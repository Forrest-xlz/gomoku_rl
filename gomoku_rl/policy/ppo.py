from __future__ import annotations

import copy
import gc
from typing import Dict, Union

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.data import DiscreteTensorSpec, TensorSpec
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate

from .base import Policy
from .common import (
    get_optimizer,
    make_critic,
    make_dataset_naive,
    make_ppo_ac,
    make_ppo_actor,
)

DeviceLike = Union[torch.device, str, int, None]


class PPO(Policy):
    """PPO with online-isolated Bsimple / gradient-noise-scale measurement.

    Compared with the previous version, the Bsimple measurement no longer runs
    forward/backward on the training actor/critic/loss_module. Instead it creates
    a temporary shadow copy of actor+critic, measures gradients on that shadow
    model, logs the result, and then deletes the shadow model.

    This isolates the normal training model from diagnostic side effects:
    - no main-parameter update from measurement;
    - no main-gradient residue from measurement;
    - no optimizer-state change from measurement;
    - no BatchNorm running_mean/running_var change on the main model.

    Random-state isolation is intentionally not implemented, because the user
    requested that RNG isolation is unnecessary.

    Optional config under cfg.algo.noise_scale:

        noise_scale:
          enabled: true
          interval: 20
          warmup_updates: 0
          ema_beta: 0.95
          min_half_batch: 2
          eps: 1e-12
          isolation: shadow_model
          empty_cuda_cache_after_measure: false
    """

    def __init__(
        self,
        cfg: DictConfig,
        action_spec: DiscreteTensorSpec,
        observation_spec: TensorSpec,
        device: DeviceLike = "cuda",
    ) -> None:
        super().__init__(cfg, action_spec, observation_spec, device)

        self.cfg: DictConfig = cfg
        self.device: DeviceLike = device

        self.clip_param: float = cfg.clip_param
        self.ppo_epoch: int = int(cfg.ppo_epochs)
        self.entropy_coef: float = cfg.entropy_coef
        self.gae_gamma: float = cfg.gamma
        self.gae_lambda: float = cfg.gae_lambda
        self.average_gae: bool = bool(cfg.average_gae)
        self.batch_size: int = int(cfg.batch_size)
        self.max_grad_norm: float = cfg.max_grad_norm

        if self.cfg.get("share_network"):
            actor_value_operator = make_ppo_ac(
                cfg,
                action_spec=action_spec,
                observation_spec=observation_spec,
                device=self.device,
            )
            self.actor = actor_value_operator.get_policy_operator()
            self.critic = actor_value_operator.get_value_head()
        else:
            self.actor = make_ppo_actor(
                cfg=cfg,
                action_spec=action_spec,
                observation_spec=observation_spec,
                device=self.device,
            )
            self.critic = make_critic(
                cfg=cfg,
                observation_spec=observation_spec,
                device=self.device,
            )

        # Initialize Lazy modules before optimizer creation / deepcopy.
        fake_input = observation_spec.zero()
        fake_input["action_mask"] = ~fake_input["action_mask"]
        fake_input = fake_input.to(self.device)

        with torch.no_grad():
            self.actor(fake_input)
            self.critic(fake_input)

        self.loss_module = self._make_loss_module(self.actor, self.critic)
        self.optim = get_optimizer(self.cfg.optimizer, self.loss_module.parameters())

        # -------------------- Bsimple / gradient-noise-scale config --------------------
        ns_cfg = self.cfg.get("noise_scale", {})
        self.noise_scale_enabled: bool = bool(ns_cfg.get("enabled", True))
        self.noise_scale_interval: int = max(1, int(ns_cfg.get("interval", 20)))
        self.noise_scale_warmup_updates: int = max(
            0, int(ns_cfg.get("warmup_updates", 0))
        )
        self.noise_scale_ema_beta: float = float(ns_cfg.get("ema_beta", 0.95))
        self.noise_scale_ema_beta = min(max(self.noise_scale_ema_beta, 0.0), 0.9999)
        self.noise_scale_min_half_batch: int = max(
            1, int(ns_cfg.get("min_half_batch", 2))
        )
        self.noise_scale_eps: float = float(ns_cfg.get("eps", 1e-12))

        # Default to the safe online isolation mode. The old main-model mode is
        # intentionally not exposed here, because it can update BatchNorm buffers.
        self.noise_scale_isolation: str = str(
            ns_cfg.get("isolation", "shadow_model")
        ).lower()
        if self.noise_scale_isolation not in {"shadow", "shadow_model", "probe"}:
            raise ValueError(
                "cfg.algo.noise_scale.isolation must be 'shadow_model' "
                "for this isolated PPO implementation."
            )

        # If true, calls torch.cuda.empty_cache() after deleting the shadow model.
        # Usually keep false: PyTorch's caching allocator may keep memory reserved
        # in nvidia-smi, but the deleted shadow memory is reusable by PyTorch.
        self.noise_scale_empty_cuda_cache_after_measure: bool = bool(
            ns_cfg.get("empty_cuda_cache_after_measure", False)
        )

        self._noise_update_counter: int = 0
        self._noise_measure_counter: int = 0
        self._noise_g2_ema: float | None = None
        self._noise_s_ema: float | None = None
        self._noise_bsimple_ema: float | None = None

    # -------------------------------------------------------------------------
    # Model / loss helpers
    # -------------------------------------------------------------------------
    def _make_loss_module(self, actor, critic) -> ClipPPOLoss:
        return ClipPPOLoss(
            actor=actor,
            critic=critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
            normalize_advantage=self.cfg.get("normalize_advantage", True),
            loss_critic_type="smooth_l1",
        )

    def __call__(self, tensordict: TensorDict):
        tensordict = tensordict.to(self.device)

        actor_input = tensordict.select("observation", "action_mask", strict=False)
        actor_output: TensorDict = self.actor(actor_input)
        actor_output = actor_output.exclude("probs")
        tensordict.update(actor_output)

        # share_network=True: critic consumes hidden.
        # share_network=False: critic consumes observation.
        critic_input = tensordict.select("hidden", "observation", strict=False)
        critic_output = self.critic(critic_input)
        tensordict.update(critic_output)

        return tensordict

    # -------------------------------------------------------------------------
    # Existing diagnostic helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_mean(values: list[torch.Tensor], *, default: float = float("nan")) -> float:
        if not values:
            return default
        return torch.stack([v.float().reshape(()) for v in values]).mean().item()

    @staticmethod
    def _get_first_existing_loss_value(loss_vals: TensorDict, keys: tuple[str, ...]):
        for key in keys:
            value = loss_vals.get(key, None)
            if value is not None:
                return value
        return None

    def _manual_clip_fraction_from_probs(self, minibatch: TensorDict) -> torch.Tensor | None:
        """Compute PPO clip fraction without calling actor.get_dist().

        Some TorchRL/TensorDict versions can build up nested functional wrappers
        around ``get_dist`` after many repeated manual calls. This fallback instead
        runs the actor once, reads the output ``probs``, and gathers the probability
        of the rollout action directly.
        """
        old_log_prob = minibatch.get("sample_log_prob", None)
        action = minibatch.get("action", None)

        if old_log_prob is None or action is None:
            return None

        try:
            with torch.no_grad():
                actor_input = minibatch.select(
                    "observation", "action_mask", "hidden", strict=False
                ).clone(False)
                actor_output: TensorDict = self.actor(actor_input)
                probs = actor_output.get("probs", None)

                if probs is None:
                    return None

                action_index = action.detach().long()

                if action_index.ndim == probs.ndim:
                    action_index = action_index.squeeze(-1)

                while action_index.ndim < probs.ndim - 1:
                    action_index = action_index.unsqueeze(-1)

                action_index = action_index.unsqueeze(-1)
                new_prob = probs.gather(-1, action_index).clamp_min(1e-12)
                new_log_prob = new_prob.log()
        except Exception:
            return None

        old_log_prob = old_log_prob.detach()
        new_log_prob = new_log_prob.detach()

        while new_log_prob.ndim < old_log_prob.ndim:
            new_log_prob = new_log_prob.unsqueeze(-1)

        while old_log_prob.ndim < new_log_prob.ndim:
            old_log_prob = old_log_prob.unsqueeze(-1)

        try:
            new_log_prob, old_log_prob = torch.broadcast_tensors(
                new_log_prob, old_log_prob
            )
        except RuntimeError:
            return None

        log_ratio = (new_log_prob - old_log_prob).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)
        clipped = (ratio < 1.0 - float(self.clip_param)) | (
            ratio > 1.0 + float(self.clip_param)
        )

        return clipped.float().mean()

    @staticmethod
    def _explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        with torch.no_grad():
            y_pred = y_pred.detach().float().reshape(-1)
            y_true = y_true.detach().float().reshape(-1)

            if y_true.numel() <= 1:
                return float("nan")

            target_var = torch.var(y_true, unbiased=False)
            if torch.isnan(target_var) or target_var.item() < 1e-8:
                return float("nan")

            residual_var = torch.var(y_true - y_pred, unbiased=False)
            return (1.0 - residual_var / target_var).item()

    # -------------------------------------------------------------------------
    # Loss / gradient helpers
    # -------------------------------------------------------------------------
    def _loss_value_from_module(
        self, loss_module: ClipPPOLoss, minibatch: TensorDict
    ) -> tuple[torch.Tensor, TensorDict]:
        loss_vals = loss_module(minibatch)
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )
        return loss_value, loss_vals

    def _loss_value(self, minibatch: TensorDict) -> tuple[torch.Tensor, TensorDict]:
        return self._loss_value_from_module(self.loss_module, minibatch)

    @staticmethod
    def _trainable_module_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
        return [p for p in module.parameters() if p.requires_grad]

    def _trainable_loss_parameters(self) -> list[torch.nn.Parameter]:
        return self._trainable_module_parameters(self.loss_module)

    @staticmethod
    def _zero_module_grad(module: torch.nn.Module) -> None:
        for p in module.parameters():
            p.grad = None

    def _zero_optimizer_grad(self) -> None:
        try:
            self.optim.zero_grad(set_to_none=True)
        except TypeError:
            self.optim.zero_grad()

    def _flat_grad_vector_from_module(self, module: torch.nn.Module) -> torch.Tensor:
        chunks = []

        for p in self._trainable_module_parameters(module):
            if p.grad is None:
                chunks.append(torch.zeros(p.numel(), device=self.device, dtype=p.dtype))
            else:
                chunks.append(p.grad.detach().reshape(-1).to(self.device))

        if not chunks:
            return torch.zeros((), device=self.device)

        return torch.cat(chunks, dim=0)

    def _flat_grad_vector(self) -> torch.Tensor:
        return self._flat_grad_vector_from_module(self.loss_module)

    @staticmethod
    def _clone_tensordict_for_measurement(minibatch: TensorDict) -> TensorDict:
        """Clone the TensorDict structure so diagnostic forward cannot add keys to it.

        clone(False) is a shallow TensorDict clone: tensor storage is shared, but
        the mapping/keys are independent. That is enough here because the loss
        should not mutate tensor values in-place.
        """
        try:
            return minibatch.clone(False)
        except TypeError:
            return minibatch.clone()

    def _grad_vector_for_batch_with_loss_module(
        self, loss_module: ClipPPOLoss, minibatch: TensorDict
    ) -> tuple[torch.Tensor, float]:
        self._zero_module_grad(loss_module)

        measurement_batch = self._clone_tensordict_for_measurement(minibatch)
        loss_value, _ = self._loss_value_from_module(loss_module, measurement_batch)
        loss_value.backward()

        grad = self._flat_grad_vector_from_module(loss_module)
        loss_item = float(loss_value.detach().item())

        self._zero_module_grad(loss_module)

        return grad, loss_item

    def _grad_vector_for_batch(self, minibatch: TensorDict) -> tuple[torch.Tensor, float]:
        self._zero_optimizer_grad()

        loss_value, _ = self._loss_value(minibatch)
        loss_value.backward()

        grad = self._flat_grad_vector()
        loss_item = float(loss_value.detach().item())

        self._zero_optimizer_grad()

        return grad, loss_item

    # -------------------------------------------------------------------------
    # Shadow-model Bsimple / gradient-noise-scale measurement
    # -------------------------------------------------------------------------
    def _should_measure_noise_scale(self) -> bool:
        if not self.noise_scale_enabled:
            return False

        if self._noise_update_counter < self.noise_scale_warmup_updates:
            return False

        return self._noise_update_counter % self.noise_scale_interval == 0

    @staticmethod
    def _first_dim_size(tensordict: TensorDict) -> int:
        if len(tensordict.shape) == 0:
            return 0
        return int(tensordict.shape[0])

    def _update_noise_ema(self, g2_hat: float, s_hat: float) -> tuple[float, float, float]:
        beta = self.noise_scale_ema_beta

        if self._noise_g2_ema is None or self._noise_s_ema is None:
            self._noise_g2_ema = float(g2_hat)
            self._noise_s_ema = float(s_hat)
        else:
            self._noise_g2_ema = beta * self._noise_g2_ema + (1.0 - beta) * float(
                g2_hat
            )
            self._noise_s_ema = beta * self._noise_s_ema + (1.0 - beta) * float(
                s_hat
            )

        g2_safe = max(float(self._noise_g2_ema), self.noise_scale_eps)
        s_safe = max(float(self._noise_s_ema), 0.0)
        self._noise_bsimple_ema = s_safe / g2_safe

        return float(self._noise_g2_ema), float(self._noise_s_ema), self._noise_bsimple_ema

    def _make_shadow_loss_module(self) -> ClipPPOLoss:
        """Create a temporary loss module for isolated Bsimple measurement.

        actor and critic are deep-copied together as one object graph. This is
        important for share_network=True: if the policy and value modules share
        an encoder/trunk, deepcopy((actor, critic)) preserves that sharing inside
        the shadow copy via Python deepcopy's memo table.
        """
        shadow_actor, shadow_critic = copy.deepcopy((self.actor, self.critic))

        shadow_actor.to(self.device)
        shadow_critic.to(self.device)
        shadow_actor.train()
        shadow_critic.train()

        shadow_loss_module = self._make_loss_module(shadow_actor, shadow_critic)
        shadow_loss_module.to(self.device)
        shadow_loss_module.train()
        self._zero_module_grad(shadow_loss_module)

        return shadow_loss_module

    def _cleanup_shadow_loss_module(self, shadow_loss_module: ClipPPOLoss | None) -> None:
        if shadow_loss_module is not None:
            self._zero_module_grad(shadow_loss_module)
            del shadow_loss_module

        gc.collect()

        if (
            self.noise_scale_empty_cuda_cache_after_measure
            and torch.cuda.is_available()
            and torch.device(self.device).type == "cuda"
        ):
            torch.cuda.empty_cache()

    def _measure_noise_scale_on_minibatch(self, minibatch: TensorDict) -> dict[str, float]:
        """Estimate Bsimple on one PPO minibatch using a temporary shadow model.

        The current minibatch is split into two equal independent chunks:

        - each half is treated as Bsmall;
        - the average gradient of both halves is treated as Bbig = 2 * Bsmall.

        This is the single-GPU analogue of the data-parallel estimator where
        Bsmall is the local per-worker batch and Bbig is the averaged global batch.

        All diagnostic forward/backward passes are performed on the shadow loss
        module, not on self.loss_module.
        """
        n_total = self._first_dim_size(minibatch)
        n_small = n_total // 2

        if n_small < self.noise_scale_min_half_batch:
            return {
                "noise_scale/measured": 0.0,
                "noise_scale/skipped_too_small": 1.0,
                "noise_scale/skipped_shadow_error": 0.0,
                "noise_scale/isolation_shadow_model": 1.0,
                "noise_scale/batch_total": float(n_total),
            }

        # Use exactly 2*n_small samples so both halves have identical size.
        part1 = minibatch[:n_small]
        part2 = minibatch[n_small : 2 * n_small]

        shadow_loss_module = None

        try:
            shadow_loss_module = self._make_shadow_loss_module()

            g1, loss1 = self._grad_vector_for_batch_with_loss_module(
                shadow_loss_module, part1
            )
            g2, loss2 = self._grad_vector_for_batch_with_loss_module(
                shadow_loss_module, part2
            )

            norm_small = 0.5 * (
                torch.dot(g1, g1).detach() + torch.dot(g2, g2).detach()
            )
            g_big = 0.5 * (g1 + g2)
            norm_big = torch.dot(g_big, g_big).detach()

            b_small = float(n_small)
            b_big = float(2 * n_small)

            norm_small_f = float(norm_small.item())
            norm_big_f = float(norm_big.item())

            g2_hat = (b_big * norm_big_f - b_small * norm_small_f) / (
                b_big - b_small
            )
            s_hat = (norm_small_f - norm_big_f) / (
                (1.0 / b_small) - (1.0 / b_big)
            )

            # Finite-sample estimates can be negative when noisy. Keep raw
            # values for diagnosis, but use clipped-positive values for the
            # stable reported ratio.
            g2_for_ratio = max(g2_hat, self.noise_scale_eps)
            s_for_ratio = max(s_hat, 0.0)

            bsimple_raw = s_hat / g2_for_ratio
            bsimple_clipped = s_for_ratio / g2_for_ratio

            g2_ema, s_ema, bsimple_ema = self._update_noise_ema(g2_hat, s_hat)

            self._noise_measure_counter += 1

            return {
                "noise_scale/measured": 1.0,
                "noise_scale/skipped_too_small": 0.0,
                "noise_scale/skipped_shadow_error": 0.0,
                "noise_scale/isolation_shadow_model": 1.0,
                "noise_scale/batch_small": b_small,
                "noise_scale/batch_big": b_big,
                "noise_scale/batch_total": float(n_total),
                "noise_scale/grad_norm_sq_small": norm_small_f,
                "noise_scale/grad_norm_sq_big": norm_big_f,
                "noise_scale/true_grad_norm_sq_hat": float(g2_hat),
                "noise_scale/grad_variance_trace_hat": float(s_hat),
                "noise_scale/bsimple_raw": float(bsimple_raw),
                "noise_scale/bsimple_clipped": float(bsimple_clipped),
                "noise_scale/true_grad_norm_sq_ema": float(g2_ema),
                "noise_scale/grad_variance_trace_ema": float(s_ema),
                "noise_scale/bsimple_ema": float(bsimple_ema),
                "noise_scale/loss_half_1": float(loss1),
                "noise_scale/loss_half_2": float(loss2),
                "noise_scale/num_measurements": float(self._noise_measure_counter),
                "noise_scale/update_index": float(self._noise_update_counter),
            }
        except Exception as exc:
            # Do not let diagnostics break training. This also avoids falling
            # back to the main model, because that would reintroduce BN-buffer
            # contamination.
            return {
                "noise_scale/measured": 0.0,
                "noise_scale/skipped_too_small": 0.0,
                "noise_scale/skipped_shadow_error": 1.0,
                "noise_scale/isolation_shadow_model": 1.0,
                "noise_scale/batch_total": float(n_total),
                "noise_scale/update_index": float(self._noise_update_counter),
                # Keep this numeric for loggers. The exception text is printed.
                "noise_scale/shadow_error_type_hash": float(
                    abs(hash(type(exc).__name__)) % 1000000
                ),
            }
        finally:
            self._cleanup_shadow_loss_module(shadow_loss_module)

            # Defensive cleanup: the main optimizer should not have grads from
            # measurement anyway, but keep the old invariant before PPO update.
            self._zero_optimizer_grad()

    # -------------------------------------------------------------------------
    # PPO learning
    # -------------------------------------------------------------------------
    def learn(self, data: TensorDict):
        value = data["state_value"].to(self.device)
        next_value = data["next", "state_value"].to(self.device)
        done = data["next", "done"].unsqueeze(-1).to(self.device)

        terminated_raw = data.get(("next", "terminated"), None)
        if terminated_raw is None:
            # Backward compatibility for old collectors that only have done.
            # New collectors should always write ("next", "terminated").
            terminated = done
        else:
            terminated = terminated_raw.unsqueeze(-1).to(self.device)

        reward = data["next", "reward"].to(self.device)

        with torch.no_grad():
            adv, value_target = vec_generalized_advantage_estimate(
                self.gae_gamma,
                self.gae_lambda,
                value,
                next_value,
                reward,
                done=done,
                terminated=terminated,
                time_dim=data.ndim - 1,
            )

        loc = adv.mean()
        scale = adv.std().clamp_min(1e-4)

        if self.average_gae:
            adv = adv - loc
            adv = adv / scale

        data.set("advantage", adv)
        data.set("value_target", value_target)

        invalid = data.get("invalid", None)
        if invalid is not None:
            data = data[~invalid]

        data = data.reshape(-1)

        critic_explained_var = self._explained_variance(
            y_pred=data.get("state_value"),
            y_true=data.get("value_target"),
        )

        self.train()

        loss_objectives = []
        loss_critics = []
        loss_entropies = []
        losses = []
        grad_norms = []
        grad_clip_flags = []
        clipfracs = []
        noise_infos = []

        for _ in range(self.ppo_epoch):
            for minibatch in make_dataset_naive(data, batch_size=self.batch_size):
                minibatch = minibatch.to(self.device)

                # Online-isolated diagnostic measurement. It is intentionally
                # placed before the normal optimizer step so the measured shadow
                # parameters match the main parameters used by this minibatch
                # update. The main model is not forward/backwarded here.
                if self._should_measure_noise_scale():
                    noise_infos.append(self._measure_noise_scale_on_minibatch(minibatch))

                self._zero_optimizer_grad()

                loss_value, loss_vals = self._loss_value(minibatch)

                loss_objectives.append(loss_vals["loss_objective"].clone().detach())
                loss_critics.append(loss_vals["loss_critic"].clone().detach())
                loss_entropies.append(loss_vals["loss_entropy"].clone().detach())
                losses.append(loss_value.clone().detach())

                clip_fraction = self._get_first_existing_loss_value(
                    loss_vals,
                    ("clip_fraction", "clipfrac", "clip_frac"),
                )
                if clip_fraction is None:
                    clip_fraction = self._manual_clip_fraction_from_probs(minibatch)
                if clip_fraction is not None:
                    clipfracs.append(clip_fraction.clone().detach().float().mean())

                loss_value.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(),
                    self.max_grad_norm,
                )
                grad_norm_detached = grad_norm.clone().detach().float()
                grad_norms.append(grad_norm_detached)
                grad_clip_flags.append(
                    (grad_norm_detached > float(self.max_grad_norm)).float()
                )

                self.optim.step()
                self._zero_optimizer_grad()

                self._noise_update_counter += 1

        self.eval()

        info = {
            "advantage_mean": loc.item(),
            "advantage_std": scale.item(),
            "grad_norm": self._safe_mean(grad_norms),
            "grad_clip_frac": self._safe_mean(grad_clip_flags),
            "clipfrac": self._safe_mean(clipfracs),
            "critic_explained_var": critic_explained_var,
            "loss": torch.stack(losses).mean().item(),
            "loss_objective": torch.stack(loss_objectives).mean().item(),
            "loss_critic": torch.stack(loss_critics).mean().item(),
            "loss_entropy": torch.stack(loss_entropies).mean().item(),
        }

        if noise_infos:
            # Log the last fresh measurement from this learn() call. EMA fields
            # already contain history across previous learn() calls.
            info.update(noise_infos[-1])
        elif self.noise_scale_enabled and self._noise_bsimple_ema is not None:
            # Keep the EMA visible on learn() calls without a fresh measurement.
            info.update(
                {
                    "noise_scale/measured": 0.0,
                    "noise_scale/isolation_shadow_model": 1.0,
                    "noise_scale/bsimple_ema": float(self._noise_bsimple_ema),
                    "noise_scale/true_grad_norm_sq_ema": float(self._noise_g2_ema),
                    "noise_scale/grad_variance_trace_ema": float(self._noise_s_ema),
                    "noise_scale/num_measurements": float(self._noise_measure_counter),
                    "noise_scale/update_index": float(self._noise_update_counter),
                }
            )

        return info

    def state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optim.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        self.critic.load_state_dict(state_dict["critic"], strict=False)
        self.actor.load_state_dict(state_dict["actor"])

        self.loss_module = self._make_loss_module(self.actor, self.critic)
        self.optim = get_optimizer(self.cfg.optimizer, self.loss_module.parameters())

        opt_state = state_dict.get("optimizer", state_dict.get("optim", None))
        if opt_state is not None:
            self.optim.load_state_dict(opt_state)
            print("optimizer state loaded successfully.")

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
