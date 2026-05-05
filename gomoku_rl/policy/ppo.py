from __future__ import annotations

import copy
import gc
from typing import Dict, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.data import DiscreteTensorSpec, TensorSpec
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


class ManualPPOLoss(torch.nn.Module):
    """PPO loss with optional critic value clipping and dual-clip PPO.

    This wrapper keeps the actor/critic modules under one torch.nn.Module so the
    existing optimizer, checkpoint and Bsimple shadow-model code can still use
    ``loss_module.parameters()`` exactly as before.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        *,
        clip_epsilon: float,
        entropy_bonus: bool,
        entropy_coef: float,
        normalize_advantage: bool,
        loss_critic_type: str = "smooth_l1",
        critic_coef: float = 1.0,
        use_critic_clip: bool = False,
        critic_clip_param: float | None = None,
        use_dual_clip: bool = False,
        dual_clip_coef: float = 3.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_bonus = bool(entropy_bonus)
        self.entropy_coef = float(entropy_coef)
        self.normalize_advantage = bool(normalize_advantage)
        self.loss_critic_type = str(loss_critic_type).lower()
        self.critic_coef = float(critic_coef)
        self.use_critic_clip = bool(use_critic_clip)
        self.critic_clip_param = (
            float(self.clip_epsilon)
            if critic_clip_param is None
            else float(critic_clip_param)
        )
        self.use_dual_clip = bool(use_dual_clip)
        self.dual_clip_coef = float(dual_clip_coef)
        self.eps = float(eps)

        if self.use_dual_clip and self.dual_clip_coef <= 1.0:
            raise ValueError("dual_clip_coef must be > 1 when use_dual_clip=True.")
        if self.use_critic_clip and self.critic_clip_param <= 0.0:
            raise ValueError("critic_clip_param must be > 0 when use_critic_clip=True.")
        if self.loss_critic_type not in {"mse", "l2", "smooth_l1", "huber"}:
            raise ValueError(
                "loss_critic_type must be one of: mse, l2, smooth_l1, huber."
            )

    @staticmethod
    def _shape_as_action_index(action: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        action_index = action.detach().long()
        if action_index.ndim == probs.ndim and action_index.shape[-1] == 1:
            action_index = action_index.squeeze(-1)
        while action_index.ndim < probs.ndim - 1:
            action_index = action_index.unsqueeze(-1)
        if action_index.ndim == probs.ndim - 1:
            action_index = action_index.unsqueeze(-1)
        return action_index

    @staticmethod
    def _broadcast_three(
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_ndim = max(x.ndim, y.ndim, z.ndim)
        while x.ndim < max_ndim:
            x = x.unsqueeze(-1)
        while y.ndim < max_ndim:
            y = y.unsqueeze(-1)
        while z.ndim < max_ndim:
            z = z.unsqueeze(-1)
        return torch.broadcast_tensors(x, y, z)

    @staticmethod
    def _broadcast_two(
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_ndim = max(x.ndim, y.ndim)
        while x.ndim < max_ndim:
            x = x.unsqueeze(-1)
        while y.ndim < max_ndim:
            y = y.unsqueeze(-1)
        return torch.broadcast_tensors(x, y)

    def _critic_error(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_critic_type in {"smooth_l1", "huber"}:
            return F.smooth_l1_loss(pred, target, reduction="none")
        return torch.square(target - pred)

    def _get_value_tensor(self, critic_output: TensorDict) -> torch.Tensor:
        value = critic_output.get("state_value", None)
        if value is None:
            value = critic_output.get("value", None)
        if value is None:
            raise KeyError(
                "critic output must contain 'state_value' or 'value' for PPO loss."
            )
        return value

    def forward(self, minibatch: TensorDict) -> TensorDict:
        actor_input = minibatch.select(
            "observation",
            "action_mask",
            "hidden",
            strict=False,
        ).clone(False)
        actor_output: TensorDict = self.actor(actor_input)
        probs = actor_output.get("probs", None)
        if probs is None:
            raise KeyError("actor output must contain 'probs' for manual PPO loss.")

        action = minibatch.get("action")
        old_log_prob = minibatch.get("sample_log_prob")
        advantage = minibatch.get("advantage")
        if action is None or old_log_prob is None or advantage is None:
            raise KeyError(
                "PPO minibatch must contain 'action', 'sample_log_prob' and 'advantage'."
            )

        action_index = self._shape_as_action_index(action, probs)
        selected_prob = probs.gather(-1, action_index).clamp_min(self.eps)
        new_log_prob = selected_prob.log()

        new_log_prob, old_log_prob, advantage = self._broadcast_three(
            new_log_prob,
            old_log_prob.detach(),
            advantage,
        )
        advantage = advantage.detach()
        if self.normalize_advantage:
            adv_mean = advantage.mean()
            adv_std = advantage.std(unbiased=False).clamp_min(1e-8)
            advantage = (advantage - adv_mean) / adv_std

        log_ratio = (new_log_prob - old_log_prob).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)
        surrogate_1 = ratio * advantage
        surrogate_2 = ratio.clamp(
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        ) * advantage
        ppo_surrogate = torch.minimum(surrogate_1, surrogate_2)

        dual_clip_applied = torch.zeros_like(ppo_surrogate, dtype=torch.bool)
        if self.use_dual_clip:
            dual_floor = self.dual_clip_coef * advantage
            negative_adv = advantage < 0.0
            dual_clip_applied = negative_adv & (ppo_surrogate < dual_floor)
            ppo_surrogate = torch.where(
                negative_adv,
                torch.maximum(ppo_surrogate, dual_floor),
                ppo_surrogate,
            )

        loss_objective = -ppo_surrogate.mean()

        probs_safe = probs.clamp_min(self.eps)
        entropy = -(probs * probs_safe.log()).sum(dim=-1)
        if self.entropy_bonus:
            loss_entropy = -float(self.entropy_coef) * entropy.mean()
        else:
            loss_entropy = probs.new_zeros(())

        critic_tensordict = minibatch.clone(False)
        hidden = actor_output.get("hidden", None)
        if hidden is not None:
            critic_tensordict.set("hidden", hidden)
        critic_input = critic_tensordict.select("hidden", "observation", strict=False)
        critic_output = self.critic(critic_input)
        value_pred = self._get_value_tensor(critic_output)

        value_target = minibatch.get("value_target", None)
        if value_target is None:
            raise KeyError("PPO minibatch must contain 'value_target'.")
        value_pred, value_target = self._broadcast_two(value_pred, value_target.detach())

        critic_clip_fraction = value_pred.new_zeros(())
        if self.use_critic_clip:
            old_value = minibatch.get("state_value", None)
            if old_value is None:
                raise KeyError(
                    "critic clipping requires rollout-time 'state_value' in minibatch."
                )
            value_pred_for_old, old_value = self._broadcast_two(value_pred, old_value.detach())
            value_clipped = old_value + (value_pred_for_old - old_value).clamp(
                -self.critic_clip_param,
                self.critic_clip_param,
            )
            value_loss_unclipped = self._critic_error(value_pred_for_old, value_target)
            value_loss_clipped = self._critic_error(value_clipped, value_target)
            value_loss = torch.maximum(value_loss_unclipped, value_loss_clipped)
            critic_clip_fraction = (
                (value_pred_for_old - old_value).abs() > self.critic_clip_param
            ).float().mean()
        else:
            value_loss = self._critic_error(value_pred, value_target)

        loss_critic = self.critic_coef * value_loss.mean()

        with torch.no_grad():
            clip_fraction = (
                (ratio < 1.0 - self.clip_epsilon)
                | (ratio > 1.0 + self.clip_epsilon)
            ).float().mean()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            ratio_mean = ratio.mean()
            ratio_max = ratio.max()
            dual_clip_fraction = dual_clip_applied.float().mean()

        return TensorDict(
            {
                "loss_objective": loss_objective,
                "loss_critic": loss_critic,
                "loss_entropy": loss_entropy,
                "clip_fraction": clip_fraction,
                "approx_kl": approx_kl,
                "ratio_mean": ratio_mean,
                "ratio_max": ratio_max,
                "dual_clip_fraction": dual_clip_fraction,
                "critic_clip_fraction": critic_clip_fraction,
            },
            batch_size=[],
        )


class PPO(Policy):
    """PPO with online-isolated Bsimple measurement and outer-epoch schedules."""

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
        self.clip_param: float = float(cfg.clip_param)
        self.ppo_epoch: int = int(cfg.ppo_epochs)
        self.entropy_coef: float = float(cfg.entropy_coef)
        self.gae_gamma: float = float(cfg.gamma)
        self.gae_lambda: float = float(cfg.gae_lambda)
        self.average_gae: bool = bool(cfg.average_gae)
        self.batch_size: int = int(cfg.batch_size)
        self.max_grad_norm: float = float(cfg.max_grad_norm)

        # -------------------- PPO clip extensions --------------------
        self.use_critic_clip: bool = bool(self.cfg.get("use_critic_clip", False))
        self.critic_clip_param: float = float(
            self.cfg.get("critic_clip_param", self.clip_param)
        )
        self.critic_loss_type: str = str(
            self.cfg.get("critic_loss_type", "smooth_l1")
        )
        self.critic_coef: float = float(self.cfg.get("critic_coef", 1.0))
        self.use_dual_clip: bool = bool(self.cfg.get("use_dual_clip", False))
        self.dual_clip_coef: float = float(self.cfg.get("dual_clip_coef", 3.0))
        # ----------------------------------------------------------------

        # -------------------- outer-epoch LR / entropy schedule --------------------
        base_lr = float(self.cfg.optimizer.kwargs.lr)
        self.lr_decay_mode: str = self._normalize_schedule_mode(
            self.cfg.get("lr_decay_mode", "constant")
        )
        self.lr_start: float = float(self.cfg.get("lr_start", base_lr))
        self.lr_end: float = float(self.cfg.get("lr_end", self.lr_start))
        self.lr_epochs: int = max(1, int(self.cfg.get("lr_epochs", 1)))

        self.entropy_decay_mode: str = self._normalize_schedule_mode(
            self.cfg.get("entropy_decay_mode", "constant")
        )
        self.entropy_start: float = float(
            self.cfg.get("entropy_start", self.entropy_coef)
        )
        self.entropy_end: float = float(
            self.cfg.get("entropy_end", self.entropy_start)
        )
        self.entropy_epochs: int = max(1, int(self.cfg.get("entropy_epochs", 1)))

        self._outer_epoch_index: int = 0
        self.current_lr: float = (
            self.lr_start if self._is_linear_mode(self.lr_decay_mode) else base_lr
        )
        self.current_entropy_coef: float = (
            self.entropy_start
            if self._is_linear_mode(self.entropy_decay_mode)
            else self.entropy_coef
        )
        self.entropy_coef = float(self.current_entropy_coef)
        # -------------------------------------------------------------------------

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
        self._apply_outer_epoch_schedule(self._outer_epoch_index)

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
    # Outer-epoch schedule helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_schedule_mode(mode) -> str:
        mode_str = str(mode).strip().lower()
        if mode_str in {"true", "on", "enable", "enabled", "linear_decay", "decay"}:
            return "linear"
        if mode_str in {"false", "off", "disable", "disabled", "none"}:
            return "constant"
        return mode_str

    @staticmethod
    def _is_linear_mode(mode: str) -> bool:
        return str(mode).strip().lower() in {"linear", "lin"}

    @staticmethod
    def _linear_value(start: float, end: float, epochs: int, epoch_index: int) -> float:
        epochs = max(1, int(epochs))
        progress = min(max(float(epoch_index), 0.0), float(epochs)) / float(epochs)
        return float(start + (end - start) * progress)

    def _scheduled_lr(self, epoch_index: int) -> float:
        if self._is_linear_mode(self.lr_decay_mode):
            return self._linear_value(
                self.lr_start,
                self.lr_end,
                self.lr_epochs,
                epoch_index,
            )
        return float(self.cfg.optimizer.kwargs.lr)

    def _scheduled_entropy_coef(self, epoch_index: int) -> float:
        if self._is_linear_mode(self.entropy_decay_mode):
            return self._linear_value(
                self.entropy_start,
                self.entropy_end,
                self.entropy_epochs,
                epoch_index,
            )
        return float(self.cfg.entropy_coef)

    def _set_optimizer_lr(self, lr: float) -> None:
        if not hasattr(self, "optim") or self.optim is None:
            return
        for group in self.optim.param_groups:
            group["lr"] = float(lr)

    @staticmethod
    def _set_loss_module_entropy_coef(loss_module: torch.nn.Module, value: float) -> None:
        """Best-effort update across TorchRL/custom loss-module versions."""
        value = float(value)
        for attr_name in ("entropy_coef", "entropy_coeff"):
            if hasattr(loss_module, attr_name):
                attr_value = getattr(loss_module, attr_name)
                if torch.is_tensor(attr_value):
                    attr_value.data.fill_(value)
                else:
                    setattr(loss_module, attr_name, value)
        buffers = getattr(loss_module, "_buffers", {})
        for buffer_name in ("entropy_coef", "entropy_coeff"):
            buffer = buffers.get(buffer_name, None)
            if torch.is_tensor(buffer):
                buffer.data.fill_(value)

    def _apply_outer_epoch_schedule(self, epoch_index: int) -> None:
        self._outer_epoch_index = max(0, int(epoch_index))
        self.current_lr = self._scheduled_lr(self._outer_epoch_index)
        self.current_entropy_coef = self._scheduled_entropy_coef(
            self._outer_epoch_index
        )
        self.entropy_coef = float(self.current_entropy_coef)
        self._set_optimizer_lr(self.current_lr)
        if hasattr(self, "loss_module") and self.loss_module is not None:
            self._set_loss_module_entropy_coef(
                self.loss_module,
                self.current_entropy_coef,
            )

    def set_outer_epoch(self, epoch_index: int) -> dict[str, float]:
        """Set LR / entropy coefficient according to the current outer epoch."""
        self._apply_outer_epoch_schedule(epoch_index)
        return self.get_schedule_info()

    def get_schedule_info(self) -> dict[str, float]:
        return {
            "schedule/outer_epoch": float(self._outer_epoch_index),
            "schedule/lr": float(self.current_lr),
            "schedule/lr_decay_linear": float(self._is_linear_mode(self.lr_decay_mode)),
            "schedule/lr_start": float(self.lr_start),
            "schedule/lr_end": float(self.lr_end),
            "schedule/lr_epochs": float(self.lr_epochs),
            "schedule/entropy_coef": float(self.current_entropy_coef),
            "schedule/entropy_decay_linear": float(
                self._is_linear_mode(self.entropy_decay_mode)
            ),
            "schedule/entropy_start": float(self.entropy_start),
            "schedule/entropy_end": float(self.entropy_end),
            "schedule/entropy_epochs": float(self.entropy_epochs),
        }

    # -------------------------------------------------------------------------
    # Model / loss helpers
    # -------------------------------------------------------------------------
    def _entropy_bonus_enabled(self) -> bool:
        if self._is_linear_mode(self.entropy_decay_mode):
            return True
        return bool(self.entropy_coef)

    def _make_loss_module(self, actor, critic) -> ManualPPOLoss:
        return ManualPPOLoss(
            actor=actor,
            critic=critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=self._entropy_bonus_enabled(),
            entropy_coef=float(self.entropy_coef),
            normalize_advantage=self.cfg.get("normalize_advantage", True),
            loss_critic_type=self.critic_loss_type,
            critic_coef=self.critic_coef,
            use_critic_clip=self.use_critic_clip,
            critic_clip_param=self.critic_clip_param,
            use_dual_clip=self.use_dual_clip,
            dual_clip_coef=self.dual_clip_coef,
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
        """Compute PPO clip fraction without calling actor.get_dist()."""
        old_log_prob = minibatch.get("sample_log_prob", None)
        action = minibatch.get("action", None)
        if old_log_prob is None or action is None:
            return None

        try:
            with torch.no_grad():
                actor_input = minibatch.select(
                    "observation",
                    "action_mask",
                    "hidden",
                    strict=False,
                ).clone(False)
                actor_output: TensorDict = self.actor(actor_input)
                probs = actor_output.get("probs", None)
                if probs is None:
                    return None
                action_index = ManualPPOLoss._shape_as_action_index(action, probs)
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
                new_log_prob,
                old_log_prob,
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
        self,
        loss_module: torch.nn.Module,
        minibatch: TensorDict,
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
        """Clone TensorDict structure so diagnostic forward cannot add keys to it."""
        try:
            return minibatch.clone(False)
        except TypeError:
            return minibatch.clone()

    def _grad_vector_for_batch_with_loss_module(
        self,
        loss_module: torch.nn.Module,
        minibatch: TensorDict,
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

    def _make_shadow_loss_module(self) -> ManualPPOLoss:
        """Create a temporary loss module for isolated Bsimple measurement."""
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

    def _cleanup_shadow_loss_module(
        self,
        shadow_loss_module: torch.nn.Module | None,
    ) -> None:
        if shadow_loss_module is not None:
            self._zero_module_grad(shadow_loss_module)
            del shadow_loss_module
        gc.collect()
        if self.noise_scale_empty_cuda_cache_after_measure and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _measure_noise_scale_on_minibatch(self, minibatch: TensorDict) -> dict[str, float]:
        """Estimate Bsimple on one PPO minibatch using a temporary shadow model."""
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
                shadow_loss_module,
                part1,
            )
            g2, loss2 = self._grad_vector_for_batch_with_loss_module(
                shadow_loss_module,
                part2,
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
            return {
                "noise_scale/measured": 0.0,
                "noise_scale/skipped_too_small": 0.0,
                "noise_scale/skipped_shadow_error": 1.0,
                "noise_scale/isolation_shadow_model": 1.0,
                "noise_scale/batch_total": float(n_total),
                "noise_scale/update_index": float(self._noise_update_counter),
                "noise_scale/shadow_error_type_hash": float(
                    abs(hash(type(exc).__name__)) % 1000000
                ),
            }
        finally:
            self._cleanup_shadow_loss_module(shadow_loss_module)
            self._zero_optimizer_grad()

    # -------------------------------------------------------------------------
    # PPO learning
    # -------------------------------------------------------------------------
    def learn(self, data: TensorDict):
        # Re-apply the current outer-epoch schedule at the start of every learn()
        # call. This protects against optimizer-state reloads or outside changes.
        self._apply_outer_epoch_schedule(self._outer_epoch_index)

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
        approx_kls = []
        ratio_means = []
        ratio_maxes = []
        dual_clip_fracs = []
        critic_clip_fracs = []
        noise_infos = []

        for _ in range(self.ppo_epoch):
            for minibatch in make_dataset_naive(data, batch_size=self.batch_size):
                minibatch = minibatch.to(self.device)
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

                for key, store in (
                    ("approx_kl", approx_kls),
                    ("ratio_mean", ratio_means),
                    ("ratio_max", ratio_maxes),
                    ("dual_clip_fraction", dual_clip_fracs),
                    ("critic_clip_fraction", critic_clip_fracs),
                ):
                    value_for_log = loss_vals.get(key, None)
                    if value_for_log is not None:
                        store.append(value_for_log.clone().detach().float().mean())

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
            "approx_kl": self._safe_mean(approx_kls),
            "ratio_mean": self._safe_mean(ratio_means),
            "ratio_max": self._safe_mean(ratio_maxes),
            "dual_clip_frac": self._safe_mean(dual_clip_fracs),
            "critic_clip_frac": self._safe_mean(critic_clip_fracs),
            "critic_explained_var": critic_explained_var,
            "loss": torch.stack(losses).mean().item(),
            "loss_objective": torch.stack(loss_objectives).mean().item(),
            "loss_critic": torch.stack(loss_critics).mean().item(),
            "loss_entropy": torch.stack(loss_entropies).mean().item(),
            "ppo/use_critic_clip": float(self.use_critic_clip),
            "ppo/critic_clip_param": float(self.critic_clip_param),
            "ppo/use_dual_clip": float(self.use_dual_clip),
            "ppo/dual_clip_coef": float(self.dual_clip_coef),
        }
        info.update(self.get_schedule_info())

        if noise_infos:
            info.update(noise_infos[-1])
        elif self.noise_scale_enabled and self._noise_bsimple_ema is not None:
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
        # Checkpoint optimizer state may carry old LR values. Restore the current
        # schedule value immediately; the runner will set the correct resumed
        # outer epoch before the next learn() call.
        self._apply_outer_epoch_schedule(self._outer_epoch_index)

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
