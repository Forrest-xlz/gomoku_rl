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
        self.average_gae: float = cfg.average_gae
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

        fake_input = observation_spec.zero()
        fake_input["action_mask"] = ~fake_input["action_mask"]
        with torch.no_grad():
            self.actor(fake_input)
            self.critic(fake_input)

        self.loss_module = ClipPPOLoss(
            actor=self.actor,
            critic=self.critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
            normalize_advantage=self.cfg.get("normalize_advantage", True),
            loss_critic_type="smooth_l1",
        )
        self.optim = get_optimizer(self.cfg.optimizer, self.loss_module.parameters())

    def __call__(self, tensordict: TensorDict):
        tensordict = tensordict.to(self.device)
        actor_input = tensordict.select("observation", "action_mask", strict=False)
        actor_output: TensorDict = self.actor(actor_input)
        actor_output = actor_output.exclude("probs")
        tensordict.update(actor_output)

        critic_input = tensordict.select("hidden", "observation", strict=False)
        critic_output = self.critic(critic_input)
        tensordict.update(critic_output)
        return tensordict

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
        around ``get_dist`` after many repeated manual calls. That may eventually
        trigger ``RecursionError``. This fallback instead runs the actor once,
        reads the output ``probs``, and gathers the probability of the rollout
        action directly.
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
            new_log_prob, old_log_prob = torch.broadcast_tensors(new_log_prob, old_log_prob)
        except RuntimeError:
            return None

        log_ratio = (new_log_prob - old_log_prob).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)
        clipped = (ratio < 1.0 - float(self.clip_param)) | (ratio > 1.0 + float(self.clip_param))
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

    def learn(self, data: TensorDict):
        value = data["state_value"].to(self.device)
        next_value = data["next", "state_value"].to(self.device)
        done = data["next", "done"].unsqueeze(-1).to(self.device)
        reward = data["next", "reward"].to(self.device)

        with torch.no_grad():
            adv, value_target = vec_generalized_advantage_estimate(
                self.gae_gamma,
                self.gae_lambda,
                value,
                next_value,
                reward,
                done=done,
                terminated=done,
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

        for _ in range(self.ppo_epoch):
            for minibatch in make_dataset_naive(data, batch_size=self.batch_size):
                minibatch = minibatch.to(self.device)
                loss_vals = self.loss_module(minibatch)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

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
                self.optim.zero_grad()

        self.eval()
        return {
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

    def state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optim.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        self.critic.load_state_dict(state_dict["critic"], strict=False)
        self.actor.load_state_dict(state_dict["actor"])
        self.loss_module = ClipPPOLoss(
            actor=self.actor,
            critic=self.critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
            normalize_advantage=self.cfg.get("normalize_advantage", True),
            loss_critic_type="smooth_l1",
        )
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
