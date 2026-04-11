import abc
import logging
import os
from typing import Any, Tuple

import torch
import wandb
from omegaconf import DictConfig
from tensordict import TensorDict
from tqdm import tqdm

from gomoku_rl.env import GomokuEnv
from gomoku_rl.policy import get_policy
from gomoku_rl.utils.misc import set_seed
from gomoku_rl.utils.policy import _policy_t, uniform_policy


class BaselineObservationAdapter:
    """Adapt eval-env observations to the baseline policy's expected input format.

    Supported cases:
    1) runner env temporal=False  -> baseline temporal=False  (direct pass-through)
    2) runner env temporal=True   -> baseline temporal=True   (equal or fewer steps; slice prefix)
    3) runner env temporal=True   -> baseline temporal=False  (convert absolute black/white history -> legacy 3ch)

    Not supported:
    - runner env temporal=False   -> baseline temporal=True
      Because the env is not producing history, so the missing past boards cannot be reconstructed.
    """

    def __init__(
        self,
        policy: _policy_t,
        target_use_temporal_feature: bool,
        target_temporal_num_steps: int,
    ) -> None:
        self.policy = policy
        self.target_use_temporal_feature = bool(target_use_temporal_feature)
        self.target_temporal_num_steps = int(target_temporal_num_steps)
        if self.target_temporal_num_steps < 1:
            raise ValueError("target_temporal_num_steps must be >= 1")

    def train(self, mode: bool = True):
        """Compat wrapper for both nn.Module-style train(mode) and custom PPO.train()."""
        if mode:
            if hasattr(self.policy, "train"):
                self.policy.train()
        else:
            if hasattr(self.policy, "eval"):
                self.policy.eval()
            elif hasattr(self.policy, "train"):
                self.policy.train()
        return self

    def eval(self):
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        elif hasattr(self.policy, "train"):
            # Fallback for policies that only expose train().
            self.policy.train()
        return self

    def _temporal_to_legacy_3ch(self, obs: torch.Tensor) -> torch.Tensor:
        """Convert absolute temporal planes [B_t,W_t,B_t-1,W_t-1,...] to legacy 3ch.

        Legacy 3ch semantics expected by the original baseline model:
        [current_player_stones, opponent_stones, last_move_one_hot]
        """
        if obs.ndim != 4 or obs.shape[1] < 2 or obs.shape[1] % 2 != 0:
            raise ValueError(
                f"Cannot convert observation with shape {tuple(obs.shape)} to legacy 3ch baseline input."
            )

        black_cur = obs[:, 0:1]
        white_cur = obs[:, 1:2]

        if obs.shape[1] >= 4:
            black_prev = obs[:, 2:3]
            white_prev = obs[:, 3:4]
            last_move = ((black_cur != black_prev) | (white_cur != white_prev)).float()
        else:
            last_move = torch.zeros_like(black_cur)

        black_count = black_cur.flatten(start_dim=1).sum(dim=1)
        white_count = white_cur.flatten(start_dim=1).sum(dim=1)
        black_to_play = (black_count == white_count).view(-1, 1, 1, 1)

        current_player = torch.where(black_to_play, black_cur, white_cur)
        opponent_player = torch.where(black_to_play, white_cur, black_cur)

        return torch.cat([current_player, opponent_player, last_move], dim=1)

    def _convert_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(f"Expected 4D observation, got shape {tuple(obs.shape)}")

        channels = obs.shape[1]

        if self.target_use_temporal_feature:
            target_channels = 2 * self.target_temporal_num_steps
            if channels == target_channels:
                return obs
            if channels > target_channels and channels % 2 == 0:
                return obs[:, :target_channels]
            raise ValueError(
                "Baseline temporal mode requires temporal observations from the env with at least "
                f"{target_channels} channels, but got {channels}."
            )

        # Baseline expects legacy 3ch.
        if channels == 3:
            return obs
        if channels >= 4 and channels % 2 == 0:
            return self._temporal_to_legacy_3ch(obs)
        raise ValueError(
            "Baseline legacy mode requires either legacy 3ch observations or temporal observations "
            f"that can be converted to legacy 3ch, but got {channels} channels."
        )

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        original_obs = tensordict.get("observation")
        adapted_obs = self._convert_observation(original_obs)
        tensordict.set("observation", adapted_obs)
        try:
            out = self.policy(tensordict)
        finally:
            tensordict.set("observation", original_obs)
        return out


class _RunnerEnvMixin:
    def _get_temporal_cfg(self) -> Tuple[bool, int]:
        use_temporal_feature = bool(self.cfg.get("use_temporal_feature", False))
        temporal_num_steps = int(self.cfg.get("temporal_num_steps", 3))
        if temporal_num_steps < 1:
            raise ValueError("temporal_num_steps must be >= 1")
        return use_temporal_feature, temporal_num_steps

    def _get_baseline_temporal_cfg(self) -> Tuple[bool, int]:
        baseline_cfg = self.cfg.get("baseline", {})
        runner_use_temporal, runner_temporal_steps = self._get_temporal_cfg()
        baseline_use_temporal = bool(
            baseline_cfg.get("use_temporal_feature", runner_use_temporal)
        )
        baseline_temporal_steps = int(
            baseline_cfg.get("temporal_num_steps", runner_temporal_steps)
        )
        if baseline_temporal_steps < 1:
            raise ValueError("baseline.temporal_num_steps must be >= 1")
        return baseline_use_temporal, baseline_temporal_steps

    def _make_env(self, num_envs: int, use_temporal_feature: bool, temporal_num_steps: int):
        action_pruning_cfg = self.cfg.get("action_pruning", None)
        return GomokuEnv(
            num_envs=num_envs,
            board_size=self.cfg.board_size,
            device=self.cfg.device,
            action_pruning=action_pruning_cfg,
            use_temporal_feature=use_temporal_feature,
            temporal_num_steps=temporal_num_steps,
        )

    def _load_policy_checkpoint(self, policy, checkpoint_path: str, tag: str):
        policy.load_state_dict(torch.load(checkpoint_path, map_location=self.cfg.device))
        logging.info(f"{tag}:{checkpoint_path}")

    def _build_baseline_policy(self):
        runner_use_temporal, runner_temporal_steps = self._get_temporal_cfg()
        baseline_use_temporal, baseline_temporal_steps = self._get_baseline_temporal_cfg()

        pretrained_dir = os.path.join(
            "pretrained_models",
            f"{self.cfg.board_size}_{self.cfg.board_size}",
            f"{self.cfg.baseline.name}",
        )
        if not os.path.isdir(pretrained_dir):
            return uniform_policy

        ckpts = [
            p
            for f in os.listdir(pretrained_dir)
            if os.path.isfile(p := os.path.join(pretrained_dir, f)) and p.endswith(".pt")
        ]
        if not ckpts:
            return uniform_policy

        # Build baseline with its own expected observation spec.
        baseline_env = self._make_env(
            num_envs=self.env.num_envs,
            use_temporal_feature=baseline_use_temporal,
            temporal_num_steps=baseline_temporal_steps,
        )
        baseline = get_policy(
            name=self.cfg.baseline.name,
            cfg=self.cfg.baseline,
            action_spec=baseline_env.action_spec,
            observation_spec=baseline_env.observation_spec,
            device=baseline_env.device,
        )

        ckpts.sort()
        logging.info(f"Baseline:{ckpts[0]}")
        baseline.load_state_dict(torch.load(ckpts[0], map_location=self.cfg.device))
        baseline.eval()

        # Same observation mode: use directly.
        if (
            baseline_use_temporal == runner_use_temporal
            and (
                (not baseline_use_temporal)
                or baseline_temporal_steps == runner_temporal_steps
            )
        ):
            return baseline

        # Env is legacy 3ch but baseline wants temporal: unsupported.
        if (not runner_use_temporal) and baseline_use_temporal:
            raise RuntimeError(
                "baseline.use_temporal_feature=True requires the runner env to also provide temporal observations. "
                "Current runner env is legacy 3ch, so baseline history cannot be reconstructed."
            )

        # Env has fewer temporal steps than baseline wants: unsupported.
        if runner_use_temporal and baseline_use_temporal and baseline_temporal_steps > runner_temporal_steps:
            raise RuntimeError(
                f"baseline.temporal_num_steps={baseline_temporal_steps} exceeds runner temporal_num_steps={runner_temporal_steps}."
            )

        logging.info(
            "Wrapping baseline with observation adapter: "
            f"runner(use_temporal={runner_use_temporal}, steps={runner_temporal_steps}) -> "
            f"baseline(use_temporal={baseline_use_temporal}, steps={baseline_temporal_steps})"
        )
        return BaselineObservationAdapter(
            policy=baseline,
            target_use_temporal_feature=baseline_use_temporal,
            target_temporal_num_steps=baseline_temporal_steps,
        )


class Runner(_RunnerEnvMixin, abc.ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        use_temporal_feature, temporal_num_steps = self._get_temporal_cfg()
        self.env = self._make_env(
            num_envs=cfg.num_envs,
            use_temporal_feature=use_temporal_feature,
            temporal_num_steps=temporal_num_steps,
        )
        self.eval_env = self._make_env(
            num_envs=512,
            use_temporal_feature=use_temporal_feature,
            temporal_num_steps=temporal_num_steps,
        )

        seed = cfg.get("seed", None)
        set_seed(seed)

        self.epochs: int = cfg.get("epochs")
        self.steps = cfg.steps
        self.save_interval: int = cfg.get("save_interval", -1)

        self.policy_black = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )
        self.policy_white = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )

        if black_checkpoint := cfg.get("black_checkpoint", None):
            self._load_policy_checkpoint(self.policy_black, black_checkpoint, "black_checkpoint")
        if white_checkpoint := cfg.get("white_checkpoint", None):
            self._load_policy_checkpoint(self.policy_white, white_checkpoint, "white_checkpoint")

        self.baseline = self._build_baseline_policy()

        run_dir = cfg.get("run_dir", None)
        if run_dir is None:
            run_dir = wandb.run.dir
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"run_dir:{run_dir}")
        self.run_dir = run_dir

    @abc.abstractmethod
    def _epoch(self, epoch: int) -> dict[str, Any]:
        ...

    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if wandb.run is not None:
            wandb.run.log(info)

    def run(self, disable_tqdm: bool = False):
        pbar = tqdm(range(self.epochs), disable=disable_tqdm)
        for i in pbar:
            info = {}
            info.update(self._epoch(epoch=i))
            self._log(info=info, epoch=i)
            if i % self.save_interval == 0 and self.save_interval > 0:
                torch.save(
                    self.policy_black.state_dict(),
                    os.path.join(self.run_dir, f"black_{i:04d}.pt"),
                )
                torch.save(
                    self.policy_white.state_dict(),
                    os.path.join(self.run_dir, f"white_{i:04d}.pt"),
                )
            pbar.set_postfix({"fps": info["fps"]})

        torch.save(
            self.policy_black.state_dict(),
            os.path.join(self.run_dir, "black_final.pt"),
        )
        torch.save(
            self.policy_white.state_dict(),
            os.path.join(self.run_dir, "white_final.pt"),
        )
        self._post_run()


class SPRunner(_RunnerEnvMixin, abc.ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        use_temporal_feature, temporal_num_steps = self._get_temporal_cfg()
        self.env = self._make_env(
            num_envs=cfg.num_envs,
            use_temporal_feature=use_temporal_feature,
            temporal_num_steps=temporal_num_steps,
        )
        self.eval_env = self._make_env(
            num_envs=512,
            use_temporal_feature=use_temporal_feature,
            temporal_num_steps=temporal_num_steps,
        )

        seed = cfg.get("seed", None)
        set_seed(seed)

        self.epochs: int = cfg.get("epochs")
        self.steps: int = cfg.steps
        self.save_interval: int = cfg.get("save_interval", -1)

        self.policy = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )

        if checkpoint := cfg.get("checkpoint", None):
            self._load_policy_checkpoint(self.policy, checkpoint, "checkpoint")

        self.baseline = self._build_baseline_policy()

        run_dir = cfg.get("run_dir", None)
        if run_dir is None:
            run_dir = wandb.run.dir
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"run_dir:{run_dir}")
        self.run_dir = run_dir

    @abc.abstractmethod
    def _epoch(self, epoch: int) -> dict[str, Any]:
        ...

    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if wandb.run is not None:
            wandb.run.log(info)

    def run(self, disable_tqdm: bool = False):
        pbar = tqdm(range(self.epochs), disable=disable_tqdm)
        for i in pbar:
            info = {}
            info.update(self._epoch(epoch=i))
            self._log(info=info, epoch=i)
            if i % self.save_interval == 0 and self.save_interval > 0:
                torch.save(
                    self.policy.state_dict(),
                    os.path.join(self.run_dir, f"{i:04d}.pt"),
                )
            pbar.set_postfix({"fps": info["fps"]})

        torch.save(self.policy.state_dict(), os.path.join(self.run_dir, "final.pt"))
        self._post_run()
