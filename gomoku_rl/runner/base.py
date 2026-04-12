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
    """Adapt runner observations to baseline observations.

    Runner temporal observation semantics:
      [current_player_board, opponent_board, last_1_move_one_hot, ..., last_n_move_one_hot]

    Legacy baseline observation semantics:
      [current_player_board, opponent_board, last_move_one_hot]
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
            self.policy.train()
        return self

    def _convert_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(f"Expected 4D observation, got shape {tuple(obs.shape)}")

        channels = obs.shape[1]

        # Target baseline expects temporal move-onehot observations.
        if self.target_use_temporal_feature:
            target_channels = 2 + self.target_temporal_num_steps
            if channels == target_channels:
                return obs
            if channels > target_channels:
                return obs[:, :target_channels]
            raise ValueError(
                f"Baseline expects {target_channels} channels, but runner only provides {channels}."
            )

        # Target baseline expects legacy 3ch.
        if channels == 3:
            return obs
        if channels >= 3:
            current_board = obs[:, :2]
            last_move = obs[:, 2:3]
            return torch.cat([current_board, last_move], dim=1)
        raise ValueError(
            "Cannot convert observation to legacy baseline format: "
            f"got shape {tuple(obs.shape)}"
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
        temporal_num_steps = int(self.cfg.get("temporal_num_steps", 6))
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

        if (
            baseline_use_temporal == runner_use_temporal
            and ((not baseline_use_temporal) or baseline_temporal_steps == runner_temporal_steps)
        ):
            return baseline

        if (not runner_use_temporal) and baseline_use_temporal:
            raise RuntimeError(
                "baseline.use_temporal_feature=True requires runner temporal observations."
            )

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
