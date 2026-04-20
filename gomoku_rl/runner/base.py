import abc
import logging
import os
from typing import Any, Tuple

import torch
import wandb
from omegaconf import DictConfig
from tensordict import TensorDict
from tqdm import tqdm

from gomoku_rl.env import (
    AZ_HISTORY_MODE,
    LEGACY_MODE,
    TEMPORAL_MOVE_HISTORY_MODE,
    GomokuEnv,
)

from gomoku_rl.policy import get_policy
from gomoku_rl.utils.misc import set_seed
from gomoku_rl.utils.policy import _policy_t, uniform_policy


class BaselineObservationAdapter:
    """Adapt runner observations to baseline observations.

    Supported runner observation modes:
    - legacy
    - temporal_move_history
    - az_history

    Supported baseline target modes:
    - legacy
    - temporal_move_history
    - az_history (only when runner is already az_history)
    """

    def __init__(
        self,
        policy: _policy_t,
        source_observation_mode: str,
        source_temporal_num_steps: int,
        target_observation_mode: str,
        target_temporal_num_steps: int,
    ) -> None:
        self.policy = policy
        self.source_observation_mode = str(source_observation_mode)
        self.source_temporal_num_steps = int(source_temporal_num_steps)
        self.target_observation_mode = str(target_observation_mode)
        self.target_temporal_num_steps = int(target_temporal_num_steps)

        if self.source_temporal_num_steps < 1:
            raise ValueError("source_temporal_num_steps must be >= 1")
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

    @staticmethod
    def _split_az_history(obs: torch.Tensor, steps: int):
        black_hist = obs[:, :steps]
        white_hist = obs[:, steps : 2 * steps]
        side = obs[:, 2 * steps : 2 * steps + 1]
        return black_hist, white_hist, side

    @staticmethod
    def _side_to_move_is_black(side_plane: torch.Tensor) -> torch.Tensor:
        return side_plane[:, 0, 0, 0] > 0.5

    @staticmethod
    def _select_current_player_planes(
        black: torch.Tensor,
        white: torch.Tensor,
        side_black: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = side_black.view(-1, 1, 1)
        current = torch.where(mask, black, white)
        opponent = torch.where(mask, white, black)
        return current, opponent

    @staticmethod
    def _pad_channel_history(x: torch.Tensor, target_steps: int) -> torch.Tensor:
        current_steps = x.shape[1]
        if current_steps == target_steps:
            return x
        if current_steps > target_steps:
            return x[:, :target_steps]
        pad_shape = list(x.shape)
        pad_shape[1] = target_steps - current_steps
        pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=1)

    def _convert_from_az_to_legacy(self, obs: torch.Tensor) -> torch.Tensor:
        black_hist, white_hist, side = self._split_az_history(
            obs, self.source_temporal_num_steps
        )
        current_black = black_hist[:, 0]
        current_white = white_hist[:, 0]
        side_black = self._side_to_move_is_black(side)
        current, opponent = self._select_current_player_planes(
            current_black,
            current_white,
            side_black,
        )

        if self.source_temporal_num_steps >= 2:
            prev_black = black_hist[:, 1]
            prev_white = white_hist[:, 1]
            last_move = ((current_black > prev_black) | (current_white > prev_white)).float()
        else:
            last_move = torch.zeros_like(current_black)

        return torch.stack([current, opponent, last_move], dim=1)

    def _convert_from_az_to_temporal_move_history(self, obs: torch.Tensor) -> torch.Tensor:
        black_hist, white_hist, side = self._split_az_history(
            obs, self.source_temporal_num_steps
        )
        current_black = black_hist[:, 0]
        current_white = white_hist[:, 0]
        side_black = self._side_to_move_is_black(side)
        current, opponent = self._select_current_player_planes(
            current_black,
            current_white,
            side_black,
        )

        max_available = max(self.source_temporal_num_steps - 1, 0)
        move_planes = []
        for i in range(min(self.target_temporal_num_steps, max_available)):
            black_now = black_hist[:, i]
            black_prev = black_hist[:, i + 1]
            white_now = white_hist[:, i]
            white_prev = white_hist[:, i + 1]
            move_plane = ((black_now > black_prev) | (white_now > white_prev)).float()
            move_planes.append(move_plane.unsqueeze(1))

        if move_planes:
            moves = torch.cat(move_planes, dim=1)
        else:
            moves = torch.zeros(
                obs.shape[0],
                0,
                obs.shape[-2],
                obs.shape[-1],
                device=obs.device,
                dtype=obs.dtype,
            )
        moves = self._pad_channel_history(moves, self.target_temporal_num_steps)
        return torch.cat([current.unsqueeze(1), opponent.unsqueeze(1), moves], dim=1)

    def _convert_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(f"Expected 4D observation, got shape {tuple(obs.shape)}")

        # Fast path
        if (
            self.source_observation_mode == self.target_observation_mode
            and (
                self.target_observation_mode == LEGACY_MODE
                or self.source_temporal_num_steps == self.target_temporal_num_steps
            )
        ):
            return obs

        if self.target_observation_mode == LEGACY_MODE:
            if self.source_observation_mode == LEGACY_MODE:
                return obs
            if self.source_observation_mode == TEMPORAL_MOVE_HISTORY_MODE:
                return obs[:, :3]
            if self.source_observation_mode == AZ_HISTORY_MODE:
                return self._convert_from_az_to_legacy(obs)

        if self.target_observation_mode == TEMPORAL_MOVE_HISTORY_MODE:
            if self.source_observation_mode == LEGACY_MODE:
                current = obs[:, :2]
                last_move = obs[:, 2:3]
                last_move = self._pad_channel_history(
                    last_move,
                    self.target_temporal_num_steps,
                )
                return torch.cat([current, last_move], dim=1)
            if self.source_observation_mode == TEMPORAL_MOVE_HISTORY_MODE:
                current = obs[:, :2]
                moves = self._pad_channel_history(
                    obs[:, 2:],
                    self.target_temporal_num_steps,
                )
                return torch.cat([current, moves], dim=1)
            if self.source_observation_mode == AZ_HISTORY_MODE:
                return self._convert_from_az_to_temporal_move_history(obs)

        if self.target_observation_mode == AZ_HISTORY_MODE:
            if self.source_observation_mode != AZ_HISTORY_MODE:
                raise RuntimeError(
                    "Cannot convert non-AZ observations to az_history baseline format."
                )
            black_hist, white_hist, side = self._split_az_history(
                obs, self.source_temporal_num_steps
            )
            black_hist = self._pad_channel_history(
                black_hist,
                self.target_temporal_num_steps,
            )
            white_hist = self._pad_channel_history(
                white_hist,
                self.target_temporal_num_steps,
            )
            return torch.cat([black_hist, white_hist, side], dim=1)

        raise RuntimeError(
            "Unsupported observation adaptation: "
            f"source={self.source_observation_mode}, target={self.target_observation_mode}"
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
    def _get_observation_cfg(self) -> Tuple[str, int]:
        observation_mode = self.cfg.get("observation_mode", None)
        if observation_mode is None:
            use_temporal_feature = bool(self.cfg.get("use_temporal_feature", False))
            observation_mode = (
                TEMPORAL_MOVE_HISTORY_MODE if use_temporal_feature else LEGACY_MODE
            )
        else:
            observation_mode = str(observation_mode)

        temporal_num_steps = int(self.cfg.get("temporal_num_steps", 6))
        if temporal_num_steps < 1:
            raise ValueError("temporal_num_steps must be >= 1")
        return observation_mode, temporal_num_steps

    def _get_baseline_observation_cfg(self) -> Tuple[str, int]:
        baseline_cfg = self.cfg.get("baseline", {})
        runner_mode, runner_steps = self._get_observation_cfg()

        observation_mode = baseline_cfg.get("observation_mode", None)
        if observation_mode is None:
            if "use_temporal_feature" in baseline_cfg:
                observation_mode = (
                    TEMPORAL_MOVE_HISTORY_MODE
                    if bool(baseline_cfg.get("use_temporal_feature", False))
                    else LEGACY_MODE
                )
            else:
                observation_mode = runner_mode
        else:
            observation_mode = str(observation_mode)

        temporal_num_steps = int(baseline_cfg.get("temporal_num_steps", runner_steps))
        if temporal_num_steps < 1:
            raise ValueError("baseline.temporal_num_steps must be >= 1")
        return observation_mode, temporal_num_steps

    def _make_env(
        self,
        num_envs: int,
        observation_mode: str,
        temporal_num_steps: int,
    ):
        action_pruning_cfg = self.cfg.get("action_pruning", None)
        return GomokuEnv(
            num_envs=num_envs,
            board_size=self.cfg.board_size,
            device=self.cfg.device,
            action_pruning=action_pruning_cfg,
            observation_mode=observation_mode,
            temporal_num_steps=temporal_num_steps,
        )

    def _load_policy_checkpoint(self, policy, checkpoint_path: str, tag: str):
        policy.load_state_dict(torch.load(checkpoint_path, map_location=self.cfg.device))
        logging.info(f"{tag}:{checkpoint_path}")

    def _normalize_checkpoint_list(self, checkpoint_paths) -> list[str]:
        if checkpoint_paths is None:
            return []
        if isinstance(checkpoint_paths, str):
            checkpoint_paths = [checkpoint_paths]
        result = []
        for path in checkpoint_paths:
            path = str(path).strip()
            if path:
                result.append(path)
        return result

    def _adapt_policy_for_runner_eval(
        self,
        policy: _policy_t,
        policy_observation_mode: str,
        policy_temporal_steps: int,
    ) -> _policy_t:
        runner_mode, runner_steps = self._get_observation_cfg()
        if (
            policy_observation_mode == runner_mode
            and (
                runner_mode == LEGACY_MODE or policy_temporal_steps == runner_steps
            )
        ):
            return policy

        if policy_observation_mode == AZ_HISTORY_MODE and runner_mode != AZ_HISTORY_MODE:
            raise RuntimeError(
                "az_history baseline requires runner observations to also be az_history."
            )

        logging.info(
            "Wrapping baseline with observation adapter: "
            f"runner(mode={runner_mode}, steps={runner_steps}) -> "
            f"baseline(mode={policy_observation_mode}, steps={policy_temporal_steps})"
        )
        return BaselineObservationAdapter(
            policy=policy,
            source_observation_mode=runner_mode,
            source_temporal_num_steps=runner_steps,
            target_observation_mode=policy_observation_mode,
            target_temporal_num_steps=policy_temporal_steps,
        )

    def _build_baseline_policy_from_checkpoint(
        self,
        checkpoint_path: str,
        *,
        tag: str,
    ) -> _policy_t:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"{tag} not found: {checkpoint_path}")

        baseline_observation_mode, baseline_temporal_steps = (
            self._get_baseline_observation_cfg()
        )
        baseline_env = self._make_env(
            num_envs=self.env.num_envs,
            observation_mode=baseline_observation_mode,
            temporal_num_steps=baseline_temporal_steps,
        )
        baseline = get_policy(
            name=self.cfg.baseline.name,
            cfg=self.cfg.baseline,
            action_spec=baseline_env.action_spec,
            observation_spec=baseline_env.observation_spec,
            device=baseline_env.device,
        )
        self._load_policy_checkpoint(baseline, checkpoint_path, tag)
        baseline.eval()
        return self._adapt_policy_for_runner_eval(
            baseline,
            policy_observation_mode=baseline_observation_mode,
            policy_temporal_steps=baseline_temporal_steps,
        )

    def _build_baseline_policy(self):
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

        ckpts.sort()
        return self._build_baseline_policy_from_checkpoint(
            ckpts[0],
            tag="Baseline",
        )

    def _build_eval_baseline_pool(self, checkpoint_paths, *, side_name: str) -> list[_policy_t]:
        pool = []
        for idx, checkpoint_path in enumerate(
            self._normalize_checkpoint_list(checkpoint_paths), start=1
        ):
            pool.append(
                self._build_baseline_policy_from_checkpoint(
                    checkpoint_path,
                    tag=f"eval_{side_name}_baseline{idx}",
                )
            )
        return pool


class Runner(_RunnerEnvMixin, abc.ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        observation_mode, temporal_num_steps = self._get_observation_cfg()

        self.env = self._make_env(
            num_envs=cfg.num_envs,
            observation_mode=observation_mode,
            temporal_num_steps=temporal_num_steps,
        )
        self.eval_env = self._make_env(
            num_envs=512,
            observation_mode=observation_mode,
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
            self._load_policy_checkpoint(
                self.policy_black,
                black_checkpoint,
                "black_checkpoint",
            )
        if white_checkpoint := cfg.get("white_checkpoint", None):
            self._load_policy_checkpoint(
                self.policy_white,
                white_checkpoint,
                "white_checkpoint",
            )

        eval_baseline_pool_cfg = cfg.get("eval_baseline_pool", {})
        self.eval_baseline_black_pool = self._build_eval_baseline_pool(
            eval_baseline_pool_cfg.get("black_pool", []),
            side_name="black_pool",
        )
        self.eval_baseline_white_pool = self._build_eval_baseline_pool(
            eval_baseline_pool_cfg.get("white_pool", []),
            side_name="white_pool",
        )

        run_dir = cfg.get("run_dir", None)
        if run_dir is None:
            run_dir = wandb.run.dir
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"run_dir:{run_dir}")
        self.run_dir = run_dir

    @abc.abstractmethod
    def _epoch(self, epoch: int) -> dict[str, Any]: ...

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
