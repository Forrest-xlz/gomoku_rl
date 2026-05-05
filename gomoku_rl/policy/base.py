"""Base runner utilities shared by InRL and PSRO.

This version keeps the original runner behavior, but allows role-specific PPO
configuration through ``cfg.algo.black`` and ``cfg.algo.white``. The role config
is built by taking the top-level ``cfg.algo`` as defaults and then overlaying the
role-specific section.
"""

from __future__ import annotations

import abc
import logging
import os
from typing import Any, Tuple

import torch
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from gomoku_rl.env import LEGACY_MODE, TEMPORAL_MOVE_HISTORY_MODE, GomokuEnv
from gomoku_rl.policy import get_policy
from gomoku_rl.utils.misc import set_seed


class _RunnerEnvMixin:
    def _get_observation_cfg(self) -> Tuple[str, int]:
        observation_mode = self.cfg.get("observation_mode", None)
        if observation_mode is None:
            use_temporal_feature = bool(self.cfg.get("use_temporal_feature", False))
            observation_mode = TEMPORAL_MOVE_HISTORY_MODE if use_temporal_feature else LEGACY_MODE
        else:
            observation_mode = str(observation_mode)

        temporal_num_steps = int(self.cfg.get("temporal_num_steps", 6))
        if temporal_num_steps < 1:
            raise ValueError("temporal_num_steps must be >= 1")
        return observation_mode, temporal_num_steps

    def _make_env(self, num_envs: int, observation_mode: str, temporal_num_steps: int):
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
        logging.info("%s:%s", tag, checkpoint_path)


class Runner(_RunnerEnvMixin, abc.ABC):
    _ROLE_CFG_KEYS = {"black", "white"}
    _NON_DEFAULT_ALGO_KEYS = {"black", "white", "common"}

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        observation_mode, temporal_num_steps = self._get_observation_cfg()
        self.env = self._make_env(
            num_envs=cfg.num_envs,
            observation_mode=observation_mode,
            temporal_num_steps=temporal_num_steps,
        )
        self.eval_env = self._make_env(
            num_envs=int(cfg.get("eval_num_envs", 512)),
            observation_mode=observation_mode,
            temporal_num_steps=temporal_num_steps,
        )

        seed = cfg.get("seed", None)
        set_seed(seed)

        self.epochs: int = cfg.get("epochs")
        self.steps = cfg.steps
        self.save_interval: int = cfg.get("save_interval", -1)
        self.pretrain_epoch_offset = int(cfg.get("pretrain_epoch_offset", 0))

        self.algo_black_cfg = self._make_role_algo_cfg("black")
        self.algo_white_cfg = self._make_role_algo_cfg("white")

        self.policy_black = get_policy(
            name=self.algo_black_cfg.name,
            cfg=self.algo_black_cfg,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )
        self.policy_white = get_policy(
            name=self.algo_white_cfg.name,
            cfg=self.algo_white_cfg,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )

        if black_checkpoint := cfg.get("black_checkpoint", None):
            self._load_policy_checkpoint(self.policy_black, black_checkpoint, "black_checkpoint")
        if white_checkpoint := cfg.get("white_checkpoint", None):
            self._load_policy_checkpoint(self.policy_white, white_checkpoint, "white_checkpoint")

        run_dir = cfg.get("run_dir", None)
        if run_dir is None:
            run_dir = wandb.run.dir if wandb.run is not None else os.getcwd()
        os.makedirs(run_dir, exist_ok=True)
        logging.info("run_dir:%s", run_dir)
        self.run_dir = run_dir

    def _make_role_algo_cfg(self, role: str) -> DictConfig:
        """Return the algorithm config for one role.

        Backward compatibility:
            If ``cfg.algo.black`` / ``cfg.algo.white`` is absent, this returns
            ``cfg.algo`` exactly as before.

        New behavior:
            If the role section exists, the top-level ``cfg.algo`` entries are
            used as defaults and the role section overrides them. This allows
            compact configs such as::

                algo:
                  name: ppo
                  clip_param: 0.2
                  optimizer:
                    name: adam
                    kwargs:
                      lr: 1e-4
                  black:
                    optimizer:
                      kwargs:
                        lr: 8e-5
                  white:
                    entropy_coef: 0.015

        """
        if role not in self._ROLE_CFG_KEYS:
            raise ValueError(f"unknown role config '{role}', expected one of {sorted(self._ROLE_CFG_KEYS)}")

        algo_cfg: DictConfig = self.cfg.algo
        role_cfg = algo_cfg.get(role, None)
        if role_cfg is None:
            return algo_cfg

        # Build defaults from cfg.algo while excluding role-specific sections.
        # If cfg.algo.common exists, it is merged between the top-level defaults
        # and the role-specific override. OmegaConf.merge keeps nested defaults,
        # so a role override like ``optimizer.kwargs.lr`` does not need to repeat
        # ``optimizer.name``.
        base_container: dict[str, Any] = {}
        for key, value in algo_cfg.items():
            if key in self._NON_DEFAULT_ALGO_KEYS:
                continue
            base_container[key] = value

        base_cfg = OmegaConf.create(base_container)
        common_cfg = algo_cfg.get("common", None)
        if common_cfg is not None:
            base_cfg = OmegaConf.merge(base_cfg, common_cfg)

        merged = OmegaConf.merge(base_cfg, role_cfg)
        with open_dict(merged):
            if merged.get("name", None) is None:
                merged.name = algo_cfg.name
            merged.role = role
        return merged

    def _completed_epoch(self, local_epoch: int) -> int:
        return int(self.pretrain_epoch_offset + local_epoch + 1)

    def _epoch_label(self, epoch_value: int) -> str:
        return f"{int(epoch_value):05d}"

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
            info: dict[str, Any] = {}
            info.update(self._epoch(epoch=i))
            self._log(info=info, epoch=i)

            if self.save_interval > 0 and (i + 1) % self.save_interval == 0:
                ckpt_epoch = self._completed_epoch(i)
                epoch_label = self._epoch_label(ckpt_epoch)
                torch.save(self.policy_black.state_dict(), os.path.join(self.run_dir, f"black_{epoch_label}.pt"))
                torch.save(self.policy_white.state_dict(), os.path.join(self.run_dir, f"white_{epoch_label}.pt"))

            pbar.set_postfix({"fps": info.get("fps", 0.0), "epoch": self._completed_epoch(i)})

        torch.save(self.policy_black.state_dict(), os.path.join(self.run_dir, "black_final.pt"))
        torch.save(self.policy_white.state_dict(), os.path.join(self.run_dir, "white_final.pt"))
        self._post_run()
