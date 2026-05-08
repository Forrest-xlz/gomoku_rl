#!/usr/bin/env python3
"""Hydra entry point for CPU-buffered mixed-teacher curriculum Gomoku training."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from gomoku_rl import CONFIG_PATH
from gomoku_rl.runner.mixed_teacher_curriculum_rl_runner_cpu_buffer import MixedTeacherCurriculumRLRunner
from gomoku_rl.utils.wandb import init_wandb


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_mixed_teacher_curriculum_cpu_buffer")
def main(cfg: DictConfig) -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    init_wandb(cfg=cfg)

    runner = MixedTeacherCurriculumRLRunner(cfg=cfg)
    runner.run()


if __name__ == "__main__":
    main()
