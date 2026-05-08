#!/usr/bin/env python3
"""Hydra entry point for ordered bucket-curriculum Gomoku training."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from gomoku_rl import CONFIG_PATH
from gomoku_rl.runner.bucket_curriculum_rl_runner import BucketCurriculumRLRunner
from gomoku_rl.utils.wandb import init_wandb


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_bucket_curriculum")
def main(cfg: DictConfig) -> None:
    # Keep this consistent with scripts/train_psro.py so interpolation and
    # command-line overrides are resolved before wandb receives the config.
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    init_wandb(cfg=cfg)

    runner = BucketCurriculumRLRunner(cfg=cfg)
    runner.run()


if __name__ == "__main__":
    main()
