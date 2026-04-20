from typing import Any

import torch
from omegaconf import DictConfig

from gomoku_rl.collector import SelfPlayCollector
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.utils.misc import add_prefix

from .base import SPRunner


class IndependentRLAZSPRunner(SPRunner):
    """Single-policy self-play runner with AZ-style absolute-color history input.

    Key differences from the original IndependentRLSPRunner:
    - observation_mode is expected to be "az_history"
    - eval metric names are aligned with IndependentRLRunner:
      * eval/black_vs_white
      * eval/black_vs_white_pool{idx}
      * eval/black_pool{idx}_vs_white
    - action pruning is inherited from cfg.action_pruning through base._make_env()
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._collector = SelfPlayCollector(
            self.env,
            self.policy,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )
        self.log_interval = int(self.cfg.get("log_interval", 5))

    def _epoch(self, epoch: int) -> dict[str, Any]:
        data, info = self._collector.rollout(self.steps)
        info = add_prefix(info, "self_play/")
        info["fps"] = info["self_play/fps"]
        del info["self_play/fps"]
        info.update(add_prefix(self.policy.learn(data.to_tensordict()), "policy/"))
        del data

        if epoch % 50 == 0 and epoch != 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return info

    def _eval_current_black_vs_white_pool(self) -> dict[str, float]:
        if not self.eval_baseline_white_pool:
            return {}

        info: dict[str, float] = {}
        scores: list[float] = []
        for idx, baseline in enumerate(self.eval_baseline_white_pool, start=1):
            score = float(
                eval_win_rate(
                    self.eval_env,
                    player_black=self.policy,
                    player_white=baseline,
                )
            )
            info[f"eval/black_vs_white_pool{idx}"] = score
            scores.append(score)

        mean_score = float(sum(scores) / len(scores))
        info["eval/black_vs_white_pool_mean"] = mean_score
        return info

    def _eval_black_pool_vs_current_white(self) -> dict[str, float]:
        if not self.eval_baseline_black_pool:
            return {}

        info: dict[str, float] = {}
        scores: list[float] = []
        for idx, baseline in enumerate(self.eval_baseline_black_pool, start=1):
            score = float(
                eval_win_rate(
                    self.eval_env,
                    player_black=baseline,
                    player_white=self.policy,
                )
            )
            info[f"eval/black_pool{idx}_vs_white"] = score
            scores.append(score)

        mean_score = float(sum(scores) / len(scores))
        info["eval/black_pool_vs_white_mean"] = mean_score
        return info

    def _format_eval_summary(self, info: dict[str, Any]) -> str:
        parts = [f"Black vs White:{info['eval/black_vs_white'] * 100.0:.2f}%"]
        if "eval/black_vs_white_pool_mean" in info:
            parts.append(
                f"Black vs WhitePool(mean):{info['eval/black_vs_white_pool_mean'] * 100.0:.2f}%"
            )
        if "eval/black_pool_vs_white_mean" in info:
            parts.append(
                f"BlackPool(mean) vs White:{info['eval/black_pool_vs_white_mean'] * 100.0:.2f}%"
            )
        parts.extend(
            [
                f"white_pool:{len(self.eval_baseline_white_pool)}",
                f"black_pool:{len(self.eval_baseline_black_pool)}",
            ]
        )
        return " ".join(parts)

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % self.log_interval == 0:
            black_vs_white = float(
                eval_win_rate(
                    self.eval_env,
                    player_black=self.policy,
                    player_white=self.policy,
                )
            )
            info.update(
                {
                    "eval/black_vs_white": black_vs_white,
                    # backward-compatible aliases with the old SP runner
                    "eval/player_vs_player": black_vs_white,
                    "eval/white_pool_size": float(len(self.eval_baseline_white_pool)),
                    "eval/black_pool_size": float(len(self.eval_baseline_black_pool)),
                }
            )
            info.update(self._eval_current_black_vs_white_pool())
            info.update(self._eval_black_pool_vs_current_white())
            print(self._format_eval_summary(info))

        return super()._log(info, epoch)
