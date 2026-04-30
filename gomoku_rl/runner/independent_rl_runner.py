from typing import Any

import torch
from omegaconf import DictConfig

from .base import Runner
from gomoku_rl.collector import VersusPlayCollector
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.utils.misc import add_prefix


class IndependentRLRunner(Runner):
    """
    InRL 双网络自博弈 runner。

    训练主线保持不变：rollout 仍然由 current_black vs current_white 产生。
    改动点是“EMA 越界时的交替更新调度”：

    - 默认：每个 rollout 后 black + white 都更新；
    - 若 eval/black_vs_white 的 EMA > upper：
        进入 white 偏置状态，后续 epoch 按
            white_only, both, white_only, both, ...
        交替更新；
    - 若 EMA < lower：
        进入 black 偏置状态，后续 epoch 按
            black_only, both, black_only, both, ...
        交替更新；
    - 若 EMA 回到 [lower, upper]：
        退出偏置状态，恢复为一直 both。

    注意：
    1) EMA 仍然只在 eval 时更新；
    2) 但 EMA 一旦越界，偏置状态会持续存在，直到后续某次 eval 把它关掉；
    3) 偏置状态里不是“持续只更一边”，而是“biased / both”交替。
    """

    _MODE_BOTH = "both"
    _MODE_BLACK_ONLY = "black_only"
    _MODE_WHITE_ONLY = "white_only"

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self._collector = VersusPlayCollector(
            self.env,
            self.policy_black,
            self.policy_white,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )

        self.log_interval = int(self.cfg.get("log_interval", 5))

        balance_cfg = self.cfg.get("balance", {})
        self.balance_enabled = bool(balance_cfg.get("enabled", True))
        self.balance_lower = float(balance_cfg.get("lower", 0.40))
        self.balance_upper = float(balance_cfg.get("upper", 0.60))
        self.balance_ema_alpha = float(balance_cfg.get("ema_alpha", 0.20))

        if not (0.0 <= self.balance_lower < self.balance_upper <= 1.0):
            raise ValueError(
                f"invalid balance bounds: lower={self.balance_lower}, upper={self.balance_upper}"
            )
        if not (0.0 < self.balance_ema_alpha <= 1.0):
            raise ValueError(
                f"invalid balance ema_alpha: {self.balance_ema_alpha}, expected in (0, 1]"
            )

        # 对 eval/black_vs_white 做 EMA；首次评估时直接用 raw 初始化。
        self.black_vs_white_ema: float | None = None

        # 当前 EMA 所决定的“偏置状态”
        # - both: 不偏置，始终双方都更新
        # - white_only: 启动 white 偏置，但实际应用为 white_only / both 交替
        # - black_only: 启动 black 偏置，但实际应用为 black_only / both 交替
        self.current_bias_mode = self._MODE_BOTH

        # 当 current_bias_mode != both 时，决定“下一轮”是走 biased 还是走 both。
        # 约定：
        #   True  -> 下一轮走 biased（white_only 或 black_only）
        #   False -> 下一轮走 both
        #
        # 因为触发 eval 的那个 epoch 已经训练完了，所以一旦新进入偏置状态，
        # 下一轮先走 biased，形成整体上的：
        #   ... 当前这个 eval epoch(通常是 both) -> biased -> both -> biased -> ...
        self.bias_turn_next = False

    def _applied_mode_to_flags(self, mode: str) -> dict[str, float]:
        return {
            "balance/applied_both": float(mode == self._MODE_BOTH),
            "balance/applied_black_only": float(mode == self._MODE_BLACK_ONLY),
            "balance/applied_white_only": float(mode == self._MODE_WHITE_ONLY),
        }

    def _bias_mode_to_flags(self, mode: str) -> dict[str, float]:
        return {
            "balance/bias_mode_both": float(mode == self._MODE_BOTH),
            "balance/bias_mode_black_only": float(mode == self._MODE_BLACK_ONLY),
            "balance/bias_mode_white_only": float(mode == self._MODE_WHITE_ONLY),
        }

    def _phase_flags(self) -> dict[str, float]:
        if self.current_bias_mode == self._MODE_BOTH:
            return {
                "balance/bias_active": 0.0,
                "balance/next_turn_biased": 0.0,
                "balance/next_turn_both": 1.0,
            }
        return {
            "balance/bias_active": 1.0,
            "balance/next_turn_biased": float(self.bias_turn_next),
            "balance/next_turn_both": float(not self.bias_turn_next),
        }

    def _ema_trigger_flags(self) -> dict[str, float]:
        if self.black_vs_white_ema is None:
            return {
                "balance/trigger_lower": 0.0,
                "balance/trigger_upper": 0.0,
            }
        return {
            "balance/trigger_lower": float(self.black_vs_white_ema < self.balance_lower),
            "balance/trigger_upper": float(self.black_vs_white_ema > self.balance_upper),
        }

    def _apply_learning(
        self,
        applied_mode: str,
        data_black,
        data_white,
        info: dict[str, Any],
    ) -> None:
        info.update(self._applied_mode_to_flags(applied_mode))
        info["balance/black_update_skipped"] = float(
            applied_mode == self._MODE_WHITE_ONLY
        )
        info["balance/white_update_skipped"] = float(
            applied_mode == self._MODE_BLACK_ONLY
        )

        if applied_mode in (self._MODE_BOTH, self._MODE_BLACK_ONLY):
            info.update(
                add_prefix(
                    self.policy_black.learn(data_black.to_tensordict()),
                    "policy_black/",
                )
            )
        del data_black

        if applied_mode in (self._MODE_BOTH, self._MODE_WHITE_ONLY):
            info.update(
                add_prefix(
                    self.policy_white.learn(data_white.to_tensordict()),
                    "policy_white/",
                )
            )
        del data_white

    def _update_black_vs_white_ema(self, black_vs_white_raw: float) -> float:
        if self.black_vs_white_ema is None:
            ema = black_vs_white_raw
        else:
            ema = (
                self.balance_ema_alpha * black_vs_white_raw
                + (1.0 - self.balance_ema_alpha) * self.black_vs_white_ema
            )
        self.black_vs_white_ema = float(ema)
        return self.black_vs_white_ema

    def _decide_bias_mode_from_ema(self) -> str:
        if not self.balance_enabled:
            return self._MODE_BOTH

        if self.black_vs_white_ema is None:
            return self._MODE_BOTH

        if self.black_vs_white_ema > self.balance_upper:
            return self._MODE_WHITE_ONLY

        if self.black_vs_white_ema < self.balance_lower:
            return self._MODE_BLACK_ONLY

        return self._MODE_BOTH

    def _update_bias_state_after_eval(self) -> None:
        """
        根据最新 EMA 更新“偏置状态”。

        规则：
        - 从 both 进入某个偏置状态时：下一轮先走 biased；
        - 同一偏置状态持续时：保持原来的交替相位，不重置；
        - 偏置方向切换时：下一轮重新从新的 biased 开始；
        - 回到区间内时：退出偏置，恢复 always both。
        """
        prev_bias_mode = self.current_bias_mode
        new_bias_mode = self._decide_bias_mode_from_ema()

        if new_bias_mode == self._MODE_BOTH:
            self.current_bias_mode = self._MODE_BOTH
            self.bias_turn_next = False
            return

        # 进入偏置，或偏置方向切换：下一轮从 biased 开始
        if prev_bias_mode != new_bias_mode:
            self.current_bias_mode = new_bias_mode
            self.bias_turn_next = True
            return

        # same bias mode: 保持原有相位，不打断当前 alternating 节奏
        self.current_bias_mode = new_bias_mode

    def _get_applied_mode_for_current_epoch(self) -> str:
        """
        给当前 epoch 计算实际应用的更新模式。

        - 若无偏置：always both
        - 若有偏置：biased / both 交替
        """
        if self.current_bias_mode == self._MODE_BOTH:
            return self._MODE_BOTH

        applied_mode = (
            self.current_bias_mode if self.bias_turn_next else self._MODE_BOTH
        )
        self.bias_turn_next = not self.bias_turn_next
        return applied_mode

    def _eval_current_black_vs_white_pool(self) -> dict[str, float]:
        if not self.eval_baseline_white_pool:
            return {}

        info: dict[str, float] = {}
        scores: list[float] = []
        for idx, baseline in enumerate(self.eval_baseline_white_pool, start=1):
            score = float(
                eval_win_rate(
                    self.eval_env,
                    player_black=self.policy_black,
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
                    player_white=self.policy_white,
                )
            )
            info[f"eval/black_pool{idx}_vs_white"] = score
            scores.append(score)

        mean_score = float(sum(scores) / len(scores))
        info["eval/black_pool_vs_white_mean"] = mean_score
        return info

    def _epoch(self, epoch: int) -> dict[str, Any]:
        data_black, data_white, info = self._collector.rollout(self.steps)
        info = add_prefix(info, "versus_play/")

        info["fps"] = info["versus_play/fps"]
        del info["versus_play/fps"]

        applied_mode = self._get_applied_mode_for_current_epoch()

        self._apply_learning(
            applied_mode=applied_mode,
            data_black=data_black,
            data_white=data_white,
            info=info,
        )

        # 每个 epoch 都把当前 balance 状态打出来，方便在 wandb 看持续状态与交替相位
        info.update(
            {
                "balance/enabled": float(self.balance_enabled),
                "balance/lower": self.balance_lower,
                "balance/upper": self.balance_upper,
                "balance/ema_alpha": self.balance_ema_alpha,
            }
        )
        info.update(self._bias_mode_to_flags(self.current_bias_mode))
        info.update(self._phase_flags())
        info.update(self._ema_trigger_flags())

        if self.black_vs_white_ema is not None:
            info["eval/black_vs_white_ema"] = self.black_vs_white_ema

        if epoch % 50 == 0 and epoch != 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return info

    def _post_run(self):
        pass

    def _format_eval_summary(self, info: dict[str, Any]) -> str:
        parts = [
            f"Black vs White:{info['eval/black_vs_white'] * 100.0:.2f}%",
            f"EMA:{info['eval/black_vs_white_ema'] * 100.0:.2f}%",
        ]
        if "eval/black_vs_white_pool_mean" in info:
            parts.append(f"Black vs WhitePool(mean):{info['eval/black_vs_white_pool_mean'] * 100.0:.2f}%")
        if "eval/black_pool_vs_white_mean" in info:
            parts.append(f"BlackPool(mean) vs White:{info['eval/black_pool_vs_white_mean'] * 100.0:.2f}%")
        parts.extend(
            [
                f"white_pool:{len(self.eval_baseline_white_pool)}",
                f"black_pool:{len(self.eval_baseline_black_pool)}",
                f"bias_mode:{self.current_bias_mode}",
                "next_turn:{}".format(
                    "biased"
                    if (
                        self.current_bias_mode != self._MODE_BOTH
                        and self.bias_turn_next
                    )
                    else "both"
                ),
            ]
        )
        return "	".join(parts)

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % self.log_interval == 0:
            black_vs_white_raw = float(
                eval_win_rate(
                    self.eval_env,
                    player_black=self.policy_black,
                    player_white=self.policy_white,
                )
            )

            black_vs_white_ema = self._update_black_vs_white_ema(black_vs_white_raw)
            self._update_bias_state_after_eval()

            info.update(
                {
                    "eval/black_vs_white": black_vs_white_raw,
                    "eval/black_vs_white_ema": black_vs_white_ema,
                    "balance/enabled": float(self.balance_enabled),
                    "balance/lower": self.balance_lower,
                    "balance/upper": self.balance_upper,
                    "balance/ema_alpha": self.balance_ema_alpha,
                    "eval/white_pool_size": float(len(self.eval_baseline_white_pool)),
                    "eval/black_pool_size": float(len(self.eval_baseline_black_pool)),
                }
            )
            info.update(self._eval_current_black_vs_white_pool())
            info.update(self._eval_black_pool_vs_current_white())
            info.update(self._ema_trigger_flags())
            info.update(self._bias_mode_to_flags(self.current_bias_mode))
            info.update(self._phase_flags())

            print(self._format_eval_summary(info))
        else:
            # 非 eval epoch 也持续记录当前状态
            info.update(
                {
                    "balance/enabled": float(self.balance_enabled),
                    "balance/lower": self.balance_lower,
                    "balance/upper": self.balance_upper,
                    "balance/ema_alpha": self.balance_ema_alpha,
                    "eval/white_pool_size": float(len(self.eval_baseline_white_pool)),
                    "eval/black_pool_size": float(len(self.eval_baseline_black_pool)),
                }
            )
            info.update(self._bias_mode_to_flags(self.current_bias_mode))
            info.update(self._phase_flags())
            info.update(self._ema_trigger_flags())

            if self.black_vs_white_ema is not None:
                info["eval/black_vs_white_ema"] = self.black_vs_white_ema

        return super()._log(info, epoch)
