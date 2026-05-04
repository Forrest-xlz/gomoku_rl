"""Independent-RL runner without baseline evaluation.

Training is still current_black vs current_white. Baseline pool evaluation has
been removed. Training-time model evaluation now uses the same metadata-driven
Elo model pool as PSRO.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import wandb
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from gomoku_rl.collector import VersusPlayCollector
from gomoku_rl.runner.base import Runner
from gomoku_rl.runner.elo_model_pool import EloEvalMixin, wandb_save_file
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.utils.misc import add_prefix


class IndependentRLRunner(EloEvalMixin, Runner):
    """InRL two-network self-play runner.

    Baseline evaluation has been removed. The only evaluation paths are:
    1. current_black vs current_white for balance control;
    2. metadata-driven Elo pool evaluation every `elo_interval` epochs.
    """

    _MODE_BOTH = "both"
    _MODE_BLACK_ONLY = "black_only"
    _MODE_WHITE_ONLY = "white_only"

    def __init__(self, cfg: DictConfig) -> None:
        # Defensive compatibility: even if an old config still contains
        # eval_baseline_pool, do not load or evaluate those models.
        with open_dict(cfg):
            cfg.eval_baseline_pool = {"black_pool": [], "white_pool": []}

        super().__init__(cfg)

        self._collector = VersusPlayCollector(
            self.env,
            self.policy_black,
            self.policy_white,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )

        self.log_interval = int(self.cfg.get("log_interval", 5))
        self.pretrain_epoch_offset = int(self.cfg.get("pretrain_epoch_offset", 0))
        if self.pretrain_epoch_offset < 0:
            raise ValueError(
                f"pretrain_epoch_offset must be >= 0, got {self.pretrain_epoch_offset}"
            )

        self._init_elo_league(source="train_inrl")

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

        self.black_vs_white_ema: float | None = None
        self.current_bias_mode = self._MODE_BOTH
        self.bias_turn_next = False

    # ---------------------------- schedule helpers ----------------------------
    def _set_policy_schedules(self, epoch: int, info: dict[str, Any]) -> None:
        """Apply per-outer-epoch LR / entropy schedules before learning.

        ``epoch`` is the zero-based local outer epoch from tqdm. When training is
        resumed from checkpoints, ``pretrain_epoch_offset`` is added so decay
        continues from the global outer epoch.
        """
        schedule_epoch = int(self.pretrain_epoch_offset + epoch)

        for policy_name, policy in (
            ("policy_black", self.policy_black),
            ("policy_white", self.policy_white),
        ):
            if hasattr(policy, "set_outer_epoch"):
                policy.set_outer_epoch(schedule_epoch)
            if hasattr(policy, "get_schedule_info"):
                info.update(add_prefix(policy.get_schedule_info(), f"{policy_name}/"))

    # ---------------------------- balance helpers ----------------------------
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
            return {"balance/trigger_lower": 0.0, "balance/trigger_upper": 0.0}
        return {
            "balance/trigger_lower": float(self.black_vs_white_ema < self.balance_lower),
            "balance/trigger_upper": float(self.black_vs_white_ema > self.balance_upper),
        }

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
        if not self.balance_enabled or self.black_vs_white_ema is None:
            return self._MODE_BOTH
        if self.black_vs_white_ema > self.balance_upper:
            return self._MODE_WHITE_ONLY
        if self.black_vs_white_ema < self.balance_lower:
            return self._MODE_BLACK_ONLY
        return self._MODE_BOTH

    def _update_bias_state_after_eval(self) -> None:
        prev_bias_mode = self.current_bias_mode
        new_bias_mode = self._decide_bias_mode_from_ema()

        if new_bias_mode == self._MODE_BOTH:
            self.current_bias_mode = self._MODE_BOTH
            self.bias_turn_next = False
            return

        if prev_bias_mode != new_bias_mode:
            self.current_bias_mode = new_bias_mode
            self.bias_turn_next = True
            return

        self.current_bias_mode = new_bias_mode

    def _get_applied_mode_for_current_epoch(self) -> str:
        if self.current_bias_mode == self._MODE_BOTH:
            return self._MODE_BOTH

        applied_mode = self.current_bias_mode if self.bias_turn_next else self._MODE_BOTH
        self.bias_turn_next = not self.bias_turn_next
        return applied_mode

    def _apply_learning(
        self,
        applied_mode: str,
        data_black,
        data_white,
        info: dict[str, Any],
    ) -> None:
        info.update(self._applied_mode_to_flags(applied_mode))
        info["balance/black_update_skipped"] = float(applied_mode == self._MODE_WHITE_ONLY)
        info["balance/white_update_skipped"] = float(applied_mode == self._MODE_BLACK_ONLY)

        if applied_mode in (self._MODE_BOTH, self._MODE_BLACK_ONLY):
            info.update(add_prefix(self.policy_black.learn(data_black.to_tensordict()), "policy_black/"))
        del data_black

        if applied_mode in (self._MODE_BOTH, self._MODE_WHITE_ONLY):
            info.update(add_prefix(self.policy_white.learn(data_white.to_tensordict()), "policy_white/"))
        del data_white

    # ------------------------------ training ---------------------------------
    def _epoch(self, epoch: int) -> dict[str, Any]:
        schedule_info: dict[str, Any] = {}
        self._set_policy_schedules(epoch=epoch, info=schedule_info)

        data_black, data_white, info = self._collector.rollout(self.steps)
        info = add_prefix(info, "versus_play/")
        info["fps"] = info["versus_play/fps"]
        del info["versus_play/fps"]

        # Log schedules for both policies even if balance control skips one side.
        info.update(schedule_info)

        applied_mode = self._get_applied_mode_for_current_epoch()
        self._apply_learning(
            applied_mode=applied_mode,
            data_black=data_black,
            data_white=data_white,
            info=info,
        )

        info.update(
            {
                "time/local_epoch_completed": float(epoch + 1),
                "time/global_epoch_completed": float(self._completed_epoch(epoch)),
                "time/pretrain_epoch_offset": float(self.pretrain_epoch_offset),
                "balance/enabled": float(self.balance_enabled),
                "balance/lower": self.balance_lower,
                "balance/upper": self.balance_upper,
                "balance/ema_alpha": self.balance_ema_alpha,
                "elo_eval/interval": float(self.elo_interval),
                "elo_eval/num_models": float(len(self.elo_league)),
                "elo_eval/payoff_coverage": self.elo_league.payoff_coverage(include_self=True),
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

    # ------------------------------- logging ---------------------------------
    def _format_eval_summary(self, info: dict[str, Any]) -> str:
        parts = [
            f"Black vs White:{info['eval/black_vs_white'] * 100.0:.2f}%",
            f"EMA:{info['eval/black_vs_white_ema'] * 100.0:.2f}%",
            f"elo_models:{len(self.elo_league)}",
            f"bias_mode:{self.current_bias_mode}",
            "next_turn:{}".format(
                "biased"
                if (self.current_bias_mode != self._MODE_BOTH and self.bias_turn_next)
                else "both"
            ),
        ]
        if "elo_eval/current_black" in info:
            parts.append(f"EloB:{info['elo_eval/current_black']:.1f}")
            parts.append(f"EloW:{info['elo_eval/current_white']:.1f}")
            parts.append(f"Brank:{int(info['elo_eval/current_black_rank'])}")
            parts.append(f"Wrank:{int(info['elo_eval/current_white_rank'])}")
        return " ".join(parts)

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
                    "elo_eval/num_models": float(len(self.elo_league)),
                    "elo_eval/payoff_coverage": self.elo_league.payoff_coverage(include_self=True),
                }
            )
            info.update(self._ema_trigger_flags())
            info.update(self._bias_mode_to_flags(self.current_bias_mode))
            info.update(self._phase_flags())

        self._maybe_update_global_elo(epoch, info)

        if epoch % self.log_interval == 0 or "elo_eval/current_black" in info:
            if "eval/black_vs_white" not in info:
                black_vs_white_raw = float(
                    eval_win_rate(
                        self.eval_env,
                        player_black=self.policy_black,
                        player_white=self.policy_white,
                    )
                )
                black_vs_white_ema = self._update_black_vs_white_ema(black_vs_white_raw)
                self._update_bias_state_after_eval()
                info["eval/black_vs_white"] = black_vs_white_raw
                info["eval/black_vs_white_ema"] = black_vs_white_ema

            print(self._format_eval_summary(info))
        else:
            info.update(self._bias_mode_to_flags(self.current_bias_mode))
            info.update(self._phase_flags())
            info.update(self._ema_trigger_flags())
            if self.black_vs_white_ema is not None:
                info["eval/black_vs_white_ema"] = self.black_vs_white_ema

        return super()._log(info, epoch)

    # ---------------------------- checkpointing ------------------------------
    def run(self, disable_tqdm: bool = False):
        pbar = tqdm(range(self.epochs), disable=disable_tqdm)
        for i in pbar:
            info: dict[str, Any] = {}
            info.update(self._epoch(epoch=i))
            self._log(info=info, epoch=i)

            if self.save_interval > 0 and (i + 1) % self.save_interval == 0:
                ckpt_epoch = self._completed_epoch(i)
                epoch_label = self._epoch_label(ckpt_epoch)

                black_path = os.path.join(self.run_dir, f"black_{epoch_label}.pt")
                white_path = os.path.join(self.run_dir, f"white_{epoch_label}.pt")

                torch.save(self.policy_black.state_dict(), black_path)
                torch.save(self.policy_white.state_dict(), white_path)

                wandb_save_file(black_path, base_path=self.run_dir)
                wandb_save_file(white_path, base_path=self.run_dir)

                self._save_elo_snapshot(epoch_label)

            pbar.set_postfix(
                {
                    "fps": info.get("fps", 0.0),
                    "eloN": len(self.elo_league),
                    "epoch": self._completed_epoch(i),
                    "lrB": info.get("policy_black/schedule/lr", float("nan")),
                    "entB": info.get("policy_black/schedule/entropy_coef", float("nan")),
                    "eloB": info.get("elo_eval/current_black", float("nan")),
                    "eloW": info.get("elo_eval/current_white", float("nan")),
                }
            )

        torch.save(
            self.policy_black.state_dict(),
            os.path.join(self.run_dir, "black_final.pt"),
        )
        torch.save(
            self.policy_white.state_dict(),
            os.path.join(self.run_dir, "white_final.pt"),
        )

        wandb_save_file(os.path.join(self.run_dir, "black_final.pt"), base_path=self.run_dir)
        wandb_save_file(os.path.join(self.run_dir, "white_final.pt"), base_path=self.run_dir)

        self._save_elo_snapshot("final")
        self._post_run()
