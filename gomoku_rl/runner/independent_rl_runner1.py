import copy
import os
from typing import Any

import torch
from omegaconf import DictConfig

from .base import SPRunner, Runner
from gomoku_rl.utils.misc import add_prefix
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.collector import SelfPlayCollector, VersusPlayCollector


class IndependentRLRunner(Runner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        # 训练仍然保持 current_black vs current_white
        self._collector = VersusPlayCollector(
            self.env,
            self.policy_black,
            self.policy_white,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )

        self.log_interval = int(self.cfg.get("log_interval", 5))

        gate_cfg = self.cfg.get("gate", {})
        self.gate_enabled = bool(gate_cfg.get("enabled", True))
        self.gate_interval = int(gate_cfg.get("interval", 20))
        self.gate_warmup_epochs = int(gate_cfg.get("warmup_epochs", 100))
        self.gate_eval_n = int(gate_cfg.get("eval_n", 3))
        self.gate_accept_margin = float(gate_cfg.get("accept_margin", 0.02))
        self.gate_collapse_margin = float(gate_cfg.get("collapse_margin", 0.08))

        # best 快照：从当前加载完 checkpoint 的策略初始化
        self.best_policy_black = copy.deepcopy(self.policy_black)
        self.best_policy_white = copy.deepcopy(self.policy_white)

        if hasattr(self.best_policy_black, "eval"):
            self.best_policy_black.eval()
        if hasattr(self.best_policy_white, "eval"):
            self.best_policy_white.eval()

        self.best_black_path = os.path.join(self.run_dir, "best_black.pt")
        self.best_white_path = os.path.join(self.run_dir, "best_white.pt")
        self._save_best_snapshots()

        self.best_pair_black_wr = self._eval_black_wr(
            self.best_policy_black, self.best_policy_white
        )
        self.best_pair_white_wr = 1.0 - self.best_pair_black_wr

    def _eval_black_wr(self, player_black, player_white) -> float:
        return float(
            eval_win_rate(
                self.eval_env,
                player_black=player_black,
                player_white=player_white,
                n=self.gate_eval_n,
            )
        )

    def _eval_white_wr(self, player_black, player_white) -> float:
        # eval_win_rate 返回的是黑方胜率，因此白方胜率 = 1 - 黑方胜率
        return 1.0 - self._eval_black_wr(player_black, player_white)

    def _save_best_snapshots(self) -> None:
        torch.save(self.best_policy_black.state_dict(), self.best_black_path)
        torch.save(self.best_policy_white.state_dict(), self.best_white_path)

    def _reset_env_after_gate(self) -> None:
        # 门控后重置环境，避免回滚后继续沿用旧 rollout 尾部状态
        self.env.reset()
        self.eval_env.reset()

    def _maybe_gate(self, epoch: int, info: dict[str, Any]) -> None:
        if not self.gate_enabled:
            return
        if epoch < self.gate_warmup_epochs:
            return
        if (epoch + 1) % self.gate_interval != 0:
            return

        # 固定参照：旧 best_black vs 旧 best_white
        ref_black_wr = self._eval_black_wr(
            self.best_policy_black, self.best_policy_white
        )
        ref_white_wr = 1.0 - ref_black_wr

        # 候选分数
        cand_black_wr = self._eval_black_wr(
            self.policy_black, self.best_policy_white
        )
        cand_white_wr = self._eval_white_wr(
            self.best_policy_black, self.policy_white
        )

        black_accept = cand_black_wr >= ref_black_wr + self.gate_accept_margin
        black_collapse = cand_black_wr <= ref_black_wr - self.gate_collapse_margin

        white_accept = cand_white_wr >= ref_white_wr + self.gate_accept_margin
        white_collapse = cand_white_wr <= ref_white_wr - self.gate_collapse_margin

        info.update(
            {
                "gate/ref_black_wr": ref_black_wr,
                "gate/ref_white_wr": ref_white_wr,
                "gate/cand_black_wr": cand_black_wr,
                "gate/cand_white_wr": cand_white_wr,
                "gate/black_accept": float(black_accept),
                "gate/black_collapse": float(black_collapse),
                "gate/white_accept": float(white_accept),
                "gate/white_collapse": float(white_collapse),
                "gate/black_keep": float((not black_accept) and (not black_collapse)),
                "gate/white_keep": float((not white_accept) and (not white_collapse)),
                "gate/black_rollback": 0.0,
                "gate/white_rollback": 0.0,
            }
        )

        any_accept = False
        any_rollback = False

        # 黑门控
        if black_accept:
            self.best_policy_black.load_state_dict(self.policy_black.state_dict())
            if hasattr(self.best_policy_black, "eval"):
                self.best_policy_black.eval()
            any_accept = True
        elif black_collapse:
            self.policy_black.load_state_dict(self.best_policy_black.state_dict())
            info["gate/black_rollback"] = 1.0
            any_rollback = True

        # 白门控
        if white_accept:
            self.best_policy_white.load_state_dict(self.policy_white.state_dict())
            if hasattr(self.best_policy_white, "eval"):
                self.best_policy_white.eval()
            any_accept = True
        elif white_collapse:
            self.policy_white.load_state_dict(self.best_policy_white.state_dict())
            info["gate/white_rollback"] = 1.0
            any_rollback = True

        if any_accept:
            self._save_best_snapshots()

        self.best_pair_black_wr = self._eval_black_wr(
            self.best_policy_black, self.best_policy_white
        )
        self.best_pair_white_wr = 1.0 - self.best_pair_black_wr

        info.update(
            {
                "gate/best_pair_black_wr": self.best_pair_black_wr,
                "gate/best_pair_white_wr": self.best_pair_white_wr,
            }
        )

        if any_accept or any_rollback:
            self._reset_env_after_gate()

        def _status(acc: bool, col: bool) -> str:
            if acc:
                return "accept"
            if col:
                return "rollback"
            return "keep"

        print(
            "[gate] epoch {:04d} | "
            "black: cand {:.3f}, ref {:.3f}, action={} | "
            "white: cand {:.3f}, ref {:.3f}, action={}".format(
                epoch,
                cand_black_wr,
                ref_black_wr,
                _status(black_accept, black_collapse),
                cand_white_wr,
                ref_white_wr,
                _status(white_accept, white_collapse),
            )
        )

    def _epoch(self, epoch: int) -> dict[str, Any]:
        data_black, data_white, info = self._collector.rollout(self.steps)
        info = add_prefix(info, "versus_play/")

        info["fps"] = info["versus_play/fps"]
        del info["versus_play/fps"]

        info.update(
            add_prefix(
                self.policy_black.learn(data_black.to_tensordict()),
                "policy_black/",
            )
        )
        del data_black

        info.update(
            add_prefix(
                self.policy_white.learn(data_white.to_tensordict()),
                "policy_white/",
            )
        )
        del data_white

        self._maybe_gate(epoch, info)

        if epoch % 50 == 0 and epoch != 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return info

    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % self.log_interval == 0:
            info.update(
                {
                    # 这三个保持和你原项目一致，便于继续看旧图
                    "eval/black_vs_white": eval_win_rate(
                        self.eval_env,
                        player_black=self.policy_black,
                        player_white=self.policy_white,
                    ),
                    "eval/black_vs_baseline": eval_win_rate(
                        self.eval_env,
                        player_black=self.policy_black,
                        player_white=self.baseline,
                    ),
                    "eval/baseline_vs_white": eval_win_rate(
                        self.eval_env,
                        player_black=self.baseline,
                        player_white=self.policy_white,
                    ),
                    # 新增：best 锚点相关评估
                    "eval/black_vs_best_white": eval_win_rate(
                        self.eval_env,
                        player_black=self.policy_black,
                        player_white=self.best_policy_white,
                        n=self.gate_eval_n,
                    ),
                    "eval/best_black_vs_white": eval_win_rate(
                        self.eval_env,
                        player_black=self.best_policy_black,
                        player_white=self.policy_white,
                        n=self.gate_eval_n,
                    ),
                    "eval/best_black_vs_best_white": self.best_pair_black_wr,
                    "eval/white_vs_best_black": 1.0
                    - eval_win_rate(
                        self.eval_env,
                        player_black=self.best_policy_black,
                        player_white=self.policy_white,
                        n=self.gate_eval_n,
                    ),
                }
            )

            print(
                "Black vs White:{:.2f}%\tBlack vs Baseline:{:.2f}%\tBaseline vs White:{:.2f}%".format(
                    info["eval/black_vs_white"] * 100,
                    info["eval/black_vs_baseline"] * 100,
                    info["eval/baseline_vs_white"] * 100,
                )
            )
            print(
                "Black vs BestWhite:{:.2f}%\tBestBlack vs White:{:.2f}%\tBestBlack vs BestWhite:{:.2f}%".format(
                    info["eval/black_vs_best_white"] * 100,
                    info["eval/best_black_vs_white"] * 100,
                    info["eval/best_black_vs_best_white"] * 100,
                )
            )

        return super()._log(info, epoch)


class IndependentRLSPRunner(SPRunner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._collector = SelfPlayCollector(
            self.env,
            self.policy,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )

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

    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % 5 == 0:
            info.update(
                {
                    "eval/player_vs_player": eval_win_rate(
                        self.eval_env,
                        player_black=self.policy,
                        player_white=self.policy,
                    ),
                    "eval/player_vs_baseline": eval_win_rate(
                        self.eval_env,
                        player_black=self.policy,
                        player_white=self.baseline,
                    ),
                    "eval/baseline_vs_player": eval_win_rate(
                        self.eval_env,
                        player_black=self.baseline,
                        player_white=self.policy,
                    ),
                }
            )

            print(
                "Player vs Player:{:.2f}%\tPlayer vs Baseline:{:.2f}%\tBaseline vs Player:{:.2f}%".format(
                    info["eval/player_vs_player"] * 100,
                    info["eval/player_vs_baseline"] * 100,
                    info["eval/baseline_vs_player"] * 100,
                )
            )

        return super()._log(info, epoch)