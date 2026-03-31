from typing import Any

import torch
from omegaconf import DictConfig

from gomoku_rl.collector import SelfPlayCollector, VersusPlayCollector
from gomoku_rl.utils.eval import eval_match, eval_win_rate
from gomoku_rl.utils.misc import add_prefix

from .base import Runner, SPRunner


class IndependentRLRunner(Runner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._collector = VersusPlayCollector(
            self.env,
            self.policy_black,
            self.policy_white,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
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

        if epoch % 50 == 0 and epoch != 0:
            torch.cuda.empty_cache()
        return info

    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % self.eval_interval == 0:
            black_white_stats = eval_match(
                self.eval_env,
                player_black=self.policy_black,
                player_white=self.policy_white,
                n=self.eval_repeat,
            )
            black_baseline_stats = eval_match(
                self.eval_env,
                player_black=self.policy_black,
                player_white=self.baseline,
                n=self.eval_repeat,
            )
            baseline_white_stats = eval_match(
                self.eval_env,
                player_black=self.baseline,
                player_white=self.policy_white,
                n=self.eval_repeat,
            )

            black_playouts = self.black_pure_mcts_ladder.current()
            white_playouts = self.white_pure_mcts_ladder.current()
            black_pure_mcts = self._make_pure_mcts_player(num_simulations=black_playouts)
            white_pure_mcts = self._make_pure_mcts_player(num_simulations=white_playouts)

            black_pure_stats = eval_match(
                self.mcts_eval_env,
                player_black=self.policy_black,
                player_white=black_pure_mcts,
                n=self.pure_mcts_repeat,
            )
            pure_white_stats = eval_match(
                self.mcts_eval_env,
                player_black=white_pure_mcts,
                player_white=self.policy_white,
                n=self.pure_mcts_repeat,
            )

            info.update(
                {
                    "eval/black_vs_white": black_white_stats.black_win_rate,
                    "eval/black_vs_baseline": black_baseline_stats.black_win_rate,
                    "eval/white_vs_baseline": baseline_white_stats.white_win_rate,
                    "eval/black_vs_pure_mcts": black_pure_stats.black_win_rate,
                    "eval/white_vs_pure_mcts": pure_white_stats.white_win_rate,
                    "eval/black_pure_mcts_playouts": black_playouts,
                    "eval/white_pure_mcts_playouts": white_playouts,
                    "eval/black_vs_white_draw": black_white_stats.draw_rate,
                    "eval/black_vs_baseline_draw": black_baseline_stats.draw_rate,
                    "eval/white_vs_baseline_draw": baseline_white_stats.draw_rate,
                    "eval/black_vs_pure_mcts_draw": black_pure_stats.draw_rate,
                    "eval/white_vs_pure_mcts_draw": pure_white_stats.draw_rate,
                }
            )

            black_promoted = self.black_pure_mcts_ladder.update(
                black_pure_stats.black_win_rate
            )
            white_promoted = self.white_pure_mcts_ladder.update(
                pure_white_stats.white_win_rate
            )

            print(
                "Black vs White:{:.2f}%\t"
                "Black vs Baseline:{:.2f}%\t"
                "White vs Baseline:{:.2f}%\t"
                "Black vs PureMCTS({}):{:.2f}%\t"
                "White vs PureMCTS({}):{:.2f}%".format(
                    black_white_stats.black_win_rate * 100.0,
                    black_baseline_stats.black_win_rate * 100.0,
                    baseline_white_stats.white_win_rate * 100.0,
                    black_playouts,
                    black_pure_stats.black_win_rate * 100.0,
                    white_playouts,
                    pure_white_stats.white_win_rate * 100.0,
                )
            )

            if black_promoted:
                print(
                    "[eval] black pure-mcts ladder promoted -> "
                    f"{self.black_pure_mcts_ladder.current()} playouts"
                )
            if white_promoted:
                print(
                    "[eval] white pure-mcts ladder promoted -> "
                    f"{self.white_pure_mcts_ladder.current()} playouts"
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
        if epoch % 50 == 0 and epoch != 0:
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
