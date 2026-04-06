import numpy as np
import copy
import logging
import os
from typing import Any

import torch
import wandb
from omegaconf import DictConfig

from .base import Runner, SPRunner
from gomoku_rl.collector import BlackPlayCollector, VersusPlayCollector, WhitePlayCollector
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.utils.misc import add_prefix, get_kwargs
from gomoku_rl.utils.policy import uniform_policy
from gomoku_rl.utils.psro import (
    ConvergedIndicator,
    PSROPolicyWrapper,
    PayoffType,
    Population,
    black_archive_similarity,
    black_mean_win_rates,
    calculate_jpc,
    eval_black_win_rates_against_population,
    eval_white_win_rates_against_population,
    get_meta_solver,
    get_new_payoffs_sp,
    init_payoffs,
    init_payoffs_sp,
    mid_ratio,
    select_active_indices,
    white_archive_similarity,
    white_mean_win_rates,
)
from gomoku_rl.utils.visual import payoff_headmap


class PSRORunner(Runner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        ci_kwargs = get_kwargs(
            cfg,
            "max_size",
            "mean_threshold",
            "std_threshold",
            "min_iter_steps",
            "max_iter_steps",
        )
        self.converged_indicator = ConvergedIndicator(**ci_kwargs)
        self.learning_player_id = cfg.get("first_id", 0)
        self.meta_solver_name = cfg.get("meta_solver", "uniform").lower()
        self.meta_solver = get_meta_solver(self.meta_solver_name)

        self.mid_low = float(cfg.get("mid_low", 0.3))
        self.mid_high = float(cfg.get("mid_high", 0.7))
        self.active_min = int(cfg.get("active_min", 8))
        self.archive_max = int(cfg.get("archive_max", 32))
        self.protect_latest = int(cfg.get("protect_latest", 5))
        if not (0.0 <= self.mid_low <= self.mid_high <= 1.0):
            raise ValueError("mid_low and mid_high must satisfy 0 <= mid_low <= mid_high <= 1")
        if self.active_min <= 0:
            raise ValueError("active_min must be positive")
        if self.archive_max <= 0:
            raise ValueError("archive_max must be positive")
        if not (0 <= self.protect_latest < self.archive_max):
            raise ValueError(
                f"protect_latest must satisfy 0 <= protect_latest < {self.archive_max}"
            )
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))

        if self.cfg.get("black_checkpoint", None):
            _policy = copy.deepcopy(self.policy_black)
            _policy.eval()
        else:
            _policy = uniform_policy
        self.player_0 = PSROPolicyWrapper(
            self.policy_black,
            population=Population(
                initial_policy=_policy,
                dir=os.path.join(self.run_dir, "population_0"),
                device=cfg.device,
            ),
        )

        if self.cfg.get("white_checkpoint", None):
            _policy = copy.deepcopy(self.policy_white)
            _policy.eval()
        else:
            _policy = uniform_policy
        self.player_1 = PSROPolicyWrapper(
            self.policy_white,
            population=Population(
                initial_policy=_policy,
                dir=os.path.join(self.run_dir, "population_1"),
                device=cfg.device,
            ),
        )

        self.player_0.set_oracle_mode(self.learning_player_id == 0)
        self.player_1.set_oracle_mode(self.learning_player_id != 0)
        self.payoffs = init_payoffs(
            env=self.eval_env,
            population_0=self.player_0.population,
            population_1=self.player_1.population,
        )
        self.archive_metrics = {}
        self.collector = VersusPlayCollector(
            self.env,
            self.player_0,
            self.player_1,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )
        self._rebuild_active_pools(log_prefix="init")

    def _build_active_masks(self):
        black_wr_vs_white = eval_black_win_rates_against_population(
            env=self.eval_env,
            current_black=self.policy_black,
            white_population=self.player_1.population,
        )
        white_wr_vs_black = eval_white_win_rates_against_population(
            env=self.eval_env,
            black_population=self.player_0.population,
            current_white=self.policy_white,
        )

        white_active_info = select_active_indices(
            win_rates=black_wr_vs_white,
            mid_low=self.mid_low,
            mid_high=self.mid_high,
            active_min=self.active_min,
            rng=self.rng,
        )
        black_active_info = select_active_indices(
            win_rates=white_wr_vs_black,
            mid_low=self.mid_low,
            mid_high=self.mid_high,
            active_min=self.active_min,
            rng=self.rng,
        )
        return black_wr_vs_white, white_wr_vs_black, black_active_info, white_active_info

    def _rebuild_active_pools(self, log_prefix: str):
        black_wr_vs_white, white_wr_vs_black, black_active_info, white_active_info = self._build_active_masks()

        self.player_0.population.set_active_indices(black_active_info["active_indices"])
        self.player_1.population.set_active_indices(white_active_info["active_indices"])

        active_black_mask = self.player_0.population.get_active_mask()
        active_white_mask = self.player_1.population.get_active_mask()
        meta_policy_0, meta_policy_1 = self.meta_solver(
            payoffs=self.payoffs,
            active_row_mask=active_black_mask,
            active_col_mask=active_white_mask,
        )
        self.player_0.set_meta_policy(meta_policy=meta_policy_0)
        self.player_1.set_meta_policy(meta_policy=meta_policy_1)

        self.archive_metrics = {
            "archive/black_mid_ratio": mid_ratio(white_wr_vs_black, self.mid_low, self.mid_high),
            "archive/white_mid_ratio": mid_ratio(black_wr_vs_white, self.mid_low, self.mid_high),
            "archive/black_similarity": black_archive_similarity(self.payoffs),
            "archive/white_similarity": white_archive_similarity(self.payoffs),
        }

        logging.info(
            f"{log_prefix} active pool | "
            f"black_vs_white_wr={np.array2string(black_wr_vs_white, precision=3)} | "
            f"white_active={white_active_info['active_indices'].tolist()} | "
            f"white_mid={white_active_info['mid_indices'].tolist()} | "
            f"white_easy={white_active_info['easy_indices'].tolist()} | "
            f"white_hard={white_active_info['hard_indices'].tolist()}"
        )
        logging.info(
            f"{log_prefix} active pool | "
            f"white_vs_black_wr={np.array2string(white_wr_vs_black, precision=3)} | "
            f"black_active={black_active_info['active_indices'].tolist()} | "
            f"black_mid={black_active_info['mid_indices'].tolist()} | "
            f"black_easy={black_active_info['easy_indices'].tolist()} | "
            f"black_hard={black_active_info['hard_indices'].tolist()}"
        )
        logging.info(
            f"{log_prefix} archive metrics | "
            f"black_mid_ratio={self.archive_metrics['archive/black_mid_ratio']:.3f} | "
            f"white_mid_ratio={self.archive_metrics['archive/white_mid_ratio']:.3f} | "
            f"black_similarity={self.archive_metrics['archive/black_similarity']:.3f} | "
            f"white_similarity={self.archive_metrics['archive/white_similarity']:.3f}"
        )
        logging.info(f"Meta Policy: Black {meta_policy_0}, White {meta_policy_1}")

        if self.payoffs.shape[0] == self.payoffs.shape[1] and active_black_mask.sum() > 1 and active_white_mask.sum() > 1 and active_black_mask.sum() == active_white_mask.sum():
            logging.info(
                f"Active JPC:{calculate_jpc((self.payoffs + 1) / 2, active_black_mask, active_white_mask)}"
            )

    def _archive_insert_and_evict(self, wrapper: PSROPolicyWrapper, mean_win_rates: np.ndarray, side: str) -> dict[str, Any]:
        population = wrapper.population
        info = {
            "side": side,
            "removed_archive_index": None,
            "removed_checkpoint_id": None,
            "archive_mean_win_rates": mean_win_rates.copy(),
        }
        if len(population) >= self.archive_max:
            protected_start = max(0, len(population) - self.protect_latest)
            candidate_indices = np.arange(protected_start, dtype=int)
            if len(candidate_indices) == 0:
                raise RuntimeError("No removable archive candidate found. Decrease protect_latest.")
            local_remove = int(np.argmin(mean_win_rates[candidate_indices]))
            remove_index = int(candidate_indices[local_remove])
            removed_entry = population.policy_sets[remove_index]
            population.remove(remove_index)
            info["removed_archive_index"] = remove_index
            info["removed_checkpoint_id"] = int(removed_entry) if isinstance(removed_entry, int) else str(removed_entry)
        wrapper.add_current_policy()
        info["archive_size"] = len(population)
        return info

    def _refresh_archives_and_payoffs(self):
        black_info = self._archive_insert_and_evict(
            wrapper=self.player_0,
            mean_win_rates=black_mean_win_rates(self.payoffs),
            side="black",
        )
        white_info = self._archive_insert_and_evict(
            wrapper=self.player_1,
            mean_win_rates=white_mean_win_rates(self.payoffs),
            side="white",
        )
        logging.info(
            "archive update | "
            f"black_removed_index={black_info['removed_archive_index']} | "
            f"black_removed_checkpoint={black_info['removed_checkpoint_id']} | "
            f"white_removed_index={white_info['removed_archive_index']} | "
            f"white_removed_checkpoint={white_info['removed_checkpoint_id']}"
        )

        self.payoffs = init_payoffs(
            env=self.eval_env,
            population_0=self.player_0.population,
            population_1=self.player_1.population,
        )
        logging.info(
            "archive mean win rate after refresh | "
            f"black={np.array2string(black_mean_win_rates(self.payoffs), precision=3)} | "
            f"white={np.array2string(white_mean_win_rates(self.payoffs), precision=3)}"
        )
        self._rebuild_active_pools(log_prefix="refresh")

    def _epoch(self, epoch: int) -> dict[str, Any]:
        if self.learning_player_id == 0:
            self.player_1.sample()
        else:
            self.player_0.sample()

        data_0, data_1, info = self.collector.rollout(steps=self.steps)
        info = add_prefix(info, "versus_play/")
        info["fps"] = info["versus_play/fps"]
        del info["versus_play/fps"]
        info.update(
            {
                "pure_strategy_black": self.player_0.population._idx if self.learning_player_id == 1 else -1,
                "pure_strategy_white": self.player_1.population._idx if self.learning_player_id == 0 else -1,
            }
        )

        if self.learning_player_id == 0:
            info.update(add_prefix(self.policy_black.learn(data_0), "black/"))
            del data_0
        else:
            info.update(add_prefix(self.policy_white.learn(data_1), "white/"))
            del data_1

        info.update(
            {
                "eval/black_vs_white": eval_win_rate(
                    self.eval_env,
                    player_black=self.player_0,
                    player_white=self.player_1,
                ),
                "eval/black_vs_baseline": eval_win_rate(
                    self.eval_env,
                    player_black=self.policy_black,
                    player_white=self.baseline,
                ),
                "eval/white_vs_baseline": 1
                - eval_win_rate(
                    self.eval_env,
                    player_black=self.baseline,
                    player_white=self.policy_white,
                ),
            }
        )
        self.converged_indicator.update(
            info["eval/black_vs_white"]
            if self.learning_player_id == 0
            else (1 - info["eval/black_vs_white"])
        )

        if self.converged_indicator.converged():
            self.collector.reset()
            self.converged_indicator.reset()
            if self.learning_player_id == 0:
                self.player_0.set_oracle_mode(False)
                self.player_1.set_oracle_mode(True)
            else:
                self.player_1.set_oracle_mode(False)
                self.player_0.set_oracle_mode(True)
            self.learning_player_id = (self.learning_player_id + 1) % 2
            logging.info(f"learning_player_id:{self.learning_player_id}")

            if self.learning_player_id == self.cfg.get("first_id", 0):
                self._refresh_archives_and_payoffs()

        info.update(self.archive_metrics)
        return info

    def _post_run(self):
        wandb.log(
            {
                "payoff": payoff_headmap(
                    (self.payoffs[-5:, -5:] + 1) / 2 * 100,
                )
            }
        )

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % 5 == 0:
            print(
                "Black vs White:{:.2f}%	Black vs baseline:{:.2f}%	White vs baseline:{:.2f}%".format(
                    info["eval/black_vs_white"] * 100,
                    info["eval/black_vs_baseline"] * 100,
                    info["eval/white_vs_baseline"] * 100,
                )
            )
        return super()._log(info, epoch)



class PSROSPRunner(SPRunner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        ci_kwargs = get_kwargs(
            cfg,
            "mean_threshold",
            "std_threshold",
            "min_iter_steps",
            "max_iter_steps",
        )
        self.converged_indicator = ConvergedIndicator(**ci_kwargs)

        if (population_dir := cfg.get("population_dir", None)) and os.path.isdir(population_dir):
            _policy = []
            for p in os.listdir(population_dir):
                tmp = copy.deepcopy(self.policy)
                tmp.load_state_dict(torch.load(os.path.join(population_dir, p), map_location=self.cfg.device))
                tmp.eval()
                _policy.append(tmp)
        elif self.cfg.get("checkpoint", None):
            _policy = copy.deepcopy(self.policy)
            _policy.eval()
        else:
            _policy = uniform_policy

        self.population = Population(
            initial_policy=_policy,
            dir=os.path.join(self.run_dir, "population"),
            device=cfg.device,
        )
        self.payoffs = init_payoffs_sp(
            env=self.eval_env,
            population=self.population,
            type=PayoffType.black_vs_white,
        )
        print(repr(self.payoffs))
        self.meta_solver = get_meta_solver(cfg.get("meta_solver", "uniform"))
        if len(self.population) > 1:
            self.meta_policy_black, self.meta_policy_white = self.meta_solver(payoffs=self.payoffs)
            logging.info(f"Meta Policy: {self.meta_policy_black}, {self.meta_policy_white}")
        else:
            self.meta_policy_black, self.meta_policy_white = None, None

        self.collector_black = BlackPlayCollector(
            self.env,
            self.policy,
            self.population,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )
        self.collector_white = WhitePlayCollector(
            copy.deepcopy(self.env),
            self.population,
            self.policy,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )

    def _epoch(self, epoch: int) -> dict[str, Any]:
        info = {}
        self.population.sample(self.meta_policy_white)
        info.update({"pure_strategy_white": self.population._idx})
        data1, info1 = self.collector_black.rollout(steps=self.steps)
        info.update(add_prefix(info1, "black_play/"))

        self.population.sample(self.meta_policy_black)
        info.update({"pure_strategy_black": self.population._idx})
        data2, info2 = self.collector_white.rollout(steps=self.steps)
        info.update(add_prefix(info2, "white_play/"))

        data = torch.cat([data1, data2], dim=-1)
        info.update(add_prefix(self.policy.learn(data.to_tensordict()), "policy/"))
        del data
        info["fps"] = (info["black_play/fps"] + info["white_play/fps"]) / 2
        del info["black_play/fps"]
        del info["white_play/fps"]

        info.update(
            {
                "eval/player_vs_opponent": eval_win_rate(
                    self.eval_env,
                    player_black=self.policy,
                    player_white=self.population,
                ),
                "eval/opponent_vs_player": eval_win_rate(
                    self.eval_env,
                    player_black=self.population,
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
        alpha = 0.5
        weighted_wr = alpha * info["eval/player_vs_opponent"] + (1 - alpha) * (1 - info["eval/opponent_vs_player"])
        info.update({"weighted_win_rate": weighted_wr})
        self.converged_indicator.update(weighted_wr)

        if self.converged_indicator.converged():
            self.converged_indicator.reset()
            _policy = copy.deepcopy(self.policy)
            _policy.eval()
            self.population.add(_policy)
            self.payoffs = get_new_payoffs_sp(
                env=self.eval_env,
                population=self.population,
                old_payoffs=self.payoffs,
                type=PayoffType.black_vs_white,
            )
            print(repr(self.payoffs))
            self.meta_policy_black, self.meta_policy_white = self.meta_solver(payoffs=self.payoffs)
            logging.info(f"Meta Policy: {self.meta_policy_black}, {self.meta_policy_white}")

        return info

    def _post_run(self):
        wandb.log(
            {
                "payoff": payoff_headmap(
                    (self.payoffs[-5:, -5:] + 1) / 2 * 100,
                )
            }
        )

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % 5 == 0:
            print(
                "Player vs Opponent:{:.2f}%\tOpponent vs Player:{:.2f}%\tPlayer vs Baseline:{:.2f}%\tBaseline vs Player:{:.2f}%".format(
                    info["eval/player_vs_opponent"] * 100,
                    info["eval/opponent_vs_player"] * 100,
                    info["eval/player_vs_baseline"] * 100,
                    info["eval/baseline_vs_player"] * 100,
                )
            )
        return super()._log(info, epoch)
