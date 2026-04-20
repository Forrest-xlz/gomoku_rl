import copy
import logging
import os
from typing import Any

import torch
from omegaconf import DictConfig

from .base import Runner, SPRunner
from gomoku_rl.collector import BlackPlayCollector, VersusPlayCollector, WhitePlayCollector
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.utils.misc import add_prefix, get_kwargs
from gomoku_rl.utils.policy import uniform_policy
from gomoku_rl.utils.psro import (
    ConvergedIndicator,
    PayoffType,
    Population,
    get_meta_solver,
    get_new_payoffs_sp,
    init_payoffs_sp,
    make_frozen_actor_policy,
)


class PSRORunner(Runner):
    """
    FIFO + uniform + staged sampling

    训练逻辑：
    1) 前期纯自博弈：
       - 黑策略 vs 白策略（当前策略）
       - 用 VersusPlayCollector 一次同时收集黑白两边数据
    2) 当模型池填充比例达到阈值后，开始池采样：
       - 黑当前策略 vs 白方模型池随机对手
       - 黑方模型池随机对手 vs 白当前策略
    3) 每隔 pool_update_interval 个 epoch，把当前黑/白策略各自压入各自 FIFO 池
    4) 若池大小超过 pool_size，则删除最老模型
    5) baseline 评估改为 pool 评估：
       - eval_baseline_pool.white_pool 里的模型作为白方 baseline，评估当前黑
       - eval_baseline_pool.black_pool 里的模型作为黑方 baseline，评估当前白
    6) 所有 eval 只在 eval_interval 执行一次
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.population_mode = str(cfg.get("population_mode", "fifo")).lower()
        self.opponent_sampling = str(cfg.get("opponent_sampling", "uniform")).lower()

        self.pool_size = int(cfg.get("pool_size", 8))
        self.pool_update_interval = int(cfg.get("pool_update_interval", 50))
        self.pool_sample_start_ratio = float(cfg.get("pool_sample_start_ratio", 0.5))

        # 新增：eval 间隔，默认 5
        self.eval_interval = int(cfg.get("eval_interval", 5))

        if self.population_mode != "fifo":
            raise ValueError(
                f"PSRORunner only supports population_mode='fifo', got {self.population_mode!r}"
            )
        if self.opponent_sampling != "uniform":
            raise ValueError(
                f"PSRORunner only supports opponent_sampling='uniform', got {self.opponent_sampling!r}"
            )
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.pool_update_interval <= 0:
            raise ValueError("pool_update_interval must be positive")
        if not (0.0 <= self.pool_sample_start_ratio <= 1.0):
            raise ValueError("pool_sample_start_ratio must be in [0, 1]")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")

        # ===== 初始化黑方模型池 =====
        if self.cfg.get("black_checkpoint", None):
            black_init_policy = make_frozen_actor_policy(self.policy_black, device=cfg.device)
        else:
            black_init_policy = uniform_policy

        self.black_population = Population(
            initial_policy=black_init_policy,
            dir=os.path.join(self.run_dir, "population_0"),
            device=cfg.device,
            interaction_type=cfg.get("population_interaction_type", "random"),
        )

        # ===== 初始化白方模型池 =====
        if self.cfg.get("white_checkpoint", None):
            white_init_policy = make_frozen_actor_policy(self.policy_white, device=cfg.device)
        else:
            white_init_policy = uniform_policy

        self.white_population = Population(
            initial_policy=white_init_policy,
            dir=os.path.join(self.run_dir, "population_1"),
            device=cfg.device,
            interaction_type=cfg.get("population_interaction_type", "random"),
        )

        # FIFO + uniform：整个池都 active，sample() 时不传 meta_policy 即均匀随机
        self.black_population.activate_all()
        self.white_population.activate_all()

        # ===== 3 套 collector，彼此使用独立 env，避免状态互相污染 =====
        self.collector_selfplay = VersusPlayCollector(
            copy.deepcopy(self.env),
            self.policy_black,
            self.policy_white,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )

        self.collector_black = BlackPlayCollector(
            copy.deepcopy(self.env),
            self.policy_black,
            self.white_population,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )

        self.collector_white = WhitePlayCollector(
            copy.deepcopy(self.env),
            self.black_population,
            self.policy_white,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )

    def _pool_fill_ratio(self) -> float:
        return min(len(self.black_population), len(self.white_population)) / float(self.pool_size)

    def _use_pool_sampling(self) -> bool:
        return self._pool_fill_ratio() >= self.pool_sample_start_ratio

    def _should_eval(self, epoch: int) -> bool:
        return (epoch + 1) % self.eval_interval == 0

    def _fifo_push_policy(
        self,
        population: Population,
        policy,
        side: str,
    ) -> dict[str, Any]:
        population.add(policy)

        removed_index = None
        removed_checkpoint_id = None

        if len(population) > self.pool_size:
            removed_entry = population.policy_sets[0]
            population.remove(0)
            removed_index = 0
            removed_checkpoint_id = (
                int(removed_entry) if isinstance(removed_entry, int) else str(removed_entry)
            )

        population.activate_all()

        return {
            "side": side,
            "pool_size": len(population),
            "removed_index": removed_index,
            "removed_checkpoint_id": removed_checkpoint_id,
        }

    def _refresh_pools(self) -> dict[str, Any]:
        black_info = self._fifo_push_policy(self.black_population, self.policy_black, "black")
        white_info = self._fifo_push_policy(self.white_population, self.policy_white, "white")

        logging.info(
            "fifo refresh | "
            f"black_pool={black_info['pool_size']} "
            f"black_removed_index={black_info['removed_index']} "
            f"black_removed_checkpoint={black_info['removed_checkpoint_id']} | "
            f"white_pool={white_info['pool_size']} "
            f"white_removed_index={white_info['removed_index']} "
            f"white_removed_checkpoint={white_info['removed_checkpoint_id']}"
        )

        self.collector_selfplay.reset()
        self.collector_black.reset()
        self.collector_white.reset()

        return {
            "fifo/black_pool_size": black_info["pool_size"],
            "fifo/white_pool_size": white_info["pool_size"],
            "psro-black-pool-size": black_info["pool_size"],
            "psro-white-pool-size": white_info["pool_size"],
            "fifo/black_removed_index": -1 if black_info["removed_index"] is None else black_info["removed_index"],
            "fifo/white_removed_index": -1 if white_info["removed_index"] is None else white_info["removed_index"],
            "fifo/updated": 1,
        }

    def _baseline_pool_metrics(self) -> dict[str, float]:
        """
        与 train_InRL.yaml 注释语义一致：
        - white_pool 里的模型作为白方 baseline，评估当前黑
        - black_pool 里的模型作为黑方 baseline，评估当前白
        """
        metrics: dict[str, float] = {}

        black_vs_baselines = []
        for i, opp_white in enumerate(self.eval_baseline_white_pool, start=1):
            wr = float(
                eval_win_rate(
                    self.eval_env,
                    player_black=self.policy_black,
                    player_white=opp_white,
                )
            )
            metrics[f"eval/black_vs_baseline{i}"] = wr
            black_vs_baselines.append(wr)

        if black_vs_baselines:
            metrics["eval/black_vs_baseline"] = sum(black_vs_baselines) / len(black_vs_baselines)
            metrics["eval/black_vs_baseline_pool_avg"] = metrics["eval/black_vs_baseline"]

        baseline_vs_white = []
        for i, opp_black in enumerate(self.eval_baseline_black_pool, start=1):
            wr = float(
                eval_win_rate(
                    self.eval_env,
                    player_black=opp_black,
                    player_white=self.policy_white,
                )
            )
            metrics[f"eval/baseline{i}_vs_white"] = wr
            baseline_vs_white.append(wr)

        if baseline_vs_white:
            metrics["eval/baseline_vs_white"] = sum(baseline_vs_white) / len(baseline_vs_white)
            metrics["eval/baseline_pool_vs_white_avg"] = metrics["eval/baseline_vs_white"]

        return metrics

    def _epoch(self, epoch: int) -> dict[str, Any]:
        info: dict[str, Any] = {}

        use_pool_sampling = self._use_pool_sampling()
        fill_ratio = self._pool_fill_ratio()

        info["fifo/use_pool_sampling"] = int(use_pool_sampling)
        info["fifo/pool_fill_ratio"] = fill_ratio
        info["fifo/black_pool_size"] = len(self.black_population)
        info["fifo/white_pool_size"] = len(self.white_population)
        info["psro-black-pool-size"] = len(self.black_population)
        info["psro-white-pool-size"] = len(self.white_population)
        info["fifo/updated"] = 0

        if not use_pool_sampling:
            data_black, data_white, info_selfplay = self.collector_selfplay.rollout(steps=self.steps)
            info.update(add_prefix(info_selfplay, "selfplay/"))

            info.update(add_prefix(self.policy_black.learn(data_black), "black/"))
            info.update(add_prefix(self.policy_white.learn(data_white), "white/"))

            del data_black
            del data_white

            info["fps"] = info["selfplay/fps"]
            del info["selfplay/fps"]

            info["sampled_white_opponent"] = -1
            info["sampled_black_opponent"] = -1
        else:
            self.white_population.sample()
            sampled_white_idx = int(self.white_population._idx)

            data_black, info_black = self.collector_black.rollout(steps=self.steps)
            info.update(add_prefix(info_black, "black_play/"))
            info["sampled_white_opponent"] = sampled_white_idx

            self.black_population.sample()
            sampled_black_idx = int(self.black_population._idx)

            data_white, info_white = self.collector_white.rollout(steps=self.steps)
            info.update(add_prefix(info_white, "white_play/"))
            info["sampled_black_opponent"] = sampled_black_idx

            info.update(add_prefix(self.policy_black.learn(data_black), "black/"))
            info.update(add_prefix(self.policy_white.learn(data_white), "white/"))

            del data_black
            del data_white

            info["fps"] = (info["black_play/fps"] + info["white_play/fps"]) / 2.0
            del info["black_play/fps"]
            del info["white_play/fps"]

        # ===== 所有 eval 只按间隔执行 =====
        if self._should_eval(epoch):
            info["eval/black_vs_white"] = float(
                eval_win_rate(
                    self.eval_env,
                    player_black=self.policy_black,
                    player_white=self.policy_white,
                )
            )
            info.update(self._baseline_pool_metrics())

        # ===== 固定间隔入池 =====
        if (epoch + 1) % self.pool_update_interval == 0:
            info.update(self._refresh_pools())

        return info

    def _post_run(self):
        # 删除 payoff 上传
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if self._should_eval(epoch):
            mode = "pool" if int(info["fifo/use_pool_sampling"]) == 1 else "selfplay"

            black_vs_baseline = info.get("eval/black_vs_baseline", float("nan"))
            baseline_vs_white = info.get("eval/baseline_vs_white", float("nan"))

            print(
                "mode:{} | fill:{:.2f} | Black vs White:{:.2f}% | "
                "Black vs BaselinePool:{:.2f}% | BaselinePool vs White:{:.2f}% | "
                "Black pool:{} White pool:{} | sampled (B<-W#{}, W<-B#{})".format(
                    mode,
                    float(info["fifo/pool_fill_ratio"]),
                    info.get("eval/black_vs_white", float("nan")) * 100,
                    black_vs_baseline * 100 if black_vs_baseline == black_vs_baseline else float("nan"),
                    baseline_vs_white * 100 if baseline_vs_white == baseline_vs_white else float("nan"),
                    int(info["psro-black-pool-size"]),
                    int(info["psro-white-pool-size"]),
                    int(info["sampled_white_opponent"]),
                    int(info["sampled_black_opponent"]),
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
        self.eval_interval = int(cfg.get("eval_interval", 5))
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")

        if (population_dir := cfg.get("population_dir", None)) and os.path.isdir(population_dir):
            _policy = []
            for p in os.listdir(population_dir):
                tmp = copy.deepcopy(self.policy)
                tmp.load_state_dict(torch.load(os.path.join(population_dir, p), map_location=self.cfg.device))
                tmp.eval()
                _policy.append(make_frozen_actor_policy(tmp, device=cfg.device))
        elif self.cfg.get("checkpoint", None):
            _policy = make_frozen_actor_policy(self.policy, device=cfg.device)
        else:
            _policy = uniform_policy

        self.population = Population(
            initial_policy=_policy,
            dir=os.path.join(self.run_dir, "population"),
            device=cfg.device,
        )

        # PSROSPRunner 的 meta_solver 仍然依赖 payoff 矩阵，但不再打印 / 上传 payoff
        self.payoffs = init_payoffs_sp(
            env=self.eval_env,
            population=self.population,
            type=PayoffType.black_vs_white,
        )

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

    def _should_eval(self, epoch: int) -> bool:
        return (epoch + 1) % self.eval_interval == 0

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

        if self._should_eval(epoch):
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
                        player_white=self.population
                        if len(self.eval_baseline_white_pool) == 0
                        else self.eval_baseline_white_pool[0],
                    ),
                    "eval/baseline_vs_player": eval_win_rate(
                        self.eval_env,
                        player_black=self.population
                        if len(self.eval_baseline_black_pool) == 0
                        else self.eval_baseline_black_pool[0],
                        player_white=self.policy,
                    ),
                }
            )

            alpha = 0.5
            weighted_wr = alpha * info["eval/player_vs_opponent"] + (1 - alpha) * (
                1 - info["eval/opponent_vs_player"]
            )
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
                self.meta_policy_black, self.meta_policy_white = self.meta_solver(payoffs=self.payoffs)
                logging.info(f"Meta Policy: {self.meta_policy_black}, {self.meta_policy_white}")

        return info

    def _post_run(self):
        # 删除 payoff 上传
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if self._should_eval(epoch):
            print(
                "Player vs Opponent:{:.2f}%\tOpponent vs Player:{:.2f}%\t"
                "Player vs Baseline:{:.2f}%\tBaseline vs Player:{:.2f}%".format(
                    info.get("eval/player_vs_opponent", float("nan")) * 100,
                    info.get("eval/opponent_vs_player", float("nan")) * 100,
                    info.get("eval/player_vs_baseline", float("nan")) * 100,
                    info.get("eval/baseline_vs_player", float("nan")) * 100,
                )
            )
        return super()._log(info, epoch)