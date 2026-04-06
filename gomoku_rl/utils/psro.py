import contextlib
import copy
import enum
import functools
import logging
import os
from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from gomoku_rl.policy import Policy
from scipy.optimize import linprog
from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule, set_interaction_type

from .eval import eval_win_rate
from .policy import _policy_t, uniform_policy

DeviceLike = Union[torch.device, str, int, None]


class PayoffType(enum.Enum):
    black_vs_white = enum.auto()
    both = enum.auto()


_meta_solver_t = Callable[..., tuple[np.ndarray, np.ndarray]]


class ConvergedIndicator:
    def __init__(
        self,
        max_size: int = 15,
        mean_threshold: float = 0.99,
        std_threshold: float = 0.005,
        min_iter_steps: int = 40,
        max_iter_steps: int = 300,
    ) -> None:
        self.win_rates = []
        self.max_size = max_size
        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold
        self.min_iter_steps = min_iter_steps
        self.max_iter_steps = max_iter_steps
        self._step_cnt = 0

    def update(self, value: float):
        self.win_rates.append(value)
        self._step_cnt += 1
        if len(self.win_rates) > self.max_size:
            self.win_rates.pop(0)

    def reset(self):
        self.win_rates = []
        self._step_cnt = 0

    def converged(self) -> bool:
        if len(self.win_rates) < self.max_size:
            return False
        if self._step_cnt < self.min_iter_steps:
            return False
        if self._step_cnt > self.max_iter_steps:
            return True
        mean = np.mean(self.win_rates)
        std = np.std(self.win_rates)
        return mean >= self.mean_threshold and std <= self.std_threshold


class Population:
    META_FILE_NAME = "population_meta.pt"

    def __init__(
        self,
        dir: str,
        initial_policy: _policy_t | list[_policy_t] = uniform_policy,
        device: DeviceLike = "cuda",
        module_prototype: Policy | TensorDictModule | None = None,
    ):
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)

        self._meta_path = os.path.join(self.dir, self.META_FILE_NAME)
        self._module_cnt = 0
        self._module = copy.deepcopy(module_prototype) if module_prototype is not None else None
        if self._module is not None:
            self._module.eval()
        self._module_backup = None
        self._idx = -1
        self.device = device
        self.restored_from_meta = False

        # this should be deterministic, as PSRO requires pure strategies.
        # But it seems it easily overfits
        self._interaction_type = InteractionType.MODE
        # self._interaction_type = InteractionType.RANDOM

        self.policy_sets: list[_policy_t | int] = []
        self.active_mask: list[bool] = []

        if not self._try_restore_from_metadata():
            if isinstance(initial_policy, (TensorDictModule, Policy)):
                self.add(initial_policy)
            elif isinstance(initial_policy, list):
                for _ip in initial_policy:
                    assert isinstance(_ip, (TensorDictModule, Policy))
                    self.add(_ip)
            else:
                self.policy_sets.append(initial_policy)
                self.active_mask.append(True)
                self._save_metadata()

        self._func = None
        self.sample()

    @classmethod
    def has_metadata(cls, dir: str) -> bool:
        return Path(dir, cls.META_FILE_NAME).is_file()

    def __len__(self) -> int:
        return len(self.policy_sets)

    def active_count(self) -> int:
        return int(np.sum(self.active_mask))

    def get_active_mask(self) -> np.ndarray:
        return np.asarray(self.active_mask, dtype=bool)

    def get_active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.get_active_mask())

    def is_active(self, index: int) -> bool:
        return bool(self.active_mask[index])

    def deactivate(self, index: int):
        if self.active_mask[index]:
            self.active_mask[index] = False
            self._save_metadata()

    def add(self, policy: Policy):
        if self._module is None:
            self._module = copy.deepcopy(policy)
            self._module.eval()
        torch.save(
            policy.state_dict(),
            os.path.join(self.dir, f"{self._module_cnt}.pt"),
        )
        self.policy_sets.append(self._module_cnt)
        self.active_mask.append(True)
        self._module_cnt += 1
        self._save_metadata()

    def _entry_to_metadata(self, entry: _policy_t | int) -> dict:
        if isinstance(entry, int):
            return {"kind": "checkpoint", "value": int(entry)}
        if entry is uniform_policy:
            return {"kind": "callable", "name": "uniform_policy"}
        raise ValueError(
            "Only checkpoint-backed policies and uniform_policy can be persisted in Population metadata."
        )

    def _entry_from_metadata(self, meta: dict) -> _policy_t | int:
        kind = meta["kind"]
        if kind == "checkpoint":
            if self._module is None:
                raise RuntimeError(
                    f"Population at {self.dir} contains checkpoint-backed policies, "
                    "but no module_prototype was provided for restoration."
                )
            return int(meta["value"])
        if kind == "callable":
            name = meta["name"]
            if name == "uniform_policy":
                return uniform_policy
            raise ValueError(f"Unknown persisted callable policy: {name}")
        raise ValueError(f"Unknown population entry kind: {kind}")

    def _save_metadata(self):
        payload = {
            "version": 1,
            "module_cnt": int(self._module_cnt),
            "entries": [self._entry_to_metadata(entry) for entry in self.policy_sets],
            "active_mask": [bool(v) for v in self.active_mask],
        }
        torch.save(payload, self._meta_path)

    def _try_restore_from_metadata(self) -> bool:
        if not os.path.isfile(self._meta_path):
            return False

        payload = torch.load(self._meta_path, map_location="cpu")
        entries = payload.get("entries", None)
        active_mask = payload.get("active_mask", None)
        if entries is None or active_mask is None:
            raise RuntimeError(f"Invalid population metadata file: {self._meta_path}")
        if len(entries) != len(active_mask):
            raise RuntimeError(
                f"Population metadata length mismatch in {self._meta_path}: "
                f"entries={len(entries)}, active_mask={len(active_mask)}"
            )

        self.policy_sets = [self._entry_from_metadata(entry) for entry in entries]
        self.active_mask = [bool(v) for v in active_mask]
        self._module_cnt = int(payload.get("module_cnt", 0))
        self.restored_from_meta = True
        return True

    def _set_policy(self, index: int):
        if self._idx == index:
            return
        self._idx = index
        if not isinstance(self.policy_sets[index], int):
            self._func = self.policy_sets[index]
        else:
            assert self._module is not None
            self._module.load_state_dict(
                torch.load(
                    os.path.join(self.dir, f"{self.policy_sets[index]}.pt"),
                    map_location=self.device,
                )
            )
            self._module.eval()
            self._func = self._module

    def sample(self, meta_policy: np.ndarray | None = None):
        active_indices = self.get_active_indices()
        if len(active_indices) == 0:
            raise RuntimeError("Population has no active policy to sample from.")

        if meta_policy is None:
            chosen = np.random.choice(active_indices)
            self._set_policy(int(chosen))
            return

        meta_policy = np.asarray(meta_policy, dtype=np.float64)
        assert len(meta_policy) == len(self.policy_sets)

        probs = np.zeros_like(meta_policy, dtype=np.float64)
        probs[active_indices] = meta_policy[active_indices]
        total = probs.sum()
        if total <= 0:
            probs[active_indices] = 1.0 / len(active_indices)
        else:
            probs /= total
        chosen = np.random.choice(np.arange(len(self.policy_sets)), p=probs)
        self._set_policy(int(chosen))

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        tensordict = tensordict.to(self.device)
        with set_interaction_type(type=self._interaction_type):
            return self._func(tensordict)

    @contextlib.contextmanager
    def fixed_behavioural_strategy(self, index: int):
        _idx = self._idx
        self._set_policy(index)
        _interaction_type = self._interaction_type
        self._interaction_type = InteractionType.RANDOM
        yield
        self._set_policy(_idx)
        self._interaction_type = _interaction_type

    def make_behavioural_strategy(self, index: int) -> _policy_t:
        """
        **share _module_backup!!!**
        ```
        s1=population.make_behavioural_strategy(0)
        s2=population.make_behavioural_strategy(1)
        then s1 and s2 are the same strategy!!!
        ```
        """
        if not isinstance(self.policy_sets[index], int):
            return self.policy_sets[index]
        if self._module_backup is None:
            self._module_backup = copy.deepcopy(self._module)
        self._module_backup.load_state_dict(
            torch.load(
                os.path.join(self.dir, f"{self.policy_sets[index]}.pt"),
                map_location=self.device,
            )
        )
        self._module_backup.eval()

        def _strategy(tensordict: TensorDict) -> TensorDict:
            tensordict = tensordict.to(self.device)
            with set_interaction_type(type=InteractionType.RANDOM):
                return self._module_backup(tensordict)

        return _strategy


class PSROPolicyWrapper:
    def __init__(self, policy: Policy, population: Population):
        self.policy = policy
        self.population = population
        self.meta_policy = None
        self._oracle_mode = True

    def set_meta_policy(self, meta_policy: np.ndarray):
        assert len(meta_policy) == len(self.population)
        self.meta_policy = meta_policy

    def set_oracle_mode(self, value: bool = True):
        self._oracle_mode = value

    def sample(self):
        assert not self._oracle_mode
        self.population.sample(meta_policy=self.meta_policy)

    def add_current_policy(self):
        actor = copy.deepcopy(self.policy)
        actor.eval()
        self.population.add(actor)

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if self._oracle_mode:
            return self.policy(tensordict)
        else:
            return self.population(tensordict)

    def eval(self):
        self.policy.eval()

    def train(self):
        self.policy.train()


def init_payoffs(
    env,
    population_0: Population,
    population_1: Population,
):
    assert len(population_0) == len(population_1)
    n = len(population_0)
    payoffs = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            with population_0.fixed_behavioural_strategy(index=i):
                with population_1.fixed_behavioural_strategy(index=j):
                    wr = eval_win_rate(
                        env=env,
                        player_black=population_0,
                        player_white=population_1,
                    )
            payoffs[i, j] = 2 * wr - 1
    return payoffs


def get_new_payoffs(
    env,
    population_0: Population,
    population_1: Population,
    old_payoffs: np.ndarray | None,
):
    assert len(population_0) == len(population_1)
    n = len(population_0)
    if old_payoffs is not None:
        assert (
            len(old_payoffs.shape) == 2
            and old_payoffs.shape[0] == old_payoffs.shape[1]
            and old_payoffs.shape[0] + 1 == n
        )
    new_payoffs = np.zeros(shape=(n, n))
    if old_payoffs is not None:
        new_payoffs[:-1, :-1] = old_payoffs
    for i in range(n):
        with population_0.fixed_behavioural_strategy(index=-1):
            with population_1.fixed_behavioural_strategy(index=i):
                wr_1 = eval_win_rate(
                    env=env,
                    player_black=population_0,
                    player_white=population_1,
                )
                wr_2 = eval_win_rate(
                    env=env,
                    player_black=population_1,
                    player_white=population_0,
                )
                new_payoffs[-1, i] = 2 * wr_1 - 1
                new_payoffs[i, -1] = 2 * wr_2 - 1
    return new_payoffs


def init_payoffs_sp(env, population: Population, type: PayoffType):
    if type == PayoffType.black_vs_white:
        return _init_payoffs_sp_bw(env, population)
    elif type == PayoffType.both:
        return _init_payoffs_sp_both(env, population)
    else:
        raise NotImplementedError()


def _init_payoffs_sp_both(env, population: Population):
    n = len(population)
    new_payoffs = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(i + 1, n):
            with population.fixed_behavioural_strategy(index=i):
                player_j = population.make_behavioural_strategy(j)
                wr_1 = eval_win_rate(
                    env=env,
                    player_black=population,
                    player_white=player_j,
                )
                wr_2 = eval_win_rate(
                    env=env,
                    player_black=player_j,
                    player_white=population,
                )
                new_payoffs[i, j] = wr_1 - wr_2
                new_payoffs[j, i] = wr_2 - wr_1
    return new_payoffs


def _init_payoffs_sp_bw(env, population: Population):
    n = len(population)
    new_payoffs = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(i + 1, n):
            with population.fixed_behavioural_strategy(index=i):
                player_j = population.make_behavioural_strategy(j)
                wr_1 = eval_win_rate(
                    env=env,
                    player_black=population,
                    player_white=player_j,
                )
                wr_2 = eval_win_rate(
                    env=env,
                    player_black=player_j,
                    player_white=population,
                )
                new_payoffs[i, j] = 2 * wr_1 - 1
                new_payoffs[j, i] = 2 * wr_2 - 1
    for i in range(n):
        with population.fixed_behavioural_strategy(index=i):
            wr = eval_win_rate(
                env=env,
                player_black=population,
                player_white=population,
            )
            new_payoffs[i, i] = 2 * wr - 1
    return new_payoffs


def get_new_payoffs_sp(
    env,
    population: Population,
    old_payoffs: np.ndarray | None,
    type: PayoffType = PayoffType.both,
):
    n = len(population)
    if old_payoffs is not None:
        assert (
            len(old_payoffs.shape) == 2
            and old_payoffs.shape[0] == old_payoffs.shape[1]
            and old_payoffs.shape[0] + 1 == n
        )
    new_payoffs = np.zeros(shape=(n, n))
    if old_payoffs is not None:
        new_payoffs[:-1, :-1] = old_payoffs
    if type == PayoffType.both:
        for i in range(n - 1):
            with population.fixed_behavioural_strategy(index=n - 1):
                player_i = population.make_behavioural_strategy(index=i)
                wr_1 = eval_win_rate(
                    env=env,
                    player_black=player_i,
                    player_white=population,
                )
                wr_2 = 1 - eval_win_rate(
                    env=env,
                    player_black=population,
                    player_white=player_i,
                )
                new_payoffs[i, -1] = wr_1 + wr_2 - 1
                new_payoffs[-1, i] = -new_payoffs[i, -1]
    elif type == PayoffType.black_vs_white:
        for i in range(n - 1):
            with population.fixed_behavioural_strategy(index=-1):
                player_i = population.make_behavioural_strategy(index=i)
                wr_1 = eval_win_rate(
                    env=env,
                    player_black=population,
                    player_white=player_i,
                )
                wr_2 = eval_win_rate(
                    env=env,
                    player_black=player_i,
                    player_white=population,
                )
                new_payoffs[-1, i] = 2 * wr_1 - 1
                new_payoffs[i, -1] = 2 * wr_2 - 1
        with population.fixed_behavioural_strategy(index=-1):
            wr = eval_win_rate(
                env=env,
                player_black=population,
                player_white=population,
            )
            new_payoffs[-1, -1] = 2 * wr - 1
    return new_payoffs


def print_payoffs(payoffs: np.ndarray):
    print(
        "payoffs:\n"
        + "\n".join(["\t".join([f"{item:+.3f}" for item in line]) for line in payoffs])
    )


def _normalize_mask(mask: np.ndarray | list[bool] | None, length: int) -> np.ndarray:
    if mask is None:
        return np.ones(length, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    assert mask.shape == (length,)
    if not mask.any():
        raise ValueError("Active mask cannot be all False.")
    return mask


def _expand_meta_policy(sub_policy: np.ndarray, active_mask: np.ndarray, length: int) -> np.ndarray:
    full = np.zeros(length, dtype=np.float64)
    full[np.flatnonzero(active_mask)] = sub_policy
    return full


def _solve_nash_submatrix(payoffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_row, num_col = payoffs.shape
    if num_row == 1:
        meta_strategy_row = np.array([1.0], dtype=np.float64)
    else:
        payoffs_row = payoffs - np.min(payoffs)
        result_row = linprog(
            c=np.ones(num_row),
            A_ub=-payoffs_row.T,
            b_ub=-np.ones(num_col),
            bounds=[(0, None)] * num_row,
        )
        meta_strategy_row = result_row.x / np.sum(result_row.x)

    if num_col == 1:
        meta_strategy_col = np.array([1.0], dtype=np.float64)
    else:
        payoffs_col = -payoffs - np.min(-payoffs)
        result_col = linprog(
            c=np.ones(num_col),
            A_ub=-payoffs_col,
            b_ub=-np.ones(num_row),
            bounds=[(0, None)] * num_col,
        )
        meta_strategy_col = result_col.x / np.sum(result_col.x)
    return meta_strategy_row, meta_strategy_col


def solve_nash(
    payoffs: np.ndarray,
    active_row_mask: np.ndarray | list[bool] | None = None,
    active_col_mask: np.ndarray | list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    row_mask = _normalize_mask(active_row_mask, payoffs.shape[0])
    col_mask = _normalize_mask(active_col_mask, payoffs.shape[1])
    sub_payoffs = payoffs[np.ix_(row_mask, col_mask)]
    sub_row, sub_col = _solve_nash_submatrix(sub_payoffs)
    return (
        _expand_meta_policy(sub_row, row_mask, payoffs.shape[0]),
        _expand_meta_policy(sub_col, col_mask, payoffs.shape[1]),
    )


def solve_uniform(
    payoffs: np.ndarray,
    active_row_mask: np.ndarray | list[bool] | None = None,
    active_col_mask: np.ndarray | list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    row_mask = _normalize_mask(active_row_mask, payoffs.shape[0])
    col_mask = _normalize_mask(active_col_mask, payoffs.shape[1])
    row_policy = np.zeros(payoffs.shape[0], dtype=np.float64)
    col_policy = np.zeros(payoffs.shape[1], dtype=np.float64)
    row_policy[row_mask] = 1.0 / np.sum(row_mask)
    col_policy[col_mask] = 1.0 / np.sum(col_mask)
    return row_policy, col_policy


def _solve_last_n_active(length: int, n: int, active_mask: np.ndarray) -> np.ndarray:
    active_indices = np.flatnonzero(active_mask)
    policy = np.zeros(length, dtype=np.float64)
    if len(active_indices) <= n:
        policy[active_indices] = 1.0 / len(active_indices)
    else:
        keep = active_indices[-n:]
        policy[keep] = 1.0 / n
    return policy


def solve_last_n(
    payoffs: np.ndarray,
    n: int,
    active_row_mask: np.ndarray | list[bool] | None = None,
    active_col_mask: np.ndarray | list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    row_mask = _normalize_mask(active_row_mask, payoffs.shape[0])
    col_mask = _normalize_mask(active_col_mask, payoffs.shape[1])
    return (
        _solve_last_n_active(payoffs.shape[0], n, row_mask),
        _solve_last_n_active(payoffs.shape[1], n, col_mask),
    )


def get_meta_solver(name: str) -> _meta_solver_t:
    tmp = {
        "uniform": solve_uniform,
        "uniform_threshold": solve_uniform,
        "nash": solve_nash,
        "iterated_best_response": functools.partial(solve_last_n, n=1),
    }
    name = name.lower()
    if name in tmp:
        return tmp[name]
    elif name.startswith("last_"):
        n = int(name.split("_")[-1])
        return functools.partial(solve_last_n, n=n)
    else:
        raise NotImplementedError()


def _black_mean_win_rates(payoffs: np.ndarray, active_white_mask: np.ndarray) -> np.ndarray:
    black_win_rates = (payoffs + 1.0) / 2.0
    return black_win_rates[:, active_white_mask].mean(axis=1)


def _white_mean_win_rates(payoffs: np.ndarray, active_black_mask: np.ndarray) -> np.ndarray:
    white_win_rates = (1.0 - payoffs[active_black_mask, :]) / 2.0
    return white_win_rates.mean(axis=0)


def _threshold_violation(mean_win_rate: float, lower: float, upper: float) -> float:
    if mean_win_rate < lower:
        return lower - mean_win_rate
    if mean_win_rate > upper:
        return mean_win_rate - upper
    return 0.0


def prune_populations_once_by_threshold_and_capacity(
    payoffs: np.ndarray,
    population_black: Population,
    population_white: Population,
    lower_threshold: float,
    upper_threshold: float,
    min_pool_size_black: int,
    max_pool_size_black: int,
    min_pool_size_white: int,
    max_pool_size_white: int,
) -> dict:
    if lower_threshold > upper_threshold:
        raise ValueError("lower_threshold cannot be greater than upper_threshold")
    if min_pool_size_black > max_pool_size_black:
        raise ValueError("min_pool_size_black cannot be greater than max_pool_size_black")
    if min_pool_size_white > max_pool_size_white:
        raise ValueError("min_pool_size_white cannot be greater than max_pool_size_white")

    active_black_mask = population_black.get_active_mask().copy()
    active_white_mask = population_white.get_active_mask().copy()

    black_mean = _black_mean_win_rates(payoffs, active_white_mask)
    white_mean = _white_mean_win_rates(payoffs, active_black_mask)

    deactivate_black_threshold: list[int] = []
    deactivate_white_threshold: list[int] = []

    active_black_indices = np.flatnonzero(active_black_mask)
    active_white_indices = np.flatnonzero(active_white_mask)

    if len(active_black_indices) > min_pool_size_black:
        black_candidates = [
            (i, _threshold_violation(float(black_mean[i]), lower_threshold, upper_threshold))
            for i in active_black_indices
            if black_mean[i] < lower_threshold or black_mean[i] > upper_threshold
        ]
        black_candidates.sort(key=lambda x: x[1], reverse=True)
        max_remove = len(active_black_indices) - min_pool_size_black
        deactivate_black_threshold = [i for i, _ in black_candidates[:max_remove]]

    if len(active_white_indices) > min_pool_size_white:
        white_candidates = [
            (j, _threshold_violation(float(white_mean[j]), lower_threshold, upper_threshold))
            for j in active_white_indices
            if white_mean[j] < lower_threshold or white_mean[j] > upper_threshold
        ]
        white_candidates.sort(key=lambda x: x[1], reverse=True)
        max_remove = len(active_white_indices) - min_pool_size_white
        deactivate_white_threshold = [j for j, _ in white_candidates[:max_remove]]

    for i in deactivate_black_threshold:
        population_black.deactivate(i)
    for j in deactivate_white_threshold:
        population_white.deactivate(j)

    remain_black_mask = population_black.get_active_mask().copy()
    remain_white_mask = population_white.get_active_mask().copy()

    deactivate_black_capacity: list[int] = []
    deactivate_white_capacity: list[int] = []

    remain_black_indices = np.flatnonzero(remain_black_mask)
    remain_white_indices = np.flatnonzero(remain_white_mask)

    if len(remain_black_indices) > max_pool_size_black:
        black_rank = sorted(remain_black_indices, key=lambda i: (float(black_mean[i]), i))
        overflow = len(remain_black_indices) - max_pool_size_black
        deactivate_black_capacity = black_rank[:overflow]

    if len(remain_white_indices) > max_pool_size_white:
        white_rank = sorted(remain_white_indices, key=lambda j: (float(white_mean[j]), j))
        overflow = len(remain_white_indices) - max_pool_size_white
        deactivate_white_capacity = white_rank[:overflow]

    for i in deactivate_black_capacity:
        population_black.deactivate(i)
    for j in deactivate_white_capacity:
        population_white.deactivate(j)

    return {
        "black_mean_win_rates": black_mean,
        "white_mean_win_rates": white_mean,
        "black_threshold_pruned": deactivate_black_threshold,
        "white_threshold_pruned": deactivate_white_threshold,
        "black_capacity_pruned": deactivate_black_capacity,
        "white_capacity_pruned": deactivate_white_capacity,
        "black_active_mask": population_black.get_active_mask(),
        "white_active_mask": population_white.get_active_mask(),
    }


def calculate_jpc(
    payoffs: np.ndarray,
    active_row_mask: np.ndarray | list[bool] | None = None,
    active_col_mask: np.ndarray | list[bool] | None = None,
):
    row_mask = _normalize_mask(active_row_mask, payoffs.shape[0])
    col_mask = _normalize_mask(active_col_mask, payoffs.shape[1])
    payoffs = payoffs[np.ix_(row_mask, col_mask)]
    assert len(payoffs.shape) == 2 and payoffs.shape[0] == payoffs.shape[1]
    n = payoffs.shape[0]
    assert n > 1
    d = np.trace(payoffs) / n
    o = (np.sum(payoffs) - n * d) / (n * (n - 1))
    r = (d - o) / d
    return r
