import contextlib
import copy
import enum
import functools
import logging
import os
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
    def __init__(
        self,
        dir: str,
        initial_policy: _policy_t | list[_policy_t] = uniform_policy,
        device: DeviceLike = "cuda",
    ):
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)

        self._module_cnt = 0
        self._module = None
        self._module_backup = None
        self._idx = -1
        self.device = device

        # this should be deterministic, as PSRO requires pure strategies.
        self._interaction_type = InteractionType.MODE

        self.policy_sets: list[_policy_t | int] = []
        self.active_mask: list[bool] = []

        if isinstance(initial_policy, (TensorDictModule, Policy)):
            self.add(initial_policy)
        elif isinstance(initial_policy, list):
            for _ip in initial_policy:
                assert isinstance(_ip, (TensorDictModule, Policy))
                self.add(_ip)
        else:
            self.policy_sets.append(initial_policy)
            self.active_mask.append(True)

        self._func = None
        self.sample()

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

    def activate_all(self):
        self.active_mask = [True for _ in self.policy_sets]

    def set_active_indices(self, indices: list[int] | np.ndarray):
        if len(self.policy_sets) == 0:
            raise RuntimeError("Cannot set active indices on an empty population.")
        mask = np.zeros(len(self.policy_sets), dtype=bool)
        if len(indices) > 0:
            mask[np.asarray(indices, dtype=int)] = True
        if not mask.any():
            raise RuntimeError("Population active set cannot be empty.")
        self.active_mask = mask.tolist()

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

    def remove(self, index: int):
        entry = self.policy_sets.pop(index)
        self.active_mask.pop(index)
        if isinstance(entry, int):
            path = os.path.join(self.dir, f"{entry}.pt")
            if os.path.isfile(path):
                os.remove(path)
        if self._idx == index:
            self._idx = -1
        elif self._idx > index:
            self._idx -= 1
        if len(self.policy_sets) == 0:
            self._func = None
            self._idx = -1
        elif self.active_count() == 0:
            self.activate_all()

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
    n_row = len(population_0)
    n_col = len(population_1)
    payoffs = np.zeros(shape=(n_row, n_col))
    for i in range(n_row):
        for j in range(n_col):
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
    n_row = len(population_0)
    n_col = len(population_1)
    new_payoffs = np.zeros(shape=(n_row, n_col))
    if old_payoffs is not None:
        if old_payoffs.shape != (n_row - 1, n_col - 1):
            raise ValueError(
                f"old_payoffs shape {old_payoffs.shape} is incompatible with new shape {(n_row, n_col)}"
            )
        new_payoffs[:-1, :-1] = old_payoffs
    for j in range(n_col):
        with population_0.fixed_behavioural_strategy(index=n_row - 1):
            with population_1.fixed_behavioural_strategy(index=j):
                wr = eval_win_rate(
                    env=env,
                    player_black=population_0,
                    player_white=population_1,
                )
        new_payoffs[-1, j] = 2 * wr - 1
    for i in range(n_row - 1):
        with population_0.fixed_behavioural_strategy(index=i):
            with population_1.fixed_behavioural_strategy(index=n_col - 1):
                wr = eval_win_rate(
                    env=env,
                    player_black=population_0,
                    player_white=population_1,
                )
        new_payoffs[i, -1] = 2 * wr - 1
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


def black_mean_win_rates(payoffs: np.ndarray) -> np.ndarray:
    return ((payoffs + 1.0) / 2.0).mean(axis=1)


def white_mean_win_rates(payoffs: np.ndarray) -> np.ndarray:
    return ((1.0 - payoffs) / 2.0).mean(axis=0)


def select_active_indices(
    win_rates: np.ndarray,
    mid_low: float,
    mid_high: float,
    active_min: int,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    if not (0.0 <= mid_low <= mid_high <= 1.0):
        raise ValueError("mid_low and mid_high must satisfy 0 <= mid_low <= mid_high <= 1")
    if active_min <= 0:
        raise ValueError("active_min must be positive")

    win_rates = np.asarray(win_rates, dtype=np.float64)
    all_indices = np.arange(len(win_rates), dtype=int)
    mid_indices = all_indices[(win_rates >= mid_low) & (win_rates <= mid_high)]
    easy_indices = all_indices[win_rates > mid_high]
    hard_indices = all_indices[win_rates < mid_low]

    active_indices = mid_indices.copy()
    target = min(active_min, len(all_indices))
    if len(active_indices) < target:
        pool = np.concatenate([easy_indices, hard_indices])
        need = target - len(active_indices)
        if len(pool) > 0:
            chosen = rng.choice(pool, size=min(need, len(pool)), replace=False)
            active_indices = np.concatenate([active_indices, np.sort(chosen.astype(int))])

    if len(active_indices) == 0 and len(all_indices) > 0:
        active_indices = all_indices.copy()

    active_indices = np.unique(active_indices.astype(int))
    return {
        "active_indices": active_indices,
        "mid_indices": mid_indices.astype(int),
        "easy_indices": easy_indices.astype(int),
        "hard_indices": hard_indices.astype(int),
    }


def eval_black_win_rates_against_population(
    env,
    current_black,
    white_population: Population,
) -> np.ndarray:
    win_rates = np.zeros(len(white_population), dtype=np.float64)
    for j in range(len(white_population)):
        with white_population.fixed_behavioural_strategy(index=j):
            win_rates[j] = eval_win_rate(
                env=env,
                player_black=current_black,
                player_white=white_population,
            )
    return win_rates


def eval_white_win_rates_against_population(
    env,
    black_population: Population,
    current_white,
) -> np.ndarray:
    win_rates = np.zeros(len(black_population), dtype=np.float64)
    for i in range(len(black_population)):
        with black_population.fixed_behavioural_strategy(index=i):
            black_wr = eval_win_rate(
                env=env,
                player_black=black_population,
                player_white=current_white,
            )
        win_rates[i] = 1.0 - black_wr
    return win_rates


def mid_ratio(win_rates: np.ndarray, mid_low: float, mid_high: float) -> float:
    win_rates = np.asarray(win_rates, dtype=np.float64)
    if win_rates.size == 0:
        return 0.0
    return float(np.mean((win_rates >= mid_low) & (win_rates <= mid_high)))


def _average_cosine_similarity_to_mean(vectors: np.ndarray) -> float:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {vectors.shape}")
    if vectors.shape[0] == 0:
        return 0.0
    if vectors.shape[0] == 1:
        return 1.0

    mean_vec = vectors.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    if mean_norm <= 1e-12:
        return 1.0

    sims = []
    for vec in vectors:
        vec_norm = np.linalg.norm(vec)
        if vec_norm <= 1e-12:
            sims.append(1.0)
        else:
            sims.append(float(np.dot(vec, mean_vec) / (vec_norm * mean_norm)))
    return float(np.mean(sims))


def black_archive_similarity(payoffs: np.ndarray) -> float:
    black_vectors = (np.asarray(payoffs, dtype=np.float64) + 1.0) / 2.0
    return _average_cosine_similarity_to_mean(black_vectors)


def white_archive_similarity(payoffs: np.ndarray) -> float:
    white_vectors = ((1.0 - np.asarray(payoffs, dtype=np.float64)) / 2.0).T
    return _average_cosine_similarity_to_mean(white_vectors)


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
