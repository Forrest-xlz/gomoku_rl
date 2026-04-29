#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Role-separated Elo evaluation for Gomoku temporal-feature model pools.

Expected local model layout, with minor typo tolerance:

    elo_models/                         # or eval_models/
        temporal_features_1/            # also supports temporal_feature_1 / temporal_feaure_1
            black/black_00000.pt        # also supports balck_00000.pt typo
            white/white_00000.pt
        temporal_features_6/
            black/black_00000.pt
            white/white_00000.pt

This script DOES NOT bind black_xxxxx.pt and white_xxxxx.pt into one agent.
Instead it builds two role-specific model pools:

    black pool = all temporal=1 black models + all temporal=6 black models
    white pool = all temporal=1 white models + all temporal=6 white models

It evaluates the real rectangular payoff matrix:

    black_vs_white_payoff[i, j] = score of black_model_i as black
                                 against white_model_j as white

Then it computes role-separated Elo by joint maximum likelihood with a shared
black-first-move alpha term.

Joint role-separated MLE:
    P(black_i scores against white_j) = sigmoid(alpha + black_skill_i - white_skill_j)

The reported black skills and white skills are centered separately, so each
role-specific Elo pool has mean 1200. Alpha is reported separately as the
fitted global black advantage in Elo points. Black Elo and white Elo are two
role-specific scales; predictions use black_elo - white_elo + black_advantage_elo.

For reporting, it also writes synthetic square model-vs-model payoff matrices:

    black_model_payoff_for_elo.csv
    white_model_payoff_for_elo.csv

Those square matrices are derived from MLE role-specific skills, not from
illegal black-vs-black or white-vs-white games.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import InteractionType, set_interaction_type


# -----------------------------------------------------------------------------
# Make the script runnable both from project root and from scripts/.
# -----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
for _candidate_root in (_THIS_FILE.parent, _THIS_FILE.parent.parent):
    if (_candidate_root / "gomoku_rl").is_dir():
        if str(_candidate_root) not in sys.path:
            sys.path.insert(0, str(_candidate_root))
        break

from gomoku_rl import CONFIG_PATH  # noqa: E402
from gomoku_rl.env import GomokuEnv  # noqa: E402
from gomoku_rl.policy import get_pretrained_policy  # noqa: E402


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RoleModel:
    """One checkpoint in either the black role pool or white role pool."""

    name: str
    color: str
    temporal_steps: int
    step: int
    path: str


@dataclass
class EvalConfig:
    model_root: str
    temporal_steps: list[int]
    board_size: int
    num_envs: int
    num_repeats: int
    device: str
    algo_cfg: str
    output_dir: str
    average_rating: float
    cache_size: int
    interaction: str
    resume: bool
    step_min: int | None
    step_max: int | None
    step_mod: int | None
    limit_per_temporal: int | None
    elo_l2: float
    elo_max_iter: int
    elo_patience: int
    elo_lr: float
    elo_min_delta: float


# -----------------------------------------------------------------------------
# Model discovery
# -----------------------------------------------------------------------------
_CKPT_RE = re.compile(r"^(?P<color>black|balck|white)_(?P<step>\d+)\.pt$", re.IGNORECASE)
_TEMPORAL_DIR_RE = re.compile(
    r"^temporal_(?:features?|feaure|feaures)_(?P<n>\d+)$",
    re.IGNORECASE,
)


def resolve_model_root(model_root_arg: str | None) -> Path:
    """Resolve model root. Supports both ./elo_models and ./eval_models by default."""
    if model_root_arg:
        root = Path(model_root_arg).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"model root does not exist: {root}")
        return root

    for p in [Path("elo_models"), Path("eval_models")]:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        "Cannot find model root. Expected ./elo_models or ./eval_models. "
        "Use --model-root to specify it explicitly."
    )


def find_temporal_dir(model_root: Path, temporal_steps: int) -> Path | None:
    """Find temporal directory and tolerate common spelling mistakes."""
    n = int(temporal_steps)
    candidates = [
        model_root / f"temporal_features_{n}",
        model_root / f"temporal_feature_{n}",
        model_root / f"temporal_feaure_{n}",
        model_root / f"temporal_feaures_{n}",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    for child in model_root.iterdir():
        if not child.is_dir():
            continue
        match = _TEMPORAL_DIR_RE.match(child.name)
        if match and int(match.group("n")) == n:
            return child.resolve()

    return None


def scan_temporal_steps(model_root: Path) -> list[int]:
    """Auto-detect temporal steps from folder names."""
    out: set[int] = set()
    for child in model_root.iterdir():
        if not child.is_dir():
            continue
        match = _TEMPORAL_DIR_RE.match(child.name)
        if match:
            out.add(int(match.group("n")))
    return sorted(out)


def _color_dir(temporal_dir: Path, color: str) -> Path:
    if color == "black":
        candidates = [temporal_dir / "black", temporal_dir / "balck"]
    else:
        candidates = [temporal_dir / "white"]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return candidates[0].resolve()


def scan_color_models(folder: Path, expected_color: str) -> dict[int, Path]:
    """Return {training_step: checkpoint_path} for one color folder."""
    out: dict[int, Path] = {}
    if not folder.exists():
        return out

    for path in folder.glob("*.pt"):
        match = _CKPT_RE.match(path.name)
        if match is None:
            continue

        color = match.group("color").lower()
        if color == "balck":
            color = "black"
        if color != expected_color:
            continue

        step = int(match.group("step"))
        out[step] = path.resolve()

    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _filter_steps(
    steps: Sequence[int],
    step_min: int | None,
    step_max: int | None,
    step_mod: int | None,
    limit_per_temporal: int | None,
) -> list[int]:
    filtered: list[int] = []
    for step in steps:
        if step_min is not None and step < step_min:
            continue
        if step_max is not None and step > step_max:
            continue
        if step_mod is not None and step_mod > 0 and step % step_mod != 0:
            continue
        filtered.append(int(step))

    if limit_per_temporal is not None and limit_per_temporal > 0 and len(filtered) > limit_per_temporal:
        indices = np.linspace(0, len(filtered) - 1, limit_per_temporal)
        keep = sorted({int(round(x)) for x in indices})
        filtered = [filtered[i] for i in keep]
    return filtered


def discover_role_models(
    model_root: Path,
    temporal_steps: Iterable[int],
    step_min: int | None = None,
    step_max: int | None = None,
    step_mod: int | None = None,
    limit_per_temporal: int | None = None,
) -> tuple[list[RoleModel], list[RoleModel]]:
    """Discover black and white model pools separately."""
    black_pool: list[RoleModel] = []
    white_pool: list[RoleModel] = []

    for temporal_n in temporal_steps:
        temporal_dir = find_temporal_dir(model_root, int(temporal_n))
        if temporal_dir is None:
            print(f"[WARN] temporal={temporal_n}: directory not found under {model_root}")
            continue

        black_dir = _color_dir(temporal_dir, "black")
        white_dir = _color_dir(temporal_dir, "white")
        black_models = scan_color_models(black_dir, "black")
        white_models = scan_color_models(white_dir, "white")

        if not black_models:
            print(f"[WARN] temporal={temporal_n}: no black checkpoints found in {black_dir}")
        if not white_models:
            print(f"[WARN] temporal={temporal_n}: no white checkpoints found in {white_dir}")

        black_steps = _filter_steps(
            sorted(black_models.keys()),
            step_min=step_min,
            step_max=step_max,
            step_mod=step_mod,
            limit_per_temporal=limit_per_temporal,
        )
        white_steps = _filter_steps(
            sorted(white_models.keys()),
            step_min=step_min,
            step_max=step_max,
            step_mod=step_mod,
            limit_per_temporal=limit_per_temporal,
        )

        if limit_per_temporal is not None:
            print(
                f"[INFO] temporal={temporal_n}: use "
                f"{len(black_steps)} black and {len(white_steps)} white checkpoints "
                f"after --limit-per-temporal={limit_per_temporal}."
            )

        for step in black_steps:
            black_pool.append(
                RoleModel(
                    name=f"black_t{int(temporal_n)}_s{step:05d}",
                    color="black",
                    temporal_steps=int(temporal_n),
                    step=int(step),
                    path=str(black_models[step]),
                )
            )
        for step in white_steps:
            white_pool.append(
                RoleModel(
                    name=f"white_t{int(temporal_n)}_s{step:05d}",
                    color="white",
                    temporal_steps=int(temporal_n),
                    step=int(step),
                    path=str(white_models[step]),
                )
            )

    black_pool.sort(key=lambda m: (m.temporal_steps, m.step, m.name))
    white_pool.sort(key=lambda m: (m.temporal_steps, m.step, m.name))
    return black_pool, white_pool


# -----------------------------------------------------------------------------
# Policy adapter and cache
# -----------------------------------------------------------------------------
class TemporalPolicyAdapter:
    """
    Wrap a policy trained with temporal_steps=n.

    The evaluation environment is created with max_history_steps, e.g. 6, so its
    observation is:

        [current_player_board, opponent_board, last_1, ..., last_max]

    A temporal=1 policy only receives:

        [current_player_board, opponent_board, last_1]
    """

    def __init__(self, policy, temporal_steps: int):
        self.policy = policy
        self.temporal_steps = int(temporal_steps)
        if hasattr(self.policy, "eval"):
            self.policy.eval()

    def eval(self):
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        return self

    def train(self):
        if hasattr(self.policy, "train"):
            self.policy.train()
        return self

    @torch.no_grad()
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        obs: torch.Tensor = tensordict.get("observation")
        action_mask: torch.Tensor = tensordict.get("action_mask")

        required_channels = 2 + self.temporal_steps
        if obs.shape[-3] < required_channels:
            raise RuntimeError(
                f"Observation has {obs.shape[-3]} channels, but temporal={self.temporal_steps} "
                f"policy needs {required_channels} channels."
            )

        model_obs = obs[:, :required_channels, :, :]
        model_td = TensorDict(
            {
                "observation": model_obs,
                "action_mask": action_mask,
            },
            batch_size=tensordict.batch_size,
            device=obs.device,
        )
        model_td = self.policy(model_td)
        tensordict.set("action", model_td.get("action").to(obs.device))
        return tensordict


class PolicyBank:
    """LRU cache to avoid keeping every checkpoint in GPU memory."""

    def __init__(
        self,
        algo_cfg: DictConfig,
        board_size: int,
        num_envs: int,
        device: str,
        cache_size: int,
    ):
        self.algo_cfg = algo_cfg
        self.board_size = int(board_size)
        self.num_envs = int(num_envs)
        self.device = device
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[tuple[str, int], TemporalPolicyAdapter] = OrderedDict()
        self._spec_envs: dict[int, GomokuEnv] = {}

    def _get_spec_env(self, temporal_steps: int) -> GomokuEnv:
        temporal_steps = int(temporal_steps)
        if temporal_steps not in self._spec_envs:
            self._spec_envs[temporal_steps] = GomokuEnv(
                num_envs=self.num_envs,
                board_size=self.board_size,
                device=self.device,
                use_temporal_feature=True,
                temporal_num_steps=temporal_steps,
                observation_mode="temporal_move_history",
            )
        return self._spec_envs[temporal_steps]

    def get(self, model: RoleModel) -> TemporalPolicyAdapter:
        key = (str(Path(model.path).resolve()), int(model.temporal_steps))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        spec_env = self._get_spec_env(model.temporal_steps)
        policy = get_pretrained_policy(
            name=str(self.algo_cfg.name),
            cfg=self.algo_cfg,
            action_spec=spec_env.action_spec,
            observation_spec=spec_env.observation_spec,
            checkpoint_path=key[0],
            device=self.device,
        )
        adapter = TemporalPolicyAdapter(policy=policy, temporal_steps=model.temporal_steps)
        self._cache[key] = adapter

        while len(self._cache) > self.cache_size:
            old_key, old_adapter = self._cache.popitem(last=False)
            del old_adapter
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            print(f"[CACHE] evicted {Path(old_key[0]).name} temporal={old_key[1]}")

        return adapter


# -----------------------------------------------------------------------------
# Evaluation and Elo
# -----------------------------------------------------------------------------
def _interaction_type(name: str) -> InteractionType:
    name = name.lower().strip()
    if name == "random":
        return InteractionType.RANDOM
    if name == "mode":
        return InteractionType.MODE
    raise ValueError(f"Unknown interaction type: {name}")


@torch.no_grad()
def eval_black_score(
    env: GomokuEnv,
    black_policy: TemporalPolicyAdapter,
    white_policy: TemporalPolicyAdapter,
    num_repeats: int,
    interaction: str,
) -> float:
    """
    Return black's average score.

    score = 1.0 if black wins, 0.0 if white wins, 0.5 if no winner is produced
    within board_size * board_size moves.
    """
    scores: list[float] = []
    interaction_value = _interaction_type(interaction)

    if hasattr(black_policy, "eval"):
        black_policy.eval()
    if hasattr(white_policy, "eval"):
        white_policy.eval()

    with set_interaction_type(type=interaction_value):
        for _ in range(int(num_repeats)):
            tensordict = env.reset()
            finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            black_score = torch.full(
                (env.num_envs,),
                0.5,
                dtype=torch.float32,
                device=env.device,
            )

            for move_index in range(env.board_size * env.board_size):
                if move_index % 2 == 0:
                    tensordict = black_policy(tensordict)
                else:
                    tensordict = white_policy(tensordict)

                tensordict = env.step_and_maybe_reset(tensordict)
                done: torch.Tensor = tensordict.get("done")
                newly_done = done & (~finished)

                if newly_done.any():
                    black_win = tensordict["stats", "black_win"].float()
                    black_score[newly_done] = black_win[newly_done]
                    finished |= newly_done

                if bool(finished.all().item()):
                    break

            scores.append(float(black_score.mean().item()))

    env.reset()
    return float(np.mean(scores))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _skill_to_elo_from_centered_skill(
    centered_skill_logit: np.ndarray,
    average_rating: float = 1200.0,
) -> np.ndarray:
    """
    Convert centered logit-scale skills to Elo.

    Standard Elo:
        p = 1 / (1 + 10^(-(R_A - R_B) / 400))

    Therefore:
        R_A - R_B = logit(p) * 400 / ln(10)
    """
    centered_skill = np.asarray(centered_skill_logit, dtype=np.float64)
    return centered_skill * (400.0 / math.log(10.0)) + float(average_rating)


def fit_role_mle_with_alpha(
    black_vs_white_payoff: np.ndarray,
    average_rating: float = 1200.0,
    games_per_pair: int = 1024,
    l2: float = 1e-4,
    max_iter: int = 5000,
    patience: int = 100,
    lr: float = 0.05,
    min_delta: float = 1e-10,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    """
    Fit one joint role-separated Bradley-Terry / logistic Elo model with alpha.

    Input:
        raw[i, j] = black_model_i's score as BLACK against white_model_j as WHITE

    Model:
        P(black_i scores against white_j) = sigmoid(alpha + black_skill_i - white_skill_j)

    Identifiability:
        black_skill is centered to mean 0.
        white_skill is centered to mean 0.

    Interpretation:
        black_elo_i = average_rating + black_skill_i * 400 / ln(10)
        white_elo_j = average_rating + white_skill_j * 400 / ln(10)
        black_advantage_elo = alpha * 400 / ln(10)

        Predicted black score:
            sigmoid((black_elo_i - white_elo_j + black_advantage_elo) * ln(10) / 400)

    Optimization:
        Adam minimizes the binomial negative log-likelihood.
        Early stopping triggers when the best training loss has not improved for
        `patience` consecutive iterations.
    """
    raw = np.asarray(black_vs_white_payoff, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"black_vs_white_payoff must be 2D, got shape={raw.shape}")
    if raw.size == 0:
        raise ValueError("empty payoff matrix")
    if np.isnan(raw).any():
        raise ValueError("black_vs_white_payoff still contains NaN values; finish evaluation first.")
    if np.any(raw < 0.0) or np.any(raw > 1.0):
        raise ValueError("black_vs_white_payoff values must be in [0, 1].")

    num_black, num_white = raw.shape
    games_np = np.full_like(raw, float(games_per_pair), dtype=np.float64)

    y = torch.tensor(raw, dtype=torch.float64)
    n = torch.tensor(games_np, dtype=torch.float64)

    black_raw = torch.zeros(num_black, dtype=torch.float64, requires_grad=True)
    white_raw = torch.zeros(num_white, dtype=torch.float64, requires_grad=True)
    alpha = torch.zeros((), dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.Adam([black_raw, white_raw, alpha], lr=float(lr))

    def objective() -> torch.Tensor:
        black_skill = black_raw - black_raw.mean()
        white_skill = white_raw - white_raw.mean()
        logits = alpha + black_skill[:, None] - white_skill[None, :]

        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y,
            reduction="none",
        )
        loss = (bce * n).sum() / n.sum().clamp_min(1.0)

        if l2 > 0:
            loss = loss + float(l2) * (
                black_skill.square().mean()
                + white_skill.square().mean()
                + alpha.square()
            )
        return loss

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_iter = 0
    bad_count = 0
    last_loss = float("inf")
    loss_history: list[float] = []

    for iteration in range(1, int(max_iter) + 1):
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        optimizer.step()

        last_loss = float(loss.detach().cpu().item())
        loss_history.append(last_loss)

        if last_loss < best_loss - float(min_delta):
            best_loss = last_loss
            best_iter = iteration
            bad_count = 0
            with torch.no_grad():
                best_state = {
                    "black_raw": black_raw.detach().clone(),
                    "white_raw": white_raw.detach().clone(),
                    "alpha": alpha.detach().clone(),
                }
        else:
            bad_count += 1

        if bad_count >= int(patience):
            break

    if best_state is not None:
        with torch.no_grad():
            black_raw.copy_(best_state["black_raw"])
            white_raw.copy_(best_state["white_raw"])
            alpha.copy_(best_state["alpha"])

    with torch.no_grad():
        black_skill = (black_raw - black_raw.mean()).detach().cpu().numpy()
        white_skill = (white_raw - white_raw.mean()).detach().cpu().numpy()
        alpha_logit = float(alpha.detach().cpu().item())

    pred_black_score = _sigmoid(alpha_logit + black_skill[:, None] - white_skill[None, :])
    pred_white_score = 1.0 - pred_black_score.T

    black_elo = _skill_to_elo_from_centered_skill(black_skill, average_rating=average_rating)
    white_elo = _skill_to_elo_from_centered_skill(white_skill, average_rating=average_rating)
    black_advantage_elo = alpha_logit * (400.0 / math.log(10.0))

    # Derived square payoff inside each same-role pool. These are not real games;
    # they are a visualization of fitted relative skills within the same role.
    black_model_payoff = _sigmoid(black_skill[:, None] - black_skill[None, :])
    white_model_payoff = _sigmoid(white_skill[:, None] - white_skill[None, :])
    np.fill_diagonal(black_model_payoff, 0.5)
    np.fill_diagonal(white_model_payoff, 0.5)

    eps = 1e-12
    pred_clip = np.clip(pred_black_score, eps, 1.0 - eps)
    nll = -np.sum(
        games_np
        * (
            raw * np.log(pred_clip)
            + (1.0 - raw) * np.log(1.0 - pred_clip)
        )
    )
    nll_per_game = float(nll / max(np.sum(games_np), 1.0))
    rmse = float(np.sqrt(np.mean((pred_black_score - raw) ** 2)))

    diagnostics = {
        "model": "sigmoid(alpha + black_skill_i - white_skill_j)",
        "average_rating": float(average_rating),
        "games_per_pair": int(games_per_pair),
        "l2": float(l2),
        "max_iter": int(max_iter),
        "patience": int(patience),
        "lr": float(lr),
        "min_delta": float(min_delta),
        "iterations_run": int(len(loss_history)),
        "best_iter": int(best_iter),
        "stopped_early": bool(len(loss_history) < int(max_iter)),
        "best_training_loss_with_l2": float(best_loss),
        "last_training_loss_with_l2": float(last_loss),
        "nll_per_game_without_l2": nll_per_game,
        "rmse": rmse,
        "alpha_logit": float(alpha_logit),
        "black_advantage_elo": float(black_advantage_elo),
        "black_skill_mean": float(np.mean(black_skill)),
        "white_skill_mean": float(np.mean(white_skill)),
        "black_skill_std": float(np.std(black_skill)),
        "white_skill_std": float(np.std(white_skill)),
        "loss_history_head": [float(x) for x in loss_history[:20]],
        "loss_history_tail": [float(x) for x in loss_history[-20:]],
        "note": (
            "Black and white skills are fitted jointly with one shared alpha. "
            "black_elo and white_elo are centered separately to average_rating. "
            "Predicted black score uses black_elo - white_elo + black_advantage_elo."
        ),
    }

    return (
        black_elo,
        white_elo,
        black_model_payoff,
        white_model_payoff,
        black_skill,
        white_skill,
        pred_black_score,
        pred_white_score,
        diagnostics,
    )


def role_elos_from_black_vs_white_payoff_mle_with_alpha(
    black_vs_white_payoff: np.ndarray,
    average_rating: float = 1200.0,
    games_per_pair: int = 1024,
    l2: float = 1e-4,
    max_iter: int = 5000,
    patience: int = 100,
    lr: float = 0.05,
    min_delta: float = 1e-10,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    """
    Compute role-separated MLE Elo from the real rectangular black-vs-white matrix.

    Input:
        raw[i, j] = black_model_i's score as black against white_model_j as white

    Joint model:
        P(black_i scores against white_j) = sigmoid(alpha + black_skill_i - white_skill_j)

    This is the alpha-based method:
        - black_skill_i compares black models inside the black role pool.
        - white_skill_j compares white models inside the white role pool.
        - alpha captures the global black first-move advantage.
    """
    return fit_role_mle_with_alpha(
        black_vs_white_payoff=black_vs_white_payoff,
        average_rating=average_rating,
        games_per_pair=games_per_pair,
        l2=l2,
        max_iter=max_iter,
        patience=patience,
        lr=lr,
        min_delta=min_delta,
    )


# -----------------------------------------------------------------------------
# Saving / loading
# -----------------------------------------------------------------------------
def save_matrix_csv(path: Path, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], row_header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row_header] + col_labels)
        for label, row in zip(row_labels, matrix):
            writer.writerow([label] + [f"{x:.8f}" if np.isfinite(x) else "nan" for x in row])


def load_matrix_csv_if_compatible(path: Path, row_labels: list[str], col_labels: list[str]) -> np.ndarray | None:
    if not path.exists():
        return None

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or len(rows[0]) < 2:
        return None

    old_cols = rows[0][1:]
    old_col_index = {label: i for i, label in enumerate(old_cols)}
    new_row_index = {label: i for i, label in enumerate(row_labels)}

    matrix = np.full((len(row_labels), len(col_labels)), np.nan, dtype=np.float64)
    for old_row in rows[1:]:
        if not old_row:
            continue
        row_label = old_row[0]
        if row_label not in new_row_index:
            continue
        new_i = new_row_index[row_label]
        for new_j, col_label in enumerate(col_labels):
            old_j = old_col_index.get(col_label)
            if old_j is None:
                continue
            value_text_index = old_j + 1
            if value_text_index >= len(old_row):
                continue
            try:
                matrix[new_i, new_j] = float(old_row[value_text_index])
            except ValueError:
                matrix[new_i, new_j] = np.nan

    return matrix


def save_role_elo_csv(
    path: Path,
    models: list[RoleModel],
    elo: np.ndarray,
    role_strength: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "color", "temporal_steps", "step", "elo", "mle_skill_logit", "path"])
        for model, rating, strength in zip(models, elo, role_strength):
            writer.writerow(
                [
                    model.name,
                    model.color,
                    model.temporal_steps,
                    model.step,
                    f"{float(rating):.6f}",
                    f"{float(strength):.8f}",
                    model.path,
                ]
            )


def save_combined_elo_csv(
    path: Path,
    black_models: list[RoleModel],
    black_elo: np.ndarray,
    black_strength: np.ndarray,
    white_models: list[RoleModel],
    white_elo: np.ndarray,
    white_strength: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "color", "temporal_steps", "step", "elo", "mle_skill_logit", "path"])
        for model, rating, strength in zip(black_models, black_elo, black_strength):
            writer.writerow([model.name, model.color, model.temporal_steps, model.step, f"{float(rating):.6f}", f"{float(strength):.8f}", model.path])
        for model, rating, strength in zip(white_models, white_elo, white_strength):
            writer.writerow([model.name, model.color, model.temporal_steps, model.step, f"{float(rating):.6f}", f"{float(strength):.8f}", model.path])


def save_meta_json(path: Path, config: EvalConfig, black_models: list[RoleModel], white_models: list[RoleModel]) -> None:
    payload = {
        "config": asdict(config),
        "black_models": [asdict(v) for v in black_models],
        "white_models": [asdict(v) for v in white_models],
        "meaning": {
            "black_vs_white_payoff": "row black model's score as black against column white model as white",
            "black_elo": "role-separated black-model MLE Elo from joint alpha model; mean centered to 1200",
            "white_elo": "role-separated white-model MLE Elo from joint alpha model; mean centered to 1200",
            "black_advantage_elo": "fitted global first-move advantage for black, used as black_elo - white_elo + black_advantage_elo",
            "black_model_payoff_for_elo": "derived square payoff among black models from MLE black skills",
            "white_model_payoff_for_elo": "derived square payoff among white models from MLE white skills",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _group_points(models: list[RoleModel], elo: np.ndarray) -> dict[int, list[tuple[int, float]]]:
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for model, rating in zip(models, elo):
        groups[int(model.temporal_steps)].append((int(model.step), float(rating)))
    return groups


def plot_role_curve(path: Path, models: list[RoleModel], elo: np.ndarray, title: str) -> None:
    groups = _group_points(models, elo)
    plt.figure(figsize=(10, 6))
    for temporal_steps in sorted(groups.keys()):
        points = sorted(groups[temporal_steps], key=lambda x: x[0])
        steps = [p[0] for p in points]
        ratings = [p[1] for p in points]
        plt.plot(steps, ratings, marker="o", linewidth=1.8, label=f"temporal={temporal_steps}")

    plt.xlabel("training step")
    plt.ylabel("Elo rating")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_combined_role_curve(
    path: Path,
    black_models: list[RoleModel],
    black_elo: np.ndarray,
    white_models: list[RoleModel],
    white_elo: np.ndarray,
) -> None:
    plt.figure(figsize=(11, 7))

    for role_name, models, ratings in [
        ("black", black_models, black_elo),
        ("white", white_models, white_elo),
    ]:
        groups = _group_points(models, ratings)
        for temporal_steps in sorted(groups.keys()):
            points = sorted(groups[temporal_steps], key=lambda x: x[0])
            steps = [p[0] for p in points]
            values = [p[1] for p in points]
            plt.plot(
                steps,
                values,
                marker="o",
                linewidth=1.8,
                label=f"{role_name}, temporal={temporal_steps}",
            )

    plt.xlabel("training step")
    plt.ylabel("Elo rating")
    plt.title("Role-separated Gomoku Elo curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate role-separated Gomoku Elo for temporal-feature checkpoint pools.")
    parser.add_argument(
        "--model-root",
        type=str,
        default=None,
        help="Model root. If omitted, the script tries ./elo_models then ./eval_models.",
    )
    parser.add_argument(
        "--temporal-steps",
        type=int,
        nargs="+",
        default=[1, 6],
        help="Temporal feature settings to include in the model pool. Default: 1 6.",
    )
    parser.add_argument(
        "--auto-temporal-steps",
        action="store_true",
        help="Ignore --temporal-steps and auto-detect temporal folders under model root.",
    )
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--algo-cfg",
        type=str,
        default=None,
        help="PPO yaml path. Default: cfg/algo/ppo.yaml from the project.",
    )
    parser.add_argument("--output-dir", type=str, default="elo_eval_outputs")
    parser.add_argument("--average-rating", type=float, default=1200.0)
    parser.add_argument("--elo-l2", type=float, default=1e-4, help="L2 regularization for MLE Elo fitting.")
    parser.add_argument("--elo-max-iter", type=int, default=5000, help="Maximum Adam iterations for MLE Elo fitting.")
    parser.add_argument("--elo-patience", type=int, default=100, help="Early stop after this many iterations without best-loss improvement.")
    parser.add_argument("--elo-lr", type=float, default=0.05, help="Adam learning rate for MLE Elo fitting.")
    parser.add_argument("--elo-min-delta", type=float, default=1e-10, help="Minimum best-loss improvement required to reset patience.")
    parser.add_argument("--cache-size", type=int, default=4)
    parser.add_argument(
        "--interaction",
        type=str,
        choices=["random", "mode"],
        default="random",
        help="random samples from the policy; mode takes the greedy categorical mode.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing black_vs_white_payoff.csv.")
    parser.add_argument("--step-min", type=int, default=None)
    parser.add_argument("--step-max", type=int, default=None)
    parser.add_argument("--step-mod", type=int, default=None, help="Only keep steps divisible by this value.")
    parser.add_argument(
        "--limit-per-temporal",
        type=int,
        default=None,
        help="For quick tests: evenly subsample at most this many black and white checkpoints for each temporal setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; fallback to CPU.")
        args.device = "cpu"

    torch.manual_seed(0)
    np.random.seed(0)

    model_root = resolve_model_root(args.model_root)
    temporal_steps = scan_temporal_steps(model_root) if args.auto_temporal_steps else [int(x) for x in args.temporal_steps]
    if not temporal_steps:
        raise RuntimeError(f"No temporal model folders found under {model_root}")

    algo_cfg_path = Path(args.algo_cfg).expanduser().resolve() if args.algo_cfg else Path(CONFIG_PATH) / "algo" / "ppo.yaml"
    output_dir = Path(args.output_dir).expanduser().resolve()

    algo_cfg = OmegaConf.load(algo_cfg_path)
    OmegaConf.set_struct(algo_cfg, False)

    eval_cfg = EvalConfig(
        model_root=str(model_root),
        temporal_steps=temporal_steps,
        board_size=int(args.board_size),
        num_envs=int(args.num_envs),
        num_repeats=int(args.num_repeats),
        device=str(args.device),
        algo_cfg=str(algo_cfg_path),
        output_dir=str(output_dir),
        average_rating=float(args.average_rating),
        cache_size=int(args.cache_size),
        interaction=str(args.interaction),
        resume=not bool(args.no_resume),
        step_min=args.step_min,
        step_max=args.step_max,
        step_mod=args.step_mod,
        limit_per_temporal=args.limit_per_temporal,
        elo_l2=float(args.elo_l2),
        elo_max_iter=int(args.elo_max_iter),
        elo_patience=int(args.elo_patience),
        elo_lr=float(args.elo_lr),
        elo_min_delta=float(args.elo_min_delta),
    )

    black_models, white_models = discover_role_models(
        model_root=model_root,
        temporal_steps=temporal_steps,
        step_min=args.step_min,
        step_max=args.step_max,
        step_mod=args.step_mod,
        limit_per_temporal=args.limit_per_temporal,
    )

    if not black_models:
        raise RuntimeError("No black checkpoints found. Check temporal_xxx/black/black_00000.pt layout.")
    if not white_models:
        raise RuntimeError("No white checkpoints found. Check temporal_xxx/white/white_00000.pt layout.")

    black_labels = [m.name for m in black_models]
    white_labels = [m.name for m in white_models]
    max_history_steps = max([m.temporal_steps for m in black_models + white_models])

    print(f"[INFO] model_root      = {model_root}")
    print(f"[INFO] algo_cfg        = {algo_cfg_path}")
    print(f"[INFO] output_dir      = {output_dir}")
    print(f"[INFO] device          = {args.device}")
    print(f"[INFO] num_envs        = {args.num_envs}")
    print(f"[INFO] num_repeats     = {args.num_repeats}")
    print(f"[INFO] interaction     = {args.interaction}")
    print(f"[INFO] temporal_steps  = {temporal_steps}")
    print(f"[INFO] max_history     = {max_history_steps}")
    print(f"[INFO] black models    = {len(black_models)}")
    print(f"[INFO] white models    = {len(white_models)}")

    for color, models in [("black", black_models), ("white", white_models)]:
        for temporal_n in sorted(set(m.temporal_steps for m in models)):
            count = sum(m.temporal_steps == temporal_n for m in models)
            print(f"       {color}, temporal={temporal_n}: {count} checkpoints")

    save_meta_json(output_dir / "meta.json", eval_cfg, black_models, white_models)

    payoff_path = output_dir / "black_vs_white_payoff.csv"
    payoff = None
    if eval_cfg.resume:
        payoff = load_matrix_csv_if_compatible(payoff_path, black_labels, white_labels)
        if payoff is not None:
            print(f"[INFO] resume from {payoff_path}")
    if payoff is None:
        payoff = np.full((len(black_models), len(white_models)), np.nan, dtype=np.float64)

    env = GomokuEnv(
        num_envs=args.num_envs,
        board_size=args.board_size,
        device=args.device,
        use_temporal_feature=True,
        temporal_num_steps=max_history_steps,
        observation_mode="temporal_move_history",
    )

    bank = PolicyBank(
        algo_cfg=algo_cfg,
        board_size=args.board_size,
        num_envs=args.num_envs,
        device=args.device,
        cache_size=args.cache_size,
    )

    total_pairs = len(black_models) * len(white_models)
    for i, black_model in enumerate(black_models):
        for j, white_model in enumerate(white_models):
            if np.isfinite(payoff[i, j]):
                continue

            progress = int(np.isfinite(payoff).sum()) + 1
            print(
                f"[{progress:>5}/{total_pairs}] "
                f"{black_model.name} as BLACK  vs  {white_model.name} as WHITE"
            )

            black_policy = bank.get(black_model)
            white_policy = bank.get(white_model)
            score = eval_black_score(
                env=env,
                black_policy=black_policy,
                white_policy=white_policy,
                num_repeats=args.num_repeats,
                interaction=args.interaction,
            )
            payoff[i, j] = score
            print(f"        black_score={score:.4f}, white_score={1.0 - score:.4f}")

            save_matrix_csv(payoff_path, payoff, black_labels, white_labels, row_header="black_model")

    (
        black_elo,
        white_elo,
        black_model_payoff,
        white_model_payoff,
        black_strength,
        white_strength,
        pred_black_score,
        pred_white_score,
        mle_diagnostics,
    ) = role_elos_from_black_vs_white_payoff_mle_with_alpha(
        payoff,
        average_rating=args.average_rating,
        games_per_pair=int(args.num_envs) * int(args.num_repeats),
        l2=args.elo_l2,
        max_iter=args.elo_max_iter,
        patience=args.elo_patience,
        lr=args.elo_lr,
        min_delta=args.elo_min_delta,
    )

    save_matrix_csv(payoff_path, payoff, black_labels, white_labels, row_header="black_model")
    save_matrix_csv(output_dir / "black_model_payoff_for_elo.csv", black_model_payoff, black_labels, black_labels, row_header="black_model")
    save_matrix_csv(output_dir / "white_model_payoff_for_elo.csv", white_model_payoff, white_labels, white_labels, row_header="white_model")

    # MLE predictions:
    #   pred_black_score rows = black models, columns = white models.
    #   pred_white_score rows = white models, columns = black models.
    save_matrix_csv(output_dir / "mle_predicted_black_score.csv", pred_black_score, black_labels, white_labels, row_header="black_model")
    save_matrix_csv(output_dir / "mle_predicted_white_score.csv", pred_white_score, white_labels, black_labels, row_header="white_model")
    (output_dir / "mle_with_alpha_diagnostics.json").write_text(
        json.dumps(mle_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_role_elo_csv(output_dir / "black_elo_ratings.csv", black_models, black_elo, black_strength)
    save_role_elo_csv(output_dir / "white_elo_ratings.csv", white_models, white_elo, white_strength)
    save_combined_elo_csv(
        output_dir / "role_elo_ratings.csv",
        black_models,
        black_elo,
        black_strength,
        white_models,
        white_elo,
        white_strength,
    )

    plot_role_curve(output_dir / "black_elo_curve.png", black_models, black_elo, "Black-model MLE Elo curve")
    plot_role_curve(output_dir / "white_elo_curve.png", white_models, white_elo, "White-model MLE Elo curve")
    plot_combined_role_curve(output_dir / "role_elo_curve.png", black_models, black_elo, white_models, white_elo)

    print("\n[RESULT] Black-model Elo ratings:")
    for model, rating in sorted(zip(black_models, black_elo), key=lambda x: (x[0].temporal_steps, x[0].step)):
        print(f"  {model.name:>20s}  Elo={rating:8.2f}")

    print("\n[RESULT] White-model Elo ratings:")
    for model, rating in sorted(zip(white_models, white_elo), key=lambda x: (x[0].temporal_steps, x[0].step)):
        print(f"  {model.name:>20s}  Elo={rating:8.2f}")

    print("\n[RESULT] Fitted global black advantage:")
    print(f"  alpha_logit={mle_diagnostics['alpha_logit']:.8f}")
    print(f"  black_advantage_elo={mle_diagnostics['black_advantage_elo']:.2f}")
    print(f"  MLE iterations={mle_diagnostics['iterations_run']}  best_iter={mle_diagnostics['best_iter']}  stopped_early={mle_diagnostics['stopped_early']}")

    print(f"\n[DONE] real payoff:       {payoff_path}")
    print(f"[DONE] black Elo csv:    {output_dir / 'black_elo_ratings.csv'}")
    print(f"[DONE] white Elo csv:    {output_dir / 'white_elo_ratings.csv'}")
    print(f"[DONE] combined Elo csv: {output_dir / 'role_elo_ratings.csv'}")
    print(f"[DONE] diagnostics:      {output_dir / 'mle_with_alpha_diagnostics.json'}")
    print(f"[DONE] combined figure:  {output_dir / 'role_elo_curve.png'}")


if __name__ == "__main__":
    main()
