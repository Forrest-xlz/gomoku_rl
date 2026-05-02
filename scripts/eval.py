#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydra eval.py for Gomoku checkpoints with per-checkpoint architecture metadata.

What this script does
---------------------
1. Reads cfg/eval.yaml.
2. Uses cfg.h as the model list. Each item can specify:
   - checkpoint: one checkpoint used as black+white, or with color: black/white/both
   - black_checkpoint and/or white_checkpoint: role-separated checkpoints
   - temporal, num_channels, num_residual_blocks
3. Evaluates every black model against every white model.
4. Fits role-separated MLE Elo using the same model as scripts/elo_eval.py:

       P(black_i scores against white_j)
       = sigmoid(alpha + black_skill_i - white_skill_j)

   black_skill and white_skill are centered separately, then converted to Elo.
5. Saves payoff, predicted payoff, Elo CSVs, diagnostics, and a resume cache.

Recommended cfg/eval.yaml example
---------------------------------
seed: 0
board_size: 15
num_envs: 512
num_repeats: 2
device: cuda

# random = sample from policy distribution; mode = greedy / argmax-like
interaction: random

# If you keep Hydra defaults, cfg.algo is used directly.
defaults:
  - _self_
  - algo: ppo

# Optional. Used only when cfg.algo does not exist.
algo_cfg: ppo

output_dir: eval_outputs
resume: true
cache_size: 4

average_rating: 1200
elo_l2: 1.0e-4
elo_max_iter: 5000
elo_patience: 100
elo_lr: 0.05
elo_min_delta: 1.0e-10

h:
  # Format A: one checkpoint, add to both black pool and white pool.
  - name: model_00000
    checkpoint: /root/autodl-tmp/gomoku_rl/train_model_pool/black/black_00000.pt
    temporal: 6
    num_channels: 64
    num_residual_blocks: 4

  # Format B: one checkpoint, role-specific.
  - name: black_00100
    color: black
    checkpoint: /root/autodl-tmp/gomoku_rl/train_model_pool/black/black_00100.pt
    temporal: 6
    num_channels: 64
    num_residual_blocks: 4

  - name: white_00100
    color: white
    checkpoint: /root/autodl-tmp/gomoku_rl/train_model_pool/white/white_00100.pt
    temporal: 6
    num_channels: 64
    num_residual_blocks: 4

  # Format C: one item contains one black checkpoint and one white checkpoint.
  - name: epoch_00200
    black_checkpoint: /root/autodl-tmp/gomoku_rl/train_model_pool/black/black_00200.pt
    white_checkpoint: /root/autodl-tmp/gomoku_rl/train_model_pool/white/white_00200.pt
    temporal: 6
    num_channels: 64
    num_residual_blocks: 4

Run
---
python scripts/eval.py

or override from CLI:
python scripts/eval.py \
  h='[{"name":"m1","color":"black","checkpoint":"/path/black_00000.pt","temporal":6,"num_channels":64,"num_residual_blocks":4},{"name":"m1","color":"white","checkpoint":"/path/white_00000.pt","temporal":6,"num_channels":64,"num_residual_blocks":4}]'
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import hydra
import numpy as np
import torch
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
class ModelSpec:
    """One checkpoint in either the black role pool or white role pool."""

    name: str
    color: str
    checkpoint: str
    temporal: int
    num_channels: int
    num_residual_blocks: int
    label: str = ""
    step: int = -1

    @property
    def display_label(self) -> str:
        if self.label:
            return self.label
        return (
            f"{self.name}"
            f"|{self.color}"
            f"|t{int(self.temporal)}"
            f"|c{int(self.num_channels)}"
            f"|r{int(self.num_residual_blocks)}"
        )


# -----------------------------------------------------------------------------
# Config parsing
# -----------------------------------------------------------------------------
def _register_resolvers() -> None:
    try:
        OmegaConf.register_new_resolver("eval", eval, replace=True)
    except TypeError:
        # Older OmegaConf may not support replace=True.
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)


def _to_container(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _parse_raw_h(raw_h: Any) -> list[dict[str, Any]]:
    if raw_h is None:
        return []

    if isinstance(raw_h, str):
        text = raw_h.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = OmegaConf.to_container(OmegaConf.create(text), resolve=True)
        raw_h = parsed

    raw_h = _to_container(raw_h)

    if not isinstance(raw_h, list):
        raise TypeError("cfg.h must be a list of dictionaries.")

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_h):
        item = _to_container(item)
        if not isinstance(item, dict):
            raise TypeError(f"cfg.h[{idx}] must be a dictionary, got {type(item)}.")
        out.append(item)
    return out


def _required_int(item: dict[str, Any], key: str, idx: int) -> int:
    if key not in item or item[key] is None:
        raise ValueError(f"cfg.h[{idx}] is missing '{key}'.")
    return int(item[key])


def _guess_step_from_path(path: str) -> int:
    """
    Extract a training step from names like:
      black_00000.pt
      white_00100.pt
      black_epoch_xxx_e06880.pt
    If no number exists, returns -1.
    """
    stem = Path(path).stem
    nums = re.findall(r"\d+", stem)
    if not nums:
        return -1
    return int(nums[-1])


def _safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unnamed"


def _make_model_spec(
    *,
    item: dict[str, Any],
    idx: int,
    color: str,
    checkpoint: str,
    default_name: str | None = None,
) -> ModelSpec:
    temporal = _required_int(item, "temporal", idx)
    num_channels = _required_int(item, "num_channels", idx)
    num_residual_blocks = _required_int(item, "num_residual_blocks", idx)

    raw_name = item.get("name", item.get("label", default_name))
    if raw_name is None:
        raw_name = Path(checkpoint).stem

    base_name = _safe_name(str(raw_name))
    name = f"{color}_{base_name}"
    label = str(item.get("label", ""))

    return ModelSpec(
        name=name,
        color=color,
        checkpoint=str(Path(str(checkpoint)).expanduser()),
        temporal=temporal,
        num_channels=num_channels,
        num_residual_blocks=num_residual_blocks,
        label=label,
        step=_guess_step_from_path(str(checkpoint)),
    )


def parse_h_models(raw_h: Any) -> tuple[list[ModelSpec], list[ModelSpec]]:
    """
    Supports three h item formats.

    1) Single checkpoint:
       - checkpoint: /path/model.pt
         color: black | white | both    # default: both

    2) Role-separated checkpoints in one item:
       - black_checkpoint: /path/black.pt
         white_checkpoint: /path/white.pt

    3) Nested role checkpoints:
       - black:
           checkpoint: /path/black.pt
         white:
           checkpoint: /path/white.pt
    """
    h_items = _parse_raw_h(raw_h)
    black_models: list[ModelSpec] = []
    white_models: list[ModelSpec] = []

    for idx, item in enumerate(h_items):
        # Nested form:
        # h:
        #   - name: xxx
        #     black:
        #       checkpoint: ...
        #     white:
        #       checkpoint: ...
        nested_black = _to_container(item.get("black"))
        nested_white = _to_container(item.get("white"))

        if isinstance(nested_black, dict) and nested_black.get("checkpoint"):
            merged = dict(item)
            merged.update(nested_black)
            black_models.append(
                _make_model_spec(
                    item=merged,
                    idx=idx,
                    color="black",
                    checkpoint=str(nested_black["checkpoint"]),
                    default_name=item.get("name", f"h{idx}"),
                )
            )

        if isinstance(nested_white, dict) and nested_white.get("checkpoint"):
            merged = dict(item)
            merged.update(nested_white)
            white_models.append(
                _make_model_spec(
                    item=merged,
                    idx=idx,
                    color="white",
                    checkpoint=str(nested_white["checkpoint"]),
                    default_name=item.get("name", f"h{idx}"),
                )
            )

        # Flat role-separated form.
        if item.get("black_checkpoint"):
            black_models.append(
                _make_model_spec(
                    item=item,
                    idx=idx,
                    color="black",
                    checkpoint=str(item["black_checkpoint"]),
                    default_name=item.get("name", f"h{idx}"),
                )
            )

        if item.get("white_checkpoint"):
            white_models.append(
                _make_model_spec(
                    item=item,
                    idx=idx,
                    color="white",
                    checkpoint=str(item["white_checkpoint"]),
                    default_name=item.get("name", f"h{idx}"),
                )
            )

        # Single checkpoint form.
        if item.get("checkpoint"):
            color = str(item.get("color", "both")).strip().lower()
            checkpoint = str(item["checkpoint"])

            if color in {"black", "b"}:
                black_models.append(
                    _make_model_spec(
                        item=item,
                        idx=idx,
                        color="black",
                        checkpoint=checkpoint,
                        default_name=item.get("name", f"h{idx}"),
                    )
                )
            elif color in {"white", "w"}:
                white_models.append(
                    _make_model_spec(
                        item=item,
                        idx=idx,
                        color="white",
                        checkpoint=checkpoint,
                        default_name=item.get("name", f"h{idx}"),
                    )
                )
            elif color in {"both", "all", ""}:
                black_models.append(
                    _make_model_spec(
                        item=item,
                        idx=idx,
                        color="black",
                        checkpoint=checkpoint,
                        default_name=item.get("name", f"h{idx}"),
                    )
                )
                white_models.append(
                    _make_model_spec(
                        item=item,
                        idx=idx,
                        color="white",
                        checkpoint=checkpoint,
                        default_name=item.get("name", f"h{idx}"),
                    )
                )
            else:
                raise ValueError(
                    f"cfg.h[{idx}].color must be black, white, or both, got {color!r}."
                )

    # Keep deterministic order.
    black_models.sort(key=lambda m: (m.step, m.name, m.checkpoint))
    white_models.sort(key=lambda m: (m.step, m.name, m.checkpoint))

    if not black_models:
        raise ValueError("No black models found. Add color: black, black_checkpoint, or checkpoint with color: both.")
    if not white_models:
        raise ValueError("No white models found. Add color: white, white_checkpoint, or checkpoint with color: both.")

    return black_models, white_models


# -----------------------------------------------------------------------------
# Algo config and policy loading
# -----------------------------------------------------------------------------
def _load_base_algo_cfg_from_name_or_path(algo_cfg: str) -> DictConfig:
    """
    Load algo config if cfg.algo is not already provided by Hydra defaults.
    """
    candidate = Path(str(algo_cfg)).expanduser()
    if candidate.exists():
        return OmegaConf.load(candidate)

    config_root = Path(CONFIG_PATH)
    if not config_root.is_absolute():
        for root in (_THIS_FILE.parent.parent, Path.cwd()):
            maybe = root / config_root
            if maybe.exists():
                config_root = maybe
                break

    candidates = [
        config_root / "algo" / f"{algo_cfg}.yaml",
        config_root / f"{algo_cfg}.yaml",
        Path("cfg") / "algo" / f"{algo_cfg}.yaml",
        Path("cfg") / f"{algo_cfg}.yaml",
    ]
    for path in candidates:
        if path.exists():
            return OmegaConf.load(path)

    raise FileNotFoundError(
        f"Cannot find algo config {algo_cfg!r}. "
        f"Use defaults: - algo: ppo, or set algo_cfg to a real yaml path."
    )


def _get_base_algo_cfg(cfg: DictConfig) -> DictConfig:
    if "algo" in cfg and cfg.algo is not None:
        return OmegaConf.create(OmegaConf.to_container(cfg.algo, resolve=True))

    algo_cfg = str(cfg.get("algo_cfg", "ppo"))
    return _load_base_algo_cfg_from_name_or_path(algo_cfg)


def _algo_cfg_for_model(base_algo_cfg: DictConfig, model: ModelSpec) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(base_algo_cfg, resolve=True))
    if "name" not in cfg or cfg.name is None:
        cfg.name = "ppo"
    cfg.num_channels = int(model.num_channels)
    cfg.num_residual_blocks = int(model.num_residual_blocks)
    return cfg


class TemporalPolicyAdapter:
    """
    The evaluation environment is created with max temporal history among all
    checkpoints. Each checkpoint receives only the number of history channels it
    was trained with.

    temporal=1 receives channels:
      [current_player_board, opponent_board, last_1]

    temporal=6 receives channels:
      [current_player_board, opponent_board, last_1, ..., last_6]
    """

    def __init__(self, policy: Any, temporal: int):
        self.policy = policy
        self.temporal = int(temporal)
        if hasattr(self.policy, "eval"):
            self.policy.eval()

    def eval(self):
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        return self

    def train(self):
        # Evaluation only. Keep eval mode even if some wrapper calls train().
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        return self

    @torch.no_grad()
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        obs: torch.Tensor = tensordict.get("observation")
        action_mask: torch.Tensor = tensordict.get("action_mask")

        required_channels = 2 + int(self.temporal)
        if obs.shape[-3] < required_channels:
            raise RuntimeError(
                f"Observation has {obs.shape[-3]} channels, "
                f"but temporal={self.temporal} needs {required_channels} channels."
            )

        model_obs = obs[..., :required_channels, :, :]
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
    """Small LRU cache so large model pools do not keep every checkpoint in VRAM."""

    def __init__(
        self,
        *,
        base_algo_cfg: DictConfig,
        board_size: int,
        num_envs: int,
        device: str,
        cache_size: int,
        action_pruning: Any = None,
    ):
        self.base_algo_cfg = base_algo_cfg
        self.board_size = int(board_size)
        self.num_envs = int(num_envs)
        self.device = device
        self.cache_size = max(1, int(cache_size))
        self.action_pruning = action_pruning
        self._cache: OrderedDict[tuple[str, int, int, int], TemporalPolicyAdapter] = OrderedDict()
        self._spec_envs: dict[tuple[int, int, int], GomokuEnv] = {}

    def _get_spec_env(self, model: ModelSpec) -> GomokuEnv:
        key = (
            int(model.temporal),
            int(model.num_channels),
            int(model.num_residual_blocks),
        )
        if key not in self._spec_envs:
            self._spec_envs[key] = GomokuEnv(
                num_envs=self.num_envs,
                board_size=self.board_size,
                device=self.device,
                action_pruning=self.action_pruning,
                use_temporal_feature=True,
                temporal_num_steps=int(model.temporal),
                observation_mode="temporal_move_history",
            )
        return self._spec_envs[key]

    def get(self, model: ModelSpec) -> TemporalPolicyAdapter:
        checkpoint = str(Path(model.checkpoint).expanduser().resolve())
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

        key = (
            checkpoint,
            int(model.temporal),
            int(model.num_channels),
            int(model.num_residual_blocks),
        )

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        spec_env = self._get_spec_env(model)
        algo_cfg = _algo_cfg_for_model(self.base_algo_cfg, model)

        policy = get_pretrained_policy(
            name=str(algo_cfg.name),
            cfg=algo_cfg,
            action_spec=spec_env.action_spec,
            observation_spec=spec_env.observation_spec,
            checkpoint_path=checkpoint,
            device=self.device,
        )

        adapter = TemporalPolicyAdapter(policy=policy, temporal=int(model.temporal))
        self._cache[key] = adapter

        while len(self._cache) > self.cache_size:
            old_key, old_adapter = self._cache.popitem(last=False)
            del old_adapter
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            print(
                f"[CACHE] evicted {Path(old_key[0]).name} "
                f"t={old_key[1]} c={old_key[2]} r={old_key[3]}"
            )

        return adapter


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def _interaction_type(name: str) -> InteractionType:
    name = str(name).lower().strip()
    if name == "random":
        return InteractionType.RANDOM
    if name == "mode":
        return InteractionType.MODE
    raise ValueError(f"Unknown interaction type: {name!r}. Use 'random' or 'mode'.")


@torch.no_grad()
def eval_black_score(
    *,
    env: GomokuEnv,
    black_policy: TemporalPolicyAdapter,
    white_policy: TemporalPolicyAdapter,
    num_repeats: int,
    interaction: str,
) -> float:
    """
    Return black's average score:
      1.0 if black wins
      0.0 if white wins
      0.5 if no winner is produced within board_size * board_size moves
    """
    scores: list[float] = []
    interaction_value = _interaction_type(interaction)

    black_policy.eval()
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

                done = tensordict.get("done").reshape(env.num_envs)
                newly_done = done & (~finished)
                if newly_done.any():
                    black_win = tensordict.get(("stats", "black_win")).float().reshape(env.num_envs)
                    black_score[newly_done] = black_win[newly_done]

                finished |= newly_done
                if bool(finished.all().item()):
                    break

            scores.append(float(black_score.mean().item()))
            env.reset()

    return float(np.mean(scores))


# -----------------------------------------------------------------------------
# Elo fitting: same model and default parameters as elo_eval.py
# -----------------------------------------------------------------------------
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
    centered_skill = np.asarray(centered_skill_logit, dtype=np.float64)
    return centered_skill * (400.0 / math.log(10.0)) + float(average_rating)


def fit_role_mle_with_alpha(
    black_vs_white_payoff: np.ndarray,
    *,
    average_rating: float = 1200.0,
    games_per_pair: int = 1024,
    l2: float = 1e-4,
    max_iter: int = 5000,
    patience: int = 100,
    lr: float = 0.05,
    min_delta: float = 1e-10,
) -> tuple[
    np.ndarray,  # black_elo
    np.ndarray,  # white_elo
    np.ndarray,  # black_skill
    np.ndarray,  # white_skill
    np.ndarray,  # pred_black_score
    np.ndarray,  # pred_white_score
    np.ndarray,  # black_model_payoff
    np.ndarray,  # white_model_payoff
    dict[str, Any],  # diagnostics
]:
    """
    Role-separated Bradley-Terry / logistic Elo with one black advantage term.

    Input:
      raw[i, j] = black_model_i's score as BLACK against white_model_j as WHITE

    Model:
      P(black_i scores against white_j)
      = sigmoid(alpha + black_skill_i - white_skill_j)

    Identifiability:
      black_skill is centered to mean 0.
      white_skill is centered to mean 0.
    """
    raw = np.asarray(black_vs_white_payoff, dtype=np.float64)

    if raw.ndim != 2:
        raise ValueError(f"black_vs_white_payoff must be 2D, got shape={raw.shape}")
    if raw.size == 0:
        raise ValueError("empty payoff matrix")
    if np.isnan(raw).any():
        raise ValueError("black_vs_white_payoff contains NaN; finish evaluation first.")
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

    black_elo = _skill_to_elo_from_centered_skill(
        black_skill,
        average_rating=average_rating,
    )
    white_elo = _skill_to_elo_from_centered_skill(
        white_skill,
        average_rating=average_rating,
    )

    black_advantage_elo = alpha_logit * (400.0 / math.log(10.0))

    # These are fitted same-role relative payoff matrices, not directly-played games.
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

    diagnostics: dict[str, Any] = {
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
            "black_model_payoff and white_model_payoff are fitted same-role relative payoff matrices, not directly played games."
        ),
    }

    return (
        black_elo,
        white_elo,
        black_skill,
        white_skill,
        pred_black_score,
        pred_white_score,
        black_model_payoff,
        white_model_payoff,
        diagnostics,
    )


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    row_names: Sequence[str],
    col_names: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(col_names))
        for row_name, row in zip(row_names, matrix):
            writer.writerow([row_name] + [float(x) for x in row])


def _write_models_csv(path: Path, models: Sequence[ModelSpec]) -> None:
    fieldnames = [
        "index",
        "name",
        "color",
        "label",
        "temporal",
        "num_channels",
        "num_residual_blocks",
        "step",
        "checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, model in enumerate(models):
            row = asdict(model)
            writer.writerow(
                {
                    "index": idx,
                    "name": row["name"],
                    "color": row["color"],
                    "label": row["label"],
                    "temporal": row["temporal"],
                    "num_channels": row["num_channels"],
                    "num_residual_blocks": row["num_residual_blocks"],
                    "step": row["step"],
                    "checkpoint": row["checkpoint"],
                }
            )


def _write_elo_csv(
    path: Path,
    models: Sequence[ModelSpec],
    elo: np.ndarray,
    skill: np.ndarray,
) -> None:
    rows: list[dict[str, Any]] = []
    for idx, (model, elo_value, skill_value) in enumerate(zip(models, elo, skill)):
        rows.append(
            {
                "index": idx,
                "rank": 0,
                "name": model.name,
                "color": model.color,
                "label": model.label,
                "temporal": model.temporal,
                "num_channels": model.num_channels,
                "num_residual_blocks": model.num_residual_blocks,
                "step": model.step,
                "elo": float(elo_value),
                "skill_logit": float(skill_value),
                "checkpoint": model.checkpoint,
            }
        )

    rows.sort(key=lambda r: r["elo"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    fieldnames = [
        "rank",
        "index",
        "name",
        "color",
        "label",
        "temporal",
        "num_channels",
        "num_residual_blocks",
        "step",
        "elo",
        "skill_logit",
        "checkpoint",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_cache(
    path: Path,
    payoff: np.ndarray,
    black_models: Sequence[ModelSpec],
    white_models: Sequence[ModelSpec],
) -> None:
    np.savez_compressed(
        path,
        payoff=payoff,
        black_names=np.array([m.name for m in black_models], dtype=object),
        white_names=np.array([m.name for m in white_models], dtype=object),
        black_checkpoints=np.array([str(Path(m.checkpoint).expanduser()) for m in black_models], dtype=object),
        white_checkpoints=np.array([str(Path(m.checkpoint).expanduser()) for m in white_models], dtype=object),
    )


def _load_cache_if_compatible(
    path: Path,
    black_models: Sequence[ModelSpec],
    white_models: Sequence[ModelSpec],
) -> np.ndarray | None:
    if not path.exists():
        return None

    try:
        data = np.load(path, allow_pickle=True)
        payoff = np.asarray(data["payoff"], dtype=np.float64)

        black_names = [str(x) for x in data["black_names"].tolist()]
        white_names = [str(x) for x in data["white_names"].tolist()]
        black_checkpoints = [str(x) for x in data["black_checkpoints"].tolist()]
        white_checkpoints = [str(x) for x in data["white_checkpoints"].tolist()]

        cur_black_names = [m.name for m in black_models]
        cur_white_names = [m.name for m in white_models]
        cur_black_checkpoints = [str(Path(m.checkpoint).expanduser()) for m in black_models]
        cur_white_checkpoints = [str(Path(m.checkpoint).expanduser()) for m in white_models]

        if (
            payoff.shape == (len(black_models), len(white_models))
            and black_names == cur_black_names
            and white_names == cur_white_names
            and black_checkpoints == cur_black_checkpoints
            and white_checkpoints == cur_white_checkpoints
        ):
            return payoff

        print("[CACHE] existing payoff_cache.npz is incompatible; ignoring it.")
        return None

    except Exception as exc:
        print(f"[CACHE] failed to load cache {path}: {exc}; ignoring it.")
        return None


def _print_top(title: str, models: Sequence[ModelSpec], elo: np.ndarray, top_k: int = 10) -> None:
    print(f"\n[{title}]")
    order = np.argsort(-elo)
    for rank, idx in enumerate(order[:top_k], start=1):
        model = models[int(idx)]
        print(
            f"{rank:>2}. elo={elo[int(idx)]:8.2f} "
            f"step={model.step:>8} "
            f"name={model.name} "
            f"path={model.checkpoint}"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="eval")
def main(cfg: DictConfig) -> None:
    _register_resolvers()
    OmegaConf.resolve(cfg)

    seed = int(cfg.get("seed", 12345))
    torch.manual_seed(seed)
    np.random.seed(seed)

    board_size = int(cfg.get("board_size", 15))
    num_envs = int(cfg.get("num_envs", 512))
    num_repeats = int(cfg.get("num_repeats", cfg.get("n", 1)))
    device = str(cfg.get("device", "cuda"))
    interaction = str(cfg.get("interaction", "random"))
    output_dir = Path(str(cfg.get("output_dir", "eval_outputs"))).expanduser()
    resume = bool(cfg.get("resume", True))
    cache_size = int(cfg.get("cache_size", 4))
    action_pruning = cfg.get("action_pruning", None)

    average_rating = float(cfg.get("average_rating", 1200.0))
    elo_l2 = float(cfg.get("elo_l2", 1e-4))
    elo_max_iter = int(cfg.get("elo_max_iter", 5000))
    elo_patience = int(cfg.get("elo_patience", 100))
    elo_lr = float(cfg.get("elo_lr", 0.05))
    elo_min_delta = float(cfg.get("elo_min_delta", 1e-10))

    black_models, white_models = parse_h_models(cfg.get("h", None))
    max_temporal = max([m.temporal for m in black_models + white_models])

    _ensure_dir(output_dir)

    print("========== Eval config ==========")
    print(f"seed={seed}")
    print(f"board_size={board_size}")
    print(f"num_envs={num_envs}")
    print(f"num_repeats={num_repeats}")
    print(f"games_per_pair={num_envs * num_repeats}")
    print(f"device={device}")
    print(f"interaction={interaction}")
    print(f"max_temporal={max_temporal}")
    print(f"black_models={len(black_models)}")
    print(f"white_models={len(white_models)}")
    print(f"output_dir={output_dir}")
    print(
        "elo params: "
        f"average_rating={average_rating}, "
        f"l2={elo_l2}, "
        f"max_iter={elo_max_iter}, "
        f"patience={elo_patience}, "
        f"lr={elo_lr}, "
        f"min_delta={elo_min_delta}"
    )

    _write_models_csv(output_dir / "black_models.csv", black_models)
    _write_models_csv(output_dir / "white_models.csv", white_models)

    base_algo_cfg = _get_base_algo_cfg(cfg)

    env = GomokuEnv(
        num_envs=num_envs,
        board_size=board_size,
        device=device,
        action_pruning=action_pruning,
        use_temporal_feature=True,
        temporal_num_steps=int(max_temporal),
        observation_mode="temporal_move_history",
    )

    policy_bank = PolicyBank(
        base_algo_cfg=base_algo_cfg,
        board_size=board_size,
        num_envs=num_envs,
        device=device,
        cache_size=cache_size,
        action_pruning=action_pruning,
    )

    cache_path = output_dir / "payoff_cache.npz"
    if resume:
        cached = _load_cache_if_compatible(cache_path, black_models, white_models)
    else:
        cached = None

    if cached is None:
        payoff = np.full((len(black_models), len(white_models)), np.nan, dtype=np.float64)
    else:
        payoff = cached.copy()
        done_count = int(np.isfinite(payoff).sum())
        total_count = int(payoff.size)
        print(f"[CACHE] loaded {done_count}/{total_count} evaluated pairs from {cache_path}")

    black_names = [m.name for m in black_models]
    white_names = [m.name for m in white_models]

    total_pairs = len(black_models) * len(white_models)
    pair_idx = 0

    for i, black_model in enumerate(black_models):
        black_policy = policy_bank.get(black_model)

        for j, white_model in enumerate(white_models):
            pair_idx += 1

            if np.isfinite(payoff[i, j]):
                print(
                    f"[SKIP {pair_idx}/{total_pairs}] "
                    f"{black_model.name} vs {white_model.name}: "
                    f"black_score={payoff[i, j]:.6f}"
                )
                continue

            white_policy = policy_bank.get(white_model)

            print(
                f"[EVAL {pair_idx}/{total_pairs}] "
                f"BLACK={black_model.name} "
                f"WHITE={white_model.name}"
            )

            score = eval_black_score(
                env=env,
                black_policy=black_policy,
                white_policy=white_policy,
                num_repeats=num_repeats,
                interaction=interaction,
            )

            payoff[i, j] = float(score)

            print(
                f"[RESULT {pair_idx}/{total_pairs}] "
                f"BLACK={black_model.name} "
                f"WHITE={white_model.name} "
                f"black_score={score:.6f}"
            )

            _save_cache(cache_path, payoff, black_models, white_models)
            _write_matrix_csv(
                output_dir / "black_vs_white_payoff.csv",
                payoff,
                black_names,
                white_names,
            )

    if np.isnan(payoff).any():
        missing = int(np.isnan(payoff).sum())
        raise RuntimeError(f"Evaluation is incomplete: {missing} payoff entries are still NaN.")

    _write_matrix_csv(
        output_dir / "black_vs_white_payoff.csv",
        payoff,
        black_names,
        white_names,
    )

    (
        black_elo,
        white_elo,
        black_skill,
        white_skill,
        pred_black_score,
        pred_white_score,
        black_model_payoff,
        white_model_payoff,
        diagnostics,
    ) = fit_role_mle_with_alpha(
        payoff,
        average_rating=average_rating,
        games_per_pair=int(num_envs * num_repeats),
        l2=elo_l2,
        max_iter=elo_max_iter,
        patience=elo_patience,
        lr=elo_lr,
        min_delta=elo_min_delta,
    )

    diagnostics.update(
        {
            "seed": seed,
            "board_size": board_size,
            "num_envs": num_envs,
            "num_repeats": num_repeats,
            "device": device,
            "interaction": interaction,
            "num_black_models": len(black_models),
            "num_white_models": len(white_models),
            "black_models": [asdict(m) for m in black_models],
            "white_models": [asdict(m) for m in white_models],
        }
    )

    _write_matrix_csv(
        output_dir / "predicted_black_scores.csv",
        pred_black_score,
        black_names,
        white_names,
    )
    _write_matrix_csv(
        output_dir / "predicted_white_scores.csv",
        pred_white_score,
        white_names,
        black_names,
    )
    _write_matrix_csv(
        output_dir / "black_model_payoff_for_elo.csv",
        black_model_payoff,
        black_names,
        black_names,
    )
    _write_matrix_csv(
        output_dir / "white_model_payoff_for_elo.csv",
        white_model_payoff,
        white_names,
        white_names,
    )

    _write_elo_csv(output_dir / "black_elo_ratings.csv", black_models, black_elo, black_skill)
    _write_elo_csv(output_dir / "white_elo_ratings.csv", white_models, white_elo, white_skill)

    with (output_dir / "elo_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    _save_cache(cache_path, payoff, black_models, white_models)

    print("\n========== Raw black-vs-white payoff ==========")
    print(payoff)

    print("\n========== Elo diagnostics ==========")
    print(json.dumps(
        {
            "alpha_logit": diagnostics["alpha_logit"],
            "black_advantage_elo": diagnostics["black_advantage_elo"],
            "rmse": diagnostics["rmse"],
            "nll_per_game_without_l2": diagnostics["nll_per_game_without_l2"],
            "iterations_run": diagnostics["iterations_run"],
            "best_iter": diagnostics["best_iter"],
        },
        ensure_ascii=False,
        indent=2,
    ))

    _print_top("Top black Elo", black_models, black_elo)
    _print_top("Top white Elo", white_models, white_elo)

    print("\n========== Output files ==========")
    for name in [
        "black_models.csv",
        "white_models.csv",
        "black_vs_white_payoff.csv",
        "predicted_black_scores.csv",
        "predicted_white_scores.csv",
        "black_elo_ratings.csv",
        "white_elo_ratings.csv",
        "black_model_payoff_for_elo.csv",
        "white_model_payoff_for_elo.csv",
        "elo_diagnostics.json",
        "payoff_cache.npz",
    ]:
        print(output_dir / name)


if __name__ == "__main__":
    main()
