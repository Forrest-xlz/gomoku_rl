#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Role-separated Elo evaluation for manually organized Gomoku model folders.

This script is meant for offline evaluation of model checkpoints that you choose
explicitly. It does not use the training-time PSRO/InRL model pool metadata.

Expected new layout:

    <f>/
      black/                         # also tolerates: balck/
        <h_folder>/
          black_00000.pt             # also tolerates: balck_00000.pt
          black_00100.pt
      white/
        <h_folder>/
          white_00000.pt
          white_00100.pt

The folders listed in config.h define both where to find checkpoints and which
network architecture must be used to load them.

Example config:

    f: /root/autodl-tmp/eval_models
    h:
      - folder: temporal1_c64_r4
        temporal: 1
        num_channels: 64
        num_residual_blocks: 4
      - folder: temporal6_c64_r4
        temporal: 6
        num_channels: 64
        num_residual_blocks: 4

    board_size: 15
    num_envs: 512
    num_repeats: 1
    device: cuda
    algo_cfg: ppo
    output_dir: elo_eval_outputs
    interaction: random             # random or mode
    resume: true
    average_rating: 1200

    # Optional checkpoint filters.
    step_min: null
    step_max: null
    step_mod: null
    limit_per_h: null

    # Optional MLE optimizer settings.
    elo_l2: 1.0e-4
    elo_max_iter: 5000
    elo_patience: 100
    elo_lr: 0.05
    elo_min_delta: 1.0e-10

CLI examples:

    python scripts/elo_eval.py --config cfg/elo_eval.yaml

    python scripts/elo_eval.py \
      --f /root/autodl-tmp/eval_models \
      --h '[{"folder":"temporal1_c64_r4","temporal":1,"num_channels":64,"num_residual_blocks":4}]'

Outputs:

    output_dir/
      black_models.csv
      white_models.csv
      black_vs_white_payoff.csv
      predicted_black_scores.csv
      black_elo_ratings.csv
      white_elo_ratings.csv
      black_model_payoff_for_elo.csv
      white_model_payoff_for_elo.csv
      elo_diagnostics.json
      black_elo_curve.png
      white_elo_curve.png
      black_vs_white_payoff_heatmap.png
      payoff_cache.npz
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
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
class HSpec:
    """One manually configured architecture/folder spec."""

    folder: str
    temporal: int
    num_channels: int
    num_residual_blocks: int
    label: str = ""

    @property
    def display_label(self) -> str:
        if self.label:
            return self.label
        return (
            f"{self.folder}"
            f"|t{int(self.temporal)}"
            f"|c{int(self.num_channels)}"
            f"|r{int(self.num_residual_blocks)}"
        )


@dataclass(frozen=True)
class RoleModel:
    """One checkpoint in either the black role pool or white role pool."""

    name: str
    color: str
    h_folder: str
    h_label: str
    temporal: int
    num_channels: int
    num_residual_blocks: int
    step: int
    path: str


@dataclass
class EvalConfig:
    f: str
    h: list[HSpec]
    board_size: int = 15
    num_envs: int = 512
    num_repeats: int = 1
    device: str = "cuda"
    algo_cfg: str = "ppo"
    output_dir: str = "elo_eval_outputs"
    average_rating: float = 1200.0
    cache_size: int = 4
    interaction: str = "random"
    resume: bool = True
    step_min: int | None = None
    step_max: int | None = None
    step_mod: int | None = None
    limit_per_h: int | None = None
    elo_l2: float = 1e-4
    elo_max_iter: int = 5000
    elo_patience: int = 100
    elo_lr: float = 0.05
    elo_min_delta: float = 1e-10


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------


def _none_if_string_null(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _parse_h_specs(raw_h: Any) -> list[HSpec]:
    if raw_h is None:
        return []
    if isinstance(raw_h, str):
        text = raw_h.strip()
        if not text:
            return []
        try:
            raw_h = json.loads(text)
        except json.JSONDecodeError:
            raw_h = OmegaConf.to_container(OmegaConf.create(text), resolve=True)
    if isinstance(raw_h, DictConfig):
        raw_h = OmegaConf.to_container(raw_h, resolve=True)
    if not isinstance(raw_h, list):
        raise TypeError("config.h must be a list of dictionaries.")

    out: list[HSpec] = []
    for idx, item in enumerate(raw_h):
        if isinstance(item, DictConfig):
            item = OmegaConf.to_container(item, resolve=True)
        if not isinstance(item, dict):
            raise TypeError(f"config.h[{idx}] must be a dictionary, got {type(item)}")

        folder = item.get("folder", item.get("name", item.get("folder_name", None)))
        if folder is None:
            raise ValueError(f"config.h[{idx}] is missing 'folder' or 'name'.")

        temporal = item.get("temporal", item.get("temporal_steps", None))
        if temporal is None:
            raise ValueError(f"config.h[{idx}] is missing 'temporal'.")

        num_channels = item.get("num_channels", None)
        if num_channels is None:
            raise ValueError(f"config.h[{idx}] is missing 'num_channels'.")

        num_residual_blocks = item.get("num_residual_blocks", None)
        if num_residual_blocks is None:
            raise ValueError(f"config.h[{idx}] is missing 'num_residual_blocks'.")

        out.append(
            HSpec(
                folder=str(folder),
                temporal=int(temporal),
                num_channels=int(num_channels),
                num_residual_blocks=int(num_residual_blocks),
                label=str(item.get("label", "")),
            )
        )
    return out


def _load_config(args: argparse.Namespace) -> EvalConfig:
    base: dict[str, Any] = {
        "f": None,
        "h": [],
        "board_size": 15,
        "num_envs": 512,
        "num_repeats": 1,
        "device": "cuda",
        "algo_cfg": "ppo",
        "output_dir": "elo_eval_outputs",
        "average_rating": 1200.0,
        "cache_size": 4,
        "interaction": "random",
        "resume": True,
        "step_min": None,
        "step_max": None,
        "step_mod": None,
        "limit_per_h": None,
        "elo_l2": 1e-4,
        "elo_max_iter": 5000,
        "elo_patience": 100,
        "elo_lr": 0.05,
        "elo_min_delta": 1e-10,
    }

    if args.config is not None:
        cfg_path = Path(args.config).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"config file not found: {cfg_path}")
        loaded = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        if not isinstance(loaded, dict):
            raise TypeError(f"config file must contain a dictionary: {cfg_path}")
        base.update(loaded)

    # CLI overrides. Use None as "not provided".
    cli_overrides = {
        "f": args.f,
        "h": args.h,
        "board_size": args.board_size,
        "num_envs": args.num_envs,
        "num_repeats": args.num_repeats,
        "device": args.device,
        "algo_cfg": args.algo_cfg,
        "output_dir": args.output_dir,
        "average_rating": args.average_rating,
        "cache_size": args.cache_size,
        "interaction": args.interaction,
        "resume": args.resume,
        "step_min": args.step_min,
        "step_max": args.step_max,
        "step_mod": args.step_mod,
        "limit_per_h": args.limit_per_h,
        "elo_l2": args.elo_l2,
        "elo_max_iter": args.elo_max_iter,
        "elo_patience": args.elo_patience,
        "elo_lr": args.elo_lr,
        "elo_min_delta": args.elo_min_delta,
    }
    for key, value in cli_overrides.items():
        value = _none_if_string_null(value)
        if value is not None:
            base[key] = value

    model_root = base.get("f", base.get("model_root", None))
    if model_root is None:
        raise ValueError("Missing model root. Set config.f or pass --f.")

    h_specs = _parse_h_specs(base.get("h"))
    if not h_specs:
        raise ValueError("Missing architecture folder list. Set config.h or pass --h.")

    return EvalConfig(
        f=str(model_root),
        h=h_specs,
        board_size=int(base["board_size"]),
        num_envs=int(base["num_envs"]),
        num_repeats=int(base["num_repeats"]),
        device=str(base["device"]),
        algo_cfg=str(base["algo_cfg"]),
        output_dir=str(base["output_dir"]),
        average_rating=float(base["average_rating"]),
        cache_size=int(base["cache_size"]),
        interaction=str(base["interaction"]),
        resume=bool(base["resume"]),
        step_min=None if base.get("step_min") is None else int(base["step_min"]),
        step_max=None if base.get("step_max") is None else int(base["step_max"]),
        step_mod=None if base.get("step_mod") is None else int(base["step_mod"]),
        limit_per_h=None if base.get("limit_per_h") is None else int(base["limit_per_h"]),
        elo_l2=float(base["elo_l2"]),
        elo_max_iter=int(base["elo_max_iter"]),
        elo_patience=int(base["elo_patience"]),
        elo_lr=float(base["elo_lr"]),
        elo_min_delta=float(base["elo_min_delta"]),
    )


# -----------------------------------------------------------------------------
# Model discovery
# -----------------------------------------------------------------------------


_CKPT_RE = re.compile(r"^(?P<color>black|balck|white)_(?P<step>\d+)\.pt$", re.IGNORECASE)


def _resolve_existing_dir(candidates: Sequence[Path]) -> Path | None:
    for p in candidates:
        if p.exists() and p.is_dir():
            return p.resolve()
    return None


def _role_h_folder(model_root: Path, role: str, h_folder: str) -> Path | None:
    """Find one configured h folder for black/white checkpoints.

    New preferred layout:
        root/black/h_folder
        root/white/h_folder

    Backward-compatible layouts tolerated:
        root/balck/h_folder
        root/h_folder/black
        root/h_folder/balck
        root/h_folder/white
    """

    role = role.lower()
    if role == "black":
        candidates = [
            model_root / "black" / h_folder,
            model_root / "balck" / h_folder,
            model_root / h_folder / "black",
            model_root / h_folder / "balck",
        ]
    elif role == "white":
        candidates = [
            model_root / "white" / h_folder,
            model_root / h_folder / "white",
        ]
    else:
        raise ValueError(f"unknown role: {role}")
    return _resolve_existing_dir(candidates)


def scan_color_models(folder: Path, expected_color: str) -> dict[int, Path]:
    """Return {training_step: checkpoint_path} for one configured h folder."""

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
    limit_per_h: int | None,
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

    if limit_per_h is not None and limit_per_h > 0 and len(filtered) > limit_per_h:
        # Keep a roughly uniform sample, including endpoints.
        indices = np.linspace(0, len(filtered) - 1, limit_per_h)
        keep = sorted({int(round(x)) for x in indices})
        filtered = [filtered[i] for i in keep]
    return filtered


def _safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unnamed"


def discover_role_models(
    model_root: Path,
    h_specs: Sequence[HSpec],
    step_min: int | None = None,
    step_max: int | None = None,
    step_mod: int | None = None,
    limit_per_h: int | None = None,
) -> tuple[list[RoleModel], list[RoleModel]]:
    black_pool: list[RoleModel] = []
    white_pool: list[RoleModel] = []

    for spec in h_specs:
        black_dir = _role_h_folder(model_root, "black", spec.folder)
        white_dir = _role_h_folder(model_root, "white", spec.folder)

        if black_dir is None:
            print(f"[WARN] h={spec.folder}: black folder not found under {model_root}")
            black_models: dict[int, Path] = {}
        else:
            black_models = scan_color_models(black_dir, "black")

        if white_dir is None:
            print(f"[WARN] h={spec.folder}: white folder not found under {model_root}")
            white_models: dict[int, Path] = {}
        else:
            white_models = scan_color_models(white_dir, "white")

        if not black_models:
            print(f"[WARN] h={spec.folder}: no black checkpoints found in {black_dir}")
        if not white_models:
            print(f"[WARN] h={spec.folder}: no white checkpoints found in {white_dir}")

        black_steps = _filter_steps(
            sorted(black_models.keys()),
            step_min=step_min,
            step_max=step_max,
            step_mod=step_mod,
            limit_per_h=limit_per_h,
        )
        white_steps = _filter_steps(
            sorted(white_models.keys()),
            step_min=step_min,
            step_max=step_max,
            step_mod=step_mod,
            limit_per_h=limit_per_h,
        )

        print(
            f"[DISCOVER] h={spec.folder} temporal={spec.temporal} "
            f"channels={spec.num_channels} blocks={spec.num_residual_blocks} "
            f"black={len(black_steps)} white={len(white_steps)}"
        )

        safe_h = _safe_name(spec.folder)
        for step in black_steps:
            black_pool.append(
                RoleModel(
                    name=(
                        f"black_{safe_h}"
                        f"_t{int(spec.temporal)}"
                        f"_c{int(spec.num_channels)}"
                        f"_r{int(spec.num_residual_blocks)}"
                        f"_s{int(step):05d}"
                    ),
                    color="black",
                    h_folder=spec.folder,
                    h_label=spec.display_label,
                    temporal=int(spec.temporal),
                    num_channels=int(spec.num_channels),
                    num_residual_blocks=int(spec.num_residual_blocks),
                    step=int(step),
                    path=str(black_models[step]),
                )
            )
        for step in white_steps:
            white_pool.append(
                RoleModel(
                    name=(
                        f"white_{safe_h}"
                        f"_t{int(spec.temporal)}"
                        f"_c{int(spec.num_channels)}"
                        f"_r{int(spec.num_residual_blocks)}"
                        f"_s{int(step):05d}"
                    ),
                    color="white",
                    h_folder=spec.folder,
                    h_label=spec.display_label,
                    temporal=int(spec.temporal),
                    num_channels=int(spec.num_channels),
                    num_residual_blocks=int(spec.num_residual_blocks),
                    step=int(step),
                    path=str(white_models[step]),
                )
            )

    black_pool.sort(key=lambda m: (m.h_folder, m.temporal, m.num_channels, m.num_residual_blocks, m.step, m.name))
    white_pool.sort(key=lambda m: (m.h_folder, m.temporal, m.num_channels, m.num_residual_blocks, m.step, m.name))
    return black_pool, white_pool


# -----------------------------------------------------------------------------
# Policy adapter and cache
# -----------------------------------------------------------------------------


def _load_base_algo_cfg(algo_cfg: str) -> DictConfig:
    """Load base algorithm config by name or file path."""

    candidate = Path(algo_cfg).expanduser()
    if candidate.exists():
        return OmegaConf.load(candidate)

    config_root = Path(CONFIG_PATH)
    if not config_root.is_absolute():
        # CONFIG_PATH is usually a path-like value inside the package. If it is
        # relative, resolve it from project root.
        for root in (_THIS_FILE.parent.parent, Path.cwd()):
            maybe = root / config_root
            if maybe.exists():
                config_root = maybe
                break

    for p in [
        config_root / "algo" / f"{algo_cfg}.yaml",
        config_root / f"{algo_cfg}.yaml",
        Path("cfg") / "algo" / f"{algo_cfg}.yaml",
    ]:
        if p.exists():
            return OmegaConf.load(p)

    raise FileNotFoundError(
        f"Cannot find algo config '{algo_cfg}'. Pass a real YAML path or use an existing cfg/algo/*.yaml name."
    )


def _algo_cfg_for_model(base_algo_cfg: DictConfig, model: RoleModel) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(base_algo_cfg, resolve=True))
    cfg.num_channels = int(model.num_channels)
    cfg.num_residual_blocks = int(model.num_residual_blocks)
    return cfg


class TemporalPolicyAdapter:
    """Wrap a policy trained with temporal=n and slice eval observation channels.

    The evaluation environment is created with max temporal history. A temporal=1
    checkpoint only receives [current_player_board, opponent_board, last_1],
    while temporal=6 receives [current_player_board, opponent_board, last_1..last_6].
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
        # These policies are used for evaluation only; keep them in eval mode.
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        return self

    @torch.no_grad()
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        obs: torch.Tensor = tensordict.get("observation")
        action_mask: torch.Tensor = tensordict.get("action_mask")
        required_channels = 2 + self.temporal
        if obs.shape[-3] < required_channels:
            raise RuntimeError(
                f"Observation has {obs.shape[-3]} channels, but temporal={self.temporal} "
                f"policy needs {required_channels} channels."
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
    """LRU cache for checkpoint policies.

    Each architecture folder can have different temporal/channel/block settings,
    so the cache key includes both the path and the architecture parameters.
    """

    def __init__(
        self,
        base_algo_cfg: DictConfig,
        board_size: int,
        num_envs: int,
        device: str,
        cache_size: int,
    ):
        self.base_algo_cfg = base_algo_cfg
        self.board_size = int(board_size)
        self.num_envs = int(num_envs)
        self.device = device
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[tuple[str, int, int, int], TemporalPolicyAdapter] = OrderedDict()
        self._spec_envs: dict[tuple[int, int, int], GomokuEnv] = {}

    def _get_spec_env(self, temporal: int, num_channels: int, num_residual_blocks: int) -> GomokuEnv:
        key = (int(temporal), int(num_channels), int(num_residual_blocks))
        if key not in self._spec_envs:
            self._spec_envs[key] = GomokuEnv(
                num_envs=self.num_envs,
                board_size=self.board_size,
                device=self.device,
                use_temporal_feature=True,
                temporal_num_steps=int(temporal),
                observation_mode="temporal_move_history",
            )
        return self._spec_envs[key]

    def get(self, model: RoleModel) -> TemporalPolicyAdapter:
        key = (
            str(Path(model.path).resolve()),
            int(model.temporal),
            int(model.num_channels),
            int(model.num_residual_blocks),
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        spec_env = self._get_spec_env(
            temporal=model.temporal,
            num_channels=model.num_channels,
            num_residual_blocks=model.num_residual_blocks,
        )
        algo_cfg = _algo_cfg_for_model(self.base_algo_cfg, model)
        policy = get_pretrained_policy(
            name=str(algo_cfg.name),
            cfg=algo_cfg,
            action_spec=spec_env.action_spec,
            observation_spec=spec_env.observation_spec,
            checkpoint_path=key[0],
            device=self.device,
        )
        adapter = TemporalPolicyAdapter(policy=policy, temporal=model.temporal)
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
# Evaluation and Elo fitting
# -----------------------------------------------------------------------------


def _interaction_type(name: str) -> InteractionType:
    name = name.lower().strip()
    if name == "random":
        return InteractionType.RANDOM
    if name == "mode":
        return InteractionType.MODE
    raise ValueError(f"Unknown interaction type: {name}. Use 'random' or 'mode'.")


@torch.no_grad()
def eval_black_score(
    env: GomokuEnv,
    black_policy: TemporalPolicyAdapter,
    white_policy: TemporalPolicyAdapter,
    num_repeats: int,
    interaction: str,
) -> float:
    """Return black's average score.

    score = 1.0 if black wins, 0.0 if white wins, 0.5 if no winner is produced
    within board_size * board_size moves.
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
                    black_win = tensordict["stats", "black_win"].float().reshape(env.num_envs)
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
    """Convert centered logit-scale skill to Elo points."""

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
    dict[str, Any],
]:
    """Fit joint role-separated Bradley-Terry / logistic Elo with alpha.

    Input:
        raw[i, j] = black_model_i's score as BLACK against white_model_j as WHITE

    Model:
        P(black_i scores against white_j) = sigmoid(alpha + black_skill_i - white_skill_j)

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
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (bce * n).sum() / n.sum().clamp_min(1.0)
        if l2 > 0:
            loss = loss + float(l2) * (
                black_skill.square().mean() + white_skill.square().mean() + alpha.square()
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
    # they visualize fitted relative skill within each role.
    black_model_payoff = _sigmoid(black_skill[:, None] - black_skill[None, :])
    white_model_payoff = _sigmoid(white_skill[:, None] - white_skill[None, :])
    np.fill_diagonal(black_model_payoff, 0.5)
    np.fill_diagonal(white_model_payoff, 0.5)

    eps = 1e-12
    pred_clip = np.clip(pred_black_score, eps, 1.0 - eps)
    nll = -np.sum(games_np * (raw * np.log(pred_clip) + (1.0 - raw) * np.log(1.0 - pred_clip)))
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


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def write_models_csv(path: Path, models: Sequence[RoleModel]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "color",
                "h_folder",
                "h_label",
                "temporal",
                "num_channels",
                "num_residual_blocks",
                "step",
                "path",
            ],
        )
        writer.writeheader()
        for m in models:
            writer.writerow(asdict(m))


def write_matrix_csv(path: Path, row_names: Sequence[str], col_names: Sequence[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(col_names))
        for name, row in zip(row_names, matrix):
            writer.writerow([name] + ["" if np.isnan(x) else f"{float(x):.8f}" for x in row])


def write_elo_csv(path: Path, models: Sequence[RoleModel], elos: np.ndarray, skills: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "color",
                "h_folder",
                "h_label",
                "temporal",
                "num_channels",
                "num_residual_blocks",
                "step",
                "elo",
                "skill_logit_centered",
                "path",
            ],
        )
        writer.writeheader()
        for model, elo, skill in zip(models, elos, skills):
            row = asdict(model)
            row["elo"] = f"{float(elo):.6f}"
            row["skill_logit_centered"] = f"{float(skill):.10f}"
            writer.writerow(row)


def _names(models: Sequence[RoleModel]) -> list[str]:
    return [m.name for m in models]


def _load_payoff_cache(cache_path: Path, black_models: Sequence[RoleModel], white_models: Sequence[RoleModel]) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        old_black = [str(x) for x in data["black_names"].tolist()]
        old_white = [str(x) for x in data["white_names"].tolist()]
        matrix = np.asarray(data["payoff"], dtype=np.float64)
    except Exception as exc:
        print(f"[WARN] failed to load payoff cache {cache_path}: {exc}")
        return None

    if old_black != _names(black_models) or old_white != _names(white_models):
        print("[WARN] payoff cache exists but model list changed; ignoring cache.")
        return None
    if matrix.shape != (len(black_models), len(white_models)):
        print("[WARN] payoff cache shape mismatch; ignoring cache.")
        return None
    return matrix


def _save_payoff_cache(cache_path: Path, black_models: Sequence[RoleModel], white_models: Sequence[RoleModel], payoff: np.ndarray) -> None:
    np.savez_compressed(
        cache_path,
        black_names=np.asarray(_names(black_models), dtype=object),
        white_names=np.asarray(_names(white_models), dtype=object),
        payoff=np.asarray(payoff, dtype=np.float64),
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_role_elo_curve(path: Path, title: str, models: Sequence[RoleModel], elos: np.ndarray) -> None:
    groups: dict[str, list[tuple[int, float, RoleModel]]] = defaultdict(list)
    for model, elo in zip(models, elos):
        groups[model.h_label].append((int(model.step), float(elo), model))

    plt.figure(figsize=(12, 7))
    for label, rows in sorted(groups.items(), key=lambda kv: kv[0]):
        rows = sorted(rows, key=lambda x: x[0])
        xs = [x[0] for x in rows]
        ys = [x[1] for x in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=label)

    plt.title(title)
    plt.xlabel("training step")
    plt.ylabel("Elo")
    plt.grid(True, alpha=0.3)
    if len(groups) <= 20:
        plt.legend(fontsize=8)
    else:
        plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_payoff_heatmap(path: Path, matrix: np.ndarray, row_names: Sequence[str], col_names: Sequence[str]) -> None:
    plt.figure(figsize=(max(8, min(20, len(col_names) * 0.35)), max(6, min(20, len(row_names) * 0.35))))
    plt.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="black score")
    plt.title("Black-vs-white payoff matrix")
    plt.xlabel("white model")
    plt.ylabel("black model")

    def _tick_positions(n: int, max_ticks: int = 25) -> list[int]:
        if n <= max_ticks:
            return list(range(n))
        return sorted({int(round(x)) for x in np.linspace(0, n - 1, max_ticks)})

    x_ticks = _tick_positions(len(col_names))
    y_ticks = _tick_positions(len(row_names))
    plt.xticks(x_ticks, [col_names[i] for i in x_ticks], rotation=90, fontsize=6)
    plt.yticks(y_ticks, [row_names[i] for i in y_ticks], fontsize=6)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------


def run_eval(cfg: EvalConfig) -> None:
    model_root = Path(cfg.f).expanduser().resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"model root f does not exist: {model_root}")

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG] f={model_root}")
    print(f"[CONFIG] output_dir={output_dir}")
    print("[CONFIG] h=")
    for spec in cfg.h:
        print(
            f"  - folder={spec.folder} temporal={spec.temporal} "
            f"num_channels={spec.num_channels} num_residual_blocks={spec.num_residual_blocks}"
        )

    black_models, white_models = discover_role_models(
        model_root=model_root,
        h_specs=cfg.h,
        step_min=cfg.step_min,
        step_max=cfg.step_max,
        step_mod=cfg.step_mod,
        limit_per_h=cfg.limit_per_h,
    )
    if not black_models:
        raise RuntimeError("No black checkpoints discovered.")
    if not white_models:
        raise RuntimeError("No white checkpoints discovered.")

    print(f"[INFO] discovered black models: {len(black_models)}")
    print(f"[INFO] discovered white models: {len(white_models)}")

    write_models_csv(output_dir / "black_models.csv", black_models)
    write_models_csv(output_dir / "white_models.csv", white_models)

    max_temporal = max([m.temporal for m in black_models] + [m.temporal for m in white_models])
    env = GomokuEnv(
        num_envs=int(cfg.num_envs),
        board_size=int(cfg.board_size),
        device=str(cfg.device),
        use_temporal_feature=True,
        temporal_num_steps=int(max_temporal),
        observation_mode="temporal_move_history",
    )

    base_algo_cfg = _load_base_algo_cfg(cfg.algo_cfg)
    bank = PolicyBank(
        base_algo_cfg=base_algo_cfg,
        board_size=int(cfg.board_size),
        num_envs=int(cfg.num_envs),
        device=str(cfg.device),
        cache_size=int(cfg.cache_size),
    )

    cache_path = output_dir / "payoff_cache.npz"
    payoff = None
    if cfg.resume:
        payoff = _load_payoff_cache(cache_path, black_models, white_models)
        if payoff is not None:
            finished = int(np.isfinite(payoff).sum())
            total = payoff.size
            print(f"[RESUME] loaded payoff cache: {finished}/{total} pairs finished")

    if payoff is None:
        payoff = np.full((len(black_models), len(white_models)), np.nan, dtype=np.float64)

    total_pairs = len(black_models) * len(white_models)
    pair_index = 0
    for i, black_model in enumerate(black_models):
        for j, white_model in enumerate(white_models):
            pair_index += 1
            if np.isfinite(payoff[i, j]):
                continue
            print(
                f"[EVAL] {pair_index}/{total_pairs} "
                f"black={black_model.name} vs white={white_model.name}"
            )
            black_policy = bank.get(black_model)
            white_policy = bank.get(white_model)
            score = eval_black_score(
                env=env,
                black_policy=black_policy,
                white_policy=white_policy,
                num_repeats=int(cfg.num_repeats),
                interaction=str(cfg.interaction),
            )
            payoff[i, j] = float(score)
            print(f"[RESULT] black_score={score:.6f}")

            _save_payoff_cache(cache_path, black_models, white_models, payoff)
            write_matrix_csv(output_dir / "black_vs_white_payoff.csv", _names(black_models), _names(white_models), payoff)

    if np.isnan(payoff).any():
        raise RuntimeError("payoff matrix still contains NaN values after evaluation.")

    games_per_pair = int(cfg.num_envs) * int(cfg.num_repeats)
    (
        black_elo,
        white_elo,
        black_model_payoff,
        white_model_payoff,
        black_skill,
        white_skill,
        pred_black_score,
        pred_white_score,
        diagnostics,
    ) = fit_role_mle_with_alpha(
        payoff,
        average_rating=float(cfg.average_rating),
        games_per_pair=games_per_pair,
        l2=float(cfg.elo_l2),
        max_iter=int(cfg.elo_max_iter),
        patience=int(cfg.elo_patience),
        lr=float(cfg.elo_lr),
        min_delta=float(cfg.elo_min_delta),
    )

    diagnostics.update(
        {
            "config": {
                "f": str(model_root),
                "h": [asdict(x) for x in cfg.h],
                "board_size": int(cfg.board_size),
                "num_envs": int(cfg.num_envs),
                "num_repeats": int(cfg.num_repeats),
                "device": str(cfg.device),
                "algo_cfg": str(cfg.algo_cfg),
                "interaction": str(cfg.interaction),
                "cache_size": int(cfg.cache_size),
                "step_min": cfg.step_min,
                "step_max": cfg.step_max,
                "step_mod": cfg.step_mod,
                "limit_per_h": cfg.limit_per_h,
            },
            "num_black_models": len(black_models),
            "num_white_models": len(white_models),
            "black_models": [asdict(x) for x in black_models],
            "white_models": [asdict(x) for x in white_models],
        }
    )

    write_matrix_csv(output_dir / "black_vs_white_payoff.csv", _names(black_models), _names(white_models), payoff)
    write_matrix_csv(output_dir / "predicted_black_scores.csv", _names(black_models), _names(white_models), pred_black_score)
    write_matrix_csv(output_dir / "predicted_white_scores.csv", _names(white_models), _names(black_models), pred_white_score)
    write_matrix_csv(output_dir / "black_model_payoff_for_elo.csv", _names(black_models), _names(black_models), black_model_payoff)
    write_matrix_csv(output_dir / "white_model_payoff_for_elo.csv", _names(white_models), _names(white_models), white_model_payoff)
    write_elo_csv(output_dir / "black_elo_ratings.csv", black_models, black_elo, black_skill)
    write_elo_csv(output_dir / "white_elo_ratings.csv", white_models, white_elo, white_skill)

    with (output_dir / "elo_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    plot_role_elo_curve(output_dir / "black_elo_curve.png", "Black role Elo", black_models, black_elo)
    plot_role_elo_curve(output_dir / "white_elo_curve.png", "White role Elo", white_models, white_elo)
    plot_payoff_heatmap(output_dir / "black_vs_white_payoff_heatmap.png", payoff, _names(black_models), _names(white_models))

    print("[DONE] Elo evaluation finished.")
    print(f"[DONE] output_dir={output_dir}")
    print(f"[DONE] black_advantage_elo={diagnostics['black_advantage_elo']:.3f}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Role-separated Elo evaluation for configured Gomoku model folders.")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config file. Recommended.")
    parser.add_argument("--f", "--model-root", dest="f", type=str, default=None, help="Root folder containing black/ and white/.")
    parser.add_argument(
        "--h",
        type=str,
        default=None,
        help=(
            "JSON/YAML list of architecture folder specs. Example: "
            "'[{\"folder\":\"temporal1_c64_r4\",\"temporal\":1,\"num_channels\":64,\"num_residual_blocks\":4}]'"
        ),
    )
    parser.add_argument("--board-size", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-repeats", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--algo-cfg", type=str, default=None, help="Algo config name such as ppo, or a YAML path.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--average-rating", type=float, default=None)
    parser.add_argument("--cache-size", type=int, default=None)
    parser.add_argument("--interaction", type=str, choices=["random", "mode"], default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--step-min", type=int, default=None)
    parser.add_argument("--step-max", type=int, default=None)
    parser.add_argument("--step-mod", type=int, default=None)
    parser.add_argument("--limit-per-h", type=int, default=None)
    parser.add_argument("--elo-l2", type=float, default=None)
    parser.add_argument("--elo-max-iter", type=int, default=None)
    parser.add_argument("--elo-patience", type=int, default=None)
    parser.add_argument("--elo-lr", type=float, default=None)
    parser.add_argument("--elo-min-delta", type=float, default=None)
    return parser


def main() -> None:
    warnings.filterwarnings("default")
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = _load_config(args)
    run_eval(cfg)


if __name__ == "__main__":
    main()
