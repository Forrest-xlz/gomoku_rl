"""
Dota-style self-play / PSRO training entry for Forrest-xlz/gomoku_rl.

This script keeps the original Independent-RL two-network style:
    current black policy + current white policy

Training:
- With probability p, collect current_black vs current_white self-play data.
- With probability 1-p, use role-separated historical pools:
    * sample a historical white policy to train current_black;
    * sample a historical black policy to train current_white.
- The training model pool keeps weights in folder f and metadata in JSON e.
- Black and white pools maintain independent quality scores. When current_black
  beats a sampled historical white, only that white entry is lowered; when
  current_white beats a sampled historical black, only that black entry is lowered.

Evaluation:
- Baseline evaluation is removed.
- Every elo_interval epochs, current black/white weights are added to a global
  elo_models folder.
- A single global black-vs-white payoff matrix is maintained in elo_models/payoff.json.
- Only missing black_i-vs-white_j entries are evaluated.
- Role-separated Elo is fitted by maximum likelihood with one shared alpha:
      P(black_i scores against white_j) = sigmoid(alpha + black_skill_i - white_skill_j)
  The MLE uses early stopping and logs role-separated black/white Elo plus alpha.

Put this file at:
    scripts/train_psro.py

Run example:
    python scripts/train_psro.py \
        p=0.8 l=0.01 m=10 \
        f=model_pool e=model_pool/pool_meta.json \
        elo_models=elo_models elo_interval=100
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from gomoku_rl import CONFIG_PATH
from gomoku_rl.collector import BlackPlayCollector, VersusPlayCollector, WhitePlayCollector
from gomoku_rl.runner.base import Runner
from gomoku_rl.policy import get_policy
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.utils.misc import add_prefix
from gomoku_rl.utils.wandb import init_wandb


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _now() -> float:
    return float(time.time())


def _sanitize_id(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_") or "run"


def _get_run_id(default_prefix: str = "run") -> str:
    if wandb.run is not None and getattr(wandb.run, "id", None):
        return _sanitize_id(str(wandb.run.id))
    return _sanitize_id(f"{default_prefix}_{time.strftime('%Y%m%d_%H%M%S')}")


def _atomic_json_dump(obj: Any, path: str | os.PathLike[str]) -> None:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)


# -----------------------------------------------------------------------------
# Training model pool: role-separated OpenAI-Five-style historical opponents
# -----------------------------------------------------------------------------


@dataclass
class RolePoolEntry:
    """One historical checkpoint for one role in the training pool.

    role="black" entries are historical black policies used as opponents for
    current_white. role="white" entries are historical white policies used as
    opponents for current_black.
    """

    id: str
    role: str
    epoch: int
    path: str
    quality: float
    created_at: float
    source: str = "train_psro"
    run_id: str = ""


class RoleSeparatedDiskPolicyPool:
    """On-disk role-separated policy pool.

    Metadata is kept in one JSON file, but black and white pools have independent
    entries and independent quality scores. This is important for Gomoku because
    black and white are not interchangeable roles.

    JSON layout:
        {
          "schema_version": 3,
          "black_entries": [...],
          "white_entries": [...]
        }

    Backward compatibility:
        If an older pair-level JSON contains "entries" with black_path and
        white_path, it is converted on load into one black entry and one white
        entry with the old pair quality copied to both roles.
    """

    SCHEMA_VERSION = 3
    ROLES = ("black", "white")

    def __init__(
        self,
        pool_dir: str | os.PathLike[str],
        meta_path: str | os.PathLike[str] | None,
        *,
        run_id: str,
        prevent_overwrite: bool = True,
    ) -> None:
        self.pool_dir = Path(pool_dir).expanduser().resolve()
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = (
            Path(meta_path).expanduser().resolve()
            if meta_path is not None and str(meta_path).strip()
            else self.pool_dir / "pool_meta.json"
        )
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = _sanitize_id(run_id)
        self.prevent_overwrite = bool(prevent_overwrite)
        self.black_entries: list[RolePoolEntry] = []
        self.white_entries: list[RolePoolEntry] = []
        self.extra: dict[str, Any] = {}
        self.load()

    def __len__(self) -> int:
        return self.total_size()

    def total_size(self) -> int:
        return len(self.black_entries) + len(self.white_entries)

    def role_size(self, role: str) -> int:
        return len(self._entries(role))

    def _entries(self, role: str) -> list[RolePoolEntry]:
        role = str(role).lower().strip()
        if role == "black":
            return self.black_entries
        if role == "white":
            return self.white_entries
        raise ValueError(f"unknown pool role: {role}")

    def _resolve_checkpoint_path(self, path: str) -> str:
        p = Path(path).expanduser()
        if p.is_absolute():
            return str(p.resolve())

        p_meta = (self.meta_path.parent / p).resolve()
        if p_meta.exists():
            return str(p_meta)
        return str(Path(to_absolute_path(str(p))).expanduser().resolve())

    def _load_role_entries(self, raw_entries: list[dict[str, Any]], role: str) -> list[RolePoolEntry]:
        out: list[RolePoolEntry] = []
        for item in raw_entries:
            out.append(
                RolePoolEntry(
                    id=str(item["id"]),
                    role=role,
                    epoch=int(item.get("epoch", -1)),
                    path=self._resolve_checkpoint_path(str(item.get("path", item.get(f"{role}_path", "")))),
                    quality=float(item.get("quality", 0.0)),
                    created_at=float(item.get("created_at", 0.0)),
                    source=str(item.get("source", "loaded")),
                    run_id=str(item.get("run_id", "")),
                )
            )
        return out

    def load(self) -> None:
        if not self.meta_path.is_file():
            self.black_entries = []
            self.white_entries = []
            self.extra = {}
            return

        with self.meta_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self.extra = {
            k: v
            for k, v in raw.items()
            if k not in {"schema_version", "entries", "black_entries", "white_entries"}
        }

        # New role-separated schema.
        if "black_entries" in raw or "white_entries" in raw:
            self.black_entries = self._load_role_entries(raw.get("black_entries", []), "black")
            self.white_entries = self._load_role_entries(raw.get("white_entries", []), "white")
            return

        # Backward compatibility with old pair-level pool_meta.json.
        black_entries: list[RolePoolEntry] = []
        white_entries: list[RolePoolEntry] = []
        for item in raw.get("entries", []):
            old_id = str(item["id"])
            epoch = int(item.get("epoch", -1))
            quality = float(item.get("quality", 0.0))
            created_at = float(item.get("created_at", 0.0))
            source = str(item.get("source", "loaded_pair_compat"))
            run_id = str(item.get("run_id", ""))
            black_entries.append(
                RolePoolEntry(
                    id=f"black_{old_id}",
                    role="black",
                    epoch=epoch,
                    path=self._resolve_checkpoint_path(str(item["black_path"])),
                    quality=quality,
                    created_at=created_at,
                    source=source,
                    run_id=run_id,
                )
            )
            white_entries.append(
                RolePoolEntry(
                    id=f"white_{old_id}",
                    role="white",
                    epoch=epoch,
                    path=self._resolve_checkpoint_path(str(item["white_path"])),
                    quality=quality,
                    created_at=created_at,
                    source=source,
                    run_id=run_id,
                )
            )
        self.black_entries = black_entries
        self.white_entries = white_entries

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "pool_dir": str(self.pool_dir),
            "meta_path": str(self.meta_path),
            "updated_at": _now(),
            "black_entries": [asdict(entry) for entry in self.black_entries],
            "white_entries": [asdict(entry) for entry in self.white_entries],
            **self.extra,
        }

    def save(self, path: str | os.PathLike[str] | None = None) -> None:
        out_path = Path(path).expanduser().resolve() if path is not None else self.meta_path
        _atomic_json_dump(self.to_dict(), out_path)

    def softmax_probs(self, role: str) -> list[float]:
        entries = self._entries(role)
        if not entries:
            return []
        qs = [float(e.quality) for e in entries]
        max_q = max(qs)
        exp_qs = [math.exp(q - max_q) for q in qs]
        denom = sum(exp_qs)
        if denom <= 0.0 or not math.isfinite(denom):
            return [1.0 / len(entries)] * len(entries)
        return [v / denom for v in exp_qs]

    def prob_variance(self, role: str) -> float:
        probs = self.softmax_probs(role)
        if not probs:
            return 0.0
        mean_p = 1.0 / len(probs)
        return float(sum((float(p) - mean_p) ** 2 for p in probs) / len(probs))

    def sample(self, role: str, rng: random.Random) -> tuple[int, RolePoolEntry, float]:
        entries = self._entries(role)
        if not entries:
            raise RuntimeError(f"Cannot sample from an empty {role} policy pool.")
        probs = self.softmax_probs(role)
        idx = rng.choices(range(len(entries)), weights=probs, k=1)[0]
        return idx, entries[idx], float(probs[idx])

    def _target_path(self, *, role: str, epoch: int, prefix: str) -> tuple[str, Path]:
        role = str(role).lower().strip()
        if role not in self.ROLES:
            raise ValueError(f"unknown role: {role}")
        safe_prefix = _sanitize_id(prefix)
        entry_id = f"{role}_{safe_prefix}_{self.run_id}_e{epoch:06d}"
        role_dir = self.pool_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = role_dir / f"{entry_id}.pt"
        return entry_id, ckpt_path

    def add_current_role(
        self,
        *,
        role: str,
        epoch: int,
        state_dict: dict[str, Any],
        prefix: str = "epoch",
        source: str = "train_psro",
    ) -> RolePoolEntry:
        entries = self._entries(role)
        entry_id, ckpt_path = self._target_path(role=role, epoch=epoch, prefix=prefix)
        if self.prevent_overwrite and ckpt_path.exists():
            raise FileExistsError(f"Refusing to overwrite {role} model-pool checkpoint: {ckpt_path}")

        torch.save(state_dict, ckpt_path)
        init_q = max((entry.quality for entry in entries), default=0.0)
        entry = RolePoolEntry(
            id=entry_id,
            role=str(role),
            epoch=int(epoch),
            path=str(ckpt_path.resolve()),
            quality=float(init_q),
            created_at=_now(),
            source=source,
            run_id=self.run_id,
        )
        entries.append(entry)
        self.save()
        return entry

    def add_current_pair(
        self,
        *,
        epoch: int,
        black_state_dict: dict[str, Any],
        white_state_dict: dict[str, Any],
        prefix: str = "epoch",
        source: str = "train_psro",
    ) -> tuple[RolePoolEntry, RolePoolEntry]:
        black_entry = self.add_current_role(
            role="black", epoch=epoch, state_dict=black_state_dict, prefix=prefix, source=source
        )
        white_entry = self.add_current_role(
            role="white", epoch=epoch, state_dict=white_state_dict, prefix=prefix, source=source
        )
        return black_entry, white_entry

    def ensure_non_empty(
        self,
        *,
        black_state_dict: dict[str, Any],
        white_state_dict: dict[str, Any],
    ) -> list[RolePoolEntry]:
        added: list[RolePoolEntry] = []
        if len(self.black_entries) == 0:
            added.append(
                self.add_current_role(
                    role="black",
                    epoch=0,
                    state_dict=black_state_dict,
                    prefix="initial",
                    source="initial_current",
                )
            )
        if len(self.white_entries) == 0:
            added.append(
                self.add_current_role(
                    role="white",
                    epoch=0,
                    state_dict=white_state_dict,
                    prefix="initial",
                    source="initial_current",
                )
            )
        return added

    def update_after_current_win_rate(
        self,
        *,
        role: str,
        index: int,
        sample_prob: float,
        pool_lr: float,
        current_win_rate: float,
        min_prob: float = 1e-8,
    ) -> float:
        entries = self._entries(role)
        if not (0 <= index < len(entries)):
            raise IndexError(f"{role} pool index out of range: {index}")
        if pool_lr <= 0.0:
            return 0.0

        win_rate = max(0.0, min(1.0, float(current_win_rate)))
        if win_rate <= 0.0:
            return 0.0

        n = max(1, len(entries))
        p_i = max(float(sample_prob), min_prob)
        delta = float(pool_lr) * win_rate / (n * p_i)
        entries[index].quality -= delta
        self.save()
        return delta


# -----------------------------------------------------------------------------
# Global Elo league: one payoff matrix, no fixed baselines
# -----------------------------------------------------------------------------


@dataclass
class EloEntry:
    """One black/white checkpoint pair used only for global Elo evaluation."""

    id: str
    epoch: int
    black_path: str
    white_path: str
    created_at: float
    source: str = "train_psro"
    run_id: str = ""
    parent_black_checkpoint: str = ""
    parent_white_checkpoint: str = ""


class GlobalEloLeague:
    """
    Global checkpoint league.

    Payoff convention:
        payoff[black_id][white_id] = black win rate when black_id.black plays
                                      against white_id.white.

    Elo convention:
        The real rectangular black-vs-white payoff matrix is fitted directly:
            P(black_i scores against white_j)
                = sigmoid(alpha + black_skill_i - white_skill_j)
        This yields role-separated black Elo, role-separated white Elo, and one
        shared black-first-move advantage alpha.
    """

    META_SCHEMA_VERSION = 1
    PAYOFF_SCHEMA_VERSION = 1

    def __init__(
        self,
        elo_dir: str | os.PathLike[str],
        *,
        run_id: str,
        prevent_overwrite: bool = True,
    ) -> None:
        self.elo_dir = Path(elo_dir).expanduser().resolve()
        self.elo_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.elo_dir / "elo_meta.json"
        self.payoff_path = self.elo_dir / "payoff.json"
        self.ratings_csv_path = self.elo_dir / "elo_ratings.csv"
        self.ratings_json_path = self.elo_dir / "elo_ratings.json"
        self.run_id = _sanitize_id(run_id)
        self.prevent_overwrite = bool(prevent_overwrite)
        self.entries: list[EloEntry] = []
        self.payoff: dict[str, dict[str, dict[str, Any]]] = {}
        self.load()

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_path(self, path: str) -> str:
        p = Path(path).expanduser()
        if p.is_absolute():
            return str(p.resolve())
        p_meta = (self.meta_path.parent / p).resolve()
        if p_meta.exists():
            return str(p_meta)
        return str(Path(to_absolute_path(str(p))).expanduser().resolve())

    def load(self) -> None:
        if self.meta_path.is_file():
            with self.meta_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            entries: list[EloEntry] = []
            for item in raw.get("entries", []):
                entries.append(
                    EloEntry(
                        id=str(item["id"]),
                        epoch=int(item.get("epoch", -1)),
                        black_path=self._resolve_path(str(item["black_path"])),
                        white_path=self._resolve_path(str(item["white_path"])),
                        created_at=float(item.get("created_at", 0.0)),
                        source=str(item.get("source", "loaded")),
                        run_id=str(item.get("run_id", "")),
                        parent_black_checkpoint=str(item.get("parent_black_checkpoint", "")),
                        parent_white_checkpoint=str(item.get("parent_white_checkpoint", "")),
                    )
                )
            self.entries = entries
        else:
            self.entries = []

        if self.payoff_path.is_file():
            with self.payoff_path.open("r", encoding="utf-8") as f:
                raw_payoff = json.load(f)
            self.payoff = raw_payoff.get("payoff", {})
        else:
            self.payoff = {}

    def save_meta(self) -> None:
        _atomic_json_dump(
            {
                "schema_version": self.META_SCHEMA_VERSION,
                "elo_dir": str(self.elo_dir),
                "updated_at": _now(),
                "entries": [asdict(entry) for entry in self.entries],
            },
            self.meta_path,
        )

    def save_payoff(self) -> None:
        _atomic_json_dump(
            {
                "schema_version": self.PAYOFF_SCHEMA_VERSION,
                "elo_dir": str(self.elo_dir),
                "updated_at": _now(),
                "payoff": self.payoff,
            },
            self.payoff_path,
        )

    def _target_paths(self, *, epoch: int, prefix: str) -> tuple[str, Path, Path]:
        safe_prefix = _sanitize_id(prefix)
        entry_id = f"{safe_prefix}_{self.run_id}_e{epoch:06d}"
        black_path = self.elo_dir / f"black_{entry_id}.pt"
        white_path = self.elo_dir / f"white_{entry_id}.pt"
        return entry_id, black_path, white_path

    def has_entry(self, entry_id: str) -> bool:
        return any(e.id == entry_id for e in self.entries)

    def add_current_pair(
        self,
        *,
        epoch: int,
        black_state_dict: dict[str, Any],
        white_state_dict: dict[str, Any],
        prefix: str = "elo",
        source: str = "train_psro",
        parent_black_checkpoint: str = "",
        parent_white_checkpoint: str = "",
    ) -> EloEntry:
        entry_id, black_path, white_path = self._target_paths(epoch=epoch, prefix=prefix)
        if self.has_entry(entry_id):
            # Idempotent behavior when the same epoch is retried in the same run.
            for entry in self.entries:
                if entry.id == entry_id:
                    return entry
        if self.prevent_overwrite and (black_path.exists() or white_path.exists()):
            raise FileExistsError(
                f"Refusing to overwrite Elo checkpoints: {black_path}, {white_path}"
            )

        torch.save(black_state_dict, black_path)
        torch.save(white_state_dict, white_path)

        entry = EloEntry(
            id=entry_id,
            epoch=int(epoch),
            black_path=str(black_path.resolve()),
            white_path=str(white_path.resolve()),
            created_at=_now(),
            source=source,
            run_id=self.run_id,
            parent_black_checkpoint=str(parent_black_checkpoint or ""),
            parent_white_checkpoint=str(parent_white_checkpoint or ""),
        )
        self.entries.append(entry)
        self.save_meta()
        return entry

    def get_score(self, black_id: str, white_id: str) -> float | None:
        item = self.payoff.get(black_id, {}).get(white_id, None)
        if item is None:
            return None
        return float(item["black_win_rate"])

    def set_score(
        self,
        *,
        black_id: str,
        white_id: str,
        black_win_rate: float,
        eval_repeats: int,
    ) -> None:
        self.payoff.setdefault(black_id, {})[white_id] = {
            "black_win_rate": float(black_win_rate),
            "eval_repeats": int(eval_repeats),
            "updated_at": _now(),
        }
        self.save_payoff()

    def missing_pairs(self, *, include_self: bool = False) -> list[tuple[EloEntry, EloEntry]]:
        pairs: list[tuple[EloEntry, EloEntry]] = []
        for black_entry in self.entries:
            for white_entry in self.entries:
                if not include_self and black_entry.id == white_entry.id:
                    continue
                if self.get_score(black_entry.id, white_entry.id) is None:
                    pairs.append((black_entry, white_entry))
        return pairs

    def payoff_coverage(self, *, include_self: bool = False) -> float:
        n = len(self.entries)
        total = n * n if include_self else n * max(n - 1, 0)
        if total <= 0:
            return 1.0
        filled = 0
        for i in self.entries:
            for j in self.entries:
                if not include_self and i.id == j.id:
                    continue
                filled += int(self.get_score(i.id, j.id) is not None)
        return float(filled / total)

    def fit_role_elo_with_alpha(
        self,
        *,
        base_elo: float = 1200.0,
        elo_scale: float = 400.0,
        l2: float = 1e-4,
        iters: int = 5000,
        lr: float = 0.05,
        patience: int = 100,
        min_delta: float = 1e-10,
        games_per_pair: int = 1,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
        """
        Fit role-separated Elo with one shared black-first-move advantage alpha.

        Payoff convention:
            payoff[black_id][white_id] = black_id.black's score as black
                                      against white_id.white as white.

        MLE model:
            P(black_i scores against white_j)
                = sigmoid(alpha + black_skill_i - white_skill_j)

        Identifiability:
            black_skill is centered to mean 0.
            white_skill is centered to mean 0.

        Reported ratings:
            black_elo_i = base_elo + black_skill_i * elo_scale / ln(10)
            white_elo_j = base_elo + white_skill_j * elo_scale / ln(10)
            black_advantage_elo = alpha * elo_scale / ln(10)

        This matches the alpha-based payoff-matrix MLE style used by the
        standalone role-separated Elo script.
        """
        n = len(self.entries)
        if n == 0:
            empty_summary = {
                "num_observations": 0.0,
                "loss": 0.0,
                "alpha_logit": 0.0,
                "black_advantage_elo": 0.0,
            }
            return {"black": [], "white": [], "combined": []}, empty_summary

        obs_black_idx: list[int] = []
        obs_white_idx: list[int] = []
        obs_score: list[float] = []
        obs_weight: list[float] = []

        for i, black_entry in enumerate(self.entries):
            for j, white_entry in enumerate(self.entries):
                score = self.get_score(black_entry.id, white_entry.id)
                if score is None:
                    continue
                obs_black_idx.append(i)
                obs_white_idx.append(j)
                obs_score.append(max(0.0, min(1.0, float(score))))
                obs_weight.append(float(max(1, int(games_per_pair))))

        if not obs_score:
            black_rows: list[dict[str, Any]] = []
            white_rows: list[dict[str, Any]] = []
            combined_rows: list[dict[str, Any]] = []
            for rank, entry in enumerate(self.entries, start=1):
                black_rows.append(self._rating_row(entry, role="black", rating=base_elo, skill=0.0, rank=rank))
                white_rows.append(self._rating_row(entry, role="white", rating=base_elo, skill=0.0, rank=rank))
                combined_rows.append(
                    self._combined_rating_row(
                        entry,
                        black_elo=base_elo,
                        white_elo=base_elo,
                        mean_elo=base_elo,
                        black_rank=rank,
                        white_rank=rank,
                        combined_rank=rank,
                    )
                )
            summary = {
                "model": "sigmoid(alpha + black_skill_i - white_skill_j)",
                "num_observations": 0.0,
                "loss": 0.0,
                "alpha_logit": 0.0,
                "black_advantage_elo": 0.0,
                "iterations_run": 0.0,
                "best_iter": 0.0,
                "stopped_early": 0.0,
            }
            self._save_role_ratings(black_rows, white_rows, combined_rows, summary)
            return {"black": black_rows, "white": white_rows, "combined": combined_rows}, summary

        device = torch.device("cpu")
        idx_b = torch.tensor(obs_black_idx, dtype=torch.long, device=device)
        idx_w = torch.tensor(obs_white_idx, dtype=torch.long, device=device)
        y = torch.tensor(obs_score, dtype=torch.float64, device=device)
        weights = torch.tensor(obs_weight, dtype=torch.float64, device=device)

        black_raw = torch.zeros(n, dtype=torch.float64, requires_grad=True, device=device)
        white_raw = torch.zeros(n, dtype=torch.float64, requires_grad=True, device=device)
        alpha = torch.zeros((), dtype=torch.float64, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([black_raw, white_raw, alpha], lr=float(lr))

        def objective() -> torch.Tensor:
            black_skill = black_raw - black_raw.mean()
            white_skill = white_raw - white_raw.mean()
            logits = alpha + black_skill[idx_b] - white_skill[idx_w]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                y,
                reduction="none",
            )
            loss = (bce * weights).sum() / weights.sum().clamp_min(1.0)
            if l2 > 0:
                loss = loss + float(l2) * (
                    black_skill.square().mean()
                    + white_skill.square().mean()
                    + alpha.square()
                )
            return loss

        best_loss = float("inf")
        best_iter = 0
        best_state: dict[str, torch.Tensor] | None = None
        bad_count = 0
        last_loss = float("inf")
        max_iter = max(1, int(iters))
        patience_i = max(1, int(patience))
        min_delta_f = float(min_delta)

        for iteration in range(1, max_iter + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            optimizer.step()

            last_loss = float(loss.detach().cpu().item())
            if last_loss < best_loss - min_delta_f:
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

            if bad_count >= patience_i:
                break

        if best_state is not None:
            with torch.no_grad():
                black_raw.copy_(best_state["black_raw"])
                white_raw.copy_(best_state["white_raw"])
                alpha.copy_(best_state["alpha"])

        with torch.no_grad():
            black_skill_t = black_raw - black_raw.mean()
            white_skill_t = white_raw - white_raw.mean()
            alpha_logit = float(alpha.detach().cpu().item())
            logits = alpha + black_skill_t[idx_b] - white_skill_t[idx_w]
            pred = torch.sigmoid(logits).detach().cpu()
            y_cpu = y.detach().cpu()
            rmse = float(torch.sqrt(torch.mean((pred - y_cpu).square())).item())
            black_skill = [float(x) for x in black_skill_t.detach().cpu().tolist()]
            white_skill = [float(x) for x in white_skill_t.detach().cpu().tolist()]

        factor = float(elo_scale) / math.log(10.0)
        black_elo_values = [float(base_elo + factor * s) for s in black_skill]
        white_elo_values = [float(base_elo + factor * s) for s in white_skill]
        black_advantage_elo = float(alpha_logit * factor)

        black_rows = [
            self._rating_row(entry, role="black", rating=rating, skill=skill, rank=0)
            for entry, rating, skill in zip(self.entries, black_elo_values, black_skill)
        ]
        white_rows = [
            self._rating_row(entry, role="white", rating=rating, skill=skill, rank=0)
            for entry, rating, skill in zip(self.entries, white_elo_values, white_skill)
        ]
        black_rows.sort(key=lambda r: r["elo"], reverse=True)
        white_rows.sort(key=lambda r: r["elo"], reverse=True)
        for rank, row in enumerate(black_rows, start=1):
            row["rank"] = rank
        for rank, row in enumerate(white_rows, start=1):
            row["rank"] = rank

        black_rank_by_id = {row["id"]: int(row["rank"]) for row in black_rows}
        white_rank_by_id = {row["id"]: int(row["rank"]) for row in white_rows}
        combined_rows = []
        for entry, b_elo, w_elo in zip(self.entries, black_elo_values, white_elo_values):
            combined_rows.append(
                self._combined_rating_row(
                    entry,
                    black_elo=b_elo,
                    white_elo=w_elo,
                    mean_elo=0.5 * (b_elo + w_elo),
                    black_rank=black_rank_by_id[entry.id],
                    white_rank=white_rank_by_id[entry.id],
                    combined_rank=0,
                )
            )
        combined_rows.sort(key=lambda r: r["mean_elo"], reverse=True)
        for rank, row in enumerate(combined_rows, start=1):
            row["combined_rank"] = rank

        summary = {
            "model": "sigmoid(alpha + black_skill_i - white_skill_j)",
            "base_elo": float(base_elo),
            "elo_scale": float(elo_scale),
            "l2": float(l2),
            "lr": float(lr),
            "max_iter": float(max_iter),
            "patience": float(patience_i),
            "min_delta": float(min_delta_f),
            "games_per_pair": float(max(1, int(games_per_pair))),
            "num_models": float(n),
            "num_observations": float(len(obs_score)),
            "loss": float(best_loss),
            "last_loss": float(last_loss),
            "iterations_run": float(iteration),
            "best_iter": float(best_iter),
            "stopped_early": float(iteration < max_iter),
            "rmse": float(rmse),
            "alpha_logit": float(alpha_logit),
            "black_advantage_elo": float(black_advantage_elo),
            "black_elo_mean": float(sum(black_elo_values) / len(black_elo_values)),
            "white_elo_mean": float(sum(white_elo_values) / len(white_elo_values)),
            "payoff_coverage_including_missing_self_policy": float(self.payoff_coverage(include_self=True)),
            "note": (
                "Black and white ratings are fitted jointly from the real black-vs-white payoff matrix. "
                "Predicted black score uses black_elo - white_elo + black_advantage_elo."
            ),
        }
        self._save_role_ratings(black_rows, white_rows, combined_rows, summary)
        return {"black": black_rows, "white": white_rows, "combined": combined_rows}, summary

    def _rating_row(
        self,
        entry: EloEntry,
        *,
        role: str,
        rating: float,
        skill: float,
        rank: int,
    ) -> dict[str, Any]:
        return {
            "rank": int(rank),
            "id": entry.id,
            "role": role,
            "run_id": entry.run_id,
            "epoch": int(entry.epoch),
            "elo": float(rating),
            "mle_skill_logit": float(skill),
            "source": entry.source,
            "path": entry.black_path if role == "black" else entry.white_path,
            "black_path": entry.black_path,
            "white_path": entry.white_path,
        }

    def _combined_rating_row(
        self,
        entry: EloEntry,
        *,
        black_elo: float,
        white_elo: float,
        mean_elo: float,
        black_rank: int,
        white_rank: int,
        combined_rank: int,
    ) -> dict[str, Any]:
        return {
            "combined_rank": int(combined_rank),
            "id": entry.id,
            "run_id": entry.run_id,
            "epoch": int(entry.epoch),
            "black_elo": float(black_elo),
            "white_elo": float(white_elo),
            "mean_elo": float(mean_elo),
            "black_rank": int(black_rank),
            "white_rank": int(white_rank),
            "source": entry.source,
            "black_path": entry.black_path,
            "white_path": entry.white_path,
        }

    def _save_role_ratings(
        self,
        black_rows: list[dict[str, Any]],
        white_rows: list[dict[str, Any]],
        combined_rows: list[dict[str, Any]],
        summary: dict[str, float],
    ) -> None:
        _atomic_json_dump(
            {
                "updated_at": _now(),
                "summary": summary,
                "black_ratings": black_rows,
                "white_ratings": white_rows,
                "combined_ratings": combined_rows,
            },
            self.ratings_json_path,
        )

        def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

        write_csv(
            self.elo_dir / "black_elo_ratings.csv",
            black_rows,
            [
                "rank",
                "id",
                "role",
                "run_id",
                "epoch",
                "elo",
                "mle_skill_logit",
                "source",
                "path",
                "black_path",
                "white_path",
            ],
        )
        write_csv(
            self.elo_dir / "white_elo_ratings.csv",
            white_rows,
            [
                "rank",
                "id",
                "role",
                "run_id",
                "epoch",
                "elo",
                "mle_skill_logit",
                "source",
                "path",
                "black_path",
                "white_path",
            ],
        )
        write_csv(
            self.ratings_csv_path,
            combined_rows,
            [
                "combined_rank",
                "id",
                "run_id",
                "epoch",
                "black_elo",
                "white_elo",
                "mean_elo",
                "black_rank",
                "white_rank",
                "source",
                "black_path",
                "white_path",
            ],
        )


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


class DotaStylePSRORunner(Runner):
    """Independent RL runner with Dota/OpenAI-Five-style model pool and global Elo."""

    _MODE_BOTH = "both"
    _MODE_BLACK_ONLY = "black_only"
    _MODE_WHITE_ONLY = "white_only"

    def __init__(self, cfg: DictConfig) -> None:
        # Runner.__init__ in the original project eagerly builds eval_baseline_pool.
        # train_psro does not use the old fixed-baseline evaluation, so force this
        # field to be empty before calling super().__init__(cfg). This makes the
        # script robust even if the active Hydra config inherited old baseline
        # paths such as pretrained_models/15_15/ppo/0.pt.
        with open_dict(cfg):
            cfg.eval_baseline_pool = {"black_pool": [], "white_pool": []}

        super().__init__(cfg)

        self.log_interval = int(self.cfg.get("log_interval", 5))
        seed = self.cfg.get("seed", None)
        self._rng = random.Random(None if seed is None else int(seed))
        self.run_id = _sanitize_id(_get_run_id())

        self.self_play_prob = float(self.cfg.get("p", self.cfg.get("self_play_prob", 0.8)))
        self.pool_lr = float(self.cfg.get("l", self.cfg.get("pool_lr", 0.01)))
        self.add_to_pool_interval = int(
            self.cfg.get("m", self.cfg.get("add_to_pool_interval", 10))
        )
        pool_dir = self.cfg.get("f", self.cfg.get("model_pool_dir", "model_pool"))
        meta_path = self.cfg.get("e", self.cfg.get("model_pool_meta", None))

        if not (0.0 <= self.self_play_prob <= 1.0):
            raise ValueError(f"p/self_play_prob must be in [0, 1], got {self.self_play_prob}")
        if self.pool_lr < 0.0:
            raise ValueError(f"l/pool_lr must be >= 0, got {self.pool_lr}")

        self.pool = RoleSeparatedDiskPolicyPool(
            pool_dir=to_absolute_path(str(pool_dir)),
            meta_path=(
                to_absolute_path(str(meta_path))
                if meta_path is not None and str(meta_path).strip()
                else None
            ),
            run_id=self.run_id,
            prevent_overwrite=True,
        )
        logging.info(
            "Role-separated PSRO training pool loaded: black=%d, white=%d, total=%d, dir=%s, meta=%s",
            self.pool.role_size("black"),
            self.pool.role_size("white"),
            self.pool.total_size(),
            self.pool.pool_dir,
            self.pool.meta_path,
        )

        self.elo_interval = int(
            self.cfg.get("elo_interval", self.cfg.get("add_to_elo_interval", 100))
        )
        self.elo_eval_repeats = int(self.cfg.get("elo_eval_repeats", 1))
        self.elo_include_self = True
        self.elo_base = float(self.cfg.get("elo_base", 1200.0))
        self.elo_scale = float(self.cfg.get("elo_scale", 400.0))
        self.elo_mle_iters = int(self.cfg.get("elo_mle_iters", 1000))
        self.elo_mle_lr = float(self.cfg.get("elo_mle_lr", 0.05))
        self.elo_mle_patience = int(self.cfg.get("elo_mle_patience", 100))
        self.elo_l2 = float(self.cfg.get("elo_l2", 1e-4))
        elo_dir = self.cfg.get("elo_models", self.cfg.get("elo_models_dir", "elo_models"))
        self.elo_league = GlobalEloLeague(
            elo_dir=to_absolute_path(str(elo_dir)),
            run_id=self.run_id,
            prevent_overwrite=True,
        )
        logging.info(
            "Global Elo league loaded: size=%d, dir=%s",
            len(self.elo_league),
            self.elo_league.elo_dir,
        )

        self._self_collector = VersusPlayCollector(
            self.env,
            self.policy_black,
            self.policy_white,
            out_device=self.cfg.get("out_device", None),
            augment=self.cfg.get("augment", False),
        )

        added_pool_entries = self.pool.ensure_non_empty(
            black_state_dict=self.policy_black.state_dict(),
            white_state_dict=self.policy_white.state_dict(),
        )
        if added_pool_entries:
            logging.info(
                "Added initial current policies to empty role pool(s): %s",
                ", ".join(entry.id for entry in added_pool_entries),
            )

        if len(self.elo_league) == 0:
            entry = self.elo_league.add_current_pair(
                epoch=0,
                black_state_dict=self.policy_black.state_dict(),
                white_state_dict=self.policy_white.state_dict(),
                prefix="elo_init",
                source="pretrained_init" if self.cfg.get("black_checkpoint", None) else "initial_current",
                parent_black_checkpoint=str(self.cfg.get("black_checkpoint", "") or ""),
                parent_white_checkpoint=str(self.cfg.get("white_checkpoint", "") or ""),
            )
            logging.info("Added initial current policies to Elo league: %s", entry.id)

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

    # ------------------------------ learn helpers -----------------------------
    def _learn_black(self, data_black, info: dict[str, Any]) -> None:
        if data_black is None:
            info["policy_black/update_skipped_empty_data"] = 1.0
            return
        info.update(add_prefix(self.policy_black.learn(data_black.to_tensordict()), "policy_black/"))

    def _learn_white(self, data_white, info: dict[str, Any]) -> None:
        if data_white is None:
            info["policy_white/update_skipped_empty_data"] = 1.0
            return
        info.update(add_prefix(self.policy_white.learn(data_white.to_tensordict()), "policy_white/"))

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
            self._learn_black(data_black, info)
        if applied_mode in (self._MODE_BOTH, self._MODE_WHITE_ONLY):
            self._learn_white(data_white, info)


    def _build_current_algo_policy_from_checkpoint(self, checkpoint_path: str, *, tag: str):
        """Build a policy with the current training algo/env specs and load a checkpoint.

        This is intentionally different from Runner._build_baseline_policy_from_checkpoint(),
        which uses cfg.baseline and may create a legacy 3-channel baseline model.
        The PSRO pool and global Elo league store checkpoints produced by the current
        train_psro run, so they must be loaded with cfg.algo and self.env.observation_spec
        to preserve temporal-feature channel count, e.g. 2 + temporal_num_steps.
        """
        checkpoint_path = str(checkpoint_path)
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"{tag} not found: {checkpoint_path}")
        policy = get_policy(
            name=self.cfg.algo.name,
            cfg=self.cfg.algo,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )
        self._load_policy_checkpoint(policy, checkpoint_path, tag)
        policy.eval()
        return policy

    # ------------------------------ rollout modes -----------------------------
    def _rollout_current_self_play(self, applied_mode: str) -> dict[str, Any]:
        data_black, data_white, raw_info = self._self_collector.rollout(self.steps)
        info: dict[str, Any] = add_prefix(raw_info, "self_play/")
        info["train/source_self_play"] = 1.0
        info["fps"] = info.get("self_play/fps", 0.0)
        info.pop("self_play/fps", None)
        self._apply_learning(applied_mode, data_black, data_white, info)
        return info

    def _load_role_pool_policy(self, entry: RolePoolEntry):
        if not Path(entry.path).is_file():
            raise FileNotFoundError(f"pool {entry.role} checkpoint not found: {entry.path}")
        return self._build_current_algo_policy_from_checkpoint(
            entry.path, tag=f"pool_{entry.role}/{entry.id}"
        )

    def _rollout_against_pool(self, applied_mode: str, epoch: int) -> dict[str, Any]:
        # Role-separated pool:
        #   white pool entries are historical WHITE opponents for current_black.
        #   black pool entries are historical BLACK opponents for current_white.
        if self.pool.role_size("black") == 0 or self.pool.role_size("white") == 0:
            self.pool.ensure_non_empty(
                black_state_dict=self.policy_black.state_dict(),
                white_state_dict=self.policy_white.state_dict(),
            )

        info: dict[str, Any] = {"train/source_self_play": 0.0}
        fps_parts: list[float] = []
        data_black = None
        data_white = None
        sampled_policies: list[Any] = []

        # Train current black against a sampled historical white.
        if applied_mode in (self._MODE_BOTH, self._MODE_BLACK_ONLY):
            white_idx, white_entry, white_sample_prob = self.pool.sample("white", self._rng)
            hist_white = self._load_role_pool_policy(white_entry)
            sampled_policies.append(hist_white)

            info.update(
                {
                    "pool_white/sampled_index": float(white_idx),
                    "pool_white/sampled_epoch": float(white_entry.epoch),
                    "pool_white/sampled_relative_epoch_ratio": float((white_entry.epoch - (epoch + 1)) / max(epoch + 1, 1)),
                    "pool_white/sampled_prob": float(white_sample_prob),
                }
            )

            black_collector = BlackPlayCollector(
                self.env,
                policy_black=self.policy_black,
                policy_white=hist_white,
                out_device=self.cfg.get("out_device", None),
                augment=self.cfg.get("augment", False),
            )
            data_black, black_info = black_collector.rollout(self.steps)
            black_prefixed_info = add_prefix(black_info, "pool_black_train/")
            black_prefixed_info.pop("pool_black_train/white_win", None)
            info.update(black_prefixed_info)
            fps_parts.append(float(black_info.get("fps", 0.0)))

            current_black_win = float(black_info.get("black_win", 0.0))
            info["pool_white/current_black_win_rate"] = current_black_win
            white_delta = self.pool.update_after_current_win_rate(
                role="white",
                index=white_idx,
                sample_prob=white_sample_prob,
                pool_lr=self.pool_lr,
                current_win_rate=current_black_win,
            )
            info["pool_white/quality_delta"] = -float(white_delta)

        # Train current white against a sampled historical black.
        if applied_mode in (self._MODE_BOTH, self._MODE_WHITE_ONLY):
            black_idx, black_entry, black_sample_prob = self.pool.sample("black", self._rng)
            hist_black = self._load_role_pool_policy(black_entry)
            sampled_policies.append(hist_black)

            info.update(
                {
                    "pool_black/sampled_index": float(black_idx),
                    "pool_black/sampled_epoch": float(black_entry.epoch),
                    "pool_black/sampled_relative_epoch_ratio": float((black_entry.epoch - (epoch + 1)) / max(epoch + 1, 1)),
                    "pool_black/sampled_prob": float(black_sample_prob),
                }
            )

            white_collector = WhitePlayCollector(
                self.env,
                policy_black=hist_black,
                policy_white=self.policy_white,
                out_device=self.cfg.get("out_device", None),
                augment=self.cfg.get("augment", False),
            )
            data_white, white_info = white_collector.rollout(self.steps)
            white_prefixed_info = add_prefix(white_info, "pool_white_train/")
            white_prefixed_info.pop("pool_white_train/black_win", None)
            info.update(white_prefixed_info)
            fps_parts.append(float(white_info.get("fps", 0.0)))

            current_white_win = float(white_info.get("white_win", 0.0))
            info["pool_black/current_white_win_rate"] = current_white_win
            black_delta = self.pool.update_after_current_win_rate(
                role="black",
                index=black_idx,
                sample_prob=black_sample_prob,
                pool_lr=self.pool_lr,
                current_win_rate=current_white_win,
            )
            info["pool_black/quality_delta"] = -float(black_delta)

        info["fps"] = float(sum(fps_parts) / len(fps_parts)) if fps_parts else 0.0
        self._apply_learning(applied_mode, data_black, data_white, info)

        for policy in sampled_policies:
            del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Temporary role-pool collectors share self.env with the persistent
        # self-play collector, so reset self-play collector state after any pool
        # rollout to avoid stale action masks and illegal moves.
        self._self_collector.reset()
        return info

    def _maybe_add_current_to_pool(self, epoch: int, info: dict[str, Any]) -> None:
        if self.add_to_pool_interval <= 0:
            return
        if (epoch + 1) % self.add_to_pool_interval != 0:
            return

        black_entry, white_entry = self.pool.add_current_pair(
            epoch=epoch + 1,
            black_state_dict=self.policy_black.state_dict(),
            white_state_dict=self.policy_white.state_dict(),
            prefix="epoch",
        )
        logging.info(
            "Added current policies to role-separated training pool: black=%s white=%s",
            black_entry.id,
            white_entry.id,
        )

    def _epoch(self, epoch: int) -> dict[str, Any]:
        applied_mode = self._get_applied_mode_for_current_epoch()
        use_self_play = (self._rng.random() < self.self_play_prob) or len(self.pool) == 0

        if use_self_play:
            info = self._rollout_current_self_play(applied_mode)
        else:
            info = self._rollout_against_pool(applied_mode, epoch)

        info.update(
            {
                "psro/self_play_prob": self.self_play_prob,
                "psro/pool_lr": self.pool_lr,
                "psro/add_to_pool_interval": float(self.add_to_pool_interval),
                "pool_black/size": float(self.pool.role_size("black")),
                "pool_black/prob_variance": self.pool.prob_variance("black"),
                "pool_white/size": float(self.pool.role_size("white")),
                "pool_white/prob_variance": self.pool.prob_variance("white"),
                "elo_eval/interval": float(self.elo_interval),
                "elo_eval/num_models": float(len(self.elo_league)),
                "elo_eval/payoff_coverage": self.elo_league.payoff_coverage(
                    include_self=True
                ),
                "balance/enabled": float(self.balance_enabled),
                "balance/lower": self.balance_lower,
                "balance/upper": self.balance_upper,
                "balance/ema_alpha": self.balance_ema_alpha,
            }
        )
        info.update(self._bias_mode_to_flags(self.current_bias_mode))
        info.update(self._phase_flags())
        info.update(self._ema_trigger_flags())
        if self.black_vs_white_ema is not None:
            info["eval/black_vs_white_ema"] = self.black_vs_white_ema

        self._maybe_add_current_to_pool(epoch, info)

        if epoch % 50 == 0 and epoch != 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return info

    # ------------------------------ global Elo eval ---------------------------
    def _load_elo_policy_pair(self, entry: EloEntry, *, load_black: bool, load_white: bool):
        black_policy = None
        white_policy = None
        if load_black:
            if not Path(entry.black_path).is_file():
                raise FileNotFoundError(f"Elo black checkpoint not found: {entry.black_path}")
            black_policy = self._build_current_algo_policy_from_checkpoint(
                entry.black_path, tag=f"elo_black/{entry.id}"
            )
        if load_white:
            if not Path(entry.white_path).is_file():
                raise FileNotFoundError(f"Elo white checkpoint not found: {entry.white_path}")
            white_policy = self._build_current_algo_policy_from_checkpoint(
                entry.white_path, tag=f"elo_white/{entry.id}"
            )
        return black_policy, white_policy

    def _evaluate_one_payoff_pair(
        self,
        black_entry: EloEntry,
        white_entry: EloEntry,
    ) -> float:
        black_policy, _ = self._load_elo_policy_pair(
            black_entry, load_black=True, load_white=False
        )
        _, white_policy = self._load_elo_policy_pair(
            white_entry, load_black=False, load_white=True
        )
        assert black_policy is not None
        assert white_policy is not None
        try:
            score = float(
                eval_win_rate(
                    self.eval_env,
                    player_black=black_policy,
                    player_white=white_policy,
                    n=max(1, self.elo_eval_repeats),
                )
            )
        finally:
            del black_policy, white_policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return score

    def _maybe_update_global_elo(self, epoch: int, info: dict[str, Any]) -> None:
        if self.elo_interval <= 0:
            return
        if (epoch + 1) % self.elo_interval != 0:
            return

        entry = self.elo_league.add_current_pair(
            epoch=epoch + 1,
            black_state_dict=self.policy_black.state_dict(),
            white_state_dict=self.policy_white.state_dict(),
            prefix="elo",
            source="train_psro",
            parent_black_checkpoint=str(self.cfg.get("black_checkpoint", "") or ""),
            parent_white_checkpoint=str(self.cfg.get("white_checkpoint", "") or ""),
        )
        info["elo_eval/added_epoch"] = float(entry.epoch)
        info["elo_eval/num_models"] = float(len(self.elo_league))

        missing = self.elo_league.missing_pairs(include_self=True)

        evaluated = 0
        start = time.perf_counter()
        for black_entry, white_entry in missing:
            score = self._evaluate_one_payoff_pair(black_entry, white_entry)
            self.elo_league.set_score(
                black_id=black_entry.id,
                white_id=white_entry.id,
                black_win_rate=score,
                eval_repeats=max(1, self.elo_eval_repeats),
            )
            evaluated += 1

        ratings, fit_summary = self.elo_league.fit_role_elo_with_alpha(
            base_elo=self.elo_base,
            elo_scale=self.elo_scale,
            l2=self.elo_l2,
            iters=self.elo_mle_iters,
            lr=self.elo_mle_lr,
            patience=self.elo_mle_patience,
            min_delta=0.0,
            games_per_pair=max(
                1,
                int(getattr(self.eval_env, "num_envs", 1)) * max(1, self.elo_eval_repeats),
            ),
        )
        elapsed = time.perf_counter() - start

        black_rows = ratings.get("black", [])
        white_rows = ratings.get("white", [])
        current_black_row = next((row for row in black_rows if row["id"] == entry.id), None)
        current_white_row = next((row for row in white_rows if row["id"] == entry.id), None)
        best_black_row = black_rows[0] if black_rows else None
        best_white_row = white_rows[0] if white_rows else None

        current_black_elo = (
            float(current_black_row["elo"]) if current_black_row is not None else float(self.elo_base)
        )
        current_white_elo = (
            float(current_white_row["elo"]) if current_white_row is not None else float(self.elo_base)
        )
        current_black_rank = (
            float(current_black_row["rank"]) if current_black_row is not None else 1.0
        )
        current_white_rank = (
            float(current_white_row["rank"]) if current_white_row is not None else 1.0
        )
        best_black_elo = (
            float(best_black_row["elo"]) if best_black_row is not None else current_black_elo
        )
        best_white_elo = (
            float(best_white_row["elo"]) if best_white_row is not None else current_white_elo
        )

        info.update(
            {
                "elo_eval/current_black": current_black_elo,
                "elo_eval/current_white": current_white_elo,
                "elo_eval/current_black_rank": current_black_rank,
                "elo_eval/current_white_rank": current_white_rank,
                "elo_eval/best_black": best_black_elo,
                "elo_eval/best_white": best_white_elo,
                "elo_eval/num_models": float(len(self.elo_league)),
                "elo_eval/missing_pairs_evaluated": float(evaluated),
                "elo_eval/payoff_coverage": self.elo_league.payoff_coverage(include_self=True),
                "elo_eval/eval_seconds": float(elapsed),
                "elo_state/black_advantage": float(fit_summary.get("black_advantage_elo", 0.0)),
                "elo_state/alpha_logit": float(fit_summary.get("alpha_logit", 0.0)),
                "elo_state/fit_observations": float(fit_summary.get("num_observations", 0.0)),
                "elo_state/fit_loss": float(fit_summary.get("loss", 0.0)),
                "elo_state/fit_rmse": float(fit_summary.get("rmse", 0.0)),
                "elo_state/fit_iterations": float(fit_summary.get("iterations_run", 0.0)),
                "elo_state/fit_best_iter": float(fit_summary.get("best_iter", 0.0)),
                "elo_state/fit_stopped_early": float(fit_summary.get("stopped_early", 0.0)),
            }
        )
        logging.info(
            "Global role Elo updated at epoch=%d: current=%s black=%.2f white=%.2f black_rank=%s white_rank=%s alpha=%.2f models=%d evaluated_pairs=%d",
            epoch + 1,
            entry.id,
            current_black_elo,
            current_white_elo,
            int(current_black_rank),
            int(current_white_rank),
            float(fit_summary.get("black_advantage_elo", 0.0)),
            len(self.elo_league),
            evaluated,
        )

    # ------------------------------- logging ---------------------------------
    def _format_eval_summary(self, info: dict[str, Any]) -> str:
        parts = [
            f"Black vs White:{info['eval/black_vs_white'] * 100.0:.2f}%",
            f"EMA:{info['eval/black_vs_white_ema'] * 100.0:.2f}%",
            f"pool_black:{self.pool.role_size('black')} pool_white:{self.pool.role_size('white')}",
            f"elo_models:{len(self.elo_league)}",
            f"p_self:{self.self_play_prob:.3f}",
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
                    "pool_black/size": float(self.pool.role_size("black")),
                    "pool_black/prob_variance": self.pool.prob_variance("black"),
                    "pool_white/size": float(self.pool.role_size("white")),
                    "pool_white/prob_variance": self.pool.prob_variance("white"),
                    "elo_eval/num_models": float(len(self.elo_league)),
                    "elo_eval/payoff_coverage": self.elo_league.payoff_coverage(
                        include_self=True
                    ),
                }
            )
            info.update(self._ema_trigger_flags())
            info.update(self._bias_mode_to_flags(self.current_bias_mode))
            info.update(self._phase_flags())

        self._maybe_update_global_elo(epoch, info)

        if epoch % self.log_interval == 0 or "elo_eval/current_black" in info:
            if "eval/black_vs_white" not in info:
                # Minimal eval if this epoch is an Elo epoch but not a log_interval epoch.
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

    # ---------------------------- metadata snapshots --------------------------
    def _save_pool_meta_snapshot(self, epoch_label: str) -> None:
        self.pool.save()
        snapshot_path = Path(self.run_dir) / f"pool_meta_{epoch_label}.json"
        self.pool.save(snapshot_path)

    def _save_elo_snapshot(self, epoch_label: str) -> None:
        self.elo_league.save_meta()
        self.elo_league.save_payoff()
        snapshot_dir = Path(self.run_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json_dump(
            {
                "elo_meta_path": str(self.elo_league.meta_path),
                "payoff_path": str(self.elo_league.payoff_path),
                "ratings_csv_path": str(self.elo_league.ratings_csv_path),
                "ratings_json_path": str(self.elo_league.ratings_json_path),
                "updated_at": _now(),
            },
            snapshot_dir / f"elo_snapshot_{epoch_label}.json",
        )

    def run(self, disable_tqdm: bool = False):
        pbar = tqdm(range(self.epochs), disable=disable_tqdm)
        for i in pbar:
            info: dict[str, Any] = {}
            info.update(self._epoch(epoch=i))
            self._log(info=info, epoch=i)

            if i % self.save_interval == 0 and self.save_interval > 0:
                torch.save(
                    self.policy_black.state_dict(),
                    os.path.join(self.run_dir, f"black_{i:04d}.pt"),
                )
                torch.save(
                    self.policy_white.state_dict(),
                    os.path.join(self.run_dir, f"white_{i:04d}.pt"),
                )
                self._save_pool_meta_snapshot(f"{i:04d}")
                self._save_elo_snapshot(f"{i:04d}")

            pbar.set_postfix(
                {
                    "fps": info.get("fps", 0.0),
                    "poolB": self.pool.role_size("black"),
                    "poolW": self.pool.role_size("white"),
                    "eloN": len(self.elo_league),
                    "eloB": info.get("elo_eval/current_black", float("nan")),
                    "eloW": info.get("elo_eval/current_white", float("nan")),
                }
            )

        torch.save(self.policy_black.state_dict(), os.path.join(self.run_dir, "black_final.pt"))
        torch.save(self.policy_white.state_dict(), os.path.join(self.run_dir, "white_final.pt"))
        self._save_pool_meta_snapshot("final")
        self._save_elo_snapshot("final")
        self._post_run()


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_psro")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    init_wandb(cfg=cfg)
    runner = DotaStylePSRORunner(cfg=cfg)
    runner.run()


if __name__ == "__main__":
    main()
