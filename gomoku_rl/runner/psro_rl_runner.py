"""PSRO-style historical-pool runner with role-separated bucket replay.

This module contains the full training logic for ``scripts/train_psro.py``.
It keeps PSRO code in the same runner style as ``IndependentRLRunner``: the
script file is only a Hydra entry point, while the runner owns rollout, update,
logging, checkpointing, model-pool snapshots, and Elo evaluation.

Compared with the previous per-epoch random historical-pool sampling, this
version keeps the original role-separated quality score update unchanged, but
changes opponent sampling to a two-level bucket structure:

1. consecutive historical models are grouped into buckets of ``bucket_size``;
2. a bucket score is the sum of the qualities of all models inside it;
3. a bucket is sampled by softmax(bucket_score);
4. the sampled bucket is locked for the whole bucket phase;
5. every epoch in that phase samples one model inside the locked bucket by
   softmax(model_quality).

The training schedule is phase-based instead of Bernoulli self-play:
``self_play_phase_epochs`` epochs of current black-vs-current white self-play,
then ``bucket_phase_epochs`` epochs of bucket replay, repeated.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from gomoku_rl.collector import BlackPlayCollector, VersusPlayCollector, WhitePlayCollector
from gomoku_rl.policy import get_policy
from gomoku_rl.runner.base import Runner
from gomoku_rl.runner.elo_model_pool import (
    EloEntry,
    EloEvalMixin,
    GlobalEloLeague,
    atomic_json_dump,
    get_run_id,
    now,
    sanitize_id,
    wandb_save_file,
)
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.utils.misc import add_prefix


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


@dataclass
class RolePoolBucket:
    """A locked role-specific bucket of historical checkpoints."""

    role: str
    bucket_index: int
    entry_indices: list[int]
    quality: float
    prob: float

    @property
    def size(self) -> int:
        return len(self.entry_indices)

    @property
    def id(self) -> str:
        return f"{self.role}_bucket_{self.bucket_index:05d}"


@dataclass
class SelectedReviewTeacher:
    """One historical teacher selected for post-Elo mixed-bucket review."""

    role: str
    index: int
    entry: RolePoolEntry
    prob: float
    env_num: int

class RoleSeparatedDiskPolicyPool:
    """On-disk role-separated policy pool.

    Metadata is kept in one JSON file, but black and white pools have independent
    entries and independent quality scores. This is important for Gomoku because
    black and white are not interchangeable roles.
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
        self.run_id = sanitize_id(run_id)
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
            "updated_at": now(),
            "black_entries": [asdict(entry) for entry in self.black_entries],
            "white_entries": [asdict(entry) for entry in self.white_entries],
            **self.extra,
        }

    def save(self, path: str | os.PathLike[str] | None = None) -> None:
        out_path = Path(path).expanduser().resolve() if path is not None else self.meta_path
        atomic_json_dump(self.to_dict(), out_path)

    @staticmethod
    def _softmax_from_scores(scores: list[float]) -> list[float]:
        if not scores:
            return []
        max_score = max(float(v) for v in scores)
        exp_scores = [math.exp(float(v) - max_score) for v in scores]
        denom = sum(exp_scores)
        if denom <= 0.0 or not math.isfinite(denom):
            return [1.0 / len(scores)] * len(scores)
        return [float(v) / denom for v in exp_scores]

    def softmax_probs(self, role: str) -> list[float]:
        entries = self._entries(role)
        if not entries:
            return []
        return self._softmax_from_scores([float(e.quality) for e in entries])

    def prob_variance(self, role: str) -> float:
        probs = self.softmax_probs(role)
        if not probs:
            return 0.0
        mean_p = 1.0 / len(probs)
        return float(sum((float(p) - mean_p) ** 2 for p in probs) / len(probs))

    def _make_buckets(self, role: str, bucket_size: int) -> list[RolePoolBucket]:
        entries = self._entries(role)
        if not entries:
            return []
        size = max(1, int(bucket_size))
        if len(entries) < size:
            return []

        # Only complete buckets are valid training buckets.
        # Example: bucket_size=10 means entries [0..9] form bucket 0,
        # entries [10..19] form bucket 1, and a trailing incomplete group
        # is ignored until it reaches bucket_size.
        raw_indices = [
            list(range(start, start + size))
            for start in range(0, len(entries) - size + 1, size)
        ]
        scores = [float(sum(entries[i].quality for i in indices)) for indices in raw_indices]
        probs = self._softmax_from_scores(scores)
        return [
            RolePoolBucket(
                role=str(role).lower().strip(),
                bucket_index=i,
                entry_indices=indices,
                quality=float(scores[i]),
                prob=float(probs[i]),
            )
            for i, indices in enumerate(raw_indices)
        ]

    def bucket_count(self, role: str, bucket_size: int) -> int:
        return len(self._make_buckets(role, bucket_size))

    def has_complete_buckets(self, bucket_size: int) -> bool:
        return (
            self.bucket_count("black", bucket_size) > 0
            and self.bucket_count("white", bucket_size) > 0
        )

    def bucket_prob_variance(self, role: str, bucket_size: int) -> float:
        buckets = self._make_buckets(role, bucket_size)
        if not buckets:
            return 0.0
        probs = [b.prob for b in buckets]
        mean_p = 1.0 / len(probs)
        return float(sum((float(p) - mean_p) ** 2 for p in probs) / len(probs))

    def sample_bucket(self, role: str, bucket_size: int, rng: random.Random) -> RolePoolBucket:
        buckets = self._make_buckets(role, bucket_size)
        if not buckets:
            raise RuntimeError(f"Cannot sample a bucket from an empty {role} policy pool.")
        idx = rng.choices(range(len(buckets)), weights=[b.prob for b in buckets], k=1)[0]
        return buckets[idx]

    def sample_from_bucket(self, bucket: RolePoolBucket, rng: random.Random) -> tuple[int, RolePoolEntry, float]:
        entries = self._entries(bucket.role)
        valid_indices = [idx for idx in bucket.entry_indices if 0 <= int(idx) < len(entries)]
        if not valid_indices:
            raise RuntimeError(f"Cannot sample from empty or stale bucket: {bucket}")
        probs = self._softmax_from_scores([float(entries[idx].quality) for idx in valid_indices])
        local_idx = rng.choices(range(len(valid_indices)), weights=probs, k=1)[0]
        entry_idx = int(valid_indices[local_idx])
        return entry_idx, entries[entry_idx], float(probs[local_idx])

    def sample(self, role: str, rng: random.Random) -> tuple[int, RolePoolEntry, float]:
        """Old single-level model-pool sampler, kept for compatibility/debugging."""

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
        safe_prefix = sanitize_id(prefix)
        entry_id = f"{role}_{safe_prefix}_{self.run_id}_e{epoch:05d}"
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
            created_at=now(),
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
            role="black",
            epoch=epoch,
            state_dict=black_state_dict,
            prefix=prefix,
            source=source,
        )
        white_entry = self.add_current_role(
            role="white",
            epoch=epoch,
            state_dict=white_state_dict,
            prefix=prefix,
            source=source,
        )
        return black_entry, white_entry

    def ensure_non_empty(
        self,
        *,
        black_state_dict: dict[str, Any],
        white_state_dict: dict[str, Any],
        epoch: int = 0,
    ) -> list[RolePoolEntry]:
        added: list[RolePoolEntry] = []
        if len(self.black_entries) == 0:
            added.append(
                self.add_current_role(
                    role="black",
                    epoch=epoch,
                    state_dict=black_state_dict,
                    prefix="initial",
                    source="initial_current",
                )
            )
        if len(self.white_entries) == 0:
            added.append(
                self.add_current_role(
                    role="white",
                    epoch=epoch,
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


class PSRORLRunner(EloEvalMixin, Runner):
    """Independent RL runner with role-separated bucket replay and global Elo."""

    _MODE_BOTH = "both"
    _MODE_BLACK_ONLY = "black_only"
    _MODE_WHITE_ONLY = "white_only"

    _PHASE_SELF_PLAY = "self_play"
    _PHASE_BUCKET = "bucket"

    def __init__(self, cfg: DictConfig) -> None:
        # Defensive compatibility: old configs may still contain eval_baseline_pool.
        # PSRO no longer loads or evaluates fixed baseline pools.
        with open_dict(cfg):
            cfg.eval_baseline_pool = {"black_pool": [], "white_pool": []}

        super().__init__(cfg)

        self.log_interval = int(self.cfg.get("log_interval", 5))
        self.pretrain_epoch_offset = int(self.cfg.get("pretrain_epoch_offset", 0))
        if self.pretrain_epoch_offset < 0:
            raise ValueError(f"pretrain_epoch_offset must be >= 0, got {self.pretrain_epoch_offset}")

        seed = self.cfg.get("seed", None)
        self._rng = random.Random(None if seed is None else int(seed))
        self.run_id = sanitize_id(get_run_id())

        self.self_play_phase_epochs = int(self.cfg.get("self_play_phase_epochs", 80))
        self.bucket_phase_epochs = int(self.cfg.get("bucket_phase_epochs", 20))
        self.bucket_size = int(self.cfg.get("bucket_size", 10))
        self.pool_lr = float(self.cfg.get("l", self.cfg.get("pool_lr", 0.01)))
        self.add_to_pool_interval = int(self.cfg.get("m", self.cfg.get("add_to_pool_interval", 10)))

        if self.self_play_phase_epochs < 0:
            raise ValueError(f"self_play_phase_epochs must be >= 0, got {self.self_play_phase_epochs}")
        if self.bucket_phase_epochs < 0:
            raise ValueError(f"bucket_phase_epochs must be >= 0, got {self.bucket_phase_epochs}")
        if self.self_play_phase_epochs + self.bucket_phase_epochs <= 0:
            raise ValueError("self_play_phase_epochs + bucket_phase_epochs must be > 0")
        if self.bucket_size <= 0:
            raise ValueError(f"bucket_size must be > 0, got {self.bucket_size}")
        if self.pool_lr < 0.0:
            raise ValueError(f"l/pool_lr must be >= 0, got {self.pool_lr}")

        # p/self_play_prob is intentionally no longer used for sampling. Keep a
        # best-effort read only for backward-compatible logging if old configs pass it.
        self.legacy_self_play_prob = self.cfg.get("p", self.cfg.get("self_play_prob", None))

        pool_dir = self.cfg.get("f", self.cfg.get("model_pool_dir", "model_pool"))
        meta_path = self.cfg.get("e", self.cfg.get("model_pool_meta", None))
        self.pool = RoleSeparatedDiskPolicyPool(
            pool_dir=to_absolute_path(str(pool_dir)),
            meta_path=(to_absolute_path(str(meta_path)) if meta_path is not None and str(meta_path).strip() else None),
            run_id=self.run_id,
            prevent_overwrite=True,
        )
        logging.info(
            "Role-separated bucket PSRO training pool loaded: black=%d, white=%d, total=%d, "
            "bucket_size=%d, self_play_phase_epochs=%d, bucket_phase_epochs=%d, dir=%s, meta=%s",
            self.pool.role_size("black"),
            self.pool.role_size("white"),
            self.pool.total_size(),
            self.bucket_size,
            self.self_play_phase_epochs,
            self.bucket_phase_epochs,
            self.pool.pool_dir,
            self.pool.meta_path,
        )

        self.elo_interval = int(self.cfg.get("elo_interval", self.cfg.get("add_to_elo_interval", 100)))
        self.elo_eval_repeats = int(self.cfg.get("elo_eval_repeats", 1))
        self.elo_include_self = True
        self.elo_base = float(self.cfg.get("elo_base", 1200.0))
        self.elo_scale = float(self.cfg.get("elo_scale", 400.0))
        self.elo_mle_iters = int(self.cfg.get("elo_mle_iters", 1000))
        self.elo_mle_lr = float(self.cfg.get("elo_mle_lr", 0.05))
        self.elo_mle_patience = int(self.cfg.get("elo_mle_patience", 100))
        self.elo_l2 = float(self.cfg.get("elo_l2", 1e-4))

        elo_dir = self.cfg.get("elo_models", self.cfg.get("elo_models_dir", "elo_models"))
        elo_meta = self.cfg.get("elo_e", self.cfg.get("elo_meta", self.cfg.get("elo_models_meta", None)))
        self.elo_league = GlobalEloLeague(
            elo_dir=to_absolute_path(str(elo_dir)),
            meta_path=(to_absolute_path(str(elo_meta)) if elo_meta is not None and str(elo_meta).strip() else None),
            run_id=self.run_id,
            prevent_overwrite=True,
        )
        logging.info(
            "Global Elo league loaded: size=%d, dir=%s, meta=%s",
            len(self.elo_league),
            self.elo_league.elo_dir,
            self.elo_league.meta_path,
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
            epoch=self.pretrain_epoch_offset,
        )
        if added_pool_entries:
            logging.info(
                "Added initial current policies to empty role pool(s): %s",
                ", ".join(entry.id for entry in added_pool_entries),
            )

        # Important: only add the starting current model if the Elo metadata is empty.
        # If `elo_e` points to an existing metadata file, that file fully controls the Elo pool.
        if len(self.elo_league) == 0:
            entry = self.elo_league.add_current_pair(
                epoch=self.pretrain_epoch_offset,
                black_state_dict=self.policy_black.state_dict(),
                white_state_dict=self.policy_white.state_dict(),
                prefix="elo_init",
                source="pretrained_init" if self.cfg.get("black_checkpoint", None) else "initial_current",
                parent_black_checkpoint=str(self.cfg.get("black_checkpoint", "") or ""),
                parent_white_checkpoint=str(self.cfg.get("white_checkpoint", "") or ""),
            )
            logging.info("Added initial current policies to empty Elo league: %s", entry.id)
        else:
            logging.info("Elo metadata is non-empty; current start checkpoint is not auto-added.")

        balance_cfg = self.cfg.get("balance", {})
        self.balance_enabled = bool(balance_cfg.get("enabled", True))
        # Direct PSRO balance for no-foul Gomoku: if enabled, do not use win-rate
        # thresholds or EMA. The update mode simply alternates between updating both
        # roles and updating white only. This gives white extra optimization steps
        # while still preventing black from being completely starved.
        self.current_bias_mode = self._MODE_WHITE_ONLY if self.balance_enabled else self._MODE_BOTH
        self.bias_turn_next = bool(self.balance_enabled)

        review_cfg = self.cfg.get("elo_review", {})
        self.elo_review_enabled = bool(review_cfg.get("enabled", self.cfg.get("elo_review_enabled", True)))
        self.elo_review_window = max(1, int(review_cfg.get("window", self.cfg.get("elo_review_window", 5))))
        self.elo_review_min_prior = max(1, int(review_cfg.get("min_prior", self.cfg.get("elo_review_min_prior", self.elo_review_window))))
        # How to reduce the most recent `window` prior Elo values into the review trigger threshold.
        #   min:  trigger only if current Elo is below the worst of recent N; preserves old behavior.
        #   mean: trigger if current Elo is below the recent-N average.
        #   max:  trigger unless current Elo reaches the best of recent N.
        self.elo_review_compare_stat = str(
            review_cfg.get(
                "compare_stat",
                review_cfg.get("stat", self.cfg.get("elo_review_compare_stat", "min")),
            )
        ).lower().strip()
        if self.elo_review_compare_stat not in {"min", "mean", "max"}:
            raise ValueError("elo_review.compare_stat must be one of: min, mean, max")
        # Number of most-recent complete teacher buckets used for post-Elo review.
        # ``recent_buckets`` is the preferred name. ``front_buckets`` is kept as a
        # backward-compatible alias for existing configs; it now means latest buckets,
        # not bucket 0..N-1.
        self.elo_review_recent_buckets = max(0, int(review_cfg.get(
            "recent_buckets",
            review_cfg.get("front_buckets", self.cfg.get("elo_review_front_buckets", self.elo_review_window)),
        )))
        self.elo_review_front_buckets = self.elo_review_recent_buckets
        self.elo_review_epochs = max(1, int(review_cfg.get("epochs", self.cfg.get("elo_review_epochs", 1))))
        self.elo_review_policy_name = str(review_cfg.get("policy_name", self.cfg.get("elo_review_policy_name", "ppo_review"))).lower().strip()
        self.elo_review_accept_margin = float(review_cfg.get("accept_margin", self.cfg.get("elo_review_accept_margin", 0.0)))
        self.elo_review_best_bucket_source = str(review_cfg.get("best_bucket_source", self.cfg.get("elo_review_best_bucket_source", "opponent"))).lower().strip()

        # Mixed-bucket review samples are collected into CPU memory first, then PPOReview moves only
        # each minibatch to GPU. These defaults mirror the previous mixed-teacher CPU-buffer runner.
        self.elo_review_base_env_num = int(review_cfg.get("base_env_num", self.cfg.get("base_env_num", 8)))
        self.elo_review_min_envs_per_teacher = int(review_cfg.get("min_envs_per_teacher", self.cfg.get("min_envs_per_teacher", 1)))
        raw_max_envs = review_cfg.get("max_envs_per_teacher", self.cfg.get("max_envs_per_teacher", None))
        self.elo_review_max_envs_per_teacher = None if raw_max_envs is None or str(raw_max_envs).lower() in {"", "none", "null"} else int(raw_max_envs)
        self.elo_review_env_rounding = str(review_cfg.get("env_rounding", self.cfg.get("env_rounding", "round"))).lower().strip()
        self.elo_review_teacher_prob_temperature = float(review_cfg.get("teacher_prob_temperature", self.cfg.get("teacher_prob_temperature", 1.0)))
        self.elo_review_teacher_chunk_size = max(1, int(review_cfg.get("teacher_chunk_size", self.cfg.get("teacher_chunk_size", 1))))
        self.elo_review_shuffle_rollout_before_ppo = bool(review_cfg.get("shuffle_rollout_before_ppo", self.cfg.get("shuffle_rollout_before_ppo", True)))
        self.elo_review_rollout_shuffle_mode = str(review_cfg.get("rollout_shuffle_mode", self.cfg.get("rollout_shuffle_mode", "env"))).lower().strip()
        self.elo_review_rollout_storage_device = str(review_cfg.get("rollout_storage_device", self.cfg.get("rollout_storage_device", "cpu"))).lower().strip()
        self.elo_review_collector_out_device = review_cfg.get("collector_out_device", self.cfg.get("collector_out_device", "cpu"))
        self.elo_review_keep_rejected_elo_entry = bool(review_cfg.get("keep_rejected_elo_entry", self.cfg.get("elo_review_keep_rejected_elo_entry", False)))

        if self.elo_review_base_env_num <= 0:
            raise ValueError(f"elo_review.base_env_num must be > 0, got {self.elo_review_base_env_num}")
        if self.elo_review_min_envs_per_teacher < 0:
            raise ValueError("elo_review.min_envs_per_teacher must be >= 0")
        if self.elo_review_max_envs_per_teacher is not None and self.elo_review_max_envs_per_teacher < self.elo_review_min_envs_per_teacher:
            raise ValueError("elo_review.max_envs_per_teacher must be >= min_envs_per_teacher")
        if self.elo_review_env_rounding not in {"round", "floor", "ceil", "stochastic"}:
            raise ValueError("elo_review.env_rounding must be one of: round, floor, ceil, stochastic")
        if self.elo_review_teacher_prob_temperature <= 0.0:
            raise ValueError("elo_review.teacher_prob_temperature must be > 0")
        if self.elo_review_best_bucket_source not in {"opponent", "current", "same_role"}:
            raise ValueError("elo_review.best_bucket_source must be 'opponent' or 'current'")
        if self.elo_review_rollout_storage_device not in {"cpu", "cuda", "cuda:0", "gpu"}:
            raise ValueError("elo_review.rollout_storage_device must be one of: cpu, cuda, cuda:0, gpu")
        if self.elo_review_rollout_shuffle_mode in {"false", "off", "no", "none", "disable", "disabled"}:
            self.elo_review_shuffle_rollout_before_ppo = False
            self.elo_review_rollout_shuffle_mode = "none"
        elif self.elo_review_rollout_shuffle_mode in {"env", "env_dim", "trajectory", "trajectory_env", "row", "rows"}:
            self.elo_review_rollout_shuffle_mode = "env"
        else:
            raise ValueError("elo_review.rollout_shuffle_mode only supports 'env' or 'none'")

        try:
            self._review_observation_mode, self._review_temporal_num_steps = self._get_observation_cfg()
        except Exception:
            self._review_observation_mode = self.cfg.get("observation_mode", None)
            self._review_temporal_num_steps = self.cfg.get("temporal_num_steps", None)

        logging.info(
            "Elo review config: enabled=%s window=%d min_prior=%d compare_stat=%s "
            "recent_buckets=%d epochs=%d policy=%s base_env_num=%d storage=%s keep_rejected_elo=%s",
            self.elo_review_enabled,
            self.elo_review_window,
            self.elo_review_min_prior,
            self.elo_review_compare_stat,
            self.elo_review_front_buckets,
            self.elo_review_epochs,
            self.elo_review_policy_name,
            self.elo_review_base_env_num,
            self.elo_review_rollout_storage_device,
            self.elo_review_keep_rejected_elo_entry,
        )

        self._locked_bucket_phase_index: int | None = None
        self._locked_buckets: dict[str, RolePoolBucket | None] = {"black": None, "white": None}
        self._current_train_phase = self._PHASE_SELF_PLAY
        self._current_phase_index = 0
        self._current_phase_pos = 0

    # ---------------------------- epoch helpers ----------------------------

    def _completed_epoch(self, local_epoch: int) -> int:
        return int(self.pretrain_epoch_offset + local_epoch + 1)

    def _checkpoint_epoch(self, local_epoch: int) -> int:
        return int(self.pretrain_epoch_offset + local_epoch + 1)

    def _epoch_label(self, epoch_value: int) -> str:
        return f"{int(epoch_value):05d}"

    def _phase_state(self, local_epoch: int) -> tuple[str, int, int]:
        cycle_len = int(self.self_play_phase_epochs + self.bucket_phase_epochs)
        global_zero_based_epoch = int(self.pretrain_epoch_offset + local_epoch)
        phase_index = global_zero_based_epoch // cycle_len
        phase_pos = global_zero_based_epoch % cycle_len
        if self.self_play_phase_epochs > 0 and phase_pos < self.self_play_phase_epochs:
            return self._PHASE_SELF_PLAY, phase_index, phase_pos
        return self._PHASE_BUCKET, phase_index, phase_pos

    def _lock_buckets_for_phase(self, phase_index: int) -> None:
        # Caller guarantees that both role pools have at least one complete bucket.
        self._locked_buckets["black"] = self.pool.sample_bucket("black", self.bucket_size, self._rng)
        self._locked_buckets["white"] = self.pool.sample_bucket("white", self.bucket_size, self._rng)
        self._locked_bucket_phase_index = int(phase_index)

        logging.info(
            "Locked PSRO buckets for phase=%d: black_bucket=%s prob=%.6f q_sum=%.6f size=%d; "
            "white_bucket=%s prob=%.6f q_sum=%.6f size=%d",
            phase_index,
            self._locked_buckets["black"].id if self._locked_buckets["black"] else "none",
            self._locked_buckets["black"].prob if self._locked_buckets["black"] else 0.0,
            self._locked_buckets["black"].quality if self._locked_buckets["black"] else 0.0,
            self._locked_buckets["black"].size if self._locked_buckets["black"] else 0,
            self._locked_buckets["white"].id if self._locked_buckets["white"] else "none",
            self._locked_buckets["white"].prob if self._locked_buckets["white"] else 0.0,
            self._locked_buckets["white"].quality if self._locked_buckets["white"] else 0.0,
            self._locked_buckets["white"].size if self._locked_buckets["white"] else 0,
        )

    def _prepare_phase(self, local_epoch: int) -> tuple[str, int, int]:
        scheduled_phase, phase_index, phase_pos = self._phase_state(local_epoch)
        phase = scheduled_phase
        self._current_phase_index = int(phase_index)
        self._current_phase_pos = int(phase_pos)

        if scheduled_phase == self._PHASE_BUCKET and self.bucket_phase_epochs > 0:
            if self.pool.has_complete_buckets(self.bucket_size):
                if self._locked_bucket_phase_index != phase_index:
                    self._lock_buckets_for_phase(phase_index)
            else:
                # No complete black/white buckets yet: bucket phase falls back to
                # ordinary current-black-vs-current-white self-play.
                phase = self._PHASE_SELF_PLAY
                self._locked_bucket_phase_index = None
                self._locked_buckets = {"black": None, "white": None}
        else:
            self._locked_bucket_phase_index = None
            self._locked_buckets = {"black": None, "white": None}

        self._current_train_phase = phase
        return phase, phase_index, phase_pos

    # ---------------------------- schedule helpers ----------------------------

    def _set_policy_schedules(self, epoch: int, info: dict[str, Any]) -> None:
        """Apply per-outer-epoch LR / entropy schedules before learning.

        ``epoch`` is the zero-based local outer epoch from tqdm. When training is
        resumed from checkpoints, ``pretrain_epoch_offset`` is added so decay
        continues from the global outer epoch instead of restarting from zero.
        """

        schedule_epoch = int(self.pretrain_epoch_offset + epoch)
        for policy_name, policy in (
            ("policy_black", self.policy_black),
            ("policy_white", self.policy_white),
        ):
            if hasattr(policy, "set_outer_epoch"):
                policy.set_outer_epoch(schedule_epoch)
            if hasattr(policy, "get_schedule_info"):
                info.update(add_prefix(policy.get_schedule_info(), f"{policy_name}/"))

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
        if not self.balance_enabled:
            return {"balance/bias_active": 0.0, "balance/next_turn_biased": 0.0, "balance/next_turn_both": 1.0}
        return {
            "balance/bias_active": 1.0,
            "balance/next_turn_biased": float(self.bias_turn_next),
            "balance/next_turn_both": float(not self.bias_turn_next),
        }

    def _balance_trigger_flags(self) -> dict[str, float]:
        return {
            "balance/trigger_black_only": 0.0,
            "balance/trigger_white_only": float(self.balance_enabled),
        }

    def _get_applied_mode_for_current_epoch(self) -> str:
        if not self.balance_enabled:
            self.current_bias_mode = self._MODE_BOTH
            return self._MODE_BOTH
        self.current_bias_mode = self._MODE_WHITE_ONLY
        applied_mode = self._MODE_WHITE_ONLY if self.bias_turn_next else self._MODE_BOTH
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

    def _apply_learning(self, applied_mode: str, data_black, data_white, info: dict[str, Any]) -> None:
        info.update(self._applied_mode_to_flags(applied_mode))
        info["balance/black_update_skipped"] = float(applied_mode == self._MODE_WHITE_ONLY)
        info["balance/white_update_skipped"] = float(applied_mode == self._MODE_BLACK_ONLY)
        if applied_mode in (self._MODE_BOTH, self._MODE_BLACK_ONLY):
            self._learn_black(data_black, info)
        if applied_mode in (self._MODE_BOTH, self._MODE_WHITE_ONLY):
            self._learn_white(data_white, info)

    # ------------------------------ rollout modes -----------------------------

    def _rollout_current_self_play(self, applied_mode: str) -> dict[str, Any]:
        data_black, data_white, raw_info = self._self_collector.rollout(self.steps)
        info: dict[str, Any] = add_prefix(raw_info, "self_play/")
        info["train/source_self_play"] = 1.0
        info["train/source_bucket"] = 0.0
        info["fps"] = info.get("self_play/fps", 0.0)
        info.pop("self_play/fps", None)
        self._apply_learning(applied_mode, data_black, data_white, info)
        return info

    def _load_role_pool_policy(self, entry: RolePoolEntry):
        if not Path(entry.path).is_file():
            raise FileNotFoundError(f"pool {entry.role} checkpoint not found: {entry.path}")
        return self._build_inference_policy_from_checkpoint(entry.path, tag=f"pool_{entry.role}/{entry.id}")

    def _get_locked_bucket(self, role: str) -> RolePoolBucket:
        bucket = self._locked_buckets.get(role)
        if bucket is None:
            # Defensive fallback for resumed runs or direct calls.
            bucket = self.pool.sample_bucket(role, self.bucket_size, self._rng)
            self._locked_buckets[role] = bucket
        return bucket

    def _rollout_against_pool(self, applied_mode: str, epoch: int) -> dict[str, Any]:
        if self.pool.role_size("black") == 0 or self.pool.role_size("white") == 0:
            self.pool.ensure_non_empty(
                black_state_dict=self.policy_black.state_dict(),
                white_state_dict=self.policy_white.state_dict(),
                epoch=self.pretrain_epoch_offset,
            )

        info: dict[str, Any] = {"train/source_self_play": 0.0, "train/source_bucket": 1.0}
        fps_parts: list[float] = []
        data_black = None
        data_white = None
        sampled_policies: list[Any] = []
        global_epoch = self._completed_epoch(epoch)

        # Train current black against a sampled historical white inside the locked white bucket.
        if applied_mode in (self._MODE_BOTH, self._MODE_BLACK_ONLY):
            white_bucket = self._get_locked_bucket("white")
            white_idx, white_entry, white_in_bucket_prob = self.pool.sample_from_bucket(white_bucket, self._rng)
            white_sample_prob = float(white_bucket.prob) * float(white_in_bucket_prob)
            hist_white = self._load_role_pool_policy(white_entry)
            sampled_policies.append(hist_white)

            info.update(
                {
                    "pool_white/bucket_index": float(white_bucket.bucket_index),
                    "pool_white/bucket_size": float(white_bucket.size),
                    "pool_white/bucket_quality_sum": float(white_bucket.quality),
                    "pool_white/bucket_prob": float(white_bucket.prob),
                    "pool_white/sampled_index": float(white_idx),
                    "pool_white/sampled_epoch": float(white_entry.epoch),
                    "pool_white/sampled_relative_epoch_ratio": float((white_entry.epoch - global_epoch) / max(global_epoch, 1)),
                    "pool_white/sampled_prob_in_bucket": float(white_in_bucket_prob),
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

        # Train current white against a sampled historical black inside the locked black bucket.
        if applied_mode in (self._MODE_BOTH, self._MODE_WHITE_ONLY):
            black_bucket = self._get_locked_bucket("black")
            black_idx, black_entry, black_in_bucket_prob = self.pool.sample_from_bucket(black_bucket, self._rng)
            black_sample_prob = float(black_bucket.prob) * float(black_in_bucket_prob)
            hist_black = self._load_role_pool_policy(black_entry)
            sampled_policies.append(hist_black)

            info.update(
                {
                    "pool_black/bucket_index": float(black_bucket.bucket_index),
                    "pool_black/bucket_size": float(black_bucket.size),
                    "pool_black/bucket_quality_sum": float(black_bucket.quality),
                    "pool_black/bucket_prob": float(black_bucket.prob),
                    "pool_black/sampled_index": float(black_idx),
                    "pool_black/sampled_epoch": float(black_entry.epoch),
                    "pool_black/sampled_relative_epoch_ratio": float((black_entry.epoch - global_epoch) / max(global_epoch, 1)),
                    "pool_black/sampled_prob_in_bucket": float(black_in_bucket_prob),
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
        self._self_collector.reset()
        return info

    def _maybe_add_current_to_pool(self, epoch: int, info: dict[str, Any]) -> None:
        if self.add_to_pool_interval <= 0:
            return
        if (epoch + 1) % self.add_to_pool_interval != 0:
            return
        global_epoch = self._completed_epoch(epoch)
        black_entry, white_entry = self.pool.add_current_pair(
            epoch=global_epoch,
            black_state_dict=self.policy_black.state_dict(),
            white_state_dict=self.policy_white.state_dict(),
            prefix="epoch",
        )
        info["pool/added_epoch"] = float(global_epoch)
        logging.info(
            "Added current policies to role-separated training pool: black=%s white=%s",
            black_entry.id,
            white_entry.id,
        )

    def _epoch(self, epoch: int) -> dict[str, Any]:
        schedule_info: dict[str, Any] = {}
        self._set_policy_schedules(epoch, schedule_info)

        applied_mode = self._get_applied_mode_for_current_epoch()
        phase, phase_index, phase_pos = self._prepare_phase(epoch)
        use_self_play = phase == self._PHASE_SELF_PLAY or len(self.pool) == 0 or self.bucket_phase_epochs <= 0

        if use_self_play:
            info = self._rollout_current_self_play(applied_mode)
        else:
            info = self._rollout_against_pool(applied_mode, epoch)

        info.update(schedule_info)
        info.update(
            {
                "psro/self_play_phase_epochs": float(self.self_play_phase_epochs),
                "psro/bucket_phase_epochs": float(self.bucket_phase_epochs),
                "psro/bucket_size": float(self.bucket_size),
                "psro/pool_lr": self.pool_lr,
                "psro/add_to_pool_interval": float(self.add_to_pool_interval),
                "psro/phase_is_self_play": float(phase == self._PHASE_SELF_PLAY),
                "psro/phase_is_bucket": float(phase == self._PHASE_BUCKET),
                "psro/phase_index": float(phase_index),
                "psro/phase_pos": float(phase_pos),
                "time/local_epoch_completed": float(epoch + 1),
                "time/global_epoch_completed": float(self._completed_epoch(epoch)),
                "time/pretrain_epoch_offset": float(self.pretrain_epoch_offset),
                "pool_black/size": float(self.pool.role_size("black")),
                "pool_black/prob_variance": self.pool.prob_variance("black"),
                "pool_black/bucket_count": float(self.pool.bucket_count("black", self.bucket_size)),
                "pool_black/bucket_prob_variance": self.pool.bucket_prob_variance("black", self.bucket_size),
                "pool_white/size": float(self.pool.role_size("white")),
                "pool_white/prob_variance": self.pool.prob_variance("white"),
                "pool_white/bucket_count": float(self.pool.bucket_count("white", self.bucket_size)),
                "pool_white/bucket_prob_variance": self.pool.bucket_prob_variance("white", self.bucket_size),
                "elo_eval/interval": float(self.elo_interval),
                "elo_eval/num_models": float(len(self.elo_league)),
                "elo_eval/payoff_coverage": self.elo_league.payoff_coverage(include_self=True),
                "balance/enabled": float(self.balance_enabled),
                "balance/fixed_alternating": float(self.balance_enabled),
                "balance/fixed_bias_white_only": float(self.balance_enabled),
            }
        )
        if self.legacy_self_play_prob is not None:
            info["psro/legacy_self_play_prob_unused"] = float(self.legacy_self_play_prob)

        info.update(self._bias_mode_to_flags(self.current_bias_mode))
        info.update(self._phase_flags())
        info.update(self._balance_trigger_flags())

        self._maybe_add_current_to_pool(epoch, info)

        if epoch % 50 == 0 and epoch != 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return info


    # -------------------------- post-Elo review helpers --------------------------

    @staticmethod
    def _opponent_role(role: str) -> str:
        role = str(role).lower().strip()
        if role == "black":
            return "white"
        if role == "white":
            return "black"
        raise ValueError(f"unknown role: {role}")

    @staticmethod
    def _prob_entropy(probs: list[float]) -> float:
        return float(-sum(float(p) * math.log(max(float(p), 1e-12)) for p in probs)) if probs else 0.0

    @staticmethod
    def _prob_effective_n(probs: list[float]) -> float:
        denom = sum(float(p) ** 2 for p in probs)
        return float(1.0 / denom) if denom > 0.0 else 0.0

    def _elo_rows_by_id(self, ratings: dict[str, list[dict[str, Any]]], role: str) -> dict[str, dict[str, Any]]:
        return {str(row["id"]): row for row in ratings.get(role, [])}

    def _elo_row(self, ratings: dict[str, list[dict[str, Any]]], role: str, entry_id: str) -> dict[str, Any] | None:
        return self._elo_rows_by_id(ratings, role).get(str(entry_id))

    def _prior_elo_rows(
        self,
        *,
        role: str,
        ratings: dict[str, list[dict[str, Any]]],
        current_entry_id: str,
    ) -> list[dict[str, Any]]:
        """Return all prior Elo rows for ``role``, ordered by training chronology.

        ``min_prior`` is intentionally checked against this full list.
        ``window`` is only applied later to choose the recent rows used for
        the regression comparison. This makes the config semantics:

            min_prior: minimum number of historical Elo models in the league
            window: number of most recent historical Elo models to compare to
        """
        rows_by_id = self._elo_rows_by_id(ratings, role)
        prior_entries = [entry for entry in self.elo_league.entries if entry.id != current_entry_id]
        prior_entries.sort(key=lambda e: (int(e.epoch), float(e.created_at), str(e.id)))
        return [rows_by_id[e.id] for e in prior_entries if e.id in rows_by_id]

    def _elo_review_needed(
        self,
        *,
        role: str,
        ratings: dict[str, list[dict[str, Any]]],
        current_entry_id: str,
        current_elo: float,
        info: dict[str, Any],
    ) -> bool:
        prefix = f"elo_review/{role}"

        all_prior_rows = self._prior_elo_rows(role=role, ratings=ratings, current_entry_id=current_entry_id)
        info[f"{prefix}/prior_count"] = float(len(all_prior_rows))
        info[f"{prefix}/compare_count"] = 0.0

        if len(all_prior_rows) < int(self.elo_review_min_prior):
            info[f"{prefix}/triggered"] = 0.0
            info[f"{prefix}/skipped_not_enough_prior"] = 1.0
            return False

        compare_rows = all_prior_rows[-int(self.elo_review_window) :]
        info[f"{prefix}/compare_count"] = float(len(compare_rows))
        if len(compare_rows) <= 0:
            info[f"{prefix}/triggered"] = 0.0
            info[f"{prefix}/skipped_not_enough_prior"] = 1.0
            return False

        prior_elos = [float(row["elo"]) for row in compare_rows]
        prior_min_elo = float(min(prior_elos))
        prior_max_elo = float(max(prior_elos))
        prior_mean_elo = float(sum(prior_elos) / len(prior_elos))

        compare_stat = str(self.elo_review_compare_stat).lower().strip()
        if compare_stat == "min":
            threshold = prior_min_elo
        elif compare_stat == "mean":
            threshold = prior_mean_elo
        elif compare_stat == "max":
            threshold = prior_max_elo
        else:
            raise ValueError(f"elo_review.compare_stat must be one of: min, mean, max, got {compare_stat!r}")

        info[f"{prefix}/prior_min_elo"] = prior_min_elo
        info[f"{prefix}/prior_max_elo"] = prior_max_elo
        info[f"{prefix}/prior_mean_elo"] = prior_mean_elo
        info[f"{prefix}/compare_threshold_elo"] = float(threshold)
        # Numeric one-hot flags are safer for wandb charts than logging a string.
        info[f"{prefix}/compare_stat_is_min"] = float(compare_stat == "min")
        info[f"{prefix}/compare_stat_is_mean"] = float(compare_stat == "mean")
        info[f"{prefix}/compare_stat_is_max"] = float(compare_stat == "max")

        needed = float(current_elo) < float(threshold)
        info[f"{prefix}/triggered"] = float(needed)
        info[f"{prefix}/skipped_not_enough_prior"] = 0.0
        return bool(needed)

    def _best_elo_entry_for_role(
        self,
        *,
        role: str,
        ratings: dict[str, list[dict[str, Any]]],
    ) -> EloEntry | None:
        rows = ratings.get(role, [])
        if not rows:
            return None
        best_id = str(rows[0]["id"])
        return next((entry for entry in self.elo_league.entries if entry.id == best_id), None)

    def _find_pool_bucket_by_epoch(self, *, role: str, epoch: int) -> int | None:
        entries = self.pool._entries(role)
        if not entries:
            return None
        complete_count = len(entries) // self.bucket_size
        if complete_count <= 0:
            return None

        exact = [(idx, entry) for idx, entry in enumerate(entries[: complete_count * self.bucket_size]) if int(entry.epoch) == int(epoch)]
        if exact:
            return int(exact[-1][0] // self.bucket_size)

        # Fallback for intervals that do not align exactly with pool snapshots: choose the nearest
        # historical entry not newer than the Elo entry; if all are newer, use the nearest by epoch.
        eligible = [(idx, entry) for idx, entry in enumerate(entries[: complete_count * self.bucket_size]) if int(entry.epoch) <= int(epoch)]
        if eligible:
            idx, _ = max(eligible, key=lambda item: (int(item[1].epoch), int(item[0])))
            return int(idx // self.bucket_size)
        idx, _ = min(
            [(idx, entry) for idx, entry in enumerate(entries[: complete_count * self.bucket_size])],
            key=lambda item: abs(int(item[1].epoch) - int(epoch)),
        )
        return int(idx // self.bucket_size)

    def _review_bucket_indices_for_train_role(
        self,
        *,
        train_role: str,
        ratings: dict[str, list[dict[str, Any]]],
        info: dict[str, Any],
    ) -> list[int]:
        teacher_role = self._opponent_role(train_role)
        bucket_count = self.pool.bucket_count(teacher_role, self.bucket_size)
        prefix = f"elo_review/{train_role}"
        if bucket_count <= 0:
            info[f"{prefix}/skipped_no_teacher_bucket"] = 1.0
            return []

        recent_n = min(int(self.elo_review_recent_buckets), bucket_count)
        recent_start = max(0, bucket_count - recent_n)
        bucket_indices = list(range(recent_start, bucket_count))

        best_source_role = teacher_role if self.elo_review_best_bucket_source == "opponent" else train_role
        best_entry = self._best_elo_entry_for_role(role=best_source_role, ratings=ratings)
        best_bucket = None
        if best_entry is not None:
            # The teacher bucket always comes from the opponent-role training pool. When best_source_role
            # is current/same_role, this maps by epoch into the opponent pool, because current-role
            # checkpoints cannot be used as opponents for this review role.
            best_bucket = self._find_pool_bucket_by_epoch(role=teacher_role, epoch=int(best_entry.epoch))
            if best_bucket is not None and 0 <= int(best_bucket) < bucket_count and int(best_bucket) not in bucket_indices:
                bucket_indices.append(int(best_bucket))

        info[f"{prefix}/teacher_role_is_white"] = float(teacher_role == "white")
        info[f"{prefix}/teacher_bucket_count"] = float(bucket_count)
        info[f"{prefix}/recent_bucket_count"] = float(recent_n)
        info[f"{prefix}/recent_bucket_start"] = float(recent_start)
        info[f"{prefix}/recent_bucket_end"] = float(bucket_count - 1 if recent_n > 0 else -1)
        # Backward-compatible metric name for old dashboards/configs.
        info[f"{prefix}/front_bucket_count"] = float(recent_n)
        info[f"{prefix}/best_bucket_index"] = float(best_bucket if best_bucket is not None else -1)
        info[f"{prefix}/mixed_bucket_count"] = float(len(bucket_indices))
        logging.info(
            "Elo review %s mixed teacher buckets: teacher_role=%s recent=%s best=%s final=%s",
            train_role,
            teacher_role,
            list(range(recent_start, bucket_count)),
            best_bucket if best_bucket is not None else None,
            bucket_indices,
        )
        return bucket_indices

    def _selected_review_teachers(
        self,
        *,
        train_role: str,
        ratings: dict[str, list[dict[str, Any]]],
        info: dict[str, Any],
    ) -> list[SelectedReviewTeacher]:
        teacher_role = self._opponent_role(train_role)
        bucket_indices = self._review_bucket_indices_for_train_role(train_role=train_role, ratings=ratings, info=info)
        if not bucket_indices:
            return []

        entries = self.pool._entries(teacher_role)
        selected: list[tuple[int, RolePoolEntry]] = []
        for bucket_idx in bucket_indices:
            start = int(bucket_idx) * int(self.bucket_size)
            end = min(start + int(self.bucket_size), len(entries))
            if end - start < int(self.bucket_size):
                continue
            selected.extend((idx, entries[idx]) for idx in range(start, end))

        if not selected:
            return []

        scores = [float(entry.quality) / float(self.elo_review_teacher_prob_temperature) for _, entry in selected]
        probs = RoleSeparatedDiskPolicyPool._softmax_from_scores(scores)
        n = len(selected)
        out: list[SelectedReviewTeacher] = []
        for (idx, entry), prob in zip(selected, probs):
            raw_envs = float(self.elo_review_base_env_num) * float(prob) * float(n)
            env_num = self._round_review_env_count(raw_envs)
            env_num = max(int(self.elo_review_min_envs_per_teacher), int(env_num))
            if self.elo_review_max_envs_per_teacher is not None:
                env_num = min(int(self.elo_review_max_envs_per_teacher), int(env_num))
            out.append(
                SelectedReviewTeacher(
                    role=teacher_role,
                    index=int(idx),
                    entry=entry,
                    prob=float(prob),
                    env_num=int(env_num),
                )
            )

        if out and all(item.env_num <= 0 for item in out):
            best = max(range(len(out)), key=lambda i: out[i].prob)
            out[best].env_num = max(1, int(self.elo_review_base_env_num))

        pfx = f"elo_review/{train_role}"
        probs_only = [float(t.prob) for t in out]
        env_counts = [int(t.env_num) for t in out]
        info[f"{pfx}/teacher_count"] = float(len(out))
        info[f"{pfx}/active_teacher_count"] = float(sum(1 for t in out if t.env_num > 0))
        info[f"{pfx}/total_review_envs"] = float(sum(env_counts))
        info[f"{pfx}/teacher_prob_entropy"] = self._prob_entropy(probs_only)
        info[f"{pfx}/teacher_prob_effective_n"] = self._prob_effective_n(probs_only)
        info[f"{pfx}/teacher_max_prob"] = float(max(probs_only) if probs_only else 0.0)
        info[f"{pfx}/teacher_min_prob"] = float(min(probs_only) if probs_only else 0.0)
        info[f"{pfx}/teacher_max_envs"] = float(max(env_counts) if env_counts else 0)
        info[f"{pfx}/teacher_min_envs"] = float(min(env_counts) if env_counts else 0)
        return out

    def _round_review_env_count(self, value: float) -> int:
        if self.elo_review_env_rounding == "floor":
            return int(math.floor(value))
        if self.elo_review_env_rounding == "ceil":
            return int(math.ceil(value))
        if self.elo_review_env_rounding == "stochastic":
            lower = math.floor(value)
            frac = float(value - lower)
            return int(lower + (1 if self._rng.random() < frac else 0))
        return int(round(value))

    def _make_review_rollout_env(self, num_envs: int):
        kwargs: dict[str, Any] = {}
        if self._review_observation_mode is not None:
            kwargs["observation_mode"] = self._review_observation_mode
        if self._review_temporal_num_steps is not None:
            kwargs["temporal_num_steps"] = self._review_temporal_num_steps
        try:
            return self._make_env(num_envs=int(num_envs), **kwargs)
        except TypeError:
            return self._make_env(num_envs=int(num_envs))

    def _store_review_rollout_buffer(self, data):
        if data is None:
            return None
        target = self.elo_review_rollout_storage_device
        if target in {"gpu", "cuda", "cuda:0"}:
            target = "cuda"
        try:
            data = data.detach()
        except Exception:
            pass
        try:
            return data.to(target)
        except TypeError:
            return data.to(torch.device(target))

    def _shuffle_review_rollout_tensordict(self, tensordict):
        if tensordict is None or not self.elo_review_shuffle_rollout_before_ppo:
            return tensordict
        if self.elo_review_rollout_shuffle_mode == "none":
            return tensordict
        if self.elo_review_rollout_shuffle_mode != "env":
            raise ValueError(f"unsupported elo_review.rollout_shuffle_mode: {self.elo_review_rollout_shuffle_mode}")
        try:
            batch_size = tensordict.batch_size
        except Exception:
            batch_size = getattr(tensordict, "shape", None)
        if batch_size is None or len(batch_size) < 1:
            return tensordict
        n_env_rows = int(batch_size[0])
        if n_env_rows <= 1:
            return tensordict
        device = getattr(tensordict, "device", None)
        if device is None:
            device = "cpu" if self.elo_review_rollout_storage_device == "cpu" else self.device
        perm = torch.randperm(n_env_rows, device=device)
        return tensordict[perm]

    def _cat_review_rollouts(self, parts: list[Any]):
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        combined = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return self._shuffle_review_rollout_tensordict(combined)

    def _review_algo_cfg_for_role(self, role: str):
        """Return the role-specific PPO config used to build a review policy.

        Normal PSRO training may use independent ``algo.black`` and
        ``algo.white`` configs. Review mirrors that behavior: black review uses
        black PPO hyperparameters and white review uses white PPO
        hyperparameters. If an older flat PPO config is used, this falls back to
        the top-level ``algo`` block for backward compatibility.
        """
        if role not in {"black", "white"}:
            raise ValueError(f"invalid review role: {role}")

        algo_cfg = self.cfg.algo
        role_cfg = None
        try:
            role_cfg = algo_cfg.get(role, None)
        except Exception:
            role_cfg = getattr(algo_cfg, role, None)

        review_cfg = copy.deepcopy(role_cfg if role_cfg is not None else algo_cfg)

        # ``get_policy`` selects the class from the explicit name argument, but
        # some policy code still reads ``cfg.name`` / ``cfg.role`` internally.
        # Keep those fields consistent with the temporary review policy.
        if isinstance(review_cfg, DictConfig):
            with open_dict(review_cfg):
                review_cfg.name = self.elo_review_policy_name
                review_cfg.role = role
        elif isinstance(review_cfg, dict):
            review_cfg["name"] = self.elo_review_policy_name
            review_cfg["role"] = role
        else:
            try:
                setattr(review_cfg, "name", self.elo_review_policy_name)
            except Exception:
                pass
            try:
                setattr(review_cfg, "role", role)
            except Exception:
                pass

        return review_cfg

    def _make_review_policy_from_current(self, role: str):
        if role not in {"black", "white"}:
            raise ValueError(f"invalid review role: {role}")

        base_policy = self.policy_black if role == "black" else self.policy_white
        review_cfg = self._review_algo_cfg_for_role(role)
        review_policy = get_policy(
            name=self.elo_review_policy_name,
            cfg=review_cfg,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )
        review_policy.load_state_dict(copy.deepcopy(base_policy.state_dict()))
        if hasattr(review_policy, "keep_rollout_buffer_on_cpu"):
            setattr(review_policy, "keep_rollout_buffer_on_cpu", True)
        if hasattr(review_policy, "gae_on_cpu"):
            setattr(review_policy, "gae_on_cpu", True)
        if hasattr(review_policy, "train"):
            review_policy.train()
        return review_policy

    def _copy_review_policy_back(self, *, role: str, review_policy) -> None:
        base_policy = self.policy_black if role == "black" else self.policy_white
        base_policy.load_state_dict(copy.deepcopy(review_policy.state_dict()))
        if hasattr(base_policy, "optim") and hasattr(review_policy, "optim"):
            try:
                base_policy.optim.load_state_dict(copy.deepcopy(review_policy.optim.state_dict()))
            except Exception as exc:
                logging.warning("Could not copy %s review optimizer state back to main policy: %s", role, exc)
        if hasattr(base_policy, "train"):
            base_policy.train()

    def _rollout_review_against_mixed_teachers(
        self,
        *,
        train_role: str,
        train_policy,
        teachers: list[SelectedReviewTeacher],
        info: dict[str, Any],
    ):
        if train_role not in {"black", "white"}:
            raise ValueError(f"train_role must be black or white, got {train_role}")
        active = [teacher for teacher in teachers if int(teacher.env_num) > 0]
        if not active:
            return None, 0.0

        teacher_role = self._opponent_role(train_role)
        data_parts: list[Any] = []
        fps_values: list[tuple[float, float]] = []
        win_values: list[tuple[float, float]] = []
        pfx = f"elo_review/{train_role}"

        for chunk_start in range(0, len(active), int(self.elo_review_teacher_chunk_size)):
            chunk = active[chunk_start : chunk_start + int(self.elo_review_teacher_chunk_size)]
            for teacher in chunk:
                hist_policy = self._load_role_pool_policy(teacher.entry)
                rollout_env = self._make_review_rollout_env(int(teacher.env_num))
                collector = None
                data = None
                try:
                    if train_role == "black":
                        collector = BlackPlayCollector(
                            rollout_env,
                            policy_black=train_policy,
                            policy_white=hist_policy,
                            out_device=self.elo_review_collector_out_device,
                            augment=self.cfg.get("augment", False),
                        )
                        data, raw_info = collector.rollout(self.steps)
                        win_key = "black_win"
                    else:
                        collector = WhitePlayCollector(
                            rollout_env,
                            policy_black=hist_policy,
                            policy_white=train_policy,
                            out_device=self.elo_review_collector_out_device,
                            augment=self.cfg.get("augment", False),
                        )
                        data, raw_info = collector.rollout(self.steps)
                        win_key = "white_win"

                    data = self._store_review_rollout_buffer(data)
                    data_parts.append(data)
                    fps_values.append((float(raw_info.get("fps", 0.0)), float(teacher.env_num)))
                    win_values.append((float(raw_info.get(win_key, 0.0)), float(teacher.env_num)))
                finally:
                    del data
                    del collector
                    del hist_policy
                    if rollout_env is not self.env:
                        del rollout_env
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        combined = self._cat_review_rollouts(data_parts)
        info[f"{pfx}/teacher_role_is_white"] = float(teacher_role == "white")
        info[f"{pfx}/review_win_rate_vs_mixed_bucket"] = self._weighted_mean_review(win_values)
        return combined, self._weighted_mean_review(fps_values)

    @staticmethod
    def _weighted_mean_review(values: list[tuple[float, float]]) -> float:
        denom = sum(float(w) for _, w in values)
        if denom <= 0.0:
            return 0.0
        return float(sum(float(v) * float(w) for v, w in values) / denom)

    def _learn_review_policy(self, *, role: str, review_policy, data, info: dict[str, Any], review_epoch: int) -> None:
        pfx = f"elo_review/{role}"
        if data is None:
            info[f"{pfx}/update_skipped_empty_data"] = 1.0
            return
        if hasattr(review_policy, "keep_rollout_buffer_on_cpu"):
            setattr(review_policy, "keep_rollout_buffer_on_cpu", True)
        if hasattr(review_policy, "gae_on_cpu"):
            setattr(review_policy, "gae_on_cpu", True)
        if hasattr(review_policy, "set_outer_epoch"):
            review_policy.set_outer_epoch(int(self.pretrain_epoch_offset + review_epoch))
        info.update(add_prefix(review_policy.learn(data), f"elo_review/{role}/policy/"))

    def _evaluate_missing_global_elo_pairs(self) -> tuple[int, float]:
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
        elapsed = time.perf_counter() - start
        return int(evaluated), float(elapsed)

    def _fit_global_elo(self) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
        return self.elo_league.fit_role_elo_with_alpha(
            base_elo=self.elo_base,
            elo_scale=self.elo_scale,
            l2=self.elo_l2,
            iters=self.elo_mle_iters,
            lr=self.elo_mle_lr,
            patience=self.elo_mle_patience,
            min_delta=0.0,
            games_per_pair=max(1, int(getattr(self.eval_env, "num_envs", 1)) * max(1, self.elo_eval_repeats)),
        )

    def _remove_elo_entry(self, entry: EloEntry) -> None:
        entry_id = str(entry.id)
        self.elo_league.entries = [e for e in self.elo_league.entries if e.id != entry_id]
        self.elo_league.payoff.pop(entry_id, None)
        for row in self.elo_league.payoff.values():
            if isinstance(row, dict):
                row.pop(entry_id, None)
        for path_str in (entry.black_path, entry.white_path):
            try:
                Path(path_str).unlink(missing_ok=True)
            except Exception:
                pass
        self.elo_league.save_meta()
        self.elo_league.save_payoff()
        self._fit_global_elo()

    def _evaluate_review_candidate(
        self,
        *,
        role: str,
        review_policy,
        baseline_elo: float,
        global_epoch: int,
        info: dict[str, Any],
    ) -> tuple[float, EloEntry]:
        black_state = review_policy.state_dict() if role == "black" else self.policy_black.state_dict()
        white_state = review_policy.state_dict() if role == "white" else self.policy_white.state_dict()
        entry = self.elo_league.add_current_pair(
            epoch=int(global_epoch),
            black_state_dict=black_state,
            white_state_dict=white_state,
            prefix=f"elo_review_{role}",
            source="elo_review_candidate",
            parent_black_checkpoint=str(self.cfg.get("black_checkpoint", "") or ""),
            parent_white_checkpoint=str(self.cfg.get("white_checkpoint", "") or ""),
        )
        evaluated, elapsed = self._evaluate_missing_global_elo_pairs()
        ratings, fit_summary = self._fit_global_elo()
        row = self._elo_row(ratings, role, entry.id)
        candidate_elo = float(row["elo"]) if row is not None else float(self.elo_base)
        candidate_rank = float(row["rank"]) if row is not None else 1.0
        pfx = f"elo_review/{role}"
        info[f"{pfx}/candidate_entry_added"] = 1.0
        info[f"{pfx}/candidate_elo"] = candidate_elo
        info[f"{pfx}/candidate_rank"] = candidate_rank
        info[f"{pfx}/candidate_delta_vs_baseline"] = float(candidate_elo - float(baseline_elo))
        info[f"{pfx}/candidate_missing_pairs_evaluated"] = float(evaluated)
        info[f"{pfx}/candidate_eval_seconds"] = float(elapsed)
        info[f"{pfx}/candidate_fit_loss"] = float(fit_summary.get("loss", 0.0))
        return candidate_elo, entry

    def _run_role_elo_review(
        self,
        *,
        role: str,
        baseline_elo: float,
        ratings: dict[str, list[dict[str, Any]]],
        global_epoch: int,
        info: dict[str, Any],
    ) -> None:
        pfx = f"elo_review/{role}"
        review_policy = None
        candidate_entry = None
        try:
            review_policy = self._make_review_policy_from_current(role)
            total_fps: list[float] = []
            for review_epoch in range(int(self.elo_review_epochs)):
                teachers = self._selected_review_teachers(train_role=role, ratings=ratings, info=info)
                if not teachers:
                    info[f"{pfx}/skipped_no_teacher"] = 1.0
                    return
                data, fps = self._rollout_review_against_mixed_teachers(
                    train_role=role,
                    train_policy=review_policy,
                    teachers=teachers,
                    info=info,
                )
                total_fps.append(float(fps))
                self._learn_review_policy(role=role, review_policy=review_policy, data=data, info=info, review_epoch=review_epoch)
                del data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            info[f"{pfx}/fps"] = float(sum(total_fps) / len(total_fps)) if total_fps else 0.0
            candidate_elo, candidate_entry = self._evaluate_review_candidate(
                role=role,
                review_policy=review_policy,
                baseline_elo=float(baseline_elo),
                global_epoch=int(global_epoch),
                info=info,
            )
            accepted = float(candidate_elo) > float(baseline_elo) + float(self.elo_review_accept_margin)
            info[f"{pfx}/accepted"] = float(accepted)
            info[f"{pfx}/rejected"] = float(not accepted)
            if accepted:
                self._copy_review_policy_back(role=role, review_policy=review_policy)
                logging.info(
                    "Accepted %s Elo review at epoch=%d: baseline=%.2f candidate=%.2f delta=%.2f",
                    role,
                    global_epoch,
                    baseline_elo,
                    candidate_elo,
                    candidate_elo - float(baseline_elo),
                )
            else:
                logging.info(
                    "Rejected %s Elo review at epoch=%d: baseline=%.2f candidate=%.2f delta=%.2f",
                    role,
                    global_epoch,
                    baseline_elo,
                    candidate_elo,
                    candidate_elo - float(baseline_elo),
                )
                if candidate_entry is not None and not self.elo_review_keep_rejected_elo_entry:
                    self._remove_elo_entry(candidate_entry)
        finally:
            del review_policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _maybe_run_elo_review(
        self,
        *,
        epoch: int,
        global_epoch: int,
        current_entry: EloEntry,
        ratings: dict[str, list[dict[str, Any]]],
        current_black_elo: float,
        current_white_elo: float,
        info: dict[str, Any],
    ) -> None:
        if not self.elo_review_enabled:
            info["elo_review/enabled"] = 0.0
            return
        info["elo_review/enabled"] = 1.0
        if self.pool.bucket_count("black", self.bucket_size) <= 0 or self.pool.bucket_count("white", self.bucket_size) <= 0:
            info["elo_review/skipped_no_complete_bucket"] = 1.0
            return

        review_black = self._elo_review_needed(
            role="black",
            ratings=ratings,
            current_entry_id=current_entry.id,
            current_elo=float(current_black_elo),
            info=info,
        )
        review_white = self._elo_review_needed(
            role="white",
            ratings=ratings,
            current_entry_id=current_entry.id,
            current_elo=float(current_white_elo),
            info=info,
        )

        if review_black:
            self._run_role_elo_review(
                role="black",
                baseline_elo=float(current_black_elo),
                ratings=ratings,
                global_epoch=int(global_epoch),
                info=info,
            )
        if review_white:
            # Use the original normal-Elo rating table for trigger/teacher selection. If black review
            # accepted first, the white review still starts from the updated current black opponent, which
            # is the desired sequential current-state behavior.
            self._run_role_elo_review(
                role="white",
                baseline_elo=float(current_white_elo),
                ratings=ratings,
                global_epoch=int(global_epoch),
                info=info,
            )

    # ------------------------------ global Elo eval ---------------------------

    def _load_elo_policy_pair(self, entry: EloEntry, *, load_black: bool, load_white: bool):
        black_policy = None
        white_policy = None
        if load_black:
            if not Path(entry.black_path).is_file():
                raise FileNotFoundError(f"Elo black checkpoint not found: {entry.black_path}")
            black_policy = self._build_inference_policy_from_checkpoint(entry.black_path, tag=f"elo_black/{entry.id}")
        if load_white:
            if not Path(entry.white_path).is_file():
                raise FileNotFoundError(f"Elo white checkpoint not found: {entry.white_path}")
            white_policy = self._build_inference_policy_from_checkpoint(entry.white_path, tag=f"elo_white/{entry.id}")
        return black_policy, white_policy

    def _evaluate_one_payoff_pair(self, black_entry: EloEntry, white_entry: EloEntry) -> float:
        black_policy, _ = self._load_elo_policy_pair(black_entry, load_black=True, load_white=False)
        _, white_policy = self._load_elo_policy_pair(white_entry, load_black=False, load_white=True)
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

        global_epoch = self._completed_epoch(epoch)
        entry = self.elo_league.add_current_pair(
            epoch=global_epoch,
            black_state_dict=self.policy_black.state_dict(),
            white_state_dict=self.policy_white.state_dict(),
            prefix="elo",
            source="train_psro",
            parent_black_checkpoint=str(self.cfg.get("black_checkpoint", "") or ""),
            parent_white_checkpoint=str(self.cfg.get("white_checkpoint", "") or ""),
        )
        info["elo_eval/added_epoch"] = float(entry.epoch)
        info["elo_eval/num_models"] = float(len(self.elo_league))

        evaluated, elapsed = self._evaluate_missing_global_elo_pairs()
        ratings, fit_summary = self._fit_global_elo()

        black_rows = ratings.get("black", [])
        white_rows = ratings.get("white", [])
        current_black_row = self._elo_row(ratings, "black", entry.id)
        current_white_row = self._elo_row(ratings, "white", entry.id)
        best_black_row = black_rows[0] if black_rows else None
        best_white_row = white_rows[0] if white_rows else None

        current_black_elo = float(current_black_row["elo"]) if current_black_row is not None else float(self.elo_base)
        current_white_elo = float(current_white_row["elo"]) if current_white_row is not None else float(self.elo_base)
        current_black_rank = float(current_black_row["rank"]) if current_black_row is not None else 1.0
        current_white_rank = float(current_white_row["rank"]) if current_white_row is not None else 1.0
        best_black_elo = float(best_black_row["elo"]) if best_black_row is not None else current_black_elo
        best_white_elo = float(best_white_row["elo"]) if best_white_row is not None else current_white_elo

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
            "Global role Elo updated at epoch=%d: current=%s black=%.2f white=%.2f "
            "black_rank=%s white_rank=%s alpha=%.2f models=%d evaluated_pairs=%d",
            global_epoch,
            entry.id,
            current_black_elo,
            current_white_elo,
            int(current_black_rank),
            int(current_white_rank),
            float(fit_summary.get("black_advantage_elo", 0.0)),
            len(self.elo_league),
            evaluated,
        )

        self._maybe_run_elo_review(
            epoch=epoch,
            global_epoch=global_epoch,
            current_entry=entry,
            ratings=ratings,
            current_black_elo=current_black_elo,
            current_white_elo=current_white_elo,
            info=info,
        )

    # ------------------------------- logging ---------------------------------

    def _format_eval_summary(self, info: dict[str, Any]) -> str:
        parts = [
            f"Black vs White:{info['eval/black_vs_white'] * 100.0:.2f}%",
            f"pool_black:{self.pool.role_size('black')} pool_white:{self.pool.role_size('white')}",
            f"bucket_size:{self.bucket_size}",
            f"phase:{self._current_train_phase}",
            f"phase_pos:{self._current_phase_pos}",
            f"elo_models:{len(self.elo_league)}",
            f"bias_mode:{self.current_bias_mode}",
            "next_turn:{}".format(
                "biased" if (self.current_bias_mode != self._MODE_BOTH and self.bias_turn_next) else "both"
            ),
        ]
        if self._current_train_phase == self._PHASE_BUCKET:
            black_bucket = self._locked_buckets.get("black")
            white_bucket = self._locked_buckets.get("white")
            if black_bucket is not None:
                parts.append(f"bucketB:{black_bucket.bucket_index}/p={black_bucket.prob:.3f}")
            if white_bucket is not None:
                parts.append(f"bucketW:{white_bucket.bucket_index}/p={white_bucket.prob:.3f}")
        if "elo_eval/current_black" in info:
            parts.append(f"EloB:{info['elo_eval/current_black']:.1f}")
            parts.append(f"EloW:{info['elo_eval/current_white']:.1f}")
            parts.append(f"Brank:{int(info['elo_eval/current_black_rank'])}")
            parts.append(f"Wrank:{int(info['elo_eval/current_white_rank'])}")
        return " ".join(parts)

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % self.log_interval == 0:
            black_vs_white_raw = float(eval_win_rate(self.eval_env, player_black=self.policy_black, player_white=self.policy_white))
            info.update(
                {
                    "eval/black_vs_white": black_vs_white_raw,
                    "pool_black/size": float(self.pool.role_size("black")),
                    "pool_black/prob_variance": self.pool.prob_variance("black"),
                    "pool_black/bucket_count": float(self.pool.bucket_count("black", self.bucket_size)),
                    "pool_black/bucket_prob_variance": self.pool.bucket_prob_variance("black", self.bucket_size),
                    "pool_white/size": float(self.pool.role_size("white")),
                    "pool_white/prob_variance": self.pool.prob_variance("white"),
                    "pool_white/bucket_count": float(self.pool.bucket_count("white", self.bucket_size)),
                    "pool_white/bucket_prob_variance": self.pool.bucket_prob_variance("white", self.bucket_size),
                    "elo_eval/num_models": float(len(self.elo_league)),
                    "elo_eval/payoff_coverage": self.elo_league.payoff_coverage(include_self=True),
                }
            )
            info.update(self._balance_trigger_flags())
            info.update(self._bias_mode_to_flags(self.current_bias_mode))
            info.update(self._phase_flags())

        self._maybe_update_global_elo(epoch, info)

        if epoch % self.log_interval == 0 or "elo_eval/current_black" in info:
            if "eval/black_vs_white" not in info:
                black_vs_white_raw = float(eval_win_rate(self.eval_env, player_black=self.policy_black, player_white=self.policy_white))
                info["eval/black_vs_white"] = black_vs_white_raw
            print(self._format_eval_summary(info))
        else:
            info.update(self._bias_mode_to_flags(self.current_bias_mode))
            info.update(self._phase_flags())
            info.update(self._balance_trigger_flags())

        return super()._log(info, epoch)

    # ---------------------------- metadata snapshots --------------------------

    def _save_pool_meta_snapshot(self, epoch_label: str) -> None:
        self.pool.save()
        snapshot_path = Path(self.run_dir) / f"pool_meta_{epoch_label}.json"
        self.pool.save(snapshot_path)
        wandb_save_file(snapshot_path, base_path=self.run_dir)

    def _save_elo_snapshot(self, epoch_label: str) -> None:
        self.elo_league.save_meta()
        self.elo_league.save_payoff()
        snapshot_dir = Path(self.run_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        meta_snapshot_path = snapshot_dir / f"elo_meta_{epoch_label}.json"
        self.elo_league.save(meta_snapshot_path)
        wandb_save_file(meta_snapshot_path, base_path=self.run_dir)

        summary_path = snapshot_dir / f"elo_snapshot_{epoch_label}.json"
        atomic_json_dump(
            {
                "elo_meta_path": str(self.elo_league.meta_path),
                "payoff_path": str(self.elo_league.payoff_path),
                "ratings_csv_path": str(self.elo_league.ratings_csv_path),
                "ratings_json_path": str(self.elo_league.ratings_json_path),
                "snapshot_meta_path": str(meta_snapshot_path),
                "updated_at": now(),
            },
            summary_path,
        )
        wandb_save_file(summary_path, base_path=self.run_dir)

    def run(self, disable_tqdm: bool = False):
        pbar = tqdm(range(self.epochs), disable=disable_tqdm)
        for i in pbar:
            info: dict[str, Any] = {}
            info.update(self._epoch(epoch=i))
            self._log(info=info, epoch=i)

            # Save after completed local epochs. With offset=1000 and
            # save_interval=100, the first periodic checkpoint is
            # black_01100.pt / white_01100.pt.
            if self.save_interval > 0 and (i + 1) % self.save_interval == 0:
                ckpt_epoch = self._checkpoint_epoch(i)
                epoch_label = self._epoch_label(ckpt_epoch)
                black_path = os.path.join(self.run_dir, f"black_{epoch_label}.pt")
                white_path = os.path.join(self.run_dir, f"white_{epoch_label}.pt")
                torch.save(self.policy_black.state_dict(), black_path)
                torch.save(self.policy_white.state_dict(), white_path)
                wandb_save_file(black_path, base_path=self.run_dir)
                wandb_save_file(white_path, base_path=self.run_dir)
                self._save_pool_meta_snapshot(epoch_label)
                self._save_elo_snapshot(epoch_label)

            pbar.set_postfix(
                {
                    "fps": info.get("fps", 0.0),
                    "poolB": self.pool.role_size("black"),
                    "poolW": self.pool.role_size("white"),
                    "bucketB": self.pool.bucket_count("black", self.bucket_size),
                    "bucketW": self.pool.bucket_count("white", self.bucket_size),
                    "phase": self._current_train_phase,
                    "epoch": self._completed_epoch(i),
                    "eloB": info.get("elo_eval/current_black", float("nan")),
                    "eloW": info.get("elo_eval/current_white", float("nan")),
                }
            )

        torch.save(self.policy_black.state_dict(), os.path.join(self.run_dir, "black_final.pt"))
        torch.save(self.policy_white.state_dict(), os.path.join(self.run_dir, "white_final.pt"))
        wandb_save_file(os.path.join(self.run_dir, "black_final.pt"), base_path=self.run_dir)
        wandb_save_file(os.path.join(self.run_dir, "white_final.pt"), base_path=self.run_dir)
        self._save_pool_meta_snapshot("final")
        self._save_elo_snapshot("final")
        self._post_run()
