"""Mixed-teacher curriculum runner for Gomoku RL.

This runner uses an existing role-separated historical model pool as a fixed
teacher set. ``start_bucket``/``end_bucket`` and ``bucket_size`` are used only
to select a contiguous metadata range:

    selected entries = entries[start_bucket * bucket_size : end_bucket * bucket_size]

After the range is selected, bucket identity is discarded. Every selected single
teacher model receives a softmax probability from its existing ``quality/q``
score. A configurable ``base_env_num`` means: if a model had the uniform
probability 1 / n over n selected teachers, it would receive ``base_env_num``
parallel environments. Other teachers receive

    round(base_env_num * probability * n)

environments, optionally clamped by min/max settings. Rollouts from all selected
teachers are concatenated along the environment dimension, optionally shuffled
across the environment/trajectory dimension, and passed to the existing PPO
learner, so PPO inner epochs/minibatches stay controlled by the original algo
config.
"""

from __future__ import annotations

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
class SelectedTeacher:
    """One selected single-model teacher after range slicing and q-softmax."""

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

    def get_bucket(self, role: str, bucket_size: int, bucket_index: int) -> RolePoolBucket:
        buckets = self._make_buckets(role, bucket_size)
        if not buckets:
            raise RuntimeError(f"No complete {role} buckets are available.")
        idx = int(bucket_index)
        if not (0 <= idx < len(buckets)):
            raise IndexError(f"{role} bucket index out of range: index={idx}, available={len(buckets)}")
        return buckets[idx]

    def sample_from_bucket(
        self,
        bucket: RolePoolBucket,
        rng: random.Random,
        *,
        mode: str = "softmax_quality",
    ) -> tuple[int, RolePoolEntry, float]:
        entries = self._entries(bucket.role)
        valid_indices = [idx for idx in bucket.entry_indices if 0 <= int(idx) < len(entries)]
        if not valid_indices:
            raise RuntimeError(f"Cannot sample from empty or stale bucket: {bucket}")
        if mode == "uniform":
            probs = [1.0 / len(valid_indices)] * len(valid_indices)
        elif mode == "softmax_quality":
            probs = self._softmax_from_scores([float(entries[idx].quality) for idx in valid_indices])
        else:
            raise ValueError(f"unknown bucket sampling mode: {mode}")
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


class MixedTeacherCurriculumRLRunner(EloEvalMixin, Runner):
    """Independent RL runner with mixed historical teacher curriculum and global Elo."""

    _MODE_BOTH = "both"
    _MODE_BLACK_ONLY = "black_only"
    _MODE_WHITE_ONLY = "white_only"

    _PHASE_SELF_PLAY = "self_play"
    _PHASE_MIXED = "mixed_teacher_curriculum"

    def __init__(self, cfg: DictConfig) -> None:
        # Defensive compatibility: old configs may still contain eval_baseline_pool.
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
        self._observation_mode, self._temporal_num_steps = self._get_observation_cfg()

        # Range selection still uses bucket_size/start_bucket/end_bucket, but
        # after slicing we flatten the selected buckets into single-model teachers.
        self.bucket_size = int(self.cfg.get("bucket_size", 10))
        self.start_bucket = int(self.cfg.get("start_bucket", 0))
        raw_end_bucket = self.cfg.get("end_bucket", None)
        self.end_bucket: int | None = None if raw_end_bucket is None or str(raw_end_bucket).lower() in {"", "none", "null"} else int(raw_end_bucket)
        self.train_roles = str(self.cfg.get("train_roles", "both")).lower().strip()

        # If p_i = 1 / n, this teacher receives base_env_num envs.
        self.base_env_num = int(self.cfg.get("base_env_num", self.cfg.get("base_env_num_per_uniform_model", 8)))
        self.min_envs_per_teacher = int(self.cfg.get("min_envs_per_teacher", 1))
        raw_max_envs = self.cfg.get("max_envs_per_teacher", None)
        self.max_envs_per_teacher: int | None = None if raw_max_envs is None or str(raw_max_envs).lower() in {"", "none", "null"} else int(raw_max_envs)
        self.env_rounding = str(self.cfg.get("env_rounding", "round")).lower().strip()
        self.teacher_prob_temperature = float(self.cfg.get("teacher_prob_temperature", 1.0))

        # Shuffle after concatenating all teacher rollouts and before PPO.learn().
        # The safe default is env/trajectory-row shuffling: it mixes teacher blocks
        # while preserving the time dimension needed by GAE. Transition-level
        # flatten-shuffle is intentionally not done here, because PPO.learn()
        # computes GAE before minibatching; flattening before that would break the
        # temporal recurrence. The PPO learner already flattens and randomizes
        # minibatches after value targets / advantages have been computed.
        self.shuffle_rollout_before_ppo = bool(self.cfg.get("shuffle_rollout_before_ppo", True))
        self.rollout_shuffle_mode = str(
            self.cfg.get("rollout_shuffle_mode", self.cfg.get("shuffle_rollout_mode", "env"))
        ).lower().strip()
        if self.rollout_shuffle_mode in {"false", "off", "no", "none", "disable", "disabled"}:
            self.shuffle_rollout_before_ppo = False
            self.rollout_shuffle_mode = "none"
        elif self.rollout_shuffle_mode in {"env", "env_dim", "trajectory", "trajectory_env", "row", "rows"}:
            self.rollout_shuffle_mode = "env"
        else:
            raise ValueError(
                "rollout_shuffle_mode only supports 'env' or 'none' in the runner. "
                "Do transition-level shuffling inside PPO after GAE/value targets are computed."
            )

        # Memory-safe mixed-teacher mode:
        #   rollout each teacher/chunk, immediately store the rollout buffer on CPU,
        #   concatenate the complete mixed batch on CPU, then let PPO move only
        #   each minibatch to GPU. This keeps the desired globally mixed batch
        #   distribution without retaining all teacher rollouts on CUDA.
        self.offload_rollout_to_cpu = bool(self.cfg.get("offload_rollout_to_cpu", True))
        self.ppo_learn_from_cpu_buffer = bool(
            self.cfg.get("ppo_learn_from_cpu_buffer", self.offload_rollout_to_cpu)
        )
        self.teacher_chunk_size = int(self.cfg.get("teacher_chunk_size", 1))
        self.rollout_storage_device = str(self.cfg.get("rollout_storage_device", "cpu" if self.offload_rollout_to_cpu else self.cfg.get("out_device", None))).lower()
        if self.rollout_storage_device in {"", "none", "null"}:
            self.rollout_storage_device = str(self.cfg.get("out_device", None)).lower()
        self.collector_out_device = self.cfg.get(
            "collector_out_device",
            "cpu" if self.offload_rollout_to_cpu else self.cfg.get("out_device", None),
        )

        self.outer_epochs = int(self.cfg.get("outer_epochs", self.cfg.get("mixed_teacher_outer_epochs", self.cfg.get("epochs", 1))))
        self.override_epochs_from_outer = bool(self.cfg.get("override_epochs_from_outer", True))
        self.update_teacher_quality = bool(self.cfg.get("update_teacher_quality", False))
        self.pool_lr = float(self.cfg.get("l", self.cfg.get("pool_lr", 0.0)))

        if self.bucket_size <= 0:
            raise ValueError(f"bucket_size must be > 0, got {self.bucket_size}")
        if self.start_bucket < 0:
            raise ValueError(f"start_bucket must be >= 0, got {self.start_bucket}")
        if self.base_env_num <= 0:
            raise ValueError(f"base_env_num must be > 0, got {self.base_env_num}")
        if self.min_envs_per_teacher < 0:
            raise ValueError(f"min_envs_per_teacher must be >= 0, got {self.min_envs_per_teacher}")
        if self.max_envs_per_teacher is not None and self.max_envs_per_teacher <= 0:
            raise ValueError(f"max_envs_per_teacher must be > 0 when set, got {self.max_envs_per_teacher}")
        if self.max_envs_per_teacher is not None and self.max_envs_per_teacher < self.min_envs_per_teacher:
            raise ValueError("max_envs_per_teacher must be >= min_envs_per_teacher")
        if self.env_rounding not in {"round", "floor", "ceil", "stochastic"}:
            raise ValueError("env_rounding must be one of: round, floor, ceil, stochastic")
        if self.teacher_prob_temperature <= 0.0:
            raise ValueError("teacher_prob_temperature must be > 0")
        if self.outer_epochs <= 0:
            raise ValueError(f"outer_epochs must be > 0, got {self.outer_epochs}")
        if self.teacher_chunk_size <= 0:
            raise ValueError(f"teacher_chunk_size must be > 0, got {self.teacher_chunk_size}")
        if self.rollout_storage_device not in {"cpu", "cuda", "cuda:0", "gpu"}:
            raise ValueError(
                "rollout_storage_device must be one of: cpu, cuda, cuda:0, gpu; "
                f"got {self.rollout_storage_device!r}"
            )
        if self.train_roles not in {self._MODE_BOTH, self._MODE_BLACK_ONLY, self._MODE_WHITE_ONLY, "black", "white"}:
            raise ValueError(
                "train_roles must be one of: both, black, white, black_only, white_only; "
                f"got {self.train_roles!r}"
            )
        if self.train_roles == "black":
            self.train_roles = self._MODE_BLACK_ONLY
        elif self.train_roles == "white":
            self.train_roles = self._MODE_WHITE_ONLY
        if self.pool_lr < 0.0:
            raise ValueError(f"l/pool_lr must be >= 0, got {self.pool_lr}")

        pool_dir = self.cfg.get("f", self.cfg.get("model_pool_dir", "model_pool"))
        meta_path = self.cfg.get("e", self.cfg.get("model_pool_meta", None))
        self.pool = RoleSeparatedDiskPolicyPool(
            pool_dir=to_absolute_path(str(pool_dir)),
            meta_path=(to_absolute_path(str(meta_path)) if meta_path is not None and str(meta_path).strip() else None),
            run_id=self.run_id,
            prevent_overwrite=True,
        )

        self.available_curriculum_buckets = self._compute_available_bucket_count()
        if self.end_bucket is None:
            self.end_bucket = self.available_curriculum_buckets
        if self.end_bucket < self.start_bucket:
            raise ValueError(f"end_bucket must be >= start_bucket, got {self.end_bucket} < {self.start_bucket}")
        if self.end_bucket > self.available_curriculum_buckets:
            raise ValueError(
                f"end_bucket={self.end_bucket} exceeds available curriculum buckets "
                f"{self.available_curriculum_buckets} for train_roles={self.train_roles}"
            )
        self.selected_bucket_count = int(self.end_bucket - self.start_bucket)
        if self.selected_bucket_count <= 0:
            raise ValueError(
                "No selected teacher range is available. Check f/e, bucket_size, "
                "train_roles, start_bucket, and end_bucket."
            )
        self.selected_black_teacher_count = len(self._selected_entries("black"))
        self.selected_white_teacher_count = len(self._selected_entries("white"))
        if self.train_roles in (self._MODE_BOTH, self._MODE_WHITE_ONLY) and self.selected_black_teacher_count <= 0:
            raise ValueError("No historical black teachers selected for current-white training.")
        if self.train_roles in (self._MODE_BOTH, self._MODE_BLACK_ONLY) and self.selected_white_teacher_count <= 0:
            raise ValueError("No historical white teachers selected for current-black training.")
        if self.override_epochs_from_outer:
            self.epochs = self.outer_epochs

        logging.info(
            "Mixed-teacher curriculum pool loaded: black=%d, white=%d, total=%d, "
            "bucket_size=%d, available_buckets=%d, selected=[%d,%d), "
            "selected_black=%d, selected_white=%d, base_env_num=%d, "
            "min_envs=%d, max_envs=%s, total_epochs=%d, train_roles=%s, "
            "temperature=%.4f, shuffle_before_ppo=%s, shuffle_mode=%s, "
            "offload_rollout_to_cpu=%s, ppo_cpu_buffer=%s, teacher_chunk_size=%d, "
            "collector_out_device=%s, storage_device=%s, dir=%s, meta=%s",
            self.pool.role_size("black"),
            self.pool.role_size("white"),
            self.pool.total_size(),
            self.bucket_size,
            self.available_curriculum_buckets,
            self.start_bucket,
            self.end_bucket,
            self.selected_black_teacher_count,
            self.selected_white_teacher_count,
            self.base_env_num,
            self.min_envs_per_teacher,
            self.max_envs_per_teacher,
            self.epochs,
            self.train_roles,
            self.teacher_prob_temperature,
            self.shuffle_rollout_before_ppo,
            self.rollout_shuffle_mode,
            self.offload_rollout_to_cpu,
            self.ppo_learn_from_cpu_buffer,
            self.teacher_chunk_size,
            self.collector_out_device,
            self.rollout_storage_device,
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
        self.balance_enabled = bool(balance_cfg.get("enabled", False))
        self.current_bias_mode = self._MODE_WHITE_ONLY if self.balance_enabled else self._MODE_BOTH
        self.bias_turn_next = bool(self.balance_enabled)

        # Compatibility fields used by shared logging/checkpoint code.
        self._current_train_phase = self._PHASE_MIXED
        self._current_cycle_index = 0
        self._current_bucket_index = self.start_bucket
        self._current_bucket_offset = 0
        self._current_epoch_in_bucket = 0
        self._current_buckets: dict[str, RolePoolBucket | None] = {"black": None, "white": None}

    # ---------------------------- epoch helpers ----------------------------

    def _completed_epoch(self, local_epoch: int) -> int:
        return int(self.pretrain_epoch_offset + local_epoch + 1)

    def _checkpoint_epoch(self, local_epoch: int) -> int:
        return int(self.pretrain_epoch_offset + local_epoch + 1)

    def _epoch_label(self, epoch_value: int) -> str:
        return f"{int(epoch_value):05d}"

    def _compute_available_bucket_count(self) -> int:
        black_bucket_count = self.pool.bucket_count("black", self.bucket_size)
        white_bucket_count = self.pool.bucket_count("white", self.bucket_size)
        if self.train_roles == self._MODE_BOTH:
            return int(min(black_bucket_count, white_bucket_count))
        if self.train_roles == self._MODE_BLACK_ONLY:
            # Current black learns from historical white teachers.
            return int(white_bucket_count)
        if self.train_roles == self._MODE_WHITE_ONLY:
            # Current white learns from historical black teachers.
            return int(black_bucket_count)
        raise ValueError(f"unknown train_roles: {self.train_roles}")

    def _selected_range(self, role: str) -> tuple[int, int]:
        entries = self.pool._entries(role)
        role_bucket_count = len(entries) // self.bucket_size
        end_bucket = self.end_bucket if self.end_bucket is not None else role_bucket_count
        end_bucket = min(int(end_bucket), role_bucket_count)
        start = int(self.start_bucket * self.bucket_size)
        end = int(end_bucket * self.bucket_size)
        return start, end

    def _selected_entries(self, role: str) -> list[tuple[int, RolePoolEntry]]:
        entries = self.pool._entries(role)
        start, end = self._selected_range(role)
        return [(idx, entries[idx]) for idx in range(start, end)]

    def _softmax_selected(self, selected: list[tuple[int, RolePoolEntry]]) -> list[float]:
        if not selected:
            return []
        scores = [float(entry.quality) / self.teacher_prob_temperature for _, entry in selected]
        max_score = max(scores)
        exp_scores = [math.exp(score - max_score) for score in scores]
        denom = sum(exp_scores)
        if denom <= 0.0 or not math.isfinite(denom):
            return [1.0 / len(selected)] * len(selected)
        return [float(v) / denom for v in exp_scores]

    def _prob_entropy(self, probs: list[float]) -> float:
        return float(-sum(float(p) * math.log(max(float(p), 1e-12)) for p in probs)) if probs else 0.0

    def _prob_effective_n(self, probs: list[float]) -> float:
        denom = sum(float(p) ** 2 for p in probs)
        return float(1.0 / denom) if denom > 0.0 else 0.0

    def _round_env_count(self, value: float) -> int:
        if self.env_rounding == "floor":
            return int(math.floor(value))
        if self.env_rounding == "ceil":
            return int(math.ceil(value))
        if self.env_rounding == "stochastic":
            lower = math.floor(value)
            frac = float(value - lower)
            return int(lower + (1 if self._rng.random() < frac else 0))
        return int(round(value))

    def _assign_teacher_envs(self, role: str) -> list[SelectedTeacher]:
        selected = self._selected_entries(role)
        probs = self._softmax_selected(selected)
        n = len(selected)
        out: list[SelectedTeacher] = []
        for (idx, entry), prob in zip(selected, probs):
            raw_envs = float(self.base_env_num) * float(prob) * float(n)
            env_num = self._round_env_count(raw_envs)
            env_num = max(int(self.min_envs_per_teacher), int(env_num))
            if self.max_envs_per_teacher is not None:
                env_num = min(int(self.max_envs_per_teacher), int(env_num))
            out.append(SelectedTeacher(role=role, index=int(idx), entry=entry, prob=float(prob), env_num=int(env_num)))
        if out and all(item.env_num <= 0 for item in out):
            # Avoid a completely empty rollout when min_envs_per_teacher=0.
            best = max(range(len(out)), key=lambda i: out[i].prob)
            out[best].env_num = max(1, int(self.base_env_num))
        return out

    def _make_rollout_env(self, num_envs: int):
        return self._make_env(
            num_envs=int(num_envs),
            observation_mode=self._observation_mode,
            temporal_num_steps=self._temporal_num_steps,
        )

    def _shuffle_rollout_tensordict(self, tensordict):
        """Shuffle concatenated teacher rollouts before PPO.learn().

        Collector outputs have batch shape like [num_envs, time]. PPO.learn()
        computes GAE over the time dimension before it creates randomized
        minibatches. Therefore this runner shuffles only dim=0, which mixes
        teacher/env trajectory rows without corrupting temporal adjacency.
        """

        if tensordict is None or not self.shuffle_rollout_before_ppo:
            return tensordict
        if self.rollout_shuffle_mode == "none":
            return tensordict
        if self.rollout_shuffle_mode != "env":
            raise ValueError(f"unsupported rollout_shuffle_mode: {self.rollout_shuffle_mode}")

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
            device = self.cfg.get("out_device", self.device)
        perm = torch.randperm(n_env_rows, device=device)
        return tensordict[perm]

    def _cat_tensordicts(self, parts: list[Any]):
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        combined = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return self._shuffle_rollout_tensordict(combined)

    def _rollout_output_device(self):
        # Collector-level output device. In CPU-buffer mode this should be CPU so
        # completed rollout TensorDicts do not remain resident on CUDA.
        return self.collector_out_device

    def _store_rollout_buffer(self, data):
        """Move completed rollout data to the configured storage device.

        The collector uses torch.no_grad(), so this is a storage/memory operation,
        not a gradient-preserving transfer. CPU storage lets us collect all teacher
        chunks into one complete mixed batch without holding the whole batch on GPU.
        """

        if data is None:
            return None
        target = self.rollout_storage_device
        if target in {"gpu", "cuda", "cuda:0"}:
            target = "cuda"
        if self.offload_rollout_to_cpu:
            target = "cpu"
        try:
            data = data.detach()
        except Exception:
            pass
        try:
            return data.to(target)
        except TypeError:
            return data.to(torch.device(target))

    def _set_policy_cpu_buffer_mode(self) -> None:
        # The patched PPO file reads these attributes at learn() time. Setting
        # them here is harmless for older policies, but the memory-saving GAE and
        # CPU-buffer minibatch behavior requires replacing gomoku_rl/policy/ppo.py
        # with the companion ppo_cpu_buffer.py file.
        for policy in (self.policy_black, self.policy_white):
            if policy is None:
                continue
            try:
                setattr(policy, "keep_rollout_buffer_on_cpu", bool(self.ppo_learn_from_cpu_buffer))
                setattr(policy, "gae_on_cpu", bool(self.ppo_learn_from_cpu_buffer))
            except Exception:
                pass

    @staticmethod
    def _weighted_mean(values: list[tuple[float, float]]) -> float:
        denom = sum(float(w) for _, w in values)
        if denom <= 0.0:
            return 0.0
        return float(sum(float(v) * float(w) for v, w in values) / denom)

    def _get_applied_mode_for_config(self) -> str:
        if self.train_roles in (self._MODE_BLACK_ONLY, self._MODE_WHITE_ONLY):
            return self.train_roles
        if not self.balance_enabled:
            self.current_bias_mode = self._MODE_BOTH
            return self._MODE_BOTH
        self.current_bias_mode = self._MODE_WHITE_ONLY
        applied_mode = self._MODE_WHITE_ONLY if self.bias_turn_next else self._MODE_BOTH
        self.bias_turn_next = not self.bias_turn_next
        return applied_mode

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
        return self._get_applied_mode_for_config()

    # ------------------------------ learn helpers -----------------------------

    def _learn_black(self, data_black, info: dict[str, Any]) -> None:
        if data_black is None:
            info["policy_black/update_skipped_empty_data"] = 1.0
            return
        self._set_policy_cpu_buffer_mode()
        info.update(add_prefix(self.policy_black.learn(data_black.to_tensordict()), "policy_black/"))

    def _learn_white(self, data_white, info: dict[str, Any]) -> None:
        if data_white is None:
            info["policy_white/update_skipped_empty_data"] = 1.0
            return
        self._set_policy_cpu_buffer_mode()
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

    def _rollout_one_role_against_teachers(
        self,
        *,
        train_role: str,
        teachers: list[SelectedTeacher],
        global_epoch: int,
        info: dict[str, Any],
    ):
        """Collect rollouts for one trainable role against many single-model teachers."""

        if train_role not in {"black", "white"}:
            raise ValueError(f"train_role must be black or white, got {train_role}")

        data_parts: list[Any] = []
        fps_values: list[tuple[float, float]] = []
        win_values: list[tuple[float, float]] = []
        active = [teacher for teacher in teachers if int(teacher.env_num) > 0]
        selected_count = len(teachers)
        active_count = len(active)
        total_envs = int(sum(int(t.env_num) for t in active))
        probs = [float(t.prob) for t in teachers]
        env_counts = [int(t.env_num) for t in teachers]
        prefix = "pool_white" if train_role == "black" else "pool_black"
        teacher_role = "white" if train_role == "black" else "black"

        info.update({
            f"{prefix}/selected_model_count": float(selected_count),
            f"{prefix}/active_model_count": float(active_count),
            f"{prefix}/total_envs": float(total_envs),
            f"{prefix}/base_env_num": float(self.base_env_num),
            f"{prefix}/min_envs_per_teacher": float(self.min_envs_per_teacher),
            f"{prefix}/max_envs_per_teacher": float(self.max_envs_per_teacher or 0),
            f"{prefix}/prob_entropy": self._prob_entropy(probs),
            f"{prefix}/prob_effective_n": self._prob_effective_n(probs),
            f"{prefix}/max_prob": float(max(probs) if probs else 0.0),
            f"{prefix}/min_prob": float(min(probs) if probs else 0.0),
            f"{prefix}/max_envs": float(max(env_counts) if env_counts else 0),
            f"{prefix}/min_envs": float(min(env_counts) if env_counts else 0),
            f"{prefix}/rollout_shuffle_before_ppo": float(self.shuffle_rollout_before_ppo),
            f"{prefix}/rollout_shuffle_env_mode": float(self.rollout_shuffle_mode == "env"),
            f"{prefix}/offload_rollout_to_cpu": float(self.offload_rollout_to_cpu),
            f"{prefix}/ppo_learn_from_cpu_buffer": float(self.ppo_learn_from_cpu_buffer),
            f"{prefix}/teacher_chunk_size": float(self.teacher_chunk_size),
        })

        if not active:
            return None, 0.0

        for chunk_start in range(0, len(active), self.teacher_chunk_size):
            chunk = active[chunk_start : chunk_start + self.teacher_chunk_size]
            for teacher in chunk:
                hist_policy = self._load_role_pool_policy(teacher.entry)
                rollout_env = self._make_rollout_env(teacher.env_num)
                collector = None
                data = None
                try:
                    if train_role == "black":
                        collector = BlackPlayCollector(
                            rollout_env,
                            policy_black=self.policy_black,
                            policy_white=hist_policy,
                            out_device=self._rollout_output_device(),
                            augment=self.cfg.get("augment", False),
                        )
                        data, raw_info = collector.rollout(self.steps)
                        win_key = "black_win"
                    else:
                        collector = WhitePlayCollector(
                            rollout_env,
                            policy_black=hist_policy,
                            policy_white=self.policy_white,
                            out_device=self._rollout_output_device(),
                            augment=self.cfg.get("augment", False),
                        )
                        data, raw_info = collector.rollout(self.steps)
                        win_key = "white_win"

                    data = self._store_rollout_buffer(data)
                    data_parts.append(data)
                    fps_values.append((float(raw_info.get("fps", 0.0)), float(teacher.env_num)))
                    current_win = float(raw_info.get(win_key, 0.0))
                    win_values.append((current_win, float(teacher.env_num)))

                    if self.update_teacher_quality:
                        delta = self.pool.update_after_current_win_rate(
                            role=teacher_role,
                            index=teacher.index,
                            sample_prob=teacher.prob,
                            pool_lr=self.pool_lr,
                            current_win_rate=current_win,
                        )
                        info[f"{prefix}/quality_delta_sum"] = float(
                            info.get(f"{prefix}/quality_delta_sum", 0.0) - float(delta)
                        )

                finally:
                    del data
                    del collector
                    del hist_policy
                    del rollout_env
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Explicit cache clear after each chunk: rollout data has already been
            # moved to CPU, so CUDA only needs temporary policy/env allocations.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        combined = self._cat_tensordicts(data_parts)
        weighted_fps = self._weighted_mean(fps_values)
        weighted_win = self._weighted_mean(win_values)
        if train_role == "black":
            info["pool_white/current_black_win_rate"] = weighted_win
        else:
            info["pool_black/current_white_win_rate"] = weighted_win
        info.setdefault(f"{prefix}/quality_delta_sum", 0.0)
        return combined, weighted_fps

    def _rollout_against_mixed_teachers(self, applied_mode: str, epoch: int) -> dict[str, Any]:
        info: dict[str, Any] = {"train/source_self_play": 0.0, "train/source_mixed_teacher": 1.0}
        fps_parts: list[float] = []
        data_black = None
        data_white = None
        global_epoch = self._completed_epoch(epoch)

        if applied_mode in (self._MODE_BOTH, self._MODE_BLACK_ONLY):
            white_teachers = self._assign_teacher_envs("white")
            data_black, fps_black = self._rollout_one_role_against_teachers(
                train_role="black",
                teachers=white_teachers,
                global_epoch=global_epoch,
                info=info,
            )
            fps_parts.append(float(fps_black))

        if applied_mode in (self._MODE_BOTH, self._MODE_WHITE_ONLY):
            black_teachers = self._assign_teacher_envs("black")
            data_white, fps_white = self._rollout_one_role_against_teachers(
                train_role="white",
                teachers=black_teachers,
                global_epoch=global_epoch,
                info=info,
            )
            fps_parts.append(float(fps_white))

        info["fps"] = float(sum(fps_parts) / len(fps_parts)) if fps_parts else 0.0
        self._apply_learning(applied_mode, data_black, data_white, info)

        self._self_collector.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return info

    def _maybe_add_current_to_pool(self, epoch: int, info: dict[str, Any]) -> None:
        # The curriculum teacher pool is fixed by default. Current checkpoints are
        # saved by ``run`` but are not appended back into ``self.pool``.
        info["pool/current_not_added_to_teacher_pool"] = 1.0

    def _epoch(self, epoch: int) -> dict[str, Any]:
        schedule_info: dict[str, Any] = {}
        self._set_policy_schedules(epoch, schedule_info)

        self._current_train_phase = self._PHASE_MIXED
        self._current_cycle_index = 0
        self._current_bucket_index = self.start_bucket
        self._current_bucket_offset = 0
        self._current_epoch_in_bucket = epoch

        applied_mode = self._get_applied_mode_for_current_epoch()
        info = self._rollout_against_mixed_teachers(applied_mode, epoch)

        info.update(schedule_info)
        info.update({
            "curriculum/bucket_size": float(self.bucket_size),
            "curriculum/start_bucket": float(self.start_bucket),
            "curriculum/end_bucket": float(self.end_bucket),
            "curriculum/selected_bucket_count": float(self.selected_bucket_count),
            "curriculum/available_bucket_count": float(self.available_curriculum_buckets),
            "curriculum/selected_black_teacher_count": float(self.selected_black_teacher_count),
            "curriculum/selected_white_teacher_count": float(self.selected_white_teacher_count),
            "curriculum/base_env_num": float(self.base_env_num),
            "curriculum/min_envs_per_teacher": float(self.min_envs_per_teacher),
            "curriculum/max_envs_per_teacher": float(self.max_envs_per_teacher or 0),
            "curriculum/teacher_prob_temperature": float(self.teacher_prob_temperature),
            "curriculum/offload_rollout_to_cpu": float(self.offload_rollout_to_cpu),
            "curriculum/ppo_learn_from_cpu_buffer": float(self.ppo_learn_from_cpu_buffer),
            "curriculum/teacher_chunk_size": float(self.teacher_chunk_size),
            "curriculum/rollout_storage_device_cpu": float(self.rollout_storage_device == "cpu"),
            "curriculum/env_rounding_round": float(self.env_rounding == "round"),
            "curriculum/env_rounding_floor": float(self.env_rounding == "floor"),
            "curriculum/env_rounding_ceil": float(self.env_rounding == "ceil"),
            "curriculum/env_rounding_stochastic": float(self.env_rounding == "stochastic"),
            "curriculum/train_roles_both": float(self.train_roles == self._MODE_BOTH),
            "curriculum/train_roles_black_only": float(self.train_roles == self._MODE_BLACK_ONLY),
            "curriculum/train_roles_white_only": float(self.train_roles == self._MODE_WHITE_ONLY),
            "curriculum/update_teacher_quality": float(self.update_teacher_quality),
            "curriculum/pool_lr": self.pool_lr,
            "time/local_epoch_completed": float(epoch + 1),
            "time/global_epoch_completed": float(self._completed_epoch(epoch)),
            "time/pretrain_epoch_offset": float(self.pretrain_epoch_offset),
            "pool_black/size": float(self.pool.role_size("black")),
            "pool_black/prob_variance": self.pool.prob_variance("black"),
            "pool_black/bucket_count": float(self.pool.bucket_count("black", self.bucket_size)),
            "pool_white/size": float(self.pool.role_size("white")),
            "pool_white/prob_variance": self.pool.prob_variance("white"),
            "pool_white/bucket_count": float(self.pool.bucket_count("white", self.bucket_size)),
            "elo_eval/interval": float(self.elo_interval),
            "elo_eval/num_models": float(len(self.elo_league)),
            "elo_eval/payoff_coverage": self.elo_league.payoff_coverage(include_self=True),
            "balance/enabled": float(self.balance_enabled),
            "balance/fixed_alternating": float(self.balance_enabled and self.train_roles == self._MODE_BOTH),
            "balance/fixed_bias_white_only": float(self.balance_enabled and self.train_roles == self._MODE_BOTH),
        })

        info.update(self._bias_mode_to_flags(self.current_bias_mode))
        info.update(self._phase_flags())
        info.update(self._balance_trigger_flags())
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
            source="train_mixed_teacher_curriculum",
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
            games_per_pair=max(1, int(getattr(self.eval_env, "num_envs", 1)) * max(1, self.elo_eval_repeats)),
        )
        elapsed = time.perf_counter() - start

        black_rows = ratings.get("black", [])
        white_rows = ratings.get("white", [])
        current_black_row = next((row for row in black_rows if row["id"] == entry.id), None)
        current_white_row = next((row for row in white_rows if row["id"] == entry.id), None)
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

    # ------------------------------- logging ---------------------------------

    def _format_eval_summary(self, info: dict[str, Any]) -> str:
        parts = [
            f"Black vs White:{info['eval/black_vs_white'] * 100.0:.2f}%",
            f"pool_black:{self.pool.role_size('black')} pool_white:{self.pool.role_size('white')}",
            f"range:[{self.start_bucket},{self.end_bucket}) size:{self.bucket_size}",
            f"phase:{self._current_train_phase}",
            f"epoch:{int(info.get('time/global_epoch_completed', 0))}",
            f"base_env:{self.base_env_num}",
            f"Bteach:{int(info.get('pool_black/active_model_count', 0))}/{self.selected_black_teacher_count}",
            f"Wteach:{int(info.get('pool_white/active_model_count', 0))}/{self.selected_white_teacher_count}",
            f"envB:{int(info.get('pool_black/total_envs', 0))}",
            f"envW:{int(info.get('pool_white/total_envs', 0))}",
            f"elo_models:{len(self.elo_league)}",
            f"bias_mode:{self.current_bias_mode}",
            "next_turn:{}".format(
                "biased" if (self.current_bias_mode != self._MODE_BOTH and self.bias_turn_next) else "both"
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
        # Do not rewrite the original teacher-pool metadata when the teacher pool
        # is fixed. We still save a run-local snapshot for reproducibility.
        if self.update_teacher_quality:
            self.pool.save()
        snapshot_path = Path(self.run_dir) / f"teacher_pool_meta_{epoch_label}.json"
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
                    "phase": self._current_train_phase,
                    "Bteach": info.get("pool_black/active_model_count", 0),
                    "Wteach": info.get("pool_white/active_model_count", 0),
                    "envB": info.get("pool_black/total_envs", 0),
                    "envW": info.get("pool_white/total_envs", 0),
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
