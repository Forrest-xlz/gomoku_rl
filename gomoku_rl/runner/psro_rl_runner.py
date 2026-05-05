"""PSRO-style Dota/OpenAI-Five historical-pool runner.

This module contains the full training logic for `scripts/train_psro.py`.
It keeps PSRO code in the same runner style as `IndependentRLRunner`: the
script file is only a Hydra entry point, while the runner owns rollout, update,
logging, checkpointing, model-pool snapshots, and Elo evaluation.

Baseline evaluation has been removed. Training-time evaluation uses the shared
metadata-driven Elo model pool from `elo_model_pool.py`.
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
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.utils.misc import add_prefix
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


@dataclass
class RolePoolEntry:
    """One historical checkpoint for one role in the training pool.

    role="black" entries are historical black policies used as opponents for current_white.
    role="white" entries are historical white policies used as opponents for current_black.
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
    """Independent RL runner with Dota/OpenAI-Five-style model pool and global Elo."""

    _MODE_BOTH = "both"
    _MODE_BLACK_ONLY = "black_only"
    _MODE_WHITE_ONLY = "white_only"

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
        self.self_play_prob = float(self.cfg.get("p", self.cfg.get("self_play_prob", 0.8)))
        self.pool_lr = float(self.cfg.get("l", self.cfg.get("pool_lr", 0.01)))
        self.add_to_pool_interval = int(self.cfg.get("m", self.cfg.get("add_to_pool_interval", 10)))

        pool_dir = self.cfg.get("f", self.cfg.get("model_pool_dir", "model_pool"))
        meta_path = self.cfg.get("e", self.cfg.get("model_pool_meta", None))
        if not (0.0 <= self.self_play_prob <= 1.0):
            raise ValueError(f"p/self_play_prob must be in [0, 1], got {self.self_play_prob}")
        if self.pool_lr < 0.0:
            raise ValueError(f"l/pool_lr must be >= 0, got {self.pool_lr}")

        self.pool = RoleSeparatedDiskPolicyPool(
            pool_dir=to_absolute_path(str(pool_dir)),
            meta_path=(to_absolute_path(str(meta_path)) if meta_path is not None and str(meta_path).strip() else None),
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
        self.balance_lower = float(balance_cfg.get("lower", 0.40))
        self.balance_upper = float(balance_cfg.get("upper", 0.60))
        self.balance_ema_alpha = float(balance_cfg.get("ema_alpha", 0.20))
        if not (0.0 <= self.balance_lower < self.balance_upper <= 1.0):
            raise ValueError(f"invalid balance bounds: lower={self.balance_lower}, upper={self.balance_upper}")
        if not (0.0 < self.balance_ema_alpha <= 1.0):
            raise ValueError(f"invalid balance ema_alpha: {self.balance_ema_alpha}, expected in (0, 1]")

        self.black_vs_white_ema: float | None = None
        self.current_bias_mode = self._MODE_BOTH
        self.bias_turn_next = False

    # ---------------------------- epoch helpers ----------------------------

    def _completed_epoch(self, local_epoch: int) -> int:
        return int(self.pretrain_epoch_offset + local_epoch + 1)

    def _checkpoint_epoch(self, local_epoch: int) -> int:
        return int(self.pretrain_epoch_offset + local_epoch + 1)

    def _epoch_label(self, epoch_value: int) -> str:
        return f"{int(epoch_value):05d}"

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
        if self.current_bias_mode == self._MODE_BOTH:
            return {"balance/bias_active": 0.0, "balance/next_turn_biased": 0.0, "balance/next_turn_both": 1.0}
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
            ema = self.balance_ema_alpha * black_vs_white_raw + (1.0 - self.balance_ema_alpha) * self.black_vs_white_ema
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
        info["fps"] = info.get("self_play/fps", 0.0)
        info.pop("self_play/fps", None)
        self._apply_learning(applied_mode, data_black, data_white, info)
        return info

    def _load_role_pool_policy(self, entry: RolePoolEntry):
        if not Path(entry.path).is_file():
            raise FileNotFoundError(f"pool {entry.role} checkpoint not found: {entry.path}")
        return self._build_inference_policy_from_checkpoint(entry.path, tag=f"pool_{entry.role}/{entry.id}")

    def _rollout_against_pool(self, applied_mode: str, epoch: int) -> dict[str, Any]:
        if self.pool.role_size("black") == 0 or self.pool.role_size("white") == 0:
            self.pool.ensure_non_empty(
                black_state_dict=self.policy_black.state_dict(),
                white_state_dict=self.policy_white.state_dict(),
                epoch=self.pretrain_epoch_offset,
            )

        info: dict[str, Any] = {"train/source_self_play": 0.0}
        fps_parts: list[float] = []
        data_black = None
        data_white = None
        sampled_policies: list[Any] = []
        global_epoch = self._completed_epoch(epoch)

        # Train current black against a sampled historical white.
        if applied_mode in (self._MODE_BOTH, self._MODE_BLACK_ONLY):
            white_idx, white_entry, white_sample_prob = self.pool.sample("white", self._rng)
            hist_white = self._load_role_pool_policy(white_entry)
            sampled_policies.append(hist_white)
            info.update(
                {
                    "pool_white/sampled_index": float(white_idx),
                    "pool_white/sampled_epoch": float(white_entry.epoch),
                    "pool_white/sampled_relative_epoch_ratio": float((white_entry.epoch - global_epoch) / max(global_epoch, 1)),
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
                    "pool_black/sampled_relative_epoch_ratio": float((black_entry.epoch - global_epoch) / max(global_epoch, 1)),
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
        logging.info("Added current policies to role-separated training pool: black=%s white=%s", black_entry.id, white_entry.id)

    def _epoch(self, epoch: int) -> dict[str, Any]:
        schedule_info: dict[str, Any] = {}
        self._set_policy_schedules(epoch, schedule_info)

        applied_mode = self._get_applied_mode_for_current_epoch()
        use_self_play = (self._rng.random() < self.self_play_prob) or len(self.pool) == 0
        if use_self_play:
            info = self._rollout_current_self_play(applied_mode)
        else:
            info = self._rollout_against_pool(applied_mode, epoch)

        info.update(schedule_info)
        info.update(
            {
                "psro/self_play_prob": self.self_play_prob,
                "psro/pool_lr": self.pool_lr,
                "psro/add_to_pool_interval": float(self.add_to_pool_interval),
                "time/local_epoch_completed": float(epoch + 1),
                "time/global_epoch_completed": float(self._completed_epoch(epoch)),
                "time/pretrain_epoch_offset": float(self.pretrain_epoch_offset),
                "pool_black/size": float(self.pool.role_size("black")),
                "pool_black/prob_variance": self.pool.prob_variance("black"),
                "pool_white/size": float(self.pool.role_size("white")),
                "pool_white/prob_variance": self.pool.prob_variance("white"),
                "elo_eval/interval": float(self.elo_interval),
                "elo_eval/num_models": float(len(self.elo_league)),
                "elo_eval/payoff_coverage": self.elo_league.payoff_coverage(include_self=True),
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
            "Global role Elo updated at epoch=%d: current=%s black=%.2f white=%.2f black_rank=%s white_rank=%s alpha=%.2f models=%d evaluated_pairs=%d",
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
            f"EMA:{info['eval/black_vs_white_ema'] * 100.0:.2f}%",
            f"pool_black:{self.pool.role_size('black')} pool_white:{self.pool.role_size('white')}",
            f"elo_models:{len(self.elo_league)}",
            f"p_self:{self.self_play_prob:.3f}",
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
                    "elo_eval/payoff_coverage": self.elo_league.payoff_coverage(include_self=True),
                }
            )
            info.update(self._ema_trigger_flags())
            info.update(self._bias_mode_to_flags(self.current_bias_mode))
            info.update(self._phase_flags())

        self._maybe_update_global_elo(epoch, info)

        if epoch % self.log_interval == 0 or "elo_eval/current_black" in info:
            if "eval/black_vs_white" not in info:
                black_vs_white_raw = float(eval_win_rate(self.eval_env, player_black=self.policy_black, player_white=self.policy_white))
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

            # Save after completed local epochs. With offset=1000 and save_interval=100,
            # the first periodic checkpoint is black_01100.pt / white_01100.pt.
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
                    "eloN": len(self.elo_league),
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


