"""
Shared metadata-driven Elo model pool for InRL and PSRO runners.

The Elo pool is intentionally separate from the training opponent pool:
- checkpoint files live in `elo_models/`;
- model registration and payoff matrix live in `elo_meta.json`;
- if metadata is empty, the runner registers the starting current model once;
- if metadata already has entries, resuming training does not auto-register
  the starting checkpoint again.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import time
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from gomoku_rl.policy import get_policy
from gomoku_rl.policy.common import make_critic, make_ppo_ac, make_ppo_actor
from gomoku_rl.utils.eval import eval_win_rate


sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))


def now() -> float:
    return float(time.time())


def sanitize_id(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_") or "run"


def get_run_id(default_prefix: str = "run") -> str:
    if wandb.run is not None and getattr(wandb.run, "id", None):
        return sanitize_id(str(wandb.run.id))
    return sanitize_id(f"{default_prefix}_{time.strftime('%Y%m%d_%H%M%S')}")


def atomic_json_dump(obj: Any, path: str | os.PathLike[str]) -> None:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)


def wandb_save_file(path: str | os.PathLike[str], *, base_path: str | os.PathLike[str] | None = None) -> None:
    if wandb.run is None:
        return
    try:
        kwargs = {}
        if base_path is not None:
            kwargs["base_path"] = str(base_path)
        wandb.save(str(path), **kwargs)
    except Exception as exc:
        logging.warning("wandb.save failed for %s: %s", path, exc)


class PPOInferencePolicy:
    """Lightweight PPO policy for fixed opponents and Elo evaluation.

    It builds only actor + critic and loads only actor/critic weights. It does
    not construct ClipPPOLoss or optimizer, so fixed Elo/pool opponents do not
    repeatedly create TorchRL loss modules during long training runs.
    """

    def __init__(self, cfg: DictConfig, action_spec, observation_spec, device: Any = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        if bool(cfg.get("share_network")):
            actor_value_operator = make_ppo_ac(
                cfg,
                action_spec=action_spec,
                observation_spec=observation_spec,
                device=self.device,
            )
            self.actor = actor_value_operator.get_policy_operator()
            self.critic = actor_value_operator.get_value_head()
        else:
            self.actor = make_ppo_actor(
                cfg=cfg,
                action_spec=action_spec,
                observation_spec=observation_spec,
                device=self.device,
            )
            self.critic = make_critic(
                cfg=cfg,
                observation_spec=observation_spec,
                device=self.device,
            )

        # Materialize lazy modules before loading checkpoint weights.
        #
        # Important: when cfg.share_network=True, make_ppo_ac() returns an
        # ActorValueOperator whose policy operator writes the shared
        # representation to the TensorDict key ``hidden``. The value head then
        # consumes that same ``hidden`` key. Therefore actor and critic must be
        # initialized on the SAME TensorDict object. Calling
        # ``self.critic(fake_input.clone())`` would create a fresh TensorDict
        # without ``hidden`` and crash with:
        #   KeyError: inputs are None: {'hidden'}
        fake_input = observation_spec.zero()
        try:
            fake_input = fake_input.to(self.device)
        except Exception:
            pass
        if "action_mask" in fake_input.keys():
            fake_input["action_mask"] = ~fake_input["action_mask"]
        with torch.no_grad():
            fake_td = fake_input.clone()
            fake_td = self.actor(fake_td)
            self.critic(fake_td)
        self.eval()
        self.requires_grad_(False)

    def requires_grad_(self, requires_grad: bool = False):
        for module in (self.actor, self.critic):
            for param in module.parameters():
                param.requires_grad_(requires_grad)
        return self

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "actor" not in state_dict or "critic" not in state_dict:
            raise KeyError("checkpoint must contain 'actor' and 'critic' keys")
        self.critic.load_state_dict(state_dict["critic"], strict=False)
        self.actor.load_state_dict(state_dict["actor"])
        self.eval()
        self.requires_grad_(False)

    def train(self):
        self.eval()
        return self

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        return self

    @torch.no_grad()
    def __call__(self, tensordict):
        tensordict = tensordict.to(self.device)
        actor_input = tensordict.select("observation", "action_mask", strict=False)
        actor_output = self.actor(actor_input)
        actor_output = actor_output.exclude("probs")
        tensordict.update(actor_output)
        critic_input = tensordict.select("hidden", "observation", strict=False)
        critic_output = self.critic(critic_input)
        tensordict.update(critic_output)
        return tensordict


@dataclass
class EloEntry:
    """One black/white checkpoint pair used by Elo evaluation."""

    id: str
    epoch: int
    black_path: str
    white_path: str
    created_at: float
    source: str = "train"
    run_id: str = ""
    parent_black_checkpoint: str = ""
    parent_white_checkpoint: str = ""


class GlobalEloLeague:
    """Metadata-driven role-separated Elo league.

    The metadata JSON is the source of truth. It stores both entries and the
    black-vs-white payoff matrix. A separate payoff.json is also written for
    compatibility with older standalone scripts.
    """

    META_SCHEMA_VERSION = 2
    PAYOFF_SCHEMA_VERSION = 1

    def __init__(
        self,
        elo_dir: str | os.PathLike[str],
        meta_path: str | os.PathLike[str] | None,
        *,
        run_id: str,
        prevent_overwrite: bool = True,
    ) -> None:
        self.elo_dir = Path(elo_dir).expanduser().resolve()
        self.elo_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = (
            Path(meta_path).expanduser().resolve()
            if meta_path is not None and str(meta_path).strip()
            else self.elo_dir / "elo_meta.json"
        )
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.payoff_path = self.elo_dir / "payoff.json"
        self.ratings_csv_path = self.elo_dir / "elo_ratings.csv"
        self.ratings_json_path = self.elo_dir / "elo_ratings.json"
        self.run_id = sanitize_id(run_id)
        self.prevent_overwrite = bool(prevent_overwrite)
        self.entries: list[EloEntry] = []
        self.payoff: dict[str, dict[str, dict[str, Any]]] = {}
        self.extra: dict[str, Any] = {}
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
        p_elo = (self.elo_dir / p).resolve()
        if p_elo.exists():
            return str(p_elo)
        return str(Path(to_absolute_path(str(p))).expanduser().resolve())

    def load(self) -> None:
        raw: dict[str, Any] = {}
        if self.meta_path.is_file():
            with self.meta_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)

        self.extra = {
            k: v
            for k, v in raw.items()
            if k not in {"schema_version", "elo_dir", "meta_path", "payoff_path", "updated_at", "entries", "payoff"}
        }

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

        self.payoff = raw.get("payoff", {}) if isinstance(raw.get("payoff", {}), dict) else {}
        if not self.payoff and self.payoff_path.is_file():
            with self.payoff_path.open("r", encoding="utf-8") as f:
                raw_payoff = json.load(f)
            self.payoff = raw_payoff.get("payoff", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.META_SCHEMA_VERSION,
            "elo_dir": str(self.elo_dir),
            "meta_path": str(self.meta_path),
            "payoff_path": str(self.payoff_path),
            "updated_at": now(),
            "entries": [asdict(entry) for entry in self.entries],
            "payoff": self.payoff,
            **self.extra,
        }

    def save(self, path: str | os.PathLike[str] | None = None) -> None:
        out_path = Path(path).expanduser().resolve() if path is not None else self.meta_path
        atomic_json_dump(self.to_dict(), out_path)

    def save_meta(self) -> None:
        self.save(self.meta_path)

    def save_payoff(self) -> None:
        atomic_json_dump(
            {
                "schema_version": self.PAYOFF_SCHEMA_VERSION,
                "elo_dir": str(self.elo_dir),
                "updated_at": now(),
                "payoff": self.payoff,
            },
            self.payoff_path,
        )
        self.save_meta()

    def _target_paths(self, *, epoch: int, prefix: str) -> tuple[str, Path, Path]:
        safe_prefix = sanitize_id(prefix)
        entry_id = f"{safe_prefix}_{self.run_id}_e{epoch:05d}"
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
        source: str = "train",
        parent_black_checkpoint: str = "",
        parent_white_checkpoint: str = "",
    ) -> EloEntry:
        entry_id, black_path, white_path = self._target_paths(epoch=epoch, prefix=prefix)
        if self.has_entry(entry_id):
            for entry in self.entries:
                if entry.id == entry_id:
                    return entry
        if self.prevent_overwrite and (black_path.exists() or white_path.exists()):
            raise FileExistsError(f"Refusing to overwrite Elo checkpoints: {black_path}, {white_path}")

        torch.save(black_state_dict, black_path)
        torch.save(white_state_dict, white_path)
        entry = EloEntry(
            id=entry_id,
            epoch=int(epoch),
            black_path=str(black_path.resolve()),
            white_path=str(white_path.resolve()),
            created_at=now(),
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

    def set_score(self, *, black_id: str, white_id: str, black_win_rate: float, eval_repeats: int) -> None:
        self.payoff.setdefault(black_id, {})[white_id] = {
            "black_win_rate": float(black_win_rate),
            "eval_repeats": int(eval_repeats),
            "updated_at": now(),
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
        n = len(self.entries)
        if n == 0:
            summary = {"num_observations": 0.0, "loss": 0.0, "alpha_logit": 0.0, "black_advantage_elo": 0.0}
            return {"black": [], "white": [], "combined": []}, summary

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
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = (bce * weights).sum() / weights.sum().clamp_min(1.0)
            if l2 > 0:
                loss = loss + float(l2) * (black_skill.square().mean() + white_skill.square().mean() + alpha.square())
            return loss

        best_loss = float("inf")
        best_iter = 0
        best_state: dict[str, torch.Tensor] | None = None
        bad_count = 0
        last_loss = float("inf")
        max_iter = max(1, int(iters))
        patience_i = max(1, int(patience))
        min_delta_f = float(min_delta)
        iteration = 0
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
            "note": "Black and white ratings are fitted jointly from the real black-vs-white payoff matrix.",
        }
        self._save_role_ratings(black_rows, white_rows, combined_rows, summary)
        return {"black": black_rows, "white": white_rows, "combined": combined_rows}, summary

    def _rating_row(self, entry: EloEntry, *, role: str, rating: float, skill: float, rank: int) -> dict[str, Any]:
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
        atomic_json_dump(
            {
                "updated_at": now(),
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
            ["rank", "id", "role", "run_id", "epoch", "elo", "mle_skill_logit", "source", "path", "black_path", "white_path"],
        )
        write_csv(
            self.elo_dir / "white_elo_ratings.csv",
            white_rows,
            ["rank", "id", "role", "run_id", "epoch", "elo", "mle_skill_logit", "source", "path", "black_path", "white_path"],
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


class EloEvalMixin:
    """Mixin that adds shared training-time Elo evaluation to runners.

    The subclass must have these fields from Runner: cfg, env, eval_env,
    policy_black, policy_white, run_dir. It should set `pretrain_epoch_offset`
    before calling `_init_elo_league`.
    """

    def _completed_epoch(self, local_epoch: int) -> int:
        return int(getattr(self, "pretrain_epoch_offset", 0) + local_epoch + 1)

    def _epoch_label(self, epoch_value: int) -> str:
        return f"{int(epoch_value):05d}"

    def _build_inference_policy_from_checkpoint(self, checkpoint_path: str, *, tag: str):
        checkpoint_path = str(checkpoint_path)
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"{tag} not found: {checkpoint_path}")

        if str(self.cfg.algo.name).lower() == "ppo":
            policy = PPOInferencePolicy(
                cfg=self.cfg.algo,
                action_spec=self.env.action_spec,
                observation_spec=self.env.observation_spec,
                device=self.env.device,
            )
            state = torch.load(checkpoint_path, map_location=self.env.device)
            policy.load_state_dict(state)
            logging.info("%s:%s", tag, checkpoint_path)
            return policy

        policy = get_policy(
            name=self.cfg.algo.name,
            cfg=self.cfg.algo,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )
        self._load_policy_checkpoint(policy, checkpoint_path, tag)
        policy.eval()
        for attr in ("loss_module", "optim", "optimizer"):
            if hasattr(policy, attr):
                try:
                    delattr(policy, attr)
                except Exception:
                    pass
        return policy

    def _init_elo_league(self, *, source: str) -> None:
        self.elo_interval = int(self.cfg.get("elo_interval", self.cfg.get("add_to_elo_interval", 100)))
        self.elo_eval_repeats = int(self.cfg.get("elo_eval_repeats", 1))
        self.elo_base = float(self.cfg.get("elo_base", 1200.0))
        self.elo_scale = float(self.cfg.get("elo_scale", 400.0))
        self.elo_mle_iters = int(self.cfg.get("elo_mle_iters", 1000))
        self.elo_mle_lr = float(self.cfg.get("elo_mle_lr", 0.05))
        self.elo_mle_patience = int(self.cfg.get("elo_mle_patience", 100))
        self.elo_l2 = float(self.cfg.get("elo_l2", 1e-4))
        self.elo_source = str(source)

        elo_dir = self.cfg.get("elo_models", self.cfg.get("elo_models_dir", "elo_models"))
        elo_meta = self.cfg.get("elo_e", self.cfg.get("elo_meta", self.cfg.get("elo_models_meta", None)))
        self.elo_league = GlobalEloLeague(
            elo_dir=to_absolute_path(str(elo_dir)),
            meta_path=(to_absolute_path(str(elo_meta)) if elo_meta is not None and str(elo_meta).strip() else None),
            run_id=sanitize_id(get_run_id()),
            prevent_overwrite=True,
        )
        logging.info(
            "Global Elo league loaded: size=%d, dir=%s, meta=%s",
            len(self.elo_league),
            self.elo_league.elo_dir,
            self.elo_league.meta_path,
        )

        if len(self.elo_league) == 0:
            entry = self.elo_league.add_current_pair(
                epoch=int(getattr(self, "pretrain_epoch_offset", 0)),
                black_state_dict=self.policy_black.state_dict(),
                white_state_dict=self.policy_white.state_dict(),
                prefix="elo_init",
                source="pretrained_init" if self.cfg.get("black_checkpoint", None) else source,
                parent_black_checkpoint=str(self.cfg.get("black_checkpoint", "") or ""),
                parent_white_checkpoint=str(self.cfg.get("white_checkpoint", "") or ""),
            )
            logging.info("Added initial current policies to empty Elo league: %s", entry.id)
        else:
            logging.info("Elo metadata is non-empty; current start checkpoint is not auto-added.")

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
            source=self.elo_source,
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
