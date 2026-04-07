from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .competition_inhibition import InhibitionPolicy, recover_inhibited_strengths
from .markdown_io import read_markdown, write_markdown


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


@dataclass(slots=True)
class ForgettingPolicy:
    decay_lambda_per_day: float = 0.08
    prune_threshold: float = 0.18
    archive_retention_days: int = 30
    renormalize_target_mean: float = 0.5
    inhibition_recovery_cap_days: int = 10


@dataclass(slots=True)
class ForgettingResult:
    decayed: int
    archived: int
    retained: int
    renormalized: int

    def as_dict(self) -> dict[str, int]:
        return {
            "decayed": self.decayed,
            "archived": self.archived,
            "retained": self.retained,
            "renormalized": self.renormalized,
        }


def run_forgetting_pass(
    *,
    events_dir: Path,
    strength_table_path: Path,
    archive_dir: Path,
    policy: ForgettingPolicy | None = None,
    inhibition_policy: InhibitionPolicy | None = None,
) -> ForgettingResult:
    cfg = policy or ForgettingPolicy()
    inh_cfg = inhibition_policy or InhibitionPolicy()
    strength_table = _read_json(strength_table_path, default={})

    decayed = 0
    archived = 0
    retained = 0
    renormalized = 0

    now = _utc_now()
    archive_dir.mkdir(parents=True, exist_ok=True)

    for event_id, payload in list(strength_table.items()):
        path = Path(str(payload.get("path", "")))
        strength = float(payload.get("strength", 0.55))
        updated_at = _parse_iso(str(payload.get("updated_at", ""))) or now
        age_days = max(0.0, (now - updated_at).total_seconds() / 86400.0)

        last_inhibited_at = _parse_iso(str(payload.get("last_inhibited_at", "")))
        if float(payload.get("inhibition_level", 0.0)) > 0 and last_inhibited_at is not None:
            recover_days = max(0.0, (now - last_inhibited_at).total_seconds() / 86400.0)
            recover_days = min(float(cfg.inhibition_recovery_cap_days), recover_days)
            if recover_days > 0:
                recover_inhibited_strengths(
                    strength_table={event_id: payload},
                    elapsed_days=recover_days,
                    policy=inh_cfg,
                )
                strength = float(payload.get("strength", strength))

        decayed_strength = strength * (2.718281828 ** (-cfg.decay_lambda_per_day * age_days))
        decayed_strength = round(max(0.0, min(1.0, decayed_strength)), 6)
        if decayed_strength != strength:
            decayed += 1

        payload["strength"] = decayed_strength
        payload["updated_at"] = now.replace(microsecond=0).isoformat()

        if decayed_strength < cfg.prune_threshold and path.exists():
            archived_path = archive_dir / path.name
            metadata, body = read_markdown(path)
            metadata["archived_at"] = now.replace(microsecond=0).isoformat()
            metadata["archive_reason"] = f"strength<{cfg.prune_threshold}"
            write_markdown(archived_path, metadata, body)
            path.unlink()
            payload["archived_path"] = str(archived_path)
            payload["archived"] = True
            archived += 1
        else:
            retained += 1

    # renormalize active strengths to target mean for stability
    active_strengths = [
        float(v.get("strength", 0.0))
        for v in strength_table.values()
        if not bool(v.get("archived", False))
    ]
    if active_strengths:
        mean_strength = sum(active_strengths) / len(active_strengths)
        if mean_strength > 0:
            scale = cfg.renormalize_target_mean / mean_strength
            for payload in strength_table.values():
                if bool(payload.get("archived", False)):
                    continue
                old = float(payload.get("strength", 0.0))
                new = round(max(0.0, min(1.0, old * scale)), 6)
                if new != old:
                    renormalized += 1
                payload["strength"] = new

    _write_json(strength_table_path, strength_table)
    return ForgettingResult(
        decayed=decayed,
        archived=archived,
        retained=retained,
        renormalized=renormalized,
    )


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return default
    return json.loads(raw)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
