from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class InhibitionPolicy:
    loser_penalty: float = 0.15
    min_strength: float = 0.05
    default_strength: float = 0.55
    recovery_rate_per_day: float = 0.06
    rebound_strength_factor: float = 0.4


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_competition_sets(cue_to_events: dict[str, list[str]]) -> dict[str, list[str]]:
    competition: dict[str, list[str]] = {}
    for cue, events in cue_to_events.items():
        unique = sorted(set(events))
        if len(unique) <= 1:
            continue
        competition[cue] = unique
    return competition


def apply_competitor_inhibition(
    *,
    winner_event_id: str,
    cue: str,
    competition_sets: dict[str, list[str]],
    strength_table: dict[str, dict[str, Any]],
    policy: InhibitionPolicy | None = None,
) -> dict[str, float]:
    cfg = policy or InhibitionPolicy()
    affected: dict[str, float] = {}
    now_iso = _utc_iso()

    competitors = competition_sets.get(cue, [])
    for event_id in competitors:
        if event_id == winner_event_id:
            continue
        entry = strength_table.setdefault(event_id, {})
        current = float(entry.get("strength", cfg.default_strength))
        updated = max(cfg.min_strength, round(current - cfg.loser_penalty, 6))
        inhibition_level = float(entry.get("inhibition_level", 0.0))
        inhibition_level = round(min(1.0, inhibition_level + cfg.loser_penalty), 6)
        entry["strength"] = updated
        entry["inhibition_level"] = inhibition_level
        entry["last_inhibited_at"] = now_iso
        affected[event_id] = updated
    return affected


def recover_inhibited_strengths(
    *,
    strength_table: dict[str, dict[str, Any]],
    elapsed_days: float,
    policy: InhibitionPolicy | None = None,
) -> dict[str, float]:
    cfg = policy or InhibitionPolicy()
    recovered: dict[str, float] = {}
    if elapsed_days <= 0:
        return recovered
    for event_id, entry in strength_table.items():
        inhibition_level = float(entry.get("inhibition_level", 0.0))
        if inhibition_level <= 0:
            continue
        recovery_amount = min(inhibition_level, cfg.recovery_rate_per_day * elapsed_days)
        if recovery_amount <= 0:
            continue
        inhibition_after = round(max(0.0, inhibition_level - recovery_amount), 6)
        strength_before = float(entry.get("strength", cfg.default_strength))
        rebound = recovery_amount * cfg.rebound_strength_factor
        strength_after = round(min(1.0, strength_before + rebound), 6)
        entry["inhibition_level"] = inhibition_after
        entry["strength"] = strength_after
        if inhibition_after == 0.0:
            entry["last_inhibited_at"] = ""
        recovered[event_id] = strength_after
    return recovered


def load_competition_sets(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    return {str(k): [str(v) for v in values] for k, values in dict(data).items()}


def write_competition_sets(path: Path, data: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
