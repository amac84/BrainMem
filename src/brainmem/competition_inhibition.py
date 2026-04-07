from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class InhibitionPolicy:
    loser_penalty: float = 0.15
    min_strength: float = 0.05
    default_strength: float = 0.55


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

    competitors = competition_sets.get(cue, [])
    for event_id in competitors:
        if event_id == winner_event_id:
            continue
        current = float(strength_table.get(event_id, {}).get("strength", cfg.default_strength))
        updated = max(cfg.min_strength, round(current - cfg.loser_penalty, 6))
        strength_table.setdefault(event_id, {})["strength"] = updated
        affected[event_id] = updated
    return affected


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
