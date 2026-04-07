from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EventBoundaryConfig:
    threshold: float = 0.55
    people_weight: float = 0.2
    project_weight: float = 0.2
    tool_weight: float = 0.15
    place_weight: float = 0.1
    emotion_weight: float = 0.1
    mode_weight: float = 0.1
    action_weight: float = 0.1
    surprise_weight: float = 0.05


@dataclass(slots=True)
class SegmentDecision:
    event: dict[str, Any]
    segment_id: str
    boundary_score: float
    boundary_reasons: list[str]
    is_new_segment: bool


def compute_boundary_score(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    config: EventBoundaryConfig | None = None,
) -> float:
    cfg = config or EventBoundaryConfig()
    people_prev = set(_as_list(previous.get("people")))
    people_curr = set(_as_list(current.get("people")))
    people_changed = 1.0 if people_prev != people_curr else 0.0

    project_changed = _changed(previous.get("project"), current.get("project"))
    tool_changed = _changed(previous.get("tool"), current.get("tool"))
    place_changed = _changed(previous.get("place"), current.get("place"))
    emotion_changed = _changed(previous.get("emotion"), current.get("emotion"))
    mode_changed = _changed(previous.get("mode"), current.get("mode"))
    action_changed = _changed(previous.get("action"), current.get("action"))
    surprise = float(current.get("factors", {}).get("prediction_error", current.get("surprise", 0.0)) or 0.0)

    score = (
        cfg.people_weight * people_changed
        + cfg.project_weight * project_changed
        + cfg.tool_weight * tool_changed
        + cfg.place_weight * place_changed
        + cfg.emotion_weight * emotion_changed
        + cfg.mode_weight * mode_changed
        + cfg.action_weight * action_changed
        + cfg.surprise_weight * _clip01(surprise)
    )
    return round(score, 6)


def _boundary_reasons(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if set(_as_list(previous.get("people"))) != set(_as_list(current.get("people"))):
        reasons.append("people_changed")
    if _changed(previous.get("project"), current.get("project")):
        reasons.append("project_changed")
    if _changed(previous.get("tool"), current.get("tool")):
        reasons.append("tool_changed")
    if _changed(previous.get("place"), current.get("place")):
        reasons.append("place_changed")
    if _changed(previous.get("emotion"), current.get("emotion")):
        reasons.append("emotion_changed")
    if _changed(previous.get("mode"), current.get("mode")):
        reasons.append("mode_changed")
    if _changed(previous.get("action"), current.get("action")):
        reasons.append("action_changed")
    surprise = float(current.get("factors", {}).get("prediction_error", current.get("surprise", 0.0)) or 0.0)
    if _clip01(surprise) >= 0.6:
        reasons.append("surprise_spike")
    return reasons


def segment_events(
    records: list[dict[str, Any]],
    *,
    boundary_threshold: float = 0.55,
    config: EventBoundaryConfig | None = None,
) -> list[SegmentDecision]:
    if not records:
        return []
    cfg = config or EventBoundaryConfig(threshold=boundary_threshold)

    decisions: list[SegmentDecision] = []
    segment_number = 1
    decisions.append(
        SegmentDecision(
            event=records[0],
            segment_id=f"seg-{segment_number:03d}",
            boundary_score=0.0,
            boundary_reasons=["initial_event"],
            is_new_segment=True,
        )
    )
    for idx in range(1, len(records)):
        previous = records[idx - 1]
        current = records[idx]
        score = compute_boundary_score(previous, current, config=cfg)
        reasons = _boundary_reasons(previous, current)
        is_new = score >= cfg.threshold
        if is_new:
            segment_number += 1
        decisions.append(
            SegmentDecision(
                event=current,
                segment_id=f"seg-{segment_number:03d}",
                boundary_score=score,
                boundary_reasons=reasons or ["low_shift"],
                is_new_segment=is_new,
            )
        )
    return decisions


def segment_stream_records(
    records: list[dict[str, Any]],
    *,
    boundary_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    decisions = segment_events(records, boundary_threshold=boundary_threshold)
    if not decisions:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for decision in decisions:
        grouped.setdefault(decision.segment_id, []).append(decision.event)
    return [
        {
            "segment_id": segment_id,
            "events": grouped[segment_id],
        }
        for segment_id in sorted(grouped.keys())
    ]


def _changed(previous: Any, current: Any) -> float:
    return 1.0 if str(previous or "").lower() != str(current or "").lower() else 0.0


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    if value in (None, ""):
        return []
    return [str(value).strip().lower()]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


boundary_score = compute_boundary_score
