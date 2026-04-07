from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import DEFAULT_RECALL_WEIGHTS
from .types import RecallMatch, RecallRequest


def _weighted_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    lset = set(left)
    rset = set(right)
    if not lset and not rset:
        return 0.0
    intersection = len(lset & rset)
    union = len(lset | rset)
    return intersection / union if union else 0.0


def _state_match(request: RecallRequest, candidate_state: dict[str, str]) -> float:
    checks = (
        ("location", request.place),
        ("device_context", request.tool),
        ("social_context", "social" if request.people else "alone"),
        ("processing_mode", request.mode),
    )
    score = 0.0
    considered = 0
    for key, expected in checks:
        if not expected or expected == "unknown":
            continue
        considered += 1
        if candidate_state.get(key, "unknown") == expected:
            score += 1.0
    return score / considered if considered else 0.0


def _recency_bonus(created_at: str) -> float:
    if not created_at:
        return 0.0
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    age_hours = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 3600.0)
    if age_hours <= 24:
        return 1.0
    if age_hours <= 72:
        return 0.5
    if age_hours <= 168:
        return 0.2
    return 0.0


def score_candidate(
    request: RecallRequest,
    *,
    memory_id: str,
    path: Path,
    cues: list[str],
    state: dict[str, str],
    anchor: str,
    excerpt: str,
    created_at: str = "",
    weights: dict[str, float] | None = None,
) -> RecallMatch:
    w = weights or DEFAULT_RECALL_WEIGHTS
    query_cues = set(request.cues)
    candidate_cues = set(cues)
    if "status:open_loop" in candidate_cues:
        candidate_cues = set(candidate_cues)
        candidate_cues.discard("status:open_loop")
    cue_overlap = _weighted_jaccard(query_cues, candidate_cues)
    state_match = _state_match(request, state)
    goal_relevance = 1.0 if f"project:{request.project.lower()}" in cues else 0.0
    open_loop_activation = 1.0 if "status:open_loop" in cues else 0.0
    recency_bonus = _recency_bonus(created_at)

    total = (
        w["cue_overlap"] * cue_overlap
        + w["state_match"] * state_match
        + w["goal_relevance"] * goal_relevance
        + w["open_loop_activation"] * open_loop_activation
        + w["recency_bonus"] * recency_bonus
    )

    return RecallMatch(
        memory_id=memory_id,
        path=path,
        total_score=round(total, 6),
        score_breakdown={
            "cue_overlap": round(cue_overlap, 4),
            "state_match": round(state_match, 4),
            "goal_relevance": round(goal_relevance, 4),
            "open_loop_activation": round(open_loop_activation, 4),
            "recency_bonus": round(recency_bonus, 4),
        },
        anchor=anchor,
        excerpt=excerpt,
        cues=list(cues),
        state=dict(state),
    )
