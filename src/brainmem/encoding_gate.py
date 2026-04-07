"""Selective encoding gate logic."""

from __future__ import annotations

from .config import DEFAULT_ENCODING_THRESHOLD, DEFAULT_ENCODING_WEIGHTS
from .types import EncodingDecision, EventRecord


def score_event_for_encoding(
    event: EventRecord,
    threshold: float = DEFAULT_ENCODING_THRESHOLD,
    weights: dict[str, float] | None = None,
) -> EncodingDecision:
    """Compute transparent encoding score for an event record."""
    weights = weights or DEFAULT_ENCODING_WEIGHTS
    factors = event.factors

    weighted = {
        "novelty": factors.novelty * weights["novelty"],
        "emotional_salience": factors.emotional_salience * weights["emotional_salience"],
        "goal_relevance": factors.goal_relevance * weights["goal_relevance"],
        "prediction_error": factors.prediction_error * weights["prediction_error"],
        "unresolved_tension": factors.unresolved_tension * weights["unresolved_tension"],
        "repetition": factors.repetition * weights["repetition"],
        "explicit_emphasis": factors.explicit_emphasis * weights["explicit_emphasis"],
        "social_importance": factors.social_importance * weights["social_importance"],
        "decision_irreversibility": factors.decision_irreversibility * weights["decision_irreversibility"],
    }
    score = max(0.0, min(1.0, round(sum(weighted.values()), 6)))
    promoted = score >= threshold
    reason = (
        f"score {score:.3f} >= threshold {threshold:.3f}"
        if promoted
        else f"score {score:.3f} < threshold {threshold:.3f}"
    )
    return EncodingDecision(
        score=score,
        threshold=threshold,
        promoted=promoted,
        weighted_factors={k: round(v, 6) for k, v in weighted.items()},
        reason=reason,
    )


def should_promote(score: float, threshold: float | None = None) -> bool:
    cutoff = DEFAULT_ENCODING_THRESHOLD if threshold is None else threshold
    return score >= cutoff


def encoding_decision(event: EventRecord, threshold: float) -> EncodingDecision:
    return score_event_for_encoding(event, threshold=threshold)
