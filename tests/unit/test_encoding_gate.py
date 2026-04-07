from __future__ import annotations

from brainmem.encoding_gate import score_event_for_encoding
from brainmem.types import EncodingFactors, EventRecord, MemoryState


def _event_with_factors(**overrides: float) -> EventRecord:
    factors = EncodingFactors(
        novelty=0.4,
        emotional_salience=0.3,
        goal_relevance=0.5,
        prediction_error=0.2,
        unresolved_tension=0.3,
        repetition=0.1,
        explicit_emphasis=0.2,
        social_importance=0.1,
        decision_irreversibility=0.1,
    )
    for key, value in overrides.items():
        setattr(factors, key, value)
    return EventRecord(
        event_id="ev-test",
        created_at="2026-04-07T00:00:00+00:00",
        text="Important decision meeting notes",
        people=["Ava"],
        project="atlas",
        place="office",
        tool="laptop",
        mode="planning",
        emotion="focused",
        action="decide",
        cues=["project:atlas"],
        state=MemoryState(location="office"),
        factors=factors,
        source="conversation",
    )


def test_encoding_gate_promotes_high_signal_event() -> None:
    event = _event_with_factors(
        novelty=0.9,
        goal_relevance=0.9,
        emotional_salience=0.8,
        prediction_error=0.7,
        unresolved_tension=0.8,
        explicit_emphasis=0.9,
    )
    decision = score_event_for_encoding(event, threshold=0.55)
    assert decision.promoted is True
    assert decision.score >= 0.55
    assert "threshold" in decision.reason
    assert "goal_relevance" in decision.weighted_factors


def test_encoding_gate_rejects_low_signal_event() -> None:
    event = _event_with_factors(
        novelty=0.05,
        emotional_salience=0.05,
        goal_relevance=0.05,
        prediction_error=0.05,
        unresolved_tension=0.0,
        explicit_emphasis=0.0,
    )
    decision = score_event_for_encoding(event, threshold=0.55)
    assert decision.promoted is False
    assert decision.score < 0.55
