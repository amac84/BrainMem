from __future__ import annotations

from brainmem.event_segmentation import compute_boundary_score, segment_stream_records


def test_boundary_score_rises_on_context_shift() -> None:
    first = {
        "people": ["Maya"],
        "project": "Atlas",
        "tool": "laptop",
        "place": "office",
        "mode": "planning",
        "emotion": "worried",
    }
    second = {
        "people": ["Noah"],
        "project": "Home",
        "tool": "phone",
        "place": "home",
        "mode": "execution",
        "emotion": "neutral",
    }
    score = compute_boundary_score(first, second)
    assert score >= 0.6


def test_segment_stream_records_splits_on_large_boundary() -> None:
    records = [
        {
            "event_id": "ev-1",
            "created_at": "2026-04-07T08:00:00+00:00",
            "people": ["Maya"],
            "project": "Atlas",
            "tool": "laptop",
            "place": "office",
            "mode": "planning",
            "emotion": "worried",
            "text": "Budget review with Maya",
            "source": "conversation",
        },
        {
            "event_id": "ev-2",
            "created_at": "2026-04-07T08:20:00+00:00",
            "people": ["Maya"],
            "project": "Atlas",
            "tool": "laptop",
            "place": "office",
            "mode": "planning",
            "emotion": "focused",
            "text": "Action items from budget review",
            "source": "conversation",
        },
        {
            "event_id": "ev-3",
            "created_at": "2026-04-07T09:00:00+00:00",
            "people": [],
            "project": "Home",
            "tool": "phone",
            "place": "home",
            "mode": "execution",
            "emotion": "neutral",
            "text": "Book dentist appointment",
            "source": "conversation",
        },
    ]
    segments = segment_stream_records(records, boundary_threshold=0.45)
    assert len(segments) == 2
    assert [event["event_id"] for event in segments[0]["events"]] == ["ev-1", "ev-2"]
    assert [event["event_id"] for event in segments[1]["events"]] == ["ev-3"]
