from __future__ import annotations

from pathlib import Path

from brainmem.engine import BrainMemEngine
from brainmem.types import IngestInput, RecallRequest


def test_similar_memory_interference_marks_pattern_separation(tmp_path: Path) -> None:
    engine = BrainMemEngine(root=tmp_path)

    first = engine.ingest(
        IngestInput(
            text="Reviewed Atlas budget with Maya and flagged vendor risk for Q2.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="planning",
            emotion="worried",
            action="review",
            novelty=0.8,
            salience=0.8,
            goal_relevance=0.9,
            surprise=0.6,
            unresolved_tension=0.7,
            emphasis=0.8,
        )
    )
    second = engine.ingest(
        IngestInput(
            text="Reviewed Atlas budget with Maya and flagged vendor risk for Q3.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="planning",
            emotion="worried",
            action="review",
            novelty=0.7,
            salience=0.7,
            goal_relevance=0.9,
            surprise=0.5,
            unresolved_tension=0.7,
            emphasis=0.7,
        )
    )
    assert first.durable and second.durable

    results = engine.recall(
        RecallRequest(
            cues=["person:maya", "project:atlas", "token:budget", "token:vendor", "token:risk"],
            people=["Maya"],
            place="unknown",
            tool="laptop",
            mode="planning",
            project="atlas",
            emotion="worried",
            top_k=3,
        )
    )
    assert len(results) >= 2
    assert any(
        match.score_breakdown.get("pattern_separation_required", 0.0) == 1.0
        for match in results
    )
