from __future__ import annotations

from brainmem.engine import BrainMemEngine
from brainmem.types import IngestInput, RecallRequest


def test_state_dependent_recall_prefers_matching_mode(tmp_path) -> None:
    engine = BrainMemEngine(root=tmp_path)
    morning = engine.ingest(
        IngestInput(
            text="Morning planning for Atlas budget and vendor call.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="planning",
            emotion="focused",
            action="plan",
            novelty=0.8,
            salience=0.7,
            goal_relevance=0.9,
            surprise=0.4,
            unresolved_tension=0.6,
            emphasis=0.7,
        )
    )
    evening = engine.ingest(
        IngestInput(
            text="Evening reflection on Atlas budget outcomes and lessons learned.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="reflection",
            emotion="neutral",
            action="review",
            novelty=0.75,
            salience=0.65,
            goal_relevance=0.85,
            surprise=0.45,
            unresolved_tension=0.55,
            emphasis=0.55,
        )
    )
    assert morning.durable and evening.durable

    planning_results = engine.recall(
        RecallRequest(
            cues=["project:atlas", "token:budget", "person:maya"],
            people=["Maya"],
            place="unknown",
            tool="laptop",
            mode="planning",
            project="atlas",
            emotion="focused",
            top_k=2,
        )
    )
    assert planning_results
    assert planning_results[0].state["processing_mode"] == "planning"
