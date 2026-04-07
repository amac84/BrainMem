from __future__ import annotations

from brainmem.engine import BrainMemEngine
from brainmem.types import IngestInput


def test_planning_rollout_returns_evidence_backed_actions(tmp_path) -> None:
    engine = BrainMemEngine(root=tmp_path)

    # Build a short transition sequence for atlas planning flow.
    first = engine.ingest(
        IngestInput(
            text="Plan Atlas vendor sync agenda.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="planning",
            emotion="focused",
            action="plan",
            novelty=0.8,
            salience=0.7,
            goal_relevance=0.9,
            surprise=0.5,
            unresolved_tension=0.6,
            emphasis=0.6,
        )
    )
    second = engine.ingest(
        IngestInput(
            text="Call vendor and confirm dependencies.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="execution",
            emotion="focused",
            action="call",
            novelty=0.7,
            salience=0.7,
            goal_relevance=0.9,
            surprise=0.4,
            unresolved_tension=0.6,
            emphasis=0.6,
        )
    )
    third = engine.ingest(
        IngestInput(
            text="Review handoff notes and close loop.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="reflection",
            emotion="happy",
            action="review",
            novelty=0.6,
            salience=0.6,
            goal_relevance=0.8,
            surprise=0.3,
            unresolved_tension=0.3,
            emphasis=0.4,
        )
    )
    if not third.durable:
        # Ensure a transition from planning -> execution exists for the planner graph.
        third = engine.ingest(
            IngestInput(
                text="Review handoff notes and close loop.",
                people=["Maya"],
                project="atlas",
                tool="laptop",
                mode="reflection",
                emotion="happy",
                action="review",
                novelty=0.8,
                salience=0.7,
                goal_relevance=0.9,
                surprise=0.5,
                unresolved_tension=0.6,
                emphasis=0.7,
            )
        )
    assert first.durable and second.durable and third.durable

    sim_result = engine.simulate(
        current_state="project:atlas|mode:planning|emotion:focused",
        goal_hint="atlas",
        depth=2,
        top_k=2,
    )
    proposals = sim_result["proposals"]
    assert proposals
    top = proposals[0]
    assert top["action_sequence"]
    assert top["evidence"]
    assert top["score"] > 0
