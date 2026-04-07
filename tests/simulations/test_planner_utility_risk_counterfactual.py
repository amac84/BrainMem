from __future__ import annotations

from brainmem.engine import BrainMemEngine
from brainmem.types import IngestInput


def test_planner_reports_utility_risk_uncertainty_and_counterfactual(tmp_path) -> None:
    engine = BrainMemEngine(root=tmp_path)

    # Build several transitions so uncertainty and risk are meaningful.
    engine.ingest(
        IngestInput(
            text="Plan Atlas vendor agenda.",
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
    engine.ingest(
        IngestInput(
            text="Call vendor and close dependency risk.",
            people=["Maya"],
            project="atlas",
            tool="phone",
            mode="execution",
            emotion="focused",
            action="call",
            novelty=0.7,
            salience=0.7,
            goal_relevance=0.9,
            surprise=0.4,
            unresolved_tension=0.5,
            emphasis=0.6,
        )
    )
    engine.ingest(
        IngestInput(
            text="Review outcomes and update plan.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="reflection",
            emotion="happy",
            action="review",
            novelty=0.7,
            salience=0.6,
            goal_relevance=0.8,
            surprise=0.3,
            unresolved_tension=0.3,
            emphasis=0.5,
        )
    )

    sim = engine.simulate(
        current_state="project:atlas|mode:planning|emotion:focused",
        goal_hint="atlas",
        depth=2,
        top_k=2,
    )
    assert sim["proposals"]
    top = sim["proposals"][0]
    assert "utility" in top
    assert "risk" in top
    assert "uncertainty" in top
    assert "counterfactual" in top
    assert top["evidence"]
