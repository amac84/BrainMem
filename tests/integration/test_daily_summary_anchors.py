from __future__ import annotations

from pathlib import Path

from brainmem.consolidation_job import ConsolidationJob
from brainmem.engine import BrainMemEngine
from brainmem.types import IngestInput


def test_consolidation_creates_daily_summary_with_anchors(tmp_path: Path) -> None:
    engine = BrainMemEngine(root=tmp_path)
    engine.ingest(
        IngestInput(
            text="Discussed Atlas budget risks with Maya and identified unresolved vendor dependency.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="planning",
            emotion="worried",
            action="review",
            novelty=0.8,
            salience=0.8,
            goal_relevance=0.9,
            surprise=0.7,
            unresolved_tension=0.9,
            emphasis=0.8,
        )
    )
    engine.ingest(
        IngestInput(
            text="Switched to roadmap drafting for Atlas launch milestones and assigned next action to call vendor.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="planning",
            emotion="focused",
            action="plan",
            novelty=0.7,
            salience=0.7,
            goal_relevance=0.9,
            surprise=0.5,
            unresolved_tension=0.8,
            emphasis=0.7,
        )
    )

    result = ConsolidationJob(root=tmp_path).run_daily()
    assert result["events_considered"] >= 2
    assert result["segments"] >= 1
    summary_path = Path(result["summary_path"])
    assert summary_path.exists()

    text = summary_path.read_text(encoding="utf-8")
    assert "anchors:" in text
    assert "#event_id=" in text
    assert "Daily gist" in text
