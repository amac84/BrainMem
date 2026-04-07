from __future__ import annotations

from pathlib import Path

from brainmem.engine import BrainMemEngine
from brainmem.reconsolidation import queue_patch
from brainmem.types import IngestInput


def test_reconsolidation_commit_and_conflict_resolution(tmp_path: Path) -> None:
    engine = BrainMemEngine(root=tmp_path)
    candidate = engine.ingest(
        IngestInput(
            text="Finalize Atlas vendor handoff scope with Maya.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="planning",
            emotion="worried",
            action="review",
            novelty=0.9,
            salience=0.8,
            goal_relevance=0.9,
            surprise=0.7,
            unresolved_tension=0.8,
            emphasis=0.8,
        )
    )
    assert candidate.durable is True

    queue_patch(
        queue_dir=engine.config.reconsolidation_queue_dir,
        memory_path=candidate.path,
        memory_id=candidate.memory_id,
        claim="Vendor handoff scope includes Q2 deliverable.",
        evidence_anchor=candidate.anchor,
        confidence_delta=0.15,
        expected_project="atlas",
    )
    queue_patch(
        queue_dir=engine.config.reconsolidation_queue_dir,
        memory_path=candidate.path,
        memory_id=candidate.memory_id,
        claim="Incorrect project mutation should conflict.",
        evidence_anchor=candidate.anchor,
        confidence_delta=0.1,
        expected_project="home",
    )

    outcome = engine.run_reconsolidation_commit()
    assert outcome["processed"] >= 2
    assert outcome["committed"] >= 1
    assert outcome["conflicts"] >= 1
    assert outcome["decisions"]
    assert (tmp_path / "indexes" / "claim_graph.json").exists()
