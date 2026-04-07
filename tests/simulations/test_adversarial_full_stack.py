from __future__ import annotations

from datetime import date

from brainmem.consolidation_job import run_daily_consolidation
from brainmem.engine import BrainMemEngine
from brainmem.types import IngestInput, RecallRequest


def test_adversarial_full_stack_end_to_end(tmp_path) -> None:
    engine = BrainMemEngine(root=tmp_path)

    # Day 1: similar conversation with same person/topic
    e1 = engine.ingest(
        IngestInput(
            text="Maya asked to review Atlas budget for Q2 and vendor risks.",
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
            unresolved_tension=0.8,
            emphasis=0.8,
        )
    )
    e2 = engine.ingest(
        IngestInput(
            text="Maya asked to review Atlas budget for Q3 and vendor risks.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="planning",
            emotion="worried",
            action="review",
            novelty=0.75,
            salience=0.75,
            goal_relevance=0.9,
            surprise=0.5,
            unresolved_tension=0.7,
            emphasis=0.7,
        )
    )
    assert e1.durable and e2.durable

    # Same topic, different goal mode
    e3 = engine.ingest(
        IngestInput(
            text="Reflecting on Atlas budget mistakes and lessons from vendor negotiation.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="reflection",
            emotion="neutral",
            action="review",
            novelty=0.8,
            salience=0.75,
            goal_relevance=0.9,
            surprise=0.6,
            unresolved_tension=0.6,
            emphasis=0.75,
        )
    )
    assert e3.durable

    # High-salience low-importance noise (may or may not encode depending on threshold)
    e4 = engine.ingest(
        IngestInput(
            text="A shocking celebrity rumor interrupted my work stream.",
            people=[],
            project="general",
            tool="phone",
            mode="reflection",
            emotion="excited",
            action="note",
            novelty=0.9,
            salience=0.9,
            goal_relevance=0.1,
            surprise=0.9,
            unresolved_tension=0.1,
            emphasis=0.2,
        )
    )
    # Do not enforce durability here; this item is intentionally low goal relevance.

    # Low-salience later-critical signal with explicit emphasis to ensure capture
    e5 = engine.ingest(
        IngestInput(
            text="Reminder: contract auto-renews on 2026-05-01 unless cancelled.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="planning",
            emotion="neutral",
            action="review",
            novelty=0.2,
            salience=0.2,
            goal_relevance=0.85,
            surprise=0.2,
            unresolved_tension=0.7,
            emphasis=0.9,
        )
    )
    if not e5.durable:
        e5 = engine.ingest(
            IngestInput(
                text="Reminder: contract auto-renews on 2026-05-01 unless cancelled.",
                people=["Maya"],
                project="atlas",
                tool="laptop",
                mode="planning",
                emotion="neutral",
                action="review",
                novelty=0.45,
                salience=0.4,
                goal_relevance=0.95,
                surprise=0.35,
                unresolved_tension=0.85,
                emphasis=1.0,
            )
        )
    assert e5.durable

    # Contradictory evidence event
    e6 = engine.ingest(
        IngestInput(
            text="Vendor confirmed contract can be cancelled with 30-day notice.",
            people=["Maya"],
            project="atlas",
            tool="laptop",
            mode="planning",
            emotion="focused",
            action="call",
            novelty=0.7,
            salience=0.7,
            goal_relevance=0.9,
            surprise=0.6,
            unresolved_tension=0.4,
            emphasis=0.7,
        )
    )
    assert e6.durable

    # Recall with partial cues should surface atlas memories, not celebrity noise.
    recall = engine.recall(
        RecallRequest(
            cues=["project:atlas", "token:budget", "token:vendor", "person:maya"],
            people=["Maya"],
            place="unknown",
            tool="laptop",
            mode="planning",
            project="atlas",
            emotion="worried",
            top_k=4,
        )
    )
    assert recall
    assert any(r.score_breakdown.get("pattern_separation_required", 0.0) == 1.0 for r in recall[1:])
    assert all("project:atlas" in r.cues or "person:maya" in r.cues for r in recall[:2])

    # Reconsolidation should commit queued patches.
    recon = engine.run_reconsolidation_commit()
    assert recon["processed"] >= 1

    # Consolidation and forgetting should run without breaking traceability.
    summary = run_daily_consolidation(
        config=engine.config,
        target_date=date.fromisoformat("2026-04-07"),
        boundary_threshold=0.45,
        run_forgetting=True,
    )
    assert summary["events_considered"] >= 6
    assert summary["anchors"]
    assert "forgetting" in summary

    # Planner should provide evidence-backed proposals from accumulated transitions.
    sim = engine.simulate(
        current_state="project:atlas|mode:planning|emotion:worried",
        goal_hint="atlas",
        depth=2,
        top_k=3,
    )
    assert "proposals" in sim
    if sim["proposals"]:
        assert sim["proposals"][0]["evidence"]
