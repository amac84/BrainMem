from __future__ import annotations

from pathlib import Path

from brainmem.markdown_io import read_markdown, write_markdown
from brainmem.reconsolidation import commit_reconsolidation_queue, queue_patch


def test_reconsolidation_patch_commit_updates_memory_and_claim_graph(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory" / "events" / "2026" / "04" / "07" / "ev-1.md"
    write_markdown(
        memory_path,
        {
            "type": "event",
            "event_id": "ev-1",
            "project": "atlas",
            "claims": ["Initial claim"],
            "evidence": ["2026-04-07.md#event_id=ev-1"],
            "confidence": {"Initial claim": 0.6},
        },
        "Event body",
    )

    patch_path = queue_patch(
        queue_dir=tmp_path / "memory" / "reconsolidation" / "queue",
        memory_path=memory_path,
        memory_id="ev-1",
        claim="Vendor dependency has high risk",
        evidence_anchor="2026-04-07.md#event_id=ev-1",
        confidence_delta=0.2,
        expected_project="atlas",
    )
    assert patch_path.exists()

    outcome = commit_reconsolidation_queue(
        queue_dir=tmp_path / "memory" / "reconsolidation" / "queue",
        claim_graph_path=tmp_path / "indexes" / "claim_graph.json",
    )
    assert outcome["processed"] == 1
    assert outcome["committed"] == 1
    assert outcome["decisions"][0]["status"] == "committed"

    meta, _ = read_markdown(memory_path)
    assert "Vendor dependency has high risk" in meta["claims"]
    assert meta["confidence"]["Vendor dependency has high risk"] > 0.5
    assert meta["labilized"] is False


def test_reconsolidation_patch_conflict_on_project_mismatch(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory" / "events" / "2026" / "04" / "07" / "ev-2.md"
    write_markdown(
        memory_path,
        {
            "type": "event",
            "event_id": "ev-2",
            "project": "home",
            "claims": [],
            "evidence": [],
            "confidence": {},
        },
        "Event body",
    )
    patch_path = queue_patch(
        queue_dir=tmp_path / "memory" / "reconsolidation" / "queue",
        memory_path=memory_path,
        memory_id="ev-2",
        claim="Atlas-specific claim",
        evidence_anchor="2026-04-07.md#event_id=ev-2",
        expected_project="atlas",
    )
    outcome = commit_reconsolidation_queue(
        queue_dir=tmp_path / "memory" / "reconsolidation" / "queue",
        claim_graph_path=tmp_path / "indexes" / "claim_graph.json",
    )
    assert outcome["processed"] == 1
    assert outcome["conflicts"] == 1
    assert outcome["decisions"][0]["status"] == "conflict"

    patch_meta, _ = read_markdown(patch_path)
    assert patch_meta["status"] == "conflict"
