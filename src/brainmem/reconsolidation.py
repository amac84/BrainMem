from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .markdown_io import read_markdown, write_markdown


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class ReconsolidationDecision:
    patch_path: Path
    status: str
    reason: str
    memory_id: str
    claim: str
    confidence_delta: float


def queue_patch(
    *,
    queue_dir: Path,
    memory_path: Path,
    memory_id: str,
    claim: str,
    evidence_anchor: str,
    confidence_delta: float = 0.05,
    expected_project: str = "",
) -> Path:
    queue_dir.mkdir(parents=True, exist_ok=True)
    patch_id = f"rp-{uuid4().hex[:10]}"
    patch_path = queue_dir / f"{patch_id}.md"
    metadata = {
        "type": "reconsolidation_patch",
        "patch_id": patch_id,
        "memory_id": memory_id,
        "memory_path": str(memory_path),
        "claim": claim,
        "evidence_anchor": evidence_anchor,
        "confidence_delta": confidence_delta,
        "expected_project": expected_project,
        "status": "queued",
        "created_at": _utc_iso(),
    }
    write_markdown(patch_path, metadata, claim.strip())
    return patch_path


def queue_reconsolidation_patch(
    *,
    root: Path,
    memory_id: str,
    event_path: Path,
    recall_anchor: str,
    observed_excerpt: str,
    evidence_note: str,
    score_breakdown: dict[str, Any],
) -> Path:
    queue_dir = Path(root) / "memory" / "reconsolidation" / "queue"
    metadata, _ = read_markdown(event_path)
    expected_project = str(metadata.get("project", ""))
    claim = (
        f"recalled_memory:{memory_id}: "
        f"{observed_excerpt[:160].strip()}"
    )
    evidence_anchor = recall_anchor or str(metadata.get("source_anchor", ""))
    patch_path = queue_patch(
        queue_dir=queue_dir,
        memory_path=event_path,
        memory_id=memory_id,
        claim=claim,
        evidence_anchor=evidence_anchor,
        confidence_delta=0.03,
        expected_project=expected_project,
    )
    patch_meta, patch_body = read_markdown(patch_path)
    patch_meta["evidence_note"] = evidence_note
    patch_meta["score_breakdown"] = score_breakdown
    write_markdown(patch_path, patch_meta, patch_body)
    labilize_memory(event_path)
    return patch_path


def labilize_memory(memory_path: Path, *, window_hours: int = 4) -> None:
    metadata, body = read_markdown(memory_path)
    metadata["labilized"] = True
    metadata["labilized_at"] = _utc_iso()
    metadata["labilization_window_hours"] = window_hours
    write_markdown(memory_path, metadata, body)


def commit_reconsolidation_queue(
    *,
    queue_dir: Path | None = None,
    claim_graph_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    if root is not None:
        root_path = Path(root)
        queue_dir = root_path / "memory" / "reconsolidation" / "queue"
        claim_graph_path = root_path / "indexes" / "claim_graph.json"
    if queue_dir is None or claim_graph_path is None:
        raise ValueError("queue_dir and claim_graph_path are required when root is not provided")

    decisions: list[ReconsolidationDecision] = []
    claim_graph = _read_json(claim_graph_path, default={})

    for patch_path in sorted(queue_dir.glob("*.md")):
        metadata, body = read_markdown(patch_path)
        if metadata.get("type") != "reconsolidation_patch":
            continue
        status = str(metadata.get("status", "queued"))
        if status != "queued":
            continue

        memory_path = Path(str(metadata.get("memory_path", "")))
        memory_id = str(metadata.get("memory_id", ""))
        claim = str(metadata.get("claim", body.strip()))
        confidence_delta = float(metadata.get("confidence_delta", 0.05))
        evidence_anchor = str(metadata.get("evidence_anchor", ""))
        expected_project = str(metadata.get("expected_project", "")).lower().strip()

        if not memory_path.exists():
            _mark_patch_status(patch_path, "rejected", "memory_missing")
            decisions.append(
                ReconsolidationDecision(
                    patch_path=patch_path,
                    status="rejected",
                    reason="memory_missing",
                    memory_id=memory_id,
                    claim=claim,
                    confidence_delta=confidence_delta,
                )
            )
            continue

        if not evidence_anchor:
            _mark_patch_status(patch_path, "deferred", "missing_evidence_anchor")
            decisions.append(
                ReconsolidationDecision(
                    patch_path=patch_path,
                    status="deferred",
                    reason="missing_evidence_anchor",
                    memory_id=memory_id,
                    claim=claim,
                    confidence_delta=confidence_delta,
                )
            )
            continue

        mem_meta, mem_body = read_markdown(memory_path)
        actual_project = str(mem_meta.get("project", "")).lower().strip()
        if expected_project and actual_project and expected_project != actual_project:
            _mark_patch_status(patch_path, "conflict", "project_mismatch")
            decisions.append(
                ReconsolidationDecision(
                    patch_path=patch_path,
                    status="conflict",
                    reason="project_mismatch",
                    memory_id=memory_id,
                    claim=claim,
                    confidence_delta=confidence_delta,
                )
            )
            continue

        mem_meta.setdefault("claims", [])
        mem_meta.setdefault("evidence", [])
        mem_meta.setdefault("confidence", {})
        claims = [str(c) for c in mem_meta.get("claims", [])]
        if claim not in claims:
            claims.append(claim)
        mem_meta["claims"] = claims

        evidence = [str(ev) for ev in mem_meta.get("evidence", [])]
        if evidence_anchor not in evidence:
            evidence.append(evidence_anchor)
        mem_meta["evidence"] = evidence

        confidence_map = dict(mem_meta.get("confidence", {}))
        existing = float(confidence_map.get(claim, 0.5))
        confidence_map[claim] = round(max(0.0, min(1.0, existing + confidence_delta)), 6)
        mem_meta["confidence"] = confidence_map
        mem_meta["labilized"] = False
        mem_meta["reconsolidated_at"] = _utc_iso()

        write_markdown(memory_path, mem_meta, mem_body)
        _mark_patch_status(patch_path, "committed", "applied")
        decisions.append(
            ReconsolidationDecision(
                patch_path=patch_path,
                status="committed",
                reason="applied",
                memory_id=memory_id,
                claim=claim,
                confidence_delta=confidence_delta,
            )
        )

        claim_graph.setdefault(memory_id, {})
        claim_graph[memory_id].setdefault("claims", {})
        claim_graph[memory_id]["claims"][claim] = {
            "confidence": mem_meta["confidence"][claim],
            "evidence": evidence,
            "updated_at": _utc_iso(),
        }

    _write_json(claim_graph_path, claim_graph)
    committed = [d for d in decisions if d.status == "committed"]
    conflicts = [d for d in decisions if d.status == "conflict"]
    deferred = [d for d in decisions if d.status == "deferred"]
    rejected = [d for d in decisions if d.status == "rejected"]
    return {
        "processed": len(decisions),
        "committed": len(committed),
        "conflicts": len(conflicts),
        "deferred": len(deferred),
        "rejected": len(rejected),
        "decisions": [
            {
                "patch_path": str(d.patch_path),
                "status": d.status,
                "reason": d.reason,
                "memory_id": d.memory_id,
                "claim": d.claim,
                "confidence_delta": d.confidence_delta,
            }
            for d in decisions
        ],
    }


def _mark_patch_status(patch_path: Path, status: str, reason: str) -> None:
    metadata, body = read_markdown(patch_path)
    metadata["status"] = status
    metadata["status_reason"] = reason
    metadata["updated_at"] = _utc_iso()
    write_markdown(patch_path, metadata, body)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return default
    return json.loads(raw)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
