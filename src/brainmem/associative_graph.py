from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .markdown_io import read_markdown


@dataclass(slots=True)
class GraphActivationResult:
    memory_id: str
    path: str
    activation_score: float
    matched_cues: list[str]
    top_supporting_cues: list[str]
    overlap_with_top: float = 0.0
    disambiguation_required: bool = False
    pattern_separation_hint: str = ""


def build_hig_adjacency(events_dir: Path) -> dict[str, Any]:
    """Build a sparse cue-memory adjacency graph from event markdown memories."""
    memories: dict[str, dict[str, Any]] = {}
    cue_to_memories: dict[str, list[str]] = defaultdict(list)
    memory_to_cues: dict[str, list[str]] = {}

    for path in sorted(events_dir.glob("**/*.md")):
        metadata, _ = read_markdown(path)
        if metadata.get("type") != "event":
            continue
        memory_id = str(metadata.get("event_id", path.stem))
        cues = sorted({str(c) for c in metadata.get("cues", []) if str(c)})
        memories[memory_id] = {
            "path": str(path),
            "cues": cues,
            "state": metadata.get("state", {}),
            "project": metadata.get("project", "general"),
        }
        memory_to_cues[memory_id] = cues
        for cue in cues:
            cue_to_memories[cue].append(memory_id)

    return {
        "memories": memories,
        "cue_to_memories": dict(cue_to_memories),
        "memory_to_cues": memory_to_cues,
    }


def activate_from_cues(
    *,
    query_cues: list[str],
    adjacency: dict[str, Any],
    top_k: int = 10,
    min_activation: float = 0.01,
) -> list[GraphActivationResult]:
    """Spreading-style activation using sparse cue overlaps."""
    cue_to_memories: dict[str, list[str]] = dict(adjacency.get("cue_to_memories", {}))
    memory_to_cues: dict[str, list[str]] = dict(adjacency.get("memory_to_cues", {}))
    memories: dict[str, dict[str, Any]] = dict(adjacency.get("memories", {}))

    if not query_cues:
        return []

    contributions: dict[str, float] = defaultdict(float)
    matched: dict[str, set[str]] = defaultdict(set)

    for cue in query_cues:
        matches = cue_to_memories.get(cue, [])
        if not matches:
            continue
        fanout = max(1, len(matches))
        contribution = 1.0 / fanout
        for memory_id in matches:
            contributions[memory_id] += contribution
            matched[memory_id].add(cue)

    results: list[GraphActivationResult] = []
    for memory_id, raw_score in contributions.items():
        cues = set(memory_to_cues.get(memory_id, []))
        if not cues:
            continue
        # normalize by memory cue count to avoid huge cue lists dominating
        normalized = raw_score / (1.0 + (len(cues) * 0.05))
        if normalized < min_activation:
            continue
        memory_ref = memories.get(memory_id, {})
        matched_cues = sorted(matched.get(memory_id, set()))
        top_supporting = sorted(matched_cues, key=lambda cue: cue_weight(cue), reverse=True)[:5]
        results.append(
            GraphActivationResult(
                memory_id=memory_id,
                path=str(memory_ref.get("path", "")),
                activation_score=round(normalized, 6),
                matched_cues=matched_cues,
                top_supporting_cues=top_supporting,
            )
        )

    results.sort(key=lambda item: item.activation_score, reverse=True)
    return results[:top_k]


def apply_pattern_separation(
    activations: list[GraphActivationResult],
    adjacency: dict[str, Any],
    *,
    overlap_threshold: float = 0.75,
) -> list[GraphActivationResult]:
    """Mark near-duplicate top activations and produce disambiguation hints."""
    if len(activations) < 2:
        return activations

    memory_to_cues: dict[str, list[str]] = dict(adjacency.get("memory_to_cues", {}))
    top_cues = set(memory_to_cues.get(activations[0].memory_id, []))
    if not top_cues:
        return activations

    for idx in range(1, len(activations)):
        contender = activations[idx]
        contender_cues = set(memory_to_cues.get(contender.memory_id, []))
        overlap = jaccard(top_cues, contender_cues)
        contender.overlap_with_top = round(overlap, 6)
        if overlap >= overlap_threshold:
            contender.disambiguation_required = True
            contenders_only = sorted(contender_cues - top_cues)[:4]
            top_only = sorted(top_cues - contender_cues)[:4]
            hint_bits = []
            if top_only:
                hint_bits.append(f"top-unique: {', '.join(top_only)}")
            if contenders_only:
                hint_bits.append(f"contender-unique: {', '.join(contenders_only)}")
            contender.pattern_separation_hint = " | ".join(hint_bits) or "request more context cues"
    return activations


def cue_weight(cue: str) -> float:
    if cue.startswith("person:"):
        return 1.0
    if cue.startswith("project:"):
        return 0.95
    if cue.startswith("action:"):
        return 0.85
    if cue.startswith("emotion:"):
        return 0.8
    if cue.startswith("tool:"):
        return 0.75
    if cue.startswith("place:"):
        return 0.7
    return 0.5


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
