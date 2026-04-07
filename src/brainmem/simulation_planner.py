from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .markdown_io import read_markdown


@dataclass(slots=True)
class TransitionEdge:
    from_state: str
    action: str
    to_state: str
    count: int
    successes: int
    evidence: list[str]

    @property
    def success_rate(self) -> float:
        if self.count == 0:
            return 0.0
        return self.successes / self.count


def build_experience_graph(events_dir: Path) -> dict[str, Any]:
    """Build a transition graph from chronological event memories."""
    records: list[dict[str, Any]] = []
    for path in sorted(events_dir.glob("**/*.md")):
        metadata, body = read_markdown(path)
        if metadata.get("type") != "event":
            continue
        record = {
            "event_id": str(metadata.get("event_id", path.stem)),
            "created_at": str(metadata.get("created_at", "")),
            "project": str(metadata.get("project", "general")).lower(),
            "mode": str(metadata.get("mode", "unknown")).lower(),
            "emotion": str(metadata.get("emotion", "neutral")).lower(),
            "action": str(metadata.get("action", "note")).lower(),
            "anchor": str(metadata.get("source_anchor", "")),
            "text": body[:240].strip(),
        }
        records.append(record)

    records.sort(key=lambda r: r["created_at"])
    edges: dict[str, dict[str, Any]] = {}
    state_nodes: set[str] = set()

    for idx in range(len(records) - 1):
        cur = records[idx]
        nxt = records[idx + 1]
        from_state = state_key(cur)
        to_state = state_key(nxt)
        state_nodes.add(from_state)
        state_nodes.add(to_state)
        key = f"{from_state}|{cur['action']}|{to_state}"
        item = edges.setdefault(
            key,
            {
                "from_state": from_state,
                "action": cur["action"],
                "to_state": to_state,
                "count": 0,
                "successes": 0,
                "evidence": [],
            },
        )
        item["count"] += 1
        # proxy success: transition to less negative emotion
        if emotion_val(nxt["emotion"]) >= emotion_val(cur["emotion"]):
            item["successes"] += 1
        anchor = cur["anchor"] or f"event_id={cur['event_id']}"
        if anchor not in item["evidence"]:
            item["evidence"].append(anchor)

    return {
        "states": sorted(state_nodes),
        "edges": sorted(edges.values(), key=lambda e: (-e["count"], e["from_state"], e["action"])),
    }


def save_experience_graph(path: Path, graph: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")


def load_experience_graph(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"states": [], "edges": []}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {"states": [], "edges": []}
    return json.loads(raw)


def simulate_plan(
    *,
    graph: dict[str, Any],
    current_state: str,
    goal_hint: str,
    depth: int = 2,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Simple rollout over transition graph with evidence-backed outputs."""
    edges = [TransitionEdge(**edge) for edge in graph.get("edges", [])]
    if not edges:
        return []

    matching_edges = [edge for edge in edges if edge.from_state == current_state]
    if not matching_edges:
        current_parts = parse_state_key(current_state)
        project_mode_matches = [
            edge
            for edge in edges
            if parse_state_key(edge.from_state).get("project") == current_parts.get("project")
            and parse_state_key(edge.from_state).get("mode") == current_parts.get("mode")
        ]
        if project_mode_matches:
            matching_edges = project_mode_matches
        else:
            project_matches = [
                edge
                for edge in edges
                if parse_state_key(edge.from_state).get("project") == current_parts.get("project")
            ]
            matching_edges = project_matches or edges

    proposals: list[dict[str, Any]] = []
    for edge in matching_edges:
        chain = [edge]
        score = edge.success_rate * 0.7 + min(1.0, edge.count / 5.0) * 0.3
        projected = edge.to_state

        if depth > 1:
            followups = [e for e in edges if e.from_state == edge.to_state]
            followups.sort(key=lambda e: (e.success_rate, e.count), reverse=True)
            if followups:
                next_edge = followups[0]
                chain.append(next_edge)
                score = (score * 0.6) + ((next_edge.success_rate * 0.7 + min(1.0, next_edge.count / 5.0) * 0.3) * 0.4)
                projected = next_edge.to_state

        goal_bonus = 0.1 if goal_hint and goal_hint.lower() in projected else 0.0
        total_score = round(min(1.0, score + goal_bonus), 6)
        proposals.append(
            {
                "action_sequence": [item.action for item in chain],
                "from_state": edge.from_state,
                "projected_state": projected,
                "score": total_score,
                "evidence": [anchor for item in chain for anchor in item.evidence][:5],
                "explanation": (
                    f"Based on {len(chain)} transition(s) with observed success patterns."
                    if edge.from_state == current_state
                    else f"Fallback from nearest state {edge.from_state}; matched by project/mode cues."
                ),
            }
        )

    proposals.sort(key=lambda item: item["score"], reverse=True)
    return proposals[:top_k]


def state_key(item: dict[str, str]) -> str:
    return f"project:{item['project']}|mode:{item['mode']}|emotion:{item['emotion']}"


def parse_state_key(value: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for chunk in value.split("|"):
        if ":" not in chunk:
            continue
        key, raw = chunk.split(":", 1)
        parts[key.strip().lower()] = raw.strip().lower()
    return parts


def emotion_val(emotion: str) -> float:
    mapping = {
        "angry": 0.1,
        "frustrated": 0.2,
        "worried": 0.3,
        "neutral": 0.5,
        "focused": 0.7,
        "happy": 0.8,
        "excited": 0.9,
    }
    return mapping.get(emotion.lower(), 0.5)
