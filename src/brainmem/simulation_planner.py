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
    avg_tension: float = 0.0
    avg_irreversibility: float = 0.0

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
            "tension": float(metadata.get("encoding_weighted_factors", {}).get("unresolved_tension", 0.0)),
            "irreversibility": float(metadata.get("encoding_weighted_factors", {}).get("decision_irreversibility", 0.0)),
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
                "tension_sum": 0.0,
                "irreversibility_sum": 0.0,
            },
        )
        item["count"] += 1
        # proxy success: transition to less negative emotion
        if emotion_val(nxt["emotion"]) >= emotion_val(cur["emotion"]):
            item["successes"] += 1
        item["tension_sum"] += float(cur.get("tension", 0.0))
        item["irreversibility_sum"] += float(cur.get("irreversibility", 0.0))
        anchor = cur["anchor"] or f"event_id={cur['event_id']}"
        if anchor not in item["evidence"]:
            item["evidence"].append(anchor)

    normalized_edges: list[dict[str, Any]] = []
    for edge in edges.values():
        count = max(1, int(edge.get("count", 1)))
        edge["avg_tension"] = round(float(edge.get("tension_sum", 0.0)) / count, 6)
        edge["avg_irreversibility"] = round(float(edge.get("irreversibility_sum", 0.0)) / count, 6)
        edge.pop("tension_sum", None)
        edge.pop("irreversibility_sum", None)
        normalized_edges.append(edge)

    return {
        "states": sorted(state_nodes),
        "edges": sorted(normalized_edges, key=lambda e: (-e["count"], e["from_state"], e["action"])),
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
    goal_priors: list[str] | None = None,
    risk_weight: float = 0.25,
    uncertainty_weight: float = 0.15,
    value_weight: float = 0.6,
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
        confidence = min(1.0, edge.count / 5.0)
        expected_value = edge.success_rate
        risk = min(1.0, (edge.avg_tension * 0.7) + (edge.avg_irreversibility * 0.3))
        uncertainty = 1.0 - confidence
        score = (
            (value_weight * expected_value)
            - (risk_weight * risk)
            - (uncertainty_weight * uncertainty)
            + (0.15 * confidence)
        )
        projected = edge.to_state

        if depth > 1:
            followups = [e for e in edges if e.from_state == edge.to_state]
            followups.sort(key=lambda e: (e.success_rate, e.count), reverse=True)
            if followups:
                next_edge = followups[0]
                chain.append(next_edge)
                next_conf = min(1.0, next_edge.count / 5.0)
                next_value = next_edge.success_rate
                next_risk = min(1.0, (next_edge.avg_tension * 0.7) + (next_edge.avg_irreversibility * 0.3))
                next_uncertainty = 1.0 - next_conf
                next_score = (
                    (value_weight * next_value)
                    - (risk_weight * next_risk)
                    - (uncertainty_weight * next_uncertainty)
                    + (0.15 * next_conf)
                )
                score = (score * 0.6) + (next_score * 0.4)
                projected = next_edge.to_state

        prior_bonus = _goal_prior_bonus(goal_priors, edge)
        goal_bonus = 0.1 if goal_hint and goal_hint.lower() in projected else 0.0
        total_score = round(min(1.0, max(0.0, score + goal_bonus)), 6)
        counterfactual = _best_counterfactual(
            edges=matching_edges,
            current=edge,
            goal_hint=goal_hint,
            goal_priors=goal_priors,
            risk_weight=risk_weight,
            uncertainty_weight=uncertainty_weight,
            value_weight=value_weight,
        )
        proposals.append(
            {
                "action_sequence": [item.action for item in chain],
                "from_state": edge.from_state,
                "projected_state": projected,
                "score": total_score,
                "utility": round(total_score, 6),
                "risk": round(risk, 6),
                "uncertainty": round(uncertainty, 6),
                "expected_value": round(expected_value, 6),
                "goal_prior_bonus": round(prior_bonus, 6),
                "evidence": [anchor for item in chain for anchor in item.evidence][:5],
                "explanation": (
                    f"Based on {len(chain)} transition(s) with observed success patterns."
                    if edge.from_state == current_state
                    else f"Fallback from nearest state {edge.from_state}; matched by project/mode cues."
                ),
                "counterfactual": counterfactual,
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


def _edge_utility(
    edge: TransitionEdge,
    *,
    goal_hint: str,
    goal_priors: list[str] | None = None,
    risk_weight: float,
    uncertainty_weight: float,
    value_weight: float,
) -> float:
    confidence = min(1.0, edge.count / 5.0)
    expected_value = edge.success_rate
    risk = min(1.0, (edge.avg_tension * 0.7) + (edge.avg_irreversibility * 0.3))
    uncertainty = 1.0 - confidence
    bonus = 0.1 if goal_hint and goal_hint.lower() in edge.to_state else 0.0
    bonus += _goal_prior_bonus(goal_priors, edge)
    return (
        (value_weight * expected_value)
        - (risk_weight * risk)
        - (uncertainty_weight * uncertainty)
        + (0.15 * confidence)
        + bonus
    )


def _best_counterfactual(
    *,
    edges: list[TransitionEdge],
    current: TransitionEdge,
    goal_hint: str,
    goal_priors: list[str] | None = None,
    risk_weight: float,
    uncertainty_weight: float,
    value_weight: float,
) -> dict[str, Any]:
    alternatives = [edge for edge in edges if edge.action != current.action or edge.to_state != current.to_state]
    if not alternatives:
        return {"action": "none", "projected_state": current.to_state, "score": 0.0}
    best = max(
        alternatives,
        key=lambda edge: _edge_utility(
            edge,
            goal_hint=goal_hint,
            goal_priors=goal_priors,
            risk_weight=risk_weight,
            uncertainty_weight=uncertainty_weight,
            value_weight=value_weight,
        ),
    )
    return {
        "action": best.action,
        "projected_state": best.to_state,
        "score": round(
            _edge_utility(
                best,
                goal_hint=goal_hint,
                goal_priors=goal_priors,
                risk_weight=risk_weight,
                uncertainty_weight=uncertainty_weight,
                value_weight=value_weight,
            ),
            6,
        ),
    }


def _goal_prior_bonus(goal_priors: list[str] | None, edge: TransitionEdge) -> float:
    if not goal_priors:
        return 0.0
    text = f"{edge.from_state} {edge.to_state} action:{edge.action}".lower()
    hits = 0
    for goal in goal_priors:
        g = goal.strip().lower()
        if g and g in text:
            hits += 1
    if hits <= 0:
        return 0.0
    return min(0.12, 0.04 * hits)


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
