from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .markdown_io import read_markdown, write_markdown


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class BeliefRecord:
    belief_id: str
    statement: str
    tags: list[str]
    confidence: float
    supports: list[str]
    disconfirms: list[str]
    updated_at: str


def ensure_identity_goal_files(identity_dir: Path, goals_dir: Path) -> None:
    identity_dir.mkdir(parents=True, exist_ok=True)
    goals_dir.mkdir(parents=True, exist_ok=True)
    identity_file = identity_dir / "identity.md"
    if not identity_file.exists():
        write_markdown(
            identity_file,
            {"type": "identity_profile", "updated_at": _utc_iso()},
            "## Roles\n- builder\n\n## Values\n- clarity\n- reliability\n",
        )
    goals_file = goals_dir / "active_goals.md"
    if not goals_file.exists():
        write_markdown(
            goals_file,
            {"type": "goal_profile", "updated_at": _utc_iso()},
            "## Active Goals\n- deliver reliable personal assistance\n",
        )


def upsert_belief(
    *,
    identity_dir: Path,
    statement: str,
    tags: list[str],
    evidence_anchor: str,
    confidence_delta: float = 0.05,
) -> Path:
    beliefs_dir = identity_dir / "beliefs"
    beliefs_dir.mkdir(parents=True, exist_ok=True)

    normalized_statement = statement.strip()
    for existing in sorted(beliefs_dir.glob("*.md")):
        meta, body = read_markdown(existing)
        if str(meta.get("statement", body.strip())).strip().lower() == normalized_statement.lower():
            confidence = float(meta.get("confidence", 0.5))
            confidence = round(max(0.0, min(1.0, confidence + confidence_delta)), 6)
            supports = [str(s) for s in meta.get("supports", [])]
            if evidence_anchor and evidence_anchor not in supports:
                supports.append(evidence_anchor)
            meta["confidence"] = confidence
            meta["supports"] = supports
            meta["tags"] = sorted(set([str(t).lower() for t in meta.get("tags", [])] + [t.lower() for t in tags]))
            meta["updated_at"] = _utc_iso()
            write_markdown(existing, meta, body)
            return existing

    belief_id = f"belief-{uuid4().hex[:10]}"
    path = beliefs_dir / f"{belief_id}.md"
    meta = {
        "type": "identity_belief",
        "belief_id": belief_id,
        "statement": normalized_statement,
        "tags": sorted(set([t.lower() for t in tags])),
        "confidence": round(max(0.0, min(1.0, 0.5 + confidence_delta)), 6),
        "supports": [evidence_anchor] if evidence_anchor else [],
        "disconfirms": [],
        "updated_at": _utc_iso(),
    }
    write_markdown(path, meta, normalized_statement)
    return path


def log_belief_conflict(
    *,
    identity_dir: Path,
    belief_statement: str,
    conflicting_evidence_anchor: str,
    note: str,
) -> Path:
    conflicts_dir = identity_dir / "conflicts"
    conflicts_dir.mkdir(parents=True, exist_ok=True)
    conflict_id = f"conflict-{uuid4().hex[:10]}"
    path = conflicts_dir / f"{conflict_id}.md"
    meta = {
        "type": "identity_conflict",
        "conflict_id": conflict_id,
        "belief_statement": belief_statement.strip(),
        "evidence_anchor": conflicting_evidence_anchor,
        "status": "open",
        "created_at": _utc_iso(),
    }
    write_markdown(path, meta, note.strip())
    return path


def load_beliefs(identity_dir: Path) -> list[BeliefRecord]:
    beliefs_dir = identity_dir / "beliefs"
    if not beliefs_dir.exists():
        return []
    beliefs: list[BeliefRecord] = []
    for path in sorted(beliefs_dir.glob("*.md")):
        meta, body = read_markdown(path)
        if meta.get("type") != "identity_belief":
            continue
        beliefs.append(
            BeliefRecord(
                belief_id=str(meta.get("belief_id", path.stem)),
                statement=str(meta.get("statement", body.strip())),
                tags=[str(t) for t in meta.get("tags", [])],
                confidence=float(meta.get("confidence", 0.5)),
                supports=[str(s) for s in meta.get("supports", [])],
                disconfirms=[str(s) for s in meta.get("disconfirms", [])],
                updated_at=str(meta.get("updated_at", "")),
            )
        )
    return beliefs


def load_goal_tags(goals_dir: Path) -> list[str]:
    goals_file = goals_dir / "active_goals.md"
    if not goals_file.exists():
        return []
    _, body = read_markdown(goals_file)
    tags: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            tags.append(stripped[2:].strip().lower())
    return tags


def compute_identity_goal_relevance(
    *,
    cues: list[str],
    identity_tags: list[str],
    goal_tags: list[str],
) -> dict[str, float]:
    cue_text = " ".join(cues).lower()
    identity_hits = 0
    for tag in identity_tags:
        t = tag.lower().strip()
        if t and t in cue_text:
            identity_hits += 1
    goal_hits = 0
    for tag in goal_tags:
        t = tag.lower().strip()
        if t and t in cue_text:
            goal_hits += 1
    identity_score = min(1.0, identity_hits / max(1, len(identity_tags))) if identity_tags else 0.0
    goal_score = min(1.0, goal_hits / max(1, len(goal_tags))) if goal_tags else 0.0
    return {
        "identity": round(identity_score, 6),
        "goal": round(goal_score, 6),
        "combined": round((0.6 * identity_score) + (0.4 * goal_score), 6),
    }


def score_identity_goal_prior(
    *,
    cues: list[str],
    project: str,
    identity_dir: Path,
    goals_dir: Path,
) -> dict[str, Any]:
    cue_text = " ".join(cues).lower()
    beliefs = load_beliefs(identity_dir)
    goal_tags = load_goal_tags(goals_dir)

    belief_score = 0.0
    matched_beliefs: list[str] = []
    for belief in beliefs:
        tokens = [belief.statement.lower()] + [tag.lower() for tag in belief.tags]
        if any(token and token in cue_text for token in tokens):
            belief_score += belief.confidence
            matched_beliefs.append(belief.belief_id)

    goal_score = 0.0
    matched_goals: list[str] = []
    for goal in goal_tags:
        if goal and (goal in cue_text or goal in project.lower()):
            goal_score += 1.0
            matched_goals.append(goal)

    normalized_belief = min(1.0, belief_score / max(1, len(beliefs))) if beliefs else 0.0
    normalized_goal = min(1.0, goal_score / max(1, len(goal_tags))) if goal_tags else 0.0
    total = round((0.6 * normalized_belief) + (0.4 * normalized_goal), 6)
    return {
        "score": total,
        "belief_component": round(normalized_belief, 6),
        "goal_component": round(normalized_goal, 6),
        "matched_beliefs": matched_beliefs,
        "matched_goals": matched_goals,
    }


def update_identity_and_goals_from_event(
    *,
    identity_dir: Path,
    goals_dir: Path,
    event_id: str,
    event_text: str,
    cues: list[str],
    project: str,
    source_anchor: str,
) -> dict[str, Any]:
    ensure_identity_goal_files(identity_dir, goals_dir)
    tags = []
    for cue in cues:
        if cue.startswith(("person:", "project:", "action:", "emotion:")):
            tags.append(cue)
    statement = f"Event {event_id}: {event_text[:140].strip()}"
    belief_path = upsert_belief(
        identity_dir=identity_dir,
        statement=statement,
        tags=tags,
        evidence_anchor=source_anchor,
        confidence_delta=0.04,
    )

    goals_file = goals_dir / "active_goals.md"
    goals_meta, goals_body = read_markdown(goals_file)
    goal_line = f"- project:{project.lower()}"
    lines = [line for line in goals_body.splitlines() if line.strip()]
    if goal_line not in lines and project and project.lower() != "general":
        lines.append(goal_line)
    goals_meta["updated_at"] = _utc_iso()
    write_markdown(goals_file, goals_meta, "\n".join(lines).strip() + "\n")
    return {
        "belief_path": str(belief_path),
        "goals_path": str(goals_file),
    }


def load_identity_beliefs(identity_dir: Path) -> list[BeliefRecord]:
    return load_beliefs(identity_dir)


def load_goal_priors(goals_dir: Path) -> list[str]:
    return load_goal_tags(goals_dir)


def update_identity_goal_evidence(
    *,
    identity_dir: Path,
    goals_dir: Path | None = None,
    belief_statement: str,
    evidence_anchor: str,
    confidence_delta: float = 0.02,
) -> dict[str, str]:
    belief_path = upsert_belief(
        identity_dir=identity_dir,
        statement=belief_statement,
        tags=[],
        evidence_anchor=evidence_anchor,
        confidence_delta=confidence_delta,
    )
    if goals_dir is not None:
        ensure_identity_goal_files(identity_dir, goals_dir)
    return {"belief_path": str(belief_path)}


def detect_identity_conflicts(
    *,
    identity_dir: Path,
    belief_statement: str,
    evidence_anchor: str,
    note: str,
) -> Path:
    return log_belief_conflict(
        identity_dir=identity_dir,
        belief_statement=belief_statement,
        conflicting_evidence_anchor=evidence_anchor,
        note=note,
    )
