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
class OpenLoopTrigger:
    cues: list[str]
    project: str = "general"


@dataclass(slots=True)
class OpenLoopRecord:
    loop_id: str
    title: str
    trigger: OpenLoopTrigger
    next_action: str
    closure_condition: str
    tension: float
    status: str = "open"
    source_anchor: str = ""
    created_at: str = ""

    @property
    def intent(self) -> str:
        return self.title

    def to_frontmatter(self) -> dict:
        return {
            "type": "open_loop",
            "loop_id": self.loop_id,
            "title": self.title,
            "trigger_cues": self.trigger.cues,
            "trigger_project": self.trigger.project,
            "next_action": self.next_action,
            "closure_condition": self.closure_condition,
            "tension": round(max(0.0, min(1.0, self.tension)), 6),
            "status": self.status,
            "source_anchor": self.source_anchor,
            "created_at": self.created_at or _utc_iso(),
        }


@dataclass(slots=True)
class OpenLoopMatch:
    loop_id: str
    trigger_score: float
    tension: float
    next_action: str
    trigger_reason: str


def create_open_loop(
    *,
    loops_dir: Path | None = None,
    open_loops_dir: Path | None = None,
    title: str | None = None,
    intent: str | None = None,
    trigger_cues: list[str] | None = None,
    trigger: OpenLoopTrigger | None = None,
    next_action: str,
    closure_condition: str,
    tension: float,
    source_anchor: str = "",
) -> OpenLoopRecord:
    target_dir = open_loops_dir or loops_dir
    if target_dir is None:
        raise ValueError("open loop directory is required")
    target_dir.mkdir(parents=True, exist_ok=True)

    trigger_obj = trigger or OpenLoopTrigger(
        cues=sorted(set(trigger_cues or [])),
        project="general",
    )
    loop = OpenLoopRecord(
        loop_id=f"ol-{uuid4().hex[:10]}",
        title=(title or intent or "Untitled open loop").strip(),
        trigger=OpenLoopTrigger(
            cues=sorted(set(trigger_obj.cues)),
            project=(trigger_obj.project or "general").lower(),
        ),
        next_action=next_action.strip(),
        closure_condition=closure_condition.strip(),
        tension=max(0.0, min(1.0, tension)),
        status="open",
        source_anchor=source_anchor,
        created_at=_utc_iso(),
    )
    save_open_loop(target_dir / f"{loop.loop_id}.md", loop)
    return loop


def save_open_loop(path: Path, loop: OpenLoopRecord) -> None:
    body = (
        f"Title: {loop.title}\n"
        f"Next action: {loop.next_action}\n"
        f"Closure: {loop.closure_condition}\n"
        f"Trigger cues: {', '.join(loop.trigger.cues)}\n"
    )
    write_markdown(path, loop.to_frontmatter(), body)


def load_open_loops(open_loops_dir: Path) -> list[OpenLoopRecord]:
    loops: list[OpenLoopRecord] = []
    if not open_loops_dir.exists():
        return loops
    for path in sorted(open_loops_dir.glob("*.md")):
        metadata, _ = read_markdown(path)
        if metadata.get("type") != "open_loop":
            continue
        loop = OpenLoopRecord(
            loop_id=str(metadata.get("loop_id", path.stem)),
            title=str(metadata.get("title", metadata.get("intent", ""))),
            trigger=OpenLoopTrigger(
                cues=[str(c) for c in metadata.get("trigger_cues", [])],
                project=str(metadata.get("trigger_project", "general")),
            ),
            next_action=str(metadata.get("next_action", "")),
            closure_condition=str(metadata.get("closure_condition", "")),
            tension=float(metadata.get("tension", 0.0)),
            status=str(metadata.get("status", "open")),
            source_anchor=str(metadata.get("source_anchor", "")),
            created_at=str(metadata.get("created_at", "")),
        )
        loops.append(loop)
    return loops


def trigger_open_loops(loops: list[OpenLoopRecord], query_cues: list[str]) -> list[OpenLoopMatch]:
    query = set(query_cues)
    ranked: list[OpenLoopMatch] = []
    for loop in loops:
        if loop.status != "open":
            continue
        trigger_set = set(loop.trigger.cues + [f"project:{loop.trigger.project}"])
        if not trigger_set:
            continue
        overlap = len(query & trigger_set) / len(query | trigger_set) if (query | trigger_set) else 0.0
        score = (0.7 * overlap) + (0.3 * loop.tension)
        if score <= 0.0:
            continue
        reason = f"overlap={overlap:.3f};tension={loop.tension:.3f}"
        ranked.append(
            OpenLoopMatch(
                loop_id=loop.loop_id,
                trigger_score=round(score, 6),
                tension=loop.tension,
                next_action=loop.next_action,
                trigger_reason=reason,
            )
        )
    ranked.sort(key=lambda item: item.trigger_score, reverse=True)
    return ranked


def evaluate_open_loop_triggers(loops: list[OpenLoopRecord], query_cues: set[str]) -> list[OpenLoopRecord]:
    matches = trigger_open_loops(loops, sorted(query_cues))
    by_id = {loop.loop_id: loop for loop in loops}
    ordered: list[OpenLoopRecord] = []
    for match in matches:
        loop = by_id.get(match.loop_id)
        if loop is not None:
            ordered.append(loop)
    return ordered


def match_open_loops(
    *,
    open_loops_dir: Path,
    query_cues: list[str],
    project: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Compatibility matcher used by engine recall flow."""
    loops = load_open_loops(open_loops_dir)
    effective_cues = set(query_cues)
    if project:
        effective_cues.add(f"project:{project.lower()}")

    raw_matches = trigger_open_loops(loops, sorted(effective_cues))
    by_id = {loop.loop_id: loop for loop in loops}

    ranked: list[dict[str, Any]] = []
    for match in raw_matches[:top_k]:
        loop = by_id.get(match.loop_id)
        if loop is None:
            continue
        ranked.append(
            {
                "loop_id": match.loop_id,
                "activation_score": match.trigger_score,
                "trigger_score": match.trigger_score,
                "tension": match.tension,
                "next_action": match.next_action,
                "trigger_reason": match.trigger_reason,
                "trigger_cues": list(loop.trigger.cues),
                "source_anchor": loop.source_anchor,
            }
        )
    return ranked


def close_open_loop(open_loop_path: Path, *, reason: str = "") -> None:
    metadata, body = read_markdown(open_loop_path)
    metadata["status"] = "closed"
    metadata["closed_at"] = _utc_iso()
    if reason:
        metadata["closure_reason"] = reason
    write_markdown(open_loop_path, metadata, body)
