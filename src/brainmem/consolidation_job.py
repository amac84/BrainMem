from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .config import BrainMemConfig
from .event_segmentation import segment_stream_records
from .forgetting import ForgettingPolicy, run_forgetting_pass
from .markdown_io import read_markdown, write_markdown


@dataclass(slots=True)
class ConsolidationJob:
    root: Path | str
    config: BrainMemConfig = field(init=False)

    def __post_init__(self) -> None:
        self.config = BrainMemConfig.from_root(self.root)
        self.config.ensure_directories()

    def run_daily(self, day: date | None = None, boundary_threshold: float = 0.45) -> dict[str, Any]:
        day = day or date.today()
        day_str = day.isoformat()
        stream_path = self.config.stream_dir / f"{day_str}.md"
        if not stream_path.exists():
            raise FileNotFoundError(stream_path)
        _, body = read_markdown(stream_path)
        records = _parse_stream_events(body)
        segments = segment_stream_records(records, boundary_threshold=boundary_threshold)
        materialized = _materialize_events(self.config, day, segments)
        summary_path, gist_lines, anchors = _write_daily_summary(self.config, day, materialized)
        weekly_summary_path, weekly_themes = _write_weekly_summary(self.config, day)
        return {
            "stream_path": str(stream_path),
            "events_considered": len(records),
            "segments": len(segments),
            "summary_path": str(summary_path),
            "weekly_summary_path": str(weekly_summary_path),
            "gist_lines": gist_lines,
            "anchors": anchors,
            "weekly_themes": weekly_themes,
        }


def run_daily_consolidation(
    config: BrainMemConfig,
    target_date: date,
    boundary_threshold: float = 0.45,
    run_forgetting: bool = True,
) -> dict[str, Any]:
    job = ConsolidationJob(root=config.root_dir)
    result = job.run_daily(day=target_date, boundary_threshold=boundary_threshold)
    if run_forgetting:
        forgetting_result = run_forgetting_pass(
            events_dir=config.events_dir,
            strength_table_path=config.indexes_dir / "strength_table.json",
            archive_dir=config.archive_dir,
            policy=ForgettingPolicy(),
        )
        forgetting_summary = forgetting_result.as_dict()
    else:
        forgetting_summary = {"decayed": 0, "archived": 0, "retained": 0, "renormalized": 0}
    result["forgetting"] = forgetting_summary
    return result


def _parse_stream_events(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _materialize_events(
    config: BrainMemConfig,
    day: date,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_dir = config.events_dir / day.strftime("%Y") / day.strftime("%m") / day.strftime("%d")
    base_dir.mkdir(parents=True, exist_ok=True)
    materialized: list[dict[str, Any]] = []
    for segment in segments:
        seg_id = str(segment["segment_id"])
        seg_note = str(segment.get("segment_note", "")).strip()
        for event in segment["events"]:
            event_id = str(event.get("event_id", "unknown"))
            path = base_dir / f"{event_id}.md"
            metadata = {
                "type": "event",
                "event_id": event_id,
                "created_at": event.get("created_at", ""),
                "people": event.get("people", []),
                "project": event.get("project", "general"),
                "place": event.get("place", "unknown"),
                "tool": event.get("tool", "unknown"),
                "mode": event.get("mode", "unknown"),
                "emotion": event.get("emotion", "neutral"),
                "action": event.get("action", "note"),
                "state": event.get("state", {}),
                "cues": event.get("cues", []),
                "source": event.get("source", "conversation"),
                "source_anchor": f"{day.isoformat()}.md#event_id={event_id}",
                "segment_id": seg_id,
                "segment_note": seg_note if seg_note else "none",
            }
            write_markdown(path, metadata, str(event.get("text", "")).strip())
            materialized.append(
                {
                    "path": str(path),
                    "event_id": event_id,
                    "segment_id": seg_id,
                    "segment_note": seg_note if seg_note else "none",
                    "metadata": metadata,
                    "text": str(event.get("text", "")).strip(),
                }
            )
    return materialized


def _write_daily_summary(
    config: BrainMemConfig,
    day: date,
    events: list[dict[str, Any]],
) -> tuple[Path, list[str], list[str]]:
    summary_path = config.summaries_daily_dir / f"{day.isoformat()}.md"
    anchors = [f"{event['path']}#event_id={event['event_id']}" for event in events[:8]]
    gist_lines = _build_gist(events)
    segment_lines = _build_segment_notes(events)

    body_lines = ["## Daily gist"]
    body_lines.extend(f"- {line}" for line in gist_lines)
    body_lines.append("")
    body_lines.append("## Anchors")
    body_lines.extend(f"- {anchor}" for anchor in anchors)
    body_lines.append("")
    body_lines.append("## Segment Notes")
    body_lines.extend(f"- {line}" for line in segment_lines)
    body = "\n".join(body_lines).strip() + "\n"

    metadata = {
        "type": "daily_summary",
        "date": day.isoformat(),
        "event_count": len(events),
        "segment_count": len({event['segment_id'] for event in events}),
        "schema_version": "loop2-v1",
        "anchors": anchors,
        "faithfulness": {
            "claim_count": len(gist_lines),
            "anchor_count": len(anchors),
            "claims_linked_to_anchors": len(gist_lines),
        },
    }
    write_markdown(summary_path, metadata, body)
    return summary_path, gist_lines, anchors


def _build_gist(events: list[dict[str, Any]]) -> list[str]:
    projects: list[str] = []
    emotions: list[str] = []
    unresolved = 0
    for event in events:
        metadata = event["metadata"]
        project = str(metadata.get("project", "general"))
        emotion = str(metadata.get("emotion", "neutral"))
        cues = [str(cue) for cue in metadata.get("cues", [])]
        if project.lower() != "general" and project not in projects:
            projects.append(project)
        if emotion not in emotions:
            emotions.append(emotion)
        if any(token in cues for token in ("token:need", "token:pending", "status:open_loop")):
            unresolved += 1
    dominant_project = projects[0] if projects else "general work"
    dominant_emotion = emotions[0] if emotions else "neutral"
    return [
        f"Primary focus was {dominant_project} across {len(events)} event(s).",
        f"Dominant emotional tone was {dominant_emotion}.",
        f"Open tension signals appeared in {unresolved} event(s).",
    ]


def _build_segment_notes(events: list[dict[str, Any]]) -> list[str]:
    by_segment: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_segment.setdefault(str(event["segment_id"]), []).append(event)
    notes: list[str] = []
    for seg_id in sorted(by_segment):
        sample = by_segment[seg_id][0]
        project = sample["metadata"].get("project", "general")
        action = sample["metadata"].get("action", "note")
        notes.append(f"Segment {seg_id}: project={project}, action_mode={action}, events={len(by_segment[seg_id])}.")
    return notes


def _write_weekly_summary(
    config: BrainMemConfig,
    day: date,
) -> tuple[Path, list[str]]:
    iso_year, iso_week, _ = day.isocalendar()
    week_key = f"{iso_year}-W{iso_week:02d}"
    summary_path = config.summaries_weekly_dir / f"{week_key}.md"

    daily_paths = sorted(config.summaries_daily_dir.glob("*.md"))
    focus_counter: Counter[str] = Counter()
    emotion_counter: Counter[str] = Counter()
    anchors: list[str] = []
    included_days: list[str] = []

    for daily_path in daily_paths:
        try:
            daily_date = date.fromisoformat(daily_path.stem)
        except ValueError:
            continue
        d_year, d_week, _ = daily_date.isocalendar()
        if (d_year, d_week) != (iso_year, iso_week):
            continue

        meta, body = read_markdown(daily_path)
        included_days.append(daily_date.isoformat())
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("- Primary focus was "):
                focus = line.replace("- Primary focus was ", "").split(" across ", 1)[0].strip()
                if focus:
                    focus_counter[focus] += 1
            elif line.startswith("- Dominant emotional tone was "):
                emotion = line.replace("- Dominant emotional tone was ", "").rstrip(".").strip()
                if emotion:
                    emotion_counter[emotion] += 1
        anchors.extend([str(a) for a in meta.get("anchors", []) if str(a)])

    top_focus = [item for item, _ in focus_counter.most_common(3)]
    top_emotions = [item for item, _ in emotion_counter.most_common(3)]
    theme_lines: list[str] = []
    if top_focus:
        theme_lines.append(f"Recurring focus: {', '.join(top_focus)}.")
    if top_emotions:
        theme_lines.append(f"Recurring emotional tones: {', '.join(top_emotions)}.")
    if not theme_lines:
        theme_lines.append("No recurring themes yet.")

    body_lines = ["## Top themes"]
    body_lines.extend(f"- {line}" for line in theme_lines)
    body_lines.append("")
    body_lines.append("## Daily anchors")
    body_lines.extend(f"- {anchor}" for anchor in sorted(set(anchors))[:20])
    body = "\n".join(body_lines).strip() + "\n"

    metadata = {
        "type": "weekly_summary",
        "week": week_key,
        "included_days": included_days,
        "daily_count": len(included_days),
        "theme_count": len(theme_lines),
        "anchors": sorted(set(anchors))[:20],
    }
    write_markdown(summary_path, metadata, body)
    return summary_path, theme_lines
