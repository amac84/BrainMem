from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .markdown_io import read_markdown, write_markdown


@dataclass(slots=True)
class SchemaCandidate:
    schema_id: str
    condition: str
    action_pattern: str
    expected_outcome: str
    support_count: int
    exception_count: int
    support_anchors: list[str]
    exception_anchors: list[str]


def extract_schemas_from_events(events_dir: Path) -> list[SchemaCandidate]:
    support_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for path in sorted(events_dir.glob("**/*.md")):
        metadata, body = read_markdown(path)
        if metadata.get("type") != "event":
            continue
        project = str(metadata.get("project", "general")).lower()
        mode = str(metadata.get("mode", "unknown")).lower()
        action = str(metadata.get("action", "note")).lower()
        emotion = str(metadata.get("emotion", "neutral")).lower()
        anchor = str(metadata.get("source_anchor", "")) or f"{path.name}#event_id={metadata.get('event_id', path.stem)}"

        condition = f"project:{project}|mode:{mode}"
        key = (condition, action)
        support_map[key].append(
            {
                "anchor": anchor,
                "emotion": emotion,
                "text": body.strip(),
            }
        )

    candidates: list[SchemaCandidate] = []
    for (condition, action), rows in support_map.items():
        emotions = Counter(row["emotion"] for row in rows if row["emotion"])
        dominant_emotion = emotions.most_common(1)[0][0] if emotions else "neutral"
        support_anchors = [row["anchor"] for row in rows[:20]]
        exception_anchors: list[str] = []
        for row in rows:
            if row["emotion"] != dominant_emotion:
                exception_anchors.append(row["anchor"])
        schema_id = f"schema-{abs(hash((condition, action))) % 10_000_000:07d}"
        candidates.append(
            SchemaCandidate(
                schema_id=schema_id,
                condition=condition,
                action_pattern=action,
                expected_outcome=f"emotion_trend:{dominant_emotion}",
                support_count=len(rows),
                exception_count=len(exception_anchors),
                support_anchors=support_anchors,
                exception_anchors=exception_anchors[:20],
            )
        )

    candidates.sort(key=lambda c: (c.support_count, -c.exception_count), reverse=True)
    return candidates


def write_schema_files(schemas_dir: Path, candidates: list[SchemaCandidate], top_k: int = 25) -> list[Path]:
    schemas_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for schema in candidates[:top_k]:
        path = schemas_dir / f"{schema.schema_id}.md"
        metadata = {
            "type": "schema",
            "schema_id": schema.schema_id,
            "condition": schema.condition,
            "action_pattern": schema.action_pattern,
            "expected_outcome": schema.expected_outcome,
            "support_count": schema.support_count,
            "exception_count": schema.exception_count,
            "support_anchors": schema.support_anchors,
            "exception_anchors": schema.exception_anchors,
        }
        body_lines = [
            "## Schema",
            f"- condition: {schema.condition}",
            f"- action: {schema.action_pattern}",
            f"- expected outcome: {schema.expected_outcome}",
            "",
            "## Supports",
        ]
        body_lines.extend(f"- {anchor}" for anchor in schema.support_anchors[:20])
        body_lines.append("")
        body_lines.append("## Exceptions")
        if schema.exception_anchors:
            body_lines.extend(f"- {anchor}" for anchor in schema.exception_anchors[:20])
        else:
            body_lines.append("- none observed")
        write_markdown(path, metadata, "\n".join(body_lines).strip() + "\n")
        written.append(path)
    return written


def extract_and_write_schemas(events_dir: Path, schemas_dir: Path) -> dict[str, Any]:
    candidates = extract_schemas_from_events(events_dir)
    written = write_schema_files(schemas_dir, candidates)
    return {
        "schema_count": len(written),
        "candidates_considered": len(candidates),
        "schema_paths": [str(p) for p in written],
    }


def extract_weekly_schema(
    *,
    summaries_daily_dir: Path,
    schemas_dir: Path,
    week_id: str,
) -> dict[str, Any]:
    """Create weekly schema from daily summaries with exception tracking."""
    schemas_dir.mkdir(parents=True, exist_ok=True)
    focus_counter: Counter[str] = Counter()
    emotion_counter: Counter[str] = Counter()
    anchors: list[str] = []
    exceptions: list[dict[str, str]] = []
    daily_count = 0

    for daily_path in sorted(summaries_daily_dir.glob("*.md")):
        try:
            d = date.fromisoformat(daily_path.stem)
        except ValueError:
            continue
        current_week = f"{d.isocalendar().year}-W{d.isocalendar().week:02d}"
        if current_week != week_id:
            continue

        daily_count += 1
        meta, body = read_markdown(daily_path)
        anchors.extend([str(a) for a in meta.get("anchors", []) if str(a)])

        local_focus = ""
        local_emotion = ""
        for line in body.splitlines():
            s = line.strip()
            if s.startswith("- Primary focus was "):
                local_focus = s.replace("- Primary focus was ", "").split(" across ", 1)[0].strip()
                if local_focus:
                    focus_counter[local_focus] += 1
            elif s.startswith("- Dominant emotional tone was "):
                local_emotion = s.replace("- Dominant emotional tone was ", "").rstrip(".").strip()
                if local_emotion:
                    emotion_counter[local_emotion] += 1

        if local_focus and local_emotion and local_emotion.lower() in {"worried", "angry", "frustrated"}:
            exceptions.append(
                {
                    "daily_summary": daily_path.name,
                    "reason": f"negative_emotion:{local_emotion}",
                    "focus": local_focus,
                }
            )

    top_focus = [name for name, _ in focus_counter.most_common(5)]
    top_emotions = [name for name, _ in emotion_counter.most_common(5)]
    schema_path = schemas_dir / f"{week_id}.md"

    metadata = {
        "type": "weekly_schema",
        "week_id": week_id,
        "daily_count": daily_count,
        "themes": {name: count for name, count in focus_counter.most_common(10)},
        "support_count": sum(focus_counter.values()),
        "exception_count": len(exceptions),
        "emotion_distribution": {name: count for name, count in emotion_counter.most_common(10)},
        "anchors": sorted(set(anchors))[:40],
        "exceptions": exceptions,
    }
    body_lines = ["## Top themes"]
    body_lines.extend(f"- {focus}" for focus in top_focus or ["none"])
    body_lines.append("")
    body_lines.append("## Emotion patterns")
    body_lines.extend(f"- {emo}" for emo in top_emotions or ["none"])
    body_lines.append("")
    body_lines.append("## Exceptions")
    if exceptions:
        body_lines.extend(
            f"- {item['daily_summary']}: {item['reason']} ({item['focus']})"
            for item in exceptions
        )
    else:
        body_lines.append("- none observed")
    body_lines.append("")
    body_lines.append("## Anchors")
    body_lines.extend(f"- {a}" for a in sorted(set(anchors))[:40])

    write_markdown(schema_path, metadata, "\n".join(body_lines).strip() + "\n")
    return {
        "week_id": week_id,
        "daily_count": daily_count,
        "themes": list(focus_counter.most_common(10)),
        "exceptions": exceptions,
        "schema_path": str(schema_path),
    }


def extract_schema_from_weekly_summaries(
    *,
    summaries_weekly_dir: Path,
    schemas_dir: Path,
    schema_id: str,
) -> dict[str, Any]:
    """Create a schema file from existing weekly summaries."""
    schemas_dir.mkdir(parents=True, exist_ok=True)
    focus_counter: Counter[str] = Counter()
    emotion_counter: Counter[str] = Counter()
    anchors: list[str] = []
    source_weeks: list[str] = []
    exceptions: list[str] = []

    for weekly_path in sorted(summaries_weekly_dir.glob("*.md")):
        meta, body = read_markdown(weekly_path)
        if meta.get("type") != "weekly_summary":
            continue
        source_weeks.append(str(meta.get("week", weekly_path.stem)))
        anchors.extend([str(a) for a in meta.get("anchors", []) if str(a)])
        for line in body.splitlines():
            s = line.strip().lower()
            if s.startswith("- recurring focus:"):
                content = s.replace("- recurring focus:", "").strip().rstrip(".")
                for bit in [b.strip() for b in content.split(",") if b.strip()]:
                    focus_counter[bit] += 1
            elif s.startswith("- recurring emotional tones:"):
                content = s.replace("- recurring emotional tones:", "").strip().rstrip(".")
                for bit in [b.strip() for b in content.split(",") if b.strip()]:
                    emotion_counter[bit] += 1
            elif "exception" in s:
                exceptions.append(line.strip())

    schema_path = schemas_dir / f"{schema_id}.md"
    metadata = {
        "type": "schema_from_weekly",
        "schema_id": schema_id,
        "source_week_count": len(source_weeks),
        "source_weeks": source_weeks,
        "focus_distribution": {k: v for k, v in focus_counter.most_common(10)},
        "emotion_distribution": {k: v for k, v in emotion_counter.most_common(10)},
        "anchors": sorted(set(anchors))[:40],
        "exceptions": exceptions[:20],
    }
    body_lines = ["## Schema from weekly rollups"]
    body_lines.append("")
    body_lines.append("### Focus patterns")
    body_lines.extend(f"- {name}: {count}" for name, count in focus_counter.most_common(10))
    if not focus_counter:
        body_lines.append("- none")
    body_lines.append("")
    body_lines.append("### Emotion patterns")
    body_lines.extend(f"- {name}: {count}" for name, count in emotion_counter.most_common(10))
    if not emotion_counter:
        body_lines.append("- none")
    body_lines.append("")
    body_lines.append("### Exceptions")
    body_lines.extend(f"- {item}" for item in exceptions[:20])
    if not exceptions:
        body_lines.append("- none observed")
    body_lines.append("")
    body_lines.append("### Anchors")
    body_lines.extend(f"- {a}" for a in sorted(set(anchors))[:40])
    write_markdown(schema_path, metadata, "\n".join(body_lines).strip() + "\n")
    return {
        "schema_id": schema_id,
        "schema_path": str(schema_path),
        "source_week_count": len(source_weeks),
    }
