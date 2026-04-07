from __future__ import annotations

from pathlib import Path

from brainmem.markdown_io import read_markdown, write_markdown
from brainmem.schema_extraction import (
    extract_schema_from_weekly_summaries,
    extract_weekly_schema,
)


def test_extract_weekly_schema_tracks_support_and_exceptions(tmp_path: Path) -> None:
    summaries_dir = tmp_path / "memory" / "summaries" / "daily"
    schemas_dir = tmp_path / "memory" / "schemas"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    schemas_dir.mkdir(parents=True, exist_ok=True)

    write_markdown(
        summaries_dir / "2026-04-07.md",
        {
            "type": "daily_summary",
            "date": "2026-04-07",
            "anchors": ["memory/events/ev-1.md#event_id=ev-1"],
        },
        "## Daily gist\n- Primary focus was Atlas across 2 event(s).\n- Dominant emotional tone was worried.\n",
    )
    write_markdown(
        summaries_dir / "2026-04-08.md",
        {
            "type": "daily_summary",
            "date": "2026-04-08",
            "anchors": ["memory/events/ev-2.md#event_id=ev-2"],
        },
        "## Daily gist\n- Primary focus was Atlas across 3 event(s).\n- Dominant emotional tone was focused.\n- Exception: No vendor follow-up completed.\n",
    )

    result = extract_weekly_schema(
        summaries_daily_dir=summaries_dir,
        schemas_dir=schemas_dir,
        week_id="2026-W15",
    )
    assert result["daily_count"] == 2
    assert result["schema_path"]

    meta, body = read_markdown(Path(result["schema_path"]))
    assert meta["type"] == "weekly_schema"
    assert meta["support_count"] >= 1
    assert meta["exception_count"] >= 1
    assert "## Exceptions" in body


def test_extract_schema_from_weekly_summaries_returns_path(tmp_path: Path) -> None:
    summaries_weekly_dir = tmp_path / "memory" / "summaries" / "weekly"
    schemas_dir = tmp_path / "memory" / "schemas"
    summaries_weekly_dir.mkdir(parents=True, exist_ok=True)
    schemas_dir.mkdir(parents=True, exist_ok=True)

    write_markdown(
        summaries_weekly_dir / "2026-W15.md",
        {
            "type": "weekly_summary",
            "week": "2026-W15",
            "included_days": ["2026-04-07"],
            "anchors": ["memory/events/ev-1.md#event_id=ev-1"],
        },
        "## Weekly themes\n- Recurring focus: Atlas.\n- Recurring emotional tones: worried.\n",
    )

    result = extract_schema_from_weekly_summaries(
        summaries_weekly_dir=summaries_weekly_dir,
        schemas_dir=schemas_dir,
        schema_id="schema-atlas",
    )
    assert result["schema_path"]
    assert Path(result["schema_path"]).exists()
