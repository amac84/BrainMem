from __future__ import annotations

from datetime import date
from pathlib import Path

from brainmem.config import BrainMemConfig
from brainmem.consolidation_job import run_daily_consolidation
from brainmem.markdown_io import read_markdown, write_markdown


def _seed_stream(config: BrainMemConfig, day: str, lines: list[str]) -> None:
    write_markdown(
        config.stream_dir / f"{day}.md",
        {"type": "stream", "date": day},
        "\n".join(lines) + "\n",
    )


def test_weekly_rollup_generated_from_multiple_daily_summaries(tmp_path: Path) -> None:
    config = BrainMemConfig.from_root(tmp_path)
    config.ensure_directories()

    _seed_stream(
        config,
        "2026-04-06",
        [
            '{"event_id":"ev-a1","created_at":"2026-04-06T09:00:00+00:00","project":"atlas","people":["Maya"],"tool":"laptop","place":"office","mode":"planning","emotion":"worried","action":"review","state":{"processing_mode":"planning"},"cues":["project:atlas","token:budget"],"text":"Atlas budget review","factors":{"prediction_error":0.5},"source":"conversation"}'
        ],
    )
    _seed_stream(
        config,
        "2026-04-07",
        [
            '{"event_id":"ev-a2","created_at":"2026-04-07T10:00:00+00:00","project":"atlas","people":["Maya"],"tool":"laptop","place":"office","mode":"execution","emotion":"focused","action":"call","state":{"processing_mode":"execution"},"cues":["project:atlas","token:vendor"],"text":"Vendor confirmation call","factors":{"prediction_error":0.4},"source":"conversation"}'
        ],
    )

    run_daily_consolidation(
        config=config,
        target_date=date(2026, 4, 6),
        boundary_threshold=0.45,
        run_forgetting=False,
    )
    run_daily_consolidation(
        config=config,
        target_date=date(2026, 4, 7),
        boundary_threshold=0.45,
        run_forgetting=False,
    )

    weekly_summary = config.summaries_weekly_dir / "2026-W15.md"
    assert weekly_summary.exists()

    meta, body = read_markdown(weekly_summary)
    assert meta["type"] == "weekly_summary"
    assert meta["daily_count"] >= 2
    assert "Top themes" in body
    assert "atlas" in body.lower()
