from __future__ import annotations

from datetime import date
from pathlib import Path

from brainmem.consolidation_job import run_daily_consolidation
from brainmem.config import BrainMemConfig
from brainmem.markdown_io import write_markdown


def test_reversible_pruning_moves_low_strength_event_to_archive(tmp_path: Path) -> None:
    config = BrainMemConfig.from_root(tmp_path)
    config.ensure_directories()

    # Seed stream so consolidation can run
    stream_path = config.stream_dir / "2026-04-07.md"
    write_markdown(
        stream_path,
        {"type": "stream", "date": "2026-04-07"},
        '{"event_id":"ev-old","created_at":"2026-04-07T00:00:00+00:00","project":"atlas","people":[],"tool":"laptop","place":"unknown","mode":"planning","emotion":"neutral","action":"review","state":{"processing_mode":"planning"},"cues":["project:atlas"],"text":"old event","factors":{"novelty":0.1},"source":"conversation"}\n',
    )

    old_event_path = config.events_dir / "2026" / "04" / "07" / "ev-old.md"
    write_markdown(
        old_event_path,
        {
            "type": "event",
            "event_id": "ev-old",
            "created_at": "2026-01-01T00:00:00+00:00",
            "project": "atlas",
            "state": {"processing_mode": "planning"},
            "cues": ["project:atlas"],
            "source_anchor": "2026-04-07.md#event_id=ev-old",
        },
        "old event",
    )

    strength_table = config.indexes_dir / "strength_table.json"
    strength_table.write_text(
        '{"ev-old":{"path":"%s","strength":0.05,"updated_at":"2026-01-01T00:00:00+00:00"}}'
        % str(old_event_path),
        encoding="utf-8",
    )

    result = run_daily_consolidation(
        config=config,
        target_date=date(2026, 4, 7),
        boundary_threshold=0.45,
        run_forgetting=True,
    )

    forgetting = result["forgetting"]
    assert forgetting["archived"] >= 1
    assert not old_event_path.exists()
    archived_files = list(config.archive_dir.glob("*.md"))
    assert archived_files, "Expected archived markdown file for reversible pruning"
