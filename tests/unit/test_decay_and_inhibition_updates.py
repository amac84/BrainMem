from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from brainmem.forgetting import ForgettingPolicy, run_forgetting_pass


def test_forgetting_decay_reduces_strength_for_older_memory(tmp_path: Path) -> None:
    events_dir = tmp_path / "memory" / "events" / "2026" / "04" / "07"
    events_dir.mkdir(parents=True, exist_ok=True)
    event_path = events_dir / "ev-old.md"
    event_path.write_text("---\ntype: event\nevent_id: ev-old\n---\nold event\n", encoding="utf-8")

    old_time = (datetime.now(timezone.utc) - timedelta(days=10)).replace(microsecond=0).isoformat()
    strength_table = tmp_path / "indexes" / "strength_table.json"
    strength_table.parent.mkdir(parents=True, exist_ok=True)
    strength_table.write_text(
        (
            '{\n'
            '  "ev-old": {\n'
            f'    "path": "{event_path}",\n'
            '    "strength": 0.8,\n'
            f'    "updated_at": "{old_time}"\n'
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    result = run_forgetting_pass(
        events_dir=events_dir,
        strength_table_path=strength_table,
        archive_dir=tmp_path / "memory" / "archive",
        policy=ForgettingPolicy(decay_lambda_per_day=0.2, prune_threshold=0.05),
    )
    assert result.decayed >= 1

    content = strength_table.read_text(encoding="utf-8")
    assert '"strength":' in content
    assert '"ev-old"' in content


def test_forgetting_archives_below_threshold(tmp_path: Path) -> None:
    events_dir = tmp_path / "memory" / "events" / "2026" / "04" / "07"
    events_dir.mkdir(parents=True, exist_ok=True)
    event_path = events_dir / "ev-low.md"
    event_path.write_text("---\ntype: event\nevent_id: ev-low\n---\nlow event\n", encoding="utf-8")

    strength_table = tmp_path / "indexes" / "strength_table.json"
    strength_table.parent.mkdir(parents=True, exist_ok=True)
    strength_table.write_text(
        (
            '{\n'
            '  "ev-low": {\n'
            f'    "path": "{event_path}",\n'
            '    "strength": 0.01,\n'
            f'    "updated_at": "{datetime.now(timezone.utc).replace(microsecond=0).isoformat()}"\n'
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    archive_dir = tmp_path / "memory" / "archive"
    result = run_forgetting_pass(
        events_dir=events_dir,
        strength_table_path=strength_table,
        archive_dir=archive_dir,
        policy=ForgettingPolicy(decay_lambda_per_day=0.0, prune_threshold=0.02),
    )
    assert result.archived == 1
    assert not event_path.exists()
    archived_files = list(archive_dir.glob("*.md"))
    assert archived_files, "Expected archived markdown file"
