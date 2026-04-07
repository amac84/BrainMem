from __future__ import annotations

from pathlib import Path

from brainmem.associative_graph import (
    activate_from_cues,
    apply_pattern_separation,
    build_hig_adjacency,
)
from brainmem.markdown_io import write_markdown


def test_hig_activation_prefers_matching_memory(tmp_path: Path) -> None:
    events_dir = tmp_path / "memory" / "events" / "2026" / "04" / "07"
    events_dir.mkdir(parents=True, exist_ok=True)

    write_markdown(
        events_dir / "ev-a.md",
        {
            "type": "event",
            "event_id": "ev-a",
            "cues": ["project:atlas", "person:maya", "token:budget", "action:review"],
            "state": {"processing_mode": "planning"},
            "project": "atlas",
        },
        "Atlas budget review with Maya",
    )
    write_markdown(
        events_dir / "ev-b.md",
        {
            "type": "event",
            "event_id": "ev-b",
            "cues": ["project:home", "person:noah", "token:dentist", "action:book"],
            "state": {"processing_mode": "execution"},
            "project": "home",
        },
        "Book dentist appointment",
    )

    adjacency = build_hig_adjacency(tmp_path / "memory" / "events")
    activated = activate_from_cues(
        query_cues=["project:atlas", "person:maya", "token:budget"],
        adjacency=adjacency,
        top_k=5,
    )

    assert activated
    assert activated[0].memory_id == "ev-a"
    assert activated[0].activation_score > 0
    assert "project:atlas" in activated[0].matched_cues


def test_pattern_separation_flags_near_duplicate_candidates(tmp_path: Path) -> None:
    events_dir = tmp_path / "memory" / "events" / "2026" / "04" / "08"
    events_dir.mkdir(parents=True, exist_ok=True)

    write_markdown(
        events_dir / "ev-1.md",
        {
            "type": "event",
            "event_id": "ev-1",
            "cues": ["project:atlas", "person:maya", "token:budget", "tool:laptop", "action:review"],
            "state": {},
            "project": "atlas",
        },
        "Budget discussion version one",
    )
    write_markdown(
        events_dir / "ev-2.md",
        {
            "type": "event",
            "event_id": "ev-2",
            "cues": ["project:atlas", "person:maya", "token:budget", "tool:laptop", "action:review", "token:vendor"],
            "state": {},
            "project": "atlas",
        },
        "Budget discussion version two",
    )

    adjacency = build_hig_adjacency(tmp_path / "memory" / "events")
    activated = activate_from_cues(
        query_cues=["project:atlas", "person:maya", "token:budget"],
        adjacency=adjacency,
        top_k=5,
    )
    separated = apply_pattern_separation(activated, adjacency, overlap_threshold=0.75)

    assert len(separated) >= 2
    assert separated[1].disambiguation_required is True
    assert separated[1].pattern_separation_hint
