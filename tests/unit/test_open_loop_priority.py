from __future__ import annotations

from pathlib import Path

from brainmem.open_loops import (
    OpenLoopRecord,
    create_open_loop,
    load_open_loops,
    save_open_loop,
    trigger_open_loops,
)


def test_open_loop_trigger_prioritizes_tension_and_deadline(tmp_path: Path) -> None:
    loops_dir = tmp_path / "memory" / "open_loops"

    loop_a = create_open_loop(
        loops_dir=loops_dir,
        title="Atlas vendor follow-up",
        trigger_cues=["project:atlas", "person:maya", "token:vendor"],
        next_action="Message vendor for dependency update",
        closure_condition="vendor confirms timeline",
        tension=0.9,
    )
    loop_b = create_open_loop(
        loops_dir=loops_dir,
        title="Personal errands",
        trigger_cues=["project:home", "token:grocery"],
        next_action="Buy groceries",
        closure_condition="fridge restocked",
        tension=0.4,
    )
    assert loop_a.loop_id != loop_b.loop_id

    active = load_open_loops(loops_dir)
    matches = trigger_open_loops(active, query_cues=["project:atlas", "token:vendor", "person:maya"])
    assert matches
    assert matches[0].loop_id == loop_a.loop_id
    assert matches[0].trigger_score > 0.0


def test_closed_open_loop_is_not_triggered(tmp_path: Path) -> None:
    loops_dir = tmp_path / "memory" / "open_loops"
    loop = create_open_loop(
        loops_dir=loops_dir,
        title="Close me",
        trigger_cues=["project:atlas"],
        next_action="done",
        closure_condition="done",
        tension=0.7,
    )
    loop.status = "closed"
    save_open_loop(loops_dir / f"{loop.loop_id}.md", loop)

    active = load_open_loops(loops_dir)
    matches = trigger_open_loops(active, query_cues=["project:atlas"])
    assert not matches
