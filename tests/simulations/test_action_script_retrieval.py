from __future__ import annotations

from pathlib import Path

from brainmem.action_scripts import extract_script_signature, rank_scripts_for_query
from brainmem.markdown_io import write_markdown


def test_action_script_retrieval_prefers_matching_tool_action(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "memory" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    write_markdown(
        scripts_dir / "atlas_budget_review.md",
        {
            "type": "action_script",
            "script_id": "script-atlas-budget",
            "actions": ["review", "call", "plan"],
            "tools": ["laptop", "zoom"],
            "projects": ["atlas"],
            "success_rate": 0.85,
        },
        "review budget -> call vendor -> plan handoff",
    )
    write_markdown(
        scripts_dir / "home_errands.md",
        {
            "type": "action_script",
            "script_id": "script-home-errands",
            "actions": ["buy", "book"],
            "tools": ["phone"],
            "projects": ["home"],
            "success_rate": 0.6,
        },
        "book dentist -> buy groceries",
    )

    query = extract_script_signature(
        text="review vendor handoff plan",
        action="review",
        tool="laptop",
        project="atlas",
    )
    ranked = rank_scripts_for_query(query, scripts_dir, top_k=3)
    assert ranked
    assert ranked[0]["script_id"] == "script-atlas-budget"
    assert ranked[0]["score"] > ranked[1]["score"]
