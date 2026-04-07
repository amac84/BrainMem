from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(repo_root: Path, args: list[str]) -> dict:
    workspace_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    project_pythonpath = str(workspace_root / "src")
    env["PYTHONPATH"] = (
        f"{project_pythonpath}:{existing_pythonpath}"
        if existing_pythonpath
        else project_pythonpath
    )
    command = [sys.executable, "-m", "brainmem.cli", "--root", str(repo_root), *args]
    proc = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(proc.stdout)


def test_ingest_and_recall_cli_round_trip(tmp_path: Path) -> None:
    ingest_payload = _run_cli(
        repo_root=tmp_path,
        args=[
            "ingest",
            "--text",
            "I need to review budget with Maya before Friday and I am worried.",
            "--people",
            "Maya",
            "--project",
            "Atlas",
            "--tool",
            "laptop",
            "--mode",
            "planning",
            "--emotion",
            "worried",
            "--action",
            "review",
            "--novelty",
            "0.9",
            "--salience",
            "0.8",
            "--goal-relevance",
            "0.9",
            "--surprise",
            "0.7",
            "--unresolved-tension",
            "0.8",
            "--emphasis",
            "0.8",
            "--social-importance",
            "0.7",
            "--decision-irreversibility",
            "0.6",
        ],
    )
    assert ingest_payload["status"] == "ok"
    assert ingest_payload["durable"] is True

    recall_payload = _run_cli(
        repo_root=tmp_path,
        args=[
            "recall",
            "--cues",
            "budget,review",
            "--people",
            "Maya",
            "--project",
            "Atlas",
            "--tool",
            "laptop",
            "--mode",
            "planning",
            "--emotion",
            "worried",
            "--top-k",
            "3",
        ],
    )
    assert recall_payload["status"] == "ok"
    assert recall_payload["results"], "Expected at least one recall result"
    top = recall_payload["results"][0]
    assert top["score"] > 0
    assert top["score_breakdown"]["cue_overlap"] > 0
    assert top["anchor"].endswith(f"event_id={top['memory_id']}")
