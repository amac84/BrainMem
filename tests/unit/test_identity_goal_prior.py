from __future__ import annotations

from pathlib import Path

from brainmem.identity_goal_prior import (
    ensure_identity_goal_files,
    load_beliefs,
    log_belief_conflict,
    score_identity_goal_prior,
    update_identity_and_goals_from_event,
    update_identity_goal_evidence,
    upsert_belief,
)


def test_identity_belief_upsert_and_scoring(tmp_path: Path) -> None:
    identity_dir = tmp_path / "memory" / "identity"
    goals_dir = tmp_path / "memory" / "goals"
    ensure_identity_goal_files(identity_dir, goals_dir)

    belief_path = upsert_belief(
        identity_dir=identity_dir,
        statement="User values reliability in project delivery",
        tags=["reliability", "delivery"],
        evidence_anchor="2026-04-07.md#event_id=ev-1",
        confidence_delta=0.2,
    )
    assert belief_path.exists()

    score = score_identity_goal_prior(
        cues=["token:reliability", "project:atlas", "token:delivery"],
        project="atlas",
        identity_dir=identity_dir,
        goals_dir=goals_dir,
    )
    assert score["score"] > 0
    assert score["matched_beliefs"]


def test_identity_conflict_logged_on_opposing_evidence(tmp_path: Path) -> None:
    identity_dir = tmp_path / "memory" / "identity"
    goals_dir = tmp_path / "memory" / "goals"
    ensure_identity_goal_files(identity_dir, goals_dir)

    update_identity_and_goals_from_event(
        identity_dir=identity_dir,
        goals_dir=goals_dir,
        event_id="ev-2",
        event_text="I always avoid rushed decisions.",
        cues=["token:always", "token:avoid", "token:rushed", "token:decisions"],
        project="atlas",
        source_anchor="2026-04-07.md#event_id=ev-2",
    )

    conflict = log_belief_conflict(
        identity_dir=identity_dir,
        belief_statement="I always avoid rushed decisions.",
        conflicting_evidence_anchor="2026-04-08.md#event_id=ev-3",
        note="Contradiction: user accepted rushed vendor change.",
    )
    assert conflict.exists()

    beliefs = load_beliefs(identity_dir)
    assert beliefs


def test_update_identity_goal_evidence_updates_existing_belief(tmp_path: Path) -> None:
    identity_dir = tmp_path / "memory" / "identity"
    goals_dir = tmp_path / "memory" / "goals"
    ensure_identity_goal_files(identity_dir, goals_dir)

    upsert_belief(
        identity_dir=identity_dir,
        statement="User values reliability in project delivery",
        tags=["reliability"],
        evidence_anchor="2026-04-07.md#event_id=ev-1",
        confidence_delta=0.1,
    )
    updated = update_identity_goal_evidence(
        identity_dir=identity_dir,
        goals_dir=goals_dir,
        belief_statement="User values reliability in project delivery",
        evidence_anchor="2026-04-08.md#event_id=ev-4",
        confidence_delta=0.15,
    )
    assert updated["belief_path"]
    assert Path(updated["belief_path"]).exists()
