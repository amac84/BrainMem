from pathlib import Path

from brainmem.recall_scoring import score_candidate
from brainmem.types import RecallRequest


def test_recall_scoring_prefers_high_overlap_and_state_match() -> None:
    request = RecallRequest(
        cues=[
            "person:alex",
            "project:apollo",
            "token:budget",
            "token:call",
            "emotion:anxious",
        ],
        people=["alex"],
        place="office",
        tool="laptop",
        mode="planning",
        project="apollo",
        emotion="anxious",
        top_k=3,
    )

    good = score_candidate(
        request,
        memory_id="ev-good",
        path=Path("memory/events/ev-good.md"),
        cues=[
            "person:alex",
            "project:apollo",
            "token:budget",
            "token:call",
            "emotion:anxious",
        ],
        state={
            "location": "office",
            "device_context": "laptop",
            "social_context": "team",
            "processing_mode": "planning",
        },
        anchor="stream#ev-good",
        excerpt="good",
    )
    weak = score_candidate(
        request,
        memory_id="ev-weak",
        path=Path("memory/events/ev-weak.md"),
        cues=[
            "person:alex",
            "project:other",
            "token:chores",
            "emotion:neutral",
        ],
        state={
            "location": "home",
            "device_context": "phone",
            "social_context": "alone",
            "processing_mode": "reflection",
        },
        anchor="stream#ev-weak",
        excerpt="weak",
    )

    assert good.total_score > weak.total_score
    assert good.score_breakdown["cue_overlap"] > weak.score_breakdown["cue_overlap"]


def test_recall_scoring_open_loop_boost_applies() -> None:
    request = RecallRequest(
        cues=["project:apollo", "token:handoff"],
        people=[],
        place="unknown",
        tool="unknown",
        mode="unknown",
        project="apollo",
        emotion="neutral",
        top_k=2,
    )
    boosted = score_candidate(
        request,
        memory_id="ev-loop",
        path=Path("memory/events/ev-loop.md"),
        cues=["project:apollo", "token:handoff", "status:open_loop"],
        state={},
        anchor="stream#ev-loop",
        excerpt="loop",
    )
    plain = score_candidate(
        request,
        memory_id="ev-plain",
        path=Path("memory/events/ev-plain.md"),
        cues=["project:apollo", "token:handoff"],
        state={},
        anchor="stream#ev-plain",
        excerpt="plain",
    )
    assert boosted.score_breakdown["open_loop_activation"] == 1.0
    assert boosted.total_score > plain.total_score
