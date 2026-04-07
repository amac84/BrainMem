from __future__ import annotations

from brainmem.state_inference import infer_state_with_confidence


def test_state_inference_confidence_with_direct_metadata() -> None:
    state, confidence = infer_state_with_confidence(
        "Reviewed plan with Maya in office on laptop",
        metadata={
            "location": "office",
            "tool": "laptop",
            "mode": "planning",
            "people": ["Maya"],
            "emotion": "focused",
        },
    )
    assert state.location == "office"
    assert state.device_context == "laptop"
    assert state.processing_mode == "planning"
    assert confidence["overall"] >= 0.9


def test_state_inference_confidence_uses_text_heuristics_when_sparse() -> None:
    state, confidence = infer_state_with_confidence(
        "Met Maya at cafe and reviewed blocker on phone while frustrated",
        metadata={},
    )
    assert state.location == "cafe"
    assert state.device_context == "phone"
    assert state.social_context == "social"
    assert state.physio_proxy == "frustrated"
    assert 0.45 <= confidence["overall"] <= 0.8
