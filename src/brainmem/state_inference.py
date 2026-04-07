"""State inference for current and encoded memory context."""

from __future__ import annotations

from datetime import datetime, timezone
import re

from .types import MemoryState


def _bucket_time_of_day(timestamp: datetime) -> str:
    hour = timestamp.hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def infer_state_from_metadata(
    *,
    now: datetime,
    place: str,
    tool: str,
    mode: str,
    people: list[str],
    emotion: str,
) -> MemoryState:
    processing_mode = mode if mode and mode != "unknown" else "general"
    social_context = "alone" if not people else "social"
    return MemoryState(
        time_of_day=_bucket_time_of_day(now),
        location=place or "unknown",
        device_context=tool or "unknown",
        social_context=str(social_context),
        physio_proxy=emotion if emotion and emotion != "neutral" else "unknown",
        processing_mode=str(processing_mode),
    )


def infer_state(text: str, metadata: dict | None = None) -> MemoryState:
    """Compatibility wrapper used by older callers/tests."""
    metadata = metadata or {}
    return infer_state_from_metadata(
        now=datetime.now(timezone.utc),
        place=str(metadata.get("place", metadata.get("location", "unknown"))),
        tool=str(metadata.get("tool", metadata.get("device_context", "unknown"))),
        mode=str(metadata.get("mode", metadata.get("processing_mode", "unknown"))),
        people=[str(v) for v in metadata.get("people", [])] if isinstance(metadata.get("people", []), list) else [],
        emotion=str(metadata.get("emotion", metadata.get("physio_proxy", "neutral"))),
    )


def infer_state_with_confidence(text: str, metadata: dict | None = None) -> tuple[MemoryState, dict[str, float]]:
    """
    Infer state and return robustness confidence per dimension.

    Confidence is estimated from direct metadata presence and weak text heuristics.
    """
    metadata = metadata or {}
    lowered = text.lower()

    location = str(metadata.get("place", metadata.get("location", "unknown")))
    tool = str(metadata.get("tool", metadata.get("device_context", "unknown")))
    mode = str(metadata.get("mode", metadata.get("processing_mode", "unknown")))
    people = [str(v) for v in metadata.get("people", [])] if isinstance(metadata.get("people", []), list) else []
    emotion = str(metadata.get("emotion", metadata.get("physio_proxy", "neutral")))

    if (not location or location == "unknown") and re.search(r"\b(home|office|onsite|remote|cafe)\b", lowered):
        location = re.search(r"\b(home|office|onsite|remote|cafe)\b", lowered).group(1)  # type: ignore[union-attr]
    if (not tool or tool == "unknown") and re.search(r"\b(laptop|phone|tablet|terminal|browser)\b", lowered):
        tool = re.search(r"\b(laptop|phone|tablet|terminal|browser)\b", lowered).group(1)  # type: ignore[union-attr]
    if (not mode or mode == "unknown") and re.search(r"\b(planning|execution|reflection|debugging|review)\b", lowered):
        mode = re.search(r"\b(planning|execution|reflection|debugging|review)\b", lowered).group(1)  # type: ignore[union-attr]
    if (not emotion or emotion == "neutral") and re.search(r"\b(worried|frustrated|focused|happy|excited|angry)\b", lowered):
        emotion = re.search(r"\b(worried|frustrated|focused|happy|excited|angry)\b", lowered).group(1)  # type: ignore[union-attr]
    if not people:
        if re.search(r"\b(with|met|called)\s+[a-z]+\b", lowered):
            people = ["inferred-social"]

    state = infer_state_from_metadata(
        now=datetime.now(timezone.utc),
        place=location,
        tool=tool,
        mode=mode,
        people=people,
        emotion=emotion,
    )
    confidence = {
        "location": 1.0 if metadata.get("place") or metadata.get("location") else (0.6 if location != "unknown" else 0.2),
        "device_context": 1.0 if metadata.get("tool") or metadata.get("device_context") else (0.6 if tool != "unknown" else 0.2),
        "processing_mode": 1.0 if metadata.get("mode") or metadata.get("processing_mode") else (0.6 if mode != "unknown" else 0.2),
        "social_context": 1.0 if metadata.get("people") else (0.55 if people else 0.3),
        "physio_proxy": 1.0 if metadata.get("emotion") or metadata.get("physio_proxy") else (0.55 if emotion != "neutral" else 0.35),
    }
    confidence["overall"] = round(sum(confidence.values()) / len(confidence), 6)
    return state, confidence
