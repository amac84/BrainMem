"""State inference for current and encoded memory context."""

from __future__ import annotations

from datetime import datetime, timezone

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
