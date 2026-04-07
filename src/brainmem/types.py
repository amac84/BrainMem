from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class EncodingFactors:
    novelty: float = 0.0
    emotional_salience: float = 0.0
    goal_relevance: float = 0.0
    prediction_error: float = 0.0
    unresolved_tension: float = 0.0
    repetition: float = 0.0
    explicit_emphasis: float = 0.0
    social_importance: float = 0.0
    decision_irreversibility: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "novelty": self.novelty,
            "emotional_salience": self.emotional_salience,
            "goal_relevance": self.goal_relevance,
            "prediction_error": self.prediction_error,
            "unresolved_tension": self.unresolved_tension,
            "repetition": self.repetition,
            "explicit_emphasis": self.explicit_emphasis,
            "social_importance": self.social_importance,
            "decision_irreversibility": self.decision_irreversibility,
        }


@dataclass(slots=True)
class MemoryState:
    time_of_day: str = "unknown"
    location: str = "unknown"
    device_context: str = "unknown"
    social_context: str = "unknown"
    physio_proxy: str = "unknown"
    processing_mode: str = "unknown"

    def as_dict(self) -> dict[str, str]:
        return {
            "time_of_day": self.time_of_day,
            "location": self.location,
            "device_context": self.device_context,
            "social_context": self.social_context,
            "physio_proxy": self.physio_proxy,
            "processing_mode": self.processing_mode,
        }


@dataclass(slots=True)
class EventRecord:
    event_id: str
    created_at: str
    text: str
    people: list[str]
    project: str
    place: str
    tool: str
    mode: str
    emotion: str
    action: str
    cues: list[str]
    state: MemoryState
    factors: EncodingFactors
    source: str
    strength: float = 0.5


@dataclass(slots=True)
class EncodingDecision:
    score: float
    threshold: float
    promoted: bool
    weighted_factors: dict[str, float]
    reason: str


@dataclass(slots=True)
class QueryContext:
    query_text: str
    cues: list[str] = field(default_factory=list)
    state: dict[str, str] = field(default_factory=dict)
    processing_mode: str = "unknown"
    active_goals: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IngestInput:
    text: str
    people: list[str] = field(default_factory=list)
    place: str = "unknown"
    tool: str = "unknown"
    mode: str = "unknown"
    project: str = "general"
    emotion: str = "neutral"
    action: str = "note"
    novelty: float = 0.3
    salience: float = 0.3
    goal_relevance: float = 0.3
    surprise: float = 0.2
    unresolved_tension: float = 0.0
    repetition: float = 0.0
    emphasis: float = 0.0
    social_importance: float = 0.1
    decision_irreversibility: float = 0.0
    source: str = "conversation"


@dataclass(slots=True)
class MemoryCandidate:
    memory_id: str
    created_at: str
    path: Path
    cues: list[str]
    state: dict[str, str]
    scores: dict[str, float]
    encoding_score: float
    durable: bool
    anchor: str
    text: str


@dataclass(slots=True)
class RecallRequest:
    cues: list[str]
    people: list[str] = field(default_factory=list)
    place: str = "unknown"
    tool: str = "unknown"
    mode: str = "unknown"
    project: str = "general"
    emotion: str = "neutral"
    top_k: int = 5


@dataclass(slots=True)
class RecallMatch:
    memory_id: str
    path: Path
    total_score: float
    score_breakdown: dict[str, float]
    anchor: str
    excerpt: str
    cues: list[str]
    state: dict[str, str]


@dataclass(slots=True)
class OpenLoopMatch:
    loop_id: str
    path: Path
    priority: float
    tension: float
    next_action: str
    trigger_reason: str
    anchor: str

