from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_ENCODING_THRESHOLD: float = 0.55

DEFAULT_ENCODING_WEIGHTS: dict[str, float] = {
    "novelty": 0.18,
    "emotional_salience": 0.14,
    "goal_relevance": 0.18,
    "prediction_error": 0.12,
    "unresolved_tension": 0.14,
    "repetition": 0.08,
    "explicit_emphasis": 0.08,
    "social_importance": 0.04,
    "decision_irreversibility": 0.04,
}

DEFAULT_RECALL_WEIGHTS: dict[str, float] = {
    "cue_overlap": 0.45,
    "state_match": 0.20,
    "goal_relevance": 0.20,
    "open_loop_activation": 0.10,
    "recency_bonus": 0.05,
}


@dataclass(slots=True)
class BrainMemConfig:
    root_dir: Path
    memory_dir: Path
    indexes_dir: Path
    stream_dir: Path
    events_dir: Path
    summaries_daily_dir: Path
    summaries_weekly_dir: Path
    schemas_dir: Path
    open_loops_dir: Path
    scripts_dir: Path
    identity_dir: Path
    goals_dir: Path
    reconsolidation_queue_dir: Path
    archive_dir: Path
    encoding_threshold: float = DEFAULT_ENCODING_THRESHOLD
    encoding_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_ENCODING_WEIGHTS.copy())
    recall_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_RECALL_WEIGHTS.copy())

    @classmethod
    def from_root(cls, root: Path | str) -> "BrainMemConfig":
        root_path = Path(root).resolve()
        memory = root_path / "memory"
        return cls(
            root_dir=root_path,
            memory_dir=memory,
            indexes_dir=root_path / "indexes",
            stream_dir=memory / "stream",
            events_dir=memory / "events",
            summaries_daily_dir=memory / "summaries" / "daily",
            summaries_weekly_dir=memory / "summaries" / "weekly",
            schemas_dir=memory / "schemas",
            open_loops_dir=memory / "open_loops",
            scripts_dir=memory / "scripts",
            identity_dir=memory / "identity",
            goals_dir=memory / "goals",
            reconsolidation_queue_dir=memory / "reconsolidation" / "queue",
            archive_dir=memory / "archive",
        )

    def ensure_directories(self) -> None:
        for path in (
            self.memory_dir,
            self.indexes_dir,
            self.stream_dir,
            self.events_dir,
            self.summaries_daily_dir,
            self.summaries_weekly_dir,
            self.schemas_dir,
            self.open_loops_dir,
            self.scripts_dir,
            self.identity_dir,
            self.goals_dir,
            self.reconsolidation_queue_dir,
            self.archive_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
