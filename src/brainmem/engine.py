from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import BrainMemConfig, DEFAULT_ENCODING_THRESHOLD
from .cue_extraction import cues_for_ingest
from .encoding_gate import score_event_for_encoding
from .markdown_io import read_markdown, write_markdown
from .recall_scoring import score_candidate
from .state_inference import infer_state
from .types import (
    EncodingFactors,
    EventRecord,
    IngestInput,
    MemoryCandidate,
    RecallMatch,
    RecallRequest,
    utc_now_iso,
)


class BrainMemEngine:
    def __init__(
        self,
        *,
        config: BrainMemConfig | None = None,
        root: Path | str | None = None,
        encoding_threshold: float = DEFAULT_ENCODING_THRESHOLD,
    ) -> None:
        if config is None:
            root_path = Path.cwd() if root is None else Path(root)
            config = BrainMemConfig.from_root(root_path)
        self.config = config
        self.config.encoding_threshold = encoding_threshold
        self.config.ensure_directories()

    def ingest(self, request: IngestInput) -> MemoryCandidate:
        event_id = f"ev-{uuid4().hex[:10]}"
        created_at = utc_now_iso()
        now = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        state = infer_state(
            request.text,
            metadata={
                "time_of_day": request.mode if request.mode in {"morning", "afternoon", "evening", "night"} else None,
                "location": request.place,
                "tool": request.tool,
                "device_context": request.tool,
                "social_context": "social" if request.people else "alone",
                "physio_proxy": request.emotion if request.emotion != "neutral" else "unknown",
                "processing_mode": request.mode
                if request.mode not in {"morning", "afternoon", "evening", "night"}
                else "general",
                "people": request.people,
            },
        )
        cues = cues_for_ingest(request)
        factors = EncodingFactors(
            novelty=request.novelty,
            emotional_salience=request.salience,
            goal_relevance=request.goal_relevance,
            prediction_error=request.surprise,
            unresolved_tension=request.unresolved_tension,
            repetition=request.repetition,
            explicit_emphasis=request.emphasis,
            social_importance=request.social_importance,
            decision_irreversibility=request.decision_irreversibility,
        )

        record = EventRecord(
            event_id=event_id,
            created_at=created_at,
            text=request.text.strip(),
            people=request.people,
            project=request.project,
            place=request.place,
            tool=request.tool,
            mode=request.mode,
            emotion=request.emotion,
            action=request.action,
            state=state,
            cues=cues,
            factors=factors,
            source=request.source,
        )

        stream_path = self._stream_file_path(date.fromisoformat(created_at[:10]))
        self._append_stream_record(stream_path, record)

        decision = score_event_for_encoding(
            record,
            threshold=self.config.encoding_threshold,
            weights=self.config.encoding_weights,
        )
        durable = decision.promoted
        target_path = stream_path
        anchor = f"{stream_path.name}#event_id={event_id}"

        if durable:
            target_path = self._event_file_path(now, event_id)
            metadata_payload = {
                "type": "event",
                "event_id": event_id,
                "created_at": created_at,
                "people": record.people,
                "project": record.project,
                "place": record.place,
                "tool": record.tool,
                "mode": record.mode,
                "emotion": record.emotion,
                "action": record.action,
                "state": record.state.as_dict(),
                "cues": record.cues,
                "source": record.source,
                "source_anchor": anchor,
                "encoding_score": decision.score,
                "encoding_threshold": decision.threshold,
                "encoding_weighted_factors": decision.weighted_factors,
                "strength": 0.55,
            }
            write_markdown(target_path, metadata_payload, record.text)
            self._update_indexes(target_path, metadata_payload)

        return MemoryCandidate(
            memory_id=event_id,
            created_at=created_at,
            path=target_path,
            cues=cues,
            state=state.as_dict(),
            scores=decision.weighted_factors,
            encoding_score=decision.score,
            durable=durable,
            anchor=anchor,
            text=record.text,
        )

    def recall(self, request: RecallRequest) -> list[RecallMatch]:
        matches: list[RecallMatch] = []
        for path in sorted(self.config.events_dir.glob("**/*.md")):
            metadata, body = read_markdown(path)
            if metadata.get("type") != "event":
                continue
            candidate = score_candidate(
                request,
                memory_id=str(metadata.get("event_id", path.stem)),
                path=path,
                cues=[str(c) for c in metadata.get("cues", [])],
                state={k: str(v) for k, v in dict(metadata.get("state", {})).items()},
                anchor=str(metadata.get("source_anchor", "")),
                excerpt=body[:280].strip(),
                weights=self.config.recall_weights,
            )
            matches.append(candidate)
        matches.sort(key=lambda item: item.total_score, reverse=True)
        return matches[: request.top_k]

    def _stream_file_path(self, day: date) -> Path:
        return self.config.stream_dir / f"{day.isoformat()}.md"

    def _event_file_path(self, now: datetime, event_id: str) -> Path:
        return (
            self.config.events_dir
            / now.strftime("%Y")
            / now.strftime("%m")
            / now.strftime("%d")
            / f"{event_id}.md"
        )

    def _append_stream_record(self, stream_path: Path, record: EventRecord) -> None:
        metadata: dict[str, Any]
        body: str
        if stream_path.exists():
            metadata, body = read_markdown(stream_path)
        else:
            metadata = {"type": "stream", "date": stream_path.stem}
            body = ""
        payload = {
            "event_id": record.event_id,
            "created_at": record.created_at,
            "people": record.people,
            "project": record.project,
            "place": record.place,
            "tool": record.tool,
            "mode": record.mode,
            "emotion": record.emotion,
            "action": record.action,
            "state": record.state.as_dict(),
            "cues": record.cues,
            "text": record.text,
            "factors": record.factors.as_dict(),
            "source": record.source,
        }
        entry = json.dumps(payload, sort_keys=True)
        new_body = f"{body.rstrip()}\n{entry}\n" if body.strip() else f"{entry}\n"
        write_markdown(stream_path, metadata, new_body)

    def _update_indexes(self, event_path: Path, metadata_payload: dict[str, Any]) -> None:
        self._update_inverted_cues(event_path, metadata_payload)
        self._update_state_buckets(event_path, metadata_payload)
        self._update_strength_table(event_path, metadata_payload)

    def _update_inverted_cues(self, event_path: Path, metadata_payload: dict[str, Any]) -> None:
        index_path = self.config.indexes_dir / "inverted_cues.json"
        data = self._read_json(index_path, {})
        for cue in metadata_payload.get("cues", []):
            entries = data.setdefault(str(cue), [])
            event_ref = str(event_path)
            if event_ref not in entries:
                entries.append(event_ref)
        self._write_json(index_path, data)

    def _update_state_buckets(self, event_path: Path, metadata_payload: dict[str, Any]) -> None:
        index_path = self.config.indexes_dir / "state_buckets.json"
        data = self._read_json(index_path, {})
        state = dict(metadata_payload.get("state", {}))
        for key, value in state.items():
            if not value:
                continue
            bucket_key = f"{key}:{value}"
            entries = data.setdefault(bucket_key, [])
            event_ref = str(event_path)
            if event_ref not in entries:
                entries.append(event_ref)
        self._write_json(index_path, data)

    def _update_strength_table(self, event_path: Path, metadata_payload: dict[str, Any]) -> None:
        index_path = self.config.indexes_dir / "strength_table.json"
        data = self._read_json(index_path, {})
        event_id = str(metadata_payload.get("event_id"))
        if event_id:
            data[event_id] = {
                "path": str(event_path),
                "strength": metadata_payload.get("strength", 0.55),
                "updated_at": utc_now_iso(),
            }
        self._write_json(index_path, data)

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return default
        return json.loads(raw)

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

