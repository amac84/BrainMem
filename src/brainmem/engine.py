from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import BrainMemConfig, DEFAULT_ENCODING_THRESHOLD
from .associative_graph import activate_from_cues, apply_pattern_separation, build_hig_adjacency
from .competition_inhibition import (
    InhibitionPolicy,
    apply_competitor_inhibition,
    build_competition_sets,
    write_competition_sets,
)
from .cue_extraction import cues_for_ingest
from .encoding_gate import score_event_for_encoding
from .markdown_io import read_markdown, write_markdown
from .open_loops import (
    create_open_loop,
    match_open_loops,
)
from .recall_scoring import score_candidate
from .reconsolidation import (
    commit_reconsolidation_queue,
    labilize_memory,
    queue_patch,
)
from .action_scripts import (
    extract_script_signature,
    rank_scripts_for_query,
    upsert_script_from_event,
)
from .simulation_planner import (
    build_experience_graph,
    load_experience_graph,
    save_experience_graph,
    simulate_plan,
)
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
            self._maybe_create_open_loop(target_path, metadata_payload, record.text)
            self._update_action_scripts(target_path, metadata_payload, record.text)

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
        open_loop_hits = match_open_loops(
            open_loops_dir=self.config.open_loops_dir,
            query_cues=request.cues,
            project=request.project,
            top_k=3,
        )
        query_signature = extract_script_signature(
            text=request.query_text if hasattr(request, "query_text") else " ".join(request.cues),
            action=(request.cues[0].replace("action:", "") if request.cues and request.cues[0].startswith("action:") else "review"),
            tool=request.tool,
            project=request.project,
        )
        action_script_hits = rank_scripts_for_query(
            query_signature=query_signature,
            scripts_dir=self.config.scripts_dir,
            top_k=2,
        )
        matches: list[RecallMatch] = []
        adjacency = self._load_hig_adjacency()
        graph_results = apply_pattern_separation(
            activate_from_cues(query_cues=request.cues, adjacency=adjacency, top_k=max(request.top_k * 3, 10)),
            adjacency,
        )
        graph_map = {result.memory_id: result for result in graph_results}

        for path in sorted(self.config.events_dir.glob("**/*.md")):
            metadata, body = read_markdown(path)
            if metadata.get("type") != "event":
                continue
            memory_id = str(metadata.get("event_id", path.stem))
            candidate = score_candidate(
                request,
                memory_id=memory_id,
                path=path,
                cues=[str(c) for c in metadata.get("cues", [])],
                state={k: str(v) for k, v in dict(metadata.get("state", {})).items()},
                anchor=str(metadata.get("source_anchor", "")),
                excerpt=body[:280].strip(),
                created_at=str(metadata.get("created_at", "")),
                weights=self.config.recall_weights,
            )
            graph_activation = graph_map.get(memory_id)
            if graph_activation:
                candidate.total_score = round(candidate.total_score + (0.2 * graph_activation.activation_score), 6)
                candidate.score_breakdown["graph_activation"] = round(graph_activation.activation_score, 6)
                candidate.score_breakdown["graph_supporting_cues"] = len(graph_activation.matched_cues)
                candidate.score_breakdown["pattern_overlap_with_top"] = round(
                    graph_activation.overlap_with_top,
                    6,
                )
                if graph_activation.disambiguation_required:
                    candidate.score_breakdown["pattern_separation_required"] = 1.0
                    candidate.score_breakdown["pattern_separation_hint"] = graph_activation.pattern_separation_hint
            else:
                candidate.score_breakdown["graph_activation"] = 0.0
            matches.append(candidate)
        matches.sort(key=lambda item: item.total_score, reverse=True)
        selected = matches[: request.top_k]
        self._update_competition_inhibition(request, selected)
        self._mark_recalled_memories_labilized(selected, request)
        self._apply_open_loop_boost(selected, open_loop_hits)
        self._apply_action_script_boost(selected, action_script_hits)
        return selected

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
        self._update_associative_indexes()

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

    def _update_associative_indexes(self) -> None:
        adjacency = build_hig_adjacency(self.config.events_dir)
        self._write_json(self.config.indexes_dir / "hig_adjacency.json", adjacency)

        cue_to_events = {
            cue: [
                str(memory_id)
                for memory_id in event_ids
            ]
            for cue, event_ids in dict(adjacency.get("cue_to_memories", {})).items()
        }
        competition_sets = build_competition_sets(cue_to_events)
        write_competition_sets(self.config.indexes_dir / "competition_sets.json", competition_sets)

    def _load_hig_adjacency(self) -> dict[str, Any]:
        path = self.config.indexes_dir / "hig_adjacency.json"
        if path.exists():
            return self._read_json(path, {})
        adjacency = build_hig_adjacency(self.config.events_dir)
        self._write_json(path, adjacency)
        return adjacency

    def _update_competition_inhibition(self, request: RecallRequest, selected: list[RecallMatch]) -> None:
        if not selected or not request.cues:
            return
        strength_path = self.config.indexes_dir / "strength_table.json"
        strength_table = self._read_json(strength_path, {})
        competition_path = self.config.indexes_dir / "competition_sets.json"
        competition_sets = self._read_json(competition_path, {})

        winner = selected[0]
        cue = request.cues[0]
        affected = apply_competitor_inhibition(
            winner_event_id=winner.memory_id,
            cue=cue,
            competition_sets=competition_sets,
            strength_table=strength_table,
            policy=InhibitionPolicy(),
        )
        if affected:
            self._write_json(strength_path, strength_table)

    def _mark_recalled_memories_labilized(self, selected: list[RecallMatch], request: RecallRequest) -> None:
        evidence_note = (
            f"recall_cues={','.join(request.cues[:8])};"
            f"people={','.join(request.people[:4])};project={request.project}"
        )
        for match in selected:
            labilize_memory(match.path)
            claim = f"Recall trace: {evidence_note}; excerpt={match.excerpt[:100]}"
            queue_patch(
                queue_dir=self.config.reconsolidation_queue_dir,
                memory_path=match.path,
                memory_id=match.memory_id,
                claim=claim,
                evidence_anchor=match.anchor or f"auto:{match.memory_id}",
                confidence_delta=0.02,
                expected_project=request.project.lower(),
            )

    def _maybe_create_open_loop(self, event_path: Path, metadata_payload: dict[str, Any], text: str) -> None:
        cues = [str(c) for c in metadata_payload.get("cues", [])]
        if not any(
            cue in {"token:need", "token:pending", "token:todo", "token:follow-up", "token:followup"}
            for cue in cues
        ):
            return
        trigger_cues = [
            cue
            for cue in cues
            if cue.startswith(("project:", "person:", "action:", "tool:", "place:", "emotion:", "token:"))
        ][:8]
        loop_path = create_open_loop(
            open_loops_dir=self.config.open_loops_dir,
            intent=f"Resolve unresolved task from {metadata_payload.get('event_id')}",
            trigger_cues=trigger_cues,
            next_action=f"Review and close: {text[:120]}",
            closure_condition="Explicit completion or user cancellation",
            tension=0.65,
        )
        cues.append("status:open_loop")
        metadata_payload["cues"] = sorted(set(cues))
        write_markdown(event_path, metadata_payload, text)
        refreshed_meta, refreshed_body = read_markdown(event_path)
        refreshed_meta["open_loop_path"] = str(loop_path)
        write_markdown(event_path, refreshed_meta, refreshed_body)
        self._update_associative_indexes()

    def _apply_open_loop_boost(self, selected: list[RecallMatch], open_loop_hits: list[dict[str, Any]]) -> None:
        if not selected or not open_loop_hits:
            return
        max_tension = max(float(loop.get("tension", 0.0)) for loop in open_loop_hits)
        for match in selected:
            cues = set(match.cues)
            if any(cue in cues for loop in open_loop_hits for cue in loop.get("trigger_cues", [])):
                match.total_score = round(match.total_score + 0.1, 6)
                match.score_breakdown["open_loop_triggered"] = 1.0
                match.score_breakdown["open_loop_tension"] = round(max_tension, 6)
        selected.sort(key=lambda item: item.total_score, reverse=True)

    def _update_action_scripts(self, event_path: Path, metadata_payload: dict[str, Any], text: str) -> None:
        event_id = str(metadata_payload.get("event_id", event_path.stem))
        upsert_script_from_event(
            scripts_dir=self.config.scripts_dir,
            event_id=event_id,
            event_text=text,
            action=str(metadata_payload.get("action", "note")),
            tool=str(metadata_payload.get("tool", "unknown")),
            mode=str(metadata_payload.get("mode", "unknown")),
            evidence_anchor=str(metadata_payload.get("source_anchor", "")),
        )

    def _apply_action_script_boost(self, selected: list[RecallMatch], script_hits: list[dict[str, Any]]) -> None:
        if not selected or not script_hits:
            return
        script_bonus = max(float(hit.get("score", 0.0)) for hit in script_hits)
        for match in selected:
            match.total_score = round(match.total_score + (0.05 * script_bonus), 6)
            match.score_breakdown["action_script_bonus"] = round(script_bonus, 6)
        selected.sort(key=lambda item: item.total_score, reverse=True)

    def run_reconsolidation_commit(self) -> dict[str, Any]:
        result = commit_reconsolidation_queue(
            queue_dir=self.config.reconsolidation_queue_dir,
            claim_graph_path=self.config.indexes_dir / "claim_graph.json",
        )
        return result

    def run_weekly_schema_consolidation(self, week_id: str) -> dict[str, Any]:
        """Generate a lightweight weekly theme/schema rollup."""
        summary_dir = self.config.summaries_daily_dir
        daily_files = sorted(summary_dir.glob("*.md"))
        if not daily_files:
            return {
                "week_id": week_id,
                "daily_count": 0,
                "themes": [],
                "schema_path": "",
            }

        themes: dict[str, int] = {}
        anchors: list[str] = []
        for daily_path in daily_files:
            metadata, body = read_markdown(daily_path)
            if metadata.get("type") != "daily_summary":
                continue
            text = body.lower()
            if "atlas" in text:
                themes["atlas"] = themes.get("atlas", 0) + 1
            if "vendor" in text:
                themes["vendor"] = themes.get("vendor", 0) + 1
            if "budget" in text:
                themes["budget"] = themes.get("budget", 0) + 1
            for anchor in metadata.get("anchors", []):
                if isinstance(anchor, str):
                    anchors.append(anchor)

        schema_path = self.config.schemas_dir / f"{week_id}.md"
        top_themes = sorted(themes.items(), key=lambda item: item[1], reverse=True)[:8]
        metadata = {
            "type": "weekly_schema",
            "week_id": week_id,
            "source_daily_count": len(daily_files),
            "themes": {k: v for k, v in top_themes},
            "anchors": anchors[:20],
        }
        body_lines = ["## Weekly Themes"]
        body_lines.extend(f"- {name}: observed {count} day(s)" for name, count in top_themes)
        body_lines.append("")
        body_lines.append("## Anchors")
        body_lines.extend(f"- {anchor}" for anchor in anchors[:20])
        write_markdown(schema_path, metadata, "\n".join(body_lines).strip() + "\n")
        return {
            "week_id": week_id,
            "daily_count": len(daily_files),
            "themes": top_themes,
            "schema_path": str(schema_path),
        }

    def refresh_identity_goal_priors(self) -> dict[str, Any]:
        """Derive simple identity/goal priors from evidence in memory."""
        events = sorted(self.config.events_dir.glob("**/*.md"))
        role_counts: dict[str, int] = {}
        goal_counts: dict[str, int] = {}
        evidence: list[str] = []
        for path in events:
            metadata, body = read_markdown(path)
            if metadata.get("type") != "event":
                continue
            project = str(metadata.get("project", "general")).lower()
            action = str(metadata.get("action", "note")).lower()
            if project and project != "general":
                goal_counts[project] = goal_counts.get(project, 0) + 1
            if action in {"plan", "review", "call", "debug", "write"}:
                role = f"role:{action}_oriented"
                role_counts[role] = role_counts.get(role, 0) + 1
            anchor = str(metadata.get("source_anchor", ""))
            if anchor:
                evidence.append(anchor)

        identity_path = self.config.identity_dir / "identity.md"
        goals_path = self.config.goals_dir / "active_goals.md"
        identity_meta = {
            "type": "identity_profile",
            "roles": role_counts,
            "evidence_count": len(evidence),
        }
        identity_body = "## Inferred Roles\n" + "\n".join(
            f"- {role} ({count})" for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True)
        )
        if not role_counts:
            identity_body += "\n- none yet"
        identity_body += "\n\n## Evidence Anchors\n" + "\n".join(f"- {a}" for a in evidence[:20])
        write_markdown(identity_path, identity_meta, identity_body.strip() + "\n")

        goals_meta = {
            "type": "goal_prior",
            "goals": goal_counts,
            "evidence_count": len(evidence),
        }
        goals_body = "## Active Goal Priors\n" + "\n".join(
            f"- {goal}: {count}" for goal, count in sorted(goal_counts.items(), key=lambda x: x[1], reverse=True)
        )
        if not goal_counts:
            goals_body += "\n- none yet"
        goals_body += "\n\n## Evidence Anchors\n" + "\n".join(f"- {a}" for a in evidence[:20])
        write_markdown(goals_path, goals_meta, goals_body.strip() + "\n")

        return {
            "identity_path": str(identity_path),
            "goals_path": str(goals_path),
            "roles": role_counts,
            "goals": goal_counts,
            "evidence_count": len(evidence),
        }

    def simulate(
        self,
        *,
        current_state: str,
        goal_hint: str,
        depth: int = 2,
        top_k: int = 3,
    ) -> dict[str, Any]:
        graph_path = self.config.indexes_dir / "experience_graph.json"
        graph = build_experience_graph(self.config.events_dir)
        save_experience_graph(graph_path, graph)
        graph = load_experience_graph(graph_path)
        proposals = simulate_plan(
            graph=graph,
            current_state=current_state,
            goal_hint=goal_hint,
            depth=depth,
            top_k=top_k,
        )
        return {
            "current_state": current_state,
            "goal_hint": goal_hint,
            "proposal_count": len(proposals),
            "proposals": proposals,
            "graph_path": str(graph_path),
        }

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

