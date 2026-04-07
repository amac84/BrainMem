from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_ENCODING_THRESHOLD
from .engine import BrainMemEngine
from .types import IngestInput, RecallRequest


def _parse_people(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brainmem", description="BrainMem memory engine CLI")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root where /memory and /indexes live (default: current directory).",
    )
    parser.add_argument(
        "--encoding-threshold",
        type=float,
        default=DEFAULT_ENCODING_THRESHOLD,
        help=f"Encoding promotion threshold (default: {DEFAULT_ENCODING_THRESHOLD}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest one event into stream and optional durable memory")
    ingest.add_argument("--text", required=True, help="Event text/body.")
    ingest.add_argument("--people", default="", help="Comma-separated people list.")
    ingest.add_argument("--place", default="unknown")
    ingest.add_argument("--tool", default="unknown")
    ingest.add_argument("--mode", default="unknown")
    ingest.add_argument("--project", default="general")
    ingest.add_argument("--emotion", default="neutral")
    ingest.add_argument("--action", default="note")
    ingest.add_argument("--novelty", type=float, default=0.3)
    ingest.add_argument("--salience", type=float, default=0.3)
    ingest.add_argument("--goal-relevance", type=float, default=0.3)
    ingest.add_argument("--surprise", type=float, default=0.2)
    ingest.add_argument("--unresolved-tension", type=float, default=0.0)
    ingest.add_argument("--repetition", type=float, default=0.0)
    ingest.add_argument("--emphasis", type=float, default=0.0)
    ingest.add_argument("--social-importance", type=float, default=0.1)
    ingest.add_argument("--decision-irreversibility", type=float, default=0.0)
    ingest.add_argument("--source", default="conversation")
    ingest.add_argument("--open-loop-intent", default="", help="Optional open loop intent to register")
    ingest.add_argument("--open-loop-next-action", default="", help="Optional open loop next action")
    ingest.add_argument("--open-loop-trigger", default="", help="Optional open loop trigger cue")
    ingest.add_argument("--open-loop-tension", type=float, default=0.0, help="Optional open loop tension score")

    recall = subparsers.add_parser("recall", help="Recall memories via cue/state scoring")
    recall.add_argument("--cues", default="", help="Comma-separated cue tokens.")
    recall.add_argument("--people", default="", help="Comma-separated people list.")
    recall.add_argument("--place", default="unknown")
    recall.add_argument("--tool", default="unknown")
    recall.add_argument("--mode", default="unknown")
    recall.add_argument("--project", default="general")
    recall.add_argument("--emotion", default="neutral")
    recall.add_argument("--top-k", type=int, default=5)

    consolidate = subparsers.add_parser(
        "consolidate",
        help="Run event segmentation and daily summary generation",
    )
    consolidate.add_argument(
        "--date",
        default="",
        help="Date in YYYY-MM-DD format (defaults to today UTC).",
    )
    consolidate.add_argument(
        "--boundary-threshold",
        type=float,
        default=0.45,
        help="Boundary score threshold for splitting stream events.",
    )
    consolidate.add_argument(
        "--apply-reconsolidation",
        action="store_true",
        help="After consolidation, commit queued reconsolidation patches.",
    )

    simulate = subparsers.add_parser(
        "simulate",
        help="Run planning simulation over experience transitions",
    )
    simulate.add_argument(
        "--project",
        default="general",
        help="Project context for current state",
    )
    simulate.add_argument(
        "--mode",
        default="planning",
        help="Current processing mode",
    )
    simulate.add_argument(
        "--emotion",
        default="neutral",
        help="Current emotional state proxy",
    )
    simulate.add_argument(
        "--goal-hint",
        default="",
        help="Optional goal text to bias proposed transitions",
    )
    simulate.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Rollout depth (default: 2)",
    )
    simulate.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum proposed plans (default: 3)",
    )

    return parser


def _build_engine(root: Path, encoding_threshold: float) -> BrainMemEngine:
    return BrainMemEngine(root=root.resolve(), encoding_threshold=encoding_threshold)


def _handle_ingest(args: argparse.Namespace, engine: BrainMemEngine) -> dict[str, Any]:
    request = IngestInput(
        text=args.text,
        people=_parse_people(args.people),
        place=args.place,
        tool=args.tool,
        mode=args.mode,
        project=args.project,
        emotion=args.emotion,
        action=args.action,
        novelty=args.novelty,
        salience=args.salience,
        goal_relevance=args.goal_relevance,
        surprise=args.surprise,
        unresolved_tension=args.unresolved_tension,
        repetition=args.repetition,
        emphasis=args.emphasis,
        social_importance=args.social_importance,
        decision_irreversibility=args.decision_irreversibility,
        source=args.source,
    )
    candidate = engine.ingest(request)
    open_loop = _maybe_register_open_loop(args, engine, candidate.memory_id)
    return {
        "status": "ok",
        "operation": "ingest",
        "memory_id": candidate.memory_id,
        "created_at": candidate.created_at,
        "durable": candidate.durable,
        "path": str(candidate.path),
        "encoding_score": candidate.encoding_score,
        "score_breakdown": candidate.scores,
        "anchor": candidate.anchor,
        "open_loop": open_loop,
    }


def _maybe_register_open_loop(args: argparse.Namespace, engine: BrainMemEngine, candidate_memory_id: str) -> dict[str, Any] | None:
    if not args.open_loop_intent:
        return None
    result = engine.register_open_loop(
        intent=args.open_loop_intent,
        next_action=args.open_loop_next_action or "Clarify next action",
        trigger=args.open_loop_trigger or "token:followup",
        tension=args.open_loop_tension,
        created_from_memory_id=candidate_memory_id,
    )
    return result


def _handle_recall(args: argparse.Namespace, engine: BrainMemEngine) -> dict[str, Any]:
    raw_cues = [c.strip() for c in args.cues.split(",") if c.strip()]
    cues = [cue if ":" in cue else f"token:{cue.lower()}" for cue in raw_cues]
    people = [f"person:{p.lower()}" for p in _parse_people(args.people)]
    if args.project:
        cues.append(f"project:{args.project.lower()}")
    if args.emotion:
        cues.append(f"emotion:{args.emotion.lower()}")
    cues.extend(people)
    cues = sorted(set(cues))
    request = RecallRequest(
        cues=cues,
        people=_parse_people(args.people),
        place=args.place,
        tool=args.tool,
        mode=args.mode,
        project=args.project,
        emotion=args.emotion,
        top_k=args.top_k,
    )
    matches = engine.recall(request)
    return {
        "status": "ok",
        "operation": "recall",
        "query": {
            "cues": request.cues,
            "people": request.people,
            "place": request.place,
            "tool": request.tool,
            "mode": request.mode,
            "project": request.project,
            "emotion": request.emotion,
        },
        "results": [
            {
                "memory_id": m.memory_id,
                "path": str(m.path),
                "score": m.total_score,
                "score_breakdown": m.score_breakdown,
                "anchor": m.anchor,
                "excerpt": m.excerpt,
                "cues": m.cues,
                "state": m.state,
            }
            for m in matches
        ],
    }


def _handle_consolidate(args: argparse.Namespace, engine: BrainMemEngine) -> dict[str, Any]:
    from datetime import datetime, timezone

    from .consolidation_job import run_daily_consolidation

    if args.date:
        target_date = datetime.fromisoformat(args.date).date()
    else:
        target_date = datetime.now(timezone.utc).date()
    result = run_daily_consolidation(
        config=engine.config,
        target_date=target_date,
        boundary_threshold=args.boundary_threshold,
    )
    recon = {"decisions": [], "committed": 0, "conflicts": 0, "deferred": 0, "rejected": 0}
    if args.apply_reconsolidation:
        recon = engine.run_reconsolidation_commit()
    return {
        "status": "ok",
        "operation": "consolidate",
        "date": target_date.isoformat(),
        "reconsolidation": recon,
        **result,
    }


def _handle_simulate(args: argparse.Namespace, engine: BrainMemEngine) -> dict[str, Any]:
    current_state = f"project:{args.project.lower()}|mode:{args.mode.lower()}|emotion:{args.emotion.lower()}"
    plans = engine.simulate(
        current_state=current_state,
        goal_hint=args.goal_hint,
        depth=args.depth,
        top_k=args.top_k,
    )
    return {
        "status": "ok",
        "operation": "simulate",
        "current_state": current_state,
        "goal_hint": args.goal_hint,
        **plans,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    engine = _build_engine(Path(args.root), args.encoding_threshold)

    if args.command == "ingest":
        payload = _handle_ingest(args, engine)
    elif args.command == "recall":
        payload = _handle_recall(args, engine)
    elif args.command == "consolidate":
        payload = _handle_consolidate(args, engine)
    elif args.command == "simulate":
        payload = _handle_simulate(args, engine)
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
