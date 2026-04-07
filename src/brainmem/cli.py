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

    recall = subparsers.add_parser("recall", help="Recall memories via cue/state scoring")
    recall.add_argument("--cues", default="", help="Comma-separated cue tokens.")
    recall.add_argument("--people", default="", help="Comma-separated people list.")
    recall.add_argument("--place", default="unknown")
    recall.add_argument("--tool", default="unknown")
    recall.add_argument("--mode", default="unknown")
    recall.add_argument("--project", default="general")
    recall.add_argument("--emotion", default="neutral")
    recall.add_argument("--top-k", type=int, default=5)

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
    }


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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    engine = _build_engine(Path(args.root), args.encoding_threshold)

    if args.command == "ingest":
        payload = _handle_ingest(args, engine)
    elif args.command == "recall":
        payload = _handle_recall(args, engine)
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
