from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .markdown_io import read_markdown, write_markdown


VERBS = {"plan", "review", "call", "write", "ship", "debug", "book", "finalize", "draft"}


@dataclass(slots=True)
class ScriptCandidate:
    script_id: str
    actions: list[str]
    tools: list[str]
    projects: list[str]
    contexts: list[str]
    evidence_event: str
    source_anchor: str
    summary: str


def extract_script_signature(*, text: str, action: str, tool: str, project: str) -> dict[str, Any]:
    tokens = text.lower().replace(",", " ").replace(".", " ").split()
    actions = [action.lower()] + [tok for tok in tokens if tok in VERBS]
    deduped_actions: list[str] = []
    for item in actions:
        if item and item not in deduped_actions:
            deduped_actions.append(item)
    return {
        "actions": deduped_actions[:8],
        "tools": [tool.lower()] if tool else [],
        "projects": [project.lower()] if project else [],
    }


def extract_script_candidate(*, text: str, metadata: dict[str, Any], source_anchor: str) -> ScriptCandidate | None:
    action = str(metadata.get("action", "")).strip().lower()
    if not action:
        return None
    signature = extract_script_signature(
        text=text,
        action=action,
        tool=str(metadata.get("tool", "")),
        project=str(metadata.get("project", "")),
    )
    if not signature["actions"]:
        return None
    event_id = str(metadata.get("event_id", "unknown"))
    return ScriptCandidate(
        script_id=f"script-{action}-{str(metadata.get('project', 'general')).lower()}",
        actions=signature["actions"],
        tools=signature["tools"],
        projects=signature["projects"],
        contexts=[str(metadata.get("mode", "unknown")).lower()],
        evidence_event=event_id,
        source_anchor=source_anchor,
        summary=text.strip()[:220],
    )


def load_action_scripts(scripts_dir: Path) -> dict[str, dict[str, Any]]:
    scripts: dict[str, dict[str, Any]] = {}
    if not scripts_dir.exists():
        return scripts
    for path in sorted(scripts_dir.glob("*.md")):
        meta, body = read_markdown(path)
        if meta.get("type") != "action_script":
            continue
        scripts[str(meta.get("script_id", path.stem))] = {
            "path": path,
            "metadata": meta,
            "body": body,
        }
    return scripts


def upsert_action_script(
    scripts_dir: Path,
    existing_scripts: dict[str, dict[str, Any]],
    candidate: ScriptCandidate,
) -> Path:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = scripts_dir / f"{candidate.script_id}.md"

    if candidate.script_id in existing_scripts:
        meta = dict(existing_scripts[candidate.script_id]["metadata"])
        body = existing_scripts[candidate.script_id]["body"]
    else:
        meta = {
            "type": "action_script",
            "script_id": candidate.script_id,
            "title": candidate.script_id.replace("-", " ").title(),
            "actions": [],
            "tools": [],
            "projects": [],
            "contexts": [],
            "evidence_events": [],
            "success_rate": 0.7,
            "failure_modes": [],
        }
        body = ""

    meta["actions"] = sorted(set([str(x).lower() for x in meta.get("actions", [])] + candidate.actions))
    meta["tools"] = sorted(set([str(x).lower() for x in meta.get("tools", [])] + candidate.tools))
    meta["projects"] = sorted(set([str(x).lower() for x in meta.get("projects", [])] + candidate.projects))
    meta["contexts"] = sorted(set([str(x).lower() for x in meta.get("contexts", [])] + candidate.contexts))
    evidence = set(str(x) for x in meta.get("evidence_events", []))
    evidence.add(candidate.source_anchor)
    meta["evidence_events"] = sorted(evidence)
    write_markdown(path, meta, (body + "\n" + candidate.summary).strip())
    return path


def rank_scripts_for_query(query_signature: dict[str, Any], scripts_dir: Path, top_k: int = 3) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    scripts = load_action_scripts(scripts_dir)
    for script_id, payload in scripts.items():
        meta = payload["metadata"]
        actions = [str(x).lower() for x in meta.get("actions", [])]
        tools = [str(x).lower() for x in meta.get("tools", [])]
        projects = [str(x).lower() for x in meta.get("projects", [])]
        action_overlap = _overlap_ratio(query_signature.get("actions", []), actions)
        tool_overlap = _overlap_ratio(query_signature.get("tools", []), tools)
        project_overlap = _overlap_ratio(query_signature.get("projects", []), projects)
        score = (0.6 * action_overlap) + (0.25 * tool_overlap) + (0.15 * project_overlap)
        results.append(
            {
                "script_id": script_id,
                "path": str(payload["path"]),
                "score": round(score, 6),
                "actions": actions,
                "tools": tools,
                "projects": projects,
                "evidence_events": [str(x) for x in meta.get("evidence_events", [])],
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def match_scripts(
    *,
    scripts_dir: Path,
    action_cues: list[str],
    tool: str,
    mode: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    query_signature = {
        "actions": [str(c).lower() for c in action_cues if c],
        "tools": [tool.lower()] if tool and tool != "unknown" else [],
        "projects": [],
        "contexts": [mode.lower()] if mode and mode != "unknown" else [],
    }
    base = rank_scripts_for_query(query_signature, scripts_dir, top_k=top_k * 2)
    if not query_signature["contexts"]:
        return base[:top_k]

    ranked: list[dict[str, Any]] = []
    for item in base:
        meta, _ = read_markdown(Path(item["path"]))
        contexts = [str(x).lower() for x in meta.get("contexts", [])]
        context_overlap = _overlap_ratio(query_signature["contexts"], contexts)
        item = dict(item)
        item["contexts"] = contexts
        item["context_overlap"] = round(context_overlap, 6)
        item["score"] = round(item["score"] + (0.1 * context_overlap), 6)
        ranked.append(item)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def _overlap_ratio(left: list[str], right: list[str]) -> float:
    l = Counter([item.lower() for item in left if item])
    r = Counter([item.lower() for item in right if item])
    if not l and not r:
        return 0.0
    intersection = sum((l & r).values())
    union = sum((l | r).values())
    return intersection / union if union else 0.0


def upsert_script_from_event(
    *,
    scripts_dir: Path,
    event_id: str,
    event_text: str,
    action: str,
    tool: str,
    mode: str,
    evidence_anchor: str,
) -> Path:
    signature = extract_script_signature(
        text=event_text,
        action=action,
        tool=tool,
        project="general",
    )
    candidate = ScriptCandidate(
        script_id=f"script-{action.lower()}-{mode.lower()}",
        actions=signature.get("actions", []),
        tools=signature.get("tools", []),
        projects=["general"],
        contexts=[mode.lower()] if mode else ["unknown"],
        evidence_event=event_id,
        source_anchor=evidence_anchor,
        summary=event_text.strip()[:220],
    )
    existing = load_action_scripts(scripts_dir)
    return upsert_action_script(scripts_dir, existing, candidate)


def match_scripts(
    *,
    scripts_dir: Path,
    action_cues: list[str],
    tool: str,
    mode: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    query_signature = {
        "actions": [cue.lower() for cue in action_cues],
        "tools": [tool.lower()] if tool else [],
        "projects": [],
    }
    results = rank_scripts_for_query(query_signature, scripts_dir, top_k=top_k)
    for result in results:
        result.setdefault("contexts", [mode.lower()] if mode else [])
    return results
