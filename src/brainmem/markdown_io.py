from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Parse simple YAML-like front matter without external dependencies."""
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, text

    metadata = _parse_yaml_like_lines(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return metadata, body


def format_front_matter(metadata: dict[str, Any], body: str) -> str:
    lines = ["---"]
    lines.extend(_dump_yaml_like_dict(metadata))
    lines.append("---")
    lines.append("")
    if body:
        lines.append(body.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def read_markdown(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return parse_front_matter(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, metadata: dict[str, Any], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_front_matter(metadata, body), encoding="utf-8")


def load_front_matter(path: Path) -> tuple[dict[str, Any], str]:
    return read_markdown(path)


def write_memory_markdown(path: Path, front_matter: dict[str, Any], body: str) -> None:
    return write_markdown(path, front_matter, body)


def _parse_yaml_like_lines(lines: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        idx += 1
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            nested: dict[str, Any] = {}
            while idx < len(lines):
                next_line = lines[idx]
                if not next_line.startswith("  "):
                    break
                idx += 1
                if ":" not in next_line:
                    continue
                nkey, nvalue = next_line.strip().split(":", 1)
                nested[nkey.strip()] = _parse_scalar(nvalue.strip())
            data[key] = nested
        else:
            data[key] = _parse_scalar(value)
    return data


def _dump_yaml_like_dict(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for nkey, nvalue in value.items():
                lines.append(f"  {nkey}: {_dump_scalar(nvalue)}")
        elif isinstance(value, list):
            lines.append(f"{key}: [{', '.join(_dump_scalar(v) for v in value)}]")
        else:
            lines.append(f"{key}: {_dump_scalar(value)}")
    return lines


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        return [_parse_scalar(part) for part in parts]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        if (value.startswith("{") and value.endswith("}")) or (
            value.startswith("[") and value.endswith("]")
        ):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        return value


def _dump_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)
