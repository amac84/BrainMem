from __future__ import annotations

from collections import Counter

from .types import IngestInput

EMOTION_WORDS = {
    "stressed",
    "anxious",
    "excited",
    "happy",
    "frustrated",
    "worried",
    "angry",
    "tired",
    "overwhelmed",
}
ACTION_WORDS = {
    "send",
    "review",
    "call",
    "buy",
    "plan",
    "write",
    "book",
    "fix",
    "schedule",
    "debug",
    "decide",
}
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "is",
    "it",
    "that",
    "this",
    "at",
    "by",
    "from",
    "be",
    "as",
}


def _tokenize(text: str) -> list[str]:
    cleaned = (
        text.lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("!", " ")
        .replace("?", " ")
        .replace("(", " ")
        .replace(")", " ")
    )
    return [tok for tok in cleaned.split() if tok]


def _text_tokens(text: str) -> list[str]:
    tokens = _tokenize(text)
    emotion_tokens = [tok for tok in tokens if tok in EMOTION_WORDS]
    action_tokens = [tok for tok in tokens if tok in ACTION_WORDS]
    sparse_tokens = [tok for tok, count in Counter(tokens).items() if count == 1 and tok not in STOPWORDS]
    return sorted(set(emotion_tokens + action_tokens + sparse_tokens[:10]))


def cues_from_text(
    text: str,
    *,
    people: list[str],
    place: str,
    tool: str,
    mode: str,
    project: str,
    emotion: str,
    action: str,
) -> list[str]:
    cues: list[str] = []
    cues.extend(f"person:{person.lower()}" for person in people if person)
    if place:
        cues.append(f"place:{place.lower()}")
    if tool:
        cues.append(f"tool:{tool.lower()}")
    if mode:
        cues.append(f"mode:{mode.lower()}")
    if project:
        cues.append(f"project:{project.lower()}")
    if emotion:
        cues.append(f"emotion:{emotion.lower()}")
    if action:
        cues.append(f"action:{action.lower()}")
    cues.extend(f"token:{token}" for token in _text_tokens(text)[:12])
    return sorted(set(cues))


def cues_for_ingest(ingest: IngestInput) -> list[str]:
    return cues_from_text(
        ingest.text,
        people=ingest.people,
        place=ingest.place,
        tool=ingest.tool,
        mode=ingest.mode,
        project=ingest.project,
        emotion=ingest.emotion,
        action=ingest.action,
    )
