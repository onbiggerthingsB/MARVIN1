"""Parse the butler's final text into {spoken, display, citations}. Pure."""
from __future__ import annotations

import json
import re

_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_wikilinks(text: str) -> list[str]:
    seen: list[str] = []
    for m in _WIKILINK.finditer(text or ""):
        name = m.group(1).strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def cap_sentences(text: str, n: int = 3) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return " ".join(_SENT_SPLIT.split(text)[:n]).strip()


def _load_object(raw: str) -> dict | None:
    """Try the whole string as JSON, then the first {...} span inside it."""
    candidates = [raw.strip()]
    span = re.search(r"\{.*\}", raw, re.DOTALL)
    if span:
        candidates.append(span.group(0))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_butler_output(text: str) -> dict:
    raw = (text or "").strip()
    obj = _load_object(raw)
    if isinstance(obj, dict) and ("spoken" in obj or "display" in obj):
        display = str(obj.get("display") or obj.get("spoken") or "").strip()
        spoken = str(obj.get("spoken") or cap_sentences(display)).strip()
        cites = obj.get("citations")
        if isinstance(cites, list):
            cites = [str(c).strip() for c in cites if str(c).strip()]
        else:
            cites = extract_wikilinks(display)
        return {"spoken": spoken, "display": display, "citations": cites}
    return {"spoken": cap_sentences(raw), "display": raw, "citations": extract_wikilinks(raw)}
