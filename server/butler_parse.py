"""Parse the butler's final text into {spoken, display, citations}. Pure."""
from __future__ import annotations

import json
import re

_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
# Three alternatives: ASCII terminators need trailing whitespace; CJK
# terminators (。！？) split with OR WITHOUT it -- Chinese sentences run the
# terminator straight into the next character, so requiring whitespace made the
# 3-sentence spoken cap a no-op for Chinese. The zero-width CJK split can emit
# empty segments; cap_sentences filters them before counting.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*|\n+")


def _normalize_citation(c: str) -> str:
    c = str(c).strip()
    if c.startswith("[[") and c.endswith("]]"):
        c = c[2:-2]
    c = c.split("|")[0]          # alias
    c = c.split("#")[0]          # heading anchor
    c = c.split("^")[0]          # block ref
    return c.strip()


def _dedupe(names: list[str]) -> list[str]:
    seen: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.append(name)
    return seen


def extract_wikilinks(text: str) -> list[str]:
    return _dedupe(
        [_normalize_citation(m.group(1)) for m in _WIKILINK.finditer(str(text or ""))]
    )


def cap_sentences(text: str, n: int = 3) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    # drop empties BEFORE slicing: the zero-width CJK split yields "" segments,
    # and counting those against `n` would under-cap real sentences
    parts = [p for p in _SENT_SPLIT.split(text) if p]
    return " ".join(parts[:n]).strip()


def _load_object(raw: str) -> dict | None:
    """Try the whole string as JSON, then the first {...} that actually decodes.

    The decoder walk prefers a dict carrying "spoken"/"display" so that an
    incidental object in the prose (a bare `{}`, a fenced example) cannot
    shadow the real reply that follows it.
    """
    raw = str(raw).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):                     # e.g. [{...}]
            for item in obj:
                if isinstance(item, dict):
                    return item
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    fallback: dict | None = None
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "spoken" in obj or "display" in obj:
            return obj
        if fallback is None:
            fallback = obj
    return fallback


def parse_butler_output(text: str) -> dict:
    raw = str(text or "").strip()
    obj = _load_object(raw)
    if isinstance(obj, dict) and ("spoken" in obj or "display" in obj):
        display = str(obj.get("display") or obj.get("spoken") or "").strip()
        if display:                                   # else: fall through to plain text
            spoken = cap_sentences(str(obj.get("spoken") or display)).strip()
            cites = obj.get("citations")
            if isinstance(cites, list):
                cites = _dedupe([_normalize_citation(c) for c in cites])
            else:
                cites = extract_wikilinks(display)
            return {"spoken": spoken, "display": display, "citations": cites}
    return {"spoken": cap_sentences(raw), "display": raw, "citations": extract_wikilinks(raw)}
