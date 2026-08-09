"""Read-only briefing over Keke's own stock system.

Spec §16: JARVIS is an ADVISOR SURFACE, never an execution path. There is no
function here that can place, modify or cancel an order, and no brokerage
credential is read, stored or transmitted. Voice is the worst possible surface
for an irreversible money action (mishearing plus no undo), and an execution
path would also hand any future prompt injection a money channel.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

MAX_PER_KIND = 5
MAX_ROWS = 8
CAVEAT = ("These are your stock system's own signals, not my recommendation, "
          "and not financial advice.")
TRADE_REFUSAL = ("Trading happens in your stock system directly, sir — I only read it. "
                 "I won't place an order by voice.")

_POSITION_TABLES = ("positions", "holdings", "portfolio", "picks", "signals")
_SYMBOL_KEYS = ("symbol", "ticker", "sym")


def find_finance_project(registry):
    if registry is None:
        return None
    return next((p for p in registry.projects
                 if p.confirmed and p.kind == "finance"), None)


def detect_outputs(root: Path) -> dict:
    root = Path(root)
    out: dict[str, list[str]] = {"sqlite": [], "json": [], "csv": [], "reports": []}
    if not root.is_dir():
        return out
    buckets = {".sqlite": "sqlite", ".db": "sqlite", ".json": "json",
               ".csv": "csv", ".md": "reports"}
    found: dict[str, list[tuple[float, str]]] = {k: [] for k in out}
    # os.walk with followlinks=False: rglob on Python < 3.13 follows symlinked
    # directories, which would let a link inside the repo escape it (or cycle).
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".")      # .venv, .git, .claude
                       and not (base / d).is_symlink()]
        for name in filenames:
            p = base / name
            # Dot check is scoped to the path RELATIVE to the repo root, so a
            # repo living under a dotted ancestor (~/.finance/...) still scans.
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                continue
            if p.is_symlink():
                continue                   # never read through a link out of the repo
            kind = buckets.get(p.suffix.lower())
            if kind is None or not p.is_file():
                continue
            try:
                found[kind].append((p.stat().st_mtime, str(p)))
            except OSError:
                continue
    for kind, items in found.items():
        items.sort(reverse=True)           # newest first
        out[kind] = [path for _, path in items[:MAX_PER_KIND]]
    return out


def _rows_from_sqlite(path: str) -> list[dict]:
    # Percent-encode the path: SQLite's URI parser treats '#' as a fragment and
    # '?' as query-string start, so a raw filename could truncate the URI and
    # silently lose mode=ro (falling back to read-write-and-create). quote()
    # keeps '/' and encodes the rest; SQLite decodes %XX in URI paths.
    uri = f"file:{quote(str(path))}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return []
    try:
        con.row_factory = sqlite3.Row
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        table = next((t for t in names if t.lower() in _POSITION_TABLES), None)
        if table is None:
            return []
        rows = con.execute(f'SELECT * FROM "{table}" LIMIT {MAX_ROWS}').fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def _rows_from_json(path: str) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []                          # a bad file degrades, never crashes the brief
    if isinstance(data, dict):
        for key in ("positions", "holdings", "picks", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        return []
    return [r for r in data[:MAX_ROWS] if isinstance(r, dict)]


def _symbol(row: dict) -> str | None:
    for k in row:
        if k.lower() in _SYMBOL_KEYS and row[k]:
            return str(row[k])
    return None


def _speak(rows: list[dict]) -> str:
    syms = [s for s in (_symbol(r) for r in rows) if s]
    if not syms:
        return f"Your stock system has {len(rows)} rows, but no ticker column I recognise."
    head = ", ".join(syms[:3])
    more = f", and {len(syms) - 3} more" if len(syms) > 3 else ""
    return f"Your screener is holding {head}{more}."


def _collect(project) -> dict:
    pinned = getattr(project, "data_source", None)
    if pinned:
        # A voice-confirmed source is authoritative: read it or say so. No
        # silent fallback to some other file Keke never approved (spec §16).
        for reader in (_rows_from_sqlite, _rows_from_json):
            rows = reader(pinned)
            if rows:
                return {"source": pinned, "rows": rows}
        return {"source": None, "rows": []}
    root = Path(project.path)
    outputs = detect_outputs(root)
    for path in outputs["sqlite"]:
        rows = _rows_from_sqlite(path)
        if rows:
            return {"source": path, "rows": rows}
    for path in outputs["json"]:
        rows = _rows_from_json(path)
        if rows:
            return {"source": path, "rows": rows}
    return {"source": None, "rows": []}


async def portfolio_brief(project, now: datetime | None = None) -> dict:
    # The "never brief an unconfirmed project" guarantee holds by construction
    # here, not by trusting the caller to have used find_finance_project():
    # anything that is not a confirmed finance project is treated like None.
    if project is None or not (getattr(project, "confirmed", False)
                               and getattr(project, "kind", None) == "finance"):
        return {"available": False, "source": None, "rows": [], "as_of": None,
                "caveat": CAVEAT,
                "spoken": "I don't have a confirmed stock system yet, sir."}
    found = await asyncio.to_thread(_collect, project)
    if not found["rows"]:
        return {"available": False, "source": None, "rows": [], "as_of": None,
                "caveat": CAVEAT,
                "spoken": f"I couldn't find readable results in {project.name}, sir."}
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
    return {"available": True, "source": found["source"], "rows": found["rows"],
            "as_of": stamp, "caveat": CAVEAT, "spoken": _speak(found["rows"])}
