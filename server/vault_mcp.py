"""The butler's ONLY tools: read-only vault search/read + append-only capture/log."""
from __future__ import annotations

import asyncio
from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

from server.vault_read import _safe_note, ensure_materialized, vault_read, vault_search
from server.vault_write import vault_capture, vault_log

VAULT_TOOL_NAMES = [
    "mcp__vault__vault_search", "mcp__vault__vault_read",
    "mcp__vault__vault_capture", "mcp__vault__vault_log",
]


def _text(s: str) -> dict:
    return {"content": [{"type": "text", "text": s}]}


def build_vault_server(vault_root: Path):
    vault_root = Path(vault_root)

    @tool("vault_search", "Search the vault's notes for a query; returns the notes that "
          "match most often, with a snippet, plus the TOTAL number of matching notes. "
          "The query is matched as a LITERAL fixed string, not a regex "
          "and not natural language: pass ONE or TWO keywords (a proper noun, a project "
          "name), never a whole question. Issue several narrow searches instead of one "
          "long one. If the total is far larger than what is shown, you are seeing a "
          "subset: narrow the query rather than assuming the rest do not exist. "
          "Use this to ground answers before replying.",
          {"query": str, "limit": int})
    async def _search(args):
        # `or 5` (not a dict default) so an explicit limit:null -- which a model
        # emits routinely -- falls back instead of raising TypeError; clamped to
        # 1..20 so a huge limit cannot drag the whole vault into the context.
        limit = max(1, min(int(args.get("limit") or 5), 20))
        found = await vault_search(args["query"], vault_root, limit)
        if found.get("error"):
            # A failed search must never read as an empty vault: "No matching
            # notes" here would make the butler affirmatively tell Keke her
            # vault has nothing on a topic it simply failed to search.
            return _text(f"Search is unavailable right now ({found['error']}). "
                         "Tell Keke you could not search rather than concluding "
                         "the vault has nothing on this.")
        results, total = found["results"], found["total"]
        if not results:
            return _text("No matching notes. Try a single different keyword before "
                         "concluding the vault has nothing on this.")
        # The total goes FIRST and in plain words. Ranking alone does not stop the
        # model from treating five hits as the whole truth; it has to be told that
        # 164 notes matched and it is looking at five of them.
        parts = [f"Found {total} notes; showing the top {len(results)} by match count."]
        parts.append("\n\n".join(
            f"[[{r['title']}]]  ({r['path']})\n{r['snippet']}" for r in results))
        if total > len(results):
            parts.append(f"{total - len(results)} more notes matched and are NOT shown. "
                         "Narrow with a more specific keyword, or an adjacent two-word "
                         "phrase, instead of accepting only the notes above.")
        return _text("\n\n".join(parts))

    @tool("vault_read", "Read one vault note by its path relative to the vault root "
          "(e.g. 'Wiki/Tibet.md'). Markdown notes only.", {"path": str})
    async def _read(args):
        rel = args["path"]
        try:
            # _safe_note FIRST: it is the containment + .md check, so a traversal
            # attempt is rejected before brctl is ever pointed at the path.
            note = _safe_note(rel, vault_root)
            # The vault lives on iCloud and a cold note can be evicted to the cloud.
            # vault_read does not materialize on its own, so the guard is inert
            # unless a caller invokes it -- do that here, before the read.
            await ensure_materialized(note)
            # vault_read is SYNCHRONOUS blocking file I/O. This process also carries
            # live audio, so an evicted-file stall must not sit on the event loop.
            return _text(await asyncio.to_thread(vault_read, rel, vault_root))
        except OSError as e:
            # PermissionError/FileNotFoundError are OSError subclasses; the wider
            # catch also covers the raw I/O errors an un-materialized iCloud file
            # can raise, which would otherwise escape as an unhandled tool crash.
            return _text(f"Cannot read: {e}")

    @tool("vault_capture", "Append a short thought to today's daily inbox. Append-only.",
          {"text": str})
    async def _capture(args):
        path = await vault_capture(args["text"], vault_root)
        return _text(f"Captured to {path}")

    @tool("vault_log", "Append an action-log entry (op + title). Append-only.",
          {"op": str, "title": str})
    async def _log(args):
        await vault_log(args["op"], args["title"], vault_root)
        return _text("Logged.")

    return create_sdk_mcp_server("vault", "1.0.0", [_search, _read, _capture, _log])
