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

    @tool("vault_search", "Search the vault's notes for a query; returns matching notes "
          "with a snippet. The query is matched as a LITERAL fixed string, not a regex "
          "and not natural language: pass ONE or TWO keywords (a proper noun, a project "
          "name), never a whole question. Issue several narrow searches instead of one "
          "long one. Use this to ground answers before replying.",
          {"query": str, "limit": int})
    async def _search(args):
        # `or 5` (not a dict default) so an explicit limit:null -- which a model
        # emits routinely -- falls back instead of raising TypeError; clamped to
        # 1..20 so a huge limit cannot drag the whole vault into the context.
        limit = max(1, min(int(args.get("limit") or 5), 20))
        results = await vault_search(args["query"], vault_root, limit)
        if not results:
            return _text("No matching notes. Try a single different keyword before "
                         "concluding the vault has nothing on this.")
        return _text("\n\n".join(
            f"[[{r['title']}]]  ({r['path']})\n{r['snippet']}" for r in results))

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
