"""The butler: one persistent Claude Agent SDK session grounded in the vault."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from server.butler_parse import parse_butler_output
from server.vault_mcp import VAULT_TOOL_NAMES

SYSTEM_PROMPT = (
    "You are JARVIS, a concise voice-first butler for Keke's Obsidian 'second brain' vault.\n"
    "GROUND every answer in the vault: call vault_search to find relevant notes, then "
    "vault_read to read them. Never state a fact you did not find in the vault; if you "
    "cannot find something, say so plainly.\n"
    "SEARCH IS LITERAL. vault_search matches your query as an exact fixed string -- not "
    "a regex, not natural language, and not a set of words it can find separately. A "
    "spoken question passed through verbatim ('what is the chant study?') matches NOTHING, "
    "and so does any phrase whose words are not adjacent in the note ('chant session'). "
    "So: NEVER pass the user's question, a sentence, or a phrase as the query. Search ONE "
    "or TWO adjacent keywords at a time -- proper nouns, project names, distinctive terms "
    "-- and issue SEVERAL separate searches instead of one long one. For 'where did I "
    "leave the Tibet chant study?', search 'Tibet', then 'chant', then 'HRV' as three "
    "calls. If a search returns nothing, that is usually the query's fault, not the "
    "vault's: retry with a single shorter keyword or a different name before you conclude "
    "the vault has nothing.\n"
    "CITE sources inline as [[Note Title]] wikilinks.\n"
    "WRITES are append-only: use vault_capture to save a thought to today's inbox and "
    "vault_log to record an action. You cannot edit or delete anything, and you have no "
    "shell or file tools beyond these four.\n"
    "RESPOND with exactly one JSON object on a single line and nothing else:\n"
    '{"spoken": "<= 3 sentences, natural, read aloud>", '
    '"display": "<fuller markdown answer with [[wikilink]] citations>", '
    '"citations": ["Note Title", ...]}\n'
    "Keep 'spoken' short and conversational; put detail in 'display'."
)


def build_options(vault_root, mcp_server, resume_session_id=None) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cwd=str(vault_root),
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"vault": mcp_server},
        # [] is the SDK's documented "disable ALL built-in tools" -- the closed
        # surface. MCP tools are server-provided, not built-ins, so the four vault
        # tools survive it and are gated by allowed_tools below.
        tools=[],
        allowed_tools=list(VAULT_TOOL_NAMES),
        # allowed_tools only auto-APPROVES; it does not remove anything. Name the
        # natives explicitly too, so they are stripped from the model's context
        # even if a future SDK default puts them back in the base set.
        disallowed_tools=["Bash", "Edit", "Write", "Read", "Grep", "Glob",
                          "WebFetch", "WebSearch"],
        strict_mcp_config=True,
        # cwd IS the vault, and the vault carries its own .claude/ with skills,
        # hooks and MCP servers. [] keeps every one of them out of this session.
        setting_sources=[],
        permission_mode="default",
        resume=resume_session_id,
    )


class Butler:
    def __init__(self, options_builder, state_path, client_factory=ClaudeSDKClient):
        self._build = options_builder          # callable(resume_session_id) -> options
        self._state_path = Path(state_path)
        self._client_factory = client_factory
        self._client = None
        self._session_id = self._load_session_id()
        self._lock = asyncio.Lock()

    def _load_session_id(self):
        try:
            return json.loads(self._state_path.read_text()).get("session_id")
        except (OSError, json.JSONDecodeError, AttributeError):
            return None

    def _save_session_id(self, sid):
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"session_id": sid}))
        tmp.replace(self._state_path)
        self._session_id = sid

    async def _ensure_client(self):
        if self._client is None:
            client = self._client_factory(self._build(self._session_id))
            # connect() FIRST, publish second. A stale `resume` id (e.g. a cleared
            # CLI session store) makes connect() raise on the first turn of a boot;
            # assigning self._client before that would leave a never-connected
            # client in place and wedge every later ask() on a dead object.
            await client.connect()
            self._client = client
        return self._client

    async def ask(self, text: str) -> dict:
        async with self._lock:
            try:
                client = await self._ensure_client()
                await client.query(text)
                chunks, session_id = [], None
                async for msg in client.receive_response():
                    for block in getattr(msg, "content", None) or []:
                        t = getattr(block, "text", None)
                        if t:
                            chunks.append(t)
                    # `subtype` is NOT unique to ResultMessage (SystemMessage and
                    # others carry it too). This works because receive_response()
                    # ends at the ResultMessage, so the LAST message carrying a
                    # subtype -- the one whose session_id wins here -- is it.
                    # Duck-typed so tests need no heavy SDK dataclasses.
                    if getattr(msg, "subtype", None) is not None:
                        session_id = getattr(msg, "session_id", None)
                if session_id and session_id != self._session_id:
                    self._save_session_id(session_id)
                return parse_butler_output("".join(chunks))
            except (Exception, asyncio.CancelledError):
                # Any failure can leave the transport half-dead; drop the client so
                # the next turn builds and connects a fresh one instead of retrying
                # query() against a corpse.
                #
                # CancelledError is spelled out because it derives from
                # BaseException, not Exception: app_brain wraps ask() in
                # asyncio.wait_for, and a timeout CANCELS this coroutine mid
                # `async for`. Without this the client would stay cached with a
                # half-consumed response stream and the next turn would query()
                # into the middle of the abandoned one. The lock is released
                # either way (`async with` unwinds on CancelledError too); this
                # is about the client, not the lock. Always re-raised -- never
                # swallow a cancellation.
                self._client = None
                raise

    async def close(self):
        if self._client is not None:
            await self._client.disconnect()
            self._client = None
