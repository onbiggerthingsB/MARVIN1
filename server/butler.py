"""The butler: one persistent Claude Agent SDK session grounded in the vault."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from server.butler_parse import _load_object, parse_butler_output
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
    "When a search reports FAR MORE matching notes than it shows you, you are looking "
    "at a subset: narrow with a more specific keyword or an adjacent two-word phrase "
    "before you answer, rather than accepting the top few as everything the vault has.\n"
    "CITE sources inline as [[Note Title]] wikilinks.\n"
    "VAULT MAP (top-level folders):\n"
    "- Daily/ — daily notes and inbox captures\n"
    "- Research/, Coursework/, Fieldwork/, NUTRITION/, Nutrition Club/, College Prep/, "
    "Projects/, Soccer/ — Keke's own notes\n"
    "- Wiki/ — distilled topic pages and MOCs; Wiki/Projects/ holds project briefs and specs\n"
    "- Sources/ — raw source material (papers, PDFs)\n"
    "- Claude Chats/, ChatGPT Chats/ — archived AI conversations; treat as secondary, "
    "not primary sources\n"
    "- _Claude/ — the agent control room: index.md (catalog of notes), "
    "memory/MEMORY.md (durable facts), log.md\n"
    "Prefer Keke's own notes and Wiki/ pages over chat archives. _Claude/index.md is a "
    "catalog worth reading when you need orientation.\n"
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


def _parse_object(text: str) -> dict | None:
    """parse_butler_output(text), but only if `text` really carried a butler
    JSON object with content. None means "nothing usable in here".

    The gate mirrors parse_butler_output's own: an object holding a non-empty
    display/spoken. Anything else would take that function's plain-text
    fallback, which is a different decision and belongs to the caller.
    """
    obj = _load_object(text)
    if isinstance(obj, dict) and str(obj.get("display") or obj.get("spoken") or "").strip():
        return parse_butler_output(text)
    return None


def _best_parse(messages: list[str]) -> dict:
    """Parse a turn's assistant messages, preferring the LAST one.

    A turn that used tools emits several assistant messages: a narration before
    each tool call ("Let me search the vault for Tibet...") and the real reply at
    the end. Concatenating them all and parsing the blob is fine while the model
    emits clean JSON -- _load_object finds the object wherever it sits -- but the
    moment it does not, parse_butler_output's plain-text fallback speaks the
    WHOLE blob, so JARVIS reads its own narration aloud as the answer.

    Order, therefore:
      1. the last non-empty message, if it carried a butler object -- the normal case;
      2. the full concatenation, if an object was emitted earlier in the turn and
         then followed by chatter -- the old behaviour, kept as a fallback;
      3. plain text of the LAST message. This is the fix: when there is no JSON
         anywhere, the thing spoken is the model's final say, not its preamble.
    parse_butler_output is the parser throughout.
    """
    last = next((m for m in reversed(messages) if m and m.strip()), "")
    joined = "".join(messages)
    return (_parse_object(last) or _parse_object(joined)
            or parse_butler_output(last or joined))


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

    def _clear_session_file(self):
        """Forget the persisted resume id. Best-effort; never raises."""
        try:
            self._state_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 — a read-only state dir must not break recovery
            pass

    async def _ensure_client(self):
        if self._client is None:
            resume = self._session_id
            client = self._client_factory(self._build(resume))
            # connect() FIRST, publish second. A stale `resume` id (e.g. a cleared
            # CLI session store) makes connect() raise on the first turn of a boot;
            # assigning self._client before that would leave a never-connected
            # client in place and wedge every later ask() on a dead object.
            try:
                await client.connect()
            except Exception:
                # ...and dropping the client alone is not enough. The id that made
                # connect() fail is still on disk and in memory, so the next turn
                # rebuilds options with the SAME stale resume and fails identically
                # -- forever, with the fallback line as the only symptom and
                # deleting state/butler.json by hand as the only cure. Clear it
                # here, where we know a resume id was actually in play, so an
                # unrelated mid-turn failure cannot wipe a good session id.
                if resume is not None:
                    self._session_id = None
                    self._clear_session_file()
                raise
            self._client = client
        return self._client

    async def ask(self, text: str) -> dict:
        async with self._lock:
            try:
                client = await self._ensure_client()
                await client.query(text)
                # Per-MESSAGE texts, not one flat list of blocks: a turn with tool
                # calls emits a preamble message ("Let me search the vault for
                # Tibet...") before the real reply, and _best_parse needs them
                # separable to prefer the last one. See _best_parse.
                messages, session_id = [], None
                async for msg in client.receive_response():
                    parts = []
                    for block in getattr(msg, "content", None) or []:
                        t = getattr(block, "text", None)
                        if t:
                            parts.append(t)
                    if parts:
                        messages.append("".join(parts))
                    # `subtype` is NOT unique to ResultMessage (SystemMessage and
                    # others carry it too). This works because receive_response()
                    # ends at the ResultMessage, so the LAST message carrying a
                    # subtype -- the one whose session_id wins here -- is it.
                    # Duck-typed so tests need no heavy SDK dataclasses.
                    if getattr(msg, "subtype", None) is not None:
                        session_id = getattr(msg, "session_id", None)
                if session_id and session_id != self._session_id:
                    self._save_session_id(session_id)
                return _best_parse(messages)
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
