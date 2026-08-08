"""The butler: one persistent Claude Agent SDK session grounded in the vault."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from server.butler_parse import _load_object, parse_butler_output
from server.vault_mcp import VAULT_TOOL_NAMES

log = logging.getLogger(__name__)


class ButlerUnavailable(RuntimeError):
    """The turn failed at the API/CLI layer — the model never answered.

    Raised instead of returning the transport's error text as an answer, which
    would otherwise be spoken aloud to the user as though it were a real reply.
    """

    def __init__(self, reason: str, detail: str = "", status: str | None = None):
        super().__init__(detail or reason)
        self.reason = reason        # short, safe to speak/display
        self.detail = detail        # raw transport text, for logs and the console
        self.status = status        # e.g. "401", when the SDK gave one


def _failure_reason(status: str | None, text: str) -> str:
    """Classify a transport failure for a human. CLASSIFICATION only: detection
    (in ask()) keys on ResultMessage fields and never string-matches, because
    these words legitimately appear in vault notes Keke might ask about. Once
    the structured fields say the turn failed, the text may refine the label."""
    low = (text or "").lower()
    if status in ("401", "403") or any(w in low for w in ("authenticat", "oauth", "login")):
        return "login expired"
    if status == "429" or "rate limit" in low:
        return "rate limited"
    if (status or "").startswith("5"):
        return "the service is having trouble"
    return "the model is unavailable"


# Strong refs to detached reaper tasks: the event loop holds tasks weakly, so a
# bare create_task could be garbage-collected mid-disconnect.
_REAPERS: set = set()


async def _dispose(client) -> None:
    """Best-effort disconnect of a dropped client, so the CLI child it spawned
    (order 100MB RSS) does not survive until process exit. Detached: nothing
    awaits it, and it must never propagate anything."""
    try:
        await client.disconnect()
    except BaseException:  # noqa: BLE001 — best effort; never propagate
        pass

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


def build_options(vault_root, mcp_server, resume_session_id=None, *,
                  use_api_key: bool = False) -> ClaudeAgentOptions:
    # Opt-in paid path: `--bare` forces the CLI to authenticate with
    # ANTHROPIC_API_KEY only (OAuth/keychain never read), which sidesteps the
    # subscription 403 on headless requests. The SDK renders a None-valued
    # extra_args entry as a bare flag, and MERGES options.env over the
    # inherited os.environ (subprocess_cli.py), so passing just the key is
    # enough -- and avoids re-injecting CLAUDECODE, which the SDK deliberately
    # strips from the child env. Everything else (the closed tool surface
    # above all) stays byte-identical to the free path.
    api_key_kwargs = {}
    if use_api_key:
        api_key_kwargs = dict(
            extra_args={"bare": None},
            env={"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]},
        )
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
        **api_key_kwargs,
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


def _accepts_use_api_key(builder) -> bool:
    """Whether options_builder takes the `use_api_key` keyword.

    Decided ONCE by signature inspection rather than per-call try/except
    TypeError: the except would also swallow a TypeError raised INSIDE a
    two-argument builder's body (e.g. a bad kwarg into ClaudeAgentOptions)
    and silently re-call it single-argument -- a confusing double invocation
    of a function that may not be idempotent. Builders whose signature cannot
    be introspected are treated as legacy single-argument.
    """
    try:
        params = inspect.signature(builder).parameters
    except (TypeError, ValueError):
        return False
    return "use_api_key" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class Butler:
    def __init__(self, options_builder, state_path, client_factory=ClaudeSDKClient):
        self._build = options_builder          # callable(resume_session_id) -> options
        self._build_accepts_mode = _accepts_use_api_key(options_builder)
        self._state_path = Path(state_path)
        self._client_factory = client_factory
        self._client = None
        self._session_id = self._load_session_id()
        # Whether the CURRENT client was built in API-key (--bare, paid) mode.
        # False on every fresh client so a fixed subscription is always tried
        # first; flipped by ask()'s auth-fallback retry only.
        self._api_key_mode = False
        self._lock = asyncio.Lock()

    @staticmethod
    def _api_key_available() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def using_api_key(self) -> bool:
        """True while the live client authenticates with ANTHROPIC_API_KEY,
        i.e. while turns bill per token. Read-only, for console indicators."""
        return self._api_key_mode

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

    def _build_options(self, resume):
        if self._build_accepts_mode:
            return self._build(resume, use_api_key=self._api_key_mode)
        return self._build(resume)             # legacy single-argument builder

    async def _ensure_client(self):
        if self._client is None:
            resume = self._session_id
            client = self._client_factory(self._build_options(resume))
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
            # The mode the FIRST attempt runs under, captured before _ask_once's
            # failure path resets it: a turn that already failed on the paid
            # path must propagate, never retry onto the same dead end.
            first_mode = self._api_key_mode
            try:
                return await self._ask_once(text)
            except ButlerUnavailable as e:
                # Free-first, paid only as a fallback: retry ONCE with --bare +
                # ANTHROPIC_API_KEY, and only for an AUTH-class refusal (the
                # headless-403 case). Rate limits, 5xx and the rest propagate
                # untouched. The dead client was already dropped and reaped by
                # _ask_once; the retry is a genuine second attempt through
                # _ensure_client, under the same lock hold.
                if (e.reason == "login expired" and not first_mode
                        and self._api_key_available()):
                    log.warning(
                        "subscription auth refused (%s); falling back to "
                        "ANTHROPIC_API_KEY -- this bills per token",
                        e.detail or e.reason)
                    self._api_key_mode = True
                    return await self._ask_once(text)
                raise

    async def _ask_once(self, text: str) -> dict:
        try:
            client = await self._ensure_client()
            await client.query(text)
            # Per-MESSAGE texts, not one flat list of blocks: a turn with tool
            # calls emits a preamble message ("Let me search the vault for
            # Tibet...") before the real reply, and _best_parse needs them
            # separable to prefer the last one. See _best_parse.
            messages, session_id, failure = [], None, None
            async for msg in client.receive_response():
                parts = []
                for block in getattr(msg, "content", None) or []:
                    t = getattr(block, "text", None)
                    if t:
                        parts.append(t)
                if parts:
                    messages.append("".join(parts))
                # Keyed on the CLASS NAME, not the old `subtype is not None`
                # probe: SystemMessage carries a subtype too ('init',
                # 'api_retry'), and current SDKs put a session_id on
                # AssistantMessage, so no single attribute is distinctive.
                # The name check is exact against the real SDK, keeps the
                # test fakes light (isinstance would force constructing the
                # full SDK dataclass), and a SystemMessage -- whatever
                # fields a future SDK gives it -- can never overwrite the
                # saved id.
                if type(msg).__name__ == "ResultMessage":
                    # Authoritative failure signals, read defensively (SDK
                    # fields move). `subtype` is NOT one of them: the live
                    # auth failure arrived with subtype='success'. And the
                    # message TEXT is never used for detection -- Keke
                    # asking about an error recorded in a vault note must
                    # not trip this.
                    status = getattr(msg, "api_error_status", None)
                    status = str(status) if status not in (None, "") else None
                    if (getattr(msg, "is_error", None) or status is not None
                            or getattr(msg, "terminal_reason", None) == "api_error"):
                        # A failed turn's id is not a resumable session:
                        # do NOT capture it. Raise only after the loop has
                        # finished draining the response.
                        detail = (str(getattr(msg, "result", None) or "").strip()
                                  or (messages[-1] if messages else ""))
                        failure = ButlerUnavailable(
                            _failure_reason(status, detail), detail, status)
                    else:
                        session_id = getattr(msg, "session_id", None)
            if failure is not None:
                # Flows through the except below: an auth-failed client is
                # dead, so it is dropped and reaped like any other failure.
                raise failure
            if session_id and session_id != self._session_id:
                # In-memory FIRST: the running session keeps continuity even
                # if the disk save below fails.
                self._session_id = session_id
                try:
                    # Best-effort: a full or read-only state dir must not
                    # discard the answer we already computed (the generic
                    # handler would drop the client and speak the fallback
                    # line on EVERY turn, forever). to_thread because the
                    # write is blocking I/O on a loop carrying live audio.
                    await asyncio.to_thread(self._save_session_id, session_id)
                except OSError as e:
                    log.warning("could not persist session id: %s", e)
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
            #
            # The dropped client goes to a DETACHED reaper rather than being
            # nulled and forgotten: ClaudeSDKClient has no __del__, so an
            # undisconnected client leaks its `claude` subprocess until
            # process exit -- and with a 120s ask timeout upstream, timeouts
            # are an expected event, not a rarity. Never awaited inline: on
            # the cancellation path an inline await could swallow or delay
            # the cancellation we are re-raising.
            old, self._client = self._client, None
            # Every dropped client resets to the free path: the next fresh
            # client retries the subscription first, so a fixed 403 stops
            # billing automatically. ask()'s fallback re-flips this AFTER the
            # drop when it decides to retry paid, so the order here is safe.
            self._api_key_mode = False
            if old is not None:
                task = asyncio.get_running_loop().create_task(_dispose(old))
                _REAPERS.add(task)                      # keep a strong ref
                task.add_done_callback(_REAPERS.discard)
            raise

    async def close(self):
        if self._client is not None:
            await self._client.disconnect()
            self._client = None
        self._api_key_mode = False                      # fresh client, free path first
