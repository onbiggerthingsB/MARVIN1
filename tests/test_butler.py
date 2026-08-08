import asyncio
import json
from pathlib import Path

import mcp.types as mt
import pytest

from server.butler import Butler, ButlerUnavailable
from server.vault_mcp import VAULT_TOOL_NAMES, build_vault_server


# --- reach the in-process MCP server's real handlers -------------------------
# create_sdk_mcp_server returns {"type": "sdk", "name": ..., "instance": Server}.
# The @server.call_tool() decorator registers under request_handlers[CallToolRequest],
# exactly as @server.list_tools() does for ListToolsRequest (see the four-tool test
# below). Going through the handler runs the REAL tool wrapper -- materialize guard,
# to_thread hop and all -- rather than the bare function it wraps.
async def _call_tool(server, name, args):
    return await server["instance"].request_handlers[mt.CallToolRequest](
        mt.CallToolRequest(method="tools/call",
                           params=mt.CallToolRequestParams(name=name, arguments=args)))


def _tool_text(result) -> str:
    return "".join(c.text for c in result.root.content if c.type == "text")


# --- lightweight fakes: the Butler reads .content/.text/.subtype/.session_id ---
# The result fake is NAMED ResultMessage on purpose: Butler.ask keys session-id
# capture and failure detection on type(msg).__name__ == "ResultMessage" (the
# real SDK class name), so the fake must carry it too -- no other field of the
# heavy SDK dataclass is needed.
class _Text:
    def __init__(self, text): self.text = text


class _Assistant:
    def __init__(self, text): self.content = [_Text(text)]


class ResultMessage:
    def __init__(self, session_id): self.subtype = "success"; self.session_id = session_id


class FakeClient:
    last = None

    def __init__(self, options):
        self.options = options
        self.queries = []
        self.connected = False
        FakeClient.last = self

    async def connect(self): self.connected = True
    async def disconnect(self): self.connected = False
    async def query(self, text, session_id="default"): self.queries.append(text)

    async def receive_response(self):
        yield _Assistant('{"spoken": "Session 2.", '
                         '"display": "You left off at [[Tibet Session 2]].", '
                         '"citations": ["Tibet Session 2"]}')
        yield ResultMessage("sess-123")


def make_butler(tmp_path, resume=None):
    def opts(resume_id):
        return {"resume": resume_id}  # a plain marker; FakeClient just stores it
    return Butler(options_builder=opts,
                  state_path=tmp_path / "butler.json",
                  client_factory=FakeClient)


async def test_ask_returns_parsed_output(tmp_path):
    b = make_butler(tmp_path)
    out = await b.ask("where did I leave the tibet study?")
    assert out["spoken"] == "Session 2."
    assert out["citations"] == ["Tibet Session 2"]
    assert FakeClient.last.queries == ["where did I leave the tibet study?"]


async def test_ask_persists_session_id(tmp_path):
    b = make_butler(tmp_path)
    await b.ask("hi")
    saved = json.loads((tmp_path / "butler.json").read_text())
    assert saved["session_id"] == "sess-123"


async def test_resume_id_loaded_and_passed_to_options(tmp_path):
    (tmp_path / "butler.json").write_text(json.dumps({"session_id": "prev-999"}))
    b = make_butler(tmp_path)
    await b.ask("hi")
    # options_builder was called with the resumed id from disk
    assert FakeClient.last.options == {"resume": "prev-999"}


async def test_vault_server_exposes_four_named_tools(tmp_path):
    server = build_vault_server(tmp_path)
    assert server is not None
    assert VAULT_TOOL_NAMES == [
        "mcp__vault__vault_search", "mcp__vault__vault_read",
        "mcp__vault__vault_capture", "mcp__vault__vault_log"]
    # ...and the server really registers those four, under exactly those names.
    # VAULT_TOOL_NAMES is the allowlist: a wrapper whose @tool name drifts from
    # it is still registered but never permitted, so the butler loses the
    # capability silently at runtime. Assert the real list_tools reply, not a
    # second copy of the constant.
    listed = await server["instance"].request_handlers[mt.ListToolsRequest](
        mt.ListToolsRequest(method="tools/list"))
    assert [f"mcp__vault__{t.name}" for t in listed.root.tools] == VAULT_TOOL_NAMES


async def test_failed_connect_does_not_wedge_the_butler(tmp_path):
    """A raising connect() must not leave a never-connected client behind.

    A stale `resume` id in state/butler.json (a session the CLI store no longer
    has) fails on the FIRST turn of a boot. If the dead client stayed cached,
    every later ask() would skip the `is None` branch and call query() on it.
    """
    attempts = []

    class FlakyClient(FakeClient):
        async def connect(self):
            attempts.append(self)
            if len(attempts) == 1:
                raise RuntimeError("No conversation found with session ID: sess-stale")
            self.connected = True

    b = Butler(options_builder=lambda resume_id: {"resume": resume_id},
               state_path=tmp_path / "butler.json",
               client_factory=FlakyClient)

    with pytest.raises(RuntimeError):
        await b.ask("first turn of the boot")
    assert b._client is None, "a never-connected client was left cached"

    # the NEXT turn must build a fresh client and work
    out = await b.ask("second turn")
    assert out["spoken"] == "Session 2."
    assert len(attempts) == 2 and attempts[0] is not attempts[1]


async def test_stale_resume_id_is_cleared_after_connect_failure(tmp_path):
    """A stale resume id must not brick the butler permanently.

    test_failed_connect_does_not_wedge_the_butler above is a FALSE NEGATIVE on
    this: its fake succeeds on attempt 2 whatever resume id it was handed, so it
    passed even when ask() dropped the client and kept the poisoned id. In the
    real world the id is what connect() rejects, so every later turn rebuilt the
    same options and failed identically -- forever, with the fallback line as the
    only symptom and `rm state/butler.json` as the only cure. Assert the SECOND
    attempt is handed resume=None.
    """
    state = tmp_path / "butler.json"
    state.write_text(json.dumps({"session_id": "stale-999"}))
    seen_resumes = []

    class FailFirstConnect(FakeClient):
        attempts = 0

        def __init__(self, options):
            super().__init__(options)
            seen_resumes.append(options["resume"])

        async def connect(self):
            FailFirstConnect.attempts += 1
            if FailFirstConnect.attempts == 1:
                raise RuntimeError("no such session")
            await super().connect()

    b = Butler(options_builder=lambda r: {"resume": r},
               state_path=state, client_factory=FailFirstConnect)
    with pytest.raises(RuntimeError):
        await b.ask("first")
    out = await b.ask("second")            # must start a FRESH session and work
    assert out["spoken"]
    assert seen_resumes == ["stale-999", None]


async def test_good_session_id_survives_an_unrelated_turn_failure(tmp_path):
    """Only a CONNECT failure clears the id, and only when a resume was in play.

    Clearing on every failure would throw away a healthy session -- and the
    butler's whole value is conversational continuity across turns.
    """
    state = tmp_path / "butler.json"

    class QueryBoom(FakeClient):
        async def query(self, text, session_id="default"):
            raise RuntimeError("mid-turn transport hiccup")

    b = Butler(options_builder=lambda r: {"resume": r},
               state_path=state, client_factory=QueryBoom)
    b._session_id = "good-123"
    state.write_text(json.dumps({"session_id": "good-123"}))
    with pytest.raises(RuntimeError):
        await b.ask("hi")
    assert b._session_id == "good-123"
    assert state.exists()


async def test_fallback_prefers_the_last_message_not_the_narration(tmp_path):
    """Without clean JSON the old code spoke the model's pre-tool-call preamble.

    Every assistant message in the turn was concatenated, so "Let me search the
    vault for Tibet..." ended up at the FRONT of the plain-text fallback and got
    read aloud as the answer.
    """
    class Narrating(FakeClient):
        async def receive_response(self):
            yield _Assistant("Let me search the vault for Tibet...")
            yield _Assistant("Session 2 is where you left it. See [[Tibet Session 2]].")
            yield ResultMessage("sess-123")

    b = Butler(options_builder=lambda r: {"resume": r},
               state_path=tmp_path / "butler.json", client_factory=Narrating)
    out = await b.ask("where did I leave off?")
    assert "Let me search" not in out["spoken"]
    assert out["display"].startswith("Session 2 is where you left it.")
    assert out["citations"] == ["Tibet Session 2"]


async def test_json_in_an_earlier_message_is_still_found(tmp_path):
    """The concatenation fallback stays: a reply followed by a trailing chatty
    message must not lose the JSON object the model did emit."""
    class TrailingChatter(FakeClient):
        async def receive_response(self):
            yield _Assistant('{"spoken": "Session 2.", "display": "At [[Tibet]].", '
                             '"citations": ["Tibet"]}')
            yield _Assistant("Let me know if you want the full note.")
            yield ResultMessage("sess-123")

    b = Butler(options_builder=lambda r: {"resume": r},
               state_path=tmp_path / "butler.json", client_factory=TrailingChatter)
    out = await b.ask("hi")
    assert out["spoken"] == "Session 2."
    assert out["citations"] == ["Tibet"]


# --- the first live failure: an API/auth error the SDK does NOT raise --------
# Captured evidence from the real failing turn: receive_response() ended with
#   ResultMessage: subtype='success'  is_error=True  api_error_status='401'
#                  terminal_reason='api_error'
#                  result='Failed to authenticate. API Error: 401 OAuth ...'
# and Butler.ask returned that text as a normal answer, which the brain then
# SPOKE ALOUD in the JARVIS voice. Detection must key on the structured fields
# (is_error / api_error_status / terminal_reason), never the text.

async def test_api_error_result_raises_instead_of_answering(tmp_path):
    class ResultMessage:                 # mirrors the captured evidence exactly
        subtype = "success"              # the SDK really does say 'success' here
        is_error = True
        api_error_status = "401"
        terminal_reason = "api_error"
        session_id = "sess-should-not-be-saved"
        result = ("Failed to authenticate. API Error: 401 OAuth access token "
                  "has been revoked.")

    class ErrClient(FakeClient):
        async def receive_response(self):
            yield _Assistant(ResultMessage.result)
            yield ResultMessage()

    state = tmp_path / "butler.json"
    b = Butler(options_builder=lambda r: {"resume": r},
               state_path=state, client_factory=ErrClient)
    with pytest.raises(ButlerUnavailable) as ei:
        await b.ask("where did I leave the tibet study?")
    assert ei.value.reason == "login expired"
    assert "401" in (ei.value.detail or "") or ei.value.status == "401"
    assert not state.exists()        # a failed turn must not persist a session id
    assert b._client is None         # the auth-dead client was dropped for reaping


async def test_rate_limited_result_classifies_as_rate_limited(tmp_path):
    class ResultMessage:
        subtype = "success"
        is_error = True
        api_error_status = "429"
        terminal_reason = "api_error"
        session_id = "sess-should-not-be-saved"
        result = "API Error: 429 rate limit exceeded"

    class ErrClient(FakeClient):
        async def receive_response(self):
            yield ResultMessage()

    state = tmp_path / "butler.json"
    b = Butler(options_builder=lambda r: {"resume": r},
               state_path=state, client_factory=ErrClient)
    with pytest.raises(ButlerUnavailable) as ei:
        await b.ask("hi")
    assert ei.value.reason == "rate limited"
    assert ei.value.status == "429"
    assert not state.exists()


async def test_answer_about_an_error_note_is_not_mistaken_for_a_failure(tmp_path):
    """Keke legitimately asking about an error recorded in a vault note must get
    the answer: detection keys on ResultMessage fields, never the message text."""
    class ErrorNoteClient(FakeClient):
        async def receive_response(self):
            yield _Assistant('{"spoken": "Your note says the token was revoked.", '
                             '"display": "Per [[Auth Bug]]: Failed to authenticate. '
                             'API Error: 401 OAuth access token has been revoked.", '
                             '"citations": ["Auth Bug"]}')
            yield ResultMessage("sess-123")

    b = Butler(options_builder=lambda r: {"resume": r},
               state_path=tmp_path / "butler.json", client_factory=ErrorNoteClient)
    out = await b.ask("what did the auth bug note say?")
    assert out["citations"] == ["Auth Bug"]
    assert b._session_id == "sess-123"   # a healthy turn still saves its id


def test_system_prompt_carries_the_vault_map():
    """setting_sources=[] keeps the vault's own CLAUDE.md out of the session (the
    right security trade), which leaves the butler with no folder map at all and
    no directory-listing tool. The prompt has to carry one."""
    from server.butler import SYSTEM_PROMPT
    for folder in ("Daily/", "Wiki/", "Sources/", "_Claude/", "Claude Chats/",
                   "Fieldwork/", "College Prep/", "Soccer/"):
        assert folder in SYSTEM_PROMPT
    assert "_Claude/index.md" in SYSTEM_PROMPT
    # and the model must be told a big total means it is seeing a subset
    assert "narrow" in SYSTEM_PROMPT.lower()


async def test_cancelled_turn_leaves_butler_usable(tmp_path):
    """A timed-out turn is a CANCELLED turn, and cancellation must reset the client.

    app_brain wraps ask() in asyncio.wait_for, which cancels the inner coroutine
    on timeout. CancelledError is a BaseException, so a bare `except Exception`
    in ask() would not fire: the persistent client would stay cached with a
    half-consumed response stream, and the next turn would query() into the
    middle of the abandoned one. The fake models exactly that -- a client
    cancelled mid-stream refuses further queries -- so this test fails if the
    handler stops covering cancellation.
    """
    class HangingThenOkClient(FakeClient):
        calls = 0
        poisoned = False

        async def query(self, text, session_id="default"):
            if self.poisoned:
                raise RuntimeError("query() into a half-consumed response stream")
            await super().query(text, session_id)

        async def receive_response(self):
            HangingThenOkClient.calls += 1
            if HangingThenOkClient.calls == 1:
                try:
                    await asyncio.sleep(60)      # will be cancelled
                except asyncio.CancelledError:
                    self.poisoned = True         # stream abandoned mid-flight
                    raise
            async for m in super().receive_response():
                yield m

    b = Butler(options_builder=lambda r: {"resume": r},
               state_path=tmp_path / "butler.json",
               client_factory=HangingThenOkClient)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(b.ask("first"), 0.05)
    assert b._client is None, "a client cancelled mid-stream was left cached"
    assert not b._lock.locked(), "the lock was not released on cancellation"

    out = await b.ask("second")              # must work, not deadlock
    assert out["spoken"] == "Session 2."
    assert HangingThenOkClient.calls == 2


def test_build_options_closes_the_tool_surface(tmp_path):
    from server.butler import build_options
    from server.vault_mcp import build_vault_server, VAULT_TOOL_NAMES

    opts = build_options(tmp_path, build_vault_server(tmp_path), None)

    # [] disables ALL built-in tools; None would OMIT the flag and restore them.
    assert opts.tools == []
    assert opts.tools is not None
    # the four vault tools are the entire allowed surface
    assert opts.allowed_tools == list(VAULT_TOOL_NAMES)
    # natives explicitly denied as belt-and-braces
    for native in ("Bash", "Edit", "Write", "Read", "Grep", "Glob"):
        assert native in opts.disallowed_tools
    # no filesystem settings, no foreign MCP servers
    assert opts.setting_sources == []
    assert opts.strict_mcp_config is True
    assert set(opts.mcp_servers.keys()) == {"vault"}
    # grounded in the vault
    assert str(opts.cwd) == str(tmp_path)


def test_build_options_passes_resume_id(tmp_path):
    from server.butler import build_options
    from server.vault_mcp import build_vault_server
    opts = build_options(tmp_path, build_vault_server(tmp_path), "sess-abc")
    assert opts.resume == "sess-abc"


async def test_search_tool_reports_unavailable_instead_of_empty(tmp_path, monkeypatch):
    """A failed search must render a DISTINCT message, not 'No matching notes':
    the latter makes the butler affirmatively claim the vault lacks the topic."""
    import server.vault_mcp as vm

    async def broken_search(query, root, limit):
        return {"total": 0, "results": [], "error": "timed out after 8s"}

    monkeypatch.setattr(vm, "vault_search", broken_search)
    result = await _call_tool(build_vault_server(tmp_path), "vault_search",
                              {"query": "chant", "limit": 5})
    text = _tool_text(result)
    assert "unavailable" in text.lower()
    assert "timed out after 8s" in text
    assert "No matching notes" not in text


async def test_state_write_failure_does_not_discard_the_answer(tmp_path, monkeypatch):
    """A full or read-only state dir must cost the SAVE, not the turn: the
    answer is already computed, and in-memory continuity must survive."""
    b = make_butler(tmp_path)

    def disk_full(sid):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(b, "_save_session_id", disk_full)
    out = await b.ask("hi")
    assert out["spoken"] == "Session 2."            # the answer survived
    assert b._session_id == "sess-123"              # in-memory continuity kept
    assert b._client is not None                    # client NOT dropped


async def test_failed_turn_disposes_the_dropped_client(tmp_path):
    """Nulling the client without disconnecting leaks the `claude` subprocess
    (no __del__ on ClaudeSDKClient) until process exit -- and with a 120s ask
    timeout upstream, failed/timed-out turns are an expected event."""
    disconnected = []

    class QueryBoom(FakeClient):
        async def query(self, text, session_id="default"):
            raise RuntimeError("mid-turn transport hiccup")

        async def disconnect(self):
            disconnected.append(self)

    b = Butler(options_builder=lambda r: {"resume": r},
               state_path=tmp_path / "butler.json", client_factory=QueryBoom)
    with pytest.raises(RuntimeError):
        await b.ask("hi")
    assert b._client is None
    for _ in range(5):                 # let the detached reaper run
        await asyncio.sleep(0)
    assert disconnected, "dropped client was never disconnected (subprocess leak)"


async def test_cancelled_turn_also_disposes_the_dropped_client(tmp_path):
    """The reaper must be DETACHED on the cancellation path -- awaiting
    disconnect inline there would delay or swallow the cancellation."""
    disconnected = []

    class Hanging(FakeClient):
        async def receive_response(self):
            await asyncio.sleep(60)
            yield _Assistant("never")

        async def disconnect(self):
            disconnected.append(self)

    b = Butler(options_builder=lambda r: {"resume": r},
               state_path=tmp_path / "butler.json", client_factory=Hanging)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(b.ask("hi"), 0.05)
    assert b._client is None
    for _ in range(5):
        await asyncio.sleep(0)
    assert disconnected, "cancelled turn leaked its client's subprocess"


# --- opt-in API-key fallback: free-first, paid only on an auth-class refusal --
# The live failure this covers: the Max subscription authenticates, but headless
# requests 403 ("Request not allowed"). `claude --bare` forces API-key auth and
# keeps --mcp-config, so a single paid retry rescues the turn -- and the moment
# the subscription works again, the paid path stops being used.

def _fallback_butler(tmp_path, factory):
    """A Butler whose options_builder takes the extended (resume, use_api_key)
    contract and records the mode as a plain marker on the options dict."""
    return Butler(
        options_builder=lambda resume, use_api_key=False: {
            "resume": resume, "use_api_key": use_api_key},
        state_path=tmp_path / "butler.json",
        client_factory=factory)


class _AuthErrResult:                    # mirrors the live headless-403 evidence
    subtype = "success"
    is_error = True
    api_error_status = "403"
    terminal_reason = "api_error"
    session_id = "sess-should-not-be-saved"
    result = "API Error: 403 Request not allowed"


async def test_no_api_key_means_no_retry_and_nothing_billed(tmp_path, monkeypatch):
    """Default behavior UNCHANGED: without ANTHROPIC_API_KEY an auth failure
    raises after exactly ONE attempt -- no paid retry can even be considered."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    built = []

    class ResultMessage(_AuthErrResult):
        pass

    class AlwaysAuthErr(FakeClient):
        def __init__(self, options):
            super().__init__(options)
            built.append(options)

        async def receive_response(self):
            yield ResultMessage()

    b = _fallback_butler(tmp_path, AlwaysAuthErr)
    with pytest.raises(ButlerUnavailable) as ei:
        await b.ask("hi")
    assert ei.value.reason == "login expired"
    assert len(built) == 1, "retried without a key -- a second attempt was built"
    assert b.using_api_key is False


async def test_auth_failure_with_key_retries_once_in_api_key_mode(tmp_path, monkeypatch):
    """Key present + subscription refused: the SAME turn is retried exactly once
    on a fresh client built in API-key mode, and the answer comes back."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    built = []

    class ResultMessage(_AuthErrResult):
        pass

    class FailsThenWorks(FakeClient):
        def __init__(self, options):
            super().__init__(options)
            built.append(options)

        async def receive_response(self):
            if len(built) == 1:          # first client: the subscription 403
                yield ResultMessage()
                return
            async for m in super().receive_response():   # second client: healthy
                yield m

    b = _fallback_butler(tmp_path, FailsThenWorks)
    out = await b.ask("where did I leave the tibet study?")
    assert out["spoken"] == "Session 2."
    assert len(built) == 2
    assert built[0]["use_api_key"] is False   # free path tried first
    assert built[1]["use_api_key"] is True    # retry carried the paid mode
    assert b.using_api_key is True            # visible to a console indicator


async def test_healthy_subscription_never_falls_back_even_with_key(tmp_path, monkeypatch):
    """The anti-billing regression: a key in the environment must be inert
    while the free subscription path works."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    built = []

    class Healthy(FakeClient):
        def __init__(self, options):
            super().__init__(options)
            built.append(options)

    b = _fallback_butler(tmp_path, Healthy)
    out = await b.ask("hi")
    assert out["spoken"] == "Session 2."
    assert len(built) == 1
    assert built[0]["use_api_key"] is False
    assert b.using_api_key is False


def test_build_options_api_key_mode_is_additive_only(tmp_path, monkeypatch):
    """use_api_key=True adds EXACTLY --bare + the key; use_api_key=False (the
    default) adds neither. In both modes the closed tool surface is identical:
    bare mode must not widen what the butler can touch."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-probe")
    from server.butler import build_options
    from server.vault_mcp import build_vault_server, VAULT_TOOL_NAMES

    server = build_vault_server(tmp_path)
    free = build_options(tmp_path, server, None)
    paid = build_options(tmp_path, server, None, use_api_key=True)

    # paid mode: the bare flag (None value == valueless flag) and the key
    assert paid.extra_args == {"bare": None}
    assert paid.env.get("ANTHROPIC_API_KEY") == "sk-probe"
    # free mode: NEITHER -- nothing bills by default
    assert not free.extra_args
    assert "ANTHROPIC_API_KEY" not in (free.env or {})

    # identical closed surface in BOTH modes
    for opts in (free, paid):
        assert opts.tools == [] and opts.tools is not None
        assert opts.allowed_tools == list(VAULT_TOOL_NAMES)
        assert opts.setting_sources == []
        assert opts.strict_mcp_config is True
        for native in ("Bash", "Edit", "Write", "Read", "Grep", "Glob"):
            assert native in opts.disallowed_tools


async def test_read_tool_materializes_and_runs_off_the_event_loop(tmp_path, monkeypatch):
    import threading
    import server.vault_mcp as vm

    (tmp_path / "Wiki").mkdir(parents=True)
    (tmp_path / "Wiki" / "N.md").write_text("hello", encoding="utf-8")

    materialized = []
    read_threads = []

    async def fake_materialize(path):
        materialized.append(str(path))

    real_read = vm.vault_read

    def spy_read(rel, root):
        read_threads.append(threading.current_thread().name)
        return real_read(rel, root)

    monkeypatch.setattr(vm, "ensure_materialized", fake_materialize)
    monkeypatch.setattr(vm, "vault_read", spy_read)

    # invoke the real vault_read tool handler through the built server
    result = await _call_tool(build_vault_server(tmp_path), "vault_read", {"path": "Wiki/N.md"})

    assert "hello" in _tool_text(result)
    assert materialized, "ensure_materialized was not called before reading"
    assert read_threads and read_threads[0] != threading.main_thread().name, \
        "vault_read ran on the event loop instead of a worker thread"
