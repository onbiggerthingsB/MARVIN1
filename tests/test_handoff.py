import asyncio
from types import SimpleNamespace

from server.fleet_state import CLOSED, DETACHED, QUIET_AFTER_S, UNKNOWN
from tests.test_app_auth import bootstrap, make_client
from tests.test_fleet import (FakeClient, ResultMessage, cleanup, make_fleet,
                              repo)


async def spawned(tmp_path, monkeypatch, **kw):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch, **kw)
    opened = []
    async def fake_terminal(cmd):
        opened.append(cmd)
    fleet._open_terminal = fake_terminal
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    clients[0].stream.put_nowait(ResultMessage(session_id="sess-42"))
    await asyncio.sleep(0.05)                     # session id captured
    return fleet, bus, router, clients, path, opened


async def test_handoff_runs_the_full_lockout_and_detaches(tmp_path, monkeypatch):
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    result = await fleet.handoff(path)
    assert result["ok"] is True
    w = fleet.workers[0]
    assert w.machine.base == DETACHED
    assert clients[0].interrupted and clients[0].disconnected
    assert "claude --resume sess-42" in result["command"]
    assert w.worktree.path in result["command"]   # cd into the WORKTREE, not the repo
    assert opened == [result["command"]]          # Terminal actually launched


async def test_handoff_refuses_without_a_resumable_session(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    async def fake_terminal(cmd): raise AssertionError("must not launch")
    fleet._open_terminal = fake_terminal
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")     # no ResultMessage → no session id
    try:
        result = await fleet.handoff(path)
        assert result["ok"] is False and "command" not in result
        assert fleet.workers[0].machine.base != DETACHED
    finally:
        await cleanup(fleet)


async def test_a_failed_close_leaves_unknown_and_no_resume_command(tmp_path, monkeypatch):
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    async def boom():
        raise RuntimeError("subprocess refused to die")
    clients[0].disconnect = boom
    result = await fleet.handoff(path)
    assert result["ok"] is False and "command" not in result
    assert fleet.workers[0].machine.base == UNKNOWN   # honest, and NOT detached
    assert opened == []                               # two drivers never happen


async def test_pending_approvals_are_rejected_before_the_handoff(tmp_path, monkeypatch):
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    w = fleet.workers[0]
    decision = asyncio.create_task(w._on_tool_request(
        "Bash", {"command": "npm test"}, SimpleNamespace(title=None)))
    await asyncio.sleep(0.05)
    assert len(router.pending_approvals()) == 1
    result = await fleet.handoff(path)
    assert result["ok"] is True
    got = await asyncio.wait_for(decision, 1)
    assert got.behavior == "deny"                     # unblocked, not abandoned
    assert router.pending_approvals() == []           # nothing left voice-addressable


async def test_a_detached_worker_refuses_steer_and_approvals(tmp_path, monkeypatch):
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    await fleet.handoff(path)
    assert "detached" in fleet.steer_path(path, "do more").lower()
    assert fleet.deliver_approval("any-nonce", True) is False
    assert "detached" in (await fleet.stop(path)).lower()


async def test_handoff_survives_a_terminal_that_will_not_open(tmp_path, monkeypatch):
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    async def no_terminal(cmd):
        raise RuntimeError("osascript missing")
    fleet._open_terminal = no_terminal
    result = await fleet.handoff(path)
    # The SDK side is already closed and locked — the session IS detached; the
    # command is on screen for Keke to run by hand.
    assert result["ok"] is True and "command" in result
    assert fleet.workers[0].machine.base == DETACHED
    assert "run the command" in result["spoken"].lower()


# ---------- the remaining failure points of the sequence ----------
def _drain(q):
    out = []
    while not q.empty():
        ev = q.get_nowait()
        if ev:
            out.append(ev)
    return out


async def test_handoff_on_a_path_with_no_worker_opens_nothing(tmp_path, monkeypatch):
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    try:
        result = await fleet.handoff(repo(tmp_path, "alethic"))
        assert result["ok"] is False and "command" not in result
        assert "Nothing is running there" in result["spoken"]
        assert opened == []
    finally:
        await cleanup(fleet)


async def test_a_second_handoff_refuses_and_never_opens_a_second_terminal(tmp_path, monkeypatch):
    """DETACHED is a one-way door: the terminal already owns the session, so a
    second press must not hand the same session to a second window."""
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    assert (await fleet.handoff(path))["ok"] is True
    result = await fleet.handoff(path)
    assert result["ok"] is False and "command" not in result
    assert "already detached" in result["spoken"]
    assert len(opened) == 1                           # exactly one window


async def test_the_handoff_event_carries_the_command_for_the_console(tmp_path, monkeypatch):
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    cid, q = bus.subscribe()
    result = await fleet.handoff(path)
    events = [ev["data"] for ev in _drain(q) if ev["type"] == "fleet.handoff"]
    bus.unsubscribe(cid)
    w = fleet.workers[0]
    assert events == [{"worker": w.id, "project": "soccer", "path": path,
                       "command": result["command"]}]


async def test_a_failed_handoff_publishes_no_handoff_event(tmp_path, monkeypatch):
    """The console prints the resume command from this event — publishing it
    for a half-dead session would put a second driver in Keke's hands."""
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    async def boom():
        raise RuntimeError("subprocess refused to die")
    clients[0].disconnect = boom
    cid, q = bus.subscribe()
    await fleet.handoff(path)
    types = [ev["type"] for ev in _drain(q)]
    bus.unsubscribe(cid)
    assert "fleet.handoff" not in types


async def test_a_stopped_worker_is_not_sent_to_a_terminal_that_does_not_exist(tmp_path, monkeypatch):
    """`locked` is set by every shutdown — stop, close_all, a FAILED handoff.
    Only a real DETACHED worker has a terminal to be steered from."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    try:
        await fleet.stop(path)
        spoken = fleet.steer_path(path, "do more")
        assert "terminal" not in spoken.lower()
        assert "detached" not in spoken.lower()
        assert "soccer" in spoken
    finally:
        await cleanup(fleet)


async def test_a_tick_during_the_lockout_cannot_shout_a_false_probe_failure(tmp_path, monkeypatch):
    """The 5s ticker lands inside the handoff's awaits. The consumer it probes
    is dead BY OUR OWN HAND, so escalating it would alarm the console about a
    teardown we are performing on purpose — and drag the worker to UNKNOWN
    mid-sequence."""
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    w = fleet.workers[0]
    w._apply("activity")                              # a long-running turn…
    w.machine.last_event -= QUIET_AFTER_S + 1         # …silent long enough to be QUIET
    real_disconnect = clients[0].disconnect

    async def slow_disconnect():
        await asyncio.sleep(0.01)                     # cancelled tasks really end
        await fleet.tick()                            # the ticker lands mid-lockout
        await real_disconnect()

    clients[0].disconnect = slow_disconnect
    cid, q = bus.subscribe()
    result = await fleet.handoff(path)
    errors = [ev["data"].get("reason") for ev in _drain(q)
              if ev["type"] == "fleet.error"]
    bus.unsubscribe(cid)
    assert result["ok"] is True
    assert w.machine.base == DETACHED
    assert errors == []


async def test_two_clicks_hand_off_once_and_open_one_terminal(tmp_path, monkeypatch):
    """The button has no debounce and the lockout is not instant. Both calls
    pass the DETACHED gate before either sets it — and two windows running
    `claude --resume` on one session IS the two-drivers accident."""
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    real_disconnect = clients[0].disconnect

    async def slow_disconnect():
        await asyncio.sleep(0.05)                     # closing takes a moment
        await real_disconnect()

    clients[0].disconnect = slow_disconnect
    first, second = await asyncio.gather(fleet.handoff(path),
                                         fleet.handoff(path))
    done = [r for r in (first, second) if r["ok"]]
    refused = [r for r in (first, second) if not r["ok"]]
    assert len(done) == 1 and len(refused) == 1
    assert "command" not in refused[0]
    assert "already handing" in refused[0]["spoken"]
    assert opened == [done[0]["command"]]             # ONE window, one session
    assert fleet.workers[0].machine.base == DETACHED


async def test_a_stop_during_the_lockout_defers_instead_of_contradicting_it(tmp_path, monkeypatch):
    """"Stop soccer" spoken inside the lockout window would tear the session
    down under the handoff: session_end lands first, `detached` bounces off
    CLOSED, and both "Stopped soccer" and "soccer is yours in the terminal"
    get spoken about one worker."""
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    real_disconnect = clients[0].disconnect

    async def slow_disconnect():
        await asyncio.sleep(0.05)
        await real_disconnect()

    clients[0].disconnect = slow_disconnect
    handing = asyncio.create_task(fleet.handoff(path))
    await asyncio.sleep(0.01)                         # inside the lockout
    stopped = await fleet.stop(path)
    steered = fleet.steer_path(path, "also fix the tests")
    result = await asyncio.wait_for(handing, 2)
    assert "handing" in stopped.lower()               # deferred, not obeyed
    assert "handing" in steered.lower()               # not "spawn it again"
    assert result["ok"] is True
    assert fleet.workers[0].machine.base == DETACHED  # NOT closed
    assert "detached" in (await fleet.stop(path)).lower()


async def test_handoff_refuses_a_worker_whose_session_already_closed(tmp_path, monkeypatch):
    """Tiles are never removed, so a stopped worker still has a button. CLOSED
    is final — `detached` bounces off it — so proceeding would open a terminal
    and speak "yours in the terminal" over a tile that reads CLOSED."""
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    await fleet.stop(path)
    assert fleet.workers[0].machine.base == CLOSED
    result = await fleet.handoff(path)
    assert result["ok"] is False and "command" not in result
    assert "closed" in result["spoken"]
    assert opened == []


async def test_handoff_refuses_a_worker_that_is_still_starting(tmp_path, monkeypatch):
    """Early registration makes a half-started worker visible here, and the
    SessionStart hook can hand us a session id while _spawn is still awaiting
    start(). Detaching under it opens a terminal and then contradicts itself
    with a spawn-failure sentence."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    opened, seen = [], {}
    async def fake_terminal(cmd): opened.append(cmd)
    fleet._open_terminal = fake_terminal
    path = repo(tmp_path)
    inner = fleet._client_factory

    def factory(options):
        c = inner(options)
        real_connect = c.connect

        async def connect():
            await real_connect()
            # the CLI's SessionStart hook lands mid-connect, WITH an id
            fleet.handle_hook({"hook_event_name": "SessionStart",
                               "session_id": "sess-early",
                               "cwd": fleet.workers[0].worktree.path})
            seen.update(await fleet.handoff(path))

        c.connect = connect
        return c

    fleet._client_factory = factory
    try:
        spoken = await fleet.spawn("soccer", path, "task")
        assert seen["ok"] is False and "command" not in seen
        assert "still starting" in seen["spoken"]
        assert opened == []
        assert spoken.startswith("On it")             # the spawn was untouched
        assert fleet.workers[0].machine.base != DETACHED
    finally:
        await cleanup(fleet)


# ---------- the endpoint (cookie-authed console click) ----------
def test_the_handoff_endpoint_is_cookie_authed_and_speaks_the_result(tmp_path):
    c = make_client(tmp_path)
    calls, published = [], []

    async def fake_handoff(path):
        calls.append(path)
        return {"ok": True, "command": "cd /wt && claude --resume s-1",
                "spoken": "soccer is yours in the terminal, sir."}

    c.app.state.fleet.handoff = fake_handoff
    real_publish = c.app.state.bus.publish
    c.app.state.bus.publish = lambda t, d: (published.append((t, d)),
                                            real_publish(t, d))[1]
    body = {"path": "/p/soccer"}
    # cookie-only, exactly like /approval: the hook bearer lives in every
    # worktree, and a worker must never be able to hand itself a terminal.
    assert c.post("/handoff", json=body).status_code == 401
    hdrs = {"Authorization": f"Bearer {c.app.state.cfg.hook_bearer}"}
    assert c.post("/handoff", json=body, headers=hdrs).status_code == 401
    bootstrap(c)
    # malformed bodies are 400s, never 500s (parity with /hooks and /approval)
    assert c.post("/handoff", json={}).status_code == 400
    assert c.post("/handoff", content=b"{{{",
                  headers={"Content-Type": "application/json"}).status_code == 400
    assert c.post("/handoff", json=["not", "a", "dict"]).status_code == 400
    assert calls == []
    r = c.post("/handoff", json=body)
    assert r.status_code == 200 and r.json()["command"].endswith("s-1")
    assert calls == ["/p/soccer"]
    assert ("fleet.spoken",
            {"text": "soccer is yours in the terminal, sir."}) in published


def test_a_refused_handoff_still_speaks_from_the_endpoint(tmp_path):
    c = make_client(tmp_path)
    published = []

    async def fake_handoff(path):
        return {"ok": False, "spoken": "Nothing is running there, sir."}

    c.app.state.fleet.handoff = fake_handoff
    real_publish = c.app.state.bus.publish
    c.app.state.bus.publish = lambda t, d: (published.append((t, d)),
                                            real_publish(t, d))[1]
    bootstrap(c)
    r = c.post("/handoff", json={"path": "/p/nope"})
    assert r.status_code == 200
    assert "command" not in r.json()
    assert ("fleet.spoken",
            {"text": "Nothing is running there, sir."}) in published
