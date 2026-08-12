import asyncio
from types import SimpleNamespace

import pytest

from server import fleet as fleet_mod
from server.fleet_state import (CLOSED, DETACHED, QUIET_AFTER_S, UNKNOWN,
                                WAITING_PERMISSION)
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


# ---------- the three real interleavings ----------
async def test_a_stop_already_inside_its_interrupt_refuses_the_handoff(tmp_path, monkeypatch):
    """stop() defers to handoff_in_flight, but nothing was symmetric: stop
    suspends inside w.shutdown's `await interrupt()` BEFORE it applies
    session_end, and it runs on the brain task while POST /handoff runs on the
    HTTP task. A click landing in that multi-second window passed every handoff
    gate — base still live, handoff_in_flight False, session id set — and then
    either ordering lies: `detached` bounces off the CLOSED the stop lands, or
    the stop collapses the freshly DETACHED worker back to CLOSED."""
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)

    async def slow_interrupt():
        await asyncio.sleep(0.05)             # the CLI takes a moment to interrupt
        clients[0].interrupted = True

    clients[0].interrupt = slow_interrupt
    stopping = asyncio.create_task(fleet.stop(path))
    await asyncio.sleep(0.01)                 # suspended inside the interrupt wait
    result = await fleet.handoff(path)
    stopped = await asyncio.wait_for(stopping, 2)

    assert result["ok"] is False and "command" not in result
    assert "stopping" in result["spoken"].lower()   # not "yours in the terminal"
    assert opened == []                       # no window on a session being closed
    assert "Stopped soccer" in stopped        # the stop was never contradicted
    assert fleet.workers[0].machine.base == CLOSED   # ONE outcome, and the tile says it


async def test_a_session_end_between_the_close_and_the_detached_apply_detaches(tmp_path, monkeypatch):
    """The handoff's own disconnect() kills the CLI, and the CLI's SessionEnd
    hook POSTs while the lockout is still inside that disconnect. The first
    shipped version read the resulting CLOSED as a failure and refused — 3 live
    attempts out of 3 — but that CLOSED is the teardown WORKING: closing the
    session on our side is a step of the handoff, and the transcript it leaves
    on disk is exactly what `claude --resume` resumes. The lockout completed
    and was verified, so the handoff must detach and publish the command. (A
    session_end on the back of a teardown that FAILED still refuses — see
    test_a_failed_teardown_never_detaches_even_when_the_session_closed.)"""
    fleet, bus, router, clients, path, opened = await spawned(tmp_path, monkeypatch)
    w = fleet.workers[0]
    real_disconnect = clients[0].disconnect

    async def slow_disconnect():
        await asyncio.sleep(0.05)             # closing the subprocess takes a moment
        await real_disconnect()

    clients[0].disconnect = slow_disconnect
    cid, q = bus.subscribe()
    handing = asyncio.create_task(fleet.handoff(path))
    await asyncio.sleep(0.01)                 # inside the lockout, mid-disconnect
    fleet.handle_hook({"hook_event_name": "SessionEnd",
                       "session_id": "sess-42", "cwd": w.worktree.path})
    result = await asyncio.wait_for(handing, 2)
    types = [ev["type"] for ev in _drain(q)]
    bus.unsubscribe(cid)

    assert result["ok"] is True
    assert w.machine.base == DETACHED         # zero drivers became exactly one
    assert opened == [result["command"]]
    assert "claude --resume sess-42" in result["command"]
    assert "fleet.handoff" in types           # the console gets the command
    # The teardown's session_end was consumed pre-detach, so no echo is owed:
    # the NEXT SessionEnd is the terminal session's own end and must close.
    fleet.handle_hook({"hook_event_name": "SessionEnd",
                       "session_id": "sess-42", "cwd": w.worktree.path})
    assert w.machine.base == CLOSED


async def test_a_handoff_in_the_query_window_refuses_for_the_whole_spawn(tmp_path, monkeypatch):
    """`Worker.starting` is False for the SECOND half of the spawn: start()
    sets consumer and pump and THEN awaits client.query(task_text). The tile and
    its button already exist (_apply("spawned") published one line earlier) and
    the SessionStart hook may already have handed us a session id — so a click
    there cleared the starting gate, detached, opened a terminal, and _spawn
    then resumed to speak a stop-or-failure sentence contradicting it."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    opened = []

    async def fake_terminal(cmd):
        opened.append(cmd)

    fleet._open_terminal = fake_terminal
    path = repo(tmp_path)
    inner = fleet._client_factory

    def factory(options):
        c = inner(options)
        real_query = c.query

        async def query(text):
            if not c.queries:                 # the spawn's own opening task text
                await asyncio.sleep(0.05)     # the CLI is still taking it
            await real_query(text)

        c.query = query
        return c

    fleet._client_factory = factory
    try:
        spawning = asyncio.create_task(fleet.spawn("soccer", path, "task"))
        await asyncio.sleep(0.02)             # consumer + pump exist; query in flight
        w = fleet.workers[0]
        fleet.handle_hook({"hook_event_name": "SessionStart",
                           "session_id": "sess-early",
                           "cwd": w.worktree.path})
        assert w.consumer is not None and not w.starting   # the gap the review named
        result = await fleet.handoff(path)
        spoken = await asyncio.wait_for(spawning, 5)

        assert result["ok"] is False and "command" not in result
        assert "still starting" in result["spoken"]
        assert opened == []
        assert spoken.startswith("On it")     # the spawn was untouched
        assert w.machine.base != DETACHED
    finally:
        await cleanup(fleet)


async def test_steer_during_the_post_handoff_queue_drain_says_detached(tmp_path, monkeypatch):
    """handoff_in_flight used to be cleared by a finally that runs AFTER
    _admit_next() — i.e. after the next queued worker's whole spawn, bounded by
    SPAWN_TIMEOUT_S. For that window steer() (which checks the flag BEFORE the
    DETACHED branch) said "I'm handing soccer over right now" about a worker
    already detached with its terminal open — false, and it invites a retry
    that is then refused with a different sentence."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    opened = []

    async def fake_terminal(cmd):
        opened.append(cmd)

    fleet._open_terminal = fake_terminal
    inner = fleet._client_factory

    def factory(options):
        c = inner(options)
        if len(clients) > 1:                  # the QUEUED worker's client
            real_connect = c.connect

            async def connect():
                await asyncio.sleep(0.2)      # its spawn takes a while
                await real_connect()

            c.connect = connect
        return c

    fleet._client_factory = factory
    path = repo(tmp_path)
    try:
        await fleet.spawn("soccer", path, "task")
        clients[0].stream.put_nowait(ResultMessage(session_id="sess-42"))
        await asyncio.sleep(0.05)             # session id captured
        await fleet.spawn("alethic", repo(tmp_path, "alethic"), "other task")
        assert len(fleet.queue) == 1          # max_workers=1: it queued

        handing = asyncio.create_task(fleet.handoff(path))
        await asyncio.sleep(0.05)             # inside _admit_next's slow spawn
        steered = fleet.steer_path(path, "do more")
        result = await asyncio.wait_for(handing, 5)

        assert result["ok"] is True and opened == [result["command"]]
        assert "detached" in steered.lower()  # the truth, from the moment it was
        assert "handing" not in steered.lower()
    finally:
        await cleanup(fleet)


# ---------- the fake that behaves like the REAL CLI ----------
class RealCLIClient(FakeClient):
    """FakeClient plus the two behaviours of the real CLI that the live run
    proved and every older fake omitted — the omission that let the shipped
    handoff pass ~30 unit tests and then fail 3 attempts out of 3 against
    real workers (state/fleet.jsonl seq 55-60):

      1. a PermissionResultDeny is terminal: the real CLI exits the session on
         a denial — globe's log shows `session_end` 19ms after the teardown's
         own rejection (seq 56 -> 57), dei's 24ms (seq 76 -> 77);
      2. the exiting CLI runs its SessionEnd hook, whose curl POSTs to /hooks
         and is handled BEFORE the SDK disconnect returns — so `session_end`
         reaches the state machine while the handoff is still inside its own
         shutdown, in all three live attempts.

    `hooks` is wired to fleet.handle_hook by the test: it stands in for the
    localhost HTTP hop, which runs on the same event loop in production too.
    """

    def __init__(self, options):
        super().__init__(options)
        self.hooks = None                    # fleet.handle_hook, wired later
        self.cli_session_id = "sess-42"
        self.exited = False

    def exit_session(self):
        """The CLI process ends: SessionEnd hook POST, then the stream ends."""
        if self.exited:
            return
        self.exited = True
        if self.hooks is not None:
            self.hooks({"hook_event_name": "SessionEnd",
                        "session_id": self.cli_session_id,
                        "cwd": self.options.cwd})
        self.stream.put_nowait(None)

    def exit_on_deny(self, decision_task):
        """Arm behaviour 1 on a pending can_use_tool decision."""
        def on_done(task):
            if task.cancelled() or task.exception() is not None:
                return
            if getattr(task.result(), "behavior", "") == "deny":
                self.exit_session()
        decision_task.add_done_callback(on_done)

    async def disconnect(self):
        self.exit_session()                  # behaviour 2
        await super().disconnect()


async def spawned_real_cli(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch,
                                             client_cls=RealCLIClient)
    opened = []

    async def fake_terminal(cmd):
        opened.append(cmd)

    fleet._open_terminal = fake_terminal
    path = repo(tmp_path)
    await fleet.spawn("globe", path, "task")
    clients[0].hooks = fleet.handle_hook
    clients[0].stream.put_nowait(ResultMessage(session_id="sess-42"))
    await asyncio.sleep(0.05)                     # session id captured
    return fleet, bus, router, clients, path, opened


async def test_handoff_survives_the_cli_exiting_on_the_teardowns_own_denial(tmp_path, monkeypatch):
    """THE live defect, replayed: worker parked on a pending approval (where a
    real worker sits most of the time), handoff rejects it, the real CLI treats
    the denial as terminal and exits, its SessionEnd hook lands `session_end`
    mid-shutdown — and the handoff must still detach and hand over exactly one
    driver, because ending the session on our side is a STEP of the handoff,
    not its failure. Before the fix this failed 3 attempts out of 3."""
    fleet, bus, router, clients, path, opened = await spawned_real_cli(
        tmp_path, monkeypatch)
    w = fleet.workers[0]
    decision = asyncio.create_task(w._on_tool_request(
        "Bash", {"command": "npm test"}, SimpleNamespace(title=None)))
    await asyncio.sleep(0.05)
    assert w.machine.base == WAITING_PERMISSION
    clients[0].exit_on_deny(decision)

    result = await fleet.handoff(path)
    got = await asyncio.wait_for(decision, 1)

    assert got.behavior == "deny"                 # unblocked, not abandoned
    assert clients[0].exited                      # the CLI really did exit
    assert result["ok"] is True
    assert w.machine.base == DETACHED
    assert "claude --resume sess-42" in result["command"]
    assert opened == [result["command"]]          # ONE terminal, one driver
    assert router.pending_approvals() == []


async def test_handoff_survives_the_session_end_hook_of_its_own_disconnect(tmp_path, monkeypatch):
    """The probe attempt (seq 19-22): NO pending approval — the CLI exits
    under the teardown's interrupt/disconnect and its SessionEnd hook still
    beats `_apply("detached")`. Same defect, no denial involved."""
    fleet, bus, router, clients, path, opened = await spawned_real_cli(
        tmp_path, monkeypatch)
    w = fleet.workers[0]
    result = await fleet.handoff(path)
    assert clients[0].exited
    assert result["ok"] is True
    assert w.machine.base == DETACHED
    assert opened == [result["command"]]
    # The teardown's session_end was consumed BEFORE the detach, so the next
    # SessionEnd for this worker is the TERMINAL session's own end — the one
    # event that may close a DETACHED tile.
    fleet.handle_hook({"hook_event_name": "SessionEnd",
                       "session_id": "sess-42", "cwd": w.worktree.path})
    assert w.machine.base == CLOSED


async def test_the_teardowns_late_session_end_echo_does_not_close_the_detached_tile(tmp_path, monkeypatch):
    """The race's OTHER order: the CLI died without its hook completing (a
    hard kill mid-hook leaves the curl to finish on its own), so the
    teardown's own `session_end` lands only AFTER `detached` was applied.
    Un-swallowed it would collapse DETACHED -> CLOSED under an open terminal:
    the tile lies, the worktree drops out of live_worktree_paths' never-delete
    set, and the next boot folds the worker to CLOSED — no DETACHED ghost, no
    resume command. The echo is swallowed ONCE, keyed to the session id we
    handed over; the terminal session's own end still closes the tile."""
    fleet, bus, router, clients, path, opened = await spawned(
        tmp_path, monkeypatch)               # plain fake: no hook mid-shutdown
    w = fleet.workers[0]
    result = await fleet.handoff(path)
    assert result["ok"] is True and w.machine.base == DETACHED

    fleet.handle_hook({"hook_event_name": "SessionEnd",
                       "session_id": "sess-42", "cwd": w.worktree.path})
    assert w.machine.base == DETACHED             # our echo, not the terminal

    fleet.handle_hook({"hook_event_name": "SessionEnd",
                       "session_id": "sess-42", "cwd": w.worktree.path})
    assert w.machine.base == CLOSED               # the terminal really ended


async def test_a_failed_teardown_never_detaches_even_when_the_session_closed(tmp_path, monkeypatch):
    """CLOSED-mid-handoff is handoff-able ONLY on the back of a verified
    teardown. Here the hook proves the CLI exited but disconnect() itself
    raised — the lockout is unverified end-to-end, so the handoff must refuse,
    return no resume command, and speak the closed-session sentence."""
    fleet, bus, router, clients, path, opened = await spawned_real_cli(
        tmp_path, monkeypatch)
    w = fleet.workers[0]
    real_exit = clients[0].exit_session

    async def hook_then_boom():
        real_exit()                               # the CLI died and said so…
        raise RuntimeError("subprocess refused to die")   # …but we can't verify

    clients[0].disconnect = hook_then_boom
    result = await fleet.handoff(path)
    assert result["ok"] is False and "command" not in result
    assert w.machine.base == CLOSED               # honest: it really did end
    assert "closed" in result["spoken"].lower()
    assert opened == []                           # never a terminal unverified


async def test_the_teardowns_denial_speaks_for_marlowe_not_for_keke(tmp_path, monkeypatch):
    """The rejected approval's deny message lands in the session transcript
    the human is about to resume. "Keke denied this by voice" is false there —
    nobody denied anything; Marlowe was detaching — and the resumed model
    would act on it."""
    fleet, bus, router, clients, path, opened = await spawned(
        tmp_path, monkeypatch)
    w = fleet.workers[0]
    decision = asyncio.create_task(w._on_tool_request(
        "Bash", {"command": "npm test"}, SimpleNamespace(title=None)))
    await asyncio.sleep(0.05)
    result = await fleet.handoff(path)
    got = await asyncio.wait_for(decision, 1)
    assert result["ok"] is True
    assert got.behavior == "deny"
    assert "keke" not in got.message.lower()      # not a decision on the tool
    assert "denied" not in got.message.lower()


# ---------- the default launcher (never launches a real Terminal) ----------
class _FakeProc:
    """Stands in for the osascript child. `hangs` makes wait() report the
    deadline the real wait_for(…, 10) would raise, without spending 10s."""

    def __init__(self, *, hangs=False):
        self.returncode = None if hangs else 0
        self.hangs = hangs
        self.killed = False

    async def wait(self):
        if self.hangs and not self.killed:
            raise asyncio.TimeoutError
        self.returncode = -9 if self.hangs else 0
        return self.returncode

    def kill(self):
        self.killed = True


async def test_the_applescript_literal_keeps_a_non_ascii_path_verbatim(monkeypatch):
    """AppleScript does not decode \\uXXXX. With json.dumps' default
    ensure_ascii, one non-ASCII character anywhere in the worktree path turns
    the whole `cd` into a literal backslash-u string — and osascript STILL
    exits 0, so Marlowe says "yours in the terminal" over a shell that never
    left home."""
    seen = {}

    async def fake_exec(*argv, **kw):
        seen["argv"] = argv
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await fleet_mod._default_open_terminal(
        "cd /Users/keke/项目/wt && claude --resume s-1")
    script = seen["argv"][2]
    assert "项目" in script and "\\u" not in script


async def test_a_hung_osascript_is_killed_instead_of_orphaned(monkeypatch):
    """wait_for cancels the WAIT, never the child: a wedged osascript (a modal
    dialog, a stuck Terminal) would outlive the handoff and hold its window for
    the rest of the session."""
    proc = _FakeProc(hangs=True)

    async def fake_exec(*argv, **kw):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(asyncio.TimeoutError):
        await fleet_mod._default_open_terminal("cd /wt && claude --resume s-1")
    assert proc.killed and proc.returncode == -9      # reaped, not left behind


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
