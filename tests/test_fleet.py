import asyncio
import json
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
from claude_agent_sdk import CanUseToolShadowedWarning

from server.bus import EventBus
from server.fleet import (APPROVAL_WAIT_S, Fleet, Worker, _FleetLogWriter,
                          _full_args, _named_paths, _risk_note, _short_args)
from server.fleet_log import FleetLog
from server.fleet_state import (ACTIVE_TURN, CLOSED, IDLE_AT_PROMPT, UNKNOWN,
                                WAITING_PERMISSION)
from server.router import Router
from server.worktrees import Worktree


# ---------- fakes (same class-name probe the Butler uses) ----------
class SystemMessage:
    def __init__(self, session_id=None):
        self.data = {"session_id": session_id} if session_id else {}
        self.content = []


class AssistantMessage:
    def __init__(self, text):
        self.content = [SimpleNamespace(text=text, name=None)]


class ResultMessage:
    def __init__(self, session_id="sess-1"):
        self.session_id = session_id
        self.content = []


class FakeClient:
    def __init__(self, options):
        self.options = options
        self.queries: list[str] = []
        self.connected = False
        self.disconnected = False
        self.interrupted = False
        self.stream: asyncio.Queue = asyncio.Queue()

    async def connect(self): self.connected = True
    async def query(self, text): self.queries.append(text)
    async def interrupt(self): self.interrupted = True
    async def disconnect(self): self.disconnected = True

    async def receive_messages(self):
        while True:
            msg = await self.stream.get()
            if msg is None:
                return
            if isinstance(msg, Exception):
                raise msg
            yield msg


def make_fleet(tmp_path, monkeypatch, max_workers=1, forbidden=None,
               client_cls=FakeClient):
    monkeypatch.setenv("MARVIN_SKIP_PROXY_CHECK", "1")
    bus, router = EventBus(), Router()
    clients: list[FakeClient] = []

    def factory(options):
        c = client_cls(options)
        clients.append(c)
        return c

    async def fake_worktree(repo, task, wtdir):
        dest = Path(wtdir) / f"wt-{len(clients)}"
        dest.mkdir(parents=True, exist_ok=True)
        return Worktree(repo=str(repo), path=str(dest), branch="marvin/test-x",
                        base_commit="abc1234def5678")

    fleet = Fleet(bus=bus, router=router,
                  log=FleetLog(tmp_path / "state" / "fleet.jsonl"),
                  worktrees_dir=tmp_path / "state" / "worktrees",
                  forbidden=forbidden or (str(tmp_path / "vault"),
                                          str(tmp_path / "marvin")),
                  max_workers=max_workers, client_factory=factory,
                  worktree_factory=fake_worktree,
                  settings_writer=lambda p, port, bearer: Path(p) / "unused",
                  hook_port=7777, hook_bearer="tok")
    return fleet, bus, router, clients


async def cleanup(fleet):
    for w in fleet.workers:
        for t in (w.consumer, w.pump):
            if t is not None:
                t.cancel()
    await asyncio.sleep(0)


def repo(tmp_path, name="soccer"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    return str(d)


# ---------- spawn ----------
async def test_spawn_speaks_on_it_only_after_the_worker_exists(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    spoken = await fleet.spawn("soccer", repo(tmp_path), "fix the login redirect")
    try:
        # Spec §3: "On it." is reserved for a successfully created worker.
        assert spoken.startswith("On it")
        assert "abc1234" in spoken                      # the recorded commit, spoken
        assert clients[0].connected is True
        assert clients[0].queries == ["fix the login redirect"]
        assert fleet.workers[0].machine.base == ACTIVE_TURN
        opts = clients[0].options
        assert opts.permission_mode == "default"        # NOT acceptEdits — spec §5
        assert opts.can_use_tool is not None
        assert opts.cwd == fleet.workers[0].worktree.path
    finally:
        await cleanup(fleet)


async def test_spawn_refuses_the_vault_and_the_marvin_repo(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    (tmp_path / "vault").mkdir()
    (tmp_path / "marvin").mkdir()
    for bad in (str(tmp_path / "vault"), str(tmp_path / "marvin")):
        spoken = await fleet.spawn("thing", bad, "do work")
        assert "don't run workers" in spoken
    assert fleet.workers == [] and clients == []


async def test_spawn_refuses_without_proxy_vars(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    monkeypatch.delenv("MARVIN_SKIP_PROXY_CHECK", raising=False)
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    spoken = await fleet.spawn("soccer", repo(tmp_path), "task")
    assert "HTTPS_PROXY" in spoken and fleet.workers == []


async def test_admission_queues_the_second_spawn(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path, "soccer"), "task one")
    spoken = await fleet.spawn("alethic", repo(tmp_path, "alethic"), "task two")
    try:
        assert "queued" in spoken.lower()
        assert len(fleet.workers) == 1 and len(fleet.queue) == 1
    finally:
        await cleanup(fleet)


async def test_stop_frees_the_slot_and_dequeues(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path1 = repo(tmp_path, "soccer")
    await fleet.spawn("soccer", path1, "task one")
    await fleet.spawn("alethic", repo(tmp_path, "alethic"), "task two")
    spoken = await fleet.stop(path1)
    try:
        assert "Stopped soccer" in spoken
        assert clients[0].interrupted and clients[0].disconnected
        assert fleet.workers[0].machine.base == CLOSED
        assert len(fleet.queue) == 0                          # drained
        assert any(w.project == "alethic" for w in fleet.workers)  # admitted
    finally:
        await cleanup(fleet)


# ---------- steer ----------
async def test_steer_feeds_the_live_session(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    try:
        spoken = fleet.steer_path(path, "also fix the tests")
        assert "Told soccer" in spoken
        await asyncio.sleep(0.05)                     # let the pump run
        assert clients[0].queries == ["task", "also fix the tests"]
        assert fleet.workers[0].machine.base == ACTIVE_TURN
    finally:
        await cleanup(fleet)


async def test_steer_refuses_a_full_backlog(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    try:
        fleet.workers[0].pump.cancel()                # nothing drains the inbox
        await asyncio.sleep(0)
        for i in range(4):                            # INPUT_QUEUE_MAX
            assert "Told soccer" in fleet.workers[0].steer(f"m{i}")
        assert "backlog" in fleet.workers[0].steer("m4")
    finally:
        await cleanup(fleet)


# ---------- approvals (can_use_tool) ----------
async def test_a_tool_request_opens_a_path_keyed_approval_and_waits(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    w = fleet.workers[0]
    cid, q = bus.subscribe()
    decision = asyncio.create_task(w._on_tool_request(
        "Bash", {"command": "npm test"}, SimpleNamespace(title=None)))
    try:
        await asyncio.sleep(0.05)
        assert w.machine.base == WAITING_PERMISSION
        pending = router.pending_approvals()
        assert len(pending) == 1 and pending[0].path == path      # PATH-keyed
        card = None
        while not q.empty():
            ev = q.get_nowait()
            if ev and ev["type"] == "approval.request":
                card = ev["data"]
        assert card is not None and "npm test" in card["question"]
        assert fleet.deliver_approval(card["nonce"], True) is True
        result = await asyncio.wait_for(decision, 1)
        assert result.behavior == "allow"
        assert w.machine.base == ACTIVE_TURN
        assert router.pending_approvals() == []       # consumed at delivery
    finally:
        await cleanup(fleet)


async def test_deny_resolves_the_tool_request_with_a_denial(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    w = fleet.workers[0]
    decision = asyncio.create_task(w._on_tool_request(
        "Bash", {"command": "rm -rf build"}, SimpleNamespace(title=None)))
    try:
        await asyncio.sleep(0.05)
        nonce = router.pending_approvals()[0].nonce
        assert fleet.deliver_approval(nonce, False) is True
        result = await asyncio.wait_for(decision, 1)
        assert result.behavior == "deny" and "denied" in result.message
    finally:
        await cleanup(fleet)


async def test_an_unanswered_approval_times_out_denied(tmp_path, monkeypatch):
    monkeypatch.setattr("server.fleet.APPROVAL_WAIT_S", 0.05)
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    w = fleet.workers[0]
    cid, q = bus.subscribe()
    try:
        result = await asyncio.wait_for(
            w._on_tool_request("Bash", {"command": "npm test"},
                               SimpleNamespace(title=None)), 2)
        assert result.behavior == "deny"
        assert router.pending_approvals() == []       # swept, not left dangling
        outcomes = []
        while not q.empty():
            ev = q.get_nowait()
            if ev and ev["type"] == "approval.resolved":
                outcomes.append(ev["data"]["outcome"])
        assert "expired" in outcomes
    finally:
        await cleanup(fleet)


# ---------- state through the stream and the hooks ----------
async def test_result_message_is_idle_at_prompt_not_closed(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    try:
        clients[0].stream.put_nowait(SystemMessage(session_id="sess-9"))
        clients[0].stream.put_nowait(AssistantMessage("done with that"))
        clients[0].stream.put_nowait(ResultMessage(session_id="sess-9"))
        await asyncio.sleep(0.05)
        assert w.machine.base == IDLE_AT_PROMPT       # NOT closed — the §5 trap
        assert w.session_id == "sess-9"
        assert list(w.transcript)[-1]["text"] == "done with that"
    finally:
        await cleanup(fleet)


async def test_a_stream_explosion_marks_unknown_and_survives(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    cid, q = bus.subscribe()
    try:
        clients[0].stream.put_nowait(RuntimeError("transport died"))
        await asyncio.sleep(0.05)
        assert w.machine.base == UNKNOWN
        reasons = []
        while not q.empty():
            ev = q.get_nowait()
            if ev and ev["type"] == "fleet.error":
                reasons.append(ev["data"]["reason"])
        assert any("stream died" in r for r in reasons)
    finally:
        await cleanup(fleet)


async def test_hooks_map_to_states_and_unknown_sessions_are_surfaced(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    cid, q = bus.subscribe()
    try:
        fleet.handle_hook({"hook_event_name": "Stop", "session_id": "s-77",
                           "cwd": w.worktree.path, "transcript_path": "/t.jsonl"})
        assert w.machine.base == IDLE_AT_PROMPT
        assert w.session_id == "s-77"                 # hooks can learn the id first
        fleet.handle_hook({"hook_event_name": "SessionEnd", "session_id": "s-77",
                           "cwd": w.worktree.path})
        assert w.machine.base == CLOSED
        fleet.handle_hook({"hook_event_name": "Stop", "session_id": "manual-1",
                           "cwd": "/somewhere/else"})
        unknown = []
        while not q.empty():
            ev = q.get_nowait()
            if ev and ev["type"] == "fleet.unknown_session":
                unknown.append(ev["data"])
        assert unknown and unknown[-1]["session_id"] == "manual-1"
    finally:
        await cleanup(fleet)


async def test_stop_rejects_pending_approvals_first(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    w = fleet.workers[0]
    decision = asyncio.create_task(w._on_tool_request(
        "Bash", {"command": "npm test"}, SimpleNamespace(title=None)))
    await asyncio.sleep(0.05)
    assert len(router.pending_approvals()) == 1
    await fleet.stop(path)
    result = await asyncio.wait_for(decision, 1)
    assert result.behavior == "deny"                  # the callback was unblocked
    assert router.pending_approvals() == []           # and the router swept
    assert w.machine.base == CLOSED


async def test_stopping_a_worker_publishes_cancelled_for_its_pending_approval(tmp_path, monkeypatch):
    """shutdown() rejects the pending future and sweeps the nonce, but only an
    approval.resolved event removes the console card — without the publish,
    "stop soccer" leaves the card on screen until a manual click 404s it or
    the page reloads."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    w = fleet.workers[0]
    decision = asyncio.create_task(w._on_tool_request(
        "Bash", {"command": "npm test"}, SimpleNamespace(title=None)))
    await asyncio.sleep(0.05)
    nonce = router.pending_approvals()[0].nonce
    cid, q = bus.subscribe()
    await fleet.stop(path)
    await asyncio.wait_for(decision, 1)
    resolved = []
    while not q.empty():
        ev = q.get_nowait()
        if ev and ev["type"] == "approval.resolved":
            resolved.append(ev["data"])
    bus.unsubscribe(cid)
    assert [r["outcome"] for r in resolved] == ["cancelled"]  # exactly once
    assert resolved[0]["nonce"] == nonce                      # the card's key
    assert resolved[0]["project"] == "soccer"
    assert "npm test" in resolved[0]["tool"]


# ---------- regression: the parked-forever WAITING_PERMISSION state ----------
async def test_notification_after_permission_done_live_consumer(tmp_path, monkeypatch):
    """A stale Notification POST landing AFTER can_use_tool resolved must not
    park the tile on WAITING_PERMISSION (which never decays)."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    decision = asyncio.create_task(w._on_tool_request(
        "Bash", {"command": "npm test"}, SimpleNamespace(title=None)))
    try:
        await asyncio.sleep(0.05)
        nonce = router.pending_approvals()[0].nonce
        assert fleet.deliver_approval(nonce, True) is True
        await asyncio.wait_for(decision, 1)
        assert w.machine.base == ACTIVE_TURN          # permission_done applied
        fleet.handle_hook({"hook_event_name": "Notification",
                           "cwd": w.worktree.path})   # the slow POST lands now
        assert w.machine.base == ACTIVE_TURN          # NOT parked
    finally:
        await cleanup(fleet)


async def test_notification_with_dead_consumer_cannot_park_waiting(tmp_path, monkeypatch):
    """Dead stream, live CLI: the CLI hits a real permission prompt and POSTs
    Notification — with no router card, no future, and no TTL. Applying it
    would park WAITING_PERMISSION forever; it must be dropped."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    try:
        clients[0].stream.put_nowait(RuntimeError("transport died"))
        await asyncio.sleep(0.05)
        assert w.machine.base == UNKNOWN and w.consumer.done()
        fleet.handle_hook({"hook_event_name": "Notification",
                           "cwd": w.worktree.path})
        assert w.machine.base == UNKNOWN              # NEVER WAITING_PERMISSION
    finally:
        await cleanup(fleet)


async def test_tick_rescues_a_parked_permission_wait(tmp_path, monkeypatch):
    """Defense in depth: if WAITING_PERMISSION is ever reached with a dead
    consumer and no pending approval future, nothing can deliver
    permission_done — tick's probe must reclassify it, not skip it."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    try:
        clients[0].stream.put_nowait(RuntimeError("transport died"))
        await asyncio.sleep(0.05)
        w._apply("permission_wait")       # the leak, however it happened
        assert w.machine.base == WAITING_PERMISSION
        assert w.consumer.done() and not w._futures
        await fleet.tick()
        assert w.machine.base == UNKNOWN  # rescued — a stop can now clear it
    finally:
        await cleanup(fleet)


async def test_tick_leaves_a_real_permission_wait_alone(tmp_path, monkeypatch):
    """A pending approval future has a TTL that WILL deliver permission_done —
    tick must not reclassify that wait even when the consumer is dead."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    decision = asyncio.create_task(w._on_tool_request(
        "Bash", {"command": "npm test"}, SimpleNamespace(title=None)))
    try:
        await asyncio.sleep(0.05)
        w.consumer.cancel()
        await asyncio.sleep(0)
        await fleet.tick()
        assert w.machine.base == WAITING_PERMISSION   # the TTL owns this one
        nonce = router.pending_approvals()[0].nonce
        fleet.deliver_approval(nonce, True)
        await asyncio.wait_for(decision, 1)
    finally:
        await cleanup(fleet)


# ---------- regression: atomic admission (max_workers=1 under concurrency) ----------
async def test_two_concurrent_spawns_admit_exactly_one(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    inner = fleet._worktree_factory

    async def slow_factory(repo_path, task, wtdir):
        await asyncio.sleep(0.01)     # a REAL suspension, like the git subprocess
        return await inner(repo_path, task, wtdir)

    fleet._worktree_factory = slow_factory
    s1, s2 = await asyncio.gather(
        fleet.spawn("soccer", repo(tmp_path, "soccer"), "task one"),
        fleet.spawn("alethic", repo(tmp_path, "alethic"), "task two"))
    try:
        assert len(fleet.workers) == 1                # EXACTLY one admitted
        assert len(fleet.queue) == 1                  # the other queued
        assert sum("On it" in s for s in (s1, s2)) == 1
        assert sum("queued" in s.lower() for s in (s1, s2)) == 1
    finally:
        await cleanup(fleet)


# ---------- regression: a cleanly-ended stream is not silent task death ----------
async def test_a_cleanly_ended_stream_is_marked_lost(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    cid, q = bus.subscribe()
    try:
        clients[0].stream.put_nowait(None)            # generator returns cleanly
        await asyncio.sleep(0.05)
        assert w.machine.base == UNKNOWN              # not parked, not "live"
        reasons = []
        while not q.empty():
            ev = q.get_nowait()
            if ev and ev["type"] == "fleet.error":
                reasons.append(ev["data"]["reason"])
        assert any("stream ended" in r for r in reasons)
    finally:
        await cleanup(fleet)


async def test_a_stream_ending_after_session_end_stays_closed_quietly(tmp_path, monkeypatch):
    """CLOSED is final: when the SessionEnd hook already landed, the stream
    ending is the EXPECTED end — no lost, no error noise."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    cid, q = bus.subscribe()
    try:
        fleet.handle_hook({"hook_event_name": "SessionEnd",
                           "cwd": w.worktree.path})
        assert w.machine.base == CLOSED
        clients[0].stream.put_nowait(None)
        await asyncio.sleep(0.05)
        assert w.machine.base == CLOSED
        while not q.empty():
            ev = q.get_nowait()
            assert not (ev and ev["type"] == "fleet.error")
    finally:
        await cleanup(fleet)


# ---------- regression: cancellation must not leak the router card ----------
async def test_cancelled_tool_request_sweeps_card_and_repairs_state(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    decision = asyncio.create_task(w._on_tool_request(
        "Bash", {"command": "npm test"}, SimpleNamespace(title=None)))
    try:
        await asyncio.sleep(0.05)
        assert w.machine.base == WAITING_PERMISSION
        assert len(router.pending_approvals()) == 1
        decision.cancel()                 # the SDK tearing down its callback
        with pytest.raises(asyncio.CancelledError):
            await decision                # never swallowed
        assert router.pending_approvals() == []       # card swept, no TTL wait
        assert w.machine.base != WAITING_PERMISSION   # state repaired
        assert w._futures == {}
    finally:
        await cleanup(fleet)


# ---------- regression: spawn plumbing the constraints demanded ----------
async def test_spawn_writes_the_bearer_shield(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    try:
        gi = Path(fleet.workers[0].worktree.path) / ".claude" / ".gitignore"
        lines = gi.read_text(encoding="utf-8").splitlines()
        # keeps .claude/settings.local.json (the bearer) out of `git add -A`,
        # and hides the shield itself
        assert "settings.local.json" in lines
        assert ".gitignore" in lines
    finally:
        await cleanup(fleet)


async def test_a_worktree_timeout_speaks_and_registers_nothing(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)

    async def timing_out(repo_path, task, wtdir):
        raise asyncio.TimeoutError("git hung")

    fleet._worktree_factory = timing_out
    spoken = await fleet.spawn("soccer", repo(tmp_path), "task")
    assert "timed out" in spoken
    assert fleet.workers == []
    assert fleet._pending_spawns == 0                 # the reservation released


# ---------- regression: a start() failure must deregister the early-registered worker ----------
async def test_a_start_failure_deregisters_and_frees_admission(tmp_path, monkeypatch):
    """The worker joins self.workers BEFORE start() (early registration, so a
    mid-connect SessionStart hook matches). If start() itself fails, the
    deregistration line is the only thing between a failed spawn and a phantom
    UNKNOWN worker that counts live forever and wedges admission at
    max_workers=1. The worktree-timeout test above fails BEFORE registration,
    so only this test covers that revert."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    inner = fleet._client_factory

    def exploding_factory(options):
        c = inner(options)

        async def connect():
            await asyncio.sleep(0)        # a real suspension INSIDE start()
            raise RuntimeError("CLI refused to boot")
        c.connect = connect
        return c

    fleet._client_factory = exploding_factory
    spoken = await fleet.spawn("soccer", repo(tmp_path), "task")
    assert "failed" in spoken.lower()     # the spoken failure, never a raise
    assert fleet.workers == []            # the phantom was deregistered
    assert fleet._pending_spawns == 0     # and the reservation released
    # Admission is NOT wedged: a healthy spawn is admitted, not queued.
    fleet._client_factory = inner
    spoken2 = await fleet.spawn("alethic", repo(tmp_path, "alethic"), "task two")
    try:
        assert spoken2.startswith("On it")
        assert len(fleet.workers) == 1 and len(fleet.queue) == 0
    finally:
        await cleanup(fleet)


# ---------- regression: stop() during connect() must not close a tile over a live CLI ----------
async def test_stop_during_connect_defers_and_never_leaves_a_zombie(tmp_path, monkeypatch):
    """Early registration makes a half-started worker visible to _find. A stop
    landing while start() still awaits connect() used to no-op shutdown, close
    the tile (CLOSED), speak "Stopped", and free the slot — while the in-flight
    start connected a real CLI behind the dead tile and a queued worker was
    admitted beside it. The stop must DEFER: refuse to tear down mid-start,
    then execute the moment start() returns, and only then admit the queue."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path, "soccer")
    gate = asyncio.Event()
    events: list[str] = []
    inner = fleet._client_factory

    def gated_factory(options):
        c = inner(options)
        idx = len(clients)
        orig_connect, orig_disconnect = c.connect, c.disconnect

        async def connect():
            if idx == 1:
                await gate.wait()         # a REAL mid-connect suspension
            events.append(f"connect-{idx}")
            await orig_connect()

        async def disconnect():
            events.append(f"disconnect-{idx}")
            await orig_disconnect()
        c.connect, c.disconnect = connect, disconnect
        return c

    fleet._client_factory = gated_factory
    spawn_task = asyncio.create_task(fleet.spawn("soccer", path, "task one"))
    try:
        for _ in range(100):              # let spawn register + park in connect
            if fleet.workers:
                break
            await asyncio.sleep(0)
        assert fleet.workers, "spawn never registered its worker"
        queued = await fleet.spawn("alethic", repo(tmp_path, "alethic"),
                                   "task two")
        assert "queued" in queued.lower()
        # Cosmetics while starting: status_line must not read "unknown", and
        # steer must not claim delivery into a pump that does not exist.
        assert "soccer is starting" in fleet.status_line()
        assert "still starting" in fleet.steer_path(path, "also lint")
        stop_spoken = await fleet.stop(path)
        assert "still starting" in stop_spoken        # deferred, not "Stopped"
        soccer = fleet.workers[0]
        assert soccer.machine.base != CLOSED          # tile NOT closed early
        assert len(fleet.queue) == 1                  # queue NOT drained early
        gate.set()                                    # connect finishes AFTER the stop
        spawn_spoken = await asyncio.wait_for(spawn_task, 5)
        # Spawn never speaks a false "On it" over a worker the user stopped.
        assert not spawn_spoken.startswith("On it")
        assert "stopped" in spawn_spoken.lower()
        assert soccer.machine.base == CLOSED
        assert clients[0].disconnected is True        # no zombie subprocess
        assert soccer.consumer.done() and soccer.pump.done()
        assert "also lint" not in clients[0].queries  # steer refused, not dropped
        # The queued worker was admitted only AFTER the stopped CLI was gone.
        assert events.index("disconnect-1") < events.index("connect-2")
        alethic = next(w for w in fleet.workers if w.project == "alethic")
        assert alethic.machine.base == ACTIVE_TURN
        assert len(fleet.live) == 1                   # exactly one real worker
    finally:
        gate.set()
        spawn_task.cancel()
        await asyncio.gather(spawn_task, return_exceptions=True)
        await cleanup(fleet)


async def test_stop_in_the_query_window_is_not_spoken_over_by_on_it(tmp_path, monkeypatch):
    """After connect() the consumer/pump exist, so a stop during the awaited
    query() takes the NORMAL teardown path — but start() then returns into
    _spawn, which must not announce "On it" over the already-CLOSED tile.

    DIVERGENCE, deliberate: FakeClient.query() succeeds on a disconnected
    client, whereas the real ClaudeSDKClient.query() after disconnect() raises.
    Production therefore usually exits this interleaving through _spawn's
    failure path — covered by the variant test two below, which pins that
    sentence. Both shapes are real (a stop can also land between query() and
    _apply("prompt")), and this one is the only one that reaches the belt, so
    it stays as the belt's revert detector."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path, "soccer")
    gate = asyncio.Event()
    inner = fleet._client_factory

    def gated_factory(options):
        c = inner(options)
        orig_query = c.query

        async def query(text):
            if not c.queries:             # first query: the spawn task text
                await gate.wait()         # a REAL suspension inside start()
            await orig_query(text)
        c.query = query
        return c

    fleet._client_factory = gated_factory
    spawn_task = asyncio.create_task(fleet.spawn("soccer", path, "task one"))
    try:
        for _ in range(100):              # park inside the gated query()
            if fleet.workers and fleet.workers[0].consumer is not None:
                break
            await asyncio.sleep(0)
        w = fleet.workers[0]
        assert w.consumer is not None     # past connect: the normal stop path
        stop_spoken = await fleet.stop(path)
        assert "Stopped soccer" in stop_spoken
        assert w.machine.base == CLOSED
        assert clients[0].disconnected is True
        gate.set()                                    # query() resumes now
        spawn_spoken = await asyncio.wait_for(spawn_task, 5)
        assert not spawn_spoken.startswith("On it")   # never a false success
        assert "stopped" in spawn_spoken.lower()
        assert w.machine.base == CLOSED
    finally:
        gate.set()
        spawn_task.cancel()
        await asyncio.gather(spawn_task, return_exceptions=True)
        await cleanup(fleet)


async def test_a_dirty_stop_in_the_query_window_is_not_spoken_over_by_on_it(tmp_path, monkeypatch):
    """The belt cannot key on CLOSED alone. When the stop's disconnect fails,
    _stop_worker applies "lost" — UNKNOWN, not CLOSED — and the resumed
    start() drives UNKNOWN → ACTIVE_TURN, so _spawn used to fall through to
    "On it, sir" one sentence after "I couldn't stop soccer cleanly", over a
    worker whose consumer and pump are already cancelled."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path, "soccer")
    gate = asyncio.Event()
    inner = fleet._client_factory

    def wedged_factory(options):
        c = inner(options)
        orig_query = c.query

        async def query(text):
            if not c.queries:             # first query: the spawn task text
                await gate.wait()         # a REAL suspension inside start()
            await orig_query(text)

        async def disconnect():           # the failure _stop_worker exists for
            c.disconnected = True
            raise RuntimeError("transport is wedged")
        c.query, c.disconnect = query, disconnect
        return c

    fleet._client_factory = wedged_factory
    spawn_task = asyncio.create_task(fleet.spawn("soccer", path, "task one"))
    try:
        for _ in range(100):              # park inside the gated query()
            if fleet.workers and fleet.workers[0].consumer is not None:
                break
            await asyncio.sleep(0)
        w = fleet.workers[0]
        assert w.consumer is not None     # past connect: the normal stop path
        queued = await fleet.spawn("alethic", repo(tmp_path, "alethic"),
                                   "task two")
        assert "queued" in queued.lower()
        stop_spoken = await fleet.stop(path)
        assert "couldn't stop soccer cleanly" in stop_spoken
        assert w.machine.base == UNKNOWN
        gate.set()                                    # query() resumes now
        spawn_spoken = await asyncio.wait_for(spawn_task, 5)
        assert not spawn_spoken.startswith("On it")   # no contradictory success
        assert "unknown" in spawn_spoken.lower()      # agrees with the stop
        assert w.machine.base == UNKNOWN              # and so does the tile
        # Fail closed: a client that would not disconnect may still hold a
        # subprocess, so UNKNOWN keeps counting live and the queue waits.
        assert len(fleet.live) == 1 and len(fleet.queue) == 1
    finally:
        gate.set()
        spawn_task.cancel()
        await asyncio.gather(spawn_task, return_exceptions=True)
        await cleanup(fleet)


async def test_a_stop_in_the_query_window_drains_the_queue(tmp_path, monkeypatch):
    """stop()'s own _admit_next is refused while _pending_spawns still counts
    the in-flight spawn, so the belt branch owns the drain. Without it a
    queued project waits forever behind ZERO live workers — and permanently,
    since tick() is not wired yet."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path, "soccer")
    gate = asyncio.Event()
    events: list[str] = []
    inner = fleet._client_factory

    def gated_factory(options):
        c = inner(options)
        idx = len(clients)
        orig_connect, orig_query, orig_disconnect = (c.connect, c.query,
                                                     c.disconnect)

        async def connect():
            events.append(f"connect-{idx}")
            await orig_connect()

        async def query(text):
            if idx == 1 and not c.queries:
                await gate.wait()         # a REAL suspension inside start()
            await orig_query(text)

        async def disconnect():
            events.append(f"disconnect-{idx}")
            await orig_disconnect()
        c.connect, c.query, c.disconnect = connect, query, disconnect
        return c

    fleet._client_factory = gated_factory
    spawn_task = asyncio.create_task(fleet.spawn("soccer", path, "task one"))
    try:
        for _ in range(100):              # park inside the gated query()
            if fleet.workers and fleet.workers[0].consumer is not None:
                break
            await asyncio.sleep(0)
        queued = await fleet.spawn("alethic", repo(tmp_path, "alethic"),
                                   "task two")
        assert "queued" in queued.lower()
        assert "Stopped soccer" in await fleet.stop(path)
        assert len(fleet.queue) == 1      # stop could not admit: spawn in flight
        gate.set()                                    # query() resumes now
        spawn_spoken = await asyncio.wait_for(spawn_task, 5)
        assert not spawn_spoken.startswith("On it")
        assert len(fleet.queue) == 0                  # the belt drained it
        alethic = next(w for w in fleet.workers if w.project == "alethic")
        assert alethic.machine.base == ACTIVE_TURN
        assert len(fleet.live) == 1                   # exactly one real worker
        # And only AFTER the stopped CLI was actually gone.
        assert events.index("disconnect-1") < events.index("connect-2")
    finally:
        gate.set()
        spawn_task.cancel()
        await asyncio.gather(spawn_task, return_exceptions=True)
        await cleanup(fleet)


async def test_a_failed_teardown_after_a_mid_start_stop_is_not_silent(tmp_path, monkeypatch):
    """Mid-start SessionEnd: the CLI's own hook closes the machine while the
    client is still connected. If the belt's teardown then fails, the fleet
    must not speak absolutely — and since the machine is already CLOSED,
    _apply("lost") bounces, so fleet.error is the ONLY place that failure can
    be recorded. It must also not admit the queue beside a client it could
    not prove dead."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path, "soccer")
    gate = asyncio.Event()
    inner = fleet._client_factory

    def wedged_factory(options):
        c = inner(options)
        orig_query = c.query

        async def query(text):
            if not c.queries:
                await gate.wait()         # a REAL suspension inside start()
            await orig_query(text)

        async def disconnect():
            c.disconnected = True
            raise RuntimeError("transport is wedged")
        c.query, c.disconnect = query, disconnect
        return c

    fleet._client_factory = wedged_factory
    spawn_task = asyncio.create_task(fleet.spawn("soccer", path, "task one"))
    cid, q = bus.subscribe()
    try:
        for _ in range(100):
            if fleet.workers and fleet.workers[0].consumer is not None:
                break
            await asyncio.sleep(0)
        w = fleet.workers[0]
        queued = await fleet.spawn("alethic", repo(tmp_path, "alethic"),
                                   "task two")
        assert "queued" in queued.lower()
        fleet.handle_hook({"hook_event_name": "SessionEnd",   # the CLI exited
                           "cwd": w.worktree.path})
        assert w.machine.base == CLOSED
        gate.set()
        spawn_spoken = await asyncio.wait_for(spawn_task, 5)
        assert not spawn_spoken.startswith("On it")
        assert "nothing is running" not in spawn_spoken       # never absolute
        reasons = []
        while not q.empty():
            ev = q.get_nowait()
            if ev and ev["type"] == "fleet.error":
                reasons.append(ev["data"]["reason"])
        assert any("teardown" in r and "wedged" in r for r in reasons)
        assert len(fleet.queue) == 1      # not admitted over a live client
    finally:
        gate.set()
        spawn_task.cancel()
        await asyncio.gather(spawn_task, return_exceptions=True)
        await cleanup(fleet)


async def test_stop_in_the_query_window_when_the_client_rejects_a_late_query(tmp_path, monkeypatch):
    """The production shape of the interleaving above: the real
    ClaudeSDKClient.query() raises once disconnect() has run, so start() dies
    and _spawn exits through its FAILURE path. What matters is the same —
    no false "On it", no zombie, no phantom worker left counting live."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path, "soccer")
    gate = asyncio.Event()
    inner = fleet._client_factory

    def realistic_factory(options):
        c = inner(options)
        orig_query = c.query

        async def query(text):
            if not c.queries:
                await gate.wait()         # a REAL suspension inside start()
            if c.disconnected:            # what the real SDK does here
                raise RuntimeError("client is not connected")
            await orig_query(text)
        c.query = query
        return c

    fleet._client_factory = realistic_factory
    spawn_task = asyncio.create_task(fleet.spawn("soccer", path, "task one"))
    try:
        for _ in range(100):
            if fleet.workers and fleet.workers[0].consumer is not None:
                break
            await asyncio.sleep(0)
        w = fleet.workers[0]
        assert "Stopped soccer" in await fleet.stop(path)
        assert w.machine.base == CLOSED
        gate.set()
        spawn_spoken = await asyncio.wait_for(spawn_task, 5)
        assert not spawn_spoken.startswith("On it")
        assert "failed" in spawn_spoken.lower()
        assert fleet.workers == []        # the phantom was deregistered
        assert fleet._pending_spawns == 0
        assert fleet.live == []
    finally:
        gate.set()
        spawn_task.cancel()
        await asyncio.gather(spawn_task, return_exceptions=True)
        await cleanup(fleet)


async def test_one_breath_says_starting_not_unknown_mid_start(tmp_path, monkeypatch):
    """status_line got this treatment; one_breath still read "soccer is
    unknown" — the exact alarm word the status_line change exists to avoid."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path, "soccer")
    gate = asyncio.Event()
    inner = fleet._client_factory

    def gated_factory(options):
        c = inner(options)
        orig_connect = c.connect

        async def connect():
            await gate.wait()             # a REAL mid-connect suspension
            await orig_connect()
        c.connect = connect
        return c

    fleet._client_factory = gated_factory
    spawn_task = asyncio.create_task(fleet.spawn("soccer", path, "task one"))
    try:
        for _ in range(100):              # register, then park in connect()
            if fleet.workers:
                break
            await asyncio.sleep(0)
        assert fleet.workers and fleet.workers[0].starting
        breath = fleet.one_breath(path)
        assert breath == "soccer is starting. Last activity: nothing yet."
        assert "unknown" not in breath.lower()
    finally:
        gate.set()
        spawn_task.cancel()
        await asyncio.gather(spawn_task, return_exceptions=True)
        await cleanup(fleet)


async def test_a_worker_registered_without_an_in_flight_spawn_is_not_starting(tmp_path, monkeypatch):
    """Task 10 registers pre-existing sessions found after a restart. Such a
    worker has no consumer, so inferring "starting" from `consumer is None`
    would make stop() a permanent no-op that still promises "I'll stop it the
    moment it's up", and steer() refuse forever. Only _spawn's own in-flight
    flag may mean starting."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path, "soccer")
    wt = Worktree(repo=path, path=str(tmp_path / "wt-recovered"),
                  branch="marvin/recovered", base_commit="abc1234def5678")
    w = Worker(project="soccer", path=path, task_text="task", wt=wt, bus=bus,
               router=router, log=fleet._log_writer,
               client_factory=fleet._client_factory, now=fleet._now)
    fleet.workers.append(w)                       # registered by another path
    try:
        assert w.consumer is None and w.spawn_in_flight is False
        assert w.starting is False
        assert "still starting" not in fleet.status_line()
        assert "still starting" not in fleet.steer_path(path, "also lint")
        stopped = await fleet.stop(path)
        assert "still starting" not in stopped    # a real stop, not a promise
        assert w.stop_requested is False          # nothing left to redeem
        assert w.machine.base == CLOSED
    finally:
        await cleanup(fleet)


async def test_session_start_during_connect_is_not_an_unknown_session(tmp_path, monkeypatch):
    """The CLI's SessionStart hook can POST while connect() is still awaited;
    the fleet's own spawn must never be published as fleet.unknown_session."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    inner = fleet._client_factory

    def hooking_factory(options):
        c = inner(options)
        orig_connect = c.connect

        async def connect():
            await orig_connect()
            fleet.handle_hook({"hook_event_name": "SessionStart",
                               "session_id": "boot-1", "cwd": options.cwd})
        c.connect = connect
        return c

    fleet._client_factory = hooking_factory
    cid, q = bus.subscribe()
    await fleet.spawn("soccer", repo(tmp_path), "task")
    try:
        unknown = []
        while not q.empty():
            ev = q.get_nowait()
            if ev and ev["type"] == "fleet.unknown_session":
                unknown.append(ev["data"])
        assert unknown == []                          # our own spawn, matched
        assert fleet.workers[0].session_id == "boot-1"
    finally:
        await cleanup(fleet)


async def test_connect_suppresses_the_shadowed_callback_warning(tmp_path, monkeypatch):
    """allowed_tools intentionally shadows can_use_tool for Read/Grep/Glob;
    the SDK's CanUseToolShadowedWarning must not surface on a real connect."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    inner = fleet._client_factory

    def warning_factory(options):
        c = inner(options)
        orig_connect = c.connect

        async def connect():
            warnings.warn(                    # what the real SDK does on connect
                "can_use_tool will not be invoked for: Read, Grep, Glob",
                CanUseToolShadowedWarning, stacklevel=2)
            await orig_connect()
        c.connect = connect
        return c

    fleet._client_factory = warning_factory
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await fleet.spawn("soccer", repo(tmp_path), "task")
    try:
        assert not [c for c in caught
                    if issubclass(c.category, CanUseToolShadowedWarning)]
    finally:
        await cleanup(fleet)


# ---------- regression: the durable log survives a whole life, in order ----------
async def test_spawn_steer_stop_leaves_an_untorn_ordered_log(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "fix the tests")
    assert "Told soccer" in fleet.steer_path(path, "also lint")
    await asyncio.sleep(0.05)                         # let the pump run
    await fleet.stop(path)
    await fleet.close_all()                           # flushes, then compacts
    # close_all compacts (Task 10): the snapshot rotates the whole life aside
    # to `.jsonl.1` and leaves a fresh, empty log behind. That rotated
    # generation IS the durable record now, so read it — this test exists to
    # prove the whole life is on disk, untorn and in order, not to pin which
    # file holds it.
    assert fleet._log.replay() == ([], False)         # the post-compaction tail
    rotated = fleet._log.path.with_suffix(".jsonl.1")
    # Read the rotated generation through a REAL FleetLog, never bare
    # json.loads: replay() verifies `v == SCHEMA_VERSION` and
    # `sum == _checksum(seq, ts, kind, data)` on EVERY record and stops at the
    # first bad one, and tamper detection is the entire point of the checksum —
    # a test named "untorn" that parses the file by hand would pass over a
    # record whose sum was rewritten. torn_on_open is asserted too because the
    # constructor REPAIRS a torn file, after which replay() honestly reports
    # the repaired remainder clean.
    verified = FleetLog(rotated)
    records, torn = verified.replay()
    assert verified.torn_on_open is False and torn is False
    assert fleet._log.load_snapshot() is not None     # ...and it is compacted
    seqs = [r["seq"] for r in records]
    assert seqs == list(range(1, len(seqs) + 1))      # untorn, gapless, ordered
    kinds = [r["kind"] for r in records]
    assert kinds[0] == "spawned"
    assert kinds.count("prompt") == 2                 # the task, then the steer
    assert kinds[-1] == "session_end"


async def test_a_worker_free_session_does_not_erase_the_last_one(tmp_path,
                                                                 monkeypatch):
    """Compaction keeps exactly ONE rotated generation, so an unconditional
    snapshot at every shutdown would let a Marvin session that never spawned
    anything rotate its own empty log over the previous session's history.
    Nothing to compact means no compaction."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "fix the tests")
    await fleet.stop(repo(tmp_path))
    await fleet.close_all()                           # session 1 compacts
    rotated = fleet._log.path.with_suffix(".jsonl.1")
    history = rotated.read_text(encoding="utf-8")
    assert "spawned" in history

    quiet, *_ = make_fleet(tmp_path, monkeypatch)     # session 2, same log path
    # Session 1's compaction RENAMED fleet.jsonl away, so without this line the
    # guard under test is never reached at all: FleetLog.snapshot already skips
    # the rotation when the log file is absent, and the test would pass with
    # Fleet.snapshot's size check deleted. A 0-byte log on disk is the case
    # that genuinely destroys the previous generation — an empty file rotates
    # perfectly happily — so put one there and pin the guard that stops it.
    quiet._log.path.write_bytes(b"")
    assert quiet._log.path.stat().st_size == 0
    await quiet.close_all()
    assert rotated.read_text(encoding="utf-8") == history


# ---------- the queue drain must not eat what it popped ----------
def _drain_events(q):
    out = []
    while not q.empty():
        ev = q.get_nowait()
        if ev:
            out.append(ev)
    return out


async def test_a_transient_refusal_puts_the_queued_item_back(tmp_path,
                                                             monkeypatch):
    """_admit_next POPS before it spawns, and _spawn's three pre-flight
    refusals (forbidden root, proxy_problem, empty task) return before the
    queue logic — so the popped project was silently dropped. A missing proxy
    is TRANSIENT: the item goes back on the FRONT, and nothing is announced,
    because it is still queued and the 5-second tick would otherwise repeat the
    same sentence forever."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path1 = repo(tmp_path, "soccer")
    await fleet.spawn("soccer", path1, "task one")
    await fleet.spawn("alethic", repo(tmp_path, "alethic"), "task two")
    assert len(fleet.queue) == 1
    cid, q = bus.subscribe()
    try:
        monkeypatch.delenv("MARVIN_SKIP_PROXY_CHECK", raising=False)
        for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            monkeypatch.delenv(var, raising=False)
        await fleet.stop(path1)                       # frees the slot, drains
        assert [item[0] for item in fleet.queue] == ["alethic"]   # NOT dropped
        spoken = [ev["data"]["text"] for ev in _drain_events(q)
                  if ev["type"] == "fleet.spoken"]
        assert spoken == []                           # still queued, no news
    finally:
        bus.unsubscribe(cid)
        await cleanup(fleet)


async def test_a_permanently_refused_queue_item_is_dropped_and_announced(
        tmp_path, monkeypatch):
    """The mirror: a refusal that can never succeed must NOT be requeued, or
    the poisoned head shadows the whole queue and re-refuses on every tick. It
    leaves the queue, and Keke is told — in one sentence that does not staple
    two vocatives together."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    (tmp_path / "vault").mkdir(exist_ok=True)
    fleet.queue.append(("notes", str(tmp_path / "vault"), "do work"))
    cid, q = bus.subscribe()
    try:
        await fleet._admit_next()
        assert list(fleet.queue) == []                # gone, not looping
        spoken = [ev["data"]["text"] for ev in _drain_events(q)
                  if ev["type"] == "fleet.spoken"]
        assert len(spoken) == 1
        assert "don't run workers in that directory" in spoken[0]
        assert spoken[0].count("sir") == 1            # not "…, sir: …, sir: …"
    finally:
        bus.unsubscribe(cid)
        await cleanup(fleet)


async def test_the_queue_announcement_reads_as_one_sentence(tmp_path,
                                                            monkeypatch):
    """"Next in the queue, sir: I couldn't prepare a worktree for X, sir: …"
    stapled two vocatives together. Success has to read cleanly too."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    path1 = repo(tmp_path, "soccer")
    await fleet.spawn("soccer", path1, "task one")
    await fleet.spawn("alethic", repo(tmp_path, "alethic"), "task two")
    cid, q = bus.subscribe()
    try:
        await fleet.stop(path1)
        spoken = [ev["data"]["text"] for ev in _drain_events(q)
                  if ev["type"] == "fleet.spoken"]
        assert len(spoken) == 1
        assert spoken[0].startswith("From the queue: On it, sir")
        assert spoken[0].count("sir") == 1
    finally:
        bus.unsubscribe(cid)
        await cleanup(fleet)


# ---------- durability: the flush and the compaction ----------
async def test_a_cancelled_drain_does_not_abort_the_flush(tmp_path):
    """contextlib.suppress(Exception) does NOT catch CancelledError, so one
    cancelled drain task raised straight out of flush() and took close_all()'s
    compaction with it — no snapshot, and no ghosts on the next boot."""
    log = FleetLog(tmp_path / "fleet.jsonl")
    writer = _FleetLogWriter(log, EventBus())
    writer.append("spawned", {"worker": "w1", "state": "IDLE_AT_PROMPT"})
    assert writer._task is not None
    writer._task.cancel()                             # the shutdown race
    await asyncio.wait_for(writer.flush(), 2)         # used to raise
    records, torn = log.replay()
    assert [r["kind"] for r in records] == ["spawned"]  # written by the retry
    assert torn is False


def test_the_snapshot_is_durable_before_the_log_rotates(tmp_path):
    """Two renames, one fsync. A crash between them left the log rotated aside
    with the new snapshot not on the platter — and recover() then reports
    nothing, with no torn flag, because both files are individually
    well-formed."""
    import server.fleet_log as fleet_log
    log = FleetLog(tmp_path / "fleet.jsonl")
    log.append("spawned", {"worker": "w1", "state": "IDLE_AT_PROMPT"})
    snap = log.path.with_suffix(".snap")
    rotated = log.path.with_suffix(".jsonl.1")
    seen = []
    real = fleet_log._fsync_dir
    fleet_log._fsync_dir = lambda p: seen.append((snap.exists(),
                                                  rotated.exists()))
    try:
        log.snapshot({"workers": {}})
    finally:
        fleet_log._fsync_dir = real
    assert seen[0] == (True, False)     # snapshot durable BEFORE the rotation
    assert seen[-1] == (True, True)     # and the rotation durable after it


# ---------- the spoken readback IS the containment (spec §5) ----------
def test_the_readback_elides_the_middle_and_keeps_a_commands_tail():
    """A trailing cut removes exactly the dangerous half of a shell line — and
    _risk_note scans the FULL blob, so Marvin would say "Careful, sir" and then
    read out a sentence with the thing it is warning about missing."""
    cmd = ("npm run build -- --verbose " + "--flag=value " * 20
           + "&& rm -rf /tmp/scratch-repo")
    spoken = _short_args({"command": cmd})
    assert _risk_note("Bash", {"command": cmd})        # the warning fires...
    assert "rm -rf /tmp/scratch-repo" in spoken        # ...and names its target
    assert spoken.startswith("npm run build")          # the head still identifies it
    assert "…" in spoken and len(spoken) < len(cmd)


def test_the_readback_keeps_a_paths_distinguishing_suffix():
    """Two worktrees for the same task differ ONLY in the `-<timestamp>` tail;
    the live smoke read both aloud as the same truncated prefix."""
    p = ("/Users/keke/marvin/state/worktrees/scratch-repo-create-a-file-named-"
         "done-txt-containing-exactly-the-word-done-20260808-141523/DONE.txt")
    spoken = _short_args({"file_path": p})
    assert spoken.endswith("20260808-141523/DONE.txt")
    assert spoken.startswith("/Users/keke/marvin")
    assert "…" in spoken and len(spoken) < len(p)


def test_a_short_argument_is_spoken_verbatim():
    assert _short_args({"command": "npm test"}) == "npm test"
    assert _short_args({"file_path": "/a/b.txt"}) == "/a/b.txt"


def test_the_full_command_survives_somewhere_even_when_speech_elides_it():
    """The elision cuts the MIDDLE, so a long enough command loses its
    destructive clause from the spoken sentence while _risk_note (which scans
    the FULL blob) still fires: "Careful, sir — this one can destroy things"
    over a sentence naming only `npm run build` and `npm test`. The console
    card is the complete record, so _full_args elides nothing."""
    cmd = ("npm run build -- --verbose " + "--flag=value " * 18
           + "&& rm -rf '/Users/likerun/Library/Mobile Documents/"
             "iCloud~md~obsidian/Documents/KEKE LI' && npm test")
    assert len(cmd) > 300
    spoken = _short_args({"command": cmd})
    assert _risk_note("Bash", {"command": cmd})     # the warning fires...
    assert "rm -rf" not in spoken                   # ...over an elided middle
    assert _full_args({"command": cmd}) == cmd      # and the card has it all
    assert _full_args({"file_path": "/a/b.txt"}) == "/a/b.txt"
    assert _full_args("not-a-dict") == "not-a-dict"


# ---------- CRITICAL 2: the spoken readback is never defeated by length ----------
async def test_a_long_command_is_click_only_and_never_read_with_the_middle_cut(
        tmp_path, monkeypatch):
    """A 200+ char command with a destructive clause parked in the middle used
    to be read aloud with exactly that clause elided — the ellipsis is
    inaudible in TTS, so the sentence sounded complete and a voice yes deleted
    the vault. The spoken line must NOT elide the middle: too long to read in
    full → refuse voice approval, say so, and point at the card. The approval
    is flagged voice_ok=False so the router refuses a voice yes."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    evil = ("python3 -c 'import shutil,os;"
            "shutil.rmtree(os.path.expanduser(\"~/vault\"))'")
    cmd = ("npm run lint && npm run typecheck && npm run build && "
           + "npm run test -- --coverage --runInBand --reporter verbose && "
           + evil + " && echo finishing-the-pipeline-run-now-okay-done")
    assert len(cmd) > 200
    cid, q = bus.subscribe()
    try:
        t = asyncio.create_task(w._on_tool_request(
            "Bash", {"command": cmd}, SimpleNamespace(title=None)))
        await asyncio.sleep(0.05)
        approval = router.pending_approvals()[0]
        assert approval.voice_ok is False                 # click-only
        card = None
        while not q.empty():
            ev = q.get_nowait()
            if ev and ev["type"] == "approval.request":
                card = ev["data"]
        assert card is not None
        # The spoken question must NOT contain a middle-elided command, and it
        # must SAY it is too long and name the card.
        assert "rmtree" not in card["question"]           # never a torn clause
        assert "…" not in card["question"]                # no inaudible ellipsis
        assert "too long" in card["question"].lower()
        assert card.get("voice_ok") is False
        assert card["full_args"] == cmd                   # the card still has it all
        fleet.deliver_approval(approval.nonce, False)
        await asyncio.wait_for(t, 1)
    finally:
        await cleanup(fleet)


def test_the_risk_scanner_sees_interpreter_form_destruction():
    """`python3 -c '...shutil.rmtree(...)'` names no path token the shell
    scanner recognises AND misses the rm blacklist — so it drew NO warning at
    all. The risk scanner now recognises interpreter-form deletion."""
    from server.fleet import _opaque_note
    assert _risk_note("Bash", {"command":
        'python3 -c "import shutil; shutil.rmtree(v)"'})
    # and the opaque note fires for interpreter one-liners and unknown shell vars
    assert _opaque_note({"command": 'python3 -c "print(1)"'})
    assert _opaque_note({"command": "rm -rf $VAULT/Daily"})
    assert _opaque_note({"command": "rm -rf $(cat target.txt)"})
    # a plain, fully-legible command draws no opaque note
    assert _opaque_note({"command": "npm test && rm -rf build"}) == ""
    # $HOME is resolved by the path scanner, so it is not "opaque"
    assert _opaque_note({"command": "rm -rf $HOME/Documents"}) == ""


async def test_the_card_carries_the_full_command_and_both_warnings(
        tmp_path, monkeypatch):
    """The click path was told LESS than the voice path: `_short_args` on the
    card too, and neither the risk note nor the outside-the-worktree note
    anywhere on it. There was no surface in Marvin showing the full command."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    # The destructive clause sits in the MIDDLE — the half _elide throws away.
    cmd = ("npm run build " + "--flag=value " * 14
           + "&& rm -rf /Users/likerun/Documents && npm test -- "
           + "--reporter=verbose " * 6)
    cid, q = bus.subscribe()
    try:
        t = asyncio.create_task(w._on_tool_request(
            "Bash", {"command": cmd}, SimpleNamespace(title=None)))
        await asyncio.sleep(0.05)
        fleet.deliver_approval(router.pending_approvals()[0].nonce, False)
        await asyncio.wait_for(t, 1)
        card = None
        while not q.empty():
            ev = q.get_nowait()
            if ev and ev["type"] == "approval.request":
                card = ev["data"]
        assert card is not None
        assert card["full_args"] == cmd                # verbatim, unelided
        assert "rm -rf /Users/likerun/Documents" not in card["args"]  # spoken half
        assert "destroy things" in card["risk"]
        assert "outside" in card["outside"].lower()
        assert card["worktree"] == w.worktree.path
    finally:
        await cleanup(fleet)


async def test_an_approval_says_when_the_target_is_outside_the_worktree(
        tmp_path, monkeypatch):
    """cwd is the worktree, but Write/Edit/Bash take absolute paths — the live
    smoke's very first real tool call was `Write /tmp/DONE.txt`. The worktree
    is not a sandbox, so the spoken line has to say when the blast lands
    outside it."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    inside = str(Path(w.worktree.path) / "sub" / "DONE.txt")
    cid, q = bus.subscribe()
    try:
        for tool, args in (("Write", {"file_path": "/tmp/DONE.txt"}),
                           ("Write", {"file_path": inside}),
                           ("Bash", {"command": "cat /etc/hosts"}),
                           ("Bash", {"command": "npm test && rm -rf build"})):
            t = asyncio.create_task(w._on_tool_request(
                tool, args, SimpleNamespace(title=None)))
            await asyncio.sleep(0.05)
            fleet.deliver_approval(router.pending_approvals()[0].nonce, False)
            await asyncio.wait_for(t, 1)
        questions = []
        while not q.empty():
            ev = q.get_nowait()
            if ev and ev["type"] == "approval.request":
                questions.append(ev["data"]["question"].lower())
        assert len(questions) == 4
        assert "outside its worktree" in questions[0]      # absolute escape
        assert "outside its worktree" not in questions[1]  # inside, no noise
        assert "outside its worktree" in questions[2]      # a command's target
        assert "outside its worktree" not in questions[3]  # relative, still inside
    finally:
        await cleanup(fleet)


async def test_the_readback_names_the_vault_and_the_marvin_repo(tmp_path,
                                                                monkeypatch):
    """One generic sentence made a Write into the owner's Obsidian vault sound
    exactly like a Write into /tmp/DONE.txt — "Outside its worktree, sir." for
    both. Fleet.forbidden already held the resolved vault root and the Marvin
    repo; it was consulted only at spawn, never in the readback."""
    vault, marvin = tmp_path / "vault", tmp_path / "marvin"
    vault.mkdir()
    marvin.mkdir()
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    fleet.protected = ((str(vault), "your Obsidian vault"),
                       (str(marvin), "the Marvin repo itself"))
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    w.protected = fleet.protected
    cid, q = bus.subscribe()
    try:
        for tool, args in (
                ("Write", {"file_path": str(vault / "Daily" / "2026-08-09.md")}),
                ("Bash", {"command": f"git -C {marvin} push --force"}),
                ("Write", {"file_path": "/tmp/DONE.txt"})):
            t = asyncio.create_task(w._on_tool_request(
                tool, args, SimpleNamespace(title=None)))
            await asyncio.sleep(0.05)
            fleet.deliver_approval(router.pending_approvals()[0].nonce, False)
            await asyncio.wait_for(t, 1)
        said = []
        while not q.empty():
            ev = q.get_nowait()
            if ev and ev["type"] == "approval.request":
                said.append(ev["data"]["question"])
        assert "inside your Obsidian vault" in said[0]
        assert "inside the Marvin repo itself" in said[1]
        assert said[2].count("Outside its worktree, sir.") == 1   # still generic
        assert "vault" not in said[2] and "Marvin repo" not in said[2]
    finally:
        await cleanup(fleet)


async def test_a_pending_card_can_be_replayed_after_the_sse_drops(tmp_path,
                                                                  monkeypatch):
    """A chatty worker evicts the browser's bus subscriber, and the console
    reconnects with NO Last-Event-ID — so an approval.request published in the
    gap never rendered, nothing resynced, and the worker sat blocked for the
    full TTL with no card and no spoken line. GET /fleet serves the pending
    cards so a reconnect (or a reload) gets them back."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    await fleet.spawn("soccer", repo(tmp_path), "task")
    w = fleet.workers[0]
    try:
        assert fleet.pending_cards() == []
        t = asyncio.create_task(w._on_tool_request(
            "Bash", {"command": "rm -rf /Users/likerun/Documents"},
            SimpleNamespace(title=None)))
        await asyncio.sleep(0.05)
        cards = fleet.pending_cards()
        assert len(cards) == 1
        assert cards[0]["nonce"] == router.pending_approvals()[0].nonce
        assert cards[0]["full_args"] == "rm -rf /Users/likerun/Documents"
        assert "destroy things" in cards[0]["risk"]
        assert "session_id" not in cards[0]          # nothing secret rides along
        fleet.deliver_approval(cards[0]["nonce"], False)
        await asyncio.wait_for(t, 1)
        # answered → never replayable again, however the page reloads
        assert fleet.pending_cards() == []
    finally:
        await cleanup(fleet)


def test_the_bash_scanner_sees_the_two_home_spellings():
    """`rm -rf $HOME/Documents` and `cd "$HOME/Documents" && rm -rf .` produced
    NO outside note at all — the scanner only recognised tokens starting `/`,
    `~/` or `../`. $HOME is the one variable worth folding: nothing else can
    start with `$HOME/`, so it costs no false positives. Every OTHER variable,
    command substitution, and interpreter form is still silent, by design and
    by documentation — see _named_paths."""
    assert _named_paths({"command": "rm -rf $HOME/Documents"}) == ["~/Documents"]
    assert _named_paths({"command": 'cd "${HOME}/Documents" && rm -rf .'}) \
        == ["~/Documents"]
    assert _named_paths({"command": "cd $HOME && ls"}) == ["~"]
    # unchanged: no path-shaped token, no claim
    assert _named_paths({"command": "git commit -m 'fix /etc/hosts parsing'"}) \
        == []
    assert _named_paths({"command": 'python -c "import shutil; shutil.rmtree(v)"'}) \
        == []
    assert _named_paths({"command": "rm -rf $VAULT/Daily"}) == []
