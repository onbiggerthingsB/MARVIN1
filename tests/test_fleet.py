import asyncio
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
from claude_agent_sdk import CanUseToolShadowedWarning

from server.bus import EventBus
from server.fleet import APPROVAL_WAIT_S, Fleet, Worker
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


def make_fleet(tmp_path, monkeypatch, max_workers=1, forbidden=None):
    monkeypatch.setenv("JARVIS_SKIP_PROXY_CHECK", "1")
    bus, router = EventBus(), Router()
    clients: list[FakeClient] = []

    def factory(options):
        c = FakeClient(options)
        clients.append(c)
        return c

    async def fake_worktree(repo, task, wtdir):
        dest = Path(wtdir) / f"wt-{len(clients)}"
        dest.mkdir(parents=True, exist_ok=True)
        return Worktree(repo=str(repo), path=str(dest), branch="jarvis/test-x",
                        base_commit="abc1234def5678")

    fleet = Fleet(bus=bus, router=router,
                  log=FleetLog(tmp_path / "state" / "fleet.jsonl"),
                  worktrees_dir=tmp_path / "state" / "worktrees",
                  forbidden=forbidden or (str(tmp_path / "vault"),
                                          str(tmp_path / "jarvis")),
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


async def test_spawn_refuses_the_vault_and_the_jarvis_repo(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    (tmp_path / "vault").mkdir()
    (tmp_path / "jarvis").mkdir()
    for bad in (str(tmp_path / "vault"), str(tmp_path / "jarvis")):
        spoken = await fleet.spawn("thing", bad, "do work")
        assert "don't run workers" in spoken
    assert fleet.workers == [] and clients == []


async def test_spawn_refuses_without_proxy_vars(tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    monkeypatch.delenv("JARVIS_SKIP_PROXY_CHECK", raising=False)
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
    await fleet.close_all()                           # flushes the log writer
    records, torn = fleet._log.replay()
    assert torn is False
    seqs = [r["seq"] for r in records]
    assert seqs == list(range(1, len(seqs) + 1))      # untorn, gapless, ordered
    kinds = [r["kind"] for r in records]
    assert kinds[0] == "spawned"
    assert kinds.count("prompt") == 2                 # the task, then the steer
    assert kinds[-1] == "session_end"
