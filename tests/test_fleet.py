import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

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
