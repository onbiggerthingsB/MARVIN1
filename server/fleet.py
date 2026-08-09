"""The worker fleet: spawn, steer, stop — real Claude Code sessions in
disposable worktrees, tracked honestly (spec §5).

Shape: every worker is one ClaudeSDKClient owned by exactly ONE asyncio task
(the consumer, reading receive_messages() for the worker's whole life —
streaming input from message one, never one-shot query()). Input arrives
through a bounded queue drained by a pump task. Permission requests block
inside can_use_tool on a future that the voice loop, a console click, or the
TTL resolves. Admission control: max_workers live workers (default 1 —
concurrency is not a documented entitlement), everything else queues.

Every public Fleet method SPEAKS its failures (returns a sentence) instead of
raising: the caller is a voice loop whose guards should be a last resort, not
the error path."""
from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections import deque
from pathlib import Path

from claude_agent_sdk import (ClaudeAgentOptions, ClaudeSDKClient,
                              PermissionResultAllow, PermissionResultDeny)

from server.fleet_state import (CLOSED, DETACHED, QUIET, WorkerStateMachine)
from server.worktrees import (WorktreeError, create_worktree, proxy_problem,
                              write_hook_settings)

APPROVAL_WAIT_S = 600.0        # == router.APPROVAL_TTL_S; unanswered → denied
INPUT_QUEUE_MAX = 4
TRANSCRIPT_KEEP = 200
SPAWN_TIMEOUT_S = 60.0
STOP_TIMEOUT_S = 15.0

# Reads may proceed without asking; EVERYTHING else lands in can_use_tool and
# becomes a voice approval. Explicit per-tool policy under permission_mode
# "default" — NOT acceptEdits, which auto-approves rm/mv/sed Bash and bypasses
# can_use_tool entirely (spec §5).
WORKER_ALLOWED_TOOLS = ["Read", "Grep", "Glob"]

_RISKY = ("rm ", "rm -", "mv ", "sudo", "--force", "--hard", "sed -i",
          "chmod", "chown", "curl", "git push")

_HOOK_KINDS = {
    "SessionStart": "spawned",
    "UserPromptSubmit": "prompt",
    "PreToolUse": "activity",
    "PostToolUse": "activity",
    "Notification": "permission_wait",
    "Stop": "turn_done",
    "SessionEnd": "session_end",
}


def _short_args(args, limit: int = 120) -> str:
    if not isinstance(args, dict):
        return str(args)[:limit]
    for key in ("command", "file_path", "path", "pattern", "url"):
        if args.get(key):
            return str(args[key])[:limit]
    return str(args)[:limit]


def _risk_note(tool: str, args) -> str:
    blob = f"{tool} {args}".lower()
    return ("Careful, sir — this one can destroy things. "
            if any(w in blob for w in _RISKY) else "")


def _shield_bearer(wt_path: Path) -> None:
    """Keep the bearer token out of the real repo's object store.

    write_hook_settings puts the bearer in .claude/settings.local.json BEFORE
    the CLI ever runs, so Claude Code's own auto-ignore may never fire — and a
    worker literally tasked to `git add -A && git commit` would commit the
    token through the worktree into the shared .git. The shield is a
    SELF-IGNORING .claude/.gitignore inside the worktree: it hides the
    settings file and itself from status/add/commit, it is worktree-scoped by
    construction (a plain file in the disposable checkout), and it mutates
    nothing shared — verified empirically, because git does NOT read a
    per-worktree .git/worktrees/<id>/info/exclude, and the repo-common
    info/exclude would change what the human's own checkout ignores.

    Idempotent and append-only: an existing .gitignore (tracked or not) keeps
    its content; only missing patterns are added. On a non-git directory this
    is inert — nothing there can commit the bearer anyway."""
    gi = Path(wt_path) / ".claude" / ".gitignore"
    gi.parent.mkdir(parents=True, exist_ok=True)
    text = gi.read_text(encoding="utf-8") if gi.exists() else ""
    have = set(text.splitlines())
    missing = [p for p in ("settings.local.json", ".gitignore")
               if p not in have]
    if missing:
        if text and not text.endswith("\n"):
            text += "\n"
        gi.write_text(text + "".join(p + "\n" for p in missing),
                      encoding="utf-8")


class _FleetLogWriter:
    """Single-writer funnel between the fleet and the SYNCHRONOUS FleetLog.

    FleetLog.append fsyncs on important kinds and assumes one writer for its
    sequence numbers. Called directly on the event loop it would stall live
    audio, and called from asyncio.to_thread ad hoc it would race the
    sequence. So: appends land in a FIFO deque on the loop thread, and ONE
    self-terminating drain task writes them via asyncio.to_thread, in order,
    one at a time (vault_write.py's to_thread posture, plus the single-writer
    guarantee). append() itself never raises and never blocks — a full disk
    dents the console via fleet.error, not the fleet.

    State transitions and bus publishes stay synchronous in the callers; only
    the durable record trails by the queue latency. A hard crash inside that
    window loses the tail exactly like any buffered write — FleetLog's torn
    handling copes; close_all() flushes on orderly shutdown."""

    def __init__(self, log, bus):
        self._log = log
        self._bus = bus
        self._backlog: deque[tuple[str, dict]] = deque()
        self._task: asyncio.Task | None = None

    def append(self, kind: str, data: dict) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._write_now(kind, dict(data))   # no loop — blocking is fine
            return
        self._backlog.append((kind, dict(data)))
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._drain(),
                                          name="fleet-log-writer")

    def _write_now(self, kind: str, data: dict) -> None:
        try:
            self._log.append(kind, data)
        except Exception as e:  # noqa: BLE001 — durability failure is reported, not fatal
            self._bus.publish("fleet.error",
                              {"reason": f"event log failed: {e}"})

    async def _drain(self) -> None:
        while self._backlog:
            kind, data = self._backlog.popleft()
            try:
                await asyncio.to_thread(self._log.append, kind, data)
            except Exception as e:  # noqa: BLE001
                self._bus.publish("fleet.error",
                                  {"reason": f"event log failed: {e}"})

    async def flush(self) -> None:
        while self._task is not None and not self._task.done():
            with contextlib.suppress(Exception):
                await self._task


class Worker:
    def __init__(self, *, project: str, path: str, task_text: str, wt,
                 bus, router, log, client_factory=ClaudeSDKClient,
                 now=time.time):
        self.id = uuid.uuid4().hex[:8]
        self.project = project
        self.path = path                  # the REAL repo path (registry key)
        self.task_text = task_text
        self.worktree = wt
        # Fresh machine per worker, ALWAYS: WorkerStateMachine bounces
        # "spawned" off CLOSED, so a machine reused from a previous worker
        # would silently ignore the new session.
        self.machine = WorkerStateMachine()
        self.session_id: str | None = None
        self.transcript: deque[dict] = deque(maxlen=TRANSCRIPT_KEEP)
        self.locked = False               # handoff lockout: no more input
        self.published_state: str | None = None
        self.consumer: asyncio.Task | None = None
        self.pump: asyncio.Task | None = None
        self._bus = bus
        self._router = router
        self._log = log
        self._now = now
        self._client_factory = client_factory
        self._client = None
        self._inbox: asyncio.Queue[str] = asyncio.Queue(maxsize=INPUT_QUEUE_MAX)
        self._futures: dict[str, asyncio.Future] = {}   # nonce -> decision

    # ---------- state plumbing ----------
    def _apply(self, kind: str, extra: dict | None = None) -> None:
        """State change + durable record + tile update, one place. The log
        append is guarded: a full disk must dent the console, not the fleet."""
        state = self.machine.apply(kind, self._now())
        data = {"worker": self.id, "project": self.project, "path": self.path,
                "state": state, "task": self.task_text,
                "worktree": self.worktree.path}
        if extra:
            data.update(extra)
        try:
            self._log.append(kind, data)
        except Exception as e:  # noqa: BLE001 — durability failure is reported, not fatal
            self._bus.publish("fleet.error", {"reason": f"event log failed: {e}"})
        self.published_state = state
        self._bus.publish("fleet.update", {
            "worker": self.id, "project": self.project, "path": self.path,
            "state": state, "task": self.task_text,
            "worktree": self.worktree.path})

    # ---------- lifecycle ----------
    def options(self) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            cwd=self.worktree.path,
            permission_mode="default",            # spec §5 — never acceptEdits
            can_use_tool=self._on_tool_request,
            allowed_tools=list(WORKER_ALLOWED_TOOLS),
            # "local" loads exactly the settings.local.json we wrote into the
            # worktree (the hook POSTs) and nothing of Keke's user or project
            # config. No vault/social/finance tools exist here (spec §9.3).
            setting_sources=["local"],
            strict_mcp_config=True,
        )

    async def start(self) -> None:
        self._client = self._client_factory(self.options())
        await self._client.connect()
        self._apply("spawned", {"base_commit": self.worktree.base_commit,
                                "branch": self.worktree.branch})
        self.consumer = asyncio.create_task(
            self._consume(), name=f"worker-{self.id}-consume")
        self.pump = asyncio.create_task(
            self._pump(), name=f"worker-{self.id}-pump")
        await self._client.query(self.task_text)
        self._apply("prompt")

    async def _consume(self) -> None:
        """Owns the message stream for the worker's whole life."""
        try:
            async for msg in self._client.receive_messages():
                kind = type(msg).__name__          # same probe Butler uses
                if kind == "SystemMessage":
                    data = getattr(msg, "data", None)
                    sid = (data.get("session_id")
                           if isinstance(data, dict) else None)
                    sid = sid or getattr(msg, "session_id", None)
                    if sid:
                        self.session_id = sid
                    continue
                if kind == "ResultMessage":
                    sid = getattr(msg, "session_id", None)
                    if sid:
                        self.session_id = sid
                    self._apply("turn_done")
                    continue
                texts = []
                for block in getattr(msg, "content", None) or []:
                    t = getattr(block, "text", None)
                    if t:
                        texts.append(t)
                    name = getattr(block, "name", None)     # ToolUseBlock
                    if name:
                        texts.append(f"[{name}] "
                                     f"{_short_args(getattr(block, 'input', {}) or {})}")
                if texts:
                    line = {"who": ("worker" if kind == "AssistantMessage"
                                    else kind.lower()),
                            "text": "\n".join(texts)[:2000]}
                    self.transcript.append(line)
                    self._bus.publish("fleet.message", {
                        "worker": self.id, "project": self.project,
                        "path": self.path, **line})
                self._apply("activity")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — a dead stream must say so on the tile
            self._bus.publish("fleet.error", {
                "worker": self.id, "project": self.project,
                "reason": f"worker stream died: {e}"})
            self._apply("lost")

    async def _pump(self) -> None:
        """Drains the bounded inbox into the live session, one steer at a time."""
        try:
            while True:
                text = await self._inbox.get()
                try:
                    await self._client.query(text)
                    self._apply("prompt")
                except Exception as e:  # noqa: BLE001 — a failed steer is reported, not fatal
                    self._bus.publish("fleet.error", {
                        "worker": self.id, "project": self.project,
                        "reason": f"steer failed: {e}"})
        except asyncio.CancelledError:
            raise

    def steer(self, text: str) -> str:
        if self.locked:
            return (f"{self.project} is detached, sir — "
                    f"steer it from its terminal.")
        try:
            self._inbox.put_nowait(text)
        except asyncio.QueueFull:
            return (f"{self.project} already has a backlog, sir — "
                    f"give it a moment.")
        return f"Told {self.project}, sir."

    # ---------- approvals ----------
    async def _on_tool_request(self, tool_name, tool_input, context):
        """The SDK blocks the worker's tool call on this coroutine. Open a
        path-keyed approval on the shared router, publish the card + spoken
        readback, and wait for the voice loop, a click, or the TTL."""
        if self.locked:
            return PermissionResultDeny(
                message="JARVIS handoff in progress", interrupt=True)
        approval = self._router.open_approval(
            self.project, f"{tool_name}: {_short_args(tool_input)}",
            now=self._now(), path=self.path)
        fut = asyncio.get_running_loop().create_future()
        self._futures[approval.nonce] = fut
        self._apply("permission_wait",
                    {"nonce": approval.nonce, "tool": tool_name,
                     "args": _short_args(tool_input)})
        title = getattr(context, "title", None) or (
            f"{self.project} wants {tool_name}")
        self._bus.publish("approval.request", {
            "nonce": approval.nonce, "worker": self.id,
            "project": self.project, "path": self.path, "tool": tool_name,
            "args": _short_args(tool_input),
            "question": (f"{_risk_note(tool_name, tool_input)}{title} — "
                         f"{_short_args(tool_input)}. Approve or deny, sir?")})
        approved = False
        try:
            approved = bool(await asyncio.wait_for(fut, APPROVAL_WAIT_S))
        except asyncio.TimeoutError:
            self._router.take_nonce(approval.nonce, self._now())
            self._bus.publish("approval.resolved", {
                "outcome": "expired", "project": self.project,
                "tool": approval.tool, "nonce": approval.nonce})
        finally:
            self._futures.pop(approval.nonce, None)
        self._apply("permission_done",
                    {"nonce": approval.nonce, "approved": approved})
        if approved:
            return PermissionResultAllow()
        return PermissionResultDeny(
            message="Keke denied this by voice", interrupt=False)

    def deliver_approval(self, nonce: str, approved: bool) -> bool:
        fut = self._futures.get(nonce)
        if fut is None or fut.done():
            return False
        # Consume the router card AT delivery, so a pending list read a
        # millisecond later cannot show an approval that is already decided.
        # The voice path (resolve_approval) removes it before calling us —
        # take_nonce on a gone nonce is a harmless no-op, never a double
        # consume.
        self._router.take_nonce(nonce, self._now())
        fut.set_result(bool(approved))
        return True

    # ---------- teardown ----------
    async def shutdown(self, *, interrupt_first: bool) -> None:
        """§5 lockout core, shared by stop and handoff: lock input → reject
        pending approvals → interrupt → cancel own tasks → disconnect →
        verify the tasks actually ended."""
        self.locked = True
        for nonce, fut in list(self._futures.items()):
            if not fut.done():
                fut.set_result(False)                 # reject, unblock the SDK
            self._router.take_nonce(nonce, self._now())
        if interrupt_first and self._client is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._client.interrupt(), 10)
        for task in (self.pump, self.consumer):
            if task is not None:
                task.cancel()
        if self._client is not None:
            await self._client.disconnect()           # raises → caller marks lost
        for task in (self.pump, self.consumer):
            if task is not None:
                with contextlib.suppress(BaseException):
                    await task                        # verify exit — really gone


class Fleet:
    def __init__(self, *, bus, router, log, worktrees_dir: Path,
                 forbidden: tuple[str, ...] = (), max_workers: int = 1,
                 client_factory=ClaudeSDKClient, now=time.time,
                 worktree_factory=create_worktree,
                 settings_writer=write_hook_settings,
                 hook_port: int = 7777, hook_bearer: str = "",
                 open_terminal=None):
        self.workers: list[Worker] = []
        self.queue: deque[tuple[str, str, str]] = deque()
        self.ghosts: list[dict] = []      # restart reports (Task 10)
        self.max_workers = max_workers
        self.worktrees_dir = Path(worktrees_dir)
        self.forbidden = tuple(str(Path(f).resolve()) for f in forbidden)
        self.hook_port = hook_port
        self.hook_bearer = hook_bearer
        self._bus = bus
        self._router = router
        self._log = log                       # raw FleetLog: replay/snapshot
        self._log_writer = _FleetLogWriter(log, bus)   # ALL appends go here
        self._now = now
        self._client_factory = client_factory
        self._worktree_factory = worktree_factory
        self._settings_writer = settings_writer
        self._open_terminal = open_terminal   # Task 9 wires the default

    # ---------- lookup ----------
    @property
    def live(self) -> list[Worker]:
        # UNKNOWN deliberately counts as live: a worker we lost track of may
        # still hold a subprocess, so it blocks admission until stopped
        # explicitly. Fail closed on quota, never over-spawn.
        return [w for w in self.workers
                if w.machine.base not in (CLOSED, DETACHED)]

    def _find(self, path: str) -> Worker | None:
        hits = [w for w in self.workers
                if w.path == path or w.worktree.path == path]
        for w in hits:
            if w.machine.base not in (CLOSED, DETACHED):
                return w
        return hits[0] if hits else None

    # ---------- commands (every failure is a sentence) ----------
    async def spawn(self, project: str, path: str, task: str) -> str:
        try:
            resolved = str(Path(path).resolve())
            for bad in self.forbidden:
                if resolved == bad or resolved.startswith(bad + "/"):
                    # spec §5/§9: never the vault, never the JARVIS repo
                    return "I don't run workers in that directory, sir."
            problem = proxy_problem()
            if problem:
                return f"I can't spawn safely, sir: {problem}."
            if not (task or "").strip():
                return "Spawn it to do what, sir?"
            if len(self.live) >= self.max_workers:
                self.queue.append((project, path, task))
                self._log_writer.append("queued", {"project": project,
                                                   "path": path, "task": task})
                return (f"The fleet is at capacity, sir — {project} is queued "
                        f"at position {len(self.queue)}.")
            wt = await self._worktree_factory(Path(path), task,
                                              self.worktrees_dir)
            self._settings_writer(Path(wt.path), self.hook_port,
                                  self.hook_bearer)
            _shield_bearer(Path(wt.path))   # before the CLI can ever git add
            worker = Worker(project=project, path=path, task_text=task, wt=wt,
                            bus=self._bus, router=self._router,
                            log=self._log_writer,
                            client_factory=self._client_factory, now=self._now)
            try:
                await asyncio.wait_for(worker.start(), SPAWN_TIMEOUT_S)
            except BaseException:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        worker.shutdown(interrupt_first=False), 5)
                raise
            self.workers.append(worker)
            # Spec §3 acknowledgment honesty: "On it" ONLY now — the worktree
            # exists, the client connected, the task is in the session.
            return (f"On it, sir — {project} is working in a fresh worktree "
                    f"from commit {wt.base_commit[:7]}.")
        except WorktreeError as e:
            return f"I couldn't prepare a worktree for {project}, sir: {e}"
        except asyncio.TimeoutError:
            # worktrees._git raises asyncio.TimeoutError, NOT WorktreeError,
            # when git itself hangs — and worker.start() has its own deadline.
            # Both must end as speech, never as an unhandled exception.
            self._bus.publish("fleet.error",
                              {"reason": f"spawn timed out for {project}"})
            return (f"Spawning {project} timed out, sir — no worker is "
                    f"running.")
        except Exception as e:  # noqa: BLE001 — spawn failures are spoken, never raised
            self._bus.publish("fleet.error", {"reason": f"spawn failed: {e}"})
            return f"Spawning {project} failed, sir — details on screen."

    def steer_path(self, path: str, text: str) -> str:
        w = self._find(path)
        if w is None:
            return "Nothing is running there, sir."
        if not (text or "").strip():
            return "Tell it what, sir?"
        return w.steer(text)

    async def stop(self, path: str) -> str:
        w = self._find(path)
        if w is None:
            return "Nothing is running there, sir."
        if w.machine.base == DETACHED:
            return f"{w.project} is detached, sir — stop it from its terminal."
        try:
            await asyncio.wait_for(w.shutdown(interrupt_first=True),
                                   STOP_TIMEOUT_S)
            w._apply("session_end")
            spoken = (f"Stopped {w.project}, sir. Its worktree is preserved "
                      f"for your review.")
        except Exception as e:  # noqa: BLE001 — a dirty stop is reported, not raised
            w._apply("lost", {"reason": f"stop failed: {e}"})
            spoken = (f"I couldn't stop {w.project} cleanly, sir — "
                      f"treat it as unknown.")
        await self._admit_next()
        return spoken

    async def _admit_next(self) -> None:
        if not self.queue or len(self.live) >= self.max_workers:
            return
        project, path, task = self.queue.popleft()
        spoken = await self.spawn(project, path, task)
        self._bus.publish("fleet.spoken",
                          {"text": f"Next in the queue, sir: {spoken}"})

    # ---------- reporting ----------
    def status_line(self) -> str:
        if not self.workers and not self.queue:
            return "The fleet is empty, sir."
        now = self._now()
        parts = [f"{w.project} is {w.machine.state(now).replace('_', ' ').lower()}"
                 for w in self.workers]
        if self.queue:
            parts.append(f"{len(self.queue)} queued")
        return "; ".join(parts) + "."

    def transcript(self, path: str) -> list[dict]:
        w = self._find(path)
        return list(w.transcript) if w is not None else []

    def one_breath(self, path: str) -> str:
        w = self._find(path)
        if w is None:
            return "Nothing is running there, sir."
        state = w.machine.state(self._now()).replace("_", " ").lower()
        last = (w.transcript[-1]["text"][:80] if w.transcript
                else "nothing yet")
        return f"{w.project} is {state}. Last activity: {last}."

    # ---------- inbound ----------
    def deliver_approval(self, nonce: str, approved: bool) -> bool:
        return any(w.deliver_approval(nonce, approved) for w in self.workers)

    def handle_hook(self, payload: dict) -> None:
        """The second detection layer: the CLI's own lifecycle, POSTed by the
        settings we wrote into the worktree. Matches by session_id first, then
        by worktree cwd; sessions we don't own are surfaced for M4."""
        event = str(payload.get("hook_event_name", ""))
        kind = _HOOK_KINDS.get(event)
        sid = payload.get("session_id")
        cwd = str(payload.get("cwd", ""))
        w = next((w for w in self.workers
                  if (sid and w.session_id == sid)
                  or (cwd and cwd == w.worktree.path)), None)
        if w is None:
            self._bus.publish("fleet.unknown_session",
                              {"session_id": sid, "cwd": cwd, "event": event})
            return
        if sid and not w.session_id:
            w.session_id = sid            # hooks can learn the id before the stream
        if kind == "permission_wait" and (w.consumer is not None
                                          and not w.consumer.done()):
            # SDK-owned session: can_use_tool is the ONLY permission source.
            # permission_wait arrives on two independent layers, and a slow
            # Notification POST landing AFTER can_use_tool already applied
            # permission_done would park the tile on WAITING_PERMISSION
            # forever (it never decays). Dropping the hook copy for sessions
            # whose consumer is alive makes that parked state impossible; a
            # worker without a live consumer (detached, stream lost) keeps
            # the hook as its only remaining signal.
            return
        if kind:
            w._apply(kind)

    # ---------- housekeeping ----------
    async def tick(self, now: float | None = None) -> None:
        """Derive QUIET (someone must look at the clock) and escalate ONLY on
        a failed health probe: consumer task dead = probe failure."""
        now = self._now() if now is None else now
        for w in self.workers:
            state = w.machine.state(now)
            if state == QUIET and (w.consumer is None or w.consumer.done()):
                state = w.machine.probe_failed(now)
                self._bus.publish("fleet.error", {
                    "worker": w.id, "project": w.project,
                    "reason": "health probe failed — worker marked unknown"})
            if state != w.published_state:
                w.published_state = state
                self._bus.publish("fleet.update", {
                    "worker": w.id, "project": w.project, "path": w.path,
                    "state": state, "task": w.task_text,
                    "worktree": w.worktree.path})
        await self._admit_next()

    async def close_all(self) -> None:
        """Server shutdown. Deliberately does NOT log session_end: the next
        boot's recover() must report these workers UNKNOWN/INTERRUPTED —
        their sessions are gone with the SDK clients (spec §5 durability).
        NEVER removes a worktree: each one holds a diff the human may still
        want to merge back."""
        for w in self.workers:
            if w.machine.base not in (CLOSED, DETACHED):
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        w.shutdown(interrupt_first=False), 10)
        await self._log_writer.flush()    # orderly shutdown keeps its records
