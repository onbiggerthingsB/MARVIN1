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
import json
import shlex
import time
import uuid
import warnings
from collections import deque
from pathlib import Path

from claude_agent_sdk import (CanUseToolShadowedWarning, ClaudeAgentOptions,
                              ClaudeSDKClient, PermissionResultAllow,
                              PermissionResultDeny)

from server.fleet_state import (CLOSED, DETACHED, QUIET, UNKNOWN,
                                WAITING_PERMISSION, WorkerStateMachine)
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


async def _default_open_terminal(cmd: str) -> None:
    """Open Terminal.app running `cmd`. json.dumps produces a double-quoted
    literal whose \" and \\ escapes AppleScript reads the same way, and the
    command never contains newlines.

    ensure_ascii=False is load-bearing: with the default, a non-ASCII worktree
    path (a project named in Chinese, an accent in a folder) is emitted as
    \\uXXXX, which AppleScript does NOT decode — it would `cd` into a literal
    backslash-u path, fail, and still exit 0, so JARVIS would confidently say
    "yours in the terminal" over a shell sitting in the wrong directory.

    The child is reaped on every exit: a hung osascript (a modal dialog, a
    wedged Terminal) outlives the wait_for otherwise, leaving an orphan
    process holding the window for the rest of the session."""
    script = f'tell application "Terminal" to do script {json.dumps(cmd, ensure_ascii=False)}'
    proc = await asyncio.create_subprocess_exec(
        "osascript", "-e", script,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        stdin=asyncio.subprocess.DEVNULL)
    try:
        await asyncio.wait_for(proc.wait(), 10)
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"osascript exited {proc.returncode}")


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


class _Snapshot:
    """Sentinel `kind` marking a queue item as a compaction, not an event.

    An object, not a string: no real event kind can ever collide with it, so
    a forged or drifting kind name cannot make an append rotate the log."""


_SNAPSHOT = _Snapshot()


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
        self._backlog: deque[tuple[object, dict]] = deque()
        self._task: asyncio.Task | None = None

    def append(self, kind: str, data: dict) -> None:
        self._enqueue(kind, dict(data))

    def snapshot(self, state: dict) -> None:
        """Compaction, queued behind every append already in flight.

        FleetLog.snapshot fsyncs a temp file, ROTATES the log out from under
        the append path, and fsyncs the directory. Run on the loop thread it
        would stall live audio; run from an ad-hoc to_thread it would rename
        the log while a concurrent append still holds the old inode — the
        record would land in `.jsonl.1` and vanish from the next replay. Same
        funnel, same single writer, same ordering."""
        self._enqueue(_SNAPSHOT, dict(state))

    def _enqueue(self, kind, payload: dict) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._write_now(kind, payload)      # no loop — blocking is fine
            return
        self._backlog.append((kind, payload))
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._drain(),
                                          name="fleet-log-writer")

    def _write_now(self, kind, payload: dict) -> None:
        try:
            if kind is _SNAPSHOT:
                self._log.snapshot(payload)
            else:
                self._log.append(kind, payload)
        except Exception as e:  # noqa: BLE001 — durability failure is reported, not fatal
            self._bus.publish("fleet.error",
                              {"reason": f"event log failed: {e}"})

    async def _drain(self) -> None:
        while self._backlog:
            kind, payload = self._backlog.popleft()
            try:
                if kind is _SNAPSHOT:
                    await asyncio.to_thread(self._log.snapshot, payload)
                else:
                    await asyncio.to_thread(self._log.append, kind, payload)
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
        self.handoff_in_flight = False    # a lockout sequence is mid-flight
        # The symmetric partner of handoff_in_flight, set by _stop_worker (so
        # it covers stop() AND _spawn's deferred stop). Both flags are pure
        # REFUSALS — neither side ever waits on the other, so they cannot
        # deadlock — and both are reserved synchronously, with no await
        # between the check and the set, so exactly one sequence can own a
        # worker's teardown at a time.
        self.stop_in_flight = False
        self.stop_requested = False       # a stop that landed mid-start, deferred
        # Set by _spawn around its own await of start(), cleared the moment
        # that await returns — the ONLY thing that may mean "starting".
        self.spawn_in_flight = False
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

    @property
    def starting(self) -> bool:
        """True while a LIVE _spawn has not yet built the session's tasks: the
        client may be absent or half-connected and there is no pump. Early
        registration makes this window visible to _find, so stop() and steer()
        must not treat it as a running session.

        Gated on the explicit flag, never on `consumer is None` alone: stop()
        answers this window with a PROMISE that only _spawn's post-start()
        check redeems, so a worker registered by any other path (Task 10's
        restart registration) must not look like one — that promise would
        never be redeemed, stop() would be a permanent no-op, and steer()
        would refuse forever."""
        return self.spawn_in_flight and self.consumer is None

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
        # The installed SDK warns (CanUseToolShadowedWarning, on connect) that
        # allowed_tools=["Read","Grep","Glob"] shadows can_use_tool for those
        # tools. Here the shadowing is INTENTIONAL: reads proceed without a
        # voice approval, everything else falls through to can_use_tool (spec
        # §5). Suppress exactly that category at this construction site only —
        # a production spawn must not open with an alarming warning — while
        # any OTHER warning the SDK raises still surfaces.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CanUseToolShadowedWarning)
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
            # The stream ended WITHOUT an exception: the CLI process exited
            # (session limit, /exit, clean crash). Returning silently here
            # would park the worker at its last state forever — still counted
            # live, still blocking the queue, with steer reporting success
            # into a dead client — and tick could never catch it (only QUIET
            # is probed, and IDLE_AT_PROMPT never derives QUIET). Apply
            # "lost" instead; if the SessionEnd hook already landed the
            # machine is CLOSED and this is a clean, expected end — apply
            # would bounce off CLOSED anyway, so skip the error noise too.
            if self.machine.base not in (CLOSED, DETACHED):
                self._bus.publish("fleet.error", {
                    "worker": self.id, "project": self.project,
                    "reason": "worker stream ended — the CLI process exited"})
                self._apply("lost")
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
        if self.starting:
            # No pump exists yet: "Told X" would be a claim of delivery into
            # an inbox nothing drains until connect() finishes — and a claim
            # silently dropped if the spawn fails. Refuse honestly instead.
            return (f"{self.project} is still starting, sir — "
                    f"give it a moment.")
        if self.handoff_in_flight:
            # Mid-lockout: "spawn it again" below would be bad advice about a
            # session that is seconds from being alive in a terminal.
            return (f"I'm handing {self.project} over to a terminal right "
                    f"now, sir — one moment.")
        if self.locked:
            if self.machine.base == DETACHED:
                return (f"{self.project} is detached, sir — "
                        f"steer it from its terminal.")
            # `locked` is set by EVERY shutdown() — stop, close_all, and a
            # FAILED handoff — and it is never cleared. Only a confirmed
            # handoff opens a terminal, so sending Keke to one here would name
            # a window that does not exist. Refuse without claiming where it
            # went; the state that matters was already spoken by stop().
            return (f"{self.project} isn't taking input any more, sir — "
                    f"spawn it again if you need more work there.")
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
        except asyncio.CancelledError:
            # The SDK tore down its callback task — a disconnect or interrupt
            # OUTSIDE stop()'s path (close_all, an SDK-initiated abort).
            # Without repair the router card would linger until TTL and the
            # base state would sit on WAITING_PERMISSION, which never decays.
            # Sweep the nonce, repair the state, and RE-RAISE: cancellation
            # is never swallowed.
            self._router.take_nonce(approval.nonce, self._now())
            self._futures.pop(approval.nonce, None)
            self._bus.publish("approval.resolved", {
                "outcome": "cancelled", "project": self.project,
                "tool": approval.tool, "nonce": approval.nonce})
            self._apply("permission_done",
                        {"nonce": approval.nonce, "approved": False,
                         "cancelled": True})
            raise
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
            rejected = not fut.done()
            if rejected:
                fut.set_result(False)                 # reject, unblock the SDK
            taken = self._router.take_nonce(nonce, self._now())
            if rejected:
                # Shutdown IS the resolver for this nonce, so it must publish
                # the resolution: nothing downstream will. The resumed
                # _on_tool_request only applies permission_done (its normal
                # path never publishes — the voice loop and /approval do),
                # and the voice/click paths can no longer fire because the
                # nonce is consumed. Without this event the console card
                # stays on screen forever after "stop soccer" — until a
                # manual click 404s it or the page reloads. An already-done
                # future was resolved (and published) elsewhere; publishing
                # again would overwrite a truthful "approved" status line
                # with "cancelled".
                self._bus.publish("approval.resolved", {
                    "outcome": "cancelled", "project": self.project,
                    "tool": taken.tool if taken is not None else "",
                    "nonce": nonce})
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
        # Spawns that have RESERVED a slot but not yet registered a worker.
        # Counted into every admission check so two overlapping spawns can
        # never both pass while the first is still awaiting its worktree —
        # max_workers=1 is the invariant this class exists to enforce.
        self._pending_spawns = 0
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
        self._open_terminal = open_terminal or _default_open_terminal

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
        return await self._spawn(project, path, task, requeue_front=False)

    async def _spawn(self, project: str, path: str, task: str,
                     requeue_front: bool) -> str:
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
            # Admission is decided and RESERVED synchronously — no await
            # between this check and the increment below — so overlapping
            # spawns (HTTP route + voice loop, Task 8) can never both admit.
            if len(self.live) + self._pending_spawns >= self.max_workers:
                if requeue_front:
                    # A queue head popped by _admit_next that bounced off
                    # capacity goes back to the FRONT — it must never be
                    # demoted behind later arrivals.
                    self.queue.appendleft((project, path, task))
                    position = 1
                else:
                    self.queue.append((project, path, task))
                    position = len(self.queue)
                self._log_writer.append("queued", {"project": project,
                                                   "path": path, "task": task})
                return (f"The fleet is at capacity, sir — {project} is queued "
                        f"at position {position}.")
            self._pending_spawns += 1
            worker = None
            try:
                wt = await self._worktree_factory(Path(path), task,
                                                  self.worktrees_dir)
                self._settings_writer(Path(wt.path), self.hook_port,
                                      self.hook_bearer)
                _shield_bearer(Path(wt.path))  # before the CLI can ever git add
                worker = Worker(project=project, path=path, task_text=task,
                                wt=wt, bus=self._bus, router=self._router,
                                log=self._log_writer,
                                client_factory=self._client_factory,
                                now=self._now)
                # Register BEFORE start(): the CLI's SessionStart hook can
                # POST during connect(), and an unregistered worker would be
                # published as fleet.unknown_session. published_state is
                # pre-seeded so a tick firing mid-connect does not publish a
                # spurious UNKNOWN tile for a healthy spawning worker.
                worker.published_state = worker.machine.base
                # THIS spawn owns the start below, so it — and only it — can
                # redeem the promise stop() makes to a starting worker. The
                # finally clears the flag the moment that await returns.
                worker.spawn_in_flight = True
                self.workers.append(worker)
                try:
                    await asyncio.wait_for(worker.start(), SPAWN_TIMEOUT_S)
                except BaseException:
                    if worker in self.workers:   # failed spawn: deregister
                        self.workers.remove(worker)
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(
                            worker.shutdown(interrupt_first=False), 5)
                    raise
            finally:
                self._pending_spawns -= 1
                if worker is not None:
                    worker.spawn_in_flight = False
            if worker.stop_requested:
                # A stop landed while start() was awaited; stop() deferred it
                # to us. Execute it now that the client, consumer and pump
                # really exist — never speak "On it" over a worker the user
                # already stopped, and admit the queue only once the CLI is
                # actually gone.
                if await self._stop_worker(worker):
                    spoken = (f"{project} is stopped as you asked, sir — it "
                              f"finished starting and I shut it down. Its "
                              f"worktree is preserved for your review.")
                else:
                    spoken = (f"You asked me to stop {project}, sir, but it "
                              f"wouldn't stop cleanly — treat it as unknown.")
                await self._admit_next()
                return spoken
            if worker.machine.base == CLOSED or worker.locked:
                # The worker was torn down under the in-flight start without a
                # deferred stop: a stop() in the query window (the consumer
                # already existed, so stop took the normal teardown path) or a
                # SessionEnd hook (the CLI itself exited). CLOSED alone is not
                # enough to see that: a DIRTY stop applies "lost" (UNKNOWN),
                # and nothing bounces off UNKNOWN, so the resuming start()'s
                # _apply("prompt") drives it to ACTIVE_TURN and "On it" would
                # contradict the "couldn't stop cleanly" Keke just heard.
                # `locked` is set ONLY by shutdown(), so it covers the clean
                # stop, the dirty stop and close_all, while CLOSED still
                # covers the hook case (no shutdown ran there).
                torn_down = True
                try:
                    await asyncio.wait_for(
                        worker.shutdown(interrupt_first=False), 5)
                except Exception as e:  # noqa: BLE001 — reported, never silent
                    # Speaking "nothing is running there" over a client that
                    # would not tear down would be a flat assertion of a dead
                    # worker on top of a live one — and on a CLOSED machine
                    # _apply("lost") bounces, so this publish is the only
                    # place the failure can be recorded at all.
                    torn_down = False
                    self._bus.publish("fleet.error", {
                        "worker": worker.id, "project": project,
                        "reason": f"teardown after a mid-start stop failed: {e}"})
                if worker.machine.base != CLOSED:
                    # A dirty stop left UNKNOWN and the resumed start() pushed
                    # it back to ACTIVE_TURN — a lie over a cancelled consumer
                    # and pump. Say so on the tile too; UNKNOWN keeps counting
                    # live, so admission stays blocked until Keke stops it.
                    worker._apply("lost", {"reason": "stopped while starting"})
                if torn_down:
                    # The belt must drain the queue exactly like the deferred
                    # branch above: stop()'s own _admit_next was refused
                    # because _pending_spawns still counted THIS spawn, so
                    # without this the queued project waits behind zero live
                    # workers — forever, until tick() is wired. _admit_next
                    # re-reads self.live, so the UNKNOWN of a dirty stop still
                    # blocks it: never admit beside a live subprocess.
                    await self._admit_next()
                if torn_down and worker.machine.base == CLOSED:
                    return (f"{project} was stopped while it was starting, "
                            f"sir — nothing is running there.")
                return (f"{project} was stopped while it was starting, sir, "
                        f"but it wouldn't shut down cleanly — treat it as "
                        f"unknown.")
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
        if w.handoff_in_flight:
            # A stop inside the lockout would tear the session down under the
            # handoff: two concurrent shutdowns, session_end lands first, and
            # the handoff's "detached" bounces off CLOSED — leaving "Stopped
            # X" and "X is yours in the terminal" both spoken about one
            # worker. The handoff finishes in seconds; a stop after it is
            # answered honestly by the DETACHED branch above.
            return (f"I'm handing {w.project} over to a terminal right now, "
                    f"sir — one moment.")
        if w.starting:
            # Early registration makes a half-started worker visible here
            # while start() is still awaited. Tearing it down NOW would
            # no-op or misfire shutdown (no consumer/pump, a client that may
            # be mid-connect), close the tile over the in-flight start, and
            # free the slot so a queued worker gets admitted beside the
            # zombie CLI. Defer instead: _spawn executes this stop the
            # moment start() returns, and only then drains the queue.
            w.stop_requested = True
            return (f"{w.project} is still starting, sir — I'll stop it the "
                    f"moment it's up.")
        spoken = (f"Stopped {w.project}, sir. Its worktree is preserved "
                  f"for your review."
                  if await self._stop_worker(w) else
                  f"I couldn't stop {w.project} cleanly, sir — "
                  f"treat it as unknown.")
        await self._admit_next()
        # Deliberately NO snapshot here (see Fleet.snapshot): stop's final
        # state is the fsync'd `session_end` this method just applied, so a
        # compaction would add nothing and would rotate the stopped worker's
        # whole event history into the single generation that the next
        # compaction overwrites.
        return spoken

    async def _stop_worker(self, w: Worker) -> bool:
        """Lockout → interrupt → disconnect → session_end, shared by stop()
        and _spawn's deferred stop. False = dirty stop, worker marked lost
        (UNKNOWN still counts live, so admission stays blocked).

        Reserves stop_in_flight SYNCHRONOUSLY, before the first await: this
        suspends for seconds inside the interrupt wait and only applies
        session_end afterwards, so without the reservation a handoff clicked
        into that window passes every one of its gates and one of the two
        sequences ends up lying about the other."""
        w.stop_in_flight = True
        try:
            await asyncio.wait_for(w.shutdown(interrupt_first=True),
                                   STOP_TIMEOUT_S)
            w._apply("session_end")
            return True
        except Exception as e:  # noqa: BLE001 — a dirty stop is reported, not raised
            w._apply("lost", {"reason": f"stop failed: {e}"})
            return False
        finally:
            w.stop_in_flight = False

    async def handoff(self, path: str) -> dict:
        """Spec §5 lockout: stop input → reject pending approvals → interrupt
        → close the SDK subprocess → verify exit → lock → `claude --resume`.
        DETACHED only after the session is provably closed on our side; any
        failure leaves the worker UNKNOWN and returns NO resume command —
        two drivers on one session interleave messages."""
        w = self._find(path)
        if w is None:
            # A restart ghost is not a Worker, so _find cannot see it — but
            # its tile IS on the console, button and all, and "nothing is
            # running there" would read as a denial that the tile means
            # anything. Answer the tile Keke is looking at. This is also the
            # ghost's handoff gate: spawn_in_flight guards a starting worker,
            # never a ghost, and there is no session id to resume — the SDK
            # client that owned it died with the old process.
            ghost = next((g for g in self.ghosts
                          if g.get("path") == path), None)
            if ghost is None:
                return {"ok": False, "spoken": "Nothing is running there, sir."}
            project = ghost.get("project") or "That worker"
            if ghost.get("state") == DETACHED:
                return {"ok": False,
                        "spoken": (f"{project} was already detached before the "
                                   f"restart, sir — it has its own terminal.")}
            return {"ok": False,
                    "spoken": (f"{project} didn't survive the restart, sir — I "
                               f"don't hold its session any more, so there's "
                               f"nothing to hand over. Its worktree is "
                               f"preserved.")}
        if w.machine.base == DETACHED:
            # DETACHED is a one-way door: the terminal owns this session now,
            # and a second window on it is the exact accident this prevents.
            return {"ok": False,
                    "spoken": f"{w.project} is already detached, sir."}
        if w.machine.base == CLOSED:
            # Tiles are never removed, so a stopped worker keeps its button.
            # CLOSED is final — "detached" bounces off it — so going on would
            # open a terminal and claim a handoff the tile flatly contradicts.
            return {"ok": False,
                    "spoken": (f"{w.project}'s session is already closed, "
                               f"sir — there's nothing to hand over.")}
        if w.handoff_in_flight:
            # Reserved synchronously below, before the first await: the button
            # has no debounce and the lockout takes seconds, so both clicks
            # would otherwise clear the DETACHED gate and open two windows on
            # ONE session — two drivers, exactly what this method prevents.
            return {"ok": False,
                    "spoken": f"I'm already handing {w.project} over, sir."}
        if w.stop_in_flight:
            # The mirror of the branch above, and of stop()'s deferral to
            # handoff_in_flight. A stop is already seconds deep in its own
            # teardown — suspended inside the interrupt wait, before it applies
            # session_end — so every gate here still reads live. Going on ends
            # in one of two lies: `detached` bounces off the CLOSED the stop
            # lands (a terminal over a tile reading CLOSED), or the stop's
            # session_end — the one event allowed off DETACHED — collapses the
            # just-detached worker back to CLOSED. Refuse; the stop finishes in
            # seconds and the CLOSED gate answers the next press honestly.
            return {"ok": False,
                    "spoken": (f"I'm stopping {w.project} right now, sir — "
                               f"one moment.")}
        if w.spawn_in_flight:
            # The WHOLE spawn, not the `starting` property: that is False for
            # the second half of start() (consumer and pump are built, then
            # client.query(task_text) is awaited), where the tile and its
            # button already exist and the SessionStart hook may already have
            # handed us a session id. Detaching in that window opens a terminal
            # and then lets _spawn resume into a spawn-failure or
            # stopped-while-starting sentence that flatly contradicts it.
            return {"ok": False,
                    "spoken": (f"{w.project} is still starting, sir — "
                               f"give it a moment.")}
        if not w.session_id:
            # Nothing to resume: `claude --resume` needs an id the CLI has not
            # reported yet. Detaching here would strand the worker forever —
            # the machine bounces everything but session_end off DETACHED.
            return {"ok": False,
                    "spoken": (f"{w.project} has no resumable session yet, "
                               f"sir — give it a moment.")}
        w.handoff_in_flight = True        # reserved BEFORE the first await
        try:
            try:
                # lock input + reject approvals + interrupt + close + verify exit
                await asyncio.wait_for(w.shutdown(interrupt_first=True),
                                       STOP_TIMEOUT_S)
                for t in (w.consumer, w.pump):
                    if t is not None and not t.done():
                        raise RuntimeError("worker tasks still running")
                # The transition is a STEP, verified like every other one. The
                # CLOSED gate above protects only this sequence's ENTRY, and
                # WorkerStateMachine.apply BOUNCES every event but session_end
                # off CLOSED *without raising*: a concurrent stop, or the CLI's
                # own SessionEnd hook fired by the disconnect() we just did,
                # can land inside these awaits. Unverified, the lines below
                # would publish the resume command, open a terminal and say
                # "yours in the terminal" over a worker recorded CLOSED.
                w._apply("detached", {"session_id": w.session_id})
                if w.machine.base != DETACHED:
                    raise RuntimeError(
                        f"the session went {w.machine.base} mid-handoff")
            except Exception as e:  # noqa: BLE001 — a half-dead session must never detach
                w._apply("lost", {"reason": f"handoff failed: {e}"})
                # Through the writer, like every other append: it never raises
                # and never blocks the loop, and handoff_failed is an fsync
                # kind the next boot reads.
                self._log_writer.append("handoff_failed",
                                        {"worker": w.id, "path": w.path,
                                         "reason": str(e)})
                if w.machine.base == CLOSED:
                    # `lost` bounced too: the session really did end under the
                    # lockout. "Marked unknown" would be the same lie pointing
                    # the other way, over a tile that reads CLOSED.
                    return {"ok": False,
                            "spoken": (f"{w.project}'s session closed while I "
                                       f"was handing it over, sir — I did not "
                                       f"open a terminal on it.")}
                return {"ok": False,
                        "spoken": (f"The handoff failed, sir — {w.project} is "
                                   f"marked unknown, and I did not open a "
                                   f"terminal on it.")}
            cmd = (f"cd {shlex.quote(w.worktree.path)} && "
                   f"claude --resume {shlex.quote(w.session_id)}")
            # Published BEFORE the launch: if osascript hangs or is missing,
            # the command must already be on screen for Keke to run by hand.
            self._bus.publish("fleet.handoff",
                              {"worker": w.id, "project": w.project,
                               "path": w.path, "command": cmd})
            try:
                await self._open_terminal(cmd)
                spoken = f"{w.project} is yours in the terminal, sir."
            except Exception:  # noqa: BLE001 — the session is detached either way
                spoken = (f"{w.project} is detached, sir — I couldn't open a "
                          f"terminal, so run the command on screen to pick "
                          f"it up.")
            # Cleared BEFORE the drain, not by the finally after it:
            # _admit_next spawns the next queued worker under SPAWN_TIMEOUT_S,
            # and for that whole minute steer() — which checks this flag before
            # the DETACHED branch — would say "I'm handing X over right now"
            # about a worker that is already detached with its terminal open.
            w.handoff_in_flight = False
            await self._admit_next()      # DETACHED freed the slot
            # No snapshot here either, for the same reason as stop(): the
            # `detached` record above is already fsync'd, and it is the exact
            # fact the next boot needs in order to leave this worker alone.
            return {"ok": True, "command": cmd, "spoken": spoken}
        finally:
            w.handoff_in_flight = False

    async def _admit_next(self) -> None:
        # No await between this capacity check, the pop, and _spawn's own
        # synchronous slot reservation — so a tick firing during stop's
        # _admit_next cannot pop and spawn a second queued item: whichever
        # runs second sees the reserved slot and returns without popping.
        if not self.queue or (len(self.live) + self._pending_spawns
                              >= self.max_workers):
            return
        project, path, task = self.queue.popleft()
        spoken = await self._spawn(project, path, task, requeue_front=True)
        self._bus.publish("fleet.spoken",
                          {"text": f"Next in the queue, sir: {spoken}"})

    # ---------- reporting ----------
    def status_line(self) -> str:
        if not self.workers and not self.queue:
            return "The fleet is empty, sir."
        now = self._now()
        # A worker mid-start reads "starting", not the machine's pre-spawned
        # UNKNOWN — "unknown" is an alarm word reserved for failed probes.
        parts = [f"{w.project} is "
                 + ("starting" if w.starting
                    else w.machine.state(now).replace('_', ' ').lower())
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
        # Same treatment as status_line: a worker mid-start reads "starting",
        # never the machine's pre-spawned UNKNOWN — "unknown" is an alarm word
        # reserved for failed probes, and this line is spoken aloud.
        state = ("starting" if w.starting
                 else w.machine.state(self._now()).replace("_", " ").lower())
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
        if kind == "permission_wait":
            # Fleet-owned worker: can_use_tool is the ONLY permission source,
            # so a hook-sourced Notification is dropped UNCONDITIONALLY. It
            # can land after permission_done — a stale POST that raced a
            # stream death, or a dead stream with a live CLI hitting a real
            # prompt — and would park the tile on WAITING_PERMISSION forever:
            # the state never decays because only a real approval's TTL
            # delivers permission_done, and a hook-only wait has no router
            # card, no future, and no TTL. A dead-consumer worker is rescued
            # by tick()'s probe, never by trusting this POST. (Post-handoff
            # the machine bounces everything but session_end off DETACHED,
            # so keeping the hook there bought nothing anyway.)
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
            # A LOCKED worker's consumer is dead by our own hand — shutdown()
            # cancelled it — so its "probe failure" is not news, it is the
            # teardown we asked for. It also lands mid-handoff: the ticker
            # fires every 5s and the lockout awaits an interrupt for up to 10,
            # so escalating here would shout a false alarm on the console and
            # drag the worker to UNKNOWN in the middle of the sequence. Every
            # locked path (stop, close_all, a failed handoff) already ends in
            # an honest state of its own.
            consumer_dead = (not w.locked
                             and (w.consumer is None or w.consumer.done()))
            if state == QUIET and consumer_dead:
                state = w.machine.probe_failed(now)
                self._bus.publish("fleet.error", {
                    "worker": w.id, "project": w.project,
                    "reason": "health probe failed — worker marked unknown"})
            elif (w.machine.base == WAITING_PERMISSION and consumer_dead
                  and not any(not f.done() for f in w._futures.values())):
                # A permission wait NOTHING can resolve: the consumer is
                # dead and no approval future is pending, so no TTL will
                # ever deliver permission_done and the state never decays.
                # Only a failed probe may honestly reclassify it.
                state = w.machine.probe_failed(now)
                self._bus.publish("fleet.error", {
                    "worker": w.id, "project": w.project,
                    "reason": ("health probe failed — permission wait had "
                               "no resolver; worker marked unknown")})
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
        # Flush BEFORE compacting: a snapshot that rotates the log while
        # records are still queued would push them into a fresh generation
        # behind the snapshot's own seq — harmless for recovery, but it makes
        # what is on disk depend on drain timing. Then compact, then flush
        # again, because the compaction rides the same queue and a drain task
        # created after the last flush would be cancelled by the shutdown.
        await self._log_writer.flush()    # orderly shutdown keeps its records
        with contextlib.suppress(Exception):
            self.snapshot()
        await self._log_writer.flush()

    # ---------- restart honesty ----------
    def snapshot(self) -> None:
        """Compaction point: current worker facts through the durable log.

        Called ONCE per server lifetime, from close_all. FleetLog.snapshot
        ROTATES the log and keeps exactly ONE generation (`.jsonl.1`), so each
        compaction destroys the one before it: snapshotting on every stop and
        handoff as well would rotate a finished worker's whole event history
        aside and then overwrite it at shutdown seconds later. Those paths need
        nothing from a snapshot anyway — `session_end` and `detached` are
        fsync'd kinds, already on the platter before either method speaks.

        `machine.base`, never `state(now)`: QUIET is a DERIVED display state
        that decays out of ACTIVE_TURN by the clock alone, and writing it down
        would record a worker as quiet in a file read minutes or days later,
        by a process with no clock context at all. Bases are what recover()
        maps: anything live becomes UNKNOWN/INTERRUPTED, which is exactly what
        close_all's refusal to log session_end intends."""
        # An empty or absent log has nothing to compact, and rotating it is not
        # a compaction — it is a deletion of the previous generation. A JARVIS
        # session that never spawned a worker must not erase the last one that
        # did.
        try:
            if self._log.path.stat().st_size == 0:
                return
        except OSError:
            return
        workers = {w.id: {"worker": w.id, "project": w.project, "path": w.path,
                          "state": w.machine.base, "task": w.task_text,
                          "worktree": w.worktree.path,
                          "session_id": w.session_id or ""}
                   for w in self.workers}
        self._log_writer.snapshot({"workers": workers})

    def recover(self) -> list[dict]:
        """Replay snapshot + log and report what a restart cannot know.

        Honesty rules (spec §5): a worker whose last recorded state was live
        is now UNKNOWN with interrupted=True — its process, stream, and every
        approval callback died with the old server, so nothing is re-armed
        and nothing is claimed to be waiting. DETACHED stays DETACHED
        (another driver owns it); CLOSED stays gone. Ghost tiles are published
        so the console shows the wreckage, and fleet.recovered lets the brain
        say it out loud.

        Ghosts are deliberately NOT registered as Workers. A hand-built Worker
        would have no client, no consumer and no pump, so steer() would answer
        "Told X, sir" into an inbox nothing drains, handoff() would open a
        terminal on a session we do not hold, and its UNKNOWN would count into
        `live` and block every future spawn against a subprocess that provably
        died with the old server. They live in `self.ghosts`, which GET /fleet
        serves beside the live workers and handoff() refuses by name.

        The fold trusts the `state` each record CARRIES, because every record
        is written by _apply AFTER the machine has decided — so a `lost` that
        bounced off CLOSED is recorded CLOSED, and a handoff that lost its
        session mid-flight reads as closed rather than as a terminal somebody
        owns. Nothing here re-runs the state machine: replaying a
        WAITING_PERMISSION verbatim would resurrect a state that never decays,
        waiting on a future that died with the process."""
        folded: dict[str, dict] = {}
        snap = self._log.load_snapshot()
        if snap:
            for wid, info in (snap.get("state", {}).get("workers", {}) or {}).items():
                if isinstance(info, dict):
                    folded[wid] = dict(info)
        records, torn = self._log.replay()
        # A tail torn by a power cut is repaired (and quarantined) by
        # FleetLog's constructor, so replay() now reports a clean file. The
        # log remembers that it was torn when this process opened it; without
        # that, recovery would report a whole log as verified when its last
        # records are sitting in a `.torn-*` file nobody read.
        torn = bool(torn or getattr(self._log, "torn_on_open", False))
        for rec in records:
            data = rec.get("data", {})
            wid = data.get("worker")
            if not wid:
                continue
            slot = folded.setdefault(wid, {})
            for key in ("project", "path", "state", "task", "worktree",
                        "session_id"):
                if data.get(key):
                    slot[key] = data[key]
        reports: list[dict] = []
        for wid, info in folded.items():
            last = info.get("state")
            if last == CLOSED:
                continue
            state = DETACHED if last == DETACHED else UNKNOWN
            report = {"worker": wid, "project": info.get("project", "?"),
                      "path": info.get("path", ""), "state": state,
                      "task": info.get("task", ""),
                      "interrupted": last != DETACHED, "torn_log": torn}
            reports.append(report)
            self._bus.publish("fleet.update",
                              {**report, "worktree": info.get("worktree", "")})
        interrupted = sum(1 for r in reports if r["interrupted"])
        if interrupted:
            self._bus.publish("fleet.recovered", {"count": interrupted})
        self.ghosts = reports
        return reports
