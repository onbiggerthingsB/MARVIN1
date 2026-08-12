"""Live handoff smoke: a REAL worker through the real SDK, torn down and
handed to a real second driver — the beat the shipped feature failed 3-for-3
(state/fleet.jsonl seq 55-60: `session_end` from the exiting CLI's own hook
landed before `_apply("detached")`, which bounced off CLOSED).

Run:      cd ~/marlowe && uv run python scripts/live_handoff_test.py
          add --real-terminal to launch Terminal.app via the default
          osascript launcher instead of the recorder (leaves a window open
          driving the session; the script then does NOT resume headlessly —
          two drivers at once is the accident this feature exists to prevent).
Needs:    working claude auth, HTTPS_PROXY/HTTP_PROXY set and NO_PROXY
          covering localhost + 127.0.0.1 (proxy_problem() gates below).

Unlike live_fleet_test.py this script DOES consume the worktree's hook POSTs:
it binds a real /hooks receiver (same shape as server/app.py's, bearer and
all) on 127.0.0.1:7788, so the CLI's SessionEnd hook races the teardown
exactly as it does under the full server. That race IS the defect under test,
so a run without the receiver would prove nothing.

Not a pytest: it costs real tokens and real minutes, and it is run
deliberately. `testpaths = ["tests"]` keeps `uv run pytest` away from it.

PASS criteria, printed at the end:
  1. the worker parked on a PENDING approval (the state real workers hold
     most of the time — where all three live failures happened)
  2. handoff() returned ok with a resume command, the tile reads DETACHED,
     and the durable log shows the CLI's own `session_end` consumed by the
     teardown followed by a `detached` record whose state IS DETACHED
  3. `claude --resume <session-id>` genuinely resumes the session as the one
     new driver (headless -p, in the worktree), while the tile STAYS
     DETACHED through that driver's hook traffic
  4. the second driver's exit — its SessionEnd hook — is what finally moves
     the tile DETACHED -> CLOSED
"""
import asyncio
import json
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn                              # noqa: E402
from fastapi import FastAPI, Request       # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from server.bus import EventBus             # noqa: E402
from server.fleet import Fleet              # noqa: E402
from server.fleet_log import FleetLog       # noqa: E402
from server.fleet_state import (CLOSED, DETACHED,        # noqa: E402
                                WAITING_PERMISSION)
from server.router import Router            # noqa: E402
from server.vault_paths import vault_root_from_env       # noqa: E402
from server.worktrees import proxy_problem  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK_PORT = 7788        # NEVER 7777: a live Marvin may be listening there


def make_repo(base: Path) -> Path:
    repo = base / "scratch-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "README.md").write_text("Scratch repo for the live handoff smoke.\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=j@j", "-c", "user.name=marvin",
                    "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def hooks_app(fleet: Fleet, bearer: str, seen: list) -> FastAPI:
    """The same /hooks surface server/app.py exposes, receiver-side verbatim:
    bearer-authed, JSON body handed to fleet.handle_hook on the loop."""
    app = FastAPI()

    @app.post("/hooks")
    async def hooks(request: Request):
        if request.headers.get("Authorization") != f"Bearer {bearer}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "bad json"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "bad shape"}, status_code=400)
        seen.append((time.time(), str(payload.get("hook_event_name", "?")),
                     str(payload.get("session_id", ""))))
        fleet.handle_hook(payload)
        return {"ok": True}

    return app


async def wait_for(predicate, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.5)
    print(f"TIMEOUT waiting for {what}")
    return False


async def main() -> int:
    problem = proxy_problem()
    if problem:
        print(f"REFUSED: I can't spawn safely, sir: {problem}.")
        return 2
    real_terminal = "--real-terminal" in sys.argv[1:]

    base = Path(tempfile.mkdtemp(prefix="marvin-handoff-smoke-"))
    repo = make_repo(base)
    bus, router = EventBus(), Router()
    bearer = secrets.token_hex(16)
    fleet = Fleet(bus=bus, router=router,
                  log=FleetLog(base / "state" / "fleet.jsonl"),
                  worktrees_dir=base / "state" / "worktrees",
                  forbidden=(str(vault_root_from_env().resolve()),
                             str(REPO_ROOT)),
                  hook_port=HOOK_PORT, hook_bearer=bearer)
    opened: list[str] = []
    if not real_terminal:
        async def record_terminal(cmd):
            opened.append(cmd)
        fleet._open_terminal = record_terminal

    hook_posts: list[tuple[float, str, str]] = []
    server = uvicorn.Server(uvicorn.Config(
        hooks_app(fleet, bearer, hook_posts), host="127.0.0.1",
        port=HOOK_PORT, log_level="warning"))
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.5)                # the receiver must be up first

    cid, q = bus.subscribe()
    states: list[str] = []

    async def pump():
        while True:
            ev = await q.get()
            if ev is None:
                print("  [bus] subscriber dropped (queue overflow)")
                return
            kind, data = ev["type"], ev.get("data", {})
            if kind == "fleet.update":
                state = data.get("state", "?")
                if not states or states[-1] != state:
                    states.append(state)
                    print(f"  [state] {state}")
            elif kind == "approval.request":
                print(f"  [APPROVAL pending, deliberately unanswered] "
                      f"{data.get('tool')}: {data.get('args')}")
            elif kind in ("fleet.error", "fleet.handoff", "approval.resolved"):
                print(f"  [{kind}] {data}")

    pump_task = asyncio.create_task(pump())
    ok = True
    try:
        spoken = await fleet.spawn(
            "scratch", str(repo),
            "Run `echo handoff-proof` with the Bash tool, then write its "
            "output into a file named PROOF.txt")
        print(f"[spawn] {spoken}")
        if not spoken.startswith("On it"):
            print("FAIL: spawn refused")
            return 1
        w = fleet.workers[0]
        wt = Path(w.worktree.path)
        print(f"[worktree] {wt} @ {w.worktree.base_commit[:7]}")

        # 1 — the live failures' exact precondition: a PENDING approval.
        ok &= await wait_for(
            lambda: (w.machine.base == WAITING_PERMISSION
                     and router.pending_approvals() and w.session_id),
            240, "a pending approval + session id")
        print(f"[check] parked on a pending approval, session {w.session_id}")

        # 2 — the handoff itself, straight into the teeth of the race.
        result = await fleet.handoff(str(repo))
        print(f"[handoff] ok={result.get('ok')} spoken={result.get('spoken')}")
        print(f"[handoff] command={result.get('command', '(none)')}")
        handed = bool(result.get("ok")) and w.machine.base == DETACHED \
            and "claude --resume" in result.get("command", "")
        ok &= handed
        print(f"[check] detached with a resume command: {handed}")
        ok &= router.pending_approvals() == []
        session_ends_pre = [p for p in hook_posts if p[1] == "SessionEnd"]
        print(f"[check] teardown SessionEnd hooks consumed pre-detach: "
              f"{len(session_ends_pre)} (echo owed: {w.session_end_echo_owed})")

        if real_terminal:
            print("[note] --real-terminal: a Terminal window now owns the "
                  "session; NOT resuming headlessly on top of it. Inspect "
                  "and close it by hand.")
        else:
            # 3 — the second driver: the resume command, run for real. The
            # driver's OWN hooks must find the tile DETACHED while it lives,
            # so the check has to happen mid-flight: a headless -p driver
            # finishes in one turn, and by the time communicate() returns its
            # SessionEnd has already (correctly) closed the tile.
            sid = w.session_id
            posts_before = len(hook_posts)
            mid_states: list[str] = []

            async def watch_driver():
                while True:
                    fresh = hook_posts[posts_before:]
                    if any(k == "SessionEnd" for _, k, _ in fresh):
                        return          # driver ended: CLOSED is correct now
                    if fresh:
                        mid_states.append(w.machine.base)  # driver is LIVE
                    await asyncio.sleep(0.2)

            watcher = asyncio.create_task(watch_driver())
            proc = await asyncio.create_subprocess_exec(
                "claude", "--resume", sid, "-p",
                "Reply with exactly the single word: RESUMED",
                cwd=str(wt), stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(proc.communicate(), 180)
            except asyncio.TimeoutError:
                proc.kill()
                out, err = b"", b"(timed out)"
            watcher.cancel()
            reply = out.decode(errors="replace").strip()
            print(f"[resume] exit={proc.returncode} reply={reply[:120]!r}")
            if proc.returncode != 0:
                print(f"[resume] stderr: {err.decode(errors='replace')[:400]}")
            ok &= proc.returncode == 0 and bool(reply)
            still_detached = bool(mid_states) and set(mid_states) == {DETACHED}
            print(f"[check] tile read DETACHED at every sample while the "
                  f"second driver's hooks arrived: {still_detached} "
                  f"({len(mid_states)} samples)")

            # 4 — the second driver's own end closes the tile.
            closed = await wait_for(lambda: w.machine.base == CLOSED, 20,
                                    "the terminal driver's SessionEnd")
            # A resumed -p session may end under a NEW session id; cwd
            # matching in handle_hook is what routes it home either way.
            ok &= still_detached and closed
            print(f"[check] second driver's exit closed the tile: {closed}")

        await fleet.close_all()
    finally:
        pump_task.cancel()
        bus.unsubscribe(cid)
        server.should_exit = True
        await asyncio.wait_for(server_task, 5)

    print("\n--- what happened ---")
    print(f"states: {' -> '.join(states) or '(none)'}")
    print(f"hook posts: {', '.join(k for _, k, _ in hook_posts) or '(none)'}")
    print("--- durable log ---")
    # close_all's compaction ROTATES the log: the run's records live in the
    # `.jsonl.1` generation afterwards, not in a fresh (absent) fleet.jsonl.
    log_path = base / "state" / "fleet.jsonl"
    if not log_path.exists():
        log_path = log_path.with_suffix(".jsonl.1")
    kinds_states: list[tuple[str, str]] = []
    for line in log_path.read_text().splitlines():
        rec = json.loads(line)
        d = rec.get("data", {})
        kinds_states.append((rec.get("kind", ""), d.get("state", "")))
        print(f"  seq {rec.get('seq'):>3} {rec.get('kind'):<16} "
              f"{d.get('state', ''):<20} {d.get('reason', '')[:60]}")
    # The durable shape of a REAL handoff, the exact inverse of the shipped
    # failure (which read: session_end CLOSED, detached CLOSED, lost,
    # handoff_failed): the teardown's session_end followed by a `detached`
    # record whose state IS DETACHED — beat 6's first evidence.
    if not real_terminal:
        shape = (("session_end", CLOSED) in kinds_states
                 and ("detached", DETACHED) in kinds_states
                 and all(k != "handoff_failed" for k, _ in kinds_states))
        ok &= shape
        print(f"[check] durable log shows session_end CLOSED then detached "
              f"DETACHED, no handoff_failed: {shape}")
    if not real_terminal and opened:
        print(f"terminal launcher recorded: {opened[0]}")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    print(f"(scratch tree: {base} — delete when done)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
