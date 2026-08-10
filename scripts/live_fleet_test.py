"""Live fleet smoke: a REAL worker on a throwaway repo, through the real SDK.

Run:      cd ~/marlowe && uv run python scripts/live_fleet_test.py
Needs:    working claude auth (subscription or ANTHROPIC_API_KEY), and this
          machine's proxy env (HTTPS_PROXY/HTTP_PROXY + NO_PROXY=localhost,127.0.0.1).
Note:     the worktree's hook curls target 127.0.0.1:7777 with an EMPTY bearer,
          so they cannot be accepted even if the real Marlowe is running there —
          this script never touches a live fleet. Hook delivery is verified in
          the milestone gate, not here.

Not a pytest: it costs real tokens and real minutes, and it is run
deliberately. `testpaths = ["tests"]` keeps `uv run pytest` away from it.

The repo under test is created fresh in a temp directory and thrown away by
the operating system; the Fleet is additionally built with the vault and the
Marlowe checkout in `forbidden`, so even an edited copy of this script cannot
aim a worker at either.

PASS criteria, printed at the end:
  1. spawn spoke "On it" and the worktree exists at the recorded commit
  2. the tool approval arrived, was auto-"voice"-approved, and DONE.txt
     appeared in the WORKTREE and NOT in the origin repo
  3. steer produced a second turn that appended to DONE.txt
  4. stop closed the session and the worktree survives

None of that is a sandbox check, and this script must never imply one. The
worktree is cwd, not a jail: Write/Edit/Bash take absolute paths, and a real
run of this script had the worker's first tool call write /tmp/DONE.txt. The
containment is the SPOKEN approval, so the summary also prints every approval
whose readback said the target was outside the worktree.
"""
import asyncio
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.bus import EventBus            # noqa: E402
from server.fleet import Fleet             # noqa: E402
from server.fleet_log import FleetLog      # noqa: E402
from server.fleet_state import CLOSED, IDLE_AT_PROMPT   # noqa: E402
from server.router import Router           # noqa: E402
from server.vault_paths import vault_root_from_env      # noqa: E402
from server.worktrees import proxy_problem              # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_repo(base: Path) -> Path:
    repo = base / "scratch-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "README.md").write_text("A scratch repo for the Marlowe fleet smoke.\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=j@j", "-c", "user.name=marlowe",
                    "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


async def wait_for(predicate, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(1)
    print(f"TIMEOUT waiting for {what}")
    return False


async def main() -> int:
    # Fail here, loudly, with the sentence Marlowe itself would speak — rather
    # than inside a worker that dies on a bare "403 Request not allowed".
    problem = proxy_problem()
    if problem:
        print(f"REFUSED: I can't spawn safely, sir: {problem}.")
        print("Set HTTPS_PROXY/HTTP_PROXY (FlClash on 127.0.0.1:7890) and "
              "NO_PROXY=localhost,127.0.0.1, then run this again.")
        return 2

    base = Path(tempfile.mkdtemp(prefix="marlowe-fleet-smoke-"))
    repo = make_repo(base)
    bus, router = EventBus(), Router()
    fleet = Fleet(bus=bus, router=router,
                  log=FleetLog(base / "state" / "fleet.jsonl"),
                  worktrees_dir=base / "state" / "worktrees",
                  # Belt and braces: the repo above is a fresh temp checkout,
                  # and these two can never be spawned into regardless.
                  forbidden=(str(vault_root_from_env().resolve()),
                             str(REPO_ROOT)))
    cid, q = bus.subscribe()
    states: list[str] = []
    tools: list[str] = []
    approvals: list[str] = []
    outside: list[str] = []        # approvals that named a target outside the worktree

    pending: set[asyncio.Task] = set()

    async def approve_later(data):
        await asyncio.sleep(2)                  # simulate the voice round-trip
        if router.take_nonce(data["nonce"], time.time()) is not None:
            fleet.deliver_approval(data["nonce"], True)
            print("  [voice] approved")

    async def pump_events():
        while True:
            ev = await q.get()
            if ev is None:
                # The bus drops a subscriber whose queue overflows. Say so —
                # silently stopping would look like a worker that went quiet,
                # and nothing would answer its next approval.
                print("  [bus] subscriber dropped (queue overflow) — "
                      "no further events, and no further approvals")
                return
            kind, data = ev["type"], ev.get("data", {})
            if kind == "fleet.update":
                state = data.get("state", "?")
                if not states or states[-1] != state:
                    states.append(state)
                print(f"  [state] {state}")
            if kind in ("fleet.error", "approval.resolved",
                        "fleet.unknown_session"):
                print(f"  [{kind}] {data}")
            if kind == "fleet.message":
                text = str(data.get("text", ""))
                if text.startswith("["):                 # a ToolUseBlock line
                    tools.append(text.split("]")[0].lstrip("["))
                print(f"  [worker] {text[:120]}")
            if kind == "approval.request":
                approvals.append(f"{data.get('tool')}: {data.get('args')}")
                if "outside its worktree" in str(data.get("question", "")).lower():
                    # The real containment report: what the approval SAID out
                    # loud about targets beyond the disposable directory.
                    outside.append(f"{data.get('tool')}: {data.get('args')}")
                print(f"  [APPROVAL] {data['question']}")
                # In its OWN task: the round-trip delay must not stop this
                # pump draining the bus. A blocked pump overflows the
                # subscriber queue on a chatty worker and gets dropped, and
                # then nothing answers the next approval at all.
                t = asyncio.create_task(approve_later(data))
                pending.add(t)
                t.add_done_callback(pending.discard)

    pump = asyncio.create_task(pump_events())
    ok, wt = True, None
    try:
        spoken = await fleet.spawn(
            "scratch", str(repo),
            "Create a file named DONE.txt containing exactly the word: done")
        print(f"[spawn] {spoken}")
        if not spoken.startswith("On it"):
            print("FAIL: spawn refused")
            return 1
        w = fleet.workers[0]
        wt = Path(w.worktree.path)
        print(f"[worktree] {wt} @ {w.worktree.base_commit[:7]}")

        ok &= await wait_for(lambda: (wt / "DONE.txt").exists(), 240, "DONE.txt")
        ok &= await wait_for(lambda: w.machine.base == IDLE_AT_PROMPT, 120,
                             "idle at prompt")
        # NOT a sandbox check, and it must not be printed as one. This compares
        # the ORIGIN REPO and nothing else: cwd is the worktree, but Write,
        # Edit and Bash take absolute paths, and the run that produced this
        # script's first transcript wrote /tmp/DONE.txt — while this line
        # printed True. The only honest containment claim is the spoken
        # approval, and the `outside` tally below is what reports on it.
        clean_origin = not (repo / "DONE.txt").exists()
        ok &= clean_origin
        print(f"[check] DONE.txt NOT in the origin repo: {clean_origin} "
              f"(says nothing about the rest of the filesystem)")

        print(f"[steer] {fleet.steer_path(str(repo), 'Append a second line to DONE.txt that says: and dusted')}")
        ok &= await wait_for(
            lambda: (wt / "DONE.txt").exists()
            and "and dusted" in (wt / "DONE.txt").read_text(errors="replace"),
            240, "the steered append")

        print(f"[stop] {await fleet.stop(str(repo))}")
        closed = w.machine.base == CLOSED
        preserved = wt.exists()
        ok &= closed and preserved
        print(f"[check] session CLOSED: {closed}; worktree preserved: {preserved}")
        await fleet.close_all()                        # flush the durable log
    finally:
        pump.cancel()
        for t in list(pending):
            t.cancel()
        bus.unsubscribe(cid)

    print("\n--- what happened ---")
    print(f"states:    {' -> '.join(states) or '(none)'}")
    print(f"tools:     {', '.join(tools) or '(none)'}")
    print(f"approvals: {'; '.join(approvals) or '(none — nothing needed one)'}")
    # Observed, never a PASS gate: whether the model reaches for an absolute
    # path is its own choice, and failing the smoke on it would make the smoke
    # flaky. What must hold is that the spoken line SAID so before the yes.
    print(f"outside:   {'; '.join(outside) or '(none — every target named was inside the worktree)'}")
    if approvals and not tools:
        print("NOTE: an approval was asked but no tool call was streamed.")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    print(f"(worktree preserved for inspection: {wt})")
    print(f"(delete the whole scratch tree when done: rm -rf {base})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
