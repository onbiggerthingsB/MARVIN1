import asyncio
from types import SimpleNamespace

from server.app_brain import run_butler_brain
from server.bus import EventBus
from server.fleet import Fleet
from server.fleet_log import FleetLog
from server.fleet_state import DETACHED, UNKNOWN
from server.router import Router
from tests.test_fleet_wiring import (FakeButler, FakeSpeaker, FakeTurnLog,
                                     confirmed_registry)


def fresh_fleet(tmp_path, bus=None, router=None):
    return Fleet(bus=bus or EventBus(), router=router or Router(),
                 log=FleetLog(tmp_path / "fleet.jsonl"),
                 worktrees_dir=tmp_path / "wts")


def seed_log(tmp_path, *kinds_and_data):
    log = FleetLog(tmp_path / "fleet.jsonl")
    for kind, data in kinds_and_data:
        log.append(kind, data)
    return log


W1 = {"worker": "w1", "project": "soccer", "path": "/p/soccer",
      "task": "fix login", "worktree": "/wt/soccer-1"}


def test_restart_reports_interrupted_unknown_never_alive(tmp_path):
    seed_log(tmp_path,
             ("spawned", {**W1, "state": "IDLE_AT_PROMPT"}),
             ("prompt", {**W1, "state": "ACTIVE_TURN"}))
    fleet = fresh_fleet(tmp_path)
    reports = fleet.recover()
    assert len(reports) == 1
    r = reports[0]
    assert r["state"] == UNKNOWN and r["interrupted"] is True
    assert r["project"] == "soccer" and r["path"] == "/p/soccer"
    assert fleet.ghosts == reports                 # served by GET /fleet


def test_closed_workers_are_not_resurrected(tmp_path):
    seed_log(tmp_path,
             ("spawned", {**W1, "state": "IDLE_AT_PROMPT"}),
             ("session_end", {**W1, "state": "CLOSED"}))
    assert fresh_fleet(tmp_path).recover() == []


def test_detached_workers_stay_detached(tmp_path):
    seed_log(tmp_path,
             ("spawned", {**W1, "state": "IDLE_AT_PROMPT"}),
             ("detached", {**W1, "state": "DETACHED", "session_id": "s-1"}))
    reports = fresh_fleet(tmp_path).recover()
    assert reports[0]["state"] == DETACHED
    assert reports[0]["interrupted"] is False      # someone else is driving it


def test_a_pending_approval_is_never_claimed_alive_after_restart(tmp_path):
    # The worker died mid-permission-wait. The callback future is gone with
    # the process: recovery must NOT re-open the approval.
    seed_log(tmp_path,
             ("spawned", {**W1, "state": "IDLE_AT_PROMPT"}),
             ("permission_wait", {**W1, "state": "WAITING_PERMISSION",
                                  "nonce": "n-1", "tool": "Bash"}))
    router = Router()
    fleet = fresh_fleet(tmp_path, router=router)
    reports = fleet.recover()
    assert reports[0]["state"] == UNKNOWN          # not WAITING — nobody is waiting
    assert router.pending_approvals() == []        # nothing voice-addressable


def test_recovery_reads_the_snapshot_plus_the_tail(tmp_path):
    log = seed_log(tmp_path, ("spawned", {**W1, "state": "IDLE_AT_PROMPT"}))
    log.snapshot({"workers": {"w1": {**W1, "state": "IDLE_AT_PROMPT"}}})
    log.append("prompt", {**W1, "state": "ACTIVE_TURN"})   # tail after rotation
    reports = fresh_fleet(tmp_path).recover()
    assert len(reports) == 1 and reports[0]["state"] == UNKNOWN


def test_a_torn_log_still_recovers_the_prefix(tmp_path):
    seed_log(tmp_path, ("spawned", {**W1, "state": "IDLE_AT_PROMPT"}))
    with open(tmp_path / "fleet.jsonl", "a", encoding="utf-8") as f:
        f.write('{"v": 1, "seq"')                  # power cut mid-write
    reports = fresh_fleet(tmp_path).recover()
    assert len(reports) == 1 and reports[0]["torn_log"] is True


def test_a_torn_log_with_nothing_left_to_report_still_says_so(tmp_path):
    """A tear early enough to swallow every record leaves reports == [] — no
    tile, no fleet.recovered, and, before this, nothing spoken: a boot
    indistinguishable from a clean one with no workers. That is the exact
    silence create_app's exception path refuses ("silence is indistinguishable
    from all clear"); torn_on_open is known at boot regardless of how many
    reports came out of it."""
    (tmp_path / "fleet.jsonl").write_bytes(b'{"v": 1, "seq"')   # nothing usable
    bus = EventBus()
    fleet = fresh_fleet(tmp_path, bus=bus)
    assert fleet._log.torn_on_open is True
    cid, q = bus.subscribe()
    assert fleet.recover() == []                   # nothing survived the tear
    events = []
    while not q.empty():
        ev = q.get_nowait()
        if ev:
            events.append((ev["type"], ev["data"]))
    spoken = [d["text"] for t, d in events if t == "fleet.spoken"]
    assert any("log was damaged" in s for s in spoken)
    assert any(t == "fleet.error" for t, _ in events)


def test_a_ghost_survives_a_clean_shutdown_that_spawns_nothing(tmp_path):
    """Run 1 crashes with a worker live; run 2 boots, reports the ghost, spawns
    nothing and shuts down CLEANLY. close_all compacts — the ghost's records go
    to `.jsonl.1`, which recover() never reads — so without carrying the ghost
    into the snapshot, run 3 says nothing at all about a worktree that still
    holds uncommitted work."""
    wt = tmp_path / "wt-a"
    wt.mkdir()                                     # the evidence, still on disk
    seed_log(tmp_path,
             ("spawned", {**W1, "worktree": str(wt), "state": "IDLE_AT_PROMPT"}),
             ("prompt", {**W1, "worktree": str(wt), "state": "ACTIVE_TURN"}))
    run2 = fresh_fleet(tmp_path)
    assert len(run2.recover()) == 1
    asyncio.run(run2.close_all())                  # compaction rotates the log

    run3 = fresh_fleet(tmp_path)
    reports = run3.recover()
    assert len(reports) == 1 and reports[0]["worker"] == "w1"
    assert reports[0]["state"] == UNKNOWN and reports[0]["worktree"] == str(wt)


def test_a_ghost_is_retired_once_its_worktree_is_gone(tmp_path):
    """The worktree IS the evidence, so it is also the retirement rule: seeded
    ghosts that persisted forever with no way to clear them would be the
    opposite failure. Delete the directory (merge-back, or plain cleanup) and
    the next compaction stops carrying it.

    The retirement has to be reachable on a WORKER-FREE boot, and this test
    used to call close_all() on run 2 — the one boot where the log still had
    records, and therefore the only boot on which Fleet.snapshot got past its
    empty-log guard at all. Every boot after that starts with a log that the
    previous compaction rotated away, so the guard returned before the
    retirement loop ever ran and the ghost was immortal. Boot 3 below is that
    boot: it spawns nothing, logs nothing, and must still retire."""
    wt = tmp_path / "wt-a"
    wt.mkdir()
    seed_log(tmp_path,
             ("spawned", {**W1, "worktree": str(wt), "state": "IDLE_AT_PROMPT"}))
    run2 = fresh_fleet(tmp_path)
    assert len(run2.recover()) == 1
    asyncio.run(run2.close_all())                  # compaction rotates the log

    run3 = fresh_fleet(tmp_path)                   # empty log, ghost from .snap
    assert len(run3.recover()) == 1                # still real: worktree exists
    assert run3._log.path.stat().st_size == 0 if run3._log.path.exists() else True
    wt.rmdir()                                     # the human cleaned it up
    asyncio.run(run3.close_all())                  # spawned nothing all boot

    assert fresh_fleet(tmp_path).recover() == []


def test_a_retired_ghost_is_not_re_announced_on_the_next_boot(tmp_path):
    """The consequence the count made visible: `fleet.recovered` fired on every
    boot, so Marvin announced "workers were interrupted by a restart" about a
    worktree that had been deleted several restarts ago — and a DETACHED ghost
    kept offering a `claude --resume` for a session that ended long before."""
    wt = tmp_path / "wt-a"
    wt.mkdir()
    seed_log(tmp_path,
             ("spawned", {**W1, "worktree": str(wt), "state": "IDLE_AT_PROMPT"}),
             ("prompt", {**W1, "worktree": str(wt), "state": "ACTIVE_TURN"}))
    counts = []
    for boot in range(4):
        bus = EventBus()
        fleet = fresh_fleet(tmp_path, bus=bus)
        cid, q = bus.subscribe()
        fleet.recover()
        events = []
        while not q.empty():
            ev = q.get_nowait()
            if ev:
                events.append(ev)
        bus.unsubscribe(cid)
        counts.append(sum(e["data"]["count"] for e in events
                          if e["type"] == "fleet.recovered"))
        if boot == 1:
            wt.rmdir()                             # cleaned up after boot 2
        asyncio.run(fleet.close_all())
    # boots 1 and 2 legitimately report it (the worktree still holds work);
    # boot 3 is the compaction that retires it; boot 4 has nothing to say.
    assert counts[:2] == [1, 1]
    assert counts[3] == 0


def test_a_detached_ghost_keeps_the_command_that_rejoins_it(tmp_path):
    """The session id lives nowhere but the log, and `claude --resume` is the
    one thing that would let Keke rejoin that session. Folding it into the slot
    and then dropping it from the report leaves the console's .tile-resume line
    empty and the ghost-handoff refusal with nothing to offer."""
    seed_log(tmp_path,
             ("spawned", {**W1, "state": "IDLE_AT_PROMPT"}),
             ("detached", {**W1, "state": "DETACHED", "session_id": "s-1"}))
    fleet = fresh_fleet(tmp_path)
    report = fleet.recover()[0]
    assert report["state"] == DETACHED
    assert "claude --resume s-1" in report["command"]
    assert "/wt/soccer-1" in report["command"]
    result = asyncio.run(fleet.handoff("/p/soccer"))
    assert result["ok"] is False                   # still never a second driver
    assert "claude --resume s-1" in result["command"]


def test_an_interrupted_ghost_is_offered_no_resume_command(tmp_path):
    """UNKNOWN means nobody knows what became of it — a resume command would
    invite Keke to drive something Marvin just said it cannot vouch for."""
    seed_log(tmp_path,
             ("spawned", {**W1, "state": "IDLE_AT_PROMPT",
                          "session_id": "s-9"}))
    report = fresh_fleet(tmp_path).recover()[0]
    assert report["state"] == UNKNOWN and "command" not in report


async def test_a_ghost_is_refused_a_terminal_by_name(tmp_path):
    """A ghost is not a Worker, so _find cannot see it and handoff's every
    gate (spawn_in_flight, session_id, DETACHED) is aimed at live workers.
    Without its own gate the console's still-visible button would answer a
    restart ghost with a bare "nothing is running there" — and, worse, any
    future ghost registered as a Worker would open a terminal on a session
    that died with the old process."""
    seed_log(tmp_path,
             ("spawned", {**W1, "state": "IDLE_AT_PROMPT"}),
             ("prompt", {**W1, "state": "ACTIVE_TURN"}))
    fleet = fresh_fleet(tmp_path)
    fleet.recover()
    result = await fleet.handoff("/p/soccer")
    assert result["ok"] is False
    assert "command" not in result                  # nothing to resume, ever
    assert "didn't survive the restart" in result["spoken"]
    assert "soccer" in result["spoken"]
    # an unknown path is still the plain sentence
    other = await fleet.handoff("/p/nowhere")
    assert other["spoken"] == "Nothing is running there, sir."


async def test_a_detached_ghost_is_told_it_already_has_its_terminal(tmp_path):
    seed_log(tmp_path,
             ("spawned", {**W1, "state": "IDLE_AT_PROMPT"}),
             ("detached", {**W1, "state": "DETACHED", "session_id": "s-1"}))
    fleet = fresh_fleet(tmp_path)
    fleet.recover()
    result = await fleet.handoff("/p/soccer")
    assert result["ok"] is False and "already detached" in result["spoken"]


def test_get_fleet_serves_live_workers_and_ghosts_to_a_cookie(tmp_path):
    """The console reconnects to /events WITHOUT a Last-Event-ID, so the
    boot-time ghost tiles are published before any browser exists and this
    route is the ONLY way they ever reach the page."""
    from tests.test_app_auth import bootstrap, make_client
    log = FleetLog(tmp_path / "state" / "fleet.jsonl")
    log.append("spawned", {**W1, "state": "IDLE_AT_PROMPT"})
    log.append("prompt", {**W1, "state": "ACTIVE_TURN"})
    c = make_client(tmp_path)                       # create_app boots on tmp_path
    assert c.get("/fleet").status_code == 401       # cookie-authed like the rest
    hdrs = {"Authorization": f"Bearer {c.app.state.cfg.hook_bearer}"}
    assert c.get("/fleet", headers=hdrs).status_code == 401   # never the hook lane
    bootstrap(c)
    with c:                                         # drives the real lifespan
        # A live worker beside the ghost: the route serves the UNION, and it
        # must read the live half through machine.state(now) — the DERIVED
        # display state, so a worker gone QUIET renders quiet — never .base.
        c.app.state.fleet.workers.append(SimpleNamespace(
            id="live-1", project="tibet", path="/p/tibet", task_text="chart it",
            worktree=SimpleNamespace(path="/wt/tibet"), starting=False,
            machine=SimpleNamespace(state=lambda now: "ACTIVE_TURN")))
        try:
            body = c.get("/fleet").json()
        finally:
            # Out again before shutdown: close_all drives real Workers, and a
            # stub left in the list would take the lifespan's teardown with it.
            c.app.state.fleet.workers.clear()
    workers = body["workers"]
    assert [w["worker"] for w in workers] == ["live-1", "w1"]
    live, ghost = workers
    assert live["state"] == "ACTIVE_TURN" and live["worktree"] == "/wt/tibet"
    assert "interrupted" not in live                # only ghosts carry it
    assert ghost["state"] == UNKNOWN and ghost["interrupted"] is True
    assert ghost["project"] == "soccer"


def test_get_fleet_serves_pending_approval_cards_to_a_cookie(tmp_path):
    """The console loses CARDS on an SSE drop and never resyncs: it reconnects
    without a Last-Event-ID and /fleet was fetched exactly once at setup. A
    worker could sit blocked for the full 600s TTL with no card on screen and
    no spoken line. Same cookie gate as the rest of the route — never the hook
    bearer, which lives in every worktree."""
    import json
    import time

    from tests.test_app_auth import bootstrap, make_client
    c = make_client(tmp_path)
    with c:
        a = c.app.state.router.open_approval(
            "soccer", "Bash: rm -rf …", now=time.time(), path="/p/soccer")
        card = {"nonce": a.nonce, "worker": "w9", "project": "soccer",
                "path": "/p/soccer", "tool": "Bash", "args": "rm -rf …",
                "full_args": "rm -rf /Users/likerun/Documents",
                "risk": "Careful, sir — this one can destroy things.",
                "outside": "That target is inside your Obsidian vault, sir.",
                "worktree": "/wt/soccer-1", "question": "…"}
        c.app.state.fleet.workers.append(SimpleNamespace(
            id="w9", project="soccer", path="/p/soccer", task_text="t",
            worktree=SimpleNamespace(path="/wt/soccer-1"), starting=False,
            machine=SimpleNamespace(state=lambda now: "WAITING_PERMISSION"),
            _cards={a.nonce: card}))
        try:
            assert c.get("/fleet").status_code == 401      # cookie required
            bootstrap(c)
            body = c.get("/fleet").json()
        finally:
            c.app.state.fleet.workers.clear()
    served = body["approvals"]
    assert len(served) == 1
    assert served[0]["full_args"] == "rm -rf /Users/likerun/Documents"
    assert "destroy things" in served[0]["risk"]
    assert "Obsidian vault" in served[0]["outside"]
    assert c.app.state.cfg.hook_bearer not in json.dumps(body)   # never the bearer
    assert "session_id" not in json.dumps(served)                # nor a session id


def test_get_fleet_never_paints_a_starting_worker_unknown(tmp_path):
    """`base` is UNKNOWN until start() applies `spawned`, and the spawn window
    is up to 60 seconds long. _spawn pre-seeds published_state, and both
    status_line() and one_breath() special-case w.starting, precisely because
    "unknown" is an alarm word reserved for failed probes — a page load
    mid-spawn must not paint a healthy worker as an UNKNOWN tile with a live
    handoff button."""
    from tests.test_app_auth import bootstrap, make_client
    c = make_client(tmp_path)
    bootstrap(c)
    with c:
        c.app.state.fleet.workers.append(SimpleNamespace(
            id="live-2", project="soccer", path="/p/soccer", task_text="fix it",
            worktree=SimpleNamespace(path="/wt/soccer"), starting=True,
            machine=SimpleNamespace(state=lambda now: UNKNOWN)))
        try:
            body = c.get("/fleet").json()
        finally:
            c.app.state.fleet.workers.clear()
    assert body["workers"][0]["state"] == "STARTING"


def test_boot_recovery_runs_in_the_lifespan(tmp_path):
    """Deleting recover() from the lifespan disables restart honesty entirely
    and every other test still passes. This one must not: it watches the real
    startup publish the ghost tile and the spoken report."""
    from tests.test_app_auth import make_client
    log = FleetLog(tmp_path / "state" / "fleet.jsonl")
    log.append("spawned", {**W1, "state": "IDLE_AT_PROMPT"})
    c = make_client(tmp_path)
    with c:
        ring = list(c.app.state.bus._ring)
    tiles = [e["data"] for e in ring if e["type"] == "fleet.update"]
    recovered = [e["data"] for e in ring if e["type"] == "fleet.recovered"]
    assert [t["worker"] for t in tiles] == ["w1"]
    assert tiles[0]["state"] == UNKNOWN and tiles[0]["worktree"] == "/wt/soccer-1"
    assert recovered == [{"count": 1}]


def test_a_recovery_that_explodes_is_reported_and_never_blocks_boot(tmp_path):
    """An unreadable log must not stop the boot — and must not pass in
    SILENCE either. Coming up with no idea what was running and saying
    nothing is indistinguishable, to Keke, from "all clear"."""
    from tests.test_app_auth import make_client
    c = make_client(tmp_path)

    def boom():
        raise RuntimeError("log unreadable")

    c.app.state.fleet.recover = boom
    with c:
        assert c.get("/health").json()["ok"] is True   # the server still came up
        ring = list(c.app.state.bus._ring)
    assert any(e["type"] == "fleet.error"
               and "recovery failed" in e["data"]["reason"] for e in ring)
    spoken = [e["data"]["text"] for e in ring if e["type"] == "fleet.spoken"]
    assert any("couldn't read my own fleet log" in s for s in spoken)


async def test_the_brain_speaks_the_recovery_report(tmp_path):
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=confirmed_registry("soccer")))
    await asyncio.sleep(0)
    bus.publish("fleet.recovered", {"count": 2})
    await asyncio.sleep(0.05)
    assert any("2 workers were interrupted" in s for s in spk.spoke)
    assert not task.done()
    task.cancel()
