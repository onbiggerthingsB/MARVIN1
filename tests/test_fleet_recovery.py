import asyncio
from pathlib import Path
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
            worktree=SimpleNamespace(path="/wt/tibet"),
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
