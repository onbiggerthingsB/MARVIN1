"""Lifespan wiring: the fleet ticker actually runs, its death is reported,
and shutdown cleanup survives a brain task that died with a real exception.

TestClient's context manager drives the real lifespan (startup + shutdown) in
a portal thread — the same path uvicorn takes in test_app_auth's live-server
tests, minus the socket.
"""
import time as time_mod

from fastapi.testclient import TestClient

from server.app import create_app


def _ring_events(app, type_):
    return [e for e in list(app.state.bus._ring) if e["type"] == type_]


def test_lifespan_starts_the_fleet_ticker(tmp_path, monkeypatch):
    """Pin the ticker's existence by observing REAL ticks. Deleting the
    lifespan's create_task(_fleet_ticker()) silently disables the only rescue
    for a resolver-less WAITING_PERMISSION worker and the only drain for a
    queued project stranded behind zero live workers — and every other test
    still passes. This one must not."""
    monkeypatch.setattr("server.app.FLEET_TICK_S", 0.01, raising=False)
    app = create_app(base_dir=tmp_path)
    ticks = []

    async def counting_tick(now=None):
        ticks.append(1)

    app.state.fleet.tick = counting_tick
    with TestClient(app, base_url="http://127.0.0.1:7777"):
        deadline = time_mod.time() + 2
        while not ticks and time_mod.time() < deadline:
            time_mod.sleep(0.01)
    assert ticks, "the lifespan never ticked the fleet"


def test_a_dead_ticker_is_reported_not_silent(tmp_path, monkeypatch):
    """tick() faults are survived inside the loop; anything that escapes it
    kills the task. Without a done-callback both safety nets go dark with no
    trace — the same treatment the brain task already has."""
    monkeypatch.setattr("server.app.FLEET_TICK_S", 0.01, raising=False)
    app = create_app(base_dir=tmp_path)

    class _PastTheGuard(BaseException):
        """Not an Exception: sails past the ticker's per-tick guard."""

    async def dying_tick(now=None):
        raise _PastTheGuard("tick exploded past the guard")

    app.state.fleet.tick = dying_tick
    with TestClient(app, base_url="http://127.0.0.1:7777"):
        deadline = time_mod.time() + 2
        while time_mod.time() < deadline:
            if any("ticker died" in e["data"]["reason"]
                   for e in _ring_events(app, "fleet.error")):
                break
            time_mod.sleep(0.01)
    assert any("ticker died" in e["data"]["reason"]
               for e in _ring_events(app, "fleet.error"))


def test_shutdown_cleanup_runs_even_when_the_brain_task_died(tmp_path, monkeypatch):
    """A brain task that ended with a real exception re-raises it from the
    shutdown's `await t`; suppressing only CancelledError skipped close_all()
    (log flush, orderly worker shutdown) and butler.close()."""
    async def dying_brain(*args, **kwargs):
        raise RuntimeError("brain exploded")

    monkeypatch.setattr("server.app.run_butler_brain", dying_brain)
    app = create_app(base_dir=tmp_path)
    closed = []

    async def recording_close_all():
        closed.append("fleet")

    app.state.fleet.close_all = recording_close_all
    with TestClient(app, base_url="http://127.0.0.1:7777"):
        pass
    assert closed == ["fleet"], "close_all was skipped on a dirty brain exit"
