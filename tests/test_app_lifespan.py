"""Lifespan wiring: the fleet ticker actually runs, its death is reported,
shutdown cleanup survives a brain task that died with a real exception, and a
first boot with nothing confirmed discovers projects and ASKS about one.

TestClient's context manager drives the real lifespan (startup + shutdown) in
a portal thread — the same path uvicorn takes in test_app_auth's live-server
tests, minus the socket.
"""
import time as time_mod

from fastapi.testclient import TestClient

from server.app import create_app
from server.registry import Project, Registry


def _ring_events(app, type_):
    return [e for e in list(app.state.bus._ring) if e["type"] == type_]


class _RecordingSpeaker:
    """Stands in for SpeakEngine so a boot test can assert what was SAID
    without driving `say` or ElevenLabs. Installed before the TestClient
    context, because the lifespan reads app.state.speaker when it starts."""

    def __init__(self):
        self.spoke = []

    async def speak(self, text):
        self.spoke.append(text)

    async def preconnect(self):
        pass


def _home_with_a_repo(tmp_path):
    """A home directory discovery can find exactly one project in."""
    home = tmp_path / "home"
    (home / "alethic" / ".git").mkdir(parents=True)
    return home


def _booted(app, spk, until, timeout=5.0):
    with TestClient(app, base_url="http://127.0.0.1:7777") as client:
        assert client.get("/health").json() == {"ok": True}   # never blocked
        deadline = time_mod.time() + timeout
        while time_mod.time() < deadline and not until():
            time_mod.sleep(0.01)
    return spk.spoke


def test_a_first_boot_discovers_and_speaks_the_repo_question(tmp_path, monkeypatch):
    """Beat 1. Nothing in the running server used to call Onboarding.refresh()
    or ask_next(), so the registry stayed empty forever and the question the
    brain knows how to answer was never asked."""
    monkeypatch.setattr("server.app.default_home",
                        lambda: _home_with_a_repo(tmp_path))
    app = create_app(base_dir=tmp_path)
    app.state.speaker = spk = _RecordingSpeaker()
    spoke = _booted(app, spk, lambda: any("correct repo" in s for s in spk.spoke))
    assert any("alethic" in s and "correct repo" in s for s in spoke), spoke
    assert _ring_events(app, "confirm.request")
    assert [p.name for p in app.state.registry.projects] == ["alethic"]
    assert app.state.onboarding.awaiting          # the answer is owned


def test_a_boot_with_a_confirmed_project_asks_nothing(tmp_path, monkeypatch):
    """Do not re-discover on every boot and pester Keke about repos he has
    already answered for. The trigger is 'nothing confirmed yet'."""
    monkeypatch.setattr("server.app.default_home",
                        lambda: _home_with_a_repo(tmp_path))
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    Registry([Project(name="soccer", path="/p/soccer", confirmed=True)]).save(
        tmp_path / "config" / "projects.json")
    app = create_app(base_dir=tmp_path)
    app.state.speaker = spk = _RecordingSpeaker()
    spoke = _booted(app, spk, lambda: bool(spk.spoke), timeout=1.0)
    assert _ring_events(app, "confirm.request") == []
    assert spoke == []
    assert [p.name for p in app.state.registry.projects] == ["soccer"]  # no rescan


def test_a_discovery_fault_is_spoken_and_never_blocks_boot(tmp_path):
    """Discovery walks a real home directory: it can raise, and a raise must
    cost the question, not the server."""
    app = create_app(base_dir=tmp_path)

    async def boom(home):
        raise RuntimeError("scan exploded")

    app.state.onboarding.refresh = boom
    app.state.speaker = spk = _RecordingSpeaker()
    spoke = _booted(app, spk, lambda: bool(spk.spoke))
    assert any("projects" in s.lower() for s in spoke), spoke
    assert any("discovery failed" in e["data"]["reason"]
               for e in _ring_events(app, "butler.error"))


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
