"""The ninth fail-open: `spoken` was set from a signal that did not mean it.

`Approval.spoken` closed the sixth fail-open — a yes may only resolve a
request the owner actually HEARD — and the brain sets it from _speak()'s
return value, whose docstring said "whether speech actually SUCCEEDED". But
_speak() returned True whenever speaker.speak() did not raise, and BOTH
engines returned normally for speech that was never delivered:

  * `say`: interrupt() (the last console leaving) kills the subprocess
    mid-word; proc.wait() then returns like a clean finish, _say_cut and the
    tts.done `interrupted` field record the cut, and the RETURN PATH consults
    neither. proc.returncode was never checked at all.
  * ElevenLabs: chunks are handed to fire-and-forget send tasks, so isFinal
    can arrive with the room already empty (or having emptied mid-stream),
    a clean socket close without isFinal ends the loop like success, and
    isFinal with zero audio is "success" with nothing ever hearable.

Codex's exploit sequence, reproduced through the real brain + real engine
below: destructive approval pending -> readback starts with a console
present -> the last console leaves and the kill lands -> speak() returns
normally -> the approval is marked spoken -> the owner reconnects, says a
bare "yes", and fleet.deliver_approval(..., True) unblocks the tool.

The fix: SpeechNotDelivered. Both engines decide delivery from evidence
(cut record, exit code, protocol completion, room occupancy re-read AFTER
speech) and raise when the utterance was cut, incomplete, or finished with
nobody there. Every speak() caller already treats a raise as "not
delivered", so `spoken` stays false and the TTL backstops the request.
"""
import asyncio
import time

import pytest

from server.bus import EventBus
from server.router import Router
from server.tts import SpeakEngine, SpeechNotDelivered
from tests.test_approval_correlation import DESTRUCTIVE, card
from tests.test_fleet_wiring import (FakeButler, FakeFleet, brain,
                                     confirmed_registry)


class FakeSay:
    """One controllable `say` subprocess. kill() is what interrupt() calls;
    finish() is the clean end of an utterance."""

    def __init__(self):
        self._done = asyncio.Event()
        self.returncode = None

    async def wait(self):
        await self._done.wait()
        return self.returncode

    def kill(self):
        self.returncode = -9
        self._done.set()

    def finish(self, rc=0):
        self.returncode = rc
        self._done.set()


def _say_factory(monkeypatch):
    """Divert `say`; returns the list of (argv, proc) spawned so far."""
    spawned = []

    async def fake_exec(*argv, **kw):
        proc = FakeSay()
        spawned.append((argv, proc))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return spawned


async def _until(cond, what: str, tries: int = 200):
    for _ in range(tries):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"never observed: {what}")


# =====================================================================
# Codex's sequence, end to end through the real brain and real engine.
# =====================================================================
async def test_a_readback_cut_by_the_last_console_leaving_is_never_marked_spoken(
        monkeypatch):
    """Steps 1-6 of the exploit. Before the fix this failed at step 4 with
    `spoken is True` — and letting it run on, the bare yes delivered the
    destructive tool (("deliver", nonce, True) in fleet.calls)."""
    spawned = _say_factory(monkeypatch)
    bus, butler = EventBus(), FakeButler()
    router, fleet = Router(), FakeFleet()
    # 1. a destructive approval is pending, spoken=False
    a = router.open_approval("soccer", DESTRUCTIVE, now=time.time(),
                             path="/p/soccer")
    present = {"now": True}
    eng = SpeakEngine("", "", "ws://unused", bus.publish,
                      send_audio=lambda b: None,
                      listening=lambda: present["now"])
    task = await brain(bus, butler, eng, router=router,
                       registry=confirmed_registry("soccer"), fleet=fleet)
    # 2. the readback begins while a console is connected (gate passes)
    bus.publish("approval.request", card(
        a, "Careful, sir — this one can destroy things. soccer wants Bash — "
           "rm -rf /Users/likerun/Documents. Approve or deny, sir?"))
    await _until(lambda: eng._say_proc is not None, "the readback in flight")
    # 3. the last console disconnects before the owner hears anything;
    #    the /audio teardown hook kills `say` mid-word
    present["now"] = False
    assert eng.interrupt("console disconnected") is True
    await _until(lambda: eng._say_proc is None, "the cut landing")
    await asyncio.sleep(0.05)
    # 4. THE FAIL-OPEN: speak() returned normally, _speak() said True, and
    #    the approval was marked spoken. It must stay unspoken.
    assert router.pending_approvals()[0].spoken is False, \
        "a readback nobody heard was marked spoken"
    # 5. the owner reconnects and later says a bare yes
    present["now"] = True
    bus.publish("command.received", {"text": "yes"})
    await _until(lambda: len(spawned) >= 2, "the brain answering the yes")
    # 6. the destructive tool is NOT delivered; Marvin says what happened
    assert all(c[0] != "deliver" for c in fleet.calls), \
        "a bare yes delivered a tool whose readback was cut mid-word"
    assert len(router.pending_approvals()) == 1
    assert "haven't read" in spawned[-1][0][-1]
    spawned[-1][1].finish(0)
    task.cancel()


async def test_a_repo_confirm_cut_mid_word_is_never_marked_spoken(
        monkeypatch, tmp_path):
    """The flag's other consumer: Onboarding.mark_spoken rides the same
    _speak() return, so a cut repo question must stay unanswerable too."""
    from server.discovery import Candidate
    from server.onboarding import Onboarding
    from server.registry import Registry
    spawned = _say_factory(monkeypatch)
    bus, butler = EventBus(), FakeButler()
    r = Registry()
    r.merge_candidates([Candidate(path="/p/quant agent", name="quant agent",
                                  sources=["a", "b"])])
    ob = Onboarding(bus, r, tmp_path / "projects.json")
    present = {"now": True}
    eng = SpeakEngine("", "", "ws://unused", bus.publish,
                      send_audio=lambda b: None,
                      listening=lambda: present["now"])
    task = await brain(bus, butler, eng, router=Router(),
                       registry=r, onboarding=ob)
    await ob.ask_next()                    # publishes confirm.request
    await _until(lambda: eng._say_proc is not None, "the question in flight")
    present["now"] = False
    assert eng.interrupt("console disconnected") is True
    await asyncio.sleep(0.05)
    assert ob._asking_spoken is False, "a cut repo question was marked spoken"
    present["now"] = True
    bus.publish("command.received", {"text": "yes"})
    await asyncio.sleep(0.05)
    assert not any(p.confirmed for p in r.projects), \
        "a yes confirmed a repo whose question was cut mid-word"
    task.cancel()


# =====================================================================
# the `say` engine: the decision consumes the evidence
# =====================================================================
async def test_a_killed_say_raises_and_still_publishes_the_cut(monkeypatch):
    spawned = _say_factory(monkeypatch)
    events = []
    present = {"now": True}
    eng = SpeakEngine("", "", "ws://unused",
                      lambda t, d: events.append((t, d)),
                      send_audio=lambda b: None,
                      listening=lambda: present["now"])
    task = asyncio.create_task(eng.speak("A long reply, sir."))
    await _until(lambda: eng._say_proc is not None, "speech in flight")
    present["now"] = False
    assert eng.interrupt("console disconnected") is True
    with pytest.raises(SpeechNotDelivered):
        await asyncio.wait_for(task, 2)
    done = dict(events)["tts.done"]                # still published, still marked
    assert done.get("interrupted") == "console disconnected"


async def test_a_say_that_exits_nonzero_is_not_delivery(monkeypatch):
    """`say` reports its own failures (unknown voice, dead audio device)
    through the exit code, which was never consulted: a `say` that printed
    an error and exited 1 counted as a delivered readback."""
    spawned = _say_factory(monkeypatch)
    events = []
    eng = SpeakEngine("", "", "ws://unused",
                      lambda t, d: events.append((t, d)),
                      send_audio=lambda b: None, listening=lambda: True)
    task = asyncio.create_task(eng.speak("Anything, sir."))
    await _until(lambda: eng._say_proc is not None, "speech in flight")
    spawned[0][1].finish(rc=1)
    with pytest.raises(SpeechNotDelivered, match="exited 1"):
        await asyncio.wait_for(task, 2)
    assert dict(events)["tts.done"].get("interrupted") == "say exited 1"


async def test_a_say_finishing_into_an_empty_room_is_not_delivery(monkeypatch):
    """The interrupt() race: the last console leaves in the same instant the
    subprocess finishes, so there is nothing left to kill and no cut is
    recorded. The presence gate is re-read AFTER speech for exactly this
    sliver — a sentence that completed into an empty room is not delivered."""
    present = {"now": True}

    class InstantProc:
        returncode = None

        async def wait(self):
            present["now"] = False         # the room empties as `say` ends
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    async def fake_exec(*argv, **kw):
        return InstantProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    events = []
    eng = SpeakEngine("", "", "ws://unused",
                      lambda t, d: events.append((t, d)),
                      send_audio=lambda b: None,
                      listening=lambda: present["now"])
    with pytest.raises(SpeechNotDelivered, match="no console at completion"):
        await eng.speak("Done just as you left, sir.")
    assert dict(events)["tts.done"].get("interrupted") == "no console at completion"


async def test_a_clean_say_with_a_listener_is_still_delivered(monkeypatch):
    """Positive control: without it, every test above passes on an engine
    that simply never reports delivery again."""
    spawned = _say_factory(monkeypatch)
    events = []
    eng = SpeakEngine("", "", "ws://unused",
                      lambda t, d: events.append((t, d)),
                      send_audio=lambda b: None, listening=lambda: True)
    task = asyncio.create_task(eng.speak("All well, sir."))
    await _until(lambda: eng._say_proc is not None, "speech in flight")
    spawned[0][1].finish(0)
    await asyncio.wait_for(task, 2)                # no raise
    assert "interrupted" not in dict(events)["tts.done"]
