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
import base64
import json
import time

import pytest
import websockets

from server import tts as tts_mod
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


async def test_the_elevenlabs_analogue_a_stream_to_an_emptied_room_is_unspoken(
        monkeypatch):
    """Same sequence, worse engine: the chunks go to fire-and-forget sends,
    so the tab dying mid-readback never raised anything — isFinal arrived,
    speak() returned, the approval was marked spoken."""
    stream_done = asyncio.Event()

    async def handler(ws):
        try:
            while True:
                data = json.loads(await ws.recv())
                if not data.get("text"):
                    continue
                for c in (b"MP3A", b"MP3B"):
                    await ws.send(json.dumps(
                        {"audio": base64.b64encode(c).decode()}))
                await ws.send(json.dumps({"isFinal": True}))
                stream_done.set()
        except websockets.ConnectionClosed:
            return

    server = await websockets.serve(handler, "127.0.0.1", 0)
    url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    bus, butler = EventBus(), FakeButler()
    router, fleet = Router(), FakeFleet()
    a = router.open_approval("soccer", DESTRUCTIVE, now=time.time(),
                             path="/p/soccer")
    present = {"now": True}
    chunks = []

    def send_audio(b):
        # the last console dies as the first chunk is handed to the fan-out:
        # every byte of this readback broadcasts to zero sockets
        chunks.append(b)
        present["now"] = False

    eng = SpeakEngine("voice1", "key", url, bus.publish,
                      send_audio=send_audio,
                      listening=lambda: present["now"])
    task = await brain(bus, butler, eng, router=router,
                       registry=confirmed_registry("soccer"), fleet=fleet)
    bus.publish("approval.request", card(
        a, "soccer wants Bash — rm -rf /Users/likerun/Documents. "
           "Approve or deny, sir?"))
    await asyncio.wait_for(stream_done.wait(), 5)
    await asyncio.sleep(0.1)
    assert chunks, "the fake stream never produced audio"
    assert router.pending_approvals()[0].spoken is False, \
        "a stream whose audio went to zero consoles was marked spoken"
    present["now"] = True                          # the owner reconnects
    bus.publish("command.received", {"text": "yes"})
    await asyncio.sleep(0.1)
    assert all(c[0] != "deliver" for c in fleet.calls)
    assert len(router.pending_approvals()) == 1
    task.cancel()
    await eng.close()
    server.close()
    await server.wait_closed()


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


# =====================================================================
# the ElevenLabs engine: audio + isFinal + an occupied room, or no claim
# =====================================================================
async def _eleven(handler):
    server = await websockets.serve(handler, "127.0.0.1", 0)
    return server, f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"


def _stream_handler(frames):
    """A fake stream-input endpoint that answers the first text frame with
    `frames` (already-encoded JSON payloads) and then idles."""
    async def handler(ws):
        try:
            while True:
                data = json.loads(await ws.recv())
                if not data.get("text"):
                    continue
                for f in frames:
                    await ws.send(json.dumps(f))
                if frames and frames[-1].get("isFinal"):
                    continue
                await ws.close(code=1000)      # clean close, no isFinal
                return
        except websockets.ConnectionClosed:
            return
    return handler


def _audio_frame(b=b"MP3A"):
    return {"audio": base64.b64encode(b).decode()}


async def test_a_room_that_empties_mid_stream_is_not_delivery():
    """The room is re-read at every chunk hand-off — and LATCHED: a console
    arriving back before isFinal missed the start, so the utterance stays
    undelivered even though the final read would pass."""
    server, url = await _eleven(_stream_handler(
        [_audio_frame(b"MP3A"), _audio_frame(b"MP3B"), {"isFinal": True}]))
    present, handed = {"now": True}, []

    def send_audio(b):
        handed.append(b)
        if len(handed) == 1:
            present["now"] = False         # dies on the first chunk...

    events = []
    eng = SpeakEngine("voice1", "key", url,
                      lambda t, d: events.append((t, d)),
                      send_audio=send_audio,
                      listening=lambda: present["now"] or len(handed) >= 2)
    # ...and a new console appears before the second chunk: the latch holds
    with pytest.raises(SpeechNotDelivered, match="mid-stream"):
        await eng.speak("Gone mid-sentence, sir.")
    assert dict(events)["tts.done"].get("interrupted") == \
        "console disconnected mid-stream"
    await eng.close()
    server.close()
    await server.wait_closed()


async def test_a_clean_close_without_isfinal_is_not_delivery():
    """The socket closing normally ends the message iterator without raising,
    which used to be indistinguishable from a finished utterance."""
    server, url = await _eleven(_stream_handler([_audio_frame()]))
    events = []
    eng = SpeakEngine("voice1", "key", url,
                      lambda t, d: events.append((t, d)),
                      send_audio=lambda b: None, listening=lambda: True)
    with pytest.raises(SpeechNotDelivered, match="before isFinal"):
        await eng.speak("Cut off by the socket, sir.")
    assert dict(events)["tts.done"].get("interrupted") == \
        "stream ended before isFinal"
    await eng.close()
    server.close()
    await server.wait_closed()


async def test_isfinal_with_zero_audio_is_not_delivery():
    server, url = await _eleven(_stream_handler([{"isFinal": True}]))
    events = []
    eng = SpeakEngine("voice1", "key", url,
                      lambda t, d: events.append((t, d)),
                      send_audio=lambda b: None, listening=lambda: True)
    with pytest.raises(SpeechNotDelivered, match="no audio"):
        await eng.speak("Silence, sir.")
    done = dict(events)["tts.done"]
    assert done["t_first_audio"] is None
    assert done.get("interrupted") == "no audio in stream"
    await eng.close()
    server.close()
    await server.wait_closed()


async def test_sends_are_drained_before_delivery_is_claimed():
    """app.py's per-console sends are fire-and-forget tasks whose failure
    discards the dead console from the presence set. The engine must AWAIT
    what send_audio hands back before judging the room, so a console whose
    socket died on the very last chunk is seen — not raced past."""
    server, url = await _eleven(_stream_handler(
        [_audio_frame(), {"isFinal": True}]))
    present = {"now": True}

    async def failing_send():
        # models _safe_send: the write fails, the console is discarded —
        # but only once the send actually RUNS. A bare coroutine (not a
        # pre-scheduled task) pins that the engine itself awaits the
        # fan-out; without the drain the discard never lands before the
        # verdict and the room reads as occupied.
        present["now"] = False

    eng = SpeakEngine("voice1", "key", url, lambda t, d: None,
                      send_audio=lambda b: failing_send(),
                      listening=lambda: present["now"])
    with pytest.raises(SpeechNotDelivered, match="no console at completion"):
        await eng.speak("To a dead socket, sir.")
    await eng.close()
    server.close()
    await server.wait_closed()


async def test_a_complete_stream_into_an_occupied_room_is_delivered():
    """Positive control for the eleven path, presence signal wired."""
    server, url = await _eleven(_stream_handler(
        [_audio_frame(b"MP3A"), _audio_frame(b"MP3B"), {"isFinal": True}]))
    events, handed = [], []
    eng = SpeakEngine("voice1", "key", url,
                      lambda t, d: events.append((t, d)),
                      send_audio=lambda b: handed.append(b),
                      listening=lambda: True)
    await eng.speak("All delivered, sir.")         # no raise
    assert handed == [b"MP3A", b"MP3B"]
    assert "interrupted" not in dict(events)["tts.done"]
    await eng.close()
    server.close()
    await server.wait_closed()
