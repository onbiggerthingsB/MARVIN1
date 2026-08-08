import asyncio
import base64
import json

import pytest
import websockets

from server.tts import SpeakEngine


async def start_fake_eleven(chunks=(b"MP3A", b"MP3B"), idle_close_after: float | None = None):
    """Fake stream-input endpoint: replies with base64 audio for each text, honors
    idle_close_after to simulate the 20s idle disconnect.

    DEVIATION from the brief (recorded in the task report): the brief's handler
    applied the idle timeout with a single `ws.recv()` *before* the message loop,
    which discarded the client's first `{"text": ...}` frame — so the first
    speak() produced no audio and its `async for raw in self._ws` blocked forever
    (empirically a hard hang). Corrected to a per-message idle timeout that closes
    only after `idle_close_after` seconds of no incoming frame, without dropping
    the frame that carries the text. Production SpeakEngine is unchanged."""
    async def handler(ws):
        try:
            while True:
                if idle_close_after is not None:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=idle_close_after)
                    except asyncio.TimeoutError:
                        await ws.close(code=1000)
                        return
                else:
                    msg = await ws.recv()
                data = json.loads(msg)
                if data.get("text"):
                    for c in chunks:
                        await ws.send(json.dumps({"audio": base64.b64encode(c).decode()}))
                    await ws.send(json.dumps({"isFinal": True}))
        except websockets.ConnectionClosed:
            return

    server = await websockets.serve(handler, "127.0.0.1", 0)
    return server, f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"


async def test_speak_streams_chunks_and_reports_first_audio():
    server, url = await start_fake_eleven()
    events, audio = [], []
    eng = SpeakEngine("voice1", "key", url, lambda t, d: events.append((t, d)),
                      send_audio=lambda b: audio.append(b))
    await eng.speak("Hello sir.")
    await eng.close()
    server.close(); await server.wait_closed()
    assert audio == [b"MP3A", b"MP3B"]
    done = dict(events)["tts.done"]
    assert done["engine"] == "elevenlabs" and done["t_first_audio"] is not None


async def test_reconnects_after_idle_close():
    server, url = await start_fake_eleven(idle_close_after=0.05)
    events, audio = [], []
    eng = SpeakEngine("voice1", "key", url, lambda t, d: events.append((t, d)),
                      send_audio=lambda b: audio.append(b))
    await eng.speak("First.")
    await asyncio.sleep(0.15)          # fake server idle-closes the socket
    await eng.speak("Second.")         # must transparently reconnect
    await eng.close()
    server.close(); await server.wait_closed()
    assert audio.count(b"MP3A") == 2


async def test_cancelled_speak_discards_the_socket_and_next_turn_is_clean():
    """The 60s speak timeout cancels _speak_eleven mid-protocol. If the socket
    stays cached, the NEXT utterance consumes the PREVIOUS turn's leftover
    frames and breaks on the old isFinal -- every later answer plays one turn
    late, forever. Cancellation must discard the socket."""
    stall = asyncio.Event()

    async def handler(ws):
        try:
            while True:
                data = json.loads(await ws.recv())
                if not data.get("text"):
                    continue          # the empty flush frame
                await ws.send(json.dumps({"audio": base64.b64encode(b"MP3A").decode()}))
                if "stall" in data["text"]:
                    await stall.wait()               # never sends isFinal
                else:
                    await ws.send(json.dumps({"isFinal": True}))
        except websockets.ConnectionClosed:
            return

    server = await websockets.serve(handler, "127.0.0.1", 0)
    url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    events, audio = [], []
    eng = SpeakEngine("voice1", "key", url, lambda t, d: events.append((t, d)),
                      send_audio=lambda b: audio.append(b))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(eng.speak("stall this one"), 0.3)
    assert eng._ws is None, "cancelled speak left a mid-stream socket cached"

    await eng.speak("Second.")   # reconnects and completes THIS turn cleanly
    dones = [d for t, d in events if t == "tts.done"]
    assert dones and dones[-1]["engine"] == "elevenlabs"

    stall.set()                  # release the stalled handler so the server can close
    await eng.close()
    server.close(); await server.wait_closed()


async def test_cancelled_say_kills_the_subprocess(monkeypatch):
    killed = []

    class HangingProc:
        async def wait(self):
            await asyncio.sleep(60)

        def kill(self):
            killed.append(True)

    async def fake_exec(*argv, **kw):
        return HangingProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    eng = SpeakEngine("", "", "ws://unused", lambda t, d: None, send_audio=lambda b: None)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(eng.speak("too long"), 0.05)
    assert killed, "a cancelled `say` was left talking over the next turn"


async def test_say_fallback_when_no_key(monkeypatch):
    calls = []

    async def fake_exec(*argv, **kw):
        calls.append(argv)
        class P:
            async def wait(self): return 0
        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    events = []
    eng = SpeakEngine("", "", "ws://unused", lambda t, d: events.append((t, d)),
                      send_audio=lambda b: None)
    await eng.speak("Fallback line.")
    assert calls and calls[0][0] == "say"
    assert dict(events)["tts.done"]["engine"] == "say"
