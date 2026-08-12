"""/mic teardown: the handler must survive a client that has already vanished.

Every one of these needs a REAL socket. Starlette's in-process test transport
enqueues whatever the app sends and never fails, so `ws.close()` on a peer that
is already gone succeeds there — the very failure being fixed cannot be
expressed through it. `_live_server` (borrowed from test_app_auth, which
documents the same limitation for /events) puts uvicorn on loopback so a closed
peer is a closed peer.

The assertion is the evidence the owner actually had: uvicorn's
"Exception in ASGI application" report. It has to be captured from INSIDE the
live-server block — uvicorn.Config runs dictConfig on startup, which strips
handlers off the loggers it configures, so a listener attached earlier is gone
by the time the server is up.
"""
import asyncio
import contextlib
import json
import logging

import httpx
import websockets

from fastapi.websockets import WebSocketState

from server import app as app_mod
from server.app import close_quietly, create_app
from tests.test_app_auth import _free_port, _live_server


class _Recorder(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _asgi_errors():
    """Every ERROR uvicorn logs for the ASGI app while this is open."""
    # uvicorn.error does not propagate to root (its `uvicorn` parent sets
    # propagate=False), so caplog cannot see it. Attach directly.
    logger = logging.getLogger("uvicorn.error")
    recorder = _Recorder()
    logger.addHandler(recorder)
    try:
        yield recorder.records
    finally:
        logger.removeHandler(recorder)


def _formatted(records) -> str:
    return "\n".join(logging.Formatter().format(r) for r in records)


class _StopThenLinger:
    """Stands in for SttRelay on the ordinary end of a press-and-hold.

    Returns on the client's `stop` frame — exactly like the real relay, which
    hands Deepgram a CloseStream and unwinds — WITHOUT ever reading the
    disconnect that follows. The sleep is the real relay's Deepgram flush, and
    it guarantees the browser's close lands first. That is the whole point of
    this case: starlette's client_state and application_state BOTH still read
    CONNECTED when the handler goes to close, because nothing has told starlette
    otherwise. A state check alone does not save this one.
    """

    def __init__(self, **kwargs):
        pass

    async def run(self, inbound, publish):
        async for kind, payload in inbound:
            if kind == "text" and json.loads(payload).get("type") == "stop":
                await asyncio.sleep(0.3)
                return


class _DrainToDisconnect:
    """Stands in for SttRelay when the browser vanishes mid-hold.

    Page reload or network blip: inbound() reads the disconnect and returns
    cleanly, so client_state IS flipped to DISCONNECTED before the close.

    The sleep stands for the real relay's own unwind — cancelling both pumps and
    awaiting `dg.close()` — and it matters: without it the handler can reach its
    close inside the same tick that the disconnect was queued, before the
    transport has actually gone, and the doomed send quietly succeeds.
    """

    def __init__(self, **kwargs):
        pass

    async def run(self, inbound, publish):
        async for _ in inbound:
            pass
        await asyncio.sleep(0.2)


class _Exploding:
    """Stands in for a relay that dies on its way up.

    The live shape of this is a bad or expired DEEPGRAM_API_KEY: websockets
    raises InvalidStatus, which is NOT an OSError, so SttRelay's `except OSError`
    misses it and it lands in the handler — silently, with the console never
    told why the microphone produced nothing.
    """

    def __init__(self, **kwargs):
        pass

    async def run(self, inbound, publish):
        await anext(inbound)
        raise RuntimeError("deepgram refused the upgrade")


def _serve(tmp_path, monkeypatch, relay_cls):
    """Build the app with `relay_cls` standing in for SttRelay. Returns (app, port)."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    monkeypatch.setattr(app_mod, "SttRelay", relay_cls)
    app = create_app(base_dir=tmp_path)
    port = _free_port()
    app.state.cfg.port = port  # keep the Host-header check consistent
    return app, port


def _session_cookie(app, port: int) -> str:
    with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
        token = app.state.bootstrap_token_plain
        r = client.get(f"/bootstrap?token={token}", follow_redirects=False)
        assert r.status_code == 303
        return r.cookies["marvin_session"]


async def _mic(port: int, cookie: str):
    return await websockets.connect(
        f"ws://127.0.0.1:{port}/mic",
        additional_headers={"Cookie": f"marvin_session={cookie}"},
    )


def _record(app) -> list[tuple]:
    """Divert the bus so a test can read the stt.* the handler published.

    Filtered: the lifespan publishes registry/confirm traffic of its own that
    has nothing to do with the microphone.
    """
    seen: list[tuple] = []

    def publish(type_, data):
        if type_.startswith("stt."):
            seen.append((type_, data))

    app.state.bus.publish = publish
    return seen


START = json.dumps({"type": "start", "encoding": "linear16",
                    "sample_rate": 16000, "channels": 1, "t_hold": 1})
STOP = json.dumps({"type": "stop", "t_release": 2})


class _FakeWS:
    def __init__(self, client_state, application_state):
        self.client_state = client_state
        self.application_state = application_state
        self.closed = 0

    async def close(self, code=1000, reason=None):
        self.closed += 1


async def test_a_socket_known_to_be_gone_is_never_written_to():
    """The half of the guard that suppression cannot cover.

    Once the reader has drained the disconnect, starlette KNOWS the peer left.
    Sending anyway would work — the exception is swallowed either way — so only
    this pins that the close is skipped rather than thrown at a dead socket.
    """
    gone = _FakeWS(WebSocketState.DISCONNECTED, WebSocketState.CONNECTED)
    await close_quietly(gone)
    assert gone.closed == 0

    already_closed = _FakeWS(WebSocketState.CONNECTED, WebSocketState.DISCONNECTED)
    await close_quietly(already_closed)
    assert already_closed.closed == 0

    live = _FakeWS(WebSocketState.CONNECTED, WebSocketState.CONNECTED)
    await close_quietly(live)
    assert live.closed == 1


async def test_close_after_a_normal_release_does_not_raise(tmp_path, monkeypatch):
    """The ordinary end of a press-and-hold: stop, then the browser closes.

    The relay is still flushing when the socket dies, so the handler's close
    lands on a peer that is gone while starlette still believes it is CONNECTED.
    """
    app, port = _serve(tmp_path, monkeypatch, _StopThenLinger)
    with _live_server(app, port), _asgi_errors() as errors:
        cookie = _session_cookie(app, port)
        ws = await _mic(port, cookie)
        await ws.send(START)
        await ws.send(b"\x00\x00" * 960)
        await ws.send(STOP)
        await ws.close()
        await asyncio.sleep(1.0)  # outlast the flush and the handler's close
        assert errors == [], _formatted(errors)


async def test_close_after_the_browser_vanishes_does_not_raise(tmp_path, monkeypatch):
    """Reload / network blip: no stop frame, the socket just dies mid-hold."""
    app, port = _serve(tmp_path, monkeypatch, _DrainToDisconnect)
    with _live_server(app, port), _asgi_errors() as errors:
        cookie = _session_cookie(app, port)
        ws = await _mic(port, cookie)
        await ws.send(START)
        await ws.send(b"\x00\x00" * 960)
        await ws.close()
        await asyncio.sleep(0.5)
        assert errors == [], _formatted(errors)


async def test_a_relay_that_dies_is_surfaced_not_raised(tmp_path, monkeypatch):
    """A relay failure is an stt.error the console can speak, never a traceback."""
    app, port = _serve(tmp_path, monkeypatch, _Exploding)
    seen = _record(app)
    with _live_server(app, port), _asgi_errors() as errors:
        cookie = _session_cookie(app, port)
        ws = await _mic(port, cookie)
        await ws.send(START)
        await asyncio.sleep(0.5)
        await ws.close()
        await asyncio.sleep(0.3)
        assert errors == [], _formatted(errors)
    assert [t for t, _ in seen] == ["stt.error"], seen


async def test_audio_sent_before_deepgram_connects_is_not_lost(tmp_path, monkeypatch):
    """The server half of the demo's clipped-front defect.

    The client's fix flushes everything spoken during its handshake the moment
    /mic opens — which is exactly when SttRelay has NOT yet dialed Deepgram:
    `relay.run` reads the start frame and then awaits `websockets.connect`
    before anything consumes `inbound()`. Frames sent in that window sit in
    uvicorn's receive queue; this pins that every one of them reaches Deepgram,
    in order, byte for byte. The fake Deepgram stalls its handshake for 300ms
    so the entire burst provably lands mid-dial, and the burst is 40 frames —
    past the websockets library's default inbound buffering — so backpressure
    queuing (not dropping) is pinned too. No stand-in: this is SttRelay itself.
    """
    received: list[bytes] = []

    async def dg_handler(ws):
        async for msg in ws:
            if isinstance(msg, bytes):
                received.append(msg)
            elif json.loads(msg).get("type") == "CloseStream":
                await ws.close()
                return

    async def stall_handshake(connection, request):
        await asyncio.sleep(0.3)  # the whole burst arrives while we sleep
        return None

    dg = await websockets.serve(dg_handler, "127.0.0.1", 0,
                                process_request=stall_handshake)
    dg_port = dg.sockets[0].getsockname()[1]
    monkeypatch.setenv("DEEPGRAM_URL", f"ws://127.0.0.1:{dg_port}")

    app, port = _serve(tmp_path, monkeypatch, app_mod.SttRelay)
    seen = _record(app)
    frames = [bytes([i]) * 1920 for i in range(40)]  # 40 distinct 60ms frames
    try:
        with _live_server(app, port), _asgi_errors() as errors:
            cookie = _session_cookie(app, port)
            ws = await _mic(port, cookie)
            await ws.send(START)
            for f in frames:  # the flush: everything buffered pre-open, at once
                await ws.send(f)
            await ws.send(STOP)
            for _ in range(100):  # outlast the stalled dial and the drain
                if len(received) >= len(frames):
                    break
                await asyncio.sleep(0.05)
            await ws.close()
            assert errors == [], _formatted(errors)
    finally:
        dg.close()
        await dg.wait_closed()
    assert received == frames, (
        f"lost {len(frames) - len(received)} of {len(frames)} frames "
        "(or reordered/corrupted them)")
    assert not [e for e in seen if e[0] == "stt.error"], seen


async def test_a_press_with_no_audio_is_surfaced_not_raised(tmp_path, monkeypatch):
    """A tap released before the browser's `start` frame ever went out.

    inbound() ends without yielding anything, so the REAL relay's opening
    `await anext(inbound)` raises StopAsyncIteration straight into the handler.
    No stand-in here — this is SttRelay itself, and it never reaches Deepgram.
    """
    app, port = _serve(tmp_path, monkeypatch, app_mod.SttRelay)
    seen = _record(app)
    with _live_server(app, port), _asgi_errors() as errors:
        cookie = _session_cookie(app, port)
        ws = await _mic(port, cookie)
        await ws.close()  # released while the socket was still coming up
        await asyncio.sleep(0.5)
        assert errors == [], _formatted(errors)
    assert [t for t, _ in seen] == ["stt.error"], seen
