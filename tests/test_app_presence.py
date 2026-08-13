"""The /audio WebSocket set is the presence signal for speech.

Which signal means "the owner is present"? The /audio socket, deliberately:
it is opened by every console page at script load (before the setup click,
unlike /events), it dies with the tab, it reconnects within about a second
across a reload, and for the ElevenLabs engine it is literally the channel
the reply's audio is delivered through — "someone can hear this" and "an
/audio socket is open" are the same fact. Bus subscribers cannot serve here:
the brain loop holds one itself, so the bus always has a subscriber even
with every console closed.

Real sockets (the borrowed live server, as in test_app_mic): presence is
about connections actually opening and dying, which the in-process
transport cannot express.
"""
import asyncio

import websockets

from server.app import create_app
from tests.test_app_auth import _free_port, _live_server
from tests.test_app_mic import _session_cookie


async def _audio(port: int, cookie: str):
    return await websockets.connect(
        f"ws://127.0.0.1:{port}/audio",
        additional_headers={"Cookie": f"marvin_session={cookie}"})


async def _until(cond, what: str, tries: int = 150):
    for _ in range(tries):
        if cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"never observed: {what}")


async def test_audio_sockets_drive_presence_and_the_last_disconnect_cuts(tmp_path):
    """DEFECT 2 wiring: the engine's `listening` reads the /audio client set,
    and the LAST console leaving cuts in-flight speech — one console leaving
    while another remains cuts nothing (two tabs are one owner)."""
    app = create_app(base_dir=tmp_path)
    port = _free_port()
    app.state.cfg.port = port  # keep the Host-header check consistent

    cut: list[str] = []
    app.state.speaker.interrupt = (
        lambda reason="": cut.append(reason) or True)  # observe the hook

    listening = app.state.speaker._listening
    assert listening is not None, "the app never wired a presence signal"
    assert listening() is False, "an empty room read as occupied at boot"

    with _live_server(app, port):
        cookie = _session_cookie(app, port)
        ws1 = await _audio(port, cookie)
        await _until(listening, "presence after the first console connected")

        ws2 = await _audio(port, cookie)   # a second tab, same owner
        await ws1.close()
        await asyncio.sleep(0.3)           # let the disconnect land
        assert listening() is True, "one of two tabs closing read as absence"
        assert cut == [], "speech was cut while a console was still connected"

        await ws2.close()                  # the LAST console leaves
        await _until(lambda: not listening(), "absence after the last close")
        await _until(lambda: cut, "the in-flight cut on the last disconnect")
        assert cut == ["console disconnected"], cut
