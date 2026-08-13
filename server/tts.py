"""SpeakEngine: ElevenLabs Flash stream-input with reconnect; `say` fallback."""
from __future__ import annotations

import asyncio
import base64
import inspect
import json
import time

import websockets

ELEVEN_BASE = "wss://api.elevenlabs.io"

# How long speak() waits for a console to (re)appear before refusing the
# reply. A page reload drops every console for well under a second — the
# /audio socket reconnects at script load, before the setup click — so the
# grace turns "reply landed during a refresh" into a normally delivered
# reply. A console that is genuinely gone stays gone past any grace, and the
# refusal is honest (NobodyListening). Checked at SPEAK time, never latched:
# a session can lose and regain its consoles any number of times.
LISTENER_GRACE_S = 5.0
LISTENER_POLL_S = 0.1


class NobodyListening(RuntimeError):
    """No console is connected to hear this reply, and none arrived within
    the grace window.

    Raised INSTEAD of speaking, and deliberately not swallowed here: every
    speak() caller already treats a raise as "not delivered" — the brain's
    _speak publishes butler.error and returns False (so an approval readback
    that nobody heard is never marked spoken), and the social digest
    publishes social.error. The reply is DROPPED with that published record,
    never queued: a queue replayed at reconnect is the talking-to-an-empty-
    room defect again, just deferred to a moment nobody asked for.
    """


class SpeechNotDelivered(RuntimeError):
    """Speech STARTED but did not verifiably reach a connected console.

    NobodyListening's after-the-fact sibling, raised once the utterance is
    over — after tts.done goes out carrying an `interrupted` field that names
    the reason — because "speak() did not raise" used to be the success
    signal, and both engines returned normally for speech nobody could have
    heard. That false signal was the ninth consent fail-open: interrupt()
    killed `say` mid-word on the last console leaving, proc.wait() returned
    like a clean finish, the brain's _speak() said True, and the approval
    readback was marked spoken — so a later bare "yes" delivered a
    destructive tool whose sentence was never voiced.

    What each engine now requires before speak() returns without this raise:
      * `say`   — the subprocess was not cut (no _say_cut), exited 0 (a kill
                  is -9, and `say` reports its own failures via the exit
                  code), and a console was still connected when it finished
                  (re-read AFTER speech: interrupt() cannot cut a process
                  that ended in the same instant the last console left).
      * eleven  — the stream carried at least one audio chunk, ended with
                  isFinal (a clean socket close mid-utterance is not a
                  finish), every chunk was handed to a non-empty room
                  (latched: a console returning mid-stream missed the
                  start), the per-console sends were drained, and the room
                  was still occupied after that drain.

    What the ABSENCE of this raise still cannot prove: that a human heard
    anything. `say` plays through the machine's own speakers next to a
    connected console; ElevenLabs bytes accepted by an open /audio socket
    may still sit unplayed in a browser buffer when the tab dies a moment
    later. No software closes that last gap — the consent design leans on
    the approval TTL and the console card, which ignore `spoken`, exactly
    for this reason.
    """


class SpeakEngine:
    def __init__(self, voice_id: str, api_key: str, base_url: str,
                 publish, send_audio, listening=None):
        self.voice_id = voice_id
        self.api_key = api_key
        self.base_url = base_url
        self.publish = publish
        self.send_audio = send_audio
        # `listening` answers "is any console connected right now?" — the app
        # wires it to the /audio WebSocket client set (see server/app.py for
        # why that set, and not bus subscribers, is the presence signal).
        # None means no presence information (direct constructions, tests):
        # speak unconditionally, the pre-gate behavior.
        self._listening = listening
        self._say_proc = None      # the in-flight `say`, if any
        self._say_cut = ""         # why it was cut mid-speech, if it was
        self._ws = None
        self._lock = asyncio.Lock()

    @property
    def _eleven_enabled(self) -> bool:
        return bool(self.api_key and self.voice_id)

    def _url(self) -> str:
        return (f"{self.base_url}/v1/text-to-speech/{self.voice_id}/stream-input"
                f"?model_id=eleven_flash_v2_5&auto_mode=true&output_format=mp3_44100_64")

    async def _connect(self):
        self._ws = await websockets.connect(
            self._url(), additional_headers={"xi-api-key": self.api_key})

    async def preconnect(self) -> None:
        if self._eleven_enabled and self._ws is None:
            try:
                await self._connect()
            except OSError:
                self._ws = None

    async def _require_listener(self) -> None:
        """Refuse to start speech into an empty room — but ride out a reload.

        Nothing here is remembered between calls: presence is re-answered for
        every reply at its own moment, so a refresh (or an hour with the
        console closed) can never mute the session for the replies that come
        after a console returns."""
        if self._listening is None or self._listening():
            return
        deadline = time.monotonic() + LISTENER_GRACE_S
        while not self._listening():
            if time.monotonic() >= deadline:
                raise NobodyListening("no console connected")
            await asyncio.sleep(LISTENER_POLL_S)

    def interrupt(self, reason: str = "console disconnected") -> bool:
        """Kill an in-flight `say` NOW. The /audio handler calls this when
        the LAST console disconnects: the only channel the owner could
        answer through just vanished, so the mouth stops with it.

        `say` only, deliberately. The two engines fail differently here: an
        in-flight ElevenLabs stream is already inaudible the moment the page
        dies — its chunks broadcast to zero sockets and the browser-side
        player died with the tab — while `say` plays through the machine's
        own speakers and keeps going unless killed. Closing the ElevenLabs
        socket mid-protocol would also trip speak()'s transparent reconnect
        into REPLAYING the utterance to the empty room. Returns whether
        anything was actually cut, and is safe to call when nothing is.
        """
        proc = self._say_proc
        if proc is None or proc.returncode is not None:
            return False
        self._say_cut = reason or "interrupted"
        try:
            proc.kill()
        except ProcessLookupError:
            return False
        return True

    async def speak(self, text: str) -> None:
        async with self._lock:
            # Before tts.start, not after: the console must never be told
            # speech began that never did (its player tears down the previous
            # utterance on tts.start).
            await self._require_listener()
            self.publish("tts.start", {"text": text})
            if self._eleven_enabled:
                for attempt in (1, 2):  # one transparent reconnect
                    try:
                        if self._ws is None:
                            await self._connect()
                        await self._speak_eleven(text)
                        return
                    except (websockets.ConnectionClosed, OSError):
                        self._ws = None
                        if attempt == 2:
                            break
                    except BaseException:
                        # CancelledError (the brain's 60s speak timeout lands
                        # here, mid-protocol) or any unexpected error leaves the
                        # socket mid-stream before `isFinal`. Keeping it cached
                        # makes the NEXT utterance consume THIS turn's leftover
                        # frames and break on the old isFinal -- every later
                        # answer plays one turn late and never self-heals.
                        # Discard the socket (best-effort close), then re-raise.
                        ws, self._ws = self._ws, None
                        if ws is not None:
                            try:
                                await ws.close()
                            except BaseException:  # noqa: BLE001 — cleanup only
                                pass
                        raise
            await self._speak_say(text)

    async def _speak_eleven(self, text: str) -> None:
        t0 = None
        await self._ws.send(json.dumps({"text": text + " "}))
        await self._ws.send(json.dumps({"text": ""}))  # flush
        async for raw in self._ws:
            msg = json.loads(raw)
            if msg.get("audio"):
                if t0 is None:
                    t0 = time.time()
                self.send_audio(base64.b64decode(msg["audio"]))
            if msg.get("isFinal"):
                break
        self.publish("tts.done", {"t_first_audio": t0, "engine": "elevenlabs"})

    async def _speak_say(self, text: str) -> None:
        proc = await asyncio.create_subprocess_exec("say", "-v", "Daniel", text)
        self._say_proc, self._say_cut = proc, ""
        t0 = time.time()
        rc = None
        try:
            # The presence gate ran before this subprocess existed, so a
            # console that left in that window was told "nothing to cut".
            # Re-check once now that there IS a process; the disconnect hook
            # covers every later moment.
            if self._listening is not None and not self._listening():
                self.interrupt()
            rc = await proc.wait()
        except asyncio.CancelledError:
            # a speak timeout must not leave `say` talking over the next turn
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            raise
        finally:
            self._say_proc = None
        # The success signal, decided from evidence rather than from "wait()
        # did not raise": wait() returns normally after a kill, so a readback
        # cut mid-word used to be indistinguishable from a delivered one —
        # the ninth fail-open. Precedence: the recorded cut names the event
        # most honestly; the exit code catches both the kill it implies and
        # `say`'s own failures (unknown voice, dead audio device — nonzero,
        # never checked before); the room is then re-read AFTER speech, with
        # no grace, for the one sliver the disconnect hook cannot see —
        # interrupt() finding the process already finished in the same
        # instant the last console left records nothing. Fail closed on an
        # unknown exit (rc None).
        if self._say_cut:
            reason = self._say_cut
        elif rc != 0:
            reason = f"say exited {rc}"
        elif self._listening is not None and not self._listening():
            reason = "no console at completion"
        else:
            reason = ""
        done = {"t_first_audio": t0, "engine": "say"}
        if reason:
            # A cut reply must not be indistinguishable from a delivered one.
            # Additive field; the event name is unchanged.
            done["interrupted"] = reason
        self.publish("tts.done", done)
        if reason:
            raise SpeechNotDelivered(reason)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
