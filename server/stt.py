"""Relay browser mic frames to Deepgram live and publish transcript events."""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from urllib.parse import urlencode

import websockets

DEFAULT_BASE = "wss://api.deepgram.com"

# The ONLY clock in the relay, and it starts at the release, never before.
# While the button is held the OWNER governs — a hold has no deadline, because
# any deadline here is the 30-second cousin of the 300ms endpointing bug: a
# machine deciding mid-sentence that the human is finished. Once the press has
# ended, the wind-down (Deepgram flushing its last finals and closing) is
# machine work and must be bounded: a wedged Deepgram may cost the tail of one
# utterance, never a hung /mic handler. Module-level so tests can shrink it.
DRAIN_TIMEOUT_S = 10.0


class SttRelay:
    def __init__(self, api_key: str, keyterms: list[str], base_url: str = DEFAULT_BASE):
        self.api_key = api_key
        self.keyterms = keyterms[:100]  # Deepgram cap
        self.base_url = base_url

    def _url(self, sample_rate: int) -> str:
        # `endpointing` is explicitly DISABLED, not omitted: omitting it means
        # Deepgram's default (10ms of silence), which is even more trigger-
        # happy than the 300ms that fragmented the live demo. This is a
        # press-and-hold interface — the release is an explicit, unambiguous
        # end-of-utterance signal from the human, and a silence heuristic has
        # no authority over it. Natural pauses (thinking, breathing, "uh")
        # exceed any threshold constantly; with endpointing off Deepgram never
        # declares `speech_final`, and utterance assembly in run() is gated on
        # the press instead. interim_results stay on: the live transcript
        # painting while the button is down is what makes holding feel heard.
        params = [("model", "nova-3"), ("encoding", "linear16"),
                  ("sample_rate", str(sample_rate)), ("channels", "1"),
                  ("interim_results", "true"), ("smart_format", "true"),
                  ("endpointing", "false")]
        params += [("keyterm", k) for k in self.keyterms]
        return f"{self.base_url}/v1/listen?{urlencode(params)}"

    async def run(self, inbound, publish) -> None:
        """inbound: async iterator of ("text", str) | ("bytes", bytes).
        publish: callable(type_: str, data: dict).

        ONE utterance per press. The press ends when the client's stop frame
        arrives (the release) or the client vanishes; Deepgram is then asked
        to flush (CloseStream), and everything it finalized — across any
        number of silence pauses — is published as a single stt.utterance
        when the stream winds down. `speech_final` is deliberately never
        acted on: endpointing is disabled in _url(), and even if a server
        sent one anyway, a silence heuristic must not override the button.
        """
        first = await anext(inbound)
        start = json.loads(first[1]) if first[0] == "text" else {}
        t_release: float | None = None
        finals: list[str] = []

        try:
            dg = await websockets.connect(
                self._url(int(start.get("sample_rate", 16000))),
                additional_headers={"Authorization": f"Token {self.api_key}"},
            )
        except OSError:
            publish("stt.error", {"reason": "deepgram unreachable"})
            return

        async def pump_up():
            nonlocal t_release
            try:
                async for kind, payload in inbound:
                    if kind == "bytes":
                        await dg.send(payload)
                    else:
                        msg = json.loads(payload)
                        if msg.get("type") == "stop":
                            t_release = msg.get("t_release")
                            return
            except websockets.ConnectionClosed:
                pass          # Deepgram died mid-hold; pump_down reports it
            except Exception:
                publish("stt.error", {"reason": "relay error"})
            finally:
                # EVERY way a press can end asks Deepgram to finalize what it
                # holds: the release (the stop frame), a client that vanished
                # mid-hold, a malformed frame. Best-effort — Deepgram may
                # already be gone, and then there is nothing to flush to.
                with contextlib.suppress(Exception):
                    await dg.send(json.dumps({"type": "CloseStream"}))

        async def pump_down():
            try:
                async for raw in dg:
                    msg = json.loads(raw)
                    if msg.get("type") != "Results":
                        continue
                    text = msg["channel"]["alternatives"][0]["transcript"].strip()
                    if not msg.get("is_final"):
                        if text:
                            publish("stt.interim", {"text": " ".join([*finals, text])})
                        continue
                    if text:
                        finals.append(text)
                        publish("stt.final", {"text": text})
            except websockets.ConnectionClosed as e:
                close_code = getattr(e, "rcvd", None) and e.rcvd.code
                if close_code not in (1000,):
                    publish("stt.error", {"reason": "deepgram disconnected"})
            except Exception:
                publish("stt.error", {"reason": "relay error"})

        up = asyncio.create_task(pump_up())
        down = asyncio.create_task(pump_down())
        try:
            # The hold: no deadline. pump_up ends on the release, on the
            # client vanishing, or on Deepgram dying under it (the next
            # frame's send raises) — never on a clock.
            await up
            # The wind-down: bounded. After CloseStream, Deepgram owes a
            # flush and a close; the timeout covers one that never closes.
            await asyncio.wait({down}, timeout=DRAIN_TIMEOUT_S)
        finally:
            for t in (up, down):
                t.cancel()
                with contextlib.suppress(BaseException):
                    await t
            await dg.close()
            # THE one publication point: everything Deepgram finalized during
            # this press, as one utterance — however many silence pauses it
            # contained, and whether the press ended in a release, a vanished
            # client, or a dead Deepgram (that last one also published
            # stt.error above, so a truncated dispatch is never silent).
            # An empty press publishes nothing.
            if finals:
                publish("stt.utterance", {"text": " ".join(finals),
                                          "t_release": t_release,
                                          "t_utterance": time.time()})
