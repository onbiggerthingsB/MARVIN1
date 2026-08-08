"""Relay browser mic frames to Deepgram live and publish transcript events."""
from __future__ import annotations

import asyncio
import json
import time
from urllib.parse import urlencode

import websockets

DEFAULT_BASE = "wss://api.deepgram.com"


class SttRelay:
    def __init__(self, api_key: str, keyterms: list[str], base_url: str = DEFAULT_BASE):
        self.api_key = api_key
        self.keyterms = keyterms[:100]  # Deepgram cap
        self.base_url = base_url

    def _url(self, sample_rate: int) -> str:
        params = [("model", "nova-3"), ("encoding", "linear16"),
                  ("sample_rate", str(sample_rate)), ("channels", "1"),
                  ("interim_results", "true"), ("smart_format", "true"),
                  ("endpointing", "300")]
        params += [("keyterm", k) for k in self.keyterms]
        return f"{self.base_url}/v1/listen?{urlencode(params)}"

    async def run(self, inbound, publish) -> None:
        """inbound: async iterator of ("text", str) | ("bytes", bytes).
        publish: callable(type_: str, data: dict)."""
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
                            await dg.send(json.dumps({"type": "CloseStream"}))
                            return
            except websockets.ConnectionClosed:
                pass

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
                    if msg.get("speech_final") and finals:
                        publish("stt.utterance", {"text": " ".join(finals),
                                                  "t_release": t_release,
                                                  "t_utterance": time.time()})
                        finals.clear()
            except websockets.ConnectionClosed as e:
                if e.code not in (1000,):
                    publish("stt.error", {"reason": "deepgram disconnected"})

        up = asyncio.create_task(pump_up())
        down = asyncio.create_task(pump_down())
        try:
            await asyncio.wait({up, down}, return_when=asyncio.ALL_COMPLETED, timeout=30)
        finally:
            for t in (up, down):
                t.cancel()
            await dg.close()
            # flush leftover finals as an utterance if Deepgram died before speech_final
            if finals:
                publish("stt.utterance", {"text": " ".join(finals),
                                          "t_release": t_release,
                                          "t_utterance": time.time()})
