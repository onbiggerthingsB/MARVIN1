"""SpeakEngine: ElevenLabs Flash stream-input with reconnect; `say` fallback."""
from __future__ import annotations

import asyncio
import base64
import json
import time

import websockets

ELEVEN_BASE = "wss://api.elevenlabs.io"


class SpeakEngine:
    def __init__(self, voice_id: str, api_key: str, base_url: str,
                 publish, send_audio):
        self.voice_id = voice_id
        self.api_key = api_key
        self.base_url = base_url
        self.publish = publish
        self.send_audio = send_audio
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

    async def speak(self, text: str) -> None:
        async with self._lock:
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
        t0 = time.time()
        await proc.wait()
        self.publish("tts.done", {"t_first_audio": t0, "engine": "say"})

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
