"""The butler brain loop: bus events -> butler.ask -> speak + answer events."""
from __future__ import annotations

import asyncio

FALLBACK_LINE = "Sorry sir, I couldn't reach my brain just now."
UNCLEAR_LINE = "Sorry sir, I didn't get a clean answer that time — it's on screen."

# A single turn must never wedge the loop. permission_mode is "default" and this
# process is headless: if a tool call ever lands outside the allowed surface
# there is nobody to answer the permission prompt and ask() would hang forever,
# taking every later utterance with it. Time it out and treat that like any
# other failure.
ASK_TIMEOUT_S = 120


def speakable(spoken) -> str:
    """What to actually read aloud, given the butler's `spoken` field.

    parse_butler_output falls back to plain text when the model's JSON has only
    empty values -- which makes the raw serialized JSON the `spoken` string. Read
    aloud that is a mouthful of braces and quotes, so anything empty or
    JSON-looking is replaced with a short canned line. `\\r` is stripped because
    bulleted replies leave stray carriage returns that some voices verbalize.
    CRLF is collapsed to a bare newline FIRST -- replacing `\\r` alone would turn
    every `\\r\\n` into a stray trailing space before the newline.
    """
    text = (spoken or "").replace("\r\n", "\n").replace("\r", " ").strip()
    if not text or text.startswith("{"):
        return UNCLEAR_LINE
    return text


async def run_butler_brain(bus, butler, speaker, turnlog):
    """Subscribe to the bus and drive the butler. Never raises out of the loop."""
    cid, q = bus.subscribe()

    async def _speak(text):
        # A TTS failure is not a reason to lose the brain for the rest of the
        # session; surface it and keep looping.
        try:
            await speaker.speak(text)
        except Exception as e:  # noqa: BLE001
            bus.publish("butler.error", {"reason": f"speak failed: {e}"})

    async def _preconnect():
        # Same reasoning as _speak: a 401, a DNS failure or a dropped network on
        # the TTS pre-warm would otherwise raise straight out of this loop and
        # end the task -- JARVIS goes deaf until the process restarts, silently,
        # because nothing publishes on that path. M1's fire-and-forget
        # create_task(preconnect()) could not kill the loop either.
        try:
            await speaker.preconnect()
        except Exception as e:  # noqa: BLE001 — a wake must never kill the brain
            bus.publish("butler.error", {"reason": f"preconnect failed: {e}"})

    def _safe(fn, *args, **kwargs):
        # Metrics bookkeeping takes possibly-None timestamps straight off the
        # wire; a TurnLog that trips over one must not take the brain with it.
        try:
            fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — metrics must never kill the brain
            bus.publish("butler.error", {"reason": f"turnlog failed: {e}"})

    def _record_audio(t_first_audio):
        turnlog.record_first_audio(t_first_audio)
        bus.publish("metrics.turn", turnlog.summary())

    try:
        while True:
            event = await q.get()
            if event is None:                       # evicted; resubscribe
                cid, q = bus.subscribe()
                continue
            etype = event["type"]
            data = event.get("data", {})
            if etype == "wake":
                await _preconnect()
                continue
            if etype == "tts.done":
                _safe(_record_audio, data.get("t_first_audio"))
                continue
            if etype in ("stt.utterance", "command.received"):
                if etype == "stt.utterance":
                    _safe(turnlog.record_utterance,
                          t_release=(data.get("t_release") or 0) / 1000 or None,
                          t_utterance=data.get("t_utterance"))
                text = (data.get("text") or "").strip()
                if not text:
                    continue
                try:
                    answer = await asyncio.wait_for(butler.ask(text), ASK_TIMEOUT_S)
                except Exception as e:  # noqa: BLE001 — never let the brain die
                    reason = "timed out" if isinstance(e, asyncio.TimeoutError) else str(e)
                    bus.publish("butler.error", {"reason": reason})
                    await _speak(FALLBACK_LINE)
                    continue
                bus.publish("butler.answer",
                            {"display": (answer or {}).get("display", ""),
                             "citations": (answer or {}).get("citations", [])})
                await _speak(speakable((answer or {}).get("spoken")))
    finally:
        bus.unsubscribe(cid)
