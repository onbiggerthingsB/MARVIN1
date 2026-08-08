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


def _record_utterance(turnlog, data):
    """Convert the wire's millisecond `t_release` and hand it to the TurnLog.

    The division lives HERE, not in the argument list of the `_safe(...)` call.
    An argument expression is evaluated in the caller's frame BEFORE the guarded
    callable is entered, so `(data.get("t_release") or 0) / 1000` written inline
    would raise straight out of the brain loop -- and `t_release` is unvalidated
    wire data (server/stt.py republishes whatever the /mic websocket JSON held).
    A string, list or dict there would kill the task silently. Inside this
    function the same TypeError is just another guarded failure.

    Behaviour for good input is unchanged: None/0 -> None (metrics skipped),
    a real millisecond epoch -> seconds.
    """
    raw = data.get("t_release")
    t_release = (raw or 0) / 1000 or None
    turnlog.record_utterance(t_release=t_release, t_utterance=data.get("t_utterance"))


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
        # Metrics bookkeeping takes possibly-None (and possibly non-numeric)
        # timestamps straight off the wire; a TurnLog that trips over one must
        # not take the brain with it.
        #
        # This publishes metrics.error, NOT butler.error: the console's
        # butler.error handler clears #answer and #citations, so routing a
        # metrics hiccup there would wipe a correct, already-rendered answer off
        # screen while JARVIS is still speaking it (tts.done arrives mid-speech).
        # Every _safe call site is a turnlog path, so the event type is fixed
        # here rather than parameterized.
        try:
            fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — metrics must never kill the brain
            bus.publish("metrics.error", {"reason": f"turnlog failed: {e}"})

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
                    _safe(_record_utterance, turnlog, data)
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
