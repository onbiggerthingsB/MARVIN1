import asyncio

from server import app_brain
from server.app_brain import FALLBACK_LINE, UNCLEAR_LINE, run_butler_brain, speakable
from server.bus import EventBus


class FakeButler:
    def __init__(self): self.asked = []
    async def ask(self, text):
        self.asked.append(text)
        return {"spoken": "Session 2.", "display": "At [[Tibet Session 2]].",
                "citations": ["Tibet Session 2"]}


class FakeSpeaker:
    def __init__(self): self.spoke = []; self.preconnects = 0
    async def speak(self, text): self.spoke.append(text)
    async def preconnect(self): self.preconnects += 1


class FakeTurnLog:
    def __init__(self): self.utter = 0; self.audio = 0
    def record_utterance(self, t_release, t_utterance): self.utter += 1
    def record_first_audio(self, t_first_audio): self.audio += 1
    def summary(self): return {"turns": self.utter}


async def _drain_until(bus_events, type_, timeout=2.0):
    async def wait():
        cid, q = bus_events.subscribe()
        try:
            while True:
                ev = await q.get()
                if ev and ev["type"] == type_:
                    return ev
        finally:
            bus_events.unsubscribe(cid)
    return await asyncio.wait_for(wait(), timeout)


async def test_utterance_drives_butler_and_speaks_and_answers(tmp_path):
    bus = EventBus()
    butler, speaker, turnlog = FakeButler(), FakeSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, butler, speaker, turnlog))
    await asyncio.sleep(0)  # let the brain subscribe
    answer_fut = asyncio.ensure_future(_drain_until(bus, "butler.answer"))
    bus.publish("stt.utterance", {"text": "where did I leave the tibet study?",
                                  "t_release": 1000.0, "t_utterance": 1000.5})
    answer = await answer_fut
    assert answer["data"]["citations"] == ["Tibet Session 2"]
    assert butler.asked == ["where did I leave the tibet study?"]
    assert speaker.spoke == ["Session 2."]
    assert turnlog.utter == 1
    task.cancel()


async def test_wake_preconnects(tmp_path):
    bus = EventBus()
    butler, speaker, turnlog = FakeButler(), FakeSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, butler, speaker, turnlog))
    await asyncio.sleep(0)
    bus.publish("wake", {})
    await asyncio.sleep(0.05)
    assert speaker.preconnects == 1
    task.cancel()


async def test_butler_failure_speaks_fallback_not_crash(tmp_path):
    class Boom(FakeButler):
        async def ask(self, text): raise RuntimeError("no brain")
    bus = EventBus()
    speaker, turnlog = FakeSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, Boom(), speaker, turnlog))
    await asyncio.sleep(0)
    err_fut = asyncio.ensure_future(_drain_until(bus, "butler.error"))
    bus.publish("command.received", {"text": "hi"})
    err = await err_fut
    assert "reason" in err["data"]
    assert speaker.spoke  # spoke a fallback line, did not crash
    task.cancel()


# --- guards added after the brief (task-review notes 1-3) -------------------

def test_speakable_never_reads_raw_json_or_nothing_aloud():
    # parse_butler_output falls back to plain text when the JSON has only empty
    # values, which puts the serialized JSON itself in `spoken`.
    assert speakable('{"spoken": "", "display": "", "citations": []}') == UNCLEAR_LINE
    assert speakable("") == UNCLEAR_LINE
    assert speakable(None) == UNCLEAR_LINE
    assert speakable("   ") == UNCLEAR_LINE


def test_speakable_strips_carriage_returns():
    assert speakable("one.\r\ntwo.") == "one. \ntwo."


async def test_json_shaped_spoken_still_publishes_display(tmp_path):
    class Jsonish(FakeButler):
        async def ask(self, text):
            return {"spoken": '{"spoken": "", "display": ""}',
                    "display": "At [[Tibet Session 2]].", "citations": ["Tibet Session 2"]}
    bus = EventBus()
    speaker, turnlog = FakeSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, Jsonish(), speaker, turnlog))
    await asyncio.sleep(0)
    answer_fut = asyncio.ensure_future(_drain_until(bus, "butler.answer"))
    bus.publish("command.received", {"text": "hi"})
    answer = await answer_fut
    assert answer["data"]["display"] == "At [[Tibet Session 2]]."
    assert speaker.spoke == [UNCLEAR_LINE]
    task.cancel()


async def test_hung_ask_times_out_instead_of_wedging_the_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(app_brain, "ASK_TIMEOUT_S", 0.01)

    class Hang(FakeButler):
        async def ask(self, text):
            await asyncio.sleep(60)  # a headless permission prompt nobody answers

    bus = EventBus()
    speaker, turnlog = FakeSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, Hang(), speaker, turnlog))
    await asyncio.sleep(0)
    err_fut = asyncio.ensure_future(_drain_until(bus, "butler.error"))
    bus.publish("command.received", {"text": "hi"})
    err = await err_fut
    assert err["data"]["reason"] == "timed out"
    assert speaker.spoke == [FALLBACK_LINE]
    task.cancel()
