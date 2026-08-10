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
    def __init__(self): self.utter = 0; self.audio = 0; self.last = None
    def record_utterance(self, t_release, t_utterance):
        self.utter += 1
        self.last = (t_release, t_utterance)
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
    # good input is unchanged by the guard: ms epoch -> seconds
    assert turnlog.last == (1.0, 1000.5)
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


def test_speakable_catches_fenced_and_bracketed_json():
    # a ```json-fenced (possibly truncated) reply must not be read aloud,
    # backticks and braces included
    assert speakable('```json\n{"spoken": "Hi.", "display": "Hi."') == UNCLEAR_LINE
    assert speakable('```\n{"spoken": "Hi."}\n```') == UNCLEAR_LINE
    assert speakable('[{"spoken": "Hi."}]') == UNCLEAR_LINE
    assert speakable("```") == UNCLEAR_LINE          # a bare fence carries nothing
    # non-JSON text is still spoken as-is
    assert speakable("Plain answer.") == "Plain answer."


def test_speakable_strips_carriage_returns():
    # CRLF collapses to a bare newline (a naive .replace("\r", " ") would leave
    # a stray trailing space); a lone CR still becomes a space.
    assert speakable("one.\r\ntwo.") == "one.\ntwo."
    assert speakable("one.\rtwo.") == "one. two."


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


# --- guards added after the M2 Task 6 review --------------------------------
# Every await and every callback inside the loop must be guarded: an unguarded
# raise ends run_butler_brain, the lifespan never restarts it, and Marlowe goes
# deaf until the process restarts -- silently.

async def test_failed_preconnect_does_not_kill_the_brain(tmp_path):
    class BadSpeaker(FakeSpeaker):
        async def preconnect(self): raise RuntimeError("no network")

    bus = EventBus()
    butler, speaker, turnlog = FakeButler(), BadSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, butler, speaker, turnlog))
    await asyncio.sleep(0)
    err_fut = asyncio.ensure_future(_drain_until(bus, "butler.error"))
    bus.publish("wake", {})
    err = await err_fut
    assert "preconnect failed" in err["data"]["reason"]
    # the loop must still be alive and still answer
    answer_fut = asyncio.ensure_future(_drain_until(bus, "butler.answer"))
    bus.publish("command.received", {"text": "still there?"})
    answer = await answer_fut
    assert answer["data"]["display"]
    assert not task.done()
    task.cancel()


async def test_failed_speak_does_not_kill_the_brain(tmp_path):
    class MuteSpeaker(FakeSpeaker):
        async def speak(self, text): raise RuntimeError("tts down")

    bus = EventBus()
    butler, speaker, turnlog = FakeButler(), MuteSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, butler, speaker, turnlog))
    await asyncio.sleep(0)
    err_fut = asyncio.ensure_future(_drain_until(bus, "butler.error"))
    bus.publish("command.received", {"text": "hi"})
    err = await err_fut
    assert "speak failed" in err["data"]["reason"]
    # a dead voice is not a dead brain: the next turn still produces an answer
    answer_fut = asyncio.ensure_future(_drain_until(bus, "butler.answer"))
    bus.publish("command.received", {"text": "still there?"})
    answer = await answer_fut
    assert answer["data"]["display"]
    assert not task.done()
    task.cancel()


async def test_failed_turnlog_does_not_kill_the_brain(tmp_path):
    """record_* take possibly-None timestamps straight off the wire.

    A metrics failure publishes metrics.error, NOT butler.error: the console's
    butler.error handler clears #answer/#citations, so a TurnLog hiccup on
    tts.done would otherwise wipe a correct answer off screen mid-speech.
    """
    class BadTurnLog(FakeTurnLog):
        def record_utterance(self, t_release, t_utterance):
            raise TypeError("unsupported operand type(s) for -: 'NoneType' and 'float'")
        def record_first_audio(self, t_first_audio):
            raise TypeError("bad timestamp")

    bus = EventBus()
    butler, speaker, turnlog = FakeButler(), FakeSpeaker(), BadTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, butler, speaker, turnlog))
    await asyncio.sleep(0)
    err_fut = asyncio.ensure_future(_drain_until(bus, "metrics.error"))
    bus.publish("tts.done", {"t_first_audio": None})
    err = await err_fut
    assert "turnlog failed" in err["data"]["reason"]
    # a broken metric must not cost the answer on the same event
    answer_fut = asyncio.ensure_future(_drain_until(bus, "butler.answer"))
    bus.publish("stt.utterance", {"text": "still there?", "t_release": None,
                                  "t_utterance": None})
    answer = await answer_fut
    assert answer["data"]["display"]
    assert not task.done()
    task.cancel()


async def test_garbage_t_release_does_not_kill_the_brain(tmp_path):
    """`t_release` is unvalidated wire data — stt.py republishes it verbatim.

    The ms->s division must happen INSIDE the guarded callable. As an argument
    expression it is evaluated in the loop's own frame, so a non-numeric value
    raises past _safe, ends run_butler_brain, and Marlowe goes silently deaf.
    """
    bus = EventBus()
    butler, speaker, turnlog = FakeButler(), FakeSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, butler, speaker, turnlog))
    await asyncio.sleep(0)
    answer_fut = asyncio.ensure_future(_drain_until(bus, "butler.answer"))
    # a non-numeric t_release arrives from the websocket
    bus.publish("stt.utterance", {"text": "hello", "t_release": "not-a-number",
                                  "t_utterance": 1000.5})
    answer = await answer_fut          # the turn must still be answered
    assert answer["data"]["display"]
    assert not task.done()             # and the loop must still be alive
    task.cancel()


async def test_zero_and_none_t_release_are_still_skipped(tmp_path):
    """The guard must not change the None/0 -> None contract metrics.py relies on."""
    for raw in (None, 0):
        bus = EventBus()
        butler, speaker, turnlog = FakeButler(), FakeSpeaker(), FakeTurnLog()
        task = asyncio.create_task(run_butler_brain(bus, butler, speaker, turnlog))
        await asyncio.sleep(0)
        answer_fut = asyncio.ensure_future(_drain_until(bus, "butler.answer"))
        bus.publish("stt.utterance", {"text": "hi", "t_release": raw,
                                      "t_utterance": 1000.5})
        await answer_fut
        assert turnlog.last == (None, 1000.5)
        task.cancel()


# --- FIX I2: a HANG is not an exception; try/except never sees it -----------

async def test_hung_speak_times_out_instead_of_wedging_the_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(app_brain, "SPEAK_TIMEOUT_S", 0.01)

    class HangingSpeaker(FakeSpeaker):
        async def speak(self, text): await asyncio.sleep(60)

    bus = EventBus()
    butler, speaker, turnlog = FakeButler(), HangingSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, butler, speaker, turnlog))
    await asyncio.sleep(0)
    err_fut = asyncio.ensure_future(_drain_until(bus, "butler.error"))
    bus.publish("command.received", {"text": "hi"})
    err = await err_fut
    assert err["data"]["reason"] == "speak failed: timed out"
    # the loop is serial: a stuck speak would have eaten every later turn
    answer_fut = asyncio.ensure_future(_drain_until(bus, "butler.answer"))
    bus.publish("command.received", {"text": "still there?"})
    assert (await answer_fut)["data"]["display"]
    assert not task.done()
    task.cancel()


async def test_hung_preconnect_times_out_instead_of_wedging_the_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(app_brain, "PRECONNECT_TIMEOUT_S", 0.01)

    class HangingSpeaker(FakeSpeaker):
        async def preconnect(self): await asyncio.sleep(60)

    bus = EventBus()
    butler, speaker, turnlog = FakeButler(), HangingSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, butler, speaker, turnlog))
    await asyncio.sleep(0)
    err_fut = asyncio.ensure_future(_drain_until(bus, "butler.error"))
    bus.publish("wake", {})
    err = await err_fut
    assert err["data"]["reason"] == "preconnect failed: timed out"
    answer_fut = asyncio.ensure_future(_drain_until(bus, "butler.answer"))
    bus.publish("command.received", {"text": "still there?"})
    assert (await answer_fut)["data"]["display"]
    assert not task.done()
    task.cancel()


# --- FIX I3: citations must be validated against real notes (spec §4) -------

class TwoCiteButler(FakeButler):
    async def ask(self, text):
        self.asked.append(text)
        return {"spoken": "Session 2.",
                "display": "At [[Tibet Session 2]] and [[Invented Note]].",
                "citations": ["Tibet Session 2", "Invented Note"]}


async def test_hallucinated_citation_is_dropped_before_publishing(tmp_path):
    """A made-up [[wikilink]] renders as a chip identical to a real one."""
    async def validator(titles):
        return [t for t in titles if t == "Tibet Session 2"]

    bus = EventBus()
    speaker, turnlog = FakeSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, TwoCiteButler(), speaker, turnlog,
                                                validate_citations=validator))
    await asyncio.sleep(0)
    answer_fut = asyncio.ensure_future(_drain_until(bus, "butler.answer"))
    bus.publish("command.received", {"text": "hi"})
    answer = await answer_fut
    assert answer["data"]["citations"] == ["Tibet Session 2"]   # the real one survives
    task.cancel()


async def test_no_validator_publishes_citations_unchanged(tmp_path):
    bus = EventBus()
    speaker, turnlog = FakeSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, TwoCiteButler(), speaker, turnlog))
    await asyncio.sleep(0)
    answer_fut = asyncio.ensure_future(_drain_until(bus, "butler.answer"))
    bus.publish("command.received", {"text": "hi"})
    answer = await answer_fut
    assert answer["data"]["citations"] == ["Tibet Session 2", "Invented Note"]
    task.cancel()


async def test_broken_validator_does_not_kill_the_brain_or_the_answer(tmp_path):
    """A validator failure costs verification, not the turn -- and it reports on
    metrics.error, because butler.error clears #answer/#citations on the console."""
    async def boom(titles): raise RuntimeError("vault walk failed")

    bus = EventBus()
    speaker, turnlog = FakeSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, TwoCiteButler(), speaker, turnlog,
                                                validate_citations=boom))
    await asyncio.sleep(0)
    err_fut = asyncio.ensure_future(_drain_until(bus, "metrics.error"))
    answer_fut = asyncio.ensure_future(_drain_until(bus, "butler.answer"))
    bus.publish("command.received", {"text": "hi"})
    assert "citation check failed" in (await err_fut)["data"]["reason"]
    answer = await answer_fut
    assert answer["data"]["display"]                       # the answer survived
    assert answer["data"]["citations"] == ["Tibet Session 2", "Invented Note"]
    assert not task.done()
    task.cancel()


async def test_app_validator_checks_real_files(tmp_path):
    """server/app.py's validator resolves each cited title to a real .md note."""
    from server.app import _existing_note_titles

    (tmp_path / "Wiki").mkdir(parents=True)
    (tmp_path / "Wiki" / "Tibet Session 2.md").write_text("hi", encoding="utf-8")
    found = _existing_note_titles(["Tibet Session 2", "Invented Note"], tmp_path)
    assert found == {"Tibet Session 2"}
    # a title full of glob metacharacters is matched literally, not as a pattern
    assert _existing_note_titles(["*"], tmp_path) == set()


# --- the first live failure: API/auth errors must be spoken usefully --------
# Butler.ask now raises ButlerUnavailable when the SDK reports a transport-level
# failure (observed live: a revoked OAuth token was read aloud, verbatim, as
# the "answer"). The brain must speak a line that tells Keke what to DO and
# route the raw error text to the console only.

async def test_auth_failure_speaks_a_useful_line_not_the_raw_error(tmp_path):
    from server.butler import ButlerUnavailable

    class DeadButler(FakeButler):
        async def ask(self, text):
            raise ButlerUnavailable(
                "login expired",
                "Failed to authenticate. API Error: 401 OAuth access token "
                "has been revoked.", "401")

    bus = EventBus()
    speaker, turnlog = FakeSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, DeadButler(), speaker, turnlog))
    await asyncio.sleep(0)
    err_fut = asyncio.ensure_future(_drain_until(bus, "butler.error"))
    bus.publish("command.received", {"text": "hi"})
    err = await err_fut
    assert "login" in err["data"]["reason"].lower()
    assert "401" in err["data"]["detail"]           # raw text goes to the console
    spoken = " ".join(speaker.spoke)
    assert "401" not in spoken and "API Error" not in spoken  # never recite the raw error
    assert "login" in spoken.lower()                # tell Keke what to do
    assert not task.done()
    task.cancel()


async def test_unmapped_unavailable_reason_speaks_the_default_line(tmp_path):
    from server.butler import ButlerUnavailable

    class DeadButler(FakeButler):
        async def ask(self, text):
            raise ButlerUnavailable("the model is unavailable", "boom", None)

    bus = EventBus()
    speaker, turnlog = FakeSpeaker(), FakeTurnLog()
    task = asyncio.create_task(run_butler_brain(bus, DeadButler(), speaker, turnlog))
    await asyncio.sleep(0)
    err_fut = asyncio.ensure_future(_drain_until(bus, "butler.error"))
    bus.publish("command.received", {"text": "hi"})
    err = await err_fut
    assert err["data"]["reason"] == "the model is unavailable"
    assert speaker.spoke == [app_brain.UNAVAILABLE_DEFAULT]
    assert not task.done()
    task.cancel()


# --- the lifespan really starts the brain -----------------------------------

def test_lifespan_starts_the_butler_brain(tmp_path):
    """`app.router.lifespan_context = _lifespan` is load-bearing but invisible.

    If that assignment ever stopped taking effect the server would boot
    identically -- /health ok, no traceback, the same uvicorn log lines -- and
    the brain would simply never run. Every other test builds a TestClient
    WITHOUT a `with` block, so the lifespan never executes under pytest at all.
    This drives a real turn THROUGH the app's own bus inside the lifespan, so it
    fails if the brain task is merely created-and-not-running, or never created.
    """
    from fastapi.testclient import TestClient

    from server.app import create_app

    class ClosableButler(FakeButler):
        async def close(self): pass

    app = create_app(base_dir=tmp_path)
    # Swap in fakes BEFORE the lifespan runs: it captures app.state.butler /
    # .speaker when it creates the brain task, and the real ones would try to
    # reach Anthropic and ElevenLabs.
    app.state.butler = ClosableButler()
    app.state.speaker = FakeSpeaker()

    with TestClient(app, base_url="http://127.0.0.1:7777") as client:  # `with` runs lifespan
        assert getattr(app.state, "butler", None) is not None

        async def probe():
            # Run inside the app's event loop (the bus queues live there).
            cid, q = app.state.bus.subscribe()
            try:
                app.state.bus.publish("command.received", {"text": "still there?"})
                while True:
                    ev = await asyncio.wait_for(q.get(), 2.0)
                    if ev and ev["type"] == "butler.answer":
                        return ev
            finally:
                app.state.bus.unsubscribe(cid)

        answer = client.portal.call(probe)
        assert answer["data"]["display"]
        assert app.state.butler.asked == ["still there?"]
