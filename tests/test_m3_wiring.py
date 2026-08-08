import asyncio
import contextlib
import time

from server.bus import EventBus
from server.discovery import Candidate
from server.registry import Registry
from server.router import Router
from server.app_brain import run_butler_brain


class FakeButler:
    def __init__(self): self.asked = []
    async def ask(self, text):
        self.asked.append(text)
        return {"spoken": "vault answer", "display": "d", "citations": []}


class FakeSpeaker:
    def __init__(self): self.spoke = []
    async def speak(self, t): self.spoke.append(t)
    async def preconnect(self): pass


class FakeTurnLog:
    def record_utterance(self, **k): pass
    def record_first_audio(self, *a): pass
    def summary(self): return {}


def confirmed_registry(*names, kind="code"):
    r = Registry()
    r.merge_candidates([Candidate(path=f"/p/{n}", name=n, sources=["t"]) for n in names])
    for n in names:
        r.confirm(n, kind=kind)
    return r


async def _drain(bus, type_, timeout=2.0):
    cid, q = bus.subscribe()
    async def wait():
        while True:
            ev = await q.get()
            if ev and ev["type"] == type_:
                return ev
    try:
        return await asyncio.wait_for(wait(), timeout)
    finally:
        bus.unsubscribe(cid)


async def test_a_question_still_reaches_the_butler():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=confirmed_registry("soccer")))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "butler.answer"))
    bus.publish("command.received", {"text": "where did I leave the Tibet study?"})
    await fut
    assert butler.asked == ["where did I leave the Tibet study?"]
    task.cancel()


async def test_a_dangerous_verb_is_routed_and_never_reaches_the_butler():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=confirmed_registry("soccer")))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "router.command"))
    bus.publish("command.received", {"text": "pull up soccer"})
    ev = await fut
    assert ev["data"]["verb"] == "pull_up" and ev["data"]["project"] == "soccer"
    assert butler.asked == []          # the model was never consulted
    task.cancel()


async def test_a_trade_request_is_refused_aloud_and_not_routed():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=confirmed_registry("quant agent", kind="finance")))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "router.command"))
    bus.publish("command.received", {"text": "buy 10 shares of NVDA"})
    ev = await fut
    assert ev["data"]["verb"] == "refuse_trade"
    await asyncio.sleep(0.05)
    assert any("stock system" in s.lower() for s in spk.spoke)
    assert butler.asked == []
    task.cancel()


async def test_capture_verb_actually_appends_to_the_daily_inbox(tmp_path, monkeypatch):
    # The day-one regression: "remember/note that ..." must WRITE, not stub.
    monkeypatch.setenv("JARVIS_VAULT", str(tmp_path))
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=confirmed_registry("soccer")))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "router.command"))
    bus.publish("command.received", {"text": "note that Dr. Kong wants the HRV protocol by Friday"})
    ev = await fut
    assert ev["data"]["verb"] == "capture"
    for _ in range(100):                       # wait for the off-loop append
        daily = list((tmp_path / "Daily").glob("*.md"))
        if daily and "Noted, sir." in spk.spoke:
            break
        await asyncio.sleep(0.01)
    assert len(daily) == 1
    assert "Dr. Kong wants the HRV protocol by Friday" in daily[0].read_text(encoding="utf-8")
    assert "Noted, sir." in spk.spoke
    assert butler.asked == []                  # the model was never consulted
    task.cancel()


async def test_the_brain_survives_a_router_explosion():
    class BoomRouter(Router):
        def parse(self, spoken, registry): raise RuntimeError("router broke")
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=BoomRouter(), registry=confirmed_registry("soccer")))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "butler.answer"))
    bus.publish("command.received", {"text": "pull up soccer"})
    await fut                       # falls through to the butler rather than dying
    assert not task.done()
    task.cancel()


async def test_brain_without_a_router_behaves_exactly_as_m2():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(bus, butler, spk, FakeTurnLog()))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "butler.answer"))
    bus.publish("command.received", {"text": "pull up soccer"})
    await fut
    assert butler.asked == ["pull up soccer"]
    task.cancel()


# ---- deny-vs-stop tie-break (binding note carried from Task 3's review) ----
# The router's denial vocabulary ("stop", "cancel") overlaps the stop VERB, so
# call order alone cannot arbitrate. The brain gates on utterance SHAPE: a
# parse that resolved a confirmed project ("stop soccer") is a fleet command
# and wins; an utterance that names no project ("yes", bare "no") may answer a
# pending approval. These tests pin both directions.

async def test_stop_project_beats_denial_when_it_names_a_confirmed_project():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router, registry = Router(), confirmed_registry("soccer")
    router.open_approval("soccer", "npm test", now=time.time())
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(), router=router, registry=registry))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "router.command"))
    bus.publish("command.received", {"text": "stop soccer"})
    ev = await fut
    assert ev["data"]["verb"] == "stop" and ev["data"]["project"] == "soccer"
    assert len(router.pending_approvals()) == 1   # approval NOT consumed as a denial
    assert butler.asked == []
    task.cancel()


async def test_a_bare_yes_resolves_the_pending_approval_not_the_butler():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router, registry = Router(), confirmed_registry("soccer")
    router.open_approval("soccer", "npm test", now=time.time())
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(), router=router, registry=registry))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "approval.resolved"))
    bus.publish("command.received", {"text": "yes"})
    ev = await fut
    assert ev["data"]["outcome"] == "approved"
    assert ev["data"]["project"] == "soccer" and ev["data"]["tool"] == "npm test"
    assert router.pending_approvals() == []       # consumed exactly once
    assert butler.asked == []                     # a bare yes never reaches the model
    task.cancel()


async def test_a_bare_no_denies_the_pending_approval():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router, registry = Router(), confirmed_registry("soccer")
    router.open_approval("soccer", "npm test", now=time.time())
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(), router=router, registry=registry))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "approval.resolved"))
    bus.publish("command.received", {"text": "no"})
    ev = await fut
    assert ev["data"]["outcome"] == "denied"
    assert router.pending_approvals() == []
    assert butler.asked == []
    task.cancel()


# ---- guard coverage: no router or finance fault may kill the loop ----------

async def test_the_brain_survives_a_pending_approvals_explosion():
    class BoomPending(Router):
        def pending_approvals(self): raise RuntimeError("pending exploded")
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=BoomPending(), registry=confirmed_registry("soccer")))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "butler.answer"))
    bus.publish("command.received", {"text": "where did I leave the Tibet study?"})
    await fut                      # falls through to the butler rather than dying
    assert not task.done()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_the_brain_survives_a_portfolio_explosion(monkeypatch):
    import server.app_brain as ab
    async def boom(project): raise RuntimeError("finance exploded")
    monkeypatch.setattr(ab, "portfolio_brief", boom)
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=confirmed_registry("quant agent", kind="finance")))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "butler.error"))
    bus.publish("command.received", {"text": "how are the picks doing?"})
    ev = await fut
    assert "portfolio brief failed" in ev["data"]["reason"]
    assert not task.done()         # the loop survives a finance fault
    await asyncio.sleep(0.05)
    assert any("stock system" in s.lower() for s in spk.spoke)
    assert butler.asked == []      # a routed verb still never reaches the model
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
