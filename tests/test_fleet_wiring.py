import asyncio
import time
from types import SimpleNamespace

from server.app_brain import run_butler_brain
from server.bus import EventBus
from server.discovery import Candidate
from server.registry import Registry
from server.router import Router


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


class FakeFleet:
    def __init__(self, worker_paths=("/p/soccer",)):
        self.calls = []
        self.workers = [SimpleNamespace(path=p, project=p.rsplit("/", 1)[-1])
                        for p in worker_paths]
    async def spawn(self, project, path, task):
        self.calls.append(("spawn", project, path, task))
        return "On it, sir — test worker running."
    def steer_path(self, path, text):
        self.calls.append(("steer", path, text))
        return "Told it, sir."
    async def stop(self, path):
        self.calls.append(("stop", path))
        return "Stopped it, sir."
    def status_line(self):
        return "soccer is active turn."
    def transcript(self, path):
        return [{"who": "worker", "text": "hello from the worktree"}]
    def one_breath(self, path):
        return "soccer is mid-task, sir."
    def deliver_approval(self, nonce, approved):
        self.calls.append(("deliver", nonce, approved))
        return True


def confirmed_registry(*names, kind="code"):
    r = Registry()
    r.merge_candidates([Candidate(path=f"/p/{n}", name=n, sources=["t"])
                        for n in names])
    for n in names:
        r.confirm(n, kind=kind)
    return r


async def brain(bus, butler, spk, **kw):
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(), **kw))
    await asyncio.sleep(0)
    return task


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


async def test_spawn_verb_reaches_the_fleet_with_the_path():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    fleet = FakeFleet()
    task = await brain(bus, butler, spk, router=Router(),
                       registry=confirmed_registry("soccer"), fleet=fleet)
    bus.publish("command.received",
                {"text": "start work in soccer: fix the login redirect"})
    await asyncio.sleep(0.05)
    assert ("spawn", "soccer", "/p/soccer", "fix the login redirect") in fleet.calls
    assert any(s.startswith("On it") for s in spk.spoke)   # the fleet's words, spoken
    assert butler.asked == []
    task.cancel()


async def test_steer_and_stop_verbs_dispatch():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    fleet = FakeFleet()
    task = await brain(bus, butler, spk, router=Router(),
                       registry=confirmed_registry("soccer"), fleet=fleet)
    bus.publish("command.received", {"text": "tell soccer to run the tests"})
    await asyncio.sleep(0.05)
    bus.publish("command.received", {"text": "stop soccer"})
    await asyncio.sleep(0.05)
    assert ("steer", "/p/soccer", "run the tests") in fleet.calls
    assert ("stop", "/p/soccer") in fleet.calls
    task.cancel()


async def test_status_verb_speaks_the_fleet_line_when_workers_exist():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = await brain(bus, butler, spk, router=Router(),
                       registry=confirmed_registry("soccer"), fleet=FakeFleet())
    bus.publish("command.received", {"text": "what's running right now?"})
    await asyncio.sleep(0.05)
    assert any("active turn" in s for s in spk.spoke)
    assert butler.asked == []
    task.cancel()


async def test_status_falls_through_to_the_butler_when_fleet_is_empty():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = await brain(bus, butler, spk, router=Router(),
                       registry=confirmed_registry("soccer"),
                       fleet=FakeFleet(worker_paths=()))
    fut = asyncio.ensure_future(_drain(bus, "butler.answer"))
    bus.publish("command.received", {"text": "what's running right now?"})
    await fut
    assert butler.asked == ["what's running right now?"]   # M3.1 behavior kept
    task.cancel()


async def test_pull_up_publishes_the_owned_transcript():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = await brain(bus, butler, spk, router=Router(),
                       registry=confirmed_registry("soccer"), fleet=FakeFleet())
    fut = asyncio.ensure_future(_drain(bus, "fleet.transcript"))
    bus.publish("command.received", {"text": "pull up soccer"})
    ev = await fut
    assert ev["data"]["path"] == "/p/soccer"
    assert ev["data"]["lines"][0]["text"] == "hello from the worktree"
    assert any("mid-task" in s for s in spk.spoke)         # the one-breath status
    task.cancel()


async def test_bare_pull_it_up_resolves_the_only_worker():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = await brain(bus, butler, spk, router=Router(),
                       registry=confirmed_registry("soccer"), fleet=FakeFleet())
    fut = asyncio.ensure_future(_drain(bus, "fleet.transcript"))
    bus.publish("command.received", {"text": "pull it up"})
    ev = await fut
    assert ev["data"]["path"] == "/p/soccer"
    task.cancel()


async def test_voice_approval_is_delivered_to_the_fleet():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router, fleet = Router(), FakeFleet()
    a = router.open_approval("soccer", "Bash: npm test",
                             now=time.time(), path="/p/soccer")
    task = await brain(bus, butler, spk, router=router,
                       registry=confirmed_registry("soccer"), fleet=fleet)
    fut = asyncio.ensure_future(_drain(bus, "approval.resolved"))
    bus.publish("command.received", {"text": "yes, go ahead"})
    ev = await fut
    assert ev["data"]["outcome"] == "approved"
    assert ("deliver", a.nonce, True) in fleet.calls       # the worker unblocks
    task.cancel()


async def test_a_mixed_polarity_answer_asks_and_leaves_the_approval_pending():
    """"sure, cancel that" / "yeah, stop it" / "sure, stop soccer" used to
    APPROVE the pending tool call (_AFFIRM anchors on the opener; _DENY never
    sees the refusal). The brain must ask a clarifying question instead —
    nothing resolved, nothing delivered, card still on screen."""
    for said in ("sure, cancel that", "yeah, stop it", "sure, stop soccer"):
        bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
        router, fleet = Router(), FakeFleet()
        router.open_approval("soccer", "Bash: rm -rf build",
                             now=time.time(), path="/p/soccer")
        task = await brain(bus, butler, spk, router=router,
                           registry=confirmed_registry("soccer"), fleet=fleet)
        cid, q = bus.subscribe()
        bus.publish("command.received", {"text": said})
        await asyncio.sleep(0.05)
        resolved = []
        while not q.empty():
            ev = q.get_nowait()
            if ev and ev["type"] == "approval.resolved":
                resolved.append(ev["data"])
        bus.unsubscribe(cid)
        assert resolved == [], said                        # nothing resolved
        assert all(c[0] != "deliver" for c in fleet.calls), said
        assert len(router.pending_approvals()) == 1, said  # card stays
        assert any("yes and a no" in s for s in spk.spoke), said
        assert "Approved, sir." not in spk.spoke, said
        assert butler.asked == [], said                    # never the model's turn
        task.cancel()


async def test_a_deliver_failure_speaks_truth_and_publishes_nothing():
    """Voice path, deliver_approval raises. Deliver runs FIRST (the /approval
    order), so no approval.resolved goes out for a worker that never got the
    decision — and the turn ends with a truthful failure sentence instead of
    falling through to butler.ask("yes, go ahead")."""
    class BoomDeliverFleet(FakeFleet):
        def deliver_approval(self, nonce, approved):
            raise RuntimeError("delivery broke")
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router, fleet = Router(), BoomDeliverFleet()
    router.open_approval("soccer", "Bash: npm test",
                         now=time.time(), path="/p/soccer")
    task = await brain(bus, butler, spk, router=router,
                       registry=confirmed_registry("soccer"), fleet=fleet)
    cid, q = bus.subscribe()
    bus.publish("command.received", {"text": "yes, go ahead"})
    await asyncio.sleep(0.05)
    resolved, errors = [], []
    while not q.empty():
        ev = q.get_nowait()
        if ev and ev["type"] == "approval.resolved":
            resolved.append(ev["data"])
        if ev and ev["type"] == "butler.error":
            errors.append(ev["data"]["reason"])
    bus.unsubscribe(cid)
    assert resolved == []                      # no false "approved" announce
    assert any("approval handling failed" in r for r in errors)
    assert "Approved, sir." not in spk.spoke   # never claims success
    assert any("failed" in s.lower() for s in spk.spoke)   # the truthful line
    assert butler.asked == []                  # no unrelated vault answer
    assert not task.done()                     # the brain survives
    task.cancel()


async def test_approval_request_cards_are_spoken():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = await brain(bus, butler, spk, router=Router(),
                       registry=confirmed_registry("soccer"), fleet=FakeFleet())
    bus.publish("approval.request",
                {"nonce": "n1", "project": "soccer", "tool": "Bash",
                 "args": "npm test",
                 "question": "soccer wants Bash — npm test. Approve or deny, sir?"})
    await asyncio.sleep(0.05)
    assert any("Approve or deny" in s for s in spk.spoke)
    task.cancel()


async def test_the_brain_survives_a_fleet_explosion():
    class BoomFleet:
        workers = []                                       # parse stays sane
        async def spawn(self, *a): raise RuntimeError("fleet broke")
        def steer_path(self, *a): raise RuntimeError("fleet broke")
        async def stop(self, *a): raise RuntimeError("fleet broke")
        def status_line(self): raise RuntimeError("fleet broke")
        def transcript(self, *a): raise RuntimeError("fleet broke")
        def one_breath(self, *a): raise RuntimeError("fleet broke")
        def deliver_approval(self, *a): raise RuntimeError("fleet broke")
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = await brain(bus, butler, spk, router=Router(),
                       registry=confirmed_registry("soccer"), fleet=BoomFleet())
    bus.publish("command.received", {"text": "start work in soccer: explode"})
    await asyncio.sleep(0.05)
    assert any("command failed" in s.lower() for s in spk.spoke)
    fut = asyncio.ensure_future(_drain(bus, "butler.answer"))
    bus.publish("command.received", {"text": "where did I leave the Tibet study?"})
    await fut                                              # still alive, still answering
    assert not task.done()
    task.cancel()


async def test_the_brain_survives_a_poisoned_approval_request_event():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = await brain(bus, butler, spk, router=Router(),
                       registry=confirmed_registry("soccer"), fleet=FakeFleet())
    bus.publish("approval.request", None)                  # data=None, no dict at all
    await asyncio.sleep(0.05)
    assert any("permission" in s.lower() for s in spk.spoke)  # fallback line
    assert not task.done()
    task.cancel()


async def test_fleet_verbs_without_a_fleet_stay_honest():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = await brain(bus, butler, spk, router=Router(),
                       registry=confirmed_registry("soccer"))   # fleet=None
    bus.publish("command.received", {"text": "start work in soccer: fix login"})
    await asyncio.sleep(0.05)
    assert any("can't run that yet" in s for s in spk.spoke)
    assert not task.done()
    task.cancel()
