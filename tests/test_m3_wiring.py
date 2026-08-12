import asyncio
import contextlib
import time

from server.bus import EventBus
from server.discovery import Candidate
from server.registry import Registry
from server.router import Router
from server.app_brain import run_butler_brain
# An approval Marvin has already read aloud. A raw open_approval() models one
# nobody has heard, which a voice yes may no longer resolve.
from tests.test_fleet_wiring import open_spoken


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
    monkeypatch.setenv("MARVIN_VAULT", str(tmp_path))
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
    open_spoken(router, "soccer", "npm test", now=time.time())
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
    open_spoken(router, "soccer", "npm test", now=time.time())
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
    open_spoken(router, "soccer", "npm test", now=time.time())
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


class PendingConfirm:
    """An onboarding that is mid-question but does not understand the reply."""
    awaiting = True
    async def handle_reply(self, text):
        return "ignored"


async def test_a_pending_confirmation_owns_affirm_shaped_speech():
    # "approved" is router-_AFFIRM but not onboarding-_YES. With a repo
    # question pending AND a tool approval pending, it must be consumed by
    # the confirmation gate — never resolve the approval, never reach the model.
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router = Router()
    open_spoken(router, "soccer", "npm test", now=time.time())
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=router, registry=confirmed_registry("soccer"),
        onboarding=PendingConfirm()))
    await asyncio.sleep(0)
    bus.publish("command.received", {"text": "approved"})
    await asyncio.sleep(0.05)
    assert len(router.pending_approvals()) == 1      # approval untouched
    assert butler.asked == []                        # model never consulted
    assert any("yes or a no" in s for s in spk.spoke)
    assert not task.done()
    task.cancel()


async def test_okay_answers_the_repo_question_not_the_tool_approval(tmp_path):
    from server.discovery import Candidate as C
    from server.onboarding import Onboarding
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = Registry()
    reg.merge_candidates([C(path="/p/soccer", name="soccer", sources=["t"])])
    ob = Onboarding(bus, reg, tmp_path / "projects.json")
    await ob.ask_next()                              # repo question now pending
    ob.mark_spoken()                                 # ...and read aloud (the barrier)
    router = Router()
    open_spoken(router, "alethic", "rm -rf build", now=time.time())
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=router, registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("command.received", {"text": "okay"})
    await asyncio.sleep(0.05)
    assert any(p.confirmed for p in reg.projects)    # the repo got its yes
    assert len(router.pending_approvals()) == 1      # the approval did NOT
    assert not task.done()
    task.cancel()


# ---- beat 1: discovery -> a SPOKEN repo confirm -> the next candidate -----
# Every piece below was already unit-tested and every unit passed while the
# flow was dead: nothing asked, nothing spoke the question, and the chain
# stopped after one answer. These pin the WIRING.

def pending_registry(*names):
    """A registry of discovered-but-unconfirmed candidates — what a first boot
    produces, and the only state the confirmation flow has anything to do."""
    r = Registry()
    r.merge_candidates([Candidate(path=f"/p/{n}", name=n, sources=["t"]) for n in names])
    return r


def onboarding_for(bus, registry, tmp_path):
    from server.onboarding import Onboarding
    return Onboarding(bus, registry, tmp_path / "projects.json")


async def _settle(seconds=0.1):
    """Let the serial brain loop drain the events we just published."""
    await asyncio.sleep(seconds)


async def test_the_repo_question_is_spoken_not_only_rendered(tmp_path):
    # Beat 1 is a SPOKEN confirm. confirm.request has always been rendered by
    # the console; the brain never read it aloud.
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry("quant agent")
    ob = onboarding_for(bus, reg, tmp_path)
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("confirm.next", {"trigger": "boot"})
    await _settle()
    assert any("quant agent" in s and "correct repo" in s for s in spk.spoke), spk.spoke
    assert ob.awaiting                                  # and the answer is owned
    task.cancel()


async def test_the_source_question_is_never_read_aloud_twice(tmp_path):
    # The §16 data-source question rides the SAME confirm.request event (the
    # console renders both), but it is spoken by the turn that asked it.
    # Reading it a second time would put a question to Keke whose answer is
    # already pending — the correlation defect this codebase keeps paying for.
    from server.finance_gate import SourceGate
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg, root = finance_fixture(tmp_path)
    gate = SourceGate(bus, reg, tmp_path / "projects.json")
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, finance=gate))
    await asyncio.sleep(0)
    bus.publish("command.received", {"text": "how are the picks doing?"})
    await _settle()
    assert sum(1 for s in spk.spoke if "picks.json" in s) == 1, spk.spoke
    task.cancel()


async def test_a_confirmed_answer_proposes_the_next_candidate(tmp_path):
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry("quant agent", "soccer")
    ob = onboarding_for(bus, reg, tmp_path)
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("confirm.next", {"trigger": "boot"})
    await _settle()
    first = ob._asking.path
    bus.publish("command.received", {"text": "yes"})
    await _settle()
    assert [p.path for p in reg.projects if p.confirmed] == [first]
    assert ob.awaiting and ob._asking.path != first     # the chain kept going
    assert any(ob._asking.name in s and "correct repo" in s
               for s in spk.spoke), spk.spoke          # and the next one was SPOKEN
    task.cancel()


async def test_a_rejection_also_proposes_the_next_candidate(tmp_path):
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry("quant agent", "soccer")
    ob = onboarding_for(bus, reg, tmp_path)
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("confirm.next", {})
    await _settle()
    first = ob._asking.path
    bus.publish("command.received", {"text": "no"})
    await _settle()
    assert not any(p.confirmed for p in reg.projects)   # a no confirms nothing
    assert ob.awaiting and ob._asking.path != first
    task.cancel()


async def test_a_rename_re_asks_the_same_repo_because_it_is_still_pending(tmp_path):
    # "no, I said the trading system" teaches an alias and deliberately leaves
    # the project PENDING — only a real yes confirms. The chain must therefore
    # re-propose THAT project, not skip past it.
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry("quant agent")
    ob = onboarding_for(bus, reg, tmp_path)
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("confirm.next", {})
    await _settle()
    asked = ob._asking.path
    bus.publish("command.received", {"text": "no, I said the trading system"})
    await _settle()
    p = next(p for p in reg.projects if p.path == asked)
    assert "the trading system" in p.aliases and not p.confirmed
    assert ob.awaiting and ob._asking.path == asked     # re-asked, knowing the alias
    task.cancel()


async def test_the_last_candidate_ends_the_chain_truthfully(tmp_path):
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry("soccer")
    ob = onboarding_for(bus, reg, tmp_path)
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("confirm.next", {})
    await _settle()
    bus.publish("command.received", {"text": "yes"})
    await _settle()
    assert not ob.awaiting
    assert any("nothing left" in s.lower() for s in spk.spoke), spk.spoke
    # and it does not loop: exactly one question was ever asked
    assert len([e for e in bus._ring if e["type"] == "confirm.request"]) == 1
    task.cancel()


async def test_a_second_yes_cannot_confirm_a_repo_that_was_never_asked(tmp_path):
    # The double-yes. Keke answers, then says "yes" again while "Noted, sir."
    # is still playing — that second yes was uttered before the next question
    # existed and must never confirm it. The next candidate is proposed
    # THROUGH the bus for exactly this reason: everything already said is
    # drained (as ordinary speech) before onboarding starts awaiting again.
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry("quant agent", "soccer")
    ob = onboarding_for(bus, reg, tmp_path)
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("confirm.next", {})
    await _settle()
    first = ob._asking.path
    bus.publish("command.received", {"text": "yes"})
    bus.publish("command.received", {"text": "yes"})    # queued behind the first
    await _settle()
    assert [p.path for p in reg.projects if p.confirmed] == [first]
    assert ob.awaiting and ob._asking.path != first     # asked, NOT confirmed
    task.cancel()


async def test_the_chain_holds_while_an_approval_is_pending_and_resumes_after(tmp_path):
    # The two questions must never compete for one yes (live demo defect,
    # 2026-08-12): a repo confirm blocks nothing and can wait; a tool approval
    # blocks a real worker and expires. So a confirm.next arriving while an
    # approval is pending HOLDS — no repo question is put on the floor — and
    # the yes goes to the approval. Once it resolves, the chain resumes and
    # the repo question is read aloud before it may be answered.
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry("soccer")
    ob = onboarding_for(bus, reg, tmp_path)
    router = Router()
    open_spoken(router, "alethic", "rm -rf build", now=time.time())
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=router, registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("confirm.next", {})
    await _settle()
    assert not any("correct repo" in s for s in spk.spoke)  # held, not asked
    assert not ob.awaiting
    bus.publish("command.received", {"text": "yes"})
    await _settle()
    assert router.pending_approvals() == []              # the worker got its answer
    assert butler.asked == []
    # ...and the chain resumed: the repo question is now on the floor, spoken
    assert any("correct repo" in s for s in spk.spoke), spk.spoke
    assert ob.awaiting
    assert not any(p.confirmed for p in reg.projects)    # that yes confirmed no repo
    bus.publish("command.received", {"text": "yes"})
    await _settle()
    assert [p.name for p in reg.projects if p.confirmed] == ["soccer"]
    task.cancel()


async def test_the_discover_verb_rescans_and_proposes_a_repo(tmp_path, monkeypatch):
    import server.app_brain as ab
    home = tmp_path / "home"
    (home / "alethic" / ".git").mkdir(parents=True)
    monkeypatch.setattr(ab, "default_home", lambda: home)
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = Registry()                                    # nothing known yet
    ob = onboarding_for(bus, reg, tmp_path)
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "router.command"))
    bus.publish("command.received", {"text": "find my projects"})
    ev = await fut
    assert ev["data"]["verb"] == "discover"
    for _ in range(200):                                # the scan runs off-loop
        if ob.awaiting:
            break
        await asyncio.sleep(0.01)
    assert [p.name for p in reg.projects] == ["alethic"]
    assert any("alethic" in s and "correct repo" in s for s in spk.spoke), spk.spoke
    assert butler.asked == []                           # the model was never consulted
    task.cancel()


async def test_the_brain_survives_an_onboarding_ask_explosion(tmp_path):
    class BoomAsk:
        awaiting = False
        async def handle_reply(self, text): return "ignored"
        async def ask_next(self): raise RuntimeError("ask exploded")
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=Registry(), onboarding=BoomAsk()))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "butler.error"))
    bus.publish("confirm.next", {})
    ev = await fut
    assert "onboarding ask failed" in ev["data"]["reason"]
    assert not task.done()
    fut2 = asyncio.ensure_future(_drain(bus, "butler.answer"))
    bus.publish("command.received", {"text": "where did I leave the Tibet study?"})
    await fut2                                          # still answering
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def finance_fixture(tmp_path):
    """A confirmed finance project with one readable output, plus the gate."""
    import json as _json
    from server.discovery import Candidate as C
    from server.finance_gate import SourceGate
    root = tmp_path / "quant agent"
    root.mkdir()
    (root / "picks.json").write_text(
        _json.dumps([{"symbol": "TSLA", "shares": 3}]), encoding="utf-8")
    reg = Registry()
    reg.merge_candidates([C(path=str(root), name="quant agent", sources=["t"])])
    reg.confirm("quant agent", kind="finance")
    return reg, root


async def test_first_portfolio_ask_confirms_the_source_before_briefing(tmp_path):
    from server.finance_gate import SourceGate
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg, root = finance_fixture(tmp_path)
    gate = SourceGate(bus, reg, tmp_path / "projects.json")
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, finance=gate))
    await asyncio.sleep(0)
    briefed = []
    bus_cid, bus_q = bus.subscribe()
    bus.publish("command.received", {"text": "how are the picks doing?"})
    await asyncio.sleep(0.05)
    assert any("picks.json" in s for s in spk.spoke)     # the question, spoken
    assert gate.awaiting is True
    # no brief yet: nothing was confirmed
    while not bus_q.empty():
        ev = bus_q.get_nowait()
        if ev and ev["type"] == "finance.brief":
            briefed.append(ev)
    assert briefed == []
    task.cancel()


async def test_confirming_the_source_briefs_immediately(tmp_path):
    from server.finance_gate import SourceGate
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg, root = finance_fixture(tmp_path)
    gate = SourceGate(bus, reg, tmp_path / "projects.json")
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, finance=gate))
    await asyncio.sleep(0)
    bus.publish("command.received", {"text": "how are the picks doing?"})
    await asyncio.sleep(0.05)
    fut = asyncio.ensure_future(_drain(bus, "finance.brief"))
    bus.publish("command.received", {"text": "yes"})
    ev = await fut                                        # the yes doubles as "go"
    assert ev["data"]["source"].endswith("picks.json")
    assert reg.projects[0].data_source == str(root / "picks.json")
    assert butler.asked == []                             # model never involved
    task.cancel()


async def test_the_confirmed_brief_targets_the_pinned_project_not_a_rederived_one(tmp_path):
    # Two confirmed finance projects. The gate asked about the SECOND one, so
    # the yes pins that one — but a re-derived find_finance_project() returns
    # the FIRST, which is unpinned and would fall back to scanning its newest
    # readable file: a brief from a source Keke never confirmed (§16). The
    # confirmed branch must brief the exact project the gate pinned.
    import json as _json
    from server.discovery import Candidate as C
    from server.finance_gate import SourceGate
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    root_a = tmp_path / "quant agent"          # what re-derivation selects
    root_a.mkdir()
    (root_a / "decoy.json").write_text(
        _json.dumps([{"symbol": "AAPL", "shares": 1}]), encoding="utf-8")
    root_b = tmp_path / "second system"        # what the gate asked about
    root_b.mkdir()
    (root_b / "picks.json").write_text(
        _json.dumps([{"symbol": "TSLA", "shares": 3}]), encoding="utf-8")
    reg = Registry()
    reg.merge_candidates([C(path=str(root_a), name="quant agent", sources=["t"]),
                          C(path=str(root_b), name="second system", sources=["t"])])
    reg.confirm("quant agent", kind="finance")
    reg.confirm("second system", kind="finance")
    gate = SourceGate(bus, reg, tmp_path / "projects.json")
    proj_b = next(p for p in reg.projects if p.path == str(root_b))
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, finance=gate))
    await asyncio.sleep(0)
    await gate.ask(proj_b)                     # question pending for B
    fut = asyncio.ensure_future(_drain(bus, "finance.brief"))
    bus.publish("command.received", {"text": "yes"})
    ev = await fut
    assert ev["data"]["source"] == str(root_b / "picks.json")   # never the decoy
    assert proj_b.data_source == str(root_b / "picks.json")
    assert butler.asked == []
    task.cancel()


async def test_the_brain_survives_a_source_gate_explosion(tmp_path):
    class BoomGate:
        @property
        def awaiting(self):
            raise RuntimeError("gate broke")
        async def handle_reply(self, text):
            raise RuntimeError("gate broke")
        async def ask(self, project):
            raise RuntimeError("gate broke")
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg, root = finance_fixture(tmp_path)
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, finance=BoomGate()))
    await asyncio.sleep(0)
    # the portfolio verb hits BoomGate.ask; the reply path hits .awaiting —
    # both must cost the turn, never the loop
    bus.publish("command.received", {"text": "how are the picks doing?"})
    await asyncio.sleep(0.05)
    fut = asyncio.ensure_future(_drain(bus, "butler.answer"))
    bus.publish("command.received", {"text": "where did I leave the Tibet study?"})
    await fut
    assert not task.done()
    task.cancel()


# ---- separator-free spawn + the spawn hint (live demo defect, 2026-08) ----

async def test_a_separator_free_spawn_is_routed_not_answered():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=confirmed_registry("soccer")))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "router.command"))
    bus.publish("command.received",
                {"text": "begin work on soccer add a comment to the readme"})
    ev = await fut
    assert ev["data"]["verb"] == "spawn" and ev["data"]["project"] == "soccer"
    assert ev["data"]["argument"] == "add a comment to the readme"
    assert butler.asked == []          # the model was never consulted
    task.cancel()


async def test_a_spawn_shaped_fallthrough_is_explained_not_answered():
    # The dangerous half of the live defect: an unresolvable spawn used to be
    # answered CONVERSATIONALLY by the butler, so the failure read like
    # intended behaviour. The brain now speaks the router's deterministic
    # hint instead, and the model is never consulted.
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=confirmed_registry("soccer")))
    await asyncio.sleep(0)
    bus.publish("command.received",
                {"text": "begin work on flimflam do the thing"})

    async def spoken():
        while not spk.spoke:
            await asyncio.sleep(0.01)
    await asyncio.wait_for(spoken(), 2.0)
    assert butler.asked == []          # never became a conversational answer
    assert "start work" in spk.spoke[0].lower()   # tells Keke how to retry
    task.cancel()


# ---- the confirm chain's exits and the floor arbitration (2026-08-12) -----
# The live first run: 69 discovered repos, one spoken confirm each, no way to
# stop the chain — and while it was open, its terminal ownership of bare
# yes/no ate the owner's answer to a blocked worker's permission request.
# These pin the wiring half of the fix.

async def test_a_mid_chain_tool_approval_is_not_eaten_by_the_repo_question(tmp_path):
    """THE LIVE DEFECT (M3P2 gate, beat 3). A repo question is on the floor
    when a worker blocks on permission. The approval readback displaces the
    repo question; the first bare yes after the switch is confirmed against
    the question it now belongs to (fail closed, one beat); the next yes
    unblocks the worker; and the chain resumes with the repo question read
    aloud AGAIN before it may be answered."""
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry("quant agent", "soccer")
    ob = onboarding_for(bus, reg, tmp_path)
    router = Router()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=router, registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("confirm.next", {"trigger": "boot"})
    await _settle()
    assert ob.awaiting                                   # repo question spoken
    # a worker blocks on permission mid-chain; its card publishes the readback
    a = router.open_approval("alethic", "Bash: npm test", now=time.time(),
                             path="/x/alethic")
    bus.publish("approval.request", {
        "nonce": a.nonce, "worker": "w1", "project": "alethic",
        "path": "/x/alethic", "tool": "Bash", "args": "npm test",
        "voice_ok": True,
        "question": "alethic wants Bash — npm test. Approve or deny, sir?"})
    await _settle()
    assert any("npm test" in s for s in spk.spoke)       # readback spoken
    assert not ob.awaiting                               # repo question yielded
    # the owner's yes is for the WORKER. Before the fix, onboarding's terminal
    # ownership confirmed the repo with it and the worker stayed blocked.
    bus.publish("command.received", {"text": "yes"})
    await _settle()
    assert not any(p.confirmed for p in reg.projects)    # repo NOT confirmed by it
    # the floor just switched, so the first bare yes costs one clarifying beat
    assert any("worker's request" in s for s in spk.spoke), spk.spoke
    assert len(router.pending_approvals()) == 1          # nothing resolved by it either
    bus.publish("command.received", {"text": "yes"})
    await _settle()
    assert router.pending_approvals() == []              # the worker got its yes
    assert not any(p.confirmed for p in reg.projects)
    # the chain resumed: the displaced question was read aloud a SECOND time
    assert sum(1 for s in spk.spoke if "correct repo" in s) >= 2, spk.spoke
    assert ob.awaiting
    bus.publish("command.received", {"text": "yes"})
    await _settle()
    assert sum(1 for p in reg.projects if p.confirmed) == 1
    assert butler.asked == []                            # none of this reached the model
    task.cancel()


async def test_a_sixty_nine_repo_discovery_does_not_chain_unbounded(tmp_path):
    """The other half of the live report: proposing 59 remaining repos one at
    a time is bad design even with an exit. One run is capped, and the close
    is honest — how many remain and how to continue."""
    from server.onboarding import PROPOSALS_PER_RUN
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry(*[f"repo{i}" for i in range(PROPOSALS_PER_RUN + 4)])
    ob = onboarding_for(bus, reg, tmp_path)
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("confirm.next", {"trigger": "boot"})
    await _settle()
    for _ in range(PROPOSALS_PER_RUN):
        assert ob.awaiting
        bus.publish("command.received", {"text": "yes"})
        await _settle()
    assert not ob.awaiting                               # the chain closed itself
    closing = [s for s in spk.spoke if "4 more" in s]
    assert closing and "find my projects" in closing[0].lower(), spk.spoke
    assert sum(1 for p in reg.projects if p.confirmed) == PROPOSALS_PER_RUN
    assert len(ob._candidates()) == 4                    # nothing discarded
    task.cancel()


async def test_stop_asking_ends_the_chain_and_find_my_projects_resumes_it(tmp_path, monkeypatch):
    import server.app_brain as ab
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(ab, "default_home", lambda: home)
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry("quant agent", "soccer", "alethic")
    ob = onboarding_for(bus, reg, tmp_path)
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("confirm.next", {"trigger": "boot"})
    await _settle()
    bus.publish("command.received", {"text": "yes"})     # confirm the first
    await _settle()
    assert ob.awaiting                                   # the second is on the floor
    bus.publish("command.received", {"text": "stop asking"})
    await _settle()
    assert not ob.awaiting                               # the way out works
    line = next(s for s in spk.spoke if "find my projects" in s.lower())
    assert "2" in line                                   # how many remain, honestly
    assert len(ob._candidates()) == 2                    # both still pending
    assert sum(1 for p in reg.projects if p.confirmed) == 1
    assert butler.asked == []                            # never model-parsed
    # the documented way back in
    bus.publish("command.received", {"text": "find my projects"})
    await _settle(0.3)                                   # the rescan runs off-loop
    assert ob.awaiting                                   # the chain re-opened
    bus.publish("command.received", {"text": "yes"})
    await _settle()
    assert sum(1 for p in reg.projects if p.confirmed) == 2
    task.cancel()


async def test_a_displaced_question_is_never_spoken_stale(tmp_path):
    """The pause lands while its question's confirm.request is still queued:
    the stale readback must not be voiced. A spoken question nothing owns
    invites a yes that lands somewhere else — the misdirection this whole
    module exists to prevent."""
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry("quant agent", "soccer")
    ob = onboarding_for(bus, reg, tmp_path)
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    # ask_next publishes confirm.request to the TAIL, so it lands BEHIND this
    # utterance — the pause is processed first and the readback goes stale
    bus.publish("confirm.next", {"trigger": "boot"})
    bus.publish("command.received", {"text": "stop asking"})
    await _settle()
    assert not any("correct repo" in s for s in spk.spoke), spk.spoke
    assert not ob.awaiting
    assert len(ob._candidates()) == 2                    # nothing lost
    task.cancel()


async def test_stop_asking_is_not_swallowed_by_a_pending_approval(tmp_path):
    """'stop asking' opens with a deny word, and a project-less utterance is
    offered to the approval resolver first — with an unread approval pending
    that used to answer 'I haven't read that request to you yet' and the
    pause never fired. A parsed pause command is positive evidence of intent,
    exactly like a resolved project name."""
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry("quant agent", "soccer")
    ob = onboarding_for(bus, reg, tmp_path)
    router = Router()
    router.open_approval("alethic", "Bash: npm test", now=time.time(),
                         path="/x/alethic")              # pending, UNSPOKEN
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=router, registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("command.received", {"text": "stop asking"})
    await _settle()
    assert not any("haven't read" in s for s in spk.spoke), spk.spoke
    assert any("find my projects" in s.lower() for s in spk.spoke), spk.spoke
    assert len(router.pending_approvals()) == 1          # the approval is untouched
    task.cancel()


async def test_find_my_projects_defers_honestly_while_an_approval_is_pending(tmp_path, monkeypatch):
    import server.app_brain as ab
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(ab, "default_home", lambda: home)
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    reg = pending_registry("quant agent")
    ob = onboarding_for(bus, reg, tmp_path)
    router = Router()
    open_spoken(router, "alethic", "Bash: npm test", now=time.time(),
                path="/x/alethic")
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=router, registry=reg, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("command.received", {"text": "find my projects"})
    await _settle(0.3)
    assert not ob.awaiting                               # deferred, not asked
    assert any("worker's request comes first" in s for s in spk.spoke), spk.spoke
    # the deferral is honest AND recoverable: resolving the approval resumes
    bus.publish("command.received", {"text": "yes"})
    await _settle()
    assert router.pending_approvals() == []
    assert ob.awaiting                                   # the chain came back
    task.cancel()


async def test_the_brain_survives_a_pause_explosion(tmp_path):
    class BoomPause:
        awaiting = False
        suspended = False
        async def handle_reply(self, text): return "ignored"
        def pause(self): raise RuntimeError("pause exploded")
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=Router(), registry=Registry(), onboarding=BoomPause()))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "butler.error"))
    bus.publish("command.received", {"text": "stop asking"})
    ev = await fut
    assert "command handling failed" in ev["data"]["reason"]
    assert not task.done()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_the_brain_survives_a_yield_explosion(tmp_path):
    class BoomYield:
        awaiting = False
        suspended = False
        async def handle_reply(self, text): return "ignored"
        def yield_floor(self): raise RuntimeError("yield exploded")
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router = Router()
    a = router.open_approval("alethic", "Bash: npm test", now=time.time(),
                             path="/x/alethic")
    task = asyncio.create_task(run_butler_brain(
        bus, butler, spk, FakeTurnLog(),
        router=router, registry=Registry(), onboarding=BoomYield()))
    await asyncio.sleep(0)
    fut = asyncio.ensure_future(_drain(bus, "butler.error"))
    bus.publish("approval.request", {
        "nonce": a.nonce, "project": "alethic", "voice_ok": True,
        "question": "alethic wants Bash — npm test. Approve or deny, sir?"})
    ev = await fut
    assert "yield failed" in ev["data"]["reason"]
    await _settle()
    # the readback still went out — a yield fault costs the yield, not the turn
    assert any("npm test" in s for s in spk.spoke)
    assert not task.done()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
