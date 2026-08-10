import asyncio
import contextlib
import time

from server.bus import EventBus
from server.discovery import Candidate
from server.registry import Registry
from server.router import Router
from server.app_brain import run_butler_brain
# An approval Marlowe has already read aloud. A raw open_approval() models one
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
    monkeypatch.setenv("MARLOWE_VAULT", str(tmp_path))
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


async def test_a_bare_yes_answers_the_spoken_repo_question_not_the_tool_approval(tmp_path):
    # Three questions can be pending at once. A bare yes with a repo confirm
    # pending belongs TERMINALLY to that confirm — it must never reach the
    # approval resolver and run someone's `rm -rf`.
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
    assert any("correct repo" in s for s in spk.spoke)   # it WAS read aloud
    bus.publish("command.received", {"text": "yes"})
    await _settle()
    assert [p.name for p in reg.projects if p.confirmed] == ["soccer"]
    assert len(router.pending_approvals()) == 1          # the rm -rf is untouched
    assert butler.asked == []
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
