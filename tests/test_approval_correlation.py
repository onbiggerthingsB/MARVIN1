"""A yes may only resolve the request Marlowe actually READ ALOUD.

The spoken approval line is the whole safety model: the worktree is not a
sandbox, so what contains a worker is the human hearing what it wants and
saying yes. Before this, the readback and the resolver were not correlated at
all — the brain spoke `approval.request.question` and recorded nothing, while
`resolve_approval` matched against whatever the router happened to be holding.
Three reproduced consequences, all pinned below:

(a) the queue holds [utterance("yeah, sure"), approval.request] while the brain
    is busy — it approved `rm -rf ~/Documents`, said "Approved, sir.", and only
    then asked;
(b) a click resolves N1 while the worker opens N2 — the sentence described
    `npm test` and the voice yes approved the `rm -rf`;
(c) the bus evicts the brain's 256-slot queue (a chatty worker publishes ~2
    events per message) and it resubscribes with no replay — the request is
    never spoken at all, and the yes delivered it anyway.
"""
import asyncio
import time

from server.bus import EventBus
from server.router import Router
from tests.test_fleet_wiring import (FakeButler, FakeFleet, FakeSpeaker, brain,
                                     confirmed_registry)

NOW = 1_000_000.0
DESTRUCTIVE = "Bash: rm -rf /Users/likerun/Documents"


class GatedSpeaker(FakeSpeaker):
    """Blocks inside the FIRST speak until `gate` is set — the long await the
    brain loop really does hold (SPEAK_TIMEOUT_S is 60s, ASK_TIMEOUT_S 120)."""

    def __init__(self):
        super().__init__()
        self.gate = asyncio.Event()
        self.started = asyncio.Event()

    async def speak(self, t):
        self.spoke.append(t)
        if not self.gate.is_set():
            self.started.set()
            await self.gate.wait()


def card(approval, question):
    return {"nonce": approval.nonce, "worker": "w1", "project": approval.project,
            "path": approval.path, "tool": "Bash", "args": "…",
            "question": question}


# ---------------- the router: only a spoken approval is resolvable ----------
def test_a_bare_yes_never_resolves_an_approval_nobody_read_aloud():
    router = Router()
    a = router.open_approval("soccer", DESTRUCTIVE, now=NOW, path="/p/soccer")
    assert router.resolve_approval("yes", NOW + 5) == ("unspoken", None)
    assert [x.nonce for x in router.pending_approvals()] == [a.nonce]
    assert router.mark_spoken(a.nonce) is True
    state, appr = router.resolve_approval("yes", NOW + 6)
    assert state == "approved" and appr.nonce == a.nonce


def test_a_bare_no_also_needs_a_readback():
    """A denial that resolves an unread approval is a smaller accident than an
    approval, but it is the same lie: it answers a question never asked."""
    router = Router()
    a = router.open_approval("soccer", "Bash: npm test", now=NOW, path="/p/soccer")
    assert router.resolve_approval("no", NOW + 5) == ("unspoken", None)
    router.mark_spoken(a.nonce)
    assert router.resolve_approval("no", NOW + 6)[0] == "denied"


def test_an_addressed_yes_never_resolves_an_unspoken_approval():
    """Both branches of resolve_approval, not just the bare one: naming the
    project does not mean the project's request was ever read out."""
    router = Router()
    a = router.open_approval("soccer", "npm test", now=NOW, path="/p/soccer")
    assert (router.resolve_approval("approve soccer npm test", NOW + 5)
            == ("unspoken", None))
    router.mark_spoken(a.nonce)
    assert router.resolve_approval("approve soccer npm test",
                                   NOW + 6)[0] == "approved"


def test_a_bare_yes_binds_to_the_one_that_was_spoken_not_the_newest():
    """Case (b): the worker opened a second approval while the first was being
    read. `self._approvals[0]` and "exactly one is pending" both say nothing
    about which sentence Keke actually heard."""
    router = Router()
    heard = router.open_approval("soccer", "Bash: npm test", now=NOW,
                                 path="/p/soccer")
    router.mark_spoken(heard.nonce)
    unheard = router.open_approval("soccer", DESTRUCTIVE, now=NOW + 1,
                                   path="/p/soccer")
    state, appr = router.resolve_approval("yes", NOW + 2)
    assert state == "approved" and appr.nonce == heard.nonce
    assert [x.nonce for x in router.pending_approvals()] == [unheard.nonce]


def test_two_spoken_approvals_are_still_ambiguous_to_a_bare_yes():
    router = Router()
    for project in ("soccer", "alethic"):
        a = router.open_approval(project, "npm test", now=NOW, path=f"/p/{project}")
        router.mark_spoken(a.nonce)
    assert router.resolve_approval("yes", NOW + 1) == ("ambiguous", None)


def test_marking_a_nonce_the_router_does_not_hold_is_a_no_op():
    assert Router().mark_spoken("no-such-nonce") is False


def test_an_expired_approval_is_still_expired_not_unspoken():
    router = Router()
    router.open_approval("soccer", "npm test", now=NOW)
    assert router.resolve_approval("yes", NOW + 601) == ("expired", None)


# ---------------- the brain: spoken is set only AFTER the readback ----------
async def test_a_yes_queued_ahead_of_the_readback_approves_nothing():
    """Case (a), through the real brain. Both events are already in the queue
    when the loop reaches them, utterance first — the observed live ordering."""
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router, fleet = Router(), FakeFleet()
    a = router.open_approval("soccer", DESTRUCTIVE, now=time.time(),
                             path="/p/soccer")
    task = await brain(bus, butler, spk, router=router,
                       registry=confirmed_registry("soccer"), fleet=fleet)
    bus.publish("command.received", {"text": "yeah, sure"})
    bus.publish("approval.request", card(
        a, "Careful, sir — this one can destroy things. soccer wants Bash — "
           "rm -rf /Users/likerun/Documents. Approve or deny, sir?"))
    await asyncio.sleep(0.05)
    assert all(c[0] != "deliver" for c in fleet.calls)   # nothing was approved
    assert "Approved, sir." not in spk.spoke
    assert len(router.pending_approvals()) == 1
    told = next(i for i, s in enumerate(spk.spoke) if "haven't read" in s)
    read = next(i for i, s in enumerate(spk.spoke) if "Approve or deny" in s)
    assert told < read                                   # said so, THEN asked
    assert butler.asked == []                            # never the model's turn
    # and now a yes lands on something Keke has actually heard
    bus.publish("command.received", {"text": "yes"})
    await asyncio.sleep(0.05)
    assert ("deliver", a.nonce, True) in fleet.calls
    assert "Approved, sir." in spk.spoke
    task.cancel()


async def test_the_yes_binds_to_the_request_that_was_read_not_the_next_one():
    """Case (b), through the real brain: the console click resolves the one
    that was read out, the worker opens another, and the voice yes arrives."""
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router, fleet = Router(), FakeFleet()
    n1 = router.open_approval("soccer", "Bash: npm test", now=time.time(),
                              path="/p/soccer")
    task = await brain(bus, butler, spk, router=router,
                       registry=confirmed_registry("soccer"), fleet=fleet)
    bus.publish("approval.request", card(
        n1, "soccer wants Bash — npm test. Approve or deny, sir?"))
    await asyncio.sleep(0.05)
    assert any("npm test" in s for s in spk.spoke)
    router.take_nonce(n1.nonce, time.time())             # the Approve click
    n2 = router.open_approval("soccer", DESTRUCTIVE, now=time.time(),
                              path="/p/soccer")
    bus.publish("command.received", {"text": "yes"})
    await asyncio.sleep(0.05)
    assert all(c[0] != "deliver" for c in fleet.calls)
    assert [x.nonce for x in router.pending_approvals()] == [n2.nonce]
    assert "Approved, sir." not in spk.spoke
    task.cancel()


async def test_an_evicted_readback_leaves_the_yes_with_nothing_to_approve():
    """Case (c). bus.py drops a subscriber whose 256-slot queue fills and the
    brain resubscribes with NO replay, so a chatty worker (~2 events per
    message) can swallow the readback entirely while the loop is blocked in a
    speak. The request is never spoken — so no yes may resolve it."""
    bus, butler, spk = EventBus(), FakeButler(), GatedSpeaker()
    router, fleet = Router(), FakeFleet()
    task = await brain(bus, butler, spk, router=router,
                       registry=confirmed_registry("soccer"), fleet=fleet)
    bus.publish("fleet.spoken", {"text": "something long, sir."})
    await asyncio.wait_for(spk.started.wait(), 2)        # blocked mid-sentence
    for i in range(300):
        bus.publish("fleet.message", {"worker": "w1", "project": "soccer",
                                      "who": "worker", "text": f"line {i}"})
    a = router.open_approval("soccer", DESTRUCTIVE, now=time.time(),
                             path="/p/soccer")
    bus.publish("approval.request", card(
        a, "soccer wants Bash — rm -rf /Users/likerun/Documents. "
           "Approve or deny, sir?"))
    spk.gate.set()
    await asyncio.sleep(0.1)
    assert not any("Approve or deny" in s for s in spk.spoke)   # never read
    bus.publish("command.received", {"text": "yes"})
    await asyncio.sleep(0.05)
    assert all(c[0] != "deliver" for c in fleet.calls)
    assert len(router.pending_approvals()) == 1
    assert any("haven't read" in s for s in spk.spoke)
    task.cancel()


async def test_a_readback_that_was_actually_spoken_is_resolvable_by_voice():
    """The positive control: without it every test above passes on a Marlowe
    that simply never resolves anything by voice again."""
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router, fleet = Router(), FakeFleet()
    a = router.open_approval("soccer", "Bash: npm test", now=time.time(),
                             path="/p/soccer")
    task = await brain(bus, butler, spk, router=router,
                       registry=confirmed_registry("soccer"), fleet=fleet)
    bus.publish("approval.request", card(
        a, "soccer wants Bash — npm test. Approve or deny, sir?"))
    await asyncio.sleep(0.05)
    assert router.pending_approvals()[0].spoken is True
    bus.publish("command.received", {"text": "yes, go ahead"})
    await asyncio.sleep(0.05)
    assert ("deliver", a.nonce, True) in fleet.calls
    assert "Approved, sir." in spk.spoke
    task.cancel()


# ---------------- CRITICAL 3: a dead request is never read aloud ------------
async def test_a_stale_approval_request_is_never_read_aloud():
    """After "stop soccer", Worker.shutdown rejects the future and TAKES the
    nonce — but the queued approval.request was read out anyway: "Stopped
    soccer, sir." followed by "soccer wants Bash — rm -rf build. Approve or
    deny, sir?" about a worker that no longer exists. A "no" in reply found no
    pending approval and was answered by the butler as conversation."""
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router, fleet = Router(), FakeFleet()
    a = router.open_approval("soccer", "Bash: rm -rf build", now=time.time(),
                             path="/p/soccer")
    router.take_nonce(a.nonce, time.time())              # the stop consumed it
    task = await brain(bus, butler, spk, router=router,
                       registry=confirmed_registry("soccer"), fleet=fleet)
    bus.publish("approval.request", card(
        a, "soccer wants Bash — rm -rf build. Approve or deny, sir?"))
    await asyncio.sleep(0.05)
    assert spk.spoke == []                               # silence is the fix
    assert not task.done()
    task.cancel()


# ---------------- CRITICAL 2: a click-only command is never voice-approvable --
async def test_a_click_only_readback_is_never_marked_spoken_and_refuses_a_yes():
    """The fleet flags a too-long command voice_ok=False and speaks a 'too long,
    it's on the card' line. The brain must NOT mark it spoken (it never read the
    command) and a later voice yes must be refused, not deliver the tool."""
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router, fleet = Router(), FakeFleet()
    a = router.open_approval("soccer", "Bash: <long>", now=time.time(),
                             path="/p/soccer", voice_ok=False)
    task = await brain(bus, butler, spk, router=router,
                       registry=confirmed_registry("soccer"), fleet=fleet)
    bus.publish("approval.request", {
        "nonce": a.nonce, "worker": "w1", "project": "soccer", "path": "/p/soccer",
        "tool": "Bash", "args": "…", "voice_ok": False,
        "question": ("soccer wants Bash — that command is too long to read "
                     "aloud, sir; it's on the card. Approve or deny it there.")})
    await asyncio.sleep(0.05)
    assert router.pending_approvals()[0].spoken is False       # never marked read
    assert any("too long" in s.lower() for s in spk.spoke)     # said so
    bus.publish("command.received", {"text": "yes"})
    await asyncio.sleep(0.05)
    assert all(c[0] != "deliver" for c in fleet.calls)         # nothing approved
    assert len(router.pending_approvals()) == 1
    assert any("on the card" in s.lower() for s in spk.spoke)  # refusal points to card
    task.cancel()


async def test_an_expired_approval_request_is_never_read_aloud():
    """Same rule for the TTL path: _on_tool_request sweeps the nonce and
    publishes `expired`, and the queued card must not be asked about."""
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    router, fleet = Router(), FakeFleet()
    a = router.open_approval("soccer", "Bash: npm test", now=time.time() - 700,
                             path="/p/soccer")
    task = await brain(bus, butler, spk, router=router,
                       registry=confirmed_registry("soccer"), fleet=fleet)
    router.take_nonce(a.nonce, time.time())              # the TTL swept it
    bus.publish("approval.request", card(
        a, "soccer wants Bash — npm test. Approve or deny, sir?"))
    await asyncio.sleep(0.05)
    assert spk.spoke == []
    task.cancel()
