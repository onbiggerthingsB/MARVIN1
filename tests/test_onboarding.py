import asyncio
from pathlib import Path

import pytest

from server.bus import EventBus
from server.discovery import Candidate
from server.onboarding import Onboarding
from server.registry import Registry


def seeded(tmp_path):
    r = Registry()
    r.merge_candidates([Candidate(path="/p/quant agent", name="quant agent", sources=["a", "b"]),
                        Candidate(path="/p/soccer", name="soccer", sources=["a"])])
    return Onboarding(EventBus(), r, tmp_path / "projects.json"), r


async def asked(ob):
    """ask_next AND mark the question spoken — the state handle_reply now
    requires, exactly as a repo question that was actually read aloud. A bare
    ask_next() models a question published but not yet voiced, which a reply may
    not resolve (see the spoken-correlation tests below)."""
    ok = await ob.ask_next()
    ob.mark_spoken()
    return ok


def test_next_prompt_names_the_project_and_path(tmp_path):
    ob, _ = seeded(tmp_path)
    p = ob.next_prompt()
    assert p["name"] in {"quant agent", "soccer"}
    assert p["path"].startswith("/p/")
    assert "correct" in p["question"].lower()      # "...that's the correct repo, right?"


async def test_ask_next_publishes_a_request(tmp_path):
    ob, _ = seeded(tmp_path)
    cid, q = ob.bus.subscribe()
    assert await ob.ask_next() is True
    ev = await asyncio.wait_for(q.get(), 1)
    assert ev["type"] == "confirm.request" and ev["data"]["name"]


async def test_yes_confirms_and_persists(tmp_path):
    f = tmp_path / "projects.json"
    ob, r = seeded(tmp_path)
    await asked(ob)
    name = ob._asking.name                         # the project we were just asked about
    assert await ob.handle_reply("yes, that's right") == "confirmed"
    assert any(p.name == name and p.confirmed for p in r.projects)
    assert Registry.load(f).projects               # written to disk


async def test_no_rejects_and_moves_on(tmp_path):
    ob, r = seeded(tmp_path)
    await asked(ob)
    first = ob._asking.name
    assert await ob.handle_reply("no, that's not it") == "rejected"
    # a rejected project is not confirmed and is not asked about again
    assert not any(p.name == first and p.confirmed for p in r.projects)
    assert ob.next_prompt() is None or ob.next_prompt()["name"] != first


@pytest.mark.parametrize("said", [
    "No.", "no!", "no thanks", "no way",
    "no, that's not right", "no, that isn't it", "no, not this one"])
async def test_spoken_rejections_never_confirm(tmp_path, said):
    ob, r = seeded(tmp_path)
    await asked(ob)
    outcome = await ob.handle_reply(said)
    assert outcome == "rejected"
    assert not any(p.confirmed for p in r.projects)      # nothing confirmed, ever


@pytest.mark.parametrize("said", [
    "Yeah, no, that's not right", "yeah no", "yes, but not that one", "sure, no"])
async def test_affirmative_prefixed_negations_never_confirm(tmp_path, said):
    ob, r = seeded(tmp_path)
    await asked(ob)
    outcome = await ob.handle_reply(said)
    assert outcome != "confirmed"
    assert not any(p.confirmed for p in r.projects)


@pytest.mark.parametrize("said", [
    "okay, where did I leave the Tibet study?",
    "sure, stop soccer",
    "go ahead and pull up composed",
    "yes, run the tests in alethic"])
async def test_an_addressed_request_is_not_a_confirmation(tmp_path, said):
    # _YES is prefix-anchored, so each of these matches it — but each carries
    # a real request. Naming something is positive evidence the speaker is not
    # answering the pending question (router.bare_yes_no's rule).
    ob, r = seeded(tmp_path)
    await asked(ob)
    outcome = await ob.handle_reply(said)
    assert outcome == "ignored"                       # falls through to the router/butler
    assert not any(p.confirmed for p in r.projects)   # nothing confirmed by a real request
    assert ob.awaiting                                # the question is still pending


@pytest.mark.parametrize("said", ["yes", "yeah", "yep", "yes, that's right", "confirm"])
async def test_a_plain_yes_still_confirms(tmp_path, said):
    ob, r = seeded(tmp_path)
    await asked(ob)
    assert await ob.handle_reply(said) == "confirmed"
    assert any(p.confirmed for p in r.projects)


async def test_mutation_lands_on_the_exact_path_asked_about(tmp_path):
    r = Registry()
    r.merge_candidates([Candidate(path="/one/marvin", name="marvin", sources=["a"]),
                        Candidate(path="/two/marvin", name="marvin", sources=["a"])])
    ob = Onboarding(EventBus(), r, tmp_path / "projects.json")
    # Reject the first marvin, then confirm the second: a name-keyed mutation
    # would land on the first (rejected!) one, since both share a basename.
    await asked(ob)
    assert await ob.handle_reply("no") == "rejected"
    await asked(ob)
    asked_path = ob._asking.path
    assert await ob.handle_reply("yes") == "confirmed"
    confirmed = [p for p in r.projects if p.confirmed]
    assert [p.path for p in confirmed] == [asked_path]      # exactly one, the right one


async def test_a_correction_records_a_mishearing(tmp_path):
    ob, r = seeded(tmp_path)
    await asked(ob)
    name = ob._asking.name
    assert await ob.handle_reply("no, I said the trading system") == "renamed"
    p = next(p for p in r.projects if p.name == name)
    assert "the trading system" in p.aliases


async def test_reply_with_nothing_pending_is_ignored(tmp_path):
    ob, _ = seeded(tmp_path)
    assert await ob.handle_reply("yes") == "ignored"


async def test_refresh_merges_and_reports_new(tmp_path, monkeypatch):
    ob, r = seeded(tmp_path)

    async def fake_discover(home, extra_roots=None):
        return [Candidate(path="/p/new", name="new", sources=["a"])]

    monkeypatch.setattr("server.onboarding.discover", fake_discover)
    assert await ob.refresh(Path("/fake/home")) == 1
    assert any(p.name == "new" for p in r.projects)


@pytest.mark.parametrize("said", ["okay", "sure", "go ahead", "do it"])
async def test_router_affirmations_confirm_a_repo(tmp_path, said):
    # Precondition 2: the router's _AFFIRM vocabulary and onboarding's _YES
    # must agree, or an affirmation that only the router understands falls
    # past the pending confirmation onto a pending tool approval.
    ob, r = seeded(tmp_path)
    await asked(ob)
    assert await ob.handle_reply(said) == "confirmed"
    assert any(p.confirmed for p in r.projects)


async def test_refresh_survives_a_null_claude_json(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude.json").write_text("null", encoding="utf-8")
    ob = Onboarding(EventBus(), Registry(), tmp_path / "projects.json")
    assert await ob.refresh(home) == 0          # no raise, no candidates


# ---------- CRITICAL 3: consent may not precede the spoken question ----------
async def test_handle_reply_refuses_until_the_question_is_spoken(tmp_path):
    """ask_next() sets awaiting synchronously but the question rides the TAIL of
    the bus queue, so a reply can arrive before it is voiced. handle_reply must
    refuse to resolve until mark_spoken — the same barrier Approval.spoken gives
    the fleet. A wrong 'confirmed' here hands over a repo (maybe the finance
    one) whose question the owner never heard."""
    ob, r = seeded(tmp_path)
    assert await ob.ask_next() is True            # asked, but NOT yet spoken
    assert ob.awaiting is True
    assert await ob.handle_reply("yes") == "ignored"     # refused: never voiced
    assert not any(p.confirmed for p in r.projects)
    assert ob.awaiting is True                    # still pending, not consumed
    # once the question is actually read aloud, the same yes confirms
    assert ob.mark_spoken() is True
    assert await ob.handle_reply("yes") == "confirmed"
    assert any(p.confirmed for p in r.projects)


async def test_mark_spoken_with_nothing_pending_is_a_no_op(tmp_path):
    ob, _ = seeded(tmp_path)
    assert ob.mark_spoken() is False


async def test_a_reply_before_the_question_never_upgrades_a_finance_repo(tmp_path):
    """The concrete report: a candidate whose name contains 'quant' is
    auto-upgraded to kind='finance' on confirm. A yes that beats the question
    must not trigger that upgrade — the whole point of the barrier."""
    r = Registry()
    r.merge_candidates([Candidate(path="/p/quant agent", name="quant agent",
                                  sources=["a", "b"])])
    ob = Onboarding(EventBus(), r, tmp_path / "projects.json")
    await ob.ask_next()                           # published, not spoken
    assert await ob.handle_reply("yeah go ahead") == "ignored"
    assert not any(p.confirmed for p in r.projects)
    assert all(getattr(p, "kind", None) != "finance" for p in r.projects)


async def test_the_confirm_next_barrier_race_confirms_nothing_before_speaking():
    """CRITICAL 3, through the REAL brain. The observed boot ordering is
    [fleet.spoken, confirm.next, 'yeah go ahead'] — the impatient yes is already
    queued when confirm.next runs, so ask_next appends confirm.request BEHIND
    it. The repo must not be confirmed (and not upgraded to finance) before its
    question is spoken."""
    from server.app_brain import run_butler_brain
    from server.router import Router
    from tests.test_fleet_wiring import FakeButler, FakeSpeaker, FakeTurnLog

    class SeqSpeaker(FakeSpeaker):
        async def speak(self, t):
            self.spoke.append(t)

    bus = EventBus()
    r = Registry()
    r.merge_candidates([Candidate(path="/p/quant agent", name="quant agent",
                                  sources=["a", "b"])])
    ob = Onboarding(bus, r, "/tmp/marvin-test-projects.json")
    spk = SeqSpeaker()
    task = asyncio.create_task(run_butler_brain(
        bus, FakeButler(), spk, FakeTurnLog(),
        router=Router(), registry=r, onboarding=ob))
    await asyncio.sleep(0)
    bus.publish("fleet.spoken", {"text": "Sir, a worker was interrupted."})
    bus.publish("confirm.next", {"trigger": "boot"})
    bus.publish("command.received", {"text": "yeah go ahead"})
    await asyncio.sleep(0.1)
    # nothing confirmed, nothing upgraded, BEFORE the question was read
    assert not any(p.confirmed for p in r.projects)
    assert all(getattr(p, "kind", None) != "finance" for p in r.projects)
    # and the question WAS eventually spoken (the barrier voiced it)
    assert any("correct repo" in s for s in spk.spoke)
    # confirmation acknowledgment ("Noted") never preceded the question
    q_idx = next((i for i, s in enumerate(spk.spoke) if "correct repo" in s), None)
    noted_idx = next((i for i, s in enumerate(spk.spoke) if "Noted" in s), None)
    assert q_idx is not None and (noted_idx is None or noted_idx > q_idx)
    task.cancel()
