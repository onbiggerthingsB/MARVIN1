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
    await ob.ask_next()
    asked = ob._asking.name                        # the project we were just asked about
    assert await ob.handle_reply("yes, that's right") == "confirmed"
    assert any(p.name == asked and p.confirmed for p in r.projects)
    assert Registry.load(f).projects               # written to disk


async def test_no_rejects_and_moves_on(tmp_path):
    ob, r = seeded(tmp_path)
    await ob.ask_next()
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
    await ob.ask_next()
    outcome = await ob.handle_reply(said)
    assert outcome == "rejected"
    assert not any(p.confirmed for p in r.projects)      # nothing confirmed, ever


@pytest.mark.parametrize("said", [
    "Yeah, no, that's not right", "yeah no", "yes, but not that one", "sure, no"])
async def test_affirmative_prefixed_negations_never_confirm(tmp_path, said):
    ob, r = seeded(tmp_path)
    await ob.ask_next()
    outcome = await ob.handle_reply(said)
    assert outcome != "confirmed"
    assert not any(p.confirmed for p in r.projects)


async def test_mutation_lands_on_the_exact_path_asked_about(tmp_path):
    r = Registry()
    r.merge_candidates([Candidate(path="/one/jarvis", name="jarvis", sources=["a"]),
                        Candidate(path="/two/jarvis", name="jarvis", sources=["a"])])
    ob = Onboarding(EventBus(), r, tmp_path / "projects.json")
    # Reject the first jarvis, then confirm the second: a name-keyed mutation
    # would land on the first (rejected!) one, since both share a basename.
    await ob.ask_next()
    assert await ob.handle_reply("no") == "rejected"
    await ob.ask_next()
    asked_path = ob._asking.path
    assert await ob.handle_reply("yes") == "confirmed"
    confirmed = [p for p in r.projects if p.confirmed]
    assert [p.path for p in confirmed] == [asked_path]      # exactly one, the right one


async def test_a_correction_records_a_mishearing(tmp_path):
    ob, r = seeded(tmp_path)
    await ob.ask_next()
    asked = ob._asking.name
    assert await ob.handle_reply("no, I said the trading system") == "renamed"
    p = next(p for p in r.projects if p.name == asked)
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
