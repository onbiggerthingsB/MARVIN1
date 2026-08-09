import asyncio
import json
from pathlib import Path

from server.bus import EventBus
from server.discovery import Candidate
from server.finance import find_finance_project
from server.finance_gate import SourceGate
from server.registry import Registry


def gated(tmp_path):
    root = tmp_path / "quant agent"
    root.mkdir()
    (root / "picks.json").write_text(
        json.dumps([{"symbol": "TSLA", "shares": 3}]), encoding="utf-8")
    r = Registry()
    r.merge_candidates([Candidate(path=str(root), name="quant agent", sources=["a"])])
    r.confirm("quant agent", kind="finance")
    gate = SourceGate(EventBus(), r, tmp_path / "projects.json")
    return gate, r, root


async def test_ask_publishes_the_question_with_the_filename(tmp_path):
    gate, r, root = gated(tmp_path)
    cid, q = gate.bus.subscribe()
    question = await gate.ask(find_finance_project(r))
    assert "picks.json" in question and "correct" in question.lower()
    assert gate.awaiting is True
    ev = await asyncio.wait_for(q.get(), 1)
    assert ev["type"] == "confirm.request" and "picks.json" in ev["data"]["question"]


async def test_yes_pins_and_persists_the_source(tmp_path):
    gate, r, root = gated(tmp_path)
    await gate.ask(find_finance_project(r))
    assert await gate.handle_reply("yes, that's right") == "confirmed"
    assert gate.awaiting is False
    assert r.projects[0].data_source == str(root / "picks.json")
    reloaded = Registry.load(tmp_path / "projects.json")
    assert reloaded.projects[0].data_source == str(root / "picks.json")


async def test_no_rejects_without_pinning(tmp_path):
    gate, r, root = gated(tmp_path)
    await gate.ask(find_finance_project(r))
    assert await gate.handle_reply("no, that's not it") == "rejected"
    assert r.projects[0].data_source is None


async def test_yeah_no_is_a_rejection(tmp_path):
    # The Part 1 lesson, applied here: an affirmative opener with a negation
    # anywhere in it must never confirm.
    gate, r, root = gated(tmp_path)
    await gate.ask(find_finance_project(r))
    assert await gate.handle_reply("yeah, no, that's not right") == "rejected"
    assert r.projects[0].data_source is None


async def test_reply_with_nothing_pending_is_ignored(tmp_path):
    gate, r, root = gated(tmp_path)
    assert await gate.handle_reply("yes") == "ignored"
