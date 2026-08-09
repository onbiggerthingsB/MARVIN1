"""The worktree verbs, wired end to end through the brain loop.

Every piece below is unit-tested elsewhere; these pin the WIRING, and above
all the FOURTH-QUESTION property: the worktree gate is reached by a destructive
VERB, never by an affirmation, so it can neither steal a pending yes nor have
its own instruction stolen by one of the three yes-gates that already exist.
"""
from __future__ import annotations

import asyncio
import contextlib
import time

import pytest

from server.app_brain import run_butler_brain
from server.bus import EventBus
from server.discovery import Candidate
from server.registry import Registry
from server.router import Router
from tests.test_fleet_wiring import (FakeButler, FakeSpeaker, FakeTurnLog,
                                     open_spoken)


def confirmed_registry(*names, kind="code"):
    r = Registry()
    r.merge_candidates([Candidate(path=f"/p/{n}", name=n, sources=["t"])
                        for n in names])
    for n in names:
        r.confirm(n, kind=kind)
    return r


class FakeCleanup:
    def __init__(self):
        self.calls = []

    async def report(self):
        self.calls.append(("report",))
        return "Sir, I found 3 worktrees."

    async def remove_empty(self):
        self.calls.append(("remove_empty",))
        return "Removed 2 empty worktrees, sir."

    async def remove_named(self, name):
        self.calls.append(("remove_named", name))
        return f"Removed the worktree for {name}, sir."


class PendingConfirm:
    """An onboarding mid-question that does not understand this reply."""
    awaiting = True

    async def handle_reply(self, text):
        return "ignored"


class PendingSource:
    """The §16 finance gate, mid-question, same posture."""
    awaiting = True

    async def handle_reply(self, text):
        return "ignored"


async def run(bus, butler, spk, **kw):
    task = asyncio.create_task(run_butler_brain(bus, butler, spk,
                                               FakeTurnLog(), **kw))
    await asyncio.sleep(0)
    return task


async def stop(task):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ------------------------------------------------------------- dispatch -----
async def test_the_survey_verb_reports_and_never_reaches_the_model():
    bus, butler, spk, cleanup = EventBus(), FakeButler(), FakeSpeaker(), FakeCleanup()
    task = await run(bus, butler, spk, router=Router(),
                     registry=confirmed_registry("soccer"), cleanup=cleanup)
    bus.publish("command.received", {"text": "clean up the worktrees"})
    await asyncio.sleep(0.05)
    assert cleanup.calls == [("report",)]
    assert "Sir, I found 3 worktrees." in spk.spoke
    assert butler.asked == []
    assert not task.done()
    await stop(task)


async def test_the_survey_verb_never_removes_anything():
    """Report before acting; never act on the utterance that asked."""
    bus, butler, spk, cleanup = EventBus(), FakeButler(), FakeSpeaker(), FakeCleanup()
    task = await run(bus, butler, spk, router=Router(),
                     registry=confirmed_registry("soccer"), cleanup=cleanup)
    for said in ("clean up the worktrees", "tidy up the worktrees",
                 "go through the worktrees"):
        bus.publish("command.received", {"text": said})
    await asyncio.sleep(0.05)
    assert {c[0] for c in cleanup.calls} == {"report"}
    await stop(task)


async def test_the_batch_verb_reaches_the_gate():
    bus, butler, spk, cleanup = EventBus(), FakeButler(), FakeSpeaker(), FakeCleanup()
    task = await run(bus, butler, spk, router=Router(),
                     registry=confirmed_registry("soccer"), cleanup=cleanup)
    bus.publish("command.received", {"text": "remove the empty worktrees"})
    await asyncio.sleep(0.05)
    assert cleanup.calls == [("remove_empty",)]
    await stop(task)


async def test_the_per_item_verb_carries_the_spoken_name():
    bus, butler, spk, cleanup = EventBus(), FakeButler(), FakeSpeaker(), FakeCleanup()
    task = await run(bus, butler, spk, router=Router(),
                     registry=confirmed_registry("soccer"), cleanup=cleanup)
    bus.publish("command.received", {"text": "remove the worktree for soccer"})
    await asyncio.sleep(0.05)
    assert cleanup.calls == [("remove_named", "soccer")]
    await stop(task)


async def test_without_a_cleanup_the_brain_says_so_honestly():
    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = await run(bus, butler, spk, router=Router(),
                     registry=confirmed_registry("soccer"))
    bus.publish("command.received", {"text": "remove the empty worktrees"})
    await asyncio.sleep(0.05)
    assert any("can't run that yet" in s for s in spk.spoke)
    assert not task.done()
    await stop(task)


async def test_a_cleanup_that_raises_is_spoken_never_raised():
    class Boom:
        async def report(self):
            raise RuntimeError("git exploded")

        async def remove_empty(self):
            raise RuntimeError("git exploded")

        async def remove_named(self, name):
            raise RuntimeError("git exploded")

    bus, butler, spk = EventBus(), FakeButler(), FakeSpeaker()
    task = await run(bus, butler, spk, router=Router(),
                     registry=confirmed_registry("soccer"), cleanup=Boom())
    for said in ("clean up the worktrees", "remove the empty worktrees",
                 "remove the worktree for soccer"):
        bus.publish("command.received", {"text": said})
    await asyncio.sleep(0.05)
    assert spk.spoke.count("Sorry sir, that command failed.") == 3
    assert not task.done()                       # the brain must never die
    await stop(task)


# -------------------------------------- the fourth-question interaction -----
async def test_the_removal_verb_survives_all_three_other_pending_questions():
    """Onboarding's repo confirm and the finance source confirm both own
    yes/no-shaped speech TERMINALLY, and a spoken tool approval is pending too.
    A destructive worktree instruction is none of those shapes, so it reaches
    the gate untouched — and none of the three is disturbed by it."""
    bus, butler, spk, cleanup = EventBus(), FakeButler(), FakeSpeaker(), FakeCleanup()
    router = Router()
    open_spoken(router, "soccer", "rm -rf build", now=time.time())
    task = await run(bus, butler, spk, router=router,
                     registry=confirmed_registry("soccer"),
                     onboarding=PendingConfirm(), finance=PendingSource(),
                     cleanup=cleanup)
    bus.publish("command.received", {"text": "remove the empty worktrees"})
    await asyncio.sleep(0.05)
    assert cleanup.calls == [("remove_empty",)]
    assert len(router.pending_approvals()) == 1   # the approval is untouched
    assert butler.asked == []
    await stop(task)


@pytest.mark.parametrize("said", ["yes", "yeah", "sure", "approved", "okay",
                                  "go ahead", "do it"])
async def test_no_affirmation_anywhere_can_remove_a_worktree(said):
    """THE fail-open this design refuses to become the seventh instance of.
    A yes belongs to whichever question is pending; it must never be able to
    reach the cleanup gate, whether one is pending or none is."""
    bus, butler, spk, cleanup = EventBus(), FakeButler(), FakeSpeaker(), FakeCleanup()
    router = Router()
    open_spoken(router, "soccer", "npm test", now=time.time())
    task = await run(bus, butler, spk, router=router,
                     registry=confirmed_registry("soccer"), cleanup=cleanup)
    bus.publish("command.received", {"text": said})
    await asyncio.sleep(0.05)
    assert cleanup.calls == []
    await stop(task)


async def test_a_yes_right_after_a_survey_still_removes_nothing():
    # The realistic shape of the accident: JARVIS reads the survey, Keke says
    # "yes". Nothing removable follows from that — the gate is never consulted,
    # and the utterance goes on to be ordinary conversation.
    bus, butler, spk, cleanup = EventBus(), FakeButler(), FakeSpeaker(), FakeCleanup()
    task = await run(bus, butler, spk, router=Router(),
                     registry=confirmed_registry("soccer"), cleanup=cleanup)
    bus.publish("command.received", {"text": "clean up the worktrees"})
    await asyncio.sleep(0.05)
    bus.publish("command.received", {"text": "yes, go ahead"})
    await asyncio.sleep(0.05)
    assert cleanup.calls == [("report",)]
    await stop(task)


async def test_a_worktree_instruction_does_not_answer_the_repo_question():
    """The mirror direction: a real Onboarding mid-question must not read
    "remove the empty worktrees" as an answer of any kind."""
    from server.onboarding import Onboarding
    bus, butler, spk, cleanup = EventBus(), FakeButler(), FakeSpeaker(), FakeCleanup()
    reg = Registry()
    reg.merge_candidates([Candidate(path="/p/soccer", name="soccer",
                                    sources=["t"])])
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        ob = Onboarding(bus, reg, Path(tmp) / "projects.json")
        await ob.ask_next()
        task = await run(bus, butler, spk, router=Router(), registry=reg,
                         onboarding=ob, cleanup=cleanup)
        bus.publish("command.received", {"text": "remove the empty worktrees"})
        await asyncio.sleep(0.05)
        assert cleanup.calls == [("remove_empty",)]
        assert not any(p.confirmed for p in reg.projects)   # still pending
        assert ob.awaiting                                   # still asking
        await stop(task)


# ------------------------------------------------------------ the route -----
def test_get_worktrees_is_cookie_authed_and_never_the_hook_lane(tmp_path):
    """Same lane as /fleet, deliberately NOT widened: the hook bearer lives in
    every worktree's settings.local.json, so a worker holding it must not be
    able to enumerate — let alone reason about — its siblings."""
    from tests.test_app_auth import bootstrap, make_client
    c = make_client(tmp_path)
    assert c.get("/worktrees").status_code == 401
    hdrs = {"Authorization": f"Bearer {c.app.state.cfg.hook_bearer}"}
    assert c.get("/worktrees", headers=hdrs).status_code == 401
    bootstrap(c)
    body = c.get("/worktrees").json()
    assert body["worktrees"] == []          # state/worktrees does not exist yet
    assert body["error"] == ""


def test_get_worktrees_reports_a_real_worktree(tmp_path):
    import subprocess
    from pathlib import Path

    from tests.test_app_auth import bootstrap, make_client
    from tests.test_worktree_survey import add_worktree

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", "."], cwd=repo, check=True)
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], cwd=repo, check=True)
    c = make_client(tmp_path)
    bootstrap(c)
    wts = tmp_path / "state" / "worktrees"
    # asyncio.run, not a loop nobody closes: this is a SYNC test (the route is
    # driven through TestClient), and the previous spelling leaked an event
    # loop per run. `wt` is bound BEFORE the try, because binding it inside
    # meant a failure in add_worktree raised NameError in the finally and
    # buried the real error.
    wt = asyncio.run(add_worktree(repo, wts, "left a draft"))
    try:
        (Path(wt.path) / "draft.md").write_text("keep\n", encoding="utf-8")
        body = c.get("/worktrees").json()
        assert [w["kind"] for w in body["worktrees"]] == ["holds-work"]
        assert body["worktrees"][0]["untracked"] == 1
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt.path)],
                       cwd=repo, check=False, capture_output=True)
        subprocess.run(["git", "worktree", "prune"], cwd=repo, check=False,
                       capture_output=True)


def test_a_survey_fault_is_an_error_field_not_a_500(tmp_path):
    from tests.test_app_auth import bootstrap, make_client
    c = make_client(tmp_path)
    bootstrap(c)

    class Boom:
        async def entries(self):
            raise RuntimeError("git exploded")

    c.app.state.cleanup = Boom()
    body = c.get("/worktrees").json()
    assert body["worktrees"] == [] and "git exploded" in body["error"]
