"""The worktree survey and its consented cleanup.

Everything here runs against REAL temporary git repos and REAL worktrees, the
way tests/test_worktrees.py already does: classification is a set of claims
about what git says, and claims about git tested against fake data prove
nothing. The `repo` fixture force-removes every worktree it finds registered
at teardown, so a test that leaks one leaks it into a directory pytest throws
away rather than into a checkout somebody uses.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from server.bus import EventBus
from server.fleet_state import DETACHED
from server.worktree_survey import (KIND_EMPTY, KIND_HOLDS_WORK, KIND_LIVE,
                                    KIND_STALE, KIND_UNRECOGNIZED,
                                    OFFER_TTL_S, WorktreeCleanup, survey)
from server.worktrees import create_worktree, write_hook_settings


# ---------------------------------------------------------------- fixtures --
def _git(cwd, *args, check=True):
    return subprocess.run(["git", *args], cwd=str(cwd), check=check,
                          capture_output=True, text=True).stdout.strip()


def _commit(cwd, message):
    _git(cwd, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam",
         message)


@pytest.fixture
def repo(tmp_path):
    """A real checkout on `main` with one commit. Teardown unregisters every
    worktree it ends up with, so nothing survives the test."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main", ".")
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(root, "add", ".")
    _commit(root, "init")
    try:
        yield root
    finally:
        for line in _git(root, "worktree", "list", "--porcelain",
                         check=False).splitlines():
            if line.startswith("worktree ") and line[9:] != str(root):
                subprocess.run(["git", "worktree", "remove", "--force",
                                line[9:]], cwd=str(root), check=False,
                               capture_output=True)
        subprocess.run(["git", "worktree", "prune"], cwd=str(root),
                       check=False, capture_output=True)


class FakeFleet:
    """Only the three things the cleanup gate reads off the real Fleet."""

    def __init__(self, worktrees_dir, live=(), repos=()):
        self.worktrees_dir = Path(worktrees_dir)
        self._live = [str(p) for p in live]
        self._repos = [str(p) for p in repos]

    def live_worktree_paths(self):
        return set(self._live)

    def known_repos(self):
        return set(self._repos)


def make_gate(worktrees_dir, live=(), repos=(), now=None):
    bus = EventBus()
    fleet = FakeFleet(worktrees_dir, live=live, repos=repos)
    gate = WorktreeCleanup(bus=bus, fleet=fleet,
                           now=now or (lambda: 1_000_000.0))
    return gate, bus, fleet


def by_path(entries):
    return {e.path: e for e in entries}


async def add_worktree(repo, wts, task):
    """A worktree exactly as the fleet makes them — same namespace, same
    hook settings, same self-ignoring shield."""
    from server.fleet import _shield_bearer
    wt = await create_worktree(repo, task, wts)
    write_hook_settings(Path(wt.path), 7777, "tok")
    _shield_bearer(Path(wt.path))
    return wt


# ---------------------------------------------------------- classification --
async def test_a_fresh_worktree_with_only_jarvis_plumbing_is_empty(repo, tmp_path):
    # The hook settings and the bearer shield are written BEFORE the CLI ever
    # runs. If they counted as work, every worktree JARVIS ever made would
    # classify as holds-work and the empty bucket would be permanently empty —
    # the feature would surface a pile it could never offer to clear.
    wt = await add_worktree(repo, tmp_path / "wts", "did nothing")
    entries = await survey(tmp_path / "wts")
    assert [e.kind for e in entries] == [KIND_EMPTY]
    e = entries[0]
    assert e.branch == wt.branch and e.ahead == 0
    assert e.dirty == 0 and e.untracked == 0
    assert e.base_commit == wt.base_commit


async def test_an_untracked_file_alone_makes_it_holds_work(repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "left a scratch file")
    (Path(wt.path) / "notes.txt").write_text("hours of work\n", encoding="utf-8")
    e = (await survey(tmp_path / "wts"))[0]
    assert e.kind == KIND_HOLDS_WORK and e.untracked == 1 and e.ahead == 0


async def test_a_modified_tracked_file_alone_makes_it_holds_work(repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "edited a file")
    (Path(wt.path) / "README.md").write_text("changed\n", encoding="utf-8")
    e = (await survey(tmp_path / "wts"))[0]
    assert e.kind == KIND_HOLDS_WORK and e.dirty == 1 and e.untracked == 0


async def test_commits_beyond_the_base_make_it_holds_work(repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "committed something")
    (Path(wt.path) / "README.md").write_text("changed\n", encoding="utf-8")
    _commit(wt.path, "worker commit")
    e = (await survey(tmp_path / "wts"))[0]
    assert e.kind == KIND_HOLDS_WORK and e.ahead == 1
    assert e.dirty == 0 and e.untracked == 0
    assert e.base_commit == wt.base_commit      # the fork point, not HEAD


async def test_a_live_worker_worktree_is_live_even_when_it_is_empty(repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "running right now")
    entries = await survey(tmp_path / "wts", live_paths=[wt.path])
    assert [e.kind for e in entries] == [KIND_LIVE]


async def test_a_foreign_branch_is_unrecognized_and_never_removable(repo, tmp_path):
    # Something in the worktrees directory that JARVIS did not create. It is
    # reported (silence would be worse) but it is in neither removable bucket.
    wts = tmp_path / "wts"
    wts.mkdir(parents=True)
    _git(repo, "worktree", "add", "-q", "-b", "feature/human", str(wts / "human"))
    e = (await survey(wts))[0]
    assert e.kind == KIND_UNRECOGNIZED


async def test_a_stale_registration_is_reported_from_the_repo(repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "gone")
    subprocess.run(["rm", "-rf", wt.path], check=True)
    entries = await survey(tmp_path / "wts", repos=[repo])
    assert [e.kind for e in entries] == [KIND_STALE]
    assert entries[0].branch == wt.branch


async def test_the_survey_reports_every_bucket_at_once(repo, tmp_path):
    wts = tmp_path / "wts"
    live = await add_worktree(repo, wts, "live one")
    holds = await add_worktree(repo, wts, "holds work")
    (Path(holds.path) / "draft.md").write_text("keep me\n", encoding="utf-8")
    await add_worktree(repo, wts, "empty one")
    gone = await add_worktree(repo, wts, "gone one")
    subprocess.run(["rm", "-rf", gone.path], check=True)
    kinds = sorted(e.kind for e in
                   await survey(wts, live_paths=[live.path], repos=[repo]))
    assert kinds == sorted([KIND_LIVE, KIND_HOLDS_WORK, KIND_EMPTY, KIND_STALE])


async def test_a_missing_worktrees_directory_surveys_to_nothing(tmp_path):
    assert await survey(tmp_path / "never-created") == []


# ------------------------------------------------------------- the offer ----
async def test_removal_refuses_before_anything_has_been_read_aloud(repo, tmp_path):
    await add_worktree(repo, tmp_path / "wts", "empty one")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    spoken = await gate.remove_empty()
    assert "survey" in spoken.lower() or "tidy" in spoken.lower()
    assert len(await survey(tmp_path / "wts")) == 1      # still there


async def test_an_offer_older_than_the_ttl_is_refused(repo, tmp_path):
    await add_worktree(repo, tmp_path / "wts", "empty one")
    clock = {"t": 1_000_000.0}
    gate, _bus, _fleet = make_gate(tmp_path / "wts", now=lambda: clock["t"])
    await gate.report()
    clock["t"] += OFFER_TTL_S + 1
    spoken = await gate.remove_empty()
    assert "stale" in spoken.lower()
    assert len(await survey(tmp_path / "wts")) == 1


async def test_one_offer_is_redeemable_exactly_once(repo, tmp_path):
    # One sentence, one instruction. A second removal may not ride the same
    # spoken survey — it has to be described again first. Pinned on the
    # NO-OFFER refusal specifically: the "that changed since I read it to you"
    # sentence also mentions the survey, so a loose match here would go green
    # with the one-shot consumption fully reverted.
    await add_worktree(repo, tmp_path / "wts", "empty one")
    await add_worktree(repo, tmp_path / "wts", "empty two")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()
    await gate.remove_empty()
    second = await gate.remove_empty()
    assert "haven't gone through" in second
    assert "removed" not in second.lower()


async def test_a_second_named_removal_cannot_ride_the_same_survey(repo, tmp_path):
    a = await add_worktree(repo, tmp_path / "wts", "soccer one")
    b = await add_worktree(repo, tmp_path / "wts", "alethic two")
    for wt in (a, b):
        (Path(wt.path) / "draft.md").write_text("keep\n", encoding="utf-8")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()
    await gate.remove_named("soccer one")
    second = await gate.remove_named("alethic two")
    assert "haven't gone through" in second
    assert Path(b.path).exists()                # the second one is untouched


# -------------------------------------------- the batch: empties only -------
async def test_the_batch_removes_the_empties_and_their_branches(repo, tmp_path):
    a = await add_worktree(repo, tmp_path / "wts", "empty a")
    b = await add_worktree(repo, tmp_path / "wts", "empty b")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()
    spoken = await gate.remove_empty()
    assert "2" in spoken
    assert not Path(a.path).exists() and not Path(b.path).exists()
    branches = _git(repo, "branch", "--list", "jarvis/*")
    assert branches == ""                       # nothing beyond base to keep


async def test_the_batch_never_touches_a_worktree_holding_work(repo, tmp_path):
    holds = await add_worktree(repo, tmp_path / "wts", "holds work")
    (Path(holds.path) / "draft.md").write_text("keep me\n", encoding="utf-8")
    empty = await add_worktree(repo, tmp_path / "wts", "empty one")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()
    await gate.remove_empty()
    assert not Path(empty.path).exists()
    assert Path(holds.path).exists()
    assert (Path(holds.path) / "draft.md").read_text(encoding="utf-8") == "keep me\n"


async def test_the_batch_skips_a_worktree_that_gained_work_after_the_report(
        repo, tmp_path):
    # THE DANGEROUS DIRECTION. The offer named this one as empty; between the
    # sentence and the yes, a worker wrote into it. One yes must never remove
    # more than the sentence described — and "empty" is part of what it said.
    wt = await add_worktree(repo, tmp_path / "wts", "empty at survey time")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()
    (Path(wt.path) / "urgent.md").write_text("wrote this after\n", encoding="utf-8")
    spoken = await gate.remove_empty()
    assert Path(wt.path).exists()
    assert (Path(wt.path) / "urgent.md").exists()
    assert "changed" in spoken.lower() or "no longer" in spoken.lower()


async def test_the_batch_skips_a_worktree_that_became_live_after_the_report(
        repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "empty at survey time")
    gate, _bus, fleet = make_gate(tmp_path / "wts")
    await gate.report()
    fleet._live = [wt.path]                     # a worker spawned into it
    await gate.remove_empty()
    assert Path(wt.path).exists()


async def test_a_detached_worktree_survives_the_batch(repo, tmp_path):
    # A human is driving that session in a real terminal right now. The fleet
    # reports it as live; the survey must agree and the batch must skip it even
    # though nothing has been written into it yet.
    wt = await add_worktree(repo, tmp_path / "wts", "detached session")
    gate, _bus, _fleet = make_gate(tmp_path / "wts", live=[wt.path])
    await gate.report()
    await gate.remove_empty()
    assert Path(wt.path).exists()


async def test_the_batch_prunes_stale_registrations(repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "gone")
    subprocess.run(["rm", "-rf", wt.path], check=True)
    gate, _bus, _fleet = make_gate(tmp_path / "wts", repos=[repo])
    await gate.report()
    await gate.remove_empty()
    listed = _git(repo, "worktree", "list", "--porcelain")
    assert "prunable" not in listed
    # Pruning destroys nothing: the branch is the record, and the directory it
    # named is already gone, so this is the ONE trace left of that worker.
    assert wt.branch in _git(repo, "branch", "--list", "jarvis/*")


# ------------------------------------------- per-item: naming one of them ---
async def test_naming_a_worktree_that_holds_work_removes_it_and_keeps_the_branch(
        repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "soccer captain page")
    (Path(wt.path) / "README.md").write_text("changed\n", encoding="utf-8")
    _commit(wt.path, "worker commit")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()
    spoken = await gate.remove_named("soccer captain page")
    assert not Path(wt.path).exists()
    assert wt.branch in _git(repo, "branch", "--list", "jarvis/*")
    assert wt.branch in spoken                  # says where the work still is


async def test_naming_something_the_entry_cannot_explain_removes_nothing(
        repo, tmp_path):
    # THE WHITELIST. "remove the worktree for soccer, and the empty ones too"
    # names a second thing this code did not understand. One unaccounted word
    # and nothing happens — the same rule _approval_vocabulary enforces.
    wt = await add_worktree(repo, tmp_path / "wts", "soccer captain page")
    (Path(wt.path) / "draft.md").write_text("keep me\n", encoding="utf-8")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()
    spoken = await gate.remove_named("soccer and everything older than a week")
    assert Path(wt.path).exists()
    assert "soccer" in spoken.lower() or "name" in spoken.lower()


async def test_naming_a_live_worktree_is_refused_by_name(repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "soccer captain page")
    gate, _bus, _fleet = make_gate(tmp_path / "wts", live=[wt.path])
    await gate.report()
    spoken = await gate.remove_named("soccer captain page")
    assert Path(wt.path).exists()
    assert "running" in spoken.lower() or "terminal" in spoken.lower()


async def test_an_ambiguous_name_removes_nothing(repo, tmp_path):
    a = await add_worktree(repo, tmp_path / "wts", "soccer one")
    b = await add_worktree(repo, tmp_path / "wts", "soccer two")
    for wt in (a, b):
        (Path(wt.path) / "draft.md").write_text("keep\n", encoding="utf-8")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()
    spoken = await gate.remove_named("soccer")
    assert Path(a.path).exists() and Path(b.path).exists()
    assert "more than one" in spoken.lower()


async def test_a_named_worktree_that_changed_since_the_report_is_refused(
        repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "soccer captain page")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()                         # reported EMPTY
    (Path(wt.path) / "urgent.md").write_text("wrote this after\n", encoding="utf-8")
    spoken = await gate.remove_named("soccer captain page")
    assert Path(wt.path).exists()
    assert "changed" in spoken.lower()


async def test_a_foreign_checkout_cannot_be_removed_by_name(repo, tmp_path):
    wts = tmp_path / "wts"
    wts.mkdir(parents=True)
    _git(repo, "worktree", "add", "-q", "-b", "feature/human", str(wts / "human"))
    (wts / "human" / "notes.txt").write_text("hours of work\n", encoding="utf-8")
    gate, _bus, _fleet = make_gate(wts)
    await gate.report()
    await gate.remove_named("human")
    assert (wts / "human" / "notes.txt").exists()


# ------------------------------------------------------------- containment --
async def test_removal_refuses_a_target_outside_the_worktrees_directory(
        repo, tmp_path):
    # A worktree that is a real jarvis one but lives somewhere else entirely.
    # Only the configured directory is ever cleaned; the offer is built from a
    # survey of it, so this can only happen to a forged offer — refuse anyway.
    outside = tmp_path / "elsewhere"
    wt = await add_worktree(repo, outside, "not under the configured dir")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    entry = (await survey(outside))[0]
    ok, why = await gate._remove(entry)
    assert ok is False and "outside" in why.lower()
    assert Path(wt.path).exists()


async def test_removal_goes_through_the_existing_guard(repo, tmp_path, monkeypatch):
    # Never reimplement, never work around: the removal path must call
    # server.worktrees.remove_worktree, whose docstring is four proven
    # failures long.
    wt = await add_worktree(repo, tmp_path / "wts", "empty one")
    calls = []

    import server.worktree_survey as mod
    real = mod.remove_worktree

    async def spy(record):
        calls.append(record)
        return await real(record)

    monkeypatch.setattr(mod, "remove_worktree", spy)
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()
    await gate.remove_empty()
    assert [c.branch for c in calls] == [wt.branch]
    assert Path(calls[0].path).is_absolute()


# ---------------------------------------------------------- spoken failures -
async def test_a_broken_fleet_is_spoken_never_raised(tmp_path):
    class Exploding:
        worktrees_dir = tmp_path / "wts"

        def live_worktree_paths(self):
            raise RuntimeError("boom")

        def known_repos(self):
            return set()

    gate = WorktreeCleanup(bus=EventBus(), fleet=Exploding())
    spoken = await gate.report()
    assert isinstance(spoken, str) and spoken
    assert "couldn't" in spoken.lower() or "could not" in spoken.lower()
    spoken = await gate.remove_empty()
    assert isinstance(spoken, str) and spoken


async def test_the_report_publishes_the_survey_for_the_console(repo, tmp_path):
    await add_worktree(repo, tmp_path / "wts", "empty one")
    gate, bus, _fleet = make_gate(tmp_path / "wts")
    cid, q = bus.subscribe()
    try:
        await gate.report()
        event = q.get_nowait()
    finally:
        bus.unsubscribe(cid)
    assert event["type"] == "worktrees.survey"
    assert event["data"]["worktrees"][0]["kind"] == KIND_EMPTY


async def test_the_report_states_what_would_be_lost(repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "soccer captain page")
    (Path(wt.path) / "README.md").write_text("changed\n", encoding="utf-8")
    _commit(wt.path, "worker commit")
    (Path(wt.path) / "scratch.md").write_text("draft\n", encoding="utf-8")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    spoken = await gate.report()
    assert "1 commit" in spoken and "1 untracked file" in spoken


# --------------------------------------------------------- the Fleet's view -
async def test_fleet_reports_detached_workers_and_ghosts_as_live():
    from types import SimpleNamespace

    from server.fleet_state import CLOSED, IDLE_AT_PROMPT

    class F:
        pass

    from server.fleet import Fleet
    fleet = Fleet.__new__(Fleet)
    fleet.workers = [
        SimpleNamespace(path="/p/a", machine=SimpleNamespace(base=IDLE_AT_PROMPT),
                        worktree=SimpleNamespace(path="/w/live")),
        SimpleNamespace(path="/p/b", machine=SimpleNamespace(base=DETACHED),
                        worktree=SimpleNamespace(path="/w/detached")),
        SimpleNamespace(path="/p/c", machine=SimpleNamespace(base=CLOSED),
                        worktree=SimpleNamespace(path="/w/closed")),
    ]
    fleet.ghosts = [
        {"path": "/p/d", "state": DETACHED, "worktree": "/w/ghost-detached"},
        {"path": "/p/e", "state": "UNKNOWN", "worktree": "/w/ghost-interrupted"},
    ]
    live = fleet.live_worktree_paths()
    assert "/w/live" in live and "/w/detached" in live
    assert "/w/ghost-detached" in live
    assert "/w/closed" not in live               # nothing left to protect
    assert "/w/ghost-interrupted" not in live    # died with the old server
    assert fleet.known_repos() >= {"/p/a", "/p/b", "/p/c", "/p/d", "/p/e"}
