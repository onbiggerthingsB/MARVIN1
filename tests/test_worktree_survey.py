"""The worktree survey and its consented cleanup.

Everything here runs against REAL temporary git repos and REAL worktrees, the
way tests/test_worktrees.py already does: classification is a set of claims
about what git says, and claims about git tested against fake data prove
nothing. The `repo` fixture force-removes every worktree it finds registered
at teardown, so a test that leaks one leaks it into a directory pytest throws
away rather than into a checkout somebody uses.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from server.bus import EventBus
from server.fleet_state import DETACHED
from server.worktree_survey import (KIND_EMPTY, KIND_HOLDS_WORK, KIND_LIVE,
                                    KIND_ORPHAN_BRANCH, KIND_STALE,
                                    KIND_UNRECOGNIZED, OFFER_TTL_S,
                                    WorktreeCleanup, spoken_report, survey)
from server.worktrees import Worktree, create_worktree, write_hook_settings


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


def add_worktree_stamped(repo, wts, task, stamp):
    """`create_worktree` with the timestamp chosen by the test.

    create_worktree stamps to the SECOND, so two worktrees cut for the same
    task in one process cannot exist unless the stamps are forced apart. Two
    that DO exist are the case this file has to cover: same repo, same slug,
    identical spoken label."""
    from server.fleet import _shield_bearer
    from server.worktrees import _slug
    slug = _slug(task)
    dest = Path(wts) / f"{Path(repo).name}-{slug}-{stamp}"
    branch = f"marvin/{slug}-{stamp}"
    base = _git(repo, "rev-parse", "HEAD")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "-q", "-b", branch, str(dest), base)
    write_hook_settings(dest, 7777, "tok")
    _shield_bearer(dest)
    return Worktree(repo=str(repo), path=str(dest), branch=branch,
                    base_commit=base)


def move_head_off_the_worktrees(repo):
    """Put the MAIN checkout on a branch that contains none of the worktree
    commits — a human switching branches, nothing more exotic.

    `git branch -d` measures merged-ness against HEAD, so after this every
    marvin branch cut from main is 'not fully merged' and git refuses to
    delete it. That refusal is what the batch used to swallow."""
    _git(repo, "checkout", "-q", "--orphan", "elsewhere")
    _commit(repo, "somewhere else entirely")


# ---------------------------------------------------------- classification --
async def test_a_fresh_worktree_with_only_marvin_plumbing_is_empty(repo, tmp_path):
    # The hook settings and the bearer shield are written BEFORE the CLI ever
    # runs. If they counted as work, every worktree Marvin ever made would
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


async def test_a_live_worktree_recorded_under_a_different_spelling_is_live(
        repo, tmp_path):
    # THE ONE FAIL-OPEN SURFACE. The fleet records the path it built; git
    # answers with its own canonical spelling. On a case-insensitive
    # filesystem git case-corrects and Path.resolve() does not, so two strings
    # for ONE directory compare unequal — and a string mismatch here classifies
    # a running worker's checkout as empty, which is removable.
    wt = await add_worktree(repo, tmp_path / "wts", "running right now")
    variant = str(Path(wt.path).with_name(Path(wt.path).name.upper()))
    if variant == wt.path or not os.path.exists(variant):
        pytest.skip("case-sensitive filesystem: the two spellings are two "
                    "different directories here, so there is nothing to miss")
    entries = await survey(tmp_path / "wts", live_paths=[variant])
    assert [e.kind for e in entries] == [KIND_LIVE]


async def test_a_stray_file_in_the_worktrees_directory_is_reported(
        repo, tmp_path):
    # Silence about a thing you cannot classify is worse than naming it — the
    # module's own words. A file (or a symlink to one) was skipped outright.
    wts = tmp_path / "wts"
    wts.mkdir(parents=True)
    (wts / "notes.txt").write_text("stray\n", encoding="utf-8")
    (wts / "pointer").symlink_to(wts / "notes.txt")
    entries = await survey(wts)
    assert [e.kind for e in entries] == [KIND_UNRECOGNIZED, KIND_UNRECOGNIZED]
    assert {Path(e.path).name for e in entries} == {"notes.txt", "pointer"}
    assert all(e.note for e in entries)         # each says why
    assert not any(e.removable for e in entries)


async def test_a_foreign_branch_is_unrecognized_and_never_removable(repo, tmp_path):
    # Something in the worktrees directory that Marvin did not create. It is
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
    branches = _git(repo, "branch", "--list", "marvin/*")
    assert branches == ""                       # nothing beyond base to keep
    # The claim is only allowed BECAUSE git agreed. HEAD is still main here,
    # which contains both base commits; the test below is the same batch with
    # HEAD moved, where git refuses and this sentence would be a lie. Pinned on
    # the WHOLE clause: "…and 2 of their branches; git wouldn't delete…" also
    # contains the words "their branches" and means the opposite.
    assert "Removed 2 empty worktrees, sir, and their branches." in spoken


async def test_the_batch_never_claims_a_branch_git_refused_to_delete(
        repo, tmp_path):
    # THE FALSE SENTENCE. `git branch -d` fails whenever HEAD has moved off
    # the commit the worktree was cut from, and the batch used to swallow the
    # failure and say "and its branch" anyway.
    wt = await add_worktree(repo, tmp_path / "wts", "empty a")
    move_head_off_the_worktrees(repo)
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()
    spoken = await gate.remove_empty()
    assert not Path(wt.path).exists()                   # the directory did go
    assert wt.branch in _git(repo, "branch", "--list", "marvin/*")
    assert "and its branch." not in spoken              # the lie
    assert "kept" in spoken.lower()
    assert wt.branch in spoken                          # names what survived


async def test_a_branch_that_survived_the_batch_stays_visible(repo, tmp_path):
    # Worse than the false sentence: once the directory is gone the branch has
    # no directory and no registration, so without a bucket of its own the
    # survey could never surface it again — the unbounded accumulation this
    # whole feature exists to fix.
    wt = await add_worktree(repo, tmp_path / "wts", "empty a")
    move_head_off_the_worktrees(repo)
    gate, _bus, _fleet = make_gate(tmp_path / "wts", repos=[repo])
    await gate.report()
    await gate.remove_empty()
    entries = await survey(tmp_path / "wts", repos=[repo])
    assert [e.kind for e in entries] == [KIND_ORPHAN_BRANCH]
    assert entries[0].branch == wt.branch
    assert entries[0].removable is False        # report-only, never cleared
    assert "branch" in spoken_report(entries).lower()


async def test_an_orphan_branch_is_refused_by_name(repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "soccer captain page")
    move_head_off_the_worktrees(repo)
    gate, _bus, _fleet = make_gate(tmp_path / "wts", repos=[repo])
    await gate.report()
    await gate.remove_empty()                   # leaves the branch behind
    await gate.report()                         # now it is an orphan branch
    spoken = await gate.remove_named("soccer captain page")
    assert wt.branch in _git(repo, "branch", "--list", "marvin/*")
    assert "branch" in spoken.lower()


async def test_the_prune_leaves_a_registration_the_survey_never_mentioned(
        repo, tmp_path):
    # A human's own worktree on the same repo, outside the directory Marvin
    # cleans, whose checkout is temporarily missing — an unplugged drive, a
    # moved directory, the case `git worktree repair` exists for. The survey
    # never mentioned it, so the batch must not clear its registration; the
    # spoken count is the count of NAMED entries, and the action has to match.
    mine = await add_worktree(repo, tmp_path / "wts", "gone")
    subprocess.run(["rm", "-rf", mine.path], check=True)
    theirs = tmp_path / "human-checkout"
    _git(repo, "worktree", "add", "-q", "-b", "feature/human", str(theirs))
    subprocess.run(["rm", "-rf", str(theirs)], check=True)
    gate, _bus, _fleet = make_gate(tmp_path / "wts", repos=[repo])
    spoken = await gate.report()
    assert "1 registration" in spoken or "One registration" in spoken
    await gate.remove_empty()
    listing = _git(repo, "worktree", "list", "--porcelain")
    assert mine.path not in listing             # ours, named aloud, is cleared
    assert str(theirs) in listing               # theirs, unmentioned, survives


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
    assert wt.branch in _git(repo, "branch", "--list", "marvin/*")


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
    assert wt.branch in _git(repo, "branch", "--list", "marvin/*")
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


async def test_an_ambiguous_name_leaves_the_offer_standing(repo, tmp_path):
    # The refusal removed NOTHING, so it must not spend the survey: the whole
    # point of refusing is to let Keke say a better name, and consuming the
    # offer answered the follow-up with "I haven't gone through your worktrees
    # yet" — advice that cannot be followed.
    a = add_worktree_stamped(repo, tmp_path / "wts", "soccer captain page",
                             "20260101-000001")
    b = add_worktree_stamped(repo, tmp_path / "wts", "soccer captain page",
                             "20260102-000002")
    for wt in (a, b):
        (Path(wt.path) / "draft.md").write_text("keep\n", encoding="utf-8")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    spoken_survey = await gate.report()
    refusal = await gate.remove_named("soccer captain page")
    assert "more than one" in refusal.lower()
    assert "haven't gone through" not in refusal
    assert Path(a.path).exists() and Path(b.path).exists()
    # …and the name the refusal offered is one the SURVEY already spoke, so
    # saying it back is an instruction this gate can act on.
    assert "soccer captain page one" in spoken_survey
    assert "soccer captain page one" in refusal
    spoken = await gate.remove_named("soccer captain page one")
    assert "haven't gone through" not in spoken
    assert not Path(a.path).exists() and Path(b.path).exists()


async def test_two_worktrees_for_one_task_are_told_apart_by_their_branch(
        repo, tmp_path):
    # "Name its branch instead" was impossible advice twice over: the second
    # reason is that router._words drops digits, so two worktrees cut for the
    # same task on the same repo had IDENTICAL branch vocabularies and the
    # branch could not disambiguate them either.
    a = add_worktree_stamped(repo, tmp_path / "wts", "soccer captain page",
                             "20260101-000001")
    b = add_worktree_stamped(repo, tmp_path / "wts", "soccer captain page",
                             "20260102-000002")
    for wt in (a, b):
        (Path(wt.path) / "draft.md").write_text("keep\n", encoding="utf-8")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()
    spoken = await gate.remove_named(b.branch)
    assert Path(a.path).exists() and not Path(b.path).exists()
    assert b.branch in spoken


async def test_a_name_made_only_of_politeness_removes_nothing(repo, tmp_path):
    # A plausible STT truncation of "remove the worktree for the soccer page".
    # The whitelist is _POLITE plus the entry's words, so "the" was explained
    # by every entry, matched all of them, and removed the only one whenever
    # the offer held exactly one.
    wt = await add_worktree(repo, tmp_path / "wts", "soccer captain page")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()
    spoken = await gate.remove_named("the")
    assert Path(wt.path).exists()
    assert "removed" not in spoken.lower()


async def test_a_named_worktree_that_gained_files_since_the_report_is_refused(
        repo, tmp_path):
    # THE SPOKEN LOSS, not the classification. The survey said "1 commit";
    # untracked files arrived afterwards, the kind is still holds-work, and
    # `worktree remove --force` would destroy files that were never in the
    # sentence Keke answered.
    wt = await add_worktree(repo, tmp_path / "wts", "soccer captain page")
    (Path(wt.path) / "README.md").write_text("changed\n", encoding="utf-8")
    _commit(wt.path, "worker commit")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    spoken_survey = await gate.report()
    assert "1 commit" in spoken_survey and "untracked" not in spoken_survey
    (Path(wt.path) / "urgent.md").write_text("wrote this after\n",
                                             encoding="utf-8")
    spoken = await gate.remove_named("soccer captain page")
    assert Path(wt.path).exists()
    assert (Path(wt.path) / "urgent.md").exists()
    assert "changed" in spoken.lower()


async def test_a_named_worktree_that_changed_since_the_report_is_refused(
        repo, tmp_path):
    wt = await add_worktree(repo, tmp_path / "wts", "soccer captain page")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    await gate.report()                         # reported EMPTY
    (Path(wt.path) / "urgent.md").write_text("wrote this after\n", encoding="utf-8")
    spoken = await gate.remove_named("soccer captain page")
    assert Path(wt.path).exists()
    assert "changed" in spoken.lower()


async def test_a_foreign_checkout_is_refused_before_the_destructive_call(
        repo, tmp_path, monkeypatch):
    # ONE guarantee, pinned once: the refusal happens in this module, BEFORE
    # remove_worktree is called at all. remove_worktree's own namespace guard
    # would refuse it too — which is exactly why asserting only "the file
    # survived" passed with either guard reverted and pinned neither.
    wts = tmp_path / "wts"
    wts.mkdir(parents=True)
    _git(repo, "worktree", "add", "-q", "-b", "feature/human", str(wts / "human"))
    (wts / "human" / "notes.txt").write_text("hours of work\n", encoding="utf-8")

    import server.worktree_survey as mod
    calls = []
    real = mod.remove_worktree

    async def spy(record):
        calls.append(record)
        return await real(record)

    monkeypatch.setattr(mod, "remove_worktree", spy)
    gate, _bus, _fleet = make_gate(wts)
    await gate.report()
    spoken = await gate.remove_named("human")
    assert calls == []                          # never reached the removal
    assert "didn't create" in spoken
    assert (wts / "human" / "notes.txt").exists()


# ------------------------------------------------------------- containment --
async def test_removal_refuses_a_target_outside_the_worktrees_directory(
        repo, tmp_path):
    # A worktree that is a real marvin one but lives somewhere else entirely.
    # Only the configured directory is ever cleaned; the offer is built from a
    # survey of it, so this can only happen to a forged offer — refuse anyway.
    outside = tmp_path / "elsewhere"
    wt = await add_worktree(repo, outside, "not under the configured dir")
    gate, _bus, _fleet = make_gate(tmp_path / "wts")
    entry = (await survey(outside))[0]
    ok, why, _branch_gone = await gate._remove(entry)
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


async def test_the_report_says_what_a_live_worktree_holds(repo, tmp_path):
    # _classify gathers ahead/dirty/untracked for live entries too, and says
    # in as many words that it does so BECAUSE the report has to state them —
    # then the sentence said only "I won't touch it", which reads as "there is
    # nothing in there" and is the opposite of why it is being left alone.
    wt = await add_worktree(repo, tmp_path / "wts", "soccer captain page")
    (Path(wt.path) / "README.md").write_text("changed\n", encoding="utf-8")
    _commit(wt.path, "worker commit")
    (Path(wt.path) / "scratch.md").write_text("draft\n", encoding="utf-8")
    gate, _bus, _fleet = make_gate(tmp_path / "wts", live=[wt.path])
    spoken = await gate.report()
    assert "belongs to a session" in spoken
    assert "1 commit" in spoken and "1 untracked file" in spoken
    assert "won't touch" in spoken


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
