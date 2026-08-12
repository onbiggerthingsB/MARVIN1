"""Surfacing what the fleet leaves behind — and a CONSENTED way to clear the
part of it that is provably worthless.

Nothing in `server/` has ever removed a worktree, deliberately: `remove_worktree`
is explicit cleanup only, because "the worktree holds the diff a human may still
want to merge back". The consequence is that every task ever run leaves a
directory and a `marvin/*` branch behind, with nothing to surface it and no way
to clear it. This module is the surfacing, and a consent path narrow enough that
losing a diff Keke wanted is not one of its outcomes.

THIS IS NOT AUTOMATIC DELETION. Nothing here runs on a timer, at boot, or on
shutdown. Every removal is reached by an utterance whose whole job is removal,
and only after a SEPARATE utterance made Marvin read the survey out loud.

The four classifications, and why each is safe:

  live              A worker in any non-final state — or a DETACHED one, where
                    a human is driving that session in a real terminal right
                    now. Never removable, consent or not. Live-ness is re-read
                    at removal time, never trusted from the survey.
  holds-work        Commits beyond its base, tracked modifications, or
                    untracked files. Removable ONLY by naming that one
                    worktree, after the survey said aloud what it holds. Its
                    branch is never deleted: after the directory goes, the
                    branch is the only remaining trace of the work.
  empty             None of the above. The worker did nothing worth keeping,
                    so there is nothing to lose — the batch's single
                    confirmation covers these and nothing else. Its branch has
                    no commits beyond base, so the branch goes too.
  stale-registration
                    Git still registers it but the directory is gone. Clearing
                    it destroys nothing — not the branch, not any object — and
                    it is cleared ONE REGISTRATION AT A TIME, by name.
                    `git worktree prune` would take the whole repo with it,
                    including registrations the survey never mentioned and a
                    human may be about to `git worktree repair`.
  orphan-branch     A `marvin/*` branch with no worktree and no registration
                    anywhere. REPORT-ONLY: nothing here ever deletes it. It
                    exists because `git branch -d` is allowed to refuse — when
                    it does, the directory is already gone, and without this
                    bucket the branch would have no directory and no
                    registration to surface it ever again. That is the
                    unbounded accumulation this module exists to end, so a
                    branch that survives a removal stays visible instead.

There is a FIFTH bucket, `unrecognized`, for anything in the worktrees
directory that is not a marvin worktree: a foreign branch, a symlink, a plain
directory, or a checkout git cannot answer questions about. It is REPORTED
(silence about a thing you cannot classify is worse than naming it) and it is
in no removable bucket. `remove_worktree` would refuse it anyway; this makes
the refusal happen before the destructive call rather than inside it.

`ahead` is counted as "commits on this branch that are reachable from NO
non-marvin ref", which needs no recorded base_commit and degrades in the safe
direction: if the commit this branch was cut from has itself become
unreachable, its ancestors count as ahead too, and the worktree classifies as
holds-work rather than empty.

EVERY SENTENCE HERE IS A CLAIM ABOUT WHAT HAPPENED, and this feature speaks
aloud, so a sentence that outruns what git actually did is a real defect and
not a wording nit. Nothing below reports an outcome it did not observe: the
branch line is phrased from `git branch -d`'s exit status, the prune count is
the count of registrations that were actually cleared, and a removal is
refused outright when the worktree no longer matches the loss that was read
out — not merely when its bucket changed.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from server.router import _POLITE, _words
from server.worktrees import WorktreeError, Worktree, _git, remove_worktree

KIND_LIVE = "live"
KIND_HOLDS_WORK = "holds-work"
KIND_EMPTY = "empty"
KIND_STALE = "stale-registration"
KIND_ORPHAN_BRANCH = "orphan-branch"
KIND_UNRECOGNIZED = "unrecognized"

# What a spoken survey may still authorise, and for how long. Same number as
# router.APPROVAL_TTL_S for the same reason: consent to destroy something goes
# stale, and an offer redeemed ten minutes after the sentence was spoken is
# answering a description of a directory that may no longer be true.
OFFER_TTL_S = 600.0

# How many holds-work items get read aloud before the sentence defers to the
# console. Speech is the expensive channel; the console shows all of them.
SPEAK_ITEM_LIMIT = 3

# `marvin/<slug>-YYYYmmdd-HHMMSS` — create_worktree's own stamp, which dates a
# worktree more honestly than any mtime (a mtime moves when anything reads or
# writes; the stamp is when the worker was spawned).
_STAMP = re.compile(r"-(\d{8}-\d{6})$")

# Runs of digits, which router._words drops by construction ([a-z']+). Two
# worktrees cut for the same task on the same repo differ ONLY in their
# timestamp, so without these their vocabularies are identical and no
# spoken name — branch included — can ever tell them apart.
_DIGITS = re.compile(r"\d+")

# Spoken tie-breakers for two things that would otherwise share a name. Words,
# not numerals: this is read aloud.
_ORDINALS = ("one", "two", "three", "four", "five", "six", "seven", "eight",
             "nine", "ten")


def _spoken_tokens(text: str) -> list[str]:
    """Words AND number runs — the unit `_explains` matches on both sides.

    router._words is the right tokenizer for consent vocabulary and the wrong
    one for identity: it drops digits, and the digits are the only thing that
    distinguishes two worktrees cut for one task. Adding them to BOTH the
    vocabulary and the spoken side keeps the whitelist rule exact — a number
    said aloud still has to be explained by the entry."""
    return _words(text) + _DIGITS.findall(text or "")


@dataclass
class SurveyEntry:
    """One thing found under the worktrees directory. Everything a human needs
    to decide, and nothing a caller has to re-derive."""
    path: str = ""            # git's own canonical absolute path
    repo: str = ""            # the real checkout it was cut from
    branch: str = ""
    base_commit: str = ""
    kind: str = KIND_UNRECOGNIZED
    ahead: int = 0            # commits reachable from no non-marvin ref
    dirty: int = 0            # tracked modifications
    untracked: int = 0        # untracked ENTRIES (a directory collapses to one)
    age_s: float = 0.0
    project: str = ""         # spoken name, when a live worker supplies one
    note: str = ""            # why something is unrecognized, when it is
    alias: str = ""           # set by _disambiguate when two labels collide

    @property
    def removable(self) -> bool:
        return self.kind in (KIND_EMPTY, KIND_HOLDS_WORK)

    @property
    def ident(self) -> str:
        """The key an offer holds this entry under.

        A branch with no worktree has no path at all, and two of those keyed
        on "" would collapse into one — so a refusal would name the wrong
        branch. Every real worktree still keys on its path, unchanged."""
        return self.path or f"{self.repo}#{self.branch}"

    @property
    def label(self) -> str:
        """What to call it out loud.

        The directory name is `<repo>-<slug>-<stamp>`, and the slug is the task
        Keke actually said — the closest thing a worktree has to a human name.
        The stamp is dropped (nobody says "dash two zero two six...") and the
        hyphens become spaces, so the label is speech.

        It must ROUND-TRIP: _explains builds its vocabulary from this same
        directory name, so every label Marvin offers is a name the per-item
        instruction can match back. Two worktrees cut for the same task on the
        same repo would otherwise share a label outright — `alias`, assigned
        once per survey by _disambiguate, is what keeps every spoken name
        pointing at exactly one entry."""
        if self.alias:
            return self.alias
        if self.project:
            return self.project
        name = _STAMP.sub("", Path(self.path).name or "")
        # An orphan branch has no directory; its branch is the only name it
        # has, and "marvin/" is a namespace, not something anybody says.
        name = name or _STAMP.sub("", self.branch.removeprefix("marvin/"))
        return name.replace("-", " ").replace("_", " ").strip() or self.path


def _age_s(branch: str, path: str, now: float) -> float:
    m = _STAMP.search(branch or "")
    if m:
        try:
            return max(0.0, now - time.mktime(
                time.strptime(m.group(1), "%Y%m%d-%H%M%S")))
        except (ValueError, OverflowError):
            pass
    if not path:
        return 0.0        # a branch with no directory has no mtime to read
    try:
        return max(0.0, now - Path(path).stat().st_mtime)
    except OSError:
        return 0.0


class _Live:
    """Which directories belong to a running worker or a live terminal.

    This is the single most safety-critical comparison in the module: a miss
    classifies a running worker's checkout as EMPTY, which is removable by a
    batch. It is therefore decided on FILESYSTEM IDENTITY — `st_dev, st_ino` —
    and not on strings.

    Strings could not be made sound. The fleet records `worktrees_dir / name`
    unresolved and git answers with its own canonical spelling, so the two
    already differ under a symlinked parent; and `Path.resolve()` closes only
    that one gap. It does not case-correct, so on a case-insensitive
    filesystem (this machine's) git's spelling and the fleet's can differ by a
    letter for one directory; it does not normalise Unicode, so a name
    recorded NFD by one process and NFC by another is two strings for one
    directory; and a restart ghost's path was recorded by an EARLIER process,
    which is exactly where a different spelling comes from. An inode is immune
    to all three at once.

    The resolved strings are kept as a fallback and the two are OR-ed, never
    intersected — a path that no longer exists cannot be stat'ed, and the only
    direction this class is allowed to be wrong in is "protected something it
    did not have to"."""

    def __init__(self, live_paths=()):
        self._ids: set[tuple[int, int]] = set()
        self._forms: set[str] = set()
        for p in live_paths or ():
            if not p:
                continue
            self._forms.add(str(p))
            try:
                self._forms.add(str(Path(p).resolve()))
            except (OSError, RuntimeError, ValueError):
                pass
            try:
                st = os.stat(p)
                self._ids.add((st.st_dev, st.st_ino))
            except (OSError, ValueError):
                pass

    def __bool__(self) -> bool:
        return bool(self._ids or self._forms)

    def holds(self, *candidates) -> bool:
        """True if ANY spelling given names a directory a worker still owns."""
        for c in candidates:
            if not c:
                continue
            if str(c) in self._forms:
                return True
            try:
                st = os.stat(c)
            except (OSError, ValueError):
                continue
            if (st.st_dev, st.st_ino) in self._ids:
                return True
        return False


async def _classify(entry_dir: Path, live: _Live, now: float) -> SurveyEntry:
    """One directory. Every git failure lands in `unrecognized`, which is in no
    removable bucket — an entry we could not interrogate is never one we offer
    to delete."""
    e = SurveyEntry(path=str(entry_dir))
    if entry_dir.is_symlink():
        # A symlink here points somewhere else by definition, and the thing it
        # points at is not something a worktree survey was asked about.
        e.note = "a symlink, not a worktree"
        return e
    try:
        top = await _git(entry_dir, "rev-parse", "--path-format=absolute",
                         "--show-toplevel")
        common = await _git(entry_dir, "rev-parse", "--path-format=absolute",
                            "--git-common-dir")
        branch = await _git(entry_dir, "rev-parse", "--abbrev-ref", "HEAD")
    except (WorktreeError, asyncio.TimeoutError, OSError) as exc:
        e.note = f"git could not read it ({exc})"
        return e
    e.path = top
    e.branch = branch
    common_p = Path(common)
    e.repo = str(common_p.parent if common_p.name == ".git" else common_p)
    e.age_s = _age_s(branch, top, now)
    if str(entry_dir.resolve()) != top and str(entry_dir) != top:
        # The directory listed is not the ROOT of the checkout it belongs to —
        # a stray subdirectory of somebody else's worktree, say. Removing by
        # this path would aim git at a checkout nobody surveyed.
        e.note = "not the root of its checkout"
        return e
    if live.holds(entry_dir, top):
        e.kind = KIND_LIVE
    elif not branch.startswith("marvin/"):
        e.note = f"branch {branch!r} is outside the marvin/ namespace"
        return e
    # Facts, gathered for EVERY entry including live ones: the report has to
    # say what a live worker has accumulated too, or "leave it alone" reads as
    # "there is nothing there".
    try:
        ahead = await _git(entry_dir, "rev-list", "HEAD", "--not",
                           "--exclude=marvin/*", "--branches", "--tags",
                           "--remotes", "--boundary")
        status = await _git(entry_dir, "status", "--porcelain")
    except (WorktreeError, asyncio.TimeoutError, OSError) as exc:
        # Half-read is not read. Anything we cannot count, we do not classify —
        # except a LIVE one, which is already classified by something better
        # than git output (the fleet's own record) and must not be demoted to
        # a bucket whose spoken name is "not mine".
        if e.kind != KIND_LIVE:
            e.kind = KIND_UNRECOGNIZED
        e.note = f"git could not read it ({exc})"
        return e
    boundary = [ln[1:] for ln in ahead.splitlines() if ln.startswith("-")]
    e.ahead = sum(1 for ln in ahead.splitlines() if ln and not ln.startswith("-"))
    if boundary:
        e.base_commit = boundary[0]
    else:
        try:
            e.base_commit = await _git(entry_dir, "rev-parse", "HEAD")
        except (WorktreeError, asyncio.TimeoutError, OSError):
            e.base_commit = ""
    lines = [ln for ln in status.splitlines() if ln.strip()]
    e.untracked = sum(1 for ln in lines if ln.startswith("??"))
    e.dirty = len(lines) - e.untracked
    if e.kind == KIND_LIVE:
        return e
    e.kind = (KIND_HOLDS_WORK if (e.ahead or e.dirty or e.untracked)
              else KIND_EMPTY)
    return e


async def _repo_scan(repos, root: Path, seen: set[str],
                     now: float) -> list[SurveyEntry]:
    """What the worktrees directory cannot show you, asked of the repos.

    Two things live here, and both are invisible to a walk of the directory
    because neither has a directory:

      stale-registration  git still registers it, the directory is gone.
      orphan-branch       a `marvin/*` branch with no registration at all —
                          which is what a branch becomes the moment its
                          worktree is removed and `git branch -d` refuses.
                          Without this it would never be surfaced again.

    Only for repos we can still reach — the ones the surviving worktrees point
    at, plus whatever the fleet still remembers. A repo whose every worktree
    directory has been deleted is unreachable from here, and the survey says
    nothing about it rather than guessing."""
    stale: list[SurveyEntry] = []
    orphans: list[SurveyEntry] = []
    for repo in sorted({str(r) for r in repos if r}):
        try:
            listing = await _git(Path(repo), "worktree", "list", "--porcelain")
        except (WorktreeError, asyncio.TimeoutError, OSError):
            continue
        registered: set[str] = set()
        path = branch = ""
        for line in listing.splitlines() + [""]:
            if line.startswith("worktree "):
                path, branch = line[9:], ""
            elif line.startswith("branch "):
                branch = line[7:].removeprefix("refs/heads/")
            elif not line.strip() and path:
                registered.add(branch)
                try:
                    gone = not Path(path).exists()
                except OSError:
                    gone = False
                inside = root == Path(path).parent or root in Path(path).parents
                if gone and inside and path not in seen:
                    seen.add(path)
                    stale.append(SurveyEntry(
                        path=path, repo=repo, branch=branch,
                        kind=KIND_STALE, age_s=_age_s(branch, path, now)))
                path = branch = ""
        try:
            refs = await _git(Path(repo), "for-each-ref",
                              "--format=%(refname:short)", "refs/heads/marvin/")
        except (WorktreeError, asyncio.TimeoutError, OSError):
            continue
        for name in refs.splitlines():
            # A registered branch is not orphaned even when its directory is
            # gone — the stale entry above is already speaking for it.
            if name and name not in registered:
                orphans.append(SurveyEntry(
                    repo=repo, branch=name, kind=KIND_ORPHAN_BRANCH,
                    age_s=_age_s(name, "", now)))
    return stale + orphans


def _disambiguate(entries: list[SurveyEntry]) -> None:
    """Give every entry a spoken name that points at exactly one of them.

    Two worktrees cut for the same task on the same repo share a directory
    name up to the timestamp, and the timestamp is precisely what speech
    throws away — so they share a label, and the old refusal told Keke to
    "name its branch instead" when the branches were equally indistinguishable
    to the matcher. An ordinal is a discriminator that survives being said out
    loud, and it is assigned HERE, once, so the name the survey speaks and the
    name the instruction may use are the same string.

    Ordered by ident, which is unique and stable, so the same survey always
    numbers the same way."""
    groups: dict[str, list[SurveyEntry]] = {}
    for e in entries:
        groups.setdefault(e.label, []).append(e)
    for base, group in groups.items():
        if len(group) < 2:
            continue
        for i, e in enumerate(sorted(group, key=lambda x: x.ident)):
            tail = _ORDINALS[i] if i < len(_ORDINALS) else str(i + 1)
            e.alias = f"{base} {tail}"


async def survey(worktrees_dir, *, live_paths=(), repos=(),
                 projects=None, now=None) -> list[SurveyEntry]:
    """Walk the worktrees directory and classify everything in it.

    Pure observation — this function removes nothing and mutates nothing, so it
    is safe to call from an HTTP GET on every page load."""
    now = time.time() if now is None else now
    root = Path(worktrees_dir)
    live = _Live(live_paths)
    names = dict(projects or {})
    entries: list[SurveyEntry] = []
    try:
        children = sorted(p for p in root.iterdir())
    except (OSError, NotADirectoryError):
        children = []
    known_repos = {str(r) for r in repos if r}
    for child in children:
        try:
            is_dir = child.is_dir()
        except OSError:
            is_dir = False
        if not is_dir:
            # A stray file, a symlink to one, a dangling symlink. NOT skipped:
            # silence about a thing you cannot classify is worse than naming
            # it, and a directory nobody can account for is exactly what this
            # survey exists to put on the screen. `unrecognized` is in no
            # removable bucket, so reporting it costs nothing.
            odd = SurveyEntry(path=str(child), age_s=_age_s("", str(child), now))
            try:
                odd.note = ("a symlink, not a worktree" if child.is_symlink()
                            else "not a directory")
            except OSError:
                odd.note = "not a directory I can read"
            entries.append(odd)
            continue
        e = await _classify(child, live, now)
        if e.repo:
            known_repos.add(e.repo)
        e.project = names.get(e.path) or names.get(str(child)) or ""
        entries.append(e)
    seen = {e.path for e in entries}
    try:
        root_resolved = root.resolve()
    except (OSError, RuntimeError, ValueError):
        root_resolved = root
    entries.extend(await _repo_scan(known_repos, root_resolved, seen, now))
    entries.sort(key=lambda e: (e.kind, e.path, e.branch))
    _disambiguate(entries)
    return entries


# --------------------------------------------------------------- speech -----
def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _loss(e: SurveyEntry) -> str:
    """What removing this one would cost, in words. Never "nothing" for a
    holds-work entry — if we cannot name the loss we do not offer the removal."""
    bits = []
    if e.ahead:
        bits.append(_plural(e.ahead, "commit"))
    if e.dirty:
        bits.append(_plural(e.dirty, "modified file"))
    if e.untracked:
        bits.append(_plural(e.untracked, "untracked file"))
    return " and ".join(bits) if bits else "nothing I can see"


def _holding(e: SurveyEntry) -> str:
    """What a worktree currently CONTAINS, for one that is not being offered.
    Same facts as _loss, phrased as a state rather than as a cost."""
    if not (e.ahead or e.dirty or e.untracked):
        return "nothing yet"
    return _loss(e)


def _join(names: list[str]) -> str:
    """'a', 'a and b', 'a, b and c' — spoken punctuation."""
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def spoken_report(entries: list[SurveyEntry]) -> str:
    """The sentence Keke hears. Everything a later "yes" may act on has to be
    IN here — that is the whole correlation rule."""
    if not entries:
        return "Nothing has piled up, sir — there are no worktrees."
    buckets: dict[str, list[SurveyEntry]] = {}
    for e in entries:
        buckets.setdefault(e.kind, []).append(e)
    live = buckets.get(KIND_LIVE, [])
    holds = buckets.get(KIND_HOLDS_WORK, [])
    empty = buckets.get(KIND_EMPTY, [])
    stale = buckets.get(KIND_STALE, [])
    orphans = buckets.get(KIND_ORPHAN_BRANCH, [])
    odd = buckets.get(KIND_UNRECOGNIZED, [])
    # An orphan branch is not a worktree and must not be counted as one; it
    # gets its own sentence below.
    found = [e for e in entries if e.kind != KIND_ORPHAN_BRANCH]
    parts = [f"Sir, I found {_plural(len(found), 'worktree')}." if found else
             "Sir, there are no worktrees left."]
    if live:
        # What a live worker HOLDS, not merely that it is live. _classify
        # gathers these facts for live entries precisely so this line can say
        # them: "I won't touch it" with no content reads as "there is nothing
        # there", which is the opposite of why it is being left alone.
        shown = live[:SPEAK_ITEM_LIMIT]
        detail = "; ".join(f"{e.label}, holding {_holding(e)}" for e in shown)
        more = ("" if len(live) <= SPEAK_ITEM_LIMIT
                else f", and {len(live) - len(shown)} more on screen")
        lead = ("One still belongs to a session" if len(live) == 1 else
                f"{len(live)} still belong to sessions")
        parts.append(f"{lead}: {detail}{more} — I won't touch "
                     f"{'it' if len(live) == 1 else 'those'}.")
    if holds:
        shown = holds[:SPEAK_ITEM_LIMIT]
        detail = "; ".join(f"{e.label}, {_loss(e)}" for e in shown)
        more = ("" if len(holds) <= SPEAK_ITEM_LIMIT
                else f", and {len(holds) - len(shown)} more on screen")
        lead = "One holds work" if len(holds) == 1 else f"{len(holds)} hold work"
        parts.append(f"{lead}: {detail}{more}.")
    if empty:
        parts.append("One did nothing worth keeping." if len(empty) == 1 else
                     f"{len(empty)} did nothing worth keeping.")
    if stale:
        parts.append("One registration points at a directory that's already "
                     "gone." if len(stale) == 1 else
                     f"{len(stale)} registrations point at directories that "
                     f"are already gone.")
    if orphans:
        parts.append("One marvin branch has no worktree left — it's on screen "
                     "and I'm keeping it." if len(orphans) == 1 else
                     f"{len(orphans)} marvin branches have no worktrees left "
                     f"— they're on screen and I'm keeping them.")
    if odd:
        parts.append("One isn't mine — it's on screen and I'll leave it alone."
                     if len(odd) == 1 else
                     f"{len(odd)} aren't mine — they're on screen and I'll "
                     f"leave them alone.")
    if empty or stale:
        clears = []
        if empty:
            clears.append("the empty one" if len(empty) == 1
                          else f"the {len(empty)} empty ones")
        if stale:
            clears.append("prune the stale registration" if len(stale) == 1
                          else f"prune the {len(stale)} stale registrations")
        parts.append(f"Say 'remove the empty worktrees' and I'll clear "
                     f"{' and '.join(clears)} — nothing that holds work.")
    if holds:
        parts.append("For one that holds work, say 'remove the worktree for' "
                     "and its name.")
    if not (empty or stale or holds) and found:
        parts.append("Nothing here is mine to clear.")
    return " ".join(parts)


# ------------------------------------------------------------- the gate -----
@dataclass
class _Offer:
    """What a spoken survey authorises, and nothing beyond it.

    `empty` is the exact set of paths the sentence called empty; `entries` is
    every entry it mentioned at all, keyed by `ident`, so a per-item
    instruction can refuse a live one BY NAME instead of pretending not to
    know it."""
    at: float = 0.0
    empty: frozenset[str] = frozenset()
    entries: dict[str, SurveyEntry] = field(default_factory=dict)
    stale: frozenset[str] = frozenset()


class WorktreeCleanup:
    """The consent gate for worktree removal.

    IT IS NOT A YES/NO GATE, and that is the design. This codebase has shipped
    six fail-opens on consent paths, every one of them a bare affirmation
    landing on something it was never meant to answer. Three questions can
    already be pending at once — the onboarding repo confirm, the finance
    source confirm, and a fleet tool approval — and all three resolve on a
    yes-shaped utterance. A fourth yes-gate would have to be arbitrated against
    those three, and arbitration is exactly where the previous six went wrong.

    So this gate never accepts an affirmation at all. Removal is reached only
    by a destructive VERB ("remove the empty worktrees", "remove the worktree
    for X") that no affirmation vocabulary can produce and that none of the
    other three gates recognises. It cannot steal their yes and they cannot
    steal its verb; a yes said anywhere in Marvin removes nothing, ever.

    The correlation rule survives intact: an instruction may only act on what
    the survey SPOKE (the offer), the offer expires, it is redeemable once, and
    every entry is re-surveyed at removal time — so a worktree that gained work
    or gained a worker between the sentence and the instruction is skipped and
    said out loud.

    Every method returns a SENTENCE. Nothing here raises into the brain."""

    def __init__(self, *, bus, fleet, now=time.time):
        self._bus = bus
        self._fleet = fleet
        self._now = now
        self._offer: _Offer | None = None

    # ---------- observation ----------
    async def entries(self) -> list[SurveyEntry]:
        """The survey, for the console. Arms NOTHING: a page render is not a
        sentence Keke heard, and only a heard sentence may authorise anything.

        Deliberately UNGUARDED around the live-paths read, and every caller
        treats a raise here as "I could not survey". A survey that could not
        learn which worktrees belong to running workers would classify a live
        worker's checkout as empty — the one mistake this module exists to
        avoid. Failing loudly costs a spoken apology; failing soft costs a
        session. Only the cosmetic project labels are optional."""
        live = set(self._fleet.live_worktree_paths())
        repos = set(self._fleet.known_repos())
        try:
            names = dict(self._fleet.worktree_projects())
        except Exception:  # noqa: BLE001 — labels are decoration, not safety
            names = {}
        return await survey(self._fleet.worktrees_dir, live_paths=live,
                            repos=repos, projects=names, now=self._now())

    async def report(self) -> str:
        """The voice verb: survey, publish, speak — and ARM the offer.

        Never acts. The utterance that asked for a survey gets a survey."""
        try:
            entries = await self.entries()
        except Exception as e:  # noqa: BLE001 — every failure is spoken, never raised
            self._publish_error(f"worktree survey failed: {e}")
            return ("Sir, I couldn't read your worktrees just now — details "
                    "are on screen.")
        try:
            self._bus.publish("worktrees.survey",
                              {"worktrees": [asdict(x) for x in entries]})
        except Exception:  # noqa: BLE001 — the console loses a render, not the sentence
            pass
        self._offer = _Offer(
            at=self._now(),
            empty=frozenset(e.path for e in entries if e.kind == KIND_EMPTY),
            stale=frozenset(e.path for e in entries if e.kind == KIND_STALE),
            entries={e.ident: e for e in entries})
        return spoken_report(entries)

    # ---------- removal ----------
    async def remove_empty(self) -> str:
        """The batch. Removes ONLY worktrees this gate called empty out loud
        AND that are still empty, still not live, and still inside the
        configured directory. Consumes the offer either way."""
        offer, refusal = self._peek_offer()
        if offer is None:
            return refusal
        self._consume_offer()      # the batch redeems the survey, act or not
        try:
            current = {e.ident: e for e in await self.entries()}
        except Exception as e:  # noqa: BLE001
            self._publish_error(f"worktree survey failed: {e}")
            return ("Sir, I couldn't re-check your worktrees, so I removed "
                    "nothing — details are on screen.")
        removed, kept_branch, changed, failed = [], [], [], []
        for path in sorted(offer.empty):
            now_entry = current.get(path)
            if now_entry is None or now_entry.kind != KIND_EMPTY:
                changed.append(offer.entries[path].label)
                continue
            ok, why, branch_gone = await self._remove(now_entry)
            if not ok:
                failed.append(now_entry.label)
                self._publish_error(why)
                continue
            removed.append(now_entry.label)
            if not branch_gone:
                # git kept it. Say so — and the survey will keep saying so,
                # because it is an orphan branch from here on.
                kept_branch.append(now_entry.branch)
        pruned = await self._prune(offer)
        return self._batch_sentence(removed, kept_branch, changed, failed,
                                    pruned)

    async def remove_named(self, spoken_name: str) -> str:
        """Per-item. One worktree, named, whose loss the survey already read
        aloud. The match must explain EVERY word said — the whitelist rule
        router._approval_vocabulary enforces for tool approvals.

        A refusal that removed NOTHING leaves the offer standing. Consuming it
        on the way in made the advice in every refusal unfollowable: the
        answer to "name it more precisely" was "I haven't gone through your
        worktrees yet". The one-shot rule is about REMOVALS — at most one per
        spoken survey — and that is where the offer is now spent."""
        offer, refusal = self._peek_offer()
        if offer is None:
            return refusal
        said = (spoken_name or "").strip()
        if not said:
            return "Which worktree, sir?"
        matched = [e for e in offer.entries.values() if _explains(e, said)]
        if not matched:
            return (f"I don't have a worktree called {said} in what I just "
                    f"read you, sir.")
        if len(matched) > 1:
            # Refusing is right; "name its branch instead" was not, because
            # the branches of two worktrees cut for one task differ only in
            # digits. _disambiguate already gave each of these a name the
            # survey SPOKE and this matcher can tell apart — offer those.
            shown = sorted(m.label for m in matched)[:SPEAK_ITEM_LIMIT]
            more = ("" if len(matched) <= SPEAK_ITEM_LIMIT
                    else f", and {len(matched) - len(shown)} more on screen")
            return (f"More than one of those matches {said}, sir — I have "
                    f"{_join(shown)}{more}. Say one of those exactly.")
        target = matched[0]
        if target.kind == KIND_LIVE:
            return (f"{target.label} still belongs to a running session, sir "
                    f"— stop it or close its terminal first.")
        if target.kind == KIND_STALE:
            return (f"{target.label}'s directory is already gone, sir — its "
                    f"branch is all that's left, and I'm keeping that.")
        if target.kind == KIND_ORPHAN_BRANCH:
            return (f"{target.label} is only a branch now, sir — its worktree "
                    f"is gone and the branch is the record, so I'm keeping it.")
        if not target.removable:
            return (f"I didn't create {target.label}, sir — I won't remove it.")
        try:
            current = {e.ident: e for e in await self.entries()}
        except Exception as e:  # noqa: BLE001
            self._publish_error(f"worktree survey failed: {e}")
            return ("Sir, I couldn't re-check that worktree, so I removed "
                    "nothing — details are on screen.")
        fresh = current.get(target.ident)
        # The re-check is against the LOSS THAT WAS SPOKEN, not against the
        # bucket. An entry read out as "1 commit" that picks up untracked
        # files is still holds-work, and removing it destroys files that were
        # never in the sentence Keke answered — one yes may never cost more
        # than it described, so any drift in the counts refuses.
        if (fresh is None or fresh.kind != target.kind
                or (fresh.ahead, fresh.dirty, fresh.untracked)
                != (target.ahead, target.dirty, target.untracked)):
            return (f"{target.label} has changed since I read it to you, sir "
                    f"— ask me to go through the worktrees again.")
        self._consume_offer()      # from here on, something is being removed
        ok, why, branch_gone = await self._remove(fresh)
        if not ok:
            self._publish_error(why)
            return (f"I couldn't remove {target.label}, sir — it's still "
                    f"there and details are on screen.")
        if fresh.kind == KIND_EMPTY:
            if branch_gone:
                return (f"Removed {target.label}, sir. It held nothing, so "
                        f"its branch went with it.")
            return (f"Removed {target.label}, sir. It held nothing, but git "
                    f"wouldn't delete branch {fresh.branch}, so I've kept "
                    f"that — it's on screen.")
        return (f"Removed {target.label}, sir. It held {_loss(fresh)} — that "
                f"work is still on branch {fresh.branch}, which I've kept.")

    # ---------- internals ----------
    def _publish_error(self, reason: str) -> None:
        try:
            self._bus.publish("fleet.error", {"reason": reason})
        except Exception:  # noqa: BLE001 — the bus is the thing that just failed
            pass

    def _peek_offer(self) -> tuple[_Offer | None, str]:
        """The standing offer, or why there isn't one. Spends nothing.

        An offer that has aged out is not the description Keke is answering,
        so it is dropped here rather than left to be found again later."""
        offer = self._offer
        if offer is None:
            return None, ("I haven't gone through your worktrees yet, sir — "
                          "ask me to tidy up the worktrees and I'll tell you "
                          "what's there first.")
        if self._now() - offer.at > OFFER_TTL_S:
            self._offer = None
            return None, ("That survey is stale now, sir — ask me to go "
                          "through the worktrees again.")
        return offer, ""

    def _consume_offer(self) -> None:
        """Spend the survey. One-shot: a second REMOVAL cannot ride one
        spoken sentence. Called at the point of action, never on a refusal
        that removed nothing — a refusal Keke is meant to answer must leave
        something for the answer to land on."""
        self._offer = None

    async def _remove(self, entry: SurveyEntry) -> tuple[bool, str, bool]:
        """The ONE destructive call, and every gate in front of it.

        Returns (removed, why-not, branch_deleted). The third value is an
        OBSERVATION, not an intention: git is allowed to refuse the branch, and
        the sentence that follows has to be phrased from what it actually did.

        Containment is checked here rather than only at the survey, because
        this is the last place before deletion: the offer is a dict a bug (or a
        future caller) could put anything into, and "never remove anything
        outside the configured worktrees directory" has to be true of the code
        that removes, not of the code that looked.

        Everything after that is `remove_worktree`'s own guard, reused rather
        than reimplemented — the marvin/ namespace check, the absolute-path
        check, and the branch-is-checked-out-there check are four proven
        failures' worth of reasoning and this module adds nothing to them."""
        if not entry.removable:
            return False, f"refusing to remove {entry.path}: it is {entry.kind}", False
        try:
            root = Path(self._fleet.worktrees_dir).resolve()
            target = Path(entry.path)
            resolved = target.resolve()
        except (OSError, RuntimeError, ValueError, AttributeError) as e:
            return False, f"refusing to remove {entry.path}: unreadable path ({e})", False
        if not target.is_absolute():
            return False, f"refusing to remove {entry.path!r}: not an absolute path", False
        if root not in resolved.parents:
            return False, (f"refusing to remove {entry.path}: it is outside "
                           f"{root}, which is the only directory I clean"), False
        record = Worktree(repo=entry.repo, path=str(target),
                          branch=entry.branch, base_commit=entry.base_commit)
        try:
            await remove_worktree(record)
        except (WorktreeError, asyncio.TimeoutError, OSError) as e:
            return False, f"could not remove {entry.path}: {e}", False
        branch_deleted = False
        if entry.kind == KIND_EMPTY:
            # The branch is the record — but an empty worktree's branch has no
            # commits beyond base, so there is no record on it. `-d`, never
            # `-D`: git's own merged-ness check is a free second opinion, and
            # when it disagrees with our count we keep the branch.
            #
            # It disagrees more often than it looks: `-d` measures merged-ness
            # against HEAD, so any human who has switched the real checkout to
            # a branch that does not contain the commit this worktree was cut
            # from makes every one of these refuse. That outcome is CAPTURED
            # and returned, because the sentence used to claim the branch went
            # regardless — and once the directory is gone a branch nobody
            # mentioned has nothing left to surface it. `orphan-branch` is the
            # other half of this: it keeps the survivor visible.
            try:
                await _git(Path(entry.repo), "branch", "-d", entry.branch)
                branch_deleted = True
            except (WorktreeError, asyncio.TimeoutError, OSError) as e:
                self._publish_error(f"kept branch {entry.branch}: {e}")
        return True, "", branch_deleted

    async def _prune(self, offer: _Offer) -> int:
        """Clear the stale registrations the survey NAMED, one at a time.

        Not `git worktree prune`: that is a whole-repo sweep, and it would
        also clear registrations this survey never mentioned — including a
        human's own checkout that is only temporarily missing, whose
        registration is exactly what `git worktree repair` needs to put it
        back. The spoken count is the count of NAMED entries, so the action
        has to be the named entries too.

        `git worktree remove` on a registration whose directory is gone
        deletes the administrative file and nothing else: not the branch, not
        a ref, not an object. The directory's absence is RE-CHECKED here
        first, because on a directory that is present the same command would
        delete it — that is a removal nobody consented to, and it is refused
        rather than raced."""
        pruned = 0
        try:
            root = Path(self._fleet.worktrees_dir).resolve()
        except (OSError, RuntimeError, ValueError, AttributeError) as e:
            self._publish_error(f"could not prune: unreadable worktrees dir ({e})")
            return 0
        for path in sorted(offer.stale):
            entry = offer.entries.get(path)
            if entry is None or not entry.repo:
                continue
            target = Path(path)
            try:
                resolved = target.resolve()
                back = target.exists()
            except (OSError, RuntimeError, ValueError) as e:
                self._publish_error(f"could not prune {path}: unreadable ({e})")
                continue
            if not target.is_absolute() or root not in resolved.parents:
                self._publish_error(f"refusing to prune {path}: it is outside "
                                    f"{root}, which is the only directory I clean")
                continue
            if back:
                self._publish_error(f"not pruning {path}: its directory is "
                                    f"back since I read the survey to you")
                continue
            try:
                await _git(Path(entry.repo), "worktree", "remove", "--force",
                           str(target))
                pruned += 1
            except (WorktreeError, asyncio.TimeoutError, OSError) as e:
                self._publish_error(f"could not prune {path}: {e}")
        return pruned

    def _batch_sentence(self, removed, kept_branch, changed, failed,
                        pruned) -> str:
        """Phrased from OUTCOMES. The branch clause in particular is never
        assumed: git is entitled to refuse `-d`, and this sentence used to
        claim the branches went anyway."""
        parts = []
        if removed:
            n, kept = len(removed), len(kept_branch)
            if not kept:
                parts.append("Removed one empty worktree, sir, and its branch."
                             if n == 1 else
                             f"Removed {n} empty worktrees, sir, and their "
                             f"branches.")
            elif kept == n:
                parts.append(f"Removed one empty worktree, sir, but git "
                             f"wouldn't delete branch {kept_branch[0]}, so "
                             f"I've kept it." if n == 1 else
                             f"Removed {n} empty worktrees, sir, but git "
                             f"wouldn't delete their branches, so I've kept "
                             f"{_join(sorted(kept_branch))}.")
            else:
                parts.append(f"Removed {n} empty worktrees, sir, and "
                             f"{n - kept} of their branches; git wouldn't "
                             f"delete {_join(sorted(kept_branch))}, so I've "
                             f"kept {'that' if kept == 1 else 'those'}.")
            if kept:
                parts.append("They're on screen." if kept > 1
                             else "It's on screen.")
        if pruned:
            parts.append("Pruned one stale registration." if pruned == 1 else
                         f"Pruned {pruned} stale registrations.")
        if not removed and not pruned:
            parts.append("I removed nothing, sir.")
        if changed:
            parts.append(f"I left {', '.join(changed)} alone — "
                         f"{'it has' if len(changed) == 1 else 'they have'} "
                         f"changed since I read the survey to you.")
        if failed:
            parts.append(f"I couldn't remove {', '.join(failed)}; "
                         f"details are on screen.")
        parts.append("Nothing holding work was touched.")
        return " ".join(parts)


def _explains(entry: SurveyEntry, said: str) -> bool:
    """True when `entry` accounts for EVERY word of `said`.

    THE WHITELIST, borrowed wholesale from router._approval_vocabulary and for
    the same reason: matching one word of a sentence says nothing about the
    rest of it. "remove the worktree for soccer and everything older than a
    week" mentions soccer, and a substring match would have removed soccer and
    silently ignored the rest of the instruction. A word explained by nothing
    here is a word this code did not understand, and consent is never inferred
    from that. Degrades safe: an unknown politeness costs one clarification, an
    unknown destructive clause costs nothing at all.

    Explaining every word is necessary and NOT sufficient. The generic half of
    the vocabulary — politeness, plus the word "worktree" itself — belongs to
    every entry equally, so "remove the worktree for the", a plausible STT
    truncation, used to tokenize to ["the"], match every entry alike, and
    remove the only one whenever the survey held exactly one. So at least one
    word must come from THIS entry's own identity: a name that names nothing
    in particular names nothing at all."""
    generic = set(_POLITE) | {"worktree", "worktrees"}
    identity: set[str] = set()
    # Deliberately NOT the base commit: a sha contributes stray one- and
    # two-letter tokens ("a", "b", "ed") that widen the whitelist for nothing
    # — nobody names a worktree by its sha out loud.
    for text in (entry.project, entry.branch, entry.label,
                 Path(entry.path).name):
        identity.update(_spoken_tokens(text or ""))
    identity -= generic
    spoken = _spoken_tokens(said)
    if not spoken:
        return False
    if not any(w in identity for w in spoken):
        return False
    return all(w in generic or w in identity for w in spoken)
