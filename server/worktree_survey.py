"""Surfacing what the fleet leaves behind — and a CONSENTED way to clear the
part of it that is provably worthless.

Nothing in `server/` has ever removed a worktree, deliberately: `remove_worktree`
is explicit cleanup only, because "the worktree holds the diff a human may still
want to merge back". The consequence is that every task ever run leaves a
directory and a `jarvis/*` branch behind, with nothing to surface it and no way
to clear it. This module is the surfacing, and a consent path narrow enough that
losing a diff Keke wanted is not one of its outcomes.

THIS IS NOT AUTOMATIC DELETION. Nothing here runs on a timer, at boot, or on
shutdown. Every removal is reached by an utterance whose whole job is removal,
and only after a SEPARATE utterance made JARVIS read the survey out loud.

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
                    Git still registers it but the directory is gone.
                    `git worktree prune` clears the administrative file and
                    destroys nothing — not the branch, not any object.

There is a FIFTH bucket, `unrecognized`, for anything in the worktrees
directory that is not a jarvis worktree: a foreign branch, a symlink, a plain
directory, or a checkout git cannot answer questions about. It is REPORTED
(silence about a thing you cannot classify is worse than naming it) and it is
in no removable bucket. `remove_worktree` would refuse it anyway; this makes
the refusal happen before the destructive call rather than inside it.

`ahead` is counted as "commits on this branch that are reachable from NO
non-jarvis ref", which needs no recorded base_commit and degrades in the safe
direction: if the commit this branch was cut from has itself become
unreachable, its ancestors count as ahead too, and the worktree classifies as
holds-work rather than empty.
"""
from __future__ import annotations

import asyncio
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
KIND_UNRECOGNIZED = "unrecognized"

# What a spoken survey may still authorise, and for how long. Same number as
# router.APPROVAL_TTL_S for the same reason: consent to destroy something goes
# stale, and an offer redeemed ten minutes after the sentence was spoken is
# answering a description of a directory that may no longer be true.
OFFER_TTL_S = 600.0

# How many holds-work items get read aloud before the sentence defers to the
# console. Speech is the expensive channel; the console shows all of them.
SPEAK_ITEM_LIMIT = 3

# `jarvis/<slug>-YYYYmmdd-HHMMSS` — create_worktree's own stamp, which dates a
# worktree more honestly than any mtime (a mtime moves when anything reads or
# writes; the stamp is when the worker was spawned).
_STAMP = re.compile(r"-(\d{8}-\d{6})$")


@dataclass
class SurveyEntry:
    """One thing found under the worktrees directory. Everything a human needs
    to decide, and nothing a caller has to re-derive."""
    path: str = ""            # git's own canonical absolute path
    repo: str = ""            # the real checkout it was cut from
    branch: str = ""
    base_commit: str = ""
    kind: str = KIND_UNRECOGNIZED
    ahead: int = 0            # commits reachable from no non-jarvis ref
    dirty: int = 0            # tracked modifications
    untracked: int = 0        # untracked ENTRIES (a directory collapses to one)
    age_s: float = 0.0
    project: str = ""         # spoken name, when a live worker supplies one
    note: str = ""            # why something is unrecognized, when it is

    @property
    def removable(self) -> bool:
        return self.kind in (KIND_EMPTY, KIND_HOLDS_WORK)

    @property
    def label(self) -> str:
        """What to call it out loud.

        The directory name is `<repo>-<slug>-<stamp>`, and the slug is the task
        Keke actually said — the closest thing a worktree has to a human name.
        The stamp is dropped (nobody says "dash two zero two six...") and the
        hyphens become spaces, so the label is speech.

        It must ROUND-TRIP: _explains builds its vocabulary from this same
        directory name, so every label JARVIS offers is a name the per-item
        instruction can match back. Two worktrees cut for the same task on the
        same repo therefore share a label and a voice instruction naming it is
        AMBIGUOUS — refused rather than guessed, with both on screen."""
        if self.project:
            return self.project
        name = _STAMP.sub("", Path(self.path).name or "") or self.branch
        return name.replace("-", " ").replace("_", " ").strip() or self.path


def _age_s(branch: str, path: str, now: float) -> float:
    m = _STAMP.search(branch or "")
    if m:
        try:
            return max(0.0, now - time.mktime(
                time.strptime(m.group(1), "%Y%m%d-%H%M%S")))
        except (ValueError, OverflowError):
            pass
    try:
        return max(0.0, now - Path(path).stat().st_mtime)
    except OSError:
        return 0.0


def _live_forms(live_paths) -> set[str]:
    """Every spelling of a live worktree path we might have to recognise.

    The fleet records `worktrees_dir / name` unresolved; git answers with its
    own canonicalised form. On a machine where the worktrees directory sits
    under a symlink (/tmp on macOS is one) those two strings differ, and this
    is the single most safety-critical comparison in the module — a miss
    classifies a running worker's checkout as removable. So hold BOTH forms and
    compare against both."""
    forms: set[str] = set()
    for p in live_paths or ():
        if not p:
            continue
        forms.add(str(p))
        try:
            forms.add(str(Path(p).resolve()))
        except (OSError, RuntimeError, ValueError):
            pass
    return forms


async def _classify(entry_dir: Path, live: set[str], now: float) -> SurveyEntry:
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
    if str(entry_dir) in live or str(top) in live:
        e.kind = KIND_LIVE
    elif not branch.startswith("jarvis/"):
        e.note = f"branch {branch!r} is outside the jarvis/ namespace"
        return e
    # Facts, gathered for EVERY entry including live ones: the report has to
    # say what a live worker has accumulated too, or "leave it alone" reads as
    # "there is nothing there".
    try:
        ahead = await _git(entry_dir, "rev-list", "HEAD", "--not",
                           "--exclude=jarvis/*", "--branches", "--tags",
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


async def _stale(repos, root: Path, seen: set[str], now: float) -> list[SurveyEntry]:
    """Registrations git still holds whose directory is gone.

    Only for repos we can still reach — the ones the surviving worktrees point
    at, plus whatever the fleet still remembers. A repo whose every worktree
    directory has been deleted is unreachable from here, and the survey says
    nothing about it rather than guessing."""
    out: list[SurveyEntry] = []
    for repo in sorted({str(r) for r in repos if r}):
        try:
            listing = await _git(Path(repo), "worktree", "list", "--porcelain")
        except (WorktreeError, asyncio.TimeoutError, OSError):
            continue
        path = branch = ""
        for line in listing.splitlines() + [""]:
            if line.startswith("worktree "):
                path, branch = line[9:], ""
            elif line.startswith("branch "):
                branch = line[7:].removeprefix("refs/heads/")
            elif not line.strip() and path:
                try:
                    gone = not Path(path).exists()
                except OSError:
                    gone = False
                inside = root == Path(path).parent or root in Path(path).parents
                if gone and inside and path not in seen:
                    seen.add(path)
                    out.append(SurveyEntry(
                        path=path, repo=repo, branch=branch,
                        kind=KIND_STALE, age_s=_age_s(branch, path, now)))
                path = branch = ""
    return out


async def survey(worktrees_dir, *, live_paths=(), repos=(),
                 projects=None, now=None) -> list[SurveyEntry]:
    """Walk the worktrees directory and classify everything in it.

    Pure observation — this function removes nothing and mutates nothing, so it
    is safe to call from an HTTP GET on every page load."""
    now = time.time() if now is None else now
    root = Path(worktrees_dir)
    live = _live_forms(live_paths)
    names = dict(projects or {})
    entries: list[SurveyEntry] = []
    try:
        children = sorted(p for p in root.iterdir())
    except (OSError, NotADirectoryError):
        children = []
    known_repos = {str(r) for r in repos if r}
    for child in children:
        try:
            if not child.is_dir():
                continue
        except OSError:
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
    entries.extend(await _stale(known_repos, root_resolved, seen, now))
    entries.sort(key=lambda e: (e.kind, e.path))
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
    odd = buckets.get(KIND_UNRECOGNIZED, [])
    parts = [f"Sir, I found {_plural(len(entries), 'worktree')}."]
    if live:
        parts.append("One still belongs to a session — I won't touch it."
                     if len(live) == 1 else
                     f"{len(live)} still belong to sessions — I won't touch "
                     f"those.")
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
    if not (empty or stale or holds):
        parts.append("Nothing here is mine to clear.")
    return " ".join(parts)


# ------------------------------------------------------------- the gate -----
@dataclass
class _Offer:
    """What a spoken survey authorises, and nothing beyond it.

    `empty` is the exact set of paths the sentence called empty; `entries` is
    every path it mentioned at all, so a per-item instruction can refuse a live
    one BY NAME instead of pretending not to know it."""
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
    steal its verb; a yes said anywhere in JARVIS removes nothing, ever.

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
            entries={e.path: e for e in entries})
        return spoken_report(entries)

    # ---------- removal ----------
    async def remove_empty(self) -> str:
        """The batch. Removes ONLY worktrees this gate called empty out loud
        AND that are still empty, still not live, and still inside the
        configured directory. Consumes the offer either way."""
        offer, refusal = self._take_offer()
        if offer is None:
            return refusal
        try:
            current = {e.path: e for e in await self.entries()}
        except Exception as e:  # noqa: BLE001
            self._publish_error(f"worktree survey failed: {e}")
            return ("Sir, I couldn't re-check your worktrees, so I removed "
                    "nothing — details are on screen.")
        removed, changed, failed = [], [], []
        for path in sorted(offer.empty):
            now_entry = current.get(path)
            if now_entry is None or now_entry.kind != KIND_EMPTY:
                changed.append(offer.entries[path].label)
                continue
            ok, why = await self._remove(now_entry)
            (removed if ok else failed).append(now_entry.label)
            if not ok:
                self._publish_error(why)
        pruned = await self._prune(offer)
        return self._batch_sentence(removed, changed, failed, pruned)

    async def remove_named(self, spoken_name: str) -> str:
        """Per-item. One worktree, named, whose loss the survey already read
        aloud. The match must explain EVERY word said — the whitelist rule
        router._approval_vocabulary enforces for tool approvals."""
        offer, refusal = self._take_offer()
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
            return (f"More than one of those matches {said}, sir — name its "
                    f"branch instead.")
        target = matched[0]
        if target.kind == KIND_LIVE:
            return (f"{target.label} still belongs to a running session, sir "
                    f"— stop it or close its terminal first.")
        if target.kind == KIND_STALE:
            return (f"{target.label}'s directory is already gone, sir — its "
                    f"branch is all that's left, and I'm keeping that.")
        if not target.removable:
            return (f"I didn't create {target.label}, sir — I won't remove it.")
        try:
            current = {e.path: e for e in await self.entries()}
        except Exception as e:  # noqa: BLE001
            self._publish_error(f"worktree survey failed: {e}")
            return ("Sir, I couldn't re-check that worktree, so I removed "
                    "nothing — details are on screen.")
        fresh = current.get(target.path)
        if fresh is None or fresh.kind != target.kind:
            return (f"{target.label} has changed since I read it to you, sir "
                    f"— ask me to go through the worktrees again.")
        ok, why = await self._remove(fresh)
        if not ok:
            self._publish_error(why)
            return (f"I couldn't remove {fresh.label}, sir — it's still there "
                    f"and details are on screen.")
        if fresh.kind == KIND_EMPTY:
            return (f"Removed {fresh.label}, sir. It held nothing, so its "
                    f"branch went with it.")
        return (f"Removed {fresh.label}, sir. It held {_loss(fresh)} — that "
                f"work is still on branch {fresh.branch}, which I've kept.")

    # ---------- internals ----------
    def _publish_error(self, reason: str) -> None:
        try:
            self._bus.publish("fleet.error", {"reason": reason})
        except Exception:  # noqa: BLE001 — the bus is the thing that just failed
            pass

    def _take_offer(self) -> tuple[_Offer | None, str]:
        """Consume the standing offer, or say why there isn't one.

        One-shot: a second instruction cannot ride the same sentence, and an
        offer that has aged out is not the description Keke is answering."""
        offer, self._offer = self._offer, None
        if offer is None:
            return None, ("I haven't gone through your worktrees yet, sir — "
                          "ask me to tidy up the worktrees and I'll tell you "
                          "what's there first.")
        if self._now() - offer.at > OFFER_TTL_S:
            return None, ("That survey is stale now, sir — ask me to go "
                          "through the worktrees again.")
        return offer, ""

    async def _remove(self, entry: SurveyEntry) -> tuple[bool, str]:
        """The ONE destructive call, and every gate in front of it.

        Containment is checked here rather than only at the survey, because
        this is the last place before deletion: the offer is a dict a bug (or a
        future caller) could put anything into, and "never remove anything
        outside the configured worktrees directory" has to be true of the code
        that removes, not of the code that looked.

        Everything after that is `remove_worktree`'s own guard, reused rather
        than reimplemented — the jarvis/ namespace check, the absolute-path
        check, and the branch-is-checked-out-there check are four proven
        failures' worth of reasoning and this module adds nothing to them."""
        if not entry.removable:
            return False, f"refusing to remove {entry.path}: it is {entry.kind}"
        try:
            root = Path(self._fleet.worktrees_dir).resolve()
            target = Path(entry.path)
            resolved = target.resolve()
        except (OSError, RuntimeError, ValueError, AttributeError) as e:
            return False, f"refusing to remove {entry.path}: unreadable path ({e})"
        if not target.is_absolute():
            return False, f"refusing to remove {entry.path!r}: not an absolute path"
        if root not in resolved.parents:
            return False, (f"refusing to remove {entry.path}: it is outside "
                           f"{root}, which is the only directory I clean")
        record = Worktree(repo=entry.repo, path=str(target),
                          branch=entry.branch, base_commit=entry.base_commit)
        try:
            await remove_worktree(record)
        except (WorktreeError, asyncio.TimeoutError, OSError) as e:
            return False, f"could not remove {entry.path}: {e}"
        if entry.kind == KIND_EMPTY:
            # The branch is the record — but an empty worktree's branch has no
            # commits beyond base, so there is no record on it. `-d`, never
            # `-D`: git's own merged-ness check is a free second opinion, and
            # if it disagrees with our count we keep the branch and say
            # nothing more. A kept branch costs a line in `git branch`; a
            # deleted one with commits on it costs the commits.
            try:
                await _git(Path(entry.repo), "branch", "-d", entry.branch)
            except (WorktreeError, asyncio.TimeoutError, OSError):
                pass
        return True, ""

    async def _prune(self, offer: _Offer) -> int:
        """`git worktree prune` on the repos whose stale registrations the
        survey actually named. Destroys nothing: it deletes the administrative
        file for a directory that is already gone, and never a branch, a ref or
        an object — so for a worktree that held work, the branch that is now
        its only trace survives this untouched.

        Counted per REPO, not per fleet: reporting every stale entry as pruned
        because one repo's prune succeeded would be a spoken claim about repos
        this never reached."""
        by_repo: dict[str, int] = {}
        for path in offer.stale:
            entry = offer.entries.get(path)
            if entry is not None and entry.repo:
                by_repo[entry.repo] = by_repo.get(entry.repo, 0) + 1
        pruned = 0
        for repo, count in sorted(by_repo.items()):
            try:
                await _git(Path(repo), "worktree", "prune")
                pruned += count
            except (WorktreeError, asyncio.TimeoutError, OSError) as e:
                self._publish_error(f"could not prune {repo}: {e}")
        return pruned

    def _batch_sentence(self, removed, changed, failed, pruned) -> str:
        parts = []
        if removed:
            parts.append("Removed one empty worktree, sir, and its branch."
                         if len(removed) == 1 else
                         f"Removed {len(removed)} empty worktrees, sir, and "
                         f"their branches.")
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
    unknown destructive clause costs nothing at all."""
    vocabulary = set(_POLITE)
    vocabulary.add("worktree")
    vocabulary.add("worktrees")
    # Deliberately NOT the base commit: _words strips digits, so a sha
    # contributes stray one- and two-letter tokens ("a", "b", "ed") that widen
    # the whitelist for nothing — nobody names a worktree by its sha out loud.
    for text in (entry.project, entry.branch, entry.label,
                 Path(entry.path).name):
        vocabulary.update(_words(text or ""))
    spoken = _words(said)
    if not spoken:
        return False
    return all(w in vocabulary for w in spoken)
