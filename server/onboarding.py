"""The confirmation beat: Marvin proposes a repo, Keke confirms it out loud.

Discovery is a guess. Nothing becomes usable — least of all the finance repo —
until a human says yes, so this module is the only path to `confirmed`.

Every registry mutation here is keyed by PATH, not name: two different
directories on this machine are both named `jarvis` (a directory basename, not
the assistant's name — the repo directory has not moved yet), and a name-keyed
call resolves to whichever comes first. We hold the exact Project we asked about,
so we always know its path precisely.
"""
from __future__ import annotations

import re
from pathlib import Path

from server.discovery import discover
from server.registry import Project, Registry
from server.router import is_addressed

# A correction must show POSITIVE evidence: an explicit connector between the
# negation and the new name. Without one ("No.", "no thanks", "no way"), the
# reply is a rejection — never a name. Speech-to-text loves trailing "."/"!",
# so leftover text after "no" must never be treated as a name by default.
_CORRECTION = re.compile(
    r"^\s*(?:no|nope)\b[,\s]*(?:it'?s|it\s+is|that'?s|that\s+is|i\s+said|call\s+it)\s+(?P<name>.+?)\s*$",
    re.I)
# `right`/`correct` only count as a yes when they stand alone or follow
# "that's" — bare `right\b` would confirm on filler like "right, hmm".
_YES = re.compile(
    r"^\s*(?:(?:yes|yeah|yep|yup|confirm|sure|ok|okay)\b"
    r"|go\s+ahead\b|do\s+it\b"
    r"|that'?s\s+(?:it|right|correct)\b"
    r"|(?:right|correct)\s*[.!?]*\s*$)", re.I)
_NO = re.compile(r"^\s*(?:no|nope|skip|not\s+(?:that|it)|wrong)\b", re.I)
# A negation ANYWHERE in the utterance disqualifies a yes. Spoken rejections
# very often open affirmatively — "Yeah, no, that's not right", "yes, but not
# that one" — and _YES anchors on that opener, so prefix order alone cannot
# save us. Confirmation is consent; it must never ride on the first word.
_NEGATION = re.compile(r"\b(?:no|nope|not|don'?t|wrong|isn'?t)\b", re.I)
# Punctuation/space between stacked affirmation phrases ("yes, that's right").
_SEPARATORS = " \t,.;:!?"

# How many repos one run may propose before closing with an honest "N more
# waiting — say find my projects to continue". Five yes/no questions is about
# thirty seconds of spoken dialogue — enough to cover the handful of repos a
# person is actually working in, and small enough that the observed first run
# (69 discovered repos) is a bounded conversation instead of an ordeal. A run
# starts at refresh(): boot discovery and the "find my projects" verb are the
# only two entrances, so each explicit ask buys a fresh budget.
PROPOSALS_PER_RUN = 5


def _bare_affirmation(text: str) -> bool:
    """True when the utterance is an affirmation and NOTHING else — the only
    shape allowed to confirm a repo.

    _YES is prefix-anchored, so "sure, stop soccer" matches it while carrying
    a stop COMMAND. Consume every leading _YES phrase (spoken confirmations
    stack them: "yes, that's right"), then apply the router's rule to the
    remainder: naming anything beyond the bare yes/no stop-words is positive
    evidence the speaker is NOT answering the pending question (the same
    principle as router.bare_yes_no). _YES itself supplies the affirmation
    vocabulary; the router supplies the stop-words — each defined once."""
    rest = text
    while True:
        m = _YES.match(rest)
        if m is None:
            break
        rest = rest[m.end():].lstrip(_SEPARATORS)
    return not is_addressed(rest)


class Onboarding:
    def __init__(self, bus, registry: Registry, registry_path: Path):
        self.bus = bus
        self.registry = registry
        self.path = Path(registry_path)
        self._asking: Project | None = None
        # Set TRUE only after the brain's confirm.request _speak await returns —
        # i.e. only once THIS repo question has actually been read to Keke. The
        # exact parallel of Approval.spoken for the fleet's approvals, and for
        # the same reason: ask_next() sets `awaiting` synchronously but appends
        # confirm.request to the TAIL of the bus queue, so an utterance already
        # sitting ahead of it (an impatient "yeah, go ahead") lands with
        # awaiting=True before the question was ever spoken. handle_reply
        # refuses to resolve until this is set, so consent can never precede the
        # question — the barrier the confirm.next indirection was meant to be.
        self._asking_spoken: bool = False
        self._rejected: set[str] = set()  # paths, not names — names can collide
        # Floor control (2026-08-12). Three facts, stored separately because
        # they answer different questions and fail differently:
        #   _paused — the owner said "stop asking". Sticky: only refresh()
        #     (boot discovery / "find my projects") lifts it, so nothing this
        #     module does on its own can resume a chain the owner put down.
        #   _yielded — the chain was displaced by a tool approval (or held
        #     while one was pending) and should resume once approvals clear.
        #     Set ONLY when the chain was actually in motion, so a dormant
        #     chain can never start asking unprompted after an approval.
        #   _asked_this_run — proposals made since the last refresh(), the
        #     counter behind PROPOSALS_PER_RUN.
        # None of this touches _rejected or the registry: pausing/yielding
        # discards no candidate, ever — they stay pending() and re-askable.
        self._paused: bool = False
        self._yielded: bool = False
        self._asked_this_run: int = 0

    @property
    def awaiting(self) -> bool:
        """True while a spoken confirmation is pending. The brain uses this to
        make the pending question OWN every bare yes/no-shaped utterance."""
        return self._asking is not None

    def mark_spoken(self) -> bool:
        """Record that the pending repo question was READ ALOUD. Called by the
        brain only after the confirm.request readback's await returns — never on
        the strength of having published it. Returns False when nothing is
        pending (already resolved, or never asked). The mirror of
        Router.mark_spoken for approvals."""
        if self._asking is None:
            return False
        self._asking_spoken = True
        return True

    def is_asking(self, path: str) -> bool:
        """True when the question for exactly `path` is the one on the floor.
        The brain's staleness check before voicing a confirm.request: a
        question displaced (pause/yield) while its readback was still queued
        must never be spoken — a voiced question that nothing owns invites a
        yes that lands somewhere else. The mirror of the nonce verification
        the approval readback already does."""
        return self._asking is not None and self._asking.path == path

    @property
    def suspended(self) -> bool:
        """True when the chain was displaced by an approval and should resume
        once approvals clear. An explicit pause always wins — the owner said
        stop, and no approval outcome may overrule that — and an empty
        candidate list means there is nothing to resume to."""
        return self._yielded and not self._paused and bool(self._candidates())

    def pause(self) -> str:
        """The owner's way out: put the chain down. PAUSES, never rejects —
        every candidate (including the one mid-question) stays pending and
        untouched in the registry, so nothing is discarded and refresh()
        ("find my projects") re-proposes them. Returns the spoken close: how
        many remain and how to continue, composed here so the count and the
        wording can never drift apart. Clearing _asking makes any queued
        confirm.request for it stale (see is_asking) and hands bare yes/no
        back to whoever else is waiting on one."""
        self._paused = True
        self._yielded = False
        self._asking = None
        self._asking_spoken = False
        n = len(self._candidates())
        self._publish_counts()
        if not n:
            return "Understood, sir."
        return (f"Understood, sir — {n} repo{'' if n == 1 else 's'} still "
                f"waiting. Say find my projects when you want to continue.")

    def yield_floor(self) -> bool:
        """A tool approval is about to be read aloud: the repo question gives
        up the floor. The approval blocks a real worker and expires in 600
        seconds; the repo confirm blocks nothing and can wait — and two open
        questions competing for one bare yes is exactly how the owner's answer
        to one resolves the other.

        Returns True only when a question the owner actually HEARD was
        displaced — the brain uses that to announce the switch and to treat
        the next bare yes as ambiguous (one clarifying beat, fail closed). A
        question asked but never voiced displaces silently: the owner heard
        nothing, so nothing is ambiguous. Either way the candidate stays
        pending and its budget slot is returned (it was counted at proposal
        but never answered), and _yielded arms the resume. With no question
        open this is a no-op that arms nothing: a dormant chain must never
        start asking unprompted after an approval clears."""
        if self._asking is None:
            return False
        self._asked_this_run = max(0, self._asked_this_run - 1)
        displaced_spoken = self._asking_spoken
        self._yielded = True
        self._asking = None
        self._asking_spoken = False
        return displaced_spoken

    def hold(self) -> None:
        """A confirm.next arrived while an approval was pending: do not put a
        repo question on the floor, but remember the chain was in motion so it
        resumes when approvals clear. Candidates-only guard for the same
        reason yield_floor checks _asking: an empty chain has nothing to
        resume to, and arming _yielded anyway would make the eventual
        confirm.next speak a pointless closing line."""
        if self._candidates():
            self._yielded = True

    def closing_line(self) -> str | None:
        """What to say when ask_next() declined to propose. None means say
        nothing (an explicit pause already spoke its own close). Composed
        here, next to the counts it reports, for the same reason pause()
        composes its line: the sentence must never claim a number the state
        doesn't hold."""
        if self._paused:
            return None
        n = len(self._candidates())
        if not n:
            return "Nothing left for me to confirm, sir."
        return (f"That's {self._asked_this_run} for now, sir — {n} more "
                f"waiting. Say find my projects when you want to continue.")

    def _publish_counts(self) -> None:
        self.bus.publish("registry.updated", {
            "confirmed": sum(1 for p in self.registry.projects if p.confirmed),
            "pending": len(self._candidates())})

    def _candidates(self) -> list[Project]:
        return [p for p in self.registry.pending() if p.path not in self._rejected]

    async def refresh(self, home) -> int:
        added = self.registry.merge_candidates(await discover(Path(home)))
        if added:
            self.registry.save(self.path)
        # A refresh IS the start of a run — boot discovery and the "find my
        # projects" verb are its only callers, and both are explicit asks. So
        # it lifts a pause (this is the documented way back in) and grants a
        # fresh PROPOSALS_PER_RUN budget. It still discards nothing:
        # merge_candidates never touches existing entries, and _rejected is
        # deliberately left alone — a repo the owner said no to stays refused
        # for the session.
        self._paused = False
        self._asked_this_run = 0
        self._publish_counts()
        return len(added)

    def next_prompt(self) -> dict | None:
        pending = self._candidates()
        if not pending:
            return None
        p = pending[0]
        return {"name": p.name, "path": p.path,
                "question": f"I found what looks like {p.name} at {p.path}. "
                            f"That's the correct repo, right?"}

    async def ask_next(self) -> bool:
        # Declines, in the order of who they answer to: the owner's pause
        # first (nothing may override it), then the per-run cap. A resumed
        # question — one yield_floor() displaced — spends no new budget: its
        # slot was returned at displacement, so an interruption never shrinks
        # the run. False from any decline leaves the caller to speak
        # closing_line(); the candidates themselves are untouched.
        if self._paused or self._asked_this_run >= PROPOSALS_PER_RUN:
            self._asking = None
            self._asking_spoken = False
            return False
        self._yielded = False          # the resume happened (or a fresh ask did)
        prompt = self.next_prompt()
        if prompt is None:
            self._asking = None
            self._asking_spoken = False
            return False
        self._asked_this_run += 1
        self._asking = next(p for p in self.registry.projects if p.path == prompt["path"])
        # A fresh question, not yet read aloud: the brain marks it spoken only
        # after the confirm.request below has actually been voiced.
        self._asking_spoken = False
        # `kind` tags WHOSE question this is. The §16 data-source gate publishes
        # the same event type for the same console widget, but it speaks its own
        # question in the turn that asked it; the brain reads THIS one aloud and
        # would otherwise read that one a second time — putting a question to
        # Keke whose answer is already pending, which is how consent gets
        # attached to the wrong thing. A tag on the speaker's own question is a
        # whitelist; deciding by what the event is NOT would be a blacklist.
        # Additive: `prompt` itself is unchanged, so next_prompt()'s contract
        # and the console's read of `question`/`name` are untouched.
        self.bus.publish("confirm.request", {**prompt, "kind": "repo"})
        return True

    async def handle_reply(self, spoken: str) -> str:
        if self._asking is None:
            return "ignored"
        if not self._asking_spoken:
            # The question has not been read aloud yet. ask_next() appends
            # confirm.request to the TAIL of the bus queue, so this utterance
            # was queued AHEAD of it and cannot be its answer — resolving now
            # would confirm a repo (maybe the finance one) whose question was
            # never spoken. Refuse and stay pending; the question is voiced next
            # and only then may a yes confirm. handle_reply is the same barrier
            # resolve_approval enforces with "unspoken".
            return "ignored"
        asked, text = self._asking, (spoken or "").strip()
        outcome = "ignored"

        # Negation is evaluated BEFORE affirmation: when both readings are
        # available, the rejecting one must win — a wrong "rejected" costs one
        # re-ask, a wrong "confirmed" hands over a repo (maybe the finance one).
        if _NO.match(text):
            m = _CORRECTION.match(text)
            corrected = (m.group("name").strip() if m else "")
            # A valid correction needs a real name: at least one word character,
            # and not itself another negation ("not right", "not this one").
            if (corrected and re.search(r"\w", corrected)
                    and not _NO.match(corrected)
                    and not re.match(r"not\b", corrected, re.I)):
                # "no, I said the trading system" teaches an alias — but an
                # utterance beginning with "no" is not an explicit yes. The
                # project stays PENDING; the flow re-asks knowing the alias,
                # and only a real "yes" can confirm it.
                self.registry.add_alias_path(asked.path, corrected)
                outcome = "renamed"
            else:
                self._rejected.add(asked.path)
                outcome = "rejected"
        elif _YES.match(text):
            if _NEGATION.search(text):
                # "Yeah, no, that's not right" / "yes, but not that one":
                # an affirmative opener with a negation anywhere in it is a
                # rejection, never a confirmation.
                self._rejected.add(asked.path)
                outcome = "rejected"
            elif not _bare_affirmation(text):
                # "sure, stop soccer" / "okay, where did I leave the Tibet
                # study?": an affirmative OPENER carrying a real request is
                # not consent, and confirming would also swallow the request.
                # Leave outcome "ignored" so the utterance falls through
                # untouched to the router/butler, and leave self._asking set
                # so the confirmation stays pending and gets re-asked.
                pass
            else:
                self._confirm(asked)
                outcome = "confirmed"

        if outcome != "ignored":
            self.registry.save(self.path)
            self.bus.publish("confirm.result", {"name": asked.name, "outcome": outcome})
            self._asking = None
            self._asking_spoken = False
            self._publish_counts()
        return outcome

    def _confirm(self, asked: Project) -> None:
        """Confirm by exact path. Upgrade to finance when it looks like finance;
        otherwise pass no kind, so an already-assigned kind is never downgraded."""
        if _looks_like_finance(asked):
            self.registry.confirm_path(asked.path, kind="finance")
        else:
            self.registry.confirm_path(asked.path)


def _looks_like_finance(p: Project) -> bool:
    hay = f"{p.name} {p.path}".lower()
    return any(w in hay for w in ("quant", "stock", "trad", "invest", "portfolio", "finance"))
