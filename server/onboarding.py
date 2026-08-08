"""The confirmation beat: JARVIS proposes a repo, Keke confirms it out loud.

Discovery is a guess. Nothing becomes usable — least of all the finance repo —
until a human says yes, so this module is the only path to `confirmed`.

Every registry mutation here is keyed by PATH, not name: two different
directories on this machine are both named `jarvis`, and a name-keyed call
resolves to whichever comes first. We hold the exact Project we asked about,
so we always know its path precisely.
"""
from __future__ import annotations

import re
from pathlib import Path

from server.discovery import discover
from server.registry import Project, Registry

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
    r"^\s*(?:(?:yes|yeah|yep|confirm)\b"
    r"|that'?s\s+(?:it|right|correct)\b"
    r"|(?:right|correct)\s*[.!?]*\s*$)", re.I)
_NO = re.compile(r"^\s*(?:no|nope|skip|not\s+(?:that|it)|wrong)\b", re.I)
# A negation ANYWHERE in the utterance disqualifies a yes. Spoken rejections
# very often open affirmatively — "Yeah, no, that's not right", "yes, but not
# that one" — and _YES anchors on that opener, so prefix order alone cannot
# save us. Confirmation is consent; it must never ride on the first word.
_NEGATION = re.compile(r"\b(?:no|nope|not|don'?t|wrong|isn'?t)\b", re.I)


class Onboarding:
    def __init__(self, bus, registry: Registry, registry_path: Path):
        self.bus = bus
        self.registry = registry
        self.path = Path(registry_path)
        self._asking: Project | None = None
        self._rejected: set[str] = set()  # paths, not names — names can collide

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
        prompt = self.next_prompt()
        if prompt is None:
            self._asking = None
            return False
        self._asking = next(p for p in self.registry.projects if p.path == prompt["path"])
        self.bus.publish("confirm.request", prompt)
        return True

    async def handle_reply(self, spoken: str) -> str:
        if self._asking is None:
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
            else:
                self._confirm(asked)
                outcome = "confirmed"

        if outcome != "ignored":
            self.registry.save(self.path)
            self.bus.publish("confirm.result", {"name": asked.name, "outcome": outcome})
            self._asking = None
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
