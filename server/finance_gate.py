"""The §16 data-source confirmation: before the FIRST brief, Marvin names the
file it intends to read and Keke says yes out loud.

This lives OUTSIDE finance.py on purpose: finance.py is source-scanned by a
test for execution/write primitives, and this module persists the registry.
Keeping the scanned file free of writes keeps that scan meaningful.

Vocabulary is imported from onboarding (aligned with the router in M3.2 Task
1) so a yes here is the same yes everywhere — and negation-anywhere still
beats an affirmative opener ("yeah, no" rejects). Only a BARE reply settles
the question, in either direction: an opener carrying a real request ("sure,
stop soccer", "no way, stop soccer") is not an answer, and acting on it would
also swallow the request — it falls through with the question still pending."""
from __future__ import annotations

import asyncio
from pathlib import Path

from server.finance import _rows_from_json, _rows_from_sqlite, detect_outputs
from server.onboarding import _NEGATION, _NO, _SEPARATORS, _YES, _bare_affirmation
from server.router import is_addressed


def _first_readable_source(root: Path) -> str | None:
    """The file portfolio_brief would read: first sqlite with rows, then first
    json with rows — same preference order, but PROVEN readable, so the spoken
    question never names a file the brief would then fail on."""
    outputs = detect_outputs(root)
    for kind, reader in (("sqlite", _rows_from_sqlite), ("json", _rows_from_json)):
        for path in outputs[kind]:
            if reader(path):
                return path
    return None


def _bare_rejection(text: str) -> bool:
    """True when the utterance is a rejection and NOTHING else — the only
    shape allowed to reject the proposed source. onboarding._bare_affirmation,
    mirrored for the other polarity: consume every leading yes/no phrase
    ("yeah, no" stacks an opener on the negation), then apply the router's
    rule to the residue. One asymmetry, and it fails SAFE: residual negation
    vocabulary ("no, that's not it") is rejection elaboration, not a request —
    a wrong "rejected" costs one re-ask, while calling it addressed would
    leave the question pending on speech that plainly answered it. _NO/_YES
    supply the yes/no vocabulary; the router supplies the stop-words — each
    defined once."""
    rest = text
    while True:
        m = _NO.match(rest) or _YES.match(rest)
        if m is None:
            break
        rest = rest[m.end():].lstrip(_SEPARATORS)
    return bool(_NEGATION.search(rest)) or not is_addressed(rest)


async def propose_source(project) -> str | None:
    if project is None:
        return None
    return await asyncio.to_thread(_first_readable_source, Path(project.path))


class SourceGate:
    def __init__(self, bus, registry, registry_path: Path):
        self.bus = bus
        self.registry = registry
        self.path = Path(registry_path)
        self._asking: tuple[str, str] | None = None   # (project_path, source)
        # The exact Project the last yes pinned. The brain's confirmed branch
        # briefs THIS, never a re-derived lookup: with two confirmed finance
        # projects, re-deriving could select the other one — unpinned, so its
        # _collect would scan for a file Keke never confirmed (spec §16).
        self.confirmed_project = None

    @property
    def awaiting(self) -> bool:
        return self._asking is not None

    async def ask(self, project) -> str | None:
        """Propose the source and speak the §16 question. None when the repo
        has no readable output yet — the caller speaks that honestly instead."""
        source = await propose_source(project)
        if source is None:
            self._asking = None
            return None
        self._asking = (project.path, source)
        question = f"I'll read the picks from {Path(source).name} — correct, sir?"
        self.bus.publish("confirm.request",
                         {"name": project.name, "path": source, "question": question})
        return question

    async def handle_reply(self, spoken: str) -> str:
        if self._asking is None:
            return "ignored"
        text = (spoken or "").strip()
        project_path, source = self._asking
        # Negation first, and negation-anywhere disqualifies a yes: consent to
        # read a money-adjacent file must never ride on the first word.
        if _NO.match(text) or (_YES.match(text) and _NEGATION.search(text)):
            if not _bare_rejection(text):
                # "no way, stop soccer": a negating OPENER carrying a real
                # request is not an answer — and rejecting on it would also
                # swallow the request. Fall through to the router/butler
                # untouched, with the question still pending.
                return "ignored"
            self._asking = None
            self.bus.publish("confirm.result", {"name": source, "outcome": "rejected"})
            return "rejected"
        if _YES.match(text):
            if not _bare_affirmation(text):
                # "sure, stop soccer" / "yes, pull up composed": an
                # affirmative OPENER carrying a real request is not consent
                # (Task 1's onboarding discipline, applied to money). Fall
                # through untouched; nothing is pinned, the question stays.
                return "ignored"
            self.registry.set_data_source(project_path, source)
            await asyncio.to_thread(self.registry.save, self.path)
            self._asking = None
            self.confirmed_project = next(
                (p for p in self.registry.projects if p.path == project_path), None)
            self.bus.publish("confirm.result", {"name": source, "outcome": "confirmed"})
            return "confirmed"
        return "ignored"
