"""The §16 data-source confirmation: before the FIRST brief, JARVIS names the
file it intends to read and Keke says yes out loud.

This lives OUTSIDE finance.py on purpose: finance.py is source-scanned by a
test for execution/write primitives, and this module persists the registry.
Keeping the scanned file free of writes keeps that scan meaningful.

Vocabulary is imported from onboarding (aligned with the router in M3.2 Task
1) so a yes here is the same yes everywhere — and negation-anywhere still
beats an affirmative opener ("yeah, no" rejects)."""
from __future__ import annotations

import asyncio
from pathlib import Path

from server.finance import _rows_from_json, _rows_from_sqlite, detect_outputs
from server.onboarding import _NEGATION, _NO, _YES


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
            self._asking = None
            self.bus.publish("confirm.result", {"name": source, "outcome": "rejected"})
            return "rejected"
        if _YES.match(text):
            self.registry.set_data_source(project_path, source)
            await asyncio.to_thread(self.registry.save, self.path)
            self._asking = None
            self.bus.publish("confirm.result", {"name": source, "outcome": "confirmed"})
            return "confirmed"
        return "ignored"
