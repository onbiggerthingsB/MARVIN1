"""The project registry: what JARVIS knows about Keke's repos, and which of
those a human has actually confirmed. Discovery proposes; only Keke disposes."""
from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
FUZZY_CUTOFF = 0.82


@dataclass
class Project:
    name: str
    path: str
    aliases: list[str] = field(default_factory=list)
    mishearings: list[str] = field(default_factory=list)
    confirmed: bool = False
    kind: str = "code"

    def spoken_forms(self) -> list[str]:
        return [self.name, *self.aliases, *self.mishearings]


class Registry:
    def __init__(self, projects: list[Project] | None = None):
        self.projects: list[Project] = projects or []

    # ---------- persistence ----------
    @classmethod
    def load(cls, path: Path) -> "Registry":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        version = raw.get("schema_version")
        if version is None or version > SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {version}")
        known = {f for f in Project.__dataclass_fields__}
        return cls([Project(**{k: v for k, v in p.items() if k in known})
                    for p in raw.get("projects", [])])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"schema_version": SCHEMA_VERSION,
             "projects": [asdict(p) for p in self.projects]}, indent=2), encoding="utf-8")
        tmp.replace(path)

    # ---------- lookup ----------
    def _by_name(self, name: str) -> Project | None:
        return next((p for p in self.projects if p.name == name), None)

    def pending(self) -> list[Project]:
        return [p for p in self.projects if not p.confirmed]

    def match(self, spoken: str) -> list[Project]:
        """Confirmed projects whose name/alias/mishearing matches. Never guesses
        an unconfirmed project — an unknown repo must be confirmed by a human."""
        q = (spoken or "").strip().lower()
        if not q:
            return []
        confirmed = [p for p in self.projects if p.confirmed]
        exact = [p for p in confirmed if any(f.lower() == q for f in p.spoken_forms())]
        if exact:
            return exact
        contained = [p for p in confirmed
                     if any(f.lower() in q or q in f.lower() for f in p.spoken_forms())]
        if contained:
            return contained
        fuzzy = [p for p in confirmed
                 if any(difflib.SequenceMatcher(None, q, f.lower()).ratio() >= FUZZY_CUTOFF
                        for f in p.spoken_forms())]
        return fuzzy

    # ---------- mutation ----------
    def merge_candidates(self, candidates) -> list[Project]:
        """Add unseen candidates as UNCONFIRMED. Existing entries are untouched,
        so a rediscovery can never undo a human confirmation or an alias."""
        known = {p.path for p in self.projects}
        added = []
        for c in candidates:
            if c.path in known:
                continue
            p = Project(name=c.name, path=c.path)
            self.projects.append(p)
            added.append(p)
            known.add(c.path)
        return added

    def confirm(self, name: str, kind: str = "code") -> Project:
        p = self._by_name(name)
        if p is None:
            raise KeyError(name)
        p.confirmed = True
        p.kind = kind
        return p

    def add_alias(self, name: str, alias: str) -> None:
        p = self._by_name(name)
        if p is not None and alias and alias not in p.aliases:
            p.aliases.append(alias)

    def add_mishearing(self, name: str, heard: str) -> None:
        p = self._by_name(name)
        if p is not None and heard and heard not in p.mishearings:
            p.mishearings.append(heard)
