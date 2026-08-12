"""The project registry: what Marvin knows about Keke's repos, and which of
those a human has actually confirmed. Discovery proposes; only Keke disposes."""
from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
FUZZY_CUTOFF = 0.82
MIN_CONTAINMENT_LEN = 3

_KEEP = object()  # sentinel: "caller did not pass a kind — keep what's there"


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _contains_all_tokens(haystack: str, needle: str) -> bool:
    h, n = _tokens(haystack), _tokens(needle)
    return bool(n) and all(t in h for t in n)


@dataclass
class Project:
    name: str
    path: str
    aliases: list[str] = field(default_factory=list)
    mishearings: list[str] = field(default_factory=list)
    confirmed: bool = False
    kind: str = "code"
    data_source: str | None = None   # §16: the voice-confirmed output file, finance only

    def spoken_forms(self) -> list[str]:
        return [self.name, *self.aliases, *self.mishearings]


class Registry:
    def __init__(self, projects: list[Project] | None = None):
        self.projects: list[Project] = projects or []

    # ---------- persistence ----------
    @classmethod
    def load(cls, path: Path) -> "Registry":
        """Quarantine-and-continue load. This is the boot path.

        A file that exists but cannot be used (unparseable, wrong shape, or a
        rejected schema version) is renamed aside to `<path>.corrupt-<unix_ts>`
        and an empty registry is returned. One policy, two failures fixed:
        silently returning empty lets the next save() PERMANENTLY overwrite
        every confirmation, alias and mishearing with no signal, and raising
        refuses to boot the whole server — quarantining does neither. The
        human's file survives on disk for repair; callers who want the raise
        use load_strict()."""
        path = Path(path)
        try:
            return cls.load_strict(path)
        except Exception:  # noqa: BLE001 — any unusable file is quarantined, never destroyed
            if path.exists():
                quarantined = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
                try:
                    path.rename(quarantined)
                except OSError:
                    pass  # cannot rename (permissions?) — still boot empty
            return cls()

    @classmethod
    def load_strict(cls, path: Path) -> "Registry":
        """Parse `path` or raise. A missing file is an empty registry; an
        existing file that cannot be used raises (ValueError for shape/schema,
        the underlying error for I/O and JSON faults)."""
        path = Path(path)
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"wrong shape: expected a JSON object, got {type(raw).__name__}")
        version = raw.get("schema_version")
        if version is None or version > SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {version}")
        known = {f for f in Project.__dataclass_fields__}
        return cls([Project(**{k: v for k, v in p.items() if k in known})
                    for p in raw.get("projects", []) if isinstance(p, dict)])

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

    def _by_path(self, path: str) -> Project | None:
        return next((p for p in self.projects if p.path == path), None)

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
        if len(q) >= MIN_CONTAINMENT_LEN:  # never trust containment on voice filler
            contained = [p for p in confirmed
                         if any(_contains_all_tokens(q, f) or _contains_all_tokens(f, q)
                                for f in p.spoken_forms())]
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

    def confirm(self, name: str, kind=_KEEP) -> Project:
        p = self._by_name(name)
        if p is None:
            raise KeyError(name)
        return self._confirm(p, kind)

    def confirm_path(self, path: str, kind=_KEEP) -> Project:
        """Like confirm(), but keyed by exact path — the only way to reach the
        right entry when two directories share a basename."""
        p = self._by_path(path)
        if p is None:
            raise KeyError(path)
        return self._confirm(p, kind)

    @staticmethod
    def _confirm(p: Project, kind) -> Project:
        p.confirmed = True
        if kind is not _KEEP:
            p.kind = kind  # an explicit kind always wins
        elif not p.kind:
            p.kind = "code"  # belt-and-braces; the dataclass default already says "code"
        return p

    def add_alias(self, name: str, alias: str) -> None:
        p = self._by_name(name)
        if p is not None and alias and alias not in p.aliases:
            p.aliases.append(alias)

    def add_alias_path(self, path: str, alias: str) -> None:
        p = self._by_path(path)
        if p is not None and alias and alias not in p.aliases:
            p.aliases.append(alias)

    def add_mishearing(self, name: str, heard: str) -> None:
        p = self._by_name(name)
        if p is not None and heard and heard not in p.mishearings:
            p.mishearings.append(heard)

    def add_mishearing_path(self, path: str, heard: str) -> None:
        p = self._by_path(path)
        if p is not None and heard and heard not in p.mishearings:
            p.mishearings.append(heard)

    def set_data_source(self, path: str, source: str) -> None:
        """Pin the §16 voice-confirmed output file on the project at `path`.
        Path-keyed like every mutation the confirmation flows make."""
        p = self._by_path(path)
        if p is not None and source:
            p.data_source = source
