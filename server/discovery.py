"""Find the projects Keke actually works in. Read-only on the user's machine.

Three independent sources, ranked by how much they corroborate each other:
  1. ~/.claude.json "projects" — REAL absolute paths, the reliable source.
  2. ~/.claude/projects/<slug>/ — a directory per repo Claude Code has run in.
     The slug encoding is LOSSY (a real directory name may contain '-'), so we
     never decode a slug; we encode a known path and test for its directory.
  3. A shallow git scan of common code roots, for repos Claude Code never saw.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

WORKTREE_MARKER = "/.claude/worktrees/"
DEFAULT_ROOTS = ("", "Desktop", "Documents", "code", "src", "projects")


@dataclass
class Candidate:
    path: str
    name: str
    sources: list[str] = field(default_factory=list)
    is_git: bool = False
    has_session_dir: bool = False
    mtime: float = 0.0

    @property
    def score(self) -> tuple:
        """Best-first ordering key. More corroboration wins; recency breaks ties."""
        return (len(self.sources), self.has_session_dir, self.is_git, self.mtime)


def slug_for(path: str) -> str:
    """Encode a real path the way Claude Code names its project directories.

    Encode-only by design: decoding is ambiguous because '-' is legal in a
    directory name (e.g. '.../Codex-2026-06-30-new-chat').
    """
    return str(path).replace("/", "-")


def scan_claude_json(home: Path) -> list[str]:
    try:
        raw = json.loads((Path(home) / ".claude.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    projects = raw.get("projects")
    return list(projects.keys()) if isinstance(projects, dict) else []


def scan_git_dirs(roots: list[Path], max_depth: int = 2) -> list[str]:
    found: list[str] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            try:
                entries = list(d.iterdir())
            except OSError:
                continue
            if any(e.name == ".git" for e in entries):
                found.append(str(d))
                continue                      # do not descend into a repo
            if depth < max_depth:
                stack += [(e, depth + 1) for e in entries
                          if e.is_dir() and not e.name.startswith(".")]
    return found


def _usable(path: str) -> bool:
    return WORKTREE_MARKER not in path and Path(path).is_dir()


def _collect(home: Path, extra_roots: list[Path] | None) -> list[Candidate]:
    by_path: dict[str, Candidate] = {}

    def add(path: str, source: str) -> None:
        if not _usable(path):
            return
        c = by_path.get(path)
        if c is None:
            p = Path(path)
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            c = Candidate(path=path, name=p.name, is_git=(p / ".git").exists(),
                          has_session_dir=(Path(home) / ".claude" / "projects"
                                           / slug_for(path)).is_dir(),
                          mtime=mtime)
            by_path[path] = c
        if source not in c.sources:
            c.sources.append(source)

    for path in scan_claude_json(home):
        add(path, "claude.json")
    roots = [Path(home) / r if r else Path(home) for r in DEFAULT_ROOTS]
    for path in scan_git_dirs(roots + list(extra_roots or [])):
        add(path, "git-scan")
    return sorted(by_path.values(), key=lambda c: c.score, reverse=True)


async def discover(home: Path, extra_roots: list[Path] | None = None) -> list[Candidate]:
    """Ranked candidates, best-first. Filesystem work runs off the event loop."""
    return await asyncio.to_thread(_collect, Path(home), extra_roots)
