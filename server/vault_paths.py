"""Vault paths and the append-only write firewall. Pure, no I/O beyond resolve()."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

_DEFAULT = "Library/Mobile Documents/iCloud~md~obsidian/Documents/KEKE LI"


def vault_root_from_env() -> Path:
    override = os.environ.get("JARVIS_VAULT")
    return Path(override) if override else Path.home() / _DEFAULT


def capture_target(vault_root: Path, now: datetime) -> Path:
    return vault_root / "Daily" / f"{now:%Y-%m-%d}.md"


def log_target(vault_root: Path) -> Path:
    return vault_root / "_Claude" / "log.md"


def _resolve(p: Path) -> Path:
    # realpath is non-strict by default: a not-yet-created target still resolves its parent symlinks
    return Path(os.path.realpath(p))


def is_within(path: Path, root: Path) -> bool:
    rp, rr = _resolve(path), _resolve(root)
    try:
        rp.relative_to(rr)
        return True
    except ValueError:
        return False


def assert_append_allowed(path: Path, vault_root: Path, now: datetime) -> None:
    # The allowlist is the RESOLVED ROOT joined with LITERAL components -- never
    # a resolve() of the candidate's own expression. Resolving the same target
    # expression the caller passed in made the membership test a tautology, so a
    # pre-existing in-vault symlink (Daily/2026-08-08.md -> Coursework/essay.md)
    # sailed through and the append landed in Keke's own prose. With literal
    # components, a symlink anywhere in the last two path segments makes
    # _resolve(path) diverge from the allowlisted path and is refused loudly.
    rr = _resolve(vault_root)
    allowed = {rr / "Daily" / f"{now:%Y-%m-%d}.md", rr / "_Claude" / "log.md"}
    resolved = _resolve(path)
    if resolved not in allowed:
        raise PermissionError(f"append not allowed (not a fixed target): {path}")
    # Second layer: still catches a symlinked vault_root itself.
    if not is_within(resolved, rr):
        raise PermissionError(f"append target resolves outside the vault: {path}")
