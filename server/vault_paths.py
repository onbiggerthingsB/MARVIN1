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
    # strict=False so a not-yet-created target still resolves its parent symlinks
    return Path(os.path.realpath(p))


def is_within(path: Path, root: Path) -> bool:
    rp, rr = _resolve(path), _resolve(root)
    try:
        rp.relative_to(rr)
        return True
    except ValueError:
        return False


def assert_append_allowed(path: Path, vault_root: Path, now: datetime) -> None:
    resolved = _resolve(path)
    allowed = {_resolve(capture_target(vault_root, now)), _resolve(log_target(vault_root))}
    if resolved not in allowed:
        raise PermissionError(f"append not allowed (not a fixed target): {path}")
    if not is_within(resolved, vault_root):
        raise PermissionError(f"append target resolves outside the vault: {path}")
