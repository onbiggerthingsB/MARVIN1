"""Append-only vault writes to two fixed destinations, serialized through one lock."""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from server.vault_paths import assert_append_allowed, capture_target, log_target

_write_lock = asyncio.Lock()


async def _append(target: Path, entry: str, header: str | None = None) -> None:
    async with _write_lock:
        target.parent.mkdir(parents=True, exist_ok=True)
        new_file = not target.exists()
        with target.open("a", encoding="utf-8") as f:
            if new_file and header:
                f.write(header)
            f.write(entry)


async def vault_capture(text: str, vault_root: Path, now: datetime | None = None) -> str:
    now = now or datetime.now()
    target = capture_target(vault_root, now)
    assert_append_allowed(target, vault_root, now)
    await _append(target, f"- {now:%H:%M} {text.strip()}\n", header=f"# {now:%Y-%m-%d}\n\n")
    return str(target)


async def vault_log(op: str, title: str, vault_root: Path, now: datetime | None = None) -> str:
    now = now or datetime.now()
    target = log_target(vault_root)
    assert_append_allowed(target, vault_root, now)
    await _append(target, f"## [{now:%Y-%m-%d %H:%M}] {op} | {title}\n")
    return str(target)
