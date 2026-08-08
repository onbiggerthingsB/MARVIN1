"""Append-only vault writes to two fixed destinations, serialized through one lock."""
from __future__ import annotations

import asyncio
import weakref
from datetime import datetime
from pathlib import Path

from server.vault_paths import assert_append_allowed, capture_target, log_target

# One lock PER EVENT LOOP, built lazily. A module-level asyncio.Lock() created at
# import time binds to whichever loop first awaits it, and every pytest-asyncio
# test gets a fresh loop -- so a second loop would hit "bound to a different event
# loop" now that the critical section actually awaits. Weak keys let a finished
# loop drop its lock instead of leaking it, and sidestep the address reuse a plain
# {id(loop): lock} dict is exposed to.
_write_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _write_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _write_locks[loop] = lock
    return lock


def _one_line(s: str) -> str:
    """Collapse every whitespace run to a single space: one entry is always one line.

    Interpolated text is untrusted (it comes from speech transcription). A newline
    followed by "# " would open a real Markdown heading inside Keke's daily note,
    and a newline followed by "## [" would forge an entry in the audit log. Neither
    can survive being flattened onto a single line.
    """
    return " ".join((s or "").split())


def _write_sync(target: Path, entry: str, header: str | None) -> None:
    """The blocking append. Called via asyncio.to_thread: the vault lives on iCloud,
    where a cold mkdir/open/write can stall for seconds and would otherwise freeze
    the event loop that is carrying live audio."""
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    size = target.stat().st_size if exists else 0
    prefix = ""
    if exists and size:
        # If the file's last line has no trailing newline, a plain append grafts our
        # entry onto the end of Keke's own sentence -- deforming both. Repair it.
        with target.open("rb") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                prefix = "\n"
    with target.open("a", encoding="utf-8") as f:
        if not exists and header:
            f.write(header)
        f.write(prefix + entry)


async def _append(target: Path, entry: str, header: str | None = None) -> None:
    async with _lock():
        await asyncio.to_thread(_write_sync, target, entry, header)


async def vault_capture(text: str, vault_root: Path, now: datetime | None = None) -> str:
    now = datetime.now() if now is None else now
    target = capture_target(vault_root, now)
    # firewall FIRST: nothing below may create a directory or touch a file until
    # the target has been proven to be one of the two fixed, in-vault destinations
    assert_append_allowed(target, vault_root, now)
    await _append(target, f"- {now:%H:%M} {_one_line(text)}\n", header=f"# {now:%Y-%m-%d}\n\n")
    return str(target)


async def vault_log(op: str, title: str, vault_root: Path, now: datetime | None = None) -> str:
    now = datetime.now() if now is None else now
    target = log_target(vault_root)
    assert_append_allowed(target, vault_root, now)
    await _append(target, f"## [{now:%Y-%m-%d %H:%M}] {_one_line(op)} | {_one_line(title)}\n")
    return str(target)
