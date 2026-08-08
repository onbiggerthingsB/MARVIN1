"""Read-only vault access: rg-backed search, guarded single-file read, iCloud guard."""
from __future__ import annotations

import asyncio
from pathlib import Path

MAX_BYTES = 200_000
SEARCH_TIMEOUT = 8.0


def _safe_note(rel_path: str, vault_root: Path) -> Path:
    root = Path(vault_root).resolve()
    resolved = (root / rel_path).resolve()
    if resolved != root and not str(resolved).startswith(str(root) + "/"):
        raise PermissionError(f"path outside vault: {rel_path}")
    if resolved.suffix.lower() != ".md":
        raise PermissionError(f"not a markdown note: {rel_path}")
    return resolved


def vault_read(rel_path: str, vault_root: Path) -> str:
    p = _safe_note(rel_path, vault_root)
    if not p.is_file():
        raise FileNotFoundError(rel_path)
    return p.read_bytes()[:MAX_BYTES].decode("utf-8", "replace")


def _first_match_snippet(path: Path, query: str) -> str:
    q = query.lower()
    try:
        text = path.read_bytes()[:MAX_BYTES].decode("utf-8", "replace")
    except OSError:
        return ""
    for line in text.splitlines():
        if q in line.lower() and line.strip():
            return line.strip()[:200]
    return ""


async def vault_search(query: str, vault_root: Path, limit: int = 5) -> list[dict]:
    root = Path(vault_root).resolve()
    if not query.strip():
        return []
    try:
        proc = await asyncio.create_subprocess_exec(
            "rg", "-i", "-l", "--glob", "*.md", "--", query,
            cwd=str(root),
            # DEVNULL, not inherit: rg searches stdin instead of cwd when fd 0 is a regular file
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=SEARCH_TIMEOUT)
    except (FileNotFoundError, asyncio.TimeoutError):
        return []
    files = [f for f in out.decode("utf-8", "replace").splitlines() if f][:limit]
    results = []
    for rel in files:
        p = root / rel
        results.append({"title": Path(rel).stem, "path": rel,
                        "snippet": _first_match_snippet(p, query)})
    return results


async def ensure_materialized(path: Path) -> None:
    """Best-effort iCloud download of an evicted file. Never raises."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "brctl", "download", str(path),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await asyncio.wait_for(proc.wait(), timeout=10)
    except Exception:
        pass


def vault_is_downloaded(vault_root: Path) -> bool:
    probe = Path(vault_root) / "_Claude" / "index.md"
    return probe.is_file() and probe.stat().st_size > 0
