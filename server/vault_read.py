"""Read-only vault access: rg-backed search, guarded single-file read, iCloud guard."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

log = logging.getLogger(__name__)

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
    """Return at most MAX_BYTES of the note, decoded as UTF-8 with replacement.

    Output is TRUNCATED SILENTLY at the byte cap: there is no marker and no
    signal to the caller, so a long note comes back cut off mid-line. Callers
    that need the whole file must read it themselves.
    """
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
    """Case-insensitive LITERAL search over the vault's .md files.

    The query is matched as a FIXED STRING, not a regex (rg runs with -F), so
    regex metacharacters like `(`, `?` or `*` match themselves and an unbalanced
    paren is a normal query rather than a syntax error. The same literal rule is
    what `_first_match_snippet` re-applies, so a hit always yields a snippet.

    Because matching is literal, a multi-word natural-language question ("what
    did the chant study find?") is searched as one long phrase and will usually
    return NOTHING. Callers should pass keywords, not sentences.

    Returns at most `limit` results; further matches are dropped silently.
    """
    root = Path(vault_root).resolve()
    if not query.strip():
        return []
    try:
        proc = await asyncio.create_subprocess_exec(
            # -F: literal/fixed-string match, so the snippet matcher agrees with rg
            # --no-ignore: parts of the vault are gitignored but still searchable
            "rg", "-F", "--no-ignore", "-i", "-l", "--glob", "*.md", "--", query,
            cwd=str(root),
            # DEVNULL, not inherit: rg searches stdin instead of cwd when fd 0 is a regular file
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError:
        log.warning("vault_search: ripgrep (rg) not installed")
        return []
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=SEARCH_TIMEOUT)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        log.warning("vault_search: rg timed out after %ss", SEARCH_TIMEOUT)
        return []
    # rg exit codes: 0 = matches, 1 = no matches, 2+ = real error
    if proc.returncode not in (0, 1):
        log.warning("vault_search: rg failed (exit %s): %s",
                    proc.returncode, err.decode("utf-8", "replace")[:200])
        return []
    files = [f for f in out.decode("utf-8", "replace").splitlines() if f][:limit]
    results = []
    for rel in files:
        p = root / rel
        # to_thread: an iCloud-evicted file can block on read for seconds
        snippet = await asyncio.to_thread(_first_match_snippet, p, query)
        results.append({"title": Path(rel).stem, "path": rel, "snippet": snippet})
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
