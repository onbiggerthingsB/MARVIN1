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


def _rank_key(path: str, count: int) -> tuple:
    """Total order over hits. Deterministic: no two distinct paths tie.

    1. `is_archive` FIRST: `Claude Chats/` and `ChatGPT Chats/` are agent-written
       transcripts, not Keke's primary notes, so they sort after everything else
       no matter how many times they mention the term. Deprioritized, never
       excluded -- a term that only appears in a transcript still surfaces.
    2. DESCENDING match count: a note that mentions "chant" on nine lines is
       about chanting; one that mentions it once is probably a passing mention.
       (`rg -c` counts matching LINES, not occurrences -- close enough for
       line-oriented markdown prose, and it is one number per file either way.)
    3. ASCENDING path depth: top-level curated notes beat deeply nested archives
       at the same count.
    4. ASCENDING path: the tiebreak that makes the order reproducible. Without
       it the order was ripgrep's parallel-walk emission order, which genuinely
       differs run to run -- the top 3 of a 164-hit query changed on three
       consecutive runs against the real vault.
    """
    is_archive = path.startswith(("Claude Chats/", "ChatGPT Chats/"))
    return (is_archive, -count, path.count("/"), path)


async def vault_search(query: str, vault_root: Path, limit: int = 5) -> dict:
    """Case-insensitive LITERAL search over the vault's .md files, RANKED.

    The query is matched as a FIXED STRING, not a regex (rg runs with -F), so
    regex metacharacters like `(`, `?` or `*` match themselves and an unbalanced
    paren is a normal query rather than a syntax error. The same literal rule is
    what `_first_match_snippet` re-applies, so a hit always yields a snippet.

    Because matching is literal, a multi-word natural-language question ("what
    did the chant study find?") is searched as one long phrase and will usually
    return NOTHING. Callers should pass keywords, not sentences.

    Returns `{"total": <matching files>, "results": [...]}`. Only the top `limit`
    entries by `_rank_key` are returned, but `total` reports how many files
    matched overall so the caller can tell the model it is seeing a subset -- a
    common word can match hundreds of notes, and silently showing five of them
    let the butler conclude the other 159 did not exist.
    """
    root = Path(vault_root).resolve()
    empty: dict = {"total": 0, "results": []}
    if not query.strip():
        return empty
    try:
        proc = await asyncio.create_subprocess_exec(
            # -F: literal/fixed-string match, so the snippet matcher agrees with rg
            # --no-ignore: parts of the vault are gitignored but still searchable
            # -c: one "path:count" line per file (matching lines), the ranking signal
            "rg", "-F", "--no-ignore", "-i", "-c", "--glob", "*.md", "--", query,
            cwd=str(root),
            # DEVNULL, not inherit: rg searches stdin instead of cwd when fd 0 is a regular file
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError:
        log.warning("vault_search: ripgrep (rg) not installed")
        return empty
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=SEARCH_TIMEOUT)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        log.warning("vault_search: rg timed out after %ss", SEARCH_TIMEOUT)
        return empty
    # rg exit codes: 0 = matches, 1 = no matches, 2+ = real error
    if proc.returncode not in (0, 1):
        log.warning("vault_search: rg failed (exit %s): %s",
                    proc.returncode, err.decode("utf-8", "replace")[:200])
        return empty
    hits: list[tuple[str, int]] = []
    for line in out.decode("utf-8", "replace").splitlines():
        if not line:
            continue
        # rsplit, not split: a note filename may legitimately contain a colon,
        # and only the LAST field is the count rg appended.
        rel, _, raw = line.rpartition(":")
        if not rel:
            continue
        try:
            hits.append((rel, int(raw)))
        except ValueError:
            continue
    hits.sort(key=lambda h: _rank_key(h[0], h[1]))
    results = []
    for rel, count in hits[:limit]:
        p = root / rel
        # to_thread: an iCloud-evicted file can block on read for seconds
        snippet = await asyncio.to_thread(_first_match_snippet, p, query)
        results.append({"title": Path(rel).stem, "path": rel,
                        "snippet": snippet, "count": count})
    return {"total": len(hits), "results": results}


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
