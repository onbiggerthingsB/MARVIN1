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

    A note longer than the cap comes back with a VISIBLE truncation marker
    appended, so the model knows it read a prefix rather than the whole note.
    Only the cap is ever read from disk (not the whole file sliced after), and
    the decoded text is re-capped on its UTF-8 length so a non-UTF-8 note --
    where each bad byte decodes to a 3-byte U+FFFD -- cannot triple in size.
    """
    p = _safe_note(rel_path, vault_root)
    if not p.is_file():
        raise FileNotFoundError(rel_path)
    size = p.stat().st_size
    with p.open("rb") as f:
        data = f.read(MAX_BYTES)
    text = data.decode("utf-8", "replace")
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_BYTES:
        # "ignore" drops the one possibly-torn trailing sequence at the cut, so
        # the result's UTF-8 length is <= MAX_BYTES with no mojibake tail
        text = encoded[:MAX_BYTES].decode("utf-8", "ignore")
    if size > MAX_BYTES:
        text += (f"\n\n[TRUNCATED: first {MAX_BYTES:,} of {size:,} bytes"
                 " — the rest of this note was NOT read]")
    return text


def _first_match_snippet(path: Path, query: str) -> str:
    q = query.lower()
    try:
        with path.open("rb") as f:
            data = f.read(MAX_BYTES)
    except OSError:
        return ""
    text = data.decode("utf-8", "replace")
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

    FAILURE is not emptiness. Every failure path (rg missing, bad cwd, timeout,
    rg exit >= 2) returns the zero-hit shape PLUS an `"error": "<reason>"` key,
    so the caller can tell "the vault has nothing on this" apart from "the
    search never ran". A genuine zero-match result carries NO `error` key.
    """
    root = Path(vault_root).resolve()
    if not query.strip():
        return {"total": 0, "results": []}

    def _failed(reason: str) -> dict:
        return {"total": 0, "results": [], "error": reason}

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
    except FileNotFoundError as e:
        # Not necessarily "rg not installed": a missing/bad cwd raises the same
        # exception. Log the actual exception so the two are distinguishable.
        log.warning("vault_search: rg failed to start: %s", e)
        return _failed(f"ripgrep not found or vault dir missing: {e}")
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=SEARCH_TIMEOUT)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        log.warning("vault_search: rg timed out after %ss", SEARCH_TIMEOUT)
        return _failed(f"timed out after {SEARCH_TIMEOUT:g}s")
    # rg exit codes: 0 = matches, 1 = no matches, 2+ = real error
    if proc.returncode not in (0, 1):
        log.warning("vault_search: rg failed (exit %s): %s",
                    proc.returncode, err.decode("utf-8", "replace")[:200])
        return _failed(f"rg exit {proc.returncode}")
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
