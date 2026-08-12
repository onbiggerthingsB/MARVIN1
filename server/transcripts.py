"""Read-only resolution of a session's on-disk transcript (spec §12, M4).

Claude Code writes every session to
    ~/.claude/projects/<project-dir>/<session-id>.jsonl
where <project-dir> is the session's cwd with every non-alphanumeric character
replaced by a dash (verified against the four real sessions in state/
fleet.jsonl — all four files exist at exactly that derivation). This module
turns (cwd, session_id) into that file and reads its tail, so the pull-up
verb can serve a DETACHED worker, a restart ghost, or anything else whose
in-memory deque is gone but whose file is not.

THE TRUST DECISION, stated once and relied on everywhere: the hook payload's
`transcript_path` field is NEVER opened. It arrives in an HTTP POST made by a
worker process — a real agent, prompt-injectable through any file it reads —
so it is attacker-influenceable input naming an arbitrary file on this disk.
Instead the location is DERIVED from facts Marvin already holds (the worktree
path it created; the session id its own SDK stream or fleet log recorded),
and the derivation is validated shut:

  * the session id must be UUID-shaped — anything else (a traversal string, a
    "sess-42") is refused before it can become path components;
  * the project-dir mangle maps every separator to a dash, so no cwd can
    escape the transcripts root by construction;
  * the derived path is then verified the way vault_paths does it: the
    RESOLVED root joined with LITERAL components must equal the realpath of
    the file — a symlink anywhere in the last two segments makes them
    diverge and is refused (the shipped containment defect this codebase
    already paid for was a check that followed symlinks and pinned nothing);
  * the open is O_NOFOLLOW | O_NONBLOCK with an lstat/fstat inode match
    around it, so a swap between check and open fails instead of following,
    and a FIFO cannot wedge the reading thread;
  * only a regular file is read, only its last TAIL_BYTES, and only KEEP
    rendered lines survive, each capped at LINE_CHAR_CAP characters.

Every refusal is a TranscriptUnavailable whose message is a short spoken
clause — the caller's discipline is that an unreadable transcript is SAID,
never rendered as an empty pane that looks like nothing happened."""
from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

# Parity with fleet.py's live pane: same retention, same per-line cap.
KEEP = 200                   # == fleet.TRANSCRIPT_KEEP
LINE_CHAR_CAP = 2000         # == the consumer's [:2000]
TAIL_BYTES = 4 * 1024 * 1024

SESSION_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

# The argument keys worth showing for a tool_use block, in the same
# preference order the live readback uses (fleet._short_args).
_ARG_KEYS = ("command", "file_path", "path", "pattern", "url")


class TranscriptUnavailable(Exception):
    """Spoken to Keke — the message is a short clause, never a traceback."""


def default_root() -> Path:
    """Where the Claude Code CLI on this machine writes session transcripts.

    Fixed, not read from any payload: pinning the root is half the
    containment. (A CLAUDE_CONFIG_DIR override would move the CLI's files —
    Marvin spawns its workers without one, so ~/.claude is the truth here.)"""
    return Path.home() / ".claude" / "projects"


def project_dir_name(cwd: str) -> str:
    """Claude Code's own project-directory encoding of a session cwd.

    Every character outside [A-Za-z0-9] becomes a dash — verified verbatim
    against the real directories on this machine (worktree paths, and the
    vault path with its spaces and tildes). No separator survives, so the
    result can never name a parent directory."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(cwd or ""))


def derive_transcript_path(cwd: str, session_id: str, root: Path) -> Path:
    """root/<mangled cwd>/<session id>.jsonl, refused unless both inputs are
    the shape Marvin's own records produce."""
    sid = str(session_id or "")
    if not SESSION_ID_RE.match(sid):
        raise TranscriptUnavailable("its session id doesn't look like one, "
                                    "so I won't turn it into a filename")
    text = str(cwd or "").strip()
    if not text or not Path(text).is_absolute():
        raise TranscriptUnavailable("its working directory was never "
                                    "recorded properly")
    return Path(root) / project_dir_name(text) / f"{sid}.jsonl"


def _args_line(args) -> str:
    """One console line for a tool_use input. This renders on SCREEN with
    textContent (never speech), so unlike the spoken readback it shows the
    argument uncut up to the shared line cap — no middle-elision games."""
    if isinstance(args, dict):
        for key in _ARG_KEYS:
            if args.get(key):
                return str(args[key])
    return str(args)


def _render(rec: dict) -> dict | None:
    """One pane line from one transcript record, or None for the records the
    live pane never showed either (attachments, tool_results, queue
    operations, last-prompt/mode trailers)."""
    rtype = rec.get("type")
    msg = rec.get("message")
    if rtype not in ("user", "assistant") or not isinstance(msg, dict):
        return None
    content = msg.get("content")
    texts: list[str] = []
    if rtype == "user":
        # A user record with STRING content is a real prompt; list content is
        # a tool_result the live pane never rendered.
        if isinstance(content, str) and content.strip():
            texts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                texts.append(str(block["text"]))
            elif block.get("type") == "tool_use" and block.get("name"):
                texts.append(f"[{block['name']}] "
                             f"{_args_line(block.get('input') or {})}")
    if not texts:
        return None
    return {"who": "user" if rtype == "user" else "worker",
            "text": "\n".join(texts)[:LINE_CHAR_CAP]}


def read_session_tail(cwd: str, session_id: str, root: Path, *,
                      keep: int = KEEP,
                      tail_bytes: int = TAIL_BYTES) -> list[dict]:
    """The last `keep` renderable lines of a session's transcript file.

    BLOCKING — call it via asyncio.to_thread; the transcript may be large
    and lives on disk. Raises TranscriptUnavailable (a spoken clause) for
    everything it refuses to or cannot read; returns [] only for a file that
    is genuinely readable and empty of renderable lines."""
    expected = derive_transcript_path(cwd, session_id, root)
    # vault_paths' containment pattern: the RESOLVED ROOT joined with LITERAL
    # components, compared against the candidate's realpath. A symlink in
    # either of the last two segments — the file, or the project dir — makes
    # realpath(expected) diverge from expected and is refused loudly. This
    # tolerates a symlinked root (macOS /var style) while pinning everything
    # below it.
    root_r = Path(os.path.realpath(root))
    expected = root_r / expected.parent.name / expected.name
    resolved = Path(os.path.realpath(expected))
    if resolved != expected:
        raise TranscriptUnavailable("its transcript path goes through a "
                                    "symlink, so I won't read it")
    try:
        pre = os.stat(expected, follow_symlinks=False)
    except FileNotFoundError:
        raise TranscriptUnavailable("its transcript file is not on disk "
                                    "any more") from None
    except OSError:
        raise TranscriptUnavailable("its transcript file can't be "
                                    "examined") from None
    if not stat.S_ISREG(pre.st_mode):
        raise TranscriptUnavailable("its transcript is not a regular file, "
                                    "so I won't read it")
    # O_NOFOLLOW: a final component swapped to a symlink between the check
    # and this open fails with ELOOP instead of following. O_NONBLOCK: a
    # FIFO swapped in cannot wedge this thread waiting for a writer (it is
    # inert on a regular file). The fstat/lstat inode match closes the rest
    # of the race: whatever this fd actually opened must be the very inode
    # the checks above examined.
    try:
        fd = os.open(expected, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        raise TranscriptUnavailable("its transcript file is not on disk "
                                    "any more") from None
    except OSError:
        raise TranscriptUnavailable("its transcript file would not open "
                                    "safely") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or \
                (st.st_dev, st.st_ino) != (pre.st_dev, pre.st_ino):
            raise TranscriptUnavailable("its transcript file changed under "
                                        "me, so I won't read it")
        with os.fdopen(fd, "rb") as f:
            fd = -1                    # ownership passed to the file object
            if st.st_size > tail_bytes:
                f.seek(st.st_size - tail_bytes)
                f.readline()           # drop the partial line the seek cut
            data = f.read(tail_bytes)
    finally:
        if fd >= 0:
            os.close(fd)
    lines: list[dict] = []
    unparsed = 0
    for raw in data.splitlines():
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            unparsed += 1
            continue
        if not isinstance(rec, dict):
            unparsed += 1
            continue
        entry = _render(rec)
        if entry:
            lines.append(entry)
            if len(lines) > keep:
                del lines[0]
    if not lines and unparsed:
        # Bytes were there and none of them could be read: that is damage,
        # and damage must be a sentence — an empty pane over a full file
        # would be silence indistinguishable from all-clear.
        raise TranscriptUnavailable("its transcript file is there but "
                                    "nothing in it could be read")
    return lines
