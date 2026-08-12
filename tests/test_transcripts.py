"""server/transcripts.py — read-only resolution of a session's on-disk
transcript (spec §12, M4).

The fixtures here are shaped from a REAL Claude Code transcript
(~/.claude/projects/-Users-likerun-marlowe-state-worktrees-probe-read-the-
source-files-an-20260812-161654/df1eae53-….jsonl, the session detached on
2026-08-12): queue-operation and attachment records, user prompts with STRING
content, assistant records with text and tool_use blocks, tool_result records
with LIST content, and the last-prompt/mode trailer records. Fake-shaped
fixtures are how two features in this project shipped broken.

The dangerous direction is tested hardest: everything read must resolve inside
the transcripts root, through no symlink, into a regular file, bounded in
size — and every refusal must be a spoken reason, never an empty pane."""
import json
import os

import pytest

from server.transcripts import (SESSION_ID_RE, TranscriptUnavailable,
                                derive_transcript_path, project_dir_name,
                                read_session_tail)

SID = "df1eae53-f390-4917-a0d1-b068a9698965"
SID2 = "8e1e8795-cd99-442f-b73d-a460d11bb688"


# ---------- realistic record builders (shapes copied from the real file) ----
def rec_user(cwd, text, sid=SID):
    return {"parentUuid": None, "isSidechain": False, "userType": "external",
            "cwd": cwd, "sessionId": sid, "version": "2.1.19",
            "gitBranch": "marvin/read-the-source-files-an-20260812-161654",
            "type": "user", "message": {"role": "user", "content": text},
            "uuid": "e3fd3566-939b-4a94-91ea-dbe004f02e94",
            "timestamp": "2026-08-12T08:16:57.001Z",
            "permissionMode": "default", "promptId": "p-1",
            "promptSource": "cli"}


def rec_assistant_text(cwd, text, sid=SID):
    return {"parentUuid": "e3fd3566-939b-4a94-91ea-dbe004f02e94",
            "isSidechain": False, "userType": "external", "cwd": cwd,
            "sessionId": sid, "version": "2.1.19",
            "gitBranch": "marvin/read-the-source-files-an-20260812-161654",
            "type": "assistant", "requestId": "req_011CdxXLMSj6",
            "effort": "high",
            "message": {"model": "claude-opus-5",
                        "id": "msg_011CdxXLMSj6tt593vgg9Qah",
                        "type": "message", "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                        "stop_reason": "tool_use", "stop_sequence": None},
            "uuid": "a1b2c3d4-0000-4a94-91ea-dbe004f02e94",
            "timestamp": "2026-08-12T08:17:02.100Z"}


def rec_assistant_tool_use(cwd, name, tool_input, sid=SID):
    rec = rec_assistant_text(cwd, "", sid)
    rec["message"]["content"] = [{"type": "tool_use",
                                  "id": "toolu_01ChcU8YCkPKTsCRQ6PMuapm",
                                  "name": name, "input": tool_input,
                                  "caller": {"type": "direct"}}]
    return rec


def rec_tool_result(cwd, text, sid=SID):
    rec = rec_user(cwd, "", sid)
    rec["message"]["content"] = [{"type": "tool_result",
                                  "tool_use_id": "toolu_01ChcU8YCkPKTsCRQ6PMuapm",
                                  "content": text, "is_error": True}]
    rec["toolUseResult"] = text
    return rec


def rec_noise(sid=SID):
    """The records the pane never shows: queue ops, attachments, trailers."""
    return [
        {"type": "queue-operation", "operation": "enqueue",
         "timestamp": "2026-08-12T08:16:56.989Z", "sessionId": sid,
         "content": "Read the source files and summarize what this project does."},
        {"type": "attachment", "attachment": {"type": "todo"}, "cwd": "/x",
         "sessionId": sid, "uuid": "u-1", "timestamp": "t", "parentUuid": None,
         "isSidechain": False, "userType": "external", "version": "2.1.19",
         "gitBranch": "b", "entrypoint": "cli"},
        {"type": "last-prompt", "lastPrompt": "Read the source files…",
         "leafUuid": "e3fd3566-939b-4a94-91ea-dbe004f02e94", "sessionId": sid},
        {"type": "mode", "mode": "normal", "sessionId": sid},
    ]


def write_transcript(root, cwd, sid, records):
    d = root / project_dir_name(str(cwd))
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sid}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in records),
                 encoding="utf-8")
    return p


# ---------- derivation ----------
def test_project_dir_name_matches_the_real_disk_layout():
    # Copied verbatim from ~/.claude/projects on this machine: the directory
    # Claude Code created for the session that was detached on 2026-08-12.
    cwd = ("/Users/likerun/marlowe/state/worktrees/"
           "probe-read-the-source-files-an-20260812-161654")
    assert project_dir_name(cwd) == (
        "-Users-likerun-marlowe-state-worktrees-"
        "probe-read-the-source-files-an-20260812-161654")
    # Spaces, tildes and dots all reduce to dashes (the vault's own dir).
    assert project_dir_name(
        "/Users/likerun/Library/Mobile Documents/iCloud~md~obsidian"
        "/Documents/KEKE LI") == (
        "-Users-likerun-Library-Mobile-Documents-iCloud-md-obsidian"
        "-Documents-KEKE-LI")


def test_derivation_is_root_slash_project_dir_slash_sid(tmp_path):
    p = derive_transcript_path("/w/t", SID, tmp_path)
    assert p == tmp_path / "-w-t" / f"{SID}.jsonl"


def test_derivation_refuses_a_non_uuid_session_id(tmp_path):
    # A traversal-shaped id must never become path components.
    for sid in ("../../../Users/likerun/.ssh/id_rsa", "sess-42", "",
                "df1eae53-f390-4917-a0d1-b068a9698965.jsonl",
                "df1eae53/f390/4917/a0d1/b068a9698965"):
        with pytest.raises(TranscriptUnavailable):
            derive_transcript_path("/w/t", sid, tmp_path)
        assert not SESSION_ID_RE.match(sid)


def test_derivation_refuses_a_relative_or_empty_cwd(tmp_path):
    for cwd in ("", "relative/dir", "   "):
        with pytest.raises(TranscriptUnavailable):
            derive_transcript_path(cwd, SID, tmp_path)


# ---------- the happy path, from realistic records ----------
def test_reads_the_tail_of_a_realistic_transcript(tmp_path):
    cwd = "/Users/likerun/marlowe/state/worktrees/probe-x-20260812-161654"
    noise = rec_noise()
    write_transcript(tmp_path, cwd, SID, [
        noise[0], noise[1],
        rec_user(cwd, "Read the source files and summarize what this project does."),
        rec_assistant_text(cwd, "I'll start by exploring the project structure."),
        rec_assistant_tool_use(cwd, "Bash",
                               {"command": "ls -la /Users/likerun/probe",
                                "description": "List project root"}),
        rec_tool_result(cwd, "total 24\ndrwxr-xr-x  5 likerun"),
        noise[2], noise[3],
    ])
    lines = read_session_tail(cwd, SID, tmp_path)
    assert [(l["who"], l["text"]) for l in lines] == [
        ("user", "Read the source files and summarize what this project does."),
        ("worker", "I'll start by exploring the project structure."),
        ("worker", "[Bash] ls -la /Users/likerun/probe"),
    ]


def test_an_empty_transcript_reads_as_empty_not_as_an_error(tmp_path):
    cwd = "/w/t"
    write_transcript(tmp_path, cwd, SID, [])
    assert read_session_tail(cwd, SID, tmp_path) == []


def test_a_missing_file_is_an_honest_refusal(tmp_path):
    with pytest.raises(TranscriptUnavailable) as e:
        read_session_tail("/w/t", SID, tmp_path)
    assert "not on disk" in str(e.value)


# ---------- the dangerous direction ----------
def test_a_symlinked_transcript_file_is_refused(tmp_path):
    """The final component is a symlink to a secret outside the root — the
    exact shape of the containment defect this codebase already shipped once
    (a check that followed symlinks and pinned nothing)."""
    secret = tmp_path / "outside" / "id_rsa"
    secret.parent.mkdir()
    secret.write_text(json.dumps(rec_user("/w/t", "PRIVATE KEY MATERIAL")) + "\n")
    root = tmp_path / "projects"
    d = root / project_dir_name("/w/t")
    d.mkdir(parents=True)
    (d / f"{SID}.jsonl").symlink_to(secret)
    with pytest.raises(TranscriptUnavailable) as e:
        read_session_tail("/w/t", SID, root)
    assert "PRIVATE KEY MATERIAL" not in str(e.value)


def test_a_symlinked_project_directory_is_refused(tmp_path):
    """A symlink one level up must fail the same way: the containment is on
    the RESOLVED path, not on the string."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / f"{SID}.jsonl").write_text(
        json.dumps(rec_user("/w/t", "secret")) + "\n")
    root = tmp_path / "projects"
    root.mkdir()
    (root / project_dir_name("/w/t")).symlink_to(outside)
    with pytest.raises(TranscriptUnavailable):
        read_session_tail("/w/t", SID, root)


def test_a_symlink_inside_the_root_is_still_refused(tmp_path):
    """Even a link that stays under the root is refused: the file opened must
    be the file derived, byte for byte — no indirection at all."""
    root = tmp_path / "projects"
    cwd_a, cwd_b = "/w/a", "/w/b"
    real = write_transcript(root, cwd_b, SID2,
                            [rec_user(cwd_b, "someone else's session", SID2)])
    d = root / project_dir_name(cwd_a)
    d.mkdir(parents=True)
    (d / f"{SID}.jsonl").symlink_to(real)
    with pytest.raises(TranscriptUnavailable):
        read_session_tail(cwd_a, SID, root)


def test_a_fifo_is_refused_without_blocking(tmp_path):
    """A non-regular file must be refused — and the open must not block on a
    FIFO with no writer, which would wedge the reader thread forever."""
    root = tmp_path / "projects"
    d = root / project_dir_name("/w/t")
    d.mkdir(parents=True)
    os.mkfifo(d / f"{SID}.jsonl")
    with pytest.raises(TranscriptUnavailable):
        read_session_tail("/w/t", SID, root)


def test_the_tail_cap_bounds_what_is_read(tmp_path):
    cwd = "/w/t"
    head = rec_user(cwd, "HEAD-SENTINEL " + "x" * 512)
    tail = [rec_assistant_text(cwd, f"tail line {i}") for i in range(20)]
    write_transcript(tmp_path, cwd, SID, [head] + tail)
    # 20 records ≈ 13KB: the 8KB byte cap must cut the head record off even
    # though all 21 lines fit inside the `keep` LINE cap — the caps are
    # separate defenses, and a whole-file read must fail here (the first
    # version of this test let `keep` evict the sentinel and mask exactly
    # that mutation).
    lines = read_session_tail(cwd, SID, tmp_path, tail_bytes=8 * 1024)
    assert lines, "the capped read must still render the tail"
    assert all("HEAD-SENTINEL" not in l["text"] for l in lines)
    assert lines[-1]["text"] == "tail line 19"    # newest survives, always


def test_the_keep_cap_bounds_how_many_lines_render(tmp_path):
    cwd = "/w/t"
    write_transcript(tmp_path, cwd, SID,
                     [rec_assistant_text(cwd, f"line {i}") for i in range(250)])
    lines = read_session_tail(cwd, SID, tmp_path)
    assert len(lines) == 200                      # TRANSCRIPT_KEEP parity
    assert lines[-1]["text"] == "line 249"        # newest survives, always


def test_each_rendered_line_is_capped(tmp_path):
    cwd = "/w/t"
    write_transcript(tmp_path, cwd, SID,
                     [rec_assistant_text(cwd, "y" * 50_000)])
    lines = read_session_tail(cwd, SID, tmp_path)
    assert len(lines) == 1 and len(lines[0]["text"]) <= 2000


def test_malformed_lines_are_skipped_but_the_rest_still_renders(tmp_path):
    cwd = "/w/t"
    p = write_transcript(tmp_path, cwd, SID, [rec_user(cwd, "kept")])
    with p.open("a", encoding="utf-8") as f:
        f.write("{this is not json\n")
        f.write(json.dumps(rec_assistant_text(cwd, "also kept")) + "\n")
    lines = read_session_tail(cwd, SID, tmp_path)
    assert [l["text"] for l in lines] == ["kept", "also kept"]


def test_a_fully_unreadable_file_refuses_instead_of_showing_nothing(tmp_path):
    """Garbage in, an honest sentence out — an empty pane over a file full of
    bytes would be silence indistinguishable from all-clear."""
    root = tmp_path / "projects"
    d = root / project_dir_name("/w/t")
    d.mkdir(parents=True)
    (d / f"{SID}.jsonl").write_bytes(b"\x00\xff garbage\nnot json either\n")
    with pytest.raises(TranscriptUnavailable):
        read_session_tail("/w/t", SID, root)
