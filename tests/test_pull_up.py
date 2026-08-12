"""Fleet.pull_up — transcript resolution for the sessions Marvin cannot see
through an owned SDK stream (spec §12, M4).

Three previously-unreachable cases: a DETACHED worker (handed to a terminal —
the moment the owner most needs the pane), a restart ghost (the deque died
with the old process; the file did not), and a session Marvin never owned
(refused honestly — nothing binds an unknown session id to a directory except
a bearer any worker holds).

The hook's `transcript_path` field is UNTRUSTED, attacker-influenceable input
and is never opened: the location is DERIVED from Marvin's own records
(worktree path + session id) under a pinned transcripts root."""
import asyncio
import json

from server.fleet_state import DETACHED
from server.transcripts import project_dir_name
from tests.test_fleet import ResultMessage, cleanup, make_fleet, repo
from tests.test_transcripts import (SID, SID2, rec_assistant_text,
                                    rec_assistant_tool_use, rec_user,
                                    write_transcript)


async def detached_fleet(tmp_path, monkeypatch, sid=SID):
    """A spawned worker, its real session id learned, handed to a terminal."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    fleet.transcripts_root = tmp_path / "claude_projects"
    async def fake_terminal(cmd):
        pass
    fleet._open_terminal = fake_terminal
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    clients[0].stream.put_nowait(ResultMessage(session_id=sid))
    await asyncio.sleep(0.05)
    result = await fleet.handoff(path)
    assert result["ok"] is True
    assert fleet.workers[0].machine.base == DETACHED
    return fleet, bus, path


def seed_disk(fleet, cwd, sid, texts):
    return write_transcript(fleet.transcripts_root, cwd, sid,
                            [rec_user(cwd, texts[0], sid)]
                            + [rec_assistant_text(cwd, t, sid)
                               for t in texts[1:]])


# ---------- DETACHED: the most valuable case ----------
async def test_pull_up_serves_a_detached_worker_from_disk(tmp_path, monkeypatch):
    fleet, bus, path = await detached_fleet(tmp_path, monkeypatch)
    wt = fleet.workers[0].worktree.path
    seed_disk(fleet, wt, SID, ["fix the tests", "on it — running pytest now"])
    view = await fleet.pull_up(path)
    assert view["source"] == "disk"
    assert [l["text"] for l in view["lines"]] == [
        "fix the tests", "on it — running pytest now"]
    assert "detached" in view["spoken"]
    assert "disk" in view["spoken"]           # never claims a live stream


async def test_a_missing_disk_transcript_is_spoken_not_an_empty_pane(
        tmp_path, monkeypatch):
    fleet, bus, path = await detached_fleet(tmp_path, monkeypatch)
    view = await fleet.pull_up(path)          # nothing seeded on disk
    assert view["lines"] == []
    assert view["note"]                       # the pane says WHY it is empty
    assert "can't show" in view["spoken"]


async def test_pull_up_still_serves_the_live_deque_for_an_owned_worker(
        tmp_path, monkeypatch):
    from tests.test_fleet import AssistantMessage
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    fleet.transcripts_root = tmp_path / "claude_projects"
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    clients[0].stream.put_nowait(AssistantMessage("hello from the stream"))
    await asyncio.sleep(0.05)
    try:
        view = await fleet.pull_up(path)
        assert view["source"] == "memory"
        assert view["lines"][-1]["text"] == "hello from the stream"
        assert view["spoken"] == fleet.one_breath(path)
    finally:
        await cleanup(fleet)


async def test_an_empty_live_pane_carries_a_note_never_silence(
        tmp_path, monkeypatch):
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    fleet.transcripts_root = tmp_path / "claude_projects"
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    try:
        view = await fleet.pull_up(path)
        assert view["lines"] == [] and view["note"]
    finally:
        await cleanup(fleet)


# ---------- the terminal session after handoff ----------
async def test_a_detached_hook_teaches_the_terminal_session_id(
        tmp_path, monkeypatch):
    """`claude --resume` may mint a NEW session id; its hooks keep POSTing
    from the same worktree. Pull-up must follow the freshest file."""
    fleet, bus, path = await detached_fleet(tmp_path, monkeypatch)
    w = fleet.workers[0]
    fleet.handle_hook({"hook_event_name": "PostToolUse", "session_id": SID2,
                       "cwd": w.worktree.path,
                       "transcript_path": "/ignored/everywhere.jsonl"})
    assert w.terminal_session_id == SID2
    seed_disk(fleet, w.worktree.path, SID, ["old", "pre-handoff tail"])
    seed_disk(fleet, w.worktree.path, SID2, ["new", "terminal is driving"])
    view = await fleet.pull_up(path)
    assert view["lines"][-1]["text"] == "terminal is driving"


async def test_a_non_uuid_hook_sid_is_never_learned(tmp_path, monkeypatch):
    fleet, bus, path = await detached_fleet(tmp_path, monkeypatch)
    w = fleet.workers[0]
    fleet.handle_hook({"hook_event_name": "PostToolUse",
                       "session_id": "../../.ssh/id_rsa",
                       "cwd": w.worktree.path})
    assert w.terminal_session_id is None


async def test_the_terminal_sid_falls_back_to_the_handoff_sid(
        tmp_path, monkeypatch):
    fleet, bus, path = await detached_fleet(tmp_path, monkeypatch)
    w = fleet.workers[0]
    fleet.handle_hook({"hook_event_name": "PostToolUse", "session_id": SID2,
                       "cwd": w.worktree.path})
    seed_disk(fleet, w.worktree.path, SID, ["only", "the handoff file exists"])
    view = await fleet.pull_up(path)
    assert view["lines"][-1]["text"] == "the handoff file exists"


# ---------- restart ghosts ----------
async def test_pull_up_serves_a_detached_restart_ghost_from_disk(
        tmp_path, monkeypatch):
    fleet, bus, path = await detached_fleet(tmp_path, monkeypatch)
    wt = fleet.workers[0].worktree.path
    seed_disk(fleet, wt, SID, ["survives restarts", "still on disk"])
    await fleet.close_all()
    # A new boot: make_fleet builds a fresh FleetLog on the SAME path.
    fleet2, _, _, _ = make_fleet(tmp_path, monkeypatch)
    fleet2.transcripts_root = fleet.transcripts_root
    reports = fleet2.recover()
    assert reports and reports[0]["state"] == DETACHED
    view = await fleet2.pull_up(path)
    assert view["source"] == "disk"
    assert [l["text"] for l in view["lines"]] == [
        "survives restarts", "still on disk"]
    assert "disk" in view["spoken"]


async def test_pull_up_serves_an_interrupted_ghost_from_disk(
        tmp_path, monkeypatch):
    """Marvin crashed mid-task: the deque is gone, the file is not."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    fleet.transcripts_root = tmp_path / "claude_projects"
    path = repo(tmp_path)
    await fleet.spawn("soccer", path, "task")
    clients[0].stream.put_nowait(ResultMessage(session_id=SID))
    await asyncio.sleep(0.05)
    wt = fleet.workers[0].worktree.path
    seed_disk(fleet, wt, SID, ["was working", "when the server died"])
    await fleet.close_all()                      # no session_end — spec §5
    fleet2, _, _, _ = make_fleet(tmp_path, monkeypatch)
    fleet2.transcripts_root = fleet.transcripts_root
    reports = fleet2.recover()
    assert reports and reports[0]["interrupted"] is True
    view = await fleet2.pull_up(path)
    assert view["lines"][-1]["text"] == "when the server died"
    assert "disk" in view["spoken"]


async def test_a_ghost_without_a_session_id_is_refused_honestly(
        tmp_path, monkeypatch):
    fleet, _, _, _ = make_fleet(tmp_path, monkeypatch)
    fleet.transcripts_root = tmp_path / "claude_projects"
    fleet.ghosts = [{"worker": "w1", "project": "soccer", "path": "/p/soccer",
                     "state": "UNKNOWN", "task": "t", "worktree": "/wt/x",
                     "session_id": "", "interrupted": True}]
    view = await fleet.pull_up("/p/soccer")
    assert view["lines"] == [] and view["note"]
    assert "session id" in view["spoken"]


# ---------- sessions Marvin never owned ----------
async def test_pull_up_refuses_a_session_it_never_owned(tmp_path, monkeypatch):
    fleet, _, _, _ = make_fleet(tmp_path, monkeypatch)
    fleet.transcripts_root = tmp_path / "claude_projects"
    view = await fleet.pull_up("/somewhere/marvin/never/ran")
    assert view["lines"] == [] and view["source"] == "none"
    assert "Nothing is running there" in view["spoken"]


async def test_a_hostile_hook_transcript_path_is_never_opened(
        tmp_path, monkeypatch):
    """The field every hook carries is attacker-influenceable. Even when it
    points at a perfectly readable, perfectly valid transcript file, nothing
    may open it: resolution derives from Marvin's own records only."""
    fleet, bus, path = await detached_fleet(tmp_path, monkeypatch)
    w = fleet.workers[0]
    bait = tmp_path / "outside" / "bait.jsonl"
    bait.parent.mkdir()
    bait.write_text(json.dumps(
        rec_user("/w/t", "SECRET-VAULT-CONTENT", SID)) + "\n")
    fleet.handle_hook({"hook_event_name": "PostToolUse",
                       "session_id": w.session_id, "cwd": w.worktree.path,
                       "transcript_path": str(bait)})
    view = await fleet.pull_up(path)
    assert all("SECRET-VAULT-CONTENT" not in l["text"]
               for l in view["lines"])
    # No file on the derived path → the honest refusal, not the bait's bytes.
    assert view["lines"] == [] and view["note"]


async def test_an_unknown_session_hook_registers_no_tile_and_reads_nothing(
        tmp_path, monkeypatch):
    """Unknown-session auto-registration is DEFERRED (see the M4 report):
    anything holding the bearer could conjure tiles. The hook is still
    surfaced on the console, exactly as before."""
    fleet, bus, router, clients = make_fleet(tmp_path, monkeypatch)
    fleet.transcripts_root = tmp_path / "claude_projects"
    cid, q = bus.subscribe()
    fleet.handle_hook({"hook_event_name": "SessionStart", "session_id": SID,
                       "cwd": "/some/unknown/dir",
                       "transcript_path": str(tmp_path / "anything.jsonl")})
    ev = q.get_nowait()
    assert ev["type"] == "fleet.unknown_session"
    assert fleet.workers == [] and fleet.ghosts == []
    view = await fleet.pull_up("/some/unknown/dir")
    assert view["lines"] == [] and view["source"] == "none"
