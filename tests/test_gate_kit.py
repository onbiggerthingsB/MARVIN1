"""The M3P2 gate kit's pure logic: beat matching, the preflight's decision
rules, and the observer's torn/rotated log handling.

Everything here runs against temp dirs and fixture text. Nothing touches the
real `state/`, starts a server, or spawns a worker — the two scripts under
test are excluded from collection by `testpaths = ["tests"]`, and this file
imports them as plain modules.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gate_observer as obs        # noqa: E402
import gate_preflight as pre       # noqa: E402

from server.fleet_log import FleetLog       # noqa: E402
from server.fleet_state import DETACHED     # noqa: E402
from server.registry import Project         # noqa: E402


# ------------------------------------------------------------ log reading ---
def write_log(path: Path, events) -> FleetLog:
    """A real FleetLog writes the fixture, so the checksums under test are the
    ones the server actually produces."""
    log = FleetLog(path)
    for kind, data in events:
        log.append(kind, data)
    return log


def test_parse_fleet_line_accepts_a_real_record(tmp_path):
    write_log(tmp_path / "fleet.jsonl", [("spawned", {"worker": "w1"})])
    line = (tmp_path / "fleet.jsonl").read_text().splitlines()[0]
    rec, status = obs.parse_fleet_line(line)
    assert status == "ok" and rec["kind"] == "spawned"


@pytest.mark.parametrize("line,reason", [
    ('{"v":1,"seq":1', "not JSON"),
    ('[1,2,3]', "not an object"),
    ('{"v":99,"seq":1,"ts":0,"kind":"x","data":{},"sum":"z"}', "schema"),
    ('{"v":1,"seq":1,"ts":0.0,"kind":"x","data":{},"sum":"deadbeef"}',
     "checksum mismatch"),
    ('{"v":1,"seq":1}', "missing fields"),
])
def test_parse_fleet_line_rejects_damage_without_raising(line, reason):
    rec, status = obs.parse_fleet_line(line)
    assert rec is None and reason.split()[0] in status


def test_read_fleet_log_keeps_going_past_a_torn_line(tmp_path):
    """Unlike FleetLog.replay, which stops at the first bad line: the observer
    is reporting, not deciding what is alive, and stopping would blind it."""
    p = tmp_path / "fleet.jsonl"
    write_log(p, [("spawned", {"worker": "w1"}),
                  ("turn_done", {"worker": "w1"})])
    lines = p.read_text().splitlines()
    p.write_text(lines[0] + "\n" + '{"v":1,"seq":2,"ts":0,"kin\n'
                 + lines[1] + "\n")
    records, damaged = obs.read_fleet_log(p)
    assert damaged == 1
    assert [r["kind"] for r in records] == ["spawned", "turn_done"]


def test_read_fleet_log_on_a_missing_file_is_empty_not_an_error(tmp_path):
    assert obs.read_fleet_log(tmp_path / "nope.jsonl") == ([], 0)


def test_fold_workers_lets_later_records_win_key_by_key(tmp_path):
    p = tmp_path / "fleet.jsonl"
    write_log(p, [
        ("spawned", {"worker": "w1", "project": "alethic", "state": "IDLE",
                     "worktree": "/wt/1"}),
        ("detached", {"worker": "w1", "state": DETACHED, "session_id": "s-9"}),
        ("spawned", {"worker": "w2", "project": "composed", "state": "IDLE"}),
    ])
    records, _ = obs.read_fleet_log(p)
    folded = obs.fold_workers(records)
    assert folded["w1"]["state"] == DETACHED
    assert folded["w1"]["project"] == "alethic"      # carried from the spawn
    assert folded["w1"]["worktree"] == "/wt/1"
    assert folded["w1"]["session_id"] == "s-9"
    assert set(folded) == {"w1", "w2"}


def test_fold_workers_ignores_records_with_no_worker():
    assert obs.fold_workers([{"data": {"reason": "boom"}}]) == {}


# ----------------------------------------------------------------- tailer ---
def test_tailer_waits_for_a_file_that_does_not_exist_yet(tmp_path):
    t = obs.Tailer(tmp_path / "later.log", "later.log")
    assert t.poll() == ([], None)
    (tmp_path / "later.log").write_text("first\n")
    lines, note = t.poll()
    assert lines == ["first"] and "appeared" in note


def test_tailer_holds_a_partial_line_until_its_newline_arrives(tmp_path):
    p = tmp_path / "a.log"
    p.write_text("one\ntw")
    t = obs.Tailer(p, "a.log")
    assert t.poll()[0] == ["one"]
    with open(p, "a") as f:
        f.write("o\n")
    assert t.poll()[0] == ["two"]


def test_tailer_follows_a_rotation_and_drains_the_old_handle(tmp_path):
    """FleetLog.snapshot renames the log aside — beat 9 does exactly this."""
    p = tmp_path / "fleet.jsonl"
    p.write_text("a\n")
    t = obs.Tailer(p, "fleet.jsonl")
    assert t.poll()[0] == ["a"]
    with open(p, "a") as f:                    # written just before the rename
        f.write("b\n")
    p.replace(tmp_path / "fleet.jsonl.1")
    p.write_text("c\n")
    lines, note = t.poll()
    assert lines == ["b", "c"]
    assert "ROTATED" in note and t.rotations == 1


def test_tailer_reports_a_torn_tail_when_the_file_is_moved_aside(tmp_path):
    """A crash mid-append leaves a fragment; the next boot's FleetLog renames
    the damaged file to `.torn-<ts>` and writes a fresh one."""
    p = tmp_path / "fleet.jsonl"
    p.write_text("good\npart")
    t = obs.Tailer(p, "fleet.jsonl")
    assert t.poll()[0] == ["good"]
    p.replace(tmp_path / "fleet.jsonl.torn-1")
    p.write_text("fresh\n")
    lines, note = t.poll()
    assert lines == ["part", "fresh"]
    assert "torn tail" in note


def test_tailer_treats_truncation_in_place_as_a_rotation(tmp_path):
    p = tmp_path / "s.log"
    p.write_text("aaaa\nbbbb\n")
    t = obs.Tailer(p, "s.log")
    assert t.poll()[0] == ["aaaa", "bbbb"]
    p.write_text("z\n")                        # same inode, smaller
    lines, note = t.poll()
    assert lines == ["z"] and note is not None


def test_tailer_never_dies_on_undecodable_bytes(tmp_path):
    p = tmp_path / "b.log"
    p.write_bytes(b"\xff\xfe not utf8\n")
    t = obs.Tailer(p, "b.log")
    lines, _ = t.poll()
    assert len(lines) == 1                     # replaced, not raised


# ------------------------------------------------------------ beat mapping --
def board_and_correlator():
    b = obs.BeatBoard()
    return b, obs.Correlator(b)


def rec(kind, data, seq=1, ts=None):
    return {"v": 1, "seq": seq, "ts": ts if ts is not None else time.time(),
            "kind": kind, "data": data}


def test_spawn_record_is_beat_2():
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("spawned", {
        "worker": "w1", "project": "alethic", "state": "IDLE_AT_PROMPT",
        "worktree": "/wt/1", "branch": "marlowe/x-1", "base_commit": "abc1234"}))
    assert board.evidence[2] and not board.evidence[7]


def test_approval_pair_is_beat_3():
    """A GRANTED approval credits beat 3 with both halves: the card that was
    raised and the grant that resolved it. The fixture carries exactly what
    fleet.py's _apply writes: permission_done always has `approved`."""
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("permission_wait", {
        "worker": "w1", "state": "WAITING_PERMISSION", "nonce": "n1"}))
    corr.on_fleet_record(rec("permission_done", {
        "worker": "w1", "state": "ACTIVE_TURN", "nonce": "n1",
        "approved": True}, 2))
    assert len(board.evidence[3]) == 2


def test_a_denied_approval_is_not_beat_3():
    """Beat 3 is the CONSENT beat: card raised → approved BY VOICE → worker
    finishes. Keke saying NO is a real event worth a visible line, but it
    cannot satisfy a beat whose whole point is that consent was granted."""
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("permission_wait", {
        "worker": "w1", "project": "alethic",
        "state": "WAITING_PERMISSION", "nonce": "n1"}))
    corr.on_fleet_record(rec("permission_done", {
        "worker": "w1", "project": "alethic", "state": "ACTIVE_TURN",
        "nonce": "n1", "approved": False}, 2))
    corr.on_fleet_record(rec("turn_done", {
        "worker": "w1", "project": "alethic", "state": "IDLE_AT_PROMPT"}, 3))
    assert not board.evidence[3]
    assert any("denied" in d for d in corr.deviations)
    text = "\n".join(board.summary_lines())
    assert "[NONE    ] 3." in text


def test_a_cancelled_approval_is_not_beat_3():
    """fleet.py's CancelledError path writes approved=False, cancelled=True —
    the SDK tore the callback down, nobody consented to anything."""
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("permission_wait", {
        "worker": "w1", "state": "WAITING_PERMISSION", "nonce": "n1"}))
    corr.on_fleet_record(rec("permission_done", {
        "worker": "w1", "state": "ACTIVE_TURN", "nonce": "n1",
        "approved": False, "cancelled": True}, 2))
    assert not board.evidence[3]
    assert any("cancelled" in d for d in corr.deviations)


def test_an_expired_approval_is_not_beat_3():
    """The TTL path writes the same record as a denial (approved=False, no
    cancelled flag): an approval nobody answered is not consent either."""
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("permission_done", {
        "worker": "w1", "state": "ACTIVE_TURN", "nonce": "n1",
        "approved": False}, 2))
    assert not board.evidence[3]


def test_a_grant_after_a_denial_still_credits_beat_3():
    """Saying no to a scary request and yes to the next one is the checklist's
    own advice — the denial must not poison the later, genuine grant."""
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("permission_wait", {
        "worker": "w1", "state": "WAITING_PERMISSION", "nonce": "n1"}))
    corr.on_fleet_record(rec("permission_done", {
        "worker": "w1", "state": "ACTIVE_TURN", "nonce": "n1",
        "approved": False}, 2))
    corr.on_fleet_record(rec("permission_wait", {
        "worker": "w1", "state": "WAITING_PERMISSION", "nonce": "n2"}, 3))
    corr.on_fleet_record(rec("permission_done", {
        "worker": "w1", "state": "ACTIVE_TURN", "nonce": "n2",
        "approved": True}, 4))
    corr.on_fleet_record(rec("turn_done", {"worker": "w1",
                                           "state": "IDLE_AT_PROMPT"}, 5))
    assert len(board.evidence[3]) == 3      # card + grant + finished turn


def test_turn_done_only_counts_for_beat_3_after_an_approval():
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("turn_done", {"worker": "w1", "state": "IDLE"}))
    assert not board.evidence[3]


def test_detached_record_is_beat_6_and_arms_beat_7():
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("detached", {
        "worker": "w1", "project": "alethic", "state": DETACHED,
        "session_id": "sess-abcdef123456"}))
    assert board.evidence[6]
    assert corr.detached["w1"]["session_id"] == "sess-abcdef123456"
    assert corr.first_detach_ts is not None


def test_a_detach_with_no_session_id_is_a_deviation():
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("detached", {"worker": "w1", "state": DETACHED,
                                          "session_id": ""}))
    assert any("session id" in d for d in corr.deviations)


def test_records_reaching_a_detached_worker_are_beat_7():
    """The whole point of beat 7: after the handoff Marlowe holds no stream for
    that session, so only a /hooks POST can still move the tile."""
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("detached", {"worker": "w1", "state": DETACHED,
                                          "session_id": "s1"}))
    corr.on_fleet_record(rec("activity", {"worker": "w1", "state": DETACHED}, 2))
    corr.on_fleet_record(rec("turn_done", {"worker": "w1", "state": DETACHED}, 3))
    assert len(board.evidence[7]) == 2
    assert corr.post_detach_records == 2


def test_activity_on_a_live_worker_is_not_beat_7():
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("activity", {"worker": "w1",
                                          "state": "ACTIVE_TURN"}))
    assert not board.evidence[7]


def test_handoff_failed_and_lost_are_deviations():
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("lost", {"worker": "w1", "state": "UNKNOWN",
                                      "reason": "handoff failed: boom"}))
    assert corr.deviations and "boom" in corr.deviations[0]


def test_post_hooks_before_a_detach_is_recorded_but_not_beat_7():
    board, corr = board_and_correlator()
    corr.on_server_line('INFO: 127.0.0.1:5 - "POST /hooks HTTP/1.1" 200 OK',
                        time.time())
    assert corr.hooks_total == 1
    assert corr.hooks_after_detach == 0
    assert not board.evidence[7]


def test_post_hooks_after_a_detach_is_beat_7():
    board, corr = board_and_correlator()
    corr.on_fleet_record(rec("detached", {"worker": "w1", "state": DETACHED,
                                          "session_id": "s1"}))
    out = corr.on_server_line('INFO: - "POST /hooks HTTP/1.1" 200 OK',
                              time.time())
    assert corr.hooks_after_detach == 1
    assert board.evidence[7]
    # The access log carries no session id; the attribution must SAY it is by
    # time rather than imply the line stated it.
    assert any("correlated by time" in line for line in out)


def test_once_mode_never_credits_beat_7_from_the_access_log():
    """In --once the whole fleet log is pumped before the first server.log
    line, so first_detach_ts is always already set — and access-log lines
    carry no timestamps to order against it. Workers POST hooks from beat 2
    onward, so 'a detach exists somewhere in the log' proves nothing about
    THIS line. Lane A must refuse rather than degrade into a false claim."""
    board = obs.BeatBoard()
    corr = obs.Correlator(board, lane_a_ordered=False)
    corr.on_fleet_record(rec("detached", {"worker": "w1", "state": DETACHED,
                                          "session_id": "s1"}))
    out = corr.on_server_line('INFO: - "POST /hooks HTTP/1.1" 200 OK',
                              time.time())
    assert corr.hooks_total == 1               # still counted, still printed
    assert corr.hooks_after_detach == 0        # but never claimed as ordered
    assert not board.evidence[7]
    assert any("lane B" in line for line in out)   # says who the authority is


def test_once_mode_lane_b_still_credits_beat_7():
    """Lane B is causally ordered by the records themselves (a DETACHED state
    can only follow the `detached` record), so --once keeps it."""
    board = obs.BeatBoard()
    corr = obs.Correlator(board, lane_a_ordered=False)
    corr.on_fleet_record(rec("detached", {"worker": "w1", "state": DETACHED,
                                          "session_id": "s1"}))
    corr.on_fleet_record(rec("activity", {"worker": "w1",
                                          "state": DETACHED}, 2))
    assert board.evidence[7]


def test_restart_after_a_shutdown_is_beat_9():
    board, corr = board_and_correlator()
    corr.on_server_line("INFO:     Started server process [1]", time.time())
    assert not board.evidence[9]               # first boot proves nothing
    corr.on_server_line("INFO:     Shutting down", time.time())
    corr.on_server_line("INFO:     Started server process [2]", time.time())
    assert board.evidence[9]


def test_registry_confirm_is_beat_1_and_a_pin_is_beat_8():
    board, corr = board_and_correlator()
    before = {"alethic": {"path": "/a", "confirmed": False,
                          "data_source": None}}
    after = {"alethic": {"path": "/a", "confirmed": True,
                         "data_source": None}}
    corr.on_registry(before, after)
    assert board.evidence[1] and not board.evidence[8]
    later = {"alethic": {"path": "/a", "confirmed": True,
                         "data_source": "/a/picks.sqlite"}}
    corr.on_registry(after, later)
    assert board.evidence[8]


def test_history_is_printed_but_credited_to_nothing():
    """state/server.log survives across runs and ends in the LAST run's
    shutdown block; a leftover fleet.jsonl carries the last run's detach. A
    crediting replay would hand out beats 6, 7 and 9 before the demo starts."""
    board = obs.BeatBoard()
    corr = obs.Correlator(board, crediting=False)
    out = corr.on_fleet_record(rec("detached", {
        "worker": "w1", "state": DETACHED, "session_id": "old"}))
    out += corr.on_fleet_record(rec("activity", {"worker": "w1",
                                                 "state": DETACHED}, 2))
    out += corr.on_server_line("INFO:     Shutting down", time.time())
    out += corr.on_server_line("INFO:     Started server process [2]",
                               time.time())
    out += corr.on_server_line('- "POST /hooks HTTP/1.1" 200 OK', time.time())
    assert all(not rows for rows in board.evidence.values())
    assert corr.detached == {} and corr.hooks_total == 0
    assert corr.saw_shutdown is False
    assert all("not credited" in line for line in out)


def test_crediting_resumes_cleanly_after_the_history_replay():
    board = obs.BeatBoard()
    corr = obs.Correlator(board, crediting=False)
    corr.on_fleet_record(rec("detached", {"worker": "w1", "state": DETACHED,
                                          "session_id": "old"}))
    corr.crediting = True
    corr.on_fleet_record(rec("activity", {"worker": "w1", "state": DETACHED}, 2))
    # the OLD detach was never recorded, so this is not beat-7 evidence
    assert not board.evidence[7]
    corr.on_fleet_record(rec("detached", {"worker": "w1", "state": DETACHED,
                                          "session_id": "new"}, 3))
    corr.on_fleet_record(rec("activity", {"worker": "w1", "state": DETACHED}, 4))
    assert board.evidence[6] and board.evidence[7]


def test_beats_4_and_5_are_reported_blind_not_failed():
    board = obs.BeatBoard()
    text = "\n".join(board.summary_lines())
    assert "[BLIND   ] 4." in text and "[BLIND   ] 5." in text
    assert "record 4 and 5 by hand" in text
    for n in (1, 2, 3, 6, 7, 8, 9):
        assert f"[NONE    ] {n}." in text


def test_beat_board_caps_its_evidence_list():
    board = obs.BeatBoard()
    assert board.note(2, "first") is True
    for i in range(20):
        assert board.note(2, f"more {i}") is False
    assert len(board.evidence[2]) == board.MAX_PER_BEAT + 1


# --------------------------------------------------------- registry reading --
def test_registry_snapshot_never_quarantines_a_broken_file(tmp_path):
    """Registry.load RENAMES an unusable file aside. An observer must not move
    the human's config during the demo — load_strict is the read-only door."""
    p = tmp_path / "projects.json"
    p.write_text("{ not json")
    assert obs.registry_snapshot(p) == {}
    assert p.exists()
    assert not list(tmp_path.glob("projects.json.corrupt-*"))


def test_registry_snapshot_reads_confirmations_and_pins(tmp_path):
    p = tmp_path / "projects.json"
    p.write_text(json.dumps({"schema_version": 1, "projects": [
        {"name": "alethic", "path": "/a", "confirmed": True},
        {"name": "quant", "path": "/q", "confirmed": True, "kind": "finance",
         "data_source": "/q/picks.sqlite"}]}))
    snap = obs.registry_snapshot(p)
    assert snap["quant"]["data_source"] == "/q/picks.sqlite"
    assert snap["alethic"]["confirmed"] is True


# --------------------------------------------------------------- preflight --
def test_parse_env_file_handles_comments_quotes_and_export():
    env = pre.parse_env_file(
        "# a comment\n"
        "\n"
        "HTTPS_PROXY=http://127.0.0.1:7890\n"
        "export NO_PROXY='localhost,127.0.0.1'\n"
        'ELEVENLABS_VOICE_ID="abc123"\n'
        "# DEEPGRAM_API_KEY=commented-out\n"
        "not an assignment\n")
    assert env == {"HTTPS_PROXY": "http://127.0.0.1:7890",
                   "NO_PROXY": "localhost,127.0.0.1",
                   "ELEVENLABS_VOICE_ID": "abc123"}


def test_merged_env_lets_the_dotenv_win_like_source_does():
    assert pre.merged_env({"A": "shell", "B": "shell"},
                          {"A": "file"}) == {"A": "file", "B": "shell"}


@pytest.mark.parametrize("url,expected", [
    ("http://127.0.0.1:7890", ("127.0.0.1", 7890)),
    ("127.0.0.1:1080", ("127.0.0.1", 1080)),
    ("http://proxy.local", ("proxy.local", 7890)),   # documented default
])
def test_proxy_endpoint_parses_what_the_env_names(url, expected):
    assert pre.proxy_endpoint({"HTTPS_PROXY": url}) == expected


def test_proxy_endpoint_is_none_when_nothing_is_set():
    assert pre.proxy_endpoint({}) is None


def test_check_proxy_env_reuses_proxy_problem_and_errors():
    c = pre.check_proxy_env({})
    assert c.level == pre.ERROR and "HTTPS_PROXY is not set" in c.detail
    c = pre.check_proxy_env({"HTTPS_PROXY": "http://127.0.0.1:7890",
                             "NO_PROXY": "localhost"})
    assert c.level == pre.ERROR and "NO_PROXY" in c.detail


def test_check_proxy_env_passes_a_correct_environment():
    assert pre.check_proxy_env({
        "HTTPS_PROXY": "http://127.0.0.1:7890",
        "NO_PROXY": "localhost,127.0.0.1"}).level == pre.OK


def test_skipping_the_proxy_check_is_a_warning_not_a_pass():
    c = pre.check_proxy_env({"MARLOWE_SKIP_PROXY_CHECK": "1"})
    assert c.level == pre.WARN and "403" in c.detail


def test_deepgram_is_a_blocker_and_elevenlabs_is_only_a_warning():
    checks = {c.name: c for c in pre.check_voice({})}
    assert checks["STT (Deepgram)"].level == pre.ERROR
    assert checks["TTS (ElevenLabs)"].level == pre.WARN
    assert "say" in checks["TTS (ElevenLabs)"].detail


def test_both_elevenlabs_keys_are_required_for_the_marlowe_voice():
    """SpeakEngine._eleven_enabled needs BOTH; one alone still falls back."""
    checks = {c.name: c for c in pre.check_voice(
        {"DEEPGRAM_API_KEY": "k", "ELEVENLABS_API_KEY": "k"})}
    assert checks["TTS (ElevenLabs)"].level == pre.WARN
    assert "ELEVENLABS_VOICE_ID" in checks["TTS (ElevenLabs)"].detail
    checks = {c.name: c for c in pre.check_voice(
        {"DEEPGRAM_API_KEY": "k", "ELEVENLABS_API_KEY": "k",
         "ELEVENLABS_VOICE_ID": "v"})}
    assert checks["STT (Deepgram)"].level == pre.OK
    assert checks["TTS (ElevenLabs)"].level == pre.OK


def test_forcing_the_say_voice_is_reported_as_a_warning():
    checks = {c.name: c for c in pre.check_voice(
        {"DEEPGRAM_API_KEY": "k", "ELEVENLABS_API_KEY": "k",
         "ELEVENLABS_VOICE_ID": "v", "MARLOWE_VOICE": "say"})}
    assert checks["TTS (ElevenLabs)"].level == pre.WARN


def test_access_log_check_flags_the_launcher_as_shipped():
    c = pre.access_log_check("uvicorn ... --port 7777 --no-access-log\n")
    assert c.level == pre.ERROR
    assert "--access-log" in c.fix and "Do NOT edit bin/marlowe" in c.fix
    assert pre.access_log_check("uvicorn ... --port 7777\n").level == pre.OK
    assert pre.access_log_check("").level == pre.WARN


def test_discovery_wiring_scan_ignores_onboardings_own_definition():
    assert pre.discovery_wired(
        {"onboarding.py": "async def ask_next(self):"}) is False
    assert pre.discovery_wired(
        {"onboarding.py": "async def ask_next(self):",
         "app.py": "await app.state.onboarding.ask_next()"}) is True


def test_unwired_discovery_with_an_empty_registry_stops_the_gate():
    c = pre.check_beat1(wired=False, projects=[])
    assert c.level == pre.ERROR
    assert "beat 2" in c.detail and "merge_candidates" in c.fix


def test_unwired_discovery_with_confirmed_repos_only_costs_beat_1():
    c = pre.check_beat1(wired=False,
                        projects=[Project(name="a", path="/a",
                                          confirmed=True)])
    assert c.level == pre.WARN and "NOT DEMONSTRATED" in c.fix


def test_wired_discovery_is_ok():
    assert pre.check_beat1(wired=True, projects=[]).level == pre.OK


def test_registry_without_a_confirmed_repo_blocks_beat_2():
    checks = {c.name: c for c in pre.summarize_registry(
        [Project(name="a", path="/a")], loaded=True)}
    assert checks["registry"].level == pre.ERROR


def test_registry_needs_a_finance_kind_for_beat_8():
    checks = {c.name: c for c in pre.summarize_registry(
        [Project(name="a", path="/a", confirmed=True)], loaded=True)}
    assert checks["registry"].level == pre.OK
    assert checks["finance project"].level == pre.ERROR


def test_an_already_pinned_source_warns_that_beat_8_will_look_skipped():
    projects = [Project(name="q", path="/q", confirmed=True, kind="finance",
                        data_source="/q/picks.sqlite")]
    checks = {c.name: c for c in pre.summarize_registry(projects, loaded=True)}
    assert checks["finance source"].level == pre.WARN
    assert "look like it was skipped" in checks["finance source"].detail


def test_an_unpinned_finance_project_is_what_beat_8_wants():
    projects = [Project(name="q", path="/q", confirmed=True, kind="finance")]
    checks = {c.name: c for c in pre.summarize_registry(projects, loaded=True)}
    assert checks["finance source"].level == pre.OK
    assert checks["finance project"].level == pre.OK


def test_an_unparseable_registry_is_a_blocker_not_a_shrug():
    checks = pre.summarize_registry([], loaded=False)
    assert checks[0].level == pre.ERROR and "quarantine" in checks[0].detail


def test_a_missing_fleet_log_is_a_clean_slate():
    checks = pre.summarize_fleet([], 0, has_snapshot=False, log_exists=False)
    assert len(checks) == 1 and checks[0].level == pre.OK


def test_a_non_closed_worker_in_the_log_is_a_ghost_warning():
    records = [{"data": {"worker": "w1", "state": "ACTIVE_TURN",
                         "project": "alethic"}}]
    checks = {c.name: c for c in pre.summarize_fleet(
        records, 0, has_snapshot=False, log_exists=True)}
    assert checks["fleet ghosts"].level == pre.WARN
    assert "interrupted by a restart" in checks["fleet ghosts"].detail


def test_a_closed_worker_leaves_no_ghost():
    records = [{"data": {"worker": "w1", "state": "CLOSED"}}]
    names = {c.name for c in pre.summarize_fleet(
        records, 0, has_snapshot=False, log_exists=True)}
    assert "fleet ghosts" not in names


def test_a_detached_worker_still_counts_as_a_ghost():
    """DETACHED survives a restart as DETACHED — still re-announced, still
    noise inside beat 9's report."""
    records = [{"data": {"worker": "w1", "state": DETACHED}}]
    names = {c.name for c in pre.summarize_fleet(
        records, 0, has_snapshot=False, log_exists=True)}
    assert "fleet ghosts" in names


def test_damaged_lines_and_a_stale_snapshot_are_both_warned_about():
    checks = {c.name: c for c in pre.summarize_fleet(
        [], 2, has_snapshot=True, log_exists=True)}
    assert checks["fleet log damage"].level == pre.WARN
    assert checks["fleet snapshot"].level == pre.WARN
    # …and the two must not share a name, or one silently hides the other in
    # the rendered table.
    assert checks["fleet log"].level == pre.OK


def test_exit_code_is_driven_by_errors_only():
    r = pre.Report()
    r.add("a", pre.OK, "fine")
    r.add("b", pre.WARN, "noted")
    assert pre.exit_code(r) == 0
    r.add("c", pre.ERROR, "broken")
    assert pre.exit_code(r) == 1
    assert [c.name for c in r.errors] == ["c"]
    assert [c.name for c in r.warnings] == ["b"]


def test_state_dir_must_exist(tmp_path):
    assert pre.check_state_dir(tmp_path / "nope")[0].level == pre.ERROR
    assert pre.check_state_dir(tmp_path)[0].level == pre.OK


def test_scan_worktrees_is_quiet_when_there_are_none(tmp_path):
    checks = pre.scan_worktrees(tmp_path / "worktrees", [])
    assert checks[0].level == pre.OK


# ---------------------------------------------------- the observer, e2e ------
def test_observer_once_writes_a_transcript_and_a_beat_summary(tmp_path):
    """--once over logs that already exist: the same code path the live run
    uses, without tailing forever."""
    state = tmp_path / "state"
    state.mkdir()
    write_log(state / "fleet.jsonl", [
        ("spawned", {"worker": "w1", "project": "alethic", "state": "IDLE",
                     "worktree": str(tmp_path / "wt"), "branch": "marlowe/x"}),
        ("permission_wait", {"worker": "w1", "state": "WAITING_PERMISSION",
                             "nonce": "n1"}),
        ("permission_done", {"worker": "w1", "state": "ACTIVE_TURN",
                             "nonce": "n1", "approved": True}),
        ("detached", {"worker": "w1", "state": DETACHED,
                      "session_id": "sess-1"}),
        ("activity", {"worker": "w1", "state": DETACHED}),
    ])
    (state / "server.log").write_text(
        'INFO:     Started server process [1]\n'
        'INFO:     127.0.0.1:1 - "POST /hooks HTTP/1.1" 200 OK\n')
    reg = tmp_path / "projects.json"
    reg.write_text(json.dumps({"schema_version": 1, "projects": [
        {"name": "quant", "path": "/q", "confirmed": True, "kind": "finance",
         "data_source": "/q/picks.sqlite"}]}))
    out = tmp_path / "transcript.md"
    code = obs.main(["--state-dir", str(state), "--registry", str(reg),
                     "--launcher", str(REPO_ROOT / "bin" / "marlowe"),
                     "--out", str(out), "--once"])
    assert code == 0
    text = out.read_text()
    assert "## Beat summary" in text
    for n in (2, 3, 6, 7):
        assert f"[EVIDENCE] {n}." in text
    assert "[BLIND   ] 4." in text
    # --once cannot order access-log lines against the detach: the POST is
    # counted but never claimed as post-detach — beat 7 above came from lane
    # B (the `activity` record on the DETACHED worker), and the transcript
    # says lane A is out.
    assert "(AFTER the detach)" not in text     # the lane-A tag, not beat 7's title
    assert "after the first detach" not in text
    assert "lane B" in text
    # the pin was already on disk before the run: warned about, not claimed
    assert "ALREADY pinned" in text
    assert "[NONE    ] 8." in text
    # and the fleet log it read is untouched
    assert not list(state.glob("fleet.jsonl.torn-*"))
    assert not list(state.glob("fleet.jsonl.1"))


def test_observer_survives_a_state_dir_that_does_not_exist_yet(tmp_path):
    out = tmp_path / "t.md"
    code = obs.main(["--state-dir", str(tmp_path / "missing"),
                     "--registry", str(tmp_path / "none.json"),
                     "--launcher", str(tmp_path / "none"),
                     "--out", str(out), "--once"])
    assert code == 0 and "## Beat summary" in out.read_text()
