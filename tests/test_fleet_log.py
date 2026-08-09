import json
from pathlib import Path

from server.fleet_log import FSYNC_KINDS, FleetLog


def test_append_replay_roundtrip_preserves_order(tmp_path):
    log = FleetLog(tmp_path / "fleet.jsonl")
    log.append("spawned", {"worker": "w1", "state": "IDLE_AT_PROMPT"})
    log.append("prompt", {"worker": "w1", "state": "ACTIVE_TURN"})
    records, torn = log.replay()
    assert torn is False
    assert [r["kind"] for r in records] == ["spawned", "prompt"]
    assert [r["seq"] for r in records] == [1, 2]
    assert records[0]["data"]["worker"] == "w1"
    assert "spawned" in FSYNC_KINDS          # important transitions hit the platter


def test_a_torn_tail_keeps_the_prefix_and_reports_torn(tmp_path):
    path = tmp_path / "fleet.jsonl"
    log = FleetLog(path)
    log.append("spawned", {"worker": "w1"})
    log.append("prompt", {"worker": "w1"})
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"v": 1, "seq": 3, "ts"')          # power died mid-write
    # replay() on a live handle reports the raw damage; constructing a new
    # FleetLog would repair the file first (see the repair tests below).
    records, torn = log.replay()
    assert len(records) == 2 and torn is True


def test_a_corrupt_middle_line_ends_replay_there(tmp_path):
    # Everything after a bad line is unordered rumor — replay must stop, not skip.
    path = tmp_path / "fleet.jsonl"
    log = FleetLog(path)
    log.append("spawned", {"worker": "w1"})
    log.append("prompt", {"worker": "w1"})
    log.append("turn_done", {"worker": "w1"})
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1][:20] + "X" + lines[1][21:]   # flip a byte mid-record
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    records, torn = log.replay()                     # raw view; construction would repair
    assert len(records) == 1 and torn is True


def test_a_tampered_record_fails_its_checksum(tmp_path):
    path = tmp_path / "fleet.jsonl"
    log = FleetLog(path)
    log.append("spawned", {"worker": "w1", "state": "IDLE_AT_PROMPT"})
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["data"]["state"] = "CLOSED"                  # valid JSON, forged content
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    records, torn = log.replay()                     # raw view; construction would repair
    assert records == [] and torn is True


def test_missing_file_replays_empty_and_untorn(tmp_path):
    records, torn = FleetLog(tmp_path / "nope.jsonl").replay()
    assert records == [] and torn is False


def test_snapshot_roundtrip_and_log_rotation(tmp_path):
    path = tmp_path / "fleet.jsonl"
    log = FleetLog(path)
    log.append("spawned", {"worker": "w1"})
    log.snapshot({"workers": {"w1": {"state": "ACTIVE_TURN"}}})
    assert not path.exists()                          # rotated aside...
    assert path.with_suffix(".jsonl.1").exists()      # ...not destroyed
    body = log.load_snapshot()
    assert body["state"] == {"workers": {"w1": {"state": "ACTIVE_TURN"}}}
    log.append("turn_done", {"worker": "w1"})
    records, torn = log.replay()
    assert [r["kind"] for r in records] == ["turn_done"] and torn is False


def test_a_tampered_snapshot_is_refused(tmp_path):
    path = tmp_path / "fleet.jsonl"
    log = FleetLog(path)
    log.snapshot({"workers": {}})
    snap = path.with_suffix(".snap")
    body = json.loads(snap.read_text(encoding="utf-8"))
    body["state"] = {"workers": {"forged": {"state": "CLOSED"}}}
    snap.write_text(json.dumps(body), encoding="utf-8")
    assert FleetLog(path).load_snapshot() is None


def test_appending_after_a_torn_tail_is_not_swallowed(tmp_path):
    p = tmp_path / "fleet.jsonl"
    log = FleetLog(p)
    log.append("spawn", {"n": 1})
    log.append("spawn", {"n": 2})
    # simulate a crash mid-write: a partial final line with no newline
    with p.open("a", encoding="utf-8") as f:
        f.write('{"v":1,"seq":3,"ts":0,"kind":"spa')
    # restart: the log must repair itself and keep accepting records
    log2 = FleetLog(p)
    log2.append("session_end", {"n": 3})
    records, torn = log2.replay()
    kinds = [r["kind"] for r in records]
    assert kinds == ["spawn", "spawn", "session_end"], kinds   # nothing swallowed
    assert torn is False
    assert list(tmp_path.glob("fleet.jsonl.torn-*")), "the damaged bytes must be preserved"


def test_non_utf8_garbage_is_a_torn_boundary_not_a_crash(tmp_path):
    path = tmp_path / "fleet.jsonl"
    log = FleetLog(path)
    log.append("spawned", {"worker": "w1"})
    log.append("permission_done", {"worker": "w1"})
    with open(path, "ab") as f:
        f.write(b"\xff\xfe crash garbage \xf0")      # not valid UTF-8
    records, torn = log.replay()                     # must not raise
    assert [r["kind"] for r in records] == ["spawned", "permission_done"]
    assert torn is True                              # an undecodable line is a torn boundary
    log2 = FleetLog(path)                            # construction repairs, never crashes
    records2, torn2 = log2.replay()
    assert [r["kind"] for r in records2] == ["spawned", "permission_done"]
    assert torn2 is False
    assert list(tmp_path.glob("fleet.jsonl.torn-*")), "the damaged bytes must be preserved"


def test_seq_survives_restart_and_rotation(tmp_path):
    path = tmp_path / "fleet.jsonl"
    log = FleetLog(path)
    log.append("spawned", {"worker": "w1"})           # seq 1
    log.append("prompt", {"worker": "w1"})            # seq 2
    log.snapshot({"workers": {}})                     # rotates; snapshot carries seq 2
    log2 = FleetLog(path)                             # a fresh process
    rec = log2.append("turn_done", {"worker": "w1"})
    assert rec["seq"] == 3                            # monotonic across restart+rotation
