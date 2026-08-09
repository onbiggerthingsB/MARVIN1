"""Durable fleet events: checksummed JSONL, fsync on important transitions,
atomic snapshots, log rotation (spec §5 durability).

Honesty contract: replay() returns EVIDENCE about the past plus a `torn` flag.
It never decides what is alive — a restart's caller maps every non-final
worker to UNKNOWN/INTERRUPTED, and nothing here re-arms an approval callback.
Everything after a torn or tampered line is unordered rumor: replay stops at
the first bad line rather than skipping it, because a skipped record could be
the CLOSED that proves a later spawn is a different worker."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

SCHEMA_VERSION = 1

# Transitions a power cut must not erase: anything the next boot needs in
# order to be honest. Routine activity may ride the OS buffer.
FSYNC_KINDS = frozenset({
    "spawned", "permission_wait", "permission_done", "turn_done",
    "session_end", "detached", "handoff_failed", "lost"})


def _checksum(seq: int, ts: float, kind: str, data: dict) -> str:
    payload = json.dumps([seq, ts, kind, data], sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class FleetLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        snap = self.load_snapshot()
        records, _ = self.replay()
        self._seq = max(int((snap or {}).get("seq", 0)),
                        records[-1]["seq"] if records else 0)

    # ---------- writing ----------
    def append(self, kind: str, data: dict) -> dict:
        self._seq += 1
        rec = {"v": SCHEMA_VERSION, "seq": self._seq, "ts": time.time(),
               "kind": kind, "data": data}
        rec["sum"] = _checksum(rec["seq"], rec["ts"], kind, data)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            f.flush()
            if kind in FSYNC_KINDS:
                os.fsync(f.fileno())
        return rec

    def snapshot(self, state: dict) -> None:
        """Atomic compaction point. The snapshot carries the current seq so
        numbering stays monotonic across rotation; the old log is rotated to
        `.jsonl.1` (one generation kept), never deleted."""
        snap_path = self.path.with_suffix(".snap")
        tmp = snap_path.with_name(snap_path.name + ".tmp")
        body = {"v": SCHEMA_VERSION, "seq": self._seq, "ts": time.time(),
                "state": state}
        body["sum"] = _checksum(body["seq"], body["ts"], "snapshot", state)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(body, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(snap_path)
        if self.path.exists():
            self.path.replace(self.path.with_suffix(".jsonl.1"))

    # ---------- reading ----------
    def replay(self) -> tuple[list[dict], bool]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return [], False
        records: list[dict] = []
        torn = False
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                ok = (isinstance(rec, dict)
                      and rec.get("v") == SCHEMA_VERSION
                      and rec.get("sum") == _checksum(rec["seq"], rec["ts"],
                                                      rec["kind"], rec["data"]))
            except (ValueError, KeyError, TypeError):
                ok = False
            if not ok:
                torn = True
                break
            records.append(rec)
        return records, torn

    def load_snapshot(self) -> dict | None:
        try:
            body = json.loads(
                self.path.with_suffix(".snap").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        try:
            ok = (isinstance(body, dict)
                  and body.get("v") == SCHEMA_VERSION
                  and body.get("sum") == _checksum(body["seq"], body["ts"],
                                                   "snapshot", body["state"]))
        except (KeyError, TypeError):
            ok = False
        return body if ok else None
