"""M3P2 milestone gate — the evidence observer.

Run:   cd ~/marlowe && uv run python scripts/gate_observer.py
       (second terminal, started BEFORE `marlowe`)
Then:  perform scripts/gate_checklist.md at the microphone. Ctrl-C at the end.
Also:  `--once` reads the logs that already exist, prints the same transcript
       and exits — for regenerating the evidence after the demo is over.

WHAT THIS IS FOR
The gate is a live human demo. Without a recorder its result is a memory, and
"I think the hooks kept arriving" is not evidence. This tails what Marlowe
already writes to disk, timestamps it, maps it onto the nine beats, and leaves
a transcript Keke can paste into the milestone report.

WHAT IT MUST NOT DO — and why
  * NO HTTP. It never calls /events, /fleet or /health, and it never reads
    state/bootstrap_url. That token is SINGLE-REDEEM (auth.redeem_bootstrap):
    spending it here would lock Keke's own browser console out of the demo it
    is supposed to be recording. Everything below is already durable on disk.
  * NO FleetLog(path). The class is the SERVER's handle, and its constructor
    REPAIRS: a torn tail gets renamed to `<log>.torn-<ts>` and the log is
    rewritten from the verified prefix. Pointing that at a live log during the
    gate would rotate the file out from under the server's single writer and
    destroy the very tail beat 9 exists to examine. So the VALIDATION is reused
    — the same SCHEMA_VERSION and the same _checksum the server computes — and
    nothing else. Every file here is opened "rb", read-only, and never written.
  * NO judgement calls it cannot support. Beats 4 and 5 are spoken/rendered
    only: they leave nothing on disk, and the summary says "observer-blind"
    rather than inventing a proof. Deviations are printed, not smoothed.

THE THREE FILES, AND WHAT EACH BEAT LEAVES BEHIND
  state/fleet.jsonl   the durable fleet log — every state transition, one
                      checksummed JSON record per line. Beats 2, 3, 6, 7, 9.
  state/server.log    uvicorn's output. `POST /hooks` lines are beat 7's
                      first evidence lane; the shutdown/startup banner pair is
                      beat 9's — credited only when a worker was LIVE (neither
                      CLOSED nor DETACHED) at the down signal, because beat 9
                      is "kill MID-WORKER" and a detached ghost is re-announced
                      as "already detached", not as interrupted.
                      NOTE: bin/marlowe starts uvicorn with
                      --no-access-log, so lane A is SILENT unless the server
                      was started with access logging on — gate_preflight.py
                      checks exactly this and prints the fix, and the banner
                      below repeats it at startup.
  config/projects.json  the registry. A repo turning `confirmed` is beat 1's
                      evidence; a `data_source` appearing is beat 8's.

BEAT 7 GETS TWO LANES, because it is the beat with no automated proof:
  A. `POST /hooks` in the access log — proves HTTP hook traffic arrived.
  B. fleet records for a worker whose state is DETACHED. After the handoff,
     Marlowe holds no SDK stream for that session at all (fleet.handoff closes
     the client and verifies the exit), so the ONLY thing that can still move
     that tile is a hook POST from the CLI process in the worktree. A record
     with state=DETACHED arriving after the `detached` record is therefore
     hook traffic, whether or not the access log is on. Lane B carries the
     session id; the access log does not, so lane A is attributed by time —
     which only works while tailing LIVE, where wall-clock interleaving
     orders the two files. In --once the whole fleet log is read before the
     first server.log line and access-log lines carry no timestamps, so lane
     A counts POSTs but credits nothing there; lane B is the authority.

Not a pytest: it tails forever and watches the real state directory.
`testpaths = ["tests"]` keeps `uv run pytest` away from it; the pure logic is
covered by tests/test_gate_kit.py against temp dirs and fixture text.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# _checksum is private, and reaching for it is deliberate: it is the ONE
# definition of "this record verifies", and a second copy here would drift
# from the server's the first time the record shape changes.
from server.fleet_log import SCHEMA_VERSION, _checksum   # noqa: E402
from server.fleet_state import CLOSED, DETACHED          # noqa: E402
from server.registry import Registry                     # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

BEATS = {
    1: "Discovery + a spoken repo confirm",
    2: '"Start work in <repo> — <task>" → "On it… fresh worktree"',
    3: "Approval card → approved BY VOICE with readback → worker finishes",
    4: '"What\'s running?" → spoken fleet line',
    5: '"Pull it up" → transcript pane fills',
    6: "[Open in Terminal] → claude --resume, tile goes DETACHED",
    7: "/hooks POSTs arrive from the worktree session AFTER the detach",
    8: "Finance source question on FIRST ask, then the brief after yes",
    9: "Kill mid-worker, restart, hear the interrupted-worker report",
}
# Spoken/rendered only. Nothing durable is written for either, so the observer
# must say it is blind rather than leave a blank that reads as a failure.
BLIND_BEATS = (4, 5)

# Kinds that can only reach a DETACHED worker through the hook dispatcher:
# the SDK stream for that session is closed and its tasks are verified dead
# before `detached` is ever recorded (fleet.handoff).
HOOK_ONLY_KINDS = ("activity", "prompt", "turn_done", "session_end",
                   "permission_wait")


# --------------------------------------------------------------- tailing ---
class Tailer:
    """Follow a file that may not exist yet, may be rotated out from under us,
    and may end in a half-written line.

    All three happen during this gate, by design:
      * the state dir is empty until the server boots (fleet.jsonl appears on
        the first transition, not at startup);
      * FleetLog.snapshot RENAMES the log to `.jsonl.1` on clean shutdown —
        beat 9 triggers exactly that, twice if Keke restarts twice;
      * `kill -9` mid-append leaves a torn final line, and the next boot's
        FleetLog constructor renames the damaged file to `.torn-<ts>` and
        writes a fresh one — a second rotation, moments after the first.

    Rotation is detected by INODE, not by size: the replacement file can be
    larger than the offset we left off at, so a size comparison alone would
    silently keep reading the old handle forever. The old handle is drained
    ONE more time before it is dropped, because bytes written between the last
    poll and the rename are still readable through it.
    """

    def __init__(self, path: Path, label: str):
        self.path = Path(path)
        self.label = label
        self.rotations = 0
        self.opened = False
        self._fh = None
        self._ino: int | None = None
        self._pos = 0
        self._buf = b""

    def _open(self) -> bool:
        try:
            fh = open(self.path, "rb")
        except OSError:
            return False
        self._fh = fh
        try:
            self._ino = os.fstat(fh.fileno()).st_ino
        except OSError:
            self._ino = None
        self._pos = 0
        self.opened = True
        return True

    def _close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = None
        self._ino = None
        self._pos = 0

    def _drain(self) -> list[str]:
        """Complete lines only. A trailing fragment stays in the buffer until
        its newline arrives — a partial JSON record parsed now would look
        exactly like a checksum failure and cry torn on every poll."""
        try:
            chunk = self._fh.read()
        except OSError:
            return []
        if not chunk:
            return []
        self._pos += len(chunk)
        data = self._buf + chunk
        parts = data.split(b"\n")
        self._buf = parts.pop()
        return [p.decode("utf-8", errors="replace") for p in parts if p.strip()]

    def _rotated(self) -> bool:
        try:
            st = self.path.stat()
        except OSError:
            return True                     # renamed or deleted
        if self._ino is not None and st.st_ino != self._ino:
            return True
        return st.st_size < self._pos       # truncated in place

    def poll(self) -> tuple[list[str], str | None]:
        """(complete lines, note). `note` is a one-off event worth printing —
        the file appearing, or a rotation and what it did to a torn tail."""
        note = None
        lines: list[str] = []
        if self._fh is not None:
            lines += self._drain()
            if self._rotated():
                lines += self._drain()      # bytes written before the rename
                leftover = self._buf
                self._buf = b""
                self._close()
                self.rotations += 1
                note = (f"{self.label} ROTATED (snapshot, or a torn log moved "
                        f"aside) — following the new file")
                if leftover.strip():
                    lines.append(leftover.decode("utf-8", errors="replace"))
                    note += " — the previous file ended mid-line (torn tail)"
        if self._fh is None and self._open():
            if note is None:
                note = f"{self.label} appeared at {self.path}"
            lines += self._drain()
        return lines, note


# ------------------------------------------------------------- log records --
def parse_fleet_line(line: str) -> tuple[dict | None, str]:
    """(record, status). status is "ok" or a short reason.

    Deliberately UNLIKE FleetLog.replay, which stops at the first bad line
    because a skipped record could be the CLOSED that proves a later spawn is
    a different worker. That caution is right for a process deciding what is
    alive; this one is only reporting what it saw, and stopping at the first
    bad byte would blind the recorder for the rest of the demo. So a bad line
    is printed as damage and reading continues — and the summary says so.
    """
    try:
        rec = json.loads(line)
    except ValueError:
        return None, "not JSON"
    if not isinstance(rec, dict):
        return None, "not an object"
    if rec.get("v") != SCHEMA_VERSION:
        return None, f"schema v{rec.get('v')!r}"
    try:
        ok = rec["sum"] == _checksum(rec["seq"], rec["ts"], rec["kind"],
                                     rec["data"])
    except (KeyError, TypeError):
        return None, "missing fields"
    return (rec, "ok") if ok else (None, "checksum mismatch")


def fold_workers(records: list[dict]) -> dict[str, dict]:
    """The last known facts per worker, folded the way Fleet.recover folds
    them: later records win, key by key. Used by the preflight to answer "is
    there already a ghost in here that beat 9 would re-announce?"."""
    folded: dict[str, dict] = {}
    for rec in records:
        data = rec.get("data", {}) or {}
        wid = data.get("worker")
        if not wid:
            continue
        slot = folded.setdefault(wid, {})
        for key in ("project", "path", "state", "task", "worktree",
                    "session_id"):
            if data.get(key):
                slot[key] = data[key]
    return folded


def read_fleet_log(path: Path) -> tuple[list[dict], int]:
    """One-shot read of a fleet log: (verified records, damaged line count).

    Read-only and non-mutating — the point of not using FleetLog here."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return [], 0
    records, damaged = [], 0
    for raw_line in raw.split(b"\n"):
        if not raw_line.strip():
            continue
        rec, status = parse_fleet_line(
            raw_line.decode("utf-8", errors="replace"))
        if rec is None:
            damaged += 1
        else:
            records.append(rec)
    return records, damaged


# ------------------------------------------------------------- beat board ---
@dataclass
class BeatBoard:
    evidence: dict[int, list[str]] = field(
        default_factory=lambda: {n: [] for n in BEATS})
    MAX_PER_BEAT: int = 6

    def note(self, beat: int, why: str) -> bool:
        """True the FIRST time a beat gets evidence — the caller prints a
        louder line for that one."""
        rows = self.evidence.setdefault(beat, [])
        first = not rows
        if len(rows) < self.MAX_PER_BEAT:
            rows.append(why)
        elif len(rows) == self.MAX_PER_BEAT:
            rows.append("… (further evidence of this beat not listed)")
        return first

    def summary_lines(self) -> list[str]:
        out = ["", "=" * 72, "BEAT SUMMARY — what this observer actually saw",
               "=" * 72]
        for n, title in BEATS.items():
            rows = self.evidence.get(n) or []
            if rows:
                mark = "EVIDENCE"
            elif n in BLIND_BEATS:
                mark = "BLIND   "
            else:
                mark = "NONE    "
            out.append(f"[{mark}] {n}. {title}")
            for r in rows:
                out.append(f"           - {r}")
            if not rows and n in BLIND_BEATS:
                out.append("           - spoken/rendered only; nothing durable "
                           "is written. Record this beat by hand.")
        seen = [n for n in BEATS if self.evidence.get(n)]
        missing = [n for n in BEATS
                   if not self.evidence.get(n) and n not in BLIND_BEATS]
        out += ["", f"beats with evidence: "
                    f"{', '.join(str(n) for n in seen) or '(none)'}",
                f"beats with none:     "
                f"{', '.join(str(n) for n in missing) or '(none)'}",
                f"observer-blind:      "
                f"{', '.join(str(n) for n in BLIND_BEATS)} "
                f"(record 4 and 5 by hand)", ""]
        out.append("Evidence here is necessary, not sufficient: the gate is "
                   "what Keke heard and saw.")
        out.append("A beat marked NONE is a finding to write down, not a "
                   "number to quietly drop.")
        return out


# ------------------------------------------------------------ correlation ---
class Correlator:
    """Turns raw lines into beat evidence, and keeps the little bit of state
    beat 7 and beat 9 need: who detached and when, and whether the server has
    been down and back up."""

    def __init__(self, board: BeatBoard, crediting: bool = True,
                 lane_a_ordered: bool = True):
        self.board = board
        # True only when this correlator sees the two files interleaved in
        # wall-clock order (the live tail). --once pumps the ENTIRE fleet log
        # before the first server.log line, so "a detach exists" says nothing
        # about whether a given access-log line came before or after it — and
        # those lines carry no timestamps to compare. With this False, lane A
        # counts POSTs but never credits beat 7; lane B is the authority.
        self.lane_a_ordered = lane_a_ordered
        # False while replaying content that was ALREADY on disk when the
        # observer started. state/server.log survives across runs and ends in
        # a "Shutting down"/"Finished server process" block from the last one,
        # so a crediting replay would hand beat 9 its restart before the demo
        # begins — and a fleet.jsonl left over from earlier testing would hand
        # out beats 2, 6 and 7 the same way. History is printed, never
        # credited; see prime_history().
        self.crediting = crediting
        self.workers: dict[str, dict] = {}
        self.detached: dict[str, dict] = {}     # worker -> {session_id, ts}
        self.first_detach_ts: float | None = None
        self.hooks_total = 0
        self.hooks_after_detach = 0
        self.post_detach_records = 0
        self.damaged_lines = 0
        self.saw_shutdown = False
        self.boots = 0
        self.deviations: list[str] = []
        # Beat 3 is the CONSENT beat: the card alone proves nothing until the
        # outcome arrives, so a raised card is held here (keyed by nonce) and
        # credited only when its permission_done says approved=True.
        self.pending_approvals: dict[str, str] = {}
        # Beat 9 is "kill MID-WORKER": a down signal (the shutdown banner, or
        # a fleet.jsonl rotation — the only signal kill -9 leaves) captures
        # which workers were live at that instant, and the next startup line
        # credits the beat only if that capture is non-empty. A restart with
        # nothing live (all CLOSED, or DETACHED — re-announced as "already
        # detached", not interrupted) is a deviation, not evidence.
        self.down_seen = False
        self.pending_interrupted: list[str] = []

    # ---- fleet.jsonl
    def on_fleet_record(self, rec: dict) -> list[str]:
        kind = str(rec.get("kind", ""))
        data = rec.get("data", {}) or {}
        wid = str(data.get("worker", "") or "?")
        state = str(data.get("state", "") or "?")
        project = data.get("project") or "?"
        seq = rec.get("seq")
        if not self.crediting:
            return [f"(pre-existing, not credited) #{seq} {kind} {wid} "
                    f"{project} -> {state}"]
        out: list[str] = []
        info = self.workers.setdefault(wid, {})
        info.update({k: v for k, v in data.items() if v})
        out.append(f"#{seq} {kind:<15} {wid} {project} -> {state}")

        if kind == "spawned":
            # A `spawned` record is NOT a spawn that took — the same shape as
            # the `detached` case below. Fleet._spawn registers the worker
            # BEFORE it awaits start(), so the CLI's own hooks can already
            # address it by worktree `cwd` while connect() is still running;
            # a SessionEnd landing in that window drives the machine to
            # CLOSED, and WorkerStateMachine.apply bounces every later event
            # off a final CLOSED without raising. `_apply("spawned")` then
            # writes a `spawned` record whose resulting `state` is CLOSED,
            # and _spawn's torn_down branch answers "was stopped while it was
            # starting, sir — nothing is running there" — so the "On it…
            # fresh worktree" sentence beat 2 IS never spoken. CLOSED and
            # DETACHED are the only two states apply() bounces off, so a
            # record carrying either is the machine's own verdict that the
            # spawn did not land; anything else means it moved the machine.
            wt = data.get("worktree", "?")
            base = str(data.get("base_commit", ""))[:7]
            branch = data.get("branch", "?")
            out.append(f"    worktree {wt}")
            out.append(f"    branch   {branch} @ {base}")
            if state in (CLOSED, DETACHED):
                self.deviate(out, f"a `spawned` record for {project} landed "
                                  f"with state {state} — the state machine "
                                  f"bounced the spawn, the session was gone "
                                  f"before it started, Keke heard \"stopped "
                                  f"while it was starting\" and NOT \"On it… "
                                  f"fresh worktree\", and this record is NOT "
                                  f"beat 2 evidence")
            else:
                self._note(out, 2, f"spawned {project} into {wt} ({branch})")
        elif kind == "permission_wait":
            # Not beat-3 evidence YET: beat 3 is "approved BY VOICE", and a
            # card that ends in a denial, a cancellation or an unanswered TTL
            # must not leave a row that reads like consent. Hold it until the
            # outcome arrives.
            key = str(data.get("nonce") or wid)
            self.pending_approvals[key] = (f"approval card raised for "
                                           f"{project} (seq {seq})")
            out.append(f"    approval card raised for {project} (seq {seq}) — "
                       f"beat 3 waits for the outcome; only a GRANTED "
                       f"approval credits it")
        elif kind == "permission_done":
            # fleet.py writes `approved` on every permission_done (and
            # `cancelled: True` on the SDK-teardown path); the TTL expiry
            # writes approved=False like a denial. Only approved=True is
            # consent — everything else is a visible finding, not evidence.
            key = str(data.get("nonce") or wid)
            raised = self.pending_approvals.pop(key, None)
            if data.get("approved") is True:
                if raised:
                    self._note(out, 3, raised)
                self._note(out, 3, f"approval GRANTED for {project} "
                                   f"(seq {seq})")
            else:
                outcome = ("cancelled (the SDK tore the request down)"
                           if data.get("cancelled")
                           else "denied, or expired unanswered")
                self.deviate(out, f"the approval for {project} was {outcome} "
                                  f"(seq {seq}) — an unapproved request does "
                                  f"NOT satisfy beat 3")
        elif kind == "turn_done" and self.board.evidence.get(3):
            self._note(out, 3, f"{project} finished a turn after the "
                               f"approval (seq {seq})")
        elif kind == "detached":
            # A `detached` record is NOT a detach. Fleet.handoff applies the
            # step and only THEN verifies it took — WorkerStateMachine.apply
            # bounces every event but session_end off a final CLOSED without
            # raising — so a worker that closed mid-handoff still leaves a
            # `detached` record whose resulting `state` is CLOSED, followed
            # one record later by `handoff_failed` (no resume command, no
            # Terminal). The record's own `state` field IS the machine's
            # verdict at the instant of the write — the same fold the handoff
            # checks — so gating on it here needs no retraction when the
            # failure record arrives, and a genuine handoff (state DETACHED)
            # still credits on its own record.
            sid = str(data.get("session_id", "") or "")
            if state != DETACHED:
                self.deviate(out, f"a `detached` record for {project} landed "
                                  f"with state {state}, not DETACHED — the "
                                  f"state machine bounced the detach, the "
                                  f"handoff did NOT take, and this record is "
                                  f"NOT beat 6 evidence")
            else:
                self.detached[wid] = {"session_id": sid,
                                      "ts": rec.get("ts", 0.0)}
                if self.first_detach_ts is None:
                    self.first_detach_ts = float(rec.get("ts") or 0.0)
                out.append(f"    session  {sid or '(none recorded!)'}")
                self._note(out, 6,
                           f"{project} DETACHED, session {sid[:12] or '?'}")
                if not sid:
                    self.deviate(out, "a `detached` record carried NO session "
                                      "id — beat 6's `claude --resume` cannot "
                                      "work")
        elif kind in ("handoff_failed", "lost"):
            reason = data.get("reason", "")
            self.deviate(out, f"{kind} for {project}: {reason}")

        # Lane B for beat 7: only hooks can still move a DETACHED tile.
        if (state == DETACHED and wid in self.detached
                and kind in HOOK_ONLY_KINDS):
            self.post_detach_records += 1
            sid = self.detached[wid].get("session_id", "")
            out.append(f"    >>> BEAT 7: `{kind}` reached a DETACHED worker — "
                       f"only a /hooks POST from the worktree session can do "
                       f"that (session {sid[:12] or '?'})")
            self._note(out, 7, f"hook-driven `{kind}` on DETACHED {project} "
                               f"after the handoff (seq {seq})")
        return out

    # ---- server.log
    def on_server_line(self, line: str, wall: float) -> list[str]:
        out: list[str] = []
        text = line.strip()
        if not self.crediting:
            return [f"(pre-existing, not credited) {text[:160]}"]
        if "POST /hooks" in text:
            self.hooks_total += 1
            if not self.lane_a_ordered:
                # Refuse, out loud, rather than degrade into a false claim:
                # workers POST hooks from beat 2 onward, so pre-detach traffic
                # ALWAYS exists, and nothing in this line says which side of
                # the detach it belongs to.
                out.append(f"POST /hooks #{self.hooks_total} — ordering "
                           f"against the detach is UNKNOWN in --once (the "
                           f"access log carries no timestamps), so lane A "
                           f"cannot credit beat 7 here; lane B (fleet "
                           f"records reaching a DETACHED worker) is the "
                           f"authority")
                return out
            after = (self.first_detach_ts is not None)
            if after:
                self.hooks_after_detach += 1
            sid = ""
            if self.detached:
                # The access log carries no session id — it is a bare HTTP
                # line. The attribution is BY TIME against the fleet log, and
                # is labelled as such so nobody reads it as something the
                # access log stated.
                last = max(self.detached.items(),
                           key=lambda kv: kv[1].get("ts", 0.0))
                sid = last[1].get("session_id", "")
            tag = "AFTER the detach" if after else "pre-detach"
            out.append(f"POST /hooks #{self.hooks_total} ({tag})"
                       + (f" — most recent detached session {sid[:12]} "
                          f"(correlated by time, not by the log line)"
                          if sid else ""))
            if after:
                self._note(out, 7,
                           f"POST /hooks #{self.hooks_total} arrived after the "
                           f"detach (access log)")
            return out
        if "Shutting down" in text or "Finished server process" in text:
            self.saw_shutdown = True
            out.append(f"server: {text}")
            self._down_signal(out, "shutdown")
            return out
        if "Started server process" in text or "startup complete" in text:
            self.boots += 1
            out.append(f"server: {text}")
            if self.pending_interrupted:
                names = ", ".join(self.pending_interrupted)
                self._note(out, 9, f"the server came back up after a kill "
                                   f"that interrupted a live worker: {names}")
                self.pending_interrupted = []
                self.down_seen = False
            elif self.down_seen:
                # One verdict per down/up pair: consume the signal so the
                # "Application startup complete" line does not repeat it.
                self.down_seen = False
                self.deviate(out, "the server restarted, but no worker was "
                                  "live at the kill — every known worker was "
                                  "CLOSED or DETACHED (a detached ghost is "
                                  "re-announced as 'already detached', not "
                                  "interrupted), so this restart is NOT "
                                  "beat 9")
            return out
        if text:
            out.append(f"server: {text}")
        return out

    # ---- config/projects.json
    def on_registry(self, before: dict, after: dict) -> list[str]:
        out: list[str] = []
        for name, p in after.items():
            was = before.get(name)
            if p["confirmed"] and (was is None or not was["confirmed"]):
                out.append(f"registry: {name} CONFIRMED ({p['path']})")
                self._note(out, 1, f"{name} confirmed in the registry")
            if p.get("data_source") and (was is None
                                         or not was.get("data_source")):
                out.append(f"registry: {name} data_source pinned "
                           f"-> {p['data_source']}")
                self._note(out, 8, f"{name} pinned its finance source "
                                   f"({Path(p['data_source']).name})")
            if was is not None and p["confirmed"] and was["confirmed"] \
                    and p["path"] != was["path"]:
                self.deviate(out, f"{name} changed path under us")
        return out

    # ---- fleet.jsonl rotation (beat 9's down signal for kill -9)
    def on_fleet_rotation(self) -> list[str]:
        """A rotation is FleetLog.snapshot's rename on clean shutdown, or the
        torn-log repair on the boot after a kill -9 — the DOWN half of beat 9
        at most, and the ONLY down signal kill -9 leaves (no 'Shutting down'
        banner is ever written). It credits nothing by itself: the capture
        below waits for the startup line, and only if a worker was live."""
        out: list[str] = []
        if not self.crediting:
            return out
        self._down_signal(out, "rotation")
        return out

    # ---- helpers
    def _live_workers(self) -> list[str]:
        """Workers whose last recorded state is neither CLOSED nor DETACHED —
        the fold Fleet.recover would call interrupted=True. Workers with no
        recorded state at all are excluded: no state, no claim."""
        return [f"{info.get('project') or '?'} ({wid})"
                for wid, info in self.workers.items()
                if (info.get("state") or "") not in ("", CLOSED, DETACHED)]

    def _down_signal(self, out: list[str], source: str) -> None:
        """The server (or its log) went down: remember who was live RIGHT NOW,
        because that — not the restart itself — is what beat 9 is about.
        Idempotent per outage: 'Shutting down', 'Finished server process' and
        the snapshot rotation all land within one shutdown, and one capture
        line is enough."""
        live = self._live_workers()
        if self.down_seen and live == self.pending_interrupted:
            return
        self.down_seen = True
        if live:
            self.pending_interrupted = live
            out.append(f"    {source} with a live worker mid-flight "
                       f"({', '.join(live)}) — the kill half of beat 9; "
                       f"waiting for the restart")
        else:
            out.append(f"    {source} with no live worker (every known "
                       f"worker CLOSED or DETACHED) — not beat 9 evidence "
                       f"by itself")

    def _note(self, out: list[str], beat: int, why: str) -> None:
        first = self.board.note(beat, why)
        out.append(f"    {'*** ' if first else '    '}beat {beat}"
                   f"{' FIRST EVIDENCE' if first else ''}: {why}")

    def deviate(self, out: list[str], text: str) -> None:
        self.deviations.append(text)
        out.append(f"    !!! DEVIATION — write this down: {text}")


# ------------------------------------------------------------- transcript ---
class Transcript:
    """Streamed to disk line by line and flushed every time, so a hard kill
    of this terminal still leaves everything observed up to that instant."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self._fenced = False

    def header(self, lines: list[str]) -> None:
        for line in lines:
            self._fh.write(line + "\n")
        self._fh.write("\n## Log\n\n```\n")
        self._fenced = True
        self._fh.flush()

    def write(self, line: str) -> None:
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self, summary: list[str]) -> None:
        if self._fenced:
            self._fh.write("```\n")
            self._fenced = False
        self._fh.write("\n## Beat summary\n\n```\n")
        for line in summary:
            self._fh.write(line + "\n")
        self._fh.write("```\n")
        self._fh.flush()
        self._fh.close()


def registry_snapshot(path: Path) -> dict:
    """Name -> {path, confirmed, data_source}. load_strict, never load():
    Registry.load RENAMES an unusable file to `<name>.corrupt-<ts>`, and an
    observer must not move the human's config aside mid-demo."""
    try:
        reg = Registry.load_strict(Path(path))
    except Exception:  # noqa: BLE001 — an unreadable registry is reported, not fatal
        return {}
    return {p.name: {"path": p.path, "confirmed": bool(p.confirmed),
                     "data_source": p.data_source} for p in reg.projects}


def access_log_disabled(launcher_text: str) -> bool:
    """True when the launcher starts uvicorn with access logging off — in
    which case beat 7 lane A (`POST /hooks` in state/server.log) is silent."""
    return "--no-access-log" in (launcher_text or "")


def _stamp(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M3P2 gate evidence observer")
    ap.add_argument("--state-dir", default=str(REPO_ROOT / "state"))
    ap.add_argument("--registry", default=str(REPO_ROOT / "config" /
                                              "projects.json"))
    ap.add_argument("--launcher", default=str(REPO_ROOT / "bin" / "marlowe"))
    ap.add_argument("--out", default="")
    ap.add_argument("--poll", type=float, default=0.4)
    ap.add_argument("--once", action="store_true",
                    help="read what is already on disk, print, and exit")
    args = ap.parse_args(argv)

    state = Path(args.state_dir)
    out_path = Path(args.out) if args.out else (
        state / f"gate-transcript-{time.strftime('%Y%m%d-%H%M%S')}.md")
    board = BeatBoard()
    # --once reads the whole fleet log before the first server.log line, so
    # lane A (access-log POSTs) has no ordering against the detach there.
    corr = Correlator(board, lane_a_ordered=not args.once)
    transcript = Transcript(out_path)

    try:
        launcher_text = Path(args.launcher).read_text(encoding="utf-8")
    except OSError:
        launcher_text = ""

    started = time.time()
    header = [
        "# Marlowe M3P2 milestone gate — observer transcript",
        "",
        f"- started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started))}",
        f"- fleet log: `{state / 'fleet.jsonl'}`",
        f"- server log: `{state / 'server.log'}`",
        f"- registry: `{args.registry}`",
        "- read-only: no HTTP, no /events, no bootstrap token, nothing written"
        " to the state dir but this file.",
    ]
    transcript.header(header)

    def emit(lane: str, text: str, t: float | None = None) -> None:
        line = f"[{_stamp(t or time.time())}] {lane:<8}| {text}"
        print(line, flush=True)
        transcript.write(line)

    for line in header:
        if line.startswith("#") or not line:
            continue
        print(line.lstrip("- "), flush=True)
    print("-" * 72, flush=True)

    if access_log_disabled(launcher_text):
        emit("gate", "NOTE: bin/marlowe starts uvicorn with --no-access-log, so "
                     "`POST /hooks` will NOT appear in state/server.log.")
        emit("gate", "      Beat 7 lane A is silent; lane B (records reaching a "
                     "DETACHED worker) still proves it.")
        emit("gate", "      Run scripts/gate_preflight.py for the exact fix if "
                     "you want the access log too.")

    if args.once:
        emit("gate", "--once: replaying finished logs. Lane A (`POST /hooks` "
                     "in the access log) cannot be ordered against the detach "
                     "here — those lines carry no timestamps and the fleet "
                     "log is read first — so POSTs are counted but NOT "
                     "credited to beat 7. Lane B (fleet records reaching a "
                     "DETACHED worker) is the authority.")

    fleet_tail = Tailer(state / "fleet.jsonl", "fleet.jsonl")
    server_tail = Tailer(state / "server.log", "server.log")
    reg_state = registry_snapshot(Path(args.registry))
    reg_mtime = 0.0
    try:
        reg_mtime = Path(args.registry).stat().st_mtime
    except OSError:
        pass
    if reg_state:
        confirmed = [n for n, p in reg_state.items() if p["confirmed"]]
        emit("registry", f"already on disk: {len(reg_state)} project(s), "
                         f"confirmed: {', '.join(confirmed) or '(none)'}")
        pinned = [n for n, p in reg_state.items() if p.get("data_source")]
        if pinned:
            emit("registry", f"finance source ALREADY pinned for "
                             f"{', '.join(pinned)} — beat 8 will go straight "
                             f"to the brief and look skipped.")

    def pump_fleet(now: float) -> None:
        lines, note = fleet_tail.poll()
        if note:
            emit("fleet", note, now)
            if "ROTATED" in note:
                # The down half of beat 9 at most (snapshot on clean shutdown,
                # or torn-log repair after kill -9) — never the beat itself.
                for text in corr.on_fleet_rotation():
                    emit("fleet", text, now)
        for raw in lines:
            rec, status = parse_fleet_line(raw)
            if rec is None:
                corr.damaged_lines += 1
                emit("fleet", f"!! unverifiable line ({status}) — kept "
                              f"reading; the server's replay would stop "
                              f"here: {raw[:120]}", now)
                continue
            for text in corr.on_fleet_record(rec):
                emit("fleet", text, now)

    def pump_server(now: float) -> None:
        lines, note = server_tail.poll()
        if note:
            emit("server", note, now)
        for raw in lines:
            for text in corr.on_server_line(raw, now):
                emit("server", text, now)

    # Everything already on disk belongs to an EARLIER run. Print it — it is
    # useful context, and a leftover ghost is exactly what the preflight warns
    # about — but credit none of it, or the summary starts the demo with beats
    # it never saw. `--once` is the deliberate exception: it exists to rebuild
    # the transcript from finished logs, where the history IS the evidence.
    if not args.once:
        corr.crediting = False
        before = time.time()
        pump_fleet(before)
        pump_server(before)
        corr.crediting = True
        emit("gate", "everything above was already on disk when I started — "
                     "read as history from an earlier run and credited to no "
                     "beat. From here on, everything counts.")

    # SIGTERM SETS A FLAG rather than raising: a handler that raises fires
    # wherever the interpreter happens to be — including inside the shutdown
    # path that is writing the summary — and a second signal there tore the
    # transcript's own close() apart with a traceback. Ctrl-C still arrives as
    # a normal KeyboardInterrupt, which the loop below catches.
    stopping: list[bool] = []
    signal.signal(signal.SIGTERM, lambda _s, _f: stopping.append(True))

    emit("gate", "watching. Ctrl-C when the demo is done.")
    code = 0
    try:
        while True:
            now = time.time()
            pump_fleet(now)
            pump_server(now)

            try:
                m = Path(args.registry).stat().st_mtime
            except OSError:
                m = 0.0
            if m and m != reg_mtime:
                reg_mtime = m
                after = registry_snapshot(Path(args.registry))
                for text in corr.on_registry(reg_state, after):
                    emit("registry", text, now)
                reg_state = after

            if args.once:
                break
            if stopping:
                emit("gate", "SIGTERM — writing the summary.", now)
                break
            time.sleep(max(0.05, args.poll))
    except KeyboardInterrupt:
        print("", flush=True)
        emit("gate", "stopped by hand — writing the summary.")
    finally:
        summary = board.summary_lines()
        hooks_line = (
            f"POST /hooks seen: {corr.hooks_total} (ordering vs the detach "
            f"unknown in --once — beat 7 rests on lane B)"
            if args.once else
            f"POST /hooks seen: {corr.hooks_total} "
            f"({corr.hooks_after_detach} after the first detach)")
        summary += [
            "",
            f"fleet records damaged/unverifiable: {corr.damaged_lines}",
            hooks_line,
            f"records reaching a DETACHED worker: {corr.post_detach_records}",
            f"log rotations: fleet.jsonl x{fleet_tail.rotations}, "
            f"server.log x{server_tail.rotations}",
        ]
        if corr.deviations:
            summary += ["", "DEVIATIONS (findings — write them into the "
                            "report, do not hide them):"]
            summary += [f"  - {d}" for d in corr.deviations]
        else:
            summary += ["", "no deviations recorded by the observer "
                            "(that is not the same as none happening — "
                            "beats 4 and 5 are unobserved here)."]
        for line in summary:
            print(line, flush=True)
        transcript.close(summary)
        print(f"\ntranscript: {out_path}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
