"""The honest per-worker state machine (spec §5).

Two traps this module exists to encode:
  1. The Stop hook fires at the end of EVERY assistant turn. It proves
     "idle at the prompt", never "finished" — CLOSED requires SessionEnd.
  2. Silence proves nothing. A worker running tests is legitimately silent,
     so silence derives QUIET (a display state, base untouched), and only a
     FAILED health probe may escalate to UNKNOWN.

Both detection layers — the owned SDK stream and the HTTP hook dispatcher —
feed the same machine through the small event-kind vocabulary below, which is
also the FleetLog kind vocabulary, so the durable log reads as a state history.
"""
from __future__ import annotations

from dataclasses import dataclass

ACTIVE_TURN = "ACTIVE_TURN"
WAITING_PERMISSION = "WAITING_PERMISSION"
IDLE_AT_PROMPT = "IDLE_AT_PROMPT"
QUIET = "QUIET"
CLOSED = "CLOSED"
UNKNOWN = "UNKNOWN"
DETACHED = "DETACHED"

QUIET_AFTER_S = 45.0

_EVENT_STATE = {
    "spawned": IDLE_AT_PROMPT,        # connected, nothing asked yet
    "prompt": ACTIVE_TURN,            # UserPromptSubmit hook / query() sent
    "activity": ACTIVE_TURN,          # PreToolUse / PostToolUse / streamed message
    "permission_wait": WAITING_PERMISSION,   # can_use_tool blocking / Notification
    "permission_done": ACTIVE_TURN,
    "turn_done": IDLE_AT_PROMPT,      # Stop hook / ResultMessage: END OF TURN ONLY
    "session_end": CLOSED,
    "detached": DETACHED,
    "lost": UNKNOWN,
}


@dataclass
class WorkerStateMachine:
    base: str = UNKNOWN
    last_event: float = 0.0

    def apply(self, kind: str, now: float) -> str:
        target = _EVENT_STATE.get(kind)
        if target is None:
            return self.state(now)    # unknown kinds must never corrupt the machine
        if self.base == CLOSED:
            return self.state(now)    # closed is final; late hooks bounce off
        if self.base == DETACHED and target != CLOSED:
            return self.state(now)    # another driver owns it; only its end matters
        self.base = target
        self.last_event = now
        return self.state(now)

    def state(self, now: float) -> str:
        """Display state. QUIET is DERIVED — only an active turn can go quiet;
        WAITING_PERMISSION is waiting on Keke and IDLE is a real resting state,
        so neither ever decays."""
        if self.base == ACTIVE_TURN and now - self.last_event > QUIET_AFTER_S:
            return QUIET
        return self.base

    def probe_failed(self, now: float) -> str:
        """The only path from silence to an alert: a health probe that actually
        failed. CLOSED and DETACHED are not ours to reclassify."""
        if self.base not in (CLOSED, DETACHED):
            self.base = UNKNOWN
            self.last_event = now
        return self.state(now)
