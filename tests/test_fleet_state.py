from server.fleet_state import (ACTIVE_TURN, CLOSED, DETACHED, IDLE_AT_PROMPT,
                                QUIET, QUIET_AFTER_S, UNKNOWN,
                                WAITING_PERMISSION, WorkerStateMachine)


def test_stop_means_idle_at_prompt_never_closed():
    # THE documented trap: Stop fires at the end of every assistant turn.
    m = WorkerStateMachine()
    m.apply("spawned", 0.0)
    m.apply("prompt", 1.0)
    m.apply("activity", 2.0)
    assert m.apply("turn_done", 3.0) == IDLE_AT_PROMPT
    assert m.state(4.0) != CLOSED


def test_a_new_prompt_after_stop_resumes_active():
    m = WorkerStateMachine()
    m.apply("turn_done", 1.0)
    assert m.apply("prompt", 2.0) == ACTIVE_TURN


def test_silence_during_a_turn_becomes_quiet_not_stalled():
    m = WorkerStateMachine()
    m.apply("activity", 0.0)
    assert m.state(QUIET_AFTER_S - 1) == ACTIVE_TURN
    assert m.state(QUIET_AFTER_S + 1) == QUIET       # derived, base untouched
    assert m.base == ACTIVE_TURN
    assert m.apply("activity", QUIET_AFTER_S + 2) == ACTIVE_TURN   # it spoke again


def test_waiting_permission_never_decays_to_quiet():
    # A worker waiting on KEKE is not stalled, no matter how long Keke takes.
    m = WorkerStateMachine()
    m.apply("permission_wait", 0.0)
    assert m.state(10_000.0) == WAITING_PERMISSION


def test_idle_never_decays_to_quiet():
    m = WorkerStateMachine()
    m.apply("turn_done", 0.0)
    assert m.state(10_000.0) == IDLE_AT_PROMPT


def test_only_a_failed_probe_escalates_to_unknown():
    m = WorkerStateMachine()
    m.apply("activity", 0.0)
    assert m.state(1_000.0) == QUIET                 # silence alone: QUIET forever
    assert m.probe_failed(1_001.0) == UNKNOWN        # the probe is the only escalator


def test_closed_is_final():
    m = WorkerStateMachine()
    m.apply("session_end", 0.0)
    assert m.apply("activity", 1.0) == CLOSED        # late hook deliveries bounce off
    assert m.probe_failed(2.0) == CLOSED


def test_detached_ignores_everything_but_session_end():
    # After handoff someone ELSE drives the session; its hooks keep POSTing.
    m = WorkerStateMachine()
    m.apply("detached", 0.0)
    assert m.apply("prompt", 1.0) == DETACHED
    assert m.apply("permission_wait", 2.0) == DETACHED
    assert m.probe_failed(3.0) == DETACHED
    assert m.apply("session_end", 4.0) == CLOSED     # the terminal session ended


def test_unknown_event_kinds_never_corrupt_the_machine():
    m = WorkerStateMachine()
    m.apply("prompt", 0.0)
    assert m.apply("banana", 1.0) == ACTIVE_TURN
    assert m.base == ACTIVE_TURN
