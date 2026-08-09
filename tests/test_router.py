import pytest

from server.discovery import Candidate
from server.registry import Registry
from server.router import Router

NOW = 1000.0


def reg(*names, kind="code"):
    r = Registry()
    r.merge_candidates([Candidate(path=f"/p/{n}", name=n, sources=["t"]) for n in names])
    for n in names:
        r.confirm(n, kind=kind)
    return r


# ---------- verb parsing ----------
def test_spawn_is_parsed_with_project_and_task():
    c = Router().parse("start work in soccer: fix the login redirect", reg("soccer"))
    assert c.verb == "spawn" and c.project == "soccer"
    assert c.argument == "fix the login redirect"


def test_pull_up_is_parsed():
    c = Router().parse("pull up soccer", reg("soccer"))
    assert c.verb == "pull_up" and c.project == "soccer"


def test_stop_is_parsed():
    assert Router().parse("stop soccer", reg("soccer")).verb == "stop"


def test_capture_is_parsed_without_a_project():
    c = Router().parse("note that Dr. Kong wants the HRV protocol by Friday", reg("soccer"))
    assert c.verb == "capture"
    assert c.argument == "Dr. Kong wants the HRV protocol by Friday"


def test_portfolio_is_parsed():
    assert Router().parse("how are the picks doing?", reg("quant agent")).verb == "portfolio"


def test_a_question_is_not_a_command_and_falls_through():
    assert Router().parse("where did I leave the Tibet study?", reg("soccer")) is None


def test_status_questions_fall_through_to_the_butler():
    # No fleet exists yet, and "what's going on..." is a natural question the
    # butler used to answer. Part 2 restores the status verb with a real fleet.
    assert Router().parse("what's going on with the fleet?", reg("soccer")) is None
    assert Router().parse("what is running right now?", reg("soccer")) is None


def test_unknown_project_does_not_become_a_spawn():
    c = Router().parse("start work in nonexistent: do a thing", reg("soccer"))
    assert c is None      # never spawn into a repo Keke has not confirmed


def test_ambiguous_project_asks_rather_than_guessing():
    r = Registry()
    r.merge_candidates([Candidate(path="/a/composed", name="composed", sources=["t"]),
                        Candidate(path="/b/composed", name="composed", sources=["t"])])
    for p in r.projects:
        p.confirmed = True
    c = Router().parse("pull up composed", r)
    assert c.verb == "pull_up" and c.project is None
    assert len(c.needs_disambiguation) == 2


# ---------- the trading refusal (spec §16) ----------
@pytest.mark.parametrize("said", [
    "buy 10 shares of NVDA", "sell all my TSLA", "place an order for AAPL"])
def test_trade_requests_are_refused_not_routed(said):
    assert Router().parse(said, reg("quant agent", kind="finance")).verb == "refuse_trade"


# ---------- approvals ----------
def test_single_pending_approval_accepts_a_bare_yes():
    router = Router()
    router.open_approval("soccer", "npm test", now=NOW)
    state, appr = router.resolve_approval("yes, go ahead", now=NOW + 5)
    assert state == "approved" and appr.project == "soccer"
    assert router.pending_approvals() == []      # consumed


def test_bare_yes_is_refused_when_two_are_pending():
    router = Router()
    router.open_approval("soccer", "npm test", now=NOW)
    router.open_approval("alethic", "rm -rf build", now=NOW)
    state, appr = router.resolve_approval("yes", now=NOW + 5)
    assert state == "ambiguous" and appr is None
    assert len(router.pending_approvals()) == 2  # nothing consumed


def test_addressed_approval_works_with_two_pending():
    router = Router()
    router.open_approval("soccer", "npm test", now=NOW)
    router.open_approval("alethic", "rm -rf build", now=NOW)
    state, appr = router.resolve_approval("approve soccer npm test", now=NOW + 5)
    assert state == "approved" and appr.project == "soccer"
    assert [a.project for a in router.pending_approvals()] == ["alethic"]


def test_denial_is_distinguished_from_approval():
    router = Router()
    router.open_approval("soccer", "npm test", now=NOW)
    state, appr = router.resolve_approval("no, deny that", now=NOW + 5)
    assert state == "denied" and appr.project == "soccer"


def test_an_expired_approval_cannot_be_accepted():
    router = Router()
    router.open_approval("soccer", "npm test", now=NOW)
    state, appr = router.resolve_approval("yes", now=NOW + 601)   # past the 600s expiry
    assert state == "expired" and appr is None
    assert router.pending_approvals() == []      # swept


def test_yes_with_nothing_pending_is_not_a_command():
    assert Router().resolve_approval("yes", now=NOW) == ("none", None)


def test_each_approval_gets_a_distinct_nonce():
    router = Router()
    a = router.open_approval("soccer", "npm test", now=NOW)
    b = router.open_approval("alethic", "npm test", now=NOW)
    assert a.nonce != b.nonce


def test_naming_an_unmatched_project_does_not_consume_the_only_pending():
    router = Router()
    router.open_approval("soccer", "npm test", now=NOW)
    state, appr = router.resolve_approval("approve alethic rm -rf build", now=NOW + 5)
    assert state == "none" and appr is None
    assert len(router.pending_approvals()) == 1      # soccer's approval untouched


def test_two_approvals_for_one_project_are_addressable_by_tool():
    router = Router()
    router.open_approval("soccer", "npm test", now=NOW)
    router.open_approval("soccer", "rm -rf build", now=NOW)
    state, appr = router.resolve_approval("approve soccer npm test", now=NOW + 5)
    assert state == "approved" and appr.tool == "npm test"
    assert [a.tool for a in router.pending_approvals()] == ["rm -rf build"]


def test_unrelated_speech_is_not_an_approval_answer():
    router = Router()
    router.open_approval("soccer", "npm test", now=NOW)
    router.open_approval("alethic", "npm test", now=NOW)
    assert router.resolve_approval("where did I leave the Tibet study?", now=NOW + 5) == ("none", None)
    assert len(router.pending_approvals()) == 2


def test_unrelated_speech_after_expiry_is_not_reported_as_expired():
    router = Router()
    router.open_approval("soccer", "npm test", now=NOW)
    assert router.resolve_approval("what's the weather?", now=NOW + 601) == ("none", None)


def test_bare_yes_no_is_shape_not_meaning():
    from server.router import bare_yes_no
    # bare affirmations and denials: owned by whichever question is pending
    assert bare_yes_no("yes") and bare_yes_no("okay") and bare_yes_no("approved")
    assert bare_yes_no("no, don't") and bare_yes_no("cancel that, please")
    # addressed utterances name something and must keep flowing to the router
    assert not bare_yes_no("approve soccer npm test")
    assert not bare_yes_no("stop soccer")
    assert not bare_yes_no("where did I leave the Tibet study?")


def dual_registry():
    """Two real directories both named jarvis — this machine's actual shape."""
    r = Registry()
    r.merge_candidates([
        Candidate(path="/Users/likerun/jarvis", name="jarvis", sources=["t"]),
        Candidate(path="/Users/likerun/Desktop/jarvis", name="jarvis", sources=["t"])])
    for p in r.projects:
        p.confirmed = True
    return r


def test_commands_carry_the_project_path():
    c = Router().parse("pull up soccer", reg("soccer"))
    assert c.path == "/p/soccer"


def test_spawn_carries_the_path():
    c = Router().parse("start work in soccer: fix the login redirect", reg("soccer"))
    assert c.verb == "spawn" and c.path == "/p/soccer"


def test_twin_basenames_never_read_back_as_the_same_label():
    c = Router().parse("pull up jarvis", dual_registry())
    assert c.project is None and c.path is None
    assert len(c.needs_disambiguation) == 2
    # the whole point: the two labels must be distinguishable when spoken
    assert c.needs_disambiguation[0] != c.needs_disambiguation[1]


def test_extra_location_words_narrow_a_twin():
    c = Router().parse("pull up jarvis in desktop", dual_registry())
    assert c.path == "/Users/likerun/Desktop/jarvis"
    assert c.needs_disambiguation == []


def test_approvals_carry_a_path():
    router = Router()
    a = router.open_approval("soccer", "npm test", now=NOW, path="/p/soccer")
    assert a.path == "/p/soccer"
    state, appr = router.resolve_approval("yes", now=NOW + 1)
    assert state == "approved" and appr.path == "/p/soccer"


def test_take_nonce_consumes_exactly_that_approval():
    router = Router()
    a = router.open_approval("soccer", "npm test", now=NOW, path="/p/soccer")
    b = router.open_approval("alethic", "pip install", now=NOW, path="/p/alethic")
    taken = router.take_nonce(a.nonce, now=NOW + 1)
    assert taken is a
    assert [x.nonce for x in router.pending_approvals()] == [b.nonce]


def test_take_nonce_unknown_or_expired_is_none():
    router = Router()
    assert router.take_nonce("deadbeef", now=NOW) is None
    a = router.open_approval("soccer", "npm test", now=NOW)
    assert router.take_nonce(a.nonce, now=NOW + 601) is None   # swept, not taken
    assert router.pending_approvals() == []
