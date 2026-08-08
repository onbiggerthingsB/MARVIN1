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
