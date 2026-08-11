import re

import pytest

from server.discovery import Candidate
from server.registry import Registry
from server.router import Router

NOW = 1000.0


def open_spoken(router, *args, **kwargs):
    """Open an approval AND mark it read aloud — the state every test below
    means. A voice yes may only resolve a request Marlowe actually SPOKE, so a
    raw open_approval() now models an approval nobody has heard; that
    correlation is pinned on its own in tests/test_approval_correlation.py."""
    a = router.open_approval(*args, **kwargs)
    router.mark_spoken(a.nonce)
    return a


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


# ---------- the discovery verb (M3 beat 1) ----------
# Discovery has to be re-runnable by voice: the registry is only ever seeded
# by a scan, and a repo cloned after boot is invisible until another one runs.

@pytest.mark.parametrize("said", [
    "find my projects", "discover my projects", "look for my repos",
    "find my repos", "scan for projects", "find all my projects",
    "find my repositories", "discover my projects again", "find my projects."])
def test_discovery_phrasings_are_parsed(said):
    c = Router().parse(said, reg("soccer"))
    assert c is not None and c.verb == "discover", said
    assert c.project is None and c.argument is None


@pytest.mark.parametrize("said", [
    "where did I leave the Tibet study?",     # ordinary conversation
    "find the HRV protocol note",             # a find that names something else
    "can you find my projects folder?",       # the phrase inside a real question
    "find my projects folder",                # a trailing word we did not parse
    "what projects am I running?",
    "note that I should find my projects"])   # capture keeps its own text
def test_ordinary_speech_never_triggers_discovery(said):
    c = Router().parse(said, reg("soccer"))
    assert c is None or c.verb != "discover", said


def test_the_discovery_verb_does_not_shadow_an_existing_verb():
    # Every other verb still parses the way it did — the discovery pattern is
    # anchored on its own closed vocabulary and shares no opener with them.
    r, registry = Router(), reg("soccer", "projects")
    assert r.parse("pull up soccer", registry).verb == "pull_up"
    assert r.parse("stop soccer", registry).verb == "stop"
    assert r.parse("open projects", registry).verb == "pull_up"   # _PULL_UP keeps "open"
    assert r.parse("note that I found my repos", registry).verb == "capture"


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
    open_spoken(router, "soccer", "npm test", now=NOW)
    state, appr = router.resolve_approval("yes, go ahead", now=NOW + 5)
    assert state == "approved" and appr.project == "soccer"
    assert router.pending_approvals() == []      # consumed


def test_bare_yes_is_refused_when_two_are_pending():
    router = Router()
    open_spoken(router, "soccer", "npm test", now=NOW)
    open_spoken(router, "alethic", "rm -rf build", now=NOW)
    state, appr = router.resolve_approval("yes", now=NOW + 5)
    assert state == "ambiguous" and appr is None
    assert len(router.pending_approvals()) == 2  # nothing consumed


def test_addressed_approval_works_with_two_pending():
    router = Router()
    open_spoken(router, "soccer", "npm test", now=NOW)
    open_spoken(router, "alethic", "rm -rf build", now=NOW)
    state, appr = router.resolve_approval("approve soccer npm test", now=NOW + 5)
    assert state == "approved" and appr.project == "soccer"
    assert [a.project for a in router.pending_approvals()] == ["alethic"]


def test_denial_is_distinguished_from_approval():
    router = Router()
    open_spoken(router, "soccer", "npm test", now=NOW)
    state, appr = router.resolve_approval("no, deny that", now=NOW + 5)
    assert state == "denied" and appr.project == "soccer"


def test_an_expired_approval_cannot_be_accepted():
    router = Router()
    open_spoken(router, "soccer", "npm test", now=NOW)
    state, appr = router.resolve_approval("yes", now=NOW + 601)   # past the 600s expiry
    assert state == "expired" and appr is None
    assert router.pending_approvals() == []      # swept


def test_yes_with_nothing_pending_is_not_a_command():
    assert Router().resolve_approval("yes", now=NOW) == ("none", None)


def test_each_approval_gets_a_distinct_nonce():
    router = Router()
    a = open_spoken(router, "soccer", "npm test", now=NOW)
    b = open_spoken(router, "alethic", "npm test", now=NOW)
    assert a.nonce != b.nonce


def test_naming_an_unmatched_project_does_not_consume_the_only_pending():
    router = Router()
    open_spoken(router, "soccer", "npm test", now=NOW)
    state, appr = router.resolve_approval("approve alethic rm -rf build", now=NOW + 5)
    assert state == "none" and appr is None
    assert len(router.pending_approvals()) == 1      # soccer's approval untouched


def test_two_approvals_for_one_project_are_addressable_by_tool():
    router = Router()
    open_spoken(router, "soccer", "npm test", now=NOW)
    open_spoken(router, "soccer", "rm -rf build", now=NOW)
    state, appr = router.resolve_approval("approve soccer npm test", now=NOW + 5)
    assert state == "approved" and appr.tool == "npm test"
    assert [a.tool for a in router.pending_approvals()] == ["rm -rf build"]


def test_unrelated_speech_is_not_an_approval_answer():
    router = Router()
    open_spoken(router, "soccer", "npm test", now=NOW)
    open_spoken(router, "alethic", "npm test", now=NOW)
    assert router.resolve_approval("where did I leave the Tibet study?", now=NOW + 5) == ("none", None)
    assert len(router.pending_approvals()) == 2


def test_unrelated_speech_after_expiry_is_not_reported_as_expired():
    router = Router()
    open_spoken(router, "soccer", "npm test", now=NOW)
    assert router.resolve_approval("what's the weather?", now=NOW + 601) == ("none", None)


# ---------- mixed polarity fails closed (Task 8 review) ----------
# _AFFIRM anchors on the FIRST word, so an affirm opener stapled to a refusal
# ("sure, cancel that") used to read as an approval — the fail-open direction
# on a path that executes real tools. Both polarities in one utterance must
# resolve NOTHING and leave the approval pending for a clarifying question.

@pytest.mark.parametrize("said", [
    "sure, cancel that",          # bare: every other token is filler
    "yeah, stop it",
    "sure, stop soccer",          # addressed: names the pending project
])
def test_an_affirm_prefixed_refusal_never_approves(said):
    router = Router()
    open_spoken(router, "soccer", "Bash: rm -rf build", now=NOW)
    state, appr = router.resolve_approval(said, now=NOW + 5)
    assert state == "unclear" and appr is None
    assert len(router.pending_approvals()) == 1      # card stays on screen


def test_a_deny_prefixed_affirmation_never_resolves():
    # The mirror direction: "no, go ahead" is genuinely two-faced and must
    # not be read as a denial (or an approval) on its opener alone.
    router = Router()
    open_spoken(router, "soccer", "Bash: npm test", now=NOW)
    state, appr = router.resolve_approval("no, go ahead", now=NOW + 5)
    assert state == "unclear" and appr is None
    assert len(router.pending_approvals()) == 1


def test_a_negated_affirm_phrase_is_still_a_plain_denial():
    # "no, don't do it now, please" carries "do it" — but negated. It is the
    # canonical bare denial from _BARE_FILLER's docstring and must stay one.
    router = Router()
    open_spoken(router, "soccer", "Bash: npm test", now=NOW)
    state, appr = router.resolve_approval("no, don't do it now, please", now=NOW + 5)
    assert state == "denied" and appr.project == "soccer"
    assert router.pending_approvals() == []


def test_mixed_polarity_with_nothing_pending_stays_none():
    # The conflict check must not leak approval state into normal speech.
    assert Router().resolve_approval("sure, cancel that", now=NOW) == ("none", None)


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
    """Two real directories both named jarvis — this machine's actual shape.
    The name here is a DIRECTORY basename, not the assistant's: both checkouts
    still live under `jarvis/` until the repo directory itself is moved."""
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


def test_every_twin_has_a_label_that_actually_selects_it():
    r = dual_registry()
    router = Router()
    labels = router.parse("pull up jarvis", r).needs_disambiguation
    assert len(labels) == 2 and labels[0] != labels[1]
    # every offered label must resolve to exactly one project when spoken back
    for label in labels:
        c = router.parse(f"pull up {label}", r)
        assert c.path is not None, f"label {label!r} is a dead end — it re-asks forever"
        assert not c.needs_disambiguation


def test_bare_name_is_still_ambiguous_for_twins():
    c = Router().parse("pull up jarvis", dual_registry())
    assert c.path is None and len(c.needs_disambiguation) == 2


def test_same_name_approvals_for_different_repos_are_ambiguous_by_voice():
    router = Router()
    a = open_spoken(router, "jarvis", "npm test", now=NOW, path="/Users/likerun/jarvis")
    b = open_spoken(router, "jarvis", "npm test", now=NOW, path="/Users/likerun/Desktop/jarvis")
    state, appr = router.resolve_approval("approve jarvis npm test", now=NOW + 5)
    assert state == "ambiguous" and appr is None
    assert len(router.pending_approvals()) == 2      # neither consumed
    # the console can still resolve precisely by nonce
    assert router.take_nonce(b.nonce, now=NOW + 5) is b


def test_approvals_carry_a_path():
    router = Router()
    a = open_spoken(router, "soccer", "npm test", now=NOW, path="/p/soccer")
    assert a.path == "/p/soccer"
    state, appr = router.resolve_approval("yes", now=NOW + 1)
    assert state == "approved" and appr.path == "/p/soccer"


def test_take_nonce_consumes_exactly_that_approval():
    router = Router()
    a = open_spoken(router, "soccer", "npm test", now=NOW, path="/p/soccer")
    b = open_spoken(router, "alethic", "pip install", now=NOW, path="/p/alethic")
    taken = router.take_nonce(a.nonce, now=NOW + 1)
    assert taken is a
    assert [x.nonce for x in router.pending_approvals()] == [b.nonce]


def test_take_nonce_unknown_or_expired_is_none():
    router = Router()
    assert router.take_nonce("deadbeef", now=NOW) is None
    a = open_spoken(router, "soccer", "npm test", now=NOW)
    assert router.take_nonce(a.nonce, now=NOW + 601) is None   # swept, not taken
    assert router.pending_approvals() == []


def test_status_intercepts_only_when_the_fleet_is_live():
    r = reg("soccer")
    assert Router().parse("what's running right now?", r) is None          # M3.1 behavior
    c = Router().parse("what's running right now?", r, has_fleet=True)
    assert c is not None and c.verb == "status"


def test_pull_it_up_is_a_bare_pull_up_only_with_a_fleet():
    r = reg("soccer")
    assert Router().parse("pull it up", r) is None
    c = Router().parse("pull it up", r, has_fleet=True)
    assert c.verb == "pull_up" and c.project is None and c.path is None


# ---------- addressed approvals must account for EVERY word (whitelist) ----------
# The mixed-polarity guard above is a BLACKLIST: it fires only when a token
# from _DENY_WORDS is present. A refusal word outside that set does not trip
# it — it merely makes the utterance ADDRESSED, and the addressed branch then
# handed back the named project's approval with deny=False. Every utterance
# below is a plain refusal that resolved to "approved" on a card whose tool
# was `rm -rf build`. Adding these words to _DENY_WORDS would be a fifth
# blacklist; the fix is that an addressed utterance may resolve ONLY when the
# match explains every non-filler word in it.

DANGEROUS = "Bash: rm -rf build"


def one_pending(tool=DANGEROUS):
    router = Router()
    open_spoken(router, "soccer", tool, now=NOW, path="/p/soccer")
    return router


# Verified live bypasses on e647617 — every one returned ("approved", <rm -rf>).
UNKNOWN_REFUSALS = [
    "sure, halt soccer",           # _STOP's own verb, absent from _DENY_WORDS
    "yeah, kill soccer",           # ditto
    "yeah, abort soccer",
    "yes, hold off on soccer",
    "okay, forget soccer",
    "sure, leave soccer alone",
    "yeah, back out of soccer",
    "sure, pause soccer",
    "yeah, never mind soccer",
    "sure, not soccer",
    "yes, wait on soccer",
    "yeah, don’t run soccer",  # U+2019: _WORD split "don’t" into don + t
]


@pytest.mark.parametrize("said", UNKNOWN_REFUSALS)
def test_a_refusal_the_guard_never_heard_of_never_approves(said):
    router = one_pending()
    state, appr = router.resolve_approval(said, now=NOW + 5)
    assert state == "unclear" and appr is None, said
    assert len(router.pending_approvals()) == 1, said     # card stays on screen


@pytest.mark.parametrize("said", UNKNOWN_REFUSALS)
def test_the_bypasses_reach_the_resolver_at_all(said):
    # The premise of the bug: router.parse claims none of these, so the brain
    # really does hand them to resolve_approval. If parse ever started
    # claiming them, the test above would be proving nothing.
    assert Router().parse(said, reg("soccer")) is None, said


def _stop_verbs():
    """The stop verbs _STOP ITSELF defines, read off the compiled pattern so a
    verb added there is automatically covered by the property below."""
    from server.router import _STOP
    m = re.search(r"\(\?:([a-z|\s]+)\)", _STOP.pattern)
    assert m, f"cannot read the verb alternation out of {_STOP.pattern!r}"
    return [v.strip() for v in m.group(1).split("|") if v.strip()]


def test_no_stop_verb_can_ride_an_affirm_opener():
    """PROPERTY, not a word list. The last round's matrix was drawn from
    _DENY's own vocabulary, which is exactly why it could not see `halt` and
    `kill`. This one is drawn from _STOP, so the next verb added there cannot
    silently reopen the hole."""
    from server.router import _STOP
    verbs = _stop_verbs()
    assert len(verbs) >= 4, verbs                 # derivation did not degrade
    for verb in verbs:
        assert _STOP.match(f"{verb} soccer"), verb   # faithful to the real regex
        router = one_pending()
        state, appr = router.resolve_approval(f"sure, {verb} soccer", now=NOW + 5)
        assert state != "approved", f"{verb!r} rode an affirm opener to approval"
        assert appr is None, verb
        assert len(router.pending_approvals()) == 1, verb


# Refusal vocabulary the guard does NOT know — none of these appear in
# _DENY_WORDS, _AFFIRM_WORDS, _BARE_FILLER, the project name, or the tool.
@pytest.mark.parametrize("verb", [
    "abort", "pause", "forget", "scrap", "skip", "hold", "ditch", "nix",
    "quit", "shelve", "freeze", "revoke", "withdraw", "undo", "retract",
    "veto", "decline", "refuse", "block", "terminate", "suspend", "cease",
    "desist", "rescind", "unapprove", "disregard", "ignore", "bail",
    "scratch", "kibosh", "squash", "yank", "postpone", "defer",
])
def test_any_unaccounted_word_refuses_to_resolve(verb):
    """The whole point of a whitelist: it holds for refusal words nobody has
    thought of yet. Not one of these is known to the router."""
    router = one_pending()
    state, appr = router.resolve_approval(f"sure, {verb} soccer", now=NOW + 5)
    assert state == "unclear" and appr is None, verb
    assert len(router.pending_approvals()) == 1, verb


def test_the_blacklist_genuinely_cannot_see_these():
    """Documents WHICH mechanism does the work. _polarity_conflict is blind to
    every one of these — if this ever starts returning True, someone widened
    _DENY_WORDS again and the whitelist stopped being what protects us."""
    from server.router import _polarity_conflict
    for said in ("sure, halt soccer", "yeah, kill soccer", "sure, pause soccer",
                 "okay, forget soccer", "sure, not soccer"):
        assert _polarity_conflict(said) is False, said


# ---------- CRITICAL 1: a refusal verb that also appears in the pending
# command's text must NOT be explained by that text (the missing axis) ----------
# Every approval fixture in this file used to open `npm test` or `rm -rf build`,
# so a refusal verb was never IN the pending tool's text — the exact blind spot
# that let "yeah, kill soccer" approve `kill -9 $(pgrep node)`. These pair each
# refusal verb with a real destructive command whose text CONTAINS it.
REFUSAL_VERB_IN_TOOL = [
    ("kill",   "Bash: kill -9 $(pgrep node)"),
    ("delete", "Bash: kubectl delete pod soccer"),
    ("remove", "Bash: git remote remove origin"),
    ("drop",   "Bash: git stash drop"),
    ("reset",  "Bash: git reset --hard HEAD"),
    ("clean",  "Bash: rm -rf build; make clean"),
    ("halt",   "Bash: halt --now"),
    ("stop",   "Bash: systemctl stop nginx"),
    ("cancel", "Bash: scancel 12345"),
]


@pytest.mark.parametrize("verb,tool", REFUSAL_VERB_IN_TOOL)
@pytest.mark.parametrize("opener", ["yeah", "sure", "yes", "okay"])
def test_a_refusal_sharing_a_verb_with_the_command_never_approves(verb, tool, opener):
    """The reproduced fail-open: the refusal verb is a token of the pending
    command, so folding tool tokens into the consent vocabulary 'explained' it
    and the utterance resolved to approved. It must fail closed instead — the
    verb is a leftover word the match cannot account for."""
    router = one_pending(tool)
    state, appr = router.resolve_approval(f"{opener}, {verb} soccer", now=NOW + 5)
    assert state != "approved", (opener, verb, tool)
    assert appr is None, (opener, verb, tool)
    assert len(router.pending_approvals()) == 1, (opener, verb, tool)


def test_the_missing_axis_reproduces_the_exact_reported_bypasses():
    """The six rows from the report, each verified to have returned
    ('approved', <destructive>) before the fix."""
    for tool, said in [
        ("Bash: kill -9 $(pgrep node)", "yeah, kill soccer"),
        ("Bash: kubectl delete pod soccer", "yeah, delete soccer"),
        ("Bash: git remote remove origin", "yeah, remove soccer"),
        ("Bash: git stash drop", "sure, drop soccer"),
        ("Bash: git reset --hard HEAD", "yeah, reset soccer"),
        ("Bash: rm -rf build; make clean", "yeah, clean soccer"),
    ]:
        router = one_pending(tool)
        state, appr = router.resolve_approval(said, now=NOW + 5)
        assert state == "unclear" and appr is None, (said, tool, state)
        assert len(router.pending_approvals()) == 1, (said, tool)


def test_naming_the_whole_tool_still_resolves_even_when_it_holds_a_verb():
    """The whitelist must not over-block: an owner who names the ENTIRE command
    is genuinely addressing it, so the tool's own words (verb included) are
    accounted for and a plain approval still resolves."""
    router = Router()
    a = router.open_approval("soccer", "Bash: git stash drop", now=NOW, path="/p/soccer")
    router.mark_spoken(a.nonce)
    state, appr = router.resolve_approval("approve soccer bash git stash drop",
                                          now=NOW + 5)
    assert state == "approved" and appr is not None


# ---------- CRITICAL 2: a command too long to read aloud is click-only ----------
def test_a_click_only_approval_refuses_a_bare_voice_yes():
    """A command the owner cannot hear in full must not be approvable by ear:
    the fleet flags it voice_ok=False and the router refuses voice resolution
    outright, pointing at the card — even after mark_spoken (the readback said
    'it's on the card', not the command)."""
    router = Router()
    a = router.open_approval("soccer", "Bash: <long>", now=NOW,
                             path="/p/soccer", voice_ok=False)
    router.mark_spoken(a.nonce)
    state, appr = router.resolve_approval("yes", now=NOW + 5)
    assert state == "too_long" and appr is None
    assert len(router.pending_approvals()) == 1


def test_a_click_only_approval_refuses_an_addressed_voice_yes():
    router = Router()
    a = router.open_approval("soccer", "Bash: <long>", now=NOW,
                             path="/p/soccer", voice_ok=False)
    router.mark_spoken(a.nonce)
    state, appr = router.resolve_approval("approve soccer", now=NOW + 5)
    assert state == "too_long" and appr is None
    assert len(router.pending_approvals()) == 1


def test_a_click_only_approval_is_still_resolvable_by_nonce():
    """Click-only means the CONSOLE still resolves it precisely — the card
    carries the full command and clicking Approve on it IS reading it."""
    router = Router()
    a = router.open_approval("soccer", "Bash: <long>", now=NOW,
                             path="/p/soccer", voice_ok=False)
    assert router.take_nonce(a.nonce, now=NOW + 5) is a
    assert router.pending_approvals() == []


def test_a_click_only_approval_beside_a_normal_one_does_not_block_it():
    """A bare yes with one click-only and one normal (spoken) approval still
    resolves the normal one — the too-long refusal is scoped to the approval
    the utterance actually addresses, not the whole pending set."""
    router = Router()
    big = router.open_approval("soccer", "Bash: <long>", now=NOW,
                               path="/p/soccer", voice_ok=False)
    router.mark_spoken(big.nonce)
    ok = router.open_approval("alethic", "npm test", now=NOW, path="/p/alethic")
    router.mark_spoken(ok.nonce)
    state, appr = router.resolve_approval("approve alethic npm test", now=NOW + 5)
    assert state == "approved" and appr.nonce == ok.nonce
    assert [x.nonce for x in router.pending_approvals()] == [big.nonce]


def test_voice_ok_defaults_true_so_normal_approvals_are_unaffected():
    router = Router()
    a = router.open_approval("soccer", "npm test", now=NOW, path="/p/soccer")
    assert a.voice_ok is True
    router.mark_spoken(a.nonce)
    assert router.resolve_approval("yes", now=NOW + 5)[0] == "approved"


# ---------- the whitelist must NOT over-block legitimate consent ----------
@pytest.mark.parametrize("said", [
    "yes", "go ahead", "approve", "do it", "yeah", "yep", "sure", "okay",
    "approved", "yes, go ahead", "sure, go ahead, sir", "approve that now",
])
def test_plain_approvals_still_approve(said):
    router = one_pending("npm test")
    state, appr = router.resolve_approval(said, now=NOW + 5)
    assert state == "approved" and appr is not None, said
    assert router.pending_approvals() == [], said


@pytest.mark.parametrize("said", [
    "no", "stop", "cancel", "cancel that", "no, don't do it now, please",
    "nope", "deny", "reject", "no, deny that", "don't", "no thank you sir",
])
def test_plain_denials_still_deny(said):
    router = one_pending("npm test")
    state, appr = router.resolve_approval(said, now=NOW + 5)
    assert state == "denied" and appr is not None, said
    assert router.pending_approvals() == [], said


def test_an_addressed_approval_that_accounts_for_everything_still_resolves():
    router = Router()
    open_spoken(router, "soccer", "npm test", now=NOW)
    open_spoken(router, "alethic", "rm -rf build", now=NOW)
    state, appr = router.resolve_approval("approve soccer npm test", now=NOW + 5)
    assert state == "approved" and appr.project == "soccer"
    assert [a.project for a in router.pending_approvals()] == ["alethic"]


def test_naming_the_wrong_tool_no_longer_approves_the_right_project():
    """Found by the whitelist, not by the bug report. With ONE pending
    approval, `matched` is decided by the project name alone — so "approve
    soccer npm test" spoken at a card reading `rm -rf build` used to approve
    the rm. The human named a tool that is not the pending one; that is not
    consent to the pending one."""
    router = one_pending(DANGEROUS)
    state, appr = router.resolve_approval("approve soccer npm test", now=NOW + 5)
    assert state == "unclear" and appr is None
    assert len(router.pending_approvals()) == 1
    # and the same words DO approve when they describe the real tool
    ok = one_pending("npm test")
    assert ok.resolve_approval("approve soccer npm test", now=NOW + 5)[0] == "approved"


def test_an_addressed_denial_that_accounts_for_everything_still_resolves():
    router = one_pending()
    state, appr = router.resolve_approval("deny soccer bash rm -rf build",
                                          now=NOW + 5)
    assert state == "denied" and appr.project == "soccer"
    assert router.pending_approvals() == []


def test_a_spoken_full_path_still_singles_out_a_twin():
    """Path tokens are part of what the match explains, so the one utterance
    that CAN tell twin checkouts apart is not blocked by the whitelist."""
    router = Router()
    open_spoken(router, "jarvis", "npm test", now=NOW, path="/Users/likerun/jarvis")
    b = open_spoken(router, "jarvis", "npm test", now=NOW,
                    path="/Users/likerun/Desktop/jarvis")
    state, appr = router.resolve_approval(
        "approve jarvis npm test /Users/likerun/Desktop/jarvis", now=NOW + 5)
    assert state == "approved" and appr is b


@pytest.mark.parametrize("said", [
    "sure, cancel that", "yeah, stop it", "sure, stop soccer", "no, go ahead"])
def test_the_previously_fixed_mixed_polarity_cases_stay_fixed(said):
    router = one_pending()
    state, appr = router.resolve_approval(said, now=NOW + 5)
    assert state == "unclear" and appr is None, said
    assert len(router.pending_approvals()) == 1, said


# ---------- curly apostrophes tokenize like their ASCII twins ----------
@pytest.mark.parametrize("curly,ascii_", [
    ("no, don’t", "no, don't"),
    ("don’t do it", "don't do it"),
    ("no, don’t do it now, please", "no, don't do it now, please"),
])
def test_a_curly_apostrophe_resolves_exactly_like_a_straight_one(curly, ascii_):
    """Deepgram nova-3 emits U+2019. _WORD is [a-z']+, so "don’t" used to split
    into "don" + "t" and the curly form of even the PINNED vocabulary escaped
    every gate."""
    got = [one_pending("npm test").resolve_approval(s, now=NOW + 5)[0]
           for s in (curly, ascii_)]
    assert got[0] == got[1] == "denied", (curly, got)


def test_curly_apostrophes_do_not_change_the_addressed_shape():
    from server.router import bare_yes_no, is_addressed
    assert bare_yes_no("no, don’t") == bare_yes_no("no, don't") is True
    assert is_addressed("don’t") == is_addressed("don't") is False


# ---------- why the BARE branch may keep using the blacklist ----------
def test_every_refusal_a_bare_utterance_can_carry_is_known_to_the_guard():
    """The bare branch is itself whitelisted — a bare utterance is one whose
    every word is in _BARE_FILLER — so _polarity_conflict's blacklist is
    COMPLETE over that closed vocabulary, but only while this holds. If a
    refusal word is ever added to _BARE_FILLER without being added to
    _DENY_WORDS, "sure, <that word>" becomes bare, unconflicted, and
    approved."""
    from server.router import (_AFFIRM_PHRASES, _AFFIRM_WORDS, _BARE_FILLER,
                               _DENY_WORDS, _POLITE)
    assert _DENY_WORDS <= _BARE_FILLER
    assert _AFFIRM_WORDS <= _BARE_FILLER
    # and the filler set is BUILT from those vocabularies plus polarity-free
    # politeness, so the invariant cannot drift by hand-editing one list
    assert _BARE_FILLER == (_AFFIRM_WORDS | _DENY_WORDS | _POLITE
                            | {w for p in _AFFIRM_PHRASES for w in p})


# ---------- the two known over-blocks from the last round ----------
def test_a_trailing_okay_tag_is_acknowledgment_not_consent():
    """"cancel that, okay?" was a working denial on 96b3f70 and the last round
    turned it into "unclear" — English mostly uses a TRAILING ok/okay as a tag,
    not as consent. Dropping trailing-tag affirm evidence can only ever turn a
    conflict into a DENIAL (the outcome polarity comes from the opener, which
    is a refusal in every such utterance), never into an approval."""
    for said in ("cancel that, okay?", "no, that's ok", "stop, ok?"):
        router = one_pending()
        state, appr = router.resolve_approval(said, now=NOW + 5)
        assert state == "denied", (said, state)
        assert appr.project == "soccer"


def test_a_leading_okay_on_a_refusal_still_fails_closed():
    # The tag exception is TRAILING-only. "okay, cancel that" keeps both
    # polarities and must still refuse to resolve.
    router = one_pending()
    assert router.resolve_approval("okay, cancel that", now=NOW + 5) == ("unclear", None)


# ---------- worktree housekeeping: three verbs, none of them a yes ----------
@pytest.mark.parametrize("said", [
    "clean up the worktrees",
    "clean up the work trees",
    "tidy up the worktrees",
    "tidy the worktrees",
    "clean up my worktrees",
    "go through the worktrees",
    "review the worktrees.",
    "check my work trees",
])
def test_the_survey_verb_is_parsed(said):
    c = Router().parse(said, reg("soccer"))
    assert c is not None and c.verb == "worktree_survey"


@pytest.mark.parametrize("said", [
    "remove the empty worktrees",
    "delete the empty worktrees",
    "clear the empty work trees",
    "clear out the empty worktrees",
    "get rid of the empty worktrees",
    "remove all empty worktrees.",
])
def test_the_batch_removal_verb_is_parsed(said):
    c = Router().parse(said, reg("soccer"))
    assert c is not None and c.verb == "worktree_remove_empty"


@pytest.mark.parametrize("said,name", [
    ("remove the worktree for soccer", "soccer"),
    ("delete the worktree for the login fix", "the login fix"),
    ("drop the work tree for jarvis in desktop", "jarvis in desktop"),
])
def test_the_per_item_removal_verb_carries_the_name(said, name):
    c = Router().parse(said, reg("soccer"))
    assert c is not None and c.verb == "worktree_remove_named"
    assert c.argument == name


@pytest.mark.parametrize("said", [
    "where did I leave the Tibet study?",
    "clean up my room",
    "what's running",
    "can you tidy the kitchen",
    "remove the empty calories from my diet",
    "delete that note about worktrees",
    "the worktrees are piling up",
    "tell soccer to clean up the worktrees",
])
def test_ordinary_speech_never_fires_a_worktree_verb(said):
    c = Router().parse(said, reg("soccer"), has_fleet=True)
    assert c is None or not c.verb.startswith("worktree")


def test_a_worktree_verb_is_never_an_affirmation_or_a_denial():
    """THE reason this gate is a verb and not a yes. Three questions can
    already be pending at once (onboarding's repo confirm, the finance source
    confirm, a fleet tool approval) and every one of them resolves on a
    yes-shaped utterance. A fourth yes-gate would need arbitrating against
    those three, and arbitration is where the previous six fail-opens came
    from. None of these phrasings can be produced by, or mistaken for, any
    consent vocabulary in the system."""
    from server.onboarding import _NO, _YES
    from server.router import _AFFIRM, _DENY, bare_yes_no, is_addressed
    for said in ("clean up the worktrees", "tidy up the worktrees",
                 "remove the empty worktrees", "delete the empty work trees",
                 "remove the worktree for soccer"):
        assert not _AFFIRM.match(said), said
        assert not _DENY.match(said), said
        assert not _YES.match(said), said
        assert not _NO.match(said), said
        assert not bare_yes_no(said), said
        assert is_addressed(said), said    # never owned by a pending yes/no


def test_a_worktree_verb_never_collides_with_another_verb():
    """Every other deterministic verb must decline these utterances, so the
    new ones can never shadow (or be shadowed by) a spawn, steer, stop,
    pull-up, capture, discovery, status, portfolio or trade."""
    from server.router import (_CAPTURE, _DISCOVER, _PORTFOLIO, _PULL_IT,
                               _PULL_UP, _SPAWN, _STATUS, _STEER, _STOP,
                               _TRADE)
    for said in ("clean up the worktrees", "tidy up the worktrees",
                 "go through the work trees", "remove the empty worktrees",
                 "clear out the empty worktrees",
                 "remove the worktree for soccer"):
        for pattern in (_SPAWN, _STEER, _PULL_UP, _STOP, _CAPTURE, _DISCOVER,
                        _STATUS, _PULL_IT):
            assert not pattern.match(said), (pattern.pattern, said)
        for pattern in (_PORTFOLIO, _TRADE):
            assert not pattern.search(said), (pattern.pattern, said)


def test_a_trade_still_outranks_a_worktree_verb():
    # _TRADE is checked first and searches anywhere: an utterance carrying a
    # trade must never be routed onward, whatever else it also says.
    c = Router().parse("sell NVDA and clean up the worktrees", reg("soccer"))
    assert c.verb == "refuse_trade"


def test_no_affirmation_vocabulary_can_ever_OPEN_a_worktree_verb():
    """The mirror of the test above, and the one that catches the fail-open
    shape directly rather than by implication.

    The other test asserts that these PHRASINGS are not affirmations. This one
    asserts the reverse — that an affirmation cannot match these PATTERNS —
    because that is the edit a future hand actually makes: widening an opener
    alternation by one friendly token ("|yes|sure") to make a natural sentence
    work. That single token would make a pending "yes" removable-shaped, which
    is the seventh instance of this codebase's oldest bug. An affirmative
    OPENER carrying a real request is not consent anywhere else in Marlowe
    either (onboarding._bare_affirmation, finance_gate._bare_rejection); it
    costs one re-ask here and it is the whole safety argument for this gate.
    """
    from server.router import _AFFIRM_WORDS, _DENY_WORDS
    router, registry = Router(), reg("soccer")
    tails = ["", " the empty worktrees", " the worktrees", " worktrees",
             " the worktree for soccer", ", the empty worktrees",
             ", remove the empty worktrees", " remove the empty worktrees",
             " clean up the worktrees", ", clean up the worktrees",
             " delete the worktree for soccer"]
    vocabulary = sorted(_AFFIRM_WORDS | _DENY_WORDS | {"go ahead", "do it"})
    for word in vocabulary:
        for tail in tails:
            said = word + tail
            c = router.parse(said, registry)
            assert c is None or not c.verb.startswith("worktree"), said
