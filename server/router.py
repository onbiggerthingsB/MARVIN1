"""Deterministic verb routing. The dangerous intents never go through the model.

Why: an LLM router can mis-resolve "yes" to the wrong pending approval, and it
cannot honestly acknowledge a command in under 150ms because it has not parsed
it yet. Anything this module does not match falls through to the butler, so
natural phrasing still works for questions, search and conversation.
"""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field
from pathlib import PurePosixPath

APPROVAL_TTL_S = 600.0

_SPAWN = re.compile(
    r"^\s*(?:start|begin|kick off)\s+(?:work|working)?\s*(?:in|on)\s+(?P<project>.+?)"
    r"\s*[:,]\s*(?P<task>.+)$", re.I)
_STEER = re.compile(r"^\s*tell\s+(?P<project>.+?)\s+to\s+(?P<task>.+)$", re.I)
_PULL_UP = re.compile(r"^\s*(?:pull\s+up|show\s+me|open)\s+(?P<project>.+?)\s*$", re.I)
# Named so the fail-closed property test can enumerate them: every verb here
# is a REFUSAL when it lands on a pending approval instead of on a worker, and
# only two of them ("stop", "cancel") happen to also live in _DENY_WORDS.
_STOP_VERBS = ("stop", "halt", "cancel", "kill")
_STOP = re.compile(r"^\s*(?:" + "|".join(_STOP_VERBS)
                   + r")\s+(?P<project>.+?)\s*$", re.I)
_CAPTURE = re.compile(r"^\s*(?:note|capture|remember)\s+(?:that\s+)?(?P<text>.+)$", re.I)
# Re-run project discovery. The registry is only ever seeded by a scan, so a
# repo cloned after boot is invisible until another one runs — and Keke has no
# console button for it, only a voice.
#
# The pattern is ANCHORED on the whole utterance and closed on both ends: a
# known opener, optional determiners, and one of a fixed set of object nouns.
# "find the HRV protocol note" and "can you find my projects folder?" leave
# words this pattern does not account for, so neither fires — the same
# "explain every word or do nothing" rule the consent gates use.
#
# The "show me"/"open"/"pull up" family is deliberately EXCLUDED even though
# "show me my projects" is natural: _PULL_UP already owns those openers and
# resolves them against the registry, so sharing them would make one sentence
# mean two different things depending on what happens to be confirmed.
_DISCOVER = re.compile(
    r"^\s*(?:find|discover|look\s+for|scan\s+for|search\s+for)"
    r"(?:\s+(?:my|all|the))*"
    r"\s+(?:projects?|repos?|repositor(?:y|ies))"
    r"(?:\s+again)?\s*[.!?]*\s*$", re.I)
# Worktree housekeeping. THREE verbs, and not one of them is a yes.
#
# Every task ever run leaves a disposable worktree and a `jarvis/*` branch
# behind — nothing in server/ has ever removed one, because "the worktree holds
# the diff a human may still want to merge back". These verbs surface the pile
# and offer a narrow, consented way to clear the part of it that is provably
# worthless.
#
# WHY VERBS AND NOT A YES/NO GATE. Three questions can already be pending at
# once (onboarding's repo confirm, the §16 finance source confirm, and a fleet
# tool approval) and all three resolve on a yes-shaped utterance, arbitrated by
# call order in the brain. A fourth yes-gate would have to be arbitrated
# against those three — and arbitration is exactly where this codebase's six
# consent fail-opens came from. None of the phrasings below can be produced by
# any affirmation or denial vocabulary in the system (pinned by a test), so
# this gate can neither steal a yes nor have its instruction stolen, and a yes
# said anywhere in JARVIS removes no worktree, ever.
#
# WHY THESE PHRASINGS. "clean up / tidy up the worktrees" is what a person
# actually says about accumulated clutter, and it is the phrasing the task's
# own framing uses. The looking verbs (check, review, go through) are included
# because the survey is READ-ONLY — widening a verb that only looks costs
# nothing — while the two removing verbs are kept deliberately narrow and
# literal: "the EMPTY worktrees" names the bucket it may touch, and "the
# worktree FOR <name>" cannot be said by accident. Each pattern is anchored on
# both ends over a closed vocabulary, the same "explain every word or do
# nothing" rule _DISCOVER and the consent gates use, so ordinary conversation
# ("clean up my room", "the worktrees are piling up") matches nothing.
_WT = r"work\s*trees?"
_WORKTREE_SURVEY = re.compile(
    r"^\s*(?:(?:clean|tidy)(?:\s+up)?|check|review|survey|go\s+through)"
    r"(?:\s+(?:my|the|all|of))*"
    r"\s+" + _WT + r"(?:\s+again)?\s*[.!?]*\s*$", re.I)
_WORKTREE_REMOVE_EMPTY = re.compile(
    r"^\s*(?:clear\s+out|clear|remove|delete|get\s+rid\s+of|purge)"
    r"(?:\s+(?:the|all|my))*"
    r"\s+empty\s+" + _WT + r"\s*[.!?]*\s*$", re.I)
_WORKTREE_REMOVE_NAMED = re.compile(
    r"^\s*(?:remove|delete|drop)(?:\s+(?:the|that))*"
    r"\s+" + _WT + r"\s+for\s+(?P<name>.+?)\s*[.!?]*\s*$", re.I)
_STATUS = re.compile(r"^\s*(?:what'?s|what is)\s+(?:running|the fleet|going on)\b.*$", re.I)
_PULL_IT = re.compile(r"^\s*pull\s+(?:it|that)\s+up\s*[.!?]*\s*$", re.I)
_PORTFOLIO = re.compile(
    r"\b(?:the\s+)?(?:picks|portfolio|positions|holdings)\b|"
    r"\bhow(?:'s| is| are)\s+(?:the\s+)?(?:market|stocks?|picks)\b", re.I)
_TRADE = re.compile(
    r"\b(?:buy|sell|short|purchase|liquidate|place\s+(?:an?\s+)?order|"
    r"execute\s+(?:a\s+)?trade|transfer|withdraw|deposit)\b", re.I)

_AFFIRM = re.compile(r"^\s*(?:yes|yeah|yep|sure|ok|okay|go ahead|do it|approve[d]?)\b", re.I)
_DENY = re.compile(r"^\s*(?:no|nope|deny|don'?t|stop|cancel|reject)\b", re.I)

# ---------- one tokenizer, one normalization, for every consent gate ----------
# Deepgram nova-3 emits the TYPOGRAPHIC right single quote (U+2019) inside
# contractions, and _WORD is [a-z']+ — so "don’t" tokenized as "don" + "t",
# and the curly form of even the PINNED refusal vocabulary walked past every
# gate ("yeah, don’t run soccer" resolved to approved). Fold every apostrophe
# variant onto ASCII "'" BEFORE anything matches or tokenizes, in ONE place
# that all three gates (this module, onboarding, finance_gate) route through.
#
# Nothing else in an STT transcript needs folding: nova-3 emits ASCII letters
# and ASCII digits, and every other typographic character it can produce
# (…, –, —, curly DOUBLE quotes) is already a separator to both tokenizers and
# to every pattern here, so it changes no decision. The map is 1:1 in
# characters, so match offsets on normalized text still index the original.
_APOSTROPHES = "’‘ʼʹ′＇"
_APOSTROPHE_MAP = {ord(c): "'" for c in _APOSTROPHES}

_WORD = re.compile(r"[a-z']+")
_TOKEN = re.compile(r"[a-z0-9]+")


def _normalized(text: str) -> str:
    return (text or "").translate(_APOSTROPHE_MAP)


def _words(text: str) -> list[str]:
    """Apostrophe-preserving words ("don't" stays one word)."""
    return _WORD.findall(_normalized(text).lower())


def _tokens(text: str) -> list[str]:
    """Alphanumeric tokens, for matching against paths and tool strings."""
    return _TOKEN.findall(_normalized(text).lower())


# Mixed-polarity guard (Task 8 review). _AFFIRM and _DENY anchor on the FIRST
# word, so "sure, cancel that" reads as an affirmation while carrying a
# refusal — and consent must never ride on the opener alone (the same rule
# onboarding's _NEGATION already enforces for repo confirmation). These sets
# are the SAME vocabulary the two prefix regexes anchor on, searched anywhere.
_AFFIRM_WORDS = frozenset({"yes", "yeah", "yep", "sure", "ok", "okay",
                           "approve", "approved"})
_DENY_WORDS = frozenset({"no", "nope", "deny", "denied", "don't", "dont",
                         "stop", "cancel", "reject"})
# Affirm evidence is discounted when negated: "no, don't do it now" carries
# "do it", but negated it is denial elaboration, not a second polarity.
# "no" itself is NOT a negator here — "no, go ahead" is genuinely two-faced
# and must stay a conflict, while "don't go ahead" is a plain denial.
_AFFIRM_PHRASES = frozenset({("go", "ahead"), ("do", "it")})
_NEGATORS = frozenset({"don't", "dont", "not", "never"})
# English mostly uses a TRAILING "ok"/"okay" as a tag, not as consent —
# "cancel that, okay?", "no, that's ok". Discounting affirm evidence in that
# one position can only ever turn a conflict into a DENIAL, never into an
# approval: the resolved polarity comes from _DENY/_AFFIRM on the FIRST word,
# and any utterance opening with an affirm word already supplies unassailable
# affirm evidence at index 0 (which is never the last index once a deny word
# is also present). So this exception cannot open a fail-open path.
_TAG_AFFIRM = frozenset({"ok", "okay"})

# Politeness that addresses nothing and carries NO polarity. Kept separate
# from the two vocabularies above so _BARE_FILLER can be DERIVED rather than
# hand-copied — the previous hand-copied list is what made it possible for the
# two to drift.
_POLITE = frozenset({"that", "that's", "thats", "this", "please", "sir",
                     "now", "the", "a", "and", "then", "thank", "thanks",
                     "you"})

# Words that may appear in a BARE affirmation/denial without addressing
# anything specific ("yes, go ahead", "no, don't do it now, please").
# Any other alphabetic word means the utterance names something, and it may
# then only resolve an approval whose match explains every word in it.
_BARE_FILLER = (_AFFIRM_WORDS | _DENY_WORDS | _POLITE
                | frozenset(w for phrase in _AFFIRM_PHRASES for w in phrase))


def _polarity_conflict(text: str) -> bool:
    """True when one utterance carries BOTH consent polarities — "sure,
    cancel that", "yeah, stop it", "no, go ahead". resolve_approval must
    FAIL CLOSED on that shape: resolve nothing, let the brain ask.

    This is a BLACKLIST (it fires only on _DENY_WORDS) and it is sound ONLY
    over the closed vocabulary of _BARE_FILLER, where every refusal word is by
    construction a _DENY_WORDS member. Outside that vocabulary it is blind —
    "halt", "kill", "pause" trip nothing — which is why the addressed branch
    is guarded by _unexplained() instead."""
    words = _words(text)
    if not any(w in _DENY_WORDS for w in words):
        return False
    last = len(words) - 1
    for i, w in enumerate(words):
        if i > 0 and words[i - 1] in _NEGATORS:
            continue                       # "don't approve", "don't do it"
        if i == last and i > 0 and w in _TAG_AFFIRM:
            continue                       # "cancel that, okay?" — a tag
        if w in _AFFIRM_WORDS:
            return True
        if i + 1 < len(words) and (w, words[i + 1]) in _AFFIRM_PHRASES:
            return True
    return False


def _unexplained(text: str, vocabulary) -> list[str]:
    """The words of `text` that `vocabulary` does not account for.

    The one mechanism behind both consent gates. Empty means the utterance
    said NOTHING the match cannot explain; anything left over means the
    speaker said something this code does not understand, and consent must
    never be inferred from an utterance we only partly understood."""
    return [w for w in _words(text) if w not in vocabulary]


def _is_addressed(text: str) -> bool:
    """True when the utterance names something beyond a bare yes/no."""
    return bool(_unexplained(text, _BARE_FILLER))


def is_addressed(text: str) -> bool:
    """Public form of _is_addressed for the other consent gates (onboarding's
    repo confirmation). Naming anything beyond the bare yes/no stop-words is
    positive evidence the speaker is NOT answering the pending question — one
    vocabulary, defined once here, so the gates can never drift apart."""
    return _is_addressed(text or "")


def bare_yes_no(text: str) -> bool:
    """A bare affirmation or denial that addresses nothing specific — the
    shape a pending spoken question (repo confirm, data-source confirm) must
    own TERMINALLY. Addressed utterances ("approve soccer npm test", "stop
    soccer") keep flowing to the router: naming something is positive evidence
    the speaker is not answering the pending question."""
    t = _normalized((text or "").strip())
    return bool((_AFFIRM.match(t) or _DENY.match(t)) and not _is_addressed(t))


def _mentions_tool(tool: str, spoken_tokens: set[str]) -> bool:
    """Token-overlap tool match: every token of the tool appears in the
    utterance ("approve soccer npm test" mentions "npm test" but not
    "rm -rf build"). Case-insensitive, punctuation-insensitive."""
    tool_tokens = _tokens(tool)
    return bool(tool_tokens) and all(t in spoken_tokens for t in tool_tokens)


def _approval_vocabulary(a) -> frozenset[str]:
    """Every word an addressed utterance may contain and still resolve `a`.

    THE WHITELIST. An addressed utterance names something, so it may only
    resolve an approval it matched — but "matched" was a substring test on the
    project name, which says nothing about the REST of the sentence. "yeah,
    kill soccer" matched soccer and approved an `rm -rf`. The rule instead: the
    match must account for every word said — the project name, the tool, the
    checkout path, plus the polarity vocabulary and politeness in
    _BARE_FILLER. A word explained by NONE of those is a word this code did
    not understand, and consent is never inferred from that.

    Whitelists degrade in the safe direction. An unknown refusal ("halt",
    "pause", or whatever the next one turns out to be) leaves a leftover word
    and the approval stays pending; an unknown POLITENESS costs one clarifying
    question. A blacklist degrades the other way — four rounds of this bug
    family are four demonstrations."""
    vocabulary = set(_BARE_FILLER)
    for said in (a.project, a.tool, a.path):
        vocabulary.update(_words(said or ""))
    return frozenset(vocabulary)


def _distinct_path_tokens(p, matches) -> set[str]:
    """Tokens of p's path that appear in NO other match's path — the words that
    can single it out ("desktop" for /Users/likerun/Desktop/jarvis)."""
    mine = set(_tokens(p.path))
    for other in matches:
        if other is not p:
            mine -= set(_tokens(other.path))
    return mine


def _label(p, matches) -> str:
    """Spoken disambiguation label. Twins by name are told apart by a location
    word the resolver can actually match back — a qualifier drawn from this
    project's DISTINCT path tokens ('jarvis in Desktop'). A parent name alone
    is not enough: 'likerun' also appears in /Users/likerun/Desktop/jarvis, so
    'jarvis in likerun' would re-ask forever. A twin with NO distinct tokens
    gets its full path — ugly to hear, but _resolve selects on a spoken or
    clicked full path, so every offered label round-trips to a selection."""
    twins = [m for m in matches if m is not p and m.name == p.name]
    if not twins:
        return p.name
    distinct = _distinct_path_tokens(p, matches)
    if distinct:
        # Prefer a whole path component that is distinct on its own — the way
        # a person names a place ("Desktop"), scanning from the parent upward.
        for part in reversed(PurePosixPath(p.path).parent.parts):
            tokens = set(_tokens(part))
            if tokens and tokens <= distinct:
                return f"{p.name} in {part}"
        return f"{p.name} in {sorted(distinct)[0]}"
    return f"{p.name} at {p.path}"


@dataclass
class Command:
    verb: str
    project: str | None = None       # spoken name, for readback
    path: str | None = None          # the REAL key — names collide on this machine
    argument: str | None = None
    needs_disambiguation: list[str] = field(default_factory=list)


@dataclass
class Approval:
    nonce: str
    project: str
    tool: str
    created: float
    path: str = ""                   # which checkout this approval belongs to
    # Set by the brain AFTER _speak(question) RETURNS — i.e. only once this
    # exact request has been read to Keke. A voice yes may resolve nothing
    # else. The whole safety model is the spoken line (the worktree is not a
    # sandbox), so "a request is pending" and "Keke heard THIS request" are
    # different facts and must be stored as such: the brain loop is serial with
    # 60-120s awaits while approvals open asynchronously in the SDK callback,
    # so the pending set routinely diverges from what has been spoken.
    #
    # The CLICK path (take_nonce) deliberately does NOT consult this: the card
    # carries the full command, and clicking Approve on it IS reading it.
    spoken: bool = False


class Router:
    def __init__(self):
        self._approvals: list[Approval] = []

    # ---------- verbs ----------
    def _resolve(self, spoken_project: str, registry):
        """-> (Project | None, ambiguous_labels). Path-keyed: the whole Project
        comes back so callers get name AND path. When several match, extra
        spoken words are tried against each candidate's DISTINCT path tokens
        ("jarvis in desktop" singles out the Desktop twin); if that fails, the
        caller gets location-qualified labels and asks."""
        matches = registry.match(spoken_project)
        if len(matches) > 1:
            lowered = spoken_project.lower()
            # A spoken (or clicked) FULL PATH is the one qualifier every twin
            # has — even one whose every path token also appears in its
            # double's path, whose distinct-token set is therefore empty.
            by_path = [m for m in matches if m.path.lower() in lowered]
            if len(by_path) == 1:
                matches = by_path
            else:
                q_tokens = set(_tokens(lowered))
                narrowed = [m for m in matches
                            if q_tokens & _distinct_path_tokens(m, matches)]
                if len(narrowed) == 1:
                    matches = narrowed
        if len(matches) == 1:
            return matches[0], []
        if len(matches) > 1:
            return None, [_label(m, matches) for m in matches]
        return None, []

    def parse(self, spoken: str, registry, has_fleet: bool = False) -> Command | None:
        # Normalized before ANY pattern runs: several patterns here spell an
        # apostrophe inline ("what'?s", "don'?t"), and nova-3's curly form
        # would slip past every one of them.
        text = _normalized((spoken or "").strip())
        if not text:
            return None

        # Refusing a trade outranks everything: it must never be routed onward,
        # and it must not depend on the model reading a system prompt (spec §16).
        if _TRADE.search(text):
            return Command(verb="refuse_trade")

        # Checked before the project-resolving verbs. Those return None the
        # moment their project does not resolve — they do not fall through to
        # later patterns — so a discovery phrasing that ever overlapped one of
        # them would be swallowed silently rather than mis-routed loudly.
        if _DISCOVER.match(text):
            return Command(verb="discover")

        # Same placement argument as _DISCOVER, and the same shape: anchored on
        # the whole utterance, closed vocabulary, no project to resolve. Checked
        # before the project-resolving verbs because those return None the
        # moment their project does not resolve — an overlap there would be
        # swallowed silently rather than mis-routed loudly. AFTER _TRADE, which
        # outranks everything.
        if _WORKTREE_SURVEY.match(text):
            return Command(verb="worktree_survey")
        if _WORKTREE_REMOVE_EMPTY.match(text):
            return Command(verb="worktree_remove_empty")
        m = _WORKTREE_REMOVE_NAMED.match(text)
        if m:
            return Command(verb="worktree_remove_named",
                           argument=m.group("name").strip())

        for pattern, verb, arg_group in((_SPAWN, "spawn", "task"), (_STEER, "steer", "task")):
            m = pattern.match(text)
            if m:
                proj, ambiguous = self._resolve(m.group("project"), registry)
                if proj is None and not ambiguous:
                    return None            # unknown/unconfirmed project → not a command
                return Command(verb=verb,
                               project=proj.name if proj else None,
                               path=proj.path if proj else None,
                               argument=m.group(arg_group).strip(),
                               needs_disambiguation=ambiguous)

        for pattern, verb in ((_PULL_UP, "pull_up"), (_STOP, "stop")):
            m = pattern.match(text)
            if m:
                proj, ambiguous = self._resolve(m.group("project"), registry)
                if proj is None and not ambiguous:
                    return None
                return Command(verb=verb,
                               project=proj.name if proj else None,
                               path=proj.path if proj else None,
                               needs_disambiguation=ambiguous)

        if has_fleet:
            # Only intercepted while workers exist: with an empty fleet these
            # are natural questions the butler answers better (Part 1 final
            # review — the status verb used to swallow them).
            if _STATUS.match(text):
                return Command(verb="status")
            if _PULL_IT.match(text):
                return Command(verb="pull_up")     # the fleet's obvious worker

        if _PORTFOLIO.search(text):
            return Command(verb="portfolio")

        m = _CAPTURE.match(text)
        if m:
            return Command(verb="capture", argument=m.group("text").strip())
        return None                          # not a dangerous verb → butler's turn

    # ---------- approvals ----------
    def _sweep(self, now: float) -> None:
        self._approvals = [a for a in self._approvals if now - a.created <= APPROVAL_TTL_S]

    def open_approval(self, project: str, tool: str, now: float, path: str = "") -> Approval:
        a = Approval(nonce=secrets.token_hex(8), project=project, tool=tool,
                     created=now, path=path)
        self._approvals.append(a)
        return a

    def pending_approvals(self) -> list[Approval]:
        return list(self._approvals)

    def mark_spoken(self, nonce: str) -> bool:
        """Record that THIS request was read aloud. Called by the brain only
        after the readback's await returns — not before it, and never on the
        strength of having published the card. Returns False for a nonce the
        router no longer holds (stopped, clicked, expired), which is exactly
        the case the readback itself must skip."""
        for a in self._approvals:
            if a.nonce == nonce:
                a.spoken = True
                return True
        return False

    def take_nonce(self, nonce: str, now: float) -> Approval | None:
        """Consume one specific approval — the console click path, the worker
        timeout, and the handoff rejection all resolve by nonce, never by
        position. Expiry is swept FIRST so a click on a stale card can no
        longer approve anything."""
        self._sweep(now)
        a = next((x for x in self._approvals if x.nonce == nonce), None)
        if a is not None:
            self._approvals.remove(a)
        return a

    def resolve_approval(self, spoken: str, now: float) -> tuple[str, Approval | None]:
        text = _normalized((spoken or "").strip())

        # FIX 3: affirm/deny FIRST. Unrelated speech is never an approval
        # answer, and must not leak approval state ("ambiguous"/"expired")
        # into normal conversation where a caller could hijack it.
        affirm, deny = bool(_AFFIRM.match(text)), bool(_DENY.match(text))
        if not (affirm or deny):
            return ("none", None)

        had = bool(self._approvals)
        self._sweep(now)
        if not self._approvals:
            return ("expired", None) if had else ("none", None)

        # THE CORRELATION. Consent answers a QUESTION, and a question that was
        # never asked cannot have been answered. Only approvals whose readback
        # has actually finished are resolvable by voice — in BOTH branches
        # below, because naming the project says nothing about whether its
        # sentence was ever spoken. An unspoken approval is left pending and
        # the caller is told ("unspoken"), never silently dropped into the
        # butler as conversation: the worker is blocked and Keke's yes deserves
        # an answer, just not that one.
        heard = [a for a in self._approvals if a.spoken]
        if not heard:
            return ("unspoken", None)

        # Mixed polarity FAILS CLOSED: an affirm opener stapled to a refusal
        # ("sure, cancel that", "yeah, stop it", "sure, stop soccer") — or the
        # mirror ("no, go ahead") — must never resolve anything. The prefix
        # regexes see only the first word; this check sees the whole
        # utterance. "unclear" leaves every approval pending so the brain can
        # ask a clarifying question and the card stays on screen. Checked
        # AFTER the empty-pending returns above so unrelated conversation
        # still cannot leak approval state.
        if _polarity_conflict(text):
            return ("unclear", None)

        lowered = text.lower()
        matched = [a for a in heard if a.project.lower() in lowered]
        if len(matched) > 1:
            # FIX 2: several approvals for one project — narrow by tool text
            # (token overlap) so "approve soccer npm test" is not a deadlock.
            spoken_tokens = set(_TOKEN.findall(lowered))
            narrowed = [a for a in matched if _mentions_tool(a.tool, spoken_tokens)]
            if len(narrowed) == 1:
                matched = narrowed
        if len(matched) > 1:
            # Twin checkouts: same project name AND same tool. Only an
            # explicit full path in the utterance may tell them apart.
            by_path = [a for a in matched if a.path and a.path.lower() in lowered]
            if len(by_path) == 1:
                matched = by_path

        if _is_addressed(text):
            # FIX 1: an utterance that names something specific may only
            # resolve an approval it actually matched — never fall back to
            # "the" pending one. "approve alethic ..." must not approve soccer.
            if not matched:
                return ("none", None)
            if len(matched) > 1:
                # Fail closed: still cannot tell which one was meant. Two
                # pending "jarvis npm test" approvals in different checkouts
                # must never resolve by voice guesswork — the console resolves
                # precisely by nonce (take_nonce); voice refuses.
                return ("ambiguous", None)
            target = matched[0]
            # THE WHITELIST. Matching the project name is not the same as
            # understanding the sentence: "yeah, kill soccer" matched soccer
            # and approved its `rm -rf build`. An addressed utterance resolves
            # only when the match explains EVERY word in it (see
            # _approval_vocabulary). One unaccounted-for word — a refusal verb
            # nobody has added to a list yet, or anything else — and we say so
            # instead of guessing: "unclear" leaves the approval pending and
            # the card on screen for the brain to ask about.
            if _unexplained(text, _approval_vocabulary(target)):
                return ("unclear", None)
        else:
            # A bare "yes"/"no" can only answer a single unambiguous question —
            # and only one that was ASKED. A second, unread approval opened
            # while the first was being spoken leaves exactly one candidate
            # here, which is the honest reading: Keke heard one sentence.
            if len(heard) > 1:
                return ("ambiguous", None)
            target = heard[0]

        self._approvals.remove(target)
        return ("denied" if deny else "approved", target)
