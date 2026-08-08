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

APPROVAL_TTL_S = 600.0

_SPAWN = re.compile(
    r"^\s*(?:start|begin|kick off)\s+(?:work|working)?\s*(?:in|on)\s+(?P<project>.+?)"
    r"\s*[:,]\s*(?P<task>.+)$", re.I)
_STEER = re.compile(r"^\s*tell\s+(?P<project>.+?)\s+to\s+(?P<task>.+)$", re.I)
_PULL_UP = re.compile(r"^\s*(?:pull\s+up|show\s+me|open)\s+(?P<project>.+?)\s*$", re.I)
_STOP = re.compile(r"^\s*(?:stop|halt|cancel|kill)\s+(?P<project>.+?)\s*$", re.I)
_CAPTURE = re.compile(r"^\s*(?:note|capture|remember)\s+(?:that\s+)?(?P<text>.+)$", re.I)
_STATUS = re.compile(r"^\s*(?:what'?s|what is)\s+(?:running|the fleet|going on)\b.*$", re.I)
_PORTFOLIO = re.compile(
    r"\b(?:the\s+)?(?:picks|portfolio|positions|holdings)\b|"
    r"\bhow(?:'s| is| are)\s+(?:the\s+)?(?:market|stocks?|picks)\b", re.I)
_TRADE = re.compile(
    r"\b(?:buy|sell|short|purchase|liquidate|place\s+(?:an?\s+)?order|"
    r"execute\s+(?:a\s+)?trade|transfer|withdraw|deposit)\b", re.I)

_AFFIRM = re.compile(r"^\s*(?:yes|yeah|yep|sure|ok|okay|go ahead|do it|approve[d]?)\b", re.I)
_DENY = re.compile(r"^\s*(?:no|nope|deny|don'?t|stop|cancel|reject)\b", re.I)


@dataclass
class Command:
    verb: str
    project: str | None = None
    argument: str | None = None
    needs_disambiguation: list[str] = field(default_factory=list)


@dataclass
class Approval:
    nonce: str
    project: str
    tool: str
    created: float


class Router:
    def __init__(self):
        self._approvals: list[Approval] = []

    # ---------- verbs ----------
    def _resolve(self, spoken_project: str, registry) -> tuple[str | None, list[str]]:
        matches = registry.match(spoken_project)
        if len(matches) == 1:
            return matches[0].name, []
        if len(matches) > 1:
            return None, [m.name for m in matches]
        return None, []

    def parse(self, spoken: str, registry) -> Command | None:
        text = (spoken or "").strip()
        if not text:
            return None

        # Refusing a trade outranks everything: it must never be routed onward,
        # and it must not depend on the model reading a system prompt (spec §16).
        if _TRADE.search(text):
            return Command(verb="refuse_trade")

        for pattern, verb, arg_group in ((_SPAWN, "spawn", "task"), (_STEER, "steer", "task")):
            m = pattern.match(text)
            if m:
                name, ambiguous = self._resolve(m.group("project"), registry)
                if name is None and not ambiguous:
                    return None            # unknown/unconfirmed project → not a command
                return Command(verb=verb, project=name,
                               argument=m.group(arg_group).strip(),
                               needs_disambiguation=ambiguous)

        for pattern, verb in ((_PULL_UP, "pull_up"), (_STOP, "stop")):
            m = pattern.match(text)
            if m:
                name, ambiguous = self._resolve(m.group("project"), registry)
                if name is None and not ambiguous:
                    return None
                return Command(verb=verb, project=name, needs_disambiguation=ambiguous)

        if _STATUS.match(text):
            return Command(verb="status")
        if _PORTFOLIO.search(text):
            return Command(verb="portfolio")

        m = _CAPTURE.match(text)
        if m:
            return Command(verb="capture", argument=m.group("text").strip())
        return None                          # not a dangerous verb → butler's turn

    # ---------- approvals ----------
    def _sweep(self, now: float) -> None:
        self._approvals = [a for a in self._approvals if now - a.created <= APPROVAL_TTL_S]

    def open_approval(self, project: str, tool: str, now: float) -> Approval:
        a = Approval(nonce=secrets.token_hex(8), project=project, tool=tool, created=now)
        self._approvals.append(a)
        return a

    def pending_approvals(self) -> list[Approval]:
        return list(self._approvals)

    def resolve_approval(self, spoken: str, now: float) -> tuple[str, Approval | None]:
        text = (spoken or "").strip()
        had = bool(self._approvals)
        self._sweep(now)
        if not self._approvals:
            return ("expired", None) if had else ("none", None)

        affirm, deny = bool(_AFFIRM.match(text)), bool(_DENY.match(text))
        addressed = [a for a in self._approvals
                     if a.project.lower() in text.lower()]
        if len(self._approvals) > 1 and len(addressed) != 1:
            # A bare "yes" with several in flight could approve the wrong command.
            return ("ambiguous", None)
        target = addressed[0] if addressed else self._approvals[0]
        if not (affirm or deny):
            return ("none", None)
        self._approvals.remove(target)
        return ("denied" if deny else "approved", target)
