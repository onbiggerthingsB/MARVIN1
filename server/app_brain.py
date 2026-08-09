"""The butler brain loop: bus events -> butler.ask -> speak + answer events."""
from __future__ import annotations

import asyncio
import time

from server.butler import ButlerUnavailable
from server.finance import TRADE_REFUSAL, find_finance_project, portfolio_brief
from server.router import bare_yes_no
from server.vault_paths import vault_root_from_env
from server.vault_write import vault_capture

FALLBACK_LINE = "Sorry sir, I couldn't reach my brain just now."
UNCLEAR_LINE = "Sorry sir, I didn't get a clean answer that time — it's on screen."

# What to SPEAK when the transport itself failed (ButlerUnavailable), keyed by
# its `reason`. The raw error text is never read aloud -- it goes to the
# console via the event's `detail` field instead. Unmapped reasons get the
# default line.
UNAVAILABLE_LINES = {
    "login expired": "I can't reach my brain, sir — my login needs refreshing.",
    "rate limited": "We're rate limited at the moment, sir. Try again shortly.",
}
UNAVAILABLE_DEFAULT = "I can't reach my brain just now, sir."

# A single turn must never wedge the loop. permission_mode is "default" and this
# process is headless: if a tool call ever lands outside the allowed surface
# there is nobody to answer the permission prompt and ask() would hang forever,
# taking every later utterance with it. Time it out and treat that like any
# other failure.
ASK_TIMEOUT_S = 120
# Same argument, applied to every other await in the loop. The loop is SERIAL:
# one await that never returns is one JARVIS that never answers again. A TTS
# socket that connects and then goes quiet, or an ElevenLabs pre-warm against a
# black-holed network, hangs indefinitely -- try/except catches the raise it
# never makes. Bound them.
SPEAK_TIMEOUT_S = 60
PRECONNECT_TIMEOUT_S = 15
# Citation validation touches the filesystem (an iCloud-evicted directory can
# stall), so it is bounded too. A slow validator must cost the chips, not the turn.
VALIDATE_TIMEOUT_S = 10


def _record_utterance(turnlog, data):
    """Convert the wire's millisecond `t_release` and hand it to the TurnLog.

    The division lives HERE, not in the argument list of the `_safe(...)` call.
    An argument expression is evaluated in the caller's frame BEFORE the guarded
    callable is entered, so `(data.get("t_release") or 0) / 1000` written inline
    would raise straight out of the brain loop -- and `t_release` is unvalidated
    wire data (server/stt.py republishes whatever the /mic websocket JSON held).
    A string, list or dict there would kill the task silently. Inside this
    function the same TypeError is just another guarded failure.

    Behaviour for good input is unchanged: None/0 -> None (metrics skipped),
    a real millisecond epoch -> seconds.
    """
    raw = data.get("t_release")
    t_release = (raw or 0) / 1000 or None
    turnlog.record_utterance(t_release=t_release, t_utterance=data.get("t_utterance"))


def speakable(spoken) -> str:
    """What to actually read aloud, given the butler's `spoken` field.

    parse_butler_output falls back to plain text when the model's JSON has only
    empty values -- which makes the raw serialized JSON the `spoken` string. Read
    aloud that is a mouthful of braces and quotes, so anything empty or
    JSON-looking is replaced with a short canned line. JSON-looking includes a
    reply wrapped in a ```json code fence: the fence line (``` plus an optional
    language tag) is stripped before probing, so a truncated fenced reply is not
    spoken verbatim, backticks and all. `\\r` is stripped because
    bulleted replies leave stray carriage returns that some voices verbalize.
    CRLF is collapsed to a bare newline FIRST -- replacing `\\r` alone would turn
    every `\\r\\n` into a stray trailing space before the newline.
    """
    text = (spoken or "").replace("\r\n", "\n").replace("\r", " ").strip()
    probe = text
    if probe.startswith("```"):
        nl = probe.find("\n")
        probe = probe[nl + 1:].lstrip() if nl != -1 else ""
    if not text or not probe or probe.startswith(("{", "[")):
        return UNCLEAR_LINE
    return text


async def run_butler_brain(bus, butler, speaker, turnlog, validate_citations=None,
                           router=None, registry=None, onboarding=None, finance=None):
    """Subscribe to the bus and drive the butler. Never raises out of the loop.

    `validate_citations` is an optional async callable taking the model's list of
    cited titles and returning the subset that resolves to real vault notes.
    None (the default) means no validation, which keeps this loop importable and
    testable without a vault behind it; server/app.py passes the real one.

    `router`/`registry`/`onboarding` wire in M3's deterministic layer: dangerous
    verbs are parsed BEFORE the model ever sees an utterance, and a pending
    confirmation or approval owns the next reply. All keyword-optional so every
    M2 caller and test keeps working. `finance` is reserved for M3 Part 2's
    injection point and unused today (the brief pins the signature).
    """
    cid, q = bus.subscribe()

    def _reason(e) -> str:
        # asyncio.TimeoutError stringifies to "", which would publish a blank
        # reason and read as "speak failed: " on the console.
        return "timed out" if isinstance(e, asyncio.TimeoutError) else str(e)

    async def _speak(text):
        # A TTS failure is not a reason to lose the brain for the rest of the
        # session; surface it and keep looping. wait_for because a hang is not a
        # failure this try/except would ever see.
        try:
            await asyncio.wait_for(speaker.speak(text), SPEAK_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            bus.publish("butler.error", {"reason": f"speak failed: {_reason(e)}"})

    async def _preconnect():
        # Same reasoning as _speak: a 401, a DNS failure or a dropped network on
        # the TTS pre-warm would otherwise raise straight out of this loop and
        # end the task -- JARVIS goes deaf until the process restarts, silently,
        # because nothing publishes on that path. M1's fire-and-forget
        # create_task(preconnect()) could not kill the loop either.
        try:
            await asyncio.wait_for(speaker.preconnect(), PRECONNECT_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — a wake must never kill the brain
            bus.publish("butler.error", {"reason": f"preconnect failed: {_reason(e)}"})

    async def _validated(citations):
        """Keep only citations that name a real note (spec §4). Never raises.

        A hallucinated [[wikilink]] renders as a chip identical to a real one, so
        an invented source is indistinguishable from a grounded one on screen.

        On validator failure the RAW list is published rather than nothing --
        losing every chip is worse than showing an unverified one -- and the
        diagnostic goes to metrics.error, not butler.error, because the console's
        butler.error handler clears #answer/#citations and this runs immediately
        before the answer is published.
        """
        if validate_citations is None or not citations:
            return citations
        try:
            return list(await asyncio.wait_for(
                validate_citations(citations), VALIDATE_TIMEOUT_S))
        except Exception as e:  # noqa: BLE001 — validation must never kill the brain
            bus.publish("metrics.error",
                        {"reason": f"citation check failed: {_reason(e)}"})
            return citations

    def _safe(fn, *args, **kwargs):
        # Metrics bookkeeping takes possibly-None (and possibly non-numeric)
        # timestamps straight off the wire; a TurnLog that trips over one must
        # not take the brain with it.
        #
        # This publishes metrics.error, NOT butler.error: the console's
        # butler.error handler clears #answer and #citations, so routing a
        # metrics hiccup there would wipe a correct, already-rendered answer off
        # screen while JARVIS is still speaking it (tts.done arrives mid-speech).
        # Every _safe call site is a turnlog path, so the event type is fixed
        # here rather than parameterized.
        try:
            fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — metrics must never kill the brain
            bus.publish("metrics.error", {"reason": f"turnlog failed: {e}"})

    def _record_audio(t_first_audio):
        turnlog.record_first_audio(t_first_audio)
        bus.publish("metrics.turn", turnlog.summary())

    try:
        while True:
            event = await q.get()
            if event is None:                       # evicted; resubscribe
                cid, q = bus.subscribe()
                continue
            etype = event["type"]
            data = event.get("data", {})
            if etype == "wake":
                await _preconnect()
                continue
            if etype == "tts.done":
                _safe(_record_audio, data.get("t_first_audio"))
                continue
            if etype in ("stt.utterance", "command.received"):
                if etype == "stt.utterance":
                    _safe(_record_utterance, turnlog, data)
                text = (data.get("text") or "").strip()
                if not text:
                    continue

                # A pending confirmation owns the next utterance ("...right?").
                # `awaiting` is read BEFORE handle_reply (which clears it on a
                # terminal outcome), and inside the guard — it is a property on
                # an injected object and can raise like anything else.
                if onboarding is not None:
                    outcome, awaiting = "ignored", False
                    try:
                        awaiting = bool(onboarding.awaiting)
                        outcome = await onboarding.handle_reply(text)
                    except Exception as e:  # noqa: BLE001 — onboarding must not kill the brain
                        bus.publish("butler.error", {"reason": f"onboarding failed: {e}"})
                    if outcome != "ignored":
                        await _speak("Noted, sir." if outcome != "rejected" else "Understood.")
                        continue
                    if awaiting and bare_yes_no(text):
                        # Precondition 2 (Part 1 final review): the pending
                        # question owns yes/no-shaped speech TERMINALLY. Without
                        # this, any affirmation onboarding doesn't parse
                        # ("approved" today; any future vocabulary drift) falls
                        # through and resolves a pending TOOL approval Keke was
                        # never read. Addressed utterances ("stop soccer") pass.
                        await _speak("About the repo, sir — is that a yes or a no?")
                        continue

                # Dangerous verbs are parsed deterministically, never by the model.
                command = None
                if router is not None and registry is not None:
                    try:
                        command = router.parse(text, registry)
                    except Exception as e:  # noqa: BLE001 — a router fault falls through
                        bus.publish("butler.error", {"reason": f"router failed: {e}"})
                        command = None

                # Deny-vs-stop tie-break (binding note from Task 3's review): the
                # router's denial vocabulary ("stop", "cancel") overlaps the stop
                # VERB, so call order alone cannot arbitrate. Gate on utterance
                # SHAPE: a parse that resolved (or nearly resolved) a confirmed
                # project — "stop soccer" — is positive evidence of a fleet
                # command and wins; an utterance that names no project ("yes",
                # bare "no", bare "cancel") may answer a pending approval first.
                # The affirm vocabulary never overlaps any verb pattern, so no
                # reading of this tie can ever APPROVE something unintended —
                # both residual misreadings are refusals, which fail safe.
                # EVERY router touch below sits inside ONE try: the pending
                # check, the resolve call, and the dereference of the returned
                # approval. Two shipped bugs came from expressions evaluated
                # OUTSIDE the guard meant to protect them (an unguarded
                # preconnect, then wire arithmetic in an argument list); the
                # bare pending_approvals() call and the approval.project /
                # .tool / .nonce reads were the third instance — a router
                # fault (or a contract-violating ("approved", None) return)
                # raised straight out of the loop and killed the brain.
                state, approval = "none", None
                if router is not None:
                    try:
                        if (router.pending_approvals()
                                and not (command is not None
                                         and (command.project
                                              or command.needs_disambiguation))):
                            state, approval = router.resolve_approval(text, time.time())
                            if state in ("approved", "denied") and approval is not None:
                                bus.publish("approval.resolved", {
                                    "outcome": state, "project": approval.project,
                                    "tool": approval.tool, "nonce": approval.nonce})
                    except Exception as e:  # noqa: BLE001 — a router fault must never kill the brain
                        bus.publish("butler.error",
                                    {"reason": f"approval handling failed: {e}"})
                        state, approval = "none", None
                if state in ("approved", "denied"):
                    await _speak("Approved, sir." if state == "approved"
                                 else "Denied, sir.")
                    continue
                if state == "ambiguous":
                    await _speak("More than one approval is pending, sir — "
                                 "name the project.")
                    continue
                if state == "expired":
                    await _speak("That approval expired, sir.")
                    continue
                # "none": not an approval answer — fall through.

                if command is not None:
                    # ONE guard around the whole dispatch: every read of
                    # command.verb/.project/.argument and the iteration of
                    # needs_disambiguation below trusts the Command contract,
                    # and the same contract-violating-return class already bit
                    # resolve_approval. A malformed Command must cost this
                    # turn, never the loop.
                    try:
                        bus.publish("router.command", {
                            "verb": command.verb, "project": command.project,
                            "path": command.path,
                            "argument": command.argument,
                            "needs_disambiguation": command.needs_disambiguation})
                        if command.verb == "refuse_trade":
                            await _speak(TRADE_REFUSAL)
                        elif command.needs_disambiguation:
                            # map(str, ... or []) because this join is an argument
                            # expression evaluated BEFORE _speak's guard is entered;
                            # a non-string element would raise out of the loop.
                            await _speak("Which one, sir? "
                                         + " or ".join(map(str, command.needs_disambiguation or []))
                                         + ".")
                        elif command.verb == "portfolio":
                            # The whole finance path — project lookup, the brief
                            # await, and every dict read — is guarded: a raise on
                            # any of it must cost this turn, never the loop. The
                            # dict reads use .get() so a shape change in the brief
                            # cannot raise either.
                            try:
                                brief = await portfolio_brief(find_finance_project(registry))
                                brief = brief or {}
                                bus.publish("finance.brief", {
                                    "rows": brief.get("rows", []),
                                    "source": brief.get("source"),
                                    "as_of": brief.get("as_of"),
                                    "caveat": brief.get("caveat", "")})
                                spoken = brief.get("spoken") or UNCLEAR_LINE
                            except Exception as e:  # noqa: BLE001 — a finance fault must never kill the brain
                                bus.publish("butler.error",
                                            {"reason": f"portfolio brief failed: {_reason(e)}"})
                                spoken = "I couldn't read your stock system just now, sir."
                            await _speak(spoken)
                        elif command.verb == "capture":
                            # Regression fix: before the router existed, "note/
                            # remember that ..." reached the butler, whose
                            # vault_capture tool did the append. The verb now
                            # short-circuits the model, so the server must run the
                            # (append-only, path-firewalled, off-loop) capture
                            # itself rather than stubbing a working feature.
                            try:
                                await vault_capture(command.argument or "",
                                                    vault_root_from_env())
                                await _speak("Noted, sir.")
                            except Exception as e:  # noqa: BLE001 — a capture fault must never kill the brain
                                bus.publish("butler.error",
                                            {"reason": f"capture failed: {_reason(e)}"})
                                await _speak("I couldn't save that note, sir.")
                        else:
                            # Fleet verbs (spawn/steer/stop/pull_up) land in M3 Part 2.
                            await _speak("Understood, sir — I can't run that yet.")
                    except Exception as e:  # noqa: BLE001 — a malformed Command must never kill the brain
                        bus.publish("butler.error",
                                    {"reason": f"command handling failed: {_reason(e)}"})
                        await _speak("Sorry sir, that command failed.")
                    continue

                try:
                    answer = await asyncio.wait_for(butler.ask(text), ASK_TIMEOUT_S)
                except ButlerUnavailable as e:
                    # The API/CLI layer failed -- there IS no answer. Speak a
                    # line that tells Keke what to actually do, keyed on the
                    # short reason; the raw transport text rides along as
                    # `detail` for the console (the browser reads d.reason, so
                    # the extra key is additive and safe) and is never spoken.
                    bus.publish("butler.error",
                                {"reason": e.reason, "detail": e.detail})
                    await _speak(UNAVAILABLE_LINES.get(e.reason, UNAVAILABLE_DEFAULT))
                    continue
                except Exception as e:  # noqa: BLE001 — never let the brain die
                    reason = "timed out" if isinstance(e, asyncio.TimeoutError) else str(e)
                    bus.publish("butler.error", {"reason": reason})
                    await _speak(FALLBACK_LINE)
                    continue
                citations = await _validated((answer or {}).get("citations", []))
                bus.publish("butler.answer",
                            {"display": (answer or {}).get("display", ""),
                             "citations": citations})
                await _speak(speakable((answer or {}).get("spoken")))
    finally:
        bus.unsubscribe(cid)
