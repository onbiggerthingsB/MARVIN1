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
FLEET_TIMEOUT_S = 90   # spawn = worktree + connect + first query; generous but bounded
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


async def _brief_and_publish(bus, project) -> str:
    """One place builds the finance turn — the portfolio verb and a
    just-confirmed source both use it, and both name the EXACT project to
    brief. No re-deriving here: with two confirmed finance projects, a fresh
    lookup could select the other one — unpinned, so its _collect would fall
    back to scanning for a file Keke never confirmed (spec §16). Returns the
    spoken line. Callers guard it; the .get() reads keep a shape drift from
    raising on their own."""
    brief = await portfolio_brief(project) or {}
    bus.publish("finance.brief", {
        "rows": brief.get("rows", []),
        "source": brief.get("source"),
        "as_of": brief.get("as_of"),
        "caveat": brief.get("caveat", "")})
    return brief.get("spoken") or UNCLEAR_LINE


async def run_butler_brain(bus, butler, speaker, turnlog, validate_citations=None,
                           router=None, registry=None, onboarding=None, finance=None,
                           fleet=None):
    """Subscribe to the bus and drive the butler. Never raises out of the loop.

    `validate_citations` is an optional async callable taking the model's list of
    cited titles and returning the subset that resolves to real vault notes.
    None (the default) means no validation, which keeps this loop importable and
    testable without a vault behind it; server/app.py passes the real one.

    `router`/`registry`/`onboarding` wire in M3's deterministic layer: dangerous
    verbs are parsed BEFORE the model ever sees an utterance, and a pending
    confirmation or approval owns the next reply. All keyword-optional so every
    M2 caller and test keeps working. `finance` takes the §16 SourceGate: the
    first portfolio ask names the output file and waits for a spoken yes.
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
            if etype == "approval.request":
                # A worker is blocked on Keke's word; the card is on screen and
                # this is the spoken half (readback of project, tool, args,
                # risk — composed by the fleet). Guarded: a malformed event
                # must cost this speech, never the loop.
                question, nonce = "", ""
                try:
                    question = str((data or {}).get("question") or "")[:400]
                    nonce = str((data or {}).get("nonce") or "")
                except Exception:  # noqa: BLE001
                    question, nonce = "", ""
                # Stale requests are never read aloud (M3P2 review, C3). This
                # event may have been queued behind a 120-second butler.ask
                # while `stop soccer` rejected the future and TOOK the nonce,
                # or a click resolved it, or its TTL swept it. Asking about a
                # dead worker is a question with no answer: a "no" in reply
                # finds nothing pending and falls through to the butler, which
                # answers it as conversation.
                verified = True
                if nonce and router is not None:
                    try:
                        verified = any(a.nonce == nonce
                                       for a in router.pending_approvals())
                    except Exception as e:  # noqa: BLE001 — a router fault must not kill the brain
                        bus.publish("butler.error",
                                    {"reason": f"approval readback check "
                                               f"failed: {_reason(e)}"})
                        # Can't verify and can't mark: say the generic line
                        # rather than assert details we cannot stand behind.
                        question, nonce = "", ""
                    if not verified:
                        continue
                await _speak(question or "A worker needs permission, sir — "
                                         "the card is on screen.")
                # AFTER the await returns, never before it: `spoken` means the
                # sentence finished, and only then may a bare yes resolve THIS
                # nonce. An event with no nonce (a malformed publish) is spoken
                # as the generic alert and marks nothing — so it can still warn
                # Keke without making anything voice-resolvable.
                if nonce and router is not None:
                    try:
                        router.mark_spoken(nonce)
                    except Exception as e:  # noqa: BLE001 — see above
                        bus.publish("butler.error",
                                    {"reason": f"approval readback failed: "
                                               f"{_reason(e)}"})
                continue
            if etype in ("fleet.spoken", "fleet.recovered"):
                text_out = ""
                try:
                    if etype == "fleet.recovered":
                        n = int((data or {}).get("count") or 0)
                        text_out = ("Sir, one worker was interrupted by a restart. "
                                    "Its worktree is preserved; I no longer hold "
                                    "its session." if n == 1 else
                                    f"Sir, {n} workers were interrupted by a "
                                    "restart. Their worktrees are preserved; I no "
                                    "longer hold their sessions.")
                    else:
                        text_out = str((data or {}).get("text") or "")[:400]
                except Exception:  # noqa: BLE001
                    text_out = ""
                if text_out:
                    await _speak(text_out)
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

                # A pending §16 data-source question owns the next yes/no.
                # Same discipline as onboarding: awaiting read inside the
                # guard and BEFORE handle_reply clears it.
                if finance is not None:
                    outcome, awaiting = "ignored", False
                    try:
                        awaiting = bool(finance.awaiting)
                        outcome = await finance.handle_reply(text)
                    except Exception as e:  # noqa: BLE001 — the gate must not kill the brain
                        bus.publish("butler.error", {"reason": f"source gate failed: {e}"})
                    if outcome == "confirmed":
                        # The yes doubles as "go": brief the EXACT project the
                        # gate just pinned. Re-deriving here could select a
                        # second confirmed finance project — unpinned, so the
                        # brief would scan a file Keke never approved (§16).
                        try:
                            spoken = await _brief_and_publish(
                                bus, getattr(finance, "confirmed_project", None))
                        except Exception as e:  # noqa: BLE001 — finance must not kill the brain
                            bus.publish("butler.error",
                                        {"reason": f"portfolio brief failed: {_reason(e)}"})
                            spoken = "I couldn't read your stock system just now, sir."
                        await _speak(spoken)
                        continue
                    if outcome == "rejected":
                        await _speak("Understood, sir — point me at the right file "
                                     "and I'll use that.")
                        continue
                    if awaiting and bare_yes_no(text):
                        await _speak("About the data file, sir — is that a yes or a no?")
                        continue

                # Dangerous verbs are parsed deterministically, never by the model.
                command = None
                if router is not None and registry is not None:
                    has_fleet = False
                    try:
                        has_fleet = bool(fleet is not None and fleet.workers)
                    except Exception as e:  # noqa: BLE001 — a fleet fault must not break parsing
                        bus.publish("butler.error",
                                    {"reason": f"fleet status failed: {e}"})
                    try:
                        command = router.parse(text, registry, has_fleet=has_fleet)
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
                # The affirm vocabulary DOES leak into this tie the other way:
                # an affirm opener defeats the anchored verb patterns, so
                # "sure, stop soccer" parses as no command and lands in
                # resolve_approval carrying both polarities. The router fails
                # closed on that shape ("unclear", below) — it never resolves,
                # and the brain asks instead.
                # EVERY router touch below sits inside ONE try: the pending
                # check, the resolve call, and the dereference of the returned
                # approval. Two shipped bugs came from expressions evaluated
                # OUTSIDE the guard meant to protect them (an unguarded
                # preconnect, then wire arithmetic in an argument list); the
                # bare pending_approvals() call and the approval.project /
                # .tool / .nonce reads were the third instance — a router
                # fault (or a contract-violating ("approved", None) return)
                # raised straight out of the loop and killed the brain.
                state, approval, resolving = "none", None, False
                resolved_event = None
                if router is not None:
                    try:
                        if (router.pending_approvals()
                                and not (command is not None
                                         and (command.project
                                              or command.needs_disambiguation))):
                            resolving = True
                            state, approval = router.resolve_approval(text, time.time())
                            if state in ("approved", "denied") and approval is not None:
                                # Deliver FIRST, publish second — the order
                                # /approval already uses (take, deliver,
                                # publish). Publish-first announced a
                                # resolution the worker might never receive:
                                # a raising deliver left the card removed,
                                # the status reading "approved", and the
                                # worker blocked until its TTL published a
                                # contradicting "expired".
                                if fleet is not None:
                                    # unblock the worker's can_use_tool future;
                                    # same try — a delivery fault is an approval
                                    # fault, handled by this guard.
                                    fleet.deliver_approval(approval.nonce,
                                                           state == "approved")
                                # The payload is BUILT here (the approval
                                # dereferences stay inside the guard — a
                                # contract-violating ("approved", None) return
                                # is the class of bug this try exists for) but
                                # PUBLISHED below, outside it. A publish that
                                # raised in here reached the handler's
                                # `resolving` branch and spoke "approval
                                # handling failed on my side" over a tool call
                                # that was in fact approved and already
                                # running — the last contradictory-sentence
                                # path in this loop.
                                resolved_event = {
                                    "outcome": state, "project": approval.project,
                                    "tool": approval.tool, "nonce": approval.nonce}
                    except Exception as e:  # noqa: BLE001 — a router fault must never kill the brain
                        bus.publish("butler.error",
                                    {"reason": f"approval handling failed: {e}"})
                        if resolving:
                            # The utterance was consent-shaped enough to reach
                            # the resolver, and resolution FAILED. Falling
                            # through would hand "yes, go ahead" to the
                            # butler, which answers some unrelated question
                            # while the worker stays blocked. Speak the truth
                            # and end the turn. (A pending_approvals() fault
                            # keeps the M3.1 fall-through: the utterance was
                            # never classified, so the butler still owns it.)
                            await _speak("Sorry sir — approval handling "
                                         "failed on my side; details are on "
                                         "screen.")
                            continue
                        state, approval = "none", None
                if resolved_event is not None:
                    # Console notification only — the decision is already with
                    # the worker and the tool is already running. Its failure
                    # may cost the card an update; it may NOT cost the brain
                    # (an unguarded raise here ends the loop) and it may NOT
                    # turn into a spoken failure, because "Approved, sir." is
                    # the truth at this point. Nothing to report it THROUGH
                    # either: the bus is the thing that just failed.
                    try:
                        bus.publish("approval.resolved", resolved_event)
                    except Exception:  # noqa: BLE001 — see above
                        pass
                if state in ("approved", "denied"):
                    await _speak("Approved, sir." if state == "approved"
                                 else "Denied, sir.")
                    continue
                if state == "unspoken":
                    # A yes arrived before (or instead of) the readback. It
                    # resolves NOTHING — but it must not vanish either: the
                    # worker is blocked and Keke thinks he just answered. Say
                    # what actually happened. When the readback is merely
                    # QUEUED behind this turn (the common case) the loop reads
                    # it out immediately after this sentence, and a second yes
                    # then lands on something he has heard; when the card was
                    # lost to a bus eviction the console is where it lives.
                    await _speak("One moment, sir — I haven't read that "
                                 "request to you yet; it's on the console.")
                    continue
                if state == "ambiguous":
                    await _speak("More than one approval is pending, sir — "
                                 "name the project.")
                    continue
                if state == "unclear":
                    # Mixed polarity ("sure, cancel that", "no, go ahead"):
                    # the router refused to resolve. The approval is still
                    # pending and the card is still on screen — ask.
                    await _speak("Sir, that sounded like both a yes and a no "
                                 "— approve or deny?")
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
                            # Whole path guarded: lookup, the gate, the brief
                            # await, every dict read. A raise costs this turn,
                            # never the loop.
                            try:
                                proj = find_finance_project(registry)
                                if (proj is not None and finance is not None
                                        and not getattr(proj, "data_source", None)):
                                    # §16: never read a file Keke has not named
                                    # and approved. Ask first; the reply lands
                                    # in the gate block above next turn.
                                    question = await finance.ask(proj)
                                    spoken = question or (
                                        f"I couldn't find a readable output file "
                                        f"in {proj.name}, sir.")
                                else:
                                    spoken = await _brief_and_publish(bus, proj)
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
                        elif command.verb in ("spawn", "steer", "stop",
                                              "status", "pull_up") and fleet is not None:
                            # Fleet dispatch. Inside the surrounding dispatch
                            # try — any raise costs this turn ("that command
                            # failed"), never the loop. FLEET_TIMEOUT_S bounds
                            # the awaits: a wedged spawn must not wedge JARVIS.
                            if command.verb == "spawn":
                                if not command.path:
                                    spoken = "Which project, sir?"
                                else:
                                    spoken = await asyncio.wait_for(
                                        fleet.spawn(command.project or "",
                                                    command.path,
                                                    command.argument or ""),
                                        FLEET_TIMEOUT_S)
                            elif command.verb == "steer":
                                spoken = fleet.steer_path(command.path or "",
                                                          command.argument or "")
                            elif command.verb == "stop":
                                spoken = await asyncio.wait_for(
                                    fleet.stop(command.path or ""), FLEET_TIMEOUT_S)
                            elif command.verb == "status":
                                spoken = fleet.status_line()
                            else:                       # pull_up
                                target = command.path
                                if target is None:
                                    workers = list(fleet.workers)
                                    if len(workers) == 1:
                                        target = workers[0].path
                                if target is None:
                                    spoken = "Which one, sir?"
                                else:
                                    bus.publish("fleet.transcript", {
                                        "path": target,
                                        "project": command.project,
                                        "lines": fleet.transcript(target)})
                                    spoken = fleet.one_breath(target)
                            await _speak(spoken)
                        else:
                            # No fleet injected (tests, degraded boot): honest.
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
