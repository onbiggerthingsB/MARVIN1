# Marvin M3P2 milestone gate — the demo script

This is the last unverified piece of M3 Part 2. Everything else in the
milestone has automated tests; these nine beats are the acceptance criterion
that only a human at a real microphone can satisfy.

Read this page while you perform it. **Deviations are findings, not failures to
hide.** A beat that behaves differently from what is written here is worth more
than a beat that goes perfectly — write it down verbatim, including the exact
sentence Marvin said. The gate's output is a transcript plus your notes, not a
feeling that it went well.

---

## Before you speak

**1. Preflight.** Read-only; it starts nothing and spawns nothing.

```
cd ~/marvin && uv run python scripts/gate_preflight.py
```

Fix every numbered blocker it prints before going on. It knows about the proxy
(env vars *and* whether FlClash is actually listening), the voice keys, the
registry, leftover worktrees, stale ghosts in the fleet log, port 7777, the
access log beat 7 needs, and the suite.

**2. Observer, in a second terminal.** Leave it running for the whole demo.

```
cd ~/marvin && uv run python scripts/gate_observer.py
```

It tails `state/fleet.jsonl`, `state/server.log` and `config/projects.json`,
timestamps everything, maps what it sees onto these nine beats, and writes a
transcript to `state/gate-transcript-<stamp>.md`. It makes no HTTP calls and
never touches the bootstrap token — that token is single-redeem, and spending
it would lock your own console out of the demo.

**Start it before `marvin`, not after.** Whatever is already on disk belongs to
an earlier run, so it is printed as `(pre-existing, not credited)` and counted
towards no beat — `state/server.log` always ends in the last run's shutdown
block, and crediting that would hand beat 9 its restart for free. Anything you
perform after that line counts. (`--once` re-reads finished logs and *does*
credit them; use it only to rebuild a transcript after the fact.)

**3. Start Marvin.**

```
marvin
```

If the preflight told you to start the server yourself first (for the access
log), do that first and *then* run `marvin` — it sees `/health` answering and
will not start a second one.

**4. Say the beat number out loud before each beat.** It lands in the STT
transcript and lines the audio up with the observer's timestamps.

---

## Beat 1 — Discovery + a spoken repo confirm

**Say:** nothing yet — this beat is Marvin asking *you*.

**Expect:** "I found what looks like `<name>` at `<path>`. Is that one of
yours, sir?", then your spoken "yes", then "Noted, sir."

**This beat was a known blocker and is now wired — it has never run live.**
Preparing this gate found that nothing in the running server called
`Onboarding.refresh()` or `ask_next()`: the brain answered the repo question
(`handle_reply`) but no code path ever *asked* it. Four breaks in one chain —
no boot trigger, no voice phrase, the question was never spoken (only
rendered), and the chain stopped after one answer. All four are fixed.

Discovery now runs **at boot when nothing is confirmed yet**, and
**"find my projects"** re-runs it any time.

**Do NOT hand-seed the registry.** An empty registry is the correct state to
start from — this beat is what fills it. Seeding it skips the beat.

Answer with a **bare** yes. Say yes to your stock repo too, if it comes up:
onboarding upgrades a repo to `kind="finance"` on confirmation when its name
or path contains quant/stock/trad/invest/portfolio/finance, and beat 8 needs
that project.

After each answer Marvin proposes the **next** candidate, so expect a short
chain rather than one question. When the candidates run out it says so and
stops.

**Sharp edge, and it is correct behaviour:** the next question is queued
through the event bus, not asked instantly. If you say "yes" twice in a row
before the second question has been read to you, the second yes resolves
nothing — a repo is never confirmed by a yes uttered before its question was
spoken.

**This is the least-proven code in the milestone.** Anything it does is worth
writing down verbatim.

**Observer:** `registry: <name> CONFIRMED (<path>)` if a confirm really
happens live.

**Deviation worth writing down:** any of it working differently from the
above — including it working at all, which would mean the wiring exists
somewhere the preflight's scan missed.

---

## Beat 2 — "Start work in `<repo>` — `<small task>`"

**Say, exactly:**

> Start work in `<confirmed repo name>`, add a one-line comment at the top of
> the README saying hello from Marvin.

**The punctuation is load-bearing.** The spawn pattern requires a comma or a
colon between the repo and the task:
`start|begin|kick off work in|on <project>[,:] <task>`. Deepgram's
`smart_format` usually inserts the comma if you pause; if the spawn does not
fire, look at the transcript line on the console before blaming the fleet —
a missing comma is the usual cause, and it is worth a note either way.

**Expect to hear:** "On it, sir — `<project>`, in a fresh worktree."

**Expect to see:** a new tile, and a fresh directory under
`~/marvin/state/worktrees/`.

**Observer:**

```
#N spawned  w1 <project> -> IDLE_AT_PROMPT
    worktree /Users/.../state/worktrees/<repo>-<slug>-<stamp>
    branch   marvin/<slug>-<stamp> @ <commit>
    *** beat 2 FIRST EVIDENCE
```

**Deviation:** a refusal ("I can't spawn safely, sir" = the proxy; "I don't
know that repo" = the registry), a worktree cut from an unexpected commit, or
a second worktree appearing for one spoken task.

---

## Beat 3 — Approve **by voice**, after the readback

**Wait for the approval card**, then **listen to the whole readback**, then
say:

> Yes.

(`yes / yeah / yep / sure / ok / okay / go ahead / do it / approved` all work.)

**Three sharp edges, all correct behaviour:**

1. **Answer only after the readback finishes.** A "yes" now resolves only an
   approval Marvin has actually read aloud. Answer early and you get
   *"One moment, sir — I haven't read that request to you yet; it's on the
   console"*, and **nothing is resolved**. That is the design working. Wait
   for the sentence to end and say yes again.
2. **Glance at the card, not just the sentence.** The spoken line elides the
   **middle** of long arguments (the tail is kept — it is the half that
   identifies things). The console card carries the **full** command. Approving
   a command you only heard summarised is the exact habit this beat exists to
   discourage.
3. **A worktree is not a sandbox.** If the readback says the target is outside
   the worktree, that is real: `Write`, `Edit` and `Bash` take absolute paths.
   Say no if you do not like it, and write down what it asked for.

**Expect after the yes:** the worker resumes and finishes the turn.

**Observer:** `permission_wait` → `permission_done` → `turn_done`, all tagged
beat 3.

**Deviation:** a yes that resolves an approval you were never read; a card with
no spoken line (or vice versa); a spoken line whose *tail* differs from the
card's command; the tile stuck on `WAITING_PERMISSION` after the yes.

---

## Beat 4 — "What's running?"

**Say:** "What's running?"

**Expect:** one spoken fleet line naming the project and its state.

**Observer:** *nothing — this beat is spoken only and leaves no durable
record.* The observer marks beats 4 and 5 `BLIND` on purpose rather than
pretending to prove them. **Write down the sentence you heard, verbatim.**

**Deviation:** a state in the sentence that contradicts the tile on screen;
"unknown" for a worker that is plainly working (that word is reserved for
failed probes).

---

## Beat 5 — "Pull it up"

**Say, exactly:** "Pull it up." (or "Pull that up.")

The phrase must stand alone; `pull up <project>` is a different, also-valid
command, but the pronoun form is the one this beat tests.

**Expect:** the transcript pane fills with that worker's messages.

**Observer:** blind again — record what you saw by hand.

**Deviation:** an empty pane, the wrong worker's transcript, or a pane that
fills only after a manual reload.

---

## Beat 6 — [Open in Terminal]

**Click** the tile's **Open in Terminal** button.

**Expect:** Terminal.app opens running
`cd <worktree> && claude --resume <session-id>`, the tile goes **DETACHED**,
and Marvin says "`<project>` is yours in the terminal, sir."

**Observer:**

```
#N detached  w1 <project> -> DETACHED
    session  <session id>
    *** beat 6 FIRST EVIDENCE
```

**Deviation:** the button refusing (each refusal sentence is specific — "still
starting", "no resumable session yet", "already detached"); a Terminal window
opening while the tile does *not* say DETACHED (that would be two drivers on
one session, the exact accident the lockout prevents); a `detached` record
carrying no session id.

---

## Beat 7 — the hooks keep arriving from the detached session

### This is the beat with no automated proof anywhere in the codebase. Take your time.

**Do:** in the Terminal window that just opened, type a small prompt at the
`claude` prompt — e.g. `list the files in this directory` — and send it. Then
watch two places at once.

**Expect on the console:** the DETACHED tile's state **keeps moving** while you
type in the terminal. Marvin holds no SDK stream for that session any more —
`handoff` closed the client and verified its tasks were dead before recording
`detached` — so anything that moves that tile arrived over HTTP from the
worktree's own hook settings.

**Expect in `state/server.log`:** `POST /hooks` lines.

**Observer — two lanes, and it labels them:**

```
POST /hooks #3 (AFTER the detach) — most recent detached session <id>
    *** beat 7 FIRST EVIDENCE
```

```
#N activity  w1 <project> -> DETACHED
    >>> BEAT 7: `activity` reached a DETACHED worker — only a /hooks POST
        from the worktree session can do that (session <id>)
```

Lane A (the access log) is silent if the server was started with
`--no-access-log` — which is what `bin/marvin` does. The preflight prints the
exact fix. **Lane B still proves the beat either way**, and lane B is the
stronger of the two because it carries the session id and is causally tied to
the tile you are watching.

**Deviation:** no hook traffic at all after the detach (check
`NO_PROXY` includes **both** `localhost` and `127.0.0.1` — curl matches the
literal URL host, so `localhost` alone sends the hook POSTs through the proxy
and they never arrive); `fleet.unknown_session` events; hook POSTs arriving but
the tile not moving.

---

## Beat 8 — "How are the picks doing?"

**Say:** "How are the picks doing?"

**Expect, in this order:**

1. **the source question, on the FIRST ask:** "I'll read the picks from
   `<file>` — correct, sir?"
2. your spoken **"yes"**
3. **then** the brief.

**Sharp edge:** the question is asked **once** and then **pinned** to the
project. If it was already confirmed during earlier testing, Marvin goes
straight to the brief and beat 8 *looks skipped*. The preflight warns when a
`data_source` is already pinned and tells you which field to clear.

Answer the source question with a **bare** yes. "Sure, stop soccer" is not
consent — an affirmative opener carrying a real request falls through with the
question still pending, deliberately, because this one is money-adjacent.

**Observer:** `registry: <name> data_source pinned -> <file>` → beat 8.

**Deviation:** the brief arriving without the question (note whether a
`data_source` was already pinned — that is the documented cause, not a bug);
the named file differing from the file the brief actually reads; any trade verb
being acted on rather than refused.

---

## Beat 9 — kill the server mid-worker, restart, hear the report

**Sequencing trap — read first.** The beat-2 worker is **DETACHED** by now, and
a detached ghost is re-announced as *"already detached before the restart"*,
**not** as interrupted. To hear the interrupted-worker report you need a worker
that is **live** at kill time. So:

1. Start a fresh worker and leave it mid-flight:
   > Start work in `<confirmed repo>`, count the lines in every file and write
   > the totals to COUNTS.txt.
2. While it is working (tile `ACTIVE_TURN`), kill the server:

```
pkill -f 'uvicorn server.app'
```

   A plain `pkill` (SIGTERM) is the clean path: the lifespan runs, `close_all`
   shuts the workers down but **deliberately does not log `session_end`**, and
   the snapshot rotates `fleet.jsonl` to `fleet.jsonl.1`. The observer prints
   `fleet.jsonl ROTATED`. `kill -9` also works and exercises the torn-tail
   repair instead — either is a valid demo, note which one you did.

3. Restart the server the same way you started it (with the access log if the
   preflight told you to), then `marvin`.

**Expect:** Marvin reports the interrupted worker out loud, and a ghost tile
appears with its worktree preserved.

**Sharp edge:** the ghost is re-announced on **every** boot while its worktree
still exists. A second restart repeats the report. That is expected and is
already a documented known-open item — note it, do not chase it.

**Observer:** `fleet.jsonl ROTATED`, then `server: Started server process`
after a shutdown → beat 9.

**Deviation:** silence about the interrupted worker; a claim that a worker is
*waiting* for something (nothing is re-armed across a restart — no approval
survives); a missing worktree; "I couldn't read my own fleet log" (that
sentence is honest, but it means the log was damaged — capture the
`state/fleet.jsonl.torn-*` file as evidence).

---

## When you are done

1. **Ctrl-C the observer.** It prints the beat summary and closes the
   transcript. Read the summary out loud to yourself — every beat marked
   `NONE` that is not 4 or 5 is a beat with no evidence.
2. **Paste the transcript** (`state/gate-transcript-<stamp>.md`) into the M3P2
   milestone report, together with:
   - the verbatim sentences you heard for beats 4 and 5 (the observer is blind
     to those);
   - every deviation, with the exact wording;
   - which beats you consider passed, and why.
3. **The worktrees stay.** Nothing removes them automatically — each one holds
   a diff you may still want. Clean them up deliberately when the report is
   written:
   `git -C <origin repo> worktree remove --force <path>` then
   `git -C <origin repo> branch -D marvin/<...>`. Leaving them means the ghosts
   keep being re-announced on every boot.
4. **`git tag m3p2` is the last step — and only if the gate genuinely passed.**
   A tag is a claim that the milestone's acceptance criterion was met by a
   human at a microphone. If any beat is unproven — beat 1 is a known blocker
   as of this writing — say so in the report and **do not tag**. An untagged
   milestone with an honest findings list is worth more than a tagged one with
   a beat quietly dropped.
