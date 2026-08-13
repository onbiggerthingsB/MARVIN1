import { makeFrameSink } from "/static/micbuffer.js";

const $ = (sel) => document.querySelector(sel);
const handlers = new Map();
let audioCtx = null;
const chimes = {};

function connectSSE() {
  const es = new EventSource("/events");
  const dispatch = (type) => (e) => {
    (handlers.get(type) || []).forEach((h) => h(JSON.parse(e.data)));
  };
  ["wake", "command.received", "stt.interim", "stt.final", "stt.utterance",
   "stt.error", "tts.start", "tts.done", "metrics.turn", "metrics.error",
   "butler.answer", "butler.error", "router.command", "confirm.request",
   "confirm.result", "registry.updated", "finance.brief",
   "fleet.update", "fleet.message", "fleet.error", "fleet.transcript",
   "fleet.unknown_session", "fleet.spoken", "fleet.recovered",
   "fleet.handoff", "approval.request", "worktrees.survey",
   "approval.resolved", "social.results", "social.error"].forEach((t) =>
    es.addEventListener(t, dispatch(t)));
  // Deliberate: reconnect fresh (no Last-Event-ID). Replaying stale tts.start/
  // stt.utterance on a live-voice UI would double-trigger playback. Server-side
  // replay (ring buffer + bus.gap) stays ready for a future non-voice consumer.
  //
  // But "no replay" also meant "no resync": everything published during the
  // gap was lost permanently, INCLUDING approval.request. The card never
  // appeared, nothing re-fetched, and a worker could sit blocked for its full
  // 600s TTL with no card and no spoken line. /fleet is the resync, and it
  // runs on every (re)connect — onopen fires for the first connection too.
  es.onopen = () => { refreshFleet(); refreshWorktrees(); };
  es.onerror = () => { es.close(); setTimeout(connectSSE, 1000); };
}

// Tiles AND still-pending approval cards, from the one cookie-authed route.
// Both renderers are idempotent — tiles keyed by worker, cards by nonce — so a
// duplicate refresh cannot double-paint. Failure is silent by design: a missing
// resync must not block the console coming online.
function refreshFleet() {
  fetch("/fleet").then((r) => r.json()).then((d) => {
    (d.workers || []).forEach(renderFleetTile);
    (d.approvals || []).forEach(renderApproval);
  }).catch(() => {});
}

async function loadChime(name) {
  const buf = await (await fetch(`/static/chimes/${name}.wav`)).arrayBuffer();
  chimes[name] = await audioCtx.decodeAudioData(buf);
}

window.marvin = {
  onEvent(type, h) { handlers.set(type, [...(handlers.get(type) || []), h]); },
  setStatus(text) { $("#status").textContent = text; },
  playChime(name) {
    if (!audioCtx || !chimes[name]) return;
    const src = audioCtx.createBufferSource();
    src.buffer = chimes[name];
    src.connect(audioCtx.destination);
    src.start();
  },
  audioCtx: () => audioCtx,
};

$("#setup-btn").addEventListener("click", async () => {
  try {
    audioCtx = new AudioContext();
    await audioCtx.resume();
    await Promise.all(["listen", "ack", "done"].map(loadChime));
    // Mic permission is requested here once; the stream is stopped immediately.
    const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    mic.getTracks().forEach((t) => t.stop());
    window.marvin.playChime("done");
    $("#setup-overlay").hidden = true;
    $("#console").hidden = false;
    connectSSE();
    // Belt to connectSSE's onopen braces. SSE carries every LATER tile, but it
    // reconnects without a Last-Event-ID, so anything published before this
    // browser existed — above all the restart ghosts, published once at boot,
    // and any approval already waiting — reaches the page only through this
    // route. Idempotent, so running it here and on every connect is safe.
    refreshFleet();
    refreshWorktrees();
    window.marvin.setStatus("online — hold to talk");
  } catch (err) {
    $("#setup-status").textContent = `setup failed: ${err.message} — fix and click again`;
  }
});

window.marvin.onEvent("wake", () => {
  window.marvin.playChime("listen");
  window.marvin.setStatus("yes?");
});

// ---- press-and-hold mic → /mic WebSocket -------------------------------
let micWS = null, micStream = null, micNode = null, analyser = null;
let micAborting = false;
// The socket of the PREVIOUS hold, still open on its flush grace. Tracked so a
// new press can retire it: two open /mic sockets mean two Deepgram relays
// publishing stt.* into one bus, and the tail of the last utterance landing on
// top of this one.
let micDraining = null;
// Is a press outstanding? micWS is not that answer — it is still null while
// setup is in flight, and it is nulled the instant a release starts. A hold
// whose release event never arrives (window switch, cmd-tab to the terminal)
// has to be recoverable, and this is what the recovery reads.
let micHeld = false;

function closeDraining() {
  if (!micDraining) return;
  const ws = micDraining;
  micDraining = null;
  ws.close();
}

// Shared, null-safe teardown. Does NOT close micWS if it has already been
// nulled by the caller (graceful stop hands the socket off to a delayed close).
function teardownMic() {
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (micNode) micNode.disconnect();
  if (micWS) micWS.close();
  micWS = null;
  micStream = null;
  micNode = null;
  analyser = null;
  $("#ptt").classList.remove("live");
}

async function startTalking() {
  if (micWS) return;
  micHeld = true;
  closeDraining();   // one live relay at a time
  micAborting = false;
  try {
    $("#ptt").classList.add("live");
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (micAborting) { teardownMic(); return; } // released during setup
    const ctx = window.marvin.audioCtx();
    await ctx.audioWorklet.addModule("/static/worklet.js");
    if (micAborting) { teardownMic(); return; } // released during setup
    const src = ctx.createMediaStreamSource(micStream);
    analyser = ctx.createAnalyser();
    src.connect(analyser);
    micNode = new AudioWorkletNode(ctx, "mic-processor");
    src.connect(micNode);
    // GAP 2 — the chime marks CAPTURE, not the press. It used to fire before
    // getUserMedia even resolved, so it invited speech into a mic that did not
    // exist yet; that audio was never captured, so no buffer could recover it
    // (unlike the pre-socket window, which micbuffer.js covers). Here the
    // worklet is wired and framing, so everything from this tone onward is
    // captured — the socket may still be CONNECTING, and that is fine. A
    // failed activation now reaches the catch below and says so instead of
    // chiming into a dead mic.
    window.marvin.playChime("listen");
    // `ws` is captured, not read back off micWS. A release lands during the
    // handshake often enough — it is the last unguarded gap in this setup —
    // and it nulls micWS, so a handler that reached for the global fired
    // `null.send(...)` and died. Compare against micWS instead: not the
    // current socket, nothing to do.
    const ws = new WebSocket(`ws://${location.host}/mic`);
    micWS = ws;
    ws.binaryType = "arraybuffer";
    // The worklet starts posting frames NOW, while the socket is still
    // CONNECTING. The old code attached this handler only inside ws.onopen,
    // which left the early frames to the browser's implicit MessagePort
    // queue — unbounded, browser-dependent, and pure luck that they arrived
    // after readyState hit 1 instead of dying on the old handler's readyState
    // guard. Attach the handler immediately and route every frame through a
    // bounded, ordered sink instead: CONNECTING buffers (capped — see
    // micbuffer.js), OPEN sends with any backlog flushed first, later states
    // discard. The sink is per-press closure state, so an abandoned press's
    // buffered audio is dropped with the press — it can never flush into the
    // next session.
    const sink = makeFrameSink();
    micNode.port.onmessage = (e) => {
      // The flush ack (GAP 1). MessagePort delivery is FIFO and the worklet
      // posts its tail frame BEFORE the ack, so by the time this lands every
      // frame is already through the sink above it. Resolve the release.
      if (e.data && e.data.type === "flushed") { resolveFlush(); return; }
      if (micWS !== ws) return;                 // press ended or superseded
      sink.frame(ws.readyState, e.data, (f) => ws.send(f));
    };
    ws.onopen = () => {
      if (micWS !== ws) { ws.close(); return; } // released while CONNECTING
      ws.send(JSON.stringify({ type: "start", encoding: "linear16",
        sample_rate: 16000, channels: 1, t_hold: Date.now() }));
      // Everything spoken during the handshake goes out here, in capture
      // order, BEFORE any live frame — and after the `start` frame above,
      // which must stay first on the wire.
      sink.flush((f) => ws.send(f));
    };
    // An abnormal close used to leave micWS pointing at a dead socket, and
    // every later press returned at the guard above — hold-to-talk simply
    // stopped, until a reload. Reset instead, so the next press starts clean.
    ws.onclose = () => { if (micWS === ws) teardownMic(); };
    ws.onerror = () => { if (micWS === ws) teardownMic(); };
    drawWave();
  } catch (err) {
    teardownMic();
    window.marvin.setStatus("mic error — " + err.message);
    return;
  }
}

// GAP 1 — the tail. The worklet only posts a frame once it has ~60ms, so at
// release it is still holding a partial, and one more frame may be in flight
// on the port. Both used to be discarded, clipping the END of every utterance
// (~<=120ms) — which is how "Start" reached the router as "Star", and the
// router anchors its commands on that opening verb. So: ask the worklet to
// flush, wait for the ack, THEN send stop. Bounded, because a worklet that
// never answers must not wedge the button: losing the tail beats a dead UI.
let resolveFlush = () => {};
const FLUSH_MS = 200;

function flushWorklet(node) {
  if (!node) return Promise.resolve();
  return new Promise((done) => {
    let settled = false;
    const finish = () => { if (!settled) { settled = true; resolveFlush = () => {}; done(); } };
    resolveFlush = finish;
    setTimeout(finish, FLUSH_MS);
    try { node.port.postMessage({ type: "flush" }); } catch { finish(); }
  });
}

let micReleasing = false;

async function stopTalking() {
  // The flush below is the first await this function has ever had, and it
  // opens a window where micHeld is already false while micWS is still set —
  // which is exactly the shape endHoldOnLostFocus fires on, so a blur during
  // the flush re-entered and double-stopped. One release per press.
  if (micReleasing) return;
  micReleasing = true;
  try {
    micHeld = false;
    // Drain the worklet BEFORE anything below nulls micWS or disconnects the
    // node — the tail frames ride the same handler and socket as live audio.
    if (micWS && micNode) await flushWorklet(micNode);
    stopTalkingNow();
  } finally {
    micReleasing = false;
  }
}

function stopTalkingNow() {
  if (!micWS) {
    // Setup still in flight (or already torn down): signal the abort AND stop
    // whatever the in-flight setup may have already opened, so the mic can't
    // be left hot.
    micAborting = true;
    teardownMic();
    return;
  }
  const ws = micWS;
  const open = ws.readyState === 1;
  if (open) ws.send(JSON.stringify({ type: "stop", t_release: Date.now() }));
  micWS = null;    // null first so teardownMic won't close it
  teardownMic();   // stops tracks, disconnects node, clears .live
  if (open) {
    micDraining = ws;
    // Give the server time to flush finals — but only until the next press.
    setTimeout(() => { if (micDraining === ws) closeDraining(); }, 3000);
  } else {
    // Released before the handshake finished. No `start` frame ever went out,
    // so there is no session and nothing to flush; parking the socket for the
    // grace period would just leave the server holding a mic that never spoke.
    ws.close();
  }
  window.marvin.setStatus("thinking…");
}

function drawWave() {
  const canvas = $("#wave"), g = canvas.getContext("2d");
  const data = new Uint8Array(analyser ? analyser.fftSize : 0);
  (function frame() {
    if (!micWS) { g.clearRect(0, 0, canvas.width, canvas.height); return; }
    analyser.getByteTimeDomainData(data);
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.strokeStyle = "#2e6da4"; g.beginPath();
    data.forEach((v, i) => g.lineTo(i / data.length * canvas.width, v / 255 * canvas.height));
    g.stroke();
    requestAnimationFrame(frame);
  })();
}

const ptt = $("#ptt");
let pttPointer = null;   // pointerId of the press that owns the mic

ptt.addEventListener("pointerdown", (e) => {
  pttPointer = e.pointerId;
  // Capture the pointer. Without it the button only hears a release that
  // happens ON the button, so `pointerleave` had to stand in as the safety
  // net — and it fires on any drift, which cut the mic mid-sentence for
  // anyone who moved the mouse while talking. With capture, pointerup and
  // pointercancel come here wherever the cursor ended up, and leave never
  // fires during the hold at all.
  try { ptt.setPointerCapture(e.pointerId); } catch (err) { /* backstop below */ }
  startTalking();
});

function releasePointer(e) {
  if (pttPointer === null || (e && e.pointerId !== pttPointer)) return;
  pttPointer = null;
  stopTalking();
}
ptt.addEventListener("pointerup", releasePointer);
ptt.addEventListener("pointercancel", releasePointer);
// Backstop for a browser that refused the capture: the release still reaches
// window. Guarded by pttPointer, so a click anywhere else cannot cut a hold.
window.addEventListener("pointerup", releasePointer);
window.addEventListener("pointercancel", releasePointer);

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && !$("#setup-overlay").hidden) return;
  if (e.code === "Space" && !e.repeat) { e.preventDefault(); startTalking(); }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space") { e.preventDefault(); stopTalking(); }
});

// A hold whose release never arrives: hold Space, cmd-tab to the terminal, and
// keyup lands in the other window. micWS stayed set, the mic stayed hot, and
// every later press returned at the `if (micWS) return` guard — hold-to-talk
// dead until a reload. Losing the window ends the hold.
function endHoldOnLostFocus() {
  if (!micHeld && !micWS) return;
  pttPointer = null;
  stopTalking();
}
window.addEventListener("blur", endHoldOnLostFocus);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) endHoldOnLostFocus();
});

window.marvin.onEvent("stt.interim", (d) => {
  const t = $("#transcript");
  t.classList.add("interim");
  t.textContent = d.text;
});
window.marvin.onEvent("stt.utterance", (d) => {
  $("#transcript").classList.remove("interim");
  $("#transcript").textContent = d.text;
  window.marvin.playChime("ack");
});
window.marvin.onEvent("stt.error", (d) => {
  window.marvin.setStatus("couldn't hear you — " + (d.reason || "audio error"));
  if (typeof playClip === "function") playClip("cannot_hear");
});

window.marvin.onEvent("butler.answer", (d) => {
  $("#answer").textContent = d.display || "";
  const box = $("#citations");
  box.textContent = "";
  (d.citations || []).forEach((name) => {
    const chip = document.createElement("span");
    chip.className = "cite";
    chip.textContent = name;
    box.appendChild(chip);
  });
  window.marvin.setStatus("online — hold to talk");
});
window.marvin.onEvent("butler.error", (d) => {
  // Clear the previous turn's answer AND its citation chips. Leaving them up
  // under an error line reads as if the stale answer belongs to the question
  // that just failed.
  $("#answer").textContent = "";
  $("#citations").textContent = "";
  window.marvin.setStatus("brain error — " + (d.reason || "unavailable"));
});
// Metrics failures are NOT brain failures: a TurnLog hiccup on tts.done must
// never blank a correct answer that Marvin is still speaking. Status line only —
// no clearing of #answer / #citations.
window.marvin.onEvent("metrics.error", (d) => {
  window.marvin.setStatus("metrics: " + (d.reason || "unavailable"));
});

// ---- /audio WebSocket → MediaSource playback ---------------------------
function connectAudio() {
  // The playback pipeline and the tts.start subscription are wired up exactly
  // once and persist across WS reconnects (state lives on the function object).
  // This keeps reconnects from stacking duplicate reset handlers or leaving the
  // drain bound to a stale, closed connection's buffer.
  if (!connectAudio.init) {
    connectAudio.init = true;
    const S = connectAudio.state = { ms: null, sb: null, el: null, queue: [] };

    // Single drain: append the next queued chunk whenever a buffer exists and is
    // idle. Driven by the WS onmessage handler, sourceopen, and updateend.
    const drain = () => {
      if (!S.sb || S.sb.updating || !S.queue.length) return;
      try {
        S.sb.appendBuffer(S.queue.shift());
      } catch (_) {
        // A late chunk arriving after the buffer/MediaSource was torn down
        // throws InvalidStateError — drop it quietly rather than break the stream.
      }
    };
    connectAudio.drain = drain;

    // Each tts.start begins a fresh utterance. Tear the previous playback down
    // FIRST — pause, revoke its object URL, and null the buffer immediately so no
    // in-flight chunk appends to the stale SourceBuffer — THEN build the new
    // MediaSource + Audio.
    window.marvin.onEvent("tts.start", () => {
      if (S.el) { S.el.pause(); URL.revokeObjectURL(S.el.src); }
      S.sb = null;
      S.queue = [];
      S.ms = new MediaSource();
      S.el = new Audio();
      S.el.src = URL.createObjectURL(S.ms);
      S.ms.addEventListener("sourceopen", () => {
        S.sb = S.ms.addSourceBuffer("audio/mpeg");
        S.sb.addEventListener("updateend", drain);
        drain();
      });
      S.el.play().catch(() => {});   // AudioContext already unlocked by setup click
    });
  }

  const S = connectAudio.state;
  const ws = new WebSocket(`ws://${location.host}/audio`);
  ws.binaryType = "arraybuffer";
  // Never drop early chunks (incl. those before the first sourceopen): always
  // queue, then drain — drain is a no-op until the buffer is ready.
  ws.onmessage = (e) => { S.queue.push(e.data); connectAudio.drain(); };
  ws.onclose = () => setTimeout(connectAudio, 1000);
  // Exactly one ping interval: clear the prior connection's before reconnecting
  // (clearInterval(undefined) is a safe no-op on the first run).
  clearInterval(connectAudio.ping);
  connectAudio.ping = setInterval(() => ws.readyState === 1 && ws.send("ping"), 10000);
}
connectAudio();
window.marvin.onEvent("tts.done", () => window.marvin.setStatus("online — hold to talk"));

// ---- canned clips + metrics footer -------------------------------------
let clipManifest = {};
fetch("/static/clips/manifest.json").then((r) => r.json())
  .then((m) => { clipManifest = m; })
  .catch(() => {});

const clipEls = {};
function playClip(slug) {
  if (!clipManifest[slug]) return;
  clipEls[slug] = clipEls[slug] || new Audio(clipManifest[slug]);
  clipEls[slug].currentTime = 0;
  clipEls[slug].play().catch(() => {});
}

// ---- M3: router / confirmation / registry / finance surfaces ------------
window.marvin.onEvent("confirm.request", (d) => {
  const box = $("#confirm");
  box.textContent = d.question || "";
  box.className = "asking";
});
window.marvin.onEvent("confirm.result", (d) => {
  const box = $("#confirm");
  box.textContent = `${d.name}: ${d.outcome}`;
  box.className = "";
});
window.marvin.onEvent("registry.updated", (d) => {
  $("#projects").textContent =
    `projects: ${d.confirmed} confirmed, ${d.pending} awaiting your yes`;
});
window.marvin.onEvent("router.command", (d) => {
  window.marvin.setStatus(
    `command: ${d.verb}${d.project ? " → " + d.project : ""}`);
});
window.marvin.onEvent("finance.brief", (d) => {
  const box = $("#finance");
  box.textContent = "";
  (d.rows || []).forEach((row) => {
    const line = document.createElement("div");
    line.className = "pos";
    line.textContent = Object.entries(row)
      .map(([k, v]) => `${k}: ${v}`).join("  ");
    box.appendChild(line);
  });
  if (d.caveat) {
    const c = document.createElement("div");
    c.className = "caveat";
    c.textContent = d.caveat;
    box.appendChild(c);
  }
});

window.marvin.onEvent("stt.utterance", () => playClip("got_it"));
window.marvin.onEvent("metrics.turn", (m) => {
  $("#metrics").textContent =
    `turns ${m.turns} · release→final p50 ${m.release_to_final_p50}ms ` +
    `p95 ${m.release_to_final_p95}ms · final→audio p50 ${m.final_to_audio_p50}ms ` +
    `p95 ${m.final_to_audio_p95}ms`;
});

// ---- M3 Part 2: fleet tiles / interrupt cards / worker transcript --------
const fleetTiles = new Map();
// Nothing can be handed to a terminal from either of these: CLOSED has no
// session left and DETACHED already has its window.
const FLEET_FINAL = new Set(["CLOSED", "DETACHED"]);
function renderFleetTile(d) {
  let tile = fleetTiles.get(d.worker);
  if (!tile) {
    tile = document.createElement("div");
    tile.className = "tile";
    ["tile-name", "tile-state", "tile-task", "tile-resume"].forEach((cls) => {
      const el = document.createElement("div");
      el.className = cls;
      tile.appendChild(el);
    });
    const handoff = document.createElement("button");
    handoff.className = "tile-open";
    handoff.textContent = "Open in Terminal";
    handoff.addEventListener("click", () => fetch("/handoff", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: tile.dataset.path }),
    }).then((r) => {
      // A refused request RESOLVES — the .catch below only ever sees a network
      // failure — so without this a 401 from an expired session (or a 400)
      // produces no feedback at all and the click looks ignored.
      if (!r.ok) window.marvin.setStatus(
        `handoff refused — HTTP ${r.status}; reload the console and retry`);
    }).catch(() => window.marvin.setStatus(
      "handoff failed to send — try again")));
    tile.appendChild(handoff);
    $("#fleet").appendChild(tile);
    fleetTiles.set(d.worker, tile);
  }
  tile.dataset.path = d.path || "";
  tile.querySelector(".tile-name").textContent = d.project || "?";
  tile.querySelector(".tile-state").textContent = d.state || "?";
  tile.querySelector(".tile-task").textContent = d.task || "";
  // Tiles are worker-keyed and never removed, but the button posts a PATH —
  // and the server prefers the non-final worker on that path. So a stale
  // CLOSED tile sitting beside a live worker on the same repo would hand off
  // the LIVE one. Hide the button wherever it has nothing of its own to give.
  // `interrupted` is the restart ghost's own marker (only recover() sets it):
  // its session died with the old server, so there is nothing to resume and
  // the button would be a promise the server can only refuse. UNKNOWN alone
  // is NOT that signal — a live worker that failed a health probe still holds
  // a session id and is genuinely handoff-able.
  tile.querySelector(".tile-open").hidden =
    FLEET_FINAL.has(d.state) || d.interrupted === true;
  // A restart ghost that was DETACHED before the restart carries the command
  // that rejoins its session — the id lives nowhere but the fleet log, so this
  // is the only copy Keke will ever see. Set ONLY when present: a later
  // fleet.update for the same worker must not blank a command the fleet.handoff
  // handler (or an earlier /fleet fetch) already parked here. textContent,
  // never innerHTML: this string is a server-built shell command.
  if (d.command) tile.querySelector(".tile-resume").textContent = d.command;
  return tile;
}
window.marvin.onEvent("fleet.update", renderFleetTile);
window.marvin.onEvent("fleet.message", (d) => {
  window.marvin.setStatus(`${d.project}: working…`);
});
window.marvin.onEvent("fleet.error", (d) => {
  window.marvin.setStatus("fleet: " + (d.reason || "error"));
});
// A session Marvin does not own POSTed a hook. Console only — never spoken:
// it is not a worker of ours dying, and a misconfigured worktree could emit
// these in a stream.
window.marvin.onEvent("fleet.unknown_session", (d) => {
  window.marvin.setStatus(
    `unowned session: ${d.event || "hook"} from ${d.cwd || "?"}`);
});
window.marvin.onEvent("fleet.transcript", (d) => {
  // #worker-transcript, NOT #transcript: that id is the live STT pane, and a
  // duplicate id would route these lines there (querySelector's first match).
  // Everything here is WORKER-SUPPLIED text (and, for M4 disk reads, the
  // on-disk transcript a worker wrote): textContent always, never innerHTML.
  // An empty pane must never read as all-clear — when there are no lines the
  // server sends `note` saying why, and that is what gets shown.
  const lines = (d.lines || []).map((l) => `${l.who}: ${l.text}`).join("\n\n");
  $("#worker-transcript").textContent =
    lines || `— ${d.note || "no transcript, and no reason given"} —`;
});
window.marvin.onEvent("fleet.handoff", (d) => {
  window.marvin.setStatus(`handed off: ${d.command}`);
  // #status is wiped back to "online — hold to talk" by the tts.done handler,
  // which fires the moment Marvin finishes saying "run the command on screen"
  // — and when osascript failed, that line is the ONLY copy of the command
  // Keke has. Park it on the worker's own tile, which nothing clears. The
  // tile already exists (fleet.update for DETACHED is published first, on the
  // same bus); build one anyway if a reconnect lost it.
  const tile = fleetTiles.get(d.worker) || renderFleetTile(
    { worker: d.worker, project: d.project, path: d.path, state: "DETACHED" });
  // textContent, never innerHTML: this string is a server-built shell command.
  tile.querySelector(".tile-resume").textContent = d.command || "";
});
// Every string below is WORKER-SUPPLIED — a tool name and its arguments, chosen
// by a model in a disposable checkout. textContent everywhere, never innerHTML.
function renderApproval(d) {
  // Idempotent by nonce: /fleet replays pending cards on every SSE (re)connect
  // and the live event may arrive too, but one request is one card.
  if (document.querySelector(`#interrupts .interrupt[data-nonce="${
      CSS.escape(d.nonce || "")}"]`)) return;
  const card = document.createElement("div");
  card.className = "interrupt";
  // The card says what the SENTENCE says. `outside` first so `risky` wins the
  // border when a command is both.
  if (d.outside) card.classList.add("outside");
  if (d.risk) card.classList.add("risky");
  card.dataset.nonce = d.nonce;
  const q = document.createElement("div");
  q.className = "interrupt-head";
  q.textContent = `${d.project}: ${d.tool}`;
  // The two warnings used to exist ONLY in the spoken half, so a click
  // approved something the console had never warned about.
  const note = document.createElement("div");
  note.className = "interrupt-note";
  note.textContent = [d.risk, d.outside].filter(Boolean).join(" ");
  // The full, UNELIDED argument. Speech cuts the middle out of a long shell
  // line — the elided middle of a real 347-character command was an `rm -rf`
  // on the vault — so this is the only surface in Marvin that shows the exact
  // thing being approved. Falls back to the spoken form for older payloads.
  const full = document.createElement("pre");
  full.className = "interrupt-args";
  full.textContent = d.full_args || d.args || "";
  const where = document.createElement("div");
  where.className = "interrupt-where";
  where.textContent = d.worktree ? `worktree: ${d.worktree}` : "";
  const send = (decision) => fetch("/approval", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nonce: d.nonce, decision }),
  }).then(() => card.remove())
    // A network failure correctly KEEPS the card (the nonce is still live and
    // the click can be retried) — but without a catch it is also an unhandled
    // rejection. Say what happened instead.
    .catch(() => window.marvin.setStatus(
      `approval ${decision} failed to send: ${d.project} — try again`));
  const yes = document.createElement("button");
  yes.textContent = "Approve";
  yes.addEventListener("click", () => send("approve"));
  const no = document.createElement("button");
  no.textContent = "Deny";
  no.addEventListener("click", () => send("deny"));
  card.append(q, note, full, where, yes, no);
  $("#interrupts").appendChild(card);
  return card;
}
window.marvin.onEvent("approval.request", renderApproval);
window.marvin.onEvent("approval.resolved", (d) => {
  // EVERY outcome removes the card — approved, denied, expired, cancelled,
  // and anything a later task adds. An unknown outcome must never leave a
  // stale card a click could still try to redeem.
  document.querySelectorAll(`#interrupts .interrupt[data-nonce="${d.nonce}"]`)
    .forEach((el) => el.remove());
  window.marvin.setStatus(`approval ${d.outcome}: ${d.project}`);
});

// ---- M5: the social cards column -----------------------------------------
// EVERYTHING in a card is attacker-authored text that merely survived the
// server's schema whitelist. textContent for every field, never innerHTML.
// The one hyperlink is a server-CONSTRUCTED https://x.com/... status URL —
// and it is re-checked here anyway before it becomes an href, so even a
// server regression cannot put a javascript: or off-allowlist link on the
// page. The digest is NOT rendered from here; it went directly to TTS.
window.marvin.onEvent("social.results", (d) => {
  const box = $("#social");
  if (!box) return;
  box.textContent = "";               // full repaint: one search, one column
  (d.cards || []).forEach((c) => {
    const card = document.createElement("div");
    card.className = "social-card";
    const head = document.createElement("div");
    head.className = "social-head";
    head.textContent = `@${c.handle} — ${c.author} · ${c.timestamp}`;
    const body = document.createElement("div");
    body.className = "social-text";
    body.textContent = c.text;
    card.append(head, body);
    if (/^https:\/\/x\.com\//.test(c.link || "")) {
      const link = document.createElement("a");
      link.href = c.link;
      link.textContent = "open on X";
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      card.appendChild(link);
    }
    box.appendChild(card);
  });
  // Spec §13: the meter shows what we actually know — counts, never an
  // invented dollar figure.
  const meter = document.createElement("div");
  meter.className = "social-meter";
  meter.textContent =
    `${d.count || 0} shown · ${d.refused || 0} refused · ` +
    `${(d.meter && d.meter.searches) || 0} search` +
    `${((d.meter && d.meter.searches) || 0) === 1 ? "" : "es"} this session` +
    ` · ${d.backend || "?"}`;
  box.appendChild(meter);
});
window.marvin.onEvent("social.error", (d) => {
  const box = $("#social");
  if (!box) return;
  // A failed search must never look like an empty result set.
  box.textContent = `search failed: ${d.reason || "unknown"}`;
});

// ---- what the fleet leaves behind ---------------------------------------
// Every task ever run leaves a disposable worktree and a marvin/* branch, on
// purpose — the worktree holds a diff a human may still want to merge back —
// and until this pane there was no way to see the pile. Read-only: removal is
// a SPOKEN instruction, never a click, so there is no button here to
// accidentally press. The order below is the order of danger: what is still
// live first, then what would be lost, then what is safe to clear.
const WORKTREE_ORDER = ["live", "holds-work", "empty", "stale-registration",
                        "orphan-branch", "unrecognized"];
const WORKTREE_WORDS = {
  "live": "in use — a worker or a terminal owns this",
  "holds-work": "holds work",
  "empty": "did nothing worth keeping",
  "stale-registration": "registration only — the directory is gone",
  // A branch git declined to delete. It has no directory and no registration,
  // so this row is the ONLY place it can still be seen.
  "orphan-branch": "branch only — no worktree left, and I am keeping it",
  "unrecognized": "not mine — I will not touch it",
};

function worktreeHolds(w) {
  const bits = [];
  if (w.ahead) bits.push(`${w.ahead} commit${w.ahead === 1 ? "" : "s"}`);
  if (w.dirty) bits.push(`${w.dirty} modified`);
  if (w.untracked) bits.push(`${w.untracked} untracked`);
  return bits.join(" · ");
}

function renderWorktrees(d) {
  const box = $("#worktrees");
  if (!box) return;
  box.textContent = "";               // full repaint: the survey is the truth
  const list = (d.worktrees || []).slice().sort(
    (a, b) => WORKTREE_ORDER.indexOf(a.kind) - WORKTREE_ORDER.indexOf(b.kind));
  if (d.error) {
    const err = document.createElement("div");
    err.className = "wt-error";
    // An empty list and a failed survey must never look alike.
    err.textContent = d.error;
    box.appendChild(err);
  }
  if (!list.length) return;
  const head = document.createElement("div");
  head.className = "wt-head";
  head.textContent = `worktrees: ${list.length}`;
  box.appendChild(head);
  list.forEach((w) => {
    const row = document.createElement("div");
    row.className = "wt";
    row.dataset.kind = w.kind || "";
    // EVERY string below is derived from a branch name, a directory name or a
    // worker's own project label — all of them chosen inside a disposable
    // checkout. textContent everywhere, never innerHTML.
    const name = document.createElement("div");
    name.className = "wt-name";
    // `alias` first: when two rows would share a name the survey gives each a
    // spoken one, and the screen has to show the same string Keke heard. A
    // branch-only row has no path at all, hence the branch fallback.
    name.textContent = `${w.alias || w.project
                          || (w.path || "").split("/").pop()
                          || w.branch || "?"}`;
    const kind = document.createElement("div");
    kind.className = "wt-kind";
    kind.textContent = `${w.kind || "?"} — ${WORKTREE_WORDS[w.kind] || ""}`;
    const holds = document.createElement("div");
    holds.className = "wt-holds";
    holds.textContent = worktreeHolds(w);
    const meta = document.createElement("div");
    meta.className = "wt-meta";
    const days = Math.floor((w.age_s || 0) / 86400);
    meta.textContent = [w.branch, (w.base_commit || "").slice(0, 7),
                        days ? `${days}d old` : "", w.note]
      .filter(Boolean).join(" · ");
    const where = document.createElement("div");
    where.className = "wt-where";
    where.textContent = w.path || "";
    row.append(name, kind, holds, meta, where);
    box.appendChild(row);
  });
}

// The voice verb publishes the same payload it spoke, so the screen and the
// sentence describe one survey. Idempotent — a full repaint, so the SSE event
// and a concurrent fetch cannot double-paint.
window.marvin.onEvent("worktrees.survey", renderWorktrees);

function refreshWorktrees() {
  fetch("/worktrees").then((r) => r.json()).then(renderWorktrees).catch(() => {});
}
