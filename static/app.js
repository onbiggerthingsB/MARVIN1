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
   "fleet.handoff", "approval.request",
   "approval.resolved"].forEach((t) =>
    es.addEventListener(t, dispatch(t)));
  // Deliberate: reconnect fresh (no Last-Event-ID). Replaying stale tts.start/
  // stt.utterance on a live-voice UI would double-trigger playback. Server-side
  // replay (ring buffer + bus.gap) stays ready for a future non-voice consumer.
  es.onerror = () => { es.close(); setTimeout(connectSSE, 1000); };
}

async function loadChime(name) {
  const buf = await (await fetch(`/static/chimes/${name}.wav`)).arrayBuffer();
  chimes[name] = await audioCtx.decodeAudioData(buf);
}

window.jarvis = {
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
    window.jarvis.playChime("done");
    $("#setup-overlay").hidden = true;
    $("#console").hidden = false;
    connectSSE();
    // Page load only. SSE carries every LATER tile, but it reconnects without
    // a Last-Event-ID, so anything published before this browser existed —
    // above all the restart ghosts, which are published once at boot — is
    // only ever visible through this fetch. Failure is silent by design: a
    // missing initial render must not block the console coming online.
    fetch("/fleet").then((r) => r.json())
      .then((d) => (d.workers || []).forEach(renderFleetTile))
      .catch(() => {});
    window.jarvis.setStatus("online — hold to talk");
  } catch (err) {
    $("#setup-status").textContent = `setup failed: ${err.message} — fix and click again`;
  }
});

window.jarvis.onEvent("wake", () => {
  window.jarvis.playChime("listen");
  window.jarvis.setStatus("yes?");
});

// ---- press-and-hold mic → /mic WebSocket -------------------------------
let micWS = null, micStream = null, micNode = null, analyser = null;
let micAborting = false;

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
  micAborting = false;
  try {
    window.jarvis.playChime("listen");
    $("#ptt").classList.add("live");
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (micAborting) { teardownMic(); return; } // released during setup
    const ctx = window.jarvis.audioCtx();
    await ctx.audioWorklet.addModule("/static/worklet.js");
    if (micAborting) { teardownMic(); return; } // released during setup
    const src = ctx.createMediaStreamSource(micStream);
    analyser = ctx.createAnalyser();
    src.connect(analyser);
    micNode = new AudioWorkletNode(ctx, "mic-processor");
    src.connect(micNode);
    micWS = new WebSocket(`ws://${location.host}/mic`);
    micWS.binaryType = "arraybuffer";
    micWS.onopen = () => {
      micWS.send(JSON.stringify({ type: "start", encoding: "linear16",
        sample_rate: 16000, channels: 1, t_hold: Date.now() }));
      micNode.port.onmessage = (e) => {
        if (micWS && micWS.readyState === 1) micWS.send(e.data);
      };
    };
    drawWave();
  } catch (err) {
    teardownMic();
    window.jarvis.setStatus("mic error — " + err.message);
    return;
  }
}

function stopTalking() {
  if (!micWS) {
    // Setup still in flight (or already torn down): signal the abort AND stop
    // whatever the in-flight setup may have already opened, so the mic can't
    // be left hot.
    micAborting = true;
    teardownMic();
    return;
  }
  if (micWS.readyState === 1)
    micWS.send(JSON.stringify({ type: "stop", t_release: Date.now() }));
  const ws = micWS; micWS = null; // null first so teardownMic won't close it
  teardownMic();                  // stops tracks, disconnects node, clears .live
  setTimeout(() => ws.close(), 3000); // give server time to flush finals
  window.jarvis.setStatus("thinking…");
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
ptt.addEventListener("pointerdown", startTalking);
ptt.addEventListener("pointerup", stopTalking);
ptt.addEventListener("pointerleave", stopTalking);
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && !$("#setup-overlay").hidden) return;
  if (e.code === "Space" && !e.repeat) { e.preventDefault(); startTalking(); }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space") { e.preventDefault(); stopTalking(); }
});

window.jarvis.onEvent("stt.interim", (d) => {
  const t = $("#transcript");
  t.classList.add("interim");
  t.textContent = d.text;
});
window.jarvis.onEvent("stt.utterance", (d) => {
  $("#transcript").classList.remove("interim");
  $("#transcript").textContent = d.text;
  window.jarvis.playChime("ack");
});
window.jarvis.onEvent("stt.error", (d) => {
  window.jarvis.setStatus("couldn't hear you — " + (d.reason || "audio error"));
  if (typeof playClip === "function") playClip("cannot_hear");
});

window.jarvis.onEvent("butler.answer", (d) => {
  $("#answer").textContent = d.display || "";
  const box = $("#citations");
  box.textContent = "";
  (d.citations || []).forEach((name) => {
    const chip = document.createElement("span");
    chip.className = "cite";
    chip.textContent = name;
    box.appendChild(chip);
  });
  window.jarvis.setStatus("online — hold to talk");
});
window.jarvis.onEvent("butler.error", (d) => {
  // Clear the previous turn's answer AND its citation chips. Leaving them up
  // under an error line reads as if the stale answer belongs to the question
  // that just failed.
  $("#answer").textContent = "";
  $("#citations").textContent = "";
  window.jarvis.setStatus("brain error — " + (d.reason || "unavailable"));
});
// Metrics failures are NOT brain failures: a TurnLog hiccup on tts.done must
// never blank a correct answer that JARVIS is still speaking. Status line only —
// no clearing of #answer / #citations.
window.jarvis.onEvent("metrics.error", (d) => {
  window.jarvis.setStatus("metrics: " + (d.reason || "unavailable"));
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
    window.jarvis.onEvent("tts.start", () => {
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
window.jarvis.onEvent("tts.done", () => window.jarvis.setStatus("online — hold to talk"));

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
window.jarvis.onEvent("confirm.request", (d) => {
  const box = $("#confirm");
  box.textContent = d.question || "";
  box.className = "asking";
});
window.jarvis.onEvent("confirm.result", (d) => {
  const box = $("#confirm");
  box.textContent = `${d.name}: ${d.outcome}`;
  box.className = "";
});
window.jarvis.onEvent("registry.updated", (d) => {
  $("#projects").textContent =
    `projects: ${d.confirmed} confirmed, ${d.pending} awaiting your yes`;
});
window.jarvis.onEvent("router.command", (d) => {
  window.jarvis.setStatus(
    `command: ${d.verb}${d.project ? " → " + d.project : ""}`);
});
window.jarvis.onEvent("finance.brief", (d) => {
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

window.jarvis.onEvent("stt.utterance", () => playClip("got_it"));
window.jarvis.onEvent("metrics.turn", (m) => {
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
      if (!r.ok) window.jarvis.setStatus(
        `handoff refused — HTTP ${r.status}; reload the console and retry`);
    }).catch(() => window.jarvis.setStatus(
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
  return tile;
}
window.jarvis.onEvent("fleet.update", renderFleetTile);
window.jarvis.onEvent("fleet.message", (d) => {
  window.jarvis.setStatus(`${d.project}: working…`);
});
window.jarvis.onEvent("fleet.error", (d) => {
  window.jarvis.setStatus("fleet: " + (d.reason || "error"));
});
window.jarvis.onEvent("fleet.transcript", (d) => {
  // #worker-transcript, NOT #transcript: that id is the live STT pane, and a
  // duplicate id would route these lines there (querySelector's first match).
  $("#worker-transcript").textContent = (d.lines || [])
    .map((l) => `${l.who}: ${l.text}`).join("\n\n");
});
window.jarvis.onEvent("fleet.handoff", (d) => {
  window.jarvis.setStatus(`handed off: ${d.command}`);
  // #status is wiped back to "online — hold to talk" by the tts.done handler,
  // which fires the moment JARVIS finishes saying "run the command on screen"
  // — and when osascript failed, that line is the ONLY copy of the command
  // Keke has. Park it on the worker's own tile, which nothing clears. The
  // tile already exists (fleet.update for DETACHED is published first, on the
  // same bus); build one anyway if a reconnect lost it.
  const tile = fleetTiles.get(d.worker) || renderFleetTile(
    { worker: d.worker, project: d.project, path: d.path, state: "DETACHED" });
  // textContent, never innerHTML: this string is a server-built shell command.
  tile.querySelector(".tile-resume").textContent = d.command || "";
});
window.jarvis.onEvent("approval.request", (d) => {
  const card = document.createElement("div");
  card.className = "interrupt";
  card.dataset.nonce = d.nonce;
  const q = document.createElement("div");
  q.textContent = `${d.project}: ${d.tool} — ${d.args}`;
  const send = (decision) => fetch("/approval", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nonce: d.nonce, decision }),
  }).then(() => card.remove())
    // A network failure correctly KEEPS the card (the nonce is still live and
    // the click can be retried) — but without a catch it is also an unhandled
    // rejection. Say what happened instead.
    .catch(() => window.jarvis.setStatus(
      `approval ${decision} failed to send: ${d.project} — try again`));
  const yes = document.createElement("button");
  yes.textContent = "Approve";
  yes.addEventListener("click", () => send("approve"));
  const no = document.createElement("button");
  no.textContent = "Deny";
  no.addEventListener("click", () => send("deny"));
  card.append(q, yes, no);
  $("#interrupts").appendChild(card);
});
window.jarvis.onEvent("approval.resolved", (d) => {
  // EVERY outcome removes the card — approved, denied, expired, cancelled,
  // and anything a later task adds. An unknown outcome must never leave a
  // stale card a click could still try to redeem.
  document.querySelectorAll(`#interrupts .interrupt[data-nonce="${d.nonce}"]`)
    .forEach((el) => el.remove());
  window.jarvis.setStatus(`approval ${d.outcome}: ${d.project}`);
});
