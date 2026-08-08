const $ = (sel) => document.querySelector(sel);
const handlers = new Map();
let lastSeq = 0;
let audioCtx = null;
const chimes = {};

function connectSSE() {
  const es = new EventSource("/events");
  const dispatch = (type) => (e) => {
    lastSeq = Number(e.lastEventId || lastSeq);
    (handlers.get(type) || []).forEach((h) => h(JSON.parse(e.data)));
  };
  ["wake", "command.received", "stt.interim", "stt.final", "stt.utterance",
   "tts.start", "tts.done", "metrics.turn"].forEach((t) =>
    es.addEventListener(t, dispatch(t)));
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
  $("#transcript").innerHTML = `<span class="interim">${d.text}</span>`;
});
window.jarvis.onEvent("stt.utterance", (d) => {
  $("#transcript").textContent = d.text;
  window.jarvis.playChime("ack");
});

// ---- /audio WebSocket → MediaSource playback ---------------------------
function connectAudio() {
  const ws = new WebSocket(`ws://${location.host}/audio`);
  ws.binaryType = "arraybuffer";
  let ms = null, sb = null, el = null, queue = [];

  const reset = () => {
    ms = new MediaSource();
    el = new Audio();
    el.src = URL.createObjectURL(ms);
    ms.addEventListener("sourceopen", () => {
      sb = ms.addSourceBuffer("audio/mpeg");
      sb.addEventListener("updateend", () => {
        if (queue.length && !sb.updating) sb.appendBuffer(queue.shift());
      });
    });
    el.play().catch(() => {});   // AudioContext already unlocked by setup click
  };

  window.jarvis.onEvent("tts.start", () => { queue = []; reset(); });
  ws.onmessage = (e) => {
    if (!sb) return;
    if (sb.updating || queue.length) queue.push(e.data);
    else sb.appendBuffer(e.data);
  };
  ws.onclose = () => setTimeout(connectAudio, 1000);
  setInterval(() => ws.readyState === 1 && ws.send("ping"), 10000);
}
connectAudio();
window.jarvis.onEvent("tts.done", () => window.jarvis.setStatus("online — hold to talk"));
