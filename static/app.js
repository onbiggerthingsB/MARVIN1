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
