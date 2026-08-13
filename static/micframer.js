// Pure framing logic for the mic AudioWorklet: accumulate context-rate
// Float32 chunks until ~60ms is held, then emit ONE downsampled 16k Int16
// frame — and, at end of utterance, flush whatever partial is still held.
//
// Why this exists: the worklet only posts a frame once a full 60ms has
// accumulated, so at the moment of release it is always holding 0–60ms of
// un-emitted audio — the end of the last word. Before flush() existed that
// tail was simply discarded, which is very likely how "Start" reached the
// router as "Star" in the live run: the router anchors on the opening verb,
// so a clipped word at either end silently turns a command into a butler
// conversation. The worklet (static/worklet.js) is a thin shell over this
// module so the behaviour is pinned from node (tests/js/micframer.test.mjs),
// where an AudioWorkletGlobalScope does not exist.

import { downsampleTo16k } from "./resample.js";

export function makeFramer({ sampleRate, frameMs = 60,
                             downsample = downsampleTo16k }) {
  let buf = [];
  let samples = 0;
  const target = Math.round(sampleRate * (frameMs / 1000));

  function emitJoined(emit) {
    const joined = new Float32Array(samples);
    let o = 0;
    for (const b of buf) { joined.set(b, o); o += b.length; }
    buf = [];
    samples = 0;
    const int16 = downsample(joined, sampleRate);
    // A sliver shorter than one output sample downsamples to nothing; a
    // zero-byte frame on the wire says nothing and is not worth a send.
    if (int16.length === 0) return false;
    emit(int16);
    return true;
  }

  return {
    // One chunk from process(). Copied — the render engine reuses the render
    // quantum's buffer. Emits exactly when a full frame is held.
    push(ch, emit) {
      buf.push(new Float32Array(ch));
      samples += ch.length;
      if (samples < target) return false;
      return emitJoined(emit);
    },
    // End of utterance: emit the partial frame still held, if any. This is
    // the worklet half of the GAP-1 flush handshake.
    flush(emit) {
      if (samples === 0) return false;
      return emitJoined(emit);
    },
    heldSamples: () => samples,
  };
}

// The worklet side of the flush handshake, extracted so node can run the
// EXACT code the worklet runs (worklet.js delegates here verbatim). Order is
// the contract: the tail frame (if any) is posted BEFORE the ack, and
// MessagePort delivery is FIFO, so by the time the main thread sees
// `flushed`, every frame — in-flight and tail alike — is already ahead of it.
// `post(message, transfer)` is the port's postMessage.
export function handleFlushRequest(framer, data, post) {
  if (!data || data.type !== "flush") return false;
  framer.flush((int16) => post(int16.buffer, [int16.buffer]));
  post({ type: "flushed" });
  return true;
}
