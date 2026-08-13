import test from "node:test";
import assert from "node:assert/strict";
import { makeFramer, handleFlushRequest } from "../../static/micframer.js";

// At 16 kHz the linear-interpolation downsampler is an identity pass
// (ratio 1), so every assertion below reads sample-for-sample: what the
// framer was fed is exactly what must come out, just quantised to Int16.
const RATE = 16000;
const TARGET = Math.round(RATE * 0.06); // 960 samples per 60ms frame

// Int16 quantisation the resampler applies to an in-range sample.
const toI16 = (v) => Math.round(v < 0 ? v * 32768 : v * 32767);
// Distinct, small, in-range values so frames are tellable apart.
const chunk = (n, from) =>
  Float32Array.from({ length: n }, (_, i) => ((from + i) % 200) / 1000);

test("nothing emits before a full 60ms frame is held", () => {
  const f = makeFramer({ sampleRate: RATE });
  const emitted = [];
  assert.equal(f.push(chunk(TARGET - 1, 0), (x) => emitted.push(x)), false);
  assert.equal(emitted.length, 0);
  assert.equal(f.heldSamples(), TARGET - 1);
});

test("a full frame emits once, joined in arrival order", () => {
  const f = makeFramer({ sampleRate: RATE });
  const emitted = [];
  f.push(chunk(500, 0), (x) => emitted.push(x));
  f.push(chunk(TARGET - 500, 500), (x) => emitted.push(x));
  assert.equal(emitted.length, 1);
  assert.equal(emitted[0].length, TARGET);
  const expected = Array.from(chunk(TARGET, 0), toI16);
  assert.deepEqual(Array.from(emitted[0]), expected,
    "the frame is the two chunks, joined, in order");
  assert.equal(f.heldSamples(), 0, "emitting empties the accumulator");
});

// GAP 1, the worklet half: the partial the worklet is holding at release —
// the end of the last word — must come out on flush, sample for sample.
test("flush emits the held partial: the end of the last word survives", () => {
  const f = makeFramer({ sampleRate: RATE });
  const emitted = [];
  const tail = chunk(700, 0); // ~44ms — the "t" of "Start"
  f.push(tail, (x) => emitted.push(x));
  assert.equal(emitted.length, 0, "700 < 960: still held, not yet emitted");
  assert.equal(f.flush((x) => emitted.push(x)), true);
  assert.equal(emitted.length, 1);
  assert.deepEqual(Array.from(emitted[0]), Array.from(tail, toI16),
    "the flushed frame IS the held audio");
  assert.equal(f.heldSamples(), 0);
});

test("flush with nothing held emits nothing", () => {
  const f = makeFramer({ sampleRate: RATE });
  assert.equal(f.flush(() => { throw new Error("must not emit"); }), false);
});

test("audio after a flush starts a clean frame — no bleed", () => {
  const f = makeFramer({ sampleRate: RATE });
  const emitted = [];
  f.push(chunk(300, 0), (x) => emitted.push(x));
  f.flush((x) => emitted.push(x));
  f.push(chunk(TARGET, 77), (x) => emitted.push(x));
  assert.equal(emitted.length, 2);
  assert.deepEqual(Array.from(emitted[1]), Array.from(chunk(TARGET, 77), toI16),
    "the next frame holds only the new audio");
});

test("a sliver too small to survive downsampling emits nothing", () => {
  // 2 samples at 48k downsample to floor(2/3) = 0 output samples: a
  // zero-byte frame says nothing, so flush must not send one.
  const f = makeFramer({ sampleRate: 48000 });
  f.push(new Float32Array(2), () => { throw new Error("must not emit"); });
  assert.equal(f.flush(() => { throw new Error("must not emit"); }), false);
  assert.equal(f.heldSamples(), 0, "the sliver is still consumed");
});

// The handshake as the worklet actually runs it (worklet.js delegates to
// handleFlushRequest verbatim): tail frame first, ack after — port delivery
// is FIFO, so the main thread's ack always trails the audio.
test("handleFlushRequest posts the tail BEFORE the flushed ack", () => {
  const f = makeFramer({ sampleRate: RATE });
  f.push(chunk(700, 0), () => {});
  const posted = [];
  assert.equal(handleFlushRequest(f, { type: "flush" },
    (m) => posted.push(m)), true);
  assert.equal(posted.length, 2);
  assert.ok(posted[0] instanceof ArrayBuffer, "first: the tail audio");
  assert.equal(new Int16Array(posted[0]).length, 700);
  assert.deepEqual(posted[1], { type: "flushed" }, "second: the ack");
});

test("handleFlushRequest with nothing held still acks — release never hangs", () => {
  const f = makeFramer({ sampleRate: RATE });
  const posted = [];
  handleFlushRequest(f, { type: "flush" }, (m) => posted.push(m));
  assert.deepEqual(posted, [{ type: "flushed" }]);
});

test("handleFlushRequest ignores everything that is not a flush", () => {
  const f = makeFramer({ sampleRate: RATE });
  f.push(chunk(700, 0), () => {});
  const posted = [];
  assert.equal(handleFlushRequest(f, { type: "other" }, (m) => posted.push(m)), false);
  assert.equal(handleFlushRequest(f, null, (m) => posted.push(m)), false);
  assert.equal(posted.length, 0);
  assert.equal(f.heldSamples(), 700, "held audio is untouched");
});
