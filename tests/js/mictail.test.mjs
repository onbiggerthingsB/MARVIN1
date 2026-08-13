// GAP 1 — the tail of an utterance must reach the wire.
//
// The worklet only posts a frame once it has ~60ms, so at release it is still
// holding a partial. That partial plus one in-flight frame used to be dropped,
// clipping the END of every utterance — which is how "Start" reached the
// deterministic router as "Star", and the router anchors commands on exactly
// that opening verb.
//
// The fix is a handshake, and its correctness rests on ONE ordering property:
// the worklet posts its tail frame BEFORE the `flushed` ack, and MessagePort
// delivery is FIFO — so when the main thread sees the ack, every frame is
// already behind it. These tests pin that property against the real
// handleFlushRequest the worklet runs (worklet.js delegates to it verbatim).
import { test } from "node:test";
import assert from "node:assert/strict";
import { makeFramer, handleFlushRequest } from "../../static/micframer.js";

const RATE = 48000;

// Feed n samples of context-rate audio, collecting whatever gets emitted.
function feed(framer, n, out) {
  framer.push(new Float32Array(n).fill(0.5), (int16) => out.push(int16));
}

test("the held partial reaches the wire on flush", () => {
  const framer = makeFramer({ sampleRate: RATE });
  const out = [];
  feed(framer, 1200, out);                       // 25ms — under one 60ms frame
  assert.equal(out.length, 0, "nothing emitted yet: still accumulating");

  const posted = [];
  handleFlushRequest(framer, { type: "flush" }, (m) => posted.push(m));
  const frames = posted.filter((m) => !(m && m.type));
  assert.equal(frames.length, 1, "the partial was emitted as a tail frame");
  assert.ok(frames[0].byteLength > 0, "and it carries real audio");
});

test("the tail frame is posted BEFORE the ack — the whole contract", () => {
  const framer = makeFramer({ sampleRate: RATE });
  feed(framer, 1200, []);

  const order = [];
  handleFlushRequest(framer, { type: "flush" },
    (m) => order.push(m && m.type === "flushed" ? "ack" : "frame"));

  assert.deepEqual(order, ["frame", "ack"],
    "ack last: FIFO delivery then means seeing the ack proves the tail landed");
});

test("flush with nothing held still acks, so release never hangs", () => {
  const framer = makeFramer({ sampleRate: RATE });
  const order = [];
  handleFlushRequest(framer, { type: "flush" },
    (m) => order.push(m && m.type === "flushed" ? "ack" : "frame"));
  assert.deepEqual(order, ["ack"], "no audio to send, but the ack still comes");
});

test("a full frame plus a partial: both reach the wire, in capture order", () => {
  const framer = makeFramer({ sampleRate: RATE });
  const live = [];
  feed(framer, 2880, live);                      // exactly 60ms -> one frame
  assert.equal(live.length, 1, "the full frame went out live");
  feed(framer, 1200, live);                      // 25ms more, held back

  const posted = [];
  handleFlushRequest(framer, { type: "flush" }, (m) => posted.push(m));
  const tail = posted.filter((m) => !(m && m.type));
  assert.equal(tail.length, 1, "the remaining partial follows as the tail");
});

test("non-flush messages are ignored, so ordinary traffic cannot fake an ack", () => {
  const framer = makeFramer({ sampleRate: RATE });
  feed(framer, 1200, []);
  const posted = [];
  for (const msg of [null, undefined, {}, { type: "stop" }, "flush", 42]) {
    assert.equal(handleFlushRequest(framer, msg, (m) => posted.push(m)), false);
  }
  assert.equal(posted.length, 0, "nothing emitted, and the partial is still held");
});
