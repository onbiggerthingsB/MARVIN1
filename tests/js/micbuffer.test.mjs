import test from "node:test";
import assert from "node:assert/strict";
import { makeFrameSink, MAX_PENDING_FRAMES } from "../../static/micbuffer.js";

// WebSocket readyState values, as app.js sees them.
const CONNECTING = 0, OPEN = 1, CLOSING = 2, CLOSED = 3;

// The defect from the live demo: audio produced before the socket opens must
// reach the server. `sent` here is the wire — everything passed to send(), in
// the order it would leave the browser.
test("frames produced while CONNECTING are flushed, in order, once open", () => {
  const sink = makeFrameSink();
  const sent = [];
  const send = (f) => sent.push(f);
  const early = ["start-", "work-", "in-soccer"];        // spoken pre-open
  for (const f of early) {
    assert.equal(sink.frame(CONNECTING, f, send), "buffered");
  }
  assert.equal(sent.length, 0, "nothing may go out before the socket opens");
  assert.equal(sink.pendingCount(), 3);

  sink.flush(send);                                      // ws.onopen
  assert.equal(sink.frame(OPEN, "fix-the-login", send), "sent"); // live frame
  assert.deepEqual(sent, ["start-", "work-", "in-soccer", "fix-the-login"],
    "pre-open audio reaches the wire ahead of every live frame, in order");
});

test("flush drains: a second flush sends nothing again", () => {
  const sink = makeFrameSink();
  const sent = [];
  sink.frame(CONNECTING, "a", (f) => sent.push(f));
  assert.equal(sink.flush((f) => sent.push(f)), 1);
  assert.equal(sink.flush((f) => sent.push(f)), 0);
  assert.deepEqual(sent, ["a"]);
  assert.equal(sink.pendingCount(), 0);
});

test("a live frame flushes any backlog first (belt for early delivery)", () => {
  // If a browser ever delivers a port message after readyState flips to OPEN
  // but before the open event handler has flushed, order must still hold.
  const sink = makeFrameSink();
  const sent = [];
  const send = (f) => sent.push(f);
  sink.frame(CONNECTING, "early", send);
  sink.frame(OPEN, "live", send);                        // no flush() ran yet
  assert.deepEqual(sent, ["early", "live"]);
});

test("the buffer is bounded: oldest drops first, newest is kept", () => {
  const cap = 5;
  const sink = makeFrameSink(cap);
  const results = [];
  for (let i = 0; i < cap + 3; i++) {
    results.push(sink.frame(CONNECTING, `f${i}`, () => {
      throw new Error("must not send while CONNECTING");
    }));
  }
  assert.deepEqual(results.slice(0, cap), Array(cap).fill("buffered"));
  assert.deepEqual(results.slice(cap), Array(3).fill("dropped-oldest"));
  assert.equal(sink.pendingCount(), cap);
  assert.equal(sink.droppedCount(), 3);
  const sent = [];
  sink.flush((f) => sent.push(f));
  assert.deepEqual(sent, ["f3", "f4", "f5", "f6", "f7"],
    "the newest audio survives, still in order");
});

test("default cap is ~10s of 60ms frames", () => {
  assert.equal(MAX_PENDING_FRAMES, 167);
  const sink = makeFrameSink();
  for (let i = 0; i < 200; i++) sink.frame(CONNECTING, i, () => {});
  assert.equal(sink.pendingCount(), 167);
  assert.equal(sink.droppedCount(), 33);
});

test("CLOSING/CLOSED discard: nothing buffered, nothing sent", () => {
  const sink = makeFrameSink();
  const send = () => { throw new Error("must not send on a dying socket"); };
  assert.equal(sink.frame(CLOSING, "late", send), "discarded");
  assert.equal(sink.frame(CLOSED, "later", send), "discarded");
  assert.equal(sink.pendingCount(), 0);
});

test("an abandoned press's buffer is never sent unless flushed", () => {
  // app.js gives each press its own sink; a press abandoned before open
  // simply never calls flush. The buffered frames must sit inert — the next
  // press's sink is a different object and cannot see them.
  const abandoned = makeFrameSink();
  abandoned.frame(CONNECTING, "stale-audio", () => {
    throw new Error("abandoned audio must not be sent");
  });
  const next = makeFrameSink();
  const sent = [];
  next.flush((f) => sent.push(f));
  assert.deepEqual(sent, [], "a fresh sink starts empty");
  assert.equal(abandoned.pendingCount(), 1, "stale audio stays inert until GC");
});
