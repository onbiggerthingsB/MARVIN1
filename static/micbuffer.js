// Pure frame-routing logic for the /mic WebSocket: buffer what the worklet
// produces while the socket is still CONNECTING, flush it — in order, ahead
// of any live frame — the moment the socket opens, and discard everything
// once the socket is past OPEN.
//
// Why this exists: the AudioWorklet starts posting ~60ms PCM frames the
// moment the mic is wired, but the WebSocket takes a handshake to open. The
// old code attached the port handler only inside ws.onopen and so leaned on
// the browser's implicit MessagePort queue to hold the early frames —
// unbounded, undocumented at the call site, and with no way to discard an
// abandoned press's audio. This makes the same window explicit, bounded,
// ordered, and testable from node (tests/js/micbuffer.test.mjs).
//
// The cap: a 60ms Int16 frame at 16 kHz is ~1920 bytes. MAX_PENDING_FRAMES
// = 167 ≈ 10 seconds ≈ 320 KB — orders of magnitude past any plausible
// localhost handshake, small enough that a socket that never opens cannot
// grow without limit. When the cap is hit the OLDEST frame is dropped (and
// counted): by then the hold is effectively dead — onclose/onerror will tear
// it down — and the newest audio is the only part still worth relaying.

export const MAX_PENDING_FRAMES = 167; // ~10s of 60ms frames ≈ 320KB

// WebSocket readyState values, inlined so the module needs no DOM to test.
const CONNECTING = 0;
const OPEN = 1;

export function makeFrameSink(maxFrames = MAX_PENDING_FRAMES) {
  let pending = [];
  let dropped = 0;

  // Send everything buffered, in arrival order, then forget it.
  function flush(send) {
    const out = pending;
    pending = [];
    for (const f of out) send(f);
    return out.length;
  }

  return {
    // Route one worklet frame given the socket's current readyState.
    // OPEN flushes any backlog FIRST, so a buffered frame can never land
    // behind a live one even if a browser delivers a port message between
    // readyState flipping to OPEN and the open event handler running.
    frame(readyState, data, send) {
      if (readyState === OPEN) {
        flush(send);
        send(data);
        return "sent";
      }
      if (readyState === CONNECTING) {
        pending.push(data);
        if (pending.length > maxFrames) {
          pending.shift();
          dropped += 1;
          return "dropped-oldest";
        }
        return "buffered";
      }
      return "discarded"; // CLOSING / CLOSED: the session is already over
    },
    flush,
    pendingCount: () => pending.length,
    droppedCount: () => dropped,
  };
}
