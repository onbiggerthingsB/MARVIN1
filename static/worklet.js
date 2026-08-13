import { makeFramer, handleFlushRequest } from "./micframer.js";

// Thin shell over micframer.js: accumulate ~60ms of context-rate audio, emit
// 16k Int16 frames to the main thread — and, on a `flush` request from the
// main thread, emit whatever partial is still held, then ack with `flushed`.
// The framing and the handshake both live in micframer.js so node can pin
// them (tests/js/micframer.test.mjs, tests/js/mictail.test.mjs); this file
// only owns what needs the AudioWorkletGlobalScope.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.framer = makeFramer({ sampleRate });
    this.emit = (int16) => this.port.postMessage(int16.buffer, [int16.buffer]);
    // GAP-1 handshake, worklet side. Port messages are handled on the render
    // thread between quanta, so this works even if no more input arrives.
    this.port.onmessage = (e) =>
      handleFlushRequest(this.framer, e.data,
        (m, transfer) => this.port.postMessage(m, transfer));
  }
  process(inputs) {
    const ch = inputs[0][0];
    if (ch) this.framer.push(ch, this.emit);
    return true;
  }
}
registerProcessor("mic-processor", MicProcessor);
