import { downsampleTo16k } from "/static/resample.js";

// Accumulates ~60ms of context-rate audio, emits 16k Int16 frames to the main thread.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = [];
    this.samples = 0;
    this.target = Math.round(sampleRate * 0.06); // 60ms at context rate
  }
  process(inputs) {
    const ch = inputs[0][0];
    if (ch) {
      this.buf.push(new Float32Array(ch));
      this.samples += ch.length;
      if (this.samples >= this.target) {
        const joined = new Float32Array(this.samples);
        let o = 0;
        for (const b of this.buf) { joined.set(b, o); o += b.length; }
        const int16 = downsampleTo16k(joined, sampleRate);
        this.port.postMessage(int16.buffer, [int16.buffer]);
        this.buf = []; this.samples = 0;
      }
    }
    return true;
  }
}
registerProcessor("mic-processor", MicProcessor);
