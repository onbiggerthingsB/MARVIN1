// Pure-function linear-interpolation downsampler to 16 kHz Int16.
export function downsampleTo16k(float32, inRate) {
  const outRate = 16000;
  const ratio = inRate / outRate;
  const outLen = Math.floor(float32.length / ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, float32.length - 1);
    const sample = float32[i0] + (float32[i1] - float32[i0]) * (pos - i0);
    const clamped = Math.max(-1, Math.min(1, sample));
    out[i] = Math.round(clamped < 0 ? clamped * 32768 : clamped * 32767);
  }
  return out;
}
