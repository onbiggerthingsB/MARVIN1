import test from "node:test";
import assert from "node:assert/strict";
import { downsampleTo16k } from "../../static/resample.js";

test("48k → 16k is 3:1 and preserves amplitude sign", () => {
  const input = new Float32Array(4800);           // 100ms at 48k
  for (let i = 0; i < input.length; i++) input[i] = Math.sin(i / 10) * 0.5;
  const out = downsampleTo16k(input, 48000);
  assert.equal(out.length, 1600);                  // 100ms at 16k
  assert.ok(out instanceof Int16Array);
  assert.ok(Math.max(...out) > 4000 && Math.min(...out) < -4000);
});

test("16k input passes through same length", () => {
  const input = new Float32Array(1600).fill(0.25);
  const out = downsampleTo16k(input, 16000);
  assert.equal(out.length, 1600);
  assert.ok(Math.abs(out[0] - 0.25 * 32767) < 2);
});

test("clipping is clamped", () => {
  const input = new Float32Array([1.5, -1.5]);
  const out = downsampleTo16k(input, 16000);
  assert.equal(out[0], 32767);
  assert.equal(out[1], -32768);
});
