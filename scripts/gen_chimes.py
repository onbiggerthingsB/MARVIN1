"""Generate the three JARVIS chimes as WAV files. Stdlib only, deterministic."""
import math
import struct
import wave
from pathlib import Path

RATE = 22050
OUT = Path(__file__).resolve().parent.parent / "static" / "chimes"


def tone(freqs: list[float], ms: int, volume: float = 0.35) -> bytes:
    n = int(RATE * ms / 1000)
    frames = bytearray()
    for i in range(n):
        fade = min(1.0, (n - i) / (0.3 * n))  # release envelope
        s = sum(math.sin(2 * math.pi * f * i / RATE) for f in freqs) / len(freqs)
        frames += struct.pack("<h", int(32767 * volume * fade * s))
    return bytes(frames)


def write(name: str, pcm: bytes) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT / name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


if __name__ == "__main__":
    write("listen.wav", tone([880.0], 120))
    write("ack.wav", tone([660.0], 90))
    write("done.wav", tone([523.25, 659.25], 200))
    print("chimes written to", OUT)
