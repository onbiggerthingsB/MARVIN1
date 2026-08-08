"""Per-turn latency records with p50/p95 over a sliding window."""
from __future__ import annotations

from collections import deque


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[idx]


class TurnLog:
    def __init__(self, window: int = 100):
        self._turns: deque[dict] = deque(maxlen=window)
        self._open: dict | None = None

    def record_utterance(self, t_release: float | None, t_utterance: float) -> None:
        if t_release is None:
            self._open = None
            return
        self._open = {"release_to_final": (t_utterance - t_release) * 1000,
                      "t_utterance": t_utterance}

    def record_first_audio(self, t_first_audio: float | None) -> None:
        if self._open is None or t_first_audio is None:
            self._open = None
            return
        delta = (t_first_audio - self._open["t_utterance"]) * 1000
        if delta < 0:
            # Mis-paired turn from a burst: audio "before" the utterance would
            # record a bogus negative latency. Drop the turn instead.
            self._open = None
            return
        self._open["final_to_audio"] = delta
        self._turns.append(self._open)
        self._open = None

    def summary(self) -> dict:
        rf = [t["release_to_final"] for t in self._turns]
        fa = [t["final_to_audio"] for t in self._turns if "final_to_audio" in t]
        return {"turns": len(self._turns),
                "release_to_final_p50": round(_pct(rf, 50)),
                "release_to_final_p95": round(_pct(rf, 95)),
                "final_to_audio_p50": round(_pct(fa, 50)),
                "final_to_audio_p95": round(_pct(fa, 95))}
