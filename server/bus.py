"""In-process event bus. Monotonic seq, ring-buffer replay, bounded subscriber queues."""
from __future__ import annotations

import asyncio
from collections import deque
from itertools import count


class EventBus:
    def __init__(self, queue_size: int = 256, ring_size: int = 512):
        self._seq = count(1)
        self._ring: deque[dict] = deque(maxlen=ring_size)
        self._subs: dict[int, asyncio.Queue] = {}
        self._ids = count(1)
        self._queue_size = queue_size

    def publish(self, type_: str, data: dict) -> int:
        event = {"seq": next(self._seq), "type": type_, "data": data}
        self._ring.append(event)
        for cid, q in list(self._subs.items()):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow client: drop it. It reconnects and replays via last_seq.
                while not q.empty():
                    q.get_nowait()
                q.put_nowait(None)
                del self._subs[cid]
        return event["seq"]

    def subscribe(self, last_seq: int | None = None) -> tuple[int, asyncio.Queue]:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        if last_seq is not None:
            for event in self._ring:
                if event["seq"] > last_seq and not q.full():
                    q.put_nowait(event)
        cid = next(self._ids)
        self._subs[cid] = q
        return cid, q

    def unsubscribe(self, client_id: int) -> None:
        self._subs.pop(client_id, None)
