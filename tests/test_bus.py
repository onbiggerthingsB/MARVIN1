import asyncio
from server.bus import EventBus


async def test_publish_assigns_increasing_seq_and_delivers():
    bus = EventBus()
    cid, q = bus.subscribe()
    s1 = bus.publish("stt.final", {"text": "a"})
    s2 = bus.publish("stt.final", {"text": "b"})
    assert s2 == s1 + 1
    e1 = await asyncio.wait_for(q.get(), 1)
    e2 = await asyncio.wait_for(q.get(), 1)
    assert e1["seq"] == s1 and e1["data"]["text"] == "a"
    assert e2["seq"] == s2
    bus.unsubscribe(cid)


async def test_replay_from_last_seq():
    bus = EventBus()
    seqs = [bus.publish("t", {"i": i}) for i in range(5)]
    cid, q = bus.subscribe(last_seq=seqs[1])  # has seen 0,1 → replay 2,3,4
    got = [await asyncio.wait_for(q.get(), 1) for _ in range(3)]
    assert [e["data"]["i"] for e in got] == [2, 3, 4]
    bus.unsubscribe(cid)


async def test_slow_client_gets_closed_not_blocked():
    bus = EventBus(queue_size=4, ring_size=16)
    cid, q = bus.subscribe()
    for i in range(10):
        bus.publish("t", {"i": i})
    # drain: must find a None sentinel (client told to reconnect), publisher never blocked
    items = []
    while True:
        item = await asyncio.wait_for(q.get(), 1)
        if item is None:
            break
        items.append(item)
    assert len(items) <= 4
    bus.unsubscribe(cid)
