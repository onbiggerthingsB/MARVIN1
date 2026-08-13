import asyncio
import json
import websockets
from server import stt
from server.stt import SttRelay


def fake_deepgram(responses):
    """Start a local WS server that records what it receives and replays canned
    Deepgram messages after the client's stop signal. Returns (server, url, received)."""
    received = []

    async def handler(ws):
        async for msg in ws:
            received.append(msg)
            if isinstance(msg, str) and json.loads(msg).get("type") == "CloseStream":
                for r in responses:
                    await ws.send(json.dumps(r))
                await ws.close()
                return

    async def start():
        server = await websockets.serve(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        return server, f"ws://127.0.0.1:{port}", received

    return start


DG_FINAL_A = {"type": "Results", "is_final": True, "speech_final": False,
              "channel": {"alternatives": [{"transcript": "start work in"}]}}
DG_FINAL_B = {"type": "Results", "is_final": True, "speech_final": True,
              "channel": {"alternatives": [{"transcript": "composed"}]}}


async def run_relay(url, frames):
    events = []

    async def inbound():
        for f in frames:
            yield f

    relay = SttRelay(api_key="test", keyterms=["Composed"], base_url=url)
    await relay.run(inbound(), lambda t, d: events.append((t, d)))
    return events


async def test_finals_concatenate_into_one_utterance_per_press():
    # DG_FINAL_B carries speech_final=True and it changes nothing: the press
    # is the boundary, so both finals land in the same single utterance.
    server, url, received = await fake_deepgram([DG_FINAL_A, DG_FINAL_B])()
    frames = [("text", json.dumps({"type": "start", "sample_rate": 16000, "t_hold": 1})),
              ("bytes", b"\x00\x00" * 960),
              ("text", json.dumps({"type": "stop", "t_release": 1000.0}))]
    events = await run_relay(url, frames)
    server.close(); await server.wait_closed()
    utterances = [d for t, d in events if t == "stt.utterance"]
    assert utterances == [{"text": "start work in composed", "t_release": 1000.0,
                           "t_utterance": utterances[0]["t_utterance"]}]
    # audio bytes were forwarded, and CloseStream was sent on stop
    assert any(isinstance(m, bytes) for m in received)
    assert any(isinstance(m, str) and json.loads(m).get("type") == "CloseStream" for m in received)


def test_deepgram_endpointing_is_disabled():
    """The design decision behind the mid-sentence interruptions: this is a
    press-and-hold interface, so the RELEASE is the end of the utterance — an
    explicit human signal — and Deepgram's silence heuristic has no authority
    over it. `endpointing` must be explicitly `false`: OMITTING the parameter
    does not mean off, it means Deepgram's default (10ms), which is even more
    trigger-happy than the 300ms that fragmented the live demo."""
    url = SttRelay(api_key="k", keyterms=[])._url(16000)
    assert "endpointing=false" in url
    assert "endpointing=300" not in url


async def test_speech_final_mid_hold_does_not_end_the_utterance():
    """The live defect: a thinking pause while the button was still HELD made
    Deepgram declare `speech_final`, and the relay dispatched half a sentence
    ("start work in soccer, fix the login redirect" became fragments). While
    the press is open the owner has not finished: nothing may publish
    stt.utterance until the client's stop frame, and the eventual utterance
    carries BOTH halves as one text. stt.interim keeps flowing meanwhile —
    that is the feedback that makes holding feel responsive."""
    part_a_final = asyncio.Event()
    events = []

    def publish(t, d):
        events.append((t, d))
        if t == "stt.final":
            part_a_final.set()

    async def handler(ws):
        async for msg in ws:
            if isinstance(msg, bytes):
                # The mid-hold pause: an interim paints, then endpointing
                # fires speech_final while the button is still down.
                await ws.send(json.dumps({
                    "type": "Results", "is_final": False, "speech_final": False,
                    "channel": {"alternatives": [{"transcript": "start work"}]}}))
                await ws.send(json.dumps({
                    "type": "Results", "is_final": True, "speech_final": True,
                    "channel": {"alternatives":
                                [{"transcript": "start work in soccer"}]}}))
            elif json.loads(msg).get("type") == "CloseStream":
                await ws.send(json.dumps({
                    "type": "Results", "is_final": True, "speech_final": True,
                    "channel": {"alternatives":
                                [{"transcript": "fix the login redirect"}]}}))
                await ws.close()
                return

    server = await websockets.serve(handler, "127.0.0.1", 0)
    url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"

    async def inbound():
        yield ("text", json.dumps({"type": "start", "sample_rate": 16000,
                                   "t_hold": 1}))
        yield ("bytes", b"\x00\x00" * 960)
        # The press is still open. Wait until the relay has PROCESSED the
        # mid-hold speech_final — if it dispatched on it, the stray utterance
        # is already in `events` and the assertion below catches it.
        await asyncio.wait_for(part_a_final.wait(), 5)
        yield ("text", json.dumps({"type": "stop", "t_release": 1000.0}))

    relay = SttRelay(api_key="test", keyterms=[], base_url=url)
    await relay.run(inbound(), publish)
    server.close(); await server.wait_closed()

    utterances = [d for t, d in events if t == "stt.utterance"]
    assert utterances == [{"text": "start work in soccer fix the login redirect",
                           "t_release": 1000.0,
                           "t_utterance": utterances[0]["t_utterance"]}], (
        "a mid-hold speech_final dispatched (or split) the utterance: "
        f"{utterances}")
    assert ("stt.interim", {"text": "start work"}) in events, (
        "the live transcript stopped painting while the button was held")


async def test_release_drain_is_bounded_when_deepgram_never_closes(monkeypatch):
    """After the release the wind-down is machine work and must be bounded.
    A Deepgram that flushes a final but never closes the socket must not hold
    the /mic handler (or the utterance) hostage: the drain times out and
    whatever was finalized is still published as the press's one utterance."""
    monkeypatch.setattr(stt, "DRAIN_TIMEOUT_S", 0.3)
    hang = asyncio.Event()

    async def handler(ws):
        try:
            async for msg in ws:
                if isinstance(msg, str) and \
                        json.loads(msg).get("type") == "CloseStream":
                    await ws.send(json.dumps(DG_FINAL_A))
                    await hang.wait()          # flushes, then never closes
        except websockets.ConnectionClosed:
            return

    server = await websockets.serve(handler, "127.0.0.1", 0)
    url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    frames = [("text", json.dumps({"type": "start", "sample_rate": 16000,
                                   "t_hold": 1})),
              ("bytes", b"\x00\x00" * 960),
              ("text", json.dumps({"type": "stop", "t_release": 7.0}))]
    events = await asyncio.wait_for(run_relay(url, frames), 5)
    hang.set()
    server.close(); await server.wait_closed()
    utterances = [d for t, d in events if t == "stt.utterance"]
    assert [u["text"] for u in utterances] == ["start work in"], events
    assert utterances[0]["t_release"] == 7.0


async def test_deepgram_mid_stream_disconnect_is_survived():
    async def handler(ws):
        await ws.recv()          # one frame, then die mid-stream
        await ws.close(code=1011)

    server = await websockets.serve(handler, "127.0.0.1", 0)
    url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    frames = [("text", json.dumps({"type": "start", "t_hold": 1})),
              ("bytes", b"\x00\x00" * 960),
              ("bytes", b"\x00\x00" * 960),
              ("text", json.dumps({"type": "stop", "t_release": 2.0}))]
    events = await run_relay(url, frames)   # must not raise
    server.close(); await server.wait_closed()
    assert ("stt.error", {"reason": "deepgram disconnected"}) in events
