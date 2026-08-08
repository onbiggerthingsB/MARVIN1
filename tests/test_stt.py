import asyncio
import json
import websockets
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


async def test_finals_concatenate_until_speech_final():
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
