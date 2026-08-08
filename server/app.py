"""JARVIS control plane. Single-process FastAPI app; all state on app.state."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from server import auth
from server.bus import EventBus
from server.config import ensure_config, load_keyterms, save_config
from server.stt import SttRelay

COOKIE = "jarvis_session"
OPEN_PATHS = {"/health", "/bootstrap"}
BEARER_PATHS = {"/wake"}


def create_app(base_dir: Path) -> FastAPI:
    app = FastAPI()
    cfg = ensure_config(base_dir / "config" / "jarvis.json")
    token_plain, bootstrap_state = auth.new_bootstrap()
    state_dir = base_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    url_file = state_dir / "bootstrap_url"
    url_file.write_text(f"http://127.0.0.1:{cfg.port}/bootstrap?token={token_plain}\n")
    url_file.chmod(0o600)
    bearer_file = state_dir / "hook_bearer"
    bearer_file.write_text(cfg.hook_bearer + "\n")
    bearer_file.chmod(0o600)

    app.state.cfg = cfg
    app.state.bus = EventBus()
    app.state.bootstrap = bootstrap_state
    app.state.bootstrap_token_plain = token_plain  # read by launcher tests only
    app.state.base_dir = base_dir

    @app.middleware("http")
    async def guard(request: Request, call_next):
        import time as _t
        path = request.url.path
        origin = request.headers.get("origin")
        host = request.headers.get("host")
        if not auth.origin_ok(origin, host, cfg.port):
            return JSONResponse({"error": "forbidden origin"}, status_code=403)
        if path in OPEN_PATHS:
            return await call_next(request)
        if path in BEARER_PATHS:
            if not auth.verify_bearer(request.headers.get("authorization"), cfg.hook_bearer):
                return JSONResponse({"error": "bearer required"}, status_code=401)
            return await call_next(request)
        if not auth.verify_session(request.cookies.get(COOKIE), cfg.session_token_hash):
            return JSONResponse({"error": "not bootstrapped"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/bootstrap")
    async def bootstrap(token: str = ""):
        import time as _t
        if not auth.redeem_bootstrap(app.state.bootstrap, token, now=_t.time()):
            return JSONResponse({"error": "invalid bootstrap token"}, status_code=403)
        cookie_value, stored_hash = auth.issue_session()
        cfg.session_token_hash = stored_hash
        save_config(cfg, base_dir / "config" / "jarvis.json")
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(COOKIE, cookie_value, httponly=True, samesite="strict")
        return resp

    @app.get("/")
    async def index():
        html = (base_dir / "static" / "index.html")
        if html.exists():
            return Response(html.read_text(), media_type="text/html")
        return Response("JARVIS: static console not built yet", media_type="text/plain")

    @app.post("/wake")
    async def wake():
        app.state.bus.publish("wake", {})
        return {"ok": True}

    @app.post("/command")
    async def command(request: Request):
        body = await request.json()
        text = str(body.get("text", "")).strip()
        if not text:
            return JSONResponse({"error": "empty"}, status_code=400)
        app.state.bus.publish("command.received", {"text": text})
        return {"ok": True}

    @app.get("/events")
    async def events(request: Request):
        last = request.headers.get("last-event-id")
        try:
            last_seq = int(last) if last else None
        except ValueError:
            last_seq = None
        cid, q = app.state.bus.subscribe(last_seq)

        async def stream():
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
                        continue
                    if event is None:
                        return  # evicted; client reconnects with Last-Event-ID
                    yield f"id: {event['seq']}\nevent: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
            finally:
                app.state.bus.unsubscribe(cid)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.websocket("/mic")
    async def mic(ws: WebSocket):
        # Cookie + origin checks: middleware does not cover WS, so verify here.
        if not auth.origin_ok(ws.headers.get("origin"), ws.headers.get("host"), cfg.port) \
           or not auth.verify_session(ws.cookies.get(COOKIE), cfg.session_token_hash):
            await ws.close(code=4401)
            return
        await ws.accept()
        key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not key:
            app.state.bus.publish("stt.error", {"reason": "no DEEPGRAM_API_KEY"})
            await ws.close(code=4500)
            return

        async def inbound():
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("bytes") is not None:
                        yield ("bytes", msg["bytes"])
                    elif msg.get("text") is not None:
                        yield ("text", msg["text"])
                        if json.loads(msg["text"]).get("type") == "stop":
                            return
                    else:
                        return
            except WebSocketDisconnect:
                return

        relay = SttRelay(api_key=key,
                         keyterms=load_keyterms(base_dir / "config" / "keyterms.json"),
                         base_url=os.environ.get("DEEPGRAM_URL", "wss://api.deepgram.com"))
        await relay.run(inbound(), app.state.bus.publish)
        await ws.close()

    static = base_dir / "static"
    if static.exists():
        app.mount("/static", StaticFiles(directory=static), name="static")
    return app


def app_factory():
    """Zero-arg entry point for `uvicorn server.app:app_factory --factory`.
    Uses ~/jarvis as base_dir so the server is launchable without arguments."""
    from pathlib import Path
    return create_app(base_dir=Path(__file__).resolve().parent.parent)
