"""JARVIS control plane. Single-process FastAPI app; all state on app.state."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from server import auth
from server.app_brain import run_butler_brain
from server.bus import EventBus
from server.butler import Butler, build_options
from server.config import ensure_config, load_keyterms, save_config
from server.stt import SttRelay
from server.vault_mcp import build_vault_server
from server.vault_paths import vault_root_from_env
from server.vault_read import vault_is_downloaded

COOKIE = "jarvis_session"
OPEN_PATHS = {"/health", "/bootstrap"}
BEARER_PATHS = {"/wake"}


def _existing_note_titles(titles, vault_root: Path) -> set:
    """Which of `titles` name a real `<title>.md` note somewhere in the vault.

    A FILENAME match is deliberate and sufficient: citations are Obsidian
    wikilinks, which address notes by basename, and the cheap check is the whole
    point -- this runs on every answered turn. Blocking `rglob`, so callers must
    hand it to a thread.

    Matching is on `Path.stem`, never on a glob pattern built from model output:
    a title containing `[`, `*` or `?` would otherwise be interpreted as glob
    syntax rather than matched literally.
    """
    wanted = {str(t).strip() for t in titles if str(t).strip()}
    if not wanted:
        return set()
    found: set = set()
    for p in Path(vault_root).rglob("*.md"):
        if p.stem in wanted:
            found.add(p.stem)
            if len(found) == len(wanted):
                break
    return found


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
    from server.metrics import TurnLog
    app.state.turnlog = TurnLog()
    app.state.bootstrap = bootstrap_state
    app.state.bootstrap_token_plain = token_plain  # read by launcher tests only
    app.state.base_dir = base_dir

    from server.tts import SpeakEngine, ELEVEN_BASE

    audio_clients: set = set()

    def send_audio(chunk: bytes) -> None:
        for ws_ in list(audio_clients):
            asyncio.ensure_future(_safe_send(ws_, chunk))

    async def _safe_send(ws_, chunk):
        try:
            await ws_.send_bytes(chunk)
        except Exception:
            audio_clients.discard(ws_)

    voice = os.environ.get("JARVIS_VOICE",
                            "elevenlabs" if os.environ.get("ELEVENLABS_API_KEY") else "say")
    app.state.speaker = SpeakEngine(
        voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "") if voice == "elevenlabs" else "",
        api_key=os.environ.get("ELEVENLABS_API_KEY", "") if voice == "elevenlabs" else "",
        base_url=os.environ.get("ELEVENLABS_URL", ELEVEN_BASE),
        publish=app.state.bus.publish,
        send_audio=send_audio,
    )

    vault_root = vault_root_from_env()
    if not vault_is_downloaded(vault_root):
        # Not fatal: the butler still runs, but grounding may stall on iCloud.
        # Surfaced once at startup; the setup screen guidance covers the fix.
        print(f"[jarvis] WARNING: vault not fully downloaded at {vault_root} "
              f"— enable 'Keep Downloaded' in Finder for reliable answers.")
    vault_server = build_vault_server(vault_root)
    app.state.butler = Butler(
        options_builder=lambda resume: build_options(vault_root, vault_server, resume),
        state_path=base_dir / "state" / "butler.json")

    @app.middleware("http")
    async def guard(request: Request, call_next):
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

    @app.get("/metrics")
    async def metrics():
        return app.state.turnlog.summary()

    @app.get("/bootstrap")
    async def bootstrap(token: str = ""):
        if not auth.redeem_bootstrap(app.state.bootstrap, token, now=time.time()):
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

    @app.websocket("/audio")
    async def audio(ws: WebSocket):
        # Cookie + origin checks before accept, matching /mic (middleware skips WS).
        if not auth.origin_ok(ws.headers.get("origin"), ws.headers.get("host"), cfg.port) \
           or not auth.verify_session(ws.cookies.get(COOKIE), cfg.session_token_hash):
            await ws.close(code=4401)
            return
        await ws.accept()
        audio_clients.add(ws)
        try:
            while True:
                await ws.receive_text()  # keepalive pings from page
        except WebSocketDisconnect:
            audio_clients.discard(ws)

    static = base_dir / "static"
    if static.exists():
        app.mount("/static", StaticFiles(directory=static), name="static")

    async def validate_citations(titles):
        """Drop cited titles that do not resolve to a real note (spec §4).

        to_thread: the vault is on iCloud and the walk is blocking file I/O that
        must not sit on the loop carrying live audio.
        """
        found = await asyncio.to_thread(_existing_note_titles, titles, vault_root)
        return [t for t in titles if str(t).strip() in found]

    @contextlib.asynccontextmanager
    async def _lifespan(_app):
        task = asyncio.create_task(
            run_butler_brain(app.state.bus, app.state.butler,
                             app.state.speaker, app.state.turnlog,
                             validate_citations=validate_citations))

        def _brain_died(t):
            # Last resort. run_butler_brain guards every await, so reaching here
            # means something outside those guards ended it -- and a dead brain
            # is otherwise perfectly silent: /health stays ok and JARVIS simply
            # never answers again. Cancellation is the normal shutdown path.
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                app.state.bus.publish("butler.error",
                                      {"reason": f"brain task died: {exc!r}"})

        task.add_done_callback(_brain_died)
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await app.state.butler.close()

    # Assigned after the app is built (rather than passed to FastAPI(lifespan=…))
    # so create_app's shape is unchanged. This is the spec §2 "lifespan-managed
    # cancellation" the M1 review asked for, and it clears the @app.on_event
    # deprecation warning the echo brain carried.
    app.router.lifespan_context = _lifespan
    return app


def app_factory():
    """Zero-arg entry point for `uvicorn server.app:app_factory --factory`.
    Uses ~/jarvis as base_dir so the server is launchable without arguments."""
    from pathlib import Path
    return create_app(base_dir=Path(__file__).resolve().parent.parent)
