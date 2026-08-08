import socket
import threading
import time as time_mod
from contextlib import contextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.app import create_app
from server.config import load_config


def make_client(tmp_path: Path) -> TestClient:
    app = create_app(base_dir=tmp_path)
    return TestClient(app, base_url="http://127.0.0.1:7777")


def bootstrap(client: TestClient) -> None:
    token = client.app.state.bootstrap_token_plain  # test hook, set in create_app
    r = client.get(f"/bootstrap?token={token}", follow_redirects=False)
    assert r.status_code == 303
    assert "jarvis_session" in r.cookies


def test_health_is_open(tmp_path):
    assert make_client(tmp_path).get("/health").json()["ok"] is True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _live_server(app: FastAPI, port: int):
    """Serve `app` on a real loopback socket in a background thread.

    DEVIATION from the brief (recorded in the task report): the installed
    test stack's in-process ASGI transports (starlette.testclient's
    `_TestClientTransport` and httpx's `ASGITransport`, as pinned in this
    repo) fully drain the ASGI callable before handing back a response —
    `handle_request`/`handle_async_request` both `portal.call`/`await` the
    whole `app(scope, receive, send)` to completion, even for `.stream()`.
    `/events` never voluntarily returns (it loops until the subscriber is
    evicted), so driving it through those transports deadlocks forever:
    confirmed by reproducing the hang under a hard `signal.alarm` guard.
    A real socket doesn't have that problem — the client closing the
    connection is a genuine disconnect, not a value the app has to produce
    itself — so this spins up uvicorn on loopback for the one assertion
    that needs a connection to actually stay open.
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time_mod.time() + 5
    while not server.started and time_mod.time() < deadline:
        time_mod.sleep(0.01)
    assert server.started, "live test server failed to start"
    try:
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_events_requires_cookie(tmp_path):
    app = create_app(base_dir=tmp_path)
    port = _free_port()
    app.state.cfg.port = port  # keep Host-header checks consistent with the bound port

    with _live_server(app, port):
        with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
            assert client.get("/events").status_code == 401
            token = app.state.bootstrap_token_plain
            r = client.get(f"/bootstrap?token={token}", follow_redirects=False)
            assert r.status_code == 303
            assert "jarvis_session" in r.cookies
            with client.stream("GET", "/events") as r:
                assert r.status_code == 200


def test_bootstrap_is_single_use_and_persists_hash(tmp_path):
    c = make_client(tmp_path)
    token = c.app.state.bootstrap_token_plain
    assert c.get(f"/bootstrap?token={token}", follow_redirects=False).status_code == 303
    assert c.get(f"/bootstrap?token={token}", follow_redirects=False).status_code == 403
    cfg = load_config(tmp_path / "config" / "jarvis.json")
    assert cfg.session_token_hash != ""


def test_wake_requires_hook_bearer(tmp_path):
    c = make_client(tmp_path)
    assert c.post("/wake").status_code == 401
    bearer = c.app.state.cfg.hook_bearer
    r = c.post("/wake", headers={"Authorization": f"Bearer {bearer}"})
    assert r.status_code == 200


def test_cross_origin_rejected(tmp_path):
    c = make_client(tmp_path)
    bootstrap(c)
    r = c.post("/command", json={"text": "hi"},
               headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_command_requires_cookie(tmp_path):
    c = make_client(tmp_path)
    assert c.post("/command", json={"text": "hi"}).status_code == 401
