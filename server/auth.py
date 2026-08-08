"""Auth primitives. Pure functions + tiny state holders; no framework imports."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field

BOOTSTRAP_TTL_S = 60


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class BootstrapState:
    token_hash: str
    created: float
    used: bool = False


def new_bootstrap() -> tuple[str, BootstrapState]:
    token = secrets.token_urlsafe(32)
    return token, BootstrapState(token_hash=_sha(token), created=time.time())


def redeem_bootstrap(state: BootstrapState, token: str, now: float) -> bool:
    if state.used or (now - state.created) > BOOTSTRAP_TTL_S:
        return False
    if not hmac.compare_digest(state.token_hash, _sha(token)):
        return False
    state.used = True
    return True


def issue_session() -> tuple[str, str]:
    cookie = secrets.token_urlsafe(32)
    return cookie, _sha(cookie)


def verify_session(cookie: str | None, stored_hash: str) -> bool:
    if not cookie or not stored_hash:
        return False
    return hmac.compare_digest(_sha(cookie), stored_hash)


def verify_bearer(header: str | None, expected: str) -> bool:
    if not header or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header.removeprefix("Bearer "), expected)


def origin_ok(origin: str | None, host: str | None, port: int) -> bool:
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if host not in allowed_hosts:
        return False
    if origin is None:
        return True
    return origin in {f"http://{h}" for h in allowed_hosts}
