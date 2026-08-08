import time
from server import auth


def test_bootstrap_single_use_and_expiry():
    token, state = auth.new_bootstrap()
    assert auth.redeem_bootstrap(state, "wrong-token", now=time.time()) is False
    assert auth.redeem_bootstrap(state, token, now=time.time()) is True
    # single use
    assert auth.redeem_bootstrap(state, token, now=time.time()) is False
    token2, state2 = auth.new_bootstrap()
    # expired
    assert auth.redeem_bootstrap(state2, token2, now=time.time() + 61) is False


def test_session_roundtrip():
    cookie, stored = auth.issue_session()
    assert auth.verify_session(cookie, stored) is True
    assert auth.verify_session("tampered", stored) is False
    assert auth.verify_session(None, stored) is False
    assert auth.verify_session(cookie, "") is False


def test_bearer():
    assert auth.verify_bearer("Bearer abc", "abc") is True
    assert auth.verify_bearer("Bearer nope", "abc") is False
    assert auth.verify_bearer(None, "abc") is False


def test_origin_and_host():
    ok = auth.origin_ok
    assert ok(None, "127.0.0.1:7777", 7777) is True            # curl, no Origin
    assert ok("http://127.0.0.1:7777", "127.0.0.1:7777", 7777) is True
    assert ok("http://localhost:7777", "localhost:7777", 7777) is True
    assert ok("https://evil.example", "127.0.0.1:7777", 7777) is False
    assert ok("http://127.0.0.1:7777", "evil.example", 7777) is False  # bad Host
    assert ok(None, None, 7777) is False
