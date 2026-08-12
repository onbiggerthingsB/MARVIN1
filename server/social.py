"""Quarantined X/Twitter search — spec §7, the last milestone on purpose.

Indirect prompt injection through feed content is the #1 known failure mode
of this product category, so the shape of this module IS the security
argument. Three zones, and nothing crosses a boundary except the narrow
thing named here:

  FETCH    — a subprocess whose stdout is UNTRUSTED, attacker-authored bytes.
             Hard timeout (the whole process group dies), a byte cap read
             incrementally (a 50MB payload is killed, not buffered), and an
             environment CONSTRUCTED from scratch: credentials come from
             agent-reach's own config file and are passed explicitly — the
             ambient process environment is never inherited, so a clean
             shell or a future LaunchAgent behaves identically, and nothing
             this server holds can leak into the child.
             Crosses the boundary: raw bytes, nothing else.

  VALIDATE — a strict whitelist schema. Anything that does not fit is
             REFUSED, never coerced: ids must be digits, handles must match
             X's own [A-Za-z0-9_]{1,15}, timestamps must parse as tz-aware
             ISO 8601 (source timestamps required), text is bounded, and
             the ONLY URL a card ever carries is CONSTRUCTED here from the
             validated handle + id and then checked against the HTTPS +
             domain allowlist anyway (whitelist, belt and braces — this
             codebase has shipped seven fail-opens on blacklist-shaped
             gates). Attacker URLs, media, metrics and every unknown field
             simply do not exist in the output schema.
             Crosses the boundary: cards with exactly six known fields.

  PRESENT  — cards go to the console over `social.results` (rendered with
             textContent, never innerHTML) and a DETERMINISTIC digest goes
             straight to TTS. The digest is built from ZERO attacker-
             authored bytes — a count, the owner's own query, a relative
             time computed from the PARSED timestamp — because any model
             summarizing attacker text can be steered into speaking
             attacker-chosen sentences, and even quoting "inert" post text
             aloud hands an attacker the speakers (and, while the owner
             holds push-to-talk, possibly the microphone).
             Crosses the boundary to the BUTLER: one fixed sentence,
             "search completed, N results on screen", where N is an integer
             this server counted. Nothing derived from result content may
             ever reach a session that holds private data and an action
             channel.

agent-reach is a VOLATILE BUNDLE, not an API: `agent-reach doctor --json`
runs before each request (cached briefly), routing follows `active_backend`,
and only backends whose exact invocation syntax was verified at build time
are in KNOWN_BACKENDS — anything else is refused with an honest sentence,
never guessed at. Verified 2026-08-12: active_backend "twitter-cli", binary
`twitter`, invocation `twitter search <query> --json -n N`, envelope
{"ok": bool, "schema_version": "1", "data"|"error": ...}; credentials read
from ~/.agent-reach/config.yaml (twitter_auth_token / twitter_ct0) and
passed as TWITTER_AUTH_TOKEN / TWITTER_CT0 (twitter_cli/auth.py reads env
FIRST — with the env set it never falls back to ambient browser-cookie
extraction).

Every failure here is a SENTENCE, never a raise. run() cannot throw.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

DOCTOR_TTL_S = 60.0          # "cached briefly": one doctor per conversation beat
DOCTOR_TIMEOUT_S = 20.0
SEARCH_TIMEOUT_S = 45.0
MAX_STDOUT_BYTES = 2_000_000     # the 50MB payload dies at 2MB, mid-read
MAX_DOCTOR_BYTES = 1_000_000
MAX_RESULTS = 5
MAX_EXAMINED = 50            # bound the validator's work, not just its output
MAX_QUERY_CHARS = 200
MAX_TEXT_CHARS = 6_000       # schema bound: longer "posts" are refused
CARD_TEXT_CHARS = 500        # display bound: capped with a visible mark
MAX_NAME_CHARS = 80

# THE WHITELISTS. Backends verified at build time — routing follows the
# doctor's active_backend but only onto syntax we have actually verified;
# an upstream that drifted to something new is refused, not guessed at.
KNOWN_BACKENDS = frozenset({"twitter-cli"})
# The only host a card link may point at. Links are CONSTRUCTED (never taken
# from the payload), so this check is a second fence, not the first.
ALLOWED_LINK_HOSTS = frozenset({"x.com"})
# Error codes we are willing to NAME in a reason string. Everything else is
# "error": the error envelope's message text is attacker-adjacent (it can
# reflect the wire) and is never displayed, spoken, or logged into a reason.
_KNOWN_ERROR_CODES = frozenset({"not_found", "api_error", "auth_error",
                                "rate_limited", "network_error"})

SOCIAL_TOOL_NAMES = ["mcp__social__social_search"]

# X's own username charset; a handle is also a URL path segment, so this
# doubles as the guarantee the constructed link cannot be steered.
_HANDLE = re.compile(r"^[A-Za-z0-9_]{1,15}$")
_ID = re.compile(r"^[0-9]{1,25}$")

# Ambient keys the subprocess env COPIES (explicitly, by name — this is the
# whole list; nothing else crosses). The backend reaches x.com through the
# local proxy on this machine; GUI-launched and LaunchAgent processes do not
# inherit these either, which is why they are named rather than inherited.
_PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
               "http_proxy", "https_proxy", "no_proxy")

# Where a bare binary name may resolve from — fixed, never $PATH.
_BIN_DIRS = ("~/.local/bin", "/opt/homebrew/bin", "/usr/local/bin")

# Bidi and directionality controls: a post can use U+202E and friends to
# visually reverse what the console shows ("exe.gpj" reading as "jpg.exe").
_BIDI = frozenset("‪‫‬‭‮"
                  "⁦⁧⁨⁩‎‏؜")

# Spoken failure lines — a CLOSED set. Nothing off the wire is ever
# interpolated into what gets said or what the butler is told.
REFUSAL_LINES = {
    "empty query": "I need something to search for, sir.",
    "bad query": "I won't pass that query to a search tool, sir.",
    "no credentials": ("I don't have X credentials configured, sir, "
                       "so I won't search."),
    "doctor unavailable": ("I couldn't check my search tools, sir — "
                           "the doctor didn't answer."),
    "search unavailable": ("X search isn't available right now, sir — "
                           "the doctor's report is on screen."),
    "unknown backend": ("My X search path has changed upstream, sir — "
                        "I won't guess at it."),
    "no backend binary": "My X search tool is missing, sir.",
    "timed out": "The X search timed out, sir.",
    "oversized": ("The X search sent back more than I'm willing to read, "
                  "sir — I refused it."),
    "unreadable output": "The X search came back malformed, sir — I refused it.",
    "bad envelope": "The X search came back malformed, sir — I refused it.",
    "internal": "The X search failed on my side, sir.",
}
_BACKEND_REFUSED_LINE = "X refused the search, sir — details are on screen."


def butler_line(n: int) -> str:
    """THE one sentence the butler may receive about a search. A fixed
    template around one integer — deliberately not even pluralized, so the
    only degree of freedom is N itself."""
    return f"search completed, {int(n)} results on screen"


def url_ok(url: str) -> bool:
    """HTTPS + domain-allowlisted, exactly (spec §7). A whitelist: scheme
    must BE https and the hostname must BE on the list — no suffix matching,
    no scheme-relative forms, no exceptions."""
    from urllib.parse import urlsplit
    try:
        parts = urlsplit(str(url))
    except ValueError:
        return False
    return parts.scheme == "https" and parts.hostname in ALLOWED_LINK_HOSTS


def scrub(text: str, cap: int) -> str:
    """Display sanitation for card text: C0/C1 controls (except newline) and
    bidi/directionality controls become U+FFFD — visibly replaced, never
    silently dropped, so a doctored post LOOKS doctored. Then a hard cap
    with a visible ellipsis. Structural fields never come through here; they
    are whitelist-validated and refused outright instead."""
    out = "".join(
        "�" if (ch in _BIDI or (ch != "\n"
                                     and unicodedata.category(ch) in ("Cc", "Cf")))
        else ch
        for ch in str(text))
    if len(out) > cap:
        out = out[: cap - 1] + "…"
    return out


def _card(item) -> dict | None:
    """One validated card, or None — refusal, not coercion. The output
    schema is CLOSED: six fields, all derived from validated input, and the
    link is constructed here rather than ever read from the payload."""
    if not isinstance(item, dict):
        return None
    tid, author = item.get("id"), item.get("author")
    text, ts = item.get("text"), item.get("createdAtISO")
    if not isinstance(tid, str) or not _ID.fullmatch(tid):
        return None
    if not isinstance(author, dict):
        return None
    handle, name = author.get("screenName"), author.get("name")
    if not isinstance(handle, str) or not _HANDLE.fullmatch(handle):
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        return None
    if not isinstance(ts, str):
        return None                       # source timestamps are REQUIRED
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None                       # a naive timestamp is not a timestamp
    link = f"https://x.com/{handle}/status/{tid}"
    if not url_ok(link):                  # the allowlist runs on the FINAL url
        return None
    return {"id": tid, "handle": handle,
            "author": scrub(name, MAX_NAME_CHARS),
            "text": scrub(text, CARD_TEXT_CHARS),
            "timestamp": dt.isoformat(),  # canonical, re-serialized — not wire bytes
            "link": link}


def validate_payload(raw: bytes) -> dict:
    """bytes off the untrusted subprocess -> {"cards", "refused", "reason"}.

    reason == "" means the ENVELOPE was sound; individual results may still
    have been refused (counted in "refused"). A non-empty reason names a
    refusal class from a closed set — never text from the wire.
    """
    refused = {"cards": [], "refused": 0}
    try:
        data = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return {**refused, "reason": "unreadable output"}
    if not isinstance(data, dict):
        return {**refused, "reason": "bad envelope"}
    if data.get("ok") is not True:
        err = data.get("error")
        code = err.get("code") if isinstance(err, dict) else None
        code = code if code in _KNOWN_ERROR_CODES else "error"
        return {**refused, "reason": f"backend refused: {code}"}
    if data.get("schema_version") != "1":
        # Upstream drift. The bundle is volatile; a shape we have not
        # verified is refused, not parsed on hope.
        return {**refused, "reason": "bad envelope"}
    items = data.get("data")
    if not isinstance(items, list):
        return {**refused, "reason": "bad envelope"}
    cards, bad = [], 0
    for item in items[:MAX_EXAMINED]:
        card = _card(item)
        if card is None:
            bad += 1
        else:
            cards.append(card)
        if len(cards) >= MAX_RESULTS:
            break
    return {"cards": cards, "refused": bad, "reason": ""}


def _relative(newest_iso: str, now: datetime | None) -> str:
    dt = datetime.fromisoformat(newest_iso)
    now = now or datetime.now(timezone.utc)
    s = max(0.0, (now - dt).total_seconds())
    if s < 90:
        return "just now"
    if s < 5400:
        n = round(s / 60)
        return f"{n} minute{'s' if n != 1 else ''} ago"
    if s < 129_600:
        n = round(s / 3600)
        return f"{n} hour{'s' if n != 1 else ''} ago"
    n = round(s / 86_400)
    return f"{n} day{'s' if n != 1 else ''} ago"


def digest_line(query: str, count: int, refused: int,
                newest_iso: str | None, now: datetime | None = None) -> str:
    """The ≤3-sentence digest that goes DIRECTLY to TTS. Deterministic, and
    built from zero attacker-authored bytes: an integer count, the owner's
    own query (scrubbed and capped — on the tool path the butler authors
    it), and a relative time computed from the PARSED newest timestamp. No
    model anywhere near it: a model summarizing attacker text can be steered
    into speaking attacker-chosen sentences through the owner's speakers,
    and the cards are already on screen for the content itself."""
    q = scrub(str(query), 80)
    if count > 0:
        posts = f"{count} post{'s' if count != 1 else ''}"
        line = (f"Search completed, sir — {posts} about {q} on screen, "
                f"the newest from {_relative(newest_iso, now)}.")
        if refused:
            line += f" I refused {refused} more that failed validation."
        return line
    if refused:
        return (f"Search completed, sir — {refused} post"
                f"{'s' if refused != 1 else ''} came back but none passed "
                f"validation, so nothing is on screen.")
    return f"Search completed, sir — no posts found for {q}."


class _Oversized(Exception):
    pass


async def _read_capped(stream, cap: int) -> bytes:
    chunks, total = [], 0
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > cap:
            raise _Oversized()
        chunks.append(chunk)


async def _kill(proc) -> None:
    """Kill the WHOLE process group (the backend may spawn its own children
    — twitter-cli's browser-cookie fallback shells out), then reap."""
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()
    with contextlib.suppress(asyncio.TimeoutError, Exception):
        await asyncio.wait_for(proc.wait(), 5)


class SocialSearch:
    """The pipeline, with every knob injectable so tests (and the canary)
    can run REAL subprocesses against fake binaries. `speak` is the direct
    TTS callable for the butler-tool path; the brain path speaks the
    returned digest itself, in its own serial order."""

    def __init__(self, bus, speak=None, *,
                 agent_reach_bin="agent-reach", backend_bin="twitter",
                 config_path=None,
                 doctor_ttl_s: float = DOCTOR_TTL_S,
                 doctor_timeout_s: float = DOCTOR_TIMEOUT_S,
                 search_timeout_s: float = SEARCH_TIMEOUT_S,
                 max_stdout_bytes: int = MAX_STDOUT_BYTES):
        self._bus = bus
        self._speak = speak
        self._agent_reach = agent_reach_bin
        self._backend = backend_bin
        self._config_path = Path(config_path) if config_path else (
            Path.home() / ".agent-reach" / "config.yaml")
        self._doctor_ttl = doctor_ttl_s
        self._doctor_timeout = doctor_timeout_s
        self._search_timeout = search_timeout_s
        self._max_stdout = max_stdout_bytes
        self._doctor_cache: tuple[float, dict] | None = None
        self.searches = 0                     # the on-screen meter; never dollars

    # ---------- fetch zone plumbing ----------

    @staticmethod
    def _resolve_bin(spec) -> Path | None:
        """A binary resolves from an absolute path or from the FIXED list of
        directories — never from ambient $PATH, for the same reason the env
        is constructed: a clean shell must behave identically."""
        p = Path(spec)
        if p.is_absolute():
            return p if os.access(p, os.X_OK) else None
        for d in _BIN_DIRS:
            cand = Path(d).expanduser() / str(spec)
            if os.access(cand, os.X_OK):
                return cand
        return None

    def _credentials(self) -> dict[str, str]:
        """X cookie state, read EXPLICITLY from agent-reach's own config —
        the file its doctor manages — and nowhere else. A two-key whitelist
        parser on purpose: no YAML library, no other keys, and missing
        either key fails closed (twitter-cli would otherwise fall back to
        ambient browser-cookie extraction, which is exactly the inherited
        state the spec forbids)."""
        try:
            text = self._config_path.read_text(encoding="utf-8")
        except OSError:
            return {}
        creds: dict[str, str] = {}
        for line in text.splitlines():
            key, sep, value = line.partition(":")
            if not sep:
                continue
            key, value = key.strip(), value.strip().strip("'\"")
            if key == "twitter_auth_token" and value:
                creds["TWITTER_AUTH_TOKEN"] = value
            elif key == "twitter_ct0" and value:
                creds["TWITTER_CT0"] = value
        return creds if len(creds) == 2 else {}

    @staticmethod
    def _env(extra: dict[str, str], *bin_dirs: str) -> dict[str, str]:
        """The subprocess environment, CONSTRUCTED — never os.environ. HOME
        (the backend keeps its transaction cache there), a fixed PATH, the
        named proxy keys copied one by one, and whatever `extra` (the
        credentials) adds. Nothing else this process holds can cross."""
        env = {"HOME": str(Path.home()),
               "PATH": os.pathsep.join([*bin_dirs, "/usr/bin", "/bin"])}
        for key in _PROXY_KEYS:
            value = os.environ.get(key)
            if value:
                env[key] = value
        env.update(extra)
        return env

    async def _run_capped(self, argv, env, timeout, cap):
        """-> (bytes | None, reason). The subprocess in its own session so a
        timeout kills the whole process group; stdout read incrementally
        against the byte cap so an oversized payload dies mid-flight instead
        of being buffered."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *[str(a) for a in argv],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
                env=env, start_new_session=True)
        except OSError:
            return None, "no backend binary"
        try:
            out = await asyncio.wait_for(
                _read_capped(proc.stdout, cap), timeout)
        except asyncio.TimeoutError:
            await _kill(proc)
            return None, "timed out"
        except _Oversized:
            await _kill(proc)
            return None, "oversized"
        except Exception:  # noqa: BLE001 — a read fault is a fetch failure, spoken
            await _kill(proc)
            return None, "internal"
        try:
            await asyncio.wait_for(proc.wait(), 10)
        except asyncio.TimeoutError:
            await _kill(proc)
        # The exit code is deliberately NOT consulted: the backend exits 1
        # while still printing a well-formed error envelope, and the
        # envelope — validated, whitelisted — is the honest signal.
        return out, ""

    async def _doctor(self) -> dict | None:
        """agent-reach doctor --json, cached briefly. Returns the twitter
        block ({"status", "active_backend", ...}) or None."""
        now = time.monotonic()
        if self._doctor_cache and now - self._doctor_cache[0] < self._doctor_ttl:
            return self._doctor_cache[1]
        bin_path = self._resolve_bin(self._agent_reach)
        if bin_path is None:
            return None
        out, reason = await self._run_capped(
            [bin_path, "doctor", "--json"],
            self._env({}, str(bin_path.parent)),
            self._doctor_timeout, MAX_DOCTOR_BYTES)
        if reason or out is None:
            return None
        try:
            doc = json.loads(out.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        block = doc.get("twitter") if isinstance(doc, dict) else None
        if not isinstance(block, dict):
            return None
        self._doctor_cache = (now, block)
        return block

    # ---------- the pipeline ----------

    def _refuse(self, reason: str) -> dict:
        spoken = (REFUSAL_LINES.get(reason)
                  or (_BACKEND_REFUSED_LINE if reason.startswith("backend refused")
                      else REFUSAL_LINES["internal"]))
        with contextlib.suppress(Exception):
            self._bus.publish("social.error", {"reason": reason})
        return {"ok": False, "count": 0, "refused": 0,
                "spoken": spoken, "butler_line": None, "reason": reason}

    async def run(self, query: str, speak_digest: bool = False) -> dict:
        """The full quarantine: guard -> doctor -> fetch -> validate ->
        present. Never raises; every failure is a sentence."""
        try:
            return await self._run_guarded(query, speak_digest)
        except Exception:  # noqa: BLE001 — the brain must never die of a search
            return self._refuse("internal")

    async def _run_guarded(self, query: str, speak_digest: bool) -> dict:
        q = str(query or "").strip()
        if not q:
            return self._refuse("empty query")
        if q.startswith("-") or len(q) > MAX_QUERY_CHARS:
            # A leading dash is an argv flag to the backend's CLI parser —
            # the one way a query could become an instruction. Refused.
            return self._refuse("bad query")

        creds = self._credentials()
        if not creds:
            return self._refuse("no credentials")

        doctor = await self._doctor()
        if doctor is None:
            return self._refuse("doctor unavailable")
        if doctor.get("status") != "ok":
            return self._refuse("search unavailable")
        backend = doctor.get("active_backend")
        if backend not in KNOWN_BACKENDS:
            # Routing follows active_backend — but only onto invocation
            # syntax verified at build time. Whitelist, never guess.
            return self._refuse("unknown backend")

        bin_path = self._resolve_bin(self._backend)
        if bin_path is None:
            return self._refuse("no backend binary")
        out, reason = await self._run_capped(
            [bin_path, "search", q, "--json", "-n", str(MAX_RESULTS)],
            self._env(creds, str(bin_path.parent)),
            self._search_timeout, self._max_stdout)
        if reason or out is None:
            return self._refuse(reason or "internal")

        validated = validate_payload(out)
        if validated["reason"]:
            return self._refuse(validated["reason"])

        cards, refused = validated["cards"], validated["refused"]
        newest = max((c["timestamp"] for c in cards), default=None)
        spoken = digest_line(q, len(cards), refused, newest)
        self.searches += 1
        with contextlib.suppress(Exception):
            self._bus.publish("social.results", {
                "query": scrub(q, MAX_QUERY_CHARS), "cards": cards,
                "count": len(cards), "refused": refused,
                "backend": backend,
                # Spec §13: metered on-screen, and never an invented dollar
                # figure — a session counter is what we actually know.
                "meter": {"searches": self.searches}})
        if speak_digest and self._speak is not None:
            # The butler-tool path: the digest goes DIRECTLY to TTS from
            # here, because the tool's return value is the fixed sentence
            # and nothing else will ever speak the digest. Guarded — a TTS
            # fault costs the digest (cards are on screen), never the turn.
            try:
                await self._speak(spoken)
            except Exception:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    self._bus.publish("social.error",
                                      {"reason": "digest speech failed"})
        return {"ok": True, "count": len(cards), "refused": refused,
                "spoken": spoken, "butler_line": butler_line(len(cards)),
                "reason": ""}


def build_social_server(search: SocialSearch):
    """The butler's ONE social tool. Its result is the entire cross-section
    of what the butler may learn from a search: the fixed sentence on
    success, a fixed failure class on failure. Post content, author names,
    URLs — none of it exists on this side of the boundary, so no session
    ever holds untrusted content + private data + an action channel."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool("social_search",
          "Run a quarantined X/Twitter search. You never see post content: "
          "validated results appear on Keke's screen as cards and a short "
          "digest is spoken automatically. This tool returns only a fixed "
          "completion note — after calling it, acknowledge briefly and do "
          "NOT invent, guess at, or summarize what the posts said.",
          {"query": str})
    async def _search(args):
        res = await search.run(str((args or {}).get("query") or ""),
                               speak_digest=True)
        text = res["butler_line"] if res["ok"] else f"search failed: {res['reason']}"
        return {"content": [{"type": "text", "text": text}]}

    return create_sdk_mcp_server("social", "1.0.0", [_search])
