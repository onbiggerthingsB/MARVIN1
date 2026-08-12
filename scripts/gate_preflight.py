"""M3P2 milestone gate — preflight.

Run:   cd ~/marlowe && uv run python scripts/gate_preflight.py
       (add --skip-tests to skip the suite when you have just run it)

WHY THIS EXISTS
The gate is a live demo with a human at the microphone: nine beats, real
Claude Code workers, real money-adjacent questions, one recorded result. Every
precondition it needs is knowable BEFORE the first word is spoken, and the
expensive failure mode is discovering a missing one at beat 6 — after a worker
has already burned tokens and the transcript is half written.

So this checks everything, changes NOTHING, and fails loudly with the exact
command to fix each problem. It is read-only: it reads files, stats a port,
opens one TCP connection to the proxy, and (unless --skip-tests) runs pytest.
It never starts the server, never spawns a worker, never redeems the bootstrap
token, and never removes a worktree — cleanup stays a human act, exactly as
worktrees.remove_worktree documents.

The proxy check is deliberately TWO checks. server.worktrees.proxy_problem is
reused verbatim (never reimplemented — it is the same sentence Marvin speaks
at spawn), but it only inspects environment variables: it cannot tell whether
FlClash is actually up. A machine with perfect HTTPS_PROXY vars and a dead
proxy spawns a worker that dies on `403 Request not allowed`, which is exactly
the mystery-inside-a-worker this is meant to prevent. So the endpoint gets a
real TCP connect too.

Not a pytest: it shells out and inspects the live machine. `testpaths =
["tests"]` keeps `uv run pytest` away from it; the decision rules are pure
functions covered by tests/test_gate_kit.py.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# APPENDED, not inserted: the scripts directory must never shadow a stdlib
# module for the server package this then imports.
sys.path.append(str(REPO_ROOT / "scripts"))

# The fleet log reader lives in the observer — one definition of "this record
# verifies", shared by the two halves of the kit.
from gate_observer import (access_log_disabled, fold_workers,  # noqa: E402
                           read_fleet_log)
from server.config import load_config                          # noqa: E402
from server.fleet_state import CLOSED                          # noqa: E402
from server.registry import Registry                           # noqa: E402
from server.worktrees import proxy_problem                     # noqa: E402

OK, WARN, ERROR = "ok", "warn", "error"
DEFAULT_PROXY_PORT = 7890


@dataclass
class Check:
    name: str
    level: str
    detail: str
    fix: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name, level, detail, fix="") -> None:
        self.checks.append(Check(name, level, detail, fix))

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.level == ERROR]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.level == WARN]


# ------------------------------------------------------------ pure rules ---
def parse_env_file(text: str) -> dict[str, str]:
    """Parse a .env the way `set -a && source .env` would, minus the shell.

    bin/marvin sources this file, so what it contains — not what happens to be
    exported in the terminal running the preflight — is what the SERVER will
    see. Getting this wrong in either direction is a false result: reading only
    os.environ would fail a perfectly configured machine, and reading only the
    file would pass one whose .env is missing keys the shell happens to have.
    """
    env: dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        env[key] = val
    return env


def merged_env(base: dict, dotenv: dict) -> dict:
    """`set -a && source .env` OVERWRITES the inherited environment, so the
    file wins. Matching that exactly is the point: this must model the process
    the launcher will start, not a compromise between the two."""
    out = dict(base)
    out.update(dotenv)
    return out


def proxy_endpoint(env: dict) -> tuple[str, int] | None:
    """(host, port) of the proxy the env points at, or None if it names none.
    Falls back to the documented FlClash port when the URL omits one."""
    url = (env.get("HTTPS_PROXY") or env.get("https_proxy")
           or env.get("HTTP_PROXY") or env.get("http_proxy") or "").strip()
    if not url:
        return None
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    return host, int(parsed.port or DEFAULT_PROXY_PORT)


def check_proxy_env(env: dict) -> Check:
    """The spawn-time check itself, run early. Same function, same sentence."""
    if env.get("MARVIN_SKIP_PROXY_CHECK") == "1":
        return Check(
            "proxy env", WARN,
            "MARVIN_SKIP_PROXY_CHECK=1 — the spawn-time proxy check is "
            "DISABLED. On this machine a worker without the proxy dies with "
            "`403 Request not allowed`, and the failure will look like a "
            "broken worker, not a broken network.",
            "unset MARVIN_SKIP_PROXY_CHECK in .env unless you are certain "
            "this network reaches Anthropic directly")
    problem = proxy_problem(env=env)
    if problem:
        return Check(
            "proxy env", ERROR,
            f"proxy_problem() says: {problem}. Marvin would refuse the beat-2 "
            f"spawn with \"I can't spawn safely, sir\".",
            "in ~/marlowe/.env set HTTPS_PROXY=http://127.0.0.1:7890, "
            "HTTP_PROXY=http://127.0.0.1:7890 and "
            "NO_PROXY=localhost,127.0.0.1 (BOTH hosts — curl matches the "
            "literal URL host, so `localhost` alone still proxies the hook "
            "POSTs)")
    return Check("proxy env", OK,
                 "HTTPS_PROXY/HTTP_PROXY set and NO_PROXY covers both "
                 "localhost and 127.0.0.1")


def summarize_fleet(records: list[dict], damaged: int, has_snapshot: bool,
                    log_exists: bool) -> list[Check]:
    """Beat 9 needs recovery to have something to recover — and needs it to be
    something the demo created, not a ghost left over from earlier testing.

    A worker whose last recorded state is neither CLOSED nor DETACHED becomes
    an "interrupted by a restart" ghost on the NEXT boot, and stays one on
    every boot after that while its worktree still exists. So a stale one does
    not merely add noise: it makes beat 9 report a worker the demo never ran,
    and Keke cannot tell that from the real thing by listening."""
    out: list[Check] = []
    if not log_exists:
        out.append(Check(
            "fleet log", OK,
            "state/fleet.jsonl does not exist yet — a clean slate. It will be "
            "created by the first transition of beat 2, and beat 9 will "
            "recover the worker this demo starts."))
        return out
    folded = fold_workers(records)
    ghosts = {w: i for w, i in folded.items()
              if i.get("state") not in (CLOSED, None)}
    detail = (f"{len(records)} verified record(s), {len(folded)} worker(s) "
              f"in the history")
    if damaged:
        out.append(Check(
            "fleet log damage", WARN,
            f"{damaged} unverifiable line(s) in state/fleet.jsonl. The next "
            f"boot's FleetLog will quarantine the damage to "
            f"`fleet.jsonl.torn-<ts>` and rebuild from the verified prefix, "
            f"and recovery will SAY the log was damaged — during beat 9 that "
            f"extra sentence is easy to misread as part of the report.",
            "either accept it (and expect the damaged-log sentence) or move "
            "state/fleet.jsonl* and state/fleet.snap aside for a clean run"))
    if ghosts:
        rows = "; ".join(
            f"{w}={i.get('state')} ({i.get('project') or '?'})"
            for w, i in ghosts.items())
        out.append(Check(
            "fleet ghosts", WARN,
            f"{len(ghosts)} worker(s) in the log are NOT closed: {rows}. "
            f"Marvin will announce these as interrupted by a restart on the "
            f"very FIRST boot — before beat 9 — and again on every boot while "
            f"their worktrees exist (a known-open item). Beat 9's report will "
            f"contain workers this demo never started.",
            "move state/fleet.jsonl, state/fleet.jsonl.1 and state/fleet.snap "
            "aside (they are gitignored evidence, not config) so beat 9 "
            "recovers only what this demo runs"))
    else:
        out.append(Check("fleet log", OK,
                         detail + " — none left open, so nothing stale will "
                                  "be reported as interrupted"))
    if has_snapshot:
        out.append(Check(
            "fleet snapshot", WARN,
            "state/fleet.snap exists from an earlier run. Its workers are "
            "folded in before the log is replayed, so anything it carries is "
            "part of beat 9's report too.",
            "move state/fleet.snap aside with the log if you want a clean "
            "beat 9"))
    return out


def discovery_wired(sources: dict[str, str]) -> bool:
    """Does anything in the RUNNING system trigger project discovery?

    Onboarding.refresh() merges discovered candidates and ask_next() speaks
    the "I found what looks like X — is that right, sir?" question that beat 1
    is. app.py constructs Onboarding and hands it to the brain, but the brain
    only ever calls handle_reply(): a reply to a question nothing asks. Pass
    every server source EXCEPT onboarding.py (which defines the methods) plus
    the console, and this answers whether a caller exists.

    Scanned rather than assumed so the check self-corrects the day the trigger
    is wired: this stops being an error the moment a caller appears."""
    return any("ask_next" in text or "onboarding.refresh" in text
               for name, text in sources.items()
               if not name.endswith("onboarding.py"))


def check_beat1(wired: bool, projects: list) -> Check:
    """Beat 1 is discovery plus a SPOKEN repo confirm, and beat 2 needs the
    repo that confirm produces. Two different failures live here."""
    confirmed = [p for p in projects if p.confirmed]
    if wired:
        return Check("beat 1 (discovery)", OK,
                     "something in the running server triggers discovery — "
                     "beat 1 can be performed as written")
    if not confirmed:
        return Check(
            "beat 1 (discovery)", ERROR,
            "NOTHING in the running server calls Onboarding.refresh() or "
            "ask_next() — the brain only answers the repo question, and no "
            "code path asks it. Discovery is unreachable by voice, the "
            "registry is empty, and Router._resolve can match no project: "
            "beat 2's \"Start work in …\" will refuse, and beat 8 has no "
            "confirmed finance project either. The gate stops at beat 2.",
            "this is a real M3P2 finding — write it into the report. To run "
            "the REST of the gate, seed the registry by hand (config data, "
            "not code — do not patch server/):\n"
            "     cd ~/marlowe && uv run python - <<'PY'\n"
            "     import asyncio\n"
            "     from pathlib import Path\n"
            "     from server.discovery import discover\n"
            "     from server.registry import Registry\n"
            "     p = Path('config/projects.json')\n"
            "     reg = Registry.load_strict(p)\n"
            "     reg.merge_candidates(asyncio.run(discover(Path.home())))\n"
            "     reg.save(p)\n"
            "     for x in reg.projects: print(x.name, '->', x.path)\n"
            "     PY\n"
            "   then confirm the two repos the gate needs (a code repo for "
            "beats 2-7, a finance repo for beat 8):\n"
            "     uv run python -c \"from pathlib import Path; from "
            "server.registry import Registry; p=Path('config/projects.json'); "
            "r=Registry.load_strict(p); r.confirm('<code repo>'); "
            "r.confirm('<finance repo>', kind='finance'); r.save(p)\"\n"
            "   Leave data_source UNSET so beat 8 still asks its question.\n"
            "   Then record beat 1 as NOT DEMONSTRATED with this reason — a "
            "deviation is a finding, not something to hide.")
    return Check(
        "beat 1 (discovery)", WARN,
        f"nothing in the running server calls Onboarding.refresh()/ask_next(), "
        f"so Marvin will never ASK the repo question — beat 1's spoken confirm "
        f"cannot be performed. The registry already holds "
        f"{len(confirmed)} confirmed repo(s), so beats 2-9 are unaffected.",
        "record beat 1 as NOT DEMONSTRATED (discovery is unreachable from the "
        "running server) and write it up as an M3P2 finding")


def summarize_registry(projects: list, loaded: bool, wired: bool = False) -> list[Check]:
    """Beat 2 needs a confirmed repo; beat 8 needs a confirmed FINANCE project
    whose source has NOT been pinned yet.

    `wired` changes what "nothing confirmed" MEANS. With discovery wired, an
    empty registry is the CORRECT starting state — beat 1 is the beat that
    fills it, live and out loud. Calling that a blocker would push Keke to
    hand-seed the registry and so skip the very beat being gated on. Only
    when discovery is unreachable is an empty registry a real blocker,
    because then nothing else can ever fill it."""
    out: list[Check] = []
    if not loaded:
        out.append(Check(
            "registry", ERROR,
            "config/projects.json exists but could not be parsed. The server "
            "would quarantine it to `projects.json.corrupt-<ts>` at boot and "
            "start with an EMPTY registry — every confirmation gone, silently, "
            "in the middle of the gate.",
            "inspect config/projects.json by hand, or move it aside "
            "deliberately so beat 1 rebuilds it"))
        return out
    confirmed = [p for p in projects if p.confirmed]
    pinned = [p for p in projects if getattr(p, "data_source", None)]
    finance = [p for p in confirmed if p.kind == "finance"]
    if not confirmed and wired:
        out.append(Check(
            "registry", OK,
            f"no confirmed repo yet ({len(projects)} discovered) — which is "
            f"the CORRECT state to start from. Beat 1 is what fills this: "
            f"Marvin asks, you say yes, and beat 2 then has a repo to name. "
            f"Do NOT hand-seed the registry; that would skip beat 1."))
    elif not confirmed:
        out.append(Check(
            "registry", ERROR,
            f"no CONFIRMED repo ({len(projects)} discovered). "
            f"Registry.match only ever returns confirmed projects, so beat "
            f"2's \"Start work in <repo>, <task>\" resolves to nothing and "
            f"Marvin refuses.",
            "confirm the repo the gate will use — see the beat 1 fix above"))
    else:
        out.append(Check(
            "registry", OK,
            f"{len(confirmed)} confirmed repo(s): "
            f"{', '.join(p.name for p in confirmed)} "
            f"({len(projects) - len(confirmed)} discovered but unconfirmed). "
            f"Beat 2 must name one of the confirmed ones."))
    if finance:
        out.append(Check("finance project", OK,
                         f"beat 8 has a confirmed finance project: "
                         f"{', '.join(p.name for p in finance)}"))
    elif wired:
        out.append(Check(
            "finance project", OK,
            "no confirmed finance project yet — expected before beat 1. "
            "Onboarding upgrades a repo to kind=\"finance\" on confirmation "
            "when its name or path looks like finance (quant/stock/trad/"
            "invest/portfolio/finance), so say yes to your stock repo during "
            "beat 1 and beat 8 has its project. If yours matches none of "
            "those words, beat 8 is the one to hand-seed — see below."))
    else:
        out.append(Check(
            "finance project", ERROR,
            "no confirmed project with kind=\"finance\". "
            "find_finance_project() returns None, so \"How are the picks "
            "doing?\" gets the no-finance-repo answer and beat 8 cannot run.",
            "confirm the finance repo with the finance kind:\n"
            "     uv run python -c \"from pathlib import Path; from "
            "server.registry import Registry; p=Path('config/projects.json'); "
            "r=Registry.load_strict(p); r.confirm('<finance repo>', "
            "kind='finance'); r.save(p)\"\n"
            "   It also needs a readable .sqlite/.db or .json output inside "
            "it, or the source question has no file to name."))
    if pinned:
        out.append(Check(
            "finance source", WARN,
            f"a data_source is already pinned for "
            f"{', '.join(p.name for p in pinned)}. The §16 question is asked "
            f"ONCE and then pinned, so beat 8 will go straight to the brief "
            f"and look like it was skipped.",
            "clear the \"data_source\" field for that project in "
            "config/projects.json (leave everything else alone) so beat 8 "
            "asks the question on the first ask"))
    else:
        out.append(Check("finance source", OK,
                         "no data_source pinned — beat 8 will ask the source "
                         "question on the first ask"))
    return out


def check_voice(env: dict) -> list[Check]:
    """STT is a hard requirement; TTS is not.

    Without DEEPGRAM_API_KEY the /mic socket publishes stt.error and closes
    with 4500 — there is no way to say anything to Marvin, so eight of the
    nine beats cannot be performed at all. Without the ElevenLabs pair,
    SpeakEngine._eleven_enabled is False and every line comes out of the macOS
    `say` voice: the demo still passes, it just does not sound like Marvin."""
    out: list[Check] = []
    if env.get("DEEPGRAM_API_KEY"):
        out.append(Check("STT (Deepgram)", OK,
                         "DEEPGRAM_API_KEY is set — the mic can transcribe"))
    else:
        out.append(Check(
            "STT (Deepgram)", ERROR,
            "DEEPGRAM_API_KEY is not set. The /mic WebSocket publishes "
            "stt.error and closes with code 4500, so nothing you say reaches "
            "Marvin — this is a voice gate and it cannot start.",
            "add DEEPGRAM_API_KEY=<key> to ~/marlowe/.env (it is currently "
            "present but commented out)"))
    forced = (env.get("MARVIN_VOICE") or "").strip().lower()
    has_pair = bool(env.get("ELEVENLABS_API_KEY")
                    and env.get("ELEVENLABS_VOICE_ID"))
    if forced == "say":
        out.append(Check(
            "TTS (ElevenLabs)", WARN,
            "MARVIN_VOICE=say forces the fallback voice. The demo will be "
            "spoken by macOS `say` — every beat still works and every "
            "readback is still read aloud, it just is not the Marvin voice.",
            "unset MARVIN_VOICE in .env to use ElevenLabs"))
    elif has_pair:
        out.append(Check("TTS (ElevenLabs)", OK,
                         "ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID are set "
                         "— the Marvin voice"))
    else:
        missing = [k for k in ("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID")
                   if not env.get(k)]
        out.append(Check(
            "TTS (ElevenLabs)", WARN,
            f"{' and '.join(missing)} not set. SpeakEngine needs BOTH, so it "
            f"degrades cleanly to the macOS `say` voice. The demo is fully "
            f"performable — beat 3's readback and beat 4's fleet line are "
            f"still spoken — it just sounds like the system voice, which is "
            f"worth a sentence in the report so nobody thinks TTS failed.",
            "add ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID to ~/marlowe/.env "
            "if you want the Marvin voice on the recording"))
    return out


def access_log_check(launcher_text: str) -> Check:
    """Beat 7's first evidence lane is `POST /hooks` in state/server.log —
    which only exists if uvicorn's access log is on. bin/marvin passes
    --no-access-log, so as shipped that lane is silent."""
    if not launcher_text:
        return Check("access log", WARN,
                     "could not read bin/marvin — cannot tell whether the "
                     "access log will be on",
                     "check by hand that uvicorn is not started with "
                     "--no-access-log")
    if not access_log_disabled(launcher_text):
        return Check("access log", OK,
                     "bin/marvin leaves uvicorn's access log on — `POST "
                     "/hooks` lines will land in state/server.log")
    return Check(
        "access log", ERROR,
        "bin/marvin starts uvicorn with --no-access-log, so `POST /hooks` "
        "will NEVER appear in state/server.log. That is beat 7's stated "
        "evidence, and the beat with no automated proof anywhere else.",
        "start the server yourself BEFORE running `marvin`, with the access "
        "log on (do the same for beat 9's restart):\n"
        "     cd ~/marlowe && set -a && . ./.env && set +a && \\\n"
        "       nohup uv run uvicorn server.app:app_factory --factory \\\n"
        "         --host 127.0.0.1 --port 7777 --access-log \\\n"
        "         >> state/server.log 2>&1 &\n"
        "   then run `marvin` — it sees /health answering and will not start "
        "a second one.\n"
        "   (If the console then says \"not bootstrapped\", open the URL in "
        "state/bootstrap_url once.)\n"
        "   Do NOT edit bin/marvin for this: the launcher is part of the "
        "system under test.\n"
        "   Beat 7 still has a second lane either way — the observer proves "
        "it from records reaching a DETACHED worker.")


def exit_code(report: Report) -> int:
    return 1 if report.errors else 0


# ---------------------------------------------------------- live probes ----
def tcp_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def port_holder(port: int) -> str:
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def git(args: list[str], cwd: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(cwd), *args],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def scan_worktrees(worktrees_dir: Path, extra_repos: list[str]) -> list[Check]:
    """Leftover worktrees and marvin/* branches from previous runs.

    Nothing removes a worktree automatically — that is deliberate, the
    worktree holds a diff a human may still want — so they accumulate, and a
    pile of them makes beat 2's "fresh worktree" and beat 9's ghost report
    hard to read. Reported and listed; never touched.

    The worktrees all live under state/worktrees regardless of which repo they
    were cut from, so that directory is the reliable index; the BRANCH lives
    in the origin repo, which each worktree can name itself."""
    out: list[Check] = []
    dirs = []
    if worktrees_dir.is_dir():
        dirs = sorted(p for p in worktrees_dir.iterdir() if p.is_dir())
    repos = {str(REPO_ROOT)} | {r for r in extra_repos if r}
    rows = []
    for d in dirs:
        branch = git(["rev-parse", "--abbrev-ref", "HEAD"], d) or "?"
        common = git(["rev-parse", "--path-format=absolute",
                      "--git-common-dir"], d)
        origin = str(Path(common).parent) if common else "?"
        if origin != "?":
            repos.add(origin)
        rows.append(f"{d.name}  [{branch}]  from {origin}")
    branches = []
    for repo in sorted(repos):
        listed = git(["branch", "--list", "marvin/*"], Path(repo))
        for line in listed.splitlines():
            name = line.strip().lstrip("* ").strip()
            if name:
                branches.append(f"{name}  in {repo}")
    if not rows and not branches:
        out.append(Check("leftover worktrees", OK,
                         "no worktrees under state/worktrees and no marvin/* "
                         "branches in the repos I can see"))
        return out
    detail = (f"{len(rows)} leftover worktree(s) and {len(branches)} "
              f"marvin/* branch(es) from earlier runs:")
    for r in rows:
        detail += f"\n     - {r}"
    for b in branches:
        detail += f"\n     - branch {b}"
    out.append(Check(
        "leftover worktrees", WARN, detail,
        "nothing removes these automatically by design (the worktree holds a "
        "diff a human may still want). If they are spent, remove each with:\n"
        "     git -C <origin repo> worktree remove --force "
        "<state/worktrees/...>\n"
        "     git -C <origin repo> branch -D marvin/<...>\n"
        "   A pile of them also keeps old ghosts re-announcing on every boot "
        "(beat 9's known-open item)."))
    return out


def check_state_dir(state: Path) -> list[Check]:
    out: list[Check] = []
    if not state.is_dir():
        out.append(Check(
            "state/", ERROR, f"{state} does not exist",
            "mkdir -p ~/marlowe/state  (the server creates it at boot, but the "
            "observer wants to tail it from before the boot)"))
        return out
    if not os.access(state, os.R_OK | os.W_OK):
        out.append(Check("state/", ERROR, f"{state} is not readable/writable",
                         f"fix the permissions on {state}"))
        return out
    out.append(Check("state/", OK, f"{state} exists and is writable"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M3P2 milestone gate preflight")
    ap.add_argument("--skip-tests", action="store_true",
                    help="do not run the suite (it is a hard precondition; "
                         "only skip it if you just ran it)")
    args = ap.parse_args(argv)

    print("=" * 72)
    print("Marvin M3P2 milestone gate — PREFLIGHT")
    print(f"repo: {REPO_ROOT}")
    print("read-only: nothing is started, spawned, removed or redeemed here.")
    print("=" * 72)

    report = Report()

    # --- environment -------------------------------------------------------
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        dotenv = parse_env_file(env_path.read_text(encoding="utf-8",
                                                   errors="replace"))
        report.add(".env", OK,
                   f"{env_path} present with {len(dotenv)} assignment(s): "
                   f"{', '.join(sorted(dotenv)) or '(none)'}")
    else:
        dotenv = {}
        report.add(".env", ERROR, f"{env_path} is missing — bin/marvin sources "
                                  f"it, so the server would start with no "
                                  f"proxy and no keys at all",
                   "create ~/marlowe/.env with HTTPS_PROXY, HTTP_PROXY, "
                   "NO_PROXY and DEEPGRAM_API_KEY")
    env = merged_env(dict(os.environ), dotenv)

    report.checks.append(check_proxy_env(env))
    endpoint = proxy_endpoint(env)
    if endpoint is None:
        report.add("proxy reachable", ERROR,
                   "no proxy URL in the environment, so there is nothing to "
                   "probe",
                   "set HTTPS_PROXY=http://127.0.0.1:7890 in ~/marlowe/.env")
    else:
        host, port = endpoint
        if tcp_open(host, port):
            report.add("proxy reachable", OK,
                       f"{host}:{port} is listening (proxy_problem() only "
                       f"reads env vars — this is the part it cannot see)")
        else:
            report.add(
                "proxy reachable", ERROR,
                f"nothing is listening on {host}:{port}. The env vars are "
                f"fine, so Marvin will happily spawn — and the worker will "
                f"die on `403 Request not allowed` somewhere inside beat 2 "
                f"with no explanation.",
                "start FlClash and confirm it is serving on "
                f"{host}:{port}, then run this preflight again")

    for c in check_voice(env):
        report.checks.append(c)

    # --- config + state ----------------------------------------------------
    cfg_path = REPO_ROOT / "config" / "marvin.json"
    port = 7777
    try:
        cfg = load_config(cfg_path)
        port = cfg.port
        report.add("config/marvin.json", OK,
                   f"loads (schema v{cfg.schema_version}, port {cfg.port}); "
                   f"hook bearer and session hash present: "
                   f"{bool(cfg.hook_bearer)}/{bool(cfg.session_token_hash)}")
    except Exception as e:  # noqa: BLE001
        report.add("config/marvin.json", ERROR,
                   f"{cfg_path} could not be loaded: {e}",
                   "the server calls ensure_config() and would fail to boot. "
                   "Move the file aside to regenerate it — note that this "
                   "invalidates the console session cookie, so you will need "
                   "the URL in state/bootstrap_url")

    state = REPO_ROOT / "state"
    for c in check_state_dir(state):
        report.checks.append(c)

    fleet_path = state / "fleet.jsonl"
    records, damaged = read_fleet_log(fleet_path)
    for c in summarize_fleet(records, damaged,
                             (state / "fleet.snap").exists(),
                             fleet_path.exists()):
        report.checks.append(c)

    # --- registry ----------------------------------------------------------
    reg_path = REPO_ROOT / "config" / "projects.json"
    projects, loaded = [], True
    if reg_path.exists():
        try:
            projects = Registry.load_strict(reg_path).projects
        except Exception:  # noqa: BLE001
            loaded = False
    sources = {}
    for p in sorted((REPO_ROOT / "server").glob("*.py")):
        try:
            sources[p.name] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    for p in sorted((REPO_ROOT / "static").glob("*.html")):
        try:
            sources[p.name] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    report.checks.append(check_beat1(discovery_wired(sources), projects))

    for c in summarize_registry(projects, loaded, wired=discovery_wired(sources)):
        report.checks.append(c)

    # --- worktrees ---------------------------------------------------------
    for c in scan_worktrees(state / "worktrees", [p.path for p in projects]):
        report.checks.append(c)

    # --- the port ----------------------------------------------------------
    holder = port_holder(port)
    if not holder:
        report.add(f"port {port}", OK, "free — the launcher will start a "
                                       "fresh server")
    else:
        report.add(
            f"port {port}", ERROR,
            f"something already holds {port}:\n     "
            + "\n     ".join(holder.splitlines()),
            f"the gate needs a clean boot (beat 9 restarts the server, and "
            f"bin/marvin skips the bootstrap URL when /health already "
            f"answers). Stop it first: kill the PID above, or "
            f"`pkill -f 'uvicorn server.app'`")

    # --- beat 7's access log ----------------------------------------------
    launcher = REPO_ROOT / "bin" / "marvin"
    try:
        launcher_text = launcher.read_text(encoding="utf-8")
    except OSError:
        launcher_text = ""
    report.checks.append(access_log_check(launcher_text))

    # --- beat 6's terminal -------------------------------------------------
    if shutil.which("osascript"):
        report.add("osascript", OK, "present — beat 6 can open Terminal.app")
    else:
        report.add("osascript", ERROR, "osascript not found; beat 6 cannot "
                                       "open a Terminal window",
                   "beat 6 would still DETACH and print the resume command on "
                   "the tile — run it by hand and note the deviation")
    claude = shutil.which("claude")
    if not claude:
        # Terminal.app opens a LOGIN shell, whose PATH may differ from the one
        # this process inherited — so a miss here is not yet a verdict.
        try:
            found = subprocess.run(["/bin/zsh", "-lc", "command -v claude"],
                                   capture_output=True, text=True, timeout=20)
            claude = found.stdout.strip() if found.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            claude = ""
    if claude:
        report.add("claude CLI", OK,
                   f"on PATH at {claude} — beat 6's `claude --resume` will "
                   f"run in the new Terminal")
    else:
        report.add("claude CLI", ERROR,
                   "`claude` is not on PATH (checked this shell and a login "
                   "zsh). Beat 6 opens a Terminal that immediately fails, and "
                   "beat 7 has nothing to POST hooks from.",
                   "install/relink the Claude Code CLI so `claude` resolves "
                   "in a login shell")

    # --- the suite ---------------------------------------------------------
    if args.skip_tests:
        report.add("test suite", WARN, "skipped by --skip-tests",
                   "run `uv run pytest -q` before the gate; a green suite is "
                   "the baseline the gate's result is read against")
    else:
        print("\nrunning the suite (uv run pytest -q) …", flush=True)
        proc = subprocess.run(["uv", "run", "pytest", "-q"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=1800)
        tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        last = tail[-1] if tail else "(no output)"
        if proc.returncode == 0:
            report.add("test suite", OK, f"green — {last}")
        else:
            report.add("test suite", ERROR,
                       f"FAILED (exit {proc.returncode}) — {last}",
                       "fix the suite before the gate: a red baseline makes "
                       "every gate deviation ambiguous")

    # --- render ------------------------------------------------------------
    print()
    width = max(len(c.name) for c in report.checks)
    for c in report.checks:
        mark = {OK: "  ok  ", WARN: " WARN ", ERROR: "ERROR "}[c.level]
        first, *rest = c.detail.split("\n")
        print(f"[{mark}] {c.name.ljust(width)}  {first}")
        for line in rest:
            print(f"{' ' * (width + 12)}{line.strip()}")
    print()

    if report.errors:
        print("=" * 72)
        print(f"NOT READY — {len(report.errors)} blocker(s). Fix, in order:")
        print("=" * 72)
        for i, c in enumerate(report.errors, 1):
            print(f"\n{i}. {c.name}: {c.detail.splitlines()[0]}")
            if c.fix:
                for line in c.fix.splitlines():
                    print(f"   {line}")
        if report.warnings:
            print(f"\n(plus {len(report.warnings)} warning(s) below the "
                  f"blockers — see the table above.)")
        print()
        return exit_code(report)

    print("=" * 72)
    print("READY — every hard precondition for the M3P2 gate is met.")
    print("=" * 72)
    for c in report.warnings:
        print(f"\nWARNING — {c.name}: {c.detail.splitlines()[0]}")
        if c.fix:
            for line in c.fix.splitlines():
                print(f"   {line}")
    if not report.warnings:
        print("no warnings.")
    print("\nNext: open a second terminal and run")
    print("      cd ~/marlowe && uv run python scripts/gate_observer.py")
    print("then run `marvin` and follow scripts/gate_checklist.md.")
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
