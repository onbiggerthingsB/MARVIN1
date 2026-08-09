"""Disposable git worktrees for workers, plus the hooks and environment glue
spawn needs (spec §5).

Worktree-per-task is v1, not v2: every worker runs in a checkout cut from a
RECORDED commit, so the real checkout's working tree and untracked files are
safe by construction. Merge-back is a human act — nothing here ever merges,
commits to, or deletes the real repo.

A worktree is NOT a sandbox, and nothing here should be read as claiming one.
`cwd` is the worktree, but Write, Edit and Bash all take absolute paths: in the
live smoke the real worker's FIRST tool call was `Write /tmp/DONE.txt`, and
neither this module nor git can stop that. The blast radius is the whole
filesystem the CLI process can reach. The actual containment is the SPOKEN
APPROVAL in Worker._on_tool_request — every non-read tool call is read aloud
and blocked until Keke answers — which is why that sentence must name the
target truthfully, tail included, and say when the target is outside here.

The hook settings go into the WORKTREE's .claude/settings.local.json:
  - local settings are never tracked, so the human's merge-back diff stays
    clean of JARVIS plumbing;
  - hooks fire from the CLI process, so they keep POSTing after a terminal
    handoff, when the owned SDK stream is gone — that is the second detection
    layer surviving the first.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

HOOK_EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse",
               "Notification", "Stop", "SessionEnd")
GIT_TIMEOUT_S = 30


class WorktreeError(RuntimeError):
    """Spoken to Keke — keep messages short and concrete."""


@dataclass
class Worktree:
    repo: str          # the real checkout this was cut from
    path: str          # the disposable directory the worker runs in
    branch: str
    base_commit: str


def _slug(text: str, max_len: int = 24) -> str:
    """ASCII branch-name slug. CJK and symbols reduce to the 'task' fallback
    rather than an empty string git would reject."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:max_len].rstrip("-") or "task"


async def _git(repo: Path, *args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(repo), *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL)
    out, err = await asyncio.wait_for(proc.communicate(), GIT_TIMEOUT_S)
    if proc.returncode != 0:
        raise WorktreeError(err.decode(errors="replace").strip()
                            or f"git {args[0]} failed")
    return out.decode(errors="replace").strip()


async def create_worktree(repo: Path, task: str, worktrees_dir: Path) -> Worktree:
    repo = Path(repo)
    if not (repo / ".git").exists():
        raise WorktreeError(f"{repo} is not a git checkout")
    base_commit = await _git(repo, "rev-parse", "HEAD")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = _slug(task)
    dest = Path(worktrees_dir) / f"{repo.name}-{slug}-{stamp}"
    branch = f"jarvis/{slug}-{stamp}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    await _git(repo, "worktree", "add", "-b", branch, str(dest), base_commit)
    return Worktree(repo=str(repo), path=str(dest), branch=branch,
                    base_commit=base_commit)


async def remove_worktree(wt: Worktree) -> None:
    """Explicit cleanup only — NEVER called automatically. The worktree holds
    the diff a human may still want to merge back.

    Guard before the destructive call: `worktree remove --force` deletes ANY
    registered linked worktree (untracked files included) — git only protects
    the MAIN working tree. A Worktree is a plain dataclass anyone can forge or
    rehydrate stale from disk, so refuse anything JARVIS did not create:
      1. the recorded branch must live in JARVIS's own jarvis/ namespace;
      2. wt.path must be ABSOLUTE, so the checkout this verifies and the
         worktree git deletes are provably the same directory;
      3. the checkout actually at wt.path must have that exact branch checked
         out RIGHT NOW — this is what stops a stale record from aiming the
         removal at somebody else's directory.

    On (2): a RELATIVE path is resolved against two different roots. The
    verification below resolves it against the server process's cwd, while
    `git -C <repo> worktree remove` finds no such directory under the repo and
    falls back to matching a registered worktree by NAME — so path="worker-3"
    can be verified against some ./worker-3 here and then delete a human's
    unrelated worker-3 checkout inside wt.repo. An absolute path gets no name
    matching, so the two can never diverge. create_worktree only ever records
    absolute paths; anything else is forged or corrupt."""
    if not wt.branch.startswith("jarvis/"):
        raise WorktreeError(
            f"refusing to remove {wt.path}: branch {wt.branch!r} is outside "
            f"the jarvis/ namespace, so JARVIS did not create it")
    if not Path(wt.path).is_absolute():
        raise WorktreeError(
            f"refusing to remove {wt.path!r}: that is not an absolute path, so "
            f"git could delete a same-named worktree in {wt.repo} instead of "
            f"the one I check — stale or forged record")
    target = str(Path(wt.path))   # ONE string: what I verify is what git removes
    try:
        checked_out = await _git(Path(target), "rev-parse", "--abbrev-ref",
                                 "HEAD")
    except WorktreeError as e:
        raise WorktreeError(
            f"refusing to remove {target}: cannot confirm what is checked "
            f"out there ({e})") from e
    if checked_out != wt.branch:
        raise WorktreeError(
            f"refusing to remove {target}: it has {checked_out!r} checked "
            f"out, not the recorded {wt.branch!r} — stale or forged record")
    await _git(Path(wt.repo), "worktree", "remove", "--force", target)


def write_hook_settings(wt_path: Path, port: int, bearer: str) -> Path:
    """Drop .claude/settings.local.json into the worktree so the CLI POSTs
    every lifecycle event to /hooks. The hook payload arrives on the command's
    stdin and carries hook_event_name/session_id/transcript_path/cwd, so ONE
    curl command serves every event. --max-time 3: a wedged server must never
    wedge the worker (a failed hook is non-blocking at exit != 2)."""
    cmd = (f"curl -s --max-time 3 -X POST "
           f"-H 'Authorization: Bearer {bearer}' "
           f"-H 'Content-Type: application/json' "
           f"--data-binary @- http://127.0.0.1:{port}/hooks")
    hooks = {}
    for ev in HOOK_EVENTS:
        entry = {"hooks": [{"type": "command", "command": cmd}]}
        if ev in ("PreToolUse", "PostToolUse"):
            entry["matcher"] = "*"          # every tool, both events
        hooks[ev] = [entry]
    claude_dir = Path(wt_path) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    out = claude_dir / "settings.local.json"
    out.write_text(json.dumps({"hooks": hooks}, indent=2), encoding="utf-8")
    out.chmod(0o600)                        # the bearer token lives in here
    return out


def proxy_problem(env=os.environ) -> str | None:
    """This machine reaches Anthropic only through a proxy; a worker CLI
    without the proxy vars dies with '403 Request not allowed'. Checked at
    spawn so the failure is a spoken sentence, not a mystery inside a worker.
    JARVIS_SKIP_PROXY_CHECK=1 disables it on networks that need no proxy."""
    if env.get("JARVIS_SKIP_PROXY_CHECK") == "1":
        return None
    if not (env.get("HTTPS_PROXY") or env.get("https_proxy")
            or env.get("HTTP_PROXY") or env.get("http_proxy")):
        return ("HTTPS_PROXY is not set — a spawned worker cannot reach "
                "Anthropic from this network")
    no_proxy = (env.get("NO_PROXY") or env.get("no_proxy") or "")
    # BOTH hosts required: hook POSTs go to http://127.0.0.1:<port>/hooks and
    # curl matches NO_PROXY against the literal URL host without resolving it,
    # so `localhost` alone still sends hook traffic through the proxy.
    if "127.0.0.1" not in no_proxy or "localhost" not in no_proxy:
        return ("NO_PROXY must include localhost,127.0.0.1 or the worker's "
                "hooks cannot reach me")
    return None
