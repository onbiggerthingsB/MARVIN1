import json
import subprocess
from pathlib import Path

import pytest

from server.worktrees import (HOOK_EVENTS, Worktree, WorktreeError, _slug,
                              create_worktree, proxy_problem, remove_worktree,
                              write_hook_settings)


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


async def test_create_worktree_records_the_base_commit(tmp_path):
    repo = make_repo(tmp_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()
    wt = await create_worktree(repo, "fix the login redirect", tmp_path / "wts")
    assert wt.base_commit == head
    assert (Path(wt.path) / "README.md").exists()
    wt_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt.path, check=True,
                             capture_output=True, text=True).stdout.strip()
    assert wt_head == head                       # cut from the RECORDED commit


async def test_worktree_excludes_untracked_files_by_construction(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "secrets.env").write_text("KEY=hunter2\n", encoding="utf-8")  # untracked
    wt = await create_worktree(repo, "task", tmp_path / "wts")
    assert not (Path(wt.path) / "secrets.env").exists()


async def test_worktree_branch_is_namespaced(tmp_path):
    repo = make_repo(tmp_path)
    wt = await create_worktree(repo, "Fix The Login!", tmp_path / "wts")
    assert wt.branch.startswith("jarvis/fix-the-login")


async def test_non_git_dir_raises_a_worktree_error(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(WorktreeError):
        await create_worktree(plain, "task", tmp_path / "wts")


async def test_remove_worktree_deletes_the_checkout(tmp_path):
    repo = make_repo(tmp_path)
    wt = await create_worktree(repo, "task", tmp_path / "wts")
    await remove_worktree(wt)
    assert not Path(wt.path).exists()


async def test_remove_worktree_refuses_a_branch_outside_the_jarvis_namespace(tmp_path):
    # A record with a foreign branch must be refused BEFORE any git call —
    # `worktree remove --force` would happily delete a real linked checkout.
    repo = make_repo(tmp_path)
    forged = Worktree(repo=str(repo), path=str(repo), branch="main",
                      base_commit="deadbeef")
    with pytest.raises(WorktreeError, match="jarvis/"):
        await remove_worktree(forged)
    assert repo.exists() and (repo / "README.md").exists()


async def test_remove_worktree_refuses_a_stale_record_on_the_wrong_checkout(tmp_path):
    # Stale/forged record: a legitimate jarvis/ branch name, but the path
    # points at a checkout that is NOT on that branch. Refuse; leave it alone.
    repo = make_repo(tmp_path)
    wt_a = await create_worktree(repo, "task alpha", tmp_path / "wts")
    wt_b = await create_worktree(repo, "task beta", tmp_path / "wts")
    try:
        forged = Worktree(repo=str(repo), path=wt_b.path, branch=wt_a.branch,
                          base_commit=wt_a.base_commit)
        # Pin the guard's OWN branch-mismatch refusal: a bare "refus" also
        # matches "cannot confirm what is checked out there", so a broken
        # rev-parse in the environment would green this test for free.
        with pytest.raises(WorktreeError, match="checked out, not the recorded"):
            await remove_worktree(forged)
        assert Path(wt_b.path).exists()             # the target survived
        assert (Path(wt_b.path) / "README.md").exists()
    finally:
        await remove_worktree(wt_a)
        await remove_worktree(wt_b)


async def test_remove_worktree_refuses_a_jarvis_record_aimed_at_the_main_checkout(tmp_path):
    # The nightmare case: a corrupted record wearing a jarvis/ branch but
    # pointing at the user's real checkout (which is on main, not the branch).
    repo = make_repo(tmp_path)
    wt = await create_worktree(repo, "task", tmp_path / "wts")
    try:
        forged = Worktree(repo=str(repo), path=str(repo), branch=wt.branch,
                          base_commit=wt.base_commit)
        # match= is load-bearing: git refuses to delete a MAIN working tree all
        # by itself ("fatal: ... is a main working tree"), so a bare raises()
        # here passes with the guard fully reverted — it would be testing git,
        # not JARVIS. Pin the guard's own branch-mismatch sentence instead.
        with pytest.raises(WorktreeError,
                           match="has 'main' checked out, not the recorded"):
            await remove_worktree(forged)
        assert repo.exists() and (repo / "README.md").exists()
    finally:
        await remove_worktree(wt)


async def test_remove_worktree_refuses_a_relative_path(tmp_path, monkeypatch):
    # A relative wt.path is resolved against two DIFFERENT roots: the guard's
    # `git -C <path> rev-parse` resolves it against the server process's cwd,
    # while `git -C <repo> worktree remove --force <path>` finds no such
    # directory in the repo and falls back to matching a registered worktree by
    # NAME. So one directory gets verified and a different one gets deleted.
    victim_root = tmp_path / "victim"
    victim_root.mkdir()
    decoy_root = tmp_path / "decoy"
    decoy_root.mkdir()
    victim_repo = make_repo(victim_root)
    decoy_repo = make_repo(decoy_root)
    victim_w = victim_root / "wts" / "W"
    decoy_w = decoy_root / "wts" / "W"
    # The human's real linked checkout in victim_repo, on their own branch.
    subprocess.run(["git", "worktree", "add", "-q", "-b", "feature/human",
                    str(victim_w)], cwd=victim_repo, check=True)
    (victim_w / "notes.txt").write_text("hours of work\n", encoding="utf-8")
    # A same-NAME decoy in an unrelated repo, wearing a jarvis/ branch so the
    # guard's rev-parse happily says yes.
    subprocess.run(["git", "worktree", "add", "-q", "-b", "jarvis/decoy",
                    str(decoy_w)], cwd=decoy_repo, check=True)
    try:
        monkeypatch.chdir(decoy_w.parent)        # "W" verifies HERE ...
        forged = Worktree(repo=str(victim_repo), path="W",
                          branch="jarvis/decoy", base_commit="deadbeef")
        with pytest.raises(WorktreeError, match="not an absolute path"):
            await remove_worktree(forged)        # ... but git would delete THERE
        assert victim_w.exists()                 # the human's checkout survived
        assert (victim_w / "notes.txt").exists()
        assert (victim_w / "README.md").exists()
        assert decoy_w.exists()                  # and nothing else went either
    finally:
        for repo, wt_dir in ((victim_repo, victim_w), (decoy_repo, decoy_w)):
            subprocess.run(["git", "worktree", "remove", "--force", str(wt_dir)],
                           cwd=repo, check=False, capture_output=True)


async def test_a_cancelled_git_call_kills_the_subprocess(tmp_path, monkeypatch):
    """Cancelling the await must cancel the WORK, not just stop waiting on it.

    app_brain wraps every worktree verb in asyncio.wait_for; on timeout it
    cancels this coroutine and speaks "that command failed" — while git, a
    separate process, carries the removal through anyway. A spoken failure
    followed by the removal succeeding is a lie the same size as a spoken
    success that never happened."""
    import asyncio as aio

    from server import worktrees as mod
    repo = make_repo(tmp_path)
    # An alias, so what blocks is a real git process and not a fake.
    subprocess.run(["git", "config", "alias.snooze", "!sleep 5"], cwd=repo,
                   check=True)
    procs = []
    real = aio.create_subprocess_exec

    async def spy(*a, **kw):
        proc = await real(*a, **kw)
        procs.append(proc)
        return proc

    monkeypatch.setattr(aio, "create_subprocess_exec", spy)
    task = aio.create_task(mod._git(repo, "snooze"))
    await aio.sleep(0.4)
    assert procs and procs[0].returncode is None      # really running
    task.cancel()
    with pytest.raises(aio.CancelledError):
        await task
    assert procs[0].returncode is not None            # killed, and reaped


def test_hook_settings_cover_all_seven_events(tmp_path):
    out = write_hook_settings(tmp_path, port=7777, bearer="tok123")
    body = json.loads(out.read_text(encoding="utf-8"))
    assert set(body["hooks"].keys()) == set(HOOK_EVENTS) == {
        "SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse",
        "Notification", "Stop", "SessionEnd"}
    cmd = body["hooks"]["Stop"][0]["hooks"][0]["command"]
    assert "/hooks" in cmd and "7777" in cmd and "--data-binary @-" in cmd


def test_hook_settings_are_private_and_carry_the_bearer(tmp_path):
    out = write_hook_settings(tmp_path, port=7777, bearer="tok123")
    assert out.name == "settings.local.json"     # never tracked, never in the diff
    assert (out.stat().st_mode & 0o777) == 0o600
    assert "Bearer tok123" in out.read_text(encoding="utf-8")


def test_slug_survives_chinese_and_empty_text():
    # Lesson from M2: a cap that no-ops on Chinese text is a shipped bug. The
    # slug is ASCII-only by construction, so CJK input must yield the fallback,
    # not an empty branch name that git rejects.
    assert _slug("修复登录跳转") == "task"
    assert _slug("") == "task"
    assert _slug("Fix the LOGIN redirect... now") == "fix-the-login-redirect-n"


def test_proxy_problem_flags_missing_proxy_and_bad_no_proxy():
    assert "HTTPS_PROXY" in proxy_problem({})
    assert "NO_PROXY" in proxy_problem({"HTTPS_PROXY": "http://proxy:8080"})


def test_proxy_problem_accepts_a_good_environment_or_the_skip_switch():
    good = {"HTTPS_PROXY": "http://proxy:8080", "NO_PROXY": "localhost,127.0.0.1"}
    assert proxy_problem(good) is None
    assert proxy_problem({"JARVIS_SKIP_PROXY_CHECK": "1"}) is None


def test_proxy_problem_requires_both_no_proxy_hosts():
    # Hooks POST to http://127.0.0.1:<port>/hooks and curl matches NO_PROXY
    # against the literal URL host without resolving it — so `localhost`
    # alone still routes hook traffic through the proxy. Both must be listed.
    proxy = {"HTTPS_PROXY": "http://proxy:8080"}
    assert "NO_PROXY" in proxy_problem({**proxy, "NO_PROXY": "localhost"})
    assert "NO_PROXY" in proxy_problem({**proxy, "NO_PROXY": "127.0.0.1"})
    assert proxy_problem({**proxy, "NO_PROXY": "127.0.0.1,localhost"}) is None
