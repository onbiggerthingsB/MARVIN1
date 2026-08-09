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
