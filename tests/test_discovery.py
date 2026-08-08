import json
from pathlib import Path

from server.discovery import Candidate, discover, scan_claude_json, scan_git_dirs, slug_for


def fake_home(tmp_path: Path, project_paths: list[str]) -> Path:
    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    (home / ".claude.json").write_text(
        json.dumps({"projects": {p: {} for p in project_paths}}), encoding="utf-8")
    return home


def test_slug_encoding_matches_claude_convention():
    assert slug_for("/Users/likerun/Desktop/dei") == "-Users-likerun-Desktop-dei"
    # a real directory name containing a dash encodes the same way it appears on disk
    assert slug_for("/Users/likerun/quant agent") == "-Users-likerun-quant agent"


def test_scan_claude_json_returns_recorded_paths(tmp_path):
    home = fake_home(tmp_path, ["/a/b", "/c/d"])
    assert sorted(scan_claude_json(home)) == ["/a/b", "/c/d"]


def test_scan_claude_json_tolerates_missing_or_broken_file(tmp_path):
    home = tmp_path / "empty"
    (home / ".claude").mkdir(parents=True)
    assert scan_claude_json(home) == []
    (home / ".claude.json").write_text("{not json", encoding="utf-8")
    assert scan_claude_json(home) == []


def test_scan_git_dirs_finds_repos(tmp_path):
    repo = tmp_path / "code" / "myrepo"
    (repo / ".git").mkdir(parents=True)
    (tmp_path / "code" / "plain").mkdir(parents=True)
    found = scan_git_dirs([tmp_path / "code"])
    assert str(repo) in found
    assert str(tmp_path / "code" / "plain") not in found


async def test_discover_excludes_worktrees_and_missing_paths(tmp_path):
    real = tmp_path / "real"
    (real / ".git").mkdir(parents=True)
    wt = tmp_path / "real" / ".claude" / "worktrees" / "abc"
    wt.mkdir(parents=True)
    home = fake_home(tmp_path, [str(real), str(wt), "/does/not/exist"])
    got = {c.path for c in await discover(home)}
    assert str(real) in got
    assert str(wt) not in got          # ephemeral agent scratch
    assert "/does/not/exist" not in got  # deleted since it was recorded


async def test_discover_ranks_by_evidence(tmp_path):
    strong = tmp_path / "strong"
    (strong / ".git").mkdir(parents=True)
    weak = tmp_path / "weak"
    weak.mkdir()
    home = fake_home(tmp_path, [str(strong), str(weak)])
    # give `strong` a session dir too, so it has two corroborating sources
    (home / ".claude" / "projects" / slug_for(str(strong))).mkdir()
    ranked = await discover(home)
    assert ranked[0].path == str(strong)
    assert ranked[0].has_session_dir is True
    assert ranked[0].is_git is True
    assert "claude.json" in ranked[0].sources


async def test_discover_dedupes_across_sources(tmp_path):
    repo = tmp_path / "dup"
    (repo / ".git").mkdir(parents=True)
    home = fake_home(tmp_path, [str(repo)])
    got = await discover(home, extra_roots=[tmp_path])   # same repo via git scan too
    assert [c.path for c in got].count(str(repo)) == 1
    assert set(got[0].sources) >= {"claude.json", "git-scan"}


async def test_candidate_name_is_the_directory_name(tmp_path):
    repo = tmp_path / "quant agent"
    (repo / ".git").mkdir(parents=True)
    home = fake_home(tmp_path, [str(repo)])
    assert (await discover(home))[0].name == "quant agent"
