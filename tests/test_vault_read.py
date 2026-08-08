import os
import shutil
from pathlib import Path

import pytest

from server.vault_read import vault_is_downloaded, vault_read, vault_search

HAS_RG = shutil.which("rg") is not None


def seed(vault: Path):
    (vault / "_Claude").mkdir(parents=True)
    (vault / "_Claude" / "index.md").write_text("# Index\n", encoding="utf-8")
    (vault / "Wiki").mkdir()
    (vault / "Wiki" / "Tibet.md").write_text(
        "# Tibet\nThe chant study replicated in session 2.\n", encoding="utf-8")


@pytest.mark.skipif(not HAS_RG, reason="ripgrep not installed")
async def test_search_finds_seeded_term(tmp_path):
    seed(tmp_path)
    results = await vault_search("chant", tmp_path, limit=5)
    assert any(r["title"] == "Tibet" for r in results)
    hit = next(r for r in results if r["title"] == "Tibet")
    assert hit["path"] == "Wiki/Tibet.md"
    assert "chant" in hit["snippet"].lower()


@pytest.mark.skipif(not HAS_RG, reason="ripgrep not installed")
async def test_search_ignores_inherited_stdin(tmp_path):
    # rg searches stdin instead of the cwd when fd 0 looks like a regular file, which
    # happens whenever the server is launched with its stdin redirected from a file.
    seed(tmp_path)
    decoy = tmp_path / "stdin.txt"
    decoy.write_text("not the vault\n", encoding="utf-8")
    saved = os.dup(0)
    try:
        with open(decoy, "rb") as f:
            os.dup2(f.fileno(), 0)
        results = await vault_search("chant", tmp_path, limit=5)
    finally:
        os.dup2(saved, 0)
        os.close(saved)
    assert any(r["path"] == "Wiki/Tibet.md" for r in results)


def test_read_returns_content(tmp_path):
    seed(tmp_path)
    assert "session 2" in vault_read("Wiki/Tibet.md", tmp_path)


def test_read_rejects_non_markdown(tmp_path):
    seed(tmp_path)
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(PermissionError):
        vault_read("secret.txt", tmp_path)


def test_read_rejects_path_escape(tmp_path):
    seed(tmp_path)
    with pytest.raises(PermissionError):
        vault_read("../../etc/hosts", tmp_path)


def test_read_rejects_md_escape_outside_vault(tmp_path):
    seed(tmp_path)
    with pytest.raises(PermissionError):
        vault_read("../../etc/hosts.md", tmp_path)  # .md suffix: only containment can reject


def test_read_rejects_sibling_prefix_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    seed(vault)
    evil = tmp_path / "vault-evil"
    evil.mkdir()
    (evil / "note.md").write_text("secret", encoding="utf-8")
    with pytest.raises(PermissionError):
        vault_read("../vault-evil/note.md", vault)


def test_read_missing_file(tmp_path):
    seed(tmp_path)
    with pytest.raises(FileNotFoundError):
        vault_read("Wiki/Nope.md", tmp_path)


def test_read_is_size_capped(tmp_path):
    seed(tmp_path)
    big = "x" * 500_000
    (tmp_path / "Wiki" / "Big.md").write_text(big, encoding="utf-8")
    out = vault_read("Wiki/Big.md", tmp_path)
    assert len(out.encode("utf-8")) <= 200_000


def test_vault_is_downloaded(tmp_path):
    assert vault_is_downloaded(tmp_path) is False
    seed(tmp_path)
    assert vault_is_downloaded(tmp_path) is True
