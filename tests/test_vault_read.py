import asyncio
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
    out = await vault_search("chant", tmp_path, limit=5)
    results = out["results"]
    assert any(r["title"] == "Tibet" for r in results)
    hit = next(r for r in results if r["title"] == "Tibet")
    assert hit["path"] == "Wiki/Tibet.md"
    assert "chant" in hit["snippet"].lower()
    assert out["total"] == 1


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
        out = await vault_search("chant", tmp_path, limit=5)
    finally:
        os.dup2(saved, 0)
        os.close(saved)
    assert any(r["path"] == "Wiki/Tibet.md" for r in out["results"])


@pytest.mark.skipif(not HAS_RG, reason="ripgrep not installed")
async def test_search_is_ranked_and_deterministic(tmp_path):
    """`rg -l` + `[:limit]` returned ripgrep's parallel-walk order.

    Against the real vault "Tibet" matches 164 files and the top 3 CHANGED on
    three consecutive runs, so the butler saw 5 arbitrary notes out of 164 and
    never learned the rest existed. Rank by match count, and break every tie
    deterministically -- with chat transcripts pushed behind Keke's own notes.
    """
    (tmp_path / "Wiki").mkdir(parents=True)
    (tmp_path / "Claude Chats").mkdir(parents=True)
    (tmp_path / "Wiki" / "Many.md").write_text("chant chant chant\n", encoding="utf-8")
    (tmp_path / "Wiki" / "One.md").write_text("chant\n", encoding="utf-8")
    (tmp_path / "Claude Chats" / "Archive.md").write_text(
        "chant chant chant chant\n", encoding="utf-8")
    out = await vault_search("chant", tmp_path, limit=5)
    paths = [r["path"] for r in out["results"]]
    assert out["total"] == 3
    assert paths[0] == "Wiki/Many.md"          # highest count among non-archives
    assert paths[1] == "Wiki/One.md"
    assert paths[-1] == "Claude Chats/Archive.md"   # archive last despite highest count
    again = await vault_search("chant", tmp_path, limit=5)
    assert [r["path"] for r in again["results"]] == paths   # deterministic


@pytest.mark.skipif(not HAS_RG, reason="ripgrep not installed")
async def test_count_outranks_alphabetical_and_depth(tmp_path):
    """Pins the count key on its own.

    `rg -c` counts matching LINES, not occurrences, so in the test above all
    three files score 1 and the expected order also falls out of the alphabetical
    tiebreak -- it would pass with the count key deleted. Here the counts differ
    for real, and both weaker keys point the other way: 'Zeta' loses on
    alphabetical order and Deep/ loses on depth.
    """
    (tmp_path / "Wiki" / "Deep").mkdir(parents=True)
    (tmp_path / "Alpha.md").write_text("chant\n", encoding="utf-8")
    (tmp_path / "Wiki" / "Deep" / "Zeta.md").write_text(
        "chant\nchant\nchant\n", encoding="utf-8")
    out = await vault_search("chant", tmp_path, limit=5)
    assert [r["path"] for r in out["results"]] == ["Wiki/Deep/Zeta.md", "Alpha.md"]
    assert [r["count"] for r in out["results"]] == [3, 1]


@pytest.mark.skipif(not HAS_RG, reason="ripgrep not installed")
async def test_depth_breaks_a_count_tie(tmp_path):
    """At equal counts the shallower, curated note wins."""
    (tmp_path / "Wiki" / "Archive" / "Old").mkdir(parents=True)
    (tmp_path / "Wiki" / "Zzz.md").write_text("chant\n", encoding="utf-8")
    (tmp_path / "Wiki" / "Archive" / "Old" / "Aaa.md").write_text(
        "chant\n", encoding="utf-8")
    out = await vault_search("chant", tmp_path, limit=5)
    assert [r["path"] for r in out["results"]] == [
        "Wiki/Zzz.md", "Wiki/Archive/Old/Aaa.md"]


@pytest.mark.skipif(not HAS_RG, reason="ripgrep not installed")
async def test_chatgpt_archive_is_deprioritized_but_not_excluded(tmp_path):
    """The other archive prefix, and the "not excluded" half of the rule: a term
    that lives ONLY in a transcript must still come back."""
    (tmp_path / "ChatGPT Chats").mkdir(parents=True)
    (tmp_path / "ChatGPT Chats" / "T.md").write_text("chant\n", encoding="utf-8")
    out = await vault_search("chant", tmp_path, limit=5)
    assert [r["path"] for r in out["results"]] == ["ChatGPT Chats/T.md"]

    (tmp_path / "Wiki").mkdir()
    (tmp_path / "Wiki" / "Note.md").write_text("chant\n", encoding="utf-8")
    out = await vault_search("chant", tmp_path, limit=5)
    assert [r["path"] for r in out["results"]] == ["Wiki/Note.md", "ChatGPT Chats/T.md"]


@pytest.mark.skipif(not HAS_RG, reason="ripgrep not installed")
async def test_search_reports_total_beyond_the_limit(tmp_path):
    """`total` is the whole point of the new shape: the caller must be able to say
    "164 matched, here are 5" instead of implying five is all there is."""
    (tmp_path / "Wiki").mkdir(parents=True)
    for i in range(9):
        (tmp_path / "Wiki" / f"N{i}.md").write_text("chant\n", encoding="utf-8")
    out = await vault_search("chant", tmp_path, limit=2)
    assert out["total"] == 9
    assert len(out["results"]) == 2


async def test_search_empty_query_returns_empty_shape(tmp_path):
    assert await vault_search("   ", tmp_path) == {"total": 0, "results": []}


async def test_search_failure_is_distinguishable_from_empty_vault(tmp_path, monkeypatch):
    """rg-not-found (or a bad cwd) must NOT return the genuine zero-match shape:
    the butler would affirmatively tell Keke her vault has nothing on a topic
    it simply failed to search."""
    import server.vault_read as vr

    async def no_rg(*args, **kwargs):
        raise FileNotFoundError("[Errno 2] No such file or directory: 'rg'")

    monkeypatch.setattr(vr.asyncio, "create_subprocess_exec", no_rg)
    out = await vault_search("chant", tmp_path)
    assert out.get("error"), "failure path returned no error key"
    assert out != {"total": 0, "results": []}   # distinguishable from a true zero-hit
    assert out["total"] == 0 and out["results"] == []


async def test_search_timeout_carries_an_error(tmp_path, monkeypatch):
    import server.vault_read as vr

    class NeverDone:
        returncode = None
        async def communicate(self):
            await asyncio.sleep(60)
        def kill(self):
            pass

    async def slow_rg(*args, **kwargs):
        return NeverDone()

    monkeypatch.setattr(vr.asyncio, "create_subprocess_exec", slow_rg)
    monkeypatch.setattr(vr, "SEARCH_TIMEOUT", 0.01)
    out = await vault_search("chant", tmp_path)
    assert "timed out" in out.get("error", "")


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


def test_read_is_size_capped_with_a_visible_marker(tmp_path):
    seed(tmp_path)
    big = "x" * 500_000
    (tmp_path / "Wiki" / "Big.md").write_text(big, encoding="utf-8")
    out = vault_read("Wiki/Big.md", tmp_path)
    # the BODY is capped; the marker rides on top of it and must be visible
    body, sep, marker = out.rpartition("\n\n[TRUNCATED: ")
    assert sep, "truncation marker missing"
    assert len(body.encode("utf-8")) <= 200_000
    assert "first 200,000 of 500,000 bytes" in out
    assert "NOT read" in out


def test_read_small_file_has_no_truncation_marker(tmp_path):
    seed(tmp_path)
    assert "[TRUNCATED" not in vault_read("Wiki/Tibet.md", tmp_path)


def test_read_non_utf8_note_cannot_inflate_past_the_cap(tmp_path):
    """Every invalid byte decodes to U+FFFD (3 bytes re-encoded), so a 200KB
    binary blob used to decode into ~600KB of replacement characters."""
    seed(tmp_path)
    (tmp_path / "Wiki" / "Blob.md").write_bytes(b"\xff" * 300_000)
    out = vault_read("Wiki/Blob.md", tmp_path)
    body, sep, _ = out.rpartition("\n\n[TRUNCATED: ")
    assert sep
    assert len(body.encode("utf-8")) <= 200_000


def test_vault_is_downloaded(tmp_path):
    assert vault_is_downloaded(tmp_path) is False
    seed(tmp_path)
    assert vault_is_downloaded(tmp_path) is True
