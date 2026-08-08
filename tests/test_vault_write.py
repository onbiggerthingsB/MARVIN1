import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from server.vault_write import vault_capture, vault_log

NOW = datetime(2026, 8, 8, 14, 30)


async def test_capture_creates_daily_with_header_then_appends(tmp_path):
    p1 = await vault_capture("first thought", tmp_path, now=NOW)
    daily = tmp_path / "Daily" / "2026-08-08.md"
    assert Path(p1) == daily
    body = daily.read_text(encoding="utf-8")
    assert body.startswith("# 2026-08-08")
    assert "- 14:30 first thought" in body

    await vault_capture("second thought", tmp_path, now=datetime(2026, 8, 8, 15, 0))
    body2 = daily.read_text(encoding="utf-8")
    assert body2.count("# 2026-08-08") == 1  # header written once
    assert "- 15:00 second thought" in body2


async def test_log_appends_canonical_line(tmp_path):
    (tmp_path / "_Claude").mkdir(parents=True)
    p = await vault_log("create", "Some Note", tmp_path, now=NOW)
    assert Path(p) == tmp_path / "_Claude" / "log.md"
    assert "## [2026-08-08 14:30] create | Some Note" in (
        tmp_path / "_Claude" / "log.md").read_text(encoding="utf-8")


async def test_capture_only_writes_the_daily_file(tmp_path):
    (tmp_path / "Research").mkdir()
    (tmp_path / "Research" / "keep.md").write_text("KEEP", encoding="utf-8")
    await vault_capture("note", tmp_path, now=NOW)
    assert (tmp_path / "Research" / "keep.md").read_text(encoding="utf-8") == "KEEP"


async def test_concurrent_captures_all_land(tmp_path):
    await asyncio.gather(*[
        vault_capture(f"n{i}", tmp_path, now=datetime(2026, 8, 8, 14, 30 + i))
        for i in range(10)])
    body = (tmp_path / "Daily" / "2026-08-08.md").read_text(encoding="utf-8")
    for i in range(10):
        assert f"n{i}" in body


async def test_symlinked_daily_note_cannot_reroute_a_capture(tmp_path):
    vault = tmp_path / "vault"
    (vault / "Daily").mkdir(parents=True)
    (vault / "Research").mkdir()
    victim = vault / "Research" / "essay.md"
    victim.write_text("Keke's own prose\n", encoding="utf-8")
    # the dated note itself is a symlink pointing at Keke's essay
    (vault / "Daily" / f"{NOW:%Y-%m-%d}.md").symlink_to(victim)
    with pytest.raises(PermissionError):
        await vault_capture("must not land", vault, now=NOW)
    assert victim.read_text(encoding="utf-8") == "Keke's own prose\n"   # byte-identical


async def test_symlinked_log_cannot_reroute_a_log_entry(tmp_path):
    vault = tmp_path / "vault"
    (vault / "_Claude").mkdir(parents=True)
    (vault / "Coursework").mkdir()
    victim = vault / "Coursework" / "essay.md"
    victim.write_text("essay body\n", encoding="utf-8")
    (vault / "_Claude" / "log.md").symlink_to(victim)
    with pytest.raises(PermissionError):
        await vault_log("create", "X", vault, now=NOW)
    assert victim.read_text(encoding="utf-8") == "essay body\n"


async def test_capture_refuses_symlinked_daily_and_creates_nothing(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (vault / "Daily").symlink_to(outside)      # Daily/ -> outside the vault
    with pytest.raises(PermissionError):
        await vault_capture("should never land", vault, now=NOW)
    assert list(outside.iterdir()) == []        # assert ran BEFORE any mkdir/open


async def test_multiline_input_is_collapsed_to_one_line(tmp_path):
    # a newline + "# " inside the transcript must not become a real heading
    await vault_capture("first\n# fake heading\nsecond", tmp_path, now=NOW)
    body = (tmp_path / "Daily" / "2026-08-08.md").read_text(encoding="utf-8")
    assert body.splitlines() == [
        "# 2026-08-08", "", "- 14:30 first # fake heading second"]

    # ...and a newline + "## [" must not forge a second audit-log entry
    (tmp_path / "_Claude").mkdir()
    await vault_log("create", "Note\n## [2026-01-01 00:00] delete | Everything",
                    tmp_path, now=NOW)
    log = (tmp_path / "_Claude" / "log.md").read_text(encoding="utf-8")
    assert log.splitlines() == [
        "## [2026-08-08 14:30] create | Note ## [2026-01-01 00:00] delete | Everything"]


async def test_append_repairs_a_missing_trailing_newline(tmp_path):
    daily = tmp_path / "Daily"
    daily.mkdir()
    # Keke's own last line, saved without a trailing newline
    (daily / "2026-08-08.md").write_text(
        "# 2026-08-08\n\n- Ran 5k before practice", encoding="utf-8")
    await vault_capture("call Dr. Kong", tmp_path, now=NOW)
    lines = (daily / "2026-08-08.md").read_text(encoding="utf-8").splitlines()
    assert lines[-2:] == ["- Ran 5k before practice", "- 14:30 call Dr. Kong"]
