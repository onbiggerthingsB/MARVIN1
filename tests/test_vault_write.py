import asyncio
from datetime import datetime
from pathlib import Path

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
