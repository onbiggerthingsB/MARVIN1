from datetime import datetime
from pathlib import Path

import pytest

from server.vault_paths import (
    assert_append_allowed, capture_target, is_within, log_target, vault_root_from_env,
)

NOW = datetime(2026, 8, 8, 14, 30)


def test_targets_are_fixed_destinations(tmp_path):
    assert capture_target(tmp_path, NOW) == tmp_path / "Daily" / "2026-08-08.md"
    assert log_target(tmp_path) == tmp_path / "_Claude" / "log.md"


def test_env_override(monkeypatch):
    monkeypatch.setenv("JARVIS_VAULT", "/tmp/some vault")
    assert vault_root_from_env() == Path("/tmp/some vault")


def test_env_default_when_unset(monkeypatch):
    monkeypatch.delenv("JARVIS_VAULT", raising=False)
    root = vault_root_from_env()
    assert root.name == "KEKE LI"
    assert "iCloud~md~obsidian" in str(root)


def test_append_allowed_for_the_two_targets(tmp_path):
    assert_append_allowed(capture_target(tmp_path, NOW), tmp_path, NOW)  # no raise
    assert_append_allowed(log_target(tmp_path), tmp_path, NOW)           # no raise


def test_append_rejects_any_other_path(tmp_path):
    for bad in [tmp_path / "Research" / "notes.md",
                tmp_path / "_Claude" / "index.md",
                tmp_path / "Daily" / "2026-08-09.md"]:  # wrong day
        with pytest.raises(PermissionError):
            assert_append_allowed(bad, tmp_path, NOW)


def test_is_within_rejects_sibling_prefix_collision(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    evil = tmp_path / "vault-evil"
    evil.mkdir()
    assert is_within(root / "Daily" / "note.md", root) is True
    assert is_within(root, root) is True
    assert is_within(evil / "note.md", root) is False   # string-prefix bug would pass this


def test_append_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir()
    daily = tmp_path / "Daily"
    daily.symlink_to(outside)  # Daily/ -> outside the vault
    # capture_target resolves through the symlink to outside → must be rejected
    with pytest.raises(PermissionError):
        assert_append_allowed(capture_target(tmp_path, NOW), tmp_path, NOW)


def test_append_rejects_symlinked_daily_note_inside_vault(tmp_path):
    """An IN-VAULT symlink defeats is_within; only the literal-component
    allowlist can catch it. The old code resolved the same target expression
    into its own allowlist, so this membership test could never fire."""
    (tmp_path / "Daily").mkdir()
    (tmp_path / "Research").mkdir()
    victim = tmp_path / "Research" / "essay.md"
    victim.write_text("Keke's own prose\n", encoding="utf-8")
    (tmp_path / "Daily" / f"{NOW:%Y-%m-%d}.md").symlink_to(victim)
    with pytest.raises(PermissionError):
        assert_append_allowed(capture_target(tmp_path, NOW), tmp_path, NOW)


def test_append_rejects_symlinked_log_inside_vault(tmp_path):
    (tmp_path / "_Claude").mkdir()
    (tmp_path / "Coursework").mkdir()
    victim = tmp_path / "Coursework" / "essay.md"
    victim.write_text("essay body\n", encoding="utf-8")
    (tmp_path / "_Claude" / "log.md").symlink_to(victim)
    with pytest.raises(PermissionError):
        assert_append_allowed(log_target(tmp_path), tmp_path, NOW)
