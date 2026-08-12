import json
from pathlib import Path
from server.config import Config, ensure_config, load_config, load_keyterms


def test_ensure_config_creates_versioned_file_with_secrets(tmp_path: Path):
    p = tmp_path / "marvin.json"
    cfg = ensure_config(p)
    assert p.exists()
    raw = json.loads(p.read_text())
    assert raw["schema_version"] == 1
    assert len(cfg.install_secret) == 64  # 32 bytes hex
    assert len(cfg.hook_bearer) == 64
    assert cfg.install_secret != cfg.hook_bearer
    # idempotent: second call loads the same secrets
    cfg2 = ensure_config(p)
    assert cfg2.install_secret == cfg.install_secret
    mode = p.stat().st_mode & 0o777
    assert mode == 0o600


def test_load_config_rejects_future_schema(tmp_path: Path):
    p = tmp_path / "marvin.json"
    p.write_text(json.dumps({"schema_version": 99}))
    try:
        load_config(p)
        assert False, "should have raised"
    except ValueError as e:
        assert "schema_version" in str(e)


def test_load_keyterms_missing_file_gives_empty(tmp_path: Path):
    assert load_keyterms(tmp_path / "keyterms.json") == []
