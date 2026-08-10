"""Versioned config for Marlowe. Secrets are generated once and persisted."""
from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class Config:
    schema_version: int = SCHEMA_VERSION
    install_secret: str = ""
    hook_bearer: str = ""
    session_token_hash: str = ""
    port: int = 7777


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text())
    version = raw.get("schema_version")
    if version is None or version > SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version}")
    known = {f: raw[f] for f in Config.__dataclass_fields__ if f in raw}
    return Config(**known)


def save_config(cfg: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(cfg), indent=2))
    tmp.replace(path)
    path.chmod(0o600)


def ensure_config(path: Path) -> Config:
    if path.exists():
        return load_config(path)
    cfg = Config(
        install_secret=secrets.token_hex(32),
        hook_bearer=secrets.token_hex(32),
    )
    save_config(cfg, path)
    return cfg


def load_keyterms(path: Path) -> list[str]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text()))
