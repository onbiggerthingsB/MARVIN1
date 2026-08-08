import json
import sqlite3
from pathlib import Path

from server.discovery import Candidate
from server.finance import (TRADE_REFUSAL, detect_outputs, find_finance_project,
                            portfolio_brief)
from server.registry import Registry


def finance_registry(path="/p/quant agent"):
    r = Registry()
    r.merge_candidates([Candidate(path=path, name="quant agent", sources=["a"])])
    r.confirm("quant agent", kind="finance")
    return r


def test_find_finance_project_requires_confirmation():
    r = Registry()
    r.merge_candidates([Candidate(path="/p/quant agent", name="quant agent", sources=["a"])])
    assert find_finance_project(r) is None          # discovered but unconfirmed
    r.confirm("quant agent", kind="finance")
    assert find_finance_project(r).name == "quant agent"


def test_find_finance_ignores_code_projects():
    r = Registry()
    r.merge_candidates([Candidate(path="/p/soccer", name="soccer", sources=["a"])])
    r.confirm("soccer", kind="code")
    assert find_finance_project(r) is None


def test_detect_outputs_finds_each_kind(tmp_path):
    (tmp_path / "signals.sqlite").write_bytes(b"")
    (tmp_path / "picks.json").write_text("{}", encoding="utf-8")
    (tmp_path / "trades.csv").write_text("a,b\n", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "daily.md").write_text("# hi", encoding="utf-8")
    out = detect_outputs(tmp_path)
    assert any(p.endswith("signals.sqlite") for p in out["sqlite"])
    assert any(p.endswith("picks.json") for p in out["json"])
    assert any(p.endswith("trades.csv") for p in out["csv"])
    assert any(p.endswith("daily.md") for p in out["reports"])


def test_detect_outputs_skips_dot_dirs_and_caps(tmp_path):
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "junk.json").write_text("{}", encoding="utf-8")
    for i in range(9):
        (tmp_path / f"f{i}.json").write_text("{}", encoding="utf-8")
    out = detect_outputs(tmp_path)
    assert all(".venv" not in p for p in out["json"])
    assert len(out["json"]) <= 5


async def test_brief_reads_a_sqlite_positions_table(tmp_path):
    db = tmp_path / "signals.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE positions (symbol TEXT, shares REAL, pnl REAL)")
    con.executemany("INSERT INTO positions VALUES (?,?,?)",
                    [("NVDA", 10, 250.5), ("AAPL", 5, -12.0)])
    con.commit(); con.close()
    r = finance_registry(str(tmp_path))
    brief = await portfolio_brief(find_finance_project(r))
    assert brief["available"] is True
    assert {row["symbol"] for row in brief["rows"]} == {"NVDA", "AAPL"}
    assert "NVDA" in brief["spoken"]
    assert brief["caveat"]                       # never presented as advice


async def test_brief_falls_back_to_json(tmp_path):
    (tmp_path / "picks.json").write_text(
        json.dumps([{"symbol": "TSLA", "shares": 3}]), encoding="utf-8")
    r = finance_registry(str(tmp_path))
    brief = await portfolio_brief(find_finance_project(r))
    assert brief["available"] is True and brief["rows"][0]["symbol"] == "TSLA"


async def test_brief_is_honest_when_nothing_is_found(tmp_path):
    r = finance_registry(str(tmp_path))
    brief = await portfolio_brief(find_finance_project(r))
    assert brief["available"] is False
    assert "couldn't find" in brief["spoken"].lower()
    assert brief["rows"] == []


async def test_brief_with_no_confirmed_finance_project_is_honest():
    brief = await portfolio_brief(None)
    assert brief["available"] is False and brief["rows"] == []


def test_module_exposes_no_trading_capability():
    import server.finance as f
    banned = ("buy", "sell", "order", "trade", "execute", "submit", "transfer", "withdraw")
    offenders = [n for n in dir(f)
                 if not n.startswith("_") and callable(getattr(f, n))
                 and any(b in n.lower() for b in banned)]
    assert offenders == []          # spec §16: no execution path exists, by construction


def test_trade_refusal_line_points_at_the_stock_system():
    assert "buy" not in TRADE_REFUSAL.lower().split()[0]
    assert len(TRADE_REFUSAL) > 20
