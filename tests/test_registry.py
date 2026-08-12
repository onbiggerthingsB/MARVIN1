import json
from pathlib import Path

import pytest

from server.discovery import Candidate
from server.registry import Project, Registry


def cand(path, name, **kw):
    return Candidate(path=path, name=name, sources=kw.pop("sources", ["claude.json"]), **kw)


def test_merge_adds_new_projects_unconfirmed(tmp_path):
    r = Registry()
    added = r.merge_candidates([cand("/a/soccer", "soccer"), cand("/b/quant agent", "quant agent")])
    assert {p.name for p in added} == {"soccer", "quant agent"}
    assert all(p.confirmed is False for p in r.projects)   # discovery never auto-confirms


def test_merge_never_clobbers_a_human_confirmation(tmp_path):
    r = Registry()
    r.merge_candidates([cand("/a/soccer", "soccer")])
    r.confirm("soccer")
    r.add_alias("soccer", "the soccer app")
    added = r.merge_candidates([cand("/a/soccer", "soccer")])   # rediscovered later
    assert added == []                                          # nothing new
    p = r.match("the soccer app")[0]
    assert p.confirmed is True and "the soccer app" in p.aliases


def test_roundtrip_persists_confirmations(tmp_path):
    f = tmp_path / "projects.json"
    r = Registry()
    r.merge_candidates([cand("/a/alethic", "alethic")])
    r.confirm("alethic")
    r.add_mishearing("alethic", "athletic")
    r.save(f)
    assert json.loads(f.read_text())["schema_version"] == 1
    r2 = Registry.load(f)
    assert r2.match("athletic")[0].name == "alethic"


def test_load_rejects_a_future_schema(tmp_path):
    f = tmp_path / "projects.json"
    f.write_text(json.dumps({"schema_version": 99, "projects": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        Registry.load_strict(f)


def test_load_quarantines_a_corrupt_file_instead_of_destroying_it(tmp_path):
    f = tmp_path / "projects.json"
    f.write_text("{not json", encoding="utf-8")
    r = Registry.load(f)
    assert r.projects == []
    quarantined = list(tmp_path.glob("projects.json.corrupt-*"))
    assert len(quarantined) == 1                 # the human's data was preserved
    assert quarantined[0].read_text(encoding="utf-8") == "{not json"


def test_load_quarantines_a_future_schema_rather_than_refusing_to_boot(tmp_path):
    f = tmp_path / "projects.json"
    f.write_text(json.dumps({"schema_version": 99, "projects": []}), encoding="utf-8")
    assert Registry.load(f).projects == []
    assert list(tmp_path.glob("projects.json.corrupt-*"))


def test_load_missing_file_gives_empty_registry(tmp_path):
    assert Registry.load(tmp_path / "nope.json").projects == []


def test_match_only_returns_confirmed_projects():
    r = Registry()
    r.merge_candidates([cand("/a/soccer", "soccer")])
    assert r.match("soccer") == []          # discovered but not yet confirmed
    r.confirm("soccer")
    assert r.match("soccer")[0].name == "soccer"


def test_match_is_case_and_fuzzy_tolerant():
    r = Registry()
    r.merge_candidates([cand("/a/quant agent", "quant agent")])
    r.confirm("quant agent")
    assert r.match("QUANT AGENT")[0].name == "quant agent"
    assert r.match("quant agnt")[0].name == "quant agent"    # one transposition
    assert r.match("completely unrelated") == []


def test_match_returns_all_candidates_when_ambiguous():
    r = Registry()
    r.merge_candidates([cand("/a/composed", "composed"), cand("/b/composed", "composed")])
    r.confirm("composed")
    for p in r.projects:
        p.confirmed = True
    assert len(r.match("composed")) == 2      # caller must disambiguate


def test_confirm_records_kind():
    r = Registry()
    r.merge_candidates([cand("/a/quant agent", "quant agent")])
    p = r.confirm("quant agent", kind="finance")
    assert p.kind == "finance"
    assert r.pending() == []


def test_match_ignores_short_filler_queries():
    r = Registry()
    r.merge_candidates([cand("/a/soccer", "soccer")])
    r.confirm("soccer")
    assert r.match("so") == []        # a two-letter filler must not route work
    assert r.match("s") == []


def test_match_still_finds_a_name_inside_a_longer_phrase():
    r = Registry()
    r.merge_candidates([cand("/a/soccer", "soccer")])
    r.confirm("soccer")
    assert r.match("pull up the soccer app")[0].name == "soccer"


def test_reconfirming_does_not_downgrade_a_finance_project():
    r = Registry()
    r.merge_candidates([cand("/a/quant agent", "quant agent")])
    r.confirm("quant agent", kind="finance")
    r.confirm("quant agent")                 # idempotent re-confirm, no kind passed
    assert r.match("quant agent")[0].kind == "finance"


def test_duplicate_basenames_are_confirmable_by_path():
    r = Registry()
    r.merge_candidates([cand("/one/marvin", "marvin"), cand("/two/marvin", "marvin")])
    r.confirm_path("/two/marvin")
    confirmed = [p for p in r.projects if p.confirmed]
    assert [p.path for p in confirmed] == ["/two/marvin"]   # the OTHER one, precisely
    assert r.match("marvin")[0].path == "/two/marvin"


def test_load_tolerates_valid_json_of_the_wrong_shape(tmp_path):
    null_file = tmp_path / "null.json"
    null_file.write_text("null", encoding="utf-8")
    assert Registry.load(null_file).projects == []

    list_file = tmp_path / "list.json"
    list_file.write_text("[1,2,3]", encoding="utf-8")
    assert Registry.load(list_file).projects == []


def test_roundtrip_persists_data_source(tmp_path):
    f = tmp_path / "projects.json"
    r = Registry()
    r.merge_candidates([cand("/p/quant agent", "quant agent")])
    r.confirm("quant agent", kind="finance")
    r.set_data_source("/p/quant agent", "/p/quant agent/signals.sqlite")
    r.save(f)
    p = Registry.load(f).projects[0]
    assert p.data_source == "/p/quant agent/signals.sqlite"
