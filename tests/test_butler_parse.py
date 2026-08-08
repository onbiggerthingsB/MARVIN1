from server.butler_parse import cap_sentences, extract_wikilinks, parse_butler_output


def test_extract_wikilinks_dedupes_and_strips_alias():
    text = "See [[Tibet Session 2]] and [[Alethic|the globe]] and [[Tibet Session 2]]."
    assert extract_wikilinks(text) == ["Tibet Session 2", "Alethic"]


def test_cap_sentences():
    assert cap_sentences("One. Two! Three? Four.", 3) == "One. Two! Three?"
    assert cap_sentences("Only one sentence here", 3) == "Only one sentence here"
    assert cap_sentences("", 3) == ""


def test_parse_json_happy_path():
    raw = ('{"spoken": "You left off at session 2.", '
           '"display": "You left off at [[Tibet Session 2]] on 2026-07-28.", '
           '"citations": ["Tibet Session 2"]}')
    out = parse_butler_output(raw)
    assert out["spoken"] == "You left off at session 2."
    assert "[[Tibet Session 2]]" in out["display"]
    assert out["citations"] == ["Tibet Session 2"]


def test_parse_json_embedded_in_prose():
    raw = 'Sure! {"spoken": "Hi.", "display": "Hi there [[Note A]]."} thanks'
    out = parse_butler_output(raw)
    assert out["spoken"] == "Hi."
    assert out["citations"] == ["Note A"]  # derived from display when absent


def test_parse_fallback_plain_text():
    raw = "You left off at [[Tibet Session 2]]. It replicated. The p-value was 0.024. Extra tail."
    out = parse_butler_output(raw)
    assert out["display"] == raw                       # full text preserved
    assert out["spoken"] == cap_sentences(raw, 3)      # spoken is capped, not full
    assert out["citations"] == ["Tibet Session 2"]


def test_parse_malformed_json_falls_back():
    raw = '{"spoken": "broken", "display": '  # invalid JSON
    out = parse_butler_output(raw)
    # parse_butler_output strips the reply first, so compare against raw.strip():
    # LLM replies routinely carry trailing newlines we do not want in `display`.
    assert out["display"] == raw.strip() and out["spoken"] == cap_sentences(raw, 3)
