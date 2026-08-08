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
    # spoken is capped at 3 sentences, not full — spelled out so a regression in
    # cap_sentences cannot hide behind a self-referential assertion.
    assert out["spoken"] == (
        "You left off at [[Tibet Session 2]]. It replicated. The p-value was 0.024."
    )
    assert out["citations"] == ["Tibet Session 2"]


def test_parse_malformed_json_falls_back():
    raw = '{"spoken": "broken", "display": '  # invalid JSON
    out = parse_butler_output(raw)
    # parse_butler_output strips the reply first, so compare against raw.strip():
    # LLM replies routinely carry trailing newlines we do not want in `display`.
    assert out["display"] == raw.strip()
    assert out["spoken"] == '{"spoken": "broken", "display":'


def test_spoken_is_capped_even_when_model_supplies_it():
    raw = ('{"spoken": "One. Two. Three. Four. Five.", "display": "d", "citations": []}')
    assert parse_butler_output(raw)["spoken"] == "One. Two. Three."


def test_trailing_brace_after_object_still_parses():
    raw = '{"spoken": "Hi.", "display": "Yo [[A]]."} also see {1,2}'
    out = parse_butler_output(raw)
    assert out["spoken"] == "Hi."           # not the raw JSON read aloud
    assert out["citations"] == ["A"]


def test_leading_brace_before_object_still_parses():
    raw = 'Note that {} means empty: {"spoken": "Hi.", "display": "Yo."}'
    assert parse_butler_output(raw)["spoken"] == "Hi."


def test_citations_are_normalized_and_deduped():
    raw = ('{"spoken": "s", "display": "d", '
           '"citations": ["[[Tibet Session 2]]", "Tibet Session 2", "Note#Results"]}')
    assert parse_butler_output(raw)["citations"] == ["Tibet Session 2", "Note"]


def test_newline_bullets_are_capped():
    raw = "Open threads:\n- Tibet session 2\n- Alethic globe\n- resume audit"
    out = parse_butler_output(raw)
    assert out["display"] == raw
    assert out["spoken"].count("\n") == 0
    assert out["spoken"] == "Open threads: - Tibet session 2 - Alethic globe"


def test_empty_json_values_fall_back_to_plain_text():
    raw = '{"spoken": "", "display": ""}'
    out = parse_butler_output(raw)
    assert out["display"] == raw          # user sees something rather than silence


def test_non_string_input_does_not_raise():
    assert parse_butler_output(None)["display"] == ""
    assert parse_butler_output(123)["display"] == "123"
