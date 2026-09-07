"""Tests for the rule-based fact extraction stub."""

from app.memory.extraction import extract_facts


def test_extracts_favorite_pattern():
    candidates = extract_facts("My favorite coffee order is an oat milk flat white.")
    assert len(candidates) == 1
    fact = candidates[0]
    assert fact.category == "preference"
    assert fact.key == "coffee_order"
    assert "oat milk flat white" in fact.value


def test_extracts_employer_pattern():
    candidates = extract_facts("I work at a small design studio downtown.")
    assert len(candidates) == 1
    assert candidates[0].category == "trait"
    assert candidates[0].key == "employer"
    assert "design studio" in candidates[0].value


def test_extracts_location_pattern():
    candidates = extract_facts("I live in Austin and love it here.")
    assert any(c.key == "location" and "Austin" in c.value for c in candidates)


def test_extracts_multiple_facts_from_one_chunk():
    text = "I work at a design studio. I live in Austin. My favorite season is autumn."
    candidates = extract_facts(text)
    keys = {c.key for c in candidates}
    assert "employer" in keys
    assert "location" in keys
    assert "season" in keys


def test_no_matches_returns_empty_list():
    candidates = extract_facts("Just a normal sentence with nothing extractable.")
    assert candidates == []


def test_all_candidates_have_low_placeholder_confidence():
    candidates = extract_facts("I live in Austin.")
    assert all(c.confidence == 0.3 for c in candidates)
