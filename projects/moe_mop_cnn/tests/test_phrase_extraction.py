"""Tests for phrase_extraction module."""
import sys
from pathlib import Path

# Add lib to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lib.phrase_extraction import (
    extract_noun_phrases,
    extract_verb_phrases,
    extract_adjective_phrases,
    extract_adverb_phrases,
    extract_prepositional_phrases,
    extract_gerund_phrases,
    extract_infinitive_phrases,
    extract_participle_phrases,
    extract_appositive_phrases,
    extract_all_phrases,
    _extract_phrases,
    PHRASE_GRAMMARS,
    PHRASE_TYPES,
)


SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog near the old barn."


class TestPhraseGrammars:
    def test_all_phrase_types_registered(self):
        assert len(PHRASE_GRAMMARS) == 9
        assert set(PHRASE_TYPES) == set(PHRASE_GRAMMARS.keys())

    def test_grammar_structure(self):
        for name, (label, grammar) in PHRASE_GRAMMARS.items():
            assert isinstance(label, str)
            assert isinstance(grammar, str)
            assert label in grammar


class TestExtractPhrases:
    def test_empty_string(self):
        assert _extract_phrases("", "NP", r"NP: {<DT>?<JJ>*<NN.*>+}") == []

    def test_none_input(self):
        assert _extract_phrases(None, "NP", r"NP: {<DT>?<JJ>*<NN.*>+}") == []

    def test_non_string_input(self):
        assert _extract_phrases(123, "NP", r"NP: {<DT>?<JJ>*<NN.*>+}") == []

    def test_returns_list_of_strings(self):
        result = _extract_phrases(SAMPLE_TEXT, "NP", r"NP: {<DT>?<JJ>*<NN.*>+}")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)


class TestNounPhrases:
    def test_basic_extraction(self):
        result = extract_noun_phrases(SAMPLE_TEXT)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_detects_noun_phrases(self):
        result = extract_noun_phrases("The big red car drove fast.")
        phrase_text = " ".join(result)
        assert "car" in phrase_text.lower()


class TestVerbPhrases:
    def test_basic_extraction(self):
        result = extract_verb_phrases("She was running quickly toward the finish line.")
        assert isinstance(result, list)


class TestAdjectivePhrases:
    def test_basic_extraction(self):
        result = extract_adjective_phrases("The very tall and strong man arrived.")
        assert isinstance(result, list)


class TestAdverbPhrases:
    def test_basic_extraction(self):
        result = extract_adverb_phrases("He ran very quickly and quite smoothly.")
        assert isinstance(result, list)


class TestPrepositionalPhrases:
    def test_basic_extraction(self):
        result = extract_prepositional_phrases("The cat sat on the mat.")
        assert isinstance(result, list)


class TestGerundPhrases:
    def test_basic_extraction(self):
        result = extract_gerund_phrases("Running marathons is good exercise.")
        assert isinstance(result, list)


class TestInfinitivePhrases:
    def test_basic_extraction(self):
        result = extract_infinitive_phrases("She wanted to run faster.")
        assert isinstance(result, list)
        assert len(result) > 0


class TestParticiplePhrases:
    def test_basic_extraction(self):
        result = extract_participle_phrases("The running water was cold.")
        assert isinstance(result, list)


class TestAppositivePhrases:
    def test_basic_extraction(self):
        result = extract_appositive_phrases("President Biden spoke today.")
        assert isinstance(result, list)


class TestExtractAllPhrases:
    def test_returns_dict(self):
        result = extract_all_phrases(SAMPLE_TEXT)
        assert isinstance(result, dict)

    def test_all_phrase_types_present(self):
        result = extract_all_phrases(SAMPLE_TEXT)
        for phrase_type in PHRASE_TYPES:
            assert phrase_type in result

    def test_values_are_lists(self):
        result = extract_all_phrases(SAMPLE_TEXT)
        for key, value in result.items():
            assert isinstance(value, list)

    def test_empty_input(self):
        result = extract_all_phrases("")
        assert isinstance(result, dict)
        for value in result.values():
            assert value == []
