"""Tests for exclusion_list module."""
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lib.exclusion_list import ExclusionList, DEFAULT_EXCLUSIONS


class TestExclusionListInit:
    def test_default_terms_loaded(self):
        el = ExclusionList()
        assert len(el) > 0
        assert "nsa" in el.terms

    def test_no_defaults(self):
        el = ExclusionList(use_defaults=False)
        assert len(el) == 0

    def test_custom_terms(self):
        el = ExclusionList(terms=["foo", "bar"], use_defaults=False)
        assert len(el) == 2
        assert "foo" in el.terms

    def test_default_mode(self):
        el = ExclusionList()
        assert el.mode == "null_vector"


class TestAddRemove:
    def test_add(self):
        el = ExclusionList(use_defaults=False)
        el.add("test term")
        assert "test term" in el.terms

    def test_add_case_insensitive(self):
        el = ExclusionList(use_defaults=False)
        el.add("TEST TERM")
        assert "test term" in el.terms

    def test_add_strips_whitespace(self):
        el = ExclusionList(use_defaults=False)
        el.add("  padded  ")
        assert "padded" in el.terms

    def test_remove(self):
        el = ExclusionList(terms=["foo"], use_defaults=False)
        el.remove("foo")
        assert "foo" not in el.terms

    def test_remove_nonexistent(self):
        el = ExclusionList(use_defaults=False)
        el.remove("nonexistent")  # Should not raise


class TestIsExcluded:
    def test_exact_match(self):
        el = ExclusionList(terms=["secret"], use_defaults=False)
        assert el.is_excluded("secret") is True

    def test_case_insensitive_match(self):
        el = ExclusionList(terms=["secret"], use_defaults=False)
        assert el.is_excluded("SECRET") is True

    def test_substring_match(self):
        el = ExclusionList(terms=["secret"], use_defaults=False)
        assert el.is_excluded("top secret document") is True

    def test_no_match(self):
        el = ExclusionList(terms=["secret"], use_defaults=False)
        assert el.is_excluded("public") is False

    def test_pattern_match(self):
        el = ExclusionList(use_defaults=False)
        el.add_pattern(r"classification:\s*(secret|top secret)")
        assert el.is_excluded("classification: secret") is True
        assert el.is_excluded("no match here") is False


class TestFilterTokens:
    def test_filters_excluded(self):
        el = ExclusionList(terms=["bad"], use_defaults=False)
        result = el.filter_tokens(["good", "bad", "ok"])
        assert result == ["good", "ok"]

    def test_empty_list(self):
        el = ExclusionList(use_defaults=False)
        assert el.filter_tokens([]) == []


class TestListTerms:
    def test_sorted(self):
        el = ExclusionList(terms=["z", "a", "m"], use_defaults=False)
        assert el.list_terms() == ["a", "m", "z"]


class TestSaveLoad:
    def test_round_trip(self):
        el = ExclusionList(terms=["alpha", "beta"], use_defaults=False)
        el.mode = "omit"
        el.add_pattern(r"test_\d+")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path = f.name

        el.save(path)
        loaded = ExclusionList.load(path)

        assert loaded.terms == el.terms
        assert loaded.mode == "omit"
        assert len(loaded.patterns) == 1

        Path(path).unlink()

    def test_load_preserves_data(self):
        data = {"mode": "null_vector", "terms": ["x", "y"], "patterns": []}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            json.dump(data, f)
            path = f.name

        loaded = ExclusionList.load(path)
        assert "x" in loaded.terms
        assert "y" in loaded.terms
        assert loaded.mode == "null_vector"

        Path(path).unlink()


class TestRepr:
    def test_repr(self):
        el = ExclusionList(terms=["a", "b"], use_defaults=False)
        r = repr(el)
        assert "n_terms=2" in r
        assert "null_vector" in r


class TestDefaultExclusions:
    def test_defaults_exist(self):
        assert len(DEFAULT_EXCLUSIONS) > 0
        assert "nsa" in DEFAULT_EXCLUSIONS
