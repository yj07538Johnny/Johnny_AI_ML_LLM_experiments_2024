"""Tests for ngram_extraction module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lib.ngram_extraction import (
    get_bigrams,
    get_ngrams,
    get_all_ngrams,
    NGRAM_SIZES,
)


SAMPLE_TEXT = "The quick brown fox jumps. The lazy dog sleeps."


class TestGetBigrams:
    def test_basic(self):
        result = get_bigrams(SAMPLE_TEXT)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_returns_tuples_of_length_2(self):
        result = get_bigrams(SAMPLE_TEXT)
        for bigram in result:
            assert isinstance(bigram, tuple)
            assert len(bigram) == 2

    def test_empty_string(self):
        assert get_bigrams("") == []

    def test_none_input(self):
        assert get_bigrams(None) == []

    def test_non_string(self):
        assert get_bigrams(42) == []

    def test_single_word(self):
        assert get_bigrams("Hello") == []


class TestGetNgrams:
    def test_trigrams(self):
        result = get_ngrams(SAMPLE_TEXT, 3)
        assert all(len(ng) == 3 for ng in result)

    def test_respects_sentence_boundaries(self):
        # Use clearly distinct sentences so NLTK splits reliably
        text = "The cat is sleeping on the mat. However, the dog ran outside quickly."
        bigrams = get_ngrams(text, 2)
        bigram_strs = [" ".join(b) for b in bigrams]
        # "mat" and "However" should never appear in the same bigram
        for b in bigram_strs:
            assert not ("mat" in b and "However" in b), f"Cross-sentence bigram found: {b}"

    def test_sentence_too_short(self):
        result = get_ngrams("Hi.", 5)
        assert result == []

    def test_large_n(self):
        result = get_ngrams("Short.", 100)
        assert result == []

    def test_whitespace_only(self):
        assert get_ngrams("   ", 2) == []


class TestGetAllNgrams:
    def test_returns_dict(self):
        result = get_all_ngrams(SAMPLE_TEXT)
        assert isinstance(result, dict)

    def test_bigram_key(self):
        result = get_all_ngrams(SAMPLE_TEXT, sizes=[2, 3])
        assert "bigrams" in result
        assert "3-grams" in result

    def test_custom_sizes(self):
        result = get_all_ngrams(SAMPLE_TEXT, sizes=[4, 5])
        assert "4-grams" in result
        assert "5-grams" in result

    def test_default_sizes(self):
        assert len(NGRAM_SIZES) == 18
        assert 2 in NGRAM_SIZES
        assert 100 in NGRAM_SIZES
