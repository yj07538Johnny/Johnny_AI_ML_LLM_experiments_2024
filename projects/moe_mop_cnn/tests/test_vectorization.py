"""Tests for vectorization module using mock Word2Vec model."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from lib.vectorization import (
    get_word_vectors,
    combine_vectors_sum,
    get_phrase_vectors,
    get_ngram_vectors,
)
from lib.exclusion_list import ExclusionList


@pytest.fixture
def mock_model():
    """Create a mock Word2Vec model with a small vocabulary."""
    model = MagicMock()
    model.vector_size = 10
    model.wv.vector_size = 10

    vocab = {
        "the": np.ones(10) * 0.1,
        "quick": np.ones(10) * 0.2,
        "brown": np.ones(10) * 0.3,
        "fox": np.ones(10) * 0.4,
        "jumps": np.ones(10) * 0.5,
        "lazy": np.ones(10) * 0.6,
        "dog": np.ones(10) * 0.7,
        "secret": np.ones(10) * 0.8,
        "nsa": np.ones(10) * 0.9,
    }

    model.wv.__contains__ = lambda self, key: key in vocab
    model.wv.__getitem__ = lambda self, key: vocab[key]
    model.wv.__len__ = lambda self: len(vocab)

    return model


class TestGetWordVectors:
    def test_basic(self, mock_model):
        result = get_word_vectors(["the", "fox"], mock_model)
        assert result is not None
        assert result.shape == (2, 10)

    def test_unknown_tokens_skipped(self, mock_model):
        result = get_word_vectors(["the", "zzzzz"], mock_model)
        assert result is not None
        assert result.shape == (1, 10)

    def test_all_unknown(self, mock_model):
        result = get_word_vectors(["aaaa", "bbbb"], mock_model)
        assert result is None

    def test_empty_list(self, mock_model):
        result = get_word_vectors([], mock_model)
        assert result is None

    def test_with_exclusion_null_vector(self, mock_model):
        excl = ExclusionList(terms=["secret"], use_defaults=False)
        excl.mode = "null_vector"
        result = get_word_vectors(["the", "secret"], mock_model)
        assert result is not None

    def test_with_exclusion_omit(self, mock_model):
        excl = ExclusionList(terms=["the"], use_defaults=False)
        excl.mode = "omit"
        result = get_word_vectors(["the", "fox"], mock_model, exclusions=excl)
        # "the" excluded via omit, only "fox" remains
        assert result is not None
        assert result.shape == (1, 10)


class TestCombineVectorsSum:
    def test_basic_sum(self, mock_model):
        result = combine_vectors_sum(["the", "fox"], mock_model)
        expected = np.ones(10) * 0.1 + np.ones(10) * 0.4
        np.testing.assert_array_almost_equal(result, expected)

    def test_unknown_returns_zeros(self, mock_model):
        result = combine_vectors_sum(["zzzzz"], mock_model)
        np.testing.assert_array_equal(result, np.zeros(10))

    def test_with_exclusion_omit(self, mock_model):
        excl = ExclusionList(terms=["the"], use_defaults=False)
        excl.mode = "omit"
        result = combine_vectors_sum(["the", "fox"], mock_model, exclusions=excl)
        expected = np.ones(10) * 0.4  # Only fox
        np.testing.assert_array_almost_equal(result, expected)


class TestGetPhraseVectors:
    def test_list_of_phrases(self, mock_model):
        result = get_phrase_vectors(["the fox", "quick brown"], mock_model)
        assert len(result) == 2
        for vec in result:
            assert vec.shape == (10,)

    def test_single_string(self, mock_model):
        result = get_phrase_vectors("the fox", mock_model)
        assert len(result) == 1


class TestGetNgramVectors:
    def test_tuple_ngrams(self, mock_model):
        ngrams = [("the", "fox"), ("quick", "brown")]
        result = get_ngram_vectors(ngrams, mock_model)
        assert len(result) == 2

    def test_string_ngrams(self, mock_model):
        result = get_ngram_vectors(["the fox", "quick brown"], mock_model)
        assert len(result) == 2

    def test_single_string(self, mock_model):
        result = get_ngram_vectors("the fox", mock_model)
        assert len(result) == 1
