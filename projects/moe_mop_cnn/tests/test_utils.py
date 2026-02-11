"""Tests for utils module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from lib.utils import apply_with_progress


class TestApplyWithProgress:
    def test_basic_function(self):
        series = pd.Series([1, 2, 3, 4])
        result = apply_with_progress(series, lambda x: x * 2, desc="test")
        assert list(result) == [2, 4, 6, 8]

    def test_preserves_index(self):
        series = pd.Series([10, 20], index=[5, 10])
        result = apply_with_progress(series, lambda x: x + 1, desc="test")
        assert list(result.index) == [5, 10]

    def test_with_kwargs(self):
        series = pd.Series(["hello", "world"])
        result = apply_with_progress(
            series, lambda x, suffix="": x + suffix, desc="test", suffix="!"
        )
        assert list(result) == ["hello!", "world!"]

    def test_empty_series(self):
        series = pd.Series([], dtype=object)
        result = apply_with_progress(series, lambda x: x, desc="test")
        assert len(result) == 0

    def test_returns_series(self):
        series = pd.Series([1])
        result = apply_with_progress(series, lambda x: x, desc="test")
        assert isinstance(result, pd.Series)
