"""Tests for cnn_classifier module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
from lib.cnn_classifier import (
    prepare_data,
    create_model,
    train_model,
    get_feature_importance,
)


class TestPrepareData:
    def test_basic_shape(self):
        tokens = [["a", "b", "c"], ["d", "e"]]
        vectors = [
            [np.ones(50), np.ones(50), np.ones(50)],
            [np.ones(50), np.ones(50)],
        ]
        labels = [0, 1]

        X, y, tok = prepare_data(tokens, vectors, labels, max_len=10, vector_dim=50)
        assert X.shape == (2, 10, 50)
        assert y.shape == (2,)

    def test_padding(self):
        tokens = [["a"]]
        vectors = [[np.ones(20)]]
        labels = [0]

        X, y, tok = prepare_data(tokens, vectors, labels, max_len=5, vector_dim=20)
        assert X.shape == (1, 5, 20)
        # Last 4 rows should be zero-padded
        np.testing.assert_array_equal(X[0, 1:, :].numpy(), np.zeros((4, 20)))

    def test_truncation(self):
        tokens = [["a", "b", "c", "d", "e"]]
        vectors = [[np.ones(10)] * 5]
        labels = [0]

        X, y, tok = prepare_data(tokens, vectors, labels, max_len=3, vector_dim=10)
        assert X.shape == (1, 3, 10)

    def test_empty_vectors(self):
        tokens = [[]]
        vectors = [[]]
        labels = [0]

        X, y, tok = prepare_data(tokens, vectors, labels, max_len=5, vector_dim=10)
        assert X.shape == (1, 5, 10)
        np.testing.assert_array_equal(X[0].numpy(), np.zeros((5, 10)))

    def test_none_vectors(self):
        tokens = [None]
        vectors = [None]
        labels = [0]

        X, y, tok = prepare_data(tokens, vectors, labels, max_len=5, vector_dim=10)
        assert X.shape == (1, 5, 10)


class TestCreateModel:
    def test_dimensions(self):
        conv, fc = create_model(embedding_dim=50, num_classes=3)
        assert conv.in_channels == 50
        assert conv.out_channels == 100
        assert fc.in_features == 100
        assert fc.out_features == 3

    def test_kernel_size(self):
        conv, fc = create_model(embedding_dim=50, num_classes=2)
        assert conv.kernel_size == (3,)


class TestTrainModel:
    def test_smoke_test(self):
        """Basic training should run without errors."""
        X = torch.randn(8, 10, 50)  # 8 samples, 10 tokens, 50-dim
        y = torch.randint(0, 2, (8,))

        conv, fc = create_model(embedding_dim=50, num_classes=2)
        train_model(X, y, conv, fc, num_classes=2, epochs=2, batch_size=4, device="cpu")

    def test_loss_decreases(self, capsys):
        """Loss should generally decrease over epochs."""
        torch.manual_seed(42)
        X = torch.randn(16, 10, 50)
        y = torch.randint(0, 2, (16,))

        conv, fc = create_model(embedding_dim=50, num_classes=2)
        train_model(X, y, conv, fc, num_classes=2, epochs=5, batch_size=8, device="cpu")

        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().split("\n") if "Loss:" in l]
        assert len(lines) == 5


class TestGetFeatureImportance:
    def test_basic(self):
        conv, fc = create_model(embedding_dim=50, num_classes=2)
        x_input = torch.randn(10, 50)

        pred_class, importance = get_feature_importance(x_input, conv, fc)
        assert isinstance(pred_class, int)
        assert pred_class in [0, 1]

    def test_with_tokens(self):
        conv, fc = create_model(embedding_dim=50, num_classes=2)
        x_input = torch.randn(10, 50)
        tokens = [f"token_{i}" for i in range(10)]

        pred_class, token_scores = get_feature_importance(
            x_input, conv, fc, tokens=tokens
        )
        assert isinstance(token_scores, list)
        assert len(token_scores) <= 10
        # Each entry is (token, score)
        for tok, score in token_scores:
            assert isinstance(tok, str)
            assert isinstance(score, (float, np.floating))

    def test_sorted_by_importance(self):
        conv, fc = create_model(embedding_dim=50, num_classes=2)
        x_input = torch.randn(10, 50)
        tokens = [f"token_{i}" for i in range(10)]

        _, token_scores = get_feature_importance(
            x_input, conv, fc, tokens=tokens
        )
        scores = [s for _, s in token_scores]
        assert scores == sorted(scores, reverse=True)

    def test_max_tokens_limit(self):
        conv, fc = create_model(embedding_dim=50, num_classes=2)
        x_input = torch.randn(20, 50)
        tokens = [f"token_{i}" for i in range(20)]

        _, token_scores = get_feature_importance(
            x_input, conv, fc, max_tokens=5, tokens=tokens
        )
        assert len(token_scores) <= 5
