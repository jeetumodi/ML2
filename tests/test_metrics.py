"""
Unit tests for evaluation metrics and numerical stability edge cases.
"""

import numpy as np
import pytest
from src.evaluation.metrics import compute_binary_metrics, compute_ece


def test_standard_binary_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["mcc"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["aupr"] == 1.0
    assert m["brier_score"] < 0.05


def test_edge_cases_handling():
    # All negatives
    y_true_all_neg = np.zeros(20, dtype=int)
    y_prob = np.random.rand(20)
    m1 = compute_binary_metrics(y_true_all_neg, y_prob)
    assert np.isnan(m1["roc_auc"])
    assert m1["n_pos"] == 0

    # All positives
    y_true_all_pos = np.ones(20, dtype=int)
    m2 = compute_binary_metrics(y_true_all_pos, y_prob)
    assert np.isnan(m2["roc_auc"])
    assert m2["n_neg"] == 0

    # Constant predictions
    y_true = np.array([0, 1, 0, 1])
    y_const = np.array([0.5, 0.5, 0.5, 0.5])
    m3 = compute_binary_metrics(y_true, y_const)
    assert not np.isnan(m3["brier_score"])
    assert m3["brier_score"] == 0.25


def test_ece_computation():
    y_true = np.array([0, 1, 0, 1])
    # Perfectly calibrated predictions:
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    ece, mce = compute_ece(y_true, y_prob, n_bins=2)
    assert ece == 0.0
