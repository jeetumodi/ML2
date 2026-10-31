"""
Unit tests for candidate prioritization and ranking strategies.
"""

import numpy as np
import pytest
from src.prioritization.ranking import (
    rank_residues_strategy_a,
    rank_residues_strategy_b,
    rank_residues_strategy_c,
    select_top_k_candidates
)
from src.prioritization.ranking_metrics import (
    compute_protein_precision_recall_at_k,
    evaluate_screening_budget_across_proteins
)


def test_prioritization_ranking_strategies():
    probs = np.array([0.9, 0.85, 0.4, 0.1])
    unc = np.array([0.4, 0.05, 0.1, 0.02]) # index 0 is high uncertainty

    # Strategy A
    scores_a = rank_residues_strategy_a(probs)
    top_a = select_top_k_candidates(scores_a, k=2)
    assert np.array_equal(top_a, [0, 1])

    # Strategy B (LCB with lambda=1.0)
    scores_b = rank_residues_strategy_b(probs, unc, lambda_param=1.0)
    # scores_b: [0.5, 0.8, 0.3, 0.08] -> index 1 (low unc) becomes #1 rank!
    top_b = select_top_k_candidates(scores_b, k=2)
    assert np.array_equal(top_b, [1, 0])

    # Strategy C (Filter out unc > 0.3)
    scores_c = rank_residues_strategy_c(probs, unc, uncertainty_threshold=0.3)
    # index 0 filtered to -1.0
    top_c = select_top_k_candidates(scores_c, k=2)
    assert top_c[0] == 1


def test_precision_recall_at_k():
    y_true = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    p_at_2, r_at_2, n_pos = compute_protein_precision_recall_at_k(y_true, scores, k=2)
    assert p_at_2 == 1.0
    assert r_at_2 == 1.0
    assert n_pos == 2

    p_at_4, r_at_4, _ = compute_protein_precision_recall_at_k(y_true, scores, k=4)
    assert p_at_4 == 0.5
    assert r_at_4 == 1.0
