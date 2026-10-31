"""
Uncertainty-Guided Interface Candidate Prioritization Strategies.

Implements:
- Strategy A: Probability-Only Ranking (score_i = p_i)
- Strategy B: Lower-Confidence-Bound (LCB) Ranking (score_i = p_i - lambda * uncertainty_i)
- Strategy C: Uncertainty Filtering (deprioritize candidates exceeding uncertainty threshold)
- Strategy D: Budget-Constrained Screening (Top-K per protein: Top-5, Top-10, Top-20)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def rank_residues_strategy_a(probs: np.ndarray) -> np.ndarray:
    """
    Strategy A: Probability-Only Ranking.
    Ranks residues in descending order of predicted interface probability.
    """
    return np.asarray(probs, dtype=float)


def rank_residues_strategy_b(
    probs: np.ndarray,
    uncertainty: np.ndarray,
    lambda_param: float = 1.0
) -> np.ndarray:
    """
    Strategy B: Lower-Confidence-Bound (LCB) Ranking.
    Penalizes candidates with high epistemic/predictive uncertainty.
    
    Formula:
        score_i = p_i - lambda * uncertainty_i
    """
    probs = np.asarray(probs, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    return probs - float(lambda_param) * uncertainty


def rank_residues_strategy_c(
    probs: np.ndarray,
    uncertainty: np.ndarray,
    uncertainty_threshold: float
) -> np.ndarray:
    """
    Strategy C: Uncertainty Filtering.
    Penalizes residues exceeding the uncertainty cutoff by assigning them a baseline minimum score.
    """
    probs = np.asarray(probs, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    filtered_scores = probs.copy()
    mask_high_unc = uncertainty > uncertainty_threshold
    filtered_scores[mask_high_unc] = -1.0  # demote to bottom of list
    return filtered_scores


def select_top_k_candidates(
    scores: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Selects top K candidate residue indices given scores.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = min(k, n)
    # Descending sort
    sorted_indices = np.argsort(-scores)
    return sorted_indices[:k]


def tune_lcb_lambda_on_validation(
    val_probs: List[np.ndarray],
    val_uncs: List[np.ndarray],
    val_trues: List[np.ndarray],
    lambda_candidates: List[float] = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
    target_k: int = 10
) -> Tuple[float, Dict[float, float]]:
    """
    Finds the optimal lambda parameter on validation/calibration data only (never test data)
    by maximizing Mean Precision@K across validation proteins.
    """
    best_lambda = lambda_candidates[0]
    best_mean_prec = -1.0
    results_by_lambda = {}

    for lam in lambda_candidates:
        prec_list = []
        for p, u, y in zip(val_probs, val_uncs, val_trues):
            scores = rank_residues_strategy_b(p, u, lambda_param=lam)
            k_eff = min(target_k, len(p))
            top_idx = select_top_k_candidates(scores, k=k_eff)
            if k_eff > 0:
                p_at_k = np.sum(y[top_idx] == 1) / k_eff
                prec_list.append(p_at_k)

        mean_prec = float(np.mean(prec_list)) if prec_list else 0.0
        results_by_lambda[lam] = mean_prec

        if mean_prec > best_mean_prec:
            best_mean_prec = mean_prec
            best_lambda = lam

    return best_lambda, results_by_lambda
