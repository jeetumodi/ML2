"""
Evaluation Metrics for Interface Candidate Prioritization and Screening.

Implements:
- Precision@K
- Recall@K
- Mean per-protein Precision@K and Recall@K
- Cumulative Recall versus Screening Budget
"""

import numpy as np
from typing import Dict, Any, List, Tuple


def compute_protein_precision_recall_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int
) -> Tuple[float, float, int]:
    """
    Computes Precision@K and Recall@K for a single protein.
    
    Returns:
        (precision_at_k, recall_at_k, n_pos)
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    n_pos = int(np.sum(y_true == 1))
    n = len(y_true)

    if n == 0:
        return 0.0, 0.0, 0

    k_eff = min(k, n)
    top_indices = np.argsort(-scores)[:k_eff]
    tp = int(np.sum(y_true[top_indices] == 1))

    prec_at_k = float(tp / k_eff) if k_eff > 0 else 0.0
    rec_at_k = float(tp / n_pos) if n_pos > 0 else 0.0

    return prec_at_k, rec_at_k, n_pos


def evaluate_screening_budget_across_proteins(
    y_trues: List[np.ndarray],
    scores_list: List[np.ndarray],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, Any]:
    """
    Evaluates prioritization performance across an entire protein set at multiple screening budgets K.
    """
    results = {}

    for k in k_values:
        precisions = []
        recalls = []
        total_tp = 0
        total_screened = 0
        total_positives = 0

        for y_t, s in zip(y_trues, scores_list):
            p_k, r_k, n_pos = compute_protein_precision_recall_at_k(y_t, s, k=k)
            precisions.append(p_k)
            recalls.append(r_k)

            k_eff = min(k, len(y_t))
            top_idx = np.argsort(-s)[:k_eff]
            tp = int(np.sum(y_t[top_idx] == 1))

            total_tp += tp
            total_screened += k_eff
            total_positives += n_pos

        results[f"mean_precision_at_{k}"] = float(np.mean(precisions))
        results[f"std_precision_at_{k}"] = float(np.std(precisions))
        results[f"mean_recall_at_{k}"] = float(np.mean(recalls))
        results[f"std_recall_at_{k}"] = float(np.std(recalls))
        results[f"micro_precision_at_{k}"] = float(total_tp / total_screened) if total_screened > 0 else 0.0
        results[f"micro_recall_at_{k}"] = float(total_tp / total_positives) if total_positives > 0 else 0.0

    return results


def compute_cumulative_recall_curve(
    y_trues: List[np.ndarray],
    scores_list: List[np.ndarray],
    max_k: int = 40
) -> Dict[str, np.ndarray]:
    """
    Computes mean cumulative recall across proteins as budget K scales from 1 to max_k.
    """
    budgets = np.arange(1, max_k + 1)
    mean_recalls = []
    mean_precisions = []

    for k in budgets:
        r_list = []
        p_list = []
        for y_t, s in zip(y_trues, scores_list):
            p_k, r_k, _ = compute_protein_precision_recall_at_k(y_t, s, k=int(k))
            r_list.append(r_k)
            p_list.append(p_k)
        mean_recalls.append(float(np.mean(r_list)))
        mean_precisions.append(float(np.mean(p_list)))

    return {
        "budgets": budgets,
        "mean_recall": np.array(mean_recalls),
        "mean_precision": np.array(mean_precisions)
    }
