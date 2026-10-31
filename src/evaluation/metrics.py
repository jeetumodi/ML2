"""
Evaluation Metrics for Protein-Protein Interaction Site (PPIS) Prediction.

Implements standard biological residue-level and protein-level classification metrics,
calibration metrics, ranking metrics, and handles edge cases (e.g., zero positives/negatives,
constant predictions, NaN values).
"""

import numpy as np
from sklearn import metrics
from typing import Dict, Any, Tuple, List, Optional


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive residue-level binary classification metrics.

    Args:
        y_true: 1D array of ground truth binary labels (0 or 1).
        y_prob: 1D array of predicted probabilities in [0, 1].
        threshold: Classification threshold. If None, computes optimal threshold via F-max.

    Returns:
        Dictionary of computed metrics.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    # Clip probabilities for numerical stability
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)

    n_samples = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_samples == 0:
        return {"error": "Empty input array"}

    # ROC-AUC and AUPR (Average Precision)
    if n_pos > 0 and n_neg > 0:
        roc_auc = float(metrics.roc_auc_score(y_true, y_prob))
        precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_prob)
        aupr = float(metrics.auc(recall_curve, precision_curve))
    else:
        roc_auc = float("nan")
        aupr = float("nan")
        precision_curve = np.array([])
        recall_curve = np.array([])

    # Brier score
    brier_score = float(np.mean((y_prob - y_true) ** 2))

    # Binary Cross Entropy / NLL
    nll = float(-np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1.0 - y_prob_clipped)))

    # Compute optimal threshold if not provided
    if threshold is None:
        best_f1 = -1.0
        best_thresh = 0.5
        best_p = 0.0
        best_r = 0.0
        best_mcc = 0.0

        for t in np.linspace(0.01, 0.99, 99):
            pred_t = (y_prob >= t).astype(int)
            tp = int(np.sum((y_true == 1) & (pred_t == 1)))
            fp = int(np.sum((y_true == 0) & (pred_t == 1)))
            fn = int(np.sum((y_true == 1) & (pred_t == 0)))
            tn = int(np.sum((y_true == 0) & (pred_t == 0)))

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            mcc_val = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)
                best_p = p
                best_r = r
                best_mcc = mcc_val

        threshold = best_thresh
        f1_score = best_f1
        precision = best_p
        recall = best_r
        mcc = best_mcc
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = (y_prob >= threshold).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = float(((tp * tn) - (fp * fn)) / denom) if denom > 0 else 0.0

    accuracy = float(np.mean(y_pred == y_true))

    # Confusion matrix elements
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    # Expected Calibration Error (ECE)
    ece, mce = compute_ece(y_true, y_prob, n_bins=10)

    return {
        "n_samples": n_samples,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": float(n_pos / n_samples) if n_samples > 0 else 0.0,
        "roc_auc": roc_auc,
        "aupr": aupr,
        "f1": f1_score,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "accuracy": accuracy,
        "brier_score": brier_score,
        "nll": nll,
        "ece": ece,
        "mce": mce,
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted probability of the positive class.
        n_bins: Number of equal-width bins in [0, 1].

    Returns:
        (ece, mce) tuple of floats.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)

    if total_samples == 0:
        return 0.0, 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            gap = np.abs(bin_acc - bin_conf)
            ece += (bin_size / total_samples) * gap
            mce = max(mce, gap)

    return float(ece), float(mce)


def compute_per_protein_metrics(
    protein_ids: List[str],
    y_trues: List[np.ndarray],
    y_probs: List[np.ndarray],
    threshold: float = 0.33
) -> List[Dict[str, Any]]:
    """
    Calculates residue classification metrics individually for each protein chain.
    """
    per_protein = []
    for pid, y_t, y_p in zip(protein_ids, y_trues, y_probs):
        m = compute_binary_metrics(y_t, y_p, threshold=threshold)
        m["protein_id"] = pid
        per_protein.append(m)
    return per_protein
