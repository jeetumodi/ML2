"""
Evaluation Metrics for Uncertainty Quantification in Protein-Protein Interaction Sites.

Implements:
- Error Detection AUROC (evaluating whether uncertainty reliably flags prediction mistakes)
- Risk-Coverage Analysis (evaluating selective prediction performance)
- Precision-Coverage Analysis
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
"""

import numpy as np
from sklearn import metrics
from typing import Dict, Any, Tuple, List


def compute_error_detection_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.33
) -> Dict[str, float]:
    """
    Evaluates whether the uncertainty metric can detect classification errors.
    
    An error occurs when: y_pred != y_true, where y_pred = (y_prob >= threshold).
    The uncertainty score is used as the predictor of error.

    Returns:
        Dict containing error_detection_auroc, error_prevalence, total_errors.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    uncertainty = np.asarray(uncertainty, dtype=float).ravel()

    y_pred = (y_prob >= threshold).astype(int)
    errors = (y_pred != y_true).astype(int)

    n_errors = int(np.sum(errors == 1))
    n_correct = int(np.sum(errors == 0))

    if n_errors > 0 and n_correct > 0:
        auroc = float(metrics.roc_auc_score(errors, uncertainty))
        # Also compute Average Precision for error detection
        prec, rec, _ = metrics.precision_recall_curve(errors, uncertainty)
        aupr = float(metrics.auc(rec, prec))
    else:
        auroc = float("nan")
        aupr = float("nan")

    return {
        "error_detection_auroc": auroc,
        "error_detection_aupr": aupr,
        "error_prevalence": float(n_errors / len(errors)) if len(errors) > 0 else 0.0,
        "n_errors": n_errors,
        "n_correct": n_correct
    }


def compute_risk_coverage_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.33,
    coverages: np.ndarray = np.linspace(0.1, 1.0, 19)
) -> Dict[str, np.ndarray]:
    """
    Computes selective prediction risk and precision as a function of coverage.
    
    Residues are ordered by ascending uncertainty (most confident first).
    For each coverage C in (0, 1], we evaluate error rate and precision
    on the top C proportion of samples.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    uncertainty = np.asarray(uncertainty, dtype=float).ravel()

    n = len(y_true)
    # Sort ascending by uncertainty (lowest uncertainty first)
    sort_idx = np.argsort(uncertainty)
    y_true_sorted = y_true[sort_idx]
    y_prob_sorted = y_prob[sort_idx]
    y_pred_sorted = (y_prob_sorted >= threshold).astype(int)

    risk_list = []
    precision_list = []
    f1_list = []
    actual_coverages = []

    for cov in coverages:
        k = max(1, int(cov * n))
        sub_true = y_true_sorted[:k]
        sub_pred = y_pred_sorted[:k]
        sub_prob = y_prob_sorted[:k]

        # Empirical risk = 1 - Accuracy
        risk = float(np.mean(sub_pred != sub_true))
        
        # Precision on retained positives
        tp = np.sum((sub_true == 1) & (sub_pred == 1))
        fp = np.sum((sub_true == 0) & (sub_pred == 1))
        fn = np.sum((sub_true == 1) & (sub_pred == 0))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        risk_list.append(risk)
        precision_list.append(prec)
        f1_list.append(f1)
        actual_coverages.append(float(k / n))

    # Trapezoidal integration compatible with NumPy 2.0+ and 1.x
    if hasattr(np, "trapezoid"):
        aurc_val = float(np.trapezoid(risk_list, actual_coverages))
    elif hasattr(np, "trapz"):
        aurc_val = float(np.trapz(risk_list, actual_coverages))
    else:
        from scipy.integrate import trapezoid
        aurc_val = float(trapezoid(risk_list, actual_coverages))

    return {
        "coverages": np.array(actual_coverages),
        "risk": np.array(risk_list),
        "precision": np.array(precision_list),
        "f1": np.array(f1_list),
        "aurc": aurc_val
    }
