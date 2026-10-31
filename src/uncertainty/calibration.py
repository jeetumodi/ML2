"""
Probability Calibration for Protein-Protein Interaction Site Prediction.

Implements:
- Temperature Scaling for binary classification logits: p_calibrated = sigmoid(logit / T)
- Platt Scaling (logistic calibration)
- Isotonic Regression Calibration
- Reliability diagram calculation and Expected Calibration Error (ECE)

Calibration models are fit strictly on held-out calibration data (never on test data).
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from typing import Tuple, Dict, Any, Optional


class TemperatureScaler:
    """
    Post-hoc Temperature Scaling for binary classification logits.
    
    Formula:
        p_calibrated = sigmoid(logit / T)
        
    Fits T > 0 by minimizing Binary Cross-Entropy (Negative Log-Likelihood)
    on a held-out calibration set.
    """
    def __init__(self, initial_t: float = 1.0):
        self.temperature = float(initial_t)
        self.is_fitted = False

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """
        Fits temperature parameter T strictly on calibration data.
        """
        logits = np.asarray(logits, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()

        eps = 1e-12

        def nll_objective(t_val):
            T = t_val[0]
            if T <= 0.01:
                return 1e9
            scaled_logits = logits / T
            # Numerically stable sigmoid BCE
            # - [y * log(sigmoid(z)) + (1-y) * log(1 - sigmoid(z))]
            # = max(z, 0) - z*y + log(1 + exp(-|z|))
            loss = np.maximum(scaled_logits, 0.0) - scaled_logits * y_true + np.log1p(np.exp(-np.abs(scaled_logits)))
            return float(np.mean(loss))

        res = minimize(
            nll_objective,
            x0=[self.temperature],
            bounds=[(0.05, 10.0)],
            method="L-BFGS-B"
        )

        if res.success:
            self.temperature = float(res.x[0])
            self.is_fitted = True
        else:
            # Fallback to initial if optimization fails
            self.temperature = max(float(res.x[0]), 0.1)
            self.is_fitted = True

        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Calibrates logits into probabilities using fitted temperature.
        """
        logits = np.asarray(logits, dtype=float)
        scaled_logits = logits / self.temperature
        # Sigmoid
        probs = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -30.0, 30.0)))
        return probs

    def get_temperature(self) -> float:
        return self.temperature


class PlattScaler:
    """
    Platt Scaling: Fits logistic regression on logits (scale and shift parameters).
    Formula: p_calibrated = sigmoid(a * logit + b)
    """
    def __init__(self):
        self.lr = LogisticRegression(solver="lbfgs")
        self.is_fitted = False

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "PlattScaler":
        logits = np.asarray(logits, dtype=float).reshape(-1, 1)
        y_true = np.asarray(y_true, dtype=int).ravel()
        self.lr.fit(logits, y_true)
        self.is_fitted = True
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float).reshape(-1, 1)
        return self.lr.predict_proba(logits)[:, 1]


class IsotonicCalibrator:
    """
    Non-parametric Isotonic Regression Calibration on raw probabilities.
    """
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        probs = np.asarray(probs, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()
        self.iso.fit(probs, y_true)
        self.is_fitted = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float).ravel()
        return self.iso.predict(probs)


def compute_reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Generates bin-wise statistics for reliability diagrams (calibration curves).
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        count = int(np.sum(in_bin))
        bin_counts[i] = count
        if count > 0:
            bin_accs[i] = float(np.mean(y_true[in_bin]))
            bin_confs[i] = float(np.mean(y_prob[in_bin]))
        else:
            bin_accs[i] = float("nan")
            bin_confs[i] = float(bin_centers[i])

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries
    }
