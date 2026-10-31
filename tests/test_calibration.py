"""
Unit tests for calibration methods (Temperature Scaling, Platt Scaling, Isotonic Calibration).
"""

import numpy as np
import pytest
from src.uncertainty.calibration import (
    TemperatureScaler, PlattScaler, IsotonicCalibrator, compute_reliability_diagram_data
)


def test_temperature_scaling_fitting():
    np.random.seed(42)
    # Generate overconfident synthetic logits
    y_true = np.random.binomial(1, 0.3, size=500)
    # Logits with excessive variance simulating uncalibrated overconfidence
    logits = np.random.randn(500) * 4.0 + (y_true * 2.0 - 1.0)

    scaler = TemperatureScaler(initial_t=1.0)
    scaler.fit(logits, y_true)

    assert scaler.is_fitted
    assert scaler.temperature > 0.05
    assert scaler.temperature < 10.0

    cal_probs = scaler.predict_proba(logits)
    assert len(cal_probs) == len(logits)
    assert np.all(cal_probs >= 0.0)
    assert np.all(cal_probs <= 1.0)
    # Monotonicity check: higher logit must yield higher calibrated probability
    sorted_idx = np.argsort(logits)
    assert np.all(np.diff(cal_probs[sorted_idx]) >= -1e-7)


def test_platt_scaler():
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    logits = np.array([-2.0, -1.0, -0.5, 0.5, 1.2, 2.5])

    scaler = PlattScaler().fit(logits, y_true)
    probs = scaler.predict_proba(logits)

    assert len(probs) == 6
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert probs[-1] > probs[0]


def test_reliability_diagram_data():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    rd = compute_reliability_diagram_data(y_true, y_prob, n_bins=5)
    assert len(rd["bin_centers"]) == 5
    assert np.sum(rd["bin_counts"]) == 4
