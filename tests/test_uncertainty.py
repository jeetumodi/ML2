"""
Unit tests for Monte Carlo Dropout and uncertainty quantification.
"""

import numpy as np
import torch
import torch.nn as nn
import pytest
from src.uncertainty.mc_dropout import enable_dropout_layers_only, compute_binary_entropy
from src.uncertainty.uncertainty_metrics import compute_error_detection_auroc, compute_risk_coverage_curve


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10)
        self.dp = nn.Dropout(0.5)
        self.out = nn.Linear(10, 1)

    def forward(self, x):
        return torch.sigmoid(self.out(self.dp(self.bn(self.fc(x)))))


def test_enable_dropout_layers_only():
    model = DummyModel()
    model.eval()

    enable_dropout_layers_only(model)

    assert model.dp.training is True, "Dropout layer should be in training mode"
    assert model.bn.training is False, "BatchNorm should remain in eval mode"
    assert model.fc.training is False, "Linear layer should remain in eval mode"


def test_compute_binary_entropy():
    probs = np.array([0.0, 0.5, 1.0])
    ent = compute_binary_entropy(probs)

    assert ent[1] > ent[0], "Entropy at p=0.5 must be maximum"
    assert ent[1] > ent[2], "Entropy at p=0.5 must be maximum"
    assert np.isclose(ent[1], np.log(2.0), atol=1e-3)
    assert np.all(ent >= 0.0)


def test_error_detection_auroc():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.1]) # index 2 is FP, index 5 is FN
    # High uncertainty on errors
    uncertainty = np.array([0.05, 0.08, 0.80, 0.10, 0.15, 0.75])

    res = compute_error_detection_auroc(y_true, y_prob, uncertainty, threshold=0.5)
    assert res["error_detection_auroc"] > 0.80, "AUROC should be high when errors have high uncertainty"
    assert res["n_errors"] == 2
