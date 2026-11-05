"""
Unit tests for paired protein representations and data validator checks.
"""

import numpy as np
import torch
import pytest
from src.partner_conditioned.paired_data import PairedProteinComplex, PairedProteinDataset
from src.partner_conditioned.cross_protein_model import CrossProteinEGCPPIS
from src.partner_conditioned.data_validator import validate_file_for_paired_condition


def test_paired_protein_complex_dataclass():
    L_A = 10
    L_B = 8
    coords_A = np.zeros((L_A, 3))
    coords_B = np.zeros((L_B, 3))
    dist_AB = np.ones((L_A, L_B))
    labels = np.zeros(L_A)

    complex_obj = PairedProteinComplex(
        complex_id="C1",
        receptor_id="A",
        partner_id="B",
        seq_receptor="A" * L_A,
        seq_partner="G" * L_B,
        coords_receptor=coords_A,
        coords_partner=coords_B,
        features_receptor=np.zeros((L_A, 115)),
        features_partner=np.zeros((L_B, 115)),
        inter_chain_distance=dist_AB,
        labels_partner_specific=labels
    )
    assert complex_obj.complex_id == "C1"
    assert complex_obj.inter_chain_distance.shape == (L_A, L_B)


def test_cross_protein_forward():
    model = CrossProteinEGCPPIS(in_dim=16, hidden_dim=32, num_heads=2)
    feat_A = torch.randn(12, 16)
    feat_B = torch.randn(10, 16)
    dist_AB = torch.rand(12, 10) * 15.0

    probs, logits = model(feat_A, feat_B, dist_AB)
    assert probs.shape == (12,)
    assert logits.shape == (12,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_data_validator_on_single_chain():
    # Single chain benchmark should be rejected as partner-conditioned
    import os
    fa_path = "Datasets/Test_60.fa" if os.path.exists("Datasets/Test_60.fa") else "ML2/Datasets/Test_60.fa"
    res = validate_file_for_paired_condition(fa_path)
    assert res["is_paired"] is False
    assert res["status"] == "REJECTED_PARTNER_INDEPENDENT_ONLY"
