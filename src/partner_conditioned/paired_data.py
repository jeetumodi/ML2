"""
Paired Protein Complex Data Representation for Partner-Conditioned PPIS.

Defines schemas and dataset abstractions for partner-conditioned interface prediction,
where a receptor protein's interface residues are predicted specifically with respect
to an interacting partner protein.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class PairedProteinComplex:
    """
    Data container for a pair of interacting proteins (Receptor A and Partner B).
    """
    complex_id: str
    receptor_id: str
    partner_id: str
    seq_receptor: str
    seq_partner: str
    coords_receptor: np.ndarray      # (L_A, 3) CA or pseudo-atom coordinates
    coords_partner: np.ndarray       # (L_B, 3) CA or pseudo-atom coordinates
    features_receptor: np.ndarray    # (L_A, D_in) sequence/structural features
    features_partner: np.ndarray     # (L_B, D_in) sequence/structural features
    inter_chain_distance: np.ndarray # (L_A, L_B) minimum atom or CA distance
    labels_partner_specific: np.ndarray # (L_A,) binary ground truth specific to this partner

    def __post_init__(self):
        L_A = len(self.seq_receptor)
        L_B = len(self.seq_partner)
        assert len(self.coords_receptor) == L_A, f"Receptor coords mismatch: {len(self.coords_receptor)} vs {L_A}"
        assert len(self.coords_partner) == L_B, f"Partner coords mismatch: {len(self.coords_partner)} vs {L_B}"
        assert self.inter_chain_distance.shape == (L_A, L_B), (
            f"Distance matrix shape mismatch: {self.inter_chain_distance.shape} vs ({L_A}, {L_B})"
        )
        assert len(self.labels_partner_specific) == L_A, (
            f"Labels length mismatch: {len(self.labels_partner_specific)} vs {L_A}"
        )


class PairedProteinDataset(Dataset):
    """
    PyTorch Dataset for paired protein complex structures.
    """
    def __init__(self, complexes: List[PairedProteinComplex]):
        self.complexes = complexes

    def __len__(self) -> int:
        return len(self.complexes)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        c = self.complexes[index]
        return {
            "complex_id": c.complex_id,
            "receptor_id": c.receptor_id,
            "partner_id": c.partner_id,
            "seq_receptor": c.seq_receptor,
            "seq_partner": c.seq_partner,
            "features_receptor": torch.from_numpy(c.features_receptor).float(),
            "features_partner": torch.from_numpy(c.features_partner).float(),
            "coords_receptor": torch.from_numpy(c.coords_receptor).float(),
            "coords_partner": torch.from_numpy(c.coords_partner).float(),
            "inter_chain_distance": torch.from_numpy(c.inter_chain_distance).float(),
            "labels": torch.from_numpy(c.labels_partner_specific).float()
        }
