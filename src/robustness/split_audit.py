"""
Dataset Partitioning and Distribution-Shift Grouping for Robustness Evaluation.

Defines biological and structural evaluation groups:
1. Conformational Shift: Bound (Test_60) vs Unbound (UBtest_31-6)
2. Sequence Length Groups on Test_60: Short (<150 aa), Medium (150-300 aa), Long (>300 aa)
3. Interface Density Groups on Test_60: Low (<12%), Medium (12-18%), High (>18%)
"""

import os
import sys
import numpy as np
from typing import Dict, Any, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils_Test import getID


def build_robustness_groups(
    test60_fasta: str,
    ubtest_fasta: str
) -> Dict[str, Dict[str, Any]]:
    """
    Constructs distinct evaluation groups from available benchmark datasets.
    """
    # 1. Parse Test_60
    ids_60, labels_60, seqs_60 = getID(test60_fasta)
    lengths_60 = [len(s) for s in seqs_60]
    densities_60 = []
    for s, l in zip(seqs_60, labels_60):
        y = [int(c) for c in str(l)]
        densities_60.append(np.mean(y) if len(y) > 0 else 0.0)

    # 2. Parse UBtest_31-6
    ids_ub, labels_ub, seqs_ub = getID(ubtest_fasta)
    lengths_ub = [len(s) for s in seqs_ub]
    densities_ub = []
    for s, l in zip(seqs_ub, labels_ub):
        y = [int(c) for c in str(l)]
        densities_ub.append(np.mean(y) if len(y) > 0 else 0.0)

    groups = {}

    # Group 1: Reference In-Distribution Bound Structures (Test_60)
    groups["Reference_Bound_Test60"] = {
        "description": "Bound crystal structures (Standard held-out test benchmark)",
        "source": "Test_60.fa",
        "indices": list(range(len(ids_60))),
        "protein_ids": ids_60,
        "is_reference": True
    }

    # Group 2: Conformational Shift: Unbound Structures (UBtest_31-6)
    groups["Conformational_Shift_Unbound"] = {
        "description": "Unbound (free) structures with side-chain and backbone conformational shifts",
        "source": "UBtest_31-6.fa",
        "indices": list(range(len(ids_ub))),
        "protein_ids": ids_ub,
        "is_reference": False
    }

    # Groups 3, 4, 5: Sequence Length Groups on Test_60
    short_idx = [i for i, L in enumerate(lengths_60) if L < 150]
    med_idx = [i for i, L in enumerate(lengths_60) if 150 <= L <= 300]
    long_idx = [i for i, L in enumerate(lengths_60) if L > 300]

    groups["Length_Short_lt150"] = {
        "description": "Short proteins (Length < 150 aa)",
        "source": "Test_60.fa",
        "indices": short_idx,
        "protein_ids": [ids_60[i] for i in short_idx],
        "is_reference": False
    }
    groups["Length_Medium_150to300"] = {
        "description": "Medium proteins (Length 150 - 300 aa)",
        "source": "Test_60.fa",
        "indices": med_idx,
        "protein_ids": [ids_60[i] for i in med_idx],
        "is_reference": False
    }
    groups["Length_Long_gt300"] = {
        "description": "Long proteins (Length > 300 aa)",
        "source": "Test_60.fa",
        "indices": long_idx,
        "protein_ids": [ids_60[i] for i in long_idx],
        "is_reference": False
    }

    # Groups 6, 7, 8: Interface Density Groups on Test_60
    low_dens_idx = [i for i, d in enumerate(densities_60) if d < 0.12]
    med_dens_idx = [i for i, d in enumerate(densities_60) if 0.12 <= d <= 0.18]
    high_dens_idx = [i for i, d in enumerate(densities_60) if d > 0.18]

    groups["Density_Low_lt12pct"] = {
        "description": "Low interface residue prevalence (< 12%)",
        "source": "Test_60.fa",
        "indices": low_dens_idx,
        "protein_ids": [ids_60[i] for i in low_dens_idx],
        "is_reference": False
    }
    groups["Density_Medium_12to18pct"] = {
        "description": "Medium interface residue prevalence (12% - 18%)",
        "source": "Test_60.fa",
        "indices": med_dens_idx,
        "protein_ids": [ids_60[i] for i in med_dens_idx],
        "is_reference": False
    }
    groups["Density_High_gt18pct"] = {
        "description": "High interface residue prevalence (> 18%)",
        "source": "Test_60.fa",
        "indices": high_dens_idx,
        "protein_ids": [ids_60[i] for i in high_dens_idx],
        "is_reference": False
    }

    return groups
