"""
Monte Carlo Dropout Inference Engine for Epistemic Uncertainty Estimation in EGCPPIS.

Selectively activates dropout layers during evaluation while preserving
deterministic evaluation behavior in normalization and feature extractors.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple


def enable_dropout_layers_only(model: nn.Module) -> None:
    """
    Sets the entire model to eval mode, then selectively sets only
    Dropout layers to training mode. Ensures BatchNorm, LayerNorm, and
    InstanceNorm statistics remain fixed.
    """
    model.eval()
    dropout_count = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()
            dropout_count += 1
    return dropout_count


def compute_binary_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes Shannon entropy H(p) = -p log(p) - (1-p) log(1-p).
    """
    p = np.clip(p, eps, 1.0 - eps)
    return - (p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


class MCDropoutInference:
    """
    Performs Monte Carlo Dropout sampling for epistemic uncertainty quantification.
    """
    def __init__(self, model: nn.Module, num_passes: int = 20, device: str = "cuda"):
        self.model = model
        self.num_passes = num_passes
        self.device = device if torch.cuda.is_available() else "cpu"

    def predict_protein(
        self,
        batch_inputs: Tuple
    ) -> Dict[str, np.ndarray]:
        """
        Runs num_passes forward passes for a single protein batch.

        Returns:
            Dictionary containing:
            - mean_probability: shape (L,)
            - predictive_variance: shape (L,)
            - predictive_std: shape (L,)
            - predictive_entropy: shape (L,)
            - mutual_information: shape (L,)
            - all_passes: shape (T, L)
        """
        (
            seq_data, dssp_data, atom_onehot_embedding, edge_index_atom, res2atom, bert_data, pseudo_dis, dis_data,
            Res_matrix, node_features, label, coords_atom, seq1
        ) = batch_inputs

        node_number = np.array(seq_data)[0].shape[0]

        if self.device == "cuda":
            node_features = node_features.cuda().float()
            bert_var = bert_data.cuda().float()
            pseudo_dis = pseudo_dis.cuda().float()
            Res_matrix = Res_matrix.cuda().float()
            atom_onehot_embedding = atom_onehot_embedding.cuda().float()
            edge_index_atom = edge_index_atom.cuda()
            coords_atom = coords_atom.cuda().float()
            res2atom = res2atom.cuda().to(torch.int64)
        else:
            node_features = node_features.float()
            bert_var = bert_data.float()
            pseudo_dis = pseudo_dis.float()
            Res_matrix = Res_matrix.float()
            atom_onehot_embedding = atom_onehot_embedding.float()
            edge_index_atom = edge_index_atom
            coords_atom = coords_atom.float()
            res2atom = res2atom.to(torch.int64)

        enable_dropout_layers_only(self.model)

        pass_probs = []
        with torch.no_grad():
            for t in range(self.num_passes):
                output, _ = self.model(
                    node_number, node_features, bert_var, pseudo_dis, Res_matrix,
                    atom_onehot_embedding, edge_index_atom, res2atom, coords_atom
                )
                prob = output.squeeze().detach().cpu().numpy().astype(float)
                if prob.ndim == 0:
                    prob = np.array([float(prob)])
                pass_probs.append(prob)

        pass_probs = np.array(pass_probs)  # (T, L)
        mean_prob = np.mean(pass_probs, axis=0)  # (L,)
        pred_variance = np.var(pass_probs, axis=0)  # (L,)
        pred_std = np.sqrt(pred_variance)  # (L,)

        # Total predictive entropy
        pred_entropy = compute_binary_entropy(mean_prob)

        # Expected entropy of individual passes (Aleatoric uncertainty estimate)
        individual_entropies = np.mean([compute_binary_entropy(p) for p in pass_probs], axis=0)

        # Mutual Information = Total Entropy - Expected Entropy (Epistemic component)
        mutual_info = np.maximum(0.0, pred_entropy - individual_entropies)

        return {
            "mean_probability": mean_prob,
            "predictive_variance": pred_variance,
            "predictive_std": pred_std,
            "predictive_entropy": pred_entropy,
            "mutual_information": mutual_info,
            "all_passes": pass_probs
        }

    def predict_dataset(self, dataset) -> Dict[str, Any]:
        """
        Runs MC Dropout across the entire dataset.
        """
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        print(f"Executing MC Dropout (T={self.num_passes}) across {len(dataset)} proteins...")
        start_time = time.time()

        protein_ids = []
        all_trues = []
        all_means = []
        all_vars = []
        all_stds = []
        all_entropies = []
        all_mis = []

        for idx, batch in enumerate(loader):
            pid = dataset.IDs[idx]
            res = self.predict_protein(batch)

            y_t = np.array(batch[10]).flatten()
            L = len(res["mean_probability"])
            if len(y_t) > L:
                y_t = y_t[:L]
            elif len(y_t) < L:
                for k in ["mean_probability", "predictive_variance", "predictive_std", "predictive_entropy", "mutual_information"]:
                    res[k] = res[k][:len(y_t)]

            protein_ids.append(pid)
            all_trues.append(y_t)
            all_means.append(res["mean_probability"])
            all_vars.append(res["predictive_variance"])
            all_stds.append(res["predictive_std"])
            all_entropies.append(res["predictive_entropy"])
            all_mis.append(res["mutual_information"])

        elapsed = time.time() - start_time
        print(f"Finished MC Dropout inference in {elapsed:.2f}s ({elapsed / len(dataset):.3f}s/protein)")

        return {
            "protein_ids": protein_ids,
            "y_trues": all_trues,
            "mean_probs": all_means,
            "predictive_variances": all_vars,
            "predictive_stds": all_stds,
            "predictive_entropies": all_entropies,
            "mutual_informations": all_mis,
            "concat_true": np.concatenate(all_trues),
            "concat_mean_prob": np.concatenate(all_means),
            "concat_variance": np.concatenate(all_vars),
            "concat_std": np.concatenate(all_stds),
            "concat_entropy": np.concatenate(all_entropies),
            "concat_mi": np.concatenate(all_mis),
            "elapsed_seconds": elapsed,
            "num_passes": self.num_passes
        }
