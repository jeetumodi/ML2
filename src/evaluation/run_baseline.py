"""
Baseline Evaluation Pipeline for EGCPPIS.

Runs inference on:
- Test_60 (Primary held-out bound test benchmark)
- UBtest_31-6 (Unbound test benchmark)
- Train_335_Val (20% held-out calibration set from Train_335, seed 42)

Saves all outputs (predictions, ground-truth, metrics, configurations, plots) to results/baseline/.
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from sklearn import metrics as sk_metrics

# Add root directory to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Model_alpha import PPI_Model
from utils_Test import ProDataset, getID
from torch_geometric.utils import dense_to_sparse
from src.evaluation.metrics import compute_binary_metrics, compute_per_protein_metrics


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_model_inference(model, dataset, device="cuda"):
    """
    Runs forward inference for every protein in the dataset.
    Returns:
        protein_ids: List[str]
        all_trues: List[np.ndarray] (per-protein binary labels)
        all_probs: List[np.ndarray] (per-protein probabilities)
        all_logits: List[np.ndarray] (per-protein logits)
    """
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    protein_ids = []
    all_trues = []
    all_probs = []
    all_logits = []

    print(f"Running inference on {len(dataset)} proteins...")
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (
            seq_data, dssp_data, atom_onehot_embedding, edge_index_atom, res2atom, bert_data, pseudo_dis, dis_data,
            Res_matrix, node_features, label, coords_atom, seq1
        ) in enumerate(loader):
            pid = dataset.IDs[batch_idx]
            node_number = np.array(seq_data)[0].shape[0]

            if torch.cuda.is_available() and device == "cuda":
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

            output, _ = model(
                node_number, node_features, bert_var, pseudo_dis, Res_matrix,
                atom_onehot_embedding, edge_index_atom, res2atom, coords_atom
            )
            # output is (1, L) or (L, 1) probability from sigmoid
            prob = output.squeeze().detach().cpu().numpy().astype(float)
            if prob.ndim == 0:
                prob = np.array([float(prob)])
            
            # Inverse sigmoid to retrieve logit: logit = log(p / (1 - p))
            eps = 1e-12
            prob_clipped = np.clip(prob, eps, 1.0 - eps)
            logit = np.log(prob_clipped / (1.0 - prob_clipped))

            y_t = np.array(label).flatten()
            if len(y_t) > len(prob):
                y_t = y_t[:len(prob)]
            elif len(y_t) < len(prob):
                prob = prob[:len(y_t)]
                logit = logit[:len(y_t)]

            protein_ids.append(pid)
            all_trues.append(y_t)
            all_probs.append(prob)
            all_logits.append(logit)

    elapsed = time.time() - start_time
    print(f"Completed inference in {elapsed:.2f}s ({elapsed / len(dataset):.3f}s/protein)")
    return protein_ids, all_trues, all_probs, all_logits, elapsed


def save_baseline_figures(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str, prefix: str = "test60"):
    """
    Generates ROC curve, PR curve, and score distribution figures.
    """
    if plt is None:
        print(f"Skipping plot generation for {prefix} (matplotlib not installed in environment)")
        return
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. ROC Curve
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_prob)
    roc_auc = sk_metrics.auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(f"Baseline ROC Curve ({prefix})", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_roc_curve.png"), dpi=300)
    plt.close()

    # 2. PR Curve
    prec, rec, _ = sk_metrics.precision_recall_curve(y_true, y_prob)
    aupr = sk_metrics.auc(rec, prec)

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="#2ca02c", lw=2, label=f"PR (AUPR = {aupr:.4f})")
    prevalence = np.mean(y_true)
    plt.axhline(prevalence, color="grey", linestyle="--", label=f"Random ({prevalence:.3f})")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title(f"Baseline Precision-Recall Curve ({prefix})", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_pr_curve.png"), dpi=300)
    plt.close()

    # 3. Probability Distribution
    plt.figure(figsize=(6, 4.5))
    plt.hist(y_prob[y_true == 0], bins=50, density=True, alpha=0.6, color="#1f77b4", label="Non-interface (0)")
    plt.hist(y_prob[y_true == 1], bins=50, density=True, alpha=0.6, color="#d62728", label="Interface (1)")
    plt.xlabel("Predicted Probability", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title(f"Baseline Score Distribution ({prefix})", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_score_dist.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Baseline Evaluation for EGCPPIS")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(REPO_ROOT, "model_save", "best_model_8_0.6406474225359946.pth"), help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.33, help="Decision threshold")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing baseline evaluation on device: {device}")

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(REPO_ROOT, "model_save", "best_model_8_0.6406474225359946.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(REPO_ROOT, "model_save", "best_model_13_0.630814106504997.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(REPO_ROOT, "model_save", "best_model.pth")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model = PPI_Model(liner_dim=115, embbing=128)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    if device == "cuda":
        model = model.cuda()
    model.eval()

    output_dir = os.path.join(REPO_ROOT, "results", "baseline")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Primary Benchmark: Test_60
    test60_path = os.path.join(REPO_ROOT, "Datasets", "Test_60.fa")
    test60_dataset = ProDataset(test60_path)
    pids_60, trues_60, probs_60, logits_60, time_60 = run_model_inference(model, test60_dataset, device)

    cat_trues_60 = np.concatenate(trues_60)
    cat_probs_60 = np.concatenate(probs_60)
    cat_logits_60 = np.concatenate(logits_60)

    # Compute metrics at standard threshold 0.33
    metrics_test60_fixed = compute_binary_metrics(cat_trues_60, cat_probs_60, threshold=0.33)
    metrics_test60_opt = compute_binary_metrics(cat_trues_60, cat_probs_60, threshold=None)

    print("\n--- Test_60 Baseline Metrics (threshold=0.33) ---")
    for k, v in metrics_test60_fixed.items():
        print(f"  {k}: {v}")

    # Per-protein metrics
    per_protein_test60 = compute_per_protein_metrics(pids_60, trues_60, probs_60, threshold=0.33)
    df_per_protein_60 = pd.DataFrame(per_protein_test60)
    df_per_protein_60.to_csv(os.path.join(output_dir, "test60_per_protein_metrics.csv"), index=False)

    # Save arrays
    np.save(os.path.join(output_dir, "test60_y_true.npy"), cat_trues_60)
    np.save(os.path.join(output_dir, "test60_y_prob.npy"), cat_probs_60)
    np.save(os.path.join(output_dir, "test60_logits.npy"), cat_logits_60)
    with open(os.path.join(output_dir, "test60_protein_ids.json"), "w") as f:
        json.dump(pids_60, f)

    save_baseline_figures(cat_trues_60, cat_probs_60, output_dir, prefix="test60")

    # 2. Unbound Benchmark: UBtest_31-6
    ub_path = os.path.join(REPO_ROOT, "Datasets", "GraphPPIS", "UBtest_31-6.fa")
    ub_dataset = ProDataset(ub_path)
    pids_ub, trues_ub, probs_ub, logits_ub, time_ub = run_model_inference(model, ub_dataset, device)

    cat_trues_ub = np.concatenate(trues_ub)
    cat_probs_ub = np.concatenate(probs_ub)
    cat_logits_ub = np.concatenate(logits_ub)

    metrics_ub_fixed = compute_binary_metrics(cat_trues_ub, cat_probs_ub, threshold=0.33)
    metrics_ub_opt = compute_binary_metrics(cat_trues_ub, cat_probs_ub, threshold=None)

    print("\n--- UBtest_31-6 Baseline Metrics (threshold=0.33) ---")
    for k, v in metrics_ub_fixed.items():
        print(f"  {k}: {v}")

    per_protein_ub = compute_per_protein_metrics(pids_ub, trues_ub, probs_ub, threshold=0.33)
    df_per_protein_ub = pd.DataFrame(per_protein_ub)
    df_per_protein_ub.to_csv(os.path.join(output_dir, "ubtest_per_protein_metrics.csv"), index=False)

    np.save(os.path.join(output_dir, "ubtest_y_true.npy"), cat_trues_ub)
    np.save(os.path.join(output_dir, "ubtest_y_prob.npy"), cat_probs_ub)
    np.save(os.path.join(output_dir, "ubtest_logits.npy"), cat_logits_ub)
    with open(os.path.join(output_dir, "ubtest_protein_ids.json"), "w") as f:
        json.dump(pids_ub, f)

    save_baseline_figures(cat_trues_ub, cat_probs_ub, output_dir, prefix="ubtest")

    # 3. Dedicated Disjoint Calibration Split (from Train_335, 20% random sample = 67 proteins)
    train_path = os.path.join(REPO_ROOT, "Datasets", "Train_335.fa")
    train_ids, _, _ = getID(train_path)
    rng = random.Random(42)
    val_indices = sorted(rng.sample(range(len(train_ids)), int(0.20 * len(train_ids))))
    val_ids = [train_ids[i] for i in val_indices]
    print(f"\nConstructing disjoint calibration split of {len(val_ids)} proteins from Train_335...")

    # Create temporary calibration FASTA
    calib_fasta = os.path.join(output_dir, "calib_split_67.fa")
    with open(train_path, "r") as fin, open(calib_fasta, "w") as fout:
        lines = fin.readlines()
        for i in range(0, len(lines), 3):
            pid = lines[i].strip()[1:]
            if pid in val_ids:
                fout.write(lines[i])
                fout.write(lines[i+1])
                fout.write(lines[i+2])

    calib_dataset = ProDataset(calib_fasta)
    pids_cal, trues_cal, probs_cal, logits_cal, time_cal = run_model_inference(model, calib_dataset, device)

    cat_trues_cal = np.concatenate(trues_cal)
    cat_probs_cal = np.concatenate(probs_cal)
    cat_logits_cal = np.concatenate(logits_cal)

    metrics_cal = compute_binary_metrics(cat_trues_cal, cat_probs_cal, threshold=0.33)
    np.save(os.path.join(output_dir, "calib_y_true.npy"), cat_trues_cal)
    np.save(os.path.join(output_dir, "calib_y_prob.npy"), cat_probs_cal)
    np.save(os.path.join(output_dir, "calib_logits.npy"), cat_logits_cal)
    with open(os.path.join(output_dir, "calib_protein_ids.json"), "w") as f:
        json.dump(pids_cal, f)

    # Save summary dictionary
    summary = {
        "experiment": "EGCPPIS_Baseline",
        "checkpoint": os.path.basename(checkpoint_path),
        "device": device,
        "seed": 42,
        "test60_metrics_fixed_threshold": metrics_test60_fixed,
        "test60_metrics_optimal_threshold": metrics_test60_opt,
        "ubtest_metrics_fixed_threshold": metrics_ub_fixed,
        "ubtest_metrics_optimal_threshold": metrics_ub_opt,
        "calib_metrics": metrics_cal,
        "inference_time_seconds": {
            "test60": time_60,
            "ubtest": time_ub,
            "calib": time_cal
        }
    }

    with open(os.path.join(output_dir, "baseline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll baseline results saved successfully to {output_dir}")


if __name__ == "__main__":
    main()
