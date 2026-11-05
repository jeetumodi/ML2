"""
Novelty Four Execution: Robustness and Generalization Across Protein Contexts.

Evaluates EGCPPIS and its uncertainty calibration across:
- Bound vs Unbound conformational shift (Test_60 vs UBtest_31-6)
- Sequence length groups (Short, Medium, Long)
- Interface density groups (Low, Medium, High)

Reports performance degradation, calibration drift, and error-detection stability.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from sklearn import metrics as sk_metrics

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.metrics import compute_binary_metrics, compute_ece
from src.uncertainty.uncertainty_metrics import compute_error_detection_auroc
from src.robustness.split_audit import build_robustness_groups
from src.prioritization.run_novelty3 import load_per_protein_data


def evaluate_single_group(
    pids: list,
    trues_list: list,
    probs_list: list,
    uncs_list: list,
    threshold: float = 0.33
) -> dict:
    """
    Computes all classification, calibration, and error-detection metrics for a subset of proteins.
    """
    cat_true = np.concatenate(trues_list)
    cat_prob = np.concatenate(probs_list)
    cat_unc = np.concatenate(uncs_list)

    base_metrics = compute_binary_metrics(cat_true, cat_prob, threshold=threshold)
    err_det = compute_error_detection_auroc(cat_true, cat_prob, cat_unc, threshold=threshold)

    return {
        "n_proteins": len(pids),
        "n_residues": len(cat_true),
        "n_positives": int(np.sum(cat_true == 1)),
        "prevalence": float(np.mean(cat_true)),
        "aupr": base_metrics["aupr"],
        "roc_auc": base_metrics["roc_auc"],
        "f1": base_metrics["f1"],
        "precision": base_metrics["precision"],
        "recall": base_metrics["recall"],
        "mcc": base_metrics["mcc"],
        "accuracy": base_metrics["accuracy"],
        "brier_score": base_metrics["brier_score"],
        "ece": base_metrics["ece"],
        "mce": base_metrics["mce"],
        "nll": base_metrics["nll"],
        "error_detection_auroc": err_det["error_detection_auroc"]
    }


def main():
    out_dir = os.path.join(REPO_ROOT, "results", "novelty_4")
    os.makedirs(out_dir, exist_ok=True)
    base_dir = os.path.join(REPO_ROOT, "results", "baseline")
    n1_dir = os.path.join(REPO_ROOT, "results", "novelty_1")

    print("=== Starting Novelty 4: Robustness and Generalization Evaluation ===")

    test60_fasta = os.path.join(REPO_ROOT, "Datasets", "Test_60.fa")
    ubtest_fasta = os.path.join(REPO_ROOT, "Datasets", "GraphPPIS", "UBtest_31-6.fa")

    groups = build_robustness_groups(test60_fasta, ubtest_fasta)

    # Load Test_60 per-protein data
    test60_df = pd.read_csv(os.path.join(n1_dir, "test60_calibrated_predictions.csv"))
    test60_probs = test60_df["prob_temperature_calibrated"].values
    test60_uncs = test60_df["predictive_std"].values
    pids_60, trues_60, probs_60, uncs_60 = load_per_protein_data(test60_fasta, test60_probs, test60_uncs)

    # Load UBtest_31-6 data
    ub_probs_raw = np.load(os.path.join(base_dir, "ubtest_y_prob.npy"))
    ub_trues_raw = np.load(os.path.join(base_dir, "ubtest_y_true.npy"))
    ub_logits_raw = np.load(os.path.join(base_dir, "ubtest_logits.npy"))
    # Load optimal T from Novelty 1
    with open(os.path.join(n1_dir, "novelty1_metrics_summary.json"), "r") as f:
        n1_sum = json.load(f)
    opt_T = n1_sum["optimal_temperature_T"]
    # Calibrate UBtest logits using fitted T
    ub_probs_cal = 1.0 / (1.0 + np.exp(-np.clip(ub_logits_raw / opt_T, -30.0, 30.0)))
    ub_uncs = 1.0 - np.abs(2.0 * ub_probs_cal - 1.0) # Margin uncertainty proxy
    pids_ub, trues_ub, probs_ub, uncs_ub = load_per_protein_data(ubtest_fasta, ub_probs_cal, ub_uncs)

    # Evaluate each group
    group_results = {}
    ref_metrics = None

    for g_name, g_info in groups.items():
        if g_info["source"] == "Test_60.fa":
            idxs = g_info["indices"]
            sub_pids = [pids_60[i] for i in idxs]
            sub_trues = [trues_60[i] for i in idxs]
            sub_probs = [probs_60[i] for i in idxs]
            sub_uncs = [uncs_60[i] for i in idxs]
        else:
            sub_pids = pids_ub
            sub_trues = trues_ub
            sub_probs = probs_ub
            sub_uncs = uncs_ub

        m = evaluate_single_group(sub_pids, sub_trues, sub_probs, sub_uncs, threshold=0.33)
        m["group_name"] = g_name
        m["description"] = g_info["description"]
        m["source"] = g_info["source"]

        if g_info.get("is_reference", False):
            ref_metrics = m

        group_results[g_name] = m

    # Calculate degradation relative to Reference_Bound_Test60
    rows = []
    for g_name, m in group_results.items():
        delta_aupr = m["aupr"] - ref_metrics["aupr"]
        delta_mcc = m["mcc"] - ref_metrics["mcc"]
        delta_ece = m["ece"] - ref_metrics["ece"]
        m["delta_aupr"] = float(delta_aupr)
        m["delta_mcc"] = float(delta_mcc)
        m["delta_ece"] = float(delta_ece)
        rows.append(m)

    df_groups = pd.DataFrame(rows)
    df_groups.to_csv(os.path.join(out_dir, "robustness_group_metrics.csv"), index=False)

    print("\n--- Robustness & Distribution Shift Metrics Summary ---")
    for r in rows:
        print(f"Group: {r['group_name']} ({r['n_proteins']} proteins, {r['n_residues']} residues)")
        print(f"  AUPR: {r['aupr']:.4f} (Delta: {r['delta_aupr']:+.4f}) | MCC: {r['mcc']:.4f} (Delta: {r['delta_mcc']:+.4f}) | ECE: {r['ece']:.4f}")

    # Plot 1: Performance Degradation across biological groups
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_groups = [
            "Reference_Bound_Test60",
            "Conformational_Shift_Unbound",
            "Length_Short_lt150",
            "Length_Long_gt300",
            "Density_Low_lt12pct",
            "Density_High_gt18pct"
        ]
        plot_labels = [
            "Ref: Bound Test_60",
            "Shift: Unbound UB_31",
            "Length < 150 aa",
            "Length > 300 aa",
            "Low Density (<12%)",
            "High Density (>18%)"
        ]
        auprs = [group_results[g]["aupr"] for g in plot_groups]
        mccs = [group_results[g]["mcc"] for g in plot_groups]
        eces = [group_results[g]["ece"] for g in plot_groups]

        x = np.arange(len(plot_labels))
        w = 0.25

        ax.bar(x - w, auprs, w, label="AUPR", color="#2ca02c", alpha=0.85)
        ax.bar(x, mccs, w, label="MCC", color="#1f77b4", alpha=0.85)
        ax.bar(x + w, eces, w, label="ECE", color="#d62728", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(plot_labels, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Metric Value", fontsize=11)
        ax.set_title("Performance & Calibration Across Distribution Shifts and Structural Groups", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "robustness_performance_degradation.png"), dpi=300)
        plt.close()

        # Plot 2: Conformation Shift PR Curves (Bound Test_60 vs Unbound UBtest_31-6)
        plt.figure(figsize=(6.5, 5))
        prec_60, rec_60, _ = sk_metrics.precision_recall_curve(np.concatenate(trues_60), np.concatenate(probs_60))
        prec_ub, rec_ub, _ = sk_metrics.precision_recall_curve(np.concatenate(trues_ub), np.concatenate(probs_ub))

        plt.plot(rec_60, prec_60, color="#1f77b4", lw=2.5, label=f"Bound Structures: Test_60 (AUPR = {group_results['Reference_Bound_Test60']['aupr']:.4f})")
        plt.plot(rec_ub, prec_ub, color="#d62728", lw=2.5, linestyle="--", label=f"Unbound Structures: UBtest_31-6 (AUPR = {group_results['Conformational_Shift_Unbound']['aupr']:.4f})")
        plt.xlabel("Recall", fontsize=11)
        plt.ylabel("Precision", fontsize=11)
        plt.title("Conformational Shift: Bound vs Unbound PPIS Performance", fontsize=12, fontweight="bold")
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "conformation_shift_pr_curves.png"), dpi=300)
        plt.close()

    # Save summary JSON
    summary = {
        "experiment": "Novelty_4_Robustness_Evaluation",
        "reference_group": "Reference_Bound_Test60",
        "groups": group_results
    }
    with open(os.path.join(out_dir, "robustness_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nNovelty 4 completed. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
