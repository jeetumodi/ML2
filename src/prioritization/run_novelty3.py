"""
Novelty Three Execution Pipeline: Uncertainty-Guided Interface Candidate Prioritization.

Compares:
1. Strategy A: Probability-Only Ranking
2. Strategy B: Lower-Confidence-Bound (LCB) Ranking (optimal lambda tuned on calibration split)
3. Strategy C: Uncertainty Filtering
4. Strategy D: Budget-Constrained Screening (K=5, 10, 20)

Includes actual real-protein case study with ground-truth validation.
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils_Test import ProDataset, getID
from src.prioritization.ranking import (
    rank_residues_strategy_a,
    rank_residues_strategy_b,
    rank_residues_strategy_c,
    select_top_k_candidates,
    tune_lcb_lambda_on_validation
)
from src.prioritization.ranking_metrics import (
    compute_protein_precision_recall_at_k,
    evaluate_screening_budget_across_proteins,
    compute_cumulative_recall_curve
)


def load_per_protein_data(dataset_path: str, prob_array: np.ndarray, unc_array: np.ndarray):
    """
    Splits concatenated predictions back into per-protein arrays based on sequence lengths.
    """
    ids, labels, seqs = getID(dataset_path)
    per_protein_trues = []
    per_protein_probs = []
    per_protein_uncs = []

    curr = 0
    for pid, lab, seq in zip(ids, labels, seqs):
        L = len(seq)
        sub_prob = prob_array[curr:curr+L]
        sub_unc = unc_array[curr:curr+L]

        y_t = [int(c) for c in str(lab)]
        if len(y_t) < L:
            y_t = [0] * (L - len(y_t)) + y_t
        elif len(y_t) > L:
            y_t = y_t[:L]
        y_t = np.array(y_t, dtype=int)

        per_protein_trues.append(y_t)
        per_protein_probs.append(sub_prob)
        per_protein_uncs.append(sub_unc)
        curr += L

    return ids, per_protein_trues, per_protein_probs, per_protein_uncs


def main():
    out_dir = os.path.join(REPO_ROOT, "results", "novelty_3")
    os.makedirs(out_dir, exist_ok=True)
    n1_dir = os.path.join(REPO_ROOT, "results", "novelty_1")
    base_dir = os.path.join(REPO_ROOT, "results", "baseline")

    print("=== Starting Novelty 3: Uncertainty-Guided Candidate Prioritization ===")

    # 1. Load calibration data to tune lambda parameter
    calib_probs = np.load(os.path.join(base_dir, "calib_y_prob.npy"))
    calib_trues = np.load(os.path.join(base_dir, "calib_y_true.npy"))
    # Margin uncertainty proxy on calibration set
    calib_unc = 1.0 - np.abs(2.0 * calib_probs - 1.0)

    calib_fasta = os.path.join(base_dir, "calib_split_67.fa")
    _, cal_trues_list, cal_probs_list, cal_uncs_list = load_per_protein_data(
        calib_fasta, calib_probs, calib_unc
    )

    print("Tuning LCB lambda parameter on held-out calibration set (maximizing Precision@10)...")
    lambda_candidates = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    best_lambda, lambda_tuning_results = tune_lcb_lambda_on_validation(
        cal_probs_list, cal_uncs_list, cal_trues_list,
        lambda_candidates=lambda_candidates, target_k=10
    )
    print(f"Optimal lambda selected on calibration data: lambda* = {best_lambda}")
    for lam, p10 in lambda_tuning_results.items():
        print(f"  lambda = {lam:4.2f}: Val Precision@10 = {p10:.4f}")

    # 2. Load Test_60 benchmark data and MC-Dropout uncertainty
    test60_df = pd.read_csv(os.path.join(n1_dir, "test60_calibrated_predictions.csv"))
    test60_probs = test60_df["prob_temperature_calibrated"].values
    test60_unc_var = test60_df["predictive_variance"].values
    test60_unc_std = test60_df["predictive_std"].values

    test60_fasta = os.path.join(REPO_ROOT, "Datasets", "Test_60.fa")
    pids_60, trues_60_list, probs_60_list, uncs_60_list = load_per_protein_data(
        test60_fasta, test60_probs, test60_unc_std
    )
    print(f"Loaded Test_60 benchmark ({len(pids_60)} proteins) for prioritization evaluation.")

    # 3. Strategy A: Probability-Only Ranking
    scores_strat_a = [rank_residues_strategy_a(p) for p in probs_60_list]
    res_strat_a = evaluate_screening_budget_across_proteins(trues_60_list, scores_strat_a, k_values=[5, 10, 20])

    # 4. Strategy B: LCB Ranking with tuned lambda
    scores_strat_b = [rank_residues_strategy_b(p, u, lambda_param=best_lambda) for p, u in zip(probs_60_list, uncs_60_list)]
    res_strat_b = evaluate_screening_budget_across_proteins(trues_60_list, scores_strat_b, k_values=[5, 10, 20])

    # Also evaluate sensitivity across other lambda values
    sensitivity_by_lambda = {}
    for lam in [0.1, 0.5, 1.0, 2.0]:
        scores_lam = [rank_residues_strategy_b(p, u, lambda_param=lam) for p, u in zip(probs_60_list, uncs_60_list)]
        sensitivity_by_lambda[f"lambda_{lam}"] = evaluate_screening_budget_across_proteins(
            trues_60_list, scores_lam, k_values=[5, 10, 20]
        )

    # 5. Strategy C: Uncertainty Filtering (cutoff = 90th percentile uncertainty)
    unc_cutoff = float(np.percentile(test60_unc_std, 90))
    scores_strat_c = [rank_residues_strategy_c(p, u, uncertainty_threshold=unc_cutoff) for p, u in zip(probs_60_list, uncs_60_list)]
    res_strat_c = evaluate_screening_budget_across_proteins(trues_60_list, scores_strat_c, k_values=[5, 10, 20])

    print("\n--- Candidate Prioritization Screening Comparison (Test_60) ---")
    for k in [5, 10, 20]:
        print(f"Top-{k} Screening Budget:")
        print(f"  Strategy A (Prob-Only):     Mean P@{k} = {res_strat_a[f'mean_precision_at_{k}']:.4f}, Mean R@{k} = {res_strat_a[f'mean_recall_at_{k}']:.4f}")
        print(f"  Strategy B (LCB lambda*):   Mean P@{k} = {res_strat_b[f'mean_precision_at_{k}']:.4f}, Mean R@{k} = {res_strat_b[f'mean_recall_at_{k}']:.4f}")
        print(f"  Strategy C (Unc-Filtered):  Mean P@{k} = {res_strat_c[f'mean_precision_at_{k}']:.4f}, Mean R@{k} = {res_strat_c[f'mean_recall_at_{k}']:.4f}")

    # 6. Cumulative Recall Curves
    curve_a = compute_cumulative_recall_curve(trues_60_list, scores_strat_a, max_k=40)
    curve_b = compute_cumulative_recall_curve(trues_60_list, scores_strat_b, max_k=40)
    curve_c = compute_cumulative_recall_curve(trues_60_list, scores_strat_c, max_k=40)

    if plt is not None:
        plt.figure(figsize=(7, 5))
        plt.plot(curve_a["budgets"], curve_a["mean_recall"], color="#1f77b4", lw=2.5, label="Strategy A: Probability-Only")
        plt.plot(curve_b["budgets"], curve_b["mean_recall"], color="#2ca02c", lw=2.5, linestyle="--", label=f"Strategy B: LCB (lambda={best_lambda})")
        plt.plot(curve_c["budgets"], curve_c["mean_recall"], color="#d62728", lw=2, linestyle=":", label="Strategy C: Uncertainty Filtered")
        plt.xlabel("Experimental Screening Budget K (Residues per Protein)", fontsize=11)
        plt.ylabel("Mean Cumulative Recall Across Proteins", fontsize=11)
        plt.title("Cumulative Interface Recall vs Screening Budget (Test_60)", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cumulative_recall_vs_budget.png"), dpi=300)
        plt.close()

        # 7. Precision@K Comparison Bar Plot
        k_labels = ["Top-5", "Top-10", "Top-20"]
        x = np.arange(len(k_labels))
        width = 0.25

        p_a = [res_strat_a[f"mean_precision_at_{k}"] for k in [5, 10, 20]]
        p_b = [res_strat_b[f"mean_precision_at_{k}"] for k in [5, 10, 20]]
        p_c = [res_strat_c[f"mean_precision_at_{k}"] for k in [5, 10, 20]]

        plt.figure(figsize=(7, 4.5))
        plt.bar(x - width, p_a, width, label="Strategy A (Prob-Only)", color="#1f77b4", alpha=0.85)
        plt.bar(x, p_b, width, label=f"Strategy B (LCB lambda={best_lambda})", color="#2ca02c", alpha=0.85)
        plt.bar(x + width, p_c, width, label="Strategy C (Unc-Filtered)", color="#d62728", alpha=0.85)
        plt.xticks(x, k_labels, fontsize=11)
        plt.ylabel("Mean Precision@K Across Proteins", fontsize=11)
        plt.title("Candidate Prioritization Precision at Screening Budgets", fontsize=12, fontweight="bold")
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(axis="y", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "precision_at_k_comparison.png"), dpi=300)
        plt.close()

    # 8. Real Protein Case Study
    # Pick a representative held-out protein where uncertainty penalized false positives (e.g. 2v9tA or 1a2yA)
    case_idx = 0
    case_pid = pids_60[case_idx]
    case_true = trues_60_list[case_idx]
    case_prob = probs_60_list[case_idx]
    case_unc = uncs_60_list[case_idx]
    case_score_a = scores_strat_a[case_idx]
    case_score_b = scores_strat_b[case_idx]

    df_case = pd.DataFrame({
        "residue_index": np.arange(len(case_true)),
        "ground_truth": case_true,
        "prob_calibrated": case_prob,
        "uncertainty_std": case_unc,
        "rank_score_prob_only": case_score_a,
        "rank_score_lcb": case_score_b
    })
    # Add ranks (1-based)
    df_case["rank_prob_only"] = df_case["rank_score_prob_only"].rank(ascending=False, method="min").astype(int)
    df_case["rank_lcb"] = df_case["rank_score_lcb"].rank(ascending=False, method="min").astype(int)
    df_case.to_csv(os.path.join(out_dir, "case_study_protein.csv"), index=False)

    # Plot case study residue profiles
    if plt is not None:
        plt.figure(figsize=(10, 4.5))
        plt.plot(df_case["residue_index"], df_case["prob_calibrated"], color="#1f77b4", lw=2, label="Calibrated Probability")
        plt.fill_between(
            df_case["residue_index"],
            df_case["prob_calibrated"] - df_case["uncertainty_std"],
            df_case["prob_calibrated"] + df_case["uncertainty_std"],
            color="#1f77b4", alpha=0.2, label="+/- 1 Std Epistemic Uncertainty"
        )
        plt.scatter(
            df_case["residue_index"][df_case["ground_truth"] == 1],
            df_case["prob_calibrated"][df_case["ground_truth"] == 1],
            color="#2ca02c", s=40, zorder=5, label="True Interface Residue (1)"
        )
        plt.axhline(0.33, color="red", linestyle=":", label="Classification Threshold (0.33)")
        plt.xlabel("Residue Position Index", fontsize=11)
        plt.ylabel("Calibrated Probability / Uncertainty", fontsize=11)
        plt.title(f"Case Study Profile: Held-Out Protein {case_pid} (Length {len(case_true)})", fontsize=12, fontweight="bold")
        plt.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "case_study_protein.png"), dpi=300)
        plt.close()

    # 9. Save JSON Summary
    summary = {
        "experiment": "Novelty_3_Candidate_Prioritization",
        "benchmark": "Test_60",
        "tuned_lambda_optimal": best_lambda,
        "lambda_validation_tuning": lambda_tuning_results,
        "uncertainty_filter_cutoff": unc_cutoff,
        "strategy_a_prob_only": res_strat_a,
        "strategy_b_lcb_optimal": res_strat_b,
        "strategy_b_sensitivity": sensitivity_by_lambda,
        "strategy_c_filtered": res_strat_c,
        "case_study_protein_id": case_pid,
        "case_study_length": len(case_true),
        "case_study_true_positives": int(np.sum(case_true == 1))
    }

    with open(os.path.join(out_dir, "ranking_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nNovelty 3 completed. Results and case studies saved to {out_dir}")


if __name__ == "__main__":
    main()
