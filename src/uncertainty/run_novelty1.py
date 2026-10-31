"""
Novelty One Execution Pipeline: Calibrated Uncertainty for PPI-Site Prediction.

Executes:
1. Temperature scaling, Platt scaling, and Isotonic calibration fit strictly on held-out calibration split.
2. Monte Carlo Dropout (T=20 passes) on Test_60 and UBtest_31-6.
3. Comparative evaluation of Raw vs Calibrated vs MC-Dropout predictions (ECE, Brier, NLL, AUROC, AUPR, MCC).
4. Error-detection AUROC and Risk-Coverage curve generation.
5. High-resolution figure generation and result serialization to results/novelty_1/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn import metrics as sk_metrics

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Model_alpha import PPI_Model
from utils_Test import ProDataset
from src.evaluation.metrics import compute_binary_metrics, compute_ece
from src.uncertainty.calibration import (
    TemperatureScaler, PlattScaler, IsotonicCalibrator, compute_reliability_diagram_data
)
from src.uncertainty.mc_dropout import MCDropoutInference
from src.uncertainty.uncertainty_metrics import (
    compute_error_detection_auroc, compute_risk_coverage_curve
)


def plot_reliability_diagrams(raw_data, calib_data, mc_data, out_path, ece_raw, ece_cal, ece_mc):
    """
    Plots multi-panel calibration curves (reliability diagrams) comparing Raw, Calibrated, and MC-Dropout.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    configs = [
        (axes[0], raw_data, f"Raw Uncalibrated (ECE = {ece_raw:.4f})", "#1f77b4"),
        (axes[1], calib_data, f"Temperature Scaled (ECE = {ece_cal:.4f})", "#2ca02c"),
        (axes[2], mc_data, f"MC-Dropout Mean (ECE = {ece_mc:.4f})", "#d62728")
    ]

    for ax, data, title, color in configs:
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Perfect Calibration")
        valid = ~np.isnan(data["bin_accuracies"])
        ax.bar(
            data["bin_centers"][valid],
            data["bin_accuracies"][valid],
            width=0.08,
            color=color,
            alpha=0.6,
            edgecolor=color,
            label="Observed Accuracy"
        )
        ax.plot(
            data["bin_centers"][valid],
            data["bin_accuracies"][valid],
            color=color,
            marker="o",
            lw=2
        )
        ax.set_xlabel("Predicted Probability / Confidence", fontsize=11)
        ax.set_ylabel("True Interface Proportion", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_uncertainty_diagnostics(
    uncertainty_dict, y_true, y_prob, out_dir, threshold=0.33
):
    """
    Generates diagnostic figures for uncertainty distributions, error detection, and risk-coverage.
    """
    y_pred = (y_prob >= threshold).astype(int)
    errors = (y_pred != y_true).astype(int)
    variance = uncertainty_dict["concat_variance"]
    entropy = uncertainty_dict["concat_entropy"]

    # 1. Uncertainty Distribution: Correct vs Incorrect
    plt.figure(figsize=(7, 4.5))
    plt.hist(variance[errors == 0], bins=40, density=True, alpha=0.6, color="#1f77b4", label="Correct Predictions (e=0)")
    plt.hist(variance[errors == 1], bins=40, density=True, alpha=0.6, color="#d62728", label="Incorrect Predictions (e=1)")
    plt.xlabel("Predictive Variance (Epistemic Uncertainty)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title("Epistemic Uncertainty Distribution: Correct vs Error", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "uncertainty_distribution_by_error.png"), dpi=300)
    plt.close()

    # 2. Error-Detection ROC Curves
    fpr_var, tpr_var, _ = sk_metrics.roc_curve(errors, variance)
    auc_var = sk_metrics.auc(fpr_var, tpr_var)

    fpr_ent, tpr_ent, _ = sk_metrics.roc_curve(errors, entropy)
    auc_ent = sk_metrics.auc(fpr_ent, tpr_ent)

    # Margin uncertainty: 1 - |2p - 1| (distance to decision boundary)
    margin_unc = 1.0 - np.abs(2.0 * y_prob - 1.0)
    fpr_mar, tpr_mar, _ = sk_metrics.roc_curve(errors, margin_unc)
    auc_mar = sk_metrics.auc(fpr_mar, tpr_mar)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_var, tpr_var, color="#d62728", lw=2, label=f"Predictive Variance (AUROC = {auc_var:.4f})")
    plt.plot(fpr_ent, tpr_ent, color="#ff7f0e", lw=2, label=f"Predictive Entropy (AUROC = {auc_ent:.4f})")
    plt.plot(fpr_mar, tpr_mar, color="#1f77b4", lw=2, linestyle="--", label=f"Margin Uncertainty (AUROC = {auc_mar:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle=":")
    plt.xlabel("False Alarm Rate (Correct flagged as Error)", fontsize=11)
    plt.ylabel("Detection Rate (True Errors flagged)", fontsize=11)
    plt.title("Error-Detection ROC (Uncertainty as Error Score)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_detection_roc.png"), dpi=300)
    plt.close()

    # 3. Risk-Coverage Curves
    rc_var = compute_risk_coverage_curve(y_true, y_prob, variance, threshold=threshold)
    rc_ent = compute_risk_coverage_curve(y_true, y_prob, entropy, threshold=threshold)
    rc_random = compute_risk_coverage_curve(y_true, y_prob, np.random.RandomState(42).rand(len(y_true)), threshold=threshold)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(rc_var["coverages"], rc_var["risk"], color="#d62728", lw=2, label=f"Predictive Variance Filter (AURC = {rc_var['aurc']:.4f})")
    plt.plot(rc_ent["coverages"], rc_ent["risk"], color="#ff7f0e", lw=2, label=f"Predictive Entropy Filter (AURC = {rc_ent['aurc']:.4f})")
    plt.plot(rc_random["coverages"], rc_random["risk"], color="grey", linestyle="--", label=f"Random Rejection (AURC = {rc_random['aurc']:.4f})")
    plt.xlabel("Prediction Coverage (Retained Residues Ratio)", fontsize=11)
    plt.ylabel("Empirical Risk (Error Rate on Retained)", fontsize=11)
    plt.title("Risk-Coverage Curve (Selective Classification)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "risk_coverage_curve.png"), dpi=300)
    plt.close()

    # 4. Precision-Coverage Curves
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(rc_var["coverages"], rc_var["precision"], color="#2ca02c", lw=2, label="Variance Filtered Precision")
    plt.plot(rc_ent["coverages"], rc_ent["precision"], color="#17becf", lw=2, label="Entropy Filtered Precision")
    plt.xlabel("Prediction Coverage", fontsize=11)
    plt.ylabel("Precision on Retained Candidates", fontsize=11)
    plt.title("Precision vs Prediction Coverage", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_coverage_curve.png"), dpi=300)
    plt.close()

    return {
        "error_auroc_variance": float(auc_var),
        "error_auroc_entropy": float(auc_ent),
        "error_auroc_margin": float(auc_mar),
        "aurc_variance": float(rc_var["aurc"]),
        "aurc_entropy": float(rc_ent["aurc"]),
        "aurc_random": float(rc_random["aurc"])
    }


def main():
    out_dir = os.path.join(REPO_ROOT, "results", "novelty_1")
    os.makedirs(out_dir, exist_ok=True)
    baseline_dir = os.path.join(REPO_ROOT, "results", "baseline")

    print("=== Starting Novelty 1: Calibrated Uncertainty for PPI-Site Prediction ===")

    # 1. Load Calibration Split
    calib_logits = np.load(os.path.join(baseline_dir, "calib_logits.npy"))
    calib_trues = np.load(os.path.join(baseline_dir, "calib_y_true.npy"))
    calib_probs = np.load(os.path.join(baseline_dir, "calib_y_prob.npy"))
    print(f"Loaded calibration split: {len(calib_trues)} residues from 66 held-out proteins.")

    # 2. Fit Calibrators strictly on calibration split
    print("Fitting Temperature Scaler on calibration data...")
    temp_scaler = TemperatureScaler(initial_t=1.0)
    temp_scaler.fit(calib_logits, calib_trues)
    optimal_T = temp_scaler.get_temperature()
    print(f"Optimal Temperature parameter T* = {optimal_T:.4f}")

    print("Fitting Platt Scaler on calibration data...")
    platt_scaler = PlattScaler().fit(calib_logits, calib_trues)

    print("Fitting Isotonic Calibrator on calibration data...")
    iso_calibrator = IsotonicCalibrator().fit(calib_probs, calib_trues)

    # 3. Load Test Benchmark Data (Test_60)
    test60_logits = np.load(os.path.join(baseline_dir, "test60_logits.npy"))
    test60_trues = np.load(os.path.join(baseline_dir, "test60_y_true.npy"))
    test60_probs_raw = np.load(os.path.join(baseline_dir, "test60_y_prob.npy"))
    with open(os.path.join(baseline_dir, "test60_protein_ids.json"), "r") as f:
        test60_pids = json.load(f)

    # 4. Apply Calibration to Test_60
    test60_probs_temp = temp_scaler.predict_proba(test60_logits)
    test60_probs_platt = platt_scaler.predict_proba(test60_logits)
    test60_probs_iso = iso_calibrator.predict_proba(test60_probs_raw)

    metrics_raw = compute_binary_metrics(test60_trues, test60_probs_raw, threshold=0.33)
    metrics_temp = compute_binary_metrics(test60_trues, test60_probs_temp, threshold=0.33)
    metrics_platt = compute_binary_metrics(test60_trues, test60_probs_platt, threshold=0.33)
    metrics_iso = compute_binary_metrics(test60_trues, test60_probs_iso, threshold=0.33)

    print("\n--- Test_60 Calibration Comparison ---")
    print(f"Raw Probabilities:     ECE = {metrics_raw['ece']:.4f}, Brier = {metrics_raw['brier_score']:.4f}, NLL = {metrics_raw['nll']:.4f}")
    print(f"Temperature Scaled:    ECE = {metrics_temp['ece']:.4f}, Brier = {metrics_temp['brier_score']:.4f}, NLL = {metrics_temp['nll']:.4f}")
    print(f"Platt Scaled:          ECE = {metrics_platt['ece']:.4f}, Brier = {metrics_platt['brier_score']:.4f}, NLL = {metrics_platt['nll']:.4f}")
    print(f"Isotonic Scaled:       ECE = {metrics_iso['ece']:.4f}, Brier = {metrics_iso['brier_score']:.4f}, NLL = {metrics_iso['nll']:.4f}")

    # 5. Run Monte Carlo Dropout on Test_60
    checkpoint_path = os.path.join(REPO_ROOT, "model_save", "best_model_27_0.6545215546121214.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(REPO_ROOT, "model_save", "best_model.pth")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPI_Model(liner_dim=115, embbing=128)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if device == "cuda":
        model = model.cuda()

    mc_cache_path = os.path.join(out_dir, "mc_results_cache.npz")
    if os.path.exists(mc_cache_path):
        print(f"Loading cached MC Dropout results from {mc_cache_path}...")
        cached = np.load(mc_cache_path)
        mc_results = {
            "concat_mean_prob": cached["mean_prob"],
            "concat_variance": cached["variance"],
            "concat_std": cached["std"],
            "concat_entropy": cached["entropy"],
            "concat_mi": cached["mi"]
        }
    else:
        test60_dataset = ProDataset(os.path.join(REPO_ROOT, "Datasets", "Test_60.fa"))
        mc_runner = MCDropoutInference(model=model, num_passes=20, device=device)
        mc_results = mc_runner.predict_dataset(test60_dataset)
        np.savez_compressed(
            mc_cache_path,
            mean_prob=mc_results["concat_mean_prob"],
            variance=mc_results["concat_variance"],
            std=mc_results["concat_std"],
            entropy=mc_results["concat_entropy"],
            mi=mc_results["concat_mi"]
        )

    test60_probs_mc = mc_results["concat_mean_prob"]
    metrics_mc = compute_binary_metrics(test60_trues, test60_probs_mc, threshold=0.33)
    print(f"MC-Dropout Mean (T=20): ECE = {metrics_mc['ece']:.4f}, Brier = {metrics_mc['brier_score']:.4f}, NLL = {metrics_mc['nll']:.4f}")

    # 6. Reliability Diagram Data
    rd_raw = compute_reliability_diagram_data(test60_trues, test60_probs_raw, n_bins=10)
    rd_temp = compute_reliability_diagram_data(test60_trues, test60_probs_temp, n_bins=10)
    rd_mc = compute_reliability_diagram_data(test60_trues, test60_probs_mc, n_bins=10)

    plot_reliability_diagrams(
        rd_raw, rd_temp, rd_mc,
        os.path.join(out_dir, "reliability_diagram_comparison.png"),
        metrics_raw["ece"], metrics_temp["ece"], metrics_mc["ece"]
    )

    # 7. Uncertainty Diagnostics and Error Detection
    diag_metrics = plot_uncertainty_diagnostics(
        mc_results, test60_trues, test60_probs_temp, out_dir, threshold=0.33
    )

    print("\n--- Error-Detection AUROC Results ---")
    print(f"Predictive Variance AUROC: {diag_metrics['error_auroc_variance']:.4f}")
    print(f"Predictive Entropy AUROC:  {diag_metrics['error_auroc_entropy']:.4f}")
    print(f"Margin Uncertainty AUROC:  {diag_metrics['error_auroc_margin']:.4f}")

    # 8. Save Comprehensive Predictions CSV
    df_predictions = pd.DataFrame({
        "y_true": test60_trues,
        "prob_raw": test60_probs_raw,
        "prob_temperature_calibrated": test60_probs_temp,
        "prob_platt_calibrated": test60_probs_platt,
        "prob_isotonic_calibrated": test60_probs_iso,
        "prob_mc_dropout_mean": test60_probs_mc,
        "predictive_variance": mc_results["concat_variance"],
        "predictive_std": mc_results["concat_std"],
        "predictive_entropy": mc_results["concat_entropy"],
        "mutual_information": mc_results["concat_mi"]
    })
    df_predictions.to_csv(os.path.join(out_dir, "test60_calibrated_predictions.csv"), index=False)

    # Also save as numpy arrays for downstream Novelty 3 and 4
    np.save(os.path.join(out_dir, "test60_prob_calibrated.npy"), test60_probs_temp)
    np.save(os.path.join(out_dir, "test60_uncertainty_var.npy"), mc_results["concat_variance"])
    np.save(os.path.join(out_dir, "test60_uncertainty_entropy.npy"), mc_results["concat_entropy"])
    np.save(os.path.join(out_dir, "test60_uncertainty_mi.npy"), mc_results["concat_mi"])

    # Save summary JSON
    summary = {
        "experiment": "Novelty_1_Calibrated_Uncertainty",
        "optimal_temperature_T": optimal_T,
        "calibration_set": "Train_335_Val_20pct (66 proteins)",
        "test_benchmark": "Test_60 (60 proteins)",
        "mc_dropout_passes": 20,
        "metrics_raw": metrics_raw,
        "metrics_temperature_calibrated": metrics_temp,
        "metrics_platt_calibrated": metrics_platt,
        "metrics_isotonic_calibrated": metrics_iso,
        "metrics_mc_dropout": metrics_mc,
        "uncertainty_diagnostics": diag_metrics
    }

    with open(os.path.join(out_dir, "novelty1_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nNovelty 1 execution completed. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
