# EGCPPIS Reproducibility and Execution Guide

**Date:** October 2026  
**Document:** `doc/REPRODUCIBILITY.md`  
**Repository:** `EGCPPIS`  

---

## 1. System Requirements & Environment Setup

### 1.1 Software & Hardware Specifications
* **Operating System:** Windows 10/11 x86_64 (or Linux / macOS with PyTorch CUDA/MPS support)
* **Python Version:** 3.10.x – 3.14.x
* **PyTorch Version:** PyTorch 2.1.x+ with CUDA support
* **GPU Requirement:** Minimum 6 GB VRAM recommended for fast inference. CPU execution is supported.

### 1.2 Python Dependencies
The core dependencies are:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or appropriate CUDA wheel
pip install torch_geometric einops scikit-learn scipy pandas numpy matplotlib seaborn nbformat nbconvert pytest
```

---

## 2. Checkpoint Verification

The canonical evaluation checkpoint is located at:
`model_save/best_model_27_0.6545215546121214.pth`

Verify checkpoint integrity:
```python
import torch
ckpt = torch.load("model_save/best_model_27_0.6545215546121214.pth", map_location="cpu")
print("Checkpoint loaded successfully. Total tensor parameters:", len(ckpt))
```

---

## 3. Step-by-Step Reproduction Pipeline

All experiments can be reproduced end-to-end using the command line:

### Step 1: Run Baseline Evaluation
Evaluates the author checkpoint on `Test_60` (bound benchmark), `UBtest_31-6` (unbound benchmark), and constructs the 66-protein disjoint calibration split from `Train_335`.
```bash
python src/evaluation/run_baseline.py
```
*Expected Outputs in `results/baseline/`:*
* `baseline_summary.json`
* `test60_per_protein_metrics.csv`
* `test60_roc_curve.png`, `test60_pr_curve.png`, `test60_score_dist.png`
* `test60_y_prob.npy`, `test60_y_true.npy`, `test60_logits.npy`
* `calib_logits.npy`, `calib_y_true.npy`, `calib_y_prob.npy`

---

### Step 2: Run Novelty 1 (Calibrated Uncertainty)
Fits post-hoc Temperature Scaling, Platt Scaling, and Isotonic Regression on the held-out calibration split, and runs 20-pass Monte Carlo Dropout.
```bash
python src/uncertainty/run_novelty1.py
```
*Expected Outputs in `results/novelty_1/`:*
* `novelty1_metrics_summary.json` (Optimal $T^* = 1.8674$, ECE dropped to $0.0225$)
* `test60_calibrated_predictions.csv`
* `reliability_diagram_comparison.png`
* `error_detection_roc.png` (AUROC $= 0.8504$)
* `risk_coverage_curve.png`, `precision_coverage_curve.png`

---

### Step 3: Run Novelty 2 (Partner-Conditioned PPIS Audit & Mechanics)
Audits benchmark files for paired-complex requirements and runs architecture mechanics verification on the modular `CrossProteinEGCPPIS` model.
```bash
python src/partner_conditioned/evaluation.py
```
*Expected Outputs in `results/novelty_2/`:*
* `partner_conditioned_status.json` (Verdict: `BLOCKED_MISSING_PAIRED_DATA`)
* `data_requirements_specification.md`

---

### Step 4: Run Novelty 3 (Candidate Prioritization & Screening Budgets)
Tunes the LCB penalty parameter $\lambda$ on calibration data, evaluates screening budgets ($K=5, 10, 20$) on `Test_60`, and generates a held-out protein case study.
```bash
python src/prioritization/run_novelty3.py
```
*Expected Outputs in `results/novelty_3/`:*
* `ranking_metrics_summary.json` (P@5 $= 0.7967$, P@10 $= 0.7817$, P@20 $= 0.7167$)
* `cumulative_recall_vs_budget.png`
* `precision_at_k_comparison.png`
* `case_study_protein.csv`, `case_study_protein.png`

---

### Step 5: Run Novelty 4 (Robustness & Distribution Shifts)
Evaluates performance across unbound conformational shifts, protein lengths, and interface densities.
```bash
python src/robustness/group_evaluation.py
```
*Expected Outputs in `results/novelty_4/`:*
* `robustness_metrics_summary.json`
* `robustness_group_metrics.csv`
* `robustness_performance_degradation.png`
* `conformation_shift_pr_curves.png`

---

### Step 6: Generate Master Comparison Tables
Aggregates all results into standard comparison tables.
```bash
python src/evaluation/comparison.py
```
*Expected Outputs in `results/comparison/`:*
* `baseline_vs_novelty.csv` (Master comparison table)
* `calibration_metrics.csv`
* `ranking_metrics.csv`
* `robustness_metrics.csv`
* `per_protein_metrics.csv`

---

### Step 7: Run Automated Unit Tests
Executes the test suite covering calibration, uncertainty, ranking, paired data, and metrics:
```bash
python -m pytest tests/
```
*Expected Output:*
```
14 passed in ~3.8s
```

---

### Step 8: Re-Execute Jupyter Notebook
To execute all cells in `notebooks/EGCPPIS_Baseline_vs_Novelty.ipynb` in a clean kernel:
```bash
python -m nbconvert --to notebook --execute --inplace notebooks/EGCPPIS_Baseline_vs_Novelty.ipynb
```
