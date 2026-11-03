# EGCPPIS Novelty Evaluation Rerun (New Model Weights)

**Execution Date:** November 3, 2026  
**Evaluated Model Weights:** [`model_save/best_model_8_0.6406474225359946.pth`](file:///c:/Users/jeetu/Desktop/Machine%20learning/EGCPPIS/model_save/best_model_8_0.6406474225359946.pth) (Trained today; best AUPR: `0.6406`, matching [`checkpoint_latest.pth`](file:///c:/Users/jeetu/Desktop/Machine%20learning/EGCPPIS/model_save/checkpoint_latest.pth))  
**Primary Test Benchmark:** `Test_60` (60 held-out bound proteins, 13,144 residues, prevalence = 15.79%)  
**Calibration Split:** 66 held-out proteins from `Train_335` (12,507 residues, seed 42)  
**Unbound Benchmark:** `UBtest_31-6` (25 unbound proteins, 5,917 residues, prevalence = 12.02%)  

---

## 1. Master Experiment Comparison Table

The unified comparison across the baseline and all four novelties re-executed with the newer weights:

| Experiment ID | Novelty Track | Method / Pipeline Configuration | AUPR | ROC-AUC | MCC | $F_1$ | Accuracy | ECE (Calib Error) | Error AUROC | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | **Baseline** | EGCPPIS Raw Inference (Test_60) | **0.6403** | 0.8845 | 0.5411 | 0.6168 | 87.10% | 0.0308 | 0.8532 | **COMPLETED** |
| **EXP-02** | **Novelty 1** | Post-Hoc Temperature Scaling ($T^*=1.2988$) | **0.6403** | 0.8845 | 0.5343 | 0.6120 | 86.11% | **0.0288** | 0.8532 | **COMPLETED** |
| **EXP-03** | **Novelty 1** | Platt Scaling (Sigmoidal calibration) | **0.6403** | 0.8845 | 0.5433 | 0.6136 | 88.07% | **0.0202** | 0.8532 | **COMPLETED** |
| **EXP-04** | **Novelty 1** | Isotonic Regression (Non-parametric) | 0.6385 | 0.8840 | 0.5390 | 0.6048 | 88.37% | **0.0131** | 0.8532 | **COMPLETED** |
| **EXP-05** | **Novelty 1** | Monte Carlo Dropout ($T=20$ passes) | **0.6415** | **0.8860** | **0.5502** | **0.6247** | **87.16%** | **0.0236** | **0.8459** | **COMPLETED** |
| **EXP-06** | **Novelty 2** | CrossProteinEGCPPIS Cross-Attention | — | — | — | — | — | — | — | **BLOCKED (Data Missing)** |
| **EXP-07** | **Novelty 3** | LCB Prioritization ($\lambda^*=0.05$, Top-10) | **0.6403** | 0.8845 | 0.5411 | 0.6168 | 87.10% | 0.0308 | 0.7675 | **COMPLETED** |
| **EXP-08** | **Novelty 4** | Conformational Shift (UBtest_31-6) | **0.2704** | 0.7130 | 0.2083 | 0.2702 | 86.12% | 0.0681 | 0.7698 | **COMPLETED** |

---

## 2. Novelty 1: Calibrated Uncertainty & Selective Classification

### 2.1 Post-Hoc Calibration Comparison on Test_60
Evaluated on the held-out calibration split (66 proteins, 12,507 residues), the optimal temperature parameter converged to **$T^* = 1.2988$** (improved from $1.8674$ on earlier weights), indicating that the newer model weights produce well-scaled, less overconfident raw logits.

| Calibration Method | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) | Brier Score | Negative Log-Likelihood (NLL) |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Uncalibrated** | 0.0308 | 0.1192 | 0.0861 | 0.3002 |
| **Temperature Scaling ($T^*=1.2988$)** | **0.0288** | **0.0904** | 0.0862 | 0.2889 |
| **Platt Scaling** | **0.0202** | **0.0952** | 0.0854 | **0.2845** |
| **Isotonic Regression** | **0.0131** | **0.0804** | **0.0851** | 0.2883 |
| **MC-Dropout Mean ($T=20$)** | **0.0236** | **0.0995** | **0.0848** | **0.2858** |

### 2.2 Epistemic Uncertainty & Selective Error Detection
Using Monte Carlo Dropout ($T=20$ stochastic forward passes), we extract epistemic uncertainty metrics (predictive variance, predictive entropy, and margin uncertainty):
- **Predictive Entropy Error-Detection AUROC:** **0.8459**
- **Margin Uncertainty Error-Detection AUROC:** **0.8532**
- **Predictive Variance Error-Detection AUROC:** **0.7675**
- **Selective Classification Risk-Coverage:** Rejection filtering yields an empirical risk reduction down to $<0.05$ at 70% coverage.

---

## 3. Novelty 2: Partner-Conditioned PPIS Feasibility & Mechanics

- **Dataset Audit Verdict:** `BLOCKED_MISSING_PAIRED_DATA`
- **Audit Findings:** Benchmark datasets (`Test_60.fa`, `Train_335.fa`, `Test_315-28.fa`, `UBtest_31-6.fa`) contain only single-protein records with partner-independent interface ground-truth labels. In compliance with scientific integrity guidelines, partner ground truth is never fabricated from single-chain labels.
- **Architecture Mechanics Verification:**
  - Implemented modular architecture: [`CrossProteinEGCPPIS`](file:///c:/Users/jeetu/Desktop/Machine%20learning/EGCPPIS/src/partner_conditioned/cross_protein_model.py)
  - Inter-chain geometric cross-attention: $A_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d}} - \gamma \cdot \text{softplus}(D_{ij})\right)$
  - Parameters: 46,341 | Forward & backward pass: **PASSED** (all gradients active).

---

## 4. Novelty 3: Uncertainty-Guided Candidate Prioritization

Evaluating candidate selection under realistic laboratory screening budgets ($K = 5, 10, 20$ mutations):
- **Strategy A:** Probability-Only Ranking.
- **Strategy B:** Lower-Confidence-Bound (LCB) Ranking: $\text{Score}_i = \hat{p}_i - \lambda^* \cdot \sigma_i$ (optimal $\lambda^* = 0.05$ tuned on calibration set).
- **Strategy C:** Epistemic Uncertainty Filtering (discarding highest 20% uncertain residues).

| Screening Budget ($K$) | Strategy A (Prob-Only) | Strategy B (LCB $\lambda^*=0.05$) | Strategy C (Unc-Filtered) | Random Baseline Enrichment |
| :---: | :---: | :---: | :---: | :---: |
| **Top-5 (P@5 / R@5)** | **78.33%** / 12.46% | **78.33%** / 12.46% | **79.00%** / 12.54% | **+4.96× to +5.00×** |
| **Top-10 (P@10 / R@10)** | **78.33%** / 24.93% | **78.17%** / 24.85% | **72.50%** / 22.85% | **+4.96×** |
| **Top-20 (P@20 / R@20)** | **70.83%** / 44.31% | **70.75%** / 44.30% | 59.58% / 36.74% | **+4.48×** |

---

## 5. Novelty 4: Robustness & Structural Generalization

Evaluates performance degradation across biological distribution shifts and structural contexts:

| Biological Stratification | Group Description | $N_{prot}$ | $N_{res}$ | AUPR | Degradation vs Ref ($\Delta\text{AUPR}$) | MCC | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference** | Bound Structures (`Test_60`) | 60 | 13,144 | **0.6403** | Reference Baseline | **0.5343** | 0.0288 |
| **Conformational Shift** | Unbound Structures (`UBtest_31-6`) | 25 | 5,917 | **0.2704** | **$-0.3700$** | **0.2119** | 0.0681 |
| **Length: Short** | $<150$ aa residues | 26 | 2,787 | **0.6379** | $-0.0024$ | 0.4441 | 0.0614 |
| **Length: Medium** | $150$ to $300$ aa residues | 21 | 4,656 | **0.7150** | **$+0.0747$** | **0.5883** | **0.0177** |
| **Length: Long** | $>300$ aa residues | 13 | 5,701 | **0.5917** | $-0.0486$ | 0.5137 | 0.0431 |
| **Density: Low** | Interface ratio $<12\%$ | 13 | 5,383 | **0.6138** | $-0.0265$ | 0.5207 | 0.0553 |
| **Density: Medium** | Interface ratio $12\%$ to $18\%$ | 10 | 2,190 | **0.4705** | $-0.1698$ | 0.4509 | 0.0347 |
| **Density: High** | Interface ratio $>18\%$ | 37 | 5,571 | **0.7071** | **$+0.0667$** | **0.5481** | **0.0270** |

---

## 6. Verification & Automated Test Status

- **Unit Test Suite:** 14/14 automated tests passed in `3.25s` via [`pytest tests/`](file:///c:/Users/jeetu/Desktop/Machine%20learning/EGCPPIS/tests/).
- **Output Artifacts:** Persisted in [`results/comparison/`](file:///c:/Users/jeetu/Desktop/Machine%20learning/EGCPPIS/results/comparison/), [`results/novelty_1/`](file:///c:/Users/jeetu/Desktop/Machine%20learning/EGCPPIS/results/novelty_1/), [`results/novelty_3/`](file:///c:/Users/jeetu/Desktop/Machine%20learning/EGCPPIS/results/novelty_3/), [`results/novelty_4/`](file:///c:/Users/jeetu/Desktop/Machine%20learning/EGCPPIS/results/novelty_4/), and [`results/baseline/`](file:///c:/Users/jeetu/Desktop/Machine%20learning/EGCPPIS/results/baseline/).

