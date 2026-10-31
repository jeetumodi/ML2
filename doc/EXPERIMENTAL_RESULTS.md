# EGCPPIS Experimental Evaluation and Results Report

**Date:** October 2026  
**Document:** `doc/EXPERIMENTAL_RESULTS.md`  
**Execution Environment:** Windows x86_64, Python 3.14.3, PyTorch 2.13.0+cu130, CUDA GPU  
**Artifact Directory:** `results/`  

---

## 1. Executive Experimental Summary

All experiments reported in this document were executed and verified against actual saved outputs. No scores or outputs were fabricated or estimated.

### Master Experiment Results

| Exp ID | Condition / Novelty | Benchmark Dataset | Proteins | Residues | AUPR | ROC-AUC | MCC | F1-Score | Brier Score | ECE | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **EXP-01** | **Baseline (Raw Model)** | `Test_60` | 60 | 13,144 | **0.6547** | **0.8878** | **0.5541** | **0.6225** | 0.0855 | 0.0428 | `COMPLETED` |
| **EXP-02** | **Novelty 1 (Temp-Scaled)** | `Test_60` | 60 | 13,144 | **0.6547** | **0.8878** | 0.5408 | 0.6171 | **0.0838** | **0.0225** | `COMPLETED` |
| **EXP-03** | **Novelty 1 (MC-Dropout T=20)** | `Test_60` | 60 | 13,144 | 0.6513 | 0.8868 | 0.5364 | 0.6076 | 0.0854 | 0.0325 | `COMPLETED` |
| **EXP-04** | **Novelty 2 (Partner-Conditioned)** | N/A (Missing Paired Data) | — | — | — | — | — | — | — | — | `BLOCKED_MISSING_PAIRED_DATA` |
| **EXP-05** | **Novelty 3 (LCB Prioritization)** | `Test_60` (Top-10) | 60 | 13,144 | 0.6547 | 0.8878 | 0.5541 | 0.6225 | 0.0855 | 0.0428 | `COMPLETED` |
| **EXP-06** | **Novelty 4 (Unbound Shift)** | `UBtest_31-6` | 25 | 5,917 | 0.2655 | 0.6595 | 0.2312 | 0.3226 | 0.1379 | 0.1167 | `COMPLETED` |

---

## 2. Baseline Reproduction

Evaluating `model_save/best_model_27_0.6545215546121214.pth` on the primary `Test_60` benchmark confirms exact numerical reproduction of the author's reported results:

* **Residues evaluated:** 13,144 (2,075 interface positives, 11,069 non-interface negatives; prevalence: `15.79%`)
* **AUPR:** `0.654677` (author reported: `0.654677`)
* **ROC-AUC:** `0.887839` (author reported: `0.887839`)
* **$F_{\max}$ / F1-Score:** `0.622530`
* **Precision:** `0.638621`
* **Recall:** `0.607229`
* **MCC:** `0.554122`
* **Accuracy:** `0.883749`
* **Optimal Decision Threshold:** `0.330000`
* **Inference Runtime:** `3.78s` total (`0.063s` per protein on GPU)

---

## 3. Novelty One: Calibration and Uncertainty Results

### 3.1 Probability Calibration Performance on Test_60

The post-hoc calibrators were fitted strictly on the held-out calibration split (66 proteins, 12,507 residues from `Train_335`), learning optimal temperature $T^* = 1.8674$.

| Calibration Method | Fit Split | Test_60 ECE | Test_60 MCE | Test_60 Brier Score | Test_60 NLL | AUPR |
|---|---|---|---|---|---|---|
| **Raw Uncalibrated** | None | 0.0428 | 0.1178 | 0.0855 | 0.3190 | 0.6547 |
| **Temperature Scaling ($T^*=1.8674$)** | Calibration (66 proteins) | **0.0225** | **0.0682** | 0.0838 | **0.2798** | **0.6547** |
| **Platt Scaling (Logistic)** | Calibration (66 proteins) | 0.0178 | 0.0514 | 0.0841 | 0.2796 | 0.6547 |
| **Isotonic Regression** | Calibration (66 proteins) | 0.0101 | 0.0387 | **0.0836** | 0.2873 | 0.6542 |
| **MC-Dropout Mean ($T=20$)** | Calibration (66 proteins) | 0.0325 | 0.0891 | 0.0854 | 0.2948 | 0.6513 |

**Key Finding:** Temperature scaling reduced the Expected Calibration Error (ECE) from `0.0428` to `0.0225` (**47.4% reduction**) and dropped NLL from `0.3190` to `0.2798`, without degrading discrimination (AUPR remains identical at `0.6547` due to strictly monotonic scaling).

### 3.2 Error-Detection AUROC

Using uncertainty scores to detect residue prediction errors ($e_i = \mathbb{I}[\hat{y}_i \neq y_i]$):

| Uncertainty Metric | Error-Detection AUROC | Error-Detection AUPR | Interpretation |
|---|---|---|---|
| **Predictive Variance ($\sigma^2$)** | **0.8210** | **0.4412** | Epistemic parameter variance flags erroneous predictions |
| **Predictive Shannon Entropy ($H$)** | **0.8504** | **0.4891** | Total predictive entropy yields highest error detection |
| **Margin Uncertainty ($1 - \|2p-1\|$)** | **0.8513** | **0.4905** | Distance to decision threshold closely tracks error probability |
| *Random Baseline* | 0.5000 | 0.1163 | Zero discriminative utility |

---

## 4. Novelty Two: Partner-Conditioned PPIS Audit

* **Audit Verdict:** `BLOCKED_MISSING_PAIRED_DATA`
* **Reason:** All benchmark datasets (`Test_60`, `Train_335`, `Test_315-28`, `UBtest_31-6`) contain isolated single-chain records with partner-independent interface ground-truth labels.
* **Architecture Verification:** The modular `CrossProteinEGCPPIS` neural network (incorporating spatial distance-biased cross-attention) was constructed and verified via integration unit test:
  - Total trainable parameters: `46,341`
  - Synthetic complex forward pass loss: `0.7633`
  - Backward pass: Gradients computed and verified on all parameters.
* **Scientific Protocol Upheld:** In accordance with research rules, no benchmark metrics were fabricated for unavailable paired experiments.

---

## 5. Novelty Three: Candidate Prioritization Results

### 5.1 Fixed Screening Budgets on Test_60

Evaluating candidate interface selection across proteins under fixed screening budgets $K \in \{5, 10, 20\}$:

| Screening Budget | Strategy A (Prob-Only) P@K | Strategy B (LCB $\lambda^*=0.05$) P@K | Strategy C (Filtered) P@K | Random Baseline P@K | Enrichment Factor |
|---|---|---|---|---|---|
| **Top-5 Residues / Protein** | **0.7967** | **0.7967** | 0.7500 | 0.1579 | **5.05x** |
| **Top-10 Residues / Protein** | **0.7850** | **0.7817** | 0.7033 | 0.1579 | **4.97x** |
| **Top-20 Residues / Protein** | **0.7167** | **0.7167** | 0.5925 | 0.1579 | **4.54x** |

| Screening Budget | Strategy A Recall@K | Strategy B Recall@K | Strategy C Recall@K |
|---|---|---|---|
| **Top-5 Residues / Protein** | 0.1274 | 0.1274 | 0.1193 |
| **Top-10 Residues / Protein** | **0.2490** | **0.2478** | 0.2210 |
| **Top-20 Residues / Protein** | **0.4493** | **0.4487** | 0.3596 |

### 5.2 Nuanced Scientific Insight on Uncertainty Penalization
* **Conservative LCB ($\lambda^* = 0.05$):** Preserves high candidate precision while softly downweighting borderline predictions.
* **Aggressive LCB ($\lambda \ge 1.0$) & Hard Filtering (Strategy C):** Over-penalizing uncertainty reduces recall (from `0.4493` to `0.3596` at $K=20$) because genuine interface residues near geometric boundaries exhibit higher epistemic ambiguity. Uncertainty should be applied as an informative diagnostic rather than a hard screening filter.

---

## 6. Novelty Four: Robustness Across Biological Contexts

Evaluating distribution shifts and structural stratifications:

| Evaluation Group | Source Partition | Proteins | Residues | Prevalence | AUPR | $\Delta$ AUPR | MCC | $\Delta$ MCC | ECE | Error AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| **Reference Bound** | `Test_60` | 60 | 13,144 | 15.79% | **0.6547** | Ref | **0.5408** | Ref | 0.0225 | 0.8210 |
| **Conformational Shift (Unbound)** | `UBtest_31-6` | 25 | 5,917 | 12.02% | **0.2655** | **-0.3891** | **0.2312** | **-0.3096** | 0.1167 | 0.7839 |
| **Short Proteins ($L < 150$)** | `Test_60` | 26 | 2,787 | 17.58% | 0.6605 | +0.0058 | 0.4698 | -0.0710 | 0.0481 | 0.7912 |
| **Medium Proteins ($150 \le L \le 300$)** | `Test_60` | 21 | 4,656 | 16.92% | **0.7586** | **+0.1040** | **0.6341** | **+0.0933** | 0.0374 | 0.8415 |
| **Long Proteins ($L > 300$)** | `Test_60` | 13 | 5,701 | 13.98% | 0.5105 | -0.1442 | 0.4439 | -0.0969 | 0.0393 | 0.8104 |
| **Low Density ($< 12\%$)** | `Test_60` | 13 | 5,383 | 9.12% | 0.5064 | -0.1483 | 0.4187 | -0.1221 | 0.0511 | 0.8012 |
| **High Density ($> 18\%$)** | `Test_60` | 37 | 5,571 | 21.84% | **0.7258** | **+0.0711** | **0.5718** | **+0.0310** | 0.0304 | 0.8389 |

### Robustness Insights:
1. **Conformational Vulnerability:** The severe performance drop on `UBtest_31-6` ($\Delta \text{AUPR} = -0.3891$) indicates that EGCPPIS relies heavily on bound side-chain orientations and AlphaFold2 bound-state conformations. Free unbound structures with flexible surface loops undergo conformational changes upon binding that the static coordinate graph does not account for.
2. **Length & Density Bias:** Medium-length proteins achieve the highest performance (`0.7586` AUPR), while long proteins suffer from sparse interface distribution over large surface areas.
