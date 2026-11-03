# EGCPPIS Comprehensive Case Study: Structural & Residue-Level Evaluation

This case study analyzes the predictive performance of **EGCPPIS** (Equivariant Graph Neural Network with Hierarchical Contrastive Representations) across four benchmark protein chains requested for deep structural evaluation:

| Protein ID | Organism | Biological System / Complex | Benchmark Dataset Split | Length ($L$) | True Interface ($N_{pos}$) |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **`4fq0_B`** | *Helicobacter pylori* | FliM Flagellar Motor Switch Subunit (FliG/FliN complex) | DeepPPISP / Train_335 | 182 aa | 49 aa (26.9%) |
| **`3uvj_A`** | *Homo sapiens* | Soluble Guanylate Cyclase (sGC) $\alpha_1$ Catalytic Domain | Test_60 (Bound) | 191 aa | 48 aa (25.1%) |
| **`6kip_A`** | *Mus musculus* | PTPRD Tyrosine Phosphatase D2 Domain (Liprin-$\alpha3$ complex) | Test_315-28 (GraphPPIS) | 293 aa | 35 aa (11.9%) |
| **`1t6e_X`** | *Triticum aestivum* | TAXI-I Xylanase Inhibitor (Unbound Conformation) | UBtest_31-6 (Unbound) | 362 aa | 39 aa (10.8%) |

---

## 1. Multi-Protein Comparative Benchmark Summary

Below is the comparative performance breakdown across the four proteins evaluated at the standard classification threshold $\tau = 0.33$ alongside optimal $F_1$ threshold tuning and experimental screening budgets (Precision@$K$):

| Evaluation Metric | `4fq0_B` | `3uvj_A` | `6kip_A` | `1t6e_X` |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC (AUROC)** | **0.7662** | **0.8983** | **0.9250** | **0.7777** |
| **PR-AUC (AUPR)** | **0.6636** | **0.8143** | **0.6704** | **0.3538** |
| Random Baseline (Prevalence) | 0.2692 | 0.2513 | 0.1195 | 0.1077 |
| **AUPR Improvement vs Random** | **+2.46×** | **+3.24×** | **+5.61×** | **+3.28×** |
| **Accuracy ($\tau=0.33$)** | 80.22% | 85.34% | 89.76% | 88.12% |
| **Precision ($\tau=0.33$)** | 69.70% | 73.81% | 57.58% | 38.89% |
| **Recall / Sensitivity ($\tau=0.33$)** | 46.94% | 64.58% | 54.29% | 17.95% |
| **Specificity ($\tau=0.33$)** | 92.48% | 92.31% | 94.57% | 96.59% |
| **$F_1$-Score ($\tau=0.33$)** | **0.5610** | **0.6889** | **0.5588** | **0.2456** |
| **MCC ($\tau=0.33$)** | **0.4538** | **0.5958** | **0.5012** | **0.2074** |
| **Optimal Threshold ($\tau_{opt}$)** | 0.36 | 0.19 | 0.20 | 0.21 |
| **Optimal $F_1$-Score** | **0.5823** | **0.7170** | **0.6067** | **0.4752** |
| Optimal Precision / Recall | 76.67% / 46.94% | 65.52% / 79.17% | 50.00% / 77.14% | 38.71% / 61.54% |
| **Precision @ Top-5 (P@5)** | **100.0% (5/5)** | **100.0% (5/5)** | **100.0% (5/5)** | **60.0% (3/5)** |
| **Precision @ Top-10 (P@10)** | **100.0% (10/10)** | **100.0% (10/10)** | **80.0% (8/10)** | **50.0% (5/10)** |
| **Precision @ Top-20 (P@20)** | **95.0% (19/20)** | **95.0% (19/20)** | **70.0% (14/20)** | **40.0% (8/20)** |
| **Recall @ Top-20 (R@20)** | 38.78% | 39.58% | 40.00% | 20.51% |

![Comparative Evaluation Panel](case_study_4protein_comparison.png)

---

## 2. Protein-by-Protein Detailed Case Studies

### 2.1 Case Study 1: `4fq0_B` (*Helicobacter pylori* FliM Flagellar Motor Switch)

#### Biological & Structural Context
- **PDB Complex:** 4FQ0 (Crystal structure of the FliM–FliG flagellar motor switch complex).
- **Subunit Role:** FliM is a core component of the flagellar C-ring rotor responsible for directional switching of bacterial flagellar rotation. Chain B forms an extensive hydrophobic and electrostatic binding surface against the middle domain of FliG.
- **Interface Structure:** Spans amino acid residues 84 to 108, forming an exposed amphipathic $\alpha$-helix and loop cluster directly engaged in torque transmission.

![Prediction Profile for 4fq0_B](case_study_profile_4fq0_B.png)

#### Confusion Matrix ($\tau = 0.33$)
| Total Residues: 182 | Predicted Positive (1) | Predicted Negative (0) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Interface (1)** | **$\text{TP} = 23$** | $\text{FN} = 26$ | 49 |
| **Actual Non-Interface (0)** | $\text{FP} = 10$ | **$\text{TN} = 123$** | 133 |
| **Total** | 33 | 149 | 182 |

#### Top-10 Prioritized Residues
| Rank | Position | Amino Acid | Predicted Probability | Ground Truth | Classification Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 86 | Leu (L) | 0.9759 | 1 | **True Positive (TP)** |
| **2** | 102 | Ile (I) | 0.9742 | 1 | **True Positive (TP)** |
| **3** | 106 | Leu (L) | 0.9510 | 1 | **True Positive (TP)** |
| **4** | 85 | Leu (L) | 0.9380 | 1 | **True Positive (TP)** |
| **5** | 87 | Gly (G) | 0.9374 | 1 | **True Positive (TP)** |
| **6** | 100 | Ser (S) | 0.9155 | 1 | **True Positive (TP)** |
| **7** | 105 | Asn (N) | 0.8863 | 1 | **True Positive (TP)** |
| **8** | 93 | Tyr (Y) | 0.8600 | 1 | **True Positive (TP)** |
| **9** | 29 | Arg (R) | 0.7932 | 1 | **True Positive (TP)** |
| **10** | 84 | Arg (R) | 0.7927 | 1 | **True Positive (TP)** |

---

### 2.2 Case Study 2: `3uvj_A` (*Homo sapiens* Soluble Guanylate Cyclase $\alpha_1$ Domain)

#### Biological & Structural Context
- **PDB Complex:** 3UVJ (Catalytic domain heterodimer of human soluble guanylate cyclase, $\alpha_1\beta_1$).
- **Subunit Role:** Soluble guanylate cyclase (sGC) is the principal intracellular receptor for nitric oxide (NO). The $\alpha_1$ catalytic domain (Chain A) pairs with the $\beta_1$ catalytic domain to form the active catalytic cleft for GTP-to-cGMP cyclization.
- **Interface Structure:** Forms an elongated, extensive contact surface containing active-site coordinating residues and inter-subunit stabilization loops (residues 20–35, 55–65, and 125–135).

![Prediction Profile for 3uvj_A](case_study_profile_3uvj_A.png)

#### Confusion Matrix ($\tau = 0.33$)
| Total Residues: 191 | Predicted Positive (1) | Predicted Negative (0) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Interface (1)** | **$\text{TP} = 31$** | $\text{FN} = 17$ | 48 |
| **Actual Non-Interface (0)** | $\text{FP} = 11$ | **$\text{TN} = 132$** | 143 |
| **Total** | 42 | 149 | 191 |

#### Top-10 Prioritized Residues
| Rank | Position | Amino Acid | Predicted Probability | Ground Truth | Classification Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 30 | Pro (P) | 0.9649 | 1 | **True Positive (TP)** |
| **2** | 34 | Ile (I) | 0.9598 | 1 | **True Positive (TP)** |
| **3** | 59 | Ile (I) | 0.9497 | 1 | **True Positive (TP)** |
| **4** | 127 | Phe (F) | 0.9375 | 1 | **True Positive (TP)** |
| **5** | 60 | Gly (G) | 0.9304 | 1 | **True Positive (TP)** |
| **6** | 25 | Cys (C) | 0.9108 | 1 | **True Positive (TP)** |
| **7** | 132 | Thr (T) | 0.8960 | 1 | **True Positive (TP)** |
| **8** | 58 | Thr (T) | 0.8780 | 1 | **True Positive (TP)** |
| **9** | 22 | Thr (T) | 0.8706 | 1 | **True Positive (TP)** |
| **10** | 31 | Leu (L) | 0.8375 | 1 | **True Positive (TP)** |

---

### 2.3 Case Study 3: `6kip_A` (*Mus musculus* PTPRD Phosphatase D2 Domain)

#### Biological & Structural Context
- **PDB Complex:** 6KIP (PTPRD phosphatase in complex with Liprin-$\alpha3$ tandem SAM domains).
- **Subunit Role:** Receptor protein tyrosine phosphatase delta (PTPRD) is an essential neuronal cell-adhesion and synapse-organizing receptor. The catalytically inactive D2 domain binds Liprin-$\alpha3$ to anchor presynaptic active zone scaffolding.
- **Interface Structure:** Compact, focused surface patch located primarily within the loop and $\alpha$-helix near the C-terminus (residues 210–220, 245–250, and 280–293).

![Prediction Profile for 6kip_A](case_study_profile_6kip_A.png)

#### Confusion Matrix ($\tau = 0.33$)
| Total Residues: 293 | Predicted Positive (1) | Predicted Negative (0) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Interface (1)** | **$\text{TP} = 19$** | $\text{FN} = 16$ | 35 |
| **Actual Non-Interface (0)** | $\text{FP} = 14$ | **$\text{TN} = 244$** | 258 |
| **Total** | 33 | 260 | 293 |

#### Top-10 Prioritized Residues
| Rank | Position | Amino Acid | Predicted Probability | Ground Truth | Classification Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 289 | Phe (F) | 0.8548 | 1 | **True Positive (TP)** |
| **2** | 211 | Lys (K) | 0.7785 | 1 | **True Positive (TP)** |
| **3** | 247 | Arg (R) | 0.7505 | 1 | **True Positive (TP)** |
| **4** | 215 | Gln (Q) | 0.7297 | 1 | **True Positive (TP)** |
| **5** | 214 | Glu (E) | 0.7228 | 1 | **True Positive (TP)** |
| **6** | 285 | Tyr (Y) | 0.6926 | 1 | **True Positive (TP)** |
| **7** | 292 | Tyr (Y) | 0.6388 | 1 | **True Positive (TP)** |
| **8** | 249 | Glu (E) | 0.5896 | 1 | **True Positive (TP)** |
| **9** | 204 | Asp (D) | 0.5524 | 0 | False Positive (FP) |
| **10** | 244 | Glu (E) | 0.5184 | 0 | False Positive (FP) |

---

### 2.4 Case Study 4: `1t6e_X` (*Triticum aestivum* TAXI-I Xylanase Inhibitor — Unbound Form)

#### Biological & Structural Context
- **PDB Entry:** 1T6E (Crystal structure of wheat xylanase inhibitor TAXI-I in its free/unbound conformation).
- **Subunit Role:** TAXI-I is a two-chain plant defense protein that potently inhibits fungal and microbial family 11 endoxylanases to protect plant cell walls.
- **The Unbound Challenge:** In the unbound state, TAXI-I's active loops (especially residues 175–195 and 350–360) are highly flexible and undergo significant rigid-body and side-chain conformation rearrangements upon binding to xylanase. Unbound prediction is notoriously challenging in structural bioinformatics due to these conformational shifts.

![Prediction Profile for 1t6e_X](case_study_profile_1t6e_X.png)

#### Confusion Matrix ($\tau = 0.33$)
| Total Residues: 362 | Predicted Positive (1) | Predicted Negative (0) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Interface (1)** | **$\text{TP} = 7$** | $\text{FN} = 32$ | 39 |
| **Actual Non-Interface (0)** | $\text{FP} = 11$ | **$\text{TN} = 312$** | 323 |
| **Total** | 18 | 344 | 362 |

#### Top-10 Prioritized Residues
| Rank | Position | Amino Acid | Predicted Probability | Ground Truth | Classification Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 188 | Tyr (Y) | 0.4781 | 0 | False Positive (FP) |
| **2** | 354 | Pro (P) | 0.4211 | 1 | **True Positive (TP)** |
| **3** | 182 | Phe (F) | 0.4142 | 1 | **True Positive (TP)** |
| **4** | 180 | Pro (P) | 0.4071 | 1 | **True Positive (TP)** |
| **5** | 192 | Val (V) | 0.3833 | 0 | False Positive (FP) |
| **6** | 157 | Arg (R) | 0.3823 | 0 | False Positive (FP) |
| **7** | 184 | Gln (Q) | 0.3792 | 1 | **True Positive (TP)** |
| **8** | 309 | Lys (K) | 0.3712 | 0 | False Positive (FP) |
| **9** | 166 | Gly (G) | 0.3684 | 1 | **True Positive (TP)** |
| **10** | 292 | Asp (D) | 0.3671 | 0 | False Positive (FP) |

---

## 3. Summary of Findings & Biological Conclusions

1. **High Discrimination Across Divergent Folds:**
   - On the three bound structures (`4fq0_B`, `3uvj_A`, `6kip_A`), EGCPPIS achieves an average **ROC-AUC of 0.8632** and an average **P@10 of 93.3%**.
2. **Robustness of Hierarchical Graph Representation:**
   - By coupling fine-grained atomic GraphSAGE features with residue-level equivariant EGNN geometric updates, EGCPPIS reliably detects structural contact pockets rather than scattered individual false positives.
3. **Wet-Lab Screening Efficiency:**
   - In practical mutagenesis or antibody binding design, wet labs typically validate only 5 to 20 candidate residues. With **100% P@5** on three of four proteins and **60% to 100% P@10**, EGCPPIS substantially minimizes wasted experimental trials.
4. **Actionable Guidance for Unbound Targets:**
   - For unbound proteins (`1t6e_X`) undergoing induced-fit conformational changes, adjusting the operating threshold to $\tau \approx 0.20$ restores high sensitivity ($61.5\%$ recall, $F_1 = 0.4752$) while maintaining candidate ranking fidelity.

