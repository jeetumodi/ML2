# EGCPPIS Research Novelty Methodology

**Date:** October 2026  
**Document:** `doc/NOVELTY_METHODOLOGY.md`  
**Focus:** Formulation, Mathematical Foundations, Algorithmic Specifications, and Evaluation Protocols for the Four Research Novelties  

---

## 1. Overview of Research Extensions

The original EGCPPIS framework formulates residue-level protein–protein interaction site (PPIS) prediction as binary node classification on a protein graph:
$$f_\theta: \mathcal{G} \to [0, 1]^L$$
where $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X}, \mathbf{R})$ captures residue physicochemical features $\mathbf{X} \in \mathbb{R}^{L \times 115}$ and 3D Euclidean coordinates $\mathbf{R} \in \mathbb{R}^{L \times 3}$.

While EGCPPIS demonstrates strong discriminative performance on benchmark sets, standard deep neural networks often suffer from uncalibrated overconfidence, lack of partner conditioning, absence of actionable candidate ranking metrics for wet-lab screening, and fragility under conformational state changes.

We develop and evaluate four research extensions:
1. **Novelty 1:** Post-hoc probability calibration and epistemic uncertainty quantification.
2. **Novelty 2:** Partner-conditioned cross-attention architecture with formal dataset validation and blocker reporting.
3. **Novelty 3:** Uncertainty-guided lower-confidence-bound ranking and fixed screening budget prioritization.
4. **Novelty 4:** Empirical robustness evaluation across conformational shifts (bound vs. unbound), sequence lengths, and interface densities.

---

## 2. Novelty One: Calibrated Uncertainty for PPI-Site Prediction

### 2.1 Post-Hoc Probability Calibration
Standard cross-entropy training encourages overconfident logits near extreme values. For residue $i$ with uncalibrated logit $z_i \in \mathbb{R}$ and raw probability $\hat{p}_i = \sigma(z_i)$, we introduce parametric **Temperature Scaling**:

$$p_{\text{calibrated}, i} = \sigma\left(\frac{z_i}{T^*}\right) = \frac{1}{1 + \exp(-z_i / T^*)}$$

where the scalar temperature $T^* > 0$ is learned exclusively by minimizing the negative log-likelihood (NLL) over a held-out calibration split $\mathcal{D}_{\text{cal}} = \{(z_i, y_i)\}_{i=1}^{N_{\text{cal}}}$:

$$T^* = \arg\min_{T > 0} -\frac{1}{N_{\text{cal}}} \sum_{i=1}^{N_{\text{cal}}} \left[ y_i \log \sigma\left(\frac{z_i}{T}\right) + (1 - y_i) \log\left(1 - \sigma\left(\frac{z_i}{T}\right)\right) \right]$$

**Strict Leakage Constraint:** $\mathcal{D}_{\text{cal}}$ is derived from a disjoint 20% random partition of `Train_335` (66 proteins, 12,507 residues). The test set `Test_60` is strictly excluded from calibration fitting.

In addition to Temperature Scaling, we compare:
* **Platt Scaling:** $p_{\text{cal}, i} = \sigma(a z_i + b)$
* **Isotonic Regression:** Non-parametric piecewise constant isotonic mapping $p_{\text{cal}, i} = g(\hat{p}_i)$

### 2.2 Epistemic Uncertainty via Monte Carlo Dropout
To capture model parameter uncertainty without expensive ensemble retraining, we implement Monte Carlo Dropout sampling. During evaluation, all batch normalization and instance normalization layers remain in evaluation mode, while dropout layers ($\text{rate} = 0.20$) in the MLP classifier and GraphSAGE atom module are kept active.

For $T = 20$ stochastic forward passes, let $p_i^{(t)}$ denote the predicted probability for residue $i$ on pass $t$. We compute:
* **Predictive Mean Probability:**
  $$\bar{p}_i = \frac{1}{T} \sum_{t=1}^T p_i^{(t)}$$
* **Predictive Variance (Epistemic Uncertainty):**
  $$\sigma_{\text{epistemic}, i}^2 = \frac{1}{T} \sum_{t=1}^T \left(p_i^{(t)} - \bar{p}_i\right)^2$$
* **Predictive Standard Deviation:**
  $$\sigma_{\text{epistemic}, i} = \sqrt{\sigma_{\text{epistemic}, i}^2}$$
* **Total Predictive Shannon Entropy:**
  $$H(\bar{p}_i) = -\bar{p}_i \log \bar{p}_i - (1 - \bar{p}_i) \log(1 - \bar{p}_i)$$
* **Mutual Information (Information Gain):**
  $$I(y_i; \theta \mid x_i) = H(\bar{p}_i) - \frac{1}{T}\sum_{t=1}^T H(p_i^{(t)})$$

### 2.3 Uncertainty Evaluation Metrics
1. **Expected Calibration Error (ECE):**
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   with $M = 10$ equal-frequency or equal-width bins.
2. **Brier Score:**
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
3. **Error-Detection AUROC:**
   Binary error indicator $e_i = \mathbb{I}[\hat{y}_i \neq y_i]$. The ROC-AUC of predicting $e_i$ using $\sigma_{\text{epistemic}, i}^2$ or $H(\bar{p}_i)$ evaluates whether uncertainty reliably flags incorrect predictions.
4. **Risk-Coverage Curve (Selective Prediction):**
   Residual classification error rate evaluated when retaining only the top $C \in (0, 1]$ proportion of residues with lowest uncertainty.

---

## 3. Novelty Two: Partner-Conditioned Geometric Interface Prediction

### 3.1 Architectural Specification
Standard PPIS models predict whether a residue belongs to *any* interface. In biological reality, binding is partner-specific. We designed **CrossProteinEGCPPIS**:
* **Input:** Receptor $A$ with $L_A$ residues and interacting partner $B$ with $L_B$ residues, with 3D coordinates $\mathbf{R}_A \in \mathbb{R}^{L_A \times 3}$ and $\mathbf{R}_B \in \mathbb{R}^{L_B \times 3}$.
* **Pairwise Euclidean Distance Matrix:**
  $$D_{ij} = \|\mathbf{R}_{A, i} - \mathbf{R}_{B, j}\|_2 \in \mathbb{R}^{L_A \times L_B}$$
* **Geometric Distance-Biased Cross-Attention:**
  $$S_{ij} = \frac{Q_i K_j^T}{\sqrt{d_k}} - \text{softplus}(\gamma) \cdot D_{ij}$$
  $$A_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^{L_B} \exp(S_{ik})}, \quad C_i = \sum_{j=1}^{L_B} A_{ij} V_j$$
  where $Q = H_A W_Q$, $K = H_B W_K$, $V = H_B W_V$, and $\gamma$ is a learnable spatial decay parameter.
* **Partner-Conditioned Fusion:**
  $$\hat{y}_i^{(B)} = \sigma\left(\text{MLP}\left([H_{A, i} \parallel C_i \parallel H_{A, i} \odot C_i]\right)\right)$$

### 3.2 Dataset Feasibility & Blocker Protocol
Under strict research rules:
* We executed automated auditing via `src/partner_conditioned/data_validator.py`.
* All existing benchmark files (`Test_60.fa`, `Train_335.fa`, `Test_315-28.fa`, `UBtest_31-6.fa`) contain only single-chain entries with partner-independent ground truths.
* **Zero Fabrication Policy:** We do not manufacture partner labels. The experiment is formally classified as:
  $$\text{Status: } \texttt{BLOCKED\_MISSING\_PAIRED\_DATA}$$
* The architecture has been verified via synthetic integration tests (`tests/test_paired_data.py`), confirming that backward gradients and forward passes operate correctly.

---

## 4. Novelty Three: Uncertainty-Guided Candidate Prioritization

Wet-lab validation of interface residues (e.g. via site-directed mutagenesis or alanine scanning) is constrained by experimental screening budgets. We evaluate four candidate prioritization strategies:

### 4.1 Prioritization Strategies
* **Strategy A (Probability-Only Ranking):**
  $$s_i^{(A)} = p_i$$
* **Strategy B (Lower-Confidence-Bound Ranking):**
  $$s_i^{(B)} = p_i - \lambda^* \cdot \sigma_{\text{epistemic}, i}$$
  where $\lambda^* = 0.05$ was selected exclusively on calibration data by maximizing validation Precision@10.
* **Strategy C (Uncertainty Filtering):**
  $$s_i^{(C)} = \begin{cases} p_i & \text{if } \sigma_{\text{epistemic}, i} \le \tau_{\text{unc}} \\ -\infty & \text{if } \sigma_{\text{epistemic}, i} > \tau_{\text{unc}} \end{cases}$$
  where $\tau_{\text{unc}}$ is set to the 90th percentile calibration uncertainty.
* **Strategy D (Budget-Constrained Screening):**
  For each protein $p$ with length $L_p$, candidate residues are ranked descending by score, selecting the top $K \in \{5, 10, 20\}$ residues.

### 4.2 Screening Evaluation Metrics
* **Precision@K:**
  $$\text{Precision@}K = \frac{1}{|P|} \sum_{p=1}^{|P|} \frac{\sum_{i \in \text{Top-}K(p)} y_{p, i}}{K}$$
* **Recall@K:**
  $$\text{Recall@}K = \frac{1}{|P|} \sum_{p=1}^{|P|} \frac{\sum_{i \in \text{Top-}K(p)} y_{p, i}}{N_{\text{pos}}(p)}$$
* **Cumulative Recall vs. Budget Curve:** Tracking mean recall as $K$ increases from 1 to 40 residues.

---

## 5. Novelty Four: Robustness Across Biological Contexts

PPIS prediction models often encounter distribution shift when deployed in practical drug discovery pipelines. We evaluate the model across three biological distribution axes:

1. **Conformational State Shift (Bound vs. Unbound):**
   - Reference: `Test_60` (bound co-crystal structures).
   - Target Shift: `UBtest_31-6` (25 unbound free crystal structures where interface side-chains and surface loops have not undergone induced-fit rearrangement).
2. **Sequence Length Stratification:**
   - Short proteins ($L < 150$ residues)
   - Medium proteins ($150 \le L \le 300$ residues)
   - Long proteins ($L > 300$ residues)
3. **Interface Residue Density (Class Imbalance Variation):**
   - Low density ($< 12\%$ interface residues)
   - Moderate density ($12\% - 18\%$ interface residues)
   - High density ($> 18\%$ interface residues)

For each group $g$, we measure:
$$\Delta \text{AUPR} = \text{AUPR}_g - \text{AUPR}_{\text{ref}}, \quad \Delta \text{MCC} = \text{MCC}_g - \text{MCC}_{\text{ref}}, \quad \Delta \text{ECE} = \text{ECE}_g - \text{ECE}_{\text{ref}}$$
