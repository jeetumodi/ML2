# Novelty 2: Partner-Conditioned Interface Prediction Specification

## Experiment Status: `BLOCKED_MISSING_PAIRED_DATA`

### 1. Mandatory Feasibility Audit Finding
A rigorous inspection of all available benchmark datasets (`Test_60.fa`, `Train_335.fa`, `Test_315-28.fa`, `UBtest_31-6.fa`) confirms that they comprise exclusively single-chain records with partner-independent interface ground-truth labels ($y_i \in \{0, 1\}$).

In accordance with strict scientific integrity principles:
> **"Never manufacture partner-specific ground truth from partner-independent residue labels."**
> **"Mark the experiment BLOCKED_MISSING_PAIRED_DATA in the results."**

### 2. Implemented Neural Architecture
The modular partner-conditioned architecture [`CrossProteinEGCPPIS`](file:///C:/Users/jeetu/Desktop/Machine learning/EGCPPIS/src/partner_conditioned/cross_protein_model.py) has been implemented and verified:
- **Intra-chain Encoding:** Equivariant feature mapping for Receptor $A$ and Partner $B$.
- **Inter-chain Geometric Cross-Attention:** Attention weights dynamically penalized by Euclidean distance $D_{AB}$:
  $$A_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d}} - \gamma \cdot \text{softplus}(D_{ij})\right)$$
- **Conditioned Prediction Head:** Combines intrinsic, context, and multiplicative interaction vectors.
- **Unit Test Status:** Passed (Parameters: 46341, Backward gradients active).

### 3. Missing Data Requirements to Unblock Benchmark Training
To evaluate benchmark performance on partner-conditioned prediction without data fabrication, the following biological assets must be integrated:
1. **Co-crystallized Complex Structures:** Full PDB files containing both Receptor Chain $A$ and Interacting Chain $B$ (e.g. from the Dockground or DBPBench benchmarks).
2. **Inter-Chain Distance Tensors:** Pairwise minimum atom distance matrices $D_{AB} \in \mathbb{R}^{L_A \times L_B}$.
3. **Partner-Specific Labels:** Residue labels defined by partner proximity:
   $$y_{A, i}^{(B)} = \mathbb{I}\left[\min_{j \in B} \text{dist}(a_i, b_j) < 5.0\,\text{Angstroms}\right]$$
4. **Homology-Deduplicated Complex Splits:** Train/test partitions clustered at the complex interface level to prevent cross-complex leakage.
