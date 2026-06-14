# 3. Methodology

## 3.1 Definitions

**Feature Absorption** [Chanin et al., 2024]: A hierarchical parent feature is suppressed when its child feature co-occurs, with the parent activation merged into the child's latent representation to increase sparsity while maintaining reconstruction.

**Feature Collision**: A measurable proxy for absorption. When multiple ground-truth concepts (e.g., first letters 'a', 'i', 'o') activate the same SAE feature index, the collision rate equals the fraction of concepts sharing features. We use collisions as validation ground truth because true absorption requires supervised detection.

**Absorption Signature**: The characteristic co-occurrence pattern of an absorbed parent: high conditional probability $\mathbb{P}(\text{child} \mid \text{parent})$ but low marginal activation frequency. This signature is structurally distinct from independent features (low co-occurrence, high marginals) and correlated features (high co-occurrence, high marginals).

## 3.2 Experimental Setup

**Primary Model.** GPT-2 Small (12 layers, 124M parameters) [Radford et al., 2019]. We extract activations at hook point $\texttt{blocks}.\{l\}.\texttt{hook\_resid\_post}$ using TransformerLens. The SAE is $\texttt{gpt2-small-res-jb}$ (JumpReLU, pretrained, $d_{\text{SAE}} = 24{,}576$) from SAELens.

**Dataset.** OpenWebText [Gokaslan and Cohen, 2019], 1,000 samples for all experiments.

**Seeds.** 42, 123, 456 for multi-seed robustness validation.

**Software.** SAELens >= 2.0, TransformerLens >= 2.0, PyTorch 2.0+, scikit-learn.

## 3.3 UAD: Unsupervised Absorption Detection

**Input.** Pre-trained SAE, unlabeled text corpus.
**Output.** List of suspected absorbed (parent, child) feature pairs.

**Algorithm.** UAD proceeds in six steps:

1. **Activation extraction.** Extract feature activation matrix $A \in \mathbb{R}^{N \times d_{\text{SAE}}}$ from the corpus, where $A_{ni} = \phi_i(x_n)$ is the activation of feature $f_i$ on example $n$.

2. **Co-occurrence computation.** Compute feature co-occurrence matrix $C = A^T A$, where $C_{ij}$ counts how often features $i$ and $j$ activate together on the same input.

3. **Phi coefficient normalization.** Normalize $C$ to phi coefficient correlation matrix $R \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{SAE}}}$. The phi coefficient $\phi_{ij}$ measures association between binary variables (feature active/inactive), equivalent to Pearson correlation for dichotomous data. This normalization accounts for differences in feature activation frequency.

4. **Hierarchical clustering.** Run hierarchical agglomerative clustering (HAC) with Ward linkage on $R$, producing $n_c = 50$ clusters. Ward linkage minimizes variance within clusters, yielding compact, coherent feature groups.

5. **Candidate pair identification.** All same-cluster feature pairs form the candidate set $\mathcal{P}_{\text{cand}}$.

6. **Validation.** Where Chanin-style supervised labels are available, validate candidates against known collision pairs.

**Hyperparameters.** We select the top 500 most active features (by total activation count) from $d_{\text{SAE}} = 24{,}576$ total, and set $n_c = 50$ clusters. Ward linkage minimizes within-cluster variance. Table 4 summarizes.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Top indices | 500 | Most active features from 24,576 total |
| Clusters | 50 | Balances granularity and statistical power |
| Linkage | Ward | Minimizes variance within clusters |

**Key insight.** Absorbed parents show anomalous co-occurrence: they fire primarily when children fire, but rarely independently. This creates a detectable signature in $R$ without ground truth.

Figure 1 illustrates the UAD pipeline from activation extraction to cluster-based pair identification.

![UAD Detection Pipeline](figures/fig1.pdf)

## 3.4 DFDA: Dynamic Feature De-Absorption (Preliminary)

**Input.** SAE, identified absorbed (parent, child) pairs.
**Output.** Compensated SAE output with recovered parent activations.

**Algorithm.** For each pair $(p, c)$:

1. Collect child-dominant examples where the child feature $c$ fires but the parent feature $p$ does not (or fires weakly).
2. Train a tiny compensation MLP: input = child activation $z_c$, output = predicted parent residual $\hat{r}_p$.
3. Architecture: 2 layers, 64 hidden units, ReLU activation:
   $$\hat{r}_p = W_2 \cdot \text{ReLU}(W_1 z_c + b_1) + b_2$$
   where $W_1 \in \mathbb{R}^{64 \times 1}$, $W_2 \in \mathbb{R}^{1 \times 64}$, $b_1 \in \mathbb{R}^{64}$, $b_2 \in \mathbb{R}$. Total: 193 parameters per pair.
4. At inference: add the predicted residual to the parent feature's SAE output:
   $$z_p^{\text{comp}} = z_p + \hat{r}_p$$

**Parameter budget.** 193 parameters per pair, or 0.004% of SAE parameters for 8 pairs.

**Known limitation.** The current evaluation metric (MSE improvement on child-dominant examples) may reflect near-zero prediction rather than true absorption recovery. A parent-positive validation protocol---evaluating on examples where the parent should activate according to ground truth---is required for conclusive validation. DFDA is reported as preliminary work pending this validation.

## 3.5 Validation Protocol

| Stage | Task | Success Criteria |
|-------|------|-----------------|
| Pilot | UAD on GPT-2 Small, layer 8 | F1 >= 0.5 |
| Full | UAD on GPT-2 Small, layer 8 | F1 >= 0.6 |
| Full | UAD multi-seed (3 seeds) | Mean F1 >= 0.6, std <= 0.1 |
| Full | UAD cross-layer (layers 4, 8, 10) | Mean F1 >= 0.5 |
| Full | DFDA on >=8 pairs | Reported with metric caveat |

<!-- FIGURES
- Figure 1: fig1_desc.md — UAD detection pipeline flow diagram (TikZ description)
- Table 4: inline — UAD hyperparameter summary
-->
