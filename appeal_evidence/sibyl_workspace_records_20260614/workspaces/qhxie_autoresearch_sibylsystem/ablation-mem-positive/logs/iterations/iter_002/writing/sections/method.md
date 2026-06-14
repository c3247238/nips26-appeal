# 3. Method

We formalize feature absorption in Sparse Autoencoders as a quasi-critical phase transition phenomenon. This section presents the theoretical framework, SAE architecture, absorption detection methodology, and experimental procedures.

## 3.1 Sparse Autoencoder Architecture

We follow the standard SAE formulation from prior work (Bricken et al., 2023; Chanin et al., 2024). Given a residual stream activation $x \in \mathbb{R}^d$ from a GPT-2 Small model (d = 768), the SAE encoder computes:

$$f(x) = \max(0, W_{enc}x + b_{enc})$$

where $W_{enc} \in \mathbb{R}^{d_{sae} \times d}$ maps to the SAE latent dimension. The decoder reconstructs the input:

$$\hat{x} = W_{dec}f(x) + b_{dec}$$

Training employs an L1 sparsity penalty to encourage sparse activations:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda \sum_j |f(x)_j|$$

where $\lambda$ is the sparsity penalty coefficient and the sum runs over all $N$ SAE latents (dictionary size).

We use pretrained SAEs from the SAELens library (hook name: `blocks.{layer}.hook_resid_pre`) with dictionary sizes $d_{sae} \in \{6144, 12288, 24576\}$. Layer 6 was identified as an absorption hotspot based on prior work; layer 8 feature-splitting SAEs enable cross-dictionary-size comparison.

## 3.2 Feature Absorption Definition

Feature absorption occurs when a general (parent) feature is subsumed by a more specific (child) feature during sparse optimization. A parent latent $j$ exhibits systematic false negatives: it fails to activate on inputs containing the parent concept when a child concept is also present.

We adopt the training-free absorption detector from Chanin et al. (2024):

$$A_j = \frac{\|d_j\|^2}{d_j^\top e_j}$$

where $d_j \in \mathbb{R}^d$ is the decoder vector and $e_j \in \mathbb{R}^d$ is the encoder vector for SAE latent $j$. High $A_j$ indicates absorption: large decoder norm relative to encoder-decoder alignment suggests the feature has been subsumed by its children.

## 3.3 Phase Transition Framework

We conceptualize feature absorption as a quasi-critical phase transition with analogy to statistical physics.

### Order Parameter and Control Parameter

Let $m(\lambda)$ denote the **order parameter**: mean absorption rate across candidate feature pairs, computed as:

$$m(\lambda) = \frac{1}{N} \sum_{j=1}^{N} \mathbb{1}[A_j > \lambda]$$

where $\lambda$ is the **control parameter** (sparsity penalty coefficient), $\mathbb{1}[\cdot]$ is the indicator function, and $N$ is the dictionary size.

### Critical Point and Susceptibility

The **critical point** $\lambda_c$ is the sparsity value where absorption onset becomes widespread. At this point, the **susceptibility**:

$$\chi = \frac{dm}{d\lambda}$$

exhibits a peak analogous to magnetic susceptibility at a phase transition. We measure susceptibility numerically as $\chi \approx \Delta m / \Delta \lambda$ between adjacent sparsity values.

### Quasi-Critical Behavior

The sharpness of the transition is quantified by:

$$\chi_{ratio} = \frac{\chi_{max}}{\bar{\chi}}$$

where $\chi_{max}$ is the peak susceptibility and $\bar{\chi}$ is the mean susceptibility across all sparsity values. Physical phase transitions exhibit $\chi_{ratio} > 3$ for sharp transitions; we observe $\chi_{ratio} = 1.88$, indicating **quasi-critical** (gradual) behavior rather than a sharp phase transition.

### Finite-Size Scaling

The transition width scales with system size $N$:

$$\delta\lambda \propto N^{-1/\nu}$$

where $\nu$ is the **critical exponent**. For SAE absorption, we test $\nu \in \{1, 2, 3\}$ and measure scaling collapse quality via $R^2$ when rescaling the sparsity axis by $N^{1/\nu}$.

## 3.4 Coefficient of Variation Analysis

The coefficient of variation (CV) characterizes activation variability:

$$CV = \frac{\sigma}{\mu}$$

where $\sigma$ is the standard deviation and $\mu$ is the mean activation magnitude across $n_{samples} = 1000$ text samples. We compute per-feature CV and compare absorbed vs non-absorbed feature distributions.

## 3.5 Co-occurrence Formula

We test a revised co-occurrence formula for predicting absorption:

$$S_{revised} = \cos(d_j, d_k) \cdot \log\left(\frac{f_j}{f_k}\right) \cdot (1 - \rho_j \rho_k)$$

where $\cos(d_j, d_k)$ is decoder cosine similarity, $f_j, f_k$ are activation frequencies, and $\rho_j, \rho_k$ are normalized activation suppressions. We correlate $S_{revised}$ with absorption rate and compare against baseline correlation from prior work.

## 3.6 Graph Topology Construction

We construct an absorption graph $G = (V, E)$ where:
- Nodes $V$ correspond to SAE features ($|V| = N$)
- Edges $E$ represent absorption relationships (parent-child pairs with $A_j > threshold$)

Graph metrics include:
- **Connected components** $C$: number of disconnected subgraphs
- **Giant component size** $S_{giant}$: size of the largest connected component
- **Mean degree** $k_{mean}$: average edge count per node

## 3.7 Experimental Procedure

### Sparsity Sweep (H1: Critical Threshold)

1. Load GPT-2 Small with SAE at layer 6 (`blocks.6.hook_resid_pre`)
2. Sample $n_{samples} = 1000$ text prompts from the OpenWebText corpus
3. For each $\lambda \in \{10^{-5}, 2 \times 10^{-5}, 5 \times 10^{-5}, 10^{-4}, 2 \times 10^{-4}, 5 \times 10^{-4}, 10^{-3}, 2 \times 10^{-3}, 5 \times 10^{-3}, 10^{-2}, 2 \times 10^{-2}, 5 \times 10^{-2}\}$:
   - Compute absorption rate $m(\lambda)$
   - Calculate susceptibility $\chi$
4. Identify $\lambda_c$ as the sparsity with peak $\chi$
5. Compute $\chi_{ratio}$ to assess transition sharpness

### Dictionary Size Sweep (H2: Finite-Size Scaling)

1. Load feature-splitting SAEs at layer 8 with $d_{sae} \in \{6144, 12288, 24576\}$
2. For each dictionary size, sweep sparsity percentiles $\{90, 92, 94, 95, 96, 97, 98, 99\}$
3. Rescale the sparsity axis: $\lambda \times N^{1/\nu}$ for each $\nu \in \{1, 2, 3\}$
4. Compute $R^2$ for scaling collapse quality across all dictionary sizes
5. Select $\nu$ with highest $R^2$ as the critical exponent

### Cross-Layer Measurement (H3: Layer Criticality)

1. Load SAEs at layers $l \in \{0, 3, 6, 9, 11\}$ with $d_{sae} = 24576$
2. Measure absorption rate at $\lambda = 10^{-3}$ (standard sparsity)
3. Test whether layers exhibit heterogeneous absorption (versus uniform saturation)

### CV Analysis (H4: Variance Paradox)

1. Classify features as absorbed ($A_j > 10^{-3}$) or non-absorbed ($A_j \leq 10^{-3}$)
2. Compute per-feature CV across 1000 samples at $\lambda = 5 \times 10^{-5}$
3. Compare CV distributions using t-test with $H_0: \mu_{absorbed} = \mu_{non-absorbed}$

### Co-occurrence Analysis (H5: Information Bottleneck)

1. For feature pairs with $n_{cooccurrence} > 50$:
   - Compute revised co-occurrence score $S_{revised}$
   - Measure absorption rate via Chanin et al. detector
2. Calculate Pearson correlation $r$ between $S_{revised}$ and absorption rate
3. Compare against baseline $r = -0.52$ from prior work

### Graph Topology Analysis (H6: Order Parameter)

1. Construct absorption graphs at each layer at $\lambda = 10^{-3}$
2. Compute connected component count $C$ and giant component size $S_{giant}$
3. Test whether $C$ peaks at layer 6 (hypothesized critical point)

## 3.8 Statistical Methods

We employ standard statistical tests:
- **t-test**: Compare CV distributions between absorbed and non-absorbed features
- **Pearson correlation**: Assess relationship between co-occurrence score and absorption
- **$R^2$ for scaling collapse**: Quantify quality of finite-size scaling universality

All experiments use seed 42 for reproducibility.

<!-- FIGURES
- Figure 1: gen_fig1_phase_transition.py, fig1_phase_transition.pdf — Quasi-critical phase transition with m(λ) curve and susceptibility peak inset
- Figure 2: gen_fig2_scaling_collapse.py, fig2_scaling_collapse.pdf — Finite-size scaling collapse across dictionary sizes (ν=3, R²=0.951)
- Figure 2b: gen_fig2_scaling_collapse.py, fig2b_scaling_nu_comparison.pdf — Scaling collapse quality comparison for ν ∈ {1, 2, 3}
- Figure 3: gen_fig3_cv_comparison.py, fig3_cv_comparison.pdf — Cross-layer CV comparison showing variance paradox (reversed direction)
- Figure 4: gen_fig4_cooccurrence.py, fig4_cooccurrence.pdf — Co-occurrence formula comparison (baseline r=-0.52 vs revised r=0.647)
-->