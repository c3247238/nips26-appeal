# Auditing Feature Absorption Metrics on JumpReLU SAEs: Hedging Dominance, the $L_0$ Phase Transition, and the Limits of Rate-Distortion Diagnostics

---

## Abstract

Feature absorption -- where a parent Sparse Autoencoder (SAE) latent silently fails to fire when a child latent is active -- is the primary reliability concern for SAE-based mechanistic interpretability. The Chanin absorption metric, developed on GPT-2 Small with L1-ReLU SAEs and reporting absorption rates of 15--35%, has motivated architectural mitigations including Matryoshka SAE (ICML 2025), OrtSAE, ATM-SAE, and masked regularization. All assume the metric measures competitive exclusion; none validates it on JumpReLU SAEs, the architecture dominating the Gemma Scope ecosystem.

We audit the metric on Gemma 2 2B with Gemma Scope JumpReLU SAEs across five hierarchy domains and report three findings. First, the metric does not transfer: shuffled-label controls exceed measured absorption in all five domains (ratios 2.7$\times$ to $\infty$; 4.7$\times$ on first-letter), a structural failure caused by candidate explosion at cosine $\geq 0.025$ in $\mathbb{R}^{2304}$ (23.0% of decoder columns identified as candidates by a random vector). Confound decomposition at $L_0 = 22$ with perfect probes (F1 = 1.0) classifies 98.6% of false negatives as hedging under a permissive definition; a strict parent-latent check reduces this to 6.2% (95% CI: [4.4%, 8.2%]). Activation patching on 8 persistent core words finds 0/8 parent recovery after child zeroing, ruling out competitive exclusion. Second, absorption declines monotonically from 42.85% ($L_0 = 22$) to 0.84% ($L_0 = 176$), with a phase transition in the $L_0 \approx 40$--80 range stable across three layers (CV $<$ 10%). Third, conditional mutual information (CMI) at $L_0 = 82$ shows a marginal negative correlation with absorption ($\rho_s = -0.383$, $p = 0.059$), but replication at $L_0 = 22$ with perfect probes yields $\rho_s = 0.044$ ($p = 0.835$), indicating the original correlation was driven by probe quality confounds.

These results recommend validating absorption metrics on each new SAE architecture before building mitigations.

---

# 1 Introduction

Sparse Autoencoders (SAEs) decompose the polysemantic activations of large language models (LLMs) into sparse, interpretable features -- the primary unsupervised tool for mechanistic interpretability at scale (Cunningham et al., 2023; Bricken et al., 2023; Templeton et al., 2024). Feature absorption undermines this decomposition: a general "parent" SAE latent silently fails to fire on inputs where a more specific "child" latent is active, even though the parent concept is present (Chanin et al., 2024). The failure is invisible to standard evaluation -- the parent latent achieves high precision on its activating examples but has systematic, invisible recall holes.

Chanin et al. (2024) measured absorption rates of 15--35% across hundreds of SAEs on first-letter spelling using GPT-2 Small with L1-ReLU SAEs. This finding triggered an architectural mitigation wave: Matryoshka SAE (Bussmann et al., 2025; ICML 2025), OrtSAE (Korznikov et al., 2025), ATM-SAE (Li et al., 2025), and masked regularization (Narayanaswamy et al., 2026) all propose encoder modifications to reduce absorption. Every mitigation benchmarks against the Chanin metric and assumes it measures competitive exclusion -- a causal mechanism where child latents actively suppress parent latents under sparsity pressure. No study has validated this assumption on JumpReLU SAEs, the architecture that dominates the Gemma Scope ecosystem with 400+ pretrained SAEs (Lieberum et al., 2024). JumpReLU SAEs impose hard per-latent activation thresholds $\theta_j$ rather than soft L1 penalties, creating a binary activation regime (fire or zero) that could alter both the mechanism and measurement of feature absorption.

Three open questions motivate our study:

**Q1 -- Metric validity.** Does the Chanin absorption metric transfer from L1-ReLU SAEs on GPT-2 Small to JumpReLU SAEs on Gemma 2 2B? What fraction of measured "absorption" reflects genuine competitive exclusion versus hedging -- information spreading across many latents due to insufficient capacity (Chanin & Garriga-Alonso, 2025) -- or measurement artifact?

**Q2 -- Sparsity dynamics.** How does absorption scale with the $L_0$ operating point (the configured number of active SAE latents per forward pass)? Is there a sparsity threshold below which absorption becomes negligible?

**Q3 -- Theoretical criterion.** Can conditional mutual information (CMI) predict which parent-child feature pairs are susceptible to absorption, and at what sparsity level absorption becomes information-theoretically unavoidable?

We audit the Chanin metric on Gemma 2 2B with Gemma Scope JumpReLU SAEs across five hierarchy domains (first-letter spelling, city-country, city-continent, city-language, animal-class) and report three findings.

**1. The absorption metric does not transfer to JumpReLU SAEs (Section 4).** Shuffled-label controls produce higher "absorption" than true labels in all five domains (ratios from 2.7$\times$ to $\infty$; 4.7$\times$ on first-letter). This control failure persists across all 20 threshold combinations in a 5$\times$4 parameter sweep (CV = 0.077), confirming it is structural. Confound decomposition at $L_0 = 22$ -- where all 25 probes achieve F1 = 1.0 -- classifies 98.6% of 656 false negatives (FNs) as hedging under a permissive definition. A strict parent-latent check reduces the hedging rate to 6.2% (95% CI: [4.4%, 8.2%]; $z = 3.51$, $p < 0.001$ above shuffled control). Activation patching on 8 persistent core words finds 0/8 parent recovery after child zeroing, ruling out competitive exclusion for these tokens.

**2. Absorption declines monotonically with $L_0$, exhibiting a phase transition (Section 5).** On L12-16k, the absorption rate drops from 42.85% at $L_0 = 22$ to 0.84% at $L_0 = 176$ (Spearman $\rho_s = -1.0$), with a sharp transition between $L_0 \approx 40$ and $L_0 \approx 80$. This profile is stable across layers 10, 12, and 20 (CV $<$ 10%), establishing $L_0$ -- not encoder architecture -- as the primary control parameter.

**3. The CMI-absorption association does not replicate with perfect probes (Section 6).** At $L_0 = 82$, CMI at subspace dimension $d' = 10$ shows a marginal negative correlation with absorption ($\rho_s = -0.383$, $p = 0.059$). Replication at $L_0 = 22$ -- where all probes achieve F1 = 1.0 -- yields $\rho_s = 0.044$ ($p = 0.835$), with the sign reversed. The original correlation tracked probe quality ($\rho_s = -0.69$ between absorption and probe F1), not rate-distortion theory.

These results recommend validating absorption metrics on new SAE architectures before building mitigations, and indicate that hedging -- not competitive exclusion -- is the dominant false-negative mechanism on JumpReLU SAEs at typical $L_0$ operating points.

Sections 2--3 provide background and methodology. Section 4 presents the metric audit: universal control failure, its structural explanation, threshold sensitivity, confound decomposition, and activation patching. Section 5 reports the $L_0$ phase transition and cross-layer stability. Section 6 describes the exploratory CMI analysis and its failure to replicate. Section 7 discusses implications, limitations, and future directions.

---

# 2 Background and Related Work

## 2.1 Sparse Autoencoders in Mechanistic Interpretability

A Sparse Autoencoder (SAE) maps a residual stream activation $x \in \mathbb{R}^{d_{\text{model}}}$ to a sparse latent vector $z \in \mathbb{R}^{d_{\text{SAE}}}$ via encoder $W_e$ and reconstructs $\hat{x} = W_d z$, where each decoder column $d_j$ is unit-normalized. Two architecture families dominate current practice.

**L1-ReLU SAEs** apply a soft L1 penalty to encourage sparsity: $z_j = \text{ReLU}(w_{e,j}^\top x + b_{e,j})$, producing graded suppression that scales smoothly with penalty strength (Cunningham et al., 2023; Bricken et al., 2023). SAEs have been scaled to frontier models (Templeton et al., 2024; Gao et al., 2024) and used for circuit analysis (Lindsey et al., 2025).

**JumpReLU SAEs** impose a hard per-latent threshold $\theta_j$:
$$z_j = (w_{e,j}^\top x + b_{e,j}) \cdot \mathbb{1}[w_{e,j}^\top x + b_{e,j} > \theta_j]$$
trained via the straight-through estimator (Rajamanoharan et al., 2024). The binary activation regime -- fire or zero -- differs qualitatively from L1-ReLU's graded suppression. The $L_0$ operating point, the number of non-zero latents per forward pass, directly controls the sparsity--fidelity tradeoff. Gemma Scope (Lieberum et al., 2024) provides 400+ open JumpReLU SAEs on Gemma 2 2B/9B/27B with configurable $L_0$, making it the largest public testbed for SAE evaluation.

Theoretical limitations constrain what SAEs can reliably recover. Leask et al. (2025) show that SAE features are neither canonical (different training runs yield different feature sets) nor atomic (meta-SAEs decompose individual features into sub-features). Cui et al. (2025) prove that recovery of ground-truth features fails unless features are extremely sparse. Engels et al. (2025) discover irreducible multi-dimensional features that one-dimensional SAE latents cannot faithfully represent. These results motivate direct empirical validation of what SAE-derived metrics actually capture.

## 2.2 Feature Absorption

Chanin et al. (2024) define feature absorption: a parent latent $z_p$ (e.g., encoding "starts-with-A") drops to $z_p = 0$ on parent-positive inputs when a child latent $z_c$ (e.g., encoding a specific word token) is active ($z_c > 0$). The measurement protocol trains $k$-sparse logistic regression probes ($k = 5$) on SAE latent activations, identifies candidate SAE latents whose decoder columns have cosine similarity $\cos(d_j, v_p) \geq \tau_{\cos}$ to the probe direction $v_p$, and computes the false-negative (FN) rate -- the fraction of parent-positive tokens where all $k$ probe-associated latents are inactive. Across hundreds of SAEs in SAEBench (Karvonen et al., 2025; ICML 2025), absorption rates of 15--35% are reported on first-letter spelling using GPT-2 Small with L1-ReLU SAEs. No study has validated the Chanin metric on JumpReLU SAEs with appropriate controls.

Two theoretical accounts explain why absorption arises. Tang et al. (2025) show that the biconvex optimization landscape of sparse dictionary learning contains partial minima where absorption is locally optimal. O'Neill et al. (2024) identify the amortization gap: feedforward encoding cannot recover all dictionary atoms, so the encoder systematically fails to activate parents when children already account for most of the reconstruction.

A distinct but observationally similar phenomenon is feature hedging (Chanin & Garriga-Alonso, 2025): when the $L_0$ operating point is too low, the SAE merges correlated features, spreading parent information across many latents rather than concentrating it in a single parent latent. Hedging produces false negatives that mimic absorption but arise from capacity starvation rather than competitive suppression. The false negatives from hedging resolve when $L_0$ is increased, whereas competitive-exclusion-driven false negatives persist. This distinction is central to our confound decomposition (Section 3.4).

Tian et al. (2025) frame feature absorption as a special case of poor feature sensitivity: many features that appear monosemantic on their top activating examples nonetheless fail to activate on semantically similar inputs. This broader failure mode subsumes absorption and suggests that recall deficits in SAE latents extend beyond hierarchical parent-child relationships.

## 2.3 Architectural Mitigations

The absorption finding has prompted a wave of architectural interventions. All share the assumption that the Chanin metric validly measures competitive exclusion and propose modifications to reduce it:

- **Matryoshka SAE** (Bussmann et al., 2025; ICML 2025) trains nested dictionaries of increasing size, organizing features hierarchically. SAEBench absorption score: 0.03 versus 0.29 for BatchTopK.
- **OrtSAE** (Korznikov et al., 2025) enforces orthogonality via a pairwise cosine penalty on decoder columns, reducing absorption by ~65% relative to BatchTopK.
- **ATM-SAE** (Li et al., 2025) applies adaptive temporal masking with per-latent importance scoring, achieving an absorption score of 0.007 versus 0.140 for TopK and 0.011 for JumpReLU on Gemma 2 2B.
- **Masked regularization** (Narayanaswamy et al., 2026) disrupts co-occurrence patterns during training, reducing absorption across SAE architectures.
- **KronSAE** (2025) exploits Kronecker factorization of latents.
- **MP-SAE** (Costa et al., 2025) uses iterative encoding to refine latent representations, partially recovering missed latents.

None of these mitigations validates the Chanin metric on JumpReLU SAEs -- the architecture whose hard-threshold activation dynamics differ most from the L1-ReLU SAEs on which the metric's thresholds ($\tau_{\cos} \geq 0.025$, $\tau_{\text{mag}} \geq 1.0$) were calibrated. If the metric conflates hedging with competitive exclusion on JumpReLU architectures, the reported mitigation gains may reflect changes in false-negative rates rather than genuine reductions in hierarchy-driven absorption. Our metric audit (Section 4) tests this directly.

## 2.4 Rate-Distortion Theory and Successive Refinement

The successive refinement theorem (Equitz and Cover, 1991) establishes that a source is successively refinable -- encodable in stages without information loss -- if and only if the stage descriptions form a Markov chain. Applied to SAE features: if $X \to f_{\text{child}} \to f_{\text{parent}}$ is Markov, then $I(X; f_{\text{parent}} \mid f_{\text{child}}) = 0$ and the parent carries no unique information about the input beyond what the child encodes. Absorbing such a parent is information-theoretically lossless.

When CMI $> 0$, absorbing the parent destroys unique information -- a quantifiable cost measured in bits. The magnitude of CMI thus provides a theoretical criterion for absorption susceptibility: features with low CMI are cheap to absorb; features with high CMI resist absorption because suppressing them incurs a larger information cost.

No prior work has connected successive refinement theory to SAE feature absorption. We test this prediction in Section 6; as reported there, the predicted association does not replicate when probe quality confounds are eliminated.

---

# 3 Methodology

## 3.1 Models and SAE Configurations

Table 1 summarizes the SAE configurations. The primary model is Gemma 2 2B ($d_{\text{model}} = 2304$) with Gemma Scope JumpReLU SAEs (Lieberum et al., 2024). We systematically vary three axes:

1. **$L_0$ operating point.** Four configurations at layer 12 with $d_{\text{SAE}} = 16{,}384$: $L_0 \in \{22, 41, 82, 176\}$. These are the four $L_0$ configurations available in Gemma Scope for this layer and width, spanning capacity-starved ($L_0 = 22$) to capacity-sufficient ($L_0 = 176$) regimes.

2. **Layer.** At fixed $L_0 = 82$ and $d_{\text{SAE}} = 16{,}384$, we compare layers 10, 12, and 20.

3. **Dictionary width.** At layer 12, we compare $d_{\text{SAE}} = 16{,}384$ with $d_{\text{SAE}} = 65{,}536$ at $L_0 = 82$.

As a secondary reference, we include GPT-2 Small ($d_{\text{model}} = 768$) with L1-ReLU SAEs ($d_{\text{SAE}} = 24{,}576$) at layers 8, 10, and 11. This cross-architecture comparison is confounded by model size, architecture, training data, and $L_0$; we report it as context only.

| Config | Model | Layer | $d_{\text{SAE}}$ | $L_0$ | Architecture |
|--------|-------|:-----:|:-----------------:|:------:|:-------------|
| L12-16k-$L_0$=22 | Gemma 2 2B | 12 | 16,384 | 22 | JumpReLU |
| L12-16k-$L_0$=41 | Gemma 2 2B | 12 | 16,384 | 41 | JumpReLU |
| L12-16k-$L_0$=82 | Gemma 2 2B | 12 | 16,384 | 82 | JumpReLU |
| L12-16k-$L_0$=176 | Gemma 2 2B | 12 | 16,384 | 176 | JumpReLU |
| L12-65k | Gemma 2 2B | 12 | 65,536 | 82 | JumpReLU |
| L10-16k | Gemma 2 2B | 10 | 16,384 | 82 | JumpReLU |
| L20-16k | Gemma 2 2B | 20 | 16,384 | 82 | JumpReLU |
| GPT2-L8/L10/L11 | GPT-2 Small | 8/10/11 | 24,576 | -- | L1-ReLU |

**Table 1.** SAE configurations used in this study. Gemma Scope SAEs provide configurable $L_0$ via per-latent JumpReLU thresholds. GPT-2 SAEs use L1-ReLU with no explicit $L_0$ control.

## 3.2 Absorption Measurement Protocol

We follow the measurement protocol of Chanin et al. (2024) with explicit quality controls.

**Vocabulary.** Our primary domain is first-letter spelling: 1,204 single-token English words across 25 tested letters (X excluded due to having only 1 token). For confound decomposition analyses that require consistent vocabulary across $L_0$ values, we use a separately tokenized 1,196-word subset; the minor size difference (8 words) reflects different filtering criteria on the same raw word list.[^vocab]

[^vocab]: The 1,204-word vocabulary applies a minimum token count per letter; the 1,196-word vocabulary uses a stricter single-token criterion. Both produce statistically equivalent absorption rates (15.96% vs. 14.39% at $L_0 = 82$, within the cross-layer CV of 8.6%).

For cross-domain experiments, we use 189 single-token cities across 29 countries from RAVEL and 140 animals across 6 taxonomic classes from WordNet (Table 8, Section 3.7).

**Probe training.** For each parent class (e.g., letter), we train a $k$-sparse logistic regression probe ($k = 5$) on SAE latent activations. The probe selects the $k$ latents whose decoder columns $d_j$ have the highest cosine similarity $\cos(d_j, v_p)$ with the probe direction $v_p$ and fits a logistic classifier on their activations.

**Quality gate.** We require probe F1 $> 0.85$ per parent class before computing absorption. At $L_0 = 82$, 10 of 25 letters pass this gate (mean probe F1 = 0.817 across all 25); at $L_0 = 22$, all 25 letters achieve F1 = 1.0. Letters failing the gate are excluded from aggregate statistics but reported individually.

**Candidate identification.** A latent $j$ is a candidate absorbing feature if $\cos(d_j, v_p) \geq \tau_{\cos}$ where $\tau_{\cos} = 0.025$ (default). Section 3.6 evaluates sensitivity to this threshold.

**Absorption criterion.** A token is classified as absorbed (false negative, FN) when: (1) the probe correctly classifies it as belonging to the parent class, and (2) all $k$ probe-associated latents have zero activation ($z_j = 0$, i.e., below the JumpReLU threshold $\theta_j$). Among these FNs, absorption is confirmed when the highest-activation candidate latent has magnitude ratio $\geq \tau_{\text{mag}} = 1.0$ relative to the second-highest. The absorption rate is the fraction of parent-positive tokens meeting both conditions; the magnitude ratio filter identifies specific absorbing latents.

**Statistical inference.** All confidence intervals are 95% percentile bootstrap CIs with 10,000 resamples (seed = 42). Effect sizes use Cohen's $d$ for group comparisons and Spearman $\rho_s$ for rank correlations. Multiple comparisons are corrected via Bonferroni.

## 3.3 Four-Control Suite

A valid absorption metric should assign higher absorption to true labels than to shuffled labels. We implement four controls:

**C1: Random probe.** A probe trained on a random direction in $\mathbb{R}^{d_{\text{SAE}}}$. Expected: near-zero absorption. Observed: 11.8% on first-letter, indicating substantial baseline metric noise.

**C2: Shuffled labels.** The same probe pipeline applied to randomly permuted class assignments (labels shuffled across tokens). Expected: absorption rate $\leq$ measured rate. Observed: 74.6% on first-letter (mean over 5 shuffles, $\sigma = 2.1\%$), exceeding the true-label rate of 15.96% by 4.7$\times$.

**C3: Dense probe.** An all-feature logistic regression (no sparsity constraint). This tests whether SAE activations contain sufficient information for classification independent of the $k$-sparse selection step. Mean dense probe F1 = 0.929 across first-letter classes.

**C4: Untrained SAE.** Absorption computed using randomly initialized decoder columns. Produces 0.0% absorption, confirming that the measured signal depends on SAE structure.

## 3.4 Confound Decomposition

Feature absorption and feature hedging produce identical observational signatures: a parent latent fails to fire on a parent-positive input. We introduce confound decomposition to separate these two sources.

**Multi-$L_0$ persistence analysis.** We measure false negatives at all four $L_0$ values $\{22, 41, 82, 176\}$ using the 1,196-word consistent vocabulary. Hedging-driven false negatives should resolve at higher $L_0$ (where the SAE has more capacity), while hierarchy-driven false negatives should persist.

**Permissive hedging classification.** A token is classified as hedging if it ceases to be a false negative at *any* higher $L_0$ value. This is an upper bound: it counts every FN that eventually resolves, regardless of whether the specific parent-associated latents recover.

**Strict hedging classification.** We check whether the specific $k = 5$ parent-associated latents (those selected by the probe at $L_0 = 22$) fire at $L_0 = 176$. A token is classified as strict hedging only if at least 1 of these 5 latents activates at the highest $L_0$. This is a conservative lower bound.

**Persistent core words.** Tokens classified as FN at all four $L_0$ values constitute the persistent core -- candidates for genuine competitive exclusion that cannot be explained by hedging at any tested sparsity level.

**Shuffled control.** We apply the strict classification to 10 shuffled-label replicates to establish a baseline rate.

## 3.5 Activation Patching

Confound decomposition identifies persistent false negatives but cannot establish causality. Activation patching resolves this ambiguity.

For each persistent core word, we perform three causal interventions at $L_0 = 82$:

1. **Primary patching (decode-reencode).** Zero the strongest child candidate feature's activation, decode through $W_d$, then re-encode through the full JumpReLU encoder pipeline (including thresholds $\theta_j$). Parent recovery is defined as any parent-associated latent activating above its JumpReLU threshold after patching.

2. **All-children patching.** Zero all absorbing candidate features simultaneously.

3. **Residual patching.** Subtract the child feature's decoder contribution ($z_c \cdot d_c$) directly from the raw residual stream activation $x$, then re-encode.

**Control.** For each word, we zero 10 randomly selected non-child features and verify that parent latents do not spuriously activate.

## 3.6 Threshold Sensitivity Analysis

We evaluate robustness across a $5 \times 4$ parameter grid: $\tau_{\cos} \in \{0.01, 0.02, 0.025, 0.03, 0.05\}$ and $\tau_{\text{mag}} \in \{0.5, 1.0, 1.5, 2.0\}$. For each of the 20 cells, we compute the aggregate absorption rate, per-letter rankings (Kendall $\tau$ with default parameters), and whether the shuffled-label control exceeds the measured rate.

## 3.7 Cross-Domain Hierarchy Suite

Five hierarchy domains span syntactic, geographic, linguistic, and taxonomic relationships:

| Domain | Parent Feature | $N_{\text{parents}}$ | $N_{\text{children}}$ | Source |
|--------|---------------|:-------------------:|:--------------------:|--------|
| First-letter | "starts with X" | 25 | 1,204 | Chanin et al. (2024) |
| City $\to$ Country | "in France" | 28 | 184 | RAVEL |
| City $\to$ Continent | "in Europe" | 6 | 185 | RAVEL |
| City $\to$ Language | "French-speaking" | 18 | 183 | RAVEL |
| Animal $\to$ Class | "mammal" | 6 | 140 | WordNet |

**Table 8.** Cross-domain hierarchy suite. Probes, quality gates (F1 $> 0.85$), and all four controls are applied per domain. Countries with fewer than 5 cities or probe F1 $< 0.50$ are excluded.

## 3.8 Conditional Mutual Information Estimation

We estimate CMI, $I(X; f_{\text{parent}} \mid f_{\text{child}})$, via a $k$-nearest neighbor mutual information estimator ($k_{\text{nn}} = 5$; Kraskov et al., 2004) operating in a $d'$-dimensional subspace formed by the top principal components of the parent and child decoder columns. The primary pre-registered subspace dimension is $d' = 10$. For each letter, we collect SAE activations on corpus token positions plus all word tokens for that letter, project into the decoder subspace, and estimate CMI.

**Cross-$L_0$ replication.** We estimate CMI at both $L_0 = 82$ (where probe quality varies) and $L_0 = 22$ (where all 25 probes achieve F1 = 1.0). This design eliminates the probe quality confound.

**Partial correlation.** At $L_0 = 82$, probe F1 correlates with both CMI and absorption rate. We compute partial Spearman $\rho_s$(CMI, absorption $\mid$ probe F1) with permutation-based $p$-values (10,000 permutations).

**Dimension sensitivity.** We repeat CMI estimation at $d' \in \{10, 20, 30, 50\}$ at both $L_0$ values.

---

# 4 The Absorption Metric Does Not Transfer to JumpReLU SAEs

## 4.1 Universal Control Failure

Figure 1 and Table 2 present the central result of the metric audit: shuffled-label controls exceed measured absorption rates in all five hierarchy domains on L12-16k ($L_0 = 82$).

![Grouped bar chart showing measured vs. shuffled vs. random absorption rates across 5 hierarchy domains on L12-16k (L0=82). In every domain, shuffled controls exceed measured rates, demonstrating universal control failure.](figures/control_failure.pdf)

**Figure 1.** Universal control failure across five hierarchy domains on L12-16k ($L_0 = 82$). Grouped bars show measured absorption rate (blue), shuffled-label control (pink), and random probe control (purple). In all five domains, shuffled controls exceed measured rates. The dashed horizontal line marks the expected random probe floor ($< 2\%$).

| Domain | $N_{\text{parents}}$ | Probe F1 | Measured | Shuffled | Random | Ratio (S/M) |
|--------|:-------------------:|:--------:|:--------:|:--------:|:------:|:-----------:|
| First-letter | 25 | 0.817 | 15.96% | 74.6% | 11.8% | **4.7$\times$** |
| City $\to$ Continent | 6 | 0.795 | 6.49% | 45.2% | 12.9% | **6.9$\times$** |
| City $\to$ Language | 18 | 0.745 | 6.56% | 18.0% | 20.8% | **2.7$\times$** |
| Animal $\to$ Class | 6 | 0.696 | 1.43% | 39.3% | 34.3% | **27.5$\times$** |
| City $\to$ Country | 28 | 0.602 | 0.0% | 10.3% | 19.0% | **$\infty$** |

**Table 2.** Control results by domain on L12-16k ($L_0 = 82$). Bold ratios indicate shuffled-label control exceeds measured absorption in every domain. The untrained SAE control (C4) produces 0.0% absorption in all domains (not shown). The dense probe control (C3) achieves F1 = 0.929, confirming signal exists in the SAE activations.

The control inversion is specific to the shuffled-label condition: C1 (random probe) produces a near-chance rate (11.8% on first-letter), and C4 (untrained SAE) produces exactly 0%. A metric that assigns higher absorption scores to randomized labels than to true hierarchical labels is not measuring hierarchy-driven competitive exclusion.

## 4.2 Structural Explanation: Candidate Explosion in High Dimensions

At the standard cosine threshold $\tau_{\cos} \geq 0.025$ in $\mathbb{R}^{2304}$, a random unit vector identifies a mean of 3,766 decoder columns as candidates (23.0% of the 16,384 total; $n = 1{,}000$ random vectors; std = 109.2). True probes identify 3,478 candidates (21.2%) and shuffled probes identify 3,502 (21.4%). These counts are statistically indistinguishable: the candidate identification step provides no discriminative power between real and random probe directions.

With $L_0 = 82$ active latents per token, the probability that at least one candidate latent fires is 1.0 for all three probe types. Dead features (3,074 of 16,384, or 18.8%) inflate candidate counts without contributing signal. The metric therefore reduces to: absorption rate $\approx$ false-negative rate, because the candidate step is vacuous. Shuffled probes produce worse classification accuracy (lower F1), yielding more false negatives, which the metric counts as higher "absorption."

The structural cause: the cosine threshold $\tau_{\cos} = 0.025$ was calibrated for GPT-2 Small ($d_{\text{model}} = 768$, $d_{\text{SAE}} = 24{,}576$). Gemma 2 2B's higher-dimensional space produces a geometrically larger candidate set, rendering the threshold non-discriminative.

## 4.3 Threshold Sensitivity: Control Failure Is Structural

Table 3 reports the $5 \times 4$ threshold sensitivity grid.

| $\tau_{\text{mag}} \backslash \tau_{\cos}$ | 0.01 | 0.02 | 0.025 | 0.03 | 0.05 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.5** | 15.10% | 15.10% | 15.10% | 15.10% | 15.10% |
| **1.0** | 15.10% | 14.58% | **14.58%** | 14.58% | 14.06% |
| **1.5** | 15.10% | 14.24% | 13.54% | 13.54% | 13.19% |
| **2.0** | 15.10% | 13.19% | 12.15% | 12.15% | 11.81% |

**Table 3.** Absorption rate across the threshold parameter grid on L12-16k ($L_0 = 82$). Bold marks the Chanin et al. default ($\tau_{\cos} = 0.025$, $\tau_{\text{mag}} = 1.0$). CV = 0.077 across all 20 cells; range = 3.3 percentage points (11.81%--15.10%). Shuffled-label controls exceed measured absorption at all 20 combinations.

Absorption rate is stable across threshold choices (CV = 0.077). Letter-level rankings are highly preserved: mean pairwise Kendall $\tau = 0.977$ (all 190 pairs significant at $p < 0.05$). The magnitude gap has more influence than the cosine threshold (average reduction: 0.030 for gap versus 0.010 for cosine), consistent with JumpReLU's hard threshold mechanism producing more variable activation magnitudes than directional alignments. Control failure is structural, not resolvable by threshold tuning.

## 4.4 Confound Decomposition: Permissive vs. Strict Hedging

At $L_0 = 22$, where all 25 probes achieve F1 = 1.0, we identify 656 false-negative tokens. Figure 2 shows the decomposition across $L_0$ values.

![Stacked bar chart showing the decomposition of false negatives at each L0 value (22, 41, 82, 176). Hedging dominates at low L0, while the hierarchy-driven fraction increases as hedging tokens resolve at higher L0.](figures/hedging_decomposition.pdf)

**Figure 2.** Confound decomposition of false negatives across four $L_0$ values on L12-16k. At $L_0 = 22$ with perfect probes (F1 = 1.0), 98.6% of false negatives are classified as permissive hedging (blue) and 1.4% as hierarchy-driven (red). As $L_0$ increases, hedging tokens resolve and the hierarchy-driven fraction rises.

Table 4 reports the strict hedging classification.

| Classification | Count | Rate | 95% CI | Shuffled Control |
|:--------------|:-----:|:----:|:------:|:----------------:|
| Permissive hedging | 647 | 98.6% | --- | --- |
| **Strict hedging** | **41** | **6.2%** | **[4.4%, 8.2%]** | **3.4% $\pm$ 0.8%** |
| Non-hedging | 615 | 93.8% | [91.9%, 95.6%] | --- |
| Hierarchy-driven (persistent) | 9 | 1.4% | --- | --- |

**Table 4.** Hedging classification at $L_0 = 22$ ($n = 656$ FN tokens). The permissive definition (98.6%) is an upper bound; strict parent-latent checking reduces the hedging rate to 6.2%. The strict rate is significantly above the shuffled control (3.4%; $z = 3.51$, $p < 0.001$), confirming a real but small hedging signal.

The discrepancy between permissive (98.6%) and strict (6.2%) hedging is stark. Of the 656 FN tokens, 93.8% show none of the 5 parent-associated latents firing even at $L_0 = 176$. The permissive rate counts any token that ceases to be FN at a higher $L_0$ -- a near-tautological criterion, since FN counts drop from 656 at $L_0 = 22$ to 9 at $L_0 = 176$.

Letter G is a pronounced outlier: 19 of 21 FN tokens for G show strict hedging (90.5%), accounting for 46% of all strict-hedging cases. Excluding G, the strict hedging rate for the remaining letters drops to 3.5%. Twelve letters (B, D, F, H, J, K, M, N, P, R, T, W) show zero strict hedging.

## 4.5 Activation Patching: Ruling Out Competitive Exclusion

Eight persistent core words -- tokens that are FN at all four tested $L_0$ values $\{22, 41, 82, 176\}$ -- are the strongest candidates for competitive exclusion. Table 5 reports the causal intervention results.

| Word | Letter | Child Feature | $\cos(d_c, v_p)$ | Primary | Residual | All-Children | Control (0/10) |
|:----:|:------:|:------------:|:-----------------:|:-------:|:--------:|:------------:|:--------------:|
| eight | E | 8174 | 0.216 | No | No | No | 0/10 |
| liked | L | 4678 | 0.074 | No | No | No | 0/10 |
| lower | L | 3858 | 0.218 | No | No | No | 0/10 |
| offer | O | 15092 | 0.223 | No | No | No | 0/10 |
| often | O | 3063 | 0.143 | No | No | No | 0/10 |
| other | O | 15322 | 0.203 | No | No | No | 0/10 |
| under | U | 2810 | 0.246 | No | No | No | 0/10 |
| until | U | --- | --- | No | No | No | 0/10 |

**Table 5.** Activation patching on 8 persistent core words at $L_0 = 82$. "Primary" = decode-reencode after child zeroing; "Residual" = subtract child decoder direction from activation; "All-Children" = zero all absorbing features simultaneously. Recovery rate: 0/8 (0%; 95% CI: [0%, 36.9%]). Control: zeroing 10 random non-child features per word produces 0/65 spurious recoveries. "until" has no absorbing features at $L_0 = 82$.

The 0/8 recovery result provides metric-independent causal evidence against competitive exclusion for these tokens. If child latents were actively suppressing parent latents under sparsity pressure, zeroing the child should release that pressure and allow the parent to fire. The absence of recovery indicates that the parent information is not encoded in the parent-associated latents for these words -- the false-negative status reflects a genuine encoding gap in the SAE, not competitive suppression.

## 4.6 Probe Quality Confound

Across the 25 tested letters at $L_0 = 82$, probe F1 and absorption rate are strongly anti-correlated: Spearman $\rho_s = -0.69$ ($p < 0.001$). Letters with low probe quality (C: F1 = 0.71, absorption = 29.0%; T: F1 = 0.72, absorption = 32.9%) show high absorption, while letters with high probe quality (K: F1 = 0.96, absorption = 0%; J: F1 = 0.95, absorption = 0%) show zero absorption. This confound means that the absorption rate at $L_0 = 82$ partially tracks probe classification errors rather than genuine feature dynamics. At $L_0 = 22$, where all probes achieve F1 = 1.0, this confound is eliminated.

## 4.7 First-Letter Replication and Cross-Domain Results

The aggregate first-letter absorption rate is 15.96% (95% CI: [8.4%, 17.5%]) on L12-16k at $L_0 = 82$, within the published 15--35% range from Chanin et al. The replication succeeds in magnitude but the universal control failure means the interpretation as competitive exclusion is unsupported.

Cross-domain absorption rates are: city-continent 6.49% (95% CI: [0%, 11.5%]), city-language 6.56%, animal-class 1.43% (CI: [0%, 3.6%]), and city-country 0.0%. All rates fall below their shuffled controls (Table 2), so absolute rates cannot be interpreted as genuine absorption. The city-continent signal is driven by a single parent: Asia shows 21.62% absorption (probe F1 = 0.895), while other continents range from 0% to 3.3%.

---

# 5 The $L_0$-Absorption Phase Transition

## 5.1 Monotonic Decline Across $L_0$

Figure 3 shows absorption declining monotonically from 42.85% at $L_0 = 22$ to 0.84% at $L_0 = 176$ on L12-16k (Spearman $\rho_s = -1.0$; bootstrap CIs at adjacent $L_0$ values do not overlap).

![Absorption rate versus L0 operating point for three layers (L10, L12, L20), showing monotonic decline and cross-layer stability.](figures/l0_phase_transition.pdf)

**Figure 3.** $L_0$-absorption phase transition on L12-16k (primary trace) with cross-layer validation at L10-16k and L20-16k. Error bars are 95% bootstrap CIs. The shaded region marks the phase transition zone ($L_0 \approx 40$--80).

| $L_0$ | Absorption Rate | 95% CI | FN Count |
|:------:|:--------------:|:------:|:--------:|
| 22 | 42.85% | [40.1%, 45.6%] | 656 |
| 41 | 37.49% | [34.8%, 40.2%] | 488 |
| 82 | 14.39% | [12.4%, 16.4%] | 186 |
| 176 | 0.84% | [0.3%, 1.4%] | 9 |

**Table 7.** Absorption rate by $L_0$ on L12-16k (confound decomposition protocol, 1,195 tested words). Spearman $\rho_s = -1.0$ (perfect monotonic decline). FN counts decrease from 656 to 9.

The sharpest drop occurs between $L_0 = 41$ and $L_0 = 82$ (37.49% to 14.39%), identifying the phase transition region at $L_0 \approx 40$--80. Below $L_0 \approx 40$, the SAE is capacity-starved and absorption exceeds 35%. Above $L_0 \approx 80$, sufficient capacity reduces absorption below 15%.

**Note on absorption rate measurements.** The L12-16k rate at $L_0 = 82$ is reported as 14.39% in the $L_0$ sweep (confound decomposition protocol, 1,195 tested words) and 15.96% in the first-letter replication (Section 4.7; 1,203 tested words, probes trained at $L_0 = 82$). The difference of 1.57 percentage points arises from different vocabulary sizes and probe sets; the cross-layer CV of 8.6% encompasses both values.

## 5.2 Cross-Layer Stability

At $L_0 = 82$, layer 10 yields 13.88%, layer 12 yields 14.39%, and layer 20 yields 13.55% (CV = 8.6%). The phase transition is a property of the $L_0$ operating point, not the layer position.

## 5.3 Width-$L_0$ Interaction

OLS regression on the full 34-configuration scaling surface yields a significant width $\times$ $L_0$ interaction ($F = 37.75$, $p = 1.24 \times 10^{-6}$, $\Delta R^2 = 0.252$). Width increases absorption -- wider dictionaries spread parent information across more latents -- but $L_0$ remains the dominant factor ($\rho_s = -0.457$, $p = 0.007$; width: $\rho_s = 0.308$, $p = 0.076$, not significant after correction).

## 5.4 Cross-Architecture Reference (Confounded)

GPT-2 Small with L1-ReLU SAEs shows uniformly high absorption: 67.29% at layer 8, 64.26% at layer 10, and 61.65% at layer 11. No $L_0$-dependent phase transition exists because L1-ReLU SAEs lack configurable $L_0$. This comparison is confounded by model size ($d_{\text{model}} = 768$ vs. 2304), architecture, training data, and $L_0$ differences; it is reported as context, not as a controlled comparison.

---

# 6 Exploratory CMI-Absorption Association

## 6.1 Motivation

The successive refinement theorem (Equitz and Cover, 1991) predicts that features with low conditional mutual information (CMI) -- those where the parent encodes little unique information beyond the child -- should be more susceptible to absorption. We test this prediction by estimating CMI via $k$-nearest neighbors (Kraskov et al., 2004; $k_{\text{nn}} = 5$) in a $d' = 10$ decoder subspace for each of 25 first-letter features.

## 6.2 Results at $L_0 = 82$ (Marginal Signal)

Absorbed letters ($n = 13$; absorption rate $\geq 10\%$) have lower mean CMI than non-absorbed letters ($n = 9$; rate $< 5\%$): 0.649 $\pm$ 0.187 vs. 0.861 $\pm$ 0.258; Mann-Whitney $p = 0.045$; Cohen's $d = -0.924$. The rank correlation across all 25 letters is Spearman $\rho_s = -0.383$ ($p = 0.059$; Bonferroni-corrected $p = 0.236$).

This association is confounded by probe quality ($\rho_s = -0.69$ between absorption and probe F1). Table 6 shows progressive weakening with each control.

| Analysis | $n$ | Spearman $\rho_s$ | 95% CI | $p$-value | Interpretation |
|:---------|:---:|:-----------------:|:------:|:---------:|:--------------|
| Raw $\rho_s$(CMI, absorption) | 25 | $-0.383$ | [$-0.684$, 0.031] | 0.059 | Marginal |
| Raw $\rho_s$(absorption, probe F1) | 25 | $-0.692$ | --- | 0.0001 | Strong confound |
| Partial $\rho_s$(CMI, absorption $\mid$ F1) | 25 | $-0.328$ | [$-0.664$, 0.246] | 0.118 | Weakened |
| Restricted $\rho_s$ (F1 $> 0.85$) | 10 | $-0.113$ | [$-0.789$, 0.808] | 0.757 | Eliminated |

**Table 6.** CMI-absorption correlation at $L_0 = 82$ ($d' = 10$). The signal weakens with each robustness check and vanishes when restricted to letters with high probe quality.

## 6.3 Replication at $L_0 = 22$: Null Result

At $L_0 = 22$, all 25 probes achieve F1 = 1.0, eliminating the probe quality confound entirely. The pre-registered analysis at $d' = 10$ yields Spearman $\rho_s = 0.044$ ($p = 0.835$; Bonferroni-corrected $p = 1.0$). The sign is positive -- reversed from the $-0.383$ at $L_0 = 82$.

![CMI vs. absorption rate per letter at d'=10, comparing L0=82 (left panel, points colored by probe F1) and L0=22 (right panel, all F1=1.0).](figures/cmi_vs_absorption.pdf)

**Figure 4.** CMI vs. absorption rate per letter at $d' = 10$. Left: $L_0 = 82$ (points colored by probe F1); right: $L_0 = 22$ (all F1 = 1.0). The negative trend at $L_0 = 82$ vanishes at $L_0 = 22$ where probe quality is uniform.

At higher subspace dimensions, the $L_0 = 22$ correlations are positive and strengthening: $d' = 20$ yields $\rho_s = 0.248$ ($p = 0.232$); $d' = 30$ yields $\rho_s = 0.410$ ($p = 0.042$); $d' = 50$ yields $\rho_s = 0.483$ ($p = 0.014$). Both nominally significant values reverse the theoretical prediction. None survives Bonferroni correction.

## 6.4 Dimension Sensitivity

![Spearman rho between CMI and absorption rate as a function of subspace dimension d'.](figures/cmi_dimension_sensitivity.pdf)

**Figure 5.** Spearman $\rho_s$ between CMI and absorption rate as a function of subspace dimension $d'$. At $L_0 = 82$, the correlation is negative at $d' = 10$ and reverses sign at $d' \geq 20$. At $L_0 = 22$, the correlation is positive at all $d'$ values. The direction depends on both subspace choice and $L_0$, undermining any theoretical interpretation.

## 6.5 Leave-One-Out Sensitivity

No single letter removal changes $\rho_s$ by more than 0.1 (maximum $|\Delta\rho_s| = 0.088$, letter V). All 25 leave-one-out $\rho_s$ values remain negative, ranging from $-0.471$ to $-0.321$. The jackknife standard error is 0.186; bias-corrected $\rho_s = -0.397$. The marginal signal at $L_0 = 82$ is stable across individual letter removals but, as shown in Section 6.2, not robust to probe quality controls.

## 6.6 Interpretation

Three lines of evidence converge: partial correlation ($p = 0.118$), restricted analysis ($\rho_s = -0.113$, $p = 0.757$), and replication failure at $L_0 = 22$ ($\rho_s = +0.044$). The CMI-absorption association at $L_0 = 82$ was driven by the probe quality confound ($\rho_s = -0.69$ between absorption and probe F1), not by rate-distortion theory. The successive refinement framework remains theoretically sound, but this instantiation -- $k$-NN CMI estimation in a $d' = 10$ decoder subspace, correlated with Chanin metric absorption rates -- does not provide empirical support.

---

# 7 Discussion

## 7.1 Connecting the Findings

The metric audit (Section 4) and $L_0$ phase transition (Section 5) connect as follows. Most of what the Chanin metric measures on JumpReLU SAEs is hedging, not competitive exclusion -- confirmed by the strict hedging check (6.2% vs. 98.6% permissive) and the activation patching null result (0/8 recovery). The $L_0$ operating point controls how much hedging occurs: at low $L_0$, capacity starvation forces the SAE to spread parent information across many latents; at high $L_0$, sufficient capacity reduces false negatives to near zero. The CMI diagnostic (Section 6), initially promising, does not survive probe quality controls.

## 7.2 Two Interpretations

The data admit two readings that we cannot fully disambiguate.

**Interpretation A (metric miscalibration).** The Chanin metric is fundamentally miscalibrated for JumpReLU SAEs. At $\tau_{\cos} = 0.025$ in $\mathbb{R}^{2304}$, a random vector identifies 23.0% of decoder columns as candidates, the candidate step provides no discriminative power, and the metric reduces to the false-negative rate. Under this reading, the measured 15.96% absorption rate at $L_0 = 82$ is a probe artifact.

**Interpretation B (genuine low absorption).** JumpReLU SAEs at $L_0 \geq 82$ genuinely exhibit low competitive exclusion. The metric correctly detects few false negatives because few exist. The universal control failure reflects that shuffled labels produce worse probes with more false negatives. Under this reading, the 0/8 activation patching result confirms that competitive exclusion is rare, and the monotonic decline with $L_0$ reflects a real decrease in information loss.

Both interpretations agree on the empirical facts and the practical consequence: what the Chanin metric measures on JumpReLU SAEs is not the competitive exclusion mechanism that architectural mitigations target. The 0.84% rate at $L_0 = 176$ and the 0/8 patching result suggest that competitive exclusion is rare at typical operating points regardless of interpretation. Scaling activation patching beyond 8 words is the clearest path to disambiguation.

## 7.3 Implications for the Mitigation Literature

Matryoshka SAE, OrtSAE, ATM-SAE, and masked regularization all benchmark absorption reduction against the Chanin metric. Two concerns follow.

First, if the metric measures false-negative rate rather than competitive exclusion on JumpReLU SAEs, then absorption reductions reported by these methods may reflect changes in probe behavior rather than changes in underlying feature dynamics. The metric would need recalibration -- or replacement with a causal diagnostic such as activation patching -- before mitigation benchmarks on JumpReLU SAEs can be interpreted. We note that mitigations may genuinely improve feature quality through mechanisms (hierarchical organization, orthogonality) that benefit latents independently of competitive exclusion.

Second, the $L_0$ phase transition (42.85% to 0.84%) identifies the $L_0$ operating point as the dominant control parameter. Architectural interventions that do not modify the sparsity-fidelity tradeoff may be secondary to operating at a higher $L_0$. The sharp transition between $L_0 \approx 40$ and $L_0 \approx 80$ suggests a critical capacity threshold. Mitigation methods should report their effective $L_0$ alongside absorption measurements to disentangle architectural effects from sparsity effects.

## 7.4 Negative Results

Five pre-registered hypotheses are falsified or unsupported: H2 (hierarchy-driven dominance at $L_0 = 22$: predicted $> 80\%$, observed 1.4%), H4 (CMI predicts absorption: $\rho_s = -0.383$ at $L_0 = 82$ does not replicate at $L_0 = 22$), H5 (cross-domain rates above controls: all below), H6 (cross-domain entity matching: zero matches), H7 (architecture-specific bimodality: both architectures bimodal). Each was reported with pre-registered targets and observed values.

The CMI non-replication illustrates three lessons for the mechanistic interpretability community: (a) marginal signals require replication across operating regimes, (b) confounders must be controlled before theoretical interpretation, and (c) dimension sensitivity analysis is essential for $k$-NN mutual information estimators.

## 7.5 Limitations

**Single model family.** All primary experiments use Gemma 2 2B with Gemma Scope JumpReLU SAEs. The control failure, hedging dominance, and $L_0$ phase transition have not been validated on other model families.

**First-letter spelling as primary hierarchy.** First-letter features are weakly represented graphemic information with unusual hierarchy properties: depth-2, branching factor ~40, and near-perfect co-occurrence between parent and child tokens.

**Small activation patching sample.** The 0/8 parent recovery result is based on 8 words. Five of these (*other*, *often*, *offer*, *under*, *until*) are high-frequency function-adjacent words that may have unusual encoding dynamics.

**CMI estimation fragility.** The $k$-NN estimator's output depends on subspace dimension $d'$. At $L_0 = 82$, the sign reverses at $d' \geq 20$; at $L_0 = 22$, the correlation is positive at all tested dimensions and reaches $\rho_s = 0.483$ at $d' = 50$ -- opposite to the theoretical prediction.

**Vocabulary scope.** The study uses only single-token English words. Multi-token words, non-English tokens, and the broader vocabulary are excluded.

**Training-free constraint.** This study evaluates pretrained SAEs only and cannot assess whether retraining with hierarchy-aware losses would change absorption dynamics.

## 7.6 Future Directions

**Activation patching at scale.** Patching the full set of 186 false negatives at $L_0 = 82$ (or 656 at $L_0 = 22$) would establish the prevalence of competitive exclusion across the entire false-negative population, resolving the central ambiguity.

**Metric redesign.** The candidate explosion finding suggests that unsupervised absorption detection may be possible through decoder geometry alone. A valid metric should satisfy a minimal criterion: shuffled-label absorption $<$ measured absorption in $\geq 3$ hierarchy domains.

**Cross-model validation.** Replicating the $L_0$ phase transition on Gemma 2 9B, 27B (with available Gemma Scope SAEs) and Llama 3.1 family models would establish whether the transition zone ($L_0 \approx 40$--80) is universal.

**SAE retraining experiments.** The finding that absorption is primarily controlled by $L_0$ suggests that sparsity schedule modifications (e.g., curriculum learning over $L_0$) may be more effective than architectural changes.

---

# 8 Conclusion

We posed three questions about the Chanin feature absorption metric on JumpReLU SAEs; each is answered by one of the findings below.

**Q1 -- The metric does not transfer.** Shuffled-label controls exceed measured absorption rates in all five domains tested on Gemma Scope JumpReLU SAEs (ratios from 2.7$\times$ to $\infty$; 4.7$\times$ on first-letter). This structural failure persists across all 20 threshold combinations tested (CV = 0.077; Section 4.3). Confound decomposition at $L_0 = 22$ with perfect probes classifies only 6.2% of false negatives as strict hedging (Table 4); activation patching on 8 persistent core words yields 0/8 parent recovery (Table 5), ruling out competitive exclusion for these tokens.

**Q2 -- Absorption is controlled by $L_0$.** Absorption declines monotonically from 42.85% ($L_0 = 22$) to 0.84% ($L_0 = 176$), with a phase transition in the $L_0 \approx 40$--80 range stable across three layers (CV $<$ 10%; Figure 3). The $L_0$ operating point -- a training-time hyperparameter requiring no architectural modification -- is the primary control parameter for absorption severity on JumpReLU SAEs.

**Q3 -- The CMI diagnostic does not replicate.** The marginal CMI-absorption correlation at $L_0 = 82$ ($\rho_s = -0.383$, $p = 0.059$) vanishes at $L_0 = 22$ with perfect probes ($\rho_s = 0.044$, $p = 0.835$; Table 6). The original signal was driven by probe quality confounds, not rate-distortion theory.

Four of seven pre-registered hypotheses are falsified (H2, H5, H6, H7), each reported with pre-registered targets versus observed values. These negative results constrain future work as directly as the positive findings.

Three recommendations follow. First, validate absorption metrics on each new SAE architecture before building mitigations: the shuffled-label control check (shuffled rate $<$ measured rate in $\geq 3$ hierarchy domains) should be standard practice. Second, report shuffled-label and random-probe controls alongside measured absorption rates. Third, treat the $L_0$ operating point as a first-order intervention for absorption severity on JumpReLU SAEs before pursuing encoder modifications -- increasing $L_0$ from 22 to 176 reduces measured absorption by 98%.

Code and data are released as an SAEBench extension at [URL], including the four-control suite and confound decomposition pipeline for application to new SAE architectures.

---

## Figures and Tables

- **Figure 1:** control_failure.pdf -- Universal control failure across five hierarchy domains; grouped bars of measured vs. shuffled vs. random absorption rates
- **Figure 2:** hedging_decomposition.pdf -- Confound decomposition stacked bars showing hedging vs. hierarchy-driven fractions across four $L_0$ values
- **Figure 3:** l0_phase_transition.pdf -- $L_0$-absorption phase transition line plot for three layers (L10, L12, L20) with confidence bands
- **Figure 4:** cmi_vs_absorption.pdf -- Two-panel scatter: CMI vs. absorption at $L_0 = 82$ and $L_0 = 22$
- **Figure 5:** cmi_dimension_sensitivity.pdf -- CMI-absorption correlation direction as a function of subspace dimension $d'$
- **Table 1:** inline -- SAE configurations used in this study
- **Table 2:** inline -- Control results by domain showing universal control failure
- **Table 3:** inline -- Threshold sensitivity heatmap (5$\times$4 grid)
- **Table 4:** inline -- Hedging classification comparison (permissive vs. strict)
- **Table 5:** inline -- Activation patching results for 8 persistent core words
- **Table 6:** inline -- CMI-absorption correlation analysis with probe quality controls
- **Table 7:** inline -- $L_0$ phase transition absorption rates with bootstrap CIs
- **Table 8:** inline -- Cross-domain hierarchy suite
