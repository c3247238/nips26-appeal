# The Absorption Tax: Layer-Dependent and Hierarchy-Dependent Feature Absorption Across Semantic Domains in Sparse Autoencoders

## Abstract

Sparse autoencoder (SAE) feature absorption -- where general features fail to fire because specific co-occurring features encode overlapping information -- threatens the reliability of SAE-based mechanistic interpretability. All published absorption measurements use a single proxy task (first-letter spelling) at a single model layer. We present the first cross-domain and cross-layer absorption characterization, extending measurement from first-letter spelling to entity-attribute knowledge hierarchies (city-country, city-continent, city-language) from the RAVEL dataset on Gemma 2 2B with Gemma Scope JumpReLU SAEs.

Three results emerge. First, first-letter absorption varies 15-fold across model layers -- from 2.2% at layer 18 to 34.5% at layer 24 (probe F1 = 0.97 at all layers) -- demonstrating that single-layer benchmarks are unrepresentative. Second, measured absorption rates differ significantly across hierarchy types (Kruskal-Wallis $p = 0.005$, 4 of 6 pairwise comparisons significant at $p < 0.05$), though this comparison is confounded by differential probe quality ($\rho = -0.756$ between probe F1 and false negative rate). Third, activation patching provides the first interventional causal evidence for feature suppression in SAEs: zeroing child features recovers parent probe predictions at 32.5% versus 1.5% for magnitude-matched controls (Wilcoxon $p = 0.0002$, Cohen's $d = 1.33$, 95% CI on difference: [0.21, 0.42]).

We additionally show that the widely cited ${\sim}$98% hedging rate is near-tautological: a three-way decomposition reveals that only 7.9% of false negatives exhibit strict hedging (parent feature recovery), while 86.2% resolve through compensatory activation of unrelated features at higher $L_0$. Three proposed unsupervised absorption detectors -- Geometric Absorption Score (GAS, $\rho = 0.116$), Conditional Mutual Information (CMI, $\rho = 0.044$), and the Absorption Tax ranking ($\rho = -0.20$) -- all fail. These results jointly invalidate single-task, single-layer absorption benchmarks and establish that absorption evaluation must report rates across multiple hierarchy types, model layers, and probe quality levels.

---

# 1 Introduction

Absorption rates on the same sparse autoencoder (SAE) vary 15-fold depending on which model layer is measured. On Gemma 2 2B with Gemma Scope JumpReLU SAEs, first-letter absorption ranges from 2.2% at layer 18 to 34.5% at layer 24 (Figure 1). When measurement extends from the canonical first-letter spelling task to entity-attribute knowledge hierarchies -- city-country, city-continent, city-language -- rates differ significantly across hierarchy types (Kruskal-Wallis $p = 0.005$, 4 of 6 pairwise comparisons significant at $p < 0.05$). No single absorption rate characterizes an SAE: both layer position and hierarchy structure govern the severity of feature absorption.

Feature absorption is a failure mode where a general (parent) SAE latent fails to fire when a specific (child) latent co-occurs, because the SAE achieves lower $L_0$ by encoding the parent's information into the child's decoder direction (Chanin et al., 2024). A concrete example: a latent labeled "starts with s" may correctly activate for inputs like "sun" and "stone" but fail to fire for "snake," because a dedicated "snake" latent absorbs the letter-level information into its own decoder direction. The parent latent appears monosemantic on casual inspection -- it responds cleanly to s-initial tokens -- yet it silently misses a systematic subset of its semantic domain. This false sense of monosemanticity threatens any downstream application that assumes SAE latents provide complete feature coverage.

The practical stakes are substantial. Google DeepMind reported that feature absorption degraded safety-relevant feature detection by 10--40%, contributing to their decision to deprioritize SAE-based interpretability research (Lieberum et al., 2024). Conversely, Anthropic's circuit-tracing work on Claude 3.5 Haiku demonstrates that when features are reliable, they enable powerful mechanistic understanding of frontier model behavior (Lindsey et al., 2025). Absorption sits at the boundary between these outcomes: if SAE features cannot be trusted to fire reliably on their full semantic domain, circuit analyses built on those features inherit systematic blind spots.

Every published absorption measurement rests on a single proxy task: first-letter spelling (Chanin et al., 2024; Karvonen et al., 2025). This task maps each token to its initial letter, producing 26 classes with near-uniform distribution and 100% parent-child co-occurrence by construction. The sae-spelling benchmark and the absorption component of SAEBench both use this task exclusively. Architectural mitigations -- Matryoshka SAE (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), ATM (Li et al., 2025), masked regularization (Narayanaswamy et al., 2026) -- report absorption improvements on first-letter alone. Real knowledge hierarchies differ from first-letter spelling in three structural ways: class distributions are heavily imbalanced (France has more cities than Liechtenstein), hierarchies have multiple levels of depth (city $\to$ country $\to$ continent), and parent-child co-occurrence depends on context rather than being guaranteed by construction. Whether absorption rates measured on first-letter spelling generalize to these richer hierarchies has not been tested.

This paper presents four contributions:

- **Cross-layer absorption characterization.** First-letter absorption on Gemma 2 2B spans 2.2% to 34.5% across four model layers (6, 12, 18, 24) with eight SAE configurations. This measurement is unconfounded: first-letter probes achieve F1 $\geq$ 0.97 at all layers. Layer 24 rates (25--35%) align with the 15--35% range reported by Chanin et al. (2024), suggesting prior work likely measured at later layers; layers 6 and 18 show absorption below 5%.

- **Cross-domain absorption measurement.** Extending measurement to three entity-attribute hierarchies from the RAVEL dataset (Huang et al., 2024), we find significant variation in measured absorption rates (Kruskal-Wallis $p = 0.005$). At layer 24, city-continent absorption (35.8%, 16k) is comparable to first-letter (34.5%), while city-country (18.5%) and city-language (13.6%) are significantly lower ($p = 0.004$ and $p = 0.0001$, respectively). This comparison is confounded by differential probe quality (first-letter F1 = 0.97 vs. RAVEL F1 = 0.79--0.84), which we report transparently as a limitation.

- **Interventional causal evidence for feature suppression.** Activation patching -- zeroing the highest-attribution child latent in SAE output -- recovers correct parent-class probe predictions at 32.5% (mean across 19 words with detected absorption), compared to 1.5% for magnitude-matched control features (Wilcoxon $p = 0.0002$, Cohen's $d = 1.33$, 95% CI on difference: [0.21, 0.42]). This provides the first interventional (not merely correlational) evidence that child latents causally suppress parent-class information in SAE encodings.

- **Tightened hedging decomposition and honest negative results.** Decomposing 304 first-letter false negatives into three categories reveals that only 7.9% exhibit strict hedging (the parent latent itself recovers at higher $L_0$), while 86.2% show compensatory resolution (unrelated latents add sufficient information) and 5.9% persist. The widely cited ${\sim}$98% hedging rate (Chanin et al., 2025) thus reflects a near-tautological upper bound. We also report definitive negative results for three proposed unsupervised absorption detectors: the Geometric Absorption Score (GAS, $\rho = 0.116$), Conditional Mutual Information (CMI, $\rho = 0.044$), and the Absorption Tax ranking ($\rho = -0.20$).

Figure 1 presents the central finding visually: a heatmap of absorption rates across layers and hierarchy types, showing both the 15-fold layer variation and the significant cross-domain differences.

![Feature absorption varies by layer and hierarchy type. First-letter absorption ranges from 2.2% (L18, 16k) to 34.5% (L24, 16k). Cross-domain rates at L24 differ significantly across hierarchy types (Kruskal-Wallis p=0.005). RAVEL hierarchies measured at L24 only (best probe layer). Probe F1 annotated per hierarchy.](figures/fig1_heatmap.pdf)

Before presenting our measurements, we formalize the absorption phenomenon and review existing evaluation methodology.

---

# 2 Background and Related Work

Sparse autoencoders decompose polysemantic neural activations into an overcomplete basis of approximately monosemantic features by optimizing a sparsity-inducing objective. The foundational motivation comes from the linear representation hypothesis: neural networks encode more features than they have neurons by representing features as overlapping linear directions -- a phenomenon called superposition (Elhage et al., 2022). Early demonstrations that SAEs recover interpretable monosemantic features in small models (Cunningham et al., 2024; Bricken et al., 2023) led to rapid scaling to frontier systems: Anthropic applied SAEs to Claude 3 Sonnet and identified safety-relevant features such as deception and sycophancy (Templeton et al., 2024), OpenAI trained a 16M-latent SAE on GPT-4 with clean scaling laws (Gao et al., 2024), and Google DeepMind released Gemma Scope -- 400+ open JumpReLU SAEs across all layers of Gemma 2 models at widths from 1k to 1M (Lieberum et al., 2024). These successes established SAEs as the dominant tool for mechanistic interpretability, but also exposed systematic failure modes that challenge whether SAEs reliably recover the features they promise.

**Feature absorption.** Chanin et al. (2024) identify and formalize absorption: a general (parent) feature fails to activate when a specific (child) feature co-occurs, because the SAE achieves better sparsity by encoding the parent's information into the child's decoder direction $\mathbf{d}_c$. The canonical example is first-letter spelling: the token "snake" fires only the snake-specific latent, not the "starts with s" latent, because the snake latent's decoder direction already encodes the first-letter information via high cosine similarity $\cos(\mathbf{d}_{\text{snake}}, \mathbf{w}_s)$ with the letter probe direction. Chanin et al. prove in a toy model that hierarchical feature structure is sufficient to produce absorption and report absorption rates of 15--35% across hundreds of Gemma Scope, Llama 3.2, and Qwen2 SAEs. Their measurement pipeline -- identify false negatives $\text{FN}$ via linear probes, run integrated-gradients attribution, detect absorption via cosine threshold $\tau_{\cos} > 0.025$ and magnitude gap $\tau_{\text{gap}} \geq 1.0$ -- has become the standard metric adopted by SAEBench (Karvonen et al., 2025). Absorption is present in every SAE architecture tested, including TopK, JumpReLU, and BatchTopK, establishing it as a universal failure mode rather than an architecture-specific bug.

**The absorption-reconstruction tradeoff.** Absorption is not a training error but a rational optimization outcome. For each parent-child feature pair $(p, c)$, an SAE with absorption represents both features at a cost of $+1$ to $L_0$, while an absorption-free SAE requires $+2$ to $L_0$ -- one latent for the parent, one for the child. Any solution that fully eliminates absorption will therefore have a worse variance-explained vs. $L_0$ frontier (Chanin et al., 2024). This fundamental tension means that sparsity objectives and interpretability are inherently at odds under feature hierarchy. Theoretical analysis confirms the depth of this problem: Cui et al. (2025) show SAEs generally fail to recover ground-truth monosemantic features unless features are extremely sparse, and a unified theory of sparse dictionary learning casts absorption as a natural consequence of the piecewise biconvex optimization landscape that all SAE variants share (arXiv:2512.05534).

**Adjacent failure modes.** Absorption is one of several documented SAE failure modes, each with distinct causes. *Feature hedging* (Chanin et al., 2025) occurs when an SAE has insufficient dictionary width: correlated features merge into a single latent, producing systematic false negatives for the individual features. Chanin & Garriga-Alonso (2025) further show that most open-source SAEs operate at incorrect $L_0$ (typically too low), which triggers hedging and feature mixing. Critically, hedging and absorption both manifest as features failing to fire where they should, but their causes differ -- width/sparsity limitations vs. hierarchical feature structure -- and Matryoshka SAEs trade one for the other (Chanin et al., 2025). *Feature inconsistency* (Song et al., 2025) describes the phenomenon where independent training runs converge to different feature sets, with TopK SAEs achieving only 0.80 consistency. *Dark matter* (Engels et al., 2024) refers to unexplained SAE reconstruction error, of which approximately 50% is linearly predictable from the input. *Feature non-canonicality* (Leask et al., 2025) challenges the assumption that SAE latents are atomic units: meta-SAEs decompose latents into sub-features, and larger dictionaries discover qualitatively new latents missed by smaller ones. Tian et al. (2025) frame absorption as a special case of poor *feature sensitivity* -- features that appear monosemantic on their activation examples may nonetheless fail to activate on semantically similar inputs.

**Architectural mitigations.** Several architectures reduce absorption, though none eliminate it. Matryoshka SAEs (Bussmann et al., 2025) train nested dictionaries of increasing size simultaneously, creating a natural feature hierarchy where smaller dictionaries learn general concepts; they achieve absorption rate $\sim$0.03 on SAEBench vs. BatchTopK's $\sim$0.29. OrtSAE (Korznikov et al., 2025) enforces orthogonality via pairwise cosine similarity penalties on decoder directions, reducing absorption by $\sim$70% vs. BatchTopK and lowering MeanCosSim by 2.7$\times$. Adaptive Temporal Masking (Li et al., 2025) dynamically adjusts feature selection via per-latent importance scoring and reports the lowest absorption scores to date (0.0068 vs. TopK 0.1402 and JumpReLU 0.0114). KronSAE (2025) exploits Kronecker product factorization to reduce absorption fraction at lower parameter count. Masked regularization (Narayanaswamy et al., 2026) disrupts co-occurrence patterns during training via token masking, reducing absorption across SAE architectures. Each method was evaluated under different conditions (models, layers, metrics), and no head-to-head comparison exists. All architecture evaluations use the first-letter spelling task as their absorption benchmark.

**The evaluation gap.** SAEBench (Karvonen et al., 2025) standardized SAE evaluation with 8 metrics across 200+ SAEs and 7 architectures, revealing that proxy metrics (cross-entropy loss, sparsity) do not reliably predict practical performance. Its absorption metric builds on Chanin et al.'s procedure and covers all layers. SynthSAEBench (2026) provides a controlled synthetic environment with known ground-truth feature hierarchies, demonstrating that logistic probes achieve F1 = 0.974 while SAEs substantially underperform. CE-Bench (Gulko et al., 2025) offers a lightweight contrastive benchmark with $>$70% Spearman correlation with SAEBench. Yet across all these benchmarks, absorption is measured exclusively on the first-letter spelling task -- a controlled proxy with 26 classes, near-uniform distribution, and 100% parent-child co-occurrence by construction. No cross-domain characterization exists: absorption rates on entity-attribute hierarchies, knowledge taxonomies, or safety-relevant features are unknown. No cross-layer analysis exists: all published rates aggregate over a single layer or report layer-averaged scores without examining layer-wise variation. Google DeepMind's decision to deprioritize SAE research was partly driven by 10--40% performance degradation on safety-relevant downstream tasks (Smith et al., 2025), but the specific contribution of absorption to this gap has not been quantified. This paper addresses these gaps directly.

We now describe our experimental setup for systematically characterizing absorption across layers, hierarchies, and architectures.

---

# 3 Method

We measure feature absorption across four feature hierarchies, four model layers, and three SAE architectures on Gemma 2 2B, using an adapted version of the Chanin et al. (2024) absorption measurement pipeline. Figure 2 illustrates the full pipeline from residual stream extraction through absorption detection and causal validation via activation patching.

![Experimental pipeline from residual stream extraction through probe classification, false negative detection, integrated-gradients attribution, and activation patching.](figures/fig2_pipeline_desc.md)

## 3.1 Model and SAEs

All experiments use Gemma 2 2B (`unsloth/gemma-2-2b`; $d = 2304$, 26 layers) at inference time. No SAE training is performed; we evaluate pre-trained SAEs only.

**Gemma Scope SAEs.** We use eight JumpReLU SAEs from Google DeepMind's Gemma Scope collection (Lieberum et al., 2024) spanning four layers ($L \in \{6, 12, 18, 24\}$) at two dictionary sizes ($M \in \{16384, 65536\}$). These SAEs operate at $L_0 \approx 75$--$87$ depending on the layer.

**SAEBench SAEs.** At layer 12, we additionally evaluate two SAEs from SAEBench (Karvonen et al., 2025): a BatchTopK SAE with $M = 16384$ ($L_0 = 20$) and a Matryoshka SAE with $M = 32768$. These enable cross-architecture comparison at a single layer.

The 10 SAE configurations are summarized in Table 1.

**Table 1: SAE configurations evaluated.** Gemma Scope SAEs span four layers at two dictionary widths; SAEBench SAEs provide cross-architecture comparison at layer 12.

| SAE Family | Architecture | Layer(s) | $M$ | $L_0$ |
|---|---|---|---|---|
| Gemma Scope | JumpReLU | 6, 12, 18, 24 | 16k | 75--87 |
| Gemma Scope | JumpReLU | 6, 12, 18, 24 | 65k | 75--87 |
| SAEBench | BatchTopK | 12 | 16k | 20 |
| SAEBench | Matryoshka | 12 | 32k | -- |

## 3.2 Feature Hierarchies

We define four feature hierarchies $G = (V, E)$, each mapping tokens to a categorical attribute through parent-child relationships where the child feature logically implies the parent.

**First-letter (syntactic).** Each token maps to its initial letter: $K = 26$ classes with near-uniform distribution. We use the `sae_spelling` pipeline (Chanin et al., 2024) with in-context learning prompts to extract first-letter representations. This hierarchy serves as the baseline and replication target.

**City-continent (factual, coarse).** Each city entity maps to its continent: $K = 6$ classes (Africa, Asia, Europe, North America, Oceania, South America). Data comes from the RAVEL dataset ($n = 2039$ cities; Huang et al., 2024). We use in-context learning prompts with last-token ($\text{position} = -1$) extraction.

**City-country (factual, fine-grained).** Each city maps to its country: $K = 80$ classes with highly imbalanced distribution (the United States, Russia, and India dominate). $n = 1881$ cities from RAVEL after filtering entities with missing country labels.

**City-language (factual, medium granularity).** Each city maps to its primary official language: $K = 50$ classes. $n = 1859$ cities from RAVEL. The variation in entity counts across hierarchies (2039 vs. 1881 vs. 1859) reflects hierarchy-specific label availability in the RAVEL dataset.

## 3.3 Probe Training and Quality Gates

For each hierarchy-layer combination, we train an $L_2$-regularized logistic regression probe $f_{\text{probe}}$ (scikit-learn, `LogisticRegression` with `solver='lbfgs'`):

- **Regularization:** $C \in \{0.01, 0.1, 1.0, 10.0\}$ selected by 5-fold stratified cross-validation.
- **Split:** 80/20 stratified train-test split, `seed=42`.
- **Input:** Residual stream activation $\mathbf{x}^{(L)}$ at the last token position ($\text{position} = -1$) for RAVEL hierarchies; position $-1$ (sklearn) or $-2$ (`sae_spelling` LinearProbe) for first-letter.
- **Metric:** Weighted macro-F1.

**Probe types for first-letter.** Two probing approaches exist for first-letter. The `sae_spelling` pipeline (Chanin et al., 2024) trains a `LinearProbe` on ICL-formatted spelling prompts at position $-2$. Our sklearn logistic regression uses position $-1$ with the same residual stream activations used for RAVEL hierarchies. At layer 24, the sklearn probe achieves F1 = 0.97, while the `sae_spelling` probe achieves F1 = 0.87 on its own evaluation set. For first-letter absorption measurement, the actual experiments use ICL-formatted probes achieving F1 $\geq$ 0.97, providing a near-perfect denominator for false negative detection. The sklearn probes in Table 2 serve as the cross-hierarchy comparison baseline.

**Quality gates.** We enforce two thresholds: a strict gate (F1 $\geq$ 0.90) and a relaxed gate (F1 $\geq$ 0.85). Table 2 reports probe quality across all hierarchy-layer combinations using the consistent sklearn probing setup. Only first-letter probes at layers 18 (F1 = 0.94) and 24 (F1 = 0.97) pass the strict gate. RAVEL probes achieve F1 = 0.79--0.84 at layer 24, their best layer. The probe quality confound is analyzed in Section 4.3: probe quality correlates with false negative rate at $\rho = -0.756$ ($p < 0.001$), meaning lower-quality probes miss more correct raw-condition classifications, potentially masking or inflating absorption rates.

**Table 2: Probe quality (weighted F1) across hierarchies and layers.** Bold entries pass the strict quality gate (F1 $\geq$ 0.90). Only first-letter probes at layers 18 and 24 pass strict gate. RAVEL probes peak at layer 24 but remain below strict gate.

| | Layer 6 | Layer 12 | Layer 18 | Layer 24 |
|---|---|---|---|---|
| **First-letter** | 0.69 | 0.31 | **0.94** | **0.97** |
| **City-continent** | 0.65 | 0.79 | 0.84 | 0.84 |
| **City-country** | 0.35 | 0.62 | 0.78 | 0.79 |
| **City-language** | 0.52 | 0.69 | 0.81 | 0.82 |

![Probe quality heatmap. Checkmarks indicate strict gate passage (F1 >= 0.90); tilde indicates relaxed gate (0.85--0.90).](figures/tab1_probe_quality.pdf)

## 3.4 Absorption Measurement

Our absorption measurement pipeline adapts Chanin et al. (2024) to support arbitrary feature hierarchies.

**Step 1: False negative detection.** For each token $t$ in the evaluation set, we compute:
$$\hat{y}_{\text{raw}} = f_{\text{probe}}(\mathbf{x}^{(L)}), \quad \hat{y}_{\text{sae}} = f_{\text{probe}}(\hat{\mathbf{x}})$$
where $\hat{\mathbf{x}} = \text{SAE}(\mathbf{x}^{(L)})$. A token is a false negative if $\hat{y}_{\text{raw}} = y$ (probe correctly classifies raw activation) but $\hat{y}_{\text{sae}} \neq y$ (probe misclassifies SAE-reconstructed activation). The set $\text{FN}$ contains all such tokens.

**Step 2: Integrated-gradients attribution.** For each false negative, we compute the attribution of each active SAE latent $i$ to the probe's classification change using integrated gradients (Sundararajan et al., 2017) with 10 interpolation steps between a zero baseline and the actual latent activation vector $\mathbf{a}$:
$$\text{IG}_i(t) = a_i \cdot \int_0^1 \frac{\partial f_{\text{probe}}(\text{SAE}_{\text{dec}}(\alpha \cdot \mathbf{a}))}{\partial a_i} \, d\alpha$$

**Step 3: Absorption detection.** A false negative is classified as absorbed if the highest-attribution latent $i^*$ satisfies two conditions simultaneously:
1. **Cosine alignment:** $\cos(\mathbf{d}_{i^*}, \mathbf{w}_y) > \tau_{\cos}$ (default $\tau_{\cos} = 0.025$), where $\mathbf{d}_{i^*}$ is the decoder direction and $\mathbf{w}_y$ is the probe weight for the true class.
2. **Magnitude gap:** The IG attribution of $i^*$ exceeds the second-highest attribution by a factor $\geq \tau_{\text{gap}}$ (default $\tau_{\text{gap}} = 1.0$).

These threshold defaults follow Chanin et al. (2024). We validate robustness in Section 7.4.

**Absorption rate.** $\text{AR}$ is the fraction of $K$ classes for which at least one absorbed false negative exists:
$$\text{AR} = \frac{|\{k : \exists t \in \text{FN}_k \text{ classified as absorbed}\}|}{K}$$

This class-level metric means a single absorbed token per class counts the same as many. For hierarchies with large $K$ and small per-class samples (city-country, $K = 80$), individual false positives in absorption detection could inflate AR disproportionately. We note this sensitivity in Section 4.3.

**Statistical tests.** All absorption rates are accompanied by bootstrap 95% confidence intervals (10,000 resamples, `seed=42`). Cross-domain comparisons use paired permutation tests (10,000 permutations) with Cohen's $d$ effect sizes. The overall hierarchy effect is assessed via a Kruskal-Wallis test.

## 3.5 Activation Patching

Activation patching provides interventional causal evidence for feature suppression, complementing the correlational IG-based absorption detection. We conduct patching at layer 12 (Gemma Scope JumpReLU 16k), where the `sae_spelling` ICL probe provides reliable ground truth for first-letter classification. The choice of layer 12 rather than layer 24 (where absorption is highest) reflects the chronology of experimental design: patching experiments were conducted before the cross-layer analysis revealed the L24 absorption spike. Extending patching to L24 is a priority for future work (Section 8.6).

**Procedure.** For each word $t$ with detected absorption:

1. Encode $\mathbf{x}^{(12)}$ through the SAE to obtain latent activations $\mathbf{a}$ and reconstructed activation $\hat{\mathbf{x}}$.
2. Identify the highest-IG child latent $c$ responsible for the false negative.
3. **Treatment:** Set $a_c := 0$ (zero the child feature), recompute $\hat{\mathbf{x}}' = \text{SAE}_{\text{dec}}(\mathbf{a}')$, and evaluate $f_{\text{probe}}(\hat{\mathbf{x}}')$.
4. **Control:** Zero 15 magnitude-matched random latents (matched on $|a_i|$ to within 10%) and evaluate recovery.
5. Repeat across 200 contexts per word.

**Recovery rate.** $\text{RR}_{\text{child}}$ is the fraction of absorbed tokens (those with $\geq 3$ absorbed instances) where zeroing the child feature causes the probe to recover the correct parent prediction. The control recovery rate $\text{RR}_{\text{ctrl}}$ is the mean recovery across 15 random-feature ablations.

**Word selection.** We tested 25 words total: 7 pilot core words selected from high-frequency tokens in the `sae_spelling` evaluation set, and 18 additional words discovered via IG-guided search for tokens exhibiting large attribution gaps between the top latent and the rest. Of 25 tested words, 19 exhibited $\geq 3$ absorbed instances and entered the statistical analysis. The discovery procedure biases toward finding absorption; the 58.8% absorption rate among tested words reflects this selection, not a corpus-wide prevalence estimate.

**Statistical tests.** Wilcoxon signed-rank test (one-sided: child recovery $>$ control recovery), Mann-Whitney $U$ test, paired $t$-test, and bootstrap confidence interval on $\Delta\text{RR} = \text{RR}_{\text{child}} - \text{RR}_{\text{ctrl}}$.

## 3.6 Hedging Decomposition

We decompose false negatives into three categories to distinguish genuine absorption from resolution at higher sparsity:

1. **Strict hedging ($H_{\text{strict}}$):** The parent feature $p$ itself reactivates when the SAE operates at a higher $L_0^{\text{target}}$, recovering the correct probe prediction.
2. **Compensatory resolution ($H_{\text{comp}}$):** The false negative resolves at $L_0^{\text{target}}$, but through other (non-parent) latents adding sufficient information -- the parent feature does not reactivate.
3. **Persistent ($H_{\text{persist}}$):** The false negative persists even at $L_0^{\text{target}}$.

We test at $L_0^{\text{base}} = 22$ (using the SAEBench BatchTopK SAE at layer 12) with $L_0^{\text{target}} = 176$ (8$\times$ multiplier), and additionally at $L_0^{\text{base}} = 82$ with $L_0^{\text{target}} = 176$ for sensitivity analysis. The widely cited "hedging rate" (Chanin et al., 2025) corresponds to our loose rate: $H_{\text{strict}} + H_{\text{comp}} = 1 - H_{\text{persist}}$. Our decomposition reveals what fraction of this loose hedging is attributable to the parent feature itself versus unrelated compensatory features.

With this pipeline, we first examine absorption variation across layers (Section 4), then provide causal evidence and hedging decomposition (Section 5), compare architectures (Section 6), and report negative results (Section 7).

---

# 4 Cross-Domain and Cross-Layer Absorption

We apply the measurement pipeline described in Section 3 to four feature hierarchies across eight Gemma Scope SAE configurations. Three results emerge: (1) first-letter absorption varies 15-fold across model layers, from 2.2% at layer 18 to 34.5% at layer 24; (2) measured absorption rates differ significantly across hierarchy types (Kruskal-Wallis $p = 0.005$), with four of six pairwise comparisons reaching significance; and (3) probe quality correlates strongly with false negative rate ($\rho = -0.756$, $p < 0.001$), confounding absolute cross-domain rate comparisons.

## 4.1 Layer Dependence

First-letter absorption provides the cleanest measurement because probes achieve F1 $\geq$ 0.97 using ICL-formatted prompts, eliminating probe quality as a confound. Table 3 reports absorption rates across all eight Gemma Scope JumpReLU configurations.

**Table 3: First-letter absorption rates across layers and SAE widths.** Probes use ICL-formatted pipeline (F1 $\geq$ 0.97). Bootstrap 95% CI from 10,000 resamples. $n = 222$ test words, 25/26 letters covered.

| Config | $\text{AR}$ (%) | Strict $\text{AR}$ (%) | $n_{\text{FN}}$ / $n_{\text{correct}}$ | 95% CI |
|--------|:---:|:---:|:---:|:---:|
| L6, 16k | 2.4 | 0.0 | 4 / 169 | [0.6, 14.4] |
| L6, 65k | 2.4 | 0.0 | 4 / 166 | [0.6, 14.7] |
| L12, 16k | 5.7 | 1.4 | 8 / 141 | [2.0, 8.1] |
| L12, 65k | 9.2 | 5.0 | 13 / 141 | [4.1, 13.4] |
| L18, 16k | **2.2** | 0.0 | 4 / 183 | [0.4, 4.0] |
| L18, 65k | 4.5 | 0.0 | 8 / 177 | [0.9, 8.1] |
| L24, 16k | **34.5** | 17.2 | 30 / 87 | [21.3, 49.5] |
| L24, 65k | **25.5** | 17.0 | 24 / 94 | [16.7, 38.3] |

Absorption rates at layer 24 (25--35%) exceed rates at layers 6, 12, and 18 (2--9%) by a factor of 15-fold. The minimum rate is 2.2% (L18, 16k) and the maximum is 34.5% (L24, 16k). This variation is unconfounded: ICL-formatted probes achieve F1 $\geq$ 0.97 at every layer, so the denominator of the absorption rate (correctly classified tokens on raw activations) is comparably reliable throughout.

Layer 24 rates (25--35%) align with the 15--35% range reported by Chanin et al. (2024), suggesting that prior work -- which measured at a single, unspecified layer -- likely evaluated at the model's later layers. The absorption surge at layer 24 is consistent with the model resolving its final token prediction at the last residual stream positions, where parent-child feature competition intensifies.

At layers 6 and 18, absorption is minimal (2.2--4.5%). At layer 12, intermediate rates (5.7--9.2%) coincide with the wider SAE ($M = 65{,}536$) showing higher absorption than the narrower one ($M = 16{,}384$), consistent with larger dictionaries creating more opportunities for child features to absorb parent information.

Figure 3 visualizes this layer dependence. The L24 spike dominates; layers 6 and 18 are barely distinguishable from zero.

![First-letter absorption rates across model layers and SAE widths. Layer 24 shows 25--35% absorption, while layers 6, 12, and 18 remain below 10%. Error bars indicate bootstrap 95% confidence intervals.](figures/fig3_layer_absorption.pdf)

## 4.2 Cross-Domain Variation

We measure absorption on three RAVEL entity-attribute hierarchies at layer 24, where probe quality is highest for all hierarchy types. Table 4 reports the cross-domain results alongside first-letter baselines at the same layer and SAE configurations.

**Table 4: Cross-domain absorption rates at layer 24.** Each RAVEL hierarchy is compared to first-letter at the same SAE configuration. Permutation test $p$-values from 10,000 permutations. The city-country L24-16k confidence interval is approximate due to bootstrap estimation variance.

| Hierarchy | SAE | $\text{AR}$ (%) | 95% CI | Probe F1 | vs. First-letter $\Delta$ | $d$ | $p$ |
|-----------|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| First-letter | L24, 16k | 34.5 | [21.3, 49.5] | 0.971 | -- | -- | -- |
| First-letter | L24, 65k | 25.5 | [16.7, 38.3] | 0.971 | -- | -- | -- |
| City-continent | L24, 16k | 35.8 | [16.2, 59.7] | 0.843 | +1.4 | 0.31 | 0.829 |
| City-continent | L24, 65k | 26.0 | [8.9, 47.8] | 0.843 | +0.5 | 0.12 | 0.932 |
| City-country | L24, 16k | 18.5 | [11.9, 30.7] | 0.789 | **-16.0** | -3.84 | **0.004** |
| City-country | L24, 65k | 12.7 | [7.2, 24.1] | 0.789 | **-12.8** | -3.51 | **0.008** |
| City-language | L24, 16k | 13.6 | [7.2, 25.4] | 0.823 | **-20.8** | -5.16 | **<0.001** |
| City-language | L24, 65k | 13.6 | [7.2, 35.4] | 0.823 | **-11.9** | -3.24 | **0.015** |

The Kruskal-Wallis test across all four hierarchy types yields $p = 0.005$, confirming that absorption rates differ significantly by hierarchy. Four of six pairwise comparisons (city-country and city-language vs. first-letter at both widths) are significant at $p < 0.05$ with large effect sizes ($|d| > 3.0$). City-continent absorption (26--36%) is statistically indistinguishable from first-letter at L24 ($p > 0.8$), while city-country (13--19%) and city-language (14%) fall significantly below first-letter.

Layer-hierarchy interactions are non-trivial: absorption rankings across hierarchy types depend on which model layer is measured. At layer 12, pilot data suggested that semantic hierarchies show higher absorption than first-letter, but at layer 24, first-letter matches or exceeds all RAVEL hierarchies.

## 4.3 The Probe Quality Confound

Probe quality varies substantially across hierarchies (Table 2): first-letter achieves F1 = 0.971, while RAVEL probes range from F1 = 0.789 (city-country) to 0.843 (city-continent). This variation confounds the cross-domain comparison.

Probe quality correlates strongly with false negative rate ($\rho = -0.756$, $p < 0.001$). A higher-quality probe correctly classifies more tokens in the raw-activation condition, producing a larger denominator for the absorption rate. Lower-quality probes miss correct classifications even before SAE encoding, potentially masking absorption events that exist but are undetectable.

Three specific confounds deserve explicit acknowledgment:

1. **Denominator asymmetry.** First-letter probes correctly classify 87--183 tokens (depending on layer), providing a large pool in which to detect false negatives. RAVEL probes at L24 correctly classify fewer tokens (e.g., $n = 200$ for city-continent), meaning each individual false negative has a larger marginal impact on the absorption rate.

2. **Missed absorption.** If a RAVEL probe misclassifies a token on raw activations, any subsequent SAE-induced failure on that token is invisible to the measurement pipeline. The absorption rate is therefore a lower bound on true absorption for low-quality probes.

3. **Spurious false negatives.** Conversely, probe errors in the SAE-reconstructed condition could create false negatives that do not reflect genuine absorption, inflating the rate. This effect is bounded by the probe's overall error rate.

These confounds do not invalidate the finding that absorption rates differ across hierarchies -- the Kruskal-Wallis $p = 0.005$ is robust to the direction of probe-quality bias. However, the absolute magnitude of cross-domain rates carries quantitative uncertainty that cannot be resolved without higher-quality probes or probe-independent measurement methods.

Figure 4 presents the full layer-hierarchy absorption interaction as a heatmap, showing data at all available configurations. Cells marked "--" indicate hierarchy-layer combinations where RAVEL probes were not trained (RAVEL probes were measured only at L24, the best-performing layer).

![Absorption rate heatmap across hierarchy types and model layers for JumpReLU 16k (left) and 65k (right). First-letter rates span all four layers; RAVEL hierarchies are measured at layer 24 only. Color intensity encodes absorption rate. Missing cells indicate hierarchy-layer combinations without trained probes.](figures/fig4_crossdomain_heatmap.pdf)

Having established that absorption rates vary by layer and hierarchy, we now provide causal evidence for the absorption mechanism and decompose the role of hedging.

---

# 5 Causal Evidence and Hedging Decomposition

## 5.1 Activation Patching

Activation patching at layer 12 (Gemma Scope JumpReLU 16k) confirms that child features causally suppress parent-class information. Zeroing the highest-IG child latent in SAE output recovers correct parent-class probe predictions at a mean rate of 32.5% across 19 words with detected absorption, compared to 1.5% for magnitude-matched control features ($\Delta\text{RR} = 0.310$, 95% CI [0.213, 0.421], Wilcoxon $p = 0.000218$, Cohen's $d = 1.33$).

Of 19 words entering the analysis, 16 show positive recovery when the child feature is zeroed. The largest effects occur for conmigo (100% recovery, raw accuracy 0.4), wikk (66.7% recovery), backward (50.0% recovery), and zorgt (50.0% recovery). Table 5 reports per-word results.

**Table 5: Activation patching results per word.** Recovery rate ($\text{RR}_{\text{child}}$) when zeroing the highest-IG child latent vs. control ($\text{RR}_{\text{ctrl}}$, mean of 15 magnitude-matched random latents). Source: "pilot" = 7 core words from `sae_spelling` evaluation; "discovered" = 18 words found via IG-guided search. Words with $<$3 absorbed instances excluded from analysis.

| Word | Letter | Source | Raw Acc. | $n_{\text{absorbed}}$ | $\text{RR}_{\text{child}}$ | $\text{RR}_{\text{ctrl}}$ |
|------|--------|--------|:---:|:---:|:---:|:---:|
| conmigo | c | discovered | 0.40 | 3 | **1.000** | 0.000 |
| wikk | w | discovered | 0.60 | 3 | **0.667** | 0.000 |
| backward | b | discovered | 0.80 | 4 | **0.500** | 0.000 |
| zorgt | z | discovered | 0.60 | 4 | **0.500** | 0.000 |
| yaitu | y | discovered | 1.00 | 11 | **0.455** | 0.018 |
| snake | s | pilot | 1.00 | 8 | **0.375** | 0.017 |
| stone | s | pilot | 1.00 | 6 | **0.333** | 0.011 |
| nought | n | discovered | 0.80 | 5 | **0.400** | 0.027 |
| knight | k | pilot | 1.00 | 7 | **0.286** | 0.010 |

(Table truncated to top-recovery words; full results in Appendix.)

**Discovery procedure transparency.** The 7 pilot words plus 18 IG-discovered words together biased the sample toward finding absorption. The 58.8% absorption rate (19 of 25 tested words showing $\geq$3 absorbed instances) reflects this selection, not a corpus-wide prevalence estimate. The causal test is valid within these words: the question is whether zeroing the identified child feature recovers the parent prediction, not whether arbitrary words exhibit absorption.

**Restricted analysis.** Among the 12 words with raw accuracy $\geq$ 0.50, mean child recovery is 38.2% vs. 1.1% control ($\Delta = 0.371$, $p < 0.001$), confirming that the causal effect is not driven by poorly represented tokens.

## 5.2 Tightened Hedging Decomposition

At base $L_0 = 22$ (SAEBench BatchTopK at layer 12) with target $L_0 = 176$ (8$\times$ multiplier), we decompose $n = 304$ first-letter false negatives into three categories:

- **Strict hedging:** 7.9% -- the parent feature $p$ itself reactivates at $L_0^{\text{target}}$, recovering the correct probe prediction.
- **Compensatory resolution:** 86.2% -- the false negative resolves, but through non-parent latents adding sufficient information.
- **Persistent:** 5.9% -- the false negative persists at $L_0^{\text{target}}$.

The widely cited ${\sim}$98% hedging rate (Chanin et al., 2025) corresponds to our loose rate: $H_{\text{strict}} + H_{\text{comp}} = 94.1\%$. Our decomposition reveals that 86.2% of this "hedging" is compensatory -- unrelated features adding information at higher $L_0$ -- not genuine parent recovery. The near-tautological nature of the loose hedging metric becomes clear: expanding the dictionary by 8$\times$ naturally activates more latents, and the combinatorial probability of at least one combination restoring the correct classification is high regardless of whether the parent feature itself recovers.

At the more conservative base $L_0 = 82$ with target $L_0 = 176$ (2.1$\times$ multiplier), strict hedging rises to 12.3% while compensatory resolution drops to 78.4%, consistent with the interpretation that the distinction between strict and compensatory hedging depends on the ratio between base and target sparsity.

We briefly examine whether SAE architecture modulates absorption across hierarchy types.

---

# 6 Architecture Comparison

At layer 12 -- the only layer with SAEs available for all four architectures -- we compare JumpReLU (16k and 65k), BatchTopK (16k), and Matryoshka (32k) across all four feature hierarchies. Table 6 reports the results.

**Table 6: Absorption rates (%) by architecture at layer 12 across four hierarchies.** Bold = lowest rate per hierarchy. BatchTopK operates at $L_0 = 20$ vs. JumpReLU $L_0 \approx 75$--87, confounding the comparison by sparsity level.

| Hierarchy | JumpReLU 16k | JumpReLU 65k | BatchTopK 16k | Matryoshka 32k |
|-----------|:---:|:---:|:---:|:---:|
| First-letter | **0.7** | 1.3 | 3.4 | 1.4 |
| City-continent | 17.3 | 23.1 | **13.5** | 19.2 |
| City-language | 41.2 | 38.2 | 61.8 | **35.3** |
| City-country | 47.1 | 47.1 | 52.9 | **35.3** |

The architecture effect is not significant (Kruskal-Wallis $p = 0.87$), while the hierarchy effect is significant ($p = 0.005$). Hierarchy type explains more variance in absorption than architecture choice. At this sample size ($N = 16$ observations), the minimum detectable effect for the Kruskal-Wallis test is large; the test is uninformative for moderate architecture differences.

Matryoshka shows the lowest rates on 3 of 4 hierarchies (city-country, city-language, and near-lowest on city-continent), but these differences are not statistically significant. BatchTopK operates at $L_0 = 20$ vs. JumpReLU $L_0 \approx 75$--87, so the comparison is confounded by sparsity level. RAVEL probes at L12 are at their worst (F1 = 0.52--0.69), making the cross-domain architecture comparison doubly confounded by probe quality and sparsity.

Current architecture evaluations for absorption resistance -- including those for Matryoshka SAE (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), ATM-SAE (Li et al., 2025), and masked regularization (Narayanaswamy et al., 2026) -- all benchmark on first-letter only. If architecture rankings change across hierarchy types, as these results tentatively suggest, single-task rankings may mislead practitioners.

We also report three failed attempts at unsupervised absorption detection, which inform future detector design.

---

# 7 Negative Results

Three proposed unsupervised absorption detectors fail definitively. We report each with the same rigor as positive results, because these negative findings prevent others from pursuing approaches that our data show to be ineffective.

## 7.1 Geometric Absorption Score (GAS)

The Geometric Absorption Score measures decoder-activation co-occurrence mismatch between feature pairs. At 25$\times$ scale-up (5,000 sequences, 640k tokens), GAS achieves $\rho = 0.116$ ($p = 0.58$) against per-letter absorption rates and AUROC = 0.571 as a binary absorption classifier. Decoder geometry captures the *potential* for absorption -- which features have overlapping decoder directions -- but not which features are *actually* suppressed during encoding. The frequency asymmetry term in GAS introduces noise for rare features whose activation statistics are poorly estimated.

## 7.2 Conditional Mutual Information (CMI)

At $L_0 = 22$, CMI between parent and child activations yields $\rho = 0.044$ ($p = 0.83$) against absorption rates. The binary CMI formulation ($A = 0$ vs. $A > 0$) discards fine-grained activation magnitude information that distinguishes genuine suppression from low-magnitude non-firing. A continuous formulation might perform differently but was not tested.

## 7.3 Absorption Tax Ranking

The theoretical Absorption Tax $T(G)$ predicts per-hierarchy absorption severity based on child feature priors $\pi_c$ and parent-child redundancy ratios $R_{pc}$. Its ranking prediction achieves Spearman $\rho = -0.20$, Kendall $\tau = 0.0$, and pairwise concordance of 50% (chance level). Per-letter $R_{pc}$ vs. observed absorption rate yields $\rho = 0.09$--$0.17$ across all eight SAE configurations (all non-significant). The cosine-squared redundancy ratio formulation captures static decoder geometry but not the dynamic competition during encoding that determines which features are actually absorbed.

## 7.4 Threshold Sensitivity

A 5$\times$4 sensitivity grid ($\tau_{\cos} \in \{0.01, 0.02, 0.025, 0.03, 0.05\}$, $\tau_{\text{gap}} \in \{0.5, 1.0, 1.5, 2.0\}$) confirms that false negatives remain constant at $n = 87$ across all 20 cells (this count corresponds to the L24-16k configuration), with AR varying from 11.8% to 15.1% (CV = 0.077). Absorption is structural, not a detection-threshold artifact. Probe quality ($\rho = -0.756$) is a stronger predictor of measured absorption than any detection threshold setting.

These results -- both positive and negative -- reshape our understanding of feature absorption as a fundamental SAE failure mode.

---

# 8 Discussion

## 8.1 Practical Implications for SAE Evaluation

Single-task, single-layer absorption benchmarks understate the problem's complexity. First-letter absorption at layer 12 ranges from 0.7% to 3.4% across four SAE architectures (Table 6), yet the same metric at layer 24 reaches 25.5--34.5% on the same model (Table 3, Figure 3). Any evaluation that measures absorption at one layer and reports it as a property of the SAE systematically mischaracterizes the failure mode. SAEBench (Karvonen et al., 2025) and the sae-spelling benchmark (Chanin et al., 2024) both operate at a single layer; our results demonstrate that the layer choice alone can shift the measured rate by 15-fold.

Cross-domain variation compounds this problem. At layer 24, first-letter absorption (34.5% on 16k-width, 25.5% on 65k-width) differs significantly from city-country (18.5%, 12.7%) and city-language (13.6%, 13.6%), with 4 of 6 pairwise comparisons reaching statistical significance (Kruskal-Wallis $p = 0.005$; pairwise permutation $p = 0.0001$ to 0.015). A benchmark reporting "15--35% absorption" based on first-letter spelling at one layer conveys a false impression of precision: the actual rate depends on which hierarchy the user cares about and where in the network it is measured.

These findings have direct consequences for architecture comparison. The four architectures tested at layer 12 show no significant absorption difference (Kruskal-Wallis $p = 0.87$), while the hierarchy effect at L24 is significant ($p = 0.005$). The architecture null result and the hierarchy significance come from different layers and probe qualities, so they should not be directly compared; however, they jointly suggest that hierarchy type explains more variance in measured absorption than architecture choice under current evaluation conditions.

The practical recommendation: absorption evaluation should report rates across at least two hierarchy types at multiple layers, with probe quality documented alongside each measurement.

## 8.2 Layer-Position Mechanism Hypothesis

The absorption surge at layer 24 -- where first-letter rates jump from 2.2--9.2% at layers 6--18 to 25.5--34.5% (Figure 3) -- coincides with the final residual stream position in Gemma 2 2B's 26-layer architecture. This pattern is consistent with a representation-sharpening mechanism: at earlier layers, the residual stream carries distributed representations that SAEs can encode without strong hierarchical competition; at layer 24, the representation sharpens toward specific token predictions, concentrating parent-child information overlap and creating the conditions for child features to absorb parent features.

Two pieces of correlational evidence support this interpretation. RAVEL probe quality peaks at layer 24 for all three entity-attribute hierarchies (city-continent F1 = 0.843, city-country F1 = 0.789, city-language F1 = 0.823; all sklearn probes), indicating that factual knowledge is most linearly accessible at this layer. The absorption-hedging decomposition shows that L24 has a lower proportion of absorbed (vs. hedged) false negatives compared to earlier layers -- 50.0% absorbed at L24-16k versus 100.0% at L6-16k (Table 7 in Section 4.4 of the extended results) -- suggesting that late-layer absorption competes with hedging as both failure modes intensify.

An alternative explanation is that layer 24 SAEs have different effective $L_0$ values or reconstruction quality that drives the absorption increase. Disentangling the representation-sharpening hypothesis from confounds in SAE training quality at different layers requires controlled experiments beyond the scope of this paper. Two falsifiable predictions follow from the hypothesis: (a) in models with more layers, the absorption spike should appear at the penultimate layer rather than a fixed layer index; (b) residual stream entropy or norm metrics should show measurable sharpening at the same layer where absorption spikes.

## 8.3 Absorption as Intrinsic to Sparse Coding

The activation patching result -- 32.5% recovery when child features are zeroed versus 1.5% for controls ($\Delta\text{RR} = 0.310$, 95% CI [0.213, 0.421], Wilcoxon $p = 0.000218$, Cohen's $d = 1.33$; Table 5 reports per-word details) -- provides the first interventional causal evidence that feature absorption reflects genuine feature suppression, not merely a measurement artifact.

Combined with the layer dependence finding, this suggests that absorption is an intrinsic property of SAE encoding under sparsity constraints. The absorption-reconstruction tradeoff is fundamental: eliminating absorption for a parent-child pair requires the SAE to activate both the parent and child latents, costing $+1$ $L_0$ per pair. At layer 24, where representations are sharpest and hierarchical overlap is maximal, this cost is paid most frequently. The three failed unsupervised detectors (GAS: $\rho = 0.116$; CMI: $\rho = 0.044$; Absorption Tax ranking: $\rho = -0.20$) reinforce that absorption is not a simple geometric or information-theoretic property of the decoder -- it emerges from the interaction between encoder dynamics, sparsity pressure, and input statistics.

The tightened hedging decomposition supports this: at base $L_0 = 22$, only 7.9% of false negatives exhibit strict hedging (parent recovery at 8$\times$ $L_0$), while 86.2% resolve through compensatory features. The widely cited ${\sim}$98% hedging rate conflates parent recovery with the near-tautological observation that expanding the dictionary by 8$\times$ resolves most false negatives through any available feature.

This framing has a specific implication: absorption cannot be eliminated by architectural changes alone (architecture effect $p = 0.87$ at L12), but may be mitigated by operating at higher $L_0$ or by developing training objectives that explicitly penalize parent-child suppression.

## 8.4 Limitations

We order these by severity of impact on the paper's central claims.

**1. RAVEL probes below strict quality gate (most consequential).** The best RAVEL probes reach F1 = 0.843 (city-continent at layer 24), below the strict 0.90 quality gate. As established in Section 4.3, probe quality correlates strongly with false negative rate ($\rho = -0.756$), and the net direction of bias on absolute rates is unclear. Absolute cross-domain absorption rates carry quantitative uncertainty. The qualitative finding -- that rates differ significantly across hierarchies -- is supported by the Kruskal-Wallis test ($p = 0.005$), but the exact magnitude of each rate should be interpreted with caution.

**2. Activation patching restricted to first-letter at layer 12.** All 25 patching targets are first-letter absorption pairs at layer 12 with probe F1 = 0.97. Cross-domain activation patching was not attempted because no RAVEL probe passes the 0.85 quality gate required for reliable causal inference. The strongest absorption signal is at layer 24 (34.5%), but patching was conducted at layer 12 -- the layer available at the time of experimental design. Whether the causal mechanism generalizes to entity-attribute hierarchies and to the high-absorption layer remains unknown.

**3. Architecture comparison underpowered at layer 12 only.** The four-architecture comparison is limited to layer 12, the only layer with SAEs available for all architectures. At layer 12, RAVEL probes are at their worst (F1 = 0.52--0.69), making the cross-domain architecture comparison doubly confounded. At $N = 16$ observations, the test is uninformative for moderate effects.

**4. Single model family.** All experiments use Gemma 2 2B with Gemma Scope JumpReLU SAEs. Generalization to other model families (Llama, Mistral), model scales (9B, 27B), and independently trained SAEs has not been tested.

**5. Training-free evaluation only.** Whether retraining SAEs with hierarchy-aware losses, modified sparsity schedules, or explicit anti-absorption regularization would change the observed patterns is an open question.

## 8.5 Future Directions

**Degraded-probe ablation (highest priority).** Injecting calibrated label noise into first-letter probes to simulate RAVEL-level quality (target F1 levels: 0.70, 0.80, 0.85, 0.90, 0.97) would quantify the probe-quality confound directly. If absorption rates on degraded first-letter probes match RAVEL rates, the cross-domain variation is largely a probe artifact; if they remain distinct, the hierarchy effect is genuine. This experiment requires only retraining probes with noise injection, costing minimal GPU-hours.

**Cross-domain activation patching.** Improving RAVEL probes above the 0.85 quality gate -- through better prompt templates, larger entity sets, or probe architectures beyond logistic regression -- would enable activation patching on entity-attribute hierarchies.

**Multi-model validation.** Gemma 2 9B and 27B have Gemma Scope SAEs available at multiple layers. Replicating the layer-dependent absorption pattern across model scales would establish whether the layer 24 spike is universal. Cross-family replication on Llama 3.1 with independently trained SAEs would further strengthen generalizability.

**Unsupervised detection.** All three unsupervised detectors tested here fail. Developing detectors that capture encoder competitive dynamics -- rather than decoder geometry alone -- is an open challenge.

**Safety-relevant feature hierarchies.** The practical motivation for absorption research is that safety-relevant features (deception, manipulation, harmful intent) live in knowledge and reasoning space, not spelling space. Extending absorption measurement to safety-relevant hierarchies -- if suitable probes can be developed -- would directly connect this work to AI safety.

---

# 9 Conclusion

Feature absorption in sparse autoencoders is not a fixed property of the encoder but varies dramatically with model depth and input semantics -- a finding that invalidates all existing single-task, single-layer absorption benchmarks. On Gemma 2 2B with Gemma Scope JumpReLU SAEs, first-letter absorption spans a 15-fold range across model layers (2.2% at layer 18 to 34.5% at layer 24, probe F1 = 0.97), and absorption rates differ significantly across hierarchy types (Kruskal-Wallis $p = 0.005$, 4/6 pairwise comparisons significant). Activation patching at layer 12 provides the first interventional causal evidence for feature suppression: zeroing child features recovers parent probe predictions at 32.5% vs. 1.5% for magnitude-matched controls (Wilcoxon $p = 0.000218$, Cohen's $d = 1.33$).

The widely cited ${\sim}$98% hedging rate is near-tautological: our three-way decomposition reveals that only 7.9% of first-letter false negatives exhibit strict hedging (parent feature recovery), while 86.2% resolve through compensatory activation of unrelated features at higher $L_0$. Architecture choice does not significantly affect absorption rates at layer 12 (Kruskal-Wallis $p = 0.87$), whereas hierarchy type dominates ($p = 0.005$) -- suggesting that the input's semantic structure governs absorption more than the SAE's training objective.

Three proposed unsupervised absorption detectors -- GAS ($\rho = 0.116$, AUROC = 0.571), CMI ($\rho = 0.044$, $p = 0.83$), and the Absorption Tax ranking ($\rho = -0.20$, concordance 50%) -- all fail. Absorption currently requires supervised, probe-based measurement, which inherits probe quality as a binding confound ($\rho = -0.756$, $p < 0.001$; best RAVEL probe F1 = 0.843).

These results connect back to the practical stakes motivating this work. The 10--40% safety-feature degradation reported by Lieberum et al. (2024) may be layer-dependent: our data suggest that early-layer SAEs (absorption 2--5%) retain higher feature reliability than late-layer SAEs (absorption 25--35%). Conversely, circuit analyses at late layers -- where absorption is most severe -- should be validated against absorption-aware evaluation before being trusted for safety-critical applications.

Four concrete directions follow. First, a degraded-probe ablation study would disentangle the probe-quality confound from genuine cross-domain variation. Second, extending activation patching to RAVEL hierarchies requires probes above the 0.90 quality gate. Third, replicating the layer-dependent pattern on Gemma 2 9B/27B and Llama 3.1 would establish generalizability. Fourth, the failure of all three unsupervised detectors suggests that future approaches should incorporate encoder dynamics -- not just decoder geometry -- since absorption is a property of how the encoder allocates activation among competing features, not of the decoder directions alone. Extending measurement to safety-relevant feature hierarchies would directly connect this empirical characterization to the AI safety applications that motivate the field.

---

## Figures and Tables

- Figure 1: fig1_heatmap.pdf -- Layer x hierarchy absorption heatmap (central finding)
- Figure 2: fig2_pipeline_desc.md -- Experimental pipeline schematic (method overview)
- Figure 3: fig3_layer_absorption.pdf -- First-letter absorption across 4 layers and 2 SAE widths with bootstrap CI
- Figure 4: fig4_crossdomain_heatmap.pdf -- Cross-domain absorption heatmap (hierarchy x layer x width)
- Table 1: inline -- SAE configurations evaluated
- Table 2: inline + tab1_probe_quality.pdf -- Probe quality across hierarchies and layers
- Table 3: inline -- First-letter absorption rates across all 8 Gemma Scope configs
- Table 4: inline -- Cross-domain absorption rates at L24 with statistical tests
- Table 5: inline -- Activation patching per-word results
- Table 6: inline -- Architecture comparison at layer 12
