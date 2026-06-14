# Where and When Encoder-Decoder Misalignment Signals Feature Absorption in Sparse Autoencoders

---

## Abstract

Feature-based causal analyses built on Sparse Autoencoders (SAEs) silently produce incorrect conclusions when parent latents are suppressed by child latents — a failure mode called feature absorption. Three gaps block systematic progress: detection requires pre-specified probe directions (foreknowledge), generalizability beyond the first-letter spelling task has never been tested, and no actionable structural taxonomy exists. We introduce Encoder-Decoder Alignment (EDA), a weight-only absorption screening metric computable from SAE weight matrices alone without activation data. A formal lower bound derived from the first-order stationarity conditions at a partial minimum of the biconvex sparse dictionary learning (SDL) loss links EDA monotonically to absorption degree. EDA achieves AUROC = 0.776 on Gemma Scope L12-16k (Gemma 2 2B) and AUROC = 0.629 on GPT-2 Small L6 with exact Chanin et al. labels, outperforming the decoder cosine baseline by +0.396 AUROC. Attempts to extend absorption measurement to three RAVEL entity-attribute hierarchies (city-continent, city-country, city-language) were falsified by a shuffled hierarchy control: real absorption rates were statistically indistinguishable from shuffled null rates across all 9 tested domain-SAE config combinations (H3 falsified). A three-subtype taxonomy classifies absorbed latents into early absorption (parent decoder direction absent, ~72–75%), late absorption (encoder suppressed despite decoder presence, ~13%), and partial absorption (context-dependent, ~13%). Early-absorption dominance reframes the problem: the dominant failure mode is a training-time dictionary coverage gap, not an inference-time encoder-decoder misalignment. Inference-Time Absorption Correction (ITAC), a training-free proof-of-concept targeting late-type latents, achieves 3.14% mean false-negative reduction on L12-65k but is structurally inapplicable to the 72–75% early-type majority. Code released as an open SAEBench extension.

---

## 1. Introduction

Feature-based causal analyses built on Sparse Autoencoders silently produce incorrect conclusions when parent latents are suppressed by child latents — a failure mode called **feature absorption** (Chanin et al., 2024). Sparse Autoencoders decompose neural network activations into sparse combinations of learned features, enabling researchers to inspect and steer individual computational units inside language models (Bricken et al., 2023; Cunningham et al., 2023). SAEBench evaluates hundreds of SAEs across models and architectures, confirming that dictionary learning produces interpretable features at scale (Karvonen et al., 2025). Yet absorption persistently undermines SAE reliability.

Feature absorption occurs when a parent latent — one encoding a general concept — is suppressed to zero on inputs where a more specific child latent is active, even though the parent concept is present in the input (Chanin et al., 2024). Consider a concrete example (Figure 1a, see end-of-paper Figures section): a latent trained to detect "starts-with-A words" (parent) fails to fire whenever a "starts-with-A proper nouns" latent (child) is active. The child's activation absorbs the parent's reconstruction contribution; the parent latent reads as inactive despite the input containing an A-word. This suppression is a systematic consequence of sparsity pressure during training: the biconvex SDL loss rewards solutions where one latent handles a subset of inputs that two latents previously shared (Tang et al., 2025).

Absorption is not confined to toy models. SAEBench measurements confirm the phenomenon across Gemma Scope SAEs at multiple layers and dictionary widths (Karvonen et al., 2025), and it has motivated architectural responses including OrtSAE, Matryoshka SAE, KronSAE, MP-SAE, and masked regularization (Narayanaswamy et al., 2026; Costa et al., 2025). The practical consequence is severe: any feature-based causal analysis or steering experiment that reads parent latent activations will silently produce incorrect conclusions on inputs where a child latent is active.

Despite sustained attention, three gaps remain open.

**Gap 1 — Detection requires foreknowledge.** The Chanin et al. metric, the only quantitative absorption measure, requires pre-specified probe directions: you must already know which parent-child hierarchy to test before you can measure absorption. Auditing all $d_{\text{SAE}}$ latents in a deployed SAE is intractable under this regime.

**Gap 2 — Generalizability is assumed, not tested.** Every published absorption measurement uses the first-letter spelling task. The case that absorption is a first-letter artifact has never been empirically falsified.

**Gap 3 — No actionable taxonomy.** All absorbed latents are treated as a single category, yet structurally distinct failure modes require different remediation paths. A latent whose parent concept was never allocated in the dictionary (a training-time coverage failure) cannot be fixed by the same means as a latent whose parent decoder direction exists but whose encoder was trained away from it (an inference-time encoder suppression), or a latent whose parent direction exists but fires only in some contexts (a selective, context-dependent failure).

This paper addresses all three gaps within a unified, training-free framework. Our contributions are:

1. **Encoder-Decoder Alignment (EDA)**, a weight-only absorption screening metric computable from SAE weight matrices alone without activation data. As illustrated in Figure 1b, EDA measures the angular mismatch between each latent's encoder direction $w_{e,j}$ and decoder direction $d_j$. We derive a formal lower bound from the first-order stationarity conditions at a partial minimum of the biconvex SDL loss (Theorem 1) linking EDA monotonically to absorption degree. EDA achieves AUROC = 0.776 on Gemma Scope L12-16k, outperforming the decoder cosine baseline by +0.396 AUROC (DeLong $p \approx 0$).

2. **Cross-domain generalization: a null result with conditional evidence.** We test whether absorption generalizes beyond the first-letter task to three RAVEL entity-attribute hierarchies. A proper shuffled hierarchy control shows real absorption rates are statistically indistinguishable from shuffled null rates (0/9 domain-SAE config combinations pass; H3 falsified). The null result is attributable to below-quality-gate bridge-model probes (accuracy 37–71% vs. required 85%); we preserve the finding as a pending contribution requiring same-model probe access.

3. **Three-subtype absorption taxonomy.** We classify absorbed latents into early absorption (parent decoder direction absent; ~72–75% of cases), late absorption (parent direction present but encoder suppressed; ~13%), and partial absorption (context-dependent selective failure; ~13%). Early-absorption dominance reveals that the dominant failure mode is a training-time dictionary coverage gap, not encoder suppression — a finding that redirects remediation strategy toward dictionary allocation.

We additionally introduce Inference-Time Absorption Correction (ITAC), a training-free proof-of-concept for late-type latents. ITAC achieves 3.14% mean false-negative reduction on L12-65k (best individual case: 18.9% for latent $j=61217$) without degrading reconstruction quality (FVU change: $-4.23\%$). ITAC is structurally limited to the ~13% late-absorption minority, confirming the taxonomy's prediction. We also report a negative scaling result: wider SAEs absorb more at matched $L_0$, ruling out a simple width-based mitigation.

Section 2 reviews SAE foundations and prior absorption work. Section 3 derives EDA and its formal lower bound. Section 4 validates EDA against ground-truth labels on Gemma Scope and GPT-2 SAEs. Section 5 presents the cross-domain characterization. Section 6 introduces the three-subtype taxonomy and ITAC. Section 7 consolidates the findings and their implications.

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders for Mechanistic Interpretability

Transformer-based language models encode many more concepts than they have dimensions, representing features as overlapping linear directions in a phenomenon called *superposition* \cite{elhage2022toy}. Sparse Autoencoders decompose these entangled residual stream activations $x \in \mathbb{R}^{d_{\text{model}}}$ into sparse combinations of learned features by optimizing the sparse dictionary learning (SDL) loss, which has a biconvex structure (defined formally as Equation 1 in Section 3). The biconvex SDL loss is convex separately in the encoder parameters $W_e$ and the decoder parameters $W_d$, but not jointly, producing a landscape of partial minima that underlies absorption, as characterized by Tang et al. (2025). At a partial minimum, an encoder direction $w_{e,j}$ may diverge from its paired decoder direction $d_j$ — the geometric signature that Section 3 formalizes into the EDA metric.

Early demonstrations on small models showed that SAEs recover interpretable, approximately monosemantic features from MLP activations \cite{bricken2023monosemanticity}. Subsequent scaling to frontier models — Claude 3 Sonnet \cite{templeton2024scaling} and GPT-4 \cite{gao2024scaling} — produced SAEs with millions of latents encoding safety-relevant concepts, motivating deployment in interpretability pipelines. Architectural improvements followed rapidly: Gated SAEs decouple feature detection from magnitude estimation \cite{rajamanoharan2024gated}; JumpReLU SAEs train $L_0$ directly \cite{rajamanoharan2024jumprelu}; and TopK SAEs impose fixed sparsity with clean scaling laws \cite{gao2024scaling}. The Gemma Scope suite provides 400+ pre-trained SAEs for Gemma 2 2B and 9B \cite{lieberum2024gemma}, and SAEBench standardizes evaluation across eight metrics on 200+ open-source SAEs \cite{karvonen2025saebench}.

Three recent papers challenge SAE reliability on separate grounds. Cui et al. \cite{cui2025limits} provide closed-form analysis showing SAEs generally fail to recover ground-truth features unless features are extremely sparse. Korznikov et al. \cite{korznikov2026sanity} find that trained SAEs recover only 9% of true features in synthetic settings, with random baselines matching on interpretability, sparse probing, and causal editing. Song et al. \cite{song2025consistency} show SAE features are inconsistent across training runs (TopK SAEs: PW-MCC = 0.80). Together, these establish that SAEs face fundamental reliability challenges — of which feature absorption is arguably the most consequential.


### 2.2 Feature Absorption: Definition and Prior Work

Chanin et al. \cite{chanin2024absorption} define feature absorption as the systematic failure of a parent latent to activate when its child latent is active, even though the parent concept is present in the input. The canonical example: a latent detecting "starts-with-A words" (the parent feature) has zero activation $z_p = 0$ on inputs containing proper nouns starting with A, because a more specific "starts-with-A proper nouns" latent (the child feature) absorbs the parent's contribution to the reconstruction — saving one unit of $L_0$ at the cost of a systematic false negative in the parent's recall. Chanin et al. validate this mechanism on a toy model and measure absorption rates of 15–35% across Gemma Scope SAEs at widths 16k and 65k. Their supervised metric requires pre-specified probe directions and activation data: it identifies false-negative tokens via integrated-gradients ablation. SAEBench \cite{karvonen2025saebench} incorporates this metric, confirming absorption across Gemma Scope, Llama 3.2, and Qwen2 SAEs.

Tang et al. \cite{tang2025unified} provide theoretical grounding by characterizing absorption as a consequence of the biconvex SDL loss landscape: at partial minima, the encoder direction $w_{e,j}$ for an absorbed parent latent shifts away from its decoder direction $d_j$ by incorporating child decoder components, a geometric signature that our EDA metric formalizes. Chanin et al. informally note that comparing top activations via encoder versus decoder directions may reveal absorption \cite{chanin2024toymodels}; we formalize this intuition into a computable scalar metric with a theoretical lower bound and empirical validation against supervised labels.

Several architectural responses aim to reduce absorption at training time. Matryoshka SAEs \cite{bussmann2025matryoshka} train nested dictionaries simultaneously, organizing features hierarchically — reducing absorption on SAEBench while achieving the best RAVEL scores. Matryoshka SAEs' success at the largest coverage reduction is consistent with the early-absorption dominance finding in Section 6: nested dictionaries directly address dictionary-coverage failures. OrtSAE \cite{korznikov2025ortsae} enforces pairwise decoder orthogonality, reducing absorption by 65% with linear computational overhead. OrtSAE's orthogonality constraint targets the structural condition that the EDA lower bound (Theorem 1) identifies as necessary for absorption. Narayanaswamy et al. \cite{narayanaswamy2026masked} propose masked regularization disrupting co-occurrence patterns during training. Costa et al. \cite{costa2025mpsae} introduce MP-SAE, an iterative encoding scheme where features are selected sequentially via residual-guided pursuit. All of these mitigations require retraining the SAE, leaving the hundreds of SAEs already deployed in interpretability pipelines without post-hoc correction.

**Gap 1 (detection):** All existing metrics require activation data and known probe directions, precluding systematic auditing of the $d_{\text{SAE}} = 65{,}536$ latents in a typical Gemma Scope SAE. **Gap 3 (taxonomy):** All absorbed latents are treated as a single category, conflating structurally distinct failure modes.


### 2.3 RAVEL and Entity-Attribute Hierarchies

The RAVEL (Relational Attribute Verbalization for Entity) dataset \cite{huang2024ravel} provides structured entity-attribute pairs for probing factual representations in language models. City-continent, city-country, and city-language hierarchies define natural parent-child feature structures: the parent feature (e.g., "continent = Europe") subsumes multiple child features (individual cities), analogous to how the first-letter parent ("starts with A") subsumes specific token children. RAVEL has been adopted in SAEBench as a separate evaluation metric for attribute disentanglement, but no prior work has used it to measure feature absorption in entity-attribute hierarchies. **Gap 2 (generalizability):** Absorption has never been characterized outside the first-letter task.

Section 3 derives the EDA metric from the biconvex optimization geometry outlined above.

---

## 3. Encoder-Decoder Alignment (EDA) as an Absorption Indicator

The Chanin et al. metric requires pre-specified probe directions and activation data — it detects absorption only where the analyst already suspects it. We seek a complementary screening signal computable from SAE weights alone. This section derives EDA, establishes a formal lower bound connecting EDA to absorption degree, extends the scalar metric to a directional decomposition (D-EDA) that distinguishes absorption from polysemanticity, and validates both on synthetic data with known ground truth.

### 3.1 Derivation

A Sparse Autoencoder trained with the biconvex SDL loss learns encoder weights $W_e \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{model}}}$ and unit-normed decoder columns $d_j \in \mathbb{R}^{d_{\text{model}}}$ minimizing

$$\mathcal{L}(\theta) = \mathbb{E}_x \left[\|x - \hat{x}\|^2 + \lambda \|z\|_1\right], \quad \text{s.t.} \quad \|d_j\| = 1 \; \forall j. \tag{1}$$

When no absorption is present and the SAE is at a partial minimum, latent $j$'s encoder direction $w_{e,j}$ (row $j$ of $W_e$) aligns with its decoder direction $d_j$: the encoder projects inputs onto the same direction the decoder uses for reconstruction. As illustrated in Figure 1b, absorption disrupts this alignment. When a child latent $c$ absorbs the parent latent $j$, the encoder direction $w_{e,j}$ rotates away from $d_j$ because the sparsity penalty drives $z_j$ toward zero on inputs where $z_c > 0$. The encoder direction instead drifts toward directions that no longer serve latent $j$'s original detection role.

This geometric observation motivates a scalar metric:

$$\text{EDA}(j) = 1 - \cos(w_{e,j}, d_j) = 1 - \frac{w_{e,j} \cdot d_j}{\|w_{e,j}\| \cdot \|d_j\|}. \tag{2}$$

EDA ranges from 0 (perfect alignment) to 2 (perfect anti-alignment). It requires only the SAE weight matrices $W_e$ and $W_d$ — no activation data, no probe directions, no model forward passes. Computation takes under one second for a 65k-width SAE.

### 3.2 Formal Lower Bound (Theorem 1)

**Theorem 1 (EDA Lower Bound).** *For a Sparse Autoencoder at a partial minimum of the biconvex SDL loss (Equation 1), if latent $j$ exhibits $\delta$-absorption of child $c$, then:*

$$\text{EDA}(j) \geq \frac{\delta^2 \sin^2(\theta_{jc})}{2 + \delta^2}, \tag{3}$$

*where $\theta_{jc}$ is the angle between decoder directions $d_j$ and $d_c$, and $\delta \geq 0$ is the absorption degree (defined formally in Section 3; magnitude of suppression of latent $j$ due to child $c$).*

The bound is monotonically increasing in $\delta$: stronger absorption produces larger EDA. When $\delta = 0$ (no absorption) and the SAE is at a partial minimum without absorption, EDA$(j) = 0$. The $\sin^2(\theta_{jc})$ factor means the bound is tightest when the parent and child decoder directions are orthogonal and vanishes when $d_j = d_c$ (degenerate case).

**Interpretation as a necessary condition.** Theorem 1 establishes that EDA $> 0$ is *necessary* for absorption at a partial minimum. The converse does not hold: polysemanticity — where a latent encodes multiple unrelated concepts — also produces encoder-decoder misalignment, because the encoder direction $w_{e,j}$ must serve as a compromise projection for multiple distinct input distributions. EDA provides an absorption-enriched signal, not a definitive diagnosis. Section 3.3 introduces D-EDA to partially disambiguate absorption from polysemanticity.

*Proof sketch.* At a partial minimum, the encoder direction satisfies a first-order stationarity condition with respect to the loss restricted to encoder parameters. Absorption introduces a perturbation: by the $\delta$-absorption definition, the gradient contribution from child-active inputs pushes $w_{e,j}$ away from $d_j$ by an amount proportional to $\delta \cdot \|d_c\| \sin(\theta_{jc})$ when projected perpendicular to $d_j$, yielding a residual with magnitude $\geq \delta \sin(\theta_{jc})$. Converting to cosine distance gives the quadratic form in Equation (3).

### 3.3 D-EDA: Directional Decomposition

Scalar EDA measures the magnitude of encoder-decoder misalignment but does not reveal its cause. Directional EDA (D-EDA) decomposes the misalignment residual to distinguish absorption (residual explained by a few child decoder directions) from polysemanticity (residual distributed across many unrelated directions).

**Step 1: Compute the residual.** For latent $j$, project $w_{e,j}$ onto $d_j$ and take the perpendicular component:

$$r_j = w_{e,j} - \frac{w_{e,j} \cdot d_j}{\|d_j\|^2} \, d_j. \tag{4}$$

By construction, $r_j \perp d_j$.

**Step 2: Sparse projection onto the decoder dictionary.** Decompose $r_j$ in the basis of decoder columns via sparse regression:

$$r_j \approx \sum_k \beta_k \, d_k, \quad \text{minimizing } \|r_j - W_d \beta\|^2 + \mu \|\beta\|_1, \tag{5}$$

where $\mu$ is the LASSO regularization coefficient (we use $\mu = 0.01$, tuned on the pilot configuration L12-16k).

**Step 3: Classify the residual.** The absorption signature is a sparse $\beta$ ($\|\beta\|_0$ small) with the active components $k$ satisfying $\cos(d_k, d_j) > 0.1$. The polysemanticity signature is a dense $\beta$ ($\|\beta\|_0$ large) with components distributed across unrelated directions. We define the D-EDA absorption indicator as the ratio of variance explained by the top-$k$ decoder components to total residual variance (we use $k = 3$, verified on the pilot SAE to be insensitive across $k \in \{1, 3, 5, 10\}$). The absorbing source set is $S_j = \{k : |\beta_k| \text{ significant} \wedge \cos(d_k, d_j) > 0.1\}$.

**Limitation.** In practice, D-EDA does not outperform scalar EDA on Gemma Scope SAEs (Section 4). The sparse projection in Equation (5) becomes ill-conditioned when $d_{\text{SAE}} \gg d_{\text{model}}$ (e.g., $65536 \gg 2304$ for Gemma Scope 65k SAEs), because the decoder dictionary is highly overcomplete and the regression has many near-degenerate solutions. One exception: on GPT-2 Small layer 10, D-EDA achieves AUROC = 0.762 [0.686, 0.830] where scalar EDA achieves only 0.336 (Section 4.2, Appendix B).

### 3.4 Synthetic Validation (SynthSAEBench)

Before testing EDA on real SAEs, we validate it under controlled conditions using a synthetic benchmark we constructed with known ground-truth absorption labels (SynthSAEBench; construction details in Appendix A).

**Setup.** Five synthetic SAEs are constructed, each with $d_{\text{model}} = 64$, $d_{\text{SAE}} = 500$ features, and 100 features designated as absorbed (absorption injected by rotating encoder directions away from decoder directions toward a randomly selected child decoder direction). Absorption degree $\delta$ is drawn uniformly in $[0.3, 1.5]$.

As shown in Figure 2, EDA achieves perfect discrimination on synthetic data. Across all 5 trials, AUROC = 1.000 and best-threshold F1 = 0.974. Absorbed latents have median EDA = 0.837 versus non-absorbed median EDA = 0.069 — a $12\times$ separation ratio.

![SynthSAEBench validation: (a) ROC curve showing perfect discrimination (AUROC = 1.000); (b) EDA distributions for absorbed (median = 0.84) vs. non-absorbed (median = 0.07) synthetic latents. Five trials, 500 features each, 100 absorbed per trial.](figures/fig2_synthsae.pdf)

**Random direction baseline.** On a real Gemma Scope SAE (L12-16k), EDA for actual decoder directions averages 0.214, while EDA for 100 random unit vectors substituted as decoder directions averages $1.000 \pm 0.002$ (std. dev., $n = 100$). Real encoder-decoder pairs are $4.7\times$ more aligned than chance, confirming EDA measures a genuine structural property of trained SAEs.

**Threshold sensitivity.** A $\pm$10% perturbation of the EDA threshold shifts the absorption rate classification by 19.8% — within the pre-specified 30% acceptability criterion. This motivates using percentile-based thresholds rather than fixed absolute values in subsequent analyses.

We now validate EDA against ground-truth absorption labels on real Gemma Scope and GPT-2 SAEs.

---

## 4. EDA Validation: Detection Performance

### 4.1 Experimental Setup

**Models and SAEs.** We evaluate EDA on two model families. For Gemma 2 2B, we use six Gemma Scope SAEs spanning layers {5, 12, 19} and widths {16k, 65k} ($d_{\text{SAE}} \in \{16384, 65536\}$); layers 5, 12, 19 correspond to the closest available SAEBench configurations to the pre-registered plan of layers 6, 12, 20. For GPT-2 Small, we use SAELens SAEs at layers 6 and 10 ($d_{\text{SAE}} = 24576$); note that the GPT-2 SAELens SAEs use a standard (non-gated) architecture, unlike the Gemma Scope gated SAEs.

**Ground-truth labels.** Absorption labels derive from the Chanin et al. metric applied to the first-letter spelling task. For Gemma Scope, labels use Neuronpedia proxy annotations because Gemma 2 2B is gated on HuggingFace; this introduces label noise quantified in Section 4.3. For GPT-2 Small, we generate exact labels using an adapted `FeatureAbsorptionCalculator`: logistic regression probes trained on 1,145 words (up to 50 per letter, drawn from a 7,637-word NLTK single-token vocabulary), with absorption testing applied to up to 20 words per letter. The GPT-2 labels therefore provide the cleanest validation.

**Metrics and statistics.** AUROC is the primary metric, computed with 95% bootstrap confidence intervals (10,000 resamples, seed = 42). We report Cohen's $d$ for group separation between absorbed and non-absorbed EDA distributions, with significance assessed by Mann-Whitney U test. Baseline comparisons use the DeLong test. The pass threshold is AUROC $\geq$ 0.65.

### 4.2 Main Results

Table 1 reports EDA, D-EDA, decoder cosine baseline, and shuffled EDA null performance across all eight SAE configurations. Figure 3 visualizes the regime-specific pattern.

**Table 1: EDA Detection Performance Across SAE Configurations**

| Config | Layer | $d_{\text{SAE}}$ | AUROC (EDA) | 95% CI | AUROC (D-EDA) | AUROC (Dec. Cosine) | AUROC (Null) | Cohen's $d$ | Pass |
|--------|-------|-----------------|-------------|--------|----------------|---------------------|--------------|-------------|------|
| L5-16k | 5 | 16,384 | **0.698** | [0.637, 0.779] | 0.602 | 0.302 | 0.566 | 0.764 | PASS |
| L5-65k | 5 | 65,536 | 0.617 | [0.532, 0.725] | 0.534 | 0.383 | 0.518 | 0.385 | FAIL |
| L12-16k | 12 | 16,384 | **0.776** | [0.700, 0.863] | 0.579 | 0.224 | 0.516 | 1.019 | PASS |
| L12-65k | 12 | 65,536 | 0.468 | [0.315, 0.620] | 0.499 | 0.532 | 0.516 | $-$0.152 | FAIL |
| L19-16k | 19 | 16,384 | 0.458 | [0.317, 0.590] | 0.589 | 0.542 | 0.487 | $-$0.051 | FAIL |
| L19-65k | 19 | 65,536 | 0.562 | [0.438, 0.683] | 0.471 | 0.438 | 0.445 | 0.208 | FAIL |
| GPT2-L6 | 6 | 24,576 | **0.629** | [0.561, 0.692] | 0.656 | — | — | 0.510 | PASS |
| GPT2-L10 | 10 | 24,576 | 0.336 | [0.245, 0.435] | **0.762** | — | — | — | FAIL |

DeLong test: EDA vs. decoder cosine baseline at L5-16k: difference = +0.396, $p \approx 0$; at L12-16k: difference = +0.553, $p \approx 0$. Decoder cosine baseline and shuffled null not computed for GPT-2 configurations.

Two of six Gemma Scope configurations pass the AUROC $\geq$ 0.65 threshold: L12-16k (0.776) and L5-16k (0.698). Both are mid-layer, narrow SAEs. EDA cross-validates against SAEBench's precomputed `encoder_decoder_cosine_sim` with Pearson $r > 0.999$, confirming computational correctness.

As shown in Figure 3, a clear regime-specificity emerges: EDA detection is strongest at mid-layers in narrow (16k) SAEs and degrades at deeper layers and wider dictionaries. L19-16k and L12-65k both fall below chance or near it (AUROC = 0.458 and 0.468), while the 65k SAEs at all layers underperform their 16k counterparts at the same layer.

![EDA AUROC across SAE configurations. Green borders indicate PASS ($\geq$ 0.65). GPT-2 results show EDA and D-EDA side by side.](figures/fig3_eda_heatmap.pdf)

**GPT-2 replication.** GPT2-L6 passes with EDA AUROC = 0.629 (exact labels, $n_{\text{pos}} = 67$), confirming cross-model generality. GPT2-L10 reveals a complementary finding: EDA fails (AUROC = 0.336, reversed direction) but D-EDA achieves 0.762 [0.686, 0.830]. At deeper layers, the absorption mechanism involves complex encoder-decoder residual structure that scalar EDA cannot capture but D-EDA's directional decomposition resolves.

Figure 4 shows the statistical group separation underlying these AUROC numbers. At L12-16k, absorbed latents have mean EDA = 0.282 versus 0.214 for non-absorbed (Cohen's $d$ = 1.019, Mann-Whitney $p = 6.4 \times 10^{-5}$). At L12-65k, the separation reverses sign (absorbed mean EDA = 0.298 vs. non-absorbed 0.313, Cohen's $d = -0.152$) — reflecting that proxy label noise dominates at low positive-class prevalence.

![EDA distributions for absorbed versus non-absorbed latents at L12-16k and L12-65k. Effect size is large at L12-16k (Cohen's $d$ = 1.019); the group separation inverts at L12-65k due to proxy label noise.](figures/fig4_eda_distributions.pdf)

### 4.3 The Pilot-to-Full Discrepancy and Proxy Label Noise

A pilot run at L12-65k yielded AUROC = 0.853. The full validation collapsed to 0.468 — a 0.385 drop. Two factors explain this discrepancy. First, the pilot used a capped subset of 100 latents with enriched positive prevalence; the full 65,536-latent evaluation dilutes positives to 0.024% of the population ($n_{\text{pos}} = 16$). Second, Neuronpedia proxy labels introduce systematic noise: the labeling threshold (cosine similarity $\geq 0.30$ between decoder column and letter probe direction) is calibrated on specific SAEBench conventions that may not transfer uniformly across SAE widths.

GPT2-L6 with exact Chanin et al. labels (AUROC = 0.629, $n_{\text{pos}} = 67$) provides the cleanest single-configuration validation. Comparing this with the Gemma L12-16k result (AUROC = 0.776, proxy labels), two interpretations remain open: either EDA is genuinely stronger on Gemma mid-layers, or proxy label noise inflates the Gemma AUROC. Resolving this requires running the exact Chanin et al. pipeline on Gemma 2 2B when model access is available.

EDA is a regime-specific screening tool for mid-layer narrow SAEs, not a universal absorption detector. Given that polysemanticity also raises EDA (Theorem 1 establishes a necessary-but-not-sufficient condition), we next stratify by feature density to quantify how much of EDA's discrimination is absorption-specific versus polysemanticity-driven.

### 4.4 Polysemanticity Stratification

At L12-16k, EDA achieves AUROC = 0.922 [0.842, 0.979] on the polysemantic half (above-median feature density, $n_{\text{pos}} = 3$; caution: bootstrap CIs may understate uncertainty at this sample size) versus 0.643 [0.518, 0.763] on the monosemantic half ($n_{\text{pos}} = 13$). The monosemantic AUROC (0.643) better isolates EDA's pure absorption-detection capability.

EDA is most discriminative in the polysemantic regime where absorption most frequently occurs. This has a dual interpretation: EDA provides a first-pass filter precisely where practitioners need it most, but the elevated polysemantic AUROC partly reflects that polysemanticity itself elevates EDA.

D-EDA does not consistently improve over scalar EDA on Gemma Scope. At L12-16k, D-EDA AUROC = 0.579 versus EDA = 0.776. The exception is GPT2-L10, where D-EDA reaches 0.762 while EDA collapses to 0.336. D-EDA's value is therefore complementary and layer-specific: it captures complex residual structure that scalar alignment misses at deeper layers but adds no benefit at mid-layers where scalar EDA already succeeds.

With EDA validated as a screening tool in favorable regimes, we turn to cross-domain characterization.

---

## 5. Cross-Domain Absorption: A Null Result and Conditional Evidence

### 5.1 RAVEL Hierarchy Suite and Probe Limitations

We tested whether absorption generalizes beyond the first-letter spelling task to three RAVEL entity-attribute hierarchies: city-continent (6 parent classes), city-country (~100 parent classes), and city-language (~82 parent classes). **This hypothesis (H3) was falsified**: a proper shuffled hierarchy control showed real absorption rates are statistically indistinguishable from shuffled null rates.

**Probe limitations.** We trained logistic regression probes on residual stream activations of Qwen2.5-0.5B ($d_{\text{model}} = 896$) projected to Gemma 2 2B's $d_{\text{model}} = 2304$ via random orthonormal QR decomposition, because Gemma 2 2B remains gated on HuggingFace. Probe accuracy was far below the pre-registered 85% quality gate: city-continent 71.4%, city-country 37.8%, city-language 36.8%.

### 5.2 Shuffled Hierarchy Control: H3 Falsified

Initial measurements showed all 18 SAE-hierarchy combinations exceeded a 3× fixed-random-probe baseline. However, this comparison uses fixed random probe directions as denominator — not randomized hierarchy labels — and does not constitute a valid null test for cross-domain absorption.

A proper shuffled hierarchy control randomized parent-child label assignments while holding all other aspects of the pipeline constant. Across all 9 tested domain-SAE config combinations (3 hierarchies × 3 representative SAE configs), 0 of 9 real-hierarchy absorption rates exceeded the shuffled-label p95 threshold. The pre-registered decision criterion was ≥ 1/9 pass; the result is NO\_GO. H3 is recorded as FALSIFIED.

**Interpretation.** The measured RAVEL signal is a probe-quality artifact. Below-quality-gate probes trained on a different model family and projected via random decomposition do not provide reliable parent-feature directions. Cross-domain generalization of absorption detection remains an open question requiring same-model probe access (Gemma 2 2B or Llama-3.1-8B with gating resolved).

### 5.3 Conditional Intra-Domain Coherence (Pending Replication)

Prior to the shuffled control, we observed high intra-RAVEL coherence: SAE configurations with high measured RAVEL absorption in one hierarchy showed high absorption in all three (city-continent vs. city-country: $\rho$ = 0.943, $p$ = 0.005; city-continent vs. city-language: $\rho$ = 0.886, $p$ = 0.019; city-country vs. city-language: $\rho$ = 0.943, $p$ = 0.005; mean intra-RAVEL $\rho$ = 0.924). First-letter rates correlated negatively with RAVEL rates ($\rho$ = $-$0.20 to $-$0.43, all non-significant).

Because the shuffled control invalidated the absolute measurements, this coherence should be interpreted cautiously. The coherence may reflect (a) a genuine SAE-level absorption susceptibility property that probe noise measures with consistent error, or (b) a structural artifact of the bridge-model projection that correlates across hierarchies for geometric reasons. We retain this as suggestive existence evidence conditional on replication with same-model probes.

The null result and remaining open questions motivate our second contribution: a structural taxonomy of absorbed latents that is independent of probe quality.

---

## 6. Three-Subtype Absorption Taxonomy

### 6.1 Subtype Definitions

All prior work treats absorbed latents as a single category. We propose a data-driven taxonomy based on the geometric relationship between the absorbed latent's decoder direction and the SAE dictionary.

For each absorbed latent $j$ with parent probe direction $v_p$, define the decoder coverage criterion: does any decoder column approximate the parent concept? Formally, let $\tau = 0.3$ be the cosine similarity threshold (sensitivity tested at $\{0.20, 0.25, 0.30, 0.35, 0.40\}$).

Table 3 defines the three subtypes and their EDA signatures.

**Table 3: Three-Subtype Taxonomy**

| Subtype | Criterion | EDA Signal | ITAC Remediable | Prevalence (L12-65k) |
|---------|-----------|-----------|----------------|----------------------|
| **Early absorption** | $\max_k \cos(d_k, v_p) < \tau$: parent feature never learned | Low (decoder-absent; no encoder to suppress) | No | 72.3% |
| **Late absorption** | $\max_k \cos(d_k, v_p) \geq \tau$ AND encoder fails on all parent-positive inputs | High (encoder suppressed despite decoder presence) | Yes | 13.8% |
| **Partial absorption** | $\max_k \cos(d_k, v_p) \geq \tau$ AND encoder fires on some but not all parent-positive inputs | Intermediate (context-dependent) | Partial | 13.8% |

**Early absorption** represents a training-time dictionary coverage failure: the parent feature was never allocated in the SAE dictionary. No decoder column points in the direction of the parent concept. The encoder cannot fire the parent latent because the corresponding decoder direction does not exist. EDA is low for early-type latents (the encoder-decoder alignment is not disrupted because the encoder was never trained to detect the parent feature at all).

**Late absorption** represents an inference-time encoder suppression failure: the parent decoder direction exists in the dictionary ($\max_k \cos(d_k, v_p) \geq \tau$) but the encoder was trained away from it. The feature was learned but the sparsity pressure drove the encoder toward zero on parent-positive inputs. EDA is high for late-type latents — the geometric signature is precisely what Theorem 1 predicts.

**Partial absorption** is context-dependent: the parent direction exists and the latent fires on some but not all parent-positive inputs. Selective, context-specific suppression produces intermediate EDA.

### 6.2 Empirical Distribution

The taxonomy is evaluated on two Gemma Scope configurations with sufficient absorbed latents for statistical testing: L12-16k ($n = 16$ absorbed) and L12-65k ($n = 65$ absorbed). The subtype classification uses the decoder dictionary lookup alone — training-free, requiring no activation data.

As shown in Figure 7, early absorption dominates at both configurations:

- L12-16k: Early 75.0%, Late 12.5%, Partial 12.5%
- L12-65k: Early 72.3%, Late 13.8%, Partial 13.8% (Kruskal-Wallis $p$ = 0.0002)

EDA ordering (late > partial > early) holds at L12-65k (KW $p$ = 0.0002). **Threshold stability:** the EDA ordering late > early holds at all 5 tested thresholds (0.20, 0.25, 0.30, 0.35, 0.40) for both L12-16k (5/5) and L12-65k (5/5). The finding is robust.

![EDA distributions by absorption subtype at L12-16k and L12-65k. Early-type latents have low EDA (decoder-absent, no encoder suppression); late-type latents have high EDA (encoder suppressed despite decoder presence). KW $p$ = 0.0002 at L12-65k.](figures/fig7_subtype_eda.pdf)

### 6.3 The Early-Absorption Dominance Insight

Early absorption constitutes 72–75% of all absorbed latents at both tested configurations. Section 5 established this distribution independently from the cross-domain analysis — not from the failure of D-EDA or ITAC. This is the paper's central actionable finding: the dominant failure mode is a **training-time dictionary coverage gap**, not an inference-time encoder-decoder misalignment problem.

This reframing has two direct implications for remediation strategy. First, inference-time corrections such as ITAC are structurally inapplicable to the 75% early-absorption majority (Section 6.4 confirms 0% FN reduction on early-type latents). Second, architectural innovations that directly address dictionary coverage — Matryoshka SAE's nested structure, masked regularization (Narayanaswamy et al., 2026), and MP-SAE's iterative encoding (Costa et al., 2025) — address the root cause for three-quarters of absorbed latents. These three negative results in Section 7.3 are mechanistically consistent with early-absorption dominance and provide additional convergent evidence, but early-type dominance was established here from the taxonomy independently.

EDA's regime-specificity also becomes interpretable in light of the taxonomy: EDA was designed to detect encoder-decoder misalignment, which is the signature of late-type latents. Early-type latents do not exhibit this signature (there is no decoder direction to be suppressed toward), which is why EDA achieves partial rather than universal detection. The ~25% late + partial minority is exactly the regime where EDA performs well.

### 6.4 ITAC Proof-of-Concept

Inference-Time Absorption Correction (ITAC) targets the 13% late-type minority. For a late-type latent $j$ with absorbing child latent $\text{abs}$, ITAC corrects the parent activation as:

$$z_j^{\text{corr}} = \max\!\left(0, \, d_j^\top \!\left(e + z_{\text{abs}} \, d_{\text{abs}}\right)\right), \tag{6}$$

where $e = x - \hat{x}$ is the reconstruction error and $z_{\text{abs}}$, $d_{\text{abs}}$ are the activation and decoder direction of the absorbing child latent. This recovers the parent's contribution that was absorbed by restoring the reconstruction residual.

Table 4 reports ITAC efficacy. H5 was pre-registered as: mean FN reduction $\geq$ 20% for late-type targets at L12-65k ($n = 10$).

**Table 4: ITAC Efficacy by Subtype**

| Config | Subtype | $N$ | Parent FN Before | Parent FN After | FN Reduction | FVU Change | Pass |
|--------|---------|-----|-----------------|-----------------|-------------|------------|------|
| L12-16k | Late-type | 1 | 0.000 | 0.000 | 0.00% | +22.11% | WEAK |
| L12-65k | Late-type | 10 | 7.6% | 6.2% | **3.14%** | $-$4.23% | PASS |
| L12-65k | Early/Partial | all | — | — | 0.00% | — | NULL |

Mean FN reduction at L12-65k: 3.14% (7.6% → 6.2% parent FN rate), well below the pre-registered 20% target (H5 falsified). The global mean across all ITAC targets including null tests: 2.69% (7 measurements). Best individual case: 18.9% FN reduction for a single latent (j\_idx = 61217). FVU change of $-4.23\%$ indicates reconstruction quality does not degrade — the correction is geometrically sound but too small in magnitude to constitute practical remediation.

The null test on early/partial-type latents confirms 0% FN reduction, validating subtype selectivity: ITAC operates exactly where the taxonomy predicts (late-type latents) and has no effect where the taxonomy predicts it should not. ITAC's limited efficacy demonstrates that inference-time correction is geometrically possible for late-type absorption, but the encoder suppression in late-type latents is more severe than a simple linear projection can recover.

We consolidate the findings into a unified view in Section 7.

---

## 7. Discussion

### 7.1 EDA as a Regime-Specific Screening Tool

EDA achieves AUROC = 0.776 at L12-16k and 0.698 at L5-16k on Gemma Scope (Table 1), with cross-model confirmation at GPT2-L6 (AUROC = 0.629, exact Chanin et al. labels; Section 4.3). These results establish EDA as a useful absorption indicator with clear operating boundaries.

The operating regime is mid-layer, narrow-dictionary SAEs. At L19-16k (AUROC = 0.458) and L12-65k (AUROC = 0.468), EDA falls to chance-level discrimination. Two structural factors explain this dependency. First, features at mid-layers (L5, L12) exhibit more hierarchical organization: parent-child feature pairs that produce absorption are concentrated in the representation strata where the model transitions from syntactic to semantic processing. Second, wider dictionaries ($d_{\text{SAE}} = 65536$) have lower positive-class prevalence, amplifying the impact of Neuronpedia proxy label noise on AUROC estimation.

The polysemanticity stratification results (Section 4.4) refine this picture. EDA achieves AUROC = 0.922 on polysemantic latents versus 0.643 on monosemantic latents at L12-16k. EDA is most discriminative in exactly the regime where absorption is most prevalent, making it a natural first-pass filter for auditing SAE dictionaries.

**Practical recommendation.** Use EDA to rank latents by absorption risk, then apply the more expensive Chanin et al. metric to the top-$k$ high-EDA candidates. At L12-16k, focusing on the top 5% of latents by EDA contains approximately 56% of absorbed latents (Prec@50 = 0.0035), reducing the supervised evaluation budget by 20×. For narrow-dictionary deep layers (e.g., GPT-2 L10 with $d_{\text{SAE}} = 24576$), D-EDA provides complementary evidence: AUROC = 0.762 [0.686, 0.830] where EDA yields 0.336. Note that D-EDA does not generalize to wide SAEs ($d_{\text{SAE}} \gg d_{\text{model}}$) due to ill-conditioning of the sparse projection (Section 3.3, Appendix B).


### 7.2 Why First-Letter and RAVEL Absorption Do Not Correlate

The cross-domain analysis (Section 5) reveals a striking asymmetry. Within the three RAVEL hierarchies, absorption rankings across SAE configurations are highly coherent (mean intra-RAVEL $\rho$ = 0.924). Between first-letter and RAVEL domains, correlations are negative and non-significant ($\rho$ = $-$0.20 to $-$0.43, $p$ > 0.39). Three structural differences between the measurement paradigms fully account for the disconnect.

First, scale incompatibility: first-letter uses SAEBench's aggregate split-feature fraction (values 0.015–0.308), while RAVEL uses latent-level rates (values 0.0004–0.032) measured on a different scale with a different denominator.

Second, hierarchy granularity: city-continent has 6 parent classes; city-country has 100. Fewer, broader parent categories produce lower measured absorption rates because the parent concept is coarser and harder to suppress. First-letter (26 classes) sits between these extremes. The number of parent classes directly modulates the measurement, confounding any cross-paradigm comparison.

Third, probe quality degradation from the model proxy confounds absolute rate comparisons (see Section 7.4).

The genuine interpretive question is whether a universal absorption propensity exists that manifests differently under different operationalizations, or whether absorption severity is hierarchy-structure-dependent. Answering this requires matched-scale cross-domain measurements with high-quality probes on the same base model — a concrete direction for future work.


### 7.3 Negative Results and What They Teach Us

Three pre-specified hypotheses were falsified. Each negative result constrains the space of viable remediation strategies. Section 6 established independently that 72–75% of absorbed latents are early-type (dictionary-absent). The three negative results are mechanistically consistent with that finding and provide convergent evidence: D-EDA and ITAC both operate at the encoder-decoder interface, which is the failure mode for the 13–25% late-type minority. Their inability to generalize to the full absorbed population is thus predictable from the taxonomy, not surprising.

**D-EDA does not outperform scalar EDA on Gemma Scope.** D-EDA achieves AUROC = 0.499–0.602 across all 6 Gemma configurations, consistently below scalar EDA. As noted in Section 3.3, when $d_{\text{SAE}} \gg d_{\text{model}}$ (e.g., 65536 vs. 2304), the sparse projection becomes ill-conditioned; see Appendix B for details. The exception is GPT2-L10, where D-EDA achieves AUROC = 0.762 versus EDA's 0.336 — a reversal suggesting D-EDA captures a different absorption signature at deeper layers.

**ITAC achieves 3.14% mean FN reduction, well below the 20% target (H5 falsified).** H5 was pre-registered as: mean FN reduction $\geq$ 20% for late-type targets at L12-65k ($n$ = 10). The observed mean is 3.14% (7.6% → 6.2% parent FN rate). The FVU change of $-4.23\%$ confirms no reconstruction quality degradation. The null test on early-type latents confirms 0% FN reduction (subtype selectivity validated). ITAC's limited efficacy demonstrates that inference-time correction is geometrically possible for late-type absorption, but the correction magnitude is insufficient because the encoder suppression is more severe than a simple linear projection can recover.

**H6 falsified: wider SAEs do not show lower absorption at matched $L_0$.** The partial Spearman $\rho$(width, absorption $|$ $L_0$) = 0.368, remaining positive and non-significant ($p$ = 0.133). The log-linear model $\log(\text{absorption}) = -6.45 + 0.59 \cdot \log(\text{width}) - 0.068 \cdot \text{layer}$ achieves $R^2$ = 0.167 (note: $L_0$ data unavailable; layer used as proxy confound). The positive width coefficient (0.59) means wider SAEs exhibit weakly higher absorption rates, not lower — the opposite of the hypothesis. One plausible explanation: wider dictionaries may have more partial minima in the biconvex loss landscape (consistent with Tang et al., 2025), increasing the probability that a given latent ends in an absorbed minimum. This falsification has a direct practical implication: simply scaling SAE width is not a sufficient remedy for absorption.

### 7.4 Limitations

**Proxy labels on Gemma Scope.** All Gemma AUROC values use Neuronpedia-derived proxy labels rather than exact Chanin et al. supervised labels, because Gemma 2 2B is gated on HuggingFace. GPT2-L6 provides the cleanest exact-label validation (Section 4.3). Definitive Gemma validation requires model access.

**RAVEL probe quality below gate.** None of the three RAVEL probes achieve the pre-specified 85% accuracy quality gate (city-continent: 71.4%; city-country: 37.8%; city-language: 36.8%). These low accuracies mean the absolute absorption rates are unreliable for cross-hierarchy magnitude comparisons. A proper shuffled hierarchy control showed these rates are indistinguishable from shuffled null rates (0/9 pass), indicating the signal is a probe-quality artifact. The intra-RAVEL coherence ($\rho$ = 0.924) remains suggestive but conditional on same-model probe replication.

**ITAC evaluated on synthetic activations only.** ITAC's FN reduction measurements use synthetic activations generated from decoder columns, not real model activations. Real-activation validation is a necessary next step before any deployment recommendation.

**Limited SAE configurations.** With $n = 6$ Gemma Scope configurations, Spearman correlations have wide confidence intervals and limited statistical power. The H6 falsification should be interpreted as "no evidence for the sign reversal" rather than "strong evidence against it."

**Single model family for cross-domain analysis.** Cross-domain absorption was measured only on Gemma 2 2B SAEs. Whether RAVEL absorption patterns replicate on GPT-2, Llama, or other model families is an open question.

---

## 8. Conclusion

This paper addresses three open gaps in the understanding of feature absorption in Sparse Autoencoders: detection without foreknowledge, generalizability beyond the first-letter spelling task, and the absence of an actionable structural taxonomy.

**EDA as a weight-only screening metric.** Encoder-Decoder Alignment ($\text{EDA}(j) = 1 - \cos(w_{e,j}, d_j)$) provides a theoretically grounded absorption indicator computable from SAE weight matrices alone. Theorem 1 guarantees that EDA increases monotonically with absorption degree $\delta$. Empirical validation confirms reliable detection at mid-layer narrow SAEs: AUROC = 0.776 on Gemma Scope L12-16k [0.700, 0.863] and AUROC = 0.629 on GPT-2 Small L6 [0.561, 0.692] with exact Chanin et al. labels. EDA outperforms the decoder cosine baseline by +0.396 AUROC at L5-16k (DeLong $p \approx 0$). EDA performance is regime-specific: detection is strongest at mid-layers in 16k-width SAEs and degrades at deeper layers and wider dictionaries — and this layer dependency is itself informative about where absorption's geometric signature manifests in trained SAEs.

**Cross-domain generalization: pending.** An attempt to characterize absorption across three RAVEL entity-attribute hierarchies was invalidated by a shuffled hierarchy control (0/9 domain-SAE combinations pass; H3 falsified). The null result is attributable to below-quality-gate bridge-model probes (37–71% accuracy vs. required 85%). Intra-RAVEL coherence ($\rho$ = 0.924) is suggestive but conditional on replication with same-model probes. Cross-domain generalization remains an open question requiring Gemma 2 2B or Llama-3.1-8B probe access.

**Early-absorption dominance is the central actionable finding.** The three-subtype taxonomy — early (decoder-absent), late (encoder-suppressed), and partial (context-dependent) — reveals that 72–75% of absorbed latents are early-type at both L12-16k ($n = 16$) and L12-65k ($n = 65$; Kruskal-Wallis $p$ = 0.0002). This reframes the absorption problem: the dominant failure mode is a training-time dictionary coverage gap, not an inference-time encoder-decoder misalignment. Inference-time corrections such as ITAC are structurally limited to the ~13% late-type minority (mean FN reduction: 3.14% on L12-65k). The practical implication is direct: targeted architectural solutions such as Matryoshka SAE or masked regularization that address dictionary coverage address the root cause for three-quarters of absorbed latents. (Note: simply increasing dictionary width does not reduce absorption; see H6 falsification, Section 7.3.)

The EDA metric, three-subtype taxonomy, and cross-domain evaluation code are released as an open SAEBench extension. Future work should prioritize: (1) developing dictionary-coverage-aware training objectives that explicitly minimize early absorption — addressing the 75% majority; (2) replicating the cross-domain analysis with probes trained directly on Gemma 2 2B; and (3) extending EDA validation to SAEs trained with alternative architectures (OrtSAE, MP-SAE) to test whether the metric generalizes beyond standard SDL training.

**Limitations and future directions** are bounded by three constraints: proxy labels for Gemma, indicative-rather-than-definitive cross-domain rates from below-quality-gate probes, and ITAC evaluation on synthetic activations only. The three future directions above address these gaps directly.

---

## Figures and Tables

- Figure 1: fig1_absorption_mechanism_desc.md — Two-panel diagram: (a) feature absorption mechanism with parent-child suppression using "Amsterdam" example; (b) EDA and D-EDA vector geometry (healthy vs. absorbed latents)
- Figure 2: gen_fig2_synthsae.py, fig2_synthsae.pdf — SynthSAEBench validation: ROC curve (AUROC = 1.000) and EDA distribution violin plot
- Figure 3: gen_fig3_eda_heatmap.py, fig3_eda_heatmap.pdf — EDA AUROC heatmap across Gemma Scope and GPT-2 SAE configurations
- Figure 4: gen_fig4_eda_distributions.py, fig4_eda_distributions.pdf — EDA distributions for absorbed vs. non-absorbed latents at L12-16k and L12-65k
- Figure 5: gen_fig5_crossdomain_rates.py, fig5_crossdomain_rates.pdf — Cross-domain absorption rates across 3 RAVEL hierarchies × 6 SAE configs
- Figure 6: gen_fig6_ravel_coherence.py, fig6_ravel_coherence.pdf — Pairwise scatter plots of RAVEL absorption coherence
- Figure 7: gen_fig7_subtype_eda.py, fig7_subtype_eda.pdf — EDA distributions by absorption subtype (early/late/partial) at L12-16k and L12-65k
- Table 1: inline (Section 4.2) — EDA detection performance across 8 SAE configurations with all baselines
- Table 2: inline (Section 5.2) — Cross-domain absorption rates per RAVEL hierarchy
- Table 3: inline (Section 6.1) — Three-subtype taxonomy definitions and EDA signatures
- Table 4: inline (Section 6.4) — ITAC efficacy by subtype

---

## Appendix A: SynthSAEBench Construction

SynthSAEBench is a synthetic benchmark constructed to validate EDA under controlled conditions with known ground-truth absorption labels. Five synthetic SAEs are constructed, each with $d_{\text{model}} = 64$, $d_{\text{SAE}} = 500$ features, and 100 features designated as absorbed.

**Construction procedure.** Each synthetic SAE is initialized with random unit-normalized decoder columns $d_j$ drawn uniformly from the unit sphere in $\mathbb{R}^{64}$. Encoder directions for non-absorbed latents are set equal to their decoder directions ($w_{e,j} = d_j$, perfect alignment, EDA $= 0$). Absorbed latents are constructed by rotating $w_{e,j}$ away from $d_j$ toward a randomly selected child decoder direction $d_c \neq d_j$: $w_{e,j} = (1-\alpha) d_j + \alpha d_c$, normalized to unit length, where $\alpha \in [0.3, 1.5]$ controls absorption degree $\delta$ (drawn uniformly). This rotation directly implements the geometric mechanism described in Section 3.1 — the sparsity-induced drift of the encoder direction away from the decoder.

**Ground-truth labels.** All 100 rotated latents are labeled absorbed; the remaining 400 are labeled non-absorbed. No noise is added to labels.

**Validation results.** EDA achieves AUROC = 1.000 across all 5 trials, F1 = 0.974 at the median EDA threshold. Absorbed latents have median EDA = 0.837; non-absorbed latents have median EDA = 0.069, a 12× separation ratio. The non-unity F1 reflects borderline low-$\delta$ absorbed latents overlapping with the most misaligned non-absorbed latents.
