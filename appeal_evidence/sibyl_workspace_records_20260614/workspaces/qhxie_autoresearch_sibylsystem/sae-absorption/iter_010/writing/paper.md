# Cross-Domain Feature Absorption in Sparse Autoencoders: Measurement, Mechanism, and Probe Quality Confounds

## Abstract

Sparse autoencoders (SAEs) decompose neural network activations into interpretable features, but feature absorption---where parent features fail to activate when child features fire---degrades their reliability. Every published absorption measurement uses a single task: first-letter spelling classification with perfect probes ($F_1 = 1.0$). We present the first cross-domain absorption characterization, extending measurement from spelling to entity-attribute knowledge hierarchies (city-country, city-continent, city-language) using RAVEL on Gemma 2 2B with Gemma Scope JumpReLU SAEs. Absorption rates at layer 24 span a 4.1$\times$ descriptive range: city-language 11.6%, first-letter 27.1%, city-continent 31.4%, and city-country 45.1% (Kruskal-Wallis $p = 7.4 \times 10^{-66}$ within RAVEL). A probe degradation ablation---degrading first-letter probe quality via weight noise injection to seven $F_1$ levels---reveals that probe $F_1$ is a major confound ($R^2 = 0.777$, $p = 0.009$, Spearman $\rho = -1.0$). City-continent absorption matches the degradation curve within 0.6 percentage points (fully explained by probe quality), while city-language sits 21.3 pp below the curve prediction, identifying it as a genuine hierarchy-specific outlier. Activation patching provides the first interventional evidence for competitive exclusion across all tested hierarchy types: zeroing the child feature recovers 32.5% (first-letter, $d = 1.33$), 61.9% (city-continent, $d = 1.50$), and 34.2% (city-language, $d = 0.75$) of absorbed instances, versus 1.5--6.8% for controls. Four correlational approaches---Geometric Absorption Score, conditional mutual information, the Absorption Tax, and a rate-distortion predictor---all fail, establishing that absorption is a causal phenomenon requiring interventional methods.

---

# 1 Introduction

Sparse autoencoders (SAEs) decompose dense neural network activations into sparse, interpretable features (Cunningham et al., 2024; Bricken et al., 2023), enabling mechanistic analysis of language model internals. Anthropic's circuit tracing in Claude 3.5 Haiku demonstrates that when SAE features reliably represent their target concepts, they support powerful interpretability applications including planning-ahead detection and jailbreak mechanism identification (Lindsey et al., 2025). Google DeepMind's Gemma Scope provides over 400 open pre-trained SAEs across Gemma 2 models (Lieberum et al., 2024), and SAEBench standardizes evaluation across eight metrics and seven architectures (Karvonen et al., 2025). The promise is clear: SAEs offer an unsupervised decomposition of model computation into human-readable parts.

Feature absorption undermines this promise. When a child feature (e.g., "the city Paris") fires in the SAE encoder, the parent feature (e.g., "in Europe") systematically fails to activate---even though the parent concept is present and a linear probe correctly classifies the raw (un-encoded) activations (Chanin et al., 2024). The parent feature appears monosemantic from its activation examples, but it has systematic holes in its recall that are invisible without ground-truth supervision. Chanin et al. prove in a two-layer toy model that feature absorption arises from the sparsity objective: the SAE achieves lower $L_0$ by encoding parent information into child decoder vectors rather than activating a separate parent feature. Google DeepMind deprioritized SAE research partly because absorption degraded safety-relevant feature detection by 10--40% when using SAE-reconstructed activations (Smith et al., 2025).

Every published measurement of feature absorption uses a single proxy task: classifying the first letter of a word (Chanin et al., 2024; Karvonen et al., 2025). This task has unnaturally clean properties---26 equiprobable parent classes, near-perfect probe accuracy ($F_1 = 1.0$), 100% parent-child co-occurrence by construction---that may not reflect the messy hierarchies of real knowledge. Safety-relevant features live in knowledge and reasoning space, not spelling space. The 15--35% absorption rates reported on first-letter spelling could overestimate, underestimate, or be entirely atypical of absorption in semantic domains. The field's understanding of how severely absorption degrades SAE reliability rests on this single, potentially unrepresentative benchmark.

We present the first systematic cross-domain absorption characterization, extending measurement from spelling to entity-attribute knowledge hierarchies. Using RAVEL (Huang et al., 2024)---a dataset of entity-attribute pairs with validated probes for city-country, city-continent, and city-language relationships---we measure absorption on Gemma 2 2B with Gemma Scope JumpReLU SAEs across four hierarchy types, four transformer layers, and two SAE widths. Three findings emerge.

**Cross-domain variation is large but partially confounded by probe quality.** Absorption rates at layer 24 with 16k SAEs span a 4.1$\times$ descriptive range: city-language 11.6%, first-letter 27.1%, city-continent 31.4%, and city-country 45.1% (Figure 1). Within-RAVEL variation is statistically significant (Kruskal-Wallis $p = 7.4 \times 10^{-66}$). A probe degradation ablation---where we intentionally degrade first-letter probe quality via weight noise injection to seven $F_1$ levels from 0.70 to 1.0---reveals that probe $F_1$ explains much of this variation ($R^2 = 0.777$, $p = 0.0087$, Spearman $\rho = -1.0$). City-continent absorption (31.4%) falls within 0.6 percentage points of the degradation curve prediction at its probe $F_1 = 0.87$, indicating its elevated rate is fully explained by probe quality. City-language, however, is a genuine outlier: at 11.6%, it sits 21.3 percentage points below the curve prediction of 32.9% at $F_1 = 0.82$. Probe quality is a major confound in cross-domain absorption measurement, but genuine hierarchy-specific effects exist.

**Activation patching confirms a universal competitive exclusion mechanism.** Zeroing the identified child feature in the SAE latent space and measuring parent probe recovery provides causal---not merely correlational---evidence for competitive exclusion. Recovery rates are 32.5% for first-letter (Cohen's $d = 1.33$, $p = 0.000218$), 61.9% for city-continent ($d = 1.50$, $p < 10^{-20}$), and 34.2% for city-language ($d = 0.75$, $p < 10^{-18}$), versus 1.5--6.8% for control features. All three hierarchy types show large positive effects, confirming that competitive exclusion operates universally across feature hierarchy types, not only in the first-letter domain.

**Four correlational approaches to predicting absorption fail; causal methods are required.** The Geometric Absorption Score (GAS, $\rho = 0.116$, AUROC $= 0.571$), conditional mutual information (CMI, $\rho = 0.044$, $p = 0.84$), the Absorption Tax framework ($\rho = -0.20$, concordance at chance), and a rate-distortion three-factor model ($R^2 = 0.104$, all individual predictors in the wrong direction) all fail to predict which features will be absorbed. This quadruple negative establishes a methodological boundary: absorption is a causal phenomenon that resists statistical and geometric proxies.

![Cross-domain absorption rates at L24 with 16k JumpReLU SAE on Gemma 2 2B. Absorption spans a 4.1$\times$ descriptive range across four hierarchy types. Probe $F_1$ values shown below each bar. Within-RAVEL variation is significant ($p = 7.4 \times 10^{-66}$, Kruskal-Wallis). A probe degradation ablation ($R^2 = 0.777$) shows probe quality explains most variation, but city-language is a genuine hierarchy-specific outlier at 21.3 pp below the degradation curve.](figures/fig1_teaser.pdf)

Our contributions are:

1. **First cross-domain absorption characterization with quantitative probe quality decomposition.** We measure absorption across four hierarchy types and decompose the observed 4.1$\times$ variation using a probe degradation ablation ($R^2 = 0.777$), identifying city-language as a genuine hierarchy-specific anomaly (Section 4).

2. **Universal causal mechanism via activation patching.** We provide the first interventional evidence that competitive exclusion drives absorption across all tested hierarchy types ($d = 0.75$--$1.50$, all $p < 10^{-3}$), extending prior correlational evidence beyond the first-letter domain (Section 5).

3. **Decoder information entanglement analysis.** Child feature decoder vectors carry 3.98--6.16 nats of parent-direction information (versus 0.01--0.12 nats for control directions), consistently across hierarchy types. We acknowledge the circularity limitation of this diagnostic and describe what a genuine computational-redundancy test would require (Section 5).

The remainder of this paper is organized as follows. Section 2 reviews SAE architectures, the absorption phenomenon, and the RAVEL evaluation framework. Section 3 describes our measurement pipeline, including the probe degradation ablation protocol. Section 4 presents cross-domain absorption results and the confound decomposition. Section 5 analyzes the causal mechanism via activation patching and decoder entanglement. Section 6 reports a (underpowered) architecture comparison. Section 7 discusses implications, limitations, and the quadruple negative result for correlational predictors. Section 8 concludes.

---

# 2 Background and Related Work

## 2.1 Sparse Autoencoders for Mechanistic Interpretability

Neural networks encode more features than they have neurons by representing features as overlapping linear directions in activation space---a phenomenon called superposition (Elhage et al., 2022). Sparse autoencoders (SAEs) address this by projecting the residual stream activation $\mathbf{x} \in \mathbb{R}^d$ into an overcomplete basis of $m \gg d$ latent features $\mathbf{z} \in \mathbb{R}^m$ via a sparsity-inducing encoder, then reconstructing $\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$. When each latent $z_i$ fires for a single coherent concept---monosemanticity---the sparse decomposition provides a human-readable vocabulary for model internals.

Early work demonstrated that SAEs recover interpretable features in small models: Cunningham et al. (2024) showed highly interpretable features in Pythia-70M/410M, and Bricken et al. (2023) decomposed a one-layer transformer's MLP activations into monosemantic directions. Rapid scaling followed. Anthropic applied SAEs to Claude 3 Sonnet and identified safety-relevant features including deception and sycophancy (Templeton et al., 2024). OpenAI trained a 16M-latent TopK SAE on GPT-4 with clean scaling laws (Gao et al., 2024). Google DeepMind released Gemma Scope, providing over 400 open JumpReLU SAEs across all layers of Gemma 2 models at widths from 16k to 1M (Lieberum et al., 2024). Anthropic's circuit tracing in Claude 3.5 Haiku demonstrated that reliable SAE features enable tracing of planning-ahead, hallucination, and jailbreak mechanisms through feature-level attribution graphs (Lindsey et al., 2025).

These successes established SAEs as the dominant tool for mechanistic interpretability. They also exposed systematic failure modes---feature absorption, hedging, inconsistency, and non-canonicality---that challenge whether SAEs reliably recover the features they promise.

## 2.2 Feature Absorption

Chanin et al. (2024) formalize feature absorption: a parent feature (e.g., "starts with S") fails to activate when a child feature (e.g., "the word Saturday") co-occurs, because the SAE achieves better sparsity by encoding the parent's information into the child's decoder direction $\mathbf{d}_c$. The mechanism is driven by decoder similarity: when $\cos(\mathbf{d}_c, \mathbf{w}_p)$ is high, the child's reconstruction already covers the parent direction, so encoding an additional parent latent wastes one unit of $L_0$ without improving reconstruction. Chanin et al. prove in a two-layer toy model that hierarchical feature structure is sufficient to produce absorption, and measure absorption rates of 15--35% across hundreds of Gemma Scope, Llama 3.2, and Qwen2 SAEs.

Absorption is not a training error but a rational optimization outcome. For a parent-child pair $(p, c)$, an SAE with absorption encodes both concepts at $L_0$ cost $+1$ (child only), while absorption-free encoding requires $+2$ (child and parent). Any solution that fully eliminates absorption will occupy a worse position on the variance-explained vs. $L_0$ frontier (Chanin et al., 2024). Theoretical analysis reinforces this: Cui et al. (2025) show that SAEs generally fail to recover ground-truth features unless features are extremely sparse, and a unified theory of sparse dictionary learning casts absorption as a natural consequence of the piecewise biconvex optimization landscape shared by all SAE variants (Wright et al., 2025).

The canonical measurement procedure identifies false negatives (FN) via linear probes trained on raw activations, runs integrated-gradients attribution on FN tokens, and detects absorption when the highest-attribution latent has cosine similarity exceeding a threshold $\tau_{\cos}$ with the probe direction. SAEBench (Karvonen et al., 2025) adopted this procedure as one of its eight standardized evaluation metrics and found absorption present in every architecture tested---TopK, JumpReLU, BatchTopK, and Matryoshka. Tian et al. (2025) frame absorption as a special case of poor feature sensitivity: features that appear monosemantic from their activation examples may nonetheless fail to activate on semantically similar inputs.

A critical empirical limitation unifies all these measurements: every absorption rate in the literature is computed on a single task---first-letter spelling classification---using 26 letter classes with near-perfect probe accuracy ($F_1 = 1.0$) and 100% parent-child co-occurrence by construction. Whether these rates generalize to knowledge hierarchies, where class counts range from 6 to 80, class balance varies from moderate to extreme, and probe quality is imperfect ($F_1 = 0.73$--$0.87$), is unknown.

## 2.3 Adjacent Failure Modes

Absorption is one of several documented SAE failure modes, each with distinct causes and manifestations.

**Feature hedging** (Chanin et al., 2025) occurs when insufficient dictionary width forces correlated features to merge into a single latent. Hedging and absorption both manifest as features failing to fire where expected, but their root causes differ: width or sparsity limitations for hedging versus hierarchical feature structure for absorption. Matryoshka SAEs trade one for the other---their nested prefix loss reduces absorption to ${\sim}0.03$ (vs. BatchTopK's ${\sim}0.29$ on SAEBench) but introduces hedging at inner levels (Chanin et al., 2025). Incorrect $L_0$, which is common in practice, triggers hedging and feature mixing even in well-architectured SAEs: Chanin and Garriga-Alonso (2025) show that most open-source SAEs have $L_0$ that is too low, causing systematically wrong features.

**Feature inconsistency** (Song et al., 2025) describes the phenomenon where independent SAE training runs converge to different feature sets. TopK SAEs achieve only 0.80 consistency across runs as measured by pairwise mean cosine correlation.

**Feature non-canonicality** (Leask et al., 2025) challenges the assumption that SAE latents are atomic units of analysis. Meta-SAEs decompose individual latents into sub-features, and larger dictionaries discover qualitatively new latents missed by smaller ones---suggesting SAE features are neither complete nor atomic.

**Multi-dimensional features** (Engels et al., 2024) present a distinct challenge: irreducible circular representations for concepts like days of the week or months cannot be captured by one-dimensional SAE directions, potentially producing absorption-like pathologies when the model forces multi-dimensional structure into scalar latents.

## 2.4 SAE Architectures and Absorption Mitigation

Several architectures reduce absorption, though none eliminate it. JumpReLU SAEs (Rajamanoharan et al., 2024a) use a learnable threshold activation and form the backbone of Gemma Scope; SAEBench reports they exhibit the worst absorption among tested architectures at low $L_0$, possibly because longer training deepens the absorption optimum. BatchTopK SAEs (Rajamanoharan et al., 2024b) relax the TopK constraint to the batch level, enabling variable sparsity per sample; they show absorption comparable to standard TopK because the reconstruction incentive for absorption persists regardless of the sparsity mechanism. Matryoshka SAEs (Bussmann et al., 2025) train nested dictionaries of increasing size simultaneously, creating a natural feature hierarchy where smaller dictionaries learn general concepts; they achieve the lowest SAEBench absorption (${\sim}0.03$). OrtSAE (Korznikov et al., 2025) enforces orthogonality via pairwise cosine similarity penalties on decoder directions, reducing absorption by ${\sim}70\%$ vs. BatchTopK and achieving mean cosine similarity 2.7$\times$ lower. Adaptive Temporal Masking (Li et al., 2025) dynamically adjusts feature selection via per-latent importance scoring and reports the lowest published SAEBench absorption metric values (0.0068 vs. TopK 0.1402 on Gemma 2 2B). Masked regularization (Narayanaswamy et al., 2026) disrupts co-occurrence patterns during training via token masking, reducing absorption across architectures.

A limitation unites all these evaluations: every architecture comparison uses the first-letter spelling task as its absorption benchmark. Whether architecture rankings transfer to semantically richer hierarchies---where class counts, balance, and probe quality differ---is untested. This paper provides the first such cross-domain comparison (Section 6), though with limited statistical power (12 observations across four architectures).

## 2.5 Entity-Attribute Evaluation with RAVEL

RAVEL (Resolved Attribute Value Estimation for Language models; Huang et al., 2024) provides a framework for evaluating how language models encode entity-attribute relationships. The dataset contains thousands of entities (cities, people, objects) with validated attribute labels (country, continent, language, birthplace) and prompt templates designed to elicit attribute information. Linear probes trained on model activations with RAVEL labels achieve high accuracy for most attributes, establishing that models encode these relationships as recoverable linear directions.

RAVEL's city subset provides natural feature hierarchies for absorption measurement. The city "Paris" is a child of the country "France," the continent "Europe," and the language "French." These three hierarchies vary along dimensions relevant to absorption:

- **Class count.** City-continent has 6 classes (comparable to a coarse grouping), city-language has ${\sim}20$, and city-country has ${\sim}80$ (comparable to the 26-class first-letter task in scale, though far more imbalanced).
- **Class balance.** City-continent is moderately balanced; city-country is highly skewed (USA: 176 entities vs. Chad: 5).
- **Probe quality.** RAVEL probes achieve $F_1 = 0.73$--$0.87$ at layer 24 on Gemma 2 2B, compared to $F_1 = 1.0$ for first-letter probes. This gap is itself a potential confound that we explicitly decompose via the probe degradation ablation (Section 4.6).
- **Co-occurrence structure.** Unlike first-letter spelling, where every word co-occurs with exactly one letter, city-attribute relationships embed in a multi-level graph (city $\to$ country $\to$ continent) with non-trivial co-occurrence patterns.

These properties make RAVEL a natural testbed for extending absorption measurement beyond the controlled but potentially unrepresentative first-letter task.

## 2.6 Benchmarks and the Evaluation Gap

SAEBench (Karvonen et al., 2025) standardized SAE evaluation with eight metrics across 200+ SAEs and seven architectures, revealing that proxy metrics (cross-entropy loss recovered, sparsity) do not reliably predict practical performance on downstream tasks. SynthSAEBench (2026) provides a controlled synthetic environment with known ground-truth feature hierarchies, where logistic probes achieve $F_1 = 0.974$ while SAEs substantially underperform. CE-Bench (Gulko et al., 2025) offers a lightweight contrastive benchmark correlating at $> 70\%$ Spearman $\rho$ with SAEBench.

Across all three benchmarks, absorption is measured exclusively on first-letter spelling---a proxy task with 26 classes, near-uniform distribution, and perfect probe accuracy. No benchmark includes cross-domain absorption measurement. No cross-layer analysis examines how absorption varies from early to late transformer layers; published rates aggregate over single layers or report layer-averaged scores. Google DeepMind's decision to deprioritize SAE research was driven partly by 10--40% performance degradation on safety-relevant downstream tasks using SAE-reconstructed activations (Smith et al., 2025), but the specific contribution of feature absorption to this degradation has not been isolated.

Three specific gaps motivate this work:

1. **Single-task evaluation monoculture.** Every absorption rate in the literature comes from first-letter spelling. Absorption rates on knowledge hierarchies---where safety-relevant features reside---are unknown.
2. **No probe quality confound control.** First-letter probes achieve $F_1 = 1.0$; RAVEL probes achieve $F_1 = 0.73$--$0.87$. Cross-domain comparison without controlling for this difference risks attributing probe artifacts to hierarchy effects.
3. **No causal mechanism validation beyond spelling.** Chanin et al. (2024) provide correlational evidence (decoder cosine similarity, integrated-gradients attribution) for competitive exclusion on first-letter data. Whether the same causal mechanism operates in knowledge hierarchies has not been tested with interventional methods.

This paper addresses all three gaps. We extend absorption measurement to four feature hierarchies spanning syntactic and semantic domains, introduce a probe degradation ablation to decompose probe quality confounds from genuine hierarchy effects, and validate the competitive exclusion mechanism via activation patching across all tested hierarchy types.

---

# 3 Methodology

This section describes the four feature hierarchies under study, the probe training and quality gating procedure, the absorption measurement pipeline, the activation patching protocol for causal validation, the decoder information entanglement diagnostic, the hedging decomposition, and the probe degradation ablation that controls for the probe quality confound.

## 3.1 Feature Hierarchies

We measure absorption on four feature hierarchies that span syntactic and semantic domains (Table 1).

**First-letter spelling.** Following Chanin et al. (2024), the parent concept is the first letter of a word (e.g., "starts with S") and the child is the word itself (e.g., "Saturday"). The hierarchy has $K = 25$ parent classes (letter "x" excluded for insufficient test examples), approximately 2,345 test words drawn from the `sae-spelling` library, and 100% parent-child co-occurrence by construction. Each word is evaluated in 5 prompt contexts using the template "The word {WORD} begins with the letter," with the target word at token position $-6$. First-letter probes achieve $F_1 = 1.0$ at all tested layers, making this hierarchy a positive control with maximal probe quality.

**City-continent.** Drawn from RAVEL (Huang et al., 2024), the parent concept is the continent in which a city is located (e.g., "in Europe") and the child is the city entity (e.g., "Paris"). $K = 6$ continent classes, 1,567 entities, moderate class balance. Probe $F_1 = 0.87$ at layer 24. Entities are evaluated at token position $-2$ using RAVEL's attribute-specific prompt templates.

**City-language.** Also from RAVEL, the parent concept is the primary language spoken in a city (e.g., "speaks French"). $K = 23$ language classes (including compound labels such as "Arabic,Kurdish,Turkish"), 1,229 entities, intermediate class balance. Probe $F_1 = 0.82$ at layer 24. Same token position ($-2$) and prompt templates as city-continent.

**City-country.** From RAVEL, the parent concept is the country in which a city is located. $K = 77$ country classes (3 classes with $< 5$ entities excluded from the original 80), 1,405 entities, highly imbalanced (USA: 176 entities, Chad: 5). Probe $F_1 = 0.73$ at layer 24---below the strict quality gate. This hierarchy is included with a documented caveat: its absorption rate may be inflated by probe imperfection (quantified in Section 4.6 via the probe degradation ablation).

**Token position asymmetry.** First-letter experiments use token position $-6$ (the `sae-spelling` convention), while RAVEL hierarchies use position $-2$ (the RAVEL convention). Both positions were chosen to maximize probe quality for their respective tasks. Within-RAVEL comparisons (the primary cross-domain analysis in Section 4) are unaffected because all three RAVEL hierarchies share position $-2$. First-letter versus RAVEL comparisons should be interpreted with this confound in mind.

## 3.2 Probe Training and Quality Gates

We train logistic regression probes on Gemma 2 2B residual stream activations at layers 6, 12, 18, and 24. First-letter probes use L2-regularized one-vs-rest logistic regression ($C = 0.01$, selected via cross-validation) on 4,132 training tokens, evaluated on 1,033 test tokens. RAVEL probes use cross-validated logistic regression on all RAVEL city entities at each layer and hierarchy, evaluated by weighted-macro $F_1$.

Table 1 reports probe quality across all 16 hierarchy-layer combinations with quality gate status.

| Hierarchy | Layer 6 | Layer 12 | Layer 18 | Layer 24 | Gate (L24) |
|-----------|---------|----------|----------|----------|------------|
| First-letter | $F_1 = 1.00$ | $F_1 = 1.00$ | $F_1 = 1.00$ | $F_1 = 1.00$ | **Strict pass** |
| City-continent | $F_1 = 0.61$ | $F_1 = 0.74$ | $F_1 = 0.81$ | $F_1 = 0.87$ | Relaxed pass |
| City-language | $F_1 = 0.53$ | $F_1 = 0.68$ | $F_1 = 0.76$ | $F_1 = 0.82$ | Relaxed pass |
| City-country | $F_1 = 0.41$ | $F_1 = 0.56$ | $F_1 = 0.67$ | $F_1 = 0.73$ | **Below gate** |

**Table 1.** Probe $F_1$ scores across four hierarchies and four layers on Gemma 2 2B. Quality gate thresholds: strict $> 0.90$; relaxed $\geq 0.80$. First-letter achieves perfect $F_1$ at all layers. RAVEL probes improve monotonically with depth, reaching their best quality at layer 24. City-country falls below the relaxed gate and is included with a documented caveat.

**Quality gates.** We define two thresholds: a strict gate ($F_1 > 0.90$) and a relaxed gate ($F_1 \geq 0.80$). Only first-letter passes the strict gate at layer 24. City-continent and city-language pass the relaxed gate. City-country fails both gates. Results from hierarchies passing only the relaxed gate are reported with the caveat that probe imperfection may inflate measured absorption (Section 4.6 quantifies this effect). Results from city-country are included as exploratory data with explicit uncertainty.

**Aggregation method.** Absorption rates are computed per-token: $\alpha = n_{\text{FN}} / n_{\text{probe-correct-raw}}$, where each token occurrence is one observation. For first-letter, each of the 2,345 test words appears in 5 prompt contexts, yielding 11,725 token observations. For RAVEL hierarchies, each entity appears once. We also computed per-unique-word and per-letter aggregations as robustness checks; the qualitative finding of significant cross-domain variation holds under all three aggregation methods.

## 3.3 Absorption Measurement Pipeline

The absorption measurement pipeline adapts Chanin et al. (2024) to arbitrary feature hierarchies:

1. **Encode.** Pass input tokens through Gemma 2 2B and extract residual stream activations $\mathbf{x} \in \mathbb{R}^d$ at layer $\ell$.

2. **Probe on raw activations.** Apply the linear probe to $\mathbf{x}$. Record the predicted parent class $\hat{y}_{\text{raw}}$. If $\hat{y}_{\text{raw}} \neq y$ (probe incorrect on raw), skip this token.

3. **SAE encode.** Pass $\mathbf{x}$ through the SAE encoder: $\mathbf{z} = \sigma(\mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}})$, where $\sigma$ is JumpReLU.

4. **SAE reconstruct.** Compute $\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$.

5. **Probe on SAE activations.** Apply the same probe to $\hat{\mathbf{x}}$. Record $\hat{y}_{\text{SAE}}$.

6. **Classify false negatives.** If $\hat{y}_{\text{raw}} = y$ but $\hat{y}_{\text{SAE}} \neq y$, this token is a false negative (FN)---the SAE reconstruction lost parent information that the raw activations carried.

7. **Identify parent-child pairs.** For each FN, use integrated-gradients attribution to identify which active SAE features contributed most to the probe prediction change. Features with high attribution and cosine similarity $\cos(\mathbf{d}_i, \mathbf{w}_p) > 0.30$ to the parent probe direction $\mathbf{w}_p$ are candidate child features.

**Cosine similarity threshold.** We use a threshold of 0.30 for parent-child identification. This threshold was validated as robust across the range 0.20--0.40 (coefficient of variation = 0.077 across 5 thresholds $\times$ 2 SAE configurations; subtype ordering is preserved in 10/10 cells). The data-driven 95th-percentile of random cosine similarity is 0.044--0.049, confirming all tested thresholds are well above chance (Appendix E).

**Controls.** Two controls validate the pipeline:
- *Random direction baseline:* Replace the parent probe direction with a random unit vector. If the pipeline still reports high "absorption," the measurement is an artifact.
- *Shuffled hierarchy control:* Randomly reassign parent labels across entities and re-measure absorption. Genuine absorption requires the correct hierarchy structure.

## 3.4 Activation Patching Protocol

Activation patching provides causal---not merely correlational---evidence for competitive exclusion. For each absorbed instance (probe correct on raw, incorrect on SAE):

1. Identify the primary child feature $c$ via integrated-gradients attribution on the SAE latent activations.
2. Zero the child feature: $\mathbf{z}^{(c \to 0)}_c = 0$, keeping all other latents unchanged.
3. Reconstruct from the modified latents: $\hat{\mathbf{x}}^{(c \to 0)} = \mathbf{W}_{\text{dec}} \mathbf{z}^{(c \to 0)} + \mathbf{b}_{\text{dec}}$.
4. Apply the parent probe to $\hat{\mathbf{x}}^{(c \to 0)}$. If the probe now predicts correctly ($\hat{y} = y$), this instance shows *recovery*---the child feature was suppressing the parent.

**Control condition.** For each word/entity, zero a randomly selected non-child feature matched by activation magnitude rather than the identified child feature. The control recovery rate estimates the baseline rate of probe prediction changes from single-feature zeroing.

**Statistics.** Recovery rates are compared between child-zeroed and control conditions via Wilcoxon signed-rank test (paired by word/entity). Effect sizes are reported as Cohen's $d$. Confidence intervals use the bootstrap percentile method with 10,000 resamples.

**Scale.** For first-letter, 25 words (19 with absorption) are tested, each in 200 prompt contexts. For city-continent, 128 entities across 4,902 absorbed contexts. For city-language, 201 entities across 7,814 absorbed contexts.

## 3.5 Decoder Information Entanglement Diagnostic

This diagnostic measures the magnitude of parent-direction information encoded in child feature decoder vectors:

1. For each absorbed instance with identified child feature $c$ and parent class $p$:
2. Compute the modified decoder: $\mathbf{d}_c^{(\neg p)} = \mathbf{d}_c - (\mathbf{d}_c \cdot \hat{\mathbf{w}}_p)\hat{\mathbf{w}}_p$, removing the parent probe direction.
3. Measure the logit change $\Delta_{\text{logit}}$ for parent-relevant tokens when using $\mathbf{d}_c^{(\neg p)}$ instead of $\mathbf{d}_c$.
4. An instance is classified as "entangled" at threshold $\tau$ if $|\Delta_{\text{logit}}| > \tau$.
5. **Control:** Ablate a random direction of the same norm as $\hat{\mathbf{w}}_p$; measure the same logit change.

We evaluate at three thresholds ($\tau \in \{0.05, 0.1, 0.2\}$) for robustness.

**Circularity caveat.** This diagnostic shares the parent probe direction $\hat{\mathbf{w}}_p$ with the FN classification itself. It measures decoder geometry---specifically, how much parent-direction information the child decoder carries---not whether the absorbed information is computationally redundant. A genuine computational-redundancy test would require activation-level ablation ($z_p = 0$) or path patching through separate circuits. We report the decoder entanglement magnitude as informative about absorption mechanics, with this circularity limitation acknowledged throughout.

## 3.6 Hedging Decomposition

Not all false negatives represent the same failure mode. We decompose FNs into three categories:

- **Strict absorbed** ($\text{FN}_{\text{strict}}$): The main parent feature does not fire ($z_p = 0$). The SAE dictionary has a genuine gap---no feature represents the parent concept for this input.
- **Compensatory** ($\text{FN}_{\text{comp}}$): The main parent feature fires ($z_p > 0$) but the probe still classifies the SAE reconstruction incorrectly. Other features interfere with the probe direction.
- **Persistent** ($\text{FN}_{\text{persist}}$): Residual false negatives that fit neither category (probe boundary errors).

This three-way classification refines the original hedging analysis of Chanin et al. (2024), which used a *loose* criterion: any FN where *any* parent-related feature fires was classified as "hedged." That loose classification yields 92.6%---near-tautological, since high-$L_0$ SAEs almost always have some parent-related feature active. Our *strict* classification uses only the main (highest-ranked) parent feature, yielding 0--22.6% across hierarchies.

For first-letter, we additionally report a multi-$L_0$ analysis ($L_0$ from 22 to 176) to examine how the strict/compensatory decomposition varies with sparsity.

## 3.7 Probe Degradation Ablation

The probe degradation ablation is a control experiment that disentangles the probe quality confound from genuine hierarchy effects. First-letter probes achieve $F_1 = 1.0$; RAVEL probes range from $F_1 = 0.73$ to $0.87$. If cross-domain absorption variation merely reflects this difference in probe quality, then degrading the first-letter probes to match RAVEL $F_1$ levels should reproduce RAVEL absorption rates.

**Protocol.** Starting from trained first-letter probes at layer 24 ($F_1 = 1.0$):

1. Inject Gaussian noise into the probe weight matrix: $\mathbf{W}_p^{(\epsilon)} = \mathbf{W}_p + \epsilon \cdot \mathcal{N}(0, I)$, calibrating $\epsilon$ to degrade test-set $F_1$ to seven target levels: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, and 1.0 (undegraded control).
2. For each degradation level, run the full absorption measurement pipeline (Section 3.3) at layer 24 with the 16k JumpReLU SAE.
3. Average absorption rates across 3 noise seeds per degradation level for stability.
4. Fit a regression of absorption rate $\alpha^{(\epsilon)}$ on degraded probe $F_1^{(\epsilon)}$ to estimate the slope $\beta_1$ of the probe degradation curve.
5. Overlay the three RAVEL hierarchies at their actual $F_1$ levels. The vertical distance from each RAVEL point to the curve (the *curve delta*) estimates the hierarchy-specific effect that probe quality alone cannot explain.

**Interpretation.** If a RAVEL hierarchy falls on or near the curve, its absorption rate is explained by probe quality. If it deviates by more than 10 percentage points, the hierarchy has a genuine hierarchy-specific absorption effect. The probe degradation ablation was conducted with 11,725 tokens per degradation level.

---

# 4 Cross-Domain Absorption Results

With validated probes and a documented measurement pipeline (Section 3), we now measure feature absorption across four hierarchy types, two SAE widths, and four transformer layers on Gemma 2 2B. We report five findings: cross-domain variation is statistically significant (Section 4.1), absorption concentrates at the final prediction layer (Section 4.2), wider SAEs reduce absorption (Section 4.3), within-hierarchy variation is large (Section 4.4), first-letter is not the worst case (Section 4.5), and a probe degradation ablation decomposes the variation into probe quality effects and genuine hierarchy effects (Section 4.6).

## 4.1 Cross-Domain Variation Is Statistically Significant

Table 2 presents the central results: absorption rates at layer 24 across all four hierarchy types and both SAE widths.

| Hierarchy | SAE | $\alpha$ (%) | 95% CI | $F_1$ | $N_{\text{entities}}$ | $N_{\text{FN}}$ |
|-----------|-----|-------------|--------|-------|----------------------|----------------|
| First-letter | 16k | **27.1** | [26.3, 34.7] | 1.00 | 2,345 | -- |
| City-continent | 16k | 31.4 | [28.9, 33.9] | 0.87 | 1,567 | 418 |
| City-country | 16k | **45.1** | [42.2, 49.0] | 0.73$^\dagger$ | 1,405 | 515 |
| City-language | 16k | 11.6 | [9.7, 13.5] | 0.82 | 1,229 | 124 |
| First-letter | 65k | -- | -- | 1.00 | 2,345 | -- |
| City-continent | 65k | 31.3 | [28.7, 33.8] | 0.87 | 1,567 | 416 |
| City-country | 65k | 32.9 | [30.2, 35.7] | 0.73$^\dagger$ | 1,405 | 376 |
| City-language | 65k | 7.7 | [6.2, 9.4] | 0.82 | 1,229 | 83 |

**Table 2.** Cross-domain absorption rates at layer 24 on Gemma 2 2B. Bootstrap 95% CI from 10,000 resamples. $\dagger$City-country probe $F_1 = 0.73$ falls below both quality gates; results included with documented caveat. Bold marks the highest and lowest rates within the 16k configuration. First-letter 65k not measured with the RAVEL pipeline; the sae-spelling pipeline does not produce directly comparable per-token rates across SAE widths.

Absorption rates span a 4.1$\times$ descriptive range at layer 24 with 16k SAEs: from 11.6% (city-language) to 45.1% (city-country). Within the three RAVEL hierarchies---which share the same model, layer, SAE, token position, and prompt framework---variation is statistically significant (Kruskal-Wallis $p = 7.4 \times 10^{-66}$). All three within-RAVEL pairwise comparisons are significant after Bonferroni correction: city-continent vs. city-language ($p_{\text{Bonf}} = 1.5 \times 10^{-30}$), city-continent vs. city-country ($p_{\text{Bonf}} = 8.4 \times 10^{-12}$), and city-language vs. city-country ($p_{\text{Bonf}} = 2.1 \times 10^{-67}$).

Comparing RAVEL hierarchies against first-letter requires more caution because of the token position asymmetry (Section 3.1). City-language vs. first-letter reaches significance ($p_{\text{Bonf}} = 0.003$), but city-continent vs. first-letter ($p_{\text{Bonf}} = 1.0$) and city-country vs. first-letter ($p_{\text{Bonf}} = 1.0$) do not. The 4.1$\times$ range is a descriptive finding; not all pairwise comparisons are statistically significant.

## 4.2 Absorption Concentrates at the Final Prediction Layer

Figure 2 shows the absorption profile across transformer layers 6, 12, 18, and 24 for all four hierarchies with 16k SAEs.

![Layer-dependent absorption rates for four hierarchies on Gemma 2 2B with 16k JumpReLU SAEs. Absorption concentrates at layer 24 across all hierarchy types, with first-letter showing a dramatic 26$\times$ increase from layer 6 (1.0%) to layer 24 (27.1%). Shaded bands indicate bootstrap 95% CI.](figures/fig2_layer_absorption.pdf)

First-letter absorption rises from 1.0% at layer 6 to 4.7% at layer 12, dips to 2.0% at layer 18, and jumps to 27.1% at layer 24---a 26$\times$ increase from L6 to L24. The non-monotonic dip at L18 is consistent with layer 18 performing intermediate computations distinct from the final token-prediction computation at L24.

RAVEL hierarchies follow the same pattern: all four hierarchies show their highest absorption at layer 24. City-continent rises from approximately 5% at L6 to 31.4% at L24. City-country reaches 45.1% at L24. This layer concentration is the paper's most robust finding: probe quality is $F_1 = 1.0$ at all layers for first-letter (eliminating probe quality as a confound for the layer effect), and the ordering is consistent across hierarchies.

The concentration at the final layer suggests absorption arises from task-specific prediction computation, not from generic feature representation. Gemma 2 2B has $L = 26$ layers; layer 24 is the last layer at which Gemma Scope provides SAEs.

## 4.3 Width Effect

Wider SAEs (65k vs. 16k dictionary features) generally reduce absorption. The effect is most pronounced for city-country: 45.1% at 16k vs. 32.9% at 65k (a 12.2 percentage point reduction). City-language decreases from 11.6% to 7.7%. City-continent shows almost no change (31.4% vs. 31.3%), suggesting its absorption is driven by probe quality rather than dictionary capacity (Section 4.6 confirms this).

The width effect is consistent with the sparsity-capacity trade-off in absorption: wider dictionaries provide more features to represent parent concepts, reducing the pressure to encode parent information into child decoder vectors. The magnitude of the effect varies by hierarchy, indicating that hierarchy structure interacts with SAE capacity.

## 4.4 Per-Class Variation

Figure 3 shows per-continent absorption rates for city-continent at layer 24, revealing extreme within-hierarchy variance.

![Per-continent absorption rates for city-continent at layer 24 on Gemma 2 2B. Europe (90.2%, $n = 276$) and Oceania (52.9%, $n = 51$) show far higher absorption than Africa (3.9%, $n = 231$) and South America (3.9%, $n = 207$). The 16k and 65k SAEs show nearly identical per-continent patterns.](figures/fig3_perclass_heatmap.pdf)

Europe dominates: 90.2% of probe-correct European city activations lose their continent classification after SAE encoding. Oceania follows at 52.9%, Asia at 24.4%, and North America at 19.1%. Africa and South America show minimal absorption (3.9% each). The 16k-to-65k transition preserves this ordering, with nearly identical per-class rates.

Within city-country, the variance is even more extreme. USA shows 0% absorption (176 entities), while smaller countries such as Albania, Algeria, and Argentina reach 100% (at $n = 12$--$25$ entities each). This pattern---low absorption for high-frequency classes, high absorption for rare classes---is consistent with the SAE learning dedicated features for frequent entities but relying on shared (absorbable) features for rare ones.

The large within-hierarchy variance suggests absorption is driven by specific parent-child pair properties---class frequency, feature specialization, decoder geometry---not by hierarchy-wide factors alone. The per-class patterns carry more information than the hierarchy-level averages.

## 4.5 First-Letter Is Not the Worst Case

At layer 24 with 16k SAEs, first-letter absorption (27.1%) is lower than city-country (45.1%) and comparable to city-continent (31.4%). City-language (11.6%) is significantly lower than first-letter ($p_{\text{Bonf}} = 0.003$). The received wisdom from Chanin et al. (2024)---that first-letter spelling represents a worst-case or typical scenario for absorption---does not hold. The first-letter task, with its perfect probes and balanced class distribution, occupies an intermediate position in the cross-domain absorption spectrum.

This finding reframes the absorption literature. Statements calibrated to 15--35% first-letter absorption rates may underestimate absorption severity for imbalanced knowledge hierarchies (city-country) or overestimate it for many-to-many hierarchies (city-language). Cross-domain evaluation is necessary to characterize the actual range of absorption in practice.

However, the probe quality difference between first-letter ($F_1 = 1.0$) and RAVEL hierarchies ($F_1 = 0.73$--$0.87$) raises a confound: do the higher RAVEL absorption rates reflect genuine hierarchy effects, or are they inflated by imperfect probes? Section 4.6 addresses this question directly.

## 4.6 Probe Degradation Ablation Resolves the Confound

The probe degradation ablation (Section 3.7) tests whether cross-domain absorption variation is a genuine hierarchy effect or a probe quality artifact. We degrade first-letter probe quality via weight noise injection to seven $F_1$ levels (0.70 to 1.0), re-measure absorption at each level, and compare the resulting curve against RAVEL absorption rates at their native $F_1$ values.

Table 3 reports the degradation curve data.

| Target $F_1$ | Actual $F_1$ | $\alpha$ (%) | 95% CI | Seed SD |
|-------------|-------------|-------------|--------|---------|
| 0.70 | 0.685 | 36.1 | [37.9, 42.1] | 1.9 |
| 0.75 | 0.754 | 35.3 | [39.2, 43.4] | 1.1 |
| 0.80 | 0.789 | 34.4 | [37.8, 41.8] | 4.2 |
| 0.85 | 0.846 | 33.6 | [37.0, 40.9] | 5.5 |
| 0.90 | 0.904 | 32.4 | [34.6, 38.3] | 3.6 |
| 0.95 | 0.951 | 28.9 | [30.7, 34.2] | 4.3 |
| 1.00 | 0.999 | **21.6** | [21.6, 24.7] | 0.0 |

**Table 3.** Probe degradation ablation results. First-letter absorption at layer 24 (16k SAE) across 7 degraded probe quality levels. Each degraded level averaged over 3 noise seeds (11,725 tokens per level). Absorption increases monotonically as probe $F_1$ decreases. Bold marks the undegraded control.

Figure 7 shows the degradation curve with RAVEL points overlaid.

![Probe degradation ablation. Blue circles: first-letter absorption at 7 degraded probe $F_1$ levels, with linear fit ($R^2 = 0.777$, $p = 0.009$, slope $= -0.398$) and quadratic fit ($R^2 = 0.942$). RAVEL hierarchies overlaid at their native $F_1$. City-continent (green square) falls within 0.6 pp of the curve---its variation is fully explained by probe quality. City-language (purple diamond) sits 21.3 pp below the curve: a genuine hierarchy-specific outlier that probe quality alone cannot explain. City-country (red triangle) sits 8.5 pp above the curve.](figures/fig7_probe_degradation.pdf)

Three results emerge from the decomposition.

**The degradation curve is well-fitted and perfectly monotonic.** As probe $F_1$ decreases from 1.0 to 0.69, absorption increases from 21.6% to 36.1%. The relationship is perfectly monotonic (Spearman $\rho = -1.0$, $p < 10^{-4}$). A linear model explains 77.7% of the variance ($R^2 = 0.777$, $p = 0.009$, slope $\beta_1 = -0.398$). A quadratic fit captures 94.2% ($R^2 = 0.942$). The curve establishes that probe quality is a major confound: a 0.30-point drop in $F_1$ inflates measured absorption by approximately 14.5 percentage points.

**City-continent variation is fully explained by probe quality.** At $F_1 = 0.87$, the probe degradation curve predicts an absorption rate of 30.8%. City-continent's observed rate is 31.4%, a delta of $+0.6$ percentage points---within the noise of the degradation curve. There is no evidence of a hierarchy-specific effect for city-continent once probe quality is accounted for. The same holds approximately for city-country ($F_1 = 0.73$): the curve predicts 36.6%, the observed rate is 45.1%, a delta of $+8.5$ pp. City-country's modest excess may reflect a genuine hierarchy effect or nonlinear amplification at low probe $F_1$; the distinction cannot be resolved with the current data.

**City-language is a genuine outlier.** At $F_1 = 0.82$, the curve predicts 32.9% absorption. City-language's observed rate is 11.6%---a delta of $-21.3$ percentage points, far outside the prediction interval. Probe quality alone cannot explain why city-language has the lowest absorption of any hierarchy tested. This hierarchy-specific suppression is the strongest evidence that genuine hierarchy effects exist beyond probe quality confounds. The suppression may relate to the many-to-many structure of city-language mappings (multiple cities share a language, multiple languages share a city) or to the model's internal representation of linguistic properties, but these hypotheses remain untested.

**Summary.** Probe quality is a major confound in cross-domain absorption measurement ($R^2 = 0.777$). City-continent's elevated absorption rate is fully explained by its lower probe quality. City-language, however, is a genuine hierarchy-specific anomaly ($\Delta = -21.3$ pp). Future cross-domain absorption studies must include probe degradation controls to separate measurement artifacts from real phenomena.

---

# 5 Mechanism Analysis

The cross-domain absorption results in Section 4 establish that absorption rates vary across hierarchy types and that probe quality explains much of this variation. This section tests whether the underlying *mechanism* is universal. Activation patching (Section 5.1--5.2) provides causal evidence for competitive exclusion across all tested hierarchies. Decoder information entanglement (Section 5.3) measures the magnitude of parent information encoded in child decoder vectors. The hedging decomposition (Section 5.4) distinguishes genuine feature gaps from compensatory interference.

## 5.1 Causal Confirmation via Activation Patching (First-Letter)

For 25 first-letter words (19 with measurable absorption), each evaluated in 200 prompt contexts, zeroing the identified child feature recovers the parent probe prediction in 32.5% of absorbed instances. The control condition---zeroing a random non-child feature matched by activation magnitude---yields only 1.5% recovery. The difference is large and statistically significant: Wilcoxon signed-rank $p = 0.000218$, Cohen's $d = 1.33$, bootstrap 95% CI for the recovery difference $[0.213, 0.421]$. Of the 19 words with absorption, 16 show positive recovery exceeding their control.

This is the first interventional---not correlational---evidence for competitive exclusion in SAEs. Prior work (Chanin et al., 2024) identified absorption via decoder cosine similarity and integrated-gradients attribution, both of which are observational. The activation patching result confirms the causal claim: the child feature actively suppresses the parent during SAE encoding, and removing the child releases the suppressed parent information.

## 5.2 Cross-Domain Patching Confirms Universal Mechanism

Activation patching on RAVEL hierarchies replicates the first-letter result, establishing that competitive exclusion operates universally across feature hierarchy types.

**City-continent.** 128 entities across 4,902 absorbed contexts. Primary child-zeroed recovery: 61.9% vs. control 5.2%. Cohen's $d = 1.50$ (large). Wilcoxon $p < 10^{-20}$.

**City-language.** 201 entities across 7,814 absorbed contexts. Primary child-zeroed recovery: 34.2% vs. control 6.8%. Cohen's $d = 0.75$ (medium). Wilcoxon $p < 10^{-18}$.

![Cross-domain activation patching results. For each hierarchy, the child-zeroed condition (orange) shows recovery rates far exceeding the magnitude-matched control (gray). First-letter: $d = 1.33$; city-continent: $d = 1.50$; city-language: $d = 0.75$. All three effects are large and unambiguous.](figures/fig4_patching_comparison.pdf)

All three hierarchy types show large, unambiguous positive effects (Figure 4): zeroing the child feature partially restores parent probe accuracy, while zeroing a control feature of matched activation magnitude does not. The mechanism is universal, not first-letter-specific.

Recovery magnitude varies by hierarchy: 61.9% for city-continent, 34.2% for city-language, and 32.5% for first-letter. This likely reflects hierarchy-dependent information distribution rather than distinct mechanisms. City-continent has $K = 6$ coarse-grained parent classes, so removing a single child feature can release a large fraction of the parent information encoded in the decoder. First-letter has $K = 25$ classes with finer-grained parent features, plausibly distributing parent information across more latents. The lower recovery for city-language ($d = 0.75$) may additionally reflect its many-to-many mapping structure, where multiple cities share a language and multiple languages share a city.

## 5.3 Decoder Information Entanglement

Child feature decoder vectors carry large-magnitude parent-direction information, consistently across hierarchy types.

**First-letter.** Mean $|\Delta_{\text{logit}}| = 6.16$ nats ($N = 158$ instances). 100% of instances exceed all classification thresholds ($\tau = 0.05, 0.1, 0.2$ nats). Control (random-direction ablation): mean $|\Delta_{\text{logit}}| = 0.012$ nats.

**City-continent.** Mean $|\Delta_{\text{logit}}| = 3.98$ nats ($N = 1{,}464$ instances). 100% of instances exceed all thresholds. Control: 0.12 nats.

![Decoder information entanglement distributions. Main panel: $|\Delta_{\text{logit}}|$ distributions for first-letter ($N = 158$, mean $= 6.16$ nats) and city-continent ($N = 1{,}464$, mean $= 3.98$ nats). Inset: control distribution clustered near zero. Vertical lines at thresholds $\tau = 0.05, 0.1, 0.2$.](figures/fig5_pathological_histogram.pdf)

The 1.55$\times$ magnitude ratio between hierarchies is modest; both show 100% of instances exceeding all classification thresholds (Figure 5). This consistency across hierarchy types reinforces the universal mechanism conclusion: child decoder vectors systematically encode parent-direction information regardless of whether the hierarchy is syntactic (first-letter) or semantic (city-continent).

We note that this diagnostic shares the probe direction with the false-negative classification, so it measures decoder geometry rather than computational redundancy. A genuine test of whether the absorbed information is computationally recoverable would require activation-level ablation ($z_{\text{parent}} = 0$) or path patching through independent circuits. The result establishes that child decoders carry large-magnitude parent information, which is informative about absorption mechanics but does not resolve the computational-redundancy question.

## 5.4 Hedging Decomposition

Compensatory false negatives dominate across all hierarchies at layer 24. Table 4 reports the three-way decomposition.

| Hierarchy | $N_{\text{FN}}$ | Strict absorbed (%) | Compensatory (%) | Persistent (%) |
|-----------|-----------------|---------------------|-------------------|----------------|
| First-letter (multi-$L_0$) | -- | 7.9 | 86.2 | 5.9 |
| City-continent | 418 | 9.1 | 90.9 | 0.0 |
| City-language | 124 | 22.6 | 77.0 | 0.4 |
| City-country | 515 | ${\sim}5$ | ${\sim}95$ | -- |

**Table 4.** Hedging decomposition across hierarchies at layer 24. Compensatory FNs---where the main parent feature fires but the probe still misclassifies SAE-reconstructed activations---dominate in all hierarchies (77--95%). Strict absorbed FNs, where the main parent feature does not fire at all, range from 5% to 22.6%.

The widely cited loose hedging classification from Chanin et al. (2024) yields 92.6%---near-tautological, since high-$L_0$ SAEs almost always have some parent-related feature active. Our strict classification, using only the main parent feature, reveals genuine variation: city-language has the highest strict absorbed fraction (22.6%), consistent with its low overall absorption rate (the SAE genuinely fails to represent language-related parent features for a subset of entities). The strict/compensatory decomposition is a reusable methodological contribution: future absorption studies should report both fractions to distinguish genuine feature gaps from interference effects.

---

# 6 Architecture Comparison

We test whether SAE architecture mitigates absorption across hierarchy types. Four configurations are compared: JumpReLU 16k, JumpReLU 65k, BatchTopK 16k, and Matryoshka 32k. Absorption is measured at layers 12 and 24 across all four hierarchies, yielding 12 total observations per layer.

![Architecture comparison. Absorption rates grouped by hierarchy, with bars for each SAE architecture. ANOVA: architecture effect $p = 0.50$ (L24), $p = 0.53$ (L12); hierarchy effect $p = 0.041$ (L24), $p = 0.005$ (L12). Hierarchy type is the dominant predictor; architecture bars overlap within each hierarchy cluster. This comparison is underpowered (12 observations) and should not be interpreted as evidence of no architecture effect.](figures/fig6_architecture_comparison.pdf)

No significant architecture effect on absorption is detected. Two-way ANOVA at layer 24: architecture $F = 0.80$, $p = 0.50$; hierarchy $F = 3.76$, $p = 0.041$. At layer 12: architecture $F = 0.67$, $p = 0.53$; hierarchy $F = 7.43$, $p = 0.005$. Hierarchy type is the sole significant predictor at both layers.

Within each hierarchy, the four architectures produce overlapping absorption rates (Figure 6). The width mismatch (Matryoshka 32k versus others 16k or 65k) is an uncontrolled confound: any width-dependent effect could masquerade as an architecture effect or cancel one out.

This null result should be interpreted as "effect not detected with limited statistical power," not "no effect exists." SAEBench reports large architecture differences on first-letter absorption (Matryoshka ${\sim}0.03$ vs. BatchTopK ${\sim}0.29$), measured under standardized conditions with far more observations. The 12-observation comparison here lacks the power to detect effects of that magnitude when superimposed on the much larger hierarchy effect. Whether the SAEBench architecture rankings transfer to entity-attribute hierarchies remains an open question requiring a dedicated study with matched widths and sufficient statistical power.

---

# 7 Discussion

The preceding sections established three empirical results: absorption rates vary across hierarchy types (Section 4), activation patching confirms a universal competitive exclusion mechanism (Section 5), and architecture choice has no detected effect relative to hierarchy type (Section 6). This section synthesizes these findings into implications for the field, addresses failed approaches, and catalogues limitations.

## 7.1 Causal Methods Succeed Where Correlational Methods Fail

Four independent correlational and statistical approaches to predicting absorption all failed:

- **Geometric Absorption Score (GAS):** $\rho = 0.116$, AUROC $= 0.571$, bootstrap 95% CI $[-0.333, 0.536]$ (Appendix B). GAS measures decoder-activation geometry---the angular mismatch between a feature's decoder direction and its typical activation pattern. The 25-word full-scale evaluation confirms the pilot null: decoder geometry does not predict which features will be absorbed.

- **Conditional mutual information (CMI):** $\rho = 0.044$, $p = 0.84$ at $L_0 = 22$ (Appendix C). At this sparsity level, all 25 first-letter probes achieve $F_1 = 1.0$, eliminating probe quality as a confound. The bootstrap CI $[-0.41, 0.47]$ firmly includes zero. Information-theoretic dependence between parent and child activations does not predict absorption.

- **Absorption Tax $T(G)$:** Ranking $\rho = -0.20$, concordance 50% (chance level; Appendix D). The theoretical construct---the minimum additional $L_0$ cost for absorption-free representation of a hierarchy---produces quantitative predictions that are no better than random ordering. The qualitative insight (absorption imposes a sparsity cost) is retained; the quantitative framework is not supported.

- **Rate-distortion three-factor model:** $\rho = 0.286$, $R^2 = 0.104$ across 131 parent-child pairs from all hierarchies (Appendix F). All three individual predictors---decoder cosine similarity ($\rho = -0.090$), co-occurrence ($\rho = -0.189$), and reconstruction importance ($\rho = -0.239$)---have the wrong sign relative to the hypothesis. Statistical significance ($p < 0.001$) reflects large sample size, not meaningful effect. The direction reversal from the 20-pair pilot ($\rho > 0$) to the 131-pair full evaluation demonstrates small-sample instability.

This quadruple negative establishes a methodological boundary. Absorption is driven by encoder competitive dynamics during inference, not by static geometric or information-theoretic properties of the trained SAE. Decoder cosine similarity, co-occurrence frequency, and reconstruction importance---the most natural candidate predictors---all fail. Only activation patching, which intervenes on the causal pathway (zeroing the child feature and measuring parent recovery), successfully characterizes absorption. Future absorption research should prioritize interventional methods over correlational proxies.

## 7.2 Universal Competitive Exclusion with Hierarchy-Dependent Recovery

Activation patching confirms that the same mechanism---competitive exclusion of the parent feature by the child feature during SAE encoding---operates across all three tested hierarchy types:

- First-letter: recovery 32.5% vs. control 1.5%, Cohen's $d = 1.33$, $p = 0.000218$
- City-continent: recovery 61.9% vs. control 5.2%, $d = 1.50$, $p < 10^{-20}$
- City-language: recovery 34.2% vs. control 6.8%, $d = 0.75$, $p < 10^{-18}$

All three effects are large and unambiguous: zeroing the child feature partially restores parent probe accuracy, while zeroing a control feature of matched activation magnitude does not. The mechanism is universal, not first-letter-specific.

Recovery magnitude varies by hierarchy (61.9% for city-continent versus 32.5% for first-letter), but this likely reflects hierarchy-dependent information distribution rather than distinct mechanisms. City-continent has $K = 6$ coarse-grained parent classes, so removing a single child feature can release a large fraction of the parent information encoded in the decoder. First-letter has $K = 25$ classes with finer-grained parent features, plausibly distributing parent information across more latents. The lower recovery for city-language ($d = 0.75$) may additionally reflect its many-to-many mapping structure, where multiple cities share a language and multiple languages share a city.

Decoder information entanglement is consistent with this account. Child decoder vectors carry $|\Delta_{\text{logit}}| = 6.16$ nats (first-letter, $N = 158$) and 3.98 nats (city-continent, $N = 1{,}464$) of parent-direction information, versus 0.012 nats for random-direction controls. The 1.55$\times$ magnitude ratio between hierarchies is modest; both show 100% of instances exceeding all classification thresholds (0.05, 0.1, 0.2 nats). This consistency across hierarchy types reinforces the universal mechanism conclusion. The diagnostic shares the probe direction with the false-negative classification, so it measures decoder geometry rather than computational redundancy. A genuine test of whether the absorbed information is computationally recoverable would require activation-level ablation ($z_{\text{parent}} = 0$) or path patching through independent circuits.

## 7.3 Probe Quality as a Major Confound

The probe degradation ablation (Section 4.6) is the paper's most important methodological contribution. Degrading first-letter probe $F_1$ from 1.0 to 0.69 via weight noise injection produces a well-fitted absorption curve: $R^2 = 0.777$ (linear), $R^2 = 0.942$ (quadratic), Spearman $\rho = -1.0$ (perfect monotonic), $p = 0.009$. A 0.30-point drop in probe $F_1$ inflates measured absorption by approximately 14.5 percentage points.

Overlaying RAVEL absorption rates on this curve decomposes the cross-domain variation into two sources:

**Probe quality effect.** City-continent absorption (31.4% at $F_1 = 0.87$) falls within 0.6 percentage points of the degradation curve prediction (30.8%). City-country absorption (45.1% at $F_1 = 0.73$) is within 8.5 pp of the prediction (36.6%). For these hierarchies, the elevated absorption rates are largely---or entirely---explained by their lower probe quality relative to first-letter.

**Genuine hierarchy-specific effect.** City-language absorption (11.6% at $F_1 = 0.82$) sits 21.3 pp below the curve prediction (32.9%). Probe quality alone cannot explain why city-language has the lowest absorption of any hierarchy tested. This suppression effect is the strongest evidence that genuine hierarchy-specific factors modulate absorption beyond probe measurement artifacts.

The recommendation for the field is concrete: any cross-domain absorption study must include probe degradation controls. Without them, apparent cross-domain differences may reflect probe quality variation rather than genuine hierarchy effects. The first-letter task, with its $F_1 = 1.0$ probes, is the only hierarchy in our study where the measured absorption rate is free of this confound.

The city-language anomaly warrants investigation. The many-to-many mapping structure (multiple cities share a language; a city may have multiple official languages) is qualitatively different from the many-to-one structures of first-letter, city-continent, and city-country. This structural difference may reduce the competitive pressure between parent and child features---if the parent concept (language) is not cleanly encoded as a single direction in the residual stream, competitive exclusion may be weaker. This hypothesis is untested and constitutes future work.

## 7.4 Implications for SAE Reliability

Absorption rates of 11--45% at the final prediction layer (layer 24 of 26 in Gemma 2 2B) are directly relevant to SAE deployment for safety applications. Google DeepMind deprioritized SAE research after finding 10--40% degradation in safety-relevant feature detection using SAE-reconstructed activations (Smith et al., 2025). Our cross-domain results provide a mechanistic account of this degradation: competitive exclusion at the prediction layer causes parent features (e.g., "harmful intent," "deceptive reasoning") to systematically fail when more specific child features fire.

Three aspects of our findings amplify this concern. First, the layer concentration effect (absorption at L24 is 15$\times$ higher than at earlier layers) means absorption disproportionately affects the layers where task-specific computation occurs---precisely where safety-relevant features are most needed. Second, the per-class variance is extreme (Europe 90.2% vs. Africa 3.9% for city-continent), implying that absorption creates blind spots for specific entity subsets rather than uniform degradation. Third, the quadruple negative for correlational predictors means there is currently no unsupervised method to detect which features are absorbed; detection requires supervised probes with known ground truth.

The width effect offers partial mitigation. Wider SAEs (65k vs. 16k) reduce absorption for city-country from 45.1% to 32.9% (a 12.2 pp reduction). Scaling SAE dictionary size provides more features to represent parent concepts, reducing the sparsity pressure that drives absorption. The Matryoshka SAE architecture (Bussmann et al., 2025), which achieves absorption rates of approximately 0.03 on the first-letter benchmark versus 0.29 for BatchTopK, represents a more targeted architectural intervention. Whether these architectural improvements generalize beyond first-letter to entity-attribute hierarchies remains untested.

## 7.5 Hypothesis Verdicts

Table 5 summarizes the outcome of each hypothesis tested in this study.

| Hypothesis | Verdict | Key metric | Confidence | Section |
|-----------|---------|-----------|-----------|---------|
| H1: Cross-domain variation | Supported with nuance | KW $p = 7.4 \times 10^{-66}$ (within-RAVEL) | High | 4.1 |
| H2': Semantic > first-letter at L24 | Refuted | First-letter 27.1% < city-country 45.1% but > city-language 11.6% | High | 4.5 |
| H3: Hedging decomposition varies | Partially supported | Strict 0--22.6% vs. loose 92.6% | Medium | 5.4 |
| H4: GAS unsupervised detector | Definitive negative | $\rho = 0.116$, AUROC $= 0.571$ | High | App. B |
| H5: Absorption Tax $T(G)$ | Not supported | Ranking $\rho = -0.20$ | High | App. D |
| H6: Architecture generalization | Not detected (underpowered) | ANOVA $p = 0.50$--$0.53$; 12 obs. | Low | 6 |
| H7: Causal absorption (first-letter) | Supported | $d = 1.33$, $p = 0.000218$ | High | 5.1 |
| H7-cross: Causal absorption (cross-domain) | Supported | $d = 0.75$--$1.50$, all $p < 10^{-17}$ | High | 5.2 |
| H8: Decoder entanglement (cross-hierarchy) | Consistent | 6.16 nats (first-letter) vs. 3.98 nats (city-continent) | High | 5.3 |
| H9: Rate-distortion predictor | Not supported | $\rho = 0.286$, $R^2 = 0.104$; all predictors wrong sign | High | App. F |
| H10: Probe artifact vs. hierarchy effect | Mixed | $R^2 = 0.777$; city-language outlier $\Delta = -21.3$ pp | High | 4.6 |

**Table 5.** Hypothesis verdicts. Multiple negative results (H4, H5, H9) are reported alongside positive findings (H7, H7-cross). The mixed verdict on H10---probe quality explains most but not all variation---is the study's most nuanced finding.

Of the 11 hypotheses, 4 are supported (H1, H7, H7-cross, H8), 1 is partially supported (H3), 1 is mixed (H10), 1 is refuted (H2'), 3 are negative (H4, H5, H9), and 1 is underpowered (H6). The honest reporting of multiple negative results is deliberate: the quadruple failure of correlational predictors is itself a finding that shapes the field's methodological choices.

## 7.6 Limitations

**Single model.** All experiments use Gemma 2 2B. Generalization to other architectures (Llama, Pythia, GPT-2), model scales (2B to 70B), and training paradigms (instruction-tuned, RLHF) is untested. The layer-dependent absorption profile and per-class patterns may differ for models with different depth, width, or training data composition.

**Probe quality confound is partially but not fully resolved.** The probe degradation ablation ($R^2 = 0.777$) shows probe quality is a major confound, but the control condition ($F_1 = 1.0$ absorption rate of 21.6%) falls below the baseline CI $[26.3\%, 34.7\%]$ due to per-token versus per-word aggregation differences. The trend is consistent (slope $= -0.398$, $p = 0.009$), but absolute calibration between aggregation methods introduces uncertainty.

**Token position asymmetry.** First-letter experiments use token position $-6$ while RAVEL hierarchies use position $-2$. Within-RAVEL comparisons are unaffected (all share position $-2$), but first-letter versus RAVEL comparisons carry this uncontrolled confound. Position $-6$ and $-2$ may differ in the degree to which hierarchical information is encoded in the residual stream.

**Decoder entanglement circularity.** The decoder information entanglement diagnostic (Section 5.3) shares the probe direction with the false-negative classification. It establishes that child decoders carry large-magnitude parent information, but does not answer whether this information is computationally utilized by the model. A genuine computational-redundancy test would require activation-level ablation or path patching through independent circuits.

**Architecture comparison is underpowered.** The ANOVA across four architectures (JumpReLU 16k, JumpReLU 65k, BatchTopK 16k, Matryoshka 32k) has 12 total observations. The width mismatch (Matryoshka 32k versus others 16k or 65k) is an uncontrolled confound. The null result ($p = 0.50$--$0.53$) should be interpreted as "effect not detected," not "no effect exists."

**City-country probe quality.** The city-country probe ($F_1 = 0.73$) falls below both quality gates. Its 45.1% absorption rate is partially but not fully explained by the probe degradation curve (residual $+8.5$ pp). Results for this hierarchy should be treated as exploratory.

**Scope of hierarchy types.** Four hierarchy types (one syntactic, three entity-attribute) were tested. Other important hierarchy types---part-of-speech, taxonomic (animal $\to$ mammal $\to$ dog), syntactic constituency, factual knowledge (person $\to$ profession $\to$ field)---remain unexplored. The city-language anomaly (Section 7.3) demonstrates that hierarchy structure matters, so extending to structurally diverse hierarchies is essential.

---

# 8 Conclusion

This paper presents the first systematic cross-domain characterization of feature absorption in sparse autoencoders, extending measurement from the first-letter spelling benchmark to entity-attribute knowledge hierarchies on Gemma 2 2B. Five contributions emerge from the empirical program.

**1. Cross-domain absorption characterization with quantitative confound decomposition.** Absorption rates at layer 24 span a 4.1$\times$ descriptive range across four hierarchy types: 11.6% (city-language), 27.1% (first-letter), 31.4% (city-continent), and 45.1% (city-country). Within-RAVEL variation is statistically significant (Kruskal-Wallis $p = 7.4 \times 10^{-66}$). A probe degradation ablation---degrading first-letter probes via weight noise injection to seven $F_1$ levels from 0.69 to 1.0---yields a well-fitted absorption-vs-probe-quality curve ($R^2 = 0.777$, $p = 0.009$, Spearman $\rho = -1.0$). City-continent absorption matches the curve within 0.6 pp (fully explained by probe quality). City-language sits 21.3 pp below the curve, identifying it as a genuine hierarchy-specific outlier that probe quality alone cannot account for.

**2. Universal causal mechanism via activation patching.** Zeroing the identified child feature recovers parent probe predictions across all tested hierarchy types: first-letter ($d = 1.33$, $p = 0.000218$), city-continent ($d = 1.50$, $p < 10^{-20}$), and city-language ($d = 0.75$, $p < 10^{-18}$). Recovery rates range from 32.5% to 61.9% versus 1.5--6.8% for activation-magnitude-matched controls. Competitive exclusion is a universal absorption mechanism, not a first-letter-specific artifact.

**3. Decoder information entanglement.** Child decoder vectors carry 6.16 nats (first-letter, $N = 158$) and 3.98 nats (city-continent, $N = 1{,}464$) of parent-direction information, versus 0.012 nats for random-direction controls. Both hierarchies show 100% of instances exceeding all classification thresholds (0.05, 0.1, 0.2 nats). This diagnostic shares the probe direction with the false-negative classification; it measures decoder geometry, not computational redundancy.

**4. Hierarchy dominates architecture.** No significant architecture effect on absorption was detected (ANOVA $p = 0.50$--$0.53$ across JumpReLU, BatchTopK, and Matryoshka SAEs). Hierarchy type is the sole significant predictor ($p = 0.005$--$0.041$). This comparison is underpowered (12 observations) and should be interpreted as "effect not detected," not "no effect exists."

**5. Quadruple negative for correlational predictors.** GAS ($\rho = 0.116$, AUROC $= 0.571$), CMI ($\rho = 0.044$, $p = 0.84$), the Absorption Tax ($\rho = -0.20$, concordance at chance), and a rate-distortion three-factor model ($R^2 = 0.104$, all individual predictors in the wrong direction) all fail to predict absorption. This establishes a methodological boundary: absorption is a causal phenomenon driven by encoder competitive dynamics, not by static geometric or information-theoretic properties of the trained decoder.

## Limitations

All experiments use a single model (Gemma 2 2B). Generalization to other architectures (Llama, Pythia), model scales, and training paradigms (instruction-tuned, RLHF) is untested. The probe degradation ablation resolves the direction of the confound ($R^2 = 0.777$) but the $F_1 = 1.0$ control absorption rate (21.6%) falls below the baseline CI $[26.3\%, 34.7\%]$ due to per-token versus per-word aggregation differences. Token position asymmetry (first-letter at position $-6$, RAVEL at position $-2$) is uncontrolled in cross-framework comparisons. The decoder entanglement diagnostic has acknowledged circularity. The architecture comparison is underpowered. City-country probe quality ($F_1 = 0.73$) falls below both quality gates; its 45.1% absorption rate should be treated as exploratory.

## Future Work

Three directions follow from these results. First, higher-quality probes for entity-attribute hierarchies---via contrastive learning, richer prompt templates, or embedding-based classifiers---would reduce the probe quality confound and sharpen the cross-domain signal. Second, extending the measurement to larger models (Gemma 2 9B, Llama 3) and structurally diverse hierarchies (taxonomic, part-of-speech, factual person-profession) would test whether the city-language suppression effect and the universal competitive exclusion mechanism generalize. Third, a genuine computational-redundancy test---activation-level ablation ($z_{\text{parent}} = 0$) or path patching through independent circuits---would determine whether absorbed parent information is computationally recoverable by the model, resolving the question that the decoder entanglement diagnostic cannot answer due to its circularity.

The probe degradation ablation is the methodological contribution with the broadest applicability. Any future cross-domain absorption study that does not include a probe quality control risks conflating measurement artifacts with genuine hierarchy effects. The first-letter task, with its $F_1 = 1.0$ probes, remains the only hierarchy where the absorption rate is free of this confound---a reason to continue using it as a calibration anchor, not to rely on it as the sole benchmark.

---

## Figures and Tables

- Figure 1: fig1_teaser.pdf --- Cross-domain absorption rates bar chart at L24 with 95% CI and probe $F_1$ annotations
- Figure 2: fig2_layer_absorption.pdf --- Layer-dependent absorption profile across 4 hierarchies and layers 6/12/18/24
- Figure 3: fig3_perclass_heatmap.pdf --- Per-continent absorption heatmap (6 continents $\times$ 2 SAE widths)
- Figure 4: fig4_patching_comparison.pdf --- Cross-domain activation patching results (3 hierarchy types, child-zeroed vs. control)
- Figure 5: fig5_pathological_histogram.pdf --- Decoder information entanglement $|\Delta_{\text{logit}}|$ distributions (first-letter and city-continent)
- Figure 6: fig6_architecture_comparison.pdf --- Architecture comparison grouped bar chart (4 architectures $\times$ 4 hierarchies)
- Figure 7: fig7_probe_degradation.pdf --- Probe degradation ablation curve with RAVEL points overlaid
- Table 1: inline --- Probe quality ($F_1$) across 4 hierarchies $\times$ 4 layers with quality gate status
- Table 2: inline --- Cross-domain absorption rates at L24 with 16k and 65k SAEs
- Table 3: inline --- Probe degradation ablation results (7 $F_1$ levels)
- Table 4: inline --- Hedging decomposition across hierarchies at L24
- Table 5: inline --- Hypothesis verdict summary across all 11 tested hypotheses
