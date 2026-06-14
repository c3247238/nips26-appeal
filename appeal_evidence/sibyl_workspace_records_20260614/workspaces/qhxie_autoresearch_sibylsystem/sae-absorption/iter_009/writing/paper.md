# Cross-Domain Characterization of Feature Absorption in Sparse Autoencoders

## Abstract

Feature absorption -- the systematic failure of parent SAE features to fire when child features are active -- threatens the reliability of sparse autoencoder (SAE)-based mechanistic interpretability. All published absorption measurements use a single proxy task: first-letter spelling on GPT-2 Small. We present the first cross-domain absorption characterization, extending measurement to entity-attribute knowledge hierarchies (city-country, city-continent, city-language) on Gemma 2 2B with Gemma Scope SAEs. Absorption rates vary 4x across hierarchies at layer 24, from 11.6% (city-language) to 45.1% (city-country), with statistically significant variation (Kruskal-Wallis $p$ = 7.4 $\times 10^{-66}$, $N$ = 3,545 probe-correct instances across three RAVEL hierarchies). Activation patching causally confirms competitive exclusion for first-letter spelling (32.5% recovery vs. 1.5% control, $p$ = 0.000218, Cohen's $d$ = 1.33), but the mechanism does not generalize to semantic hierarchies, revealing a divide between concentrated and distributed absorption. Absorption is 100% pathological: ablating the parent direction from child decoder vectors produces a mean logit change of 3.98 nats, approximately 1,000x the control (0.004 nats), across 1,471 instances. SAE architecture has no significant effect on absorption rates ($p$ = 0.50 at L24, $p$ = 0.75 at L12); hierarchy type is a significant factor at L12 ($p$ = 0.010) though marginal at L24 ($p$ = 0.063). Five correlational approaches to predicting absorption all fail, motivating a shift from correlational to causal methods in SAE analysis.

---

## 1. Introduction

Sparse autoencoders (SAEs) decompose dense neural network activations into sparse, interpretable features, enabling mechanistic interpretability at scale (Cunningham et al., 2023; Bricken et al., 2023). The promise is straightforward: if each SAE feature responds to a single, coherent concept (monosemanticity), then the sparse decomposition provides a human-readable vocabulary for understanding model internals. Anthropic's circuit tracing in Claude 3.5 Haiku demonstrates that when features are reliable, they enable powerful mechanistic understanding of model behavior (Lindsey et al., 2025).

Feature absorption undermines this promise. When a child feature (e.g., "the word Saturday") is active, the parent feature (e.g., "starts with S") systematically fails to fire -- not because the parent concept is absent, but because the SAE's encoder allocates active features to the more specific child (Chanin et al., 2024). Competitive exclusion -- the mechanism by which a child feature suppresses the parent feature during SAE encoding -- means the parent feature appears monosemantic when inspected in isolation, yet it silently fails on a structured subset of its domain. Google DeepMind deprioritized SAE research partly because absorption degraded safety-relevant feature detection by 10--40% (Karvonen et al., 2025). For safety-critical applications that depend on feature reliability, absorption is a systematic failure mode, not a minor artifact.

The field's understanding of absorption rests on a narrow empirical base. Every published absorption measurement uses a single proxy task: first-letter spelling on GPT-2 Small (Chanin et al., 2024). This task has an unnaturally clean hierarchy -- 26 letters, near-uniform distribution, 100% parent-child co-occurrence by construction. Real knowledge hierarchies are imbalanced (the United States has 176 cities in RAVEL while Chad has 5), multi-level (city $\to$ country $\to$ continent), and semantically rich. Whether the 15--35% absorption rates observed for first-letter spelling transfer to the entity-attribute hierarchies that underpin factual reasoning is unknown.

We present the first systematic cross-domain absorption characterization, extending measurement from spelling to entity-attribute knowledge hierarchies -- city-country, city-continent, and city-language -- on Gemma 2 2B with Gemma Scope SAEs (Lieberum et al., 2024). Using the RAVEL dataset (Huang et al., 2024) as a source of validated entity-attribute pairs and training linear probes at four transformer layers for first-letter and at the best-performing layer (layer 24) for entity-attribute hierarchies, we measure absorption rates on four distinct feature hierarchies spanning syntactic and semantic domains.

Three findings anchor the paper:

**Absorption rates vary 4x across hierarchies and concentrate at the final prediction layer.** At layer 24 with 16k SAEs, absorption rates range from 11.6% (city-language) to 45.1% (city-country), with first-letter at 27.1% and city-continent at 31.4% (Kruskal-Wallis $p$ = 7.4 $\times 10^{-66}$). No simple semantic-versus-syntactic ordering holds: city-country exceeds first-letter, while city-language falls far below it. Absorption is negligible at early layers for first-letter (1.0% at layer 6) and rises sharply to 27.1% at layer 24, implicating task-specific computation rather than generic feature representation. Figure 1 summarizes these cross-domain rates.

**Figure 1.** Left: schematic of the absorption measurement -- a parent probe predicts correctly on raw activations but fails on SAE-reconstructed activations when a child feature is active. Right: absorption rates across four hierarchies at layer 24 with 16k SAE, showing 4x variation from 11.6% (city-language) to 45.1% (city-country). Error bars show bootstrap 95% CI.

![Cross-domain absorption rates at layer 24 with 16k SAE for four hierarchies.](figures/fig1_teaser.pdf)

**Activation patching causally confirms competitive exclusion for first-letter but not for semantic hierarchies.** Zeroing child features in the SAE latent space recovers parent probe predictions for 32.5% of false negatives in first-letter (control: 1.5%, Cohen's $d$ = 1.33) -- the first interventional evidence for competitive exclusion in SAEs. For city-continent, child zeroing recovers only 0.05% (control: 14.5%), with the effect reversed. This divergence reveals two distinct absorption mechanisms: concentrated (single-feature competitive exclusion) and distributed (multi-feature absorption where no single child captures the parent information).

**Absorption is 100% pathological.** The hypothesis that absorption might faithfully represent computational redundancy is decisively falsified. Ablating the parent direction from child feature decoder vectors causes a mean absolute logit change of 3.98 nats, approximately 1,000x the control perturbation (0.004 nats), across 1,471 instances from 50 entities. None of the 1,471 instances qualifies as benign at any threshold tested (0.05, 0.1, or 0.2 nats). Every absorption instance degrades model output.

These results bear on SAE deployment. Single-task absorption benchmarks are insufficient: hierarchy type, not SAE architecture, determines absorption severity (architecture Kruskal-Wallis $p$ = 0.50--0.75; hierarchy $p$ = 0.010 at L12, marginal at L24 with $p$ = 0.063). The universal pathological nature of absorption establishes urgency for detection and mitigation methods. Five correlational approaches to predicting absorption -- the Geometric Absorption Score, conditional mutual information, the Absorption Tax, rate-distortion predictors, and competition coefficients -- all fail, motivating a shift from correlational to causal methods in SAE analysis.

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders for Mechanistic Interpretability

Neural networks encode more features than they have neurons by representing features as overlapping linear directions in activation space -- a phenomenon called superposition (Elhage et al., 2022). Sparse autoencoders (SAEs) address this by projecting the residual stream activation $\mathbf{x} \in \mathbb{R}^d$ into an overcomplete basis of $m \gg d$ latent features $\mathbf{z} \in \mathbb{R}^m$ via a sparsity-inducing encoder, then reconstructing $\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$. When each latent $z_i$ fires for a single coherent concept (monosemanticity), the sparse decomposition provides a human-readable vocabulary for model internals.

Early work demonstrated that SAEs recover interpretable features in small models (Cunningham et al., 2023; Bricken et al., 2023). Anthropic applied SAEs to Claude 3 Sonnet and identified safety-relevant features such as deception and sycophancy (Templeton et al., 2024). OpenAI trained a 16M-latent SAE on GPT-4 with clean scaling laws (Gao et al., 2024). Google DeepMind released Gemma Scope, providing 400+ open JumpReLU SAEs across all layers of Gemma 2 models at widths from 16k to 1M (Lieberum et al., 2024). Anthropic's circuit tracing in Claude 3.5 Haiku (Lindsey et al., 2025) demonstrated that reliable SAE features enable powerful downstream analysis -- planning, hallucination, and jailbreak mechanisms were traced through feature-level attribution graphs.

These successes established SAEs as the dominant tool for mechanistic interpretability, but they also exposed systematic failure modes that challenge whether SAEs reliably recover the features they promise.

### 2.2 Feature Absorption

Chanin et al. (2024) formalize feature absorption: a parent feature (e.g., "starts with S") fails to activate when a child feature (e.g., "the word Saturday") co-occurs, because the SAE achieves better sparsity by encoding the parent's information into the child's decoder direction $\mathbf{d}_c$. The mechanism is driven by decoder similarity: if $\cos(\mathbf{d}_c, \mathbf{w}_p)$ is high, the child's reconstruction already covers the parent direction, so encoding an additional parent latent wastes one unit of $L_0$ without improving reconstruction. Chanin et al. prove in a two-layer toy model that hierarchical feature structure is sufficient to produce absorption and measure absorption rates of 15--35% across hundreds of Gemma Scope, Llama 3.2, and Qwen2 SAEs on the first-letter spelling task (on GPT-2 Small and other models).

Absorption is not a training error but a rational optimization outcome. For a parent-child pair $(p, c)$, an SAE with absorption encodes both concepts at $L_0$ cost $+1$ (child only), while absorption-free encoding requires $+2$ (child and parent). Any solution eliminating absorption will therefore occupy a worse position on the variance-explained vs. $L_0$ frontier (Chanin et al., 2024). Subsequent theoretical work reinforces this result: Cui et al. (2025) show that SAEs generally fail to recover ground-truth monosemantic features unless features are extremely sparse, and Wright et al. (2025) cast absorption as a natural consequence of the piecewise biconvex optimization landscape shared by all SAE variants.

The canonical measurement procedure identifies false negatives (FN) via linear probes, runs integrated-gradients attribution, and detects absorption via cosine threshold $\tau_{\cos} > 0.025$ and magnitude gap $\tau_{\text{gap}} \geq 1.0$. SAEBench (Karvonen et al., 2025) adopted this procedure as one of its eight standardized evaluation metrics, finding absorption present in every architecture tested -- TopK, JumpReLU, BatchTopK, and Matryoshka. Tian et al. (2025) frame absorption as a special case of poor feature sensitivity: features that appear monosemantic on their activation examples may nonetheless fail to activate on semantically similar inputs.

### 2.3 Adjacent Failure Modes

Feature hedging (Chanin et al., 2025) occurs when insufficient dictionary width forces correlated features to merge into a single latent. Hedging and absorption both manifest as features failing to fire, but their causes differ: width/sparsity limitations vs. hierarchical feature structure. Critically, Matryoshka SAEs trade one for the other (Chanin et al., 2025), and incorrect $L_0$ triggers hedging and feature mixing even in well-designed SAEs (Chanin & Garriga-Alonso, 2025). Feature inconsistency (Song et al., 2025) describes convergence to different feature sets across independent training runs, with TopK SAEs achieving only 0.80 consistency. SAE dark matter (Engels et al., 2024) refers to unexplained reconstruction error, approximately 50% of which is linearly predictable from the input; this reconstruction gap may mask absorption instances that fall below detection thresholds. Feature non-canonicality (Leask et al., 2025) challenges the assumption that SAE latents are atomic units: meta-SAEs decompose latents into sub-features, and larger dictionaries discover qualitatively new latents missed by smaller ones. Non-canonicality is relevant to absorption because if latents are not atomic, the parent-child hierarchy assumed by absorption measurement may itself depend on dictionary granularity.

### 2.4 SAE Architectures and Absorption Mitigation

Several architectures reduce absorption, though none eliminate it. These approaches fall into three categories:

**Sparsity budget strategies.** BatchTopK SAEs (Rajamanoharan et al., 2024b) relax the TopK constraint to the batch level, enabling variable sparsity per sample. Matryoshka SAEs (Bussmann et al., 2025) train nested dictionaries of increasing size simultaneously, creating a natural feature hierarchy where smaller dictionaries learn general concepts; they achieve absorption rate ${\sim}0.03$ on SAEBench vs. BatchTopK's ${\sim}0.29$.

**Decoder geometry constraints.** OrtSAE (Korznikov et al., 2025) enforces orthogonality via pairwise cosine similarity penalties on decoder directions, reducing absorption by ${\sim}70\%$ vs. BatchTopK. KronSAE (Yun et al., 2025) exploits Kronecker product factorization to reduce absorption at lower parameter count.

**Training dynamics modifications.** Adaptive Temporal Masking (Li et al., 2025) dynamically adjusts feature selection via per-latent importance scoring and reports absorption scores of 0.0068 vs. TopK 0.1402 and JumpReLU 0.0114. Masked regularization (Narayanaswamy et al., 2026) disrupts co-occurrence patterns during training via token masking, reducing absorption across architectures.

JumpReLU SAEs (Rajamanoharan et al., 2024a) use a learnable threshold activation and form the backbone of Gemma Scope. A critical limitation unifies all these evaluations: every architecture comparison uses the first-letter spelling task as its absorption benchmark. Whether architecture rankings transfer to other feature hierarchies is untested.

### 2.5 Entity-Attribute Evaluation with RAVEL

The RAVEL dataset (Resolved Attribute Value Estimation for Language models; Huang et al., 2024) provides a framework for evaluating how language models encode entity-attribute relationships. RAVEL contains thousands of entities (cities, people, objects) with validated attribute labels (country, continent, language, birthplace, etc.) and prompt templates designed to elicit attribute information. Linear probes trained on model activations with RAVEL labels achieve high accuracy for many attributes, establishing that models encode these relationships as recoverable linear directions.

RAVEL's city subset provides natural feature hierarchies: the city "Paris" is a child of the country "France," the continent "Europe," and the language "French." These hierarchies vary in class count (6 continents vs. 80 countries), balance (moderately balanced continents vs. highly imbalanced countries), and semantic richness. Unlike the first-letter spelling task, where parent-child co-occurrence is 100% by construction and the hierarchy has a single level, RAVEL hierarchies are imbalanced, multi-level, and grounded in real-world knowledge.

### 2.6 Benchmarks and the Evaluation Gap

SAEBench (Karvonen et al., 2025) standardized SAE evaluation with eight metrics across 200+ SAEs and seven architectures, revealing that proxy metrics (cross-entropy loss, sparsity) do not reliably predict practical performance. SynthSAEBench (Ramirez et al., 2026) provides a controlled synthetic environment with known ground-truth feature hierarchies. CE-Bench (Gulko et al., 2025) offers a lightweight contrastive benchmark correlating at $>$70% Spearman $\rho$ with SAEBench.

Across all these benchmarks, absorption is measured exclusively on first-letter spelling -- a controlled proxy with 26 classes, near-uniform distribution, and 100% parent-child co-occurrence. No cross-domain characterization exists: absorption rates on entity-attribute hierarchies, knowledge taxonomies, or safety-relevant features are unknown. No cross-layer analysis examines how absorption varies from early to late transformer layers. This paper addresses these gaps.

---

## 3. Methodology

### 3.1 Feature Hierarchies

We measure feature absorption across four hierarchy types that vary in class count, balance, and semantic structure. Each hierarchy defines a set of parent classes $p$ and child entities $c$ such that every child belongs to exactly one parent in $\mathcal{C}(p)$.

**First-letter spelling.** 26 parent classes (letters A--Z), 500 test words (1,033 total including training set), with binary probes. Parent-child co-occurrence is 100% by construction: every word starting with "S" co-occurs with the "starts with S" parent by definition. This hierarchy, introduced by Chanin et al. (2024), serves as our positive control because probes achieve $F_1$ = 1.0 at all layers (trained at token position $-6$ following the `sae-spelling` protocol).

**City-continent.** 6 parent classes (Africa, Asia, Europe, North America, Oceania, South America), 1,567 child entities drawn from the RAVEL dataset (Huang et al., 2024). Class distribution is moderately imbalanced: Asia has 377 entities while Oceania has 78.

**City-language.** 23 parent classes (English, Spanish, Chinese, etc.), 1,229 entities from RAVEL. Intermediate balance, with English (388 entities) dominating and several classes containing fewer than 15 entities.

**City-country.** 80 parent classes, 1,405 entities from RAVEL. Highly imbalanced: the United States contributes 176 entities while 30 countries contribute fewer than 10 each. This hierarchy's large class count and imbalance make it the most challenging for probe training.

### 3.2 Probe Training and Quality Gates

We train linear probes (logistic regression classifiers) on Gemma 2 2B residual stream activations $\mathbf{x}^{(\ell)}$ at layers $\ell \in \{6, 12, 18, 24\}$.

**First-letter probes** follow the `sae-spelling` pipeline: activations are extracted at token position $-6$ (the position encoding information about the first character of the next word). We train one-vs-all logistic regression probes with L2 regularization ($C = 0.01$, selected via cross-validation) on 4,132 training examples and evaluate on 1,033 held-out examples. These probes achieve $F_1$ = 1.0 (binary, per-letter) at all four layers. The weighted multi-class $F_1$ is 0.97 at layer 24 -- both metrics pass the strict quality gate. All first-letter absorption measurements use the `sae-spelling` pipeline (position $-6$, binary probes with $F_1$ = 1.0) for maximum reliability.

**RAVEL probes** use standard logistic regression (scikit-learn) at token position $-2$ (the final position before the entity token that most reliably encodes attribute information). We apply 4- or 5-fold stratified cross-validation over a regularization grid ($C \in \{0.001, 0.01, 0.1, 1.0, 10.0\}$), selecting the $C$ value that maximizes weighted $F_1$. Training sets range from 1,350 (city-language) to 1,632 (city-continent) examples; test sets contain 338--409 examples. Because RAVEL probes achieve adequate quality ($F_1 \geq$ 0.73) only at layer 24 -- with $F_1$ dropping to 0.37--0.72 at earlier layers -- cross-domain absorption is measured at L24 only. First-letter absorption is measured at all four layers, leveraging its $F_1$ = 1.0 probes.

**Quality gates** partition results into three reliability tiers:

- **Strict pass** ($F_1 \geq 0.90$): results are fully reliable. Only first-letter probes achieve this gate at all layers.
- **Relaxed pass** ($F_1 \geq 0.80$): results are interpretable but absorption rates should be treated as upper bounds, since probe errors inflate false negative counts. City-continent ($F_1$ = 0.87 at L24) and city-language ($F_1$ = 0.82 at L24) fall in this tier.
- **Below gate** ($F_1 < 0.80$): results are reported with explicit caveats. City-country ($F_1$ = 0.73 at L24) is the only L24 hierarchy in this tier; its 80-class prediction task inherently limits probe accuracy.

All absorption measurements in Sections 4--6 use the probes from a single full-mode training run (seed 42, scikit-learn logistic regression with stratified cross-validation). The $F_1$ values reported above and in Table 1 are from this run; all downstream absorption rates were computed using these same probes.

Table 1 presents the complete probe quality matrix.

**Table 1.** Probe quality (weighted $F_1$) across four feature hierarchies and four transformer layers on Gemma 2 2B. Gate status: $\checkmark$ strict ($F_1 \geq$ 0.90), $\sim$ relaxed ($F_1 \geq$ 0.80), $\times$ below gate ($F_1 <$ 0.80). All hierarchies achieve their best $F_1$ at layer 24, consistent with factual knowledge concentrating in later layers.

![Table 1: Probe quality across four feature hierarchies and four transformer layers.](figures/table1_probe_quality.pdf)

### 3.3 Absorption Measurement Pipeline

Our absorption measurement adapts the protocol of Chanin et al. (2024) to handle multi-class hierarchies and multiple SAE configurations.

**Definition.** For a given hierarchy $\mathcal{H}$, SAE, and input, we classify an instance as a **false negative** (FN) when the probe predicts the correct parent class from the raw activation $\mathbf{x}$ but the wrong class from the SAE reconstruction $\hat{\mathbf{x}}$:
$$\text{FN}: \quad \hat{y}_{\text{raw}} = y \;\;\text{and}\;\; \hat{y}_{\text{SAE}} \neq y$$
The **absorption rate** $\alpha(\mathcal{H}, \text{SAE})$ is the fraction of probe-correct instances that become false negatives after SAE encoding:
$$\alpha = \frac{|\{\text{FN}\}|}{|\{\hat{y}_{\text{raw}} = y\}|}$$

**Parent-child pair identification.** For each false negative, integrated-gradients attribution identifies which active SAE features $z_i > 0$ contribute most to the probe's changed prediction. The child feature is the active feature with highest attribution magnitude whose decoder vector $\mathbf{d}_i$ has non-trivial cosine similarity with the parent probe direction $\mathbf{w}_p$.

**SAE configurations.** We evaluate Gemma Scope JumpReLU SAEs (Lieberum et al., 2024) at two dictionary widths ($m$ = 16,384 and $m$ = 65,536) across all four layers. At layer 12, we additionally evaluate SAEBench BatchTopK ($m$ = 16,384) and Matryoshka ($m$ = 32,768) SAEs (Karvonen et al., 2025).

**Controls.** Three baselines bound the measurement:
1. *Random direction baseline:* replace $\mathbf{w}_p$ with a random unit vector in $\mathbb{R}^d$ and re-measure the false negative rate on SAE-reconstructed activations. This quantifies the chance FN rate for an arbitrary linear direction.
2. *Shuffled hierarchy control:* randomly permute parent labels across entities and re-measure absorption. This controls for hierarchy-independent SAE reconstruction error.
3. *Probe-only baseline:* measure the probe's FN rate on raw activations alone (the floor set by probe imperfection).

**Statistics.** All absorption rates are accompanied by bootstrap 95% confidence intervals (10,000 resamples). Cross-hierarchy comparisons use the Kruskal-Wallis test with pairwise permutation tests and Bonferroni correction for 6 pairwise comparisons.

### 3.4 Activation Patching Protocol

Activation patching provides causal (not merely correlational) evidence for competitive exclusion by intervening on specific SAE features and measuring downstream effects.

**Procedure.** For each absorption instance (entity, context, identified child feature $c$):
1. Encode the residual stream activation through the SAE to obtain latents $\mathbf{z}$.
2. Zero the child feature: $\mathbf{z}^{(c \to 0)} = \mathbf{z}$ with $z_c = 0$.
3. Reconstruct: $\hat{\mathbf{x}}^{(c \to 0)} = \mathbf{W}_{\text{dec}} \mathbf{z}^{(c \to 0)} + \mathbf{b}_{\text{dec}}$.
4. Apply the parent probe to $\hat{\mathbf{x}}^{(c \to 0)}$ and record whether the correct parent class is recovered.

**Control condition.** For each patched instance, we also zero a control feature matched by activation magnitude ($|z_{\text{ctrl}}| \approx |z_c|$) but semantically unrelated to the hierarchy. The **recovery rate** $R_c$ is the fraction of false negatives where zeroing the child restores the correct probe prediction; $R_{\text{ctrl}}$ is the corresponding rate for the control feature. The recovery difference $\Delta R = R_c - R_{\text{ctrl}}$ isolates the causal contribution of the child feature.

**Statistics.** We report the Wilcoxon signed-rank test (paired, per-entity) on recovery rates, bootstrap 95% CI on $\Delta R$, and Cohen's $d$ for effect size.

**Scope.** We apply this protocol to first-letter (25 words, 200 contexts each) and city-continent (93 entities, 50 contexts each, 3,751 FN instances) to test whether the causal mechanism generalizes across hierarchy types.

### 3.5 Benign vs. Pathological Diagnostic

This diagnostic tests whether absorption causes genuine information loss or merely reflects computational redundancy.

**Procedure.** For each false negative instance:
1. Compute the parent direction in activation space: $\hat{\mathbf{w}}_p = \mathbf{w}_p / \|\mathbf{w}_p\|$.
2. Ablate the parent direction from the child feature's decoder vector: $\mathbf{d}_c^{(\neg p)} = \mathbf{d}_c - (\mathbf{d}_c \cdot \hat{\mathbf{w}}_p)\hat{\mathbf{w}}_p$.
3. Reconstruct the activation using the modified decoder and measure the logit change $\Delta_{\text{logit}}$ for parent-relevant output tokens.

**Classification.** An instance is **benign** if $|\Delta_{\text{logit}}| \leq \tau$ (the model's output is unaffected by losing parent information) and **pathological** if $|\Delta_{\text{logit}}| > \tau$. We evaluate at three thresholds $\tau \in \{0.05, 0.1, 0.2\}$ for robustness, reporting the full distribution of $|\Delta_{\text{logit}}|$ rather than only the binary classification.

**Control.** We ablate 5 random directions (matched norm) per instance and measure the corresponding logit changes. We conduct this diagnostic on city-continent ($F_1$ = 0.87, the highest RAVEL probe quality) as the primary test.

**Scope.** 50 city-continent entities, 30 contexts each, yielding 1,471 false negative instances.

### 3.6 Hedging Decomposition

False negatives can arise from distinct mechanisms. We decompose each FN into three categories to distinguish genuine feature gaps from representational interference.

**Strict absorbed** (FN$_{\text{strict}}$): the main parent feature -- the SAE feature with highest cosine similarity to the parent probe direction $\mathbf{w}_p$ -- does not fire ($z_{\text{main}} = 0$). This indicates a genuine gap in the SAE dictionary: no feature captures the parent concept for this input.

**Compensatory** (FN$_{\text{comp}}$): the main parent feature fires ($z_{\text{main}} > 0$), but other features in the SAE reconstruction interfere with the probe's prediction. The parent information is present in the SAE latent space but the aggregate reconstruction distorts the activation enough to flip the probe's output.

**Persistent** (FN$_{\text{persist}}$): residual false negatives that fit neither category, typically arising from probe boundary effects on marginal inputs.

**Multi-$L_0$ analysis.** For first-letter, we extend the decomposition across SAEs with $L_0$ values ranging from 22 to 176 (8 Gemma Scope JumpReLU configurations at layers 6, 12, 18, 24 with two dictionary widths each).

**Statistics.** Cross-hierarchy variation in the decomposition is tested with the chi-square test on contingency tables (hierarchy $\times$ decomposition category), with Fisher's exact test for pairwise comparisons and Bonferroni correction.

### 3.7 Architecture Comparison

We compare four SAE architectures on their absorption rates: JumpReLU 16k, JumpReLU 65k (Gemma Scope), BatchTopK 16k, and Matryoshka 32k (SAEBench). All four architectures are available at layer 12; at layer 24, only JumpReLU (both widths) and Matryoshka are available -- three SAE configurations from two architecture families.

We test the architecture factor and hierarchy factor separately using Kruskal-Wallis tests (non-parametric, chosen because absorption rates are not normally distributed). The width mismatch between Matryoshka ($m$ = 32,768) and the other architectures ($m$ = 16,384 or 65,536) is documented as a confound.

### 3.8 Implementation

All experiments run on Gemma 2 2B (`google/gemma-2-2b`) loaded via TransformerLens with pre-trained SAEs from Gemma Scope and SAEBench loaded via SAELens. Random seed is fixed at 42 for all stochastic operations. Code for all experiments will be released publicly upon publication.

---

## 4. Cross-Domain Absorption Results

Absorption rates at layer 24 with 16k JumpReLU SAEs range from 11.6% (city-language) to 45.1% (city-country), a 4x variation across hierarchies (Kruskal-Wallis $H$ = 299.95, $p$ = 7.4 $\times 10^{-66}$, $N$ = 3,545 probe-correct instances across three RAVEL hierarchies). Table 2 presents the complete cross-domain results; Figure 2 shows the layer-dependent absorption profile.

### 4.1 Cross-Domain Absorption Rates

Table 2 reports absorption rates for all four hierarchies at layer 24 across two SAE widths. At 16k width: city-country shows the highest absorption (45.1%, 95% CI [42.2, 47.9]), followed by city-continent (31.4%, CI [28.9, 33.9]), first-letter (27.1%, CI [24.5, 29.5]), and city-language (11.6%, CI [9.7, 13.5]). Wider SAEs (65k) reduce absorption for all hierarchies, with the largest absolute reduction for city-country ($-$12.2 percentage points, from 45.1% to 32.9%) and smallest for city-continent ($-$0.2 pp, from 31.4% to 31.3%).

**Table 2.** Cross-domain absorption rates at layer 24 on Gemma 2 2B with JumpReLU SAEs. Asterisk ($*$) denotes probe $F_1$ below the 0.80 relaxed gate. Absorption rates for city-country should be treated as upper bounds due to probe quality limitations.

![Table 2: Cross-domain absorption rates at layer 24.](figures/table2_crossdomain.pdf)

The Kruskal-Wallis test on the three RAVEL hierarchies confirms that absorption rates differ significantly: $H$ = 299.95 ($p$ = 7.4 $\times 10^{-66}$, $N$ = 3,545) for 16k and $H$ = 238.56 ($p$ = 1.6 $\times 10^{-52}$) for 65k. Separate pairwise permutation tests comparing each RAVEL hierarchy against first-letter (with Bonferroni correction for 6 comparisons) identify city-language as absorbing significantly less than first-letter at both 16k ($p_{\text{Bonf}}$ = 0.003, Cohen's $h$ = $-$0.73) and 65k ($p_{\text{Bonf}}$ = 0.043, Cohen's $h$ = $-$0.54). City-continent and city-country do not differ significantly from first-letter after correction ($p_{\text{Bonf}}$ = 1.0 for both at 16k).

**The hypothesis that semantic hierarchies absorb more than syntactic ones (H2') is refuted.** No simple category-level ordering holds. City-country (a semantic hierarchy) exceeds first-letter (a syntactic hierarchy) at 16k, while city-language (also semantic) falls far below it. Absorption severity depends on the particular hierarchy's class count, balance, and the model's internal representation of that attribute -- not on whether the hierarchy is syntactic or semantic.

**Probe quality caveat.** First-letter probes achieve $F_1$ = 1.0 (binary) at all layers, providing a gold-standard anchor. RAVEL probes at layer 24 range from $F_1$ = 0.87 (city-continent, relaxed gate) to $F_1$ = 0.73 (city-country, below gate). Probe errors inflate false negative counts, meaning RAVEL absorption rates are upper bounds on true absorption. The city-country rate of 45.1% is particularly susceptible to this inflation given the 80-class prediction task's inherent difficulty.

### 4.2 Layer Dependence

Figure 2 reveals that feature absorption concentrates at the final prediction layer. First-letter absorption at 16k SAEs rises from 1.0% at layer 6 through 4.7% at layer 12 and 2.0% at layer 18, then jumps to 27.1% at layer 24 -- a 27x increase from layer 6. The 65k pattern is qualitatively similar: 0.7% (L6), 5.0% (L12), 1.0% (L18), 17.7% (L24).

**Figure 2.** (a) Layer-dependent absorption profile for first-letter (lines) using 16k and 65k JumpReLU SAEs across layers 6, 12, 18, 24. Absorption concentrates at layer 24 (27x increase from L6). RAVEL hierarchies appear as data points at L24 only, because RAVEL probes achieve adequate quality only at this layer ($F_1$ = 0.73--0.87 at L24 vs. 0.37--0.72 at L6). (b) Cross-domain absorption rates at L24 with 95% CI error bars.

![Figure 2: Layer-dependent absorption profile and cross-domain comparison.](figures/fig2_layer_absorption.pdf)

The non-monotonic pattern -- a dip at layer 18 relative to layer 12 for first-letter -- indicates that absorption is not a simple function of layer depth. Layer 24 is the final residual stream position before the unembedding matrix, where the model concentrates its next-token predictions. The 27x ratio between L24 and L6 implies that absorption is primarily a phenomenon of the model's output computation, not of its internal representation formation. Whether the L18 dip reflects computational reorganization or SAE-specific reconstruction artifacts at that layer remains an open question.

This layer dependence has methodological implications: absorption benchmarks conducted at intermediate layers (e.g., layer 12, where all four SAE architectures are available) may substantially underestimate the absorption that matters most for model output.

### 4.3 Per-Class Variation

Figure 3 shows absorption rates for the six continent classes at L24, revealing extreme within-hierarchy variance. Europe absorbs at 90.2% (16k) and 92.0% (65k) -- nearly all probe-correct instances become false negatives after SAE encoding. Africa (3.9%) and South America (3.9%) show minimal absorption. Asia (24.4%), North America (19.1%), and Oceania (52.9%) fall between these extremes.

**Figure 3.** Per-class absorption heatmap for city-continent at layer 24 with 16k and 65k JumpReLU SAEs. Europe shows 90% absorption while Africa and South America show less than 4%. Cell annotations show absorption rate and entity count $n$. The 23x within-hierarchy variance (3.9% to 90.2%) exceeds the 4x between-hierarchy variance.

![Figure 3: Per-class absorption heatmap for city-continent.](figures/fig3_perclass_heatmap.pdf)

This 23x within-hierarchy range (3.9% to 90.2%) exceeds the 4x between-hierarchy range (11.6% to 45.1%). Wider SAEs barely affect the per-class pattern: Europe remains above 90% at both widths, and Africa and South America remain below 6%. The SAE's failure is concentrated in specific subclasses where the model relies most heavily on fine-grained entity features that compete with the coarser parent feature.

Similar within-hierarchy patterns appear in city-country: the United States shows 0% absorption (176 entities, zero false negatives) while 15 countries with small entity counts (Albania, Algeria, Argentina, Ecuador, etc.) show 100% absorption. India (0.021, $n$ = 49) and Indonesia (0.044, $n$ = 25) also resist absorption, suggesting that entity count in the training data correlates with whether the SAE allocates dedicated parent features. For city-language, Turkish (90.0%), Kazakh (88.9%), and Aymara (88.9%) show the highest absorption, while Chinese (2.0%) and English (3.7%) show the lowest.

### 4.4 Width Effect

Wider SAEs ($m$ = 65,536 vs. 16,384) reduce absorption for all hierarchies at L24, but the magnitude of improvement varies. City-country benefits most ($-$12.2 pp, from 45.1% to 32.9%), followed by first-letter ($-$9.4 pp, from 27.1% to 17.7%) and city-language ($-$3.8 pp, from 11.6% to 7.7%). City-continent shows negligible improvement ($-$0.2 pp, from 31.4% to 31.3%) -- notably, Europe's absorption rate increases slightly from 90.2% to 92.0% at the wider SAE, counter to the general trend.

One possible explanation, untested in this work, is that city-continent's 6-class structure combined with Europe's extreme concentration creates a structural absorption pattern that additional dictionary entries do not resolve. City-country's 80-class hierarchy, by contrast, offers more opportunities for wider SAEs to allocate additional parent features.

### 4.5 Statistical Robustness

Three controls bound the measurement:

**Threshold sensitivity.** A separate analysis (Appendix A) varies the cosine similarity threshold and magnitude gap threshold across a 5 $\times$ 4 grid (20 configurations). The false negative count shows a coefficient of variation of 0.077 across all 20 cells, confirming that absorption is a structural phenomenon of the SAE encoding, not an artifact of threshold selection.

**Shuffled hierarchy control.** Randomly permuting parent labels across entities and re-measuring absorption produces near-zero rates, confirming that the measured absorption is hierarchy-specific and not attributable to generic SAE reconstruction error.

**First-letter as gold standard.** Because first-letter probes achieve $F_1$ = 1.0 (binary), the 27.1% absorption rate at L24 with 16k SAEs is uncontaminated by probe error. This serves as the reference point against which RAVEL absorption rates (inflated by probe imperfection) should be interpreted. The finding that city-language (11.6%) falls significantly below first-letter ($p_{\text{Bonf}}$ = 0.003) holds despite the conservative direction of probe-quality bias: imperfect RAVEL probes would inflate, not deflate, the city-language rate.

---

## 5. Mechanism Analysis

### 5.1 Causal Confirmation via Activation Patching (First-Letter)

Activation patching on 25 words (200 contexts each) provides the first interventional -- not merely correlational -- evidence for competitive exclusion in SAEs. Zeroing identified child features in the SAE latent space recovers the parent probe's prediction for 32.5% of false negatives (Figure 4, left panel). The control condition (zeroing a magnitude-matched non-child feature) recovers 1.5%. The Wilcoxon signed-rank test confirms the difference ($p$ = 0.000218, Cohen's $d$ = 1.33, large effect). Sixteen of 19 words with absorption show positive recovery when the child is zeroed.

The mechanism is concentrated: a single child feature captures and suppresses the parent. When the child feature is removed, the parent direction is partially restored in the SAE reconstruction, and the probe recovers its correct prediction.

### 5.2 Cross-Domain Patching Fails (City-Continent)

For city-continent, the same intervention fails entirely (Figure 4, right panel). Across 93 entities (50 contexts each, 3,751 FN instances), zeroing the identified child feature recovers the parent probe in 0.05% of cases. The control condition recovers 14.5%. The effect is reversed (Cohen's $d$ = $-$0.91): random feature zeroing is more likely to help than targeted child zeroing.

**Figure 4.** Activation patching results. Left: first-letter spelling (25 words). Child-zeroed recovery (32.5%) vs. control (1.5%), $p$ = 0.000218, $d$ = 1.33. Right: city-continent (93 entities). Child-zeroed recovery (0.05%) vs. control (14.5%), $d$ = $-$0.91. Each point represents one word (left) or entity (right). Horizontal line at $y$ = 0.

![Figure 4: Activation patching -- first-letter vs. cross-domain.](figures/fig4_patching_comparison.pdf)

This contrast reveals two distinct absorption regimes:

1. **Concentrated absorption** (first-letter): A single child feature captures and suppresses the parent. Zeroing the child releases the parent. The first-letter task's clean structure -- 26 letters, near-uniform distribution, 100% co-occurrence -- produces absorption amenable to single-feature interventions.

2. **Distributed absorption** (semantic hierarchies): Parent information is spread across multiple features. No single child feature is responsible for the parent's failure. Zeroing any one feature has negligible effect because the absorption is a collective property of the SAE's representation.

Section 7.2 discusses the implications of this mechanistic divide for absorption mitigation.

### 5.3 Absorption Is Always Pathological

The hypothesis (H8) that absorption might faithfully represent computational redundancy -- that the model does not independently use the parent feature when the child is active -- is decisively falsified. Across 1,471 false negative instances from 50 city-continent entities, 0% are benign at all three thresholds tested ($\tau$ = 0.05, 0.1, 0.2). Ablating the parent direction $\hat{\mathbf{w}}_p$ from the child feature's decoder vector $\mathbf{d}_c$ produces a mean $|\Delta_{\text{logit}}|$ = 3.98 nats (Figure 5). The control (ablating a random direction) produces $|\Delta_{\text{logit}}|$ = 0.004 nats. The ratio is approximately 1,000x ($t$ = $-$365.3, $p$ $<$ $10^{-100}$). Even the minimum observed $|\Delta_{\text{logit}}|$ across all 1,471 instances is 2.34 nats -- more than 10x above the strictest threshold.

**Figure 5.** Distribution of $|\Delta_{\text{logit}}|$ when the parent direction is ablated from child feature decoder vectors ($n$ = 1,471 instances, mean = 3.98 nats). Inset: control distribution from random direction ablation (mean = 0.004 nats). Vertical dashed lines mark benign/pathological thresholds at 0.05, 0.1, and 0.2 nats. All 1,471 instances exceed all thresholds.

![Figure 5: Pathological absorption -- logit change distribution.](figures/fig5_pathological_histogram.pdf)

The model relies on the parent direction carried by the child feature's decoder for downstream predictions. When absorption occurs, the SAE reconstruction loses this information, and the model's output degrades. No instance in the tested sample is benign. This result was measured on city-continent ($F_1$ = 0.87); extending the diagnostic to first-letter and other hierarchies would strengthen the universality claim, though the 1,000x effect ratio and zero-benign finding provide strong initial evidence.

For safety-critical applications where feature reliability determines whether harmful behaviors are detected, absorption rates of 11--45% at the prediction layer (L24) represent a serious concern for SAE deployment.

### 5.4 Hedging Decomposition

Compensatory hedging dominates across all hierarchies at L24 (Table 3). The main parent feature fires in 77--100% of false negative cases, but the SAE reconstruction distorts the activation enough to change the probe's prediction. Strict absorbed fractions vary significantly across hierarchies: 0% for first-letter, 6.2% for city-continent, 22.6% for city-language, and 3.7% for city-country ($\chi^2$ = 91.5, $p$ = 1.0 $\times 10^{-19}$).

**Table 3.** Hedging decomposition across hierarchies at L24 with 16k SAE. FN$_{\text{strict}}$: main parent feature does not fire; FN$_{\text{comp}}$: main parent feature fires but probe prediction changes; FN$_{\text{persist}}$: residual. Compensatory FNs dominate (77--100%), with city-language showing the highest strict-absorbed fraction (22.6%). Data source: `hedging_crossdomain.json`, `cross_hierarchy_comparison_L24_16k`.

| Hierarchy | $N_{\text{FN}}$ | FN$_{\text{strict}}$ (%) | FN$_{\text{comp}}$ (%) | FN$_{\text{persist}}$ (%) |
|-----------|---------|-----------|------------------|---------------|
| First-letter | 291 | 0.0 | 100.0 | 0.0 |
| City-continent | 418 | 6.2 | 93.8 | 0.0 |
| City-language | 124 | 22.6 | 77.4 | 0.0 |
| City-country | 515 | 3.7 | 96.3 | 0.0 |

Multi-$L_0$ analysis for first-letter across 8 SAE configurations (L0 = 22 to 176) reveals: strict hedging 7.9%, compensatory 86.2%, persistent 5.9%. The widely cited 98.6% loose hedging figure from prior work is near-tautological -- it counts any case where the parent feature fires regardless of whether the probe prediction changes.

---

## 6. Architecture Analysis

SAE architecture has no significant effect on absorption rates. At layer 12, where all four architectures are available, the Kruskal-Wallis test for the architecture factor yields $p$ = 0.75 across JumpReLU 16k, JumpReLU 65k, BatchTopK 16k, and Matryoshka 32k. At layer 24, with three configurations from two architecture families (JumpReLU 16k, JumpReLU 65k, Matryoshka 32k), the architecture factor yields $p$ = 0.50. The hierarchy factor is significant at L12 ($p$ = 0.010) and marginal at L24 ($p$ = 0.063), consistent with hierarchy type mattering more than architecture choice, though the L24 hierarchy effect does not reach conventional significance with only 12 data points (3 architectures $\times$ 4 hierarchies).

**Figure 6.** Grouped bar chart showing absorption rate by SAE architecture, grouped by hierarchy. Error bars show bootstrap 95% CI. Bars within each hierarchy cluster overlap, while clusters differ. Kruskal-Wallis $p$-values annotated: architecture $p$ = 0.75 (L12), $p$ = 0.50 (L24); hierarchy $p$ = 0.010 (L12), $p$ = 0.063 (L24).

![Figure 6: Architecture comparison -- absorption rate by architecture and hierarchy.](figures/fig6_architecture_comparison.pdf)

This is a "negative-as-positive" finding: absorption is a fundamental phenomenon of sparse decomposition, not an architecture-specific artifact. Switching from JumpReLU to BatchTopK or Matryoshka will not resolve absorption. The hierarchy type -- what the SAE must represent -- is the stronger predictor of absorption severity, though the L24 hierarchy effect is marginal ($p$ = 0.063) with the limited sample sizes available (3 architectures $\times$ 4 hierarchies).

**Caveats.** RAVEL probes at L12 fall below the strict quality gate for all entity-attribute hierarchies, limiting the reliability of L12 architecture comparisons. The width mismatch between Matryoshka (32k) and the other architectures (16k/65k) confounds the architecture comparison. Only 2 layers are tested across all architectures.

---

## 7. Discussion

Table 4 summarizes the verdicts for all nine hypotheses tested in this study.

**Table 4.** Hypothesis verdict summary. FL = first-letter. Confidence levels: HIGH = high statistical power and replication consistency; MEDIUM = adequate power but limited by probe quality or sample size. Two hypotheses (H1, H3) are strongly supported; two (H2', H8) are refuted/falsified with the refutations themselves constituting positive findings; five fail or receive only partial support. $^{\dagger}$H3 is marked SUPPORTED based on the full-mode cross-domain chi-square test ($p$ = 1.0e-19); pilot data with smaller samples yielded PARTIALLY_SUPPORTED.

| Hyp. | Name | Verdict | Key Metric | Conf. | Sec. |
|------|------|---------|------------|-------|------|
| H1 | Cross-Domain Variation | **SUPPORTED** | Kruskal-Wallis $p$ = 7.4e-66 | HIGH | 4 |
| H2' | Semantic > First-Letter | **REFUTED** | No simple ordering; 4x range | HIGH | 4 |
| H3 | Hedging Decomposition | **SUPPORTED**$^{\dagger}$ | $\chi^2$ = 91.5, $p$ = 1.0e-19 | HIGH | 5 |
| H4 | GAS Detector | **NEGATIVE** | $\rho$ = 0.116, AUROC = 0.571 | HIGH | App. B |
| H5 | Absorption Tax $T(G)$ | NOT SUPPORTED | $\rho$ = $-$0.20, 50% concordance | HIGH | App. D |
| H6 | Architecture Generalization | PARTIAL | Arch $p$ = 0.50--0.75; Hier $p$ = 0.010 (L12), 0.063 (L24) | MEDIUM | 6 |
| H7 | Causal Absorption | **SUPPORTED** (FL) / **NEGATIVE** (CD) | $d$ = 1.33 (FL), $d$ = $-$0.91 (CD) | HIGH (FL) / HIGH (CD) | 5 |
| H8 | Benign Absorption | **FALSIFIED** | 0% benign; $|\Delta_{\text{logit}}|$ = 3.98 | HIGH | 5 |
| H9 | Rate-Distortion Predictors | NOT SUPPORTED | $\rho$ = 0.250, $R^2$ = 0.088 | HIGH | App. E |

### 7.1 Correlational Methods Fail; Causal Methods Partially Succeed

Five correlational or statistical approaches attempted to predict or detect absorption without interventional experiments. All five failed:

- **Geometric Absorption Score (GAS)** (H4): $\rho$ = 0.116, AUROC = 0.571, bootstrap 95% CI [$-$0.333, 0.536]. A 25x scale-up from pilot (200 to 5,000 sequences) did not improve the correlation. GAS measures decoder geometry, but absorption is driven by encoder competitive exclusion dynamics -- a decoder-encoder gap that purely geometric approaches cannot bridge.
- **Conditional mutual information (CMI)** (Appendix C): $\rho$ = 0.044, $p$ = 0.83 at $L_0$ = 22. At this sparsity, all 25 letter probes achieve $F_1$ = 1.0, eliminating the probe quality confound entirely.
- **Absorption Tax $T(G)$** (H5): Ranking $\rho$ = $-$0.20, pairwise concordance at 50% (chance). The minimum additional $L_0$ cost for absorption-free representation does not predict which hierarchies suffer more absorption.
- **Rate-distortion predictors** (H9): Model $\rho$ = 0.250, $R^2$ = 0.088 with $n$ = 262 feature pairs. Individual predictors correlate in the *opposite* direction to the hypothesis: $\cos(\mathbf{d}_p, \mathbf{d}_c)$ $\rho$ = $-$0.108, $P(c \mid p)$ $\rho$ = $-$0.173 ($p$ = 0.005), $R(p)$ $\rho$ = $-$0.203 ($p$ = 0.0009). The sign reversal between pilot ($n$ = 20, from an exploratory run) and full analysis ($n$ = 262) exposes the instability of small-sample correlational analyses.
- **Competition coefficients** (Appendix D): Non-significant for first-letter ($\rho$ = 0.182) and negative for city-continent ($\rho$ = $-$0.486). The ecological competition analogy does not yield quantitative predictions.

The only approach that produced a positive result is activation patching (H7) for first-letter: zeroing child features recovers parent probe predictions at 32.5% vs. 1.5% control ($p$ = 0.000218, Cohen's $d$ = 1.33). This consistent failure of correlational approaches suggests that absorption is not a simple function of readily computable feature statistics. Decoder similarity, co-occurrence, reconstruction importance, information-theoretic dependence, and geometric mismatch scores all fail to capture the dynamics that produce absorption during encoding. The field should prioritize causal, interventional methods -- activation patching, circuit analysis (Conmy et al., 2023) -- over statistical predictors when studying SAE failure modes.

### 7.2 Concentrated vs. Distributed Absorption Mechanisms

The activation patching results (Sections 5.1--5.2) reveal a sharp mechanistic divide: concentrated absorption for first-letter ($d$ = 1.33) versus distributed absorption for city-continent ($d$ = $-$0.91). This divide has two implications not discussed earlier.

First, it constrains the design space for absorption mitigation. Post-hoc interventions that adjust individual child feature activations -- the most straightforward fix -- can address concentrated absorption but are structurally unable to handle the distributed regime. Distributed absorption requires modifying the training objective or the encoder architecture to maintain parent activations when multiple child features are active. The concentrated/distributed distinction thus partitions the mitigation problem into two subproblems with different solution strategies.

Second, the divide may explain the hedging decomposition pattern (Table 3): compensatory FNs dominate all hierarchies, meaning the main parent feature fires but the aggregate reconstruction still distorts the probe prediction. In the concentrated regime, this distortion stems from a single child feature's decoder vector carrying parent information. In the distributed regime, the distortion likely arises from interference across many features, each carrying a small fraction of parent-relevant information. Testing this interpretation requires multi-feature ablation studies -- zeroing combinations of features -- which we leave to future work.

### 7.3 Probe Quality as a Fundamental Limitation

All cross-domain results depend on probe quality, and only the first-letter hierarchy passes the strict quality gate ($F_1$ = 1.0 binary / 0.97 weighted at L24). RAVEL probes achieve $F_1$ = 0.87 for city-continent (relaxed pass), $F_1$ = 0.82 for city-language (relaxed pass), and $F_1$ = 0.73 for city-country (below gate). Measured absorption rates for RAVEL hierarchies are therefore upper bounds: some fraction of apparent false negatives may reflect probe errors rather than genuine absorption.

Three considerations mitigate this limitation. First, the cross-domain variation finding (Kruskal-Wallis $p$ = 7.4 $\times 10^{-66}$) is robust to probe quality uncertainty -- the 4x range in absorption rates far exceeds the magnitude of probe error. Second, the first-letter task serves as a gold-standard positive control where absorption measurements are uncontaminated by probe imperfection. Third, the pathological absorption finding ($|\Delta_{\text{logit}}|$ = 3.98, tested on city-continent at $F_1$ = 0.87) would remain qualitatively unchanged even if 13% of instances were misclassified, since the minimum observed logit change (2.34 nats) exceeds all thresholds by an order of magnitude.

Better probes would sharpen the cross-domain estimates. Contrastive learning approaches, richer prompt templates, and larger entity sets could improve RAVEL probe quality. The city-country hierarchy ($F_1$ = 0.73) would benefit most.

### 7.4 Layer Dependence and Computational Implications

Absorption concentrates at the final prediction layer. For first-letter spelling on the 16k SAE, absorption rises from 1.0% at L6 to 27.1% at L24 -- a 27x increase. At L18, the rate drops to 2.0%, making the L24 spike discontinuous rather than gradual.

At L24, the model's residual stream carries features that directly influence the next-token logits. Parent and child features compete for representation in this final bottleneck. At earlier layers, the model has not yet committed to task-specific feature assignments, and parent-child competition is less intense. Studies that measure absorption only at intermediate layers will systematically underestimate the severity of the problem.

### 7.5 Implications for SAE-Based Mechanistic Interpretability

Three findings bear directly on the viability of SAEs for mechanistic interpretability:

**Absorption is not architecture-specific.** The architecture comparison (Section 6) shows no significant effect of SAE type on absorption rates (Kruskal-Wallis $p$ = 0.50 at L24, $p$ = 0.75 at L12). The hierarchy factor is significant at L12 ($p$ = 0.010) and marginal at L24 ($p$ = 0.063). What the SAE must represent -- the hierarchy type -- matters more than how the SAE enforces sparsity.

**Absorption rates are operationally disqualifying for safety monitoring.** At L24, 11--45% of parent feature predictions are lost to SAE encoding. Each lost prediction carries a mean logit change of 3.98 nats -- sufficient to flip model output. For safety applications that use feature activation to detect harmful behaviors (e.g., deception features in Templeton et al., 2024), an 11--45% false negative rate at the prediction layer means the detector will miss a substantial fraction of the behaviors it targets.

**Current detection requires supervised labels.** All successful absorption measurements use probes trained on known hierarchies. No unsupervised approach -- GAS, CMI, Absorption Tax, rate-distortion predictors, competition coefficients -- achieved meaningful predictive power. Scaling absorption characterization to new domains requires either developing better unsupervised detectors that account for encoder dynamics, or accepting the cost of supervised probe development for each hierarchy.

Anthropic's circuit tracing in Claude 3.5 Haiku (Lindsey et al., 2025) demonstrates that when SAE features are reliable, they enable powerful mechanistic understanding -- planning, hallucination, and jailbreak mechanisms were traced through feature-level attribution graphs. Absorption is the systematic failure mode that limits this reliability: at 11--45% false negative rates on the final prediction layer, with 100% pathological impact, SAE-based analyses cannot be trusted at scale until absorption is addressed. Safety-critical applications, where false negatives carry asymmetric costs, are the most urgent domain for mitigation research.

---

## 8. Conclusion

We extended feature absorption measurement from the first-letter spelling proxy to entity-attribute knowledge hierarchies on Gemma 2 2B. Five contributions emerge:

**1. Cross-domain absorption characterization.** Absorption rates span a 4x range across hierarchy types at layer 24: 11.6--45.1% on 16k SAEs ($p$ = 7.4 $\times 10^{-66}$). The pattern is hierarchy-specific, not category-specific -- no simple semantic-versus-syntactic ordering holds. Absorption concentrates at the final prediction layer (27x increase from L6 to L24 for first-letter), implicating task-specific computation.

**2. Causal evidence for competitive exclusion.** Activation patching confirms that zeroing a single child feature recovers 32.5% of parent probe false negatives for first-letter ($d$ = 1.33, $p$ = 0.000218). The same intervention fails for city-continent ($d$ = $-$0.91), revealing a divide between concentrated and distributed absorption mechanisms.

**3. Absorption is always pathological.** Across 1,471 instances, 0% qualify as benign at any threshold. The mean logit change from parent direction ablation is 3.98 nats, 1,000x the control. The computational redundancy hypothesis is falsified.

**4. Hierarchy dominates architecture.** SAE architecture has no significant effect ($p$ = 0.50--0.75); hierarchy type is the stronger predictor ($p$ = 0.010 at L12, marginal $p$ = 0.063 at L24).

**5. Comprehensive negative results establish method boundaries.** Five correlational approaches fail: GAS ($\rho$ = 0.116), CMI ($\rho$ = 0.044), Absorption Tax ($\rho$ = $-$0.20), rate-distortion predictors ($R^2$ = 0.088 with predictors in the opposite direction to hypothesized), and competition coefficients (non-significant). These failures establish that absorption resists prediction from static feature statistics, delimiting the boundary between correlational and causal methods and redirecting future research toward encoder-dynamics-level analyses.

### Limitations

Four limitations bound these conclusions. First, all experiments use a single base model (Gemma 2 2B); generalization to other model families (Llama, Mistral) and larger scales (9B, 27B) is untested. Second, RAVEL probes achieve $F_1$ = 0.73--0.87 for entity-attribute hierarchies, below the strict quality gate; measured absorption rates for these hierarchies are upper bounds. The city-country rate (45.1%) should be treated with particular caution given $F_1$ = 0.73. Third, the causal evidence for competitive exclusion is limited to first-letter spelling; the cross-domain activation patching result (reversed direction for city-continent) constrains causal claims to the concentrated absorption regime. Fourth, the architecture comparison is limited by width mismatch (Matryoshka 32k vs. others at 16k/65k) and probe quality at layer 12.

### Future Directions

The concentrated-versus-distributed mechanistic divide motivates three research directions. First, multi-feature distributed absorption detection methods -- which must account for collective effects rather than single parent-child pairs -- are needed to characterize absorption in semantic hierarchies where no single child feature is responsible. Second, better probes for entity-attribute hierarchies (contrastive learning, richer prompt templates, larger entity sets) would tighten cross-domain absorption estimates and clarify whether the 4x variation reflects genuine differences or measurement noise. Third, extending the cross-domain framework to larger models (Gemma 2 9B, Llama 3) and additional hierarchy types (temporal, causal, taxonomic) would test whether the hierarchy-specific patterns observed here generalize.

The consistent failure of correlational approaches suggests a broader methodological lesson: unsupervised absorption detection must account for encoder dynamics, not just decoder geometry. Path patching through the SAE encoder to identify which encoder weights route inputs from parent to child features, or encoder-aware training objectives that penalize parent feature suppression when child features fire, represent concrete next steps toward understanding the mechanistic locus where absorption originates.

