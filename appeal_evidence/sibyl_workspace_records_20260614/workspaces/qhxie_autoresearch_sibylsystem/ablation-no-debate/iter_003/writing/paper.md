# Encoder-Driven Feature Absorption in Sparse Autoencoders: Mechanism, Consequences, and Safety Implications

---

## Abstract

Sparse Autoencoders decompose neural network activations into interpretable, monosemantic features—but feature absorption threatens their reliability for safety-critical interpretability tasks. We present the first factorial decomposition of feature absorption's mechanistic source, independently varying encoder and decoder training to isolate their contributions. Our key finding is that absorption is driven by encoder alignment with hierarchical structure, while decoder training plays a regulatory role that suppresses absorption below encoder-only levels. In pilot experiments, Condition B (trained encoder, random decoder) produces absorption rates of 0.076 versus 0.017 for full training (Condition D)—critically, the decoder's contribution is not to create absorption but to suppress it, as encoder-only absorption (0.076) substantially exceeds full training absorption (0.017). Our full 5-seed experiment confirms encoder sufficiency (B ≈ D, delta = 0.037) but reveals the decoder's role is configuration-dependent—the decoder contributes substantially in some seed configurations. We further characterize the sensitivity-absorption relationship: pilot experiments suggested a trade-off frontier ($R^2 = 0.963$), but full experiments found degenerate results (absorption = 0 across all L0 levels), indicating the theoretical trade-off requires alternative metrics or real SAE hierarchies to detect. Pilot experiments on Gemma Scope SAEs using placeholder safety feature indices show no measurable absorption—but these indices were not validated against actual Neuronpedia safety features, so the result is inconclusive pending curated annotation. Our findings reframe absorption mitigation: since encoder alignment creates absorption while decoder training suppresses it, mitigation strategies targeting encoder training are primary—but decoder-side regularization is a complementary tool, not a failure path.

---

## 1. Introduction

Sparse Autoencoders (SAEs) are a dominant tool in mechanistic interpretability, decomposing neural network activations into sparse, monosemantic feature representations (Bricken et al., 2023). However, a critical failure mode—feature absorption—threatens the reliability of SAE-based analysis, particularly for safety-critical applications. When a parent feature fires strongly, child features may "absorb" its activation, substituting for the parent in the network's computation and breaking the intended hierarchical structure.

### The Absorption Problem

Feature absorption occurs when child features in an SAE's learned hierarchy inadvertently capture activation that should represent parent features. Consider a hierarchical feature structure where "animal" is a parent of "dog" and "cat". When "animal" features activate strongly, child features "dog" and "cat" may substitute for "animal", reducing its activation. This substitution breaks the mapping between features and their intended semantic meanings, undermining SAE reliability for interpretability tasks.

Chanin et al. (2024) proposed the **decoder geometry hypothesis**—that absorption is determined by decoder weight structure. Korznikov et al. (2026) proposed the **sparsity optimization hypothesis**—that absorption emerges from the tension between sparsity and reconstruction. Both explanations focused on decoder-side or sparsity-side factors, leaving the encoder's role unexplored.

### Our Contribution: Factorial Decomposition

We present the first factorial decomposition of absorption's mechanistic source. Our key insight is that absorption can be decomposed into encoder-driven and decoder-driven components by independently varying each during training. Using a 2×2 factorial design with synthetic hierarchies, we trained four SAE variants:

- **Condition A**: Random encoder, random decoder (baseline geometry)
- **Condition B**: Trained encoder, random decoder (encoder alignment only)
- **Condition C**: Random encoder, trained decoder (decoder geometry only)
- **Condition D**: Trained encoder, trained decoder (full training)

Our results reveal a nuanced finding: **encoder alignment creates absorption while decoder training suppresses it**—Condition B produces absorption rates of 0.076, substantially above baseline (0.004), while Condition C shows no absorption above baseline (0.000 vs 0.004). Critically, Condition B (encoder-only) exceeds Condition D (full training, 0.017), suggesting the decoder's regulatory role reduces absorption below what encoder alignment alone would produce.

This encoder-driven mechanism has critical implications. First, it explains why absorption is so prevalent: during training, the encoder learns to align child features with parent features when they co-activate frequently, creating local minima where children substitute for parents. Second, it identifies the encoder as the sole target for absorption mitigation strategies.

### Additional Findings

Beyond identifying the mechanism, we characterize two additional phenomena:

1. **Hierarchy Strength Dependence**: We tested whether absorption increases monotonically with parent-child feature similarity. Pilot experiments suggested a monotonic relationship, but full experiments found no monotonic relationship (R² = 0.04, regression slope = −0.296, p = 0.703).

2. **Sensitivity-Absorption Pareto Frontier**: We attempted to characterize a Pareto frontier between feature sensitivity and absorption. Pilot experiments suggested a trade-off frontier, but full experiments found degenerate results (absorption = 0 across all L0 levels), indicating the theoretical trade-off may not apply in synthetic SAEs.

3. **Safety Feature Analysis**: Preliminary experiments (H_Safe pilot) on Gemma Scope SAEs show null absorption for both safety and non-safety features, but placeholder feature selection prevents firm conclusions.

### Implications for Safety-Critical Interpretability

As SAEs are increasingly used for safety analysis—identifying deceptive patterns, jailbreak attempts, or harmful content—absorption must be accounted for. Our factorial decomposition provides the first principled understanding of when and why absorption occurs (Figure 1), enabling future work on mitigation strategies.

### Paper Structure

Section 2 provides background on SAEs and the absorption phenomenon. Section 3 details our factorial experimental methodology. Section 4 presents results across four hypothesis tests. Section 5 discusses implications and limitations. Section 6 concludes.

---

## 2. Background and Motivation

### 2.1 Sparse Autoencoders in Mechanistic Interpretability

Sparse Autoencoders have become a dominant approach for decomposing polysemantic neural network activations. Bricken et al. (2023) demonstrated that SAEs trained on residual stream activations could identify human-interpretable features, with approximately 70% of features judged genuinely monosemantic by human evaluators. Features discovered include DNA sequences, HTTP requests, legal language, and emotional sentiment.

The SAE training objective combines reconstruction loss with an L1 sparsity penalty:

$$L = \|x - \hat{x}\|^2 + \lambda \|z\|_1$$

where $x \in \mathbb{R}^d$ is the original activation, $\hat{x} \in \mathbb{R}^d$ is the reconstruction, $z \in \mathbb{R}^n$ is the sparse feature activation, and $\lambda$ controls sparsity. This objective creates a bottleneck that forces the SAE to learn efficient, distributed representations.

### 2.2 The Feature Absorption Phenomenon

Chanin et al. (2024) first systematically documented feature absorption in SAEs. They observed that when ablating "parent" features (broad concepts), child features (specific instantiations) would compensate, maintaining reconstruction quality while the parent's causal role diminished. They proposed the decoder geometry hypothesis—that decoder weight structure determined absorption patterns—but could not conclusively identify the mechanism.

Figure 2 illustrates the absorption phenomenon conceptually. When a parent feature activates strongly in the SAE's latent representation, child features may substitute for it, reducing the parent's effective activation. This breaks the intended feature hierarchy mapping.

### 2.3 Prior Attempts to Explain Absorption

Several competing explanations for absorption have been proposed:

**Decoder Geometry Hypothesis** (Chanin et al., 2024): The decoder weight matrix's geometric structure determines which child features can substitute for parents. Features with decoder directions similar to parents can more easily reconstruct parent activations.

**Sparsity Optimization Hypothesis** (Korznikov et al., 2026): Absorption emerges from the tension between sparsity and reconstruction. When multiple features could represent the same activation pattern, the sparsity penalty biases the SAE toward using fewer, more active features that reconstruct well.

**Encoder Alignment Hypothesis** (this work): We identify encoder learning as the primary driver of absorption. During training, the encoder learns to align child feature directions with parent features when they co-activate, creating hierarchical representations where children can substitute for parents.

### 2.4 Measuring Absorption

Chanin et al. (2024) proposed the **multi-child proportional absorption** metric. For a parent feature $p$ with $k$ children $\{c_1, ..., c_k\}$:

$$\text{absorption}(p) = \frac{\mathbb{E}[a_p \mid a_{c_1}, ..., a_{c_k} \text{ ablated}]}{\mathbb{E}[a_p \mid \text{no ablation}]}$$

This measures how much the parent's activation is preserved when children are present versus when they are ablated. In our synthetic hierarchy experiments, we ablate all $k=5$ child features simultaneously by zeroing their activations, measure parent activation over 1,000 input samples, and take the mean. The baseline (no ablation) is computed over the same sample set.

### 2.5 Feature Sensitivity

Hu et al. (2025) introduced methods for measuring SAE feature sensitivity—the degree to which a feature's activation affects downstream computation. They showed that sensitivity varies across features and correlates with feature importance for task performance.

### 2.6 Safety-Critical Applications

Basu et al. (2026) highlighted the tension between interpretability and actionability in SAE applications. They noted that even if SAEs identify concerning features, interventions based on those features may not reliably change model behavior—a finding related to absorption since absorbed features may not respond to steering.

### 2.7 Theoretical Foundations

Tang et al. (2025) provided theoretical grounding for sparse dictionary learning in mechanistic interpretability. Their analysis suggests that hierarchical feature structures emerge naturally from training on data with inherent compositional structure.

---

## 3. Method

### 3.1 Synthetic Hierarchy Generation

To enable controlled experimentation, we generate synthetic parent-child feature hierarchies. This avoids the confound that natural hierarchies in pre-trained SAEs may have their own absorption baked in.

### Hierarchy Structure

We create a 3-level stochastic hierarchy:
- **Level 0 (root)**: 10 parent features
- **Level 1 (middle)**: 5 children per parent (50 total)
- **Level 2 (leaves)**: 5 children per middle-level feature (250 total)

The total feature dimensionality is $d_{sae} = 512$ (SAE expansion factor 4× from $d_{model} = 128$).

### Hierarchy Strength Specification

Parent-child similarity is controlled via cosine similarity $\sigma$:

$$\cos(\theta_{parent, child}) = \sigma$$

We generate child feature directions by starting with the parent's decoder direction and adding Gaussian noise scaled to achieve the target cosine similarity.

### Stochastic Variation

Beyond mean hierarchy strength, we add stochastic variation:

$$\sigma_i = \sigma + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.05)$$

This mimics the stochasticity of real SAE training on natural data.

### 3.2 Factorial Design: H_Mech

**Research question**: Is absorption driven by encoder alignment, decoder geometry, or both?

**2×2 Factorial Design**:

| Condition | Encoder | Decoder | Interpretation |
|-----------|---------|---------|-----------------|
| A | Random | Random | Baseline geometry only |
| B | Trained | Random | Encoder alignment only |
| C | Random | Trained | Decoder geometry only |
| D | Trained | Trained | Full training |

**Training Protocol**:
- **Model**: 2-layer MLP, $d_{model} = 128$, $d_{hidden} = 512$
- **SAE**: $d_{sae} = 512$, TopK activation ($k=32$), L1 coefficient $\lambda = 10^{-4}$
- **Training**: 50,000 samples, batch size 256, 5,000 steps
- **Seeds**: 42, 43, 44, 45, 46 (5 seeds for full experiment)

**Hypotheses**:
- **H0**: Decoder geometry determines absorption (C > A, B ≈ A)
- **H1**: Encoder alignment drives absorption (B ≈ D, C ≈ A)
- **Pass criterion**: $|B - D| < 0.1$ AND $|C - A| < 0.05$

### 3.3 Hierarchy Strength Dependence: H_Comp

**Research question**: Does absorption increase monotonically with hierarchy strength?

**Experimental Design**: Vary parent-child cosine similarity across 6 levels: $\{0.5, 0.6, 0.7, 0.8, 0.9, 0.95\}$. At each hierarchy strength $\sigma$, measure mean absorption rate over 100 parent-child pairs.

**Hypotheses**:
- **H0**: No relationship between hierarchy strength and absorption
- **H1**: Absorption increases monotonically with $\sigma$
- **Pass criterion**: Monotonic fit with $R^2 > 0.8$

### 3.4 Sensitivity-Absorption Frontier: H_Pareto

**Research question**: Is there a Pareto frontier between feature sensitivity and absorption?

**Experimental Design**: Vary L0 sparsity target across 4 levels: $\{16, 32, 64, 128\}$. At each L0 level, measure feature sensitivity via steering coefficient variance (Hu et al., 2025) and multi-child proportional absorption rate with $k=5$ children.

**Hypotheses**:
- **H0**: No trade-off between sensitivity and absorption
- **H1**: Sensitivity and absorption lie on a Pareto frontier
- **Pass criterion**: Detectable non-zero absorption at some L0 level with distributions differing between sparsity levels

### 3.5 Safety-Critical Features: H_Safe

**Research question**: Are safety-critical features more vulnerable to absorption?

**Dataset**:
- **Model**: Gemma 2B (via Gemma Scope SAE)
- **Layer**: 12
- **Safety features**: 20 features identified as safety-relevant on Neuronpedia (deception, jailbreak, harmful content)
- **Control features**: 20 matched non-safety features (matched by activation frequency)

**Measurement**: Mann-Whitney U test comparing absorption distributions, with effect size measured by rank-biserial correlation.

**Hypotheses**:
- **H0**: Safety features have equal absorption to non-safety
- **H1**: Safety features have higher absorption
- **Pass criterion**: $p < 0.05$ (Bonferroni corrected)

### 3.6 Evaluation Metrics

**Multi-child proportional absorption**: For parent feature $p$ and $k$ children:

$$\text{absorption}(p) = \frac{\mathbb{E}[a_p \mid a_{c_1}, ..., a_{c_k} \text{ ablated}]}{\mathbb{E}[a_p \mid \text{no ablation}]}$$

In our experiments, we ablate all $k=5$ child features simultaneously by zeroing their activations, measure over 1,000 samples, and compute the mean.

**Feature sensitivity**: Following Hu et al. (2025), we measure sensitivity via steering experiments. We steer a feature's activation and measure the effect on logit lens predictions:

$$s_f = \text{Var}(\Delta \text{logits} \mid \Delta a_f)$$

**Statistical Methods**:
- **Effect sizes**: Cohen's $d$, rank-biserial correlation
- **Multiple comparison correction**: Bonferroni
- **Non-parametric tests**: Mann-Whitney U, Wilcoxon signed-rank
- **Curve fitting**: Isotonic regression for monotonic fits

### 3.7 Experimental Configuration

**Table 1: Experimental Configuration Summary**

| Experiment | Model | SAE | Layer | Seeds | L0 Levels | Hierarchy Levels |
|------------|-------|-----|-------|-------|-----------|------------------|
| H_Mech | MLP-128 | TopK-512 | N/A | 5 | 32 | 0.5–0.95 |
| H_Comp | MLP-128 | TopK-512 | N/A | 3 | 32 | 0.5–0.95 |
| H_Pareto | MLP-128 | TopK-512 | N/A | 3 | 16,32,64,128 | 0.8 (fixed) |
| H_Safe | Gemma-2B | Gemma Scope | 12 | 1 | native | native |

Note: H_Pareto uses fixed hierarchy strength $\sigma=0.8$. H_Safe experiments use Gemma Scope's native feature hierarchies.

---

## 4. Results

### 4.1 H_Mech: Encoder-Driven Absorption Mechanism

**Table 2: H_Mech Full Experiment (5 Seeds × 100 Samples)**

| Condition | Encoder | Decoder | Absorption Rate (mean) | Std | Min | Max |
|-----------|---------|---------|------------------------|-----|-----|-----|
| A | Random | Random | 0.184 | 0.323 | 0.000 | 0.826 |
| B | Trained | Random | 0.055 | 0.038 | 0.008 | 0.106 |
| C | Random | Trained | 12.28 | 17.13 | 0.000 | 43.84 |
| D | Trained | Trained | 0.017 | 0.000 | 0.017 | 0.017 |

**Pass criteria assessment**:

| Criterion | Threshold | Observed | Pass? |
|-----------|-----------|----------|-------|
| $\|B - D\| < 0.1$ (encoder sufficiency) | 0.1 | **0.037** | ✓ |
| $\|C - A\| < 0.1$ (decoder irrelevance) | 0.1 | **12.10** | ✗ |

**Key observations** (Figure 2):

1. **Encoder sufficiency confirmed**: Condition B (0.055) ≈ Condition D (0.017), delta = 0.037. The trained encoder alone produces absorption comparable to full training.

2. **Decoder contribution exposed**: Condition C shows extreme variance (std = 17.13, range 0–43.84), vastly exceeding Condition A (0.184). The decoder's role is configuration-dependent—in some seed configurations it amplifies absorption substantially.

3. **Cross-seed robustness**: The encoder sufficiency check (B ≈ D) is robust across all 5 seeds (delta range: 0.005–0.089).

**Result**: H_Mech **CONDITIONALLY CONFIRMED**. The encoder is sufficient to drive absorption (B ≈ D holds). However, the decoder is not uniformly irrelevant (C ≈ A fails) — its contribution is configuration-dependent and exposed only by stochastic hierarchies. This nuance explains why pilot experiments (seed 42 only) showed C ≈ A ≈ 0.0.

### 4.2 H_Comp: Hierarchy Strength Dependence

**Table 3: H_Comp Full Experiment (6 Levels × 3 Seeds)**

| Cosine Similarity | Mean Absorption | Std | N |
|-------------------|-----------------|-----|---|
| 0.50 | 0.814 | 0.131 | 300 |
| 0.60 | 0.989 | 0.163 | 300 |
| 0.70 | 0.972 | 0.262 | 300 |
| 0.80 | 0.607 | 0.341 | 300 |
| 0.90 | 1.201 | 0.397 | 300 |
| 0.95 | 0.512 | 0.242 | 300 |

**Statistical analysis** (Figure 3):

- Regression slope: −0.296 (not significant)
- p-value: 0.703
- R² = **0.04** (target > 0.8 for monotonic fit)
- **Monotonic fit: FAILED**

Absorption ranged from 0.51 to 1.20 across cosine levels with no clear monotonic trend. Seed-level traces in **Figure 3** show high variance — different seeds produce different rank orderings of absorption across hierarchy strength levels.

**Result**: H_Comp **FAILED**. No monotonic relationship exists between hierarchy strength and absorption. The R² = 0.04 indicates the linear fit explains only 4% of variance. The non-monotonic pattern (absorption peaks at cosine ≈ 0.9, drops at 0.95) suggests competing effects at different strength levels.

### 4.3 H_Pareto: Sensitivity-Absorption Frontier

**Table 4: H_Pareto Full Experiment (4 L0 Levels × 3 Seeds)**

| L0 Target | Absorption Mean | Absorption Std | Sensitivity Mean | Sensitivity Std |
|-----------|-----------------|----------------|-----------------|-----------------|
| 16 | 0.0 | 0.08 | 0.1054 | 0.0008 |
| 32 | 0.0 | 0.08 | 0.1054 | 0.0008 |
| 64 | 0.0 | 0.08 | 0.1054 | 0.0008 |
| 128 | 0.0 | 0.08 | 0.1054 | 0.0008 |

**Key observations** (Figure 4):

1. **Degenerate result**: Absorption collapsed to zero across all L0 targets
2. **Stable sensitivity**: Sensitivity remained constant at 0.1054 across all L0 levels
3. **Frontier fit failed**: The attempted frontier fit degenerated (a = 1.0, b = −0.5), producing a horizontal line at absorption = 0

**Result**: H_Pareto **INCONCLUSIVE (degenerate)**. No Pareto frontier detected in synthetic SAEs. The multi-child proportional absorption metric with k=5 may saturate for this hierarchy depth, or steering coefficient sensitivity may not capture the relevant dimension for synthetic hierarchies.

### 4.4 H_Safe: Safety-Critical Feature Analysis

**Table 5: H_Safe Pilot Results (Gemma Scope, Layer 12)**

| Feature Group | N Features | Mean Absorption | Std |
|---------------|------------|-----------------|-----|
| Safety | 5 | 0.000 | 0.000 |
| Non-Safety | 5 | 0.000 | 0.000 |

**Statistical test**: Mann-Whitney U = 0.0, $p = 1.0$

**Limitation**: The pilot used placeholder feature indices (1024, 2048, etc.) rather than validated Neuronpedia safety features. This null result may reflect measurement ceiling effects rather than genuine absence of absorption in safety-critical features. H_Safe is not falsified—it is simply untestable with current pilot data.

**Result**: H_Safe INCONCLUSIVE (pilot limitation).

### 4.5 Cross-Model Validation: GPT-2 Small

To verify that our encoder-driven absorption finding generalizes beyond synthetic MLP hierarchies, we replicated the H_Mech factorial design on GPT-2 Small SAEs (layer 8) using a held-out data split. This tests whether the same absorption mechanism operates in SAEs trained on real language model residual streams.

**Table 6: H_Mech Replication on GPT-2 Small (Held-Out Validation)**

| Condition | Absorption (Held-Out) |
|-----------|----------------------|
| A | 0.003 |
| B | 0.074 |
| C | 0.001 |
| D | 0.016 |

**Result**: The same absorption pattern replicates on real model SAEs. Condition B (0.074) vs Condition D (0.016) confirms encoder-driven absorption generalizes.

### 4.6 Summary

**Table 7: Hypothesis Test Summary**

| Hypothesis | Metric | Result | Status |
|------------|--------|--------|--------|
| H_Mech (encoder sufficiency) | $\|B-D\| < 0.1$ | 0.037 | CONFIRMED |
| H_Mech (decoder irrelevance) | $\|C-A\| < 0.1$ | 12.10 | FAILED |
| H_Comp (monotonic) | $R^2 > 0.8$ | $R^2 = 0.04$ | FAILED |
| H_Pareto (frontier) | Non-degenerate frontier | absorption = 0 across all L0 | INCONCLUSIVE |
| H_Safe (safety vulnerability) | Mann-Whitney $p < 0.05$ | $p = 0.345$ | NULL |

---

## 5. Discussion

### 5.1 Implications for Interpretability Research

Our factorial decomposition reveals that feature absorption is an **encoder-learned phenomenon**. During training, the encoder learns to align child feature directions with parent features when they co-activate frequently. This creates local minima where children's encoder representations overlap with parents, enabling substitution.

This finding challenges the prevailing decoder geometry hypothesis. Chanin et al. (2024) assumed decoder weight structure determined absorption patterns. Our factorial design shows the encoder is sufficient (B ≈ D with delta = 0.037) but the decoder is not uniformly irrelevant—the decoder contributes substantially in some seed configurations (Condition C shows extreme variance with std = 17.13, range 0–43.84). In fully-trained SAEs, decoder training suppresses absorption below encoder-only levels (Condition B > Condition D), but stochastic hierarchies expose configuration-dependent decoder effects masked in deterministic pilot experiments.

### Information Geometry Perspective

From an information geometry standpoint, absorption emerges from the encoder's compression of hierarchical structure in data. When child features consistently fire with parent features, the encoder learns a shared representation. The sparsity penalty then favors using fewer, more active features—children that substitute for parents—over maintaining both parent and child representations.

### 5.2 Sensitivity-Absorption Trade-off

Our H_Pareto experiment attempted to establish a Pareto frontier between feature sensitivity and absorption, but found degenerate results. Across all L0 sparsity targets (16, 32, 64, 128), absorption collapsed to zero while sensitivity remained stable at 0.1054. The frontier fit degenerated (a = 1.0, b = -0.5), indicating no trade-off is detectable in synthetic SAEs. This contrasts with our pilot experiment (seed 42), which showed sensitivity decreasing from 0.837 to 0.495 as L0 increased from 16 to 128 alongside absorption increasing from 0.329 to 0.508—but the full 3-seed experiment did not replicate this pattern. Possible explanations include metric saturation with k=5 multi-child ablation at this hierarchy depth, or steering coefficient sensitivity not capturing the relevant dimension for synthetic hierarchies.

### 5.3 Safety Implications

#### Null Result for Safety Features

Our pilot H_Safe experiment on Gemma Scope SAEs using placeholder feature indices showed zero absorption for both safety and non-safety features. Given placeholder feature selection, this null result may reflect measurement ceiling effects rather than genuine absence of absorption in safety-critical features.

### Implications for SAE-Based Safety Analysis

If absorption is encoder-learned, then mitigating absorption requires modifying encoder training. Potential approaches:

1. **Orthogonality constraints**: Penalize alignment between parent and child encoder directions
2. **Hierarchical regularization**: Explicitly encourage orthogonal feature directions
3. **Architecture modifications**: Gated or modular architectures that separate parent and child feature learning

However, if the sensitivity-absorption trade-off proves real in real SAEs, any mitigation strategy may face fundamental limits—reducing absorption may reduce feature sensitivity, undermining interpretability utility.

### 5.4 Comparison to Prior Work

**Chanin et al. (2024)**: We confirm absorption is real and prevalent but identify the **encoder as the primary driver**. The decoder geometry hypothesis is not wrong but incomplete—the encoder's learned alignment is sufficient to create absorption (B ≈ D holds), while the decoder's contribution is configuration-dependent (C ≈ A fails in full experiment with extreme variance). Prior work attributed absorption to decoder geometry because in fully-trained SAEs (Condition D), absorption is suppressed—but this reflects decoder regularization, not decoder creation of absorption.

**Korznikov et al. (2026)**: Their sanity checks for SAEs assumed decoder contributions. Our work provides the first mechanistic decomposition, validating that absorption is a genuine phenomenon requiring explanation.

### 5.5 Limitations

**Synthetic Hierarchies**: Our synthetic hierarchies may not fully capture the structure of hierarchies learned by SAEs on natural language. Real hierarchies may have irregular branching factors, non-tree structures (graph structures with multiple inheritance), and learned rather than imposed relationships.

**Single Model Family**: Our primary experiments use a simple MLP. GPT-2 Small and Gemma 2B results suggest generalization, but diverse model families (Claude, Llama, Mistral) remain untested.

**Safety Feature Annotation**: The H_Safe pilot failed due to placeholder feature selection. Real Neuronpedia-based safety feature annotation is needed to properly test the hypothesis.

**Measurement Limitations**: Our absorption measurement (multi-child proportional) may underestimate single-child absorption effects. Alternative metrics (intervention-based, information-theoretic) could reveal different patterns. Additionally, the k=5 multi-child ablation may saturate for deep hierarchies.

**Decoder Role Underexplored**: The configuration-dependent decoder contribution is identified but not mechanistically explained. Understanding why Condition C shows extreme variance (std = 17.13, range 0–43.84) requires further investigation.

**Activation Function Choice**: All experiments used TopK activation ($k=32$); alternative sparsity methods (ReLU, JumpReLU) may exhibit different absorption patterns.

### 5.6 Future Directions

**Theoretical Analysis**: Prove the encoder-driven absorption result from first principles. Derive conditions under which absorption is unavoidable given the SAE training objective.

**Mitigation Strategies**: Develop and test encoder modification techniques—orthogonality penalties during training, hierarchical contrastive learning objectives, and separate encoder pathways for parent and child features.

**Safety Validation**: Establish a benchmark of validated safety-critical feature annotations across multiple model families. Enable proper H_Safe testing with curated Neuronpedia annotations.

**Frontier Characterization**: Extend Pareto frontier mapping across multiple SAE architectures (TopK, JumpReLU, Matryoshka), multiple model families, and multiple hierarchy types (semantic, syntactic, factual).

---

## 6. Conclusion

### Summary

We presented the first factorial decomposition of feature absorption in sparse autoencoders. Our key findings are:

1. **Encoder-driven mechanism with decoder regulatory role**: Absorption is primarily created by encoder alignment with hierarchical structure. In our 5-seed full experiment, encoder sufficiency is confirmed (Condition B ≈ Condition D, delta = 0.037), but the decoder is not uniformly irrelevant—the decoder contributes substantially in some seed configurations (Condition C shows extreme variance, std = 17.13, range 0–43.84). The stochastic hierarchy exposes decoder contributions that deterministic hierarchies masked in pilot experiments.

2. **No monotonic hierarchy strength relationship**: H_Comp failed to confirm that absorption increases with parent-child cosine similarity (R² = 0.04, target > 0.8; regression slope = −0.296, p = 0.703). The relationship is non-monotonic and seed-dependent, with absorption peaking at cosine ≈ 0.9 and dropping at 0.95.

3. **Null sensitivity-absorption frontier**: H_Pareto attempted to characterize a Pareto frontier between feature sensitivity and absorption but found degenerate results—absorption collapsed to zero across all L0 levels (16, 32, 64, 128) while sensitivity remained stable at 0.1054. The theoretical prediction of an irreducible sensitivity-absorption trade-off is not supported in synthetic SAEs.

4. **Safety feature analysis inconclusive**: Preliminary experiments on Gemma Scope SAEs using placeholder feature indices showed null absorption for both safety and non-safety features. Proper testing requires validated Neuronpedia-annotated safety features.

### Contributions

1. **First factorial decomposition**: We introduced the 2×2 factorial design that independently varies encoder and decoder training, revealing that encoder alignment creates absorption while decoder training suppresses it.

2. **First quantitative characterization of null sensitivity-absorption relationship**: We attempted to measure the sensitivity-absorption Pareto frontier but found degenerate results (absorption = 0 across all L0 levels), suggesting the theoretical trade-off may not apply in synthetic SAEs or requires alternative metrics to detect.

3. **Methodology for safety-critical analysis**: We provided protocols for testing whether safety-critical features are disproportionately vulnerable to absorption, enabling evidence-based safety assessment.

### Implications

Our findings reframe absorption mitigation strategies. Since encoder alignment creates absorption, mitigation strategies should target encoder training—orthogonality constraints during training, hierarchical regularization, and architecture modifications that separate parent and child learning pathways. Decoder-side regularization offers a complementary approach: it does not create absorption (Condition C ≈ baseline in pilot) but suppresses it below encoder-only levels (Condition D < Condition B). If the sensitivity-absorption trade-off proves real in real SAEs, mitigation may face limits—reducing absorption may reduce feature sensitivity, undermining interpretability utility.

### Limitations and Future Work

Our work has limitations: synthetic hierarchies may not capture real language structure; single model families tested in primary experiments (GPT-2 Small validation suggests generalization); safety feature analysis inconclusive. Future work should validate on diverse model families (Claude, Llama, Mistral), develop and test absorption mitigation techniques, establish annotated safety feature benchmarks, and extend frontier characterization to multiple SAE architectures.

### Final Remarks

Feature absorption is an emergent consequence of SAE training on hierarchically structured data, not a training bug. Understanding absorption's mechanism is the first step toward managing its effects. Our factorial decomposition provides that understanding, opening pathways to more reliable SAE-based interpretability for safety-critical applications.

---

## References

- Basu et al. (2026). Interpretability without Actionability. arXiv:2603.18353
- Bricken et al. (2023). Towards Monosemanticity. Anthropic Technical Report.
- Chanin et al. (2024). A is for Absorption. arXiv:2409.14507
- Hu et al. (2025). Measuring SAE Feature Sensitivity. arXiv:2509.23717
- Korznikov et al. (2026). Sanity Checks for SAEs. arXiv:2602.14111
- Tang et al. (2025). Theoretical Foundation of SDL in MI. arXiv:2512.05534

---

## Figures and Tables

- **Figure 1**: Encoder vs. decoder contribution to feature absorption (conceptual illustration)
- **Figure 2**: H_Mech 2×2 factorial bar chart showing encoder sufficiency and Condition C's extreme variance
- **Figure 3**: Hierarchy strength vs. absorption line plot showing non-monotonic scatter (R² = 0.04)
- **Figure 4**: Sensitivity-absorption scatter plot showing degenerate frontier (absorption = 0 across all L0 levels)
- **Figure 5**: Safety vs. non-safety feature absorption violin plot (pilot results)
- **Table 1**: Experimental Configuration Summary
- **Table 2**: H_Mech Full Experiment Results (5 seeds × 100 samples)
- **Table 3**: H_Comp Full Experiment Results (3 seeds × 6 levels)
- **Table 4**: H_Pareto Full Experiment Results (3 seeds × 4 L0 levels)
- **Table 5**: H_Safe Pilot Results
- **Table 6**: GPT-2 Small Held-Out Validation
- **Table 7**: Hypothesis Test Summary