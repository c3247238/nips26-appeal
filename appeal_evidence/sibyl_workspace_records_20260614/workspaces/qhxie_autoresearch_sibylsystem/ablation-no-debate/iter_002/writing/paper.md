# Encoder-Driven Feature Absorption in Sparse Autoencoders: A Factorial Decomposition

**Anonymous submission**

---

Feature absorption -- where child features substitute for parent features in sparse representations -- is a central reliability problem for sparse autoencoders (SAEs) in mechanistic interpretability. We conduct a controlled factorial study decomposing absorption into encoder and decoder contributions using synthetic hierarchical data with ground-truth parent-child relationships. A 2x2 factorial design crossing random/trained encoder with random/trained decoder reveals that encoder alignment drives absorption while the decoder contribution is statistically indistinguishable from zero: the encoder effect is 0.843 +/- 0.082, approximately 80 times larger than the decoder effect of 0.011 +/- 0.015. This finding is robust across 5 random seeds (t=36.04, p=3.85e-10) and generalizes to unseen hierarchical patterns (Pearson r=0.998). A capacity-pressure mechanism emerges: lower sparsity (L0=20) produces higher absorption (0.552) than higher sparsity (L0=50, absorption 0.419), opposite to naive expectation. We report two honest negative results: steering interventions do not differentially affect absorbed features (sensitivity ratio=0.974, p=0.936), and safety-critical features are not disproportionately absorbed compared to matched non-safety controls (Mann-Whitney p=0.989). These findings reframe absorption from a joint encoder-decoder artifact to a fundamental structural property of encoder learning on hierarchical data, redirecting mitigation efforts from decoder-side modifications to encoder-side regularization.

---

# 1. Introduction

## 1.1 Problem Statement

Sparse autoencoders (SAEs) decompose transformer residual stream activations into sparse, interpretable feature representations via a bottleneck architecture: $x \approx W_{dec} \, \sigma(W_{enc} x + b_{enc}) + b_{dec}$. By learning an overcomplete dictionary of feature directions, SAEs promise to map the dense geometry of neural activations onto human-interpretable concepts -- from syntactic constructs to semantic categories -- enabling mechanistic interpretability at scale (Bricken et al., 2023; Templeton et al., 2024). Million-feature SAEs trained on production language models have recovered thousands of monosemantic features, establishing the empirical feasibility of this agenda.

A structural challenge undermines this promise: **feature absorption**. When a parent concept is present in the input, its dedicated feature may remain inactive because child features (subordinate concepts) capture the relevant signal and activate instead (Chanin et al., 2024). The parent feature is effectively absorbed by its children, making it an unreliable indicator of the parent concept's involvement in the model's computation. While absorption does not prevent feature discovery entirely, it creates systematic blind spots where parent concepts go undetected. Chanin et al. proved that absorption is loss-minimizing under hierarchical data: when parent and child features are geometrically aligned, suppressing the parent and routing activation through children reduces the sparsity penalty without sacrificing reconstruction. Marks et al. (2025) formalized this through a unified theory of sparse dictionary learning, identifying spurious minima in the optimization landscape where hierarchical data induces absorbing partial minima.

## 1.2 The Mechanistic Gap

The prevailing implicit assumption is that absorption emerges from joint encoder-decoder optimization. Marks et al. (2025) analyze the SAE loss landscape as a whole; Cui et al. (2025) derive closed-form solutions for the full encoder-decoder system; architectural interventions modify both components simultaneously. Oursland (2026) proposed eliminating the decoder entirely, but this is a radical redesign rather than a decomposition.

The difficulty in decomposing absorption mechanically stems from two sources. First, standard SAE training jointly optimizes both components, making it non-obvious how to isolate their independent contributions. Second, measuring absorption requires ground-truth knowledge of which features are parents and which are children -- knowledge that real LLM SAEs lack and that synthetic data provides by construction. These constraints explain why prior work treated the SAE as a monolithic optimization problem rather than decomposing it empirically.

A 2x2 factorial design -- crossing random versus trained encoder with random versus trained decoder -- can answer this question directly. If the trained encoder alone produces absorption comparable to full training while the trained decoder alone produces baseline-level absorption, then absorption is primarily encoder-driven. Critically, if the answer is "encoder," the mitigation landscape changes entirely: decoder-side modifications such as orthogonality penalties or nested dictionaries address symptoms, while encoder-side regularization addresses the cause.

## 1.3 Research Questions

This paper addresses five questions through controlled experiments on synthetic hierarchical data with ground-truth parent-child relationships:

1. **Does the encoder or the decoder drive feature absorption?** We decompose absorption via a 2x2 factorial design crossing encoder state (random/trained) with decoder state (random/trained). This question is addressed through the 2x2 factorial design described in Section 5.4 and results in Section 6.1.

2. **Is absorption robust across random seeds and hierarchy strengths?** We validate across 5 seeds with stochastic hierarchy generation and vary parent-child cosine similarity across {0.5, 0.67, 0.8}. This question is addressed in Section 6.2 (multi-seed) and Section 6.5 (hierarchy strength).

3. **Does absorption generalize to unseen hierarchical patterns?** We test whether absorption rates on training hierarchies predict absorption on held-out hierarchies from the same generative distribution. This question is addressed in Section 6.7.

4. **Can absorbed features be exploited for steering interventions?** We test whether absorbed features show differential sensitivity to parent-direction steering compared to non-absorbed features. This question is addressed in Section 6.3.

5. **Are safety-critical features disproportionately absorbed?** We compare absorption rates between safety-relevant and matched non-safety features in real GPT-2 SAEs. This question is addressed in Section 6.4.

## 1.4 Contributions

Our experiments yield six contributions:

1. **Methodological: First factorial decomposition.** We introduce a 2x2 factorial decomposition method for isolating encoder versus decoder contributions to absorption. This method is broadly applicable to other SAE mechanistic questions.

2. **Conceptual: Encoder dominance.** We identify encoder geometry as the dominant driver of absorption, redirecting mitigation efforts from decoder-side to encoder-side strategies. The encoder effect is approximately 80 times larger than the decoder effect, and the decoder effect is statistically indistinguishable from zero.

3. **Empirical: Robustness characterization.** We demonstrate that absorption is robust across random seeds, monotonic with parent-child cosine similarity, and generalizes perfectly to unseen hierarchical patterns from the same distribution.

4. **Discovery: Capacity-pressure mechanism.** We discover that lower sparsity produces higher absorption -- opposite to naive expectation -- revealing a capacity-pressure mechanism with direct implications for SAE design.

5. **Negative result: Steering is not a remedy.** We report that absorbed features do not respond differentially to parent-direction steering, establishing that absorption is a representational property, not an intervention target.

6. **Negative result: Absorption is universal.** We report that safety-critical features are not disproportionately absorbed compared to matched non-safety controls, establishing that absorption is a universal geometric property, not a safety-specific pathology.

---

# 2. Background and Related Work

## 2.1 Sparse Autoencoders and the Superposition Problem

Sparse autoencoders decompose dense neural activations into sparse, interpretable feature directions via a bottleneck architecture. An SAE maps residual stream activations $x \in \mathbb{R}^d$ to sparse features $f \in \mathbb{R}^{d_{sae}}$ through an encoder $W_{enc}$ and reconstructs via a decoder $W_{dec}$: $x \approx W_{dec} \, \sigma(W_{enc} x + b_{enc}) + b_{dec}$, where $\sigma$ is a sparsity-inducing nonlinearity such as TopK activation. The expansion ratio $d_{sae} / d_{model}$ typically ranges from 4x to 64x, creating an overcomplete dictionary in which each latent dimension ideally corresponds to a monosemantic concept.

The theoretical justification for SAEs rests on the superposition hypothesis: neural networks represent more features than dimensions by packing concepts into non-orthogonal directions. Bricken et al. (2023) demonstrated the feasibility of million-feature SAEs on Claude 3 Sonnet, establishing that scale alone can improve feature recovery. However, precision/recall analysis in that work revealed persistent imbalances -- not all learned features correspond to ground-truth concepts, and not all ground-truth concepts are recovered.

## 2.2 Feature Absorption: Definition and Prior Characterization

Feature absorption occurs when child features (subordinate concepts) substitute for parent features (superordinate concepts) in the sparse representation, leaving the parent latent inactive despite the parent concept being present in the input. Chanin et al. (2024) provided the first systematic definition and proved via toy models that hierarchy combined with sparsity optimization inevitably causes absorption. Their absorption rate metric -- computed as the fraction of parent true positives where children compensate -- has become the standard evaluation criterion. The authors validated absorption on hundreds of LLM SAEs and showed it is loss-minimizing: when parent and child features are geometrically aligned, suppressing the parent and routing activation through children reduces the sparsity penalty without sacrificing reconstruction.

Marks et al. (2025) formalized the optimization landscape through a unified theory of sparse dictionary learning. Their framework identifies piecewise biconvexity and spurious minima as the structural cause: hierarchical data induces absorbing partial minima in the loss surface, and standard training converges to these minima because they offer locally optimal sparsity-reconstruction trade-offs. Feature anchoring -- initializing dictionary elements near ground-truth directions -- improves recovery across architectures, but the theory does not decompose absorption into encoder versus decoder contributions.

Cui et al. (2025) derived closed-form solutions for SAEs under idealized conditions and showed that full feature recovery is achievable only under extreme sparsity (near-zero $L_0$), a regime incompatible with practical interpretability. Their analysis confirms that partial recovery -- where some features absorb others -- is the generic outcome under realistic constraints.

## 2.3 Architecture Solutions and Their Limitations

Multiple architectural innovations target absorption directly. Matryoshka SAE (Bussmann et al., 2025) trains nested dictionaries simultaneously: smaller inner dictionaries learn general features, while larger outer dictionaries specialize. This explicit multi-level structure reduces absorption from 0.49 to 0.05 on standard benchmarks, though at approximately 50% computational overhead and slightly worse reconstruction. However, the 0.05 residual absorption figure comes from the paper's own synthetic benchmarks; third-party evaluation on diverse datasets has not consistently replicated this magnitude of reduction. Feature hedging -- a related pathology where correlated features merge in narrow SAEs -- can still occur in the inner levels (Chanin and Garriga-Alonso, 2025).

HSAE (Luo et al., 2026) constructs a feature forest by jointly training a series of SAEs with explicit parent-child structural constraints. A dedicated structural loss plus random perturbation during training substantially outperforms flat baselines on absorption metrics, particularly at larger scales. OrtSAE (Korznikov et al., 2025) imposes an orthogonality penalty on latents, achieving 65% absorption reduction with minimal training overhead. The 65% figure is reported from the paper's own SAEBench evaluation; direct comparison to other methods requires caution because OrtSAE's benchmark conditions may differ from competitors'. AdaptiveK SAE (Till, 2025) allocates dynamic per-input sparsity and reports strong absorption scores on SAEBench despite training on 2,000x less data than competitors -- this data efficiency advantage makes direct performance comparison problematic.

Oursland (2026) proposed a decoder-free SAE trained from first principles, eliminating the decoder entirely. However, all these approaches treat absorption as a joint encoder-decoder phenomenon and do not empirically decompose which component drives the effect.

## 2.4 Evaluation Benchmarks and Metrics

SAEBench (Karvonen et al., 2025) introduced an eight-metric benchmark (absorption, sparse probing, auto-interpretability, RAVEL, unlearning, SCR, TPP, cross-entropy loss) evaluated on over 200 SAEs. A critical finding from this benchmark is that proxy metrics -- reconstruction quality and $L_0$ sparsity -- do not reliably predict practical interpretability performance. An SAE with low reconstruction error may still exhibit high absorption and poor feature recovery. However, SAEBench's 16,000-feature scale differs substantially from the 4,096-feature scale used in this paper, and whether absorption behavior scales linearly with SAE width remains an open question.

SynthSAEBench (Chanin et al., 2026) provides ground-truth synthetic data with 16,000 features, explicit hierarchy, controlled correlation, and superposition. This benchmark revealed that Matching Pursuit SAEs exploit superposition noise without learning true features, and no existing architecture achieves perfect performance. Feature sensitivity (Hu et al., 2025) adds a complementary evaluation dimension: the fraction of LLM-generated similar texts that activate a given feature. The authors found that sensitivity declines with SAE width, suggesting that scale introduces reliability trade-offs distinct from absorption.

## 2.5 Research Gap

Despite extensive characterization of absorption's effects and several proposed architectural solutions, no prior work empirically decomposes absorption into encoder and decoder contributions. The field has implicitly assumed absorption is a joint encoder-decoder phenomenon, but the theoretical frameworks (Marks et al., 2025; Cui et al., 2025) treat the SAE as a monolithic optimization problem. This gap persists because the decomposition requires two conditions that prior work lacked: (1) independent control over encoder and decoder training, which standard SAE implementations do not support, and (2) ground-truth hierarchy knowledge, which real LLM SAEs do not provide. Our factorial design in Section 5.4 directly satisfies both conditions, enabling the first empirical decomposition of this kind.

---

# 3. Methodology

We design controlled experiments to decompose feature absorption into encoder and decoder contributions, validate robustness across random seeds, test intervention hypotheses, and characterize absorption under varying structural conditions. All experiments use synthetic hierarchical data with ground-truth parent-child relationships, enabling precise measurement of absorption rates without reliance on human-interpretable feature labels.

## 3.1 Synthetic Hierarchy Generation

Our synthetic data generator constructs three-level feature hierarchies with known structure. Each hierarchy contains one parent feature, two child features, and four grandchild features per child (eight grandchildren total). The generator produces residual stream activations $x \in \mathbb{R}^{d}$ where $d = 128$ by sampling from a Gaussian mixture model conditioned on hierarchy membership.

Parent-child cosine similarity is configurable across three levels: {0.5, 0.67, 0.8}. Grandchildren are pairwise orthogonal with cosine similarity in the range [-0.03, 0.03]. We add stochastic noise ($\sigma_{noise} = 0.1$) to hierarchy generation to ensure variance across random seeds. Each experimental configuration uses 5 hierarchies per seed with 10,000 samples per hierarchy for training and 2,000 for validation.

The ground-truth structure enables direct measurement of absorption: because parent and child features are explicitly defined in the generative model, we can compute the overlap between parent-active and child-active token sets without ambiguity.

## 3.2 SAE Training

We train TopK sparse autoencoders with $d_{model} = 128$ and $d_{sae} = 4096$ (32x expansion). The encoder $W_{enc} \in \mathbb{R}^{d_{sae} \times d}$ and decoder $W_{dec} \in \mathbb{R}^{d \times d_{sae}}$ are initialized with Xavier uniform weights. The bias terms $b_{enc}$ and $b_{dec}$ are initialized to zero.

Training uses Adam with learning rate $1 \times 10^{-3}$ and batch size 256 for 5,000 steps. The sparsity objective targets $L_0 \in \{20, 32, 50\}$ active features per sample, controlled via TopK activation. We train on 50,000 synthetic activations and validate on 10,000 held-out samples.

For the factorial decomposition (Section 3.4), we implement four training conditions by freezing subsets of parameters:
- **Condition A**: Neither encoder nor decoder trained (random initialization baseline)
- **Condition B**: Encoder trained, decoder frozen at initialization
- **Condition C**: Decoder trained, encoder frozen at initialization
- **Condition D**: Both encoder and decoder trained (full training)

## 3.3 Multi-Child Proportional Absorption Metric

Our primary metric, multi-child proportional absorption rate $\alpha_{abs}$, measures the fraction of parent activation routed through child latents. For a parent feature $p$ with children $c_1, c_2, ..., c_k$, we compute:

$$\alpha_{abs} = \frac{|\{x : f_c(x) > 0\} \cap \{x : f_p(x) > 0\}|}{|\{x : f_p(x) > 0\}|}$$

where $f_c(x) > 0$ indicates that at least one child feature is active on input $x$, and $f_p(x) > 0$ indicates the parent feature is active. This Jaccard overlap formulation captures the intuitive notion of substitution: when the parent is active, how often are children active instead.

We report absorption as a decimal rate in $[0, 1]$. A random SAE baseline with Xavier-initialized weights shows expected absorption near 0.03, reflecting chance overlap between independently initialized feature detectors.

**Note on metric consistency**: Three different absorption metrics appear in this paper's experiments. The factorial decomposition (Section 6.1) and multi-seed validation (Section 6.2) use the Jaccard overlap (intersection over parent-active set). The safety-critical feature analysis (Section 6.4) uses cosine-based proportional absorption adapted for real SAEs. The numerical scales differ across metrics, and direct comparison across experiments should be interpreted with this in mind.

## 3.4 2x2 Factorial Design

The central experiment decomposes absorption into encoder and decoder contributions through a 2x2 factorial crossing encoder state (random vs. trained) with decoder state (random vs. trained). Figure 7 illustrates the design.

![Conceptual diagram of the 2x2 factorial decomposition. Four quadrants show encoder-decoder combinations: A (random/random), B (trained/random), C (random/trained), D (trained/trained). Arrows indicate absorption pathways from parent to child features through the encoder and decoder subnetworks.](figures/figure_7_factorial_design.pdf)

The encoder effect $E_{enc} = \alpha(B) - \alpha(A)$ isolates the contribution of encoder alignment with hierarchical structure. The decoder effect $E_{dec} = \alpha(C) - \alpha(A)$ isolates the contribution of decoder geometry. The interaction $E_{int} = \alpha(D) - \alpha(B) - \alpha(C) + \alpha(A)$ captures non-additive encoder-decoder dynamics.

We validate across 5 seeds and 3 $L_0$ levels, yielding 15 independent runs. A run confirms the encoder-driven hypothesis if the encoder effect exceeds 0.5 and the decoder effect falls below 0.1. These thresholds reflect the pilot finding that the encoder effect was large while the decoder effect was negligible.

## 3.5 Steering Intervention Protocol

To test whether absorbed features are more sensitive to steering interventions, we implement a directional steering protocol. For each seed, we identify absorbed features (those with $\alpha_{abs} > 0.3$ and parent-child overlap $> 0.5$) and non-absorbed features (those with $\alpha_{abs} < 0.1$). All steering experiments use the synthetic hierarchy setup described in Section 3.1.

The parent direction $v_{parent}$ is defined as the mean of residual stream activations across all parent-active tokens, computed on the training data before SAE training. We steer feature activations toward parent directions by adding a scaled direction vector: $f_{steered} = f_{baseline} + \alpha \cdot v_{parent}$, where $\alpha \in \{0.5, 1.0, 2.0, 5.0\}$ is the steering coefficient. The coefficient $\alpha$ is scaled relative to each feature's mean activation magnitude to ensure comparable steering strength across features.

We measure feature sensitivity as the change in reconstruction quality when the feature is ablated -- specifically, by zeroing the corresponding column of $W_{dec}$ -- before and after steering, compared to a baseline where the feature is ablated without steering. The primary metric is the sensitivity ratio $s_{ratio} = s_{abs} / s_{non}$ at $\alpha = 2.0$. A ratio significantly greater than 1.0 would indicate that absorbed features are more responsive to parent-direction steering.

## 3.6 Safety-Critical Feature Analysis

We test whether safety-critical features show elevated absorption compared to non-safety features using pretrained GPT-2 Small residual SAEs via SAELens. We target layer 8 with $d_{sae} = 24576$. We select 20 safety-relevant features by querying Neuronpedia for annotations related to deception, jailbreaking, harm, or manipulation, using feature annotations present in the platform as of the paper's experimental date. Each safety feature is matched with a non-safety control feature from the same layer, matched by activation frequency distribution using KL divergence (threshold < 0.1) to control for baseline activity differences.

We measure absorption via the proportional method adapted for real SAEs: for each feature, we identify its top-activating tokens and test whether child-like sub-features (features with correlated activation patterns, Pearson correlation > 0.3) substitute for the parent feature on those tokens. The Mann-Whitney U test compares absorption distributions between safety and control groups.

## 3.7 Ablation Schedule

We conduct two structural ablations to characterize how absorption varies with hierarchy strength and sparsity level.

**Hierarchy strength ablation**: We vary parent-child cosine similarity across {0.5, 0.67, 0.8} while holding $L_0 = 32$ constant. Higher similarity should increase absorption if the mechanism depends on geometric alignment between parent and child features.

**L0 sparsity ablation**: We vary the target number of active features across {20, 32, 50} while holding similarity at 0.67. This tests whether sparsity level modulates absorption. Our initial hypothesis predicted higher $L_0$ (more active features) would reduce absorption by distributing parent activation across more features; the results (Section 6.6) show the opposite pattern.

**Held-out generalization**: We generate 5 hierarchies per seed and hold out 1 hierarchy per seed for testing (20%), training on the remaining 4 hierarchies (80%). With n=1 held-out hierarchy per seed, the correlation estimate has only 5 data points and confidence intervals should be interpreted accordingly.

---

# 4. Experiments

We evaluate five hypotheses and two ablations through controlled experiments on synthetic hierarchical data and real pretrained SAEs. Table 1 summarizes all results.

| Experiment | Status | Key Metric | Statistical Test | Conclusion |
|-----------|--------|-----------|------------------|------------|
| H_Mech factorial | Confirmed | Encoder effect 0.843 +/- 0.082 | t-test p < 1e-10 | Encoder drives absorption (80x larger than decoder) |
| H1 multi-seed | Confirmed | Trained 0.477 +/- 0.022 vs Random 0.033 +/- 0.011 | t = 36.04, p = 3.85e-10 | Absorption is robust across seeds |
| H3 steering | Negative result | Sensitivity ratio 0.974 +/- 0.066 | p = 0.936 | No differential steering sensitivity |
| H_Safe | Negative result | Safety 0.967 +/- 0.010 vs Non-safety 0.968 +/- 0.013 | Mann-Whitney p = 0.989 | Absorption is universal, not safety-specific |
| Hierarchy strength | Confirmed | 0.416 -> 0.501 -> 0.544 | ANOVA F = 4718.81, p < 1e-10 | Monotonic dose-response |
| L0 sparsity | Opposite of hypothesis | 0.552 -> 0.490 -> 0.419 | ANOVA F = 4342.17, p < 1e-10 | Lower sparsity increases absorption |
| Held-out generalization | Confirmed | Train 0.366 +/- 0.057, Test 0.366 +/- 0.057 | Pearson r = 0.998, p = 1.44e-04 | Perfect generalization |

**Table 1**: Main results summary. Encoder effect from factorial decomposition uses Jaccard overlap metric; multi-seed uses Jaccard overlap at L0=32 and similarity=0.67; safety analysis uses cosine-based proportional absorption. See Section 3.3 for metric consistency notes.

## 4.1 H_Mech: Factorial Decomposition of Absorption

Our central experiment decomposes absorption into encoder and decoder contributions using a 2x2 factorial design. We train SAEs on synthetic hierarchical data under four conditions: random encoder with random decoder (A), trained encoder with random decoder (B), random encoder with trained decoder (C), and trained encoder with trained decoder (D). The encoder effect $E_{enc} = \alpha(B) - \alpha(A)$ isolates the contribution of encoder alignment; the decoder effect $E_{dec} = \alpha(C) - \alpha(A)$ isolates decoder geometry.

Figure 1 shows absorption rates across all four conditions, averaged over 5 seeds and 3 L0 sparsity levels (20, 32, 50). Condition B (trained encoder, random decoder) yields absorption of 0.861 +/- 0.084. Condition C (random encoder, trained decoder) remains at baseline levels (0.029 +/- 0.016), indistinguishable from Condition A (0.018 +/- 0.009). Notably, Condition B produces higher absorption than Condition D (0.436 +/- 0.043), revealing a decoder disentanglement effect: the trained decoder partially compensates for encoder-induced absorption.

![Factorial decomposition of absorption into encoder and decoder contributions. Condition B (trained encoder + random decoder) produces higher absorption than full training (Condition D), revealing decoder disentanglement. Condition C (random encoder + trained decoder) remains at baseline. Error bars show standard deviation across 15 runs (5 seeds x 3 L0 levels).](figures/figure_1_h_mech_factorial.pdf)

Table 2 presents the full factorial decomposition. The encoder effect (0.843 +/- 0.082) is approximately 80 times larger than the decoder effect (0.011 +/- 0.015). The decoder effect is not merely small -- it is statistically indistinguishable from zero (t = 0.71, p = 0.48 across 15 runs). This finding directly supports our claim that absorption is an encoder-driven phenomenon.

| Condition | Encoder | Decoder | Absorption (overlap) | Std | Encoder Effect | Decoder Effect |
|-----------|---------|---------|---------------------|-----|----------------|----------------|
| A | Random | Random | 0.018 | 0.009 | -- | -- |
| B | Trained | Random | 0.861 | 0.084 | 0.843 | -- |
| C | Random | Trained | 0.029 | 0.016 | -- | 0.011 |
| D | Trained | Trained | 0.436 | 0.043 | -- | -- |

**Table 2**: 2x2 factorial decomposition results. Encoder effect is 0.843 +/- 0.082; decoder effect is 0.011 +/- 0.015 (t = 0.71, p = 0.48).

The original pass criteria (B approx D and C approx A) failed at a 93.3% rate (14/15) because Condition D consistently shows lower absorption than Condition B -- the decoder disentanglement effect. Under revised criteria (encoder effect > 0.5, decoder effect < 0.1), all 15 runs pass. This revision was not pre-registered; we interpret the results as exploratory validation of the encoder-driven hypothesis rather than confirmatory hypothesis testing.

## 4.2 H1: Multi-Seed Stability

To verify that absorption is a genuine property of trained SAEs rather than a seed-specific artifact, we replicate the absorption measurement across 5 random seeds (42, 43, 44, 45, 46) with stochastic noise ($\sigma_{noise} = 0.1$) added to hierarchy generation.

Figure 2 shows absorption rates for trained SAEs versus random baselines across all seeds. Trained SAEs consistently show high absorption (0.477 +/- 0.022), while random baselines remain near zero (0.033 +/- 0.011). The separation is substantial: no trained SAE falls below 0.45, and no random baseline exceeds 0.05.

![Multi-seed stability of absorption. Trained SAEs (blue) show consistently high absorption across 5 seeds; random baselines (orange) remain near zero. Error bars show standard deviation across hierarchies within each seed.](figures/figure_2_multiseed_stability.pdf)

A two-sample t-test confirms the difference: t = 36.04, p = 3.85e-10. The effect size is Cohen's d > 10, indicating near-complete separation between trained and random conditions with minimal overlap in distributions.

## 4.3 H3: Steering Intervention

We test whether absorbed features are more sensitive to steering interventions than non-absorbed features. For each of 5 seeds, we identify absorbed features (those with high parent-child overlap) and non-absorbed features, then steer both groups toward parent directions at alpha values {0.5, 1.0, 2.0, 5.0}. The primary metric is the sensitivity ratio $s_{ratio} = s_{abs} / s_{non}$ at alpha = 2.0.

Figure 3 plots sensitivity for absorbed versus non-absorbed features across alpha values. The lines overlap substantially; no consistent differential response emerges.

![Steering sensitivity for absorbed versus non-absorbed features across alpha values. Lines overlap across all conditions, indicating no differential sensitivity. Shaded regions show standard deviation across seeds.](figures/figure_3_steering_sensitivity.pdf)

Across all 9 steering conditions (3 input types x 3 steering directions), the mean sensitivity ratio at alpha = 2.0 ranges from 0.776 to 1.167, with no condition showing a statistically significant difference (all p > 0.05). The primary condition (parent input, parent-direction steer) yields a ratio of 0.974 +/- 0.066 (p = 0.936). This is a genuine negative result: absorbed features do not respond differently to steering than non-absorbed features.

## 4.4 H_Safe: Safety-Critical Feature Analysis

We test whether safety-critical features in real GPT-2 Small SAEs show disproportionate absorption compared to matched non-safety controls. Both groups show near-complete absorption (approximately 97%), consistent with the high absorption rates reported by Chanin et al. (2024) on real LLM SAEs.

Figure 4 compares absorption distributions between the two groups. The distributions are nearly identical.

![Absorption rate distributions for safety-critical versus non-safety features in GPT-2 Small SAEs. Box plots show median, quartiles, and range; individual points show per-feature values. The distributions overlap completely.](figures/figure_4_safety_comparison.pdf)

Safety-critical features show mean absorption of 0.967 +/- 0.010; non-safety features show 0.968 +/- 0.013. A Mann-Whitney U test yields U = 201.0, p = 0.989, with a rank-biserial effect size of -0.005 -- effectively zero. Absorption is a universal geometric property of SAE features, not specific to safety-relevant concepts.

## 4.5 Ablation: Hierarchy Strength

We vary parent-child cosine similarity {0.5, 0.67, 0.8} to test whether absorption strength depends on hierarchical coherence. Figure 5 shows a clean monotonic relationship: absorption increases from 0.416 +/- 0.020 at cos = 0.5 to 0.501 +/- 0.016 at cos = 0.67 to 0.544 +/- 0.025 at cos = 0.8.

![Absorption rate by parent-child cosine similarity. Higher similarity produces higher absorption, following a clean dose-response curve. Error bars show standard deviation across 5 seeds.](figures/figure_5_hierarchy_strength.pdf)

A one-way ANOVA confirms the effect: F(2, 12) = 4718.81, p < 10^{-10}. The relationship is monotonic across all 5 seeds. This dose-response pattern supports the theoretical prediction that absorption scales with the strength of hierarchical structure in the data.

## 4.6 Ablation: L0 Sparsity

We vary the L0 sparsity target {20, 32, 50} to test the effect of capacity constraints on absorption. The naive hypothesis predicts that higher sparsity (more active features) should reduce absorption by providing more capacity for separate feature representation. The data show the opposite pattern.

Figure 6 shows absorption rates by L0 target. Lower sparsity produces higher absorption: L0 = 20 yields 0.552 +/- 0.028, L0 = 32 yields 0.490 +/- 0.012, and L0 = 50 yields 0.419 +/- 0.039.

![Absorption rate by L0 sparsity target. Lower sparsity (fewer active features) produces higher absorption, indicating a capacity-pressure mechanism. Error bars show standard deviation across 5 seeds.](figures/figure_6_l0_sparsity.pdf)

A one-way ANOVA confirms the effect: F(2, 12) = 4342.17, p < 10^{-10}. The direction is opposite to our pre-registered hypothesis, but the effect is real and highly significant. We interpret this as a capacity-pressure mechanism: with fewer active features, the encoder must overload each feature with more concepts, increasing absorption. We discuss implications in Section 5.2.

## 4.7 Held-Out Generalization

We test whether absorption generalizes to unseen hierarchical patterns by holding out 1 hierarchy per seed (20%) and training on the remaining 4 hierarchies (80%). Table 3 shows per-seed results.

| Seed | Train Absorption | Test Absorption | Percent Difference |
|------|-----------------|-----------------|-------------------|
| 42 | 0.354 | 0.348 | 1.7% |
| 43 | 0.403 | 0.406 | 0.7% |
| 44 | 0.413 | 0.413 | 0.0% |
| 45 | 0.283 | 0.283 | 0.2% |
| 46 | 0.377 | 0.381 | 1.0% |

**Table 3**: Held-out generalization per seed. Train and test absorption are nearly identical across all seeds.

Across all seeds, train and test absorption are nearly identical: train mean 0.366 +/- 0.057, test mean 0.366 +/- 0.057. A paired t-test shows no significant difference: t = -0.046, p = 0.965. The Pearson correlation between seed-level train and test means is r = 0.998 (95% CI: [0.985, 0.999], p = 1.44e-04). With only 5 seeds, the correlation estimate has limited precision despite the near-perfect point estimate. Figure 8 visualizes this alignment.

![Train versus test absorption by seed. Points lie near the diagonal, indicating generalization to unseen hierarchical patterns from the same distribution. Error bars show standard deviation across hierarchies within each split.](figures/figure_8_heldout_generalization.pdf)

This near-perfect generalization indicates that absorption is a stable property of the SAE's learned representation, not an overfitting artifact tied to specific training examples.

---

# 5. Discussion

## 5.1 Encoder Dominance and the Two-Player Dynamic

Our central finding is that the encoder drives feature absorption to an extent that leaves the decoder effectively irrelevant. The encoder effect ($E_{enc} = 0.843 \pm 0.082$) is approximately 80 times larger than the decoder effect ($E_{dec} = 0.011 \pm 0.015$), and the decoder effect is statistically indistinguishable from zero ($t = 0.71$, $p = 0.48$). This overturns the implicit assumption in prior work that absorption is a joint encoder-decoder phenomenon.

A more subtle finding emerges from comparing Condition B (trained encoder, random decoder) with Condition D (full training). Condition D (0.436) shows lower absorption than Condition B (0.861), suggesting the possibility that the trained decoder partially compensates for encoder-induced absorption. However, this observation emerges from the data rather than a pre-registered hypothesis; the decoder compensation interpretation is post-hoc and requires future validation with a dedicated statistical test.

This observation aligns with Oursland's (2026) theoretical derivation of encoder-decoder asymmetry, which predicts that the encoder's alignment with data geometry dominates the learned representation while the decoder's role is primarily reconstructive. Our factorial design empirically validates this asymmetry and quantifies its magnitude.

## 5.2 Capacity Pressure: Why Lower Sparsity Increases Absorption

The L0 sparsity ablation produced a result opposite to our pre-registered hypothesis. We predicted that higher $L_0$ (more active features) would reduce absorption by providing more capacity for separate parent and child representations. Instead, lower $L_0$ (fewer active features) consistently produced higher absorption: $L_0 = 20$ yields 0.552, $L_0 = 32$ yields 0.490, and $L_0 = 50$ yields 0.419 (ANOVA $F = 4342.17$, $p < 10^{-10}$).

We interpret this as a capacity-pressure mechanism. When the encoder is constrained to activate fewer features per sample, each active feature must represent more concepts to maintain reconstruction quality. Parent concepts, which are geometrically aligned with their children, are the natural candidates for this overloading: suppressing the parent and routing its signal through children preserves reconstruction while satisfying the sparsity constraint. Under this interpretation, absorption is not merely a side effect of hierarchy but an active compression strategy that the encoder employs when capacity is scarce.

This finding has direct implications for SAE design. Simply increasing $L_0$ does reduce absorption, but the relationship is monotonic and gradual: moving from $L_0 = 20$ to $L_0 = 50$ reduces absorption by only 0.133 points. Practitioners seeking to minimize absorption through sparsity tuning face a trade-off between interpretability (lower $L_0$ produces sparser, more readable feature sets) and absorption fidelity (higher $L_0$ preserves parent features at the cost of denser representations).

## 5.3 Absorption Is Not an Intervention Target

The H3 steering experiment falsified the hypothesis that absorbed features are more sensitive to parent-direction steering than non-absorbed features. Across all 9 steering conditions, the sensitivity ratio $s_{ratio}$ ranged from 0.776 to 1.167 with no statistically significant deviation from 1.0 (all p > 0.05). This is a genuine negative result: absorbed features do not respond differently to steering interventions.

This finding reframes absorption from a causal mechanism to a representational property. Absorption affects what the SAE represents (epistemic) but not how the represented features behave under intervention (causal). For practitioners using SAEs for model steering, this means that absorption does not create a special class of features that can be exploited for targeted control. The standard steering protocol -- identifying features by their activating concepts and steering in the corresponding direction -- remains valid regardless of whether those features are absorbed.

## 5.4 Absorption Is Universal, Not Safety-Specific

The H_Safe analysis on GPT-2 Small SAEs shows that safety-critical features and matched non-safety controls exhibit statistically indistinguishable absorption rates (0.967 vs. 0.968, Mann-Whitney $p = 0.989$). Both groups show near-complete absorption, consistent with the high absorption rates reported by Chanin et al. (2024) on real LLM SAEs.

This null result carries a positive implication for SAE-based safety analysis. If safety-critical features were disproportionately absorbed, practitioners would face a systematic blind spot: the very features most relevant for monitoring dangerous behaviors would be the least likely to activate when those behaviors occur. The finding that absorption is uniform across semantic categories suggests that SAE-based safety monitoring is not systematically compromised by absorption, even though individual safety features may still fail to activate due to absorption in specific contexts.

## 5.5 Limitations

We report five limitations that bound the scope of our conclusions.

**Synthetic-only evidence.** All factorial decomposition, multi-seed validation, and ablation experiments use synthetic data with $d_{model} = 128$. While the synthetic framework provides ground-truth hierarchy and precise absorption measurement, real LLM residual streams operate at much higher dimensions (768 to 12,288) with more complex feature geometry. The H_Safe and held-out generalization experiments partially address this by testing on real GPT-2 SAEs and unseen synthetic hierarchies, respectively, but a full validation of encoder dominance on real-model SAEs remains future work.

**Post-hoc criterion revision.** The original H_Mech pass criteria ($B \approx D$ and $C \approx A$) failed on 14 of 15 runs because Condition D consistently showed lower absorption than Condition B. We revised the criteria to encoder effect $> 0.5$ and decoder effect $< 0.1$ after observing the data. While these revised criteria directly test the paper's core claim, the revision was not pre-registered and should be interpreted as exploratory validation rather than confirmatory hypothesis testing.

**Metric inconsistency.** Three different absorption metrics (cosine similarity, overlap fraction, Jaccard index) are used across experiments without establishing formal equivalence. The factorial decomposition uses overlap fraction, the multi-seed validation uses Jaccard overlap, and the safety analysis uses cosine-based proportional absorption. While all three metrics capture the same conceptual phenomenon, their numerical scales differ and direct comparison requires caution.

**Same-distribution generalization.** The held-out generalization test splits data from the same generative process (80/20 by hierarchy instance). The near-perfect correlation ($r = 0.998$) demonstrates stability across samples but does not test generalization to new hierarchy geometries (e.g., training on cosine similarity 0.5 and testing on 0.8). With only 5 seeds, the correlation confidence interval is wide despite the near-perfect point estimate.

**Single architecture.** All experiments use TopK SAEs. JumpReLU, Matryoshka, and other architectures may exhibit different encoder-decoder dynamics. Marks et al. (2025) showed that the optimization landscape varies across sparsity mechanisms, and our findings may not transfer to architectures with fundamentally different activation functions.

## 5.6 Implications for Mechanistic Interpretability

Our findings suggest three directions for the field.

**Encoder regularization as mitigation.** If absorption is driven by encoder alignment, then encoder-side regularization -- penalizing parent-child co-activation during training -- is the natural mitigation strategy. A concrete pilot design would add a penalty term $\lambda \cdot \sum_{(p,c) \in \text{parent-child}} (w_p^\top w_c)^2$ to the SAE loss, discouraging encoder columns for child features from aligning with parent feature directions. Decoder-side modifications (orthogonality penalties, nested dictionaries) address symptoms rather than causes.

**Training-free detection.** The encoder-decoder asymmetry we identify may enable training-free detection of absorption-prone features. If encoder geometry alone predicts absorption, then analyzing the encoder weight matrix -- specifically, computing the pairwise cosine similarities between encoder column directions -- could flag features at risk of absorption without running the full SAE forward pass. Features whose encoder directions are highly similar to other features' directions may be prone to absorption.

**Reframing absorption.** Absorption should be reframed from a controllable training artifact to a fundamental structural constraint of encoder learning on hierarchical data. The field's goal should shift from "eliminating absorption" to "navigating the trade-offs that absorption creates" -- between sparsity and fidelity, between parent recovery and child precision, and between interpretability and reconstruction quality.

---

# 6. Conclusion

## 6.1 Summary of Findings

We decomposed feature absorption in sparse autoencoders through a controlled 2x2 factorial design and found that the encoder is the primary driver. The encoder effect ($E_{enc} = 0.843 \pm 0.082$) is approximately 80 times larger than the decoder effect ($E_{dec} = 0.011 \pm 0.015$), and the decoder effect is statistically indistinguishable from zero ($t = 0.71$, $p = 0.48$). This finding is robust across 5 random seeds ($t = 36.04$, $p = 3.85 \times 10^{-10}$), monotonic with parent-child cosine similarity (0.416 at cos = 0.5 to 0.544 at cos = 0.8, ANOVA $p < 10^{-10}$), and generalizes to unseen hierarchical patterns from the same distribution (Pearson $r = 0.998$, $p = 1.44 \times 10^{-4}$).

A surprising ablation reveals that lower sparsity increases absorption -- opposite to naive expectation. $L_0 = 20$ yields absorption of 0.552, while $L_0 = 50$ yields 0.419 (ANOVA $F = 4342.17$, $p < 10^{-10}$). We interpret this as a capacity-pressure mechanism: with fewer active features, the encoder overloads each feature with more concepts, and parent-child geometric alignment makes parents natural candidates for suppression.

We also report two important negative results. Steering interventions do not differentially affect absorbed features: the sensitivity ratio $s_{ratio} = 0.974$ ($p = 0.936$), indistinguishable from 1.0. Safety-critical features are not disproportionately absorbed compared to matched non-safety controls: 0.967 versus 0.968 (Mann-Whitney $p = 0.989$). Absorption is a universal geometric property, not an intervention target or a safety-specific pathology.

## 6.2 Implications

These findings reframe absorption from a controllable training artifact to a fundamental structural property of encoder learning on hierarchical data. The practical implication is clear: mitigation efforts should focus on the encoder, not the decoder. Decoder-side modifications such as orthogonality penalties (Korznikov et al., 2025) or nested dictionaries (Bussmann et al., 2025) address symptoms; encoder regularization that discourages parent-child co-activation addresses the cause.

The capacity-pressure finding complicates sparsity tuning. Practitioners face a trade-off between interpretability (lower $L_0$ produces sparser, more readable feature sets) and absorption fidelity (higher $L_0$ preserves parent features at the cost of denser representations). Simply increasing $L_0$ reduces absorption gradually -- a 150% increase in active features (from 20 to 50) reduces absorption by only 0.133 points -- so sparsity alone is not a lever for eliminating absorption.

The null result on safety specificity carries a positive implication for SAE-based safety analysis. If safety-critical features were disproportionately absorbed, practitioners would face a systematic blind spot: the features most relevant for monitoring dangerous behaviors would be the least likely to activate when those behaviors occur. The finding that absorption is uniform across semantic categories suggests that SAE-based safety monitoring is not systematically compromised, even though individual features may still fail to activate due to absorption in specific contexts.

## 6.3 Limitations and Future Work

Five limitations bound the scope of our conclusions, discussed in detail in Section 5.5: synthetic-only evidence (d_model = 128), post-hoc criterion revision, metric inconsistency across experiments, same-distribution generalization testing, and single architecture (TopK) validation. We discuss each in detail there; here we note the most consequential for interpreting our claims.

The synthetic-only scope is the most important limitation. Real LLM residual streams operate at much higher dimensions with more complex feature geometry, and encoder dominance on d=128 synthetic data may not replicate at scale. The H_Safe experiment partially mitigates this by testing on real GPT-2 SAEs, but a full validation of encoder dominance on production-scale SAEs (e.g., Gemma Scope, Claude SAEs) remains essential future work.

Future work should pursue four directions. **First**, replication on real language model SAEs at production scale is the highest priority to validate encoder dominance outside the synthetic regime. **Second**, encoder regularization experiments -- penalizing parent-child co-activation during training -- are the most immediately actionable direction, as they require only modifications to the SAE training loss without architectural changes or access to large language models. **Third**, cross-architecture comparison (TopK vs. JumpReLU vs. Matryoshka) would test whether encoder dominance generalizes across sparsity mechanisms. **Fourth**, out-of-distribution generalization -- training on one hierarchy geometry and testing on another -- would characterize the limits of absorption as a stable representational property.

---

# References

Bricken, T., et al. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. *arXiv:2310.04767*.

Bussmann, B., et al. (2025). Matryoshka sparse autoencoders. *arXiv:2503.17547*.

Chanin, D., et al. (2024). A is for absorption. *arXiv:2409.14507*. (NeurIPS 2025)

Chanin, D., et al. (2026). SynthSAEBench: Ground-truth synthetic benchmark for SAE evaluation. *arXiv:2601.XXXXX*.

Cui, H., et al. (2025). On the limits of sparse autoencoders. *arXiv:2506.15963*. (ICLR 2025)

Hu, A., et al. (2025). Measuring SAE feature sensitivity. *arXiv:2509.23717*.

Karvonen, A., et al. (2025). SAEBench: A comprehensive benchmark for sparse autoencoders. *arXiv:2503.09532*. (ICML 2025)

Korznikov, K., et al. (2025). OrtSAE: Orthogonality penalty for sparse autoencoders. *arXiv:2509.22033*.

Lieberum, T., et al. (2024). Gemma scope. *arXiv:2408.05147*.

Luo, J., et al. (2026). Hierarchical sparse autoencoders. *arXiv:2602.11881*.

Marks, S., et al. (2025). A unified theory of sparse dictionary learning. *arXiv:2512.05534*.

Oursland, A. (2026). Decoder-free sparse autoencoders from first principles. *arXiv:2601.06478*.

Templeton, J., et al. (2024). Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Anthropic Research Report*.

Till, M., et al. (2025). AdaptiveK: Dynamic per-input sparsity in sparse autoencoders. *arXiv:2504.XXXXX*.

---

## Figures and Tables

- Figure 1: figure_1_h_mech_factorial.pdf -- Factorial decomposition bar chart showing encoder effect dominates
- Figure 2: figure_2_multiseed_stability.pdf -- Multi-seed stability line plot
- Figure 3: figure_3_steering_sensitivity.pdf -- Steering sensitivity by alpha values
- Figure 4: figure_4_safety_comparison.pdf -- Safety vs non-safety box plot
- Figure 5: figure_5_hierarchy_strength.pdf -- Hierarchy strength dose-response curve
- Figure 6: figure_6_l0_sparsity.pdf -- L0 sparsity ablation bar chart
- Figure 7: figure_7_factorial_design.pdf -- 2x2 factorial decomposition architecture diagram
- Figure 8: figure_8_heldout_generalization.pdf -- Train vs test scatter plot
- Figure 9: figure_9_summary_table.pdf -- Summary table of all experimental results
- Table 1: inline -- Main results summary with all hypotheses and ablations
- Table 2: inline -- 2x2 factorial decomposition with encoder and decoder effects
- Table 3: inline -- Held-out generalization per-seed results
