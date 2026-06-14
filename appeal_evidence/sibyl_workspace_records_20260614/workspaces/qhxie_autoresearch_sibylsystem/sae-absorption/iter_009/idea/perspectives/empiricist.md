# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024) "A is for Absorption"** (arXiv:2409.14507, NeurIPS 2025) --- Defines the canonical absorption metric: k-sparse probing to find feature splits, false-negative identification, integrated-gradients ablation to detect absorbing latents. Thresholds (cosine similarity > 0.025, ablation gap >= 1.0) chosen by manual inspection. Key limitation: metric requires known probe directions and does not work well in later layers where ablation effects vanish.

2. **Karvonen et al. (2025) "SAEBench"** (arXiv:2503.09532, ICML 2025) --- Extends Chanin's metric to all model layers by replacing ablation-based detection with projection of firing latents against LR probe direction. Introduces fractional absorption score (mean across test inputs) to handle partial absorption and distributed responsibility across multiple absorbing latents. Evaluates 200+ SAEs across 7 architectures, 3 widths, 6 sparsities on Pythia-160M and Gemma-2-2B. Critical finding: dead features confound absorption measurement (ReLU SAEs appeared low-absorption due to dead features, not genuine mitigation).

3. **Chanin & Garriga-Alonso (2025) "Sparse but Wrong"** (arXiv:2508.16560) --- Demonstrates incorrect L0 causes SAEs to learn systematically wrong features. Proposes proxy metric for finding correct L0. Critical methodological implication: most open-source SAEs have too-low L0, meaning observed "absorption" may partly be L0-induced hedging. Any absorption study must control for L0 correctness.

4. **Tian et al. (2025) "Measuring SAE Feature Sensitivity"** (arXiv:2509.23717) --- Frames absorption as a special case of poor feature sensitivity. Develops scalable sensitivity evaluation that does not require integrated-gradients ablation. Finds many features with good activation examples still have poor sensitivity. Provides a complementary metric dimension missed by absorption-only evaluations.

5. **Huang et al. (2024) "RAVEL"** (ACL 2024) --- Entity-attribute disentanglement benchmark with cities (country, continent, language), Nobel laureates, verbs, objects, occupations. Uses interchange interventions with cause score and isolation score. Provides the exact kind of hierarchical entity-attribute structure needed to test absorption beyond first-letter spelling.

6. **SynthSAEBench (2026)** (arXiv:2602.14687) --- Large-scale synthetic data model with configurable feature hierarchy, correlation, and Zipfian firing distributions. Ground-truth features known. Logistic probes achieve 0.974 F1 vs. best SAE ~0.88. Enables controlled ablation of hierarchy depth and co-occurrence frequency on absorption.

7. **Bussmann et al. (2025) "Matryoshka SAE"** (arXiv:2503.17547, ICML 2025) --- Absorption rate ~0.03 vs. BatchTopK ~0.29. Nested prefix losses create natural feature hierarchy. Evaluated on SAEBench absorption, RAVEL, sparse probing, SCR. Inner levels suffer feature hedging --- demonstrates the absorption-hedging tradeoff.

8. **Chanin et al. (2025) "Feature Hedging"** (arXiv:2505.11756) --- Defines hedging as a complementary failure mode to absorption. Provides theoretical and empirical demonstration that narrow SAEs merge correlated features. Proposes balanced Matryoshka SAE with compound multiplier ~0.75. Essential for disentangling hedging from absorption in any measurement study.

9. **Korznikov et al. (2026) "Sanity Checks for SAEs"** (arXiv:2602.14111) --- Shows SAEs recover only 9% of true features in synthetic settings; random baselines match trained SAEs on interpretability, sparse probing, and causal editing. Forces any absorption study to include random-baseline controls.

10. **Narayanaswamy et al. (2026) "Masked Regularization"** (arXiv:2604.06495) --- Most recent mitigation: masking-based regularization to disrupt co-occurrence patterns during training. Reduces absorption and improves OOD robustness. Demonstrates that training-time intervention on co-occurrence statistics directly reduces absorption.

11. **Li et al. (2025) "ATM SAE"** (arXiv:2510.08855) --- Adaptive Temporal Masking achieves absorption score 0.0068 vs. TopK 0.1402 and JumpReLU 0.0114 on Gemma-2-2B. Best reported absorption scores. Evaluated only on one model --- generalization unknown.

12. **Song et al. (2025) "Feature Consistency"** (arXiv:2505.20254) --- Shows SAE features are inconsistent across training runs. Proposes PW-MCC metric. Essential control: any absorption measurement must account for run-to-run variability with multiple seeds.

### Experimental Landscape

**What has been properly tested:**
- Absorption rate on the first-letter spelling task across Gemma Scope, Llama 3.2, Qwen2 SAEs (Chanin et al.)
- SAEBench's 8-metric suite on 200+ SAEs including absorption (Karvonen et al.)
- Matryoshka SAE vs. BatchTopK absorption rates under SAEBench conditions
- SynthSAEBench probing F1 vs. logistic probe baseline in synthetic settings with known hierarchy

**What is accepted without proper evidence:**
- The assumption that first-letter absorption rates generalize to other feature hierarchies (entity types, knowledge, safety)
- The claim that absorption is purely caused by sparsity optimization --- no study has disentangled L0-induced hedging from hierarchy-driven absorption in real SAEs
- ATM SAE's absorption score of 0.0068 --- evaluated on a single model (Gemma-2-2B); no independent replication
- The causal link between absorption and SAE underperformance on downstream safety tasks (DeepMind blog claims this but provides no quantitative decomposition)

**Where methodological gaps exist:**
- No cross-domain absorption characterization beyond first-letter spelling
- No head-to-head comparison of all absorption-mitigating architectures under matched conditions
- No unsupervised absorption detection method (all require known probe directions)
- No statistical power analysis for absorption rate estimates (sample size, confidence intervals)
- No study disentangling absorption from hedging and L0 confounds in the same experimental setup
- No absorption measurement that accounts for run-to-run feature inconsistency

---

## Phase 2: Initial Candidates

### Candidate A: Cross-Domain Absorption Taxonomy --- Quantifying Absorption Across Semantic Hierarchy Types

- **Hypothesis**: Absorption rate varies systematically with feature hierarchy properties (hierarchy depth, parent-child co-occurrence frequency, semantic distance between parent and child). Specifically, absorption rate will be higher for hierarchies with (1) higher parent-child co-occurrence and (2) shallower hierarchy depth, following the sparsity-gain argument from Chanin et al. Falsifiable prediction: on Gemma-2-2B layer 12 with Gemma Scope 16k SAE, absorption rate on entity-type hierarchies (city -> country) will be >= 0.15 (comparable to first-letter rates of 0.15--0.35), while absorption on deeper hierarchies (city -> continent) will be lower by at least 0.05.

- **Falsification criterion**: If absorption rate on entity-type hierarchies is < 0.05 across all tested SAEs and layers (i.e., absorption is negligible outside first-letter tasks), the hypothesis that absorption is a general SAE pathology is weakened --- it may be specific to syntactic-level features.

- **Evaluation protocol**:
  - Benchmarks: RAVEL dataset (cities: country/continent/language; Nobel laureates: country/field), first-letter spelling task (as positive control replicating Chanin et al.)
  - Metrics: Absorption rate (Chanin metric + SAEBench fractional variant), sparse probing F1, LR probe baseline F1, false-negative rate per hierarchy level
  - Statistical tests: Bootstrap 95% CI on absorption rates (10,000 resamples), paired Wilcoxon signed-rank test comparing absorption rates across hierarchy types within the same SAE, Bonferroni correction for multiple comparisons
  - Models: Gemma-2-2B (Gemma Scope SAEs, widths 16k and 65k), GPT-2 Small (as secondary validation)

- **Ablation plan**:
  - Ablation 1: Remove hierarchy (flat classification, e.g., city identity only) --- if absorption persists without hierarchy, it is not hierarchy-driven
  - Ablation 2: Vary co-occurrence frequency by selecting high-frequency vs. low-frequency entities --- tests whether co-occurrence drives absorption
  - Ablation 3: Vary SAE width (16k vs. 65k) --- tests width dependence of cross-domain absorption
  - Ablation 4: Vary layer (early/middle/late) --- tests layer dependence

- **Confounders**:
  - Probe quality: LR probe accuracy may differ across hierarchy types, making absorption rates incomparable. Control: report probe accuracy alongside absorption rate; only compare hierarchies where probe accuracy > 0.90.
  - Feature hedging: Narrow SAEs may show apparent "absorption" that is actually hedging. Control: measure absorption at multiple widths; use Chanin's hedging metric alongside absorption metric.
  - L0 confound: SAEs may have incorrect L0. Control: evaluate across 3+ sparsity levels; report L0 alongside absorption.
  - Entity frequency: High-frequency entities may have dedicated SAE features, masking absorption. Control: stratify analysis by entity frequency.

- **Pilot design**: Load Gemma-2-2B + Gemma Scope 16k SAE (layer 12). Train LR probes for 3 RAVEL attributes (city->country, city->continent, city->language) and 3 first-letter tasks (letters s, a, t). Compute absorption rate for each. Total: ~10 min on single GPU (probe training + SAE encoding + absorption calculation for 6 tasks). Success signal: non-zero absorption rate (> 0.02) on at least one RAVEL hierarchy.

### Candidate B: Disentangling Absorption from Hedging --- A Controlled Decomposition

- **Hypothesis**: Observed false-negative rates in SAE features conflate two distinct phenomena: (1) true hierarchy-driven absorption (specific child feature absorbs parent) and (2) L0-induced hedging (insufficient sparsity budget merges correlated features). These can be distinguished by their dependence on SAE width and L0: hedging decreases monotonically with width at fixed L0, while absorption is width-independent at fixed hierarchy structure. Falsifiable prediction: at matched L0, doubling SAE width from 16k to 65k will reduce hedging-attributed false negatives by > 30% but reduce absorption-attributed false negatives by < 10%.

- **Falsification criterion**: If doubling width reduces both hedging and absorption false negatives by similar magnitudes (within 10% of each other), the distinction between the two phenomena is empirically vacuous at the granularity we can measure.

- **Evaluation protocol**:
  - Benchmarks: First-letter spelling task (established baseline), 2 RAVEL hierarchies
  - Metrics: (a) Total false-negative rate, (b) absorption-attributed false-negative rate (using Chanin's integrated-gradients criterion: absorbing latent has high cosine similarity with probe direction), (c) hedging-attributed false-negative rate (remaining false negatives where no single absorbing latent is identified, suggesting feature merging), (d) cosine similarity between SAE decoder vectors for correlated features (high cosine = hedging signature)
  - SAE configurations: 4k, 16k, 65k width at 3 matched L0 levels (L0 ~ 25, 50, 100) --- 9 configurations total, using SAEBench's pre-trained SAEs on Gemma-2-2B layer 12
  - Statistical tests: Two-way ANOVA (width x L0) on absorption rate and hedging rate, with post-hoc Tukey HSD

- **Ablation plan**:
  - Ablation 1: Width sweep at fixed L0 --- isolates width effect
  - Ablation 2: L0 sweep at fixed width --- isolates sparsity effect
  - Ablation 3: Decoder cosine similarity analysis --- characterizes the geometric signature of hedging vs. absorption
  - Ablation 4: Random baseline comparison --- what fraction of false negatives occur with a random sparse encoder? (following Korznikov et al., 2026)

- **Confounders**:
  - SAE architecture differences between widths: SAEBench uses the same architecture (BatchTopK) across widths, controlling this.
  - Training duration: Wider SAEs may have been trained longer. Control: verify training token counts are matched.
  - Dead features: More dead features at larger width could confound. Control: report alive-feature ratio.
  - The operational definition of "hedging-attributed false negative" is a residual category --- it may contain other failure modes. Control: report the fraction of false negatives that are unclassified.

- **Pilot design**: Load 2 SAEBench SAEs (16k and 65k at matched L0 ~ 50) on Gemma-2-2B layer 12. Compute absorption rate on first-letter task for letters a, s, t. Compare total false-negative rate, absorption-attributed FN rate, hedging-attributed FN rate. ~15 min. Success signal: absorption-attributed FN rate is measurably different from hedging-attributed FN rate in at least one configuration.

### Candidate C: Absorption Scaling Laws --- Predicting Absorption Rate from SAE Configuration

- **Hypothesis**: Absorption rate follows a predictable scaling law as a function of SAE width (W), L0, and feature co-occurrence frequency (f_co). Specifically, absorption rate ~ C * f_co^alpha * W^(-beta) * L0^(-gamma), where alpha > 0 (more co-occurrence = more absorption), beta ~ 0 (weak width dependence, following Chanin et al.'s observation), and gamma < 0 (higher sparsity = more absorption, since absorption saves L0). Falsifiable prediction: the fitted scaling law will achieve R^2 > 0.7 on held-out SAE configurations.

- **Falsification criterion**: If R^2 < 0.3 on held-out configurations, absorption rate is not predictable from these variables --- it depends on other factors (initialization, training dynamics, or interactions we have not modeled).

- **Evaluation protocol**:
  - Benchmarks: First-letter spelling task (26 letters provide natural replication), 3 RAVEL hierarchies
  - Metrics: Absorption rate per letter/attribute, fitted power-law coefficients, R^2 on held-out data
  - SAE grid: Use all available SAEBench SAEs (3 widths x 6 sparsities x 7 architectures on Gemma-2-2B layer 12) --- approximately 126 SAEs, with 80% train / 20% test split
  - Statistical tests: Nonlinear least squares fitting with bootstrap CIs on exponents, leave-one-architecture-out cross-validation
  - Co-occurrence frequency: Estimated from OpenWebText token frequency for first-letter task; from Wikidata entity co-occurrence for RAVEL

- **Ablation plan**:
  - Ablation 1: Fit model with width only --- tests whether width alone predicts absorption
  - Ablation 2: Fit model with L0 only --- tests whether sparsity alone predicts absorption
  - Ablation 3: Fit model with co-occurrence only --- tests whether co-occurrence alone predicts absorption
  - Ablation 4: Full model vs. reduced models --- tests whether the interaction terms matter
  - Ablation 5: Per-architecture analysis --- do different architectures follow different scaling laws?

- **Confounders**:
  - Architecture as a confound: Matryoshka SAEs have structurally different absorption behavior. Control: fit separate laws per architecture family and test whether they differ.
  - Training token budget: SAEs trained with different budgets may have different absorption. Control: normalize by training tokens where available.
  - Feature co-occurrence estimation error: Frequency estimates from OpenWebText may not match the actual training distribution. Control: use multiple frequency estimation methods and report sensitivity.
  - Overfitting with small sample: 126 SAEs with multiple free parameters. Control: use cross-validation and report prediction intervals.

- **Pilot design**: Load 6 SAEBench SAEs (2 widths x 3 L0 levels, all BatchTopK on Gemma-2-2B layer 12). Compute absorption rate on 10 letters. Fit power law to the 6 x 10 = 60 data points. ~15 min. Success signal: R^2 > 0.3 on training data, suggesting a signal worth pursuing.

---

## Phase 3: Self-Critique

### Against Candidate A (Cross-Domain Absorption Taxonomy)

- **Confound attack**: The RAVEL entity-attribute hierarchy may not exhibit absorption the same way as first-letter spelling because the model's internal representation of "city->country" is fundamentally different from "token->first letter." The first-letter task creates a clean parent-child structure where the parent (first letter) is a strict function of the child (token identity). In RAVEL, "city->country" is a factual association, not a deterministic function --- the model may encode these differently (distributed factual knowledge vs. syntactic feature). I searched for papers on how LLMs internally represent factual vs. syntactic knowledge hierarchies but found no direct comparison of absorption behavior across these types. This is a genuine unknown, not a fatal flaw --- but it means the results may show zero absorption on RAVEL tasks even if absorption is a real phenomenon, because the representational structure differs.

- **Statistical attack**: The RAVEL dataset has 500 entities per type. If absorption rate is low (say 5%), we need ~200 false-negative tokens to detect it with reasonable power. With 500 entities and, say, 30% false-negative rate on the probe, we have ~150 false-negative tokens --- borderline sufficient. For rarer entity types, statistical power could be inadequate. Mitigation: expand the dataset with additional cities from RAVEL's HuggingFace release (3000+ cities available) and compute power analysis before running full experiments.

- **Benchmark attack**: RAVEL was designed for disentanglement evaluation, not absorption measurement. The interchange-intervention protocol measures a different thing (causal effect of swapping features) than the absorption metric (whether features fail to fire). We need to adapt the absorption metric to RAVEL, not use RAVEL's native metric. This is doable but requires careful validation that the adapted metric is still measuring absorption and not some other failure mode.

- **Ablation completeness attack**: The ablation plan does not include a comparison between absorption measured via Chanin's original method (integrated-gradients) and SAEBench's modified method (probe projection). If the two methods disagree on cross-domain tasks, we cannot tell which is correct. Mitigation: run both methods and report agreement rate.

- **Verdict**: **STRONG** --- This addresses the most important gap (Gap 2: absorption only studied on first-letter task) with a clear experimental design. The main risk is that RAVEL hierarchies may not trigger absorption, but this would itself be an important finding. The pilot can quickly determine feasibility.

### Against Candidate B (Disentangling Absorption from Hedging)

- **Confound attack**: The operational definition of "hedging-attributed false negative" (false negatives not explained by absorption) is a residual category that could contain many things besides hedging: dead features, multi-dimensional features, features the probe fails to detect, etc. Without a positive signature for hedging (not just the absence of absorption signature), the decomposition is not clean. Chanin et al. (2025) define hedging via decoder cosine similarity (merged features have high cosine similarity), which provides a partial positive signature, but it is not incorporated into the false-negative decomposition.

- **Statistical attack**: The 2-way ANOVA requires sufficient data points per cell. With 3 widths x 3 L0 levels = 9 cells, and 26 letters providing the replicates within each cell, we have 26 replicates per cell --- adequate for ANOVA. However, the letters are not independent (frequency varies by orders of magnitude), violating the i.i.d. assumption. Mitigation: use a mixed-effects model with letter as a random effect.

- **Benchmark attack**: SAEBench's pre-trained SAEs may not have exactly matched L0 across widths. The "matched L0" claim needs verification. If L0 is only approximately matched, the width effect is confounded with L0. Mitigation: verify actual L0 values and include L0 as a continuous covariate.

- **Ablation completeness attack**: Missing ablation: what happens when you artificially create a known hedging scenario (two highly correlated features forced to share a latent) and verify the metric correctly identifies it as hedging? Without this validation on a controlled case, we do not know if the decomposition works. Mitigation: create a synthetic control using SynthSAEBench with known feature structure.

- **Verdict**: **MODERATE** --- Conceptually important but operationally tricky. The residual-category problem for hedging attribution is a real weakness. The experiment would be significantly strengthened by adding synthetic validation (SynthSAEBench) where ground truth is known.

### Against Candidate C (Absorption Scaling Laws)

- **Confound attack**: Feature co-occurrence frequency is estimated, not known, in real LLM SAEs. For the first-letter task, token frequency is a reasonable proxy, but for RAVEL, entity co-occurrence in the training corpus is much harder to estimate. Errors in co-occurrence estimation would corrupt the scaling law fit. Additionally, the number of free parameters (C, alpha, beta, gamma) relative to the number of distinct SAE configurations (9 per architecture, with 7 architectures but very different behaviors) makes overfitting a real risk.

- **Statistical attack**: With 126 SAEs and a power-law model with 4 parameters, we have decent degrees of freedom. However, the data points are not independent --- SAEs at different widths trained from the same codebase share initialization biases, and the 26 letters within each SAE are correlated. Effective sample size may be much smaller than nominal. R^2 > 0.7 is ambitious for such a noisy, confounded regression.

- **Benchmark attack**: First-letter absorption is already well-characterized by Chanin et al. and SAEBench. A scaling law that only works for first-letter tasks adds incremental value. The real test is whether the law generalizes to other hierarchies, but we do not have enough data points across hierarchies to fit a law there.

- **Ablation completeness attack**: The per-architecture ablation is essential but may reveal that the scaling law parameters differ so much across architectures that no universal law exists. This is likely the case given that Matryoshka SAEs (~0.03) vs. JumpReLU (~0.29) show order-of-magnitude differences that are architecture-driven, not configuration-driven.

- **Verdict**: **WEAK** --- The ambition is admirable, but the noise level, confounds, and risk of architecture-specific laws make this unlikely to produce a clean, generalizable result within the project's constraints. Better to focus on the controlled cross-domain characterization (Candidate A) first and see if enough data emerges to attempt scaling laws in a later iteration.

---

## Phase 4: Refinement

### Dropped

**Candidate C (Absorption Scaling Laws)**: Dropped due to (1) high risk of overfitting with limited data, (2) architecture-specific effects that likely dominate over configuration-driven scaling, (3) difficulty estimating co-occurrence frequency for non-spelling hierarchies. The scaling-law question is better addressed after Candidate A establishes whether absorption generalizes across domains --- without that, there is nothing to fit a law to.

### Strengthened Survivors

**Candidate A (Cross-Domain Absorption Taxonomy)** --- strengthened as front-runner:

1. **Added missing controls:**
   - Include random-baseline SAE control (following Korznikov et al., 2026): compute "absorption rate" for a randomly initialized sparse encoder to establish the null distribution.
   - Include both Chanin's integrated-gradients method and SAEBench's probe-projection method, reporting agreement rate.
   - Add LR probe accuracy as a qualifying threshold: only analyze hierarchies where probe accuracy > 0.90.
   - Stratify all results by entity frequency (high/medium/low tertiles).

2. **Tightened falsification criterion:**
   - Primary: If absorption rate < 0.02 on ALL tested RAVEL hierarchies across ALL tested SAEs AND the first-letter positive control shows absorption > 0.10, then absorption is specific to syntactic features.
   - Secondary: If absorption rate on RAVEL hierarchies is indistinguishable from the random-baseline rate (overlapping 95% CIs), then the signal is not real.

3. **Additional benchmarks:**
   - RAVEL: cities (country, continent, language) --- 3 hierarchy types with varying depth
   - RAVEL: Nobel laureates (country, field) --- different entity domain
   - First-letter spelling task (26 letters) --- positive control
   - Token-frequency hierarchy (common vs. rare tokens of the same type) --- tests co-occurrence frequency effect
   - Total: 5+ hierarchy types providing a taxonomy of absorption behavior.

4. **Analysis plan:**
   - Figure 1: Absorption rate heatmap (hierarchy type x SAE configuration), with 95% bootstrap CIs
   - Figure 2: Absorption rate vs. parent-child co-occurrence frequency scatter plot, testing correlation
   - Figure 3: Absorption rate by layer (early/middle/late) for each hierarchy type
   - Table 1: Per-hierarchy breakdown of total false-negative rate, absorption-attributed rate, unclassified rate
   - Table 2: Method agreement (Chanin IG vs. SAEBench projection) per hierarchy type

**Candidate B (Disentangling Absorption from Hedging)** --- incorporated as a sub-experiment of Candidate A:

Rather than a standalone study, the absorption-hedging decomposition becomes a component of the cross-domain analysis:
- For each hierarchy type, report: (a) total false-negative rate, (b) absorption-attributed FN rate, (c) remaining (hedging-candidate) FN rate
- Decoder cosine similarity analysis between correlated features as hedging signature
- Width-dependence analysis (16k vs. 65k) to separate width-sensitive hedging from width-insensitive absorption
- Synthetic validation: Run the decomposition on SynthSAEBench with known feature hierarchy to validate the method before applying to real SAEs

### Selected Front-Runner

**Candidate A: Cross-Domain Absorption Taxonomy** --- because:
1. Directly addresses the most important research gap (Gap 2: absorption only studied on first-letter tasks)
2. Uses existing infrastructure (SAELens + Gemma Scope + RAVEL) --- no new training required (training-free constraint satisfied)
3. The pilot is fast (10 min) and the full experiment is feasible within time budget
4. Negative results are also publishable (if absorption is first-letter-specific, that constrains the scope of the problem)
5. Provides the empirical foundation needed for any future scaling law or theoretical work

---

## Phase 5: Final Proposal

- **Title**: Beyond First Letters: Systematically Characterizing Feature Absorption Across Semantic Hierarchy Types in Sparse Autoencoders

- **Hypothesis**: Feature absorption in SAEs is not confined to the first-letter spelling task but occurs systematically across semantic feature hierarchies with predictable severity. Specifically, absorption rate is positively correlated with parent-child feature co-occurrence frequency (Spearman rho > 0.3, p < 0.05) and is measurably present (absorption rate > 0.05) in at least 2 of 4 tested entity-attribute hierarchy types on Gemma-2-2B Gemma Scope SAEs.

- **Falsification criterion**: If absorption rate is < 0.02 on ALL entity-attribute hierarchies across ALL tested SAEs AND layers, while the first-letter positive control shows absorption rate > 0.10, then absorption is specific to syntactic-level features and does not generalize to semantic knowledge hierarchies. This would fundamentally change how the community should prioritize absorption mitigation research.

- **Method**: Training-free analysis of pre-trained SAEs using adapted absorption metrics applied to multiple feature hierarchy types.

  **Phase 1 --- Metric Validation (replicate and extend Chanin et al.):**
  - Replicate Chanin et al.'s first-letter absorption measurement on Gemma Scope 16k SAE, Gemma-2-2B layer 12, as a positive control.
  - Implement both Chanin's IG-based absorption metric and SAEBench's projection-based absorption metric.
  - Validate agreement between the two methods on the first-letter task.
  - Compute random-baseline absorption rate (randomly initialized sparse encoder) to establish the null distribution.

  **Phase 2 --- Cross-Domain Probe Construction:**
  - Train LR probes for each target hierarchy on Gemma-2-2B residual stream activations:
    - First-letter spelling (26 letters) --- positive control
    - City -> Country (RAVEL, 500+ entities, ~50 countries)
    - City -> Continent (RAVEL, 500+ entities, 6 continents)
    - City -> Language (RAVEL, 500+ entities, ~30 languages)
    - Nobel Laureate -> Country (RAVEL, 400+ entities)
  - Quality gate: Only proceed with hierarchies where LR probe accuracy > 0.90.
  - For each hierarchy, perform k-sparse probing on SAE latents (k = 1, 2, ..., 10) to identify feature splits.

  **Phase 3 --- Cross-Domain Absorption Measurement:**
  - For each qualified hierarchy, compute:
    - Absorption rate (SAEBench fractional method, extended to the new hierarchies)
    - False-negative rate (fraction of inputs where all k-split latents fail to activate but probe is correct)
    - Absorption-attributed false-negative rate (absorbing latent identified via probe projection)
    - Unclassified false-negative rate (residual, potentially hedging)
  - Run across SAE configurations: 2 widths (16k, 65k) x 3 layers (early/middle/late, e.g., layers 6, 12, 18) = 6 configurations per hierarchy type.
  - Include random-baseline control for each configuration.

  **Phase 4 --- Correlation and Decomposition Analysis:**
  - Estimate parent-child co-occurrence frequency from model training data statistics.
  - Test correlation between co-occurrence frequency and absorption rate (Spearman rank correlation, with bootstrap CI).
  - Width-dependence analysis: Compare 16k vs. 65k absorption rates per hierarchy. Width-sensitive component = hedging; width-insensitive component = absorption.
  - Decoder cosine similarity analysis for correlated features to identify hedging signature.

- **Evaluation protocol**:
  - Primary benchmarks: RAVEL (HuggingFace: hij/ravel, 5 entity types, 4-6 attributes each), first-letter spelling task (Chanin et al. implementation)
  - Metrics: Absorption rate (SAEBench fractional), false-negative rate, k-sparse probe F1, LR probe accuracy, decoder cosine similarity
  - Statistical test plan:
    - Bootstrap 95% CIs on all absorption rate estimates (B = 10,000)
    - Paired Wilcoxon signed-rank test comparing absorption rates across hierarchy types within the same SAE (non-parametric, does not assume normality)
    - Spearman rank correlation between co-occurrence frequency and absorption rate
    - Bonferroni correction for multiple comparisons (5 hierarchy types x 6 SAE configurations = 30 comparisons, corrected alpha = 0.05/30 = 0.0017)
    - Two-way mixed ANOVA (hierarchy type x SAE width) with letter/attribute as random effect

- **Ablation schedule**:

  | Ablation | What It Tests | Expected Outcome |
  |----------|---------------|------------------|
  | Remove hierarchy (flat classification) | Is absorption hierarchy-driven? | Absorption rate drops to near-zero for flat tasks |
  | High-frequency vs. low-frequency entities | Does co-occurrence frequency drive absorption? | Higher absorption for high-frequency entities |
  | Width 16k vs. 65k | Width dependence (hedging vs. absorption) | Hedging-attributed FN decreases with width; absorption-attributed FN stable |
  | Layer sweep (6, 12, 18) | Layer dependence of cross-domain absorption | Early/middle layers show absorption; late layers may not (per Chanin et al.) |
  | Chanin IG metric vs. SAEBench projection metric | Method robustness | High agreement (>80%) on first-letter; may diverge on new tasks |
  | Random-baseline control | Is absorption signal real? | Random baseline shows near-zero absorption rate |

- **Control experiments**:
  1. **Random sparse encoder baseline**: Replace SAE encoder with random projection + TopK sparsification. Compute "absorption rate." This establishes the null: any absorption rate below this is indistinguishable from chance.
  2. **Dense probe baseline**: For each hierarchy, compare LR probe on raw activations vs. SAE-based sparse probe. The gap quantifies information loss from the SAE, contextualizing absorption's practical impact.
  3. **Positive control replication**: Replicate Chanin et al.'s first-letter absorption rates on Gemma Scope 16k/65k to confirm our implementation matches published results (within 10% relative error).
  4. **L0-controlled comparison**: For each hierarchy, evaluate absorption at 3 L0 levels to ensure results are not confounded by sparsity choice.

- **Pilot design**: The <15-minute experiment that gives early signal:
  1. Load Gemma-2-2B + Gemma Scope 16k SAE (layer 12) via SAELens (~2 min with cached weights).
  2. Train LR probes for 3 first-letter tasks (s, a, t) and 3 RAVEL tasks (city->country, city->continent, city->language) on model activations (~3 min).
  3. Perform k-sparse probing (k=1..5) on SAE latents for each task (~3 min).
  4. Compute absorption rate using SAEBench projection method for all 6 tasks (~5 min).
  5. Compare first-letter absorption rates to RAVEL absorption rates.
  - **Success criteria**: (a) First-letter absorption rate > 0.10 (confirming replication), AND (b) at least one RAVEL hierarchy shows absorption rate > 0.02 (signal worth investigating further).
  - **Abort criteria**: If LR probe accuracy < 0.80 on all RAVEL tasks, the probes are not capturing the hierarchy in the model's representations, and the absorption metric cannot be meaningfully applied. Pivot to a different set of entity-attribute pairs or a different model layer.

- **Resource estimate**:
  - Pilot: 1 GPU, ~15 min, Gemma-2-2B + 1 SAE
  - Phase 1 (metric validation): 1 GPU, ~20 min (replication + random baseline)
  - Phase 2 (probe construction): 1 GPU, ~30 min (LR probes + k-sparse probes for 5 hierarchy types)
  - Phase 3 (cross-domain measurement): 1 GPU, ~2-3 hours (6 SAE configurations x 5 hierarchies x 2 methods)
  - Phase 4 (analysis): CPU only, ~30 min (correlation analysis, bootstrap CIs, ANOVA)
  - Total: ~4-5 hours of single-GPU compute on Gemma-2-2B. Well within time budget with parallelization across SAE configurations.
  - Model: Gemma-2-2B (2.6B parameters, fits on single A100/A6000 with SAE)
  - Pre-trained SAEs: Gemma Scope (HuggingFace: google/gemma-scope), SAEBench pre-trained SAEs
  - Software: SAELens v6, TransformerLens, sae-spelling (Chanin et al. codebase)

- **Risk assessment**:

  | Risk | Severity | Likelihood | Mitigation |
  |------|----------|------------|------------|
  | LR probes fail to learn RAVEL hierarchies from Gemma-2-2B activations | High | Low | RAVEL was validated on Gemma models; try multiple layers; expand entity set |
  | Absorption signal on RAVEL is indistinguishable from random baseline | Medium | Medium | This is itself a publishable finding; pivot to investigating why |
  | Method disagreement between Chanin IG and SAEBench projection | Medium | Medium | Report both; use agreement rate as a reliability metric |
  | SAEBench SAEs not available for all target configurations | Low | Low | Fall back to Gemma Scope SAEs (400+ available); use SAELens to load |
  | Gemma-2-2B too large for available GPU | Low | Low | Fall back to GPT-2 Small; GPT-2 SAEs are well-characterized |
  | Co-occurrence frequency estimation is noisy | Medium | High | Use multiple estimation methods; treat as ordinal (high/medium/low) rather than continuous |

- **Novelty claim**: This is the first systematic empirical characterization of feature absorption across multiple semantic hierarchy types in real LLM SAEs. All prior absorption work (Chanin et al., SAEBench) focuses exclusively on the first-letter spelling task. By establishing (or refuting) the generality of absorption across entity-attribute hierarchies, knowledge taxonomies, and syntactic tasks, this work provides the empirical foundation needed to (a) prioritize absorption mitigation research, (b) develop hierarchy-aware absorption metrics, and (c) connect absorption to downstream interpretability and safety task performance. The cross-domain taxonomy also enables the first empirical test of the theoretical prediction that absorption severity depends on parent-child co-occurrence frequency.
