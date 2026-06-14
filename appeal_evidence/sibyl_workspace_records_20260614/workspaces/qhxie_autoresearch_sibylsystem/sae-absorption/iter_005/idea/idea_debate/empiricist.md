# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024), "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025)** -- Defines the canonical absorption metric: k-sparse probing + integrated-gradients ablation + cosine/magnitude thresholds. Validated on Gemma Scope, Llama 3.2, Qwen2 SAEs. Key methodological note: absorption rate 15-35% across architectures, but the metric requires known probe directions and its thresholds (cosine >= 0.025, magnitude gap >= 1.0) have never been formally justified. iter_001 Phase 0 validated threshold robustness (absorption rate stable within 30%), confirming metric usability.

2. **SAEBench (Karvonen et al., 2025, arXiv:2503.09532)** -- 8-metric evaluation suite across 200+ SAEs. Critical finding: dead features in ReLU SAEs initially confounded absorption measurements; after improved training, ReLU SAEs showed HIGHEST absorption. This dead-feature confound is the single most important methodological lesson for any absorption experiment. Also: proxy metrics (CE loss, sparsity) do NOT reliably predict absorption scores. Matryoshka SAEs are the best-performing architecture on absorption + RAVEL + sparse probing + SCR.

3. **SynthSAEBench (arXiv:2602.14687, Feb 2026)** -- Synthetic data with known ground-truth feature hierarchies, Zipfian firing distributions, controlled correlation structure. Logistic probes achieve F1=0.974, best SAEs substantially underperform. iter_001 Phase 0 validated the absorption metric here (F1=0.974 on known-absorption features), confirming metric calibration. Essential for any controlled ablation study.

4. **RAVEL (Huang et al., ACL 2024, arXiv:2402.17700)** -- Entity-attribute disentanglement benchmark with interchange interventions. Evaluates via causal intervention, not just correlational probing. Key SAEBench finding: RAVEL disentanglement unexpectedly prefers higher L0 (>400), diverging from conventional 20-200 range. Attribute pairs "country-language" and "latitude-longitude" are inherently entangled even for MDAS skyline -- MUST be excluded or reported separately.

5. **Chaudhary & Geiger (2024, arXiv:2409.04478)** -- First direct SAE evaluation on RAVEL using GPT-2 small. SAEs struggled to reach the neuron baseline and none approached the DAS skyline. Provides the critical lesson that SAE feature spaces may be fundamentally insufficient for causal disentanglement -- this is the null hypothesis against which any cross-domain absorption improvement must be tested.

6. **Tian et al. (2025, arXiv:2509.23717)** -- Feature sensitivity metric. Frames absorption as a special case of poor feature sensitivity. Many interpretable features have poor sensitivity. Sensitivity captures a dimension missed by existing evaluations. Important implication: "absorbed" features may simply be features with generally low sensitivity, making the absorption label itself questionable. Any absorption study should measure sensitivity as a covariate.

7. **Chanin & Garriga-Alonso (2025, arXiv:2508.16560)** -- Most open-source SAEs have L0 that is too low, causing feature hedging (correlated feature mixing). Feature hedging and absorption are distinct phenomena but both manifest as features failing to fire where they should. Critical confounder: observed "absorption" in practice may be partially L0-induced hedging. No study has systematically disentangled the two.

8. **Song et al. (2025, arXiv:2505.20254)** -- SAE features are inconsistent across training runs; only 30% of features are shared across seeds; TopK SAEs achieve 0.80 PW-MCC consistency. If absorption patterns differ across SAE training seeds, cross-SAE comparisons are unreliable. No prior study has measured absorption consistency across seeds.

9. **Korznikov et al. (2026, arXiv:2602.14111)** -- "Sanity Checks for SAEs": SAEs recover only 9% of true features in synthetic settings; random baselines match SAEs on interpretability and sparse probing. This forces the question: are we measuring genuine absorption or just measuring how poorly SAEs work in general? Every metric must include a random baseline control.

10. **iter_001 Experimental Results (This Project)** -- Three rounds of experiments produced: (a) EDA (encoder-decoder alignment) = 1 - cos(w_e, d) is mathematically identical to SAEBench encoder_decoder_cosine_sim (Pearson r > 0.999), eliminating it as a novel metric; (b) EDA AUROC passes 0.65 threshold at only 2/6 Gemma configs; (c) three-subtype taxonomy shows 72-75% early absorption (decoder-absent) but rests on only 2 configs (one non-significant); (d) ITAC achieves only 3% FN reduction vs. 20% target; (e) cross-domain RAVEL probes were trained on wrong model (Qwen2.5-0.5B projected to Gemma 2B); (f) shuffled hierarchy control correctly falsified H3 (0/9 pass), invalidating the cross-domain existence claim.

11. **iter_001 Reflection & Critic Reports** -- Score 5.5/10. Critical issues: (i) EDA is not novel (identical to SAEBench metric); (ii) Theorem 1 proof is circular; (iii) EDA ordering claim contradicts source data; (iv) taxonomy rests on two configurations, one underpowered; (v) AUPRC not reported for class-imbalanced settings; (vi) "reframes the absorption problem" overclaims from n=2 configs. The most damaging finding: the entire novelty claim rests on a proof sketch that a NeurIPS reviewer would reject "in under 30 seconds."

12. **Anthropic Feature Manifold Scaling (arXiv:2509.02565, Sept 2025)** -- Feature manifolds (multi-dimensional features) cause pathological scaling where dictionary size grows but doesn't proportionally capture more features. Zipfian frequency slopes of -0.7 to -0.74 for Gemma Scope SAEs. Feature absorption complicates this interpretation. Relevant to understanding whether dictionary coverage gaps scale predictably.

### Experimental Landscape

**What we know with high confidence from iter_001:**
- The canonical Chanin et al. absorption metric is robust (threshold sensitivity < 30%, random baseline < 5%, SynthSAEBench F1 = 0.974). Phase 0 is done.
- EDA = 1 - cos(w_e, d) is mathematically identical to SAEBench encoder_decoder_cosine_sim. It is NOT a novel contribution.
- EDA has AUROC > 0.65 in mid-layer, narrow-SAE regimes (L5-16k: 0.698, L12-16k: 0.776), but fails at 4/6 Gemma configs. It is a regime-specific signal, not a universal detector.
- Early absorption (decoder-absent) dominates at ~75% in two Gemma configs at tau=0.3, but this is highly threshold-sensitive (32% at tau=0.2, 95% at tau=0.4) and has n=2 configs.
- ITAC is a near-null result (3% FN reduction vs. 20% target). Structurally limited by early-absorption dominance.
- Cross-domain RAVEL analysis was invalid (wrong-model probes). Shuffled control correctly falsified existence claim.
- GPT-2 L6 exact-label validation (AUROC=0.629, n_pos=67) is the cleanest single data point.

**What iter_001 identified as blocking problems:**
- No Gemma 2B model access for ground-truth label replication
- Theorem 1 is unprovable as stated (circular proof sketch)
- EDA novelty is zero without the theorem
- Taxonomy needs replication on a third model family
- AUPRC is the appropriate metric at low prevalence (0.024%), not AUROC

**What remains genuinely unknown and experimentally testable:**
- Does absorption generalize to entity-attribute hierarchies when probes are trained on the correct model?
- Does early-absorption dominance hold across model families (GPT-2, Llama, Gemma)?
- What predicts absorption severity across different hierarchy types? (Co-occurrence statistics? Hierarchy depth? Feature frequency?)
- Can an unsupervised absorption detector achieve practically useful precision?
- What is the absorption-hedging tradeoff quantitatively?

---

## Phase 2: Initial Candidates

### Candidate A: Absorption Anatomy v2 -- Regime-Conditioned Screening, Correct Cross-Domain Validation, and Multi-Family Taxonomy Replication

This candidate addresses the critical failures of iter_001 by fixing the experimental design rather than the conceptual framework.

- **Core hypothesis**: (H1) Encoder-decoder cosine similarity (the existing SAEBench metric, honestly cited) achieves AUROC >= 0.65 for absorption detection in mid-layer, narrow-SAE regimes across at least 2 model families (Gemma + GPT-2). (H2) Absorption generalizes to RAVEL entity-attribute hierarchies when probes are correctly trained on the target model, with absorption rates exceeding 3x the shuffled-hierarchy random baseline. (H3) Early absorption (decoder-absent) dominates at > 60% across at least 3 SAE configurations from 2 model families.
- **Falsification criterion**: (H1) AUROC < 0.60 in all tested mid-layer narrow-SAE regimes. (H2) No RAVEL hierarchy shows absorption exceeding 3x the shuffled-hierarchy baseline after probe quality gate. (H3) Early absorption fraction < 40% in >= 2/3 of configurations tested. Any one falsification removes the corresponding claim; all three falsified kills the paper.
- **Evaluation protocol**:
  - Primary: Chanin et al. absorption labels on GPT-2 Small (exact labels available, n_pos=67 at L6); Gemma 2B with direct label replication (blocking dependency: model access)
  - Extension: RAVEL city-continent, city-country, city-language on GPT-2 Small (model access guaranteed) and Gemma 2B (if accessible)
  - Metrics: AUROC + AUPRC (primary at low prevalence), precision@50% recall, all with bootstrap 95% CI (10,000 resamples)
  - Statistical tests: DeLong for AUROC comparison, Mann-Whitney U for group separation, Kruskal-Wallis + Dunn for taxonomy, Bonferroni correction
  - Baselines: Random unit vectors, shuffled encoder rows, decoder cosine similarity, encoder norm, dead feature indicator
  - Minimum sample sizes: >= 30 latents per subtype for taxonomy; >= 100 tokens per hierarchy for probe training; >= 3 probe seeds
- **Ablation plan**:
  1. AUROC vs. AUPRC reporting at different prevalence levels (exposes class-imbalance artifact)
  2. Dead feature stratification (dead/rare/common separately, per SAEBench lesson)
  3. Encoder norm as alternative predictor (tests whether alignment or just encoder magnitude drives signal)
  4. Shuffled hierarchy control per RAVEL domain (establishes true false-positive rate)
  5. Probe-only false-negative baseline (separates probe failure from genuine absorption)
  6. Taxonomy threshold sensitivity (tau = 0.2, 0.25, 0.3, 0.35, 0.4)
  7. Taxonomy across model families (GPT-2 L6 + Gemma L12 + one additional)
- **Confounders identified**:
  - (C1) Polysemanticity causes high encoder-decoder misalignment without absorption
  - (C2) Dead features have undefined behavior -- exclude and report separately
  - (C3) Probe quality differences across hierarchies -- partial correlation with probe accuracy
  - (C4) L0 confound with width -- partial correlation analysis (but insufficient L0 variation in canonical SAEs)
  - (C5) Wrong-model probes invalidate cross-domain results (iter_001 lesson) -- must train on target model
  - (C6) Prevalence-dependent metric instability -- use AUPRC as primary at < 1% prevalence
- **Pilot design**: Load GPT-2 Small + SAELens SAEs. Run Chanin et al. exact-label absorption metric at L6. Compute encoder-decoder cosine similarity AUROC and AUPRC. Train RAVEL city-continent probe on GPT-2 activations (3 seeds). Measure probe accuracy. If probe accuracy > 85% and encoder-decoder AUROC > 0.55 on GPT-2, proceed. ~15 min.

### Candidate B: The Absorption-Hedging Unified Phase Diagram

A novel empirical contribution that maps the joint landscape of absorption and hedging as a function of SAE width and L0 -- the experiment nobody has run.

- **Core hypothesis**: Absorption and hedging are not independent failure modes but represent two ends of a capacity-allocation failure continuum. At low width (narrow SAEs), hedging dominates; at high width, absorption dominates; at intermediate widths with correct L0, both are minimized. The transition can be characterized as a "phase diagram" in (width, L0) space with quantitative boundaries.
- **Falsification criterion**: (a) Absorption and hedging rates are uncorrelated across SAE configurations (|rho| < 0.15); (b) No configuration achieves both low absorption AND low hedging; (c) The "phase diagram" has no discernible structure (absorption and hedging show no systematic dependence on width or L0).
- **Evaluation protocol**:
  - Use existing pre-trained SAEs from Gemma Scope (widths: 1k, 4k, 16k, 65k; L0 range: 25-300 where available) and GPT-2 SAELens SAEs
  - Absorption: Chanin et al. metric (validated in Phase 0)
  - Hedging: Chanin & Dulka hedging metric (from feature-hedging-paper codebase)
  - Feature sensitivity: Tian et al. sensitivity metric as covariate
  - Measure all three on identical SAE configurations
  - Statistical tests: Spearman correlation (absorption vs. hedging), 2D kernel density estimation for phase diagram, permutation test for structure
- **Ablation plan**:
  1. Absorption alone vs. hedging alone (are they measuring different things?)
  2. Joint measurement at matched L0 across widths (isolates width effect)
  3. Joint measurement at matched width across L0 settings (isolates L0 effect)
  4. Feature sensitivity as third axis (does low sensitivity subsume both?)
  5. Dead feature ratio as covariate (does dead-feature fraction mediate the absorption-hedging tradeoff?)
- **Confounders identified**:
  - Different metric implementations may capture overlapping signal
  - L0 confound with architecture (TopK fixes L0; L1 tunes a coefficient)
  - Pre-trained SAEs have limited L0 variation within a given width
  - Hedging metric has not been independently validated on SynthSAEBench
- **Pilot design**: Load Gemma Scope 16k and 65k at layer 12. Compute absorption rate (Chanin metric) and hedging score (Chanin & Dulka metric) on identical data. Test correlation. If |rho| > 0.2, phase diagram investigation is warranted. ~20 min.

**CAVEAT**: This candidate relies on the hedging metric codebase being available and functional. Must verify feature-hedging-paper repo is usable.

### Candidate C: Dictionary Coverage as the Dominant Predictor of Absorption -- A Training-Free Analysis

A focused, falsifiable hypothesis directly testing the single most important finding from iter_001: that ~75% of absorption is "early" (decoder-absent, meaning the SAE dictionary never allocated a latent for the parent feature). This candidate turns the iter_001 observation into a properly powered, multi-model empirical study.

- **Core hypothesis**: The fraction of absorbed latents that are "early-type" (no decoder direction within cosine > tau of the parent probe direction) exceeds 60% across multiple SAE configurations, model families, and hierarchy types. Dictionary coverage failure -- not encoder alignment failure -- is the dominant cause of measured absorption. This fraction is predictable from SAE width relative to the number of distinct features the model encodes.
- **Falsification criterion**: Early-type fraction < 40% in >= 2/3 of tested configurations. Alternatively: early-type fraction shows no correlation with SAE width or dictionary utilization rate.
- **Evaluation protocol**:
  - Model families: GPT-2 Small (exact labels), Gemma 2B (if accessible), Pythia-160M/410M (publicly available, SAELens SAEs exist)
  - SAE widths: Multiple per model (e.g., GPT-2: 768x4, 768x8, 768x16; Gemma: 16k, 65k)
  - Layers: Mid-layers (L6 for GPT-2, L12 for Gemma, L4/L8 for Pythia)
  - Taxonomy criterion: max cos(d_k, parent_probe) with threshold tau
  - Report at tau = {0.2, 0.25, 0.3, 0.35, 0.4} to characterize threshold sensitivity
  - Statistical tests: Binomial CI for early-type fraction per config; mixed-effects logistic regression across configs; Spearman correlation between early-type fraction and SAE width
  - Target: >= 8 SAE configurations across >= 3 model families
- **Ablation plan**:
  1. Threshold sensitivity (tau = 0.2-0.4) -- core robustness test
  2. Per-model-family breakdown (is early dominance universal or model-specific?)
  3. Per-width breakdown (does wider dictionary reduce early-type fraction? This tests the coverage hypothesis)
  4. Dead feature exclusion (dead features are trivially "early" -- must exclude)
  5. Dictionary utilization rate (alive features / total features) as predictor
  6. Feature frequency stratification (rare vs. common absorbed features)
  7. Alternative taxonomy criterion: use k-nearest decoder vectors instead of cosine threshold
- **Confounders identified**:
  - (C1) Threshold choice tau dramatically affects early-type fraction (iter_001 showed 32% at tau=0.2, 95% at tau=0.4). Must report full sweep and justify chosen operating point.
  - (C2) Dead features confound "early" classification -- exclude features with activation frequency < 1e-5.
  - (C3) Probe quality: low-quality probes make parent direction noisy, artificially inflating "early" classification. Must gate on probe accuracy > 85%.
  - (C4) Non-orthogonal decoder: cos(d_k, parent_probe) > tau does not guarantee the SAE "knows" the parent feature if the decoder is not orthogonal. Must use multiple cosine thresholds and report sensitivity.
  - (C5) Small n: iter_001 had n=16 (L12-16k) and n=65 (L12-65k). Must achieve n >= 30 per subtype per config for statistical power.
- **Pilot design**: Load GPT-2 Small SAE at layer 6 (exact Chanin labels, n_pos=67). Classify absorbed latents by decoder proximity. Report early-type fraction at 5 threshold values. If early-type > 50% at tau=0.3 with n >= 20 early-type latents, proceed to multi-model expansion. ~10 min.

---

## Phase 3: Self-Critique & Adversarial Testing

### Against Candidate A (Absorption Anatomy v2)

**Confound attack:**
- The encoder-decoder cosine similarity is an existing SAEBench metric. Even with honest citation, a reviewer may perceive this as "empirical characterization of someone else's metric" rather than a novel contribution. The paper needs a unique angle that SAEBench did not provide -- the regime characterization and cross-domain validation are the only differentiators.
- Cross-domain validation via RAVEL introduces the model-access problem. GPT-2 Small is publicly available but small; its representation quality for entity-attribute hierarchies may be weak (Chaudhary & Geiger showed GPT-2 SAEs struggle on RAVEL). If GPT-2 probes are low-quality, the "cross-domain" results are worthless. Fallback models (Pythia, Llama) each have their own SAE availability constraints.
- The 2/6 Gemma config pass rate from iter_001 raises the question: is 2/6 a "regime-specific success" or a "mostly-failing metric"? The framing matters enormously. A reviewer hostile to the work will say "4/6 failure rate" rather than "regime-specific success." Must pre-empt with honest quantitative framing.

**Statistical attack:**
- AUPRC is the correct primary metric at 0.024% prevalence, but expected AUPRC values at such low prevalence will be very small in absolute terms -- likely < 0.05. This looks unimpressive even if statistically significant. Must contextualize against random AUPRC baseline.
- GPT-2 L6 has n_pos=67, which gives bootstrap CI width of ~0.12 for AUROC. Not tight enough for definitive claims. Need larger positive samples.

**Benchmark attack:**
- The first-letter spelling task was DESIGNED to exhibit absorption. It has near-complete co-occurrence (every token has a first letter). Natural hierarchies (entity-type, sentiment-topic) have much sparser co-occurrence and may exhibit qualitatively different absorption patterns.
- RAVEL's causal intervention evaluation is stronger than correlational probing, but SAEBench already showed SAEs perform poorly on RAVEL disentanglement. If SAEs fail RAVEL generally, absorption within RAVEL is measuring failure-within-failure.

**Ablation completeness attack:**
- Missing: comparison of absorption metric with feature sensitivity metric (Tian et al.). If sensitivity subsumes absorption, the absorption metric adds no signal.
- Missing: test of whether L0 confound explains the 4/6 Gemma failures (iter_001 noted insufficient L0 variation).
- Missing: cross-seed absorption consistency. Even 2 seeds per config would address Song et al.'s concern.

**Verdict: MODERATE.** Addresses iter_001 failures honestly but risks being perceived as incremental empirical extension of existing metrics. The cross-domain validation is the unique contribution, but its success depends on model access and probe quality -- both uncertain.

### Against Candidate B (Absorption-Hedging Phase Diagram)

**Confound attack:**
- The hedging metric (Chanin & Dulka, 2025) has been validated in a different experimental context than the absorption metric. Measuring both on the same data is straightforward, but their interaction may be confounded by shared underlying causes (e.g., both are worse at low L0).
- Pre-trained SAEs have limited (width, L0) coverage. Gemma Scope has ~4 width settings with ~2-3 L0 settings each. A "phase diagram" with ~10-12 data points is not a diagram -- it's a scatter plot.
- The hypothesis that absorption and hedging trade off is plausible but has a simpler alternative explanation: both are caused by insufficient dictionary capacity, and wider SAEs reduce hedging (capacity) while potentially increasing absorption (hierarchy). This is descriptive, not mechanistic.

**Statistical attack:**
- With ~10-12 SAE configurations, correlations need |rho| > 0.6 to be significant at alpha=0.05. Only strong effects are detectable.
- 2D kernel density estimation with 12 points will be severely under-determined.

**Benchmark attack:**
- Still limited to first-letter task as the primary evaluation target (hedging metric may not generalize to other hierarchies).

**Ablation completeness attack:**
- Missing: SynthSAEBench validation of the hedging metric (it has not been tested on known-ground-truth data with controlled correlation structure).
- Missing: feature consistency analysis across seeds. If the same latent is "absorbed" in seed A and "hedged" in seed B, the taxonomy is meaningless.

**Verdict: MODERATE.** Novel and valuable framing, but under-powered due to limited (width, L0) coverage of pre-trained SAEs. Requires SAE training for a proper phase diagram (conflicts with training-free constraint). Best as a supplementary analysis within a larger paper.

### Against Candidate C (Dictionary Coverage Dominance)

**Confound attack:**
- The tau threshold sensitivity is the CRITICAL weakness. iter_001 showed: tau=0.2 gives 32% early, tau=0.3 gives 72%, tau=0.4 gives 95%. The "early dominance" claim is a function of threshold choice, not an empirical fact. A reviewer will ask: "What is the correct tau?" and the paper has no principled answer.
- Alternative to cosine threshold: use top-k nearest decoder vectors by cosine. If k=5 and none of the top-5 decoders are close to the parent probe, "early" classification is more robust than any single cosine threshold.
- Dead features are trivially "early" (no decoder is close to anything meaningful). Excluding dead features is necessary but reduces the effective sample size.
- Non-orthogonal decoder means cos(d_k, parent_probe) can be artificially high even for irrelevant decoder vectors (because decoders are not orthogonal). Must control for average decoder-probe cosine, not just max.

**Statistical attack:**
- Mixed-effects logistic regression is the correct statistical model (latents nested within configs, configs nested within models). But the number of configs (~8) and models (~3) means the random-effects structure is thin. May need to use fixed effects for model family.
- n >= 30 per subtype per config is achievable only at wider SAEs (65k has ~65 absorbed latents, 16k has ~16). The 16k SAE is underpowered by design.

**Benchmark attack:**
- Reliance on first-letter task for the initial classification limits generalizability. However, extending to RAVEL hierarchies (where "early" = no decoder near the entity-type probe) would strengthen the claim IF the probes are high-quality.

**Ablation completeness attack:**
- Strong ablation design with 7 components.
- Missing: comparison with unsupervised clustering of decoder vectors to identify "topic areas" covered by the dictionary. If the decoder vector space does not contain a cluster near the parent feature direction, this is a more principled "dictionary coverage gap" detection than max cosine.
- Missing: correlation with dictionary utilization rate (fraction of latents that are alive). If early absorption correlates with low utilization, the prescription is clear: train denser SAEs.

**Verdict: STRONG.** The most falsifiable, testable, and practically impactful hypothesis. The threshold sensitivity is a real weakness but addressable with robust reporting across multiple thresholds and a data-driven threshold selection method (e.g., elbow in the early-type-fraction vs. tau curve). Multi-model replication directly addresses the n=2 weakness of iter_001.

---

## Phase 4: Refinement

**Dropped:** Candidate B (Absorption-Hedging Phase Diagram). Reasons: (1) Under-powered due to limited (width, L0) coverage of pre-trained SAEs (~12 data points is insufficient for a "phase diagram"); (2) requires SAE training for proper coverage (conflicts with training-free constraint); (3) the hedging metric has not been validated on SynthSAEBench ground-truth data. Elements retained: absorption-hedging correlation measurement incorporated as a supplementary analysis in the selected proposal.

**Merged:** Candidates A and C into a unified proposal, with C as the primary contribution and A providing the validation framework. The rationale: Candidate C's "dictionary coverage dominance" hypothesis is the single most impactful and falsifiable claim from iter_001, but it was critically under-powered. Candidate A's cross-domain validation and regime characterization provide the scaffolding for a comprehensive study.

**Strengthened Proposal (A+C merged):**

1. **Primary novelty repositioned to dictionary coverage dominance (from Candidate C).** This is the iter_001 finding with the highest practical impact and the cleanest novelty claim. No prior work has tested whether "early absorption" (decoder-absent) dominates across model families. The prescriptive implication (wider dictionaries > encoder fixes) directly challenges the implicit framing of OrtSAE, ITAC, and MP-SAE literature.

2. **Threshold robustness addressed via multi-threshold reporting + data-driven selection.**
   - Report early-type fraction at tau = {0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45}
   - Plot early-type fraction as a function of tau for each SAE config (sensitivity curve)
   - Use the tau at which early-type fraction stabilizes (inflection point) as the principled operating point
   - Additionally: use k-nearest-decoder alternative (k=3, 5, 10) to verify robustness to threshold choice

3. **Multi-model replication (from Candidate A's framework).**
   - GPT-2 Small L6: exact Chanin labels, n_pos=67 -- cleanest data point
   - GPT-2 Small L8, L10: extend to multiple layers within GPT-2
   - Pythia-160M, Pythia-410M: SAELens SAEs available, expand model family coverage
   - Gemma 2B (if accessible): 16k and 65k SAEs at layers 5, 12
   - Target: >= 10 SAE configurations across >= 3 model families

4. **Cross-domain validation as secondary contribution (from Candidate A).**
   - RAVEL probes trained on GPT-2 Small (guaranteed access, avoid wrong-model probe issue)
   - If GPT-2 probes have accuracy > 85%: measure absorption + early-type fraction on RAVEL hierarchies
   - Shuffled hierarchy control mandatory for every domain
   - Report cross-domain absorption only if shuffled control shows < 5% false-positive rate

5. **Honest encoder-decoder cosine similarity framing (from iter_001 lesson).**
   - Cite SAEBench encoder_decoder_cosine_sim explicitly
   - Contribution: NOT the metric itself, but the systematic empirical characterization of its operating regime (layer, width, model family) and its relationship to dictionary coverage subtypes
   - If a theoretical contribution is desired: replace Theorem 1 with a simpler, provable proposition (e.g., "For a tied SAE with W_enc = W_dec^T, encoder-decoder cosine = 1 for all latents. For an untied SAE, EDA > 0 is a necessary condition for absorption. Proof: if the encoder perfectly detects and the decoder perfectly reconstructs, then the optimal encoder direction IS the decoder direction."). This is trivially true but states the baseline clearly.

6. **Additional controls from Phase 3 critique:**
   - Feature sensitivity (Tian et al.) as covariate -- measure for all absorbed latents
   - Dead feature exclusion with separate reporting
   - AUPRC as primary metric at < 1% prevalence
   - Random baseline (shuffled encoder rows) for every metric
   - Dictionary utilization rate (alive fraction) as predictor of early-type fraction

---

## Phase 5: Final Proposal

### Title

**Dictionary Coverage Gaps, Not Encoder Misalignment, Drive Feature Absorption in Sparse Autoencoders: A Multi-Model Taxonomy and Cross-Domain Characterization**

### Hypothesis

**Primary (H1):** The fraction of absorbed latents classified as "early-type" (no decoder direction within cosine > tau of the parent probe direction, with dead features excluded) exceeds 60% at the data-driven threshold across at least 8 SAE configurations from >= 3 model families. This establishes dictionary coverage failure as the dominant absorption mechanism.

**Secondary (H2):** Encoder-decoder cosine similarity (SAEBench metric) achieves AUROC >= 0.65 for absorption detection in mid-layer, narrow-SAE regimes across >= 2 model families (GPT-2 + at least one of Gemma/Pythia/Llama).

**Tertiary (H3):** Absorption generalizes to RAVEL entity-attribute hierarchies on GPT-2 Small, with absorption rates exceeding 3x the shuffled-hierarchy random baseline, and early-type dominance holds for cross-domain absorbed latents as well.

### Falsification Criterion

**H1 killed if:** Early-type fraction < 40% in >= 5/10 configurations tested at any threshold in {0.2, 0.3, 0.4}. This would mean encoder misalignment, not dictionary coverage, dominates absorption, and the field's focus on encoder-side fixes would be justified.

**H2 killed if:** AUROC < 0.60 in all tested mid-layer narrow-SAE regimes across all model families. This would mean encoder-decoder alignment carries no practically useful absorption signal.

**H3 killed if:** No RAVEL hierarchy shows absorption exceeding 3x shuffled baseline after probe quality gate (accuracy >= 85%). This would mean absorption is specific to the first-letter task's unique co-occurrence structure.

**Paper killed if:** All three hypotheses are falsified simultaneously.

### Method

Training-free analysis of pre-trained SAEs from multiple model families. No SAE retraining. Code released as extension of sae-spelling + SAELens.

**Models and SAEs:**
- GPT-2 Small: SAELens SAEs at layers 6, 8, 10 (multiple widths available)
- Pythia-160M and Pythia-410M: SAELens SAEs at mid-layers
- Gemma 2 2B: Gemma Scope 16k, 65k at layers 5, 12 (requires model access)
- Fallback: Llama 3.2 1B SAELens SAEs if Gemma access remains blocked

**Core analysis pipeline (per SAE configuration):**
1. Load pre-trained SAE weights
2. Compute encoder-decoder cosine similarity for all latents (weight-only, no activations needed)
3. Run Chanin et al. absorption metric to obtain ground-truth absorption labels
4. Compute AUROC + AUPRC of encoder-decoder cosine against absorption labels
5. For absorbed latents: compute max cos(d_k, parent_probe) for all decoder vectors d_k
6. Classify: early (max cosine < tau) vs. late (max cosine >= tau, latent fails to fire) vs. partial
7. Report at tau = {0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45}
8. Compute k-nearest-decoder alternative classification (k = 3, 5, 10)

### Evaluation Protocol

**Primary benchmarks:**
- First-letter spelling task (Chanin et al.) -- established, validated, exact labels available for GPT-2
- RAVEL city-continent, city-country, city-language (GPT-2 Small probes) -- causal evaluation

**Metrics with statistical test plan:**

| Metric | Statistical Test | CI Method | Multiple Comparison |
|--------|-----------------|-----------|---------------------|
| Early-type fraction per config | Binomial exact CI | Wilson interval | None |
| Early-type fraction across configs | Mixed-effects logistic regression | Profile likelihood CI | None |
| Early-type vs. SAE width | Spearman rho | Permutation test (10,000) | None |
| Early-type vs. utilization rate | Spearman rho | Permutation test | None |
| Enc-dec cosine AUROC per config | Bootstrap 95% CI | 10,000 resamples | DeLong for pairwise comparison |
| AUPRC per config (primary at low prevalence) | Bootstrap 95% CI | 10,000 resamples | None |
| Taxonomy KW test (late vs. early EDA) | Kruskal-Wallis + Dunn | Bootstrap CI on medians | Holm-Bonferroni |
| Cross-domain absorption vs. shuffled | Permutation test | 10,000 permutations | Bonferroni over 3 RAVEL domains |
| Sensitivity as covariate | Spearman partial correlation | Permutation p-value | None |

**Minimum sample sizes:**
- >= 30 latents per subtype for taxonomy (KW test power)
- >= 100 tokens per hierarchy for probe training
- >= 8 SAE configurations total
- >= 3 model families
- >= 3 probe seeds per hierarchy
- >= 10,000 bootstrap resamples for all CIs

**Number of random seeds:** Probe training: 3 seeds, report mean +/- std. SAEs: pre-trained (no seed control available). Bootstrap: 10,000 resamples.

### Ablation Schedule

| # | Component Ablated | What It Tests | Expected Outcome |
|---|-------------------|---------------|------------------|
| A1 | Taxonomy threshold tau (7 values) | Robustness of early-dominance claim | Early > 60% at tau in [0.25, 0.35] range; monotone increase with tau |
| A2 | k-nearest-decoder alternative (k=3,5,10) | Threshold-free robustness | Qualitatively same conclusion as cosine threshold at data-driven tau |
| A3 | Dead feature exclusion | SAEBench lesson: dead features are trivially "early" | Early fraction decreases by 5-15pp after exclusion but remains > 60% |
| A4 | Per-model-family breakdown | Is early dominance universal? | Holds in >= 2/3 model families |
| A5 | Per-width breakdown | Does wider dictionary reduce early fraction? | Early fraction decreases with width (tests coverage hypothesis) |
| A6 | Dictionary utilization rate as predictor | Mechanistic link to coverage | Spearman |rho| > 0.4 between utilization and early fraction |
| A7 | Feature frequency stratification | Rare vs. common absorbed features | Early absorption concentrated in rare features |
| A8 | Encoder norm as alternative predictor | Is alignment or magnitude the signal? | Encoder norm AUROC < enc-dec cosine AUROC |
| A9 | AUPRC vs. AUROC at different prevalence | Class-imbalance artifact | AUPRC rankings differ from AUROC rankings at < 1% prevalence |
| A10 | Feature sensitivity as covariate | Does sensitivity subsume absorption? | Partial correlation of enc-dec cosine with absorption, controlling for sensitivity, remains significant |
| A11 | Probe-only FN baseline per domain | Separates probe failure from absorption | Probe FN rate < total measured absorption rate |
| A12 | Shuffled hierarchy control per domain | Cross-domain false-positive rate | < 5% absorption rate with randomized parent-child labels |

### Control Experiments

| Control | Purpose | Expected Result |
|---------|---------|-----------------|
| Random unit vector probes (per domain) | False-positive rate of absorption metric | < 5% absorption rate |
| Shuffled encoder rows for enc-dec cosine | Verifies cosine uses alignment information | AUROC ~0.50 |
| Shuffled hierarchy labels (per RAVEL domain) | Cross-domain false-positive rate | < 5% absorption rate with randomized labels |
| Probe-only FN baseline | Separates probe failure from absorption | Probe FN < measured absorption rate |
| Dead feature exclusion (all analyses) | SAEBench cautionary finding | Qualitative results unchanged after exclusion |
| SynthSAEBench calibration (if taxonomy can be applied) | Ground-truth validation of taxonomy | F1 > 0.70 for early/late classification |

### Pilot Design

**Pilot 1: Taxonomy replication on GPT-2 (10 min, 1 GPU)**
- Load GPT-2 Small + SAELens SAE at layer 6
- Run Chanin et al. exact-label absorption metric
- Classify absorbed latents by decoder proximity at tau = {0.2, 0.25, 0.3, 0.35, 0.4}
- Report early-type fraction with binomial CI at each tau
- **Go**: Early-type > 50% at tau=0.3 with n_early >= 20
- **No-go**: Early-type < 30% at tau=0.3 OR n_absorbed < 20 total

**Pilot 2: Cross-domain feasibility on GPT-2 (15 min, 1 GPU)**
- Train RAVEL city-continent probe on GPT-2 Small layer 6 activations (3 seeds)
- Check probe accuracy
- If accuracy > 85%: run absorption metric on city-continent hierarchy
- Run shuffled control (randomize city-continent labels)
- **Go**: Probe accuracy > 85% AND absorption > 3x shuffled rate
- **No-go**: Probe accuracy < 75% OR absorption < 2x shuffled rate

**Pilot 3: Multi-model feasibility (10 min, 1 GPU)**
- Load Pythia-160M + SAELens SAE at mid-layer
- Run first-letter absorption metric
- If n_absorbed >= 15: classify by taxonomy
- **Go**: Absorption detected, n >= 15
- **No-go**: n_absorbed < 10 (insufficient signal for taxonomy)

### Resource Estimate

| Resource | Quantity | Notes |
|----------|---------|-------|
| Models | GPT-2 Small (~500MB), Pythia-160M/410M (~600MB/1.6GB) | Public, no access issues |
| SAEs | SAELens pre-trained SAEs per model + Gemma Scope (if accessible) | ~200MB each |
| GPU | 1 GPU with >= 16GB VRAM | ~8-12 GPU-hours total |
| Wall-clock | ~8 hours sequential, ~4 hours with 2 GPUs | All individual tasks < 1 hour |
| Software | SAELens, sae-spelling, TransformerLens | All MIT, pip-installable |
| Storage | ~20GB (SAE weights + activation caches) | |

Target: <= 1 hour per individual experiment task.

### Risk Assessment

| Risk | Severity | Likelihood | Mitigation | Empiricist Concern |
|------|----------|------------|------------|-------------------|
| Early-type fraction < 40% on GPT-2 (falsifies H1) | High | 25% | Report honestly as negative result; pivot to "encoder misalignment matters more than thought" | Would invalidate iter_001's most impactful finding |
| GPT-2 RAVEL probes have accuracy < 75% | High | 30% | Try multiple layers; use city-continent (simplest hierarchy); fallback to absorption-only paper | GPT-2 Small may lack representational capacity for entity hierarchies |
| n_absorbed < 20 on Pythia models | Medium | 35% | Use wider SAEs (more latents); try different layers; report available n with power analysis | Small models may not exhibit sufficient absorption for taxonomy |
| Tau threshold sensitivity makes early-dominance fragile | High | 40% | Full 7-point tau sweep + k-nearest-decoder alternative; data-driven tau selection; emphasize sensitivity curve shape rather than single number | This is the most honest threat; the only defense is transparent multi-threshold reporting |
| Reviewer says "this is just empirical characterization of SAEBench metric" | Medium | 50% | Lead with dictionary coverage finding (novel); position enc-dec cosine as known tool applied to new question; emphasize multi-model taxonomy replication | Perception risk; mitigate with framing |
| Dead features inflate early-type fraction | Medium | 60% | Explicit exclusion with separate reporting in every analysis | Known confound from SAEBench |
| Feature sensitivity subsumes absorption signal | Medium | 20% | Partial correlation controlling for sensitivity; if sensitivity dominates, report as finding | Would reframe absorption as subset of sensitivity |
| Cross-seed inconsistency | Medium | 30% | Use multiple SAE variants where available; report consistency | Cannot fully control without training SAEs |

### Novelty Claim

The experimental contribution -- what specific empirical questions are being answered for the first time:

1. **Is dictionary coverage failure the dominant absorption mechanism?** (Primary) First multi-model, multi-family empirical test of whether "early" absorption (decoder-absent) dominates over "late" absorption (encoder-suppressed). iter_001 observed this at n=2 configs, one non-significant. This proposal provides properly powered evidence across >= 10 configs and >= 3 model families. If confirmed, this redirects the field's mitigation strategy from encoder fixes (OrtSAE, ITAC, MP-SAE) to dictionary coverage solutions (wider SAEs, hierarchical training objectives, Matryoshka SAEs).

2. **How does early-type fraction depend on SAE width and dictionary utilization?** (Primary) First empirical test of whether dictionary coverage gaps scale predictably with SAE capacity. If early-type fraction decreases monotonically with width, the prescription is simple: use wider SAEs. If it does not decrease, the problem is structural (hierarchy in the optimization landscape).

3. **Does absorption generalize to entity-attribute hierarchies?** (Secondary) First properly controlled cross-domain absorption measurement with probes trained on the correct model and shuffled hierarchy controls. Directly tests the generalizability assumption that the entire absorption literature relies on.

4. **What is the operating regime for encoder-decoder cosine as an absorption screener?** (Secondary) Systematic characterization of an existing SAEBench metric across model families, layers, and widths, with honest reporting of where it works and where it fails.

**What would convince a skeptic:** The convergence of multi-model taxonomy replication (GPT-2 + Pythia + Gemma), threshold-robust early-dominance finding (stable across multiple tau values AND k-nearest-decoder alternative), and correctly controlled cross-domain generalization (probes on target model + shuffled control) would collectively establish that dictionary coverage gaps are the primary driver of feature absorption. The key evidence is NOT any single number but the consistency pattern across models, widths, layers, and threshold choices. If the pattern breaks (e.g., GPT-2 shows early-dominance but Pythia does not), that is equally informative and would be reported as a model-family-dependent finding.
