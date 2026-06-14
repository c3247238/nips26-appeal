# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024), "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025)** — Defines the canonical absorption metric: k-sparse probing + integrated-gradients ablation + cosine/magnitude thresholds. The gold-standard experimental design for absorption measurement, but restricted to the first-letter spelling task and dependent on multiple threshold choices whose sensitivity has never been systematically characterized. Critical methodological note: the metric conflates several distinct failure modes (probe error, dead latents, genuine absorption) under a single "absorption rate" number. The thresholds (cosine >= 0.025, magnitude gap >= 1.0) are stated without justification and could dramatically affect reported rates.

2. **Karvonen et al. (2025), SAEBench (arXiv:2503.09532)** — 8-metric evaluation suite across 200+ SAEs; absorption is one metric implemented as an extension of Chanin et al. Critical cautionary finding: dead features in ReLU SAEs confounded early absorption measurements. When training was improved, ReLU SAEs showed the HIGHEST absorption levels — the earlier low-absorption finding was an artifact of dead features, not genuine absorption resistance. This is the single most important methodological lesson: any metric comparison MUST control for dead feature ratio. Also reveals inverse scaling: wider SAEs show worse absorption, contradicting naive capacity arguments.

3. **Korznikov et al. (2026), "Sanity Checks for SAEs" (arXiv:2602.14111)** — SAEs recover only 9% of true features in synthetic settings; random baselines match SAEs on interpretability and sparse probing. This forces a critical question: are we measuring genuine absorption or just measuring how poorly SAEs work in general? Our experimental design must include a "random baseline" control for every metric.

4. **SynthSAEBench (arXiv:2602.14687, Feb 2026)** — Synthetic data with known ground-truth feature hierarchies, Zipfian firing distributions, and controlled correlation structure. Logistic probes achieve F1=0.974 while best SAEs substantially underperform. Key for Phase 0 validation: if we cannot detect known-ground-truth absorption in SynthSAEBench, the metric is broken before we apply it to real LLMs.

5. **RAVEL (Huang et al., ACL 2024, arXiv:2402.17700)** — Entity-attribute disentanglement benchmark with interchange interventions. Provides pre-validated entity hierarchies (city-continent, city-country, city-language) with Cause and Isolation scores. Critically, RAVEL evaluates via causal intervention, not just correlational probing — a stronger standard. Key finding from SAEBench integration: RAVEL disentanglement unexpectedly prefers higher L0 (>400), and attribute pairs "country-language" and "latitude-longitude" are inherently entangled even for MDAS — we must exclude these.

6. **Song et al. (2025), "Feature Consistency in SAEs" (arXiv:2505.20254)** — SAE features are inconsistent across training runs; only 30% of features are shared across seeds; TopK SAEs achieve 0.80 PW-MCC consistency. This is a critical confounder for absorption measurement: if absorption patterns differ across SAE training seeds, cross-SAE comparisons are unreliable. We must measure absorption consistency, not just absorption rate.

7. **Engels et al. (2024), "Dark Matter of SAEs" (arXiv:2410.14670)** — ~50% of SAE reconstruction error is linearly predictable from input. This "dark matter" may overlap with absorbed features (parent features that contribute to reconstruction error because they are silently dropped). We should check whether high-EDA latents correlate with dark matter patterns.

8. **Chanin & Dulka (2025), "Feature Hedging" (arXiv:2505.11756)** — Hedging is complementary to absorption: narrow SAEs merge correlated features, while hierarchical structure causes absorption. Their balanced Matryoshka SAE shows reducing absorption can INCREASE hedging. This absorption-hedging tradeoff MUST be measured in any mitigation evaluation — reducing absorption that merely shifts error to hedging is not a genuine improvement.

9. **DeepMind Safety Research (2025), "Negative Results for SAEs"** — 1-sparse SAE probes fail catastrophically at detecting harmful intent while dense linear probes succeed even OOD. Absorption is identified as likely culprit. Strongest practical motivation, but hardest to evaluate: we cannot replicate the exact safety evaluation, but we can correlate absorption rate with probe performance gap.

10. **Unified SDL Theory (arXiv:2512.05534)** — Casts all SDL variants as piecewise biconvex optimization; provides theoretical basis for EDA (encoder-decoder misalignment at partial minima). Feature anchoring proposed as remedy but validated only on synthetic benchmarks. Important for theoretical grounding but NOT empirical evidence until validated on real LLMs.

11. **Korznikov et al. (2025), OrtSAE (arXiv:2509.22033)** — 65% absorption reduction via orthogonality penalty on decoder vectors. Evaluation measures absorption rate, atomicity (clustering coefficient), and uniqueness (% distinct features). Uses decoder-decoder cosine similarity as unsupervised signal — directly relevant as a baseline comparison for EDA.

12. **"Toy Models of Feature Absorption" (LessWrong, Chanin et al., Oct 2024)** — Informal precursor to the NeurIPS paper containing the seed observation: encoder and decoder directions diverge under absorption. Specifically suggests comparing top activations using encoder vs. decoder directions. Comment thread discusses tied SAEs solving absorption in the toy case, implying EDA=0 is the absorption-free baseline. This must be cited and differentiated from EDA formalization.

### Experimental Landscape

**What has been properly tested with adequate controls:**
- Absorption rate on first-letter spelling task across Gemma Scope 16k/65k, Llama 3.2 1B, Qwen2 0.5B (Chanin et al.) — reproducible, consistent, well-controlled.
- SAEBench absorption metric across 200+ SAEs, 8 architectures (Karvonen et al.) — standardized, publicly auditable.
- OrtSAE's 65% absorption reduction on Gemma Scope (Korznikov et al.) — ablation of orthogonality coefficient included.
- Matryoshka SAE's superior absorption scores across SAEBench (Bussmann et al.) — multi-metric evaluation, confirmed by SAEBench.
- Synthetic ground-truth absorption detection via SynthSAEBench — controlled hierarchy confounds.

**What is accepted without rigorous experimental evidence:**
- That first-letter absorption rates generalize to other feature hierarchies. This is universally assumed but has **never been tested**. Every absorption measurement in every published paper uses the first-letter task.
- That wider SAEs should have less absorption based on capacity arguments. Empirical data shows the **opposite** (wider SAEs have MORE absorption), and the confound with correlated L0 has never been disentangled via partial correlation.
- That absorption is independent of SAE training seed. Never tested jointly with Song et al.'s consistency analysis. If 70% of features differ across seeds, "absorbed" features may simply be unstable features.
- That the canonical metric's thresholds (cosine > 0.025, gap >= 1.0) produce stable detection. **Threshold sensitivity has never been analyzed.**
- That dead features do not systematically confound absorption measurement. SAEBench discovered this confound post-hoc and had to revise published results.
- That EDA captures genuine absorption rather than polysemanticity, noise, or other weight-space phenomena. **Completely untested.**

**Where methodological gaps exist:**
1. No absorption evaluation exists for any semantic hierarchy beyond first-letter spelling.
2. No unsupervised absorption detection method has been validated against supervised ground truth.
3. No metric reliability characterization (threshold sensitivity, split-half reliability, random baseline rate) exists for the canonical absorption metric.
4. No partial correlation analysis disentangles SAE width from L0 in absorption scaling.
5. No joint analysis of absorption and feature consistency across training seeds.
6. No measurement of the absorption-hedging tradeoff under matched conditions.

---

## Phase 2: Initial Candidates

### Candidate A: Anatomy of Feature Absorption (EDA/D-EDA + Cross-Domain + Taxonomy + ITAC)

This is the existing front-runner proposal, evaluated here from a pure methodology standpoint.

- **Core hypothesis**: EDA (1 - cos(w_e, d)) provides a weight-only absorption detector with AUROC >= 0.65 against Chanin et al. labels. D-EDA (directional residual decomposition) improves precision by >= 10pp at matched 50% recall. Absorption generalizes across RAVEL hierarchies. Three subtypes (early/late/partial) are mechanistically distinct. ITAC corrects late absorption at inference time.
- **Falsification criterion**: EDA AUROC < 0.60 after polysemanticity filter AND all RAVEL probes fail quality gate AND three subtypes show identical EDA distributions (Kruskal-Wallis p > 0.10). Meeting all three conditions kills the paper.
- **Evaluation protocol**:
  - Primary: Chanin et al. absorption labels as ground truth (first-letter task) on Gemma Scope SAEs
  - Extension: RAVEL entity-attribute hierarchies (city-continent, city-country, city-language)
  - Metrics: AUROC, AUPRC, precision@recall=0.50, all with bootstrap 95% CI (10,000 resamples)
  - Baselines: random EDA (shuffled encoder rows), decoder cosine similarity, dead feature indicator, encoder norm
  - Statistical tests: DeLong test for AUROC comparison, Mann-Whitney U, Kruskal-Wallis + post-hoc Dunn, Bonferroni correction
- **Ablation plan**:
  1. Scalar EDA alone vs. D-EDA (contribution of directional decomposition)
  2. EDA with/without polysemanticity filter (polysemanticity confound magnitude)
  3. ITAC on late-absorbed vs. early-absorbed vs. non-absorbed (subtype-specificity)
  4. EDA across layers (layer-dependence of absorption signal)
  5. Subtype threshold sensitivity (cosine 0.2, 0.25, 0.3, 0.35, 0.4)
  6. Encoder norm as EDA alternative (is alignment the signal, or just encoder magnitude?)
  7. Dead feature stratification (dead / rare / common features separately)
- **Confounders identified**:
  - (C1) Polysemanticity causes EDA > 0 without absorption -> D-EDA decomposition filters this
  - (C2) Dead features have undefined EDA behavior -> exclude dead features (freq < 1e-5), report separately
  - (C3) Probe quality differences across RAVEL hierarchies -> partial correlation with probe accuracy
  - (C4) SAE training seed variance -> check consistency across available SAE variants
  - (C5) Layer-dependent effects -> test across layers 6, 12, 20
  - (C6) Encoder norm rather than alignment could drive EDA signal -> test encoder norm baseline
- **Pilot design**: Load Gemma Scope 16k layer 12. Compute EDA for all latents (~1 min). Load Chanin et al. first-letter labels. Compute AUROC with bootstrap CI. If AUROC > 0.60, proceed. If < 0.55, diagnose. ~15 min total.

### Candidate B: Controlled Absorption Benchmarking (the experiment nobody has run)

This is the "controlled experiment" candidate I propose as the empiricist — testing whether published architecture-specific absorption claims are reproducible under matched conditions.

- **Core hypothesis**: Current claims about architecture-specific absorption mitigation (Matryoshka ~65% reduction, OrtSAE 65% reduction, KronSAE "reduced mean absorption fraction") are NOT reproducible under controlled conditions because they were measured with different base models, layers, L0 ranges, dead feature ratios, and metric implementations. Under matched conditions, the architecture ranking on absorption may change.
- **Falsification criterion**: Architecture rankings are stable (Kendall tau > 0.8 between published claims and our controlled measurement). This would validate the field's existing comparisons despite methodological heterogeneity.
- **Evaluation protocol**:
  - Train 5 architectures (Vanilla L1, TopK, Gated, Matryoshka, BatchTopK) on Gemma 2 2B layer 12 at matched L0 (50, 100, 200) and matched training compute
  - Evaluate all with the SAME metric implementation (SAEBench absorption + Chanin et al. + KronSAE metrics)
  - Control for dead features by reporting absorption including and excluding dead features
  - 3 training seeds per configuration
- **Ablation plan**:
  1. Architecture at matched L0 (isolates architectural effect)
  2. L0 sweep within architecture (isolates sparsity effect)
  3. Dead feature exclusion (isolates dead feature confound)
  4. Metric implementation variant (Chanin vs. SAEBench vs. KronSAE — metric agreement)
- **Confounders identified**:
  - Training compute differences (equalize by training tokens)
  - L0 measurement differences (consistent definition)
  - Dead feature ratio varies dramatically between architectures
  - Metric implementation details can create apparent architecture differences
- **Pilot design**: Train Vanilla L1 and TopK at L0=100, 3 seeds each. Measure absorption. Compare to SAEBench numbers. ~30 min, 1 GPU.

**CRITICAL CAVEAT**: This candidate requires SAE training, conflicting with the "training-free" project constraint. If training is truly off-limits, this candidate must be dropped. However, it represents the most rigorous experimental design possible.

### Candidate C: Metric Reliability Audit for Feature Absorption

A pure measurement-science contribution providing the foundation all absorption research depends on.

- **Core hypothesis**: The Chanin et al. absorption metric has uncharacterized sensitivity to its threshold parameters, and reported absorption rates may vary by more than 50% across reasonable threshold ranges. The metric produces a non-trivial false-positive rate (> 5%) on random baselines.
- **Falsification criterion**: Absorption rate varies < 15% across the full reasonable threshold range AND random baseline rate < 2%. This would validate the metric as robust and well-calibrated.
- **Evaluation protocol**:
  - Threshold sensitivity: cosine x magnitude gap sweep (5x4 = 20 configurations)
  - Random direction baseline: 100 random unit vectors as "probe directions" with the full metric pipeline
  - Split-half reliability: randomly split tokens into two halves, measure absorption rate on each, compute ICC
  - Cross-seed stability: measure absorption on same SAE config with different seeds in the probing step
  - SynthSAEBench validation: apply metric to synthetic data with known ground-truth, report precision/recall/F1
- **Ablation plan**:
  1. Cosine threshold alone (fix magnitude gap at 1.0)
  2. Magnitude gap alone (fix cosine at 0.025)
  3. Number of integrated-gradients steps (10, 50, 100, 200)
  4. k-sparse probing k value (5, 10, 20)
  5. Probe training set size (100, 500, 1000, 5000)
- **Confounders identified**:
  - Integrated gradients stochasticity
  - k-sparse probing hyperparameter sensitivity
  - Dead features creating edge cases
  - Tokenizer differences between models affecting the first-letter task
- **Pilot design**: Load Gemma Scope 16k layer 12. Run Chanin et al. metric at 4 threshold settings + 1 random baseline. If variance > 30%, metric reliability contribution confirmed. ~15 min.

---

## Phase 3: Self-Critique & Adversarial Testing

### Against Candidate A (Anatomy of Feature Absorption)

**Confound attack:**
- The EDA metric measures encoder-decoder misalignment, but polysemanticity is a MAJOR confound. Polysemantic features also have misaligned encoder/decoder because the encoder must "select" among multiple concepts while the decoder must "reconstruct" a mixture. D-EDA's residual decomposition onto the decoder dictionary introduces its own confound: decoder directions are not orthogonal (non-zero pairwise cosine similarities), so sparse regression coefficients beta_k are not uniquely determined. The "absorption signature" (sparse beta, high cosine of active components) vs. "polysemanticity signature" (distributed beta, low cosine) distinction may not cleanly separate the two phenomena in practice.
- Cross-domain generalization via RAVEL introduces probe quality as a confound. If city-continent probes have 90% accuracy and first-letter probes have 98%, the 8% gap creates systematic false-negative rate differences that are NOT absorption. The proposal handles this via partial correlation, but partial correlation assumes a linear relationship between probe quality and absorption rate — the true relationship may be nonlinear (below some threshold, absorption becomes unmeasurable, not just noisier).
- Feature consistency (Song et al.) is an unaddressed confound. If 70% of SAE features differ across training seeds, "absorbed" features in one seed may not be absorbed in another. The proposal does not analyze absorption consistency across seeds — it implicitly assumes absorption is a property of the SAE architecture, not the training run.

**Statistical attack:**
- AUROC of 0.65-0.70 is a modest target achievable by weak correlated signals (e.g., "unusual" features tend to have both higher EDA and higher absorption likelihood). AUPRC is the more informative metric under class imbalance — absorption affects a minority of latents. The proposal should pre-specify expected class prevalence and power analysis.
- The three-subtype taxonomy (Kruskal-Wallis) requires sufficient per-subtype sample sizes. If early absorption is rare relative to late absorption, the test is underpowered. Minimum per-subtype N >= 30 should be pre-specified.
- Partial correlation analysis (H6) has low power: only ~10-20 Gemma Scope configurations per layer. With N=20, only large effects (|r| > 0.45) are detectable at alpha=0.05.

**Benchmark attack:**
- The first-letter spelling task was DESIGNED to exhibit absorption (it has a clean, known hierarchy). RAVEL hierarchies are better but still artificial. Neither tells us about absorption of safety-relevant features, reasoning patterns, or syntactic structures — the domains where absorption matters most for the SAE safety case.
- SynthSAEBench validation is valuable but limited: synthetic data models may not capture specific hierarchy structures of real LLMs.

**Ablation completeness attack:**
- **Missing ablation**: EDA from different training stages of the same SAE (early vs. mid vs. late training). If EDA changes dramatically, the "static weight analysis" interpretation is misleading.
- **Missing ablation**: EDA on SAEs with different random seeds. If EDA is not consistent across seeds, it measures training noise rather than structural absorption.
- **Missing control**: encoder weight norm (||w_e||) as a trivially computable alternative. If encoder norm achieves comparable AUROC to EDA, the encoder-decoder misalignment story is weaker than it appears.
- **Missing baseline**: comparison against decoder-decoder cosine similarity (used by OrtSAE). This is an existing unsupervised signal for absorption-related pathology. EDA must demonstrably outperform it.

**Verdict: STRONG.** The proposal is well-designed with clear falsification criteria, appropriate statistical tests, and reasonable controls. Main weaknesses: (1) EDA's specificity against polysemanticity is theoretically unresolved even with D-EDA; (2) cross-domain generalization relies on RAVEL probes whose quality on Gemma 2 2B at the specific layers is assumed, not guaranteed; (3) ITAC's practical value depends on the prevalence of late-absorbed features, unknown a priori; (4) feature consistency across seeds is not addressed.

### Against Candidate B (Controlled Benchmarking)

**Confound attack:**
- Matching L0 across architectures is non-trivial: TopK fixes k directly; L1 tunes a coefficient; Gated has separate gating. "Matched L0" may mean different things for different architectures.
- Training compute matching is imperfect: Gated SAEs have ~2x parameters at same dictionary size; Matryoshka trains nested levels with different loss weights. "Same training tokens" does not equalize effective training.

**Statistical attack:**
- 3 training seeds is minimal. With 5 architectures x 3 L0 x 3 seeds = 45 runs, compute is ~45 GPU-hours — conflicts with project efficiency constraints.
- Kendall tau on 5 architectures has limited power to detect ranking instability.

**Benchmark attack:**
- Still limited to first-letter spelling task as the only validated absorption benchmark.

**Ablation completeness attack:**
- Solid. Dead feature exclusion ablation is particularly important.
- Missing: cross-metric agreement analysis (are different absorption metrics measuring the same thing?).

**Verdict: MODERATE.** Rigorous and valuable, but (a) requires SAE training against project constraint; (b) high compute cost; (c) "careful replication" framing is lower novelty than novel methods.

### Against Candidate C (Metric Reliability Audit)

**Confound attack:**
- Threshold sensitivity is important and well-constrained. The random direction baseline is an excellent control that should have been in the original Chanin et al. paper.

**Statistical attack:**
- Split-half reliability (ICC) is the correct statistic. SynthSAEBench F1 is the strongest validation possible.

**Benchmark attack:**
- Self-referential (the candidate IS about the benchmark). Main risk: finding the metric is unreliable invalidates all prior absorption research — high impact but potentially unwelcome.

**Ablation completeness attack:**
- Comprehensive for the metric itself.
- Missing: comparison to alternative absorption detection approaches (encoder norm, activation anti-correlation, decoder graph metrics).

**Verdict: STRONG.** Most methodologically sound because it addresses a prerequisite question. However, a pure metric audit paper may be perceived as lower novelty unless paired with a positive contribution.

---

## Phase 4: Refinement

**Dropped:** Candidate B (Controlled Benchmarking). Reasons: (1) Requires SAE training, conflicting with training-free project constraint; (2) high compute cost (~45 GPU-hours); (3) "careful replication" framing is lower novelty. Elements incorporated as controls in Candidate A.

**Strengthened survivor (Candidate A, with elements of C as Phase 0):**

The following refinements address the adversarial attacks:

1. **Metric Reliability as Phase 0 (from Candidate C):** The threshold sensitivity sweep, random direction baseline, and SynthSAEBench validation are incorporated as mandatory Phase 0. This is already in the current proposal but I elevate its importance: Phase 0 should produce a quantitative metric reliability characterization (sensitivity curves, ICC values, SynthSAEBench F1) that is reported in supplementary materials. If the metric proves threshold-sensitive, report results at multiple thresholds, not just Chanin et al. defaults.

2. **Additional controls for EDA validation:**
   - **Encoder norm baseline**: Compute AUROC of ||w_e|| against absorption labels. If comparable to EDA, the encoder-decoder alignment story needs revision.
   - **Random EDA**: Compute EDA with randomly permuted encoder rows (breaking encoder-decoder correspondence). This must yield AUROC ~0.50 to confirm EDA uses alignment information.
   - **Decoder cosine similarity baseline**: Compute max cos(d_j, d_k) for each latent j. This is OrtSAE's unsupervised signal. EDA must outperform it (DeLong test).
   - **Dead feature stratification**: Report all metrics separately for dead (freq < 1e-5), rare (1e-5 to 1e-3), and common (> 1e-3) features.

3. **Strengthened cross-domain protocol:**
   - **DAS ceiling baseline**: Before measuring absorption, confirm DAS score > 80% on target layer. Establishes linear separability as a prerequisite.
   - **Probe-only false-negative baseline**: Measure the probe's FN rate alone (without the attribution step). Separates probe failure from genuine absorption. If probe FN = 20%, measured "absorption rate" of 25% is only 5pp genuine.
   - **Within-accuracy-matched comparison**: Compare absorption across domains only within probe-accuracy-matched subsets (e.g., subsets where probe accuracy > 95%).
   - **Random-direction control per domain**: 100 random unit vectors as probes for each RAVEL hierarchy, not just first-letter.

4. **Strengthened ITAC evaluation:**
   - **Downstream probe recovery**: After ITAC correction, retrain sparse probes on corrected SAE features. Measure whether probe accuracy increases on absorbed tokens specifically. Stronger than just FN rate.
   - **Per-token FVU monitoring**: Report FVU change specifically for tokens where ITAC intervenes, not just globally.
   - **Absorption-hedging tradeoff check**: Verify ITAC does not increase feature hedging (measure hedging metric before/after).

5. **Missing ablations added:**
   - **Encoder norm as EDA alternative** (as specified above).
   - **Multi-seed EDA consistency**: If multiple SAE training seeds available, measure EDA Spearman rho across seeds.
   - **Layer-wise EDA profile**: Test whether EDA signal strength varies systematically with layer depth.
   - **AUPRC as primary metric** alongside AUROC, given class imbalance. Pre-specify expected absorption prevalence.

**Selected front-runner: Candidate A (refined), with Candidate C elements as Phase 0.**

---

## Phase 5: Final Proposal

### Title

**Anatomy of Feature Absorption in Sparse Autoencoders: A Weight-Based Detector, Cross-Domain Characterization, and Mechanistic Taxonomy**

### Hypothesis

**Primary (H2):** Encoder-Decoder Alignment (EDA) achieves AUROC >= 0.65 against Chanin et al. supervised absorption labels on Gemma Scope SAEs, and Directional EDA (D-EDA) improves precision at matched 50% recall by >= 10 percentage points.

**Falsification criterion:** AUROC < 0.60 for both scalar EDA and D-EDA after polysemanticity stratification on first-letter task. This kills the EDA contribution. Cross-domain anatomy and ITAC could independently carry the paper if cross-domain results are strong and ITAC reduces FN rate by >= 20%.

### Method

Training-free analysis of pre-trained Gemma Scope SAEs (16k and 65k widths, layers 6/12/20) on Gemma 2 2B, with GPT-2 Small replication. No SAE retraining required. All code released as SAEBench extension.

### Evaluation Protocol

**Primary benchmark:** Chanin et al. first-letter absorption metric on Gemma Scope SAEs — established, public, reproducible.

**Extension benchmark:** RAVEL entity-attribute hierarchies (city-continent, city-country, city-language) — pre-validated, causal evaluation via interchange interventions.

**Metrics with statistical test plan:**

| Metric | Statistical Test | CI Method | Multiple Comparison |
|--------|-----------------|-----------|---------------------|
| AUROC (EDA vs. labels) | DeLong test vs. baselines | Bootstrap 95% CI, 10,000 resamples | None (primary metric) |
| AUPRC | Bootstrap CI | 10,000 resamples | None |
| Precision at 50% recall | Bootstrap CI | 10,000 resamples | EDA vs. D-EDA comparison |
| Absorption rate by domain | Spearman rho | Permutation test, 10,000 permutations | Bonferroni over 3 RAVEL domains |
| Three-subtype EDA ordering | Kruskal-Wallis + post-hoc Dunn | Bootstrap CI on medians | Holm-Bonferroni for 3 pairwise |
| ITAC FN rate reduction | Paired Wilcoxon signed-rank | Bootstrap CI | None |
| FVU change from ITAC | Paired comparison | Bootstrap CI | None |
| Partial correlation (H6) | Spearman partial | Permutation p-value | None (secondary) |

**Minimum sample sizes:** >= 30 latents per subtype for Kruskal-Wallis; >= 10 SAE configurations per layer for partial correlation; >= 100 tokens per hierarchy for probe training.

**Number of random seeds:** N/A for SAE analysis (pre-trained). Probe training: 3 seeds per probe, report mean +/- std. Bootstrap: 10,000 resamples for all CIs.

### Ablation Schedule

| # | Component Ablated | What It Tests | Expected Outcome |
|---|-------------------|---------------|------------------|
| A1 | Scalar EDA only (no D-EDA) | Value of directional decomposition | AUROC similar; precision at 50% recall drops >= 10pp |
| A2 | EDA without polysemanticity filter | Polysemanticity confound magnitude | AUROC drops 5-10pp; precision drops more |
| A3 | D-EDA with different beta sparsity thresholds | Sensitivity to decomposition parameter | Results stable across beta_0 in {3, 5, 10, 20} |
| A4 | Encoder norm (||w_e||) as EDA alternative | Is alignment the signal, or just encoder magnitude? | Encoder norm AUROC < EDA AUROC by >= 5pp |
| A5 | Random EDA (permuted encoder rows) | Verifies EDA uses alignment information | Random EDA AUROC ~0.50 |
| A6 | Decoder cosine sim baseline (max cos(d_j, d_k)) | Is EDA better than OrtSAE's signal? | EDA AUROC > decoder cosine AUROC (DeLong p < 0.05) |
| A7 | Dead feature stratification (dead / rare / common) | Dead feature confound (SAEBench lesson) | Absorption concentrated in rare features; dead excluded |
| A8 | ITAC on late-absorbed only vs. early-absorbed only | Subtype-specific remediability | ITAC improves FN for late-absorbed; null for early |
| A9 | ITAC without reconstruction residual term | Value of residual information | FN improvement decreases >= 30% relative |
| A10 | Cross-domain with vs. without probe quality matching | Probe quality confound | Absorption rates change in matched vs. unmatched |
| A11 | Subtype threshold sensitivity (cosine 0.2-0.4) | Robustness of taxonomy | Qualitative ordering late > partial > early preserved |

### Control Experiments

| Control | Purpose | Expected Result |
|---------|---------|-----------------|
| Random direction baseline (Phase 0) | False-positive rate of absorption metric | < 5% absorption rate with random "probes" |
| Threshold sensitivity sweep (Phase 0) | Metric robustness | Absorption rate varies < 30% across cosine {0.005-0.10} x magnitude {0.5-2.0} |
| SynthSAEBench validation (Phase 0) | Ground-truth calibration | F1 > 0.70 on known-absorption synthetic features |
| Split-half reliability (Phase 0) | Measurement consistency | ICC > 0.80 between halves |
| Shuffled hierarchy control (Phase 3) | Cross-domain false-positive rate | < 3% absorption rate with randomized parent-child labels |
| Probe-only FN baseline (Phase 3) | Separates probe failure from absorption | Probe FN rate < measured absorption rate; difference = genuine absorption |
| DAS ceiling check (Phase 3) | Confirms hierarchy is linearly separable | DAS > 80% for at least 2 of 3 RAVEL hierarchies |
| Dead feature exclusion (all phases) | SAEBench cautionary finding | Qualitative results unchanged after dead feature exclusion |
| Absorption-hedging tradeoff (Phase 2) | ITAC does not shift error to hedging | Hedging metric does not worsen after ITAC correction |

### Pilot Design

**Pilot 1: Metric validation (15 min, CPU+GPU)**
- Load Gemma Scope 16k layer 12
- Run Chanin et al. metric at 4 threshold settings (cosine: 0.01, 0.025, 0.05, 0.10 at magnitude gap 1.0)
- Run metric with 100 random unit vectors as probe directions
- **Go**: threshold sensitivity < 50% AND random baseline < 10%
- **No-go**: threshold sensitivity > 100% OR random baseline > 20%

**Pilot 2: EDA viability (10 min, GPU)**
- Same SAE: compute EDA for all latents from weights (~1 min)
- Load Chanin et al. first-letter labels
- Compute AUROC with bootstrap CI
- Also compute encoder norm AUROC and random EDA AUROC as sanity checks
- **Go**: AUROC > 0.60 AND encoder norm AUROC < EDA AUROC
- **No-go**: AUROC < 0.55 OR encoder norm AUROC >= EDA AUROC

**Pilot 3: Cross-domain feasibility (10 min, GPU)**
- Load RAVEL city-continent data for Gemma 2 2B layer 12
- Train LR probe (3 seeds)
- Check probe accuracy and DAS score
- **Go**: accuracy > 85% AND DAS > 75%
- **No-go**: accuracy < 75%

### Resource Estimate

| Resource | Quantity | Notes |
|----------|---------|-------|
| Model | Gemma 2 2B (~5GB VRAM) | Via TransformerLens / HuggingFace |
| SAEs | Gemma Scope 16k, 65k (layers 6, 12, 20) | ~12 SAEs total, ~200MB each |
| GPT-2 replication | GPT-2 Small + SAELens SAEs | ~500MB VRAM |
| GPU | 1-2 A100/H100 (16GB+ VRAM) | ~7 GPU-hours total |
| Wall-clock | ~6 hours sequential, ~3.5 hours with 2 GPUs | All individual tasks < 1 hour |
| Software | SAELens, sae-spelling, TransformerLens, RAVEL | All MIT/Apache, pip-installable |
| Storage | ~30GB (SAE weights + activation caches) | |

Target: <=1 hour per individual experiment task. Override: project spec allows this budget.

### Risk Assessment

| Risk | Severity | Likelihood | Mitigation | Empiricist Concern |
|------|----------|------------|------------|-------------------|
| EDA AUROC < 0.60 | High | 35% | D-EDA polysemanticity filter; fallback to cross-domain + ITAC | Theoretical motivation may not translate to empirical signal because real SAEs are not at exact partial minima |
| Polysemanticity dominates EDA signal | High | 40% | D-EDA residual decomposition; stratified analysis | Most likely failure mode; D-EDA effectiveness is unproven |
| RAVEL probes fail quality gate | High | 15% | DAS ceiling check; try multiple layers; fallback to city-continent | Gemma 2 2B may not represent RAVEL entities linearly at layer 12 |
| ITAC over-corrects (FVU > 5%) | Medium | 30% | Conservative correction (only high-D-EDA); per-token FVU | Non-orthogonal decoder can amplify noise |
| Three subtypes not separable | Medium | 25% | Sensitivity analysis on threshold; data-driven clustering fallback | Real absorption may be continuous, not categorical |
| Metric itself is unreliable (Phase 0) | High | 15% | Phase 0 characterizes reliability; if unreliable, report as primary finding | If Phase 0 fails, all downstream results are unreliable |
| Dead features confound results | High | 60% | Explicit exclusion in all analyses; report with/without | SAEBench already found this confound |
| Feature consistency across seeds | Medium | 30% | Use all available variants; report consistency where possible | Absorption may be a seed-dependent artifact |
| Absorption-hedging tradeoff from ITAC | Medium | 20% | Monitor hedging metric before/after ITAC | Reducing absorption that shifts to hedging is not genuine improvement |

### Novelty Claim

The experimental contribution — what specific empirical questions are answered for the first time:

1. **Is the canonical absorption metric reliable?** (Phase 0) — First systematic characterization of threshold sensitivity, random baseline rate, split-half reliability, and SynthSAEBench calibration. This benefits the entire field regardless of other results.

2. **Can absorption be detected from SAE weights alone?** (Phase 1) — First formal metric (EDA/D-EDA) validated against supervised ground truth with AUROC, baselines, and ablations. Differentiated from the informal LessWrong suggestion by formalization, theoretical bounds, D-EDA decomposition, and empirical validation.

3. **Does absorption generalize beyond first-letter spelling?** (Phase 3) — First systematic cross-domain absorption characterization using RAVEL hierarchies. Directly tests the "absorption is general" hypothesis that the entire field assumes.

4. **Are there mechanistically distinct absorption subtypes?** (Phase 2) — First formal taxonomy (early/late/partial) with falsifiable predictions linking geometric properties (EDA signal) to functional properties (ITAC remediability).

5. **Can absorption be corrected without retraining?** (Phase 2) — ITAC is the first training-free post-hoc absorption correction method, applicable to any deployed SAE.

**What would convince a skeptic:** Phase 0 showing the metric is reliable and well-calibrated. EDA AUROC with tight bootstrap CI demonstrating discriminative power above encoder norm and decoder cosine baselines. RAVEL absorption rates significantly above random-direction baseline with probe quality controlled. ITAC reducing FN rate with downstream probe accuracy improvement on the corrected features. The convergence of these independent lines of evidence — metric reliability, weight-only detection, cross-domain generalization, training-free correction — is what makes the case compelling. Any single piece alone is incrementally valuable; the full anatomy constitutes a comprehensive empirical characterization of a phenomenon the field depends on understanding.
