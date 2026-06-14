# Result Debate Synthesis: Iteration 5

**Synthesizer**: Result Debate Synthesizer (sibyl-heavy)
**Date**: 2026-04-15
**Iteration**: 5
**Perspectives Analyzed**: Optimist, Skeptic, Strategist, Methodologist, Comparativist, Revisionist

---

## 1. Consensus Map

The following conclusions are shared across all 6 perspectives with high confidence:

### Universal Agreement

1. **The scaling surface (H3) is the single strongest result.** All 6 perspectives rate the GAM interaction (p = 3.11e-15, N = 420, R-squared progression 0.488 -> 0.620 -> 0.693) as strong, robust, and publication-ready. No perspective raises a fatal flaw for this finding. The Comparativist confirms no prior work has constructed a formal 2D absorption surface. The Methodologist rates the baseline comparison (linear -> additive -> interaction) as GOOD. The Skeptic's only concern is the architecture confound (standard vs. JumpReLU), which is addressable in < 15 minutes.

2. **The cross-domain metric (H2) is invalidated.** All 6 perspectives agree that the dominance-based metric producing 100% on shuffled controls is fatal. The measurement does not discriminate real from random hierarchies. The Optimist frames this as a "publishable negative result"; the Skeptic calls it "untested, not partially supported"; all others agree the H2 verdict must be downgraded from "PARTIALLY_SUPPORTED" to "METRIC_INVALIDATED" or "UNTESTED." The 51-85% cross-domain rates are meaningless.

3. **There is a critical data reporting error in the taxonomy correction (H5).** All perspectives that examined the data (Skeptic, Methodologist, Revisionist, Comparativist) independently identified that `final_results_summary.md` reports "corrected rate = 92.3%, delta = 0.0%" while `P4_taxonomy_correction.json` shows corrected rate = 19.2% (delta = -73.1%). This is not a judgment call -- it is a bug in the integration script. The actual corrected Type II rate is 15.4% (down from 88.5%).

4. **The GPT-2 Small model choice fundamentally weakens Phase 2.** Universal agreement that using GPT-2 Small (124M params, 98% dead SAE features) instead of the originally proposed Gemma 2B invalidates the cross-domain contribution as designed. The model lacks sufficient factual knowledge representation and SAE feature coverage.

5. **Writing should proceed now.** The Strategist recommends immediate writing (0 GPU-hours critical path). The Optimist sees two papers' worth of material. The Comparativist positions the confound resolution and scaling surface as independently strong contributions. No perspective argues for a pivot or major delay.

---

## 2. Conflict Resolution

### Conflict A: How strong is the causal claim for H1?

**Optimist position**: The suppression effect (sparse probing r strengthening from -0.664 to -0.746 after L0 control) is "the single most compelling result." Full mediation for SCR and TPP, Rosenbaum Gamma = 2.65, and Bradford Hill 3 strong / 5 moderate criteria collectively establish a strong causal case.

**Skeptic position**: The within-width matching null (all 3 within-width strategies fail, p > 0.10 for every metric) "directly contradicts the headline claim." The Rosenbaum Gamma = 2.65 is computed on cross-width pairs that are NOT width-balanced, making it inflated. The proportion mediated for sparse probing (4.785) is "mathematically incoherent." The sample size (n = 48) with 50+ tests and no multiple comparison correction is problematic.

**Revisionist position**: "Confirmed with caveats" at 70% confidence. The three questions -- does absorption correlate with quality (yes), does it cause quality degradation independent of width (uncertain), does it generalize (unknown) -- were conflated.

**My judgment**: The Skeptic raises the most consequential point. The within-width matching null is not a minor caveat -- it is substantive evidence that the absorption-quality association may be entirely driven by cross-width differences. However, the Skeptic does not adequately address the suppression effect, which is robust to this critique: L0 control strengthening the correlation is a property of the pooled sample that does not require within-width significance. The correct framing is:

- **Absorption is robustly associated with quality** (partial r = -0.746, survives L0/width/layer control) -- this is a STRONG claim.
- **Whether this association is causal and independent of width** is INCONCLUSIVE -- the within-width null prevents strong causal language.
- **The mediation analysis is valid for SCR and TPP** but must exclude sparse probing (proportion mediated = 4.785 is uninterpretable due to near-zero total effect).

**Resolution**: Downgrade H1 from "SUPPORTED" to "ASSOCIATION_CONFIRMED, CAUSATION_INCONCLUSIVE." Lead with the suppression effect as the headline finding (it is surprising and robust), but explicitly present the within-width null as a boundary on the causal interpretation. Drop "causal chain" from the paper title.

### Conflict B: What is the correct absorption prevalence?

**Optimist**: Reports 73.1% Chanin-validated absorption alongside the corrected 19.2%.

**Skeptic**: Identifies the reporting error (92.3% -> 19.2%) but calls it a "fatal flaw" requiring correction before writing.

**Revisionist**: Frames the 92.3% -> 19.2% correction as "arguably the most consequential result" that should be promoted to a headline finding.

**Comparativist**: Labels it a MODERATE contribution (1-5%).

**My judgment**: The Revisionist is correct that this is more important than the other perspectives credit. A 73.1 percentage point correction to a published number is significant. However, the two numbers (19.2% magnitude-based, 73.1% Chanin false-negative-based) measure different things. The paper should report BOTH:

- **19.2%**: The corrected Type II classification rate using proper non-letter-context baselines. This measures how often a magnitude-ratio criterion detects absorption. The original 92.3% was inflated by the n_comparison_tokens=0 fallback.
- **73.1%**: The Chanin metric rate measuring false-negative detection (whether a letter's activation pattern shows signs of feature absorption). This is the more semantically meaningful number.

**Resolution**: Report the taxonomy correction as a significant methodological contribution. The canonical prevalence number should be 73.1% (Chanin metric) with 19.2% as the conservative lower bound from magnitude-based classification. The 92.3% original figure was an artifact.

### Conflict C: Is the Rosenbaum Gamma = 2.65 reliable?

**Optimist**: "Strong robustness" exceeding the conventional 2.0 threshold.

**Skeptic**: "Cherry-picking the favorable number" because within-width Gamma = 1.0 (non-significant).

**My judgment**: The Skeptic is right. The Mahalanobis-matched Gamma = 2.65 conflates width effects with absorption effects. It should be reported alongside the within-width null, NOT as the headline sensitivity metric. The honest statement is: "Cross-width Rosenbaum analysis suggests robustness to hidden confounders (Gamma = 2.65), but within-width matching fails to detect any absorption-quality association, indicating that the cross-width result may be driven by width differences."

**Resolution**: Report both Gamma values transparently. Do not claim "strong robustness" without the within-width qualification.

### Conflict D: Should Phase 2 be re-run on Gemma 2B before writing?

**Strategist**: Do NOT pursue Gemma 2B (deferred). Write now. Optionally run IG-based method on GPT-2 in parallel.

**Comparativist**: Re-run on Gemma 2B is "CRITICAL" Priority 1, could elevate from workshop to main conference.

**Methodologist**: Preferred action is Gemma 2B; fallback is reframing as diagnostic.

**My judgment**: The Strategist is correct about the immediate action. Writing should not be gated on Gemma 2B access, which is uncertain. However, the Comparativist is correct about the impact assessment -- the current Phase 2 results are too weak for a main conference submission. The compromise:

**Resolution**: Begin writing immediately with the current three-pillar structure (confound resolution, scaling surface, cross-domain diagnostic). In parallel, attempt IG-based absorption on GPT-2 Small (2-3 GPU-hours). If a Gemma 2B HF token becomes available, run Phase 2 on Gemma 2B as a high-priority enhancement. Do not delay the paper for this.

---

## 3. Result Quality Score

### Score: 6.5 / 10

**Justification**:

| Component | Score | Weight | Rationale |
|-----------|-------|--------|-----------|
| H1: Confound resolution | 7/10 | 30% | Strong association evidence (suppression effect, partial r = -0.746). Mediation valid for 2/4 metrics. Within-width null prevents strong causal claims. n = 48 is limited. |
| H2: Cross-domain absorption | 2/10 | 20% | Metric invalidated by shuffled controls. GPT-2 Small was the wrong model. No valid cross-domain measurement achieved. |
| H3: Scaling surface | 9/10 | 30% | Excellent: N = 420, p = 3.11e-15, clear model progression, actionable phase boundary. Only concern is architecture confound (easily addressed). |
| H5: Taxonomy correction | 6/10 | 10% | The correction is real and significant (73.1pp drop), but the reporting error in integration files undermines trust. Chanin 73.1% is credible. |
| Methodology / novelty | 7/10 | 10% | Epidemiological methods genuinely novel in SAE literature. Bradford Hill, Rosenbaum, mediation all applied for the first time. Reproducibility 3.6/5. |

**Weighted score**: 0.3(7) + 0.2(2) + 0.3(9) + 0.1(6) + 0.1(7) = 2.1 + 0.4 + 2.7 + 0.6 + 0.7 = **6.5**

The score is dragged down primarily by the failed Phase 2 (H2). Without Phase 2, the remaining contributions (H1 + H3 + H5) would score approximately 7.5/10.

---

## 4. Key Findings

1. **L0 is a suppressor, not a confounder, for the absorption-quality relationship in sparse probing.** Adding L0 as a covariate strengthened the partial correlation from r = -0.664 to r = -0.746 (p = 1.16e-09). This reversal of the expected confounding direction is the most surprising and informative finding: the "naive" absorption-quality correlation was an underestimate, not an overestimate.

2. **The absorption scaling surface has a highly significant width-L0 interaction (p = 3.11e-15, N = 420).** Absorption depends on the joint configuration of width and L0, not either alone. The practical implication: absorption increases dramatically at high width (1M) with low L0 (< 6.5), but this width effect vanishes at high L0. A phase boundary exists at log2(L0) in [2.7, 3.8].

3. **The published 92.3% absorption rate is inflated by a factor of ~5x.** The corrected Type II rate is 19.2% using proper non-letter-context baselines. The Chanin false-negative metric detects absorption in 73.1% of letters, providing a more reliable estimate. The original inflation was caused by the n_comparison_tokens=0 fallback defaulting to a biased global baseline.

4. **Cross-domain absorption measurement on GPT-2 Small fails entirely.** The dominance-based metric cannot distinguish real from shuffled knowledge hierarchies (100% on both). The cosine-calibrated metric finds 0%. GPT-2 Small with 98% dead SAE features lacks the capacity for meaningful knowledge-feature absorption measurement. Whether absorption generalizes beyond spelling remains an open question.

5. **Mediation analysis supports L0 -> Absorption -> Quality for SCR and TPP, but not for sparse probing or unlearning.** Full Baron & Kenny mediation is achieved for 2/4 metrics (SCR Sobel p = 2.94e-04; TPP Sobel p = 3.73e-02). Sparse probing mediation is uninterpretable due to opposing causal pathways (proportion mediated = 4.785). The dual-pathway finding (L0 simultaneously helps quality directly and hurts quality through absorption) is itself scientifically interesting.

---

## 5. Methodology Gaps

### Critical (Must Fix Before Writing)

1. **Fix the taxonomy reporting error.** The integration script incorrectly propagates 92.3% as the corrected rate. The actual corrected rate is 19.2%. This is a data integrity issue, not a judgment call. Estimated fix time: < 1 hour.

2. **Resolve the phase_boundary_detected inconsistency.** P3_scaling_surface.json reports `true` (443 ridge points); final_results.json reports `false`. One of these is wrong.

3. **Resolve mediation count inconsistency.** P1_synthesis_summary.md says "2/4 full mediations"; final_results.json says "n_full_mediations: 0." The raw data shows SCR and TPP achieve full mediation.

### High Priority (Should Fix During Writing)

4. **Add architecture-specific GAM analysis.** Test whether the interaction term holds within standard SAEs alone (n = 360) and JumpReLU alone (n = 54). Zero-GPU, 15-minute analysis. Critical for ruling out architecture confound.

5. **Compute power analysis for within-width matching.** Report the minimum detectable effect size at 80% power for n = 15-18 per width stratum. This determines whether the within-width null is substantive evidence against causation or simply low-powered.

6. **Add sensitivity check for 6 excluded canonical SAEs.** Report whether re-including these SAEs (with imputed L0 or as a separate analysis) changes Phase 1 conclusions.

### Desirable (For Paper Strengthening)

7. **Layer-specific mediation analysis.** Given that layer is identified as the primary suppressor for SCR, running mediation within layers would strengthen the causal argument.

8. **Explicit SEM for dual-pathway L0 decomposition.** The finding that L0 has opposing direct and indirect effects on sparse probing quality deserves formal structural equation modeling.

9. **Multiple comparison correction.** Apply FDR correction across the 50+ tests in Phase 1. Report which findings survive and which become marginal.

---

## 6. Competitive Position

### Versus SOTA (from Comparativist)

| Dimension | Our Work | Best Existing | Assessment |
|-----------|----------|---------------|------------|
| Confound-controlled absorption-quality analysis | r = -0.746 after L0 control (n = 48) | Chanin et al. r = -0.595 uncontrolled (n = 54) | **STRONG novelty**: first confound-controlled analysis, first mediation/Rosenbaum/Bradford Hill in SAE evaluation |
| Absorption scaling surface | GAM interaction p = 3.11e-15, N = 420 | Chanin Figs 9b/c (visual only) | **STRONG novelty**: first formal 2D surface with interaction testing |
| Cross-domain absorption | Metric fails on GPT-2 Small | No prior attempt | **MARGINAL**: informative negative on wrong model |
| Absorption prevalence correction | 19.2% (magnitude) / 73.1% (Chanin) vs. original 92.3% | 92.3% (Chanin et al.) | **MODERATE**: important calibration of a published number |
| Absorption mitigation | None (characterization only) | ATM: 20x reduction, OrtSAE: 65% reduction, Matryoshka: positive scaling | **NOT COMPETING**: different contribution type |

### Concurrent Work Threat: LOW

No concurrent paper (Oct 2025 - Apr 2026) addresses the same three questions. The closest is SynthSAEBench (Feb 2026), which studies absorption in synthetic settings but does not perform causal analysis or map the scaling surface. Masked Regularization (Apr 2026) is a mitigation method, complementary to our characterization.

### Venue Assessment

- **With current results**: Strong workshop submission (NeurIPS/ICML MechInterp workshop, 70% acceptance). Borderline for AAAI/EMNLP main (50%).
- **With Gemma 2B Phase 2 fix**: Competitive for NeurIPS/ICML main (35%) or ICLR main (30%).
- **Minimum viable paper**: The confound resolution + scaling surface form a solid two-contribution paper even without cross-domain results.

---

## 7. Hypothesis Update

### Hypotheses That Survived

| Hypothesis | Status | Revision Needed |
|-----------|--------|-----------------|
| H1: Absorption is associated with quality degradation after L0 control | **Confirmed** | Weaken from "causal" to "robust association." Add within-width null as explicit caveat. |
| H3: Width and L0 interact nonlinearly in determining absorption | **Confirmed** | Add architecture ablation. Consider reframing phase boundary range. |
| H5b: The 92.3% taxonomy rate is inflated | **Confirmed** (strongly) | The correction is far larger than expected (73.1pp vs. predicted < 12.3pp). |

### Hypotheses That Need Revision

| Hypothesis | Original Claim | Revised Claim |
|-----------|---------------|---------------|
| H1 (causal) | "Absorption causally mediates L0 -> quality" | "Absorption is robustly associated with quality after confound control; full mediation confirmed for SCR and TPP but causal interpretation limited by within-width null" |
| H2 (cross-domain) | "Absorption generalizes to knowledge hierarchies at 10-35%" | "Cross-domain absorption measurement is untested due to metric failure on GPT-2 Small; requires Gemma 2B replication" |
| H5a (taxonomy correction) | "Correction will reduce rate to < 80%" | "Correction reduced the magnitude-based rate to 19.2%, far more than predicted; Chanin metric gives 73.1% as alternative prevalence estimate" |

### Hypotheses Falsified or Untested

| Hypothesis | Status |
|-----------|--------|
| H2: Cross-domain absorption exists at 10-35% | **Untested** (metric invalidated, not hypothesis falsified) |
| H4: Early layers show Type I dominance | **Not tested** in this iteration |

### New Hypotheses Generated (from Revisionist)

1. **Model-Scale Threshold**: Cross-domain absorption requires >5% alive SAE features and >85% probe accuracy.
2. **Suppression Generality**: The L0 suppression effect replicates across 2+ additional quality metrics on the 420-SAE dataset.
3. **Bimodal Absorption**: Absorption prevalence follows a bimodal distribution (present/absent), not a continuous gradient, consistent with the hurdle model finding.

---

## 8. Action Plan

### Verdict: **PROCEED** to writing

The evidence supports proceeding to paper writing immediately. Two strong contributions (confound resolution, scaling surface) are publication-ready. The failed Phase 2 constrains the paper's scope but does not prevent publication.

### Immediate Actions (Before Writing)

| # | Action | Priority | Effort | Blocker? |
|---|--------|----------|--------|----------|
| 1 | Fix taxonomy reporting error in final_results_summary.md and final_results.json | **CRITICAL** | 30 min | YES -- data integrity |
| 2 | Fix phase_boundary_detected inconsistency | **CRITICAL** | 15 min | YES -- data integrity |
| 3 | Fix n_full_mediations count in final_results.json | **CRITICAL** | 15 min | YES -- data integrity |

### During Writing (Parallel)

| # | Action | Priority | Effort | GPU |
|---|--------|----------|--------|-----|
| 4 | Architecture-specific GAM ablation (standard-only, JumpReLU-only) | **HIGH** | 15 min | 0 |
| 5 | Power analysis for within-width matching | **HIGH** | 30 min | 0 |
| 6 | Sensitivity check for 6 excluded canonical SAEs | **MEDIUM** | 30 min | 0 |
| 7 | IG-based absorption on GPT-2 Small knowledge hierarchies (optional) | **MEDIUM** | 3-5h | 2-3h |
| 8 | FDR correction across Phase 1 tests | **MEDIUM** | 30 min | 0 |

### Paper Structure

- **Contribution 1**: Confound resolution via epidemiological methods -- suppression effect headline, mediation for SCR/TPP, within-width null as honest caveat
- **Contribution 2**: Absorption scaling surface with nonlinear interaction -- N = 420, phase boundary, practical guidance
- **Contribution 3**: Cross-domain diagnostic + taxonomy correction -- metric limitation, GPT-2 Small negative result, prevalence recalibration from 92.3% to 73.1%/19.2%

### Deferred to Next Iteration

- Gemma 2B Phase 2 replication (conditional on model access)
- Multi-model validation of confound resolution (GPT-2 SAEs from SAEBench)
- SEM for dual-pathway L0 decomposition
- Hurdle model extension to 420-SAE dataset
- Immunodominance R* analysis

### Risk Mitigation

| Risk | Probability | Mitigation |
|------|------------|------------|
| Reviewers reject "causal" language for H1 | 60% | Pre-emptively weaken to "robust association"; present within-width null transparently |
| Reviewers dismiss Phase 2 as inadequate | 40% | Frame as methodological contribution (metric validity gap); acknowledge model limitation |
| Architecture confound invalidates H3 interaction | 10% | Run architecture ablation before submission; likely survives given N = 360 standard SAEs |
| Taxonomy reporting error undermines credibility | 80% if unfixed | Fix immediately -- this is the single fastest credibility win |
