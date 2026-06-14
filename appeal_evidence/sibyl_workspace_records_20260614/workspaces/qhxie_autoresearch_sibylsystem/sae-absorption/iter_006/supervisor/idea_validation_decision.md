# Idea Validation Decision

## Pilot Evidence Summary

All 17 planned experiments completed successfully. The evidence paints a mixed but informative picture:

### First-Letter Baseline (Validation)
- Aggregate absorption rate: **13.4%** (95% CI: [7.2%, 18.1%]) on Gemma 2 2B, L12-16k
- Within published 15-35% range (lower end)
- Mean probe F1: 0.565 (only 4/25 letters pass the F1 > 0.85 gate)
- 11 letters show absorption > 10%: C(40%), S(40%), I(40%), L(36%), E(32%), P(24%), D(16%), O(16%), T(16%), F(12%), U(12%)
- **CRITICAL CONTROL FAILURE**: Shuffled label control = 59.6%, random probe control = 9.2% -- both far above expected ~0%

### Cross-Domain: City-Country
- Absorption rate: **0.0%** (9/28 countries pass probe gate)
- Mean probe F1: 0.602
- Shuffled control: 10.3%, random control: 19.0%

### Cross-Domain: City-Continent
- Absorption rate: **6.5%** (CI: [0%, 11.5%]) -- 4/6 continents pass probe gate
- Mean probe F1: 0.795
- Shuffled control: 45.2% (higher than signal!)

### Cross-Domain: City-Language
- Absorption rate: **6.6%** (CI: [0%, 4.3%]) -- 11/18 languages pass probe gate
- Mean probe F1: 0.745
- Shuffled control: 18.0%

### Cross-Domain: Animal-Class
- Absorption rate: **1.4%** (CI: [0%, 3.6%]) -- 4/6 classes pass probe gate
- Mean probe F1: 0.696
- Shuffled control: 39.3% (far higher than signal!)

### Unsupervised Detection Pipeline
- Best Spearman rho against gold standard: **-0.10** (full pipeline), **-0.13** (n_matching_pairs)
- Best AUROC (strict): 0.47
- Only 6/25 letters matched by unsupervised pairs
- ITAC candidate median: 1.35 vs random median: 1.14 (no significant separation, MW p > 0.05)
- **VERDICT: FAIL** -- does not meet minimum rho > 0.3 threshold

### Successive Refinement (CMI)
- Mann-Whitney U test: p = 0.247 (not significant at 0.05)
- Cohen's d = 0.33 (small effect, well below target 0.5)
- Spearman rho (CMI vs absorption rate): 0.14 (p = 0.51)
- **VERDICT: FAIL** -- CMI does not discriminate absorbed from non-absorbed letters

### Confound Decomposition
- 34 SAEs evaluated across Gemma Scope configurations
- L0 is dominant confound: partial r = -0.494 (p = 0.003, FDR-significant)
- 2/4 metrics retain |partial_r| > 0.2 (H4 technically met)
- Absorption decomposition: 96.9% hierarchy-driven, 3.1% hedging, 0% reconstruction error
- Key finding: absorption is predominantly driven by hierarchy structure, not L0 confounds

### Scaling Surface
- GAM pseudo-R2: 0.85 (good fit)
- OLS interaction (width x L0): F = 37.75, p = 1.2e-6 (significant)
- GAM interaction test: p = 1.0 (non-significant -- regularization absorbed the interaction)
- L0 correlation: rho = -0.457 (p = 0.007, significant)
- Width correlation: rho = 0.308 (p = 0.076, not significant)

### Ablations
- Threshold sensitivity: CV computed across 5x4 grid (20 cells)
- Probe sparsity: tested k={1,3,5,10}
- Unsupervised components: no component achieves rho > 0.3 independently

---

## Decision Matrix

### Candidate: `cand_cross_domain_unsupervised` (Front-Runner)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | Cross-domain absorption rates are 0-6.6%, far below the >= 10% target for H1. The first-letter baseline works (13.4%) but controls are broken (shuffled = 59.6%). Net signal vs shuffled is NEGATIVE in all domains. |
| Hypothesis survival | 0.25 | 1 | H1: BORDERLINE FALSIFIED (city-country = 0%, animal = 1.4%, language/continent ~6.5% but all below 10% target). H3: FALSIFIED (rho = -0.10, far below 0.3 threshold). H6: FALSIFIED (p = 0.25, d = 0.33). H4: technically met but with degenerate quality proxies. H5: mixed (OLS significant, GAM not). |
| Path to full result | 0.20 | 2 | The cross-domain characterization provides a legitimate negative result (absorption does not generalize at comparable rates). But the broken controls undermine the entire measurement validity. The unsupervised pipeline has completely failed. 3/6 primary hypotheses are falsified. |
| Novelty (from report) | 0.15 | 3 | Novelty score 7/10. Cross-domain absorption on Gemma 2 2B is genuinely novel. The negative result is informative. However, the unsupervised detection (core novelty claim) failed completely, removing the strongest novel contribution. |
| Resource efficiency | 0.10 | 4 | All experiments completed efficiently within time budget. GPU costs were minimal (single A100, ~3 hours total). |

**Weighted Score: 0.30(2) + 0.25(1) + 0.20(2) + 0.15(3) + 0.10(4) = 0.60 + 0.25 + 0.40 + 0.45 + 0.40 = 2.10**

### Candidate: `cand_decomposition_reframing` (Backup)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | The confound decomposition results (96.9% hierarchy-driven) and threshold sensitivity data are solid. The hedging-absorption tradeoff data exists. |
| Hypothesis survival | 0.25 | 3 | H_alt1 (hierarchy-driven < 50% of total): FALSIFIED -- 96.9% is hierarchy-driven. But H_alt2 (metric sensitivity CV > 0.5) and H_alt3 (model-faithful absorption) are untested. |
| Path to full result | 0.20 | 2 | Chanin et al. (2025) already published hedging-absorption decomposition, significantly reducing novelty. The causal patching angle remains novel but is complex to execute. |
| Novelty (from report) | 0.15 | 2 | Novelty score 6/10. Chanin et al.'s hedging paper partially anticipates this candidate. |
| Resource efficiency | 0.10 | 3 | Would require activation patching experiments not yet started. |

**Weighted Score: 0.30(3) + 0.25(3) + 0.20(2) + 0.15(2) + 0.10(3) = 0.90 + 0.75 + 0.40 + 0.30 + 0.30 = 2.65**

### Candidate: `cand_impossibility_theorem` (Backup)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot data for this theoretical candidate. The scaling surface GAM fit is supportive but indirect. |
| Hypothesis survival | 0.25 | 2 | Theoretical landscape is now crowded (Cui et al. 2025, Tang et al. 2025 provide competing results). |
| Path to full result | 0.20 | 1 | Novelty report recommends "drop" due to crowded theoretical landscape. |
| Novelty (from report) | 0.15 | 2 | Novelty score 5/10. Multiple competing theoretical papers already published. |
| Resource efficiency | 0.10 | 3 | Low compute cost (theoretical + toy model). |

**Weighted Score: 0.30(1) + 0.25(2) + 0.20(1) + 0.15(2) + 0.10(3) = 0.30 + 0.50 + 0.20 + 0.30 + 0.30 = 1.60**

### Candidate: `cand_successive_refinement` (Backup)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | CMI estimation FAILED to discriminate absorbed vs non-absorbed letters (p = 0.25, d = 0.33). The core mechanism of this candidate does not work. |
| Hypothesis survival | 0.25 | 1 | H_sr1 (CMI discriminates): FALSIFIED. The Markov chain violation shows no significant difference between absorbed and non-absorbed features. |
| Path to full result | 0.20 | 1 | Without discriminative CMI signal, there is no path to the theoretical contribution. |
| Novelty (from report) | 0.15 | 4 | Novelty score 8/10 (highest). The idea itself is novel, but the negative empirical result eliminates publishability. |
| Resource efficiency | 0.10 | 5 | Already completed. Minimal additional cost. |

**Weighted Score: 0.30(1) + 0.25(1) + 0.20(1) + 0.15(4) + 0.10(5) = 0.30 + 0.25 + 0.20 + 0.60 + 0.50 = 1.85**

---

## Decision Rationale

### Evidence Synthesis

The experimental evidence reveals systemic problems that undermine all candidates:

1. **Broken controls invalidate the absorption metric on this model/SAE**: The shuffled label control yields 59.6% absorption on first-letter (expected ~0%), and the random probe control yields 9.2%. In EVERY cross-domain experiment, the net signal vs shuffled control is NEGATIVE. This means the measured "absorption" rates are LOWER than what random label assignment produces. This is not a signal -- it is noise.

2. **Probe quality is inadequate across all domains**: Only 4/25 letters pass the F1 > 0.85 gate on first-letter. Mean probe F1 ranges from 0.565 (first-letter) to 0.795 (city-continent). The probe quality gate -- which the proposal itself pre-specified as mandatory -- is failing systematically.

3. **The unsupervised detection pipeline has completely failed**: rho = -0.10, AUROC = 0.47 (below chance). No component achieves rho > 0.3. The ITAC signal does not separate candidate pairs from random pairs. This was the core methodological novelty of the front-runner.

4. **The successive refinement theoretical framing has failed**: CMI does not discriminate absorbed from non-absorbed features (p = 0.25). This eliminates the theoretical contribution.

5. **Cross-domain absorption is near zero**: City-country = 0%, animal-class = 1.4%. The H1 hypothesis (>= 10% absorption in knowledge hierarchies) is effectively falsified. The only domains showing >5% absorption (city-continent 6.5%, city-language 6.6%) have shuffled controls that are multiples of the signal.

### Critical Assessment

The fundamental problem is that the Gemma 2 2B + Gemma Scope L12-16k SAE configuration, while running without errors, produces degenerate probes and broken control baselines. The probes are not reliable enough to measure absorption, and the controls show that the metric itself is not well-calibrated for this model. This is not merely a "proceed with caution" situation -- it invalidates the core measurement methodology.

The confound decomposition (H4) is the only hypothesis that technically passes, but it operates on degenerate quality proxy metrics and its practical significance is limited.

### Sunk Cost Check
Prior effort across 6 iterations is irrelevant to this decision. The evidence clearly shows the current approach is not producing publishable results.

### Why Not ADVANCE?
- 3/6 hypotheses are falsified (H1 borderline, H3, H6)
- Controls are broken in every domain
- The strongest novelty claim (unsupervised detection) has failed
- Probe quality fails the pre-specified gate in all domains

### Why Not REFINE the front-runner?
- The control failures suggest the measurement methodology itself needs fundamental rethinking, not incremental refinement
- Better probes alone will not fix the shuffled control problem
- The unsupervised pipeline failure is structural (the geometric signals simply do not correlate with absorption)

### Why PIVOT?
- The evidence overwhelmingly indicates the current candidate cannot produce a publishable paper
- The strongest remaining contribution is an informative negative result about cross-domain absorption and unsupervised detection, but this alone is insufficient for a top venue
- A fresh direction should leverage the substantial infrastructure already built (Gemma 2 2B, SAELens, probe framework)

### Recommended Pivot Direction
The most promising pivot is toward **cand_decomposition_reframing** (weighted score 2.65, highest of all candidates), but with significant modifications:
1. Focus on the **activation patching / causal test** angle, which remains genuinely novel
2. Use the negative unsupervised detection result as evidence that absorption is harder to detect than assumed
3. Investigate WHY controls are broken (is it the metric, the SAE, or the model?) -- this itself could be a contribution
4. Consider an alternative: a focused **negative result paper** documenting the failure of absorption generalization, unsupervised detection, and successive refinement theory across domains

## Next Actions

1. **Immediate**: Diagnose why shuffled label control produces 59.6% absorption (this may reveal a fundamental issue with the Chanin metric on Gemma 2 2B or with the implementation)
2. **If control bug found**: Fix implementation, re-run pilot on first-letter only. If controls now pass, reassess ADVANCE for cand_cross_domain_unsupervised
3. **If controls are inherently broken**: Pivot to cand_decomposition_reframing with emphasis on:
   - Threshold sensitivity as primary contribution (already have 5x4 grid data)
   - Activation patching for model-faithful vs SAE-induced absorption (novel)
   - Honest negative results about cross-domain generalization and unsupervised detection
4. **Alternative**: Frame the entire body of work as a systematic negative result / "hard lessons" paper about absorption measurement methodology

SELECTED_CANDIDATE: none
CONFIDENCE: 0.75
DECISION: PIVOT
