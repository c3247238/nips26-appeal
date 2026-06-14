# Strategist Analysis: Iter 10 FULL-Mode Results

**Agent**: Strategist (Strategic Research Advisor)
**Timestamp**: 2026-04-16
**Scope**: Strategic assessment of all iter_010 results, resource allocation, and next-step recommendation

---

## 1. Signal Strength Assessment

| Result | Signal Strength | Justification |
|--------|----------------|---------------|
| H10 probe degradation curve (R^2=0.777, rho=-1.0) | **Strong** | Perfect monotonicity across 7 levels, p=0.0087. FULL mode massively improved over PILOT (R^2 0.077 to 0.777). The SHAPE of the curve is real and methodologically important, regardless of whether quantitative extrapolation to RAVEL is defensible. |
| H7 universal competitive exclusion (d=0.75-1.50) | **Strong (first-letter) / Moderate-to-Strong (cross-domain)** | First-letter d=1.33 is stable across iterations. Cross-domain d=1.50 and d=0.75 are from corrected data with a sign reversal from a previously-failed task -- strong IF provenance holds, but data integrity was skipped. I discount cross-domain by one tier until provenance is verified. |
| H1 cross-domain variation (4.1x range) | **Moderate** | The Kruskal-Wallis p=7.4e-66 is sample-size driven. The robust pairwise comparison (first-letter 27.1% vs city-continent 31.4%) is NOT significant after Bonferroni (p=1.0). Only city-language vs first-letter reaches significance (p_Bonf=0.003). The 4.1x headline is inflated by city-country (probe F1=0.726, below gate). |
| H8 decoder direction magnitude (6.16 nats first-letter) | **Moderate** | Cross-hierarchy consistency (6.16 vs 3.98 nats) is informative about decoder geometry. But circularity acknowledged; 100% pathological is tautological. Not a standalone pillar. |
| H9 rate-distortion (rho=0.286, R^2=0.104) | **Strong as negative** | 131 pairs, all individual predictors reversed direction. Fourth confirmed failure of correlational approaches. |
| Quadruple negative (GAS, CMI, T(G), rate-distortion) | **Strong as negative** | Consistent pattern: all static/correlational predictors fail. Only causal intervention succeeds. This IS a finding, not merely absence of results. |
| City-language anomaly (-21.3 pp below curve) | **Moderate** | Interesting signal but open to alternative explanations (composite labels, fewer categories, different probing task structure). Needs validation. |
| Layer-dependent absorption (15x, 0.7% to 42.9%) | **Strong** | Robust across hierarchies. F1>=0.96 at all layers. Most reproducible finding in the paper. |

---

## 2. Opportunity Cost Analysis

| Direction | GPU Cost | Wall-Clock | Information Gain per GPU-Hour | Notes |
|-----------|----------|------------|-------------------------------|-------|
| **A. Verify cross-domain patching provenance** (SC1 from skeptic) | 0.5 hr | 1.5 hr | **EXTREMELY HIGH** | 0.5 GPU-hours for spot-check + CPU script. If provenance fails, the paper's second pillar collapses. If verified, the universal mechanism claim becomes defensible. Highest risk-adjusted ROI of any possible action. |
| **B. Aggregation consistency pass** (SC2 from skeptic) | 0-1 hr | 2 hr | **HIGH** | Zero or minimal GPU. Resolves the three-values-for-one-quantity problem (34.5%, 27.1%, 21.6%). Without this, any reviewer can trivially attack the absolute numbers. |
| **C. City-continent probe degradation** (FF1 remediation from skeptic) | 1 hr | 2 hr | **HIGH** | Tests whether probe degradation curves are domain-independent. If slopes match, the cross-domain extrapolation becomes defensible. If slopes differ, the paper pivots to "methodological warning" framing. Either way, very informative. |
| **D. Within-hierarchy variance analysis** (MC3 from skeptic) | 0 hr | 1 hr | **MEDIUM** | CPU only. ICC computation from existing data. If ICC<0.3, the paper needs reframing. Cheap insurance. |
| **E. Relegate city-country** (SC3 from skeptic) | 0 hr | 0.5 hr | **MEDIUM** | Pure writing. Probe F1=0.726 is indefensible for the primary comparison. Relegation removes a guaranteed reviewer attack vector. |
| **F. validate_integration.py** | 0 hr | 2 hr | **MEDIUM** | Pure CPU. Addresses a 10-iteration systemic weakness. Catches data mismatches before submission. |
| **G. New experiments (cross-model validation, etc.)** | 2+ hr | 4+ hr | **LOW at this stage** | Diminishing returns. The paper is in completion phase. New experimental directions would open iteration 11. |

---

## 3. Decision Matrix

| Direction | Signal Strength | GPU Cost | Risk of NOT Doing It | Expected Outcome |
|-----------|----------------|----------|---------------------|-----------------|
| **A. Verify patching provenance** | Critical dependency | 0.5 hr | **CATASTROPHIC**: Sign reversal exposed in review = rejection | Provenance confirmed OR error found early. Both outcomes valuable. |
| **B. Aggregation consistency** | Foundational | 0-1 hr | **HIGH**: Reviewer asks "why three different numbers?" with no answer | Single-method recomputation. Paper numbers become defensible. |
| **C. City-continent probe degradation** | High for H10 claims | 1 hr | **HIGH**: FF1 from skeptic goes unrebutted. Section 4.6 claims weakened. | If slopes match: cross-domain extrapolation defensible. If not: honest reframing. |
| **D. Within-hierarchy ICC** | Medium for framing | 0 hr | **MEDIUM**: MC3 left unaddressed. Paper claims hierarchy > class effects without evidence. | ICC value determines whether "hierarchy type is primary driver" framing is justified. |
| **E. Relegate city-country** | Medium for credibility | 0 hr | **MEDIUM**: Guaranteed reviewer attack on F1=0.726 in primary comparison. | Headline range becomes 2.7x (more honest). Removes weakest evidence from spotlight. |
| **F. validate_integration.py** | Medium for integrity | 0 hr | **MEDIUM**: 10 iterations of deferred tech debt. Risk of undetected data errors. | Automated cross-check catches mismatches. |

---

## 4. PROCEED vs PIVOT Verdict

### **PROCEED**

**Rationale:**

The paper has at least two hypotheses with strong signal and a clear path to publication-quality results:

1. **Probe degradation curve (H10)** -- R^2=0.777, perfect monotonic, statistically significant. This is the first quantitative demonstration that probe quality confounds absorption measurement. Even under the skeptic's strictest interpretation (methodological warning, not decomposition tool), this is a publishable methodological contribution. With city-continent probe degradation (Direction C below), it becomes a quantitative tool.

2. **Causal competitive exclusion (H7)** -- First-letter d=1.33 is rock-solid. Cross-domain data is strong IF provenance is verified (Direction A below). Even in the worst case where cross-domain data is suspect, first-letter causal evidence plus the cross-domain absorption rate characterization remains a contribution.

3. **Transparent negative results** -- The quadruple failure of correlational predictors is itself a finding: absorption is a causal phenomenon requiring causal methods. Combined with the probe degradation confound, this provides genuine guidance to the field.

**Why not PIVOT:**
- The front-runner idea has accumulated 10 iterations of empirical evidence. Pivoting now discards ~100 GPU-hours of validated results.
- No backup idea provides higher expected value at this maturity stage. The controlled dictionary experiment (Backup 1) is strong but would require a fresh experimental campaign.
- The paper's core story -- "absorption measurement is confounded by probe quality; when controlled, cross-domain variation exists with hierarchy-specific anomalies; and only causal methods (activation patching) successfully characterize the mechanism" -- is coherent and addresses a real gap in the literature.
- The skeptic's criticisms are ADDRESSABLE within 1-2 days of targeted work. They do not invalidate the core findings; they require more honest framing and specific verification experiments.

---

## 5. PROCEED: Priority-Ordered Next Steps

### Priority 1: Verification and Integrity (BLOCKING, ~2 GPU-hours + ~5 CPU-hours)

These must be completed before paper submission. Failing to do so leaves the paper fatally vulnerable to reviewer attacks on data provenance and measurement consistency.

**Step 1.1: Verify cross-domain patching provenance** (~0.5 GPU-hours + ~1 CPU-hour)
- Write a 50-line verification script that loads the corrected JSON, checks tensor shapes, verifies entity counts against RAVEL ground truth.
- Re-run cross-domain patching on 20 randomly sampled entities from city-continent as a spot-check.
- Document the specific bug (what line of code changed, what the incorrect behavior was) in an appendix.
- If spot-check reproduces d>0 and recovery>20%: universal mechanism confirmed. If not: downgrade to "preliminary corrected results pending independent verification."
- **Estimated time**: 0.5 GPU-hours + 1 hour CPU.

**Step 1.2: Aggregation consistency pass** (~0-1 GPU-hours + ~2 CPU-hours)
- Choose per-token as the canonical aggregation method (more robust to entity-frequency imbalance, used in iter_010 FULL).
- Re-compute ALL headline numbers using per-token aggregation from stored data.
- Create a consistency table in an appendix showing per-word vs per-token rates for all hierarchies.
- Update the paper to use a single consistent set of numbers.
- **Estimated time**: 0-1 GPU-hours (if re-computation needed) + 2 hours CPU.

**Step 1.3: validate_integration.py** (~2 CPU-hours)
- Automated cross-check of every numerical claim in the paper against source JSON files.
- Maps each table entry, in-text metric, and figure data point to a specific file:field path.
- Run as CI-style check before every paper revision.
- **Estimated time**: 2 hours CPU.

### Priority 2: Strengthen the Weakest Pillar (HIGH, ~1 GPU-hour)

The probe degradation curve's quantitative extrapolation to RAVEL hierarchies is the paper's most vulnerable claim (the skeptic's FF1). A single targeted experiment resolves this.

**Step 2.1: City-continent probe degradation curve** (~1 GPU-hour)
- Apply the same weight-noise degradation pipeline to city-continent probes (F1=0.871).
- Degrade to 5-7 F1 levels and re-measure absorption.
- Compare the SLOPE of the city-continent degradation curve to the first-letter curve.
- If slopes are similar (within 2x): cross-domain extrapolation is defensible. The paper can claim "the probe-quality confound operates with similar magnitude across domains."
- If slopes differ significantly: the paper reframes to "methodological warning" rather than "decomposition tool." Still publishable but with weaker Section 4.6.
- **Estimated time**: 1 GPU-hour.

### Priority 3: Paper Framing Corrections (MEDIUM, ~2 CPU-hours)

These are zero-GPU writing changes that remove guaranteed reviewer attack vectors.

**Step 3.1: Relegate city-country from headline comparison** (~30 min)
- Lead with first-letter (F1=1.0) vs city-continent (F1=0.87) vs city-language (F1=0.82).
- Report city-country (F1=0.726) in a supplementary table with explicit probe quality caveat.
- Reframe headline range as "2.7x range (11.6% to 31.4%) across quality-gated hierarchies." Note the full range extends to 45.1% with low-probe-quality caveat.

**Step 3.2: Compute within-hierarchy ICC** (~1 hr CPU)
- Report per-class absorption rates for all hierarchies.
- Compute ICC or eta-squared for within vs between hierarchy variance.
- If ICC<0.3: reframe from "hierarchy type is primary driver" to "hierarchy type and within-hierarchy class properties jointly determine absorption."
- If ICC>0.5: current framing is supported.

**Step 3.3: Strengthen claim language** (~30 min)
- "Universal" mechanism -> "mechanism confirmed across three hierarchy types in Gemma 2 2B."
- "4.1x range" -> "2.7x range across quality-gated hierarchies" (primary), "4.1x including low-quality-probe hierarchy" (secondary).
- Quadruple negative -> "four static-geometry predictors tested" (not "ALL correlational methods").
- H8 decoder magnitude -> demote from numbered contribution to methodological observation.
- Quadratic R^2=0.942 -> remove or footnote (overfitting 7 points with 3 parameters).

---

## 6. Resource Allocation Summary

| Step | GPU-Hours | CPU-Hours | Priority | Blocks |
|------|-----------|-----------|----------|--------|
| 1.1 Verify patching provenance | 0.5 | 1 | **P0 (BLOCKING)** | Universal mechanism claim |
| 1.2 Aggregation consistency | 0-1 | 2 | **P0 (BLOCKING)** | All absolute absorption numbers |
| 1.3 validate_integration.py | 0 | 2 | **P0 (BLOCKING)** | Paper integrity |
| 2.1 City-continent probe degradation | 1 | 0.5 | **P1 (HIGH)** | Section 4.6 quantitative claims |
| 3.1 Relegate city-country | 0 | 0.5 | **P2 (MEDIUM)** | Headline framing |
| 3.2 Within-hierarchy ICC | 0 | 1 | **P2 (MEDIUM)** | Hierarchy vs class framing |
| 3.3 Strengthen claim language | 0 | 0.5 | **P2 (MEDIUM)** | Overclaiming risk |
| **Total** | **1.5-2.5** | **~7.5** | | |

Total wall-clock estimate: ~2-3 days of focused work. Total new GPU budget: 1.5-2.5 hours. This is well within the project's resource envelope (config: max_gpus=4, local compute).

---

## 7. Risk Assessment for Recommended Path

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Patching spot-check FAILS (d<0 on subset) | 15% | CRITICAL: Universal mechanism claim collapses | Downgrade to "first-letter causal evidence confirmed; cross-domain evidence preliminary." Paper still submittable with this framing. Backup 1 (controlled dictionary) becomes supplementary experiment. |
| City-continent probe degradation slope differs 2x+ from first-letter | 30% | HIGH: Section 4.6 claims about decomposition become invalid | Reframe to "methodological warning" (still a contribution). The EXISTENCE of the confound is the finding. |
| Within-hierarchy ICC < 0.3 | 25% | MEDIUM: "Hierarchy type is primary driver" framing challenged | Reframe to joint determination. Add per-class analysis as supplementary. Within-hierarchy variation (Backup 4) becomes a future work recommendation. |
| Aggregation re-computation changes hierarchy ordering | 10% | MEDIUM: Cross-domain comparison reshuffled | Report new ordering honestly. The paper's claim is about VARIATION, not about a specific ordering. |
| Competing paper appears during revision | 5% | LOW: Verified no competing work as of April 2026 | Execute promptly. Submission in 2-3 weeks mitigates. |

---

## 8. Venue and Timeline Strategy

**Target**: NeurIPS 2026 (submission deadline typically May/June 2026)

**Timeline**:
- Days 1-2: Priority 1 (verification and integrity). All blocking.
- Days 2-3: Priority 2 (city-continent probe degradation). 1 GPU-hour.
- Days 3-4: Priority 3 (writing corrections). Zero GPU.
- Days 4-5: Full paper revision integrating all changes. Run validate_integration.py.
- Days 5-7: Internal review cycle. Final polish.

If NeurIPS deadline has passed by the time the paper is ready, fallback to ICLR 2027 or EMNLP 2026.

**Submission readiness assessment**: The paper is 85% ready. The remaining 15% is verification (5%), one targeted experiment (5%), and writing corrections (5%). No structural changes needed. The core story is sound.

---

## 9. Strategic Assessment: What This Paper IS and IS NOT

### What this paper IS:
1. **The first systematic demonstration that probe quality confounds absorption measurement.** The probe degradation curve (R^2=0.777) is a genuine methodological contribution. Every future absorption paper must now account for probe quality.
2. **The first causal evidence for competitive exclusion in SAEs beyond first-letter spelling.** Even under conservative interpretation, first-letter d=1.33 with cross-domain evidence on 2 additional hierarchies is novel.
3. **An honest catalogue of what does NOT predict absorption.** Four failed correlational approaches vs. one successful causal approach. This guides future research toward interventional methods.
4. **A cross-domain absorption characterization** showing hierarchy-dependent rates, with the specific finding that city-language is anomalously low. Whether this is framed as "3 hierarchies" or "4 hierarchies" depends on whether city-country is included.

### What this paper is NOT:
1. **A theory of absorption.** The rate-distortion framework failed. The ecological competitive exclusion model (Backup 2) was never tested. The paper characterizes and measures but does not predict.
2. **A cross-model study.** Everything is on Gemma 2 2B. Universal claims must be scoped accordingly.
3. **A solution to absorption.** The paper diagnoses but does not treat. The absorption-aware correction (Backup 3) remains untested.
4. **A definitive cross-domain characterization.** With 3-4 hierarchy types on one model, this is a first step, not a comprehensive survey.

### Strategic framing for maximum impact:
The paper's greatest strategic strength is the COMBINATION of (a) methodological rigor (probe degradation demonstrates the confound) and (b) genuine discovery (city-language anomaly, universal mechanism). A reviewer who is skeptical of the cross-domain claim (the skeptic's position) should still find the probe degradation methodology and the negative result catalogue valuable. A reviewer who is enthusiastic about the cross-domain findings (the optimist's position) should appreciate the honest accounting of what does not work.

The paper should be framed as: **"We set out to characterize absorption across domains. In doing so, we discovered that the measurement itself is confounded, developed a method to quantify the confound, and found that genuine hierarchy-specific effects persist after accounting for it. Along the way, we confirmed the causal mechanism is universal and documented four failed predictive approaches."** This narrative positions the probe degradation finding as the methodological centerpiece and the cross-domain characterization as the empirical centerpiece, with appropriate caveats on each.

---

## 10. Backup Idea Recommendations (Contingency Only)

| Scenario | Trigger | Recommended Action |
|----------|---------|-------------------|
| Patching spot-check fails | d<0 on 20-entity subset | Downgrade H7-crossdomain to "preliminary." Paper proceeds with first-letter causal + cross-domain descriptive + probe degradation curve. Still submittable to NeurIPS MI Workshop. |
| ICC < 0.3 AND city-continent slope mismatch | Within-hierarchy dominates AND probe curves are domain-specific | Activate Backup 4 (within-hierarchy variation) as supplementary analysis. Reframe paper as "absorption is class-driven, not hierarchy-driven." |
| All verification steps pass | Everything holds | Proceed to submission. Backups 2 (ecological phase transitions) and 4 (within-hierarchy) recommended as follow-up papers or extended version. |
| Score regression continues (reviewer score <6.0) | Next review cycle | Activate Backup 1 (controlled dictionary) as a second independent contribution to strengthen the paper. |
