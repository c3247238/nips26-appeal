# Lessons from Iteration 10

**Date**: 2026-04-15 | **Score**: 6.5/10 | **Trajectory**: 7.0 -> 6.5 (regression)
**Score history**: 5.5 x3 -> 6.5 -> 6.0 -> 6.5 x3 -> 7.0 -> **6.5**

**Root cause of regression**: New data integrity errors (Table 3 CI inversion) introduced during iter 10 + failure to reframe contributions after probe degradation result undermined cross-domain headline.

---

## Must Improve

- **[DATA INTEGRITY -- P0, ZERO GPU, 0.5 hours] Fix Table 3 CI inversion**: In 5 of 7 rows, CI lower bound exceeds point estimate (e.g., 36.1% with CI [37.9%, 42.1%]). Caused by mixing per-token point estimates with per-word bootstrap CIs. Recompute all CIs using per-token aggregation. A reviewer will flag this within minutes -- mathematically impossible CIs trigger immediate rejection.

- **[DATA INTEGRITY -- P0, ZERO GPU, 2 hours] Reconcile aggregation methods**: Three different first-letter L24 16k rates: 27.1% (per-word), 21.6% (per-token), 34.5% (per-unique-word). Choose per-token as canonical. Recompute ALL rates consistently. Tables 2 and 3 must report the same number for first-letter F1=1.0.

- **[PIPELINE -- P0, ZERO GPU, 2 hours] Implement validate_integration.py**: 11th iteration recommending. Five findings in this review are data integrity failures it would have caught. Create data_manifest.json mapping every paper claim to source file. Run BEFORE every paper revision.

- **[FRAMING -- P0, ZERO GPU, 3 hours] Restructure contributions**: Lead with probe degradation methodology (R^2=0.777) as Contribution #1, not cross-domain characterization (which H10 shows is mostly a probe quality confound). Replace 4.1x headline with 2.7x quality-gated (31.4/11.6). Present city-language anomaly as the genuinely novel finding.

- **[VERIFICATION -- P1, 0.5 GPU-hours] Run 20-entity cross-domain patching spot-check**: The sign reversal from d=-0.91 to d=+1.50 has never been independently verified. This is the paper's single biggest vulnerability. Budget: 0.5 GPU-hours.

- **[EXPERIMENT -- P1, 0-1 GPU-hours] Validate or reframe probe degradation extrapolation**: Either run city-continent probe degradation to validate cross-domain transferability (1 GPU-hour), or reframe Section 4.6 as directional evidence rather than quantitative decomposition (zero GPU).

- **[DATA INTEGRITY -- P2, ZERO GPU, 0.5 hours] Fix layer multiplier**: 26x, 15x, 11x cannot all be correct for L6-to-L24 first-letter. Trace correct L6 rate from source data. One consistent number throughout.

---

## Watch Out

- **[DATA INTEGRITY] CI inversion is a new failure mode**: The probe degradation experiment produced correct results but the CI computation used a different aggregation method than the point estimates. This was introduced in iter 10. Always verify CIs contain their point estimates.

- **[FRAMING] The system's incentive toward strong headlines systematically biases claims beyond evidence**: '4.1x range' (stale), 'cross-domain characterization' as #1 contribution (undermined by own probe degradation), 'universal mechanism' (city-country excluded). Every iteration, critics flag overclaiming. The bias is structural, not occasional.

- **[EXPERIMENT] Probe degradation extrapolation assumption untested**: The claim that city-continent 'matches the curve within 0.6 pp' assumes Gaussian noise on binary probes transfers to multi-class probes with different failure modes. A domain expert reviewer will challenge this immediately.

- **[VERIFICATION] Skipping data integrity 'for speed' is a false economy**: 2 hours of validation work would have prevented 5+ findings in this review. Always prioritize validate_integration.py over writing revisions -- it catches errors that writing revisions introduce.

- **[WRITING] Verbatim repetition across sections wastes page budget**: Results appear in 4 locations (Sections 5.1, 5.2, 7.2, 8). Discussion should add implications, not restate numbers. Conclusion should summarize at sentence level.

- **[WRITING] Figure-text inconsistencies accumulate**: Figure 4 layer mismatch, Figure 5 caption-content mismatch, Figure 6 p-value mismatch, N=1471 vs N=1464. Each individually is minor; collectively they signal careless data handling.

---

## Keep Doing

- **Honest negative results (consecutive 10 iterations)**: 11 hypotheses with 4 SUPPORTED, 3 DEFINITIVE_NEGATIVE, 1 REFUTED, 1 MIXED, 1 PARTIALLY_SUPPORTED, 1 NOT_SUPPORTED. Table 5 is the paper's strongest credibility signal. **Never compromise this.**

- **Experiment-first strategy (validated 3 times)**: Score stagnation at 5.5 broken by experiments. Score stagnation at 6.5 broken by experiments. H10 probe degradation (R^2 from 0.077 to 0.777) is the strongest result of iter 10. Score improvements = experiments. Writing-only iterations = stagnation.

- **Full-mode execution (validated 3 times)**: FULL probe degradation R^2=0.777 vs PILOT R^2=0.077 (10x improvement). Adding F1=0.75 and F1=0.95 was critical. Running at full scale is not optional.

- **Statistical rigor**: Bootstrap CIs (10k, seed=42), Bonferroni correction, Cohen's d, Wilcoxon, Mann-Whitney, Kruskal-Wallis. Consistently praised.

- **Preemptive methodology section**: Section 3 documents all procedures before results appear. Prevents many reviewer complaints.

- **Zero experiment failures (consecutive 5 iterations)**: Infrastructure fully reliable. 7/7 executed this iteration.

- **Probe degradation three-way decomposition**: City-continent explained, city-country mostly explained, city-language genuine outlier. Creative, rigorous, directly addresses reviewer #1 objection.

---

## Next Iteration Gate Structure

### Gate 0 -- Data Integrity (ZERO GPU, ~7-8 hours, BLOCKING)
1. Implement validate_integration.py + data_manifest.json [2h]
2. Fix Table 3 CI inversion [0.5h]
3. Choose per-token canonical, reconcile all rates [2h]
4. Restructure contributions + replace 4.1x headline [3h]
5. Fix layer multiplier [0.5h]

### Gate 1 -- Verification Experiments (2-3 GPU-hours)
6. 20-entity cross-domain patching spot-check [0.5 GPU-h]
7. City-continent probe degradation OR reframe Section 4.6 [0-1 GPU-h]
8. Fix Figure 4 layer mismatch OR caption [0-0.5 GPU-h]
9. City-country patching OR document exclusion [0-0.5 GPU-h]

### Gate 2 -- Writing Polish (ZERO GPU, ~3-4 hours)
10. Eliminate Section 7.2 repetition [1h]
11. Fix Figure 5, Figure 6, N discrepancy [1h]
12. Reduce abstract density [0.5h]
13. Code availability, SAEBench, hype words [0.5h]

### Gate 3 -- Run validate_integration.py (must pass with zero discrepancies)

**Expected trajectory:**
- Gate 0 completed: 7.0-7.5
- Gates 0+1 completed: 7.5-8.0
- All gates + Gate 3 clean: 8.0 (Accept-level)
