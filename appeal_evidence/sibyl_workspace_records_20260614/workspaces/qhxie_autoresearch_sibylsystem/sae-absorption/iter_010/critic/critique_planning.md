# Planning Critique -- Iteration 10

## Overall Assessment

The planning for iteration 10 was focused on the right priorities: fixing the score regression from 7.0 to 6.5, executing the probe degradation ablation (H10), and completing missing experiments. The phased structure (Phase 0: blocking paper fixes, Phase 1: highest-priority experiment, Phase 2: remaining experiments, Phase 3: writing) is well-organized. Total compute budget (~3.5 GPU-hours) is reasonable for the project's maturity.

However, the planning perpetuated a systemic weakness: data integrity verification was planned (Phase 0.1) but then skipped for speed, continuing a 10-iteration pattern of deferring validation. The most consequential planning decision -- skipping phase0_data_integrity -- is directly responsible for several critical findings in this review.

---

## Strength 1: Priority Ordering Was Correct

Phase 0 (zero-GPU paper fixes) as a blocking prerequisite was the right call. The score regression was caused by stale data in the paper, not experimental weakness. Fixing paper-level issues before running new experiments prevents compounding errors.

H10 (probe degradation ablation) as Phase 1 highest priority was also correct. This was the single most important remaining experiment, recommended by the iter-9 review. The result (R^2=0.777, MIXED verdict) justifies the prioritization and substantially improves the paper's self-awareness about confounds.

---

## Strength 2: Reduced Compute Budget

Reducing from 10.5 GPU-hours (iter-9) to 3.5 GPU-hours reflects appropriate project maturity. The emphasis on zero-GPU paper fixes (7 hours CPU) correctly identifies that the bottleneck is writing and data integrity, not experiments.

---

## Weakness 1: Data Integrity Check Skipped (10 Consecutive Iterations)

The methodology explicitly plans validate_integration.py in Step 0.1 (~1.5 hours CPU). The task was "skipped for speed" (consolidation_summary open_issues OI1). This is the 10th consecutive iteration where this validation was recommended and deferred.

Consequences of not implementing it:
- Table 3 CI inversion (would have been caught by checking that CI contains point estimate)
- City-continent 42.9% vs 31.4% discrepancy between proposal and paper (would have been caught)
- Layer multiplier inconsistency 26x vs 15x vs 11x (would have been caught)
- N=1471 vs N=1464 figure legend discrepancy (would have been caught)
- Three different first-letter absorption rates (would have been caught)

5 of the 18 findings in this review are data integrity failures that validate_integration.py would have prevented.

**Assessment:** The decision to skip validation "for speed" was a false economy. 2 hours of CPU work would have prevented ~5 hours of reviewer-facing data integrity issues. This is a systemic planning failure.

---

## Weakness 2: No Verification of Cross-Domain Patching Correction

The methodology (Step 0.2) plans to "Update paper with corrected FULL-mode data" but does not include independent verification that the corrected data is correct. Given that the correction involves a sign reversal (d=-0.91 to d=+1.50) from a previously FAILED task, a spot-check should have been part of the plan, not an afterthought.

The strategist in the result debate recommended a 20-entity spot-check (0.5 GPU-hours) as the highest-priority verification task. This was not in the original methodology and has not been executed.

---

## Weakness 3: City-Continent Probe Degradation Not Planned

The methodology plans probe degradation only for first-letter probes. The result debate verdict explicitly recommended "Run city-continent probe degradation (1 GPU-hr)" as a critical verification step. This was not incorporated into the methodology or task plan.

Without cross-domain probe degradation, the claim that the first-letter probe degradation curve can be used to "decompose" RAVEL absorption rates rests on an untested assumption (that the slope is domain-independent). The planning should have included this as Phase 1.2.

---

## Weakness 4: Writing Phase Underbudgeted

Phase 3 allocates 2.5 hours CPU for "Write appendix sections" (2 hours) and "Document methodological details" (30 min). The actual writing issues identified in the review (Section 7.2 verbatim repetition, results reported in 4 locations, abstract information density, notation inconsistencies) would require ~4-5 hours to fix properly. The plan underestimates writing revision effort.

---

## Weakness 5: No City-Country Patching Decision

The methodology explicitly plans activation patching for first-letter (existing), city-continent (existing), and city-language (existing), but does not mention city-country. Given that city-country has the highest absorption rate (45.1%), the omission should have been an explicit decision with documented rationale, not a silent gap.

---

## Task Plan Assessment

The task plan (selected_candidate.json: "kept_task_count": 10) executed 7/9 tasks successfully with 2 skipped. Key outcomes:

| Task | Status | Assessment |
|------|--------|-----------|
| phase0_paper_corrections | Completed | 27 corrections applied. Good. |
| phase0_figures | Completed | 4 required figures OK. Good. |
| phase0_data_integrity | Skipped | Critical failure. See Weakness 1. |
| phase1_probe_degradation | Completed (FULL) | R^2=0.777, rho=-1.0. Excellent. |
| phase2_decoder_magnitude | Completed (FULL) | 6.16 nats, N=158. Good. |
| phase2_rate_distortion | Completed (FULL) | 131 pairs, NOT_SUPPORTED confirmed. Good. |
| phase3_appendix_writing | Completed | 5 appendix sections. Good. |
| phase3_methodology_docs | Completed | 7 methodology notes. Good. |
| setup_env_check | Skipped | Acceptable (reused iter_009 validation). |

The 2 failures (skipping data integrity and env check) are asymmetric in importance. The env check skip is fine; the data integrity skip is directly responsible for multiple critical issues.

---

## Resource Utilization

Estimated: 3.5 GPU-hours, 13 hours wall-clock total.
The FULL experiments completed within budget. The zero-GPU phases were partially completed (27 corrections applied, but validation skipped). Overall resource utilization was efficient for the experiments that were executed.

---

## Recommendations for Next Iteration

1. **Implement validate_integration.py BEFORE any new experiments.** This is blocking. Budget 2 hours CPU.
2. **Run cross-domain patching spot-check.** Budget 0.5 GPU-hours.
3. **Run city-continent probe degradation.** Budget 1 GPU-hour.
4. **Fix Table 3 CI inversion.** Budget 0.5 hours CPU (recompute + update).
5. **Reconcile all aggregation-dependent numbers.** Budget 2 hours CPU.
6. **Decide city-country patching: run or document exclusion.** Budget 0-0.5 GPU-hours.
