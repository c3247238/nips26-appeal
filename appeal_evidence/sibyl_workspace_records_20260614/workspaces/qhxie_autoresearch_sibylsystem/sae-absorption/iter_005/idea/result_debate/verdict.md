# Result Debate Verdict: Iteration 5

**Date**: 2026-04-15 | **Iteration**: 5 | **Decision**: PROCEED

---

## Overall Quality Score: 6.5 / 10

Two strong contributions (confound resolution 7/10, scaling surface 9/10) offset by a failed cross-domain measurement (2/10) and a taxonomy correction with a reporting bug (6/10). Without Phase 2, the remaining pillars would score 7.5/10.

---

## Key Conclusion

This iteration produced two publication-ready contributions and one honest negative result. The headline finding is that L0 was masking, not driving, the absorption-quality relationship: controlling for L0 strengthened the absorption-sparse probing correlation from r = -0.664 to r = -0.746 (a suppression effect). Combined with a 420-SAE scaling surface showing highly significant nonlinear interaction (p = 3.11e-15), this provides the most rigorous empirical evidence to date that absorption is a genuine, structured property of sparse autoencoders correlated with downstream quality. The cross-domain measurement failed due to metric and model limitations, not due to absorption being absent -- the question remains open.

Three data integrity errors in the integration files must be fixed immediately: (1) the taxonomy corrected rate is 19.2%, not 92.3%; (2) the phase_boundary_detected flag is inconsistent; (3) the mediation count is wrong.

---

## Action Plan

### Phase 1: Fix Data Integrity (< 1 hour, BLOCKING)

1. Correct `final_results_summary.md` and `final_results.json` to report the actual taxonomy corrected rate of 19.2% (not 92.3%)
2. Resolve `phase_boundary_detected` inconsistency between P3 and final results
3. Fix `n_full_mediations` count (should be 2, currently 0)

### Phase 2: Write Paper (4-6 hours, CRITICAL PATH)

Structure as three contributions:
- **C1**: Confound resolution -- suppression effect, mediation (SCR + TPP), within-width null as honest caveat. Drop "causal chain" from title; use "robust association after confound control."
- **C2**: Scaling surface -- 420-SAE GAM interaction, phase boundary, practical guidance for SAE hyperparameter selection.
- **C3**: Cross-domain diagnostic + taxonomy correction -- metric limitation as methodological contribution, prevalence recalibration (92.3% -> 73.1%/19.2%).

### Phase 3: Strengthen During Writing (parallel, 0 GPU-hours)

- Architecture-specific GAM ablation (standard-only vs. JumpReLU-only)
- Power analysis for within-width matching to quantify whether the null is informative or under-powered
- FDR correction across Phase 1 tests

### Deferred

- Gemma 2B cross-domain replication (conditional on access)
- SEM for dual-pathway L0 decomposition
- Hurdle model extension to 420-SAE dataset

---

## Venue Target

- **Current form**: NeurIPS/ICML MechInterp Workshop (70% confidence) or AAAI/EMNLP main (50%)
- **With Gemma 2B fix**: NeurIPS/ICML main (35%) or ICLR main (30%)
- **Minimum viable**: Two-contribution paper (confound resolution + scaling surface) is independently publishable

---

## Critical Caveats for the Paper

1. **Within-width matching null**: The absorption-quality association cannot be cleanly separated from width effects within a single width stratum. This limits causal claims.
2. **n = 48 from one model**: Confound resolution uses only Gemma 2B SAEs. Cross-model validation is needed for generalizability.
3. **Sparse probing mediation is uninterpretable**: Proportion mediated = 4.785 due to opposing causal pathways. Only SCR and TPP mediation results should be featured.
4. **Cross-domain absorption is untested, not falsified**: The metric failure on GPT-2 Small tells us about measurement limitations, not about whether absorption generalizes.
