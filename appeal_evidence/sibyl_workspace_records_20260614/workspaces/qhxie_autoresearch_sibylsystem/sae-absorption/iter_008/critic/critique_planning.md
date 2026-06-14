# Critique: Planning

**Reviewer:** sibyl-critic
**Date:** 2026-04-15
**Target:** `current/plan/methodology.md`, `current/plan/task_plan.json`, `current/plan/pilot_plan.json`

---

## Overall Assessment: 6/10

The methodology document is thorough in specifying statistical tests, quality gates, and expected outcomes. The experimental plan is well-organized into phases with clear dependencies. However, the plan has a fundamental structural problem: it does not include the one experiment (degraded-probe ablation) that would resolve the study's most critical confound, and it overcommits to experiments on falsified hypotheses.

---

## Strengths

### S1. Clear quality gates with documented relaxation criteria
The F1 >= 0.90 strict / 0.85 relaxed quality gate is explicitly defined, and the plan specifies what happens if probes fail to meet it (relax with documentation, test alternative layers, fall back to GPT-2 Small). This is good experimental planning.

### S2. Comprehensive statistical analysis plan
Each experiment has pre-specified statistical tests: bootstrap CIs (10k resamples), paired permutation tests, Cohen's d, Wilcoxon signed-rank, ANOVA. Seeds are fixed (42). The plan even specifies the number of interpolation steps for integrated gradients (10). This level of detail supports reproducibility.

### S3. Appropriate experimental controls
Random direction baseline, shuffled hierarchy control, probe-only baseline, and loose hedging classification are all included. The activation patching experiment uses magnitude-matched controls (15 random features matched to within 10% of the child feature's activation). These controls are well-designed.

### S4. Honest demotion of failed hypotheses
GAS, CMI, and Absorption Tax are explicitly demoted to appendix with stated reasons. Phase 2 and Phase 3 are labeled "APPENDIX" and given minimal resource allocation.

---

## Weaknesses

### W1. CRITICAL -- No degraded-probe ablation in the plan

The single most important experiment for resolving the primary confound is absent from the methodology. The plan acknowledges the probe quality problem:

> "Probe quality is the critical blocking dependency. Pilot probes are below the 0.90 quality gate."

But the response is only to try to find better probes (more layers, more prompt templates), not to control for probe quality as a confound. A degraded-probe ablation -- degrade high-quality first-letter probes to match RAVEL quality levels and remeasure absorption -- would:
1. Cost 1-2 GPU-hours (less than Phase 2 GAS or Phase 3 Tax)
2. Definitively resolve whether cross-domain variation is genuine or probe-driven
3. Strengthen the paper's primary claim if the confound is ruled out, or redirect the narrative if confirmed

This experiment should have been Priority 0 in the plan, before any cross-domain measurement.

### W2. MAJOR -- Resource misallocation: falsified hypotheses get experiments, critical confounds do not

The resource allocation:
- Phase 2 (GAS, already FAILED in pilot): 1 GPU-hour budgeted
- Phase 3 (Absorption Tax, already NOT SUPPORTED): 1 GPU-hour budgeted
- Degraded-probe ablation: 0 GPU-hours budgeted

GAS achieved rho=0.12 in pilot; re-running at 25x scale confirmed the same result (rho=0.116). The plan allocates resources to re-confirming a known failure while omitting the experiment that would resolve an open question. This is a planning failure.

The 2 GPU-hours spent on GAS and Tax could have been spent on the degraded-probe ablation, which would have had 10x the impact on paper quality.

### W3. MAJOR -- Architecture comparison plan specifies L0 matching but execution does not achieve it

The methodology states:
> "Report L0 alongside absorption rates to control for sparsity differences"
> "2-way ANOVA: absorption ~ architecture * hierarchy_type"

But the plan does not include a concrete step for L0 matching. BatchTopK operates at L0=20 while JumpReLU operates at L0=75-87 -- a 4x difference. The plan recognizes this ("BatchTopK operates at L0=20 vs. JumpReLU L0~75-87") but proposes no mitigation beyond "report L0 alongside." For a valid architecture comparison, the plan should have either:
1. Selected SAEs at matched L0 values (SAEBench contains multiple L0 configurations)
2. Included L0 as a covariate in the statistical test
3. Restricted comparison to architectures at similar L0

### W4. MAJOR -- Multi-layer cross-domain plan was not achievable given probe quality

The plan specifies: "Layers 6, 12, 18, 24 for all 4 hierarchies" and "12+ SAEs x 4 hierarchies" for cross-domain absorption. But the plan's own quality gate (F1 >= 0.85 relaxed) means most layer-hierarchy combinations cannot be measured:
- At L6: all RAVEL probes are F1 = 0.35-0.65 (all FAIL)
- At L12: RAVEL probes are F1 = 0.62-0.79 (all FAIL)
- At L18: RAVEL probes are F1 = 0.78-0.84 (city-continent barely passes relaxed)
- At L24: F1 = 0.79-0.84 (city-continent and city-language at relaxed level)

The plan should have anticipated that cross-domain measurement would be restricted to L24 (and possibly L18) based on pilot probe quality data. Planning for "4 layers x 4 hierarchies" when only "1-2 layers x 2-3 hierarchies" were achievable creates a gap between planned scope and actual execution.

### W5. MINOR -- Hedging decomposition across hierarchies planned without sample size consideration

Step 1.4 plans "hedging decomposition per hierarchy type and SAE config." For city-continent (6 classes) and city-language (50 classes), the number of false negatives per SAE is often 0-3. The plan does not specify a minimum sample size for hedging analysis. A minimum of N=20 FNs should have been required for any hedging classification to be meaningful.

### W6. MINOR -- Activation patching sample size target was too low

The plan specifies "Expand from 7 words to >= 20 words" (Step 0.1). The execution achieved 25 words (19 with absorption). But 18 of the 25 words were discovered via IG-guided search biased toward finding absorption. The plan should have specified criteria for word selection quality (e.g., minimum raw accuracy >= 0.50, English words only, or stratified sampling across absorption severity).

---

## Planning Quality Summary

| Dimension | Score | Comment |
|-----------|-------|---------|
| Statistical analysis plan | 8/10 | Comprehensive, pre-specified, appropriate |
| Resource allocation | 4/10 | Spends on confirmed negatives, omits critical ablation |
| Risk mitigation | 4/10 | Identifies risks correctly, mitigates with "try harder" rather than controlled experiments |
| Scope realism | 5/10 | Plans for 4 layers x 4 hierarchies when pilot data showed 1 layer x 2-3 hierarchies achievable |
| Control specification | 7/10 | Good controls for patching and hedging; missing L0 matching for architecture |
