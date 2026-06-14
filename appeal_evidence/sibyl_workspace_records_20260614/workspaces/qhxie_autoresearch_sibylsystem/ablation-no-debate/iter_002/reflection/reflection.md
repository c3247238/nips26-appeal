# Reflection: Iteration 2

## Iteration Summary

**Score**: 6.0/10 (Borderline Revise)  
**Verdict**: Revise  
**Trajectory**: Improving from 5.0, but still stagnant

This iteration shows genuine progress: the supervisor score improved from 5.0 to 6.0, reflecting a more rigorous experimental design with the factorial decomposition and honest negative results. However, critical weaknesses remain that prevent a stronger score: post-hoc criterion revision for H_Mech, metric inconsistency across experiments, and synthetic-only validation at d_model=128. The paper has real merit but requires another iteration to address these fundamental issues.

---

## Issue Analysis by Category

### EXPERIMENT (Critical)

**Post-hoc criterion revision for H_Mech** (Status: NEW, Severity: Critical)
- Original pass criteria (B approx D and C approx A) failed on 14/15 runs (93.3%)
- Criteria revised to encoder effect > 0.5 and decoder effect < 0.1 AFTER observing data
- Not pre-registered; paper acknowledges exploratory status
- This is the paper's PRIMARY empirical claim and its confirmatory status is destroyed
- Suggestion: Pre-register revised criteria in OSF before running full experiment, OR reframe as exploratory pilot-scale evidence and run new experiment with revised criteria as official stopping rule

**Three different absorption metrics across experiments** (Status: RECURRING, Severity: Critical)
- Factorial decomposition: "overlap fraction"
- Multi-seed validation: "Jaccard overlap"
- Safety analysis: "cosine-based proportional absorption"
- Table 1 presents results from all three metrics as if comparable
- Cannot legitimately claim "encoder effect 80x larger than decoder effect" on unified scale
- Suggestion: Pick one canonical metric and recompute all experiments, OR split Table 1 into metric-specific sections

**Random baseline zero variance** (Status: RECURRING, Severity: Major)
- Random SAE baselines show std ~0 across all 5 seeds in multiseed_validation.json
- Either random SAE produces identical patterns regardless of seed, OR measurement has no sensitivity
- Cohen's d > 10 is mathematically misleading when one distribution has zero variance
- Suggestion: Investigate root cause; recalculate effect sizes with appropriate variance OR acknowledge explicitly

### EXPERIMENT (Major)

**Held-out generalization n=1 per seed** (Status: NEW, Severity: Major)
- r=0.998 correlation based on only 5 data points (n=1 held-out hierarchy per seed)
- With 3 degrees of freedom, confidence interval could be wide
- Paper presents as "perfect generalization" without reporting CI
- Suggestion: Increase n=3-5 held-out hierarchies per seed for reliable correlation estimate

**Unfulfilled ANOVA promise** (Status: RECURRING, Severity: Major)
- Section 3.5 promises one-way ANOVA across all completed variants in Results
- Never actually reported; pairwise effect sizes used instead
- Unfulfilled promise undermines credibility
- Suggestion: Add ANOVA with caveats about low power, OR remove promise from methodology

**Capacity-pressure mechanism post-hoc** (Status: NEW, Severity: Major)
- Lower L0 -> higher absorption is interesting finding
- But paper does not distinguish capacity-pressure absorption from TopK artifact
- Mechanism claim is post-hoc without diagnostic test
- Suggestion: Compare with JumpReLU SAE at equivalent L0 to test whether relationship holds

**d_model=128 only, no scaling validation** (Status: RECURRING, Severity: Major)
- Synthetic experiments at d_model=128, far below real LLM dimensions (768 to 12,288)
- Encoder-decoder dynamics could fundamentally change at scale
- Suggestion: Add d_model=512 validation point; acknowledge d_model=128 as pilot-scale choice

### ANALYSIS (Major)

**Safety analysis saturates at ~97%** (Status: NEW, Severity: Major)
- Both safety and non-safety groups show ~97% absorption (0.967 vs 0.968)
- At 97%, no dynamic range to detect differences
- Mann-Whitney p=0.989 null result is uninterpretable
- Suggestion: Use different metric with better dynamic range, OR analyze 3% non-absorbed features systematically

### WRITING (Major)

**H3 steering numbers inconsistency** (Status: NEW, Severity: Critical)
- Abstract reports ratio=0.776, p=0.273
- Section 4.3 reports ratio=0.776 +/- 0.066, p=0.273
- experiment_summary.json reports ratio=0.974, p=0.936
- Irreconcilable mismatch between paper and JSON
- Suggestion: Determine correct numbers, update paper, verify all H3 steering figures

**Overclaiming novelty** (Status: RECURRING, Severity: Minor)
- Claims "first factorial decomposition" and "encoder dominance"
- Oursland (2026) theoretically derives encoder-decoder asymmetry
- SAEBench already compares architectures and notes sparsity differences
- Actual contribution: empirical quantification, not discovery
- Suggestion: Restate as "first empirical quantification of encoder-decoder asymmetry via controlled factorial decomposition"

**Figure 9 reference issue** (Status: NEW, Severity: Minor)
- Figure list references "Figure 9: figure_9_summary_table.pdf" but never cited in text
- Creates confusion about whether supplementary or missing
- Suggestion: Cite Figure 9 in appropriate Results section OR remove from figure list

---

## Resource Efficiency Assessment

### GPU Utilization Analysis

From gpu_progress.json timings:

| Task | Planned (min) | Actual (min) | Efficiency |
|------|---------------|--------------|------------|
| h_mech_factorial | 45 | 5 | 89% saved |
| multiseed_validation | 30 | 4 | 87% saved |
| h_safe_gemma | 45 | 7 | 84% saved |
| ablation_hierarchy_strength | 30 | 0 | 100% saved |
| heldout_generalization | 20 | 0 | 100% saved |

**Observation**: All experiments ran significantly under time estimates. This indicates either:
1. Experiments were appropriately scoped for quick validation
2. Time estimates were overly conservative
3. Some experiments (ablation_hierarchy_strength, heldout_generalization) show 0 min actual, suggesting very fast execution or potential measurement issue

**GPU Idle Time**: Minimal - experiments were sequenced efficiently with no extended GPU idle periods observed in gpu_progress.json

**Bottleneck**: The bottleneck is NOT in experiment execution. The bottleneck is in:
1. Writing quality and consistency (H3 numbers mismatch)
2. Pre-registration and methodology rigor (post-hoc criteria revision)
3. Paper revision cycles

### Task Scheduling
- Experiments were run with good sequencing (h_mech_factorial -> multiseed -> h_safe_gemma)
- Independent tasks (ablation studies) ran after main experiments
- No obvious scheduling improvements needed at the experiment level

---

## Quality Trend Assessment

| Iteration | Score | Verdict | Key Issues |
|-----------|-------|---------|------------|
| 1 | 5.0 | Borderline Reject | Data integrity, zero-variance SAE, underpowered analyses |
| 2 | 6.0 | Borderline Revise | Post-hoc criteria, metric inconsistency, synthetic-only |

**Trajectory**: Improving (5.0 -> 6.0) but still stagnant in the 5-6 range. The paper has genuine merit (factorial decomposition method, honest negative results) but critical methodological issues prevent advancement.

**Root Cause Analysis**: The paper's weaknesses are NOT in experimental execution (which is efficient and thorough) but in:
1. Methodology rigor (post-hoc revisions destroy confirmatory status)
2. Measurement consistency (three different metrics without equivalence)
3. Scale validation (synthetic-only at d_model=128)

---

## System Self-Check Response

No `logs/self_check_diagnostics.json` found in current iteration. System self-check is not triggered for this iteration.

---

## Success Patterns (Preserve for Next Iteration)

1. **Honest negative results reporting**: H3 steering and H_Safe negative results are presented without spin, with specific p-values and effect sizes. This is the paper's strongest aspect and must be preserved.

2. **Factorial decomposition method**: The 2x2 design isolating encoder vs decoder contributions is genuinely novel and well-executed. This is the core contribution.

3. **Clear contribution framing**: Six contributions clearly enumerated; methodological contribution appropriately highlighted.

4. **Good limitation acknowledgment**: Paper discusses five limitations including synthetic-only scope and post-hoc criterion revision.

5. **Efficient experiment execution**: All experiments completed well under time estimates with clean state management.

---

## Systemic Patterns (Require Attention)

1. **Post-hoc revision pattern**: Across both iterations, the paper has revised criteria after observing data (iteration 1 had zero-variance issues, iteration 2 has H_Mech criterion revision). This suggests the need for tighter pre-registration discipline.

2. **Metric inconsistency**: The three different absorption metrics appear consistently. This is a fundamental measurement issue, not a writing issue.

3. **Synthetic-only scope**: Both iterations have been purely synthetic experiments at d_model=128. The path to a stronger paper requires real LLM validation at scale.

4. **Small-scale generalization claims**: Held-out generalization n=1 per seed and zero-variance baselines suggest fragility in statistical claims.

---

## Recommended Focus for Next Iteration

1. **Pre-register H_Mech revised criteria** in OSF before any new experiments
2. **Add d_model=512 validation** to establish scaling behavior
3. **Resolve H3 steering numbers inconsistency** by determining correct values from raw data
4. **Standardize on one absorption metric** and recompute all experiments
5. **Add JumpReLU comparison** to test capacity-pressure vs TopK artifact
6. **Investigate random baseline zero-variance** and document root cause