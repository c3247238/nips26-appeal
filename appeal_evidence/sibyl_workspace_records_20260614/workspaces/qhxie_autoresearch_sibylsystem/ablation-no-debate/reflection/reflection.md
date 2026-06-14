# Iteration 1 Reflection Report: Feature Absorption in SAEs

## Iteration Summary

This iteration investigated feature absorption in Sparse Autoencoders using multi-child proportional ablation. The project completed 4 pilot experiments (H1 multi-child absorption, H2 frequency correlation, H3 steering, H_Safe safety analysis) plus 4 follow-up pilots (H3 fix, multi-seed validation, H_Mech factorial, H_Safe on Gemma Scope). A paper was drafted, reviewed by critic and supervisor, and scored 6.0/10 (Borderline Reject) with a CONTINUE verdict.

**Key experimental finding**: H_Mech 2x2 factorial revealed absorption is **encoder-driven, not decoder-geometric** -- a genuinely novel and counter-intuitive result. However, the paper narrative was not updated to reflect this finding, creating a critical data-narrative mismatch.

**Supervisor score**: 6.0/10 (Novelty: 7, Soundness: 5, Experiments: 5, Reproducibility: 6)
**Critic score**: 4/10 (Experiments), 5/10 (Planning, Ideation, Writing)

---

## Issue Analysis by Category

### EXPERIMENT (4 critical, 3 major)

1. **Deterministic absorption (CRITICAL)**: Trained SAE absorption is exactly 0.5 across all 100 samples and all 3 seeds with std=0.0. The metric appears to reduce to a geometric constant: (overlap_c1 + overlap_c2) / 2 = (0.4 + 0.6) / 2 = 0.5. The reported t-test (t=63.2, p=3e-133) compares a constant against a random variable -- statistically valid but scientifically meaningless. The paper presents this as evidence of a "genuine phenomenon" without acknowledging the deterministic nature.

2. **Paper narrative contradicts own data (CRITICAL)**: Sections 4.4 and 5.1 claim "decoder geometry contributes 0.299" and "absorption is primarily geometric." But H_Mech data shows Condition A=Condition C=0.299 (decoder effect = 0.0) and Condition B nearly equals Condition D=0.484 (encoder effect = 0.191). The project's own EXPERIMENT_SUMMARY.md states "Absorption is ENCODER-DRIVEN, NOT decoder geometry." This is a factually incorrect narrative.

3. **H_Safe uses arbitrary indices (CRITICAL)**: Features 500-519 labeled "safety" and 100-119 labeled "non-safety" with no Neuronpedia validation. The experiment summary admits "feature selection was heuristic, not based on actual Neuronpedia annotations." Presenting arbitrary indices as annotated safety features is methodologically invalid.

4. **H_Mech circular design (MAJOR)**: Condition C reuses the trained encoder from Condition B, making proper decoder isolation impossible. The factorial is presented as cleanly isolating contributions when it does not achieve this.

5. **H2 zero-inflation (MAJOR)**: 98.5% of 1024 features show zero absorption. The rho=+0.17 correlation is driven entirely by 15 outlier features. The competitive exclusion hypothesis was never properly testable.

6. **H3 percentage bug (MAJOR)**: Steering verification table shows "Mean Delta Percent" of 134,717,856% and 1.52 billion% -- clear division-by-near-zero artifacts.

7. **Config discrepancy (MAJOR)**: Paper claims d_model=512, d_sae=4096, 15 configurations. Data shows d_model=128, d_sae=1024, single configuration. The discrepancy is never explained.

### WRITING (1 critical, 3 minor)

1. **Typo contradicting core methodology (CRITICAL)**: Section 5.1 states "Single-child ablation, which ablates the top-k children simultaneously" -- this describes multi-child ablation. The typo was flagged in the prior critic review but remains unfixed.

2. **Section order (MINOR)**: Limitations (5.5) appears after Future Directions (5.6).

3. **Banned patterns (MINOR)**: "a growing body of work" appears twice.

4. **Figure order confusion (MINOR)**: Figure 5 referenced before Figure 1.

### ANALYSIS (2 major, 1 minor)

1. **H1 framing overstates finding (MAJOR)**: Shuffled (0.487) and permuted encoder (0.484) baselines are nearly identical to trained SAE (0.500). Only random decoder (0.059) differs. The differentiation is about encoder training, not "training" overall.

2. **Pilot results presented as full experiments (MAJOR)**: The pilot summary recommended "proceed to full experiment (5 seeds x 3 L0 x 4 conditions)" but this was never done. The paper presents pilot results without clear labeling.

3. **H3 power understatement (MINOR)**: "Only 7 absorbed features were tested, limiting statistical power" understates the problem -- power is ~18% for d=0.5.

### IDEATION (1 major)

1. **Contribution claims oversell results (MAJOR)**: Contribution #2 claims "First decomposition of absorption into geometric vs. learned contributions" but the data shows decoder effect = 0.0. This is not a decomposition -- it is a refutation of the geometric hypothesis. Contribution #3 "Causal validation via steering" is also unsupported.

### REPRODUCIBILITY (1 major)

1. **Missing data availability statement (MAJOR)**: Experiment code exists in iter_001/exp/code/ but is not referenced in the paper. No data availability statement is provided.

---

## Resource Efficiency Assessment

### GPU Utilization
- **Total GPU time**: ~15 minutes for 4 pilot tasks (h_safe: 2min, h3_fix: 3min, multiseed: 8min, h_mech: 2min)
- **GPU idle time**: Minimal (~5 minutes between task dispatch)
- **Utilization**: High (85%). All experiments ran sequentially on a single GPU (cuda:0) with no idle waiting.

### Bottleneck Analysis
The iteration's wall-clock time was dominated by non-GPU stages:
- `idea_debate`: ~4.4 hours (iteration 0) + ~4.5 hours (iteration 1)
- `writing_sections`: ~1 hour (iteration 0) + ~1.7 hours (iteration 1)
- `writing_integrate`: ~1.6 hours (iteration 0) + ~21 minutes (iteration 1)
- `writing_final_review`: ~12.4 hours (iteration 0) + ~1.8 hours (iteration 1)
- `review` (critic + supervisor): ~45 minutes

The writing_final_review stage in iteration 0 took 12.4 hours -- an outlier likely due to review agent complexity or token accumulation.

### Scheduling Improvements
- Pilot experiments are well-optimized and fit within the 1-hour guideline.
- Writing and review stages could be parallelized with experiment execution.
- Review feedback (critic findings) is generated but not incorporated before the next iteration's writing begins. The pipeline should enforce a "fix before proceed" gate for critical issues.

---

## Quality Trend Assessment

**Trajectory: STAGNANT**

This is the first iteration, so no historical trend exists. However, the quality assessment reveals:
- **Strengths**: Genuine methodological novelty (multi-child ablation), honest negative result reporting, strong claim-evidence alignment.
- **Weaknesses**: Critical data-narrative mismatch, deterministic measurement presented as statistical result, broken experiment presented as valid negative result, arbitrary feature indices presented as validated annotations.

The supervisor score of 6.0/10 (Borderline Reject) with CONTINUE verdict indicates the project has a viable path forward but requires significant remediation before publishability.

---

## Root Cause Analysis

### Why the paper narrative contradicts the data
The H_Mech pilot was run AFTER the initial paper draft was written. The paper was drafted based on the first pilot's interpretation (geometric dominance from shuffled/permuted baselines ~0.48 vs trained 0.50). When the H_Mech factorial revealed encoder-driven absorption, the paper text was never updated. This suggests a **pipeline gap**: experimental findings that emerge during writing are not propagated back to the paper draft.

### Why critical typos persist
The Section 5.1 typo was identified in `iter_001/critic/critique_writing.md` but the writing integration stage did not incorporate this feedback. The writing pipeline appears to generate critiques but lacks an enforcement mechanism to ensure fixes are applied before the paper is finalized.

### Why deterministic absorption was not caught
The multi-seed pilot was designed to "address zero-variance concern" but the pass criteria (`non_zero_variance: false`) explicitly allowed zero variance. The pilot was treated as validating stability (all seeds agree) rather than flagging a potential metric design flaw. The deterministic nature should have triggered an investigation into the formula.

### Why H_Safe used arbitrary indices
The Gemma Scope SAE was not available during the first iteration's pilot phase. When H_Safe was run in the follow-up pilots, the code used heuristic index selection (500-519 vs 100-119) rather than actual Neuronpedia annotations. The experiment summary honestly admits this, but the paper text falsely claims Neuronpedia validation.

---

## System Self-Check Response

No `logs/self_check_diagnostics.json` file was found in this workspace. The system self-check system either did not run or did not produce output for this iteration.

---

## Path Forward

### To reach score 7.0 (Weak Accept)
1. Fix the typo in Section 5.1
2. Update paper narrative to correctly reflect encoder-driven H_Mech finding
3. Add clear explanation for deterministic absorption
4. Remove or fix H_Safe (validate with Neuronpedia or remove)

### To reach score 8.0 (Accept)
1. Redesign H_Mech with independently trained conditions
2. Run multi-seed validation for H3
3. Include experiment code reference in paper
4. Resolve config discrepancies

### To reach score 9.0 (Strong Accept)
1. Demonstrate encoder-driven finding on real language model SAEs (Gemma Scope) with validated feature annotations
2. Show generalization beyond synthetic hierarchies
3. Add held-out generalization experiments
