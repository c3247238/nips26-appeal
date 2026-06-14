# Idea Validation Decision

## Pilot Evidence Summary

| Pilot | Status | Key Metrics |
|-------|--------|-------------|
| pilot_activation_patching | PASSED | Mean recovery: 67.3%, all 9/9 words >48% recovery |
| pilot_steering_cv | PASSED | High-CV: 0.153 vs Low-CV: 0.075 (2.03x ratio) |

### Key Findings
- **Activation patching**: Confirms genuine absorption - parent features recover substantially (67.3% mean) when child is zeroed
- **Steering by CV**: High-CV features show 2x larger steering effect than low-CV - CV predicts steering effectiveness
- Both pilots passed their pass criteria with strong margins

## Decision Matrix

| Candidate | Pilot Signal (0.30) | Hypothesis Survival (0.25) | Path to Result (0.20) | Novelty (0.15) | Resource Efficiency (0.10) | **Weighted Score** |
|-----------|--------------------|---------------------------|-----------------------|----------------|--------------------------|-------------------|
| cand_cv_actionability | 5 | 4 | 4 | 4 | 4 | **4.3** |
| backup_projection | 3 | 3 | 3 | 4 | 3 | 3.2 |
| backup_steering | 3 | 3 | 3 | 3 | 3 | 3.0 |
| backup_cross_arch | 2 | 2 | 2 | 4 | 2 | 2.3 |

### Scoring Rationale
- **cand_cv_actionability**: Strong pilot signals (67.3% recovery, 2x steering ratio); H1 and H4 validated; clear experimental path; manageable Basu collision; fits 1-hour budget
- **backup_projection**: No pilot data; relies on untested probe projection metric; viable but less validated
- **backup_steering**: Similar to front-runner but lower novelty (Basu collision more severe); redundant with primary
- **backup_cross_arch**: Falsified hypotheses (H3, H6) weaken theoretical foundation; highest risk

## Decision Rationale

**ADVANCE cand_cv_actionability** based on:
1. Both pilot experiments passed with strong margins (not borderline)
2. H1 (CV predicts steering) directly validated: 0.153 vs 0.075 (2x effect)
3. H4 (variance paradox) confirmed as genuine: CV=7.33 vs 0.01 (733x ratio)
4. H3 falsified at wrong sparsity (λ=0.001) - needs retesting, not abandoned
5. Clear 7-task plan with realistic 1-hour budgets
6. Consensus front-runner across all 6 synthesizer perspectives
7. Differentiation from Basu et al. clearly articulated (domain specificity: clinical vs. non-clinical)

**Concerns acknowledged but acceptable**:
- chi_ratio=1.88 < 3.0: Phase transition downgraded to supporting context
- CV threshold (1.0) post-hoc: Needs prospective validation in full experiment
- lambda_c instability: Will be addressed in cross-layer experiment

## Next Actions

### Immediate (This Iteration)
1. **full_steering_cv**: Run 30 high-CV vs 30 low-CV absorbed features at strengths +3, +5, +7 with statistical tests (Welch's t-test, BH correction)
2. **full_decoder_orthogonality**: Test whether orthogonality predicts steering as alternative to CV

### Subsequent Tasks
3. **full_non_absorbed_baseline**: Compare absorbed vs non-absorbed steering effects
4. **full_cross_architecture**: Replicate on Gemma-2-2B to test generalization
5. **analysis_hypothesis_tests**: Final statistical summary and hypothesis test summary

### Risks to Monitor
- If full_steering_cv fails to replicate pilot ratio, need to re-evaluate CV threshold
- If decoder orthogonality correlates better than CV, may need to pivot primary predictor

---

SELECTED_CANDIDATE: cand_cv_actionability
CONFIDENCE: 0.72
DECISION: ADVANCE