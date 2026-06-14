# Planning Critique

## Overall Assessment: 7/10

The experimental plan is well-structured with clear phases, time estimates, and falsification criteria. The pilot-to-full-mode progression is sound. Resource estimates are reasonable. However, three planning issues affected the paper's outcome.

## Strengths

### 1. Phased Execution with Clear Priorities
The four-phase plan (cross-domain characterization, causal mechanism, rate-distortion predictors, negative results documentation) with priority ordering is well-designed. Primary contributions are prioritized over tertiary ones. GPU-hour estimates are realistic (10.5 GPU-hours total, ~12 hours wall-clock).

### 2. Pre-Specified Falsification Criteria
Every hypothesis has explicit falsification criteria (e.g., "H1 falsified if rates within 5 pp across all types after controlling for L0 and width"). This is exemplary planning practice.

### 3. Risk Assessment Is Calibrated
The risk table identifies 7 risks with severity, likelihood, and mitigation strategies. The likelihood estimates proved roughly accurate (e.g., RAVEL probes below 0.90 at 30% likelihood -- they ended at 0.73-0.87, partially confirmed).

### 4. Quality Gates Are Defined
The three-tier quality gate (strict F1>=0.90, relaxed F1>=0.80, below gate F1<0.80) was defined before data collection and applied consistently. This prevents post-hoc rationalization of poor probe quality.

## Weaknesses

### 1. Probe Quality Gate Too Low

The relaxed quality gate (F1>=0.80) and the "below gate with caveat" tier (F1<0.80) are too permissive for a paper whose primary contribution depends on distinguishing real absorption from probe artifacts. City-country at F1=0.73 is 80-class prediction -- essentially random for rare classes (27 countries have <=10 entities). Including city-country in the headline "4x range" finding (from 11.6% to 45.1%) is misleading because the 45.1% rate is heavily confounded by probe imperfection.

A better plan would have:
- Set the strict gate at F1>=0.90 for inclusion in headline claims
- Set the relaxed gate at F1>=0.80 for inclusion in appendix/supplementary analysis
- Excluded F1<0.80 hierarchies from the main text entirely

The plan acknowledged the risk ("RAVEL probes still below 0.90 at all layers" at 30% likelihood) but the mitigation ("Relax to 0.85 with documented caveat") was further relaxed in execution to 0.73 with caveat. This scope creep in quality standards undermines the primary finding.

### 2. No Power Analysis for Architecture Comparison

The architecture comparison (H6) was planned with "~1 hour" GPU time and no power analysis. At L24, only 3 SAE configurations were available (JumpReLU 16k, JumpReLU 65k, Matryoshka 32k) -- two from the same architecture family. A Kruskal-Wallis test with n=12 (3 architectures x 4 hierarchies) has very low power to detect moderate effects. The plan should have recognized that the architecture comparison was underpowered before execution and either (a) obtained more SAE configurations or (b) explicitly stated that the comparison would be exploratory, not confirmatory.

The paper's conclusion ("hierarchy dominates architecture") overstates what a p=0.50 from 12 observations can establish.

### 3. Benign/Pathological Test Design Was Not Validated

Phase 2 plans the benign/pathological diagnostic but does not specify how to validate the diagnostic itself. The logit change threshold (0.1 nats) was chosen without justification. The control (ablating random directions) measures the wrong baseline -- it controls for ablation magnitude but not for directional relevance. A better control would ablate a direction orthogonal to the parent but with the same geometric relationship to the child feature.

The plan also did not anticipate the possibility that the diagnostic would be partially circular (the parent direction is part of the child decoder by construction in absorbed features).

### 4. Missing: Probe-Only Baseline Priority

Control 3 ("probe-only baseline: absorption rate attributable to probe imperfection alone") is listed in Section 3.3 but never prioritized in the experimental plan. It should have been a Phase 0 experiment: measure the probe-only FN rate before running any SAE-based measurements. This would have established the noise floor for all downstream analyses.

## Plan-Execution Gaps

| Planned | Executed | Gap |
|---------|----------|-----|
| Probe F1 > 0.90 required | City-country F1=0.73 included | Quality gate relaxed 17pp below plan |
| Architecture comparison at matched L0 | Width mismatch (Matryoshka 32k vs. JumpReLU 16k/65k) | L0 matching not enforced |
| Absorption rates per hierarchy with Bonferroni correction | Only city-language significant after Bonferroni vs. first-letter | 3 of 6 pairwise comparisons significant (not 4+ as implied) |
| Rate-distortion rho > 0.5 target | Model rho = 0.250 | Below falsification threshold |
| H8 tested across hierarchies | Tested on city-continent only | Single-hierarchy generalization |
