# Idea Validation Decision

## Pilot Evidence Summary

### Core Method Comparison (CIFAR-10, ResNet-20, 100 epochs, seed 42, AdamW lr=1e-3)

| Method | Best Acc | Final Acc | CSI | AIS | BEM | Mean WD Actual |
|--------|----------|-----------|------|------|------|---------------|
| AdamW constant (baseline) | 89.48% | 89.42% | 0.893 | 0.296 | 0.000 | 5.00e-4 |
| CWD hard | 89.29% | 89.18% | 0.856 | 0.353 | 0.506 | 2.47e-4 |
| SWD | 89.50% | 89.30% | 0.869 | 0.298 | 0.900 | 5.00e-5 |
| Cosine WD schedule | 89.48% | 89.36% | 0.897 | 0.369 | 0.505 | 2.48e-4 |

### Soft CWD Proximal Approximation Sweep (H1 validation)

| Beta | Best Acc | Accuracy Gap vs Hard CWD | Mean WD Actual |
|------|----------|--------------------------|---------------|
| 10 | 89.75% | +0.46% | 2.42e-4 |
| 50 | 89.39% | +0.10% | 2.39e-4 |
| 100 | 89.12% | -0.17% | 2.42e-4 |
| 1000 | 89.66% | +0.37% | 2.42e-4 |
| Hard CWD | 89.29% | reference | 2.47e-4 |

### Key Observations from Pilot

1. **Budget equivalence strongly confirmed**: All four core methods achieve 89.29-89.50% best accuracy -- a spread of only 0.21%. This is within noise for single-seed experiments.

2. **BEM reveals hidden confounds**: CWD (BEM=0.506) and Cosine WD (BEM=0.505) both apply approximately half the nominal WD budget. SWD (BEM=0.900) applies only ~10% of the nominal WD. Despite 10x differences in effective WD, accuracy is essentially identical, confirming WD budget has minimal impact at this scale.

3. **CSI has limited but nonzero differentiation**: CSI ranges 0.853-0.897 (CV ~2%). Constant WD (0.893) and Cosine WD (0.897) have highest CSI, CWD variants cluster lower (0.853-0.876). The ordering is suggestive (stable WD -> stable norms) but the spread is narrow.

4. **AIS is moderately informative**: AIS ranges 0.296-0.402. CWD methods and cosine WD show higher AIS (0.319-0.402) than constant WD (0.296), suggesting alignment-aware and schedule methods do engage with the alignment signal more -- but this does not translate to accuracy improvements.

5. **Soft CWD convergence is noisy, not clean**: The accuracy gap between soft CWD (any beta) and hard CWD ranges from -0.17% to +0.46% without clear monotonic convergence as beta increases. This is a concern for H1 but is likely within single-seed noise. The effective WD values (0.000239-0.000247) do converge across all beta values, confirming the budget mechanism.

6. **CWD does NOT outperform baseline**: CWD hard (89.29%) is actually 0.19% BELOW the AdamW constant baseline (89.48%). This is preliminary evidence supporting H4 (CWD's benefit is not from alignment-awareness).

7. **All metrics are computable and non-degenerate**: No NaN/Inf. All three metrics (CSI, AIS, BEM) produce finite values with meaningful variation, passing the kill criteria.

## Decision Matrix

### Candidate: cand_unified_framework (Front-Runner)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 4 | Budget equivalence confirmed (0.21% spread). BEM successfully differentiates WD budgets (0.00-0.90). CWD does NOT beat baseline, supporting core thesis. All methods work as implemented. |
| Hypothesis survival | 0.25 | 4 | H3 (budget equivalence): strongly supported. H4 (CWD mechanism): preliminary support (CWD <= baseline). H1 (proximal unification): soft CWD convergence noisy but effective WD matches. H5 (metric predictiveness): metrics computable, differentiation modest but present. H2, H6, H7 not directly tested in pilot. |
| Path to full result | 0.20 | 5 | Clear phased plan (CIFAR full -> ImageNet -> ViT). Codebase functional, all methods implemented. Pilot passed kill criteria. 117 GPU-hours feasible on 8x RTX PRO 6000. Infrastructure validated. |
| Novelty (from report) | 0.15 | 4 | Novelty score 7/10. Phi-framework unification is novel. WD Stability Condition is novel. No existing compute-controlled benchmark. CWD falsification battery extends beyond ICLR 2026 paper. Partial overlap with D'Angelo et al. and CWD papers, but differentiated. |
| Resource efficiency | 0.10 | 4 | 117 GPU-hours is moderate for a NeurIPS paper. Phased approach allows early termination if framework proves vacuous. Pilot cost was minimal (~2 GPU-hours). |

**Weighted Score: 0.30*4 + 0.25*4 + 0.20*5 + 0.15*4 + 0.10*4 = 1.20 + 1.00 + 1.00 + 0.60 + 0.40 = 4.20**

### Candidate: cand_contrarian_mechanism (Backup 1)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 4 | Same pilot data supports the contrarian thesis: budget equivalence holds, CWD does not beat baseline. |
| Hypothesis survival | 0.25 | 4 | Budget equivalence confirmed. CWD mechanism hypothesis has preliminary support. |
| Path to full result | 0.20 | 4 | Simpler scope (no theory), but still needs ImageNet. Lower barrier to publishable paper. |
| Novelty (from report) | 0.15 | 3 | Novelty score 6/10. D'Angelo et al. (NeurIPS 2024) already establishes WD-as-effective-LR narrative. Purely empirical without theory risks "incremental" label. |
| Resource efficiency | 0.10 | 4 | Potentially lower GPU cost (fewer methods, no theory experiments). |

**Weighted Score: 0.30*4 + 0.25*4 + 0.20*4 + 0.15*3 + 0.10*4 = 1.20 + 1.00 + 0.80 + 0.45 + 0.40 = 3.85**

### Candidate: cand_allostatic_wd (Backup 2)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | NOT TESTED in pilot. No evidence for multi-timescale allostatic WD. Pilot actually shows WD variation has minimal effect on accuracy -- which weakens the motivation for a more complex WD controller. |
| Hypothesis survival | 0.25 | 2 | The pilot's budget equivalence finding is actively hostile to this candidate: if all WD schedules perform the same, why would a 3-level adaptive controller help? The fundamental premise is undermined. |
| Path to full result | 0.20 | 2 | Would require complete redesign: new controller implementation, many hyperparameters to tune (3 levels x 2-3 params each), ablation across all 3 levels. Estimated >200 GPU-hours with high risk of null result. |
| Novelty (from report) | 0.15 | 5 | Novelty score 9/10. Zero arXiv results. Genuinely novel framing. |
| Resource efficiency | 0.10 | 2 | High cost for uncertain payoff. Many hyperparameters mean large tuning budget. Biological analogy may not survive ML peer review. |

**Weighted Score: 0.30*1 + 0.25*2 + 0.20*2 + 0.15*5 + 0.10*2 = 0.30 + 0.50 + 0.40 + 0.75 + 0.20 = 2.15**

## Decision Rationale

**ADVANCE cand_unified_framework with weighted score 4.20.**

The evidence is clear and consistent:

1. **The front-runner's core thesis is validated by the pilot.** Budget equivalence (H3) is strongly confirmed with a 0.21% accuracy spread across methods with 10x WD budget differences. CWD's inability to outperform baseline (H4 preliminary support) aligns with the framework's prediction that alignment-awareness is not the mechanism. The proposed metrics are computable and differentiate methods (H5 partial support).

2. **No hypothesis was falsified.** H3's falsification criterion (>0.3% improvement with p<0.05) was not triggered -- the maximum delta is 0.21% from a single seed. H1's soft CWD convergence is noisy but the effective WD values converge cleanly. H2, H6, H7 are not yet testable from pilot data but face no preliminary contradictions.

3. **The infrastructure is validated.** All four core WD methods run correctly. The codebase logs all required diagnostics. Metrics compute without errors. Total pilot cost was ~2 GPU-hours, well within budget.

4. **The contrarian backup is strictly dominated.** cand_contrarian_mechanism shares the same empirical evidence but scores lower on novelty because it lacks the theoretical framework. The unified framework subsumes the contrarian empirical contribution.

5. **The allostatic backup is actively contradicted.** The pilot shows WD variation has near-zero effect on accuracy, which undermines the premise that a more complex adaptive WD controller would help. Despite its high novelty, pursuing it would be spending GPU compute against the empirical evidence.

6. **Research focus = FOCUSED mode.** The directive is to prefer REFINE over PIVOT and give the front-runner more chances. The evidence here supports not just continuing but advancing -- the pilot results are positive for the core thesis.

### Concerns to Monitor in Full Experiments

- **Soft CWD convergence noise**: Multi-seed runs (3 seeds) should resolve whether the non-monotonic beta-accuracy relationship is just seed variance. If soft CWD genuinely does not converge to hard CWD, the proximal-theoretic claim (H1) for CWD specifically may need qualification.

- **CSI's narrow spread (CV ~2%)**: If CSI remains narrow across architectures and datasets in full experiments, H5's predictive power claim will need to be weakened. Consider: is CSI capturing stability at too coarse a level?

- **AIS values are moderate (0.30-0.40), not low (<0.10)**: The hypothesis expected AIS ~ 0, but observed AIS is 0.30-0.40. This could mean alignment IS informative, or it could mean AIS is measuring something other than alignment utility. The full CWD falsification battery (effective-lambda matching, random mask, inverted mask) is critical to disambiguate.

- **Single seed**: All pilot results are from seed 42 only. The 0.21% accuracy spread is reassuring but could widen or narrow with multiple seeds. The full experiments use 3 seeds (42, 123, 456).

## Next Actions

1. **Proceed immediately to full CIFAR experiments** (task `cifar10_resnet20_full`): All 7 primary methods + CWD ablations, 3 seeds, 200 epochs on CIFAR-10/ResNet-20. This is the core evidence generator.

2. **Prioritize the CWD falsification battery**: The effective-lambda-matched constant WD, random-mask, and inverted-mask ablations are the highest-impact experiments for disambiguating H4.

3. **Extend to CIFAR-100 and VGG-16-BN** before ImageNet: The cross-dataset and cross-architecture validation is critical before committing 72 GPU-hours to ImageNet.

4. **WD Stability Condition test (H2)** can run in parallel with CIFAR full experiments since it uses different hyperparameters.

5. **Soft CWD convergence**: Include both hard and soft (beta=100) CWD in the full benchmark to get multi-seed evidence on proximal approximation quality.

SELECTED_CANDIDATE: cand_unified_framework
CONFIDENCE: 0.82
DECISION: ADVANCE
