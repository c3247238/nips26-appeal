# Supervisor Review -- Iteration 17

## Executive Summary

**Score: 7.0/10 | Verdict: CONTINUE**

Iter_017 applies targeted editorial fixes to the iter_016 paper: the LR schedule is corrected from "cosine annealing" to "step decay," lambda is consistently stated as 1e-4, the future work numbering is fixed, and the cross-dataset figure now labels "45 epochs." These are genuine corrections that resolve three of iter_016's issues. However, no new experiments were run, and the fundamental barriers to scoring above 7.0 are experimental, not editorial. The paper has hit the 7.0 ceiling for the fourth time (iter_005, iter_012, iter_014, iter_016, now iter_017). Breaking through requires filling the experimental evidence gaps identified below.

## Dimension Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Novelty** | 6.5 | EqWD is the natural next step after Defazio's equilibrium result: track deviation via EMA, modulate WD proportionally. The additive form is the simplest possible controller. No alternative formulations are tested. The CAWD negative result is a genuine contribution but needs deeper mechanistic analysis. Defazio's corrective term is closely related and insufficiently differentiated. |
| **Soundness** | 7.0 | The theoretical framing is improved: Proposition 2 has been downgraded to "Empirical Observation." Proposition 1 is trivially true by definition (acknowledged). The AIS diagnostic is the real contribution but is validated only on CIFAR where EqWD does NOT win, not on ImageNet where it does. The WD heatmap lambda values (5.2e-4 to 6.2e-4) are inconsistent with lambda_base=1e-4, which needs investigation. |
| **Experiments** | 7.0 | Comprehensive 7-method x 3-seed ImageNet comparison is a genuine achievement. Statistical methodology (bootstrap CIs, Cohen's d, honest caveats) is above community norms. However: (1) the WD inflation confound is uncontrolled; (2) no AdamW experiments; (3) EqWD missing from CIFAR-10 table; (4) beta=5.0 unvalidated; (5) VGG baselines missing; (6) 45-epoch count not stated in setup section. |
| **Reproducibility** | 7.0 | Lambda and LR schedule now correctly described. Training configurations documented. But: epoch count not explicit in setup section, no code release, BO details missing, no raw experiment data in workspace. |

## What Changed from Iter_016

### Fixed
1. LR schedule: "cosine annealing" -> "step decay (10x reduction at epoch 30)" -- correct
2. Lambda: consistently 1e-4 throughout (was 5e-4/1e-4 discrepancy in iter_016 text)
3. Future work numbering: "First...Second...Third...Fourth...Finally" (was skipping "Fourth")
4. Cross-dataset figure subtitle: now shows "45 epochs"

### Not Fixed (Recurring Issues)
1. **Budget-matched FixedWD** -- P0 experiment, still not run (4+ iterations)
2. **AdamW experiment** -- still absent
3. **CIFAR-10 table missing EqWD** -- still absent
4. **Beta=5.0 multi-seed** -- still single-seed only
5. **AIS on ImageNet** -- still CIFAR-only
6. **45-epoch count in Section 4.1** -- still implicit
7. **VGG baselines** -- still missing
8. **DefazioCorrective in CIFAR-10 table** -- still undefined

### New Issue Discovered
- **WD heatmap lambda discrepancy**: Figure 5 shows lambda_t values of 5.2e-4 to 6.2e-4. With lambda_base = 1e-4, this implies phi_l(t) = 5.2-6.2x, which contradicts the claim that "phi_l(t) rarely exceeds 1.1" on CIFAR-100. This needs immediate investigation -- it could indicate a data integrity issue or a figure labeling error.

## Critical Issue: Effective WD Inflation

This is the single most important issue in the paper and has been unresolved for 4+ iterations despite being flagged as P0 in every review and reflection.

**The problem**: EqWD's modulation factor phi_l(t) >= 1 by design. This means EqWD always applies MORE total weight decay than FixedWD with the same lambda_base. The paper's +0.38% ImageNet improvement could be entirely explained by EqWD simply applying more regularization, not by adaptive modulation.

**Why it matters**: The paper's central contribution is "adaptive per-layer modulation improves generalization." Without a budget-matched control, this claim is indistinguishable from "more weight decay is better" -- a trivially true finding for under-regularized models.

**The experiment**: Measure EqWD's average effective lambda over training. Run FixedWD at that lambda. 3 seeds, ImageNet, ~9 GPU-hours. This is the paper's highest-ROI experiment by far.

**Both outcomes are publishable**: If budget-matched FixedWD matches EqWD, the narrative shifts to "EqWD automatically discovers the optimal WD strength without tuning" -- still a valid contribution. If EqWD still wins, the adaptive modulation claim is validated.

## Figure Quality Assessment

1. **Cross-dataset comparison (Figure 1)**: Good. Now correctly labels "45 epochs." Numbers match Table 2. Error bars visible and proportionate.

2. **Training curves (Figure 2)**: Good. Clearly shows 45 epochs and step decay at epoch 30. EqWD (dark magenta) achieves highest final accuracy. All methods similar in training loss -- only separating in test accuracy.

3. **Ratio trajectories (Figure 4)**: Problematic. After the initial transient (~20 steps), all 6 layers converge to nearly identical r_t values (0.01-0.02) with minimal visible differentiation. The text claims "later layers show larger ratios with higher variance" -- this is not supported by the figure. The figure currently supports the counter-argument (minimal heterogeneity = per-layer modulation unnecessary) more than the paper's narrative.

4. **WD heatmap (Figure 5)**: Data discrepancy. Lambda_t values range from 5.2e-4 to 6.2e-4. With lambda_base=1e-4, this implies modulation factors of 5.2-6.2x. But the paper says modulation "rarely exceeds 1.1" on CIFAR-100. Either the heatmap is showing a different quantity (lambda*gamma? lambda*gamma*phi?), or lambda_base is not 1e-4, or there is a plotting error. Must be resolved.

5. **Accuracy vs Stability (Appendix)**: Good. EqWD correctly in upper-left quadrant. Clear separation from other methods.

6. **Radar chart (Figure 6)**: Acceptable but adds limited information beyond the main table.

## Path to 7.5

Cost: ~12 GPU-hours + 2 hours of text edits

1. Budget-matched FixedWD on ImageNet (9 GPU-hours) -- resolves the critical confound
2. EqWD + CAWD on CIFAR-10 (1 GPU-hour) -- fills table gap
3. Beta=5.0 multi-seed on CIFAR-100 (0.5 GPU-hours) -- validates or retracts claim
4. State 45-epoch count in Section 4.1 (0 GPU) -- reproducibility fix
5. Resolve WD heatmap lambda discrepancy (0 GPU) -- data integrity fix

## Path to 8.0

Additional cost: ~22 GPU-hours

6. AdamW experiment on CIFAR-100 (0.5 GPU-hours)
7. AIS on ImageNet (3 GPU-hours)
8. ImageNet ratio trajectories figure (0 GPU if data exists from training logs)
9. 90-epoch ImageNet comparison (18 GPU-hours)
10. VGG baselines (1 GPU-hour)
11. Differentiate from Defazio's corrective term in Related Work (0 GPU)

## Scoring Calibration Note

This paper has scored 7.0 in four separate iterations. The score is not arbitrary -- it reflects a paper with a clean method, honest writing, and decent ImageNet results, but with a fundamental experimental gap (the WD inflation confound) that prevents acceptance at a top venue. A NeurIPS reviewer would write: "The method is simple and well-described. The ImageNet results are promising. However, since phi >= 1, I cannot distinguish whether the improvement comes from adaptive modulation or simply from applying more regularization. A budget-matched FixedWD control is essential. Additionally, the paper should validate on AdamW and include EqWD in all benchmark tables." This is a borderline reject/accept paper that needs ~12-30 GPU-hours of additional experiments to become a clear accept.
