# Result Debate Verdict

**Score: 5.5 / 10**

---

## Key Conclusion

The study's most valuable finding is **not** what was originally hypothesized. The cross-domain absorption characterization (H1) is invalidated by universal control failure, and the hypothesis that absorption is mostly hierarchy-driven (H2) is refuted by a 69x margin. Instead, the data has produced an unexpected and more important result: **the standard absorption metric (Chanin et al.) conflates hedging artifacts with genuine hierarchy-driven absorption**, with 98.6% of detected "absorption" at low L0 being hedging, not competitive exclusion. Only 9 out of 1,195 tested words (0.75%) exhibit genuine hierarchy-driven absorption.

This confound decomposition, combined with the discovery that the metric does not transfer cleanly to JumpReLU SAEs (all shuffled controls exceed measured rates), constitutes a genuine methodological contribution to the mechanistic interpretability community. The rate-distortion CMI diagnostic shows correct qualitative direction (rho=-0.383, large effect) but needs one clean replication to be credible.

---

## Action Plan

**PROCEED** with three blocking experiments totaling 3.5-5 GPU-hours:

1. **Control failure diagnosis** (2-3h): Determine WHY shuffled controls exceed measured absorption. Analytical expected-rate computation + null-domain benchmark + threshold recalibration.
2. **CMI replication at L0=22** (1h): Pre-registered d'=10 with perfect probes (F1=1.0). If rho < -0.3 and p < 0.05, the theoretical pillar is secured.
3. **Activation patching on 9 core absorbed words** (0.5-1h): Zero child features, check parent recovery. Provides metric-independent ground truth.

**Do not write the paper until P1 and P2 are complete.** Paper framing is conditional on these results.

### Paper Reframing

**Title direction**: "Decomposing Feature Absorption: Hedging Artifacts, Metric Portability, and Rate-Distortion Diagnostics for Sparse Autoencoders"

**Three pillars** (ordered by evidence strength):
1. Multi-L0 confound decomposition (98.6% hedging at low L0, monotonic L0 profile)
2. Metric portability failure (universal control failure on JumpReLU SAEs)
3. Rate-distortion CMI diagnostic (conditional on replication)

### Expected Venue

| Outcome | Probability | Target |
|---|---|---|
| Best (all 3 experiments pass) | 25% | NeurIPS/ICML main |
| Expected (1-2 pass) | 55% | AAAI/EMNLP or workshop |
| Floor (all fail) | 20% | TMLR or workshop |

The floor is still publishable. The ceiling is a competitive main-conference submission.
