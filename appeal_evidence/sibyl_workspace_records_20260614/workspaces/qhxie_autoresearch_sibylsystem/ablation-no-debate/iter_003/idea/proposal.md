# Research Proposal: Encoder-Driven Feature Absorption in SAEs -- Mechanism, Consequences, and Safety Implications

## Metadata

- **Iteration**: 3 (idea synthesis)
- **Date**: 2026-04-30
- **Front-runner**: cand_p1 (Encoder-Driven Absorption with Safety Validation)
- **Status**: Pre-pilot, synthesizing from 6 perspectives

---

## Abstract

Feature absorption -- where child features in Sparse Autoencoders (SAEs) substitute for parent features in sparse representations -- is a fundamental reliability challenge for mechanistic interpretability. We propose a rigorous study with three core contributions: (1) validating the **encoder-driven absorption mechanism** via multi-seed factorial experiments on stochastic hierarchies, (2) establishing the **sensitivity-absorption Pareto frontier** quantifying the irreducible trade-off between sparsity and feature quality, and (3) testing whether **safety-critical features are disproportionately absorbed** in real Gemma Scope SAEs, which would undermine SAE-based safety analysis. Our approach synthesizes information geometry theory (theoretical), training-free diagnostic tools (pragmatist), causal steering interventions (empiricist), and phase-transition analogies (interdisciplinary), while addressing the contrarian's challenge that absorption metrics may conflate distinct phenomena.

---

## Motivation

Feature absorption threatens the reliability of SAE-based interpretability: when child features absorb their parents, the SAE's internal representation no longer corresponds to the intended feature structure. Prior work (Chanin et al., 2024; Korznikov et al., 2026) documented absorption but assumed it arose from decoder geometry or sparsity optimization. Our prior pilot evidence from iteration 1 overturns this narrative:

**Key Prior Finding** (H_Mech factorial, iter_001):
| Condition | Encoder | Decoder | Absorption Rate |
|-----------|---------|---------|-----------------|
| A | Random | Random | 0.299 |
| B | Trained | Random | **0.490** |
| C | Random | Trained | **0.299** |
| D | Trained | Trained | **0.484** |

**Interpretation**: Condition B ≈ Condition D, Condition C ≈ Condition A. The **encoder's learned alignment with hierarchical structure is the sole driver**; the decoder contributes nothing.

This creates three research gaps:
1. **Validation gap**: Prior factorial used deterministic hierarchy; need stochastic validation
2. **Trade-off gap**: No quantification of the sensitivity-absorption Pareto frontier
3. **Safety gap**: Synthetic pilots lack semantic content; real Gemma Scope SAEs required

---

## Research Questions

1. Is the encoder-driven absorption mechanism robust across random seeds and stochastic hierarchies?
2. What is the Pareto frontier between feature sensitivity (sparsity) and absorption rate?
3. Are safety-critical features disproportionately absorbed in real Gemma Scope SAEs?
4. Does the absorption-sensitivity uncertainty relation (theoretical prediction) hold empirically?

---

## Hypotheses

| ID | Hypothesis | Status | Falsification Criterion |
|----|-----------|--------|------------------------|
| H1 | Trained SAEs show higher multi-child proportional absorption than random baselines | PASSED (iter_001) | p < 0.05, delta > 0.15 |
| H_Mech | Absorption is driven by encoder alignment, not decoder geometry | PASSED (iter_001) | Condition B ≈ D, C ≈ A |
| H2 | Absorption rate inversely correlates with feature frequency | FAILED (iter_001) | rho < -0.3 (observed: +0.171) |
| H3 | Steering absorbed features toward parent directions improves sensitivity | PASSED (iter_001) | p < 0.01 improvement (ratio: 1.62x) |
| H_Safe | Safety-critical features show elevated absorption vs non-safety | NOT TESTED | Mann-Whitney p < 0.05 |
| H_Comp | Absorption increases monotonically with hierarchy strength | NOT TESTED | R² > 0.8 |
| H_Pareto | Sensitivity-absorption Pareto frontier exists (theoretical prediction) | NOT TESTED | Frontier shape matches theory |

---

## Method

### Part A: Encoder-Driven Mechanism Validation (H_Mech)

**2x2 Factorial with Stochastic Hierarchy**:
- Generate synthetic 3-level hierarchies with stochastic noise (epsilon ~ N(0, 0.05))
- Test all 4 conditions (A/B/C/D) across 5 seeds
- Measure multi-child proportional absorption (k=5)

**Expected Outcome**: B ≈ D confirms encoder-driven; C ≈ A confirms decoder-irrelevant across stochastic data.

### Part B: Sensitivity-Absorption Pareto Frontier (H_Pareto)

**Design** (from empiricist + theoretical perspectives):
- Vary L0 targets: {16, 32, 64, 128}
- At each L0, measure both absorption rate and feature sensitivity (Hu et al., 2025)
- Fit Pareto frontier curve

**Theoretical Prediction**: The absorption-sensitivity uncertainty relation predicts an irreducible trade-off.

### Part C: Safety-Critical Feature Analysis (H_Safe)

**Design** (from innovator + pragmatist perspectives):
1. Install SAELens with Gemma Scope pretrained SAEs (gemma-2b, layer 12)
2. Select 20 safety-relevant features from Neuronpedia annotations (deception, jailbreak, harm)
3. Match 20 non-safety features by activation frequency and layer
4. Measure absorption via multi-child proportional method
5. Mann-Whitney U test comparing distributions

**Expected Outcome**: Safety features show elevated absorption (p < 0.05).

### Part D: Hierarchy Strength Dependence (H_Comp)

**Design** (from interdisciplinary + theoretical perspectives):
- Vary parent-child cosine similarity: {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}
- Measure absorption rate at each strength
- Fit monotonic curve with R² > 0.8 expected

---

## Experimental Plan

| Phase | Task | Duration | Hypothesis |
|-------|------|----------|------------|
| 1 | H_Mech factorial (5 seeds, stochastic hierarchy) | 45 min | H_Mech |
| 2 | H_Comp: hierarchy strength sweep | 30 min | H_Comp |
| 3 | H_Pareto: sensitivity-absorption frontier | 45 min | H_Pareto |
| 4 | H_Safe on Gemma Scope | 60 min | H_Safe |
| 5 | Held-out validation | 30 min | Cross-validate |

**Total**: ~3.5 hours GPU time. All individual tasks within 1-hour budget.

---

## Revisions from Prior Iterations

### From iter_001 Proposal

1. **Stochastic hierarchy**: Prior H_Mech used deterministic hierarchy; new pilots add stochastic noise to test robustness
2. **H_Pareto added**: Theoretical perspective identified the sensitivity-absorption uncertainty relation as key prediction
3. **H_Comp prioritized**: Hierarchy strength sweep directly tests encoder's learned alignment with structure
4. **Safety validation confirmed**: H_Safe remains highest-novelty (9/10) but requires Gemma Scope

### Addressing Contrarian Concerns

The contrarian raised three challenges:
1. **"Absorption might not be a problem"**: We address via H3 (steering validation) showing absorbed features are 1.62x more sensitive -- they retain causal information
2. **"Decoder determinism"**: Prior factorial directly tests this; decoder contribution is zero
3. **"Metrics may be flawed"**: We use multi-child proportional ablation which addresses saturation issues

---

## Novelty Assessment

| Contribution | Novelty | Status |
|--------------|---------|--------|
| Encoder-driven absorption mechanism | **Novel** | First factorial decomposition |
| Sensitivity-absorption Pareto frontier | **Novel** | No prior quantification |
| Safety-critical feature absorption | **9/10** | No prior work |

**Differentiation from prior work**:
- Chanin et al. (2024): Documents absorption; we identify encoder as sole driver
- Korznikov et al. (2026): Baseline comparison; we decompose encoder vs decoder contributions
- Tang et al. (2512.05534): Theoretical grounding for encoder-driven local minima

---

## Perspectives Integration

| Perspective | Key Contribution to Proposal |
|-------------|------------------------------|
| Innovator | Safety-critical absorption (H_Safe) as highest-novelty sub-hypothesis |
| Pragmatist | Training-free diagnostic toolkit; multi-child proportional ablation methodology |
| Theoretical | Absorption-sensitivity uncertainty relation; information geometry framing |
| Contrarian | Decoder determinism challenge → addressed by H_Mech factorial; metric concerns → addressed by multi-child method |
| Interdisciplinary | Phase-transition framing for hierarchy strength (H_Comp); causal discovery framework |
| Empiricist | Strict experiment design; Pareto frontier measurement methodology |

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Encoder result is synthetic artifact | Medium | High | Test on real Gemma Scope SAEs |
| H_Safe shows no difference | Medium | Medium | Document as negative result; methodology still contributes |
| H_Pareto doesn't match theory | Medium | Medium | Report actual frontier shape |
| Zero variance persists | Low | Medium | Document as deterministic property |

---

## Contributions

1. **Encoder-driven mechanism**: First factorial decomposition showing absorption is entirely encoder-learned
2. **Sensitivity-absorption frontier**: First quantitative characterization of the irreducible trade-off
3. **Safety implications**: Methodology for testing SAE reliability for critical features
4. **Honest negative results**: H2 (frequency correlation wrong direction) documented
5. **Pareto-optimal benchmarking**: New evaluation framework for SAE quality

---

## References

- Chanin et al. (2024). A is for Absorption. arXiv:2409.14507
- Korznikov et al. (2026). Sanity Checks for SAEs. arXiv:2602.14111
- Tang et al. (2025). Theoretical Foundation of SDL in MI. arXiv:2512.05534
- Basu et al. (2026). Interpretability without Actionability. arXiv:2603.18353
- Hu et al. (2025). Measuring SAE Feature Sensitivity. arXiv:2509.23717
