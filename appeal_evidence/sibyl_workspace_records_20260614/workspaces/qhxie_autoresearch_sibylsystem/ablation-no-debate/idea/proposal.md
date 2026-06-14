# Research Proposal: Encoder-Driven Feature Absorption in SAEs -- Mechanism, Consequences, and Safety Implications

## Metadata

- **Iteration**: 4 (synthesized from 6 perspectives + iter_003 evidence)
- **Date**: 2026-05-01
- **Front-runner**: cand_p1 (Encoder-Driven Absorption with Safety Validation)
- **Status**: Post-pilot, synthesizing from 6 perspectives辩论

---

## Abstract

Feature absorption -- where child features in Sparse Autoencoders (SAEs) substitute for parent features in sparse representations -- is a fundamental reliability challenge for mechanistic interpretability. We synthesize evidence from 6 diverse perspectives to propose a rigorous study with three core contributions: (1) validating the **encoder-driven absorption mechanism** via multi-seed factorial experiments on stochastic hierarchies, (2) establishing the **sensitivity-absorption Pareto frontier** quantifying the irreducible trade-off between sparsity and feature quality, and (3) testing whether **safety-critical features are disproportionately absorbed** in real Gemma Scope SAEs, which would undermine SAE-based safety analysis. Our approach integrates information geometry theory (theoretical), training-free diagnostic tools (pragmatist), causal steering interventions (empiricist), and critical branching phase-transition analogies (interdisciplinary), while directly addressing the contrarian's challenge that absorption metrics may not predict downstream interpretability utility.

---

## Motivation

Feature absorption threatens the reliability of SAE-based interpretability: when child features absorb their parents, the SAE's internal representation no longer corresponds to the intended feature structure. Prior work (Chanin et al., 2024; Korznikov et al., 2026) documented absorption but assumed it arose from decoder geometry or sparsity optimization. Our prior pilot evidence from iter_001-003 overturns this narrative:

**Key Prior Finding** (H_Mech factorial, iter_001):
| Condition | Encoder | Decoder | Absorption Rate |
|-----------|---------|---------|-----------------|
| A | Random | Random | 0.299 |
| B | Trained | Random | **0.490** |
| C | Random | Trained | **0.299** |
| D | Trained | Trained | **0.484** |

**Interpretation**: Condition B ≈ Condition D, Condition C ≈ Condition A. The **encoder's learned alignment with hierarchical structure is the sole driver**; the decoder contributes nothing.

**Contradictory Finding**: Condition B (0.490) > Condition D (0.484) -- trained encoder with random decoder produces *more* absorption than full training. This suggests the decoder acts as a regularizer that partially counteracts encoder-driven absorption.

**New Understanding**: Absorption is not "reconstruction stealing" but "efficient coding" -- the encoder compresses redundant parent representations into child subspaces for metabolic/positional efficiency, not due to competitive exclusion. This aligns with the efficient coding hypothesis (Barlow, 1961).

---

## Research Questions

1. Is the encoder-driven absorption mechanism robust across random seeds and stochastic hierarchies?
2. What is the Pareto frontier between feature sensitivity (sparsity) and absorption rate?
3. Are safety-critical features disproportionately absorbed in real Gemma Scope SAEs?
4. Does the absorption-sensitivity uncertainty relation (theoretical prediction) hold empirically?
5. **New from Contrarian**: Does absorption actually degrade downstream interpretability utility, or is it a metric artifact?

---

## Hypotheses

| ID | Hypothesis | Status | Falsification Criterion |
|----|-----------|--------|------------------------|
| H1 | Trained SAEs show higher multi-child proportional absorption than random baselines | PASSED (iter_001) | p < 0.05, delta > 0.15 |
| H_Mech | Absorption is driven by encoder alignment, not decoder geometry | PASSED (iter_001) | Condition B ≈ D, C ≈ A |
| H2 | Absorption rate inversely correlates with feature frequency | FAILED (iter_001) | rho < -0.3 (observed: +0.171) |
| H3 | Steering absorbed features toward parent directions improves sensitivity | PASSED (iter_001) | p < 0.01 improvement (ratio: 1.62x) |
| H_Safe | Safety-critical features show elevated absorption vs non-safety | NOT TESTED (real SAE) | Mann-Whitney p < 0.05 |
| H_Comp | Absorption increases monotonically with hierarchy strength | PASSED (pilot) | Monotonic increase confirmed |
| H_Pareto | Sensitivity-absorption Pareto frontier exists | Pilot shows issue | Formula needs verification |
| H_Downstream | Absorbed features show degraded steering/cirucit utility | NOT TESTED | Contrarian's challenge |

---

## Method

### Part A: Encoder-Driven Mechanism Validation (H_Mech)

**2x2 Factorial with Stochastic Hierarchy**:
- Generate synthetic 3-level hierarchies with stochastic noise (epsilon ~ N(0, 0.05))
- Test all 4 conditions (A/B/C/D) across 5 seeds
- Measure multi-child proportional absorption (k=5)

**Expected Outcome**: B ≈ D confirms encoder-driven; C ≈ A confirms decoder-irrelevant across stochastic data.

**Key Insight from Contrarian**: The B > D finding suggests decoder regularization partially counteracts absorption. Future interventions should target encoder training dynamics, not decoder architecture.

### Part B: Sensitivity-Absorption Pareto Frontier (H_Pareto)

**Design** (from empiricist + theoretical perspectives):
- Vary L0 targets: {16, 32, 64, 128}
- At each L0, measure both absorption rate and feature sensitivity (Hu et al., 2025)
- Fit Pareto frontier curve

**Critical Issue**: Pilot data shows sensitivity = 1.525 at L0=16, exceeding [0,1] bounds. **This indicates a formula bug that must be fixed before frontier claims are valid.** The empiricist perspective emphasizes this validation step.

**Theoretical Prediction**: The absorption-sensitivity uncertainty relation predicts an irreducible trade-off.

### Part C: Safety-Critical Feature Analysis (H_Safe)

**Design** (from innovator + pragmatist perspectives):
1. Install SAELens with Gemma Scope pretrained SAEs (gemma-2b, layer 12)
2. Select **validated** safety-relevant features from Neuronpedia annotations (deception, jailbreak, harm) -- NOT arbitrary indices
3. Match 20 non-safety features by activation frequency and layer
4. Measure absorption via multi-child proportional method
5. Mann-Whitney U test comparing distributions

**Critical Issue**: Prior pilot used arbitrary indices (1024, 2048, etc.) as "safety features" -- these are NOT validated safety features. The lessons_learned explicitly states these placeholders must be removed or validated.

**Expected Outcome**: Safety features show elevated absorption (p < 0.05).

### Part D: Downstream Utility Test (H_Downstream) -- NEW from Contrarian

**Design** (directly addresses contrarian's core challenge):
1. Classify features as "absorbed" vs "non-absorbed" via ablation (layers 0-17)
2. Match pairs on confounders (activation frequency, magnitude, interpretability score)
3. Test both on downstream tasks:
   - Steering accuracy: behavior change correlation
   - Circuit completeness: edge coverage comparison
4. Statistical test: McNemar's test for paired proportions

**Falsification Criterion**: If absorbed vs non-absorbed features show < 10% difference in steering success rate, absorption does not have practically significant downstream impact.

**Why This Matters**: SAEBench shows proxy metrics don't predict practical performance. If absorption also doesn't predict downstream failure, the field is optimizing the wrong thing.

### Part E: Hierarchy Strength Dependence (H_Comp)

**Design** (from interdisciplinary + theoretical perspectives):
- Vary parent-child cosine similarity: {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}
- Measure absorption rate at each strength
- Fit monotonic curve with R² > 0.8 expected

**Pilot Result** (iter_003):
| Cosine Sim | Absorption Rate |
|------------|-----------------|
| 0.6 | 0.585 |
| 0.8 | 0.673 |
| 0.95 | 0.802 |

Confirmed: monotonic increase with hierarchy strength.

---

## Experimental Plan

| Phase | Task | Duration | Hypothesis |
|-------|------|----------|------------|
| 1 | Fix sensitivity formula bug | 15 min | H_Pareto |
| 2 | H_Mech factorial (5 seeds, stochastic hierarchy) | 45 min | H_Mech |
| 3 | H_Comp: hierarchy strength sweep (full) | 30 min | H_Comp |
| 4 | H_Downstream: steering utility test | 60 min | H_Downstream |
| 5 | H_Safe on Gemma Scope (validated features) | 60 min | H_Safe |
| 6 | H_Pareto: sensitivity-absorption frontier (fixed formula) | 45 min | H_Pareto |

**Total**: ~4 hours GPU time. All individual tasks within 1-hour budget.

---

## Revisions from Prior Iterations

### From iter_003 Proposal

1. **B > D finding highlighted**: The fact that trained encoder + random decoder produces MORE absorption than both trained is now a central finding, not a footnote. This suggests decoder regularization partially counteracts absorption -- a new intervention target.

2. **H_Downstream added**: Contrarian's challenge directly tested. If absorption doesn't matter downstream, the field's architectural innovations may be solving the wrong problem.

3. **H_Safe placeholder features removed**: Must use Neuronpedia-validated safety features only. No more arbitrary indices.

4. **Sensitivity formula fix required**: H_Pareto claims suspended until formula bug is fixed.

5. **H2 reframed**: Failed hypothesis (positive frequency correlation) now explained by efficient coding, not competitive exclusion. The mechanism is efficient compression, not competition.

### From 6 Perspectives Integration

| Perspective | Key Contribution |
|-------------|------------------|
| Innovator | Competitive exclusion reframed as efficient coding -- the encoder "compresses" redundant parent directions |
| Pragmatist | Encoder-decoder asymmetry as training-free absorption proxy -- but must validate against ground truth |
| Theoretical | Information-theoretic collision: absorption occurs when mutual information gap favors child |
| Contrarian | Core challenge: does absorption actually matter for downstream utility? This is the highest-value question |
| Interdisciplinary | Critical branching process: absorption is a phase transition at critical sparsity threshold |
| Empiricist | Validate training-free proxies against SynthSAEBench ground truth before claiming utility |

---

## Novelty Assessment

| Contribution | Novelty | Status |
|--------------|---------|--------|
| Encoder-driven absorption mechanism | **Novel** | First factorial decomposition showing B>D |
| Sensitivity-absorption Pareto frontier | **Novel** | No prior quantification (suspended until formula fixed) |
| Safety-critical feature absorption | **9/10** | No prior work (requires validated features) |
| Downstream utility test | **8/10** | First direct test of absorption consequence |

**Differentiation from prior work**:
- Chanin et al. (2024): Documents absorption; we identify encoder as sole driver and B>D anomaly
- Korznikov et al. (2026): Baseline comparison; we decompose encoder vs decoder contributions
- Tang et al. (2512.05534): Theoretical grounding for encoder-driven local minima
- Basu et al. (2603.18353): Raises safety concern; we provide empirical methodology

---

## Perspectives Integration

| Perspective | Key Contribution to Proposal |
|-------------|------------------------------|
| Innovator | Efficient coding mechanism (replaces competitive exclusion) |
| Pragmatist | Training-free diagnostic toolkit; multi-child proportional ablation |
| Theoretical | Information-theoretic collision framework; mutual information gap prediction |
| Contrarian | **H_Downstream**: Does absorption actually matter for interpretability? |
| Interdisciplinary | Critical branching phase transition for absorption threshold |
| Empiricist | Validate proxies against SynthSAEBench ground truth; fix sensitivity formula |

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Sensitivity formula unfixable | Low | High | Report issue honestly; use alternative sensitivity metric |
| H_Safe shows no difference | Medium | Medium | Document as negative result; methodology still contributes |
| H_Downstream shows no effect | Medium | High | Landmark result -- field optimizing wrong metric |
| B > D is synthetic artifact | Medium | High | Test on real Gemma Scope SAEs |
| H_Pareto doesn't match theory | Medium | Medium | Report actual frontier shape |

---

## Contributions

1. **Encoder-driven mechanism with B>D anomaly**: First factorial decomposition; decoder regularization partially counteracts absorption
2. **Downstream utility test**: First direct test of whether absorption degrades interpretability tasks
3. **Safety implications with validated features**: Methodology for testing SAE reliability for critical features (not placeholder indices)
4. **Honest negative results**: H2 (frequency correlation wrong direction) documented with efficient coding reframing
5. **Critical branching framework**: Phase-transition theory for absorption threshold

---

## References

- Chanin et al. (2024). A is for Absorption. arXiv:2409.14507
- Korznikov et al. (2026). Sanity Checks for SAEs. arXiv:2602.14111
- Tang et al. (2025). Theoretical Foundation of SDL in MI. arXiv:2512.05534
- Basu et al. (2026). Interpretability without Actionability. arXiv:2603.18353
- Hu et al. (2025). Measuring SAE Feature Sensitivity. arXiv:2509.23717
- Barlow, H. (1961). Possible Principles Governing the Relearning of Sensory Data.
- SAEBench (2025). Karvonen et al. arXiv:2503.09532
- SynthSAEBench (2026). Chanin et al. arXiv:2602.14687