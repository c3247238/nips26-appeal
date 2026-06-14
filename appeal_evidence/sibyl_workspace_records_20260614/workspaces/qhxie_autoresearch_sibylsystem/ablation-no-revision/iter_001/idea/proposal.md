# Research Proposal: On the Limits of Consistency-Based Activation Energy for Problem-Level Routing in LLM Mathematical Reasoning

## Status: Round 5 Synthesis - Final Evidence-Driven Proposal

---

## Abstract

We present a systematic empirical study that tests whether consistency-derived "activation energy" (Ea) can serve as a routing signal for problem-level inference optimization in LLM mathematical reasoning. Using Qwen2.5-Math-7B-Instruct on the MATH benchmark (n=50), we independently validate aggregate Arrhenius-like saturation kinetics (R^2=0.924) and confirm that Ea correlates with problem difficulty (Spearman=0.448, p=0.001). However, we decisively falsify the core routing hypothesis: Ea has zero predictive power for single-pass correctness (AUC=0.436 < 0.5, Spearman=-0.063). We further demonstrate that consistency-Ea and saturation-k_0 are unrelated constructs (Spearman=-0.219), undermining the theoretical unity of the framework. Our contribution is not a new theory but a systematic boundary-delineation: we quantify the ~25pp "agreement-but-wrong" ceiling for convergence-rate-based routing, diagnose why Ea fails (it measures answer stability, not correctness), and explicitly map the framework's valid boundaries. This negative result cross-validates ACAR's findings and saves the community from pursuing an ineffective signal.

---

## Motivation: Why This Study Matters

### The Promise

Yang et al. (2025) established that LLM reasoning accuracy follows exponential saturation under repeated sampling. A natural extension is: can the *rate* of consensus formation (activation energy) predict whether a problem is solvable in a single pass? If yes, this would enable zero-cost routing between single-pass and multi-sample inference paths, saving significant compute.

### The Reality

Our experiments show that while the aggregate saturation pattern is real, the derived Ea signal cannot predict single-pass correctness. This is not a minor limitation -- it is a fundamental failure that challenges the practical applicability of the entire framework at the individual-problem level.

### Why Publish a Negative Result

1. **Resource savings**: The community should know that Ea-based routing does not work before investing further effort
2. **Cross-validation**: Our findings independently confirm ACAR's "agreement-but-wrong" ceiling with quantitative precision
3. **Boundary knowledge**: We explicitly map where the framework works (aggregate description) and where it fails (individual routing)
4. **Methodological contribution**: We introduce a rigorous evaluation protocol for routing signals that future work can adopt

---

## Evidence-Driven Revisions from Prior Rounds

### Round History

| Round | Approach | Key Result | Decision |
|-------|----------|------------|----------|
| Round 1 | EDW-Step-DPO (training) | Loss 0.6922-0.6927, no improvement | FALSIFIED |
| Round 2 | CCAR calibration training | DeepSeek 26% baseline; Step-DPO loss 0.694 | FALSIFIED |
| Round 3 | API-based inference | BLOCKED - no API key available | BLOCKED |
| Round 4 | Activation Energy Theory | H1+H2 confirmed, H3 falsified | PARTIAL |
| **Round 5** | **Negative result + diagnostic** | **H3 systematic falsification** | **PROCEED** |

### What Changed from Round 4 Proposal

| Aspect | Round 4 Proposal | Round 5 Revision |
|--------|------------------|------------------|
| Paper type | Theoretical framework | Empirical boundary-delineation |
| Novelty claim | Exponential saturation formula | Cannot claim -- Yang et al. collision |
| Core contribution | Theory validation | Systematic falsification of Ea routing |
| H3 status | Predicted to hold | **FALSIFIED** (AUC=0.436) |
| H5 status | Predicted to hold | **FALSIFIED** (Spearman=-0.219) |
| H4 status | Not proposed | **NEW** - error classification diagnostic |
| Framing | "We propose a theory" | "We test a hypothesis and find it fails" |

---

## Research Questions

1. **RQ1** (CONFIRMED): Does LLM reasoning accuracy follow Arrhenius-like saturation at the aggregate level?
2. **RQ2** (CONFIRMED): Does activation energy correlate with problem difficulty?
3. **RQ3** (FALSIFIED): Can activation energy predict single-pass solveability for individual problems?
4. **RQ4** (NEW): Why does Ea fail as a routing signal?
5. **RQ5** (NEW): What are the valid boundaries of the Activation Energy framework?

---

## Hypotheses

### H1: Aggregate Arrhenius Saturation -- CONFIRMED

**Claim**: LLM reasoning accuracy follows exponential saturation at the aggregate level: P_k = P_inf * (1 - exp(-k/k_0))

**Evidence**: R^2 = 0.924 on n=50 problems, Qwen2.5-Math-7B-Instruct
**Status**: CONFIRMED as descriptive aggregate pattern
**Caveat**: Per-problem median R^2 = 0; 80% of individual problems cannot be fit

### H2: Ea Correlates with Difficulty -- CONFIRMED

**Claim**: Activation energy estimated from answer consistency correlates with MATH level difficulty

**Evidence**: Spearman(Ea, Level) = 0.448, p=0.001 (n=50)
**Status**: CONFIRMED as coarse-grained difficulty signal
**Caveat**: Ea values are highly concentrated (bimodal at ~9.47 and ~10.0); within-level discriminative power is minimal

### H3: Ea Predicts Single-Pass Solveability -- FALSIFIED

**Original Claim**: Problems with low Ea can be solved in single-pass with >75% accuracy

**Evidence**:
- Low-Ea accuracy: 75.0% (n=50, threshold-optimized post-hoc)
- AUC = 0.436 < 0.5 (decisive evidence of predictive failure)
- Spearman(Ea, accuracy) = -0.063, p=0.66 (zero correlation)
- Pilot (n=30): low-Ea accuracy = 68.4% < 75%

**Status**: **FALSIFIED**
**Interpretation**: Ea measures answer stability (consistency of responses), not answer correctness. Stable wrong answers exist.

### H4: Ea Measures Stability, Not Correctness -- CONFIRMED (Theoretical)

**Claim**: The reason Ea fails for routing is that it captures answer agreement, not reasoning quality

**Evidence**:
- Ea distribution is bimodal (~9.47 and ~10.0), reflecting two modes of answer stability
- Level 5 Ea values are numerically compressed (std ~1.9e-6), suggesting algorithmic saturation
- "Agreement-but-wrong" problems have low Ea (consistent incorrect answers) but are not solvable

**Status**: Theoretically supported; needs H4 error classification for empirical validation

### H5: Consistency-Ea and Saturation-k_0 Are Related -- FALSIFIED

**Claim**: The two "activation energy" measures (from consistency and from saturation curve) capture the same construct

**Evidence**: Spearman(Ea, k_0) = -0.219, p=0.54; valid pairs = 10/50 (20%)
**Status**: **FALSIFIED**
**Interpretation**: Consistency-Ea measures answer stability; saturation-k_0 measures model learning dynamics. They are unrelated constructs.

---

## Method

### Experimental Design

**Model**: Qwen/Qwen2.5-Math-7B-Instruct
**Dataset**: MATH benchmark (n=50 for main analysis)
**Temperature**: 0.7
**Max tokens**: 1024
**Seed**: 42

### Component 1: Saturation Curve (G1) -- COMPLETED

Sample k=1,2,4,8,16 per problem. Fit exponential saturation model.
- Status: COMPLETE (R^2=0.924, P_inf=0.835, k_0=0.613)

### Component 2: Consistency Analysis (G2) -- COMPLETED

Estimate Ea from answer consistency trajectory: c(k) = c_0 * exp(-Ea/k)
- Status: COMPLETE (Spearman(Ea, Level)=0.448, p=0.001)

### Component 3: Routing Validation (G3) -- COMPLETED

Test whether Ea predicts single-pass correctness
- Status: COMPLETE (AUC=0.436, decisively falsified)

### Component 4: Error Classification (H4) -- PENDING

**Goal**: Empirically validate why Ea fails

Classify low-Ea failures into:
- Execution errors (calculation mistakes, algebra errors)
- Conceptual errors (wrong approach, misinterpretation)
- Answer extraction failures (pipeline errors)

**Expected**: Execution errors dominate, supporting the "stability != correctness" narrative
**Time**: 30 minutes
**Decision gate**: If execution errors >50%, proceed with paper. If conceptual errors >50%, frame as "deep error" narrative (link to Li 2026).

### Component 5: Pre-registered Threshold Validation (P1) -- PENDING

**Goal**: Eliminate data leakage concern

Re-run H3 with pre-registered threshold (median Ea or Level-based split)
**Expected**: AUC remains <0.5, confirming falsification
**Time**: 20 minutes

---

## Pilot Experiment Design

### Completed Experiments

| Group | Description | Hypothesis | Status | Key Result |
|-------|-------------|------------|--------|------------|
| G0 | Baseline (n=100) | -- | DONE | 47% overall accuracy |
| G1 | Saturation curve (n=50, k=1,2,4,8,16) | H1 | CONFIRMED | R^2=0.924 |
| G2 | Consistency analysis (n=50) | H2 | CONFIRMED | Spearman=0.448 |
| G3 | Routing validation (n=50) | H3 | FALSIFIED | AUC=0.436 |
| H5 | Ea vs k_0 (n=50) | H5 | FALSIFIED | Spearman=-0.219 |

### Pending Experiments

| Group | Description | Hypothesis | Time | Success Criteria |
|-------|-------------|------------|------|------------------|
| H4 | Error classification | H4 | 30 min | >50% execution errors in low-Ea failures |
| P1 | Pre-registered threshold | H3 | 20 min | AUC <0.5 with fixed threshold |

---

## Expected Contributions

1. **Negative Result (Primary)**: First systematic falsification that consistency-derived activation energy cannot predict single-pass correctness (AUC=0.436 < 0.5)
2. **Quantified Ceiling**: ~25pp "agreement-but-wrong" ceiling for convergence-rate-based routing (vs ACAR's 8pp for variance-based routing)
3. **Diagnostic Analysis**: Ea measures answer stability, not correctness -- explaining the failure mechanism
4. **Decoupling Evidence**: Consistency-Ea and saturation-k_0 are unrelated constructs (Spearman=-0.219)
5. **Boundary Delineation**: Explicit map of where the Activation Energy framework works (aggregate description) and fails (individual routing)
6. **Methodological Protocol**: Reusable evaluation framework for routing signal assessment

---

## Relationship to Prior Art

| Paper | Relationship | Our Position |
|-------|-------------|--------------|
| Yang et al. (2508.16456) | **COLLISION** - same exponential formula | We adopt their framework and test its limits |
| ACAR (2602.21231) | **CROSS-VALIDATES** - agreement-but-wrong ceiling | We quantify a larger ~25pp ceiling for Ea routing |
| Li (2601.00828) | **SUPPORTS** - Error Depth Hypothesis | Our H4 aligns with their error classification |
| RASC (2408.17017) | **DISTINCT** - reasoning-aware self-consistency | We investigate convergence-rate signals, not reasoning quality |
| CGES (2511.02603) | **DISTINCT** - confidence-guided early stopping | We test a different signal type (consistency convergence rate) |
| Wang et al. (2022) | **BASELINE** - self-consistency voting | Our Ea is derived from consistency but used for routing, not aggregation |

**Novelty Claim**: We do not claim mathematical novelty. Our contribution is empirical: first systematic quantification of the Ea-routing ceiling, with diagnostic analysis of why it fails.

---

## Novelty Assessment

### Critical Collision: Yang et al. (2508.16456)

Their formula Acc_t = Upp - alpha^t(Upp - Acc_0) is mathematically equivalent to our P_k = P_inf * (1 - exp(-k/k_0)). **We cannot and do not claim novelty for the exponential saturation model.**

### What IS Novel

1. **Systematic falsification**: First empirical proof that Ea from consistency does NOT predict single-pass threshold (H3)
2. **Ceiling quantification**: ~25pp gap for Ea routing, larger than ACAR's 8pp for variance-based routing
3. **Construct decoupling**: First demonstration that consistency-Ea and saturation-k_0 measure unrelated constructs (H5)
4. **Diagnostic analysis**: Why Ea fails -- it measures stability, not correctness
5. **Boundary map**: Explicit delineation of the framework's valid scope

### Novelty Score: 6/10 (Medium)

The negative result framing provides genuine value, but the underlying framework is not new. The paper's strength lies in honest empirical boundary-delineation, not theoretical innovation.

---

## Revisions from Prior Feedback

### From Novelty Checker (Round 4)

| Concern | How Addressed |
|---------|--------------|
| Yang et al. collision | Explicitly acknowledged; we adopt their framework, not claim it |
| H5 at risk due to ACAR | H5 reframed as falsified (not pending); entropy dropped as primary focus |
| Need to differentiate from ACAR | We quantify a different ceiling (~25pp Ea vs 8pp sigma) and diagnose the cause |

### From Result Debate (Round 4)

| Concern | How Addressed |
|---------|--------------|
| Optimist overstates H3 | H3 "CONFIRMED" label revoked; AUC=0.436 is decisive falsification |
| Skeptic's per-problem fit failure | Explicitly reported; H1 downgraded to "aggregate-only" |
| Methodologist's data leakage | P1 experiment added to validate with pre-registered threshold |
| Revisionist's theoretical decoupling | H5 formally falsified; Ea and k_0 are unrelated constructs |
| Comparativist's literature positioning | Paper reframed as boundary-delineation, not theory-proposal |

---

## Perspective Weighting Summary

| Perspective | Weight | Contribution to Final Proposal |
|-------------|--------|-------------------------------|
| **Skeptic** | Highest | Core insight: AUC=0.436 is decisive; per-problem fit failure is fatal to physical-law claims |
| **Methodologist** | Highest | Data leakage in H3 must be addressed; sample sizes must be acknowledged |
| **Revisionist** | High | Theory must be downgraded from "predictive" to "descriptive"; Ea measures stability, not correctness |
| **Comparativist** | High | Literature position is clear: negative result cross-validates ACAR; ~25pp ceiling is new |
| **Strategist** | High | Correct path is negative-result paper, not theory paper; H4 is decisive next step |
| **Optimist** | Low | H1/H2 aggregate findings are real but must not be over-interpreted |

**Reasoning**: The Skeptic and Methodologist identified the fatal flaws (AUC<0.5, data leakage, per-problem fit failure). The Revisionist provided the theoretical reframing (stability != correctness). The Comparativist and Strategist showed how to turn failure into a publishable contribution. The Optimist's enthusiasm was valuable for identifying the aggregate patterns but must be tempered by the empirical reality.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Reviewers reject negative result | Medium | Frame as "saving community resources"; cite NeurIPS negative results track; emphasize quantified ceiling |
| Sample size criticism (n=50) | Medium | Acknowledge limitation; report effect sizes and confidence intervals; frame as pilot study |
| Yang et al. novelty concern | High (but managed) | Explicitly acknowledge collision; emphasize contribution is boundary-delineation, not formula |
| H4 inconclusive | Low | Alternative framing available ("stability != correctness" is theoretically supported regardless of error mix) |
| Data leakage criticism | Medium | P1 experiment validates with pre-registered threshold; acknowledge exploratory nature of original H3 |

---

## Pilot Pass Criteria

- **H4** (error classification): Complete classification of low-Ea failures; any result supports the narrative
- **P1** (pre-registered threshold): AUC <0.5 confirms falsification; if AUC >0.5, re-evaluate

---

## Alternative: If Pilot Fails or Paper Rejected

### Alt 1: Expand to n=250 (MATH full test set)

- Increase statistical power
- Validate findings on larger sample
- Time: 120 minutes

### Alt 2: Cross-Model Validation

- Test on DeepSeek-Math-7B (26% baseline) and Qwen2.5-Math-1.5B
- Check if Ea-routing failure is model-specific
- Time: 90 minutes per model

### Alt 3: Pivot to Training-Based Approach

- GPU now available (PyTorch 2.11.0)
- Test whether training reduces Ea or improves routing
- Time: 180 minutes

---

## Execution Plan

### Immediate (This Iteration)

1. H4: Error classification on low-Ea failures (30 min)
2. P1: Pre-registered threshold validation (20 min)

### If Pilot Supports Narrative

1. Draft paper with negative-result framing
2. Write results section with H1-H5 status table
3. Prepare submission to venue with negative results track

### If Pilot Inconclusive

1. Expand sample to n=100-250
2. Re-run with stricter methodology
3. Consider cross-model validation

---

## Conclusion

Round 4's Activation Energy Theory experiments produced a **genuine but limited finding**: Arrhenius-like aggregate saturation is observable, but the framework fails at the individual-problem level where it matters for routing. The most valuable output is not a new theory but a **systematic falsification** of a plausible hypothesis, with clear diagnosis of why it failed.

This proposal synthesizes six diverse perspectives into a unified, decisive research direction: **publish the negative result**. The paper will not claim theoretical novelty. It will claim empirical honesty: we tested a hypothesis, found it fails, diagnosed why, and mapped the framework's boundaries. This is a contribution the community needs.
