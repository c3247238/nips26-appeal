# Planning Critique: SAE Feature Absorption Methodology

## Overall Assessment

The methodology document is comprehensive and well-organized, but several planned experiments were not executed as specified, and the experimental design contains structural flaws that the planning phase should have caught.

---

## Critical Issues

### 1. Planned 5 Architectures, Executed 2

**Plan** (methodology.md Section 3.1): "JumpReLU, TopK, BatchTopK, Gated, Matryoshka"

**Actual**: Only JumpReLU and TopK were evaluated. BatchTopK, Gated, and Matryoshka are mentioned in Related Work but never tested.

**Why this matters**: The paper's title and framing promise a "cross-architecture benchmark" but only compare 2 architectures, one of which is pretrained and the other trained from scratch. This is not a benchmark; it is a single comparison with multiple confounds.

**Fix**: Either (a) train the remaining 3 architectures, or (b) reframe as "A Comparison of Pretrained JumpReLU and Trained TopK SAEs" rather than a "cross-architecture benchmark."

---

### 2. Gemma-2-2B Primary Model Was Not Used

**Plan** (methodology.md Section 2.1): "Primary Model: Gemma-2-2B (via TransformerLens)... Layers evaluated: 5, 10, 15, 20"

**Actual**: All experiments use GPT-2 Small (12 layers). Gemma-2-2B was abandoned due to "API issues."

**Why this matters**: The methodology promises evaluation on a modern 2B-parameter model but delivers results on a 124M-parameter model from 2019. This significantly limits the relevance and generalizability of findings.

**Fix**: Make the GPT-2 Small focus explicit in the title and abstract. Do not mention Gemma-2-2B in the abstract.

---

### 3. Planned 100 Concepts, Executed 26

**Plan** (methodology.md Section 3.2): "Define 100 concepts across 5 domains (animals, colors, numbers, countries, emotions)"

**Actual**: Only 26 first-letter concepts (a-z) were used.

**Why this matters**: The generalization from 26 first-letter features to "downstream interpretability tasks" is tenuous. First letters are syntactic, not semantic, concepts. The planned 100-concept evaluation would have provided much stronger evidence.

**Fix**: Add at least one additional concept domain, or scope claims to "first-letter feature detection."

---

### 4. Steering Efficacy Was Planned But Not Executed

**Plan** (methodology.md Section 3.2): "Model Steering: Select 10 steering directions (5 sentiment, 5 topic). Measure steering efficacy as logit difference shift."

**Actual**: No steering experiments were conducted. The paper notes this as a future work item.

**Why this matters**: Steering is a key downstream interpretability task. Without it, the "causal impact on downstream tasks" claim rests solely on sparse probing of first-letter features---a very narrow basis.

**Fix**: Remove steering from the methodology if not executed, or execute the planned experiments.

---

### 5. Causal Design Was Not Actually Causal

**Plan** (methodology.md Section 3.2): "Causal Design: Fix model and input data; vary only SAE architecture/configuration; control for reconstruction quality and sparsity via partial correlation."

**Actual**: The E2 "causal" experiment varies k, which directly controls both sparsity AND reconstruction quality. There is no independent variation of absorption.

**Why this matters**: The paper claims a "causal assessment" as a contribution, but the design does not support causal inference. This is misleading.

**Fix**: Reframe as "correlational assessment" or redesign to independently manipulate absorption (e.g., via DFDA compensation) while holding reconstruction constant.

---

### 6. Time Budgets Were Unrealistic

**Plan** (task_plan.json):
- F1 (CAAB): 45 minutes for 5 architectures on Gemma-2-2B
- F2 (Causal): 50 minutes for 100 concepts + steering
- F4 (Layer): 30 minutes for all 18 layers of Gemma-2-2B

**Actual**: The full experiment suite took ~15 minutes total, suggesting the planned experiments were severely scaled down.

**Why this matters**: The planning phase overcommitted to experiments that could not be completed in the time budget. The resulting paper is a scaled-down version that does not match the planned methodology.

**Fix**: The planning phase should have been more realistic about time constraints and designed experiments that fit within the budget.

---

## Strengths

1. **Clear experimental protocol** with step-by-step procedures
2. **Well-defined metrics** with mathematical definitions
3. **Risk mitigation table** anticipates problems (though some mitigations failed)
4. **Reproducibility section** with fixed seeds and package versions
5. **Expected visualizations** map clearly to experiments

---

## Deviation Summary

| Planned | Actual | Impact |
|---------|--------|--------|
| 5 architectures | 2 architectures | "Benchmark" claim weakened |
| Gemma-2-2B primary | GPT-2 Small only | Generalizability limited |
| 100 concepts, 5 domains | 26 first letters | Downstream claims weakened |
| Steering efficacy | Not executed | Causal claim weakened |
| 6-8 hours full experiments | ~15 minutes total | Severely scaled down |
| Chanin et al. absorption detection | Collision rate proxy | Metric validity weakened |

---

## Recommendation

The planning phase should have:
1. Been more realistic about time constraints
2. Required matched conditions for architecture comparison
3. Scoped claims to what could actually be measured
4. Included a pilot validation of the collision-absorption proxy relationship
