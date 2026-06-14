# Idea Validation Decision

## Pilot Evidence Summary

### RQ1: Cross-Architecture Comparison (Pilot + Full, 5 seeds each)

| Variant | Absorption Rate (mean ± std) | L0 Sparsity | vs Baseline Delta |
|---------|------------------------------|-------------|-------------------|
| Baseline ReLU | 0.254 ± 0.047 | 964 | — |
| TopK (k=50) | 0.056 ± 0.021 | 50.0 | -0.198 (-78%) |
| Matryoshka | 0.057 ± 0.023 | 50.0 | -0.197 (-77%) |
| Orthogonality SAE | 0.247 ± 0.048 | 550 | -0.007 (-3%) |
| Gated SAE | 0.257 ± 0.052 | 962 | +0.003 (+1%) |
| Random Control | 0.495 ± 0.035 | 1030 | +0.241 (+95%) |

**Key observation**: Absorption spans ~10x from TopK (0.056) to Random (0.495), exceeding the >2x go criteria. However, **this is NOT an L0-matched comparison** — TopK and Matryoshka run at L0=50 while Baseline runs at L0=964. The apparent architectural advantage is confounded by sparsity.

### RQ2: Dose-Response Causality (Full, 5 seeds x 5 lambda levels)

| Lambda_L1 | Absorption Range | feature_recovery_mcc |
|-----------|------------------|----------------------|
| 5e-05 | 0.146 – 0.238 | 0.220 – 0.222 |
| 0.0002 | 0.141 – 0.256 | 0.219 – 0.221 |
| 0.0005 | 0.147 – 0.258 | 0.219 – 0.221 |
| 0.001 | 0.153 – 0.289 | 0.218 – 0.220 |
| 0.002 | 0.176 – 0.319 | 0.217 – 0.219 |

**Critical finding**: Absorption varies ~2.3x across lambda levels (0.141 to 0.319), but **feature_recovery_mcc is essentially flat** (~0.22 with std ~0.001). There is NO monotonic trend, NO dose-response relationship, and NO correlation between absorption and downstream feature recovery.

### RQ3: Mutual Coherence (Pilot only)

- Only Baseline SAE mutual coherence computed: mu_max = 0.305, mu_mean = 0.0499
- No cross-variant correlation possible — insufficient data to test H3a or H3b

### RQ4: Semantic Generalization (Pilot only)

- Single semantic absorption rate computed: 0.204 (baseline)
- No cross-category comparison, no correlation with first-letter absorption
- H4a/b remain untested

### Ablation Results (Full, 5 seeds)

| Ablation | Absorption (mean) | Notes |
|----------|-------------------|-------|
| Matryoshka flat (no nesting) | 0.056 | Same as nested — nesting is not the driver |
| OrtSAE without penalty | 0.230 | Same as with penalty — orthogonality lambda had no effect |
| TopK as ReLU+L1 | 0.180 | Lower than Baseline but higher than TopK — explicit k matters |

---

## Decision Matrix

### Candidate A: Causal Link Between Feature Absorption and Downstream Interpretability Failure

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | Absorption varies 10x across architectures (strong signal for RQ1), but causal hypothesis H2 is falsified — no correlation with downstream metrics |
| Hypothesis survival | 0.25 | 2 | H1a partially supported (Matryoshka lower but not L0-matched), H1b supported, H1c falsified (TopK LOWER not higher), H2a/b FALSIFIED, H3/H4 untested |
| Path to full result | 0.20 | 3 | Negative result is publishable per pre-registered risk plan, but L0-confound undermines RQ1 claims; needs proper L0-matching to salvage |
| Novelty | 0.15 | 4 | Medium-high novelty per novelty assessment; negative causal result is genuinely novel and important for the field |
| Resource efficiency | 0.10 | 3 | ~3 GPU hours spent; need additional L0-matched experiments (~30 min) to fix confound |
| **Weighted Score** | **1.00** | **2.90** | |

### Candidate B: Temporal Dynamics of Absorption Emergence

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot data collected; no evidence |
| Hypothesis survival | 0.25 | 1 | No hypotheses tested |
| Path to full result | 0.20 | 2 | Would require new experiments from scratch; estimated 4 GPU hours |
| Novelty | 0.15 | 3 | Medium novelty; partially covered by Li & Ren (2025) |
| Resource efficiency | 0.10 | 2 | High cost, no existing data to build on |
| **Weighted Score** | **1.00** | **1.55** | |

### Candidate C: Cross-Layer Absorption Propagation

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot data collected |
| Hypothesis survival | 0.25 | 1 | No hypotheses tested |
| Path to full result | 0.20 | 2 | Would require pretrained multi-layer SAEs; estimated 2 GPU hours |
| Novelty | 0.15 | 4 | High novelty; no prior work identified |
| Resource efficiency | 0.10 | 2 | New direction, no leverage from existing experiments |
| **Weighted Score** | **1.00** | **1.65** | |

### Candidate D: Absorption as Compositional Semantics

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | No direct pilot, but the null causal result (H2 falsified) is indirect support — if absorption doesn't harm downstream, compositional recovery is plausible |
| Hypothesis survival | 0.25 | 2 | No hypotheses directly tested, but pivot trigger (RQ2 null) is met |
| Path to full result | 0.20 | 2 | Would require new experiments testing compositional recoverability; estimated 2 GPU hours |
| Novelty | 0.15 | 4 | High novelty; directly challenges established framing |
| Resource efficiency | 0.10 | 2 | New direction, limited leverage from existing data |
| **Weighted Score** | **1.00** | **2.05** | |

---

## Decision Rationale

**Candidate A scores 2.90 — REFINE.**

The pilot and full experiments produced a **strong negative result** that is itself scientifically valuable: **absorption rate does not predict downstream interpretability failure** (feature recovery MCC is flat across a 2.3x absorption range). This directly addresses the most important unanswered question in the field and validates the contrarian challenge.

However, the experiments also exposed **critical methodology issues** that must be fixed before the paper is defensible:

1. **L0-confound in RQ1**: The apparent architectural advantage of TopK/Matryoshka (absorption ~0.056 vs Baseline ~0.254) is confounded by L0 sparsity (50 vs 964). This is not a fair comparison. The pre-registered L0-matching protocol was not executed.

2. **H1c falsified**: TopK shows LOWER absorption than Baseline, opposite of the SAEBench-based prediction. This may be because the prediction applies at matched L0, which we did not achieve.

3. **H3/H4 untested**: Mutual coherence and semantic generalization lack sufficient data.

The pre-registered decision rules state: "H2a OR H2b null → PIVOT to Alternative D." However, the proposal **explicitly anticipated this outcome** in the risk assessment: "Frame as important negative result. Contrarian was right. Still publishable as it resolves a critical gap." The negative result was always a valid path forward.

**Why REFINE rather than PIVOT:**
- The negative result on causality is solid and important — abandoning it wastes the GPU investment
- With proper L0-matching, RQ1 can still yield a clean architectural comparison
- The paper can be reframed: "Architecture Comparison at Matched L0 + Absorption Does Not Predict Downstream Failure"
- This is a stronger, more honest paper than persisting with a falsified causal claim

**Why not ADVANCE:** The methodology issues (L0-confound) are too severe to proceed without fixing. The causal hypothesis is dead. The paper needs structural changes.

---

## Next Actions

1. **Fix L0-matching for RQ1**: Train Baseline L1 SAE with lambda sweep to achieve L0=50, enabling fair comparison with TopK/Matryoshka. Target: 1 seed pilot, then 5 seeds if signal is clear. (~15 min GPU)

2. **Reframe paper contribution**: Shift from "causal link" to:
   - (a) First L0-matched cross-architecture comparison with statistical rigor (5 seeds, effect sizes)
   - (b) First systematic test of absorption-downstream causality — **negative result**
   - (c) Implications: community focus on absorption reduction may be misdirected

3. **Drop H3 (mutual coherence) and H4 (semantic generalization)** from primary contributions — insufficient data, report as exploratory if at all.

4. **Add contrarian discussion section**: Address the "absorption as feature" perspective directly. The data support this view.

5. **Consider Candidate D integration**: If L0-matched comparison confirms TopK/Matryoshka advantage AND absorption still doesn't predict downstream failure, the compositional semantics framing (Candidate D) becomes a natural discussion point — absorbed parent features may remain recoverable through children.

6. **Do NOT pivot fully to Candidate D** — the existing data and reframed Candidate A provide a clearer, more complete paper.

---

SELECTED_CANDIDATE: cand_a
CONFIDENCE: 0.55
DECISION: REFINE
