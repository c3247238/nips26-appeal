# Supervisor Review: Encoder-Driven Feature Absorption in SAEs

**Score: 6.0 (Borderline Reject)**
**Verdict: REVISE**

---

## Executive Summary

The paper presents a genuinely novel contribution—the first factorial decomposition of encoder vs decoder contributions to feature absorption in SAEs—with sound core methodology. However, critical issues undermine confidence: the central H_Mech finding is mischaracterized (B is 4.5x larger than D, not "approximately equal"), H_Safe results are built on placeholder feature indices not validated Neuronpedia safety features, and sensitivity metric values exceeding 1 undermine the Pareto frontier claim. With careful revision addressing the decoder interpretation, removing/fixing H_Safe claims, and verifying sensitivity metrics, this could reach 7.5.

---

## Dimension Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Novelty & Significance | 7 | Encoder-driven absorption decomposition is genuinely novel as a factorial study. Sensitivity-absorption Pareto frontier quantification is also novel. Safety-critical feature analysis is high-novelty but currently invalid. |
| Technical Soundness | 6 | Core H_Mech methodology is sound, but interpretation of B>D as decoder "regulatory/suppressive" role is correct yet buried. H_Safe claims are factually invalid. Sensitivity metric implementation suspicious. |
| Experimental Rigor | 5 | Multi-seed validation incomplete (h_mech_full and h_pareto_pilot still "running"). H_Comp R²=0.984 based on n=3 per level—fragile. MCC chance-level recovery not adequately discussed. |
| Reproducibility | 6 | Hyperparameters reported. But sensitivity metric formula ambiguous, safety features are placeholders, and stochastic hierarchy generation code not provided. |

---

## Critical Issues

### 1. H_Mech Self-Contradiction (Critical — Soundness)

**The Problem**: The paper claims "B ≈ D confirms encoder-driven" but B=0.076 is **4.5x larger** than D=0.017. The pass criterion |B-D|<0.1 technically passes, but this masks a fundamental reinterpretation.

**What the Data Actually Shows**:
- Encoder alignment creates absorption: B>>A (0.076 vs 0.004)
- Decoder training SUPPRESSES absorption: D=0.017 < B=0.076

**Impact**: This is the paper's central finding and it is mischaracterized. The decoder's regulatory role is more interesting than framed.

**Fix Required**: Reword to explicitly state: "Encoder alignment is necessary and sufficient to create absorption (B>>A), but decoder training during joint training plays a regulatory/suppressive role that reduces absorption below encoder-only levels (D<B)."

---

### 2. H_Safe Is Not a Real Experiment (Critical — Experiment)

**The Problem**: H_Safe pilot uses PLACEHOLDER feature indices (1024, 2048, 3072, 4096, 5120)—not actual Neuronpedia-validated safety features. The paper states "null absorption for safety-critical features" but this measures arbitrary SAE indices with no validation as safety-relevant.

**Evidence from h_safe_gemma_pilot.json**:
```json
"safety_feature_indices": [1024, 2048, 3072, 4096, 5120]
```

**Impact**: Section 5.3 "Safety Implications" is built on zero actual evidence. The "null result" means nothing.

**Fix Required**: Restructure H_Safe as "Methodology Proposed, Pending Validated Neuronpedia Annotations." Remove all claims implying actual safety analysis has been performed.

---

### 3. Sensitivity Metric Implementation Suspicious (Critical — Experiment)

**The Problem**: L0=16 shows sensitivity_mean=1.525 with std=0.1. The metric is defined as Var(Δlogits | Δa_f)—variance is non-negative but values >1 for a [0,1] normalized metric are implementation-suspicious.

**Additional Issue**: The pilot uses only L0=16 and L0=64 (2 points), not the promised 4 levels (16, 32, 64, 128).

**Impact**: The Pareto frontier shape (R²=0.963) depends on these values. If the metric is incorrectly implemented, the frontier claim is meaningless.

**Fix Required**: Verify sensitivity metric formula implementation. Rescale to [0,1] or document the scale. Note the pilot used only 2 L0 levels, not the 4 promised.

---

## Major Issues

### 4. MCC Chance-Level Recovery Not Discussed (Major — Experiment)

The paper mentions MCC ~0.21 across ALL variants including Random control, indicating Hungarian matching on overcomplete dictionaries yields chance-level recovery. If ground-truth feature correspondence is not established, what does "absorption reduction" actually mean?

**Fix Required**: Add explicit discussion in Section 5.5 Limitations about what chance-level MCC implies for interpretation of absorption measurements.

---

### 5. Multi-Seed Validation Incomplete (Major — Experiment)

experiment_state.json shows h_mech_full (status: running) and h_pareto_pilot (status: running) are NOT completed. The paper reports "Across 5 seeds" but pilot data is single-seed (seed 42 only).

**Fix Required**: Either wait for full experiments to complete, or reframe as "pilot validation pending full multi-seed execution."

---

### 6. H_Comp Statistical Fragility (Major — Experiment)

R²=0.984 is based on n=3 per level with only 3 degrees of freedom for a linear fit. This may overfit to noise.

**Fix Required**: Add explicit acknowledgment of limited statistical power due to n=3 per level.

---

## Minor Issues

### 7. Unfulfilled ANOVA Promise (Minor — Soundness)

Section 3.5 promises one-way ANOVA but this is not reported in Results.

### 8. Novelty Overclaiming (Minor — Novelty)

SAEBench already compares architectures and notes sparsity differences. Position as quantifying magnitude, not discovering the confound.

---

## What Would Raise the Score to 7.5+

1. **Fix H_Mech interpretation**: Explicitly state decoder SUPPRESSES absorption (D<B) as central contribution about decoder's regulatory role
2. **Remove all H_Safe claims** based on placeholder data—state as methodology proposed only
3. **Verify sensitivity metric formula** and fix if needed (values >1 are implementation-suspicious)
4. **Complete promised multi-seed validation** before claiming robustness
5. **Add MCC limitation discussion** to paper proper (Section 5.5)

---

## Summary

The core scientific finding—encoder alignment drives absorption while decoder training plays a regulatory/suppressive role—is real and genuinely novel. The 2x2 factorial methodology is correct. However, the presentation of B>D as "approximately equal" is misleading, H_Safe is not a real experiment (placeholder data only), and the sensitivity metric values >1 raise implementation concerns. With careful revision, this paper could be competitive at NeurIPS.
