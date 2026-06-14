# Verdict: Executive Summary

> Result Debate Synthesis — One-page executive summary
> Date: 2026-04-29

---

## Overall Score: 6.5 / 10

A solid negative result with one strong positive finding. Theoretically grounded, empirically supported, but limited in scope.

---

## Key Conclusion

**Co-occurrence clustering (UAD) cannot detect hierarchical feature absorption in pre-trained SAEs.** This is not a bug or parameter issue — it is a fundamental mismatch. Absorption features are token-level mutually exclusive (they fire on different child-concept tokens and never co-occur), while UAD searches for features that fire together.

**The collision rate proxy metric is validated** (Spearman r = 0.869, n=56, CI=[0.780, 0.938]) and can serve as a fast screening tool for candidate absorption pairs.

---

## Evidence at a Glance

| Experiment | Result | Interpretation |
|------------|--------|----------------|
| P1: Proxy validation (10 pairs) | r = 0.711 | Collision rate correlates with absorption |
| F4: Extended proxy (56 pairs) | r = 0.869 | Strong correlation across 2 hierarchy types |
| P2: UAD reproduction | F1 = 0.00048 | UAD fails to detect absorption |
| P3: Random baseline | UAD F1 = random F1 | UAD provides zero value over random |
| F2: Ablations | Best variant F1 = 0.0037 | All UAD variants fail |
| F5: False positive analysis | 99.98% FP rate | UAD produces 4154 false positives |

---

## What the 6 Perspectives Agree On

1. UAD fails — this is certain and theoretically explained
2. Token-level mutual exclusivity is the root cause — this is an intrinsic property, not a dataset artifact
3. Collision rate proxy works — validated across multiple hierarchy types
4. This is publishable as a negative result — all perspectives concur

## Where They Disagreed (Resolved)

- **Quality**: Optimist (8/10) vs Skeptic (5/10) → **Resolved at 6.5/10**. Core finding is robust; limitations are real but non-fatal.
- **Timing**: Strategist (write now) vs Skeptic (more validation) → **Resolved: write now with explicit limitations.** The theoretical finding is complete; delaying risks losing time-value.
- **Proxy strength**: Optimist ("proven") vs Skeptic ("may be artifact") → **Resolved: "reliably correlated but not causally proven."** GT definition concerns are valid and must be disclosed.

---

## Action Plan

### Immediate (This Week)
**Write the negative result paper.** Target: ICBINB 2026 or NeurIPS/ICML Interpretability Workshop.

**Title**: "Why Co-occurrence Clustering Cannot Detect Feature Absorption in Sparse Autoencoders"

**Core contributions**:
1. Empirical falsification of UAD for absorption detection (F1 = 0.00048)
2. Validation of collision rate as absorption proxy (r = 0.869)
3. Theoretical explanation: token-level mutual exclusivity
4. Constructive proposals: decoder weight similarity, causal intervention

**Must include**: Explicit limitations section (sample size, single model, GT definition concerns).

### Parallel (Next 1-2 Weeks)
**Pilot alternative method**: Decoder weight cosine similarity on 100 pairs.
- Decision gate: F1 > 0.5 → expand; else, document and try next alternative.

### Post-Submission
- Expand GT to 100+ pairs
- Cross-model validation
- Causal validation of best alternative

---

## Bottom Line

**PROCEED with the negative result paper.** The finding is honest, theoretically grounded, and valuable to the community. Do not let perfect be the enemy of good — the core insight (token-level mutual exclusivity makes co-occurrence clustering unsuitable) is complete and does not require more experiments. Write it, submit it, and keep exploring alternatives in parallel.
