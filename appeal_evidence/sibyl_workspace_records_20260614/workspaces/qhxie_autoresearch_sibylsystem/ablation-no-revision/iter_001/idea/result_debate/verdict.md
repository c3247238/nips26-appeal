# Verdict: Round 4 Activation Energy Theory Results

**Date**: 2026-04-29
**Quality Score**: 4/10
**Recommendation**: PROCEED (as negative-result paper, with caution)

---

## Executive Summary

Round 4's Activation Energy Theory experiments yield a **salvageable negative result**, not a successful theory. The aggregate Arrhenius pattern (R^2=0.924) is real but describes group-average behavior only — 80% of individual problems cannot be fit. The core routing hypothesis (H3) is decisively falsified (AUC=0.436). The most valuable contribution is **systematically proving that consistency-derived "activation energy" cannot predict single-pass correctness** and diagnosing why.

---

## Score & Justification

**4/10** — Weak empirical foundation (small samples, data leakage, per-problem fit failures, sample mismatch) but the negative-result framing is well-positioned in the literature and cross-validates ACAR's "agreement-but-wrong" ceiling.

---

## Key Conclusions

### What Survives Scrutiny
1. **Aggregate Arrhenius saturation** is a real statistical pattern (R^2=0.924), equivalent to Yang et al. (2025).
2. **Ea correlates with MATH Level** (Spearman=0.448, p=0.001), but effect is modest and Ea values are highly concentrated.
3. **Ea measures answer stability, not correctness** — the critical theoretical insight from this round.

### What Is Falsified
1. **H3 (Ea routing)**: AUC=0.436 < 0.5, Spearman(Ea, accuracy)=-0.063. Ea has zero predictive power for single-pass correctness.
2. **H5 (Ea == k_0)**: Spearman=-0.219. Consistency-Ea and saturation-k_0 are unrelated constructs.
3. **Arrhenius as physical law**: 80% per-problem fit failure (median R^2=0) proves it is not a universal mechanism.

---

## Literature Position

We occupy a defensible niche: **first systematic quantification of Ea-based routing ceiling** (~25pp, vs ACAR's 8pp for variance-based routing), **first Ea-MATH Level correlation**, and **first demonstration of Ea/k_0 decoupling**. We cannot claim mathematical novelty (Yang et al. collision) or a new method, but we can claim empirical boundary-delineation.

---

## Action Plan

### Immediate (Next 90 minutes)
1. **H4 Error Classification** (30 min): Classify 50 low-Ea failures. Go/No-Go gate for narrative direction.
2. **Fix H3 with pre-registered threshold** (20 min): Eliminate data leakage; confirm AUC < 0.5.
3. **Answer extraction audit** (20 min): Fix pipeline; ensure correctness labels are reliable.
4. **Draft paper outline** (30 min): Negative-result framing with "limits of Ea routing" narrative.

### Paper Framing
**Title**: *The Limits of Consistency-Based Activation Energy for Problem-Level Routing in Mathematical Reasoning*

**Core claim**: We systematically test whether consistency-derived activation energy can route LLM reasoning between single-pass and multi-sample paths. We find it cannot (AUC=0.436), diagnose why (Ea measures stability, not correctness), and delineate the theory's valid boundaries (aggregate description only).

### Decision Rule
- If H4 supports the "deep error" narrative (execution errors >50%): **Proceed to paper writing**.
- If H4 is inconclusive: **Pivot to broader routing signal comparison** (Plan B).
- Under no circumstances should we claim theoretical novelty or predictive power for Ea routing.

---

*Synthesis complete. The truth is: the theory failed, but the failure is informative.*
