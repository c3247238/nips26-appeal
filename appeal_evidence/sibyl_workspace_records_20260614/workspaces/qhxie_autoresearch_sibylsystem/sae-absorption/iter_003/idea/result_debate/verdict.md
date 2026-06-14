# Result Debate Verdict: SAE Feature Absorption (iter_003)

**Date:** 2026-04-14  
**Decision:** PROCEED

---

## Overall Score: 6.5 / 10

Solid empirical package with one genuinely novel mechanistic finding (H2 falsification) and a practical detection improvement (encoder_norm > EDA). Small positive-class sample size and hook confound prevent a higher score, but do not block publication.

---

## Key Conclusion

**Feature absorption in GPT-2 SAEs is primarily a training-time dictionary coverage problem (sparsity landscape / partial minima), not an encoder approximation failure (amortization gap).** This conclusion is supported by: (1) OMP oracle achieving 0% false-negative reduction across 18 absorbed features (C2), and (2) encoder_norm elevation in absorbed features (Cohen's d=0.97-1.23) consistent with gradient competition during training. Encoder_norm provides the best weight-only detection proxy (AUROC=0.757-0.837 across architectures, DeLong p=0.0012 vs. EDA).

---

## Top 3 Strengths

1. **H2 falsification is clean and novel.** OMP oracle at mean feedforward sparsity achieves identical absorption rate (0.978) as feedforward encoder across all tested letters. First empirical evidence adjudicating O'Neill vs. Tang. Directly actionable: practitioners should improve SAE training objectives, not SAE encoder inference.

2. **Encoder_norm detection replicates across three independent contexts.** Standard/L1 SAE with IG labels (n=18, AUROC=0.757), Standard/L1 at L10 with proxy labels (n=39, AUROC=0.645), TopK-32k with proxy labels (n=77, AUROC=0.837). The signal is consistent despite different label sources and layer positions.

3. **O_jaccard provides independent, complementary detection signal.** AUROC=0.721, AUPRC=0.075, near-zero correlation with encoder_norm (Spearman ρ=0.044). Two uncorrelated signals both detect absorption: supports structural reality of the phenomenon.

---

## Top 3 Risks

1. **n_pos=18 at gold-label L6.** Bootstrap CI width ≈ ±0.09 AUROC. Three mislabeled features could shift the point estimate meaningfully. Mitigated by multi-layer/multi-architecture replication, but must include power analysis in paper.

2. **Hook confound in A3.** Standard SAE uses resid_pre, TopK uses resid_post. Cannot cleanly attribute AUROC difference to architecture. Fix: report encoder_norm works on both (unconfounded), do not compare AUROC values directly across architectures.

3. **H3 entity-type result is a methodology failure.** Qwen→GPT-2 cross-model probe transfer via random QR decomposition was invalid. Entity-type absorption remains an open question, not a negative result.

---

## Action Plan (Prioritized)

| Priority | Action | Timeline |
|---|---|---|
| 1 | Begin writing — lead with H2 falsification, then encoder_norm detection | Immediately |
| 2 | Hook-confound correction: compute resid_pre vs. resid_post cosine similarity at L6, or load matched SAE | 1-2 hours |
| 3 | Power analysis: bootstrap minimum detectable effect at n=18 | 30 minutes |
| 4 | Frame H3/D2 as "future work" with one paragraph; do not attempt to rerun | Writing phase |
| 5 | Target venue: ICML 2026 MI Workshop (now) + ICLR 2026 (after hook correction) | Staggered |

---

## Recommended Paper Framing

**Primary claim:** SAE feature absorption is a training-time dictionary coverage problem, detectable by encoder weight geometry, not resolvable by improving feedforward encoder inference.

**Narrative structure:**
1. Background: Absorption degrades SAE interpretability
2. Encoder norm: weight-only detector, outperforms EDA, cross-architecture (with hook caveat)
3. Co-occurrence: independent structural signal (O_jaccard)
4. **H2 oracle test:** amortization gap is not the cause — sparsity landscape is
5. F1: wider SAEs partially help (67%), but training-time solutions are needed
6. Discussion: implications for SAE training objectives

**Title option:** *"Feature Absorption in Sparse Autoencoders is Primarily a Sparsity Landscape Problem, Not an Amortization Gap"*
