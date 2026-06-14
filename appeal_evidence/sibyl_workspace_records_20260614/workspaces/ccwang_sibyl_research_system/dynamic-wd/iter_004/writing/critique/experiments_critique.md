# Critique: Experiments (Sections 5 & 6) — Revision Round 4 (Blocker Resolution)

**Critic**: Sibyl Section Critic (sibyl-light)
**Section**: 5. Experimental Setup + 6. Results
**Date**: 2026-03-18
**Revision Round**: 4 (focused verification of three critical blockers from final review)
**Prior Rounds**: Round 3 (5 critical, 5 major, 5 minor; 0 resolved)
**Data verified against**: `exp/results/analysis/sgd_baseline_analysis.json`, per-seed `summary.json` files in `iter_003/exp/results/sgd_baseline/`

---

## Executive Summary

Round 4 focuses exclusively on the three blockers raised in the final review. All three have been verified against raw data. The findings are:

1. **18.3× ratio**: Confirmed arithmetic error. 10.29/0.16 = 64.3, not 18.3. The 18.3× figure is arithmetically consistent only with AdamW d = 0.56, but Table 2 reports d = 0.16. These cannot both be correct. The paper is internally inconsistent and must choose one value and propagate it consistently everywhere.

2. **swd p_adj = 0.054**: Confirmed incorrect. The raw p = 0.000399 corrected under any standard Holm application gives p_adj = 0.0036 (significant). The value 0.054 is arithmetically unreachable from raw p = 0.000399 via Holm or Bonferroni regardless of family size. SGD swd IS a second significant comparison; the paper's "statistical honesty statement" is factually wrong about the count of significant results.

3. **φ notation**: Minor inconsistency confirmed. The formal definition uses φ(t, θ, s_t) but §3.4 BEM definition (lines 154, 157) uses φ(t, θ, g_t). This is a localized inconsistency in one definition, not pervasive throughout the paper.

**Score remains: 5/10.** The two confirmed factual errors (ratio arithmetic, swd significance) are publication blockers.

---

## Blocker 1: The 18.3× Ratio Arithmetic Error — Confirmed

### Verification Method
Raw data verification using `sgd_baseline_analysis.json` plus arithmetic computation.

### Confirmed Facts
- SGD constant vs. no_wd: Cohen's d = 10.2867 (JSON field `effect_sizes`, cifar10/no_wd). This matches the paper's reported d = 10.29.
- AdamW CIFAR-10: constant mean = 90.1333 ± 0.3083; no_wd mean = 90.0833 ± 0.3166 (JSON `table_data`).
- Pooled-std Cohen's d for AdamW: delta = 0.05%, pooled_std = 0.3125%, d = 0.05/0.3125 = **0.1600**. This matches Table 2's reported value of 0.16.
- Arithmetic: 10.29 / 0.16 = **64.3**, not 18.3.
- The 18.3× figure requires AdamW d = 10.29/18.3 = **0.562**.

### Root Cause Diagnosis
The paper contains two incompatible values that have not been reconciled:

| Location | AdamW d value | Source |
|----------|--------------|--------|
| Table 2, no_wd row (line 293) | 0.16 | Pooled-std formula: Δ/σ_pooled |
| §6.2 Key Finding 3 (line 333) | 0.16 | Same as above |
| §6.2 ratio claim (line 333) | 0.562 (implied) | Required to produce 18.3× |
| Bootstrap BCa CI in abstract | requires d=0.562 | CI [12.1×, 28.7×] was computed with d=0.56 |
| `iter_003/writing/sections/experiments.md` | 0.56 (draft) | Pre-integration paired d value |

The integration step preserved the ratio (18.3×) and CI from an earlier draft that used d = 0.56, while separately computing and inserting the pooled-std d = 0.16 into Table 2 and the §6.2 narrative. These two values use different formulas:
- **Pooled-std d** (Table 2): d = Δ / √[(σ₁² + σ₂²)/2] = 0.05/0.3125 = 0.16 (verifiable from JSON)
- **Paired d** (needed for 18.3×): d = Δ̄ / s_Δ, where s_Δ = std of per-seed differences

Table 2's caption says "Cohen's d (paired)" — so the Table 2 value of 0.16 is claimed to be the paired formula. For paired d = 0.16, the std of pairwise differences would need to be 0.05/0.16 = 0.3125%, identical to the pooled std (meaning zero correlation between conditions). For paired d = 0.56, the std of differences would be 0.05/0.56 = 0.089%, implying an implied correlation of r ≈ 0.918 between same-seed runs.

**Per-seed AdamW data is not available** in the workspace (no per-seed AdamW `summary.json` files exist — only the aggregated JSON). The correct paired d cannot be verified without that data.

### Required Fix (Choose One Option)

**Option A — Use pooled-std d = 0.16 (verifiable, conservative):**
- Keep Table 2 d = 0.16 for no_wd.
- Update §6.2 Key Finding 3: change "ratio ≈ 18.3×" to "ratio ≈ 64×."
- Recompute Bootstrap BCa 95% CI for the 64× ratio (requires code, cannot be hand-waived).
- Update all six occurrences of "18.3×" in the paper (Abstract, §1.4, §4.3, §6.2, §7.1, §8.1, §8.2) to "~64×."
- Update CI "[12.1×, 28.7×]" everywhere with the newly computed CI.
- This is the most conservative and most verifiable choice.

**Option B — Use paired d = 0.56 (matches current ratio, requires per-seed data):**
- Re-run the per-seed AdamW comparison to compute paired d directly.
- If paired d = 0.56 is confirmed: update Table 2 no_wd from 0.16 to 0.56.
- Ensure all other Table 2 d-values also use the paired formula (currently unverified).
- Add footnote clarifying why d differs from the pooled-std intuition (high within-pair correlation).
- Keep 18.3× ratio and current CI.

**Minimum acceptable fix (if not rerunning experiments):**
- In §6.2, change the quoted value from "AdamW d = 0.16" to "AdamW d = 0.56" to make the ratio arithmetic self-consistent.
- Add footnote: "This paired d = 0.56 differs from the pooled-std estimate of 0.16 in Table 2 (column 'Cohen's d (paired)'), which is inconsistent with the paired formula; we plan to recompute Table 2 d-values using paired differences in the revision."
- This is not a complete fix but removes the arithmetic contradiction.

**Under no circumstance** should the paper state "AdamW d = 0.16; ratio ≈ 18.3×" simultaneously, as 10.29/0.16 = 64.3, not 18.3.

---

## Blocker 2: SGD swd p_adj — Confirmed Error

### Verification Method
Direct computation from the `corrections` array in `sgd_baseline_analysis.json`, cross-checked with Holm algorithm applied from first principles.

### Confirmed Facts
From `sgd_baseline_analysis.json`, field `corrections` (SGD comparisons, sorted by raw p):

| Comparison | Raw p | Holm adj_p (JSON) | Significant? |
|---|---|---|---|
| cifar10/no_wd | 0.0 | 0.0 | YES |
| cifar100/swd | 2.45e-7 | 2.45e-6 | YES |
| cifar10/swd | **3.99e-4** | **3.59e-3** | **YES** |
| cifar10/half_lambda | 9.20e-3 | 7.36e-2 | no |
| cifar100/half_lambda | 9.37e-3 | 6.56e-2 | no |
| ... | ... | ... | ... |

The JSON corrections array uses k = 11 comparisons (all non-constant SGD comparisons across both datasets). Under this family:
- cifar10/swd is rank 3 of 11.
- Holm multiplier = (11 - 2) = 9.
- adj_p = 3.99e-4 × 9 = **0.003593** (significant at α = 0.05).

If the paper uses k = 6 (CIFAR-10 only):
- cifar10/swd is rank 2 of 6.
- Holm multiplier = (6 - 1) = 5.
- adj_p = 3.99e-4 × 5 = **0.001996** (significant at α = 0.05).

**The paper-reported value of p_adj = 0.054 is arithmetically unreachable.** For Holm correction, the multiplier needed to reach 0.054 from raw p = 3.99e-4 would be 135.3 — impossible for any k ≤ 135. For Bonferroni, k = 135 experiments would be needed. This is not rounding error; it is a factor of ~15 discrepancy.

The paper's "statistical honesty statement" (§6.2) claims: "only one pairwise comparison achieves significance: constant vs. no_wd." This is factually wrong. **Two SGD comparisons are significant on CIFAR-10**: no_wd (p_adj ≈ 0) and swd (p_adj = 0.0036). Three are significant when CIFAR-100 is included: cifar10/no_wd, cifar100/swd, and cifar10/swd.

### Impact on Paper's Narrative
The "only one comparison significant" claim is central to the paper's framing of SGD's weight decay sensitivity: the paper argues that swd just barely misses significance (p = 0.054), and uses this to say "weight decay presence matters for SGD but dynamic strategy choice is unconfirmed." If swd is actually significant (p = 0.0036), this moderates the argument: both weight decay presence AND one specific dynamic strategy (swd) significantly differ from constant under SGD on CIFAR-10.

This requires updating:
1. The "statistical honesty statement" paragraph in §6.2.
2. Table 3's p_adj column for swd: change 0.054 to 0.004 (or 0.002 if k=6).
3. Point 2 of §6.2 Key Findings: revise "Weight decay strategy choice is not confirmed under SGD" — it IS confirmed for swd (p = 0.0036, d = 3.48).
4. The abstract's description of the central finding may need adjustment.
5. §7.1 practical recommendations: add that swd is significantly different from constant under SGD.

### Required Fix
- Re-state the Holm correction family explicitly: is it k=6 (CIFAR-10 only) or k=11 (both datasets)?
- Report the correct adj_p: 0.004 (k=11) or 0.002 (k=6), not 0.054.
- Update Table 3 swd row accordingly.
- Revise the "statistical honesty statement" to say two (or three) comparisons reach significance.
- Update §6.2 Key Finding 2 from "strategy choice unconfirmed" to acknowledge swd significance.

---

## Blocker 3: φ Notation Inconsistency — Confirmed, Localized

### Verification Method
Full-text search of `paper.md` for all occurrences of `\varphi(`.

### Findings

The formal definition (§3.1, line 115) correctly uses φ(t, θ_t, s_t) with the explanation that s_t may include raw gradients g_t, preconditioned updates u_t, or other statistics. The abstract (line 5), §1.4 (line 43), Table 1 (line 132), and §3.1 axioms (line 121) all correctly use φ(t, θ, s_t) or φ(t, θ, s).

**The inconsistency is localized to §3.3 BEM definition (lines 154 and 157)**, which uses φ(t, θ_t, g_t) instead of φ(t, θ_t, s_t):
- Line 154: BEM budget equivalence formula uses `\varphi_1(t, \boldsymbol{\theta}_t, \mathbf{g}_t)`
- Line 157: effective weight decay definition uses `\varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t)`

This is **not the abstract/§1.4 notation issue described in the final review** — the abstract already uses s_t. The review's description of "abstract uses g, §3.1 uses s_t" is incorrect: both the abstract and §3.1 formal definition use s_t. The actual inconsistency is only in the BEM section.

**Severity**: Minor but incorrect. The BEM definition uses g_t (raw gradient) where it should use s_t (optimizer state), since BEM applies to all methods including those that depend on preconditioned updates rather than raw gradients (e.g., CWD depends on the sign of preconditioned update u_t, not raw g_t).

### Required Fix
In §3.3 (lines 154 and 157), replace `\mathbf{g}_t` with `\mathbf{s}_t` in both occurrences. This makes BEM consistent with the formal framework definition that s_t encompasses g_t as a special case.

---

## Issue Priority Summary After Round 4

### Confirmed New Blockers (from this round's verification)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| B1 | 18.3× ratio arithmetic: 10.29/0.16 = 64.3, not 18.3 | Critical | **Confirmed error** |
| B2 | SGD swd p_adj = 0.054 is wrong; correct is 0.0036 (significant) | Critical | **Confirmed error** |
| B3 | φ notation: g_t vs s_t in BEM definition only | Minor | **Confirmed, localized** |

### Carried Forward Critical Issues (from Round 3)

| # | Issue | Status |
|---|-------|--------|
| C1 | All 5 figures absent | Unresolved |
| C2 | SGD swd and cifar100/swd significance missed (consistent with B2) | Confirmed by B2 |
| C3 | CIFAR-100 SGD no_wd n=1 (missing seed) not disclosed | Unresolved |
| C4 | BN confound in §6.4 mechanistic claim | Unresolved |
| C5 | 18.3× ratio conflates optimizer mechanism with ρ operating-point | Partially addressed (limitation in §7.4), but must be disclosed at the finding point |

### Minimum Fix Path to Publication-Ready Draft

The paper can reach a submittable state with these text-only fixes (no new experiments):

1. **Fix the ratio arithmetic (B1)**: Either update "18.3×" to "~64×" everywhere with corrected CI, or update Table 2 d=0.16 to d=0.56 with a footnote explaining the formula difference.
2. **Fix swd p_adj (B2)**: Change Table 3 swd p_adj from 0.054 to 0.004; update "statistical honesty statement" to report two/three significant comparisons; revise Key Finding 2 accordingly.
3. **Fix φ notation (B3)**: Replace g_t with s_t in BEM definition (2 lines).
4. **Soften mechanistic claim (C4)**: Replace "confirms" with "is consistent with" in §6.4.
5. **Disclose ρ operating-point at point of finding (C5)**: Add one sentence in §6.2 18.3× paragraph.
6. **Disclose n=1 for CIFAR-100 no_wd (C3)**: Add one sentence in §6.2 or a table footnote.

Figures (C1) remain the single largest gap between current state and a strong NeurIPS submission, but they do not block a workshop or short-paper submission.

---

## What Works Well (Unchanged from Round 3)

1. **Statistical design is exemplary**: Holm correction, Cohen's d, TOST with explicit MDE, bootstrap BCa — all structurally correct. The errors are in the parameter values, not the methodology.
2. **BEM diagnostic narrative (§6.4)**: The observation that BEM spans [0, -1] yet produces <0.3% accuracy spread under AdamW is concrete, falsifiable, and directly operationalizes the Phi Invariance Conjecture.
3. **Control methods (random_mask, half_lambda)**: Including these controls directly tests budget effects vs. strategy effects. Methodologically rigorous.
4. **VGG pilot framing (§6.3)**: Correctly presented as infrastructure check, not scientific confirmation.

---

*Critique prepared by: Sibyl Section Critic (sibyl-light)*
*Revision round: 4 (blocker resolution, fresh arithmetic verification)*
*Raw data verified: sgd_baseline_analysis.json, per-seed summary.json in iter_003/*
*Date: 2026-03-18*
