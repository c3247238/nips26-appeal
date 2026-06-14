# Writing Critique — Iteration 7 (Updated)

**Critic**: sibyl-critic (Opus 4.6)
**Date**: 2026-03-19
**Paper version**: current/writing/paper.md (post-VGG completion, post-NoBN partial, post-matched-rho partial)

---

## Critical Data-Paper Inconsistencies

### W-CRIT-1: VGG-16-BN Table 3 — All Numbers Wrong

Every entry in Table 3 is incorrect. The paper systematically inflates means by 0.04-0.10 percentage points. Verified against epoch_metrics.jsonl:

| Method | Paper Reports | Actual Data (3-seed mean +/- std) | Delta |
|--------|--------------|-----------------------------------|-------|
| constant | 92.05 +/- 0.06 | 91.98 +/- 0.08 (91.94, 91.94, 92.07) | -0.07 |
| cosine | 91.99 +/- 0.32 | 91.89 +/- 0.30 (91.54, 92.03, 92.09) | -0.10 |
| CWD | 92.06 +/- 0.26 | 92.02 +/- 0.27 (92.31, 91.97, 91.77) | -0.04 |
| SWD | 92.11 +/- 0.28 | 92.02 +/- 0.24 (92.04, 91.78, 92.25) | -0.09 |
| half_lambda | 92.15 +/- 0.13 | 92.06 +/- 0.13 (91.92, 92.16, 92.11) | -0.09 |
| random_mask | 92.05 +/- 0.27 | 91.96 +/- 0.36 (91.55, 92.10, 92.23) | -0.09 |
| no_wd | 92.03 +/- 0.04 | 91.96 +/- 0.09 (91.97, 91.86, 92.04) | -0.07 |
| **Phi spread** | **0.16%** | **0.18%** | +0.02 |

Root cause: likely an earlier analysis run or a different "best accuracy" extraction metric (e.g., smoothed best vs. raw best). The qualitative conclusion survives (spread < 0.25%), but the systematic upward bias pattern is a reviewer red flag.

**Fix**: Regenerate Table 3 and all VGG-related text from actual epoch_metrics.jsonl. Audit the analysis pipeline to identify the source of the discrepancy.

### W-CRIT-2: NoBN Table 5 — Reports Seed Maxima, Not 3-Seed Means

Table 5 reports constant=87.74+/-0.20 and CWD=87.64+/-0.17. Actual 3-seed data:
- constant: seeds = [87.74, 87.45, 87.28], mean = **87.49 +/- 0.23**
- CWD: seeds = [87.64, 87.26, 87.49], mean = **87.46 +/- 0.19**

The paper uses the maximum seed value (87.74 = seed_42 for constant, 87.64 = seed_42 for CWD) as the "mean." The claimed constant-CWD gap shrinks from 0.10% to 0.03%.

**Fix**: Recompute using actual 3-seed means. Update Table 5 and all NoBN discussion.

---

## Major Writing Issues

### W-MAJ-1: Four Missing Appendices

The following appendices are cited but do not exist in the manuscript:
- Appendix B.1: Full proof of Theorem 1 (cited in Section 3.3)
- Appendix B.3: PMP-WD derivation from stochastic PMP (cited in Section 3.5, Theorem 3)
- Appendix B.4: RG beta function derivation (cited in Remark 3.1)
- Appendix A.3: CSI weight sensitivity analysis (cited in Section 3.2)

A theory-heavy paper with four missing proofs is not submittable. Appendices B.1 and B.3 are the most critical (they back the two main theorems).

### W-MAJ-2: Figure 1 Integrity — 2/3 Zones Unvalidated

Figure 1 (ratio regime diagram) shows three zones:
- Inhibition (rho < 0.1): partial data (rho_low constant 3 seeds + CWD 1-2 seeds)
- Transition (0.1 < rho < 2.0): complete data (main experiments)
- Differentiation (rho > 2.0): 5-epoch pilot only

The figure is presented as empirically grounded but 2/3 of the zones are extrapolated. Must be annotated with data coverage or relabeled as "hypothesized."

### W-MAJ-3: Cohen's d Mislabeling (Still Uncorrected)

Section 5.1 (previously 5.3) describes "paired formula d = delta_bar/s_delta" but all values match the unpaired pooled formula. This was flagged in the previous critique and remains uncorrected. Specific verification: SWD d=0.88 matches pooled(90.13, 89.88, 0.31, 0.25) but not paired computation.

### W-MAJ-4: Abstract Overclaims Regime Scaling

The abstract states "method sensitivity scales with rho" as confirmed. Evidence: two points (rho=0.005 SGD, rho=0.5 AdamW) differing in both optimizer AND rho. This is a confounded correlation. Should read "method sensitivity co-varies with rho and optimizer type; matched-rho experiments are ongoing."

### W-MAJ-5: PMP-WD Positioning

The paper lists PMP-WD as Contribution 3 but explicitly states "empirical evaluation is deferred to future work." Proposing an algorithm without any evaluation is unusual in ML venues. Either implement (30 LOC, ~3 GPU-hours) or demote to "future direction."

---

## Minor Writing Issues

### W-MIN-1: "3.7x" Headline Conflation

The 3.7x SGD/AdamW spread ratio conflates optimizer and rho differences. Currently qualified only in Section 6.5. Should be qualified at every occurrence (abstract, introduction, conclusion).

### W-MIN-2: Phi Spread Text vs Data

Paper says VGG-16-BN Phi spread = 0.16%; actual data = 0.18%. Small in isolation but part of the Table 3 inconsistency pattern.

### W-MIN-3: CWD phi Definition Ambiguity

CWD phi uses sign(theta) = sign(u_t) where u_t is the AdamW preconditioned update. The framework signature uses g_t (raw gradient). The evolution lessons correctly flag this: "CWD phi formula uses u_t (preconditioned update) but framework signature takes g_t (raw gradient)." Reconcile explicitly.

### W-MIN-4: Proposition 1 Formality

"CV(delta_hat) >> 1 for most training steps" lacks quantitative precision. Specify what "most" means (e.g., >90% of steps) or downgrade to "Observation 1."

### W-MIN-5: Method Naming Inconsistency

Multiple names for the same method appear across the paper: "cosine_schedule" / "cosine schedule" / "Cosine WD" / "cosine annealing"; "no_wd" / "no-WD" / "no weight decay". Standardize.

---

## What the Paper Does Well

1. **Limitations section (Section 6.5)**: Thorough, prioritized, and honest. Ordering by blocking priority is reviewer-friendly.
2. **Data gap table (Table 6, Section 5.8)**: Consolidating all incomplete configurations in one table is excellent practice. Few papers are this transparent.
3. **Diagnostic metric analysis (Section 5.7)**: Correctly identifying the architecture confound in pooled correlations and reporting within-architecture near-zero correlations is statistically sophisticated.
4. **Dual derivation narrative (Remark 3.1)**: Two independent mathematical routes converging on the same formula is compelling, even without proofs.
5. **Proposition 1 as design constraint**: Converting the Contrarian's noise critique into a positive contribution (EMA requirement) is elegant framing.
