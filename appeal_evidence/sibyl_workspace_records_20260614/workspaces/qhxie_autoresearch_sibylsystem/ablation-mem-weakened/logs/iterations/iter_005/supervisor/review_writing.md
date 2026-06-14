# Supervisor Review: Competitive Suppression in SAEs

**Iteration:** 5
**Score:** 5.5 / 10 (Borderline Reject)
**Verdict:** Continue
**Date:** 2026-04-30

---

## Executive Summary

The paper pivoted to a new LCA/competitive suppression framing with H6 correctly executed and reported as falsified. The mechanistic framework (H7, H9) is empirically supported by honest null-result reporting. However, **the proposal and paper are incoherent documents** with fundamentally different titles, primary claims, and hypotheses — they read as different papers. Additionally, H10 is claimed as Contribution #4 despite being explicitly unexecuted. These critical issues persist from prior iterations and undermine the paper's credibility.

**Bottom line:** The empirical execution is mostly sound and honest. The narrative is fractured. Fix the coherence problem first.

---

## Dimension Scores

| Dimension | Score | Assessment |
|-----------|-------|-----------|
| Novelty & Significance | 7 | LCA-SAE connection is genuinely novel. The correspondence is exact by definition for tied-weight SAEs — its scientific value depends entirely on whether it yields validated predictions. |
| Technical Soundness | 6 | Structural correspondence proof is correct. H6 correctly falsified. H10 unexecuted but claimed as contribution. H1b reporting is self-contradictory across documents. |
| Experimental Rigor | 5 | H6 executed and reported honestly. Probe ceiling effect at k>=5 undermines precision invariance claim. Steering metric too coarse. Random baseline confounded. n=26 underpowered for correlations. |
| Reproducibility | 5 | No supplementary materials despite references. No per-feature raw data files. Single model (GPT-2 Small) with failed Pythia replication not discussed. |

---

## Critical Issues

### 1. Proposal-Paper Incoherence (Novelty, Critical)

**proposal.md** is titled "Feature Absorption as Optimal Compression: Evidence that Training Reduces Structural Artifacts" with H7-H8 framing where the primary claim is trained SAE < random SAE absorption. **writing/paper.md** is titled "Competitive Suppression in SAEs: Connecting LCA to Absorption" with H6-H10 framing where the primary claim is the LCA mechanism. These are different papers with different primary claims, hypotheses, and evidence. A reader of proposal.md would not recognize the same project in paper.md.

**Impact:** This suggests the project has not resolved its core narrative. Reviewers will notice the disconnect and may question whether the authors themselves know what the paper is about.

### 2. H10 Claimed as Contribution Without Execution (Experiment, Critical)

Section 1.4 Contribution #4 states "First training-free post-hoc repair for absorption." Section 4.7 explicitly states "The homeostatic rebalancing experiment was not executed due to the negative H6 result." The contribution list presents H10 as a done thing. The abstract says "we propose" but Section 1.4 says "we contribute" — inconsistent framing that overstates what was actually done.

**Impact:** Reviewers may desk-reject this paper for overclaiming. "We contribute" means "we did this," not "we propose this."

### 3. Abstract Contradicts Itself (Writing, Critical)

The abstract states "a local inhibition graph constructed from decoder correlations predicts absorption pairs" as if H6 validated. The very next sentence says "is falsified: precision@20 = 0.0 with no enrichment over chance (p = 1.0)." This contradiction makes the abstract confusing and signals carelessness.

**Impact:** A confused abstract signals a confused paper. Reviewers may stop reading.

---

## Major Issues

### 4. H1b Contradictory Reporting Persists (Soundness, Major)

proposal.md says "zero hypotheses survive multiple comparison correction (12 tests)" but paper.md Section 4.2 and the abstract present delta-corrected L8 (r=-0.431, p=0.028 uncorrected) as evidence of a real effect. The proposal simultaneously claims "null hypothesis supported" and presents the same test as evidence against the null. This was flagged in iterations 0-2 and persists.

### 5. Probe Ceiling Effect Undermines Precision Invariance Claim (Experiment, Major)

At k>=5, 25 of 26 features achieve precision = 1.0. This is a ceiling effect: the k-sparse probe has saturated capacity for 26 features. It can perfectly distinguish positive from negative examples when it fires. This trivially produces precision = 1.0 — it does not demonstrate that absorption preserves selectivity in any meaningful sense. The precision invariance claim is methodology-bound.

**Evidence from raw data:** At k=1 (before saturation), precision std = 0.195 at L8, precision_min = 0.0 at L8. The paper never discusses k=1 results prominently, burying them in Table 4.

### 6. Steering Success Criterion Too Coarse (Experiment, Major)

The criterion (top-5 token contains target letter) measures directional correctness only, not steering magnitude. Steering could reduce target letter probability by 80% while keeping it in top-5, masking absorption effects. A finer-grained metric (probability mass on target, or rank position) would detect subtler effects.

### 7. Random SAE Baseline Confounded (Experiment, Major)

The random SAE uses a frozen orthonormal decoder, which is architecturally different from the trained SAE's learned decoder. The 8x absorption difference (trained mean=0.034 vs random mean=0.278) could be due to the orthonormal constraint, not "training reduces artifacts." The confound was identified in prior iterations and has not been addressed.

### 8. No Supplementary Materials Despite References (Reproducibility, Major)

The paper references supplementary materials for raw graph data and detailed statistics. Per-feature absorption rates, steering correlations, and probe F1 scores needed to reproduce Table 3, Table 4, and hypothesis tests are not accessible in the workspace.

---

## Cross-Validation with Raw Data

I verified the paper's reported numbers against raw data files:

- **Table 3 (graph statistics):** Mean edge weight 0.384 at L8, 0.312 at L0 — consistent with descriptive text. Cross-layer trend is real.
- **Table 4 (precision-recall):** precision_std=0.016 at k=20, recall_std=0.167 at L8 — correctly reported. However, k=1 data shows precision_std=0.195 at L8, with one feature at precision=0.0 — this is hidden by focusing on k>=5.
- **H6 result:** precision@20=0.0, p=1.0 — correctly reported from raw pilot data.
- **H1b delta-corrected L8:** r=-0.431, p=0.028 (uncorrected) — correctly reported but not clearly flagged as not surviving correction.

---

## What Would Raise the Score

1. **Resolve proposal-paper incoherence** (most critical): Reconcile the two documents into one coherent narrative with matching title, abstract, and contributions.
2. **Remove H10 from contributions** or execute it before claiming it.
3. **Add probe ceiling analysis**: Show precision at k=1, compute dictionary activation fractions, explicitly acknowledge saturation.
4. **Add finer-grained steering metrics**: Probability mass or rank position, not just top-5 hit rate.
5. **Save all per-feature data to accessible JSON files** in exp/results/.
6. **Resolve H1b contradictory reporting**: Be consistent across all documents.
7. **Discuss Pythia null result** (r=-0.041, p=0.841) in the paper body as negative evidence about generalizability.

These changes would address most critical and major issues. Estimated score after fixes: **6.5-7.0**.

---

## Risks

- Proposal-paper incoherence suggests narrative has not been resolved — reviewers will notice
- H10 overclaiming invites desk rejection
- LCA structural correspondence is exact by definition for tied-weight SAEs — without H6 validation, reviewers may call it rebranding
- Probe ceiling undermines the central evidence for competitive suppression mechanism
- Abstract contradiction signals carelessness
