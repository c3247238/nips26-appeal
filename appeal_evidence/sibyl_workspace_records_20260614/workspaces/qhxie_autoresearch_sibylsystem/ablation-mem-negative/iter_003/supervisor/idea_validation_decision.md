# Idea Validation Decision

## Pilot Evidence Summary

Only one candidate was tested in this pilot round: **UAD with co-occurrence clustering** (`uad_cooccurrence`).

### P1: Collision Rate — Absorption Rate Proxy Validation (PASS)
- Spearman r = 0.711, Pearson r = 0.733
- Bootstrap 95% CI = [0.219, 0.887] (excludes zero)
- All 10 vowel pairs detected (10/10)
- **Verdict**: Collision rate is a valid proxy for absorption rate. This is a genuine positive finding.

### P2: UAD Reproducibility Validation (FAIL)
- Precision = 0.0%, Recall = 14.3%, F1 = 0.0
- True Positives = 1, False Positives = 4154, Ground Truth = 7 pairs
- **Verdict**: UAD completely fails to detect hierarchical absorption. Recall below 80% threshold by a massive margin.

### P3: Random Baseline Validation (FAIL)
- UAD F1 = 0.00048
- Same-cluster random F1 = 0.00048 (IDENTICAL)
- UAD minus global random = 0.00037 (required >= 0.3)
- **Verdict**: UAD is statistically indistinguishable from random sampling within clusters. The clustering step provides zero value.

### Root Cause
Absorption features are **mutually exclusive at the token level** — they fire on different tokens representing different child concepts. UAD's co-occurrence clustering detects features that fire TOGETHER, not features that fire on mutually exclusive instances of a parent concept. This is a **fundamental mismatch**, not a tunable parameter issue.

---

## Decision Matrix

### Candidate: `uad_cooccurrence`

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | F1=0.0, 4154 false positives, 1 true positive out of 7 ground truth pairs. UAD F1 identical to same-cluster random baseline (0.00048). |
| Hypothesis survival | 0.25 | 1 | All three hypotheses falsified: H1 (UAD detects absorption via co-occurrence) — FAIL; H2 (absorption features co-occur) — FAIL (phi ~ 0); H3 (UAD outperforms random) — FAIL (UAD == same-cluster random). |
| Path to full result | 0.20 | 1 | No credible path. The co-occurrence mechanism is fundamentally incompatible with token-level mutual exclusivity of absorption features. No parameter tuning can fix this. |
| Novelty | 0.15 | 2 | The negative result itself has novelty value: demonstrating why co-occurrence clustering cannot detect absorption is a useful contribution to the SAE community. |
| Resource efficiency | 0.10 | 1 | Any additional GPU budget on this candidate is wasted. The method has been shown to have zero discriminative power. |

**Weighted Score = 1.15**

---

## Decision Rationale

The decision is **PIVOT**. The evidence is overwhelming and conclusive:

1. **UAD F1 = Same-Cluster Random F1** (0.00048). This is the smoking gun. The entire UAD pipeline — co-occurrence matrix, hierarchical clustering, phi filtering, dead feature filtering — provides exactly zero value over randomly sampling pairs from within the same clusters. This is not a "needs more tuning" situation; it is a fundamental architectural mismatch.

2. **Token-level mutual exclusivity** is the root cause. Absorption features fire on DIFFERENT tokens (e.g., feature 11513 only on "three", feature 24189 only on "four"-"eight"). They never co-occur. UAD clusters features that co-occur. The method literally cannot detect the phenomenon it targets.

3. **P1's positive result is separable**. The collision proxy validation (r=0.71) is a genuine finding, but it does not rescue UAD. It merely validates that collision rate correlates with absorption rate — a useful observation, but not a detection method.

4. **The proposal.md is now obsolete**. It was written before pilot execution and assumed UAD would work (citing iteration 1's F1=0.704 on a different, likely flawed setup). The pilot has falsified that assumption.

5. **No backup candidates exist** in `candidates.json` (file not present). A fresh ideation round is needed.

---

## Next Actions

1. **Do NOT proceed with any full experiments** (E1-E6 from task_plan.json). All are predicated on UAD working, which it does not.

2. **Report the negative result honestly**. The pilot has produced a valuable finding: co-occurrence clustering cannot detect hierarchical feature absorption in SAEs. This should be documented as a negative result.

3. **Preserve P1's positive finding**. The collision rate proxy validation (r=0.71) is a genuine empirical contribution and should be retained.

4. **Pivot paper focus** to: "Why Co-occurrence Clustering Cannot Detect Feature Absorption: A Negative Result and Conceptual Analysis"

5. **Start fresh ideation** for a new candidate approach that does NOT rely on co-occurrence. Potential directions (from pilot recommendations):
   - Semantic similarity between feature decoder weights
   - Causal intervention (zeroing child features, measuring parent recovery)
   - Activation patching between related concepts
   - Direct analysis of feature geometry in activation space

6. **Consider whether this project should continue** or if the negative result should be written up and the project concluded. The spec's goal was to detect absorption; if the main approach is fundamentally flawed, the project may need a broader pivot.

SELECTED_CANDIDATE: none
CONFIDENCE: 0.95
DECISION: PIVOT
