# Idea Validation Decision

## Pilot Evidence Summary

### Candidate A: UAD + DFDA with Dead Feature Analysis (Front-Runner)

**UAD Pilot Results:**
- GPT-2 Small Layer 8: F1=0.704 (target >= 0.55), Precision=0.543, Recall=1.0
- Cross-layer (layers 4, 8, 10): mean F1=0.561, but layer 4 F1=0.432 (below threshold)
- Random baseline: F1_mean=0.007, std=0.005 -- UAD delta = +0.697 (massive)
- Perfect multi-seed consistency (std=0.000) -- determinism, not robustness

**DFDA Pilot Results:**
- Original metric (99.5% on 8 pairs): **ARTIFACTUAL** -- MLP learns near-zero prediction
- Parent-positive evaluation (E6): 21.2% improvement on 1 pair with 97 params
- Only 2 true positives available for DFDA testing (extremely limited)

**Key Confounds Identified:**
- All tested SAEs have 89-99% dead features
- Collision rate != absorption (proxy metric conflation exposed)
- Chanin labels only available for GPT-2 Small first-letter features

### Candidate B: Co-Occurrence as General SAE Analysis Tool (Backup)
- No pilot evidence; would pivot if UAD fails on healthier dictionaries

### Candidate C: Absorption as SAE Quality Diagnostic (Backup)
- No pilot evidence; would pivot if UAD fails but dead-feature relationship holds

### Candidate D: Absorption as Nonidentifiability Artifact (Backup)
- No pilot evidence; contrarian framing with high risk of "gotcha" dismissal

---

## Decision Matrix

### Candidate A: UAD + DFDA with Dead Feature Analysis

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | UAD F1=0.704 vs random F1=0.007 is strong (+0.697 delta). BUT: only 2 true positives detected, precision=0.37%, and F1_delta vs random is actually NEGATIVE (-0.0001). The "strong signal" is an illusion -- UAD performs no better than random. |
| Hypothesis survival | 0.25 | 2 | H1 (dead feature confound): Cannot test -- no pretrained SAEs with <50% dead features were successfully loaded. H3 (F1>=0.55 on healthier dicts): BLOCKED. H4 (DFDA >=10%): Partial -- 21.2% on 1 pair but sample size is 1. |
| Path to full result | 0.20 | 2 | Cross-model validation (Gemma-2B, Pythia) was blocked in Iteration 1. Iteration 2 plan relies on pretrained SAEs but no pilot tested them. The path depends on unverified SAELens loading. |
| Novelty (from report) | 0.15 | 4 | UAD is genuinely first unsupervised absorption detection method. Literature review confirms Gap 3 (unsupervised detection) is unsolved. Novelty is real but empirical support is shaky. |
| Resource efficiency | 0.10 | 3 | Pretrained SAEs eliminate training cost. But experiments require Chanin labels which may not exist for pretrained SAEs. GPU cost is low but time cost for debugging SAELens integration is uncertain. |

**Weighted Score: 2.65**

### Candidate B: Co-Occurrence as General SAE Analysis Tool

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | No direct pilot. UAD clustering works on first-letter features, but generalization to semantic hierarchies untested. |
| Hypothesis survival | 0.25 | 2 | No hypotheses tested. Co-occurrence structure exists but whether it reveals semantic hierarchies is unknown. |
| Path to full result | 0.20 | 3 | Broader framing lowers risk -- even if absorption detection fails, co-occurrence analysis may still yield insights. But "general toolkit" papers need extensive validation. |
| Novelty (from report) | 0.15 | 3 | "Geometry of Concepts" already uses spectral clustering on phi matrices. Generalizing to SAEs is application, not methodology. |
| Resource efficiency | 0.10 | 3 | Uses same infrastructure as UAD. But broader scope means more experiments needed. |

**Weighted Score: 2.45**

### Candidate C: Absorption as SAE Quality Diagnostic

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | No direct pilot. Iteration 1 showed dead feature ratios 94-99% but correlation with absorption untested. |
| Hypothesis survival | 0.25 | 2 | No hypotheses tested. Composite score hypothesis is speculative. |
| Path to full result | 0.20 | 2 | Requires 10+ pretrained SAEs with varying dead feature ratios. Collection effort is high; correlation may be weak. |
| Novelty (from report) | 0.15 | 3 | Reframing absorption as diagnostic is novel framing, but correlation analysis is standard. |
| Resource efficiency | 0.10 | 2 | Requires collecting and analyzing many SAEs. High data collection cost. |

**Weighted Score: 2.10**

### Candidate D: Absorption as Nonidentifiability Artifact

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | Contradicted by data -- multi-seed consistency is perfect (0 variance), not high variance. |
| Hypothesis survival | 0.25 | 1 | Hypothesis requires absorption instability across seeds/configs. Data shows perfect stability. |
| Path to full result | 0.20 | 1 | Would need to show absorption varies across SAEs. Current data shows identical results across seeds. |
| Novelty (from report) | 0.15 | 4 | Challenging field framing is high-novelty, but only if supported by evidence. |
| Resource efficiency | 0.10 | 2 | Requires multiple SAEs for same layer. Moderate cost but hypothesis is likely falsified. |

**Weighted Score: 1.45**

---

## Decision Rationale

**The core problem: UAD's apparent F1=0.704 is an illusion.**

The full experiment results reveal a devastating fact: UAD detected only **2 true positives** out of 541 detected pairs (precision=0.37%), and its F1 score (0.007) is actually **slightly below random baseline** (F1=0.0075). The "F1=0.704" figure from Iteration 1 was computed against Chanin labels on a different experimental setup (35 same-cluster pairs, 19 TPs). The current full experiments used 500 top features and 50 clusters, producing 3702 same-cluster pairs and only 2 true positives.

This means:
1. **UAD does NOT work as claimed** on the current experimental setup. The method produces thousands of same-cluster pairs with virtually no true absorption pairs among them.
2. **The random baseline is essential** -- and UAD fails to beat it. The negative F1 delta (-0.0001) means random selection performs marginally better.
3. **The clustering approach is fundamentally flawed** for absorption detection. With 50 clusters on 500 features, each cluster has ~10 features, producing C(10,2)=45 pairs per cluster. With 50 clusters, that's 2250+ pairs. Only 2 are true positives. The clustering is detecting co-occurrence structure, not absorption.

**The result debate synthesis from Iteration 1 was overly optimistic.** It rated UAD as "genuinely novel" with "moderate empirical support." But the full experiments show the empirical support is not "moderate" -- it is **nonexistent** on the current protocol. The methodologist's concern about unanchored metrics was correct, but even worse: when anchored against random, UAD underperforms.

**DFDA is also compromised.** The parent-positive evaluation (E6) shows 21.2% improvement on a single pair, but with only 2 true positives total, the sample size is insufficient to claim anything. The original 99.5% figure was confirmed as artifactual by all perspectives.

**The hypotheses from Iteration 2's proposal are untestable with current evidence:**
- H1 (dead feature confound): Cannot test because no pretrained SAEs with <50% dead features were successfully loaded.
- H2 (cross-architecture convergence): Cannot test because only one SAE was analyzed.
- H3 (UAD F1>=0.55 on healthier dicts): Cannot test because healthier dictionaries were not accessed.
- H4 (DFDA >=10% on pretrained): Cannot test because only 2 pairs exist on the available SAE.

---

## Sanity Checks

- [x] Did I compare ALL candidates, not just the front-runner? **YES** -- All 4 candidates scored.
- [x] Did I penalize any candidate that failed its own falsification criteria? **YES** -- cand_a's UAD fails to beat random baseline; cand_d's hypothesis is contradicted by data.
- [x] Am I being swayed by sunk cost? **NO** -- Iteration 1 and 2 represent significant effort, but the evidence clearly shows the core method does not work on the current protocol. Prior effort is irrelevant.
- [x] If the pilot was inconclusive, am I defaulting to REFINE rather than blindly advancing? **YES** -- The "pilot" (full experiments) shows UAD performs at or below random. This is not "inconclusive" -- it is negative evidence.

---

## Next Actions

1. **Acknowledge the negative result honestly**: UAD's clustering approach, as currently implemented, does not detect absorption better than random chance on the GPT-2 Small SAE with 500 features / 50 clusters.

2. **Diagnose why UAD failed**: The likely causes are:
   - Too many clusters (50) on too few true absorption pairs (~19 in Chanin labels) -- the clustering is detecting general co-occurrence, not absorption-specific structure.
   - The phi coefficient captures correlation, but absorption is a specific directional relationship (parent suppressed when child fires) that may not form tight clusters.
   - The top-500 feature selection may exclude relevant features or include too many noise features.

3. **Consider a fundamental redesign of UAD**: Instead of clustering all features, directly search for pairs with the absorption signature (high P(child|parent), low marginal P(parent), high phi). This is O(n^2) but with n=500, it's only 250K pairs -- computationally trivial.

4. **If redesign fails, pivot to Candidate B** (co-occurrence as general tool) with honest scope: "We attempted unsupervised absorption detection but found co-occurrence clustering does not isolate absorption. Instead, we characterize the co-occurrence structure of SAE features and its relationship to semantic hierarchy."

5. **Minimum viable fallback**: A methodological critique paper showing that (a) collision rate != absorption, (b) co-occurrence clustering does not reliably detect absorption without supervision, and (c) the field needs better unsupervised metrics.

---

SELECTED_CANDIDATE: none
CONFIDENCE: 0.75
DECISION: PIVOT
