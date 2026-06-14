# Supervisor Review: Feature Absorption in Sparse Autoencoders

## Overall Assessment

**Score: 6.0 / 10 — Revise**

The paper introduces a legitimate, reproducible absorption metric (Af) with honestly reported negative results. Significant improvement from prior iteration: H1 now correctly distinguishes layer 8 (falsified at 0.19%) from layer 4 (not falsified at 49.3%), H4 is correctly labeled "uninformative," and Section 5.5 explicitly acknowledges layer 4 was the correct H2 target but was never attempted.

However, three critical/major issues from the prior review persist and prevent the score from advancing:

1. **H4 causal conclusion remains** in Abstract, Section 6.3, and Conclusion despite prior review identifying it as critical
2. **8 perfect-score latents** (Af=1.0, each firing on exactly 100 tokens) remain as "open question" instead of being investigated — h5_pilot_output.log confirms this at ALL dictionary sizes
3. **Only pilot-scale (n=100)** experiments reported despite methodology stating "The full experiment uses 1,024 sequences"

The contribution is valid but incomplete. The paper needs another iteration to address the critical issues before it can be evaluated as a publication-quality submission.

---

## Dimension Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | 6 | The absorption metric is a legitimate contribution, but the paper does not position it relative to existing superposition measures (Elhage et al. 2022, Sharkey et al. 2023, Templeton 2025). H2 (the most novel hypothesis about token frequency correlation) was never tested. The contribution is primarily negative results (absorption is rare), which is scientifically valid but limits novelty ceiling. |
| **Soundness** | 5.5 | The metric definition is clear and validated against random controls. However, the H4 causal conclusion is unsupported by the experiment design — comparing full SAE (100%) vs 10% subsets tests reconstruction capacity, not absorption's causal role. The paper itself acknowledges the correctly designed experiment was never conducted. |
| **Experiments** | 5.5 | Pilot-scale only (n=100), no seed ablation, H2 not tested despite layer 4 offering 260x more absorbed latents. The 8 perfect-score latents (each firing on exactly 100 tokens) are confirmed at all dictionary sizes but remain uninvestigated. |
| **Reproducibility** | 6.0 | Metric is well-defined and validated. Code availability not mentioned. No seed ablation means layer-dependence finding cannot be trusted without confirmation. |

---

## Critical Issues

### 1. H4 Causal Conclusion Is Unsupported (Critical)

**Issue**: The Abstract states "dictionary completeness — not absorption level — drives patching fidelity" and Section 6.3 repeats this. This is a causal conclusion drawn from an experiment comparing full SAE (100% of latents) against 10% subsets (which destroy ALL signal, not just absorption-related signal).

**Why this is critical**: The comparison tests whether reconstruction capacity matters — not whether absorption level predicts circuit importance. Both absorption subsets yield 0.0 faithfulness, making any comparison impossible. The paper itself acknowledges in Section 5.5 that "layer 4 (49.3% absorption) was the correct target but was never attempted."

**Required fix**: Remove all variants of the causal conclusion from Abstract, Section 5.3, Section 6.3, and Conclusion. Report H4 as strictly inconclusive: "Both absorption subsets yield 0.0 faithfulness, preventing any conclusion about whether absorption level predicts circuit-level causal importance. The correctly designed experiment (comparing full SAE at layers with different absorption profiles) was not conducted. H4 is inconclusive, not falsified."

### 2. 8 Perfect-Score Latents Remain Uninvestigated (Critical)

**Issue**: Section 6.5 lists the 8 perfect-score latents (Af=1.0, each firing on exactly 100 tokens) as an "open question." h5_pilot_output.log confirms this pattern occurs at ALL dictionary sizes (2K, 8K, 24K), with each top-5 absorbed latent firing on exactly 100 tokens at every dictionary size (lines 77-96 of h5_pilot_output.log).

**Evidence from h5_pilot_output.log**:
```
Top 5 most absorbed latents (dict=2048):
  1. Subsampled idx 665: score=1.0000, activations=100
  2. Subsampled idx 1495: score=1.0000, activations=100
  ...
Top 5 most absorbed latents (dict=8192):
  1. Subsampled idx 2517: score=1.0000, activations=100
  ...
Top 5 most absorbed latents (dict=24576):
  1. Subsampled idx 6955: score=1.0000, activations=100
  ...
```

**Why this is critical**: 100 = number of sequences in the pilot corpus. If these latents fired on semantically meaningful tokens, activation counts would vary across sequences. The regularity strongly suggests positional artifact (position-embedding alignment), not semantic absorption. If confirmed as artifacts, the metric may be detecting corpus structure rather than genuine feature absorption.

**Required fix**: Investigate immediately:
1. Compute token position consistency across sequences for each perfect-score latent
2. Check whether encoder outputs exceed threshold only at that consistent position
3. If confirmed as artifacts, exclude from primary analysis and document in Appendix

### 3. Pilot-Scale Only Despite Methodology Claim (Major)

**Issue**: Section 4.2 states "The full experiment uses 1,024 sequences" but all results use only 100 sequences (12,800 tokens). The 10x scale gap is material for rare phenomenon characterization.

**Required fix**: Either run full-scale experiments (n=1,024) with pre-committed analysis plan, or explicitly scope the paper as a pilot study with commitment to full-scale replication before publication.

### 4. No Seed Ablation (Major)

**Issue**: All experiments use seed=42 with no confirmation across seeds. The dramatic layer discrepancy (layer 4 at 49.3% vs layer 8 at 0.19%) is too large to trust without seed confirmation.

**Required fix**: Add seed ablation (seeds 42, 43, 44) for key findings. If not feasible before next milestone, explicitly acknowledge as limitation.

---

## Minor Issue

### 5. H5 Framing Overstates Practical Significance (Minor)

"10-fold reduction" (2K=2.25% vs 24K=0.19%) is directionally correct but practically negligible: 97.75% non-absorbed at 2K vs 99.81% at 24K. Lead with absolute numbers in Abstract and Section 5.4.

---

## Evidence Cross-Validation

| Source File | Finding | Paper Value | Verified |
|------------|---------|-------------|----------|
| h1_pilot.json | 46/24,576 = 0.19% at layer 8 | 0.19% | Confirmed |
| h3_pilot.json | Layer 4 = 49.3% | 49.3% | Confirmed |
| h4_pilot_output.log | raw=0.400, sae_all=0.289, low=0.0, high=0.0 | Same | Confirmed |
| h5_pilot_output.log | 2K=2.25%, 8K=0.56%, 24K=0.19% | Same | Confirmed |
| h5_pilot_output.log | All top-5 absorbed latents fire on exactly 100 tokens at ALL dict sizes (lines 77-96) | Acknowledged in Section 6.5 | Confirmed |

All paper-reported numbers match source data. No data integrity issues detected in the reported results.

---

## What Worked Well (Preserve in Revision)

1. **H1 layer-dependence correctly reported**: Abstract now reports both layer 8 falsification (0.19%) and layer 4 confirmation (49.3%). This was the most critical prior issue and has been properly fixed.

2. **H4 correctly labeled as uninformative**: The paper explicitly acknowledges H4 is uninformative and that the correct experiment was never conducted. This is honest and appropriate.

3. **Honest negative results**: Falsification criteria are pre-registered, results reported with specific numbers, conclusions match data. This remains the paper's strongest aspect.

4. **Validated metric with random dictionary control**: The Af metric is sound and reproducible. Random controls at 0.00% at all dictionary sizes confirm the metric detects genuine structure.

5. **Inverted-U finding**: The non-monotonic relationship between layer depth and absorption is a genuine and interesting finding, correctly highlighted.

---

## What Would Raise the Score to ~7.5

1. **Remove H4 causal conclusion** or run the correctly designed experiment (compare full SAE at layer 4 vs layer 8)
2. **Investigate 8 perfect-score latents** as positional artifacts — confirm token position consistency, exclude if confirmed
3. **Run full-scale (n=1,024) experiments** or explicitly scope as pilot study
4. **Add seed ablation** for key layer-dependent findings

These four steps address all critical and major issues. The paper has genuine merit (validated metric, honest negative results, reproducible framework) but the experimental gaps prevent a stronger score at this time.

---

## Recommendation

**VERDICT: REVISE**

The paper has genuine merit — the Af metric is sound, the negative results are honestly reported, and the overall contribution (systematic characterization of feature absorption in GPT-2 small SAEs) is scientifically valuable. However, the persistent critical issues (H4 causal conclusion, uninvestigated perfect-score latents, pilot-scale only) prevent advancement. Another iteration is needed to address these issues before the paper can be evaluated as a publication-quality submission.
