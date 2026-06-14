# Detailed Critique: Experiment Design and Execution

## Summary

The experimental setup (GPT-2 small + SAELens SAEs on Pile data) is sound and the absorption score metric is well-motivated with validation against random controls. However, critical design flaws undermine several key conclusions: H4's comparison isolates dictionary completeness (not absorption level) as the causal variable but the correct experiment was never run; H2 was abandoned despite layer 4 having 260x more absorbed latents than layer 8; 8 perfect-score latents likely represent positional artifacts; and only pilot scale (100 sequences) was used instead of the planned full-scale (1,024 sequences).

## Critical Issues

### 1. H4 Experiment Does Not Isolate Absorption as Causal Variable

**Location**: Section 5.3, Table 4

**Results**:
| Patching Method | Faithfulness |
|---|---|
| Raw residual stream | 0.400 |
| SAE all latents | 0.289 |
| SAE low-absorption (bottom 10%) | 0.000 |
| SAE high-absorption (top 10%) | 0.000 |

**Problem**: The paper concludes "absorption level does not predict which SAE latents are causally important for circuit tracing" and "dictionary completeness — not absorption level — drives patching fidelity." However, the experiment compared different-sized subsets of the SAME SAE at the SAME layer, not different SAEs at different layers with different absorption profiles.

**What the experiment actually tested**: Whether subsetting a dictionary (keeping 10% vs 100% of latents) preserves patching signal. The 0.289 (all latents) vs 0.000 (10% subset) comparison demonstrates that reconstruction quality is what matters, not which 10% is kept.

**The correct experiment**: Compare FULL SAE patching at layer 4 (49.3% absorption, ~12,000 absorbed latents) vs layer 8 (0.19% absorption, ~47 absorbed latents), holding dictionary size constant. This would isolate whether overall absorption level drives patching fidelity. This experiment was never conducted.

**Impact**: The H4 conclusion about absorption not predicting importance is unsupported. The data supports "subsetting destroys signal" but does not support "absorption level does not predict importance." The correct conclusion is that the experiment is uninformative because the design does not isolate the hypothesized variable. H4 is inconclusive, not falsified.

### 2. 8 Perfect-Score Latents Are Likely Positional Artifacts

**Location**: Section 5.1, h5_pilot_output.log

**Finding**: Each of the 8 latents with A_f = 1.0 fires on exactly 100 tokens. This equals the number of sequences in the pilot corpus (100 sequences x 128 tokens each, one activation per sequence).

**Analysis**: The regularity strongly suggests a positional artifact:
- If these latents fired on semantically meaningful tokens, the token counts would vary across latents
- If these latents fired on syntactic tokens (e.g., beginning-of-sentence), position regularity would produce identical counts
- "Always-on" features (>90% of tokens) are already excluded; these 8 latents firing on exactly 100 tokens (0.78% of 12,800 tokens) may represent position-specific features that should be similarly excluded

**Current treatment**: The paper acknowledges "suspicious uniformity" but does not investigate further.

**Recommendation**: Compute token position consistency across sequences for each perfect-score latent. If confirmed as positional artifacts (e.g., all firing at token position X across all sequences), exclude from primary analysis and document in Appendix.

### 3. H2 Could Have Been Tested at Layer 4 (260x More Data)

**Location**: Section 5.5, pilot_summary.json

**Paper's rationale for not testing H2**: "insufficient absorption variance at layer 8 (only 46/24,576 latents with A_f > 0.5)"

**Problem**: This rationale is self-defeating. Layer 4 has 49.3% absorption (~12,000 latents with A_f > 0.5) — 260x more absorbed latents than layer 8. The paper never attempted H2 analysis at layer 4.

**H2 at layer 4 would have**: 12,000 absorbed latents with which to compute token frequency correlations, vs. only 46 at layer 8. Even using a continuous absorption metric (mean RVE across all activating tokens), layer 4 provides vastly more statistical power.

**The early termination rationale is therefore invalid**: H2 was abandoned because layer 8 had insufficient variance, when layer 4 had ample variance. The decision to abandon H2 was not based on the actual data from the most informative layer.

**Required action**: Either (a) run H2 on the existing layer 4 data (which has sufficient variance for Spearman correlation with either binary threshold or continuous mean RVE metric), or (b) explicitly retire H2 with justification: "H2 was not tested at layer 8 due to insufficient absorption variance (46/24,576 latents). Layer 4 (49.3% absorption, ~12,000 latents) was identified as the appropriate alternative but the experiment was not conducted. H2 is formally retired: insufficient statistical power even at the highest-absorption layer tested."

## Major Issues

### 4. Only Pilot Scale (100 sequences) Used, Not Full Scale (1,024)

**Location**: Section 4.2, pilot_summary.json

**Methodology states**: Full experiments use 1,024 sequences; pilot experiments use 100 sequences.

**Actual execution**: All results reported in the paper use only 100 sequences (pilot scale), not 1,024 sequences (full scale). The 10x scale gap is material for rare phenomenon characterization.

**Impact**: The paper's findings are scoped to pilot experiments. Any claims about absorption prevalence, sparsity relationships, or downstream impact should be qualified as "pilot-scale findings pending full-scale replication." The 100x gap for H1 (0.19% vs >20%) is too large to bridge with 10x scaling, but other hypotheses (H3 inverted-U, H5 direction) might change at full scale.

**Recommendation**: Either run full-scale experiments (n=1,024) or explicitly scope the paper as a pilot study with commitment to full-scale replication.

### 5. H3 Inverted-U Pattern Has No Mechanistic Explanation

**Finding**: Absorption peaks at layer 4 (49.3%) and declines toward both shallow (layer 0: 19.5%) and deep layers (layer 10: 17.3%). This is an inverted-U, not the hypothesized monotonic increase.

**Problem**: The paper provides no theory for why mid-layers would have peak absorption. The speculation ("mid-layers process densest abstract and semantic content") is post-hoc and unsupported by transformer architecture literature.

**Impact**: The falsification of H3 (no monotonic increase) is clean, but the positive finding (inverted-U) is unexplained. Without a mechanistic account, the inverted-U is an interesting observation, not a contribution.

**Recommendation**: Either (1) propose a mechanistic hypothesis for the inverted-U with citations to transformer architecture literature on layer specialization, attention vs MLP composition at different depths, or residual stream evolution across blocks; or (2) explicitly frame the inverted-U as exploratory: "The inverted-U pattern at mid-layers is an unexpected empirical observation requiring future investigation — we report it as a finding, not a contribution with mechanistic backing."

### 6. No Seed Ablation for Key Findings

**Issue**: All experiments use seed=42 with no confirmation across seeds. The dramatic layer difference (layer 4 at 49.3% vs layer 8 at 0.19%) is particularly concerning for reproducibility. If this is a seed artifact, the layer-dependence conclusion is invalid.

**Recommendation**: Add seed ablation (2-3 seeds) for key findings (H1 layer comparison, H3 inverted-U) before publication. If seed ablation is not feasible before next milestone, explicitly acknowledge as limitation.

## Minor Issues

### 7. H5 "10-fold Reduction" Framing Overstates Practical Significance

**Location**: Section 5.4, Abstract

**Finding**: Absorption decreases from 2.25% (2K) to 0.19% (24K) — paper calls this a "10-fold reduction."

**Problem**: The absolute numbers are what matter for practical significance:
- 97.75% non-absorbed at 2K
- 99.81% non-absorbed at 24K

The "10-fold reduction" framing makes a 2% absolute difference sound dramatic. In practical terms, both dictionary sizes have negligible absorption rates — the phenomenon is rare regardless of dictionary size.

**Recommendation**: Lead with absolute numbers: "97.75% non-absorbed at 2K vs 99.81% at 24K" and note that "the effect explains <1% of variance."

### 8. H5 Subsampling Limits Conclusiveness

**Location**: Section 5.4

**Method**: "cumulatively subsampling latents, prioritizing absorbable latents" to simulate smaller dictionary sizes.

**Problem**: The 2K, 8K, 24K results are not independent dictionaries — the smaller dicts are subsets of the full dict, not separately trained SAEs. This confounds dictionary size effect with superset/subset artifact.

**Recommendation**: Acknowledge explicitly: "The comparison is between a full dictionary and its subselections, not between independently trained SAEs of different sizes. This limits the conclusiveness of the H5 finding."

## Summary of Required Changes

| Issue | Severity | Required Action |
|-------|----------|-----------------|
| H4 conclusion unsupported | Critical | Remove "absorption does not predict importance" — report as inconclusive, note correct experiment never conducted |
| 8 perfect-score latents | Critical | Investigate as positional artifacts; exclude if confirmed |
| H2 abandoned despite layer 4 data | Critical | Run H2 at layer 4 (12,000 absorbed latents) or formally retire with correct justification |
| Pilot-only scale, not full-scale | Major | Run 1,024-sequence experiments or explicitly scope as pilot study |
| H3 inverted-U unexplained | Major | Add mechanistic hypothesis or frame as exploratory |
| No seed ablation | Minor | Add seed ablation for key findings |
| H5 framing overstates | Minor | Use absolute numbers (97.75% vs 99.81% non-absorbed) |
| H5 subsampling limits | Minor | Acknowledge comparison is between subsets, not independent dictionaries |