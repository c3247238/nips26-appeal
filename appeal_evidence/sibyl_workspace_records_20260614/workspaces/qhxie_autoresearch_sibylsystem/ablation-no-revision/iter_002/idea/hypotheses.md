# Testable Hypotheses

## H1: Layer-Dependent Absorption Prevalence

**Original Hypothesis**: Feature absorption affects >20% of latents in mid-to-deep layers of GPT-2 small SAEs.

**Evidence-Driven Revision**:
- **Layer 8 (pre-registered test layer)**: Falsified. 0.19% < 10% falsification threshold.
- **Layer 4 (exploratory finding)**: Confirmed. 49.3% > 20% threshold.

**Key Finding**: Absorption is highly layer-dependent with inverted-U pattern (100x difference between layer 4 at 49.3% and layer 8 at 0.19%).

| Layer | Mean L0 | Mean Af | % with Af > 0.5 |
|-------|---------|---------|-----------------|
| 0 | 18.9 | 0.229 | 19.5% |
| 2 | 29.1 | 0.470 | 45.5% |
| 4 | 37.8 | 0.503 | 49.3% |
| 6 | 57.0 | 0.430 | 41.0% |
| 8 | 71.9 | 0.305 | 20.9% |
| 10 | 56.0 | 0.287 | 17.3% |

---

## H2: Token Frequency Drives Absorption (CRITICAL PATH -- MUST RUN AT LAYER 4)

**Hypothesis**: Latents corresponding to low-frequency tokens are absorbed at higher rates than high-frequency ones.

**Status**: **NOT YET ANALYZED**. Critical path -- was incorrectly attempted at layer 8 (46 latents) instead of layer 4 (12,000 latents).

**CRITICAL CORRECTION**: Prior rounds tested H2 at layer 8 with only 46 absorbed latents. Layer 4 has 12,000 absorbed latents -- **260x more data**. **H2 MUST be analyzed at layer 4.**

**Operationalization**:
- Bin latents by log-frequency of top-activating tokens (quartile bins)
- Compute Spearman correlation between token frequency and absorption score
- **Use layer 4 data** (n~12,000 absorbed latents) -- NOT layer 8

**Falsification criterion**: Spearman r >= 0, or lowest-frequency bin does not show >=2x higher absorption than highest-frequency bin.

**Analysis plan**: CPU analysis on existing layer 4 data (~2 hours).

**Impact**: If H2 shows no significant negative correlation, the mechanistic story for absorption weakens considerably. The paper needs a mechanistic account of WHY mid-layers show peak absorption.

---

## H3: Sparsity-Absorption Relationship (Inverted-U Pattern)

**Original Hypothesis**: Increasing L1 sparsity penalty (lambda) increases absorption monotonically.

**Finding**: Falsified as stated. Non-monotonic inverted-U pattern detected.

| Layer | Mean L0 | Mean Af | % with Af > 0.5 |
|-------|---------|---------|-----------------|
| 0 | 18.9 | 0.229 | 19.5% |
| 2 | 29.1 | 0.470 | 45.5% |
| 4 | 37.8 | 0.503 | 49.3% |
| 6 | 57.0 | 0.430 | 41.0% |
| 8 | 71.9 | 0.305 | 20.9% |
| 10 | 56.0 | 0.287 | 17.3% |

**Correlation**: Spearman r=0.086 (p=0.872) -- no monotonic relationship.

**Revised interpretation**: Absorption peaks at mid-layers (layer 4) and declines in deeper layers. The finding applies to L0 as measured, not L1 penalty strength directly.

**Limitation**: H3 uses L0 as a proxy for L1 sparsity penalty, which conflates training hyperparameter strength (lambda) with outcome (L0). Layer-specific effects may also contribute.

---

## H4: Circuit Faithfulness (UNINFORMATIVE -- Design Flaw, Not Falsification)

**Status**: **UNINFORMATIVE**, not falsified. Original experiment design was flawed.

**Design flaw**: Task-agnostic subset selection (top/bottom 10% by absorption) does not isolate circuit-relevant absorption. Both low-absorption and high-absorption latent subsets yielded 0.0 faithfulness, which is uninterpretable.

**What the experiment actually tested**: Whether subsetting a dictionary (keeping 10% vs 100% of latents) preserves patching signal. The comparison (all latents=0.289 vs 10% subset=0.000) demonstrates that reconstruction quality matters, not which 10% is selected.

**The correct experiment (never conducted)**: Compare FULL SAE at layers with different absorption profiles (layer 4 at 49.3% vs layer 8 at 0.19%) on circuit patching tasks, holding dictionary size constant.

**Conclusion**: H4 cannot support any causal claim. Paper must explicitly acknowledge: "The experiment compared different-sized subsets of the same SAE, not different SAEs at different layers. The correct experiment was never conducted. H4 is inconclusive."

---

## H5: Dictionary Size Reduces Absorption

**Finding**: Confirmed. Monotonic decrease across dictionary sizes.

| Dictionary Size | Mean Af | % with Af > 0.5 |
|----------------|--------|-----------------|
| 2,048 | 0.0268 | 2.25% |
| 8,192 | 0.0067 | 0.56% |
| 24,576 | 0.0022 | 0.19% |

**Limitation**: These are subsets of a single 24K SAE, not independently trained dictionaries. The finding is "larger subselections show lower absorption" -- whether this holds for independently trained SAEs is an open question.

---

## H6: Perfect-Score Latent Investigation (CRITICAL PATH)

**Observation**: 8 latents have Af=1.0, each firing on exactly 100 tokens.

**Hypothesis**: Positional artifacts rather than genuine semantic absorption.

**Status**: **NOT YET ANALYZED**. Critical path.

**Confirmation in h5_pilot_output.log**: This pattern occurs at ALL dictionary sizes (2K, 8K, 24K). Each of the top-5 absorbed latents at each dictionary size fires on exactly 100 tokens. 100 = number of sequences in the pilot corpus. This regularity is too systematic to be coincidence.

**Investigation plan** (~2 hours CPU on existing layer 4 data):
1. Compute token position consistency across sequences for each perfect-score latent
2. Check if latents fire at same absolute position in each sequence
3. Compare position distribution to matched random latents
4. If confirmed as artifacts, exclude from primary analysis, document in Appendix

---

## Falsification Status Summary

| Hypothesis | Status | Key Numbers |
|------------|--------|-------------|
| H1 (layer 8) | Falsified | 0.19% vs >20% predicted; below 10% falsification threshold |
| H1 (layer 4) | Confirmed | 49.3% vs >20% threshold |
| H2 (frequency) | **PENDING** | Must analyze at layer 4 (12,000 latents), not layer 8 (46 latents) |
| H3 (monotonic L0) | Falsified | Inverted-U, r=0.086, p=0.872 |
| H4 (faithfulness) | UNINFORMATIVE | Design flaw -- correct experiment never conducted |
| H5 (dict size) | Confirmed (with limitation) | 2.25% to 0.19% monotonic decrease; subsets not independent dicts |
| H6 (positional) | **PENDING** | Investigation needed -- 8 latents firing on exactly 100 tokens |

## Critical Path (This Iteration)

**H2 + H6**: Both CPU-only on existing layer 4 data (~4 hours total), no new experiments needed.

**H2 CORRECTION**: Must run at layer 4 (12,000 absorbed latents), not layer 8 (46 latents). Prior rounds incorrectly analyzed at layer 8.

**Go/No-Go**:
- H2 confirmed (negative correlation): Continue layer-dependence + token frequency story
- H2 falsified: Pivot to metric-centric (cand_metric_centric)
- H6 confirms positional artifact: Exclude 8 latents, strengthen metric validity