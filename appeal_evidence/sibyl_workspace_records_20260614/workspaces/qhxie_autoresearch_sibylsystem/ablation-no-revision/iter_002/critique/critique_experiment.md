# Experiment Critique

## Summary

The experimental program is methodologically sound in its core metric design and validation strategy, but suffers from three critical flaws: (1) H1's hypothesis was not calibrated against pilot data, leading to a 100x overestimate; (2) H4's design conflates dictionary completeness with absorption level, making the key comparison uninterpretable, and importantly, H4 is not "falsified" — it is genuinely untested because the causal variable was never isolated; and (3) H2 was abandoned despite existing data at layer 4 with 260x more absorption variance than layer 8. The pilot-before-full-experiment structure is correct and should be preserved. The NO-GO recommendation from pilot_summary.json was correct and should have triggered redesign before any further experimentation.

---

## Critical Issues

### Issue 1: H1 Hypothesis Not Calibrated (100x Miss)

**Evidence**: pilot_summary.json shows H1 predicted >20% absorption prevalence, observed 0.19% — a 100x overestimate.

**Root Cause**: The hypothesis was stated without calibration against existing evidence or a small pilot. In research design, hypotheses should be informed by prior evidence. A 100x miss suggests the prior evidence was ignored or the hypothesis was stated aspirationally.

**What should have happened**:
1. Before proposing H1, run a 10-latent pilot to estimate the true absorption rate
2. Calibrate the threshold against random dictionary controls
3. Review literature on SAE quality metrics for bounds

**Impact on paper credibility**: A 100x miss, even in a negative result paper, raises questions about hypothesis calibration methodology. The falsification is valid but the overestimation suggests poor prior calibration.

---

### Issue 2: H4 Is Untested, Not Falsified (Design Conflates Variables)

**Evidence**: Table 4 shows:
- SAE all latents: 0.289 faithfulness
- SAE low-absorption: 0.000
- SAE high-absorption: 0.000
- Raw residual: 0.400

**Root Cause**: The experiment selects latent subsets based on corpus-wide absorption scores, not circuit-specific importance. This means:
1. Subsets capture "general absorption behavior" not "circuit-relevant absorption"
2. Zeroing 90% of latents destroys reconstruction regardless of which 10% is kept
3. The experiment cannot distinguish "absorption doesn't affect circuit importance" from "subset selection method is wrong"

**Critical framing issue**: The paper says H4 is "falsified" but this is incorrect. A hypothesis is falsified when the experiment tests it and the results contradict the prediction. H4's experiment does NOT test the hypothesis because the causal variable (circuit-relevant absorption) was never isolated. Both subsets yielded 0.0, which means the comparison is uninterpretable — not that absorption doesn't affect faithfulness.

**What should have been done**: Compare full SAE representations at layers with different absorption profiles (layer 4 at 49.3% vs. layer 8 at 20.9%) while holding dictionary size constant. This would isolate absorption level as the causal variable.

**Impact**: H4 is uninterpretable and remains genuinely untested, not falsified. The paper correctly identifies the design flaw in Section 6.2 but should explicitly state: "H4 remains an unconducted experiment — the pilot results do not support any conclusion about whether absorption level predicts circuit-level causal importance."

---

### Issue 3: H2 Abandoned Despite Existing Data (Layer 4 Has 260x More Variance)

**Evidence**: H3 layer sweep collected data at layer 4 with 49.3% absorption (~12,000 latents with Af > 0.5). H1 at layer 8 has only 46 latents (0.19%). Layer 4 has 260x more absorbed latents than layer 8.

**Root Cause**: "Early termination" rationale. The paper's own Section 5.5 correctly identifies that H2 could be run on layer 4 data but does not act on it.

**What should have happened**: Before marking H2 as pending, check whether existing H3 data can answer it. With ~12,000 absorbed latents at layer 4, there is ample statistical power for Spearman correlation against token frequency.

**Impact**: H2 remains pending in the paper despite the data to resolve it existing. This is a missed opportunity and a gap in the research coverage.

---

## Major Issues

### Issue 4: H3 Uses L0 as Proxy for Sparsity Penalty (Lambda)

**Evidence**: Table 2 shows layer 8 has highest L0 (71.9) but lowest absorption (20.9% > 0.5). Layer 4 has L0=37.8 but highest absorption (49.3%). Spearman r=0.086 (p=0.872).

**Problem**: H3 hypothesizes "higher L1 sparsity penalty (lambda) monotonically increases absorption." The experiment uses L0 (non-zero activations per token) as a proxy. This conflates:
- The sparsity penalty strength (lambda, a training hyperparameter)
- The resulting sparsity level (L0, an outcome)
- Layer-specific effects independent of sparsity

**Implication**: The finding is "absorption does not increase monotonically with L0" — not "absorption does not increase monotonically with L1 penalty." These are different claims with different implications.

**Suggestion**: Add explicit language in Discussion clarifying that H3 tested L0 as a proxy, not L1 directly. The inverted-U pattern reflects L0 outcomes, not lambda sweep. Whether the same pattern holds for actual L1 penalty variations requires controlled training experiments.

---

### Issue 5: H5 Dictionary Size Comparison Uses Subsets Not Independent Dictionaries

**Evidence**: pilot_summary.json and Table 3 show 2K, 8K, 24K comparison using subsampled 24K SAE.

**Problem**: 2K and 8K are subselections of the 24K SAE, not independently trained dictionaries. This confounds:
- Dictionary size effect
- Superset/subset artifact (subsets may not be representative of independently trained SAEs)
- The "upper bound" framing (prioritizing absorbable latents) introduces additional bias

**Impact**: The H5 finding is "larger subselections show lower absorption" not "larger dictionaries reduce absorption." The distinction matters for generalization.

**Suggestion**: Acknowledge explicitly that the comparison is between a full dictionary and its subselections, limiting the conclusiveness of the dictionary size effect. The finding is conditional on the 24K SAE — whether it holds for independently trained SAEs at different dictionary sizes is an open question.

---

### Issue 6: Pilot Scale May Miss Rare Effects

**Evidence**: H1 pilot used 100 sequences (12,800 tokens). With 0.19% prevalence (46/24,576 latents), rare effects at the 0.01% level would need full corpus to detect.

**Problem**: The paper correctly notes the 100x gap is unlikely to close with 10x scale increase. However, effects at the 0.1% level may be real and worth characterizing if they exist in larger models.

**Suggestion**: Acknowledge that pilot scale limits detection of rare absorption events. Full 1,024-sequence corpus may reveal effects at the 0.1% level that the pilot misses.

---

## What Works

1. **Metric validation strategy**: Random dictionary control is excellent. The metric correctly yields 0.00% on random decoders, confirming it detects real structure. This validation approach should be standard in SAE research.

2. **Pilot-before-full structure**: Running pilots before full experiments is correct. The NO-GO recommendation from pilot_summary.json was appropriate given the 100x miss and H4 uninterpretability.

3. **Multi-layer sweep**: Auditing 6 layers (0, 2, 4, 6, 8, 10) provides good coverage of depth-dependent effects. The inverted-U finding (layer 4 peak) would have been missed with single-layer sampling.

4. **Honest negative result reporting**: pilot_summary.json's specific numbers (0.19% vs >20%) and falsification status are exactly what is needed. This is the paper's strongest aspect.

5. **Edge case handling**: The always-on feature exclusion (90% threshold) and the 8 perfect-score latents discussion are appropriate. These edge cases are correctly flagged as potentially artifactual.

---

## Recommendations

1. **Calibrate hypotheses against pilots before running full experiments**: The 100x H1 miss would have been caught with a 10-latent preliminary analysis.

2. **Redesign H4 to isolate absorption as causal variable**: Full SAE at layer 4 vs. layer 8, not latent subsets at a single layer. If redesign is not possible, explicitly retire H4 as untested.

3. **Run H2 on existing layer 4 data**: With ~12,000 absorbed latents, there is ample power. Either run it or explicitly retire it with justification.

4. **Distinguish L0 (proxy) from L1 (actual)**: H3's finding applies to L0 outcomes, not lambda directly. Clarify this distinction in the Discussion.

5. **Acknowledge H5 subset limitation**: The dictionary size comparison is between subselections of a single SAE, not independently trained dictionaries.

6. **Investigate the 8 perfect-score latents**: 100 activations each (= number of sequences) suggests position-embedding artifact. Verify whether these are genuine absorption or artifacts.

---

## Data Integrity Check

All major quantitative claims trace to pilot_summary.json:
- H1: 0.19% (46/24,576), 99.4% have Af=0.0 (H1 condition with d_sae=8,192) ✓
- H3: Layer 4 mean Af=0.503, 49.3% > 0.5 (d_sae=24,576) ✓
- H4: faithfulness (0.400, 0.289, 0.000, 0.000) ✓
- H5: dict size breakdown (2K: 2.25%, 8K: 0.56%, 24K: 0.19%) ✓

All numbers match source data exactly. No cherry-picking detected. One noted inconsistency: the 99.4% figure for layer 8 applies to the H1 condition (d_sae=8,192), while Table 2's 76.8% for layer 8 applies to d_sae=24,576 — the paper should clarify this distinction clearly.