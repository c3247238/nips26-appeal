# Planning Critique

## Summary

The research plan has a sound overall structure — five hypotheses covering prevalence, causation, and impact, with pilot-before-full execution and pre-registered falsification criteria. However, the plan suffers from critical deficiencies: (1) H1's hypothesis magnitude was not calibrated against existing evidence, leading to a 100x miss; (2) H4's design conflates dictionary completeness with absorption level, making the experiment impossible to interpret, and critically, H4 is framed as "falsified" when it is actually an unconducted experiment; and (3) H2 was prematurely abandoned despite existing data with 260x more statistical power at layer 4.

---

## What Works

1. **Pre-registered falsification criteria**: Every hypothesis has a binary falsification criterion (e.g., <10% for H1, Spearman r >= 0 for H2). This is good scientific practice and enables clean verdicts.

2. **Pilot-before-full structure**: Running pilots (100 sequences) before full experiments (1,024 sequences) is correct. The pilot scale correctly identifies that H1 is 100x below threshold and H4 is uninterpretable before committing full resources.

3. **Comprehensive scope**: Five hypotheses cover prevalence (H1), token frequency causation (H2), sparsity causation (H3), downstream circuit impact (H4), and dictionary size effect (H5). The scope is appropriate for a first systematic study.

4. **Logical ordering**: The hypothesis ordering (H1 → H3 → H4 → H5 → H2) follows a logical dependency: establish prevalence first, then causes, then impact, then dictionary size, then pending frequency correlation.

5. **Risk mitigation section**: The plan identifies key risks (SAELens release availability, VRAM constraints, token frequency computation) with mitigations. This is good.

---

## Critical Issues

### Issue 1: H1 Hypothesis Not Calibrated

**Problem**: H1 predicted >20% of mid-layer latents would have Af > 0.5. The observed rate was 0.19%. This is a 100x overestimate.

**Why this happened**: The hypothesis was stated without calibration against existing evidence, literature, or a small preliminary analysis. In research design, hypotheses should be informed by prior evidence. A 100x miss suggests:
- Prior evidence was ignored
- The hypothesis was stated aspirationally
- The definition of absorption differs from what was expected

**What should have happened**:
1. Review existing SAE quality metrics (e.g., Bloomfield et al. 2024's reconstruction vs. sparsity tradeoffs) for bounds
2. Run a 10-latent pilot to estimate true absorption rate before proposing a quantitative threshold
3. Calibrate the threshold against random dictionary controls to understand what "unexpectedly high absorption" would look like

**Impact on the paper**: The falsification is valid and the negative result is meaningful, but the 100x miss raises questions about hypothesis calibration methodology. A reviewer might ask: "How was the 20% threshold chosen? What prior evidence suggested this was plausible?"

---

### Issue 2: H4 Is Untested, Not Falsified (Design Conflates Variables)

**Problem**: H4's design compares low-absorption vs. high-absorption latent subsets at a single layer. However, the results show the key comparison is full SAE (0.289) vs. 10% subset (0.000), not low-absorption vs. high-absorption. Dictionary completeness — not absorption level — drives the difference.

**Why this happened**: The latent subset selection was task-agnostic (corpus-wide absorption scores), not task-specific (latents relevant to the France/Paris circuit). This conflation means:
- The subsets capture "general absorption behavior" not "circuit-relevant absorption"
- Zeroing 90% of latents destroys reconstruction regardless of which 10% is kept
- The experiment cannot distinguish between "absorption doesn't affect circuit importance" and "subset selection method is wrong"

**Critical framing issue**: The plan says H4 is "falsified" but this is incorrect. A hypothesis is falsified when the experiment tests it and results contradict the prediction. H4's experiment does NOT test the hypothesis because the causal variable was never isolated. Both subsets yielded 0.0 — this means the comparison is uninterpretable, not that absorption doesn't affect faithfulness.

**What should have been done**: Compare full SAE representations at layers with different absorption profiles (layer 4 at 49.3% vs. layer 8 at 20.9%) while holding dictionary size and training run constant. This would isolate absorption level as the causal variable.

**Impact**: H4 is uninterpretable and remains genuinely untested, not falsified. The plan should explicitly state this.

---

### Issue 3: H2 Abandoned Despite Existing Data (260x More Variance at Layer 4)

**Problem**: H2 (token frequency correlation) was marked as "pending" due to "early termination." However, the H3 layer sweep collected data at layer 4 with 49.3% absorption (~12,000 latents with Af > 0.5). Layer 8 has only 46 latents (0.19%). Layer 4 has 260x more absorbed latents.

**Why this happened**: The "early termination" rationale is self-defeating. With layer 4 data already collected, the decision to not run H2 was not data-driven — it was a missed opportunity.

**What should have happened**: Before marking H2 as pending, check whether existing H3 data can answer it. With ~12,000 absorbed latents at layer 4, there is ample statistical power for Spearman correlation against token frequency. The pre-registered analysis plan (bin latents by median token frequency, compute Spearman r) could be applied to layer 4 data directly.

**Impact**: H2 remains pending in the paper despite the data to resolve it existing. This is a gap in the research coverage that undermines the paper's completeness.

---

## Major Issues

### Issue 4: H3 Conflates L0 (Proxy) with L1 (Actual)

**Problem**: H3 hypothesizes "higher L1 sparsity penalty (lambda) monotonically increases absorption" but uses L0 (non-zero activations per token) as a proxy. This conflates:
- The sparsity penalty strength (lambda, a training hyperparameter)
- The resulting sparsity level (L0, an outcome)
- Layer-specific effects independent of sparsity

**Impact**: The finding is "absorption does not increase monotonically with L0" — not "absorption does not increase monotonically with L1 penalty." These are different claims.

**Suggestion**: In the Discussion, clarify that H3 tested L0 as a proxy, not L1 directly. The inverted-U pattern reflects L0 outcomes, not lambda sweep.

---

### Issue 5: H5 Uses Subsets Not Independent Dictionaries

**Problem**: H5 compares 2K, 8K, and 24K dictionary sizes, but the 2K and 8K are subselections of the 24K SAE, not independently trained dictionaries. This confounds dictionary size effect with superset/subset artifact.

**Impact**: The H5 finding is "larger subselections show lower absorption" not "larger dictionaries reduce absorption." The distinction matters for generalization.

**Suggestion**: Acknowledge this limitation explicitly in the plan and in the paper.

---

### Issue 6: Task Decomposition Does Not Include H2 Analysis on Existing Data

**Problem**: The task decomposition table does not include a task to analyze H2 on existing layer 4 data. H2 is listed as "pending" but no task exists to resolve it.

**Impact**: The plan fails to capture an opportunity to complete the research coverage using existing data.

**Suggestion**: Add a task "h2_analysis_existing" to apply the pre-registered Spearman correlation to layer 4 data from the H3 layer sweep.

---

## Minor Issues

### Issue 7: Figure Plan Does Not Match Actual Figure Numbering

**Problem**: The figure plan in outline.md describes Figure 2 as "Absorption Rate vs. Layer" but the actual figure (fig1_layer_absorption.pdf) is the inverted-U across all layers. The layer-4 histogram is Figure 4.

**Suggestion**: Align the figure plan with actual file numbering, or update text references to match actual figures.

---

### Issue 8: Sensitivity Analysis Listed as Appendix But Not in Task Decomposition

**Problem**: The methodology.md includes a sensitivity analysis plan (RVE thresholds 0.70, 0.80, 0.90; co-firer counts top-3, top-5, top-10) but this is not listed as a separate task in the task decomposition table.

**Impact**: The task decomposition may undercount required work.

**Suggestion**: Add sensitivity_analysis task to the task decomposition table.

---

## Recommendations for Planning Improvement

1. **Add hypothesis calibration step**: Before running experiments, add a "calibration pilot" step to estimate plausible magnitude ranges for each hypothesis. H1's 100x miss would have been caught.

2. **Redesign H4 in the plan**: The plan should explicitly state that H4 compares full SAE representations at layers with different absorption profiles (layer 4 vs. layer 8), not latent subsets. If redesign is not possible, explicitly state that H4 is an unconducted experiment.

3. **Add H2 analysis on existing data**: Include a task to run H2 on existing layer 4 data before marking it as pending.

4. **Distinguish proxy measures from actual variables**: H3's plan should clarify that L0 is a proxy for L1, not the actual variable.

5. **Acknowledge H5 subset limitation in plan**: The dictionary size comparison uses subselections, not independently trained SAEs.

6. **Align figure plan with actual files**: Match figure numbering in the plan with actual file names.

7. **Add sensitivity analysis task**: Include it in the task decomposition table.

---

## Risk Assessment

The plan identifies technical risks (SAELens release, VRAM, token frequency computation) but does not identify:
- Hypothesis calibration risk (100x miss)
- Experimental design risk (H4 conflating dictionary completeness with absorption level — and critically, the risk that H4 is untested not falsified)
- Abandoned hypothesis risk (H2 not run despite existing data)

**Suggestion**: Add "hypothesis calibration risk" and "experimental design risk" to the risk mitigation table.