# Ideation Critique

## Summary

The research question — systematically quantifying feature absorption in SAEs — is well-motivated and original. No prior work has systematically measured absorption prevalence or correlated it with downstream causal validity. The five-hypothesis structure is logical and covers the key dimensions: prevalence (H1), causation (H2, H3), and impact (H4, H5). However, the ideation suffers from critical flaws: (1) H1 hypothesis magnitude was not calibrated against existing evidence, leading to a 100x overestimation; (2) H4's design conflates dictionary completeness with absorption level, making the experiment impossible to interpret; (3) H4 is framed as "falsified" when it is actually an unconducted experiment; and (4) H2 was abandoned despite existing data with 260x more statistical power at layer 4.

---

## What Works

1. **Novel contribution**: The absorption metric and systematic characterization across layers/dictionary sizes is genuinely novel. No prior work has quantified absorption per latent using reconstruction variance explained by co-firers. The metric design is sound — top-5 co-firers with 80% RVE threshold was calibrated against random dictionary controls.

2. **Comprehensive scope**: Five hypotheses covering prevalence, causes (frequency and sparsity), and downstream impact (circuit faithfulness and dictionary size) provide good coverage of the research space.

3. **Clear operationalization**: Each hypothesis has a pre-registered falsification criterion, enabling binary verdicts. This is good scientific practice.

4. **Honest negative results**: The paper correctly identifies when hypotheses fail or experiments become uninformative, rather than spinning negative results. This is the paper's strongest quality.

---

## Critical Issues

### Issue 1: H1 Hypothesis Not Calibrated (100x Miss)

**Problem**: The H1 hypothesis predicted >20% of mid-layer latents would have $A_f > 0.5$. The observed rate was 0.19% — a 100x overestimate. This suggests the hypothesis was not calibrated against existing evidence or literature.

**Why this matters**: In research design, hypotheses should be informed by prior evidence or theoretical bounds. A 100x miss suggests either (a) the prior evidence was ignored, (b) the hypothesis was stated aspirationally rather than empirically, or (c) the definition of absorption differs from what was expected.

**What should have happened**: Before proposing H1, the researchers should have:
- Reviewed literature on SAE quality metrics (e.g., Bloomfield et al. 2024's reconstruction vs. sparsity tradeoffs) for bounds
- Estimated absorption rates from smaller pilots or pilot-scale analyses
- Calibrated the threshold against random dictionary controls to understand what "unexpectedly high absorption" would look like

**Suggestion**: In the Discussion, add a section on "Hypothesis Calibration" explaining why H1 was overstated by 100x and what this implies for hypothesis formation in SAE research. A 10-latent pilot would have caught this.

---

### Issue 2: H4 Is Untested, Not Falsified (Design Conflates Variables)

**Problem**: H4 aims to test whether absorption level predicts circuit causal importance. The experiment compares low-absorption vs. high-absorption latent subsets at the same layer. However, the results show that full SAE (0.289) vs. 10% subset (0.000) drives the difference, not absorption level. The key test (low-absorption vs. high-absorption) is uninterpretable because both subsets yield 0.0.

**Why the framing is wrong**: The paper says H4 is "falsified" but this is incorrect. A hypothesis is falsified when the experiment tests it and results contradict the prediction. H4's experiment does NOT test the hypothesis because the causal variable was never isolated. Both subsets yielded 0.0 — this means the comparison is uninterpretable, not that absorption doesn't affect faithfulness.

**What should have been done**: Compare full SAE representations at layers with different absorption profiles (e.g., layer 4 with 49.3% vs. layer 8 with 20.9%) while holding dictionary size and training run constant. This would isolate absorption level as the causal variable.

**Suggestion**: Retire the current H4 finding as uninterpretable. Either redesign the experiment (full SAE at layer 4 vs. layer 8) or explicitly state that H4 remains an unconducted experiment.

---

### Issue 3: H2 Abandoned Despite Available Data

**Problem**: H2 (token frequency correlation) was listed as "pending" due to "early termination." However, the H3 layer sweep collected data at layer 4, which has 49.3% absorption (~12,000 latents with $A_f > 0.5$) — 260x more absorbed latents than layer 8 (46 latents). The data to run H2 already exists in the H3 dataset.

**Why this is a problem**: The "early termination" rationale is self-defeating. With layer 4 data already collected, the decision to not run H2 was not data-driven — it was a missed opportunity. The paper's own Section 5.5 correctly identifies this but does not act on it.

**Suggestion**: Either (a) run H2 on the existing layer 4 data, or (b) explicitly retire H2 with justification: "insufficient statistical power even at the highest-absorption layer tested." Do not leave H2 in pending status when the data to resolve it exists. The pre-registered analysis (bin latents by median token frequency, compute Spearman r) can be applied to layer 4 data directly.

---

## Major Issues

### Issue 4: H3 Uses L0 as Proxy for Sparsity Penalty

**Problem**: H3 hypothesizes "higher L1 sparsity penalty (lambda) monotonically increases absorption." The experiment proxies this with L0 (average number of non-zero activations per token). This proxy conflates:
- The sparsity penalty strength (lambda, a training hyperparameter)
- The resulting sparsity level (L0, an outcome measure)
- Layer-specific effects (layer 4 has different feature structure than layer 8 independent of sparsity)

**Why this matters**: The inverted-U pattern (absorption peaks at layer 4) could reflect:
- Layer-specific semantic density (mid-layers handle abstract content)
- The proxy being wrong (L0 is not a good proxy for lambda)
- A genuine non-monotonic relationship between sparsity and absorption

**Suggestion**: In the Discussion, clarify that the H3 finding applies to L0 as measured, not to L1 penalty strength directly. The interpretation is "absorption does not increase monotonically with L0" not "absorption does not increase monotonically with L1 penalty." These are different claims, and whether the same pattern holds for actual L1 penalty variations requires controlled training experiments.

---

### Issue 5: H5 Uses Subsets Not Independent Dictionaries

**Problem**: H5 compares 2K, 8K, and 24K dictionary sizes. However, the 2K and 8K are subselections of the 24K SAE, not independently trained dictionaries. This means:
- The comparison confounds dictionary size with superset/subset effects
- Subsets may not be representative of independently trained 2K SAEs
- The "upper bound" framing (prioritizing absorbable latents) introduces additional bias

**Suggestion**: Acknowledge this limitation explicitly. The H5 finding is "larger subselections of the 24K dictionary show lower absorption than smaller subselections" — not "larger dictionaries reduce absorption." The distinction matters for generalization to independently trained SAEs.

---

### Issue 6: Perfect-Score Latents ($A_f = 1.0$) Need Investigation

**Problem**: Eight latents score $A_f = 1.0$, each firing on exactly 100 tokens. This regularity (100 = number of sequences) suggests a position-embedding artifact, not genuine semantic absorption.

**Suggestion**: The paper mentions these in Section 5.1 but does not investigate them. Add a brief analysis (e.g., do these latents fire at consistent token positions across sequences?) to determine whether they are artifacts or genuine absorption. This is important because these high-absorption cases could skew the metric's interpretation if they are artifacts.

---

## Recommendations for Future Iterations

1. **Calibrate hypotheses against pilot data**: Before running full experiments, run small pilots to calibrate hypothesis magnitudes. H1's 100x miss would have been caught with a 10-latent pilot. This is the single most important improvement.

2. **Design H4 to isolate absorption as causal variable**: If comparing subsets, they must be task-specific (relevant to the circuit being traced), not corpus-wide absorption scores. Or compare full SAE representations at different layers.

3. **Use existing data before abandoning hypotheses**: Before marking H2 as pending, check whether existing H3 data can answer it. The 260x more absorbed latents at layer 4 vs. layer 8 would have provided ample statistical power.

4. **Distinguish proxy measures from actual variables**: H3 conflates L0 (proxy) with L1 (actual). Make this distinction explicit in hypothesis framing.

5. **Use independently trained SAEs for dictionary size comparisons**: Subsets of a single SAE conflate dictionary size with superset artifacts.

---

## Novelty Assessment

**Strong aspects**: The absorption metric (top-5 co-firers, 80% RVE threshold) is genuinely novel — no prior work has measured absorption per latent using reconstruction variance explained. The systematic layer sweep (6 layers) and dictionary size sweep (3 sizes) provide comprehensive coverage.

**Weakened aspects**: H2 (token frequency correlation) is the most novel hypothesis and was not tested. The paper's novelty ceiling is limited to H1/H3/H4/H5 findings, which are largely negative or uninterpretable. H4's design flaw means the downstream impact question is genuinely unanswered.

**Positioning**: The paper does not explicitly compare against or reference prior work that might have used different operationalizations of absorption or related phenomena (e.g., Elhage et al. 2022's polysemy measure, Sharkey et al. 2023's correlation-based metrics). Adding this comparison in Section 2.2 would clarify what the paper contributes beyond existing superposition measures.