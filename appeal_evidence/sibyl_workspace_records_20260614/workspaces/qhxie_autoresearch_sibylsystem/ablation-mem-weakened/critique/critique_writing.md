# Critique of Writing: Competitive Suppression in SAEs

## Overview

The paper's writing quality is generally high but has several critical framing issues that misrepresent the empirical results.

## Critical Issues

### 1. Title-Abstract-Content Mismatch (Critical)

**Problem**: The title "The Local Inhibition Graph" (and LaTeX variant "Competitive Suppression in Sparse Autoencoders") creates an expectation of a predictive tool. The abstract and introduction explicitly claim the graph "predicts absorption pairs." The results show precision@20=0.0.

**Specific misrepresentations**:
- Abstract: "the primary predictive hypothesis---that a local inhibition graph constructed from decoder correlations predicts absorption pairs---is falsified" — but the abstract presents this as one of several results, not as the decisive falsification it is.
- Section 1.4 Contributions: "First local inhibition graph for SAE diagnostics" — this is listed as a contribution even though the graph predicts nothing.
- Section 6.3: "Even in its current form, the graph identifies latents with high total incoming inhibition as candidates" — but H8 found no correlation (r=+0.12, p=0.55). This is a direct contradiction.

**Verdict**: The writing misleads readers into expecting a predictive tool that does not exist.

### 2. Overstating the H1b Result (Critical)

**Problem**: The paper presents the H1b uncorrected p=0.028 as evidence that delta-corrected steering correlates with absorption. After Bonferroni correction, the lowest p is 0.107 (BH-FDR). The paper should not claim statistical significance it does not have.

**Text**: Section 1.5: "aligning with the prior finding that layer 8 exhibits the strongest absorption-steering correlation ($r = -0.431$, $p = 0.028$ for delta-corrected steering)" — This omits that p=0.028 is uncorrected.

**Verdict**: The writing selectively highlights uncorrected p-values while omitting correction context.

### 3. Framing Falsification as Positive (Major)

**Problem**: H6 is presented as a "valuable negative result" in the discussion, but the title, abstract, contributions, and structure all promise a predictive tool. This creates cognitive dissonance.

**Specific issues**:
- Section 4.3: "The failure mode is informative" — but a 0% precision rate is not informative for the stated goal.
- The paper pivots mid-stream from "predictive tool" to "mechanistic framework" — the writing should have been structured around the theoretical contribution from the start.

**Verdict**: The writing should choose one framing and stick to it.

### 4. Missing Limitations Section Honesty (Major)

**Problem**: The limitations section (5.5) mentions 5 limitations but omits the most critical:
- Zero significant results after MCP
- n=26 provides insufficient power for correlation analysis
- The matched L0 confound in OrtSAE comparison

**Verdict**: Limitations are selectively framed to avoid the most damaging admissions.

## What Works

1. **Clear mechanistic explanation** (Section 3.2): The four-step competitive suppression mechanism is intuitive and well-structured.

2. **Honest null-result reporting** (Section 4.3-4.5): The paper explicitly states H6 and H8 are falsified.

3. **Table 5 integration**: Prior findings explained by the competitive suppression framework.

## Recommendations

1. **Retitle** to emphasize the theoretical contribution, not the failed graph.
2. **Revise abstract** to lead with LCA-theoretical framework, not graph predictions.
3. **Add MCP context** wherever p-values are cited.
4. **Drop the graph as diagnostic** — the graph failed, so stop claiming it works.
5. **Acknowledge power limitations** explicitly in the limitations section.
