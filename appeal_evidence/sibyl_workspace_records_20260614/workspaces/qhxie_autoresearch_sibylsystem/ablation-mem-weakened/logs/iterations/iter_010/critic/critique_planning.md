# Critique: Planning (Iteration 9/10)

## Overview

The planning documents (iteration 9 proposal and iteration 10 methodology) are thorough but have critical gaps: the revision strategy addresses feedback from iteration 9 (score: 6/10) but does not correct the fundamental framing problem, and the data files referenced do not all exist or contain the claimed statistics.

## Critical Issues

### 1. Revision Strategy Doesn't Fix Title-Content Mismatch (CRITICAL)

**Problem**: The iteration 10 methodology (lines 46-48) identifies the title problem but proposes only a surface fix:

> **Old**: "The Local Inhibition Graph" or similar graph-emphasis
> **New**: "Feature Absorption as Optimal Compression: Evidence that Training Reduces Structural Artifacts"

This new title still has the same problem: it implies the "optimal compression" framing is the contribution, but H6 (the primary prediction of that framing) is falsified. The revision should acknowledge the falsification in the title itself, or change the title to match what actually succeeded (null results + metric validation).

**What actually happened in iteration 9**:
- cand_g front-runner: "Feature Absorption as Optimal Compression"
- H6 falsified: precision@20=0.0
- The title remains unchanged in the writing/paper.md

**Recommendation**: Choose title based on what succeeded:
- Option A: "Feature Absorption Does Not Degrade SAE Interpretability: Null Results and Metric Validation" (emphasizes null result)
- Option B: "The Mechanism of SAE Feature Absorption: Competitive Suppression Explains Precision-Recall Asymmetry" (emphasizes mechanistic explanation with H6 falsification acknowledged)

### 2. Missing Data Source for H8 Claim (CRITICAL)

**Problem**: The iteration 10 methodology (line 13) flags: "Missing Data Source for H8 Claim: Section 4.5 claims 'r=+0.12, p=0.55' but no data file contains this."

**What the paper says** (writing/paper.md line 259):
> "Total incoming inhibition shows no reliable relationship with absorption rate (descriptive r = +0.12, p = 0.55; computed from per-feature graph statistics at layer 8)"

**Where this should be in h6_inhibition_graph.json**: The file contains per-feature graph statistics (clustering coefficient, total incoming inhibition, etc.) correlated with absorption rates. If this file doesn't have the r=+0.12 value, the claim is unverifiable.

**What to do**: Either (a) compute the correlation from h6_inhibition_graph.json and verify/add the result, or (b) remove the claim entirely.

### 3. Planned Revisions Don't Address Multiple Comparisons Problem (CRITICAL)

**Problem**: The iteration 10 methodology mentions "Multiple comparison correction: Bonferroni (alpha=0.00417) and BH-FDR (q<0.05) applied to all 12 tests" as a baseline, but doesn't plan to fix the presentation of H1b p=0.028.

The methodology (lines 67-69) says:
> "Data Verification: Verify or remove H8 claim about total incoming inhibition"

But doesn't say: "Remove H1b p=0.028 as evidence of effect (not significant after MCP)".

**Recommendation**: Add to revision plan:
> "Remove all uncorrected p-values from results discussion. H1b at L8 (raw p=0.028) does not survive Bonferroni correction (p=0.334) and should not be cited as evidence of effect."

### 4. Figure Files May Not Exist (MAJOR)

**Problem**: The paper references:
- `fig1_lca_correspondence.pdf` (Figure 1)
- `fig2_suppression_mechanism.pdf` (Figure 2)
- `fig7_precision_recall.pdf` (Figure 3)

But I cannot verify these files exist. The exp/results/figures/ directory exists but may be empty or contain different files.

**What I see**: `exp/results/figures/` exists but I didn't list its contents.

**Recommendation**: Check which figures actually exist. If they don't exist, the paper needs to either generate them or remove references.

### 5. Planned Visualizations Don't Match Existing Data (MAJOR)

**Problem**: The methodology (lines 77-86) lists expected visualizations:
- Figure 1: Architecture diagram (LCA-SAE correspondence) - if needed
- Table 1: Hypothesis testing summary with corrected p-values
- Figure 2: Precision-recall asymmetry scatter
- Table 2: Feature-level absorption and downstream data
- Figure 3: Random vs trained SAE absorption comparison

But the writing/paper.md doesn't have:
- Table 1 (hypothesis testing summary with corrected p-values)
- Figure 3 (random vs trained comparison)

The paper has inline tables but no formal Table 1 with all corrected p-values.

**Recommendation**: Generate the missing table (corrected p-values for all 12 tests) and figure (random vs trained absorption comparison).

## Summary

The planning documents identify issues correctly but the revision strategy is incomplete. Critical fixes missing:
1. Title should reflect actual findings, not hypothesized success
2. H8 statistic needs data source verification or removal
3. H1b p=0.028 needs to be explicitly removed as evidence
4. Figure files need verification
5. Missing table (corrected p-values) and figure (random vs trained) need to be generated

**Action items**:
1. Update title to match actual findings or explicitly acknowledge H6 falsification
2. Verify H8 statistic from h6_inhibition_graph.json or remove claim
3. Add explicit note to remove H1b as evidence (not significant after MCP)
4. Verify all figure files exist
5. Generate Table 1 (corrected p-values) and Figure 3 (random vs trained)