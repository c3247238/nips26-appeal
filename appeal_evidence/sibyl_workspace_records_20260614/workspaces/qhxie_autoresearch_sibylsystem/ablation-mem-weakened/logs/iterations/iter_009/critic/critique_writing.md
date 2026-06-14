# Critique of Writing: Competitive Suppression in Sparse Autoencoders

## Overview

The paper has a strong theoretical foundation (LCA-SAE structural correspondence) but systematically misrepresents its empirical results. The most critical problems are framing a null-result study as if it had significant findings, and claiming a predictive graph tool that predicts nothing.

## Critical Issues

### 1. Title-Abstract-Contributions Mismatch (CRITICAL)

**Problem**: The title "Competitive Suppression in Sparse Autoencoders" and Section 1.4 contribution "First local inhibition graph for SAE diagnostics" create an expectation of a predictive tool. The abstract and introduction explicitly claim the graph "predicts absorption pairs." Results show precision@20=0.0 (H6 decisively falsified, Fisher p=1.0).

**Specific misrepresentations**:
- Abstract: "the primary predictive hypothesis---that a local inhibition graph constructed from decoder correlations predicts absorption pairs---is falsified" — presented as one of several results, not as the decisive falsification that undermines the paper's stated contribution
- Section 1.4: "First local inhibition graph for SAE diagnostics" — listed as a contribution despite the graph predicting nothing
- Section 6.3: "Even in its current form, the graph identifies latents with high total incoming inhibition as candidates" — but H8 found no correlation (r=+0.12, p=0.55), so the graph does NOT identify at-risk features
- Section 4.8 Table 5 claims the LCA framework explains prior findings — but the mechanism is inferred from results, not validated by the graph predictions that were the stated contribution

**Verdict**: The writing systematically misleads readers into expecting a predictive tool that does not exist. The paper should be retitled to emphasize the LCA-theoretical framework, not graph predictions.

### 2. Overstating the H1b Result (CRITICAL)

**Problem**: The paper presents H1b uncorrected p=0.028 as evidence of a real effect. After Bonferroni correction (12 tests, alpha=0.00417), the corrected p=0.334. After BH-FDR, q=0.107. The paper has ZERO statistically significant results after proper multiple comparison correction.

**Specific misrepresentations**:
- Section 1.5: "aligning with the prior finding that layer 8 exhibits the strongest absorption-steering correlation ($r = -0.431$, $p = 0.028$ for delta-corrected steering)" — omits that p=0.028 is uncorrected and does not survive MCP
- Section 4.5: "the correlation is weak and non-significant" is stated for H8 but the same honesty is not applied to H1b
- Section 3.6: Post-hoc power analysis ("approximately 20% power") is used to explain away null results — this is methodologically questionable (power should be computed before the experiment, not after to explain away failures)

**Verdict**: The writing selectively highlights uncorrected p-values while omitting critical MCP context. Every p-value citation in the paper should include: "p=X uncorrected; Bonferroni-corrected p=Y; BH-FDR q=Z."

### 3. Framing Falsification as Contribution (MAJOR)

**Problem**: H6 is presented as a "valuable negative result" in the discussion (Section 5.1), but the title, abstract, contributions, and structure all promise a predictive tool.

**Specific issues**:
- Section 4.3: "The failure mode is informative" — a 0% precision rate is not informative for the stated goal of predicting absorption pairs
- The paper pivots mid-stream from "predictive tool" to "mechanistic framework" — the writing should have been structured around the theoretical contribution from the start
- Section 4.8 Table 5 claims the LCA framework "integrates all prior findings" — but the mechanism is post-hoc inference, not the graph-based validation that was the stated contribution

**Verdict**: Either restructure the paper around the LCA-theoretical framework (legitimate) or be honest that the graph predictions failed. Do not claim both.

### 4. Missing Limitations Section Honesty (MAJOR)

**Problem**: The limitations section (5.5) mentions 5 limitations but omits the most critical:
- Zero significant results after MCP (the paper's main empirical finding)
- n=26 provides insufficient power for reliable correlation analysis
- The matched L0 confound in OrtSAE ablation comparison
- MCC~0.21 at chance level for Random SAEs invalidates the matching pipeline

**Verdict**: Limitations are selectively framed to avoid the most damaging admissions. The paper should be honest about zero significant results.

## What Works

1. **Clear mechanistic explanation** (Section 3.1-3.2): The LCA-SAE structural correspondence proof and four-step competitive suppression mechanism are intuitive and well-structured.

2. **H6/H8 falsification honesty** (Section 4.3-4.5): The paper explicitly states H6 precision@20=0.0 and H8 r=+0.12, p=0.55 are falsified/unsupported.

3. **Table 5 integration**: Prior findings explained by the competitive suppression framework — this is the paper's strongest contribution.

4. **Writing quality**: Prose is clear, citations are appropriate, structure is logical.

## Recommendations

1. **Retitle** to emphasize LCA-theoretical framework, not graph predictions. E.g., "Feature Absorption as Competitive Suppression: An LCA-Theoretic Framework for SAE Interpretation"

2. **Revise abstract** to lead with structural correspondence (G=W_dec^T W_dec), not graph predictions. Explicitly state: "The primary predictive hypothesis (graph edges predict absorption pairs) is falsified; the LCA-theoretical framework provides the contribution."

3. **Add MCP context** to every p-value citation. Example: "$r = -0.431$, p=0.028 uncorrected; Bonferroni-corrected p=0.334; BH-FDR q=0.107"

4. **Remove graph as diagnostic** — H6 precision@20=0.0 and H8 r=+0.12, p=0.55 show the graph does not work as a diagnostic. Do not claim it does in Section 6.3.

5. **Remove post-hoc power analysis** — this is methodologically questionable. Instead, honestly acknowledge: "n=26 features provides insufficient power to detect medium effects; null results should be interpreted as inconclusive, not as evidence of no effect."

6. **Remove H9** (co-occurrence tautology) from all tables and discussions. It is a mathematical identity, not an empirical finding.

7. **Investigate MCC pipeline** before claiming any matching-based comparisons. If Random SAEs yield MCC~0.21 (chance level), the matching procedure itself is invalid.

8. **Match L0 in OrtSAE comparison** or explicitly acknowledge the confound invalidates the ablation conclusion.

9. **Remove Feature U generalization** — it is n=1 evidence. Report it as a single case, not evidence for general absorption benignity.

10. **Control for probe quality** in CMI-absorption analysis or restrict to quality-gated letters (F1>0.85).

## Summary

The paper has two distinct contributions that are being conflated:
1. **LCA-theoretical framework** (valid): G=W_dec^T W_dec is the LCA inhibition matrix; competitive suppression explains precision-recall asymmetry
2. **Local inhibition graph as predictive tool** (falsified): precision@20=0.0, H8 r=+0.12, p=0.55

The writing must choose: either drop the graph predictions and reframe around the theoretical contribution, or honestly report that both predictive claims failed and the paper is a null-result study with theoretical discussion.