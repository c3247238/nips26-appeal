# Lessons from Iteration 4

## Must Improve

- **[CRITICAL] Add methodology evolution paragraph**: The empirical pivot from r=+0.35 to p=0.299 is correct but must be documented. State explicitly: original analysis had confounds (activation frequency, decoder L2 norm) that matched design controlled. After control, the positive correlation disappears — this means r=+0.35 was a confound artifact. Estimated effort: 30 min.

- **[CRITICAL] Apply Bonferroni correction and fix beta=20 framing**: Report corrected p-values for 5 beta-value comparisons. Update title/abstract: 'Across most steering magnitudes... with a reversal at high steering magnitudes (beta=20, p=0.015 uncorrected, p=0.075 Bonferroni-corrected).' The beta-conditional effect is the paper's most novel finding — it should be highlighted, not hidden. Estimated effort: 15 min.

- **[CRITICAL] Demote entanglement hypothesis**: The hypothesis was designed for r=+0.35 which is no longer the main finding. Either add mechanistic experiments (activation patching, path tracing) or demote to 2-3 sentences discussing saturation confound for beta=20. Do not retain as a named section. Estimated effort: 30 min (demote) or 2-3 hours (experiments).

- **[MAJOR] Delete paper/paper.md and designate LaTeX canonical**: Two contradictory versions persist. writing/latex/main.tex must be the only canonical version. Estimated effort: 30 min.

- **[MAJOR] Generate 2 key figures before next writing iteration**: Scatter plot of UAS vs steering sensitivity + grouped bar chart by beta value. Tables cannot convey the distribution of 200 features. Desk-rejection risk at NeurIPS/ICLR without figures. Estimated effort: 2-3 hours.

## Watch Out

- **Beta=20 p=0.015 is NOT significant after Bonferroni correction** (threshold p<0.01). The paper currently claims a significant finding that loses significance under appropriate correction.

- **Stagnation pattern is broken but ceiling remains**: Score improved 6.0->6.5 through honest empirical pivot. The next 0.5 points require fixing the 3 new critical issues — not more prose polish.

- **Entanglement hypothesis is doubly orphaned**: No positive finding to explain (r=+0.35 abandoned) AND no mechanism evidence. This is pure speculation presented as a named contribution — top venue reviewers will flag it.

- **Saturation confound is the most plausible explanation for beta=20 reversal**: High-absorption features have higher decoder L2 norms by construction. At high steering magnitudes, they saturate faster. This is worth discussing explicitly rather than leaving readers to infer.

## Keep Doing (success patterns)

- **Honest empirical pivot**: Changing the central claim from r=+0.35 to "no significant difference" is exemplary scientific integrity. This is the paper's credibility cornerstone and should be maintained.

- **UAS validation E2 meta-analysis**: N=314 with r=0.65-0.79 remains the paper's strongest empirical signal. This is genuinely novel and worth publishing.

- **Null controls with full beta range**: Now correctly matches main H3 protocol. Critical methodological flaw from prior iterations is resolved.

- **Honest negative results reporting (H2, H4, H6, H7)**: Exemplary across ALL iterations. This is the paper's strongest credibility signal and must be maintained.

- **Claim-evidence integrity**: All reported r, p-values, means match source data. This discipline must be maintained.
