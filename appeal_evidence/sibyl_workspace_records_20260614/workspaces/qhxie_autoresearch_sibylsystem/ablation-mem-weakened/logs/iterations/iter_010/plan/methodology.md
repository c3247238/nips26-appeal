# Methodology for Iteration 10: Paper Revision

## Context

This iteration addresses the review feedback from the iteration 9 paper draft (score: 6/10). All experiments are complete. The front-runner remains **cand_g** ("Feature Absorption as Optimal Compression").

## Review Feedback Summary

### Critical Issues (Must Fix)
1. **Title-Content Mismatch**: Title emphasizes "Local Inhibition Graph" but H6 (graph predictions) was falsified (precision@20=0.0)
2. **Abstract Overstates Predictive Success**: Abstract leads with graph predictions but they failed
3. **Missing Data Source for H8 Claim**: Section 4.5 claims "r=+0.12, p=0.55" but no data file contains this

### Major Issues
4. **Figure Filename Mismatch**: Text references `fig7_precision_recall.pdf` but calls it "Figure 2"
5. **Contradictory Claim**: Section 6.3 says graph identifies at-risk features, but H8 found no correlation
6. **Missing Figure Files**: LCA-SAE correspondence diagram, precision-recall plot, competitive suppression illustration
7. **Table 3 Values Unverified**: clustering coefficient, std edge weight not from verifiable source

### Minor Issues
8. **H6 "Chance" Baseline Ambiguity**: chance_precision=0.1538 vs expected ~0.004 (20/24000) needs clarification

## Completed Experiments (Source of Truth)

| Experiment | Status | Key Result |
|---|---|---|
| Absorption detection (L0, L4, L8, L10) | Completed | Mean 2.1-3.9%, max 24.2% |
| Feature steering (L4, L8) | Completed | No significant correlation with absorption |
| Sparse probing (L4, L8) | Completed | No significant correlation with absorption |
| EC50 analysis (L4, L8) | Completed | No significant correlation with absorption |
| Precision-recall decomposition | Completed | H5: precision=1.0, recall varies |
| Decoder correlation graph (H6) | Completed | FALSIFIED: precision@20=0.0 |
| Random SAE baseline (H10) | Completed | trained=0.034, random=0.278, p<0.001 |
| H9 co-occurrence | Completed | TAUTOLOGICAL; excluded from paper |

## Data Files Reference

- `exp/results/full/correlation_report_full.json` - H1-H4 correlation results
- `exp/results/full/precision_recall_analysis.json` - H5 precision-recall data
- `exp/results/full/h6_inhibition_graph.json` - H6 graph falsification data
- `exp/results/full/ablation_random_baseline.json` - H10 random baseline data
- `exp/results/full/paper_summary_stats.json` - Summary statistics

## Planned Revisions

### Title Change
**Old**: "The Local Inhibition Graph" or similar graph-emphasis
**New**: "Feature Absorption as Optimal Compression: Evidence that Training Reduces Structural Artifacts"

### Abstract Rewrite
Lead with:
1. Absorption is real but does NOT degrade downstream tasks (null results)
2. Trained SAEs have 8x lower absorption than random baselines
3. Reframe as rate-distortion optimal behavior

### Section Restructuring
1. Move LCA-SAE correspondence to background (not as predictive tool)
2. Be explicit: graph predictions were falsified (precision@20=0.0)
3. Emphasize H5 (precision invariant, recall variable) as robust finding
4. Include H10 (trained < random) as key empirical contribution

### Data Verification
- Verify or remove H8 claim about total incoming inhibition
- Clarify chance baseline definition in H6
- Add data source citations for all statistics

## Expected Visualizations

From existing results:
1. **Figure 1**: Architecture diagram (LCA-SAE correspondence) - if needed
2. **Table 1**: Hypothesis testing summary with corrected p-values
3. **Figure 2**: Precision-recall asymmetry scatter
4. **Table 2**: Feature-level absorption and downstream data
5. **Figure 3**: Random vs trained SAE absorption comparison
6. **Figure 4**: Steering success vs absorption rate

## Output
- `writing/sections/` - Updated section files
- `writing/paper.md` - Revised complete paper
- `writing/latex/main.tex` - Final LaTeX submission
