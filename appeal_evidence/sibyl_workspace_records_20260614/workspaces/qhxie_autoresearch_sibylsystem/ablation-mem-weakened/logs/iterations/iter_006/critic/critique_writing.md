# Critique: Writing Quality

## Major Issues

### 1. Abstract Misrepresents Statistical Status
The abstract claims evidence of effects throughout while presenting correlation results with p=0.028, but:
- 12 statistical tests were performed
- After Bonferroni correction (alpha=0.00417), ZERO results are significant
- After BH-FDR correction, ZERO results are significant
- The paper's framing throughout suggests significant findings exist when they do not

**Fix required**: The abstract must clearly state "After multiple comparison correction, no test reached statistical significance" rather than presenting uncorrected p-values as evidence.

### 2. Overclaiming Contributions
The paper uses "contribution" language for null results:
- "honest null-result reporting" - this is not a contribution, it is responsible practice
- "metric validation insight" - the validation shows the metric may be invalid, which is concerning not validating
- "methodological framework" - baseline correction and precision-recall are standard practice

**Fix required**: Use neutral language. Report what was found, not how it should be framed.

### 3. Section 5.2: Post-Hoc Justification
"Why Prior Work Found Null Results" reads as excuse-making:
- "Low absorption variance constrains statistical power" - possible but unproven
- "Raw metrics confounded by generic directional bias" - speculation
- "Underpowered study" - after-the-fact acknowledgment

**Fix required**: Present these as hypotheses requiring testing, not established explanations.

### 4. Vague Novelty Claims
The introduction claims multiple contributions:
- First connection between LCA and SAEs - the structural correspondence (G=W^T W) is from 2008
- First local inhibition graph - but it does not work (H6 fails)
- Mechanistic explanation - but only in a vague sense; the graph fails to predict anything

**Fix required**: Be specific about what is actually novel. The LCA connection is theoretically interesting but empirically unsupported.

### 5. Missing Caveats in Abstract
The abstract presents trained<random absorption finding (p<0.001) without caveats:
- MCC~0.21 for random SAE suggests metric validity issues
- The 8x difference may reflect metric sensitivity to dictionary structure, not learning
- Without partial correlation controlling for dictionary geometry, interpretation is unclear

**Fix required**: Add caveat about metric validity limitations.

## Minor Issues

### 6. Figure Quality
Three figures are referenced:
- `fig1_lca_correspondence.pdf` - not actually included
- `fig2_suppression_mechanism.pdf` - not actually included
- `fig7_precision_recall.pdf` - not actually included

Actual figures appear to be placeholder references.

### 7. Redundant Discussion
Section 5.3 and 5.4 repeat content from Section 4.8 (Table 5 integration). Consider consolidating.

### 8. Conclusion Overreach
"Absorption is not a critical failure mode for SAE interpretability" is too strong given:
- Only tested on 26 first-letter features in GPT-2 Small
- Only tested steering and probing downstream tasks
- The 89-99% dead feature ratio may invalidate conclusions for active features

## Summary

The writing has a systematic positivity bias that contradicts the paper's own null results. The paper found nothing statistically significant, but presents this as "honest null-result reporting" as if finding nothing is a contribution. The writing should be reframed to:
1. State clearly: 12 tests, 0 significant after correction
2. Report descriptive statistics without significance framing
3. Acknowledge limitations and alternative interpretations prominently
4. Remove "contribution" language for standard practices