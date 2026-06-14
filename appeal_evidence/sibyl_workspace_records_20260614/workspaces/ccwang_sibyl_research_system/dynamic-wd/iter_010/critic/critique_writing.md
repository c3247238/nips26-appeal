# Writing Critique

## Title
The title claims causation ("Why") that the paper does not deliver. The paper shows THAT methods are equivalent, not WHY. The mechanistic hypothesis is correlational, not causal. Change to "When" or remove the causal framing.

## Abstract
1. **Misleading factorial description**: "7 methods, 2 optimizers, 2 architectures, 2 datasets, 3 seeds" implies a fully crossed 168-run design. The actual design is partial: 84 fully crossed ResNet-20 runs + 21 targeted VGG-16-BN runs. This must be disambiguated.
2. **Missing power qualification**: "statistically equivalent" without stating the detection threshold (~0.7%) is absence of evidence presented as evidence of absence.
3. **"including the degenerate case of no weight decay at all"**: This is the paper's strongest empirical point and should be emphasized more prominently.

## Section 5.1
The spread annotation "0.25pp" is correct and consistent with Table 2 data. This was a previously flagged triple-inconsistency (0.49pp/0.25pp/0.25pp) that appears to be fixed in the text, but Figure 3(a) annotation says "Spread: 0.25pp" -- confirmed consistent.

## Section 5.7
Drawing a directional conclusion from |Delta rho| = 0.424 between two non-significant correlations is statistically improper. The honest conclusion: "neither measure predicts generalization gap in our sample." If the directional comparison matters, a formal Steiger's z-test is required.

## Section 6.2 (Non-BN ablation)
The preliminary non-BN results (spread = 0.11pp with 2/7 methods) actually contradict the BN-mechanism hypothesis, since the spread is narrower WITHOUT BN than with BN under AdamW (0.25pp). The paper dismisses this with "limiting generalizability" instead of addressing the contradiction. Either test all 7 methods or remove the contradictory preliminary results.

## Figure-Text Consistency
- Figure 8 legend includes "PMP-WD" which is not a method described in the paper. Desk-reject level.
- Figure 4: half_lambda BEM position appears incorrect (at ~0.0 instead of 0.500). Persists from iter_007.
- Figure file naming: theorem2_validation.png references a nonexistent Theorem 2.

## Overall Writing Quality
The prose is clean, well-structured, and above average for the field. Statistical methodology is rigorous and well-presented. The limitations section is honest and comprehensive (6 items). The paper reads as a mature, well-edited piece -- the problems are at the data/figure/claims level, not the prose level.
