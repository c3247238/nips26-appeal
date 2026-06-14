# Iteration 2 Proposal: Cross-Model Validation of Unsupervised Absorption Detection (UAD) with Dead Feature Analysis

## Problem Statement

Iteration 1 established two key findings:
1. **UAD (Unsupervised Absorption Detection)** achieves F1=0.704 with perfect recall on GPT-2 Small, demonstrating that co-occurrence clustering can detect absorbed feature pairs without supervised probe directions.
2. **DFDA (Dynamic Feature De-Absorption)** achieves 11.1% average MSE improvement on absorbed pairs using only 388 parameters.

However, Iteration 1 also revealed a critical confound: all trained SAEs exhibited 89-99% dead features, and the "collision rate" metric used in early CAAB experiments is a proxy, not true absorption. The 4x collision rate difference between pretrained JumpReLU (15.4%) and trained TopK (3.8%) may be entirely explained by dead feature ratios rather than architecture differences.

The 6 perspectives in Iteration 2 debate converge on a single question: **How do dead features confound absorption measurements, and what does this mean for the UAD+DFDA pipeline?**

## Research Question

Does the UAD+DFDA pipeline remain valid when accounting for dead feature confounds, and can it generalize to pretrained SAEs with healthier dictionaries?

## Hypotheses

- **H1 (Dead Feature Confound)**: High dead feature ratios (>90%) systematically distort co-occurrence clustering by reducing the effective dictionary size, but UAD's recall remains robust because absorption signatures persist even in sparse dictionaries.
- **H2 (Cross-Architecture Convergence)**: When comparing pretrained SAEs with similar dead feature ratios, collision rates converge across architectures (JumpReLU vs TopK), confirming that the Iteration 1 architecture difference was a dead-feature artifact.
- **H3 (UAD Generalization)**: UAD achieves F1 >= 0.55 on pretrained SAEs with lower dead feature ratios (e.g., GemmaScope), confirming that the method generalizes beyond the high-dead-feature regime of Iteration 1.
- **H4 (DFDA on Pretrained)**: DFDA achieves >=10% MSE improvement on absorbed pairs identified in pretrained SAEs, confirming that compensation works on healthier dictionaries.

## Methods

1. **Analyze dead feature impact on UAD**: Quantify how dead feature ratio affects UAD precision/recall by comparing results across SAEs with varying dead feature ratios.
2. **Cross-validation on pretrained SAEs**: Run UAD on pretrained SAEs with known low dead feature ratios (GemmaScope JumpReLU, GPT-2 Small residual SAEs from SAELens).
3. **Dead feature ablation**: Systematically filter out dead features before co-occurrence clustering and measure the impact on detection quality.
4. **DFDA on pretrained pairs**: Apply DFDA to absorbed pairs identified in pretrained SAEs and measure compensation efficacy.
5. **Direct comparison**: Plot UAD performance vs dead feature ratio to test whether detection quality degrades with dictionary health.

## Expected Contributions

1. **Quantification of dead feature confound on absorption metrics**: First systematic analysis of how dictionary health affects absorption detection.
2. **Validation of UAD on healthier dictionaries**: Confirm UAD works beyond the high-dead-feature regime.
3. **Unified framework**: Connect dead feature analysis (training quality) to absorption detection (interpretability quality).
4. **Practical recommendation**: Guidelines for minimum dictionary health thresholds when studying absorption.

## Risk Assessment

- **Low**: Pretrained SAEs are readily available in SAELens; no training needed.
- **Medium**: If UAD precision drops significantly on healthier dictionaries, the false positive rate may be driven by something other than dead features.
- **Mitigation**: Compare against Chanin labels where available; if UAD fails, the backup is to pivot to absorption-as-quality-diagnostic (Alternative A from Iteration 1).

## Time Budget

- UAD on pretrained SAEs (2-3 models): 45 min
- Dead feature ablation analysis: 20 min
- DFDA on pretrained pairs: 20 min
- Analysis + figures: 25 min
- Total: ~2 hours

## What Changed from Iteration 1

1. **Shifted from training SAEs to analyzing pretrained SAEs**: Respects the project spec's "training-free analysis为主" constraint. No AuxK training needed.
2. **Reframed dead features as confound rather than problem to fix**: Instead of trying to eliminate dead features (which requires retraining), we quantify their impact and validate UAD on dictionaries that are already healthy.
3. **Maintained UAD+DFDA as core contribution**: The 6 perspectives debated whether to abandon UAD for CAAB-v2; the synthesis retains UAD as primary while incorporating dead feature insights.
4. **Dropped the "living dictionary CAAB" as primary**: The perspectives' focus on training new SAEs with AuxK is feasible but violates project constraints and distracts from the already-validated UAD contribution.

## Revisions from Prior Feedback

- **Contrarian's concern**: "Fixing dead features may not fix the core problem" — addressed by not trying to fix dead features but instead validating UAD on pretrained SAEs with naturally low dead feature ratios.
- **Theoretical's H2 critique**: "Convergence may be too strong" — acknowledged; we test H2a (convergence direction) rather than claiming full convergence.
- **Empiricist's concern**: "Both had >94% dead features" — directly addressed by testing on pretrained SAEs with different dead feature profiles.
- **Interdisciplinary's species-area hypothesis**: Retained as exploratory; smaller effective dictionary size (due to dead features) may indeed increase collision rates.
