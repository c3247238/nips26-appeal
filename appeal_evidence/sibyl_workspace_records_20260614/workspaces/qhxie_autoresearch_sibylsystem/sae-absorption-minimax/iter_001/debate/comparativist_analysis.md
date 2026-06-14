# Comparativist Analysis: Contextualizing with Prior Work

## Overview
Comparison with related work in SAE interpretability and feature steering.

## Anthropic's Monosemanticity Work

### Original Absorption Definition
- Absorption: child feature activations subsume parent feature activations
- Concern: Absorbed features become less interpretable and less reliable
- **Our finding challenges this**: absorbed features are MORE steerable

### Implications for Anthropic's Framework
- H3 REVERSED suggests absorption ≠ unreliability
- Instead: absorption may indicate feature importance
- High-absorption features are "load-bearing" for model behavior

### Related: Feature Geometry
- Anthropic found features cluster in geometric structures
- Our results: high-absorption features have distinct geometry (higher UAS)
- This supports the geometric interpretation

## Steering Literature

### Subramani et al. (2022) - Steering Vectors
- Found: Adding concept directions to residual stream steers model behavior
- Our contribution: Characterize which features make effective steering targets
- UAS provides a priori selection criterion

### Dream却说 et al. - Intervention Strategies
- Ablation vs steering tradeoffs
- Our finding: High-absorption features respond well to steering
- Suggests steering > ablation for absorbed features

### Zou et al. - Representation Engineering
- Concept vectors for steering
- Our work: How to identify effective concept directions from SAEs
- UAS predicts steering effectiveness

## SAE Architecture Comparison

### Standard vs TopK vs OrtSAE
- TopK showed 70.9% reduction in absorption effects
- OrtSAE showed 8x MSE improvement
- **Implication**: Different architectures trade absorption for other properties
- Future work: Does OrtSAE's lower absorption reduce steering effectiveness?

## Field Expectations

### Expected: Absorption Degrades Reliability
- Traditional view: absorbed features are noisy
- **Our finding**: Counter-intuitive positive correlation
- Requires explanation and replication

### Expected: Layer-wise Ordering
- Deeper layers = more abstract features
- **Our finding**: Contradictory L4 vs L8 results
- Suggests absorption is layer-dependent in complex ways

## Recommendations for Paper

1. **Cite Anthropic's foundational work** on absorption
2. **Position H3 REVERSED as challenging** conventional understanding
3. **Connect to steering literature** as practical application
4. **Acknowledge limitations** relative to prior work

## Conclusion
H3 REVERSED is interesting precisely because it contradicts prior expectations. The paper should engage seriously with why absorbed features are more steerable.
