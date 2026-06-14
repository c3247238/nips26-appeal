# Glossary

This document defines unified terminology for the paper.

## Key Terms

### Feature Absorption
Feature absorption occurs when child features in a hierarchical feature organization subsume parent features, causing the parent feature to rarely activate. Also known as "feature splitting" or "parent-child absorption." Detected using the Chanin protocol (Chanin et al., 2024).

### Feature Sensitivity
Feature sensitivity measures how reliably a feature activates across semantically equivalent inputs. A feature with low sensitivity may activate strongly on one sentence but fail to activate on a paraphrase with the same meaning. Measured using the Tian protocol (Tian et al., 2025).

### Compound Failure Mode
A scenario where multiple failure modes affect the same feature simultaneously. For example, a doubly-compromised feature is both absorbed (parent subsumed by child) AND low-sensitivity (unreliable across contexts).

### Sanity Check
The Sanity Check finding (Korznikov et al., 2026) demonstrates that random SAE baselines match trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72). This suggests that SAE features may not reliably decompose model internals.

### Steering Intervention
A causal intervention where $\beta \times W_{\text{dec}}[f]$ is added to the residual stream to modify model behavior. The steering effect is measured by the max absolute logit difference at the last position.

### Decoder L2 Norm
The L2 norm of the decoder weight vector for a feature: $\|W_{\text{dec}}[f]\|_2$. Features with larger decoder norms produce larger steering effects at the same beta value.

## Abbreviations

| Abbreviation | Full Form |
|--------------|-----------|
| SAE | Sparse Autoencoder |
| UAS | Unsupervised Absorption Score |
| Q1-Q4 | Quadrant 1-4 classification |
| CI | Confidence Interval |
| SE | Standard Error |

## Paper Conventions

1. Use "absorption" not "feature absorption" after first mention
2. Use "sensitivity" not "feature sensitivity" after first mention
3. Use "compound failure" to describe the joint occurrence of multiple failure modes
4. "Doubly-compromised" refers specifically to Q1 features (both absorbed and low-sensitivity)
5. "Steering effect" refers to the logit change, not the activation change