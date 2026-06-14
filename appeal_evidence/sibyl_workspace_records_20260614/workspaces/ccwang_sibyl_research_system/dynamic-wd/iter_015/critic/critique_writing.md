# Writing Critique

## Executive Summary

The paper is well-structured and unusually honest about negative results. The writing quality is above average for ML submissions. However, several critical integrity and scope issues undermine the presentation.

## Strengths

1. **Negative result transparency**: The paper explicitly reports UDWDC underperforming NoWD, states "engineering patches, not principled solutions," and documents CSI_combined = -2.41. This honesty is rare and commendable.

2. **Clear narrative arc**: The paper flows logically from fragmentation observation (Section 1) to unification attempt (Section 3) to experimental validation with honest limitations (Sections 5-6).

3. **Statistical rigor**: Paired t-tests with Bonferroni correction, Cohen's d, TOST equivalence testing, and explicit power analysis exceed community norms.

4. **Self-contained methodology**: Each metric (BEM, CSI, AIS) is defined with an equation, motivated with a use case, and evaluated with data.

## Critical Issues

### 1. Title-Content Mismatch

"Unified Feedback Control Framework" implies comprehensive unification. Reality: 2/5 methods fail the 20% fitting threshold. The framework unifies alignment-based and constraint-based families only. The title creates expectations the paper cannot meet. Suggested alternatives:
- "A PID-Style Taxonomy for Dynamic Weight Decay"
- "Gradient-to-Weight Ratio Control: Diagnostic Framework for Comparing Dynamic WD Methods"

### 2. Contribution List Overclaims

- **Contribution 2** presents UDWDC as a "simple proportional controller" -- but it underperforms NoWD. Listing a method that harms performance as a contribution is overclaiming. Reframe as a diagnostic probe that reveals proportional-only control is insufficient.
- **Contribution 4** lists "Theoretical analysis" -- but Theorem 1 and Propositions 2-3 have no proofs. Either write proofs or downgrade to conjectures and remove from contributions.
- **Contribution 5** claims "Comprehensive experiments" -- but ImageNet has only 4/7 methods and CWD has 1 seed. "Controlled comparison with partial ImageNet coverage" is accurate.

### 3. Fabricated AIS Value

AIS = 0.566 appears in Section 4.3 and is referenced throughout the paper. The actual measured value is 0.123. This is not a rounding error; it is 4.6x the true value. This must be corrected before any other revision. The LOO-CV R^2 being negative means AIS has zero out-of-sample predictive power, which the paper does not acknowledge.

### 4. Figure 1 Is Not Data

Figure 1 uses TikZ hand-drawn curves with a caption claiming they represent experimental data from "10-epoch pilot, seed 42." A schematic is not data. Real trajectory plots exist (latex/figures/ratio_trajectories.png). Using a schematic as if it were data is a presentation integrity issue.

### 5. Notation Inconsistency

Section 3.4 introduces delta_t as alias for alpha_t and references a "notation table" that does not exist in the paper. This creates confusion, especially since delta_t and alpha_t serve different conceptual roles in Sun et al.'s framework vs. this paper.

## Minor Writing Issues

- Section 4.2: CSI definition uses three different formulations across the paper. Standardize.
- Section 5.7: Claims directional conclusions from p > 0.1 correlations, which contradicts the rigorous statistical standard established elsewhere.
- UDWDC-v2's WD budget of 98599 needs clearer context (e.g., "205,000x FixedWD due to BN layer bug").
- The paper uses both "GW ratio" and "gradient-to-weight ratio" interchangeably; pick one.
- Abstract is 250+ words; NeurIPS limit is typically 250. Check compliance.
