# Ideation Critique

## Proposal-Paper Scope Gap (Critical)

The proposal (iter_009) promises a paper grounded in "Lyapunov stability theory and Pontryagin's Maximum Principle" with 6 contributions:
1. Phi-Modulator Taxonomy with steady-state formula (Theorem 1)
2. Lyapunov Variance Decomposition (Theorem 2)
3. Budget Equivalence Theorem (Theorem 3)
4. PMP Optimal Control Derivation (Theorem 4)
5. Controller Taxonomy (R_t as plant output)
6. AIS threshold characterization

The delivered paper contains ZERO of these theoretical contributions. It delivers:
1. Phi modulator taxonomy (without steady-state formula)
2. Three diagnostic metrics (BEM, CSI, AIS -- no theorems)
3. Empirical null result (Phi Invariance observation)
4. SGD boundary condition

This is a fundamentally different paper. The pivot to an empirical-only paper was correct (removing Lyapunov/PMP was praised in lessons_learned), but the proposal has not been updated. Future iterations will continue to penalize this gap.

## Novelty Assessment

The paper's novelty rests on three pillars:

### 1. Phi Modulator Framework (Moderate Novelty)
Organizational contribution, not mathematical. Defining phi(t, theta, g) and showing methods are special cases is a taxonomy exercise. The composition closure property (Proposition 1) is the only formal result. However, the framework *does* enable the systematic comparison that produces the empirical finding, so it has instrumental value.

### 2. Phi Invariance Observation (Genuine Novelty, Limited Scope)
The finding that all 7 methods (including no_wd) are equivalent under AdamW is genuinely surprising and useful. The CWD/random-mask comparison is the paper's sharpest insight: if alignment-conditioned masking (CWD) performs identically to random masking, the alignment signal provides no value for WD decisions under AdamW. However, this is limited to CIFAR-scale with BN architectures.

### 3. Diagnostic Metrics (Mixed Novelty)
- BEM: Simple and interpretable. Not mathematically novel but fills a genuine gap.
- AIS: Standard Spearman correlation applied to standard quantities. Overclaimed as a "standardized diagnostic metric."
- CSI: Arbitrary weights, zero predictive value, combines incompatible quantities. Not a contribution.

## Differentiation from D'Angelo et al. (2024)
D'Angelo showed WD is "never useful as regularization." The current paper extends this: even when WD acts as a dynamics modifier (the D'Angelo finding), the specific modulation strategy is irrelevant under AdamW. This distinction is genuine but underarticulated in the paper.

## Alternative Framing
The paper could be more impactful if framed as: "A Systematic Empirical Test of Dynamic Weight Decay Methods: The Case Against Modulation Under AdamW." This drops the "conjecture" overclaiming and positions it as a definitive negative-result benchmark paper.
