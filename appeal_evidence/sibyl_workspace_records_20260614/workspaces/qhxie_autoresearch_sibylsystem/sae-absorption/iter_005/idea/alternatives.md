# Backup Research Ideas for Pivot

## Alternative 1: Feature Immunodominance -- Predicting Absorption Hierarchies from Ecological Competition Theory

### Summary

If the confound resolution (Contribution 1) shows absorption is an epiphenomenon of width/L0 and the cross-domain experiments yield near-zero absorption rates, the paper pivots to a purely theoretical contribution. This alternative formalizes feature absorption as a competitive exclusion phenomenon using Lotka-Volterra dynamics and immunodominance theory from ecology/immunology, producing a quantitative model that predicts WHICH features absorb which from measurable SAE properties.

### Core Idea

The structural isomorphism between immune repertoire dynamics and SAE feature dynamics is deeper than a metaphor: the SAE's sparsity-penalized reconstruction objective creates exactly the resource-mediated competition that Lotka-Volterra models describe. The key novel predictions are:

1. **R* rule**: The feature that can persist at the lowest activation magnitude (minimum reconstruction error per L0 unit) wins in each competing pair. For parent-child pairs, the child has lower R* because it explains more specific, lower-variance signal.

2. **Active suppression signature**: If absorption involves active encoder suppression (not just passive neglect), absorbed parent features should show learned negative correlation between their encoder direction and the absorbing child's encoder direction.

3. **Epitope focusing prediction**: Disrupting parent-child co-occurrence during training (masked regularization) works because it is the SAE analogue of epitope focusing in vaccinology -- removing the immunodominant stimulus allows the suppressed response to recover.

### Why This Is a Viable Pivot

- Requires zero new experiments: all analysis operates on existing pre-trained SAEs
- The R* rule and active suppression signature are testable in ~2-3 GPU-hours
- Even if the main proposal succeeds, these predictions can be included as supplementary analyses
- Novelty is strong: searched arXiv for "immunodominance" AND "sparse autoencoder" -- zero results

### Risk

The analogy may be "decorative rather than load-bearing" (interdisciplinary self-critique). If the R* rule achieves only marginally better prediction than simpler baselines (frequency alone or cosine similarity alone), the ecological framing adds vocabulary without explanatory power. Mitigation: require R* rule AUROC to exceed single-variable baselines by at least 0.05 before claiming the multi-variable ecological model adds value.


## Alternative 2: Absorption Phase Diagram with Formal Scaling Laws

### Summary

If the cross-domain experiments fail (probes too noisy, absorption near zero on knowledge tasks) but the confound analysis succeeds (H1 survives), the paper pivots to making the absorption scaling surface the primary contribution, enriched with theoretical scaffolding from compressed sensing phase transition theory.

### Core Idea

Construct a formal phase diagram in (W/|F|, L0) space with three empirically identifiable regimes:

1. **Hedging regime** (W too small): Features are merged; absorption is not separately measurable. Identifiable via high inter-feature correlation.

2. **Absorption regime** (W sufficient, L0 insufficient): Features are distinct but parent features are systematically excluded by the sparsity budget. Absorption rate follows alpha ~ max(0, 1 - L0/(d_eff + 1)).

3. **Recovery regime** (W and L0 both sufficient): Absorption approaches zero. All features recovered.

The phase boundaries are predicted from the Donoho-Tanner compressed sensing analogy: the absorption phase diagram maps the boundary in (delta=W/d_model, rho=L0/W) space where absorption transitions from avoidable to inevitable.

### Why This Is a Viable Pivot

- Uses SAEBench precomputed data for 200+ SAEs (zero GPU cost)
- The three-regime framework is novel: no paper unifies hedging, absorption, and recovery into a single phase diagram
- Produces immediately actionable guidance: "for features at hierarchy depth d, use L0 >= d+1 and W >= W_recover(d)"
- The scaling law predictions are testable on existing data

### Risk

The phase boundaries may not be sharp (smooth crossover rather than phase transition). In this case, the "phase diagram" language is overpromising; fallback to "absorption scaling law" framing. Additionally, the theoretical predictions (L0 budget argument) may not match the empirical surface, requiring the paper to be purely empirical rather than theory-driven.


## Alternative 3: The Honest Audit -- Absorption Is Less Severe, Less General, and Less Causal Than Believed

### Summary

If BOTH the confound analysis kills H1 (absorption is not a causal quality mediator) AND the cross-domain experiments show absorption < 5% on knowledge tasks, the paper pivots to a comprehensive "empirical audit" of absorption claims.

### Core Idea

The paper becomes a systematic challenge to the absorption literature, structured around three deflationary findings:

1. **The width/L0 confound explains the absorption-quality correlation**: Absorption is an epiphenomenon of L0, not a causal quality driver. The practical recommendation is to use higher L0, not to develop absorption-specific mitigations.

2. **Absorption is specific to deterministic hierarchies**: The first-letter task's crisp parent-child structure maximizes the sparsity incentive for absorption. Real-world knowledge hierarchies are probabilistic, and the overlap provides insufficient sparsity savings to drive absorption.

3. **The 92.3% taxonomy rate is an artifact**: Proper comparison tokens reduce the combined absorption rate by 30-50%, and the corrected rate with honest uncertainty bounds is substantially lower.

### Why This Is a Viable Pivot

- Negative results in this area are highly publishable because the community is making investment decisions based on absorption rates (DeepMind deprioritized SAEs partly due to absorption)
- An honest audit that recharacterizes a well-known problem is more valuable than overclaiming
- All analyses are already planned in the main proposal; the difference is framing
- The contrarian perspective explicitly designs this pivot

### Risk

A purely deflationary paper may be perceived as "tearing down without building up." Mitigation: include constructive recommendations (use correct L0, measure on task-specific hierarchies, report confound-controlled correlations) and position as "clearing the empirical ground for productive absorption research."


## Alternative 4: Dictionary Coverage as the Dominant Absorption Mechanism -- Multi-Model Replication

### Summary

If Gemma model access remains blocked for cross-domain experiments but GPT-2 and Pythia are available, pivot to the empiricist's dictionary coverage hypothesis as the primary contribution, replicated across multiple model families.

### Core Idea

The iter_001 finding that ~75% of absorption is "early-type" (decoder-absent, meaning the SAE dictionary never allocated a latent for the parent feature) is the most practically impactful observation from the project. If confirmed across model families, it redirects the field's mitigation strategy from encoder fixes (OrtSAE, ITAC) to dictionary coverage solutions (wider SAEs, Matryoshka).

This alternative focuses on:
1. Multi-model replication: GPT-2 Small (L6, L8, L10), Pythia-160M, Pythia-410M
2. Multi-threshold robustness: tau = {0.15, 0.20, ..., 0.45} + k-nearest-decoder alternative
3. Width scaling: does early-type fraction decrease with wider dictionaries?
4. Dead feature exclusion with separate reporting

### Why This Is a Viable Pivot

- No Gemma model access needed (GPT-2 and Pythia are fully public)
- The iter_001 n=2 finding is elevated to n >= 10 configurations across >= 3 model families
- Direct practical implication: if early absorption dominates, the prescription is "use wider SAEs" rather than "use fancier encoder architectures"
- Addresses the iter_3 lesson: "taxonomy needs replication on a third model family"

### Risk

The tau threshold sensitivity is the critical weakness (iter_001: 32% at tau=0.2, 95% at tau=0.4). If the early-type fraction depends entirely on threshold choice and no principled threshold exists, the finding is unstable. Mitigation: emphasize the sensitivity curve shape rather than a single number, and use the k-nearest-decoder alternative for threshold-free validation.
