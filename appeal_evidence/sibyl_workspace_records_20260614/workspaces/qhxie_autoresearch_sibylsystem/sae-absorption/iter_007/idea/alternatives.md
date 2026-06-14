# Backup Ideas for Pivot

## Alternative 1: Geometric Forensics of Feature Absorption -- Unsupervised Detection via Decoder Weight Topology

**Status**: Strong backup. The highest-novelty candidate across all perspectives. Promoted to front-runner if the metric audit paper reaches diminishing returns or if a reviewer independently publishes cross-domain absorption measurements.

### Summary

Develop the first unsupervised absorption detection method using four complementary geometric signals computed from the SAE's decoder weight matrix alone, without requiring supervised probes or known feature hierarchies:

1. **Decoder Gram matrix** G_ij = cos(w_i, w_j): pairwise cosine similarity identifying feature pairs in competition (from the LCA lateral inhibition literature).
2. **Activation-Decoder Misalignment (ADM)**: comparing each feature's empirical activation PCA direction with its decoder vector. Absorbing features show misalignment because their decoder points partly toward the absorbed feature's direction.
3. **Asymmetric Co-activation Deficit (ACD)**: unexpectedly low co-activation rates for geometrically similar features -- the specific signature of absorption.
4. **Residual Absorption Score (RAS)**: G_ij * (expected_coactivation - observed_coactivation), combining geometry with activation statistics.

### Validation Plan
- Validate against Chanin et al. supervised metric on first-letter spelling (target: AUROC > 0.7).
- Deploy on knowledge hierarchies (RAVEL city-country, entity types) where supervised probes cannot easily be constructed.
- Map the "absorption landscape" across all Gemma Scope SAE widths and layers using the geometric detector as a scalable measurement tool.

### Key Strengths
- Fills the most critical gap in the field (Gap 7: no unsupervised absorption detection).
- Perfectly aligned with training-free constraint.
- Cross-domain connections (ecological competitive exclusion, neuroscience lateral inhibition) provide rich theoretical narrative.
- All experiments use only SAELens, TransformerLens, and pretrained SAE weights.

### Key Risks
- Geometric signals may be too subtle for reliable detection (main empirical risk).
- OrtSAE (2025) already uses decoder geometry for absorption *mitigation* -- need clear differentiation as *detection*.
- Validation against supervised ground truth creates chicken-and-egg dependency.

### Origin
Innovator perspective (primary), with supporting elements from Pragmatist (computational feasibility), Theoretical (LCA connection), and Interdisciplinary (ecological competitive exclusion analogy).

---

## Alternative 2: Immunological Imprinting Theory of Feature Absorption -- Cross-Reactive Absorption and Frequency-Ratio Scaling Laws

**Status**: Strong backup with unique predictions. Promoted if empirical evidence for "cross-reactive absorption" (absorption between semantically unrelated co-occurring features) is found during exploratory analysis.

### Summary

Apply the Original Antigenic Sin (OAS) framework from immunology to generate novel predictions about SAE feature absorption that the standard hierarchical model does not make:

1. **Cross-reactive absorption**: SAE features absorb co-occurring features regardless of semantic/hierarchical relationship, driven by frequency-based competitive advantage. This extends absorption beyond the current hierarchical-only model.
2. **Frequency-ratio scaling law**: P(absorption of f_j by f_i) scales with log(freq(f_i)/freq(f_j)), analogous to antigenic seniority.
3. **Training-order imprinting**: Features established early in training have disproportionate absorptive power over later-emerging features.
4. **Masked regularization as "conserved epitope priming"**: Disrupting co-occurrence during training reduces absorption by the same mechanism that heterologous prime-boost vaccination reduces OAS.

### Diagnostic Experiment
Construct controlled feature pairs:
- **Hierarchical pairs**: parent-child (standard absorption, e.g., first-letter)
- **Cross-reactive pairs**: semantically unrelated but co-occurring (e.g., common determiners absorbing rare nouns)
- **Control pairs**: matched frequency ratio but no co-occurrence

If cross-reactive absorption exists at rates predicted by frequency ratio (independent of decoder similarity and hierarchical relationship), the OAS model explains a phenomenon the standard model cannot.

### Key Strengths
- Generates a genuinely novel prediction (cross-reactive absorption) that no other perspective produces.
- No prior work connects OAS to SAE/dictionary learning (verified via systematic search).
- Provides a unified framework for multiple SAE failure modes: absorption (=OAS), hedging (=insufficient repertoire diversity), dead features (=clonal deletion), inconsistency (=private repertoire variation).
- Suggests novel mitigation strategies inspired by vaccine design.

### Key Risks
- Cross-reactive absorption may not exist or be very rare. If only hierarchical pairs absorb, the analogy is decorative.
- The frequency-ratio confound with decoder similarity is the main methodological challenge.
- Training-order effects may be washed out by convergence (would need checkpoint analysis).

### Origin
Interdisciplinary perspective (primary), with elements from Contrarian (frequency-driven rather than hierarchy-driven absorption).

---

## Alternative 3: Quantitative Theory of Feature Absorption -- Coherence, Frequency, and the Absorption Phase Boundary

**Status**: Moderate backup. Valuable as a standalone theoretical contribution if empirical predictions are validated by the L0=22 CMI experiment.

### Summary

Develop the first quantitative theory predicting absorption probability as a function of measurable SAE and feature properties:

**Main Proposition**: For a parent-child feature pair with hierarchical coherence mu_H = |<d_parent, d_child>|, relative frequency rho = freq(child)/freq(parent), and sparsity penalty lambda:

P(absorption) >= Phi(lambda * mu_H * sqrt(rho) / sigma - Phi^{-1}(1 - mu_H^2))

This formula yields five testable predictions:
1. High-coherence pairs have higher absorption (correlation rho >= 0.5)
2. Higher freq(child)/freq(parent) ratio increases absorption within coherence bins
3. OrtSAE reduces absorption by driving mu_H -> 0; Matryoshka by reducing effective hierarchy depth
4. Critical width M_c = K * (1 + D * mu_avg^2) / (1 - epsilon) for absorption-free recovery
5. Cross-domain generalization: sigmoid parameters fitted on spelling task transfer to city-country

**Supporting proposition**: For fixed SAE width, reducing coherence (OrtSAE) reduces absorption but increases hedging, formalizing the absorption-hedging tradeoff discovered empirically by Chanin et al. (2025).

### Key Strengths
- First quantitative absorption bound connecting classical dictionary identifiability theory to SAE absorption.
- The critical width formula provides actionable SAE sizing guidance.
- The absorption-hedging tradeoff formalization explains why Matryoshka SAEs trade absorption for hedging.
- All predictions testable on existing pretrained SAEs (training-free).

### Key Risks
- Mean-field approximation (independent absorption events) may fail for dense hierarchies.
- Gaussian activation model may be quantitatively wrong for real LLM activations.
- The bound may be vacuous for practical SAE sizes (M*L0 >> K + R_hierarchy).

### Origin
Theoretical perspective (primary), with elements from Innovator (phase transition analysis) and Empiricist (testing protocol).

---

## Pivot Decision Criteria

The current front-runner (metric audit paper) should be pivoted to an alternative if:

1. **Alternative 1 (Geometry)**: The metric audit reaches 8.0+ and the team seeks a follow-up paper, OR a competing group publishes cross-domain absorption measurements.
2. **Alternative 2 (Immunological)**: Exploratory analysis within the current paper discovers cross-reactive absorption between semantically unrelated co-occurring features.
3. **Alternative 3 (Theory)**: The L0=22 CMI experiment shows strong predictive power (rho < -0.5, p < 0.01) AND decoder coherence correlates with absorption at rho >= 0.5, enabling a standalone theoretical paper.

The current front-runner remains the strongest choice because: (a) it has 8 iterations of accumulated evidence, (b) all perspectives agree cross-domain metric validation is the field's most urgent need, (c) the honest negative results are consistently rated as the paper's strongest aspect, and (d) the path to 8.0 is explicitly specified (3 GPU-hours of experiments).
