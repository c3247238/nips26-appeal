# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Predictive Coding Theory ( Rao & Ballard, 1999; Friston, 2005 )** — The brain uses hierarchical predictive hierarchies where higher levels predict lower-level representations. Prediction errors propagate upward when predictions fail. This suggests hierarchical feature organization is fundamental to neural computation, not an artifact.

2. **Efficient Coding Hypothesis ( Barlow, 1961; Laughlin, 1981 )** — Sensory neurons optimize for sparse, efficient representations given metabolic constraints. The brain's sparse coding strategy directly parallels SAE training objectives. The hierarchical organization emerges from minimizing metabolic cost while maximizing information preservation.

3. **Sparse Coding in V1 ( Olshausen & Field, 1996 )** — Classic sparse coding paper showing simple cells in primary visual cortex emerge from optimizing for sparse, factorial codes. Ground truth for SAEs. However, these models didn't account for the hierarchical structure observed in deeper cortex.

4. **Neural Manifold Hypothesis ( Gallego et al., 2017 )** — Low-dimensional neural trajectories embedded in high-dimensional population activity. The "absorption" phenomenon in SAEs may reflect the geometry of these manifolds — when features project onto overlapping directions, absorption is geometrically inevitable.

5. **Critical Brain Hypothesis ( Chklovskii et al., 2002; Shew & Plenz, 2013 )** — Neural networks operate near critical points (phase transitions) for optimal information processing. This provides the theoretical backbone for the phase transition framework in H1-H4.

#### Physics / Information Theory

6. **Phase Transitions in Sparse Systems ( Mézard et al., 1987 )** — Statistical physics of constraint satisfaction problems exhibits sharp phase transitions. The analogy to SAEs: as sparsity penalty λ increases, the system transitions from "disordered" (independent features) to "ordered" (absorbed features).

7. **Finite-Size Scaling ( Cardy, 1996 )** — Phase transitions in finite systems are broadened according to scaling laws. This provides the exact mathematical framework for H2: transition width δλ ∝ N^(-1/ν).

8. **Rate-Distortion Theory ( Shannon, 1959; Cover & Thomas, 1991 )** — Fundamental limits of compression: there's a minimum description length required for any representation. The absorption phenomenon may be a manifestation of the fundamental tradeoff between sparsity (compression) and reconstruction quality (distortion).

9. **Renormalization Group ( Wilson, 1975 )** — Coarse-graining at different scales reveals universal behavior. Hierarchical features in SAEs may renormalize differently — parent features at coarse scales, child features at fine scales. The absorption may be a renormalization artifact.

10. **Spin Glass Theory ( Parisi, 1983 )** — The energy landscape of correlated features exhibits frustration. When parent-child features are correlated, the system cannot independently optimize both — leading to the absorption compromise.

#### Biology / Evolution

11. **Immune Network Theory ( Jerne, 1974 )** — The immune system uses a network where antibodies recognize other antibodies, creating hierarchical abstractions. This is analogous to features recognizing other features in superposition.

12. **Evolutionary Pressure on Feature Organization** — If certain feature hierarchies are evolutionarily more useful (e.g., "predator" before "lion"), the network pressure to represent them efficiently could drive absorption as an adaptation.

13. **Sparse Settlement in Ecological Systems** — Analogous to sparse resource allocation: when resources (dimensions) are limited, hierarchical bundling emerges to maximize coverage, similar to absorption.

#### Computer Science / Engineering

14. **Matching Pursuit ( Mallat & Zhang, 1993 )** — Greedy sparse coding algorithms from signal processing. MP-SAE (Costa et al., 2025) applies this to SAEs, demonstrating improved hierarchical recovery.

15. **Compressed Sensing ( Donoho, 2006; Candes & Tao, 2006 )** — Theory showing sparse signals can be recovered from incomplete measurements under certain conditions. The phase transition in compressed sensing is mathematically analogous to the absorption transition.

### Cross-Disciplinary Gaps

1. **Neuroscience has not studied "feature absorption"** — The field studies hierarchical processing but not the specific SAE phenomenon. The closest analog is predictive coding where abstract features "absorb" lower-level predictions.

2. **Statistical physics has not analyzed SAE phase transitions** — While phase transitions in neural networks are well-studied, the specific absorption phenomenon in SAEs has not been formally analyzed using methods from statistical physics.

3. **No cross-pollination between immune network theory and SAEs** — Despite both dealing with hierarchical feature recognition and superposition, no work has applied immune network insights to SAE analysis.

4. **Information bottleneck in neural networks vs. SAE training** — The information bottleneck principle hasn't been formally connected to L1 sparsity-driven absorption.

## Phase 2: Initial Candidates

### Candidate A: Critical Phenomena Theory for SAE Absorption (from Statistical Physics)

- **Source principle**: Phase transitions in finite-size systems, specifically the critical threshold behavior and finite-size scaling laws from statistical physics.
- **Structural correspondence**: The L1 sparsity penalty λ in SAEs acts like a "temperature" or external field in a physical system. The absorption rate m(λ) corresponds to the order parameter (magnetization). The dictionary size N corresponds to system size. The susceptibility χ = dm/dλ corresponds to the response function.
- **Hypothesis**: SAE absorption exhibits critical threshold behavior analogous to phase transitions in condensed matter. Below λ_c, absorption is minimal; above λ_c, absorption grows sharply. The transition sharpness scales with dictionary size as δλ ∝ N^(-1/ν).
- **Why it's not just a metaphor**: The mathematical formalism is preserved — same differential equations, same scaling laws, same finite-size corrections. The phase transition framework makes precise quantitative predictions (λ_c, critical exponent ν) that can be empirically tested.
- **Novelty estimate**: 8/10 — While phase transitions in neural networks have been studied, the specific application to SAE absorption and the phase transition framework (H1-H6) is novel.

### Candidate B: Predictive Coding Hierarchical Absorption (from Neuroscience)

- **Source principle**: The predictive coding framework where higher layers generate predictions that are compared against lower-level representations. Mismatches (prediction errors) propagate upward.
- **Structural correspondence**: In SAEs, parent features (general concepts) predict child features (specific instances). When child features activate strongly, they "explain away" the parent feature's signal through the sparse bottleneck — analogous to prediction error suppression in predictive coding.
- **Hypothesis**: Feature absorption in SAEs is a consequence of the hierarchical predictive structure in the underlying LLM activations. Parent features that predict child feature activations will be suppressed by the sparse bottleneck because the child captures the residual.
- **Why it's not just a metaphor**: The causal direction is preserved — in predictive coding, higher-level predictions cause lower-level suppression; in absorption, specific features suppress general features through the same mechanism. The directionality is mathematically identical.
- **Novelty estimate**: 6/10 — MP-SAE (Costa et al., 2025) implicitly uses similar ideas with residual-guided selection, but hasn't been framed as predictive coding.

### Candidate C: Information Bottleneck Theory of Absorption (from Information Theory)

- **Source principle**: The information bottleneck (IB) principle (Tishby et al., 1999) — optimal compression trades off relevance (predicting the output) against compression (minimizing information about the input).
- **Structural correspondence**: The SAE bottleneck acts as an information bottleneck. When parent and child features co-occur frequently, the "relevant" information for reconstruction is contained in the child, so the parent is compressed away (absorbed).
- **Hypothesis**: The negative correlation between co-occurrence and absorption score (r=-0.52) is explained by the IB principle: high co-occurrence means the child latent captures most of the mutual information I(parent; reconstruction), leaving little "relevant" information for the parent latent to encode.
- **Why it's not just a metaphor**: The IB framework provides a formal information-theoretic objective function. The absorbed features are those where I(feature; reconstruction | absorbed_latent) ≈ 0.
- **Novelty estimate**: 7/10 — Connection to IB has been hinted at (Cui et al., ICLR 2026 theoretical limits) but not formally connected to the negative co-occurrence correlation.

### Candidate D: Immune Network-inspired Feature Disentanglement (from Immunology)

- **Source principle**: Jerne's immune network theory — antibodies recognize other antibodies, creating a hierarchical network where higher-order antibodies represent abstractions over lower-order ones.
- **Structural correspondence**: Features in SAEs can be viewed as "antibodies" that recognize patterns in activations. Hierarchical features recognize more abstract patterns. The absorption phenomenon occurs when the immune network (SAE decoder) learns that the more specific antibody (child feature) is sufficient to represent the pattern, suppressing the more general one (parent feature).
- **Hypothesis**: Immune network-inspired regularization could reduce absorption by enforcing explicit recognition structure between hierarchical features, analogous to how immune networks enforce explicit binding between antibodies.
- **Why it's not just a metaphor**: The structural correspondence is at the level of graph topology — both are recognition networks with hierarchical abstraction. The mathematical structure of immune network interactions maps to feature co-activation patterns in SAEs.
- **Novelty estimate**: 9/10 — This analogy has NOT been explored in the SAE literature. While metaphorical connections between neural networks and immune systems are common, the specific structural mapping to feature absorption is novel.

## Phase 3: Self-Critique

### Against Candidate A (Critical Phenomena)

- **Shallow analogy attack**: The phase transition framework requires that absorption exhibits sharp threshold behavior. The pilot data shows susceptibility peaks at λ=5e-4 with χ=1.38, suggesting a broad peak rather than a sharp transition. This is consistent with finite-size broadening but requires further validation.
- **Scale mismatch attack**: The phase transition analogy assumes the system is large enough to exhibit sharp transitions. With N ~ 10k latents, finite-size effects are significant. The observed "peak" rather than discontinuity is consistent with this.
- **Prior transplant check**: Phase transitions in neural networks are well-studied, but the specific application to SAE absorption is novel. No direct prior work exists.
- **Testability attack**: The predictions (λ_c identification, ν estimation, curve collapse) are precisely testable. The full sparsity sweep (12 values, 1000 samples) will directly validate or falsify the framework.
- **Verdict**: STRONG — The pilot data (χ peak at λ=5e-4, monotonic decrease) supports the framework. The testability is high. Novelty is substantial.

### Against Candidate B (Predictive Coding)

- **Shallow analogy attack**: The predictive coding framework in neuroscience involves bidirectional synaptic connections and active inference. SAEs have unidirectional encoder-decoder architecture. The analogy may be superficial.
- **Scale mismatch attack**: Predictive coding operates at the level of individual synapses and circuits. The absorption phenomenon involves population-level latent activations across thousands of features.
- **Prior transplant check**: MP-SAE (Costa et al., 2025) uses residual-guided greedy selection which is conceptually similar to prediction error minimization. The analogy is not entirely novel.
- **Testability attack**: Hard to distinguish "predictive coding explains absorption" from "sparsity optimization causes absorption" — the mechanisms overlap.
- **Verdict**: MODERATE — Conceptually interesting but may not add explanatory power beyond the phase transition framework.

### Against Candidate C (Information Bottleneck)

- **Shallow analogy attack**: The IB principle is a general framework. The specific prediction (negative co-occurrence correlation) was already observed (r=-0.52). The IB explanation may be post-hoc.
- **Scale mismatch attack**: IB theory typically operates on continuous information quantities. The discrete latent activation patterns in SAEs may not cleanly map to IB quantities.
- **Prior transplant check**: Cui et al. (ICLR 2026) established theoretical limits on SAE feature recovery using information-theoretic arguments. This work is directly prior.
- **Testability attack**: The IB prediction is consistent with the observed r=-0.52. But can it predict NEW phenomena? The test is whether the revised scoring formula performs better than existing metrics.
- **Verdict**: MODERATE — Explains existing data but may not generate novel predictions beyond what's already known.

### Against Candidate D (Immune Network)

- **Shallow analogy attack**: Immune networks involve physical binding between antibody and antigen. SAEs involve mathematical similarity between feature vectors. The analogy may be too abstract.
- **Scale mismatch attack**: Immune networks operate at the cellular level with specific binding chemistry. SAEs operate at the level of floating-point vectors in high-dimensional space. Scale mismatch is significant.
- **Prior transplant check**: No direct prior work connecting immune networks to SAE absorption. The analogy is genuinely novel.
- **Testability attack**: The practical test would involve implementing immune-network-inspired regularization on SAEs and measuring absorption reduction. This requires retraining SAEs, conflicting with the training-free analysis constraint.
- **Verdict**: WEAK for immediate application — Novel and interesting but not testable in the current training-free framework. Could inspire future work but not actionable now.

## Phase 4: Refinement

### Dropped Candidates
- **Candidate D (Immune Network)**: Not testable in training-free framework. Interesting for future work but not actionable.

### Strengthened Candidates
- **Candidate A (Critical Phenomena)**: Strongest candidate. Pilot data supports phase transition framework. Makes precise, testable predictions. Novel contribution to the literature.

### Formalized Mapping for Candidate A

The phase transition framework maps to SAE absorption as follows:

| Physical System | SAE Analog | Mathematical Relation |
|----------------|------------|----------------------|
| Temperature T | Sparsity penalty λ | λ plays role of inverse temperature |
| Order parameter M | Absorption rate m(λ) | m(λ) = fraction of absorbed features |
| External field H | Dictionary size N | Effective field depends on N |
| Susceptibility χ | dm/dλ | χ measures system response to λ |
| Critical exponent ν | Critical exponent ν | δλ ∝ N^(-1/ν) |
| Correlation length ξ | Feature correlation length | ξ diverges at λ_c |

### Selected Front-Runner

**Candidate A: Critical Phenomena Theory for SAE Absorption**

This is the front-runner because:
1. Direct empirical support from pilot (χ=1.38 at λ=5e-4)
2. Makes falsifiable predictions (H1-H6)
3. Novel contribution (phase transition framework for SAEs)
4. Aligns with existing literature on critical phenomena in neural networks
5. Testable via standard sparsity sweep experiments

## Phase 5: Final Proposal

### Title

**Phase Transition Theory of SAE Feature Absorption: A Critical Phenomena Framework**

### Source Principle

Critical threshold behavior in finite-size systems. When a system undergoes a phase transition, the order parameter exhibits sharp changes at a critical point. For finite systems, the transition is broadened but still exhibits predictable scaling behavior (finite-size scaling).

### Structural Correspondence

The SAE sparsity penalty λ acts as an external field controlling the system's proximity to the absorption phase transition. The absorption rate m(λ) is the order parameter. The dictionary size N controls finite-size effects.

Mathematical mapping:
- Order parameter: m(λ) = n_absorbed(λ) / N_total
- Susceptibility: χ(λ) = dm/dλ
- Finite-size scaling: δλ(N) ∝ N^(-1/ν)
- Critical point: λ_c where χ peaks

### Hypothesis

SAE feature absorption exhibits critical threshold behavior analogous to phase transitions in condensed matter systems:
1. Below λ_c: absorption rate near zero (disordered phase)
2. Above λ_c: absorption rate increases sharply (ordered phase)
3. Finite-size scaling: transition sharpness increases with dictionary size N

### Method

1. Sparsity sweep: λ ∈ [1e-5, 5e-2] with 12 values
2. Absorption rate measurement: fraction of features with absorption score > threshold
3. Susceptibility calculation: χ = Δm/Δλ at each λ
4. Critical point identification: λ_c = argmax(χ)
5. Finite-size scaling validation: compare transition width across dictionary sizes

### Diagnostic Experiment

**Full Sparsity Sweep (H1-validation)**:
- 12 λ values from 1e-5 to 5e-2
- 1000 samples per λ (prompts from first-letter spelling task)
- Primary metric: m(λ) and χ(λ)
- Success criterion: identify λ_c where χ peaks; observe finite-size scaling

The diagnostic experiment succeeds if and only if:
- χ peaks sharply at some λ_c (not gradual monotonic increase)
- m(λ) exhibits threshold behavior (not smooth increase)
- The peak is robust across multiple random seeds

### Experimental Plan

| Experiment | Duration | Model | SAE Config | Hypothesis Validated |
|------------|----------|-------|------------|---------------------|
| full_sparsity_sweep | 45 min | GPT-2-small | layer 6, 16k latents | H1 (critical threshold) |
| cross_layer_absorption | 30 min | GPT-2-small | layers 0,3,6,9,11 | H3 (layer depth effect) |
| finite_size_scaling | 60 min | GPT-2-small | 8k,16k,32k latents | H2 (scaling law) |

### Risk Assessment

1. **Sharpness risk**: If absorption increases gradually with no clear threshold, the phase transition framework fails. Mitigation: fall back to rate-distortion interpretation.
2. **Scale mismatch risk**: With N ~ 10k, finite-size effects may dominate, preventing sharp transitions. Mitigation: use scaling analysis to account for finite-size broadening.
3. **Model-specific risk**: The phase transition may be specific to GPT-2 layer 6. Mitigation: replicate across multiple layers (H3) and models.
4. **Confounding risk**: Other factors (feature frequency, co-occurrence) may affect absorption, not just λ. Mitigation: control for co-occurrence in H5 analysis.

### Novelty Claim

This is the **first application of critical phenomena theory to SAE feature absorption**. While phase transitions in neural networks have been studied (Shew & Plenz, 2013), the specific mapping to SAE absorption and the prediction of critical threshold behavior is novel. The phase transition framework provides a new theoretical lens for understanding and quantifying absorption, distinguishing it from prior work that focuses on architectural modifications or empirical characterizations.

### Cross-Disciplinary Contribution

This work bridges:
1. **Statistical physics** → providing mathematical framework for phase transitions and finite-size scaling
2. **Neuroscience** → connecting to predictive coding and hierarchical processing theories
3. **Information theory** → complementing rate-distortion and information bottleneck perspectives

The novelty lies in synthesizing these perspectives into a coherent, testable framework for SAE absorption that neither field has previously addressed.