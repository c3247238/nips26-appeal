# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Chanin et al., 2024/2025. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507 (NeurIPS 2025)** — First systematic definition and quantification of absorption; proves hierarchy + sparsity inevitably causes absorption via toy models; validated across hundreds of LLM SAEs. Found absorption rate of 0.14 for TopK Gemma-2 2B SAEs.

2. **Hu et al., 2025. "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717** — Introduced feature sensitivity as evaluation dimension; found sensitivity declines with SAE width; low-sensitivity features may indicate absorption.

3. **Karvonen et al., 2025. "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders." arXiv:2503.09532 (ICML 2025)** — Eight-metric benchmark on 200+ SAEs; revealed proxy metrics (reconstruction, L0) do not predict practical interpretability.

4. **Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547** — Nested dictionaries simultaneously; reduced absorption 0.49→0.05; smaller dicts learn general features, larger specialize.

5. **Luo et al., 2026. "Building a Structured Feature Forest with Hierarchical Sparse Autoencoders." arXiv:2602.11881** — HSAE with explicit parent-child relationships; substantially outperforms baselines on absorption.

6. **Korznikov et al., 2025. "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033** — Orthogonality penalty; 65% absorption reduction; suggests absorbed features are less orthogonal.

7. **Chanin et al., 2026. "SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data." arXiv:2602.14687** — Ground-truth benchmark (16k features, hierarchy, correlation, superposition); MP-SAEs exploit superposition noise.

8. **Lieberum et al., 2024. "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2." arXiv:2408.05147** — Full-layer Gemma 2 SAEs (16k/65k/1m, 27 layers); dominant experimental platform; absorption observed across all layers.

9. **Chanin & Garriga-Alonso, 2025. "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders." arXiv:2505.11756** — Related failure mode; correlated features merge in narrow SAEs; compound multiplier can balance Matryoshka SAE performance.

10. **Marks et al., 2025/2026. "A Unified Theory of Sparse Dictionary Learning." arXiv:2512.05534** — Theoretical framework with piecewise biconvexity and spurious minima; feature anchoring improves recovery.

### Landscape Summary

The SAE field has evolved from a period of architectural proliferation (TopK, JumpReLU, Matryoshka, HSAE, OrtSAE, AdaptiveK) toward a mature evaluation paradigm (SAEBench, SynthSAEBench). Feature absorption is now recognized as a core unsolved problem—it is not merely a metric artifact but a structural consequence of optimizing sparsity on hierarchically-structured feature distributions.

Three converging insights define the current frontier:

1. **Hierarchical supervision is necessary**: Both Matryoshka and HSAE explicitly model parent-child relationships, confirming that absorption stems from the SAE's inability to distinguish within-hierarchy features without explicit structural constraints.

2. **Proxy metrics are misleading**: Reconstruction and L0 do not predict absorption. The field lacks a training-free proxy that correlates with absorption severity.

3. **Absorption has downstream consequences**: Beyond classification accuracy, absorbed features may undermine circuit analysis, steering, and model editing applications—a gap not yet systematically explored.

**Critical gap for ideation**: No existing work treats absorption as a predictable equilibrium phenomenon with quantifiable leading indicators. Current approaches are architectural (design new SAE types) or computational (train with hierarchical constraints). The possibility that absorption follows discoverable laws governed by feature correlation statistics remains underexplored.

---

## Phase 2: Initial Candidates

### Candidate A: Feature Absorption as Competitive Exclusion in High-Dimensional Niches

- **Hypothesis**: Feature absorption in SAEs follows the competitive exclusion principle from ecology—the same niche cannot sustain two features without one driving out the other. Specific (child) features and general (parent) features occupy overlapping representational niches; when optimization pushes for sparsity, the specific feature "excludes" the general one by exploiting the shared direction.

- **Cross-domain insight**: In ecology, Gause's competitive exclusion principle states that two species competing for the same limited resources cannot coexist at constant population values. In SAE feature space, general features ("starts with S") and specific features ("short word starting with S") compete for the same residual direction in high-dimensional space. When L0 sparsity forces selection, the more specialized feature outcompetes the generalist.

- **Why it might work**: Bussmann et al. found smaller Matryoshka dicts learn general features, larger dicts learn specific ones—this niche differentiation suggests competition dynamics. If absorption is competitive exclusion, then features with higher correlation should show higher absorption rates. The 65% absorption reduction in OrtSAE (orthogonality enforcement) may work by artificially expanding niche separation.

- **Novelty estimate**: 8/10 — No paper explicitly frames absorption as competitive exclusion. The ecological analogy has not been explored despite clear structural correspondence (limited representational "resources," multiple "species" competing for direction space).

### Candidate B: Absorption as Currency Debasement — Feature Inflation Dynamics

- **Hypothesis**: Feature absorption mimics currency debasement—when the representational "value" of a general feature gets diluted by many child features "minting" claims on its direction. Absorbed features are "debased" the same way an over-printed currency loses purchasing power. The absorption rate follows an inflation-like growth curve determined by the ratio of child features to parent feature "reserves."

- **Cross-domain insight**: In monetary economics, debasement occurs when the circulating currency exceeds the backing reserves, diluting each unit's value. In SAEs, when multiple specific features (children) point in approximately the same direction as a general feature (parent), they collectively "dilute" the parent's representational capacity. The decoder weights for the parent feature become smeared across child directions, reducing its independence.

- **Why it might work**: Chanin et al.'s absorption rate metric directly measures directional "dilution." The mathematical form of absorption (compensated vs. main features) mirrors how currency debasement ratios are computed. If absorption follows inflation dynamics, then features with more "children" should show higher absorption rates in a predictable nonlinear pattern.

- **Novelty estimate**: 6/10 — Some connection to "feature anchoring" in Marks et al. (unified theory) but no explicit debasement/inflation framing. The mathematical analogy is clear but underexplored.

### Candidate C: Encoder-Decoder Asymmetry as Absorption Early Warning System

- **Hypothesis**: Absorbed features exhibit predictable encoder-decoder asymmetry—specifically, the decoder reconstructs the feature's output with higher fidelity than the encoder encodes it. This asymmetry is detectable without ablation and serves as a training-free absorption indicator. Features with decoder/encoder ratio > threshold are "warning signs" of absorption.

- **Cross-domain insight**: In quality control systems, the relationship between input precision and output fidelity often reveals manufacturing defects—a product that comes out more perfect than the process should allow suggests hidden rework or quality transfer. Absorbed features similarly "over-deliver" on reconstruction because the child feature has borrowed the parent's direction, allowing the decoder to reconstruct using a path the encoder did not actually encode.

- **Why it might work**: This directly addresses Gap 4 from the literature (training-free absorption detection). The approach is testable using existing Gemma Scope SAEs and SAELens. If confirmed, it provides a simple metric (encoder-decoder correlation divergence) that doesn't require ablation or ground truth.

- **Novelty estimate**: 7/10 — Addresses Gap 4 but no paper explicitly proposes encoder-decoder asymmetry as absorption signal. Several papers hint at encoder-decoder behavior differences (Marks et al., Hu et al.) but none frame it as a diagnostic tool.

---

## Phase 3: Self-Critique & Adversarial Testing

### Against Candidate A: Competitive Exclusion

- **Prior work attack**: Search for "SAE feature competition" and "feature hierarchy competition" — the hierarchical SAE papers (Bussmann, Luo) do use competitive framing but in the context of training dynamics, not ecological principle. Is the ecological analogy merely metaphorical or mechanistically predictive?

- **Methodological attack**: Competitive exclusion in ecology requires defining the "niche"—how do we measure feature niche overlap in SAE representation space? Without an operationalized measure, the hypothesis is unfalsifiable. The niche definition must be grounded in SAE geometry (cosine similarity, projection magnitudes).

- **Theoretical attack**: The analogy may be superficial. Ecological competition is about survival in fixed environments; SAE optimization actively reshapes the feature space. The competition isn't between pre-existing features fighting over resources—it's an emergent outcome of gradient-based optimization on a superposition-encoded representation.

- **Scalability attack**: Even if the basic principle holds for simple cases, high-dimensional feature spaces (1M latents) may exhibit different dynamics. Competition that works at 16k features may not scale predictably.

- **Verdict**: MODERATE — The insight is genuinely novel and cross-domain, but requires operationalizing "niche" in SAE terms and validating the competition dynamics. The ecological analogy is inspiring but may be metaphorical rather than predictive. Could be strengthened by quantifying niche overlap and testing on SynthSAEBench ground truth.

### Against Candidate B: Currency Debasement

- **Prior work attack**: Search for "feature inflation" and "absorptions inflation dynamics" — marks et al. "feature anchoring" concept is related but does not use inflation/debasement framing. Does any paper quantify absorption as a function of child/parent ratio?

- **Methodological attack**: The "reserve" analogy requires quantifying how much "backing" each general feature has. How do we measure the representational "reserves" of a feature direction? If we can't measure it, we can't test the inflation hypothesis.

- **Theoretical attack**: Currency debasement is a deliberate policy decision; absorption is an emergent gradient phenomenon. The analogy may be too anthropomorphic—optimization doesn't "choose" to debase features. The mechanism is gradient-based, not policy-based.

- **Scalability attack**: Inflation dynamics in economics are influenced by complex socioeconomic factors beyond simple ratios. The child/parent ratio may not capture all relevant dynamics—feature correlation structure, decoder weight magnitudes, and training dynamics may all contribute.

- **Verdict**: WEAK — While intuitively appealing, the currency debasement framing doesn't add predictive power over the direct statistical relationship between child features and absorption. The mechanism (gradient optimization) is so different from monetary policy that the analogy may be misleading rather than illuminating. Not falsifiable without operationalizing "reserves."

### Against Candidate C: Encoder-Decoder Asymmetry

- **Prior work attack**: Search for "encoder decoder asymmetry SAE" — marks et al. discuss piecewise biconvexity and spurious minima. Hu et al. discuss sensitivity but not encoder-decoder divergence. Is there direct prior work?

- **Methodological attack**: The asymmetry metric needs to be precisely defined. Encoder fidelity vs. decoder fidelity could be measured as reconstruction error on feature-specific inputs vs. feature reconstruction from latent activations. But the threshold (what ratio indicates absorption) is arbitrary without ground truth.

- **Theoretical attack**: The asymmetry could be a symptom of absorption OR a consequence of normal feature specialization. The causal direction is unclear—is absorption causing the asymmetry, or does asymmetry precede and predict absorption?

- **Scalability attack**: At deep layers (attention layers), encoder-decoder behavior may fundamentally differ due to information integration patterns. If absorption primarily affects attention layers (Chanin notes mediation effects disappear at >layer 17), the asymmetry signal may be weak or absent there.

- **Verdict**: MODERATE — The approach is training-free and addresses a real gap, but needs ground truth validation. Could be tested on SynthSAEBench where we know which features are "absorbed" and can check if asymmetry predicts it. Strong potential if validated.

---

## Phase 4: Refinement

### Dropped Ideas
- **Currency Debasement (Candidate B)** dropped because: The analogy does not add predictive power and the "reserves" concept is not operationalizable. The mechanistic difference between gradient optimization and monetary policy makes the analogy misleading.

### Strengthened Ideas
- **Competitive Exclusion (Candidate A)**: Refined to focus on quantifiable niche overlap. Instead of vague ecological analogy, operationalize as: niche width = average cosine similarity between feature decoder direction and its k-nearest neighbors in child feature set. Absorption occurs when child features collectively reduce the parent's "exclusive direction territory." This becomes testable with existing Gemma Scope SAEs.

- **Encoder-Decoder Asymmetry (Candidate C)**: Strengthened by grounding in information theory. Define asymmetry as mutual information divergence: I(encoder | feature) vs. I(decoder | feature). Use SynthSAEBench ground truth to calibrate the threshold and validate predictive power.

### Additional Evidence Found
- No explicit prior work on ecological competition framing for absorption
- No prior work on encoder-decoder asymmetry as absorption signal
- Marks et al. "feature anchoring" is the closest conceptual work to both candidates—provides theoretical grounding

### Selected Front-Runner
**Candidate A: Feature Absorption as Competitive Exclusion in High-Dimensional Niches**

Reason: This idea has the strongest cross-domain novelty and generates concrete, falsifiable predictions. The ecological competitive exclusion principle provides a clear theoretical frame that (a) explains *why* absorption happens (competition for limited representational resources), (b) predicts *when* it will happen (features with high niche overlap with hierarchy members), and (c) suggests *how* to detect it (niche overlap metrics). Unlike the debasement analogy, the competition mechanism has direct mapping to gradient optimization dynamics. Unlike the encoder-decoder asymmetry approach, it addresses the root cause rather than a symptom.

The idea is actionable: we can operationalize niche overlap using decoder direction cosine similarity, test whether high-overlap features show higher absorption rates, and use this to build a training-free absorption prediction metric.

---

## Phase 5: Final Proposal

### Title

**Competitive Exclusion in Feature Space: Quantifying Absorption as Niche Overlap in Sparse Autoencoders**

### Hypothesis

Feature absorption in SAEs is a predictable consequence of competitive exclusion dynamics: specific (child) features and general (parent) features occupy overlapping representational niches; when L0 sparsity optimization forces selection, the feature with higher specialization (typically the child) dominates the shared direction, effectively "excluding" the parent feature from meaningful activation. The absorption rate of a parent feature is a monotonic function of its niche overlap with its child feature set.

### Motivation

Current understanding of absorption treats it as an architectural side effect to be mitigated through hierarchical training (Matryoshka, HSAE) or orthogonality constraints (OrtSAE). However, no work explains *why* absorption follows the patterns observed (e.g., absorption rate increases with hierarchy depth, absorption is worse for correlated features). The competitive exclusion framework provides a unifying explanation: absorption is not a bug but a predictable consequence of optimization on superposition-encoded features with hierarchical structure.

This matters because: (1) if absorption is predictable, it is also preventable or detectable; (2) understanding the mechanism suggests interventions beyond architectural changes; (3) quantifying niche overlap may serve as a training-free absorption early warning system.

### Method

**Define Feature Niche**: For a given feature f with decoder direction w_f, its niche is the spherical region in activation space where w_f provides the primary explanation. Niche overlap between parent feature p and child feature c is measured as:

```
overlap(p, c) = |cosine(w_p, w_c)| * sqrt(|w_c|_2 / |w_p|_2) * activation_correlation(p, c)
```

Where activation_correlation measures how often p and c activate on the same inputs (from SAE feature logs).

**Predict Absorption**: For each feature p with children C(p), compute:

```
absorption_risk(p) = sum_{c in C(p)} overlap(p, c) / |C(p)|
```

Test whether features with high absorption_risk show higher absorption rates on Gemma 2 2B SAEs across layers 0-17 (where ablation-based ground truth exists).

**Training-Free Detection**: If the correlation holds, absorption_risk becomes a training-free absorption predictor usable on any pretrained SAE without ablation experiments.

### Experimental Plan

1. **Compute Niche Overlap**: Load Gemma Scope SAEs (16k width) via SAELens. For features with identified children (using Neuronpedia hierarchical annotations or the feature forest structure from HSAE), compute niche overlap metrics.

2. **Validate Against Ground Truth**: On layers 0-17 where ablation-based absorption rates exist, test whether absorption_risk predicts actual absorption rates. Compute Pearson correlation between predicted risk and measured absorption.

3. **Cross-Architecture Comparison**: Repeat on TopK, JumpReLU, and Matryoshka SAEs. If competitive exclusion holds, architectures with explicit niche separation (Matryoshka nested dicts) should show lower absorption_risk values for equivalent hierarchy relationships.

4. **Circuit Completeness Test**: Test whether absorbed features (high risk, confirmed by ablation) show degraded circuit completeness when used for steering. Compare steering success rates for high-risk vs. low-risk features within the same hierarchy.

5. **Synthetic Validation**: Use SynthSAEBench ground truth to validate the niche overlap calculation. Known absorbed features in synthetic data should have high absorption_risk.

### Resource Estimate

- **Model**: Gemma-2-2B with Gemma Scope SAEs (16k width, layer 12 as test case)
- **Dataset**: Activation logs from SAELens eval runs (~10k tokens); SynthSAEBench for ground truth validation
- **Time**: ~45 minutes for full layer analysis on 1 GPU (niche computation is matrix operations, no training)
- **Compute**: Feature extraction + cosine similarity computation: ~2M operations per layer; negligible cost

### Risk Assessment

1. **Niche Definition Arbitrariness**: The niche overlap formula may not capture the true geometric relationship. *Mitigation*: Test multiple formulations (cosine only, L2 only, activation correlation only) and select the one with highest correlation to ground truth absorption.

2. **Hierarchical Annotations Required**: Need child/parent relationships for features—this may require using HSAE-style feature forest annotations or Neuronpedia hierarchical data. *Mitigation*: Test on SynthSAEBench which has explicit ground-truth hierarchy, then transfer to real SAEs.

3. **Attention Layer Extension Uncertain**: The competitive exclusion model may not apply to deep attention layers where information integration patterns differ. *Mitigation*: Validate on layers 0-17 first; if confirmed, investigate whether adaptation to attention layers requires modified niche definitions.

### Novelty Claim

This is the first work to explicitly frame SAE feature absorption as competitive exclusion following ecological principles. While hierarchical SAE architectures implicitly address absorption through structural constraints, no prior work:
1. Provides a quantitative competitive exclusion model for absorption prediction
2. Derives a training-free absorption risk metric from first principles
3. Validates the niche overlap → absorption correlation across multiple SAE architectures

If validated, the framework connects absorption to a broader class of resource-competition phenomena, potentially allowing Borrowing from ecological niche modeling literature to predict and prevent absorption before training.