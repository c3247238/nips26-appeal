# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **[Maes et al., 2026. LeWorldModel: Stable End-to-End JEPA from Pixels. arXiv:2603.19312]** -- The target model: first end-to-end JEPA from raw pixels using SIGReg for Gaussian latent regularization. 15M params, trains on single GPU. Probes physical states but has no cross-domain compositional evaluation. Fails in low-diversity Two-Room environment.

2. **[Uselis et al., 2026. Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models. arXiv:2602.24264]** -- Proves that compositional generalization *necessitates* linear, orthogonal per-concept decomposition. Validates on CLIP/SigLIP/DINO but not on world model predictors. This is the key theoretical bridge: SIGReg enforces isotropic Gaussian, which may or may not produce the required linear orthogonal structure.

3. **[Liang & Liu, 2024. How Diffusion Models Learn to Factorize and Compose. NeurIPS 2024, arXiv:2408.13256]** -- Shows diffusion models learn "hyper-factorized" orthogonal representations from factorized data. Demonstrates that observing the full range of each independent factor plus a few compositional examples suffices for compositional generalization. Critical insight: data structure drives representation structure.

4. **[Liang et al., 2025. Compositional Generalization via Forced Rendering of Disentangled Latents. arXiv:2501.18797]** -- Disentangled latents in an abstract bottleneck are insufficient; the model must actively maintain factorization in the representational space. Challenges the naive assumption that disentanglement = composition.

5. **[Knopp et al., 2025. Predictive Learning Enables Compositional Representations. bioRxiv:2025.09.26.678731]** -- Neuroscience-inspired: RNNs trained *only* to predict future sensory inputs autonomously develop modular, disentangled representations with winner-take-all dynamics. Each module implements a single dynamic, enabling compositional generalization to unseen combinations. This is the biological precedent for JEPA-style predictive learning inducing compositional structure.

6. **[Delliaux et al., 2025. Learning Abstract World Models with a Group-Structured Latent Space. arXiv:2506.01529]** -- Imposes geometric (group) priors on latent transition models. Disentangles Z_sym x Z_unsym, improving prediction and downstream RL. Shows structured latent spaces outperform unstructured ones for generalization.

7. **[Schwarcz, 2026. Semantic Interaction Information Mediates Compositional Generalization in Latent Space. arXiv:2603.27134]** -- RCCs: JEPA-style architecture separating variable inference from variable embeddings. Demonstrates compositional generalization to novel variable combinations. Introduces Semantic Interaction Information (SII) as a measure of interaction-driven accuracy gains.

8. **[Balsells Rodas et al., 2024. Dreaming of Many Worlds: Learning Contextual World Models Aids Zero-Shot Generalization. arXiv:2403.10967]** -- cRSSM: contextual world model on Dreamer that incorporates physical context (gravity, actuator strength) for zero-shot generalization. Disentangles latent state from physical context. Evaluated on CARL/DMControl Walker with varying gravity.

9. **[Balestriero et al., 2025. LeJEPA: Scalable, Provable Self-Supervised Learning. arXiv:2511.08544]** -- The theoretical foundation for SIGReg: proves isotropic Gaussian is the optimal embedding distribution for foundation models. SIGReg uses Cramer-Wold theorem via random projections + Epps-Pulley normality test.

10. **[Zhang et al., 2026. Weak-SIGReg: Covariance Regularization for Stable Deep Learning. arXiv:2603.05924]** -- Extends SIGReg to supervised learning; recovers collapsed ViT from 20.73% to 72.02% accuracy. Views collapse prevention as constraining representation drift toward isotropic Gaussian.

11. **[Becigneul et al., 2025. PrivilegedDreamer: Explicit Imagination of Privileged Information. arXiv:2502.11377]** -- Handles hidden physical parameters (mass, friction) via explicit prediction module in DreamerV2. Relevant baseline for physical parameter adaptation in world models.

12. **[JMLR 2025. Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability]** -- Formalizes intervention-based probing, including the linear representation hypothesis and modular features, using causal abstraction. Provides the methodological framework for probing what LeWM's latent space actually encodes.

### Landscape Summary

The field of JEPA world models has converged on a clear architecture: ViT encoder + transformer predictor, trained end-to-end with regularization to prevent collapse. LeWorldModel represents the current frontier, achieving this with remarkable simplicity (2 loss terms, 15M params). However, a critical theoretical gap exists: the compositional generalization community (Uselis et al., 2026; Liang & Liu, 2024) has established that composition *requires* linear, orthogonal per-concept decomposition, yet nobody has tested whether SIGReg's isotropic Gaussian enforcement actually produces this structure in a world model's latent space.

The neuroscience literature (Knopp et al., 2025) provides a powerful hint: predictive learning -- the very objective JEPA optimizes -- autonomously induces modular, compositional representations in recurrent networks trained on environments with independent latent factors. If this finding transfers from biological networks to JEPA architectures, then LeWM *should* develop compositional structure, but only when trained on data with sufficient factor diversity. This would explain the Two-Room failure: low diversity prevents the emergence of the modular dynamics required for composition.

A critical missing piece is the *interventional* evaluation of compositional structure. Existing work probes representations observationally (linear regression on frozen embeddings), but never tests whether manipulating one physical concept in latent space independently affects the corresponding output while leaving others unchanged. The causal abstraction framework (JMLR 2025) provides the tools for this, but nobody has applied it to world model latent spaces. This is where the real innovation opportunity lies: not just measuring *whether* physical concepts are encoded, but measuring *how independently they are encoded* and whether this independence enables composition.

## Phase 2: Initial Candidates

### Candidate A: Causal Factorization Probing -- Interventional Tests of Compositional Independence in LeWM's Latent Space

- **Hypothesis**: LeWM's SIGReg-regularized latent space encodes physical concepts (gravity, friction, object properties) in approximately linear, orthogonal subspaces, and interventions on one concept subspace selectively alter the corresponding physical prediction while leaving other concepts invariant. This compositional factorization degrades predictably as training data diversity decreases.
- **Cross-domain insight**: From causal abstraction / mechanistic interpretability -- rather than only *observing* what information the latent space contains (standard probing), we *intervene* on latent dimensions to test causal independence of physical concept representations. This transplants the activation patching methodology from LLM interpretability into world model evaluation.
- **Evidence for**: (1) Uselis et al. 2026 proves linear orthogonal decomposition is *necessary* for composition; (2) Knopp et al. 2025 shows predictive learning autonomously induces modular representations; (3) SIGReg enforces isotropic Gaussian which, by distributing variance uniformly across dimensions, may implicitly encourage orthogonal factorization; (4) Causal abstraction (JMLR 2025) provides mature methodology for intervention-based probing.
- **Novelty estimate**: 8/10 -- Interventional probing of world model latent spaces for physical concept independence is genuinely new. Existing probing is purely observational. The connection between SIGReg geometry and compositional factorization is untested.

### Candidate B: Modular LoRA Surgery -- Diagnosing Compositional Knowledge Localization via Selective Adaptation

- **Hypothesis**: Physical concept knowledge in LeWM is localized: adapting specific LoRA modules (encoder-only vs. predictor-only vs. specific attention heads) to a novel physical domain reveals which components encode domain-general vs. domain-specific knowledge. If encoder LoRA alone recovers generalization, physical concepts are encoder-localized; if predictor LoRA is needed, the predictor has learned domain-specific dynamics that fail to compose.
- **Cross-domain insight**: From neurosurgery / ablation studies in neuroscience -- selectively lesioning or adapting specific neural circuits to identify functional localization. Also draws from the "LoRA as a diagnostic tool" idea in NLP, where adaptation efficiency reveals what pre-training captured.
- **Evidence for**: (1) No prior work has applied LoRA to LeWM encoder/predictor separately; (2) V-JEPA 2-AC shows action-conditioning fine-tuning of predictor alone enables new capabilities, suggesting modularity; (3) ProLoRA (2025) shows LoRA transfers can be analyzed structurally; (4) The encoder-predictor split in JEPA creates a natural anatomy for surgical adaptation.
- **Novelty estimate**: 6/10 -- LoRA adaptation for domain transfer is common in NLP/vision; applying it as a *diagnostic tool* for knowledge localization in world models is moderately novel. The encoder-vs-predictor dissection is the specific contribution.

### Candidate C: Compositional Phase Transitions -- Predicting Generalization Boundaries from Latent Geometry

- **Hypothesis**: LeWM's compositional generalization exhibits a sharp phase transition as training data diversity crosses a critical threshold. Below the threshold, latent representations collapse into entangled, non-factorized codes; above it, they spontaneously factorize into orthogonal concept subspaces. The transition point can be predicted from the principal angle spectrum of concept subspaces in the latent space, and the boundary maps to a specific combinatorial coverage ratio (as predicted by the data diversity theory of Redhardt et al., 2025).
- **Cross-domain insight**: From statistical physics / phase transitions -- the spontaneous emergence of order (factorized structure) above a critical temperature (data diversity). Also draws on the "swing-by dynamics" observation (arXiv:2410.08309) where concept learning shows non-monotonic loss behavior during training, analogous to symmetry-breaking transitions.
- **Evidence for**: (1) Liang & Liu 2024 show factorization emerges from data structure in diffusion models; (2) LeWM's Two-Room failure is consistent with being below a diversity threshold; (3) Redhardt et al. 2025 show compositional generalization is driven by data diversity not scale; (4) Phase transitions in representation learning have been documented in other architectures (grokking, sudden capability emergence).
- **Novelty estimate**: 9/10 -- Framing compositional generalization in world models as a phase transition with a predictable critical point is highly novel. No prior work has mapped the "generalization boundary" in physical concept factor space for JEPA models.

## Phase 3: Self-Critique

### Against Candidate A: Causal Factorization Probing

- **Prior work attack**: Searched for "interventional probing world model latent space" and "activation patching world model physical concepts." Found CausalARC (arXiv:2509.03636) which uses causal interventions for abstract reasoning evaluation, but in grid worlds, not in continuous physical world models. Found the JMLR causal abstraction paper which formalizes the methodology but applies it to LLMs, not world models. The AIM framework (arXiv:2603.20327) probes V-JEPA 2 but uses *passive* quantization, not active interventions. **Verdict: Genuinely novel application to world model latent spaces.**
- **Methodological attack**: The main risk is that LeWM's latent space is too low-dimensional (ViT-Tiny produces ~192-dim embeddings) to cleanly separate physical concepts into orthogonal subspaces, making interventions noisy. Mitigation: use principal component analysis to identify concept-aligned directions first, then intervene along those directions. Additionally, the intervention protocol requires generating counterfactual rollouts, which means we need to decode from latent space -- but JEPA has no decoder. Mitigation: use downstream probing accuracy as the readout instead of pixel reconstruction.
- **Theoretical attack**: The Cramer-Wold theorem underlying SIGReg guarantees the joint distribution matches isotropic Gaussian, but this does not *logically entail* orthogonal factorization by physical concept. An isotropic Gaussian can be achieved by many non-factorized representations. The connection between Gaussianity and concept orthogonality is a plausible hypothesis, not a guaranteed outcome. This is actually what makes it interesting to test empirically.
- **Scalability attack**: The intervention protocol requires training concept-aligned probes first, then performing controlled perturbations and measuring downstream effects. This is computationally feasible with LeWM's 15M params but requires careful experimental design to avoid confounds. Should scale to 5-10 physical concepts without issues.
- **Verdict**: STRONG -- The main weakness (latent space may not cleanly separate concepts) is precisely what the experiment is designed to test. The lack of a decoder is a real constraint but can be worked around with probing-based readouts.

### Against Candidate B: Modular LoRA Surgery

- **Prior work attack**: Searched for "LoRA adaptation world model encoder predictor" and "knowledge localization LoRA vision transformer." Found extensive work on LoRA for vision transformers (LoRA Recycle, ProLoRA) and for video world models (1X World Model Challenge). However, the specific encoder-vs-predictor diagnostic in JEPA has not been done. The concept of using adaptation efficiency as a diagnostic is established in NLP (e.g., "probing via fine-tuning") but novel for world models. **Verdict: Moderately novel.**
- **Methodological attack**: LoRA rank selection introduces a confound: low-rank adaptation may be insufficient to capture the adaptation needed, making "failure to adapt" ambiguous (is it because the knowledge isn't there, or because LoRA rank is too low?). Mitigation: sweep rank from 1 to 64 and measure adaptation curves. Additionally, the encoder and predictor interact; adapting one affects the other's input distribution, complicating causal attribution. Mitigation: use frozen encoder + adapted predictor (and vice versa) as controlled conditions.
- **Theoretical attack**: The encoder-predictor decomposition in JEPA is not a clean separation of "what" and "how" -- the encoder produces temporally-aware embeddings that already encode dynamics implicitly (because SIGReg acts on step-wise embeddings). So the hypothesis that "physical concepts are encoder-localized" may be too simplistic.
- **Scalability attack**: Each LoRA configuration requires fine-tuning on the target domain, which adds computational cost. With LeWM's small size, each adaptation run is fast (~10 min). But systematically testing all combinations (encoder/predictor/both x multiple domains x multiple LoRA ranks) creates a combinatorial explosion. Need to be strategic about which conditions to test.
- **Verdict**: MODERATE -- Useful as a supporting experiment but the central hypothesis (clean encoder-predictor localization) may be too simplistic for JEPA where both components are tightly coupled. The adaptation efficiency curves would still provide valuable diagnostic information.

### Against Candidate C: Compositional Phase Transitions

- **Prior work attack**: Searched for "phase transition representation learning compositional generalization" and "critical threshold data diversity factorization." Found the "swing-by dynamics" paper (arXiv:2410.08309) which documents non-monotonic learning dynamics for concept learning, but in diffusion models. The data diversity work (arXiv:2507.07102) shows compositional generalization correlates with combinatorial coverage, but does not frame it as a phase transition with a sharp critical point. **Verdict: Novel framing, building on existing empirical observations.**
- **Methodological attack**: Detecting a "phase transition" requires training many models at different data diversity levels and measuring the sharpness of the factorization onset. This is computationally expensive. With LeWM training in "hours on a single GPU," we need perhaps 20-30 training runs at different diversity levels. If each takes 2-4 hours, this is 40-120 GPU-hours. Feasible but significant. Mitigation: use a faster proxy metric (e.g., factorization score measured during training, not at convergence) and a coarser grid.
- **Theoretical attack**: Phase transitions in finite-size systems (like neural networks) are typically crossovers rather than sharp transitions. The "critical point" may be a broad crossover region, reducing the predictive power and elegance of the framing. Additionally, the analogy to statistical physics is metaphorical -- there is no partition function or free energy to derive the transition from first principles.
- **Scalability attack**: The benchmark requires creating environments with systematically varying physical factor combinations. DMControl supports this (gravity, friction via MuJoCo XML), but creating a clean factored benchmark with 3+ independently varying factors each at 3+ levels requires 27+ environment configurations. This is doable but requires careful engineering.
- **Verdict**: STRONG -- Despite the computational cost, this is the most novel and highest-impact idea. The phase transition framing, even if approximate, would provide the first principled framework for predicting *where* world model generalization breaks down. The Two-Room failure becomes a natural data point.

## Phase 4: Refinement

### Dropped Ideas

- No ideas dropped. All three survived self-critique, but with different roles:
  - Candidate C (phase transitions) is the main conceptual contribution
  - Candidate A (causal factorization probing) provides the measurement methodology
  - Candidate B (LoRA surgery) serves as a complementary diagnostic

### Strengthened Ideas

**Primary idea: Merge Candidates A and C into a unified framework -- "Compositional Phase Diagrams for JEPA World Models via Interventional Probing"**

The core insight is that Candidates A and C are naturally complementary: Candidate C asks *where* compositional generalization breaks down (the boundary), and Candidate A provides *how* to measure it (interventional probing of concept independence). Together, they form a complete research program:

1. Build a factored benchmark varying 3 physical parameters (gravity scale, friction coefficient, object mass) across 3-4 levels each in DMControl environments
2. Train LeWM on varying subsets of factor combinations (controlling combinatorial coverage)
3. At each coverage level, measure:
   - Standard probing accuracy (observational)
   - Concept subspace orthogonality (principal angle analysis from Uselis et al.)
   - **Interventional concept independence** (the novel metric): perturb the latent representation along a concept-aligned direction and measure whether only the corresponding physical prediction changes
   - Zero-shot planning success on held-out combinations
4. Plot these metrics as functions of combinatorial coverage to identify the compositional phase boundary
5. Use LoRA adaptation (Candidate B) as a post-hoc diagnostic: apply LoRA at the boundary to determine what specific knowledge is missing

**Specific strengthening changes:**
- Added LoRA as a supporting diagnostic rather than a standalone contribution, addressing its "moderate" novelty
- Added principal angle analysis (from Uselis et al.) as an existing metric to complement the novel interventional metric, providing continuity with prior work
- Added the specific DMControl parameterization (gravity, friction, mass via MuJoCo XML) based on the Dreaming of Many Worlds and PrivilegedDreamer precedents
- Sharpened the falsifiable prediction: if SIGReg does NOT produce orthogonal concept factorization even at high data diversity, then the Gaussian prior is actively harmful for composition (a strong negative result that would redirect the field)

### Additional Evidence Found

- **Dreaming of Many Worlds (cRSSM)** confirms that DMControl Walker with varying gravity and actuator strength is a practical testbed for contextual world model evaluation
- **Weak-SIGReg** demonstrates that SIGReg's covariance regularization can be decomposed into a mean-matching and covariance-matching component, which may help isolate whether it is the covariance constraint (pushing toward orthogonality) or the marginal constraint (pushing toward Gaussianity) that drives factorization
- **Forced Rendering paper (arXiv:2501.18797)** warns that abstract bottleneck disentanglement is insufficient -- we should also probe whether factorization is maintained in the *representational space* used for prediction, not just the compressed latent code

### Selected Front-Runner

**"Compositional Phase Diagrams for JEPA World Models via Interventional Probing"** -- the merged A+C idea, with B as supporting diagnostic.

This is selected because:
1. It makes the strongest theoretical contribution (connecting SIGReg geometry to compositional factorization theory)
2. It introduces a genuinely new measurement methodology (interventional concept probing for world models)
3. It produces a practical deliverable (the phase diagram) that the community can immediately use
4. It subsumes the project spec's core questions (zero-shot probing, LoRA adaptation, failure modes, generalization boundaries) within a unified framework
5. The computational cost is manageable with LeWM's efficient training

## Phase 5: Final Proposal

### Title

Compositional Phase Diagrams for JEPA World Models: Interventional Probing of Physical Concept Factorization Under SIGReg Regularization

### Hypothesis

LeWM's SIGReg regularization induces approximately linear, orthogonal physical concept subspaces in the latent space, enabling compositional generalization to unseen factor combinations, but only above a critical data diversity threshold. Below this threshold, concept representations become entangled and interventional independence breaks down, predicting a measurable compositional phase boundary in the space of physical factor coverage.

Specifically:
- H1: At high combinatorial coverage, interventions on gravity-aligned latent directions change predicted gravitational effects without altering friction-dependent or mass-dependent predictions (and symmetrically for other concepts).
- H2: Below a critical combinatorial coverage ratio c*, the principal angle between concept subspaces drops sharply, interventional independence degrades, and zero-shot planning success on held-out combinations collapses.
- H3: LoRA adaptation efficiency at the phase boundary reveals that the predictor (not the encoder) is the primary bottleneck for cross-domain generalization, because the encoder learns domain-general visual features while the predictor must compose dynamics.

### Motivation

World models are increasingly central to embodied AI, yet we lack a principled framework for understanding when they can compose learned physical knowledge to handle novel situations. The compositional generalization theory (Uselis et al., 2026) has established *necessary geometric conditions* for composition, and neuroscience (Knopp et al., 2025) has shown predictive learning autonomously induces compositional structure in biological networks. But nobody has tested these predictions in state-of-the-art JEPA world models. The SIGReg regularizer -- which enforces an isotropic Gaussian distribution -- creates a unique theoretical testbed: does the Gaussian constraint help or hinder the linear orthogonal factorization that composition demands?

This question has immediate practical implications. If the phase boundary can be predicted from data statistics, engineers can determine *in advance* whether their training data is sufficient for a world model to generalize compositionally -- avoiding expensive trial-and-error. If the Gaussian prior helps, it validates the JEPA/SIGReg design philosophy; if it hurts, it suggests specific architectural modifications (e.g., group-structured latent spaces from Delliaux et al., 2025) as alternatives.

### Method

**1. Factored Physical Benchmark (CompoPhys-DMC)**

Construct a systematically factored benchmark using DMControl environments with MuJoCo XML parameter variation:
- 3 physical factors: gravity scale (0.5x, 1.0x, 1.5x, 2.0x), friction coefficient (0.3, 0.6, 1.0, 1.5), object mass (0.5x, 1.0x, 1.5x, 2.0x)
- Full factorial: 4 x 4 x 4 = 64 configurations
- Training subsets at 5 combinatorial coverage levels: 20%, 40%, 60%, 80%, 100% of combinations
- Hold-out splits stratified to ensure novel *combinations* (not novel *levels*) at each coverage level
- Environments: DMControl Walker (gravity + friction), DMControl Cartpole (gravity + mass), and a custom Pusher (friction + mass + gravity) for 3-factor interaction

**2. Training Protocol**

For each coverage level, train LeWM from the official codebase:
- Encoder: ViT-Tiny (5M params), Predictor: Transformer (10M params)
- SIGReg regularization with lambda=1.0 (default)
- Training: 200 epochs per configuration, ~2h per run on single GPU
- Ablation: one run per coverage level *without* SIGReg (vanilla VIC loss) as control
- Total: 5 coverage levels x 2 regularizers x 3 environments = 30 training runs

**3. Observational Probing (Baseline Metrics)**

For each trained model:
- Linear probing: Freeze encoder, train linear heads to predict each physical state variable (position, velocity, gravity, friction, mass) from latent embeddings
- Non-linear probing: 2-layer MLP heads for the same variables
- Compute per-concept R-squared on in-distribution vs. held-out combinations

**4. Geometric Analysis**

- Extract concept subspaces: For each physical concept (gravity, friction, mass), compute the directions in latent space along which that concept varies, using the weight vectors of the trained linear probes
- Principal angle analysis: Measure angles between concept subspace pairs (Uselis et al. metric)
- Factorization score: Compute the linear factorization index (ratio of between-concept to within-concept variance explained)

**5. Interventional Probing (Novel Metric)**

For each concept C and each test trajectory:
1. Encode the trajectory to get latent sequence z_1, ..., z_T
2. Project z_t onto concept-C's probe direction to get the concept value
3. Shift z_t along concept-C's direction by delta (corresponding to a known change in the physical parameter, e.g., gravity 1.0x -> 1.5x)
4. Use the predictor to roll out from the modified z_t
5. Measure: (a) does the predicted trajectory change in the expected way for concept C? (b) do predicted values for concepts D, E remain unchanged?
6. **Interventional Independence Score (IIS)**: For each concept pair (C, D), IIS(C,D) = 1 - |correlation between intervention on C and change in D's prediction|. Perfect factorization yields IIS = 1.0 for all off-diagonal pairs.

**6. Phase Boundary Mapping**

- Plot IIS, principal angles, probing R-squared, and zero-shot planning success as functions of combinatorial coverage
- Fit a sigmoid to each metric to estimate the critical threshold c*
- Compare c* across metrics to test whether geometric factorization precedes, coincides with, or follows behavioral generalization
- Compare SIGReg vs. no-SIGReg models to isolate the regularizer's effect on the phase boundary

**7. LoRA Diagnostic (Supporting)**

At the phase boundary (coverage ~ c*):
- Apply LoRA (rank 4, 8, 16) to encoder-only, predictor-only, and both
- Fine-tune on 10, 50, 100 samples from held-out combinations
- Measure recovery of IIS and planning success
- Compare adaptation efficiency across components to localize the generalization bottleneck

### Cross-Domain Insight

The core transplanted principle is from **statistical physics phase transitions** + **causal interpretability**:

1. **Phase transitions (statistical physics -> representation learning)**: Just as magnetic materials undergo spontaneous symmetry breaking above a critical temperature, producing ordered (factorized) phases, we hypothesize that neural representations undergo spontaneous factorization above a critical data diversity threshold. The order parameter is the Interventional Independence Score, and the control parameter is combinatorial coverage. The structural correspondence holds because both systems involve a high-dimensional state space with a symmetry (permutation invariance of physical concepts) that can be spontaneously broken by the data distribution.

2. **Causal intervention (mechanistic interpretability -> world model evaluation)**: LLM interpretability uses activation patching to identify which components causally mediate specific behaviors. We transplant this to world model latent spaces: instead of patching activations to test whether a circuit mediates a linguistic behavior, we patch latent dimensions to test whether a subspace mediates a physical concept. The structural correspondence holds because both involve testing causal independence of factored representations via targeted perturbation.

### Experimental Plan

| Experiment | Metric | Baselines | Falsification criterion |
|-----------|--------|-----------|------------------------|
| Observational probing (coverage sweep) | R-squared on held-out combinations | Random encoder, DINO-WM frozen encoder | R-squared < random encoder on held-out at low coverage |
| Geometric analysis (coverage sweep) | Principal angles between concept subspaces | Untrained model, no-SIGReg model | Angles not significantly different from no-SIGReg |
| Interventional probing (coverage sweep) | IIS for all concept pairs | Random perturbation baseline, no-SIGReg model | IIS not significantly above random perturbation baseline |
| Phase boundary detection | Sigmoid fit to IIS vs. coverage | Null model: linear relationship (no phase transition) | AIC of sigmoid not significantly better than linear |
| SIGReg ablation | All metrics above | Same model without SIGReg | SIGReg and no-SIGReg produce same phase boundary |
| LoRA diagnostic | Adaptation efficiency (samples to 90% IIS recovery) | Encoder-only vs. predictor-only vs. both | No significant difference between encoder and predictor adaptation |

### Resource Estimate

- **Training**: 30 LeWM training runs x ~2h each = ~60 GPU-hours on single GPU (RTX 3090 or equivalent)
- **Probing + geometric analysis**: ~5h total (linear algebra on frozen embeddings)
- **Interventional probing**: ~10h (predictor rollouts from perturbed latents, parallelizable)
- **LoRA diagnostic**: ~10h (12 conditions x ~50 min each)
- **Total**: ~85 GPU-hours, achievable in ~4 days on a single GPU or ~1 day with 4 GPUs
- **Model size**: ViT-Tiny encoder (5M) + Transformer predictor (10M) = 15M params total
- **Individual experiment task**: Each training run ~2h; each probing/intervention batch ~30 min. All within 1-hour task target.

### Risk Assessment

1. **Risk: LeWM's latent space is too entangled for clean concept separation even at 100% coverage**
   - Likelihood: Medium (the Two-Room failure suggests SIGReg may over-constrain low-complexity representations)
   - Mitigation: If factorization never emerges, this is a *strong negative result* that directly challenges the JEPA/SIGReg design. We pivot to testing whether group-structured latent spaces (Delliaux et al.) or RCC architectures (Schwarcz, 2026) fare better under the same protocol. The evaluation framework itself remains valuable.
   - Impact: Would redirect the field toward architectural solutions for compositional world models.

2. **Risk: Interventional probing is confounded by correlated concept dimensions**
   - Likelihood: Medium-high (physical concepts like gravity and mass jointly affect trajectories)
   - Mitigation: Use orthogonalized probe directions (Gram-Schmidt on concept probe vectors). Validate that the probe directions are sufficiently accurate (>0.9 R-squared) before running interventions. Include a "random direction" baseline to establish the noise floor.
   - Impact: If confounded, IIS becomes unreliable, and we fall back to purely geometric analysis (principal angles), which is still informative but less novel.

3. **Risk: No sharp phase transition -- the relationship between coverage and factorization is gradual**
   - Likelihood: Medium (phase transitions in finite systems are typically crossovers)
   - Mitigation: Even a gradual transition is informative if we can identify the coverage level at which factorization *begins* to emerge (using an inflection point rather than a sharp threshold). The "phase diagram" framing remains useful as a visualization tool even without a sharp critical point. We soften claims from "phase transition" to "compositional phase diagram" to accommodate gradual crossovers.
   - Impact: Reduces the theoretical elegance but preserves the practical utility of the framework.

### Novelty Claim

This proposal is novel in three specific ways, each supported by evidence that it has not been done before:

1. **Interventional probing of physical concept independence in world model latent spaces**: Prior probing work (LeWM, DINO-WM, AIM/V-JEPA 2) is purely observational -- training regression heads on frozen embeddings to predict physical states. No prior work has *intervened* on latent dimensions to test whether physical concepts are causally independent in the representation. The closest work is CausalARC (arXiv:2509.03636), which applies causal interventions to abstract reasoning, not to continuous physical world models.

2. **Empirical test of the SIGReg-orthogonality connection**: The theoretical community has established that compositional generalization requires linear, orthogonal representations (Uselis et al., 2026). SIGReg enforces isotropic Gaussian distributions (Balestriero et al., 2025). Whether the Gaussian constraint promotes or hinders the required orthogonal structure is a completely open empirical question. No prior work has measured the factorization geometry of SIGReg-regularized latent spaces as a function of data diversity.

3. **Compositional phase diagram for world models**: The concept of mapping generalization boundaries as a function of combinatorial data coverage, producing a "phase diagram" in factor space, has not been applied to world models. The closest work is Redhardt et al. (2025) showing data diversity drives compositional generalization in vision models, but they do not map the transition curve, do not use interventional metrics, and do not study world models.
