# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: LeWM's SIGReg-enforced isotropic Gaussian latent space naturally supports compositional generalization**
   - Evidence challenging it: The isotropic Gaussian prior is proven optimal only for minimizing worst-case risk over *linear probes* (LeJEPA, arXiv:2511.08544). This says nothing about compositional generalization specifically. In the VAE literature, isotropic Gaussian priors are widely recognized as overly restrictive -- they over-smooth latent spaces, cause posterior collapse, and fail to capture multimodal structure (PH-VAE, arXiv:2502.02856; GM-VAE, arXiv:2511.21883). The compositional generalization theory (Uselis et al., arXiv:2602.24264) requires *linear, orthogonal* per-concept representations -- which is a geometric constraint on factor *alignment*, not on the marginal distribution being Gaussian. A latent space can be perfectly isotropic Gaussian yet have entangled factors; conversely, a non-Gaussian anisotropic space could have perfectly orthogonal factor subspaces.

2. **Assumption: Linear probing accuracy on physical states demonstrates that LeWM "understands" physics and these representations will transfer compositionally**
   - Evidence challenging it: The ICLR 2024 work on spatial representations in LLMs (arXiv:2310.02207) showed that the probe itself can learn linear combinations of simpler features, giving a misleading impression that the model has rich internal representations. The COLM 2024 paper demonstrated that probes may capture correlates rather than the target concept itself. The survey on Representation Engineering (arXiv:2502.17601) explicitly notes that "probing accuracy alone may not capture the full picture" and "naive geometric assumptions can lead to misleading conclusions." In the world model context, high probing accuracy for agent position or block pose within the training distribution tells us nothing about whether these representations decompose compositionally across domains.

3. **Assumption: "Cross-domain compositional generalization" is a well-defined and testable property for LeWM**
   - Evidence challenging it: LeWM was trained and evaluated on PushT (2D push manipulation), TwoRoom (navigation), and OGBench-Cube (6-DOF manipulation). These are *different tasks*, not parametric variations of a shared physical domain. The concept of "compositional generalization" requires independently varying factors that can be recombined -- e.g., gravity x friction x object shape. But PushT's physics (2D quasi-static contact) shares almost no physical vocabulary with OGBench-Cube's (3D arm kinematics + gravity). Calling transfer between these "compositional generalization of physical concepts" is a category error. It is closer to multi-task transfer, which is a fundamentally different (and much harder) problem.

4. **Assumption: LoRA adaptation can serve as a meaningful "generalization diagnostic" for world models**
   - Evidence challenging it: "LoRA Learns Less and Forgets Less" (TMLR 2024) showed that LoRA substantially underperforms full fine-tuning, especially for domain-shifting tasks where the model needs to significantly deviate from its training distribution. The ICLR 2025 paper "LoRA vs Full Fine-tuning: An Illusion of Equivalence" demonstrated that LoRA and full fine-tuning access fundamentally different parts of the solution space. If LoRA cannot recover performance, this could mean (a) the model lacks transferable knowledge, OR (b) LoRA's low-rank constraint is simply insufficient to reparameterize the needed adaptation. These explanations are confounded and cannot be disentangled without additional controls (e.g., varying rank, comparing to full fine-tuning, comparing to training from scratch).

5. **Assumption: Video generation models / world models that produce visually realistic rollouts have learned physical principles**
   - Evidence challenging it: The Physics-IQ benchmark (arXiv:2501.09038) demonstrated that "visual realism does not imply physical understanding" across Sora, Runway, Pika, and other models. The ICML 2025 PhyWorld study showed that AI models "do not truly learn general physics principles but instead rely heavily on remembering specific examples." The RBench benchmark (arXiv:2601.15282) showed consumer-oriented models perform poorly on physically-grounded tasks despite high visual quality. If even large-scale video models fail at physical understanding, why should a 15M-parameter JEPA trained on single-environment demonstrations fare better?

6. **Assumption: Disentangled representations are sufficient for compositional generalization**
   - Evidence challenging it: Liang et al. (arXiv:2501.18797) directly demonstrated that even with *fully disentangled (x,y) inputs*, standard generative architectures fail in OOD regions because subsequent layers re-entangle representations. They showed that "disentangled latents in an abstract representation are insufficient" -- the model must be *forced* to render these latents into the output space. The ICLR 2025 kernel theory paper (arXiv:2405.16391) showed that for compositional tasks with spurious correlations, model behavior is highly sensitive to architectural hyperparameters, and compositionally structured models are fundamentally limited to "conjunction-wise additive" functions.

7. **Assumption: The Two-Room failure is an isolated edge case explained by "low diversity"**
   - Evidence challenging it: The LeWM paper offers a hypothesis (low intrinsic dimensionality makes the Gaussian constraint hard to satisfy) but no controlled experiment varying diversity while holding other factors constant. This failure may actually reveal a fundamental tension: SIGReg forces the latent distribution toward high-dimensional isotropy, but many physically meaningful environments have low-dimensional structure. If physical state spaces are inherently low-rank (as they often are -- a 2D navigation task lives on a 2-manifold), then forcing high-dimensional isotropy could systematically destroy the task-relevant structure. This would predict failures not just in Two-Room but in any environment with low intrinsic dimensionality.

8. **Assumption: LeWM's 48x speed advantage over foundation-model-based world models comes without representational cost**
   - Evidence challenging it: LeWM uses a ViT-Tiny encoder (~5M parameters). V-JEPA 2 uses ViT-H (632M parameters) trained on 1M+ hours of video. DINOv2 (used in DINO-WM) uses ViT-G (1.1B parameters) pre-trained on 142M images. The representational capacity gap is enormous (100-200x). If compositional generalization requires learning factored representations of multiple physical concepts, a 5M-parameter encoder may simply lack the capacity to develop separate, linearly independent subspaces for each concept. The "efficiency" narrative may be fundamentally at odds with the "generalization" narrative.

### Landscape of Doubt

The core landscape of doubt centers on a **triple conflation** in the project proposal:

1. **Conflating "encoding physical states" with "understanding physics"**: High linear probing accuracy for agent/block position is a necessary but nowhere near sufficient condition for physical understanding. A lookup table would achieve 100% probing accuracy without any physical reasoning.

2. **Conflating "multi-task transfer" with "compositional generalization"**: Transfer from PushT to OGBench is multi-task transfer across fundamentally different physical systems, not compositional recombination of shared physical concepts. True compositional generalization requires shared primitives that can be independently varied and recombined.

3. **Conflating "LoRA adaptation efficiency" with "transferable knowledge"**: Low LoRA adaptation efficiency could indicate either (a) absence of transferable knowledge or (b) LoRA's intrinsic inability to capture the needed adaptation. Without controlling for LoRA's known limitations, the diagnostic is confounded.

**The most uncomfortable question**: What if LeWM's success on its evaluation benchmarks comes entirely from *task-specific pattern matching* rather than learning any general physical principles? The 15M parameter budget, single-environment training regime, and task-specific evaluation suite are all consistent with a model that has learned good task heuristics without any compositionally generalizable physical knowledge.

---

## Phase 2: Initial Candidates

### Candidate A: "The Probing Delusion: Why Linear Probing Accuracy in JEPA World Models Does Not Predict Cross-Domain Transfer"

- **Challenged assumption**: That high linear probing accuracy for physical states within the training distribution implies the model has learned transferable physical representations.
- **Evidence against**: (1) Probes can learn to compose simpler features that the model itself doesn't represent (arXiv:2310.02207). (2) Probing accuracy within-distribution is trivially achievable by any sufficiently smooth function approximator -- it measures *encoding* not *understanding*. (3) No prior work has tested whether in-distribution probing accuracy predicts out-of-distribution probing accuracy for JEPA world models. (4) The compositional generalization literature (arXiv:2602.24264) shows that the *geometric structure* of representations (orthogonality, divisibility) matters, not just their decodability.
- **Contrarian hypothesis**: In-distribution linear probing accuracy is a poor predictor of compositional transfer. Models with identical probing accuracy can have vastly different compositional generalization performance, because probing measures marginal decodability while composition requires factorial structure.
- **Exploitation plan**: Train LeWM on multiple DMControl environments with parametrically varied physics (gravity, friction, damping). Measure probing accuracy within-distribution and cross-domain. Show that high in-distribution probing accuracy does not predict cross-domain probing accuracy. Introduce geometric metrics (principal angle analysis, linear factorization scores from arXiv:2602.24264) that *do* predict transfer.
- **Novelty estimate**: 8/10

### Candidate B: "The Isotropy-Compositionality Tension: Why SIGReg May Actively Harm Compositional Generalization"

- **Challenged assumption**: That SIGReg's enforcement of isotropic Gaussian latents helps (or at least doesn't hurt) compositional generalization.
- **Evidence against**: (1) Isotropic Gaussian optimality is proven only for worst-case risk over linear probes, not for compositional transfer (arXiv:2511.08544). (2) In VAEs, isotropic Gaussian priors are known to over-smooth and destroy structured latent geometry. (3) Compositional generalization requires orthogonal per-concept subspaces (arXiv:2602.24264), but isotropy provides no guarantee of factor alignment -- it only ensures uniform variance. (4) LeWM's Two-Room failure suggests SIGReg actively harms low-dimensional tasks, which are common in physical environments.
- **Contrarian hypothesis**: SIGReg's isotropic constraint creates a fundamental tension with compositional structure. By forcing all latent dimensions to have equal variance, it prevents the model from allocating disproportionate representational capacity to dominant physical factors (e.g., allocating more dimensions to contact physics than to background color). This "representational egalitarianism" actually *de-factorizes* the latent space by spreading each physical concept across all dimensions uniformly.
- **Exploitation plan**: Compare LeWM with SIGReg vs. ablated variants (no regularizer, VICReg, anisotropic Gaussian) on compositional generalization benchmarks. Measure the linear factorization structure (per-concept orthogonality) of each variant's latent space. Show that SIGReg produces latent spaces with *lower* factorization scores than alternatives.
- **Novelty estimate**: 9/10

### Candidate C: "Compositional Generalization is a Category Error for Single-Environment JEPA World Models"

- **Challenged assumption**: That it is meaningful to study "compositional generalization of physical concepts" in models trained on single environments like PushT.
- **Evidence against**: (1) PushT involves only 2D quasi-static contact physics -- there is no gravity, no fluid dynamics, no deformation. The "physical concepts" are limited to agent-block contact and friction. (2) True compositional generalization requires multiple independently varying factors that can be recombined (arXiv:2602.24264). A single environment provides one fixed combination. (3) The CompWoB benchmark (ICLR 2024) showed that even LLMs suffer 70% performance drops on compositional tasks despite 94% on base tasks -- and those at least had multiple training tasks to compose.
- **Contrarian hypothesis**: Single-environment JEPA world models like LeWM learn task-specific predictive heuristics, not compositionally generalizable physical concepts. The concept of "compositional generalization" is only meaningful when the training set includes multiple independent physical factors that the model can learn to separate. Without multi-factor training data, there is nothing to compose.
- **Exploitation plan**: Design a controlled experiment with DMControl environments parameterized by 3 independent physical factors (gravity, friction, damping). Show that (a) LeWM trained on a single factor combination has no compositional transfer, (b) LeWM trained on multiple single-factor variations develops partial factored structure, (c) only with sufficient combinatorial coverage does compositional generalization emerge -- replicating the vision result of arXiv:2507.07102 in the world model setting.
- **Novelty estimate**: 7/10

---

## Phase 3: Self-Critique

### Against Candidate A: "The Probing Delusion"

- **Steelman**: LeWM's probing results are not limited to linear probes -- they also include non-linear MLP probes. If non-linear probes also achieve high accuracy, the argument that "the probe is doing the work" is weakened because MLP probes have more capacity to reveal genuine representational structure. Furthermore, the AIM framework (arXiv:2603.20327) uses a passive quantization probe that is closer to a structural analysis than a learned decoder. The fact that multiple probe architectures agree increases confidence that the representations are genuinely informative.

- **Cherry-picking check**: I am selectively citing the skeptical probing literature while ignoring substantial evidence that probing works well in practice. In the NLP interpretability literature, linear probes have successfully predicted model behaviors in many settings. The concern about probes "doing the work" is more acute for complex probes on high-dimensional spaces; for low-dimensional physical states (2D position), linear probing is a much more reliable indicator.

- **Confounding check**: There is a confound between "probe complexity" and "representation quality." The claim that probing accuracy doesn't predict transfer could be tested by comparing models with similar probing accuracy but different architectures. However, if all JEPA variants show similar probing accuracy *and* similar transfer performance, the critique is moot.

- **Actionability check**: This idea leads to a clear experiment (in-distribution vs. cross-domain probing) and a concrete alternative metric (geometric factorization). Even if the negative result is modest, the positive contribution (better metrics for predicting transfer) is valuable.

- **Verdict**: **MODERATE**. The core concern is valid but the specific claim about LeWM may be overstated for low-dimensional physical states. The positive contribution (geometric metrics) is strong.

### Against Candidate B: "The Isotropy-Compositionality Tension"

- **Steelman**: The LeJEPA paper (arXiv:2511.08544) provides rigorous theoretical justification for isotropic Gaussian embeddings minimizing worst-case downstream risk. If compositional generalization is viewed as one of many possible downstream tasks, then isotropy is at worst neutral -- it optimizes for the hardest possible downstream task, which should include compositional tasks. Furthermore, isotropy does NOT preclude factorial structure: a latent space can be both isotropic (equal variance along all axes) and factored (each concept aligned with a different axis). Isotropy constrains the *variance structure* while factorization constrains the *alignment structure* -- these are mathematically independent.

- **Cherry-picking check**: I am citing VAE limitations of the Gaussian prior, but the JEPA context is fundamentally different. In VAEs, the prior is imposed on a generative model that must decode back to pixel space; in JEPAs, the constraint is on a discriminative predictor that operates entirely in latent space. The pathologies of the Gaussian prior in VAEs (posterior collapse, over-smoothing) may not transfer to the JEPA setting where there is no decoder.

- **Confounding check**: The Two-Room failure could have many causes beyond SIGReg. The paper suggests low diversity as the cause. Without an ablation removing SIGReg in Two-Room (and showing recovery), attributing the failure to SIGReg specifically is speculative.

- **Actionability check**: The ablation experiment (SIGReg vs. alternatives on compositional benchmarks) is well-defined and highly informative regardless of outcome. If SIGReg *does* harm compositionality, this is a significant finding. If it doesn't, the negative result still clarifies the field.

- **Verdict**: **MODERATE-STRONG**. The mathematical independence argument (isotropy vs. factorization) weakens the strongest version of the claim, but the empirical ablation is valuable regardless. The key insight -- that isotropy is orthogonal to factorization -- is itself an important clarification.

### Against Candidate C: "Compositional Generalization is a Category Error"

- **Steelman**: Even within a single environment like PushT, there are multiple physical factors that vary: agent position, block position, block orientation, contact dynamics at different angles. The model must predict across all these combinations within the environment. This is a form of within-environment compositional generalization (position x orientation x contact angle). Furthermore, the project proposal explicitly plans to train on multiple environments and test on held-out combinations -- it is not proposing to study composition within a single environment.

- **Cherry-picking check**: I am attacking a strawman version of the proposal. The actual proposal acknowledges the need to "construct cross-domain combination test sets" with novel combinations of physical factors. The critique is valid for evaluating LeWM *as published* but not for the proposed experimental design.

- **Confounding check**: The distinction between "multi-task transfer" and "compositional generalization" is real but may be less sharp than I claim. If tasks share physical primitives (gravity, contact), then transfer between them *is* a form of compositional generalization over those shared primitives.

- **Actionability check**: The constructive proposal (training with sufficient combinatorial coverage) essentially reduces to the same experiment the optimists would propose, just motivated differently. The unique contribution is the prediction that single-environment training fails at composition, which is somewhat obvious.

- **Verdict**: **WEAK-MODERATE**. The critique of the "category error" is partially a strawman of the proposal. The constructive element (emphasizing combinatorial coverage in training) is valid but not surprising.

---

## Phase 4: Refinement

### Dropped
- **Candidate C** ("Category Error") was partially a strawman of the actual proposal and its unique contribution was limited. The core insight (need for multi-factor training data) is already implicit in the proposal.

### Strengthened

**Candidate A** and **Candidate B** are merged and strengthened into a unified proposal that attacks the *methodology* of evaluating compositional generalization in JEPA world models:

**Core refined claim**: The standard evaluation pipeline for JEPA world models (train on environment X, probe latent space, report accuracy) has a fundamental validity gap for compositional generalization claims. Specifically:

1. **Probing accuracy is necessary but not sufficient**: High probing accuracy within the training distribution does not predict cross-domain compositional transfer. The *geometric structure* of the latent space (factorization, orthogonality) is the relevant diagnostic, not accuracy.

2. **SIGReg's relationship to compositionality is an open question, not an assumed benefit**: The isotropic Gaussian constraint is mathematically orthogonal to the factor alignment constraint required for compositional generalization. It could help, hurt, or be irrelevant -- this must be empirically determined, not assumed.

3. **The adaptation gap (zero-shot vs. LoRA vs. full fine-tuning) must be carefully controlled**: LoRA's known limitations in domain-shifting tasks mean that a large "zero-shot to LoRA" performance gap could reflect either (a) genuine lack of transferable knowledge or (b) LoRA's intrinsic inability to adapt. Only with full fine-tuning and training-from-scratch controls can these be disentangled.

**Additional corroboration**: The PNAS 2024 paper on "Decomposing dynamical subprocesses for compositional generalization" supports the idea that compositional generalization in dynamical systems requires explicitly decomposed subprocess representations, not just monolithic latent encodings.

### Selected Front-runner

The merged and strengthened proposal: **"Rethinking the Evaluation of Compositional Generalization in JEPA World Models: Probing Accuracy, Geometric Structure, and the Isotropy-Factorization Distinction"**

---

## Phase 5: Final Proposal

### Title
**When Probing Passes but Transfer Fails: Rethinking Compositional Generalization Evaluation for JEPA World Models**

### Challenged Assumption
The JEPA world model community (LeWM, DINO-WM, V-JEPA 2) evaluates physical understanding primarily through linear/MLP probing accuracy on latent representations. The implicit assumption is that high probing accuracy for physical state variables indicates that the model has learned transferable physical representations that will compose across domains. We challenge this assumption on three fronts: (1) probing accuracy is a poor predictor of compositional transfer, (2) the isotropic Gaussian constraint (SIGReg) is orthogonal to the geometric conditions required for composition, and (3) LoRA-based adaptation diagnostics are confounded by LoRA's known domain-shift limitations.

### Evidence

**For the conventional view (steelman):**
- LeWM achieves strong linear probing accuracy on physical states (position, velocity, orientation) across PushT and OGBench environments, outperforming DINO-WM substantially.
- The LeJEPA theoretical framework (arXiv:2511.08544) proves that isotropic Gaussian embeddings minimize worst-case risk over downstream linear predictors.
- V-JEPA 2-AC demonstrates zero-shot robotic planning in unseen labs, suggesting some degree of environment transfer.
- The AIM framework (arXiv:2603.20327) shows that physical dimensions are statistically separable in V-JEPA 2 latent space.

**Against the conventional view:**
- Probing accuracy can reflect probe learning rather than model representation quality (ICLR 2024, arXiv:2310.02207; COLM 2024; Representation Engineering Survey arXiv:2502.17601).
- Visual realism does not imply physical understanding (Physics-IQ, arXiv:2501.09038; PhyWorld, ICML 2025).
- Disentangled representations alone are insufficient for compositional OOD generalization -- models re-entangle in subsequent layers (arXiv:2501.18797).
- Compositional generalization requires specific geometric conditions (linear, orthogonal per-concept) that are not guaranteed by isotropy (arXiv:2602.24264).
- LoRA substantially underperforms full fine-tuning for domain-shifting tasks and accesses a fundamentally different solution space (TMLR 2024; ICLR 2025).
- Model behavior on compositional tasks is highly sensitive to architectural hyperparameters (ICLR 2025, arXiv:2405.16391).

### Hypothesis
**In-distribution probing accuracy is a poor predictor of cross-domain compositional transfer in JEPA world models.** Models with identical probing accuracy can have vastly different compositional generalization performance because probing measures marginal decodability while composition requires factorial geometric structure. Furthermore, SIGReg's isotropic Gaussian constraint does not promote (and may actively compete with) the orthogonal per-concept factorization required for compositional generalization.

### Method

**1. Controlled Environment Suite**: Use DMControl environments (Walker, Hopper, Cheetah) parameterized by 3 independent physical factors:
   - Gravity magnitude (0.5x, 1.0x, 2.0x of default)
   - Joint friction coefficient (0.5x, 1.0x, 2.0x)
   - Body density/mass (0.5x, 1.0x, 2.0x)

This gives 27 total combinations. Train on a subset (e.g., 8 combinations covering all individual factor levels) and test on held-out combinations.

**2. Multi-Level Evaluation Protocol**:
   - **Level 1 (Probing)**: Standard linear and MLP probing for physical states within-distribution and cross-domain.
   - **Level 2 (Geometric)**: Measure linear factorization structure using principal angle analysis, CKA, and divisibility scores from Uselis et al. (arXiv:2602.24264).
   - **Level 3 (Causal)**: Counterfactual intervention -- swap specific factor representations between environments and measure prediction consistency.
   - **Level 4 (Behavioral)**: Planning success rate (CEM) on held-out combinations.

**3. SIGReg Ablation**: Compare latent geometry and compositional transfer across:
   - LeWM with SIGReg (isotropic Gaussian)
   - LeWM with VICReg (variance-invariance-covariance)
   - LeWM with no regularizer (collapse-prone baseline)
   - LeWM with GMM prior (multi-modal alternative)

**4. Adaptation Controls**: For the LoRA diagnostic, include:
   - Zero-shot (no adaptation)
   - LoRA rank-4, rank-16, rank-64 (varying capacity)
   - Full fine-tuning (upper bound)
   - Train from scratch on target (absolute ceiling)

This allows separating "lack of transferable knowledge" from "LoRA capacity limitation."

### Experimental Plan
- **Phase 1 (Pilot, ~15 min)**: Train LeWM on 2 DMControl Walker configurations (default, 2x gravity). Probe latent space. Measure cross-config probing accuracy. Verify the setup works.
- **Phase 2 (Main, ~45 min each)**: Train LeWM on 8/27 factor combinations for Walker. Run the full 4-level evaluation protocol. This is the core experiment.
- **Phase 3 (Ablation, ~45 min each)**: Run the SIGReg ablation (4 regularizer variants). Compare geometric factorization scores.
- **Phase 4 (Adaptation Controls, ~30 min each)**: Run the full adaptation hierarchy (zero-shot through train-from-scratch) on 3-4 held-out combinations.

Use ViT-Tiny encoder (matching LeWM's architecture) throughout. Single GPU. Each phase fits within the 1-hour constraint.

### Baselines
- **LeWM** (original SIGReg configuration): The primary subject of analysis.
- **DINO-WM** (frozen DINOv2 encoder): Pre-trained representation baseline -- expected to have different geometric structure due to massive pre-training.
- **Random encoder** (untrained ViT-Tiny): Controls for geometric artifacts of the architecture itself.
- **PLDM** (7-term VIC objective): Alternative JEPA training objective for regularizer comparison.

All baselines properly tuned with their recommended hyperparameters -- no strawman comparisons.

### Risk Assessment
**If the conventional view turns out to be correct** -- i.e., probing accuracy *does* predict compositional transfer and SIGReg *does* promote factorization -- then we still contribute:
1. The first systematic compositional generalization benchmark for JEPA world models.
2. Empirical validation that SIGReg promotes the geometric conditions for composition (a non-obvious positive result).
3. A multi-level evaluation protocol (probing + geometric + causal + behavioral) that is more rigorous than probing alone.
4. Controlled adaptation experiments with proper baselines.

The paper is publishable regardless of direction because the field currently lacks these controlled experiments entirely.

### Novelty Claim
The specific insight that **probing accuracy and compositional generalization are decoupled diagnostics** -- that a model can achieve high probing accuracy without compositional structure, and that the geometric conditions for composition (factor orthogonality, divisibility) are mathematically independent from the conditions optimized by SIGReg (isotropy, Gaussianity) -- has not been articulated or tested in the JEPA world model literature. This work would be the first to:
1. Empirically test the probing-accuracy-to-composition-transfer gap in world models.
2. Directly measure whether SIGReg promotes or hinders factor orthogonality.
3. Apply the geometric compositionality framework (arXiv:2602.24264) to JEPA world models rather than static vision encoders.
4. Provide a properly controlled LoRA adaptation experiment that separates model limitations from adapter limitations.
