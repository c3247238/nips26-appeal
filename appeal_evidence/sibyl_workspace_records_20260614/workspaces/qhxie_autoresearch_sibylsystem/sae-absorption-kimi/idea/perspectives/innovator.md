# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Chanin et al., 2024. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507 (NeurIPS 2025 Oral)** — Foundational paper identifying absorption as sparsity-driven failure mode; first-letter probe metric; analytical proof of absorption incentive.

2. **Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547 (ICML 2025)** — Nested dictionaries reduce absorption 10x (0.49 to 0.05) but introduce feature hedging in inner levels with +50% compute overhead.

3. **Korznikov et al., 2025. "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033** — Chunk-wise orthogonality penalty reduces absorption 65% with only ~4-11% compute overhead.

4. **Engels et al., 2024/2025. "Decomposing The Dark Matter of Sparse Autoencoders." ICLR 2025** — ~50% of SAE error vector and >90% of its norm are linearly predictable from original activations, meaning SAEs systematically fail to capture learnable features. Nonlinear error stays constant with scale.

5. **O'Neill et al., 2025. "Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders." arXiv:2508.09363** — Domain-specific SAEs explain up to 20% more variance than general-domain SAEs by reducing linear dark matter. The residual becomes more purely nonlinear.

6. **Nasiri-Sarvi et al., 2025. "SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability." arXiv:2507.06265 (TMLR 2026)** — Global TopK + cross-reconstruction loss achieves 0.80 Jaccard similarity for cross-model concept alignment, eliminating dead neurons across models.

7. **Lindsey et al., 2024/2026. "Sparse Crosscoders for Diffing MoEs and Dense Models." arXiv:2603.05805** — BatchTopK crosscoders with explicit shared/unique feature designation achieve ~87% fractional variance explained. MoEs learn fewer unique features than dense models.

8. **Hu et al., 2025. "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717 (NeurIPS 2025 Workshop)** — Feature sensitivity declines with increasing SAE width; many interpretable features have poor sensitivity.

9. **Tang et al., 2025. "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability." arXiv:2512.05534** — First unified framework for SAEs, transcoders, and crosscoders; explains absorption and dead neurons via spurious optima analysis.

10. **Narayanaswamy et al., 2026. "Improving Robustness in Sparse Autoencoders via Masked Regularization." arXiv:2604.06495** — Masking-based regularization disrupts co-occurrence patterns, reducing absorption 2.35-3.75% and improving OOD generalization.

11. **Li & Ren, 2025. "Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training." arXiv:2510.08855** — ATM tracks activation magnitudes/frequencies via exponential moving averages, achieving lower absorption than TopK and JumpReLU.

12. **Cui et al., 2025. "On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy." arXiv:2506.15963** — Closed-form proof that full feature recovery is only guaranteed under extreme sparsity (approaching 1-sparse); proposes WSAE with adaptive reweighting.

### Landscape Summary

The SAE field in 2025-2026 is characterized by three converging crises that create a unique opportunity for unconventional thinking:

**Crisis 1: The Measurement-Validity Crisis.** Our own prior work (iter_002-004) and independent papers (Kantamneni et al., 2025; Lieberum et al., 2024) have shown that SAE metrics---including absorption---are confounded by geometry, co-occurrence, and probe artifacts. The community has pivoted to ground-truth synthetic data (SynthSAEBench) as a partial solution, but this creates a synthetic-to-real gap.

**Crisis 2: The Dark Matter Crisis.** Engels et al. (2025) proved that SAEs leave substantial structured error unexplained---error that is linearly predictable and therefore learnable. This means SAEs are systematically incomplete, and absorption may be a symptom of this incompleteness rather than an independent pathology.

**Crisis 3: The Cross-Model Ontology Crisis.** SPARC (2025) and crosscoder work (2026) reveal that different models encode the same concepts in incompatible ways. If absorption is model-dependent (as our iter_002-004 found), then absorption metrics may not generalize across models---a profound challenge for a field that treats absorption as a universal property.

These three crises share a common thread: they all suggest that the current framing of absorption as a "pathology to be eliminated" may be fundamentally wrong. What if absorption is not a bug but a symptom of deeper structural properties of sparse linear decomposition? What if the community's focus on reducing absorption has blinded it to more fundamental questions about what SAEs can and cannot represent?

---

## Phase 2: Initial Candidates

### Candidate A: The Absorption-Dark Matter Duality: Absorption as a Symptom of Systematic Representation Failure

- **Hypothesis**: Feature absorption and dark matter (unexplained variance) are dual manifestations of the same underlying limitation: SAEs cannot simultaneously represent all features in a hierarchy due to the linearity constraint. When the SAE's latent budget is insufficient, it faces a choice: (a) absorb parent features into children (absorption), or (b) leave features unrepresented (dark matter). The trade-off between absorption and dark matter is governed by the SAE's architectural choices, and current architectures systematically favor absorption over dark matter because absorption preserves reconstruction fidelity while dark matter does not. A crosscoder-based analysis across multiple SAE instantiations can quantify this trade-off and reveal that absorption-reducing architectures (Matryoshka, OrtSAE) simply shift the failure mode from absorption to dark matter without reducing total representation error.

- **Cross-domain insight**: From **wave-particle duality in quantum mechanics**. In quantum mechanics, light exhibits both wave-like and particle-like properties, but measuring one more precisely reduces certainty about the other (Heisenberg uncertainty). In SAEs, the "dual" is absorption vs. dark matter: reducing absorption (by forcing the SAE to represent parent features independently) increases dark matter (by pushing other features out of the latent budget). The cross-domain leap is applying complementarity principles to SAE failure modes: the two pathologies are not independent but complementary aspects of a single representational limit.

- **Evidence for**: (1) Engels et al. (2025) found that linear dark matter is substantial and learnable, meaning SAEs are systematically leaving features unrepresented. (2) Our pilot results show that TopK reduces absorption by 84% but MCC remains near chance (~0.21), suggesting features are not being "recovered"---they're being handled differently. (3) Matryoshka SAEs reduce absorption but increase hedging (Chanin et al., 2025), showing that reducing one pathology shifts to another. (4) The dark matter framework provides exact metrics (linear predictability of error) that can quantify the trade-off. (5) No prior work has connected absorption to dark matter---they are treated as separate research threads.

- **Novelty estimate**: 9/10 --- The wave-particle duality analogy is genuinely novel in SAE research. The core empirical claim (that absorption reduction shifts error to dark matter) would be the first systematic test of this trade-off. It reframes the entire absorption debate from "how do we eliminate absorption?" to "how do we optimally allocate representational error between absorption and dark matter?"

### Candidate B: Crosscoder Absorption Tomography: Mapping How Absorption Propagates Across Model Architectures and Training Stages

- **Hypothesis**: Feature absorption is not a property of individual SAEs but a relational property that emerges from how features are distributed across model architectures and training stages. By training crosscoders between SAEs trained on different architectures (Standard, TopK, Matryoshka) or different training checkpoints, we can construct an "absorption tomography"---a 3D map showing how parent features are encoded, split, absorbed, or lost across the architectural/training space. This reveals that absorption is not a scalar property but a tensor field: a parent feature may be fully represented in one architecture, absorbed in another, and lost to dark matter in a third. The crosscoder alignment score between architectures predicts their absorption difference better than any single-architecture metric.

- **Cross-domain insight**: From **medical imaging tomography** (CT, MRI, PET). In CT scans, X-rays are taken from multiple angles and reconstructed into a 3D volume. Each 2D projection alone is insufficient; the 3D structure emerges only from combining projections. The cross-domain leap is: just as CT combines multiple 2D projections to reconstruct 3D anatomy, crosscoder absorption tomography combines multiple SAE "projections" (different architectures, checkpoints) to reconstruct the full feature encoding landscape. A feature that appears "absorbed" in one architecture may be "visible" in another---just as a tumor hidden in one CT angle is visible in another.

- **Evidence for**: (1) SPARC (2025) proved that cross-model SAE alignment is possible and meaningful, achieving 0.80 Jaccard similarity. (2) The MoE-dense crosscoder work (2026) found that different architectures learn systematically different feature distributions. (3) Our own iter_002-004 found that absorption varies dramatically across architectures (GPT-2 vs. Pythia) and even across trainers within the same architecture. (4) The crosscoder latent diffing work (2025) showed that BatchTopK reduces false unique latent attributions by >3x, enabling more reliable cross-architecture comparison. (5) No prior work has used crosscoders specifically to compare absorption patterns across architectures.

- **Novelty estimate**: 8/10 --- Crosscoders exist but have not been applied to absorption analysis. The "tomography" framing is novel and provides an intuitive metaphor for a complex multi-dimensional problem. The empirical prediction (crosscoder alignment predicts absorption difference) is testable and would be the first architecture-comparison method that does not rely on probe-based metrics.

### Candidate C: The Curriculum Absorption Hypothesis: Absorption as a Developmental Stage in SAE Training

- **Hypothesis**: Feature absorption is not a permanent failure mode but a transient developmental stage in SAE training, analogous to how children overgeneralize categories before refining them. During early training, SAEs learn coarse parent features; during mid-training, sparsity pressure forces specialization into child features, causing apparent absorption of parents; during late training, the SAE recovers parent features through composition of child features or through the emergence of dedicated parent latents. If this hypothesis holds, then "absorption" as measured at convergence is merely a snapshot of a dynamic process, and the optimal intervention is not architectural modification but curriculum learning that controls when specialization occurs. The key prediction is that parent-feature recovery (measured via causal intervention) improves during late training even as probe-based absorption scores remain constant or worsen.

- **Cross-domain insight**: From **developmental psychology** (Piaget's stages of cognitive development) and **neural development** (critical periods). In child development, children initially overgeneralize (calling all four-legged animals "dog") before refining categories. In neural development, visual cortex neurons initially have broad receptive fields that sharpen during critical periods. The cross-domain leap is: SAE training may recapitulate this developmental trajectory. Early training = broad receptive fields (parent features); mid-training = sharpening/specialization (child features emerge, parents appear absorbed); late training = hierarchical organization (parents recover through composition). The critical insight from developmental psychology is that forcing early specialization (via strong sparsity) disrupts normal development---just as depriving kittens of visual stimuli during the critical period permanently impairs vision.

- **Evidence for**: (1) Ge et al. (2026) used cross-snapshot crosscoders to track feature evolution across pre-training, finding two-phase learning: statistical learning followed by feature learning. (2) Li & Ren (2025) showed that temporal dynamics matter: ATM's exponential moving averages capture time-evolving feature importance and reduce absorption. (3) The "Temporal Sparse Autoencoders" paper (arXiv:2511.05541) found features exhibit "phase transitions between sequences," directly supporting developmental-stage framing. (4) Our pilot data shows dead latents (1232/2048 in MultiScale), suggesting training is far from converged---absorption measured at this stage may not reflect the final state. (5) The unified theory (Tang et al., 2025) frames absorption as a spurious optimum, but spurious optima are by definition escapable with sufficient training time or landscape perturbation.

- **Novelty estimate**: 8/10 --- The developmental analogy is genuinely novel in SAE research. The key empirical prediction (parent recovery via composition in late training) is falsifiable and would fundamentally change how the community thinks about absorption. If confirmed, it suggests that current absorption measurements are systematically biased toward mid-training snapshots.

---

## Phase 3: Self-Critique

### Against Candidate A (Absorption-Dark Matter Duality)

- **Prior work attack**: Engels et al. (2025) established the dark matter framework but did not connect it to absorption. The duality claim is novel. However, the "wave-particle duality" analogy may be criticized as superficial: quantum complementarity is a fundamental physical law, while absorption-dark matter trade-off may be an artifact of specific architectures rather than a universal property. The Heisenberg uncertainty principle has a precise mathematical form (Delta x * Delta p >= hbar/2); does absorption-dark matter have an analogous bound?

- **Methodological attack**: Measuring the trade-off requires quantifying both absorption and dark matter on the same SAE. Absorption requires ground-truth hierarchies (synthetic data); dark matter requires linear predictability of error (Engels et al. protocol). These are compatible but have not been combined before. The key challenge is ensuring that dark matter measurement is not confounded by the same geometric effects that plague absorption metrics. If dark matter is also model-dependent, the duality may not generalize.

- **Theoretical attack**: The claim that architectures "shift error from absorption to dark matter" assumes a conservation law: total representation error = absorption + dark matter + other pathologies. Is this sum actually conserved? If not, the duality is merely a correlation, not a complementarity. The wave-particle analogy may overclaim.

- **Scalability attack**: Measuring dark matter requires training linear probes on the error vector, which is computationally feasible but adds overhead. For a 6-variant x 5-replicate design, this doubles the measurement cost. However, the dark matter computation is CPU-bound and can be done offline.

- **Verdict**: STRONG --- The core idea is novel and empirically testable. The wave-particle analogy should be treated as conceptual framing, not a formal claim. The key deliverable is the first measurement of absorption-dark matter correlation across architectures, which is valuable regardless of whether a strict "conservation law" holds.

### Against Candidate B (Crosscoder Absorption Tomography)

- **Prior work attack**: SPARC (2025) and the MoE-dense crosscoder work (2026) established crosscoder methodology but did not apply it to absorption. The crosscoder latent diffing work (2025) is closest but focuses on feature correspondence, not absorption. The application to absorption is genuinely new. However, crosscoders between architectures with different decoder structures (e.g., Standard vs. Matryoshka nested dictionaries) may not align meaningfully---the shared latent space assumption may fail.

- **Methodological attack**: Crosscoder training is itself an optimization problem with its own failure modes. If the crosscoder fails to align features between architectures, the "tomography" will be blank or misleading. The method requires that crosscoder alignment succeeds before absorption can be compared---a circular dependency. Additionally, crosscoders require significant compute (training an additional encoder-decoder pair per architecture pair).

- **Theoretical attack**: The CT scan analogy breaks down because CT projections are mathematically invertible (Radon transform), while crosscoder "projections" are learned approximations with no guarantee of invertibility or completeness. The 3D reconstruction metaphor may create false confidence in the method's completeness.

- **Scalability attack**: For N architectures, crosscoder training requires O(N^2) pairs. With 6 variants, this is 15 crosscoders---each requiring GPU training. This exceeds the 1-hour-per-experiment constraint unless crosscoders are very small or trained on subsets.

- **Verdict**: MODERATE --- The idea is novel and the SPARC results suggest cross-model alignment is feasible. The main weaknesses are computational cost and the risk that crosscoder alignment fails for structurally different architectures. Could be strengthened by focusing on a smaller set of architectures (3-4) with compatible decoder structures, or by using lightweight crosscoder approximations.

### Against Candidate C (Curriculum Absorption Hypothesis)

- **Prior work attack**: Ge et al. (2026) already track feature evolution across training checkpoints, but they focus on feature emergence/rotation/degeneration, not absorption specifically. The "Temporal Sparse Autoencoders" paper (arXiv:2511.05541) found phase transitions in feature learning. The application to absorption as a developmental stage is new, but the underlying method (checkpointing during training) is established.

- **Methodological attack**: Testing the hypothesis requires training SAEs with frequent checkpointing and measuring absorption at each checkpoint. This is training-intensive and violates the project spec's training-free constraint. However, one could use existing pretrained SAEs with publicly available training logs/checkpoints if available. Alternatively, one could train small SAEs on SynthSAEBench specifically for this analysis (training is fast on synthetic data).

- **Theoretical attack**: The developmental psychology analogy may not map cleanly. Children's overgeneralization is driven by limited world experience, while SAE "overgeneralization" is driven by sparsity constraints. The mechanisms are different. The "critical period" analogy is particularly strained: there's no evidence that early training sparsity permanently impairs SAE feature learning.

- **Scalability attack**: Training SAEs from scratch with checkpoint storage is compute-intensive. For a GPT-2 small SAE trained on 100M tokens, storing checkpoints every 1M tokens requires 100 checkpoints. The analysis itself is lightweight, but the training cost is high. However, SynthSAEBench training is much faster (2M samples, ~2 minutes per variant).

- **Verdict**: MODERATE --- The idea is novel and the developmental framing is evocative. The main weakness is the training requirement. Could be strengthened by focusing on SynthSAEBench (fast training) and using lightweight checkpoints (every 10% of training). The key prediction (parent recovery via composition in late training) is falsifiable and would be a genuine contribution if confirmed.

---

## Phase 4: Refinement

### Dropped Ideas

- **None dropped** --- All three candidates survive the adversarial testing with at least MODERATE ratings. However, Candidates B and C have significant methodological weaknesses that limit their standalone viability.

### Strengthened Ideas

- **Candidate A (Absorption-Dark Matter Duality)**: Address the "conservation law" concern by framing the relationship as a correlation/trade-off rather than a strict complementarity. The deliverable is not a physical law but an empirical finding: "Architectures that reduce absorption increase dark matter, and the sum of the two is more stable than either alone." This is testable without claiming a conservation law. Use the wave-particle analogy only in the Discussion as conceptual framing.

- **Candidate B (Crosscoder Tomography)**: Narrow scope to 3 architectures (Standard, TopK, Matryoshka) with compatible decoder structures to reduce crosscoder training cost. Use lightweight crosscoders (smaller latent dimension) trained on SynthSAEBench activations. Frame the contribution as "first cross-architecture absorption comparison via crosscoder alignment" rather than "full tomographic reconstruction."

- **Candidate C (Curriculum Absorption)**: Focus exclusively on SynthSAEBench (fast training, ground-truth hierarchies). Use 10 checkpoints (every 10% of training) rather than fine-grained checkpointing. Measure not just probe-based absorption but also causal parent-feature recovery via decoder composition (can child latents reconstruct parent features?). This directly tests the "recovery via composition" prediction.

### Additional Evidence Found

- **"Priors in Time: Missing Inductive Biases for Language Model Interpretability"** (arXiv:2511.01836, Nov 2025): Temporal structure significantly affects feature learning in SAEs. Supports Candidate C's developmental-stage framing.

- **"Training Dynamics of Sparse Autoencoders: A Phase Transition Perspective"** (arXiv:2509.18473, Sep 2025): SAE training exhibits phase transitions in loss landscape. Features emerge in bursts rather than gradually. Supports the "critical window" aspect of Candidate C.

- **"Feature Suppression in Sparse Autoencoders"** (arXiv:2601.09437, Jan 2026): Identifies feature suppression (distinct from absorption) as a training-dynamic phenomenon where features are learned then suppressed by competing features. Directly supports Candidate C's transient-absorption hypothesis.

- **"The Geometry of Sparse Autoencoder Features"** (arXiv:2508.20192, Aug 2025): Decoder geometry determines feature recoverability. Features with high decoder norm are more robust to suppression. Suggests that absorption may be reversible if decoder geometry permits.

### Selected Front-Runner

**Candidate A: The Absorption-Dark Matter Duality** is selected as the front-runner for the following reasons:

1. **Directly addresses the project's core finding**: Our pilot results show TopK reduces absorption by 84% but MCC stays at chance (~0.21). The dark matter hypothesis explains this: TopK is not "recovering" parent features---it is shifting them into the unexplained variance. This reframes our strongest negative result as a positive finding.

2. **Best fit with project constraints**: The project spec emphasizes training-free analysis. Measuring dark matter on existing pretrained SAEs (or the already-trained pilot SAEs) requires no additional training. The experiment is purely analytical.

3. **Connects two major research threads**: Absorption and dark matter are the two most consequential SAE failure modes, yet no paper has connected them. This synthesis would be a major contribution.

4. **Explains the pilot anomalies**: The near-chance MCC across all variants (including Random) suggests that no architecture is genuinely recovering features---they're just distributing the error differently. The dark matter framework quantifies this.

5. **Complements the current paper**: The existing paper focuses on component-isolated absorption reduction. Adding a dark matter analysis would transform it from "which component reduces absorption?" to "what happens to the error that absorption reduction displaces?" This is a deeper and more surprising question.

6. **Methodologically tractable**: Engels et al. provide exact protocols for measuring linear dark matter. The only novel step is computing absorption and dark matter on the same SAEs and testing their correlation.

---

## Phase 5: Final Proposal

### Title

**The Absorption-Dark Matter Duality: Where Does the Error Go When SAEs Reduce Absorption?**

### Hypothesis

When sparse autoencoders reduce feature absorption through architectural modifications (TopK sparsity, multi-scale decomposition, orthogonality penalties), the "missing" parent-feature information is not recovered but displaced into dark matter---the structured, linearly predictable error that SAEs systematically fail to represent. Formally, for an SAE with absorption rate A and linear dark matter fraction D (fraction of error variance linearly predictable from activations), the total representation deficit R = A + D is more stable across architectures than either A or D alone. Specifically:

**H1 (Trade-off):** Across SAE architectures, absorption rate A and dark matter fraction D are negatively correlated (r < -0.5), with architectures that minimize A exhibiting elevated D.

**H2 (Conservation):** The sum R = A + D (total representation deficit) has lower variance across architectures than either A or D individually, suggesting a representational budget that architectures allocate differently but do not expand.

**H3 (Mechanism):** TopK sparsity reduces A by suppressing parent-feature activation in the latent space, but the suppressed information persists in the residual error where it is linearly predictable (elevated D). Multi-scale decomposition reduces A by distributing parent features across dictionary levels, but this increases D at inner levels due to hedging.

### Motivation

The SAE community has treated absorption as an independent pathology to be eliminated. Matryoshka SAEs, OrtSAE, HSAE, GBA, and masked regularization all report absorption reduction as a primary contribution. Our own component-isolated study found that TopK sparsity reduces absorption by 84% (Cohen's d = 5.51)---a dramatic effect.

Yet a critical question has been ignored: **Where does the error go?** If parent features are no longer absorbed, are they recovered? Our pilot data says no: feature recovery MCC remains at chance (~0.21) across all variants including Random. This suggests that absorption reduction does not recover parent features---it merely displaces the error into a different form.

Engels et al. (2025) established that SAEs leave substantial "dark matter": error that is linearly predictable from the original activations and therefore represents learnable features that the SAE failed to capture. Dark matter is the natural destination for displaced parent-feature information: if the SAE's latent budget is insufficient to independently encode parent and child features, and architectural modifications prevent absorption (merging parent into child), the only remaining option is to leave the parent feature unrepresented (dark matter).

This reframes the entire absorption debate. The community has been asking "how do we reduce absorption?" when it should be asking "how do we optimally allocate representational error between absorption and dark matter?" For interpretability, dark matter may be preferable to absorption: an absorbed parent feature is hidden but recoverable (via child latents), while dark matter is completely outside the SAE's representation.

### Method

**Step 1: Measure Absorption and Dark Matter on the Same SAEs**

Use the existing pilot SAEs (Baseline, TopK, MultiScale, Orthogonality, Random) and any additional pretrained SAEs available via SAELens. For each SAE:

- **Absorption (A)**: Compute ground-truth absorption rate on SynthSAEBench-16k using known parent-child relationships. This is already implemented from the pilot experiments.
- **Dark Matter (D)**: Following Engels et al. (2025), train a linear probe to predict the SAE reconstruction error e = x - x_hat from the original activation x. Compute R^2_linear = Var(linear_pred(e|x)) / Var(e). The linear dark matter fraction is D = R^2_linear. Also compute the scalar dark matter: train a probe to predict ||e|| from x, reporting R^2_scalar.

**Step 2: Test the Trade-off Hypothesis (H1)**

Compute Pearson correlation between A and D across variants. Test whether the correlation is negative and significant. Controls: partial correlation controlling for L0 sparsity and dictionary size.

**Step 3: Test the Conservation Hypothesis (H2)**

Compute R = A + D for each variant. Test whether Var(R) < Var(A) and Var(R) < Var(D) using Levene's test. If the sum is more stable than its components, this supports the representational budget hypothesis.

**Step 4: Test the Mechanism Hypothesis (H3)**

For TopK variants: measure parent-feature activation magnitude in the latent space. If TopK suppresses parent activation (low parent firing rate) but the parent information is linearly predictable in the error, this supports the suppression mechanism.

For MultiScale variants: measure D at each dictionary level (inner, middle, outer). If inner levels show elevated D due to hedging, this supports the distribution mechanism.

**Step 5: Real-LLM Validation (Exploratory)**

If time permits, measure A and D on 3-4 pretrained SAEs from SAELens (Standard, TopK, JumpReLU, BatchTopK) on GPT-2 small or Pythia-160M. For real LLMs, absorption is measured via SAEBench protocol; dark matter is measured via Engels et al. protocol. Test whether the A-D trade-off holds on real models.

### Cross-Domain Insight

The key transplanted principle is from **wave-particle duality** and **complementarity in quantum mechanics**. In quantum mechanics, precise measurement of a particle's position reduces certainty about its momentum, and vice versa. The two properties are not independent but complementary aspects of a single quantum system.

In SAEs, absorption and dark matter are complementary aspects of a single representational limit: the SAE's latent budget. When the budget is insufficient to independently encode all features in a hierarchy, the SAE faces a choice: merge parent into child (absorption) or leave parent unrepresented (dark matter). Architectural modifications that prevent merging do not expand the budget---they merely change how the budget allocates error.

The structural correspondence holds because:
1. Both involve a finite resource (latent budget / Planck constant) that constrains simultaneous representation of dual properties.
2. Both exhibit a trade-off: improving one measurement worsens the other.
3. Both have a "total uncertainty" that may be bounded below (Var(R) >= constant).

The critical difference is that quantum complementarity is a fundamental physical law, while absorption-dark matter trade-off is an empirical property of current architectures. It may not hold for all possible SAE designs (just as quantum mechanics does not apply to classical systems). This is why we frame it as a testable hypothesis, not a theorem.

### Experimental Plan

**Experiment 1: Absorption-Dark Matter Trade-off on SynthSAEBench (Pilot: 15 min, Full: 30 min)**
- Use existing pilot SAEs (Baseline, TopK, MultiScale, Orthogonality, Random)
- Compute absorption A (already done)
- Compute dark matter D via linear probe on error vector (CPU-bound, ~5 min per SAE)
- Test H1: Pearson correlation between A and D
- Test H2: Variance comparison of R = A + D vs. A and D individually
- **Falsification criterion**: If r >= 0 (positive or zero correlation), the trade-off hypothesis is rejected.

**Experiment 2: Mechanism Analysis (Pilot: 15 min, Full: 30 min)**
- For TopK: measure parent-feature activation rates in latent space
- For MultiScale: measure D per dictionary level
- Test H3: specific mechanism predictions
- **Falsification criterion**: If parent information is NOT linearly predictable in TopK error, the suppression mechanism is rejected.

**Experiment 3: Real-LLM Validation (Optional, 45 min)**
- Load 3-4 pretrained SAEs from SAELens on GPT-2 small
- Compute absorption via SAEBench protocol
- Compute dark matter via Engels et al. protocol
- Test whether A-D trade-off transfers to real LLMs
- **Falsification criterion**: If r >= 0 on real LLMs, the trade-off may be synthetic-data-specific.

### Resource Estimate

| Experiment | Model | GPU Time | Notes |
|---|---|---|---|
| E1 Trade-off | Existing pilot SAEs | ~0 min GPU | CPU-bound linear probes on error vectors |
| E2 Mechanism | Existing pilot SAEs | ~0 min GPU | CPU-bound analysis of latent activations |
| E3 Real-LLM | GPT-2 Small SAEs | ~30 min | SAEBench eval + dark matter probes |
| **Total** | | **~30 min GPU** | Fits within 1-hour constraint |

All experiments are training-free (analysis of existing SAEs), fitting the project spec constraint perfectly. E1 and E2 use the already-trained pilot SAEs, requiring zero additional GPU time.

### Risk Assessment

**Risk 1: Dark matter measurement is confounded by reconstruction quality**
- Mitigation: Control for explained variance in all correlations. Report partial correlations. The dark matter metric (R^2 of error prediction) is normalized by error variance, so it is already invariant to overall reconstruction quality.

**Risk 2: The A + D sum is not actually conserved**
- Mitigation: Frame H2 as exploratory. If R is not more stable than A or D, report this as a negative result. The core contribution (H1: the trade-off) does not depend on H2.

**Risk 3: The trade-off is specific to SynthSAEBench and does not transfer to real LLMs**
- Mitigation: E3 tests transfer. If it fails, the paper's scope is narrowed to synthetic data but the contribution remains valid. The Discussion should discuss the synthetic-to-real gap explicitly.

**Risk 4: The wave-particle analogy is criticized as superficial**
- Mitigation: Use the analogy only in Discussion, not in the formal hypotheses. The empirical claims (H1-H3) are stated without quantum terminology. The analogy serves as conceptual scaffolding, not evidence.

**Risk 5: Dark matter is already well-studied and the absorption connection is trivial**
- Mitigation: No prior paper has connected absorption to dark matter. A literature search confirms these are separate research threads. The synthesis is genuinely novel.

### Novelty Claim

This proposal makes four specific novel contributions:

1. **First connection between absorption and dark matter**: While absorption and dark matter are each major SAE failure modes, no prior work has tested whether they are related. This paper provides the first empirical test.

2. **First evidence that absorption reduction displaces error rather than recovering features**: Our pilot data shows MCC ~0.21 across all variants (chance level), suggesting no architecture genuinely recovers features. The dark matter framework explains where the "missing" information goes.

3. **Reframing of absorption from pathology to trade-off**: Rather than treating absorption as unambiguously bad, this paper asks "what is the optimal allocation of representational error?" This shifts the research question from elimination to optimization.

4. **Methodological contribution**: The combined absorption-dark matter measurement protocol can be applied to any SAE architecture, providing a more complete picture of representational failure than absorption alone.

No prior paper has proposed or evaluated an absorption-dark matter trade-off. The closest work (Engels et al.'s dark matter framework) measures unexplained variance but does not connect it to absorption. Our component-isolated design provides the ideal testbed for this connection because we have multiple architectures with varying absorption rates on the same ground-truth data.
