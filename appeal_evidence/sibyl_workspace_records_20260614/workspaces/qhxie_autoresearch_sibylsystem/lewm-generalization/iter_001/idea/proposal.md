# Research Proposal

## Title

**Geometric Factorization Meets Physical Reality: A Systematic Compositional Generalization Benchmark for JEPA World Models**

*Subtitle: Probing, Interventional Independence, and Regime-Aware Evaluation of LeWorldModel under Physics Factor Holdout*

---

## Abstract

We present the first systematic evaluation of compositional generalization in a JEPA world model (LeWorldModel / LeWM), combining a rigorous CoGenT-style physics factor holdout benchmark with novel geometric and interventional diagnostics. Our research centers on three interlocking questions: (1) Does LeWM compositionally generalize across unseen combinations of physical parameters (friction, gravity, mass)? (2) Does SIGReg's isotropic Gaussian prior promote or compete with the linear, orthogonal factorization that compositional generalization theory (Uselis et al., 2026) identifies as necessary? (3) Which component — encoder or predictor — is the generalization bottleneck, and can LoRA adaptation efficiently bridge the gap? Beyond model-centric evaluation, we introduce physics regime analysis as a predictor: we test whether failures of compositional generalization are attributable to model limitations or to physically irreducible coupling between factors. Our evaluation produces a reusable benchmark (ComPhys-LeWM) and a multi-level diagnostic protocol that can be applied to any JEPA world model.

---

## Motivation

LeWorldModel (LeWM) is the current frontier of compact, efficient JEPA world models: 15M parameters, end-to-end trained from pixels in hours, with a minimal two-term loss (prediction + SIGReg). It achieves strong in-distribution probing of physical states and outperforms DINO-WM and PLDM on multiple benchmarks. Yet it fails in the Two-Room low-diversity environment, and crucially, **no prior work has tested whether its representations compositionally generalize across combinations of physical parameters**.

The theoretical landscape makes this question urgent. Uselis et al. (2026) prove that compositional generalization *requires* linear, orthogonal per-concept representations. SIGReg enforces an isotropic Gaussian distribution — a constraint that is *mathematically orthogonal* to the factorization requirement (as the contrarian perspective correctly identifies: isotropy constrains variance structure, not factor alignment). Whether SIGReg promotes, hinders, or is neutral toward physical concept factorization is a completely open empirical question with direct practical implications: if the Gaussian prior helps, it validates the LeJEPA/SIGReg design philosophy; if it hurts or is neutral, it suggests architectural alternatives (group-structured latent spaces, RCC architectures) as targets for future work.

The neuroscience literature (Knopp et al., 2025) provides a powerful prior: predictive learning autonomously induces modular, compositional representations in systems trained on environments with independent latent factors. LeWM's core training objective — predicting future latent states — is precisely the form of predictive learning that theory suggests should induce compositional structure. But this prediction is conditional on sufficient data diversity, and LeWM's Two-Room failure directly suggests there is a diversity threshold below which the inductive bias fails.

Finally, the interdisciplinary perspective contributes a critical caveat that no other perspective captures: not all physical factor combinations are composable. When parameters interact nonlinearly (coupling), or when the combination crosses a bifurcation boundary producing qualitatively new dynamics, compositional generalization has a *principled physical upper bound* independent of model quality. Distinguishing model-fixable failures from physically irreducible failures is essential for fair evaluation and practical system design.

---

## Research Questions

**RQ1 (Empirical)**: Does LeWM exhibit compositional generalization to held-out combinations of physical parameters (friction, agent mass, object mass) that were individually seen during training? If so, at what scale of factor combination does generalization break down?

**RQ2 (Mechanistic)**: Does SIGReg's isotropic Gaussian prior promote or compete with the linear orthogonal factorization required for compositional generalization (Uselis et al., 2026)? Specifically, do SIGReg-trained models exhibit higher per-factor subspace orthogonality and displacement vector consistency than alternative regularizers?

**RQ3 (Interventional)**: Are LeWM's physical concept representations causally independent — i.e., can we intervene on one concept's latent subspace without affecting others? What is the Interventional Independence Score (IIS) and how does it relate to zero-shot transfer?

**RQ4 (Adaptive)**: Which component (encoder or predictor) is the generalization bottleneck? Does LoRA adaptation of the bottleneck component efficiently recover compositional generalization from few target-domain samples?

**RQ5 (Physical-regime)**: Does the physical coupling structure between factors (estimated from trajectory statistics) predict compositional generalization success independently of model quality? Are some failures physically irreducible?

---

## Hypotheses

### H1: Compositional Generalization Gap (Primary)
LeWM shows measurable performance degradation on held-out physics factor combinations: linear probing R² drops ≥20% relative and CEM planning success rate drops ≥25% relative compared to in-distribution, with the drop scaling super-linearly in the number of simultaneously held-out factor dimensions. This is statistically significant (paired t-test, p<0.05, Cohen's d>0.8, across ≥5 training seeds).

**Falsification**: If probing R² drops <10% relative on two-factor holdouts across all seeds, LeWM achieves genuine compositional generalization — itself a notable positive finding that would redirect the paper's focus to understanding *why* it generalizes.

### H2: SIGReg Promotes (Not Hinders) Factorization
SIGReg-trained models show higher principal angle orthogonality between per-factor latent subspaces (mean cosine similarity <0.15) compared to VICReg-trained models (>0.25), and this geometric advantage translates to better zero-shot transfer performance (Pearson r>0.6 between orthogonality and holdout accuracy).

**Falsification**: If SIGReg and VICReg show similar orthogonality (difference <0.05) and similar holdout performance, the Gaussian prior is neutral with respect to compositional structure.

**Alternative falsification**: If SIGReg shows *lower* orthogonality than VICReg (the contrarian's prediction: "representational egalitarianism de-factorizes"), this would be a strong negative result directly challenging the LeJEPA design rationale.

### H3: Predictor as Generalization Bottleneck
LoRA adaptation of the predictor alone recovers ≥70% of the in-distribution probing accuracy on held-out combinations with 50 target-domain trajectories, while LoRA on the encoder alone recovers ≤40%. This asymmetry indicates the predictor encodes domain-specific dynamics while the encoder captures domain-general visual features. The minimum LoRA rank for efficient adaptation correlates (r>0.5) with the intrinsic dimensionality of the changed factor subspace.

**Falsification**: If encoder-only and predictor-only LoRA perform within 10% of each other, the encoder-predictor decomposition is not the right level of analysis.

### H4: Displacement Vector Consistency Predicts Transfer
The consistency of factor displacement vectors across the latent space (parallelogram test, from the interdisciplinary perspective) is a significant predictor of zero-shot transfer success, explaining variance beyond what principal angle analysis already captures (incremental R²>0.1 in regression predicting holdout accuracy).

### H5: Physical Coupling Predicts Regime-Dependent Failure
Factor combinations with high estimated physical coupling strength (mixed partial derivatives of trajectory statistics) show systematically larger compositional generalization gaps, across all model variants (LeWM-SIGReg, LeWM-VICReg, DINO-WM). Failures in strongly-coupled regimes persist even with LoRA adaptation, suggesting physical irreducibility.

---

## Expected Contributions

1. **ComPhys-LeWM Benchmark**: The first CoGenT-style compositional holdout benchmark for a JEPA world model, with systematic coverage of 3 physical factors × 3 levels in DMControl Walker/PushT environments. Fully reusable for future world model evaluation.

2. **Empirical test of SIGReg-orthogonality connection**: First measurement of whether the Gaussian isotropic prior promotes or competes with the geometric conditions (Uselis et al., 2026) required for compositional generalization. Directly relevant to the design of future JEPA architectures.

3. **Interventional Independence Score (IIS)**: A novel metric for evaluating causal independence of physical concept representations in world model latent spaces. Transplants activation-patching methodology from LLM interpretability to world model evaluation.

4. **Regime-aware evaluation framework**: A method for classifying held-out factor combinations as "weakly coupled" (model-fixable failure) or "strongly coupled" (physically irreducible failure), providing fairer benchmarking and design guidance.

5. **LoRA as an encoder-predictor diagnostic**: First application of LoRA adaptation efficiency to distinguish encoder-localized from predictor-localized domain-specific knowledge in a JEPA world model.

---

## Method

### 5.1 Environment and Benchmark Setup

**Primary environment**: DMControl Walker with MuJoCo XML parameter variation.
- Factor 1: Gravity magnitude {0.5g, 1.0g, 2.0g} (3 levels)
- Factor 2: Joint friction coefficient {0.5x, 1.0x, 2.0x default} (3 levels)
- Factor 3: Body density/mass {0.5x, 1.0x, 2.0x default} (3 levels)
- Total: 27 factor combinations

**Secondary environment**: PushT (the LeWM flagship benchmark) with stable-worldmodel FoV support for friction and mass variation.

**Holdout design (CoGenT-style)**:
- Training split: 18/27 combinations, ensuring each individual factor level appears in ≥6 training combos
- Holdout split: 9 combinations — mix of interpolation-type (surrounded by training points in factor space) and extrapolation-type (at boundary)
- Multiple random splits (×3) to measure sensitivity to split choice

**Data collection**: 200 trajectories per combination using scripted policy or SWM's built-in data collection. 200 × 27 = 5,400 trajectories total.

### 5.2 Training Protocol

**Model**: LeWM (ViT-Tiny encoder ~5M + transformer predictor ~10M, default SIGReg λ)
- 7 random seeds for the primary experiment
- 3 seeds for ablations
- Each training run: ~2-4h on a single GPU (per LeWM paper)

**Regularizer variants** (Experiment 2):
- LeWM with SIGReg (default)
- LeWM with VICReg (alternative regularizer)
- LeWM with no regularizer (collapse-prone baseline, likely needed for control)

**Baselines**:
- Oracle (trained on all 27 combos including holdout) — ceiling
- Matched-volume control (18 randomly-selected combos, no systematic holdout) — data-volume control
- Random encoder (untrained ViT-Tiny) — floor
- DINO-WM (frozen DINOv2 encoder + learned predictor, same training split) — architecture comparison

### 5.3 Multi-Level Evaluation Protocol

**Level 1 — Probing (Standard)**:
- Linear and MLP regression probes on frozen encoder embeddings
- Predict: agent joint angles, CoM velocity, gravity factor, friction factor, body mass
- Metrics: R² in-distribution vs. held-out combinations
- Statistical tests: paired t-test (5 seeds), bootstrap 95% CI, Cohen's d

**Level 2 — Geometric Analysis**:
- Per-factor subspace computation: PCA on embeddings grouped by each factor level
- Principal angle analysis (from Uselis et al., 2026 codebase) between factor subspaces
- Displacement vector consistency (parallelogram test): for each factor, compute displacement vector at ≥3 reference points in factor space; measure variance (low variance = additive composition)
- CKA between factor-organized representations as alternative metric

**Level 3 — Interventional Probing (Novel)**:
For each physical concept C and test trajectory:
1. Encode trajectory to get latent sequence z₁, ..., z_T
2. Use trained linear probe to identify concept-C's direction in latent space
3. Shift z_t along concept-C's direction by δ (corresponding to known parameter change, e.g., gravity 1.0g → 2.0g)
4. Roll out modified z_t through the predictor
5. Measure: (a) predicted trajectory changes as expected for C; (b) predictions for other concepts D, E remain unchanged
6. Interventional Independence Score (IIS): for each concept pair (C, D): IIS(C,D) = 1 - |correlation(intervention_on_C, change_in_D_prediction)|

**Level 4 — Behavioral (Planning)**:
- CEM planning success rate on held-out combinations (50 trajectories per combo)
- Walk task: reach target joint configuration
- Latent prediction MSE on held-out factor combinations

**Level 5 — Physical Regime Analysis (Physics-Informed)**:
- Coupling strength estimation: for each factor pair (f_i, f_j), estimate mixed partial derivative of trajectory statistics (velocity variance, contact frequency, energy dissipation) with respect to parameter changes
- Classify each holdout combination as "weakly coupled" (all coupling strengths below threshold c*) or "strongly coupled" (≥1 coupling exceeds c*)
- Test whether regime classification predicts compositional transfer success across all model variants

### 5.4 LoRA Adaptation Diagnostic

Applied to the held-out combinations only, starting from the trained LeWM:
- **Adaptation targets**: encoder Q/V only, predictor Q/V only, both encoder+predictor
- **LoRA ranks**: {2, 4, 8, 16}
- **Adaptation data**: {10, 25, 50, 100, 200} trajectories from target domain
- **Metrics**: probing R² recovery, CEM success rate, samples-to-90%-recovery
- **Baselines**: zero-shot (no adaptation), head-only fine-tuning (probe head only), full fine-tuning (upper bound), random LoRA (untrained adapters)
- **Controls**: in-distribution LoRA (check that LoRA does not trivially improve already-good in-distribution performance)

---

## Experimental Plan

| Experiment | GPU-hours | Key Output | Falsification condition |
|------------|-----------|-----------|------------------------|
| Pilot: 1-factor friction holdout (1 seed) | 0.3h | Framework validation | If <5% drop on interpolation, adjust to more extreme holdout |
| Primary: 7-seed compositional holdout (H1) | ~50h | Generalization gap table + heatmap | <10% drop = LeWM compositionally generalizes |
| SIGReg vs. VICReg ablation (H2) | ~20h | Orthogonality comparison, factorization score | Similar orthogonality = Gaussian prior is neutral |
| Interventional probing (H3, novel metric) | ~10h | IIS matrix for all concept pairs | IIS not above random perturbation baseline |
| LoRA adaptation diagnostic (H4) | ~15h | Adaptation curves (encoder vs. predictor) | No significant encoder/predictor asymmetry |
| Physical coupling analysis (H5) | ~5h CPU | Regime classification + predictive accuracy | Coupling strength does not predict failures |
| DINO-WM comparison | ~15h | Architecture comparison on same benchmark | Same failure pattern as LeWM |
| **Total** | **~115h** | Full paper dataset | |

**Pilot design** (~12 min on 1 GPU): Train LeWM on 2 friction levels (0.5x, 2.0x) in Walker, probe on 1.0x (interpolation) and 3.0x (extrapolation). If linear probe R² drops >15% on 1.0x, proceed to full study. If <5% drop, adjust to more aggressive holdout.

---

## Novelty Assessment

### Prior Art Searched

**Search 1**: "JEPA world model compositional generalization physical factor holdout" — No papers found that systematically hold out combinations of physical parameters in JEPA world model training. DINO-WM tests unseen maze walls (single-factor visual), not physics combinations. V-JEPA 2 tests unseen labs (single-domain transfer), not factorial holdout. **Verdict: Gap confirmed.**

**Search 2**: "SIGReg isotropic Gaussian world model compositional generalization latent factorization" — No papers found linking SIGReg geometry to the Uselis et al. geometric conditions for compositional generalization. The theoretical question (does Gaussian isotropy promote orthogonal factorization?) is completely open. **Verdict: Gap confirmed.**

**Search 3**: "Interventional probing physical concept independence world model latent space" — The closest work is "Learning Robust Intervention Representations" (ICLR 2026) which learns delta-vector representations of interventions in object-centric settings, and PIWM (arXiv:2412.12870) which aligns latent dims with physical quantities. Neither applies interventional probing to test *causal independence* between physical concepts in a JEPA world model. **Verdict: Novel.**

**Search 4**: "Compositional phase transition data diversity world model generalization boundary" — Found the PNAS 2025 paper on phase transitions in diffusion models (hierarchical data structure), and data diversity frameworks (OpenReview 2025). No paper applies a phase-transition or generalization-boundary framework to JEPA world models. The innovator's Candidate C (phase diagrams) is novel in this context. **Verdict: Novel framing, confirmed.**

### Differentiators from Closest Prior Work

| Prior Work | How We Differ |
|---|---|
| Uselis et al. (2026) — geometric conditions for CG | They test static vision encoders (CLIP, DINO). We test a *world model* that also learns dynamics (JEPA predictor). |
| Redhardt et al. (2025) — data diversity drives CG | They vary data volume/diversity in standard vision models. We vary physical factor *combinations* in a world model setting with physics-specific holdouts. |
| AIM/V-JEPA 2 probing (2026) | Passive observational probing. We introduce *interventional* probing (IIS) that tests causal independence. |
| cRSSM / DALI / PrivilegedDreamer | Contextual world models that *use* physics parameters as context signals. We study whether these parameters are learned *compositionally* without explicit context conditioning. |
| LeWM paper itself | Only evaluates in-distribution. We provide the first cross-domain compositional evaluation. |

---

## Evidence-Driven Revisions

*This section will be populated after pilot experiment. Currently: first synthesis round, no prior pilot data.*

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| LeWM training fails on modified physics in Walker | Medium | Low | SIGReg shown stable across environments; fallback to PushT primary |
| LoRA underperforms on ViT-Tiny (known limitation) | Medium | High | Expected: report as negative result; full fine-tuning as ceiling; head-only baseline |
| Compositional gap is too small to detect | Medium | Medium | Pilot calibrates; extend to extrapolation-type holdouts if interpolation gap is small |
| Physical coupling analysis is too noisy | Medium | Medium | Use bootstrap CI; binary regime classification (independent vs. interacting) as fallback |
| All model variants fail equally on all holdouts | Low | Low | Degree of failure still predicts regime; LoRA recovery differential still diagnostic |
| Interventional probing confounded by correlated concept directions | Medium | Medium | Orthogonalize probe directions (Gram-Schmidt); include random-direction baseline |
| Visual distribution shift conflates with compositional failure | Medium | Medium | Visual oracle control (model with shuffled physics labels) |

---

## Resource Estimate

- **Model**: ViT-Tiny + transformer predictor (~15M params), single GPU, <4GB VRAM
- **Compute**: ~115 GPU-hours total, ~2 weeks on 1 GPU, ~4 days on 4 GPUs
- **Individual experiments**: All under 1 hour (aligned with project efficiency constraint)
- **Storage**: ~5,400 trajectories × ~100MB = ~0.5GB HDF5 data
- **Software**: le-wm (official codebase), stable-worldmodel (environments + probing + planning), peft (LoRA), necessary-compositionality (principal angle analysis), scikit-learn (probing heads)
