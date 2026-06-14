# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **LeWorldModel (le-wm) official repo** ([github.com/lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)) -- MIT-compatible, ViT-Tiny (~15M params), trains on single GPU in hours. **Code exists and is usable.** Supports PushT, TwoRoom, Reacher, OGBench-Cube. Built on `stable-worldmodel` + `stable-pretraining`.

2. **stable-worldmodel (SWM)** ([github.com/galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)) -- The infrastructure layer LeWM depends on. **Critically, SWM explicitly exposes "Factors of Variation" (FoV) including physical parameters (friction, damping, mass, gravity) as controllable options** during data collection and environment reset. The `World` abstraction supports vectorized environments and domain randomization. This is the key enabler for the entire project: we do not need to hack environment code to vary physics.

3. **DINO-WM** ([dino-wm.github.io](https://dino-wm.github.io/)) -- Frozen DINOv2 + learned predictor; strong zero-shot planning baseline; physical state probing protocol reusable. **Code exists.**

4. **Compositional Generalization Requires Linear, Orthogonal Representations** (arXiv:2602.24264, Uselis et al., 2026) -- Formalizes geometric conditions for compositional generalization. Code at [github.com/oshapio/necessary-compositionality](https://github.com/oshapio/necessary-compositionality). ~50 lines of numpy for principal angle analysis. **Directly reusable.**

5. **Does Data Scaling Lead to Visual Compositional Generalization?** (arXiv:2507.07102, 2025) -- Shows data diversity (not scale) drives compositional generalization. Code at [github.com/oshapio/visual-compositional-generalization](https://github.com/oshapio/visual-compositional-generalization). Provides the holdout split methodology. **Reusable for factor combination holdout design.**

6. **DALI: Dynamics-Aligned Latent Imagination** (arXiv:2508.20294, NeurIPS 2025) -- Contextual world model that infers latent context (gravity, friction) from interaction history. Achieves zero-shot generalization to unseen physics parameters **without explicit context variables**. Counterfactual latent probing shows individual dimensions encode physical properties. **Key comparator and conceptual benchmark**, but operates within DreamerV3 (different architecture family from JEPA).

7. **LoRA-ViT** ([github.com/JamesQFreeman/LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT)) -- Drop-in LoRA for Vision Transformers via timm. **Critical caveat: literature shows LoRA degrades significantly on smaller ViTs (DeiT-T, DeiT-S) due to limited representational capacity.** LeWM uses ViT-Tiny (~5M encoder), so LoRA adaptation may be weak. This is a real engineering risk.

8. **HuggingFace PEFT** ([github.com/huggingface/peft](https://github.com/huggingface/peft)) -- Standard LoRA/QLoRA implementation. Apache 2.0. Can apply to any PyTorch model. **Usable as fallback if LoRA-ViT doesn't integrate cleanly.**

9. **Factored World Models for Zero-Shot Generalization** (Biza et al., 2022) -- Object-centric world model with GNN for compositional generalization in manipulation. Foundational reference but uses fundamentally different architecture (GNN, not JEPA). Not directly reusable code-wise.

10. **DreMa: Compositional World Models via Gaussian Splatting** (ICLR 2025, [github.com/leobarcellona/drema_code](https://github.com/leobarcellona/drema_code)) -- Compositional world model using Gaussian Splatting + physics simulator. Different approach (explicit 3D reconstruction, not latent prediction). Useful as a comparison point for "what compositional world models look like" but not directly applicable.

11. **DMControl Generalization Benchmark** ([github.com/nicklashansen/dmcontrol-generalization-benchmark](https://github.com/nicklashansen/dmcontrol-generalization-benchmark)) and **DMC-GB2** ([github.com/aalmuzairee/dmcgb2](https://github.com/aalmuzairee/dmcgb2)) -- Benchmarks for visual generalization in DMControl. Focused on visual shifts (color, background), not physics parameter variation. Limited relevance.

12. **Probing the Latent World: AIM Framework** (arXiv:2603.20327, 2026) -- Passive quantization probe for V-JEPA 2 latent space. Shows physical dimensions are statistically separable. Reusable probing methodology. **No cross-domain transfer evaluation.**

### Landscape Summary

**What actually works and has usable code:**
- LeWM trains from pixels on a single GPU in hours (~15M params). The official codebase is clean and well-maintained.
- The `stable-worldmodel` library (SWM) underneath LeWM **already supports parameterized physical factors of variation** (friction, damping, mass, gravity) in PushT and other environments. This is the single most important engineering fact: we do not need to build custom environments from scratch.
- Linear/MLP probing of frozen latent embeddings is a standard, well-understood evaluation protocol. LeWM already includes probing in its evaluation suite.
- Principal angle analysis for measuring linear factorization of representations is ~50 lines of numpy (Uselis et al. codebase).

**What does NOT exist and we would need to build:**
- No one has done a **systematic holdout of physical factor combinations** for LeWM training. We need to design the train/test split over factor combinations.
- No one has applied **LoRA specifically to LeWM's ViT-Tiny encoder or transformer predictor**. This is novel but carries real risk given LoRA's known weakness on small ViTs.
- No one has measured the **linear orthogonality of LeWM latent representations organized by physical factor**. This is the mechanistic contribution.

**Where the practical gaps are:**
- The gap between "SWM supports physical FoV" and "someone has actually trained LeWM on multiple physical factor combinations and evaluated cross-domain transfer" is the entire contribution space.
- The gap between "LoRA works on ViT-Base" and "LoRA works on ViT-Tiny in a world model context" is an engineering risk that must be addressed early.

---

## Phase 2: Initial Candidates

### Candidate A: Systematic Physical Factor Holdout Probing of LeWM

- **Hypothesis**: LeWM trained on a subset of physical factor combinations (e.g., {low friction, medium gravity} and {high friction, low gravity}) will show degraded zero-shot probing accuracy on held-out combinations (e.g., {low friction, low gravity}), and the degradation pattern will correlate with the dissimilarity of the held-out combination to training combinations.

- **Implementation sketch**:
  1. Use `stable-worldmodel` PushT environment with its built-in FoV system to create datasets with 2-3 physical factors varied (friction x gravity x damping) at 2-3 levels each.
  2. Design a systematic holdout: train on N-k combinations, test zero-shot on k held-out combinations. Use the combinatorial holdout methodology from Uselis et al.
  3. Train LeWM (ViT-Tiny, ~15M params) on the training split using the existing le-wm codebase. Training takes ~2-4 hours on a single GPU.
  4. Freeze the trained encoder, run linear + MLP probing on held-out combinations. Probe for agent position, block position, block velocity.
  5. Measure CEM planning success rate on held-out combinations as a downstream metric.
  6. Apply principal angle analysis (Uselis et al. code) to measure whether latent representations factorize linearly by physical factor.

- **Simplest version**: Train LeWM on 2 friction levels x 2 gravity levels (4 combinations), hold out 1 combination, probe zero-shot. This is a 2x2 factorial design with 1 holdout. Total: 3 training datasets + 1 test dataset.

- **Time estimate**: ~4 hours training per LeWM model (single GPU). 4 factor combinations = 4 training runs = ~16 GPU-hours total. Probing is <30 min per model. Principal angle analysis is <5 min. **Total: ~20 GPU-hours, feasible in 2-3 days on a single GPU.**

- **Reusable components**: le-wm codebase (model + training), stable-worldmodel (environment + FoV + probing + planning), necessary-compositionality (principal angle analysis).

### Candidate B: LoRA Adaptation Efficiency as a Generalization Diagnostic

- **Hypothesis**: Applying LoRA to the LeWM encoder (or predictor, or both) and fine-tuning on a small number of samples from held-out physical factor combinations will partially recover zero-shot performance, and the recovery efficiency (samples needed) will be diagnostic of which physical concepts are transferable vs. domain-specific.

- **Implementation sketch**:
  1. Start from the trained LeWM model from Candidate A (pretrained on subset of factor combinations).
  2. Add LoRA adapters to the ViT-Tiny encoder's attention Q/V projections using PEFT or LoRA-ViT.
  3. Fine-tune on {10, 50, 100, 500} samples from the held-out combination.
  4. Measure probing accuracy and planning success rate after LoRA adaptation.
  5. Compare: encoder-only LoRA vs. predictor-only LoRA vs. both. This reveals whether the generalization bottleneck is in perception (encoder) or dynamics prediction (predictor).

- **Simplest version**: Single held-out combination, LoRA on encoder Q/V only, 3 sample sizes. ~30 min fine-tuning per run on a single GPU (LoRA on 15M model is fast).

- **Time estimate**: ~6-8 LoRA fine-tuning runs at ~30 min each = ~4 GPU-hours. Plus Candidate A as prerequisite.

- **Reusable components**: le-wm (base model), PEFT library (LoRA implementation), stable-worldmodel (data + probing + evaluation).

- **Risk flag**: LoRA on ViT-Tiny is known to underperform relative to LoRA on larger ViTs. The adaptation signal may be weak. **Mitigation**: Also test full fine-tuning of the encoder as a ceiling, and linear probing as a floor. If LoRA fails, the comparison of "full fine-tune vs. zero-shot" still provides the adaptation efficiency diagnostic.

### Candidate C: SIGReg Geometry Analysis -- Does the Gaussian Prior Enable Compositional Structure?

- **Hypothesis**: LeWM's SIGReg regularizer, which enforces Gaussian-distributed latent embeddings, implicitly encourages the linear orthogonal factorization that theory (Uselis et al., 2026) identifies as necessary for compositional generalization. Comparing LeWM (with SIGReg) to an ablated version (without SIGReg, or with VICReg instead) will show that SIGReg produces representations with higher per-factor orthogonality.

- **Implementation sketch**:
  1. Train LeWM with SIGReg (default) and LeWM without SIGReg (ablation) on the same multi-factor dataset.
  2. Extract latent embeddings for each physical factor combination.
  3. Compute per-factor representation subspaces, measure principal angles between them, and compute the linear factorization score from Uselis et al.
  4. Compare: does SIGReg produce more orthogonal factor subspaces? Does this correlate with better zero-shot probing on held-out combinations?

- **Simplest version**: 2 models (SIGReg vs. no-SIGReg), same dataset, same probing. The analysis itself is <30 min of numpy computation.

- **Time estimate**: 2 additional training runs = ~8 GPU-hours. Analysis is negligible.

- **Reusable components**: le-wm (model + training with configurable loss terms), necessary-compositionality (orthogonality metrics).

---

## Phase 3: Self-Critique

### Against Candidate A: Systematic Physical Factor Holdout Probing

- **Implementation reality check**: The critical dependency is whether SWM's PushT FoV system actually supports programmatic variation of friction and gravity during data collection, and whether the le-wm training pipeline accepts data from modified environments without code changes. **Searched for SWM documentation**: SWM explicitly documents FoV control via `options` argument in `world.reset()` and `world.record_dataset()`, including physical parameters. The le-wm training pipeline reads HDF5 datasets, which are environment-agnostic. **Verdict: implementation path is clear and low-risk.**

- **Reproducibility attack**: The main hyperparameter sensitivity is in LeWM training itself (lambda for SIGReg). The paper reports robustness across lambda in [0.01, 0.2]. Our factorial design with multiple factor combinations actually strengthens reproducibility -- each training run is on a well-defined data distribution. **Verdict: reproducible with standard seeds (3 seeds per run recommended for full experiments, 1 for pilot).**

- **Baseline sanity check**: The zero-shot probing baseline is the LeWM trained only on the training-split factor combinations. The ceiling is LeWM trained on ALL factor combinations (including the test ones). This bracketing ensures the comparison is fair. Additionally, a random embedding baseline (untrained encoder) provides the floor. **Verdict: clean baseline structure.**

- **Scope attack**: PushT is a 2D environment. Results may not generalize to 3D manipulation (OGBench) or locomotion. **Mitigation**: Run a secondary experiment on OGBench-Cube (3D) or Reacher (DMControl) to test scope. **Verdict: moderate risk, but PushT + one additional environment is sufficient for a first paper.**

- **Verdict: STRONG**

### Against Candidate B: LoRA Adaptation Efficiency

- **Implementation reality check**: LoRA on ViT-Tiny is documented to underperform. The le-wm ViT-Tiny encoder has ~5M parameters. With LoRA rank r=4, we add ~147K trainable parameters to an already small model. The signal-to-noise ratio may be poor. **Searched for practical experience**: The empirical study by Preprints.org (2025) confirms "LoRA's accuracy degrades significantly with smaller DeiT-T and DeiT-S models" due to "reduced representational capacity." **Verdict: real engineering risk.**

- **Reproducibility attack**: LoRA fine-tuning is sensitive to learning rate and rank. With a 15M parameter model, the hyperparameter space is small, but the interaction between LoRA rank, learning rate, and number of fine-tuning samples creates a non-trivial search space. **Mitigation**: Fix rank at r=4 and r=16, use standard PEFT learning rates. **Verdict: manageable with fixed hyperparameters.**

- **Baseline sanity check**: If LoRA fails (no improvement over zero-shot), the experiment still provides useful information: it means the generalization gap is not easily bridgeable by low-rank perturbations, implying the learned representations are not linearly separable by physical factor. Include full fine-tuning as a ceiling comparison. **Verdict: even a negative result is informative.**

- **Scope attack**: The LoRA component adds complexity without guaranteed payoff. If the zero-shot probing in Candidate A already shows strong results, LoRA may be unnecessary. If it shows weak results, LoRA may not help on a tiny model. **Verdict: LoRA is a supporting experiment, not the main contribution.**

- **Verdict: MODERATE** (high risk on LoRA-for-small-ViT, but informative either way)

### Against Candidate C: SIGReg Geometry Analysis

- **Implementation reality check**: Ablating SIGReg from LeWM requires modifying the loss function. The le-wm codebase uses only 2 loss terms (prediction + SIGReg), so removing SIGReg means training with prediction loss only. **Risk**: Without SIGReg, the model may collapse (this is the whole point of SIGReg). A collapsed model cannot be meaningfully compared. **Mitigation**: Use VICReg as an alternative regularizer (le-wm's predecessor PLDM uses VICReg-style losses). This provides a non-collapsed baseline with a different regularizer.

- **Reproducibility attack**: The principal angle analysis depends on how factor subspaces are defined. Need to collect enough samples per factor combination to estimate subspaces reliably. With 192-dim latent space and ~1000 samples per combination, this should be fine. **Verdict: reproducible.**

- **Baseline sanity check**: Comparing SIGReg to VICReg is a meaningful ablation. But the causal claim ("SIGReg causes compositional structure") requires controlling for all other differences. Since LeWM is minimal (only 2 loss terms), the ablation is clean. **Verdict: clean ablation design.**

- **Scope attack**: This is a mechanistic analysis. It explains WHY compositional generalization works (or doesn't), but doesn't directly demonstrate practical utility. It's scientifically interesting but may not stand alone as a paper. **Verdict: strong as a supporting analysis for Candidate A, weak as standalone.**

- **Verdict: MODERATE** (strong science, but best as a component of Candidate A, not standalone)

---

## Phase 4: Refinement

### Dropped Ideas
None fully dropped. Candidates B and C are both MODERATE but serve as important supporting experiments for the STRONG Candidate A.

### Strengthened Design

The refined idea is a **unified study** that combines all three candidates into a single coherent paper:

1. **Core contribution (Candidate A)**: Systematic physical factor holdout evaluation of LeWM. This is the main experiment.
2. **Diagnostic experiment (Candidate B, simplified)**: LoRA + full fine-tuning adaptation curves on held-out combinations. If LoRA fails, report the negative result and use full fine-tuning as the adaptation diagnostic.
3. **Mechanistic analysis (Candidate C, simplified)**: Principal angle analysis of LeWM representations organized by physical factor. SIGReg vs. no-SIGReg ablation.

### Simplifications Applied
- **Candidate B**: Instead of a full LoRA sweep, do LoRA-r4 and LoRA-r16 on encoder Q/V only, plus full fine-tuning as a ceiling. 3 configurations x 3 sample sizes = 9 runs. If LoRA is weak, just report the negative finding and focus on full fine-tuning curves.
- **Candidate C**: No need for a separate VICReg model. Just analyze the SIGReg model's latent geometry. If there's time, add the no-SIGReg ablation.

### Pilot Experiment Design (< 15 min)
1. Use SWM PushT with 2 friction levels (low, high) -- single factor, 2 levels.
2. Train LeWM for reduced epochs (25% of default) on low-friction data only.
3. Probe on high-friction data (zero-shot).
4. If probing accuracy drops meaningfully (>10% relative degradation), the experimental framework is validated.
5. Time estimate: ~10 min training + ~2 min probing = 12 min total.

### Selected Front-Runner
**Candidate A** (with B and C as supporting experiments) -- highest success probability because:
- All code exists and is usable
- The experimental design is a standard factorial design with well-understood statistical properties
- Even partial results (zero-shot probing only, without LoRA) constitute a publishable contribution
- The SWM FoV system eliminates the biggest engineering risk (building custom environments)

---

## Phase 5: Final Proposal

### Title
Compositional Physical Generalization in Latent World Models: A Systematic Evaluation of LeWorldModel under Factor-Combination Holdout

### Hypothesis
LeWorldModel's latent representations, constrained by SIGReg to follow a Gaussian distribution, encode physical concepts (friction, gravity, damping) in a partially factored manner that supports moderate zero-shot generalization to unseen factor combinations, but with systematic failure modes when held-out combinations require extrapolation rather than interpolation in factor space. LoRA adaptation of the encoder can partially recover performance on interpolation-type holdouts but not extrapolation-type holdouts.

### Motivation
World models that learn physics from pixels are increasingly capable (LeWM, DINO-WM, V-JEPA 2), but whether they genuinely understand physical concepts compositionally -- i.e., can recognize gravity in a new friction context -- is untested. This matters practically: a robot trained in one factory setting should adapt to a different factory without full retraining. The compositional generalization literature (Uselis et al., 2026; Okawa et al., 2025) provides theoretical conditions (linear orthogonal representations) but has only been tested on vision encoders, never on world model predictors that must also model dynamics. LeWM is the ideal testbed: small enough to train quickly, clean enough to analyze mechanistically (only 2 loss terms), and supported by SWM infrastructure that already exposes physical factors of variation.

### Method

**Step 1: Environment + Data Setup (Day 1)**
- Use SWM PushT environment with FoV control.
- Define 3 physical factors: friction coefficient (3 levels: low/medium/high), gravity magnitude (3 levels: 0.5x/1.0x/1.5x standard), damping coefficient (2 levels: low/high).
- Total: 3 x 3 x 2 = 18 factor combinations.
- Training split: 14 combinations (covering all individual factor levels). Holdout: 4 combinations chosen to include both interpolation-type (surrounded by training points in factor space) and extrapolation-type (at the boundary of the factor space).
- Collect 1000 expert trajectories per combination using SWM's `world.record_dataset()` with FoV `options`. Total: 18,000 trajectories saved as HDF5.
- Libraries: `stable-worldmodel` (environment + data collection), `gymnasium`.

**Step 2: LeWM Training (Days 2-4)**
- Train LeWM (ViT-Tiny encoder + transformer predictor, ~15M params total) on the 14-combination training split.
- Use default le-wm hyperparameters (lambda_SIGReg in [0.01, 0.2]). Training config via YAML.
- Train 3 seeds for the full experiment, 1 seed for pilot.
- Hardware: single GPU (the le-wm paper reports training in "a few hours" on a single GPU).
- As an ablation, train one model on ALL 18 combinations (ceiling) and one on a single combination (floor).
- Libraries: `le-wm` (model + training), `stable-pretraining` (training utilities).

**Step 3: Zero-Shot Probing (Days 4-5)**
- Freeze the trained encoder.
- Train linear and MLP regression probes on frozen latent embeddings to predict: agent position (x, y), block position (x, y), block orientation, block velocity.
- Evaluate on training combinations (in-distribution) and held-out combinations (OOD).
- Metric: R-squared for probe regression accuracy, per-factor and per-combination.
- Also run CEM planning on held-out combinations to measure downstream task performance (success rate, coverage percentage).
- Libraries: `scikit-learn` (linear probing), `stable-worldmodel` (planning evaluation, CEM solver).

**Step 4: LoRA Adaptation (Days 5-6)**
- Apply LoRA (rank 4 and rank 16) to the frozen LeWM encoder's Q and V projection matrices using HuggingFace PEFT.
- Fine-tune on {10, 50, 100, 500} samples from each held-out combination.
- Measure probing accuracy and planning success after adaptation.
- Compare: (a) LoRA on encoder only, (b) LoRA on predictor only, (c) full fine-tuning of encoder as ceiling, (d) zero-shot as floor.
- If LoRA shows weak signal (expected risk on ViT-Tiny), report the negative result and focus on the full fine-tuning adaptation curves as the diagnostic.
- Libraries: `peft` (LoRA implementation), `le-wm` (model loading).

**Step 5: Representation Geometry Analysis (Day 6)**
- Extract latent embeddings from the trained LeWM for all 18 factor combinations.
- For each physical factor (friction, gravity, damping), compute the representation subspace (PCA of embeddings at each factor level).
- Measure principal angles between factor-level subspaces.
- Compute the linear factorization score (Uselis et al.) to quantify how compositionally structured the representations are.
- Correlate factorization score with zero-shot probing accuracy on held-out combinations.
- Libraries: `numpy`, `scipy.linalg` (principal angles), code from `necessary-compositionality` repo.

### Simplest Version
Train LeWM on 3 PushT friction levels (low/medium/high), hold out the "high" level, probe zero-shot. This is a 1-factor, 3-level design with 1 holdout. Can be done in under 8 hours on a single GPU including data collection, training, and probing.

### Baselines

1. **Zero-shot baseline**: LeWM trained only on training-split combinations, evaluated directly on held-out combinations without any adaptation. Expected: moderate degradation on held-out combinations, with larger drops on extrapolation-type holdouts.

2. **Oracle baseline**: LeWM trained on ALL 18 combinations including the test combinations. Expected: strong performance everywhere. The gap between this and the zero-shot baseline quantifies the "compositional generalization gap."

3. **Random embedding baseline**: Untrained LeWM encoder (random weights). Expected: near-chance probing accuracy. Establishes the floor.

4. **DINO-WM baseline** (if compute allows): Frozen DINOv2 encoder + learned predictor, trained on same data. Tests whether pre-trained visual features provide better compositional structure than end-to-end training. DINO-WM code is available.

### Experimental Plan

| Phase | Duration | GPU-hours | Output |
|-------|----------|-----------|--------|
| Pilot: single-factor holdout | Day 1 | ~2 | Validates framework feasibility |
| Data collection (18 combinations) | Day 1-2 | ~2 | 18 HDF5 datasets |
| LeWM training (3 seeds, 3 models) | Days 2-4 | ~36 | Trained models (training-split, all-split, single-split) |
| Zero-shot probing | Days 4-5 | ~4 | Probing accuracy table, generalization heatmap |
| LoRA / fine-tuning adaptation | Days 5-6 | ~8 | Adaptation curves |
| Geometry analysis | Day 6 | ~1 | Principal angle analysis, factorization scores |
| **Total** | **~6 days** | **~53** | |

Metrics:
- Primary: R-squared of linear/MLP probes on held-out combinations (relative to in-distribution)
- Secondary: CEM planning success rate on held-out combinations
- Diagnostic: Adaptation efficiency (samples to reach 90% of oracle performance)
- Mechanistic: Linear factorization score (principal angle analysis)

Ablation schedule:
1. Number of held-out combinations (1, 2, 4)
2. Training data size (500, 1000, 2000 trajectories per combination)
3. SIGReg lambda (0.01, 0.05, 0.1, 0.2)
4. LoRA rank (4, 16) -- may be dropped if LoRA fails on ViT-Tiny
5. Adaptation component (encoder only, predictor only, both)

### Resource Estimate
- **GPU**: Single GPU (the paper trains LeWM on a single GPU). A100 or RTX 3090 recommended. Total ~53 GPU-hours for full experiment, ~2 GPU-hours for pilot.
- **Wall-clock time**: ~6 working days for full experiment, ~4 hours for pilot.
- **Model sizes**: LeWM is ~15M parameters (ViT-Tiny encoder ~5M + transformer predictor ~10M). Fits comfortably in <4GB VRAM. LoRA adds negligible overhead.
- **Storage**: ~18 HDF5 datasets at ~100MB each = ~1.8GB total.

### Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| SWM FoV system doesn't support gravity/damping variation in PushT | High | Low | SWM documentation explicitly lists physical parameters as FoV. Fallback: use DMControl Reacher which natively supports physics parameter modification via MuJoCo XML. |
| LeWM training collapses on modified-physics data | Medium | Low | SIGReg is designed to prevent collapse. The paper shows robustness across environments. If specific factor combinations cause issues, reduce the factor range. |
| LoRA provides no improvement on ViT-Tiny | Medium | High | **Expected.** Report as a negative result. Use full fine-tuning as the adaptation diagnostic instead. The paper's contribution does not depend on LoRA succeeding. |
| Zero-shot probing shows no degradation on held-out combinations (LeWM already generalizes perfectly) | Medium | Medium | Would be a strong positive result ("LeWM compositionally generalizes"). Pivot to studying WHY it generalizes (SIGReg geometry analysis becomes the main contribution). Also test with more extreme holdouts (extrapolation rather than interpolation). |
| PushT is too simple to show interesting compositional effects | Medium | Medium | Add OGBench-Cube (3D) or Reacher (DMControl) as a second environment. PushT results still contribute even if effects are smaller than expected. |
| Compute budget exceeded (>60 GPU-hours) | Low | Low | Reduce number of seeds from 3 to 1 for ablation experiments. Reduce factor combinations from 18 to 8 (2x2x2 instead of 3x3x2). |

### Novelty Claim

**What is new:**
1. The first systematic evaluation of cross-domain compositional generalization for a JEPA-family world model (LeWM), using a controlled factorial design over physical parameters.
2. The first application of compositional generalization theory (linear orthogonal representations) to world model latent spaces, testing whether SIGReg's Gaussian prior induces the geometric structure required for compositional transfer.
3. The first attempt to use LoRA adaptation efficiency as a diagnostic tool for world model generalization, revealing which components (encoder vs. predictor) encode domain-specific vs. transferable physical knowledge.

**What is NOT new (and we should be honest about):**
- The model (LeWM) is not ours.
- The environments (SWM PushT) are not ours.
- The probing protocol is standard.
- The principal angle analysis code exists.

**The novelty is in the experimental design and the questions asked, not in the methods used.** This is a strength for an evaluation/analysis paper: the contribution is empirical insight, not a new algorithm. The closest analogue in style is the "Does Data Scaling Lead to Visual Compositional Generalization?" paper (arXiv:2507.07102), which used existing tools to answer a novel question about compositional generalization.
