# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **LeWorldModel (arXiv:2603.19312)** -- The primary subject. Ablation studies on SIGReg projection count and embedding dimensionality are clean; probing protocol uses linear and MLP regression heads on frozen latent embeddings to predict agent position, block pose, end-effector position. Evaluation via CEM planning success rate on 50 trajectories, 3 training seeds. Key limitation: no cross-domain compositional evaluation; only in-distribution probing per environment.

2. **stable-worldmodel-v1 (arXiv:2602.08968)** -- The evaluation infrastructure LeWM depends on. Crucially, it already supports DMControl environments with "factors of variation" that change visual, geometric, and physical properties. This is the backbone for any controlled compositional experiment: environment parameterization is built in. Supports HDF5 data format, Gymnasium interface, CEM/MPPI planning solvers.

3. **DINO-WM (arXiv:2411.04983)** -- Physical state probing protocol and zero-shot planning evaluation (CEM in latent space). Tested generalization to unseen maze wall configurations (SR=0.82), but only single-factor visual generalization, not multi-factor physics combinations.

4. **Compositional Generalization Requires Linear, Orthogonal Representations (Uselis et al., arXiv:2602.24264)** -- Formalizes the geometric conditions: divisibility, transferability, stability. Representations must decompose linearly with orthogonal per-concept components. Provides concrete metrics (principal angle analysis, DCI disentanglement score) applicable to any latent space. Validated on CLIP, SigLIP, DINO -- but not on world model predictors.

5. **Does Data Scaling Lead to Visual Compositional Generalization? (arXiv:2507.07102)** -- Shows compositional generalization driven by data *diversity* not scale; increased combinatorial coverage forces linear factored representations. Directly relevant: training data diversity (number of factor combinations seen) may be the key variable, not model capacity.

6. **CARL Benchmark (arXiv:2110.02102)** -- Contextual Adaptive RL benchmark that exposes DMControl physics parameters (gravity, friction, joint stiffness, torso mass, actuator strength) as configurable context features. Provides the infrastructure for systematic hold-out of parameter combinations. The key missing piece: nobody has used CARL-style factor variation to evaluate *world model* latent representations (only RL policy generalization).

7. **CLEVR-CoGenT** -- The canonical protocol for compositional holdout evaluation: train on Condition A (certain attribute-shape pairings), test on Condition B (swapped pairings). The A->B accuracy drop quantifies reliance on spurious conjunctions vs. factorized representations. This evaluation design maps directly to physics factor combinations.

8. **TD-JEPA (arXiv:2510.00739)** -- Zero-shot RL from JEPA latent representations. Provides a rigorous ablation template: (i) zero-shot evaluation, (ii) ablation over prediction target, (iii) symmetric variant comparison, (iv) fast adaptation from pretrained representations. The adaptation evaluation is relevant to our LoRA diagnostic.

9. **V-JEPA 2 (arXiv:2506.09985)** -- Evaluation methodology for zero-shot physical planning: 4-layer probe classifiers on frozen encoders for appearance and motion understanding. Zero-shot deployment on unseen labs. Statistical reporting: success rate with 95% CI over multiple rollouts.

10. **AIM Framework / Probing the Latent World (arXiv:2603.20327)** -- Passive quantization probe for JEPA latent space. Chi-squared test (p < 1e-4) for statistical separability of physical dimensions. Limited to 3 dimensions on Kinetics-mini -- but the methodology (discretize latent dims, test independence via chi-squared) is directly transferable.

11. **Scalable Evaluation for Compositional Generalization (arXiv:2511.02667)** -- Reduces evaluation cost from combinatorial to constant complexity. Introduces distribution-based compositionality assessment (DBCA) for standardized train/test splits. Trained >5000 models for statistical robustness.

12. **The Linear Representation Hypothesis (Park et al., ICML 2024)** -- Causal inner product formalization: causally separable concepts are represented as orthogonal vectors. Linear probing and model steering connect to this structure. The causal inner product can be estimated empirically and compared against Euclidean inner product to test whether SIGReg's Gaussian prior induces the right geometry.

### Experimental Landscape

**What has been properly tested:**
- LeWM in-distribution probing: linear and MLP probes for physical state variables (agent pos, block pos/vel, EE pos) within each environment separately. 3 seeds, 50 trajectories.
- LeWM planning: CEM success rate on PushT, Reacher, TwoRoom, OGBench-Cube. LeWM beats PLDM on all challenging tasks, beats DINO-WM on PushT and Reacher.
- SIGReg ablation: projection count and embedding dimension sensitivity. Stable across variations.
- DINO-WM generalization: unseen maze walls, unseen object shapes (single-factor visual perturbations).

**What is accepted without proper evidence:**
- The claim that SIGReg's Gaussian prior encourages compositional structure. This is speculated but never measured. Nobody has quantified the linear factorization degree of LeWM's latent space by physical factor.
- The assumption that LeWM's physical probing accuracy on in-distribution data implies the model "understands" the underlying physics. Probing accuracy within the training distribution could be driven by superficial correlations, not factorized physical understanding.
- The implicit claim that LeWM generalizes across physical parameter variations. The paper only tests within the default parameter settings of each environment. No gravity/friction/mass variation has been tested.
- The claim that the Two-Room failure is due to "low diversity." This is hypothesized but not confirmed by a systematic diversity sweep.

**Methodological gaps:**
1. No CoGenT-style compositional holdout evaluation exists for any JEPA world model.
2. No statistical significance testing (bootstrap CI, permutation tests) accompanies the probing or planning results in LeWM -- only mean +/- std over 3 seeds.
3. No systematic study varying environment diversity along a controlled gradient.
4. No ablation separating encoder vs. predictor contributions to generalization/failure.
5. LoRA as a diagnostic tool for identifying which components encode domain-specific vs. transferable knowledge has not been attempted.

## Phase 2: Initial Candidates

### Candidate A: CoGenT-style Compositional Holdout Benchmark for LeWM

- **Hypothesis**: LeWM's probing accuracy will drop by >= 25% (relative) and planning success rate by >= 30% (relative) on held-out *combinations* of physics parameters that were individually seen during training, compared to in-distribution performance. This drop will be statistically significant (paired t-test, p < 0.05, across 5 seeds).

- **Falsification criterion**: If held-out combination performance drops by < 10% relative on both probing and planning, the hypothesis is falsified -- LeWM achieves compositional generalization without additional intervention.

- **Evaluation protocol**:
  - **Primary environment**: PushT (best LeWM results, rich physical state, well-established baselines).
  - **Physical factors**: Friction coefficient (3 levels: 0.5x, 1x, 2x default), agent mass (3 levels: 0.5x, 1x, 2x), block mass (3 levels: 0.5x, 1x, 2x). Total: 27 combinations.
  - **Train set**: 18 of 27 combinations (2/3), ensuring each individual factor level appears in >= 6 training combinations.
  - **Holdout set**: 9 combinations where at least 2 factors simultaneously take values not co-occurring in training.
  - **Metrics**: (1) Linear probing R^2 for agent pos, block pos, block orientation. (2) MLP probing R^2 for same variables. (3) CEM planning success rate (50 trajectories per combination, goal 25 steps ahead). (4) Latent prediction MSE (predictor accuracy on held-out factor combos).
  - **Statistical tests**: Paired t-test (in-dist vs. holdout) across 5 training seeds. Bootstrap 95% CI on all metrics. Effect size (Cohen's d).

- **Ablation plan**:
  1. Single-factor holdout: hold out one factor level at a time. Isolates per-factor generalization.
  2. Two-factor holdout: hold out pairs. Tests pairwise composition.
  3. Three-factor holdout: the full compositional test.
  4. Each ablation tells us: does generalization degrade gracefully (linear in held-out factors) or catastrophically (super-linear)?

- **Confounders**:
  - Data volume: holding out 9/27 combos reduces training data by 33%. Must control for this with a "matched volume" baseline trained on 18/27 combos drawn randomly (no systematic holdout).
  - Visual confounds: physics parameter changes may subtly alter visual appearance (object speed, trajectory shape). Must verify that probing failure is not driven by OOD visual features.
  - Hyperparameter sensitivity: lambda (SIGReg weight) may need re-tuning per physics regime. Test with fixed vs. re-tuned lambda.

- **Pilot design**: Train on 2 friction levels (0.5x, 1.5x), test on 1x (interpolation) and 2.5x (extrapolation) in PushT. 100 trajectories each, 1 seed. ~10 min on single GPU. If probing R^2 drops < 5% on interpolation but > 15% on extrapolation, we have signal for a systematic study.

### Candidate B: LoRA Adaptation Efficiency as Generalization Diagnostic

- **Hypothesis**: Applying LoRA (rank 4) to the LeWM predictor alone will recover >= 80% of the in-distribution probing accuracy on held-out factor combinations using only 50 target-domain trajectories, while applying LoRA to the encoder alone will recover <= 40%. This asymmetry reveals that the predictor, not the encoder, is the bottleneck for cross-domain transfer.

- **Falsification criterion**: If encoder-only LoRA and predictor-only LoRA perform within 10% of each other on held-out combinations, the hypothesis about predictor-specific bottleneck is falsified.

- **Evaluation protocol**:
  - Use the same holdout split from Candidate A.
  - Four conditions: (1) Zero-shot (no adaptation), (2) LoRA on encoder only (ViT attention layers), (3) LoRA on predictor only (transformer attention layers), (4) LoRA on both.
  - Vary adaptation data: 10, 25, 50, 100, 200 trajectories from target domain.
  - Metrics: same as Candidate A + adaptation efficiency curve (probing R^2 vs. number of adaptation trajectories).
  - Statistical tests: Two-way ANOVA (adaptation target x data volume), Tukey HSD post-hoc.

- **Ablation plan**:
  1. LoRA rank ablation: rank 1, 2, 4, 8, 16. Tests capacity requirements.
  2. Selective layer ablation: LoRA on first half vs. second half of encoder/predictor layers. Tests where domain-specific information is encoded.
  3. Fine-tuning baseline: full fine-tuning with same data budget. Tests whether LoRA's low-rank constraint acts as beneficial regularizer.

- **Confounders**:
  - Overfitting with small adaptation sets. Must use held-out validation within target domain.
  - LoRA may improve performance through generic fine-tuning rather than domain-specific adaptation. Control: apply LoRA with *in-distribution* data and measure improvement (should be minimal if model is already well-trained).

- **Pilot design**: Zero-shot vs. LoRA-predictor (rank 4, 50 trajectories) on 1 held-out friction level in PushT. 1 seed. ~12 min (5 min for LoRA fine-tuning + 7 min for evaluation).

### Candidate C: SIGReg Geometry and Compositional Structure Analysis

- **Hypothesis**: LeWM trained with SIGReg will exhibit higher principal angle orthogonality (measured as mean cosine similarity between per-factor subspaces < 0.15) between latent dimensions encoding different physical factors, compared to a LeWM variant trained without SIGReg (using VICReg instead), where mean cosine similarity will be > 0.30. This geometric structure will predict compositional generalization performance (Pearson r > 0.7 between orthogonality score and holdout accuracy).

- **Falsification criterion**: If SIGReg and VICReg models show similar orthogonality scores (difference < 0.05 in mean cosine similarity), the mechanistic hypothesis linking Gaussian prior to compositional structure is falsified.

- **Evaluation protocol**:
  - Train LeWM on PushT with 3 factor levels per 3 factors (27 combos).
  - Extract latent embeddings for each factor combination.
  - Compute per-factor subspaces via PCA on embeddings grouped by factor level.
  - Measure: (1) Mean cosine similarity between per-factor principal components (lower = more orthogonal). (2) DCI disentanglement score (Eastwood & Williams, 2018). (3) Mutual information between latent dimensions and ground-truth factors (sklearn mutual_info_regression).
  - Compare SIGReg vs. VICReg vs. no-regularizer (collapse expected).
  - Correlate geometry metrics with holdout probing/planning accuracy from Candidate A.

- **Ablation plan**:
  1. SIGReg lambda sweep (0.01, 0.1, 1.0, 10.0): does stronger Gaussian enforcement increase orthogonality?
  2. Embedding dimension sweep (64, 128, 192, 256): does overcomplete representation help factorization?
  3. Training data diversity sweep: do we need more factor combinations to see factorization?

- **Confounders**:
  - Correlation between orthogonality and performance could be driven by a third variable (e.g., overall representation quality). Control: measure R^2 of in-distribution probing as a covariate.
  - PCA-based subspace estimation may be unreliable with small sample sizes per factor. Use >= 500 embeddings per factor combination.

- **Pilot design**: Train SIGReg vs. VICReg on default PushT (no factor variation). Extract embeddings, compute DCI score on agent-pos vs. block-pos dimensions. 1 seed, ~8 min. If DCI differs by > 0.1 between the two, proceed to full study.

## Phase 3: Self-Critique

### Against Candidate A

- **Confound attack**: The biggest uncontrolled variable is *visual confound from physics changes*. When friction changes, object trajectories change shape; when mass changes, speeds change. The model may fail on held-out combinations because the *visual* distribution shifted, not because compositional physical reasoning failed. **Mitigation**: add a "visual oracle" control -- train a separate model on the same images but with randomized physics labels. If this oracle's probing accuracy also drops on holdout combos, the drop is visual, not compositional. Additionally, searched for papers on visual confounds in physics prediction (none found specifically for JEPA models, confirming this is a genuine gap).

- **Statistical attack**: 5 training seeds with 50 evaluation trajectories per combo gives 5 x 50 = 250 data points per condition. For a 25% relative drop in probing R^2 (e.g., from 0.80 to 0.60), Cohen's d ~ 1.0, which is detectable with n=25 per condition at power=0.99. The sample size is adequate. However, 5 seeds may be marginal for the paired t-test across seeds -- consider 7 seeds.

- **Benchmark attack**: PushT is a 2D environment with relatively simple physics. Results may not generalize to 3D environments. However, PushT is the LeWM flagship benchmark with the best-established baselines, making it the right first target. Should include at least one 3D environment (OGBench-Cube or Reacher) as a secondary benchmark. The stable-worldmodel library supports these, so infrastructure cost is low.

- **Ablation completeness attack**: The single-factor vs. two-factor vs. three-factor progression is well-designed. Missing: an ablation for training data *composition* -- does it matter *which* 18/27 combos are in training, or just the count? Should run 3 different random splits and compare variance.

- **Verdict**: **STRONG** -- This is a well-defined, falsifiable experiment with clear controls. The visual confound is the main risk, addressable with the oracle control. Statistical power is adequate.

### Against Candidate B

- **Confound attack**: LoRA adaptation on a small dataset risks *memorizing* the target distribution rather than learning transferable adaptation. With only 50 trajectories, the effective "adaptation" may be fitting to specific trajectories rather than learning the physics change. **Mitigation**: evaluate on held-out trajectories from the target domain (not used in LoRA training). Also, the "in-distribution LoRA" control is critical -- if LoRA improves in-distribution performance substantially, the diagnostic value is undermined.

- **Statistical attack**: The two-way ANOVA requires balanced cell sizes. With 5 adaptation data sizes x 4 LoRA targets x 5 seeds = 100 experimental runs, this is manageable but expensive. Each LoRA training takes ~5 min, so total ~500 min = ~8 hours. Within the 1-hour-per-task constraint, this must be batched across multiple GPU hours. Pilot first with 2 data sizes x 2 targets x 2 seeds = 8 runs (~40 min).

- **Benchmark attack**: LoRA rank 4 on a 15M parameter model is a very aggressive bottleneck (~0.1% of parameters). The rank may be too low to learn meaningful adaptation, leading to a false negative. Should include rank 8 and 16 in the design.

- **Ablation completeness attack**: Missing a "random LoRA" control -- LoRA with random (untrained) adapters. This would test whether the structured low-rank perturbation itself is beneficial vs. any perturbation. Also missing: head-only fine-tuning baseline (just the probing head, not LoRA on the backbone).

- **Verdict**: **MODERATE** -- Interesting diagnostic but relies on several assumptions about what LoRA is actually learning. The rank sensitivity and overfitting risks reduce confidence. Strengthen by adding the random LoRA and head-only baselines.

### Against Candidate C

- **Confound attack**: Orthogonality of per-factor subspaces can be an artifact of PCA: if factors have different variance scales, PCA may separate them into orthogonal components for purely statistical reasons unrelated to "compositional understanding." **Mitigation**: use CKA (Centered Kernel Alignment) as an alternative metric that is invariant to isotropic scaling.

- **Statistical attack**: The Pearson correlation between orthogonality and holdout performance (r > 0.7) requires enough data points. With 27 factor combinations, we have 27 data points for the correlation -- marginal but feasible. However, the correlation may be confounded by difficulty gradient across factor combinations (harder combos have both worse orthogonality and worse performance for trivial reasons).

- **Benchmark attack**: Comparing SIGReg to VICReg is not apples-to-apples -- VICReg has different hyperparameters and a different loss landscape. The no-SIGReg baseline might simply be a worse model overall, making the comparison uninformative about the *mechanistic* role of the Gaussian prior.

- **Ablation completeness attack**: The SIGReg lambda sweep is important but incomplete without a separate measure of model quality (e.g., in-distribution probing R^2) at each lambda. Otherwise, we cannot distinguish "SIGReg helps orthogonality" from "SIGReg helps everything."

- **Verdict**: **MODERATE** -- Mechanistically interesting but methodologically tricky. The correlation between geometry and performance is confounded. Better as a secondary analysis within Candidate A rather than a standalone experiment.

## Phase 4: Refinement

### Dropped

**Candidate C as a standalone experiment** is dropped. The SIGReg geometry analysis is better folded into Candidate A as an analysis component -- extract latent geometry as part of the compositional holdout evaluation, correlate with performance, but do not make it the central claim. The standalone version has too many confounds and insufficient statistical power for the correlation analysis.

### Strengthened

**Candidate A (front-runner)** -- Strengthened with:
1. **Visual oracle control**: Train a model with randomized physics labels but identical image distribution. If this control also shows probing degradation on holdout combos, the effect is visual, not compositional.
2. **Multiple random splits**: Run 3 different 18/9 train/holdout splits to measure sensitivity to split choice.
3. **Matched-volume control**: Train on 18 randomly selected combos (not systematic holdout) to control for data volume reduction.
4. **Secondary benchmark**: Add Reacher (2D, different physical state structure) as a secondary environment.
5. **Geometry analysis** (from Candidate C): Compute DCI disentanglement and principal angle orthogonality on the trained model, correlate with holdout performance as secondary analysis.
6. **Increase seeds from 5 to 7** for better statistical power on the paired comparison.

**Candidate B** -- Retained as a follow-up diagnostic, strengthened with:
1. **Head-only baseline**: Fine-tune only the probing head (not LoRA on backbone) to separate representation adaptation from probe adaptation.
2. **Random LoRA control**: Apply LoRA with random untrained adapters to test whether structured perturbation matters.
3. **Expanded rank range**: Rank {2, 4, 8, 16} instead of just 4.
4. **Subset analysis**: Correlate LoRA improvement with per-combination difficulty from Candidate A.

### Selected Front-Runner

**Candidate A: CoGenT-style Compositional Holdout Benchmark for LeWM** is the front-runner because:
1. It directly answers the research question with the strongest possible evidence.
2. The falsification criterion is clear and decided before seeing results.
3. The controls (visual oracle, matched volume, multiple splits) address the most serious confounds.
4. It produces a reusable benchmark that the community can adopt.
5. Candidate B flows naturally from Candidate A's results (use the holdout split to diagnose adaptation).

## Phase 5: Final Proposal

### Title

Measuring the Compositional Generalization Boundary of Latent JEPA World Models via Systematic Physics Factor Holdout

### Hypothesis

LeWM achieves strong in-distribution probing and planning but fails to compositionally generalize: when tested on held-out combinations of physical factors (friction x agent-mass x object-mass) where each individual factor level was seen during training but the specific combination was not, probing R^2 will drop by >= 25% relative and CEM planning success rate will drop by >= 30% relative, compared to in-distribution performance. This drop will be statistically significant (paired t-test across 7 training seeds, p < 0.05, Cohen's d > 0.8).

### Falsification Criterion

The hypothesis is KILLED if:
- Holdout probing R^2 drops by < 10% relative on the primary benchmark (PushT), AND
- Holdout CEM success rate drops by < 15% relative on PushT,
- Across >= 5 of 7 training seeds.

This would mean LeWM achieves genuine compositional generalization of physical concepts, which would itself be a notable positive finding.

### Method

Train LeWM on parameterized physics environments using the stable-worldmodel library's factor-of-variation support. The key innovation is a *systematic compositional holdout* protocol inspired by CLEVR-CoGenT, applied for the first time to a JEPA world model's physical predictions.

1. **Environment parameterization**: Modify PushT (and Reacher) environments via MuJoCo XML to support 3 levels of 3 physics factors:
   - Friction coefficient: {0.5x, 1.0x, 2.0x default}
   - Agent mass: {0.5x, 1.0x, 2.0x default}
   - Block/target mass: {0.5x, 1.0x, 2.0x default}
   - This yields 3^3 = 27 physics configurations per environment.

2. **Data collection**: For each of the 27 configurations, collect 200 expert trajectories using a scripted policy or an RL-trained policy from the stable-worldmodel toolkit. Total: 5400 trajectories.

3. **Compositional holdout split**: Train on 18/27 configurations (each individual factor level appears in >= 6 training configurations). Holdout 9 configurations where at least 2 factors simultaneously take unseen co-occurring values.

4. **Training**: Train LeWM with default hyperparameters (ViT-Tiny encoder, transformer predictor, SIGReg lambda=optimized). 7 random seeds.

5. **Evaluation**: On each of 27 configurations (18 train, 9 holdout), measure:
   - Linear probing R^2 (agent pos, block pos, block orientation)
   - MLP probing R^2 (same state variables)
   - CEM planning success rate (50 trajectories, goal 25 steps ahead)
   - Latent prediction MSE (predictor accuracy)

### Evaluation Protocol

- **Primary benchmark**: PushT (established LeWM flagship)
- **Secondary benchmark**: Reacher (different physical state structure: joint angles, end-effector position)
- **Metrics**:
  | Metric | Description | Statistical Test |
  |--------|-------------|-----------------|
  | Linear probe R^2 | Regress ground-truth state from frozen encoder embeddings | Paired t-test (train vs. holdout), 7 seeds |
  | MLP probe R^2 | Same with 2-layer MLP head | Same |
  | CEM success rate | Planning success over 50 trajectories per config | Bootstrap 95% CI, paired comparison |
  | Latent prediction MSE | Predictor error on holdout vs. train configs | Paired t-test |
  | DCI disentanglement | Per-factor disentanglement of latent space | Reported for analysis, not hypothesis test |
  | Principal angle orthogonality | Mean cosine similarity between per-factor subspaces | Same |
- **Number of random seeds**: 7 (for training); 50 evaluation trajectories per config per seed
- **Significance level**: alpha = 0.05, with Bonferroni correction for 4 primary metrics

### Ablation Schedule

| # | Ablation | What It Tests | Expected Outcome |
|---|----------|--------------|-----------------|
| 1 | Single-factor holdout | Per-factor generalization | Moderate drop (10-20%), graceful degradation |
| 2 | Two-factor holdout | Pairwise compositional generalization | Larger drop (20-35%), non-linear increase |
| 3 | Three-factor holdout | Full compositional generalization | Largest drop (>30%), confirms hypothesis |
| 4 | Matched-volume control | Confound: reduced training data | Small drop (<10%), rules out data volume as explanation |
| 5 | Multiple random splits (x3) | Sensitivity to holdout choice | Low variance across splits (std < 5% absolute) |
| 6 | Visual oracle control | Confound: visual distribution shift | Minimal drop on probing, confirming effect is physics-driven |
| 7 | SIGReg lambda sweep (0.01, 0.1, 1.0, 10.0) | Does regularization strength affect compositionality | Optimal lambda for in-dist may not be optimal for holdout |

### Control Experiments

1. **Matched-volume control**: Train on 18 randomly selected (not systematically held-out) configs. If performance is similar to the systematic holdout model on in-distribution test, then the performance drop on holdout combos is due to composition, not data scarcity.

2. **Visual oracle control**: Train a model on the same image distribution but with shuffled physics-config labels. If this model also shows a probing drop on "holdout" configs, the effect is visual distribution shift, not compositional physical reasoning failure.

3. **In-distribution LoRA control** (for Candidate B follow-up): Apply LoRA with in-distribution data. If LoRA substantially improves already-good in-distribution performance, the diagnostic interpretation of LoRA improvement on holdout is weakened.

### Pilot Design

**Quick validation (~12 min on 1 GPU):**
1. Parameterize PushT friction to 3 levels: {0.5x, 1.0x, 2.0x}.
2. Collect 100 trajectories per level (300 total, ~2 min).
3. Train LeWM on 0.5x and 2.0x friction only, 1 seed (~5 min, reduced epochs).
4. Probe on all 3 levels: interpolation (1.0x) and in-distribution (0.5x, 2.0x) (~3 min).
5. Measure: if linear probe R^2 drops > 15% on 1.0x (interpolation), proceed to full study.
6. Also test 3.0x (extrapolation) to calibrate degradation curve.

**Decision rule**: If pilot shows < 5% drop on interpolation, the model generalizes well along single factors, and the more interesting question becomes whether *combinations* of factors cause failure (proceed to full study with multi-factor holdout). If pilot shows > 30% drop on interpolation, single-factor generalization is already poor, suggesting the challenge is simpler than hypothesized (still proceed, but adjust expectations).

### Resource Estimate

| Phase | Task | GPU-hours | Wall-clock |
|-------|------|-----------|------------|
| Pilot | 1-factor, 1 seed, reduced scale | 0.2 | 12 min |
| Data collection | 27 configs x 200 trajectories | 1.0 | 1 hour |
| Training | 7 seeds x 1 environment | 7.0 | 7 hours (parallel: 1 hour on 7 GPUs, 3.5 hours on 2 GPUs) |
| Probing | 7 seeds x 27 configs x 3 probes | 2.0 | 2 hours |
| Planning eval | 7 seeds x 27 configs x 50 traj | 3.0 | 3 hours |
| Controls | Matched-volume + visual oracle | 4.0 | 4 hours |
| Secondary env (Reacher) | Repeat key experiments | 5.0 | 5 hours |
| Ablations | Lambda sweep, split variations | 6.0 | 6 hours |
| **Total** | | **28.2** | **~28 hours serial, ~8 hours with 4 GPUs** |

Model size: ~15M parameters (ViT-Tiny + transformer predictor). All experiments feasible on a single consumer GPU (RTX 3090 / A5000). Individual tasks all under 1 hour.

### Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| stable-worldmodel does not support physics factor parameterization out-of-box | High | Medium | MuJoCo XML modification is straightforward; CARL benchmark shows it is possible for DMControl environments |
| Expert trajectories unavailable for all physics configs | Medium | Medium | Use random policy data or train lightweight RL agents per config (~10 min each); random policy data still enables probing |
| LeWM training fails on modified physics | Medium | Low | SIGReg is shown stable across configs; fallback to reduced factor range |
| Effect size too small to detect | Medium | Low | Pilot calibrates expected effect; 7 seeds provides adequate power for d > 0.8 |
| Visual confound dominates | High | Medium | Visual oracle control explicitly tests this; if confirmed, shift claim to "visual-physics entanglement" rather than "compositional failure" |
| Probing head overfits | Low | Low | Standard train/val split on probing data; reported in LeWM paper as stable |

### Novelty Claim

This work provides the **first systematic evaluation of compositional generalization in a JEPA world model**. The specific empirical questions being answered for the first time:

1. **Can LeWM compositionally generalize across physics factor combinations?** No prior work tests held-out *combinations* of physical parameters on any JEPA world model. Existing evaluations test single-factor generalization (DINO-WM: unseen maze walls) or in-distribution probing (LeWM: default physics only).

2. **Does the SIGReg Gaussian prior encourage the linear orthogonal structure required for compositional generalization?** This connects the theoretical framework of Uselis et al. (2026) to the practical design of JEPA world models for the first time.

3. **What is the "generalization boundary" in physics factor space?** The single-factor vs. two-factor vs. three-factor ablation progression maps the degradation curve, providing the first empirical characterization of where JEPA world model generalization breaks down.

4. **Is LoRA adaptation efficiency diagnostic of encoder vs. predictor generalization bottlenecks?** This adapts the LoRA-as-probe paradigm from vision (LoRA Recycle, CVPR 2025) to world models, identifying which components encode transferable vs. domain-specific knowledge.

The benchmark design itself is a contribution: the CoGenT-style compositional holdout protocol, adapted for physics factor combinations in continuous control environments, can be applied to any world model.
