# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **[Chanin et al., NeurIPS 2025] "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders"** (arXiv:2409.14507) — First systematic definition of feature absorption; introduced absorption rate metric via causal ablation with integrated gradients; validated on hundreds of LLM SAEs. **Key methodological insight**: The ablation-dependent metric is limited to layers 0-17 (attention moves information after layer 18), making it a conservative underestimate that misses multi-latent absorption and deep-layer phenomena.

2. **[Hu et al., 2025] "Measuring Sparse Autoencoder Feature Sensitivity"** (arXiv:2509.23717) — Proposed feature sensitivity as a new evaluation dimension using LLM-generated similar texts; found sensitivity declines with SAE width across 7 variants up to 1M features. **Key methodological insight**: Uses a 4-step protocol (collect activating examples → generate similar texts → test activation → compute sensitivity score) but lacks explicit statistical testing or confidence intervals; relies on GPT-4.1-mini generation introducing non-determinism.

3. **[Karvonen et al., ICML 2025] "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders"** (arXiv:2503.09532) — Eight-metric benchmark (absorption, sparse probing, auto-interpretability, RAVEL, unlearning, SCR, TPP, CE loss) on 200+ SAEs. **Key methodological insight**: Revealed proxy metrics (reconstruction, L0) do not reliably predict practical interpretability; LLM-dependent evaluation (GPT-4o-mini as judge) introduces non-determinism and reproducibility issues; metrics designed for same-model comparison, not cross-model.

4. **[Chanin et al., 2026] "SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data"** (arXiv:2602.14687) — Ground-truth synthetic benchmark with 16k features, hierarchy, correlation, and superposition. **Key methodological insight**: Enables controlled ablations impossible with LLMs; identified MP-SAEs exploit superposition noise without learning true features; no architecture achieves perfect performance even when Linear Representation Hypothesis is perfectly satisfied.

5. **[Bussmann et al., 2025] "Learning Multi-Level Features with Matryoshka Sparse Autoencoders"** (arXiv:2503.17547) — Nested dictionaries trained simultaneously; reduced absorption from 0.49 to 0.05. **Key methodological insight**: ~50% computational overhead; slightly worse reconstruction; feature hedging in narrow inner levels — demonstrates that architectural solutions have trade-offs that must be quantified.

6. **[Luo et al., 2026] "Building a Structured Feature Forest with Hierarchical Sparse Autoencoders"** (arXiv:2602.11881) — HSAE with explicit parent-child relationships; substantially outperforms baselines on absorption. **Key methodological insight**: Newer work with less community validation; training complexity is a confound when comparing to simpler architectures.

7. **[Korznikov et al., 2025] "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features"** (arXiv:2509.22033) — Orthogonality penalty on latents; 65% absorption reduction. **Key methodological insight**: Only training-time modification; orthogonality may be too strong for correlated features — a case where the cure may have side effects.

8. **[CE-Bench, BlackboxNLP 2025] "CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark"** — Fully LLM-free contrastive evaluation using semantically contrastive story pairs; deterministic and reproducible. **Key methodological insight**: Addresses the core reproducibility pitfall of LLM-dependent evaluation in SAEBench.

9. **[Marks et al., 2024/2025] "A Unified Theory of Sparse Dictionary Learning"** (arXiv:2512.05534) — Theoretical framework: piecewise biconvexity and spurious minima; feature anchoring improves recovery. **Key methodological insight**: Explains why simple linear encoders are hard to outperform; provides theoretical grounding for why absorption occurs.

10. **[Lieberum et al., 2024] "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2"** (arXiv:2408.05147) — Released full-layer SAEs (16k/65k/1m, 27 layers); became dominant experimental platform. **Key methodological insight**: Feature absorption observed across all layers and sizes; provides standardized pretrained SAEs essential for controlled comparisons.

11. **[Till, 2025] "Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders"** (arXiv:2508.09363) — Argued broad-domain training produces representations vulnerable to nonlinear error and feature absorption. **Key methodological insight**: Domain-specific SAEs as mitigation, but this limits generalization — a confound in cross-domain comparisons.

12. **[Song et al., 2025] "Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs"** (arXiv:2505.20254) — Emphasized feature consistency across training runs; proposed PW-MCC metric. **Key methodological insight**: Different runs learn different feature sets, hurting reproducibility; consistency is related to but distinct from absorption reliability.

### Experimental Landscape

**What has been properly tested:**
- Absorption rate on early layers (0-17) of Gemma 2 2B via causal ablation (Chanin et al.)
- Feature sensitivity on generated similar texts across multiple SAE widths (Hu et al.)
- Cross-architecture comparison on SAEBench 8 metrics (Karvonen et al.)
- Ground-truth feature recovery on synthetic data (SynthSAEBench)
- Sparsity-fidelity frontier characterization across architectures

**What is accepted without evidence:**
- That absorption is the primary cause of poor downstream interpretability (no direct causal link established)
- That deep-layer absorption follows the same patterns as early layers (ablation metric fails past layer 17)
- That feature sensitivity and absorption are correlated (never jointly measured on the same features)
- That training-free methods can reliably detect absorption (post-hoc detection methods lack ground-truth validation)
- That architectural improvements (Matryoshka, HSAE, OrtSAE) generalize across all model families and tasks

**Where methodological gaps exist:**
1. **No training-free absorption metric with ground-truth validation** — All existing metrics require either causal ablation (limited depth) or training new SAEs
2. **No joint measurement of absorption and sensitivity** — These related concepts have been studied in isolation
3. **No statistical testing framework** — Absorption rates reported as point estimates without confidence intervals or significance tests
4. **No controlled downstream impact study** — Absorption's effect on circuit discovery, model editing, and concept detection is assumed but not quantified
5. **No multi-seed reproducibility analysis** — Most studies report single-run results, masking variance across random initializations

---

## Phase 2: Initial Candidates

### Candidate A: Training-Free Absorption Detection via Encoder-Decoder Asymmetry

- **Hypothesis**: Absorbed features exhibit encoder-decoder asymmetry — the encoder fails to activate the parent feature while the decoder still contains its direction in the reconstruction. This asymmetry can be quantified without causal ablation and validated at all layers.
- **Falsification criterion**: If encoder-decoder asymmetry scores do not correlate with ground-truth absorption labels (from SynthSAEBench or Chanin et al.'s ablation-based labels on layers 0-17), the hypothesis is false.
- **Evaluation protocol**:
  - Primary: Pearson/Spearman correlation between asymmetry score and ground-truth absorption labels on SynthSAEBench-16k (where ground truth is known)
  - Secondary: Correlation with Chanin et al.'s ablation-based absorption rate on Gemma 2 2B layers 0-17
  - Statistical test: Bootstrap 95% CI for correlation coefficient; significance at p < 0.05
  - Minimum 3 random seeds for SAE training (if any training required); for pretrained SAEs, bootstrap resampling of feature samples
- **Ablation plan**:
  - Ablate the asymmetry metric components (encoder norm, decoder norm, cosine similarity) to identify which contributes most
  - Test on multiple SAE architectures (TopK, JumpReLU, ReLU) to check generalization
  - Test at multiple layers (early, middle, late) to verify depth-independence
- **Confounders**:
  - Dead features may show similar asymmetry — must filter out dead features
  - Feature frequency may correlate with both asymmetry and absorption — must control for frequency
  - SAE width may affect asymmetry independently — must test across widths
- **Pilot design**: Load Gemma Scope layer 5 SAE (16k width), compute encoder-decoder asymmetry for 100 features with known absorption status from sae-spelling dataset. Compare asymmetry distributions between absorbed and non-absorbed features. Runtime: ~10 minutes.

### Candidate B: Joint Absorption-Sensitivity Measurement and Unified Quality Score

- **Hypothesis**: Low-sensitivity features are disproportionately absorbed, and a unified score combining both metrics better predicts true feature quality (as measured by ground-truth recovery) than either metric alone.
- **Falsification criterion**: If the correlation between absorption and sensitivity is near zero (|rho| < 0.1) and the unified score does not outperform individual metrics on ground-truth feature recovery (MCC on SynthSAEBench), the hypothesis is false.
- **Evaluation protocol**:
  - Primary: Spearman correlation between absorption rate and sensitivity score on the same feature set (Gemma Scope layers 0-17, 16k width)
  - Secondary: Predictive power of unified score for ground-truth feature recovery on SynthSAEBench-16k (AUROC for distinguishing well-recovered vs poorly-recovered features)
  - Statistical test: Bootstrap 95% CI for correlation; paired t-test comparing unified score AUROC vs individual metric AUROCs
  - Minimum 3 random seeds for sensitivity text generation; report variance
- **Ablation plan**:
  - Ablate each component of the unified score (absorption-only, sensitivity-only, interaction term)
  - Test different weighting schemes for the unified score
  - Test on multiple SAE widths (16k, 65k) to check scaling behavior
- **Confounders**:
  - Feature frequency affects both metrics — must partial out frequency effects
  - LLM-generated text for sensitivity introduces non-determinism — must fix LLM version and temperature
  - Layer depth affects absorption detectability — restrict to layers 0-17 for absorption ground truth
- **Pilot design**: Select 50 features from Gemma Scope layer 5 with known absorption status; measure sensitivity on 10 LLM-generated similar texts per feature; compute correlation. Runtime: ~15 minutes.

### Candidate C: Controlled Experiment — Does Absorption Actually Degrade Downstream Circuit Completeness?

- **Hypothesis**: SAEs with higher absorption rates produce less complete sparse feature circuits (as measured by circuit faithfulness on targeted model editing tasks) than SAEs with lower absorption rates, when matched for reconstruction quality and sparsity.
- **Falsification criterion**: If circuit completeness (measured by faithfulness score on RAVEL or targeted probe perturbation) does not differ significantly between high-absorption and low-absorption SAEs matched for reconstruction and sparsity, the hypothesis is false.
- **Evaluation protocol**:
  - Primary: Compare circuit faithfulness on RAVEL benchmark between high-absorption (TopK) and low-absorption (JumpReLU) SAEs matched for L0 and reconstruction MSE on the same layer
  - Secondary: Targeted probe perturbation (TPP) score from SAEBench on the same matched pairs
  - Statistical test: Paired t-test across 5 matched SAE pairs; bootstrap 95% CI for mean difference; Cohen's d for effect size
  - Minimum 3 random seeds per architecture; minimum 5 architecture pairs
- **Ablation plan**:
  - Ablate absorption level by comparing same-architecture SAEs with different sparsity penalties (high L1 → high absorption vs low L1 → low absorption)
  - Ablate architecture effect by comparing different architectures at matched absorption levels
  - Ablate layer effect by testing at early (layer 5) and late (layer 17) layers
- **Confounders**:
  - Reconstruction quality differences may drive circuit completeness, not absorption — must match on reconstruction MSE
  - Sparsity level affects both absorption and circuit completeness — must match on L0
  - Different architectures may have different inherent circuit completeness — must control for architecture
  - Task selection bias — must use established benchmark (RAVEL) rather than cherry-picked tasks
- **Pilot design**: Compare one TopK vs one JumpReLU SAE on Gemma 2 2B layer 5, matched for L0 ≈ 100, on a single RAVEL relation. Measure circuit faithfulness difference. Runtime: ~15 minutes.

---

## Phase 3: Self-Critique

### Against Candidate A: Training-Free Absorption Detection

- **Confound attack**: What variables haven't been controlled? Dead features, feature frequency, and SAE width all affect encoder-decoder norms independently of absorption. The metric may simply detect rare features rather than absorbed ones. Search for papers on SAE encoder-decoder properties: [Marks et al. 2024] shows decoder directions are not unit-normalized and encoder/decoder are not exact transposes — this structural asymmetry is normal and must be distinguished from absorption-induced asymmetry.
- **Statistical attack**: The expected effect size (difference in asymmetry between absorbed and non-absorbed features) is unknown. If the effect is small (Cohen's d < 0.3), detecting it reliably may require thousands of features, making the pilot underpowered. A bootstrap CI on the correlation coefficient with only 100 features will be wide and potentially misleading.
- **Benchmark attack**: SynthSAEBench ground truth may not capture the full complexity of LLM absorption. The synthetic data has perfect hierarchy, while real LLM features have noisy, partial hierarchies. Validation on synthetic data may overestimate performance.
- **Ablation completeness attack**: Testing across architectures is good, but what about across models (Gemma vs GPT-2 vs Pythia)? If the metric only works on Gemma, its utility is limited. Also, the metric may detect "feature hedging" (Chanin & Garriga-Alonso 2025) rather than absorption — these are related but distinct failure modes.
- **Verdict**: MODERATE — The idea addresses a real gap (training-free detection) but the confound of dead features and frequency is serious. The lack of a clear effect size estimate makes power analysis impossible. Strengthening requires: (1) rigorous frequency matching, (2) explicit distinction from hedging, (3) cross-model validation.

### Against Candidate B: Joint Absorption-Sensitivity Measurement

- **Confound attack**: The biggest uncontrolled variable is **feature frequency**. Both absorption (Chanin et al. focus on first-letter features which are relatively frequent) and sensitivity (Hu et al. filter to features with ≥15 activating examples) are frequency-dependent. Low-frequency features may appear both "absorbed" (few examples to verify) and "insensitive" (hard to generate similar texts). Without stratified sampling by frequency, the correlation may be spurious.
- **Statistical attack**: The sensitivity score is a proportion (k/10 generated texts activate). With only 10 trials per feature, the sampling variance is high (standard error up to 0.16 for p=0.5). Correlating a noisy proportion with a binary absorption label will attenuate the true correlation. A bootstrap CI on Spearman's rho with 50-100 features may be too wide to detect a moderate correlation (rho = 0.3).
- **Benchmark attack**: Restricting absorption ground truth to layers 0-17 (where ablation works) limits generalizability. If the correlation only holds in early layers, its utility for deep-layer analysis is unclear. Also, using LLM-generated texts for sensitivity introduces non-determinism that may swamp the true signal.
- **Ablation completeness attack**: The unified score combines two metrics that may capture the same underlying phenomenon (poor feature quality) rather than complementary aspects. If absorption and sensitivity are highly correlated (rho > 0.7), the unified score adds no value. If they are uncorrelated, the unified score may dilute signal. The interaction term must be justified theoretically.
- **Verdict**: WEAK — The frequency confound is severe and difficult to fully control. The sensitivity metric's high variance with only 10 samples per feature makes correlation detection underpowered. The unified score lacks theoretical justification for its functional form. This candidate should be dropped unless a much larger sample size and rigorous frequency stratification can be achieved.

### Against Candidate C: Absorption's Impact on Downstream Circuit Completeness

- **Confound attack**: The most serious uncontrolled variable is **architecture-specific inductive bias**. JumpReLU may produce more complete circuits not because of lower absorption, but because its per-feature thresholds better preserve directional information. Matched reconstruction and sparsity do not guarantee matched internal representations. A better control would be to vary absorption within the same architecture by varying sparsity penalty.
- **Statistical attack**: The expected effect size is plausible but unquantified. If the true effect is small (faithfulness difference < 5%), detecting it with 5 matched pairs requires high precision. RAVEL itself has variance across relations — some relations may show large effects while others show none. Without a pre-registered analysis plan selecting specific relations, this risks cherry-picking.
- **Benchmark attack**: RAVEL measures relational knowledge circuits, which may not be the tasks most affected by absorption. Absorption primarily affects hierarchical features (general → specific), while RAVEL focuses on factual relations. A more targeted benchmark would measure hierarchical concept completion (e.g., "starts with S" → "short").
- **Ablation completeness attack**: The ablation of "same architecture, different sparsity" is informative but confounds absorption with other sparsity-related effects (dead features, feature splitting). The ablation of "different architectures, matched absorption" is confounded by architecture. Neither ablation isolates absorption cleanly.
- **Verdict**: MODERATE — The core question (does absorption matter downstream?) is important and underexplored. However, cleanly isolating absorption from confounds is extremely difficult. Strengthening requires: (1) within-architecture sparsity variation as primary comparison, (2) a benchmark targeting hierarchical concepts, (3) pre-registered analysis plan with specific relations/tasks, (4) effect size estimation from pilot data before full study.

---

## Phase 4: Refinement

### Dropped
- **Candidate B (Joint Absorption-Sensitivity)** is dropped due to unfixable frequency confound and underpowered sensitivity metric. The 10-sample sensitivity score has too much variance, and frequency stratification would require an order of magnitude more features than feasible in a pilot.

### Strengthened

**Candidate A (Training-Free Detection) — Strengthened to Front-Runner:**
- **Added control**: Explicit frequency matching using stratified sampling — compare absorbed and non-absorbed features only within frequency quartiles
- **Added control**: Filter out dead features (zero activation across 2M tokens) before analysis
- **Added control**: Test on both synthetic (SynthSAEBench, ground truth known) and real (Gemma Scope, ground truth from ablation on layers 0-17) data
- **Tightened falsification criterion**: Correlation |rho| < 0.3 with ground-truth absorption labels on SynthSAEBench kills the hypothesis
- **Additional benchmark**: Validate on GPT-2 SAEs (different model family) to check cross-model generalization
- **Analysis plan**: Report receiver operating characteristic (ROC) curve for asymmetry score as absorption classifier; report AUC with bootstrap 95% CI
- **Pilot redesign**: Use 500 features from Gemma Scope layer 5 with known absorption status; compute asymmetry; measure AUC. Runtime: ~15 minutes.

**Candidate C (Downstream Impact) — Strengthened to Secondary:**
- **Primary comparison changed**: Within-architecture (TopK) varying sparsity penalty (high L1 vs low L1) to isolate absorption from architecture effects
- **Benchmark changed**: Use hierarchical concept completion task from sae-spelling dataset (general letter → specific word) rather than RAVEL, as this directly targets the domain where absorption occurs
- **Pre-registration**: Define exact task set (all first-letter relations A-Z) and exact metric (mean accuracy difference) before seeing results
- **Effect size estimation**: Pilot on 2 sparsity levels × 3 letters to estimate Cohen's d; proceed to full study only if d > 0.5
- **Statistical test**: Paired t-test across 26 letters; bootstrap 95% CI for mean difference; report Cohen's d
- **Control added**: Match on reconstruction MSE and L0 using hyperparameter search; reject pairs with >5% difference in either metric

### Selected Front-Runner
**Candidate A (Training-Free Absorption Detection via Encoder-Decoder Asymmetry)** is selected because:
1. It directly addresses Gap 4 (training-free analysis of pretrained SAEs) from the literature survey
2. It aligns with the project's training-free constraint
3. The experimental evidence would be most convincing if successful — a training-free metric validated against ground truth would be immediately useful to the community
4. The confounds (frequency, dead features) are identifiable and controllable
5. The pilot can give clear signal in <15 minutes

---

## Phase 5: Final Proposal

### Title
"Training-Free Detection of Feature Absorption in Sparse Autoencoders via Encoder-Decoder Asymmetry"

### Hypothesis
Absorbed SAE features exhibit a quantifiable encoder-decoder asymmetry — specifically, a mismatch between the encoder's failure to activate and the decoder's retention of the feature direction — that correlates with ground-truth absorption labels (Pearson |rho| > 0.5) and can be detected without causal ablation at any layer depth.

### Falsification Criterion
If the encoder-decoder asymmetry score achieves an AUC < 0.65 for classifying absorbed vs non-absorbed features on SynthSAEBench-16k ground truth, or if the Pearson correlation with Chanin et al.'s ablation-based absorption labels on Gemma 2 2B layers 0-17 is |rho| < 0.3 with 95% CI excluding 0.5, the hypothesis is falsified.

### Method
For each SAE feature, compute an asymmetry score based on the divergence between encoder and decoder behavior. The score combines three components:
1. **Encoder activation deficit**: The ratio of expected activation (from decoder direction projected onto inputs) to actual activation
2. **Directional mismatch**: Cosine similarity between encoder weight and decoder weight (expected to be lower for absorbed features)
3. **Norm imbalance**: Ratio of encoder norm to decoder norm (absorbed features may have suppressed encoder norms)

The final score is a weighted combination, with weights determined via logistic regression on a held-out validation set from SynthSAEBench.

### Evaluation Protocol

**Primary Benchmarks (established public benchmarks):**
- **SynthSAEBench-16k**: Ground-truth feature labels for hierarchical absorption; used for metric training/validation and primary AUC evaluation
- **Gemma Scope SAEs (Gemma 2 2B)**: Layers 0-17 with Chanin et al.'s ablation-based absorption labels; used for real-world correlation validation
- **GPT-2 SAEs (SAELens)**: Cross-model generalization test

**Metrics with Statistical Test Plan:**
- Primary metric: AUC-ROC for absorption classification (absorbed vs non-absorbed)
- Secondary metric: Pearson correlation between asymmetry score and ground-truth absorption rate
- Statistical tests:
  - Bootstrap 95% confidence intervals for AUC and correlation (10,000 resamples)
  - Permutation test for significance (p < 0.05, 10,000 permutations)
  - Cross-validation: 5-fold on SynthSAEBench; report mean ± std AUC
- Number of random seeds: 3 seeds for any trained components (logistic regression weights); for pretrained SAEs, bootstrap resampling of feature samples provides variance estimate

**Ablation Schedule:**
| Ablation | What It Tests | Expected Outcome |
|----------|---------------|------------------|
| Remove directional mismatch component | Whether cosine similarity adds signal beyond norms | AUC drops by < 0.03 if redundant, > 0.05 if critical |
| Remove norm imbalance component | Whether norm ratio adds signal | Similar interpretation |
| Use only encoder deficit | Whether single-component metric suffices | AUC lower than full combination |
| Test on dead features included vs excluded | Whether metric confounds with feature death | AUC drops significantly if dead features included |
| Test by frequency quartile | Whether metric works across frequency ranges | AUC stable across quartiles |

**Control Experiments:**
1. **Frequency-matched control**: Compare absorbed and non-absorbed features only within the same frequency quartile to rule out frequency confound
2. **Architecture control**: Test on TopK, JumpReLU, and ReLU SAEs to verify metric generalizes across architectures
3. **Depth control**: Test on layers 5, 12, and 20 (where layer 20 ground truth is unavailable, use synthetic proxy) to verify depth-independence
4. **Random baseline**: Compare to asymmetry scores computed on random feature permutations to establish chance-level performance

### Pilot Design
1. Load Gemma Scope layer 5 SAE (16k width, TopK) via SAELens
2. Load sae-spelling dataset first-letter labels for 500 features
3. Compute encoder-decoder asymmetry score for each feature
4. Compare score distributions between absorbed (n ≈ 50-100) and non-absorbed (n ≈ 400-450) features
5. Compute AUC and Pearson correlation
6. Runtime target: <15 minutes on single GPU
7. Decision rule: If AUC > 0.65 on pilot, proceed to full study; else, diagnose and iterate metric design

### Resource Estimate
- **Pilot**: 1 GPU × 15 minutes = 0.25 GPU-hours
- **Full study (SynthSAEBench validation)**: 1 GPU × 30 minutes = 0.5 GPU-hours
- **Full study (Gemma Scope layers 0-17)**: 1 GPU × 2 hours = 2 GPU-hours
- **Full study (GPT-2 cross-model)**: 1 GPU × 1 hour = 1 GPU-hour
- **Total**: ~4 GPU-hours, well within the ≤1 hour per task constraint when split appropriately
- **Model sizes**: GPT-2 small (124M), Gemma 2 2B — both small to moderate

### Risk Assessment

| Threat | Likelihood | Mitigation |
|--------|-----------|------------|
| Asymmetry confounded with dead features | High | Explicit dead feature filtering; report results with/without dead features |
| Asymmetry confounded with feature frequency | High | Frequency stratification; partial correlation controlling for frequency |
| Metric fails on deep layers (>17) where ablation ground truth unavailable | Medium | Use SynthSAEBench for deep-layer validation; acknowledge limitation |
| Low effect size (AUC 0.55-0.65) — real but weak signal | Medium | Pre-specify AUC > 0.65 as success threshold; report effect size honestly |
| Metric overfits to SynthSAEBench synthetic hierarchy | Medium | Cross-validate on real LLM SAEs; use separate train/validation splits |
| LLM-dependent sensitivity component introduces non-determinism | Low (not used) | This proposal avoids LLM-dependent evaluation entirely |

### Novelty Claim
This work would answer the following empirical question for the first time: **Can feature absorption be detected in pretrained SAEs without causal ablation?** Current methods require either (a) ablation experiments limited to early layers, or (b) training new SAEs with architectural modifications. A validated training-free metric would enable:
- Large-scale screening of existing pretrained SAEs (Gemma Scope, GPT-2, Pythia) for absorption
- Community adoption without requiring compute for ablation studies
- Layer-depth analysis beyond layer 17 where ablation fails
- Integration into automated SAE quality assessment pipelines

The experimental contribution is methodological: establishing a new class of training-free absorption metrics and validating it against multiple ground-truth sources with rigorous statistical controls.
