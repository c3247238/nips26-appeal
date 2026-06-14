# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al., 2024/2025. "A is for Absorption." arXiv:2409.14507 (NeurIPS 2025)** — First systematic definition of absorption rate metric; ablation-based detection protocol; validated on hundreds of LLM SAEs. **Key insight**: absorption requires causal ablation to definitively detect; proxy metrics fail.

2. **Karvonen et al., 2025. "SAEBench." arXiv:2503.09532 (ICML 2025)** — Eight-metric benchmark including absorption, sparse probing, RAVEL, CE loss; reveals proxy metrics (reconstruction, L0) do not predict interpretability performance. **Key insight**: standardized evaluation protocol across 200+ SAEs.

3. **Hu et al., 2025. "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717** — Feature sensitivity as evaluation dimension; sensitivity declines with SAE width; proposes sensitivity as absorption indicator. **Key insight**: low-sensitivity features may indicate absorption.

4. **Chanin et al., 2026. "SynthSAEBench." arXiv:2602.14687** — Ground-truth synthetic benchmark with 16k features, known hierarchy, correlation, superposition. **Key insight**: MP-SAEs exploit superposition noise without learning true features; no architecture achieves perfect performance.

5. **Bussmann et al., 2025. "Matryoshka SAE." arXiv:2503.17547** — Nested dictionaries reduce absorption 0.49→0.05; establishes hierarchy-absorption connection. **Key insight**: explicit parent-child modeling is effective intervention.

6. **Korznikov et al., 2025. "OrtSAE." arXiv:2509.22033** — Orthogonality penalty achieves 65% absorption reduction. **Key insight**: architectural modifications can mitigate absorption.

7. **Lieberum et al., 2024. "Gemma Scope." arXiv:2408.05147** — Full-layer Gemma 2 SAEs (16k/65k/1m, 27 layers); dominant experimental platform; absorption observed across all layers.

8. **Luo et al., 2026. "HSAE Feature Forest." arXiv:2602.11881** — Explicit parent-child constraints substantially reduce absorption; structural constraint loss approach.

### Experimental Landscape

**What has been properly tested:**
- Ablation-based absorption detection on layers 0-17 (Gemma 2B) via Chanin et al.
- Cross-architecture absorption comparisons (JumpReLU, TopK, ReLU) on SAEBench
- Hierarchical SAE training effects (Matryoshka, HSAE) on absorption reduction
- Feature sensitivity correlation with width across 7 SAE variants

**What is accepted without evidence:**
- That absorption rate increases monotonically with layer depth (unverified beyond layer 17)
- That encoder drives absorption while decoder faithfully decodes absorbed directions (single iter finding, not replicated)
- That training-free proxies (sensitivity, encoder-decoder asymmetry) reliably predict absorption
- That absorbed features have degraded downstream utility (circuit completeness, steering success)

**Methodological gaps:**
- No systematic controlled comparison across all major architectures (ReLU, TopK, JumpReLU, Matryoshka, HSAE, OrtSAE) on identical layers/data
- No validation of training-free absorption proxies against ground truth (SynthSAEBench)
- No quantification of absorption's downstream impact on interpretability tasks
- No layer-by-layer absorption profile with consistent methodology

---

## Phase 2: Initial Candidates

### Candidate A: Training-Free Proxy Validation Study

- **Core hypothesis**: Encoder-decoder asymmetry (measured by reconstruction fidelity divergence) predicts absorption with AUC > 0.7 against ablation-based ground truth.

- **Falsification criterion**: If encoder-decoder asymmetry shows < 0.6 AUC correlation with true absorption rate, the proxy fails and ablation remains necessary.

- **Evaluation protocol**:
  - Primary benchmark: SynthSAEBench (ground truth absorption labels)
  - Metrics: AUC-ROC, Pearson correlation, precision@10
  - Baseline comparison: Feature sensitivity (Hu et al.) as existing proxy
  - Statistical test: Bootstrap 95% CI, 1000 resamples

- **Ablation plan**:
  - Ablate: Vary asymmetry metric thresholds to find optimal cutpoint
  - Test: Does asymmetry predict absorption across different architectures (ReLU, TopK, JumpReLU)?
  - Ablate: Vary dataset composition (correlated vs. independent features)

- **Confounders identified**:
  - Architecture-specific encoding artifacts may create asymmetry independent of absorption
  - Dead features create asymmetric encoder-decoder behavior (not absorption)
  - Layer-specific information integration patterns may affect asymmetry signal

- **Pilot design**: 15-min test on SynthSAEBench subset (500 features) to estimate AUC before full run.

### Candidate B: Cross-Architecture Controlled Absorption Comparison

- **Core hypothesis**: Absorption rate differences between SAE architectures (ReLU, TopK, JumpReLU) are explained by differences in sparsity enforcement timing (online vs. post-hoc) and encoder direction orthogonality, not by architectural choice per se.

- **Falsification criterion**: If architectures with similar sparsity enforcement mechanisms show < 10% absorption rate difference, the hypothesis is supported; if architectures with different mechanisms show large differences, the specific architectural choice matters.

- **Evaluation protocol**:
  - Primary benchmark: SAEBench absorption metric on Gemma 2B layers 0-17
  - Metrics: Mean absorption rate, absorption rate variance across features
  - Control variables: identical training data (OpenWebText), identical layer, identical width (16k)
  - Statistical test: Paired t-test across features, ANOVA across architectures

- **Ablation plan**:
  - Ablate width: 16k vs 65k vs 1M to test capacity effects
  - Ablate L0 target: k=32 vs k=64 vs k=128 to test sparsity pressure
  - Ablate training tokens: 10M vs 100M vs 500M to test data scaling

- **Confounders identified**:
  - Different architectures trained by different groups (codex vs. SAELens vs. Anthropic)
  - Training hyperparameter differences (lr, warmup, batch size)
  - Random seed variation across training runs

- **Pilot design**: Compare 2 architectures (TopK vs ReLU) on single layer (layer 12) with matched training configs; 30-min runtime.

### Candidate C: Downstream Impact Quantification — Circuit Completeness

- **Core hypothesis**: SAE features identified as absorbed (via ablation) produce degraded circuit analysis results compared to non-absorbed features, measurable by steering success rate differential.

- **Falsification criterion**: If absorbed and non-absorbed features show < 10% difference in steering success rate on matched test cases, absorption does not have practically significant downstream impact.

- **Evaluation protocol**:
  - Primary benchmark: Custom circuit completeness test using steering
  - Target: 50 absorbed features + 50 matched non-absorbed features from Gemma Scope
  - Metrics: Steering success rate (% of cases where feature direction produces expected behavioral change)
  - Statistical test: McNemar's test for paired proportions

- **Ablation plan**:
  - Ablate steering magnitude: 1x vs 5x vs 10x baseline
  - Ablate context: single token vs. sentence vs. paragraph
  - Ablate feature selection: random absorbed vs. high-risk absorbed (maximum variance)

- **Confounders identified**:
  - Feature activation frequency: high-frequency features may be easier to steer regardless of absorption
  - Feature semantic clarity: more interpretable features may steer more reliably
  - Model layer effects: some layers may be more amenable to steering

- **Pilot design**: Test 10 absorbed vs 10 non-absorbed features for basic steering signal; 20-min runtime.

---

## Phase 3: Self-Critique & Adversarial Testing

### Against Candidate A: Training-Free Proxy Validation

- **Confound attack**: Search for papers on encoder-decoder behavior differences in autoencoders — autoencoders generally show encoder-decoder asymmetry even without absorption. The asymmetry signal may be dominated by architectural defaults, not absorption. *Rebuttal*: Restrict to SAE-specific asymmetry by computing relative to matched random SAE baseline.

- **Statistical attack**: With 16k features in SynthSAEBench, even small correlations may achieve statistical significance. Effect size (AUC) matters more than p-value. *Rebuttal*: Focus on AUC > 0.7 threshold as practical usefulness criterion.

- **Benchmark attack**: SynthSAEBench uses synthetic data; absorption patterns may differ in real LLMs. The proxy validated on synthetic data may not transfer. *Rebuttal*: Cross-validate on real SAE ablation data (layers 0-17) where available.

- **Ablation completeness attack**: Testing one proxy (asymmetry) vs. one baseline (sensitivity) is incomplete. Many other proxies exist. *Rebuttal*: Expand to include 3-5 candidate proxies in validation study.

- **Verdict**: MODERATE — The study directly addresses the field's need for training-free absorption detection. Main risk is synthetic-to-real transfer. Design should include cross-validation.

### Against Candidate B: Cross-Architecture Comparison

- **Confound attack**: Different architectures are trained by different groups with different hyperparameters. Without matched training, we cannot isolate architecture effects from training effects. *Rebuttal*: Use SAELens pretrained SAEs which share training pipeline, or retrain with matched configs.

- **Statistical attack**: Comparing mean absorption rates across 3+ architectures with different width configurations creates multiple comparison issues. *Rebuttal*: Pre-registered contrasts with Bonferroni correction; focus on pairwise architecture comparisons.

- **Benchmark attack**: SAEBench absorption metric for JumpReLU may not be comparable to TopK due to different threshold mechanisms. The metric itself may be architecture-biased. *Rebuttal*: Use multiple absorption metrics (Chanin ablation, SAEBench mean absorption fraction) to check consistency.

- **Ablation completeness attack**: The "sparsity enforcement timing" hypothesis is specific but may not capture all relevant architectural differences. *Rebuttal*: Measure actual sparsity enforcement timing in each architecture explicitly (when does L0 drop to target?).

- **Verdict**: MODERATE — The study would provide valuable systematic comparison, but confounding between architecture and training setup is the main threat. Requires matched training or careful selection of same-source SAEs.

### Against Candidate C: Downstream Impact Quantification

- **Confound attack**: Selecting "matched" non-absorbed features is tricky. If absorbed features are systematically less interpretable to begin with, steering difficulty may reflect interpretability, not absorption. *Rebuttal*: Match on activation frequency and feature interpretability score (via Neuronpedia ratings) before comparing absorption status.

- **Statistical attack**: Steering success is a binary outcome with high variance; 50 features may be underpowered to detect < 10% effects. *Rebuttal*: Power analysis: with baseline steering rate of 60%, 50 features gives 80% power to detect 25% relative improvement (too coarse). Need 200+ features for 10% detection.

- **Benchmark attack**: Steering is one downstream task; circuit analysis and model editing may show different sensitivity to absorption. *Rebuttal*: This is a pilot for one downstream task; if positive, extend to circuit completeness and unlearning tasks.

- **Ablation completeness attack**: Steering magnitude is arbitrary; may not reflect natural feature strength in the model. *Rebuttal*: Calibrate steering magnitude to feature activation strength on natural inputs.

- **Verdict**: MODERATE — The research question is important (does absorption matter downstream?) but the measurement is challenging. Steering success may not capture the full picture of circuit analysis utility. Needs careful feature matching and sufficient sample size.

---

## Phase 4: Refinement

### Dropped Candidates
- **None dropped outright** — all three candidates address genuine gaps with falsifiable hypotheses.

### Strengthened Candidates

**Candidate A (Proxy Validation)** strengthened by:
- Adding feature sensitivity (Hu et al.) as a comparison baseline
- Adding multiple candidate proxies: encoder-decoder cosine divergence, reconstruction error per feature, activation-entropy
- Including cross-validation on both SynthSAEBench and real SAE data
- Tightening success criterion to AUC > 0.75 (practical usefulness)

**Candidate B (Cross-Architecture)** strengthened by:
- Narrowing to SAELens-pretrained SAEs only (matched training pipeline)
- Adding width ablation (16k vs 65k) as within-architecture comparison
- Adding orthogonal analysis: measure encoder direction orthogonality as mediator variable

**Candidate C (Downstream Impact)** strengthened by:
- Adding feature matching protocol: match on activation frequency + Neuronpedia interpretability score
- Power analysis to determine minimum feature sample size
- Adding qualitative circuit analysis alongside steering (check if absorbed features produce interpretable circuits)

### Additional Controls Needed
- Dead feature exclusion (dead features inflate absorption statistics)
- Random baseline (measure absorption in randomly initialized SAE to establish floor)
- Layer-wise controls (information integration patterns differ by layer)

### Selected Front-Runner
**Candidate A: Training-Free Proxy Validation Study**

Reason: This directly addresses the most pressing practical gap in SAE absorption research — the field needs a reliable, computationally cheap proxy for the ablation-based gold standard. The experiment is achievable in 1 hour on a single GPU, uses existing benchmarks (SynthSAEBench, SAEBench), and produces a binary classifier (proxy vs. ground truth) that can be immediately validated and deployed. A successful outcome (AUC > 0.75) would be a significant practical contribution. Even a negative outcome (proxy fails) tells us something important about what absorption actually measures.

---

## Phase 5: Final Proposal

### Title

**Training-Free Absorption Detection: Validating Proxy Metrics Against Ground Truth**

### Hypothesis

Encoder-decoder asymmetry — specifically the ratio of reconstruction fidelity when encoding vs. decoding the same feature-specific activations — predicts absorption status with AUC > 0.75 against ablation-based ground truth on SynthSAEBench features. If validated, this provides a computationally cheap, training-free absorption detection method.

### Falsification Criterion

If no combination of asymmetry metrics (encoder-decoder cosine divergence, reconstruction error ratio, activation-entropy divergence) achieves AUC > 0.70 against ground truth absorption, the training-free proxy approach fails and ablation-based detection remains necessary for reliable absorption measurement.

### Method

**Proxy Metrics to Test:**

1. **Encoder-Decoder Cosine Divergence (ECD)**: For each feature f, compute the cosine similarity between the feature's encoder direction and decoder direction. Absorbed features should show higher divergence (encoder and decoder "disagree" on what the feature represents).

2. **Reconstruction Error Ratio (RER)**: For each feature f, measure reconstruction error when (a) encoding then decoding the feature's activating inputs vs. (b) directly decoding from the feature's latent. Absorbed features should show larger error in (a).

3. **Activation-Entropy Divergence (AED)**: For each feature f, compute the entropy of activation patterns. Absorbed features may show different entropy profiles.

4. **Feature Sensitivity Score (FSS)** (baseline comparison from Hu et al.): Fraction of LLM-generated similar texts that activate the feature.

**Evaluation Protocol:**

- **Primary benchmark**: SynthSAEBench-16k (ground truth absorption labels for 16k features)
- **Secondary validation**: Real SAE ablation data (Gemma 2B layers 0-17, via SAEBench)
- **Metrics**:
  - AUC-ROC against binary absorption label
  - Pearson correlation with continuous absorption rate
  - Precision@10 (top 10 predicted absorbed features, how many are truly absorbed)
- **Statistical test**: Bootstrap 95% CI (1000 resamples); DeLong test for AUC comparison

### Experimental Plan

**Tier 1: Synthetic Benchmark Validation (45 min)**

1. Load SynthSAEBench-16k with ground truth features, hierarchy, and absorption labels
2. For each of 16k features, compute all 4 proxy metrics
3. Evaluate AUC-ROC for each metric against ground truth absorption
4. Test metric combinations (logistic regression ensemble) for improved AUC
5. Report which metric or combination best predicts absorption

**Tier 2: Real SAE Cross-Validation (30 min)**

1. Load Gemma Scope SAE (16k width, layer 12) via SAELens
2. For features with ablation-based absorption rates (layers 0-17), compute proxy metrics
3. Evaluate same metrics against real absorption rates
4. Compare synthetic vs. real performance to assess transfer

**Tier 3: Threshold Calibration (if positive) (15 min)**

1. Use SynthSAEBench validation set to calibrate decision threshold
2. Test threshold transfer to real SAEs
3. Report operating point (sensitivity/specificity tradeoff)

### Ablation Schedule

| Ablation | What It Tests | Expected Outcome |
|----------|---------------|-----------------|
| Metric removed: ECD only | Does ECD alone predict absorption? | AUC baseline |
| Metric removed: FSS only | How much does sensitivity add over ECD? | ECD vs. ECD+FSS AUC |
| Architecture: ReLU vs TopK | Does proxy work across architectures? | Cross-arch robustness |
| Width: 16k vs 65k | Does proxy work at different widths? | Scale robustness |
| Layer: shallow vs deep | Does proxy work on attention layers (>17)? | Layer range |

### Control Experiments

1. **Random SAE baseline**: Compute proxy metrics on randomly initialized (untrained) SAE to establish floor values
2. **Dead feature exclusion**: Remove features with < 0.1% activation frequency to control for dead feature artifacts
3. **Matched feature comparison**: Compare absorbed vs. non-absorbed features matched on activation frequency to control for frequency confounds

### Pilot Design

**15-min pilot on 500-feature subset:**

```
Data: SynthSAEBench first 500 features
Metrics: ECD, RER, FSS
Evaluation: AUC-ROC, precision@10
Success criterion: Any metric shows AUC > 0.65 on subset (indicating signal)
```

Pilot output: Binary go/no-go for full 16k run, and preliminary metric ranking.

### Resource Estimate

- **Model**: Gemma-2-2B with Gemma Scope SAE (16k width)
- **Data**: SynthSAEBench-16k (loaded from HuggingFace)
- **GPU memory**: < 4GB (encoding/decoding operations are memory-efficient)
- **Time**:
  - Pilot (500 features): 15 min
  - Full SynthSAEBench (16k features): 30 min
  - Real SAE validation (5000 features): 30 min
- **Total**: ~75 min (within 1-hour target with margin)

### Risk Assessment

1. **Synthetic-to-real gap**: Proxy validated on synthetic data may not transfer to real LLMs. *Mitigation*: Include real SAE data in Tier 2; report transfer performance explicitly.

2. **Metric collinearity**: ECD, RER, and AED may measure the same underlying phenomenon. *Mitigation*: Check metric correlations; if > 0.9, select best-performing one.

3. **Threshold instability**: Calibrated threshold on synthetic data may not apply to real SAEs. *Mitigation*: Report confidence intervals on threshold; test robustness.

4. **Negative result**: All proxies fail to predict absorption. *Mitigation*: This is a valid scientific result; it confirms ablation remains necessary and motivates architecture improvements.

### Novelty Claim

This is the **first systematic validation study for training-free absorption detection proxies**. While Hu et al. (2025) proposed sensitivity as a proxy and Marks et al. (2025) hinted at encoder-decoder divergence, no prior work:

1. **Systematically compares multiple proxy candidates** against ground truth
2. **Validates on both synthetic and real SAE data** to assess transfer
3. **Reports explicit performance thresholds** (AUC > 0.75) for practical usefulness
4. **Ablates metric contributions** to determine which aspects of absorption proxy metrics capture

If successful, the result is immediately useful: practitioners can screen pretrained SAEs for absorption risk without expensive ablation experiments. If unsuccessful, the result clarifies that architectural solutions (Matryoshka, HSAE) remain the only reliable absorption mitigation.