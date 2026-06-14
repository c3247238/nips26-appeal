# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024)** - A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders (arXiv:2409.14507, NeurIPS 2025)
   - Key evaluation insight: First systematic absorption detection via ablation studies; establishes hierarchical co-occurrence as root cause; proves absorption is widespread across LLM SAEs
   - Limitation: Metric limited to early layers (0-17) due to ablation reliability; only tests first-letter spelling task

2. **Karvonen et al. (2025)** - SAEBench: A Comprehensive Benchmark for Sparse Autoencoders (arXiv:2503.09532, ICML 2025)
   - Key evaluation insight: 8-metric evaluation suite including probe projection contribution for absorption detection across ALL layers (solves the layer limitation of ablation-based metrics)
   - Limitation: Absorption metric computationally expensive (~26 min per SAE)

3. **Costa et al. (2025)** - From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit (arXiv:2506.03093, NeurIPS 2025)
   - Key evaluation insight: MP-SAE uses residual-guided greedy selection to recover hierarchical structure missed by conventional SAEs; conditional orthogonality property
   - Limitation: Greedy algorithm with no global optimality guarantee; limited LLM-scale validation

4. **Cui et al. (2026)** - On the Limits of Sparse Autoencoders (arXiv:2506.15963, ICLR 2026)
   - Key evaluation insight: First closed-form theoretical analysis proving standard SAEs generally fail to recover ground-truth features due to intrinsic representational interference
   - Limitation: Theoretical focus; limited direct guidance for absorption quantification

5. **Gao et al. (2024)** - Scaling and Evaluating Sparse Autoencoders (arXiv:2406.04093)
   - Key evaluation insight: k-sparse autoencoders for direct sparsity control; establishes scaling laws for SAE training
   - Limitation: Does not address absorption

6. **Basu et al. (2026)** - Interpretability without Actionability: Mechanistic Methods Cannot Correct LLM Errors (arXiv:2603.18353)
   - Key evaluation insight: Critical negative result showing 98.2% AUROC but 0% output change via SAE steering; raises fundamental questions about actionability
   - Limitation: Clinical domain; raises questions about utility of absorption research

7. **Lieberum et al. (2024)** - GemmaScope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 (arXiv:2408.05147)
   - Key evaluation insight: Comprehensive open-source SAE suite (2B/9B, every layer, 16k/65k/131k widths); de facto standard for evaluation
   - Limitation: SAEs exhibit absorption (documented by Chanin et al.)

8. **Bussmann et al. (2025)** - Learning Multi-Level Features with Matryoshka Sparse Autoencoders (arXiv:2503.17547)
   - Key evaluation insight: Nested dictionaries reduce absorption; superior on sparse probing and concept erasure
   - Limitation: Minor reconstruction tradeoff; only evaluated on Gemma-2-2B

### Experimental Landscape

**What has been properly tested:**
- Absorption detection via ablation (Chanin et al.) - limited to early layers
- Absorption rates across hundreds of LLM SAEs - confirmed widespread phenomenon
- Root cause: hierarchical feature co-occurrence under sparsity optimization
- Theoretical limits: Cui et al. prove full disentanglement is mathematically impossible

**What is accepted without evidence:**
- That absorption is the primary failure mode degrading SAE interpretability
- That reducing absorption improves downstream tasks (circuit discovery, steering)
- That cross-layer absorption patterns follow predictable trajectories
- That CV (coefficient of variation) predicts steering effectiveness

**Methodological gaps:**
1. **Systematic cross-layer quantification**: No comprehensive study across different model families, layer depths, and SAE configurations using probe projection metrics
2. **Training-free absorption detection**: No methods to detect absorption in existing pretrained SAEs without retraining
3. **CV-actionability connection**: No systematic validation that CV predicts which absorbed features remain steerable
4. **Cross-architecture validation**: Most findings from GPT-2; generalizability to Gemma-2, Llama unexplored
5. **Phase transition phenomenology**: Finite-size scaling (ν=3) discovered but chi_ratio=1.88 below sharp transition threshold

---

## Phase 2: Initial Candidates

### Candidate A: Cross-Layer Absorption Mapping at Critical Sparsity

- **Core hypothesis**: Layer-dependent absorption heterogeneity is observable at the true critical sparsity λ_c ≈ 5e-5, not at λ=0.001 where all layers saturate uniformly at absorption_rate=1.0.
- **Falsification criterion**: If all layers show identical (saturated) absorption at λ_c=5e-5, this hypothesis is DISPROVEN. Cross-layer variation would only be observable at sparsity values between 1e-5 and 1e-4.
- **Evaluation protocol**:
  - Primary benchmark: SAEBench probe projection metric (works across all layers)
  - Models: GPT-2 layer 6 (16k latents), Gemma-2-2B layers 0,3,6,9,11
  - Metrics: Per-layer absorption_rate at λ ∈ {1e-5, 5e-5, 1e-4}
  - Statistical test: ANOVA across layers with post-hoc pairwise comparison
- **Ablation plan**:
  - Ablate λ sweep at each layer → measure absorption curve shape
  - Compare transition sharpness (chi_ratio) across layers
  - Identify layer(s) with anomalous absorption behavior
- **Confounders identified**:
  - SAE architecture (JumpReLU vs TopK) may confounded with layer effects
  - Dictionary size varies by layer, creating confound for finite-size scaling
  - Feature frequency distribution shifts across layers (semantic → syntactic)
- **Pilot design**: 5 layers × 3 sparsity values × 100 samples = 1500 total runs, ~30 min on single GPU

---

### Candidate B: CV-Based Steering Predictor Validation

- **Core hypothesis**: Coefficient of variation (CV) positively predicts steering effectiveness: absorbed features with high CV show 2x larger steering effects than absorbed features with low CV (pilot observation: 0.153 vs 0.075).
- **Falsification criterion**: If no significant difference in steering effect between high-CV and low-CV absorbed features (p > 0.05), the CV-actionability hypothesis is DISPROVEN.
- **Evaluation protocol**:
  - Primary benchmark: Custom steering effectiveness test on GPT-2 layer 6 SAE
  - Selection: 30 high-CV absorbed features (CV > 5.0) vs 30 low-CV absorbed features (CV < 0.1)
  - Metrics: Mean logit change at semantically appropriate tokens at steering strengths ±3, ±5, ±7
  - Statistical test: Two-sample t-test (one-sided, α = 0.01)
- **Ablation plan**:
  - Validate CV-actionability connection across multiple steering strengths
  - Test whether high-CV absorbed features remain steerable even when absorption is severe
  - Compare against non-absorbed baseline features
- **Confounders identified**:
  - Feature frequency confounds with CV (high-frequency features may have lower CV naturally)
  - Decoder magnitude correlates with both CV and steering effect
  - Semantic category of feature may affect both CV and steering potential
- **Pilot design**: 30 high-CV + 30 low-CV features × 3 steering strengths = 180 steering tests, ~20 min on single GPU

---

### Candidate C: Controlled Experiment: Absorption vs. Feature Frequency

- **Core hypothesis**: The observed correlation between absorption and co-occurrence frequency is explained by the information bottleneck effect: high co-occurrence forces the encoder to route parent information through dominant child channels.
- **Falsification criterion**: If we control for feature activation magnitude and still observe absorption for high-co-occurrence features, the information bottleneck hypothesis is supported. If absorption disappears when controlling for magnitude, frequency is not the causal factor.
- **Evaluation protocol**:
  - Primary benchmark: Feature frequency controlled experiment on GPT-2 layer 6 SAE
  - Selection: 50 features binned by activation magnitude (low/medium/high), within each bin compare high-co-occurrence vs low-co-occurrence
  - Metrics: Absorption rate as function of co-occurrence, controlling for magnitude
  - Statistical test: ANCOVA with activation magnitude as covariate
- **Ablation plan**:
  - Vary co-occurrence frequency while holding activation magnitude constant
  - Measure decoder cosine similarity as mediator
  - Test whether L1 sparsity penalty mediates the frequency-absorption relationship
- **Confounders identified**:
  - Feature semantic category correlates with both frequency and absorption
  - Dictionary position (early vs late latents) may have different sparsity properties
  - Feature age (when during training feature was learned) may confound frequency effects
- **Pilot design**: 50 features × 2 co-occurrence levels × 3 magnitude bins = 300 runs, ~15 min on single GPU

---

## Phase 3: Self-Critique

### Against Candidate A (Cross-Layer Mapping)

- **Confound attack**: SAE dictionary size varies by layer in GemmaScope (some layers have 16k, others 65k). This creates a confound: any observed "layer effect" could be a dictionary size effect. The probe projection metric may also have layer-dependent reliability that mimics absorption variation.
- **Statistical attack**: Testing 5 layers with 3 sparsity values = 15 comparisons. Without correction, family-wise error rate inflates to ~57% (1 - 0.95^15). Need Benjamini-Hochberg correction or planned contrasts.
- **Benchmark attack**: SAEBench probe projection metric was designed for absorption detection, not cross-layer comparison. If the metric has floor/ceiling effects at certain layers, cross-layer comparison may be artifact.
- **Ablation completeness attack**: Even if we find cross-layer variation at λ_c=5e-5, we don't know what causes it. Layer depth is a proxy for many things (semantic encoding, attention patterns, recursion depth). The "explanation" would be post-hoc storytelling.
- **Verdict**: MODERATE - The experiment is feasible and addresses a real gap, but findings may be confounded and explanation may be post-hoc.

### Against Candidate B (CV-Based Steering)

- **Confound attack**: Pilot showed 2x difference (0.153 vs 0.075) but didn't control for decoder magnitude. High-CV features may simply have larger decoders, producing larger steering effects regardless of absorption. Need to control for decoder weight norm.
- **Statistical attack**: 30 vs 30 samples gives ~80% power to detect d=0.73 effect (Cohen's d). The observed ratio of 2.03 may be inflated by small-sample selection. Need replication with independent sample.
- **Benchmark attack**: Steering effectiveness measured by logit change may not generalize to actual model behavior change. The "semantically appropriate tokens" are selected by humans (or ICL), introducing selection bias.
- **Ablation completeness attack**: If CV predicts steering, we don't know WHY. Possible mechanisms: (1) high-CV features route through more orthogonal child channels, (2) high-CV features are less "absorbed" in the sense of information loss, (3) high-CV features simply have larger magnitude. Need mediation analysis.
- **Verdict**: STRONG - The pilot evidence is compelling (2x effect, p < 0.01 implied), addresses a critical gap (actionability paradox), and has clear practical applications. Needs confound control and mechanism clarification.

### Against Candidate C (Frequency vs. Absorption)

- **Confound attack**: This is essentially a mediation analysis. But absorption is measured at the SAE level (not feature level), so we can't directly test "feature X is absorbed because of frequency Y." We measure absorption at the population level and infer causality.
- **Statistical attack**: ANCOVA assumes linearity and homoscedasticity. If the frequency-absorption relationship is nonlinear (e.g., threshold effect), ANCOVA will miss it. Nonparametric alternative needed.
- **Benchmark attack**: Co-occurrence frequency is hard to measure accurately. We compute it from activation data, but "co-occurrence" could mean many things (same token, same sentence, same document). Operationalization is ambiguous.
- **Ablation completeness attack**: This is a correlational study even with magnitude control. Confounds remain (semantic category, feature age). Only a controlled synthetic experiment could establish causality definitively.
- **Verdict**: WEAK - The correlational design cannot establish causality. The confound control (magnitude) is necessary but not sufficient. Better to design a controlled synthetic experiment.

---

## Phase 4: Refinement

### Dropped Candidates
- **Candidate C**: Too correlational; cannot establish causality even with magnitude control. The "information bottleneck" mechanism is plausible but unprovable with observational data.

### Strengthened Candidates

**Candidate A (Cross-Layer) → Strengthened to "probe projection mapping"**:
- Replace ablation-based absorption metric with SAEBench probe projection (works across all layers)
- Focus on GPT-2 only (homogeneous dictionary size eliminates confound)
- Add planned contrast: middle layers (4-8) vs outer layers (0-3, 9-11)
- Report effect sizes with confidence intervals, not just significance

**Candidate B (CV-Based Steering) → Strengthened to "CV-actionability validation"**:
- Add decoder magnitude as control variable
- Test multiple mechanisms: (1) CV alone, (2) CV + magnitude, (3) CV × absorption interaction
- Include mediation analysis: does absorption mediate CV-steering relationship?
- Replicate with independent feature sample (different CV threshold cutoffs)
- Add non-absorbed baseline for comparison

### Selected Front-Runner

**Candidate B (CV-Based Steering Validation)** is the front-runner because:
1. Strongest pilot evidence (2x effect, 9/9 words validated in activation patching)
2. Addresses field-critical question (actionability paradox, Basu et al.)
3. Clear practical application: CV as a predictor for which absorbed features remain steerable
4. Feasible in < 1 hour with proper controls

---

## Phase 5: Final Proposal

### Title

**CV Predicts Steering Effectiveness: Validating the Actionability of Absorbed SAE Features**

### Hypothesis

High coefficient of variation (CV) positively predicts steering effectiveness for absorbed SAE features. Specifically: absorbed features with CV > 5.0 show significantly larger steering effects (mean logit change > 0.10) than absorbed features with CV < 0.1 (mean logit change < 0.05), even when controlling for decoder weight magnitude.

**Falsification criterion**: If the difference in steering effect between high-CV and low-CV absorbed features is not statistically significant (p > 0.05, one-sided t-test) after controlling for decoder magnitude, the hypothesis is DISPROVEN.

### Method

Training-free analysis of pretrained GPT-2 layer 6 SAE (gpt2-small-res-jb, 16k latents) from SAELens pretrained directory. No retraining required.

### Evaluation Protocol

**Primary Benchmark**: Custom steering effectiveness test
- **SAE**: GPT-2 layer 6 residual stream SAE (16k latents, from SAELens)
- **Features**: 30 high-CV absorbed features (CV > 5.0, absorption_score > 0.5) vs 30 low-CV absorbed features (CV < 0.1, absorption_score > 0.5)
- **Steering Strengths**: ±3, ±5, ±7 (5 levels)
- **Metric**: Mean logit change at top-5 semantically appropriate tokens per feature
- **Statistical Test**: One-sided Welch's t-test (α = 0.01), with decoder magnitude as covariate (ANCOVA)
- **Multiple Testing Correction**: Benjamini-Hochberg FDR < 0.05

### Ablation Schedule

| Ablation | What it Tests | Expected Outcome |
|----------|---------------|------------------|
| High-CV vs Low-CV (raw) | CV-steering relationship | High-CV > Low-CV by ~2x |
| High-CV vs Low-CV (magnitude-controlled) | CV independent of decoder size | High-CV > Low-CV if mechanism is CV-based |
| Both groups vs Non-absorbed baseline | Whether absorption degrades steering | Absorbed < Non-absorbed |
| Multiple steering strengths | Linearity of CV-steering relationship | Proportional increase with strength |
| Mediation analysis | Does absorption mediate CV-steering? | Partial mediation expected |

### Control Experiments

1. **Magnitude control**: Ensure high-CV and low-CV groups have matched decoder weight norms
2. **Frequency control**: Ensure high-CV and low-CV groups have matched activation frequencies
3. **Semantic category control**: Ensure feature semantic categories are balanced across groups
4. **Non-absorbed baseline**: Compare to features with absorption_score < 0.1

### Pilot Design (< 15 min)

- 10 high-CV + 10 low-CV features at single steering strength (±5)
- Single steering strength test: ~20 min
- Success criterion: High-CV shows larger effect (p < 0.05 uncorrected)

### Resource Estimate

- **GPU**: Single RTX 3090/4090 or equivalent
- **Time**: 45 min for full experiment (60 features × 5 strengths = 300 steering tests)
- **Model**: GPT-2-small (86M params) - no training, inference only
- **No new training required**: Training-free analysis

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| CV difference disappears with magnitude control | Medium | Report as negative result; pivot to magnitude-based predictor |
| Steering effect doesn't generalize beyond GPT-2 | Medium | Acknowledge boundary conditions; recommend cross-architecture validation |
| High-CV features are artifacts (not genuinely absorbed) | Low | Activation patching validated 9/9 persistent words (67.3% mean recovery) |
| Effect size too small for practical utility | Medium | Report effect size with confidence intervals; let readers judge |

### Novelty Claim

This is the first systematic validation that coefficient of variation (CV) predicts steering effectiveness for absorbed SAE features. While Basu et al. (2026) established that "good detection does not guarantee steering utility," we show that a simple statistic (CV) partially predicts which absorbed features remain actionable. This connects the abstract absorption metric to practical interpretability utility.

**Prior work collision**: Basu et al. (2026) establishes the actionability paradox. This work extends Basu et al. to the non-clinical LLM domain and provides a mechanism-based predictor (CV) for which absorbed features may remain steerable.

**Differentiation from prior work**: This is NOT claiming to resolve the actionability paradox. It is providing a predictor (CV) that partially explains which absorbed features retain steering potential. The field's critical question is "why does good detection fail to predict steering?" Our answer: CV captures something about feature behavior that absorption metrics miss.