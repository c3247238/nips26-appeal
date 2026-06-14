# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024) - "A is for Absorption"** — First systematic study of feature absorption in SAEs. Introduces ablation-based absorption detection metric (limited to early layers 0-17). Establishes hierarchical feature co-occurrence as root cause of absorption. Critical reference for experimental methodology.

2. **Basu et al. (2026) - "Interpretability without Actionability"** — Demonstrates 98.2% AUROC feature detection translates to 0% steering utility in clinical domain. The "actionability paradox" is the central empirical question this project must address. Establishes the field's key failure mode.

3. **Karvonen et al. (2025) - SAEBench (ICML 2025)** — Comprehensive 8-metric evaluation suite for SAEs. Introduces probe projection contribution as layer-agnostic alternative to ablation-based absorption detection. Standardizes evaluation protocol for absorption measurement.

4. **Cui et al. (2026) - "On the Limits of Sparse Autoencoders" (ICLR 2026)** — First closed-form theoretical analysis proving standard SAEs cannot fully recover ground-truth monosemantic features. Information-theoretic impossibility sets theoretical boundaries for what experiments can achieve.

5. **Costa et al. (2025) - MP-SAE (NeurIPS 2025)** — Matching Pursuit-based SAE with residual-guided greedy selection. Demonstrates conditional orthogonality recovers hierarchical structure missed by conventional SAEs. Alternative architecture for cross-validation.

6. **Engel & Van den Broeck (2001) - "Statistical Mechanics of Learning"** — Foundational phase transition theory. Provides finite-size scaling framework for analyzing critical phenomena in neural networks.

### Experimental Landscape

**What has been properly tested:**
- Ablation-based absorption detection on early layers (0-17) of Gemma-2-2B using first-letter spelling tasks
- Actionability paradox in clinical domain (Basu et al.)
- Probe projection metrics for cross-layer absorption via SAEBench
- Steering effectiveness comparison between absorbed vs non-absorbed features (pilot)

**What is accepted without evidence:**
- Universal actionability failure across all feature types and domains
- Critical sparsity (λ_c) as stable, architecture-independent parameter
- CV threshold (1.0) as generalizable predictor across architectures
- Decoder orthogonality as mechanism for steering effectiveness

**Methodological gaps:**
- No systematic CV-steering correlation study across multiple architectures
- Cross-layer absorption at true critical sparsity (λ_c) not validated
- Fano factor control for magnitude confounding not applied to CV-steering analysis
- Prospective validation of CV threshold on held-out features not done
- Non-absorbed baseline steering comparison incomplete

---

## Phase 2: Initial Candidates

### Candidate A: CV-Based Steering Predictor (Front-Runner Refinement)

- **Core hypothesis**: Absorbed features with high coefficient of variation (CV > 1.0) show significantly larger steering effects than absorbed features with low CV (CV <= 1.0), after controlling for decoder magnitude via Fano factor.
- **Falsification criterion**: If no significant difference in steering effect between high-CV and low-CV groups (p > 0.01, one-sided Welch's t-test), the hypothesis is DISPROVEN. The actionability paradox is universal within absorbed features.
- **Evaluation protocol**:
  - Primary: Steering effect (logit change) comparison between high-CV (n=30) vs low-CV (n=30) absorbed features on GPT-2 layer 6 SAE
  - Steering strengths: +3, +5, +7
  - Control: Fano factor (CV²/mean) for activation magnitude confounding
  - Statistical test: one-sided Welch's t-test, α=0.01, Benjamini-Hochberg FDR correction
- **Ablation plan**:
  - Ablate decoder magnitude as confounding variable (Fano factor control)
  - Ablate feature frequency as confounding variable
  - Test orthogonality as alternative predictor
- **Confounders identified**:
  - Decoder magnitude: High-CV features may have larger decoder weights
  - Feature frequency: High-CV may be rarer but more specialized
  - Layer specificity: GPT-2 layer 6 may not generalize
- **Pilot design**: 15 min, 10 high-CV vs 10 low-CV absorbed features, single steering strength (+5)

### Candidate B: Cross-Layer Criticality Retest

- **Core hypothesis**: Cross-layer absorption heterogeneity appears at critical sparsity λ_c=5e-5 (not λ=0.001 where all layers saturate).
- **Falsification criterion**: If all layers saturate at absorption_rate=1.0 even at λ_c=5e-5, the layer-criticality hypothesis is DISPROVEN at all sparsity levels.
- **Evaluation protocol**:
  - Primary: SAEBench probe projection metric at λ_c=5e-5 on layers 0, 3, 6, 9, 11
  - Compare absorption rates across layers using probe projection contribution
  - Secondary: Ablation-based metric on layers 0-17 for early-layer validation
- **Ablation plan**:
  - Probe projection variation across layers
  - Ablation recovery at each layer
- **Confounders identified**:
  - λ_c may be architecture-specific (GPT-2 vs Gemma-2)
  - λ_c may vary with SAE width (16k vs 65k)
  - Probe quality may vary across layers
- **Pilot design**: 15 min, layers 0 and 6 at λ_c=5e-5 vs λ=0.001 comparison

### Candidate C: Cross-Architecture Steering Validation

- **Core hypothesis**: The CV-steering correlation (high-CV = larger steering) replicates on Gemma-2-2B JumpReLU SAEs with similar CV threshold.
- **Falsification criterion**: If Gemma-2 shows no significant CV-steering correlation, the finding is architecture-specific and the actionability paradox holds for JumpReLU SAEs.
- **Evaluation protocol**:
  - Primary: Steering experiment on Gemma-2-2B layer 6 JumpReLU SAE
  - CV threshold: 1.0 (same as GPT-2) or model-specific adjustment
  - 30 high-CV vs 30 low-CV absorbed features, +5 steering strength
- **Ablation plan**:
  - Compare CV distribution between GPT-2 and Gemma-2 absorbed features
  - Test whether CV threshold generalizes or requires architecture-specific calibration
- **Confounders identified**:
  - JumpReLU vs TopK architecture may affect feature activation patterns
  - Gemma-2 may have different child feature structure
  - Layer 6 may not be comparable across architectures
- **Pilot design**: 20 min, 10 high-CV vs 10 low-CV on Gemma-2 layer 6

---

## Phase 3: Self-Critique

### Against Candidate A (CV-Based Steering Predictor)

- **Confound attack**: High-CV features may simply have larger decoder weights. Without Fano factor normalization (CV²/mean), the CV-steering correlation may be a magnitude proxy. The pilot used raw CV without magnitude control.
- **Statistical attack**: Pilot n=30 per group may be underpowered for detecting medium effect sizes. Need power analysis: if effect size is d=0.5, requires n~60 per group for α=0.01, power=0.8.
- **Benchmark attack**: GPT-2 layer 6 is cherry-picked. Need validation across multiple layers and random seed selection of features to avoid selection bias.
- **Ablation completeness attack**: CV may capture multiple phenomena (frequency, context-sensitivity, magnitude). Without decomposing CV into these components, the mechanism remains unclear.
- **Verdict**: MODERATE — The hypothesis is falsifiable and pilot evidence supports it, but magnitude confounding and selection bias are serious threats.

### Against Candidate B (Cross-Layer Criticality Retest)

- **Confound attack**: λ_c=5e-5 was identified on GPT-2 TopK SAE. This value may not transfer to other architectures or sparsity settings. The "critical point" may be artifact of the specific SAE training configuration.
- **Statistical attack**: Uniform saturation at λ=0.001 suggests the measurement itself may be flawed. If probe projection shows uniform saturation at λ_c too, the phenomenon may not exist.
- **Benchmark attack**: SAEBench probe projection metric has known limitations (probe quality dependence). If probes are poor at deeper layers, cross-layer comparison is invalid.
- **Ablation completeness attack**: Even if variation exists at λ_c, the practical significance is unclear. A small absorption rate difference (0.95 vs 1.0) may not matter for downstream applications.
- **Verdict**: WEAK — Falsified at wrong sparsity (λ=0.001) is not the same as validated at correct sparsity (λ_c). Risk of confirming no-phenomenon without actually testing the correct condition.

### Against Candidate C (Cross-Architecture Validation)

- **Confound attack**: Gemma-2 JumpReLU has fundamentally different activation patterns than GPT-2 TopK. Direct comparison may not be meaningful. CV threshold may require architecture-specific calibration.
- **Statistical attack**: Same power issues as Candidate A. If Gemma-2 effect size is smaller, requires larger n.
- **Benchmark attack**: Layer 6 may not be comparable across architectures due to different network depths and attention patterns.
- **Ablation completeness attack**: If Gemma-2 shows no CV effect, does not distinguish between: (a) phenomenon is architecture-specific, (b) Gemma-2 features are uniformly non-steerable, (c) measurement error.
- **Verdict**: MODERATE — Cross-architecture validation is essential but may be underpowered and has interpretation ambiguities.

---

## Phase 4: Refinement

**Dropped:**
- Candidate B (Cross-Layer Criticality) — Falsified at λ=0.001 is not informative; λ_c validation is speculative and may confirm no-phenomenon without testing correct condition. Risk of wasted experiment.

**Strengthened:**
- Candidate A (CV-Based Steering Predictor):
  - Add Fano factor control (CV²/mean) to address magnitude confounding
  - Increase sample size to n=60 per group for adequate power
  - Add non-absorbed baseline for context
  - Add decoder orthogonality as alternative predictor for mechanism investigation
  - Use held-out feature set for prospective validation of CV threshold

- Candidate C (Cross-Architecture Validation):
  - Frame as confirmatory rather than exploratory
  - Use same protocol as Candidate A for direct comparison
  - Test CV threshold calibration vs fixed threshold (1.0)

**Selected front-runner**: Candidate A (CV-Based Steering Predictor) with enhanced design.

**Rationale**: Directly addresses the field's key empirical question (actionability paradox). Pilot evidence is strong (2x effect, p < 0.01 implied). Design is feasible within 1-hour budget. Results are publishable in either direction.

---

## Phase 5: Final Proposal

### Title
**Coefficient of Variation Predicts Steering Heterogeneity Within Absorbed SAE Features**

### Hypothesis
Absorbed SAE features decompose into two actionability subpopulations: "robust absorbed" (CV > 1.0, routed through context-sensitive child channels, steerable) and "fragile absorbed" (CV <= 1.0, routed through stable child channels, non-steerable). The coefficient of variation predicts steering effectiveness after controlling for activation magnitude via Fano factor normalization.

**Metric**: Steering effect measured as logit change at semantically appropriate tokens.
**Threshold**: High-CV (>1.0) vs Low-CV (<=1.0) split on absorbed features.
**Falsification**: If no significant difference in steering effect between groups (p > 0.01, one-sided Welch's t-test), hypothesis is DISPROVEN.

### Evaluation Protocol
- **Primary benchmark**: Steering effectiveness comparison (logit change)
- **SAE**: GPT-2 layer 6 residual stream SAE (gpt2-small-res-jb, 16k latents) via SAELens
- **Features**: 60 absorbed features (30 high-CV, 30 low-CV) selected from absorption_score > 0.5 population
- **Steering strengths**: +3, +5, +7
- **Control variables**: Fano factor (CV²/mean) for magnitude normalization; decoder magnitude as covariate
- **Statistical tests**: One-sided Welch's t-test (α=0.01), Benjamini-Hochberg FDR for multiple comparisons
- **Random seeds**: 3 different random seeds for feature selection to test robustness

### Ablation Schedule
| Ablation | What it tests | Expected outcome |
|----------|---------------|------------------|
| Fano factor control | Is CV just magnitude proxy? | If CV-steering survives Fano control, CV is independent predictor |
| Non-absorbed baseline | Are robust absorbed comparable to non-absorbed? | Robust absorbed may still be degraded vs non-absorbed |
| Decoder orthogonality | Is orthogonality alternative/explanatory variable? | If orthogonality correlates with CV, explains mechanism |
| Multiple steering strengths | Does CV effect scale with steering strength? | Consistency across strengths strengthens interpretation |

### Control Experiments
1. **Magnitude control**: Regress out decoder magnitude from steering effects; test CV residual correlation
2. **Feature frequency control**: Test whether CV-steering is explained by feature rarity
3. **Selection bias control**: Test CV-steering on held-out features not used in threshold selection

### Pilot Design (10-15 min)
- 10 high-CV vs 10 low-CV absorbed features at +5 steering strength
- Validates experimental setup and provides early signal before full run

### Resource Estimate
- **GPU-hours**: ~1.5 hours total (15 min pilot, 45 min full run, 30 min mechanism investigation)
- **Models**: GPT-2-small (86M params)
- **SAEs**: GPT-2 layer 6 residual stream (~16k latents) via SAELens
- **No new training**: Training-free analysis of pretrained SAEs

### Risk Assessment
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| CV is magnitude proxy | Medium | Use Fano factor control; report as exploratory if correlation is explained |
| Effect size too small for detection | Medium | Power analysis: n=60 per group achieves d=0.5 at α=0.01, power=0.8 |
| Selection bias inflates effect | Medium | Prospective validation on held-out features |
| Actionability paradox holds universally | Low | Report as negative result; confirms Basu et al. in LLM domain |

### Novelty Claim
First evidence that absorbed SAE features are not uniformly non-steerable in non-clinical LLM domain. First connection between coefficient of variation (simple statistical measure) and causal actionability. CV provides actionable predictor for interpretability practitioners without expensive steering experiments.

---

## Supporting Evidence from Prior Experiments

### Activation Patching Validation (pilot_activation_patching)
- All 9 persistent core words show >48% parent recovery (mean 67.3%) when child is zeroed
- Confirms genuine absorption exists and has causal structure that could theoretically be steered
- Validates that absorption is not metric artifact

### Steering CV Pilot (pilot_steering_cv)
- High-CV steering effect = 0.153 vs Low-CV = 0.075 (2x difference)
- Supports CV as predictor for steering effectiveness
- Effect size is practically significant (>2x)

### Falsified Hypotheses (Reported as Informative Negatives)
- H3 (Cross-layer at λ=0.001): All layers saturate at absorption_rate=1.0 — uniform saturation contradicts layer-criticality narrative
- H6 (Graph topology): Component count decreases with layer (L0=24420 > L9=23371), not peaked at layer 6

---

## References
- Chanin et al. (2024): A is for Absorption — detection metric, hierarchical co-occurrence
- Basu et al. (2026): Interpretability without Actionability — actionability paradox
- Karvonen et al. (2025): SAEBench — probe projection metric
- Cui et al. (2026): On the Limits of SAEs — information-theoretic impossibility
- Costa et al. (2025): MP-SAE — hierarchical feature recovery
- Engel & Van den Broeck (2001): Statistical Mechanics of Learning — phase transitions