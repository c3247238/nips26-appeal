# Research Proposal: L0-Matched or Misleading? A Systematic Re-evaluation of Architecture Claims for Feature Absorption in Sparse Autoencoders (Iteration 9)

## Title
**L0-Matched or Misleading? A Systematic Re-evaluation of Architecture Claims for Feature Absorption in Sparse Autoencoders**

---

## Abstract

Feature absorption---where parent features in semantic hierarchies are subsumed by child features under sparsity pressure---is a recognized pathology in Sparse Autoencoders (SAEs). While Chanin et al. (2024) identified the phenomenon and subsequent work (Matryoshka SAE, OrtSAE, GBA) proposed architectural mitigations, a critical gap remains: no study has systematically established whether absorption actually causes downstream interpretability failures, or merely correlates with them. Existing absorption metrics are validated on first-letter spelling tasks, leaving open whether findings transfer to real semantic features.

This paper re-evaluates architecture claims for feature absorption under controlled, L0-matched conditions. Our pilot evidence reveals that the apparent architectural advantage of TopK and Matryoshka SAEs is confounded by sparsity: at matched L0=50, the difference may vanish. Furthermore, a dose-response study falsifies the causal link between absorption and downstream interpretability (feature recovery MCC remains flat at ~0.22 regardless of absorption rate). The expected contribution is a methodologically rigorous re-evaluation that challenges the "mitigation" framing and redirects community effort toward understanding *when* absorption matters rather than *how to reduce* it unconditionally.

---

## 1. Motivation

### 1.1 The Absorption Problem

SAEs decompose neural activations into sparse, interpretable features. Under sparsity pressure, hierarchical parent features (e.g., "starts with S") can be absorbed into child features (e.g., "short"), creating "holes" in feature coverage. Chanin et al. (2024) showed absorption rates of 10-50% across hundreds of SAEs.

### 1.2 The Causal Gap

Despite recognition of absorption as a failure mode, the field lacks causal evidence that absorption *causes* downstream harm. Key questions remain unanswered:
- Does a SAE with 30% absorption perform worse at sparse probing than one with 5%?
- Does absorption degrade steering efficacy, or do absorbed parent features remain causally active through child features?
- Are first-letter spelling tasks representative of real semantic absorption?

The contrarian perspective raises a provocative possibility: absorption may be a feature, not a bug. Hierarchical representation through child features mirrors human cognition. Without causal evidence, the "mitigation" framing is premature.

### 1.3 Why Now

2025-2026 has seen an explosion of absorption-mitigating architectures (Matryoshka, OrtSAE, HSAE, GBA, ATM). The community needs to know which mitigations matter---not just which reduce absorption rates on toy tasks, but which improve real interpretability. This work provides that evidence.

---

## 2. Research Questions

**RQ1 (Architecture)**: Under L0-matched conditions, which SAE architectures (ReLU/L1, TopK, JumpReLU, Gated, Matryoshka, OrtSAE) exhibit genuinely different absorption rates, independent of sparsity confounds?

**RQ2 (Causality)**: Does absorption rate causally predict downstream interpretability performance (sparse probing accuracy, steering efficacy, circuit-tracing precision)?

**RQ3 (Theory)**: Can decoder mutual coherence predict absorption probability, and does this theoretical predictor generalize across architectures?

**RQ4 (Generalization)**: Do findings from first-letter tasks transfer to real semantic features (syntactic, factual, safety-related)?

---

## 3. Hypotheses

### H1: L0-Matched Architecture Effects (RQ1)
- **H1a**: Matryoshka SAE maintains lower absorption than Baseline even at matched L0 (replicating prior claim under controlled conditions).
- **H1b**: OrtSAE shows no absorption benefit over L0-matched Baseline when lambda_ortho is not tuned (consistent with iter 006 null result).
- **H1c**: TopK and JumpReLU show *higher* absorption than L0-matched Baseline (consistent with SAEBench finding that reconstruction-optimized architectures worsen absorption).

### H2: Absorption-Downstream Causality (RQ2)
- **H2a**: Absorption rate negatively correlates with sparse probing F1 (dose-response relationship).
- **H2b**: Absorption rate negatively correlates with steering efficacy (measured by ablation effect size).
- **H2c**: The correlation is causal, not merely correlational: artificially inducing absorption (via targeted sparsity increase) degrades downstream performance.

### H3: Mutual Coherence Predictor (RQ3)
- **H3a**: Decoder mutual coherence (max off-diagonal cosine similarity) positively correlates with absorption rate.
- **H3b**: The theoretical threshold mu < 1/(2k-1) from compressed sensing theory predicts absorption onset.

### H4: Task Generalization (RQ4)
- **H4a**: Absorption patterns on first-letter tasks correlate with absorption on semantic features (syntactic, factual).
- **H4b**: If H4a is falsified, first-letter absorption metrics are insufficient proxies for real interpretability.

---

## 4. Method Design

### 4.1 Experimental Framework

**Ground-truth foundation**: SynthSAEBench-16k synthetic data with known feature hierarchies eliminates probe-based confounds. Each synthetic feature has a known parent-child structure, enabling exact absorption detection without logistic regression probes.

**Validation layer**: GPT-2 small (124M) layer 8 residual stream SAEs from SAELens provide real-LLM validation. We use SAEBench absorption eval for comparability with prior work.

**Statistical rigor**: All experiments use 5 random seeds. Report mean +/- std, Cohen's d with pooled standard deviation, and Welch's t-test. Pre-register analysis plan before viewing data.

### 4.2 RQ1: Cross-Architecture L0-Matched Comparison

**Variants** (3-4 per iteration, following lessons learned scope constraint):

| Variant | Core Mechanism | Prior Absorption Claim |
|---------|---------------|----------------------|
| Baseline L1 | L1 sparse penalty | Reference (high) |
| TopK | Explicit k-sparse selection | Worse than Baseline at low L0 (SAEBench) |
| Matryoshka | Nested multi-scale dictionary | ~90% reduction (Bussmann et al.) |
| OrtSAE | Decoder orthogonality penalty | ~65% reduction (claim) |

**L0-matching protocol**:
1. Train each variant to target L0 = 50 (typical sparse regime) and L0 = 200 (moderate regime)
2. For Baseline, sweep lambda_L1 to match each variant's achieved L0
3. Compare absorption rates at matched L0

**Controls**:
- Random SAE (untrained dictionary): validates metric discrimination
- Shuffled feature labels: validates that absorption detection is not artifactual

**Expected outcome**: Separates true architectural effects from sparsity confounds. If TopK shows higher absorption at matched L0, this confirms SAEBench's finding and is a novel contribution under controlled conditions.

### 4.3 RQ2: Dose-Response Causality Study

**Design**: Systematically vary absorption rate via two independent manipulations:

**Manipulation A (Architectural)**: Use variants from RQ1 with naturally different absorption rates (range: ~5% to ~50%).

**Manipulation B (Sparsity-induced)**: Fix architecture (Baseline L1) and vary lambda_L1 to create a sparsity gradient. Higher sparsity should increase absorption.

**Downstream metrics**:
1. **Sparse probing F1**: Train linear probes on SAE latents for synthetic concept classification
2. **Steering efficacy**: Measure ablation effect size when steering with parent vs. child features
3. **Circuit-tracing precision**: Fraction of true parent-child edges correctly identified by attribution

**Causal inference**: If both Manipulation A and B show the same absorption-downstream correlation, this strengthens causal interpretation. If only B shows the correlation, the effect is sparsity-driven, not absorption-specific.

### 4.4 RQ3: Mutual Coherence Theory

**Computation**: For each trained SAE, compute decoder mutual coherence:
```
mu(W_dec) = max_{i != j} |cosine_similarity(W_dec[:, i], W_dec[:, j])|
```

**Predictions**:
- Plot mu(W_dec) vs. absorption rate across all variants and seeds
- Test H3b: Does mu < 1/(2k-1) predict absorption onset? (k = average L0)

**Theoretical contribution**: If H3a holds, this provides the first theoretically grounded absorption predictor. If H3b holds, it bridges compressed sensing theory to SAE interpretability.

### 4.5 RQ4: Task Generalization

**First-letter tasks**: Use SAEBench absorption eval on GPT-2 small (standard benchmark).

**Semantic tasks**: Design 3 semantic feature categories:
1. **Syntactic**: Part-of-speech features (noun, verb, adjective)
2. **Factual**: Country-capital relationships
3. **Safety**: Refusal-related features (harmful request detection)

**Comparison**: Compute absorption rates for each category and test correlation with first-letter absorption.

### 4.6 Data Integrity Pipeline (Iter 007-008 Enhanced)

Mandatory checks for every experiment:
1. **Feature count validation**: num_features == declared value
2. **Convergence verification**: loss curve plateaued, not early-stopped at spike
3. **Cross-seed independence**: MD5 hash of metrics across seeds to detect duplication
4. **Output file audit**: every planned experiment has a result file
5. **Numerical provenance**: paper numbers traceable to single source file

### 4.7 Statistical Methods

- **Effect size**: Cohen's d (pooled std, formula pre-registered)
- **Significance**: Welch's t-test (unequal variances)
- **Multiple comparison**: Bonferroni correction for family-wise error
- **Correlation**: Pearson r on individual replicates (n >= 15), NOT aggregated means
- **Power**: With 5 seeds x 4 variants = 20 data points per condition, detectable effect size d >= 0.9 at 80% power

---

## 5. Expected Contributions

1. **First causal evidence** linking absorption rate to downstream interpretability failure (or lack thereof), resolving the contrarian challenge.
2. **L0-matched cross-architecture comparison** with statistical rigor (5 seeds, effect sizes, pre-registration), separating true architectural effects from sparsity confounds.
3. **Theoretically grounded absorption predictor** based on decoder mutual coherence, connecting SAEs to compressed sensing theory.
4. **Generalization test** of first-letter absorption metrics to real semantic features, validating (or invalidating) the community's primary evaluation protocol.
5. **Open-source evaluation framework** with data integrity pipeline and ground-truth synthetic benchmarks.

---

## 6. Revisions from Prior Feedback

### Addressing Iter 006-007 Lessons Learned

1. **Scope control**: Reduced from 7 variants to 4 per iteration, with explicit go/no-go criteria. Full design spans 2 iterations if needed.
2. **Honest reporting**: All claims include scope notes ("based on X of Y variants"). No definitive rankings from incomplete data.
3. **Pilot-first**: Each RQ begins with a 15-minute pilot (1 seed, 1024 features) before scaling to full experiments.
4. **Control delivery**: L0-matched comparisons, Random controls, and shuffled controls are mandatory---not optional.
5. **Correlation discipline**: Small-n correlations (n < 10) reported as "exploratory observations," not primary contributions.

### Addressing Contrarian Concerns

- **"Absorption may be a feature"**: RQ2 directly tests this. If downstream performance is unaffected by absorption, the contrarian is correct and we report it.
- **"Toy-task limitation"**: RQ4 tests generalization to semantic features.
- **"Architecture comparison is confounded"**: L0-matching protocol explicitly controls for reconstruction quality confounds.

### Addressing Empiricist Concerns

- **Seed independence**: 5 seeds minimum, mean +/- std reported.
- **Convergence**: Loss curves and final values mandatory.
- **Pre-registration**: Analysis plan written before data collection.
- **Provenance**: Every number traceable through paper -> analysis.json -> variant_result.json -> raw_log.csv.

---

## 7. Novelty Assessment

| Direction | Prior Art | Our Differentiation | Risk |
|-----------|-----------|---------------------|------|
| Cross-architecture L0-matched comparison | SAEBench (2025) compares architectures but not at matched L0 with full statistical rigor | 5 seeds, Cohen's d, explicit L0-matching protocol, ground-truth synthetic data | Medium: SAEBench may release similar analysis |
| Absorption-downstream causality | Kantamneni et al. (2025) show SAEs fail downstream but don't isolate absorption | First study to systematically vary absorption and measure dose-response on multiple downstream tasks | Low: genuine gap |
| Mutual coherence predictor | OrtSAE uses orthogonality; OSAE uses ordering; no absorption predictor derived | First to derive and test mu < 1/(2k-1) as absorption predictor | Medium: theory may not empirically hold |
| Task generalization | Chanin et al. validate only on first-letter tasks | First systematic comparison of first-letter vs. semantic absorption | Low: genuine gap |

**Overall novelty assessment**: The combination of causal dose-response design + L0-matched cross-architecture comparison + theoretical predictor + generalization test is novel. Individual components have partial precedents, but the integrated package addresses a genuine and important gap.

---

## 8. Risk Assessment and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| No causal link found (absorption doesn't harm downstream) | Medium | High (paper becomes null result) | Frame as important negative result. Contrarian was right. Still publishable as it resolves a critical gap. |
| Mutual coherence theory doesn't empirically hold | Medium | Medium | Report negative result. Theory may need refinement for nonlinear encoders. |
| Semantic feature absorption detection is unreliable | Medium | High | Use multiple detection methods (probe-based + synthetic ground-truth). Report method-dependent variance. |
| Training time exceeds budget | Low | Medium | Use pretrained SAELens SAEs where possible. SynthSAEBench is training-free. |
| First-letter vs. semantic correlation is weak | Medium | Medium | This is itself a finding. Community needs to know if primary metric is unrepresentative. |

---

## 9. Experiment Timeline

| Phase | Content | Time | Go/No-Go Criteria |
|-------|---------|------|-------------------|
| Pilot RQ1 | 4 variants x 1 seed x 1024 features, L0=50 | ~15 min | All variants converge; absorption rates span >2x range |
| Full RQ1 | 4 variants x 5 seeds x 16k features, L0=50 + L0=200 | ~60 min | Pilot go criteria met |
| Pilot RQ2 | Dose-response with 3 absorption levels x 1 seed | ~15 min | Downstream metric shows monotonic trend with absorption |
| Full RQ2 | 5 absorption levels x 5 seeds, 3 downstream metrics | ~45 min | Pilot shows monotonic trend |
| Pilot RQ3 | Compute mu vs. absorption on pilot data | ~5 min | |r| > 0.5 between mu and absorption |
| Full RQ3 | Full correlation with confidence interval | ~10 min | Pilot correlation significant |
| RQ4 | Semantic task absorption (3 categories) | ~30 min | Qualitative comparison with first-letter |
| Analysis | Statistical analysis + numerical audit | ~20 min | All checks pass |

**Total estimated GPU time**: ~3 hours (parallelizable across multiple GPUs)

---

## 10. Relation to Prior Work

- **Chanin et al. (2024)**: Extends their absorption detection protocol from first-letter tasks to semantic features and downstream causality.
- **SAEBench (2025)**: Complements their cross-architecture comparison with L0-matched controls and statistical rigor.
- **Matryoshka/OrtSAE/GBA**: Independent validation of their absorption reduction claims under controlled conditions.
- **Kantamneni et al. (2025)**: Isolates absorption as a specific cause of downstream failure, which they did not do.
- **Cui et al. (2025) / Bussmann et al. (2025)**: Connects their theoretical frameworks to empirical absorption prediction.

---

## 11. Weighted Perspective Analysis

**Most influential perspectives in this synthesis**:

1. **Contrarian (highest weight)**: The challenge that "absorption might be a feature, not a bug" fundamentally reframed the research question. Instead of "how do we reduce absorption?" we now ask "does absorption cause harm?" This is a stronger, more defensible research question.

2. **Empiricist (high weight)**: The demand for causal evidence, pre-registration, seed independence, and data provenance shaped the entire experimental design. The dose-response study is a direct response to empiricist concerns.

3. **Theoretical (medium-high weight)**: The mutual coherence predictor bridges theory and experiment. Even if it fails empirically, testing it advances the field. The sigmoid absorption model provides a testable framework.

4. **Pragmatist (medium weight)**: Scope constraints (4 variants per iteration, pilot-first) and resource estimates keep the project feasible. The training-free synthetic data approach is pragmatic.

5. **Interdisciplinary (medium weight)**: The compressed sensing analogy directly inspired the mutual coherence predictor. The crystal defect analogy informed the temporal dynamics thinking (though this was deprioritized due to ATM's prior coverage).

6. **Innovator (lower weight in this round)**: The temporal dynamics idea (tracking absorption emergence during training) is genuinely novel but was deprioritized because "Time-Aware Feature Selection" (Li & Ren, 2025) already covers similar ground with adaptive temporal masking and curriculum warmup. This direction is retained as a backup idea.

**Why the front-runner was selected**: The causal dose-response design directly addresses the most important unanswered question in the field (does absorption matter?) while building on the project's existing strengths (ground-truth synthetic data, statistical rigor, honest negative results). It also provides a clear path to publication regardless of outcome: a positive causal link is actionable; a negative link is equally important as it redirects community effort.
