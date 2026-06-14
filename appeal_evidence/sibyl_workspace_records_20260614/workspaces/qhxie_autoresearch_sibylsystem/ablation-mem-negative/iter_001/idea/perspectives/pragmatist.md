# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **[SAELens](https://github.com/jbloomAus/SAELens)** (1,100+ stars, MIT) — Foundational library for training and analyzing SAEs. Integrates GemmaScope. Essential infrastructure. **Code exists and is mature.**

2. **[sae-spelling](https://github.com/lasr-spelling/sae-spelling)** (Chanin et al., 2024) — Official code for "A is for Absorption." Implements `FeatureAbsorptionCalculator`, k-sparse probing, integrated gradients ablation. **Direct absorption tooling available.**

3. **[SAEBench](https://github.com/adamkarvonen/SAEBench)** (Karvonen et al., 2025, MIT) — Comprehensive benchmark with 8+ metrics including absorption. Evaluated 200+ SAEs. **Standard evaluation suite.**

4. **[matryoshka_sae](https://github.com/bartbussmann/matryoshka_sae)** (Bussmann et al., 2025) — Hierarchical nested SAEs achieving ~90% absorption reduction. Based on SAELens. **Primary absorption solution with code.**

5. **[TransformerLens](https://github.com/neelnanda-io/TransformerLens)** (MIT) — Mechanistic interpretability library. Required for activation extraction and interventions.

6. **GemmaScope** (Google, 2024, Apache 2.0) — Pre-trained JumpReLU SAEs for Gemma-2-2B and Gemma-2-9B, every layer, widths 16k-1M. **Eliminates training cost.**

7. **"A is for Absorption"** (Chanin et al., arXiv:2409.14507) — Foundational paper identifying absorption. Metric relies on ablation (limited to early layers), conservative underestimate, focused on GPT-2.

8. **"Feature Hedging"** (Chanin et al., arXiv:2505.11756) — Complement to absorption. Shows Matryoshka trades absorption for hedging. Proposes balanced loss coefficients (beta_m ~ 0.75).

9. **"Are Sparse Autoencoders Useful?"** (Kantamneni et al., arXiv:2502.16681, ICML 2025) — **Critical finding**: SAEs consistently underperform logistic regression baselines on sparse probing. SAEs should be used for discovery, not acting on known concepts.

10. **"Does higher interpretability imply better utility?"** (arXiv:2510.03659) — Weak correlation (tau_b ~ 0.298) between interpretability and steering performance. For most effective steering features, correlation vanishes or becomes negative.

11. **"Orthogonal Sparse Autoencoders"** (Korznikov et al., arXiv:2509.22033) — ~65% absorption reduction via cosine similarity penalty. ~50% less compute than Matryoshka.

12. **"Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts"** (arXiv:2506.23845) — Categorizes SAE use cases. Concept detection and steering are "acting on known concepts" where SAEs fail.

### Landscape Summary

**What works:**
- SAELens + GemmaScope provide a mature, training-free analysis pipeline
- Chanin et al.'s absorption metric (k-sparse probe + integrated gradients) is the established standard
- SAEBench provides standardized evaluation across architectures
- Matryoshka SAEs reduce absorption ~90% but introduce hedging and high compute cost
- Orthogonal SAEs offer a cheaper alternative (~65% reduction, ~50% less compute)

**What doesn't:**
- SAEs underperform simple baselines (logistic regression) on downstream tasks
- Absorption metric is limited to early layers (ablation fails past layer 17 in Gemma-2-2B)
- Current solutions (Matryoshka, OrtSAE) require architectural changes and retraining
- No systematic cross-architecture absorption comparison exists
- No causal evidence links absorption to downstream interpretability task degradation

**Practical gaps:**
- No training-free post-hoc absorption detection without ground truth
- No training-free mitigation method
- Absorption metric layer limitation not addressed
- The "absorption is benign compression" hypothesis untested

---

## Phase 2: Initial Candidates

### Candidate A: Cross-Architecture Absorption Benchmark (CAAB)

**Core hypothesis**: Different SAE architectures exhibit significantly different absorption rates, with a clear Pareto frontier between absorption and reconstruction quality.

**Implementation sketch**:
- Start from SAEBench evaluation pipeline + sae-spelling absorption calculator
- Load pre-trained SAEs via SAELens: GemmaScope (JumpReLU), GPT-2 small SAEs (various architectures)
- Run Chanin et al. absorption detection on first-letter features across all architectures
- Compare: JumpReLU, TopK, Gated, BatchTopK, Matryoshka, OrtSAE

**Simplest version**:
- Load 2 pre-trained SAEs (JumpReLU + TopK) on GPT-2 Small layer 8
- Run absorption detection on 26 first-letter features
- Compare absorption rates and runtime
- **Time estimate**: 15 minutes (pilot), 2 hours (full: 5 architectures x 4 layers)

**Reusable components**:
- SAELens SAE loading (1,000+ pre-trained SAEs)
- SAEBench evaluation pipeline
- sae-spelling FeatureAbsorptionCalculator
- GemmaScope pre-trained SAEs

---

### Candidate B: Absorption-Downstream Causal Impact Assessment

**Core hypothesis**: Higher feature absorption rates quantitatively degrade downstream interpretability tasks (sparse probing, steering efficacy), controlling for reconstruction quality and sparsity.

**Implementation sketch**:
- Use pre-trained SAEs with varying absorption rates (different architectures or same architecture different sparsity)
- Measure sparse probing accuracy on 100 concepts across 5 semantic domains
- Measure steering efficacy (logit difference shift) on sentiment/topic directions
- Use partial correlation to control for reconstruction MSE, L0 sparsity, dead feature ratio

**Simplest version**:
- Take 2 SAEs with known different absorption rates (e.g., JumpReLU high vs Matryoshka low)
- Run sparse probing on 20 concepts
- Check if probe accuracy correlates with absorption rate
- **Time estimate**: 15 minutes (pilot), 2 hours (full: 100 concepts, 3 tasks)

**Reusable components**:
- SAEBench sparse probing metric
- TransformerLens for steering interventions
- SAELens for SAE loading

---

### Candidate C: Training-Free Post-Hoc Absorption Mitigation

**Core hypothesis**: A lightweight residual compensation method can recover absorbed parent feature activations at inference time without retraining the SAE.

**Implementation sketch**:
- Identify absorbed feature pairs using co-occurrence patterns (unsupervised detection)
- Train a small MLP (<1% of SAE parameters) to predict absorbed parent activation from child features
- At inference, add predicted residual to SAE output
- Validate via probe accuracy improvement

**Simplest version**:
- Identify 5 known absorbed pairs from Chanin et al. first-letter analysis
- Train tiny MLP (2-layer, 64 hidden units) to predict parent from children
- Measure if probe accuracy improves post-compensation
- **Time estimate**: 15 minutes (pilot), 1 hour (full)

**Reusable components**:
- sae-spelling absorption detection (for identifying pairs)
- PyTorch for MLP training
- SAELens for activation extraction

---

## Phase 3: Self-Critique

### Against Candidate A (CAAB)

**Implementation reality check**: SAEBench already exists and evaluates absorption. However, SAEBench's absorption metric uses probe projection (not ablation), which loses causal certainty. A systematic comparison using the original Chanin et al. ablation metric across architectures does NOT exist. The sae-spelling repo provides the calculator but only for GPT-2.

**Reproducibility attack**: High. Pre-trained SAEs are fixed artifacts. SAELens provides deterministic loading. The first-letter task is standardized. Main risk: SAELens API version compatibility (observed in pilot: pre-trained SAE loading failed due to API mismatch).

**Baseline sanity check**: The baseline is not a method but a measurement. The question is whether cross-architecture differences are large enough to be meaningful. Chanin et al. showed absorption varies with width/sparsity; extending to architecture is natural.

**Scope attack**: Limited to first-letter features (English alphabet). However, this is the established test case in the literature. Extension to other concept types would be future work.

**Verdict**: STRONG — fills a clear gap, uses existing tools, low risk.

---

### Against Candidate B (Causal Impact)

**Implementation reality check**: The 2025 paper "Are Sparse Autoencoders Useful?" already showed SAEs underperform logistic regression on sparse probing. Our hypothesis is more specific: does absorption *within* SAEs cause this underperformance? This is a finer-grained question that has not been asked.

**Reproducibility attack**: Medium. Steering efficacy depends on feature selection, which is somewhat arbitrary. Partial correlation requires sufficient variation in absorption rates across SAEs.

**Baseline sanity check**: The baseline is linear probes on raw activations. SAE features already underperform this. We're asking whether absorption explains *part* of the gap. Even if the answer is "no," it's a publishable negative result.

**Scope attack**: If absorption doesn't affect downstream tasks, the contrarian hypothesis ("absorption is benign compression") gains support. The paper pivots to analyzing why the community overemphasizes absorption.

**Verdict**: STRONG — directly testable, addresses a live debate, negative results are publishable.

---

### Against Candidate C (Training-Free Mitigation)

**Implementation reality check**: No prior work attempts training-free absorption mitigation. The closest is Matryoshka (architectural change, requires retraining). The idea of residual compensation is inspired by model editing literature (e.g., ROME, MEMIT) but applied to SAE features.

**Reproducibility attack**: High risk. The MLP may overfit to the small number of known absorbed pairs. Generalization to new pairs is unproven. The "<1% parameters" claim needs validation.

**Baseline sanity check**: Baseline is the original SAE without compensation. Improvement must be statistically significant and not increase reconstruction error.

**Scope attack**: If it only works on first-letter features, it's a one-trick demonstration. The method needs to generalize to other concept types.

**Verdict**: MODERATE — innovative but high risk. Should be deprioritized relative to A and B.

---

## Phase 4: Refinement

**Dropped**: None fully dropped, but Candidate C is deprioritized to "exploratory."

**Strengthened Candidate A (CAAB)**:
- Focus on 4 architectures: JumpReLU, TopK, BatchTopK, Matryoshka (OrtSAE if pre-trained available)
- Use Gemma-2-2B as primary model (more relevant than GPT-2)
- Sample 4 layers: early (2), middle (12), late (20), final (25)
- Add reconstruction quality and sparsity to the comparison table
- Target: clear Pareto frontier plot

**Strengthened Candidate B (Causal Impact)**:
- Use 3 downstream tasks: sparse probing, steering efficacy, feature attribution consistency
- Control for: reconstruction MSE, L0 sparsity, dead feature ratio, SAE width
- Pre-register analysis: partial correlation with Bonferroni correction
- Include contrarian hypothesis test: if absorption has no effect, report prominently

**Pilot plan**:
- P1: Load GemmaScope JumpReLU + train TopK on GPT-2 Small layer 8; run absorption detection (15 min)
- P2: Compare sparse probing on 20 concepts between high-absorption and low-absorption SAEs (15 min)
- P3: Test co-occurrence-based unsupervised detection on 100 features (15 min)

---

## Phase 5: Final Proposal

### Title
**Systematic Analysis of Feature Absorption Across SAE Architectures: A Cross-Architecture Benchmark and Causal Impact Assessment**

### Hypothesis
Different SAE architectures exhibit significantly different absorption rates (H1), and higher absorption quantitatively degrades downstream interpretability tasks when controlling for reconstruction quality and sparsity (H2).

### Motivation
Feature absorption is a known failure mode of SAEs, but no systematic cross-architecture comparison exists. The community has proposed architectural solutions (Matryoshka, OrtSAE) without rigorous benchmarking. Meanwhile, recent work shows SAEs underperform simple baselines on downstream tasks — but whether absorption is the cause remains untested. This work fills both gaps with a training-free analysis using existing pre-trained SAEs.

### Method

**Step 1: Cross-Architecture Absorption Benchmark (CAAB)**
- Load pre-trained SAEs via SAELens for 4 architectures: JumpReLU, TopK, BatchTopK, Matryoshka
- Primary model: Gemma-2-2B; fallback: GPT-2 Small
- Layers: 4 representative layers (early, middle, late, final)
- Run Chanin et al. absorption detection on first-letter features (26 letters)
- Report: absorption rate, mean/full absorption, reconstruction MSE, L0 sparsity

**Step 2: Causal Impact Assessment**
- Sparse probing: Train k-sparse probes on 100 concepts across 5 semantic domains
- Steering efficacy: Measure logit difference shift on sentiment/topic directions
- Feature attribution consistency: Integrated gradients stability across semantically similar inputs
- Statistical analysis: Partial correlation controlling for reconstruction MSE, L0, dead feature ratio

**Step 3: Exploratory — Unsupervised Detection Pilot**
- Build feature activation co-occurrence matrix
- Run hierarchical clustering to identify potential absorbed pairs
- Validate against Chanin et al. supervised labels on known cases

### Simplest Version
The absolute minimum experiment:
1. Load 2 pre-trained SAEs (JumpReLU + TopK) on GPT-2 Small layer 8
2. Run absorption detection on 26 first-letter features
3. Compare absorption rates
4. Run sparse probing on 20 concepts
5. Check correlation between absorption rate and probe accuracy

**Runtime**: 30 minutes total.

### Baselines
- **Baseline 1**: Linear regression probe on raw model activations (establishes upper bound for probing)
- **Baseline 2**: Random feature selection from SAE (establishes lower bound)
- **Baseline 3**: No absorption control (raw SAE features without any mitigation)

Expected performance ranges:
- Linear probe on activations: 85-95% accuracy
- SAE features (high absorption): 60-75% accuracy
- SAE features (low absorption): 70-85% accuracy

### Experimental Plan

| Phase | Experiment | Datasets | Metrics | Time |
|-------|-----------|----------|---------|------|
| Pilot | CAAB feasibility | GPT-2 Small, 2 architectures | Absorption rate | 15 min |
| Pilot | Causal impact feasibility | 20 concepts | Probe accuracy | 15 min |
| Pilot | UAD feasibility | 100 features | Precision/recall | 15 min |
| Full | CAAB | Gemma-2-2B, 4 architectures, 4 layers | Absorption rate, MSE, L0 | 2 h |
| Full | Causal impact | 100 concepts, 3 tasks | Partial correlation | 2 h |
| Full | UAD validation | Known absorbed pairs | Precision vs. supervised | 1 h |

### Resource Estimate
- **GPU**: Single GPU (RTX 3090 or better)
- **Model sizes**: GPT-2 Small (124M), Gemma-2-2B (2B)
- **Per-task time**: <1 hour (well within budget)
- **Pilot time**: 45 minutes total
- **Full experiment time**: ~6 hours total
- **No training required** for core experiments (uses pre-trained SAEs)

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Gemma-2-2B SAE loading fails | Medium | High | Fallback to GPT-2 Small |
| SAELens API version mismatch | Medium | Medium | Pin version; use pip install sae-lens==x.y.z |
| Absorption rates too similar across architectures | Low | High | Report as negative result; discuss why architecture doesn't matter |
| Absorption unrelated to downstream performance | Low | Medium | Pivot to "absorption is benign compression" analysis |
| Chanin metric layer limitation | High | Low | Use SAEBench probe-projection variant for late layers; note limitation |

### Novelty Claim
1. **First systematic cross-architecture absorption comparison** using standardized metrics (Gap 3 from literature survey)
2. **First causal assessment** of absorption's impact on downstream interpretability tasks (Gap 5)
3. **Training-free analysis** using existing pre-trained SAEs — no training cost, fully reproducible
4. **Open-source benchmark dataset** enabling community follow-up

### What Exactly Is New
Even if the results are negative (no architecture differences, no causal impact), the paper provides the first rigorous evidence for these questions. The field currently assumes absorption is a problem that needs fixing; this work tests that assumption.

---

## Sources

- [SAELens GitHub](https://github.com/jbloomAus/SAELens)
- [sae-spelling GitHub](https://github.com/lasr-spelling/sae-spelling)
- [SAEBench GitHub](https://github.com/adamkarvonen/SAEBench)
- [matryoshka_sae GitHub](https://github.com/bartbussmann/matryoshka_sae)
- ["A is for Absorption" arXiv:2409.14507](https://arxiv.org/abs/2409.14507)
- ["SAEBench" arXiv:2503.09532](https://arxiv.org/abs/2503.09532)
- ["Feature Hedging" arXiv:2505.11756](https://arxiv.org/abs/2505.11756)
- ["Are Sparse Autoencoders Useful?" arXiv:2502.16681](https://arxiv.org/abs/2502.16681)
- ["Does higher interpretability imply better utility?" arXiv:2510.03659](https://arxiv.org/abs/2510.03659)
- ["Orthogonal Sparse Autoencoders" arXiv:2509.22033](https://arxiv.org/abs/2509.22033)
- ["Use Sparse Autoencoders to Discover Unknown Concepts" arXiv:2506.23845](https://arxiv.org/abs/2506.23845)
