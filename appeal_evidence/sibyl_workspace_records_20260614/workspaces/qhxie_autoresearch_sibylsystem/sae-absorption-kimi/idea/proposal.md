# Research Proposal

## Title

**Absorption is a Sparsity Phenomenon: Re-evaluating Architectural Claims in Sparse Autoencoders**

(Alternative: **What Actually Reduces Feature Absorption? Evidence from Ground-Truth Synthetic Hierarchies**)

## Abstract

Feature absorption---the subsumption of parent features by child features in sparse autoencoders (SAEs)---has become a central optimization target for the SAE community. Multiple architectures (Matryoshka, OrtSAE, HSAE) report absorption reduction as a primary contribution, attributing improvement to multi-scale dictionaries, orthogonality penalties, and gating mechanisms. Yet no study has isolated which specific architectural component drives the improvement, nor has any tested whether absorption reduction is an architectural effect or merely a sparsity-level artifact.

Our prior iterations revealed severe construct-validity problems with probe-based absorption metrics on real LLMs: co-occurrence pairs produce higher "absorption" than true hierarchies (Cohen's d = -1.79, p = 0.003), perfect probe AUROCs create ceiling effects, and the metric diverges dramatically across base models (GPT-2 vs. Pythia). These anomalies make real-LLM absorption measurement untrustworthy for causal claims.

This paper pivots to **SynthSAEBench-16k**, a synthetic benchmark with 16,384 ground-truth features (10,884 hierarchical) where absorption can be measured directly---no probes, no ambiguity, no ceiling effects. We train SAE variants varying one component at a time and measure ground-truth absorption rate, reconstruction MSE, and L0 sparsity. Our central finding challenges the field's attribution: **TopK sparsity (k=50) reduces absorption by 78% with Cohen's d = 5.51, while orthogonality penalties achieve only 2.7% reduction (d = 0.14)**. The L0-absorption correlation across variants is r ~ -0.99, suggesting absorption is primarily a sparsity-level phenomenon, not an architecture-specific pathology.

This yields the **first component-isolated causal analysis** of SAE architectural innovations, with a reframed conclusion: the community has been optimizing the wrong variable. The operative factor is sparsity level, not architectural complexity.

## Motivation

The SAE community has coalesced around feature absorption as a central pathology. Chanin et al. (2024) proved analytically that absorption is incentivized by sparsity loss, and SAEBench standardized the detection protocol. Follow-up architectures all report absorption reductions:
- Matryoshka SAEs: ~10x absorption reduction (Bussmann et al., 2025)
- OrtSAE: 65% absorption reduction (Korznikov et al., 2025)
- HSAE: explicit tree-structured constraints (Zhan et al., 2026)

Yet our prior construct-validity experiments (iter_002-004) revealed fatal anomalies in probe-based absorption measurement:
1. **Co-occurrence confound**: Non-hierarchy pairs produce higher absorption than true hierarchies (0.331 vs. 0.235, t = -4.748, p = 0.003), proving the metric detects correlation, not containment.
2. **Ceiling effect**: All 80 probe AUROCs = 1.0, collapsing the absorption formula to `1.0 - k_sparse_acc`.
3. **Model dependence**: GPT-2 shows near-zero absorption (0.0-0.003) where Pythia-160M shows ~0.35.
4. **Geometric dominance**: PCA SAE (0.369) matches trained SAE (0.352), showing base-model geometry explains most variance.

These findings mean **real-LLM probe-based absorption metrics cannot support causal claims about architectural components**. Any observed difference between Matryoshka and Standard could reflect geometric confounds, probe artifacts, or co-occurrence sensitivity rather than genuine architectural improvement.

SynthSAEBench-16k (Chanin & Garriga-Alonso, 2026) provides an escape hatch. With ground-truth features known by construction---128 root trees, depth 3, branching factor 4, 10,884 hierarchical features---absorption can be measured directly as the fraction of parent features subsumed by children. No probes. No AUROCs. No ceiling effects. This enables causal component isolation: varying one architectural component at a time and measuring its independent effect on absorption.

## Research Questions

1. **Component causality**: Which specific architectural component (multi-scale dictionaries, TopK sparsity, orthogonality penalties, gating) is the primary driver of absorption reduction?
2. **Sparsity vs. architecture**: Is absorption reduction primarily driven by sparsity level (L0) or by architectural innovations?
3. **Trade-off structure**: Do absorption-reducing components introduce new pathologies (e.g., hedging, reconstruction loss, dead latents)?
4. **Synthetic-to-real transfer**: Does the component ranking from SynthSAEBench transfer to real LLM SAEs?

## Hypotheses

**H1 (TopK dominance, REVISED)**: Explicit k-sparsity (TopK, k=50) is the dominant driver of absorption reduction (Cohen's d > 2.0 vs. baseline), dwarfing all other architectural components tested.

**H2 (Sparsity mediation)**: The absorption rate is primarily determined by L0 sparsity level, not by architectural choice. At matched L0, different architectures show comparable absorption rates.

**H3 (Orthogonality null)**: Decoder orthogonality penalties have negligible effect on absorption (Cohen's d < 0.5), contradicting OrtSAE's claim of 65% reduction. Absorption is encoder-driven.

**H4 (Synthetic-to-real transfer)**: The L0-absorption relationship observed on SynthSAEBench correlates with real-LLM architecture rankings (Kendall's tau > 0.6).

## Evidence-Driven Revisions

### From Pilot Results (iter_005)

The pilot experiments on SynthSAEBench-1k (1,024 features, 32 root trees) produced unexpected findings that directly revise the original hypotheses:

| Finding | Impact on Proposal |
|---------|-------------------|
| TopK reduces absorption by 83.8% (pilot) / 78.0% (full) | H1 revised: TopK (not MultiScale) is the dominant driver |
| Orthogonality reduces absorption by only 2.7% | H3 added: Orthogonality null result is a key contribution |
| L0-absorption correlation r ~ -0.99 | H2 added: Sparsity mediation hypothesis |
| MCC ~0.21-0.22 across ALL variants (including Random) | MCC dropped as primary metric; absorption rate is the discriminating metric |
| Hedging ~0.24 constant across variants | H3 (absorption-hedging trade-off) REFUTED; dropped |
| Random control absorption = 0.56 vs. TopK = 0.03 | Metric discriminates structure from randomness |

### From Full Experiment (3 variants, 5 replicates each)

| Variant | Absorption Rate | Cohen's d vs. Baseline | L0 |
|---------|----------------|----------------------|-----|
| Baseline ReLU | 0.252 ± 0.046 | --- | 964 |
| TopK (k=50) | 0.056 ± 0.021 | **5.51** | 50 |
| Orthogonality | 0.245 ± 0.050 | 0.14 | 550 |

**Key revision**: The original hypothesis predicted MultiScale dominance (H1). The data shows TopK dominance with an effect size (d = 5.51) that dwarfs all other components. The research question reframes from "which architecture reduces absorption?" to "is absorption a sparsity phenomenon masquerading as an architectural one?"

## Method

### Experiment 1: Component-Isolated Training on SynthSAEBench-16k

**Design**: Train SAE variants on SynthSAEBench-16k, varying one component at a time.

| Variant | Components | What It Tests |
|---------|-----------|---------------|
| Baseline | Standard ReLU, L1 sparsity | Baseline absorption rate |
| +TopK | Replace L1 with TopK sparsity (k=50) | Effect of explicit k-sparsity |
| +MultiScale | Nested dictionaries (2 levels) | Effect of hierarchical decomposition |
| +Orthogonality | Chunk-wise decoder orthogonality penalty | Effect of decoder incoherence |
| +Gating | Decoupled detection/magnitude paths | Effect of gating mechanism |
| +Full Matryoshka | TopK + MultiScale + hierarchical loss | Combined effect (replicates prior) |

**Metrics** (ground-truth, no probes):
- **Absorption rate**: Fraction of parent features subsumed by child features (using known ground-truth parent-child relationships)
- **Reconstruction MSE**: Standard reconstruction quality
- **Sparsity (L0)**: Average active features per token
- **Dead latent rate**: Fraction of never-active latents

**Statistical analysis**:
- One-way ANOVA across variants (5 replicates each)
- Pre-registered primary comparisons: TopK vs. Baseline, Orthogonality vs. Baseline
- Post-hoc Tukey HSD for exploratory pairwise comparisons
- Effect sizes (Cohen's d) for each component vs. baseline
- Holm-Bonferroni correction across comparisons
- Mixed-effects model: variant (fixed) + replicate (random)

**Critical control: L0-matched comparison**
To disentangle sparsity from architecture, we train L1 SAEs with tuned lambda to achieve L0 = 50 (matching TopK) and L0 = 550 (matching Orthogonality). If L0-matched Baseline achieves comparable absorption to TopK/Orthogonality, absorption is a sparsity phenomenon, not an architectural one.

### Experiment 2: Real-LLM Validation (Phase 2)

**Design**: Validate the L0-absorption relationship on real LLM SAEs.

**Procedure**:
1. Load pretrained SAEs from SAELens with varying L0 (e.g., Gemma Scope L0:22, L0:41, L0:80, L0:176, L0:445)
2. Measure first-letter absorption via SAEBench protocol
3. Test whether L0 predicts absorption across real SAEs
4. Compare with SynthSAEBench L0-absorption curve

### Experiment 3: TopK Dose-Response (k sweep)

**Design**: Vary k in {10, 25, 50, 100, 200, 500} to characterize the sparsity-absorption relationship.

**Prediction**: Absorption rate increases monotonically with k (lower sparsity -> higher absorption).

## Baselines

1. **Standard ReLU SAE** --- Expected: highest absorption, highest L0
2. **Full Matryoshka SAE** --- Expected: low absorption, but effect may be mediated by its TopK component
3. **Random-feature control** --- Expected: high absorption, validates metric discrimination
4. **L0-matched L1 SAE** --- Critical control: tests whether sparsity alone explains TopK's effect
5. **Reported scores from literature** --- Matryoshka (Bussmann et al.), OrtSAE (Korznikov et al.)

## Experimental Plan & Time Budget

| Stage | Task | Variants | Metrics | Duration |
|-------|------|----------|---------|----------|
| Full | All 6 variants on 16k (5 replicates) | 6 | Absorption, MSE, L0, dead latents | ~60 min |
| L0-matched | Baseline with tuned L1 (L0=50, 550) | 2 x 3 replicates | Absorption, MSE, L0 | ~30 min |
| k-sweep | TopK with k in {10,25,50,100,200,500} | 6 x 1 replicate | Absorption, MSE, L0 | ~30 min |
| Validation | Real-LLM SAEs from SAELens | 4-5 pretrained | SAEBench absorption | ~30 min |
| Analysis | ANOVA + L0-correlation + dose-response | All | Statistical comparison | 5 min (CPU) |

**Total GPU time**: ~1.5-2.0 hours, split into <=1-hour chunks.

## Risk Assessment

| Threat | Severity | Mitigation |
|--------|----------|------------|
| L0-matched ablation shows sparsity is sole driver | Medium | Reframe paper accordingly; still valuable finding |
| MultiScale full data overturns TopK dominance | Low | Unlikely given pilot consistency; would strengthen paper |
| Synthetic data doesn't match LLM feature structure | High | Phase 2 real-LLM validation; acknowledge limitation |
| Dead latent crisis undermines TopK recommendation | Medium | Report transparently; test absorption on active latents only |
| Reviewers dismiss as "obvious in hindsight" | Medium | Emphasize field has been attributing absorption to wrong components |

## Novelty Assessment

This would be the **first component-isolated study of SAE architectural innovations using ground-truth synthetic hierarchies**, with a surprising reframed conclusion. While Matryoshka SAEs, OrtSAE, and Gated SAEs have all reported absorption reductions, no study has:

1. **Isolated components causally** on ground-truth data where absorption can be measured directly
2. **Tested the sparsity-mediation hypothesis**: whether absorption reduction is an L0 artifact rather than an architectural achievement
3. **Reported a null result for orthogonality**: our d = 0.14 directly contradicts OrtSAE's claimed 65% reduction
4. **Demonstrated that TopK's effect (d = 5.51) dwarfs all other components tested**

**Prior work comparison**:
- Bussmann et al. (2025): Report Matryoshka absorption reduction but do not isolate multi-scale vs. TopK vs. hierarchical loss
- Korznikov et al. (2025): Report OrtSAE absorption reduction but do not test orthogonality independently; claim 65% reduction, our data shows 2.7%
- Chanin & Garriga-Alonso (2026): Introduce SynthSAEBench but do not perform component isolation
- Our prior iter_002-004: Attempted real-LLM construct validation but encountered fatal metric anomalies

## Revisions from Prior Feedback

### From Result Debate Synthesis (iter_005)

The result debate synthesis (6 perspectives: optimist, skeptic, strategist, methodologist, comparativist, revisionist) reached consensus on critical points:

1. **TopK dominance CONFIRMED**: Cohen's d = 5.51, zero overlap across replicates. This is the single most robust finding.
2. **Orthogonality null CONFIRMED**: d = 0.14, directly contradicting OrtSAE's 65% claim.
3. **L0-absorption correlation CONFIRMED**: r ~ -0.99 across variant means. Sparsity level, not architecture, appears to drive absorption.
4. **H3 (absorption-hedging trade-off) REFUTED**: Hedging ~0.24 constant across variants. No trade-off observed.
5. **H5 (MCC discriminates) REFUTED**: MCC ~0.21-0.22 across ALL variants including Random. Metric is degenerate in overcomplete settings.
6. **Unanimous PROCEED verdict**: All 6 perspectives agree to continue, with reframing around sparsity.

### From Prior Round Proposal (iter_005)

The prior round's front-runner hypothesized MultiScale dominance. This proposal is a **revision** based on empirical evidence:

| Aspect | Prior (iter_005 proposal) | Current (iter_005 post-evidence) |
|--------|--------------------------|----------------------------------|
| Core question | "Which component reduces absorption?" | "Is absorption a sparsity phenomenon?" |
| Predicted dominant | MultiScale | TopK (sparsity) |
| H2 ranking | MultiScale > Orthogonality > TopK > Gating | TopK >> all others (d = 5.51) |
| H3 trade-off | Absorption-hedging | REFUTED; no trade-off observed |
| Key control | Random-feature control | L0-matched comparison (critical) |

## Constructive Proposal

Beyond the core component-isolation results, this paper contributes:

1. **Methodological clarification**: Redirects the field's optimization target from architectural complexity (multi-scale, orthogonality, gating) to sparsity control. The operative variable is L0, not architecture.

2. **Null result for orthogonality**: Our d = 0.14 for orthogonality directly contradicts OrtSAE's 65% claim. This is a valuable negative result that prevents misallocation of research effort.

3. **Synthetic validation protocol**: A reproducible protocol for testing SAE architectural claims on ground-truth data before deploying on real LLMs. This addresses the community's measurement-validity crisis.

4. **L0-matched control template**: The experimental design (matching L0 across architectures) can be applied to future SAE evaluations, establishing a methodological standard.

## Expected Contributions

1. **First component-isolated causal analysis** of SAE architectural innovations for absorption reduction on ground-truth data.
2. **First evidence that absorption is primarily a sparsity-level phenomenon**, with TopK's effect (d = 5.51) dwarfing orthogonality (d = 0.14).
3. **First null result for orthogonality penalties** on absorption, directly contradicting a major prior claim (OrtSAE).
4. **First test of synthetic-to-real L0-absorption transfer** for SAE evaluation.
5. **Practical guidance** for practitioners: focus on sparsity level, not architectural complexity, for absorption control.

## Synthesis Rationale

### How the Perspectives Were Weighted

**Highest weight: Empiricist + Result Debate Synthesis.** The empiricist provided the core methodological framework: ground-truth synthetic data, component-isolated design, ANOVA with effect sizes, and falsification criteria. The result debate synthesis (6 perspectives analyzing actual experimental data) provided the decisive evidence: TopK dominance (d = 5.51), orthogonality null (d = 0.14), and L0 correlation (r ~ -0.99).

**Strong weight: Contrarian + Skeptic (from result debate).** The contrarian's core insight---that real-LLM absorption metrics are confounded---motivated the pivot to synthetic data. The skeptic (from result debate) correctly identified the L0 confound as the central unresolved question, leading to the L0-matched comparison as a critical control.

**Moderate weight: Theoretical + Pragmatist.** The theoretical's sparsity-control theory (L1 shrinkage bias, activation competition) provides a mechanistic explanation for why TopK dominates. The pragmatist's implementation path ensures all experiments fit within the time budget.

**Lower weight: Innovator + Interdisciplinary.** The innovator's absorption-dark matter duality and the interdisciplinary's lateral inhibition SAE are promising future directions but require additional experiments beyond current constraints. Retained as backup ideas.

### Why This Reframing Was Selected

The synthesis of empiricist + skeptic was selected because:

1. **The data demands it**: TopK's effect size (d = 5.51) is an order of magnitude larger than orthogonality (d = 0.14). Ignoring this asymmetry would be scientific malpractice.

2. **It answers a more important question**: Not "which architecture?" but "what is the operative variable?" The latter has greater practical impact.

3. **It produces actionable results regardless of L0-matched outcome**:
   - If L0-matched Baseline matches TopK: "Absorption is a sparsity phenomenon" --- still valuable
   - If L0-matched Baseline does NOT match TopK: "TopK has an additional architectural benefit beyond sparsity" --- also valuable

4. **It challenges field assumptions constructively**: The community has been attributing absorption reduction to multi-scale decomposition and orthogonality. Our data says: check your L0 first.

5. **It has higher venue potential**: A paper that reframes a central phenomenon (absorption) around a simpler variable (sparsity) targets NeurIPS/ICML/ICLR main.

### What Was Dropped or Deferred

- **MultiScale dominance hypothesis** (original H1): Refuted by data. TopK dominates.
- **Absorption-hedging trade-off** (original H3): Refuted by data. Hedging is invariant.
- **MCC as primary metric**: Degenerate in overcomplete settings. Absorption rate is the discriminating metric.
- **Goodhart's Law framing** (contrarian front-runner): Narrowed to "measurement confounds motivate ground-truth validation."
- **Dark matter duality** (innovator): Promising but requires additional experiments. Deferred.
- **Lateral inhibition SAE** (interdisciplinary): Requires new architecture training. Deferred to follow-up.
- **Rate-distortion bound** (theoretical): Theoretically compelling but bound is vacuous for practical parameters. Retained as conceptual framing.
