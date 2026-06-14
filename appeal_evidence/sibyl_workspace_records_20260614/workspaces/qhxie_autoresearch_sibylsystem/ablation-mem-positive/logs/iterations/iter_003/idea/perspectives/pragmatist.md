# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. [SAELens (jbloomAus/SAELens)](https://github.com/jbloomAus/SAELens) — 1,354 stars, MIT license, active. Primary library for training and analyzing SAEs; direct pretrained SAE loading via `SAE.from_pretrained`. **Use as core framework.**

2. [SAEBench (adamkarvonen/SAEBench)](https://github.com/adamkarvonen/SAEBench) — 162 stars, 8-metric evaluation suite including probe projection absorption metric that works across all layers. **Use for standardized evaluation.**

3. [sae-spelling (lasr-spelling/sae-spelling)](https://github.com/lasr-spelling/sae-spelling) — 14 stars, original absorption detection metric implementation. **Extend for custom probe tasks.**

4. [MP-SAE (mpsae/MP-SAE)](https://github.com/mpsae/MP-SAE) — Matching Pursuit SAE; conditional orthogonality reduces absorption. **Alternative approach if current methods fail.**

5. [GemmaScope (google/gemma-scope)](https://github.com/google/gemma-scope) — Comprehensive pretrained SAEs for Gemma-2-2B/9B, all layers, JumpReLU architecture. **Cross-architecture validation.**

6. [Chanin et al., 2024. A is for Absorption. arXiv:2409.14507](https://arxiv.org/abs/2409.14507) — First systematic study; ablation-based detection limited to layers 0-17. **Foundation, not sole reference.**

7. [Basu et al., 2026. Interpretability without Actionability. arXiv:2603.18353](https://arxiv.org/abs/2603.18353) — 98.2% AUROC but 0% steering effect; clinical domain. **Critical negative result; frames actionability question.**

8. [Cui et al., ICLR 2026. On the Limits of SAEs. arXiv:2506.15963](https://arxiv.org/abs/2506.15963) — Theoretical limits: full disentanglement mathematically impossible. **Cited as theoretical foundation.**

### Landscape Summary

The feature absorption literature is rich but fragmented. Key findings:
- Absorption is real and validated (67.3% mean recovery in activation patching)
- The "actionability paradox" (Basu et al.) shows detection != steering utility
- Theoretical limits are established (Cui et al.); absorption is mathematically unavoidable
- Pilot data shows CV predicts steering effectiveness (2x difference, high-CV vs low-CV)
- Cross-layer analysis at λ=0.001 shows saturation (all layers = 1.0); need finer λ measurement

**Practical gap**: No one has systematically tested whether absorbed features are uniformly unsteerable or if some subpopulation remains steerable. The Basu et al. finding is in clinical domain; non-clinical LLM domain may differ.

### GitHub Implementation Reality

- SAELens is the de facto standard; extensive pretrained SAE support
- SAEBench is the standard evaluation; absorption metric implementation exists
- sae-spelling has the original metric; can be extended to new probe tasks
- All code is available; no missing infrastructure

---

## Phase 2: Initial Candidates

### Candidate A: CV-Based Absorption Decomposition (Front-Runner)

- **Hypothesis**: Absorbed features split into "robust" (high-CV, steerable) and "fragile" (low-CV, non-steerable) subpopulations. Pilot data supports: 0.153 vs 0.075 steering effect.
- **Implementation sketch**: Use SAELens to load GPT-2/Gemma SAEs, classify absorbed features by CV, run steering experiments on 30+30 features.
- **Simplest version**: Single experiment: 30 high-CV vs 30 low-CV absorbed features, measure steering effect. Fits in 30 minutes.
- **Time estimate**: 2-3 hours total for full validation (CV classification, steering, mechanism investigation, cross-model)
- **Reusable components**: SAELens pretrained SAEs, existing steering infrastructure, SAEBench evaluation code

### Candidate B: Projection-Based Cross-Layer Quantification (Backup)

- **Hypothesis**: Use SAEBench probe projection metric (works across all layers) to measure absorption at λ_c=5e-5 where layers may show heterogeneity.
- **Implementation sketch**: SAEBench already has probe projection metric; apply to layers 0,3,6,9,11 at critical sparsity.
- **Simplest version**: 5 layers × 1 sparsity = quick measurement, ~30 minutes.
- **Time estimate**: 1-2 hours for cross-layer validation
- **Reusable components**: SAEBench probe projection implementation, GemmaScope SAEs

### Candidate C: Cross-Architecture Phase Transition Validation (Longer-Term)

- **Hypothesis**: ν=3 finite-size scaling discovered on GPT-2 TopK SAEs may generalize to Gemma-2-2B JumpReLU SAEs.
- **Implementation sketch**: Run sparsity sweep on both architectures; compare critical exponents.
- **Simplest version**: Replicate GPT-2 sparsity sweep, then run Gemma-2 sweep (2× the time)
- **Time estimate**: 4-5 hours total; exceeds 1-hour guideline per task but fits project budget

---

## Phase 3: Self-Critique

### Against Candidate A: CV-Based Absorption Decomposition

- **Implementation reality check**: 
  - Has anyone tested CV-steering correlation? Search shows no prior work. Novel territory.
  - Practical issue: CV computation requires sufficient samples (1000+) for stable estimates
  - Steering infrastructure already exists (based on pilot_steering_cv passing)
  
- **Reproducibility attack**:
  - The 2x difference (0.153 vs 0.075) needs larger n for confirmation
  - CV threshold (1.0) was chosen post-hoc; may not generalize
  - Need to validate on held-out features, not the same features used to identify the correlation
  
- **Baseline sanity check**:
  - What is the steering effect for NON-absorbed features? Need this comparison for context
  - If non-absorbed features show similar steering effects, the "robust absorbed" story weakens
  - Baseline steering for GPT-2 layer 6: need to establish what "normal" looks like
  
- **Scope attack**:
  - Pilot used only GPT-2; Gemma-2 validation is critical for generalization
  - Clinical domain (Basu et al.) may be fundamentally different from LLM domain
  - One SAE layer (6) is not representative; need multi-layer validation
  
- **Verdict**: MODERATE — The CV-steering finding is genuine and novel, but needs rigorous validation on held-out data and multiple SAE layers. The steering effect size (0.078 difference) is measurable but small.

### Against Candidate B: Projection-Based Cross-Layer Quantification

- **Implementation reality check**:
  - SAEBench probe projection metric is already implemented; minimal implementation effort
  - Works across all layers (unlike ablation limited to early layers)
  - No new code required; just parameterize existing infrastructure
  
- **Reproducibility attack**:
  - If no cross-layer variation is found at λ_c=5e-5, what does that mean?
  - Could be: (a) layer heterogeneity doesn't exist, or (b) probe projection metric isn't sensitive enough
  - Negative result is publishable but less exciting
  
- **Baseline sanity check**:
  - Compare to Chanin et al.'s finding (layers 0-17 only); does probe projection agree in overlap region?
  - Need ground-truth absorption ground-truth to validate the metric
  
- **Scope attack**:
  - Only tests one sparsity value (λ_c); what about other λ values?
  - If all layers saturate at λ_c as well, the approach fails entirely
  
- **Verdict**: MODERATE — Clean, standard methodology. Publishable in either direction (variation found or not). However, if layers still saturate at λ_c, this backup also fails.

### Against Candidate C: Cross-Architecture Phase Transition Validation

- **Implementation reality check**:
  - GPT-2 sparsity sweep took ~45 min; Gemma-2 sweep will be similar
  - Total runtime 4-5 hours exceeds "1 hour per task" guideline
  - But the project allows longer experiments; this is within budget
  
- **Reproducibility attack**:
  - If ν differs significantly between architectures, is that a failure or a genuine finding?
  - R²=0.951 for GPT-2 may not replicate for Gemma-2
  - Need to report architectural variation as finding, not just "failed to replicate"
  
- **Baseline sanity check**:
  - GemmaScope uses JumpReLU (different from GPT-2's TopK)
  - Architecture difference may confound the comparison
  - Need to account for architectural variation in analysis
  
- **Scope attack**:
  - Only tests two architectures; need more for universal claims
  - LlamaScope available for third architecture validation if needed
  
- **Verdict**: STRONG — This is the most publishable backup because finding is significant regardless of outcome:
  - If ν=3 generalizes: confirm theoretical framework
  - If ν differs: first evidence of architectural dependence, publishable as-is

---

## Phase 4: Refinement

### Dropped Ideas

- **Long theoretical extension (Candidate B from innovator)**: Information-theoretic decomposition dropped because computing mutual information for thousands of feature pairs is not tractable. Too heavy for available compute.

- **Temporal absorption dynamics**: Operationally undefined. Would require new metric development. Not feasible within project timeline.

### Strengthened Ideas

- **Candidate A (CV-Based Decomposition)**: 
  - Strengthened by pilot_steering_cv (0.153 vs 0.075, 2x difference)
  - Activation patching confirms genuine absorption (67.3% recovery)
  - Direct connection to actionability paradox (the field's key question)
  
- **Candidate B (Projection-Based)**: 
  - Clean methodology using standard SAEBench metric
  - No new code required
  - Publishable negative result if layers still saturate

### Additional Pragmatist Considerations

1. **What can we actually ship in 1-hour budget?**
   - CV classification: 15 min
   - Steering on 30+30 features: 30 min
   - Mechanism investigation: 20 min
   - Cross-model validation: 30 min (Gemma-2)
   - **Total: ~95 min** — fits within project time budget

2. **What existing code can we reuse?**
   - SAELens for SAE loading and activation extraction
   - SAEBench probe projection for cross-layer metrics
   - Existing steering infrastructure (from pilot_steering_cv)
   - No new infrastructure development needed

3. **What could go wrong in implementation?**
   - Steering infrastructure may not generalize from pilot to full
   - CV threshold (1.0) may be wrong; need to tune
   - Gemma-2 may show different CV distribution than GPT-2

4. **What is the simplest possible publishable result?**
   - **Minimum**: 30 high-CV vs 30 low-CV steering comparison on GPT-2 layer 6
   - **If positive**: CV predicts steering → submit to AAAI/EMNLP
   - **If negative**: CV does not predict steering → Basu et al. confirmed for LLM domain → submit negative result

### Selected Front-Runner

**Candidate A: CV-Based Absorption Decomposition** is the pragmatist's choice because:

1. **Implementation path is clear**: Use SAELens + existing steering code; no new infrastructure
2. **Time fits budget**: ~95 min total across all validation experiments
3. **Publishable in either direction**: Positive (CV predicts steering) or negative (confirms Basu et al. for LLM domain)
4. **Directly addresses field's key question**: Actionability of absorbed features
5. **Novel finding if positive**: First evidence that absorbed features are not uniformly non-steerable (in non-clinical domain)
6. **Low engineering risk**: Pilot already validated the steering infrastructure

---

## Phase 5: Final Proposal

### Title

**CV Predicts Steering Heterogeneity Within Absorbed SAE Features: Evidence from GPT-2 and Gemma-2 SAEs**

### Hypothesis

Feature absorption creates steering heterogeneity: some absorbed features (high-CV, CV > 1.0) remain steerable while others (low-CV, CV ≤ 1.0) do not. CV (coefficient of variation) is a simple statistical predictor of steering effectiveness. This challenges the universal actionability failure reported by Basu et al. (2026) in clinical domain and suggests the paradox may be domain-specific or feature-type-specific.

### Motivation

Basu et al. (2026) demonstrated 98.2% feature detection AUROC but 0% steering utility — the "actionability paradox." If absorbed features are uniformly non-steerable, SAE-based interpretability has limited practical value. However, their result is in clinical domain; non-clinical LLM domain may differ. Our pilot data (0.153 vs 0.075 steering effect for high-CV vs low-CV) suggests absorbed features are not uniformly non-steerable. Identifying which absorbed features remain steerable would provide actionable guidance for interpretability practitioners.

### Method

**Step 1: Feature Classification (15 min)**
- Load GPT-2 layer 6 SAE via SAELens
- Compute per-feature CV across 1000 text samples at λ=5e-5 (critical sparsity)
- Classify absorbed features (absorption_score > 0.5) into high-CV (CV > 1.0) and low-CV (CV ≤ 1.0)
- Aim for 30+ features in each group

**Step 2: Steering Effectiveness Test (30 min)**
- Run steering experiments on 30 high-CV and 30 low-CV absorbed features
- Steering strength: +3, +5, +7
- Metric: logit change at semantically appropriate tokens
- Compare mean steering effect between groups

**Step 3: Mechanism Investigation (20 min)**
- Test whether CV-steering correlation is explained by:
  - Decoder orthogonality (robust features have more orthogonal decoders)
  - Feature frequency (robust features are rarer but more specialized)
  - Context sensitivity (robust features activate in narrower distributions)

**Step 4: Cross-Model Validation (30 min)**
- Replicate on Gemma-2-2B layer 6 JumpReLU SAE
- Test whether CV threshold (1.0) generalizes or model-specific

**Step 5: Baseline Comparison (15 min)**
- Compare steering effects for absorbed features vs non-absorbed features
- Establishes whether "robust absorbed" is actually comparable to non-absorbed or still degraded

### Simplest Version

Single experiment: 30 high-CV vs 30 low-CV absorbed features on GPT-2 layer 6. Steering at +5 strength. Logit change measurement. ~30 min runtime.

**Expected outcome**:
- Positive: High-CV shows larger steering effect → CV predicts steering heterogeneity
- Negative: No difference → Basu et al. actionability paradox may be universal

### Baselines

1. **Non-absorbed features**: Expected steering effect ~0.2-0.3 (established from prior work)
2. **All absorbed features (Basu et al.)**: Expected steering effect ~0 (universal failure)
3. **High-CV absorbed (our hypothesis)**: Expected steering effect >0.1 (some steering retained)
4. **Low-CV absorbed (our hypothesis)**: Expected steering effect ~0.05 (near-baseline failure)

### Experimental Plan

| Experiment | Duration | Validates |
|-----------|----------|-----------|
| E1: CV classification | 15 min | High/low CV split on absorbed features |
| E2: Steering comparison | 30 min | Robust vs fragile hypothesis |
| E3: Mechanism analysis | 20 min | Orthogonality, frequency, context |
| E4: Gemma-2 validation | 30 min | Cross-model generalization |
| E5: Non-absorbed baseline | 15 min | Context for interpreting absorbed results |

**Total: ~110 min** (within project budget)

### Resource Estimate

- **Models**: GPT-2-small (86M params, fast), Gemma-2-2B (2B params, slower but acceptable)
- **SAEs**: GPT-2 layer 6 residual stream (~16k latents), GemmaScope layer 6
- **Compute**: ~2 GPU hours total
- **No new training**: Training-free analysis of pretrained SAEs via SAELens

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| CV threshold (1.0) wrong | Medium | Tune on held-out; report as exploratory if not predictive |
| Steering effect too small | Medium | Compare to non-absorbed baseline; report normalized effect |
| Gemma-2 shows no CV effect | Medium | Report as negative result; Basu et al. confirmed for LLM domain |
| Steering infrastructure fails on Gemma | Low | Use GemmaScope directly via SAELens; similar API |

### Novelty Claim

**First evidence that absorbed SAE features are not uniformly non-steerable in non-clinical LLM domain.** The CV-based decomposition provides a simple, computationally cheap predictor (CV) of steering effectiveness within absorbed features. This advances the field by:
1. Showing the Basu et al. actionability paradox may not be universal
2. Providing a simple metric (CV) for practitioners to assess steering feasibility
3. Establishing heterogeneity within absorbed features as a research direction

**Prior work**: Basu et al. (clinical domain, uniform failure). Our contribution: demonstrating heterogeneity in non-clinical LLM domain and identifying CV as predictor.

**What is NOT novel**: Phase transitions, finite-size scaling, general absorption phenomenon — these are established in the literature.

### Practical Notes for Implementation

1. **Use SAELens** (`SAE.from_pretrained`) for all SAE loading — standard API, extensive documentation
2. **Use existing steering code** from pilot_steering_cv — validated in pilot
3. **CV computation**: scipy.stats.variation (coefficient of variation) or manual computation
4. **Absorption score**: From sae-spelling or SAEBench probe projection
5. **Steering metric**: Logit change at top-k activating tokens per feature (standard approach)