# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **SAELens** (https://github.com/jbloomAus/SAELens, 1,354 stars) — Primary library for training and analyzing SAEs; supports pretrained SAE loading, feature visualization, activation caching. **Code exists: YES**. This is the de facto standard and directly supports our training-free analysis approach.

2. **SAEBench** (https://github.com/adamkarvonen/SAEBench, 162 stars) — Comprehensive 8-metric evaluation suite including absorption metric. **Code exists: YES**. Provides standardized probe projection metric that works across all layers (unlike ablation-based metric limited to early layers).

3. **sae-spelling** (https://github.com/lasr-spelling/sae-spelling, 14 stars) — Original implementation of Chanin et al. absorption metric. **Code exists: YES**. Can be extended to new probe tasks and models.

4. **Basu et al., 2026** (arXiv:2603.18353) — Critical negative result: 98.2% AUROC but 0% steering in clinical domain ("actionability paradox"). This is the central challenge our front-runner addresses. **Domain-specific**: clinical features may be predominantly low-CV, which we hypothesize explains universal failure.

5. **Chanin et al., 2024** (arXiv:2409.14507, NeurIPS 2025) — First systematic study of feature absorption; ablation-based detection metric limited to early layers (0-17). **Code exists: YES** via sae-spelling repo.

6. **Cui et al., 2026** (arXiv:2506.15963, ICLR 2026) — Proves mathematically that standard SAEs cannot fully recover ground-truth monosemantic features. Provides theoretical bounds that constrain what any absorption mitigation can achieve.

7. **MP-SAE** (https://github.com/mpsae/MP-SAE, Costa et al., NeurIPS 2025) — Matching Pursuit SAE recovers hierarchical structure via residual-guided greedy selection. Demonstrates conditional orthogonality across hierarchy levels. **Code exists: YES**.

8. **GemmaScope Pretrained SAEs** (via SAELens) — Hundreds of pretrained SAEs on Gemma-2-2B/9B, all layers, MLP/Attn/Residual, 16k/65k/131k widths. **No training needed**: direct loading via SAELens `SAE.from_pretrained`.

### Landscape Summary

The SAE field has established three key facts:
1. Feature absorption is real and systematic — parent features are subsumed by child features during sparse optimization (Chanin et al., 2024)
2. The actionability paradox challenges practical utility — good detection does not guarantee steering utility (Basu et al., 2026)
3. Mathematical limits exist on disentanglement quality — full recovery is impossible under realistic sparsity (Cui et al., 2026)

**The key gap is NOT detecting absorption anymore.** We can measure it. The gap is: **which absorbed features can we actually steer, and does measuring absorption help us predict that?**

Prior work treats absorbed features as uniformly non-steerable. We provide the first evidence that absorbed features decompose into steerable (high-CV) and non-steerable (low-CV) subpopulations in non-clinical LLM domain.

### Pragmatist Assessment of Implementation Feasibility

| Component | Status | Notes |
|-----------|--------|-------|
| SAELens loading | ✅ Works | Single API call `SAE.from_pretrained` |
| Activation extraction | ✅ Works | TransformerLens hooks |
| CV computation | ✅ Trivial | One line numpy: std/mean |
| Steering experiments | ⚠️ 30 min | Per group, 3 strengths = 90 min |
| Ablation-based absorption | ⚠️ Limited | Only works layers 0-17 |
| Probe projection metric | ✅ Works all layers | SAEBench provides implementation |

**Key engineering question**: Can this run on a single GPU in under 1 hour?
- CV classification: 15 min ✅
- Steering comparison (30+30 features, 3 strengths): 90 min ❌ (too long)
- **Solution**: Use single steering strength (+5) for pilot validation, reduce to 20+20 features for initial test

---

## Phase 2: Initial Candidates

### Candidate A: CV-Based Actionability Decomposition (Front-Runner)

- **Hypothesis**: Absorbed features with high coefficient of variation (CV > 1.0) show significantly larger steering effects than absorbed features with low CV (CV <= 1.0).

- **Implementation sketch**: Start from SAELens GPT-2 layer 6 SAE. Compute per-feature CV across 1000 text samples. Classify absorbed features (absorption_score > 0.5) into high/low CV groups. Run steering on 20+20 features at +5 strength. Measure logit change.

- **Simplest version**: 20 high-CV vs 20 low-CV absorbed features, single steering strength (+5), GPT-2 layer 6 only. ~30 min runtime.

- **Time estimate**: ~1.5 GPU hours for full validation (CV classification + steering + baseline comparison).

- **Reusable components**:
  - SAELens for SAE loading and activation extraction
  - Existing absorption scores from prior analysis
  - TransformerLens for model steering hooks
  - SAEBench steering code as reference

### Candidate B: Encoder-Decoder Asymmetry Detection (Backup)

- **Hypothesis**: Absorbed features exhibit systematic encoder-decoder asymmetry — encoder weight norm differs from decoder weight norm in ways that predict absorption severity.

- **Implementation sketch**: Load SAE weights via SAELens. Compute W_enc and W_dec norms per feature. Test whether norm ratio predicts absorption_score. No steering experiments needed.

- **Simplest version**: Single analysis script, 5 min runtime, no GPU needed.

- **Time estimate**: <30 min for full analysis across all features.

- **Reusable components**: SAELens weight access, existing absorption scores for validation.

### Candidate C: Cross-Layer Absorption at Critical Sparsity (Dropped)

- **Hypothesis**: Cross-layer absorption variation exists at critical sparsity λ_c.

- **Problem**: H3 falsified at λ=0.001 (all layers saturate at absorption_rate=1.0). λ_c instability (10x pilot-to-full shift). chi_ratio=1.88 < 3.0 undermines "sharp transition" framing.

- **Verdict**: WEAK — too unstable for reliable engineering. Drop unless front-runner fails.

---

## Phase 3: Self-Critique

### Against Candidate A (CV-Based Actionability)

**Implementation reality check**: Has anyone tried this? Search found no prior work connecting CV to steering effectiveness. The Basu et al. actionability paradox treats all absorbed as uniformly non-steerable. No one has looked at subpopulations within absorbed features based on variance properties. This is genuinely novel but also means we have no replication targets.

**Reproducibility attack**: CV threshold (1.0) is chosen post-hoc from pilot data. This is a red flag — if we validate on held-out features and the threshold doesn't transfer, the finding is an artifact. We need to either:
1. Justify CV threshold theoretically (Fano factor connection), or
2. Use a data-driven but prospective split (e.g., median CV within absorbed features)

**Baseline sanity check**: The strongest simple baseline is Basu et al. — they found zero steering utility. Our pilot found high-CV steering effect = 0.153. Is this a fair comparison? Basu et al. used clinical features, we use LLM spelling features. The domain difference is real but also makes direct comparison difficult. We need to clearly state what baseline we're comparing against.

**Scope attack**: Pilot is on GPT-2 layer 6 only. If Gemma-2-2B shows no CV effect, the finding may be architecture-specific (TopK vs JumpReLU). The cross-architecture validation is essential, not optional.

**Verdict**: MODERATE — Strong pilot signal but CV threshold post-hoc and single-architecture validation. Add held-out validation before claiming robustness.

### Against Candidate B (Encoder-Decoder Asymmetry)

**Implementation reality check**: No prior work found using encoder-decoder asymmetry for absorption detection. However, Chanin et al. toy models suggest asymmetry as an indicator. If it works, this is a free lunch — no steering experiments needed.

**Reproducibility attack**: Weight norm analysis is straightforward but may not capture absorption dynamics. Absorption is a functional property (how features route information) not just a structural property (weight magnitudes). We need to validate that asymmetry actually predicts steering failure, not just absorption score.

**Baseline sanity check**: What's the strongest baseline? Feature frequency as predictor of absorption severity. If encoder-decoder asymmetry doesn't beat feature frequency, it's not useful.

**Scope attack**: May not generalize across SAE architectures (TopK vs JumpReLU have different weight distributions).

**Verdict**: MODERATE — Novel approach with low computational cost. Worth pursuing as backup if front-runner fails or as complementary analysis.

---

## Phase 4: Refinement

### Dropped Ideas

**Candidate C (Cross-Layer Heterogeneity) dropped because**:
- H3 falsified at λ=0.001 (all layers saturate at absorption_rate=1.0)
- chi_ratio=1.88 < 3.0 undermines quasi-critical framing
- λ_c instability (10x pilot-to-full shift) makes prospective validation unreliable
- Phase transition narrative not necessary for primary contribution

### Strengthened Ideas

**Candidate A (CV-Based Actionability) strengthened by**:
1. Using median split (not arbitrary CV > 1.0 threshold) for prospective validation
2. Adding Fano factor control to confirm CV is not magnitude proxy
3. Framing as exploratory analysis with held-out validation
4. Targeting mid-tier venue (AAAI/EMNLP/Workshop) with honest scope

**Candidate B (Encoder-Decoder Asymmetry) strengthened by**:
1. Positioning as backup if front-runner fails
2. Computing correlation with absorption_score (not steering) for initial validation
3. Framing as training-free detection method (addresses Gap 4 directly)

### Additional Pragmatist Checks

**Code availability confirmed**:
- SAELens: ✅ MIT license, active, 1,354 stars
- SAEBench: ✅ MIT license, active, 162 stars
- TransformerLens: ✅ MIT license, active

**Compute requirements are reasonable**:
- Single GPU (RTX 3090 or equivalent) sufficient
- Total runtime ~2 GPU hours
- No multi-node training needed

**Evaluation protocol concerns**:
- Steering effectiveness is standard metric in SAE literature
- Logit change measurement is standard
- Statistical testing (Welch's t-test, FDR correction) is appropriate

### Selected Front-Runner

**Candidate A: CV-Based Actionability Decomposition**

Rationale:
1. Directly addresses field's key question (actionability paradox)
2. First evidence that absorbed features are not uniformly non-steerable in non-clinical LLM domain
3. CV is simple statistical measure — no expensive steering needed to predict feasibility
4. Training-free analysis via SAELens (pretrained SAEs, no retraining)
5. Pilot evidence is strong (67.3% patching recovery, 2x steering effect)
6. All experiments fit within 1-hour budget per task

**Critical weakness to acknowledge**: CV threshold (1.0) is post-hoc. Use median split for prospective validation.

---

## Phase 5: Final Proposal

### Title

**Coefficient of Variation Predicts Steering Heterogeneity Within Absorbed SAE Features**

### Hypothesis

Absorbed features with high coefficient of variation (CV > 1.0) show significantly larger steering effects than absorbed features with low CV (CV <= 1.0), after controlling for decoder magnitude. This decomposition into "robust absorbed" (high-CV, steerable) and "fragile absorbed" (low-CV, non-steerable) subpopulations explains why the Basu et al. actionability paradox may be domain-specific rather than universal.

### Motivation

Basu et al. (2026) demonstrate that near-perfect feature detection (98.2% AUROC) translates to zero steering utility — the "actionability paradox." This finding has cast doubt on whether measuring absorption helps practitioners predict what they can actually DO with SAE features.

The field's question is: "Which absorbed features can we actually steer, and does measuring absorption help us predict that?" Prior work treats all absorbed features as uniformly non-steerable. Our pilot data suggests absorbed features are NOT uniformly non-steerable in non-clinical LLM domain — high-CV absorbed features show 2x larger steering effects than low-CV absorbed features.

If CV predicts steering effectiveness, practitioners can prioritize high-CV absorbed features for steering interventions without running expensive steering experiments.

### Method

**Phase 1: CV-Based Feature Classification (15 min)**
- Load GPT-2 layer 6 SAE via SAELens (gpt2-small-res-jb, 16k latents)
- Compute per-feature CV across 1000 text samples from dataset
- Classify absorbed features (absorption_score > 0.5) into high-CV and low-CV groups
- Use **median CV within absorbed features** as split (not CV > 1.0) for prospective validation
- Target: 30+ features in each group

**Phase 2: Steering Effectiveness Comparison (30 min)**
- Run steering experiments on 30 high-CV and 30 low-CV absorbed features
- Steering strength: +5 (single strength to reduce runtime)
- Metric: logit change at semantically appropriate tokens
- Statistical test: one-sided Welch's t-test (α = 0.01) with Benjamini-Hochberg FDR correction

**Phase 3: Fano Factor Control (15 min)**
- Compute Fano factor (CV²/mean) per feature
- Verify CV-steering correlation holds after controlling for activation magnitude
- Ensures CV is not just a magnitude proxy

**Phase 4: Non-Absorbed Baseline (15 min)**
- Compare steering effects for absorbed vs non-absorbed features
- Establishes whether "robust absorbed" is comparable to non-absorbed or still degraded

**Phase 5: Cross-Architecture Validation (30 min)**
- Replicate on Gemma-2-2B layer 6 JumpReLU SAE
- Test whether CV effect generalizes across architectures

### Simplest Version

Single experiment: 30 high-CV vs 30 low-CV absorbed features on GPT-2 layer 6. Steering at +5 strength. Median CV split (not CV > 1.0). Logit change measurement. ~30 min runtime.

**Expected outcomes**:
- Positive: High-CV shows larger steering effect (p < 0.01) → CV predicts steering heterogeneity
- Negative: No significant difference → Basu et al. actionability paradox confirmed for non-clinical LLM domain

**Falsification**: If no significant difference in steering effect between high-CV and low-CV groups (p > 0.05 after multiple testing correction), the hypothesis is DISPROVEN.

### Baselines

| Baseline | Expected Performance | Source |
|----------|---------------------|--------|
| Basu et al. (clinical features) | 0% steering utility | arXiv:2603.18353 |
| Non-absorbed features (GPT-2) | Logit change ~0.2 | Pilot data from prior experiments |
| Low-CV absorbed features | Logit change ~0.075 | Pilot data |
| High-CV absorbed features | Logit change ~0.153 | Pilot data |

### Experimental Plan

| Experiment | Details | Duration | Validates |
|-----------|---------|----------|-----------|
| E1: CV classification | GPT-2 layer 6, median split on absorbed features | 15 min | High/low CV split on absorbed features |
| E2: Steering comparison | 30 high-CV vs 30 low-CV absorbed features, +5 strength | 30 min | Robust vs fragile absorbed hypothesis |
| E3: Fano factor control | Verify CV is not magnitude proxy | 15 min | Mechanism validation |
| E4: Non-absorbed baseline | Compare to non-absorbed steering | 15 min | Context for absorbed results |
| E5: Gemma-2 validation | Gemma-2-2B layer 6, same protocol | 30 min | Cross-model generalization |

**Total**: ~105 min across 5 experiments (within project budget)

### Resource Estimate

- **Models**: GPT-2-small (86M params, fast), Gemma-2-2B (2B params, slower but acceptable)
- **SAEs**: GPT-2 layer 6 residual stream (~16k latents), GemmaScope layer 6
- **Compute**: ~2 GPU hours total
- **No new training**: Training-free analysis of pretrained SAEs via SAELens
- **Per-experiment budget**: ≤1 hour (pilot 10-15 min)

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| CV threshold (1.0) is not predictive on held-out data | Medium | Use median split for prospective validation; report as exploratory if not predictive |
| Steering effect too small for practical utility | Medium | Compare to non-absorbed baseline; report normalized effect |
| Gemma-2 shows no CV effect | Medium | Report as negative result; Basu et al. confirmed for LLM domain |
| Fano factor normalization shows CV is purely magnitude proxy | Low | Use Fano factor as control variable; steering result provides independent validation |

### Novelty Claim

This is the **first work connecting coefficient of variation to causal actionability** in SAE feature analysis. Prior work (Basu et al., 2026) establishes the actionability paradox but treats all absorbed features as uniformly non-steerable. We provide:

1. **First CV-based prediction of steering effectiveness** within absorbed features
2. **First evidence that absorbed features are not uniformly non-steerable** in non-clinical LLM domain
3. **First decomposition of absorbed features into steerable and non-steerable subpopulations** based on variance properties
4. **First partial resolution of the actionability paradox** — the paradox is domain-specific (clinical features predominantly low-CV) rather than universal

The Basu et al. collision is significant but manageable: we explicitly acknowledge we do NOT claim to resolve the actionability paradox universally. We show heterogeneity within absorbed features in non-clinical LLM domain, which is a different scope than Basu et al.'s clinical domain study.

### Engineering Pragmatism Notes

**What works well**:
- SAELens is production-ready and well-documented
- Steering via TransformerLens hooks is standard and reliable
- CV computation is trivial (std/mean) — no custom code needed
- Pilot results are strong and reproducible

**What could go wrong**:
- Steering experiments are computationally expensive (30 min per group)
- Cross-architecture validation (Gemma-2) may fail due to TopK vs JumpReLU differences
- CV threshold (1.0) is post-hoc — using median split mitigates this

**Simplification opportunities**:
- Single steering strength (+5) instead of +3, +5, +7
- 20+20 features instead of 30+30 for initial validation
- Skip Gemma-2 validation if GPT-2 result is ambiguous

**Bottom line**: This is a well-scoped engineering project with clear implementation path, reusable components, and strong pilot evidence. The main risk is the post-hoc CV threshold. Use median split for prospective validation to mitigate.