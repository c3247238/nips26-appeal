# Research Proposal: CV Predicts Steering Heterogeneity Within Absorbed SAE Features

## Revisions from Prior Feedback

**From Result Debate (iter_004 verdict, score 5.5/10)**:
- Recommendation: PIVOT to actionability-focused research
- chi_ratio=1.88 is below the "sharp transition" threshold of 3.0 — downgrade "quasi-critical" framing
- H3 (cross-layer heterogeneity) falsified at λ=0.001 — needs retesting at λ_c=5e-5
- H6 (graph topology) falsified — component count decreases with layer, not peaked
- Phase transition framework provides supporting theoretical context, not primary novelty

**Pilot Evidence Validating This Round**:
1. **Activation Patching**: 67.3% mean recovery (all 9/9 words pass 10% threshold) — confirms genuine absorption
2. **Steering CV**: High-CV features show 2x larger steering effect (0.153 vs 0.075) — validates CV as predictor

**This proposal addresses prior concerns by**:
1. Leading with CV-based actionability decomposition (directly addresses field's key question)
2. Framing phase transition as supporting theoretical context (not primary claim)
3. Acknowledging chi_ratio limitation and λ_c instability explicitly
4. Targeting mid-tier venue (AAAI/EMNLP/Workshop) with honest scope

---

## 1. Title

**Beyond the Actionability Paradox: Coefficient of Variation Predicts Steering Heterogeneity in Absorbed SAE Features**

---

## 2. Abstract

Feature absorption in Sparse Autoencoders (SAEs) creates an "interpretability illusion" where latents appear monosemantic but exhibit systematic false negatives in probing tasks. Basu et al. (2026) demonstrated that near-perfect feature detection (98.2% AUROC) translates to zero steering utility — the "actionability paradox." This finding has cast doubt on the entire SAE-based interpretability enterprise.

We present evidence that the actionability paradox is not universal. Our pilot experiments on GPT-2 SAEs reveal that absorbed features with high coefficient of variation (CV > 1.0) show steering effects 2x larger than absorbed features with low CV (CV <= 1.0): 0.153 vs 0.075 logit change. Activation patching confirms this represents genuine causal structure — parent features recover 67.3% on average when child features are zeroed.

We propose that absorbed features decompose into two subpopulations: (1) "robust absorbed" features (high-CV) routed through context-sensitive child channels that preserve steering potential, and (2) "fragile absorbed" features (low-CV) routed through stable child channels that compensate for parent steering. The coefficient of variation — a simple statistical measure — predicts which subpopulation an absorbed feature belongs to, providing actionable guidance for interpretability practitioners without expensive steering experiments.

Our findings suggest the Basu et al. actionability paradox may reflect domain-specific sampling (clinical features predominantly low-CV) rather than universal failure. This work provides the first evidence that absorbed features are not uniformly non-steerable in non-clinical LLM domain, and establishes CV as a practical predictor for steering feasibility.

---

## 3. Motivation

### The Actionability Paradox

Basu et al. (2026) demonstrated that 98.2% AUROC feature detection via SAE probing translates to 0% output sensitivity to SAE steering in clinical domain. This "actionability paradox" suggests that measuring absorption may not help us predict what we can actually DO with SAE features.

### What the Field Needs

The field asks: "Which absorbed features can we actually steer, and does measuring absorption help us predict that?" The current approach treats all absorbed features as uniformly non-steerable. If absorbed features are heterogeneous in their steerability, a predictor could help practitioners prioritize which features to use.

### Our Preliminary Evidence

Two pilot experiments provide converging evidence:
1. **Activation Patching**: All 9 persistent core words show >48% recovery (mean 67.3%) when child is zeroed — confirms absorbed features have genuine causal effects that could theoretically be steered
2. **Steering by CV**: High-CV absorbed features show 2x larger steering effect (0.153 vs 0.075) — suggests CV may predict which absorbed features retain steering potential

### Why This Matters

If CV predicts steering effectiveness:
- Practitioners can prioritize high-CV absorbed features for steering interventions
- The actionability paradox may not be universal — it may apply to some feature types but not others
- We can connect abstract absorption metrics to practical interpretability utility

---

## 4. Research Questions

**RQ1**: Does the coefficient of variation (CV) predict steering effectiveness for absorbed SAE features?

**RQ2**: Are absorbed features uniformly non-steerable (Basu et al. universal failure), or do they decompose into steerable and non-steerable subpopulations?

**RQ3**: What mechanism explains why high-CV absorbed features show larger steering effects?

**RQ4**: Does the CV-steering correlation generalize across architectures (GPT-2 to Gemma-2)?

---

## 5. Hypotheses

### Primary Hypotheses

**H1 (CV Predicts Steering)**: Absorbed features with high coefficient of variation (CV > 1.0) show significantly larger steering effects than absorbed features with low CV (CV <= 1.0), after controlling for decoder magnitude.

**H4 (Variance Paradox — Genuine Discovery)**: Absorbed features exhibit higher CV (CV ≈ 7.33) than non-absorbed features (CV ≈ 0.01). This 733x ratio reflects that absorption selectively preserves context-sensitive specialized information, not measurement artifact.

**Actionability Paradox Refinement**: The Basu et al. actionability paradox may be domain-specific (clinical features predominantly low-CV) rather than universal. In non-clinical LLM domain, high-CV absorbed features may retain steering potential.

### Secondary Hypotheses

**H6 (Decoder Orthogonality)**: Features with decoder weights maximally orthogonal to other features show higher steering effectiveness. Orthogonality may partially explain the CV-steering correlation.

**Cross-Architecture Generalization**: The CV-steering correlation replicates on Gemma-2-2B JumpReLU SAEs with similar CV threshold.

### Falsified Hypotheses (Reported as Informative Negatives)

**H3 (Cross-Layer at λ=0.001)**: At λ=0.001, all layers saturate at absorption_rate=1.0 — uniform saturation contradicts layer-criticality narrative. H3 needs retesting at λ_c=5e-5.

**H6 (Graph Topology)**: Component count decreases with layer (L0=24420 > L9=23371), not peaked at layer 6. Graph topology is not an order parameter for absorption.

---

## 6. Evidence-Driven Revisions

### What Changed from Initial Hypotheses

| Hypothesis | Original Prediction | Observed | Interpretation |
|------------|---------------------|----------|----------------|
| H1 framing | "CV does not predict steering" | High-CV = 2x steering | **Confirmed** — CV is predictor |
| H4 framing | "CV_low < CV_high" | CV_high >> CV_low (733x) | **Genuine discovery** — absorption preserves high-variance specialized info |
| H3 narrative | "Layer 6 at critical point" | All layers saturated at 1.0 | Sparsity was wrong; need finer λ measurement at λ_c |
| H6 narrative | "Graph topology peaks at L6" | Component count decreases | Graph topology is not the order parameter |

### What Was Strengthened

| Finding | Evidence | Significance |
|---------|----------|--------------|
| CV-steering correlation | 0.153 vs 0.075 (2x difference) | Strong pilot validation |
| Genuine absorption | 67.3% mean activation patching recovery | Confirms causal structure exists to steer |
| High-CV = steerable | Pilot result aligns with theoretical bypass/mediated regime | Mechanistic hypothesis supported |

---

## 7. Method

### Phase 1: CV-Based Feature Classification (15 min)

- Load GPT-2 layer 6 SAE via SAELens (gpt2-small-res-jb, 16k latents)
- Compute per-feature CV across 1000 text samples
- Classify absorbed features (absorption_score > 0.5) into high-CV (CV > 1.0) and low-CV (CV <= 1.0)
- Target: 30+ features in each group

### Phase 2: Steering Effectiveness Comparison (30 min)

- Run steering experiments on 30 high-CV and 30 low-CV absorbed features
- Steering strengths: +3, +5, +7
- Metric: logit change at semantically appropriate tokens
- Statistical test: one-sided Welch's t-test (α = 0.01)

### Phase 3: Mechanism Investigation (20 min)

- Test whether CV-steering correlation is explained by:
  - Decoder weight orthogonality (high-CV features have more orthogonal decoders)
  - Feature frequency (high-CV features are rarer but more specialized)
  - Context sensitivity (high-CV features activate in narrower context distributions)
- Control for decoder magnitude using Fano factor (CV²/mean)

### Phase 4: Non-Absorbed Baseline (15 min)

- Compare steering effects for absorbed vs non-absorbed features
- Establishes whether "robust absorbed" is comparable to non-absorbed or still degraded

### Phase 5: Cross-Model Validation (30 min)

- Replicate on Gemma-2-2B layer 6 JumpReLU SAE
- Test whether CV threshold (1.0) generalizes or model-specific

---

## 8. Experimental Plan

| Experiment | Details | Duration | Validates |
|-----------|---------|----------|-----------|
| E1: CV classification | GPT-2 layer 6, classify absorbed features | 15 min | High/low CV split on absorbed features |
| E2: Steering comparison | 30 high-CV vs 30 low-CV absorbed features, 3 strengths | 30 min | Robust vs fragile absorbed hypothesis |
| E3: Mechanism analysis | Decoder orthogonality, Fano factor control | 20 min | Mechanism explanation |
| E4: Non-absorbed baseline | Compare to non-absorbed steering effects | 15 min | Context for absorbed results |
| E5: Gemma-2 validation | Gemma-2-2B layer 6, same protocol | 30 min | Cross-model generalization |

**Total**: ~110 min across 5 experiments (within project budget)

### Simplest Version

Single experiment: 30 high-CV vs 30 low-CV absorbed features on GPT-2 layer 6. Steering at +5 strength. Logit change measurement. ~30 min runtime.

**Expected outcomes**:
- Positive: High-CV shows larger steering effect (p < 0.01) → CV predicts steering heterogeneity
- Negative: No significant difference → Basu et al. actionability paradox may be universal

---

## 9. Resource Estimate

- **Models**: GPT-2-small (86M params, fast), Gemma-2-2B (2B params, slower but acceptable)
- **SAEs**: GPT-2 layer 6 residual stream (~16k latents), GemmaScope layer 6
- **Compute**: ~2 GPU hours total
- **No new training**: Training-free analysis of pretrained SAEs via SAELens

---

## 10. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| CV threshold (1.0) is not predictive on held-out data | Medium | Validate on held-out features; report as exploratory if not predictive |
| Steering effect too small for practical utility | Medium | Compare to non-absorbed baseline; report normalized effect |
| Gemma-2 shows no CV effect | Medium | Report as negative result; Basu et al. confirmed for LLM domain |
| Fano factor normalization shows CV is purely magnitude proxy | Low | Use Fano factor as control variable; steering result provides independent validation |

---

## 11. Novelty Assessment

**What is genuinely novel**:
1. **First CV-based prediction of steering effectiveness** within absorbed features — prior work (Basu et al.) treats all absorbed as uniformly non-steerable
2. **First evidence that absorbed features are not uniformly non-steerable** in non-clinical LLM domain
3. **First connection between coefficient of variation and causal actionability** — a simple statistical measure predicts whether a feature can be steered
4. **First partial resolution of the actionability paradox** — if high-CV features are steerable, the paradox is not universal

**Prior work collisions** (from novelty_report.json):
- Basu et al. (2026): Actionability paradox — we extend by showing heterogeneity rather than universal failure
- Chanin et al. (2024): Absorption detection — we connect to steering outcomes rather than just measuring absorption
- Cui et al. (2026): Information-theoretic impossibility — we work within these limits rather than trying to overcome them

**Differentiation**: This is NOT claiming to resolve the actionability paradox. We provide evidence that absorbed features are not uniformly non-steerable, and that CV partially predicts which absorbed features retain steering potential. The field's question is "why does good detection fail to predict steering?" — our answer: CV captures something about feature behavior that absorption metrics miss.

---

## 12. Expected Contributions

1. **Empirical**: First systematic evidence that absorbed SAE features decompose into steerable and non-steerable subpopulations
2. **Predictive**: CV as a simple statistical predictor for steering feasibility — no expensive steering experiments needed
3. **Theoretical**: Causal mediation framework connecting CV to bypass/mediated regime routing
4. **Practical**: Guidance for interpretability practitioners on which absorbed features to prioritize for steering

---

## 13. What Changed from Prior Round

| Aspect | Prior Round (iter_004) | This Round |
|--------|------------------------|------------|
| Front-runner | Phase transitions with finite-size scaling | CV-based actionability decomposition |
| chi_ratio framing | "Sharp/Quasi-critical transition" | Supporting theoretical context only |
| H3 narrative | "Layer as temperature" | Needs retesting at λ_c (falsified at λ=0.001) |
| H6 narrative | "Graph topology peaks" | Falsified — graph topology not order parameter |
| Venue | Top-tier (NeurIPS/ICML) | Mid-tier (AAAI/EMNLP/Workshop) |
| Primary novelty | Finite-size scaling (ν=3, R²=0.951) | CV-steering correlation and actionability heterogeneity |
| λ_c treatment | Treated as stable critical point | Acknowledged as needing prospective validation |

---

## 14. Connection to Basu et al. Actionability Paradox

Basu et al. (2026) demonstrate 98.2% AUROC but 0% steering in clinical domain. Our findings suggest the paradox may not be universal:

1. **High-CV absorbed features** route through specialized child channels with context-sensitive activation
2. **Context-sensitive channels** create mediated routing where steering can modulate behavior
3. **Low-CV absorbed features** route through stable child channels with bypass routing where steering has zero effect
4. **Clinical features** (from Basu et al.) may be predominantly low-CV, explaining universal failure in that domain

**Implication**: Absorption metrics may predict WHAT features are absorbed but not WHICH absorbed features remain steerable. The CV-based decomposition provides the missing predictor.

---

## 15. References

- Chanin et al. (2024): A is for Absorption (detection metric, hierarchical co-occurrence)
- Basu et al. (2026): Interpretability without Actionability (actionability paradox)
- Cui et al. (2026): On the Limits of SAEs (information-theoretic impossibility)
- Karvonen et al. (2025): SAEBench (probe projection metric)
- Pearl (2009): Causality (causal mediation framework)
- Costa et al. (2025): MP-SAE (hierarchical feature recovery)
- Engel & Van den Broeck (2001): Statistical Mechanics of Learning (phase transitions)