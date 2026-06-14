# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Chanin et al., 2024. A is for Absorption. arXiv:2409.14507 (NeurIPS 2025)** — First systematic study of feature absorption; establishes hierarchical co-occurrence as root cause; introduces ablation-based detection metric limited to early layers. Critical for understanding what we're trying to detect and quantify.

2. **Basu et al., 2026. Interpretability without Actionability. arXiv:2603.18353** — Establishes the "actionability paradox": 98.2% AUROC detection but 0% steering utility in clinical domain. This is the central challenge our front-runner addresses. We argue the paradox is domain-specific (clinical features predominantly low-CV) rather than universal.

3. **Cui et al., 2026. On the Limits of SAEs. arXiv:2506.15963 (ICLR 2026)** — Proves mathematically that standard SAEs cannot fully recover ground-truth monosemantic features. Provides theoretical bounds that constrain what any absorption mitigation can achieve. We work within these limits rather than trying to overcome them.

4. **Costa et al., 2025. From Flat to Hierarchical: MP-SAE. arXiv:2506.03093 (NeurIPS 2025)** — Matching Pursuit SAE recovers hierarchical structure via residual-guided greedy selection. Demonstrates conditional orthogonality across hierarchy levels. MP-SAE's success suggests absorbed features have recoverable structure if we can detect it.

5. **Karvonen et al., 2025. SAEBench. arXiv:2503.09532 (ICML 2025)** — Comprehensive 8-metric benchmark including probe projection absorption metric that works across all layers (unlike ablation-based metric). Provides standardized evaluation framework for our experiments.

6. **Korznikov et al., 2025. OrtSAE. arXiv:2509.22033** — Orthogonality penalty reduces absorption by 65%. Demonstrates that architectural modifications can address absorption but requires retraining. We focus on training-free detection in existing pretrained SAEs.

7. **Bussmann et al., 2025. Matryoshka SAE. arXiv:2503.17547** — Nested dictionaries organize features hierarchically, reducing absorption. Training-time solution not applicable to our analysis of pretrained SAEs.

8. **Engel & Van den Broeck, 2001. Statistical Mechanics of Learning** — Phase transition theory underlying our finite-size scaling analysis. Provides mathematical framework for understanding critical points in sparse coding.

### Landscape Summary

The SAE field has established three key facts: (1) Feature absorption is real and systematic — parent features are subsumed by child features during sparse optimization (Chanin et al., 2024). (2) The actionability paradox challenges the practical utility of absorption research — good detection does not guarantee steering utility (Basu et al., 2026). (3) Mathematical limits exist on disentanglement quality — full recovery is impossible under realistic sparsity (Cui et al., 2026).

The key gap is NOT detecting absorption anymore — we can measure it. The gap is: **which absorbed features can we actually steer, and does measuring absorption help us predict that?** This is exactly what our front-runner addresses.

Prior work treats absorbed features as uniformly non-steerable. We provide the first evidence that absorbed features decompose into steerable (high-CV) and non-steerable (low-CV) subpopulations in non-clinical LLM domain.

---

## Phase 2: Initial Candidates

### Candidate A: CV-Based Actionability Decomposition (Front-Runner)

- **Hypothesis**: Absorbed features with high coefficient of variation (CV > 1.0) show significantly larger steering effects than absorbed features with low CV (CV <= 1.0), after controlling for decoder magnitude.

- **Cross-domain insight**: The variance paradox (absorbed features have 733x higher CV than non-absorbed) reflects that absorption selectively preserves **context-sensitive specialized information**. When a parent feature P is absorbed into child C, the routing preserves high-variance specialized channels (e.g., "letter A at word start") over low-variance general channels (e.g., "any first letter"). This is analogous to **signal processing**: compressed sensing preserves high-frequency components essential for reconstruction.

- **Why it might work**: Pilot data shows high-CV steering effect = 0.153 vs low-CV = 0.075 (2x difference, p < 0.01). Activation patching confirms 67.3% mean recovery — absorbed features have genuine causal structure. The mechanism is that high-CV features route through context-sensitive child channels in mediated regime; steering the parent modulates child behavior. Low-CV features route through stable child channels in bypass regime; steering has zero effect because child compensates identically.

- **Novelty estimate**: 8/10 — First connection between coefficient of variation and causal actionability. Basu et al. treat all absorbed as uniformly non-steerable. We show heterogeneity within absorbed features in non-clinical domain.

### Candidate B: Encoder-Decoder Asymmetry for Training-Free Absorption Detection

- **Hypothesis**: Absorbed features exhibit systematic encoder-decoder asymmetry — the encoder weight norm differs from decoder weight norm in ways that predict absorption severity. This provides a training-free detection method without requiring probe tasks.

- **Cross-domain insight**: In compressed sensing (Candes & Romberg, 2006), the analysis operator (encoder) and synthesis operator (decoder) have known relationships to sparsity patterns. If SAE absorption is analogous to compressed sensing failure modes, encoder-decoder asymmetry may indicate which features are absorbed.

- **Why it might work**: Chanin et al. toy models suggest encoder-decoder asymmetry as absorption indicator, but no prior work systematically uses this as detection method. SAELens provides easy access to W_enc and W_dec weights for all pretrained SAEs. If asymmetry correlates with absorption score, we have a free lunch — no steering experiments needed.

- **Novelty estimate**: 7/10 — Novel application of signal processing theory to SAE analysis. Gap 4 (training-free detection in pretrained SAEs) is identified but underexplored.

### Candidate C: Cross-Layer Absorption Heterogeneity at Critical Sparsity

- **Hypothesis**: Cross-layer absorption variation exists but requires measurement at critical sparsity λ_c (not λ=0.001). At λ_c, layer 6 exhibits maximum heterogeneity; at λ=0.001 all layers saturate at absorption_rate=1.0.

- **Cross-domain insight**: Phase transitions in neural networks (Engel & Van den Broeck, 2001) show that critical points reveal underlying structure invisible at non-critical regimes. SAE sparsity acts as "temperature" control.

- **Why it might work**: Pilot data shows chi_ratio=1.88 (< 3.0 threshold) undermines "sharp transition" framing, but the phase transition framework provides theoretical context. H3 falsified at λ=0.001 but needs retesting at λ_c=5e-5. The backup proposal addresses this.

- **Novelty estimate**: 6/10 — Finite-size scaling is established technique; novelty is applying to SAE absorption specifically. Risk: if λ_c is unstable (10x pilot-to-full shift), finding may not replicate.

---

## Phase 3: Self-Critique

### Against Candidate A (CV-Based Actionability)

- **Prior work attack**: Basu et al. (2026) directly establishes actionability paradox. Our claim that absorbed features are NOT uniformly non-steerable is contrarian. However, Basu et al. study clinical features; we study non-clinical LLM features. Domain specificity is key.

- **Methodological attack**: CV threshold (1.0) is chosen post-hoc from pilot data. Needs prospective validation on held-out features. 30 vs 30 feature comparison is underpowered for definitive conclusions. Steering experiments are computationally expensive (30 min per group).

- **Theoretical attack**: Bypass vs. mediated regime routing explanation is compelling but speculative. No direct evidence that high-CV features actually route through context-sensitive channels. The Fano factor control (CV²/mean) may reveal that CV is just a magnitude proxy.

- **Scalability attack**: Pilot验证 on GPT-2 layer 6. If Gemma-2-2B shows no CV effect, the finding may be architecture-specific. TopK (GPT-2) vs JumpReLU (Gemma-2) architecture differences may affect CV distribution.

- **Verdict**: MODERATE — Strong pilot evidence, but CV threshold post-hoc and mechanism speculative. Address by validating on held-out features and confirming mechanism with Fano factor control.

### Against Candidate B (Encoder-Decoder Asymmetry)

- **Prior work attack**: No prior work uses encoder-decoder asymmetry for absorption detection. This is genuinely novel. However, the signal processing analogy may be superficial — SAEs don't have the same restricted isometry property as compressed sensing matrices.

- **Methodological attack**: Weight norm analysis is straightforward but may not capture absorption dynamics. Absorption is a functional property (how features route information) not just a structural property (weight magnitudes). Need to validate that asymmetry actually predicts steering failure.

- **Theoretical attack**: The analogy to compressed sensing failure modes is suggestive but not proven. If absorption is caused by hierarchical co-occurrence during training (not mathematical impossibility), encoder-decoder asymmetry may not correlate with absorption severity.

- **Scalability attack**: Easy to compute for all features, but may not generalize across SAE architectures (TopK vs JumpReLU have different weight distributions).

- **Verdict**: MODERATE — Novel approach but theoretical grounding is weak. Could be high-value if it works as training-free detection.

### Against Candidate C (Cross-Layer Heterogeneity)

- **Prior work attack**: Finite-size scaling is established in statistical mechanics. Applying it to SAEs is novel but the mathematical framework is not new. Engel & Van den Broeck (2001) covers this extensively.

- **Methodological attack**: λ_c instability (10x pilot-to-full shift) is a red flag. chi_ratio=1.88 < 3.0 undermines "sharp transition" framing. H3 falsified at λ=0.001 — need λ_c=5e-5 but not confident this will show layer variation.

- **Theoretical attack**: Phase transition framework provides supporting context but may be overengineering the problem. If CV-based actionability works, we may not need layer-criticality narrative.

- **Scalability attack**: Requires sparsity sweeps across multiple layers — computationally expensive. The benefit is limited if the cross-layer variation is small or unreliable.

- **Verdict**: WEAK — Falsified at wrong λ; χ² ratio below threshold; λ_c instability. Drop unless CV-based actionability fails.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (Cross-Layer Heterogeneity) dropped because**: H3 falsified at λ=0.001 (all layers saturate at absorption_rate=1.0). chi_ratio=1.88 < 3.0 undermines quasi-critical framing. λ_c instability (10x pilot-to-full shift) makes prospective validation unreliable. The phase transition narrative is not necessary for the primary contribution — CV-based actionability decomposition stands on its own as the primary novelty.

### Strengthened Ideas

- **Candidate A (CV-Based Actionability) strengthened by**:
  1. Explicitly framing phase transition as supporting theoretical context, not primary claim
  2. Acknowledging chi_ratio limitation and λ_c instability in the proposal
  3. Targeting mid-tier venue (AAAI/EMNLP/Workshop) with honest scope
  4. Adding Fano factor control to validate CV is not purely magnitude proxy
  5. Proposing held-out validation for CV threshold (1.0)

- **Candidate B (Encoder-Decoder Asymmetry) strengthened by**:
  1. Positioning as backup if front-runner fails
  2. Acknowledging theoretical grounding is weak but signal processing analogy is suggestive
  3. Framing as training-free detection (addresses Gap 4 directly)

### Additional Evidence Found

- Pilot activation patching (9/9 words >48% recovery, mean 67.3%) confirms genuine absorption for persistent core words
- Pilot steering CV (high-CV 0.153 vs low-CV 0.075, 2x difference) confirms CV-steering correlation
- SynthSAEBench (arXiv:2602.14687, 2026) provides ground-truth validation framework for new metrics
- LatentScalpel (2026-04-26) shows growing interest in applying SAEs to diffusion models — cross-domain potential

### Selected Front-Runner

**Candidate A: CV-Based Actionability Decomposition**

Rationale:
1. Directly addresses the field's key question (actionability paradox, Basu et al.)
2. First evidence that absorbed features are not uniformly non-steerable in non-clinical LLM domain
3. CV is simple statistical measure — no expensive steering experiments needed to predict feasibility
4. Training-free analysis via SAELens (pretrained SAEs, no retraining)
5. All experiments fit within 1-hour budget per task
6. Pilot evidence is strong (67.3% patching recovery, 2x steering effect)
7. Mechanism hypothesis (bypass vs. mediated regime) is testable

The backup (Candidate B) provides an alternative approach if front-runner fails or reveals unexpected limitations.

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

### Cross-Domain Insight

The variance paradox (absorbed features have CV ≈ 7.33 vs non-absorbed CV ≈ 0.01, 733x ratio) reflects that absorption selectively preserves **context-sensitive specialized information**. When a parent feature is absorbed into a child feature during sparse optimization, the routing preserves high-variance specialized channels (e.g., "letter A at word start") over low-variance general channels (e.g., "any first letter").

This is analogous to **signal processing in compressed sensing**: when reconstruction is imperfect, high-frequency/variance components are preserved because they carry critical discriminative information. The coefficient of variation captures this selection pressure — absorbed features with high CV are those where specialized information survived the compression.

The mechanism is: high-CV features route through context-sensitive child channels in **mediated regime** (steering the parent modulates child behavior); low-CV features route through stable child channels in **bypass regime** (steering has zero effect because child compensates identically).

### Method

**Phase 1: CV-Based Feature Classification (15 min)**
- Load GPT-2 layer 6 SAE via SAELens (gpt2-small-res-jb, 16k latents)
- Compute per-feature CV across 1000 text samples from dataset
- Classify absorbed features (absorption_score > 0.5) into high-CV (CV > 1.0) and low-CV (CV <= 1.0)
- Control for decoder magnitude using Fano factor (CV²/mean) — ensures CV is not just magnitude proxy
- Target: 30+ features in each group

**Phase 2: Steering Effectiveness Comparison (30 min)**
- Run steering experiments on 30 high-CV and 30 low-CV absorbed features
- Steering strengths: +3, +5, +7
- Metric: logit change at semantically appropriate tokens
- Statistical test: one-sided Welch's t-test (α = 0.01) with Benjamini-Hochberg FDR correction

**Phase 3: Non-Absorbed Baseline (15 min)**
- Compare steering effects for absorbed vs non-absorbed features
- Establishes whether "robust absorbed" is comparable to non-absorbed or still degraded

**Phase 4: Cross-Architecture Validation (30 min)**
- Replicate on Gemma-2-2B layer 6 JumpReLU SAE
- Test whether CV threshold (1.0) generalizes or model-specific

### Experimental Plan

| Experiment | Details | Duration | Validates |
|-----------|---------|----------|-----------|
| E1: CV classification | GPT-2 layer 6, classify absorbed features | 15 min | High/low CV split on absorbed features |
| E2: Steering comparison | 30 high-CV vs 30 low-CV absorbed features, 3 strengths | 30 min | Robust vs fragile absorbed hypothesis |
| E3: Fano factor control | Verify CV is not magnitude proxy | 15 min | Mechanism validation |
| E4: Non-absorbed baseline | Compare to non-absorbed steering | 15 min | Context for absorbed results |
| E5: Gemma-2 validation | Gemma-2-2B layer 6, same protocol | 30 min | Cross-model generalization |

**Total**: ~105 min across 5 experiments (within project budget)

**Simplest version**: Single experiment: 30 high-CV vs 30 low-CV absorbed features on GPT-2 layer 6. Steering at +5 strength. Logit change measurement. ~30 min runtime.

**Expected outcomes**:
- Positive: High-CV shows larger steering effect (p < 0.01) → CV predicts steering heterogeneity
- Negative: No significant difference → Basu et al. actionability paradox confirmed for non-clinical LLM domain

**Falsification**: If no significant difference in steering effect between high-CV and low-CV groups (p > 0.05 after multiple testing correction), the hypothesis is DISPROVEN.

### Resource Estimate

- **Models**: GPT-2-small (86M params, fast), Gemma-2-2B (2B params, slower but acceptable)
- **SAEs**: GPT-2 layer 6 residual stream (~16k latents), GemmaScope layer 6
- **Compute**: ~2 GPU hours total
- **No new training**: Training-free analysis of pretrained SAEs via SAELens
- **Per-experiment budget**: ≤1 hour (pilot 10-15 min)

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| CV threshold (1.0) is not predictive on held-out data | Medium | Validate on held-out features; report as exploratory if not predictive |
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

### References

- Chanin et al. (2024): A is for Absorption — absorption detection, hierarchical co-occurrence
- Basu et al. (2026): Interpretability without Actionability — actionability paradox
- Cui et al. (2026): On the Limits of SAEs — information-theoretic impossibility
- Karvonen et al. (2025): SAEBench — probe projection metric
- Costa et al. (2025): MP-SAE — hierarchical feature recovery
- Engel & Van den Broeck (2001): Statistical Mechanics of Learning — phase transitions
- Candes & Romberg (2006): Compressed sensing — signal processing analogy
- Pearl (2009): Causality — causal mediation framework