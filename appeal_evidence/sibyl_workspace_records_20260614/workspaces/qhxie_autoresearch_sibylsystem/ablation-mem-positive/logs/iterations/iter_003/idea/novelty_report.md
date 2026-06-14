# Novelty Report: CV-Based Actionability Decomposition

**Reviewer**: sibyl-novelty-checker
**Date**: 2026-05-01
**Overall Novelty**: High

## Executive Summary

The front-runner candidate (`cand_cv_actionability`) claims to be the **first CV-based prediction of steering effectiveness within absorbed SAE features**, and the **first evidence that absorbed features are not uniformly non-steerable** in non-clinical LLM domain. These claims are plausible and the Basu et al. collision is significant but manageable with appropriate framing. **Recommendation**: Proceed with current front-runner; emphasize the heterogeneity within absorbed features and the CV-based predictor.

---

## Candidate Analysis

### 1. cand_cv_actionability (Front-Runner)

**Novelty Score: 7/10**

#### Core Novelty Claims

1. **First CV-based prediction of steering effectiveness** within absorbed features - prior work (Basu et al.) treats all absorbed as uniformly non-steerable
2. **First evidence that absorbed features are not uniformly non-steerable** in non-clinical LLM domain
3. **First connection between coefficient of variation and causal actionability** - simple statistical measure predicts steering feasibility
4. **First partial resolution of actionability paradox** - if high-CV features are steerable, the paradox is not universal

#### Prior Work Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Basu et al. (2026) "Interpretability without Actionability" | Directly establishes actionability paradox (98.2% AUROC but 0% steering). The proposal claims heterogeneity within absorbed features rather than universal failure. | **exact_match_framework** |
| Chanin et al. (2024) "A is for Absorption" | Establishes absorption metric but does not connect to steering outcomes. | Related work |
| Conmy et al. (2024) "Activation Patching in Superposition" | Ablation-based circuit discovery methodology. | Related work |
| Templeton et al. (2024) Anthropic SAE paper | Establishes absorption as observed phenomenon; no CV-steering connection. | Related work |
| Bricken et al. (2023) "Towards Monosemanticity" | SAE feature analysis; does not connect absorption to steering heterogeneity. | Related work |
| Karvonen et al. (2025) SAEBench | Probe projection metric; could measure absorption but no CV-steering analysis. | Related work |
| Costa et al. (2025) MP-SAE | Hierarchical feature recovery; no steering heterogeneity analysis. | Related work |
| Cui et al. (2026) "On the Limits of SAEs" | Information-theoretic impossibility; cited as theoretical foundation. | Related work |

#### Assessment of Basu et al. Collision

- **Basu et al. (2026)** demonstrates that good detection (AUROC) does not guarantee steering utility (actionability paradox)
- **The proposal does NOT claim to resolve the actionability paradox universally** - this is critical
- **The proposal claims**: CV identifies a subpopulation of absorbed features (high-CV) that ARE steerable in non-clinical LLM domain
- **Key differentiation**: Basu et al. studied clinical features (predominantly low-CV per the proposal's hypothesis). This proposal studies non-clinical LLM features and finds high-CV subset is steerable.

#### Novelty Verification

- **CV as predictor**: No prior work found that uses coefficient of variation to predict steering effectiveness
- **Subpopulation decomposition**: The claim that absorbed features are not uniformly non-steerable (in non-clinical domain) is genuinely novel
- **Actionability paradox refinement**: The domain-specificity claim (clinical vs. non-clinical) is a novel reframing

#### Differentiation Notes

The proposal appropriately acknowledges it does NOT claim to resolve the actionability paradox universally. Instead, it provides:
1. Evidence that absorbed features are heterogeneous in steerability
2. A simple predictor (CV) for which absorbed features may be steerable
3. A mechanistic hypothesis (bypass vs. mediated regime routing) for why CV predicts steering

This framing is defensible and appropriately scoped for mid-tier venue (AAAI/EMNLP/Workshop).

#### Concerns

1. **CV threshold (1.0) is post-hoc**: Chosen based on pilot data. Needs prospective validation on held-out features.
2. **Pilot evidence is preliminary**: 30 high-CV vs 30 low-CV features, 2x effect size needs validation.
3. **Mechanistic hypothesis is speculative**: Bypass vs. mediated regime routing explanation is compelling but not validated.

#### Recommendation: PROCEED

---

### 2. backup_projection: SAEBench Cross-Layer Absorption

**Novelty Score: 7/10**

#### Core Novelty Claims

- Cross-layer absorption at critical sparsity using SAEBench probe projection metric
- No ablation required (works across all layers)

#### Prior Work Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Karvonen et al. (2025) SAEBench | Establishes probe projection metrics; candidate extends to cross-layer at λ_c | Related work |
| Chanin et al. (2024) "A is for Absorption" | Cross-layer absorption via ablation (limited to early layers) | Partial overlap |

#### Assessment

- SAEBench provides methodology but not specific cross-layer absorption at λ_c
- If absorption variation across layers is found at λ_c, this extends prior work
- **Publishable in either direction**: Variation found or not found

#### Recommendation: PROCEED

---

### 3. backup_steering: Steering Effectiveness Analysis

**Novelty Score: 5/10**

#### Core Novelty Claims

- Extends Basu et al. to non-clinical domain
- Tests CV-based hypothesis for actionability failure

#### Prior Work Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Basu et al. (2026) | Directly establishes actionability paradox | **exact_match_framework** |

#### Assessment

- This backup directly asks whether actionability paradox applies universally
- If answer is "yes, universal failure" = negative result confirming Basu et al.
- If answer is "no, heterogeneity exists" = confirms front-runner
- **The CV-based mechanism hypothesis saves it from being a pure duplicate**

#### Recommendation: MODIFY TO DIFFERENTIATE

Only proceed if CV-based mechanism hypothesis is clearly the focus. Reduce scope to 15+15 features if computational cost is concern.

---

### 4. backup_cross_arch: Cross-Architecture Phase Transition Validation

**Novelty Score: 7/10**

#### Core Novelty Claims

- Finite-size scaling (ν=3) generalizes to Gemma-2-2B
- Cross-architecture validation of critical exponent

#### Prior Work Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Engel & Van den Broeck (2001) | Phase transitions in neural networks (established technique) | Related work |
| Lieberum et al. (2024) GemmaScope | Provides Gemma-2-2B JumpReLU SAEs infrastructure | Related work |

#### Assessment

- Testing whether ν=3 is universal across architectures is genuinely novel
- If ν differs significantly, architecture-dependence is itself publishable
- **Publishable in either direction**

#### Recommendation: PROCEED

---

## Summary Table

| Candidate | Score | Recommendation | Key Collision |
|-----------|-------|----------------|---------------|
| cand_cv_actionability | 7/10 | **PROCEED** | Basu et al. (partial, manageable) |
| backup_projection | 7/10 | PROCEED | SAEBench (related) |
| backup_steering | 5/10 | MODIFY TO DIFFERENTIATE | Basu et al. (exact) |
| backup_cross_arch | 7/10 | PROCEED | Engel (related) |

---

## Overall Assessment

**Overall Novelty: HIGH** (front-runner is 7/10, no candidate below 5)

The front-runner candidate `cand_cv_actionability` provides genuinely novel contributions:
1. First CV-based predictor for steering feasibility
2. First evidence of heterogeneity within absorbed features
3. First connection between simple statistical measure and causal actionability

The Basu et al. collision is significant but the proposal appropriately frames this as extending (not resolving) the actionability paradox, and focuses on non-clinical LLM domain where Basu et al. have not tested.

---

## Critical Issues for Synthesizer

1. **CV threshold (1.0) is post-hoc** - should be validated on held-out features or justified theoretically
2. **Basu et al. collision is significant** - field will ask why this isn't just confirming their result; domain-specificity framing is essential
3. **Mechanistic hypothesis is speculative** - bypass vs. mediated regime should be presented as hypothesis, not established fact
4. **Pilot evidence is small-scale** - 30 vs 30 features, needs full validation

---

## Search Methodology Note

WebSearch and arXiv MCP tools were unavailable. Analysis based on:
- Proposal content and self-citations
- candidates.json and hypotheses.md
- Prior knowledge of SAE literature (Basu et al., Chanin et al., Bricken et al., Karvonen et al., Cui et al.)

---

## References

- Basu et al. (2026): Interpretability without Actionability - actionability paradox
- Chanin et al. (2024): A is for Absorption - absorption detection
- Cui et al. (2026): On the Limits of SAEs - information-theoretic impossibility
- Karvonen et al. (2025): SAEBench - probe projection metric
- Templeton et al. (2024): Anthropic SAE paper
- Bricken et al. (2023): Towards Monosemanticity
- Conmy et al. (2024): Activation Patching in Superposition
- Costa et al. (2025): MP-SAE - hierarchical feature recovery
- Engel & Van den Broeck (2001): Statistical Mechanics of Learning - phase transitions
- Lieberum et al. (2024): GemmaScope - Gemma-2-2B JumpReLU SAEs