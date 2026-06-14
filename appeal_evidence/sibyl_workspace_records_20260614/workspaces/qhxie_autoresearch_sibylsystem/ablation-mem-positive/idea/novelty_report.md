# Novelty Report: CV-Based Actionability Decomposition

**Reviewer**: sibyl-novelty-checker
**Date**: 2026-05-01
**Overall Novelty**: High

## Executive Summary

The front-runner candidate (`cand_cv_actionability`) claims to be the **first CV-based prediction of steering effectiveness within absorbed SAE features**, and the **first evidence that absorbed features are not uniformly non-steerable** in non-clinical LLM domain. These claims are plausible and the Basu et al. collision is significant but manageable with appropriate framing.

**Critical Framing Requirement**: The proposal must NOT claim to resolve the actionability paradox universally. Instead, it must focus on demonstrating heterogeneity within absorbed features in non-clinical LLM domain, with CV as a predictor.

**Recommendation**: Proceed with current front-runner; emphasize the heterogeneity within absorbed features and the CV-based predictor. Address all critical concerns before submission.

---

## Novelty Scores Summary

| Candidate | Score | Recommendation | Key Collision |
|-----------|-------|----------------|---------------|
| cand_cv_actionability | **7/10** | PROCEED | Basu et al. (exact_match_framework) |
| backup_projection | **7/10** | PROCEED | SAEBench (related_work) |
| backup_steering | **5/10** | MODIFY TO DIFFERENTIATE | Basu et al. (exact_match_framework) |
| backup_cross_arch | **7/10** | PROCEED | Engel & Van den Broeck (related_work) |

**Overall Novelty**: HIGH (all candidates >= 5/10)

---

## Candidate Analysis

### 1. cand_cv_actionability (Front-Runner)

**Novelty Score: 7/10**

#### Core Novelty Claims

1. **First CV-based prediction of steering effectiveness** within absorbed features - prior work (Basu et al.) treats all absorbed as uniformly non-steerable
2. **First evidence that absorbed features are not uniformly non-steerable** in non-clinical LLM domain
3. **First connection between coefficient of variation and causal actionability** - simple statistical measure predicts steering feasibility
4. **First rate-distortion theoretic explanation** for the variance paradox in SAE absorption

#### Prior Work Collision Assessment

| Paper | Overlap | Severity | Manageable? |
|-------|---------|----------|-------------|
| Basu et al. (2026) | Actionability paradox (98.2% AUROC but 0% steering). Proposal claims heterogeneity, not universal failure. | **exact_match_framework** | Yes - with domain-specificity framing |
| Chanin et al. (2024) | Absorption metric but no CV-steering connection | related_work | N/A |
| Cui et al. (2026) | Information-theoretic impossibility. Cited as constraint, not collision. | related_work | N/A |
| Karvonen et al. (2025) | SAEBench probe projection. No CV-steering analysis exists. | related_work | N/A |
| Templeton et al. (2024) | Absorption as phenomenon. No CV-steering connection. | related_work | N/A |
| Bricken et al. (2023) | SAE feature analysis. No steering heterogeneity. | related_work | N/A |
| Shannon (1948) | Rate-distortion theory foundation. Novel application to SAE absorption. | related_work | N/A |
| Pearl (2009) | Causal mediation framework. Novel application to SAE steering routing. | related_work | N/A |

#### Basu et al. Collision Analysis

**The collision is significant but manageable**:

- **Basu et al. (2026)** demonstrates that good detection (AUROC) does not guarantee steering utility (actionability paradox)
- **The proposal does NOT claim to resolve the actionability paradox universally** - this is critical
- **The proposal claims**: CV identifies a subpopulation of absorbed features (high-CV) that ARE steerable in non-clinical LLM domain
- **Key differentiation**: Basu et al. studied clinical features (predominantly low-CV per the proposal's hypothesis). This proposal studies non-clinical LLM features and finds the high-CV subset is steerable

**If the field asks**: "Why isn't this just confirming Basu et al.?"

**Answer**: Basu et al. showed universal failure in clinical domain. We show the failure is NOT universal in non-clinical LLM domain - there exists a high-CV subpopulation that IS steerable. This is an extension, not a contradiction.

#### What is Genuinely Novel

1. **CV as predictor for steering**: No prior work found that uses coefficient of variation to predict steering effectiveness. This is the core novel contribution.

2. **Subpopulation decomposition**: The claim that absorbed features are not uniformly non-steerable (in non-clinical domain) is genuinely novel. Basu et al. showed 0% steering on clinical features. We show some steering on non-clinical features with high-CV.

3. **Rate-distortion application**: While rate-distortion theory is established (Shannon 1948), applying it to explain the variance paradox (733x CV ratio) in SAE absorption is novel.

4. **Bypass/mediated regime routing**: While causal mediation framework is established (Pearl 2009), applying it to explain differential steering effectiveness in absorbed features is novel.

#### Critical Concerns

1. **CV threshold is still prospective**: Median CV split is better than post-hoc CV > 1.0, but still needs validation on held-out features.

2. **Steering effects are small**: 0.153 vs 0.075 logit change. The contrarian magnitude critique is valid. Must frame as "small effect vs zero effect" not "CV works vs doesn't work".

3. **Mechanistic hypothesis is speculative**: Bypass vs. mediated regime routing explanation is compelling but not validated. Present as hypothesis, not fact.

4. **lambda_c instability**: 10x pilot-to-full shift undermines critical point reliability. Critical for phase transition framing but less important now that rate-distortion is primary.

5. **Pilot evidence is preliminary**: 30 high-CV vs 30 low-CV features, 2x effect size needs full validation.

#### Recommendation: **PROCEED** with addressable concerns

---

### 2. backup_projection: SAEBench Cross-Layer Absorption

**Novelty Score: 7/10**

#### Core Novelty Claims

- Cross-layer absorption at critical sparsity using SAEBench probe projection metric
- No ablation required (works across all layers)
- Addresses H3 falsification by using lambda_c=5e-5

#### Prior Work Collision Assessment

| Paper | Overlap | Severity |
|-------|---------|----------|
| Karvonen et al. (2025) SAEBench | Probe projection methodology; candidate extends to cross-layer at lambda_c | related_work |
| Chanin et al. (2024) "A is for Absorption" | Cross-layer absorption via ablation (limited to early layers) | partial_overlap |

#### Assessment

- SAEBench provides methodology but not specific cross-layer absorption at lambda_c
- If absorption variation across layers is found at lambda_c, this extends prior work
- **Publishable in either direction**: Variation found or not found (negative result is also valuable)

#### Recommendation: **PROCEED**

---

### 3. backup_steering: Steering Effectiveness Analysis

**Novelty Score: 5/10**

#### Core Novelty Claims

- Extends Basu et al. to non-clinical domain
- Tests CV-based hypothesis for actionability failure

#### Prior Work Collision Assessment

| Paper | Overlap | Severity |
|-------|---------|----------|
| Basu et al. (2026) | Directly establishes actionability paradox | **exact_match_framework** |
| Conmy et al. (2024) | Ablation-based methodology | related_work |

#### Assessment

- This backup directly asks whether actionability paradox applies universally
- If answer is "yes, universal failure" = negative result confirming Basu et al.
- If answer is "no, heterogeneity exists" = confirms front-runner
- **The CV-based mechanism hypothesis saves it from being a pure duplicate**

#### Recommendation: **MODIFY TO DIFFERENTIATE**

Only proceed if CV-based mechanism hypothesis is clearly the focus, not just reproducing Basu et al. Consider reducing scope to 15+15 features if computational cost is concern.

---

### 4. backup_cross_arch: Cross-Architecture Phase Transition Validation

**Novelty Score: 7/10**

#### Core Novelty Claims

- Finite-size scaling (nu=3) generalizes to Gemma-2-2B
- Cross-architecture validation of critical exponent
- Tests artifact hypothesis: are phase transitions GPT-2/TopK-specific or universal?

#### Prior Work Collision Assessment

| Paper | Overlap | Severity |
|-------|---------|----------|
| Engel & Van den Broeck (2001) | Phase transitions in neural networks (established technique) | related_work |
| Lieberum et al. (2024) GemmaScope | Provides Gemma-2-2B JumpReLU SAEs infrastructure | related_work |

#### Assessment

- Testing whether nu=3 is universal across architectures is genuinely novel
- Phase transitions in neural networks are established via statistical physics, but specific application to SAE absorption is novel
- If nu differs significantly, architecture-dependence is itself publishable
- **Publishable in either direction**

#### Recommendation: **PROCEED**

---

## Summary of Changes from Prior Report

1. **Updated candidate structure** to match current proposal
2. **Added Shannon (1948) and Pearl (2009)** as related work foundations
3. **Clarified rate-distortion theory as PRIMARY explanation** (replacing phase transitions as primary)
4. **Acknowledged median CV split** is prospective but still needs held-out validation
5. **Added concerns about small absolute steering effects** and lambda_c instability
6. **Overall novelty remains HIGH** (no candidate below 5/10)

---

## Critical Issues for Synthesizer

1. **CV threshold (median split) needs prospective validation** - should be validated on held-out features or justified theoretically

2. **Basu et al. collision is significant** - field will ask why this isn't just confirming their result; domain-specificity framing is essential

3. **Mechanistic hypothesis is speculative** - bypass vs. mediated regime should be presented as hypothesis, not established fact

4. **Steering effects are small in absolute terms** - must frame as "small effect vs zero effect" not "CV works vs doesn't work"

5. **lambda_c instability acknowledged** - critical point not fully reliable, phase transition framing is now supporting context only

---

## References

- Basu et al. (2026): Interpretability without Actionability - actionability paradox
- Chanin et al. (2024): A is for Absorption - absorption detection
- Cui et al. (2026): On the Limits of SAEs - information-theoretic impossibility
- Karvonen et al. (2025): SAEBench - probe projection metric
- Templeton et al. (2024): Anthropic SAE paper - absorption as phenomenon
- Bricken et al. (2023): Towards Monosemanticity - SAE feature analysis
- Conmy et al. (2024): Activation Patching in Superposition - ablation methodology
- Costa et al. (2025): MP-SAE - hierarchical feature recovery
- Engel & Van den Broeck (2001): Statistical Mechanics of Learning - phase transitions
- Lieberum et al. (2024): GemmaScope - Gemma-2-2B JumpReLU SAEs
- Shannon (1948): Mathematical Theory of Communication - rate-distortion foundation
- Pearl (2009): Causality - causal mediation framework