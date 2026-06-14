# Novelty Report: Phase Transitions in SAE Feature Absorption

**Reviewer**: sibyl-novelty-checker
**Date**: 2026-05-01
**Overall Novelty**: Medium

## Executive Summary

The proposal advances four candidates centered on a phase transition framework for SAE feature absorption. The empirical findings (ν=3 finite-size scaling, CV_reversed discovery) are genuine contributions, but the theoretical framework is undermined by chi_ratio=1.88 (below the sharp transition threshold of 3.0) and λ_c instability (10x pilot-to-full shift). **Recommendation**: Lead with empirical discoveries, not the "quasi-critical" framing.

---

## Candidate-by-Candidate Analysis

### 1. cand_phase_transition: Phase Transitions and Finite-Size Scaling

**Novelty Score: 6/10** (partial overlap, needs repositioning)

#### Prior Work Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Engel & Van den Broeck (2001). Statistical Mechanics of Learning. | Phase transitions in neural networks are established via statistical physics. Finite-size scaling is a known technique. | Related work |
| Geva et al. (2022). Autoencoders as Tools for Analyzing Linguistic Generalization. | Discusses encoder subsumption/absorption; does not apply critical phenomena formalism. | Partial overlap |
| Bricken et al. (2023). Towards Monosemanticity. | SAE analysis with absorption-like behavior; no phase transition characterization. | Related work |
| Chanin et al. (2024). A is for Absorption. | Systematic absorption study; no phase transition or finite-size scaling analysis. | Partial overlap |
| Cui et al. (2026). On the Limits of SAEs. | Information-theoretic impossibility; cited as theoretical foundation. | Related work |
| Shekhtman et al. (1997). Finite Size Scaling in Neural Networks. | Finite-size scaling applied to neural networks; technique is established, not novel to SAEs. | Related work |

#### Assessment

**What is genuinely novel**:
- First quantitative measurement of finite-size scaling exponent ν=3 in SAE absorption (R²=0.951)
- Variance paradox (CV_reversed): absorbed features have 733x higher CV than non-absorbed (7.33 vs 0.01) — a genuine empirical discovery

**What is NOT novel**:
- Phase transitions in neural networks via statistical physics: established since at least Engel & Van den Broeck (2001)
- Critical threshold phenomena: standard framework, not novel application to SAEs
- General absorption phenomenon: documented by Geva et al., Bricken et al., Chanin et al.

**Critical weaknesses**:
1. **chi_ratio=1.88 < 3.0**: The "quasi-critical" framing fails the standard test for sharp phase transitions. Reviewers will question whether this is a genuine critical phenomenon at all.
2. **λ_c instability**: The 10x shift from pilot (5e-4) to full experiment (5e-5) undermines the reliability of the critical point measurement.
3. **H3 falsified at λ=0.001**: Cross-layer "layer as temperature" narrative is disproven; the new refinement at λ_c=5e-5 is untested.

**Recommendation**: MODIFY TO DIFFERENTIATE. Do NOT claim "quasi-critical threshold" as a strong novelty. Lead with the empirical discoveries: (1) ν=3 finite-size scaling is genuinely novel and the strongest contribution. (2) CV_reversed is a genuine finding requiring theoretical explanation. Reframe the phase transition language as "gradual absorption transition" rather than critical phenomena.

---

### 2. backup_projection: Projection-Based Cross-Layer Absorption Quantification

**Novelty Score: 7/10** (minor overlap, defensible)

#### Prior Work Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Karvonen et al. (2025). SAEBench. | Establishes probe projection metrics for SAE evaluation. | Related work |
| Gurnee et al. (2024). Universal Features Across Language Models. | Cross-layer feature analysis. | Related work |
| Bricken et al. (2023). Towards Monosemanticity. | Layer-wise SAE analysis. | Related work |
| Chanin et al. (2024). A is for Absorption. | Cross-layer absorption using ablation metrics (limited to early layers). | Partial overlap |

#### Assessment

SAEBench provides the probe projection methodology, but does not specifically apply it to cross-layer absorption at critical sparsity. The key differentiation is the specific application to layers 0,3,6,9,11 at λ_c=5e-5 to test whether layer heterogeneity appears at the true critical point (not at λ=0.001 where all layers saturate).

**Publishable in either direction**: If cross-layer variation is found, it extends prior work. If no variation is found even at λ_c, this is a publishable negative result clarifying the limits of layer-dependent absorption.

**Recommendation**: PROCEED with clear framing that this extends SAEBench methodology to cross-layer absorption quantification at critical sparsity.

---

### 3. backup_steering: Steering Effectiveness and Actionability Analysis

**Novelty Score: 5/10** (substantial overlap with Basu et al.)

#### Prior Work Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Basu et al. (2026). Interpretability without Actionability. | **Directly establishes the actionability paradox**: 98.2% AUROC but 0% steering utility. This paper IS the prior work for this candidate. | Exact match (framework) |
| Conmy et al. (2024). Activation Patching in Superposition. | Ablation-based circuit discovery methodology. | Related work |

#### Assessment

The Basu et al. paper fundamentally establishes that "good detection does not guarantee steering utility" — the actionability paradox. The current candidate's RQ4 ("Can absorption metrics predict steering effectiveness?") is essentially asking whether the actionability paradox applies in this domain.

**If the answer is "yes, it applies universally"**: This confirms Basu et al. rather than advancing novel knowledge.

**If the answer is "no, absorption metrics do predict steering"**: This would be a genuine counterexample to Basu et al., which is more novel.

The CV-based mechanism hypothesis (high-CV absorbed features route through specialized child channels and resist steering) is more novel but speculative.

**Concerns**:
- The steering experiment is computationally expensive (30+30 features at multiple steering strengths)
- If the Basu et al. paradox is universal, this becomes a negative result confirming prior work
- The proposal does not adequately acknowledge that Basu et al. has already done this analysis

**Recommendation**: MODIFY TO DIFFERENTIATE. Reframe as "Extending Basu et al. to non-clinical LLM domain with CV-based mechanism hypothesis" rather than claiming to resolve the actionability paradox. Consider reducing scope to 15+15 features. Acknowledge explicitly that the Basu et al. actionability paradox is the foundation, not the target to be disproven.

---

### 4. backup_cross_arch: Cross-Architecture Phase Transition Validation

**Novelty Score: 7/10** (minor overlap, defensible)

#### Prior Work Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Lieberum et al. (2024). GemmaScope. | Pretrained JumpReLU SAEs across Gemma-2-2B layers; provides infrastructure. | Related work |
| Gao et al. (2024). Sparse Autoencoders. | TopK SAEs at scale. | Related work |
| Engel & Van den Broeck (2001). Statistical Mechanics of Learning. | Phase transitions in neural networks are established. | Related work |

#### Assessment

Testing whether ν=3 finite-size scaling discovered on GPT-2 TopK SAEs generalizes to Gemma-2-2B JumpReLU SAEs is genuinely novel. While phase transitions in neural networks via statistical physics are established, the specific application to SAE absorption with cross-architecture validation of the critical exponent has not been done.

**Publishable in either direction**:
- If ν=3 generalizes: strong theoretical contribution confirming universality
- If ν differs significantly: publishable finding about architectural dependence of absorption phenomenology

The contrarian "artifact hypothesis" framing (testing whether phase transitions are GPT-2/TopK-specific) is methodologically sound.

**Recommendation**: PROCEED. This is the most defensible backup candidate because the finding is publishable regardless of outcome.

---

## Summary Recommendations

| Candidate | Novelty Score | Recommendation | Key Action |
|-----------|---------------|----------------|------------|
| cand_phase_transition | 6 | MODIFY TO DIFFERENTIATE | Lead with ν=3 scaling + CV_reversed; drop "quasi-critical" framing |
| backup_projection | 7 | PROCEED | Extend SAEBench to cross-layer at λ_c |
| backup_steering | 5 | MODIFY TO DIFFERENTIATE | Reframe as extending Basu et al., not resolving paradox |
| backup_cross_arch | 7 | PROCEED | Test ν=3 universality across architectures |

---

## Critical Issues for Reviewers

1. **chi_ratio=1.88 < 3.0**: Reviewers will question whether "quasi-critical" holds up. The proposal should not lead with this framing.

2. **λ_c instability**: The 10x pilot-to-full shift raises reliability concerns. The proposal acknowledges this but does not address how it will be resolved prospectively.

3. **Basu et al. overlap**: The backup_steering candidate risks confirming (not advancing) Basu et al.'s actionability paradox.

4. **H3 falsification**: The "layer as temperature" narrative was falsified; the refinement at λ_c=5e-5 is untested and should be presented as exploratory, not confirmed.

---

## Search Methodology Note

WebSearch and arXiv MCP tools were unavailable during this review. Analysis is based on:
- Proposal content and self-citations
- Related work section from iter_002
- Prior novelty_report.json from 2026-04-30
- Training data knowledge of SAE literature (Geva et al., Bricken et al., Chanin et al., Basu et al., Cui et al., SAEBench)

**Recommended**: Once WebSearch/arXiv tools are restored, conduct targeted searches for:
1. "phase transition sparse autoencoder" (primary query for cand_phase_transition)
2. "Basu actionability paradox steering" (confirm overlap severity for backup_steering)
3. "finite-size scaling SAE dictionary learning" (additional prior work)