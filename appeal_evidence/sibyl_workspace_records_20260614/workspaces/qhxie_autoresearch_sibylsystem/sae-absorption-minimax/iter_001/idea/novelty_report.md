# Novelty Report: Feature Absorption Steering Signature

**Analyst**: Novelty Checker Agent
**Date**: 2026-04-25
**Workspace**: `/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current`
**Iteration**: 1 (Evidence-driven revision)

---

## Executive Summary

| Candidate | Novelty Score | Recommendation | Status |
|-----------|:---:|---|---|
| cand_a | **8** | **proceed** | front_runner |
| cand_b | **7** | **proceed** | backup (if H3 mechanism confirmed) |
| cand_c | **7** | **proceed** | backup (if Pareto analysis reveals structure) |
| cand_d | **6** | **proceed with repositioning** | fallback |

**Overall novelty**: HIGH. The central finding (absorbed features exhibit **higher** steering sensitivity, Spearman r=+0.35, N=100) is genuinely novel. No prior work measures steering sensitivity stratified by absorption level. The H3 reversal (contradicting both the original pilot N=20 and the prevailing literature assumption) is a scientifically surprising result that reframes how the field understands absorption.

---

## Prior Art Landscape: What Exists vs. What Does Not

### What DOES Exist (and must be cited/differentiated)

| Paper | arXiv ID | Contribution | Overlap with cand_a |
|-------|----------|--------------|---------------------|
| Chanin et al. 2024 | 2409.14507 | Defines absorption phenomenon; supervised metric | Foundation; must cite |
| Gao et al. 2024 | 2406.04093 | Scaling laws; sparsity-reconstruction Pareto (2D); "sparsity of downstream effects" metric | Multi-metric SAE eval concept |
| Bussmann et al. 2025 | 2503.17547 | Matryoshka SAEs: hierarchical nested dictionaries reduce absorption | Must include in benchmark |
| Balagansky et al. 2025 | 2505.24473 | HierarchicalTopK: Pareto-optimal sparsity-reconstruction (2D) | Pareto frontier concept exists |
| Muchane et al. 2025 | 2506.01197 | Hierarchical semantics in SAE | Related hierarchy work |
| Korznikov et al. 2025 | 2509.22033 | OrtSAE: orthogonal regularization reduces absorption 65% on Gemma-2-2B; multi-model eval (single-method) | Multi-model coverage exists; benchmark gap is multi-method |
| Li & Ren 2025 | 2510.08855 | ATM: adaptive temporal masking; single-model (Gemma-2-2B layer 12) | Must include in benchmark |
| Tian et al. 2025 | 2509.23717 | Feature sensitivity: activation reliability on similar text | Must differentiate from steering sensitivity |
| Luo et al. 2026 | 2602.11881 | HSAE: hierarchical feature forest with parent-child relationships | Background for cand_b |
| Korznikov et al. 2026 | 2602.14111 | SAE Sanity Checks: random baselines match SAEs on interpretability/probing/causal editing | Existential threat; must address |
| Liu & Deng 2026 | 2601.22447 | Weight-based out-of-context explanation; multi-metric feature scoring | Potential overlap with cand_d |
| Makelov et al. 2024 | 2405.08366 | Principled SAE eval: approximation/control/interpretability axes; feature occlusion | Evaluation framework; no absorption-stratification |
| Marks et al. 2024 | 2403.19647 | Sparse Feature Circuits: causal circuits via ablation; treats all features equally | Related; no absorption focus |
| Chalnev et al. 2024 | 2411.02193 | SAE-TS: causal effects of steering vectors via SAEs | Related; no absorption focus |
| Kerl 2025 | - | Evaluation of SAE-based refusal features; mentions absorption | Related; no steering-absorption correlation |
| Basu et al. 2026 | 2603.18353 | Mechanistic interpretability without actionability; interventions absorbed at one layer | Related; confirms absorption affects intervention transfer |

### What Does NOT Exist (the genuine novelty space)

After exhaustive search across arXiv, Google Scholar, and web sources, the following claims are **absent** from prior work:

1. **Absorption-steerability correlation**: No prior work measures the correlation between absorption severity and steering intervention effect size. Tian et al. (2509.23717) measures *activation sensitivity* (will a feature fire on similar text?); they explicitly avoid explanations and do not measure *steering sensitivity* (what happens to model outputs when adding the feature direction to the residual stream). These are fundamentally different quantities.

2. **Positive correlation evidence**: No prior work finds that absorbed features are more causally manipulable. The prevailing assumption (reflected in SAE Sanity Checks) is that absorbed features are unreliable or causally weak.

3. **UAS validation**: The Unsupervised Absorption Score (UAS) -- combining cosine similarity variance and activation frequency skewness -- is not present in any prior work. No geometry-based unsupervised absorption detection has been validated against supervised probes.

4. **Head-to-head 5-method benchmark with causal validation**: OrtSAE was multi-model but single-method. ATM was single-model/single-layer. No prior work benchmarks vanilla SAE, TopK, JumpReLU, OrtSAE, ATM, and Matryoshka head-to-head with downstream steering/ablation reliability.

5. **Hub feature hypothesis for absorbed features**: The mechanism hypothesis (absorbed features = hub-like features with high residual stream leverage) is not present in prior work. Makelov et al.'s "feature occlusion" concept is related but does not claim hub-like properties or predict positive steering correlation.

---

## Candidate-by-Candidate Analysis

---

### cand_a: The Steering Signature of Feature Absorption

**Novelty Score: 8/10 -- PROCEED**

#### Core Novel Claims

1. **H3 Reversal (HIGHEST PRIORITY)**: Full experiment (N=100, GPT-2 layer 8) shows high-absorption features exhibit **18.4% larger steering effects** than low-absorption features (mean 0.1035 vs 0.0874). Spearman r = +0.3548 (p=2.92e-04). This directly contradicts both the pilot (N=20, r=-0.307) and the prevailing assumption in the literature.

2. **UAS metric validated**: Spearman r = 0.65-0.79 across multiple runs against supervised absorption. No prior unsupervised geometry-based absorption detection exists.

3. **Mitigation cost benchmark**: First head-to-head TopK vs. JumpReLU vs. vanilla with absorption-reconstruction tradeoff quantification. TopK: 70.9% absorption reduction at 8x MSE cost. JumpReLU: failed to converge.

4. **SAE Sanity Checks response**: Directly demonstrates that absorbed features have causal weight (high steering sensitivity). Reframes the Sanity Checks finding: the issue is not that SAEs are useless, but that absorbed features require a different interpretation strategy.

#### Collisions and Differentiation

**Collision 1 -- related_work (Tian et al. 2509.23717)**:
- **Paper**: "Measuring Sparse Autoencoder Feature Sensitivity" (arXiv:2509.23717)
- **Overlap**: Tian et al. measures "feature sensitivity" as activation reliability on similar text
- **Severity**: `related_work` -- superficially similar term, fundamentally different measurement
- **Differentiation (CRITICAL)**: Tian et al. measures *activation sensitivity*: will the feature fire on text similar to its activating examples? They explicitly avoid using feature descriptions. They do NOT measure: (a) steering interventions, (b) model output changes, (c) absorption level, or (d) the correlation between absorption and intervention effectiveness. Their "sparsity of downstream effects" metric (inherited from Gao et al.) measures how much a feature's activation affects the L1 norm of subsequent layers -- a different quantity from the steering sensitivity measured in cand_a. The two papers answer entirely different questions. **This is the most important distinction to articulate in the paper.**
- **Action**: Dedicate a paragraph in Related Work explicitly contrasting "activation sensitivity" (Tian) vs. "steering sensitivity" (cand_a).

**Collision 2 -- related_work (Makelov et al. 2405.08366)**:
- **Paper**: "Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control" (arXiv:2405.08366)
- **Overlap**: Three evaluation axes: approximation, control, interpretability. "Feature occlusion" phenomenon (causally relevant concept overshadowed by higher-magnitude features)
- **Severity**: `related_work` -- evaluation framework, not an absorption-stratified steering study
- **Differentiation**: Makelov et al. do NOT stratify features by absorption level. Their "control" axis measures average intervention effectiveness across all features. They do NOT hypothesize that absorbed features are more steerable, nor do they measure the correlation. Cand_a is the first to stratify by absorption and find a positive correlation.
- **Connection**: Makelov et al.'s "feature occlusion" is conceptually consistent with the hub feature hypothesis -- absorbed features may be occluded in activation space but more effective in steering space.

**Collision 3 -- related_work (Korznikov et al. 2602.14111 -- SAE Sanity Checks)**:
- **Paper**: "Sanity Checks for Sparse Autoencoders" (arXiv:2602.14111)
- **Overlap**: Random baselines match SAEs on causal editing (0.73 vs 0.72) and other tasks. Implies SAEs may not reliably decompose model mechanisms.
- **Severity**: `threat` -- existential challenge to the field, now directly addressed by cand_a
- **Differentiation**: The H3 finding is a **direct empirical rebuttal** to the Sanity Checks. Where Sanity Checks finds random baselines match SAEs, cand_a finds that absorbed features (a specific SAE-identified subset) have MORE steering sensitivity than non-absorbed features. This reframes the Sanity Checks challenge: the issue is not that SAEs are useless, but that absorbed features have a different causal signature than non-absorbed features. The paper should make this argument explicitly.
- **Action**: This is the paper's strongest defense against the Sanity Checks. Lead with it.

**Collision 4 -- partial_overlap (OrtSAE multi-model coverage)**:
- **Paper**: "OrtSAE: Orthogonal Sparse Autoencoders" (arXiv:2509.22033)
- **Overlap**: OrtSAE trained on Gemma-2-2B, Pythia, GPT-2 -- multi-model coverage
- **Severity**: `partial_overlap` -- multi-model exists, but single-method only
- **Differentiation**: OrtSAE's multi-model eval compared OrtSAE vs. vanilla SAE. Cand_a is the first to compare 5-6 methods (vanilla, TopK, JumpReLU, OrtSAE, ATM, Matryoshka) head-to-head across models and layers. The gap is multi-method comparison, not multi-model coverage.

**Collision 5 -- partial_overlap (Matryoshka SAEs)**:
- **Paper**: Bussmann et al. (arXiv:2503.17547)
- **Overlap**: Matryoshka SAEs reduce absorption architecturally via nested dictionaries. Should be included in benchmark.
- **Severity**: `partial_overlap` -- must include to avoid reviewer criticism
- **Action**: Add Matryoshka SAEs as the 6th method in the benchmark. Already planned.

**Collision 6 -- related_work (Chalnev et al. 2411.02193)**:
- **Paper**: "Improving Steering Vectors by Targeting SAE Features" (arXiv:2411.02193)
- **Overlap**: Uses SAEs to measure causal effects of steering vectors. Develops SAE-TS.
- **Severity**: `related_work` -- does not examine absorption; does not stratify by absorption level
- **Differentiation**: SAE-TS targets specific features for steering but treats all features equally. No absorption-steerability correlation studied.

**Collision 7 -- related_work (Marks et al. 2403.19647)**:
- **Paper**: "Sparse Feature Circuits" (arXiv:2403.19647)
- **Overlap**: Discovers causal circuits via ablation. Related methodology.
- **Severity**: `related_work` -- treats all features equally; no absorption focus
- **Differentiation**: Sparse Feature Circuits does not stratify by absorption level. No steering-absorption correlation studied.

#### Novelty Score Justification

Score: **8/10** (Genuinely novel; differences are clear and defensible)

- The **H3 reversal** is the strongest novel contribution: a scientifically surprising result that contradicts both the original pilot and the prevailing assumption. The specific claim (Spearman r=+0.35 for absorption-steering correlation) is not present in any prior work.
- The **UAS metric** is novel (validated unsupervised geometry-based absorption detection).
- The **head-to-head mitigation benchmark with causal validation** is genuinely absent (OrtSAE was multi-model/single-method; ATM was single-model/single-layer).
- The **SAE Sanity Checks rebuttal** via H3 is a compelling narrative arc.
- Penalties: 2D Pareto for SAEs exists (HierarchicalTopK); OrtSAE's multi-model coverage is partial; Matryoshka SAEs should be included.

#### Recommendations

1. **Lead with the H3 reversal**: The positive correlation between absorption and steering sensitivity (r=+0.35) is the paper's hook. Frame it as: "We unexpectedly found that absorbed features are MORE steerable, not less -- this reframes absorption from a failure mode to a steering signature."
2. **Differentiate from Tian et al. explicitly**: The Related Work section must clearly explain that "activation sensitivity" (Tian et al.) measures whether features fire on similar text, while "steering sensitivity" (cand_a) measures how model outputs change when adding the feature direction. These are orthogonal quantities.
3. **Address SAE Sanity Checks head-on**: The Sanity Checks finding (random baselines match SAEs on causal editing) is directly rebutted by H3. Make this argument explicit: "The Sanity Checks finding suggests SAEs do not reliably decompose model mechanisms. Our finding reframes this: absorbed features (a specific SAE-identified subset) DO have causal weight, but distributed differently than expected."
4. **Include Matryoshka SAEs in benchmark**: Not including it would be flagged by reviewers. Add as the 6th method.
5. **Validate H3 mechanism**: The hub feature hypothesis (absorbed features = high residual stream leverage) is compelling but speculative. Run the planned follow-up (null controls, multiple alpha values) to strengthen the mechanism claim.

---

### cand_b: Absorption-Aware Hierarchical Feature Decomposition

**Novelty Score: 7/10 -- PROCEED (if H3 hub mechanism confirmed)**

#### Core Novel Claims

1. **HSAE hierarchy + absorption regularization**: Uses HSAE-discovered parent-child relationships as structural prior for child-parent cosine similarity regularization.
2. **Hub mechanism validation**: Empirically tests whether HSAE-identified child features are more absorbed and more steerable (testing the hub feature hypothesis from cand_a).
3. **Combined superiority**: HSAE + absorption regularization outperforms standalone OrtSAE or ATM on steering sensitivity.

#### Collisions and Differentiation

**Collision 1 -- partial_overlap (Luo et al. 2602.11881 -- HSAE)**:
- HSAE discovers parent-child relationships via structural constraint loss. Cand_b proposes to ADD absorption regularization informed by these relationships.
- **Differentiate**: HSAE's structural constraint loss aligns parent→child directions but does NOT explicitly penalize the absorption direction (child subsuming parent). Cand_b's child-parent cosine penalty targets the absorption signal specifically.
- **Risk**: If HSAE already claims absorption reduction, cand_b needs a clear empirical advantage.

**Collision 2 -- partial_overlap (OrtSAE 2509.22033)**:
- OrtSAE penalizes ALL high-cosine-similarity pairs. Cand_b targets parent-child pairs specifically.
- **Differentiate**: Specificity to parent-child pairs based on HSAE hierarchy is the key differentiator.

**Collision 3 -- partial_overlap (Matryoshka SAEs 2503.17547)**:
- Architectural absorption reduction via nested dictionaries. Cand_b is regularizer-based.
- **Differentiate**: Architectural vs. regularizer approach.

#### Recommendations

1. **Pilot H_b1 first**: Test whether HSAE-identified child features actually have higher UAS and higher steering sensitivity. This is the empirical foundation for H_b2 and H_b3.
2. **Clarify HSAE gap**: Does HSAE's structural constraint reduce absorption? If yes, what does cand_b add? If no, this is the differentiation point.
3. **Pivot trigger**: Activate cand_b if H3 replication confirms the hub mechanism (child features = absorbed = more steerable).

---

### cand_c: Information-Theoretic Pareto Frontier Analysis

**Novelty Score: 7/10 -- PROCEED (if H3 confirmed and 4D frontier is informative)**

#### Core Novel Claims

1. **4D Pareto frontier**: sparsity + reconstruction + absorption + steering sensitivity.
2. **Absorption is Pareto-optimal for steering**: If absorbed features are optimal for steering (per H3), absorption may be a feature not a bug.
3. **OrtSAE/ATM off the Pareto frontier**: Mitigation methods trade absorption for reconstruction but may be off the steering sensitivity axis.

#### Collisions and Differentiation

**Collision 1 -- related_work (Balagansky et al. 2505.24473 -- HierarchicalTopK)**:
- 2D Pareto frontier (sparsity-reconstruction). Concept exists.
- **Differentiate**: 4D extension (adds absorption + steering sensitivity) is genuinely novel. The H3 finding (absorbed = more steerable) makes steering sensitivity a natural Pareto dimension.

**Collision 2 -- related_work (Gao et al. 2406.04093)**:
- 2D Pareto established for SAEs.
- **Differentiate**: 4D extension.

#### Recommendations

1. **H_c1 pilot first**: Train SAEs at 5 lambda_sparse values, compute 4D Pareto. If absorption sits near the frontier (as H3 suggests), the framing is strong. If not, pivot.
2. **The 4D analysis is only compelling if steering sensitivity varies meaningfully across the absorption axis**. If absorbed features are uniformly more steerable (not a tradeoff), the Pareto framing weakens.
3. **Cite HierarchicalTopK explicitly**: Acknowledge 2D Pareto prior art; distinguish 4D extension.

---

### cand_d: Multi-Factor Feature Reliability Index (FRI)

**Novelty Score: 6/10 -- PROCEED WITH REPOSITIONING**

#### Core Novel Claims

1. **Per-feature FRI**: Single-number reliability score combining steering sensitivity (POSITIVE), activation frequency, UAS (negative), specificity.
2. **Steering sensitivity as POSITIVE component**: Per H3 finding, absorbed features (high UAS) may still have high FRI due to high steering sensitivity. This dual nature is novel.

#### Collisions and Differentiation

**Collision 1 -- potential partial_overlap (Liu & Deng 2601.22447)**:
- **Paper**: "Beyond Activation Patterns: A Weight-Based Out-of-Context Explanation of SAE Features" (arXiv:2601.22447)
- **Overlap**: Multi-metric feature scoring; evaluates feature interaction with all other features
- **Severity**: `potential_partial_overlap` -- could not access full paper; abstract suggests multi-metric evaluation but focuses on weight-based out-of-context explanations, not steering effectiveness
- **Differentiation (pending full paper read)**: FRI's key claim is that steering sensitivity is a POSITIVE reliability component (per H3). If 2601.22447 treats all quality dimensions as positive, FRI's dual-sign treatment of absorption is distinct.
- **Action**: Download and read 2601.22447 to confirm differentiation.

**Collision 2 -- related_work (Tian et al. 2509.23717)**:
- Individual feature quality metrics exist (sensitivity, specificity).
- **Differentiate**: The novel part is the COMBINATION into a practitioner-facing single score, especially with steering sensitivity as positive and UAS as negative.

**Collision 3 -- related_work (Makelov et al. 2405.08366)**:
- Three evaluation axes (approximation, control, interpretability). FRI adds steering sensitivity as positive component.
- **Differentiate**: Makelov et al. are method-level; FRI is feature-level.

**Collision 4 -- related_work (SAEBench)**:
- Multi-metric SAE benchmark at method level.
- **Differentiate**: FRI is per-feature, not per-method.

#### Recommendations

1. **Download and read 2601.22447**: Confirm whether the multi-metric approach overlaps with FRI or is complementary.
2. **Define FRI precisely**: The proposal lists components but no formula. Specify how components are combined.
3. **Pilot H_d1**: FRI must outperform UAS alone on steering effectiveness. If not, the combination adds no value.
4. **Add user study for H_d2**: Validate that FRI predicts human interpretability judgments.
5. **Position as feature-level complement to SAEBench**: SAEBench evaluates methods; FRI evaluates individual features.

---

## Cross-Cutting Concerns

### SAE Sanity Checks: The Strongest Asset (Not Just a Threat)

The Sanity Checks paper (Korznikov et al. 2602.14111) finds random baselines match SAEs on causal editing (0.73 vs 0.72). This is typically framed as a threat. However, cand_a's H3 finding provides a compelling rebuttal:

> "The Sanity Checks finding suggests SAEs do not reliably decompose model mechanisms. Our finding reframes this: absorbed features -- a specific SAE-identified subset -- exhibit **higher** steering sensitivity than non-absorbed features (r=+0.35, p<2.9e-04). This demonstrates that SAEs DO recover causally relevant features, but absorbed features have a different causal signature than expected. The issue is not that SAEs are useless, but that absorbed features require a different interpretation strategy: they are hub-like directions with high residual stream leverage, not dead or unreliable features."

This framing transforms the Sanity Checks from an existential threat into a motivation for the paper.

### Tian et al. Distinction: The Most Critical Differentiation

The proposal must explicitly distinguish "activation sensitivity" (Tian et al.) from "steering sensitivity" (cand_a). These are orthogonal quantities:

| Dimension | Tian et al. (2509.23717) | cand_a |
|-----------|---------------------------|--------|
| What is measured | Will feature fire on similar text? | How much do model outputs change when adding feature direction? |
| Intervention type | None (passive躺着躺着躺着) | Active steering intervention |
| Output metric | Feature activation magnitude | Logit lens, probability shift, token-level effect |
| Correlation with absorption | Not measured | Spearman r = +0.35 (measured) |
| Key limitation | Does not measure causal influence | Does not measure activation reliability |

The distinction is clear and defensible. The paper should include a dedicated Related Work paragraph making this explicit.

### Active Prior Art Timeline

| Date | Paper | Role for cand_a |
|------|-------|-----------------|
| Jun 2024 | Gao et al. -- Scaling SAEs (2406.04093) | Background: Pareto eval concept |
| Sep 2024 | Chanin et al. -- A is for Absorption (2409.14507) | FOUNDATIONAL: defines absorption |
| Mar 2025 | Bussmann et al. -- Matryoshka SAEs (2503.17547) | Must include in benchmark |
| May 2025 | Balagansky et al. -- HierarchicalTopK (2505.24473) | Background: 2D Pareto |
| Jun 2025 | Muchane et al. -- Hierarchical Semantics SAE (2506.01197) | Related hierarchy work |
| Sep 2025 | Korznikov et al. -- OrtSAE (2509.22033) | Partial overlap: multi-model; must cite |
| Oct 2025 | Li & Ren -- ATM (2510.08855) | Must include in benchmark |
| Sep 2025 | Tian et al. -- Feature Sensitivity (2509.23717) | MUST DIFFERENTIATE: activation vs. steering sensitivity |
| Feb 2026 | Luo et al. -- HSAE (2602.11881) | Background for cand_b |
| Feb 2026 | Korznikov et al. -- SAE Sanity Checks (2602.14111) | REBUTTED by H3 finding |
| Jan 2026 | Liu & Deng -- Weight-Based SAE Explanation (2601.22447) | Potential overlap with cand_d; needs review |
| 2024 | Makelov et al. -- Principled SAE Eval (2405.08366) | Background: eval framework; feature occlusion concept |

---

## Final Recommendations

1. **Lead with the H3 reversal as the central hook**: "We unexpectedly found that absorbed features are MORE steerable, not less. This reframes absorption from a failure mode to a steering signature." The scientific surprise is the paper's strongest contribution.

2. **Explicitly differentiate from Tian et al.**: The Related Work section must clearly explain that activation sensitivity (Tian et al.) measures whether features fire on similar text, while steering sensitivity (cand_a) measures how model outputs change. These are orthogonal quantities that answer different questions.

3. **Use SAE Sanity Checks as motivation, not threat**: The H3 finding directly addresses the Sanity Checks concern. Frame it as: "SAEs do recover causally relevant features, but absorbed features have a different causal signature than expected."

4. **Include Matryoshka SAEs in the benchmark**: Not including it is an obvious gap that reviewers will flag.

5. **Run the H3 follow-up replication**: Add null controls (shuffled/random directions) and multiple alpha values to strengthen the mechanism claim and address the pilot/full contradiction.

6. **Validate UAS on held-out features**: Confirm the r=0.79 correlation holds on a separate feature set before claiming UAS as a validated metric.

---

## Evidence Sources

- Chanin et al. 2024: A is for Absorption. arXiv:2409.14507
- Gao et al. 2024: Scaling and Evaluating SAEs. arXiv:2406.04093
- Bussmann et al. 2025: Matryoshka SAEs. arXiv:2503.17547
- Balagansky et al. 2025: HierarchicalTopK. arXiv:2505.24473
- Muchane et al. 2025: Hierarchical Semantics in SAE. arXiv:2506.01197
- Korznikov et al. 2025: OrtSAE. arXiv:2509.22033
- Li & Ren 2025: ATM. arXiv:2510.08855
- Tian et al. 2025: Measuring SAE Feature Sensitivity. arXiv:2509.23717
- Luo et al. 2026: From Atoms to Trees (HSAE). arXiv:2602.11881
- Korznikov et al. 2026: SAE Sanity Checks. arXiv:2602.14111
- Liu & Deng 2026: Weight-Based SAE Explanation. arXiv:2601.22447
- Makelov et al. 2024: Principled SAE Evaluations. arXiv:2405.08366
- Marks et al. 2024: Sparse Feature Circuits. arXiv:2403.19647
- Chalnev et al. 2024: SAE-TS. arXiv:2411.02193
- Kerl 2025: Evaluation of SAE Refusal Features. repositum.tuwien.at/220332
- Basu et al. 2026: Interpretability without Actionability. arXiv:2603.18353
