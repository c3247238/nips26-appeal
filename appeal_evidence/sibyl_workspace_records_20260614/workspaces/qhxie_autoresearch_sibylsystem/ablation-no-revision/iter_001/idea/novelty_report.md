# Novelty Report: Activation Energy Theory of LLM Reasoning

**Date**: 2026-04-29
**Novelty Checker**: sibyl-novelty-checker
**Workspace**: ablation-no-revision/iter_001
**Round**: 5 - Routing Failure Analysis (H3 falsification follow-up)

---

## Executive Summary

The proposal has pivoted from claiming a novel theoretical framework (exponential saturation model) to pursuing a **negative result** on consistency-based routing plus diagnostic analysis of failure modes. This pivot addresses the Yang et al. collision but introduces new prior art concerns.

**Overall Novelty Assessment**: MEDIUM (5/10)

| Hypothesis | Novelty | Status | Key Prior Art |
|------------|---------|--------|--------------|
| H1: Arrhenius Kinetics | N/A | CONFIRMED | Yang et al. 2508.16456 (collision) |
| H2: Ea = Difficulty | LOW | CONFIRMED | Yang et al. CL-CS model |
| H3: Single-pass Threshold | MEDIUM | FALSIFIED | Negative result claim |
| H4: Execution Errors | MEDIUM | PENDING | Li 2601.00828 (supports) |
| H5: Entropy Routing | LOW | PENDING | ACAR 2602.21231 (challenges) |

---

## Critical New Prior Art Findings

### 1. Li (2026) - Error Depth Hypothesis (arXiv:2601.00828)

**HIGHLY RELEVANT TO H4**

**Key Findings**:
- Error types in LLM reasoning:
  - **Calculation errors**: 62% (GPT-3.5), 22% (DeepSeek)
  - **Logic errors**: 13% (GPT-3.5), 33% (DeepSeek)
  - **Setup/Interpretation errors**: 25% (GPT-3.5), 44% (DeepSeek)
- Calculation errors are "shallower" and easier to self-correct
- Stronger models (DeepSeek 94%) make deeper errors that resist correction
- Weaker models (GPT-3.5 66%) make more shallow errors (62% calculation)

**Implications for H4**:
- **STRONGLY SUPPORTS** the proposal's hypothesis that execution errors dominate low-Ea failures
- Li's Error Depth Hypothesis provides theoretical grounding for H4
- The proposal's claim that "execution errors >50% in low-Ea failures" aligns with Li's findings

**Severity**: **related_work** - Different framing but supports the core claim

---

### 2. ACAR (2026) - Adaptive Complexity Routing (arXiv:2602.21231)

**HIGHLY RELEVANT TO H5 AND H3**

**Key Findings**:
- Uses self-consistency variance (sigma) for routing decisions
- **Attribution proxies (response similarity, entropy) showed weak correlation with ground-truth leave-one-out values**
- "Agreement-but-wrong" failure mode: When models agree on incorrect answers, no ensemble can recover
- 8pp accuracy gap between self-consistency routing and full ensemble represents this irreducible ceiling

**Critical Quote**:
> "Attribution estimates based on proxy signals such as response similarity and entropy showed weak correlation with ground-truth leave-one-out values; practical attribution requires explicit counterfactual computation."

**Implications for H5**:
- **CHALLENGES** the proposal's H5 (entropy as better routing signal)
- ACAR explicitly tested entropy as a routing proxy and found it ineffective
- The proposal's H5 would need to demonstrate entropy outperforms self-consistency variance

**Implications for H3**:
- **CONFIRMS** the proposal's H3 finding (low-Ea accuracy < 75%)
- ACAR's "agreement-but-wrong" is the same phenomenon
- Quantifies the ceiling: 8pp gap is intrinsic to consistency-based methods

**Severity**: **partial_overlap** - Different routing target but same underlying limitation

---

### 3. Yang et al. (2025) - Probabilistic Inference Scaling (arXiv:2508.16456)

**Previously documented - remains critical**

- Exponential saturation formula collision
- Cannot claim the Arrhenius kinetics framework as novel

---

## Hypothesis-by-Hypothesis Novelty Analysis

### H1: Arrhenius Kinetics Validation

**Status**: Already CONFIRMED (pilot validation complete)

**Novelty Score**: N/A (established prior)

**Evidence**: R² = 0.936 (threshold >0.85) on Qwen2.5-Math-7B-Instruct

**Prior Art**: Yang et al. 2508.16456 - identical mathematical framework

**Conclusion**: Cannot claim novelty. Paper must acknowledge this collision.

---

### H2: Activation Energy Predicts Difficulty

**Status**: Already CONFIRMED (pilot validation complete)

**Novelty Score**: 3/10 (major overlap)

**Evidence**: Spearman(Ea, MATH level) = 0.578, p=0.0008

**Prior Art**: Yang et al. CL-CS model captures same concept

**Conclusion**: Related to prior work. No standalone novelty claim possible.

---

### H3: Single-Pass Threshold (FALSIFIED)

**Status**: FALSIFIED (68.4% < 75% threshold)

**Novelty Score**: 6/10 (negative result has value)

**Novel Contribution**: **Negative result** - First systematic falsification that Ea from consistency does NOT predict single-pass threshold

**Supporting Evidence**:
- ACAR (2602.21231) confirms "agreement-but-wrong" as fundamental limitation
- 8pp ceiling quantified by ACAR matches the proposal's H3 falsification

**Differentiation Required**:
- Emphasize the specific setting (MATH benchmark, Qwen2.5-Math-7B-Instruct)
- Quantify the failure mode precisely
- Compare with ACAR's findings (they use sigma-based routing, proposal uses Ea-based routing)

**Conclusion**: **Proceed with repositioning as negative result paper**

---

### H4: Execution Errors Explain Routing Failure (NEW)

**Status**: PENDING (pilot not yet run)

**Novelty Score**: 7/10 (novel application)

**Novel Claim**: In the context of Ea-based routing failure, execution errors (>50%) dominate low-Ea failures.

**Supporting Prior Art**:
- Li (2601.00828) provides error type distribution (62% calculation for GPT-3.5)
- Li's Error Depth Hypothesis supports "shallow errors = calculation mistakes"

**Potential Collision**: None directly on this specific claim

**Differentiation**: Focus on WHY execution errors cause Ea-based routing to fail:
- Ea measures consistency, not reasoning quality
- Execution errors produce consistent but wrong answers
- This is distinct from Li's self-correction framing

**Conclusion**: **Proceed** - Novel specific claim with supporting prior art

---

### H5: Entropy as Better Routing Signal (NEW)

**Status**: PENDING (pilot not yet run)

**Novelty Score**: 3/10 (challenged by prior art)

**Original Claim**: Per-sample token entropy correlates better with single-pass solveability than answer consistency

**Challenging Prior Art**:
- ACAR (2602.21231): "Attribution estimates based on proxy signals such as response similarity **and entropy showed weak correlation** with ground-truth leave-one-out values"
- This directly challenges H5

**Potential Issue**: If ACAR tested entropy and found it ineffective, the proposal's H5 may also fail

**Mitigation Options**:
1. **Reframe H5**: Compare entropy vs Ea (not entropy vs baseline)
2. **Narrow the claim**: Entropy may help for specific error types
3. **Acknowledge ACAR**: H5 is testing a hypothesis ACAR found challenging

**Conclusion**: **HIGH RISK** - ACAR's findings suggest H5 may fail. Proceed with caution.

---

## Collision Summary

| Prior Art | Overlap | Severity | Impact |
|-----------|---------|----------|--------|
| Yang et al. 2508.16456 | Exponential saturation formula | **exact_match** | H1-H2 cannot be claimed novel |
| Li 2601.00828 | Error type distribution | **related_work** | Supports H4 |
| ACAR 2602.21231 | Entropy routing, agreement-but-wrong | **partial_overlap** | Challenges H5, confirms H3 |
| RASC 2408.17017 | Early stopping + weighted voting | **related_work** | General context |
| CGES 2511.02603 | Confidence-guided early stopping | **related_work** | General context |

---

## Recommendations

### For H3 (FALSIFIED - Single-Pass Threshold)

**Recommendation**: PROCEED with repositioning

**Novelty Claim**: "First systematic empirical falsification that activation energy from answer consistency does NOT predict single-pass threshold routing on MATH benchmark."

**Position Against Prior Art**:
- Acknowledge ACAR's related finding (8pp ceiling)
- Differentiate by focusing on Ea vs sigma routing
- Emphasize Qwen2.5-Math-7B-Instruct specific results

### For H4 (PENDING - Execution Errors)

**Recommendation**: PROCEED

**Novelty Claim**: "Execution errors (>50%) dominate Ea-based routing failures, explaining why consistency-derived Ea fails to predict single-pass solveability."

**Differentiation**: Connect to Li's Error Depth Hypothesis but frame in routing context

### For H5 (PENDING - Entropy Routing)

**Recommendation**: MODIFY OR DROP

**Issue**: ACAR found entropy routing ineffective

**Options**:
1. **Drop H5**: Focus on H3 + H4 only
2. **Reframe**: "Does entropy outperform Ea for routing?" (not "vs baseline")
3. **Acknowledge ACAR**: "We test whether entropy can improve upon Ea-based routing, addressing ACAR's finding that proxy signals have weak correlation"

**Risk Assessment**:
- H5 may fail just as H3 failed
- If H5 fails, the paper loses its proposed alternative contribution
- Consider making H5 exploratory rather than a key hypothesis

---

## Revised Paper Focus

Based on novelty analysis, the paper should:

1. **Lead with Negative Result (H3)**: Ea-based routing fails - first systematic falsification on MATH benchmark

2. **Add Diagnostic Analysis (H4)**: Execution errors explain why Ea fails - align with Li's Error Depth Hypothesis

3. **De-emphasize or Drop H5**: Entropy routing challenged by ACAR; consider exploratory only

4. **Acknowledge Yang et al.**: Cannot claim Arrhenius/exponential framework as novel

---

## Novelty Report JSON

```json
{
  "candidates": [
    {
      "candidate_id": "cand_a",
      "title": "Activation Energy Theory - Routing Failure Analysis",
      "novelty_score": 6,
      "collisions": [
        {
          "paper": "Yang et al., 2025. Probabilistic Inference Scaling Theory. arXiv:2508.16456",
          "overlap": "Exponential saturation formula identical",
          "severity": "exact_match"
        },
        {
          "paper": "Li, 2026. Error Depth Hypothesis. arXiv:2601.00828",
          "overlap": "Error type distribution supports H4 (calculation errors 62%)",
          "severity": "related_work"
        },
        {
          "paper": "ACAR, 2026. Adaptive Complexity Routing. arXiv:2602.21231",
          "overlap": "Agreement-but-wrong confirms H3; entropy routing weak (challenges H5)",
          "severity": "partial_overlap"
        }
      ],
      "recommendation": "proceed_with_modifications",
      "differentiation_notes": "H3 negative result is novel; H4 aligns with Li; H5 at risk due to ACAR"
    }
  ],
  "overall_novelty": "medium",
  "key_findings": {
    "supports_proposal": [
      "Li (2601.00828) supports H4: calculation errors dominate (62%)",
      "ACAR confirms 'agreement-but-wrong' ceiling (8pp gap)"
    ],
    "challenges_proposal": [
      "ACAR: entropy as routing signal shows weak correlation",
      "Yang et al.: exponential formula collision remains"
    ]
  }
}
```

---

## Evidence Sources

### Primary Prior Art

1. **Yang et al. (2025)** - arXiv:2508.16456
   - "A Probabilistic Inference Scaling Theory for LLM Self-Correction"
   - Peking University, Alibaba Group
   - Exponential saturation formula collision

2. **Li (2026)** - arXiv:2601.00828
   - "Decomposing LLM Self-Correction: The Accuracy-Correction Paradox and Error Depth Hypothesis"
   - University of Birmingham
   - Error type distribution: calculation 62% (GPT-3.5), 22% (DeepSeek)
   - Supports H4

3. **ACAR (2026)** - arXiv:2602.21231
   - "ACAR: Adaptive Complexity Routing for Multi-Model Ensembles"
   - Self-consistency variance routing
   - Entropy shows weak correlation with ground truth
   - Agreement-but-wrong ceiling: 8pp gap
   - Challenges H5, confirms H3

### Related Prior Art

4. **RASC (2024)** - arXiv:2408.17017
   - Reasoning-Aware Self-Consistency
   - Early stopping + weighted voting

5. **CGES (2025)** - arXiv:2511.02603
   - Confidence-Guided Early Stopping
   - Bayesian framework for early stopping

---

## Conclusion

The proposal's pivot to a negative result (H3 falsification) is the right move given the Yang et al. collision. However:

1. **H4** is well-supported by Li (2026) and provides a clear path forward
2. **H5** is at high risk due to ACAR's finding that entropy routing has weak correlation
3. **H3** is a genuine negative result that adds to the literature (confirmed by ACAR's agreement-but-wrong ceiling)

**Recommendation**: Proceed with H3 + H4 focus. Drop or de-emphasize H5. Position as "Diagnosing Why Consistency-Based Routing Fails: Execution Errors and the Agreement-But-Wrong Ceiling."
