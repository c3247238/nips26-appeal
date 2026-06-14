# Strategist Analysis: CV-Based Actionability Decomposition

## Executive Summary

**Verdict**: PROCEED with targeted validation experiments before paper submission.

**Rationale**: The CV-steering correlation is a genuine, statistically robust finding (1.47x effect ratio, p < 0.01 across all strengths) that directly addresses a timely open problem (Basu et al. actionability paradox). However, two critical concerns require resolution before publication: (1) post-hoc CV threshold and (2) single-architecture validation. These are addressable within the current project timeline.

**Recommended next 3 experiments** (priority order):
1. **Gemma-2-2B replication** (2.0 GPU hours) - cross-architecture validation
2. **Held-out feature validation** (1.0 GPU hours) - address post-hoc threshold concern
3. **Fano factor control** (1.0 GPU hours) - investigate mechanism / rule out magnitude confound

---

## 1. Signal Strength Assessment

| Finding | Evidence | Signal Strength | Justification |
|---------|----------|-----------------|---------------|
| CV predicts steering heterogeneity | 1.47x ratio, p < 0.01 (all strengths) | **STRONG** | Effect replicated at +3, +5, +7; BH-corrected significant; not an artifact of single steering strength |
| Absorbed high-CV ≈ non-absorbed steering | 0.0975 vs 0.1020 (4.5% diff) | **MODERATE** | Parity is positive for interpretability practice, but measured on different prompts |
| H4: Absorbed have higher CV | 733x ratio (CV=7.33 vs 0.01) | **STRONG** | Unambiguous signal; genuinely surprising (reversed direction) |
| Phase transition with critical λ | chi_ratio=1.88, R²=0.951 | **MODERATE** | Below "sharp transition" threshold of 3.0; framing should be "soft transition" |
| H3/H6 falsified | Uniform saturation, decreasing components | **N/A** | Honest negative results; reported as informative |
| Decoder orthogonality predictor | r = -0.136 (not significant) | **WEAK** | Negative result; rules out orthogonality as mechanism |
| Cross-architecture generalization | Not completed | **PENDING** | Critical missing validation |

**Overall signal assessment**: The primary finding (CV predicts steering heterogeneity) is robust and replicated across multiple steering strengths. The signal is NOT noise.

---

## 2. Opportunity Cost Analysis

| Direction | GPU Cost | Info Gain | Risk | Expected Outcome |
|------------|----------|-----------|------|------------------|
| Gemma-2-2B replication | 2.0 hrs | HIGH - determines if finding generalizes | Medium (may fail on different architecture) | If succeeds: story becomes top-tier; if fails: clear boundary condition |
| Held-out feature validation | 1.0 hrs | MEDIUM - addresses post-hoc threshold concern | Low (pure validation) | If holds: threshold concern resolved; if fails: must reframe as exploratory |
| Fano factor control | 1.0 hrs | MEDIUM - investigates mechanism | Low (measurement) | If CV effect persists: mechanism is CV itself; if disappears: CV is magnitude proxy |
| Test ALL 106 low-CV features | 0.5 hrs | LOW - addresses sampling concern | Low (confirms or refutes) | If confirms current: sampling concern overstated; if different: current estimate biased |
| Cross-layer at λ_c=5e-5 | 2.0 hrs | LOW - H3 retest, low priority given falsification | Medium | Likely still falsified; low information gain at this stage |
| Clinical domain replication | 3.0 hrs | MEDIUM - tests Basu et al. domain hypothesis | High (may not have clinical data) | Speculative; not critical for current submission |

**Ranked by expected information gain per GPU-hour**:
1. Gemma-2 replication (2.0 hrs, HIGH gain)
2. Held-out validation (1.0 hrs, MEDIUM gain)
3. Fano factor control (1.0 hrs, MEDIUM gain)
4. Test all 106 low-CV (0.5 hrs, LOW gain)

---

## 3. Decision Matrix

| Criterion | Gemma-2 Replication | Held-out Validation | Fano Factor Control |
|-----------|--------------------|--------------------|--------------------|
| Signal strength | HIGH (1.47x confirmed) | MEDIUM | MEDIUM |
| GPU cost | 2.0 hrs | 1.0 hrs | 1.0 hrs |
| Failure risk | Medium (architectural differences) | Low | Low |
| Blocking concern resolved | Cross-architecture generalization | Post-hoc threshold | Mechanism unknown |
| Publication impact | HIGH (justifies mid-tier → higher) | MEDIUM (strengthens credibility) | MEDIUM (informs mechanism discussion) |

---

## 4. PIVOT vs PROCEED Verdict

**PROCEED** because:
- At least one hypothesis has STRONG signal: CV-based steering heterogeneity (1.47x, p < 0.01)
- Clear path to publication-quality results: AAAI/EMNLP with honest scope
- The finding directly addresses Basu et al. actionability paradox (field's key question)
- Activation patching confirms genuine causal structure (67.3% mean recovery)
- 4/6 hypotheses confirmed (H1, H2, H4, H5)

**PIVOT if**: Gemma-2 replication fails AND held-out validation fails. This combination would undermine both generalization and threshold validity.

---

## 5. Recommended Experiments (Priority Order)

### Experiment 1: Gemma-2-2B Cross-Architecture Replication
**GPU cost**: 2.0 hours
**Objective**: Test whether CV > 1.0 threshold predicts steering heterogeneity on Gemma-2-2B JumpReLU SAE
**Success criteria**:
- High-CV vs Low-CV shows >30% improvement in steering effect at +5 strength
- Statistical significance at p < 0.05

**If succeeds**: Cross-architecture validation justifies claiming generalization; paper can target ICLR workshop or EMNLP
**If fails**: Clear boundary condition established; paper remains valid for GPT-2 scope with honest generalization caveat

### Experiment 2: Held-Out Feature Validation
**GPU cost**: 1.0 hour
**Objective**: Validate CV threshold on held-out features (split 50/50 before steering)
**Protocol**:
1. Split absorbed features into train/hold-out sets (random 50/50)
2. Compute CV on training set, select threshold
3. Apply threshold to hold-out set WITHOUT seeing steering results
4. Test steering on hold-out features

**Success criteria**:
- High-CV vs Low-CV on hold-out shows similar effect size (within 20% of training)
- Statistical significance at p < 0.05

**If succeeds**: Post-hoc threshold concern resolved; threshold is predictive, not overfitted
**If fails**: Must reframe as exploratory finding; threshold is likely overfitted to training data

### Experiment 3: Fano Factor Mechanism Investigation
**GPU cost**: 1.0 hour
**Objective**: Test whether CV-steering correlation survives magnitude control
**Protocol**:
1. Compute Fano factor (CV²/mean) for all features
2. Use Fano factor as covariate in analysis
3. Test partial correlation of CV with steering controlling for magnitude

**Success criteria**:
- CV-steering correlation persists after controlling for Fano factor
- Fano factor does NOT explain steering (r < 0.3)

**If succeeds**: CV is a genuine predictor, not a magnitude proxy
**If fails**: CV may be proxy for feature magnitude; mechanism is activation variance, not CV per se

---

## 6. Alternative Direction (if all 3 experiments fail)

If Gemma-2 replication, held-out validation, and Fano factor control all fail, pivot to:

**Backup: "Absorbed features are not uniformly non-steerable"** (descriptive claim only)

- Abandon CV threshold claim
- Focus on empirical finding: absorbed high-CV features show steering effects comparable to non-absorbed (0.0975 vs 0.1020)
- Acknowledge mechanism is unknown and CV may be a proxy
- Target a shorter paper / workshop submission with honest scope
- Venue: Workshop paper only

---

## 7. Resource Summary

| Experiment | GPU Hours | Total |
|------------|-----------|-------|
| Gemma-2-2B replication | 2.0 | 2.0 |
| Held-out validation | 1.0 | 3.0 |
| Fano factor control | 1.0 | 4.0 |
| **Total** | **4.0** | |

**Remaining budget**: Project has sufficient budget for all 3 validation experiments within current iteration.

---

## 8. Key Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Gemma-2 shows no CV effect | Medium | Frame as boundary condition; paper still valid for GPT-2 scope |
| Held-out validation fails | Medium | Reframe as exploratory; use continuous CV in analysis |
| Fano factor explains away CV | Medium | Investigate context-sensitivity mechanism (revisionist New H1) |
| Pilot-to-full replication concern (methodologist) | Low | Full experiment is more reliable (5 prompts × 3 strengths vs pilot's 1 strength) |

---

## 9. What This Analysis Concludes

**The core story is real and publishable at AAAI/EMNLP level**, but requires:
1. Cross-architecture validation (Gemma-2) to justify generalization claims
2. Honest acknowledgment of post-hoc threshold limitation (held-out test addresses this)
3. Mechanism discussion (Fano factor investigation)

**The story is NOT**:
- A confirmed universal finding (needs cross-architecture)
- A resolved paradox (only partial evidence that Basu et al. is domain-specific)
- A mechanistic explanation (orthogonality ruled out, mechanism unknown)

**The story IS**:
- First evidence that absorbed features decompose into steerable/non-steerable subpopulations
- CV as a cheap predictor for steering feasibility (no steering experiments needed)
- Statistically robust finding on GPT-2 that warrants cross-architecture validation
- Directly addresses the field's key question (actionability paradox) with novel evidence

---

## 10. Final Recommendation

**PROCEED** with the 3 validation experiments prioritized above. Total GPU investment: ~4 hours. This is the minimum needed to justify the paper's claims before submission to AAAI/EMNLP.

If all 3 succeed: Strengthen paper to potentially target ICLR workshop or EMNLP spotlight.
If Gemma-2 fails but held-out + Fano succeed: Paper remains valid for AAAI/EMNLP with honest boundary condition acknowledgment.
If held-out + Fano fail: Consider pivoting to descriptive claim only.

---

*Strategist Analysis - Result Debate (iter_005)*
*Based on: skeptic.md, optimist.md, methodologist.md, comparativist.md, revisionist.md*
*Key artifacts: exp/results/full_steering_cv.json, exp/results/full/hypothesis_test_summary.json*