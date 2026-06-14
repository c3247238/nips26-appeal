# Strategist Analysis: Result Debate

## Signal Strength Assessment

| Experiment | Signal | Metric Delta | Justification |
|---|---|---|---|
| H_Mech Factorial (FULL) | **STRONG** | Encoder effect 0.843 +/- 0.082 (80x decoder) | Effect size is massive and consistent. The trained encoder drives absorption; decoder contribution is negligible. Even though the original pass criteria (B≈D) fails at L0=32/50 due to decoder disentanglement, the core finding -- encoder as sole driver -- is unambiguous. |
| Multi-seed Validation (FULL) | **STRONG** | Trained 0.477 +/- 0.022 vs Random 0.033 +/- 0.011 | t=36.04, p=3.85e-10. Robust across 5 seeds. Stochastic noise (0.1) successfully resolved the zero-variance concern from pilot. |
| Hierarchy Strength Ablation (FULL) | **STRONG** | Monotonic increase: 0.416 → 0.501 → 0.544 | ANOVA p < 1e-10. Effect is monotonic and significant across all seeds. Dose-response relationship is clean. |
| Held-Out Generalization (FULL) | **STRONG** | Train 0.366 vs Test 0.366, Pearson r=0.998 | Perfect correlation across seed means. Generalization is not just good -- it is near-perfect. |
| L0 Sparsity Ablation (FULL) | **MODERATE** | Inverse monotonic: 0.552 → 0.490 → 0.419 | ANOVA p < 1e-10. Effect is real but direction is opposite of hypothesis. This is a genuine finding, not noise, but it complicates the narrative. |
| H3 Steering (FULL) | **NOISE** | Mean ratio 0.915 +/- 0.396 across 5 seeds | Only 1/5 seeds significant (p<0.05). Direction is inconsistent. The pilot's 1.62x ratio did NOT replicate at full scale. This is a failed replication. |
| H_Safe on GPT-2 SAE (FULL) | **NOISE** | Safety 0.967 vs Non-safety 0.968, p=0.989 | Mann-Whitney p=0.989, effect size ~0. No difference whatsoever. Near-ceiling absorption (~97%) suggests the metric saturates on real SAEs. |

## Opportunity Cost Analysis

| Direction | GPU Hours | Info Gain / GPU-hr | Risk |
|---|---|---|---|
| Real-model absorption on Gemma Scope | ~1.5 | **HIGH** | All evidence is synthetic; field will reject without real-model validation |
| Encoder regularization intervention | ~1.5 | HIGH | Constructive contribution; but needs new training code |
| Re-run H3 with fixed metric | ~0.5 | LOW | Already failed replication; unlikely to recover |
| H_Safe on Gemma Scope | ~1.0 | MEDIUM | Highest novelty, but pilot on GPT-2 showed ceiling effect |
| Deeper hierarchy strength sweep | ~0.5 | LOW | Already confirmed monotonic; diminishing returns |
| Absorption as steerability diagnostic | ~1.0 | MEDIUM | Interesting pivot, but H3 failure undermines premise |

## Decision Matrix

| Direction | Signal | GPU Cost | Risk | Expected Outcome |
|---|---|---|---|---|
| **PROCEED with encoder-driven narrative + real-model validation** | Strong (H_Mech, H1, hierarchy, generalization) | ~2.0 hrs | Medium | Paper with solid mechanism + real-world relevance |
| Pivot to encoder regularization | N/A (constructive) | ~1.5 hrs | Medium | Stronger contribution but higher implementation risk |
| Pivot to absorption-as-diagnostic | Weak (H3 failed) | ~1.0 hrs | High | Unlikely to work; premise contradicted by data |
| Continue H3 / H_Safe on current setup | Noise | ~0.5 hrs | Low | Waste of GPU; results will not improve |

## PIVOT vs PROCEED Verdict: **PROCEED with reframing**

**Criteria check:**
- At least one hypothesis with moderate+ signal? **YES** -- H_Mech (strong), H1 (strong), hierarchy (strong), generalization (strong)
- Clear path to publication-quality results? **YES, but requires real-model validation**

**However, the project must reframe:**

1. **Drop H3 steering as a primary claim.** The full experiment failed replication. The pilot's 1.62x ratio was a false positive. Document as negative result.
2. **Drop H_Safe on GPT-2.** Near-ceiling absorption (~97%) indicates the multi-child proportional metric does not discriminate on real SAEs with d_sae=24576. The feature space is too large; almost everything appears "absorbed" by the metric's definition.
3. **Add real-model validation as mandatory next step.** The entire contribution is currently synthetic (d_model=128, synthetic hierarchy). Without Gemma Scope or similar real SAE evidence, the paper will be rejected at any top venue for lacking real-world relevance.

## If PROCEED: Next 3 Experiments (Priority Order)

### 1. Real-Model Absorption on Gemma Scope (Priority 1) -- ~1.0 GPU hr
**What**: Measure absorption on Gemma 2 2B layer 12 SAE (16k features) using multi-child proportional metric.
**Why**: The entire paper is synthetic. This is the single highest-ROI experiment.
**Risk**: Metric may saturate (as seen on GPT-2). Mitigation: test on narrower SAEs first (e.g., 4k features) where absorption may be more discriminable.
**Expected outcome**: Either (a) absorption is measurable and discriminable on real SAEs → paper is viable, or (b) metric saturates → need to redesign measurement for real SAEs.

### 2. Encoder Regularization Pilot (Priority 2) -- ~0.5 GPU hr
**What**: Add a penalty term to encoder training that discourages parent-child activation correlation. Measure absorption reduction vs reconstruction degradation.
**Why**: The H_Mech finding opens a direct intervention target. A constructive contribution ("how to fix absorption") is stronger than pure diagnosis.
**Risk**: May degrade reconstruction >10% or show <10% absorption reduction.
**Expected outcome**: If absorption drops >30% with <5% reconstruction loss, this becomes the paper's centerpiece contribution.

### 3. Hierarchy Strength on Real SAEs (Priority 3) -- ~0.5 GPU hr
**What**: Test whether absorption correlates with semantic hierarchy strength on real SAEs (e.g., using Neuronpedia annotations to identify parent-child feature pairs).
**Why**: Validates that the synthetic hierarchy finding transfers to real semantic features.
**Risk**: Finding semantically hierarchical feature pairs in real SAEs is hard and may require manual curation.
**Expected outcome**: Correlation would strengthen the encoder-driven narrative; null result would suggest synthetic hierarchy is artifactual.

## If PIVOT: Backup Direction

If real-model validation shows metric saturation (absorption ~100% on all features), pivot to **Pivot A: Encoder Architecture Modifications** from `alternatives.md`.

**Why**: If absorption is universally high on real SAEs, the measurement itself is not informative. The constructive contribution (encoder regularization) becomes the only viable path.

## Key Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Real SAE metric saturation | High | High | Test on narrower SAEs first; redesign metric if needed |
| Encoder regularization degrades reconstruction | Medium | Medium | Start with small penalty coefficient; sweep gradually |
| Paper rejected for synthetic-only evidence | High if unaddressed | High | Real-model validation is non-negotiable |
| H3 failure weakens causal claims | Already realized | Medium | Reframe as "absorbed features are not differentially steerable" -- still a valid negative result |

## Resource Allocation Recommendation

| Activity | GPU Hours | Priority |
|---|---|---|
| Real-model absorption (Gemma Scope) | 1.0 | **P0 - Must do** |
| Encoder regularization pilot | 0.5 | P1 |
| Hierarchy strength on real SAEs | 0.5 | P2 |
| Paper writing (with current results) | 0 | P1 (parallel) |
| **Total** | **2.0** | |

## Summary

The encoder-driven absorption mechanism is **confirmed and robust**. Four of seven experiments show strong, consistent signals. However, the project has two critical weaknesses:

1. **All evidence is synthetic** -- real-model validation is mandatory.
2. **H3 steering failed replication** -- the causal claim must be dropped or reframed.

**Recommendation**: PROCEED with the encoder-driven narrative, but immediately pivot resources to real-model validation. Do not invest more GPU in H3 or H_Safe on the current synthetic/GPT-2 setup. The paper's viability depends on whether absorption is measurable and meaningful on real SAEs.
