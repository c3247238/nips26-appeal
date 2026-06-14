# Result Debate Synthesis: Encoder-Driven Feature Absorption in SAEs

## 1. Consensus Map: High-Confidence Conclusions

All six perspectives agree on the following:

1. **Encoder training is sufficient to produce high absorption rates in synthetic SAEs.** The 2x2 factorial (H_Mech) shows that trained encoder + random decoder achieves ~0.80 absorption overlap, while random encoder + trained decoder stays near baseline (~0.01). This directional finding is uncontested.

2. **The decoder's contribution to absorption is negligible compared to the encoder.** The encoder effect (0.843 +/- 0.082) is ~80x larger than the decoder effect (0.011 +/- 0.015). All six analysts accept this magnitude gap.

3. **Trained SAEs show significantly higher absorption than random baselines.** Multi-seed validation (5 seeds, stochastic noise=0.1) yields t=36.04, p=3.85e-10. The effect is robust and replicable.

4. **Absorption increases monotonically with hierarchy strength.** Similarity 0.5 -> 0.67 -> 0.8 produces absorption 0.416 -> 0.501 -> 0.544 (ANOVA p < 1e-10). The dose-response relationship is clean.

5. **All evidence is synthetic (d_model=128).** No perspective disputes this. It is universally recognized as the project's critical weakness.

6. **H3 steering failed replication.** The pilot's 1.62x ratio did not replicate at full scale (5 seeds, 9 conditions). All perspectives classify this as a genuine negative result, not a hidden positive.

7. **H_Safe shows no safety-specific elevation.** Safety and non-safety features both show ~97% absorption on GPT-2 SAE (p=0.989). This is a null result.

---

## 2. Conflict Resolution: Where Perspectives Disagree

### Conflict 1: Is H_Mech "Confirmed" or "Post-Hoc Rationalized"?

- **Optimist**: "Encoder-driven absorption CONFIRMED" -- the 80x ratio is unambiguous.
- **Skeptic**: "Fatal flaw" -- original pass criteria (B~D) failed on 14/15 runs; revised criteria are post-hoc and cannot fail.
- **Methodologist**: "Post-hoc criterion revision undermines confirmation claim."

**Resolution**: The skeptic and methodologist are correct that the *original* criteria failed, and this must be reported honestly. However, the revised criteria (encoder effect > 0.5, decoder effect < 0.1) are not arbitrary -- they directly test the paper's core claim (encoder dominance). The fairest assessment: **the encoder-dominance finding is robust, but the confirmation protocol was flawed**. The paper should report both the original criteria failure (6.7% pass rate) and the revised criteria success, with explicit justification for the revision.

### Conflict 2: Is Generalization "Perfect" or a "Ceiling Effect"?

- **Optimist**: "Near-perfect generalization (r=0.998)" -- proves absorption is structural, not overfitting.
- **Skeptic**: "Ceiling effect" -- test data is from the same generative process; the SAE has already seen the hierarchy structure.

**Resolution**: The skeptic is correct that the generalization test does not test *new geometry*, only new samples from the same geometry. However, the optimist's claim that absorption is structural (not sample-specific) is still supported -- the SAE learns a compression strategy for the given hierarchy, not memorization of individual samples. The paper should **reframe** the claim from "generalizes perfectly" to "is stable across train/test splits from the same distribution." True generalization to new hierarchy geometries remains untested.

### Conflict 3: Is the Paper Top-Tier or Workshop-Level?

- **Comparativist**: "Workshop or mid-tier conference (AAAI/EMNLP/Findings)" -- synthetic-only, partially preempted by Oursland (2026).
- **Optimist**: "Publishable story" -- the encoder-driven mechanism is genuinely new.
- **Strategist**: "PROCEED with reframing" -- real-model validation is mandatory for top-tier.

**Resolution**: The comparativist's assessment is the most realistic. The core insight is novel but (a) synthetic-only, (b) partially preempted by concurrent theory work, and (c) missing real-model validation. **Without real-model validation, the paper is workshop/mid-tier at best. With real-model validation, it could reach top-tier.** The strategist's recommendation to prioritize real-model validation is correct.

### Conflict 4: Is the Inverse L0 Effect a "Discovery" or "Variance Confound"?

- **Optimist**: "Refinement, not failure" -- consistent with encoder-driven mechanism (capacity pressure).
- **Skeptic**: "Minor caveat" -- higher variance at low L0 may inflate the mean.
- **Revisionist**: "Capacity pressure" -- fewer features force each to represent more concepts.

**Resolution**: The effect is real (ANOVA p < 1e-10) and consistent across seeds. The skeptic's variance concern is partially addressed by the data (variance is actually lower at L0=20). The revisionist's "capacity pressure" interpretation is the most coherent explanation. **Consensus: genuine finding, opposite direction from hypothesis.**

---

## 3. Result Quality Score: 6/10

**Justification**:
- **Strengths (+3)**: Robust encoder-dominance finding (80x ratio), clean dose-response (hierarchy strength), multi-seed replication (5 seeds), novel factorial design.
- **Weaknesses (-2)**: All synthetic data, post-hoc criteria revision for H_Mech, metric inconsistency across experiments, failed H3 replication.
- **Critical gap (-2)**: No real-model validation. The entire contribution rests on d_model=128 synthetic data.
- **Partial preemption (-1)**: Oursland (2026) provides theoretical support for encoder-decoder asymmetry, reducing exclusivity.
- **Mitigation (+1)**: Negative results (H3, H_Safe) are reported honestly, which adds methodological credibility.

**Score breakdown**: Base 5 (solid empirical finding) + 3 (robustness) - 2 (methodological flaws) - 2 (synthetic-only) - 1 (preemption) + 1 (honest negatives) = **6/10**.

A score of 6 means: "Genuine finding with solid experimental design, but limited scope and critical gaps prevent top-tier publication without additional work."

---

## 4. Key Findings: What We Actually Learned

1. **Encoder alignment drives absorption; decoder geometry is negligible.** In synthetic SAEs, the encoder's learned weight matrix is the sole driver of parent-child feature absorption. The decoder contributes ~80x less to absorption and may even partially disentangle encoder-induced absorption. This overturns the implicit assumption that absorption is a joint encoder-decoder phenomenon.

2. **Absorption is a universal structural property, not a safety-specific vulnerability.** Both safety and non-safety features in real GPT-2 SAEs show ~97% absorption. Absorption does not discriminate by semantic content -- it is a geometric property of how encoders handle hierarchical structure.

3. **Absorption cannot be exploited for steering interventions.** The H3 steering experiment (5 seeds, 9 conditions) shows no differential sensitivity between absorbed and non-absorbed features. Absorption is a representational property, not a control handle.

4. **Absorption increases with hierarchy strength but decreases with sparsity.** Higher parent-child similarity -> more absorption (expected). But fewer active features (lower L0) -> more absorption (unexpected), suggesting a "capacity pressure" mechanism where the encoder overloads features when capacity is constrained.

5. **The decoder actively compensates for encoder absorption.** Condition D (full training) shows lower absorption overlap than Condition B (trained encoder + random decoder), suggesting the trained decoder learns to redistribute parent activations across more features during reconstruction. This is a two-player dynamic, not a one-player game.

---

## 5. Methodology Gaps: Critical Experimental Improvements Needed

### Gap 1: Post-Hoc Criterion Revision (H_Mech) -- **HIGH PRIORITY**
The original pass criteria (B~D) failed on 14/15 runs. The revised criteria were adopted after seeing the data. **Remediation**: Pre-register criteria for any future replication. Report the original criteria failure honestly in the paper.

### Gap 2: Metric Inconsistency -- **HIGH PRIORITY**
Three different metrics (cosine, overlap, Jaccard) are used interchangeably without establishing equivalence. **Remediation**: Compute all three metrics on the same data and report inter-correlation. Commit to one primary metric with justification.

### Gap 3: No Real-Model Validation -- **CRITICAL**
All experiments are on synthetic data (d_model=128). The field will reject without real SAE evidence. **Remediation**: Replicate core findings on Gemma Scope or GPT-2 SAEs. This is the single highest-impact addition.

### Gap 4: Generalization Tests Same Geometry -- **MEDIUM PRIORITY**
The held-out test uses the same generative process. **Remediation**: Test generalization to different hierarchy geometries (e.g., train on cosine=0.5, test on cosine=0.8).

### Gap 5: Reproducibility Gaps -- **MEDIUM PRIORITY**
No code repository, incomplete hyperparameters, no environment specification. **Remediation**: Release code and full hyperparameters.

### Gap 6: H3 Steering No Pre-Registered Alpha -- **MEDIUM PRIORITY**
Effect direction reverses with alpha; no primary analysis was pre-registered. **Remediation**: Pre-register alpha=1.0 for any future steering experiment.

---

## 6. Competitive Position: Where Do We Stand vs SOTA?

### Direct Comparisons
- **Chanin et al. (NeurIPS 2025)**: Defines the standard "absorption rate" metric. Our metric (Jaccard overlap) is different and non-standard. We cannot compare numerically.
- **Oursland (2026)**: Theoretically derives encoder-decoder asymmetry and proposes decoder-free SAEs. Our empirical 2x2 factorial supports their theoretical claim, but they go further by proposing a solution. **Our contribution is partially preempted.**
- **Matryoshka SAE / OrtSAE / HSAE**: Architectural solutions to absorption. We diagnose the mechanism but do not propose a solution. **Our contribution is narrower.**

### Contribution Margin
| Finding | Novelty | Strength |
|---------|---------|----------|
| Encoder-driven mechanism (empirical) | High | Strong |
| Factorial methodology | High | Strong |
| Inverse L0-absorption | Moderate | Surprising |
| Steering negative result | Moderate | Honest |
| Safety null result | Low | Informative |

### Venue Realism
- **Workshop (MEI, XAI, SAE-dedicated)**: Best fit. Narrow scope, single mechanism, synthetic data.
- **Mid-tier (AAAI, EMNLP, COLING)**: Good fit if real-model validation is added.
- **Top-tier (NeurIPS/ICML/ICLR)**: Poor fit without real-model validation and a constructive contribution (e.g., encoder regularization).

---

## 7. Hypothesis Update: Which Survived, Which Need Revision?

| Hypothesis | Verdict | Confidence | Update |
|-----------|---------|------------|--------|
| H1: Trained > Random | **Confirmed** | High | Survives intact. Robust across 5 seeds. |
| H_Mech: Encoder drives absorption | **Confirmed (revised)** | High | Core claim survives, but criteria need honest reporting. |
| H2: Absorption inversely correlates with frequency | **Refuted** | High | Pilot showed positive correlation; not re-tested at full scale. Drop from paper. |
| H3: Steering improves absorbed sensitivity | **Refuted** | High | Failed replication. Absorption is not a control property. Reframe as negative result. |
| H_Safe: Safety features more absorbed | **Refuted** | High | Null result. Absorption is universal, not safety-specific. |
| H_Comp: Absorption increases with hierarchy strength | **Confirmed** | High | Survives. Monotonic dose-response is clean. |
| L0 Sparsity: Higher L0 -> higher absorption | **Refuted (opposite)** | High | Genuine finding opposite to hypothesis. "Capacity pressure" interpretation. |
| Generalization: Absorption generalizes | **Partially confirmed** | Medium | Stable across train/test splits, but not tested on new geometry. |

### Mental Model Revisions
1. **Decoder is not passive** -- it actively disentangles encoder-induced absorption. Two-player dynamic.
2. **Absorption is not controllable** -- it is a representational property, not a steering handle.
3. **Absorption is driven by capacity pressure** -- fewer active features force more absorption, not fewer.

### New Hypotheses Generated
1. **Decoder disentanglement can be optimized** via regularization that penalizes parent-child co-activation in decoder output.
2. **Absorption has a critical capacity threshold** -- catastrophic above a certain d_sae/d_model ratio.
3. **Real SAEs exhibit near-universal absorption** (>0.9) due to deeper natural language hierarchies.

---

## 8. Action Plan: Prioritized Next Steps

### Verdict: **PROCEED with mandatory reframing**

The core finding (encoder-driven absorption) is robust and novel enough to justify continuation. However, the project must reframe from "absorption as a controllable mechanism" to "absorption as a fundamental structural constraint." Two hypotheses (H3, H_Safe) must be dropped or reframed as negative results.

### Immediate Actions (Next 1-2 Iterations)

| Priority | Action | GPU Hours | Rationale |
|----------|--------|-----------|-----------|
| **P0** | Real-model absorption on Gemma Scope | ~1.0 | Single highest-ROI experiment. Without this, paper is workshop-level at best. |
| **P1** | Standardize absorption metric | ~0.2 | Compute all three metrics on same data; commit to one primary metric. |
| **P1** | Honest reporting of H_Mech criteria | 0 | Report original criteria failure (6.7% pass) alongside revised criteria. |
| **P2** | Encoder regularization pilot | ~0.5 | Constructive contribution: can we reduce absorption via encoder penalty? |
| **P2** | Generalization to new geometry | ~0.3 | Train on one cosine, test on another. |

### If Real-Model Validation Succeeds
- **Path A**: Absorption is measurable and discriminable on real SAEs -> Elevate to mid-tier venue (AAAI/EMNLP).
- **Path B**: Add encoder regularization experiment showing >30% absorption reduction with <5% reconstruction loss -> Elevate to top-tier venue.

### If Real-Model Validation Fails (Metric Saturation)
- **Pivot**: Encoder regularization becomes the primary contribution. Reframe paper as "How to Fix Absorption" rather than "What Causes Absorption."

### What NOT to Do
- Do NOT invest more GPU in H3 steering on current setup. It has failed replication.
- Do NOT invest more GPU in H_Safe on GPT-2. The metric saturates at ~97%.
- Do NOT present revised H_Mech criteria as the primary validation without acknowledging the original failure.

### Resource Allocation
| Activity | GPU Hours | Priority |
|----------|-----------|----------|
| Real-model validation (Gemma Scope) | 1.0 | **P0 - Must do** |
| Metric standardization | 0.2 | P1 |
| Encoder regularization pilot | 0.5 | P1 |
| Generalization to new geometry | 0.3 | P2 |
| Paper writing (reframed narrative) | 0 | P1 (parallel) |
| **Total** | **2.0** | |

---

## Summary

The encoder-driven absorption mechanism is **confirmed, robust, and novel**. Four of seven experiments show strong, consistent signals. However, the project has two critical weaknesses that prevent top-tier publication: (1) all evidence is synthetic, and (2) the H_Mech confirmation relied on post-hoc criteria revision. The most honest and productive path forward is to **proceed with the encoder-driven narrative while immediately prioritizing real-model validation**. The paper should reframe absorption as a "fundamental structural constraint" rather than a "controllable mechanism," honestly report negative results, and use the decoder disentanglement observation as a springboard for constructive contributions.
