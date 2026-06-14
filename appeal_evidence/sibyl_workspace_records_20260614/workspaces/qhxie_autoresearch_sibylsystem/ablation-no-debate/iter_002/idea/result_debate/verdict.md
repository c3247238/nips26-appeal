# Result Debate Verdict: Executive Summary

## Overall Result Quality Score: 6/10

A genuine and novel mechanistic finding with solid experimental design, but limited to synthetic data and marred by post-hoc criteria revision. Without real-model validation, the paper cannot reach top-tier venues.

---

## Key Conclusion

**Feature absorption in SAEs is driven by encoder alignment, not decoder geometry.** The encoder effect (0.843 +/- 0.082) is approximately 80x larger than the decoder effect (0.011 +/- 0.015). This finding is robust across a 2x2 factorial design, 5 independent seeds, and a hierarchy strength dose-response curve. However, all evidence is synthetic (d_model=128), and the confirmation protocol for H_Mech had to be revised post-hoc after the original criteria failed on 14/15 runs.

---

## What We Learned

1. **Encoder dominance**: The encoder's learned weight matrix is the sole driver of parent-child absorption in synthetic SAEs. The decoder contributes negligibly and may even partially disentangle encoder-induced absorption.

2. **Absorption is universal, not safety-specific**: Both safety and non-safety features in real GPT-2 SAEs show ~97% absorption. Absorption does not discriminate by semantic content.

3. **Absorption is not a control handle**: The H3 steering experiment (5 seeds, 9 conditions) completely failed replication. Absorbed and non-absorbed features respond identically to steering interventions.

4. **Capacity pressure drives absorption**: Lower L0 (fewer active features) leads to higher absorption -- the opposite of the original hypothesis. The encoder overloads features when capacity is constrained.

5. **Decoder compensation is real**: The trained decoder reduces absorption overlap by ~30-50% compared to a random decoder, suggesting a two-player dynamic where the encoder compresses and the decoder decompresses.

---

## Critical Weaknesses

1. **Synthetic-only evidence**: All experiments use d_model=128 synthetic data. No validation on real LLM SAEs.
2. **Post-hoc criteria revision**: H_Mech original pass criteria (B~D) failed on 93.3% of runs. The revised criteria were adopted after seeing the data.
3. **Metric inconsistency**: Three different absorption metrics (cosine, overlap, Jaccard) are used interchangeably without establishing equivalence.
4. **Partial preemption**: Oursland (2026) provides theoretical support for encoder-decoder asymmetry, reducing the exclusivity of our empirical contribution.
5. **Failed replication**: H3 steering pilot (1.62x ratio) did not replicate at full scale (0.91x ratio).

---

## Competitive Position

- **vs Chanin et al. (NeurIPS 2025)**: We diagnose the mechanism; they define the standard metric and benchmark. Our metric is non-standard; direct comparison is impossible.
- **vs Oursland (2026)**: They theoretically derive encoder-decoder asymmetry and propose decoder-free SAEs. Our 2x2 factorial empirically supports their claim, but they go further by proposing a solution.
- **vs Matryoshka/OrtSAE/HSAE**: They propose architectural solutions. We diagnose but do not solve.

**Venue realism**: Workshop or mid-tier conference (AAAI/EMNLP/Findings) without real-model validation. Top-tier (NeurIPS/ICML/ICLR) only if real-model validation + constructive contribution (e.g., encoder regularization) are added.

---

## Action Plan: PROCEED with Mandatory Reframing

### Immediate Priority (P0): Real-Model Validation
Replicate core findings on Gemma Scope or GPT-2 SAEs. This is the single highest-ROI experiment. Without it, the paper is workshop-level at best.

### Priority 1: Honest Methodological Reporting
- Report H_Mech original criteria failure (6.7% pass rate) alongside revised criteria.
- Standardize absorption metric: compute all three metrics on same data, commit to one primary metric.
- Pre-register criteria for any future experiments.

### Priority 2: Constructive Contribution
Pilot encoder regularization: add a penalty term that discourages parent-child co-activation during encoder training. If absorption drops >30% with <5% reconstruction loss, this becomes the paper's centerpiece.

### What NOT to Do
- Do not invest more GPU in H3 steering or H_Safe on current setup. Both have failed.
- Do not present revised H_Mech criteria as primary validation without acknowledging original failure.
- Do not claim "perfect generalization" -- the test uses the same generative process.

### Resource Budget
| Activity | GPU Hours | Priority |
|----------|-----------|----------|
| Real-model validation (Gemma Scope) | 1.0 | **P0 - Must do** |
| Metric standardization | 0.2 | P1 |
| Encoder regularization pilot | 0.5 | P1 |
| Generalization to new geometry | 0.3 | P2 |
| **Total** | **2.0** | |

---

## Bottom Line

The encoder-driven absorption mechanism is **confirmed, robust, and novel enough to proceed**. However, the project must reframe from "absorption as a controllable mechanism" to "absorption as a fundamental structural constraint." The paper's viability depends entirely on whether real-model validation confirms the synthetic findings. If real SAEs show metric saturation (absorption ~100% universally), the project must pivot to encoder regularization as the primary constructive contribution.
