# Result Debate Synthesis: SAE Architecture Comparison

## 1. Consensus Map (High-Confidence Conclusions)

All six perspectives agree on the following:

1. **L0 mismatch is the critical confound**: Every perspective identifies that TopK/Matryoshka (L0~50) vs Baseline (L0~964) is an apples-to-oranges comparison. This is not a minor issue --- it invalidates all architecture comparison claims until resolved.

2. **OrtSAE orthogonal penalty shows no benefit**: All perspectives converge on this negative result. OrtSAE absorption (0.247) is statistically indistinguishable from Baseline (0.254). This directly challenges the OrtSAE paper's claim of ~65% reduction.

3. **The absorption metric has discriminative power**: Random control (0.495) vs TopK/Matryoshka (~0.056) shows the metric can distinguish extreme cases. The metric itself is not broken.

4. **MCC is flat across all variants (~0.22)**: All perspectives note this. The disagreement is on interpretation: optimist sees it as "absorption does not predict downstream performance" (a finding); skeptic sees it as "the metric may be saturated or insensitive" (a problem).

5. **Data integrity mechanisms worked**: The MD5 hash checking and feature count validation successfully prevented the iter006 data replication bug from recurring. This is a methodological win.

---

## 2. Conflict Resolution

### Conflict A: Are the architecture differences real or just L0 effects?
- **Optimist**: "TopK and Matryoshka consistently show low absorption" (treating as real)
- **Skeptic + Methodologist**: "This is entirely explainable by L0 differences; no conclusion possible without L0 matching"
- **Resolution**: The skeptic/methodologist position is stronger. The dose-response relationship (absorption varies with lambda_L1) demonstrates that sparsity is a major driver of absorption. Until L0-matched experiments are run, architecture claims are speculative. The optimist's finding should be reframed as "sparse configurations show low absorption" rather than "TopK/Matryoshka reduce absorption."

### Conflict B: Is flat MCC a meaningful finding or a methodological failure?
- **Optimist + Revisionist**: "Absorption does not predict downstream performance" (H2 falsified --- this is a contribution)
- **Skeptic + Methodologist**: "MCC may be saturated at ~0.22; the metric lacks dynamic range to detect differences"
- **Resolution**: Both can be partially true. The flat MCC is a real finding *under these experimental conditions*, but its interpretation is ambiguous. It could mean (a) absorption truly does not affect feature recovery, or (b) the synthetic task is too easy/hard and MCC has hit a ceiling/floor. The comparativist notes this is the first causal test of absorption-downstream links, so even a null result is novel. However, the methodologist's concern about metric sensitivity must be addressed with additional analysis (e.g., MCC distribution, per-feature breakdowns) before strong claims are made.

### Conflict C: Is this paper publishable as-is?
- **Optimist + Comparativist**: "Yes --- negative results and L0-confound warning are valuable contributions"
- **Skeptic + Methodologist**: "No --- core claims lack L0-matched evidence; statistical methods are incomplete"
- **Strategist**: "Publishable if L0 matching is completed; otherwise reposition as methodology warning"
- **Resolution**: The paper is NOT publishable in its current form for a top-tier venue. The L0-missing issue is fatal for RQ1 claims. However, the paper has a viable path to publishability: (1) complete L0 matching, (2) add statistical rigor, (3) reframe contributions around "L0 confound" as a primary contribution rather than a caveat. The strategist's conditional plan is correct.

---

## 3. Result Quality Score: 4/10

**Justification**:
- **Strengths**: Component isolation design is clean; 5-seed replication provides basic reliability; negative result on OrtSAE is solid; data integrity checks worked; first causal test of absorption-downstream link.
- **Weaknesses**: L0 mismatch invalidates the central architecture comparison; no significance testing or effect sizes; training convergence unverified (2-3s runs suspicious); MCC interpretation ambiguous; dead% display bug indicates QA issues; sample size (5 seeds) is minimal for the claims made.

The score is low not because the work is bad, but because the most important claim (architecture effects on absorption) rests on a confounded comparison. The score would rise to 6-7 with L0 matching and to 8+ with full statistical rigor and GPT-2 validation.

---

## 4. Key Findings (What We Actually Learned)

1. **L0 (sparsity) is a dominant driver of absorption**: The dose-response relationship and the large L0 differences across variants strongly suggest that controlling for L0 is essential before making any architecture claims. This is itself a methodological contribution.

2. **OrtSAE's orthogonal penalty does not reduce absorption** (under untuned lambda conditions): A solid negative result that contradicts the original paper's claim. This finding does not depend on L0 matching because OrtSAE and Baseline have similar L0s.

3. **Absorption and feature recovery MCC show no correlation** in this experimental setup: All variants cluster at MCC~0.22 regardless of absorption rate. Whether this means absorption is irrelevant to downstream performance or the metric is insensitive remains to be determined.

4. **TopK and Matryoshka produce much sparser representations than Baseline** (L0~50 vs ~964): This is a reliable descriptive finding. Whether this sparsity is architecturally intrinsic or tunable is an open question.

5. **Matryoshka nesting (flat vs nested) does not affect absorption**: When L0 is held constant within the Matryoshka family, the hierarchical structure makes no difference. The key factor is k-selection, not nesting.

---

## 5. Methodology Gaps (Critical)

1. **L0 matching is mandatory before any architecture comparison**: This is the highest-priority gap. Without it, RQ1 conclusions are unsupported.

2. **Convergence verification**: 2-3 second training runs on 2M samples need loss curve validation. Are models actually converging?

3. **Statistical rigor**: Add Welch's t-tests, Cohen's d with pooled standard deviation, 95% confidence intervals, and multiple comparison correction (Benjamini-Hochberg).

4. **Sample size**: Increase from 5 to at least 10 seeds, or use bootstrap confidence intervals.

5. **Metric sensitivity analysis**: For MCC, report per-feature distributions, check for ceiling/floor effects, and consider alternative metrics (e.g., AUROC, F1 at different thresholds).

6. **Dead latent analysis**: Fix the dead% display bug and analyze whether dead latents confound absorption calculations.

7. **Real model validation**: The GPT-2 model was loaded but SAEBench eval was not run. At minimum, verify that key findings replicate on a real LLM.

---

## 6. Competitive Position vs SOTA

| Dimension | Our Position | Assessment |
|-----------|-------------|------------|
| L0-matched comparison | **First** | No prior work systematically controls L0 across architectures |
| Absorption-downstream causality | **First** | Kantamneni et al. (2025) tested SAE vs baseline but did not isolate absorption |
| Architecture coverage | Moderate | 6 variants is reasonable but misses recent methods (JumpReLU, Gated SAE) |
| Scale | Weak | Synthetic data only; no real LLM results yet |
| Statistical rigor | Weak | Missing significance tests, effect sizes, and adequate sample size |

**Overall**: We are positioned to make a genuine methodological contribution (L0 confound warning) and a novel empirical contribution (first causal test). However, the current execution is not yet competitive with top-tier standards. The gap is closable with the improvements listed above.

---

## 7. Hypothesis Update

| Hypothesis | Status | Update |
|------------|--------|--------|
| H1a (Matryoshka < Baseline at matched L0) | **Un testable** | Cannot evaluate until L0 matching is done. Current low absorption is confounded by low L0. |
| H1b (OrtSAE == Baseline) | **Supported** | Negative result is reliable. OrtSAE orthogonal penalty does not reduce absorption under these conditions. |
| H1c (TopK > Baseline at matched L0) | **Contradicted (confounded)** | Current data shows TopK < Baseline, but this is likely due to L0 differences. Original prediction may still hold after L0 matching. |
| H2a (Absorption negatively correlates with MCC) | **Falsified** | No correlation observed. However, metric sensitivity concerns temper this conclusion. |
| H2b (Absorption negatively correlates with steering) | **Untested** | Steering efficacy was not evaluated in this iteration. |
| H3a (Mutual coherence positively correlates with absorption) | **Partially tested** | Only Baseline data available. Cannot compute correlation across variants. |
| H3b (Threshold predicts absorption onset) | **Untested** | Requires mutual coherence data across all variants. |

**Revised mental model**:
- Absorption is primarily driven by sparsity (L0), not architecture per se.
- Architecture may have secondary effects, but they are smaller than L0 effects and require careful isolation to detect.
- The link between absorption and downstream interpretability is weaker than hypothesized, but this conclusion needs more sensitive metrics to be definitive.

---

## 8. Action Plan

### Immediate (This Iteration --- ~30 minutes)

1. **Run L0-matched pilot**: Train Baseline with adjusted lambda_L1 to achieve L0~50. Run 1 seed to verify feasibility (~5 min).
2. **Fix dead% display bug**: Correct the percentage calculation in the analysis script.
3. **Add convergence diagnostics**: Log final loss and generate loss curves for all training runs.
4. **Compute proper statistics**: Add Welch's t-test, Cohen's d (pooled std), and 95% CI to the analysis pipeline.

### Short-Term (Next Iteration --- ~1-2 hours)

5. **Run L0-matched full experiment**: If pilot succeeds, run 5 seeds of Baseline-L0=50 and compare to TopK/Matryoshka.
6. **Sensitivity analysis for MCC**: Check MCC distribution for ceiling/floor effects; consider alternative metrics.
7. **Increase sample size**: Run 10 seeds for the key comparison (Baseline-L0=50 vs TopK).

### Medium-Term (Before Paper Submission)

8. **GPT-2 validation**: Run SAEBench eval on GPT-2 small to verify key findings on a real LLM.
9. **Add missing architectures**: Consider JumpReLU or Gated SAE for completeness.
10. **Steering efficacy experiment**: Test H2b if resources allow.

### Decision Gate

| Condition | Decision |
|-----------|----------|
| L0-matched Baseline absorbs at similar rate to TopK/Matryoshka | **PIVOT** to "L0 Confound" as primary contribution; de-emphasize architecture claims |
| L0-matched Baseline absorbs significantly more than TopK/Matryoshka | **PROCEED** with architecture comparison as primary contribution |
| MCC remains flat after L0 matching and metric sensitivity analysis | Add "absorption may not predict downstream performance" as a secondary contribution; discuss metric limitations |

---

*Synthesized from 6 perspectives: Optimist, Skeptic, Strategist, Methodologist, Comparativist, Revisionist*
*Date: 2026-04-25*
