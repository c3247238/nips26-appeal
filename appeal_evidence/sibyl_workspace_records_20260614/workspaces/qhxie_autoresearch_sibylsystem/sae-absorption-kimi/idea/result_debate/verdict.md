# Verdict: Component-Isolated SAE Absorption Study on SynthSAEBench

**Date:** 2026-04-25
**Result Quality Score:** 6.5 / 10
**Workspace:** `/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi`
**Data:** iter_006 Full Experiment (7 variants x 5 replicates each, seeds 42/123/456/789/1011)

---

## Key Conclusion

**TopK sparsity -- not multi-scale decomposition, orthogonality, or gating -- is the dominant driver of absorption reduction in sparse autoencoders, with an effect size (Cohen's d = 4.93, 78% reduction) that dwarfs all other components tested.** At matched L0=50, TopK (0.056), MultiScale (0.055), and Full Matryoshka (0.066) show statistically indistinguishable absorption rates, suggesting that sparsity level -- not architectural innovation -- is the operative variable. This finding challenges the field's attribution of absorption reduction to architectural complexity (Matryoshka's multi-scale decomposition, OrtSAE's orthogonality penalties) and reframes the optimization target from "which architecture?" to "what sparsity level?"

**Critical caveat:** The causal claim that "absorption is a sparsity phenomenon" rests on correlational evidence (r = 0.865 across 7 variant means). The L0-matched control experiment (Baseline L1 tuned to L0=50) has NOT been executed and is required to disentangle sparsity effects from architectural effects. Additionally, TopK's 81.7% dead latent rate raises questions about whether the absorption reduction is artifactual.

---

## What We Actually Learned

1. **TopK sparsity reduces absorption by 78% with overwhelming statistical evidence** (Cohen's d = 4.93, all 5 replicates below lowest baseline replicate, zero range overlap). This is the single most robust finding.

2. **MultiScale and Full Matryoshka do not outperform TopK at matched L0** (0.055 and 0.066 vs 0.056, all within error bars at L0=50). The multi-scale decomposition and hierarchical loss provide no marginal absorption benefit beyond explicit k-sparsity.

3. **Orthogonality penalties have negligible effect on absorption** (2.7% reduction, d = 0.13, p = 0.845), directly contradicting OrtSAE's claim of 65% reduction. However, this result is confounded by a custom training loop and should be treated as preliminary.

4. **Gating has no effect on absorption** (0.261 vs 0.252 baseline, d = -0.17, p = 0.797). The decoupled detection/magnitude paths neither improve nor worsen absorption.

5. **Absorption is primarily encoder-driven**. Orthogonality achieves near-perfect reconstruction (MSE = 3.2e-5) by constraining decoder geometry, but absorption remains unchanged (0.245 vs 0.252). Decoder improvements do not affect encoder activation patterns.

6. **MCC is structurally degenerate in overcomplete settings** (~0.214-0.222 across ALL 7 variants including Random Control at 0.221). Hungarian matching on overcomplete dictionaries produces chance-level correlations. This metric should not be used for feature recovery validation here.

7. **No absorption-hedging trade-off is observed** (hedging ~0.235-0.240 constant across all 7 variants). The predicted trade-off does not manifest on synthetic hierarchical data.

---

## PIVOT or PROCEED?

**RECOMMENDATION: PROCEED with L0-matched ablation and dead-latent analysis.**

### Why PROCEED (Unanimous across all 6 perspectives)

- The TopK finding is robust (d = 4.93) and consistent across 5 replicates
- All 7 variants now have complete 5-replicate data; component ranking is clear
- Clear path to publication-quality results: First component-isolated causal analysis with ground-truth data
- The remaining critical experiments are low-cost (~1 hour total) and high-information
- Even the "worst case" (all components equivalent at matched L0) yields a publishable reframing: "Absorption is a Sparsity Phenomenon, Not an Architectural One"
- No backup idea has higher expected value at this point

### Why NOT PIVOT

- The full dataset strengthens (not weakens) the core finding
- Backup ideas (narrow diagnostic paper, rate-distortion theory) have lower expected impact
- The ground-truth measurement methodology is novel and addresses a real community need
- Marginal cost of remaining experiments is very low

---

## Action Plan

| Priority | Action | GPU Hours | Expected Outcome |
|---|---|---|---|
| **P0** | **L0-matched ablation**: Baseline with tuned L1 to achieve L0=50 and L0=550 (3 replicates each) | ~0.3h | Disentangles sparsity from architecture; addresses central confound |
| **P1** | **Dead-latent-adjusted absorption** for TopK, MultiScale, Full Matryoshka | ~0.1h (CPU) | Tests whether improvement is artifactual |
| **P2** | **TopK k-sweep** (k in {10, 25, 100, 200, 500}) | ~0.3h | Dose-response relationship; characterizes sparsity-absorption curve |
| **P2** | **Real-LLM validation** (TopK + Baseline on Gemma-2-2B or Pythia-160M) | ~0.5h | Tests synthetic-to-real transfer (H4) |
| **P3** | **Orthogonality fair comparison** (SAELens runner plugin or lambda sweep) | ~0.3h | Resolves whether OrtSAE contradiction is genuine or artifactual |
| **Total** | | **~1.5h** | |

**Decision Gates:**

- **Gate 1** (after P0): If L0-matched Baseline (L0=50) achieves absorption ~0.05-0.10: "sparsity is sole driver." If > 0.15: "TopK has independent architectural benefits beyond sparsity."
- **Gate 2** (after P1): If active-only TopK absorption < 0.10: signal is genuine. If > 0.20: largely artifactual.
- **Gate 3** (after P2 real-LLM): If real-LLM validation confirms synthetic ranking: target NeurIPS/ICLR main. If not: acknowledge gap; paper still valuable.

---

## Target Venue

| Scenario | Venue | Probability |
|---|---|---|
| Full data + L0-matched + dead-latent validation + real-LLM validation | NeurIPS/ICLR/ICML main conference | 25% |
| Full data + L0-matched + dead-latent validation, no real-LLM | ICLR/NeurIPS Workshop or EMNLP Findings | 45% |
| All components equivalent at matched L0 (sparsity is sole driver) | Reframed paper: "Absorption is a Sparsity Phenomenon" -- still main-conference viable | 30% |

**Hard constraints for any submission:**
- Do NOT claim "absorption is a sparsity phenomenon" without L0-matched control; say "strong correlational evidence" instead
- Do NOT claim "architecture is irrelevant" without L0-matched control
- Report dead latent rates (81.7% TopK, 56.4% MultiScale, 56.7% Matryoshka) transparently
- Acknowledge MCC degeneracy; do not use MCC as validation metric
- Correct all "r ~ -0.99" claims to "r = 0.865 (p = 0.012)"
- Apply Holm-Bonferroni correction for multiple comparisons (pre-registered)
- Acknowledge orthogonality custom training loop limitation

---

## Biggest Strengths and Weaknesses

**Biggest strength:** A surprising, large-effect finding (TopK d=4.93 dominates, orthogonality d=0.13 is null) that challenges field assumptions, based on ground-truth measurement without probe confounds. First component-isolated causal analysis in SAE literature. All 7 variants now have complete data.

**Biggest weakness:** The unresolved L0 confound. We cannot yet distinguish "TopK architecture reduces absorption" from "enforcing L0=50 sparsity reduces absorption" without the L0-matched control.

**Second weakness:** TopK's 81.7% dead latent rate raises questions about practical utility. The "best" variant may achieve its result through dictionary crippling.

**Third weakness:** No real-LLM validation yet. Synthetic-to-real transfer (H4) is untested.

**Fourth weakness:** Orthogonality null result is confounded by custom training loop (decoder renormalization, no L1 warm-up, fixed LR). The 300x lower MSE (3.2e-5 vs 0.0104) is a red flag.

---

*Synthesized from 6 perspectives: Optimist, Skeptic, Strategist, Methodologist, Comparativist, Revisionist*
*Iter_006, SynthSAEBench Component-Isolated Study, Full Dataset (7 variants x 5 replicates)*
