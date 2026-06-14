# Experiment Critique (Iteration 3 — Post-DTA Full-Scale Results)

**Reviewer**: Critic Agent
**Date**: 2026-03-10
**Scope**: All experimental results (`exp/results/`), experimental code (`exp/code/`), methodology (`plan/methodology.md`)

---

## Overall Assessment: 6/10

The experimental infrastructure is solid: proper seeding, multiple benchmarks, statistical testing protocols, and well-organized result files. The full-scale Countdown-500 evaluation (500 problems x 3 seeds x 7 methods) is rigorous for DLM research. However, the experiment suffers from **incomplete execution** (most cross-benchmark results are pilot-only), **proxy metric concerns** (DTA's MLM self-supervision vs task accuracy), and **a critical data integrity issue** (the paper still presents completed results as "pending").

---

## CRITICAL Issues

### E1. Full-Scale Results Available But Not Reflected in Paper

`full_scale_summary.json` contains completed results for ALL 7 methods on Countdown-500:

```json
"dta":        {"s42": 4.4, "s123": 4.6, "s456": 5.4, "avg": 4.8},
"dta_remdm":  {"s42": 3.6, "s123": 2.4, "s456": 4.8, "avg": 3.6},
"scp":        {"s42": 8.4, "s123": 9.4, "s456": 9.4, "avg": 9.1}
```

But the paper's Table 1 shows DTA, DTA+ReMDM, and SCP as "pending" or "interim." This is a data integrity problem. Whether the delay is due to writing lag or intentional omission, it must be fixed immediately.

### E2. DTA+ReMDM Produces Actively Harmful Results

DTA+ReMDM (3.6%) is significantly *worse* than vanilla (4.7%) and even worse than ReMDM alone (4.4%). The proposal claimed DTA and remasking are "orthogonal and complementary" (Section 3.4). The full-scale data falsifies this claim: combining parameter-space and token-space interventions produces interference, not complementarity.

This requires analysis. Hypothesis: DTA's LoRA updates, optimized on already-revealed tokens, create a parameter-space bias. When ReMDM then remasks and re-predicts tokens, the biased model produces worse predictions than the original model would have. The DTA adapter has learned patterns from tokens that ReMDM subsequently erases, creating a mismatch between the adapter's "memory" and the current token state.

### E3. Proxy Metric Gaming Check — DTA's Self-Supervised Signal

**This is the most fundamental experimental concern.** DTA's M-step computes masked LM loss on tokens the model itself just predicted. This is quasi-circular: the model learns to predict tokens it has already decided to predict. The pilot experiments acknowledged that "self-consistency loss yields near-zero gradients because the model already agrees with its own predictions." The mask-and-predict variant provides a non-zero signal, but the information content of this signal is questionable.

Evidence of the problem:
- DTA's LoRA norms are well-controlled (0.05-0.25)---the adapter is learning *something*
- DTA's prediction confidence increases monotonically (0.969 to 0.995)---the model is becoming more confident
- DTA's accuracy does not improve (4.8% vs 4.7%)---but what it learns is not useful for task correctness

The model is learning to be more confident in its own predictions, but this confidence is not correlated with correctness. This is the textbook definition of a calibration failure, and it should have been caught by measuring calibration curves (confidence vs accuracy) for DTA-augmented predictions. **This analysis is missing from the experiments.**

---

## HIGH Issues

### E4. Cross-Benchmark Results Are Almost Entirely Pilot-Scale

The only full-scale results are on Countdown-500. Everything else is N=16 pilot:

| Benchmark | Full-Scale Status |
|-----------|-------------------|
| Countdown-500 | COMPLETE (7 methods x 3 seeds) |
| GSM8K | PARTIAL (vanilla 1300/1319; DTA 900/1319; others partial) |
| MBPP | PILOT ONLY (N=16) |
| HumanEval | NOT STARTED |

The paper's most favorable DTA claim (MBPP +12.5pp) is from N=16. Given the documented 24pp pilot-to-full-scale discrepancy on Countdown (DMI: 0% pilot vs 9.3% full-scale), the MBPP pilot signal has a >50% chance of not surviving full-scale evaluation.

The partial GSM8K data (`full_scale_summary.json`) already shows DTA (29.0%) = vanilla (29.6%) at N=900. This pattern is consistent with Countdown: DTA does nothing.

### E5. Statistical Tests Missing for Key Comparisons

The paper promises McNemar tests with Bonferroni correction, but no statistical test results appear in the experiments section. The following comparisons need formal testing:
- DMI vs vanilla on Countdown-500 (likely significant, but needs p-value)
- SCP vs vanilla on Countdown-500
- DTA vs vanilla on Countdown-500 (likely not significant)
- DMI vs SCP (are they significantly different?)
- DTA+ReMDM vs vanilla (is the degradation significant?)

Without p-values and confidence intervals, the results are descriptive only.

### E6. Missing Calibration Analysis

Section 5.4 reports that DTA's prediction confidence increases from 0.969 to 0.995 across denoising steps. But if accuracy doesn't improve, this means the model is becoming *overconfident*. A calibration analysis (binned confidence vs actual accuracy) would reveal whether DTA degrades calibration, which would explain why DTA+ReMDM fails: ReMDM relies on confidence scores for remasking decisions, and if DTA makes these scores less reliable, remasking becomes worse.

### E7. DMI Mechanism Not Fully Ablated

DMI is the paper's strongest result, but its mechanism is not systematically ablated:
- What is the effect of the mixing weight alpha? (Only alpha=0.3 tested)
- What is the effect of the temperature tau_soft? (Only tau=0.5 tested)
- Does DMI help because of the soft embedding, or because it breaks the discrete information bottleneck? (Could a simple "carry over last step's top-1 hard prediction" achieve similar results?)
- Does DMI benefit from multi-step lookback (carrying embeddings from the last 2-3 steps)?

If DMI is to be the paper's primary contribution, it needs the level of ablation currently given to DTA.

### E8. SCP Computational Overhead Is Extreme

SCP is listed as "~7x" overhead in the method section (45.9s vs 3.7s per sample = 12.4x actual). For comparable accuracy to DMI (9.1% vs 9.3%), SCP costs approximately 10x more compute. This makes SCP completely impractical and adds little to the spectrum ablation. The paper should acknowledge this more directly---SCP is not a viable method, it's a data point in the spectrum analysis.

---

## MEDIUM Issues

### E9. Pilot-to-Full-Scale Discrepancy for DTA Not Analyzed

The paper extensively analyzes the pilot-to-full-scale discrepancy for DMI (0% pilot vs 9.3% full-scale). But it does not analyze the discrepancy for DTA. At pilot scale (N=16), DTA showed mixed results (6.2% on some benchmarks). At full scale, DTA is 4.8%. The shrinkage patterns for different methods may reveal systematic biases in pilot evaluation.

### E10. Lack of Qualitative Analysis

The paper claims DTA "creates zero unstable token positions" and ReMDM-conf creates 94.8. But no qualitative examples are provided. What does a DTA-generated Countdown solution look like versus a vanilla one? Do they differ at all? If DTA changes model behavior but not accuracy, understanding *how* it changes behavior is important for diagnosing the failure mode.

### E11. Incomplete Ablation at Full Scale

All ablation studies (LoRA rank, decay gamma, update frequency, layer count) are at pilot scale (N=16). Given that pilot effects don't survive scaling, these ablation results are unreliable for making design recommendations. The default configuration (r=4, gamma=0.95) was chosen based on pilot data that turned out to be misleading.

### E12. No Negative Control for DMI

To verify that DMI's improvement is due to the soft embedding mechanism and not simply due to the implementation introducing a random perturbation (which could sometimes break the model out of bad local optima), a negative control should be tested: inject random embeddings with the same norm as DMI's soft embeddings. If this also improves accuracy, the DMI mechanism is not what we think it is.

---

## Positive Notes

1. **Statistical methodology**: The plan for 3 seeds x 500 samples with McNemar + Bonferroni is excellent for DLM research. The execution on Countdown-500 is commendable.
2. **Result organization**: JSON files with per-seed breakdowns enable easy verification and further analysis.
3. **Honest pilot-to-full-scale analysis**: The documentation of 24pp discrepancies between pilot and full-scale results is a genuine methodological contribution.
4. **Diversity/degeneration monitoring**: All experiments track distinct-n and rep-n alongside accuracy, preventing the PPL gaming seen in earlier iterations.
5. **LoRA norm tracking**: Monitoring adapter parameter norms across denoising steps provides useful stability evidence.
