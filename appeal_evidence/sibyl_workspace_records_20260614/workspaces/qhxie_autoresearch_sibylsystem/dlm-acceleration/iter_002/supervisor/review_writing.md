# Supervisor Review: ComposeAccel (Iteration 2)

**Score: 5.5 / 10** | **Verdict: CONTINUE** (significant revisions required)

---

## Executive Summary

ComposeAccel presents the first controlled factorial study of training-free DLM acceleration composition -- a genuine gap in the literature that this work fills. The Ortho metric, interference taxonomy, and honest AR comparison are all valuable contributions. However, the paper suffers from five interconnected problems that collectively place it below the acceptance threshold: (1) the primary "acceleration axis" M1 is effectively non-functional (1.16x measured speedup), undermining the flagship composition result; (2) IGSD's confidence gate provides zero benefit at the primary operating point; (3) most results are pilot-scale with insufficient statistical power; (4) the AR comparison is so devastating that it undercuts the practical motivation; and (5) cross-method comparisons use inconsistent evaluation scales. The paper is commendably honest about limitations but needs substantial additional experimental work and reframing.

---

## Dimension Scores

| Dimension | Score | Assessment |
|-----------|-------|------------|
| Novelty & Significance | 6 | First composition study fills a real gap. Ortho metric is useful. But the composition taxonomy has only 3 data points (one being a near-no-op). IGSD's algorithmic contribution is undermined by the tau=0.0 ablation. |
| Technical Soundness | 5 | QAS and Ortho definitions are clean. But M1+IGSD Ortho=0.96 is misleading when M1 contributes <5% speedup. Confidence gate ablation shows tau=0.0 = tau=0.9 at T_draft=32, undercutting the IGSD mechanism. |
| Experimental Rigor | 5 | Comprehensive scope (15 experiment groups, 2 models, AR comparison). But pairwise results are pilot-scale (100 samples, 1 seed). Inconsistent sample sizes across methods. Wide per-seed variance in three-way results. |
| Reproducibility | 6 | Hardware, hyperparameters, and protocol well-documented. IGSD implementation described as 50 lines. Code release promised. Missing bibliography and placeholder figures reduce reproducibility score. |

---

## Critical Issues (Would Cause Rejection)

### C1: M1 Is Not a Functional Acceleration Method

M1 achieves 1.16x measured speedup with full forward passes. The d2Cache kernel integration failed (15.2x overhead). Despite this, the paper treats M1 as a genuine "acceleration axis" and presents M1+IGSD Ortho=0.96 as evidence of successful two-axis composition.

**Cross-validation with raw data:** M1+IGSD at eta=0.5, tau=0.7, T_draft=16 achieves 2.75x speedup (from `m1_igsd_full.json`). Standalone IGSD at the same IGSD config achieves 2.81x (`igsd_pareto_corrected.json`). The M1 contribution is 2.75/2.81 = 0.98x -- a 2% slowdown, not acceleration. The Ortho=0.96 simply means "adding entropy computation to IGSD costs 4% QAS."

**Impact:** The paper's flagship result ("M1+IGSD achieves near-orthogonal composition") is technically correct but practically misleading. A reviewer who computes 2.75/2.81 will immediately identify this.

### C2: IGSD Confidence Gate Provides Zero Benefit at T_draft=32

From `igsd_ablation_refined.json`, at T_draft=32 with tau=0.0 (no confidence gate) and tau=0.9, both produce 49.5% GSM8K accuracy (99/200 correct for both). The paper reports this honestly in Section 4.4 but calls it a "confidence gate ablation" rather than acknowledging it means IGSD = naive step truncation at this operating point.

**Impact:** The "information-geometric" framing implies a sophisticated mechanism. The ablation data shows the mechanism adds nothing. The paper's third contribution ("IGSD: a 50-line training-free step scheduler using inter-step KL divergence") is undermined.

### C3: Inconsistent Evaluation Scales

From the raw data files:
- M1: 1319 GSM8K samples, 1 seed (from `m1_pareto_full.json`)
- IGSD: 200 GSM8K samples, 1 seed (from `igsd_pareto_corrected.json`)
- M3: 100 GSM8K samples, 1 seed (inferred from M3+IGSD and M1+M3 files)
- Pairwise: 100 GSM8K + 100 MATH500, 1 seed
- Three-way: 100 GSM8K + 100 MATH500, 3 seeds

Table 3 presents M1 at 1319 samples alongside M3 at 100 samples as if they have equal statistical authority. The 103.9% AccRet claim for M3 (73/100 correct) has a 95% CI of approximately [95%, 113%] by normal approximation, while M1's 94.5% AccRet (888/1319 correct) has a 95% CI of approximately [92.9%, 96.1%].

---

## Major Issues (Significantly Weaken the Paper)

### M1: Three-Way Composition Statistical Instability

Per-seed GSM8K accuracy for the Max-Speed recipe: {42: 45%, 123: 52%, 456: 58%}. Per-seed Ortho: {42: 0.91, 123: 1.11, 456: 1.05}. The verdict flips between "near-orthogonal" and "synergistic" depending on seed. No confidence intervals are reported.

### M2: MATH500 Noise Dominance

MATH500 11.1% baseline means the "task-dependent" finding for M3+IGSD (MATH500 Ortho=0.76 vs GSM8K Ortho=0.96) is very likely noise-driven. On 100 MATH500 samples, 3 additional correct answers change AccRet by 27%.

### M3: M3 AccRet Measurement Confound

M3 at wg=0.3 achieves 73/100 correct on GSM8K (seed 42). The full baseline at seed 42 achieves 73.3% on 1319 samples. If the 100-sample pilot subset was drawn with seed 42, the baseline on that subset is likely 73/100 as well. The "103.9% AccRet" is computed against the 3-seed mean (71.2%), not the same-seed pilot baseline.

### M4: Ortho Metric Degeneracy at QAS~1.0

When QAS(M1) = 0.98, Ortho(M1+B) = QAS(M1+B)/QAS(B) for any method B. The metric reduces to "does adding M1 hurt?" which is uninformative about composition synergy.

### M5: Thin Composition Taxonomy

Three pairwise measurements with one near-no-op reduces the effective taxonomy to 2 data points. This is insufficient for a "first systematic composition taxonomy" contribution.

### M6: AR Comparison Understates the Gap

HF Transformers baseline understates the AR advantage. With vLLM, Qwen2.5-7B would likely achieve 200+ TPS at batch=1, widening the QAS gap to 6-10x.

---

## Minor Issues

- Figure 2 and Figure 7 are TODO placeholders
- All citations are [CITE:xxx] placeholders
- M1/M3/IGSD naming without mnemonics (M2 excluded but M1/M3 numbering retained)
- Batch sensitivity accuracy increase at batch=4 is unexplained and possibly artifactual
- No qualitative output examples (sample texts in JSON show "ske ske ske" repetitions at low accept rates)

---

## Positive Aspects

1. **Honest negative results.** M2 NO_GO, M1+M3 interference, and the AR comparison gap are reported transparently. This is commendable and rare.

2. **M1+M3 destructive interference is a genuinely useful finding.** The result that entropy caching + AR guidance produces a net slowdown (Ortho=0.41-0.52) is non-obvious and practically important.

3. **Cross-model validation on Dream-7B.** The consistent pattern (M1+IGSD optimal, M3 guidance counterproductive) across two architectures strengthens generalizability.

4. **Comprehensive experimental scope.** 15 experiment groups covering individual, pairwise, three-way, cross-model, AR comparison, batch sensitivity, and ablation is thorough design.

5. **Well-structured related work.** The categorization of DLM acceleration into four families with Table 1 comparing published protocols is useful to the community.

---

## What Would Raise the Score

**To 7.0 (+1.5):** Full-scale pairwise evaluations (1319 GSM8K, 3 seeds) for all three pairs with confidence intervals on Ortho. Honest IGSD reframing acknowledging the confidence gate provides no benefit at T_draft=32. Clear separation of measured vs projected M1 results throughout. Placeholder figures and citations completed.

**To 7.5 (+2.0):** Above, plus finding an operating point where confidence partitioning demonstrably outperforms naive truncation, OR adding more method families (FastDLLM, SSD) to the composition matrix to strengthen the taxonomy.

**To 8.0 (+2.5):** Above, plus kernel-level M1 implementation demonstrating genuine speedup, making M1+IGSD a true two-axis composition result.

---

## Verdict: CONTINUE

The paper addresses a genuine gap (no prior DLM acceleration composition study) and has comprehensive experimental scope. The interference taxonomy and honest AR comparison are valuable. However, the three critical issues (M1 non-functional, IGSD gate non-contributory, pilot-scale evaluations) must be addressed before the quality gate can pass. The most impactful next step is full-scale pairwise replication to put the Ortho values on firm statistical ground.
