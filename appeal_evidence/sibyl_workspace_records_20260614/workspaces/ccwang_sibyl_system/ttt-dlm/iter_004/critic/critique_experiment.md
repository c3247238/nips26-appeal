# Experiment Critique — Iteration 4

## Overall Assessment: INSUFFICIENT SCALE, SOUND METHODOLOGY

The experimental methodology is rigorous where applied — proper baselines, quality metrics (rep-n, distinct-n), compute-fair comparisons, and honest reporting of negative results. However, the core problem is that the key proposed methods (BSD, A-CFG) were only evaluated at n=16, making all conclusions directional at best.

---

## 1. Sample Size Crisis

**Severity: CRITICAL**

The paper's central claims rest on Countdown-16 (n=16) pilot data:

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Vanilla | 0 | 16 | 0.0% |
| BSD | 1 | 16 | 6.2% |
| A-CFG | 2 | 16 | 12.5% |

At n=16 with binary outcomes:
- The probability of observing 0/16 correct by chance when true rate is 4.7% (vanilla at full scale) = (0.953)^16 = 46%. So Countdown-16 vanilla=0% is entirely expected.
- The probability of observing 2/16 or more correct by chance when true rate is 4.7% = 1 - Σ_{k=0}^{1} C(16,k)·0.047^k·0.953^(16-k) ≈ 15%. **A-CFG's 2/16 could plausibly be noise even if the true rate is 4.7%.**
- BSD's 1/16 is even less distinguishable from vanilla's true rate.

The full-scale DMI result (9.3% mean, s42=7.8%, s123=9.6%, s456=10.6%) is the gold standard. Everything else in this iteration is statistically underpowered.

**Recommendation**: Full-scale Countdown-500 × 3 seeds for BSD and A-CFG is the single most important next step.

## 2. Pilot-to-Full-Scale Effect Size Shrinkage

**Severity: CRITICAL**

The paper mentions "documented pilot-to-full-scale effect size shrinkage across prior iterations (up to 24pp)" but does not analyze this pattern quantitatively. From earlier iterations:

- Several methods showed promising pilot results that collapsed at full scale
- DMI is the *only* method that survived full-scale validation

This track record strongly suggests that BSD's 6.2% and A-CFG's 12.5% on the pilot may not survive scaling. The paper should include a meta-analysis of pilot-to-full-scale shrinkage across all 4 iterations to calibrate expectations.

## 3. Compute-Fair Comparison Reveals the Core Problem

**Severity: HIGH**

Table 4 shows that at matched FLOPs on the pilot:
- BSD (1.1x FLOPs): 6.2% vs vanilla at 141 steps: 12.5% → BSD **loses**
- A-CFG (2.0x FLOPs): 12.5% vs vanilla at 256 steps: 12.5% → **tie**
- DMI (1.05x FLOPs): 12.5% vs vanilla at 134 steps: 12.5% → **tie**

This is devastating for the information island thesis. If vanilla with more steps matches or beats the proposed methods, then the "representation poverty" framing is either wrong or the proposed solutions do not adequately address it. DMI at full scale (9.3% at 1.05x) vs vanilla (4.7% at 1.0x) remains the only compute-efficient win.

**The paper undersells this finding.** The discussion (Section 5.4) says "the bottleneck is real (evidence: entropy metrics)" but this is circular — entropy improvement is input-side, accuracy is output-side, and the output-side shows no advantage over more compute.

## 4. Ablation Quality

**Severity: MEDIUM**

The ablations are well-designed but suffer from the n=16 problem:
- BSD k_frac ablation: 0%, 0%, 6.2% — the "sharp threshold" at k=0.75 is based on 0/16 vs 1/16. This is noise-level variation.
- BSD alpha ablation: All four schedules give identical results (6.2%). This is uninformative — it could mean alpha doesn't matter, or it could mean n=16 cannot distinguish any of them.
- A-CFG schedule ablation: Fixed=12.5%, all others=0%. The "decisive failure of scheduling" is 2/16 vs 0/16 — a 2-sample difference.

At n=100, these ablations would be informative. At n=16, they identify only catastrophic failures (which may or may not be real).

## 5. GSM8K Evaluation

**Severity: HIGH**

GSM8K is evaluated on 16 samples only. Key problems:
- The full-scale vanilla GSM8K result (29.6% from summary.md on 1300/1319 samples) differs from the pilot (25.0% from 4/16). This 4.6pp gap illustrates exactly why n=16 is insufficient.
- A-CFG at 37.5% (6/16) with 95% CI [-12.5%, 37.5%] — the lower bound includes a substantial *negative* effect.
- The paper claims A-CFG shows "cross-benchmark generalization" — this is the strongest claim in the paper, based on 6 vs 4 correct samples out of 16.

## 6. Missing Experiments

**Severity: HIGH**

From the proposal's Phase 2-4 plan, the following are unexecuted:
- [ ] BSD vs DMI vs vanilla on Countdown-500 × 3 seeds (the main experiment)
- [ ] RACFG vs A-CFG vs vanilla on Countdown-500 × 3 seeds
- [ ] BSD+A-CFG on Countdown-500 × 3 seeds
- [ ] Best method on GSM8K-1319 (full scale)
- [ ] McNemar test for A-CFG (only available for DMI)
- [ ] Difficulty-stratified subgroup analysis
- [ ] Comparison against MetaState, CORE, Self-Rewarding SMC

The experiment plan budgets ~58 GPU·h across 4 phases. Based on the results, it appears only Phase 1 (pilots, ~4 GPU·h) was completed. What happened to the remaining ~54 GPU·h?

## 7. Entropy Analysis

**Severity: MEDIUM**

The entropy trajectory analysis is the paper's strongest analytical contribution. Issues:
- r=0.78 (entropy-accuracy correlation) with a binary outcome variable and 2/16 correct samples means the correlation is driven by exactly 2 data points. This is extremely fragile.
- The "near-monotonic decrease" (ρ=-0.95, 15/16 samples) is more robust but expected: EMA of softmax outputs will naturally converge as predictions stabilize, regardless of accuracy impact.
- Terminal entropy BSD=0.001 vs vanilla=0.002 is a directionally interesting but minuscule difference. The paper presents this as validation of information accumulation, but the practical significance is unclear.

## 8. Degeneration Diagnostics

**Severity: LOW — Well Done**

The quality metrics (rep-2, rep-3, distinct-1/2/3) are consistently reported and show no degeneration for any method. This is responsible experimental practice and should be maintained.

## 9. Reproducibility

**Severity: MEDIUM**

- Random seeds specified (42, 123, 456) — good
- Hardware specified (RTX PRO 6000 Blackwell, 98GB) — good
- Model specified (Dream-v0-Instruct-7B) — good
- However, the BSD and A-CFG implementation details (exact code for modifying the denoising loop) are not provided. For reproducibility, code should be released or at minimum pseudo-code should be more detailed about integration with Dream-7B's specific architecture.
