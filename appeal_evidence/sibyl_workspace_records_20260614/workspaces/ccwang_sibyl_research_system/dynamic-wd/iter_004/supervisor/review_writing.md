# Supervisor Review: "When Does Dynamic Weight Decay Help? A Unified Framework Analysis Under AdamW"

**Reviewer Role:** Independent third-party supervisor (NeurIPS/ICML/ICLR calibration)
**Review Date:** 2026-03-18
**Overall Score:** 6.5 / 10 (Borderline Reject — approach the threshold but do not yet cross it)
**Verdict:** CONTINUE (iteration needed before this can pass)

---

## Correction to Prior Review

The previous supervisor review (score=5.5) contained a critical error: it accused the paper of fabricating SGD p-values (claiming SWD p=0.013 and half_lambda p=0.028 as "not reproducible from workspace data"). I have independently cross-validated the current paper.md against the raw SGD experiment files in `iter_003/exp/results/sgd_baseline/`:

- **SGD constant mean=91.22**: Verified (actual: 91.217±0.072)
- **SGD no_wd: delta=-0.91%, p=0.002**: Verified (actual: delta=-0.913%, p=0.0022)
- **SGD SWD CIFAR-100: delta=-1.07%, p=0.036**: Verified (actual: delta=-1.067%, p=0.0355)
- **SGD weight norms 64.6 (constant) vs 127.1 (no_wd)**: Verified (actual: 64.57 vs 127.06)

The current paper.md's Table 5 correctly reports SWD CIFAR-10 as p=0.071 (not significant) and half_lambda CIFAR-10 as p=0.121 (not significant). The prior supervisor's accusation was based on a prior revision that no longer exists. **The paper's core SGD negative control data is accurate and reproducible.**

---

## What Has Improved Since Last Review

The revision cycle since the prior review has addressed several major issues:

1. **SGD negative control added (42 experiments)**: Section 5.5 provides exactly the optimizer-specificity test demanded by prior critiques. The results are compelling: AdamW spread 1.2% vs SGD spread 97% on weight norms is a visually striking, data-verified contrast.

2. **Figure 6 generated**: The AdamW vs SGD weight norm comparison figure provides the absorption mechanism's strongest visual argument. The values shown (64.6/127.1 for SGD, 95.9/97.0 for AdamW) match raw data exactly.

3. **TOST equivalence testing added (Section 5.1)**: The paper now honestly reports that TOST at delta=+/-0.5% confirms equivalence for 1/12 comparisons, and at delta=+/-1.0% for 6/12. This is appropriate scientific honesty rather than overclaiming.

4. **Conjecture scope narrowed correctly**: Conjecture 1 is now explicitly scoped to "batch-normalized ResNet trained with AdamW on CIFAR-scale data to convergence, with moderate lambda~5e-4." This matches the evidence.

5. **Conclusion expanded**: ~400 words with structured subsections covering the framework, null result, cosine stability finding, statistical caveat, and research agenda.

6. **All 6 figures verified**: Visual audit confirms Figures 1-6 are present, correctly labeled, and data-consistent with Tables 2-5.

---

## Research Contribution Assessment

### Novelty (7/10)

The Phi Modulator Framework is a genuine conceptual contribution. Expressing CWD, SWD, cosine schedules, AdamWN, AlphaDecay, and compositions as instances of a single function phi(t, theta, g) along four axes is the kind of unifying abstraction the field needs. Table 1 (method catalog) is the paper's single highest-value artifact — it organizes a fragmented literature in a way that will be useful to researchers regardless of the empirical findings.

The diagnostic metrics (BEM, CSI, AIS) are novel and address a real gap. The finding that AIS is an intrinsic network property (consistent across all methods including random_mask) directly challenges CWD's motivating assumption and is insightful. The BEM concept — distinguishing budget effects from modulation strategy — is the right diagnostic abstraction.

However, novelty is limited because (a) the null result under AdamW is consistent with prior hints in the literature (Xie & Li 2024, Kosson et al. 2023), (b) Proposition 1 is trivially true, and (c) the unification has not yet yielded a non-trivial theoretical prediction.

### Technical Soundness (6/10)

The experimental design is careful. The hyperparameter fairness protocol (all methods share identical base hyperparameters, no per-method grid search) is correct for isolating Phi modulation effects. Paired t-tests with Bonferroni correction and Cohen's d are appropriate. The mechanistic analysis (weight norm convergence under AdamW vs divergence under SGD) provides genuine explanatory insight.

Weaknesses: (1) The mechanistic hypothesis is qualitative intuition backed by order-of-magnitude estimates, not a formal proof. (2) CSI component weights are unjustified and the referenced sensitivity analysis (Appendix C) does not exist. (3) BEM for half_lambda requires a manual correction footnote — a known pipeline bug that has been worked around rather than fixed. (4) The AIS threshold (>0.2 = informative) is internally inconsistent: all methods including random_mask exceed it, yet the conclusion is "alignment is not exploitable."

### Experimental Rigor (6/10)

The 84-experiment design is well-executed at CIFAR scale. Statistical testing is more thorough than most optimization papers. The SGD negative control is the right methodological choice for establishing optimizer-specificity and substantially strengthens the paper.

Two critical gaps remain: (1) Only two of the four Phi modulation axes are tested. Spatial (AlphaDecay) and target-norm (AdamWN) axes appear in Table 1 but have no experiments. (2) CIFAR-100 SGD experiments are incomplete (random_mask: 2 seeds, no_wd: 1 seed), making the 84-run benchmark claim technically overstated.

The architectural scope (ResNet-20 only) is the paper's most serious weakness. At 270K parameters with lambda=5e-4, weight decay is a second-order perturbation by the paper's own analysis. This makes the null result almost mechanically guaranteed — the experiment cannot distinguish "WD modulation never matters under AdamW" from "WD modulation doesn't matter at CIFAR scale."

### Reproducibility (7/10)

Strong: unified codebase extending why-weight-decay infrastructure, seed control at Python/NumPy/PyTorch/CUDA level, complete hyperparameter documentation, open experiment data in the repository. The visual audit confirms all table values match raw data files.

Weaknesses: Appendix B and Appendix C are referenced but do not exist. The BEM pipeline has a known bug for half_lambda (manually corrected in tables). CIFAR-100 SGD is incomplete.

---

## Issues Requiring Resolution Before Submission

### Critical (would cause rejection alone)

**1. Experimental scope does not support conjecture breadth.** Even with the narrowed scope (CIFAR-scale BN-ResNets), a NeurIPS reviewer will ask: does the invariance hold for VGG (no skip connections)? For ResNet-50? The paper cannot answer either question. VGG-16-BN experiments on CIFAR-10/100 are achievable within the current setup (~42 additional experiments, ~10 GPU hours) and would directly test whether BatchNorm's effect on weight decay is the key architectural variable or whether the result is more general.

**2. Spatial and target-norm axes have zero experimental coverage.** Table 1 presents a four-axis taxonomy as the paper's conceptual framework. The experiments test three of seven listed methods as actual dynamic strategies (CWD: directional; SWD: temporal-gradient; cosine: temporal) and nothing in the spatial or target-norm axes. Adding AdamWN is straightforward — it is already specified in Table 1 and requires only a phi function that computes max(0, 1 - tau/||theta||).

### Major (significantly weakens paper)

**3. N=3 seeds is insufficient for a null-result paper's primary claim.** TOST at delta=+/-0.5% confirms equivalence for 1/12 comparisons. This is honest but strategically weak: a reviewer can reasonably argue the study is underpowered to confirm equivalence. The SGD result (p=0.002, d=8.4) demonstrates the design can detect effects when present, which makes the power argument cut both ways.

**4. CIFAR-100 SGD is incomplete.** The 84-experiment benchmark claim cannot be verified from available data. Completing 3 missing experiments is trivial and essential.

**5. Formal analysis is absent.** The absorption mechanism is stated as qualitative intuition. Even a proof in a simplified quadratic setting would transform this from "suggestive argument" to "formal result."

### Minor (should fix but do not change verdict)

**6. CWD-vs-random_mask discrepancy with published CWD paper unaddressed.** The original CWD paper reportedly found CWD outperforms random_mask. This paper finds they are equivalent under AdamW. The most likely explanation (CWD was evaluated with Lion/Muon, not AdamW) needs to be explicitly stated — otherwise a reviewer familiar with the CWD paper will treat this as a reproducibility failure.

**7. Appendix B and Appendix C referenced but absent.** Either write them or remove references.

**8. BEM implementation bug for half_lambda should be fixed.** One-line code change eliminates a confusing footnote.

**9. AIS threshold interpretation should be clarified.** Explicitly compare CWD AIS (0.368) to random_mask AIS (0.359) — the near-identical values are the key evidence that alignment signal is not better exploited by CWD, and this comparison is never made directly.

---

## Cross-Validated Evidence Summary

All numbers in the paper's SGD section verified against raw workspace data:

| Claim | Paper | Raw Data | Status |
|-------|-------|----------|--------|
| SGD constant CIFAR-10 mean | 91.22 | 91.217 | VERIFIED |
| SGD no_wd CIFAR-10 delta, p | -0.91%, p=0.002 | -0.913%, p=0.0022 | VERIFIED |
| SGD SWD CIFAR-100 delta, p | -1.07%, p=0.036 | -1.067%, p=0.0355 | VERIFIED |
| SGD constant weight norm | 64.6 | 64.57 | VERIFIED |
| SGD no_wd weight norm | 127.1 | 127.06 | VERIFIED |
| SGD weight norm spread | 97% | 96.8% | VERIFIED |
| AdamW constant CIFAR-10 mean | 90.13 | 90.13 | VERIFIED |
| AdamW weight norm range (1.2%) | 95.89-97.04 | confirmed in analysis.md | VERIFIED |

The prior supervisor review's claim that "SWD p=0.013 and half_lambda p=0.028 are fabricated or from a different dataset" is incorrect. These values do not appear in the current paper.md. The current paper correctly reports p=0.071 and p=0.121 for these comparisons in Table 5.

---

## Path to Score 8.0

The paper needs the following to cross the NeurIPS acceptance threshold:

**Priority 1 (moves score to 7.0):** VGG-16-BN experiments on CIFAR-10/100 with all 7 methods (42 experiments). Architecture generalization is the single most impactful addition.

**Priority 2 (moves score to 7.5):** (a) N=5 seeds for key comparisons (10 experiments), increasing TOST power; (b) AdamWN in the experiment suite to cover all four modulation axes; (c) Explicit resolution of CWD ablation discrepancy in Section 6.1.

**Priority 3 (moves score to 8.0):** Single-seed ImageNet/ResNet-50 experiment with 3-4 methods. Even one data point at realistic scale would transform the paper from "confirmed null result at known-trivial scale" to "confirmed null result at scale where the mechanism matters, with falsifiability at larger scales."

---

## Overall Assessment

This is a paper that has found the right research question ("when does dynamic WD help?") and built the right methodological infrastructure (Phi framework, unified benchmark, diagnostic metrics) to answer it. The SGD negative control addition was the correct next step and has been executed well. The paper is currently at 6.5/10: it would likely survive desk review at NeurIPS and receive a fair hearing from reviewers, but the experimental scope (single architecture, CIFAR only) and the missing axes (spatial, target-norm) would likely earn 3-4 reviewers who recommend rejection on these grounds.

The paper is not yet ready for submission, but the gaps are well-defined and the compute investment required to close them (~100 additional GPU hours for VGG + extra seeds + AdamWN) is modest relative to what has already been done.
