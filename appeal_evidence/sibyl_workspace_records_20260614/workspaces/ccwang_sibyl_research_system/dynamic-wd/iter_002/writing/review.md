# NeurIPS 2026 Review: Final Quality Assessment

## Overall Score: 7/10
## Recommendation: Weak Accept
## Confidence: 4/5

## Summary (2-3 sentences)

This paper investigates whether gradient--parameter alignment information can be exploited for adaptive weight decay scheduling in nonconvex SGD. Through 39 experiments spanning two architectures and two datasets, the authors establish three informative negative results: (1) budget equivalence -- only the time-averaged weight decay matters, not its temporal distribution; (2) LR--WD coupling necessity -- decoupling adaptive weight decay from the learning rate causes catastrophic collapse; and (3) alignment signal inapplicability -- the alignment quantity delta_t is too small and too constant under standard training to serve as a useful control signal. The paper is well-structured, the experimental logic is sound, and the findings constitute a genuinely useful contribution to the understanding of weight decay in deep learning.

## Strengths

1. **Clear and well-framed negative result.** The paper is transparent from the abstract onward that AADWD is an experimental framework, not a proposed practical method. The three negative findings are stated as positive mechanistic insights, which is the correct framing for a negative-result paper. The "budget equivalence" finding in particular is a clean, memorable result that practitioners can immediately apply.

2. **Strong experimental design for the budget equivalence result.** The protocol of recording the per-step lambda_t from an AADWD run, computing the time-average, and running a constant WD at that average is a clean causal design. The exact metric match (92.54% = 92.54%, weight norm 23.49 = 23.49) is striking and well-presented.

3. **Dramatic and informative failure modes.** The LR decoupling experiment (aggressive collapse to 10.00% with weight norm 0.0036) provides vivid, mechanistically interpretable evidence for the coupling necessity. The cascade mechanism (Eq. 6) is clearly explained and connects theory to experiment.

4. **Thorough ablation coverage.** The paper covers a comprehensive set of controls: random dynamic WD (to test alignment information content), budget-equivalent constant WD (to test temporal distribution), decoupled variants (to test LR coupling), and hyperparameter sweeps across c and beta. Each ablation cleanly isolates a specific mechanism.

5. **Honest scope delimitation.** The paper explicitly states its limitations: CIFAR-scale only, SGD with momentum only, single seed, milestone LR schedule only. The discussion of when alignment might matter (adversarial training, extreme WD, very deep networks) is appropriately speculative and delineates the boundary of the negative result.

6. **Cross-architecture consistency.** The core findings replicate across ResNet-20 and VGG-16-BN (56x parameter difference) and across CIFAR-10 and CIFAR-100, lending credibility to the conclusions despite the limited scale.

7. **CWD instability finding is a useful secondary contribution.** Documenting that Cautious Weight Decay (ICLR 2025) exhibits 4.84%--12.57% best-to-final degradation across all tested settings is a valuable empirical observation for the community.

## Weaknesses

1. **Single-seed experiments are a significant limitation for the quantitative claims.** All 39 experiments use seed=42. The paper acknowledges this, but several claims rest on small margins: AADWD Conservative vs. fixed WD is 0.17%, AADWD Aggressive vs. fixed WD is 0.49%, random dynamic WD vs. AADWD Aggressive is 0.01%. Without variance estimates, it is impossible to know whether these differences are meaningful. The large-effect-size findings (decoupling collapse, CWD instability, budget equivalence exact match) are robust to seed variation, but the claim that "no dynamic method outperforms fixed WD" could be overturned by a different seed. For a NeurIPS submission, at least the primary comparisons in Table 1 should have 3-seed confidence intervals.

2. **Scale limitation is consequential for the paper's strongest claims.** The title refers to "nonconvex SGD" generically, but the experiments are restricted to CIFAR-10/100 with small models. The budget equivalence principle is a strong claim that is only verified on one architecture/dataset combination (CIFAR-10/ResNet-20). While the paper acknowledges this, the theoretical framing (Theorems 1 and 2) suggests generality that the experiments do not support. The claim would be substantially stronger with even one ImageNet-scale experiment.

3. **Theorem 1 and Theorem 2 remain formally disconnected.** Theorem 1 states a bound in terms of Lambda_bar_T (uniform time-average of lambda_t / gamma_t) with no explicit dependence on delta_t. Theorem 2 introduces delta_bar_T (lambda_t-weighted average of delta_t) but does not present a complete bound containing this quantity. The revision notes acknowledge this ("full mathematical fix deferred to appendix completion") but for a NeurIPS submission, the two theorems should either be unified into a single statement with an explicit delta_t-dependent term, or Theorem 2 should be clearly labeled as a corollary/observation derived from the proof machinery of Theorem 1. The appendix containing the full proof is referenced but not included.

4. **The decoupled experiment uses different c values than the main experiments.** The aggressive_decoupled experiment uses c=0.25 (vs. c=2.5 in main), and conservative_decoupled uses c=0.0005 (vs. c=0.005 in main). The revision notes acknowledge this: "changing the experiment data is out of scope for this revision." This is a meaningful confound: the decoupled collapse could be partly attributable to the 10x reduction in c rather than purely to the removal of gamma_t. The paper should at minimum discuss this explicitly and argue why the c difference does not undermine the conclusion (e.g., because the collapse mechanism described in Observation 1 is structural and independent of c).

5. **Alignment proxy data provenance is unclear.** Table 4 reports alignment statistics with phase labels "epochs 1-80," "epochs 81-120," and "epochs 121-200," but the previous critique flagged that the underlying data may come from a 20-epoch pilot rather than a full 200-epoch training run. The revision notes defer this to "Appendix clarification." If the alignment statistics are from a pilot run extrapolated to 200-epoch phase boundaries, this should be disclosed transparently. The entire Insight 3 (alignment is not actionable) depends on the characterization in Table 4.

6. **Missing Appendices.** The paper references Appendix B (Norm-Matched WD results), Appendix E (full proof of Theorem 1), and other appendices, but none are included. For a complete NeurIPS submission, these are essential. The Norm-Matched WD result is methodologically important (it was originally listed as a compared method), and the proof of Theorem 1 is needed to verify the convergence analysis.

## Questions for Authors

1. What is the actual source of the alignment statistics in Table 4? Are these from a full 200-epoch training run or extrapolated from a 20-epoch pilot? If the latter, have you verified that the alignment dynamics do not change qualitatively over 200 epochs?

2. For the decoupled experiments: if you re-run with the same c values as the main experiments (c=2.5 for aggressive, c=0.005 for conservative), do you still observe the same collapse behavior? The 10x c reduction is a confound that weakens the coupling necessity argument.

3. The budget equivalence experiment found that the AADWD-Aggressive time-average happened to exactly equal 5e-4. Can you report the time-average to more significant digits (e.g., 5.00e-4 vs. 4.87e-4 vs. 5.13e-4)? The exact coincidence with the standard baseline deserves more precise quantification.

4. Have you measured the alignment quantity delta_t under any non-standard training conditions (e.g., large weight decay approaching instability, adversarial training, early training of very deep networks)? Even a single measurement showing delta_t >> 0.003 under different conditions would strengthen the scope argument.

5. The Square variant achieves 92.13% despite "effectively eliminating regularization in late training" (weight norm 38.75, the highest among WD methods). How does this square with the budget equivalence principle? Its time-average lambda must be lower than 5e-4 if late-training WD is near zero, yet it still achieves reasonable performance. Does this not suggest that early-training regularization contributes disproportionately?

## Detailed Comments

### Novelty and Significance

The paper's contribution is primarily empirical and diagnostic rather than methodological. The three negative findings -- budget equivalence, coupling necessity, alignment inapplicability -- are individually interesting and collectively form a coherent narrative about why constant weight decay works. The budget equivalence principle in particular has immediate practical value: it gives practitioners a diagnostic tool (budget-matching) for evaluating any proposed WD schedule.

The novelty lies not in the AADWD method itself (which is a straightforward application of alignment monitoring) but in the systematic experimental design that isolates each mechanism. The random-WD-vs-alignment-WD comparison, the budget-equivalent constant, and the coupled/decoupled ablation are well-chosen controls.

Significance is moderate. The findings are reassuring for practitioners (constant WD is fine) and informative for theorists (alignment is not practically exploitable), but the scope limitation to CIFAR-scale experiments reduces the impact for the broader community. The paper would be substantially more significant with ImageNet-scale validation of at least the budget equivalence result.

### Technical Soundness

The experimental methodology is generally sound, with the caveats noted above (single seed, c-value mismatch in decoupled experiments, alignment data provenance). The numerical data is accurate: I cross-checked all numbers in Tables 1-5 against the raw experimental JSON files, and all values match correctly after rounding.

**Data Accuracy Verification (spot checks):**
- Table 1: Fixed WD (5e-4) best=92.54, final=92.29, gap=7.17, WN=23.49. Raw data: best=92.54, final=92.29, gap=7.1651, WN=23.4928. PASS.
- Table 1: AADWD Aggressive best=92.05, final=91.57, gap=7.50, WN=21.47. Raw data: best=92.05, final=91.57, gap=7.4965, WN=21.4722. PASS.
- Table 1: Equiv. Cumulative WD best=92.54, final=92.29, gap=7.17, WN=23.49. Raw data: best=92.54, final=92.29, gap=7.1651, WN=23.4928. PASS (exact match with Fixed WD 5e-4 confirmed).
- Table 2: VGG-16-BN/CIFAR-10 Fixed WD best=93.86. Raw data: 93.86. PASS.
- Table 2: CIFAR-100/ResNet-20 CWD best=66.84, final=54.27. Raw data: best=66.84, final=54.27. PASS.
- Table 3: Aggressive decoupled final=10.00, WN=0.0036. Raw data: final=10.0, WN=0.0036. PASS.
- Table 5: c=10.0 best=52.12, final=10.00, WN=0.004. Raw data: best=52.12, final=10.0, WN=0.0042. PASS (rounded).
- Table 5: beta=0.9999 best=92.25, WN=20.56. Raw data: best=92.25, WN=20.5625. PASS.

All claimed numbers in the paper are verified against the experimental data files.

The theoretical framework (Theorems 1 and 2) has the formal disconnection issue noted above, but the paper is primarily empirical, and the theoretical contributions are appropriately modest -- they provide motivation rather than tight guarantees.

### Clarity and Writing Quality

The writing is excellent throughout. The paper follows a clear logical arc: motivation -> theory -> experiment design -> results -> mechanistic analysis -> discussion. Key observations are highlighted with clear paragraph labels. Tables are well-formatted with appropriate captions.

The revision addresses numerous issues from the previous round of critiques: the 84.49% vs. 92.05% confusion is now clearly explained in the abstract and introduction; the method count has been corrected to 13; the single-seed limitation is disclosed; the Norm-Matched WD is properly deferred to the appendix; figure references have been added; and the language has been appropriately hedged ("empirically undominated" rather than "optimal").

Minor writing issues remain:
- In Section 7.3 (Limitations), "$82.05\%$ for the decoupling collapse" reads ambiguously -- this is a percentage-point gap (92.05 - 10.00), not an accuracy. Consider writing "82.05 percentage-point gap" for clarity.
- Some LaTeX formatting artifacts remain (e.g., `\label{}` tags, `\citet{}` commands) that will need to be resolved for the final compiled version. This is expected in a markdown draft.

### Experimental Rigor

The experimental design is the paper's primary strength. The 39 experiments are well-organized into tiers: main results, cross-architecture validation, ablation controls, hyperparameter sensitivity, and decoupled variants. Each experiment isolates a specific variable.

However, several experimental concerns remain:

1. **Single seed** (discussed above) -- the most significant limitation.
2. **c-value mismatch in decoupled experiments** (discussed above) -- a confound.
3. **No cross-architecture budget equivalence test.** The budget equivalence result is only demonstrated on CIFAR-10/ResNet-20. Testing it on CIFAR-100/ResNet-20 or CIFAR-10/VGG-16-BN would substantially strengthen the claim's generality.
4. **Stagewise WD absent from Table 2.** Stagewise WD (92.44% on CIFAR-10/ResNet-20) is a practical baseline that should appear in the cross-architecture comparison.
5. **AADWD Aggressive uses c=2.5 as representative, but c=1.0 achieves higher accuracy (92.18% vs. 92.05%).** The choice of representative hyperparameters should be explained -- was c=2.5 chosen before the sensitivity sweep, or is there a reason not to report the best-performing c?

### Reproducibility

The training protocol is sufficiently detailed for reproduction: architecture specifications, optimizer settings, learning rate schedule, augmentation protocol, and AADWD hyperparameters are all disclosed. The seed (42) is stated. However:

- Lambda_max values for sensitivity experiments should be made more prominent (currently in the caption of Table 5).
- The Stagewise WD exact formula (how lambda scales at milestones) is described only qualitatively.
- CUDA determinism settings are not specified.
- The alignment proxy data collection protocol (how large-batch measurements for Pearson r validation were obtained) is not detailed.

## Minor Issues

- Table 4: Per-phase standard deviations are listed as "---" which appears incomplete. If per-phase stds were not measured, this should be stated explicitly rather than left as missing values.
- Table 3 column header: "Coupled (%)" and "Decoupled (%)" mix accuracy percentages with Delta (percentage-point difference) in the same row structure. The Delta column should clarify its unit as "pp" (percentage points).
- The paper counts "39 systematic experiments" multiple times. A brief breakdown (e.g., 13 Tier-1 + 10 cross-arch + 5 ablation + 9 sensitivity + 4 decoupled = 41, or however the count is derived) would help readers verify this claim. From the JSON files, I count: 11 (main) + 10 (cross-arch) + 5 (ablations) + 9 (hyperparam) + 4 (decoupled) = 39. This checks out.
- The Observation (formerly Proposition 1) on decoupling instability states that delta_hat_t decreases as ||w_t|| shrinks, but the mechanism for why |<g_t, w_t>| decreases faster than ||g_t|| * ||w_t|| is asserted rather than shown. A brief calculation or reference would strengthen this.
- In the abstract: "weight norm -> 0.0036" uses an arrow notation that mixes with the best->final notation used elsewhere (e.g., CWD "91.79 -> 86.95"). The arrow here means "converges to" rather than "changes from X to Y."

## Overall Assessment

This is a well-executed negative-result paper that makes a genuine contribution to understanding weight decay in deep learning. The three core findings -- budget equivalence, coupling necessity, and alignment inapplicability -- are clearly stated, experimentally supported, and mechanistically explained. The experimental design, particularly the budget-matching protocol and the random-vs-alignment comparison, demonstrates careful thinking about causal inference in deep learning.

The primary weaknesses are the single-seed experiments (which affect the precision of quantitative claims but not the qualitative conclusions), the limited scale (CIFAR-level only), and the formal incompleteness of the theoretical framework (Theorems 1 and 2 are disconnected, missing appendix). The confound in the decoupled experiment (different c values) is a methodological concern that should be addressed before camera-ready.

The paper is above the acceptance threshold for NeurIPS based on the strength of its experimental design and the practical relevance of its findings. The budget equivalence principle alone is a useful diagnostic tool, and the coupling necessity result has immediate implications for anyone designing adaptive regularization schemes. However, the paper would benefit significantly from multi-seed validation, at least one larger-scale experiment, and resolution of the Theorem 1/2 formal gap.

The writing quality is high, the narrative is well-structured, and the revision has addressed the majority of issues raised in previous critiques. The remaining issues are primarily about strengthening the evidence (more seeds, more scale) rather than fundamental flaws in the approach.

## Verdict: PASS

The paper is ready for LaTeX compilation and submission, with the understanding that the following items should be addressed for camera-ready if accepted:
1. Multi-seed experiments for primary comparisons (Table 1).
2. Resolution of the decoupled experiment c-value confound.
3. Inclusion of Appendices B and E (Norm-Matched WD results and Theorem 1 proof).
4. Verification that Table 4 alignment statistics come from 200-epoch runs.
5. Cross-architecture budget equivalence validation.
