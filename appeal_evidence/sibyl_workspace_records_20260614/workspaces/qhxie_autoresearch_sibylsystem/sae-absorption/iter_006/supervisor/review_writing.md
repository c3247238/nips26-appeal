# Supervisor Review: Iteration 6

## Score: 6.5 / 10 (Borderline Reject)

**Verdict: CONTINUE** -- The paper has a defensible core contribution but needs one more iteration to resolve three critical issues before it could survive NeurIPS review.

---

## Executive Summary

This paper audits the Chanin absorption metric on Gemma 2 2B with JumpReLU SAEs and reports three findings: (1) universal control failure across five hierarchy domains, (2) an L0-absorption phase transition from 42.9% to 0.8%, and (3) a CMI-absorption correlation at d'=10. The metric audit and L0 phase transition are genuine contributions that the SAE interpretability community needs. However, the paper is undermined by three critical problems: the CMI pillar does not survive multiple comparison correction and reverses at other tested dimensions; numerical discrepancies between source data files erode trust in all reported numbers; and the confound decomposition's hedging classification is permissive enough that its 98.6% figure is near-guaranteed by construction.

---

## Dimension Scores

### Novelty: 7/10

The paper occupies a genuinely useful niche. Key novelty elements:

- **First audit of the Chanin absorption metric on JumpReLU SAEs** -- no prior work validates this metric beyond L1-ReLU on GPT-2 Small. The universal control failure (shuffled > measured across all 5 domains) is a finding that should change community practice.
- **First cross-domain absorption measurements** -- city-continent, city-language, animal-class absorption rates are first-of-their-kind, even if the control failure renders the absolute rates uninterpretable.
- **L0 phase transition** -- the monotonic decline from 42.9% to 0.8% with cross-layer stability (CV < 10%) has not been reported before.
- **Rate-distortion framework application** -- applying successive refinement theory to absorption is conceptually novel, though the empirical validation is insufficient.

The novelty would be stronger if the paper committed to being a metric-audit paper rather than trying to also be a theory paper. The cross-domain measurement novelty is partially undermined by the paper's own admission that the rates are uninterpretable.

### Technical Soundness: 5/10

Three critical soundness issues:

1. **CMI dimension instability.** The CMI-absorption correlation at d'=10 (rho=-0.383, p=0.059) reverses to positive at d'=20,30,50. Bonferroni-corrected p=0.236. A finding that reverses sign under a modest parameter change and does not survive standard corrections is not evidence for the underlying theory, regardless of effect size. The paper presents Cohen's d=-0.924 as if it validates the result, but effect size estimates for non-robust effects are meaningless.

2. **Numerical discrepancies.** I independently verified: `cmi_estimation.json` computes absorbed mean CMI = 0.6492 using its own partition (13 absorbed, 9 non-absorbed letters from first_letter_improved rates). The paper reports 0.687 with U=41 and p=0.042 -- numbers that match `phase_transition_prediction.json`, which uses a different partition. This means the paper's primary CMI statistics are computed from an inconsistent analysis pipeline. The L0=82 absorption rate appears as both 14.39% and 15.96% depending on which pipeline is used. These are not rounding differences but different analyses.

3. **Hedging classification bias.** The operational definition classifies any false negative that resolves at any higher L0 as "hedging." Since L0=176 (8x more active features) resolves nearly everything, this guarantees a high hedging percentage. The classification does not check whether the specific parent feature fires -- only whether the token stops being a false negative at higher L0. The 98.6% hedging figure is therefore an upper bound, potentially a very loose one.

### Experimental Rigor: 6/10

Positive elements:
- The control suite (random probe, shuffled labels, untrained SAE, dense probe) is well-designed and the universal failure across 5 domains is convincing.
- The L0 sweep across four operating points with consistent methodology is rigorous.
- Cross-layer validation (layers 10, 12, 20) with CV < 10% demonstrates stability.
- Bootstrap CIs (10,000 resamples) are properly reported.
- Negative results (H2, H4, H6, H7) are reported transparently with specific expected vs. observed values.

Missing experiments and gaps:
- **Activation patching on 9 core words** is planned but not executed -- this is the single most impactful missing experiment. It would provide causal evidence for or against competitive exclusion on JumpReLU SAEs.
- **Partial correlation between CMI and absorption controlling for probe F1** is not computed despite the paper reporting rho=-0.67 between absorption and probe F1.
- **Tightened hedging classification** checking whether the specific parent latent fires at higher L0 is not implemented.
- **Leave-one-out sensitivity** for the CMI correlation is not reported (letter S is a major outlier: high CMI, high absorption).
- **CMI estimation diagnostics** (bootstrap CIs, convergence curves) are absent.
- **Unsupervised pipeline** produced literally zero matching pairs -- worse than a weak detection system, it is a non-functional one.

### Reproducibility: 5/10

- The SAE configurations are clearly specified (Gemma Scope model/layer/width identifiers).
- The measurement protocol is described in sufficient detail for reimplementation.
- However, the pipeline-level inconsistency (two different absorption rates for the same configuration, two different partitions yielding different CMI statistics) means even the authors cannot reproduce consistent numbers from their own data.
- The CMI estimation's sensitivity to d' undermines any claim that the analysis is reproducible in the scientific sense -- small, apparently arbitrary choices (d'=10 vs d'=20) change the conclusion.
- No code release is mentioned, though the experimental code exists in `exp/code/`.

---

## What Works Well

1. **The control suite and its universal failure** -- this is the paper's strongest argument, presented clearly and convincingly across all five domains. The grouped bar charts and Table 2 make the point inescapable.

2. **The L0 phase transition** -- the monotonic decline from 42.9% to 0.8% with cross-layer stability is the most robust finding. The shaded phase-transition zone in Figure 3 communicates the result effectively.

3. **Negative results reporting** (Section 7.4) -- exemplary. Four falsified hypotheses, each with specific pre-registered target vs. actual result. This is model scientific practice.

4. **The confound decomposition conceptual framework** -- even though the operational definition needs tightening, the idea of decomposing false negatives by cross-L0 persistence is sound and novel.

5. **The 9 persistent core words** -- these are a compelling finding that grounds the abstract statistical analysis in concrete, inspectable examples.

---

## Critical Path to 8.0

Three actions that would move the score from 6.5 to approximately 7.5-8.0:

1. **Activation patching on the 9 core words.** This is the highest-value experiment remaining. Positive results confirm competitive exclusion exists at small scale and validate the confound decomposition. Negative results strengthen the "it's all hedging" narrative. Either outcome strengthens the paper.

2. **Data pipeline reconciliation and CMI reframing.** Choose one canonical pipeline, recompute all statistics consistently, fix the numerical discrepancies, and downgrade the CMI finding from "primary" to "suggestive directional evidence." Add Bonferroni-corrected p-values wherever uncorrected ones appear.

3. **Tightened hedging definition.** Check whether the specific parent latent fires at higher L0 (not just whether the token stops being a false negative). Report both permissive and strict definitions. This could either strengthen or weaken the 98.6% figure, but either outcome is more credible than the current permissive-only report.

---

## Hypothesis Scorecard (Cross-Validated)

| Hypothesis | Pre-registered Target | Result | Verdict |
|---|---|---|---|
| H1: Cross-domain >=5% in 2+ domains | >=5% | City-continent 6.49%, city-language 6.56% | Formally met but invalidated by control failure |
| H2: Hierarchy-driven >80% at L0=22 | >80% | 1.4% | **FALSIFIED** (reversal from pilot's 96.9%) |
| H3: CMI rho < -0.3 | < -0.3 | -0.383 at d'=10 only | Marginal at one dimension, does not survive correction |
| H4: Unsupervised rho > 0.3 | > 0.3 | -0.125, AUROC=0.47 | **FALSIFIED** |
| H5: Width-L0 interaction | p < 0.05 | OLS p=1.24e-6, GAM p=1.0 | Mixed (model-dependent) |
| H6: Hierarchy predictor rho > 0.3 | > 0.3 | Underpowered (n=5) | **UNDERPOWERED** |
| H7: JumpReLU bimodal, L1 continuous | KS p < 0.1 | Both bimodal | **FALSIFIED as stated** |

The fact that 4/7 hypotheses are falsified is not itself a problem -- honest reporting of negative results is valuable. But it signals that the research plan was not well calibrated to the experimental reality, and the paper needs to reorganize around what the data actually show rather than what was predicted.

---

## Recommended Paper Structure

If the paper were restructured around its actual strengths:

1. **Primary contribution: Metric audit** -- universal control failure, hedging decomposition (with tightened definition), practical recommendation to validate controls before measuring absorption on new architectures.

2. **Secondary contribution: L0 phase transition** -- most robust empirical finding, cross-layer stable, directly actionable (increase L0 to reduce metric output).

3. **Tertiary contribution: Rate-distortion diagnostic** -- frame as "suggestive evidence motivating future work," not a validated diagnostic. Report all d' results equally.

4. **Supporting: Cross-domain characterization** -- reframe as the experimental vehicle for the metric audit, not an independent contribution.

This reframing is more honest about the evidence and positions the paper as a strong methods-audit contribution rather than a weaker three-pillar paper where two pillars are fragile.
