# Experiment Result Analysis

## Key Results Summary

Iteration 6 executed experiments across 5 hierarchy domains (first-letter, city-country, city-continent, city-language, animal-class), multiple SAE configurations (L0 = 22/41/82/176, widths 16k/65k, layers 10/12/20), and a rate-distortion theoretical framework (CMI estimation, phase transition prediction, geometric constant, bifurcation analysis). The numbers:

- **First-letter absorption**: 15.96% at L0=82 (improved, 1204 words); 42.85% at L0=22 with perfect probes (F1=1.0)
- **Cross-domain absorption**: city-continent 6.49%, city-language 6.56%, animal-class 1.43%, city-country 0.0%
- **Control failure**: Shuffled-label controls exceed measured rates in ALL 5 domains (first-letter: 74.6% shuffled vs 15.96% measured; random probe: 11.8% vs target <2%)
- **Confound decomposition**: At L0=22, 1.4% hierarchy-driven vs 98.6% hedging (pilot claimed 96.9% hierarchy-driven -- 69x discrepancy)
- **CMI-absorption correlation**: Spearman rho = -0.383 (p=0.059, Cohen's d = -0.924) at d'=10, but reverses to positive at d'=20/30/50
- **Phase transition prediction**: L0_crit = 24.7 vs empirical 22.4 (10.2% error), but binary classification accuracy = 36% (below 64% chance baseline)
- **Geometric constant**: Degenerates for unit-normalized Gemma Scope decoders (CV=2.16%), simplifying the rate-distortion criterion
- **Bifurcation**: Both JumpReLU and L1 show bimodal distributions (original prediction falsified); JumpReLU shows L0-phase-transition, L1 uniformly high (61-67%)
- **Unsupervised detection (H4)**: Best rho = -0.125, AUROC = 0.47 -- definitively failed
- **L0-absorption profile**: Monotonic decline from 42.9% (L0=22) to 0.8% (L0=176); cross-layer stable (L10: 13.9%, L12: 16.0%, L20: 13.6%)
- **Hypothesis scorecard**: H2 falsified (1.4% vs >80% threshold), H4 falsified, H7 falsified as stated, H1 inconclusive (controls fail), H3 marginal (one dimension only), H5 mixed (OLS p=1.24e-6, GAM p=1.0), H6 underpowered (n=5)

## Debate Perspectives Summary

- **Optimist**: Emphasizes the L0=22 perfect-probe finding (42.85% true absorption), the CMI large effect size (Cohen's d = -0.924), cross-layer consistency, SAE probes outperforming dense probes, and the China-specific absorption case study. Frames the data as supporting three publishable pillars. Claims a publishable story exists around cross-domain characterization, rate-distortion diagnostics, and the multi-L0 confound decomposition.

- **Skeptic**: Identifies two fatal flaws: (1) universal control failure invalidating all absorption rate claims (shuffled 4.7x higher than measured), and (2) CMI dimension dependence (rho reverses sign at d'>10, p=0.059 uncorrected, 0.236 Bonferroni-corrected). H2 falsified by 69x. Concludes that H1 is invalidated, H3 is not supported, and recommends the paper must pivot to a methodological critique framing if controls cannot be resolved.

- **Strategist**: Recommends PROCEED (conditional) with three priority experiments totaling 4-5 GPU-hours: (1) control investigation, (2) CMI replication at L0=22, (3) activation patching on 9 core absorbed words. Notes the evidence has substantially improved since the prior PIVOT decision (CMI rho went from +0.14 to -0.383, probe F1 from 0.565 to 0.817). Proposes a conditional three-pillar paper reframing. Assigns 55% probability for main conference, 75% for workshop, 98% for any venue.

- **Comparativist**: Positions the work against SOTA (Chanin et al., Matryoshka SAE, OrtSAE, ATM-SAE, Unified SDL Theory). The cross-domain measurement is novel (first ever) but compromised by controls. The CMI diagnostic occupies a genuinely unique niche (no prior work applies successive refinement theorem to absorption). Current venue assessment: workshop paper as-is, NeurIPS/ICML if controls and CMI are resolved.

- **Methodologist**: Identifies three critical validity threats: (1) control failure across all domains, (2) CMI instability across subspace dimensions (post-hoc d' selection is researcher degrees of freedom), (3) pilot-to-full decomposition inconsistency. Rates reproducibility at 3/5. Notes that at the quality-gated level, first-letter absorption is actually 0% (zero parents pass F1>0.85 AND show absorption). Recommends resolving the control failure as an absolute prerequisite.

- **Revisionist**: The most radical perspective. Argues that 98.6% of what the metric calls "absorption" is hedging, meaning the true absorption rate is approximately 9/1196 = 0.75%, not 13-43%. Proposes reframing the entire paper around "what does the absorption metric actually measure?" Generates three new testable hypotheses (hedging dominance, shuffled control diagnostic, constant-absorption core). Concludes the previous mental model was fundamentally wrong.

## Analysis

### 1. Method Feasibility

The core measurement method -- the Chanin probe-based absorption metric -- does NOT transfer reliably from GPT-2 Small (where it was developed) to Gemma 2 2B with JumpReLU SAEs. The universal control failure (shuffled > measured in all 5 domains) is definitive evidence that the metric, in its current form on this architecture, cannot distinguish genuine hierarchical absorption from noise. This is not a marginal calibration issue: shuffled labels produce 4.7x the "absorption" of true labels on the improved first-letter experiment. The method is measuring something real (consistent across layers), but the interpretation as hierarchy-driven competitive exclusion is unwarranted without control validation.

The rate-distortion theoretical framework (CMI estimation, phase transition prediction) is conceptually sound but empirically fragile. The CMI correlation exists only at d'=10 out of four tested dimensions, which constitutes post-hoc optimization over estimator hyperparameters. The phase transition prediction's "excellent scale match" is partially circular (lambda fitted from the same data).

### 2. Performance

Against baselines, the results are decidedly mixed:

- **Replication**: First-letter absorption at 15.96% (L0=82) replicates the Chanin et al. range (15-35%) in magnitude, which is a positive signal that the basic measurement pipeline works.
- **Cross-domain**: Rates of 0-6.6% are first-of-their-kind measurements, but they all fall below their own shuffled controls, meaning net signal is negative.
- **Theory**: The CMI rho of -0.383 at d'=10 shows a large effect size (Cohen's d = -0.924), substantially improved from the previous rho = +0.14. But the sign reversal at d'>10 and Bonferroni-corrected p=0.236 prevent any robust claim.
- **Against the proposal's own pre-registered gates**: H1 target was >=5% in 2+ domains -- formally met (city-continent 6.5%, city-language 6.6%) but invalidated by control failure. H2 target was >80% hierarchy-driven -- falsified at 1.4%. H3 target was rho < -0.3 -- formally met at d'=10 only, not after correction.

### 3. Improvement Headroom

There is a clear, specific path to improvement -- the Strategist and Synthesis both identify three experiments (control investigation, CMI replication at L0=22, activation patching) that could decisively resolve the two blocking issues:

- **Control investigation** (2-3 GPU-hours): Could reveal either that the metric needs threshold recalibration (fixable) or that the control failure is a structural property of high-dimensional SAE feature spaces (itself a publishable finding). Either outcome advances the paper.
- **CMI at L0=22** (1 GPU-hour): At this operating point, all 25 letter probes achieve F1=1.0, eliminating the probe quality confound. Absorption rates span 0-80%, providing maximum variance. This is the cleanest possible test of the rate-distortion theory.
- **Activation patching** (0.5-1 GPU-hour): The only metric-independent validation possible for genuine absorption.

Total: 4-5 GPU-hours. This is a small, bounded investment with high information gain.

### 4. Time-Cost Tradeoff

The project has already invested 6 iterations of substantial compute and analysis. The question is whether 4-5 additional GPU-hours justify continuing versus pivoting.

Arguments for continuing:
- The evidence landscape has genuinely changed since the prior PIVOT decision. The CMI signal went from rho=+0.14 (wrong direction) to rho=-0.383 (correct, large effect). Probe quality improved from F1=0.565 to 0.817. These are not marginal improvements.
- The L0-absorption monotonic profile, the hedging dominance finding (98.6%), and the geometric constant degeneration are all publishable regardless of the control and CMI outcomes.
- The diagnostic/characterization niche is genuinely unoccupied (Comparativist confirms no concurrent paper does this).
- 4-5 GPU-hours is a trivial investment compared to the 6 iterations already completed.

Arguments for pivoting:
- The original 7-hypothesis structure is dead (3 falsified, 2 inconclusive).
- The control failure has persisted and worsened despite targeted efforts.
- Even the "best case" scenario (Strategist's 35% probability for controls fixable + CMI replicates) is a minority outcome.

### 5. Critical Objections

The Skeptic's two "fatal flaws" are the most serious objections:

**F1 (Universal control failure)**: This is real and blocking. However, it is not fatal to the project as a whole because: (a) the control failure itself is a novel finding about metric reliability, (b) the L0-absorption profile is valid independent of the control interpretation, and (c) the planned control investigation has well-defined outcomes in either direction.

**F2 (CMI dimension dependence)**: This is real and concerning. However, the Strategist correctly identifies that the L0=22 replication eliminates the largest confound (probe quality) and tests the theory at its strongest operating point. The outcome of this one experiment is decisive.

The Revisionist's reframing (98.6% hedging, true absorption ~ 0.75%) is intellectually compelling but represents one interpretation of the confound decomposition. The Methodologist notes that the classification depends on which L0 values are in the sweep -- adding L0=300 could reclassify the remaining 9 "hierarchy-driven" words. The 9 persistent core words need activation patching validation.

## Decision Rationale

I weigh the following factors:

**1. The evidence has materially improved since the prior PIVOT decision (confidence 0.75).**

The prior PIVOT was based on: CMI rho = +0.14 (wrong direction), probe F1 = 0.565 (poor), and 3/6 hypotheses falsified. The current state: CMI rho = -0.383 (correct, large effect), probe F1 = 0.817, and new positive results (phase transition prediction, bifurcation analysis, geometric constant) that did not exist at PIVOT time. Overturning the prior PIVOT is justified by this evidence shift.

**2. Three robust findings already exist, independent of the blocking issues.**

Regardless of what happens with the control investigation and CMI replication:
- The hedging dominance finding (98.6% at L0=22) is a genuine empirical contribution.
- The L0-absorption monotonic profile (42.9% to 0.8%) is the most robust finding in the dataset.
- The universal control failure is itself a methodological contribution (no prior work has attempted to validate the Chanin metric on a different architecture).

These three findings constitute a floor that is publishable at minimum at the workshop level, and plausibly at TMLR.

**3. The remaining experiments have bounded cost (4-5 GPU-hours) with high expected information gain.**

The three priority experiments are well-designed, have clear pass/fail criteria, and their outcomes map cleanly to paper scenarios. This is not an open-ended research direction -- it is a bounded validation exercise.

**4. The project occupies a genuinely distinct niche.**

The Comparativist's analysis confirms that no concurrent work does cross-domain absorption measurement with confound decomposition and information-theoretic diagnostics. The field has moved toward mitigation (Matryoshka, OrtSAE, ATM-SAE), leaving the diagnostic space open.

**5. The risks are bounded and the floor is acceptable.**

Even in the worst case (Scenario D: controls not fixable AND CMI does not replicate, 25% probability per Strategist), the paper is still publishable as a methodology audit paper. The expected value of continuing is clearly positive.

**Counterargument addressed**: The Skeptic argues that the control failure invalidates "ALL absorption rate claims." This is technically correct for absolute rate claims but does not invalidate relative comparisons (e.g., the L0-absorption monotonic decline), the hedging dominance finding (which is about the composition of false negatives, not their absolute rate), or the control failure itself as a finding. The Revisionist's reframing ("what does the metric actually measure?") is a viable paper regardless.

**Key conditional**: The PROCEED verdict is explicitly conditional on completing the three priority experiments BEFORE writing. If both the control investigation and CMI replication fail (Scenario D), the paper is reframed to a methodology audit. This decision will be revisited after Phase 1 experiments.

DECISION: PROCEED
