# Experiment Critique

## What Was Done Well
- 105 controlled experiments with identical base hyperparameters is a genuinely fair comparison.
- Three seeds with paired t-tests, Bonferroni correction, Cohen's d, TOST equivalence testing, and explicit power analysis is above community norm.
- The 7-method selection covers the four modulation axes well: temporal (cosine), directional (CWD), budget control (half_lambda, no_wd), and spatial (random_mask).
- SGD experiments provide a genuine boundary condition test that validates the AdamW-specific claim.

## Critical Gaps

### 1. ImageNet (Critical)
Missing for the 5th consecutive iteration. Project constraints mandate ImageNet. The paper's claim of "sufficiently overparameterized problems" is untested beyond CIFAR. Minimum viable: constant + cwd_hard + no_wd x 3 seeds on ResNet-50/ImageNet (9 runs). Must diagnose failure mode before retrying.

### 2. Figure 8 Data Contamination (Critical)
Figure 8 includes PMP-WD data points that are not in the paper's method set. This means:
- The rho=-0.379 and p=0.121 statistics may be computed over a different method set than described
- N=18 implies 6 methods x 3 seeds, but the paper has 7 methods -- the math doesn't work either way
- If PMP-WD is included in the computation, the results are for an experiment the paper doesn't describe

### 3. Figure 4 BEM Bug (Critical)
half_lambda is positioned near BEM~0.0 in the scatter plot but Table 6 reports BEM=0.500. This has persisted since iter_007. The regenerate_figures.py script needs debugging.

### 4. Incomplete VGG-16-BN Design
VGG-16-BN tested only with SGD/CIFAR-10 (21/168 possible runs). The claim that "BN enables Phi Invariance even without adaptive optimization" requires a 2x2 design: (BN/no-BN) x (AdamW/SGD). Without VGG-16-BN + AdamW, the attribution to BN vs architecture vs optimizer is confounded.

### 5. Statistical Power
N=3 gives ~15-20% power for TOST. The paper can only detect effects >= 0.7%. Adding 2 seeds for the 4 key comparisons (8 runs, ~2 GPU-hours) would substantially improve credibility.

### 6. Non-BN Contradiction
The 2/7 non-BN results show 0.11pp spread -- narrower than BN+AdamW (0.25pp). This directly contradicts the hypothesis that BN enables invariance. Testing all 7 methods would either resolve or confirm this contradiction.

## Missing Experiments (Prioritized)
1. Fix Figure 8 data (zero compute, just regeneration)
2. Fix Figure 4 data (zero compute, just regeneration)
3. Add 2 seeds for key CIFAR-10/AdamW comparisons (~2 GPU-hours)
4. ImageNet smoke test + 9 runs (~12-24 GPU-hours)
5. VGG-16-BN + AdamW/CIFAR-10 (~3 GPU-hours)
6. Complete non-BN ablation all 7 methods (~2 GPU-hours)
