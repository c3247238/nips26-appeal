# Writing Critique

## Overall Assessment: 5/10 (Not ready for submission)

The prose is technically competent and well-structured, but the paper suffers from a fundamental gap between what it promises and what it delivers. The abstract and introduction set up an ambitious 4-theorem framework validated across architectures and scales; the experiments deliver a single architecture on CIFAR with no validation of 2 of 4 theorems.

## Critical Issues

### 1. Promise-Delivery Gap
The abstract promises:
- "certified convergence band" -- but V_t empirically increases, undermining the certificate
- Subsumption proof for ">=95% of training steps" -- no supporting table/figure
- "weight decay illusion" as a general finding -- tested on exactly 1 architecture + 1 optimizer

The introduction lists 3 contributions; contribution 2 (cumulative alignment bound, Theorem 2) has zero empirical validation.

### 2. Missing Figures and Appendix
- Figure 1 is a markdown placeholder, not an actual image
- No certified band visualization (the core theoretical contribution)
- No appendix despite "Full proof in Appendix A" reference in Section 4.2
- Only 6 of 8 planned figures present; cumulative alignment scatter and BN vs NoBN comparison are missing

### 3. Data Inconsistencies
- CIFAR-100 spread reported as both 0.76pp and 0.58pp in different locations
- Random mask (90.12%) cited in Section 7.3 but absent from all tables
- Half_lambda appears in taxonomy table but not results table, with no explanation
- Iter_003 vs iter_006 method ordering reversal acknowledged in supervisor analysis but hidden from the reader

### 4. Theorem Numbering
Theorems appear out of order (1, 3, 2, 4) which is needlessly confusing.

### 5. Theory-Experiment Mismatch
The entire theoretical framework assumes SGD dynamics. The experiments use AdamW. This is never discussed or justified. The PMP-WD costate approximation (momentum buffer as costate proxy) is derived for SGD momentum but applied to AdamW's first moment estimate, which is bias-corrected and scaled by the inverse second moment.

## Structural Issues

- Sections 3 and 4 boundary is blurred. Theorem 2 (generalization) is in Section 4 (convergence band), creating a thematic mismatch.
- Section 6 (Experiments) covers only CIFAR-10 in detail; CIFAR-100 gets 1 paragraph with no table.
- No dedicated results table for CIFAR-100 -- numbers are embedded in prose.
- Discussion (Section 7) makes predictions about non-BN architectures, ImageNet, and large models but provides zero supporting evidence.

## What Works
- Clear notation throughout (cross-checked against notation.md)
- Well-organized related work section with fair treatment of prior methods
- The "weight decay illusion" framing is compelling and attention-grabbing
- Algorithm 1 (PMP-WD) is clearly stated and implementable
- Table 1 (phi taxonomy) is genuinely useful and illuminating

## Recommendations (Priority Order)
1. Add certified band visualization as a figure (this IS the paper)
2. Create actual Figure 1 diagram
3. Add CIFAR-100 results table
4. Add half_lambda + random_mask to main results
5. Add subsumption verification table
6. Add Appendix A with Theorem 1 proof
7. Fix CIFAR-100 spread consistency
8. Renumber theorems sequentially
9. Discuss SGD-vs-AdamW theory gap explicitly
10. Either validate Theorem 2 empirically or remove it as a contribution
