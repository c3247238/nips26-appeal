# Experiment Critique: Unified Dynamic Weight Decay Framework

## Critical Issue: SGD Data Integrity Violation

The most serious problem is a direct contradiction between the paper's reported SGD statistics (Table 5) and the raw workspace data.

### Verified from workspace (`current/exp/results/sgd_baseline/cifar10/resnet20/*/seed_*/summary.json`):

| Method | Seed 42 | Seed 123 | Seed 456 | Mean | Std | Delta vs. const | p-value (paired t) |
|--------|---------|---------|---------|------|-----|-----------------|-------------------|
| constant | 91.18 | 90.98 | 91.16 | **91.107** | 0.110 | --- | --- |
| cosine_schedule | 91.07 | 91.26 | 91.08 | 91.137 | 0.106 | +0.03% | 0.833 |
| cwd_hard | 91.04 | 90.86 | 90.32 | 90.740 | 0.370 | -0.37% | 0.262 |
| half_lambda | 90.65 | 90.78 | 90.63 | 90.687 | 0.082 | -0.42% | **0.062** |
| no_wd | 90.27 | 90.00 | 90.22 | 90.163 | 0.144 | -0.94% | **0.0005** |
| random_mask | 90.29 | 90.53 | 91.12 | 90.647 | 0.429 | -0.46% | 0.202 |
| swd | 90.40 | 90.36 | 90.85 | 90.537 | 0.271 | -0.57% | **0.054** |

### Paper Table 5 claims:

| Method | Acc (%) | Delta | p-value | Significant? |
|--------|---------|-------|---------|--------------|
| constant | **91.22 ± 0.06** | --- | --- | --- |
| swd | 90.71 ± 0.16 | -0.51% | **0.013*** | YES (claimed) |
| half_lambda | 90.84 ± 0.15 | -0.37% | **0.028*** | YES (claimed) |
| no_wd | 90.30 ± 0.08 | -0.91% | <0.001** | YES (confirmed) |

### Discrepancies:

1. **SGD constant baseline**: Paper claims 91.22 ± 0.06. Raw data gives 91.107 ± 0.110. Unexplained +0.11% inflation.

2. **SGD SWD significance**: Paper claims p=0.013 (significant). Raw data gives p=0.054 (NOT significant). The paper's narrative that "SWD is significantly worse under SGD" rests on this fabricated result.

3. **SGD half_lambda significance**: Paper claims p=0.028 (significant). Raw data gives p=0.062 (NOT significant). Paper claims "weight decay magnitude matters under SGD (half_lambda: -0.37%, p=0.028)" -- this specific claim is not supported by raw data.

4. **SWD std**: Paper claims std=0.16; raw data gives std=0.271.

### Impact on the conjecture

The paper's central argument is that the SGD experiments validate the Phi Invariance Conjecture as AdamW-specific: "The same N=3 design that yields inconclusive TOST results for AdamW yields highly significant differences under SGD." With the corrected data, only **no_wd** is highly significant (p=0.0005). SWD and half_lambda are NOT significant, weakening the "SGD is clearly different from AdamW" narrative considerably.

The paper still has one valid SGD comparison (no_wd: -0.94%, p=0.0005), but the claim of "three statistically significant SGD effects" becomes "one statistically significant SGD effect."

---

## Major Issue: Experimental Scope vs. Claim Scope

The Phi Invariance Conjecture is framed as a claim about AdamW generally, but all 42 AdamW experiments use:
- ResNet-20 (~270K parameters)
- CIFAR-10 and CIFAR-100 (32x32 images, 10/100 classes)
- λ = 5×10⁻⁴ (moderate)
- 200 epochs

At this scale and lambda, the paper's own mechanistic analysis (Section 5.4) shows the Phi perturbation is at most 5-50% of the gradient update magnitude. The upper bound is only reached at late training when learning rate is small. At early training (large learning rate), Phi is overwhelmed. This means the null result is **almost mechanically guaranteed** by the experimental setting. The experiment is not testing "does WD modulation matter under AdamW" -- it is testing "does WD modulation matter when it is already a small perturbation."

**Missing experiments that would actually test the conjecture:**
1. ImageNet/ResNet-50: larger lambda is needed for larger models; weight decay's role is more consequential
2. Large λ (1e-2): the absorption argument explicitly weakens at large λ
3. VGG-16-BN or ViT: different architecture biases
4. Severely overfitting regime: weight decay as regularization (not just dynamics modifier)

---

## Major Issue: Incomplete Method Coverage

The Phi framework taxonomy covers four axes. The experiments cover:
- Temporal: SWD, cosine_schedule ✓
- Directional: cwd_hard ✓
- Spatial: **MISSING** (AlphaDecay is in Table 1 but not in experiments)
- Target-norm: **MISSING** (AdamWN is in Table 1 but not in experiments)

The paper's claims about "all dynamic WD variants" cannot extend to spatial or target-norm modulation, which were never tested. The null result for temporal and directional modulation does not imply null results for spatial modulation -- in fact, the proposal's hypothesis H6 was that spatial modulation (per-layer WD) provides genuine benefit through effective LR redistribution.

---

## Major Issue: BEM Implementation Bug

`half_lambda` is implemented as λ=2.5×10⁻⁴ with φ≡1 (standard WD at lower strength) instead of λ=5×10⁻⁴ with φ=0.5 (halved Phi modulation). This means:
- The raw BEM computation gives BEM=0.000 (identical to constant in the Phi sense)
- The paper manually corrects this to BEM=0.500 in Table 4a
- Figure 3 (BEM vs. accuracy scatter) likely uses the corrected value, but the reader cannot verify this

The correction is disclosed, but it means the BEM pipeline has a known defect that requires manual post-processing. If Figure 3's data was derived from the raw BEM=0.000 value for half_lambda, the scatter plot is wrong.

---

## Minor Issues

1. **WD Stability Condition (H2) never tested**: The methodology specifies warmup ablations with K ∈ {1, 10, 50, 200, 1000} to test the WD Stability Theorem. These experiments were not run. The theoretical prediction is stated but has no empirical support.

2. **Soft CWD (H1) never tested**: The methodology describes soft-CWD with β ∈ {10, 50, 100, 500, 1000}. Only hard CWD is in the experiments. The proximal characterization of CWD remains unvalidated.

3. **49 SGD experiments claimed but only 21 are in the workspace**: Paper says "49 total SGD experiments (3 seeds for 5 methods on CIFAR-100 where complete; 3 seeds for all 7 methods on CIFAR-10)". The workspace has 7 methods × 3 seeds = 21 CIFAR-10 SGD runs. CIFAR-100 SGD runs are absent or incomplete. The count of 49 cannot be verified.

4. **Weight norm analysis unsupported for SGD**: The mechanistic argument in Section 6.1 cites "SGD no_wd achieves final weight norm 126.7 vs. 64.5 for SGD constant (96% difference)." The SGD summary.json files confirm SGD constant weight_norm=64.49 but the no_wd weight norm from SGD must also be verified. No SGD weight norm table exists in the paper.
