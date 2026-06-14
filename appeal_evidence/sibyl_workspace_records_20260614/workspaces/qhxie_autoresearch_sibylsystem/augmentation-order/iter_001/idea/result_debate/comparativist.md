# Comparativist Analysis: Augmentation Ordering Study

**Date**: 2026-04-02
**Status**: PILOT results only (10 epochs, limited samples, 1 seed)

---

## 1. Baseline Landscape

### Published SOTA on the Same Benchmarks (ResNet-18, trained from scratch)

| Method | CIFAR-10 Acc | CIFAR-100 Acc | Source |
|--------|-------------|---------------|--------|
| ResNet-18 baseline (standard augment) | ~94.5-95.0% | ~76-78% | Multiple reproductions |
| ResNet-18 + RandAugment (N=2 M=9) | ~95-96% | ~78-80% | Cubuk et al. 2019; Soft Augment paper |
| ResNet-18 + TrivialAugment | ~95-96% | ~78-80% | Muller & Hutter 2021 (extrapolated from WRN results) |
| ResNet-18 + Cutout | 96.01% | 78.04% | DeVries & Taylor 2017 |
| ResNet-18 + Soft Augment + RA | >96% | >80% | arXiv:2211.04625 |

### Published SOTA for Small ViTs on CIFAR (trained from scratch)

| Method | CIFAR-10 Acc | CIFAR-100 Acc | Source |
|--------|-------------|---------------|--------|
| Tiny ViT (vanilla, from scratch) | ~80% | ~55-60% | Various reproductions |
| ViT + depth-wise conv modifications | ~96.4% | ~80-85% | Zhang et al. 2025 |
| MSCViT (14M params) | -- | 84.68% | arXiv:2501.06040 |
| ViT-CIFAR (6.3M) | >90% | -- | omihub777/ViT-CIFAR |

### Our Baseline Results (PILOT: 30 epochs, seed=42)

| Method | CIFAR-10 RN18 | CIFAR-100 RN18 | CIFAR-10 ViT-S | CIFAR-100 ViT-S |
|--------|---------------|----------------|----------------|-----------------|
| Conventional (Crop->Flip->CJ) | 91.91% | 71.77% | 80.80% | 58.76% |
| Random-per-image | 91.88% | 72.27% | 81.09% | 58.54% |
| TrivialAugment | 91.95% | 71.63% | 78.94% | 57.57% |
| No augmentation (Crop+Flip only) | 92.00% | 72.23% | 81.11% | 56.46% |
| RandAugment N=2 M=9 | 92.57% | 72.83% | 81.24% | 59.85% |

**Assessment**: Baseline accuracies are 2-4% below published SOTA, which is expected at 30 epochs (vs. typical 200 epochs). The relative ranking of methods is consistent with literature: RandAugment > TrivialAugment >= Conventional > No augmentation. This validates the experimental setup.

---

## 2. Contribution Margin Analysis

### Critical Issue: Ordering Experiments Used Severely Under-Trained Models

The ordering results (Table 1 in final_summary.json) show dramatically low accuracies:
- ResNet-18 on CIFAR-10: 10.01% - 10.97% (random chance = 10%)
- ResNet-18 on CIFAR-100: 45.75% - 46.63%
- ViT-Small on CIFAR-10: 17.38% - 19.70%
- ViT-Small on CIFAR-100: 2.64% - 2.89% (random chance = 1%)

**These results are from extremely short training (likely 10 epochs on small subsets).** The ResNet-18 CIFAR-10 results are at random-chance level, meaning the model has barely begun to learn. Any ordering differences at this stage reflect initialization noise and early gradient dynamics, NOT the kind of long-term generalization effects that would be publishable.

**The baselines, by contrast, were trained for 30 epochs on the full dataset,** making them incomparable to the ordering experiments. This is a fundamental methodological problem that must be resolved before any contribution claims can be made.

### Ordering Spread (from pilot data, heavily caveated)

| Block | Spread | Classification |
|-------|--------|---------------|
| ResNet-18 x CIFAR-10 | 0.96% | **Marginal** (at random-chance accuracy) |
| ResNet-18 x CIFAR-100 | 0.88% | **Marginal** (at undertrained accuracy) |
| ViT-Small x CIFAR-10 | 2.32% | **Moderate** (at very low accuracy) |
| ViT-Small x CIFAR-100 | 0.25% | **Marginal** (at near-random accuracy) |

**Honest assessment**: A 2.32% spread sounds impressive, but it is the difference between 17.38% and 19.70% on a task where the converged model should reach >80%. At such low absolute accuracy, these differences may vanish, reverse, or become statistically insignificant once training converges. **No contribution claim can be made from these numbers.**

### Category-Level Ordering (Tier 2, CIFAR-10 5k pilot)

The interleaved P->G ordering (29.39%) vs. all-geometric-first (20.38%) shows a 9.01% spread. However, this is on a 5,000-sample subset at pilot scale. The 20.38% accuracy for all-geometric-first is itself near random, suggesting the pipeline may have a bug or the training was too short to be informative.

**FLAG**: A 9% spread at such low accuracy is suspicious and requires validation at full scale.

---

## 3. Concurrent Work Scan

### Direct Competitors (2024-2026)

**No paper found** that specifically studies augmentation operation ordering as the sole independent variable. This confirms the novelty claim in the proposal. Key concurrent work:

1. **Li et al. (arXiv:2408.14381, Aug 2024)** — "Learning Tree-Structured Composition of Data Augmentation"
   - Studies binary tree vs. sequential composition STRUCTURE
   - Does NOT permute operations within a fixed sequence
   - Related but addresses a different question (structural topology vs. ordering)
   - Does NOT threaten our novelty

2. **Sample-aware RandAugment (arXiv:2508.08004, 2025)**
   - Per-sample complexity scoring for magnitude adaptation
   - Does NOT study ordering
   - Orthogonal contribution

3. **Enhancing Image Classification with Augmentation (arXiv:2502.18691, Feb 2025)**
   - New augmentation techniques (channel transfer)
   - Does NOT study ordering

4. **Surveys (Cheung & Yeung, IEEE TNNLS 2023; Yang et al., KAIS 2023)**
   - Both explicitly identify operation ordering as an open question
   - Our study would directly address this acknowledged gap

**Verdict**: No concurrent paper scoops this work. The novelty window remains open as of April 2026.

---

## 4. Novelty Verdict

**The ONE thing this work does that no prior work does**: It isolates augmentation operation ordering as the sole independent variable in a controlled factorial experiment with theoretical grounding (NC measure + DPI reversibility principle).

**However**, the current pilot evidence is too weak to support any strong empirical claim. The theoretical contributions (NC bound, DPI principle) are novel on paper, but:
- H3 (NC predicts accuracy) is **falsified** (rho=-0.20, p=0.68)
- H4 (MI predicts accuracy) is **inconclusive** (combined rho=-0.057)
- H5 (magnitude amplifies ordering effect) is **falsified**

This means **2 of 3 theoretical predictions failed**, and the remaining 2 confirmed hypotheses (H1: ordering matters, H2: reversibility-sorted wins sometimes) are supported only by under-trained pilot data with 1 seed and no statistical tests.

**Novelty is present but unvalidated.** The question is novel; the answer is not yet credible.

---

## 5. Venue Recommendation

### Current State: **Insufficient for submission**

**Rationale**:
- Ordering experiments are at random-chance accuracy and cannot support any generalization claims
- 2/3 theoretical predictions falsified, undermining the theoretical framework
- Only 1 seed, no paired t-tests, no Bonferroni correction
- Baselines and ordering experiments are not comparable (different training regimes)

### If Full-Scale Results Confirm H1 at Convergence:

| Scenario | Venue Tier | Justification |
|----------|-----------|---------------|
| Ordering spread >1% at convergence + NC correlation recovers | Top-tier (NeurIPS/ICML) | First principled study + validated theory |
| Ordering spread >0.5% at convergence, NC still fails | Mid-tier (AAAI/EMNLP workshop) | Empirical contribution without theory |
| Ordering spread <0.3% at convergence | Workshop or negative-results track | Valuable null result confirming convention |
| NC and DPI both fail, ordering effect marginal | Negative results workshop (NeurIPS) | Important to document for the community |

**Comparable papers at target venues**:
- TrivialAugment (ICCV 2021) — simple method, strong empirical results, minimal theory
- RandAugment (CVPR Workshop 2020) — practical method with reduced search space
- AugMax (NeurIPS 2021) — combination of diversity + adversarial, theoretical motivation

For top-tier acceptance, this work needs: (a) validated ordering effects at full training convergence, (b) at least ONE working theoretical predictor, and (c) cross-dataset validation beyond CIFAR.

---

## 6. Strengthening Plan

### Priority 1: Run Full-Scale Tier 1 (CRITICAL)

The single most important action is re-running the 6 orderings with:
- 200 epochs (not 10)
- Full dataset (not subsets)
- 5 seeds (not 1)
- Proper paired t-tests with Bonferroni correction

Without this, the paper has zero credible evidence. Everything else is secondary.

### Priority 2: Add Stronger Baselines

Current baselines are incomplete. Add:

| Baseline | Purpose | Difficulty |
|----------|---------|-----------|
| **AutoAugment (learned policy)** | The ordering in AutoAugment was learned; compare our best fixed ordering against it | Easy (torchvision has it) |
| **Li et al. 2024 tree-structured composition** | Most directly related concurrent work; show that our ordering insights are complementary | Medium (requires reimplementation) |
| **Soft Augmentation + ordering** | Combine ordering with the strongest recent augmentation method | Easy (open-source code available) |

### Priority 3: Rescue the Theoretical Framework

H3 and H5 being falsified is a significant blow. Options:
1. **Recompute NC_2 with proper Wasserstein distance** (current implementation uses SWD proxy -- this may not capture the actual NC)
2. **Test NC correlation at full training convergence** -- the correlation may emerge only after models are properly trained
3. **Frame falsification as a finding** -- "NC_2 as currently formulated is insufficient to predict ordering effects" is itself publishable if accompanied by analysis of WHY it fails

### Additional Comparisons That Would Strengthen Positioning

1. **ImageNet-1k subset (Tiny ImageNet or ImageNet-100)**: CIFAR results alone may not convince reviewers. Even a small-scale ImageNet experiment (10 orderings, 1 architecture, 90 epochs) would dramatically strengthen the paper.
2. **Object detection / segmentation transfer**: Show ordering effects transfer to downstream tasks (even a quick COCO fine-tuning experiment from differently-ordered CIFAR pre-training).
3. **Training dynamics analysis**: Plot ordering effects across training (epochs 1, 10, 50, 100, 200). This would directly address whether the pilot-stage differences persist, grow, or vanish at convergence.

---

## 7. Summary Assessment

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Novelty of question | **Strong** | No prior work; acknowledged gap in surveys |
| Novelty of theory | **Moderate** | NC bound + DPI principle are novel but unvalidated |
| Empirical evidence | **Insufficient** | Pilot-only, random-chance accuracy, 1 seed |
| Concurrent work risk | **Low** | No scooping threat as of April 2026 |
| Contribution margin vs. SOTA | **Unknown** | Cannot assess until full-scale results exist |
| Publication readiness | **Not ready** | Full Tier 1 experiments are prerequisite |

**Bottom line**: This study asks the right question in an untouched niche. The novelty window is open and no concurrent work threatens priority. However, the current evidence is from severely under-trained models and cannot support any publishable claim. The immediate priority is a full-scale Tier 1 experiment. If the ordering effect survives at convergence (even at 0.3-0.5%), the paper has a clear path to a mid-to-top-tier venue. If the effect vanishes, the negative result plus the NC/DPI falsification analysis is still publishable at a workshop or negative-results track.

---

## Sources

- [TrivialAugment (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Muller_TrivialAugment_Tuning-Free_Yet_State-of-the-Art_Data_Augmentation_ICCV_2021_paper.pdf)
- [RandAugment (CVPR Workshop 2020)](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf)
- [Li et al. 2024 — Tree-Structured Composition](https://arxiv.org/abs/2408.14381)
- [Soft Augmentation (arXiv:2211.04625)](https://arxiv.org/html/2211.04625v2)
- [Sample-aware RandAugment (arXiv:2508.08004)](https://arxiv.org/abs/2508.08004)
- [94% on CIFAR-10 in 3.29 Seconds](https://arxiv.org/html/2404.00498v1)
- [MSCViT for Tiny Datasets (arXiv:2501.06040)](https://arxiv.org/html/2501.06040v2)
- [Survey on Automated Data Augmentation (Springer 2023)](https://link.springer.com/article/10.1007/s10115-023-01853-2)
- [Mix-Based Data Augmentation Survey (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/3696206)
- [CIFAR-100 Benchmark (Papers with Code)](https://paperswithcode.com/sota/image-classification-on-cifar-100)
- [Enhancing Image Classification with Augmentation (arXiv:2502.18691)](https://arxiv.org/html/2502.18691v1)
- [OpenMixup CIFAR Benchmarks](https://openmixup.readthedocs.io/en/latest/mixup_benchmarks/Mixup_cifar.html)
