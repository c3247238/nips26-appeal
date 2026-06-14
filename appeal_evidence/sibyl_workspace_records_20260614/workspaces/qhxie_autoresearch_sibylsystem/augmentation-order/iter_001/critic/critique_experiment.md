# Experiment Critique: Augmentation Ordering Study

## Executive Summary

The experimental design is sound in conception but the execution is entirely pilot-scale. No full-scale experiments have been run. All findings rest on 10-epoch, 100-sample, single-seed runs — a regime where most architecture-dataset blocks have not learned meaningful representations. The critical flaw is that hypothesis verdicts ('confirmed', 'falsified') have been written into JSON artifacts and the paper as if these pilot signals constitute conclusions.

---

## Critical Problems

### 1. Sample Size and Training Scale Are Catastrophically Insufficient

**Evidence from data:**
- Tier 1 ordering experiments: 100 training samples, 10 epochs, 1 seed
- ResNet-18/CIFAR-10: accuracy 10.01–10.97% (random chance = 10% for 10 classes)
- ViT-S/4/CIFAR-100: accuracy 2.64–2.89% (random chance = 1% for 100 classes)
- ViT-S/4/CIFAR-10: accuracy 17.38–19.70% (a 22M parameter model trained on 100 samples)

These are not experiments in any statistically meaningful sense. The ResNet-18/CIFAR-10 block and ViT-S/4/CIFAR-100 block are both operating at or near random chance. Any spread measured in these blocks is noise, not ordering signal.

The pre-registered protocol called for 200 epochs, full datasets (50k training), and 5 seeds. The pilot ran at 0.2% of the intended training data and 5% of the intended epochs. No hypothesis can be evaluated from this.

**Verdict inflation in artifacts:** final_summary.json marks H1 as "confirmed" and H2 as "confirmed" based on single-seed pilot point estimates. This is incorrect. The paper's Results section appropriately labels these as "directional signals", but the JSON artifacts — which are consumed by downstream automation — claim confirmed hypotheses.

### 2. No Valid Statistical Tests Are Possible from n=1

The pre-registered statistical plan specifies:
- Paired t-tests (requires n ≥ 2)
- Two-way ANOVA (requires n ≥ 2 per cell)
- Bonferroni correction (multiple comparison correction)
- Cohen's d (requires within-group variance)

With n=1 seed, none of these are computable. The tier1_analysis.json reports Cohen's d values of 2.27–2.71 for all four blocks despite t_stat=null and p_value=null. These d values appear to be computed from between-ordering variance assuming zero within-ordering variance (std_val_acc=0.0 for all single-seed runs). Effect sizes computed this way are undefined and misleading — they create a false impression of statistically validated large effects.

### 3. Tier 3 Best/Worst Ordering Labels Are Internally Inconsistent

Tier 3 tests "best" ordering (Flip→Crop→CJ, order_2) vs. "worst" ordering (CJ→Flip→Crop, order_5) from Tier 1. The selection is based on the CIFAR-100/ResNet-18 block:
- CIFAR-100/RN18 best: order_2 (0.4663), worst: order_5 (0.4575) ✓

But Tier 3 was run on CIFAR-100 with full dataset (not 100 samples), 10 epochs — a different training regime from Tier 1. At M5, the "worst" ordering (CJ→Flip→Crop) achieves 0.5123, outperforming the "best" ordering (Flip→Crop→CJ) at 0.5088. The tier3 notes acknowledge this as "stochastic variation with only 1 seed", but this is precisely the problem: a single seed cannot distinguish a genuine ordering effect from seed-level noise.

Additionally, in the CIFAR-10/ResNet-18 block, order_5 (CJ→Flip→Crop) IS the best ordering (10.97%), not the worst. The best/worst labels are block-specific and were not locked before Tier 3 executed.

### 4. Tier 3 Uses Inconsistent Training Conditions vs. Tier 1

Tier 3 uses the full CIFAR-100 dataset (50k training samples), while Tier 1 used 100-sample subsets for the CIFAR-100/ResNet-18 block. This means:
- Tier 1 CIFAR-100 accuracies: 45.75–46.63% (10 epochs, 100 samples)
- Tier 3 M9 accuracies: 45.75–46.63% (10 epochs, full 50k dataset — same values!)

Wait: the values are identical. Looking at the data: Tier 3/M9/order_2 = 0.4663, Tier 3/M9/order_5 = 0.4575. These are the same values as Tier 1 CIFAR-100 runs. The Tier 3 M9 condition IS the Tier 1 condition (same seed 42, same epochs 10) — which means Tier 3 and Tier 1 share data rather than being independent experiments. This needs verification.

### 5. ViT-S/4/CIFAR-100 Block Is Degenerate But Counted in H1

The paper correctly identifies the ResNet-18/CIFAR-10 block as degenerate (near random chance floor) and excludes it from the H1 assessment. However, the ViT-S/4/CIFAR-100 block (2.64–2.89% accuracy, random chance = 1%) is counted in the 3/4 blocks with spread > 0.5%. This block's 0.25% spread is below threshold and doesn't affect the count, but including a near-degenerate block while excluding another inflates confidence in the experimental validity.

The ViT-S/4/CIFAR-100 block should also be flagged as operating far below meaningful learning. A 22M parameter vision transformer cannot learn CIFAR-100 features from 100 samples in 10 epochs; the 2.64–2.89% range represents essentially random feature extraction plus minor class bias, not a genuine ordering effect.

---

## Major Problems

### 6. Tier 2 Results Are Single-Condition and Overclaimed

The 9.01 percentage point spread between interleaved P→G (29.39%) and all-geometric-first (20.38%) in Tier 2 is based on:
- Architecture: ResNet-18 only
- Dataset: CIFAR-10 only  
- Sample size: 5k training samples
- Training: 10 epochs
- Seeds: 1

This is a single point on the experimental grid. The paper uses this as motivation for a practical recommendation about interleaved orderings. The recommendation is presented in the Discussion as if it were validated across the full experimental space.

Furthermore, the 20–30% accuracy range suggests these models are still in an early learning phase. The Tier 2 results at epoch 10 may not reflect the asymptotic ordering effect at convergence.

### 7. The Tier 0 Pilot Summary Is Inconsistent with Later Tier 1 Results

The pilot_summary.json (Tier 0) reports:
- Best ordering: CJ→Crop→Flip (0.2804, spread 2.68%)
- Worst ordering: Flip→CJ→Crop (0.2536)
- "CJ-first orderings dominate"

The Tier 1 results report:
- Best ordering across blocks: Flip→Crop→CJ
- The CJ→Flip→Crop ordering is the WORST on CIFAR-100/ResNet-18

These are opposite conclusions about the role of CJ-first orderings. The Tier 0 used 5k samples (vs 100 for Tier 1), suggesting different data regimes produce different ordering preferences. This instability should be flagged as a major uncertainty about whether any ordering finding will replicate at 200 epochs / full data.

---

## Minor Problems

### 8. M14 Exact Tie Is Unexplained

Both orderings achieve exactly 0.245 at M14. This could be:
(a) The network collapsed to a learned class balance distribution under extreme augmentation
(b) Both runs happened to converge to the same epoch-10 checkpoint (check if the loss curves are identical)
(c) The random seed is inducing the same path under heavy stochasticity

The exact match to 4 decimal places (0.2450 = 0.2450) is suspicious. This should be investigated.

### 9. Stochastic Augmentation and n=1 Interaction

ColorJitter, RandomCrop, and RandomHorizontalFlip all introduce stochasticity at each sample. With n=1 seed and 100 training samples, the ordering effect is confounded with the stochastic augmentation realization. Two runs with the same seed but different orderings will see different augmented images because the augmentation randomness interacts with ordering. The paired seed design is correct in principle but requires multiple seeds to separate ordering signal from augmentation-realization noise.

---

## What Needs to Happen Before the Paper Is Publishable

1. Run the full-scale Tier 1 protocol (200 epochs, 50k training samples, 5 seeds) — this is non-negotiable
2. Run full-scale Tier 2 (both architectures, both datasets, 3+ seeds)
3. Compute NC_2 with 10k+ samples (Tier 4a)
4. Compute MI with full-scale encoder checkpoints (Tier 4b)
5. Report Tier 3 with matched conditions and seed count

Without steps 1-3, no hypothesis can be declared confirmed or falsified, and no practical recommendation should be made.
