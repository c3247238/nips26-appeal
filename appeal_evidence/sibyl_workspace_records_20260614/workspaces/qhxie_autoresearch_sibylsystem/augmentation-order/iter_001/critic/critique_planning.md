# Planning Critique: Augmentation Ordering Study

## Executive Summary

The experimental plan is well-conceived: the four-tier hierarchical design, paired seed protocol, pre-registered falsification criteria, and multi-metric evaluation are all appropriate. The resource estimates (51 GPU-hours, 26h wall clock) are plausible. The primary planning failure is that the pipeline has advanced beyond the pilot stage and produced paper-formatted results while treating pilot-scale signals as study-level conclusions. Execution discipline broke down between Tier 0 and Tier 1 — the Tier 1 runs were conducted at pilot scale (100 samples, 10 epochs) rather than full scale, but the full-scale analysis code was applied to produce hypothesis verdicts.

---

## What the Plan Gets Right

### 1. Paired Seed Design

The paired seed design (same seed across all orderings) is the correct choice for detecting small effects. Unpaired designs with 5 seeds per ordering would have substantially higher variance and require larger sample sizes to detect the same effect size. This design choice is methodologically sound.

### 2. Pre-Registered Falsification Criteria

The explicit, quantitative falsification criteria are rare in the ML literature and strengthen the paper's credibility. The specific thresholds (H1: spread > 0.2%, H3: rho > 0.5) are reasonable, though H1's threshold was informally changed from 0.2% in the proposal to 0.5% in the paper without documentation.

### 3. Four-Tier Hierarchy

The hierarchy (pilot → full factorial → category ordering → magnitude interaction → theoretical validation) is logical. The idea of running NC measurement (Tier 4) before the accuracy-heavy Tiers 1-3 (to validate the theoretical framework first) would have been even better, but the current sequence is workable.

### 4. Resource Budget Is Realistic

The pre-registered 51 GPU-hours for the full study is achievable on the stated hardware (RTX PRO 6000). The cost scaling is appropriate: Tier 1 (most expensive at 20h) → Tier 2 (18h) → Tier 3 (12h) → Tier 4 (<2h).

---

## Critical Problems

### 1. The Pipeline Has Not Followed Its Own Tier Structure

The plan specifies:
- Tier 0 (pilot): ResNet-18 only, 5k subset, 10 epochs, 2 seeds → GO/NO-GO decision
- Tier 1 (full factorial): all 6 orderings × 2 archs × 2 datasets × 5 seeds × 200 epochs

What actually happened:
- Tier 0: ResNet-18, 5k subset, 10 epochs (appears executed as planned)
- Tier 1: all 6 orderings × 2 archs × 2 datasets × **1 seed** × **10 epochs** × **100-sample subset**
- Tier 2: ResNet-18 only, CIFAR-10 only, 1 seed, 10 epochs, 5k samples
- Tier 3: 2 orderings × 3 magnitudes × ResNet-18 × CIFAR-100 (full dataset), 1 seed, 10 epochs
- Tier 4a: NC_2 computed with 100 samples (spec called for 10k)
- Tier 4b: InfoNCE MI with 100-sample encoders (spec called for Tier 1 checkpoints)

The execution is approximately 0.5–2% of the planned scale at every tier except Tier 0. This is not a pilot extension of Tier 0; it is a new, smaller-scale pilot that was mislabeled as Tier 1.

### 2. The GO Decision Was Misapplied

The Tier 0 pilot summary gives "overall_recommendation: GO" with confidence 0.85, based on 2.68% spread at 10 epochs on 5k samples. The GO decision should have triggered the **full-scale** Tier 1 (200 epochs, 50k samples, 5 seeds). Instead, the system ran a 100-sample, 10-epoch, 1-seed version of Tier 1.

The GO decision tree in alternatives.md says: "Spread > 0.5%: proceed with full proposal as planned." The 2.68% pilot spread triggered a GO — but the execution did not follow the plan. This is a pipeline discipline failure.

### 3. Hypothesis Thresholds Were Changed Without Documentation

The original proposal (proposal.md) specifies H1 threshold: "accuracy spread > 0.3% on CIFAR-10 with ResNet-18 (200 epochs, 5 seeds)." The methodology.md revises this to "spread > 0.2% with 95% CI." The paper (writing/paper.md) uses "> 0.5% in at least 3 of 4 architecture-dataset blocks." Three different thresholds for H1, with no documented rationale for changes.

Similarly, H3 threshold changed from "rho > 0.5" (proposal.md) to "rho > 0.6" (tier4_correlation.json) without documentation.

### 4. Tier 3 Best/Worst Ordering Not Pre-Committed Before Execution

The plan says "Best and worst orderings from Tier 1 (determined post-Tier 1)" for Tier 3. The selection was made from the CIFAR-100/ResNet-18 block, but this choice was not documented before Tier 3 ran. The fact that CJ→Flip→Crop is "worst" on CIFAR-100/ResNet-18 but "best" on CIFAR-10/ResNet-18 means the Tier 3 result is specifically testing behavior under the CIFAR-100/ResNet-18 criteria — not a universal best/worst.

---

## Major Problems

### 5. No Mechanism to Ensure Full-Scale Tier 1 Actually Runs

The system declared pilot-level Tier 1 results as "confirmed" hypotheses without any checkpoint or gate to ensure the full-scale Tier 1 (200 epochs, 5 seeds) actually executes. The research pipeline should have a hard gate: "pilot results are labeled pilot; hypothesis verdicts require full-scale." This gate does not appear to exist.

### 6. Tier 2 Architecture Coverage Is Insufficient

The plan specifies Tier 2 as "2 architectures × 2 datasets × 5 seeds = 100 runs." Only ResNet-18/CIFAR-10/1 seed ran. This is 2% of the planned Tier 2 coverage. The practical recommendation for interleaved orderings is based entirely on this 2% coverage.

### 7. Baseline Training Conditions Are Not Matched

The baselines were trained on full datasets for 30 epochs, while ordering experiments used 100 samples for 10 epochs. The plan does not specify that baselines should be run at the same scale as ordering experiments. This should be fixed: run baselines under identical conditions for valid within-scale comparisons, even if the absolute numbers are low.

---

## Recommendations

1. **Immediate priority**: Execute the pre-registered full-scale Tier 1. This is the entire point of the study. All the theoretical development, pilot work, and paper writing is wasted without this step.

2. **Gate mechanism**: Add an explicit mechanism in the pipeline to prevent hypothesis verdicts from being computed on pilot-scale data. The JSON artifacts should refuse to output "confirmed" or "falsified" unless the n_seeds field in the analysis equals the pre-registered value (5).

3. **Document all threshold changes**: Any deviation from the pre-registered thresholds should be logged in a "Protocol Deviations" section of the methods. The three different H1 thresholds need consolidation.

4. **Lock Tier 3 selection criteria**: Before running Tier 3, explicitly commit to which block determines "best" and "worst" orderings and log this decision.

5. **Reschedule**: Given the current state (all pilot), the paper should be treated as a technical report documenting the experimental design and pilot findings. Submission to a venue should be scheduled after full-scale results are available.
