# Backup Ideas for Pivot (Revised — Iteration 1)

**Last updated**: 2026-04-03 (post-pilot evidence integration)

## When to Pivot

**Revised pivot criteria** (based on pilot evidence):

The pilot shows 2.32% spread (ViT/CIFAR-10) and 9.01% category-level spread — these are large effects that do NOT recommend an early pivot. Pivot only if full-scale (5 seeds, 200 epochs) results reveal:
- ALL 4 blocks with spread < 0.2% (highly unlikely given pilot)
- AND random-per-image ordering is statistically indistinguishable from all fixed orderings in all blocks
- AND Tier 2 category-ordering confirmation pilot also shows < 1% spread at 30 epochs

The primary proposal (cand_a) is strongly supported by current evidence. The backup ideas below are retained for completeness but their activation threshold is now much higher than before pilot evidence.

---

## Backup A: Variance Decomposition of Augmentation Pipeline Design

**Activation condition**: Main study finds null ordering effects (spread < 0.2%) in the full 5-seed experiment. Probability: Low (< 20% given pilot evidence).

**Idea**: Instead of asking "does ordering matter?", ask "how much does ordering matter relative to operation selection, magnitude, and random seed?" Full factorial design with 4 independent variables on ResNet-18/CIFAR-10. ANOVA-based variance decomposition.

**Expected finding** (consistent with pilot evidence): Magnitude >> Category-level ordering >> Within-category permutation ≈ Seed. The 9.01% category-level spread from Tier 2 pilot dwarfs the 0.96% within-category spread from Tier 1, supporting a two-level ordering hierarchy.

**Method**:
1. Fix a 4-operation set and independently vary: (a) permutation (24 orderings), (b) magnitude (3 levels), (c) operation selection (4 subsets of 3 from 4), (d) random seed (10 seeds)
2. Run the full factorial design on ResNet-18/CIFAR-10
3. ANOVA-based variance decomposition. Report eta-squared for each factor.

**Integration with main study**: This can be run as a complementary analysis using data already collected in Tiers 1 and 3, with minimal additional runs. It strengthens the paper regardless of the direction of H1.

**Compute**: ~150 runs on CIFAR-10, ~20 GPU-hours. The variance decomposition analysis adds no new training — it reuses Tier 1 runs.

**Publication path**: Section in the main paper ("How Much Does Each Design Choice Matter?") or NeurIPS workshop.

---

## Backup B: Class-Level Effects of Augmentation Ordering (Zero Additional Training)

**Activation condition**: Main study shows null or marginal aggregate effects but per-class variation is visible. OR: Can always be added as zero-cost secondary analysis.

**Idea**: Augmentation ordering may have null *aggregate* effects but meaningful *per-class* effects that cancel out in the mean accuracy. Per-class accuracy from CIFAR-100 runs is already available from Tier 1 at zero additional training cost.

**Core hypothesis**: Specific orderings disproportionately help or hurt certain semantic categories. For example, geometric-first ordering may help spatial-structure-dependent classes (vehicles, buildings) while hurting texture-rich classes (animals). Inspired by Kirichenko et al. (NeurIPS 2023) — which showed augmentation *strength* has per-class effects — this extends to ordering sensitivity.

**Key differentiation from Kirichenko et al. (NeurIPS 2023)**: Their independent variable is augmentation strength (a scalar). Ours is ordering sequence. The independent variable is fundamentally different; must be foregrounded in related work to avoid confusion.

**Method**:
1. Extract per-class accuracy from all 6 orderings in Tier 1 CIFAR-100 runs (already available at zero cost)
2. Compute "class ordering sensitivity" = max per-class spread across orderings
3. Group CIFAR-100 classes by semantic taxonomy (animals, vehicles, household objects, plants) and test group differences
4. Statistical test: paired per-class t-tests across orderings, with FDR correction

**Contribution**: Reveals per-class ordering sensitivity patterns invisible to aggregate accuracy metrics. Practically important for fairness and long-tail learning.

**Compute**: Zero additional GPU runs. Analysis-only cost. Runs in minutes.

**Publication path**: Section 4 of main paper ("Class-Level Ordering Effects") — zero-cost addition that strengthens the analysis.

---

## Backup C: Learned Ordering Policy via Dataset Fingerprint (Requires Main Study)

**Activation condition**: Main study finds meaningful effects (H1 confirmed, spread > 0.5%) AND full-scale Tier 2 confirms category-level effects. Depends entirely on main study results.

**Idea**: Given the architecture-differential findings (H2) and category-level effects (H5b), develop a "Meta-Ordering" lookup — a simple k-nearest-neighbor regressor that predicts the best ordering from dataset/architecture fingerprint features.

**Approach**:
- Fingerprint features: (a) texture-shape bias score (Geirhos et al. style), (b) class separability in pixel space (inter-class Fisher ratio), (c) architecture type (CNN/ViT flag), (d) patch size if ViT, (e) dataset scale (n_classes, resolution)
- Train meta-regressor on CIFAR-10, CIFAR-100, and Tiny ImageNet runs from main study
- Evaluate on held-out dataset (e.g., Oxford Flowers, SVHN)

**Why this is a backup**: It requires the main study's positive findings as training data. It is a less original contribution than the theoretical framework. The implementation is simple (kNN regressor) but the concept is practically appealing.

**Compute**: Requires main study runs as training data. Meta-regressor fitting: trivial. Extension datasets: ~20 GPU-hours.

**Publication path**: Section 5 "Towards Ordering-Aware Augmentation" in the main paper.

---

## Revised Pivot Decision Tree

```
Full Tier 1 result (5 seeds, 200 epochs)
│
├── H1 confirmed (spread > 0.3% in ≥2 blocks):
│   └── Confirm H2 (architecture differential)
│       ├── H2 confirmed: Strong paper with H1+H2+H5b+H6 + NC/DPI theory framing
│       └── H2 not confirmed: Weaker but still valid with H1+H5b+H6; add Backup B
│
├── H1 marginal (0.1–0.3% in 1–2 blocks):
│   ├── Add Backup B (class-level analysis) as primary finding
│   ├── Amplify H5b/H6 as the larger effects
│   └── Frame as "category-level ordering matters; within-category ordering is marginal"
│
└── H1 null (spread < 0.1% in all blocks):
    ├── Activate Backup A (variance decomposition) as primary framing
    ├── H5b/H6 remain as sub-findings (even if within-category ordering is null,
    │   category-level interleaving may still show large effects)
    └── Frame: "Augmentation category selection and interleaving dominate;
         within-category permutation is absorbed by SGD stochasticity"
```

**Current recommendation**: Do NOT pivot. Execute Full Tier 1 and Tier 2 confirmation pilot immediately. Pilot evidence strongly supports continuing with the primary proposal.
