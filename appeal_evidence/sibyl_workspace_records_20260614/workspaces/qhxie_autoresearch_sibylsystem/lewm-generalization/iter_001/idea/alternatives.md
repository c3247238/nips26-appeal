# Backup Ideas and Pivot Options

## Strategic Context

The front-runner proposal ("Geometric Factorization Meets Physical Reality: A Systematic Compositional Generalization Benchmark for JEPA World Models") integrates the strongest elements from all six perspectives. These backup ideas represent pivot options if the front-runner encounters specific blocking problems, or serve as natural extensions/follow-up papers.

---

## Backup A: SIGReg vs. Group-Structured Latent Space Comparison

**When to pivot here**: If the primary experiment finds that SIGReg provides no advantage (or disadvantage) over VICReg for compositional factorization, the most interesting follow-up is testing whether *structured* latent priors (designed explicitly for factorization) outperform isotropic priors.

**Core idea**: Implement a group-structured latent space variant of LeWM (following Delliaux et al., 2025 — "Learning Abstract World Models with a Group-Structured Latent Space") and compare it against the standard SIGReg model on the ComPhys-LeWM benchmark. The group structure explicitly decomposes the latent into Z_sym × Z_unsym, providing a geometric prior aligned with factored physical parameters.

**Hypotheses**:
- Group-structured LeWM shows higher IIS and principal angle orthogonality than SIGReg-LeWM
- Group-structured LeWM shows better compositional generalization on held-out factor combinations
- The advantage is larger for combinations involving physically independent factors and smaller for strongly-coupled factors (where no geometric prior helps)

**Effort estimate**: ~30 GPU-hours beyond the primary experiment. Implementation requires modifying the le-wm training loop to add the group-structured prior.

**Novelty**: 8/10 — First comparison of isotropic Gaussian vs. group-structured priors on a compositional physical generalization benchmark.

**Papers to build on**:
- Delliaux et al. (2025, arXiv:2506.01529): Group-Structured Latent Space
- Schwarcz (2026, arXiv:2603.27134): RCC architectures with Semantic Interaction Information

---

## Backup B: Curriculum-Ordered Factor Introduction

**When to pivot here**: If the primary experiment shows that compositional generalization fails *even at moderate training coverage*, but LoRA adaptation succeeds with small amounts of target data. This would suggest the model can learn the relevant structure when shown target examples, but fails to extract it from random multi-factor training — consistent with an ordering/curriculum effect.

**Core idea**: Inspired by Baillargeon (2004) and Knopp et al. (2025), test whether training with sequentially introduced factors (one factor at a time, then combinations) produces better factorized representations than training on all factor combinations simultaneously. Three training orderings: (1) single-factor variations first, then pairwise, then triple, (2) reverse, (3) simultaneous (baseline).

**Hypotheses**:
- Sequential single→pairwise→triple factor introduction produces higher IIS and better holdout accuracy than simultaneous training
- The factorization quality at the end of stage k (single-factor stage) predicts final compositional generalization performance
- The benefit of curriculum ordering is larger for factors with higher physical coupling strength (where order helps establish stable factor representations before coupling is introduced)

**Effort estimate**: 6 training orderings × 3 seeds × ~3h each = ~54 GPU-hours. Feasible as a standalone contribution or Section 5 of the primary paper.

**Novelty**: 7/10 — Curriculum learning is established, but its application to physical factor factorization in world models is new. The developmental psychology motivation provides a clear theoretical frame.

**Risk**: Effect size may be small on a 15M model. The null result (ordering does not matter) would still be informative and publishable.

---

## Backup C: Physics-Informed Compositional Evaluation Framework (Standalone)

**When to pivot here**: If the primary experiment finds that LeWM compositionally generalizes well (falsifying H1), the most interesting remaining question shifts from "does it work?" to "why does it work, and what are the limits?" In this scenario, the physical regime analysis becomes the main contribution.

**Core idea**: Develop the regime-aware evaluation framework as a standalone contribution:
1. Build a taxonomy of "composable" vs. "non-composable" physical dynamics in DMControl environments
2. Show that the taxonomy predicts model-agnostic failure patterns (across LeWM, DINO-WM, cRSSM)
3. Provide a calibrated evaluation protocol that reports compositional generalization separately for weakly-coupled (achievable) vs. strongly-coupled (physically limited) combinations
4. Demonstrate the parallelogram test and IIS as standard metrics for the field

**Framing**: "Towards Fair Evaluation of World Model Compositional Generalization: Disentangling Model Limitations from Physical Irreducibility"

**Effort estimate**: ~40 GPU-hours (shared with primary experiment) + ~10h CPU for physical analysis.

**Novelty**: 9/10 — The idea that some evaluation failures are physically irreducible (not model failures) has not been articulated in the world model community. This is a conceptual contribution with practical implications for fair benchmarking.

---

## Backup D: Probing-vs-Transfer Dissociation (The Contrarian's Main Contribution)

**When to pivot here**: If the primary experiment finds a dissociation between in-distribution probing accuracy and compositional generalization — i.e., models with high in-distribution probing accuracy show poor holdout performance, or the geometric factorization metrics are better predictors of transfer than probing accuracy itself.

**Core idea**: Shift the paper's framing to "The Probing Illusion": demonstrate systematically that in-distribution probing accuracy is a poor proxy for compositional generalization in JEPA world models, and introduce geometric + interventional metrics as the proper diagnostic. This becomes a *methodology* paper rather than a benchmark paper.

**Hypotheses**:
- In-distribution probing R² and holdout probing R² are weakly correlated (r < 0.3) across different model variants, training runs, and factor combinations
- Principal angle orthogonality and IIS are better predictors of holdout accuracy (r > 0.7) than in-distribution probing accuracy
- The probing-accuracy/transfer-accuracy dissociation is larger for multi-factor holdouts than single-factor holdouts (because compositional structure matters more when more factors must be composed)

**Framing**: "When Probing Passes but Transfer Fails: Geometric Diagnostics for Compositional Generalization in JEPA World Models"

**Novelty**: 8/10 — The disconnect between probing and generalization is implicit in the contrarian critique but has not been quantified for JEPA world models specifically.

---

## Decision Matrix for Pivoting

| Pilot Outcome | Recommended Pivot |
|---|---|
| H1 falsified (LeWM generalizes perfectly) | Backup C (why does it work, physical limits) → Backup A (group structure extensions) |
| H1 confirmed, H2 falsified (SIGReg neutral/harmful) | Backup A (group-structured alternatives) as main paper |
| H1 confirmed, H3 falsified (no encoder/predictor asymmetry) | Backup B (curriculum ordering) as additional experiment |
| Large probing-generalization dissociation observed | Backup D (methodology paper on probing illusion) |
| All hypotheses confirmed | Primary paper with Backups B and D as supplementary experiments |
| LoRA completely fails (even rank 64) | Focus on full fine-tuning as adaptation diagnostic; LoRA as supporting negative result |
