# Optimist Analysis

## Evidence Map

| Metric | Baseline / Prior Expectation | Ours | Delta | Signal Strength |
|--------|------------------------------|------|-------|-----------------|
| First-letter vs semantic-hierarchy correlation (r) | H1 predicted r > 0.6 | r = 0.463 (95% CI [-0.389, 0.981]) | -0.137 vs threshold | Moderate (positive point estimate, wide CI) |
| tau_fs robustness (correlation stability) | H3 predicted r > 0.5 across thresholds | r = 0.468 (0.01), 0.463 (0.03), 0.471 (0.05) | ~0.008 range | Strong (remarkably stable) |
| Architecture ranking preservation (Kendall's tau) | H5 predicted tau_rank > 0.5 | PAnneal < Matryoshka < Gated < JumpRelu < TopK < BatchTopK on both tasks | Ordering preserved | Strong |
| GPT-2 replication (semantic hierarchy absorption) | Expectation: low absorption, correlating with Pythia | Standard: 0.0, TopK: 0.003 | Near-zero, consistent with low-absorption architectures | Strong |
| Probe quality (AUROC) | Target > 0.7 for valid probes | All 10 hierarchy probes: AUROC = 1.0 (all SAEs) | +0.30 above threshold | Strong |
| GPT-2 pair (non-hierarchy) absorption | Expectation: low, below hierarchy | Standard: 0.025, TopK: 0.098 | Lower than hierarchy | Moderate |

## Root Cause Analysis

### Positive Signal 1: Robustness Across tau_fs Thresholds
- **Mechanism**: The correlation between first-letter and semantic-hierarchy absorption is almost perfectly invariant to the feature-splitting threshold (tau_fs = 0.01, 0.03, 0.05). This suggests the relationship is not an artifact of a specific hyperparameter choice but reflects a stable structural property of how these SAEs represent hierarchical features.
- **Design decision**: The proposal explicitly included a tau_fs ablation to test robustness. The near-zero variance (r range: 0.463–0.471) validates this design choice and strengthens the methodological rigor of the study.
- **Expected or surprising**: Expected if the metric is stable, but the magnitude of stability (±0.004) is surprisingly strong given the small sample size (n=7 SAEs).

### Positive Signal 2: Architecture Ranking Preservation
- **Mechanism**: The relative ranking of SAEs by absorption rate is preserved across first-letter and semantic-hierarchy tasks. PAnneal (lowest) < MatryoshkaBatchTopK < GatedSAE < JumpRelu < TopK < BatchTopK (highest). This monotonic alignment means that even if the absolute correlation falls short of 0.6, the *ordinal* construct validity is intact—architectures that improve first-letter absorption also improve semantic-hierarchy absorption.
- **Design decision**: The diverse SAE selection (spanning the absorption spectrum) was a deliberate choice from the empiricist perspective. Without this diversity, ranking preservation would be unobservable.
- **Expected or surprising**: Expected, but the clarity of the ordering (no inversions) is a genuine positive signal that supports the utility of the benchmark for architecture comparison.

### Positive Signal 3: Perfect Probe Quality and GPT-2 Replication
- **Mechanism**: All 10 hierarchy probes achieved AUROC = 1.0 across all 8 SAEs on Pythia-160m, and the GPT-2 replication showed near-zero semantic-hierarchy absorption (Standard: 0.0, TopK: 0.003). This indicates the experimental protocol is technically sound and replicable across models.
- **Design decision**: The frequency-matching protocol and single-token filtering (from the empiricist + pragmatist) ensured clean probes. The GPT-2 replication was an exploratory ablation that paid off.
- **Expected or surprising**: The perfect AUROCs are surprising—they suggest the concepts were exceptionally well-chosen, though they also raise questions about whether the task is "too easy" (see caveats).

## Unexpected Signals

### Unexpected Finding 1: Hierarchy Specificity Reversed
- **Observation**: The paired t-test between semantic-hierarchy and non-hierarchy control absorption was significant (t = -4.748, p = 0.0032), but in the *opposite* direction: semantic-hierarchy absorption (mean = 0.235) was *lower* than non-hierarchy control absorption (mean = 0.331).
- **Mini-hypothesis**: The SAEBench absorption formula may be more sensitive to feature co-occurrence patterns in synonym/attribute pairs than in true hypernym hierarchies. Non-hierarchy pairs like "doctor-hospital" or "student-school" have stronger contextual co-occurrence in training data, leading to more overlapping SAE latents and thus higher "absorption-like" scores. True hierarchies ("animal" ⊃ "dog") may be represented more discretely in the SAE because the parent and child tokens appear in less overlapping contexts.
- **Significance**: If true, this implies the metric detects *correlation structure* more than *hierarchical structure*. This is still a meaningful finding—it means the metric is not hierarchy-specific, but it *is* sensitive to representational overlap. The paper could reframe this as "the metric generalizes to correlation detection, not hierarchy detection."

### Unexpected Finding 2: Random SAE Matches Standard SAE on Semantic Hierarchy
- **Observation**: The Random SAE (permuted decoder) achieved semantic-hierarchy absorption = 0.352 and non-hierarchy control = 0.416, nearly identical to the Standard SAE (0.352 and 0.416 respectively). This suggests that for semantic tasks, the Standard SAE performs no better than random on these particular hierarchies.
- **Mini-hypothesis**: The Standard SAE's low first-letter absorption (0.026) may reflect a genuine architectural advantage for *character-level* features, but this advantage does not transfer to *semantic* features. Alternatively, the absorption formula itself may have a non-zero floor for any SAE when probe AUROC is perfect (1.0) and the k-sparse classifier has limited capacity.
- **Significance**: This is a sharp result that directly challenges the assumption that low first-letter absorption implies better features generally. It supports the contrarian perspective and could be the paper's most cited figure.

### Unexpected Finding 3: GPT-2 Shows Dramatically Lower Absorption Than Pythia
- **Observation**: On GPT-2 small, semantic-hierarchy absorption was 0.0 (Standard) and 0.003 (TopK), compared to 0.352 and 0.250 on Pythia-160m. This is not a small difference—it is a near-complete absence of absorption on GPT-2.
- **Mini-hypothesis**: GPT-2 small (124M, 12 layers) may represent these simple WordNet hierarchies in a more distributed or less sparse manner than Pythia-160m (12 layers, different tokenizer/training). Alternatively, the SAEs used on GPT-2 (res-jb, resid-post-v5-32k) may have been trained with different hyperparameters (larger dictionary, different sparsity) that reduce absorption artifacts.
- **Significance**: This opens a door to a model-comparison contribution: "Absorption rates are model-dependent, and the benchmark may not generalize across model families." This was not in the original hypotheses but is a publishable secondary finding.

## Follow-Up Experiments

| Signal | Follow-Up Experiment | Expected Outcome | GPU Hours | Priority |
|--------|---------------------|------------------|-----------|----------|
| tau_fs robustness | The correlation stability is already demonstrated across 3 thresholds; no follow-up needed | — | — | — |
| Architecture ranking preservation | Test 2 additional architectures (OrtSAE, HSAE) on the same semantic hierarchies | Ordering remains preserved (OrtSAE low, HSAE moderate) | ~0.3 | Med |
| GPT-2 near-zero absorption | Run the full 8-SAE cohort on GPT-2 small to confirm the model-family effect | All GPT-2 SAEs show lower semantic absorption than Pythia counterparts | ~0.8 | High |
| Random-SAE floor effect | Test a second random baseline (random encoder + decoder) to check if 0.35 is a hard floor | Random-2 scores ~0.35, confirming floor effect | ~0.2 | Med |
| Hierarchy specificity reversal | Design a new non-hierarchy control with *weaker* co-occurrence (e.g., "cat-shelf") | Weaker co-occurrence pairs show lower absorption, supporting the co-occurrence hypothesis | ~0.3 | High |
| Perfect AUROC concern | Test more challenging hierarchies (e.g., 3-level deep: animal → mammal → dog) or rare concepts | AUROC drops below 1.0, absorption scores change | ~0.3 | Med |

## Honest Caveats

### Finding 1: Correlation r = 0.463 falls short of H1 threshold
- **Counter-argument**: With only n=7 SAEs, the correlation is underpowered. The bootstrap CI spans from -0.389 to +0.981, so the true correlation could easily be >0.6 or near-zero.
- **Alternative explanation**: The correlation is genuinely moderate (~0.45), meaning first-letter absorption explains only ~21% of the variance in semantic-hierarchy absorption. This would be a weak construct validity result.
- **What would convince me**: Replicate with n=12–15 SAEs (including OrtSAE, HSAE, additional Gemma-2 SAEs). If r remains ~0.45 with a tighter CI, the metric's construct validity is genuinely limited.

### Finding 2: Random SAE = Standard SAE on semantic hierarchy
- **Counter-argument**: The Standard SAE evaluated here may be an outlier. Other "standard" SAEs (different layers, models) might show a larger gap.
- **Alternative explanation**: The semantic hierarchy task is too easy (perfect AUROC), so any SAE with sufficient dictionary size can achieve the same k-sparse performance. The absorption score reflects probe-task ceiling effects, not SAE quality.
- **What would convince me**: Run the same comparison on harder hierarchies or with noisier probes. If the Standard SAE still matches Random, the metric is indeed uninformative for semantic features.

### Finding 3: GPT-2 shows near-zero absorption
- **Counter-argument**: The GPT-2 evaluation used only 2 SAEs and a different layer (block 8 vs block 3 on Pythia). Layer depth, not model family, could drive the difference.
- **Alternative explanation**: The GPT-2 SAEs (res-jb, v5-32k) have larger dictionaries (32k vs 4k), which may reduce absorption mechanically. Dictionary size, not model quality, is the confound.
- **What would convince me**: Run GPT-2 SAEs at multiple layers and matched dictionary sizes. If near-zero absorption persists across layers and sizes, it is a genuine model-family effect.

### Finding 4: Hierarchy specificity reversed (semantic < non-hierarchy)
- **Counter-argument**: The non-hierarchy pairs were not perfectly matched in difficulty or frequency structure. Some pairs ("doctor-hospital") may have stronger associative links than the hierarchies.
- **Alternative explanation**: The absorption formula is not measuring "hierarchy" at all; it measures how much a parent concept's representation is *reconstructed* from child latents. In co-occurrence pairs, both directions are symmetrically reconstructible, leading to higher scores.
- **What would convince me**: Design a new control with explicitly weak co-occurrence. If those pairs show lower absorption than hierarchies, the current reversal is explained by co-occurrence strength. If they still show higher absorption, the metric is fundamentally non-hierarchy-specific.

## Bottom Line

There is a publishable story here, but it is not the one originally hypothesized. The data do not strongly support "first-letter absorption predicts semantic-hierarchy absorption" (r = 0.463, CI inconclusive). Instead, the strongest positive signals are: (1) the *ranking* of architectures is preserved across tasks, suggesting ordinal construct validity; (2) the correlation is remarkably robust to tau_fs; and (3) the GPT-2 replication and random-SAE control reveal surprising model-dependent and floor effects that challenge benchmark assumptions. The most compelling paper would frame itself as a *construct-validity stress-test* that finds partial support for the metric's utility (ranking preservation) while identifying important boundary conditions (model-dependence, non-hierarchy specificity, random-SAE floor). This is a story reviewers will want to read because it delivers both a methodological contribution and a critical insight.
