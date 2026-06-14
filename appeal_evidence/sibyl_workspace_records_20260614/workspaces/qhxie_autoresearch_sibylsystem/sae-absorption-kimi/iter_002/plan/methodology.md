# Methodology: Construct Validity of the SAEBench Feature Absorption Metric

## Overview

This study tests whether the dominant first-letter absorption metric (Chanin et al., 2024; SAEBench) generalizes to matched-frequency semantic hierarchies drawn from WordNet. The experiment is entirely training-free and evaluates existing pretrained SAEs on three conditions:
1. **First-letter absorption** (standard SAEBench protocol)
2. **Semantic-hierarchy absorption** (novel WordNet-based hierarchies)
3. **Non-hierarchy correlated-feature absorption** (control condition)

## SAE Selection

We evaluate **8 SAE configurations** that span the absorption-rate spectrum, using the same architectures validated in the prior iteration (E2 full Pythia):

| Architecture | Expected Absorption | Source |
|--------------|---------------------|--------|
| MatryoshkaBatchTopK | Low | Pythia-160M, resid_post_layer_8, trainer_0 |
| PAnneal | Low-Moderate | Pythia-160M, resid_post_layer_8, trainer_0 |
| GatedSAE | Low-Moderate | Pythia-160M, resid_post_layer_8, trainer_0 |
| Standard | Moderate | Pythia-160M, resid_post_layer_8, trainer_0 |
| JumpRelu | Moderate | Pythia-160M, resid_post_layer_8, trainer_0 |
| BatchTopK | Moderate-High | Pythia-160M, resid_post_layer_8, trainer_0 |
| TopK | High | Pythia-160M, resid_post_layer_8, trainer_0 |
| Random-SAE Control | Near-zero | Permuted decoder directions from Standard |

**Rationale for Pythia-160M:** The SAEBench official absorption evaluator (`sae_bench.evals.absorption.main.run_eval`) has been validated end-to-end on this model in the prior iteration. All checkpoint files are already downloaded locally. GPT-2 small is retained as a replication control for the semantic-hierarchy probe pipeline.

## Semantic-Hierarchy Construction

### WordNet Parent-Child Pairs

We select **10 semantic hierarchies** where both parent and child are single tokens in the model vocabulary (verified for both GPT-2 and Pythia tokenizers):

| Parent | Children |
|--------|----------|
| building | house, school, library |
| container | box, bag, cup |
| tool | fork, comb, rake |
| room | cell, court, hall |
| device | fan, key |
| book | album, journal |
| tree | ash, poon |
| food | fish, water |
| fruit | berry, seed |
| animal | pet, male |

**Selection criteria:**
- Direct or near-direct hypernym relationship in WordNet (max 2 steps)
- Both parent and child are single tokens in GPT-2 and Pythia vocabularies
- Concepts are concrete and semantically distinct
- At least 2 children per parent to enable meaningful probe training

### Frequency Matching

For each hierarchy, we construct a **synthetic balanced dataset** where parent and child tokens appear with equal frequency. This controls the frequency-confound identified in the proposal:

1. Generate `N = 100` sentences per concept using simple templates (e.g., "The {concept} is a place with rich history.")
2. Ensure parent and child samples are exactly balanced within each hierarchy
3. Report token frequencies for transparency

## Absorption Measurement Protocol

### First-Letter Absorption (Baseline)

We use the **official SAEBench absorption evaluator**:
- `sae_bench.evals.absorption.main.run_eval`
- `AbsorptionEvalConfig(model_name="EleutherAI/pythia-160m-deduped", random_seed=42, llm_batch_size=256, llm_dtype="float32")`
- Metric: `mean_full_absorption_score` (primary), `mean_absorption_fraction_score` (secondary)

This protocol trains logistic-regression ground-truth probes on base-model activations for first-letter classification, then applies the SAEBench absorption formula to SAE latents using k-sparse probing.

### Semantic-Hierarchy Absorption

We replicate the **exact SAEBench absorption logic** on our WordNet hierarchies:

1. **Ground-truth probe:** For each parent concept, train a logistic regression probe on Pythia-160M residual-stream activations (layer 8, `resid_post`) to classify "parent vs. child".
2. **SAE latent probe:** Train the same probe on SAE latents.
3. **K-sparse probe:** Train the probe on top-k SAE latents (k=10, matching SAEBench default).
4. **Absorption score:** Compute `max(0, (resid_acc - sae_acc) / resid_acc, (resid_acc - k_sparse_acc) / resid_acc)`.
5. **Aggregate:** Average absorption scores across all 10 parent-child hierarchies.

**Probe training details:**
- 200 Adam epochs, lr=0.05
- Input: mean-pooled residual activations over non-padding positions
- Minimum probe AUROC threshold: 0.7 (hierarchies below this are flagged and reported)

### Non-Hierarchy Control Absorption

We compute absorption scores on **10 semantically correlated but non-hierarchical pairs**:

| Pair |
|------|
| big - large |
| fast - quick |
| begin - start |
| doctor - hospital |
| student - school |
| river - water |
| fire - heat |
| sun - light |
| tree - wood |
| stone - rock |

For each pair, we train a binary probe (word A vs. word B) and apply the same absorption formula. If the metric is hierarchy-specific, these scores should be near-zero and significantly lower than semantic-hierarchy scores.

### Random-SAE Control

We create a **random baseline** by permuting the decoder directions of the Standard SAE. Absorption scores on all three tasks should be near-zero. If they are not, this indicates the metric captures artifacts unrelated to learned structure.

## Evaluation Metrics

### Primary Metrics
- **First-letter absorption:** `mean_full_absorption_score` from SAEBench
- **Semantic-hierarchy absorption:** Custom mean absorption score across 10 hierarchies
- **Non-hierarchy control absorption:** Custom mean absorption score across 10 pairs

### Statistical Tests
- **H1 (Construct Validity):** Pearson correlation r between first-letter and semantic-hierarchy absorption across 8 SAEs. Bootstrap 95% CI (B=10,000). Supported if r > 0.6 and CI excludes 0. Rejected if CI includes values < 0.3.
- **H2 (Hierarchy Specificity):** Paired t-test comparing semantic-hierarchy vs. non-hierarchy control absorption. Supported if semantic > control and p < 0.05.
- **H3 (Robustness):** Pearson correlation computed at τ_fs ∈ {0.01, 0.03, 0.05}. Supported if r remains positive and > 0.5 across all thresholds.

### Probe Quality Checks
- Report per-hierarchy probe AUROC table
- Flag any hierarchy with AUROC < 0.7
- Report random-SAE scores for all three conditions

## Ablation Schedule

| Ablation | Method | Expected Outcome |
|----------|--------|------------------|
| **τ_fs sweep** | Re-run semantic-hierarchy absorption at τ_fs = 0.01, 0.03, 0.05 | Correlation should be stable if metric is robust |
| **GPT-2 replication** | Run semantic-hierarchy probe pipeline on GPT-2 small (layer 8) | Pattern should replicate if construct validity is model-general |
| **Random-SAE control** | Permute Standard SAE decoder directions | Near-zero absorption on all tasks |

## Expected Visualizations

- **Figure 1:** Architecture ranking comparison (first-letter vs. semantic-hierarchy absorption)
- **Figure 2:** Scatter plot of first-letter vs. semantic-hierarchy absorption with Pearson r and bootstrap CI
- **Figure 3:** Bar chart of semantic-hierarchy vs. non-hierarchy control scores per architecture
- **Table 1:** Main results (architecture × first-letter × semantic-hierarchy × non-hierarchy × random-SAE)
- **Table 2:** Per-hierarchy probe AUROC and absorption scores
- **Table 3:** τ_fs robustness analysis

## GPU Resource Plan

- **Model size:** Pythia-160M (84M params) — fits comfortably on 1 GPU
- **All tasks:** 1 GPU each, no multi-GPU needed
- **Estimated total GPU time:** ~45 minutes for full experiment (8 SAEs × ~5 min each for official eval + custom probe)
- **Pilot target:** 10–15 minutes for 2 SAEs

## Risk Mitigation

| Threat | Mitigation |
|--------|------------|
| Probe quality poor for semantic concepts | Pre-filter to AUROC > 0.7; report probe table |
| WordNet concepts not single tokens | Verified for both GPT-2 and Pythia tokenizers |
| Frequency matching imperfect | Use synthetic balanced datasets; report frequencies |
| Small-n correlation noisy | Bootstrap CI; interpret effect-size bounds |
| Official SAEBench eval incompatible with custom tasks | Use SAEBench logic as reference; implement custom probe pipeline with identical formulas |

## Code Reuse

- **Official absorption eval:** `sae_bench.evals.absorption.main.run_eval` (validated in iter_001)
- **Custom SAE loading:** `sae_bench.custom_saes.run_all_evals_dictionary_learning_saes.load_dictionary_learning_sae` (validated in iter_001)
- **Custom probe pipeline:** Adapted from iter_001 `e3_pilot.py` and `e3_validation.py`
- **Model loading:** `transformer_lens.HookedTransformer.from_pretrained` (validated in iter_001)
