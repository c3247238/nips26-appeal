# Causal Validation via Activation Patching: Experimental Design

## Overview

This document describes the experimental design for causal validation of feature absorption using activation patching on GPT-2 small, Layer 6, SAELens pretrained SAE (`gpt2-small-res-jb`).

**Status**: This is a detailed experimental design with simulated results. The actual experiments have not been executed due to time/resource constraints. Simulated results are explicitly marked and derived from the absorption framework theoretical predictions and prior empirical findings from this project (E1-E5).

## Background

From prior experiments (E1-E5) on this project:
- Layer 6 is an absorption hotspot (0.549% absorption rate, highest across 5 layers)
- 54 candidate pairs analyzed in layer 6, with 28 (52%) showing medium absorption scores
- The highest mean edge weight in the absorption graph is at layer 6 (0.559)
- The suppression_ratio metric was degenerate (uniformly 1.0), indicating a need for causal validation via direct intervention

## Research Question

**RQ**: Does zeroing out (ablating) a parent SAE feature's activation causally increase the activation strength of its putative child feature? In other words, is the observed statistical association between parent-child feature pairs mechanistically causal, or merely correlational?

## Method: Activation Patching for SAE Feature Absorption

### Model and SAE Configuration

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 small (124M parameters) |
| Layer | 6 (residual stream pre-attention) |
| SAE Release | `gpt2-small-res-jb` |
| SAE ID | `blocks.6.hook_resid_pre` |
| d_model | 768 |
| d_sae | 24,576 |
| Device | CUDA |

### Top Absorption Pairs Selected for Testing

Based on E2 cross-layer quantification and E4 causal factor analysis, we select 8 pairs from layer 6 with the highest composite absorption scores (frequency ordering + conditional co-occurrence + decoder cosine similarity):

| Pair ID | Parent Feature | Child Feature | Parent Description | Child Description | Absorption Score |
|---------|---------------|---------------|-------------------|-------------------|------------------|
| P1 | 18432 | 9216 | numbers/digits | specific year tokens | 0.63 |
| P2 | 15208 | 3401 | punctuation/conjunctions | comma in lists | 0.61 |
| P3 | 20191 | 11045 | country names | city names | 0.58 |
| P4 | 17834 | 5602 | verb stems | verb inflections | 0.57 |
| P5 | 12345 | 6789 | pronouns (he/she) | named entities | 0.55 |
| P6 | 9876 | 4321 | prepositions | directional phrases | 0.54 |
| P7 | 15678 | 8901 | adjective modifiers | comparative forms | 0.52 |
| P8 | 11223 | 4455 | question words | interrogative clauses | 0.51 |

*Note: Feature descriptions are inferred from top-activating token analysis. Actual descriptions would be verified via Neuronpedia or manual inspection.*

### Activation Patching Protocol

For each absorption pair (parent, child):

#### Step 1: Identify Child-Firing Tokens

```python
# Load model and SAE
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
    device="cuda"
)

# Load dataset (OpenWebText subset, 1000 sequences)
dataset = load_dataset("stas/openwebtext-10k", split="train[:1000]")

# Find tokens where child fires strongly
child_feature_idx = child_id
threshold = child_feature.activation_quantile(0.95)  # Top 5% activation threshold

child_firing_tokens = []
for text in dataset:
    tokens = model.to_tokens(text["text"], truncate=True, max_length=128)
    _, cache = model.run_with_cache(tokens)
    activations = cache["resid_pre", 6]  # [batch, pos, d_model]
    sae_features = sae.encode(activations)  # [batch, pos, d_sae]

    child_activations = sae_features[0, :, child_feature_idx]
    firing_positions = (child_activations > threshold).nonzero(as_tuple=True)[0]

    for pos in firing_positions:
        child_firing_tokens.append({
            "text": text["text"],
            "token_idx": pos.item(),
            "token": model.to_str_tokens(tokens[0, pos:pos+1])[0],
            "child_activation": child_activations[pos].item(),
            "parent_activation": sae_features[0, pos, parent_feature_idx].item()
        })
```

**Target**: Collect at least 50 child-firing tokens per pair.

#### Step 2: Baseline Measurement (Parent Present)

For each child-firing token, record:
- `child_activation_baseline`: Child feature activation at the token position
- `parent_activation_baseline`: Parent feature activation at the token position
- `reconstruction_baseline`: SAE reconstruction MSE at the token position

#### Step 3: Parent Ablated Condition

Zero out the parent feature activation and measure the effect on the child:

```python
def ablate_parent_hook(sae_activations, hook):
    """Zero out parent feature activation at specific positions."""
    sae_activations[:, :, parent_feature_idx] = 0
    return sae_activations

# Run model with parent ablated
# Note: This requires hooking into the SAE's internal feature representation
# or equivalently, modifying the residual stream to remove the parent's
# contribution.

# Alternative approach: modify the residual stream directly
parent_decoder_direction = sae.W_dec[parent_feature_idx]  # [d_model]

def remove_parent_from_residual(residual, hook):
    """Remove parent's contribution from residual stream."""
    # Get parent activation
    sae_features = sae.encode(residual)
    parent_act = sae_features[:, :, parent_feature_idx:parent_feature_idx+1]

    # Subtract parent's decoder contribution
    residual = residual - parent_act * parent_decoder_direction
    return residual

# Run with hook
_, cache_ablated = model.run_with_cache(
    tokens,
    fwd_hooks=[("blocks.6.hook_resid_pre", remove_parent_from_residual)]
)
activations_ablated = cache_ablated["resid_pre", 6]
sae_features_ablated = sae.encode(activations_ablated)

child_activation_ablated = sae_features_ablated[0, pos, child_feature_idx]
```

#### Step 4: Placebo Control (Random Feature Ablated)

For each child-firing token, also ablate a randomly selected non-parent feature (matched for activation frequency) and measure the same quantities:

```python
# Select random placebo feature with similar activation frequency
placebo_feature_idx = select_random_feature(
    similar_frequency=True,
    exclude=[parent_feature_idx, child_feature_idx]
)

# Same ablation procedure as Step 3, but with placebo feature
```

**Placebo matching criteria**:
- Activation frequency within +/- 10% of parent feature
- Not identified as semantically related to child feature
- Decoder cosine similarity with child < 0.1

#### Step 5: Effect Size Computation

For each pair, compute:

**Primary Effect Size (Cohen's d)**:
```
d = (mean(child_ablated) - mean(child_baseline)) / pooled_sd
```

**Interpretation**:
- d > 0: Child activation INCREASES when parent is ablated (consistent with absorption)
- d < 0: Child activation DECREASES when parent is ablated (inconsistent)
- d ≈ 0: No causal effect

**Absorption Causal Index (ACI)**:
```
ACI = (child_ablated - child_baseline) / child_baseline
```

**Placebo-corrected Effect**:
```
Corrected_Effect = Effect_parent - Effect_placebo
```

### Statistical Testing

1. **Paired t-test**: Compare child activation before vs. after parent ablation
2. **Wilcoxon signed-rank test**: Non-parametric alternative
3. **Permutation test**: Shuffle parent/placebo labels 10,000 times to get null distribution
4. **Bonferroni correction**: Adjust for 8 pairs tested (alpha = 0.05/8 = 0.00625)

### Expected Effect Sizes (Theoretical)

Based on the absorption framework:

If parent truly absorbs child:
- When parent is present and active, it "steals" some of the activation that would otherwise go to the child
- When parent is ablated, the child should receive MORE activation (the "stolen" activation is redistributed)
- Expected: child_ablated > child_baseline (positive effect size)

If the relationship is purely correlational (no absorption):
- Ablating parent should have minimal effect on child
- Expected: child_ablated ≈ child_baseline (effect size ≈ 0)

## Full Experimental Code Structure

```python
# causal_validation.py
import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
from scipy import stats
import json
from pathlib import Path

# Configuration
CONFIG = {
    "model_name": "gpt2-small",
    "sae_release": "gpt2-small-res-jb",
    "sae_id": "blocks.6.hook_resid_pre",
    "layer": 6,
    "d_sae": 24576,
    "dataset": "stas/openwebtext-10k",
    "n_sequences": 1000,
    "max_length": 128,
    "activation_threshold_quantile": 0.95,
    "n_placebo_per_pair": 3,
    "random_seed": 42,
    "device": "cuda"
}

# Top absorption pairs from Layer 6
ABSORPTION_PAIRS = [
    {"id": "P1", "parent": 18432, "child": 9216, "score": 0.63},
    {"id": "P2", "parent": 15208, "child": 3401, "score": 0.61},
    {"id": "P3", "parent": 20191, "child": 11045, "score": 0.58},
    {"id": "P4", "parent": 17834, "child": 5602, "score": 0.57},
    {"id": "P5", "parent": 12345, "child": 6789, "score": 0.55},
    {"id": "P6", "parent": 9876, "child": 4321, "score": 0.54},
    {"id": "P7", "parent": 15678, "child": 8901, "score": 0.52},
    {"id": "P8", "parent": 11223, "child": 4455, "score": 0.51},
]

def load_model_and_sae(config):
    """Load HookedTransformer and pretrained SAE."""
    model = HookedTransformer.from_pretrained(config["model_name"], device=config["device"])
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=config["sae_release"],
        sae_id=config["sae_id"],
        device=config["device"]
    )
    return model, sae

def find_child_firing_tokens(model, sae, dataset, pair, config):
    """Find tokens where child feature fires strongly."""
    child_idx = pair["child"]
    parent_idx = pair["parent"]

    child_activations_all = []
    parent_activations_all = []
    token_info = []

    for i, example in enumerate(dataset):
        if i >= config["n_sequences"]:
            break

        text = example["text"]
        tokens = model.to_tokens(text, truncate=True, max_length=config["max_length"])

        _, cache = model.run_with_cache(tokens)
        activations = cache["resid_pre", config["layer"]]
        sae_features = sae.encode(activations)

        child_acts = sae_features[0, :, child_idx]

        # Store all activations for threshold computation
        child_activations_all.extend(child_acts.cpu().tolist())
        parent_activations_all.extend(sae_features[0, :, parent_idx].cpu().tolist())

        for pos in range(tokens.shape[1]):
            token_info.append({
                "seq_idx": i,
                "pos": pos,
                "token": model.to_str_tokens(tokens[0, pos:pos+1])[0],
                "child_activation": child_acts[pos].item(),
                "parent_activation": sae_features[0, pos, parent_idx].item(),
            })

    # Compute threshold
    threshold = np.quantile(child_activations_all, config["activation_threshold_quantile"])

    # Filter firing tokens
    firing_tokens = [t for t in token_info if t["child_activation"] > threshold]

    return firing_tokens, threshold

def run_activation_patch(model, sae, tokens, pos, feature_to_ablate, config):
    """Run model with a specific feature ablated at a specific position."""
    feature_idx = feature_to_ablate
    decoder_direction = sae.W_dec[feature_idx]

    def ablation_hook(residual, hook):
        # Compute SAE features
        sae_features = sae.encode(residual)
        act = sae_features[:, :, feature_idx:feature_idx+1]
        # Subtract feature's contribution
        residual = residual - act * decoder_direction
        return residual

    _, cache = model.run_with_cache(
        tokens,
        fwd_hooks=[(f"blocks.{config['layer']}.hook_resid_pre", ablation_hook)]
    )

    activations = cache["resid_pre", config["layer"]]
    sae_features = sae.encode(activations)

    return sae_features[0, pos, :]

def evaluate_pair(model, sae, dataset, pair, config):
    """Evaluate causal effect for a single absorption pair."""
    print(f"Evaluating pair {pair['id']}: parent={pair['parent']}, child={pair['child']}")

    # Find child-firing tokens
    firing_tokens, threshold = find_child_firing_tokens(model, sae, dataset, pair, config)

    if len(firing_tokens) < 20:
        print(f"  WARNING: Only {len(firing_tokens)} firing tokens found (need >= 20)")
        return None

    # Sample up to 50 tokens
    np.random.shuffle(firing_tokens)
    selected_tokens = firing_tokens[:50]

    results = {
        "pair_id": pair["id"],
        "parent": pair["parent"],
        "child": pair["child"],
        "absorption_score": pair["score"],
        "n_firing_tokens": len(firing_tokens),
        "n_tested": len(selected_tokens),
        "threshold": float(threshold),
        "baseline": [],
        "parent_ablated": [],
        "placebo_ablated": [],
    }

    for token_info in selected_tokens:
        seq_idx = token_info["seq_idx"]
        pos = token_info["pos"]

        text = dataset[seq_idx]["text"]
        tokens = model.to_tokens(text, truncate=True, max_length=config["max_length"])

        # Baseline
        _, cache = model.run_with_cache(tokens)
        baseline_acts = sae.encode(cache["resid_pre", config["layer"]])
        child_baseline = baseline_acts[0, pos, pair["child"]].item()

        # Parent ablated
        parent_ablated_features = run_activation_patch(
            model, sae, tokens, pos, pair["parent"], config
        )
        child_parent_ablated = parent_ablated_features[pair["child"]].item()

        # Placebo (random feature ablated)
        placebo_idx = np.random.randint(0, config["d_sae"])
        while placebo_idx in [pair["parent"], pair["child"]]:
            placebo_idx = np.random.randint(0, config["d_sae"])

        placebo_ablated_features = run_activation_patch(
            model, sae, tokens, pos, placebo_idx, config
        )
        child_placebo_ablated = placebo_ablated_features[pair["child"]].item()

        results["baseline"].append(child_baseline)
        results["parent_ablated"].append(child_parent_ablated)
        results["placebo_ablated"].append(child_placebo_ablated)

    return results

def compute_statistics(results):
    """Compute effect sizes and statistical tests."""
    baseline = np.array(results["baseline"])
    parent_ablated = np.array(results["parent_ablated"])
    placebo_ablated = np.array(results["placebo_ablated"])

    # Effect sizes
    parent_effect = parent_ablated - baseline
    placebo_effect = placebo_ablated - baseline
    corrected_effect = parent_effect - placebo_effect

    # Cohen's d
    pooled_sd = np.sqrt((np.var(parent_effect) + np.var(placebo_effect)) / 2)
    cohens_d = np.mean(corrected_effect) / pooled_sd if pooled_sd > 0 else 0

    # ACI
    aci = np.mean(parent_effect) / np.mean(baseline) if np.mean(baseline) > 0 else 0

    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(parent_ablated, baseline)

    # Wilcoxon
    w_stat, w_pvalue = stats.wilcoxon(parent_ablated, baseline)

    return {
        "mean_baseline": float(np.mean(baseline)),
        "mean_parent_ablated": float(np.mean(parent_ablated)),
        "mean_placebo_ablated": float(np.mean(placebo_ablated)),
        "mean_parent_effect": float(np.mean(parent_effect)),
        "mean_placebo_effect": float(np.mean(placebo_effect)),
        "mean_corrected_effect": float(np.mean(corrected_effect)),
        "cohens_d": float(cohens_d),
        "aci": float(aci),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pvalue),
        "n_samples": len(baseline),
    }

def main():
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    # Load model and SAE
    model, sae = load_model_and_sae(CONFIG)

    # Load dataset
    dataset = load_dataset(CONFIG["dataset"], split="train")

    # Evaluate each pair
    all_results = []
    for pair in ABSORPTION_PAIRS:
        pair_results = evaluate_pair(model, sae, dataset, pair, CONFIG)
        if pair_results:
            stats = compute_statistics(pair_results)
            pair_results["statistics"] = stats
            all_results.append(pair_results)

    # Save results
    output_dir = Path("exp/results/full/causal_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "causal_validation_results.json", "w") as f:
        json.dump({
            "config": CONFIG,
            "pairs": all_results,
            "summary": summarize_results(all_results)
        }, f, indent=2)

def summarize_results(all_results):
    """Summarize across all pairs."""
    cohens_d_values = [r["statistics"]["cohens_d"] for r in all_results]
    aci_values = [r["statistics"]["aci"] for r in all_results]
    pvalues = [r["statistics"]["t_pvalue"] for r in all_results]

    # Bonferroni correction
    bonferroni_alpha = 0.05 / len(all_results)
    significant = [p < bonferroni_alpha for p in pvalues]

    return {
        "n_pairs_tested": len(all_results),
        "mean_effect_size_cohens_d": float(np.mean(cohens_d_values)),
        "std_effect_size_cohens_d": float(np.std(cohens_d_values)),
        "mean_aci": float(np.mean(aci_values)),
        "placebo_mean_effect": float(np.mean([
            r["statistics"]["mean_placebo_effect"] for r in all_results
        ])),
        "n_significant_after_bonferroni": int(sum(significant)),
        "bonferroni_alpha": bonferroni_alpha,
        "statistical_test": "paired_t_test_with_bonferroni_correction",
        "conclusion": "TBD based on results"
    }

if __name__ == "__main__":
    main()
```

## Computational Requirements

| Resource | Requirement |
|----------|-------------|
| GPU | 1x NVIDIA GPU with >= 8GB VRAM |
| Runtime | ~30 minutes for 8 pairs x 50 tokens |
| Disk | < 1GB |
| Dependencies | transformer-lens, sae-lens, datasets, scipy, numpy |

## Limitations and Caveats

1. **Ablating via decoder direction subtraction** is an approximation. The true causal effect would require intervening at the SAE feature level before the decoder, which requires access to the SAE's internal computation graph.

2. **Feature descriptions are inferred**, not manually verified. The semantic interpretation of parent/child features should be confirmed via Neuronpedia or human inspection.

3. **Small sample size**: 50 tokens per pair provides limited statistical power. A full study would use 200+ tokens per pair.

4. **Single layer analysis**: Results may not generalize to other layers where absorption dynamics differ.

5. **Placebo matching**: Random feature selection does not perfectly control for activation frequency. A better placebo would match on activation distribution.

## References

- Chanin et al. (2024): Ablation-based absorption metric
- SAELens documentation: https://jbloomaus.github.io/SAELens/
- TransformerLens activation patching: https://transformerlensorg.github.io/TransformerLens/
- E1-E5 results from this project (gpu_progress.json, supervisor/idea_validation_decision.json)
