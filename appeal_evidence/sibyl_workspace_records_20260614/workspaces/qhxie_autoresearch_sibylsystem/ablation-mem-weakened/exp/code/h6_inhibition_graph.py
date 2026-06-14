"""
H6 Gatekeeper Experiment: Inhibition Graph Predicts Absorption Pairs

Constructs the local inhibition graph from decoder correlations and tests whether
graph edges predict Chanin absorption pairs.

Expected time: ~5-10 minutes (training-free, GPU optional)
"""

import json
import torch
import numpy as np
from sae_lens import SAE


def load_absorption_data():
    """Load existing absorption data from iteration 1."""
    with open("iter_001/exp/results/pilots/absorption_layer8_16k.json") as f:
        data = json.load(f)

    letter_features = data["letter_features"]  # A-Z -> feature_id, cosine_similarity

    with open("exp/results/full/correlation_report_full.json") as f:
        report = json.load(f)

    absorption_rates = report["layer_results"]["layer_8"]["absorption_rates"]

    return letter_features, absorption_rates


def load_sae():
    """Load GPT-2 Small res-jb SAE at layer 8."""
    print("Loading SAE (gpt2-small-res-jb, layer 8)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=device,
    )
    print(f"SAE loaded: d_model={cfg_dict['d_in']}, d_dict={cfg_dict['d_sae']}")
    print(f"Device: {device}")
    return sae, device


def compute_inhibition_graph(sae, device, letter_features, absorption_rates, top_k=20):
    """
    Compute decoder correlations and test if top-k neighbors predict absorption.

    For each first-letter feature:
    1. Get its decoder direction
    2. Compute correlation with all other decoder directions
    3. Find top-k most correlated neighbors
    4. Check if those neighbors are HIGH-absorption features
    """
    W_dec = sae.W_dec.data  # shape: (d_dict, d_model)
    d_dict = W_dec.shape[0]

    # Normalize decoder directions for cosine similarity
    W_dec_norm = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-8)

    # Build feature_id -> letter mapping
    fid_to_letter = {v["feature_id"]: k for k, v in letter_features.items()}

    # Identify HIGH absorption features
    high_abs_letters = [k for k, v in absorption_rates.items() if v > 0.10]
    print(f"\nHIGH absorption features (>{0.10}): {high_abs_letters}")
    print(f"Count: {len(high_abs_letters)} out of 26")

    results = {}

    for letter, info in letter_features.items():
        feature_id = info["feature_id"]

        # Map to layer-local index if needed
        # For gpt2-small-res-jb with 24576 latents, feature_ids may be global
        # Use modulo to get local index
        local_idx = feature_id % d_dict

        # Get decoder direction for this feature
        d_i = W_dec_norm[local_idx]

        # Compute correlation with all other directions
        correlations = W_dec_norm @ d_i  # shape: (d_dict,)

        # Exclude self
        correlations[local_idx] = -1.0

        # Get top-k neighbors
        top_k_values, top_k_indices = torch.topk(correlations, top_k)

        # Check which of these neighbors are first-letter features
        neighbor_letters = []
        neighbor_absorption = []
        for idx in top_k_indices.tolist():
            # Map back to global feature_id
            global_fid = idx  # Approximate mapping
            # Find if this index corresponds to a known letter feature
            for l, info2 in letter_features.items():
                if info2["feature_id"] % d_dict == idx:
                    neighbor_letters.append(l)
                    neighbor_absorption.append(absorption_rates.get(l, 0.0))
                    break

        # Count how many top-k neighbors are HIGH absorption
        high_in_topk = sum(1 for a in neighbor_absorption if a > 0.10)

        results[letter] = {
            "feature_id": feature_id,
            "local_idx": local_idx,
            "absorption_rate": absorption_rates.get(letter, 0.0),
            "top_k_indices": top_k_indices.tolist(),
            "top_k_correlations": top_k_values.tolist(),
            "neighbor_letters": neighbor_letters,
            "neighbor_absorption": neighbor_absorption,
            "high_absorption_in_topk": high_in_topk,
            "precision_at_k": high_in_topk / top_k if top_k > 0 else 0.0,
        }

    return results, high_abs_letters


def analyze_results(results, high_abs_letters, top_k=20):
    """Compute aggregate statistics."""
    print(f"\n{'='*60}")
    print("H6 RESULTS: Inhibition Graph Predicts Absorption Pairs")
    print(f"{'='*60}")

    # Overall precision@k
    total_high_in_topk = sum(r["high_absorption_in_topk"] for r in results.values())
    total_predictions = len(results) * top_k
    overall_precision = total_high_in_topk / total_predictions

    # Precision for HIGH absorption features only
    high_results = {k: v for k, v in results.items() if k in high_abs_letters}
    high_precision = (
        sum(r["high_absorption_in_topk"] for r in high_results.values())
        / (len(high_results) * top_k)
        if high_results
        else 0.0
    )

    # Random baseline (chance)
    n_high = len(high_abs_letters)
    n_total = 26
    chance_precision = n_high / n_total  # Expected if random

    # Enrichment factor
    enrichment = overall_precision / chance_precision if chance_precision > 0 else 0.0

    print(f"\nTop-k = {top_k}")
    print(f"Total predictions: {total_predictions}")
    print(f"HIGH absorption hits in top-k: {total_high_in_topk}")
    print(f"Overall precision@{top_k}: {overall_precision:.4f}")
    print(f"Precision@{top_k} for HIGH-abs features: {high_precision:.4f}")
    print(f"Random chance: {chance_precision:.4f}")
    print(f"Enrichment over chance: {enrichment:.1f}x")

    # Fisher exact test approximation
    from scipy.stats import fisher_exact

    table = [
        [total_high_in_topk, total_predictions - total_high_in_topk],
        [n_high * top_k, (n_total - n_high) * top_k],
    ]
    try:
        _, p_value = fisher_exact(table, alternative="greater")
        print(f"Fisher exact test p-value: {p_value:.6f}")
    except Exception as e:
        print(f"Fisher test failed: {e}")
        p_value = 1.0

    print(f"\n{'='*60}")
    print("VERDICT:")
    if overall_precision >= 0.10:
        print(f"  PASS: precision@{top_k} = {overall_precision:.4f} >= 0.10 threshold")
        print(f"  GO: Proceed with full H6-H10 framework")
    elif overall_precision >= 0.05:
        print(f"  CAUTION: precision@{top_k} = {overall_precision:.4f} (0.05-0.10)")
        print(f"  Proceed with diagnostic-only claims")
    else:
        print(f"  FAIL: precision@{top_k} = {overall_precision:.4f} < 0.05")
        print(f"  NO-GO: Pivot to trade-off analysis")
    print(f"{'='*60}")

    return {
        "top_k": top_k,
        "overall_precision": overall_precision,
        "high_precision": high_precision,
        "chance_precision": chance_precision,
        "enrichment": enrichment,
        "fisher_p_value": p_value,
        "total_predictions": total_predictions,
        "total_hits": total_high_in_topk,
        "verdict": "PASS" if overall_precision >= 0.10 else ("CAUTION" if overall_precision >= 0.05 else "FAIL"),
    }


def main():
    print("=" * 60)
    print("H6 Gatekeeper Experiment")
    print("Local Inhibition Graph Predicts Absorption Pairs")
    print("=" * 60)

    # Load data
    letter_features, absorption_rates = load_absorption_data()
    print(f"Loaded {len(letter_features)} first-letter features")

    # Load SAE
    sae, device = load_sae()

    # Compute inhibition graph
    results, high_abs_letters = compute_inhibition_graph(
        sae, device, letter_features, absorption_rates, top_k=20
    )

    # Analyze
    summary = analyze_results(results, high_abs_letters, top_k=20)

    # Save results
    output = {
        "h6_results": results,
        "summary": summary,
        "high_absorption_letters": high_abs_letters,
    }

    output_path = "exp/results/full/h6_inhibition_graph.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return summary


if __name__ == "__main__":
    main()
