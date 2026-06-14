#!/usr/bin/env python3
"""Fix missing dead-neuron metrics for 3 GPT-2 128k checkpoints - memory-efficient version."""

import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current")
E2_FULL_DIR = WORKSPACE / "exp" / "e2_full_gpt2"
RESULTS_PATH = E2_FULL_DIR / "absorption_results.json"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS_TO_FIX = [
    {"release": "gpt2-small-resid-post-v5-128k", "sae_id": "blocks.4.hook_resid_post", "model_name": "gpt2-small", "family": "TopK"},
    {"release": "gpt2-small-resid-post-v5-128k", "sae_id": "blocks.8.hook_resid_post", "model_name": "gpt2-small", "family": "TopK"},
    {"release": "gpt2-small-mlp-out-v5-128k", "sae_id": "blocks.8.hook_mlp_out", "model_name": "gpt2-small", "family": "TopK_MLP"},
]

DEAD_NEURON_TOKENS = 50_000


def parse_hook_name_and_layer(sae_id: str):
    base = sae_id.split("__")[0]
    return base, int(base.split(".")[1])


@torch.no_grad()
def compute_dead_neuron_and_recon_mem_efficient(model, sae, sae_id: str, n_tokens=50_000, batch_size=8, context_size=128):
    """Memory-efficient dead-neuron and reconstruction metrics."""
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    ds = ds.shuffle(seed=SEED)

    hook_name, _ = parse_hook_name_and_layer(sae_id)
    total_tokens = 0

    # For incremental statistics
    acts_sum = None
    acts_count = 0
    mse_sum = 0.0
    ever_active = None
    l0_sum = 0.0
    l0_count = 0

    for batch in ds.iter(batch_size=batch_size):
        if total_tokens >= n_tokens:
            break
        texts = batch["text"]
        tokens = model.to_tokens(texts, truncate=True)[:, :context_size]
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        acts = cache[hook_name]

        # Process one sample at a time to avoid OOM
        for i in range(acts.shape[0]):
            single_act = acts[i:i+1]
            sae_acts = sae.encode(single_act)
            recon_acts = sae.decode(sae_acts)

            # Flatten for statistics
            acts_flat = single_act.reshape(-1, single_act.shape[-1]).cpu()
            recon_flat = recon_acts.reshape(-1, recon_acts.shape[-1]).cpu()

            if acts_sum is None:
                acts_sum = acts_flat.sum(dim=0)
                acts_count = acts_flat.shape[0]
            else:
                acts_sum += acts_flat.sum(dim=0)
                acts_count += acts_flat.shape[0]

            mse_sum += ((acts_flat - recon_flat) ** 2).sum().item()

            # Dead neuron tracking
            sae_acts_cpu = (sae_acts > 0).cpu()
            if ever_active is None:
                ever_active = sae_acts_cpu.any(dim=0).any(dim=0)
            else:
                ever_active = ever_active | sae_acts_cpu.any(dim=0).any(dim=0)

            # L0 tracking
            l0_sum += (sae_acts_cpu.float().sum(dim=-1)).sum().item()
            l0_count += sae_acts_cpu.shape[0] * sae_acts_cpu.shape[1]

            del sae_acts, recon_acts, acts_flat, recon_flat, sae_acts_cpu
            torch.cuda.empty_cache()

        total_tokens += tokens.numel()
        if total_tokens >= n_tokens:
            break

    acts_mean = acts_sum / acts_count
    # Compute total variance using Welford's algorithm would be better, but let's do a second pass
    # Actually, for simplicity, let's compute variance from the accumulated MSE and mean
    # We need total_var = E[(x - mean)^2]
    # We'll do a lightweight second pass

    total_var_sum = 0.0
    ds2 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    ds2 = ds2.shuffle(seed=SEED)
    token_count2 = 0
    for batch in ds2.iter(batch_size=batch_size):
        if token_count2 >= n_tokens:
            break
        texts = batch["text"]
        tokens = model.to_tokens(texts, truncate=True)[:, :context_size]
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        acts = cache[hook_name]
        for i in range(acts.shape[0]):
            single_act = acts[i:i+1]
            acts_flat = single_act.reshape(-1, single_act.shape[-1]).cpu()
            total_var_sum += ((acts_flat - acts_mean) ** 2).sum().item()
            token_count2 += acts_flat.shape[0]
            if token_count2 >= acts_count:
                break
        if token_count2 >= acts_count:
            break

    total_var = total_var_sum / acts_count
    mse = mse_sum / acts_count
    explained_variance = 1.0 - (mse / total_var) if total_var > 0 else 0.0

    dead_neuron_fraction = 1.0 - ever_active.float().mean().item()
    l0 = l0_sum / l0_count if l0_count > 0 else 0.0

    return {
        "dead_neuron_fraction": dead_neuron_fraction,
        "l0": l0,
        "explained_variance": explained_variance,
        "n_tokens_evaluated": total_tokens,
    }


def main():
    results = json.loads(RESULTS_PATH.read_text())
    checkpoints = results["checkpoints"]

    print("Loading GPT-2 Small model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE, dtype=torch.float32)

    for cp in CHECKPOINTS_TO_FIX:
        release = cp["release"]
        sae_id = cp["sae_id"]
        print(f"\n=== Fixing {release}_{sae_id} ===")

        torch.cuda.empty_cache()
        gc.collect()

        try:
            sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
            if isinstance(sae, tuple):
                sae = sae[0]
        except Exception as e:
            print(f"  ERROR loading SAE: {e}")
            continue

        try:
            recon_metrics = compute_dead_neuron_and_recon_mem_efficient(
                model, sae, sae_id, n_tokens=DEAD_NEURON_TOKENS, batch_size=4, context_size=128
            )
            print(f"  Dead neurons: {recon_metrics['dead_neuron_fraction']:.4f}")
            print(f"  L0: {recon_metrics['l0']:.2f}")
            print(f"  Explained variance: {recon_metrics['explained_variance']:.4f}")

            for c in checkpoints:
                if c["release"] == release and c["sae_id"] == sae_id:
                    c["dead_neuron_fraction"] = recon_metrics["dead_neuron_fraction"]
                    c["l0"] = recon_metrics["l0"]
                    c["explained_variance"] = recon_metrics["explained_variance"]
                    c["n_tokens_evaluated"] = recon_metrics["n_tokens_evaluated"]
                    break
        except Exception as e:
            print(f"  ERROR in recon metrics: {e}")
            import traceback
            traceback.print_exc()

        del sae
        torch.cuda.empty_cache()
        gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nUpdated results saved to {RESULTS_PATH}")

    # Update summary.md
    summary_md = E2_FULL_DIR / "summary.md"
    ok_count = sum(1 for m in checkpoints if m.get("dead_neuron_fraction") is not None)
    summary_md.write_text(
        f"""# E2 Full GPT-2 Summary

**Task:** e2_full_gpt2
**Total Time:** {results['total_time_sec']/60:.1f} min
**Successful:** {ok_count}/{len(checkpoints)}

## Results

| Checkpoint | Family | Absorption (Full) | L0 | Dead Neurons | Explained Var |
|---|---|---|---|---|---|
"""
        + "\n".join(
            [
                f"| {m['release']}/{m['sae_id']} | {m['family']} | {m.get('official_absorption_full', 'N/A') if m.get('official_absorption_full') is not None else 'N/A'} | {m.get('l0', 'N/A') if m.get('l0') is not None else 'N/A'} | {m.get('dead_neuron_fraction', 'N/A') if m.get('dead_neuron_fraction') is not None else 'N/A'} | {m.get('explained_variance', 'N/A') if m.get('explained_variance') is not None else 'N/A'} |"
                for m in checkpoints
            ]
        )
        + "\n\n## Family-Level Statistics\n\n"
    )
    print("Summary updated.")


if __name__ == "__main__":
    main()
