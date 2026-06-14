#!/usr/bin/env python3
"""Fix missing dead-neuron metrics for 3 GPT-2 128k checkpoints."""

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
def compute_dead_neuron_and_recon(model, sae, sae_id: str, n_tokens=50_000, batch_size=16, context_size=128):
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    ds = ds.shuffle(seed=SEED)

    hook_name, _ = parse_hook_name_and_layer(sae_id)
    total_tokens = 0
    acts_all = []
    recon_acts_all = []

    for batch in ds.iter(batch_size=batch_size):
        if total_tokens >= n_tokens:
            break
        texts = batch["text"]
        tokens = model.to_tokens(texts, truncate=True)[:, :context_size]
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        acts = cache[hook_name]

        sae_acts = sae.encode(acts)
        recon_acts = sae.decode(sae_acts)

        acts_all.append(acts.detach().cpu())
        recon_acts_all.append(recon_acts.detach().cpu())

        total_tokens += tokens.numel()
        if total_tokens >= n_tokens:
            break

    acts_cat = torch.cat(acts_all, dim=0)
    recon_cat = torch.cat(recon_acts_all, dim=0)

    acts_flat = acts_cat.reshape(-1, acts_cat.shape[-1])
    recon_flat = recon_cat.reshape(-1, recon_cat.shape[-1])

    mse = ((acts_flat - recon_flat) ** 2).mean()
    total_var = ((acts_flat - acts_flat.mean(0)) ** 2).mean()
    explained_variance = 1.0 - (mse / total_var).item()

    sae_acts_full = sae.encode(acts_cat.to(DEVICE))
    l0 = (sae_acts_full > 0).float().sum(dim=-1).mean().item()

    ever_active = (sae_acts_full > 0).any(dim=0).any(dim=0)
    dead_neuron_fraction = 1.0 - ever_active.float().mean().item()

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
            recon_metrics = compute_dead_neuron_and_recon(
                model, sae, sae_id, n_tokens=DEAD_NEURON_TOKENS, batch_size=16, context_size=128
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
