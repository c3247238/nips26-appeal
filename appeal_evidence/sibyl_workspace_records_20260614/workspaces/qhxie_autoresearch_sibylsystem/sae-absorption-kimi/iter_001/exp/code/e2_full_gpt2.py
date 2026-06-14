#!/usr/bin/env python3
"""E2 Full: Official absorption metric on 14 GPT-2 Small checkpoints.

Runs sae-bench absorption eval across Standard, TopK, TopK_MLP, and TopK_Attn families.
Computes dead-neuron rate on >=50k tokens and basic reconstruction metrics.
"""

import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer

# Monkey-patch sae-bench to support gpt2-small
from sae_bench.sae_bench_utils import activation_collection
activation_collection.LLM_NAME_TO_BATCH_SIZE["gpt2-small"] = 128
activation_collection.LLM_NAME_TO_DTYPE["gpt2-small"] = "float32"

from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.absorption.main import run_eval

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_ID = "e2_full_gpt2"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
E2_FULL_DIR = WORKSPACE / "exp" / "e2_full_gpt2"
E2_FULL_DIR.mkdir(parents=True, exist_ok=True)

PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 14 checkpoints across 4 families with layer diversity
CHECKPOINTS = [
    # Standard (res-jb)
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.0.hook_resid_pre", "model_name": "gpt2-small", "family": "Standard"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.4.hook_resid_pre", "model_name": "gpt2-small", "family": "Standard"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.8.hook_resid_pre", "model_name": "gpt2-small", "family": "Standard"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.11.hook_resid_pre", "model_name": "gpt2-small", "family": "Standard"},
    # TopK (resid-post v5 32k)
    {"release": "gpt2-small-resid-post-v5-32k", "sae_id": "blocks.0.hook_resid_post", "model_name": "gpt2-small", "family": "TopK"},
    {"release": "gpt2-small-resid-post-v5-32k", "sae_id": "blocks.4.hook_resid_post", "model_name": "gpt2-small", "family": "TopK"},
    {"release": "gpt2-small-resid-post-v5-32k", "sae_id": "blocks.8.hook_resid_post", "model_name": "gpt2-small", "family": "TopK"},
    {"release": "gpt2-small-resid-post-v5-32k", "sae_id": "blocks.11.hook_resid_post", "model_name": "gpt2-small", "family": "TopK"},
    # TopK (resid-post v5 128k)
    {"release": "gpt2-small-resid-post-v5-128k", "sae_id": "blocks.4.hook_resid_post", "model_name": "gpt2-small", "family": "TopK"},
    {"release": "gpt2-small-resid-post-v5-128k", "sae_id": "blocks.8.hook_resid_post", "model_name": "gpt2-small", "family": "TopK"},
    # TopK_MLP (mlp-out v5 32k)
    {"release": "gpt2-small-mlp-out-v5-32k", "sae_id": "blocks.4.hook_mlp_out", "model_name": "gpt2-small", "family": "TopK_MLP"},
    {"release": "gpt2-small-mlp-out-v5-32k", "sae_id": "blocks.8.hook_mlp_out", "model_name": "gpt2-small", "family": "TopK_MLP"},
    {"release": "gpt2-small-mlp-out-v5-128k", "sae_id": "blocks.8.hook_mlp_out", "model_name": "gpt2-small", "family": "TopK_MLP"},
    # TopK_Attn (attn-out v5 32k)
    {"release": "gpt2-small-attn-out-v5-32k", "sae_id": "blocks.4.hook_attn_out", "model_name": "gpt2-small", "family": "TopK_Attn"},
    {"release": "gpt2-small-attn-out-v5-32k", "sae_id": "blocks.8.hook_attn_out", "model_name": "gpt2-small", "family": "TopK_Attn"},
]

DEAD_NEURON_TOKENS = 50_000
K_SPARSE_EPOCHS = 10


# ---------------------------------------------------------------------------
# Process tracking
# ---------------------------------------------------------------------------
PID_FILE.write_text(str(os.getpid()))
start_time_iso = datetime.now().isoformat()


def report_progress(epoch, total_epochs, step=0, total_steps=0, message=""):
    PROGRESS_FILE.write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": step,
                "total_steps": total_steps,
                "message": message,
                "updated_at": datetime.now().isoformat(),
            }
        )
    )


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    progress = {}
    if PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "status": status,
                "summary": summary,
                "final_progress": progress,
                "timestamp": datetime.now().isoformat(),
            }
        )
    )


# ---------------------------------------------------------------------------
# Dead-neuron and reconstruction metrics
# ---------------------------------------------------------------------------
def parse_hook_name_and_layer(sae_id: str):
    """Parse hook_name and hook_layer from sae_id string."""
    base = sae_id.split("__")[0]
    return base, int(base.split(".")[1])


@torch.no_grad()
def compute_dead_neuron_and_recon(model, sae, sae_id: str, n_tokens=50_000, batch_size=32, context_size=128):
    """Compute dead-neuron fraction, L0, explained variance on C4 validation."""
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    report_progress(1, len(CHECKPOINTS) + 2, message="Starting E2 full GPT-2 absorption benchmark")

    metrics_list = []
    total_start = time.time()

    # Load model once
    print("Loading GPT-2 Small model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE, dtype=torch.float32)

    for idx, cp in enumerate(CHECKPOINTS, start=2):
        release = cp["release"]
        sae_id = cp["sae_id"]
        model_name = cp["model_name"]
        family = cp["family"]
        checkpoint_label = f"{release}_{sae_id}"

        report_progress(idx, len(CHECKPOINTS) + 2, message=f"Evaluating {checkpoint_label}")
        print(f"\n=== [{idx-1}/{len(CHECKPOINTS)}] {checkpoint_label} ===")

        # Run official absorption eval
        print("Running official absorption eval...")
        config = AbsorptionEvalConfig(
            model_name=model_name,
            random_seed=SEED,
            k_sparse_probe_num_epochs=K_SPARSE_EPOCHS,
            min_GT_probe_f1=0.6,
            min_feats_for_eval=5,
        )
        if model_name == "gpt2-small":
            config.llm_batch_size = 128
            config.llm_dtype = "float32"

        output_path = str(E2_FULL_DIR / "absorption_raw")
        os.makedirs(output_path, exist_ok=True)

        abs_start = time.time()
        try:
            abs_results = run_eval(
                config,
                [(release, sae_id)],
                DEVICE,
                output_path,
                force_rerun=True,
            )
            abs_time = time.time() - abs_start

            abs_data = abs_results.get(checkpoint_label, {})
            mean_metrics = abs_data.get("eval_result_metrics", {}).get("mean", {})
            official_absorption = mean_metrics.get("mean_full_absorption_score")
            official_absorption_frac = mean_metrics.get("mean_absorption_fraction_score")
            print(f"  Official absorption (full): {official_absorption}")
            print(f"  Official absorption (fraction): {official_absorption_frac}")
        except Exception as e:
            print(f"  ERROR in absorption eval: {e}")
            official_absorption = None
            official_absorption_frac = None
            abs_time = time.time() - abs_start

        # Load SAE for dead-neuron and reconstruction metrics
        print("Loading SAE for dead-neuron metrics...")
        try:
            sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
            if isinstance(sae, tuple):
                sae = sae[0]
        except Exception as e:
            print(f"  ERROR loading SAE: {e}")
            metrics_list.append({
                "release": release,
                "sae_id": sae_id,
                "model": model_name,
                "family": family,
                "error": str(e),
            })
            continue

        print(f"Computing dead-neuron rate on {DEAD_NEURON_TOKENS} tokens...")
        recon_start = time.time()
        try:
            recon_metrics = compute_dead_neuron_and_recon(
                model, sae, sae_id, n_tokens=DEAD_NEURON_TOKENS, batch_size=32, context_size=128
            )
            recon_time = time.time() - recon_start
            print(f"  Dead neurons: {recon_metrics['dead_neuron_fraction']:.4f}")
            print(f"  L0: {recon_metrics['l0']:.2f}")
            print(f"  Explained variance: {recon_metrics['explained_variance']:.4f}")
        except Exception as e:
            print(f"  ERROR in recon metrics: {e}")
            recon_metrics = {"dead_neuron_fraction": None, "l0": None, "explained_variance": None, "n_tokens_evaluated": 0}
            recon_time = time.time() - recon_start

        metrics_list.append({
            "release": release,
            "sae_id": sae_id,
            "model": model_name,
            "family": family,
            "official_absorption_full": official_absorption,
            "official_absorption_fraction": official_absorption_frac,
            "dead_neuron_fraction": recon_metrics["dead_neuron_fraction"],
            "l0": recon_metrics["l0"],
            "explained_variance": recon_metrics["explained_variance"],
            "n_tokens_evaluated": recon_metrics["n_tokens_evaluated"],
            "absorption_eval_time_sec": abs_time,
            "recon_eval_time_sec": recon_time,
        })

        del sae
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # Save results
    results_path = E2_FULL_DIR / "absorption_results.json"
    output = {
        "task_id": TASK_ID,
        "model": "gpt2-small",
        "checkpoints": metrics_list,
        "total_time_sec": total_time,
        "timestamp": datetime.now().isoformat(),
    }
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved results to {results_path}")

    # Update gpu_progress
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    if gpu_progress_path.exists():
        gpu_progress = json.loads(gpu_progress_path.read_text())
    else:
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID in gpu_progress.get("running", {}):
        del gpu_progress["running"][TASK_ID]
    if TASK_ID not in gpu_progress.get("completed", []):
        gpu_progress.setdefault("completed", []).append(TASK_ID)
    gpu_progress["timings"][TASK_ID] = {
        "planned_min": 20,
        "actual_min": round(total_time / 60),
        "start_time": start_time_iso,
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "checkpoints": len(CHECKPOINTS),
            "k_sparse_epochs": K_SPARSE_EPOCHS,
            "dead_neuron_tokens": DEAD_NEURON_TOKENS,
            "device": DEVICE,
        },
    }
    gpu_progress_path.write_text(json.dumps(gpu_progress, indent=2))

    # Summary markdown
    summary_md = E2_FULL_DIR / "summary.md"
    ok_count = sum(1 for m in metrics_list if "error" not in m)
    summary_md.write_text(
        f"""# E2 Full GPT-2 Summary

**Task:** {TASK_ID}
**Total Time:** {total_time/60:.1f} min
**Successful:** {ok_count}/{len(CHECKPOINTS)}

## Results

| Checkpoint | Family | Absorption (Full) | L0 | Dead Neurons | Explained Var |
|---|---|---|---|---|---|
"""
        + "\n".join(
            [
                f"| {m['release']}/{m['sae_id']} | {m['family']} | {m.get('official_absorption_full', 'N/A') if m.get('official_absorption_full') is not None else 'N/A'} | {m.get('l0', 'N/A') if m.get('l0') is not None else 'N/A'} | {m.get('dead_neuron_fraction', 'N/A') if m.get('dead_neuron_fraction') is not None else 'N/A'} | {m.get('explained_variance', 'N/A') if m.get('explained_variance') is not None else 'N/A'} |"
                for m in metrics_list
            ]
        )
        + "\n\n## Family-Level Statistics\n\n"
    )

    mark_done(
        status="success",
        summary=f"E2 full GPT-2 completed on {ok_count}/{len(CHECKPOINTS)} checkpoints in {total_time/60:.1f} min",
    )
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        mark_done(status="failed", summary=str(e))
        raise
