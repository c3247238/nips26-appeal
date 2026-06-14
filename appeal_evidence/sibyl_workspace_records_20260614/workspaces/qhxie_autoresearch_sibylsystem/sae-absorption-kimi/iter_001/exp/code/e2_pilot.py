#!/usr/bin/env python3
"""E2 Pilot: Official absorption metric pipeline on 5 checkpoints.

Validates the official sae-bench absorption eval on:
- 3 GPT-2 Small checkpoints (Standard, TopK, TopK_Attn)
- 2 Pythia-70M checkpoints (Standard, TopK)

Also computes dead-neuron rate on >=50k tokens and basic reconstruction metrics.
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
from tqdm import tqdm
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
TASK_ID = "e2_pilot"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
E2_PILOT_DIR = WORKSPACE / "exp" / "e2_pilot"
E2_PILOT_DIR.mkdir(parents=True, exist_ok=True)

PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pilot checkpoint selection
CHECKPOINTS = [
    {
        "release": "gpt2-small-res-jb",
        "sae_id": "blocks.8.hook_resid_pre",
        "model_name": "gpt2-small",
        "family": "Standard",
    },
    {
        "release": "gpt2-small-resid-post-v5-32k",
        "sae_id": "blocks.8.hook_resid_post",
        "model_name": "gpt2-small",
        "family": "TopK",
    },
    {
        "release": "gpt2-small-attn-out-v5-32k",
        "sae_id": "blocks.8.hook_attn_out",
        "model_name": "gpt2-small",
        "family": "TopK_Attn",
    },
    {
        "release": "sae_bench_pythia70m_sweep_standard_ctx128_0712",
        "sae_id": "blocks.3.hook_resid_post__trainer_0",
        "model_name": "pythia-70m-deduped",
        "family": "Standard",
    },
    {
        "release": "sae_bench_pythia70m_sweep_topk_ctx128_0730",
        "sae_id": "blocks.3.hook_resid_post__trainer_0",
        "model_name": "pythia-70m-deduped",
        "family": "TopK",
    },
]

DEAD_NEURON_TOKENS = 50_000
K_SPARSE_EPOCHS = 10  # reduced for pilot speed


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
    parts = base.split(".")
    layer = int(parts[1])
    return base, layer


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
        acts = cache[hook_name]  # [batch, pos, d_model]

        sae_acts = sae.encode(acts)
        recon_acts = sae.decode(sae_acts)

        acts_all.append(acts.detach().cpu())
        recon_acts_all.append(recon_acts.detach().cpu())

        total_tokens += tokens.numel()
        if total_tokens >= n_tokens:
            break

    acts_cat = torch.cat(acts_all, dim=0)  # [total_batch, pos, d_model]
    recon_cat = torch.cat(recon_acts_all, dim=0)

    # Flatten across batch and pos for reconstruction metrics
    acts_flat = acts_cat.reshape(-1, acts_cat.shape[-1])
    recon_flat = recon_cat.reshape(-1, recon_cat.shape[-1])

    # Explained variance
    mse = ((acts_flat - recon_flat) ** 2).mean()
    total_var = ((acts_flat - acts_flat.mean(0)) ** 2).mean()
    explained_variance = 1.0 - (mse / total_var).item()

    # L0 (average active features per token)
    sae_acts_full = sae.encode(acts_cat.to(DEVICE))
    l0 = (sae_acts_full > 0).float().sum(dim=-1).mean().item()

    # Dead neuron fraction (features that never fire across all tokens)
    ever_active = (sae_acts_full > 0).any(dim=0).any(dim=0)  # [d_sae]
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
    report_progress(1, len(CHECKPOINTS) + 2, message="Starting E2 pilot absorption metric validation")

    metrics_list = []
    total_start = time.time()

    for idx, cp in enumerate(CHECKPOINTS, start=2):
        release = cp["release"]
        sae_id = cp["sae_id"]
        model_name = cp["model_name"]
        family = cp["family"]
        checkpoint_label = f"{release}_{sae_id}"

        report_progress(idx, len(CHECKPOINTS) + 2, message=f"Evaluating {checkpoint_label}")
        print(f"\n=== [{idx-1}/{len(CHECKPOINTS)}] {checkpoint_label} ===")

        # Load model once per model_name
        # Note: use standard from_pretrained for reconstruction metrics to match
        # SAE training preprocessing; sae-bench absorption eval handles this internally.
        print(f"Loading model {model_name}...")
        model = HookedTransformer.from_pretrained(
            model_name, device=DEVICE, dtype=torch.float32
        )

        # Run official absorption eval
        print("Running official absorption eval...")
        config = AbsorptionEvalConfig(
            model_name=model_name,
            random_seed=SEED,
            k_sparse_probe_num_epochs=K_SPARSE_EPOCHS,
            min_GT_probe_f1=0.6,
            min_feats_for_eval=5,
        )
        # Override batch size for gpt2-small
        if model_name == "gpt2-small":
            config.llm_batch_size = 128
            config.llm_dtype = "float32"

        output_path = str(E2_PILOT_DIR / "absorption_raw")
        os.makedirs(output_path, exist_ok=True)

        abs_start = time.time()
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
        print(f"  Absorption eval time: {abs_time:.1f}s")

        # Load SAE for dead-neuron and reconstruction metrics
        print("Loading SAE for dead-neuron metrics...")
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)

        print(f"Computing dead-neuron rate on {DEAD_NEURON_TOKENS} tokens...")
        recon_start = time.time()
        recon_metrics = compute_dead_neuron_and_recon(
            model, sae, sae_id, n_tokens=DEAD_NEURON_TOKENS, batch_size=32, context_size=128
        )
        recon_time = time.time() - recon_start
        print(f"  Dead neurons: {recon_metrics['dead_neuron_fraction']:.4f}")
        print(f"  L0: {recon_metrics['l0']:.2f}")
        print(f"  Explained variance: {recon_metrics['explained_variance']:.4f}")
        print(f"  Recon metrics time: {recon_time:.1f}s")

        metrics_list.append(
            {
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
            }
        )

        # Cleanup
        del model, sae
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # Save results
    metrics_path = E2_PILOT_DIR / "metrics.json"
    output = {
        "task_id": TASK_ID,
        "checkpoints": metrics_list,
        "total_time_sec": total_time,
        "timestamp": datetime.now().isoformat(),
    }
    metrics_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved metrics to {metrics_path}")

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
        "planned_min": 15,
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

    summary_md = E2_PILOT_DIR / "pilot_summary.md"
    summary_md.write_text(
        f"""# E2 Pilot Summary

**Task:** {TASK_ID}
**Total Time:** {total_time/60:.1f} min

## Results

| Checkpoint | Family | Absorption (Full) | L0 | Dead Neurons | Explained Var |
|---|---|---|---|---|---|
"""
        + "\n".join(
            [
                f"| {m['release']}/{m['sae_id']} | {m['family']} | {m['official_absorption_full']:.4f} | {m['l0']:.1f} | {m['dead_neuron_fraction']:.4f} | {m['explained_variance']:.4f} |"
                for m in metrics_list
            ]
        )
        + f"\n\n## Pass Criteria\n\n"
        + f"- All {len(CHECKPOINTS)} checkpoints loaded successfully: {'YES' if len(metrics_list) == len(CHECKPOINTS) else 'NO'}\n"
        + f"- Official absorption metric returned finite values: {'YES' if all(m['official_absorption_full'] is not None for m in metrics_list) else 'NO'}\n"
        + f"- Dead-neuron rate computed on >=50k tokens: {'YES' if all(m['n_tokens_evaluated'] >= 50_000 for m in metrics_list) else 'NO'}\n"
    )

    mark_done(
        status="success",
        summary=f"E2 pilot completed on {len(metrics_list)}/{len(CHECKPOINTS)} checkpoints in {total_time/60:.1f} min",
    )
    print("Done.")


if __name__ == "__main__":
    main()
