#!/usr/bin/env python3
"""E2 Full Pythia: Official absorption metric on 14 Pythia-160M SAEBench checkpoints.

Evaluates 2 checkpoints per architecture family across 7 families:
BatchTopK, GatedSAE, JumpRelu, MatryoshkaBatchTopK, PAnneal, Standard, TopK.
Computes official absorption metric and dead-neuron rate on >=50k tokens.
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
from tqdm import tqdm
from transformer_lens import HookedTransformer

import sae_bench.custom_saes.run_all_evals_dictionary_learning_saes as dl_saes
from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.absorption.main import run_eval
from sae_bench.sae_bench_utils import general_utils

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_ID = "e2_full_pythia"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR = WORKSPACE / "exp" / "e2_full_pythia"
EXP_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = EXP_DIR / "absorption_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ID = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"

# 2 trainers per family = 14 checkpoints
FAMILIES = [
    "BatchTopK", "GatedSAE", "JumpRelu", "MatryoshkaBatchTopK",
    "PAnneal", "Standard", "TopK",
]
TRAINERS = ["trainer_0", "trainer_3"]

CHECKPOINTS = []
for family in FAMILIES:
    for trainer in TRAINERS:
        location = f"{family}_pythia-160m-deduped__0108/resid_post_layer_8/{trainer}"
        CHECKPOINTS.append({
            "location": location,
            "family": family,
            "trainer": trainer,
        })

DEAD_NEURON_TOKENS = 50_000
BATCH_SIZE = 32
CONTEXT_SIZE = 128


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
@torch.no_grad()
def compute_dead_neuron_and_recon(model, sae, n_tokens=50_000, batch_size=32, context_size=128):
    """Compute dead-neuron fraction, L0, explained variance on C4 validation."""
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    ds = ds.shuffle(seed=SEED)

    hook_name = sae.cfg.hook_name
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
    report_progress(1, len(CHECKPOINTS) + 2, message="Starting E2 full Pythia absorption evaluation")

    # Pre-download configs so load_dictionary_learning_sae works
    report_progress(2, len(CHECKPOINTS) + 2, message="Downloading SAE configs")
    dl_saes.get_all_hf_repo_autoencoders(REPO_ID, download_location="downloaded_saes")

    metrics_list = []
    total_start = time.time()

    # Load model once for dead-neuron metrics
    print(f"Loading model {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)

    for idx, cp in enumerate(CHECKPOINTS, start=3):
        location = cp["location"]
        family = cp["family"]
        trainer = cp["trainer"]
        sae_label = f"{family}_{trainer}"

        report_progress(idx, len(CHECKPOINTS) + 2, message=f"Evaluating {sae_label}")
        print(f"\n=== [{idx-2}/{len(CHECKPOINTS)}] {sae_label} ===")

        # Load custom SAE
        print(f"Loading SAE {location}...")
        sae = dl_saes.load_dictionary_learning_sae(
            repo_id=REPO_ID,
            location=location,
            model_name=MODEL_NAME,
            device=DEVICE,
            dtype=torch.float32,
        )

        # Run official absorption eval
        print("Running official absorption eval...")
        config = AbsorptionEvalConfig(
            model_name=MODEL_NAME,
            random_seed=SEED,
            llm_batch_size=256,
            llm_dtype="float32",
        )

        abs_start = time.time()
        abs_results = run_eval(
            config,
            [(sae_label, sae)],
            DEVICE,
            str(RAW_DIR),
            force_rerun=True,
        )
        abs_time = time.time() - abs_start

        abs_data = abs_results.get(f"{sae_label}_custom_sae", {})
        mean_metrics = abs_data.get("eval_result_metrics", {}).get("mean", {})
        official_absorption = mean_metrics.get("mean_full_absorption_score")
        official_absorption_frac = mean_metrics.get("mean_absorption_fraction_score")

        print(f"  Official absorption (full): {official_absorption}")
        print(f"  Official absorption (fraction): {official_absorption_frac}")
        print(f"  Absorption eval time: {abs_time:.1f}s")

        # Compute dead-neuron and reconstruction metrics
        print(f"Computing dead-neuron rate on {DEAD_NEURON_TOKENS} tokens...")
        recon_start = time.time()
        recon_metrics = compute_dead_neuron_and_recon(
            model, sae, n_tokens=DEAD_NEURON_TOKENS, batch_size=BATCH_SIZE, context_size=CONTEXT_SIZE
        )
        recon_time = time.time() - recon_start
        print(f"  Dead neurons: {recon_metrics['dead_neuron_fraction']:.4f}")
        print(f"  L0: {recon_metrics['l0']:.2f}")
        print(f"  Explained variance: {recon_metrics['explained_variance']:.4f}")
        print(f"  Recon metrics time: {recon_time:.1f}s")

        metrics_list.append(
            {
                "repo_id": REPO_ID,
                "location": location,
                "family": family,
                "trainer": trainer,
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
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # Save results
    metrics_path = EXP_DIR / "absorption_results.json"
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
        "planned_min": 30,
        "actual_min": round(total_time / 60),
        "start_time": start_time_iso,
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "checkpoints": len(CHECKPOINTS),
            "repo_id": REPO_ID,
            "model": MODEL_NAME,
            "dead_neuron_tokens": DEAD_NEURON_TOKENS,
            "device": DEVICE,
        },
    }
    gpu_progress_path.write_text(json.dumps(gpu_progress, indent=2))

    # Summary markdown
    summary_md = EXP_DIR / "summary.md"
    summary_md.write_text(
        f"""# E2 Full Pythia Summary

**Task:** {TASK_ID}
**Total Time:** {total_time/60:.1f} min

## Results

| Family | Trainer | Absorption (Full) | Absorption (Fraction) | L0 | Dead Neurons | Explained Var |
|---|---|---|---|---|---|---|
"""
        + "\n".join(
            [
                f"| {m['family']} | {m['trainer']} | {m['official_absorption_full']:.4f} | {m['official_absorption_fraction']:.4f} | {m['l0']:.1f} | {m['dead_neuron_fraction']:.4f} | {m['explained_variance']:.4f} |"
                for m in metrics_list
            ]
        )
        + f"\n\n## Family Averages\n\n"
    )
    # Append family averages
    from collections import defaultdict
    fam_stats = defaultdict(lambda: {"abs": [], "abs_frac": [], "l0": [], "dead": [], "ev": []})
    for m in metrics_list:
        fam_stats[m["family"]]["abs"].append(m["official_absorption_full"] or 0.0)
        fam_stats[m["family"]]["abs_frac"].append(m["official_absorption_fraction"] or 0.0)
        fam_stats[m["family"]]["l0"].append(m["l0"])
        fam_stats[m["family"]]["dead"].append(m["dead_neuron_fraction"])
        fam_stats[m["family"]]["ev"].append(m["explained_variance"])

    avg_lines = []
    for fam, stats in sorted(fam_stats.items()):
        avg_lines.append(
            f"- **{fam}**: Abs={np.mean(stats['abs']):.4f} (±{np.std(stats['abs']):.4f}), "
            f"AbsFrac={np.mean(stats['abs_frac']):.4f} (±{np.std(stats['abs_frac']):.4f}), "
            f"L0={np.mean(stats['l0']):.1f}, Dead={np.mean(stats['dead']):.4f}, EV={np.mean(stats['ev']):.4f}"
        )

    content = summary_md.read_text() + "\n".join(avg_lines) + "\n"
    summary_md.write_text(content)

    mark_done(
        status="success",
        summary=f"E2 full Pythia completed on {len(metrics_list)}/{len(CHECKPOINTS)} checkpoints in {total_time/60:.1f} min",
    )
    print("Done.")


if __name__ == "__main__":
    main()
