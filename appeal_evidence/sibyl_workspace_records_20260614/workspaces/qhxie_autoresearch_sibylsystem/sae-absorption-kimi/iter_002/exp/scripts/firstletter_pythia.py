#!/usr/bin/env python3
"""First-Letter Absorption on Pythia-160M SAEs.

Runs the official SAEBench absorption evaluator on 8 SAE configurations:
7 families (trainer_0 each) + 1 random-SAE control (permuted Standard trainer_0).
Records mean_full_absorption_score and mean_absorption_fraction_score.
"""

import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

import sae_bench.custom_saes.run_all_evals_dictionary_learning_saes as dl_saes
from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.absorption.main import run_eval

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_ID = "firstletter_pythia"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = WORKSPACE / "exp" / "absorption_raw" / TASK_ID
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
DOWNLOAD_LOCATION = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/downloaded_saes"

# 7 families + random control = 8 checkpoints
FAMILIES = [
    "BatchTopK", "GatedSAE", "JumpRelu", "MatryoshkaBatchTopK",
    "PAnneal", "Standard", "TopK",
]

CHECKPOINTS = []
for family in FAMILIES:
    location = f"{family}_pythia-160m-deduped__0108/resid_post_layer_8/trainer_0"
    CHECKPOINTS.append({
        "location": location,
        "family": family,
        "trainer": "trainer_0",
        "is_random": False,
    })

# Random-SAE control: permuted Standard trainer_0
CHECKPOINTS.append({
    "location": "Standard_pythia-160m-deduped__0108/resid_post_layer_8/trainer_0",
    "family": "Random",
    "trainer": "trainer_0",
    "is_random": True,
})


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


def make_random_sae(base_sae):
    """Create a random-SAE control by permuting decoder directions."""
    import copy
    random_sae = copy.deepcopy(base_sae)
    with torch.no_grad():
        d_sae = random_sae.W_dec.shape[0]
        perm = torch.randperm(d_sae)
        random_sae.W_dec = torch.nn.Parameter(random_sae.W_dec[perm])
        random_sae.W_enc = torch.nn.Parameter(random_sae.W_enc[:, perm])
        random_sae.b_enc = torch.nn.Parameter(random_sae.b_enc[perm])
    return random_sae


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    report_progress(1, len(CHECKPOINTS) + 2, message="Starting firstletter_pythia absorption evaluation")

    metrics_list = []
    total_start = time.time()

    print(f"Loading model {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)

    for idx, cp in enumerate(CHECKPOINTS, start=3):
        location = cp["location"]
        family = cp["family"]
        trainer = cp["trainer"]
        is_random = cp["is_random"]
        sae_label = f"{family}_{trainer}"
        if is_random:
            sae_label = "Random_control"

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
            download_location=DOWNLOAD_LOCATION,
        )

        if is_random:
            print("Permuting decoder directions for random control...")
            sae = make_random_sae(sae)

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

        metrics_list.append(
            {
                "family": family,
                "trainer": trainer,
                "is_random": is_random,
                "official_absorption_full": official_absorption,
                "official_absorption_fraction": official_absorption_frac,
                "absorption_eval_time_sec": abs_time,
            }
        )

        # Cleanup
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # Save results
    output = {
        "task_id": TASK_ID,
        "checkpoints": metrics_list,
        "total_time_sec": total_time,
        "timestamp": datetime.now().isoformat(),
    }
    results_path = RESULTS_DIR / "firstletter_pythia_results.json"
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved metrics to {results_path}")

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
        "planned_min": 45,
        "actual_min": round(total_time / 60),
        "start_time": start_time_iso,
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "checkpoints": len(CHECKPOINTS),
            "repo_id": REPO_ID,
            "model": MODEL_NAME,
            "device": DEVICE,
            "gpu_model": "local",
            "gpu_count": 1,
        },
    }
    gpu_progress_path.write_text(json.dumps(gpu_progress, indent=2))

    # Summary markdown
    summary_md = RESULTS_DIR / "firstletter_pythia_summary.md"
    summary_md.write_text(
        f"""# First-Letter Absorption (Pythia-160M) Summary

**Task:** {TASK_ID}
**Total Time:** {total_time/60:.1f} min

## Results

| Architecture | Absorption (Full) | Absorption (Fraction) |
|---|---|---|
"""
        + "\n".join(
            [
                f"| {m['family']} | {m['official_absorption_full']:.4f} | {m['official_absorption_fraction']:.4f} |"
                if m['official_absorption_full'] is not None
                else f"| {m['family']} | N/A | N/A |"
                for m in metrics_list
            ]
        )
        + "\n"
    )

    mark_done(
        status="success",
        summary=f"firstletter_pythia completed on {len(metrics_list)}/{len(CHECKPOINTS)} checkpoints in {total_time/60:.1f} min",
    )
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        mark_done(status="failure", summary=str(e))
        raise
