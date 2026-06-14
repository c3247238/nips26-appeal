#!/usr/bin/env python3
"""Robustness Analysis: Varying tau_fs (feature_split_f1_jump_threshold).

Re-runs the official SAEBench first-letter absorption evaluator at tau_fs = 0.01
and tau_fs = 0.05 for all 8 SAEs, then computes Pearson correlation between
first-letter and semantic-hierarchy absorption at each threshold (including
the baseline tau_fs = 0.03 from firstletter_pythia).
"""

import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.utils import resample
from transformer_lens import HookedTransformer

import sae_bench.custom_saes.run_all_evals_dictionary_learning_saes as dl_saes
from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.absorption.main import run_eval

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_ID = "tau_fs_robustness"
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

TAU_FS_VALUES = [0.01, 0.05]

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


def bootstrap_pearson_r(x, y, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute Pearson r with bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(x)
    observed_r, _ = stats.pearsonr(x, y)
    bootstrapped = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            r, _ = stats.pearsonr(x[idx], y[idx])
            bootstrapped.append(r)
        else:
            bootstrapped.append(observed_r)
    bootstrapped = np.array(bootstrapped)
    alpha = 1 - ci
    lower = np.percentile(bootstrapped, alpha / 2 * 100)
    upper = np.percentile(bootstrapped, (1 - alpha / 2) * 100)
    return observed_r, float(lower), float(upper)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    total_steps = len(CHECKPOINTS) * len(TAU_FS_VALUES) + 3
    report_progress(1, total_steps, message="Starting tau_fs robustness analysis")

    total_start = time.time()

    print(f"Loading model {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)
    print("Model loaded.")

    # Run first-letter absorption at tau_fs = 0.01 and 0.05
    tau_fs_results = {}
    step = 2

    for tau_fs in TAU_FS_VALUES:
        tau_fs_results[tau_fs] = []
        print(f"\n=== tau_fs = {tau_fs} ===")
        for idx, cp in enumerate(CHECKPOINTS, start=1):
            location = cp["location"]
            family = cp["family"]
            trainer = cp["trainer"]
            is_random = cp["is_random"]
            sae_label = f"{family}_{trainer}"
            if is_random:
                sae_label = "Random_control"

            report_progress(step, total_steps, message=f"tau_fs={tau_fs}, evaluating {sae_label}")
            print(f"\n  [{idx}/{len(CHECKPOINTS)}] {sae_label}")

            sae = dl_saes.load_dictionary_learning_sae(
                repo_id=REPO_ID,
                location=location,
                model_name=MODEL_NAME,
                device=DEVICE,
                dtype=torch.float32,
                download_location=DOWNLOAD_LOCATION,
            )

            if is_random:
                print("  Permuting decoder directions for random control...")
                sae = make_random_sae(sae)

            config = AbsorptionEvalConfig(
                model_name=MODEL_NAME,
                random_seed=SEED,
                llm_batch_size=256,
                llm_dtype="float32",
                f1_jump_threshold=tau_fs,
            )

            tau_fs_raw_dir = RAW_DIR / f"tau_fs_{tau_fs}"
            tau_fs_raw_dir.mkdir(parents=True, exist_ok=True)

            abs_start = time.time()
            abs_results = run_eval(
                config,
                [(sae_label, sae)],
                DEVICE,
                str(tau_fs_raw_dir),
                force_rerun=True,
            )
            abs_time = time.time() - abs_start

            abs_data = abs_results.get(f"{sae_label}_custom_sae", {})
            mean_metrics = abs_data.get("eval_result_metrics", {}).get("mean", {})
            official_full = mean_metrics.get("mean_full_absorption_score")
            official_frac = mean_metrics.get("mean_absorption_fraction_score")

            print(f"    full: {official_full}, fraction: {official_frac}, time: {abs_time:.1f}s")

            tau_fs_results[tau_fs].append({
                "family": family,
                "trainer": trainer,
                "is_random": is_random,
                "official_absorption_full": official_full,
                "official_absorption_fraction": official_frac,
                "absorption_eval_time_sec": abs_time,
            })

            del sae
            gc.collect()
            torch.cuda.empty_cache()
            step += 1

    # Load baseline first-letter results (tau_fs = 0.03)
    report_progress(step, total_steps, message="Loading baseline and custom results")
    step += 1

    baseline_path = RESULTS_DIR / "firstletter_pythia_results.json"
    baseline_data = json.loads(baseline_path.read_text())
    baseline_results = baseline_data["checkpoints"]

    # Load semantic-hierarchy results
    semantic_path = RESULTS_DIR / "semantic_hierarchy_pythia_results.json"
    semantic_data = json.loads(semantic_path.read_text())
    semantic_by_family = {}
    for r in semantic_data["sae_results"]:
        semantic_by_family[r["family"]] = r["mean_hierarchy_absorption"]

    # Load non-hierarchy control results
    nonhierarchy_path = RESULTS_DIR / "nonhierarchy_control_pythia_results.json"
    nonhierarchy_data = json.loads(nonhierarchy_path.read_text())
    nonhierarchy_by_family = {}
    for r in nonhierarchy_data["sae_results"]:
        nonhierarchy_by_family[r["family"]] = r["mean_pair_absorption"]

    # Compute correlations
    report_progress(step, total_steps, message="Computing correlations")
    step += 1

    correlation_results = []

    all_tau_fs = [0.03] + TAU_FS_VALUES
    all_results = {0.03: baseline_results}
    all_results.update(tau_fs_results)

    for tau_fs in all_tau_fs:
        results = all_results[tau_fs]
        # Exclude random control from correlation (as in methodology)
        families = []
        first_letter = []
        semantic = []
        nonhierarchy = []
        for r in results:
            if r["is_random"]:
                continue
            families.append(r["family"])
            first_letter.append(r["official_absorption_full"])
            semantic.append(semantic_by_family[r["family"]])
            nonhierarchy.append(nonhierarchy_by_family[r["family"]])

        fl_arr = np.array(first_letter)
        sem_arr = np.array(semantic)
        nh_arr = np.array(nonhierarchy)

        # Pearson r: first-letter vs semantic-hierarchy
        r_fl_sem, ci_lower_fl_sem, ci_upper_fl_sem = bootstrap_pearson_r(fl_arr, sem_arr)
        # Pearson r: first-letter vs non-hierarchy
        r_fl_nh, ci_lower_fl_nh, ci_upper_fl_nh = bootstrap_pearson_r(fl_arr, nh_arr)
        # Paired t-test: semantic vs non-hierarchy
        t_stat, t_pvalue = stats.ttest_rel(sem_arr, nh_arr)

        correlation_results.append({
            "tau_fs": tau_fs,
            "families": families,
            "first_letter_absorption": first_letter,
            "semantic_hierarchy_absorption": semantic,
            "nonhierarchy_control_absorption": nonhierarchy,
            "pearson_r_firstletter_semantic": float(r_fl_sem),
            "bootstrap_ci_lower_firstletter_semantic": float(ci_lower_fl_sem),
            "bootstrap_ci_upper_firstletter_semantic": float(ci_upper_fl_sem),
            "pearson_r_firstletter_nonhierarchy": float(r_fl_nh),
            "bootstrap_ci_lower_firstletter_nonhierarchy": float(ci_lower_fl_nh),
            "bootstrap_ci_upper_firstletter_nonhierarchy": float(ci_upper_fl_nh),
            "paired_ttest_semantic_vs_nonhierarchy_t": float(t_stat),
            "paired_ttest_semantic_vs_nonhierarchy_p": float(t_pvalue),
        })

        print(f"\ntau_fs={tau_fs}:")
        print(f"  r(first-letter, semantic) = {r_fl_sem:.3f} [{ci_lower_fl_sem:.3f}, {ci_upper_fl_sem:.3f}]")
        print(f"  r(first-letter, non-hierarchy) = {r_fl_nh:.3f} [{ci_lower_fl_nh:.3f}, {ci_upper_fl_nh:.3f}]")
        print(f"  paired t-test (semantic vs non-hierarchy): t={t_stat:.3f}, p={t_pvalue:.4f}")

    total_time = time.time() - total_start

    # Save results
    output = {
        "task_id": TASK_ID,
        "tau_fs_results": tau_fs_results,
        "correlation_analysis": correlation_results,
        "total_time_sec": total_time,
        "timestamp": datetime.now().isoformat(),
    }
    results_path = RESULTS_DIR / "tau_fs_robustness_results.json"
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
        "planned_min": 30,
        "actual_min": round(total_time / 60),
        "start_time": start_time_iso,
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "checkpoints": len(CHECKPOINTS),
            "tau_fs_values": all_tau_fs,
            "model": MODEL_NAME,
            "device": DEVICE,
            "gpu_model": "local",
            "gpu_count": 1,
        },
    }
    gpu_progress_path.write_text(json.dumps(gpu_progress, indent=2))

    # Summary markdown
    summary_md = RESULTS_DIR / "tau_fs_robustness_summary.md"
    md_lines = [
        f"# tau_fs Robustness Analysis Summary",
        f"",
        f"**Task:** {TASK_ID}",
        f"**Total Time:** {total_time/60:.1f} min",
        f"",
        "## Correlation across tau_fs values",
        "",
        "| tau_fs | Pearson r (FL vs Semantic) | 95% CI Lower | 95% CI Upper | Pearson r (FL vs Non-Hierarchy) | 95% CI Lower | 95% CI Upper |",
        "|---|---|---|---|---|---|---|",
    ]
    for cr in correlation_results:
        md_lines.append(
            f"| {cr['tau_fs']} | {cr['pearson_r_firstletter_semantic']:.3f} | "
            f"{cr['bootstrap_ci_lower_firstletter_semantic']:.3f} | "
            f"{cr['bootstrap_ci_upper_firstletter_semantic']:.3f} | "
            f"{cr['pearson_r_firstletter_nonhierarchy']:.3f} | "
            f"{cr['bootstrap_ci_lower_firstletter_nonhierarchy']:.3f} | "
            f"{cr['bootstrap_ci_upper_firstletter_nonhierarchy']:.3f} |"
        )
    md_lines.append("")
    md_lines.append("## Paired t-test: Semantic-Hierarchy vs Non-Hierarchy Control")
    md_lines.append("")
    md_lines.append("| tau_fs | t-statistic | p-value |")
    md_lines.append("|---|---|---|")
    for cr in correlation_results:
        md_lines.append(
            f"| {cr['tau_fs']} | {cr['paired_ttest_semantic_vs_nonhierarchy_t']:.3f} | "
            f"{cr['paired_ttest_semantic_vs_nonhierarchy_p']:.4f} |"
        )
    md_lines.append("")
    summary_md.write_text("\n".join(md_lines))

    mark_done(
        status="success",
        summary=f"tau_fs_robustness completed in {total_time/60:.1f} min",
    )
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        mark_done(status="failure", summary=str(e))
        raise
