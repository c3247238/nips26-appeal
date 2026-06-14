"""
Merge results from GPU 3 (orderings 0-1) and GPU 6 (orderings 2-5)
into the final tier1_vit_cifar100_full.json

Run after both GPU runners complete.
Usage: python merge_vit_cifar100_results.py
"""
import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

REMOTE_BASE = "/home/qhxie/sibyl_system"
PROJECT = "augmentation-order"
RESULTS_DIR = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/results")
GPU_PROGRESS_PATH = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/gpu_progress.json")
EXPERIMENT_STATE_PATH = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/experiment_state.json")

SEEDS = [42, 43, 44, 45, 46]
NUM_EPOCHS = 200

ORDERING_LABELS = {
    "order_0": "Crop->Flip->CJ",
    "order_1": "Crop->CJ->Flip",
    "order_2": "Flip->Crop->CJ",
    "order_3": "Flip->CJ->Crop",
    "order_4": "CJ->Crop->Flip",
    "order_5": "CJ->Flip->Crop",
}

ORDERINGS_OPS = {
    "order_0": ["crop", "flip", "cj"],
    "order_1": ["crop", "cj", "flip"],
    "order_2": ["flip", "crop", "cj"],
    "order_3": ["flip", "cj", "crop"],
    "order_4": ["cj", "crop", "flip"],
    "order_5": ["cj", "flip", "crop"],
}

TASK_ID = "tier1_vit_cifar100_full"


def load_partial_results(gpu_label):
    """Load partial results from GPU-specific file."""
    path = RESULTS_DIR / f"tier1_vit_cifar100_full_partial_{gpu_label}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def check_readiness():
    """Check if both GPU runners have completed."""
    gpu3_done = (RESULTS_DIR / "tier1_vit_cifar100_full_gpu3_DONE").exists()
    gpu6_done = (RESULTS_DIR / "tier1_vit_cifar100_full_gpu6_DONE").exists()
    print(f"GPU3 runner done: {gpu3_done}")
    print(f"GPU6 runner done: {gpu6_done}")
    return gpu3_done and gpu6_done


def merge_and_write():
    """Merge results from both GPU runners and write final output."""
    gpu3_results = load_partial_results("gpu3")
    gpu6_results = load_partial_results("gpu6")

    if not gpu3_results:
        print("ERROR: GPU3 results not found!")
        return False
    if not gpu6_results:
        print("ERROR: GPU6 results not found!")
        return False

    # Merge all orderings
    all_orderings = {}
    for key in ["order_0", "order_1"]:
        if key in gpu3_results.get("orderings", {}):
            all_orderings[key] = gpu3_results["orderings"][key]
        else:
            print(f"WARNING: {key} not found in GPU3 results")

    for key in ["order_2", "order_3", "order_4", "order_5"]:
        if key in gpu6_results.get("orderings", {}):
            all_orderings[key] = gpu6_results["orderings"][key]
        else:
            print(f"WARNING: {key} not found in GPU6 results")

    # Compute statistics
    all_means = []
    for ordering_key, ordering_data in all_orderings.items():
        seed_data = ordering_data.get("per_seed", {})
        accs = [seed_data[str(s)]["final_val_acc"]
                for s in SEEDS if str(s) in seed_data and "error" not in seed_data[str(s)]]
        if accs:
            ordering_data["mean_val_acc"] = round(float(np.mean(accs)), 4)
            ordering_data["std_val_acc"] = round(float(np.std(accs)), 4)
            ordering_data["n_seeds_ok"] = len(accs)
            all_means.append((ordering_key, ordering_data["mean_val_acc"]))
        else:
            ordering_data["mean_val_acc"] = 0.0
            ordering_data["std_val_acc"] = 0.0
            ordering_data["n_seeds_ok"] = 0

    valid_means = [(k, v) for k, v in all_means if v > 0.0]
    spread = round(max(v for _, v in valid_means) - min(v for _, v in valid_means), 4) if len(valid_means) >= 2 else 0.0
    best_ordering = max(all_orderings, key=lambda k: all_orderings[k]["mean_val_acc"])
    worst_ordering = min(all_orderings, key=lambda k: all_orderings[k]["mean_val_acc"])

    flip_first_orders = ["order_2", "order_3"]
    crop_first_orders = ["order_0", "order_1"]
    flip_first_mean = np.mean([all_orderings[k]["mean_val_acc"] for k in flip_first_orders if k in all_orderings])
    crop_first_mean = np.mean([all_orderings[k]["mean_val_acc"] for k in crop_first_orders if k in all_orderings])
    h4_flip_wins = bool(flip_first_mean > crop_first_mean)

    all_errors = gpu3_results.get("errors", []) + gpu6_results.get("errors", [])
    total_runs = 6 * 5
    best_acc = all_orderings[best_ordering]["mean_val_acc"]
    worst_acc = all_orderings[worst_ordering]["mean_val_acc"]
    pass_criteria_met = spread > 0.002 and best_acc > 0.55 and len(all_errors) < total_runs // 2

    summary = {
        "task_id": TASK_ID,
        "mode": "full",
        "model": "vit_small_patch4",
        "dataset": "cifar100",
        "n_classes": 100,
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 0.05,
        "timestamp": datetime.now().isoformat(),
        "spread": spread,
        "spread_pct": round(spread * 100, 2),
        "best_ordering": best_ordering,
        "worst_ordering": worst_ordering,
        "best_label": ORDERING_LABELS[best_ordering],
        "worst_label": ORDERING_LABELS[worst_ordering],
        "best_acc": best_acc,
        "worst_acc": worst_acc,
        "flip_first_mean": round(float(flip_first_mean), 4),
        "crop_first_mean": round(float(crop_first_mean), 4),
        "h4_flip_wins": h4_flip_wins,
        "pass_criteria_met": pass_criteria_met,
        "total_runs": total_runs,
        "errors": all_errors,
        "n_errors": len(all_errors),
        "orderings": all_orderings,
        "seeds": SEEDS,
        "epochs": NUM_EPOCHS,
        "note": "Results merged from GPU3 (orderings 0-1) and GPU6 (orderings 2-5) parallel runners",
    }

    # Write to full/ directory (expected by task plan)
    full_dir = RESULTS_DIR / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    out_path = full_dir / "tier1_vit_cifar100_full.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nFinal merged results written to {out_path}")
    print(f"Spread: {spread:.4f} ({spread*100:.2f}%)")
    print(f"Best:  {best_ordering} ({ORDERING_LABELS[best_ordering]}) = {best_acc:.4f}")
    print(f"Worst: {worst_ordering} ({ORDERING_LABELS[worst_ordering]}) = {worst_acc:.4f}")
    print(f"H4 Flip-first wins: {h4_flip_wins} ({flip_first_mean:.4f} vs {crop_first_mean:.4f})")
    print(f"Pass criteria met: {pass_criteria_met}")

    # Write DONE marker for the main task
    done_marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    done_marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success" if pass_criteria_met else "partial",
        "summary": f"full: spread={spread:.4f} ({spread*100:.2f}%) best={best_ordering} errors={len(all_errors)}",
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"DONE marker written to {done_marker}")

    # Update experiment_state.json
    try:
        if EXPERIMENT_STATE_PATH.exists():
            data = json.loads(EXPERIMENT_STATE_PATH.read_text())
        else:
            data = {"schema_version": 1, "tasks": {}}
        if TASK_ID not in data["tasks"]:
            data["tasks"][TASK_ID] = {}
        data["tasks"][TASK_ID]["status"] = "completed"
        data["tasks"][TASK_ID]["completed_at"] = datetime.now().isoformat()
        EXPERIMENT_STATE_PATH.write_text(json.dumps(data, indent=2))
        print("experiment_state.json updated")
    except Exception as e:
        print(f"Warning: could not update experiment_state.json: {e}")

    # Update gpu_progress.json
    try:
        if GPU_PROGRESS_PATH.exists():
            data = json.loads(GPU_PROGRESS_PATH.read_text())
        else:
            data = {"completed": [], "failed": [], "running": {}, "timings": {}}
        if TASK_ID not in data["completed"]:
            data["completed"].append(TASK_ID)
        data.get("running", {}).pop(TASK_ID, None)
        data.get("running", {}).pop("tier1_vit_cifar100_full_gpu3", None)
        data.get("running", {}).pop("tier1_vit_cifar100_full_gpu6", None)
        GPU_PROGRESS_PATH.write_text(json.dumps(data, indent=2))
        print("gpu_progress.json updated")
    except Exception as e:
        print(f"Warning: could not update gpu_progress.json: {e}")

    return True


if __name__ == "__main__":
    print("=== tier1_vit_cifar100_full merge script ===")
    ready = check_readiness()
    if not ready:
        print("\nNot all GPU runners have completed yet. Checking partial results...")
        # Check if partial results exist
        for gpu_label in ["gpu3", "gpu6"]:
            p = load_partial_results(gpu_label)
            if p:
                orderings = list(p.get("orderings", {}).keys())
                print(f"  {gpu_label}: found partial results for orderings {orderings}")
            else:
                print(f"  {gpu_label}: no partial results yet")
        print("\nPlease wait for both GPU runners to complete, then run this script again.")
        sys.exit(1)
    else:
        print("\nBoth GPU runners completed. Merging results...")
        success = merge_and_write()
        if success:
            print("\nMerge successful!")
        else:
            print("\nMerge failed. Check errors above.")
            sys.exit(1)
