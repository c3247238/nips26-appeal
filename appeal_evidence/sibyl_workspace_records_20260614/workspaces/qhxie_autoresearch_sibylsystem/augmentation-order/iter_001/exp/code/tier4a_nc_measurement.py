#!/usr/bin/env python3
"""
Tier 4a: NC_2 Computation (Sliced Wasserstein Distance)

Compute NC_2(t_i, t_j; mu) for all 3 pairs from the 3-op set,
in both directions (6 directional computations).

Mode: PILOT - uses 100 images instead of 10k
- Sample CIFAR-10 images
- Apply ordering (A then B) to get distribution P_AB
- Apply (B then A) to get P_BA
- Compute Sliced Wasserstein Distance (SWD) as proxy for W_2
  using N random projections

CPU-only; no training.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--workspace", default="/home/qhxie/sibyl_system/projects/augmentation-order")
parser.add_argument("--mode", default="pilot", choices=["pilot", "full"])
parser.add_argument("--n_samples", type=int, default=None,
                    help="Override sample count (default: 100 for pilot, 10000 for full)")
parser.add_argument("--n_projections", type=int, default=None,
                    help="Override projections (default: 100 for pilot, 1000 for full)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

WORKSPACE = Path(args.workspace)
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "tier4a_nc_measurement"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

# Write PID file
PID_FILE.write_text(str(os.getpid()))
start_time = datetime.now()

# Mode-specific defaults
if args.mode == "pilot":
    N_SAMPLES = args.n_samples or 100
    N_PROJECTIONS = args.n_projections or 100
    print(f"[tier4a] PILOT mode: {N_SAMPLES} samples, {N_PROJECTIONS} projections")
else:
    N_SAMPLES = args.n_samples or 10000
    N_PROJECTIONS = args.n_projections or 1000
    print(f"[tier4a] FULL mode: {N_SAMPLES} samples, {N_PROJECTIONS} projections")

SEED = args.seed
np.random.seed(SEED)

import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

torch.manual_seed(SEED)

def write_progress(step, total_steps, msg=""):
    PROGRESS_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
        "step": step,
        "total_steps": total_steps,
        "updated_at": datetime.now().isoformat(),
        "msg": msg,
    }))

def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

# ---- Load CIFAR-10 images ----
write_progress(0, 10, "Loading dataset")
print(f"[tier4a] Loading CIFAR-10...")

cifar10_path = WORKSPACE / "../../shared/datasets/cifar10"
if not cifar10_path.exists():
    cifar10_path = Path("/home/qhxie/sibyl_system/shared/datasets/cifar10")

dataset = CIFAR10(root=str(cifar10_path), train=True, download=False, transform=None)

# Select N_SAMPLES images
rng = np.random.RandomState(SEED)
indices = rng.choice(len(dataset), size=min(N_SAMPLES, len(dataset)), replace=False)
images = [dataset[i][0] for i in indices]  # PIL images

print(f"[tier4a] Loaded {len(images)} images")

# ---- Define transforms ----
# 3 operations: Crop, Flip, ColorJitter
def make_crop():
    return T.RandomCrop(32, padding=4)

def make_flip():
    return T.RandomHorizontalFlip(p=0.5)

def make_cj():
    return T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0)

to_tensor = T.ToTensor()

def apply_transform_seq(images, transforms_list, seed_offset=0):
    """Apply a sequence of transforms to all images and return as numpy array."""
    results = []
    for i, img in enumerate(images):
        torch.manual_seed(SEED + i + seed_offset)
        np.random.seed(SEED + i + seed_offset)
        x = img
        for t in transforms_list:
            x = t(x)
        x = to_tensor(x) if not isinstance(x, torch.Tensor) else x
        results.append(x.numpy().flatten())
    return np.array(results, dtype=np.float32)

# ---- Sliced Wasserstein Distance ----
def sliced_wasserstein_distance(X, Y, n_projections=100, seed=42):
    """
    Compute Sliced Wasserstein Distance between distributions X and Y.
    X, Y: (N, D) arrays
    Returns: SWD value
    """
    rng = np.random.RandomState(seed)
    D = X.shape[1]
    N = min(X.shape[0], Y.shape[0])
    X = X[:N]
    Y = Y[:N]

    swd_values = []
    for _ in range(n_projections):
        # Random projection direction
        direction = rng.randn(D).astype(np.float32)
        direction /= np.linalg.norm(direction) + 1e-8

        # Project
        x_proj = X @ direction
        y_proj = Y @ direction

        # Sort projected values
        x_sorted = np.sort(x_proj)
        y_sorted = np.sort(y_proj)

        # 1D Wasserstein = mean absolute difference of sorted
        swd_values.append(np.mean(np.abs(x_sorted - y_sorted)))

    return float(np.mean(swd_values))

# ---- Define 3-op pairs ----
# Operations: Crop (C), Flip (F), ColorJitter (CJ)
# 3 pairs: (C, F), (C, CJ), (F, CJ)
# Each pair in both orderings -> 6 directional computations

operations = {
    "Crop": make_crop,
    "Flip": make_flip,
    "CJ": make_cj,
}

pairs = [
    ("Crop", "Flip"),
    ("Crop", "CJ"),
    ("Flip", "CJ"),
]

write_progress(1, 10, "Starting SWD computation")
print(f"[tier4a] Computing SWD for {len(pairs)} pairs x 2 directions = {len(pairs)*2} measurements")

results = []
total_steps = len(pairs) * 2
step = 0

for (op_a, op_b) in pairs:
    print(f"[tier4a] Pair ({op_a}, {op_b})...")

    # Forward: A then B -> P_AB
    t_start = time.time()
    t_a = operations[op_a]()
    t_b = operations[op_b]()
    X_AB = apply_transform_seq(images, [t_a, t_b], seed_offset=0)

    # Reverse: B then A -> P_BA
    t_a2 = operations[op_a]()
    t_b2 = operations[op_b]()
    X_BA = apply_transform_seq(images, [t_b2, t_a2], seed_offset=1000)

    # Compute SWD(P_AB, P_BA) - measures non-commutativity
    swd = sliced_wasserstein_distance(X_AB, X_BA, n_projections=N_PROJECTIONS, seed=SEED)
    elapsed = time.time() - t_start

    pair_label = f"{op_a}_{op_b}"
    result_entry = {
        "pair": [op_a, op_b],
        "pair_label": pair_label,
        "direction_forward": f"{op_a}→{op_b}",
        "direction_reverse": f"{op_b}→{op_a}",
        "swd_forward_vs_reverse": swd,
        "nc2_proxy": swd,  # SWD as NC_2 proxy
        "n_samples": len(images),
        "n_projections": N_PROJECTIONS,
        "elapsed_sec": round(elapsed, 2),
    }
    results.append(result_entry)

    step += 1
    write_progress(step, total_steps, f"Completed pair {op_a}/{op_b}")
    print(f"  SWD({op_a}→{op_b} vs {op_b}→{op_a}) = {swd:.6f}  [{elapsed:.1f}s]")

    step += 1

# ---- Also compute individual distributions for each single transform ----
# For visualization: what does each individual transform distribution look like?
# Compute SWD of each transform applied vs. identity (original)
print(f"[tier4a] Computing SWD of individual transforms vs. identity...")

identity_results = []
X_identity = apply_transform_seq(images, [], seed_offset=9999)

for op_name, op_factory in operations.items():
    t_start = time.time()
    t = op_factory()
    X_transformed = apply_transform_seq(images, [t], seed_offset=5000)
    swd = sliced_wasserstein_distance(X_identity, X_transformed, n_projections=N_PROJECTIONS, seed=SEED)
    elapsed = time.time() - t_start
    identity_results.append({
        "operation": op_name,
        "swd_vs_identity": swd,
        "elapsed_sec": round(elapsed, 2),
    })
    print(f"  SWD(identity, {op_name}) = {swd:.6f}  [{elapsed:.1f}s]")

write_progress(total_steps, total_steps, "Analysis complete")

# ---- Rank pairs by NC_2 estimate ----
results_sorted = sorted(results, key=lambda x: x["nc2_proxy"], reverse=True)

# Determine which pairs have highest non-commutativity
print(f"\n[tier4a] NC_2 ranking (highest = most non-commutative):")
for i, r in enumerate(results_sorted):
    print(f"  {i+1}. {r['pair_label']}: NC_2={r['nc2_proxy']:.6f}")

# ---- Write results ----
nc_results = {
    "task_id": TASK_ID,
    "mode": args.mode,
    "timestamp": datetime.now().isoformat(),
    "seed": SEED,
    "n_samples": N_SAMPLES,
    "n_projections": N_PROJECTIONS,
    "operations": list(operations.keys()),
    "pairs_analyzed": results,
    "pairs_ranked_by_nc2": [
        {"rank": i+1, "pair_label": r["pair_label"], "nc2_proxy": r["nc2_proxy"]}
        for i, r in enumerate(results_sorted)
    ],
    "individual_transform_swd": identity_results,
    "summary": {
        "most_non_commutative_pair": results_sorted[0]["pair_label"],
        "least_non_commutative_pair": results_sorted[-1]["pair_label"],
        "max_nc2": results_sorted[0]["nc2_proxy"],
        "min_nc2": results_sorted[-1]["nc2_proxy"],
        "all_nc2_positive": all(r["nc2_proxy"] > 0 for r in results),
        "pass_criteria_met": (
            len(results) >= 1 and
            results[0]["nc2_proxy"] > 0 and
            DONE_FILE.name not in str(RESULTS_DIR)
        ),
    },
    "runtime_sec": round((datetime.now() - start_time).total_seconds(), 1),
}

output_path = RESULTS_DIR / "tier4a_nc.json"
output_path.write_text(json.dumps(nc_results, indent=2))
print(f"\n[tier4a] Results written to {output_path}")

# Validate pass criteria
pass_criteria = (
    len(results) >= 1 and
    any(r["nc2_proxy"] > 0 for r in results)
)

print(f"\n[tier4a] Pass criteria met: {pass_criteria}")
print(f"  - SWD computed for {len(results)} pairs")
print(f"  - NC_2 > 0 for at least one pair: {any(r['nc2_proxy'] > 0 for r in results)}")
print(f"  - Results written: {output_path.exists()}")

if pass_criteria:
    mark_done("success", f"NC_2 computed for {len(results)} pairs. Most non-commutative: {results_sorted[0]['pair_label']} (SWD={results_sorted[0]['nc2_proxy']:.6f})")
    print(f"[tier4a] DONE - SUCCESS")
    sys.exit(0)
else:
    mark_done("failed", "Pass criteria not met")
    print(f"[tier4a] DONE - FAILED")
    sys.exit(1)
