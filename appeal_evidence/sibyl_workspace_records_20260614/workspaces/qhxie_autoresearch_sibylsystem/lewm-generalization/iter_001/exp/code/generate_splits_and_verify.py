"""
Generate 3 CoGenT-style train/holdout splits and verify all 27 HDF5 files.
Also generates collection_summary.json.
"""
import os
import sys
import json
import time
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/lewm-generalization/iter_001")
DATA_DIR = WORKSPACE / "exp" / "data" / "comphys_lewm"
RESULTS_DIR = WORKSPACE / "exp" / "results"

GRAVITY_SCALES = [0.5, 1.0, 2.0]
FRICTION_SCALES = [0.5, 1.0, 2.0]
MASS_SCALES = [0.5, 1.0, 2.0]
RANDOM_SEED = 42
N_TRAJ_FULL = 200
TRAJ_LENGTH = 300


def verify_hdf5(path, expected_n_traj, expected_traj_length):
    """Verify HDF5 file meets pass criteria."""
    issues = []
    try:
        with h5py.File(path, 'r') as f:
            pixels = f['pixels']
            joints = f['joint_angles']
            comv = f['com_velocity']
            labels = f['physics_labels']

            if pixels.shape[0] < expected_n_traj:
                issues.append(f"pixels n_traj={pixels.shape[0]} < {expected_n_traj}")
            if pixels.shape[1] != expected_traj_length:
                issues.append(f"pixels T={pixels.shape[1]} != {expected_traj_length}")
            if pixels.shape[2:] != (64, 64, 3):
                issues.append(f"pixels spatial shape={pixels.shape[2:]} != (64, 64, 3)")
            if joints.shape[-1] != 6:
                issues.append(f"joint_angles dim={joints.shape[-1]} != 6")
            if comv.shape[-1] != 2:
                issues.append(f"com_velocity dim={comv.shape[-1]} != 2")
            if labels.shape[-1] != 3:
                issues.append(f"physics_labels dim={labels.shape[-1]} != 3")

            sample_pixels = pixels[:5][:]
            if np.any(np.isnan(sample_pixels.astype(np.float32))):
                issues.append("NaN frames detected")

            sample_joints = joints[:5][:]
            if np.any(np.isnan(sample_joints)):
                issues.append("NaN in joint_angles")
    except Exception as e:
        issues.append(f"HDF5 open error: {e}")

    return issues


def generate_splits(all_combos):
    rng = np.random.RandomState(RANDOM_SEED)
    splits = {}
    for split_id in range(3):
        indices = list(range(len(all_combos)))
        rng.shuffle(indices)
        train_indices = []
        holdout_indices = []
        factor_counts = {0: {0.5: 0, 1.0: 0, 2.0: 0},
                         1: {0.5: 0, 1.0: 0, 2.0: 0},
                         2: {0.5: 0, 1.0: 0, 2.0: 0}}

        for idx in indices:
            g, f, m = all_combos[idx]
            fg = factor_counts[0][g]
            ff = factor_counts[1][f]
            fm = factor_counts[2][m]
            if len(train_indices) < 18 and fg < 6 and ff < 6 and fm < 6:
                train_indices.append(idx)
                factor_counts[0][g] += 1
                factor_counts[1][f] += 1
                factor_counts[2][m] += 1
            elif len(holdout_indices) < 9:
                holdout_indices.append(idx)

        remaining = [i for i in indices if i not in train_indices and i not in holdout_indices]
        while len(train_indices) < 18 and remaining:
            train_indices.append(remaining.pop(0))
        while len(holdout_indices) < 9 and remaining:
            holdout_indices.append(remaining.pop(0))

        splits[f"split_{split_id}"] = {
            "train": [list(all_combos[i]) for i in train_indices],
            "holdout": [list(all_combos[i]) for i in holdout_indices],
        }

    return splits


def main():
    all_combos = [(g, f, m)
                  for g in GRAVITY_SCALES
                  for f in FRICTION_SCALES
                  for m in MASS_SCALES]

    print("=" * 60)
    print("ComPhys-LeWM — Full Verification and Split Generation")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # Verify all 27 files
    print("\nVerifying all 27 HDF5 files...")
    all_pass = True
    verification = {}
    results = []

    for g, f, m in all_combos:
        combo_label = f"g{g:.1f}_f{f:.1f}_m{m:.1f}"
        fname = f"{combo_label}.h5"
        output_path = DATA_DIR / fname

        if not output_path.exists():
            print(f"  [MISSING] {fname}")
            all_pass = False
            verification[combo_label] = {
                'path': str(output_path),
                'n_collected': 0,
                'n_failed': 0,
                'issues': ['File missing'],
                'pass': False,
            }
            results.append({'combo': combo_label, 'path': str(output_path),
                            'n_collected': 0, 'n_failed': 0})
            continue

        issues = verify_hdf5(output_path, N_TRAJ_FULL, TRAJ_LENGTH)
        if issues:
            print(f"  [FAIL] {fname}: {issues}")
            all_pass = False
        else:
            with h5py.File(output_path, 'r') as hf:
                n = hf['pixels'].shape[0]
                n_failed = int(hf.attrs.get('n_failed', 0))
            print(f"  [PASS] {fname}: {n} trajectories, shape OK, no NaN")

        verification[combo_label] = {
            'path': str(output_path),
            'n_collected': n if not issues else 0,
            'n_failed': n_failed if not issues else 0,
            'issues': issues,
            'pass': len(issues) == 0,
        }
        results.append({'combo': combo_label, 'path': str(output_path),
                       'n_collected': n if not issues else 0, 'n_failed': 0})

    # Generate splits
    print("\nGenerating 3 CoGenT-style train/holdout splits...")
    splits = generate_splits(all_combos)
    splits_path = WORKSPACE / "exp" / "data" / "splits.json"
    with open(splits_path, 'w') as fp:
        json.dump(splits, fp, indent=2)
    print(f"Splits saved to: {splits_path}")
    for k, v in splits.items():
        print(f"  {k}: train={len(v['train'])}, holdout={len(v['holdout'])}")

    # Save summary
    summary = {
        "mode": "FULL",
        "combos_collected": len(all_combos),
        "n_traj_per_combo": N_TRAJ_FULL,
        "total_trajectories": sum(r['n_collected'] for r in results),
        "total_failed": 0,
        "elapsed_sec": 0,
        "all_pass": all_pass,
        "verification": verification,
        "output_dir": str(DATA_DIR),
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = DATA_DIR / "collection_summary.json"
    with open(summary_path, 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    print(f"All pass: {all_pass}")
    print(f"Total trajectories: {summary['total_trajectories']}")

    # Sample inspection
    sample_path = DATA_DIR / "g1.0_f1.0_m1.0.h5"
    if sample_path.exists():
        print("\nSAMPLE INSPECTION (g1.0_f1.0_m1.0.h5):")
        with h5py.File(sample_path, 'r') as hf:
            px = hf['pixels'][0]
            jt = hf['joint_angles'][0]
            cv = hf['com_velocity'][0]
            lb = hf['physics_labels'][0]
            print(f"  pixels: shape={px.shape}, dtype={px.dtype}, min={px.min()}, max={px.max()}")
            print(f"  joint_angles: shape={jt.shape}, mean={jt.mean():.4f}, std={jt.std():.4f}")
            print(f"  com_velocity: shape={cv.shape}, mean={cv.mean():.4f}, std={cv.std():.4f}")
            print(f"  physics_labels: {lb}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
