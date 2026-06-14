"""
PILOT Data Collection Script: ComPhys-LeWM Benchmark
==============================================
Pilot scope:
  - 100 trajectories for 3 friction combos (0.5x, 1.0x, 2.0x)
  - gravity=1.0g (nominal), mass=1.0x (nominal)
  - Walker-walk environment in DMControl
  - Output: 3 HDF5 files in exp/data/comphys_lewm/pilot/
  - Pass criteria: HDF5 files readable; frame shape (100, T, 64, 64, 3);
                   physics labels stored correctly; no NaN frames

Full-run parameters are defined at top but only the pilot subset is executed
when PILOT=True.
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py

# Set EGL for headless rendering BEFORE importing dm_control
os.environ['MUJOCO_GL'] = 'egl'

from dm_control import suite


# ========================= Config =========================
WORKSPACE = Path(__file__).parent.parent.parent  # .../current
RESULTS_DIR = WORKSPACE / "exp" / "results"
DATA_DIR = WORKSPACE / "exp" / "data" / "comphys_lewm"
PILOT_DIR = DATA_DIR / "pilot"

TASK_ID = "data_collection"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

# =================== Physics Factors ======================
GRAVITY_SCALES = [0.5, 1.0, 2.0]    # {0.5g, 1.0g, 2.0g}
FRICTION_SCALES = [0.5, 1.0, 2.0]   # {0.5x, 1.0x, 2.0x}
MASS_SCALES = [0.5, 1.0, 2.0]       # {0.5x, 1.0x, 2.0x}

BASE_GRAVITY = 9.81  # m/s^2, MuJoCo uses -z convention

# Pilot: only 3 friction combos, gravity=1.0g, mass=1.0x
PILOT_COMBOS = [
    (1.0, 0.5, 1.0),  # (gravity_scale, friction_scale, mass_scale)
    (1.0, 1.0, 1.0),
    (1.0, 2.0, 1.0),
]

# Trajectory collection settings
TRAJ_LENGTH = 300   # 300 steps per trajectory
N_TRAJ_PILOT = 100  # 100 trajectories per combo for pilot
N_TRAJ_FULL = 200   # 200 trajectories per combo for full run
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
RANDOM_SEED = 42


# =================== Helper Functions ======================

def write_pid():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    print(f"[PID] Written PID {os.getpid()} to {PID_FILE}")


def write_progress(combo_idx, total_combos, traj_idx, total_traj, elapsed_sec):
    prog = {
        "task_id": TASK_ID,
        "epoch": combo_idx,
        "total_epochs": total_combos,
        "step": traj_idx,
        "total_steps": total_traj,
        "loss": None,
        "metric": {
            "combo": combo_idx,
            "trajectory": traj_idx,
            "elapsed_sec": round(elapsed_sec, 1),
        },
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(prog))


def write_done(status="success", summary="", final_progress=None):
    if PID_FILE.exists():
        PID_FILE.unlink()
    fp = {}
    if PROGRESS_FILE.exists():
        try:
            fp = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    marker = {
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": fp,
        "timestamp": datetime.now().isoformat(),
    }
    DONE_FILE.write_text(json.dumps(marker, indent=2))
    print(f"[DONE] Written DONE marker: status={status}, summary={summary[:100]}")


# =================== Environment Builder ===================

def make_walker_env(gravity_scale=1.0, friction_scale=1.0, mass_scale=1.0, seed=42):
    """Create Walker-walk environment with custom physics parameters."""
    env = suite.load('walker', 'walk', task_kwargs={'random': seed})

    # Modify gravity (MuJoCo uses -z convention)
    env.physics.model.opt.gravity[2] = -BASE_GRAVITY * gravity_scale

    # Modify friction for all geoms
    # Default MuJoCo geom friction ~ [1.0, 0.005, 0.0001] (sliding, torsional, rolling)
    env.physics.model.geom_friction[:, 0] *= friction_scale  # sliding friction only

    # Modify body mass
    env.physics.model.body_mass[:] *= mass_scale

    return env


# =================== Trajectory Collector =================

def collect_trajectory(env, traj_length=TRAJ_LENGTH, seed=42, rng=None):
    """
    Collect a single trajectory using random-walk policy.
    Returns: dict with pixels, joint_angles, com_velocity, contact_forces
    Raises ValueError if NaN detected.
    """
    if rng is None:
        rng = np.random.RandomState(seed)

    ts = env.reset()
    action_spec = env.action_spec()
    action_dim = action_spec.shape[0]

    pixels_list = []
    joint_angles_list = []
    com_velocity_list = []
    contact_forces_list = []

    for step in range(traj_length):
        # Random-walk policy: smooth random actions
        action = rng.uniform(action_spec.minimum, action_spec.maximum)

        ts = env.step(action)

        # Render frame
        frame = env.physics.render(height=FRAME_HEIGHT, width=FRAME_WIDTH, camera_id=0)

        # Joint angles (degrees of freedom)
        # Walker has 9 DOF: rootz, rootx, rooty, right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle
        # We take the 6 actuated joints (excluding root)
        qpos = env.physics.data.qpos.copy()
        joint_angles = qpos[3:]  # 6 actuated joints

        # CoM velocity (from velocity observation - 9-dim; we use [0,1] for horizontal/vertical)
        # Actually the velocity obs gives: rootz_vel, rootx_vel, rooty_vel, joint_vels...
        # We want CoM velocity = x and z translational velocities
        qvel = env.physics.data.qvel.copy()
        com_vel = qvel[0:2]  # x-velocity, z-velocity (CoM translational)

        # Contact forces (sum of external forces on feet/floor contacts)
        cfrc = env.physics.data.cfrc_ext.copy()
        # Sum force magnitudes on foot bodies
        contact_force_mag = np.linalg.norm(cfrc, axis=1)[:4]  # first 4 bodies

        pixels_list.append(frame)
        joint_angles_list.append(joint_angles.astype(np.float32))
        com_velocity_list.append(com_vel.astype(np.float32))
        contact_forces_list.append(contact_force_mag.astype(np.float32))

        if ts.last():
            # Re-initialize remaining steps with reset
            ts = env.reset()

    pixels_arr = np.array(pixels_list, dtype=np.uint8)  # (T, 64, 64, 3)
    joint_arr = np.array(joint_angles_list, dtype=np.float32)  # (T, 6)
    com_arr = np.array(com_velocity_list, dtype=np.float32)    # (T, 2)
    cf_arr = np.array(contact_forces_list, dtype=np.float32)   # (T, 4)

    # Check for NaN
    for name, arr in [('joint_angles', joint_arr), ('com_velocity', com_arr), ('contact_forces', cf_arr)]:
        if np.any(np.isnan(arr)):
            raise ValueError(f"NaN detected in {name}")

    return {
        'pixels': pixels_arr,
        'joint_angles': joint_arr,
        'com_velocity': com_arr,
        'contact_forces': cf_arr,
    }


# =================== Main Collection ======================

def collect_combo(gravity_scale, friction_scale, mass_scale,
                  n_traj, output_dir, combo_label,
                  combo_idx, total_combos, start_time,
                  seed_offset=0):
    """Collect n_traj trajectories for one physics combo and save to HDF5."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"g{gravity_scale:.1f}_f{friction_scale:.1f}_m{mass_scale:.1f}.h5"
    output_path = output_dir / fname

    if output_path.exists():
        print(f"  [SKIP] {fname} already exists, verifying...")
        with h5py.File(output_path, 'r') as f:
            n_existing = f['pixels'].shape[0]
        if n_existing >= n_traj:
            print(f"  [OK] {fname} has {n_existing} trajectories, skipping collection.")
            return str(output_path), n_existing, 0
        else:
            print(f"  [INCOMPLETE] {fname} has {n_existing}/{n_traj}, re-collecting.")
            output_path.unlink()

    print(f"  Collecting {n_traj} trajectories for {combo_label}...")

    # Pre-allocate arrays
    all_pixels = np.zeros((n_traj, TRAJ_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    all_joints = np.zeros((n_traj, TRAJ_LENGTH, 6), dtype=np.float32)
    all_comv = np.zeros((n_traj, TRAJ_LENGTH, 2), dtype=np.float32)
    all_cfrc = np.zeros((n_traj, TRAJ_LENGTH, 4), dtype=np.float32)

    env = make_walker_env(gravity_scale, friction_scale, mass_scale, seed=RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED + seed_offset)

    n_collected = 0
    n_failed = 0

    for i in range(n_traj):
        traj_seed = RANDOM_SEED + seed_offset + i
        try:
            traj = collect_trajectory(env, traj_length=TRAJ_LENGTH, seed=traj_seed, rng=rng)
            all_pixels[i] = traj['pixels']
            all_joints[i] = traj['joint_angles']
            all_comv[i] = traj['com_velocity']
            all_cfrc[i] = traj['contact_forces']
            n_collected += 1
        except ValueError as e:
            print(f"    [WARN] Traj {i} failed: {e}, using zeros")
            n_failed += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            write_progress(combo_idx, total_combos, i + 1, n_traj, elapsed)
            print(f"    Progress: {i+1}/{n_traj} trajectories "
                  f"({n_failed} failed) | elapsed: {elapsed:.1f}s")

    # Save to HDF5
    print(f"  Saving {n_collected} trajectories to {fname}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('pixels', data=all_pixels, compression='gzip', compression_opts=4)
        f.create_dataset('joint_angles', data=all_joints)
        f.create_dataset('com_velocity', data=all_comv)
        f.create_dataset('contact_forces', data=all_cfrc)
        f.create_dataset('physics_labels',
                         data=np.array([[gravity_scale, friction_scale, mass_scale]] * n_traj,
                                       dtype=np.float32))
        f.attrs['gravity_scale'] = gravity_scale
        f.attrs['friction_scale'] = friction_scale
        f.attrs['mass_scale'] = mass_scale
        f.attrs['n_trajectories'] = n_collected
        f.attrs['n_failed'] = n_failed
        f.attrs['traj_length'] = TRAJ_LENGTH
        f.attrs['frame_height'] = FRAME_HEIGHT
        f.attrs['frame_width'] = FRAME_WIDTH
        f.attrs['created_at'] = datetime.now().isoformat()
        f.attrs['combo_label'] = combo_label

    print(f"  [OK] Saved {n_collected} trajectories ({n_failed} failed) to {fname}")
    return str(output_path), n_collected, n_failed


def verify_hdf5(path, expected_n_traj, expected_traj_length):
    """Verify HDF5 file meets pass criteria."""
    issues = []
    with h5py.File(path, 'r') as f:
        pixels = f['pixels']
        joints = f['joint_angles']
        comv = f['com_velocity']
        labels = f['physics_labels']

        # Shape checks
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

        # NaN checks
        sample_pixels = pixels[:10]
        if np.any(np.isnan(sample_pixels.astype(np.float32))):
            issues.append("NaN frames detected in pixels")

        sample_joints = joints[:10]
        if np.any(np.isnan(sample_joints[:])):
            issues.append("NaN in joint_angles")

        sample_comv = comv[:10]
        if np.any(np.isnan(sample_comv[:])):
            issues.append("NaN in com_velocity")

        # Label check
        lbl_sample = labels[:5][:]
        if not np.all(lbl_sample > 0):
            issues.append("physics_labels contain non-positive values")

    return issues


def generate_splits(all_combos, output_path):
    """Generate 3 CoGenT-style train/holdout splits (18/27 train, 9/27 holdout each)."""
    rng = np.random.RandomState(RANDOM_SEED)
    splits = {}
    for split_id in range(3):
        # CoGenT-style: ensure each factor level appears in >= 6 training combos
        # Strategy: for each factor, hold out the combo where that factor is at level 1 (mid)
        # combined with specific patterns to create 9 holdout combos
        indices = list(range(len(all_combos)))
        rng.shuffle(indices)
        # Simple split: 18 train, 9 holdout
        # Ensure balanced factor coverage
        train_indices = []
        holdout_indices = []
        factor_counts = {0: {0.5: 0, 1.0: 0, 2.0: 0},
                         1: {0.5: 0, 1.0: 0, 2.0: 0},
                         2: {0.5: 0, 1.0: 0, 2.0: 0}}

        for idx in indices:
            g, f, m = all_combos[idx]
            # Add to train if factor counts allow
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

        # If still not 18/9, fill greedily
        remaining = [i for i in indices if i not in train_indices and i not in holdout_indices]
        while len(train_indices) < 18 and remaining:
            train_indices.append(remaining.pop(0))
        while len(holdout_indices) < 9 and remaining:
            holdout_indices.append(remaining.pop(0))

        splits[f"split_{split_id}"] = {
            "train": [all_combos[i] for i in train_indices],
            "holdout": [all_combos[i] for i in holdout_indices],
        }

    with open(output_path, 'w') as fp:
        json.dump(splits, fp, indent=2)

    print(f"[SPLITS] Saved 3 CoGenT-style splits to {output_path}")
    for k, v in splits.items():
        print(f"  {k}: train={len(v['train'])}, holdout={len(v['holdout'])}")
    return splits


# ========================= Main ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true', default=True,
                        help='Run pilot only (default)')
    parser.add_argument('--full', action='store_true', default=False,
                        help='Run full data collection (27 combos x 200 traj)')
    args = parser.parse_args()

    is_pilot = not args.full
    mode = "PILOT" if is_pilot else "FULL"

    print("=" * 60)
    print(f"ComPhys-LeWM Data Collection — {mode} MODE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Workspace: {WORKSPACE}")
    print("=" * 60)

    # Write PID
    write_pid()
    start_time = time.time()

    try:
        if is_pilot:
            combos = PILOT_COMBOS
            n_traj = N_TRAJ_PILOT
            output_dir = PILOT_DIR
            print(f"Pilot: {len(combos)} combos x {n_traj} trajectories each")
        else:
            # Full: all 27 combos
            combos = [(g, f, m)
                      for g in GRAVITY_SCALES
                      for f in FRICTION_SCALES
                      for m in MASS_SCALES]
            n_traj = N_TRAJ_FULL
            output_dir = DATA_DIR
            print(f"Full: {len(combos)} combos x {n_traj} trajectories each")

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        print()

        # Collect data
        results = []
        total_failed = 0

        for combo_idx, (g, f, m) in enumerate(combos):
            combo_label = f"g{g:.1f}_f{f:.1f}_m{m:.1f}"
            print(f"[{combo_idx+1}/{len(combos)}] Collecting combo: "
                  f"gravity={g:.1f}x, friction={f:.1f}x, mass={m:.1f}x")

            path, n_collected, n_failed = collect_combo(
                g, f, m, n_traj, output_dir, combo_label,
                combo_idx, len(combos), start_time,
                seed_offset=combo_idx * 1000
            )
            total_failed += n_failed
            results.append({
                'combo': combo_label,
                'path': path,
                'n_collected': n_collected,
                'n_failed': n_failed,
            })

        print()
        print("=" * 60)
        print("VERIFICATION")
        print("=" * 60)

        # Verify all files
        all_pass = True
        verification = {}
        for res in results:
            path = Path(res['path'])
            issues = verify_hdf5(path, n_traj, TRAJ_LENGTH)
            if issues:
                print(f"  [FAIL] {path.name}: {issues}")
                all_pass = False
            else:
                print(f"  [PASS] {path.name}: {res['n_collected']} trajectories, shape OK, no NaN")
            verification[res['combo']] = {
                'path': str(path),
                'n_collected': res['n_collected'],
                'n_failed': res['n_failed'],
                'issues': issues,
                'pass': len(issues) == 0,
            }

        # Generate splits if full run
        splits_path = WORKSPACE / "exp" / "data" / "splits.json"
        if not is_pilot:
            all_combos = [(g, f, m)
                          for g in GRAVITY_SCALES
                          for f in FRICTION_SCALES
                          for m in MASS_SCALES]
            generate_splits(all_combos, splits_path)

        # Summary
        elapsed = time.time() - start_time
        summary = {
            "mode": mode,
            "combos_collected": len(combos),
            "n_traj_per_combo": n_traj,
            "total_trajectories": sum(r['n_collected'] for r in results),
            "total_failed": total_failed,
            "elapsed_sec": round(elapsed, 1),
            "all_pass": all_pass,
            "verification": verification,
            "output_dir": str(output_dir),
            "timestamp": datetime.now().isoformat(),
        }

        summary_path = output_dir / "collection_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print()
        print(f"Elapsed: {elapsed:.1f}s")
        print(f"Total trajectories: {summary['total_trajectories']}")
        print(f"Total failed: {total_failed}")
        print(f"All pass: {all_pass}")
        print(f"Summary saved to: {summary_path}")

        # Quick inspection: print sample stats for sanity
        print()
        print("SAMPLE INSPECTION (first collected file):")
        sample_file = output_dir / f"g{combos[0][0]:.1f}_f{combos[0][1]:.1f}_m{combos[0][2]:.1f}.h5"
        if sample_file.exists():
            with h5py.File(sample_file, 'r') as f:
                px = f['pixels'][0]  # first trajectory
                jt = f['joint_angles'][0]
                cv = f['com_velocity'][0]
                lb = f['physics_labels'][0]
                print(f"  pixels shape: {px.shape}, dtype: {px.dtype}, "
                      f"min={px.min()}, max={px.max()}")
                print(f"  joint_angles shape: {jt.shape}, "
                      f"mean={jt.mean():.4f}, std={jt.std():.4f}")
                print(f"  com_velocity shape: {cv.shape}, "
                      f"mean={cv.mean():.4f}, std={cv.std():.4f}")
                print(f"  physics_labels: {lb}")

        final_status = "success" if all_pass else "partial"
        write_done(
            status=final_status,
            summary=f"{mode}: {len(combos)} combos, "
                    f"{summary['total_trajectories']} trajectories, "
                    f"all_pass={all_pass}"
        )

        return 0 if all_pass else 1

    except Exception as e:
        print(f"\n[ERROR] Data collection failed: {e}")
        traceback.print_exc()
        write_done(status="failed", summary=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
