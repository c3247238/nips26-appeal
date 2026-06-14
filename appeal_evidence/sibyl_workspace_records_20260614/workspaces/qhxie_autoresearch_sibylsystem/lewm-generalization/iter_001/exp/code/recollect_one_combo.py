"""
Recollect a single missing combo: g1.0_f0.5_m2.0 (gravity=1.0, friction=0.5, mass=2.0)
"""
import os
import sys
import json
import time
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

os.environ['MUJOCO_GL'] = 'egl'

from dm_control import suite

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/lewm-generalization/iter_001")
DATA_DIR = WORKSPACE / "exp" / "data" / "comphys_lewm"
RESULTS_DIR = WORKSPACE / "exp" / "results"

BASE_GRAVITY = 9.81
TRAJ_LENGTH = 300
N_TRAJ_FULL = 200
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
RANDOM_SEED = 42

TASK_ID = "data_collection"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"


def make_walker_env(gravity_scale=1.0, friction_scale=1.0, mass_scale=1.0, seed=42):
    env = suite.load('walker', 'walk', task_kwargs={'random': seed})
    env.physics.model.opt.gravity[2] = -BASE_GRAVITY * gravity_scale
    env.physics.model.geom_friction[:, 0] *= friction_scale
    env.physics.model.body_mass[:] *= mass_scale
    return env


def collect_trajectory(env, traj_length=TRAJ_LENGTH, rng=None):
    if rng is None:
        rng = np.random.RandomState(RANDOM_SEED)
    ts = env.reset()
    action_spec = env.action_spec()
    pixels_list, joints_list, comv_list, cfrc_list = [], [], [], []

    for step in range(traj_length):
        action = rng.uniform(action_spec.minimum, action_spec.maximum)
        ts = env.step(action)
        frame = env.physics.render(height=FRAME_HEIGHT, width=FRAME_WIDTH, camera_id=0)
        qpos = env.physics.data.qpos.copy()
        joint_angles = qpos[3:]
        qvel = env.physics.data.qvel.copy()
        com_vel = qvel[0:2]
        cfrc = env.physics.data.cfrc_ext.copy()
        contact_force_mag = np.linalg.norm(cfrc, axis=1)[:4]
        pixels_list.append(frame)
        joints_list.append(joint_angles.astype(np.float32))
        comv_list.append(com_vel.astype(np.float32))
        cfrc_list.append(contact_force_mag.astype(np.float32))
        if ts.last():
            ts = env.reset()

    return {
        'pixels': np.array(pixels_list, dtype=np.uint8),
        'joint_angles': np.array(joints_list, dtype=np.float32),
        'com_velocity': np.array(comv_list, dtype=np.float32),
        'contact_forces': np.array(cfrc_list, dtype=np.float32),
    }


def main():
    g, f, m = 1.0, 0.5, 2.0
    combo_label = f"g{g:.1f}_f{f:.1f}_m{m:.1f}"
    fname = f"{combo_label}.h5"
    output_path = DATA_DIR / fname

    print(f"Recollecting: {combo_label}")
    print(f"Output: {output_path}")
    print(f"N trajectories: {N_TRAJ_FULL}")

    start_time = time.time()

    all_pixels = np.zeros((N_TRAJ_FULL, TRAJ_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    all_joints = np.zeros((N_TRAJ_FULL, TRAJ_LENGTH, 6), dtype=np.float32)
    all_comv = np.zeros((N_TRAJ_FULL, TRAJ_LENGTH, 2), dtype=np.float32)
    all_cfrc = np.zeros((N_TRAJ_FULL, TRAJ_LENGTH, 4), dtype=np.float32)

    env = make_walker_env(g, f, m, seed=RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED + 11000)  # seed_offset = combo_idx * 1000 = 11*1000

    n_collected = 0
    n_failed = 0

    for i in range(N_TRAJ_FULL):
        try:
            traj = collect_trajectory(env, rng=rng)
            all_pixels[i] = traj['pixels']
            all_joints[i] = traj['joint_angles']
            all_comv[i] = traj['com_velocity']
            all_cfrc[i] = traj['contact_forces']
            n_collected += 1
        except ValueError as e:
            print(f"  [WARN] Traj {i} failed: {e}")
            n_failed += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{N_TRAJ_FULL} ({n_failed} failed) | {elapsed:.1f}s")
            # Update progress file
            prog = {
                "task_id": TASK_ID,
                "epoch": 26,
                "total_epochs": 27,
                "step": i + 1,
                "total_steps": N_TRAJ_FULL,
                "loss": None,
                "metric": {"combo": 26, "trajectory": i + 1, "elapsed_sec": round(elapsed, 1)},
                "updated_at": datetime.now().isoformat(),
            }
            PROGRESS_FILE.write_text(json.dumps(prog))

    # Save HDF5
    print(f"Saving {n_collected} trajectories to {fname}...")
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('pixels', data=all_pixels, compression='gzip', compression_opts=4)
        hf.create_dataset('joint_angles', data=all_joints)
        hf.create_dataset('com_velocity', data=all_comv)
        hf.create_dataset('contact_forces', data=all_cfrc)
        hf.create_dataset('physics_labels',
                          data=np.array([[g, f, m]] * N_TRAJ_FULL, dtype=np.float32))
        hf.attrs['gravity_scale'] = g
        hf.attrs['friction_scale'] = f
        hf.attrs['mass_scale'] = m
        hf.attrs['n_trajectories'] = n_collected
        hf.attrs['n_failed'] = n_failed
        hf.attrs['traj_length'] = TRAJ_LENGTH
        hf.attrs['frame_height'] = FRAME_HEIGHT
        hf.attrs['frame_width'] = FRAME_WIDTH
        hf.attrs['created_at'] = datetime.now().isoformat()
        hf.attrs['combo_label'] = combo_label

    elapsed = time.time() - start_time
    print(f"[OK] Saved {n_collected} trajectories ({n_failed} failed) to {fname}")
    print(f"Total time: {elapsed:.1f}s")

    # Verify
    with h5py.File(output_path, 'r') as hf:
        n = hf['pixels'].shape[0]
        shape = hf['pixels'].shape[1:]
        labels = hf['physics_labels'][0]
        print(f"Verification: n_traj={n}, shape={shape}, labels={labels}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
