"""
Environment Verification Script for LeWM Generalization Project
Verifies all required dependencies are installed and functional.
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add local packages to path
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

WORKSPACE = Path(__file__).parent.parent.parent
RESULTS_DIR = WORKSPACE / "exp" / "code"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results = {
    "timestamp": datetime.now().isoformat(),
    "python_version": sys.version,
    "checks": {},
    "errors": [],
    "status": "PENDING"
}

def check(name, fn):
    try:
        result = fn()
        results["checks"][name] = {"status": "PASS", "info": str(result) if result else "OK"}
        print(f"  [PASS] {name}: {result if result else 'OK'}")
        return True
    except Exception as e:
        results["checks"][name] = {"status": "FAIL", "error": str(e)}
        results["errors"].append(f"{name}: {e}")
        print(f"  [FAIL] {name}: {e}")
        return False


print("=" * 60)
print("LeWM Generalization Environment Verification")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Workspace: {WORKSPACE}")
print()

# === Core Libraries ===
print("1. Core libraries:")

check("torch", lambda: __import__('torch').__version__)
check("torchvision", lambda: __import__('torchvision').__version__)

def check_cuda():
    import torch
    if torch.cuda.is_available():
        return f"CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s): {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    return "CPU only (no CUDA)"
check("cuda", check_cuda)

check("numpy", lambda: __import__('numpy').__version__)
check("h5py", lambda: __import__('h5py').__version__)
check("matplotlib", lambda: __import__('matplotlib').__version__)
check("scikit-learn", lambda: __import__('sklearn').__version__)
check("einops", lambda: __import__('einops').__version__)
check("peft", lambda: __import__('peft').__version__)
check("timm", lambda: __import__('timm').__version__)

print()

# === RL Environments ===
print("2. RL environments:")

check("dm-control", lambda: __import__('dm_control').__version__ if hasattr(__import__('dm_control'), '__version__') else "installed")
check("mujoco", lambda: __import__('mujoco').__version__)

def check_walker_env():
    """Verify Walker environment with default physics params."""
    from dm_control import suite
    env = suite.load('walker', 'walk', task_kwargs={'random': 42})
    timestep = env.reset()
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()
    obs_keys = list(obs_spec.keys())
    return f"Walker-walk loaded. Obs keys: {obs_keys[:3]}... Action dim: {action_spec.shape}"
check("dm_control.walker_walk", check_walker_env)

def check_walker_gravity():
    """Verify Walker environment with custom gravity."""
    import numpy as np
    from dm_control import suite
    # Use default physics modification via XML
    env = suite.load('walker', 'walk')
    # Modify gravity in physics
    env.physics.model.opt.gravity[2] = -9.81 * 2.0  # 2x gravity
    env.reset()
    actual_gravity = env.physics.model.opt.gravity[2]
    return f"Custom gravity={actual_gravity:.2f} (expected -19.62)"
check("walker_custom_gravity", check_walker_gravity)

def check_walker_friction():
    """Verify Walker environment with custom friction."""
    from dm_control import suite
    import numpy as np
    env = suite.load('walker', 'walk')
    # Modify friction
    env.physics.model.geom_friction[:] *= 0.5  # 0.5x friction
    env.reset()
    return f"Custom friction applied (scale=0.5x)"
check("walker_custom_friction", check_walker_friction)

print()

# === stable-worldmodel ===
print("3. stable-worldmodel:")
check("stable_worldmodel", lambda: __import__('stable_worldmodel').__version__ if hasattr(__import__('stable_worldmodel'), '__version__') else "installed")
check("stable_pretraining", lambda: __import__('stable_pretraining').__version__ if hasattr(__import__('stable_pretraining'), '__version__') else "installed")

print()

# === le-wm ===
print("4. le-wm:")

LE_WM_DIR = CODE_DIR / "le-wm"
sys.path.insert(0, str(LE_WM_DIR))

def check_lewm_import():
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP, SIGReg
    from utils import get_img_preprocessor
    return "jepa, module, utils imported successfully"
check("lewm_imports", check_lewm_import)

def check_sigreg():
    """Test SIGReg instantiation and forward pass."""
    import torch
    from module import SIGReg
    sigreg = SIGReg(knots=17, num_proj=128)
    # (T, B, D) = (5, 4, 192)
    x = torch.randn(5, 4, 192)
    loss = sigreg(x)
    return f"SIGReg forward pass OK, loss={loss.item():.4f}"
check("sigreg_forward", check_sigreg)

def check_lewm_1step():
    """Verify le-wm model runs for 1 step without error."""
    import torch
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP, SIGReg

    # Build a minimal LeWM model
    from transformers import ViTConfig, ViTModel

    # Use tiny ViT config (matching le-wm's encoder_scale='tiny')
    vit_config = ViTConfig(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
        image_size=64,
        patch_size=16,
    )
    encoder = ViTModel(vit_config)

    # Minimal action encoder using Embedder (correct API)
    action_dim = 6  # Walker action dim
    embed_dim = 192
    action_encoder = Embedder(
        input_dim=action_dim,
        smoothed_dim=action_dim,
        emb_dim=embed_dim,
        mlp_scale=4,
    )

    # Predictor (correct ARPredictor API)
    history_size = 3  # matches config wm.history_size
    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=2,
        heads=4,
        mlp_dim=512,
        dim_head=48,
        dropout=0.0,
        emb_dropout=0.0,
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
    )

    # 1-step forward pass (CPU only since CUDA incompatible with current torch)
    B, T, C, H, W = 2, 4, 3, 64, 64
    pixels = torch.randn(B, T, C, H, W)
    actions = torch.randn(B, T, action_dim)

    info = {"pixels": pixels, "action": actions}
    with torch.no_grad():
        output = model.encode(info)
        emb = output["emb"]       # (B, T, D)
        act_emb = output["act_emb"]  # (B, T, A_emb)
        pred = model.predict(emb[:, :3], act_emb[:, :3])  # predict from 3 context frames

    return f"1-step forward pass OK. emb={emb.shape}, pred={pred.shape}"
check("lewm_1step", check_lewm_1step)

print()

# === necessary-compositionality (local) ===
print("5. necessary-compositionality (local):")

def check_principal_angles():
    """Verify principal angle analysis works."""
    import numpy as np
    import sys
    sys.path.insert(0, str(CODE_DIR))
    from necessary_compositionality import principal_angles, mean_cosine_similarity, principal_angle_matrix

    rng = np.random.RandomState(42)
    # Two random subspaces in 192-d space
    A = rng.randn(50, 192)
    B = rng.randn(50, 192)

    cos_sim = mean_cosine_similarity(A, B, n_components=8)

    # Factor-grouped test
    factor_embeddings = {
        "gravity": rng.randn(60, 192),
        "friction": rng.randn(60, 192),
        "mass": rng.randn(60, 192),
    }
    matrix = principal_angle_matrix(factor_embeddings, n_components=4)

    assert matrix.shape == (3, 3), f"Expected (3,3), got {matrix.shape}"
    assert 0.0 <= matrix[0, 1] <= 1.0, f"Cosine sim out of range: {matrix[0, 1]}"

    return f"Principal angle matrix (3x3) computed. cos_sim(A,B)={cos_sim:.4f}. Matrix diagonal all 1.0: {all(abs(matrix[i,i]-1.0) < 1e-6 for i in range(3))}"
check("necessary_compositionality", check_principal_angles)

print()

# === Data format verification (HDF5 shape check) ===
print("6. HDF5 data format check:")

def check_hdf5_write_read():
    """Verify HDF5 file creation for trajectory data."""
    import h5py
    import numpy as np

    test_file = WORKSPACE / "exp" / "code" / "test_data_format.h5"
    T_traj = 50  # trajectory length
    n_traj = 5

    with h5py.File(test_file, 'w') as f:
        f.create_dataset('pixels', data=np.random.randint(0, 255, (n_traj, T_traj, 64, 64, 3), dtype=np.uint8))
        f.create_dataset('joint_angles', data=np.random.randn(n_traj, T_traj, 6).astype(np.float32))
        f.create_dataset('com_velocity', data=np.random.randn(n_traj, T_traj, 2).astype(np.float32))
        f.create_dataset('contact_forces', data=np.random.randn(n_traj, T_traj, 4).astype(np.float32))
        f.create_dataset('physics_labels', data=np.array([[1.0, 1.0, 1.0]] * n_traj, dtype=np.float32))  # [gravity_scale, friction_scale, mass_scale]
        f.attrs['gravity_scale'] = 1.0
        f.attrs['friction_scale'] = 1.0
        f.attrs['mass_scale'] = 1.0
        f.attrs['n_trajectories'] = n_traj

    with h5py.File(test_file, 'r') as f:
        frames = f['pixels'][:]
        labels = f['physics_labels'][:]

    test_file.unlink()  # cleanup

    assert frames.shape == (n_traj, T_traj, 64, 64, 3), f"Frame shape wrong: {frames.shape}"
    assert labels.shape == (n_traj, 3), f"Labels shape wrong: {labels.shape}"

    return f"HDF5 write/read OK. pixels={frames.shape}, labels={labels.shape}"
check("hdf5_format", check_hdf5_write_read)

print()

# === GPU test ===
print("7. GPU availability for experiments:")

def check_gpu_for_training():
    import torch
    if not torch.cuda.is_available():
        return "No GPU available, will use CPU"

    # Check GPU 2 specifically
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # Re-check after setting env
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        gpus = result.stdout.strip().split('\n')
        for gpu in gpus:
            print(f"    GPU: {gpu.strip()}")

    device = torch.device('cuda')
    x = torch.randn(64, 192).to(device)
    y = torch.matmul(x, x.T)
    return f"GPU test OK on {torch.cuda.get_device_name(0)}: matrix mul {x.shape} -> {y.shape}"
check("gpu_training_test", check_gpu_for_training)

print()
print("=" * 60)

# Compute pass/fail
n_pass = sum(1 for v in results["checks"].values() if v["status"] == "PASS")
n_total = len(results["checks"])

if len(results["errors"]) == 0:
    results["status"] = "ALL_PASS"
    print(f"RESULT: ALL {n_total} checks PASSED")
elif n_pass >= n_total * 0.8:
    results["status"] = "MOSTLY_PASS"
    print(f"RESULT: {n_pass}/{n_total} checks PASSED (mostly OK)")
    print("FAILURES:")
    for err in results["errors"]:
        print(f"  - {err}")
else:
    results["status"] = "FAIL"
    print(f"RESULT: {n_pass}/{n_total} checks PASSED")
    print("FAILURES:")
    for err in results["errors"]:
        print(f"  - {err}")

results["pass_count"] = n_pass
results["total_count"] = n_total

# Save results
output_path = WORKSPACE / "exp" / "code" / "setup_verification.txt"
with open(output_path, 'w') as f:
    f.write(f"LeWM Generalization Environment Setup Verification\n")
    f.write(f"Timestamp: {results['timestamp']}\n")
    f.write(f"Status: {results['status']}\n")
    f.write(f"Pass: {n_pass}/{n_total}\n\n")
    f.write("Checks:\n")
    for name, check_result in results["checks"].items():
        status = check_result["status"]
        info = check_result.get("info", check_result.get("error", ""))
        f.write(f"  [{status}] {name}: {info}\n")
    if results["errors"]:
        f.write("\nErrors:\n")
        for err in results["errors"]:
            f.write(f"  - {err}\n")

output_json = WORKSPACE / "exp" / "code" / "setup_verification.json"
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_path}")
print(f"JSON saved to: {output_json}")
