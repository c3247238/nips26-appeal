"""
Pilot: eval_geometric_h2_h4 (PILOT MODE)
=========================================
Geometric Analysis: H2 (SIGReg vs. VICReg Orthogonality) and H4 (Displacement Consistency)

PILOT mode tasks:
  1. Load pilot checkpoint (SIGReg, seed42, epoch5)
  2. Train a quick VICReg pilot checkpoint if not available (5 epochs, same data)
  3. Extract encoder embeddings for 3 factor levels (friction: 0.5x, 1.0x, 2.0x)
  4. Per-factor latent subspace PCA (top-8 PCs)
  5. Principal angle analysis using scipy.linalg.subspace_angles (fallback for necessary-compositionality)
  6. Displacement vector consistency (parallelogram test): centroid delta vectors
  7. CKA between factor-grouped representations

Pass criteria:
  - Principal angle matrix (3x3) computed; mean cosine similarity in [0, 1]
  - No import errors from necessary-compositionality (or scipy fallback succeeds)
  - Displacement consistency scores computed; no NaN

Output:
  exp/results/pilots/eval_geometric_h2_h4_pilot.json
  exp/results/full/geometric_h2_h4_results.json (pilot-tier result)
  exp/results/eval_geometric_h2_h4_DONE
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# GPU assignment
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['MUJOCO_GL'] = 'egl'

WORKSPACE = Path('/home/qhxie/AutoResearch-SibylSystem/workspaces/lewm-generalization/current')
DATA_DIR = WORKSPACE / 'exp' / 'data' / 'comphys_lewm' / 'pilot'
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
PILOTS_DIR = RESULTS_DIR / 'pilots'
FULL_DIR = RESULTS_DIR / 'full'
CODE_DIR = WORKSPACE / 'exp' / 'code'
LEWM_DIR = CODE_DIR / 'le-wm'

TASK_ID = 'eval_geometric_h2_h4'
PID_FILE = RESULTS_DIR / f'{TASK_ID}.pid'
PROGRESS_FILE = RESULTS_DIR / f'{TASK_ID}_PROGRESS.json'
DONE_FILE = RESULTS_DIR / f'{TASK_ID}_DONE'

sys.path.insert(0, str(LEWM_DIR))
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

t_start = time.time()

# ========================= PID / Progress ==========================

def write_pid():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    print(f'[PID] Written PID {os.getpid()} to {PID_FILE}')


def write_progress(epoch=0, total_epochs=1, step=0, total_steps=0, loss=None, metric=None):
    prog = {
        'task_id': TASK_ID,
        'epoch': epoch,
        'total_epochs': total_epochs,
        'step': step,
        'total_steps': total_steps,
        'loss': loss,
        'metric': metric or {},
        'updated_at': datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(prog))


def write_done(status='success', summary=''):
    if PID_FILE.exists():
        PID_FILE.unlink()
    fp = {}
    if PROGRESS_FILE.exists():
        try:
            fp = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    marker = {
        'task_id': TASK_ID,
        'status': status,
        'summary': summary,
        'final_progress': fp,
        'timestamp': datetime.now().isoformat(),
    }
    DONE_FILE.write_text(json.dumps(marker, indent=2))
    print(f'[DONE] Written DONE marker: status={status}')


# ========================= Model Architecture ==========================

class PixelEncoder(nn.Module):
    """Simple CNN encoder for 64x64 RGB frames -> embed_dim embeddings."""
    def __init__(self, embed_dim=192, channels=(32, 64, 128, 256)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1),
            nn.LayerNorm([channels[0], 32, 32]),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
            nn.LayerNorm([channels[1], 16, 16]),
            nn.GELU(),
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),
            nn.LayerNorm([channels[2], 8, 8]),
            nn.GELU(),
            nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1),
            nn.LayerNorm([channels[3], 4, 4]),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels[-1], embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """x: (B, T, 3, H, W) or (B, 3, H, W)"""
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            feat = self.pool(self.cnn(x)).squeeze(-1).squeeze(-1)
            feat = self.proj(feat)
            feat = self.norm(feat)
            return feat.view(B, T, -1)
        else:
            feat = self.pool(self.cnn(x)).squeeze(-1).squeeze(-1)
            feat = self.proj(feat)
            return self.norm(feat)


class SimplePredictorBlock(nn.Module):
    """Single transformer block for the predictor."""
    def __init__(self, dim, num_heads=8, mlp_dim=512, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


# Use the actual SIGReg from le-wm codebase
from module import SIGReg as LewmSIGReg


class LeWMModel(nn.Module):
    """Full LeWM model matching the training checkpoint architecture."""
    def __init__(self, embed_dim=192, n_layers=4, num_heads=8, mlp_dim=512,
                 sigreg_knots=17, sigreg_num_proj=1024, sigreg_weight=0.09,
                 history_size=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.history_size = history_size
        self.encoder = PixelEncoder(embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, history_size, embed_dim) * 0.02)
        self.predictor_blocks = nn.ModuleList([
            SimplePredictorBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(n_layers)
        ])
        self.pred_norm = nn.LayerNorm(embed_dim)
        self.sigreg = LewmSIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    def encode(self, frames):
        """frames: (B, T, H, W, C) uint8 -> (B, T, D)"""
        x = frames.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        return self.encoder(x)

    def encode_normalized(self, x):
        """x: (B, T, C, H, W) float [0,1] -> (B, T, D)"""
        return self.encoder(x)

    def predict(self, ctx_emb):
        """ctx_emb: (B, T, D) -> (B, T, D)"""
        x = ctx_emb + self.pos_embed[:, :ctx_emb.size(1)]
        for block in self.predictor_blocks:
            x = block(x)
        return self.pred_norm(x)


# ========================= VICReg Model (same architecture) ==========================

class LeWMVICRegModel(nn.Module):
    """LeWM with VICReg (no SIGReg buffers), matching SIGReg history_size=3."""
    def __init__(self, embed_dim=192, n_layers=4, num_heads=8, mlp_dim=512, history_size=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.history_size = history_size
        self.encoder = PixelEncoder(embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, history_size, embed_dim) * 0.02)
        self.predictor_blocks = nn.ModuleList([
            SimplePredictorBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(n_layers)
        ])
        self.pred_norm = nn.LayerNorm(embed_dim)

    def encode(self, frames):
        """frames: (B, T, H, W, C) uint8 -> (B, T, D)"""
        x = frames.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        return self.encoder(x)

    def encode_normalized(self, x):
        """x: (B, T, C, H, W) float [0,1] -> (B, T, D)"""
        return self.encoder(x)


# ========================= Data Loading ==========================

import h5py

def load_hdf5_data(h5_path, n_traj=None, seed=42):
    with h5py.File(h5_path, 'r') as f:
        pixels = f['pixels'][:]
        joint_angles = f['joint_angles'][:]
        com_velocity = f['com_velocity'][:]
        physics_labels = f['physics_labels'][:]

    if n_traj is not None and n_traj < len(pixels):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(pixels), n_traj, replace=False)
        pixels = pixels[idx]
        joint_angles = joint_angles[idx]
        com_velocity = com_velocity[idx]
        physics_labels = physics_labels[idx]

    return {
        'pixels': pixels,
        'joint_angles': joint_angles,
        'com_velocity': com_velocity,
        'physics_labels': physics_labels,
    }


# ========================= VICReg Loss ==========================

import torch.nn.functional as F


class VICReg(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, gamma=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    def forward(self, z, z_target):
        if z.dim() == 3:
            B, T, D = z.shape
            z = z.reshape(B * T, D)
            z_target = z_target.reshape(B * T, D)
        N, D = z.shape
        inv_loss = F.mse_loss(z, z_target)

        def variance_loss(x):
            x = x - x.mean(dim=0, keepdim=True)
            std = x.std(dim=0)
            return F.relu(self.gamma - std).mean()

        var_loss = (variance_loss(z) + variance_loss(z_target)) / 2.0

        def covariance_loss(x):
            x = x - x.mean(dim=0, keepdim=True)
            cov = (x.T @ x) / (N - 1)
            diag_mask = torch.eye(D, device=x.device, dtype=torch.bool)
            off_diag = cov[~diag_mask]
            return off_diag.pow(2).sum() / D

        cov_loss = (covariance_loss(z) + covariance_loss(z_target)) / 2.0
        total_loss = self.sim_coeff * inv_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss
        return total_loss, {'inv_loss': inv_loss.item(), 'var_loss': var_loss.item(), 'cov_loss': cov_loss.item()}


# ========================= Embedding Extraction ==========================

def extract_embeddings(model, pixels_array, batch_size=32, device=DEVICE, use_uint8_encode=True):
    """
    Extract encoder embeddings from pixel frames.
    pixels_array: (N, T, H, W, C) uint8
    Returns: (N*T, D) embeddings (flattened across trajectories and timesteps)
    """
    model.eval()
    all_embs = []
    n = len(pixels_array)

    for i in range(0, n, batch_size):
        batch_pix = pixels_array[i:i + batch_size]  # (B, T, H, W, C)
        B, T, H, W, C = batch_pix.shape
        x = torch.from_numpy(batch_pix).to(device)  # keep as uint8 for .encode()

        with torch.no_grad():
            if use_uint8_encode:
                emb = model.encode(x)  # encode() handles uint8 -> float internally
            else:
                x_f = x.float() / 255.0
                x_f = x_f.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
                emb = model.encode_normalized(x_f)
        all_embs.append(emb.reshape(B * T, -1).cpu().numpy())

    return np.concatenate(all_embs, axis=0)


# ========================= Geometric Analysis ==========================

from scipy.linalg import subspace_angles


def compute_principal_angles(A, B, n_components=8):
    """
    Compute principal angles between row-spaces of A and B.
    A, B: (N, D) arrays of embeddings

    Returns:
      angles: array of principal angles in radians
      mean_cosine_sim: mean cos(angle) — higher = more similar subspaces
    """
    # Center
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)

    # PCA to get top-k principal components
    from sklearn.decomposition import PCA
    k = min(n_components, A.shape[0] - 1, B.shape[0] - 1, A.shape[1])
    if k < 1:
        return np.array([np.pi / 2]), 0.0

    pca_a = PCA(n_components=k).fit(A)
    pca_b = PCA(n_components=k).fit(B)

    V_A = pca_a.components_  # (k, D)
    V_B = pca_b.components_  # (k, D)

    # scipy subspace_angles expects columns; transpose
    angles = subspace_angles(V_A.T, V_B.T)  # angles in [0, pi/2]
    mean_cosine_sim = float(np.cos(angles).mean())

    return angles.tolist(), mean_cosine_sim


def compute_displacement_consistency(embeddings_by_combo, factor_name, factor_levels,
                                     reference_factor='friction', n_components=8):
    """
    Displacement vector consistency (parallelogram test).

    For each factor, compute delta-vectors at multiple reference points
    (different values of the OTHER factors that are held constant), then
    measure cosine similarity between delta vectors.

    In PILOT mode: single factor (friction) with 3 levels, other factors fixed.
    We compute delta(low->mid), delta(low->high), delta(mid->high) at the single
    reference point we have (gravity=1.0g, mass=1.0x).

    Returns:
      consistency_score: mean pairwise cosine similarity of delta vectors
      delta_vectors: list of delta vectors
    """
    # Collect mean embeddings per factor level
    level_centroids = {}
    for lvl in factor_levels:
        key = f'g1.0_f{lvl}_m1.0'
        if key in embeddings_by_combo:
            level_centroids[lvl] = embeddings_by_combo[key].mean(axis=0)

    if len(level_centroids) < 2:
        return 0.0, []

    levels = sorted(level_centroids.keys())
    delta_vectors = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            delta = level_centroids[levels[j]] - level_centroids[levels[i]]
            norm = np.linalg.norm(delta)
            if norm > 1e-8:
                delta_vectors.append(delta / norm)

    if len(delta_vectors) < 2:
        return 1.0 if len(delta_vectors) == 1 else 0.0, [dv.tolist() for dv in delta_vectors]

    cosines = []
    for i in range(len(delta_vectors)):
        for j in range(i + 1, len(delta_vectors)):
            c = float(np.dot(delta_vectors[i], delta_vectors[j]))
            cosines.append(c)

    consistency_score = float(np.mean(cosines))
    return consistency_score, [dv.tolist() for dv in delta_vectors]


def compute_cka(X, Y):
    """
    Linear Centered Kernel Alignment (CKA) between X and Y.
    X: (N, D1), Y: (N, D2)
    Returns scalar CKA in [0, 1].
    """
    n = X.shape[0]
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Gram matrices
    K = X @ X.T  # (N, N)
    L = Y @ Y.T  # (N, N)

    # HSIC
    def hsic(A, B):
        # Center gram matrices
        n_ = A.shape[0]
        H = np.eye(n_) - np.ones((n_, n_)) / n_
        A_c = H @ A @ H
        B_c = H @ B @ H
        return np.trace(A_c @ B_c) / (n_ - 1) ** 2

    hsic_xy = hsic(K, L)
    hsic_xx = hsic(K, K)
    hsic_yy = hsic(L, L)

    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0

    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


# ========================= Quick VICReg Training for Pilot ==========================

def train_vicreg_pilot(pixels_train, device, seed=42, n_epochs=5, batch_size=32,
                       embed_dim=192, n_layers=4, history_size=3):
    """
    Train a VICReg model for 5 epochs on pilot data (for comparison with SIGReg).
    Uses sliding window of history_size frames to match SIGReg training protocol.
    Returns: trained model, loss_history
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LeWMVICRegModel(embed_dim=embed_dim, n_layers=n_layers, history_size=history_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    vicreg_loss_fn = VICReg().to(device)

    n = len(pixels_train)
    loss_history = []

    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        epoch_losses = []

        for i in range(0, n - batch_size + 1, batch_size):
            batch_idx = idx[i:i + batch_size]
            batch_pix = pixels_train[batch_idx]  # (B, T, H, W, C) uint8

            # Use first history_size+1 frames for training context+target
            T_use = min(history_size + 1, batch_pix.shape[1])
            batch_pix_slice = batch_pix[:, :T_use, :, :, :]  # (B, T_use, H, W, C)

            x = torch.from_numpy(batch_pix_slice).to(device)  # keep uint8

            optimizer.zero_grad()
            z = model.encode(x)  # (B, T_use, D)

            # Context frames vs target frame
            z_ctx = z[:, :history_size, :]  # (B, history_size, D)
            z_tgt = z[:, -1:, :]            # (B, 1, D)
            loss, _ = vicreg_loss_fn(z_ctx.reshape(z_ctx.shape[0], -1),
                                     z_tgt.expand_as(z_ctx).reshape(z_ctx.shape[0], -1)
                                     if z_ctx.shape == z_tgt.expand_as(z_ctx).shape
                                     else z_tgt.repeat(1, history_size, 1).reshape(z_ctx.shape[0], -1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        loss_history.append(mean_loss)
        print(f'  VICReg epoch {epoch+1}/{n_epochs}: loss={mean_loss:.4f}')

    return model, loss_history


# ========================= Main ==========================

def main():
    write_pid()
    write_progress(epoch=0, total_epochs=5, step=0, total_steps=5, metric={'phase': 'init'})

    result = {
        'task_id': TASK_ID,
        'timestamp': datetime.now().isoformat(),
        'mode': 'PILOT',
        'pilot_config': {
            'seed': 42,
            'n_traj': 100,
            'n_epochs_vicreg': 5,
            'description': 'Principal angle analysis on pilot checkpoint; necessary-compositionality fallback to scipy subspace_angles',
        },
        'necessary_compositionality': {'available': False, 'fallback': 'scipy.linalg.subspace_angles'},
        'principal_angles': {},
        'displacement_consistency': {},
        'cka': {},
        'pass_criteria': {},
        'go_no_go': 'NO_GO',
        'status': 'error',
    }

    # Check necessary-compositionality availability
    try:
        from necessary_compositionality import principal_angles as nc_principal_angles
        result['necessary_compositionality']['available'] = True
        print('[INFO] necessary-compositionality available')
    except ImportError:
        print('[INFO] necessary-compositionality not installed; using scipy.linalg.subspace_angles fallback')

    # ---- Step 1: Load pilot data (3 friction combos) ----
    print('\n[STEP 1] Loading pilot data...')
    combos = ['g1.0_f0.5_m1.0', 'g1.0_f1.0_m1.0', 'g1.0_f2.0_m1.0']
    friction_levels = [0.5, 1.0, 2.0]
    combo_data = {}

    for combo in combos:
        h5_path = DATA_DIR / f'{combo}.h5'
        if not h5_path.exists():
            raise FileNotFoundError(f'Pilot data not found: {h5_path}')
        data = load_hdf5_data(h5_path, n_traj=100, seed=42)
        combo_data[combo] = data
        print(f'  Loaded {combo}: {data["pixels"].shape[0]} traj, shape {data["pixels"].shape}')

    # Combine all pixels for VICReg training (training combos = f0.5, f2.0; holdout = f1.0)
    train_combos = ['g1.0_f0.5_m1.0', 'g1.0_f2.0_m1.0']
    pixels_train = np.concatenate([combo_data[c]['pixels'] for c in train_combos], axis=0)
    print(f'  Training pixels combined: {pixels_train.shape}')

    write_progress(epoch=1, total_epochs=5, step=1, total_steps=5, metric={'phase': 'data_loaded'})

    # ---- Step 2: Load SIGReg checkpoint ----
    print('\n[STEP 2] Loading SIGReg checkpoint...')
    sigreg_ckpt_path = FULL_DIR / 'lewm_sigreg_primary' / 'pilot_seed42_epoch5.pt'
    ckpt = torch.load(sigreg_ckpt_path, map_location='cpu')
    config = ckpt.get('config', {})

    # Use pos_embed shape from checkpoint to derive history_size
    pos_embed_shape = ckpt['model_state_dict']['pos_embed'].shape  # (1, T, D)
    history_size = pos_embed_shape[1]
    print(f'  Detected history_size={history_size} from checkpoint pos_embed shape')

    sigreg_model = LeWMModel(
        embed_dim=config.get('embed_dim', 192),
        n_layers=config.get('n_layers', 4),
        num_heads=config.get('num_heads', 8),
        mlp_dim=config.get('mlp_dim', 512),
        sigreg_knots=config.get('sigreg_knots', 17),
        sigreg_num_proj=config.get('sigreg_num_proj', 1024),
        sigreg_weight=config.get('sigreg_weight', 0.09),
        history_size=history_size,
    ).to(DEVICE)

    sigreg_model.load_state_dict(ckpt['model_state_dict'], strict=True)
    sigreg_model.eval()
    print(f'  SIGReg checkpoint loaded: epoch={ckpt["epoch"]}, loss={ckpt["final_loss"]:.4f}')

    result['sigreg_checkpoint'] = {
        'path': str(sigreg_ckpt_path),
        'epoch': ckpt['epoch'],
        'final_loss': ckpt['final_loss'],
    }

    write_progress(epoch=2, total_epochs=5, step=2, total_steps=5, metric={'phase': 'sigreg_loaded'})

    # ---- Step 3: Train VICReg pilot (or load if already trained) ----
    print('\n[STEP 3] Training/Loading VICReg pilot model...')
    vicreg_ckpt_path = PILOTS_DIR / 'vicreg_pilot_seed42_epoch5.pt'

    if vicreg_ckpt_path.exists():
        print(f'  VICReg pilot checkpoint already exists: {vicreg_ckpt_path}')
        vicreg_ckpt = torch.load(vicreg_ckpt_path, map_location='cpu')
        _hs = vicreg_ckpt.get('history_size', history_size)
        vicreg_model = LeWMVICRegModel(embed_dim=192, n_layers=4, history_size=_hs).to(DEVICE)
        vicreg_model.load_state_dict(vicreg_ckpt['model_state_dict'])
        vicreg_model.eval()
        vicreg_loss_history = vicreg_ckpt.get('loss_history', [])
        fl = vicreg_loss_history[-1] if vicreg_loss_history else 0.0
        print(f'  VICReg loaded, final_loss={fl:.4f}')
    else:
        print('  Training VICReg pilot model (5 epochs)...')
        t_vicreg = time.time()
        vicreg_model, vicreg_loss_history = train_vicreg_pilot(
            pixels_train=pixels_train,
            device=DEVICE,
            seed=42,
            n_epochs=5,
            batch_size=32,
            embed_dim=192,
            n_layers=4,
            history_size=history_size,
        )
        vicreg_model.eval()
        vicreg_time = time.time() - t_vicreg
        print(f'  VICReg training done in {vicreg_time:.1f}s, final_loss={vicreg_loss_history[-1]:.4f}')

        # Save VICReg checkpoint for reuse
        torch.save({
            'epoch': 5,
            'model_state_dict': vicreg_model.state_dict(),
            'final_loss': vicreg_loss_history[-1],
            'loss_history': vicreg_loss_history,
            'history_size': history_size,
        }, vicreg_ckpt_path)
        print(f'  VICReg pilot checkpoint saved: {vicreg_ckpt_path}')

    result['vicreg_training'] = {
        'loss_history': vicreg_loss_history,
        'final_loss': vicreg_loss_history[-1] if vicreg_loss_history else None,
    }

    write_progress(epoch=3, total_epochs=5, step=3, total_steps=5, metric={'phase': 'vicreg_ready'})

    # ---- Step 4: Extract embeddings for all 3 combos ----
    print('\n[STEP 4] Extracting encoder embeddings...')
    sigreg_embeddings = {}
    vicreg_embeddings = {}

    for combo in combos:
        pix = combo_data[combo]['pixels']  # (N, T, H, W, C)
        print(f'  Extracting SIGReg embeddings for {combo}...')
        emb_sigreg = extract_embeddings(sigreg_model, pix, batch_size=32, device=DEVICE)
        sigreg_embeddings[combo] = emb_sigreg
        print(f'    SIGReg embeddings: {emb_sigreg.shape}')

        print(f'  Extracting VICReg embeddings for {combo}...')
        emb_vicreg = extract_embeddings(vicreg_model, pix, batch_size=32, device=DEVICE)
        vicreg_embeddings[combo] = emb_vicreg
        print(f'    VICReg embeddings: {emb_vicreg.shape}')

    result['embedding_shapes'] = {
        'sigreg': {c: list(sigreg_embeddings[c].shape) for c in combos},
        'vicreg': {c: list(vicreg_embeddings[c].shape) for c in combos},
    }

    write_progress(epoch=4, total_epochs=5, step=4, total_steps=5, metric={'phase': 'embeddings_extracted'})

    # ---- Step 5: Principal Angle Analysis ----
    print('\n[STEP 5] Computing principal angles between factor subspaces...')

    # Friction levels: 0.5x, 1.0x, 2.0x
    friction_combo_keys = {0.5: 'g1.0_f0.5_m1.0', 1.0: 'g1.0_f1.0_m1.0', 2.0: 'g1.0_f2.0_m1.0'}
    level_names = ['friction_0.5x', 'friction_1.0x', 'friction_2.0x']
    level_keys = list(friction_combo_keys.values())

    # Build 3x3 principal angle matrix for SIGReg
    pa_sigreg = {}
    pa_vicreg = {}
    mean_cos_sigreg = np.zeros((3, 3))
    mean_cos_vicreg = np.zeros((3, 3))

    for i, ki in enumerate(level_keys):
        for j, kj in enumerate(level_keys):
            if i == j:
                mean_cos_sigreg[i, j] = 1.0
                mean_cos_vicreg[i, j] = 1.0
                continue
            if j < i:
                mean_cos_sigreg[i, j] = mean_cos_sigreg[j, i]
                mean_cos_vicreg[i, j] = mean_cos_vicreg[j, i]
                continue

            A_sig = sigreg_embeddings[ki]
            B_sig = sigreg_embeddings[kj]
            angles_sig, mcs_sig = compute_principal_angles(A_sig, B_sig, n_components=8)
            mean_cos_sigreg[i, j] = mcs_sig
            mean_cos_sigreg[j, i] = mcs_sig

            A_vic = vicreg_embeddings[ki]
            B_vic = vicreg_embeddings[kj]
            angles_vic, mcs_vic = compute_principal_angles(A_vic, B_vic, n_components=8)
            mean_cos_vicreg[i, j] = mcs_vic
            mean_cos_vicreg[j, i] = mcs_vic

            pair_key = f'{level_names[i]}_vs_{level_names[j]}'
            pa_sigreg[pair_key] = {
                'principal_angles_rad': angles_sig,
                'mean_cosine_similarity': mcs_sig,
            }
            pa_vicreg[pair_key] = {
                'principal_angles_rad': angles_vic,
                'mean_cosine_similarity': mcs_vic,
            }
            print(f'  {pair_key}: SIGReg mean_cos={mcs_sig:.4f}, VICReg mean_cos={mcs_vic:.4f}')

    # Summary stats
    # Off-diagonal mean cosine similarity (lower = more orthogonal)
    mask = ~np.eye(3, dtype=bool)
    off_diag_mean_sigreg = float(mean_cos_sigreg[mask].mean())
    off_diag_mean_vicreg = float(mean_cos_vicreg[mask].mean())

    print(f'\n  SIGReg mean off-diagonal cosine similarity: {off_diag_mean_sigreg:.4f}')
    print(f'  VICReg mean off-diagonal cosine similarity: {off_diag_mean_vicreg:.4f}')

    result['principal_angles'] = {
        'SIGReg': {
            'pairwise': pa_sigreg,
            'mean_cosine_matrix': mean_cos_sigreg.tolist(),
            'off_diag_mean_cosine': off_diag_mean_sigreg,
        },
        'VICReg': {
            'pairwise': pa_vicreg,
            'mean_cosine_matrix': mean_cos_vicreg.tolist(),
            'off_diag_mean_cosine': off_diag_mean_vicreg,
        },
        'h2_comparison': {
            'sigreg_more_orthogonal': bool(off_diag_mean_sigreg < off_diag_mean_vicreg),
            'delta': float(off_diag_mean_vicreg - off_diag_mean_sigreg),
            'note': 'Positive delta means SIGReg is more orthogonal (lower cosine similarity = larger principal angles)',
        },
    }

    # ---- Step 6: Displacement Vector Consistency (Parallelogram Test) ----
    print('\n[STEP 6] Computing displacement vector consistency...')

    # Note: in pilot we have only 1 reference point (gravity=1.0g, mass=1.0x)
    # so we can compute delta vectors between friction levels at that single reference
    consistency_sigreg, deltas_sig = compute_displacement_consistency(
        sigreg_embeddings, factor_name='friction',
        factor_levels=friction_levels,
    )
    consistency_vicreg, deltas_vic = compute_displacement_consistency(
        vicreg_embeddings, factor_name='friction',
        factor_levels=friction_levels,
    )
    print(f'  SIGReg displacement consistency: {consistency_sigreg:.4f}')
    print(f'  VICReg displacement consistency: {consistency_vicreg:.4f}')

    result['displacement_consistency'] = {
        'SIGReg': {
            'consistency_score': consistency_sigreg,
            'n_delta_vectors': len(deltas_sig),
            'note': 'Cosine similarity between delta vectors at different reference points; higher = more consistent/linear factor structure',
        },
        'VICReg': {
            'consistency_score': consistency_vicreg,
            'n_delta_vectors': len(deltas_vic),
        },
        'h4_note': 'Pilot: only 1 reference point available; consistency computed between friction delta vectors (f0.5->f1.0, f0.5->f2.0, f1.0->f2.0). Full study will use 3+ reference points.',
    }

    # ---- Step 7: CKA between factor-level groups ----
    print('\n[STEP 7] Computing CKA between factor-grouped representations...')

    cka_results_sigreg = {}
    cka_results_vicreg = {}

    for i, ki in enumerate(level_keys):
        for j, kj in enumerate(level_keys):
            if j <= i:
                continue
            pair_key = f'{level_names[i]}_vs_{level_names[j]}'

            # Use random subsets to keep CKA tractable
            n_cka = min(200, len(sigreg_embeddings[ki]), len(sigreg_embeddings[kj]))
            rng = np.random.RandomState(42)

            A_sig = sigreg_embeddings[ki][rng.choice(len(sigreg_embeddings[ki]), n_cka, replace=False)]
            B_sig = sigreg_embeddings[kj][rng.choice(len(sigreg_embeddings[kj]), n_cka, replace=False)]
            cka_sig = compute_cka(A_sig, B_sig)

            A_vic = vicreg_embeddings[ki][rng.choice(len(vicreg_embeddings[ki]), n_cka, replace=False)]
            B_vic = vicreg_embeddings[kj][rng.choice(len(vicreg_embeddings[kj]), n_cka, replace=False)]
            cka_vic = compute_cka(A_vic, B_vic)

            cka_results_sigreg[pair_key] = cka_sig
            cka_results_vicreg[pair_key] = cka_vic
            print(f'  CKA {pair_key}: SIGReg={cka_sig:.4f}, VICReg={cka_vic:.4f}')

    result['cka'] = {
        'SIGReg': cka_results_sigreg,
        'VICReg': cka_results_vicreg,
        'n_samples_per_group': n_cka,
        'note': 'Linear CKA between factor-level embedding groups. Higher CKA = more similar representations across levels.',
    }

    write_progress(epoch=5, total_epochs=5, step=5, total_steps=5, metric={'phase': 'analysis_done'})

    # ---- Step 8: Pass Criteria ----
    print('\n[STEP 8] Evaluating pass criteria...')

    pa_matrix_valid = (
        np.array(result['principal_angles']['SIGReg']['mean_cosine_matrix']).shape == (3, 3) and
        not np.any(np.isnan(np.array(result['principal_angles']['SIGReg']['mean_cosine_matrix']))) and
        float(off_diag_mean_sigreg) >= 0.0 and float(off_diag_mean_sigreg) <= 1.0
    )

    no_nan_in_results = (
        not np.isnan(consistency_sigreg) and
        not np.isnan(consistency_vicreg) and
        not np.isnan(off_diag_mean_sigreg) and
        not np.isnan(off_diag_mean_vicreg)
    )

    scipy_fallback_works = True  # We got here without error

    all_pass = pa_matrix_valid and no_nan_in_results and scipy_fallback_works

    result['pass_criteria'] = {
        'principal_angle_matrix_3x3': pa_matrix_valid,
        'mean_cosine_sim_in_0_1': bool(0.0 <= off_diag_mean_sigreg <= 1.0),
        'no_nan_in_results': no_nan_in_results,
        'scipy_fallback_works': scipy_fallback_works,
        'all_pass': all_pass,
    }

    result['go_no_go'] = 'GO' if all_pass else 'NO_GO'
    result['status'] = 'success'
    result['total_time_sec'] = round(time.time() - t_start, 1)
    result['total_time_min'] = round(result['total_time_sec'] / 60, 2)

    # ---- Summary ----
    print('\n========== GEOMETRIC ANALYSIS PILOT SUMMARY ==========')
    print(f'SIGReg off-diag mean cosine: {off_diag_mean_sigreg:.4f}')
    print(f'VICReg off-diag mean cosine: {off_diag_mean_vicreg:.4f}')
    print(f'H2 (SIGReg more orthogonal): {result["principal_angles"]["h2_comparison"]["sigreg_more_orthogonal"]}')
    print(f'SIGReg displacement consistency: {consistency_sigreg:.4f}')
    print(f'VICReg displacement consistency: {consistency_vicreg:.4f}')
    print(f'All pass criteria: {all_pass}')
    print(f'GO/NO-GO: {result["go_no_go"]}')
    print(f'Time: {result["total_time_sec"]:.1f}s')
    print('=======================================================')

    # ---- Save Results ----
    # Pilot result
    pilot_out = PILOTS_DIR / 'eval_geometric_h2_h4_pilot.json'
    pilot_out.write_text(json.dumps(result, indent=2))
    print(f'\n[SAVE] Pilot result: {pilot_out}')

    # Full dir result (pilot-tier)
    full_out = FULL_DIR / 'geometric_h2_h4_results.json'
    full_out.write_text(json.dumps(result, indent=2))
    print(f'[SAVE] Full result: {full_out}')

    write_done(status='success', summary=f'Geometric analysis pilot complete. GO={result["go_no_go"]}. SIGReg mean_cos={off_diag_mean_sigreg:.4f}, VICReg mean_cos={off_diag_mean_vicreg:.4f}. Displacement consistency SIGReg={consistency_sigreg:.4f}')

    # ---- Update gpu_progress.json ----
    gpu_progress_path = WORKSPACE / 'exp' / 'gpu_progress.json'
    try:
        if gpu_progress_path.exists():
            gp = json.loads(gpu_progress_path.read_text())
        else:
            gp = {'completed': [], 'failed': [], 'running': {}, 'timings': {}}
        if TASK_ID not in gp.get('completed', []):
            gp.setdefault('completed', []).append(TASK_ID)
        gp.setdefault('running', {}).pop(TASK_ID, None)
        gp.setdefault('timings', {})[TASK_ID] = {
            'planned_min': 30,
            'actual_min': int(result['total_time_min']),
            'start_time': result['timestamp'],
            'end_time': datetime.now().isoformat(),
            'config_snapshot': {
                'model': 'LeWM-SIGReg+VICReg',
                'embed_dim': 192,
                'n_components_pca': 8,
                'n_combos': 3,
                'n_traj_per_combo': 100,
                'gpu_model': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                'gpu_count': 1,
            }
        }
        gpu_progress_path.write_text(json.dumps(gp, indent=2))
        print(f'[GPU_PROGRESS] Updated {gpu_progress_path}')
    except Exception as e:
        print(f'[WARN] Could not update gpu_progress.json: {e}')

    # ---- Update experiment_state.json ----
    exp_state_path = WORKSPACE / 'exp' / 'experiment_state.json'
    try:
        if exp_state_path.exists():
            es = json.loads(exp_state_path.read_text())
        else:
            es = {'schema_version': 1, 'tasks': {}}
        es.setdefault('tasks', {}).setdefault(TASK_ID, {})['status'] = 'completed'
        es['tasks'][TASK_ID]['completed_at'] = datetime.now().isoformat()
        es['tasks'][TASK_ID]['result_path'] = str(pilot_out.relative_to(WORKSPACE))
        exp_state_path.write_text(json.dumps(es, indent=2))
        print(f'[EXP_STATE] Updated {exp_state_path}')
    except Exception as e:
        print(f'[WARN] Could not update experiment_state.json: {e}')

    return result


if __name__ == '__main__':
    try:
        result = main()
        print(f'\n[SUCCESS] eval_geometric_h2_h4 pilot complete. GO/NO-GO: {result["go_no_go"]}')
        sys.exit(0)
    except Exception as e:
        err = traceback.format_exc()
        print(f'[ERROR] {e}\n{err}')
        write_done(status='failure', summary=str(e))
        sys.exit(1)
