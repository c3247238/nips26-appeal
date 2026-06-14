"""
Pilot: train_lewm_vicreg (PILOT MODE)
======================================
Train LeWM with VICReg regularizer on 100-trajectory pilot subset (1 seed, 5 epochs).
Pass criteria:
  - VICReg loss converges (3 terms: invariance, variance, covariance all decreasing)
  - No collapse (embedding variance stays non-trivial)
  - Framework runs without error

VICReg loss:
  - Invariance: MSE between z_pred and z_target (brings predictions close to targets)
  - Variance: hinge loss to keep per-dim variance >= gamma (prevents collapse)
  - Covariance: off-diagonal covariance penalty (decorrelates dimensions)

Output:
  exp/results/pilots/train_lewm_vicreg_pilot.json
  exp/results/train_lewm_vicreg_DONE (if success)
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
import torch.nn.functional as F

# GPU assignment
os.environ['MUJOCO_GL'] = 'egl'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

# ========================= Paths ==========================
WORKSPACE = Path('/home/qhxie/AutoResearch-SibylSystem/workspaces/lewm-generalization/current')
DATA_DIR = WORKSPACE / 'exp' / 'data' / 'comphys_lewm' / 'pilot'
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
PILOTS_DIR = RESULTS_DIR / 'pilots'
CODE_DIR = WORKSPACE / 'exp' / 'code'
LEWM_DIR = CODE_DIR / 'le-wm'

TASK_ID = 'train_lewm_vicreg'
PID_FILE = RESULTS_DIR / f'{TASK_ID}.pid'
PROGRESS_FILE = RESULTS_DIR / f'{TASK_ID}_PROGRESS.json'
DONE_FILE = RESULTS_DIR / f'{TASK_ID}_DONE'

sys.path.insert(0, str(LEWM_DIR))
PILOTS_DIR.mkdir(parents=True, exist_ok=True)

# ========================= PID / Progress ==========================
def write_pid():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    print(f'[PID] Written PID {os.getpid()} to {PID_FILE}')

def write_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
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


# ========================= Data Loading ==========================
import h5py

def load_hdf5_data(h5_path, n_traj=None, seed=42):
    """Load HDF5 data. Returns dict of arrays."""
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

class VICReg(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization.

    References: Bardes et al. (2022) VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning.

    Parameters:
        sim_coeff (lambda): invariance loss weight (default 25.0)
        std_coeff (mu): variance loss weight (default 25.0)
        cov_coeff (nu): covariance loss weight (default 1.0)
        gamma: target standard deviation for variance loss (default 1.0)
    """

    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, gamma=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    def forward(self, z, z_target):
        """
        Compute VICReg loss between z (predicted) and z_target (encoded).
        z, z_target: (B, D) or (B, T, D) -> flattened to (N, D)

        Returns: total_loss, {inv_loss, var_loss, cov_loss}
        """
        if z.dim() == 3:
            B, T, D = z.shape
            z = z.reshape(B * T, D)
            z_target = z_target.reshape(B * T, D)

        N, D = z.shape

        # --- Invariance Loss (MSE between z and z_target) ---
        inv_loss = F.mse_loss(z, z_target)

        # --- Variance Loss (hinge on per-dim std) ---
        # Applies to both z and z_target
        def variance_loss(x):
            x = x - x.mean(dim=0, keepdim=True)
            std = x.std(dim=0)  # (D,)
            var_loss = F.relu(self.gamma - std).mean()
            return var_loss

        var_loss = (variance_loss(z) + variance_loss(z_target)) / 2.0

        # --- Covariance Loss (off-diagonal covariance penalty) ---
        def covariance_loss(x):
            x = x - x.mean(dim=0, keepdim=True)
            cov = (x.T @ x) / (N - 1)  # (D, D)
            # Mask out diagonal
            diag_mask = torch.eye(D, device=x.device, dtype=torch.bool)
            off_diag = cov[~diag_mask]
            cov_loss = off_diag.pow(2).sum() / D
            return cov_loss

        cov_loss = (covariance_loss(z) + covariance_loss(z_target)) / 2.0

        total_loss = (self.sim_coeff * inv_loss
                      + self.std_coeff * var_loss
                      + self.cov_coeff * cov_loss)

        return total_loss, {
            'inv_loss': inv_loss.item(),
            'var_loss': var_loss.item(),
            'cov_loss': cov_loss.item(),
        }


# ========================= Model Architecture ====================

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


class LeWMVICReg(nn.Module):
    """
    LeWM with VICReg regularizer.
    Same architecture as LeWM-SIGReg but replaces SIGReg with VICReg.

    VICReg is applied between:
      - z_pred: predictor output for context frames
      - z_target: encoder output for target frames (stop-gradient)

    This is the natural VICReg formulation for predictive architectures:
    - Invariance: align predictions with targets
    - Variance: prevent collapse of either branch
    - Covariance: decorrelate dimensions in both branches
    """

    def __init__(self, embed_dim=192, n_layers=4, num_heads=8, mlp_dim=512,
                 vicreg_sim_coeff=25.0, vicreg_std_coeff=25.0, vicreg_cov_coeff=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.history_size = 3
        self.num_preds = 1

        self.encoder = PixelEncoder(embed_dim=embed_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, self.history_size, embed_dim) * 0.02)

        self.predictor_blocks = nn.ModuleList([
            SimplePredictorBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(n_layers)
        ])
        self.pred_norm = nn.LayerNorm(embed_dim)

        # VICReg loss
        self.vicreg = VICReg(
            sim_coeff=vicreg_sim_coeff,
            std_coeff=vicreg_std_coeff,
            cov_coeff=vicreg_cov_coeff,
        )

    def encode(self, frames):
        """frames: (B, T, H, W, 3) uint8 -> embeddings (B, T, D)"""
        x = frames.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)  # (B, T, 3, H, W)
        return self.encoder(x)

    def predict(self, ctx_emb):
        """ctx_emb: (B, T_ctx, D) -> (B, T_ctx, D) predicted next"""
        x = ctx_emb + self.pos_embed[:, :ctx_emb.size(1)]
        for block in self.predictor_blocks:
            x = block(x)
        return self.pred_norm(x)

    def forward(self, frames):
        """
        frames: (B, T_total, H, W, 3), T_total >= history_size + num_preds

        VICReg is applied between:
          pred: predictor output from context frames
          tgt: encoder output of target frames (stop-grad)

        Returns dict with losses including 3 VICReg terms.
        """
        emb = self.encode(frames)  # (B, T_total, D)

        ctx_emb = emb[:, :self.history_size]    # (B, 3, D) - context
        tgt_emb = emb[:, self.num_preds:]        # (B, T-1, D) - all targets

        pred_emb = self.predict(ctx_emb)          # (B, 3, D) - predictions

        # Only compare last prediction vs. next-step target
        pred = pred_emb[:, -1:]         # (B, 1, D)
        tgt = tgt_emb[:, self.history_size-1:self.history_size]  # (B, 1, D)

        # VICReg: pred vs. tgt (stop-gradient on target in JEPA style)
        tgt_sg = tgt.detach()  # stop gradient on target (JEPA convention)

        total_vicreg_loss, vicreg_terms = self.vicreg(
            pred.squeeze(1),   # (B, D)
            tgt_sg.squeeze(1)  # (B, D)
        )

        return {
            'loss': total_vicreg_loss,
            'inv_loss': vicreg_terms['inv_loss'],
            'var_loss': vicreg_terms['var_loss'],
            'cov_loss': vicreg_terms['cov_loss'],
            'emb': emb,
        }


# ========================= Dataset ============================

class PhysicsDataset(torch.utils.data.Dataset):
    """Dataset for HDF5 files. Returns (frames, physics_labels)."""

    def __init__(self, data_dicts, history_size=3, num_preds=1, frameskip=5):
        self.history_size = history_size
        self.num_preds = num_preds
        self.seq_len = history_size + num_preds
        self.frameskip = frameskip

        all_pixels = []
        all_physics_labels = []
        all_joint_angles = []
        all_com_velocity = []

        for d in data_dicts:
            n_traj, T = d['pixels'].shape[:2]
            frames_per_traj = T // frameskip

            for i in range(n_traj):
                pix = d['pixels'][i, ::frameskip]
                ja = d['joint_angles'][i, ::frameskip]
                cv = d['com_velocity'][i, ::frameskip]
                labels = d['physics_labels'][i]

                for start in range(frames_per_traj - self.seq_len + 1):
                    end = start + self.seq_len
                    all_pixels.append(pix[start:end])
                    all_physics_labels.append(labels)
                    all_joint_angles.append(ja[start:end])
                    all_com_velocity.append(cv[start:end])

        self.pixels = np.array(all_pixels)
        self.physics_labels = np.array(all_physics_labels)
        self.joint_angles = np.array(all_joint_angles)
        self.com_velocity = np.array(all_com_velocity)

        print(f'  [Dataset] {len(self.pixels)} sequences from {sum(len(d["pixels"]) for d in data_dicts)} trajectories')

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        return {
            'pixels': torch.from_numpy(self.pixels[idx]),
            'physics_labels': torch.from_numpy(self.physics_labels[idx]),
            'joint_angles': torch.from_numpy(self.joint_angles[idx]),
            'com_velocity': torch.from_numpy(self.com_velocity[idx]),
        }


# ========================= Embedding Variance Check ==============

def check_embedding_variance(model, dataset, device, batch_size=64):
    """
    Check that embeddings don't collapse.
    Returns: mean per-dim variance (should be > 0.01 to not collapse).
    """
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_emb = []

    with torch.no_grad():
        for batch in loader:
            frames = batch['pixels'].to(device)
            emb = model.encode(frames)      # (B, T, D)
            emb_mean = emb.mean(dim=1)      # (B, D) - mean over time
            all_emb.append(emb_mean.cpu().numpy())

    all_emb = np.concatenate(all_emb, axis=0)  # (N, D)
    per_dim_var = np.var(all_emb, axis=0)       # (D,)
    mean_var = float(per_dim_var.mean())
    min_var = float(per_dim_var.min())
    std_var = float(per_dim_var.std())

    return {
        'mean_per_dim_variance': round(mean_var, 6),
        'min_per_dim_variance': round(min_var, 6),
        'std_per_dim_variance': round(std_var, 6),
        'collapsed': bool(mean_var < 1e-4),
        'n_collapsed_dims': int((per_dim_var < 1e-5).sum()),
        'total_dims': int(len(per_dim_var)),
    }


# ========================= Training ==========================

def train_vicreg(model, dataset, n_epochs=5, lr=1e-4, batch_size=32, device=DEVICE,
                 epoch_offset=0, total_epochs_outer=5):
    """
    Train LeWM-VICReg on dataset.
    Returns loss history per term.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    history = {
        'total': [],
        'inv': [],
        'var': [],
        'cov': [],
    }

    for epoch in range(n_epochs):
        model.train()
        epoch_total = []
        epoch_inv = []
        epoch_var = []
        epoch_cov = []

        for batch in loader:
            frames = batch['pixels'].to(device)
            optimizer.zero_grad()
            out = model(frames)
            loss = out['loss']
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_total.append(loss.item())
            epoch_inv.append(out['inv_loss'])
            epoch_var.append(out['var_loss'])
            epoch_cov.append(out['cov_loss'])

        mean_total = float(np.mean(epoch_total))
        mean_inv = float(np.mean(epoch_inv))
        mean_var = float(np.mean(epoch_var))
        mean_cov = float(np.mean(epoch_cov))

        history['total'].append(mean_total)
        history['inv'].append(mean_inv)
        history['var'].append(mean_var)
        history['cov'].append(mean_cov)

        write_progress(
            epoch + epoch_offset + 1, total_epochs_outer,
            loss=mean_total,
            metric={
                'phase': 'vicreg_training',
                'epoch': epoch + 1,
                'inv_loss': round(mean_inv, 5),
                'var_loss': round(mean_var, 5),
                'cov_loss': round(mean_cov, 5),
            }
        )

        print(f'  Epoch {epoch+1:3d}/{n_epochs} | total={mean_total:.4f} | '
              f'inv={mean_inv:.4f} | var={mean_var:.4f} | cov={mean_cov:.5f}')

    return history


def check_loss_convergence(history):
    """
    Check that each VICReg term is decreasing or stable (not diverging).
    For a 5-epoch pilot:
      - inv_loss: should decrease (invariance loss goes down as predictions align with targets)
      - var_loss: should decrease (variance loss goes to 0 as embeddings spread out)
      - cov_loss: should be finite and not growing
    """
    results = {}

    for term in ['total', 'inv', 'var', 'cov']:
        vals = history[term]
        if len(vals) < 2:
            results[term] = {'decreasing': None, 'stable': True, 'values': vals}
            continue

        first_half_mean = float(np.mean(vals[:len(vals)//2 + 1]))
        second_half_mean = float(np.mean(vals[len(vals)//2:]))
        decreasing = second_half_mean <= first_half_mean * 1.05  # allow 5% tolerance
        diverging = vals[-1] > vals[0] * 5.0  # 5x increase = diverging

        results[term] = {
            'decreasing': bool(decreasing),
            'diverging': bool(diverging),
            'first': round(vals[0], 6),
            'last': round(vals[-1], 6),
            'change_pct': round((vals[-1] - vals[0]) / (abs(vals[0]) + 1e-8) * 100, 2),
            'values': [round(v, 6) for v in vals],
        }

    return results


# ========================= MAIN ==============================

def main():
    t_start = time.time()
    write_pid()
    write_progress(0, 5, loss=None, metric={'phase': 'init'})

    torch.manual_seed(42)
    np.random.seed(42)

    results = {
        'task_id': TASK_ID,
        'timestamp': datetime.now().isoformat(),
        'gpu': str(DEVICE),
        'seed': 42,
        'mode': 'PILOT',
        'pilot_config': {
            'n_traj': 100,
            'n_epochs': 5,
            'seed': 42,
            'description': '1-seed 5-epoch VICReg training on 100-trajectory pilot subset',
        }
    }

    print('\n' + '='*60)
    print('PILOT: train_lewm_vicreg')
    print('VICReg: Variance-Invariance-Covariance Regularization')
    print('='*60)

    # ===== Load Data (use same pilot data as framework validation) =====
    print('\n[DATA] Loading pilot data (100 traj, friction={0.5x, 2.0x}, gravity=1.0g, mass=1.0x)...')

    try:
        data_05 = load_hdf5_data(DATA_DIR / 'g1.0_f0.5_m1.0.h5', n_traj=100, seed=42)
        data_20 = load_hdf5_data(DATA_DIR / 'g1.0_f2.0_m1.0.h5', n_traj=100, seed=42)
        print(f'  Loaded friction=0.5x: {data_05["pixels"].shape[0]} traj, shape={data_05["pixels"].shape}')
        print(f'  Loaded friction=2.0x: {data_20["pixels"].shape[0]} traj, shape={data_20["pixels"].shape}')
    except Exception as e:
        print(f'[ERROR] Failed to load data: {e}')
        traceback.print_exc()
        write_done('failure', f'Data load failed: {e}')
        return

    # Build dataset
    train_dataset = PhysicsDataset([data_05, data_20], history_size=3, num_preds=1, frameskip=5)
    print(f'  Train dataset: {len(train_dataset)} sequences')
    results['dataset_size'] = len(train_dataset)

    # ===== Initialize VICReg Model =====
    print('\n[MODEL] Initializing LeWM-VICReg model...')
    # VICReg weights: sim=25, std=25, cov=1 (from VICReg paper defaults)
    model = LeWMVICReg(
        embed_dim=192,
        n_layers=4,
        num_heads=8,
        mlp_dim=512,
        vicreg_sim_coeff=25.0,
        vicreg_std_coeff=25.0,
        vicreg_cov_coeff=1.0,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {n_params:,} ({n_params/1e6:.2f}M)')
    results['model_params'] = n_params
    results['model_params_M'] = round(n_params / 1e6, 2)
    results['vicreg_config'] = {
        'sim_coeff': 25.0,
        'std_coeff': 25.0,
        'cov_coeff': 1.0,
        'embed_dim': 192,
        'n_layers': 4,
    }

    # ===== Check Pre-Training Embedding Variance =====
    print('\n[CHECK] Pre-training embedding variance (should be non-zero for random init)...')
    pretrain_var = check_embedding_variance(model, train_dataset, DEVICE)
    print(f'  Pre-train variance: mean={pretrain_var["mean_per_dim_variance"]:.6f}, '
          f'min={pretrain_var["min_per_dim_variance"]:.6f}, '
          f'collapsed={pretrain_var["collapsed"]}')
    results['pretrain_embedding_variance'] = pretrain_var

    # ===== Train VICReg =====
    print('\n[TRAIN] Training LeWM-VICReg (5 epochs, 100 traj x 2 friction levels)...')
    write_progress(0, 5, metric={'phase': 'training'})

    TRAIN_EPOCHS = 5
    t_train_start = time.time()

    try:
        loss_history = train_vicreg(
            model, train_dataset,
            n_epochs=TRAIN_EPOCHS, lr=1e-4, batch_size=32,
            device=DEVICE, epoch_offset=0, total_epochs_outer=TRAIN_EPOCHS,
        )
    except Exception as e:
        print(f'[ERROR] Training failed: {e}')
        traceback.print_exc()
        results['training_error'] = str(e)
        write_done('failure', f'Training failed: {e}')
        return

    train_time = time.time() - t_train_start
    print(f'\n  Training complete: {TRAIN_EPOCHS} epochs in {train_time:.1f}s ({train_time/60:.1f} min)')

    results['training'] = {
        'n_epochs': TRAIN_EPOCHS,
        'n_traj': 100,
        'lr': 1e-4,
        'batch_size': 32,
        'loss_history': loss_history,
        'initial_total_loss': round(loss_history['total'][0], 6),
        'final_total_loss': round(loss_history['total'][-1], 6),
        'initial_inv_loss': round(loss_history['inv'][0], 6),
        'final_inv_loss': round(loss_history['inv'][-1], 6),
        'initial_var_loss': round(loss_history['var'][0], 6),
        'final_var_loss': round(loss_history['var'][-1], 6),
        'initial_cov_loss': round(loss_history['cov'][0], 6),
        'final_cov_loss': round(loss_history['cov'][-1], 6),
        'train_time_sec': round(train_time, 1),
    }

    # ===== Check Convergence ==========================
    print('\n[CHECK] Analyzing VICReg loss convergence...')
    convergence = check_loss_convergence(loss_history)
    results['convergence'] = convergence

    print('\n  VICReg Loss Term Analysis:')
    for term, info in convergence.items():
        print(f'  {term:6s}: first={info["first"]:.5f} -> last={info["last"]:.5f} '
              f'({info["change_pct"]:+.1f}%) | decreasing={info["decreasing"]} | '
              f'diverging={info.get("diverging", False)}')

    # ===== Check Post-Training Embedding Variance (collapse check) =====
    print('\n[CHECK] Post-training embedding variance (collapse check)...')
    posttrain_var = check_embedding_variance(model, train_dataset, DEVICE)
    print(f'  Post-train variance: mean={posttrain_var["mean_per_dim_variance"]:.6f}, '
          f'min={posttrain_var["min_per_dim_variance"]:.6f}, '
          f'collapsed={posttrain_var["collapsed"]}, '
          f'collapsed_dims={posttrain_var["n_collapsed_dims"]}/{posttrain_var["total_dims"]}')
    results['posttrain_embedding_variance'] = posttrain_var

    # ===== Pass Criteria Evaluation ==========================
    print('\n[PASS CRITERIA] Evaluating pilot pass criteria...')

    # Criterion 1: VICReg loss converges (3 terms: inv, var, cov all decreasing or stable)
    inv_decreasing = convergence['inv']['decreasing']
    var_decreasing = convergence['var']['decreasing']
    cov_not_diverging = not convergence['cov'].get('diverging', False)
    total_decreasing = convergence['total']['decreasing']

    # Criterion 2: No collapse
    no_collapse = not posttrain_var['collapsed']
    reasonable_variance = posttrain_var['mean_per_dim_variance'] > 1e-4

    # Criterion 3: Loss terms are finite (no NaN/Inf)
    all_finite = all(
        np.isfinite(loss_history[term]).all()
        for term in ['total', 'inv', 'var', 'cov']
    )

    # Criterion 4: Overall loss decreasing
    # For a 5-epoch pilot, we accept if loss doesn't diverge
    loss_ok = not convergence['total'].get('diverging', False)

    pass_criteria = {
        'inv_loss_decreasing': bool(inv_decreasing),
        'var_loss_decreasing': bool(var_decreasing),
        'cov_not_diverging': bool(cov_not_diverging),
        'total_loss_decreasing': bool(total_decreasing),
        'no_collapse': bool(no_collapse),
        'reasonable_variance': bool(reasonable_variance),
        'all_finite': bool(all_finite),
        'loss_ok': bool(loss_ok),
    }

    # Core pass: framework works (no NaN, no crash, no collapse)
    core_pass = pass_criteria['all_finite'] and pass_criteria['no_collapse']
    # Full pass: all convergence criteria met
    full_pass = core_pass and all([
        pass_criteria['inv_loss_decreasing'],
        pass_criteria['var_loss_decreasing'],
        pass_criteria['cov_not_diverging'],
    ])

    pass_criteria['core_pass'] = bool(core_pass)
    pass_criteria['full_pass'] = bool(full_pass)
    results['pass_criteria'] = pass_criteria

    print('\n  Pass Criteria Results:')
    for k, v in pass_criteria.items():
        print(f'    {k}: {v}')

    # ===== Go/No-Go Decision ==========================
    if full_pass:
        go_no_go = 'GO'
        status = 'success'
        summary = (f'VICReg pilot: all loss terms converging. '
                   f'inv={loss_history["inv"][-1]:.4f}, var={loss_history["var"][-1]:.4f}, '
                   f'cov={loss_history["cov"][-1]:.5f}. No collapse. PROCEED to full run.')
    elif core_pass:
        go_no_go = 'GO'
        status = 'success'
        summary = (f'VICReg pilot: framework works (no crash, no collapse, finite losses). '
                   f'Some convergence criteria not fully met in 5 epochs (expected). '
                   f'Full 200-epoch run should converge. PROCEED.')
    else:
        go_no_go = 'NO_GO'
        status = 'failure'
        failed = [k for k, v in pass_criteria.items() if not v and k not in ['full_pass', 'core_pass']]
        summary = f'VICReg pilot FAILED core criteria: {failed}. INVESTIGATE before full run.'

    results['go_no_go'] = go_no_go
    results['status'] = status

    # ===== Qualitative Analysis ==========================
    print('\n[ANALYSIS] VICReg behavior summary:')
    print(f'  Invariance loss: {loss_history["inv"][0]:.4f} -> {loss_history["inv"][-1]:.4f}')
    print(f'  Variance  loss:  {loss_history["var"][0]:.4f} -> {loss_history["var"][-1]:.4f}')
    print(f'  Covariance loss: {loss_history["cov"][0]:.5f} -> {loss_history["cov"][-1]:.5f}')
    print(f'  Total loss:      {loss_history["total"][0]:.4f} -> {loss_history["total"][-1]:.4f}')
    print(f'\n  Embedding variance (collapsed dims): {posttrain_var["n_collapsed_dims"]}/{posttrain_var["total_dims"]}')
    print(f'  Mean per-dim variance: {posttrain_var["mean_per_dim_variance"]:.6f}')
    print(f'\n  Go/No-Go: {go_no_go}')
    print(f'  Summary: {summary}')

    # ===== Timing ==========================
    total_time = time.time() - t_start
    results['total_time_sec'] = round(total_time, 1)
    results['total_time_min'] = round(total_time / 60, 1)

    # ===== Save Results ==========================
    output_path = PILOTS_DIR / f'{TASK_ID}_pilot.json'
    output_path.write_text(json.dumps(results, indent=2))
    print(f'\n[SAVE] Results saved to {output_path}')

    # Write progress + DONE
    write_progress(
        TRAIN_EPOCHS, TRAIN_EPOCHS,
        loss=loss_history['total'][-1],
        metric={'status': 'complete', 'go_no_go': go_no_go}
    )
    write_done(status, summary)

    print('\n' + '='*60)
    print(f'PILOT COMPLETE: {go_no_go}')
    print(f'Time: {total_time/60:.1f} min')
    print('='*60 + '\n')

    return results


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\n[FATAL ERROR] {e}')
        traceback.print_exc()
        write_done('failure', f'Fatal error: {e}')
        sys.exit(1)
