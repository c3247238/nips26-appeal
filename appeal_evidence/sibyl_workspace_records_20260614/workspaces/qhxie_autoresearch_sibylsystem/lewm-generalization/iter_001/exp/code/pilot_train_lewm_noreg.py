"""
Pilot: train_lewm_noreg (PILOT MODE)
=======================================
Train LeWM without any regularizer (no SIGReg, no VICReg) on 100-trajectory pilot subset.
Used as collapse control baseline.

Pass criteria:
  - Training proceeds without crash
  - Log embedding variance at epoch 1 and 5 for collapse detection

Key questions:
  - Does the model collapse without regularization?
  - How does the embedding variance evolve compared to SIGReg/VICReg?

Output:
  exp/results/pilots/train_lewm_noreg_pilot.json
  exp/results/train_lewm_noreg_DONE (if success)
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

# GPU assignment (assigned: GPU 2)
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

# ========================= Paths ==========================
WORKSPACE = Path('/home/qhxie/AutoResearch-SibylSystem/workspaces/lewm-generalization/current')
DATA_DIR = WORKSPACE / 'exp' / 'data' / 'comphys_lewm' / 'pilot'
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
PILOTS_DIR = RESULTS_DIR / 'pilots'
CODE_DIR = WORKSPACE / 'exp' / 'code'
LEWM_DIR = CODE_DIR / 'le-wm'

TASK_ID = 'train_lewm_noreg'
PID_FILE = RESULTS_DIR / f'{TASK_ID}.pid'
PROGRESS_FILE = RESULTS_DIR / f'{TASK_ID}_PROGRESS.json'
DONE_FILE = RESULTS_DIR / f'{TASK_ID}_DONE'

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


class LeWMNoReg(nn.Module):
    """
    LeWM without any regularizer (no SIGReg, no VICReg).

    Only prediction loss: MSE between predictor output and target encoder embedding.
    This is the minimal JEPA formulation — prone to collapse (trivial constant solutions).

    The model predicts the NEXT latent from the current context.
    Loss = MSE(predict(z_ctx), sg(z_target))

    With no regularizer:
    - Encoder and predictor can trivially learn to output zero/constant vectors
    - Collapse is expected and is the KEY behavior we want to observe
    - We track embedding variance at epoch 1 and epoch 5 for comparison
    """

    def __init__(self, embed_dim=192, n_layers=4, num_heads=8, mlp_dim=512):
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

        Only prediction loss: MSE between pred and target (no regularizer).
        This will encourage trivial solutions (collapse to constant) without regularization.
        """
        emb = self.encode(frames)  # (B, T_total, D)

        ctx_emb = emb[:, :self.history_size]    # (B, 3, D)
        tgt_emb = emb[:, self.num_preds:]        # (B, T-1, D)

        pred_emb = self.predict(ctx_emb)          # (B, 3, D)

        # Predict next-step target
        pred = pred_emb[:, -1:]         # (B, 1, D)
        tgt = tgt_emb[:, self.history_size-1:self.history_size]  # (B, 1, D)

        # Stop-gradient on target (JEPA convention), but no regularizer
        tgt_sg = tgt.detach()

        # Pure prediction loss — no regularizer
        pred_loss = F.mse_loss(pred.squeeze(1), tgt_sg.squeeze(1))

        return {
            'loss': pred_loss,
            'pred_loss': pred_loss.item(),
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
    Check embedding variance — key metric for collapse detection.
    Returns: per-dim variance statistics.
    Collapsed: mean_per_dim_variance < 1e-4
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
    n_collapsed = int((per_dim_var < 1e-5).sum())

    # Also compute embedding mean (collapse to non-zero constant is still collapse)
    mean_emb = all_emb.mean(axis=0)
    mean_abs_mean = float(np.abs(mean_emb).mean())

    return {
        'mean_per_dim_variance': round(mean_var, 6),
        'min_per_dim_variance': round(min_var, 6),
        'std_per_dim_variance': round(std_var, 6),
        'collapsed': bool(mean_var < 1e-4),
        'n_collapsed_dims': n_collapsed,
        'total_dims': int(len(per_dim_var)),
        'mean_abs_embedding_mean': round(mean_abs_mean, 6),
    }


# ========================= Training ==========================

def train_noreg(model, dataset, n_epochs=5, lr=1e-4, batch_size=32, device=DEVICE,
                epoch_offset=0, total_epochs_outer=5,
                variance_snapshots=None):
    """
    Train LeWM-NoReg on dataset (pure prediction loss, no regularizer).
    Captures embedding variance at specific epochs for collapse monitoring.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    history = {'pred_loss': []}
    epoch_variance = {}  # epoch -> variance stats

    if variance_snapshots is None:
        variance_snapshots = set()

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        for batch in loader:
            frames = batch['pixels'].to(device)
            optimizer.zero_grad()
            out = model(frames)
            loss = out['loss']
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        history['pred_loss'].append(mean_loss)

        write_progress(
            epoch + epoch_offset + 1, total_epochs_outer,
            loss=mean_loss,
            metric={
                'phase': 'noreg_training',
                'epoch': epoch + 1,
                'pred_loss': round(mean_loss, 6),
            }
        )

        print(f'  Epoch {epoch+1:3d}/{n_epochs} | pred_loss={mean_loss:.6f}')

        # Capture variance snapshot at specified epochs
        actual_epoch = epoch + 1
        if actual_epoch in variance_snapshots:
            var_stats = check_embedding_variance(model, dataset, device)
            epoch_variance[actual_epoch] = var_stats
            print(f'    [COLLAPSE CHECK] Epoch {actual_epoch}: '
                  f'mean_var={var_stats["mean_per_dim_variance"]:.6f}, '
                  f'collapsed={var_stats["collapsed"]}, '
                  f'n_collapsed_dims={var_stats["n_collapsed_dims"]}/{var_stats["total_dims"]}')

    return history, epoch_variance


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
            'description': '1-seed 5-epoch no-regularizer training; monitor embedding variance collapse',
        }
    }

    print('\n' + '='*60)
    print('PILOT: train_lewm_noreg')
    print('No regularizer (pure prediction loss)')
    print('Monitoring embedding variance collapse at epochs 1 and 5')
    print('='*60)

    # ===== Load Data =====
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

    # ===== Initialize NoReg Model =====
    print('\n[MODEL] Initializing LeWM-NoReg model (no regularizer)...')
    model = LeWMNoReg(
        embed_dim=192,
        n_layers=4,
        num_heads=8,
        mlp_dim=512,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {n_params:,} ({n_params/1e6:.2f}M)')
    results['model_params'] = n_params
    results['model_params_M'] = round(n_params / 1e6, 2)
    results['model_config'] = {
        'embed_dim': 192,
        'n_layers': 4,
        'num_heads': 8,
        'mlp_dim': 512,
        'regularizer': 'none',
        'loss': 'pure_prediction_mse',
    }

    # ===== Pre-Training Embedding Variance =====
    print('\n[CHECK] Pre-training embedding variance (random init baseline)...')
    pretrain_var = check_embedding_variance(model, train_dataset, DEVICE)
    print(f'  Pre-train variance: mean={pretrain_var["mean_per_dim_variance"]:.6f}, '
          f'min={pretrain_var["min_per_dim_variance"]:.6f}, '
          f'collapsed={pretrain_var["collapsed"]}')
    results['pretrain_embedding_variance'] = pretrain_var

    # ===== Train NoReg =====
    print('\n[TRAIN] Training LeWM-NoReg (5 epochs, 100 traj x 2 friction levels)...')
    print('Expected behavior: loss may drop rapidly (trivial collapse) or stabilize at near-zero')
    write_progress(0, 5, metric={'phase': 'training'})

    TRAIN_EPOCHS = 5
    t_train_start = time.time()

    # Capture variance at epochs 1 and 5 (pass criteria requirement)
    VARIANCE_SNAPSHOTS = {1, 5}

    try:
        loss_history, epoch_variance = train_noreg(
            model, train_dataset,
            n_epochs=TRAIN_EPOCHS, lr=1e-4, batch_size=32,
            device=DEVICE, epoch_offset=0, total_epochs_outer=TRAIN_EPOCHS,
            variance_snapshots=VARIANCE_SNAPSHOTS,
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
        'loss_history': {k: [round(v, 6) for v in vs] for k, vs in loss_history.items()},
        'initial_pred_loss': round(loss_history['pred_loss'][0], 6),
        'final_pred_loss': round(loss_history['pred_loss'][-1], 6),
        'loss_drop_pct': round(
            (loss_history['pred_loss'][0] - loss_history['pred_loss'][-1])
            / (loss_history['pred_loss'][0] + 1e-8) * 100, 2
        ),
        'train_time_sec': round(train_time, 1),
    }
    results['epoch_variance_snapshots'] = {
        str(k): v for k, v in epoch_variance.items()
    }

    # ===== Post-Training Embedding Variance =====
    print('\n[CHECK] Post-training embedding variance (collapse detection)...')
    posttrain_var = check_embedding_variance(model, train_dataset, DEVICE)
    print(f'  Post-train variance: mean={posttrain_var["mean_per_dim_variance"]:.6f}, '
          f'min={posttrain_var["min_per_dim_variance"]:.6f}, '
          f'collapsed={posttrain_var["collapsed"]}, '
          f'collapsed_dims={posttrain_var["n_collapsed_dims"]}/{posttrain_var["total_dims"]}')
    results['posttrain_embedding_variance'] = posttrain_var

    # ===== Collapse Analysis ==========================
    print('\n[ANALYSIS] Collapse trajectory analysis...')

    epoch1_var = epoch_variance.get(1, {}).get('mean_per_dim_variance', None)
    epoch5_var = epoch_variance.get(5, {}).get('mean_per_dim_variance', None)

    all_pred_losses = loss_history['pred_loss']
    loss_final = all_pred_losses[-1]
    loss_initial = all_pred_losses[0]

    # Collapse signals:
    # 1. Variance drops dramatically (mean_var epoch5 << epoch1)
    # 2. Loss drops to near-zero quickly (trivial solution: predict mean embedding)
    variance_collapsed = posttrain_var['collapsed']
    loss_collapsed = loss_final < 1e-5  # near-zero loss = perfect prediction = collapse
    substantial_loss_drop = (loss_initial - loss_final) / (loss_initial + 1e-8) > 0.90

    # Variance trajectory
    pretrain_mean_var = pretrain_var['mean_per_dim_variance']
    posttrain_mean_var = posttrain_var['mean_per_dim_variance']
    variance_drop_pct = 0.0
    if pretrain_mean_var > 0:
        variance_drop_pct = (pretrain_mean_var - posttrain_mean_var) / pretrain_mean_var * 100

    collapse_assessment = {
        'variance_collapsed': bool(variance_collapsed),
        'loss_near_zero': bool(loss_collapsed),
        'substantial_loss_drop': bool(substantial_loss_drop),
        'pretrain_mean_var': pretrain_mean_var,
        'epoch1_mean_var': epoch1_var,
        'epoch5_mean_var': epoch5_var,
        'posttrain_mean_var': posttrain_mean_var,
        'variance_drop_pct': round(variance_drop_pct, 2),
        'initial_loss': round(loss_initial, 6),
        'final_loss': round(loss_final, 6),
        'loss_drop_pct': round((loss_initial - loss_final) / (loss_initial + 1e-8) * 100, 2),
    }

    # Overall collapse assessment
    strong_collapse = variance_collapsed or loss_collapsed
    partial_collapse = substantial_loss_drop and variance_drop_pct > 50
    no_collapse = not strong_collapse and not partial_collapse

    if strong_collapse:
        collapse_assessment['verdict'] = 'STRONG_COLLAPSE'
        collapse_assessment['description'] = (
            'Model collapsed: embeddings degenerate or loss near-zero. '
            'Confirms regularizer (SIGReg/VICReg) is NECESSARY for non-trivial representations.'
        )
    elif partial_collapse:
        collapse_assessment['verdict'] = 'PARTIAL_COLLAPSE'
        collapse_assessment['description'] = (
            'Partial collapse observed: substantial loss drop with variance decrease. '
            'Full 200-epoch run likely shows complete collapse.'
        )
    else:
        collapse_assessment['verdict'] = 'NO_COLLAPSE_5EPOCHS'
        collapse_assessment['description'] = (
            'No collapse in 5 epochs — model maintains non-trivial embeddings. '
            'Collapse may emerge in longer training (200 epochs). '
            'Full run needed to confirm control baseline behavior.'
        )

    results['collapse_analysis'] = collapse_assessment

    print(f'\n  Collapse Analysis:')
    print(f'    Pre-train variance: {pretrain_mean_var:.6f}')
    print(f'    Epoch 1 variance:   {epoch1_var}')
    print(f'    Epoch 5 variance:   {epoch5_var}')
    print(f'    Post-train variance: {posttrain_mean_var:.6f}')
    print(f'    Variance drop: {variance_drop_pct:.1f}%')
    print(f'    Loss drop: {collapse_assessment["loss_drop_pct"]:.1f}%')
    print(f'    Verdict: {collapse_assessment["verdict"]}')

    # ===== Pass Criteria Evaluation ==========================
    print('\n[PASS CRITERIA] Evaluating pilot pass criteria...')

    # Pass criteria from task plan:
    # 1. Training proceeds without crash (always true if we reach here)
    # 2. Log embedding variance at epoch 1 and 5 for collapse detection

    all_losses_finite = all(np.isfinite(v) for v in loss_history['pred_loss'])
    epoch1_variance_logged = 1 in epoch_variance
    epoch5_variance_logged = 5 in epoch_variance

    pass_criteria = {
        'training_no_crash': True,  # If we reach here, training didn't crash
        'all_losses_finite': bool(all_losses_finite),
        'epoch1_variance_logged': bool(epoch1_variance_logged),
        'epoch5_variance_logged': bool(epoch5_variance_logged),
        'epoch1_variance': epoch1_var,
        'epoch5_variance': epoch5_var,
        'collapse_verdict': collapse_assessment['verdict'],
    }

    core_pass = (
        pass_criteria['training_no_crash']
        and pass_criteria['all_losses_finite']
        and pass_criteria['epoch1_variance_logged']
        and pass_criteria['epoch5_variance_logged']
    )
    pass_criteria['core_pass'] = bool(core_pass)

    results['pass_criteria'] = pass_criteria

    print('\n  Pass Criteria Results:')
    for k, v in pass_criteria.items():
        print(f'    {k}: {v}')

    # ===== Go/No-Go Decision ==========================
    if core_pass:
        go_no_go = 'GO'
        status = 'success'
        summary = (
            f'NoReg pilot: framework runs without crash. '
            f'Variance epoch1={epoch1_var}, epoch5={epoch5_var}. '
            f'Loss: {loss_initial:.4f} -> {loss_final:.6f} ({collapse_assessment["loss_drop_pct"]:.1f}% drop). '
            f'Collapse: {collapse_assessment["verdict"]}. '
            f'PROCEED to full 200-epoch run as collapse control baseline.'
        )
    else:
        go_no_go = 'NO_GO'
        status = 'failure'
        failed = [k for k, v in pass_criteria.items() if not v and k not in ['core_pass', 'epoch1_variance', 'epoch5_variance', 'collapse_verdict']]
        summary = f'NoReg pilot FAILED: {failed}'

    results['go_no_go'] = go_no_go
    results['status'] = status

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
        loss=loss_history['pred_loss'][-1],
        metric={'status': 'complete', 'go_no_go': go_no_go}
    )
    write_done(status, summary)

    print('\n' + '='*60)
    print(f'PILOT COMPLETE: {go_no_go}')
    print(f'Collapse verdict: {collapse_assessment["verdict"]}')
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
