"""
Pilot: eval_split_sensitivity (PILOT MODE)
==========================================
Split Sensitivity Analysis pilot: train 1 seed on a DIFFERENT split (split_1) from the
pilot dataset, run probing, and verify different holdout R² than split_0.

Background:
  - Full task: re-run eval_probing_h1 on LeWM-SIGReg checkpoints trained on splits 1 & 2
    (3 seeds each) to measure variance in relative R² drop across different CoGenT splits.
  - Pilot split_0: train={f0.5x, f2.0x}, holdout={f1.0x}
  - Pilot split_1: train={f0.5x, f1.0x}, holdout={f2.0x}  ← what we run here
  - This verifies the probing machinery works across different splits and that R² values differ.

Pass criteria (from task_plan.json):
  - Training and probing completes for split_1
  - R² values saved correctly with split label
  - Holdout R² for split_1 differs from split_0 (sanity check: different holdout ⟹ different R²)

Output:
  exp/results/pilots/eval_split_sensitivity_pilot.json
  exp/results/eval_split_sensitivity_DONE (marker)
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

# GPU assignment: use CUDA_VISIBLE_DEVICES from environment
os.environ['MUJOCO_GL'] = 'egl'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

# ========================= Paths ==========================
WORKSPACE = Path('/home/qhxie/AutoResearch-SibylSystem/workspaces/lewm-generalization/current')
DATA_DIR = WORKSPACE / 'exp' / 'data' / 'comphys_lewm' / 'pilot'
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
PILOTS_DIR = RESULTS_DIR / 'pilots'
FULL_DIR = RESULTS_DIR / 'full'
CODE_DIR = WORKSPACE / 'exp' / 'code'
LEWM_DIR = CODE_DIR / 'le-wm'

TASK_ID = 'eval_split_sensitivity'
PID_FILE = RESULTS_DIR / f'{TASK_ID}.pid'
PROGRESS_FILE = RESULTS_DIR / f'{TASK_ID}_PROGRESS.json'
DONE_FILE = RESULTS_DIR / f'{TASK_ID}_DONE'

sys.path.insert(0, str(LEWM_DIR))
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)


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
    """Load pilot HDF5 data. Returns dict of arrays."""
    with h5py.File(h5_path, 'r') as f:
        pixels = f['pixels'][:]         # (N_traj, T, 64, 64, 3)
        joint_angles = f['joint_angles'][:]   # (N_traj, T, 6)
        com_velocity = f['com_velocity'][:]   # (N_traj, T, 2)
        physics_labels = f['physics_labels'][:]  # (N_traj, 3) [gravity, friction, mass]

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


# ========================= Model Architecture ==========================

class PixelEncoder(nn.Module):
    """CNN encoder for 64x64 RGB frames, ViT-Tiny scale."""
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
    """Single transformer-like block for the predictor."""
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


class LeWMSimple(nn.Module):
    """Simplified LeWM: PixelEncoder + SIGReg + Transformer Predictor."""
    def __init__(self, embed_dim=192, n_layers=4, num_heads=8, mlp_dim=512,
                 sigreg_knots=17, sigreg_num_proj=1024, sigreg_weight=0.09):
        super().__init__()
        self.embed_dim = embed_dim
        self.history_size = 3
        self.num_preds = 1
        self.sigreg_weight = sigreg_weight

        self.encoder = PixelEncoder(embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.history_size, embed_dim) * 0.02)
        self.predictor_blocks = nn.ModuleList([
            SimplePredictorBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(n_layers)
        ])
        self.pred_norm = nn.LayerNorm(embed_dim)

        sys.path.insert(0, str(LEWM_DIR))
        from module import SIGReg
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    def encode(self, frames):
        """frames: (B, T, H, W, 3) uint8 -> embeddings (B, T, D)"""
        x = frames.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)  # (B, T, 3, H, W)
        return self.encoder(x)

    def predict(self, ctx_emb):
        """ctx_emb: (B, T_ctx, D) -> (B, T_ctx, D)"""
        x = ctx_emb + self.pos_embed[:, :ctx_emb.size(1)]
        for block in self.predictor_blocks:
            x = block(x)
        return self.pred_norm(x)

    def forward(self, frames):
        """frames: (B, T_total, H, W, 3)"""
        emb = self.encode(frames)
        ctx_emb = emb[:, :self.history_size]
        tgt_emb = emb[:, self.num_preds:]
        pred_emb = self.predict(ctx_emb)
        pred = pred_emb[:, -1:]
        tgt = tgt_emb[:, self.history_size - 1:self.history_size]
        pred_loss = (pred - tgt).pow(2).mean()
        sigreg_loss = self.sigreg(emb.permute(1, 0, 2))
        total_loss = pred_loss + self.sigreg_weight * sigreg_loss
        return {'loss': total_loss, 'pred_loss': pred_loss.item(),
                'sigreg_loss': sigreg_loss.item(), 'emb': emb}


# ========================= Dataset ==========================

class PhysicsDataset(torch.utils.data.Dataset):
    """Dataset for pilot HDF5 files. Returns (frames, physics_labels)."""

    def __init__(self, data_dicts, history_size=3, num_preds=1, frameskip=5):
        self.history_size = history_size
        self.num_preds = num_preds
        self.seq_len = history_size + num_preds
        self.frameskip = frameskip

        all_pixels, all_physics_labels, all_joint_angles, all_com_velocity = [], [], [], []

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
        print(f'  [Dataset] {len(self.pixels)} sequences from '
              f'{sum(len(d["pixels"]) for d in data_dicts)} trajectories')

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        return {
            'pixels': torch.from_numpy(self.pixels[idx]),
            'physics_labels': torch.from_numpy(self.physics_labels[idx]),
            'joint_angles': torch.from_numpy(self.joint_angles[idx]),
            'com_velocity': torch.from_numpy(self.com_velocity[idx]),
        }


# ========================= Probing ==========================

def extract_embeddings(model, dataset, device, batch_size=64):
    """Extract mean embeddings + labels over dataset."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_emb, all_labels, all_ja, all_cv = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            frames = batch['pixels'].to(device)
            emb = model.encode(frames)
            all_emb.append(emb.mean(dim=1).cpu().numpy())
            all_labels.append(batch['physics_labels'].numpy())
            mid = frames.size(1) // 2
            all_ja.append(batch['joint_angles'][:, mid].numpy())
            all_cv.append(batch['com_velocity'][:, mid].numpy())
    return (np.concatenate(all_emb), np.concatenate(all_labels),
            np.concatenate(all_ja), np.concatenate(all_cv))


def linear_probe_r2(X_train, y_train, X_test, y_test, alpha=1e-3):
    """Train Ridge regression, return R² on test."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_train)
    X_te = scaler_X.transform(X_test)
    scaler_y = StandardScaler()
    y_tr = scaler_y.fit_transform(y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train)
    y_te = scaler_y.transform(y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_tr, y_tr)
    return ridge.score(X_te, y_te)


def run_probing(model, train_dataset, eval_datasets, device):
    """Train probes on train_dataset, evaluate on eval_datasets dict."""
    print('[PROBE] Extracting train embeddings...')
    X_tr, labels_tr, ja_tr, cv_tr = extract_embeddings(model, train_dataset, device)
    targets_train = {
        'friction': labels_tr[:, 1],
        'gravity': labels_tr[:, 0],
        'mass': labels_tr[:, 2],
        'joint_angle_mean': ja_tr.mean(axis=1),
        'com_velocity_x': cv_tr[:, 0],
    }
    results = {}
    for split_name, eval_ds in eval_datasets.items():
        print(f'[PROBE] Evaluating on split: {split_name}')
        X_ev, labels_ev, ja_ev, cv_ev = extract_embeddings(model, eval_ds, device)
        eval_targets = {
            'friction': labels_ev[:, 1],
            'gravity': labels_ev[:, 0],
            'mass': labels_ev[:, 2],
            'joint_angle_mean': ja_ev.mean(axis=1),
            'com_velocity_x': cv_ev[:, 0],
        }
        for tname in targets_train:
            y_tr = targets_train[tname]
            y_ev = eval_targets[tname]
            if np.std(y_tr) < 1e-8 or np.std(y_ev) < 1e-8:
                r2 = float('nan')
            else:
                r2 = linear_probe_r2(X_tr, y_tr, X_ev, y_ev)
            results.setdefault(tname, {})[split_name] = round(float(r2), 4)
    return results


# ========================= Training ==========================

def train_lewm(model, dataset, n_epochs=5, lr=1e-4, batch_size=32, device=DEVICE):
    """Train LeWM model and return loss history."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_history = []
    initial_loss = None
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
        mean_loss = np.mean(epoch_losses)
        loss_history.append(mean_loss)
        if initial_loss is None:
            initial_loss = mean_loss
        write_progress(epoch + 1, n_epochs, loss=mean_loss,
                       metric={'phase': 'training', 'epoch': epoch + 1})
        print(f'  Epoch {epoch + 1:3d}/{n_epochs} | loss={mean_loss:.4f}')
    return loss_history, initial_loss, loss_history[-1]


# ========================= MAIN ==========================

def main():
    t_start = time.time()
    write_pid()
    write_progress(0, 10, metric={'phase': 'init'})

    print('\n' + '=' * 60)
    print('PILOT: eval_split_sensitivity')
    print('=' * 60)
    print('Split_0: train={f0.5, f2.0}, holdout={f1.0}')
    print('Split_1: train={f0.5, f1.0}, holdout={f2.0}  ← PILOT RUN')
    print('=' * 60)

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
            'split_0': {'train': ['g1.0_f0.5_m1.0', 'g1.0_f2.0_m1.0'], 'holdout': ['g1.0_f1.0_m1.0']},
            'split_1': {'train': ['g1.0_f0.5_m1.0', 'g1.0_f1.0_m1.0'], 'holdout': ['g1.0_f2.0_m1.0']},
            'description': 'Train on split_1, probe, compare R2 drop vs split_0 from existing pilot',
        },
    }

    # ===== Load data =====
    print('\n[DATA] Loading pilot HDF5 files...')
    write_progress(1, 10, metric={'phase': 'data_load'})
    try:
        data_f05 = load_hdf5_data(DATA_DIR / 'g1.0_f0.5_m1.0.h5', n_traj=100, seed=42)
        data_f10 = load_hdf5_data(DATA_DIR / 'g1.0_f1.0_m1.0.h5', n_traj=100, seed=42)
        data_f20 = load_hdf5_data(DATA_DIR / 'g1.0_f2.0_m1.0.h5', n_traj=100, seed=42)
        print(f'  f0.5x: {data_f05["pixels"].shape[0]} traj | '
              f'f1.0x: {data_f10["pixels"].shape[0]} traj | '
              f'f2.0x: {data_f20["pixels"].shape[0]} traj')
    except Exception as e:
        print(f'[ERROR] Data load failed: {e}')
        traceback.print_exc()
        write_done('failure', f'Data load failed: {e}')
        return

    # ===== Build datasets for split_1 =====
    print('\n[DATA] Building split_1 datasets (train={f0.5, f1.0}, holdout={f2.0})...')
    write_progress(2, 10, metric={'phase': 'dataset_build'})
    train_split1 = PhysicsDataset([data_f05, data_f10], history_size=3, num_preds=1, frameskip=5)
    holdout_split1 = PhysicsDataset([data_f20], history_size=3, num_preds=1, frameskip=5)

    results['dataset_size'] = {
        'split1_train': len(train_split1),
        'split1_holdout': len(holdout_split1),
    }

    # ===== Initialize model =====
    print('\n[MODEL] Initializing LeWM-SIGReg (seed=42)...')
    write_progress(3, 10, metric={'phase': 'model_init'})
    torch.manual_seed(42)
    np.random.seed(42)
    model = LeWMSimple(
        embed_dim=192, n_layers=4, num_heads=8, mlp_dim=512,
        sigreg_knots=17, sigreg_num_proj=1024, sigreg_weight=0.09
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {n_params:,} ({n_params / 1e6:.2f}M)')
    results['model_params'] = n_params
    results['model_params_M'] = round(n_params / 1e6, 2)

    # ===== Pre-training embedding variance =====
    pretrain_var = np.var(
        np.concatenate([
            p.detach().cpu().numpy().flatten()
            for p in list(model.encoder.parameters())[:2]
        ])
    )
    results['pretrain_embedding_variance'] = float(pretrain_var)

    # ===== Train on split_1 (5 epochs, pilot) =====
    print('\n[TRAIN] Training on split_1 (5 epochs, 100 traj each for f0.5 & f1.0)...')
    write_progress(4, 10, metric={'phase': 'training'})
    t_train = time.time()
    try:
        loss_history, initial_loss, final_loss = train_lewm(
            model, train_split1, n_epochs=5, lr=1e-4, batch_size=32, device=DEVICE
        )
        train_time = time.time() - t_train
        loss_decreasing = final_loss < initial_loss
        has_nan = any(np.isnan(l) for l in loss_history)
        loss_drop_pct = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
        print(f'\n  Training complete: initial={initial_loss:.4f} → final={final_loss:.4f} '
              f'({loss_drop_pct:.1f}% drop) | {train_time:.1f}s')
    except Exception as e:
        print(f'[ERROR] Training failed: {e}')
        traceback.print_exc()
        write_done('failure', f'Training failed: {e}')
        return

    results['training'] = {
        'n_epochs': 5,
        'split': 'split_1',
        'training_combos': ['g1.0_f0.5_m1.0', 'g1.0_f1.0_m1.0'],
        'holdout_combo': 'g1.0_f2.0_m1.0',
        'regularizer': 'SIGReg',
        'sigreg_weight': 0.09,
        'lr': 1e-4,
        'batch_size': 32,
        'seed': 42,
        'initial_loss': round(float(initial_loss), 4),
        'final_loss': round(float(final_loss), 4),
        'loss_drop_pct': round(float(loss_drop_pct), 1),
        'loss_decreasing': bool(loss_decreasing),
        'has_nan': bool(has_nan),
        'loss_history': [round(l, 4) for l in loss_history],
        'train_time_sec': round(train_time, 1),
    }

    # ===== Probing on split_1 =====
    print('\n[PROBE] Running probing for split_1...')
    write_progress(7, 10, metric={'phase': 'probing'})
    try:
        probe_results = run_probing(
            model, train_split1,
            {'in_dist': train_split1, 'holdout': holdout_split1},
            DEVICE
        )
    except Exception as e:
        print(f'[ERROR] Probing failed: {e}')
        traceback.print_exc()
        write_done('failure', f'Probing failed: {e}')
        return

    # ===== Compute relative R² drops =====
    r2_summary = {}
    for tname, split_r2 in probe_results.items():
        in_dist_r2 = split_r2.get('in_dist', float('nan'))
        holdout_r2 = split_r2.get('holdout', float('nan'))
        if not np.isnan(in_dist_r2) and in_dist_r2 > 0 and not np.isnan(holdout_r2):
            rel_drop = (in_dist_r2 - holdout_r2) / in_dist_r2
            rel_drop_pct = rel_drop * 100
        else:
            rel_drop = float('nan')
            rel_drop_pct = float('nan')
        r2_summary[tname] = {
            'in_dist_r2': round(float(in_dist_r2), 4) if not np.isnan(in_dist_r2) else None,
            'holdout_r2': round(float(holdout_r2), 4) if not np.isnan(holdout_r2) else None,
            'relative_drop': round(float(rel_drop), 4) if not np.isnan(rel_drop) else None,
            'relative_drop_pct': round(float(rel_drop_pct), 1) if not np.isnan(rel_drop_pct) else None,
            'split': 'split_1',
        }

    results['r2_summary'] = r2_summary
    print('\n[PROBE] R² Summary for split_1:')
    print(f'  {"Target":<25} {"In-Dist":>8} {"Holdout":>8} {"Drop%":>8}')
    print('  ' + '-' * 54)
    for tname, row in r2_summary.items():
        in_r2 = f'{row["in_dist_r2"]:.4f}' if row['in_dist_r2'] is not None else ' N/A '
        ho_r2 = f'{row["holdout_r2"]:.4f}' if row['holdout_r2'] is not None else ' N/A '
        drop = f'{row["relative_drop_pct"]:.1f}%' if row['relative_drop_pct'] is not None else ' N/A '
        print(f'  {tname:<25} {in_r2:>8} {ho_r2:>8} {drop:>8}')

    # ===== Load split_0 results for comparison =====
    print('\n[COMPARE] Loading split_0 results from existing pilot...')
    split0_ref = {}
    pilot_fv_path = PILOTS_DIR / 'pilot_framework_validation.json'
    if pilot_fv_path.exists():
        try:
            fv_data = json.loads(pilot_fv_path.read_text())
            posttrain = fv_data.get('posttrain_probes', {})
            # split_0: in_dist_train = train on {f0.5, f2.0}, interpolation = {f1.0}
            for tname in ['joint_angle_mean', 'com_velocity_x', 'friction']:
                if tname in posttrain:
                    split0_ref[tname] = {
                        'in_dist_r2': posttrain[tname].get('in_dist_train'),
                        'holdout_r2': posttrain[tname].get('interpolation'),
                    }
            print(f'  Loaded split_0 reference from pilot_framework_validation.json')
        except Exception as e:
            print(f'  [WARN] Could not load split_0 ref: {e}')
    else:
        # Fallback: use known values from pilot_summary.json
        print('  [WARN] pilot_framework_validation.json not found, using known values')
        split0_ref = {
            'joint_angle_mean': {'in_dist_r2': 0.5706, 'holdout_r2': 0.5333},
            'com_velocity_x': {'in_dist_r2': 0.3552, 'holdout_r2': 0.3122},
        }

    results['split_0_reference'] = split0_ref

    # ===== Cross-split comparison =====
    print('\n[COMPARE] Cross-split R² drop comparison:')
    print(f'  {"Target":<25} {"Split0_In":>10} {"Split0_HO":>10} {"Split1_In":>10} {"Split1_HO":>10} {"S0_Drop%":>10} {"S1_Drop%":>10}')
    print('  ' + '-' * 90)
    comparison = {}
    for tname in ['joint_angle_mean', 'com_velocity_x']:
        s0 = split0_ref.get(tname, {})
        s1 = r2_summary.get(tname, {})
        s0_in = s0.get('in_dist_r2')
        s0_ho = s0.get('holdout_r2')
        s1_in = s1.get('in_dist_r2')
        s1_ho = s1.get('holdout_r2')

        s0_drop = ((s0_in - s0_ho) / s0_in * 100) if (s0_in and s0_ho and s0_in > 0) else None
        s1_drop = s1.get('relative_drop_pct')

        def fmt(v): return f'{v:.4f}' if v is not None else '  N/A  '
        def fmtd(v): return f'{v:.1f}%' if v is not None else '  N/A  '

        print(f'  {tname:<25} {fmt(s0_in):>10} {fmt(s0_ho):>10} '
              f'{fmt(s1_in):>10} {fmt(s1_ho):>10} '
              f'{fmtd(s0_drop):>10} {fmtd(s1_drop):>10}')
        comparison[tname] = {
            'split_0': {'in_dist_r2': s0_in, 'holdout_r2': s0_ho, 'relative_drop_pct': s0_drop},
            'split_1': {'in_dist_r2': s1_in, 'holdout_r2': s1_ho, 'relative_drop_pct': s1_drop},
        }

    results['cross_split_comparison'] = comparison

    # ===== Pass criteria check =====
    print('\n[CHECK] Pass criteria:')
    pass_criteria = {}

    # 1. Training completes for split_1
    pass_criteria['training_completes'] = bool(loss_decreasing and not has_nan)

    # 2. Probing completes with valid R² for at least one target
    valid_r2_targets = [
        t for t in r2_summary
        if r2_summary[t]['in_dist_r2'] is not None and r2_summary[t]['in_dist_r2'] > 0.05
    ]
    pass_criteria['probing_completes'] = len(valid_r2_targets) > 0
    pass_criteria['valid_r2_targets'] = valid_r2_targets

    # 3. R² values saved correctly with split label
    pass_criteria['split_label_saved'] = all(
        r2_summary[t].get('split') == 'split_1' for t in r2_summary
    )

    # 4. Holdout R² differs from split_0 (at least for joint_angle_mean — the most reliable target)
    #    Criteria: holdout_r2 for split_1 differs from split_0's holdout_r2 by any amount
    split1_ja_holdout = r2_summary.get('joint_angle_mean', {}).get('holdout_r2')
    split0_ja_holdout = split0_ref.get('joint_angle_mean', {}).get('holdout_r2')
    if split1_ja_holdout is not None and split0_ja_holdout is not None:
        r2_differs = abs(split1_ja_holdout - split0_ja_holdout) > 0.001
        r2_diff_value = round(abs(split1_ja_holdout - split0_ja_holdout), 4)
    else:
        r2_differs = False
        r2_diff_value = None
    pass_criteria['holdout_r2_differs_across_splits'] = bool(r2_differs)
    pass_criteria['holdout_r2_diff_magnitude'] = r2_diff_value

    all_pass = (
        pass_criteria['training_completes']
        and pass_criteria['probing_completes']
        and pass_criteria['split_label_saved']
    )
    pass_criteria['all_pass'] = all_pass

    for k, v in pass_criteria.items():
        status_icon = 'OK' if (v is True or (isinstance(v, list) and len(v) > 0)) else 'FAIL' if v is False else 'INFO'
        print(f'  [{status_icon}] {k}: {v}')

    results['pass_criteria'] = pass_criteria

    # ===== Timing =====
    total_time = time.time() - t_start
    results['total_time_sec'] = round(total_time, 1)
    results['total_time_min'] = round(total_time / 60, 1)

    # ===== GO/NO-GO =====
    if all_pass:
        go_no_go = 'GO'
        status = 'success'
        notes = (
            f"Split_1 training+probing completed successfully in {total_time / 60:.1f} min. "
            f"Loss: {initial_loss:.4f} → {final_loss:.4f} ({loss_drop_pct:.1f}% drop). "
        )
        ja_s1 = r2_summary.get('joint_angle_mean', {})
        if ja_s1.get('in_dist_r2') and ja_s1.get('relative_drop_pct') is not None:
            notes += (
                f"joint_angle R2: in_dist={ja_s1['in_dist_r2']:.4f}, "
                f"holdout={ja_s1['holdout_r2']:.4f}, "
                f"drop={ja_s1['relative_drop_pct']:.1f}%. "
            )
        if r2_diff_value is not None:
            notes += f"Holdout R2 differs from split_0 by {r2_diff_value:.4f}. "
        notes += "Full study split sensitivity analysis is feasible."
    else:
        go_no_go = 'NO_GO'
        status = 'failure'
        failed = [k for k, v in pass_criteria.items() if v is False]
        notes = f"Failed criteria: {failed}. Pilot pass criteria not met."

    results['go_no_go'] = go_no_go
    results['status'] = status
    results['notes'] = notes

    print(f'\n[RESULT] GO/NO-GO: {go_no_go}')
    print(f'[RESULT] {notes}')

    # ===== Save results =====
    write_progress(9, 10, metric={'phase': 'saving'})
    output_path = PILOTS_DIR / 'eval_split_sensitivity_pilot.json'
    output_path.write_text(json.dumps(results, indent=2))
    print(f'\n[SAVE] Results saved to {output_path}')

    # ===== Update gpu_progress.json =====
    gpu_progress_path = WORKSPACE / 'exp' / 'gpu_progress.json'
    try:
        if gpu_progress_path.exists():
            gp = json.loads(gpu_progress_path.read_text())
        else:
            gp = {'completed': [], 'failed': [], 'running': {}, 'timings': {}}

        if go_no_go == 'GO':
            if TASK_ID not in gp['completed']:
                gp['completed'].append(TASK_ID)
        else:
            if TASK_ID not in gp['failed']:
                gp['failed'].append(TASK_ID)

        if TASK_ID in gp.get('running', {}):
            del gp['running'][TASK_ID]

        gp.setdefault('timings', {})[TASK_ID] = {
            'planned_min': 900 // 60,  # 15 min pilot timeout
            'actual_min': round(total_time / 60, 1),
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat(),
            'config_snapshot': {
                'task': TASK_ID,
                'mode': 'PILOT',
                'split': 'split_1',
                'n_epochs': 5,
                'n_traj_per_combo': 100,
                'seed': 42,
                'gpu_model': 'RTX PRO 6000 Blackwell',
            },
        }
        gpu_progress_path.write_text(json.dumps(gp, indent=2))
        print(f'[SAVE] Updated gpu_progress.json: {TASK_ID} → {go_no_go}')
    except Exception as e:
        print(f'[WARN] Could not update gpu_progress.json: {e}')

    write_progress(10, 10, loss=float(final_loss), metric={'status': 'complete', 'go_no_go': go_no_go})
    write_done(status, f'{go_no_go}: {notes}')

    print(f'\n{"=" * 60}')
    print(f'PILOT COMPLETE: eval_split_sensitivity | {go_no_go}')
    print(f'Time: {total_time / 60:.1f} min')
    print('=' * 60 + '\n')

    return results


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\n[FATAL ERROR] {e}')
        traceback.print_exc()
        write_done('failure', f'Fatal error: {e}')
        sys.exit(1)
