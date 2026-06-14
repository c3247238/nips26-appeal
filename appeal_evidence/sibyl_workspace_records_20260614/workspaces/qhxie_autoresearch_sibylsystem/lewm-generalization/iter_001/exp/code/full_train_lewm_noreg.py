"""
FULL: train_lewm_noreg
=======================
Train LeWM without any regularizer (no SIGReg, no VICReg) as collapse control baseline.
3 seeds {42, 7, 13}, 200 epochs each, on the primary 18/27 CoGenT training split (seed_split_0).

Task: Control baseline — confirms regularizer (SIGReg/VICReg) necessity.
Expected behavior: embedding collapse (trivial constant solutions) with pure MSE prediction loss.

Output:
  exp/results/full/lewm_noreg/seed{seed}_epoch200.pt (3 checkpoints)
  exp/results/full/lewm_noreg/training_summary.json
  exp/results/train_lewm_noreg_DONE
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
import h5py

# GPU assignment
os.environ['MUJOCO_GL'] = 'egl'
GPU_ID = int(os.environ.get('CUDA_VISIBLE_DEVICES', '4').split(',')[0])
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

# ========================= Paths ==========================
WORKSPACE = Path('/home/qhxie/AutoResearch-SibylSystem/workspaces/lewm-generalization/current')
DATA_DIR = WORKSPACE / 'exp' / 'data' / 'comphys_lewm'
SPLITS_FILE = WORKSPACE / 'exp' / 'data' / 'splits.json'
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
FULL_DIR = RESULTS_DIR / 'full'
NOREG_DIR = FULL_DIR / 'lewm_noreg'
CODE_DIR = WORKSPACE / 'exp' / 'code'

TASK_ID = 'train_lewm_noreg'
PID_FILE = RESULTS_DIR / f'{TASK_ID}.pid'
PROGRESS_FILE = RESULTS_DIR / f'{TASK_ID}_PROGRESS.json'
DONE_FILE = RESULTS_DIR / f'{TASK_ID}_DONE'

NOREG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 13]
N_EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-4
N_TRAJ_PER_COMBO = 200  # full run: all 200 trajectories per combination

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

def load_hdf5_data(h5_path, n_traj=None, seed=42):
    """Load HDF5 trajectory data."""
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

def combo_to_filename(gravity, friction, mass):
    """Convert factor combination to HDF5 filename."""
    return f'g{gravity}_f{friction}_m{mass}.h5'


# ========================= Model Architecture ====================

class PixelEncoder(nn.Module):
    """CNN encoder: 64x64 RGB -> embed_dim embeddings."""
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

    Loss = MSE(predict(z_ctx), sg(z_target))

    With no regularizer:
    - Encoder and predictor can trivially learn to output zero/constant vectors
    - Collapse is expected and is the KEY behavior we want to observe
    - Collapse control baseline to confirm regularizer necessity
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
        """frames: (B, T, H, W, 3) uint8 -> (B, T, D) embeddings"""
        x = frames.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)  # -> (B, T, 3, H, W)
        return self.encoder(x)

    def predict(self, ctx_emb):
        x = ctx_emb + self.pos_embed[:, :ctx_emb.size(1)]
        for block in self.predictor_blocks:
            x = block(x)
        return self.pred_norm(x)

    def forward(self, frames):
        emb = self.encode(frames)  # (B, T_total, D)
        ctx_emb = emb[:, :self.history_size]    # (B, 3, D)
        tgt_emb = emb[:, self.num_preds:]        # (B, T-1, D)
        pred_emb = self.predict(ctx_emb)          # (B, 3, D)

        pred = pred_emb[:, -1:]   # (B, 1, D)
        tgt = tgt_emb[:, self.history_size-1:self.history_size]  # (B, 1, D)
        tgt_sg = tgt.detach()  # stop-gradient on target (JEPA)

        # Pure prediction loss — no regularizer
        pred_loss = F.mse_loss(pred.squeeze(1), tgt_sg.squeeze(1))

        return {
            'loss': pred_loss,
            'pred_loss': pred_loss.item(),
            'emb': emb,
        }


# ========================= Dataset ============================

class PhysicsDataset(torch.utils.data.Dataset):
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

def check_embedding_variance(model, dataset, device, batch_size=128):
    """
    Check embedding variance — key metric for collapse detection.
    Returns: per-dim variance statistics.
    Collapsed: mean_per_dim_variance < 1e-4
    """
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                          num_workers=2)
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

    mean_emb = all_emb.mean(axis=0)
    mean_abs_mean = float(np.abs(mean_emb).mean())

    return {
        'mean_per_dim_variance': round(mean_var, 8),
        'min_per_dim_variance': round(min_var, 8),
        'std_per_dim_variance': round(std_var, 8),
        'collapsed': bool(mean_var < 1e-4),
        'n_collapsed_dims': n_collapsed,
        'total_dims': int(len(per_dim_var)),
        'mean_abs_embedding_mean': round(mean_abs_mean, 8),
    }


# ========================= Training ==========================

def train_one_seed(model, dataset, seed, n_epochs=200, lr=1e-4, batch_size=64,
                   device=DEVICE, seed_idx=0, total_seeds=3, save_dir=NOREG_DIR):
    """
    Train LeWM-NoReg for one seed.
    Captures embedding variance at epochs 1, 10, 50, 100, 200 for collapse monitoring.
    Returns training summary dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    # Cosine annealing LR scheduler (same as vicreg for fair comparison)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    loss_history = []

    # Variance checkpoints for collapse trajectory analysis
    VARIANCE_CHECKPOINTS = {1, 10, 50, 100, 200}
    epoch_variance = {}

    t_seed_start = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        for batch in loader:
            frames = batch['pixels'].to(device)
            optimizer.zero_grad()
            out = model(frames)
            loss = out['loss']

            if torch.isnan(loss) or torch.isinf(loss):
                print(f'[WARN] NaN/Inf loss at epoch {epoch+1}, seed {seed}. Skipping batch.')
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()

        if not epoch_losses:
            print(f'[ERROR] No valid batches at epoch {epoch+1}!')
            break

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        actual_epoch = epoch + 1

        # Global progress tracking (over all seeds)
        global_epoch = seed_idx * n_epochs + actual_epoch
        total_epochs_global = total_seeds * n_epochs

        write_progress(
            global_epoch, total_epochs_global,
            loss=mean_loss,
            metric={
                'phase': 'full_noreg_training',
                'seed': seed,
                'seed_idx': seed_idx + 1,
                'total_seeds': total_seeds,
                'epoch': actual_epoch,
                'pred_loss': round(mean_loss, 6),
                'lr': scheduler.get_last_lr()[0],
            }
        )

        # Variance snapshot at key epochs
        if actual_epoch in VARIANCE_CHECKPOINTS:
            var_stats = check_embedding_variance(model, dataset, device)
            epoch_variance[actual_epoch] = var_stats
            print(f'  [Seed {seed}] Epoch {actual_epoch:3d} | '
                  f'pred_loss={mean_loss:.6f} | '
                  f'emb_var={var_stats["mean_per_dim_variance"]:.8f} '
                  f'(collapsed={var_stats["collapsed"]}, '
                  f'n_coll={var_stats["n_collapsed_dims"]}/{var_stats["total_dims"]})')
        elif (actual_epoch) % 10 == 0 or actual_epoch == 1:
            elapsed = time.time() - t_seed_start
            eta = elapsed / actual_epoch * (n_epochs - actual_epoch)
            print(f'  [Seed {seed}] Epoch {actual_epoch:3d}/{n_epochs} | '
                  f'pred_loss={mean_loss:.6f} | '
                  f'ETA: {eta/60:.1f}min')

    seed_time = time.time() - t_seed_start
    print(f'\n  [Seed {seed}] Done: {seed_time/60:.1f} min | '
          f'loss: {loss_history[0]:.6f} → {loss_history[-1]:.6f}')

    # Save checkpoint
    ckpt_path = save_dir / f'seed{seed}_epoch{n_epochs}.pt'
    torch.save({
        'epoch': n_epochs,
        'seed': seed,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': loss_history[-1],
        'loss_history': loss_history,
        'epoch_variance': {str(k): v for k, v in epoch_variance.items()},
        'config': {
            'embed_dim': 192,
            'n_layers': 4,
            'num_heads': 8,
            'mlp_dim': 512,
            'regularizer': 'none',
            'loss': 'pure_prediction_mse',
            'seed': seed,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'mode': 'FULL',
        }
    }, str(ckpt_path))
    print(f'  [Seed {seed}] Checkpoint saved: {ckpt_path}')

    # Collapse analysis
    loss_initial = loss_history[0]
    loss_final = loss_history[-1]
    variance_collapsed = epoch_variance.get(n_epochs, {}).get('collapsed', None)
    loss_near_zero = loss_final < 1e-5

    collapse_verdict = 'UNKNOWN'
    if variance_collapsed is True or loss_near_zero:
        collapse_verdict = 'STRONG_COLLAPSE'
    elif loss_initial > 0 and (loss_initial - loss_final) / (loss_initial + 1e-8) > 0.90:
        if epoch_variance.get(n_epochs, {}).get('mean_per_dim_variance', 1.0) < 1e-3:
            collapse_verdict = 'PARTIAL_COLLAPSE'
        else:
            collapse_verdict = 'NO_COLLAPSE'
    else:
        collapse_verdict = 'NO_COLLAPSE'

    return {
        'seed': seed,
        'n_epochs': n_epochs,
        'initial_loss': round(loss_initial, 8),
        'final_loss': round(loss_final, 8),
        'loss_decrease_pct': round((loss_initial - loss_final) / (loss_initial + 1e-8) * 100, 2),
        'loss_history': [round(v, 8) for v in loss_history],
        'epoch_variance_snapshots': {str(k): v for k, v in epoch_variance.items()},
        'collapse_verdict': collapse_verdict,
        'train_time_min': round(seed_time / 60, 1),
        'checkpoint_path': str(ckpt_path),
    }


# ========================= VRAM Probing ==========================

def probe_max_batch_size(model, dataset, device, start=256, min_bs=8):
    """Binary search for max batch size that fits in VRAM."""
    import gc
    high, best = start, min_bs

    def sample_input_fn(bs):
        items = [dataset[i] for i in range(min(bs, len(dataset)))]
        frames = torch.stack([it['pixels'] for it in items]).to(device)
        return frames

    while min_bs <= high:
        mid = (min_bs + high) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            frames = sample_input_fn(mid)
            with torch.no_grad():
                _ = model(frames)
            best = mid
            min_bs = mid + 1
        except torch.cuda.OutOfMemoryError:
            high = mid - 1
            torch.cuda.empty_cache()
            gc.collect()

    return best


# ========================= MAIN ==============================

def main():
    t_start = time.time()
    write_pid()
    write_progress(0, len(SEEDS) * N_EPOCHS, loss=None, metric={'phase': 'init'})

    print('\n' + '='*70)
    print('FULL: train_lewm_noreg')
    print('No regularizer (pure MSE prediction): 3 seeds x 200 epochs on 18/27 CoGenT split')
    print('Control baseline — monitoring embedding collapse trajectory')
    print('='*70)

    # ===== Load splits =====
    print('\n[DATA] Loading split definitions...')
    splits = json.loads(SPLITS_FILE.read_text())
    train_combos = splits['split_0']['train']  # 18 combos: [gravity, friction, mass]
    holdout_combos = splits['split_0']['holdout']  # 9 combos

    print(f'  Training combos: {len(train_combos)}')
    print(f'  Holdout combos: {len(holdout_combos)}')

    # ===== Load training data =====
    print(f'\n[DATA] Loading {len(train_combos)} training HDF5 files ({N_TRAJ_PER_COMBO} traj each)...')
    data_dicts = []
    t_data = time.time()

    for combo in train_combos:
        g, f, m = combo
        fname = combo_to_filename(g, f, m)
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f'  [WARN] Missing file: {fpath}')
            continue
        d = load_hdf5_data(fpath, n_traj=N_TRAJ_PER_COMBO, seed=42)
        data_dicts.append(d)

    print(f'  Loaded {len(data_dicts)} HDF5 files in {time.time()-t_data:.1f}s')

    if len(data_dicts) < 10:
        msg = f'Too few training files: {len(data_dicts)}. Need at least 10.'
        print(f'[ERROR] {msg}')
        write_done('failure', msg)
        return

    # ===== Build dataset =====
    print('\n[DATA] Building PhysicsDataset...')
    train_dataset = PhysicsDataset(data_dicts, history_size=3, num_preds=1, frameskip=5)
    print(f'  Total sequences: {len(train_dataset)}')

    # ===== VRAM probing =====
    print('\n[VRAM] Probing max batch size for GPU...')
    test_model = LeWMNoReg(embed_dim=192, n_layers=4, num_heads=8, mlp_dim=512).to(DEVICE)
    probed_bs = probe_max_batch_size(test_model, train_dataset, DEVICE, start=512, min_bs=8)
    del test_model
    torch.cuda.empty_cache()

    effective_bs = min(probed_bs, 256)
    if effective_bs < BATCH_SIZE:
        effective_bs = BATCH_SIZE
    print(f'  Probed max batch size: {probed_bs}')
    print(f'  Using batch size: {effective_bs}')

    # Save GPU profile
    gpu_profile = {
        'gpu_name': torch.cuda.get_device_name(0),
        'vram_total_mb': torch.cuda.get_device_properties(0).total_memory // (1024*1024),
        'probed_max_batch_size': probed_bs,
        'effective_batch_size': effective_bs,
    }
    (NOREG_DIR / 'gpu_profile.json').write_text(json.dumps(gpu_profile, indent=2))

    # ===== Train each seed =====
    all_seed_results = []
    n_params = None

    for seed_idx, seed in enumerate(SEEDS):
        print(f'\n{"="*60}')
        print(f'[TRAIN] Seed {seed} ({seed_idx+1}/{len(SEEDS)}) — NoReg Control')
        print(f'{"="*60}')

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = LeWMNoReg(
            embed_dim=192,
            n_layers=4,
            num_heads=8,
            mlp_dim=512,
        ).to(DEVICE)

        if n_params is None:
            n_params = sum(p.numel() for p in model.parameters())
            print(f'  Model parameters: {n_params:,} ({n_params/1e6:.2f}M)')

        seed_result = train_one_seed(
            model, train_dataset,
            seed=seed,
            n_epochs=N_EPOCHS,
            lr=LR,
            batch_size=effective_bs,
            device=DEVICE,
            seed_idx=seed_idx,
            total_seeds=len(SEEDS),
            save_dir=NOREG_DIR,
        )
        all_seed_results.append(seed_result)

        # Free GPU memory between seeds
        del model
        torch.cuda.empty_cache()

        # Save intermediate summary
        intermediate_summary = {
            'task_id': TASK_ID,
            'mode': 'FULL',
            'seeds_completed': SEEDS[:seed_idx+1],
            'seeds_remaining': SEEDS[seed_idx+1:],
            'results_so_far': all_seed_results,
            'updated_at': datetime.now().isoformat(),
        }
        (NOREG_DIR / 'training_summary_partial.json').write_text(
            json.dumps(intermediate_summary, indent=2))

    # ===== Save final summary =====
    total_time = time.time() - t_start

    # Aggregate collapse analysis across seeds
    collapse_verdicts = [r['collapse_verdict'] for r in all_seed_results]
    strong_collapse_count = sum(1 for v in collapse_verdicts if v == 'STRONG_COLLAPSE')
    partial_collapse_count = sum(1 for v in collapse_verdicts if v == 'PARTIAL_COLLAPSE')

    if strong_collapse_count >= 2:
        overall_collapse = 'STRONG_COLLAPSE'
        collapse_description = (
            'Model consistently collapsed across seeds. '
            'Confirms SIGReg/VICReg regularization is NECESSARY for non-trivial representations.'
        )
    elif strong_collapse_count + partial_collapse_count >= 2:
        overall_collapse = 'PARTIAL_COLLAPSE'
        collapse_description = (
            'Partial collapse observed in most seeds. '
            'Full collapse likely at longer training horizons.'
        )
    else:
        overall_collapse = 'NO_COLLAPSE'
        collapse_description = (
            'No collapse detected — model maintains non-trivial embeddings even without regularization. '
            'This is an unexpected result; LayerNorm may be preventing collapse.'
        )

    summary = {
        'task_id': TASK_ID,
        'mode': 'FULL',
        'timestamp': datetime.now().isoformat(),
        'gpu': str(DEVICE),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'model_params': n_params,
        'model_params_M': round(n_params / 1e6, 2) if n_params else None,
        'model_config': {
            'embed_dim': 192,
            'n_layers': 4,
            'num_heads': 8,
            'mlp_dim': 512,
            'regularizer': 'none',
            'loss': 'pure_prediction_mse',
        },
        'training_config': {
            'seeds': SEEDS,
            'n_epochs': N_EPOCHS,
            'batch_size': effective_bs,
            'lr': LR,
            'n_traj_per_combo': N_TRAJ_PER_COMBO,
            'n_training_combos': len(data_dicts),
            'n_holdout_combos': len(holdout_combos),
            'dataset_size': len(train_dataset),
            'split': 'split_0',
        },
        'gpu_profile': gpu_profile,
        'seed_results': all_seed_results,
        'overall': {
            'final_losses': [r['final_loss'] for r in all_seed_results],
            'mean_final_loss': round(float(np.mean([r['final_loss'] for r in all_seed_results])), 8),
            'std_final_loss': round(float(np.std([r['final_loss'] for r in all_seed_results])), 8),
            'all_seeds_complete': len(all_seed_results) == len(SEEDS),
            'collapse_verdicts': collapse_verdicts,
            'overall_collapse_verdict': overall_collapse,
            'collapse_description': collapse_description,
        },
        'total_time_sec': round(total_time, 1),
        'total_time_min': round(total_time / 60, 1),
        'total_time_hrs': round(total_time / 3600, 2),
    }

    summary_path = NOREG_DIR / 'training_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'\n[SAVE] Summary saved to {summary_path}')

    # ===== Update gpu_progress.json =====
    gp_path = WORKSPACE / 'exp' / 'gpu_progress.json'
    try:
        gp = json.loads(gp_path.read_text()) if gp_path.exists() else {
            'completed': [], 'failed': [], 'running': {}, 'timings': {}
        }
        if TASK_ID not in gp['completed']:
            gp['completed'].append(TASK_ID)
        gp['running'].pop(TASK_ID, None)
        gp['timings'][TASK_ID] = {
            'planned_min': 240,
            'actual_min': round(total_time / 60),
            'start_time': summary['timestamp'],
            'end_time': datetime.now().isoformat(),
            'config_snapshot': {
                'task': TASK_ID,
                'mode': 'FULL',
                'model': 'LeWM-NoReg',
                'regularizer': 'none',
                'seeds': SEEDS,
                'n_epochs': N_EPOCHS,
                'batch_size': effective_bs,
                'n_traj_per_combo': N_TRAJ_PER_COMBO,
                'n_training_combos': len(data_dicts),
                'gpu': 'NVIDIA RTX PRO 6000 Blackwell',
                'gpu_count': 1,
                'mean_final_loss': summary['overall']['mean_final_loss'],
                'overall_collapse_verdict': overall_collapse,
            }
        }
        gp_path.write_text(json.dumps(gp, indent=2))
        print('[SAVE] Updated gpu_progress.json')
    except Exception as e:
        print(f'[WARN] Could not update gpu_progress.json: {e}')

    # ===== Update experiment_state.json =====
    es_path = WORKSPACE / 'exp' / 'experiment_state.json'
    try:
        es = json.loads(es_path.read_text()) if es_path.exists() else {
            'schema_version': 1, 'tasks': {}, 'last_recovery_at': '', 'recovery_log': []
        }
        es['tasks'][TASK_ID] = {
            'status': 'completed',
            'gpu_ids': [GPU_ID],
            'completed_at': datetime.now().isoformat(),
            'result_path': f'exp/results/full/lewm_noreg/training_summary.json',
        }
        es_path.write_text(json.dumps(es, indent=2))
        print('[SAVE] Updated experiment_state.json')
    except Exception as e:
        print(f'[WARN] Could not update experiment_state.json: {e}')

    # ===== Final print =====
    print('\n' + '='*70)
    print('FULL TRAINING COMPLETE: train_lewm_noreg')
    for r in all_seed_results:
        print(f'  Seed {r["seed"]}: loss {r["initial_loss"]:.6f} → {r["final_loss"]:.8f} '
              f'({r["loss_decrease_pct"]:.1f}% drop) | {r["train_time_min"]:.1f} min | '
              f'collapse={r["collapse_verdict"]}')
    print(f'  Mean final loss: {summary["overall"]["mean_final_loss"]:.8f} ± {summary["overall"]["std_final_loss"]:.8f}')
    print(f'  Overall collapse verdict: {overall_collapse}')
    print(f'  {collapse_description}')
    print(f'  Total time: {total_time/3600:.2f} hours')
    print('='*70 + '\n')

    write_progress(
        len(SEEDS) * N_EPOCHS, len(SEEDS) * N_EPOCHS,
        loss=summary['overall']['mean_final_loss'],
        metric={
            'status': 'complete',
            'seeds_done': len(all_seed_results),
            'go_no_go': 'GO',
            'overall_collapse_verdict': overall_collapse,
        }
    )
    write_done('success',
               f'NoReg full training complete: {len(all_seed_results)}/{len(SEEDS)} seeds, '
               f'mean final loss={summary["overall"]["mean_final_loss"]:.6f}, '
               f'collapse={overall_collapse}')

    return summary


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\n[FATAL ERROR] {e}')
        traceback.print_exc()
        write_done('failure', f'Fatal error: {e}')
        sys.exit(1)
