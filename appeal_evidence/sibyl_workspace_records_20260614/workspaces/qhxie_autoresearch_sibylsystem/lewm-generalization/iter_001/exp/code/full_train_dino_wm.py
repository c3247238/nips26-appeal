"""
FULL: train_dino_wm
====================================
Train DINO-WM (frozen DINOv2-s encoder + learned transformer predictor)
on the primary 18/27 CoGenT training split (split_0), 3 seeds, 200 epochs.

Architecture:
  - Encoder: frozen DINOv2-s (timm: vit_small_patch14_dinov2) -> 384-dim patch embeddings
             -> mean-pool across patches -> 384-dim frame embedding
  - Predictor: learned transformer (4 layers, same as LeWM architecture)
             -> predict next-step embedding from 3-frame context
  - Loss: MSE between predicted embedding and target (stop-gradient, JEPA convention)

Outputs:
  exp/results/full/dino_wm/seed_{seed}_epoch{epoch}.pt (every 50 epochs)
  exp/results/full/dino_wm/seed_{seed}_final.pt
  exp/results/full/dino_wm/seed_{seed}_training_log.json
  exp/results/full/dino_wm/full_results.json
  exp/results/train_dino_wm_DONE
"""

import os
import sys
import json
import time
import gc
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# GPU assignment - CUDA_VISIBLE_DEVICES set externally, use cuda:0
os.environ['MUJOCO_GL'] = 'egl'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

# ========================= Paths ==========================
WORKSPACE = Path('/home/qhxie/AutoResearch-SibylSystem/workspaces/lewm-generalization/current')
DATA_DIR = WORKSPACE / 'exp' / 'data' / 'comphys_lewm'
SPLITS_FILE = WORKSPACE / 'exp' / 'data' / 'splits.json'
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
FULL_DIR = RESULTS_DIR / 'full' / 'dino_wm'
GPU_PROGRESS_FILE = WORKSPACE / 'exp' / 'gpu_progress.json'

TASK_ID = 'train_dino_wm'
PID_FILE = RESULTS_DIR / f'{TASK_ID}.pid'
PROGRESS_FILE = RESULTS_DIR / f'{TASK_ID}_PROGRESS.json'
DONE_FILE = RESULTS_DIR / f'{TASK_ID}_DONE'

FULL_DIR.mkdir(parents=True, exist_ok=True)

# ========================= Config ==========================
SEEDS = [42, 7, 13]
N_EPOCHS = 200
BATCH_SIZE = 64  # will be probed/adjusted
LR = 1e-4
WEIGHT_DECAY = 1e-3
EMBED_DIM = 384
N_LAYERS = 4
NUM_HEADS = 8
MLP_DIM = 512
IMG_SIZE = 64
HISTORY_SIZE = 3
NUM_PREDS = 1
FRAMESKIP = 5
CHECKPOINT_INTERVAL = 50  # save every 50 epochs
N_TRAJ_PER_COMBO = 200    # full data


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


# ========================= GPU Progress ==========================
def update_gpu_progress(status, actual_min=None, config_snapshot=None):
    """Update gpu_progress.json with task completion."""
    try:
        if GPU_PROGRESS_FILE.exists():
            data = json.loads(GPU_PROGRESS_FILE.read_text())
        else:
            data = {'completed': [], 'failed': [], 'running': {}, 'timings': {}}

        if status == 'completed':
            if TASK_ID not in data.get('completed', []):
                data.setdefault('completed', []).append(TASK_ID)
            if TASK_ID in data.get('running', {}):
                del data['running'][TASK_ID]
        elif status == 'failed':
            if TASK_ID not in data.get('failed', []):
                data.setdefault('failed', []).append(TASK_ID)
            if TASK_ID in data.get('running', {}):
                del data['running'][TASK_ID]

        if actual_min is not None:
            data.setdefault('timings', {})[TASK_ID] = {
                'planned_min': 360,
                'actual_min': actual_min,
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'config_snapshot': config_snapshot or {},
            }

        GPU_PROGRESS_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f'[WARN] Failed to update gpu_progress.json: {e}')


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


def load_split_data(split_combos, data_dir, n_traj_per_combo=200, seed=42):
    """Load all combos for a split."""
    data_dicts = []
    for combo in split_combos:
        g, f, m = combo
        filename = f'g{g}_f{f}_m{m}.h5'
        h5_path = data_dir / filename
        if not h5_path.exists():
            print(f'  [WARN] Missing data file: {h5_path}')
            continue
        d = load_hdf5_data(h5_path, n_traj=n_traj_per_combo, seed=seed)
        data_dicts.append(d)
        print(f'  Loaded {filename}: {d["pixels"].shape[0]} traj')
    return data_dicts


# ========================= Dataset ==========================

class PhysicsDataset(torch.utils.data.Dataset):
    """Dataset for HDF5 files. Returns (frames, physics_labels, joint_angles, com_velocity)."""

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


# ========================= DINOv2 Encoder ==========================

class FrozenDINOv2Encoder(nn.Module):
    """
    Frozen DINOv2-s encoder that extracts 384-dim frame embeddings from 64x64 RGB frames.
    """

    def __init__(self, embed_dim=384, img_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size

        print(f'  [DINOv2] Loading DINOv2-s via timm (img_size={img_size})...')
        try:
            import timm
            self.dino = timm.create_model(
                'vit_small_patch14_dinov2.lvd142m',
                pretrained=True,
                img_size=img_size,
                num_classes=0,
            )
            print(f'  [DINOv2] Loaded via timm successfully')
        except Exception as e:
            print(f'  [DINOv2] timm load failed ({e}), trying torch.hub...')
            try:
                self.dino = torch.hub.load(
                    'facebookresearch/dinov2',
                    'dinov2_vits14',
                    pretrained=True,
                )
                print(f'  [DINOv2] Loaded via torch.hub successfully')
            except Exception as e2:
                raise RuntimeError(f'Could not load DINOv2: timm error: {e}, hub error: {e2}')

        # Freeze ALL encoder parameters
        for param in self.dino.parameters():
            param.requires_grad = False

        n_frozen = sum(p.numel() for p in self.dino.parameters())
        print(f'  [DINOv2] Frozen {n_frozen:,} params ({n_frozen/1e6:.1f}M)')

        # Projection to match predictor embed_dim if needed
        dino_dim = self._get_dino_output_dim()
        self.proj = None
        if dino_dim != embed_dim:
            self.proj = nn.Linear(dino_dim, embed_dim)
            print(f'  [DINOv2] Added projection: {dino_dim} -> {embed_dim}')

    def _get_dino_output_dim(self):
        try:
            return self.dino.num_features
        except AttributeError:
            return 384

    def forward(self, x):
        """
        x: (B, T, 3, H, W) or (B, 3, H, W) - float32, [0, 1]
        Returns: (B, T, embed_dim) or (B, embed_dim)
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
        else:
            B = x.shape[0]
            T = None
            x_flat = x

        with torch.no_grad():
            feat = self.dino(x_flat)

        if self.proj is not None:
            feat = self.proj(feat)

        if T is not None:
            return feat.view(B, T, -1)
        return feat


# ========================= Predictor ==========================

class PredictorBlock(nn.Module):
    """Single transformer block for the predictor (same as LeWM)."""

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


# ========================= DINO-WM Model ==========================

class DINOWM(nn.Module):
    """
    DINO-WM: Frozen DINOv2-s encoder + learned transformer predictor.

    Architecture:
      - Encoder: frozen DINOv2-s -> 384-dim frame embeddings
      - Predictor: 4-layer transformer -> predict next-step embedding
      - Loss: MSE between predicted embedding and target (stop-gradient)
    """

    def __init__(self, embed_dim=384, n_layers=4, num_heads=8, mlp_dim=512,
                 img_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.history_size = HISTORY_SIZE
        self.num_preds = NUM_PREDS

        # Frozen DINOv2 encoder
        self.encoder = FrozenDINOv2Encoder(embed_dim=embed_dim, img_size=img_size)

        # Learned position embeddings for predictor context
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.history_size, embed_dim) * 0.02
        )

        # Learned transformer predictor
        self.predictor_blocks = nn.ModuleList([
            PredictorBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(n_layers)
        ])
        self.pred_norm = nn.LayerNorm(embed_dim)

    def encode(self, frames):
        """
        frames: (B, T, H, W, 3) uint8 -> embeddings (B, T, D)
        """
        x = frames.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)  # (B, T, 3, H, W)
        return self.encoder(x)

    def predict(self, ctx_emb):
        """
        ctx_emb: (B, T_ctx, D) -> (B, T_ctx, D) predicted next
        """
        x = ctx_emb + self.pos_embed[:, :ctx_emb.size(1)]
        for block in self.predictor_blocks:
            x = block(x)
        return self.pred_norm(x)

    def forward(self, frames):
        """
        frames: (B, T_total, H, W, 3), T_total = history_size + num_preds = 4
        """
        emb = self.encode(frames)  # (B, T_total, D)
        ctx_emb = emb[:, :self.history_size]
        tgt_emb = emb[:, self.history_size:self.history_size + self.num_preds]

        pred_emb = self.predict(ctx_emb)
        pred = pred_emb[:, -1:]

        tgt_sg = tgt_emb.detach()
        loss = F.mse_loss(pred, tgt_sg)

        return {
            'loss': loss,
            'emb': emb,
            'pred': pred,
            'tgt': tgt_emb,
        }

    def get_predictor_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ========================= VRAM Probing ==========================

def probe_max_batch_size(model, sample_dataset, device, start=128, min_bs=8):
    """Binary search for max stable batch size."""
    high, best = start, min_bs
    while min_bs <= high:
        mid = (min_bs + high) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            loader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=mid, shuffle=False, num_workers=0
            )
            batch = next(iter(loader))
            frames = batch['pixels'].to(device)
            with torch.no_grad():
                out = model(frames)
                loss = out['loss']
            del frames, out, loss, batch
            torch.cuda.empty_cache()
            best = mid
            min_bs = mid + 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            high = mid - 1
    return best


def get_gpu_profile(device):
    """Get GPU VRAM info."""
    try:
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        free = total - reserved
        return {
            'gpu_name': torch.cuda.get_device_name(device),
            'vram_total_mb': round(total / 1024**2),
            'vram_free_mb': round(free / 1024**2),
        }
    except Exception:
        return {}


# ========================= Training ==========================

def train_one_seed(model, train_dataset, seed, n_epochs, device, task_id,
                   output_dir, batch_size=64, lr=LR, weight_decay=WEIGHT_DECAY):
    """Train DINO-WM for one seed. Returns training log dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f'  [Seed {seed}] Trainable params: {n_trainable:,} ({n_trainable/1e6:.2f}M)')

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    loss_history = []
    t_start_seed = time.time()

    for epoch in range(n_epochs):
        model.train()
        model.encoder.dino.eval()  # Always keep DINOv2 in eval mode

        epoch_losses = []

        for batch in loader:
            frames = batch['pixels'].to(device)
            optimizer.zero_grad()
            out = model(frames)
            loss = out['loss']
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        # Progress reporting (every epoch)
        elapsed_min = (time.time() - t_start_seed) / 60
        print(f'  [Seed {seed}] Epoch {epoch+1:3d}/{n_epochs} | '
              f'loss={mean_loss:.5f} | elapsed={elapsed_min:.1f}min')

        write_progress(
            epoch + 1, n_epochs,
            loss=mean_loss,
            metric={
                'phase': f'training_seed{seed}',
                'seed': seed,
                'epoch': epoch + 1,
                'loss': round(mean_loss, 5),
                'elapsed_min': round(elapsed_min, 1),
            }
        )

        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_path = output_dir / f'seed_{seed}_epoch{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'seed': seed,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
                'loss_history': loss_history,
            }, ckpt_path)
            print(f'  [Seed {seed}] Checkpoint saved: {ckpt_path}')

    # Save final checkpoint
    final_path = output_dir / f'seed_{seed}_final.pt'
    torch.save({
        'epoch': n_epochs,
        'seed': seed,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_history[-1],
        'loss_history': loss_history,
    }, final_path)
    print(f'  [Seed {seed}] Final checkpoint saved: {final_path}')

    total_time = time.time() - t_start_seed
    return {
        'seed': seed,
        'n_epochs': n_epochs,
        'loss_history': [round(l, 6) for l in loss_history],
        'initial_loss': round(loss_history[0], 6),
        'final_loss': round(loss_history[-1], 6),
        'loss_drop_pct': round((loss_history[0] - loss_history[-1]) / loss_history[0] * 100, 1),
        'total_time_min': round(total_time / 60, 1),
        'checkpoint_final': str(final_path),
    }


# ========================= Linear Probing ==========================

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def extract_embeddings(model, dataset, device, batch_size=128):
    """Extract encoder embeddings for a dataset."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    all_emb = []
    all_joint = []
    all_com = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            frames = batch['pixels'].to(device)
            emb = model.encode(frames)          # (B, T, D)
            emb_mean = emb.mean(dim=1)          # (B, D)

            all_emb.append(emb_mean.cpu().numpy())
            all_joint.append(batch['joint_angles'].mean(dim=1).numpy())
            all_com.append(batch['com_velocity'].mean(dim=1).numpy())
            all_labels.append(batch['physics_labels'].numpy())

    return {
        'embeddings': np.concatenate(all_emb, axis=0),
        'joint_angles': np.concatenate(all_joint, axis=0),
        'com_velocity': np.concatenate(all_com, axis=0),
        'physics_labels': np.concatenate(all_labels, axis=0),
    }


def run_linear_probe(train_feats, test_feats, target='joint_angle_mean'):
    """Fit linear regression probe and return R²."""
    X_train = train_feats['embeddings']
    X_test = test_feats['embeddings']

    if target == 'joint_angle_mean':
        y_train = train_feats['joint_angles'].mean(axis=-1)
        y_test = test_feats['joint_angles'].mean(axis=-1)
    elif target == 'com_velocity_x':
        y_train = train_feats['com_velocity'][:, 0]
        y_test = test_feats['com_velocity'][:, 0]
    elif target == 'gravity_label':
        y_train = train_feats['physics_labels'][:, 0]
        y_test = test_feats['physics_labels'][:, 0]
    elif target == 'friction_label':
        y_train = train_feats['physics_labels'][:, 1]
        y_test = test_feats['physics_labels'][:, 1]
    elif target == 'mass_label':
        y_train = train_feats['physics_labels'][:, 2]
        y_test = test_feats['physics_labels'][:, 2]
    else:
        return None

    # Check for constant target (R² undefined)
    if np.std(y_test) < 1e-8:
        return {'train_r2': None, 'test_r2': None, 'note': 'constant_target'}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    probe = Ridge(alpha=1e-3)
    probe.fit(X_train_s, y_train)

    train_r2 = float(probe.score(X_train_s, y_train))
    test_r2 = float(probe.score(X_test_s, y_test))

    return {
        'train_r2': round(train_r2, 4),
        'test_r2': round(test_r2, 4),
    }


# ========================= MAIN ==========================

def main():
    t_start = time.time()
    write_pid()
    write_progress(0, N_EPOCHS * len(SEEDS), loss=None, metric={'phase': 'init'})

    print('\n' + '='*60)
    print('FULL: train_dino_wm')
    print(f'Seeds: {SEEDS}, Epochs: {N_EPOCHS}, Split: split_0 (18/27 combos)')
    print('Architecture: Frozen DINOv2-s encoder + learned transformer predictor')
    print('='*60)

    # ===== Load Splits =====
    print('\n[DATA] Loading splits...')
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    split0 = splits['split_0']
    train_combos = split0['train']
    holdout_combos = split0['holdout']
    print(f'  Train combos: {len(train_combos)}, Holdout combos: {len(holdout_combos)}')

    # ===== Load Training Data =====
    print('\n[DATA] Loading training data (18 combos x 200 traj each)...')
    try:
        train_data_dicts = load_split_data(
            train_combos, DATA_DIR, n_traj_per_combo=N_TRAJ_PER_COMBO, seed=42
        )
        print(f'  Loaded {len(train_data_dicts)} training combos')
    except Exception as e:
        print(f'[ERROR] Failed to load training data: {e}')
        traceback.print_exc()
        write_done('failure', f'Data load failed: {e}')
        return

    # Build training dataset
    print('\n[DATA] Building training dataset...')
    train_dataset = PhysicsDataset(
        train_data_dicts, history_size=HISTORY_SIZE,
        num_preds=NUM_PREDS, frameskip=FRAMESKIP
    )
    n_train_seq = len(train_dataset)
    print(f'  Training sequences: {n_train_seq:,}')

    # ===== Load Holdout Data (for final probing) =====
    print('\n[DATA] Loading holdout data (9 combos x 200 traj each)...')
    try:
        holdout_data_dicts = load_split_data(
            holdout_combos, DATA_DIR, n_traj_per_combo=N_TRAJ_PER_COMBO, seed=42
        )
        holdout_dataset = PhysicsDataset(
            holdout_data_dicts, history_size=HISTORY_SIZE,
            num_preds=NUM_PREDS, frameskip=FRAMESKIP
        )
        print(f'  Holdout sequences: {len(holdout_dataset):,}')
    except Exception as e:
        print(f'[WARN] Failed to load holdout data: {e}')
        holdout_dataset = None

    # ===== Initialize Model =====
    print('\n[MODEL] Initializing DINO-WM model...')
    try:
        model = DINOWM(
            embed_dim=EMBED_DIM,
            n_layers=N_LAYERS,
            num_heads=NUM_HEADS,
            mlp_dim=MLP_DIM,
            img_size=IMG_SIZE,
        ).to(DEVICE)
    except Exception as e:
        print(f'[ERROR] Model initialization failed: {e}')
        traceback.print_exc()
        write_done('failure', f'Model init failed: {e}')
        return

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = n_total - n_trainable
    print(f'  Total params: {n_total:,} ({n_total/1e6:.2f}M)')
    print(f'  Trainable (predictor): {n_trainable:,} ({n_trainable/1e6:.2f}M)')
    print(f'  Frozen (DINOv2): {n_frozen:,} ({n_frozen/1e6:.2f}M)')

    # ===== VRAM Probing =====
    print('\n[VRAM] Probing max batch size...')
    gpu_profile = get_gpu_profile(DEVICE)
    print(f'  GPU: {gpu_profile.get("gpu_name", "unknown")}')
    print(f'  VRAM total: {gpu_profile.get("vram_total_mb", "?"):,} MiB')
    print(f'  VRAM free: {gpu_profile.get("vram_free_mb", "?"):,} MiB')

    try:
        sample_dataset = torch.utils.data.Subset(train_dataset, list(range(min(256, len(train_dataset)))))
        max_bs = probe_max_batch_size(model, sample_dataset, DEVICE, start=256, min_bs=8)
        # Use 85% of max for stability
        effective_bs = max(8, int(max_bs * 0.85))
        # But also not less than BATCH_SIZE (configured)
        effective_bs = max(effective_bs, BATCH_SIZE)
        effective_bs = min(effective_bs, max_bs)
        print(f'  Max batch size: {max_bs}, Using: {effective_bs}')
        gpu_profile['max_batch_size'] = max_bs
        gpu_profile['effective_batch_size'] = effective_bs
    except Exception as e:
        print(f'  [WARN] VRAM probe failed ({e}), using default batch size {BATCH_SIZE}')
        effective_bs = BATCH_SIZE

    # Save GPU profile
    gpu_profile_path = FULL_DIR / 'train_dino_wm_gpu_profile.json'
    gpu_profile_path.write_text(json.dumps(gpu_profile, indent=2))

    # ===== Train 3 Seeds =====
    all_seed_results = []
    t_train_start = time.time()

    for i, seed in enumerate(SEEDS):
        print(f'\n{"="*60}')
        print(f'[TRAIN] Seed {seed} ({i+1}/{len(SEEDS)}) - {N_EPOCHS} epochs')
        print(f'{"="*60}')

        # Re-initialize model weights for each seed
        if i > 0:
            print(f'  Re-initializing model for seed {seed}...')
            model = DINOWM(
                embed_dim=EMBED_DIM,
                n_layers=N_LAYERS,
                num_heads=NUM_HEADS,
                mlp_dim=MLP_DIM,
                img_size=IMG_SIZE,
            ).to(DEVICE)

        try:
            seed_result = train_one_seed(
                model, train_dataset, seed, N_EPOCHS, DEVICE,
                TASK_ID, FULL_DIR, batch_size=effective_bs
            )
            all_seed_results.append(seed_result)
            print(f'\n[SEED {seed}] Complete: loss={seed_result["final_loss"]:.5f} '
                  f'({seed_result["loss_drop_pct"]:.1f}% drop), '
                  f'time={seed_result["total_time_min"]:.1f}min')

            # Quick probing after each seed for sanity check
            if holdout_dataset is not None:
                print(f'\n[PROBE] Quick linear probe for seed {seed}...')
                try:
                    sample_train = torch.utils.data.Subset(
                        train_dataset, list(range(min(5000, len(train_dataset))))
                    )
                    sample_holdout = torch.utils.data.Subset(
                        holdout_dataset, list(range(min(2000, len(holdout_dataset))))
                    )
                    train_feats = extract_embeddings(model, sample_train, DEVICE)
                    holdout_feats = extract_embeddings(model, sample_holdout, DEVICE)

                    ja_probe = run_linear_probe(train_feats, holdout_feats, 'joint_angle_mean')
                    com_probe = run_linear_probe(train_feats, holdout_feats, 'com_velocity_x')

                    seed_result['linear_probe'] = {
                        'joint_angle_mean': ja_probe,
                        'com_velocity_x': com_probe,
                    }
                    if ja_probe:
                        print(f'  Joint angle R² - train={ja_probe["train_r2"]}, '
                              f'holdout={ja_probe["test_r2"]}')
                    if com_probe:
                        print(f'  CoM velocity R² - train={com_probe["train_r2"]}, '
                              f'holdout={com_probe["test_r2"]}')
                except Exception as e:
                    print(f'  [WARN] Probing failed: {e}')
                    seed_result['linear_probe'] = {'error': str(e)}

        except Exception as e:
            print(f'[ERROR] Training seed {seed} failed: {e}')
            traceback.print_exc()
            all_seed_results.append({
                'seed': seed,
                'status': 'failed',
                'error': str(e),
            })

    # ===== Compute Summary =====
    total_time = time.time() - t_start
    successful_seeds = [r for r in all_seed_results if 'error' not in r]

    if successful_seeds:
        final_losses = [r['final_loss'] for r in successful_seeds]
        drop_pcts = [r['loss_drop_pct'] for r in successful_seeds]
        summary_stats = {
            'n_seeds_success': len(successful_seeds),
            'n_seeds_total': len(SEEDS),
            'final_loss_mean': round(float(np.mean(final_losses)), 6),
            'final_loss_std': round(float(np.std(final_losses)), 6),
            'loss_drop_pct_mean': round(float(np.mean(drop_pcts)), 1),
        }
    else:
        summary_stats = {
            'n_seeds_success': 0,
            'n_seeds_total': len(SEEDS),
            'error': 'all seeds failed',
        }

    full_results = {
        'task_id': TASK_ID,
        'timestamp': datetime.now().isoformat(),
        'mode': 'FULL',
        'seeds': SEEDS,
        'n_epochs': N_EPOCHS,
        'batch_size': effective_bs,
        'lr': LR,
        'split': 'split_0',
        'n_train_combos': len(train_combos),
        'n_holdout_combos': len(holdout_combos),
        'n_train_sequences': n_train_seq,
        'model_params': {
            'total': n_total,
            'trainable': n_trainable,
            'frozen': n_frozen,
        },
        'gpu_profile': gpu_profile,
        'seed_results': all_seed_results,
        'summary': summary_stats,
        'total_time_min': round(total_time / 60, 1),
        'checkpoint_dir': str(FULL_DIR),
    }

    # Save full results
    results_path = FULL_DIR / 'full_results.json'
    results_path.write_text(json.dumps(full_results, indent=2))
    print(f'\n[SAVE] Results saved to {results_path}')

    # ===== Update GPU Progress =====
    actual_min = round(total_time / 60)
    config_snapshot = {
        'model': 'DINO-WM (frozen DINOv2-s + transformer predictor)',
        'seeds': SEEDS,
        'n_epochs': N_EPOCHS,
        'batch_size': effective_bs,
        'n_train_combos': len(train_combos),
        'gpu_model': gpu_profile.get('gpu_name', 'unknown'),
        'gpu_count': 1,
        'n_seeds_success': summary_stats.get('n_seeds_success', 0),
    }
    update_gpu_progress(
        'completed' if summary_stats.get('n_seeds_success', 0) > 0 else 'failed',
        actual_min=actual_min,
        config_snapshot=config_snapshot,
    )

    # ===== Write DONE marker =====
    n_success = summary_stats.get('n_seeds_success', 0)
    if n_success > 0:
        status = 'success'
        summary = (
            f'DINO-WM full training complete: {n_success}/{len(SEEDS)} seeds. '
            f'Final loss: {summary_stats.get("final_loss_mean", "N/A"):.5f} '
            f'(±{summary_stats.get("final_loss_std", 0):.5f}). '
            f'Loss drop: {summary_stats.get("loss_drop_pct_mean", 0):.1f}%. '
            f'Total time: {round(total_time/60, 1)} min. '
            f'Checkpoints: {FULL_DIR}'
        )
    else:
        status = 'failure'
        summary = f'DINO-WM full training FAILED for all seeds.'

    write_progress(
        N_EPOCHS * len(SEEDS), N_EPOCHS * len(SEEDS),
        loss=summary_stats.get('final_loss_mean', None),
        metric={'status': 'complete', 'n_seeds_success': n_success}
    )
    write_done(status, summary)

    print('\n' + '='*60)
    print(f'FULL TRAINING COMPLETE: {status.upper()}')
    print(f'Seeds: {n_success}/{len(SEEDS)} successful')
    if successful_seeds:
        print(f'Final loss: {summary_stats["final_loss_mean"]:.5f} ± {summary_stats["final_loss_std"]:.5f}')
        print(f'Loss drop: {summary_stats["loss_drop_pct_mean"]:.1f}%')
    print(f'Total time: {round(total_time/60, 1)} min')
    print(f'Checkpoints: {FULL_DIR}')
    print('='*60 + '\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\n[FATAL ERROR] {e}')
        traceback.print_exc()
        write_done('failure', f'Fatal error: {e}')
        sys.exit(1)
