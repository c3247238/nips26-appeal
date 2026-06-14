"""
Pilot: train_dino_wm (PILOT MODE)
====================================
Train DINO-WM (frozen DINOv2-s encoder + learned transformer predictor) on
the same primary 18/27 split (using available pilot data).

Pilot pass criteria:
  - DINOv2 encoder frozen (no grad update on encoder params)
  - Predictor loss decreases over 5 epochs
  - Model runs forward pass on held-out combo without error

Architecture:
  - Encoder: frozen DINOv2-s (timm: vit_small_patch14_dinov2) -> 384-dim patch embeddings
             -> mean-pool across patches -> 384-dim frame embedding
  - Predictor: learned transformer (4 layers, same as LeWM architecture)
             -> predict next-step embedding from 3-frame context

Output:
  exp/results/pilots/train_dino_wm_pilot.json
  exp/results/train_dino_wm_DONE
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

# GPU assignment - CUDA_VISIBLE_DEVICES set externally, use cuda:0
os.environ['MUJOCO_GL'] = 'egl'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

# ========================= Paths ==========================
WORKSPACE = Path('/home/qhxie/AutoResearch-SibylSystem/workspaces/lewm-generalization/current')
DATA_DIR = WORKSPACE / 'exp' / 'data' / 'comphys_lewm' / 'pilot'
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
PILOTS_DIR = RESULTS_DIR / 'pilots'
CODE_DIR = WORKSPACE / 'exp' / 'code'

TASK_ID = 'train_dino_wm'
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


# ========================= Dataset ==========================

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

    DINOv2-s expects 224x224 by default, but we can use it with 64x64 via
    interpolation of position embeddings (supported by timm).

    The encoder is completely frozen - no parameters are updated during training.
    This implements the DINO-WM architecture: frozen visual backbone + learned predictor.
    """

    def __init__(self, embed_dim=384, img_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size

        print(f'  [DINOv2] Loading DINOv2-s via timm (img_size={img_size})...')
        try:
            import timm
            # vit_small_patch14_dinov2 - 384-dim, patch14, pretrained
            # We use img_size=64 with dynamic img_size interpolation
            self.dino = timm.create_model(
                'vit_small_patch14_dinov2.lvd142m',
                pretrained=True,
                img_size=img_size,
                num_classes=0,          # remove classification head
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
        else:
            print(f'  [DINOv2] No projection needed (embed_dim={embed_dim})')

    def _get_dino_output_dim(self):
        """Determine DINOv2 output dimension."""
        try:
            return self.dino.num_features
        except AttributeError:
            return 384  # vit_small_patch14 default

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

        # DINOv2 forward (frozen)
        with torch.no_grad():
            feat = self.dino(x_flat)  # (B*T, dino_dim) - CLS token or pooled

        # Apply projection if needed
        if self.proj is not None:
            feat = self.proj(feat)

        if T is not None:
            return feat.view(B, T, -1)
        return feat


def verify_encoder_frozen(encoder):
    """Verify that the DINOv2 encoder has no gradients flowing through it."""
    frozen_params = []
    trainable_params = []

    for name, param in encoder.named_parameters():
        if 'proj' in name:  # projection layer is trainable
            continue
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    return {
        'n_frozen_params': len(frozen_params),
        'n_trainable_encoder_params': len(trainable_params),
        'encoder_frozen': len(trainable_params) == 0,
        'frozen_params_sample': frozen_params[:3],
        'trainable_params_sample': trainable_params[:3],
    }


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

    This follows the DINO-WM architecture from the original paper:
    "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning"
    (Zhou et al., 2024, ICLR 2025)
    """

    def __init__(self, embed_dim=384, n_layers=4, num_heads=8, mlp_dim=512,
                 img_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.history_size = 3
        self.num_preds = 1

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
        x = frames.float() / 255.0          # normalize to [0, 1]
        x = x.permute(0, 1, 4, 2, 3)        # (B, T, 3, H, W)
        return self.encoder(x)              # (B, T, embed_dim)

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

        Loss: MSE between predicted embedding (from context) and
              target embedding (stop-gradient).

        Returns dict with 'loss', 'emb'.
        """
        emb = self.encode(frames)           # (B, T_total, D)

        ctx_emb = emb[:, :self.history_size]             # (B, 3, D) - context
        # Target is the next frame after context
        tgt_emb = emb[:, self.history_size:self.history_size + self.num_preds]  # (B, 1, D)

        pred_emb = self.predict(ctx_emb)    # (B, 3, D)
        pred = pred_emb[:, -1:]             # (B, 1, D) - last step prediction

        # Stop-gradient on target (JEPA convention)
        tgt_sg = tgt_emb.detach()

        # MSE prediction loss
        loss = F.mse_loss(pred, tgt_sg)

        return {
            'loss': loss,
            'emb': emb,
            'pred': pred,
            'tgt': tgt_emb,
        }

    def get_encoder_params(self):
        """Returns only the frozen DINOv2 parameters (for verification)."""
        return list(self.encoder.dino.parameters())

    def get_predictor_params(self):
        """Returns only the trainable predictor parameters."""
        trainable = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable.append((name, p))
        return trainable


# ========================= Training ==========================

def train_dino_wm(model, dataset, n_epochs=5, lr=1e-4, batch_size=32, device=DEVICE):
    """
    Train DINO-WM predictor on dataset (encoder is frozen).
    Returns loss history.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )

    # Only optimize trainable params (predictor + pos_embed + projection if any)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f'  Trainable params: {n_trainable:,} ({n_trainable/1e6:.2f}M)')

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-3)

    loss_history = []
    initial_loss = None

    for epoch in range(n_epochs):
        model.train()
        model.encoder.dino.eval()  # Always keep DINOv2 in eval mode (frozen BN stats)

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

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        if initial_loss is None:
            initial_loss = mean_loss

        write_progress(
            epoch + 1, n_epochs,
            loss=mean_loss,
            metric={
                'phase': 'dino_wm_training',
                'epoch': epoch + 1,
                'loss': round(mean_loss, 5),
            }
        )

        print(f'  Epoch {epoch+1:3d}/{n_epochs} | pred_loss={mean_loss:.5f}')

    return loss_history, initial_loss


def verify_encoder_not_updated(model, dataset, device, n_samples=8):
    """
    Verify DINOv2 encoder parameters are actually frozen by checking gradients.
    """
    model.train()
    model.encoder.dino.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=True)
    batch = next(iter(loader))
    frames = batch['pixels'].to(device)

    # Zero all gradients
    model.zero_grad()

    out = model(frames)
    out['loss'].backward()

    # Check: DINOv2 params should have no gradient
    dino_grads = []
    for name, param in model.encoder.dino.named_parameters():
        if param.grad is not None:
            dino_grads.append((name, param.grad.abs().max().item()))

    # Check: predictor params should have gradients
    pred_grads = []
    for name, param in model.predictor_blocks.named_parameters():
        if param.grad is not None:
            pred_grads.append((name, param.grad.abs().max().item()))

    return {
        'encoder_has_no_grad': len(dino_grads) == 0,
        'predictor_has_grad': len(pred_grads) > 0,
        'n_dino_params_with_grad': len(dino_grads),
        'n_predictor_params_with_grad': len(pred_grads),
        'dino_grad_samples': dino_grads[:2],
        'pred_grad_sample': pred_grads[:2] if pred_grads else [],
    }


def run_holdout_forward_pass(model, holdout_data, device, n_samples=20):
    """
    Run forward pass on held-out combo (friction=1.0x) without error.
    This verifies the model can process unseen physics combinations.
    """
    model.eval()

    dataset = PhysicsDataset([holdout_data], history_size=3, num_preds=1, frameskip=5)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(n_samples, len(dataset)), shuffle=False
    )

    with torch.no_grad():
        batch = next(iter(loader))
        frames = batch['pixels'].to(device)
        out = model(frames)
        loss = out['loss'].item()
        emb_shape = out['emb'].shape

    return {
        'holdout_forward_pass_ok': True,
        'holdout_loss': round(loss, 5),
        'embedding_shape': list(emb_shape),
        'n_samples': frames.shape[0],
    }


# ========================= Linear Probing ==========================

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def extract_embeddings(model, dataset, device, batch_size=64):
    """Extract encoder embeddings for a dataset."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_emb = []
    all_joint = []
    all_com = []

    with torch.no_grad():
        for batch in loader:
            frames = batch['pixels'].to(device)
            emb = model.encode(frames)          # (B, T, D)
            emb_mean = emb.mean(dim=1)          # (B, D) - mean over time

            all_emb.append(emb_mean.cpu().numpy())
            all_joint.append(batch['joint_angles'].mean(dim=1).numpy())
            all_com.append(batch['com_velocity'].mean(dim=1).numpy())

    return {
        'embeddings': np.concatenate(all_emb, axis=0),
        'joint_angles': np.concatenate(all_joint, axis=0),
        'com_velocity': np.concatenate(all_com, axis=0),
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
    else:
        return None

    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Ridge regression probe
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
            'description': '1-seed 5-epoch DINO-WM training; verify DINOv2 encoder frozen '
                           'and predictor trains correctly',
        }
    }

    print('\n' + '='*60)
    print('PILOT: train_dino_wm')
    print('Architecture: Frozen DINOv2-s encoder + learned transformer predictor')
    print('='*60)

    # ===== Load Data =====
    print('\n[DATA] Loading pilot data (100 traj, friction={0.5x, 2.0x}, gravity=1.0g, mass=1.0x)...')

    try:
        data_05 = load_hdf5_data(DATA_DIR / 'g1.0_f0.5_m1.0.h5', n_traj=100, seed=42)
        data_20 = load_hdf5_data(DATA_DIR / 'g1.0_f2.0_m1.0.h5', n_traj=100, seed=42)
        data_holdout = load_hdf5_data(DATA_DIR / 'g1.0_f1.0_m1.0.h5', n_traj=20, seed=42)
        print(f'  Loaded friction=0.5x: {data_05["pixels"].shape[0]} traj, shape={data_05["pixels"].shape}')
        print(f'  Loaded friction=2.0x: {data_20["pixels"].shape[0]} traj, shape={data_20["pixels"].shape}')
        print(f'  Loaded holdout (friction=1.0x): {data_holdout["pixels"].shape[0]} traj for forward pass test')
    except Exception as e:
        print(f'[ERROR] Failed to load data: {e}')
        traceback.print_exc()
        write_done('failure', f'Data load failed: {e}')
        return

    # Build training dataset
    train_dataset = PhysicsDataset([data_05, data_20], history_size=3, num_preds=1, frameskip=5)
    holdout_dataset = PhysicsDataset([data_holdout], history_size=3, num_preds=1, frameskip=5)
    results['dataset_size'] = len(train_dataset)

    # ===== Initialize DINO-WM Model =====
    print('\n[MODEL] Initializing DINO-WM model...')

    try:
        model = DINOWM(
            embed_dim=384,
            n_layers=4,
            num_heads=8,
            mlp_dim=512,
            img_size=64,
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

    results['model_params'] = {
        'total': n_total,
        'trainable': n_trainable,
        'frozen': n_frozen,
        'total_M': round(n_total / 1e6, 2),
        'trainable_M': round(n_trainable / 1e6, 2),
        'frozen_M': round(n_frozen / 1e6, 2),
    }

    # ===== Verify Encoder is Frozen (requires_grad check) =====
    print('\n[VERIFY] Checking encoder parameter freeze status...')
    freeze_verify = verify_encoder_frozen(model.encoder)
    print(f'  Encoder frozen: {freeze_verify["encoder_frozen"]}')
    print(f'  Frozen DINOv2 params: {freeze_verify["n_frozen_params"]}')
    print(f'  Trainable encoder params (unexpected): {freeze_verify["n_trainable_encoder_params"]}')
    if freeze_verify["trainable_params_sample"]:
        print(f'  WARN: Unexpected trainable params: {freeze_verify["trainable_params_sample"]}')
    results['encoder_freeze_verify'] = freeze_verify

    # ===== Test Forward Pass Before Training =====
    print('\n[VERIFY] Pre-training forward pass test...')
    try:
        dummy_batch = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=4)))
        with torch.no_grad():
            out = model(dummy_batch['pixels'].to(DEVICE))
        print(f'  Forward pass OK: loss={out["loss"].item():.5f}, emb_shape={out["emb"].shape}')
        results['pretrain_forward_pass'] = {
            'ok': True,
            'initial_pred_loss': round(out['loss'].item(), 5),
            'emb_shape': list(out['emb'].shape),
        }
    except Exception as e:
        print(f'  [ERROR] Forward pass failed: {e}')
        traceback.print_exc()
        write_done('failure', f'Pre-training forward pass failed: {e}')
        return

    # ===== Train Predictor =====
    print('\n[TRAIN] Training DINO-WM predictor (5 epochs, 100 traj x 2 friction levels)...')
    print('  (DINOv2 encoder remains frozen throughout)')
    write_progress(0, 5, metric={'phase': 'training'})

    t_train_start = time.time()

    try:
        loss_history, initial_loss = train_dino_wm(
            model, train_dataset,
            n_epochs=5, lr=1e-4, batch_size=32, device=DEVICE
        )
    except Exception as e:
        print(f'[ERROR] Training failed: {e}')
        traceback.print_exc()
        results['training_error'] = str(e)
        write_done('failure', f'Training failed: {e}')
        return

    train_time = time.time() - t_train_start
    print(f'\n  Training complete: 5 epochs in {train_time:.1f}s ({train_time/60:.1f} min)')

    final_loss = loss_history[-1]
    loss_drop_pct = (initial_loss - final_loss) / initial_loss * 100

    results['training'] = {
        'n_epochs': 5,
        'n_traj': 100,
        'lr': 1e-4,
        'batch_size': 32,
        'loss_history': [round(l, 6) for l in loss_history],
        'initial_loss': round(initial_loss, 6),
        'final_loss': round(final_loss, 6),
        'loss_drop_pct': round(loss_drop_pct, 1),
        'train_time_sec': round(train_time, 1),
    }

    # ===== Verify Encoder Still Frozen After Training =====
    print('\n[VERIFY] Verifying encoder not updated during training...')
    try:
        grad_verify = verify_encoder_not_updated(model, train_dataset, DEVICE)
        print(f'  Encoder has no grad: {grad_verify["encoder_has_no_grad"]}')
        print(f'  Predictor has grad: {grad_verify["predictor_has_grad"]}')
        print(f'  DINOv2 params with grad (should be 0): {grad_verify["n_dino_params_with_grad"]}')
        print(f'  Predictor params with grad: {grad_verify["n_predictor_params_with_grad"]}')
        results['grad_verify'] = grad_verify
    except Exception as e:
        print(f'  [WARN] Gradient verification failed: {e}')
        results['grad_verify'] = {'error': str(e)}

    # ===== Holdout Forward Pass =====
    print('\n[VERIFY] Running forward pass on holdout combo (friction=1.0x)...')
    try:
        holdout_result = run_holdout_forward_pass(model, data_holdout, DEVICE, n_samples=20)
        print(f'  Holdout forward pass OK: loss={holdout_result["holdout_loss"]:.5f}')
        print(f'  Embedding shape: {holdout_result["embedding_shape"]}')
        results['holdout_forward_pass'] = holdout_result
    except Exception as e:
        print(f'  [ERROR] Holdout forward pass failed: {e}')
        traceback.print_exc()
        results['holdout_forward_pass'] = {'ok': False, 'error': str(e)}

    # ===== Linear Probing (quick sanity check) =====
    print('\n[PROBE] Running quick linear probe (sanity check)...')
    try:
        train_feats = extract_embeddings(model, train_dataset, DEVICE)
        holdout_feats = extract_embeddings(model, holdout_dataset, DEVICE)

        ja_probe = run_linear_probe(train_feats, holdout_feats, 'joint_angle_mean')
        com_probe = run_linear_probe(train_feats, holdout_feats, 'com_velocity_x')

        print(f'  Joint angle R² - train={ja_probe["train_r2"]:.4f}, holdout={ja_probe["test_r2"]:.4f}')
        print(f'  CoM velocity R² - train={com_probe["train_r2"]:.4f}, holdout={com_probe["test_r2"]:.4f}')

        results['linear_probe'] = {
            'joint_angle_mean': ja_probe,
            'com_velocity_x': com_probe,
        }
    except Exception as e:
        print(f'  [WARN] Linear probe failed: {e}')
        traceback.print_exc()
        results['linear_probe'] = {'error': str(e)}

    # ===== Pass Criteria Evaluation =====
    print('\n[PASS CRITERIA] Evaluating pilot pass criteria...')

    loss_decreasing = (loss_history[-1] < loss_history[0]) if len(loss_history) >= 2 else False

    # Core criterion 1: encoder frozen (no grad on DINOv2 params)
    encoder_frozen_ok = results.get('encoder_freeze_verify', {}).get('encoder_frozen', False)
    grad_frozen_ok = results.get('grad_verify', {}).get('encoder_has_no_grad', False)
    encoder_really_frozen = encoder_frozen_ok and grad_frozen_ok

    # Core criterion 2: predictor loss decreases
    predictor_trains = loss_decreasing

    # Core criterion 3: holdout forward pass succeeds
    holdout_ok = results.get('holdout_forward_pass', {}).get('holdout_forward_pass_ok', False)

    # Bonus: predictor loss drop meaningful (>5% in 5 epochs)
    meaningful_drop = loss_drop_pct > 5.0

    pass_criteria = {
        'encoder_frozen_requires_grad_check': bool(encoder_frozen_ok),
        'encoder_frozen_gradient_check': bool(grad_frozen_ok),
        'encoder_frozen': bool(encoder_really_frozen),
        'predictor_loss_decreasing': bool(predictor_trains),
        'holdout_forward_pass': bool(holdout_ok),
        'loss_drop_meaningful': bool(meaningful_drop),
        'loss_drop_pct': round(loss_drop_pct, 1),
        'initial_loss': round(initial_loss, 5),
        'final_loss': round(final_loss, 5),
    }

    core_pass = encoder_really_frozen and predictor_trains and holdout_ok
    pass_criteria['core_pass'] = bool(core_pass)

    results['pass_criteria'] = pass_criteria

    print('\n  Pass Criteria:')
    for k, v in pass_criteria.items():
        print(f'    {k}: {v}')

    # ===== Go/No-Go =====
    if core_pass:
        go_no_go = 'GO'
        status = 'success'
        summary = (
            f'DINO-WM pilot PASSED core criteria. '
            f'DINOv2 encoder frozen (no grad on {n_frozen:,} params). '
            f'Predictor ({n_trainable:,} params) loss decreased {loss_drop_pct:.1f}% '
            f'({initial_loss:.5f} -> {final_loss:.5f}). '
            f'Holdout forward pass OK (loss={results.get("holdout_forward_pass", {}).get("holdout_loss", "N/A")}). '
            f'PROCEED to full 3-seed 200-epoch run.'
        )
    else:
        go_no_go = 'NO_GO'
        status = 'failure'
        failed = []
        if not encoder_really_frozen:
            failed.append('encoder_not_frozen')
        if not predictor_trains:
            failed.append('predictor_loss_not_decreasing')
        if not holdout_ok:
            failed.append('holdout_forward_pass_failed')
        summary = f'DINO-WM pilot FAILED: {failed}. INVESTIGATE before full run.'

    results['go_no_go'] = go_no_go
    results['status'] = status

    # ===== Timing =====
    total_time = time.time() - t_start
    results['total_time_sec'] = round(total_time, 1)
    results['total_time_min'] = round(total_time / 60, 1)

    # ===== Save Results =====
    output_path = PILOTS_DIR / f'{TASK_ID}_pilot.json'
    output_path.write_text(json.dumps(results, indent=2))
    print(f'\n[SAVE] Results saved to {output_path}')

    write_progress(
        5, 5,
        loss=final_loss,
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
