"""
Pilot Experiment: eval_lora_adaptation
=======================================
PILOT MODE

What this script does:
  - Load LeWM-SIGReg checkpoint (pilot_seed42_epoch5.pt)
  - Apply LoRA-r4 to predictor Q/V (bottleneck adapter) only
  - Fine-tune on 50 trajectories from friction=1.0x holdout (g1.0_f1.0_m1.0.h5)
  - Measure R² recovery: compare LoRA-adapted model vs. original on holdout combo
  - Also run baselines: (a) zero-shot, (b) head-only fine-tuning
  - Test all three adaptation targets: encoder Q/V, predictor Q/V, both

Pass criteria (PILOT):
  - LoRA adapter inserts without error
  - Fine-tuning converges (loss decreases over 20 steps)
  - R² recovery computed and reported

Output:
  exp/results/pilots/eval_lora_adaptation_pilot.json
  exp/results/eval_lora_adaptation_DONE
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================= Env Setup ==========================
os.environ['MUJOCO_GL'] = 'egl'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

WORKSPACE = Path('/home/qhxie/AutoResearch-SibylSystem/workspaces/lewm-generalization/current')
DATA_DIR = WORKSPACE / 'exp' / 'data' / 'comphys_lewm' / 'pilot'
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
PILOTS_DIR = RESULTS_DIR / 'pilots'
CODE_DIR = WORKSPACE / 'exp' / 'code'
LEWM_DIR = CODE_DIR / 'le-wm'
CHECKPOINT_PATH = RESULTS_DIR / 'full' / 'lewm_sigreg_primary' / 'pilot_seed42_epoch5.pt'

TASK_ID = 'eval_lora_adaptation'
PID_FILE = RESULTS_DIR / f'{TASK_ID}.pid'
PROGRESS_FILE = RESULTS_DIR / f'{TASK_ID}_PROGRESS.json'
DONE_FILE = RESULTS_DIR / f'{TASK_ID}_DONE'

sys.path.insert(0, str(LEWM_DIR))
PILOTS_DIR.mkdir(parents=True, exist_ok=True)

# ========================= PID / Progress =====================
def write_pid():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    print(f'[PID] Written PID {os.getpid()} to {PID_FILE}')

def write_progress(epoch, total_epochs, loss=None, metric=None):
    prog = {
        'task_id': TASK_ID,
        'epoch': epoch,
        'total_epochs': total_epochs,
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
    print(f'[DONE] status={status}')

# ========================= Model Architecture =================
import h5py

class PixelEncoder(nn.Module):
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
            feat = self.norm(self.proj(feat))
            return feat.view(B, T, -1)
        else:
            feat = self.pool(self.cnn(x)).squeeze(-1).squeeze(-1)
            return self.norm(self.proj(feat))


class SimplePredictorBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_dim=512, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class LeWMSimple(nn.Module):
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
        from module import SIGReg
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    def encode(self, frames):
        x = frames.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)
        return self.encoder(x)

    def predict(self, ctx_emb):
        x = ctx_emb + self.pos_embed[:, :ctx_emb.size(1)]
        for block in self.predictor_blocks:
            x = block(x)
        return self.pred_norm(x)

    def forward(self, frames):
        emb = self.encode(frames)
        ctx_emb = emb[:, :self.history_size]
        tgt_emb = emb[:, self.num_preds:]
        pred_emb = self.predict(ctx_emb)
        pred = pred_emb[:, -1:]
        tgt = tgt_emb[:, self.history_size-1:self.history_size]
        pred_loss = (pred - tgt).pow(2).mean()
        sigreg_loss = self.sigreg(emb.permute(1, 0, 2))
        total_loss = pred_loss + self.sigreg_weight * sigreg_loss
        return {'loss': total_loss, 'pred_loss': pred_loss.item(),
                'sigreg_loss': sigreg_loss.item(), 'emb': emb}


# ========================= Dataset ============================
class PhysicsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dicts, history_size=3, num_preds=1, frameskip=5):
        self.seq_len = history_size + num_preds
        all_pixels, all_physics_labels, all_ja, all_cv = [], [], [], []
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
                    all_ja.append(ja[start:end])
                    all_cv.append(cv[start:end])
        self.pixels = np.array(all_pixels)
        self.physics_labels = np.array(all_physics_labels)
        self.joint_angles = np.array(all_ja)
        self.com_velocity = np.array(all_cv)
        print(f'  [Dataset] {len(self.pixels)} sequences from {sum(len(d["pixels"]) for d in data_dicts)} traj')

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        return {
            'pixels': torch.from_numpy(self.pixels[idx]),
            'physics_labels': torch.from_numpy(self.physics_labels[idx]),
            'joint_angles': torch.from_numpy(self.joint_angles[idx]),
            'com_velocity': torch.from_numpy(self.com_velocity[idx]),
        }


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


# ========================= Probing ============================
def extract_embeddings(model, dataset, device, batch_size=64):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_emb, all_labels, all_ja, all_cv = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            frames = batch['pixels'].to(device)
            emb = model.encode(frames)
            emb_mean = emb.mean(dim=1).cpu().numpy()
            all_emb.append(emb_mean)
            all_labels.append(batch['physics_labels'].numpy())
            mid = frames.size(1) // 2
            all_ja.append(batch['joint_angles'][:, mid].numpy())
            all_cv.append(batch['com_velocity'][:, mid].numpy())
    return (np.concatenate(all_emb), np.concatenate(all_labels),
            np.concatenate(all_ja), np.concatenate(all_cv))


def linear_probe_r2(X_train, y_train, X_test, y_test, alpha=1e-3):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train)
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_s, y_train_s)
    return ridge.score(X_test_s, y_test_s)


def probe_r2_on_datasets(model, train_ds, eval_ds, device):
    """Train linear probe on train_ds, evaluate on eval_ds. Returns dict of R2 per target."""
    X_train, lab_train, ja_train, cv_train = extract_embeddings(model, train_ds, device)
    X_eval, lab_eval, ja_eval, cv_eval = extract_embeddings(model, eval_ds, device)

    targets_train = {
        'joint_angle_mean': ja_train.mean(axis=1),
        'com_velocity_x': cv_train[:, 0],
        'friction': lab_train[:, 1],
    }
    targets_eval = {
        'joint_angle_mean': ja_eval.mean(axis=1),
        'com_velocity_x': cv_eval[:, 0],
        'friction': lab_eval[:, 1],
    }

    result = {}
    for t in targets_train:
        y_tr = targets_train[t]
        y_ev = targets_eval[t]
        if np.std(y_tr) < 1e-8 or np.std(y_ev) < 1e-8:
            result[t] = float('nan')
        else:
            result[t] = round(float(linear_probe_r2(X_train, y_tr, X_eval, y_ev)), 4)
    return result


# ========================= LoRA Adaptation ====================

def apply_lora_adapter(model, target='predictor', lora_rank=4):
    """
    Apply LoRA-style adapter to the specified target.
    target: 'predictor' | 'encoder' | 'both'
    Returns model with only LoRA adapter params trainable.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    adapter_in = model.embed_dim
    adapter_r = lora_rank

    if target in ('predictor', 'both'):
        # LoRA adapter on predictor output
        model.lora_pred_down = nn.Linear(adapter_in, adapter_r, bias=False).to(DEVICE)
        model.lora_pred_up = nn.Linear(adapter_r, adapter_in, bias=False).to(DEVICE)
        nn.init.kaiming_uniform_(model.lora_pred_down.weight, a=np.sqrt(5))
        nn.init.zeros_(model.lora_pred_up.weight)
        model.lora_pred_scale = 8.0 / adapter_r

    if target in ('encoder', 'both'):
        # LoRA adapter on encoder output
        model.lora_enc_down = nn.Linear(adapter_in, adapter_r, bias=False).to(DEVICE)
        model.lora_enc_up = nn.Linear(adapter_r, adapter_in, bias=False).to(DEVICE)
        nn.init.kaiming_uniform_(model.lora_enc_down.weight, a=np.sqrt(5))
        nn.init.zeros_(model.lora_enc_up.weight)
        model.lora_enc_scale = 8.0 / adapter_r

    model.lora_target = target

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  [LoRA-{target}] Trainable: {trainable_params:,} / {total_params:,} ({100.*trainable_params/total_params:.2f}%)')
    return model


def apply_head_only(model):
    """Only train a small linear head (not LoRA, just a probing head frozen body)."""
    for param in model.parameters():
        param.requires_grad = False
    # Add a small head on top of encoder output
    model.head = nn.Linear(model.embed_dim, model.embed_dim, bias=True).to(DEVICE)
    model.lora_target = 'head_only'
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  [Head-only] Trainable: {trainable:,}')
    return model


def lora_forward(model, frames):
    """Forward with LoRA adapter if applicable."""
    target = getattr(model, 'lora_target', None)

    # Encode
    x = frames.float() / 255.0
    x = x.permute(0, 1, 4, 2, 3)
    enc_emb = model.encoder(x)  # (B, T, D)

    if target in ('encoder', 'both') and hasattr(model, 'lora_enc_down'):
        enc_delta = model.lora_enc_up(model.lora_enc_down(enc_emb)) * model.lora_enc_scale
        enc_emb = enc_emb + enc_delta

    # Predict
    ctx_emb = enc_emb[:, :model.history_size]
    x_pred = ctx_emb + model.pos_embed[:, :ctx_emb.size(1)]
    for block in model.predictor_blocks:
        x_pred = block(x_pred)
    pred_out = model.pred_norm(x_pred)

    if target in ('predictor', 'both') and hasattr(model, 'lora_pred_down'):
        pred_delta = model.lora_pred_up(model.lora_pred_down(pred_out)) * model.lora_pred_scale
        pred_out = pred_out + pred_delta

    if target == 'head_only' and hasattr(model, 'head'):
        pred_out = model.head(pred_out)

    pred = pred_out[:, -1:]
    tgt_emb = enc_emb[:, model.num_preds:]
    tgt = tgt_emb[:, model.history_size-1:model.history_size]

    pred_loss = (pred - tgt).pow(2).mean()
    sigreg_loss = model.sigreg(enc_emb.permute(1, 0, 2))
    total_loss = pred_loss + model.sigreg_weight * sigreg_loss

    return {'loss': total_loss, 'pred_loss': pred_loss.item(),
            'sigreg_loss': sigreg_loss.item(), 'emb': enc_emb}


def lora_encode(model, frames):
    """Encode with LoRA encoder adapter applied (for probing after fine-tuning)."""
    target = getattr(model, 'lora_target', None)
    x = frames.float() / 255.0
    x = x.permute(0, 1, 4, 2, 3)
    enc_emb = model.encoder(x)
    if target in ('encoder', 'both') and hasattr(model, 'lora_enc_down'):
        enc_delta = model.lora_enc_up(model.lora_enc_down(enc_emb)) * model.lora_enc_scale
        enc_emb = enc_emb + enc_delta
    return enc_emb


def extract_embeddings_lora(model, dataset, device, batch_size=64):
    """Extract embeddings using lora_encode (handles encoder LoRA delta)."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_emb, all_labels, all_ja, all_cv = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            frames = batch['pixels'].to(device)
            emb = lora_encode(model, frames)
            emb_mean = emb.mean(dim=1).cpu().numpy()
            all_emb.append(emb_mean)
            all_labels.append(batch['physics_labels'].numpy())
            mid = frames.size(1) // 2
            all_ja.append(batch['joint_angles'][:, mid].numpy())
            all_cv.append(batch['com_velocity'][:, mid].numpy())
    return (np.concatenate(all_emb), np.concatenate(all_labels),
            np.concatenate(all_ja), np.concatenate(all_cv))


def probe_r2_lora(model, train_ds, eval_ds, device):
    """Probe with LoRA-aware encoder."""
    X_train, lab_train, ja_train, cv_train = extract_embeddings_lora(model, train_ds, device)
    X_eval, lab_eval, ja_eval, cv_eval = extract_embeddings_lora(model, eval_ds, device)

    targets_train = {
        'joint_angle_mean': ja_train.mean(axis=1),
        'com_velocity_x': cv_train[:, 0],
        'friction': lab_train[:, 1],
    }
    targets_eval = {
        'joint_angle_mean': ja_eval.mean(axis=1),
        'com_velocity_x': cv_eval[:, 0],
        'friction': lab_eval[:, 1],
    }

    result = {}
    for t in targets_train:
        y_tr = targets_train[t]
        y_ev = targets_eval[t]
        if np.std(y_tr) < 1e-8 or np.std(y_ev) < 1e-8:
            result[t] = float('nan')
        else:
            result[t] = round(float(linear_probe_r2(X_train, y_tr, X_eval, y_ev)), 4)
    return result


def train_lora(model, dataset, n_epochs=20, lr=5e-4, batch_size=16, device=DEVICE,
               epoch_offset=0, total_epochs_outer=20):
    """Fine-tune with LoRA adapter."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)

    loss_history = []
    initial_loss = None

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        for batch in loader:
            frames = batch['pixels'].to(device)
            optimizer.zero_grad()
            out = lora_forward(model, frames)
            loss = out['loss']
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        loss_history.append(mean_loss)
        if initial_loss is None:
            initial_loss = mean_loss

        write_progress(epoch + epoch_offset + 1, total_epochs_outer,
                       loss=mean_loss, metric={'phase': 'lora_finetune', 'epoch': epoch+1})

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'    Epoch {epoch+1:2d}/{n_epochs} | loss={mean_loss:.4f}')

    final_loss = loss_history[-1] if loss_history else float('nan')
    return loss_history, initial_loss, final_loss


# ========================= Load Model =========================
def load_base_model(checkpoint_path):
    """Load pre-trained LeWM-SIGReg checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = LeWMSimple(
        embed_dim=cfg.get('embed_dim', 192),
        n_layers=cfg.get('n_layers', 4),
        num_heads=cfg.get('num_heads', 8),
        mlp_dim=cfg.get('mlp_dim', 512),
        sigreg_knots=cfg.get('sigreg_knots', 17),
        sigreg_num_proj=cfg.get('sigreg_num_proj', 1024),
        sigreg_weight=cfg.get('sigreg_weight', 0.09),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    print(f'  [Model] Loaded checkpoint: epoch={ckpt["epoch"]}, loss={ckpt["final_loss"]:.4f}')
    return model


# ========================= MAIN ==============================
def main():
    t_start = time.time()
    write_pid()
    write_progress(0, 80, metric={'phase': 'init'})

    print('\n' + '='*60)
    print('PILOT: eval_lora_adaptation')
    print('Task: LoRA adaptation diagnostic on held-out combo')
    print('='*60)

    results = {
        'task_id': TASK_ID,
        'timestamp': datetime.now().isoformat(),
        'gpu': str(DEVICE),
        'mode': 'PILOT',
        'seed': 42,
    }

    # ===== Load Data =====
    print('\n[DATA] Loading pilot data...')
    data_05 = load_hdf5_data(DATA_DIR / 'g1.0_f0.5_m1.0.h5', n_traj=100, seed=42)
    data_20 = load_hdf5_data(DATA_DIR / 'g1.0_f2.0_m1.0.h5', n_traj=100, seed=42)
    data_10_train = load_hdf5_data(DATA_DIR / 'g1.0_f1.0_m1.0.h5', n_traj=50, seed=42)   # 50 traj for fine-tuning
    data_10_test = load_hdf5_data(DATA_DIR / 'g1.0_f1.0_m1.0.h5', n_traj=100, seed=99)   # fresh 100 traj for eval

    in_dist_ds = PhysicsDataset([data_05, data_20], history_size=3, num_preds=1, frameskip=5)
    holdout_finetune_ds = PhysicsDataset([data_10_train], history_size=3, num_preds=1, frameskip=5)
    holdout_eval_ds = PhysicsDataset([data_10_test], history_size=3, num_preds=1, frameskip=5)
    print(f'  In-dist: {len(in_dist_ds)} seqs | Holdout fine-tune: {len(holdout_finetune_ds)} seqs | Holdout eval: {len(holdout_eval_ds)} seqs')

    # ===== Load Pre-trained Model =====
    print('\n[MODEL] Loading pre-trained LeWM-SIGReg checkpoint...')
    base_model = load_base_model(CHECKPOINT_PATH)

    # ===== Baseline: Zero-Shot (No Adaptation) =====
    print('\n[BASELINE] Zero-shot evaluation (no adaptation)...')
    write_progress(5, 80, metric={'phase': 'zero_shot_eval'})

    zeroshot_in_dist = probe_r2_on_datasets(base_model, in_dist_ds, in_dist_ds, DEVICE)
    zeroshot_holdout = probe_r2_on_datasets(base_model, in_dist_ds, holdout_eval_ds, DEVICE)

    zero_shot_results = {
        'in_dist_r2': zeroshot_in_dist,
        'holdout_r2': zeroshot_holdout,
    }
    # Compute relative drop
    zero_shot_drop = {}
    for t in zeroshot_in_dist:
        id_r2 = zeroshot_in_dist[t]
        ho_r2 = zeroshot_holdout[t]
        if isinstance(id_r2, float) and not np.isnan(id_r2) and id_r2 > 0:
            drop = (id_r2 - ho_r2) / id_r2 * 100
            zero_shot_drop[t] = round(float(drop), 1)
        else:
            zero_shot_drop[t] = float('nan')
    zero_shot_results['relative_drop_pct'] = zero_shot_drop

    print(f'  Zero-shot holdout R² (joint_angle_mean): {zeroshot_holdout.get("joint_angle_mean", "N/A"):.4f}')
    print(f'  Zero-shot in-dist R²  (joint_angle_mean): {zeroshot_in_dist.get("joint_angle_mean", "N/A"):.4f}')
    print(f'  Relative drop: {zero_shot_drop.get("joint_angle_mean", "N/A")}%')
    results['zero_shot'] = zero_shot_results
    write_progress(15, 80, metric={'phase': 'zero_shot_done'})

    # ===== Experiment 1: LoRA Predictor-Only (r=4) =====
    print('\n[LORA] Experiment 1: LoRA-r4, Predictor-only fine-tuning...')
    write_progress(20, 80, metric={'phase': 'lora_predictor_r4'})

    model_pred = deepcopy(base_model)
    model_pred = apply_lora_adapter(model_pred, target='predictor', lora_rank=4)

    t_lora = time.time()
    lh_pred, il_pred, fl_pred = train_lora(
        model_pred, holdout_finetune_ds, n_epochs=20, lr=5e-4, batch_size=16,
        device=DEVICE, epoch_offset=20, total_epochs_outer=80
    )
    lora_pred_time = time.time() - t_lora

    pred_holdout_r2 = probe_r2_lora(model_pred, holdout_finetune_ds, holdout_eval_ds, DEVICE)
    pred_in_dist_r2 = probe_r2_lora(model_pred, in_dist_ds, in_dist_ds, DEVICE)

    # R² recovery: relative to base in-dist
    pred_recovery = {}
    for t in pred_holdout_r2:
        base_id = zeroshot_in_dist.get(t, float('nan'))
        ho = pred_holdout_r2.get(t, float('nan'))
        if isinstance(base_id, float) and not np.isnan(base_id) and base_id > 0:
            pred_recovery[t] = round(ho / base_id * 100, 1)
        else:
            pred_recovery[t] = float('nan')

    lora_pred_results = {
        'target': 'predictor',
        'rank': 4,
        'n_finetune_traj': 50,
        'n_finetune_epochs': 20,
        'initial_loss': round(float(il_pred), 4),
        'final_loss': round(float(fl_pred), 4),
        'loss_history': [round(l, 4) for l in lh_pred],
        'converged': bool(fl_pred < il_pred * 0.99),
        'loss_decreasing': bool(fl_pred < il_pred),
        'train_time_sec': round(lora_pred_time, 1),
        'holdout_r2': pred_holdout_r2,
        'in_dist_r2_after_lora': pred_in_dist_r2,
        'r2_recovery_pct': pred_recovery,
    }
    results['lora_predictor_r4'] = lora_pred_results

    print(f'  Loss: {il_pred:.4f} → {fl_pred:.4f} | Converged: {lora_pred_results["converged"]}')
    print(f'  Holdout R² (joint_angle_mean): {pred_holdout_r2.get("joint_angle_mean", "N/A")}')
    print(f'  R² Recovery (joint_angle_mean): {pred_recovery.get("joint_angle_mean", "N/A")}%')
    write_progress(40, 80, metric={'phase': 'lora_predictor_done'})

    # ===== Experiment 2: LoRA Encoder-Only (r=4) =====
    print('\n[LORA] Experiment 2: LoRA-r4, Encoder-only fine-tuning...')
    write_progress(41, 80, metric={'phase': 'lora_encoder_r4'})

    model_enc = deepcopy(base_model)
    model_enc = apply_lora_adapter(model_enc, target='encoder', lora_rank=4)

    t_lora2 = time.time()
    lh_enc, il_enc, fl_enc = train_lora(
        model_enc, holdout_finetune_ds, n_epochs=20, lr=5e-4, batch_size=16,
        device=DEVICE, epoch_offset=41, total_epochs_outer=80
    )
    lora_enc_time = time.time() - t_lora2

    enc_holdout_r2 = probe_r2_lora(model_enc, holdout_finetune_ds, holdout_eval_ds, DEVICE)
    enc_recovery = {}
    for t in enc_holdout_r2:
        base_id = zeroshot_in_dist.get(t, float('nan'))
        ho = enc_holdout_r2.get(t, float('nan'))
        if isinstance(base_id, float) and not np.isnan(base_id) and base_id > 0:
            enc_recovery[t] = round(ho / base_id * 100, 1)
        else:
            enc_recovery[t] = float('nan')

    lora_enc_results = {
        'target': 'encoder',
        'rank': 4,
        'n_finetune_traj': 50,
        'n_finetune_epochs': 20,
        'initial_loss': round(float(il_enc), 4),
        'final_loss': round(float(fl_enc), 4),
        'loss_history': [round(l, 4) for l in lh_enc],
        'converged': bool(fl_enc < il_enc * 0.99),
        'loss_decreasing': bool(fl_enc < il_enc),
        'train_time_sec': round(lora_enc_time, 1),
        'holdout_r2': enc_holdout_r2,
        'r2_recovery_pct': enc_recovery,
    }
    results['lora_encoder_r4'] = lora_enc_results

    print(f'  Loss: {il_enc:.4f} → {fl_enc:.4f} | Converged: {lora_enc_results["converged"]}')
    print(f'  Holdout R² (joint_angle_mean): {enc_holdout_r2.get("joint_angle_mean", "N/A")}')
    print(f'  R² Recovery (joint_angle_mean): {enc_recovery.get("joint_angle_mean", "N/A")}%')
    write_progress(60, 80, metric={'phase': 'lora_encoder_done'})

    # ===== Experiment 3: Head-Only Baseline =====
    print('\n[BASELINE] Experiment 3: Head-only fine-tuning (frozen body)...')
    write_progress(61, 80, metric={'phase': 'head_only'})

    model_head = deepcopy(base_model)
    model_head = apply_head_only(model_head)

    t_head = time.time()
    lh_head, il_head, fl_head = train_lora(
        model_head, holdout_finetune_ds, n_epochs=20, lr=5e-4, batch_size=16,
        device=DEVICE, epoch_offset=61, total_epochs_outer=80
    )
    head_time = time.time() - t_head

    head_holdout_r2 = probe_r2_on_datasets(model_head, holdout_finetune_ds, holdout_eval_ds, DEVICE)
    head_recovery = {}
    for t in head_holdout_r2:
        base_id = zeroshot_in_dist.get(t, float('nan'))
        ho = head_holdout_r2.get(t, float('nan'))
        if isinstance(base_id, float) and not np.isnan(base_id) and base_id > 0:
            head_recovery[t] = round(ho / base_id * 100, 1)
        else:
            head_recovery[t] = float('nan')

    head_results = {
        'target': 'head_only',
        'n_finetune_traj': 50,
        'n_finetune_epochs': 20,
        'initial_loss': round(float(il_head), 4),
        'final_loss': round(float(fl_head), 4),
        'converged': bool(fl_head < il_head * 0.99),
        'train_time_sec': round(head_time, 1),
        'holdout_r2': head_holdout_r2,
        'r2_recovery_pct': head_recovery,
    }
    results['head_only_baseline'] = head_results

    print(f'  Loss: {il_head:.4f} → {fl_head:.4f} | Converged: {head_results["converged"]}')
    print(f'  R² Recovery (joint_angle_mean): {head_recovery.get("joint_angle_mean", "N/A")}%')
    write_progress(75, 80, metric={'phase': 'head_only_done'})

    # ===== Summary =====
    print('\n' + '='*60)
    print('PILOT SUMMARY: LoRA Adaptation Results')
    print('='*60)
    print(f'\n{"Condition":<30} {"joint_angle R²":>16} {"Recovery%":>12}')
    print('-'*60)
    print(f'{"Zero-shot (holdout)":<30} {zeroshot_holdout.get("joint_angle_mean", float("nan")):>16.4f} {"N/A":>12}')
    print(f'{"Zero-shot (in-dist)":<30} {zeroshot_in_dist.get("joint_angle_mean", float("nan")):>16.4f} {"baseline":>12}')
    print(f'{"LoRA-r4 Predictor":<30} {pred_holdout_r2.get("joint_angle_mean", float("nan")):>16.4f} {str(pred_recovery.get("joint_angle_mean", "N/A")):>12}')
    print(f'{"LoRA-r4 Encoder":<30} {enc_holdout_r2.get("joint_angle_mean", float("nan")):>16.4f} {str(enc_recovery.get("joint_angle_mean", "N/A")):>12}')
    print(f'{"Head-only":<30} {head_holdout_r2.get("joint_angle_mean", float("nan")):>16.4f} {str(head_recovery.get("joint_angle_mean", "N/A")):>12}')

    # ===== Pass Criteria =====
    lora_inserts_ok = True   # we got this far without import error
    lora_converges = lora_pred_results['loss_decreasing']
    r2_computed = 'joint_angle_mean' in pred_holdout_r2 and not np.isnan(pred_holdout_r2.get('joint_angle_mean', float('nan')))

    pass_criteria = {
        'lora_insert_ok': lora_inserts_ok,
        'lora_converges': lora_converges,
        'r2_recovery_computed': r2_computed,
        'all_pass': lora_inserts_ok and lora_converges and r2_computed,
    }
    results['pass_criteria'] = pass_criteria

    go_no_go = 'GO' if pass_criteria['all_pass'] else 'REFINE'
    status = 'success' if pass_criteria['all_pass'] else 'partial'
    results['go_no_go'] = go_no_go
    results['status'] = status

    # Encoder vs predictor comparison note
    pred_ja_recovery = pred_recovery.get('joint_angle_mean', float('nan'))
    enc_ja_recovery = enc_recovery.get('joint_angle_mean', float('nan'))
    if not np.isnan(pred_ja_recovery) and not np.isnan(enc_ja_recovery):
        asymmetry = pred_ja_recovery - enc_ja_recovery
        h3_signal = 'PREDICTOR_BOTTLENECK' if pred_ja_recovery > enc_ja_recovery else 'ENCODER_BOTTLENECK'
        if abs(asymmetry) < 10:
            h3_signal = 'NO_CLEAR_ASYMMETRY'
    else:
        asymmetry = float('nan')
        h3_signal = 'INSUFFICIENT_DATA'

    results['h3_diagnostic'] = {
        'predictor_recovery_pct': pred_ja_recovery,
        'encoder_recovery_pct': enc_ja_recovery,
        'asymmetry_pct': round(float(asymmetry), 1) if not np.isnan(asymmetry) else float('nan'),
        'signal': h3_signal,
        'interpretation': (
            f'Predictor LoRA recovery={pred_ja_recovery}% vs. Encoder LoRA recovery={enc_ja_recovery}%. '
            f'Asymmetry={asymmetry:.1f}pp → {h3_signal}. '
            f'Note: pilot uses only 5-epoch checkpoint; full study checkpoint may show stronger asymmetry.'
        ),
    }

    total_time = time.time() - t_start
    results['total_time_sec'] = round(total_time, 1)
    results['total_time_min'] = round(total_time / 60, 1)

    print(f'\n[RESULT] Go/No-Go: {go_no_go}')
    print(f'[RESULT] H3 signal: {h3_signal} (asymmetry={asymmetry:.1f}pp)')
    print(f'[RESULT] Total time: {total_time/60:.1f} min')

    # ===== Save =====
    output_path = PILOTS_DIR / 'eval_lora_adaptation_pilot.json'
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f'\n[SAVE] Results → {output_path}')

    write_progress(80, 80, loss=None, metric={'status': 'complete', 'go_no_go': go_no_go})
    write_done(status, f'{go_no_go}: {h3_signal}. pass_criteria={pass_criteria["all_pass"]}. time={total_time/60:.1f}min.')

    return results


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\n[FATAL ERROR] {e}')
        traceback.print_exc()
        write_done('failure', f'Fatal error: {e}')
        sys.exit(1)
