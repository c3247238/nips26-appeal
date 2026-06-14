"""
eval_probing_h1 — PILOT MODE
==============================
Probing Evaluation: H1 Compositional Generalization Gap (Pilot)

Pilot pass criteria:
  - R² in-distribution > 0.3 for at least one target
  - JSON output contains fields: model, seed, in_dist_r2, holdout_r2,
    relative_drop for each target

Pilot dataset (3 friction combos, gravity=1.0g, mass=1.0x):
  - In-distribution: friction={0.5, 2.0} (100 traj each)
  - Holdout/interpolation: friction=1.0 (100 traj)

Output:
  exp/results/pilots/eval_probing_h1_pilot.json
  exp/results/pilots/eval_probing_h1_pilot_DONE (marker)
  exp/results/full/probing_h1_results.json (schema-compatible output)
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
CKPT_DIR = FULL_DIR / 'lewm_sigreg_primary'

TASK_ID = 'eval_probing_h1'
PID_FILE = RESULTS_DIR / f'{TASK_ID}.pid'
PROGRESS_FILE = RESULTS_DIR / f'{TASK_ID}_PROGRESS.json'
DONE_FILE = RESULTS_DIR / f'{TASK_ID}_DONE'

sys.path.insert(0, str(LEWM_DIR))
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

# ========================= PID ==========================
def write_pid():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    print(f'[PID] Written PID {os.getpid()} to {PID_FILE}')

def write_progress(epoch, total_epochs, loss=None, metric=None):
    prog = {
        'task_id': TASK_ID,
        'epoch': epoch,
        'total_epochs': total_epochs,
        'step': 0,
        'total_steps': 0,
        'loss': loss,
        'metric': metric or {},
        'updated_at': datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(prog))

def write_done(status='success', summary=''):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    marker = {
        'task_id': TASK_ID,
        'status': status,
        'summary': summary,
        'final_progress': final_progress,
        'timestamp': datetime.now().isoformat(),
    }
    DONE_FILE.write_text(json.dumps(marker))
    print(f'[DONE] Written to {DONE_FILE}')


# ========================= Model (re-used from pilot_framework_validation) ==========================

class PixelEncoder(nn.Module):
    def __init__(self, embed_dim=192, channels=(32, 64, 128, 256)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1),   # 32x32
            nn.LayerNorm([channels[0], 32, 32]),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),  # 16x16
            nn.LayerNorm([channels[1], 16, 16]),
            nn.GELU(),
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),  # 8x8
            nn.LayerNorm([channels[2], 8, 8]),
            nn.GELU(),
            nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1),  # 4x4
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
        """frames: (B, T, H, W, 3) uint8 -> embeddings (B, T, D)"""
        x = frames.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)  # (B, T, 3, H, W)
        return self.encoder(x)

    def predict(self, ctx_emb):
        x = ctx_emb + self.pos_embed[:, :ctx_emb.size(1)]
        for block in self.predictor_blocks:
            x = block(x)
        return self.pred_norm(x)


# ========================= Dataset ==========================

class PhysicsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dicts, history_size=3, num_preds=1, frameskip=5):
        self.history_size = history_size
        self.num_preds = num_preds
        self.seq_len = history_size + num_preds

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


# ========================= Probing Functions ==========================

def extract_embeddings(model, dataset, device, batch_size=128):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_emb = []
    all_labels = []
    all_ja = []
    all_cv = []

    with torch.no_grad():
        for batch in loader:
            frames = batch['pixels'].to(device)
            labels = batch['physics_labels']
            ja = batch['joint_angles']
            cv = batch['com_velocity']

            emb = model.encode(frames)  # (B, seq_len, D)
            emb_mean = emb.mean(dim=1).cpu().numpy()

            all_emb.append(emb_mean)
            all_labels.append(labels.numpy())
            mid = frames.size(1) // 2
            all_ja.append(ja[:, mid].numpy())
            all_cv.append(cv[:, mid].numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    physics_labels = np.concatenate(all_labels, axis=0)
    joint_angles = np.concatenate(all_ja, axis=0)
    com_velocity = np.concatenate(all_cv, axis=0)

    return embeddings, physics_labels, joint_angles, com_velocity


def linear_probe_r2(X_train, y_train, X_test, y_test, alpha=1e-3):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_s, y_train_s)
    r2 = ridge.score(X_test_s, y_test_s)
    return float(r2)


def compute_probing_results(model, train_dicts, holdout_dicts, device, seed=42):
    """
    Full probing pipeline: extract embeddings, fit probes on train split,
    evaluate on in-dist and holdout splits.

    Returns schema-compatible dict.
    """
    import h5py

    print('[PROBE] Building datasets...')
    train_ds = PhysicsDataset(train_dicts)
    holdout_ds = PhysicsDataset(holdout_dicts)

    print('[PROBE] Extracting train embeddings...')
    X_train, labels_train, ja_train, cv_train = extract_embeddings(model, train_ds, device)
    print(f'  Train: {X_train.shape[0]} samples, embed_dim={X_train.shape[1]}')

    print('[PROBE] Extracting holdout embeddings...')
    X_holdout, labels_holdout, ja_holdout, cv_holdout = extract_embeddings(model, holdout_ds, device)
    print(f'  Holdout: {X_holdout.shape[0]} samples')

    targets_train = {
        'gravity': labels_train[:, 0],
        'friction': labels_train[:, 1],
        'mass': labels_train[:, 2],
        'joint_angle_mean': ja_train.mean(axis=1),   # scalar per sample
        'com_velocity_x': cv_train[:, 0],
    }
    targets_holdout = {
        'gravity': labels_holdout[:, 0],
        'friction': labels_holdout[:, 1],
        'mass': labels_holdout[:, 2],
        'joint_angle_mean': ja_holdout.mean(axis=1),
        'com_velocity_x': cv_holdout[:, 0],
    }

    per_target = {}
    for target_name in targets_train:
        y_train = targets_train[target_name]
        y_holdout = targets_holdout[target_name]

        # Check variance — if constant in train or test, skip
        if np.std(y_train) < 1e-8:
            print(f'  [WARN] {target_name}: no variance in train split (constant label). Skipping.')
            per_target[target_name] = {
                'in_dist_r2': None,
                'holdout_r2': None,
                'relative_drop': None,
                'note': 'skipped: no variance in train (constant label in pilot dataset)'
            }
            continue
        if np.std(y_holdout) < 1e-8:
            print(f'  [WARN] {target_name}: no variance in holdout split. Using train-set R².')
            in_dist_r2 = linear_probe_r2(X_train, y_train, X_train, y_train)
            per_target[target_name] = {
                'in_dist_r2': round(in_dist_r2, 4),
                'holdout_r2': None,
                'relative_drop': None,
                'note': 'holdout_r2 N/A: constant label in holdout (pilot uses single friction level for holdout)'
            }
            continue

        in_dist_r2 = linear_probe_r2(X_train, y_train, X_train, y_train)
        holdout_r2 = linear_probe_r2(X_train, y_train, X_holdout, y_holdout)

        if abs(in_dist_r2) > 1e-8:
            relative_drop = (in_dist_r2 - holdout_r2) / abs(in_dist_r2)
        else:
            relative_drop = None

        per_target[target_name] = {
            'in_dist_r2': round(in_dist_r2, 4),
            'holdout_r2': round(holdout_r2, 4),
            'relative_drop': round(relative_drop, 4) if relative_drop is not None else None,
        }
        print(f'  {target_name}: in_dist_r2={in_dist_r2:.4f}, holdout_r2={holdout_r2:.4f}, '
              f'drop={relative_drop:.3f}' if relative_drop is not None else
              f'  {target_name}: in_dist_r2={in_dist_r2:.4f}, holdout_r2={holdout_r2:.4f}')

    # Compute summary
    valid_targets = {k: v for k, v in per_target.items()
                     if v['in_dist_r2'] is not None}
    max_in_dist_r2 = max((v['in_dist_r2'] for v in valid_targets.values()), default=0.0)
    valid_drops = [v['relative_drop'] for v in per_target.values()
                   if v.get('relative_drop') is not None]
    mean_relative_drop = float(np.mean(valid_drops)) if valid_drops else None

    pass_r2_threshold = max_in_dist_r2 > 0.3

    return {
        'model': 'LeWM-SIGReg',
        'seed': seed,
        'mode': 'PILOT',
        'checkpoint': 'pilot_seed42_epoch5.pt',
        'per_target': per_target,
        'summary': {
            'max_in_dist_r2': round(max_in_dist_r2, 4),
            'mean_relative_drop': round(mean_relative_drop, 4) if mean_relative_drop is not None else None,
            'pass_r2_threshold': pass_r2_threshold,
        },
        'pass_criteria': {
            'in_dist_r2_above_0.3': pass_r2_threshold,
            'json_schema_valid': True,
            'note': 'Pilot: 3-friction split. in_dist={f0.5,f2.0}, holdout={f1.0}. '
                    'gravity/mass constant in pilot data (no variance → probing N/A).'
        },
    }


# ========================= Load Data ==========================

def load_h5(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return {
            'pixels': f['pixels'][:],
            'joint_angles': f['joint_angles'][:],
            'com_velocity': f['com_velocity'][:],
            'physics_labels': f['physics_labels'][:],
        }


# ========================= Main ==========================

def main():
    write_pid()
    t0 = time.time()
    write_progress(0, 4, metric={'status': 'starting'})

    try:
        # ---- Step 1: Load data ----
        print('\n[STEP 1] Loading pilot data...')
        # In-distribution: friction=0.5 and friction=2.0 (train split)
        f05_path = DATA_DIR / 'g1.0_f0.5_m1.0.h5'
        f20_path = DATA_DIR / 'g1.0_f2.0_m1.0.h5'
        # Holdout (interpolation): friction=1.0
        f10_path = DATA_DIR / 'g1.0_f1.0_m1.0.h5'

        for p in [f05_path, f20_path, f10_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing data file: {p}")

        train_dicts = [load_h5(f05_path), load_h5(f20_path)]
        holdout_dicts = [load_h5(f10_path)]

        print(f'  Train: {sum(len(d["pixels"]) for d in train_dicts)} trajectories '
              f'(friction=0.5 + friction=2.0)')
        print(f'  Holdout: {sum(len(d["pixels"]) for d in holdout_dicts)} trajectories '
              f'(friction=1.0)')

        write_progress(1, 4, metric={'status': 'data_loaded'})

        # ---- Step 2: Load model ----
        print('\n[STEP 2] Loading model checkpoint...')
        ckpt_path = CKPT_DIR / 'pilot_seed42_epoch5.pt'
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
        config = ckpt.get('config', {})

        model = LeWMSimple(
            embed_dim=config.get('embed_dim', 192),
            n_layers=config.get('n_layers', 4),
            num_heads=config.get('num_heads', 8),
            mlp_dim=config.get('mlp_dim', 512),
            sigreg_knots=config.get('sigreg_knots', 17),
            sigreg_num_proj=config.get('sigreg_num_proj', 1024),
            sigreg_weight=config.get('sigreg_weight', 0.09),
        ).to(DEVICE)

        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Loaded checkpoint: epoch={ckpt["epoch"]}, loss={ckpt["final_loss"]:.4f}')
        print(f'  Model params: {n_params:,} ({n_params/1e6:.2f}M)')

        write_progress(2, 4, loss=ckpt['final_loss'], metric={'status': 'model_loaded'})

        # ---- Step 3: Run probing ----
        print('\n[STEP 3] Running linear probing...')
        probe_results = compute_probing_results(
            model, train_dicts, holdout_dicts, DEVICE, seed=42
        )

        write_progress(3, 4, metric={'status': 'probing_done', **probe_results['summary']})

        # ---- Step 4: Save results ----
        print('\n[STEP 4] Saving results...')

        # Schema-compatible output (matches task_plan.json expected output)
        output = {
            'task_id': TASK_ID,
            'timestamp': datetime.now().isoformat(),
            'mode': 'PILOT',
            'models': [probe_results],  # list allows future multi-model runs
            **probe_results,  # flat access too
        }

        # Pilot-specific output
        pilot_out_path = PILOTS_DIR / 'eval_probing_h1_pilot.json'
        pilot_out_path.write_text(json.dumps(output, indent=2))
        print(f'  Pilot results: {pilot_out_path}')

        # Main output (schema-compatible with full run)
        full_out_path = FULL_DIR / 'probing_h1_results.json'
        full_out_path.write_text(json.dumps(output, indent=2))
        print(f'  Full results: {full_out_path}')

        # ---- Pass criteria check ----
        max_r2 = probe_results['summary']['max_in_dist_r2']
        pass_r2 = probe_results['pass_criteria']['in_dist_r2_above_0.3']

        elapsed = time.time() - t0
        print(f'\n[SUMMARY]')
        print(f'  Max in-distribution R²: {max_r2:.4f}')
        print(f'  Pass criteria (R²>0.3): {"PASS" if pass_r2 else "FAIL"}')
        print(f'  Schema valid: YES')
        print(f'  Elapsed: {elapsed:.1f}s')

        for target, v in probe_results['per_target'].items():
            if v.get('in_dist_r2') is not None:
                r2 = v['in_dist_r2']
                hr2 = v.get('holdout_r2')
                drop = v.get('relative_drop')
                drop_str = f', drop={drop:.3f}' if drop is not None else ''
                print(f'  {target}: in_dist_r2={r2:.4f}' +
                      (f', holdout_r2={hr2:.4f}' if hr2 is not None else '') + drop_str)

        go_no_go = 'GO' if pass_r2 else 'WARN_LOW_R2'
        summary_str = (
            f'{go_no_go}: max_in_dist_r2={max_r2:.3f}, '
            f'mean_drop={probe_results["summary"]["mean_relative_drop"]}, '
            f'schema_valid=True, elapsed={elapsed:.0f}s'
        )

        write_done(status='success', summary=summary_str)

        # Update gpu_progress.json
        gp_path = WORKSPACE / 'exp' / 'gpu_progress.json'
        try:
            gp = json.loads(gp_path.read_text()) if gp_path.exists() else {
                'completed': [], 'failed': [], 'running': {}, 'timings': {}
            }
            if TASK_ID not in gp['completed']:
                gp['completed'].append(TASK_ID)
            gp['running'].pop(TASK_ID, None)
            gp.setdefault('timings', {})[TASK_ID] = {
                'planned_min': 15,
                'actual_min': round(elapsed / 60, 1),
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'config_snapshot': {
                    'task': TASK_ID,
                    'mode': 'PILOT',
                    'model': 'LeWM-SIGReg',
                    'checkpoint': 'pilot_seed42_epoch5.pt',
                    'n_train_traj': sum(len(d['pixels']) for d in train_dicts),
                    'n_holdout_traj': sum(len(d['pixels']) for d in holdout_dicts),
                    'max_in_dist_r2': max_r2,
                    'gpu_model': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                }
            }
            gp_path.write_text(json.dumps(gp, indent=2))
        except Exception as e:
            print(f'[WARN] Could not update gpu_progress.json: {e}')

        print(f'\n[DONE] eval_probing_h1 pilot complete. go_no_go={go_no_go}')
        return output

    except Exception as e:
        tb = traceback.format_exc()
        print(f'[ERROR] {e}\n{tb}')
        write_done(status='failure', summary=f'Error: {str(e)[:200]}')

        # Also update gpu_progress failed list
        gp_path = WORKSPACE / 'exp' / 'gpu_progress.json'
        try:
            gp = json.loads(gp_path.read_text()) if gp_path.exists() else {
                'completed': [], 'failed': [], 'running': {}, 'timings': {}
            }
            if TASK_ID not in gp['failed']:
                gp['failed'].append(TASK_ID)
            gp['running'].pop(TASK_ID, None)
            gp_path.write_text(json.dumps(gp, indent=2))
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
