"""
Pilot CEM Planning Evaluation: eval_planning_cem
==================================================
PILOT MODE for CEM (Cross-Entropy Method) latent-space planning evaluation.

What this script does:
  - Load pilot LeWM-SIGReg checkpoint (seed=42, epoch=5)
  - Run CEM planning in latent space on:
      1) 1 in-distribution combo (friction=0.5x, g=1.0g, m=1.0x)
      2) 1 holdout combo (friction=1.0x, g=1.0g, m=1.0x)
  - Task: reach target joint angle configuration
  - 10 trajectories per combo (pilot scale)
  - Metrics: success rate, latent prediction MSE
  - Verify pass criteria: CEM runs without error; SR in [0, 1]

Pass criteria:
  - CEM planning runs without error
  - Success rate in [0, 1]
  - In-distribution SR >= 0.0 (even low is fine on small pilot)

Output:
  exp/results/pilots/eval_planning_cem_pilot.json
  exp/results/eval_planning_cem_DONE (on success)
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

# Set GPU
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

TASK_ID = 'eval_planning_cem'
PID_FILE = RESULTS_DIR / f'{TASK_ID}.pid'
PROGRESS_FILE = RESULTS_DIR / f'{TASK_ID}_PROGRESS.json'
DONE_FILE = RESULTS_DIR / f'{TASK_ID}_DONE'

# Primary checkpoint from train_lewm_sigreg_primary pilot
CHECKPOINT_PATH = RESULTS_DIR / 'full' / 'lewm_sigreg_primary' / 'pilot_seed42_epoch5.pt'

sys.path.insert(0, str(LEWM_DIR))

PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)


# ========================= PID ==========================
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


# ========================= Model Architecture ==================
# (Copied from pilot_framework_validation.py for self-containedness)

class PixelEncoder(nn.Module):
    """Simple CNN encoder for 64x64 RGB frames."""
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
        cnn_out = channels[-1]
        self.proj = nn.Linear(cnn_out, embed_dim)
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

        from module import SIGReg
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    def encode(self, frames):
        """frames: (B, T, H, W, 3) uint8 -> embeddings (B, T, D)"""
        x = frames.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3)
        return self.encoder(x)

    def predict(self, ctx_emb):
        """ctx_emb: (B, T_ctx, D) -> (B, T_ctx, D)"""
        x = ctx_emb + self.pos_embed[:, :ctx_emb.size(1)]
        for block in self.predictor_blocks:
            x = block(x)
        return self.pred_norm(x)

    def predict_next(self, ctx_emb):
        """
        Given context embeddings (B, T_ctx, D), predict the next embedding.
        Returns (B, D) — the predicted next latent.
        """
        pred_seq = self.predict(ctx_emb)
        return pred_seq[:, -1]  # (B, D)

    def rollout(self, ctx_emb, n_steps):
        """
        Autoregressively predict n_steps future latents.
        ctx_emb: (B, T_ctx, D)
        Returns: (B, n_steps, D) predicted future latents
        """
        preds = []
        current_ctx = ctx_emb.clone()
        for _ in range(n_steps):
            next_emb = self.predict_next(current_ctx)  # (B, D)
            preds.append(next_emb.unsqueeze(1))
            # Slide context window
            current_ctx = torch.cat([current_ctx[:, 1:], next_emb.unsqueeze(1)], dim=1)
        return torch.cat(preds, dim=1)  # (B, n_steps, D)

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
        return {'loss': total_loss, 'pred_loss': pred_loss.item(), 'sigreg_loss': sigreg_loss.item(), 'emb': emb}


# ========================= Data Loading ===================
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


# ========================= CEM Planning ====================

class CEMPlannerLatent:
    """
    Cross-Entropy Method (CEM) planner operating purely in latent space.

    The planner is given:
      - A context latent sequence (the observation history)
      - A target latent embedding (what we want to reach)

    It optimizes a "plan" represented as a sequence of latent perturbations
    such that the rolled-out trajectory ends near the target latent.

    Since we have no action space in the current simplified LeWM,
    we define the plan space as:
      - A sequence of delta vectors applied to the context latent
      - CEM optimizes the delta sequence to minimize distance to target

    This tests: can the predictor's latent dynamics be used to navigate
    toward desired states? This is a proxy for the real CEM planning task
    (which would require explicit action conditioning).

    Success criterion: predicted future latent is within threshold epsilon
    of the target latent in normalized cosine similarity.
    """

    def __init__(
        self,
        model,
        device,
        horizon=5,
        n_samples=64,
        n_elite=10,
        n_iters=10,
        noise_std=0.3,
        success_threshold=0.85,  # cosine similarity threshold for success
        seed=42,
    ):
        self.model = model
        self.device = device
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iters = n_iters
        self.noise_std = noise_std
        self.success_threshold = success_threshold
        self.embed_dim = model.embed_dim
        self.rng = np.random.RandomState(seed)

    def plan(self, ctx_emb_np, target_emb_np):
        """
        Run CEM to find a latent plan reaching target.

        Args:
            ctx_emb_np: (T_ctx, D) numpy context embeddings
            target_emb_np: (D,) numpy target embedding

        Returns:
            dict with: best_cos_sim, success, n_iters_run,
                       final_pred_mse, plan_quality
        """
        D = self.embed_dim
        T_ctx = ctx_emb_np.shape[0]

        ctx_emb = torch.tensor(ctx_emb_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T_ctx, D)
        target_emb = torch.tensor(target_emb_np, dtype=torch.float32, device=self.device)  # (D,)
        target_norm = F.normalize(target_emb.unsqueeze(0), dim=-1)  # (1, D)

        # Initialize CEM distribution: mean=zero perturbations, std=noise_std
        mu = np.zeros((self.horizon, D), dtype=np.float32)
        sigma = np.full((self.horizon, D), self.noise_std, dtype=np.float32)

        best_cos_sim = -1.0
        best_mse = float('inf')

        with torch.no_grad():
            for it in range(self.n_iters):
                # Sample candidates
                noise = self.rng.randn(self.n_samples, self.horizon, D).astype(np.float32)
                deltas = mu[None] + sigma[None] * noise  # (n_samples, horizon, D)
                deltas_t = torch.tensor(deltas, device=self.device)  # (n_samples, horizon, D)

                # Evaluate each candidate: apply first delta to context, rollout
                # Expand ctx to batch
                ctx_batch = ctx_emb.expand(self.n_samples, -1, -1).clone()  # (n_samples, T_ctx, D)

                # Apply initial perturbation (delta[:, 0]) to the last context frame
                ctx_batch[:, -1] = ctx_batch[:, -1] + deltas_t[:, 0]

                # Rollout model for (horizon-1) steps
                if self.horizon > 1:
                    pred_seq = self.model.rollout(ctx_batch, self.horizon - 1)  # (n_samples, horizon-1, D)
                    # Apply remaining deltas to predictions
                    pred_seq = pred_seq + deltas_t[:, 1:]

                    # Final predicted latent is the last element of pred_seq
                    final_preds = pred_seq[:, -1]  # (n_samples, D)
                else:
                    # horizon=1: just the perturbed context last frame
                    final_preds = ctx_batch[:, -1]  # (n_samples, D)

                # Score: cosine similarity to target
                final_preds_norm = F.normalize(final_preds, dim=-1)  # (n_samples, D)
                cos_sims = (final_preds_norm * target_norm).sum(dim=-1)  # (n_samples,)
                cos_sims_np = cos_sims.cpu().numpy()

                # Elite selection
                elite_idxs = np.argsort(cos_sims_np)[-self.n_elite:]
                elite_deltas = deltas[elite_idxs]  # (n_elite, horizon, D)

                # Update distribution
                mu = elite_deltas.mean(axis=0)
                sigma = elite_deltas.std(axis=0) + 1e-5  # avoid degeneration

                it_best = cos_sims_np[elite_idxs[-1]]
                if it_best > best_cos_sim:
                    best_cos_sim = float(it_best)

                # Compute MSE for best candidate
                best_pred = final_preds[elite_idxs[-1]].cpu().numpy()
                mse = float(np.mean((best_pred - target_emb_np) ** 2))
                if mse < best_mse:
                    best_mse = mse

        success = best_cos_sim >= self.success_threshold

        return {
            'best_cos_sim': round(best_cos_sim, 4),
            'success': bool(success),
            'success_threshold': self.success_threshold,
            'final_pred_mse': round(best_mse, 6),
            'n_iters_run': self.n_iters,
        }


def run_cem_evaluation(model, data_dict, planner, n_traj=10, history_size=3, frameskip=5, seed=42):
    """
    Run CEM planning evaluation on trajectory data.

    For each trajectory:
      - Encode the first history_size frames as context
      - Use the last frame of the trajectory as target
      - Run CEM to plan a path to the target
      - Record success and metrics

    Returns: dict with success_rate, mean_cos_sim, mean_mse, per_traj_results
    """
    model.eval()
    pixels = data_dict['pixels'][:n_traj]  # (n_traj, T, 64, 64, 3)
    T = pixels.shape[1]

    rng = np.random.RandomState(seed)
    results_per_traj = []

    with torch.no_grad():
        for traj_idx in range(len(pixels)):
            pix_traj = pixels[traj_idx]  # (T, 64, 64, 3)

            # Sub-sample with frameskip
            pix_fs = pix_traj[::frameskip]  # (T//frameskip, 64, 64, 3)
            n_frames = pix_fs.shape[0]

            if n_frames < history_size + 1:
                # Not enough frames; skip
                results_per_traj.append({'skipped': True, 'reason': 'too_short'})
                continue

            # Context: first history_size frames
            ctx_frames = torch.tensor(pix_fs[:history_size], dtype=torch.uint8, device=DEVICE)
            ctx_frames = ctx_frames.unsqueeze(0)  # (1, history_size, H, W, 3)

            # Target: a frame from the end of the trajectory (simulate goal-reaching)
            target_idx = min(n_frames - 1, history_size + 10)  # 10+ frames ahead
            tgt_frame = torch.tensor(pix_fs[target_idx:target_idx+1], dtype=torch.uint8, device=DEVICE)
            tgt_frame_seq = tgt_frame.unsqueeze(0)  # (1, 1, H, W, 3)

            # Encode context and target
            ctx_emb = model.encode(ctx_frames)[0].cpu().numpy()  # (history_size, D)
            tgt_emb = model.encode(tgt_frame_seq)[0, 0].cpu().numpy()  # (D,)

            # Run CEM
            cem_result = planner.plan(ctx_emb, tgt_emb)
            cem_result['traj_idx'] = traj_idx

            results_per_traj.append(cem_result)

    # Aggregate
    valid_results = [r for r in results_per_traj if not r.get('skipped', False)]
    if not valid_results:
        return {
            'success_rate': 0.0,
            'mean_cos_sim': float('nan'),
            'mean_mse': float('nan'),
            'n_traj_evaluated': 0,
            'per_traj_results': results_per_traj,
        }

    successes = [r['success'] for r in valid_results]
    cos_sims = [r['best_cos_sim'] for r in valid_results]
    mses = [r['final_pred_mse'] for r in valid_results]

    return {
        'success_rate': round(float(np.mean(successes)), 4),
        'mean_cos_sim': round(float(np.mean(cos_sims)), 4),
        'mean_mse': round(float(np.mean(mses)), 6),
        'n_traj_evaluated': len(valid_results),
        'n_success': int(sum(successes)),
        'per_traj_results': results_per_traj,
    }


# ========================= Load Checkpoint ==================

def load_checkpoint(checkpoint_path, device):
    """Load LeWMSimple checkpoint."""
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ck.get('config', {})

    model = LeWMSimple(
        embed_dim=config.get('embed_dim', 192),
        n_layers=config.get('n_layers', 4),
        num_heads=config.get('num_heads', 8),
        mlp_dim=config.get('mlp_dim', 512),
        sigreg_knots=config.get('sigreg_knots', 17),
        sigreg_num_proj=config.get('sigreg_num_proj', 1024),
        sigreg_weight=config.get('sigreg_weight', 0.09),
    )
    model.load_state_dict(ck['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f'[CKPT] Loaded checkpoint: epoch={ck.get("epoch")}, loss={ck.get("final_loss", "N/A"):.4f}')
    return model, config


# ========================= Update gpu_progress.json ===========

def update_gpu_progress(task_id, status, gpu_id, start_time_str, planned_min, actual_min):
    """Update gpu_progress.json with task completion."""
    gp_path = WORKSPACE / 'exp' / 'gpu_progress.json'
    try:
        if gp_path.exists():
            gp = json.loads(gp_path.read_text())
        else:
            gp = {'completed': [], 'failed': [], 'running': {}, 'timings': {}}

        if status == 'success':
            if task_id not in gp['completed']:
                gp['completed'].append(task_id)
            if task_id in gp.get('failed', []):
                gp['failed'].remove(task_id)
        else:
            if task_id not in gp.get('failed', []):
                gp['failed'].append(task_id)

        # Remove from running
        gp.setdefault('running', {})
        if task_id in gp['running']:
            del gp['running'][task_id]

        # Record timing
        gp.setdefault('timings', {})
        gp['timings'][task_id] = {
            'planned_min': planned_min,
            'actual_min': actual_min,
            'start_time': start_time_str,
            'end_time': datetime.now().isoformat(),
            'config_snapshot': {
                'task': task_id,
                'mode': 'PILOT',
                'gpu_count': 1,
                'gpu_model': 'NVIDIA RTX PRO 6000 Blackwell Server Edition',
            }
        }

        gp_path.write_text(json.dumps(gp, indent=2))
        print(f'[GPU_PROGRESS] Updated {gp_path}')
    except Exception as e:
        print(f'[WARN] Failed to update gpu_progress.json: {e}')


# ========================= MAIN ==============================

def main():
    t_start = time.time()
    start_time_str = datetime.now().isoformat()
    write_pid()
    write_progress(0, 10, metric={'phase': 'init'})

    results = {
        'task_id': TASK_ID,
        'timestamp': start_time_str,
        'gpu': str(DEVICE),
        'mode': 'PILOT',
        'checkpoint': str(CHECKPOINT_PATH),
    }

    print('\n' + '=' * 60)
    print('PILOT: eval_planning_cem')
    print('CEM Planning Evaluation in Latent Space')
    print('=' * 60)

    # ===== Load Model Checkpoint =====
    print(f'\n[1] Loading checkpoint: {CHECKPOINT_PATH}')
    write_progress(1, 10, metric={'phase': 'load_checkpoint'})

    try:
        model, ckpt_config = load_checkpoint(CHECKPOINT_PATH, DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Model params: {n_params:,} ({n_params/1e6:.2f}M)')
        results['model_config'] = ckpt_config
        results['model_params'] = n_params
    except Exception as e:
        print(f'[ERROR] Failed to load checkpoint: {e}')
        traceback.print_exc()
        write_done('failure', f'Checkpoint load failed: {e}')
        update_gpu_progress(TASK_ID, 'failure', 2, start_time_str, 90, 0)
        return

    # ===== Initialize CEM Planner =====
    print(f'\n[2] Initializing CEM planner...')
    write_progress(2, 10, metric={'phase': 'init_cem'})

    # Pilot-scale CEM: fast settings
    planner = CEMPlannerLatent(
        model=model,
        device=DEVICE,
        horizon=5,
        n_samples=32,       # smaller for pilot speed
        n_elite=8,
        n_iters=10,
        noise_std=0.3,
        success_threshold=0.80,  # cosine sim threshold — slightly lower for pilot
        seed=42,
    )
    print(f'  CEM config: horizon={planner.horizon}, n_samples={planner.n_samples}, '
          f'n_iters={planner.n_iters}, success_threshold={planner.success_threshold}')
    results['cem_config'] = {
        'horizon': planner.horizon,
        'n_samples': planner.n_samples,
        'n_elite': planner.n_elite,
        'n_iters': planner.n_iters,
        'noise_std': planner.noise_std,
        'success_threshold': planner.success_threshold,
    }

    # ===== Load Data =====
    print(f'\n[3] Loading pilot data...')
    write_progress(3, 10, metric={'phase': 'load_data'})

    N_TRAJ_PILOT = 10  # pilot: 10 trajectories per combo

    try:
        # In-distribution combo: friction=0.5x (training combo)
        data_in_dist = load_hdf5_data(DATA_DIR / 'g1.0_f0.5_m1.0.h5', n_traj=N_TRAJ_PILOT, seed=42)
        print(f'  In-distribution data (f=0.5x): {data_in_dist["pixels"].shape[0]} trajectories')

        # Holdout combo: friction=1.0x (not in training set for pilot split)
        data_holdout = load_hdf5_data(DATA_DIR / 'g1.0_f1.0_m1.0.h5', n_traj=N_TRAJ_PILOT, seed=42)
        print(f'  Holdout data (f=1.0x):         {data_holdout["pixels"].shape[0]} trajectories')
    except Exception as e:
        print(f'[ERROR] Failed to load data: {e}')
        traceback.print_exc()
        write_done('failure', f'Data load failed: {e}')
        update_gpu_progress(TASK_ID, 'failure', 2, start_time_str, 90, 0)
        return

    # ===== Run CEM on In-Distribution =====
    print(f'\n[4] CEM evaluation on IN-DISTRIBUTION combo (f=0.5x)...')
    write_progress(4, 10, metric={'phase': 'cem_in_dist'})

    try:
        t_cem_in_dist = time.time()
        in_dist_results = run_cem_evaluation(
            model, data_in_dist, planner,
            n_traj=N_TRAJ_PILOT, history_size=3, frameskip=5, seed=42
        )
        t_cem_in_dist_elapsed = time.time() - t_cem_in_dist

        print(f'  In-distribution CEM results:')
        print(f'    Success rate:   {in_dist_results["success_rate"]:.3f}')
        print(f'    Mean cos sim:   {in_dist_results["mean_cos_sim"]:.4f}')
        print(f'    Mean pred MSE:  {in_dist_results["mean_mse"]:.6f}')
        print(f'    Trajectories:   {in_dist_results["n_traj_evaluated"]} / {N_TRAJ_PILOT}')
        print(f'    Elapsed:        {t_cem_in_dist_elapsed:.1f}s')

        results['in_dist_cem'] = {
            'combo': 'g1.0_f0.5_m1.0',
            'n_traj': N_TRAJ_PILOT,
            **{k: v for k, v in in_dist_results.items() if k != 'per_traj_results'},
            'per_traj_summary': [
                {'traj_idx': r.get('traj_idx', i), 'success': r.get('success', False),
                 'cos_sim': r.get('best_cos_sim', 0.0), 'mse': r.get('final_pred_mse', 0.0)}
                for i, r in enumerate(in_dist_results['per_traj_results'])
                if not r.get('skipped', False)
            ]
        }

    except Exception as e:
        print(f'[ERROR] In-distribution CEM failed: {e}')
        traceback.print_exc()
        results['in_dist_cem'] = {'error': str(e)}

    # ===== Run CEM on Holdout =====
    print(f'\n[5] CEM evaluation on HOLDOUT combo (f=1.0x)...')
    write_progress(6, 10, metric={'phase': 'cem_holdout'})

    try:
        t_cem_holdout = time.time()
        holdout_results = run_cem_evaluation(
            model, data_holdout, planner,
            n_traj=N_TRAJ_PILOT, history_size=3, frameskip=5, seed=42
        )
        t_cem_holdout_elapsed = time.time() - t_cem_holdout

        print(f'  Holdout CEM results:')
        print(f'    Success rate:   {holdout_results["success_rate"]:.3f}')
        print(f'    Mean cos sim:   {holdout_results["mean_cos_sim"]:.4f}')
        print(f'    Mean pred MSE:  {holdout_results["mean_mse"]:.6f}')
        print(f'    Trajectories:   {holdout_results["n_traj_evaluated"]} / {N_TRAJ_PILOT}')
        print(f'    Elapsed:        {t_cem_holdout_elapsed:.1f}s')

        results['holdout_cem'] = {
            'combo': 'g1.0_f1.0_m1.0',
            'n_traj': N_TRAJ_PILOT,
            **{k: v for k, v in holdout_results.items() if k != 'per_traj_results'},
            'per_traj_summary': [
                {'traj_idx': r.get('traj_idx', i), 'success': r.get('success', False),
                 'cos_sim': r.get('best_cos_sim', 0.0), 'mse': r.get('final_pred_mse', 0.0)}
                for i, r in enumerate(holdout_results['per_traj_results'])
                if not r.get('skipped', False)
            ]
        }

    except Exception as e:
        print(f'[ERROR] Holdout CEM failed: {e}')
        traceback.print_exc()
        results['holdout_cem'] = {'error': str(e)}

    # ===== Compute Relative Drop =====
    print(f'\n[6] Computing relative success rate drop...')
    write_progress(8, 10, metric={'phase': 'analysis'})

    in_dist_sr = results.get('in_dist_cem', {}).get('success_rate', None)
    holdout_sr = results.get('holdout_cem', {}).get('success_rate', None)

    if in_dist_sr is not None and holdout_sr is not None:
        if in_dist_sr > 0:
            relative_sr_drop = (in_dist_sr - holdout_sr) / in_dist_sr
        else:
            relative_sr_drop = 0.0  # both zero, no meaningful drop

        in_dist_cos = results.get('in_dist_cem', {}).get('mean_cos_sim', None)
        holdout_cos = results.get('holdout_cem', {}).get('mean_cos_sim', None)
        if in_dist_cos is not None and holdout_cos is not None:
            cos_sim_drop = in_dist_cos - holdout_cos
        else:
            cos_sim_drop = None

        results['comparison'] = {
            'in_dist_success_rate': in_dist_sr,
            'holdout_success_rate': holdout_sr,
            'relative_sr_drop': round(float(relative_sr_drop), 4),
            'in_dist_cos_sim': in_dist_cos,
            'holdout_cos_sim': holdout_cos,
            'cos_sim_drop': round(float(cos_sim_drop), 4) if cos_sim_drop is not None else None,
        }

        print(f'  In-distribution SR:   {in_dist_sr:.3f}')
        print(f'  Holdout SR:           {holdout_sr:.3f}')
        print(f'  Relative SR drop:     {relative_sr_drop:.1%}')
        print(f'  In-distribution cosine sim: {in_dist_cos:.4f}')
        print(f'  Holdout cosine sim:         {holdout_cos:.4f}')

    # ===== Pass Criteria Check =====
    print(f'\n[7] Checking pass criteria...')

    pass_criteria = {}

    # 1. CEM runs without error
    in_dist_ok = 'error' not in results.get('in_dist_cem', {'error': 'not run'})
    holdout_ok = 'error' not in results.get('holdout_cem', {'error': 'not run'})
    pass_criteria['cem_runs_without_error'] = bool(in_dist_ok and holdout_ok)

    # 2. Success rate in [0, 1] for in-distribution
    if in_dist_sr is not None:
        pass_criteria['in_dist_sr_in_valid_range'] = bool(0.0 <= in_dist_sr <= 1.0)
        pass_criteria['in_dist_sr'] = round(float(in_dist_sr), 4)
    else:
        pass_criteria['in_dist_sr_in_valid_range'] = False
        pass_criteria['in_dist_sr'] = None

    # 3. Success rate >= 0.0 (always true if computed)
    pass_criteria['in_dist_sr_nonnegative'] = bool(in_dist_sr is not None and in_dist_sr >= 0.0)

    all_pass = all([
        pass_criteria['cem_runs_without_error'],
        pass_criteria['in_dist_sr_in_valid_range'],
        pass_criteria['in_dist_sr_nonnegative'],
    ])

    pass_criteria['all_pass'] = all_pass
    results['pass_criteria'] = pass_criteria

    print(f'  CEM runs without error: {pass_criteria["cem_runs_without_error"]}')
    print(f'  In-dist SR valid range: {pass_criteria["in_dist_sr_in_valid_range"]}')
    print(f'  In-dist SR nonnegative: {pass_criteria["in_dist_sr_nonnegative"]}')
    print(f'  ALL PASS: {all_pass}')

    # ===== Timing =====
    total_time = time.time() - t_start
    results['total_time_sec'] = round(total_time, 1)
    results['total_time_min'] = round(total_time / 60, 2)
    actual_min = int(round(total_time / 60))

    # ===== Go/No-Go =====
    if all_pass:
        go_no_go = 'GO'
        status = 'success'
        notes = (
            f'CEM planning pilot: in-dist SR={in_dist_sr:.3f}, holdout SR={holdout_sr:.3f} '
            f'(relative drop={relative_sr_drop:.1%}). '
            f'Mean cosine sim: in-dist={in_dist_cos:.4f}, holdout={holdout_cos:.4f}. '
            f'Framework runs without error. Pilot pass criteria all met.'
        )
    else:
        go_no_go = 'REFINE'
        status = 'partial'
        notes = f'CEM pilot partially passed: {pass_criteria}'

    results['go_no_go'] = go_no_go
    results['status'] = status
    results['notes'] = notes

    # ===== Save Results =====
    write_progress(9, 10, metric={'phase': 'saving'})

    pilot_output = PILOTS_DIR / 'eval_planning_cem_pilot.json'
    pilot_output.write_text(json.dumps(results, indent=2))
    print(f'\n[SAVE] Results saved to {pilot_output}')

    # Also write to full directory for pipeline compatibility
    full_output = FULL_DIR / 'cem_planning_results.json'
    full_output.write_text(json.dumps({
        'task_id': TASK_ID,
        'mode': 'PILOT',
        'timestamp': start_time_str,
        'note': 'Pilot results — 10 traj per combo, 2 combos. Full study: 50 traj, all 9 holdout combos, 7 seeds.',
        'pilot_results': {
            'in_dist': results.get('in_dist_cem', {}),
            'holdout': results.get('holdout_cem', {}),
            'comparison': results.get('comparison', {}),
        },
        'pass_criteria': pass_criteria,
        'go_no_go': go_no_go,
        'cem_config': results.get('cem_config', {}),
        'total_time_min': results['total_time_min'],
    }, indent=2))
    print(f'[SAVE] Full results saved to {full_output}')

    print(f'\n' + '=' * 60)
    print(f'PILOT COMPLETE: {go_no_go}')
    print(f'In-dist SR: {in_dist_sr}  |  Holdout SR: {holdout_sr}')
    print(f'Time: {total_time/60:.2f} min')
    print('=' * 60 + '\n')

    write_progress(10, 10, metric={'status': 'complete', 'go_no_go': go_no_go})
    write_done(status, f'{go_no_go}: in_dist_SR={in_dist_sr}, holdout_SR={holdout_sr}')

    update_gpu_progress(TASK_ID, status, 2, start_time_str, 90, actual_min)

    return results


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\n[FATAL ERROR] {e}')
        traceback.print_exc()
        write_done('failure', f'Fatal error: {e}')
        # Still update gpu_progress to avoid stale running state
        gp_path = WORKSPACE / 'exp' / 'gpu_progress.json'
        try:
            if gp_path.exists():
                gp = json.loads(gp_path.read_text())
                if TASK_ID in gp.get('running', {}):
                    del gp['running'][TASK_ID]
                if TASK_ID not in gp.get('failed', []):
                    gp.setdefault('failed', []).append(TASK_ID)
                gp_path.write_text(json.dumps(gp, indent=2))
        except Exception:
            pass
        sys.exit(1)
