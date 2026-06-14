"""
Task C1: Amortization Gap Pilot — OMP vs. Feedforward on 100 Tokens

Implement and validate the Orthogonal Matching Pursuit (OMP) encoding experiment on GPT-2 L6.
Load gpt2-small-res-jb L6 SAE; extract frozen decoder matrix W_dec.
For 100 tokens from OpenWebText:
  (A) Feedforward encoding: z_A = ReLU(W_enc @ x + b_enc)
  (B) OMP encoding: z_B = OrthogonalMatchingPursuit(W_dec, x, n_nonzero_coefs=K)
  (C) 2-pass encoding: z_A_1 = z_A; r = x - W_dec @ z_A_1; z_A_2 = ReLU(W_enc @ r + b_enc); z_C = z_A_1 + 0.5 * z_A_2
Validate: OMP reconstruction error < feedforward; OMP sparsity ~ K
Compute absorption on 3 letters {a, e, s} using per-token active-set comparison with exact Chanin labels.

Mode: PILOT
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
import torch.nn.functional as F
from sklearn.linear_model import OrthogonalMatchingPursuit

TASK_ID = "task_C1_omp_pilot"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR_PILOTS = WORKSPACE / "exp/results/pilots"
RESULTS_DIR_FULL = WORKSPACE / "exp/results/full"
RESULTS_DIR_PILOTS.mkdir(parents=True, exist_ok=True)
RESULTS_DIR_FULL.mkdir(parents=True, exist_ok=True)

LABEL_FILE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_001/exp/results/r4/r4a_direct_labels.json")
SEED = 42
N_TOKENS = 100
TARGET_LETTERS = ['a', 'e', 's']

np.random.seed(SEED)
torch.manual_seed(SEED)

start_time = time.time()

# Write PID file
pid_file = RESULTS_DIR_FULL / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def write_progress(step, total_steps, metric=None):
    progress_file = RESULTS_DIR_FULL / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
        "step": step,
        "total_steps": total_steps,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR_FULL / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR_FULL / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = RESULTS_DIR_FULL / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def load_absorbed_feature_ids():
    """Load the 18 absorbed feature IDs for GPT-2 L6 from exact Chanin IG labels."""
    with open(LABEL_FILE) as f:
        d = json.load(f)
    psr = d['per_sae_results']
    # L6 is the first entry
    l6_entry = psr[0]
    assert l6_entry['config']['hook_name'] == 'blocks.6.hook_resid_pre', f"Expected L6, got {l6_entry['config']}"
    absorbed_ids = set(l6_entry['absorbed_latent_ids'])
    print(f"Loaded {len(absorbed_ids)} absorbed feature IDs for GPT-2 L6: {sorted(absorbed_ids)[:10]}...")
    return absorbed_ids, l6_entry['n_absorbed'], l6_entry['n_non_absorbed']


def feedforward_encode(x, W_enc, b_enc):
    """Standard feedforward SAE encoder: z = ReLU(x @ W_enc + b_enc)
    SAELens convention: W_enc: [d_in, d_sae], so x @ W_enc gives [d_sae]
    """
    # x: [768], W_enc: [768, d_sae], b_enc: [d_sae]
    return F.relu(x @ W_enc + b_enc)


def omp_encode(x_np, W_dec_np, K):
    """OMP encoding: find K-sparse code z such that z @ W_dec ≈ x.

    SAELens convention: W_dec: [d_sae, d_model]
    Reconstruction: x_hat = z @ W_dec (where z is [d_sae])
    This is equivalent to: x ≈ W_dec.T @ z (transposing)

    sklearn OMP: solves min ||y - D @ coef||^2 with ||coef||_0 <= K
    where D is [n_samples_features, n_atoms] = [d_model, d_sae]
    and y is [d_model]

    So dictionary D = W_dec.T [d_model, d_sae], and we solve for coef = z [d_sae]
    """
    # W_dec: [d_sae, d_model]
    D = W_dec_np.T  # [d_model, d_sae]
    # Normalize columns (atoms) for OMP
    norms = np.linalg.norm(D, axis=0, keepdims=True)  # [1, d_sae]
    norms = np.clip(norms, 1e-8, None)
    D_norm = D / norms  # [d_model, d_sae]

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=K, fit_intercept=False)
    omp.fit(D_norm, x_np)
    z_norm = omp.coef_  # [d_sae]
    # Unnormalize: z[j] = z_norm[j] / norms[0,j]
    z = z_norm / norms[0]
    z = np.clip(z, 0, None)  # Ensure non-negative (consistent with ReLU SAE)
    return z


def twopass_encode(x, W_enc, b_enc, W_dec):
    """2-pass residual encoding (vectorized over batch dimension):
    SAELens convention: W_enc [d_in, d_sae], W_dec [d_sae, d_in]
    z1 = ReLU(x @ W_enc + b_enc)      # [batch, d_sae]
    x_hat1 = z1 @ W_dec               # [batch, d_in] reconstruction
    r = x - x_hat1                    # [batch, d_in] residual
    z2 = ReLU(r @ W_enc + b_enc)      # [batch, d_sae]
    z_combined = z1 + 0.5 * z2        # [batch, d_sae]
    """
    z1 = F.relu(x @ W_enc + b_enc)    # [batch, d_sae]
    x_hat1 = z1 @ W_dec               # [batch, d_in]
    r = x - x_hat1                    # [batch, d_in]
    z2 = F.relu(r @ W_enc + b_enc)    # [batch, d_sae]
    z_combined = z1 + 0.5 * z2        # [batch, d_sae]
    return z_combined


def get_active_latents(z, threshold=0.01):
    """Get set of active latent indices above threshold."""
    if isinstance(z, np.ndarray):
        return set(np.where(z > threshold)[0].tolist())
    else:
        return set(torch.where(z > threshold)[0].cpu().numpy().tolist())


def compute_reconstruction_error(x, z, W_dec):
    """Compute relative reconstruction error ||x - z @ W_dec|| / ||x||"""
    if isinstance(z, np.ndarray):
        z_t = torch.from_numpy(z).float()
    else:
        z_t = z
    x_hat = z_t @ W_dec.cpu()  # [d_model]
    x_cpu = x.cpu()
    err = torch.norm(x_cpu - x_hat).item()
    norm_x = torch.norm(x_cpu).item()
    return err / (norm_x + 1e-8)


def compute_absorption_rate(active_sets_per_token, absorbed_ids, letter_tokens):
    """
    Compute absorption rate for a set of letter tokens.

    Absorption: a token where the letter is 'active in representation' but absorbed features don't fire.

    Simplified: For tokens that are letter tokens:
    - absorption rate = fraction where NONE of the absorbed features fire

    This is a simplified version of the Chanin IG pipeline.
    For the pilot, we measure: what fraction of letter-containing tokens have NO absorbed features active?
    """
    if not letter_tokens:
        return None, 0, 0

    n_absorbed_fire = 0
    n_total = len(letter_tokens)

    for tok_idx in letter_tokens:
        active = active_sets_per_token[tok_idx]
        # Check if any absorbed feature fires on this token
        if absorbed_ids & active:
            n_absorbed_fire += 1

    absorption_rate = 1.0 - (n_absorbed_fire / n_total)
    return absorption_rate, n_absorbed_fire, n_total


def main():
    device = torch.device(f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '4').split(',')[0]}"
                          if torch.cuda.is_available() else "cpu")
    # If CUDA_VISIBLE_DEVICES is set to a single GPU, use cuda:0
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    write_progress(1, 10, {"status": "loading_labels"})

    # === Step 1: Load absorbed feature IDs ===
    print("\n=== Step 1: Loading absorbed feature IDs ===")
    absorbed_ids, n_absorbed, n_non_absorbed = load_absorbed_feature_ids()

    write_progress(2, 10, {"status": "loading_sae"})

    # === Step 2: Load GPT-2 L6 SAE via SAELens ===
    print("\n=== Step 2: Loading GPT-2 L6 SAE ===")
    from sae_lens import SAE

    sae, cfg_dict, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",
        device=str(device),
    )
    sae.eval()
    print(f"SAE loaded: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # Extract weight matrices
    # W_enc: [d_sae, d_in], b_enc: [d_sae]
    # W_dec: [d_sae, d_in] (decoder weight, normalized rows)
    W_enc = sae.W_enc.detach()  # SAELens: [d_in, d_sae] = [768, 24576]
    b_enc = sae.b_enc.detach()  # [d_sae]
    W_dec = sae.W_dec.detach()  # SAELens: [d_sae, d_in] = [24576, 768]

    print(f"W_enc shape: {W_enc.shape}, W_dec shape: {W_dec.shape}")
    # Verify shapes
    d_in = W_enc.shape[0]
    d_sae = W_enc.shape[1]
    print(f"d_in={d_in}, d_sae={d_sae}")

    # Compute K = mean L0 (will be done from data, use heuristic ~50 for now)
    # We'll compute it from actual feedforward outputs

    write_progress(3, 10, {"status": "loading_gpt2"})

    # === Step 3: Load GPT-2 and get activations ===
    print("\n=== Step 3: Loading GPT-2 and getting activations ===")
    import transformer_lens
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained("gpt2", device=str(device))
    model.eval()
    print("GPT-2 loaded")

    write_progress(4, 10, {"status": "loading_tokens"})

    # === Step 4: Get 100 tokens from OpenWebText or generate simple text ===
    print("\n=== Step 4: Preparing tokens ===")

    # Try to use cached tokens first
    cached_tokens_path = WORKSPACE / "iter_002/exp/results/owt_tokens_cache.pt"
    if cached_tokens_path.exists():
        print(f"Loading cached tokens from {cached_tokens_path}")
        tokens = torch.load(cached_tokens_path)
        if tokens.dim() > 1:
            tokens = tokens.flatten()
        tokens = tokens[:N_TOKENS + 50]  # extra buffer
    else:
        # Fallback: generate a short corpus of text about letters a, e, s
        print("No cached tokens found. Generating text corpus.")
        texts = [
            "The animal at the edge of the field sat quietly, eating small seeds.",
            "Every element of the sentence starts with a specific sound.",
            "Some abstract things are actually quite simple to understand.",
            "An apple a day keeps the doctor away, as the saying goes.",
            "Every student in the class eagerly awaited the exam results.",
            "Small animals sometimes escape from their enclosures at the zoo.",
            "The astronaut saw a strange sight as she entered the space station.",
            "An energetic cat sat on the mat, swatting at everything around.",
            "She sat at the edge of the sea, watching the waves arrive.",
            "Abstract art sometimes shows elements of the artist's soul.",
        ] * 20
        full_text = " ".join(texts)
        tokens = model.to_tokens(full_text, prepend_bos=True).squeeze(0)
        print(f"Generated {len(tokens)} tokens")

    # Limit to N_TOKENS
    if len(tokens) > N_TOKENS:
        tokens = tokens[:N_TOKENS]
    tokens = tokens.to(device)
    print(f"Using {len(tokens)} tokens")

    write_progress(5, 10, {"status": "extracting_activations"})

    # === Step 5: Extract residual stream activations at L6 ===
    print("\n=== Step 5: Extracting residual stream activations at L6 ===")

    # Run model and capture hook at blocks.6.hook_resid_pre
    residual_activations = []

    def hook_fn(value, hook):
        residual_activations.append(value.detach().cpu())
        return value

    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens.unsqueeze(0),  # [1, seq_len]
            names_filter=lambda name: name == "blocks.6.hook_resid_pre",
        )

    # cache["blocks.6.hook_resid_pre"] has shape [1, seq_len, d_model]
    acts = cache["blocks.6.hook_resid_pre"].squeeze(0)  # [seq_len, 768]
    print(f"Residual activations shape: {acts.shape}")

    # Get the tokenized string for letter detection
    all_tokens = tokens.cpu().numpy().tolist()
    token_strings = [model.tokenizer.decode([t]) for t in all_tokens]

    write_progress(6, 10, {"status": "finding_letter_tokens"})

    # === Step 6: Identify which tokens correspond to our target letters ===
    print("\n=== Step 6: Finding letter tokens ===")

    letter_token_indices = {letter: [] for letter in TARGET_LETTERS}
    for idx, tok_str in enumerate(token_strings):
        tok_lower = tok_str.lower().strip()
        for letter in TARGET_LETTERS:
            # Token starts with the letter (including " a", " e", " s" etc.)
            if tok_lower.startswith(letter) or tok_lower == letter:
                letter_token_indices[letter].append(idx)

    for letter, idxs in letter_token_indices.items():
        print(f"  Letter '{letter}': {len(idxs)} tokens")

    write_progress(7, 10, {"status": "encoding_all_conditions"})

    # === Step 7: Run all 3 encoding conditions ===
    print("\n=== Step 7: Running encoding conditions ===")

    # Move activations to device
    acts_device = acts.to(device)  # [seq_len, 768]
    W_dec_np = W_dec.cpu().numpy()  # [24576, 768]

    # 7A: Feedforward encoding for all tokens
    # Use sae.encode() which correctly handles apply_b_dec_to_input and other preprocessing
    print("7A: Feedforward encoding...")
    with torch.no_grad():
        z_A_all = sae.encode(acts_device)  # [seq_len, d_sae]

    # Compute K = mean L0 of feedforward encoding (use strict threshold > 0)
    l0_per_token = (z_A_all > 0).sum(dim=1).float()
    mean_l0 = l0_per_token.mean().item()
    K = max(1, round(mean_l0))
    print(f"  Mean L0 of feedforward: {mean_l0:.1f}, using K={K} for OMP")

    # Reconstruction errors for feedforward using SAE decode
    with torch.no_grad():
        x_hat_A = sae.decode(z_A_all)  # [seq_len, 768]
    recon_err_A_per_token = torch.norm(acts_device - x_hat_A, dim=1) / (torch.norm(acts_device, dim=1) + 1e-8)
    mean_recon_err_A = recon_err_A_per_token.mean().item()
    print(f"  Mean feedforward reconstruction error: {mean_recon_err_A:.4f}")

    # Active sets for feedforward
    z_A_np = z_A_all.cpu().numpy()
    active_sets_A = [get_active_latents(z_A_np[i]) for i in range(len(token_strings))]

    # 7B: OMP encoding
    # For OMP, we operate on the SAE input (after b_dec preprocessing if applicable)
    # SAE input: x_in = x - b_dec (when apply_b_dec_to_input=True)
    # OMP finds sparse code z_B such that z_B @ W_dec ≈ x_in
    # Then reconstruction: x_hat = z_B @ W_dec + b_dec
    print(f"7B: OMP encoding (K={K})...")
    b_dec_np = sae.b_dec.detach().cpu().numpy()  # [d_in]
    apply_b_dec = sae.cfg.apply_b_dec_to_input

    acts_cpu = acts.cpu()  # [seq_len, 768]
    acts_np = acts_cpu.numpy()

    z_B_all = np.zeros((len(token_strings), d_sae))
    omp_errors = []

    for i in range(len(token_strings)):
        xi = acts_np[i]  # [768]
        # Apply same preprocessing as SAE encode
        xi_in = xi - b_dec_np if apply_b_dec else xi
        z_b = omp_encode(xi_in, W_dec_np, K)
        z_B_all[i] = z_b
        # Reconstruction: x_hat = z @ W_dec + b_dec
        x_hat = z_b @ W_dec_np + b_dec_np  # [768]
        err = np.linalg.norm(xi - x_hat) / (np.linalg.norm(xi) + 1e-8)
        omp_errors.append(err)

    mean_recon_err_B = np.mean(omp_errors)
    mean_l0_B = np.mean([(z_B_all[i] > 0).sum() for i in range(len(token_strings))])
    print(f"  Mean OMP reconstruction error: {mean_recon_err_B:.4f}")
    print(f"  Mean OMP L0: {mean_l0_B:.1f} (target K={K})")

    active_sets_B = [get_active_latents(z_B_all[i]) for i in range(len(token_strings))]

    # 7C: 2-pass encoding
    # For 2-pass, we use the SAE's preprocessing correctly
    print("7C: 2-pass residual encoding...")
    with torch.no_grad():
        # First pass: standard encode
        z1_all = sae.encode(acts_device)  # [seq_len, d_sae]
        x_hat1 = sae.decode(z1_all)       # [seq_len, 768]
        r_all = acts_device - x_hat1       # [seq_len, 768] residual
        # Second pass: encode residual
        z2_all = sae.encode(r_all)         # [seq_len, d_sae]
        z_C_all = z1_all + 0.5 * z2_all   # [seq_len, d_sae]

    with torch.no_grad():
        x_hat_C = sae.decode(z_C_all)
    recon_err_C_per_token = torch.norm(acts_device - x_hat_C, dim=1) / (torch.norm(acts_device, dim=1) + 1e-8)
    mean_recon_err_C = recon_err_C_per_token.mean().item()
    mean_l0_C = (z_C_all > 0).sum(dim=1).float().mean().item()
    print(f"  Mean 2-pass reconstruction error: {mean_recon_err_C:.4f}")
    print(f"  Mean 2-pass L0: {mean_l0_C:.1f}")

    z_C_np = z_C_all.cpu().numpy()
    active_sets_C = [get_active_latents(z_C_np[i]) for i in range(len(token_strings))]

    write_progress(8, 10, {"status": "computing_absorption"})

    # === Step 8: Compute absorption rates ===
    print("\n=== Step 8: Computing absorption rates ===")

    # Sanity check: OMP reconstruction < feedforward?
    omp_better = mean_recon_err_B < mean_recon_err_A
    print(f"OMP reconstruction error ({mean_recon_err_B:.4f}) < feedforward ({mean_recon_err_A:.4f}): {omp_better}")

    absorption_results = {}

    for letter in TARGET_LETTERS:
        l_tokens = letter_token_indices[letter]
        print(f"\nLetter '{letter}': {len(l_tokens)} tokens")

        if len(l_tokens) == 0:
            absorption_results[letter] = {
                "n_tokens": 0,
                "absorption_rate_A": None,
                "absorption_rate_B": None,
                "absorption_rate_C": None,
                "note": "No tokens found for this letter"
            }
            continue

        ar_A, n_fire_A, n_total = compute_absorption_rate(active_sets_A, absorbed_ids, l_tokens)
        ar_B, n_fire_B, _ = compute_absorption_rate(active_sets_B, absorbed_ids, l_tokens)
        ar_C, n_fire_C, _ = compute_absorption_rate(active_sets_C, absorbed_ids, l_tokens)

        print(f"  Feedforward: AR={ar_A:.3f} ({n_total-n_fire_A}/{n_total} tokens absorbed)")
        print(f"  OMP:         AR={ar_B:.3f} ({n_total-n_fire_B}/{n_total} tokens absorbed)")
        print(f"  2-pass:      AR={ar_C:.3f} ({n_total-n_fire_C}/{n_total} tokens absorbed)")

        omp_reduction = None
        if ar_A is not None and ar_A > 0:
            omp_reduction = (ar_A - ar_B) / ar_A if ar_B is not None else None

        absorption_results[letter] = {
            "n_tokens": n_total,
            "absorption_rate_A": ar_A,
            "absorption_rate_B": ar_B,
            "absorption_rate_C": ar_C,
            "n_fire_A": n_fire_A,
            "n_fire_B": n_fire_B,
            "n_fire_C": n_fire_C,
            "omp_reduction_ratio": omp_reduction,
        }

    # Overall absorption rate across letters
    all_letter_tokens = []
    for letter in TARGET_LETTERS:
        all_letter_tokens.extend(letter_token_indices[letter])
    all_letter_tokens = list(set(all_letter_tokens))

    ar_A_all, n_fire_A_all, n_total_all = compute_absorption_rate(active_sets_A, absorbed_ids, all_letter_tokens)
    ar_B_all, n_fire_B_all, _ = compute_absorption_rate(active_sets_B, absorbed_ids, all_letter_tokens)
    ar_C_all, n_fire_C_all, _ = compute_absorption_rate(active_sets_C, absorbed_ids, all_letter_tokens)

    overall_omp_reduction = None
    if ar_A_all is not None and ar_A_all > 0 and ar_B_all is not None:
        overall_omp_reduction = (ar_A_all - ar_B_all) / ar_A_all

    print(f"\nOverall (all 3 letters, {n_total_all} tokens):")
    print(f"  Feedforward AR: {ar_A_all:.3f}")
    print(f"  OMP AR: {ar_B_all:.3f}")
    print(f"  2-pass AR: {ar_C_all:.3f}")
    print(f"  OMP reduction ratio: {overall_omp_reduction}")

    write_progress(9, 10, {"status": "saving_results"})

    # === Step 9: Save results ===
    print("\n=== Step 9: Saving results ===")

    elapsed = time.time() - start_time

    # Determine pass/fail
    sanity_pass = omp_better  # OMP recon < feedforward
    absorption_measured = all(
        absorption_results[l].get('absorption_rate_A') is not None
        for l in TARGET_LETTERS
        if absorption_results[l]['n_tokens'] > 0
    )

    # Determine direction of effect
    if ar_A_all is not None and ar_B_all is not None:
        direction = "OMP_reduces_absorption" if ar_B_all < ar_A_all else "OMP_increases_absorption"
    else:
        direction = "undetermined"

    # Significant difference criterion: > 5 pp
    omp_differs_significantly = None
    if ar_A_all is not None and ar_B_all is not None:
        omp_differs_significantly = abs(ar_A_all - ar_B_all) > 0.05

    result = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "seed": SEED,
        "n_tokens": N_TOKENS,
        "target_letters": TARGET_LETTERS,
        "absorbed_feature_ids": sorted(absorbed_ids),
        "n_absorbed_features": len(absorbed_ids),

        "encoding_stats": {
            "K_omp": K,
            "mean_l0_feedforward": mean_l0,
            "mean_l0_omp": float(mean_l0_B),
            "mean_l0_twopass": float(mean_l0_C),
            "mean_recon_err_feedforward": float(mean_recon_err_A),
            "mean_recon_err_omp": float(mean_recon_err_B),
            "mean_recon_err_twopass": float(mean_recon_err_C),
            "omp_recon_better_than_feedforward": bool(omp_better),
        },

        "absorption_by_letter": absorption_results,

        "overall_absorption": {
            "n_letter_tokens_total": n_total_all,
            "absorption_rate_feedforward": float(ar_A_all) if ar_A_all is not None else None,
            "absorption_rate_omp": float(ar_B_all) if ar_B_all is not None else None,
            "absorption_rate_twopass": float(ar_C_all) if ar_C_all is not None else None,
            "omp_reduction_ratio": float(overall_omp_reduction) if overall_omp_reduction is not None else None,
            "direction": direction,
            "omp_differs_by_gt_5pp": bool(omp_differs_significantly) if omp_differs_significantly is not None else None,
        },

        "pilot_pass_criteria": {
            "omp_recon_lt_feedforward": bool(sanity_pass),
            "absorption_measured_for_letters": bool(absorption_measured),
            "direction_reported": bool(direction != "undetermined"),
            "pass": bool(sanity_pass and absorption_measured),
        },

        "proceed_to_C2": bool(omp_differs_significantly) if omp_differs_significantly is not None else False,
        "notes": []
    }

    # Add interpretation notes
    if overall_omp_reduction is not None:
        if overall_omp_reduction >= 0.50:
            result["notes"].append(f"OMP reduces absorption by {overall_omp_reduction:.1%} — amortization gap dominant signal")
        elif overall_omp_reduction >= 0.20:
            result["notes"].append(f"OMP reduces absorption by {overall_omp_reduction:.1%} — mixed signal (amortization + sparsity landscape)")
        elif overall_omp_reduction >= 0.05:
            result["notes"].append(f"OMP reduces absorption by {overall_omp_reduction:.1%} — marginal effect, borderline for C2")
        elif overall_omp_reduction < 0:
            result["notes"].append(f"OMP INCREASES absorption by {abs(overall_omp_reduction):.1%} — unexpected direction")
        else:
            result["notes"].append(f"OMP reduces absorption by only {overall_omp_reduction:.1%} — sparsity landscape likely dominant")

    if not sanity_pass:
        result["notes"].append(
            f"WARNING: OMP recon error ({mean_recon_err_B:.4f}) NOT better than feedforward ({mean_recon_err_A:.4f}) — "
            "check OMP implementation (W_dec orientation)"
        )

    # Save to pilots/
    output_path = RESULTS_DIR_PILOTS / "C1_omp_pilot.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {output_path}")

    write_progress(10, 10, {"status": "done", "pass": result["pilot_pass_criteria"]["pass"]})

    print("\n=== PILOT SUMMARY ===")
    print(f"Sanity check (OMP recon < feedforward): {sanity_pass}")
    print(f"Absorption measured for target letters: {absorption_measured}")
    print(f"Direction of OMP effect: {direction}")
    print(f"OMP reduction ratio: {overall_omp_reduction}")
    print(f"Proceed to C2: {result['proceed_to_C2']}")
    print(f"Pilot PASS: {result['pilot_pass_criteria']['pass']}")
    print(f"Elapsed: {elapsed:.1f}s")

    return result


if __name__ == "__main__":
    try:
        result = main()
        mark_done(
            status="success",
            summary=f"OMP pilot complete. sanity_pass={result['pilot_pass_criteria']['omp_recon_lt_feedforward']}, "
                    f"direction={result['overall_absorption']['direction']}, "
                    f"proceed_to_C2={result['proceed_to_C2']}"
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nFATAL ERROR: {e}\n{tb}")
        mark_done(status="failure", summary=f"Error: {str(e)[:200]}")
        sys.exit(1)
