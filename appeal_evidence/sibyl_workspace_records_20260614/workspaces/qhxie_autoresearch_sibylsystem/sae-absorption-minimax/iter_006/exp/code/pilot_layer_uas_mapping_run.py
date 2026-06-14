#!/usr/bin/env python3
"""
Pilot H-L1: Layer-wise UAS Saturation Mapping
Iteration 10 - Map UAS variance across layers 4, 6, 8, 10 to find which layer(s) permit measurable absorption variation.

Critical diagnostic: if ALL layers saturate, absorption metric is degenerate.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

# TransformerLens and SAE Lens imports
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/iter_006")
RESULTS_DIR = WORKSPACE / "exp" / "results"
DEVICE = "cuda:0"  # Use GPU 0
SEED = 42
N_FEATURES_PER_LAYER = 30
N_TOKENS = 500
TAU_FS = 0.03

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def write_pid(results_dir):
    pid_file = Path(results_dir) / "pilot_layer_uas_mapping.pid"
    pid_file.write_text(str(os.getpid()))


def report_progress(epoch=0, total_epochs=1, step=0, metric=None):
    progress_file = Path(RESULTS_DIR) / "pilot_layer_uas_mapping_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": "pilot_layer_uas_mapping",
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "metric": metric,
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    results_dir = RESULTS_DIR
    pid_file = results_dir / "pilot_layer_uas_mapping.pid"
    if pid_file.exists():
        pid_file.unlink()
    marker = results_dir / "pilot_layer_uas_mapping_DONE"
    marker.write_text(json.dumps({
        "task_id": "pilot_layer_uas_mapping",
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))


def get_model():
    """Load GPT-2 Small."""
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    model.eval()
    return model


def get_sae(layer):
    """Load SAE for a specific layer."""
    print(f"Loading SAE for layer {layer}...")
    sae, cfg, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=DEVICE
    )
    sae.eval()
    return sae, cfg


def get_tokenizer():
    """Get GPT-2 tokenizer."""
    return HookedTransformer.from_pretrained("gpt2-small", device=DEVICE).tokenizer


def compute_absorption_score(resid_acc, sae_acc):
    """Compute UAS (Universal Absorption Score) as per Chanin 2024."""
    if resid_acc is None or sae_acc is None:
        return None
    denominator = 1 - sae_acc
    if denominator == 0:
        return 0.0
    uas = (resid_acc - sae_acc) / denominator
    return float(uas)


def chanin_absorption_protocol(model, sae, tokens, layer, tau_fs=0.03):
    """
    Implements Chanin 2024 absorption detection protocol.
    Task: Classify tokens into A-M vs N-Z first-letter bins.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    tokenizer = get_tokenizer()
    n_features = sae.W_dec.shape[0]

    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    # Get first letter classification
    first_letters = []
    for i in range(tokens.shape[0]):
        for j in range(tokens.shape[1]):
            tok_id = int(tokens[i, j].item())
            tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)
            if tok_text and len(tok_text) > 0 and tok_text[0].isalpha():
                first_letters.append(1 if tok_text[0].upper() in 'MNOPQRSTUVWXYZ' else 0)
            else:
                first_letters.append(-1)

    first_letters = np.array(first_letters)
    batch_size, seq_len = tokens.shape
    first_letters = first_letters.reshape(batch_size, seq_len)
    valid_mask = first_letters != -1

    if valid_mask.sum() < 50:
        return {}

    # Get SAE activations
    _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{layer}.hook_resid_pre"])
    resid_post = cache[f"blocks.{layer}.hook_resid_pre"]

    with torch.no_grad():
        sae_acts = sae.encode(resid_post)

    # Flatten
    sae_acts = sae_acts.reshape(batch_size * seq_len, n_features)
    resid_post = resid_post.reshape(batch_size * seq_len, -1)
    first_letters_flat = first_letters.reshape(-1)
    valid_mask_flat = valid_mask.reshape(-1)

    valid_labels = first_letters_flat[valid_mask_flat]
    valid_sae_acts = sae_acts[valid_mask_flat]
    valid_resid_acts = resid_post[valid_mask_flat]

    # Pre-identify features with sufficient activations
    feature_means = valid_sae_acts.mean(dim=0).cpu().numpy()
    candidate_features = np.where(feature_means > tau_fs)[0]
    print(f"Found {len(candidate_features)} features with mean activation > {tau_fs}")

    results = {}

    for f_idx in tqdm(candidate_features[:N_FEATURES_PER_LAYER], desc=f"Layer {layer} absorption"):
        f_idx_int = int(f_idx)
        feature_acts = valid_sae_acts[:, f_idx].cpu().numpy()

        active_mask = feature_acts > tau_fs
        if active_mask.sum() < 10:
            continue

        active_labels = valid_labels[active_mask]
        if len(active_labels) < 10:
            continue

        if len(np.unique(active_labels)) < 2:
            continue

        try:
            X_sae_active = feature_acts[active_mask].reshape(-1, 1)
            X_resid_active = valid_resid_acts[active_mask].cpu().numpy()
            y_active = active_labels

            clf_sae = LogisticRegression(max_iter=500, random_state=SEED)
            clf_sae.fit(X_sae_active, y_active)
            acc_sae = clf_sae.score(X_sae_active, y_active)

            resid_dim = min(50, X_resid_active.shape[1])
            clf_resid = LogisticRegression(max_iter=500, random_state=SEED)
            clf_resid.fit(X_resid_active[:, :resid_dim], y_active)
            acc_resid = clf_resid.score(X_resid_active[:, :resid_dim], y_active)

            uas = compute_absorption_score(acc_resid, acc_sae)

            results[f_idx_int] = {
                'uas': uas,
                'acc_resid': float(acc_resid),
                'acc_sae': float(acc_sae),
                'n_active': int(active_mask.sum()),
            }
        except Exception as e:
            continue

    return results


def run_pilot_layer_uas_mapping():
    """
    Task: Map UAS variance across layers 4, 6, 8, 10.
    Pass criteria: UAS std > 0.1 at any layer (falsification: ALL layers std < 0.05)
    """
    print("\n" + "="*80)
    print("PILOT H-L1: Layer-wise UAS Saturation Mapping")
    print("="*80)

    write_pid(RESULTS_DIR)
    report_progress(epoch=0, total_epochs=1, step=0, metric={"status": "starting"})

    model = get_model()
    tokenizer = get_tokenizer()

    # Diverse text samples
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The pen is mightier than the sword.",
        "Actions speak louder than words.",
        "Better late than never.",
        "Birds of a feather flock together.",
    ] * (N_TOKENS // 50 + 1)

    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)["input_ids"].to(DEVICE)

    layers = [4, 6, 8, 10]
    results = {
        'task_id': 'pilot_layer_uas_mapping',
        'hypothesis': 'H-L1, H-L2',
        'timestamp': datetime.now().isoformat(),
        'layers': {}
    }

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        report_progress(step=layers.index(layer), total_epochs=len(layers), metric={"status": f"processing layer {layer}"})
        sae, _ = get_sae(layer)

        layer_results = chanin_absorption_protocol(model, sae, tokens, layer, TAU_FS)

        # Compute statistics
        uas_values = [r['uas'] for r in layer_results.values() if r['uas'] is not None and not np.isnan(r['uas'])]
        if len(uas_values) > 0:
            uas_mean = np.mean(uas_values)
            uas_std = np.std(uas_values)
            uas_min = np.min(uas_values)
            uas_max = np.max(uas_values)
        else:
            uas_mean = uas_std = uas_min = uas_max = np.nan

        results['layers'][str(layer)] = {
            'n_features': len(layer_results),
            'n_valid_uas': len(uas_values),
            'uas_mean': float(uas_mean) if not np.isnan(uas_mean) else None,
            'uas_std': float(uas_std) if not np.isnan(uas_std) else None,
            'uas_min': float(uas_min) if not np.isnan(uas_min) else None,
            'uas_max': float(uas_max) if not np.isnan(uas_max) else None,
            'features': {str(k): v for k, v in layer_results.items()}
        }

        print(f"Layer {layer}: {len(layer_results)} features, UAS std = {uas_std:.4f}")

    # Overall assessment
    all_stds = [results['layers'][str(l)]['uas_std'] for l in layers if results['layers'][str(l)]['uas_std'] is not None]
    max_std = max(all_stds) if all_stds else 0.0
    best_layer = layers[[results['layers'][str(l)]['uas_std'] for l in layers].index(max_std)] if all_stds else None

    results['overall'] = {
        'max_uas_std': float(max_std),
        'best_layer': best_layer,
        'all_stds': {str(l): results['layers'][str(l)]['uas_std'] for l in layers},
        'pass_criteria': 'UAS std > 0.1 at any layer',
        'falsification': 'ALL layers std < 0.05'
    }

    # Hypothesis evaluation
    h_l1_passed = max_std > 0.1
    h_l2_passed = best_layer == 4 if best_layer is not None else False

    results['hypothesis_results'] = {
        'H-L1': {
            'pass': h_l1_passed,
            'max_std': float(max_std),
            'criterion': 'UAS std > 0.1 at any layer',
            'falsification': 'ALL layers std < 0.05'
        },
        'H-L2': {
            'pass': h_l2_passed,
            'best_layer': best_layer,
            'criterion': 'Layer 4 has highest variance',
            'actual_best_layer': best_layer
        }
    }

    # Save results
    output_file = RESULTS_DIR / "pilots" / "layer_uas_mapping.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print(f"\nH-L1 (any layer std > 0.1): {'PASS' if h_l1_passed else 'FAIL'} (max std = {max_std:.4f})")
    print(f"H-L2 (layer 4 best): {'PASS' if h_l2_passed else 'FAIL'} (best layer = {best_layer})")

    mark_done(status="success", summary=f"H-L1: {'PASS' if h_l1_passed else 'FAIL'} (max_std={max_std:.4f}), H-L2: best_layer={best_layer}")

    return results


if __name__ == "__main__":
    results = run_pilot_layer_uas_mapping()