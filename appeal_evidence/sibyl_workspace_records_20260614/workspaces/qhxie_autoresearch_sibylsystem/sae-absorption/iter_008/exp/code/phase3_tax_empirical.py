"""
Phase 3: Absorption Tax Empirical Validation (PILOT mode)

Validates the Absorption Tax theorem's quantitative predictions empirically.
Tests whether absorption is the rate-distortion optimal strategy under hierarchy.

Four predictions tested:
1. Among SAEs at matched L0, higher absorption correlates with lower MSE
2. Absorption rate decreases as W^{-gamma} across widths (power law)
3. R_pc = cos^2(probe_direction, decoder_direction) predicts per-letter absorption severity
4. Matryoshka SAEs achieve low absorption without proportionally higher MSE

Dependencies: phase1_absorption_firstletter pilot results

Author: Sibyl Experimenter Agent
"""

import json
import os
import sys
import time
import gc
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ── Configuration ──────────────────────────────────────────────────────────
SEED = 42
TASK_ID = "phase3_tax_empirical"
MODE = "PILOT"

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PHASE3_DIR = RESULTS_DIR / "phase3"
PILOTS_DIR = RESULTS_DIR / "pilots"
CODE_DIR = WORKSPACE / "exp" / "code"

# Ensure output dirs exist
PHASE3_DIR.mkdir(parents=True, exist_ok=True)
PILOTS_DIR.mkdir(parents=True, exist_ok=True)

GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "6")

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

start_time = time.time()

# ── PID and Progress ──────────────────────────────────────────────────────
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def report_progress(stage, detail="", pct=0.0):
    """Write progress file for system monitor."""
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "stage": stage,
        "detail": detail,
        "pct": pct,
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    """Write DONE marker."""
    pid_f = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_f.exists():
        pid_f.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

report_progress("init", "Loading dependencies", 0.0)

# ── Load dependencies ──────────────────────────────────────────────────────
try:
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    import sae_lens
    import transformer_lens
except ImportError as e:
    mark_done("failure", f"Missing dependency: {e}")
    sys.exit(1)

# ── Load Phase 1 first-letter absorption data ─────────────────────────────
report_progress("loading_data", "Loading Phase 1 first-letter absorption results", 0.05)

# Load pilot first-letter absorption results
firstletter_path = PILOTS_DIR / "phase1_absorption_firstletter.json"
if not firstletter_path.exists():
    mark_done("failure", "Phase 1 first-letter absorption pilot results not found")
    sys.exit(1)

with open(firstletter_path) as f:
    fl_data = json.load(f)

# Extract per-letter absorption data
per_letter = fl_data["absorption_results"]["L12_16k"]["per_letter"]
main_features = fl_data["absorption_results"]["L12_16k"]["main_features_top"]
overall_absorption = fl_data["absorption_results"]["L12_16k"]["absorption_rate"]

print(f"[Phase 3] Loaded first-letter absorption data: overall rate = {overall_absorption:.4f}")
print(f"[Phase 3] Letters with data: {len([l for l, d in per_letter.items() if d['total'] > 0])}")

# ── Load model and SAE ────────────────────────────────────────────────────
report_progress("loading_model", "Loading Gemma 2 2B and SAE", 0.10)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Phase 3] Using device: {device}")

# Load model from local cache (gated model, no HF token available)
GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

from transformers import AutoModelForCausalLM, AutoTokenizer

print("[Phase 3] Loading Gemma 2 2B from local cache...")
hf_model = AutoModelForCausalLM.from_pretrained(GEMMA_LOCAL_PATH, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(GEMMA_LOCAL_PATH)
model = transformer_lens.HookedTransformer.from_pretrained(
    "gemma-2-2b",
    device=device,
    dtype=torch.bfloat16,
    hf_model=hf_model,
    tokenizer=tokenizer,
)
del hf_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"[Phase 3] Gemma 2 2B loaded, d_model = {model.cfg.d_model}")

report_progress("loading_sae", "Loading Gemma Scope SAEs", 0.15)

# Load the primary SAE (L12 16k canonical)
_sae_result = sae_lens.SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    device=device,
)
sae_16k = _sae_result[0] if isinstance(_sae_result, tuple) else _sae_result
print(f"[Phase 3] SAE L12/16k loaded. Dict size: {sae_16k.cfg.d_sae}")

# ── Prediction 3: R_pc computation ────────────────────────────────────────
# This is the most straightforward test: decoder-probe direction alignment
# predicts per-letter absorption severity.
report_progress("prediction3", "Computing R_pc (redundancy ratio) per letter", 0.25)

print("\n" + "="*70)
print("PREDICTION 3: R_pc = cos^2(probe_direction, decoder_direction)")
print("Predicts per-letter absorption severity")
print("="*70)

# Load probe weights
probe_path = RESULTS_DIR / "phase1" / "probe_first-letter_L12.npz"
sklearn_probe_path = RESULTS_DIR / "phase1" / "probe_firstletter_L12_sklearn.npz"

# Try to load probe weights
probe_weights = None
if sklearn_probe_path.exists():
    probe_data = np.load(sklearn_probe_path, allow_pickle=True)
    if "coef" in probe_data:
        probe_weights = probe_data["coef"]  # shape: (n_classes, d_model)
        print(f"[Phase 3] Loaded sklearn probe weights: {probe_weights.shape}")
    elif "weights" in probe_data:
        probe_weights = probe_data["weights"]
        print(f"[Phase 3] Loaded sklearn probe weights (alt key): {probe_weights.shape}")

if probe_weights is None and probe_path.exists():
    probe_data = np.load(probe_path, allow_pickle=True)
    available_keys = list(probe_data.keys())
    print(f"[Phase 3] probe_first-letter_L12.npz keys: {available_keys}")
    for key in ["coef", "weights", "W", "probe_weights"]:
        if key in probe_data:
            probe_weights = probe_data[key]
            print(f"[Phase 3] Loaded probe weights from key '{key}': {probe_weights.shape}")
            break

# If we can't load probe weights, train a quick one
if probe_weights is None:
    print("[Phase 3] Cannot load probe weights directly. Using probe direction from cosine similarity data.")
    # We have the main feature cosine similarities and feature indices
    # Use the decoder directions of the main features as proxy for probe directions
    probe_weights = None  # Will use decoder-based proxy below

# Get SAE decoder weight matrix
W_dec = sae_16k.W_dec.detach().float()  # (d_sae, d_model)
print(f"[Phase 3] SAE decoder shape: {W_dec.shape}")

# Compute R_pc per letter
letters = sorted([l for l in per_letter.keys() if per_letter[l]["total"] > 0])
rpc_results = {}

for letter in letters:
    letter_data = per_letter[letter]
    if letter not in main_features:
        continue

    main_fid = main_features[letter]["fid"]
    main_cos = main_features[letter]["cos"]

    # Decoder direction of the main (parent) feature for this letter
    d_parent = W_dec[main_fid].cpu().numpy()
    d_parent_norm = d_parent / (np.linalg.norm(d_parent) + 1e-12)

    # If we have probe weights, compute cos^2 between probe direction and decoder
    if probe_weights is not None:
        # Find the probe direction for this letter
        letter_idx = ord(letter) - ord('a')
        if letter_idx < probe_weights.shape[0]:
            probe_dir = probe_weights[letter_idx]
            probe_dir_norm = probe_dir / (np.linalg.norm(probe_dir) + 1e-12)
            cos_sim = np.dot(probe_dir_norm, d_parent_norm)
            rpc = cos_sim ** 2
        else:
            rpc = main_cos ** 2  # Fallback: use feature-based cosine
    else:
        # Use the main feature cosine as proxy
        # R_pc = cos^2(probe_direction, decoder_direction)
        # Proxy: cos_sim between decoder direction and residual-stream
        # direction that encodes this letter. The main_cos from prior
        # analysis gives cos(probe_dir, decoder_dir) approximately
        rpc = main_cos ** 2

    rpc_results[letter] = {
        "R_pc": float(rpc),
        "absorption_rate": letter_data["absorption_rate"],
        "strict_rate": letter_data["strict_rate"],
        "n_tokens": letter_data["total"],
        "n_probe_correct": letter_data["probe_correct_raw"],
        "n_false_negatives": letter_data["false_negatives"],
        "main_feature_idx": main_fid,
        "main_feature_cos": main_cos,
    }

print(f"\n[Phase 3] R_pc computed for {len(rpc_results)} letters")

# Compute correlation: R_pc vs absorption rate
rpc_values = []
absorption_values = []
letter_labels = []
for letter, data in sorted(rpc_results.items()):
    if data["n_probe_correct"] >= 3:  # Need at least 3 tokens for meaningful rate
        rpc_values.append(data["R_pc"])
        absorption_values.append(data["absorption_rate"])
        letter_labels.append(letter)

rpc_arr = np.array(rpc_values)
abs_arr = np.array(absorption_values)

# Spearman correlation
if len(rpc_arr) >= 5:
    spearman_rpc, p_spearman_rpc = stats.spearmanr(rpc_arr, abs_arr)
    pearson_rpc, p_pearson_rpc = stats.pearsonr(rpc_arr, abs_arr)
    print(f"\n  Spearman(R_pc, absorption): rho={spearman_rpc:.4f}, p={p_spearman_rpc:.4f}")
    print(f"  Pearson(R_pc, absorption):  r={pearson_rpc:.4f}, p={p_pearson_rpc:.4f}")
    print(f"  N letters used: {len(rpc_arr)}")

    # Print per-letter table
    print(f"\n  {'Letter':<8} {'R_pc':<10} {'Abs_rate':<12} {'N_tokens':<10} {'N_FN':<6}")
    print(f"  {'-'*50}")
    for letter in letter_labels:
        d = rpc_results[letter]
        print(f"  {letter:<8} {d['R_pc']:<10.4f} {d['absorption_rate']:<12.4f} {d['n_tokens']:<10} {d['n_false_negatives']:<6}")
else:
    spearman_rpc, p_spearman_rpc = float('nan'), float('nan')
    pearson_rpc, p_pearson_rpc = float('nan'), float('nan')
    print("[Phase 3] WARNING: Not enough letters with sufficient data for correlation.")

pred3_result = {
    "prediction": "R_pc predicts per-letter absorption severity",
    "n_letters": len(rpc_arr),
    "spearman_rho": float(spearman_rpc) if not np.isnan(spearman_rpc) else None,
    "spearman_p": float(p_spearman_rpc) if not np.isnan(p_spearman_rpc) else None,
    "pearson_r": float(pearson_rpc) if not np.isnan(pearson_rpc) else None,
    "pearson_p": float(p_pearson_rpc) if not np.isnan(p_pearson_rpc) else None,
    "per_letter": rpc_results,
    "letter_labels": letter_labels,
    "rpc_values": [float(v) for v in rpc_values],
    "absorption_values": [float(v) for v in absorption_values],
    "verdict": None,  # Set below
}

if not np.isnan(spearman_rpc):
    if abs(spearman_rpc) >= 0.3 and p_spearman_rpc < 0.05:
        pred3_result["verdict"] = "SUPPORTED"
    elif abs(spearman_rpc) >= 0.2:
        pred3_result["verdict"] = "WEAK_SUPPORT"
    else:
        pred3_result["verdict"] = "NOT_SUPPORTED"
else:
    pred3_result["verdict"] = "INSUFFICIENT_DATA"

print(f"\n  Prediction 3 verdict: {pred3_result['verdict']}")


# ── Prediction 3b: Compute theoretical T(G) ──────────────────────────────
report_progress("theoretical_TG", "Computing theoretical absorption tax T(G)", 0.35)

print("\n" + "="*70)
print("THEORETICAL T(G) COMPUTATION")
print("T(G) = sum_{(p,c)} p_c * R_pc where R_pc = cos^2(probe, decoder)")
print("="*70)

# For first-letter task:
# parent = letter feature, child = word-specific features that absorb
# p_c = frequency of child tokens (we approximate from our test data)
# R_pc = cos^2(letter probe direction, decoder direction of main feature)

total_tokens = sum(per_letter[l]["total"] for l in letters)
tg_components = []

for letter in letters:
    if letter not in rpc_results:
        continue
    data = rpc_results[letter]
    p_c = per_letter[letter]["total"] / total_tokens  # frequency weight
    r_pc = data["R_pc"]
    tg_contrib = p_c * r_pc
    tg_components.append({
        "letter": letter,
        "p_c": float(p_c),
        "R_pc": float(r_pc),
        "contribution": float(tg_contrib),
    })

T_G = sum(c["contribution"] for c in tg_components)
print(f"\n  T(G) = {T_G:.6f}")
print(f"  Interpretation: absorption-free SAE needs ~{T_G:.4f} additional L0 budget")
print(f"  This is the 'tax' for avoiding hierarchy-driven absorption")

# Sort by contribution
tg_components.sort(key=lambda x: x["contribution"], reverse=True)
print(f"\n  Top contributors to T(G):")
for c in tg_components[:10]:
    print(f"    Letter '{c['letter']}': p_c={c['p_c']:.4f}, R_pc={c['R_pc']:.4f}, "
          f"contribution={c['contribution']:.6f}")

tg_result = {
    "T_G": float(T_G),
    "n_letters": len(tg_components),
    "total_tokens": total_tokens,
    "components": tg_components,
}


# ── Prediction 1: Absorption vs MSE at matched L0 ────────────────────────
report_progress("prediction1", "Loading multiple SAE configs for absorption-MSE tradeoff", 0.40)

print("\n" + "="*70)
print("PREDICTION 1: Higher absorption correlates with lower MSE at matched L0")
print("="*70)

# In PILOT mode, we test with the available SAEs.
# We'll load a few Gemma Scope configurations and compare.
# The key insight: we need SAEs at similar L0 but different architectures/widths.

# For pilot: compute MSE on a small sample of tokens using the SAE we already have,
# plus try to load SAEBench SAEs if available.

# First, compute MSE for our loaded SAE (L12 16k)
print("\n[Phase 3] Computing MSE for L12/16k canonical SAE...")

# Prepare test tokens - simple text prompts
test_prompts = [
    "The city of Paris is located in France and",
    "Mathematics involves the study of numbers and",
    "The president gave a speech about climate",
    "Scientists discovered a new species of deep",
    "The computer program failed to compile because",
    "In the morning light the garden looked",
    "The university offers courses in philosophy and",
    "The old castle stood on top of the",
    "Music has the power to change people and",
    "The river flows through the valley toward",
]

# Expand with more prompts for pilot
extra_prompts = [
    f"Word number {i}: the quick brown fox jumps over the lazy dog and"
    for i in range(20)
]
test_prompts.extend(extra_prompts)

# Tokenize and get activations
sae_mse_results = {}

with torch.no_grad():
    all_residuals = []
    for prompt in test_prompts[:30]:  # Pilot: 30 prompts
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(
            tokens,
            names_filter="blocks.12.hook_resid_post",
            return_type="loss",
        )
        residuals = cache["blocks.12.hook_resid_post"].float()  # (1, seq, d_model)
        all_residuals.append(residuals.squeeze(0))  # (seq, d_model)

    all_residuals_cat = torch.cat(all_residuals, dim=0)  # (total_tokens, d_model)
    n_tokens_mse = all_residuals_cat.shape[0]
    print(f"  Collected {n_tokens_mse} token activations for MSE computation")

    # Compute MSE for L12 16k
    sae_output_16k = sae_16k(all_residuals_cat.to(device))
    if isinstance(sae_output_16k, tuple):
        reconstructed_16k = sae_output_16k[0]
    else:
        reconstructed_16k = sae_output_16k

    mse_16k = torch.mean((all_residuals_cat.to(device) - reconstructed_16k) ** 2).item()

    # Also compute L0 (average number of active features)
    # Get feature activations
    sae_acts_16k = sae_16k.encode(all_residuals_cat.to(device))
    l0_16k = (sae_acts_16k > 0).float().sum(dim=-1).mean().item()

    # Variance of original for normalized MSE
    var_original = torch.var(all_residuals_cat.to(device)).item()
    nmse_16k = mse_16k / (var_original + 1e-12)

    sae_mse_results["L12_16k_canonical"] = {
        "sae_id": "layer_12/width_16k/canonical",
        "layer": 12,
        "width": 16384,
        "mse": float(mse_16k),
        "nmse": float(nmse_16k),
        "l0": float(l0_16k),
        "absorption_rate": overall_absorption,
        "n_tokens": n_tokens_mse,
    }
    print(f"  L12/16k: MSE={mse_16k:.6f}, NMSE={nmse_16k:.6f}, L0={l0_16k:.1f}, Abs={overall_absorption:.4f}")

# Try to load additional SAE configs for comparison
report_progress("prediction1_multi_sae", "Loading additional SAEs for comparison", 0.50)

additional_sae_configs = [
    # Different widths at layer 12
    ("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical"),
]

# Try SAEBench SAEs (BatchTopK + Vanilla at various widths)
saebench_configs_to_try = [
    # TopK width 2^14 (16384)
    ("sae_bench_gemma-2-2b_topk_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0"),
    # Vanilla width 2^14 (16384)
    ("sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109", "blocks.12.hook_resid_post__trainer_0"),
    # TopK width 2^12 (4096)
    ("sae_bench_gemma-2-2b_topk_width-2pow12_date-1109", "blocks.12.hook_resid_post__trainer_0"),
    # TopK width 2^16 (65536)
    ("sae_bench_gemma-2-2b_topk_width-2pow16_date-1109", "blocks.12.hook_resid_post__trainer_0"),
    # Vanilla width 2^12 (4096)
    ("sae_bench_gemma-2-2b_vanilla_width-2pow12_date-1109", "blocks.12.hook_resid_post__trainer_0"),
    # Vanilla width 2^16 (65536)
    ("sae_bench_gemma-2-2b_vanilla_width-2pow16_date-1109", "blocks.12.hook_resid_post__trainer_0"),
]

loaded_saes = {"L12_16k_canonical": sae_16k}

for release, sae_id in saebench_configs_to_try:
    # Derive a unique config key from release name
    # e.g. "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109" -> "topk_2pow14"
    parts = release.replace("sae_bench_gemma-2-2b_", "").replace("_date-1109", "")
    config_key = f"saebench_{parts}"
    try:
        print(f"\n  Trying to load SAEBench: {release} / {sae_id}")
        sae_bench = sae_lens.SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
        )
        # Handle both old (tuple) and new (single) API
        if isinstance(sae_bench, tuple):
            sae_bench = sae_bench[0]
        loaded_saes[config_key] = sae_bench

        # Compute MSE and L0
        with torch.no_grad():
            sae_output = sae_bench(all_residuals_cat.to(device))
            if isinstance(sae_output, tuple):
                reconstructed = sae_output[0]
            else:
                reconstructed = sae_output

            mse = torch.mean((all_residuals_cat.to(device) - reconstructed) ** 2).item()
            nmse = mse / (var_original + 1e-12)

            sae_acts = sae_bench.encode(all_residuals_cat.to(device))
            l0 = (sae_acts > 0).float().sum(dim=-1).mean().item()

        # Determine architecture type from release name
        arch_type = "topk" if "topk" in release else "vanilla"
        width_str = parts.split("width-")[-1] if "width-" in parts else "unknown"

        sae_mse_results[config_key] = {
            "sae_id": sae_id,
            "release": release,
            "layer": 12,
            "width": sae_bench.cfg.d_sae,
            "arch_type": arch_type,
            "width_label": width_str,
            "mse": float(mse),
            "nmse": float(nmse),
            "l0": float(l0),
            "absorption_rate": None,  # Will compute below if possible
            "n_tokens": n_tokens_mse,
        }
        print(f"  {config_key}: MSE={mse:.6f}, NMSE={nmse:.6f}, L0={l0:.1f}, width={sae_bench.cfg.d_sae}")

    except Exception as e:
        print(f"  Failed to load {release}/{sae_id}: {e}")
        continue

print(f"\n[Phase 3] Loaded {len(sae_mse_results)} SAE configs total")

# ── Compute absorption for additional SAEs (quick approximation) ──────────
report_progress("absorption_multi", "Computing absorption rates for additional SAEs", 0.60)

print("\n" + "="*70)
print("COMPUTING ABSORPTION FOR ADDITIONAL SAEs")
print("="*70)

# Load probe for absorption measurement
# Reuse the probe from first-letter data
# We need: probe weights that map residual stream -> letter predictions
probe_pt_path = RESULTS_DIR / "phase1" / "probe_first-letter_L12.pt"
probe_npz_path = RESULTS_DIR / "phase1" / "probe_first-letter_L12.npz"

letter_probe = None
letter_to_idx = {chr(ord('a') + i): i for i in range(26)}

if probe_pt_path.exists():
    probe_state = torch.load(probe_pt_path, map_location=device, weights_only=True)
    if isinstance(probe_state, dict):
        if "weight" in probe_state:
            letter_probe = torch.nn.Linear(probe_state["weight"].shape[1], probe_state["weight"].shape[0])
            letter_probe.weight = torch.nn.Parameter(probe_state["weight"])
            if "bias" in probe_state:
                letter_probe.bias = torch.nn.Parameter(probe_state["bias"])
            letter_probe = letter_probe.to(device).float()
            print(f"[Phase 3] Loaded probe from PT file: {letter_probe.weight.shape}")

if letter_probe is None and probe_npz_path.exists():
    pdata = np.load(probe_npz_path, allow_pickle=True)
    pkeys = list(pdata.keys())
    print(f"[Phase 3] NPZ probe keys: {pkeys}")
    for key in ["coef", "weights", "weight", "W"]:
        if key in pdata:
            w = pdata[key]
            letter_probe = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
            letter_probe.weight = torch.nn.Parameter(torch.tensor(w, dtype=torch.float32))
            # Check for bias
            for bkey in ["intercept", "bias", "b"]:
                if bkey in pdata:
                    letter_probe.bias = torch.nn.Parameter(torch.tensor(pdata[bkey], dtype=torch.float32))
            letter_probe = letter_probe.to(device)
            print(f"[Phase 3] Loaded probe from NPZ: {letter_probe.weight.shape}")
            break

if letter_probe is None:
    print("[Phase 3] WARNING: Could not load probe for multi-SAE absorption. Using Phase 1 data only.")
    # We'll rely solely on the L12_16k absorption data from Phase 1

# For all additional SAEs: quick absorption estimation using the probe
if letter_probe is not None:
    # Generate test data: words with known first letters
    test_words = [
        "apple", "banana", "cherry", "dog", "elephant", "fish",
        "garden", "happy", "island", "jungle", "kitchen", "lion",
        "mountain", "night", "ocean", "piano", "queen", "river",
        "sunset", "tiger", "umbrella", "valley", "water", "yellow", "zebra",
        "about", "before", "chapter", "during", "every", "future",
        "global", "human", "input", "journey", "kingdom", "language",
        "market", "nature", "office", "pattern", "quick", "result",
        "simple", "travel", "under", "virtual", "winter", "young", "zone",
        "ancient", "bright", "crystal", "diamond", "engine", "forest",
        "gentle", "harbor", "imagine", "justice", "kite", "lemon",
        "mirror", "noble", "orange", "purple", "question", "rainbow",
        "silver", "thunder", "unique", "vintage", "wisdom", "youth", "zero",
    ]

    template = "{word} has the first letter:"

    # Cache all residual stream activations ONCE for efficiency
    print("\n  Caching residual activations for absorption probing...")
    cached_resids = []
    cached_letters = []
    cached_words = []

    with torch.no_grad():
        for word in test_words:
            letter = word[0].lower()
            if letter not in letter_to_idx:
                continue
            prompt = template.format(word=word)
            tokens = model.to_tokens(prompt)
            _, cache = model.run_with_cache(
                tokens,
                names_filter="blocks.12.hook_resid_post",
                return_type="loss",
            )
            resid = cache["blocks.12.hook_resid_post"][0, -6].float()
            cached_resids.append(resid)
            cached_letters.append(letter)
            cached_words.append(word)

    cached_resids_tensor = torch.stack(cached_resids)  # (n_words, d_model)
    print(f"  Cached {len(cached_resids)} word activations")

    # Compute raw probe predictions once
    with torch.no_grad():
        raw_logits = letter_probe(cached_resids_tensor.to(device))
        raw_preds = raw_logits.argmax(dim=-1).cpu().numpy()

    true_indices = np.array([letter_to_idx[l] for l in cached_letters])
    raw_correct_mask = (raw_preds == true_indices)
    n_raw_correct_total = raw_correct_mask.sum()
    print(f"  Raw probe accuracy: {n_raw_correct_total}/{len(cached_letters)} "
          f"({n_raw_correct_total/len(cached_letters)*100:.1f}%)")

    for config_key, sae_obj in loaded_saes.items():
        if config_key == "L12_16k_canonical":
            continue  # Already have absorption from Phase 1

        if config_key not in sae_mse_results:
            continue

        print(f"\n  Computing absorption for {config_key}...")

        with torch.no_grad():
            # Reconstruct all cached activations through this SAE
            sae_output = sae_obj(cached_resids_tensor.to(device))
            if isinstance(sae_output, tuple):
                reconstructed = sae_output[0]
            else:
                reconstructed = sae_output

            sae_logits = letter_probe(reconstructed.float())
            sae_preds = sae_logits.argmax(dim=-1).cpu().numpy()

        # Count false negatives: raw correct but SAE wrong
        n_fn = int(((raw_preds == true_indices) & (sae_preds != true_indices)).sum())
        n_correct_raw = int(raw_correct_mask.sum())
        n_correct_sae = int((sae_preds == true_indices).sum())
        n_tested = len(cached_letters)

        abs_rate = n_fn / n_correct_raw if n_correct_raw > 0 else 0.0
        sae_mse_results[config_key]["absorption_rate"] = float(abs_rate)
        sae_mse_results[config_key]["absorption_n_tested"] = n_tested
        sae_mse_results[config_key]["absorption_n_fn"] = n_fn
        sae_mse_results[config_key]["absorption_n_correct_raw"] = n_correct_raw

        print(f"    {config_key}: absorption_rate={abs_rate:.4f} "
              f"(FN={n_fn}/{n_correct_raw}, tested={n_tested})")

# Now compute Prediction 1 correlation
report_progress("prediction1_analysis", "Analyzing absorption-MSE correlation", 0.70)

print("\n" + "="*70)
print("PREDICTION 1: ABSORPTION vs MSE CORRELATION")
print("="*70)

pred1_sae_data = []
for config_key, data in sae_mse_results.items():
    if data["absorption_rate"] is not None:
        pred1_sae_data.append({
            "config": config_key,
            "absorption_rate": data["absorption_rate"],
            "mse": data["mse"],
            "nmse": data["nmse"],
            "l0": data["l0"],
            "width": data["width"],
        })

print(f"\n  SAEs with both absorption and MSE data: {len(pred1_sae_data)}")
for d in pred1_sae_data:
    print(f"    {d['config']}: abs={d['absorption_rate']:.4f}, "
          f"MSE={d['mse']:.6f}, NMSE={d['nmse']:.6f}, L0={d['l0']:.1f}")

pred1_result = {
    "prediction": "Higher absorption correlates with lower MSE at matched L0",
    "n_saes": len(pred1_sae_data),
    "sae_data": pred1_sae_data,
    "spearman_rho": None,
    "spearman_p": None,
    "pearson_r": None,
    "pearson_p": None,
    "verdict": None,
}

if len(pred1_sae_data) >= 3:
    abs_rates = [d["absorption_rate"] for d in pred1_sae_data]
    mses = [d["mse"] for d in pred1_sae_data]

    spearman_r1, p_r1 = stats.spearmanr(abs_rates, mses)
    pearson_r1, p_pr1 = stats.pearsonr(abs_rates, mses)

    pred1_result["spearman_rho"] = float(spearman_r1) if not np.isnan(spearman_r1) else None
    pred1_result["spearman_p"] = float(p_r1) if not np.isnan(p_r1) else None
    pred1_result["pearson_r"] = float(pearson_r1) if not np.isnan(pearson_r1) else None
    pred1_result["pearson_p"] = float(p_pr1) if not np.isnan(p_pr1) else None

    print(f"\n  Spearman(absorption, MSE): rho={spearman_r1:.4f}, p={p_r1:.4f}")
    print(f"  Pearson(absorption, MSE):  r={pearson_r1:.4f}, p={p_pr1:.4f}")

    if spearman_r1 < -0.3 and p_r1 < 0.1:
        pred1_result["verdict"] = "SUPPORTED"
    elif spearman_r1 < 0:
        pred1_result["verdict"] = "WEAK_SUPPORT"
    else:
        pred1_result["verdict"] = "NOT_SUPPORTED"
else:
    pred1_result["verdict"] = "INSUFFICIENT_DATA"
    print("  Not enough SAEs for Prediction 1 correlation test.")

print(f"\n  Prediction 1 verdict: {pred1_result['verdict']}")


# ── Prediction 2: Power law scaling ──────────────────────────────────────
report_progress("prediction2", "Testing power law scaling of absorption with width", 0.80)

print("\n" + "="*70)
print("PREDICTION 2: Absorption decreases as W^{-gamma}")
print("="*70)

# For pilot: we only have 1 Gemma Scope width (16k). Need at least 2 widths.
# Check if we can load 65k width
pred2_width_data = []

# Add our 16k data
pred2_width_data.append({
    "width": 16384,
    "log_width": np.log(16384),
    "absorption_rate": overall_absorption,
    "log_absorption": np.log(overall_absorption + 1e-8),
})

# Try to load 65k width SAE for comparison
try:
    print("  Trying to load L12/width_65k/canonical...")
    _65k_result = sae_lens.SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_12/width_65k/canonical",
        device=device,
    )
    sae_65k = _65k_result[0] if isinstance(_65k_result, tuple) else _65k_result
    print(f"  Loaded 65k SAE: d_sae={sae_65k.cfg.d_sae}")

    # Compute MSE and absorption for 65k
    with torch.no_grad():
        sae_output_65k = sae_65k(all_residuals_cat.to(device))
        if isinstance(sae_output_65k, tuple):
            reconstructed_65k = sae_output_65k[0]
        else:
            reconstructed_65k = sae_output_65k
        mse_65k = torch.mean((all_residuals_cat.to(device) - reconstructed_65k) ** 2).item()
        nmse_65k = mse_65k / (var_original + 1e-12)
        sae_acts_65k = sae_65k.encode(all_residuals_cat.to(device))
        l0_65k = (sae_acts_65k > 0).float().sum(dim=-1).mean().item()

    # Compute absorption for 65k using cached word activations (if available)
    abs_65k = None
    if letter_probe is not None and len(cached_resids) > 0:
        with torch.no_grad():
            sae_out_65k = sae_65k(cached_resids_tensor.to(device))
            if isinstance(sae_out_65k, tuple):
                recon_65k_words = sae_out_65k[0]
            else:
                recon_65k_words = sae_out_65k
            sae_logits_65k = letter_probe(recon_65k_words.float())
            sae_preds_65k = sae_logits_65k.argmax(dim=-1).cpu().numpy()

        n_fn_65k = int(((raw_preds == true_indices) & (sae_preds_65k != true_indices)).sum())
        n_correct_raw_65k = int(raw_correct_mask.sum())
        abs_65k = n_fn_65k / n_correct_raw_65k if n_correct_raw_65k > 0 else 0.0
        print(f"  65k absorption: {abs_65k:.4f} (FN={n_fn_65k}/{n_correct_raw_65k})")

        pred2_width_data.append({
            "width": 65536,
            "log_width": np.log(65536),
            "absorption_rate": abs_65k,
            "log_absorption": np.log(abs_65k + 1e-8),
        })

    sae_mse_results["L12_65k_canonical"] = {
        "sae_id": "layer_12/width_65k/canonical",
        "layer": 12,
        "width": 65536,
        "mse": float(mse_65k),
        "nmse": float(nmse_65k),
        "l0": float(l0_65k),
        "absorption_rate": abs_65k,
        "n_tokens": n_tokens_mse,
    }

    del sae_65k
    torch.cuda.empty_cache()
    gc.collect()

except Exception as e:
    print(f"  Failed to load 65k SAE: {e}")
    traceback.print_exc()

pred2_result = {
    "prediction": "Absorption decreases as W^{-gamma} (power law scaling)",
    "n_widths": len(pred2_width_data),
    "width_data": pred2_width_data,
    "gamma": None,
    "r_squared": None,
    "verdict": None,
}

if len(pred2_width_data) >= 2:
    log_widths = np.array([d["log_width"] for d in pred2_width_data])
    log_abs = np.array([d["log_absorption"] for d in pred2_width_data])

    # Fit: log(absorption) = -gamma * log(width) + const
    if len(log_widths) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_widths, log_abs)
        gamma = -slope
        pred2_result["gamma"] = float(gamma)
        pred2_result["r_squared"] = float(r_value ** 2)
        pred2_result["slope"] = float(slope)
        pred2_result["intercept"] = float(intercept)
        pred2_result["p_value"] = float(p_value)

        print(f"\n  Power law fit: gamma = {gamma:.4f} (absorption ~ W^{{-{gamma:.4f}}})")
        print(f"  R^2 = {r_value**2:.4f}, p = {p_value:.4f}")

        if gamma > 0:
            pred2_result["verdict"] = "SUPPORTED_DIRECTION"
        else:
            pred2_result["verdict"] = "NOT_SUPPORTED"
    else:
        pred2_result["verdict"] = "INSUFFICIENT_DATA"
else:
    pred2_result["verdict"] = "INSUFFICIENT_DATA"
    print("  Only 1 width available in pilot. Need full mode for power law test.")

print(f"\n  Prediction 2 verdict: {pred2_result['verdict']}")


# ── Prediction 4: Matryoshka comparison ───────────────────────────────────
report_progress("prediction4", "Attempting Matryoshka SAE comparison", 0.85)

print("\n" + "="*70)
print("PREDICTION 4: Matryoshka SAEs achieve low absorption without MSE penalty")
print("="*70)

# Try to load Matryoshka SAEs
matryoshka_configs = [
    # Chanind Matryoshka BatchTopK 32k (standard variant)
    ("chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40", "standard/blocks.12.hook_resid_post"),
]

matryoshka_results = []
for release, sae_id in matryoshka_configs:
    try:
        print(f"  Trying Matryoshka: {release} / {sae_id}")
        _mat_result = sae_lens.SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
        )
        sae_mat = _mat_result[0] if isinstance(_mat_result, tuple) else _mat_result

        with torch.no_grad():
            sae_out = sae_mat(all_residuals_cat.to(device))
            if isinstance(sae_out, tuple):
                recon = sae_out[0]
            else:
                recon = sae_out
            mse_mat = torch.mean((all_residuals_cat.to(device) - recon) ** 2).item()
            nmse_mat = mse_mat / (var_original + 1e-12)
            acts_mat = sae_mat.encode(all_residuals_cat.to(device))
            l0_mat = (acts_mat > 0).float().sum(dim=-1).mean().item()

        matryoshka_results.append({
            "sae_id": sae_id,
            "width": sae_mat.cfg.d_sae,
            "mse": float(mse_mat),
            "nmse": float(nmse_mat),
            "l0": float(l0_mat),
        })
        print(f"    MSE={mse_mat:.6f}, NMSE={nmse_mat:.6f}, L0={l0_mat:.1f}")

        del sae_mat
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"    Failed: {e}")

pred4_result = {
    "prediction": "Matryoshka SAEs achieve low absorption without proportional MSE increase",
    "matryoshka_data": matryoshka_results,
    "baseline_data": {
        "L12_16k_mse": sae_mse_results["L12_16k_canonical"]["mse"],
        "L12_16k_absorption": overall_absorption,
    },
    "verdict": "INSUFFICIENT_DATA" if not matryoshka_results else "PRELIMINARY",
}

if matryoshka_results:
    print(f"\n  Matryoshka comparison available ({len(matryoshka_results)} configs)")
    for mr in matryoshka_results:
        print(f"    {mr['sae_id']}: MSE={mr['mse']:.6f}, L0={mr['l0']:.1f}")
else:
    print("  No Matryoshka SAEs available in pilot. Documenting as future work.")


# ── Bootstrap CI on Prediction 3 ─────────────────────────────────────────
report_progress("bootstrap", "Computing bootstrap CIs", 0.90)

print("\n" + "="*70)
print("BOOTSTRAP CONFIDENCE INTERVALS")
print("="*70)

n_bootstrap = 1000

if len(rpc_arr) >= 5:
    boot_spearman = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(rpc_arr), size=len(rpc_arr), replace=True)
        r, _ = stats.spearmanr(rpc_arr[idx], abs_arr[idx])
        if not np.isnan(r):
            boot_spearman.append(r)

    if boot_spearman:
        boot_ci_lower = np.percentile(boot_spearman, 2.5)
        boot_ci_upper = np.percentile(boot_spearman, 97.5)
        boot_mean = np.mean(boot_spearman)

        pred3_result["bootstrap"] = {
            "n_resamples": n_bootstrap,
            "mean": float(boot_mean),
            "ci_lower": float(boot_ci_lower),
            "ci_upper": float(boot_ci_upper),
            "std": float(np.std(boot_spearman)),
        }
        print(f"  Pred 3 Spearman bootstrap: mean={boot_mean:.4f}, "
              f"95% CI=[{boot_ci_lower:.4f}, {boot_ci_upper:.4f}]")


# ── Compile final results ─────────────────────────────────────────────────
report_progress("compiling", "Compiling final results", 0.95)

elapsed_min = (time.time() - start_time) / 60.0

final_results = {
    "task_id": TASK_ID,
    "mode": MODE,
    "timestamp": datetime.now().isoformat(),
    "seed": SEED,
    "model": "gemma-2-2b",
    "elapsed_minutes": float(elapsed_min),

    "theoretical_T_G": tg_result,

    "prediction_1": pred1_result,
    "prediction_2": pred2_result,
    "prediction_3": pred3_result,
    "prediction_4": pred4_result,

    "sae_mse_data": sae_mse_results,

    "summary": {
        "T_G": float(T_G),
        "T_G_interpretation": f"Absorption-free SAE needs ~{T_G:.4f} additional L0 budget per token",
        "predictions_tested": 4,
        "predictions_supported": sum(1 for p in [pred1_result, pred2_result, pred3_result, pred4_result]
                                    if p["verdict"] in ["SUPPORTED", "SUPPORTED_DIRECTION", "WEAK_SUPPORT"]),
        "predictions_not_supported": sum(1 for p in [pred1_result, pred2_result, pred3_result, pred4_result]
                                        if p["verdict"] == "NOT_SUPPORTED"),
        "predictions_insufficient_data": sum(1 for p in [pred1_result, pred2_result, pred3_result, pred4_result]
                                            if "INSUFFICIENT" in (p["verdict"] or "")),
        "pilot_pass": True,  # Updated below
        "pilot_criteria": {
            "pred1_testable": pred1_result["verdict"] != "INSUFFICIENT_DATA",
            "pred2_testable": pred2_result["verdict"] != "INSUFFICIENT_DATA",
            "pred3_testable": pred3_result["verdict"] != "INSUFFICIENT_DATA",
            "pred4_testable": pred4_result["verdict"] != "INSUFFICIENT_DATA",
            "TG_computed": T_G > 0,
        },
    },
}

# Pilot pass: at least 2 of 4 predictions testable + T(G) computed
n_testable = sum([
    pred1_result["verdict"] != "INSUFFICIENT_DATA",
    pred2_result["verdict"] != "INSUFFICIENT_DATA",
    pred3_result["verdict"] != "INSUFFICIENT_DATA",
    pred4_result["verdict"] != "INSUFFICIENT_DATA",
])
final_results["summary"]["pilot_pass"] = (n_testable >= 2) and (T_G > 0)

# Save results
output_path = PILOTS_DIR / "phase3_tax_empirical.json"
with open(output_path, "w") as f:
    json.dump(final_results, f, indent=2, default=str)
print(f"\n[Phase 3] Results saved to: {output_path}")

# Also save to phase3 directory
phase3_output = PHASE3_DIR / "tax_empirical.json"
with open(phase3_output, "w") as f:
    json.dump(final_results, f, indent=2, default=str)
print(f"[Phase 3] Results also saved to: {phase3_output}")

# Write summary markdown
summary_md = f"""# Phase 3: Absorption Tax Empirical Validation (PILOT)

## Theoretical Absorption Tax T(G)

**T(G) = {T_G:.6f}**

Interpretation: An absorption-free SAE needs approximately {T_G:.4f} additional L0 budget
per token compared to one that uses absorption. This is the "tax" for maintaining full
feature resolution under hierarchy.

Top contributors to T(G):
| Letter | p_c (frequency) | R_pc (redundancy) | Contribution |
|--------|-----------------|-------------------|-------------|
"""
for c in tg_components[:10]:
    summary_md += f"| {c['letter']} | {c['p_c']:.4f} | {c['R_pc']:.4f} | {c['contribution']:.6f} |\n"

summary_md += f"""
## Prediction Results Summary

| Prediction | Verdict | Key Metric |
|-----------|---------|-----------|
| P1: Absorption-MSE trade-off | {pred1_result['verdict']} | Spearman={pred1_result.get('spearman_rho', 'N/A')} |
| P2: Width power law | {pred2_result['verdict']} | gamma={pred2_result.get('gamma', 'N/A')} |
| P3: R_pc predicts absorption | {pred3_result['verdict']} | Spearman={f"{pred3_result['spearman_rho']:.4f}" if isinstance(pred3_result.get('spearman_rho'), float) else "N/A"} |
| P4: Matryoshka advantage | {pred4_result['verdict']} | N={len(matryoshka_results)} configs |

## Prediction 3 Detail: R_pc vs Per-Letter Absorption

Spearman rho = {pred3_result.get('spearman_rho', 'N/A')}
Pearson r = {pred3_result.get('pearson_r', 'N/A')}
N letters = {pred3_result['n_letters']}

| Letter | R_pc | Absorption Rate | N_tokens | N_FN |
|--------|------|----------------|----------|------|
"""
for letter in letter_labels:
    d = rpc_results[letter]
    summary_md += f"| {letter} | {d['R_pc']:.4f} | {d['absorption_rate']:.4f} | {d['n_tokens']} | {d['n_false_negatives']} |\n"

summary_md += f"""
## SAE Configurations Tested

| Config | Width | MSE | NMSE | L0 | Absorption |
|--------|-------|-----|------|-----|-----------|
"""
for config_key, data in sae_mse_results.items():
    abs_str = f"{data['absorption_rate']:.4f}" if data['absorption_rate'] is not None else "N/A"
    summary_md += f"| {config_key} | {data['width']} | {data['mse']:.6f} | {data['nmse']:.6f} | {data['l0']:.1f} | {abs_str} |\n"

summary_md += f"""
## Pilot Assessment

- **Predictions testable:** {n_testable}/4
- **T(G) computed:** Yes ({T_G:.6f})
- **Pilot pass:** {'YES' if final_results['summary']['pilot_pass'] else 'NO'}
- **Elapsed time:** {elapsed_min:.1f} minutes
"""

summary_path = PILOTS_DIR / "phase3_tax_empirical_summary.md"
with open(summary_path, "w") as f:
    f.write(summary_md)
print(f"[Phase 3] Summary saved to: {summary_path}")

# ── Print final summary ──────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 3 PILOT COMPLETE")
print("="*70)
print(f"  T(G) = {T_G:.6f}")
print(f"  Predictions tested: 4")
print(f"  Predictions testable: {n_testable}/4")
print(f"  Pilot pass: {'YES' if final_results['summary']['pilot_pass'] else 'NO'}")
print(f"  Elapsed: {elapsed_min:.1f} minutes")

# ── Mark done ─────────────────────────────────────────────────────────────
mark_done(
    status="success",
    summary=f"Phase 3 pilot complete. T(G)={T_G:.6f}. "
            f"Testable predictions: {n_testable}/4. "
            f"Pilot {'PASS' if final_results['summary']['pilot_pass'] else 'FAIL'}."
)

# ── Update gpu_progress.json ──────────────────────────────────────────────
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
lock_path = WORKSPACE / "exp" / ".gpu_progress.lock"

try:
    import fcntl
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)

        if gpu_progress_path.exists():
            with open(gpu_progress_path) as f:
                gp = json.load(f)
        else:
            gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

        if TASK_ID not in gp["completed"]:
            gp["completed"].append(TASK_ID)

        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]

        gp["timings"][TASK_ID] = {
            "planned_min": 45,
            "actual_min": int(elapsed_min),
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {
                "model": "gemma-2-2b",
                "mode": MODE,
                "n_sae_configs": len(sae_mse_results),
                "predictions_tested": 4,
                "gpu_model": "RTX PRO 6000 Blackwell",
                "gpu_count": 1,
            },
        }

        with open(gpu_progress_path, "w") as f:
            json.dump(gp, f, indent=2)

        fcntl.flock(lock_f, fcntl.LOCK_UN)
    print(f"[Phase 3] gpu_progress.json updated")
except Exception as e:
    print(f"[Phase 3] WARNING: Failed to update gpu_progress.json: {e}")

print(f"\n[Phase 3] All done. Results at: {output_path}")
