"""
Phase 3: Absorption Tax Qualitative Validation (PILOT mode)

Tests whether T(G) predicts cross-domain absorption RANKING, even if quantitative
predictions fail (rho=0.08, 0.16 from prior pilot).

Pipeline:
1. Compute T(G) for each hierarchy type using improved probes from Phase 1.1
2. T(G) = sum p_c * R_pc where R_pc = cos^2(probe_dir, decoder_dir)
3. Rank hierarchies by T(G) and compare to observed absorption ranking
4. Recompute R_pc for all 26 letters with improved probes + correlation with per-letter absorption
5. Bootstrap CIs on all ranking comparisons

Dependencies:
- phase1_probe_training_full (probe weights for all hierarchies x layers)
- phase1_absorption_firstletter (first-letter absorption rates per SAE config)
- phase1_absorption_crossdomain (cross-domain absorption rates)

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

# ── Configuration ──────────────────────────────────────────────────────────
SEED = 42
TASK_ID = "phase3_tax_qualitative"
MODE = "PILOT"

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PHASE3_DIR = RESULTS_DIR / "phase3"
PILOTS_DIR = RESULTS_DIR / "pilots"
PHASE1_DIR = RESULTS_DIR / "phase1"
CODE_DIR = WORKSPACE / "exp" / "code"

# Ensure output dirs exist
PHASE3_DIR.mkdir(parents=True, exist_ok=True)
PILOTS_DIR.mkdir(parents=True, exist_ok=True)

GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "6")

np.random.seed(SEED)

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

report_progress("init", "Loading dependencies and data", 0.0)

# ── Load dependencies ──────────────────────────────────────────────────────
try:
    import torch
    from scipy import stats
except ImportError as e:
    mark_done("failure", f"Missing dependency: {e}")
    sys.exit(1)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Phase 3 Tax Qualitative] Device: {device}")

# ── Load Phase 1 results ──────────────────────────────────────────────────
report_progress("loading_data", "Loading Phase 1 probe and absorption results", 0.05)

# Load probe training results (tells us best layer per hierarchy)
probe_training_path = PHASE1_DIR / "probe_training_full.json"
if not probe_training_path.exists():
    probe_training_path = PILOTS_DIR / "phase1_probe_training_full.json"
if not probe_training_path.exists():
    mark_done("failure", "Phase 1 probe training results not found")
    sys.exit(1)

with open(probe_training_path) as f:
    probe_data = json.load(f)

print(f"[Phase 3] Loaded probe training data: {len(probe_data['probes'])} probes")

# Load first-letter absorption results
fl_path = PILOTS_DIR / "phase1_absorption_firstletter.json"
if not fl_path.exists():
    mark_done("failure", "Phase 1 first-letter absorption results not found")
    sys.exit(1)

with open(fl_path) as f:
    fl_data = json.load(f)

print(f"[Phase 3] Loaded first-letter absorption data")

# Load cross-domain absorption results
cd_path = PILOTS_DIR / "phase1_absorption_crossdomain.json"
if not cd_path.exists():
    mark_done("failure", "Phase 1 cross-domain absorption results not found")
    sys.exit(1)

with open(cd_path) as f:
    cd_data = json.load(f)

print(f"[Phase 3] Loaded cross-domain absorption data: {cd_data['n_hierarchies_tested']} hierarchies")

# ── Extract observed absorption rates per hierarchy ───────────────────────
report_progress("extract_observed", "Extracting observed absorption rates", 0.10)

print("\n" + "="*70)
print("OBSERVED ABSORPTION RATES FROM PHASE 1")
print("="*70)

# We use L24_16k as the primary comparison point (all cross-domain data is at L24)
# For first-letter, we also need L24_16k

observed_absorption = {}

# First-letter at L24_16k (matching the cross-domain layer)
fl_results = fl_data.get("absorption_results", {})
for config_key in ["L24_16k", "L24_65k", "L12_16k", "L18_16k", "L6_16k"]:
    if config_key in fl_results and isinstance(fl_results[config_key], dict):
        rate = fl_results[config_key].get("absorption_rate", None)
        if rate is not None:
            observed_absorption[f"first-letter_{config_key}"] = rate
            print(f"  first-letter {config_key}: {rate:.4f}")

# Cross-domain hierarchies
cd_results = cd_data.get("crossdomain_results", {})
for hier_name, configs in cd_results.items():
    for config_key, data in configs.items():
        if isinstance(data, dict) and "absorption_rate" in data and not data.get("skipped", False):
            observed_absorption[f"{hier_name}_{config_key}"] = data["absorption_rate"]
            print(f"  {hier_name} {config_key}: {data['absorption_rate']:.4f}")

# Aggregate per hierarchy at L24_16k (primary comparison)
hierarchy_absorption_primary = {}
if "first-letter_L24_16k" in observed_absorption:
    hierarchy_absorption_primary["first-letter"] = observed_absorption["first-letter_L24_16k"]
for hier in ["city-continent", "city-country", "city-language"]:
    key = f"{hier}_L24_16k"
    if key in observed_absorption:
        hierarchy_absorption_primary[hier] = observed_absorption[key]

print(f"\n  Primary comparison (L24_16k):")
for h, r in sorted(hierarchy_absorption_primary.items(), key=lambda x: x[1], reverse=True):
    print(f"    {h}: {r:.4f}")

# Also aggregate across all SAE configs (mean absorption per hierarchy)
hierarchy_absorption_mean = {}
for hier in ["first-letter", "city-continent", "city-country", "city-language"]:
    rates = [v for k, v in observed_absorption.items() if k.startswith(f"{hier}_")]
    if rates:
        hierarchy_absorption_mean[hier] = np.mean(rates)
        print(f"  {hier} mean across configs: {np.mean(rates):.4f} (n={len(rates)})")

# ── Load model and SAE for T(G) computation ──────────────────────────────
report_progress("loading_model", "Loading Gemma 2B and SAE for T(G) computation", 0.15)

print("\n" + "="*70)
print("LOADING MODEL AND SAE")
print("="*70)

import sae_lens
import transformer_lens
from transformers import AutoModelForCausalLM, AutoTokenizer

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

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
print(f"[Phase 3] Gemma 2B loaded, d_model = {model.cfg.d_model}")

# Load SAE at layer 24 (primary layer for cross-domain comparisons)
report_progress("loading_sae", "Loading Gemma Scope SAE L24/16k", 0.20)

_sae_result = sae_lens.SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_24/width_16k/canonical",
    device=device,
)
sae_24_16k = _sae_result[0] if isinstance(_sae_result, tuple) else _sae_result
W_dec_24 = sae_24_16k.W_dec.detach().float()  # (d_sae, d_model)
print(f"[Phase 3] SAE L24/16k loaded. Dict size: {sae_24_16k.cfg.d_sae}, decoder shape: {W_dec_24.shape}")

# Also load L12/16k for the per-letter R_pc comparison (matching prior pilot)
_sae_result_12 = sae_lens.SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    device=device,
)
sae_12_16k = _sae_result_12[0] if isinstance(_sae_result_12, tuple) else _sae_result_12
W_dec_12 = sae_12_16k.W_dec.detach().float()
print(f"[Phase 3] SAE L12/16k loaded. Dict size: {sae_12_16k.cfg.d_sae}")


# ── Part 1: Compute T(G) per hierarchy using improved probes ─────────────
report_progress("tg_computation", "Computing T(G) per hierarchy", 0.25)

print("\n" + "="*70)
print("PART 1: T(G) COMPUTATION PER HIERARCHY")
print("T(G) = sum_{(p,c)} p_c * R_pc, where R_pc = cos^2(probe_dir, decoder_dir)")
print("="*70)

# For each hierarchy, we need:
#   - probe_weights (from trained probe at best layer)
#   - SAE decoder weights (at same layer)
#   - frequency weights p_c for each child class
#   - R_pc = cos^2 between probe direction for class c and the SAE decoder direction
#     of the most aligned feature

tg_per_hierarchy = {}

# ── T(G) for first-letter hierarchy ─────────────────────────────────────
print("\n--- First-Letter Hierarchy ---")

# Load first-letter probe at multiple layers
for layer in [6, 12, 18, 24]:
    # Try sae_spelling probes (best quality for first-letter)
    probe_path = PHASE1_DIR / f"probe_first-letter_L{layer}_sae_spelling.pt"
    sklearn_path = PHASE1_DIR / f"probe_first-letter_L{layer}_sklearn.npz"
    sklearn_icl_path = PHASE1_DIR / f"probe_firstletter_L{layer}_sklearn_icl.npz"

    # Try loading probe weights
    probe_weights_fl = None

    if sklearn_path.exists():
        try:
            pdata = np.load(sklearn_path, allow_pickle=True)
            for key in ["coef", "weights", "weight", "W"]:
                if key in pdata:
                    probe_weights_fl = pdata[key]
                    print(f"  Loaded sklearn probe L{layer}: shape={probe_weights_fl.shape}")
                    break
        except Exception as e:
            print(f"  Failed to load sklearn probe L{layer}: {e}")

    if probe_weights_fl is None and sklearn_icl_path.exists():
        try:
            pdata = np.load(sklearn_icl_path, allow_pickle=True)
            for key in ["coef", "weights", "weight", "W"]:
                if key in pdata:
                    probe_weights_fl = pdata[key]
                    print(f"  Loaded sklearn ICL probe L{layer}: shape={probe_weights_fl.shape}")
                    break
        except Exception as e:
            print(f"  Failed to load sklearn ICL probe L{layer}: {e}")

    if probe_weights_fl is None and probe_path.exists():
        try:
            state = torch.load(probe_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "weight" in state:
                probe_weights_fl = state["weight"].numpy()
                print(f"  Loaded sae_spelling probe L{layer}: shape={probe_weights_fl.shape}")
            elif isinstance(state, torch.Tensor):
                probe_weights_fl = state.numpy()
                print(f"  Loaded sae_spelling probe L{layer}: shape={probe_weights_fl.shape}")
        except Exception as e:
            print(f"  Failed to load sae_spelling probe L{layer}: {e}")

    if probe_weights_fl is None:
        print(f"  WARNING: No probe weights found for first-letter L{layer}")
        continue

    # Load SAE at matching layer
    sae_id_str = f"layer_{layer}/width_16k/canonical"
    try:
        if layer == 24:
            W_dec_layer = W_dec_24
        elif layer == 12:
            W_dec_layer = W_dec_12
        else:
            _sae_r = sae_lens.SAE.from_pretrained(
                release="gemma-scope-2b-pt-res-canonical",
                sae_id=sae_id_str,
                device=device,
            )
            sae_temp = _sae_r[0] if isinstance(_sae_r, tuple) else _sae_r
            W_dec_layer = sae_temp.W_dec.detach().float()
            del sae_temp
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        print(f"  Failed to load SAE at L{layer}: {e}")
        continue

    # Compute R_pc for each letter
    # R_pc = cos^2(probe_dir_c, best_decoder_dir)
    # The "best decoder" is the feature whose decoder direction is most aligned with
    # the probe direction for that class.
    n_classes = probe_weights_fl.shape[0]

    # Get per-letter frequency from absorption data
    fl_config_key = f"L{layer}_16k"
    fl_per_letter = fl_results.get(fl_config_key, {}).get("per_letter", {})

    # If no per-letter data at this layer, compute frequency from total tokens
    total_tokens_fl = sum(
        fl_per_letter.get(chr(ord('a') + i), {}).get("total", 0)
        for i in range(26)
    )
    if total_tokens_fl == 0:
        total_tokens_fl = 1  # avoid div-by-zero

    rpc_per_letter = {}
    tg_components = []

    for c_idx in range(min(n_classes, 26)):
        letter = chr(ord('a') + c_idx)
        probe_dir = probe_weights_fl[c_idx]
        probe_dir_norm = probe_dir / (np.linalg.norm(probe_dir) + 1e-12)

        # Convert to torch for efficient cosine sim with all decoder directions
        probe_dir_t = torch.tensor(probe_dir_norm, dtype=torch.float32, device=W_dec_layer.device)

        # Compute cosine similarity with ALL decoder directions
        # W_dec_layer: (d_sae, d_model)
        dec_norms = W_dec_layer.norm(dim=1, keepdim=True).clamp(min=1e-12)
        dec_normalized = W_dec_layer / dec_norms
        cos_sims = (dec_normalized @ probe_dir_t).cpu().numpy()

        # R_pc = max_feature cos^2(probe, decoder)
        best_fid = int(np.argmax(np.abs(cos_sims)))
        best_cos = float(cos_sims[best_fid])
        R_pc = best_cos ** 2

        # Get frequency weight
        letter_data = fl_per_letter.get(letter, {})
        n_tokens = letter_data.get("total", 0)
        p_c = n_tokens / total_tokens_fl if total_tokens_fl > 0 else 1.0 / 26.0

        # Get observed absorption at this letter (if available)
        absorption_rate = letter_data.get("absorption_rate", None)

        rpc_per_letter[letter] = {
            "R_pc": float(R_pc),
            "best_cos": float(best_cos),
            "best_feature_idx": best_fid,
            "p_c": float(p_c),
            "n_tokens": n_tokens,
            "absorption_rate": absorption_rate,
        }

        contribution = p_c * R_pc
        tg_components.append({
            "class_label": letter,
            "p_c": float(p_c),
            "R_pc": float(R_pc),
            "contribution": float(contribution),
        })

    T_G_fl = sum(c["contribution"] for c in tg_components)

    # Compute R_pc vs absorption correlation
    rpc_vals = []
    abs_vals = []
    for letter, data in rpc_per_letter.items():
        if data["absorption_rate"] is not None and data["n_tokens"] >= 3:
            rpc_vals.append(data["R_pc"])
            abs_vals.append(data["absorption_rate"])

    rpc_abs_corr = None
    if len(rpc_vals) >= 5:
        rho_rpc, p_rpc = stats.spearmanr(rpc_vals, abs_vals)
        rpc_abs_corr = {
            "spearman_rho": float(rho_rpc) if not np.isnan(rho_rpc) else None,
            "spearman_p": float(p_rpc) if not np.isnan(p_rpc) else None,
            "n_letters": len(rpc_vals),
        }
        print(f"  L{layer} T(G) = {T_G_fl:.6f}, R_pc-absorption rho={rho_rpc:.4f} (p={p_rpc:.4f}, n={len(rpc_vals)})")
    else:
        print(f"  L{layer} T(G) = {T_G_fl:.6f}, insufficient data for R_pc-absorption correlation")

    tg_per_hierarchy[f"first-letter_L{layer}"] = {
        "hierarchy": "first-letter",
        "layer": layer,
        "T_G": float(T_G_fl),
        "n_classes": n_classes,
        "n_components": len(tg_components),
        "components": sorted(tg_components, key=lambda x: x["contribution"], reverse=True),
        "per_class_rpc": rpc_per_letter,
        "rpc_absorption_correlation": rpc_abs_corr,
    }

# ── T(G) for cross-domain hierarchies ───────────────────────────────────
report_progress("tg_crossdomain", "Computing T(G) for cross-domain hierarchies", 0.40)

cross_domain_hierarchies = {
    "city-continent": {"n_classes": 6},
    "city-country": {"n_classes": 80},
    "city-language": {"n_classes": 50},
}

for hier_name, hier_info in cross_domain_hierarchies.items():
    print(f"\n--- {hier_name} Hierarchy ---")

    # Best layer from probe training is always 24 for all RAVEL hierarchies
    best_layer = probe_data["recommended_layers"].get(hier_name, {}).get("layer", 24)

    # Load probe weights
    probe_path = PHASE1_DIR / f"probe_{hier_name}_L{best_layer}.npz"
    probe_weights_cd = None

    if probe_path.exists():
        try:
            pdata = np.load(probe_path, allow_pickle=True)
            for key in ["coef", "weights", "weight", "W"]:
                if key in pdata:
                    probe_weights_cd = pdata[key]
                    print(f"  Loaded probe L{best_layer}: shape={probe_weights_cd.shape}")
                    break
            # Also try to load class labels
            class_labels = None
            for key in ["classes", "class_labels", "labels"]:
                if key in pdata:
                    class_labels = list(pdata[key])
                    break
            if class_labels is None:
                class_labels = [f"class_{i}" for i in range(probe_weights_cd.shape[0])]
        except Exception as e:
            print(f"  Failed to load probe: {e}")
            traceback.print_exc()

    if probe_weights_cd is None:
        print(f"  WARNING: No probe found for {hier_name} L{best_layer}. Skipping.")
        continue

    # Use layer-24 SAE decoder (all cross-domain probes are at L24)
    W_dec_layer = W_dec_24

    # Get absorption data per class
    cd_hier_results = cd_results.get(hier_name, {})
    # Use L24_16k as primary
    cd_config = cd_hier_results.get("L24_16k", {})
    per_class_absorption = cd_config.get("per_class", {})

    # Compute total tokens for frequency
    total_tokens_cd = sum(
        cls_data.get("total", 0)
        for cls_data in per_class_absorption.values()
    )
    if total_tokens_cd == 0:
        total_tokens_cd = 1

    n_classes = probe_weights_cd.shape[0]
    tg_components = []
    rpc_per_class = {}

    for c_idx in range(n_classes):
        c_label = class_labels[c_idx] if c_idx < len(class_labels) else f"class_{c_idx}"

        probe_dir = probe_weights_cd[c_idx]
        probe_dir_norm = probe_dir / (np.linalg.norm(probe_dir) + 1e-12)

        probe_dir_t = torch.tensor(probe_dir_norm, dtype=torch.float32, device=W_dec_layer.device)

        dec_norms = W_dec_layer.norm(dim=1, keepdim=True).clamp(min=1e-12)
        dec_normalized = W_dec_layer / dec_norms
        cos_sims = (dec_normalized @ probe_dir_t).cpu().numpy()

        best_fid = int(np.argmax(np.abs(cos_sims)))
        best_cos = float(cos_sims[best_fid])
        R_pc = best_cos ** 2

        # Frequency weight
        cls_data = per_class_absorption.get(c_label, {})
        n_tokens = cls_data.get("total", 0)
        p_c = n_tokens / total_tokens_cd if total_tokens_cd > 0 else 1.0 / n_classes

        # Observed absorption for this class
        abs_rate = cls_data.get("absorption_rate", None)

        rpc_per_class[c_label] = {
            "R_pc": float(R_pc),
            "best_cos": float(best_cos),
            "best_feature_idx": best_fid,
            "p_c": float(p_c),
            "n_tokens": n_tokens,
            "absorption_rate": abs_rate,
        }

        contribution = p_c * R_pc
        tg_components.append({
            "class_label": c_label,
            "p_c": float(p_c),
            "R_pc": float(R_pc),
            "contribution": float(contribution),
        })

    T_G = sum(c["contribution"] for c in tg_components)

    # Per-class R_pc vs absorption correlation
    rpc_vals = []
    abs_vals = []
    for cls_label, data in rpc_per_class.items():
        if data["absorption_rate"] is not None and data["n_tokens"] >= 3:
            rpc_vals.append(data["R_pc"])
            abs_vals.append(data["absorption_rate"])

    rpc_abs_corr = None
    if len(rpc_vals) >= 5:
        rho_rpc, p_rpc = stats.spearmanr(rpc_vals, abs_vals)
        rpc_abs_corr = {
            "spearman_rho": float(rho_rpc) if not np.isnan(rho_rpc) else None,
            "spearman_p": float(p_rpc) if not np.isnan(p_rpc) else None,
            "n_classes": len(rpc_vals),
        }
        print(f"  T(G) = {T_G:.6f}, per-class R_pc-absorption rho={rho_rpc:.4f} (p={p_rpc:.4f}, n={len(rpc_vals)})")
    else:
        print(f"  T(G) = {T_G:.6f}, insufficient data for per-class R_pc-absorption correlation (n={len(rpc_vals)})")

    tg_per_hierarchy[f"{hier_name}_L{best_layer}"] = {
        "hierarchy": hier_name,
        "layer": best_layer,
        "T_G": float(T_G),
        "n_classes": n_classes,
        "n_components": len(tg_components),
        "components": sorted(tg_components, key=lambda x: x["contribution"], reverse=True)[:20],  # Top 20
        "per_class_rpc_sample": {k: v for k, v in list(rpc_per_class.items())[:20]},  # Top 20
        "rpc_absorption_correlation": rpc_abs_corr,
    }

# ── Part 2: Ranking comparison ───────────────────────────────────────────
report_progress("ranking", "Comparing T(G) ranking vs observed absorption ranking", 0.60)

print("\n" + "="*70)
print("PART 2: RANKING COMPARISON")
print("T(G) ranking vs observed absorption ranking")
print("="*70)

# Build ranking table: hierarchy -> T(G), observed absorption
# Use L24 as the primary comparison layer
ranking_data = []

for hier in ["first-letter", "city-continent", "city-country", "city-language"]:
    key = f"{hier}_L24"
    tg_val = tg_per_hierarchy.get(key, {}).get("T_G", None)
    obs_val = hierarchy_absorption_primary.get(hier, None)
    obs_mean = hierarchy_absorption_mean.get(hier, None)

    if tg_val is not None:
        ranking_data.append({
            "hierarchy": hier,
            "T_G": tg_val,
            "observed_absorption_L24_16k": obs_val,
            "observed_absorption_mean": obs_mean,
        })

print(f"\n  {'Hierarchy':<20} {'T(G)':<12} {'Obs Abs (L24_16k)':<20} {'Obs Abs (mean)':<15}")
print(f"  {'-'*65}")
for r in ranking_data:
    obs_str = f"{r['observed_absorption_L24_16k']:.4f}" if r['observed_absorption_L24_16k'] is not None else "N/A"
    obs_mean_str = f"{r['observed_absorption_mean']:.4f}" if r['observed_absorption_mean'] is not None else "N/A"
    print(f"  {r['hierarchy']:<20} {r['T_G']:<12.6f} {obs_str:<20} {obs_mean_str:<15}")

# Compute rankings
tg_values = [(r["hierarchy"], r["T_G"]) for r in ranking_data]
obs_values_primary = [(r["hierarchy"], r["observed_absorption_L24_16k"])
                      for r in ranking_data if r["observed_absorption_L24_16k"] is not None]
obs_values_mean = [(r["hierarchy"], r["observed_absorption_mean"])
                   for r in ranking_data if r["observed_absorption_mean"] is not None]

# Sort by value to get ranking
tg_ranking = sorted(tg_values, key=lambda x: x[1], reverse=True)
obs_ranking_primary = sorted(obs_values_primary, key=lambda x: x[1], reverse=True)
obs_ranking_mean = sorted(obs_values_mean, key=lambda x: x[1], reverse=True)

print(f"\n  T(G) ranking (highest first):")
for i, (h, v) in enumerate(tg_ranking):
    print(f"    {i+1}. {h}: {v:.6f}")

print(f"\n  Observed absorption ranking (L24_16k, highest first):")
for i, (h, v) in enumerate(obs_ranking_primary):
    print(f"    {i+1}. {h}: {v:.4f}")

print(f"\n  Observed absorption ranking (mean across SAEs, highest first):")
for i, (h, v) in enumerate(obs_ranking_mean):
    print(f"    {i+1}. {h}: {v:.4f}")

# Compute rank correlation (Spearman)
ranking_correlation = {}

# T(G) vs primary obs absorption
if len(tg_values) >= 3:
    # Match hierarchies
    matched_tg = []
    matched_obs = []
    matched_hier = []
    for h, tg in tg_values:
        for h2, obs in obs_values_primary:
            if h == h2:
                matched_tg.append(tg)
                matched_obs.append(obs)
                matched_hier.append(h)
                break

    if len(matched_tg) >= 3:
        rho_rank, p_rank = stats.spearmanr(matched_tg, matched_obs)
        tau_rank, p_tau = stats.kendalltau(matched_tg, matched_obs)
        ranking_correlation["vs_L24_16k"] = {
            "spearman_rho": float(rho_rank) if not np.isnan(rho_rank) else None,
            "spearman_p": float(p_rank) if not np.isnan(p_rank) else None,
            "kendall_tau": float(tau_rank) if not np.isnan(tau_rank) else None,
            "kendall_p": float(p_tau) if not np.isnan(p_tau) else None,
            "n_hierarchies": len(matched_tg),
            "matched_hierarchies": matched_hier,
            "T_G_values": [float(v) for v in matched_tg],
            "obs_values": [float(v) for v in matched_obs],
        }
        print(f"\n  Rank correlation (T(G) vs L24_16k absorption):")
        print(f"    Spearman rho = {rho_rank:.4f}, p = {p_rank:.4f}")
        print(f"    Kendall tau  = {tau_rank:.4f}, p = {p_tau:.4f}")

        # Check if direction is correct: higher T(G) -> higher absorption
        tg_rank_order = [h for h, _ in sorted(zip(matched_hier, matched_tg), key=lambda x: x[1], reverse=True)]
        obs_rank_order = [h for h, _ in sorted(zip(matched_hier, matched_obs), key=lambda x: x[1], reverse=True)]
        ranking_match = tg_rank_order == obs_rank_order
        print(f"    Exact ranking match: {ranking_match}")
        print(f"    T(G) order:  {tg_rank_order}")
        print(f"    Obs order:   {obs_rank_order}")
        ranking_correlation["vs_L24_16k"]["exact_match"] = ranking_match

# T(G) vs mean obs absorption
if len(tg_values) >= 3:
    matched_tg_mean = []
    matched_obs_mean = []
    matched_hier_mean = []
    for h, tg in tg_values:
        for h2, obs in obs_values_mean:
            if h == h2:
                matched_tg_mean.append(tg)
                matched_obs_mean.append(obs)
                matched_hier_mean.append(h)
                break

    if len(matched_tg_mean) >= 3:
        rho_rank_m, p_rank_m = stats.spearmanr(matched_tg_mean, matched_obs_mean)
        tau_rank_m, p_tau_m = stats.kendalltau(matched_tg_mean, matched_obs_mean)
        ranking_correlation["vs_mean_absorption"] = {
            "spearman_rho": float(rho_rank_m) if not np.isnan(rho_rank_m) else None,
            "spearman_p": float(p_rank_m) if not np.isnan(p_rank_m) else None,
            "kendall_tau": float(tau_rank_m) if not np.isnan(tau_rank_m) else None,
            "kendall_p": float(p_tau_m) if not np.isnan(p_tau_m) else None,
            "n_hierarchies": len(matched_tg_mean),
        }
        print(f"\n  Rank correlation (T(G) vs mean absorption):")
        print(f"    Spearman rho = {rho_rank_m:.4f}, p = {p_rank_m:.4f}")
        print(f"    Kendall tau  = {tau_rank_m:.4f}, p = {p_tau_m:.4f}")

# ── Part 3: Bootstrap CI on ranking correlation ──────────────────────────
report_progress("bootstrap", "Bootstrap CIs on ranking correlations", 0.70)

print("\n" + "="*70)
print("PART 3: BOOTSTRAP CONFIDENCE INTERVALS")
print("="*70)

n_bootstrap = 10000

# Bootstrap the ranking correlation by resampling the per-hierarchy (T_G, obs) pairs
if len(matched_tg) >= 3:
    boot_rhos = []
    boot_taus = []
    matched_tg_arr = np.array(matched_tg)
    matched_obs_arr = np.array(matched_obs)

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(len(matched_tg_arr), size=len(matched_tg_arr), replace=True)
        # Need at least 2 unique indices for meaningful correlation
        if len(set(idx)) < 2:
            continue
        tg_boot = matched_tg_arr[idx]
        obs_boot = matched_obs_arr[idx]
        try:
            r, _ = stats.spearmanr(tg_boot, obs_boot)
            t, _ = stats.kendalltau(tg_boot, obs_boot)
            if not np.isnan(r):
                boot_rhos.append(r)
            if not np.isnan(t):
                boot_taus.append(t)
        except Exception:
            pass

    if boot_rhos:
        boot_rho_ci = (float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5)))
        boot_rho_mean = float(np.mean(boot_rhos))
        print(f"  Spearman rho bootstrap: mean={boot_rho_mean:.4f}, "
              f"95% CI=[{boot_rho_ci[0]:.4f}, {boot_rho_ci[1]:.4f}]")
    else:
        boot_rho_ci = (None, None)
        boot_rho_mean = None

    if boot_taus:
        boot_tau_ci = (float(np.percentile(boot_taus, 2.5)), float(np.percentile(boot_taus, 97.5)))
        boot_tau_mean = float(np.mean(boot_taus))
        print(f"  Kendall tau bootstrap:  mean={boot_tau_mean:.4f}, "
              f"95% CI=[{boot_tau_ci[0]:.4f}, {boot_tau_ci[1]:.4f}]")
    else:
        boot_tau_ci = (None, None)
        boot_tau_mean = None

    ranking_correlation["bootstrap"] = {
        "n_resamples": n_bootstrap,
        "spearman_mean": boot_rho_mean,
        "spearman_ci_lower": boot_rho_ci[0],
        "spearman_ci_upper": boot_rho_ci[1],
        "kendall_mean": boot_tau_mean,
        "kendall_ci_lower": boot_tau_ci[0],
        "kendall_ci_upper": boot_tau_ci[1],
    }
else:
    print("  Insufficient data for bootstrap")

# ── Part 4: Per-letter R_pc analysis with improved probes ─────────────────
report_progress("per_letter_rpc", "Recomputing per-letter R_pc with improved probes", 0.80)

print("\n" + "="*70)
print("PART 4: PER-LETTER R_pc WITH IMPROVED PROBES")
print("="*70)

# Use best first-letter probe (L24 sklearn) for R_pc computation
# Compare with absorption rates from L12_16k (pilot comparison) and L24_16k
best_fl_key = "first-letter_L24"
best_fl_rpc = tg_per_hierarchy.get(best_fl_key, {}).get("per_class_rpc", {})

# Also compute at L12 for comparison with prior pilot
l12_fl_key = "first-letter_L12"
l12_fl_rpc = tg_per_hierarchy.get(l12_fl_key, {}).get("per_class_rpc", {})

# Compute correlation between R_pc (from improved L24 probe) and absorption at different SAE configs
per_letter_analysis = {}

for sae_config_key in ["L6_16k", "L12_16k", "L18_16k", "L24_16k", "L6_65k", "L12_65k", "L18_65k", "L24_65k"]:
    fl_config_data = fl_results.get(sae_config_key, {})
    if not isinstance(fl_config_data, dict) or "per_letter" not in fl_config_data:
        continue

    per_letter_fl = fl_config_data["per_letter"]

    # Use R_pc from the probe at the SAE's layer
    sae_layer = int(sae_config_key.split("_")[0].replace("L", ""))
    rpc_key = f"first-letter_L{sae_layer}"
    rpc_data = tg_per_hierarchy.get(rpc_key, {}).get("per_class_rpc", {})

    if not rpc_data:
        continue

    rpc_vals = []
    abs_vals = []
    letters_used = []

    for letter in sorted(rpc_data.keys()):
        letter_abs_data = per_letter_fl.get(letter, {})
        n_tokens = letter_abs_data.get("total", 0)
        abs_rate = letter_abs_data.get("absorption_rate", None)

        if abs_rate is not None and n_tokens >= 3:
            rpc_vals.append(rpc_data[letter]["R_pc"])
            abs_vals.append(abs_rate)
            letters_used.append(letter)

    if len(rpc_vals) >= 5:
        rho, p_val = stats.spearmanr(rpc_vals, abs_vals)
        pearson_r, pearson_p = stats.pearsonr(rpc_vals, abs_vals)

        per_letter_analysis[sae_config_key] = {
            "sae_config": sae_config_key,
            "probe_layer": sae_layer,
            "n_letters": len(rpc_vals),
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "spearman_p": float(p_val) if not np.isnan(p_val) else None,
            "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else None,
            "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else None,
            "verdict": None,
        }

        if not np.isnan(rho):
            if abs(rho) >= 0.3 and p_val < 0.05:
                per_letter_analysis[sae_config_key]["verdict"] = "SUPPORTED"
            elif abs(rho) >= 0.2:
                per_letter_analysis[sae_config_key]["verdict"] = "WEAK_SUPPORT"
            else:
                per_letter_analysis[sae_config_key]["verdict"] = "NOT_SUPPORTED"

        print(f"  {sae_config_key} (probe L{sae_layer}): "
              f"rho={rho:.4f} (p={p_val:.4f}), r={pearson_r:.4f} (p={pearson_p:.4f}), "
              f"n={len(rpc_vals)}, verdict={per_letter_analysis[sae_config_key]['verdict']}")

# ── Part 5: Qualitative assessment ───────────────────────────────────────
report_progress("qualitative", "Generating qualitative assessment", 0.90)

print("\n" + "="*70)
print("PART 5: QUALITATIVE ASSESSMENT")
print("="*70)

# Does T(G) predict the absorption DIRECTION?
# - Higher T(G) -> higher absorption?
# - Does the hierarchy with highest T(G) have highest absorption?

qualitative_verdict = "INCONCLUSIVE"
qualitative_details = []

# Check if T(G) ranking matches observed ranking direction
if len(ranking_data) >= 3:
    # Get the actual values for comparison
    tg_sorted = sorted(ranking_data, key=lambda x: x["T_G"], reverse=True)
    obs_sorted = sorted(
        [r for r in ranking_data if r["observed_absorption_L24_16k"] is not None],
        key=lambda x: x["observed_absorption_L24_16k"],
        reverse=True
    )

    if len(obs_sorted) >= 3:
        # Check top-1 match
        top1_match = tg_sorted[0]["hierarchy"] == obs_sorted[0]["hierarchy"]
        # Check bottom-1 match
        bottom1_match = tg_sorted[-1]["hierarchy"] == obs_sorted[-1]["hierarchy"]

        qualitative_details.append(f"Top hierarchy: T(G)={tg_sorted[0]['hierarchy']}, Obs={obs_sorted[0]['hierarchy']} (match={top1_match})")
        qualitative_details.append(f"Bottom hierarchy: T(G)={tg_sorted[-1]['hierarchy']}, Obs={obs_sorted[-1]['hierarchy']} (match={bottom1_match})")

        # Count pairwise ordering matches
        n_pairs = 0
        n_concordant = 0
        for i in range(len(ranking_data)):
            for j in range(i + 1, len(ranking_data)):
                h1 = ranking_data[i]["hierarchy"]
                h2 = ranking_data[j]["hierarchy"]
                tg1 = ranking_data[i]["T_G"]
                tg2 = ranking_data[j]["T_G"]
                obs1 = ranking_data[i].get("observed_absorption_L24_16k")
                obs2 = ranking_data[j].get("observed_absorption_L24_16k")
                if obs1 is not None and obs2 is not None:
                    n_pairs += 1
                    if (tg1 - tg2) * (obs1 - obs2) > 0:
                        n_concordant += 1

        concordance_rate = n_concordant / n_pairs if n_pairs > 0 else 0
        qualitative_details.append(f"Pairwise concordance: {n_concordant}/{n_pairs} = {concordance_rate:.2%}")

        if concordance_rate >= 0.8:
            qualitative_verdict = "QUALITATIVE_SUPPORT"
        elif concordance_rate >= 0.5:
            qualitative_verdict = "PARTIAL_SUPPORT"
        elif concordance_rate < 0.5:
            qualitative_verdict = "NOT_SUPPORTED"

        # Compute overall T(G) direction prediction
        # Does first-letter have lowest T(G) AND lowest absorption? (consistent with pilot finding)
        fl_tg_rank = next((i for i, r in enumerate(tg_sorted) if r["hierarchy"] == "first-letter"), None)
        fl_obs_rank = next((i for i, r in enumerate(obs_sorted) if r["hierarchy"] == "first-letter"), None)
        if fl_tg_rank is not None and fl_obs_rank is not None:
            qualitative_details.append(
                f"First-letter: T(G) rank={fl_tg_rank+1}/{len(tg_sorted)}, "
                f"Obs rank={fl_obs_rank+1}/{len(obs_sorted)}"
            )

print(f"\n  Qualitative verdict: {qualitative_verdict}")
for d in qualitative_details:
    print(f"    {d}")

# ── Compile final results ─────────────────────────────────────────────────
report_progress("compiling", "Compiling final results", 0.95)

elapsed_min = (time.time() - start_time) / 60.0

# Determine overall pilot pass
# Pass criteria: T(G) computed for >= 3 hierarchy types. Ranking comparison with observed absorption.
# R_pc for >= 20 letters. Any result informative for appendix.
n_hierarchies_with_tg = len(tg_per_hierarchy)
n_letters_rpc = len(best_fl_rpc) if best_fl_rpc else 0

pilot_pass = (n_hierarchies_with_tg >= 3 and n_letters_rpc >= 20)

final_results = {
    "task_id": TASK_ID,
    "mode": MODE,
    "timestamp": datetime.now().isoformat(),
    "seed": SEED,
    "model": "gemma-2-2b",
    "elapsed_minutes": float(elapsed_min),

    "tg_per_hierarchy": tg_per_hierarchy,

    "observed_absorption": {
        "primary_L24_16k": hierarchy_absorption_primary,
        "mean_across_configs": {k: float(v) for k, v in hierarchy_absorption_mean.items()},
        "all_configs": {k: float(v) for k, v in observed_absorption.items()},
    },

    "ranking_comparison": {
        "ranking_data": ranking_data,
        "tg_ranking": [(h, float(v)) for h, v in tg_ranking],
        "obs_ranking_primary": [(h, float(v)) for h, v in obs_ranking_primary],
        "obs_ranking_mean": [(h, float(v)) for h, v in obs_ranking_mean],
        "correlation": ranking_correlation,
    },

    "per_letter_rpc_analysis": per_letter_analysis,

    "qualitative_assessment": {
        "verdict": qualitative_verdict,
        "details": qualitative_details,
    },

    "summary": {
        "n_hierarchies_with_TG": n_hierarchies_with_tg,
        "n_letters_rpc": n_letters_rpc,
        "qualitative_verdict": qualitative_verdict,
        "ranking_spearman_rho": ranking_correlation.get("vs_L24_16k", {}).get("spearman_rho"),
        "ranking_kendall_tau": ranking_correlation.get("vs_L24_16k", {}).get("kendall_tau"),
        "pilot_pass": pilot_pass,
        "pilot_criteria": {
            "tg_for_3_plus_hierarchies": n_hierarchies_with_tg >= 3,
            "ranking_comparison_done": len(ranking_data) >= 3,
            "rpc_for_20_plus_letters": n_letters_rpc >= 20,
            "result_informative": True,
        },
    },
}

# Save results
output_path = PHASE3_DIR / "tax_qualitative.json"
with open(output_path, "w") as f:
    json.dump(final_results, f, indent=2, default=str)
print(f"\n[Phase 3] Results saved to: {output_path}")

pilot_output = PILOTS_DIR / "phase3_tax_qualitative.json"
with open(pilot_output, "w") as f:
    json.dump(final_results, f, indent=2, default=str)
print(f"[Phase 3] Results also saved to: {pilot_output}")

# Write summary markdown
summary_md = f"""# Phase 3: Absorption Tax Qualitative Validation (PILOT)

## Key Result: T(G) Ranking vs Observed Absorption

**Qualitative Verdict: {qualitative_verdict}**

### Ranking Table

| Hierarchy | T(G) | T(G) Rank | Observed Abs (L24_16k) | Obs Rank |
|-----------|------|-----------|----------------------|----------|
"""

for r in sorted(ranking_data, key=lambda x: x["T_G"], reverse=True):
    tg_rank = next(i+1 for i, (h, _) in enumerate(tg_ranking) if h == r["hierarchy"])
    obs_str = f"{r['observed_absorption_L24_16k']:.4f}" if r['observed_absorption_L24_16k'] is not None else "N/A"
    obs_rank = "N/A"
    for i, (h, _) in enumerate(obs_ranking_primary):
        if h == r["hierarchy"]:
            obs_rank = str(i + 1)
            break
    summary_md += f"| {r['hierarchy']} | {r['T_G']:.6f} | {tg_rank} | {obs_str} | {obs_rank} |\n"

summary_md += f"""
### Ranking Correlation

- Spearman rho: {ranking_correlation.get('vs_L24_16k', {}).get('spearman_rho', 'N/A')}
- Kendall tau: {ranking_correlation.get('vs_L24_16k', {}).get('kendall_tau', 'N/A')}
"""

if "bootstrap" in ranking_correlation:
    bs = ranking_correlation["bootstrap"]
    summary_md += f"- Bootstrap 95% CI (Spearman): [{bs.get('spearman_ci_lower', 'N/A')}, {bs.get('spearman_ci_upper', 'N/A')}]\n"

summary_md += f"""
### Qualitative Details
"""
for d in qualitative_details:
    summary_md += f"- {d}\n"

summary_md += f"""
### T(G) Per Hierarchy (at best probe layer)

| Hierarchy | Layer | T(G) | n_classes |
|-----------|-------|------|-----------|
"""
for key, data in sorted(tg_per_hierarchy.items(), key=lambda x: x[1]["T_G"], reverse=True):
    if "_L" in key:
        h = key.rsplit("_L", 1)[0]
        layer = data["layer"]
        summary_md += f"| {h} | {layer} | {data['T_G']:.6f} | {data['n_classes']} |\n"

summary_md += f"""
### Per-Letter R_pc vs Absorption (Improved Probes)

| SAE Config | Probe Layer | n_letters | Spearman rho | p-value | Verdict |
|-----------|-------------|-----------|-------------|---------|---------|
"""
for config_key, analysis in sorted(per_letter_analysis.items()):
    rho_str = f"{analysis['spearman_rho']:.4f}" if analysis['spearman_rho'] is not None else "N/A"
    p_str = f"{analysis['spearman_p']:.4f}" if analysis['spearman_p'] is not None else "N/A"
    summary_md += f"| {config_key} | {analysis['probe_layer']} | {analysis['n_letters']} | {rho_str} | {p_str} | {analysis['verdict']} |\n"

summary_md += f"""
## Pilot Assessment

- **Hierarchies with T(G):** {n_hierarchies_with_tg}
- **Letters with R_pc:** {n_letters_rpc}
- **Pilot pass:** {'YES' if pilot_pass else 'NO'}
- **Elapsed time:** {elapsed_min:.1f} minutes
"""

summary_path = PILOTS_DIR / "phase3_tax_qualitative_summary.md"
with open(summary_path, "w") as f:
    f.write(summary_md)
print(f"[Phase 3] Summary saved to: {summary_path}")

# ── Print final summary ──────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 3 TAX QUALITATIVE PILOT COMPLETE")
print("="*70)
print(f"  Hierarchies with T(G): {n_hierarchies_with_tg}")
print(f"  Letters with R_pc: {n_letters_rpc}")
print(f"  Qualitative verdict: {qualitative_verdict}")
print(f"  Ranking Spearman rho: {ranking_correlation.get('vs_L24_16k', {}).get('spearman_rho', 'N/A')}")
print(f"  Pilot pass: {'YES' if pilot_pass else 'NO'}")
print(f"  Elapsed: {elapsed_min:.1f} minutes")

# ── Mark done ─────────────────────────────────────────────────────────────
mark_done(
    status="success",
    summary=f"Phase 3 tax qualitative pilot complete. "
            f"N_hierarchies={n_hierarchies_with_tg}, N_letters_rpc={n_letters_rpc}. "
            f"Verdict: {qualitative_verdict}. "
            f"Ranking rho={ranking_correlation.get('vs_L24_16k', {}).get('spearman_rho', 'N/A')}. "
            f"Pilot {'PASS' if pilot_pass else 'FAIL'}."
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
            "planned_min": 30,
            "actual_min": int(elapsed_min),
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {
                "model": "gemma-2-2b",
                "mode": MODE,
                "n_hierarchies": n_hierarchies_with_tg,
                "n_letters_rpc": n_letters_rpc,
                "qualitative_verdict": qualitative_verdict,
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
