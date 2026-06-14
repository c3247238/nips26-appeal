"""
Phase 3.2: Absorption Tax T(G) Per Hierarchy -- FULL VERSION

Comprehensive analysis across ALL 4 hierarchies x multiple SAE configs.
Computes Absorption Tax T(G) = sum of excess L0 needed for absorption-free representation.

T(G) = sum_{c} p_c * R_pc where R_pc = cos^2(probe_dir_c, best_decoder_dir)

FULL mode extensions over PILOT:
  - All 4 hierarchies: first-letter, city-continent, city-country, city-language
  - All SAE configs: L6/L12/L18/L24 x 16k for first-letter probes; L24 x 16k/65k for RAVEL
  - Also load 65k SAEs where available for broader coverage
  - 10,000 bootstrap resamples (vs 2000 in pilot)
  - Spearman + Kendall rank correlations with 4+ hierarchies (meaningful statistical power)
  - Full competition coefficient analysis from 262 rate-distortion pairs
  - Per-class R_pc vs absorption correlation within each hierarchy
  - Cross-layer T(G) comparison (L6, L12, L18, L24)

Dependencies:
  - phase1_probe_training (DONE): probe weights at 4 layers x 4 hierarchies
  - phase1_absorption_firstletter (DONE): 8 SAE configs
  - phase1_absorption_crossdomain (DONE): 3 RAVEL hierarchies x L24 x 16k/65k
  - phase3_rate_distortion_predictors (DONE, FULL): 262 pairs across all hierarchies
"""

import os
import sys
import json
import time
import gc
import random
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

# ============================================================
# Configuration
# ============================================================
TASK_ID = "phase3_absorption_tax"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE3_DIR = RESULTS_DIR / "phase3"
PHASE1_DIR = RESULTS_DIR / "phase1"
FULL_DIR = RESULTS_DIR / "full"
for d in [PILOT_DIR, PHASE3_DIR, FULL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODE = "FULL"  # This is the FULL version

GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "1")

# SAE configurations -- expanded for FULL mode
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_CONFIGS = {
    "L6_16k":  {"layer": 6,  "sae_id": "layer_6/width_16k/canonical"},
    "L12_16k": {"layer": 12, "sae_id": "layer_12/width_16k/canonical"},
    "L18_16k": {"layer": 18, "sae_id": "layer_18/width_16k/canonical"},
    "L24_16k": {"layer": 24, "sae_id": "layer_24/width_16k/canonical"},
    "L6_65k":  {"layer": 6,  "sae_id": "layer_6/width_65k/canonical"},
    "L12_65k": {"layer": 12, "sae_id": "layer_12/width_65k/canonical"},
    "L18_65k": {"layer": 18, "sae_id": "layer_18/width_65k/canonical"},
    "L24_65k": {"layer": 24, "sae_id": "layer_24/width_65k/canonical"},
}

# All 4 hierarchies
HIERARCHIES = ["first-letter", "city-continent", "city-country", "city-language"]

# Bootstrap parameters
N_BOOTSTRAP = 10000

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

start_time = time.time()

# ============================================================
# Process tracking (standard Sibyl protocol)
# ============================================================
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def report_progress(step, total_steps, status="running", metrics=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total_steps,
        "status": status,
        "mode": MODE,
        "metric": metrics or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    done_data = json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "mode": MODE,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    })
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(done_data)
    logger.info(f"DONE (status={status}): {summary}")

def update_gpu_progress(elapsed_seconds, status="completed"):
    """Update gpu_progress.json with completion status."""
    try:
        import fcntl
        progress_path = WORKSPACE / "exp" / "gpu_progress.json"
        lock_path = WORKSPACE / "exp" / ".gpu_progress.lock"
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            data = json.loads(progress_path.read_text()) if progress_path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}
            }
            if status == "completed":
                if TASK_ID not in data.get("completed", []):
                    data.setdefault("completed", []).append(TASK_ID)
            elif status == "failed":
                if TASK_ID not in data.get("failed", []):
                    data.setdefault("failed", []).append(TASK_ID)
            if TASK_ID in data.get("running", {}):
                del data["running"][TASK_ID]
            data.setdefault("timings", {})[TASK_ID] = {
                "planned_min": 20,
                "actual_min": round(elapsed_seconds / 60.0, 1),
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "config_snapshot": {
                    "model": "gemma-2-2b",
                    "mode": MODE,
                    "task": "absorption_tax_full",
                    "n_hierarchies": 4,
                    "n_sae_configs": len(SAE_CONFIGS),
                },
            }
            with open(progress_path, "w") as f:
                json.dump(data, f, indent=2)
            fcntl.flock(lock_f, fcntl.LOCK_UN)
        logger.info("gpu_progress.json updated")
    except Exception as e:
        logger.warning(f"Failed to update gpu_progress.json: {e}")

def update_experiment_state(status="completed"):
    """Update experiment_state.json with task status."""
    try:
        import fcntl
        state_path = WORKSPACE / "exp" / "experiment_state.json"
        lock_path = WORKSPACE / "exp" / "experiment_state.lock"
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            if state_path.exists():
                state = json.loads(state_path.read_text())
            else:
                state = {"schema_version": 1, "tasks": {}}
            if TASK_ID in state.get("tasks", {}):
                state["tasks"][TASK_ID]["status"] = status
                state["tasks"][TASK_ID][f"{status}_at"] = datetime.now().isoformat()
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            fcntl.flock(lock_f, fcntl.LOCK_UN)
        logger.info(f"experiment_state.json updated: {TASK_ID} -> {status}")
    except Exception as e:
        logger.warning(f"Failed to update experiment_state.json: {e}")


# ============================================================
# Main execution
# ============================================================
write_pid()
report_progress(0, 12, "init")
logger.info(f"Phase 3.2 Absorption Tax (FULL) starting on GPU {GPU_ID}")

try:
    import torch
    from scipy import stats
except ImportError as e:
    mark_done("failure", f"Missing dependency: {e}")
    update_gpu_progress(time.time() - start_time, "failed")
    update_experiment_state("failed")
    sys.exit(1)

torch.manual_seed(SEED)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

# ============================================================
# Step 1: Load Phase 1 absorption results (all sources)
# ============================================================
report_progress(1, 12, "loading_phase1_results")
logger.info("Step 1: Loading Phase 1 absorption results")

# Load first-letter absorption (pilot has all 8 SAE configs)
fl_path = PILOT_DIR / "phase1_absorption_firstletter.json"
if not fl_path.exists():
    mark_done("failure", "Phase 1 first-letter absorption results not found")
    update_gpu_progress(time.time() - start_time, "failed")
    update_experiment_state("failed")
    sys.exit(1)

with open(fl_path) as f:
    fl_data = json.load(f)
fl_results = fl_data.get("absorption_results", {})
logger.info(f"Loaded first-letter absorption: {len(fl_results)} SAE configs")

# Load cross-domain absorption (pilot has all 3 hierarchies x 2 widths)
cd_path = PILOT_DIR / "phase1_absorption_crossdomain.json"
if not cd_path.exists():
    # Try full directory
    cd_path = FULL_DIR / "phase1_absorption_crossdomain.json"
if not cd_path.exists():
    mark_done("failure", "Phase 1 cross-domain absorption results not found")
    update_gpu_progress(time.time() - start_time, "failed")
    update_experiment_state("failed")
    sys.exit(1)

with open(cd_path) as f:
    cd_data = json.load(f)
cd_results = cd_data.get("absorption_results", {})
logger.info(f"Loaded cross-domain absorption: {len(cd_results)} SAE configs")
logger.info(f"  Keys: {sorted(cd_results.keys())}")

# Load rate-distortion predictors (FULL: 262 pairs)
rd_path = PHASE3_DIR / "rate_distortion_predictors.json"
rd_data = None
if rd_path.exists():
    with open(rd_path) as f:
        rd_data = json.load(f)
    logger.info(f"Loaded rate-distortion predictor data: {len(rd_data.get('pairs_with_predictors', []))} pairs")
else:
    # Fallback to FULL results file
    rd_fallback = RESULTS_DIR / "phase3_rate_distortion_predictors_FULL.json"
    if rd_fallback.exists():
        with open(rd_fallback) as f:
            rd_data = json.load(f)
        logger.info(f"Loaded FULL rate-distortion predictor data: {len(rd_data.get('pairs_with_predictors', []))} pairs")
    else:
        logger.warning("Rate-distortion predictor data not found; competition coefficients will be approximate")

# ============================================================
# Step 2: Extract observed absorption rates per hierarchy
# ============================================================
report_progress(2, 12, "extracting_observed_rates")
logger.info("Step 2: Extracting observed absorption rates")

observed_absorption = {}

# First-letter at all SAE configs
for config_key, config_data in fl_results.items():
    if isinstance(config_data, dict) and "absorption_rate" in config_data:
        observed_absorption[f"first-letter_{config_key}"] = config_data["absorption_rate"]

# Cross-domain hierarchies
for config_key, config_data in cd_results.items():
    if isinstance(config_data, dict) and "absorption_rate" in config_data:
        # config_key format: "city-continent_L24_16k"
        observed_absorption[config_key] = config_data["absorption_rate"]

# Aggregate per hierarchy at L24_16k (primary comparison)
hierarchy_absorption_primary = {}
for hier in HIERARCHIES:
    key = f"{hier}_L24_16k"
    if key in observed_absorption:
        hierarchy_absorption_primary[hier] = observed_absorption[key]

# Mean across all SAE configs per hierarchy
hierarchy_absorption_mean = {}
for hier in HIERARCHIES:
    rates = [v for k, v in observed_absorption.items() if k.startswith(f"{hier}_")]
    if rates:
        hierarchy_absorption_mean[hier] = float(np.mean(rates))

logger.info(f"Primary comparison (L24_16k): {hierarchy_absorption_primary}")
logger.info(f"Mean across SAEs: {hierarchy_absorption_mean}")
logger.info(f"All observed rates: {len(observed_absorption)} configs")

# ============================================================
# Step 3: Load SAE decoder matrices
# ============================================================
report_progress(3, 12, "loading_sae_decoders")
logger.info("Step 3: Loading SAE decoder matrices")

import sae_lens

W_dec = {}  # sae_key -> (d_sae, d_model) tensor

for sae_key, sae_info in SAE_CONFIGS.items():
    logger.info(f"Loading SAE {sae_key}...")
    try:
        _result = sae_lens.SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=sae_info["sae_id"],
            device=device,
        )
        sae_obj = _result[0] if isinstance(_result, tuple) else _result
        W_dec[sae_key] = sae_obj.W_dec.detach().float().cpu()
        logger.info(f"  {sae_key}: decoder shape = {W_dec[sae_key].shape}")
        del sae_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        logger.error(f"  Failed to load {sae_key}: {e}")

logger.info(f"Loaded {len(W_dec)} SAE decoder matrices")

# Free GPU memory after loading decoders
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# ============================================================
# Step 4: Load probe weights (all 4 hierarchies x 4 layers)
# ============================================================
report_progress(4, 12, "loading_probes")
logger.info("Step 4: Loading probe weights")

probes = {}  # (hierarchy, layer) -> {"coef": ndarray, "classes": list}

# First-letter probes (sklearn)
for layer in [6, 12, 18, 24]:
    path = PHASE1_DIR / f"probe_first-letter_L{layer}_sklearn.npz"
    if path.exists():
        pdata = np.load(path, allow_pickle=True)
        coef = pdata["coef"]  # (26, d_model)
        classes_raw = pdata["classes"]
        classes = [chr(ord('a') + int(c)) for c in classes_raw]
        probes[("first-letter", layer)] = {"coef": coef, "classes": classes}
        logger.info(f"  first-letter L{layer}: coef shape={coef.shape}, {len(classes)} classes")

# RAVEL probes (sklearn)
for hier in ["city-continent", "city-country", "city-language"]:
    for layer in [6, 12, 18, 24]:
        path = PHASE1_DIR / f"probe_{hier}_L{layer}.npz"
        if path.exists():
            pdata = np.load(path, allow_pickle=True)
            coef = pdata["coef"]  # (n_classes, d_model)
            classes = list(pdata["classes"])
            probes[(hier, layer)] = {"coef": coef, "classes": classes}
            logger.info(f"  {hier} L{layer}: coef shape={coef.shape}, {len(classes)} classes")

logger.info(f"Loaded {len(probes)} probe sets total")

# ============================================================
# Step 5: Compute T(G) per hierarchy
# ============================================================
report_progress(5, 12, "computing_tg")
logger.info("Step 5: Computing T(G) per hierarchy (all combos)")


def compute_tg(probe_info, W_dec_matrix, hierarchy_name, layer, sae_key, per_class_data=None):
    """Compute T(G) for one hierarchy at one layer/SAE combination."""
    coef = probe_info["coef"]  # (n_classes, d_model)
    classes = probe_info["classes"]
    n_classes = len(classes)

    # Normalize probe directions
    coef_norms = np.linalg.norm(coef, axis=1, keepdims=True)
    coef_norms = np.clip(coef_norms, 1e-12, None)
    coef_normalized = coef / coef_norms  # (n_classes, d_model)

    # Normalize decoder directions
    dec_norms = W_dec_matrix.norm(dim=1, keepdim=True).clamp(min=1e-12)
    dec_normalized = W_dec_matrix / dec_norms  # (d_sae, d_model)

    # Compute cosine similarity: (n_classes, d_sae)
    coef_t = torch.tensor(coef_normalized, dtype=torch.float32)
    cos_sim_matrix = (coef_t @ dec_normalized.T).numpy()  # (n_classes, d_sae)

    # For each class: find best feature, compute R_pc
    components = []
    per_class_rpc = {}

    # Compute frequency weights
    total_tokens = 0
    if per_class_data:
        total_tokens = sum(per_class_data.get(c, {}).get("total", 0) for c in classes)
    if total_tokens == 0:
        total_tokens = 1  # Use uniform weights

    for c_idx in range(n_classes):
        c_label = classes[c_idx]
        cos_sims = cos_sim_matrix[c_idx]

        # R_pc = max cos^2 (absolute cosine to handle sign)
        best_fid = int(np.argmax(np.abs(cos_sims)))
        best_cos = float(cos_sims[best_fid])
        R_pc = best_cos ** 2

        # Frequency weight
        if per_class_data and total_tokens > 1:
            n_tokens = per_class_data.get(c_label, {}).get("total", 0)
            p_c = n_tokens / total_tokens if total_tokens > 0 else 1.0 / n_classes
        else:
            p_c = 1.0 / n_classes
            n_tokens = 0

        # Observed absorption
        absorption_rate = None
        if per_class_data:
            absorption_rate = per_class_data.get(c_label, {}).get("absorption_rate", None)

        per_class_rpc[c_label] = {
            "R_pc": float(R_pc),
            "best_cos": float(best_cos),
            "best_feature_idx": best_fid,
            "p_c": float(p_c),
            "n_tokens": int(n_tokens),
            "absorption_rate": absorption_rate,
        }

        components.append({
            "class_label": c_label,
            "p_c": float(p_c),
            "R_pc": float(R_pc),
            "contribution": float(p_c * R_pc),
        })

    T_G = sum(c["contribution"] for c in components)

    # Per-class R_pc vs absorption correlation
    rpc_vals = []
    abs_vals = []
    for c_label, data in per_class_rpc.items():
        if data["absorption_rate"] is not None and data["n_tokens"] >= 3:
            rpc_vals.append(data["R_pc"])
            abs_vals.append(data["absorption_rate"])

    rpc_abs_corr = None
    if len(rpc_vals) >= 4:
        try:
            rho, p_val = stats.spearmanr(rpc_vals, abs_vals)
            pearson_r, pearson_p = stats.pearsonr(rpc_vals, abs_vals)
            rpc_abs_corr = {
                "spearman_rho": float(rho) if not np.isnan(rho) else None,
                "spearman_p": float(p_val) if not np.isnan(p_val) else None,
                "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else None,
                "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else None,
                "n_classes_used": len(rpc_vals),
            }
        except Exception:
            pass

    # Distribution stats on R_pc
    all_rpc_vals = [d["R_pc"] for d in per_class_rpc.values()]
    rpc_stats = {
        "mean": float(np.mean(all_rpc_vals)),
        "std": float(np.std(all_rpc_vals)),
        "median": float(np.median(all_rpc_vals)),
        "min": float(np.min(all_rpc_vals)),
        "max": float(np.max(all_rpc_vals)),
        "q25": float(np.percentile(all_rpc_vals, 25)),
        "q75": float(np.percentile(all_rpc_vals, 75)),
    }

    return {
        "hierarchy": hierarchy_name,
        "layer": layer,
        "sae_key": sae_key,
        "T_G": float(T_G),
        "n_classes": n_classes,
        "n_components": len(components),
        "rpc_distribution": rpc_stats,
        "components": sorted(components, key=lambda x: x["contribution"], reverse=True)[:30],  # top 30
        "per_class_rpc": per_class_rpc,
        "rpc_absorption_correlation": rpc_abs_corr,
    }


tg_all = {}  # (hierarchy, layer, sae_key) -> result

# First-letter: probes at all 4 layers, SAEs at all layers
for layer in [6, 12, 18, 24]:
    probe_key = ("first-letter", layer)
    if probe_key not in probes:
        continue

    for width in ["16k", "65k"]:
        sae_key = f"L{layer}_{width}"
        if sae_key not in W_dec:
            continue

        fl_config = fl_results.get(sae_key, {})
        per_class_data = fl_config.get("per_letter", {}) if isinstance(fl_config, dict) else None

        result = compute_tg(probes[probe_key], W_dec[sae_key], "first-letter", layer, sae_key, per_class_data)
        tg_all[("first-letter", layer, sae_key)] = result

        corr_str = ""
        if result["rpc_absorption_correlation"]:
            rho = result["rpc_absorption_correlation"]["spearman_rho"]
            corr_str = f", R_pc-abs rho={rho:.4f}"
        logger.info(f"  first-letter L{layer} {width}: T(G) = {result['T_G']:.6f}{corr_str}")

# RAVEL hierarchies: probes at L24 (best layer), SAEs at L24 (16k, 65k)
for hier in ["city-continent", "city-country", "city-language"]:
    for layer in [24]:  # Primary analysis at L24
        probe_key = (hier, layer)
        if probe_key not in probes:
            continue

        for width in ["16k", "65k"]:
            sae_key = f"L{layer}_{width}"
            if sae_key not in W_dec:
                continue

            cd_config_key = f"{hier}_{sae_key}"
            cd_config = cd_results.get(cd_config_key, {})
            per_class_data = cd_config.get("per_class", {}) if isinstance(cd_config, dict) else None

            result = compute_tg(probes[probe_key], W_dec[sae_key], hier, layer, sae_key, per_class_data)
            tg_all[(hier, layer, sae_key)] = result

            corr_str = ""
            if result["rpc_absorption_correlation"]:
                rho = result["rpc_absorption_correlation"]["spearman_rho"]
                corr_str = f", R_pc-abs rho={rho:.4f}"
            logger.info(f"  {hier} L{layer} {width}: T(G) = {result['T_G']:.6f}{corr_str}")

    # Also try at L12 if probe is decent
    for layer in [12]:
        probe_key = (hier, layer)
        if probe_key not in probes:
            continue
        sae_key = f"L{layer}_16k"
        if sae_key not in W_dec:
            continue
        # No per-class absorption data for L12 RAVEL, so use None
        result = compute_tg(probes[probe_key], W_dec[sae_key], hier, layer, sae_key, None)
        tg_all[(hier, layer, sae_key)] = result
        logger.info(f"  {hier} L{layer} 16k: T(G) = {result['T_G']:.6f} (no absorption data at L12)")

logger.info(f"Computed T(G) for {len(tg_all)} hierarchy x layer x SAE combinations")

# ============================================================
# Step 6: Primary ranking comparison (T(G) vs observed absorption at L24_16k)
# ============================================================
report_progress(6, 12, "ranking_comparison")
logger.info("Step 6: Ranking comparison (T(G) vs observed absorption)")

print("\n" + "=" * 80)
print("RANKING COMPARISON: T(G) vs Observed Absorption at L24_16k")
print("=" * 80)

# Build ranking table at L24_16k (primary comparison)
ranking_data = []
for hier in HIERARCHIES:
    key = (hier, 24, "L24_16k")
    tg_val = tg_all.get(key, {}).get("T_G", None)
    obs_primary = hierarchy_absorption_primary.get(hier, None)
    obs_mean = hierarchy_absorption_mean.get(hier, None)

    if tg_val is not None:
        ranking_data.append({
            "hierarchy": hier,
            "T_G": float(tg_val),
            "observed_absorption_L24_16k": obs_primary,
            "observed_absorption_mean": obs_mean,
        })

print(f"\n  {'Hierarchy':<20} {'T(G)':<12} {'Obs Abs (L24_16k)':<20} {'Obs Abs (mean)':<15}")
print(f"  {'-' * 65}")
for r in ranking_data:
    obs_str = f"{r['observed_absorption_L24_16k']:.4f}" if r['observed_absorption_L24_16k'] is not None else "N/A"
    obs_mean_str = f"{r['observed_absorption_mean']:.4f}" if r['observed_absorption_mean'] is not None else "N/A"
    print(f"  {r['hierarchy']:<20} {r['T_G']:<12.6f} {obs_str:<20} {obs_mean_str:<15}")

# Sort to get rankings
tg_ranking = sorted([(r["hierarchy"], r["T_G"]) for r in ranking_data], key=lambda x: x[1], reverse=True)
obs_ranking_primary = sorted(
    [(r["hierarchy"], r["observed_absorption_L24_16k"]) for r in ranking_data
     if r["observed_absorption_L24_16k"] is not None],
    key=lambda x: x[1], reverse=True
)

print(f"\n  T(G) ranking (highest first):")
for i, (h, v) in enumerate(tg_ranking):
    print(f"    {i + 1}. {h}: {v:.6f}")

print(f"\n  Observed absorption ranking (L24_16k, highest first):")
for i, (h, v) in enumerate(obs_ranking_primary):
    print(f"    {i + 1}. {h}: {v:.4f}")

# Compute rank correlation (NOW with 4 hierarchies, meaningful Spearman)
ranking_correlation = {}
matched_tg = []
matched_obs = []
matched_hier = []

for r in ranking_data:
    if r["observed_absorption_L24_16k"] is not None:
        matched_tg.append(r["T_G"])
        matched_obs.append(r["observed_absorption_L24_16k"])
        matched_hier.append(r["hierarchy"])

if len(matched_tg) >= 3:
    try:
        rho_rank, p_rank = stats.spearmanr(matched_tg, matched_obs)
        tau_rank, p_tau = stats.kendalltau(matched_tg, matched_obs)
    except Exception:
        rho_rank, p_rank = float("nan"), float("nan")
        tau_rank, p_tau = float("nan"), float("nan")

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

    print(f"\n  Spearman rho = {rho_rank:.4f} (p = {p_rank:.4f})")
    print(f"  Kendall tau  = {tau_rank:.4f} (p = {p_tau:.4f})")
    print(f"  n_hierarchies = {len(matched_tg)}")
elif len(matched_tg) >= 2:
    # Fallback for 2 hierarchies
    tg_order = [h for h, _ in tg_ranking]
    obs_order = [h for h, _ in obs_ranking_primary]
    order_match = tg_order == obs_order

    try:
        rho_rank, p_rank = stats.spearmanr(matched_tg, matched_obs)
        tau_rank, p_tau = stats.kendalltau(matched_tg, matched_obs)
    except Exception:
        rho_rank, p_rank = float("nan"), float("nan")
        tau_rank, p_tau = float("nan"), float("nan")

    ranking_correlation["vs_L24_16k"] = {
        "spearman_rho": float(rho_rank) if not np.isnan(rho_rank) else None,
        "spearman_p": float(p_rank) if not np.isnan(p_rank) else None,
        "kendall_tau": float(tau_rank) if not np.isnan(tau_rank) else None,
        "kendall_p": float(p_tau) if not np.isnan(p_tau) else None,
        "n_hierarchies": len(matched_tg),
        "matched_hierarchies": matched_hier,
        "T_G_values": [float(v) for v in matched_tg],
        "obs_values": [float(v) for v in matched_obs],
        "exact_order_match": order_match,
        "note": f"Only {len(matched_tg)} hierarchies, limited statistical power.",
    }

    print(f"\n  Spearman rho = {rho_rank:.4f} (p = {p_rank:.4f})")
    print(f"  NOTE: With only {len(matched_tg)} hierarchies, rank correlation has low statistical power.")

# Also ranking using mean across SAEs
matched_tg_mean = []
matched_obs_mean = []
matched_hier_mean = []
for hier in HIERARCHIES:
    key = (hier, 24, "L24_16k")
    tg_val = tg_all.get(key, {}).get("T_G", None)
    obs_mean_val = hierarchy_absorption_mean.get(hier, None)
    if tg_val is not None and obs_mean_val is not None:
        matched_tg_mean.append(tg_val)
        matched_obs_mean.append(obs_mean_val)
        matched_hier_mean.append(hier)

if len(matched_tg_mean) >= 3:
    try:
        rho_mean, p_mean = stats.spearmanr(matched_tg_mean, matched_obs_mean)
    except Exception:
        rho_mean, p_mean = float("nan"), float("nan")

    ranking_correlation["vs_mean_absorption"] = {
        "spearman_rho": float(rho_mean) if not np.isnan(rho_mean) else None,
        "spearman_p": float(p_mean) if not np.isnan(p_mean) else None,
        "n_hierarchies": len(matched_tg_mean),
        "matched_hierarchies": matched_hier_mean,
        "T_G_values": [float(v) for v in matched_tg_mean],
        "obs_mean_values": [float(v) for v in matched_obs_mean],
    }
    print(f"\n  vs Mean Absorption: Spearman rho = {rho_mean:.4f} (p = {p_mean:.4f})")

# Pairwise concordance
n_pairs = 0
n_concordant = 0
for i in range(len(ranking_data)):
    for j in range(i + 1, len(ranking_data)):
        obs1 = ranking_data[i].get("observed_absorption_L24_16k")
        obs2 = ranking_data[j].get("observed_absorption_L24_16k")
        if obs1 is not None and obs2 is not None:
            n_pairs += 1
            if (ranking_data[i]["T_G"] - ranking_data[j]["T_G"]) * (obs1 - obs2) > 0:
                n_concordant += 1

concordance_rate = n_concordant / n_pairs if n_pairs > 0 else 0
ranking_correlation["pairwise_concordance"] = {
    "n_concordant": n_concordant,
    "n_pairs": n_pairs,
    "concordance_rate": float(concordance_rate),
}
print(f"\n  Pairwise concordance: {n_concordant}/{n_pairs} = {concordance_rate:.2%}")

# ============================================================
# Step 7: Also compare at L24_65k
# ============================================================
report_progress(7, 12, "ranking_65k")
logger.info("Step 7: Ranking comparison at L24_65k")

ranking_data_65k = []
for hier in HIERARCHIES:
    key = (hier, 24, "L24_65k")
    tg_val = tg_all.get(key, {}).get("T_G", None)
    obs_key = f"{hier}_L24_65k"
    obs_val = observed_absorption.get(obs_key, None)
    if tg_val is not None and obs_val is not None:
        ranking_data_65k.append({
            "hierarchy": hier,
            "T_G": float(tg_val),
            "observed_absorption_L24_65k": float(obs_val),
        })

if len(ranking_data_65k) >= 3:
    tg_65k = [r["T_G"] for r in ranking_data_65k]
    obs_65k = [r["observed_absorption_L24_65k"] for r in ranking_data_65k]
    try:
        rho_65k, p_65k = stats.spearmanr(tg_65k, obs_65k)
    except Exception:
        rho_65k, p_65k = float("nan"), float("nan")

    ranking_correlation["vs_L24_65k"] = {
        "spearman_rho": float(rho_65k) if not np.isnan(rho_65k) else None,
        "spearman_p": float(p_65k) if not np.isnan(p_65k) else None,
        "n_hierarchies": len(ranking_data_65k),
        "matched_hierarchies": [r["hierarchy"] for r in ranking_data_65k],
        "T_G_values": tg_65k,
        "obs_values": obs_65k,
    }
    print(f"\n  L24_65k ranking: Spearman rho = {rho_65k:.4f} (p = {p_65k:.4f})")

# ============================================================
# Step 8: Competition coefficients (ecological framework)
# ============================================================
report_progress(8, 12, "competition_coefficients")
logger.info("Step 8: Computing competition coefficients from rate-distortion data")

competition_coefficients = {}

if rd_data and "pairs_with_predictors" in rd_data:
    pairs = rd_data["pairs_with_predictors"]

    for hier in HIERARCHIES:
        hier_pairs = [p for p in pairs if p["hierarchy"] == hier]
        if not hier_pairs:
            continue

        # competition_coeff = cos_sim * co_occur (ecological framework)
        comp_coeffs = [p["cos_sim"] * p["co_occur"] for p in hier_pairs]
        abs_rates = [p["observed_absorption_rate"] for p in hier_pairs]

        competition_coefficients[hier] = {
            "n_pairs": len(hier_pairs),
            "mean_competition_coeff": float(np.mean(comp_coeffs)),
            "std_competition_coeff": float(np.std(comp_coeffs)),
            "median_competition_coeff": float(np.median(comp_coeffs)),
            "mean_cos_sim": float(np.mean([p["cos_sim"] for p in hier_pairs])),
            "mean_co_occur": float(np.mean([p["co_occur"] for p in hier_pairs])),
            "mean_absorption_rate": float(np.mean(abs_rates)),
        }

        # Correlation between competition coeff and absorption
        if len(comp_coeffs) >= 4:
            try:
                rho, p_val = stats.spearmanr(comp_coeffs, abs_rates)
                pearson_r, pearson_p = stats.pearsonr(comp_coeffs, abs_rates)
                competition_coefficients[hier]["comp_absorption_corr"] = {
                    "spearman_rho": float(rho) if not np.isnan(rho) else None,
                    "spearman_p": float(p_val) if not np.isnan(p_val) else None,
                    "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else None,
                    "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else None,
                }
            except Exception:
                pass

        logger.info(f"  {hier}: mean_comp_coeff={np.mean(comp_coeffs):.4f}, "
                    f"n_pairs={len(hier_pairs)}, mean_absorption={np.mean(abs_rates):.4f}")

    # Cross-hierarchy aggregate competition analysis
    all_comp = [p["cos_sim"] * p["co_occur"] for p in pairs]
    all_abs = [p["observed_absorption_rate"] for p in pairs]
    if len(all_comp) >= 10:
        try:
            rho_all, p_all = stats.spearmanr(all_comp, all_abs)
            competition_coefficients["_aggregate"] = {
                "n_pairs_total": len(all_comp),
                "spearman_rho": float(rho_all) if not np.isnan(rho_all) else None,
                "spearman_p": float(p_all) if not np.isnan(p_all) else None,
                "mean_competition_coeff": float(np.mean(all_comp)),
                "mean_absorption_rate": float(np.mean(all_abs)),
            }
            logger.info(f"  AGGREGATE: rho={rho_all:.4f} (p={p_all:.4f}), n={len(all_comp)}")
        except Exception:
            pass

# ============================================================
# Step 9: Bootstrap CIs on ranking correlation
# ============================================================
report_progress(9, 12, "bootstrap")
logger.info(f"Step 9: Bootstrap CIs on ranking correlation (n={N_BOOTSTRAP})")

bootstrap_results = {}

if len(matched_tg) >= 3:
    boot_rhos = []
    boot_taus = []
    matched_tg_arr = np.array(matched_tg)
    matched_obs_arr = np.array(matched_obs)

    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(len(matched_tg_arr), size=len(matched_tg_arr), replace=True)
        if len(set(idx)) < 2:
            continue
        try:
            r, _ = stats.spearmanr(matched_tg_arr[idx], matched_obs_arr[idx])
            t, _ = stats.kendalltau(matched_tg_arr[idx], matched_obs_arr[idx])
            if not np.isnan(r):
                boot_rhos.append(r)
            if not np.isnan(t):
                boot_taus.append(t)
        except Exception:
            pass

    if boot_rhos:
        bootstrap_results = {
            "n_resamples": N_BOOTSTRAP,
            "n_valid_rho": len(boot_rhos),
            "n_valid_tau": len(boot_taus),
            "spearman_mean": float(np.mean(boot_rhos)),
            "spearman_std": float(np.std(boot_rhos)),
            "spearman_ci_lower": float(np.percentile(boot_rhos, 2.5)),
            "spearman_ci_upper": float(np.percentile(boot_rhos, 97.5)),
            "kendall_mean": float(np.mean(boot_taus)) if boot_taus else None,
            "kendall_ci_lower": float(np.percentile(boot_taus, 2.5)) if boot_taus else None,
            "kendall_ci_upper": float(np.percentile(boot_taus, 97.5)) if boot_taus else None,
            "proportion_positive_rho": float(np.mean(np.array(boot_rhos) > 0)),
        }
        logger.info(f"  Bootstrap Spearman: mean={np.mean(boot_rhos):.4f}, "
                    f"95% CI=[{np.percentile(boot_rhos, 2.5):.4f}, {np.percentile(boot_rhos, 97.5):.4f}], "
                    f"prop_positive={np.mean(np.array(boot_rhos) > 0):.2%}")
    else:
        bootstrap_results = {
            "n_resamples": N_BOOTSTRAP,
            "note": "Bootstrap produced no valid correlations",
        }
elif len(matched_tg) >= 2:
    # Degenerate bootstrap with n=2
    boot_rhos = []
    matched_tg_arr = np.array(matched_tg)
    matched_obs_arr = np.array(matched_obs)

    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(len(matched_tg_arr), size=len(matched_tg_arr), replace=True)
        if len(set(idx)) < 2:
            continue
        try:
            r, _ = stats.spearmanr(matched_tg_arr[idx], matched_obs_arr[idx])
            if not np.isnan(r):
                boot_rhos.append(r)
        except Exception:
            pass

    bootstrap_results = {
        "n_resamples": N_BOOTSTRAP,
        "n_valid_rho": len(boot_rhos),
        "note": f"With only {len(matched_tg)} hierarchies, bootstrap correlation has limited interpretability",
        "spearman_mean": float(np.mean(boot_rhos)) if boot_rhos else None,
        "spearman_ci_lower": float(np.percentile(boot_rhos, 2.5)) if boot_rhos else None,
        "spearman_ci_upper": float(np.percentile(boot_rhos, 97.5)) if boot_rhos else None,
    }

ranking_correlation["bootstrap"] = bootstrap_results

# ============================================================
# Step 10: Per-class R_pc analysis (expanded across all configs)
# ============================================================
report_progress(10, 12, "per_class_rpc_analysis")
logger.info("Step 10: Per-class R_pc vs absorption analysis")

per_class_analysis = {}

# First-letter per-letter analysis
for sae_config_key in ["L6_16k", "L6_65k", "L12_16k", "L12_65k", "L18_16k", "L18_65k", "L24_16k", "L24_65k"]:
    fl_config_data = fl_results.get(sae_config_key, {})
    if not isinstance(fl_config_data, dict) or "per_letter" not in fl_config_data:
        continue

    per_letter_fl = fl_config_data["per_letter"]
    layer = int(sae_config_key.split("_")[0].replace("L", ""))

    tg_key = ("first-letter", layer, sae_config_key)
    rpc_data = tg_all.get(tg_key, {}).get("per_class_rpc", {})
    if not rpc_data:
        continue

    rpc_vals = []
    abs_vals = []
    letters_used = []

    for letter in sorted(rpc_data.keys()):
        letter_abs_data = per_letter_fl.get(letter, {})
        n_tokens = letter_abs_data.get("total", 0)
        abs_rate = letter_abs_data.get("absorption_rate", None)

        if abs_rate is not None and n_tokens >= 2:
            rpc_vals.append(rpc_data[letter]["R_pc"])
            abs_vals.append(abs_rate)
            letters_used.append(letter)

    if len(rpc_vals) >= 4:
        try:
            rho, p_val = stats.spearmanr(rpc_vals, abs_vals)
            pearson_r, pearson_p = stats.pearsonr(rpc_vals, abs_vals)
        except Exception:
            rho, p_val = float("nan"), float("nan")
            pearson_r, pearson_p = float("nan"), float("nan")

        verdict = "NOT_SUPPORTED"
        if not np.isnan(rho):
            if abs(rho) >= 0.3 and p_val < 0.05:
                verdict = "SUPPORTED"
            elif abs(rho) >= 0.2:
                verdict = "WEAK_SUPPORT"

        per_class_analysis[f"first-letter_{sae_config_key}"] = {
            "hierarchy": "first-letter",
            "sae_config": sae_config_key,
            "probe_layer": layer,
            "n_classes": len(rpc_vals),
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "spearman_p": float(p_val) if not np.isnan(p_val) else None,
            "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else None,
            "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else None,
            "verdict": verdict,
        }
        logger.info(f"  first-letter {sae_config_key}: rho={rho:.4f} (p={p_val:.4f}), n={len(rpc_vals)}, verdict={verdict}")

# RAVEL per-class analysis (at L24 only, where we have per-class absorption data)
for hier in ["city-continent", "city-country", "city-language"]:
    for width in ["16k", "65k"]:
        sae_key = f"L24_{width}"
        cd_config_key = f"{hier}_{sae_key}"
        cd_config = cd_results.get(cd_config_key, {})
        if not isinstance(cd_config, dict) or "per_class" not in cd_config:
            continue

        per_class_cd = cd_config["per_class"]
        tg_key = (hier, 24, sae_key)
        rpc_data = tg_all.get(tg_key, {}).get("per_class_rpc", {})
        if not rpc_data:
            continue

        rpc_vals = []
        abs_vals = []
        classes_used = []

        for cls_name, cls_data in per_class_cd.items():
            if cls_name in rpc_data:
                n_tokens = cls_data.get("total", 0)
                abs_rate = cls_data.get("absorption_rate", None)
                if abs_rate is not None and n_tokens >= 3:
                    rpc_vals.append(rpc_data[cls_name]["R_pc"])
                    abs_vals.append(abs_rate)
                    classes_used.append(cls_name)

        if len(rpc_vals) >= 4:
            try:
                rho, p_val = stats.spearmanr(rpc_vals, abs_vals)
                pearson_r, pearson_p = stats.pearsonr(rpc_vals, abs_vals)
            except Exception:
                rho, p_val = float("nan"), float("nan")
                pearson_r, pearson_p = float("nan"), float("nan")

            verdict = "NOT_SUPPORTED"
            if not np.isnan(rho):
                if abs(rho) >= 0.3 and p_val < 0.05:
                    verdict = "SUPPORTED"
                elif abs(rho) >= 0.2:
                    verdict = "WEAK_SUPPORT"

            per_class_analysis[f"{hier}_{sae_key}"] = {
                "hierarchy": hier,
                "sae_config": sae_key,
                "probe_layer": 24,
                "n_classes": len(rpc_vals),
                "spearman_rho": float(rho) if not np.isnan(rho) else None,
                "spearman_p": float(p_val) if not np.isnan(p_val) else None,
                "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else None,
                "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else None,
                "verdict": verdict,
            }
            logger.info(f"  {hier} {sae_key}: rho={rho:.4f} (p={p_val:.4f}), n={len(rpc_vals)}, verdict={verdict}")
        else:
            logger.info(f"  {hier} {sae_key}: insufficient classes with absorption data (n={len(rpc_vals)})")

# ============================================================
# Step 11: Cross-layer T(G) comparison
# ============================================================
report_progress(11, 12, "cross_layer_analysis")
logger.info("Step 11: Cross-layer T(G) comparison")

cross_layer = {}

# First-letter across all layers (16k)
fl_layer_tg = {}
for layer in [6, 12, 18, 24]:
    key = ("first-letter", layer, f"L{layer}_16k")
    if key in tg_all:
        fl_layer_tg[layer] = tg_all[key]["T_G"]

if fl_layer_tg:
    cross_layer["first-letter_16k"] = {
        "layers": sorted(fl_layer_tg.keys()),
        "T_G_values": {str(k): float(v) for k, v in sorted(fl_layer_tg.items())},
        "trend": "increasing" if all(fl_layer_tg.get(a, 0) <= fl_layer_tg.get(b, 0)
                                     for a, b in zip([6, 12, 18], [12, 18, 24])
                                     if a in fl_layer_tg and b in fl_layer_tg) else "non-monotonic",
    }
    print(f"\n  First-letter T(G) by layer (16k):")
    for layer in sorted(fl_layer_tg.keys()):
        print(f"    L{layer}: {fl_layer_tg[layer]:.6f}")

# First-letter across all layers (65k)
fl_layer_tg_65k = {}
for layer in [6, 12, 18, 24]:
    key = ("first-letter", layer, f"L{layer}_65k")
    if key in tg_all:
        fl_layer_tg_65k[layer] = tg_all[key]["T_G"]

if fl_layer_tg_65k:
    cross_layer["first-letter_65k"] = {
        "layers": sorted(fl_layer_tg_65k.keys()),
        "T_G_values": {str(k): float(v) for k, v in sorted(fl_layer_tg_65k.items())},
    }
    print(f"\n  First-letter T(G) by layer (65k):")
    for layer in sorted(fl_layer_tg_65k.keys()):
        print(f"    L{layer}: {fl_layer_tg_65k[layer]:.6f}")

# RAVEL hierarchies at L12 vs L24 (16k)
for hier in ["city-continent", "city-country", "city-language"]:
    hier_layers = {}
    for layer in [12, 24]:
        key = (hier, layer, f"L{layer}_16k")
        if key in tg_all:
            hier_layers[layer] = tg_all[key]["T_G"]
    if hier_layers:
        cross_layer[f"{hier}_16k"] = {
            "layers": sorted(hier_layers.keys()),
            "T_G_values": {str(k): float(v) for k, v in sorted(hier_layers.items())},
        }

# ============================================================
# Step 12: Final compilation
# ============================================================
report_progress(12, 12, "compiling_output")
logger.info("Step 12: Compiling final results")

# Qualitative verdict
qualitative_verdict = "INCONCLUSIVE"
qualitative_details = []

if len(ranking_data) >= 2:
    tg_sorted = sorted(ranking_data, key=lambda x: x["T_G"], reverse=True)
    obs_sorted = sorted(
        [r for r in ranking_data if r["observed_absorption_L24_16k"] is not None],
        key=lambda x: x["observed_absorption_L24_16k"], reverse=True
    )

    if len(obs_sorted) >= 2:
        top1_match = tg_sorted[0]["hierarchy"] == obs_sorted[0]["hierarchy"]
        qualitative_details.append(
            f"Top hierarchy: T(G)={tg_sorted[0]['hierarchy']} ({tg_sorted[0]['T_G']:.6f}), "
            f"Obs={obs_sorted[0]['hierarchy']} ({obs_sorted[0]['observed_absorption_L24_16k']:.4f}) "
            f"(match={top1_match})"
        )

        concordance = ranking_correlation.get("pairwise_concordance", {}).get("concordance_rate", 0)
        rho_val = ranking_correlation.get("vs_L24_16k", {}).get("spearman_rho", None)
        p_val = ranking_correlation.get("vs_L24_16k", {}).get("spearman_p", None)

        if concordance >= 0.8 and rho_val is not None and rho_val >= 0.7 and (p_val is None or p_val < 0.1):
            qualitative_verdict = "STRONG_SUPPORT"
        elif concordance >= 0.7 and rho_val is not None and rho_val >= 0.5:
            qualitative_verdict = "MODERATE_SUPPORT"
        elif concordance >= 0.5 and rho_val is not None and rho_val >= 0.2:
            qualitative_verdict = "WEAK_SUPPORT"
        elif concordance >= 0.5:
            qualitative_verdict = "PARTIAL_SUPPORT"
        else:
            qualitative_verdict = "NOT_SUPPORTED"

        qualitative_details.append(f"Concordance: {concordance:.2%}")
        if rho_val is not None:
            qualitative_details.append(f"Spearman rho: {rho_val:.4f} (p={p_val})")

# Per-class R_pc evidence summary
n_supported = sum(1 for a in per_class_analysis.values() if a.get("verdict") == "SUPPORTED")
n_weak = sum(1 for a in per_class_analysis.values() if a.get("verdict") == "WEAK_SUPPORT")
n_total = len(per_class_analysis)
qualitative_details.append(f"Per-class R_pc support: {n_supported} supported + {n_weak} weak out of {n_total} configs")

# Comparison with iter_008
qualitative_details.append(
    "iter_008 reference: ranking rho=-0.20, concordance=50%. "
    f"Current FULL: {len(ranking_data)} hierarchies."
)

print(f"\n{'=' * 80}")
print(f"QUALITATIVE VERDICT: {qualitative_verdict}")
print(f"{'=' * 80}")
for d in qualitative_details:
    print(f"  {d}")

# Build T(G) summary table
tg_summary = []
for key, result in sorted(tg_all.items(), key=lambda x: x[1]["T_G"], reverse=True):
    hier, layer, sae_key = key
    entry = {
        "hierarchy": hier,
        "layer": layer,
        "sae_key": sae_key,
        "T_G": result["T_G"],
        "n_classes": result["n_classes"],
        "rpc_mean": result["rpc_distribution"]["mean"],
        "rpc_std": result["rpc_distribution"]["std"],
    }
    if result.get("rpc_absorption_correlation"):
        entry["rpc_absorption_rho"] = result["rpc_absorption_correlation"].get("spearman_rho")
        entry["rpc_absorption_p"] = result["rpc_absorption_correlation"].get("spearman_p")
    tg_summary.append(entry)

# Elapsed time
elapsed = time.time() - start_time
elapsed_min = elapsed / 60.0

# Build serializable tg_per_hierarchy (compact)
tg_per_hierarchy_compact = {}
for key, result in tg_all.items():
    hier, layer, sae_key = key
    compact_key = f"{hier}_L{layer}_{sae_key.split('_')[1]}"
    tg_per_hierarchy_compact[compact_key] = {
        "hierarchy": result["hierarchy"],
        "layer": result["layer"],
        "sae_key": result["sae_key"],
        "T_G": result["T_G"],
        "n_classes": result["n_classes"],
        "rpc_distribution": result["rpc_distribution"],
        "rpc_absorption_correlation": result.get("rpc_absorption_correlation"),
    }

final_results = {
    "task_id": TASK_ID,
    "mode": MODE,
    "timestamp": datetime.now().isoformat(),
    "seed": SEED,
    "model": "gemma-2-2b",
    "elapsed_seconds": float(elapsed),
    "elapsed_minutes": float(elapsed_min),

    # T(G) summary table (sorted by T(G))
    "tg_summary": tg_summary,

    # T(G) per hierarchy (compact, no per-class data)
    "tg_per_hierarchy": tg_per_hierarchy_compact,

    # Observed absorption rates
    "observed_absorption": {
        "primary_L24_16k": {k: float(v) for k, v in hierarchy_absorption_primary.items()},
        "mean_across_configs": hierarchy_absorption_mean,
        "all_configs": {k: float(v) for k, v in observed_absorption.items()},
    },

    # Primary ranking comparison (L24_16k)
    "ranking_comparison": {
        "ranking_data": ranking_data,
        "tg_ranking": [(h, float(v)) for h, v in tg_ranking],
        "obs_ranking_primary": [(h, float(v)) for h, v in obs_ranking_primary],
        "correlation": ranking_correlation,
    },

    # L24_65k ranking data (if available)
    "ranking_comparison_65k": {
        "ranking_data": ranking_data_65k,
        "correlation": ranking_correlation.get("vs_L24_65k", {}),
    },

    # Cross-layer T(G) analysis
    "cross_layer_analysis": cross_layer,

    # Competition coefficients (ecological framework)
    "competition_coefficients": competition_coefficients,

    # Per-class R_pc analysis
    "per_class_rpc_analysis": per_class_analysis,

    # Qualitative assessment
    "qualitative_assessment": {
        "verdict": qualitative_verdict,
        "details": qualitative_details,
        "n_hierarchies": len(ranking_data),
        "n_tg_computations": len(tg_all),
        "n_sae_configs_loaded": len(W_dec),
    },

    # FULL criteria
    "full_criteria": {
        "n_hierarchies_with_TG_L24": sum(1 for h in HIERARCHIES if (h, 24, "L24_16k") in tg_all),
        "n_total_tg_computations": len(tg_all),
        "has_4_hierarchies": len([h for h in HIERARCHIES if (h, 24, "L24_16k") in tg_all]) >= 4,
        "has_ranking_comparison": len(ranking_data) >= 3,
        "has_competition_coefficients": len(competition_coefficients) > 0,
        "has_cross_layer_analysis": len(cross_layer) > 0,
        "target": "T(G) computed for all 4 hierarchies at L24_16k. Ranking comparison with statistical power. Competition coefficients from 262 pairs.",
    },

    # iter_008 comparison context
    "iter008_reference": {
        "ranking_rho": -0.20,
        "concordance": 0.50,
        "note": "iter_008 had 4 hierarchies with different probes. Current FULL uses improved probes and expanded data.",
    },

    "methodology_notes": {
        "T_G_formula": "T(G) = sum_c p_c * R_pc where R_pc = cos^2(probe_dir_c, best_decoder_dir)",
        "p_c_definition": "Frequency weight: n_tokens(class_c) / total_tokens. Uniform if no frequency data.",
        "R_pc_definition": "max_feature cos^2(normalized_probe_weight, normalized_decoder_direction)",
        "competition_coeff_definition": "cos_sim(probe, decoder) * co_occur(parent, child) -- ecological framework",
        "ranking_comparison": "Spearman rank correlation between T(G) and observed absorption across hierarchies",
        "bootstrap_n": N_BOOTSTRAP,
        "sae_configs_loaded": list(W_dec.keys()),
        "n_probes_loaded": len(probes),
    },
}

# Save to phase3 directory
output_path = PHASE3_DIR / "absorption_tax.json"
with open(output_path, "w") as f:
    json.dump(final_results, f, indent=2, default=str)
logger.info(f"Results saved to: {output_path}")

# Also save to full directory
full_output = FULL_DIR / "phase3_absorption_tax.json"
with open(full_output, "w") as f:
    json.dump(final_results, f, indent=2, default=str)
logger.info(f"Results also saved to: {full_output}")

# Also save FULL marker
full_marker = RESULTS_DIR / f"{TASK_ID}_FULL.json"
with open(full_marker, "w") as f:
    json.dump({
        "task_id": TASK_ID,
        "mode": "FULL",
        "timestamp": datetime.now().isoformat(),
        "tg_summary_compact": [
            {"hierarchy": e["hierarchy"], "layer": e["layer"], "sae_key": e["sae_key"], "T_G": e["T_G"]}
            for e in tg_summary[:10]
        ],
        "verdict": qualitative_verdict,
        "ranking_rho": ranking_correlation.get("vs_L24_16k", {}).get("spearman_rho"),
        "concordance": concordance_rate,
        "elapsed_minutes": elapsed_min,
    }, f, indent=2)

# Print final summary
print(f"\n{'=' * 80}")
print("PHASE 3.2 ABSORPTION TAX FULL COMPLETE")
print(f"{'=' * 80}")
print(f"  Mode: FULL")
print(f"  Hierarchies with T(G) at L24_16k: {sum(1 for h in HIERARCHIES if (h, 24, 'L24_16k') in tg_all)}")
print(f"  Total T(G) computations: {len(tg_all)}")
print(f"  SAE decoders loaded: {len(W_dec)}")
print(f"  T(G) ranking (L24_16k):")
for h, v in tg_ranking:
    print(f"    {h}: {v:.6f}")
print(f"  Qualitative verdict: {qualitative_verdict}")
rho_display = ranking_correlation.get('vs_L24_16k', {}).get('spearman_rho', 'N/A')
print(f"  Ranking Spearman rho: {rho_display}")
print(f"  Pairwise concordance: {concordance_rate:.2%}")
print(f"  Per-class R_pc: {n_supported} SUPPORTED + {n_weak} WEAK out of {n_total}")
print(f"  Elapsed: {elapsed_min:.1f} minutes")

# Update tracking files
mark_done(
    status="success",
    summary=f"Phase 3.2 absorption tax FULL complete. "
            f"N_hierarchies={len(ranking_data)}. "
            f"Verdict: {qualitative_verdict}. "
            f"Ranking rho={rho_display}. "
            f"Concordance={concordance_rate:.2%}. "
            f"T(G) computations={len(tg_all)}."
)
update_gpu_progress(elapsed, "completed")
update_experiment_state("completed")

logger.info(f"All done. Results at: {output_path}")
