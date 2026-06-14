"""
Phase 3.2: Absorption Tax T(G) Per Hierarchy (Qualitative) -- Pilot

CPU-only analysis (with optional GPU for SAE decoder loading).
Computes Absorption Tax T(G) = sum of excess L0 needed for absorption-free representation
for each hierarchy type.

T(G) = sum_{c} p_c * R_pc where R_pc = cos^2(probe_dir_c, best_decoder_dir)

Pipeline:
  1. Load trained probe weights for each hierarchy (first-letter, city-continent)
  2. Load SAE decoder matrices (L24 16k primary, L12 16k secondary)
  3. For each hierarchy: compute R_pc for each class, then T(G) = weighted sum
  4. Compare T(G) ranking with observed absorption ranking
  5. Compute competition coefficients (cos_sim x co_occur overlap) per hierarchy
  6. Bootstrap CIs on ranking correlations
  7. Per-class R_pc vs absorption correlation (within each hierarchy)

Dependencies:
  - phase1_probe_training (DONE): probe weights
  - phase1_absorption_firstletter (DONE): first-letter per-letter absorption rates
  - phase1_absorption_crossdomain (DONE): city-continent per-class absorption rates
  - phase3_rate_distortion_predictors (DONE): co-occurrence data for competition coefficients

PILOT mode: Uses L24_16k and L12_16k SAEs. Two hierarchies (first-letter, city-continent).
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
for d in [PILOT_DIR, PHASE3_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODE = os.environ.get("PILOT_MODE", "PILOT").upper()
if "--full" in sys.argv:
    MODE = "FULL"

GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "4")

# SAE configurations
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_CONFIGS = {
    "L24_16k": {"layer": 24, "sae_id": "layer_24/width_16k/canonical"},
    "L12_16k": {"layer": 12, "sae_id": "layer_12/width_16k/canonical"},
}
PRIMARY_SAE = "L24_16k"

# Bootstrap parameters
N_BOOTSTRAP = 2000 if MODE == "PILOT" else 10000

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
                "planned_min": 15,
                "actual_min": round(elapsed_seconds / 60.0, 1),
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "config_snapshot": {
                    "model": "gemma-2-2b",
                    "mode": MODE,
                    "task": "absorption_tax_qualitative",
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
report_progress(0, 10, "init")
logger.info(f"Phase 3.2 Absorption Tax ({MODE}) starting on GPU {GPU_ID}")

try:
    import torch
    from scipy import stats
except ImportError as e:
    mark_done("failure", f"Missing dependency: {e}")
    update_gpu_progress(time.time() - start_time, "failed")
    update_experiment_state("failed")
    sys.exit(1)

torch.manual_seed(SEED)
# When CUDA_VISIBLE_DEVICES is set, the visible GPU becomes device 0
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

# ============================================================
# Step 1: Load Phase 1 absorption results
# ============================================================
report_progress(1, 10, "loading_phase1_results")
logger.info("Step 1: Loading Phase 1 absorption results")

# Load first-letter absorption data
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

# Load cross-domain (city-continent) absorption data
cd_path = PILOT_DIR / "phase1_absorption_crossdomain.json"
if not cd_path.exists():
    mark_done("failure", "Phase 1 cross-domain absorption results not found")
    update_gpu_progress(time.time() - start_time, "failed")
    update_experiment_state("failed")
    sys.exit(1)

with open(cd_path) as f:
    cd_data = json.load(f)
cd_results = cd_data.get("absorption_results", {})
logger.info(f"Loaded cross-domain absorption: {len(cd_results)} SAE configs")

# Load rate-distortion predictors (for co-occurrence data / competition coefficients)
rd_path = PHASE3_DIR / "rate_distortion_predictors.json"
rd_data = None
if rd_path.exists():
    with open(rd_path) as f:
        rd_data = json.load(f)
    logger.info("Loaded rate-distortion predictor data for competition coefficients")
else:
    logger.warning("Rate-distortion predictor data not found; competition coefficients will be approximate")

# ============================================================
# Step 2: Extract observed absorption rates per hierarchy
# ============================================================
report_progress(2, 10, "extracting_observed_rates")
logger.info("Step 2: Extracting observed absorption rates")

observed_absorption = {}

# First-letter at all SAE configs
for config_key, config_data in fl_results.items():
    if isinstance(config_data, dict) and "absorption_rate" in config_data:
        observed_absorption[f"first-letter_{config_key}"] = config_data["absorption_rate"]
        logger.info(f"  first-letter {config_key}: {config_data['absorption_rate']:.4f}")

# City-continent at L24_16k
for config_key, config_data in cd_results.items():
    if isinstance(config_data, dict) and "absorption_rate" in config_data:
        observed_absorption[f"city-continent_{config_key}"] = config_data["absorption_rate"]
        logger.info(f"  city-continent {config_key}: {config_data['absorption_rate']:.4f}")

# Aggregate per hierarchy at L24_16k (primary comparison)
hierarchy_absorption_primary = {}
if "first-letter_L24_16k" in observed_absorption:
    hierarchy_absorption_primary["first-letter"] = observed_absorption["first-letter_L24_16k"]
if "city-continent_L24_16k" in observed_absorption:
    hierarchy_absorption_primary["city-continent"] = observed_absorption["city-continent_L24_16k"]

# Mean across all SAE configs
hierarchy_absorption_mean = {}
for hier in ["first-letter", "city-continent"]:
    rates = [v for k, v in observed_absorption.items() if k.startswith(f"{hier}_")]
    if rates:
        hierarchy_absorption_mean[hier] = float(np.mean(rates))

logger.info(f"Primary comparison (L24_16k): {hierarchy_absorption_primary}")
logger.info(f"Mean across SAEs: {hierarchy_absorption_mean}")

# ============================================================
# Step 3: Load SAE decoder matrices
# ============================================================
report_progress(3, 10, "loading_sae_decoders")
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

# Free GPU memory after loading decoders (rest is CPU-only)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# ============================================================
# Step 4: Load probe weights
# ============================================================
report_progress(4, 10, "loading_probes")
logger.info("Step 4: Loading probe weights")

probes = {}  # (hierarchy, layer) -> {"coef": ndarray, "classes": list}

# First-letter probes (sklearn)
for layer in [12, 24]:
    path = PHASE1_DIR / f"probe_first-letter_L{layer}_sklearn.npz"
    if path.exists():
        pdata = np.load(path, allow_pickle=True)
        coef = pdata["coef"]  # (26, d_model)
        classes_raw = pdata["classes"]
        # Classes are int indices 0-25 -> map to letters
        classes = [chr(ord('a') + int(c)) for c in classes_raw]
        probes[("first-letter", layer)] = {"coef": coef, "classes": classes}
        logger.info(f"  first-letter L{layer}: coef shape={coef.shape}, {len(classes)} classes")

# City-continent probes (sklearn)
for layer in [12, 24]:
    path = PHASE1_DIR / f"probe_city-continent_L{layer}.npz"
    if path.exists():
        pdata = np.load(path, allow_pickle=True)
        coef = pdata["coef"]  # (n_classes, d_model)
        classes = list(pdata["classes"])
        probes[("city-continent", layer)] = {"coef": coef, "classes": classes}
        logger.info(f"  city-continent L{layer}: coef shape={coef.shape}, {len(classes)} classes")

logger.info(f"Loaded {len(probes)} probe sets")

# ============================================================
# Step 5: Compute T(G) per hierarchy
# ============================================================
report_progress(5, 10, "computing_tg")
logger.info("Step 5: Computing T(G) per hierarchy")

# T(G) = sum_c p_c * R_pc
# R_pc = max_feature cos^2(probe_dir_c, decoder_dir_feature)
# p_c = frequency weight of class c

tg_per_hierarchy = {}

def compute_tg(probe_info, W_dec_matrix, hierarchy_name, layer, per_class_data=None):
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
        if per_class_data:
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
            rpc_abs_corr = {
                "spearman_rho": float(rho) if not np.isnan(rho) else None,
                "spearman_p": float(p_val) if not np.isnan(p_val) else None,
                "n_classes_used": len(rpc_vals),
            }
        except Exception:
            pass

    return {
        "hierarchy": hierarchy_name,
        "layer": layer,
        "T_G": float(T_G),
        "n_classes": n_classes,
        "n_components": len(components),
        "components": sorted(components, key=lambda x: x["contribution"], reverse=True),
        "per_class_rpc": per_class_rpc,
        "rpc_absorption_correlation": rpc_abs_corr,
    }


# Compute T(G) for each hierarchy x layer x SAE config
for (hier_name, layer), probe_info in probes.items():
    sae_key = f"L{layer}_16k"
    if sae_key not in W_dec:
        logger.warning(f"  No decoder for {sae_key}, skipping {hier_name} L{layer}")
        continue

    # Get per-class absorption data
    per_class_data = None
    if hier_name == "first-letter":
        fl_config = fl_results.get(sae_key, {})
        per_class_data = fl_config.get("per_letter", {})
    elif hier_name == "city-continent":
        cd_config = cd_results.get(sae_key, {})
        per_class_data = cd_config.get("per_class", {})

    result = compute_tg(probe_info, W_dec[sae_key], hier_name, layer, per_class_data)
    tg_key = f"{hier_name}_L{layer}"
    tg_per_hierarchy[tg_key] = result

    corr_str = ""
    if result["rpc_absorption_correlation"]:
        rho = result["rpc_absorption_correlation"]["spearman_rho"]
        p = result["rpc_absorption_correlation"]["spearman_p"]
        n = result["rpc_absorption_correlation"]["n_classes_used"]
        corr_str = f", R_pc-absorption rho={rho:.4f} (p={p:.4f}, n={n})"

    logger.info(f"  {tg_key}: T(G) = {result['T_G']:.6f}{corr_str}")

# ============================================================
# Step 6: Ranking comparison (T(G) vs observed absorption)
# ============================================================
report_progress(6, 10, "ranking_comparison")
logger.info("Step 6: Ranking comparison (T(G) vs observed absorption)")

print("\n" + "=" * 70)
print("RANKING COMPARISON: T(G) vs Observed Absorption at L24_16k")
print("=" * 70)

# Build ranking table at L24 (primary comparison layer)
ranking_data = []
for hier in ["first-letter", "city-continent"]:
    tg_key = f"{hier}_L24"
    tg_val = tg_per_hierarchy.get(tg_key, {}).get("T_G", None)
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

# Compute rank correlation (need at least 3 hierarchies for meaningful Spearman)
ranking_correlation = {}
matched_tg = []
matched_obs = []
matched_hier = []

# With only 2 hierarchies, we can only check direction agreement
if len(ranking_data) >= 2:
    # Check if ordering matches
    tg_order = [h for h, _ in tg_ranking]
    obs_order = [h for h, _ in obs_ranking_primary]
    order_match = tg_order == obs_order

    # Also compute actual Spearman/Kendall even for n=2
    matched_tg = []
    matched_obs = []
    matched_hier = []
    for r in ranking_data:
        if r["observed_absorption_L24_16k"] is not None:
            matched_tg.append(r["T_G"])
            matched_obs.append(r["observed_absorption_L24_16k"])
            matched_hier.append(r["hierarchy"])

    if len(matched_tg) >= 2:
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
        }

        print(f"\n  Ordering match: {order_match}")
        print(f"  Spearman rho = {rho_rank:.4f} (p = {p_rank:.4f})")
        print(f"  Kendall tau  = {tau_rank:.4f} (p = {p_tau:.4f})")
        print(f"  NOTE: With only {len(matched_tg)} hierarchies, rank correlation has low statistical power.")

    # Pairwise concordance (count how many pairs have consistent ordering)
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
    print(f"  Pairwise concordance: {n_concordant}/{n_pairs} = {concordance_rate:.2%}")

# ============================================================
# Step 7: Competition coefficients (cos_sim x co_occur)
# ============================================================
report_progress(7, 10, "competition_coefficients")
logger.info("Step 7: Computing competition coefficients from rate-distortion data")

competition_coefficients = {}

if rd_data and "pairs_with_predictors" in rd_data:
    pairs = rd_data["pairs_with_predictors"]

    # Compute per-hierarchy mean competition coefficient
    for hier in ["first-letter", "city-continent"]:
        hier_pairs = [p for p in pairs if p["hierarchy"] == hier]
        if hier_pairs:
            # competition_coeff = cos_sim * co_occur (ecological framework)
            comp_coeffs = [p["cos_sim"] * p["co_occur"] for p in hier_pairs]
            abs_rates = [p["observed_absorption_rate"] for p in hier_pairs]

            competition_coefficients[hier] = {
                "n_pairs": len(hier_pairs),
                "mean_competition_coeff": float(np.mean(comp_coeffs)),
                "std_competition_coeff": float(np.std(comp_coeffs)),
                "mean_cos_sim": float(np.mean([p["cos_sim"] for p in hier_pairs])),
                "mean_co_occur": float(np.mean([p["co_occur"] for p in hier_pairs])),
                "mean_absorption_rate": float(np.mean(abs_rates)),
                "individual_pairs": [
                    {
                        "class_name": p["class_name"],
                        "cos_sim": p["cos_sim"],
                        "co_occur": p["co_occur"],
                        "competition_coeff": p["cos_sim"] * p["co_occur"],
                        "absorption_rate": p["observed_absorption_rate"],
                    }
                    for p in hier_pairs
                ],
            }

            # Correlation between competition coeff and absorption
            if len(comp_coeffs) >= 4:
                try:
                    rho, p_val = stats.spearmanr(comp_coeffs, abs_rates)
                    competition_coefficients[hier]["comp_absorption_corr"] = {
                        "spearman_rho": float(rho) if not np.isnan(rho) else None,
                        "spearman_p": float(p_val) if not np.isnan(p_val) else None,
                    }
                except Exception:
                    pass

            logger.info(f"  {hier}: mean_comp_coeff={np.mean(comp_coeffs):.4f}, "
                        f"n_pairs={len(hier_pairs)}, mean_absorption={np.mean(abs_rates):.4f}")

# ============================================================
# Step 8: Bootstrap CIs on ranking correlation
# ============================================================
report_progress(8, 10, "bootstrap")
logger.info("Step 8: Bootstrap CIs on ranking correlation")

bootstrap_results = {}

if len(matched_tg) >= 2:
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
            "spearman_ci_lower": float(np.percentile(boot_rhos, 2.5)),
            "spearman_ci_upper": float(np.percentile(boot_rhos, 97.5)),
            "kendall_mean": float(np.mean(boot_taus)) if boot_taus else None,
            "kendall_ci_lower": float(np.percentile(boot_taus, 2.5)) if boot_taus else None,
            "kendall_ci_upper": float(np.percentile(boot_taus, 97.5)) if boot_taus else None,
        }
        logger.info(f"  Bootstrap Spearman: mean={np.mean(boot_rhos):.4f}, "
                    f"95% CI=[{np.percentile(boot_rhos, 2.5):.4f}, {np.percentile(boot_rhos, 97.5):.4f}]")
    else:
        logger.warning("  Bootstrap produced no valid correlations (expected with n=2)")
        bootstrap_results = {
            "n_resamples": N_BOOTSTRAP,
            "note": "With only 2 hierarchies, bootstrap correlation is degenerate",
        }

ranking_correlation["bootstrap"] = bootstrap_results

# ============================================================
# Step 9: Per-letter R_pc analysis
# ============================================================
report_progress(9, 10, "per_letter_analysis")
logger.info("Step 9: Per-letter R_pc vs absorption analysis")

per_letter_analysis = {}

for sae_config_key in ["L12_16k", "L24_16k"]:
    fl_config_data = fl_results.get(sae_config_key, {})
    if not isinstance(fl_config_data, dict) or "per_letter" not in fl_config_data:
        continue

    per_letter_fl = fl_config_data["per_letter"]
    sae_layer = int(sae_config_key.split("_")[0].replace("L", ""))

    probe_key = ("first-letter", sae_layer)
    if probe_key not in probes:
        continue

    tg_key = f"first-letter_L{sae_layer}"
    rpc_data = tg_per_hierarchy.get(tg_key, {}).get("per_class_rpc", {})
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

        per_letter_analysis[sae_config_key] = {
            "sae_config": sae_config_key,
            "probe_layer": sae_layer,
            "n_letters": len(rpc_vals),
            "letters_used": letters_used,
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "spearman_p": float(p_val) if not np.isnan(p_val) else None,
            "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else None,
            "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else None,
            "verdict": verdict,
        }

        logger.info(f"  {sae_config_key}: rho={rho:.4f} (p={p_val:.4f}), n={len(rpc_vals)}, verdict={verdict}")
    else:
        logger.info(f"  {sae_config_key}: insufficient letters with absorption data (n={len(rpc_vals)})")

# ============================================================
# Step 10: Qualitative assessment and final compilation
# ============================================================
report_progress(10, 10, "compiling_output")
logger.info("Step 10: Compiling final results")

# Qualitative verdict
qualitative_verdict = "INCONCLUSIVE"
qualitative_details = []

if len(ranking_data) >= 2:
    # Direction check: does higher T(G) correspond to higher absorption?
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
        if concordance >= 0.8:
            qualitative_verdict = "QUALITATIVE_SUPPORT"
        elif concordance >= 0.5:
            qualitative_verdict = "PARTIAL_SUPPORT"
        else:
            qualitative_verdict = "NOT_SUPPORTED"

        qualitative_details.append(f"Concordance: {concordance:.2%}")

# Also assess per-letter R_pc evidence
rpc_support_count = sum(1 for a in per_letter_analysis.values() if a.get("verdict") in ["SUPPORTED", "WEAK_SUPPORT"])
qualitative_details.append(f"Per-letter R_pc support: {rpc_support_count}/{len(per_letter_analysis)} SAE configs")

# Comparison with iter_008 (rho=-0.20)
qualitative_details.append(
    "iter_008 reference: ranking rho=-0.20, concordance=50%. "
    "Current pilot has only 2 hierarchies so direct ranking comparison is limited."
)

print(f"\n{'=' * 70}")
print(f"QUALITATIVE VERDICT: {qualitative_verdict}")
print(f"{'=' * 70}")
for d in qualitative_details:
    print(f"  {d}")

# Compile final results
elapsed = time.time() - start_time
elapsed_min = elapsed / 60.0

# Pilot pass criteria
n_hierarchies_with_tg = len(set(r.split("_L")[0] for r in tg_per_hierarchy.keys()))
has_ranking_comparison = len(ranking_data) >= 2
has_competition_coefficients = len(competition_coefficients) > 0

pilot_pass = n_hierarchies_with_tg >= 2 and has_ranking_comparison

final_results = {
    "task_id": TASK_ID,
    "mode": MODE,
    "timestamp": datetime.now().isoformat(),
    "seed": SEED,
    "model": "gemma-2-2b",
    "elapsed_seconds": float(elapsed),
    "elapsed_minutes": float(elapsed_min),

    # T(G) computation results
    "tg_per_hierarchy": {
        k: {key: val for key, val in v.items() if key != "components" or len(v.get("components", [])) <= 30}
        for k, v in tg_per_hierarchy.items()
    },

    # Summary T(G) table
    "tg_summary": [
        {
            "hierarchy": v["hierarchy"],
            "layer": v["layer"],
            "T_G": v["T_G"],
            "n_classes": v["n_classes"],
            "rpc_absorption_rho": (v.get("rpc_absorption_correlation") or {}).get("spearman_rho"),
            "rpc_absorption_p": (v.get("rpc_absorption_correlation") or {}).get("spearman_p"),
        }
        for k, v in sorted(tg_per_hierarchy.items(), key=lambda x: x[1]["T_G"], reverse=True)
    ],

    # Observed absorption rates
    "observed_absorption": {
        "primary_L24_16k": {k: float(v) for k, v in hierarchy_absorption_primary.items()},
        "mean_across_configs": hierarchy_absorption_mean,
        "all_configs": {k: float(v) for k, v in observed_absorption.items()},
    },

    # Ranking comparison
    "ranking_comparison": {
        "ranking_data": ranking_data,
        "tg_ranking": [(h, float(v)) for h, v in tg_ranking],
        "obs_ranking_primary": [(h, float(v)) for h, v in obs_ranking_primary],
        "correlation": ranking_correlation,
    },

    # Competition coefficients (ecological framework)
    "competition_coefficients": competition_coefficients,

    # Per-letter R_pc analysis
    "per_letter_rpc_analysis": per_letter_analysis,

    # Qualitative assessment
    "qualitative_assessment": {
        "verdict": qualitative_verdict,
        "details": qualitative_details,
    },

    # Pilot criteria
    "pilot_criteria_met": pilot_pass,
    "pilot_criteria_details": {
        "n_hierarchies_with_TG": n_hierarchies_with_tg,
        "has_ranking_comparison": has_ranking_comparison,
        "has_competition_coefficients": has_competition_coefficients,
        "target": "T(G) computed for all 4 hierarchies. Ranking comparison with observed absorption. Competition coefficients computed.",
        "note": "PILOT has only 2 hierarchies (first-letter, city-continent). Full mode will add city-country and city-language.",
    },

    # iter_008 comparison context
    "iter008_reference": {
        "ranking_rho": -0.20,
        "concordance": 0.50,
        "note": "iter_008 had 4 hierarchies. Current pilot has 2. Direct comparison limited.",
    },

    "methodology_notes": {
        "T_G_formula": "T(G) = sum_c p_c * R_pc where R_pc = cos^2(probe_dir_c, best_decoder_dir)",
        "p_c_definition": "Frequency weight: n_tokens(class_c) / total_tokens",
        "R_pc_definition": "max_feature cos^2(normalized_probe_weight, normalized_decoder_direction)",
        "competition_coeff_definition": "cos_sim(probe, decoder) * co_occur(parent, child) -- ecological framework",
        "ranking_comparison": "Spearman rank correlation between T(G) and observed absorption across hierarchies",
    },
}

# Save results
output_path = PHASE3_DIR / "absorption_tax.json"
with open(output_path, "w") as f:
    json.dump(final_results, f, indent=2, default=str)
logger.info(f"Results saved to: {output_path}")

pilot_output = PILOT_DIR / "phase3_absorption_tax.json"
with open(pilot_output, "w") as f:
    json.dump(final_results, f, indent=2, default=str)
logger.info(f"Results also saved to: {pilot_output}")

# Print final summary
print(f"\n{'=' * 70}")
print("PHASE 3.2 ABSORPTION TAX PILOT COMPLETE")
print(f"{'=' * 70}")
print(f"  Hierarchies with T(G): {n_hierarchies_with_tg}")
print(f"  T(G) values:")
for entry in final_results["tg_summary"]:
    print(f"    {entry['hierarchy']} L{entry['layer']}: T(G) = {entry['T_G']:.6f}")
print(f"  Qualitative verdict: {qualitative_verdict}")
print(f"  Ranking correlation: {ranking_correlation.get('vs_L24_16k', {}).get('spearman_rho', 'N/A')}")
print(f"  Pilot pass: {'YES' if pilot_pass else 'NO'}")
print(f"  Elapsed: {elapsed_min:.1f} minutes")

# Update tracking files
mark_done(
    status="success",
    summary=f"Phase 3.2 absorption tax pilot complete. "
            f"N_hierarchies={n_hierarchies_with_tg}. "
            f"Verdict: {qualitative_verdict}. "
            f"Ranking rho={ranking_correlation.get('vs_L24_16k', {}).get('spearman_rho', 'N/A')}. "
            f"Pilot {'PASS' if pilot_pass else 'FAIL'}."
)
update_gpu_progress(elapsed, "completed")
update_experiment_state("completed")

logger.info(f"All done. Results at: {output_path}")
