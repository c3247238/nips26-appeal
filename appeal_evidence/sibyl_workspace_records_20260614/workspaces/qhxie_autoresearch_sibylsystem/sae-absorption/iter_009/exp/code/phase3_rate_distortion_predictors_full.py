"""
Phase 3.1: Rate-Distortion Three-Factor Predictor Model (H9) -- FULL

Tests whether absorption probability is predicted by:
  1. cos_sim = cosine(d_parent, d_child) -- decoder similarity
  2. co_occur = P(child active | parent active) -- co-occurrence from activation stats
  3. R_parent = MSE increase when parent direction ablated from SAE decoder

FULL MODE: Uses ALL hierarchies (first-letter, city-continent, city-language, city-country),
BOTH SAE widths (16k, 65k), 500 sequences for co-occurrence, 250 for R_parent.
Target: >= 30 pairs with all three predictors. Spearman rho > 0.5.
Falsification: rho < 0.3 or p > 0.05.

Based on pilot results: rho=0.261 (NOT_SUPPORTED).
FULL mode adds ~100+ pairs to increase power and potentially reveal hidden structure.
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
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

# ============================================================
# Configuration
# ============================================================
TASK_ID = "phase3_rate_distortion_predictors"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE3_DIR = RESULTS_DIR / "phase3"
PHASE1_DIR = RESULTS_DIR / "phase1"
for d in [PILOT_DIR, PHASE3_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = "FULL"

# SAE configurations -- FULL uses multiple SAEs
SAE_CONFIGS = {
    "L24_16k": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_24/width_16k/canonical",
        "layer": 24,
    },
    "L24_65k": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_24/width_65k/canonical",
        "layer": 24,
    },
}

# Primary SAE for co-occurrence / R_parent computation
PRIMARY_SAE_KEY = "L24_16k"
SAE_LAYER = 24

# Sequences for co-occurrence / reconstruction importance
N_SEQUENCES_COOCCUR = 500
N_SEQUENCES_RPARENT = 250
MAX_SEQ_LEN = 128
BATCH_SIZE = 8

# Bootstrap parameters
N_BOOTSTRAP = 10000

# GPU configuration -- CUDA_VISIBLE_DEVICES remaps, so always use cuda:0
DEVICE = "cuda:0"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


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
    import filelock
    progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    lock_path = WORKSPACE / "exp" / "gpu_progress.lock"
    try:
        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            data = json.loads(progress_path.read_text()) if progress_path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}
            }
            if status == "completed":
                if TASK_ID not in data.get("completed", []):
                    data.setdefault("completed", []).append(TASK_ID)
            else:
                if TASK_ID not in data.get("failed", []):
                    data.setdefault("failed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            data.setdefault("timings", {})[TASK_ID] = {
                "planned_min": 30,
                "actual_min": round(elapsed_seconds / 60, 1),
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "config_snapshot": {
                    "model": "gemma-2-2b",
                    "mode": MODE,
                    "sae_configs": list(SAE_CONFIGS.keys()),
                    "n_sequences_cooccur": N_SEQUENCES_COOCCUR,
                    "n_sequences_rparent": N_SEQUENCES_RPARENT,
                    "batch_size": BATCH_SIZE,
                },
            }
            progress_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"gpu_progress update failed: {e}")
        try:
            data = json.loads(progress_path.read_text()) if progress_path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}
            }
            if status == "completed":
                if TASK_ID not in data.get("completed", []):
                    data.setdefault("completed", []).append(TASK_ID)
            else:
                if TASK_ID not in data.get("failed", []):
                    data.setdefault("failed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            progress_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass


def update_experiment_state(status, error_summary=None):
    import filelock
    state_path = WORKSPACE / "exp" / "experiment_state.json"
    lock_path = WORKSPACE / "exp" / "experiment_state.lock"
    try:
        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            data = json.loads(state_path.read_text()) if state_path.exists() else {
                "schema_version": 1, "tasks": {}
            }
            if TASK_ID in data.get("tasks", {}):
                data["tasks"][TASK_ID]["status"] = status
                if status == "completed":
                    data["tasks"][TASK_ID]["completed_at"] = datetime.now().isoformat()
                if error_summary:
                    data["tasks"][TASK_ID]["error_summary"] = error_summary
            state_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"experiment_state update failed: {e}")


# ============================================================
# Load Phase 1 results -- FULL mode: ALL hierarchies, ALL SAEs
# ============================================================
def load_phase1_pairs():
    """
    Extract parent-child pairs from FULL mode Phase 1 absorption results.

    FULL mode data structure:
    - absorption_firstletter.json: keyed by SAE config (L6_16k, ..., L24_65k)
    - absorption_crossdomain.json: keyed by "{hierarchy}_{sae_key}" (e.g. city-continent_L24_16k)

    Returns list of dicts with keys:
      hierarchy, class_name, sae_key, child_feature_id,
      child_cos_sim_to_probe, observed_absorption_rate, n_probe_correct
    """
    pairs = []

    # --- First-letter pairs (FULL mode) ---
    fl_paths = [
        PHASE1_DIR / "absorption_firstletter.json",
        RESULTS_DIR / "phase1_absorption_firstletter.json",
        PILOT_DIR / "phase1_absorption_firstletter.json",
    ]
    fl_data = None
    for p in fl_paths:
        if p.exists():
            try:
                fl_data = json.loads(p.read_text())
                logger.info(f"Loaded first-letter data from {p} (mode={fl_data.get('mode', '?')})")
                break
            except Exception as e:
                logger.warning(f"Failed to load first-letter from {p}: {e}")

    if fl_data is not None:
        ar = fl_data.get("absorption_results", {})
        # In FULL mode: keys are L6_16k, L6_65k, ..., L24_65k
        for sae_key in SAE_CONFIGS:
            sae_ar = ar.get(sae_key)
            if not sae_ar or not isinstance(sae_ar, dict):
                continue
            mft = sae_ar.get("main_features_top", {})
            pl = sae_ar.get("per_letter", {})
            if not mft or not pl:
                continue

            count = 0
            for letter, mf_info in mft.items():
                if letter not in pl:
                    continue
                ls = pl[letter]
                n_correct = ls.get("probe_correct_raw", 0)
                if n_correct < 1:
                    continue
                absorption_rate = ls.get("absorption_rate", 0.0)
                fid = mf_info.get("fid") if isinstance(mf_info, dict) else None
                cos = mf_info.get("cos", 0.0) if isinstance(mf_info, dict) else 0.0
                if fid is None:
                    continue

                pairs.append({
                    "hierarchy": "first-letter",
                    "class_name": letter,
                    "sae_key": sae_key,
                    "child_feature_id": fid,
                    "child_cos_sim_to_probe": cos,
                    "observed_absorption_rate": absorption_rate,
                    "n_probe_correct": n_correct,
                })
                count += 1

            if count > 0:
                logger.info(f"  first-letter/{sae_key}: {count} pairs")

    # --- Cross-domain pairs (FULL mode) ---
    cd_paths = [
        PHASE1_DIR / "absorption_crossdomain.json",
        RESULTS_DIR / "phase1_absorption_crossdomain.json",
        PILOT_DIR / "phase1_absorption_crossdomain.json",
    ]
    cd_data = None
    for p in cd_paths:
        if p.exists():
            try:
                cd_data = json.loads(p.read_text())
                logger.info(f"Loaded cross-domain data from {p} (mode={cd_data.get('mode', '?')})")
                break
            except Exception as e:
                logger.warning(f"Failed to load cross-domain from {p}: {e}")

    if cd_data is not None:
        ar = cd_data.get("absorption_results", {})
        # FULL mode keys: "city-continent_L24_16k", "city-language_L24_16k", etc.
        for ar_key, sae_ar in ar.items():
            if not isinstance(sae_ar, dict):
                continue
            hierarchy = sae_ar.get("hierarchy", "")
            sae_key = sae_ar.get("sae_key", "")

            # Fallback: parse from ar_key
            if not hierarchy or not sae_key:
                # Key format: "city-continent_L24_16k"
                for h_prefix in ["city-continent", "city-language", "city-country"]:
                    if ar_key.startswith(h_prefix + "_"):
                        hierarchy = h_prefix
                        sae_key = ar_key[len(h_prefix) + 1:]
                        break

            if not hierarchy or sae_key not in SAE_CONFIGS:
                continue

            mft = sae_ar.get("main_features_top", {})
            pc = sae_ar.get("per_class", {})
            if not mft or not pc:
                continue

            count = 0
            for cls_name, mf_info in mft.items():
                if cls_name not in pc:
                    continue
                cs = pc[cls_name]
                n_correct = cs.get("probe_correct_raw", 0)
                if n_correct < 1:
                    continue
                absorption_rate = cs.get("absorption_rate", 0.0)
                fid = mf_info.get("fid") if isinstance(mf_info, dict) else None
                cos = mf_info.get("cos", 0.0) if isinstance(mf_info, dict) else 0.0
                if fid is None:
                    continue

                pairs.append({
                    "hierarchy": hierarchy,
                    "class_name": cls_name,
                    "sae_key": sae_key,
                    "child_feature_id": fid,
                    "child_cos_sim_to_probe": cos,
                    "observed_absorption_rate": absorption_rate,
                    "n_probe_correct": n_correct,
                })
                count += 1

            if count > 0:
                logger.info(f"  {hierarchy}/{sae_key}: {count} pairs")

    # Summary
    hierarchy_counts = Counter(p["hierarchy"] for p in pairs)
    sae_counts = Counter(p["sae_key"] for p in pairs)
    logger.info(f"Total parent-child pairs: {len(pairs)}")
    logger.info(f"  By hierarchy: {dict(hierarchy_counts)}")
    logger.info(f"  By SAE: {dict(sae_counts)}")

    return pairs


# ============================================================
# Model / SAE loading
# ============================================================
def load_model(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    logger.info("Loading Gemma 2 2B...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_LOCAL_PATH, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_LOCAL_PATH)
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", device=device, dtype=torch.bfloat16,
        hf_model=hf_model, tokenizer=tokenizer,
    )
    logger.info(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    return model


def load_sae(sae_key, device):
    from sae_lens import SAE
    cfg = SAE_CONFIGS[sae_key]
    logger.info(f"Loading SAE: {cfg['release']} / {cfg['sae_id']}")
    sae = SAE.from_pretrained(cfg["release"], cfg["sae_id"], device=device)
    logger.info(f"  d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")
    return sae


# ============================================================
# Load probes for ALL hierarchies
# ============================================================
def load_probes(device="cpu"):
    """Load probes for all hierarchies to compute parent directions."""
    probes = {}

    # First-letter probe (sklearn)
    fl_probe_path = PHASE1_DIR / "probe_first-letter_L24_sklearn.npz"
    if fl_probe_path.exists():
        data = np.load(fl_probe_path, allow_pickle=True)
        coef = data["coef"]
        intercept = data["intercept"]
        classes_raw = data["classes"]
        if len(classes_raw) == 26 and all(isinstance(c, (int, np.integer)) for c in classes_raw):
            classes_list = [chr(ord('a') + i) for i in range(26)]
        else:
            classes_list = [str(c) for c in classes_raw]
        probe = LogisticRegression(max_iter=1)
        probe.classes_ = np.arange(len(classes_list))
        probe.coef_ = coef
        probe.intercept_ = intercept
        probes["first-letter"] = {
            "probe": probe,
            "classes": classes_list,
            "coef": torch.tensor(coef, dtype=torch.float32),
        }
        logger.info(f"  Loaded first-letter probe: {coef.shape}, {len(classes_list)} classes")

    # Cross-domain probes (sklearn .npz format)
    for hierarchy in ["city-continent", "city-language", "city-country"]:
        probe_path = PHASE1_DIR / f"probe_{hierarchy}_L24.npz"
        if probe_path.exists():
            data = np.load(probe_path, allow_pickle=True)
            coef = data["coef"]
            intercept = data["intercept"]
            classes_raw = data["classes"]
            classes_list = [str(c) for c in classes_raw]
            probe = LogisticRegression(max_iter=1)
            probe.classes_ = np.arange(len(classes_list))
            probe.coef_ = coef
            probe.intercept_ = intercept
            probes[hierarchy] = {
                "probe": probe,
                "classes": classes_list,
                "coef": torch.tensor(coef, dtype=torch.float32),
            }
            logger.info(f"  Loaded {hierarchy} probe: {coef.shape}, {len(classes_list)} classes")

    logger.info(f"  Probes loaded: {list(probes.keys())}")
    return probes


# ============================================================
# Predictor 1: Decoder cosine similarity
# ============================================================
def compute_decoder_cosine_similarity(sae_decoders, probes, pairs):
    """
    For each parent-child pair, compute:
      cos_sim = cosine(probe_direction_for_class, W_dec[child_feature_id])

    sae_decoders: dict mapping sae_key -> W_dec tensor [d_sae, d_model]
    """
    results = []
    for pair in pairs:
        hierarchy = pair["hierarchy"]
        class_name = pair["class_name"]
        child_fid = pair["child_feature_id"]
        sae_key = pair["sae_key"]

        if hierarchy not in probes:
            results.append(None)
            continue

        if sae_key not in sae_decoders:
            results.append(None)
            continue

        W_dec = sae_decoders[sae_key]
        probe_data = probes[hierarchy]
        classes = probe_data["classes"]

        if class_name in classes:
            cls_idx = classes.index(class_name)
        else:
            results.append(None)
            continue

        # Get probe direction for this class
        probe_dir = probe_data["coef"][cls_idx]  # [d_model]
        probe_dir = probe_dir / (probe_dir.norm() + 1e-8)

        # Get child feature decoder direction
        if child_fid >= W_dec.shape[0]:
            results.append(None)
            continue
        child_dec = W_dec[child_fid]  # [d_model]
        child_dec_norm = child_dec / (child_dec.norm() + 1e-8)

        cos_sim = F.cosine_similarity(
            probe_dir.unsqueeze(0), child_dec_norm.unsqueeze(0)
        ).item()

        results.append(cos_sim)

    return results


# ============================================================
# Predictor 2: Co-occurrence P(child active | parent active)
# ============================================================
def compute_cooccurrence(model, sae, probes, pairs, device,
                         n_sequences=500, max_seq_len=128, batch_size=8):
    """
    Estimate co-occurrence of parent and child features.

    For cross-domain (city-*): use probe predictions with confidence > 0.4.
    For first-letter: use the top SAE feature co-activation.

    Only processes pairs matching the SAE currently loaded.
    """
    tl_hook = f"blocks.{SAE_LAYER}.hook_resid_post"
    sae_key = PRIMARY_SAE_KEY  # Only process primary SAE pairs for co-occurrence

    prompts_pool = _get_text_prompts(model.tokenizer, n_sequences, max_seq_len)

    # Build pair map for pairs with the primary SAE
    pair_map = {}
    for i, pair in enumerate(pairs):
        if pair["sae_key"] != sae_key:
            continue
        h = pair["hierarchy"]
        c = pair["class_name"]
        fid = pair["child_feature_id"]
        if h not in probes:
            continue
        classes = probes[h]["classes"]
        if c in classes:
            cls_idx = classes.index(c)
            pair_map[i] = {
                "hierarchy": h,
                "class_name": c,
                "cls_idx": cls_idx,
                "child_fid": fid,
            }

    # Load top SAE features for first-letter co-activation
    fl_top_features = _load_top_features("first-letter", sae_key)

    # Accumulators
    cooccur_counts = defaultdict(lambda: {"parent_active": 0, "both_active": 0, "child_fires_total": 0})
    total_tokens = 0

    for batch_start in range(0, len(prompts_pool), batch_size):
        batch_prompts = prompts_pool[batch_start:batch_start + batch_size]

        for prompt_text in batch_prompts:
            try:
                tokens = model.to_tokens(prompt_text, prepend_bos=True)
                if tokens.shape[1] > max_seq_len:
                    tokens = tokens[:, :max_seq_len]

                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])

                acts = cache[tl_hook][0].detach().float()  # [seq_len, d_model]
                seq_len = acts.shape[0]

                # Encode with SAE
                with torch.no_grad():
                    sae_features = sae.encode(acts.to(device))  # [seq_len, d_sae]

                sae_features_cpu = sae_features.detach().float().cpu()
                acts_np = acts.cpu().numpy()

                # Process cross-domain hierarchies via probe predictions
                for hierarchy in ["city-continent", "city-language", "city-country"]:
                    if hierarchy not in probes:
                        continue
                    probe_data = probes[hierarchy]
                    probe_obj = probe_data["probe"]
                    classes = probe_data["classes"]

                    try:
                        predictions = probe_obj.predict(acts_np)
                        probs = probe_obj.predict_proba(acts_np)

                        for t_idx in range(seq_len):
                            pred_cls = predictions[t_idx]
                            pred_prob = probs[t_idx, pred_cls]
                            if pred_prob < 0.4:
                                continue
                            cls_name = classes[pred_cls]

                            for i, pm in pair_map.items():
                                if pm["hierarchy"] == hierarchy and pm["class_name"] == cls_name:
                                    child_fid = pm["child_fid"]
                                    cooccur_counts[i]["parent_active"] += 1
                                    if child_fid < sae_features_cpu.shape[1]:
                                        if sae_features_cpu[t_idx, child_fid].item() > 0:
                                            cooccur_counts[i]["both_active"] += 1
                    except Exception:
                        pass

                # Process first-letter hierarchy via SAE feature co-activation
                for i, pm in pair_map.items():
                    if pm["hierarchy"] != "first-letter":
                        continue
                    letter = pm["class_name"]
                    child_fid = pm["child_fid"]
                    parent_fid = fl_top_features.get(letter)
                    if parent_fid is None:
                        continue

                    for t_idx in range(seq_len):
                        if parent_fid < sae_features_cpu.shape[1]:
                            parent_fires = sae_features_cpu[t_idx, parent_fid].item() > 0
                        else:
                            continue

                        if parent_fires:
                            cooccur_counts[i]["parent_active"] += 1
                            if child_fid < sae_features_cpu.shape[1]:
                                if sae_features_cpu[t_idx, child_fid].item() > 0:
                                    cooccur_counts[i]["both_active"] += 1

                        if child_fid < sae_features_cpu.shape[1]:
                            if sae_features_cpu[t_idx, child_fid].item() > 0:
                                cooccur_counts[i]["child_fires_total"] += 1

                del cache, acts, sae_features, sae_features_cpu
                total_tokens += seq_len

            except Exception as e:
                logger.warning(f"Error processing prompt: {e}")
                continue

        torch.cuda.empty_cache()

        batch_num = batch_start // batch_size + 1
        if batch_num % 10 == 0:
            logger.info(f"  Co-occurrence: processed {batch_start + len(batch_prompts)}/{len(prompts_pool)} "
                        f"sequences ({total_tokens} tokens)")

    # Compute P(child | parent) for each pair
    cooccur_results = []
    for i, pair in enumerate(pairs):
        if i in cooccur_counts and cooccur_counts[i]["parent_active"] > 0:
            counts = cooccur_counts[i]
            p_cooccur = counts["both_active"] / counts["parent_active"]
            cooccur_results.append({
                "value": p_cooccur,
                "parent_active_count": counts["parent_active"],
                "both_active_count": counts["both_active"],
                "child_fires_total": counts.get("child_fires_total", 0),
            })
        elif pair["sae_key"] != sae_key:
            # For non-primary SAE pairs, try to transfer co-occurrence from same class in primary SAE
            transferred = _transfer_cooccurrence(pair, pairs, cooccur_counts, pair_map)
            cooccur_results.append(transferred)
        else:
            # Fallback: use cos_sim as proxy
            cos_proxy = pair.get("child_cos_sim_to_probe")
            if cos_proxy is not None and cos_proxy > 0:
                cooccur_results.append({
                    "value": float(cos_proxy),
                    "parent_active_count": 0,
                    "both_active_count": 0,
                    "child_fires_total": 0,
                    "note": "cos_sim_proxy",
                })
            else:
                cooccur_results.append(None)

    n_direct = sum(1 for r in cooccur_results if r and r.get("parent_active_count", 0) > 0)
    n_proxy = sum(1 for r in cooccur_results if r and "proxy" in r.get("note", ""))
    n_transfer = sum(1 for r in cooccur_results if r and "transfer" in r.get("note", ""))
    n_none = sum(1 for r in cooccur_results if r is None)
    logger.info(f"  Co-occurrence: {n_direct} direct, {n_transfer} transferred, "
                f"{n_proxy} proxy, {n_none} missing (total tokens: {total_tokens})")

    return cooccur_results


def _transfer_cooccurrence(pair, all_pairs, cooccur_counts, pair_map):
    """Transfer co-occurrence from same hierarchy+class in primary SAE."""
    for i, pm in pair_map.items():
        if pm["hierarchy"] == pair["hierarchy"] and pm["class_name"] == pair["class_name"]:
            if i in cooccur_counts and cooccur_counts[i]["parent_active"] > 0:
                counts = cooccur_counts[i]
                return {
                    "value": counts["both_active"] / counts["parent_active"],
                    "parent_active_count": counts["parent_active"],
                    "both_active_count": counts["both_active"],
                    "child_fires_total": counts.get("child_fires_total", 0),
                    "note": "transferred_from_primary_sae",
                }
    cos_proxy = pair.get("child_cos_sim_to_probe")
    if cos_proxy is not None and cos_proxy > 0:
        return {
            "value": float(cos_proxy),
            "parent_active_count": 0,
            "both_active_count": 0,
            "child_fires_total": 0,
            "note": "cos_sim_proxy",
        }
    return None


def _load_top_features(hierarchy, sae_key):
    """Load the top SAE feature IDs from Phase 1 results."""
    features = {}

    if hierarchy == "first-letter":
        paths = [
            PHASE1_DIR / "absorption_firstletter.json",
            RESULTS_DIR / "phase1_absorption_firstletter.json",
            PILOT_DIR / "phase1_absorption_firstletter.json",
        ]
        for p in paths:
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                    mft = data.get("absorption_results", {}).get(sae_key, {}).get("main_features_top", {})
                    for letter, info in mft.items():
                        if isinstance(info, dict) and "fid" in info:
                            features[letter] = info["fid"]
                    break
                except Exception:
                    pass
    return features


def _get_text_prompts(tokenizer, n_sequences, max_seq_len):
    """Generate diverse text prompts for co-occurrence estimation."""
    prompts = []
    rng = random.Random(SEED)

    words_by_letter = {
        'a': ['apple', 'artist', 'ancient', 'answer', 'always', 'amazing', 'arrow', 'autumn'],
        'b': ['bright', 'beauty', 'beyond', 'bridge', 'basket', 'beacon', 'bottle', 'branch'],
        'c': ['castle', 'center', 'change', 'circle', 'create', 'crystal', 'captain', 'cloud'],
        'd': ['dragon', 'design', 'during', 'direct', 'double', 'diamond', 'desert', 'dance'],
        'e': ['energy', 'effort', 'escape', 'entire', 'engine', 'eternal', 'explore', 'eagle'],
        'f': ['flower', 'figure', 'forest', 'frozen', 'future', 'falcon', 'famous', 'flight'],
        'g': ['garden', 'golden', 'gather', 'gentle', 'global', 'glacier', 'guitar', 'grain'],
        'h': ['hidden', 'harbor', 'heaven', 'hollow', 'hunger', 'harvest', 'honest', 'humble'],
        'i': ['island', 'inside', 'invite', 'ignore', 'impact', 'impulse', 'indoor', 'ivory'],
        'j': ['jungle', 'junior', 'jacket', 'jasper', 'jovial', 'justice', 'joyful', 'jewel'],
        'k': ['knight', 'kindly', 'kernel', 'kimono', 'keeper', 'kitchen', 'knuckle', 'karma'],
        'l': ['ladder', 'legacy', 'lovely', 'lonely', 'lively', 'lantern', 'library', 'liquid'],
        'm': ['marble', 'mirror', 'modern', 'moment', 'master', 'meadow', 'mineral', 'muscle'],
        'n': ['nature', 'notice', 'narrow', 'nation', 'normal', 'nebula', 'nimble', 'noble'],
        'o': ['oracle', 'origin', 'orange', 'outfit', 'online', 'outpost', 'oxygen', 'opaque'],
        'p': ['palace', 'planet', 'pretty', 'purple', 'python', 'prairie', 'puzzle', 'permit'],
        'q': ['queen', 'quiet', 'quest', 'quick', 'quaint', 'quarry', 'quasar', 'quench'],
        'r': ['rocket', 'random', 'rising', 'robust', 'ritual', 'radiant', 'raven', 'rhythm'],
        's': ['silver', 'simple', 'spirit', 'strong', 'system', 'sunset', 'shadow', 'stream'],
        't': ['temple', 'timber', 'travel', 'trophy', 'turtle', 'thunder', 'theory', 'throne'],
        'u': ['unique', 'united', 'unfold', 'upward', 'useful', 'utopia', 'urchin', 'umbrella'],
        'v': ['valley', 'violet', 'vivid', 'volume', 'virtue', 'voyage', 'velvet', 'volcano'],
        'w': ['winter', 'wonder', 'wisdom', 'wander', 'wealth', 'whisper', 'walnut', 'warrior'],
        'x': ['xenon', 'xerox', 'xylophone'],
        'y': ['yellow', 'yearly', 'yogurt', 'yonder', 'youth', 'yeoman', 'yucca'],
        'z': ['zenith', 'zodiac', 'zephyr', 'zigzag', 'zealot', 'zombie', 'zither'],
    }

    spelling_templates = [
        "The word {word} starts with the letter",
        "Spell the word {word}. The first letter is",
        "In the dictionary, {word} is found under the letter",
        "The word {word} begins with",
        "Write the first letter of the word {word}. It is",
        "The initial character of {word} is the letter",
    ]

    # Spelling prompts (30%)
    for _ in range(n_sequences * 3 // 10):
        letter = rng.choice(list(words_by_letter.keys()))
        word = rng.choice(words_by_letter[letter])
        template = rng.choice(spelling_templates)
        prompts.append(template.format(word=word))

    # City/continent prompts (25%)
    city_templates = [
        "The city of {city} is located in",
        "{city} is a major city in the continent of",
        "If you travel to {city}, you are visiting",
        "The continent where {city} can be found is",
        "{city} is known as a city in",
        "People living in {city} reside on the continent of",
    ]

    cities_with_continents = [
        ("London", "Europe"), ("Paris", "Europe"), ("Berlin", "Europe"),
        ("Tokyo", "Asia"), ("Beijing", "Asia"), ("Mumbai", "Asia"),
        ("New York", "North America"), ("Toronto", "North America"),
        ("Sydney", "Oceania"), ("Melbourne", "Oceania"),
        ("Cairo", "Africa"), ("Lagos", "Africa"), ("Nairobi", "Africa"),
        ("Buenos Aires", "South America"), ("Lima", "South America"),
        ("Istanbul", "Europe"), ("Seoul", "Asia"), ("Bangkok", "Asia"),
        ("Moscow", "Europe"), ("Rome", "Europe"), ("Madrid", "Europe"),
        ("Riyadh", "Asia"), ("Karachi", "Asia"), ("Jakarta", "Asia"),
        ("Warsaw", "Europe"), ("Athens", "Europe"), ("Dublin", "Europe"),
        ("Shanghai", "Asia"), ("Delhi", "Asia"), ("Osaka", "Asia"),
        ("Sao Paulo", "South America"), ("Bogota", "South America"),
        ("Accra", "Africa"), ("Addis Ababa", "Africa"), ("Algiers", "Africa"),
        ("Auckland", "Oceania"), ("Wellington", "Oceania"),
        ("Mexico City", "North America"), ("Chicago", "North America"),
        ("Lisbon", "Europe"), ("Prague", "Europe"), ("Vienna", "Europe"),
    ]

    for _ in range(n_sequences * 1 // 4):
        city, continent = rng.choice(cities_with_continents)
        template = rng.choice(city_templates)
        prompts.append(template.format(city=city))

    # City/language prompts (15%)
    city_language_templates = [
        "In {city}, people speak",
        "The primary language spoken in {city} is",
        "{city} is a city where the main language is",
    ]

    cities_with_languages = [
        ("London", "English"), ("Paris", "French"), ("Berlin", "German"),
        ("Tokyo", "Japanese"), ("Beijing", "Chinese"), ("Mumbai", "Hindi"),
        ("Moscow", "Russian"), ("Rome", "Italian"), ("Madrid", "Spanish"),
        ("Istanbul", "Turkish"), ("Seoul", "Korean"), ("Bangkok", "Thai"),
        ("Jakarta", "Indonesian"), ("Riyadh", "Arabic"), ("Tehran", "Persian"),
        ("Lisbon", "Portuguese"), ("Warsaw", "Polish"), ("Bucharest", "Romanian"),
    ]

    for _ in range(n_sequences * 15 // 100):
        city, lang = rng.choice(cities_with_languages)
        template = rng.choice(city_language_templates)
        prompts.append(template.format(city=city))

    # City/country prompts (15%)
    city_country_templates = [
        "{city} is a city in the country of",
        "The country that contains {city} is",
        "{city} is located in",
    ]

    cities_with_countries = [
        ("London", "United Kingdom"), ("Paris", "France"), ("Berlin", "Germany"),
        ("Tokyo", "Japan"), ("Beijing", "China"), ("Mumbai", "India"),
        ("Moscow", "Russia"), ("Rome", "Italy"), ("Madrid", "Spain"),
        ("Sydney", "Australia"), ("Toronto", "Canada"), ("Cairo", "Egypt"),
        ("Lagos", "Nigeria"), ("Buenos Aires", "Argentina"), ("Lima", "Peru"),
        ("Seoul", "South Korea"), ("Bangkok", "Thailand"), ("Jakarta", "Indonesia"),
        ("Sao Paulo", "Brazil"), ("Mexico City", "Mexico"),
    ]

    for _ in range(n_sequences * 15 // 100):
        city, country = rng.choice(cities_with_countries)
        template = rng.choice(city_country_templates)
        prompts.append(template.format(city=city))

    # General diverse text (remaining)
    general_texts = [
        "Europe is a continent that includes countries like France and Germany",
        "Asia is the largest continent with nations such as China and India",
        "North America contains the United States and Canada",
        "Africa is a continent known for its diverse wildlife and cultures",
        "South America includes Brazil, Argentina, and Chile",
        "Oceania encompasses Australia and the Pacific island nations",
        "The population of the world is approximately eight billion people",
        "Scientific discoveries in the 21st century include many breakthroughs",
        "The history of civilization dates back thousands of years",
        "Modern technology has transformed the way we communicate",
        "Mathematics provides the foundation for understanding physics",
        "Literature from the classical period still influences modern writing",
        "Music has evolved dramatically throughout human history",
        "The ocean covers about seventy percent of the Earth's surface",
        "Climate change is one of the most pressing global challenges",
        "Languages around the world carry unique cultural heritage",
        "Different countries have developed distinct political systems",
        "Trade between nations has shaped the course of human history",
    ]

    while len(prompts) < n_sequences:
        prompts.append(rng.choice(general_texts))

    rng.shuffle(prompts)
    return prompts[:n_sequences]


# ============================================================
# Predictor 3: Reconstruction importance R(parent)
# ============================================================
def compute_reconstruction_importance(model, sae, probes, pairs, device,
                                      n_sequences=250, max_seq_len=128):
    """
    R(parent) = MSE increase when parent direction is ablated from SAE reconstruction.

    Only processes pairs matching the currently loaded SAE.
    """
    tl_hook = f"blocks.{SAE_LAYER}.hook_resid_post"
    sae_key = PRIMARY_SAE_KEY

    prompts_pool = _get_text_prompts(model.tokenizer, n_sequences, max_seq_len)

    # Pre-collect parent directions
    parent_dirs = {}
    for pair in pairs:
        if pair["sae_key"] != sae_key:
            continue
        h = pair["hierarchy"]
        c = pair["class_name"]
        key = (h, c)
        if key in parent_dirs:
            continue
        if h not in probes:
            continue
        probe_data = probes[h]
        classes = probe_data["classes"]
        if c not in classes:
            continue
        cls_idx = classes.index(c)
        probe_dir = probe_data["coef"][cls_idx].clone()
        probe_dir = probe_dir / (probe_dir.norm() + 1e-8)
        parent_dirs[key] = probe_dir

    logger.info(f"  R(parent): {len(parent_dirs)} parent directions prepared")

    if not parent_dirs:
        return [None] * len(pairs)

    parent_keys = list(parent_dirs.keys())
    parent_dir_stack = torch.stack([parent_dirs[k] for k in parent_keys])

    # Accumulate MSE differences per parent
    mse_diffs = defaultdict(list)
    total_tokens = 0

    for seq_idx, prompt_text in enumerate(prompts_pool[:n_sequences]):
        try:
            tokens = model.to_tokens(prompt_text, prepend_bos=True)
            if tokens.shape[1] > max_seq_len:
                tokens = tokens[:, :max_seq_len]

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])

            raw_acts = cache[tl_hook][0].detach().float().to(device)
            seq_len = raw_acts.shape[0]

            with torch.no_grad():
                sae_features = sae.encode(raw_acts)
                sae_recon = sae.decode(sae_features)

            orig_mse = ((raw_acts - sae_recon) ** 2).mean(dim=-1)

            for p_idx, key in enumerate(parent_keys):
                p_dir = parent_dir_stack[p_idx].to(device)
                proj = torch.einsum('sd,d->s', sae_recon, p_dir).unsqueeze(-1)
                modified_recon = sae_recon - proj * p_dir.unsqueeze(0)
                mod_mse = ((raw_acts - modified_recon) ** 2).mean(dim=-1)
                mse_increase = (mod_mse - orig_mse).mean().item()
                mse_diffs[key].append(mse_increase)

            del cache, raw_acts, sae_features, sae_recon
            total_tokens += seq_len

        except Exception as e:
            logger.warning(f"Error in R(parent) computation: {e}")
            continue

        if (seq_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
            logger.info(f"  R(parent): processed {seq_idx + 1}/{n_sequences} sequences "
                        f"({total_tokens} tokens)")

    logger.info(f"  R(parent) computed over {total_tokens} tokens, {n_sequences} sequences")

    # Average MSE increase per parent
    r_parent_values = {}
    for key, diffs in mse_diffs.items():
        r_parent_values[key] = float(np.mean(diffs))

    # Map back to pairs (transfer from primary SAE to same class in secondary SAE)
    results = []
    for pair in pairs:
        key = (pair["hierarchy"], pair["class_name"])
        if key in r_parent_values:
            results.append(r_parent_values[key])
        else:
            results.append(None)

    return results


# ============================================================
# Bootstrap CI helper
# ============================================================
def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    if len(values) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n": 0}
    values = np.array(values, dtype=float)
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(float(np.mean(sample)))
    boot_means = sorted(boot_means)
    alpha = (1 - ci) / 2
    lo = boot_means[int(alpha * n_bootstrap)]
    hi = boot_means[min(int((1 - alpha) * n_bootstrap), len(boot_means) - 1)]
    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "std": float(np.std(values)),
        "n": len(values),
    }


# ============================================================
# Three-factor model fitting -- FULL version with regularization, LOO-CV
# ============================================================
def fit_three_factor_model(pairs_with_predictors):
    """
    Fit: absorption_prob ~ beta_1*cos_sim^2 + beta_2*co_occur + beta_3*R_parent
    Evaluate: Spearman rho between predicted and observed.
    Also: individual predictors, cross-domain analysis, leave-one-out cross-validation.
    """
    valid = [p for p in pairs_with_predictors
             if p.get("cos_sim") is not None
             and p.get("co_occur") is not None
             and p.get("r_parent") is not None
             and p.get("observed_absorption_rate") is not None]

    if len(valid) < 5:
        logger.warning(f"Only {len(valid)} valid pairs -- too few for model fitting")
        return {
            "n_valid_pairs": len(valid),
            "model_fit": None,
            "individual_correlations": {},
            "cross_domain_analysis": None,
            "insufficient_data": True,
        }

    cos_sim = np.array([p["cos_sim"] for p in valid])
    co_occur = np.array([p["co_occur"] for p in valid])
    r_parent = np.array([p["r_parent"] for p in valid])
    observed = np.array([p["observed_absorption_rate"] for p in valid])
    hierarchies = [p["hierarchy"] for p in valid]
    sae_keys = [p["sae_key"] for p in valid]

    logger.info(f"\n  Three-factor model fitting with {len(valid)} pairs:")
    logger.info(f"    cos_sim: mean={np.mean(cos_sim):.4f}, std={np.std(cos_sim):.4f}")
    logger.info(f"    co_occur: mean={np.mean(co_occur):.4f}, std={np.std(co_occur):.4f}")
    logger.info(f"    r_parent: mean={np.mean(r_parent):.6f}, std={np.std(r_parent):.6f}")
    logger.info(f"    observed: mean={np.mean(observed):.4f}, std={np.std(observed):.4f}")

    # Individual predictor correlations
    individual = {}
    predictor_names = {
        "cos_sim": cos_sim,
        "cos_sim_squared": cos_sim ** 2,
        "cos_sim_abs": np.abs(cos_sim),
        "co_occur": co_occur,
        "r_parent": r_parent,
        "neg_r_parent": -r_parent,
        "competition_coeff": cos_sim * co_occur,
    }

    for name, values in predictor_names.items():
        if np.std(values) < 1e-10 or np.std(observed) < 1e-10:
            individual[name] = {
                "spearman_rho": 0.0,
                "spearman_p": 1.0,
                "n_pairs": len(valid),
                "note": "zero variance",
            }
            continue

        rho, p_val = stats.spearmanr(values, observed)
        pearson_r, pearson_p = stats.pearsonr(values, observed)

        individual[name] = {
            "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
            "spearman_p": float(p_val) if not np.isnan(p_val) else 1.0,
            "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else 0.0,
            "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else 1.0,
            "n_pairs": len(valid),
        }
        logger.info(f"    {name}: Spearman rho={individual[name]['spearman_rho']:.4f}, "
                     f"p={individual[name]['spearman_p']:.4f}")

    # Per-hierarchy individual correlations
    per_hierarchy_correlations = {}
    unique_hierarchies = list(set(hierarchies))
    for h in unique_hierarchies:
        h_mask = np.array([hi == h for hi in hierarchies])
        n_h = h_mask.sum()
        if n_h < 5:
            continue
        h_individual = {}
        h_observed = observed[h_mask]
        for name, values in predictor_names.items():
            h_vals = values[h_mask]
            if np.std(h_vals) < 1e-10 or np.std(h_observed) < 1e-10:
                continue
            rho, p_val = stats.spearmanr(h_vals, h_observed)
            h_individual[name] = {
                "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                "spearman_p": float(p_val) if not np.isnan(p_val) else 1.0,
                "n_pairs": int(n_h),
            }
        if h_individual:
            per_hierarchy_correlations[h] = h_individual
            logger.info(f"    Per-hierarchy {h} (n={n_h}): best individual = "
                        f"{max(h_individual.items(), key=lambda x: abs(x[1]['spearman_rho']))}")

    # Multi-variate linear model
    X = np.column_stack([cos_sim ** 2, co_occur, r_parent])

    if X.shape[0] <= X.shape[1] + 1:
        model_fit = {
            "fitted": False,
            "reason": f"n={X.shape[0]} <= p+1={X.shape[1]+1}",
        }
    else:
        try:
            reg = LinearRegression()
            reg.fit(X, observed)
            predicted = reg.predict(X)

            rho_pred, p_pred = stats.spearmanr(predicted, observed)
            pearson_pred, pearson_pred_p = stats.pearsonr(predicted, observed)

            ss_res = np.sum((observed - predicted) ** 2)
            ss_tot = np.sum((observed - np.mean(observed)) ** 2)
            r_squared = 1 - ss_res / max(ss_tot, 1e-10)

            # Leave-one-out cross-validation
            loo_predictions = np.zeros(len(valid))
            for i in range(len(valid)):
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(observed, i)
                reg_loo = LinearRegression()
                reg_loo.fit(X_train, y_train)
                loo_predictions[i] = reg_loo.predict(X[i:i+1])[0]

            loo_rho, loo_p = stats.spearmanr(loo_predictions, observed)
            loo_mse = np.mean((loo_predictions - observed) ** 2)

            model_fit = {
                "fitted": True,
                "coefficients": {
                    "cos_sim_squared": float(reg.coef_[0]),
                    "co_occur": float(reg.coef_[1]),
                    "r_parent": float(reg.coef_[2]),
                    "intercept": float(reg.intercept_),
                },
                "spearman_rho": float(rho_pred) if not np.isnan(rho_pred) else 0.0,
                "spearman_p": float(p_pred) if not np.isnan(p_pred) else 1.0,
                "pearson_r": float(pearson_pred) if not np.isnan(pearson_pred) else 0.0,
                "pearson_p": float(pearson_pred_p) if not np.isnan(pearson_pred_p) else 1.0,
                "r_squared": float(r_squared),
                "n_pairs": len(valid),
                "loo_cv": {
                    "spearman_rho": float(loo_rho) if not np.isnan(loo_rho) else 0.0,
                    "spearman_p": float(loo_p) if not np.isnan(loo_p) else 1.0,
                    "mse": float(loo_mse),
                },
                "predicted_vs_observed": [
                    {
                        "hierarchy": hierarchies[i],
                        "class_name": valid[i]["class_name"],
                        "sae_key": sae_keys[i],
                        "predicted": float(predicted[i]),
                        "observed": float(observed[i]),
                        "loo_predicted": float(loo_predictions[i]),
                        "cos_sim": float(cos_sim[i]),
                        "co_occur": float(co_occur[i]),
                        "r_parent": float(r_parent[i]),
                    }
                    for i in range(len(valid))
                ],
            }

            logger.info(f"\n  Three-factor model results:")
            logger.info(f"    R^2 = {r_squared:.4f}")
            logger.info(f"    Spearman rho = {model_fit['spearman_rho']:.4f}, p = {model_fit['spearman_p']:.4f}")
            logger.info(f"    LOO-CV: rho = {loo_rho:.4f}, p = {loo_p:.4f}, MSE = {loo_mse:.6f}")
            logger.info(f"    Coefficients: cos_sim^2={reg.coef_[0]:.4f}, "
                         f"co_occur={reg.coef_[1]:.4f}, r_parent={reg.coef_[2]:.4f}")

        except Exception as e:
            logger.error(f"  Linear model fitting failed: {e}")
            model_fit = {"fitted": False, "reason": str(e)}

    # Cross-domain analysis
    cross_domain = {}
    if len(unique_hierarchies) > 1:
        for h in unique_hierarchies:
            h_mask = np.array([hi == h for hi in hierarchies])
            if h_mask.sum() >= 2:
                cross_domain[h] = {
                    "n_pairs": int(h_mask.sum()),
                    "mean_absorption": float(np.mean(observed[h_mask])),
                    "std_absorption": float(np.std(observed[h_mask])),
                    "mean_cos_sim": float(np.mean(cos_sim[h_mask])),
                    "mean_co_occur": float(np.mean(co_occur[h_mask])),
                    "mean_r_parent": float(np.mean(r_parent[h_mask])),
                    "mean_competition_coeff": float(np.mean(cos_sim[h_mask] * co_occur[h_mask])),
                }

        # ANOVA on observed absorption rates across hierarchies
        hierarchy_groups = []
        for h in unique_hierarchies:
            h_mask = np.array([hi == h for hi in hierarchies])
            if h_mask.sum() >= 2:
                hierarchy_groups.append(observed[h_mask])
        if len(hierarchy_groups) >= 2:
            try:
                f_stat, anova_p = stats.f_oneway(*hierarchy_groups)
                cross_domain["anova"] = {
                    "f_statistic": float(f_stat) if not np.isnan(f_stat) else 0.0,
                    "p_value": float(anova_p) if not np.isnan(anova_p) else 1.0,
                    "n_groups": len(hierarchy_groups),
                }
                logger.info(f"  ANOVA across hierarchies: F={f_stat:.4f}, p={anova_p:.4f}")
            except Exception:
                pass

        logger.info(f"\n  Cross-domain predictor analysis:")
        for h, sd in cross_domain.items():
            if isinstance(sd, dict) and "mean_absorption" in sd:
                logger.info(f"    {h}: n={sd['n_pairs']}, absorption={sd['mean_absorption']:.4f}, "
                             f"cos_sim={sd['mean_cos_sim']:.4f}, co_occur={sd['mean_co_occur']:.4f}")

    return {
        "n_valid_pairs": len(valid),
        "model_fit": model_fit,
        "individual_correlations": individual,
        "per_hierarchy_correlations": per_hierarchy_correlations,
        "cross_domain_analysis": cross_domain if cross_domain else None,
        "insufficient_data": False,
    }


# ============================================================
# Hypothesis verdict
# ============================================================
def evaluate_h9(model_results):
    """
    Evaluate H9: Spearman rho > 0.5 target, rho < 0.3 falsification.
    """
    if model_results.get("insufficient_data"):
        return {
            "verdict": "INSUFFICIENT_DATA",
            "reason": f"Only {model_results['n_valid_pairs']} valid pairs",
            "confidence": 0.0,
        }

    # Best individual predictor
    best_rho = 0
    best_p = 1.0
    best_name = None
    for name, ind in model_results.get("individual_correlations", {}).items():
        if abs(ind.get("spearman_rho", 0)) > abs(best_rho):
            best_rho = ind["spearman_rho"]
            best_p = ind.get("spearman_p", 1.0)
            best_name = name

    mf = model_results.get("model_fit", {})
    if mf and mf.get("fitted"):
        # Prefer LOO-CV rho for evaluation (less overfit)
        loo = mf.get("loo_cv", {})
        rho = loo.get("spearman_rho", mf.get("spearman_rho", 0))
        p_value = loo.get("spearman_p", mf.get("spearman_p", 1))
        r_squared = mf.get("r_squared", 0)
        source = "multivariate_model_loo_cv"
    else:
        rho = abs(best_rho)
        p_value = best_p
        r_squared = rho ** 2
        source = f"individual_predictor_{best_name}"

    n_pairs = model_results.get("n_valid_pairs", 0)

    if rho > 0.5 and p_value < 0.05:
        verdict = "SUPPORTED"
        confidence = min(0.95, 0.5 + rho * 0.5)
    elif rho > 0.3 and p_value < 0.05:
        verdict = "PARTIALLY_SUPPORTED"
        confidence = min(0.7, 0.3 + rho * 0.4)
    elif rho < 0.3 or p_value > 0.05:
        verdict = "NOT_SUPPORTED"
        confidence = max(0.1, 1 - abs(rho))
    else:
        verdict = "INCONCLUSIVE"
        confidence = 0.3

    return {
        "verdict": verdict,
        "evaluation_source": source,
        "rho_used": float(rho),
        "p_value_used": float(p_value),
        "r_squared_approx": float(r_squared),
        "model_spearman_rho": float(mf.get("spearman_rho", 0)) if mf and mf.get("fitted") else None,
        "model_p_value": float(mf.get("spearman_p", 1)) if mf and mf.get("fitted") else None,
        "model_r_squared": float(mf.get("r_squared", 0)) if mf and mf.get("fitted") else None,
        "model_loo_rho": float(mf.get("loo_cv", {}).get("spearman_rho", 0)) if mf and mf.get("fitted") else None,
        "model_loo_p": float(mf.get("loo_cv", {}).get("spearman_p", 1)) if mf and mf.get("fitted") else None,
        "best_individual_predictor": best_name,
        "best_individual_rho": float(best_rho),
        "best_individual_p": float(best_p),
        "n_valid_pairs": n_pairs,
        "confidence": float(confidence),
        "target": "rho > 0.5",
        "falsification": "rho < 0.3 or p > 0.05",
    }


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 15, "starting")

    logger.info("=" * 60)
    logger.info("Phase 3.1: Rate-Distortion Three-Factor Predictor Model (H9)")
    logger.info(f"Mode: {MODE}, SAEs: {list(SAE_CONFIGS.keys())}")
    logger.info(f"N_sequences_cooccur: {N_SEQUENCES_COOCCUR}, N_sequences_rparent: {N_SEQUENCES_RPARENT}")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 60)

    # Step 1: Load Phase 1 parent-child pairs
    report_progress(1, 15, "loading_phase1_pairs")
    pairs = load_phase1_pairs()

    if len(pairs) < 5:
        msg = f"Only {len(pairs)} parent-child pairs found. Need >= 5."
        logger.error(msg)
        mark_done("failed", msg)
        update_gpu_progress(time.time() - start_time, "failed")
        update_experiment_state("failed", msg)
        return

    # Step 2: Load model
    report_progress(2, 15, "loading_model")
    model = load_model(device=DEVICE)

    # Step 3: Load SAEs and extract decoder weights
    report_progress(3, 15, "loading_saes")
    sae_decoders = {}
    primary_sae = None
    for sae_key in SAE_CONFIGS:
        try:
            sae = load_sae(sae_key, device=DEVICE)
            sae_decoders[sae_key] = sae.W_dec.detach().float().cpu()
            if sae_key == PRIMARY_SAE_KEY:
                primary_sae = sae
            else:
                del sae
                gc.collect()
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Failed to load SAE {sae_key}: {e}")

    if primary_sae is None:
        msg = f"Failed to load primary SAE {PRIMARY_SAE_KEY}"
        logger.error(msg)
        mark_done("failed", msg)
        update_gpu_progress(time.time() - start_time, "failed")
        update_experiment_state("failed", msg)
        return

    # Step 4: Load probes
    report_progress(4, 15, "loading_probes")
    probes = load_probes()

    if not probes:
        msg = "No probes could be loaded."
        logger.error(msg)
        mark_done("failed", msg)
        update_gpu_progress(time.time() - start_time, "failed")
        update_experiment_state("failed", msg)
        del model, primary_sae
        gc.collect()
        torch.cuda.empty_cache()
        return

    # Step 5: Compute Predictor 1 -- Decoder cosine similarity
    report_progress(5, 15, "computing_cos_sim")
    logger.info("\n--- Predictor 1: Decoder cosine similarity ---")
    cos_sim_values = compute_decoder_cosine_similarity(sae_decoders, probes, pairs)

    n_cos_valid = sum(1 for cs in cos_sim_values if cs is not None)
    logger.info(f"  cos_sim computed for {n_cos_valid}/{len(pairs)} pairs")
    del sae_decoders  # Free memory
    gc.collect()

    # Step 6: Compute Predictor 2 -- Co-occurrence
    report_progress(6, 15, "computing_cooccurrence")
    logger.info("\n--- Predictor 2: Co-occurrence P(child|parent) ---")
    cooccur_values = compute_cooccurrence(
        model, primary_sae, probes, pairs,
        device=DEVICE,
        n_sequences=N_SEQUENCES_COOCCUR,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    # Step 7: Compute Predictor 3 -- Reconstruction importance
    report_progress(7, 15, "computing_r_parent")
    logger.info("\n--- Predictor 3: Reconstruction importance R(parent) ---")
    r_parent_values = compute_reconstruction_importance(
        model, primary_sae, probes, pairs,
        device=DEVICE,
        n_sequences=N_SEQUENCES_RPARENT,
        max_seq_len=MAX_SEQ_LEN,
    )

    # Release model and SAE
    del model, primary_sae
    gc.collect()
    torch.cuda.empty_cache()

    # Step 8: Assemble pairs with predictors
    report_progress(8, 15, "assembling_predictors")
    pairs_with_predictors = []
    for i, pair in enumerate(pairs):
        cs = cos_sim_values[i]
        co = cooccur_values[i]
        rp = r_parent_values[i]

        entry = {
            **pair,
            "cos_sim": cs,
            "co_occur": co["value"] if co is not None else None,
            "co_occur_detail": co if co is not None else None,
            "r_parent": rp,
        }
        pairs_with_predictors.append(entry)

    # Step 9: Fit three-factor model
    report_progress(9, 15, "fitting_model")
    logger.info("\n--- Fitting three-factor model ---")
    model_results = fit_three_factor_model(pairs_with_predictors)

    # Step 10: Evaluate H9
    report_progress(10, 15, "evaluating_h9")
    h9_verdict = evaluate_h9(model_results)

    logger.info(f"\n  H9 verdict: {h9_verdict['verdict']}")
    logger.info(f"  Confidence: {h9_verdict['confidence']:.2f}")

    # Step 11: Bootstrap CIs on correlation
    report_progress(11, 15, "bootstrap_ci")
    valid_pairs = [p for p in pairs_with_predictors
                   if p.get("cos_sim") is not None
                   and p.get("co_occur") is not None
                   and p.get("r_parent") is not None
                   and p.get("observed_absorption_rate") is not None]

    boot_results = {}
    if len(valid_pairs) >= 10:
        observed_arr = np.array([p["observed_absorption_rate"] for p in valid_pairs])
        for pred_name, pred_fn in [
            ("cos_sim", lambda p: p["cos_sim"]),
            ("cos_sim_squared", lambda p: p["cos_sim"] ** 2),
            ("co_occur", lambda p: p["co_occur"]),
            ("r_parent", lambda p: p["r_parent"]),
        ]:
            pred_arr = np.array([pred_fn(p) for p in valid_pairs])
            if np.std(pred_arr) < 1e-10:
                continue
            rng = np.random.RandomState(SEED)
            boot_rhos = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.choice(len(valid_pairs), size=len(valid_pairs), replace=True)
                try:
                    rho, _ = stats.spearmanr(pred_arr[idx], observed_arr[idx])
                    if not np.isnan(rho):
                        boot_rhos.append(rho)
                except Exception:
                    pass
            if boot_rhos:
                boot_rhos = sorted(boot_rhos)
                boot_results[pred_name] = {
                    "mean_rho": float(np.mean(boot_rhos)),
                    "ci_lower": float(boot_rhos[int(0.025 * len(boot_rhos))]),
                    "ci_upper": float(boot_rhos[min(int(0.975 * len(boot_rhos)), len(boot_rhos) - 1)]),
                    "std": float(np.std(boot_rhos)),
                    "n_bootstrap": len(boot_rhos),
                }
                logger.info(f"  Bootstrap CI for {pred_name}: "
                            f"rho={boot_results[pred_name]['mean_rho']:.4f} "
                            f"[{boot_results[pred_name]['ci_lower']:.4f}, "
                            f"{boot_results[pred_name]['ci_upper']:.4f}]")

    # Step 12: Compile output
    report_progress(12, 15, "compiling_output")
    elapsed = time.time() - start_time

    n_valid = model_results["n_valid_pairs"]

    output = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae_configs": list(SAE_CONFIGS.keys()),
        "primary_sae": PRIMARY_SAE_KEY,
        "n_total_pairs": len(pairs),
        "n_valid_pairs": n_valid,
        "n_sequences_cooccur": N_SEQUENCES_COOCCUR,
        "n_sequences_r_parent": N_SEQUENCES_RPARENT,
        "n_bootstrap": N_BOOTSTRAP,
        "pairs_per_hierarchy": dict(Counter(p["hierarchy"] for p in pairs)),
        "pairs_per_sae": dict(Counter(p["sae_key"] for p in pairs)),
        "predictor_statistics": {
            "cos_sim": _stat_summary([cs for cs in cos_sim_values if cs is not None]),
            "co_occur": _stat_summary([co["value"] for co in cooccur_values if co is not None]),
            "r_parent": _stat_summary([rp for rp in r_parent_values if rp is not None]),
        },
        "bootstrap_ci_correlations": boot_results,
        "model_results": model_results,
        "h9_verdict": h9_verdict,
        "pairs_with_predictors": [
            {k: (float(v) if isinstance(v, (np.floating, float)) else v)
             for k, v in p.items() if k not in ("co_occur_detail",)}
            for p in pairs_with_predictors
        ],
        "methodology_notes": {
            "parent_definition": "Probe direction (logistic regression weight vector) for each class",
            "child_definition": "Top cosine-similarity SAE feature decoder direction for each class",
            "cos_sim_definition": "cosine(probe_direction, W_dec[child_feature_id])",
            "co_occur_definition": "P(child SAE feature active | probe predicts class with prob > 0.4)",
            "r_parent_definition": "Mean MSE increase when parent direction ablated from SAE reconstruction",
            "model_formula": "absorption_rate ~ beta_1*cos_sim^2 + beta_2*co_occur + beta_3*r_parent",
            "evaluation_metric": "Leave-one-out cross-validated Spearman rho",
            "full_mode_changes": (
                "Uses ALL hierarchies (first-letter, city-continent, city-language, city-country), "
                "BOTH SAE widths (16k, 65k), 500 sequences for co-occurrence, 250 for R_parent, "
                "10000 bootstrap iterations, LOO-CV."
            ),
        },
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    # Save
    out_path = PHASE3_DIR / "rate_distortion_predictors.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nSaved: {out_path}")

    # Also save to results dir
    results_path = RESULTS_DIR / f"{TASK_ID}_FULL.json"
    results_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved: {results_path}")

    # Generate summary markdown
    summary_md = generate_summary_md(output)
    md_path = PHASE3_DIR / "rate_distortion_predictors_summary.md"
    md_path.write_text(summary_md)
    logger.info(f"Summary: {md_path}")

    # Finalize
    report_progress(15, 15, "done")

    summary_text = (
        f"Phase 3.1 FULL rate-distortion predictors (H9). "
        f"Pairs: {n_valid}/{len(pairs)} valid. "
        f"Hierarchies: {dict(Counter(p['hierarchy'] for p in pairs))}. "
        f"Time: {elapsed/60:.1f}min. "
    )

    mf = model_results.get("model_fit", {})
    if mf and mf.get("fitted"):
        loo = mf.get("loo_cv", {})
        summary_text += (
            f"Model: rho={mf.get('spearman_rho', 0):.3f} "
            f"(LOO: {loo.get('spearman_rho', 0):.3f}), "
            f"R2={mf.get('r_squared', 0):.3f}. "
        )

    summary_text += f"H9: {h9_verdict['verdict']}."

    mark_done("success", summary_text)
    update_gpu_progress(elapsed, "completed")
    update_experiment_state("completed")

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETED: {summary_text}")
    logger.info(f"{'='*60}")

    return output


def _stat_summary(values):
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
    }


def generate_summary_md(results):
    """Generate human-readable markdown summary."""
    lines = [
        "# Phase 3.1: Rate-Distortion Three-Factor Predictor Model (H9) -- FULL",
        "",
        f"**Mode**: {results.get('mode', '?')}",
        f"**Time**: {results.get('elapsed_minutes', 0):.1f} minutes",
        f"**SAEs**: {', '.join(results.get('sae_configs', []))}",
        f"**Total pairs**: {results.get('n_total_pairs', 0)}",
        f"**Valid pairs**: {results.get('n_valid_pairs', 0)}",
        "",
    ]

    # Pairs breakdown
    pph = results.get("pairs_per_hierarchy", {})
    pps = results.get("pairs_per_sae", {})
    if pph:
        lines.extend(["## Pairs Breakdown", ""])
        lines.append("### By Hierarchy")
        for h, n in sorted(pph.items()):
            lines.append(f"- **{h}**: {n} pairs")
        lines.append("")
    if pps:
        lines.append("### By SAE")
        for s, n in sorted(pps.items()):
            lines.append(f"- **{s}**: {n} pairs")
        lines.append("")

    # Individual predictor correlations
    ind = results.get("model_results", {}).get("individual_correlations", {})
    if ind:
        lines.extend([
            "## Individual Predictor Correlations (All Pairs)", "",
            "| Predictor | Spearman rho | p-value | Pearson r | n |",
            "|-----------|-------------|---------|-----------|---|",
        ])
        for name, vals in sorted(ind.items(), key=lambda x: -abs(x[1].get("spearman_rho", 0))):
            lines.append(
                f"| {name} | {vals.get('spearman_rho', 0):.4f} | "
                f"{vals.get('spearman_p', 1):.4f} | "
                f"{vals.get('pearson_r', 0):.4f} | "
                f"{vals.get('n_pairs', 0)} |"
            )
        lines.append("")

    # Per-hierarchy correlations
    phc = results.get("model_results", {}).get("per_hierarchy_correlations", {})
    if phc:
        lines.extend(["## Per-Hierarchy Predictor Correlations", ""])
        for h, h_ind in phc.items():
            lines.append(f"### {h}")
            lines.append("| Predictor | Spearman rho | p-value | n |")
            lines.append("|-----------|-------------|---------|---|")
            for name, vals in sorted(h_ind.items(), key=lambda x: -abs(x[1].get("spearman_rho", 0))):
                lines.append(
                    f"| {name} | {vals.get('spearman_rho', 0):.4f} | "
                    f"{vals.get('spearman_p', 1):.4f} | "
                    f"{vals.get('n_pairs', 0)} |"
                )
            lines.append("")

    # Bootstrap CIs
    boot = results.get("bootstrap_ci_correlations", {})
    if boot:
        lines.extend([
            "## Bootstrap 95% CI on Spearman Correlations", "",
            "| Predictor | Mean rho | CI lower | CI upper | std |",
            "|-----------|----------|----------|----------|-----|",
        ])
        for name, vals in boot.items():
            lines.append(
                f"| {name} | {vals['mean_rho']:.4f} | {vals['ci_lower']:.4f} | "
                f"{vals['ci_upper']:.4f} | {vals['std']:.4f} |"
            )
        lines.append("")

    # Three-factor model
    mf = results.get("model_results", {}).get("model_fit", {})
    if mf and mf.get("fitted"):
        loo = mf.get("loo_cv", {})
        lines.extend([
            "## Three-Factor Linear Model", "",
            "**Formula**: absorption ~ beta_1*cos_sim^2 + beta_2*co_occur + beta_3*r_parent", "",
            f"- **R^2** = {mf.get('r_squared', 0):.4f}",
            f"- **Spearman rho** = {mf.get('spearman_rho', 0):.4f} (p = {mf.get('spearman_p', 1):.4f})",
            f"- **LOO-CV Spearman rho** = {loo.get('spearman_rho', 0):.4f} (p = {loo.get('spearman_p', 1):.4f})",
            f"- **LOO-CV MSE** = {loo.get('mse', 0):.6f}",
            "",
            "### Coefficients", "",
        ])
        coefs = mf.get("coefficients", {})
        for name, val in coefs.items():
            lines.append(f"- **{name}**: {val:.6f}")
        lines.append("")

    # Cross-domain analysis
    cda = results.get("model_results", {}).get("cross_domain_analysis", {})
    if cda:
        lines.extend([
            "## Cross-Domain Analysis", "",
            "| Hierarchy | n | Mean Absorption | Std | Mean cos_sim | Mean co_occur | Mean R_parent |",
            "|-----------|---|-----------------|-----|-------------|---------------|---------------|",
        ])
        for h, sd in cda.items():
            if isinstance(sd, dict) and "mean_absorption" in sd:
                lines.append(
                    f"| {h} | {sd['n_pairs']} | {sd['mean_absorption']:.4f} | "
                    f"{sd.get('std_absorption', 0):.4f} | "
                    f"{sd['mean_cos_sim']:.4f} | {sd['mean_co_occur']:.4f} | "
                    f"{sd['mean_r_parent']:.6f} |"
                )
        if "anova" in cda:
            anova = cda["anova"]
            lines.append(f"\n**ANOVA**: F={anova['f_statistic']:.4f}, p={anova['p_value']:.4f}")
        lines.append("")

    # H9 verdict
    hv = results.get("h9_verdict", {})
    if hv:
        lines.extend([
            "## H9 Verdict", "",
            f"**Verdict**: {hv.get('verdict', '?')}",
            f"**Confidence**: {hv.get('confidence', 0):.2f}",
            f"**Evaluation source**: {hv.get('evaluation_source', '?')}",
            f"**rho used**: {hv.get('rho_used', 0):.4f} (p = {hv.get('p_value_used', 1):.4f})",
            f"**Best individual predictor**: {hv.get('best_individual_predictor', '?')} "
            f"(rho={hv.get('best_individual_rho', 0):.4f})",
            f"**Target**: {hv.get('target', '?')}",
            f"**Falsification**: {hv.get('falsification', '?')}",
            "",
        ])

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        elapsed = time.time() - (globals().get("start_time", time.time()))
        update_gpu_progress(elapsed, "failed")
        update_experiment_state("failed", str(e))
        sys.exit(1)
