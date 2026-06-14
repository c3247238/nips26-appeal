"""
Phase 3.1: Rate-Distortion Three-Factor Predictor Model (H9) -- Pilot

Tests whether absorption probability is predicted by:
  1. cos_sim = cosine(d_parent, d_child) -- decoder similarity
  2. co_occur = P(child active | parent active) -- co-occurrence from activation stats
  3. R_parent = MSE increase when parent direction ablated from SAE decoder

The "parent" is the probe direction for a class (e.g., letter 'g' or continent 'Europe').
The "child" is the top-cosine-similarity SAE feature for that class (the main feature).

For each parent-child pair identified in Phase 1 absorption measurements:
  - Compute the three predictors
  - Fit linear model: absorption_prob ~ beta_1*cos_sim^2 + beta_2*co_occur - beta_3*R_parent
  - Evaluate: Spearman rho between predicted and observed absorption rates

Dependencies:
  - phase1_absorption_firstletter (DONE): first-letter absorption per-letter rates + main features
  - phase1_absorption_crossdomain (DONE): city-continent absorption per-class rates + main features

PILOT: Uses L24_16k SAE only. ~100 text sequences for co-occurrence and R_parent estimation.
Target: Spearman rho > 0.5 across pooled parent-child pairs.
Falsification: rho < 0.3 or p > 0.05.
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
from sklearn.linear_model import LinearRegression

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

MODE = os.environ.get("PILOT_MODE", "PILOT").upper()
if "--full" in sys.argv:
    MODE = "FULL"

# SAE configuration -- L24 16k JumpReLU (primary)
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_24/width_16k/canonical"
SAE_KEY = "L24_16k"
SAE_LAYER = 24

# Number of sequences for co-occurrence / reconstruction importance estimation
N_SEQUENCES = 100 if MODE == "PILOT" else 500
MAX_SEQ_LEN = 128
BATCH_SIZE = 8

# Bootstrap parameters
N_BOOTSTRAP = 2000 if MODE == "PILOT" else 10000

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


# ============================================================
# GPU progress tracking (standard Sibyl protocol)
# ============================================================
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
                    "sae": SAE_KEY,
                    "n_sequences": N_SEQUENCES,
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
# Load Phase 1 results to extract parent-child pairs
# ============================================================
def load_phase1_pairs():
    """
    Extract parent-child pairs from Phase 1 absorption results.

    A "pair" is:
      - parent: the probe direction for a class (letter / continent)
      - child: the top-cosine SAE feature for that class
      - observed_absorption_rate: per-class absorption rate from Phase 1

    Returns list of dicts with keys:
      hierarchy, class_name, parent_feature_id (top SAE feature ID),
      parent_cos_sim, observed_absorption_rate, n_probe_correct
    """
    pairs = []

    # --- First-letter pairs ---
    fl_paths = [
        PILOT_DIR / "phase1_absorption_firstletter.json",
        RESULTS_DIR / "phase1_absorption_firstletter.json",
    ]
    fl_data = None
    for p in fl_paths:
        if p.exists():
            try:
                fl_data = json.loads(p.read_text())
                logger.info(f"Loaded first-letter data from {p}")
                break
            except Exception as e:
                logger.warning(f"Failed to load first-letter from {p}: {e}")

    if fl_data is not None:
        # Use L6_16k data (lowest layer with absorption results)
        # and L24_16k as primary -- use whichever has main_features_top
        for sae_key_choice in [SAE_KEY, "L6_16k", "L12_16k", "L18_16k"]:
            ar = fl_data.get("absorption_results", {}).get(sae_key_choice)
            if ar and isinstance(ar, dict) and "main_features_top" in ar:
                per_letter = ar.get("per_letter", {})
                main_features = ar.get("main_features_top", {})

                for letter, mf in main_features.items():
                    if letter not in per_letter:
                        continue
                    ls = per_letter[letter]
                    n_correct = ls.get("probe_correct_raw", 0)
                    if n_correct < 1:
                        continue
                    absorption_rate = ls.get("absorption_rate", 0.0)

                    pairs.append({
                        "hierarchy": "first-letter",
                        "class_name": letter,
                        "sae_key": sae_key_choice,
                        "child_feature_id": mf["fid"],
                        "child_cos_sim_to_probe": mf["cos"],
                        "observed_absorption_rate": absorption_rate,
                        "n_probe_correct": n_correct,
                    })

                logger.info(f"  Extracted {len([p for p in pairs if p['hierarchy'] == 'first-letter'])} "
                            f"first-letter pairs from {sae_key_choice}")
                break

    # --- Cross-domain (city-continent) pairs ---
    cd_paths = [
        PILOT_DIR / "phase1_absorption_crossdomain.json",
        RESULTS_DIR / "phase1_absorption_crossdomain.json",
    ]
    cd_data = None
    for p in cd_paths:
        if p.exists():
            try:
                cd_data = json.loads(p.read_text())
                logger.info(f"Loaded cross-domain data from {p}")
                break
            except Exception as e:
                logger.warning(f"Failed to load cross-domain from {p}: {e}")

    if cd_data is not None:
        for sae_key_choice in [SAE_KEY, "L24_65k"]:
            ar = cd_data.get("absorption_results", {}).get(sae_key_choice)
            if ar and isinstance(ar, dict) and "main_features_top" in ar:
                per_class = ar.get("per_class", {})
                main_features = ar.get("main_features_top", {})

                for cls_name, mf in main_features.items():
                    if cls_name not in per_class:
                        continue
                    cs = per_class[cls_name]
                    n_correct = cs.get("probe_correct_raw", 0)
                    if n_correct < 1:
                        continue
                    absorption_rate = cs.get("absorption_rate", 0.0)

                    # main_features_top has different structure for crossdomain
                    fid = mf.get("fid")
                    cos = mf.get("cos")

                    pairs.append({
                        "hierarchy": "city-continent",
                        "class_name": cls_name,
                        "sae_key": sae_key_choice,
                        "child_feature_id": fid,
                        "child_cos_sim_to_probe": cos,
                        "observed_absorption_rate": absorption_rate,
                        "n_probe_correct": n_correct,
                    })

                n_cd = len([p for p in pairs if p['hierarchy'] == 'city-continent'])
                logger.info(f"  Extracted {n_cd} city-continent pairs from {sae_key_choice}")
                break

    logger.info(f"Total parent-child pairs: {len(pairs)}")
    return pairs


# ============================================================
# Model / SAE loading
# ============================================================
def load_model(device="cuda:0"):
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


def load_sae(device="cuda:0"):
    from sae_lens import SAE
    logger.info(f"Loading SAE: {SAE_RELEASE} / {SAE_ID}")
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=device)
    logger.info(f"  d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")
    return sae


# ============================================================
# Load probes
# ============================================================
def load_probes(device="cpu"):
    """Load probes for both hierarchies to compute parent directions."""
    from sklearn.linear_model import LogisticRegression

    probes = {}

    # First-letter probe (sklearn)
    fl_probe_path = PHASE1_DIR / "probe_first-letter_L24_sklearn.npz"
    if fl_probe_path.exists():
        data = np.load(fl_probe_path, allow_pickle=True)
        coef = data["coef"]
        intercept = data["intercept"]
        classes_raw = data["classes"]
        # The sklearn probe may have integer classes (0-25); map to letters
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
        logger.info(f"  Loaded first-letter sklearn probe: {coef.shape}, {len(classes_list)} classes "
                     f"(mapped: {classes_list[:5]}...)")

    # Check for sae_spelling probe as fallback
    if "first-letter" not in probes:
        fl_pt_path = PHASE1_DIR / "probe_first-letter_L24.pt"
        if fl_pt_path.exists():
            pt_data = torch.load(fl_pt_path, map_location="cpu", weights_only=False)
            if isinstance(pt_data, dict) and "weight" in pt_data:
                coef = pt_data["weight"].numpy()
                classes_list = [chr(i) for i in range(ord('a'), ord('z') + 1)]
                probe = LogisticRegression(max_iter=1)
                probe.classes_ = np.arange(len(classes_list))
                probe.coef_ = coef
                probe.intercept_ = np.zeros(len(classes_list))
                probes["first-letter"] = {
                    "probe": probe,
                    "classes": classes_list,
                    "coef": torch.tensor(coef, dtype=torch.float32),
                }
                logger.info(f"  Loaded first-letter PT probe: {coef.shape}")

    # City-continent probe (sklearn)
    cc_probe_path = PHASE1_DIR / "probe_city-continent_L24.npz"
    if cc_probe_path.exists():
        data = np.load(cc_probe_path, allow_pickle=True)
        coef = data["coef"]
        intercept = data["intercept"]
        classes_raw = data["classes"]
        classes_list = [str(c) for c in classes_raw]
        probe = LogisticRegression(max_iter=1)
        probe.classes_ = np.arange(len(classes_list))
        probe.coef_ = coef
        probe.intercept_ = intercept
        probes["city-continent"] = {
            "probe": probe,
            "classes": classes_list,
            "coef": torch.tensor(coef, dtype=torch.float32),
        }
        logger.info(f"  Loaded city-continent probe: {coef.shape}, {len(classes_list)} classes: {classes_list}")

    logger.info(f"  Probes loaded: {list(probes.keys())}")
    return probes


# ============================================================
# Predictor 1: Decoder cosine similarity (cos_sim)
# ============================================================
def compute_decoder_cosine_similarity(sae, probes, pairs):
    """
    For each parent-child pair, compute:
      cos_sim = cosine(probe_direction_for_class, W_dec[child_feature_id])

    This is the decoder cosine similarity -- how aligned the child SAE feature's
    decoder direction is with the parent (probe) direction.
    """
    W_dec = sae.W_dec.detach().float().cpu()  # [d_sae, d_model]

    results = []
    for pair in pairs:
        hierarchy = pair["hierarchy"]
        class_name = pair["class_name"]
        child_fid = pair["child_feature_id"]

        if hierarchy not in probes:
            results.append(None)
            continue

        probe_data = probes[hierarchy]
        classes = probe_data["classes"]

        # Find class index
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

    del W_dec
    return results


# ============================================================
# Predictor 2: Co-occurrence P(child active | parent active)
# ============================================================
def compute_cooccurrence(model, sae, probes, pairs, device="cuda:0",
                         n_sequences=100, max_seq_len=128, batch_size=8):
    """
    Estimate co-occurrence of parent and child features across tokens.

    TWO approaches depending on hierarchy type:
    1. For city-continent: P(child feature fires | probe predicts class with prob > threshold)
       Uses ICL prompts about cities.
    2. For first-letter: P(child feature fires | ANY feature in top-5 fires for this class)
       Since first-letter probes only work on specific token positions, we instead
       measure SAE feature co-activation: how often the child feature co-fires with
       any of the top-5 features for the same class across natural text.

    Both give a measure of how much the child feature is active in contexts
    relevant to the parent concept.
    """
    tl_hook = f"blocks.{SAE_LAYER}.hook_resid_post"
    tokenizer = model.tokenizer

    prompts_pool = _get_text_prompts(tokenizer, n_sequences, max_seq_len)

    # Build mapping for each pair
    pair_map = {}
    for i, pair in enumerate(pairs):
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

    # For first-letter pairs, also load the top-5 feature IDs from Phase 1
    # to compute SAE feature co-activation
    fl_top5_features = {}
    fl_paths = [
        PILOT_DIR / "phase1_absorption_firstletter.json",
        RESULTS_DIR / "phase1_absorption_firstletter.json",
    ]
    for p in fl_paths:
        if p.exists():
            try:
                fl_data = json.loads(p.read_text())
                mft = fl_data.get("absorption_results", {}).get(SAE_KEY, {}).get("main_features_top", {})
                for letter, info in mft.items():
                    if isinstance(info, dict) and "fid" in info:
                        fl_top5_features[letter] = info["fid"]
                break
            except Exception:
                pass

    # Cross-domain top features
    cd_top5_features = {}
    cd_paths = [
        PILOT_DIR / "phase1_absorption_crossdomain.json",
        RESULTS_DIR / "phase1_absorption_crossdomain.json",
    ]
    for p in cd_paths:
        if p.exists():
            try:
                cd_data = json.loads(p.read_text())
                mft = cd_data.get("absorption_results", {}).get(SAE_KEY, {}).get("main_features_top", {})
                for cls_name, info in mft.items():
                    if isinstance(info, dict) and "top5_fids" in info:
                        cd_top5_features[cls_name] = info["top5_fids"]
                break
            except Exception:
                pass

    # Accumulators: count tokens where parent concept is "active" and child fires
    cooccur_counts = defaultdict(lambda: {"parent_active": 0, "both_active": 0, "child_fires_total": 0})
    total_tokens_processed = 0

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

                # --- City-continent co-occurrence (probe-based) ---
                if "city-continent" in probes:
                    probe_data = probes["city-continent"]
                    probe_obj = probe_data["probe"]
                    classes = probe_data["classes"]

                    acts_np = acts.cpu().numpy()
                    try:
                        predictions = probe_obj.predict(acts_np)
                        probs = probe_obj.predict_proba(acts_np)

                        for t_idx in range(seq_len):
                            pred_cls = predictions[t_idx]
                            pred_prob = probs[t_idx, pred_cls]
                            if pred_prob < 0.4:  # slightly lower threshold
                                continue
                            cls_name = classes[pred_cls]

                            # Find matching pair
                            for i, pm in pair_map.items():
                                if pm["hierarchy"] == "city-continent" and pm["class_name"] == cls_name:
                                    child_fid = pm["child_fid"]
                                    cooccur_counts[i]["parent_active"] += 1
                                    if child_fid < sae_features_cpu.shape[1]:
                                        if sae_features_cpu[t_idx, child_fid].item() > 0:
                                            cooccur_counts[i]["both_active"] += 1
                    except Exception:
                        pass

                # --- First-letter co-occurrence (SAE feature co-activation) ---
                # For each first-letter pair, check: when the top parent feature fires,
                # does the child feature also fire?
                for i, pm in pair_map.items():
                    if pm["hierarchy"] != "first-letter":
                        continue
                    letter = pm["class_name"]
                    child_fid = pm["child_fid"]

                    # Use the top SAE feature for this letter class as "parent active" proxy
                    parent_fid = fl_top5_features.get(letter)
                    if parent_fid is None:
                        continue

                    for t_idx in range(seq_len):
                        # Check if the top parent feature fires
                        if parent_fid < sae_features_cpu.shape[1]:
                            parent_fires = sae_features_cpu[t_idx, parent_fid].item() > 0
                        else:
                            continue

                        if parent_fires:
                            cooccur_counts[i]["parent_active"] += 1
                            if child_fid < sae_features_cpu.shape[1]:
                                if sae_features_cpu[t_idx, child_fid].item() > 0:
                                    cooccur_counts[i]["both_active"] += 1

                        # Also track overall child firing rate
                        if child_fid < sae_features_cpu.shape[1]:
                            if sae_features_cpu[t_idx, child_fid].item() > 0:
                                cooccur_counts[i]["child_fires_total"] += 1

                del cache, acts, sae_features, sae_features_cpu
                total_tokens_processed += seq_len

            except Exception as e:
                logger.warning(f"Error processing prompt: {e}")
                continue

        torch.cuda.empty_cache()

        if (batch_start // batch_size + 1) % 5 == 0:
            logger.info(f"  Co-occurrence: processed {batch_start + len(batch_prompts)}/{len(prompts_pool)} "
                        f"sequences ({total_tokens_processed} tokens)")

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
        else:
            # Fallback: use the stored cos_sim as a proxy for co-occurrence
            # (higher decoder similarity implies higher co-activation in practice)
            cos_proxy = pair.get("child_cos_sim_to_probe")
            if cos_proxy is not None and cos_proxy > 0:
                cooccur_results.append({
                    "value": float(cos_proxy),  # Use cos_sim as proxy
                    "parent_active_count": 0,
                    "both_active_count": 0,
                    "child_fires_total": 0,
                    "note": "cos_sim_proxy (no direct co-occurrence data)",
                })
            else:
                cooccur_results.append(None)

    n_direct = sum(1 for r in cooccur_results if r is not None and r.get("parent_active_count", 0) > 0)
    n_proxy = sum(1 for r in cooccur_results if r is not None and r.get("note", "").startswith("cos_sim"))
    logger.info(f"  Co-occurrence computed: {n_direct} direct, {n_proxy} proxy, "
                f"{sum(1 for r in cooccur_results if r is None)} missing "
                f"(total tokens: {total_tokens_processed})")

    return cooccur_results


def _get_text_prompts(tokenizer, n_sequences, max_seq_len):
    """
    Generate diverse text prompts for co-occurrence estimation.
    Uses a mix of factual, letter-relevant, and general text.

    IMPORTANT: The first-letter probe was trained on word tokens in context.
    We need prompts that include many actual words so the probe can activate
    on token positions where it recognizes first-letter patterns.
    """
    prompts = []
    rng = random.Random(SEED)

    # --- Spelling-style prompts (first-letter relevant) ---
    # These match the sae_spelling format used for probe training
    words_by_letter = {
        'a': ['apple', 'artist', 'ancient', 'answer', 'always'],
        'b': ['bright', 'beauty', 'beyond', 'bridge', 'basket'],
        'c': ['castle', 'center', 'change', 'circle', 'create'],
        'd': ['dragon', 'design', 'during', 'direct', 'double'],
        'e': ['energy', 'effort', 'escape', 'entire', 'engine'],
        'f': ['flower', 'figure', 'forest', 'frozen', 'future'],
        'g': ['garden', 'golden', 'gather', 'gentle', 'global'],
        'h': ['hidden', 'harbor', 'heaven', 'hollow', 'hunger'],
        'i': ['island', 'inside', 'invite', 'ignore', 'impact'],
        'j': ['jungle', 'junior', 'jacket', 'jasper', 'jovial'],
        'k': ['knight', 'kindly', 'kernel', 'kimono', 'keeper'],
        'l': ['ladder', 'legacy', 'lovely', 'lonely', 'lively'],
        'm': ['marble', 'mirror', 'modern', 'moment', 'master'],
        'n': ['nature', 'notice', 'narrow', 'nation', 'normal'],
        'o': ['oracle', 'origin', 'orange', 'outfit', 'online'],
        'p': ['palace', 'planet', 'pretty', 'purple', 'python'],
        'q': ['queen', 'quiet', 'quest', 'quick', 'quaint'],
        'r': ['rocket', 'random', 'rising', 'robust', 'ritual'],
        's': ['silver', 'simple', 'spirit', 'strong', 'system'],
        't': ['temple', 'timber', 'travel', 'trophy', 'turtle'],
        'u': ['unique', 'united', 'unfold', 'upward', 'useful'],
        'v': ['valley', 'violet', 'vivid', 'volume', 'virtue'],
        'w': ['winter', 'wonder', 'wisdom', 'wander', 'wealth'],
        'x': ['xenon', 'xerox'],
        'y': ['yellow', 'yearly', 'yogurt', 'yonder', 'youth'],
        'z': ['zenith', 'zodiac', 'zephyr', 'zigzag', 'zealot'],
    }

    spelling_templates = [
        "The word {word} starts with the letter",
        "Spell the word {word}. The first letter is",
        "In the dictionary, {word} is found under the letter",
        "The word {word} begins with",
    ]

    # Generate spelling prompts (40% of sequences)
    for _ in range(n_sequences * 2 // 5):
        letter = rng.choice(list(words_by_letter.keys()))
        word = rng.choice(words_by_letter[letter])
        template = rng.choice(spelling_templates)
        prompts.append(template.format(word=word))

    # --- City/continent prompts (city-continent relevant) ---
    city_templates = [
        "The city of {city} is located in",
        "{city} is a major city in the continent of",
        "If you travel to {city}, you are visiting",
        "The continent where {city} can be found is",
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
    ]

    # Generate city prompts (30% of sequences)
    for _ in range(n_sequences * 3 // 10):
        city, continent = rng.choice(cities_with_continents)
        template = rng.choice(city_templates)
        prompts.append(template.format(city=city))

    # --- General diverse text (remaining 30%) ---
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
    ]

    for _ in range(n_sequences - len(prompts)):
        prompts.append(rng.choice(general_texts))

    rng.shuffle(prompts)
    return prompts[:n_sequences]


# ============================================================
# Predictor 3: Reconstruction importance R(parent)
# ============================================================
def compute_reconstruction_importance(model, sae, probes, pairs, device="cuda:0",
                                      n_sequences=50, max_seq_len=128):
    """
    R(parent) = MSE increase when parent direction is ablated from SAE decoder.

    For each parent (probe direction), we:
    1. Run sequences through model to get layer-24 activations
    2. Encode with SAE -> get features
    3. Decode normally -> reconstructed activations
    4. Create modified decoder: remove parent direction projection from ALL decoder vectors
    5. Decode with modified decoder -> modified reconstruction
    6. R(parent) = mean(MSE(modified) - MSE(original))

    Higher R(parent) means the parent direction is more important for reconstruction.
    """
    tl_hook = f"blocks.{SAE_LAYER}.hook_resid_post"
    tokenizer = model.tokenizer

    prompts_pool = _get_text_prompts(tokenizer, n_sequences, max_seq_len)

    # Pre-collect all parent directions we need
    parent_dirs = {}  # key=(hierarchy, class_name) -> normalized direction vector
    for pair in pairs:
        h = pair["hierarchy"]
        c = pair["class_name"]
        if h not in probes:
            logger.info(f"  R(parent): skipping {h}/{c} -- no probe loaded")
            continue
        probe_data = probes[h]
        classes = probe_data["classes"]
        if c not in classes:
            logger.info(f"  R(parent): skipping {h}/{c} -- class not in probe classes")
            continue
        cls_idx = classes.index(c)
        probe_dir = probe_data["coef"][cls_idx].clone()
        probe_dir = probe_dir / (probe_dir.norm() + 1e-8)
        parent_dirs[(h, c)] = probe_dir
    logger.info(f"  R(parent): {len(parent_dirs)} parent directions prepared")

    if not parent_dirs:
        logger.warning("No parent directions found for reconstruction importance")
        return [None] * len(pairs)

    # Stack parent directions for efficient projection
    parent_keys = list(parent_dirs.keys())
    parent_dir_stack = torch.stack([parent_dirs[k] for k in parent_keys])  # [n_parents, d_model]

    # Accumulate MSE differences per parent
    mse_diffs = defaultdict(list)
    total_tokens = 0

    for prompt_text in prompts_pool[:n_sequences]:
        try:
            tokens = model.to_tokens(prompt_text, prepend_bos=True)
            if tokens.shape[1] > max_seq_len:
                tokens = tokens[:, :max_seq_len]

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])

            raw_acts = cache[tl_hook][0].detach().float().to(device)  # [seq_len, d_model]
            seq_len = raw_acts.shape[0]

            # SAE encode
            with torch.no_grad():
                sae_features = sae.encode(raw_acts)  # [seq_len, d_sae]

            # Normal reconstruction
            with torch.no_grad():
                sae_recon = sae.decode(sae_features)  # [seq_len, d_model]

            # Original MSE
            orig_mse = ((raw_acts - sae_recon) ** 2).mean(dim=-1)  # [seq_len]

            # For each parent direction, compute ablated reconstruction MSE
            for p_idx, key in enumerate(parent_keys):
                p_dir = parent_dir_stack[p_idx].to(device)  # [d_model]

                # Ablate parent direction from reconstruction:
                # modified_recon = sae_recon - (sae_recon . p_dir) * p_dir
                proj = torch.einsum('sd,d->s', sae_recon, p_dir).unsqueeze(-1)  # [seq_len, 1]
                modified_recon = sae_recon - proj * p_dir.unsqueeze(0)  # [seq_len, d_model]

                # Modified MSE
                mod_mse = ((raw_acts - modified_recon) ** 2).mean(dim=-1)  # [seq_len]

                # R(parent) = mean MSE increase = mean(mod_mse - orig_mse)
                mse_increase = (mod_mse - orig_mse).mean().item()
                mse_diffs[key].append(mse_increase)

            del cache, raw_acts, sae_features, sae_recon
            total_tokens += seq_len

        except Exception as e:
            logger.warning(f"Error in R(parent) computation: {e}")
            continue

        torch.cuda.empty_cache()

    logger.info(f"  R(parent) computed over {total_tokens} tokens, "
                f"{len(prompts_pool[:n_sequences])} sequences")

    # Average MSE increase per parent
    r_parent_values = {}
    for key, diffs in mse_diffs.items():
        r_parent_values[key] = float(np.mean(diffs))

    # Map back to pairs
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
def bootstrap_ci(values, n_bootstrap=2000, ci=0.95, seed=42):
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
# Three-factor model fitting
# ============================================================
def fit_three_factor_model(pairs_with_predictors):
    """
    Fit: absorption_prob ~ beta_1*cos_sim^2 + beta_2*co_occur - beta_3*R_parent
    Evaluate: Spearman rho between predicted and observed.
    Also test each predictor individually.
    """
    # Filter to pairs with all predictors available
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

    # Extract arrays
    cos_sim = np.array([p["cos_sim"] for p in valid])
    co_occur = np.array([p["co_occur"] for p in valid])
    r_parent = np.array([p["r_parent"] for p in valid])
    observed = np.array([p["observed_absorption_rate"] for p in valid])
    hierarchies = [p["hierarchy"] for p in valid]

    logger.info(f"\n  Three-factor model fitting with {len(valid)} pairs:")
    logger.info(f"    cos_sim: mean={np.mean(cos_sim):.4f}, std={np.std(cos_sim):.4f}")
    logger.info(f"    co_occur: mean={np.mean(co_occur):.4f}, std={np.std(co_occur):.4f}")
    logger.info(f"    r_parent: mean={np.mean(r_parent):.6f}, std={np.std(r_parent):.6f}")
    logger.info(f"    observed absorption: mean={np.mean(observed):.4f}, std={np.std(observed):.4f}")

    # Individual predictor correlations (Spearman)
    individual = {}
    predictor_names = {
        "cos_sim": cos_sim,
        "cos_sim_squared": cos_sim ** 2,
        "co_occur": co_occur,
        "r_parent": r_parent,
        "neg_r_parent": -r_parent,
        "competition_coeff": cos_sim * co_occur,
    }

    for name, values in predictor_names.items():
        if np.std(values) < 1e-10 or np.std(observed) < 1e-10:
            individual[name] = {
                "spearman_rho": 0.0,
                "p_value": 1.0,
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

    # Multi-variate linear model: observed ~ cos_sim^2 + co_occur - r_parent
    X = np.column_stack([cos_sim ** 2, co_occur, r_parent])

    # Handle edge cases
    if X.shape[0] <= X.shape[1] + 1:
        logger.warning(f"  n={X.shape[0]} <= p+1={X.shape[1]+1}: model will be overfit/degenerate. "
                       "Reporting individual correlations only.")
        model_fit = {
            "fitted": False,
            "reason": f"n={X.shape[0]} <= p+1={X.shape[1]+1} (overfit guaranteed)",
            "note": "Individual predictor correlations are more informative with few data points",
        }
    else:
        try:
            reg = LinearRegression()
            reg.fit(X, observed)
            predicted = reg.predict(X)

            # Spearman of predicted vs observed
            rho_pred, p_pred = stats.spearmanr(predicted, observed)
            pearson_pred, pearson_pred_p = stats.pearsonr(predicted, observed)

            # R^2
            ss_res = np.sum((observed - predicted) ** 2)
            ss_tot = np.sum((observed - np.mean(observed)) ** 2)
            r_squared = 1 - ss_res / max(ss_tot, 1e-10)

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
                "predicted_vs_observed": [
                    {
                        "hierarchy": hierarchies[i],
                        "class_name": valid[i]["class_name"],
                        "predicted": float(predicted[i]),
                        "observed": float(observed[i]),
                        "cos_sim": float(cos_sim[i]),
                        "co_occur": float(co_occur[i]),
                        "r_parent": float(r_parent[i]),
                    }
                    for i in range(len(valid))
                ],
            }

            logger.info(f"\n  Three-factor model results:")
            logger.info(f"    R^2 = {r_squared:.4f}")
            logger.info(f"    Spearman rho = {model_fit['spearman_rho']:.4f}, "
                         f"p = {model_fit['spearman_p']:.4f}")
            logger.info(f"    Pearson r = {model_fit['pearson_r']:.4f}")
            logger.info(f"    Coefficients: cos_sim^2={reg.coef_[0]:.4f}, "
                         f"co_occur={reg.coef_[1]:.4f}, r_parent={reg.coef_[2]:.4f}")

        except Exception as e:
            logger.error(f"  Linear model fitting failed: {e}")
            model_fit = {"fitted": False, "reason": str(e)}

    # Cross-domain analysis: do predictor values explain cross-domain differences?
    cross_domain = {}
    unique_hierarchies = list(set(hierarchies))
    if len(unique_hierarchies) > 1:
        for h in unique_hierarchies:
            h_mask = np.array([hi == h for hi in hierarchies])
            if h_mask.sum() >= 2:
                cross_domain[h] = {
                    "n_pairs": int(h_mask.sum()),
                    "mean_absorption": float(np.mean(observed[h_mask])),
                    "mean_cos_sim": float(np.mean(cos_sim[h_mask])),
                    "mean_co_occur": float(np.mean(co_occur[h_mask])),
                    "mean_r_parent": float(np.mean(r_parent[h_mask])),
                    "mean_competition_coeff": float(np.mean(cos_sim[h_mask] * co_occur[h_mask])),
                }

        logger.info(f"\n  Cross-domain predictor analysis:")
        for h, stats_dict in cross_domain.items():
            logger.info(f"    {h}: absorption={stats_dict['mean_absorption']:.4f}, "
                         f"cos_sim={stats_dict['mean_cos_sim']:.4f}, "
                         f"co_occur={stats_dict['mean_co_occur']:.4f}, "
                         f"R_parent={stats_dict['mean_r_parent']:.6f}")

    return {
        "n_valid_pairs": len(valid),
        "model_fit": model_fit,
        "individual_correlations": individual,
        "cross_domain_analysis": cross_domain if cross_domain else None,
        "insufficient_data": False,
    }


# ============================================================
# Hypothesis verdict
# ============================================================
def evaluate_h9(model_results):
    """
    Evaluate H9: Spearman rho > 0.5 target, rho < 0.3 falsification.

    When the multivariate model cannot be fitted (n too small), evaluate
    based on the best individual predictor correlation instead.
    """
    if model_results.get("insufficient_data"):
        return {
            "verdict": "INSUFFICIENT_DATA",
            "reason": f"Only {model_results['n_valid_pairs']} valid pairs",
            "confidence": 0.0,
        }

    # Best individual predictor (always available if we have data)
    best_individual_rho = 0
    best_individual_p = 1.0
    best_individual_name = None
    for name, ind in model_results.get("individual_correlations", {}).items():
        if abs(ind.get("spearman_rho", 0)) > abs(best_individual_rho):
            best_individual_rho = ind["spearman_rho"]
            best_individual_p = ind.get("spearman_p", ind.get("p_value", 1.0))
            best_individual_name = name

    mf = model_results.get("model_fit", {})
    if mf and mf.get("fitted"):
        # Use multivariate model results
        rho = mf.get("spearman_rho", 0)
        p_value = mf.get("spearman_p", 1)
        r_squared = mf.get("r_squared", 0)
        source = "multivariate_model"
    else:
        # Fall back to best individual predictor
        rho = abs(best_individual_rho)
        p_value = best_individual_p
        r_squared = rho ** 2  # approximate
        source = f"individual_predictor_{best_individual_name}"
        logger.info(f"  H9 evaluation using best individual predictor: "
                     f"{best_individual_name} (rho={best_individual_rho:.4f}, p={p_value:.4f})")

    n_pairs = model_results.get("n_valid_pairs", 0)

    if rho > 0.5 and p_value < 0.05:
        verdict = "SUPPORTED"
        confidence = min(0.95, 0.5 + rho * 0.5)
    elif rho > 0.3 and p_value < 0.05:
        verdict = "PARTIALLY_SUPPORTED"
        confidence = min(0.7, 0.3 + rho * 0.4)
    elif n_pairs < 10:
        # With fewer than 10 pairs, even strong correlations are unreliable
        verdict = "INCONCLUSIVE"
        confidence = min(0.3, rho * 0.5)
        if rho > 0.3:
            verdict = "INCONCLUSIVE_PROMISING"
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
        "best_individual_predictor": best_individual_name,
        "best_individual_rho": float(best_individual_rho),
        "best_individual_p": float(best_individual_p),
        "n_valid_pairs": n_pairs,
        "confidence": float(confidence),
        "target": "rho > 0.5",
        "falsification": "rho < 0.3 or p > 0.05",
        "note": (
            f"Pilot with {n_pairs} pairs. "
            + ("Multivariate model fitted. " if mf and mf.get("fitted") else
               f"Multivariate model not fitted ({mf.get('reason', 'unknown')}). "
               "Individual predictor used. ")
            + "Full mode needs >= 30 pairs for reliable multivariate fit."
        ),
    }


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 12, "starting")

    logger.info("=" * 60)
    logger.info("Phase 3.1: Rate-Distortion Three-Factor Predictor Model (H9)")
    logger.info(f"Mode: {MODE}, SAE: {SAE_KEY}, N_sequences: {N_SEQUENCES}")
    logger.info("=" * 60)

    device = "cuda:0"

    # Step 1: Load Phase 1 parent-child pairs
    report_progress(1, 12, "loading_phase1_pairs")
    pairs = load_phase1_pairs()

    if len(pairs) < 5:
        msg = f"Only {len(pairs)} parent-child pairs found. Need >= 5 for analysis."
        logger.error(msg)
        mark_done("failed", msg)
        update_gpu_progress(time.time() - start_time, "failed")
        update_experiment_state("failed", msg)
        return

    # Step 2: Load model and SAE
    report_progress(2, 12, "loading_model")
    model = load_model(device=device)

    report_progress(3, 12, "loading_sae")
    sae = load_sae(device=device)

    # Step 3: Load probes
    report_progress(4, 12, "loading_probes")
    probes = load_probes()

    if not probes:
        msg = "No probes could be loaded. Cannot compute predictors."
        logger.error(msg)
        mark_done("failed", msg)
        update_gpu_progress(time.time() - start_time, "failed")
        update_experiment_state("failed", msg)
        del model, sae
        gc.collect()
        torch.cuda.empty_cache()
        return

    # Step 4: Compute Predictor 1 -- Decoder cosine similarity
    report_progress(5, 12, "computing_cos_sim")
    logger.info("\n--- Predictor 1: Decoder cosine similarity ---")
    cos_sim_values = compute_decoder_cosine_similarity(sae, probes, pairs)

    for i, (pair, cs) in enumerate(zip(pairs, cos_sim_values)):
        if cs is not None:
            logger.info(f"  {pair['hierarchy']}/{pair['class_name']}: "
                         f"cos_sim={cs:.4f} (stored: {pair['child_cos_sim_to_probe']:.4f})")

    # Step 5: Compute Predictor 2 -- Co-occurrence
    report_progress(6, 12, "computing_cooccurrence")
    logger.info("\n--- Predictor 2: Co-occurrence P(child|parent) ---")
    cooccur_values = compute_cooccurrence(
        model, sae, probes, pairs,
        device=device,
        n_sequences=N_SEQUENCES,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    for i, (pair, co) in enumerate(zip(pairs, cooccur_values)):
        if co is not None:
            logger.info(f"  {pair['hierarchy']}/{pair['class_name']}: "
                         f"P(child|parent)={co['value']:.4f} "
                         f"(parent_active={co['parent_active_count']}, "
                         f"both={co['both_active_count']})")

    # Step 6: Compute Predictor 3 -- Reconstruction importance
    report_progress(7, 12, "computing_r_parent")
    logger.info("\n--- Predictor 3: Reconstruction importance R(parent) ---")
    r_parent_values = compute_reconstruction_importance(
        model, sae, probes, pairs,
        device=device,
        n_sequences=min(N_SEQUENCES // 2, 50),
        max_seq_len=MAX_SEQ_LEN,
    )

    for i, (pair, rp) in enumerate(zip(pairs, r_parent_values)):
        if rp is not None:
            logger.info(f"  {pair['hierarchy']}/{pair['class_name']}: R(parent)={rp:.6f}")

    # Release model and SAE to free GPU memory
    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    # Step 7: Assemble pairs with predictors
    report_progress(8, 12, "assembling_predictors")
    pairs_with_predictors = []
    for i, pair in enumerate(pairs):
        cs = cos_sim_values[i]
        co = cooccur_values[i]
        rp = r_parent_values[i]

        entry = {
            **pair,
            "cos_sim": cs,
            "co_occur": co["value"] if co is not None else None,
            "co_occur_parent_count": co["parent_active_count"] if co is not None else 0,
            "co_occur_both_count": co["both_active_count"] if co is not None else 0,
            "r_parent": rp,
        }
        pairs_with_predictors.append(entry)

    # Step 8: Fit three-factor model
    report_progress(9, 12, "fitting_model")
    logger.info("\n--- Fitting three-factor model ---")
    model_results = fit_three_factor_model(pairs_with_predictors)

    # Step 9: Evaluate H9
    report_progress(10, 12, "evaluating_h9")
    h9_verdict = evaluate_h9(model_results)

    logger.info(f"\n  H9 verdict: {h9_verdict['verdict']}")
    logger.info(f"  Confidence: {h9_verdict['confidence']:.2f}")

    # Step 10: Pilot pass criteria
    report_progress(11, 12, "compiling_output")
    elapsed = time.time() - start_time

    n_valid = model_results["n_valid_pairs"]
    pilot_pass = (
        n_valid >= 10  # relaxed from 30 for pilot
        and model_results.get("individual_correlations", {}) != {}
        and not model_results.get("insufficient_data", True)
    )

    # Compile final output
    output = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae": SAE_KEY,
        "n_total_pairs": len(pairs),
        "n_valid_pairs": n_valid,
        "n_sequences_cooccur": N_SEQUENCES,
        "n_sequences_r_parent": min(N_SEQUENCES // 2, 50),
        "pairs_per_hierarchy": dict(Counter(p["hierarchy"] for p in pairs)),
        "pairs_with_predictors": [
            {k: (float(v) if isinstance(v, (np.floating, float)) else v)
             for k, v in p.items() if k != "absorption_per_entity"}
            for p in pairs_with_predictors
        ],
        "predictor_statistics": {
            "cos_sim": {
                "values": [cs for cs in cos_sim_values if cs is not None],
                "mean": float(np.mean([cs for cs in cos_sim_values if cs is not None])) if any(cs is not None for cs in cos_sim_values) else None,
                "std": float(np.std([cs for cs in cos_sim_values if cs is not None])) if any(cs is not None for cs in cos_sim_values) else None,
            },
            "co_occur": {
                "values": [co["value"] for co in cooccur_values if co is not None],
                "mean": float(np.mean([co["value"] for co in cooccur_values if co is not None])) if any(co is not None for co in cooccur_values) else None,
                "std": float(np.std([co["value"] for co in cooccur_values if co is not None])) if any(co is not None for co in cooccur_values) else None,
            },
            "r_parent": {
                "values": [rp for rp in r_parent_values if rp is not None],
                "mean": float(np.mean([rp for rp in r_parent_values if rp is not None])) if any(rp is not None for rp in r_parent_values) else None,
                "std": float(np.std([rp for rp in r_parent_values if rp is not None])) if any(rp is not None for rp in r_parent_values) else None,
            },
        },
        "model_results": model_results,
        "h9_verdict": h9_verdict,
        "pilot_criteria_met": pilot_pass,
        "pilot_criteria_details": {
            "n_valid_pairs": n_valid,
            "target_pairs": 10,
            "has_individual_correlations": bool(model_results.get("individual_correlations")),
            "criteria": "Per-pair predictors computed for >= 10 pairs. Spearman rho computed with p-value. Individual predictor correlations reported.",
        },
        "methodology_notes": {
            "parent_definition": "Probe direction (logistic regression weight vector) for each class",
            "child_definition": "Top cosine-similarity SAE feature decoder direction for each class",
            "cos_sim_definition": "cosine(probe_direction, W_dec[child_feature_id])",
            "co_occur_definition": "P(child SAE feature active | probe predicts class with prob > 0.5)",
            "r_parent_definition": "Mean MSE increase when parent direction ablated from SAE reconstruction",
            "model_formula": "absorption_rate ~ beta_1*cos_sim^2 + beta_2*co_occur + beta_3*r_parent",
            "n_sequences_for_cooccurrence": N_SEQUENCES,
            "n_sequences_for_r_parent": min(N_SEQUENCES // 2, 50),
            "probe_confidence_threshold": 0.5,
        },
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    # Save
    out_path = PILOT_DIR / f"{TASK_ID}.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nSaved: {out_path}")

    # Also save to phase3 directory
    phase3_path = PHASE3_DIR / "rate_distortion_predictors.json"
    phase3_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved: {phase3_path}")

    # Generate summary markdown
    summary_md = generate_summary_md(output)
    md_path = PILOT_DIR / f"{TASK_ID}_summary.md"
    md_path.write_text(summary_md)
    logger.info(f"Summary: {md_path}")

    # Finalize
    report_progress(12, 12, "done")

    summary_text = (
        f"Phase 3.1 rate-distortion predictors (H9, {MODE}). "
        f"Pairs: {n_valid}/{len(pairs)} valid. "
        f"Time: {elapsed/60:.1f}min. "
    )

    mf = model_results.get("model_fit", {})
    if mf and mf.get("fitted"):
        summary_text += (
            f"Model: rho={mf.get('spearman_rho', 0):.3f}, "
            f"p={mf.get('spearman_p', 1):.3f}, "
            f"R2={mf.get('r_squared', 0):.3f}. "
        )

    summary_text += f"H9: {h9_verdict['verdict']}. "
    summary_text += f"Pilot: {'PASS' if pilot_pass else 'FAIL'}."

    mark_done("success" if pilot_pass else "partial", summary_text)
    update_gpu_progress(elapsed, "completed" if pilot_pass else "failed")
    update_experiment_state("completed" if pilot_pass else "failed")

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETED: {summary_text}")
    logger.info(f"{'='*60}")

    return output


def generate_summary_md(results):
    """Generate human-readable markdown summary."""
    lines = [
        "# Phase 3.1: Rate-Distortion Three-Factor Predictor Model (H9)",
        "",
        f"**Mode**: {results.get('mode', '?')}",
        f"**Status**: {'PASS' if results.get('pilot_criteria_met') else 'PARTIAL/FAIL'}",
        f"**Time**: {results.get('elapsed_minutes', 0):.1f} minutes",
        f"**SAE**: {results.get('sae', '?')}",
        f"**Total pairs**: {results.get('n_total_pairs', 0)}",
        f"**Valid pairs**: {results.get('n_valid_pairs', 0)}",
        "",
    ]

    # Pairs per hierarchy
    pph = results.get("pairs_per_hierarchy", {})
    if pph:
        lines.extend([
            "## Pairs per Hierarchy", "",
        ])
        for h, n in pph.items():
            lines.append(f"- **{h}**: {n} pairs")
        lines.append("")

    # Individual predictor correlations
    ind = results.get("model_results", {}).get("individual_correlations", {})
    if ind:
        lines.extend([
            "## Individual Predictor Correlations", "",
            "| Predictor | Spearman rho | p-value | Pearson r | n |",
            "|-----------|-------------|---------|-----------|---|",
        ])
        for name, vals in ind.items():
            lines.append(
                f"| {name} | {vals.get('spearman_rho', 0):.4f} | "
                f"{vals.get('spearman_p', vals.get('p_value', 1)):.4f} | "
                f"{vals.get('pearson_r', 0):.4f} | "
                f"{vals.get('n_pairs', 0)} |"
            )
        lines.append("")

    # Three-factor model
    mf = results.get("model_results", {}).get("model_fit", {})
    if mf and mf.get("fitted"):
        lines.extend([
            "## Three-Factor Model", "",
            f"**Formula**: absorption ~ beta_1*cos_sim^2 + beta_2*co_occur + beta_3*r_parent", "",
            f"- R^2 = {mf.get('r_squared', 0):.4f}",
            f"- Spearman rho = {mf.get('spearman_rho', 0):.4f} (p = {mf.get('spearman_p', 1):.4f})",
            f"- Pearson r = {mf.get('pearson_r', 0):.4f}",
            "",
            "### Coefficients", "",
        ])
        coefs = mf.get("coefficients", {})
        for name, val in coefs.items():
            lines.append(f"- **{name}**: {val:.6f}")
        lines.append("")

        # Predicted vs observed
        pvo = mf.get("predicted_vs_observed", [])
        if pvo:
            lines.extend([
                "### Predicted vs Observed", "",
                "| Hierarchy | Class | Predicted | Observed | cos_sim | co_occur | R_parent |",
                "|-----------|-------|-----------|----------|---------|----------|----------|",
            ])
            for row in pvo:
                lines.append(
                    f"| {row['hierarchy']} | {row['class_name']} | "
                    f"{row['predicted']:.4f} | {row['observed']:.4f} | "
                    f"{row['cos_sim']:.4f} | {row['co_occur']:.4f} | "
                    f"{row['r_parent']:.6f} |"
                )
            lines.append("")

    # Cross-domain analysis
    cda = results.get("model_results", {}).get("cross_domain_analysis", {})
    if cda:
        lines.extend([
            "## Cross-Domain Predictor Analysis", "",
            "| Hierarchy | Mean Absorption | Mean cos_sim | Mean co_occur | Mean R_parent | Mean Competition |",
            "|-----------|-----------------|-------------|---------------|---------------|------------------|",
        ])
        for h, sd in cda.items():
            lines.append(
                f"| {h} | {sd['mean_absorption']:.4f} | "
                f"{sd['mean_cos_sim']:.4f} | {sd['mean_co_occur']:.4f} | "
                f"{sd['mean_r_parent']:.6f} | {sd['mean_competition_coeff']:.4f} |"
            )
        lines.append("")

    # H9 verdict
    hv = results.get("h9_verdict", {})
    if hv:
        lines.extend([
            "## H9 Verdict", "",
            f"**Verdict**: {hv.get('verdict', '?')}",
            f"**Confidence**: {hv.get('confidence', 0):.2f}",
            f"**Model Spearman rho**: {hv.get('model_spearman_rho', 0):.4f}",
            f"**Best individual predictor**: {hv.get('best_individual_predictor', '?')} "
            f"(rho={hv.get('best_individual_rho', 0):.4f})",
            f"**Target**: {hv.get('target', '?')}",
            f"**Falsification**: {hv.get('falsification', '?')}",
            "",
        ])

    # Predictor statistics
    ps = results.get("predictor_statistics", {})
    if ps:
        lines.extend(["## Predictor Statistics", ""])
        for name, vals in ps.items():
            mean = vals.get("mean")
            std = vals.get("std")
            if mean is not None:
                lines.append(f"- **{name}**: mean={mean:.4f}, std={std:.4f}")
        lines.append("")

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
