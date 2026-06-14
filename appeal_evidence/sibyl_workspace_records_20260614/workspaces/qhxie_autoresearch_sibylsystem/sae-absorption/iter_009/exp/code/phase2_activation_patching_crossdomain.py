"""
Phase 2.1: Cross-Domain Activation Patching -- Iteration 9 Pilot

Extends iter_008's first-letter activation patching to city-continent hierarchy.
For absorbed entity-label pairs identified in Phase 1.2b (phase1_absorption_crossdomain):
  1. For each false-negative (FN) entity, identify child SAE features most aligned
     with the probe direction (top-5 cosine between decoder columns and probe weights).
  2. Zero child feature(s) in SAE activation space, decode back, re-encode, and check
     if the probe's parent prediction now recovers.
  3. Control: zero random non-child features of matched activation magnitude.
  4. Also try residual patching: subtract child's decoder contribution from raw
     residual stream, re-encode, and check parent recovery.
  5. Statistics: Wilcoxon signed-rank test, bootstrap CI, Cohen's d.

PILOT configuration:
- Hierarchy: city-continent at L24 (28.6% absorption rate from Phase 1)
- SAE: L24-16k JumpReLU
- Uses FN pairs from phase1_absorption_crossdomain pilot
- Target: >= 10 pairs processed
- Token position: -2 (matching probe training)

Prior evidence (iter_008):
- First-letter activation patching at L12: p=0.000218, d=1.33
- Cross-domain now extends to city-continent at L24

Dependencies:
- phase1_probe_training (COMPLETED): probe_city-continent_L24.npz
- phase1_absorption_crossdomain (COMPLETED): FN pairs + main feature IDs
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
from sklearn.linear_model import LogisticRegression
from scipy import stats

# ============================================================
# Configuration
# ============================================================
TASK_ID = "phase2_activation_patching_crossdomain"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE1_DIR = RESULTS_DIR / "phase1"
PHASE2_DIR = RESULTS_DIR / "phase2"
for d in [PILOT_DIR, PHASE2_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = os.environ.get("PILOT_MODE", "PILOT").upper()
if "--full" in sys.argv:
    MODE = "FULL"

# SAE config
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_24/width_16k/canonical"
SAE_KEY = "L24_16k"
SAE_LAYER = 24

# Token position -- must match probe training
TOKEN_POS = -2

# Number of ICL examples per prompt
N_ICL = 5

# Number of input contexts per entity
N_CONTEXTS = 50 if MODE == "PILOT" else 100

# Control parameters
N_CONTROL_FEATURES = 10

# Bootstrap
N_BOOTSTRAP = 2000 if MODE == "PILOT" else 10000

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Process tracking
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


def load_sae(release, sae_id, device="cuda:0"):
    from sae_lens import SAE
    logger.info(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release, sae_id, device=device)
    logger.info(f"  d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")
    return sae


# ============================================================
# Load pre-trained probe
# ============================================================
def load_probe(hierarchy_name, layer=24):
    """Load saved sklearn LogisticRegression probe."""
    probe_path = PHASE1_DIR / f"probe_{hierarchy_name}_L{layer}.npz"
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}")

    data = np.load(probe_path, allow_pickle=True)
    coef = data["coef"]
    intercept = data["intercept"]
    classes = data["classes"]

    probe = LogisticRegression(max_iter=1)
    probe.classes_ = np.arange(len(classes))
    probe.coef_ = coef
    probe.intercept_ = intercept

    logger.info(f"  Loaded probe {hierarchy_name}_L{layer}: "
                f"coef={coef.shape}, classes={list(classes)}")
    return probe, classes


# ============================================================
# Load Phase 1 absorption results to find FN pairs
# ============================================================
def load_phase1_fn_pairs():
    """Load false-negative entity-label pairs from phase1_absorption_crossdomain."""
    # Try pilot results first, then full
    for src in [PILOT_DIR / "phase1_absorption_crossdomain.json",
                PHASE1_DIR / "absorption_crossdomain.json"]:
        if src.exists():
            logger.info(f"Loading Phase 1 results from {src.name}")
            data = json.loads(src.read_text())
            break
    else:
        raise FileNotFoundError("No phase1_absorption_crossdomain results found")

    # Extract FN examples from L24_16k
    ar = data.get("absorption_results", {}).get("L24_16k", {})
    if "error" in ar:
        raise ValueError(f"L24_16k had error: {ar['error']}")

    fn_examples = ar.get("fn_examples", [])
    main_features_top = ar.get("main_features_top", {})
    per_class = ar.get("per_class", {})

    # Also get overall stats
    absorption_rate = ar.get("absorption_rate", 0)
    total_fn = ar.get("total_false_negatives", 0)
    total_correct = ar.get("total_probe_correct_raw", 0)

    logger.info(f"  Phase 1 L24_16k: absorption={absorption_rate:.4f}, "
                f"FN={total_fn}/{total_correct}")
    logger.info(f"  FN examples available: {len(fn_examples)}")
    logger.info(f"  Main features per class: {list(main_features_top.keys())}")

    return {
        "fn_examples": fn_examples,
        "main_features_top": main_features_top,
        "per_class": per_class,
        "absorption_rate": absorption_rate,
        "total_fn": total_fn,
        "total_correct": total_correct,
        "source": str(src),
    }


# ============================================================
# RAVEL data loading
# ============================================================
def prepare_ravel_data(max_entities=2000):
    """Load RAVEL dataset for city-continent."""
    from datasets import load_dataset

    logger.info("Loading RAVEL dataset (hij/ravel, city_entity)...")
    ds = load_dataset("hij/ravel", "city_entity", split="train")

    cities = list(ds["City"])
    continents = list(ds["Continent"])

    valid = [
        (c, l) for c, l in zip(cities, continents)
        if c and l and str(l).strip() and str(c).strip()
    ]
    cities_valid = [x[0] for x in valid]
    labels_valid = ["Oceania" if x[1] == "Australia" else x[1] for x in valid]

    label_counts = Counter(labels_valid)
    valid_labels = {l for l, n in label_counts.items() if n >= 3}
    keep = [(c, l) for c, l in zip(cities_valid, labels_valid) if l in valid_labels]

    return {
        "cities": [x[0] for x in keep],
        "labels": [x[1] for x in keep],
    }


# ============================================================
# ICL prompt construction with variation
# ============================================================
def build_icl_prompts_varied(city, label, all_cities, all_labels, n_contexts=50,
                              n_icl=5):
    """Build multiple varied ICL prompts for one city by varying ICL examples."""
    base_template = "The city of {entity} is on the continent of"
    answer_template = " {label}"

    examples = [(c, l) for c, l in zip(all_cities, all_labels) if c != city]
    rng = random.Random(SEED + hash(city))

    prompts = []
    for ctx_idx in range(n_contexts):
        rng_ctx = random.Random(SEED + hash(city) + ctx_idx * 7919)
        rng_ctx.shuffle(examples)
        icl_selected = examples[:n_icl]

        icl_parts = []
        for ex_city, ex_label in icl_selected:
            ex_text = base_template.format(entity=ex_city) + answer_template.format(label=ex_label)
            icl_parts.append(ex_text)

        full_prompt = "\n".join(icl_parts) + "\n" + base_template.format(entity=city)
        prompts.append(full_prompt)

    return prompts


# ============================================================
# Core patching experiment for one entity
# ============================================================
def patching_experiment_for_entity(
    model, sae, probe, class_labels,
    city, true_label, label_to_idx,
    all_cities, all_labels,
    main_feature_ids,
    n_contexts=50, n_icl=5,
    layer=24, token_pos=-2,
    n_control=10,
    device="cuda:0",
):
    """
    Activation patching for one city-continent entity.

    For each input context:
      1. Run model, get raw residual stream at L24 pos=-2
      2. Probe predicts on raw -> if correct (probe_correct_raw):
         a. Encode through SAE, decode, probe on SAE reconstruction
         b. If probe wrong on SAE reconstruction (false negative):
            - This is an absorption instance
            - Identify child features: features most aligned with probe direction for
              the TRUE class that are active (potential absorbers)
            - Zero child features, decode, re-encode, check if parent (correct class)
              probe prediction recovers
            - Control: zero random non-child feature of matched magnitude
    """
    tl_hook = f"blocks.{layer}.hook_resid_post"
    cls_list = class_labels.tolist() if hasattr(class_labels, 'tolist') else list(class_labels)

    if true_label not in cls_list:
        return {"status": "label_not_in_probe", "city": city, "true_label": true_label}

    probe_true_idx = cls_list.index(true_label)

    # Probe direction for the true class
    probe_coefs = torch.tensor(probe.coef_, dtype=torch.float32)
    probe_dir = probe_coefs[probe_true_idx]
    probe_dir_normalized = probe_dir / (probe_dir.norm() + 1e-8)

    # W_dec for cosine similarity
    W_dec = sae.W_dec.detach().float().cpu()

    # Build prompts
    prompts = build_icl_prompts_varied(
        city, true_label, all_cities, all_labels,
        n_contexts=n_contexts, n_icl=n_icl
    )

    # Per-context results
    context_results = []
    n_probe_correct_raw = 0
    n_probe_correct_sae = 0
    n_fn = 0
    n_fn_child_zeroed_recovered = 0
    n_fn_control_recovered = 0
    n_fn_residual_recovered = 0
    recovery_rates_child = []  # 1 if recovered, 0 if not (for FN contexts)
    recovery_rates_control = []

    for ctx_idx, prompt in enumerate(prompts):
        try:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])

            raw_act = cache[tl_hook][0, token_pos, :].detach().float()
            del cache

            # Probe on raw activation
            raw_np = raw_act.cpu().numpy().reshape(1, -1)
            raw_pred = probe.predict(raw_np)[0]
            probe_correct_raw = (raw_pred == probe_true_idx)

            if not probe_correct_raw:
                continue  # Only process probe-correct-raw contexts

            n_probe_correct_raw += 1

            # SAE encode/decode
            raw_act_device = raw_act.to(device)
            with torch.no_grad():
                sae_features = sae.encode(raw_act_device.unsqueeze(0))
                sae_out = sae.decode(sae_features)

            sae_np = sae_out[0].detach().float().cpu().numpy().reshape(1, -1)
            sae_pred = probe.predict(sae_np)[0]
            probe_correct_sae = (sae_pred == probe_true_idx)

            if probe_correct_sae:
                n_probe_correct_sae += 1
                continue  # Not a false negative, no absorption

            # ---- FALSE NEGATIVE (absorption instance) ----
            n_fn += 1
            sae_features_cpu = sae_features[0].detach().float().cpu()

            # Identify child features: active features aligned with probe direction
            active_mask = sae_features_cpu.abs() > 1e-6
            active_indices = torch.where(active_mask)[0].numpy()

            if len(active_indices) == 0:
                recovery_rates_child.append(0)
                recovery_rates_control.append(0)
                continue

            # Cosine similarity between each active feature's decoder column
            # and the probe direction
            active_decoder = W_dec[active_indices]  # [n_active, d_model]
            cos_sims = F.cosine_similarity(
                probe_dir_normalized.unsqueeze(0),
                active_decoder,
                dim=-1
            )

            # Child features: those with highest positive cosine (aligned with probe direction)
            # Select top-5 most aligned active features
            n_child_candidates = min(5, len(active_indices))
            top_cos_vals, top_cos_idx = cos_sims.topk(n_child_candidates)
            child_feature_indices = active_indices[top_cos_idx.numpy()]
            child_cos_sims = top_cos_vals.numpy()

            # Also use main features from Phase 1 if they are active
            main_fids_for_class = main_feature_ids.get(true_label, [])
            active_main_fids = [
                fid for fid in main_fids_for_class
                if fid in active_indices
            ]

            # Combine: child features = union of probe-aligned + active main features
            all_child_fids = list(set(child_feature_indices.tolist()) | set(active_main_fids))

            if len(all_child_fids) == 0:
                recovery_rates_child.append(0)
                recovery_rates_control.append(0)
                continue

            # ---- PATCHING: Zero child features ----
            patched_sae_act = sae_features_cpu.clone()
            for fid in all_child_fids:
                patched_sae_act[fid] = 0.0

            with torch.no_grad():
                patched_out = sae.decode(patched_sae_act.unsqueeze(0).to(device))

            patched_np = patched_out[0].detach().float().cpu().numpy().reshape(1, -1)
            patched_pred = probe.predict(patched_np)[0]
            child_zeroed_recovered = (patched_pred == probe_true_idx)

            if child_zeroed_recovered:
                n_fn_child_zeroed_recovered += 1
            recovery_rates_child.append(1 if child_zeroed_recovered else 0)

            # ---- RESIDUAL PATCHING: Subtract child decoder contribution from raw ----
            raw_act_patched = raw_act.clone().to(device)
            for fid in all_child_fids:
                child_act_val = sae_features_cpu[fid].item()
                child_dec_vec = W_dec[fid].to(device).to(raw_act_patched.dtype)
                raw_act_patched = raw_act_patched - child_act_val * child_dec_vec

            with torch.no_grad():
                resid_re_encoded = sae.encode(raw_act_patched.unsqueeze(0))
                resid_decoded = sae.decode(resid_re_encoded)

            resid_np = resid_decoded[0].detach().float().cpu().numpy().reshape(1, -1)
            resid_pred = probe.predict(resid_np)[0]
            residual_recovered = (resid_pred == probe_true_idx)

            if residual_recovered:
                n_fn_residual_recovered += 1

            # ---- CONTROL: Zero random non-child feature ----
            non_child_active = [
                int(idx) for idx in active_indices
                if idx not in all_child_fids
            ]

            ctrl_recovered_any = False
            ctrl_recovered_count = 0
            ctrl_total = 0

            if len(non_child_active) > 0:
                rng_ctrl = np.random.RandomState(SEED + ctx_idx)
                n_ctrl = min(n_control, len(non_child_active))
                ctrl_indices = rng_ctrl.choice(non_child_active, size=n_ctrl, replace=False)

                for ctrl_idx in ctrl_indices:
                    ctrl_sae_act = sae_features_cpu.clone()
                    ctrl_sae_act[ctrl_idx] = 0.0

                    with torch.no_grad():
                        ctrl_out = sae.decode(ctrl_sae_act.unsqueeze(0).to(device))

                    ctrl_np = ctrl_out[0].detach().float().cpu().numpy().reshape(1, -1)
                    ctrl_pred = probe.predict(ctrl_np)[0]
                    if ctrl_pred == probe_true_idx:
                        ctrl_recovered_count += 1
                        ctrl_recovered_any = True
                    ctrl_total += 1

            control_recovery_rate = ctrl_recovered_count / max(ctrl_total, 1)
            recovery_rates_control.append(control_recovery_rate)

            if ctrl_recovered_any:
                n_fn_control_recovered += 1

            # Store detailed context result (first 10 only to save space)
            if len(context_results) < 10:
                context_results.append({
                    "ctx_idx": ctx_idx,
                    "n_active_features": len(active_indices),
                    "n_child_features": len(all_child_fids),
                    "top_child_cos_sims": child_cos_sims[:3].tolist(),
                    "child_zeroed_recovered": bool(child_zeroed_recovered),
                    "residual_recovered": bool(residual_recovered),
                    "control_recovery_rate": float(control_recovery_rate),
                    "sae_pred_label": cls_list[sae_pred] if sae_pred < len(cls_list) else f"idx_{sae_pred}",
                    "patched_pred_label": cls_list[patched_pred] if patched_pred < len(cls_list) else f"idx_{patched_pred}",
                })

            del patched_out, resid_re_encoded, resid_decoded
            torch.cuda.empty_cache()

        except Exception as e:
            if ctx_idx < 3:
                logger.warning(f"  Context {ctx_idx} error: {e}")
            continue

    # Aggregate for this entity
    child_recovery_rate = n_fn_child_zeroed_recovered / max(n_fn, 1)
    control_aggregate_rate = n_fn_control_recovered / max(n_fn, 1)
    residual_recovery_rate = n_fn_residual_recovered / max(n_fn, 1)

    return {
        "status": "completed",
        "city": city,
        "true_label": true_label,
        "n_contexts_total": len(prompts),
        "n_probe_correct_raw": n_probe_correct_raw,
        "n_probe_correct_sae": n_probe_correct_sae,
        "n_false_negatives": n_fn,
        "absorption_rate_raw": n_fn / max(n_probe_correct_raw, 1),
        "n_fn_child_zeroed_recovered": n_fn_child_zeroed_recovered,
        "n_fn_control_recovered": n_fn_control_recovered,
        "n_fn_residual_recovered": n_fn_residual_recovered,
        "child_recovery_rate": float(child_recovery_rate),
        "control_recovery_rate": float(control_aggregate_rate),
        "residual_recovery_rate": float(residual_recovery_rate),
        "recovery_rates_child": recovery_rates_child,
        "recovery_rates_control": recovery_rates_control,
        "context_examples": context_results,
    }


# ============================================================
# Bootstrap CI
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
# GPU progress tracking
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
                "planned_min": 15,
                "actual_min": round(elapsed_seconds / 60, 1),
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "config_snapshot": {
                    "model": "gemma-2-2b",
                    "mode": MODE,
                    "sae": SAE_KEY,
                    "layer": SAE_LAYER,
                    "token_pos": TOKEN_POS,
                    "n_contexts": N_CONTEXTS,
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
# Main
# ============================================================
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 12, "starting")

    logger.info("=" * 70)
    logger.info("Phase 2.1: Cross-Domain Activation Patching (city-continent, iter_009 pilot)")
    logger.info(f"Mode: {MODE}, Layer: {SAE_LAYER}, Token pos: {TOKEN_POS}")
    logger.info(f"N_CONTEXTS per entity: {N_CONTEXTS}, N_CONTROL: {N_CONTROL_FEATURES}")
    logger.info("=" * 70)

    device = "cuda:0"

    # Step 1: Load model
    report_progress(1, 12, "loading_model")
    model = load_model(device=device)

    # Step 2: Load probe
    report_progress(2, 12, "loading_probe")
    probe, class_labels = load_probe("city-continent", layer=SAE_LAYER)
    cls_list = class_labels.tolist() if hasattr(class_labels, 'tolist') else list(class_labels)
    logger.info(f"  Probe classes: {cls_list}")

    # Step 3: Load Phase 1 results to identify FN entities
    report_progress(3, 12, "loading_phase1_fn_pairs")
    phase1_data = load_phase1_fn_pairs()
    fn_examples = phase1_data["fn_examples"]
    main_features_top = phase1_data["main_features_top"]

    # Build main feature ID lookup
    main_feature_ids = {}
    for cls_name, mf in main_features_top.items():
        if isinstance(mf, dict):
            fids = mf.get("top5_fids", [mf.get("fid", -1)])
            main_feature_ids[cls_name] = [int(f) for f in fids]

    logger.info(f"  FN examples from Phase 1: {len(fn_examples)}")
    logger.info(f"  Main feature IDs: {main_feature_ids}")

    # Step 4: Load SAE
    report_progress(4, 12, "loading_sae")
    sae = load_sae(SAE_RELEASE, SAE_ID, device=device)

    # Step 5: Load RAVEL data for ICL prompts
    report_progress(5, 12, "loading_ravel_data")
    ravel_data = prepare_ravel_data()
    all_cities = ravel_data["cities"]
    all_labels = ravel_data["labels"]

    # Build label_to_idx using probe class ordering
    label_to_idx = {l: cls_list.index(l) for l in set(all_labels) if l in cls_list}

    # Get FN cities to patch -- use both the explicitly recorded FN examples AND
    # re-identify FN entities by re-running absorption on a broader set
    fn_cities = set()
    fn_city_labels = {}
    for fn in fn_examples:
        city = fn.get("city", "")
        label = fn.get("true_label", "")
        if city and label and label in label_to_idx:
            fn_cities.add(city)
            fn_city_labels[city] = label

    logger.info(f"  Initial FN cities from Phase 1 examples: {len(fn_cities)}")

    # Also find additional FN entities by scanning the full RAVEL dataset
    # (Phase 1 only stored 30 examples, we may have more)
    report_progress(6, 12, "identifying_additional_fn_entities")
    logger.info("  Scanning for additional FN entities...")

    tl_hook = f"blocks.{SAE_LAYER}.hook_resid_post"
    additional_fn_found = 0
    scan_count = 0
    scan_limit = 200  # Scan up to 200 entities for additional FN pairs

    # Prioritize classes with known high absorption (Europe had 100% in Phase 1)
    per_class = phase1_data.get("per_class", {})
    high_absorption_classes = sorted(
        per_class.keys(),
        key=lambda c: per_class[c].get("absorption_rate", 0),
        reverse=True
    )
    logger.info(f"  Absorption rates by class: {[(c, per_class[c].get('absorption_rate', 0)) for c in high_absorption_classes]}")

    # Build city -> label map from RAVEL
    city_label_map = dict(zip(all_cities, all_labels))

    # Scan entities not already in FN set
    entities_to_scan = [
        (c, city_label_map[c]) for c in all_cities
        if c not in fn_cities and city_label_map[c] in label_to_idx
    ]
    # Prioritize high-absorption classes
    class_priority = {c: i for i, c in enumerate(high_absorption_classes)}
    entities_to_scan.sort(key=lambda x: class_priority.get(x[1], 99))

    for city, label in entities_to_scan[:scan_limit]:
        try:
            base_template = "The city of {entity} is on the continent of"
            prompt = base_template.format(entity=city)
            tokens = model.to_tokens(prompt, prepend_bos=True)

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])

            raw_act = cache[tl_hook][0, TOKEN_POS, :].detach().float()
            del cache

            raw_np = raw_act.cpu().numpy().reshape(1, -1)
            raw_pred = probe.predict(raw_np)[0]
            probe_true_idx = cls_list.index(label)

            if raw_pred != probe_true_idx:
                scan_count += 1
                continue

            # Probe correct on raw -- check SAE
            with torch.no_grad():
                sae_features = sae.encode(raw_act.to(device).unsqueeze(0))
                sae_out = sae.decode(sae_features)

            sae_np = sae_out[0].detach().float().cpu().numpy().reshape(1, -1)
            sae_pred = probe.predict(sae_np)[0]

            if sae_pred != probe_true_idx:
                # Found a new FN entity
                fn_cities.add(city)
                fn_city_labels[city] = label
                additional_fn_found += 1

            scan_count += 1
            if scan_count % 50 == 0:
                torch.cuda.empty_cache()

        except Exception:
            scan_count += 1
            continue

    logger.info(f"  Scanned {scan_count} additional entities, found {additional_fn_found} new FN entities")
    logger.info(f"  Total FN entities for patching: {len(fn_cities)}")

    # Step 7: Run activation patching on each FN entity
    report_progress(7, 12, "activation_patching",
                    {"n_fn_entities": len(fn_cities)})

    entity_results = []
    all_child_recovery = []
    all_control_recovery = []
    all_residual_recovery = []

    fn_city_list = sorted(fn_cities)
    logger.info(f"\n{'='*70}")
    logger.info(f"Running activation patching on {len(fn_city_list)} FN entities...")
    logger.info(f"{'='*70}")

    for i, city in enumerate(fn_city_list):
        label = fn_city_labels[city]
        logger.info(f"\n  [{i+1}/{len(fn_city_list)}] Patching '{city}' (true: {label})...")

        report_progress(7, 12, f"patching_{i+1}/{len(fn_city_list)}",
                        {"city": city, "label": label})

        result = patching_experiment_for_entity(
            model=model,
            sae=sae,
            probe=probe,
            class_labels=class_labels,
            city=city,
            true_label=label,
            label_to_idx=label_to_idx,
            all_cities=all_cities,
            all_labels=all_labels,
            main_feature_ids=main_feature_ids,
            n_contexts=N_CONTEXTS,
            n_icl=N_ICL,
            layer=SAE_LAYER,
            token_pos=TOKEN_POS,
            n_control=N_CONTROL_FEATURES,
            device=device,
        )

        entity_results.append(result)

        if result["status"] == "completed":
            logger.info(f"    FN: {result['n_false_negatives']}/{result['n_probe_correct_raw']}")
            logger.info(f"    Child-zeroed recovery: {result['n_fn_child_zeroed_recovered']}/{result['n_false_negatives']} "
                        f"({result['child_recovery_rate']:.4f})")
            logger.info(f"    Control recovery: {result['n_fn_control_recovered']}/{result['n_false_negatives']} "
                        f"({result['control_recovery_rate']:.4f})")
            logger.info(f"    Residual recovery: {result['n_fn_residual_recovered']}/{result['n_false_negatives']} "
                        f"({result['residual_recovery_rate']:.4f})")

            # Collect per-FN-context recovery rates
            all_child_recovery.extend(result["recovery_rates_child"])
            all_control_recovery.extend(result["recovery_rates_control"])
            if result["n_false_negatives"] > 0:
                all_residual_recovery.append(result["residual_recovery_rate"])
        else:
            logger.info(f"    Status: {result['status']}")

        if (i + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Step 8: Aggregate statistics
    report_progress(8, 12, "aggregating_statistics")
    logger.info(f"\n{'='*70}")
    logger.info("Aggregate Statistics")
    logger.info(f"{'='*70}")

    completed = [r for r in entity_results if r["status"] == "completed"]
    n_completed = len(completed)
    total_fn = sum(r["n_false_negatives"] for r in completed)
    total_child_recovered = sum(r["n_fn_child_zeroed_recovered"] for r in completed)
    total_control_recovered = sum(r["n_fn_control_recovered"] for r in completed)
    total_residual_recovered = sum(r["n_fn_residual_recovered"] for r in completed)
    total_probe_correct = sum(r["n_probe_correct_raw"] for r in completed)

    overall_child_recovery_rate = total_child_recovered / max(total_fn, 1)
    overall_control_rate = total_control_recovered / max(total_fn, 1)
    overall_residual_rate = total_residual_recovered / max(total_fn, 1)

    logger.info(f"  Entities completed: {n_completed}/{len(fn_city_list)}")
    logger.info(f"  Total FN instances across contexts: {total_fn}")
    logger.info(f"  Child-zeroed recovery: {total_child_recovered}/{total_fn} "
                f"({overall_child_recovery_rate:.4f})")
    logger.info(f"  Control recovery: {total_control_recovered}/{total_fn} "
                f"({overall_control_rate:.4f})")
    logger.info(f"  Residual recovery: {total_residual_recovered}/{total_fn} "
                f"({overall_residual_rate:.4f})")

    # Bootstrap CI on per-FN-context child recovery
    child_ci = bootstrap_ci(all_child_recovery, n_bootstrap=N_BOOTSTRAP, seed=SEED)
    control_ci = bootstrap_ci(all_control_recovery, n_bootstrap=N_BOOTSTRAP, seed=SEED)

    logger.info(f"  Child recovery CI: [{child_ci['ci_lower']:.4f}, {child_ci['ci_upper']:.4f}]")
    logger.info(f"  Control recovery CI: [{control_ci['ci_lower']:.4f}, {control_ci['ci_upper']:.4f}]")

    # Step 9: Statistical tests
    report_progress(9, 12, "statistical_tests")

    stat_tests = {}

    # Wilcoxon signed-rank: child recovery vs control recovery per entity
    child_rates = [r["child_recovery_rate"] for r in completed if r["n_false_negatives"] > 0]
    control_rates = [r["control_recovery_rate"] for r in completed if r["n_false_negatives"] > 0]

    if len(child_rates) >= 3 and len(control_rates) >= 3:
        child_arr = np.array(child_rates)
        control_arr = np.array(control_rates)
        diffs = child_arr - control_arr

        # Only run Wilcoxon if there are non-zero differences
        non_zero_diffs = diffs[diffs != 0]
        if len(non_zero_diffs) >= 3:
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                    child_arr, control_arr, alternative='greater'
                )
                stat_tests["wilcoxon_signed_rank"] = {
                    "statistic": float(wilcoxon_stat),
                    "p_value": float(wilcoxon_p),
                    "n_pairs": len(child_rates),
                    "n_nonzero_diffs": len(non_zero_diffs),
                    "alternative": "greater (child > control)",
                    "significant_005": wilcoxon_p < 0.05,
                    "significant_001": wilcoxon_p < 0.01,
                }
                logger.info(f"\n  Wilcoxon signed-rank test:")
                logger.info(f"    stat={wilcoxon_stat:.4f}, p={wilcoxon_p:.6f}")
                logger.info(f"    Significant (p<0.05): {wilcoxon_p < 0.05}")
            except Exception as e:
                stat_tests["wilcoxon_signed_rank"] = {"error": str(e)}
                logger.warning(f"  Wilcoxon test failed: {e}")
        else:
            stat_tests["wilcoxon_signed_rank"] = {
                "note": f"Too few non-zero differences ({len(non_zero_diffs)})",
                "n_pairs": len(child_rates),
            }
            logger.info(f"  Wilcoxon: too few non-zero differences ({len(non_zero_diffs)})")

        # Cohen's d (paired)
        if np.std(diffs) > 0:
            cohens_d = float(np.mean(diffs) / np.std(diffs))
        else:
            cohens_d = float('inf') if np.mean(diffs) > 0 else 0.0

        stat_tests["cohens_d"] = {
            "value": cohens_d,
            "mean_diff": float(np.mean(diffs)),
            "std_diff": float(np.std(diffs)),
            "interpretation": (
                "large" if abs(cohens_d) >= 0.8 else
                "medium" if abs(cohens_d) >= 0.5 else
                "small" if abs(cohens_d) >= 0.2 else "negligible"
            ),
        }
        logger.info(f"  Cohen's d: {cohens_d:.4f} ({stat_tests['cohens_d']['interpretation']})")

        # Permutation test on recovery rate difference
        n_perm = 5000
        observed_diff = np.mean(child_arr) - np.mean(control_arr)
        combined = np.concatenate([child_arr, control_arr])
        n_child = len(child_arr)
        rng_perm = np.random.RandomState(SEED)
        n_extreme = 0
        for _ in range(n_perm):
            rng_perm.shuffle(combined)
            perm_diff = np.mean(combined[:n_child]) - np.mean(combined[n_child:])
            if perm_diff >= observed_diff:
                n_extreme += 1
        perm_p = (n_extreme + 1) / (n_perm + 1)

        stat_tests["permutation_test"] = {
            "observed_diff": float(observed_diff),
            "p_value": float(perm_p),
            "n_permutations": n_perm,
            "significant_005": perm_p < 0.05,
        }
        logger.info(f"  Permutation test: diff={observed_diff:.4f}, p={perm_p:.4f}")

    else:
        stat_tests["note"] = f"Too few entities with FN ({len(child_rates)}) for statistical tests"
        logger.info(f"  Too few entities for statistical tests: {len(child_rates)}")

    # Step 10: Per-class breakdown
    report_progress(10, 12, "per_class_analysis")

    per_class_patching = {}
    for r in completed:
        if r["n_false_negatives"] > 0:
            label = r["true_label"]
            if label not in per_class_patching:
                per_class_patching[label] = {
                    "n_entities": 0,
                    "total_fn": 0,
                    "total_child_recovered": 0,
                    "total_control_recovered": 0,
                    "total_residual_recovered": 0,
                    "entity_child_rates": [],
                    "entity_control_rates": [],
                }
            pcd = per_class_patching[label]
            pcd["n_entities"] += 1
            pcd["total_fn"] += r["n_false_negatives"]
            pcd["total_child_recovered"] += r["n_fn_child_zeroed_recovered"]
            pcd["total_control_recovered"] += r["n_fn_control_recovered"]
            pcd["total_residual_recovered"] += r["n_fn_residual_recovered"]
            pcd["entity_child_rates"].append(r["child_recovery_rate"])
            pcd["entity_control_rates"].append(r["control_recovery_rate"])

    for label, pcd in per_class_patching.items():
        pcd["mean_child_rate"] = float(np.mean(pcd["entity_child_rates"])) if pcd["entity_child_rates"] else 0.0
        pcd["mean_control_rate"] = float(np.mean(pcd["entity_control_rates"])) if pcd["entity_control_rates"] else 0.0
        pcd["child_rate_overall"] = pcd["total_child_recovered"] / max(pcd["total_fn"], 1)
        pcd["control_rate_overall"] = pcd["total_control_recovered"] / max(pcd["total_fn"], 1)
        pcd["residual_rate_overall"] = pcd["total_residual_recovered"] / max(pcd["total_fn"], 1)
        # Clean up list fields
        del pcd["entity_child_rates"]
        del pcd["entity_control_rates"]

    logger.info(f"\n  Per-class patching results:")
    for label in sorted(per_class_patching.keys()):
        pcd = per_class_patching[label]
        logger.info(f"    {label}: {pcd['n_entities']} entities, "
                    f"FN={pcd['total_fn']}, "
                    f"child_recovery={pcd['child_rate_overall']:.4f}, "
                    f"control={pcd['control_rate_overall']:.4f}")

    # Step 11: Comparison with iter_008 first-letter results
    report_progress(11, 12, "comparison_with_firstletter")

    firstletter_comparison = {}
    fl_path = PILOT_DIR / "phase1_absorption_firstletter.json"
    if fl_path.exists():
        try:
            fl_data = json.loads(fl_path.read_text())
            fl_l24 = fl_data.get("absorption_results", {}).get("L24_16k", {})
            if "absorption_rate" in fl_l24:
                firstletter_comparison = {
                    "firstletter_absorption_rate": fl_l24["absorption_rate"],
                    "crossdomain_absorption_rate": phase1_data["absorption_rate"],
                    "note": "First-letter patching was at L12 in iter_008 (p=0.000218). "
                            "Cross-domain now at L24.",
                }
                logger.info(f"\n  First-letter comparison:")
                logger.info(f"    First-letter L24 absorption: {fl_l24['absorption_rate']:.4f}")
                logger.info(f"    Cross-domain L24 absorption: {phase1_data['absorption_rate']:.4f}")
        except Exception as e:
            logger.warning(f"  Could not load first-letter data: {e}")

    # Step 12: Compile output
    report_progress(12, 12, "compiling_output")

    elapsed = time.time() - start_time

    # Pilot pass criteria
    has_enough_entities = n_completed >= 10
    has_fn_data = total_fn >= 5
    has_significant_result = stat_tests.get("wilcoxon_signed_rank", {}).get("significant_005", False)
    has_recovery_above_control = overall_child_recovery_rate > overall_control_rate

    pilot_pass = has_enough_entities and has_fn_data and has_recovery_above_control

    output = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae": {"release": SAE_RELEASE, "sae_id": SAE_ID, "key": SAE_KEY},
        "layer": SAE_LAYER,
        "token_pos": TOKEN_POS,
        "n_contexts_per_entity": N_CONTEXTS,
        "n_control_features": N_CONTROL_FEATURES,
        "hierarchy": "city-continent",
        "phase1_source": phase1_data["source"],
        "entity_counts": {
            "n_fn_entities_identified": len(fn_city_list),
            "n_completed": n_completed,
            "n_skipped": len(fn_city_list) - n_completed,
        },
        "aggregate": {
            "total_fn_instances": total_fn,
            "total_probe_correct": total_probe_correct,
            "child_zeroed_recovery": {
                "n_recovered": total_child_recovered,
                "rate": float(overall_child_recovery_rate),
                "bootstrap_ci": child_ci,
            },
            "control_recovery": {
                "n_recovered": total_control_recovered,
                "rate": float(overall_control_rate),
                "bootstrap_ci": control_ci,
            },
            "residual_recovery": {
                "n_recovered": total_residual_recovered,
                "rate": float(overall_residual_rate),
            },
            "recovery_difference": float(overall_child_recovery_rate - overall_control_rate),
        },
        "statistical_tests": stat_tests,
        "per_class_patching": per_class_patching,
        "per_entity_results": [
            {k: v for k, v in r.items() if k != "recovery_rates_child" and k != "recovery_rates_control"}
            for r in entity_results
        ],
        "firstletter_comparison": firstletter_comparison,
        "pilot_criteria": {
            "met": pilot_pass,
            "details": {
                "enough_entities_10": has_enough_entities,
                "enough_fn_5": has_fn_data,
                "recovery_above_control": has_recovery_above_control,
                "wilcoxon_significant": has_significant_result,
            },
        },
        "interpretation": {
            "verdict": (
                "CAUSAL_ABSORPTION_CONFIRMED" if has_significant_result and has_recovery_above_control
                else "EVIDENCE_FOR_ABSORPTION" if has_recovery_above_control
                else "INSUFFICIENT_EVIDENCE"
            ),
            "detail": (
                f"Zeroing child features recovers correct continent prediction in "
                f"{overall_child_recovery_rate:.1%} of absorbed contexts (vs "
                f"{overall_control_rate:.1%} control). "
                f"{'Wilcoxon p < 0.05 confirms causal effect.' if has_significant_result else 'Statistical significance not reached with current sample size.'}"
            ),
        },
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    # Save
    out_path = PILOT_DIR / f"{TASK_ID}.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nSaved: {out_path}")

    # Also save to phase2 directory
    phase2_path = PHASE2_DIR / f"activation_patching_crossdomain.json"
    phase2_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Also saved: {phase2_path}")

    # Generate summary markdown
    summary_md = generate_summary_md(output)
    md_path = PILOT_DIR / f"{TASK_ID}_summary.md"
    md_path.write_text(summary_md)
    logger.info(f"Summary: {md_path}")

    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    summary_text = (
        f"Phase 2.1 cross-domain activation patching (city-continent, {MODE}). "
        f"Entities: {n_completed}, FN instances: {total_fn}. "
        f"Child recovery: {overall_child_recovery_rate:.4f} "
        f"(CI [{child_ci['ci_lower']:.3f},{child_ci['ci_upper']:.3f}]). "
        f"Control: {overall_control_rate:.4f}. "
        f"Diff: {overall_child_recovery_rate - overall_control_rate:+.4f}. "
    )
    if "wilcoxon_signed_rank" in stat_tests and "p_value" in stat_tests["wilcoxon_signed_rank"]:
        summary_text += f"Wilcoxon p={stat_tests['wilcoxon_signed_rank']['p_value']:.4f}. "
    if "cohens_d" in stat_tests:
        summary_text += f"Cohen's d={stat_tests['cohens_d']['value']:.4f}. "
    summary_text += f"Verdict: {output['interpretation']['verdict']}. "
    summary_text += f"Pilot: {'PASS' if pilot_pass else 'FAIL'}. "
    summary_text += f"Time: {elapsed/60:.1f}min."

    mark_done("success" if pilot_pass else "partial", summary_text)
    update_gpu_progress(elapsed, "completed" if pilot_pass else "failed")
    update_experiment_state("completed" if pilot_pass else "failed")

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETED: {summary_text}")
    logger.info(f"{'='*70}")

    return output


def generate_summary_md(results):
    """Generate human-readable markdown summary."""
    lines = [
        "# Phase 2.1: Cross-Domain Activation Patching -- Iter 9 Pilot",
        "",
        f"**Verdict**: {results.get('interpretation', {}).get('verdict', 'UNKNOWN')}",
        f"**Pilot**: {'PASS' if results.get('pilot_criteria', {}).get('met') else 'FAIL'}",
        f"**Time**: {results.get('elapsed_minutes', 0):.1f} minutes",
        "",
        "## Design",
        "",
        f"- **Hierarchy**: city-continent at L{results.get('layer', '?')}",
        f"- **SAE**: {results.get('sae', {}).get('key', '?')}",
        f"- **Token position**: {results.get('token_pos', '?')}",
        f"- **Contexts per entity**: {results.get('n_contexts_per_entity', '?')}",
        "",
        "## Aggregate Results",
        "",
    ]

    agg = results.get("aggregate", {})
    lines.extend([
        f"- **FN instances**: {agg.get('total_fn_instances', 0)}",
        f"- **Child-zeroed recovery**: {agg.get('child_zeroed_recovery', {}).get('rate', 0):.4f} "
        f"(CI [{agg.get('child_zeroed_recovery', {}).get('bootstrap_ci', {}).get('ci_lower', 0):.3f}, "
        f"{agg.get('child_zeroed_recovery', {}).get('bootstrap_ci', {}).get('ci_upper', 0):.3f}])",
        f"- **Control recovery**: {agg.get('control_recovery', {}).get('rate', 0):.4f} "
        f"(CI [{agg.get('control_recovery', {}).get('bootstrap_ci', {}).get('ci_lower', 0):.3f}, "
        f"{agg.get('control_recovery', {}).get('bootstrap_ci', {}).get('ci_upper', 0):.3f}])",
        f"- **Residual recovery**: {agg.get('residual_recovery', {}).get('rate', 0):.4f}",
        f"- **Recovery difference**: {agg.get('recovery_difference', 0):+.4f}",
    ])

    # Statistical tests
    st = results.get("statistical_tests", {})
    if "wilcoxon_signed_rank" in st and "p_value" in st["wilcoxon_signed_rank"]:
        ws = st["wilcoxon_signed_rank"]
        lines.extend([
            "", "## Statistical Tests", "",
            f"- **Wilcoxon signed-rank**: p={ws['p_value']:.6f} "
            f"(n={ws.get('n_pairs', '?')} pairs, "
            f"significant p<0.05: {ws.get('significant_005', False)})",
        ])
    if "cohens_d" in st:
        cd = st["cohens_d"]
        lines.append(f"- **Cohen's d**: {cd['value']:.4f} ({cd['interpretation']})")
    if "permutation_test" in st:
        pt = st["permutation_test"]
        lines.append(f"- **Permutation test**: diff={pt['observed_diff']:.4f}, p={pt['p_value']:.4f}")

    # Per-class breakdown
    pcp = results.get("per_class_patching", {})
    if pcp:
        lines.extend([
            "", "## Per-Class Patching", "",
            "| Class | Entities | FN | Child Recovery | Control | Residual |",
            "|-------|----------|-----|---------------|---------|----------|",
        ])
        for cls in sorted(pcp.keys()):
            d = pcp[cls]
            lines.append(
                f"| {cls} | {d['n_entities']} | {d['total_fn']} | "
                f"{d['child_rate_overall']:.4f} | {d['control_rate_overall']:.4f} | "
                f"{d['residual_rate_overall']:.4f} |"
            )

    # Per-entity table
    entity_results = results.get("per_entity_results", [])
    completed_entities = [r for r in entity_results if r.get("status") == "completed" and r.get("n_false_negatives", 0) > 0]
    if completed_entities:
        lines.extend([
            "", "## Per-Entity Results", "",
            "| City | Label | FN | Child Rec | Control Rec | Residual Rec |",
            "|------|-------|-----|-----------|-------------|--------------|",
        ])
        for r in completed_entities[:30]:
            lines.append(
                f"| {r['city']} | {r['true_label']} | {r['n_false_negatives']} | "
                f"{r['child_recovery_rate']:.4f} | {r['control_recovery_rate']:.4f} | "
                f"{r['residual_recovery_rate']:.4f} |"
            )

    # Interpretation
    interp = results.get("interpretation", {})
    lines.extend([
        "", "## Interpretation", "",
        f"**{interp.get('verdict', 'UNKNOWN')}**: {interp.get('detail', '')}",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        try:
            update_gpu_progress(0, "failed")
            update_experiment_state("failed", str(e))
        except Exception:
            pass
        sys.exit(1)
