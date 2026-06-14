#!/usr/bin/env python3
"""
Phase 1.1 PILOT: Cross-Domain Patching Spot-Check (20 City-Continent Entities)
===============================================================================

Verify the sign-reversed city-continent activation patching data from iter_009:
  - iter_009 pilot: d=-0.91 (FAILED - bug: zeroed features supporting correct class)
  - iter_009 corrected FULL: d=+1.50, mean primary recovery=61.9%

This is the single highest-ROI verification task in the entire iteration.

Method:
  1. Randomly sample 20 city-continent entities (seed=42, stratified by continent)
  2. Load Gemma 2 2B + Gemma Scope L24 16k SAE using TransformerLens + SAELens
  3. For each entity:
     a. Run forward pass, collect L24 activations
     b. Encode through SAE, identify top child feature (absorber) via contribution analysis
     c. Zero the child feature activation
     d. Decode back to L24
     e. Measure parent probe prediction recovery on absorbed contexts
  4. Control: zero a random non-child feature (magnitude-matched) for each entity
  5. Report: mean recovery rate, Cohen's d, p-value (paired t-test)
  6. Compare with claimed 61.9% recovery rate

Pass criteria:
  - Mean recovery rate within 10 pp of 61.9%
  - Cohen's d > 0.5
  - p < 0.01
  - If recovery off by > 15 pp: FLAG CRITICAL

MODE: PILOT (20 entities spot-check)
GPU: cuda:0
"""

import gc
import json
import os
import sys
import time
import random
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import LogisticRegression

# ============================================================
# Configuration
# ============================================================
TASK_ID = "phase1_patching_spotcheck"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE1_DIR = RESULTS_DIR / "phase1"
for d in [PILOT_DIR, PHASE1_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Path to iter_009 probes
ITER009_PHASE1 = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_009/exp/results/phase1")
# Path to iter_009 full patching results (ground truth to verify against)
ITER009_FULL_PATCHING = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_009/exp/results/full/activation_patching_crossdomain_full.json")

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = "PILOT"

# SAE config -- match iter_009
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_24/width_16k/canonical"
SAE_KEY = "L24_16k"
SAE_LAYER = 24
HOOK_POINT = f"blocks.{SAE_LAYER}.hook_resid_post"

# Token position -- must match probe training (iter_009 used -2)
TOKEN_POS = -2

# Spot-check config
N_ENTITIES = 20            # 20 entities for spot-check
N_CONTEXTS = 50            # contexts per entity (reduced from 100 for pilot speed)
N_DISCOVERY_CONTEXTS = 30  # contexts for absorber discovery per class
N_CONTROL_FEATURES = 10    # control features per entity
N_BOOTSTRAP = 5000         # bootstrap resamples
MIN_ABSORPTION_CONTEXTS = 2  # min absorbed contexts (relaxed for pilot)
MAX_ENTITIES_PER_CLASS = 30  # for absorber discovery phase
N_ICL = 5                  # ICL examples per prompt

# Expected values from iter_009 FULL results
EXPECTED_RECOVERY_RATE = 0.619  # 61.9% mean primary recovery
EXPECTED_COHENS_D = 1.50

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# Process tracking
# ============================================================
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))


def report_progress(step, total_steps, status="running", metrics=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
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
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))
    logger.info(f"DONE (status={status}): {summary}")


def update_gpu_progress(elapsed_seconds, status="completed"):
    try:
        import filelock
    except ImportError:
        logger.warning("filelock not available, skipping gpu_progress update")
        return
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
                    "layer": SAE_LAYER,
                    "token_pos": TOKEN_POS,
                    "n_entities": N_ENTITIES,
                    "n_contexts": N_CONTEXTS,
                    "hierarchy": "city-continent",
                    "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
                    "gpu_count": 1,
                },
            }
            progress_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"gpu_progress update failed: {e}")


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
# Load pre-trained probe from iter_009
# ============================================================
def load_probe(hierarchy_name, layer=24):
    """Load probe from iter_009 phase1 results."""
    probe_path = ITER009_PHASE1 / f"probe_{hierarchy_name}_L{layer}.npz"
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
# RAVEL data loading
# ============================================================
def prepare_ravel_data(hierarchy="city-continent"):
    from datasets import load_dataset

    logger.info(f"Loading RAVEL dataset for {hierarchy}...")
    ds = load_dataset("hij/ravel", "city_entity", split="train")

    cities = list(ds["City"])
    labels_raw = list(ds["Continent"])
    # Normalize: Australia -> Oceania (matching iter_009)
    labels = ["Oceania" if l == "Australia" else l for l in labels_raw]

    valid = [
        (c, str(l).strip()) for c, l in zip(cities, labels)
        if c and l and str(c).strip() and str(l).strip()
    ]
    label_counts = Counter([x[1] for x in valid])
    valid_labels = {l for l, n in label_counts.items() if n >= 3}
    keep = [(c, l) for c, l in valid if l in valid_labels]

    # Deduplicate by city name (keep first occurrence)
    seen = set()
    unique = []
    for c, l in keep:
        if c not in seen:
            seen.add(c)
            unique.append((c, l))

    logger.info(f"  {hierarchy}: {len(unique)} entities, "
                f"{len(set(l for _, l in unique))} classes")

    return {
        "cities": [x[0] for x in unique],
        "labels": [x[1] for x in unique],
    }


def stratified_sample(cities, labels, n_total, seed=42):
    """Stratified sampling: proportional to class size, minimum 1 per class."""
    rng = random.Random(seed)

    # Group by label
    by_label = defaultdict(list)
    for c, l in zip(cities, labels):
        by_label[l].append(c)

    # Proportional allocation with minimum 1
    n_classes = len(by_label)
    total_entities = len(cities)

    # Ensure at least 1 per class, distribute rest proportionally
    allocation = {}
    remaining = n_total
    for label in sorted(by_label.keys()):
        allocation[label] = 1
        remaining -= 1

    if remaining > 0:
        # Distribute remaining proportionally
        for label in sorted(by_label.keys()):
            extra = int(remaining * len(by_label[label]) / total_entities)
            allocation[label] += extra

        # Adjust to hit exactly n_total
        allocated = sum(allocation.values())
        diff = n_total - allocated
        # Add remainder to largest classes
        sorted_labels = sorted(by_label.keys(), key=lambda l: len(by_label[l]), reverse=True)
        for i in range(abs(diff)):
            if diff > 0:
                allocation[sorted_labels[i % n_classes]] += 1
            elif diff < 0:
                if allocation[sorted_labels[i % n_classes]] > 1:
                    allocation[sorted_labels[i % n_classes]] -= 1

    # Sample from each class
    sampled = []
    for label in sorted(by_label.keys()):
        pool = by_label[label][:]
        rng.shuffle(pool)
        n_take = min(allocation.get(label, 1), len(pool))
        for city in pool[:n_take]:
            sampled.append((city, label))

    # If we still need more (due to small classes), sample from remaining
    rng.shuffle(sampled)
    return sampled[:n_total]


# ============================================================
# ICL prompt construction (matching iter_009)
# ============================================================
PROMPT_TEMPLATE = "The city of {entity} is on the continent of"


def build_icl_prompts(city, label, all_cities, all_labels,
                      n_contexts=50, n_icl=5):
    """Build varied ICL prompts for one city."""
    answer_template = " {label}"
    examples = [(c, l) for c, l in zip(all_cities, all_labels) if c != city]

    prompts = []
    for ctx_idx in range(n_contexts):
        rng_ctx = random.Random(SEED + hash(city) + ctx_idx * 7919)
        rng_ctx.shuffle(examples)
        icl_selected = examples[:n_icl]

        icl_parts = []
        for ex_city, ex_label in icl_selected:
            ex_text = PROMPT_TEMPLATE.format(entity=ex_city) + answer_template.format(label=ex_label)
            icl_parts.append(ex_text)

        full_prompt = "\n".join(icl_parts) + "\n" + PROMPT_TEMPLATE.format(entity=city)
        prompts.append(full_prompt)

    return prompts


# ============================================================
# Batch activation caching
# ============================================================
def cache_entity_activations(model, sae, city, label, all_cities, all_labels,
                             n_contexts, device="cuda:0"):
    """Cache raw and SAE activations for one entity across multiple contexts."""
    prompts = build_icl_prompts(city, label, all_cities, all_labels,
                                n_contexts=n_contexts, n_icl=N_ICL)

    raw_acts = []
    for prompt in prompts:
        try:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[HOOK_POINT],
                    stop_at_layer=SAE_LAYER + 1,
                )
            act = cache[HOOK_POINT][0, TOKEN_POS, :].float().cpu()
            raw_acts.append(act)
            del cache
        except Exception as e:
            logger.debug(f"  Context error for {city}: {e}")
            continue

    if len(raw_acts) == 0:
        return None

    raw_acts_tensor = torch.stack(raw_acts).to(device)  # [n_ctx, d_model]

    with torch.no_grad():
        sae_features = sae.encode(raw_acts_tensor)     # [n_ctx, d_sae]
        sae_recon = sae.decode(sae_features)            # [n_ctx, d_model]

    return {
        "raw_acts": raw_acts_tensor,
        "sae_features": sae_features,
        "sae_recon": sae_recon,
        "n_contexts": len(raw_acts),
    }


# ============================================================
# Absorber discovery (matching iter_009 corrected methodology)
# ============================================================
def discover_absorber_features(model, sae, probe, cls_list, label_to_idx,
                               all_cities, all_labels, device="cuda:0"):
    """
    Discover per-class absorber features using contribution analysis.
    This matches the iter_009 FULL methodology exactly.
    """
    logger.info(f"\n{'='*60}")
    logger.info("ABSORBER DISCOVERY for city-continent")
    logger.info(f"{'='*60}")

    # Pre-compute decoder cosine similarities for all probe directions
    W_dec = sae.W_dec.detach().float().cpu()  # [d_sae, d_model]
    W_dec_normalized = W_dec / (W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8))

    probe_coefs = torch.tensor(probe.coef_, dtype=torch.float32)  # [n_classes, d_model]
    probe_dirs = probe_coefs / (probe_coefs.norm(dim=1, keepdim=True).clamp(min=1e-8))
    cos_per_class = (W_dec_normalized @ probe_dirs.T)  # [d_sae, n_classes]
    cos_per_class = cos_per_class.numpy()

    # Group cities by label
    cities_by_label = defaultdict(list)
    for c, l in zip(all_cities, all_labels):
        if l in label_to_idx:
            cities_by_label[l].append(c)

    absorber_results = {}

    for cls_label in sorted(label_to_idx.keys()):
        cls_idx = label_to_idx[cls_label]
        cls_cities = cities_by_label.get(cls_label, [])

        if len(cls_cities) == 0:
            logger.info(f"  {cls_label}: no cities, skipping")
            continue

        rng = random.Random(SEED + hash(cls_label))
        rng.shuffle(cls_cities)
        scan_cities = cls_cities[:MAX_ENTITIES_PER_CLASS]

        logger.info(f"\n  {cls_label}: scanning {len(scan_cities)} entities...")

        # Collect absorbed contexts
        all_absorbed_features = []
        n_raw_correct = 0
        n_absorbed = 0
        n_scanned = 0

        for city in scan_cities:
            cached = cache_entity_activations(
                model, sae, city, cls_label, all_cities, all_labels,
                n_contexts=N_DISCOVERY_CONTEXTS, device=device,
            )
            if cached is None:
                continue

            n_scanned += 1

            raw_np = cached["raw_acts"].cpu().numpy()
            sae_np = cached["sae_recon"].cpu().numpy()
            raw_preds = probe.predict(raw_np)
            sae_preds = probe.predict(sae_np)

            raw_correct_mask = (raw_preds == cls_idx)
            sae_correct_mask = (sae_preds == cls_idx)
            absorbed_mask = raw_correct_mask & (~sae_correct_mask)

            n_raw_correct += int(raw_correct_mask.sum())
            n_absorbed += int(absorbed_mask.sum())

            if absorbed_mask.sum() > 0:
                absorbed_feats = cached["sae_features"][absorbed_mask].cpu().numpy()
                all_absorbed_features.append(absorbed_feats)

            del cached
            if n_scanned % 10 == 0:
                torch.cuda.empty_cache()

        absorption_rate = n_absorbed / max(n_raw_correct, 1)
        logger.info(f"  {cls_label}: scanned {n_scanned} cities, "
                    f"raw_correct={n_raw_correct}, absorbed={n_absorbed} "
                    f"(rate={absorption_rate:.4f})")

        if n_absorbed < MIN_ABSORPTION_CONTEXTS:
            absorber_results[cls_label] = {
                "status": "insufficient_absorption",
                "n_absorbed": n_absorbed,
                "absorption_rate": absorption_rate,
            }
            continue

        # Stack all absorbed feature vectors
        absorbed_features = np.concatenate(all_absorbed_features, axis=0)
        mean_absorbed_acts = absorbed_features.mean(axis=0)

        # Compute contribution: mean_activation * cosine_with_true_class_probe_direction
        # NEGATIVE contribution = feature pushes prediction AWAY from true class -> absorber
        cos_true_class = cos_per_class[:, cls_idx]
        contributions = mean_absorbed_acts * cos_true_class

        # Sort ascending: most negative contribution first
        sorted_indices = np.argsort(contributions)

        # Select absorber candidates
        absorber_candidates = []
        for feat_idx in sorted_indices:
            feat_idx = int(feat_idx)
            if mean_absorbed_acts[feat_idx] > 0.1:
                absorber_candidates.append({
                    "feature_idx": feat_idx,
                    "contribution": float(contributions[feat_idx]),
                    "mean_activation": float(mean_absorbed_acts[feat_idx]),
                    "cosine_true_class": float(cos_true_class[feat_idx]),
                })
            if len(absorber_candidates) >= 10:
                break

        primary_absorber = absorber_candidates[0]["feature_idx"] if absorber_candidates else None

        absorber_results[cls_label] = {
            "status": "found",
            "n_scanned": n_scanned,
            "n_raw_correct": n_raw_correct,
            "n_absorbed": n_absorbed,
            "absorption_rate": float(absorption_rate),
            "primary_absorber": primary_absorber,
            "top_absorbers": absorber_candidates[:5],
        }

        logger.info(f"  {cls_label}: primary absorber = feature {primary_absorber}")
        for ac in absorber_candidates[:3]:
            logger.info(f"    feature {ac['feature_idx']}: contribution={ac['contribution']:.4f}, "
                        f"mean_act={ac['mean_activation']:.3f}, cos={ac['cosine_true_class']:.4f}")

    return absorber_results


# ============================================================
# Patching for one entity
# ============================================================
def run_patching_for_entity(model, sae, probe, cls_list,
                            city, true_label, label_to_idx,
                            all_cities, all_labels,
                            absorber_info, device="cuda:0"):
    """
    Run activation patching for one entity using discovered absorber features.
    Returns per-entity result dict.
    """
    cls_idx = label_to_idx[true_label]

    cached = cache_entity_activations(
        model, sae, city, true_label, all_cities, all_labels,
        n_contexts=N_CONTEXTS, device=device,
    )
    if cached is None:
        return {"status": "cache_failed", "city": city, "true_label": true_label}

    raw_acts = cached["raw_acts"]
    sae_features = cached["sae_features"]
    sae_recon = cached["sae_recon"]
    n_ctx = cached["n_contexts"]

    # Probe predictions
    raw_np = raw_acts.cpu().numpy()
    sae_np = sae_recon.cpu().numpy()
    raw_preds = probe.predict(raw_np)
    raw_probs = probe.predict_proba(raw_np)
    sae_preds = probe.predict(sae_np)
    sae_probs = probe.predict_proba(sae_np)

    raw_correct = (raw_preds == cls_idx)
    sae_correct = (sae_preds == cls_idx)
    absorbed_mask = raw_correct & (~sae_correct)

    n_raw_correct = int(raw_correct.sum())
    n_sae_correct = int(sae_correct.sum())
    n_absorbed = int(absorbed_mask.sum())

    if n_absorbed < MIN_ABSORPTION_CONTEXTS:
        return {
            "status": "insufficient_absorption",
            "city": city,
            "true_label": true_label,
            "n_contexts": n_ctx,
            "n_raw_correct": n_raw_correct,
            "n_absorbed": n_absorbed,
        }

    if absorber_info is None or absorber_info.get("status") != "found":
        return {
            "status": "no_absorber_info",
            "city": city,
            "true_label": true_label,
            "n_absorbed": n_absorbed,
        }

    primary_absorber = absorber_info["primary_absorber"]
    all_absorbers = [a["feature_idx"] for a in absorber_info.get("top_absorbers", [])]

    # Raw and SAE probabilities for true class
    raw_true_prob = raw_probs[:, cls_idx]
    sae_true_prob = sae_probs[:, cls_idx]

    # ---- PRIMARY ABSORBER ZEROING ----
    if primary_absorber is not None:
        modified_features = sae_features.clone()
        modified_features[:, primary_absorber] = 0.0

        with torch.no_grad():
            modified_recon = sae.decode(modified_features)

        mod_preds = probe.predict(modified_recon.cpu().numpy())
        mod_probs = probe.predict_proba(modified_recon.cpu().numpy())
        mod_correct = (mod_preds == cls_idx)
        mod_true_prob = mod_probs[:, cls_idx]

        primary_recovery = absorbed_mask & mod_correct
        n_primary_recovered = int(primary_recovery.sum())
        primary_recovery_rate = n_primary_recovered / n_absorbed

        primary_prob_change = float(
            (mod_true_prob[absorbed_mask] - sae_true_prob[absorbed_mask]).mean()
        ) if n_absorbed > 0 else 0.0

        # Degradation check
        was_correct_sae = sae_correct & raw_correct
        degradation = was_correct_sae & (~mod_correct)
        n_degraded_primary = int(degradation.sum())

        del modified_features, modified_recon
    else:
        n_primary_recovered = 0
        primary_recovery_rate = 0.0
        primary_prob_change = 0.0
        n_degraded_primary = 0

    # ---- ALL ABSORBERS ZEROING (top-5) ----
    if len(all_absorbers) > 0:
        modified_features_all = sae_features.clone()
        for absorber_fid in all_absorbers:
            modified_features_all[:, absorber_fid] = 0.0

        with torch.no_grad():
            modified_recon_all = sae.decode(modified_features_all)

        all_preds = probe.predict(modified_recon_all.cpu().numpy())
        all_correct = (all_preds == cls_idx)

        all_recovery = absorbed_mask & all_correct
        n_all_recovered = int(all_recovery.sum())
        all_recovery_rate = n_all_recovered / n_absorbed

        del modified_features_all, modified_recon_all
    else:
        n_all_recovered = 0
        all_recovery_rate = 0.0

    # ---- CONTROL: zero random non-absorber features (magnitude-matched) ----
    absorber_set = set(all_absorbers)
    if primary_absorber is not None:
        absorber_set.add(primary_absorber)

    primary_mean_act = float(sae_features[:, primary_absorber].mean().item()) if primary_absorber is not None else 0.0
    all_mean_acts = sae_features.detach().mean(dim=0).cpu().numpy()

    active_mask = all_mean_acts > 0.01
    active_indices = np.where(active_mask)[0]
    non_absorber = [int(idx) for idx in active_indices if idx not in absorber_set]

    # Magnitude-matched control selection
    if primary_mean_act > 0 and len(non_absorber) > 0:
        mag_diffs = np.abs(all_mean_acts[non_absorber] - primary_mean_act)
        sorted_by_mag = np.argsort(mag_diffs)
        matched_pool = [non_absorber[i] for i in sorted_by_mag[:N_CONTROL_FEATURES * 3]]
    else:
        matched_pool = non_absorber

    rng_ctrl = np.random.RandomState((SEED + abs(hash(city))) % (2**32))
    rng_ctrl.shuffle(matched_pool)
    control_features = matched_pool[:N_CONTROL_FEATURES]

    control_recovery_rates = []
    for ctrl_feat in control_features:
        ctrl_modified = sae_features.clone()
        ctrl_modified[:, ctrl_feat] = 0.0

        with torch.no_grad():
            ctrl_recon = sae.decode(ctrl_modified)

        ctrl_preds = probe.predict(ctrl_recon.cpu().numpy())
        ctrl_correct = (ctrl_preds == cls_idx)

        ctrl_recovery = absorbed_mask & ctrl_correct
        ctrl_recovery_rate = int(ctrl_recovery.sum()) / n_absorbed

        control_recovery_rates.append(ctrl_recovery_rate)
        del ctrl_modified, ctrl_recon

    mean_control_recovery = float(np.mean(control_recovery_rates)) if control_recovery_rates else 0.0

    # Clean up
    del raw_acts, sae_features, sae_recon
    torch.cuda.empty_cache()

    return {
        "status": "completed",
        "city": city,
        "true_label": true_label,
        "n_contexts": n_ctx,
        "n_raw_correct": n_raw_correct,
        "n_sae_correct": n_sae_correct,
        "n_absorbed": n_absorbed,
        "absorption_rate": n_absorbed / max(n_raw_correct, 1),
        "primary_absorber": primary_absorber,
        "n_absorbers_zeroed_all": len(all_absorbers),
        "primary_recovery": {
            "n_recovered": n_primary_recovered,
            "rate": float(primary_recovery_rate),
            "prob_change": float(primary_prob_change),
            "n_degraded": n_degraded_primary,
        },
        "all_absorbers_recovery": {
            "n_recovered": n_all_recovered,
            "rate": float(all_recovery_rate),
        },
        "control_recovery": {
            "mean_rate": float(mean_control_recovery),
            "rates": [float(r) for r in control_recovery_rates],
            "n_controls": len(control_features),
        },
        "recovery_diff_primary": float(primary_recovery_rate - mean_control_recovery),
    }


# ============================================================
# Statistical tests
# ============================================================
def bootstrap_ci(values, n_bootstrap=5000, ci=0.95, seed=42):
    if len(values) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n": 0}
    values = np.array(values, dtype=float)
    rng = np.random.RandomState(seed)
    boot_means = [float(np.mean(rng.choice(values, size=len(values), replace=True)))
                  for _ in range(n_bootstrap)]
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


def compute_stats(child_rates, control_rates):
    """Compute statistical tests comparing child vs control recovery rates."""
    stats_dict = {}
    child_arr = np.array(child_rates)
    control_arr = np.array(control_rates)
    diffs = child_arr - control_arr

    if len(child_rates) < 3:
        stats_dict["note"] = f"Too few entities ({len(child_rates)}) for tests"
        return stats_dict

    # Wilcoxon signed-rank
    non_zero = diffs[diffs != 0]
    if len(non_zero) >= 3:
        try:
            stat, p = stats.wilcoxon(child_arr, control_arr, alternative='greater')
            stats_dict["wilcoxon"] = {
                "statistic": float(stat),
                "p_value": float(p),
                "n_pairs": len(child_rates),
                "n_nonzero": len(non_zero),
                "significant_005": bool(p < 0.05),
                "significant_001": bool(p < 0.01),
            }
        except Exception as e:
            stats_dict["wilcoxon"] = {"error": str(e)}
    else:
        stats_dict["wilcoxon"] = {"note": f"Too few non-zero diffs ({len(non_zero)})"}

    # Paired t-test
    try:
        t_stat, t_p = stats.ttest_rel(child_arr, control_arr)
        # One-sided: recovery > control
        t_p_onesided = t_p / 2 if t_stat > 0 else 1 - t_p / 2
        stats_dict["paired_ttest"] = {
            "t_statistic": float(t_stat),
            "p_value_twosided": float(t_p),
            "p_value_onesided": float(t_p_onesided),
            "significant_001": bool(t_p_onesided < 0.01),
        }
    except Exception as e:
        stats_dict["paired_ttest"] = {"error": str(e)}

    # Cohen's d (paired)
    if np.std(diffs) > 0:
        d = float(np.mean(diffs) / np.std(diffs))
    else:
        d = float('inf') if np.mean(diffs) > 0 else 0.0
    stats_dict["cohens_d"] = {
        "value": d,
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs)),
        "interpretation": (
            "large" if abs(d) >= 0.8 else
            "medium" if abs(d) >= 0.5 else
            "small" if abs(d) >= 0.2 else "negligible"
        ),
    }

    # Bootstrap CI on difference
    stats_dict["bootstrap_diff"] = bootstrap_ci(diffs, N_BOOTSTRAP, seed=SEED)

    return stats_dict


# ============================================================
# Load iter_009 reference data
# ============================================================
def load_reference_data():
    """Load iter_009 FULL patching results for comparison."""
    if not ITER009_FULL_PATCHING.exists():
        logger.warning(f"iter_009 reference not found: {ITER009_FULL_PATCHING}")
        return None

    data = json.loads(ITER009_FULL_PATCHING.read_text())
    city_continent = data.get("per_hierarchy", {}).get("city-continent", {})
    if city_continent.get("status") != "completed":
        logger.warning("iter_009 city-continent not completed")
        return None

    agg = city_continent.get("aggregate", {})
    stats_primary = city_continent.get("statistical_tests_primary", {})

    return {
        "mean_primary_recovery": agg.get("mean_primary_rate", 0),
        "mean_control_recovery": agg.get("mean_control_rate", 0),
        "overall_primary_recovery": agg.get("overall_primary_recovery_rate", 0),
        "n_entities": city_continent.get("n_entities_completed", 0),
        "total_absorbed": city_continent.get("total_absorbed", 0),
        "cohens_d": stats_primary.get("cohens_d", {}).get("value", 0),
        "wilcoxon_p": stats_primary.get("wilcoxon", {}).get("p_value", None),
    }


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 25, "starting")

    device = "cuda:0"

    logger.info("=" * 70)
    logger.info("Phase 1.1 PILOT: Cross-Domain Patching Spot-Check")
    logger.info(f"N_ENTITIES: {N_ENTITIES}, N_CONTEXTS: {N_CONTEXTS}")
    logger.info(f"SAE: {SAE_KEY}, Layer: {SAE_LAYER}, Token pos: {TOKEN_POS}")
    logger.info(f"Expected recovery: {EXPECTED_RECOVERY_RATE:.1%}")
    logger.info("=" * 70)

    # Load reference data from iter_009
    reference = load_reference_data()
    if reference:
        logger.info(f"\niter_009 reference:")
        logger.info(f"  mean_primary_recovery={reference['mean_primary_recovery']:.4f}")
        logger.info(f"  mean_control_recovery={reference['mean_control_recovery']:.4f}")
        logger.info(f"  cohens_d={reference['cohens_d']:.4f}")
        logger.info(f"  n_entities={reference['n_entities']}")

    # Step 1: Load model
    report_progress(1, 25, "loading_model")
    model = load_model(device=device)

    # Step 2: Load SAE
    report_progress(2, 25, "loading_sae")
    sae = load_sae(SAE_RELEASE, SAE_ID, device=device)

    # Step 3: Load probe
    report_progress(3, 25, "loading_probe")
    probe, class_labels = load_probe("city-continent", layer=SAE_LAYER)
    cls_list = class_labels.tolist() if hasattr(class_labels, 'tolist') else list(class_labels)
    label_to_idx = {l: i for i, l in enumerate(cls_list)}

    # Step 4: Load RAVEL data
    report_progress(4, 25, "loading_data")
    ravel_data = prepare_ravel_data("city-continent")
    all_cities = ravel_data["cities"]
    all_labels = ravel_data["labels"]

    # Step 5: Stratified sample of 20 entities
    report_progress(5, 25, "sampling_entities")
    sampled_entities = stratified_sample(all_cities, all_labels, N_ENTITIES, seed=SEED)

    logger.info(f"\nSampled {len(sampled_entities)} entities:")
    label_counts = Counter([l for _, l in sampled_entities])
    for label, count in sorted(label_counts.items()):
        cities_in_label = [c for c, l in sampled_entities if l == label]
        logger.info(f"  {label} ({count}): {cities_in_label}")

    # Step 6: Discover absorber features per class
    # (Run on a broader set, not just the 20 sampled entities)
    report_progress(6, 25, "discovering_absorbers")
    absorber_results = discover_absorber_features(
        model, sae, probe, cls_list, label_to_idx,
        all_cities, all_labels, device=device,
    )

    # Step 7-26: Run patching for each sampled entity
    entity_results = []
    primary_recovery_rates = []
    all_recovery_rates = []
    control_recovery_rates_per_entity = []

    for i, (city, label) in enumerate(sampled_entities):
        report_progress(7 + i, 25, f"patching_{city}",
                        {"entity": city, "label": label, "idx": i + 1, "total": N_ENTITIES})
        logger.info(f"\n  [{i+1}/{N_ENTITIES}] Patching {city} ({label})...")

        absorber_info = absorber_results.get(label)
        result = run_patching_for_entity(
            model, sae, probe, cls_list,
            city, label, label_to_idx,
            all_cities, all_labels,
            absorber_info, device=device,
        )
        entity_results.append(result)

        if result["status"] == "completed" and result["n_absorbed"] >= MIN_ABSORPTION_CONTEXTS:
            primary_recovery_rates.append(result["primary_recovery"]["rate"])
            all_recovery_rates.append(result["all_absorbers_recovery"]["rate"])
            control_recovery_rates_per_entity.append(result["control_recovery"]["mean_rate"])

            logger.info(f"    absorbed={result['n_absorbed']}, "
                        f"primary_recovery={result['primary_recovery']['rate']:.4f}, "
                        f"all_recovery={result['all_absorbers_recovery']['rate']:.4f}, "
                        f"control={result['control_recovery']['mean_rate']:.4f}")
        else:
            logger.info(f"    status={result['status']}, n_absorbed={result.get('n_absorbed', 0)}")

        if (i + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ============================================================
    # Aggregate Results
    # ============================================================
    completed = [r for r in entity_results if r["status"] == "completed"
                 and r["n_absorbed"] >= MIN_ABSORPTION_CONTEXTS]
    n_completed = len(completed)

    logger.info(f"\n{'='*70}")
    logger.info(f"AGGREGATE RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Completed entities: {n_completed}/{N_ENTITIES}")

    if n_completed == 0:
        logger.error("No entities with sufficient absorption! Check data and probes.")
        output = {
            "task_id": TASK_ID,
            "mode": MODE,
            "status": "FAILED_NO_ABSORPTION",
            "n_sampled": N_ENTITIES,
            "n_completed": 0,
            "timestamp": datetime.now().isoformat(),
            "entity_results": entity_results,
        }
        out_path = PHASE1_DIR / "patching_spotcheck.json"
        out_path.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))
        mark_done("failed", "No entities with sufficient absorption")
        update_gpu_progress(time.time() - start_time, "failed")
        return output

    total_absorbed = sum(r["n_absorbed"] for r in completed)
    total_primary_recovered = sum(r["primary_recovery"]["n_recovered"] for r in completed)
    total_all_recovered = sum(r["all_absorbers_recovery"]["n_recovered"] for r in completed)

    overall_primary_rate = total_primary_recovered / max(total_absorbed, 1)
    mean_primary_rate = float(np.mean(primary_recovery_rates))
    mean_all_rate = float(np.mean(all_recovery_rates))
    mean_control_rate = float(np.mean(control_recovery_rates_per_entity))

    logger.info(f"Total absorbed contexts: {total_absorbed}")
    logger.info(f"Mean primary recovery: {mean_primary_rate:.4f} (expected: {EXPECTED_RECOVERY_RATE:.4f})")
    logger.info(f"Overall primary recovery: {overall_primary_rate:.4f}")
    logger.info(f"Mean all-absorbers recovery: {mean_all_rate:.4f}")
    logger.info(f"Mean control recovery: {mean_control_rate:.4f}")
    logger.info(f"Diff (primary-control): {mean_primary_rate - mean_control_rate:+.4f}")

    # Statistical tests
    stat_results = compute_stats(primary_recovery_rates, control_recovery_rates_per_entity)

    if "wilcoxon" in stat_results and "p_value" in stat_results["wilcoxon"]:
        logger.info(f"Wilcoxon p={stat_results['wilcoxon']['p_value']:.6f}")
    if "paired_ttest" in stat_results and "p_value_onesided" in stat_results["paired_ttest"]:
        logger.info(f"Paired t-test p (one-sided)={stat_results['paired_ttest']['p_value_onesided']:.6f}")
    if "cohens_d" in stat_results:
        logger.info(f"Cohen's d={stat_results['cohens_d']['value']:.4f} "
                    f"({stat_results['cohens_d']['interpretation']})")

    # ============================================================
    # Comparison with iter_009 reference
    # ============================================================
    recovery_diff_from_expected = mean_primary_rate - EXPECTED_RECOVERY_RATE
    abs_diff = abs(recovery_diff_from_expected)

    if abs_diff <= 0.10:
        verification_verdict = "VERIFIED"
        verdict_detail = (f"Mean primary recovery {mean_primary_rate:.4f} is within 10 pp "
                          f"of expected {EXPECTED_RECOVERY_RATE:.4f} (diff={recovery_diff_from_expected:+.4f})")
    elif abs_diff <= 0.15:
        verification_verdict = "MARGINAL"
        verdict_detail = (f"Mean primary recovery {mean_primary_rate:.4f} is 10-15 pp from "
                          f"expected {EXPECTED_RECOVERY_RATE:.4f} (diff={recovery_diff_from_expected:+.4f}). "
                          f"Investigate but may be sampling noise on 20 entities.")
    else:
        verification_verdict = "FLAG_CRITICAL"
        verdict_detail = (f"Mean primary recovery {mean_primary_rate:.4f} is >15 pp from "
                          f"expected {EXPECTED_RECOVERY_RATE:.4f} (diff={recovery_diff_from_expected:+.4f}). "
                          f"ENTIRE PATCHING DATASET IS SUSPECT.")

    logger.info(f"\nVERIFICATION VERDICT: {verification_verdict}")
    logger.info(f"  {verdict_detail}")

    # Check pass criteria
    cohens_d_val = stat_results.get("cohens_d", {}).get("value", 0)
    wilcoxon_p = stat_results.get("wilcoxon", {}).get("p_value", 1.0)
    ttest_p = stat_results.get("paired_ttest", {}).get("p_value_onesided", 1.0)

    pass_criteria = {
        "recovery_within_10pp": abs_diff <= 0.10,
        "cohens_d_gt_0.5": cohens_d_val > 0.5,
        "p_lt_0.01_wilcoxon": wilcoxon_p < 0.01 if isinstance(wilcoxon_p, float) else False,
        "p_lt_0.01_ttest": ttest_p < 0.01 if isinstance(ttest_p, float) else False,
    }
    all_pass = all(pass_criteria.values())

    logger.info(f"\nPass criteria:")
    for criterion, passed in pass_criteria.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {criterion}: {status}")
    logger.info(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    # Per-class breakdown
    per_class = defaultdict(lambda: {
        "n_entities": 0, "total_absorbed": 0,
        "total_primary_recovered": 0,
        "primary_rates": [], "control_rates": [],
    })
    for r in completed:
        cls = r["true_label"]
        per_class[cls]["n_entities"] += 1
        per_class[cls]["total_absorbed"] += r["n_absorbed"]
        per_class[cls]["total_primary_recovered"] += r["primary_recovery"]["n_recovered"]
        per_class[cls]["primary_rates"].append(r["primary_recovery"]["rate"])
        per_class[cls]["control_rates"].append(r["control_recovery"]["mean_rate"])

    per_class_summary = {}
    for cls, data in per_class.items():
        per_class_summary[cls] = {
            "n_entities": data["n_entities"],
            "total_absorbed": data["total_absorbed"],
            "primary_recovery_rate": data["total_primary_recovered"] / max(data["total_absorbed"], 1),
            "mean_primary_rate": float(np.mean(data["primary_rates"])),
            "mean_control_rate": float(np.mean(data["control_rates"])),
        }

    # ============================================================
    # Compile output
    # ============================================================
    elapsed = time.time() - start_time

    output = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae": {"release": SAE_RELEASE, "sae_id": SAE_ID, "key": SAE_KEY},
        "layer": SAE_LAYER,
        "token_pos": TOKEN_POS,
        "hierarchy": "city-continent",
        "n_entities_sampled": N_ENTITIES,
        "n_entities_completed": n_completed,
        "n_contexts_per_entity": N_CONTEXTS,
        "n_discovery_contexts": N_DISCOVERY_CONTEXTS,
        "n_control_features": N_CONTROL_FEATURES,
        "sampling": {
            "method": "stratified_by_continent",
            "seed": SEED,
            "entities_per_class": dict(label_counts),
            "sampled_entities": [{"city": c, "continent": l} for c, l in sampled_entities],
        },
        "aggregate": {
            "total_absorbed": total_absorbed,
            "overall_primary_recovery_rate": float(overall_primary_rate),
            "mean_primary_recovery_rate": float(mean_primary_rate),
            "mean_all_absorbers_recovery_rate": float(mean_all_rate),
            "mean_control_recovery_rate": float(mean_control_rate),
            "primary_bootstrap_ci": bootstrap_ci(primary_recovery_rates, N_BOOTSTRAP, seed=SEED),
            "control_bootstrap_ci": bootstrap_ci(control_recovery_rates_per_entity, N_BOOTSTRAP, seed=SEED),
        },
        "statistical_tests": stat_results,
        "verification": {
            "verdict": verification_verdict,
            "detail": verdict_detail,
            "expected_recovery_rate": EXPECTED_RECOVERY_RATE,
            "observed_recovery_rate": float(mean_primary_rate),
            "difference_pp": float(recovery_diff_from_expected * 100),
            "pass_criteria": pass_criteria,
            "all_criteria_pass": all_pass,
        },
        "iter009_reference": reference,
        "per_class": per_class_summary,
        "entity_results": entity_results,
        "absorber_discovery": {
            cls: {k: v for k, v in info.items()}
            for cls, info in absorber_results.items()
        },
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    # Save
    out_path = PHASE1_DIR / "patching_spotcheck.json"
    out_path.write_text(json.dumps(output, indent=2, cls=NumpyEncoder, default=str))
    logger.info(f"\nSaved: {out_path}")

    # Also save pilot summary
    pilot_summary = {
        "overall_recommendation": "GO" if all_pass else ("REFINE" if verification_verdict != "FLAG_CRITICAL" else "NO_GO"),
        "verification_verdict": verification_verdict,
        "mean_recovery_rate": float(mean_primary_rate),
        "expected_recovery_rate": EXPECTED_RECOVERY_RATE,
        "difference_pp": float(recovery_diff_from_expected * 100),
        "cohens_d": cohens_d_val,
        "wilcoxon_p": wilcoxon_p if isinstance(wilcoxon_p, float) else None,
        "n_entities_completed": n_completed,
        "pass_criteria": pass_criteria,
        "all_criteria_pass": all_pass,
    }
    pilot_path = PILOT_DIR / "pilot_patching_spotcheck.json"
    pilot_path.write_text(json.dumps(pilot_summary, indent=2, cls=NumpyEncoder, default=str))

    # Summary markdown
    summary_md = generate_summary_md(output)
    md_path = PHASE1_DIR / "patching_spotcheck_summary.md"
    md_path.write_text(summary_md)

    # Cleanup
    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    # Summary text
    summary_text = (
        f"Patching spot-check: {verification_verdict}. "
        f"Mean recovery={mean_primary_rate:.4f} (expected {EXPECTED_RECOVERY_RATE:.4f}, "
        f"diff={recovery_diff_from_expected:+.4f}). "
        f"d={cohens_d_val:.4f}, "
        f"n={n_completed}/{N_ENTITIES}. "
        f"Time: {elapsed/60:.1f}min."
    )

    mark_done("success" if all_pass else "partial", summary_text)
    update_gpu_progress(elapsed, "completed")

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETED: {summary_text}")
    logger.info(f"{'='*70}")

    return output


def generate_summary_md(results):
    """Generate human-readable markdown summary."""
    verif = results.get("verification", {})
    agg = results.get("aggregate", {})
    stats_r = results.get("statistical_tests", {})

    lines = [
        "# Phase 1.1: Cross-Domain Patching Spot-Check",
        "",
        f"**Verification Verdict**: {verif.get('verdict', 'UNKNOWN')}",
        f"**Time**: {results.get('elapsed_minutes', 0):.1f} minutes",
        "",
        "## Purpose",
        "",
        "Verify the sign-reversed city-continent activation patching data from iter_009.",
        f"Expected mean primary recovery rate: {EXPECTED_RECOVERY_RATE:.1%}",
        f"Expected Cohen's d: {EXPECTED_COHENS_D}",
        "",
        "## Sampling",
        "",
        f"- **Method**: Stratified by continent (seed={SEED})",
        f"- **N entities**: {results.get('n_entities_sampled', 0)}",
        f"- **N completed**: {results.get('n_entities_completed', 0)}",
        f"- **N contexts per entity**: {results.get('n_contexts_per_entity', 0)}",
        "",
        "### Entities sampled per continent",
        "",
    ]

    for entry in results.get("sampling", {}).get("sampled_entities", []):
        lines.append(f"- {entry['city']} ({entry['continent']})")

    lines.extend([
        "",
        "## Aggregate Results",
        "",
        f"- **Mean primary recovery**: {agg.get('mean_primary_recovery_rate', 0):.4f}",
        f"- **Overall primary recovery**: {agg.get('overall_primary_recovery_rate', 0):.4f}",
        f"- **Mean all-absorbers recovery**: {agg.get('mean_all_absorbers_recovery_rate', 0):.4f}",
        f"- **Mean control recovery**: {agg.get('mean_control_recovery_rate', 0):.4f}",
        f"- **Total absorbed contexts**: {agg.get('total_absorbed', 0)}",
        "",
        "## Statistical Tests",
        "",
    ])

    if "cohens_d" in stats_r:
        lines.append(f"- **Cohen's d**: {stats_r['cohens_d']['value']:.4f} "
                      f"({stats_r['cohens_d']['interpretation']})")
    if "wilcoxon" in stats_r and "p_value" in stats_r["wilcoxon"]:
        lines.append(f"- **Wilcoxon**: p={stats_r['wilcoxon']['p_value']:.6f} "
                      f"(sig at 0.01: {stats_r['wilcoxon'].get('significant_001', False)})")
    if "paired_ttest" in stats_r and "p_value_onesided" in stats_r["paired_ttest"]:
        lines.append(f"- **Paired t-test**: p={stats_r['paired_ttest']['p_value_onesided']:.6f} "
                      f"(sig at 0.01: {stats_r['paired_ttest'].get('significant_001', False)})")

    lines.extend([
        "",
        "## Verification Against iter_009",
        "",
        f"- **Expected recovery rate**: {verif.get('expected_recovery_rate', 0):.1%}",
        f"- **Observed recovery rate**: {verif.get('observed_recovery_rate', 0):.1%}",
        f"- **Difference**: {verif.get('difference_pp', 0):+.1f} pp",
        f"- **Verdict**: {verif.get('verdict', 'UNKNOWN')}",
        "",
        "## Pass Criteria",
        "",
    ])

    for criterion, passed in verif.get("pass_criteria", {}).items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"- {criterion}: **{status}**")

    lines.extend([
        "",
        f"**Overall**: {'ALL PASS' if verif.get('all_criteria_pass') else 'SOME FAIL'}",
        "",
        "## Per-Class Breakdown",
        "",
        "| Continent | Entities | Absorbed | Primary Rate | Control Rate |",
        "|-----------|----------|----------|--------------|-------------|",
    ])

    for cls, data in sorted(results.get("per_class", {}).items()):
        lines.append(
            f"| {cls} | {data['n_entities']} | {data['total_absorbed']} | "
            f"{data['mean_primary_rate']:.4f} | {data['mean_control_rate']:.4f} |"
        )

    lines.extend([
        "",
        "## Per-Entity Results",
        "",
        "| City | Continent | Absorbed | Primary Rec | Control | Diff |",
        "|------|-----------|----------|-------------|---------|------|",
    ])

    for r in results.get("entity_results", []):
        if r.get("status") == "completed":
            lines.append(
                f"| {r['city']} | {r['true_label']} | {r['n_absorbed']} | "
                f"{r['primary_recovery']['rate']:.4f} | {r['control_recovery']['mean_rate']:.4f} | "
                f"{r['recovery_diff_primary']:+.4f} |"
            )
        else:
            lines.append(
                f"| {r.get('city', '?')} | {r.get('true_label', '?')} | "
                f"{r.get('n_absorbed', 0)} | {r.get('status', '?')} | - | - |"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        try:
            update_gpu_progress(0, "failed")
        except Exception:
            pass
        sys.exit(1)
