#!/usr/bin/env python3
"""
Phase 2.1 FULL: Cross-Domain Activation Patching
=================================================

Extends iter_008's first-letter causal activation patching (p=0.000218, d=1.33)
to RAVEL hierarchies: city-continent and city-language at L24.

CRITICAL FIX from PILOT:
  Pilot recovered 0.05% (2/3751) because it zeroed features with highest
  POSITIVE cosine to the probe direction. This is backwards -- those features
  SUPPORT the correct class. We need to find features that COMPETE with the
  parent feature (absorbers).

Corrected methodology (matching iter_008):
  1. DISCOVERY PHASE: For each hierarchy, collect SAE activations on all
     absorbed contexts (probe correct on raw, wrong on SAE reconstruction).
     Compute per-feature contribution = mean_activation * cosine_with_probe.
     Features with most NEGATIVE contribution are the absorber candidates.
  2. PATCHING PHASE: For each entity, zero the discovered absorber feature(s)
     across all contexts, check if probe recovers on absorbed contexts.
  3. CONTROL: Zero magnitude-matched non-absorber features.
  4. Statistics: Wilcoxon signed-rank, bootstrap CI, Cohen's d.

Hierarchies:
  - city-continent (PRIMARY): L24-16k, 6 classes, ~31% absorption
  - city-language (SECONDARY): L24-16k, 23 classes, ~12% absorption
  - Also re-run first-letter at L24 for cross-hierarchy comparison

MODE: FULL (100 contexts per entity, 10k bootstrap, all available entities)
GPU: cuda:4 (mapped via CUDA_VISIBLE_DEVICES or directly)
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

MODE = "FULL"

# SAE config
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_24/width_16k/canonical"
SAE_KEY = "L24_16k"
SAE_LAYER = 24
HOOK_POINT = f"blocks.{SAE_LAYER}.hook_resid_post"

# Token position -- must match probe training
TOKEN_POS = -2

# Patching config
N_CONTEXTS = 100          # contexts per entity for patching
N_DISCOVERY_CONTEXTS = 50  # contexts per entity for absorber discovery
N_CONTROL_FEATURES = 15   # control features per word
N_BOOTSTRAP = 10000       # bootstrap resamples
MIN_ABSORPTION_CONTEXTS = 3  # min absorbed contexts to include entity
MAX_ENTITIES_PER_CLASS = 30  # cap entities per class for efficiency
N_ICL = 5                 # ICL examples per prompt

# Hierarchies to test
HIERARCHIES = ["city-continent", "city-language"]

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
                "planned_min": 60,
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
                    "hierarchies": HIERARCHIES,
                },
            }
            progress_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"gpu_progress update failed: {e}")


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
# RAVEL data loading
# ============================================================
def prepare_ravel_data(hierarchy="city-continent"):
    from datasets import load_dataset

    logger.info(f"Loading RAVEL dataset for {hierarchy}...")
    ds = load_dataset("hij/ravel", "city_entity", split="train")

    cities = list(ds["City"])

    if hierarchy == "city-continent":
        labels_raw = list(ds["Continent"])
        # Normalize: Australia -> Oceania
        labels = ["Oceania" if l == "Australia" else l for l in labels_raw]
    elif hierarchy == "city-language":
        labels = list(ds["Language"])
    elif hierarchy == "city-country":
        labels = list(ds["Country"])
    else:
        raise ValueError(f"Unknown hierarchy: {hierarchy}")

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


# ============================================================
# ICL prompt construction
# ============================================================
PROMPT_TEMPLATES = {
    "city-continent": "The city of {entity} is on the continent of",
    "city-language": "The primary language spoken in {entity} is",
}


def build_icl_prompts(city, label, all_cities, all_labels, hierarchy,
                      n_contexts=50, n_icl=5):
    """Build varied ICL prompts for one city."""
    template = PROMPT_TEMPLATES.get(hierarchy, "The city of {entity} is on the continent of")
    answer_template = " {label}"

    examples = [(c, l) for c, l in zip(all_cities, all_labels) if c != city]

    prompts = []
    for ctx_idx in range(n_contexts):
        rng_ctx = random.Random(SEED + hash(city) + ctx_idx * 7919)
        rng_ctx.shuffle(examples)
        icl_selected = examples[:n_icl]

        icl_parts = []
        for ex_city, ex_label in icl_selected:
            ex_text = template.format(entity=ex_city) + answer_template.format(label=ex_label)
            icl_parts.append(ex_text)

        full_prompt = "\n".join(icl_parts) + "\n" + template.format(entity=city)
        prompts.append(full_prompt)

    return prompts


# ============================================================
# Batch activation caching
# ============================================================
def cache_entity_activations(model, sae, city, label, all_cities, all_labels,
                             hierarchy, n_contexts, device="cuda:0"):
    """
    Cache raw and SAE activations for one entity across multiple contexts.
    Returns dict with raw_acts, sae_features, sae_recon tensors.
    """
    prompts = build_icl_prompts(
        city, label, all_cities, all_labels, hierarchy,
        n_contexts=n_contexts, n_icl=N_ICL
    )

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
# Absorber discovery (CORRECTED -- matching iter_008 methodology)
# ============================================================
def discover_absorber_features(model, sae, probe, cls_list, label_to_idx,
                               all_cities, all_labels, hierarchy,
                               device="cuda:0"):
    """
    Discover per-class absorber features using aggregate contribution analysis.

    For each class:
      1. Collect all absorbed contexts (probe correct raw, wrong SAE)
      2. Compute per-feature: contribution = mean_activation_on_absorbed * cos(decoder, probe_dir)
      3. Features with most NEGATIVE contribution push probe AWAY from true class
      4. These negative-contribution features are the absorber candidates

    Returns dict mapping class_label -> {primary_absorber, all_absorbers, stats}
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"ABSORBER DISCOVERY for {hierarchy}")
    logger.info(f"{'='*60}")

    # Pre-compute decoder cosine similarities for all probe directions
    # Move everything to CPU for this computation to avoid device mismatches
    W_dec = sae.W_dec.detach().float().cpu()  # [d_sae, d_model]
    W_dec_normalized = W_dec / (W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8))

    probe_coefs = torch.tensor(probe.coef_, dtype=torch.float32)  # [n_classes, d_model]

    # For each class, compute cosine between all decoder vectors and that class's probe direction
    # cos_per_class[c, f] = cosine(W_dec[f], probe_dir[c])
    probe_dirs = probe_coefs / (probe_coefs.norm(dim=1, keepdim=True).clamp(min=1e-8))
    cos_per_class = (W_dec_normalized @ probe_dirs.T)  # [d_sae, n_classes]
    cos_per_class = cos_per_class.numpy()

    # Build city -> label map
    city_label_map = dict(zip(all_cities, all_labels))

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

        # Cap entities for efficiency
        rng = random.Random(SEED + hash(cls_label))
        rng.shuffle(cls_cities)
        scan_cities = cls_cities[:MAX_ENTITIES_PER_CLASS]

        logger.info(f"\n  {cls_label}: scanning {len(scan_cities)} entities...")

        # Collect absorbed contexts
        all_absorbed_features = []  # SAE feature vectors on absorbed contexts
        n_raw_correct = 0
        n_absorbed = 0
        n_scanned = 0

        for city in scan_cities:
            cached = cache_entity_activations(
                model, sae, city, cls_label, all_cities, all_labels,
                hierarchy, n_contexts=N_DISCOVERY_CONTEXTS, device=device,
            )
            if cached is None:
                continue

            n_scanned += 1

            # Probe predictions
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

            # Clean up
            del cached
            if n_scanned % 10 == 0:
                torch.cuda.empty_cache()

        absorption_rate = n_absorbed / max(n_raw_correct, 1)
        logger.info(f"  {cls_label}: scanned {n_scanned} cities, "
                    f"raw_correct={n_raw_correct}, absorbed={n_absorbed} "
                    f"(rate={absorption_rate:.4f})")

        if n_absorbed < MIN_ABSORPTION_CONTEXTS:
            logger.info(f"  {cls_label}: too few absorbed contexts ({n_absorbed}), skipping")
            absorber_results[cls_label] = {
                "status": "insufficient_absorption",
                "n_absorbed": n_absorbed,
                "absorption_rate": absorption_rate,
            }
            continue

        # Stack all absorbed feature vectors
        absorbed_features = np.concatenate(all_absorbed_features, axis=0)  # [N, d_sae]
        mean_absorbed_acts = absorbed_features.mean(axis=0)  # [d_sae]

        # Compute contribution: mean_activation * cosine_with_true_class_probe_direction
        # NEGATIVE contribution = feature pushes prediction AWAY from true class
        cos_true_class = cos_per_class[:, cls_idx]  # [d_sae]
        contributions = mean_absorbed_acts * cos_true_class  # [d_sae]

        # Sort ascending: most negative contribution first (strongest absorbers)
        sorted_indices = np.argsort(contributions)

        # Select absorber candidates: features with negative contribution AND
        # that are actually active on absorbed contexts (mean_act > threshold)
        absorber_candidates = []
        for feat_idx in sorted_indices:
            feat_idx = int(feat_idx)
            if mean_absorbed_acts[feat_idx] > 0.1:  # Must be meaningfully active
                absorber_candidates.append({
                    "feature_idx": feat_idx,
                    "contribution": float(contributions[feat_idx]),
                    "mean_activation": float(mean_absorbed_acts[feat_idx]),
                    "cosine_true_class": float(cos_true_class[feat_idx]),
                })
            if len(absorber_candidates) >= 10:
                break

        # Primary absorber is the one with most negative contribution
        primary_absorber = absorber_candidates[0]["feature_idx"] if absorber_candidates else None

        # Also check: which features have highest cosine with WRONG class predictions?
        # This gives us features that might be "stealing" the representation
        sae_pred_classes = []
        for features_block in all_absorbed_features:
            # Decode and probe
            with torch.no_grad():
                decoded = sae.decode(torch.tensor(features_block, device=device))
            pred = probe.predict(decoded.cpu().numpy())
            sae_pred_classes.extend(pred.tolist())

        wrong_class_counts = Counter(sae_pred_classes)
        logger.info(f"  {cls_label}: SAE predictions on absorbed -> {dict(wrong_class_counts)}")

        absorber_results[cls_label] = {
            "status": "found",
            "n_scanned": n_scanned,
            "n_raw_correct": n_raw_correct,
            "n_absorbed": n_absorbed,
            "absorption_rate": float(absorption_rate),
            "primary_absorber": primary_absorber,
            "top_absorbers": absorber_candidates[:5],
            "wrong_class_distribution": {
                cls_list[k] if k < len(cls_list) else str(k): v
                for k, v in wrong_class_counts.items()
            },
        }

        logger.info(f"  {cls_label}: primary absorber = feature {primary_absorber}")
        for ac in absorber_candidates[:3]:
            logger.info(f"    feature {ac['feature_idx']}: contribution={ac['contribution']:.4f}, "
                        f"mean_act={ac['mean_activation']:.3f}, cos={ac['cosine_true_class']:.4f}")

    return absorber_results


# ============================================================
# Patching experiment for one entity (iter_008 style)
# ============================================================
def run_patching_for_entity(model, sae, probe, cls_list,
                            city, true_label, label_to_idx,
                            all_cities, all_labels, hierarchy,
                            absorber_info, device="cuda:0"):
    """
    Run activation patching for one entity using discovered absorber features.

    Methodology (matching iter_008):
      1. Cache raw + SAE activations across N_CONTEXTS
      2. Identify absorbed contexts (probe correct raw, wrong SAE)
      3. Zero primary absorber across ALL contexts, check recovery on absorbed
      4. Zero ALL absorber candidates, check recovery
      5. Control: zero random features, magnitude-matched
    """
    cls_idx = label_to_idx[true_label]

    # Cache activations
    cached = cache_entity_activations(
        model, sae, city, true_label, all_cities, all_labels,
        hierarchy, n_contexts=N_CONTEXTS, device=device,
    )
    if cached is None:
        return {"status": "cache_failed", "city": city, "true_label": true_label}

    raw_acts = cached["raw_acts"]        # [n_ctx, d_model]
    sae_features = cached["sae_features"]  # [n_ctx, d_sae]
    sae_recon = cached["sae_recon"]      # [n_ctx, d_model]
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

    # Get absorber features for this class
    if absorber_info is None or absorber_info.get("status") != "found":
        return {
            "status": "no_absorber_info",
            "city": city,
            "true_label": true_label,
            "n_absorbed": n_absorbed,
        }

    primary_absorber = absorber_info["primary_absorber"]
    all_absorbers = [a["feature_idx"] for a in absorber_info.get("top_absorbers", [])]

    # Raw probability for true class
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

        # Recovery on absorbed contexts
        primary_recovery = absorbed_mask & mod_correct
        n_primary_recovered = int(primary_recovery.sum())
        primary_recovery_rate = n_primary_recovered / n_absorbed

        # Probability change on absorbed contexts
        primary_prob_change = float(
            (mod_true_prob[absorbed_mask] - sae_true_prob[absorbed_mask]).mean()
        ) if n_absorbed > 0 else 0.0

        # Degradation on non-absorbed (were correct, now wrong?)
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
        all_probs = probe.predict_proba(modified_recon_all.cpu().numpy())
        all_correct = (all_preds == cls_idx)
        all_true_prob = all_probs[:, cls_idx]

        all_recovery = absorbed_mask & all_correct
        n_all_recovered = int(all_recovery.sum())
        all_recovery_rate = n_all_recovered / n_absorbed

        all_prob_change = float(
            (all_true_prob[absorbed_mask] - sae_true_prob[absorbed_mask]).mean()
        ) if n_absorbed > 0 else 0.0

        del modified_features_all, modified_recon_all
    else:
        n_all_recovered = 0
        all_recovery_rate = 0.0
        all_prob_change = 0.0

    # ---- CONTROL: zero random non-absorber features (magnitude-matched) ----
    absorber_set = set(all_absorbers)
    if primary_absorber is not None:
        absorber_set.add(primary_absorber)

    # Get mean activation magnitude of primary absorber
    primary_mean_act = float(sae_features[:, primary_absorber].mean().item()) if primary_absorber is not None else 0.0
    all_mean_acts = sae_features.detach().mean(dim=0).cpu().numpy()

    # Find active non-absorber features
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
    control_prob_changes = []

    for ctrl_feat in control_features:
        ctrl_modified = sae_features.clone()
        ctrl_modified[:, ctrl_feat] = 0.0

        with torch.no_grad():
            ctrl_recon = sae.decode(ctrl_modified)

        ctrl_preds = probe.predict(ctrl_recon.cpu().numpy())
        ctrl_probs = probe.predict_proba(ctrl_recon.cpu().numpy())
        ctrl_correct = (ctrl_preds == cls_idx)
        ctrl_true_prob = ctrl_probs[:, cls_idx]

        ctrl_recovery = absorbed_mask & ctrl_correct
        ctrl_recovery_rate = int(ctrl_recovery.sum()) / n_absorbed
        ctrl_prob_change = float(
            (ctrl_true_prob[absorbed_mask] - sae_true_prob[absorbed_mask]).mean()
        ) if n_absorbed > 0 else 0.0

        control_recovery_rates.append(ctrl_recovery_rate)
        control_prob_changes.append(ctrl_prob_change)

        del ctrl_modified, ctrl_recon

    mean_control_recovery = float(np.mean(control_recovery_rates)) if control_recovery_rates else 0.0
    mean_control_prob_change = float(np.mean(control_prob_changes)) if control_prob_changes else 0.0

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
            "prob_change": float(all_prob_change),
        },
        "control_recovery": {
            "mean_rate": float(mean_control_recovery),
            "rates": [float(r) for r in control_recovery_rates],
            "mean_prob_change": float(mean_control_prob_change),
            "n_controls": len(control_features),
        },
        "recovery_diff_primary": float(primary_recovery_rate - mean_control_recovery),
        "recovery_diff_all": float(all_recovery_rate - mean_control_recovery),
    }


# ============================================================
# Bootstrap CI
# ============================================================
def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
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


# ============================================================
# Statistical tests
# ============================================================
def compute_stats(child_rates, control_rates, label=""):
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

    # Permutation test
    n_perm = 5000
    observed = float(np.mean(diffs))
    rng_perm = np.random.RandomState(SEED)
    n_extreme = 0
    for _ in range(n_perm):
        signs = rng_perm.choice([-1, 1], size=len(diffs))
        perm_mean = float(np.mean(diffs * signs))
        if perm_mean >= observed:
            n_extreme += 1
    perm_p = (n_extreme + 1) / (n_perm + 1)
    stats_dict["permutation"] = {
        "observed_mean_diff": observed,
        "p_value": float(perm_p),
        "n_permutations": n_perm,
        "significant_005": bool(perm_p < 0.05),
    }

    return stats_dict


# ============================================================
# Run patching for one hierarchy
# ============================================================
def run_hierarchy_patching(model, sae, hierarchy, absorber_results,
                           device="cuda:0"):
    """Run the full patching pipeline for one hierarchy."""
    logger.info(f"\n{'='*70}")
    logger.info(f"PATCHING PIPELINE: {hierarchy}")
    logger.info(f"{'='*70}")

    # Load probe
    probe, class_labels = load_probe(hierarchy, layer=SAE_LAYER)
    cls_list = class_labels.tolist() if hasattr(class_labels, 'tolist') else list(class_labels)

    # Load RAVEL data
    ravel_data = prepare_ravel_data(hierarchy)
    all_cities = ravel_data["cities"]
    all_labels = ravel_data["labels"]

    label_to_idx = {l: cls_list.index(l) for l in set(all_labels) if l in cls_list}

    # Filter to classes with discovered absorbers
    classes_with_absorbers = [
        cls for cls, info in absorber_results.items()
        if info.get("status") == "found" and info.get("primary_absorber") is not None
    ]

    logger.info(f"  Classes with absorbers: {classes_with_absorbers}")

    # Select entities for patching: cities from classes with absorbers
    # that show absorption in Phase 1
    entities_to_patch = []
    city_label_map = dict(zip(all_cities, all_labels))

    for cls_label in classes_with_absorbers:
        cls_cities = [c for c, l in zip(all_cities, all_labels) if l == cls_label]
        rng = random.Random(SEED + hash(cls_label))
        rng.shuffle(cls_cities)
        selected = cls_cities[:MAX_ENTITIES_PER_CLASS]
        for city in selected:
            entities_to_patch.append((city, cls_label))

    logger.info(f"  Total entities to patch: {len(entities_to_patch)}")

    # Run patching
    entity_results = []
    primary_recovery_rates = []  # per-entity primary absorber recovery
    all_recovery_rates = []      # per-entity all-absorbers recovery
    control_recovery_rates_per_entity = []  # per-entity mean control recovery

    for i, (city, label) in enumerate(entities_to_patch):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"\n  [{i+1}/{len(entities_to_patch)}] Patching {city} ({label})...")
            report_progress(i, len(entities_to_patch), f"patching_{hierarchy}",
                            {"city": city, "label": label, "hierarchy": hierarchy})

        absorber_info = absorber_results.get(label)
        result = run_patching_for_entity(
            model, sae, probe, cls_list,
            city, label, label_to_idx,
            all_cities, all_labels, hierarchy,
            absorber_info, device=device,
        )
        entity_results.append(result)

        if result["status"] == "completed" and result["n_absorbed"] >= MIN_ABSORPTION_CONTEXTS:
            primary_recovery_rates.append(result["primary_recovery"]["rate"])
            all_recovery_rates.append(result["all_absorbers_recovery"]["rate"])
            control_recovery_rates_per_entity.append(result["control_recovery"]["mean_rate"])

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"    absorbed={result['n_absorbed']}, "
                            f"primary_recovery={result['primary_recovery']['rate']:.4f}, "
                            f"all_recovery={result['all_absorbers_recovery']['rate']:.4f}, "
                            f"control={result['control_recovery']['mean_rate']:.4f}")

        if (i + 1) % 15 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Aggregate statistics
    completed = [r for r in entity_results if r["status"] == "completed"
                 and r["n_absorbed"] >= MIN_ABSORPTION_CONTEXTS]
    n_completed = len(completed)

    logger.info(f"\n  {hierarchy}: {n_completed} entities with sufficient absorption")

    if n_completed == 0:
        return {
            "hierarchy": hierarchy,
            "status": "no_entities_with_absorption",
            "n_entities_tested": len(entities_to_patch),
        }

    # Aggregate
    total_absorbed = sum(r["n_absorbed"] for r in completed)
    total_primary_recovered = sum(r["primary_recovery"]["n_recovered"] for r in completed)
    total_all_recovered = sum(r["all_absorbers_recovery"]["n_recovered"] for r in completed)

    overall_primary_rate = total_primary_recovered / max(total_absorbed, 1)
    overall_all_rate = total_all_recovered / max(total_absorbed, 1)

    mean_primary_rate = float(np.mean(primary_recovery_rates))
    mean_all_rate = float(np.mean(all_recovery_rates))
    mean_control_rate = float(np.mean(control_recovery_rates_per_entity))

    logger.info(f"\n  AGGREGATE ({hierarchy}):")
    logger.info(f"    Entities: {n_completed}, Total absorbed: {total_absorbed}")
    logger.info(f"    Primary absorber recovery: {overall_primary_rate:.4f} "
                f"(mean per-entity: {mean_primary_rate:.4f})")
    logger.info(f"    All absorbers recovery: {overall_all_rate:.4f} "
                f"(mean per-entity: {mean_all_rate:.4f})")
    logger.info(f"    Control recovery: mean={mean_control_rate:.4f}")
    logger.info(f"    Diff (primary-control): {mean_primary_rate - mean_control_rate:+.4f}")
    logger.info(f"    Diff (all-control): {mean_all_rate - mean_control_rate:+.4f}")

    # Statistical tests
    primary_stats = compute_stats(primary_recovery_rates, control_recovery_rates_per_entity,
                                  label=f"{hierarchy}_primary")
    all_stats = compute_stats(all_recovery_rates, control_recovery_rates_per_entity,
                              label=f"{hierarchy}_all")

    if "wilcoxon" in primary_stats and "p_value" in primary_stats["wilcoxon"]:
        logger.info(f"    Wilcoxon (primary): p={primary_stats['wilcoxon']['p_value']:.6f}")
    if "cohens_d" in primary_stats:
        logger.info(f"    Cohen's d (primary): {primary_stats['cohens_d']['value']:.4f} "
                    f"({primary_stats['cohens_d']['interpretation']})")
    if "wilcoxon" in all_stats and "p_value" in all_stats["wilcoxon"]:
        logger.info(f"    Wilcoxon (all): p={all_stats['wilcoxon']['p_value']:.6f}")

    # Per-class breakdown
    per_class = defaultdict(lambda: {
        "n_entities": 0, "total_absorbed": 0,
        "total_primary_recovered": 0, "total_all_recovered": 0,
        "primary_rates": [], "control_rates": [],
    })
    for r in completed:
        cls = r["true_label"]
        per_class[cls]["n_entities"] += 1
        per_class[cls]["total_absorbed"] += r["n_absorbed"]
        per_class[cls]["total_primary_recovered"] += r["primary_recovery"]["n_recovered"]
        per_class[cls]["total_all_recovered"] += r["all_absorbers_recovery"]["n_recovered"]
        per_class[cls]["primary_rates"].append(r["primary_recovery"]["rate"])
        per_class[cls]["control_rates"].append(r["control_recovery"]["mean_rate"])

    per_class_summary = {}
    for cls, data in per_class.items():
        per_class_summary[cls] = {
            "n_entities": data["n_entities"],
            "total_absorbed": data["total_absorbed"],
            "primary_recovery_rate": data["total_primary_recovered"] / max(data["total_absorbed"], 1),
            "all_recovery_rate": data["total_all_recovered"] / max(data["total_absorbed"], 1),
            "mean_primary_rate": float(np.mean(data["primary_rates"])),
            "mean_control_rate": float(np.mean(data["control_rates"])),
        }

    return {
        "hierarchy": hierarchy,
        "status": "completed",
        "n_entities_tested": len(entities_to_patch),
        "n_entities_completed": n_completed,
        "total_absorbed": total_absorbed,
        "aggregate": {
            "overall_primary_recovery_rate": float(overall_primary_rate),
            "overall_all_recovery_rate": float(overall_all_rate),
            "mean_primary_rate": float(mean_primary_rate),
            "mean_all_rate": float(mean_all_rate),
            "mean_control_rate": float(mean_control_rate),
            "primary_bootstrap_ci": bootstrap_ci(primary_recovery_rates, N_BOOTSTRAP, seed=SEED),
            "all_bootstrap_ci": bootstrap_ci(all_recovery_rates, N_BOOTSTRAP, seed=SEED),
            "control_bootstrap_ci": bootstrap_ci(control_recovery_rates_per_entity, N_BOOTSTRAP, seed=SEED),
        },
        "statistical_tests_primary": primary_stats,
        "statistical_tests_all": all_stats,
        "per_class": per_class_summary,
        "entity_results": [
            {k: v for k, v in r.items()
             if k not in ("control_recovery",) or not isinstance(v, dict) or "rates" not in v}
            for r in entity_results
        ],
        "absorber_discovery": {
            cls: {k: v for k, v in info.items() if k != "wrong_class_distribution"}
            for cls, info in absorber_results.items()
        },
    }


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 20, "starting")

    device = "cuda:0"

    logger.info("=" * 70)
    logger.info("Phase 2.1 FULL: Cross-Domain Activation Patching")
    logger.info(f"Hierarchies: {HIERARCHIES}")
    logger.info(f"SAE: {SAE_KEY}, Layer: {SAE_LAYER}, Token pos: {TOKEN_POS}")
    logger.info(f"N_CONTEXTS: {N_CONTEXTS}, N_DISCOVERY_CONTEXTS: {N_DISCOVERY_CONTEXTS}")
    logger.info(f"N_CONTROL: {N_CONTROL_FEATURES}, N_BOOTSTRAP: {N_BOOTSTRAP}")
    logger.info("=" * 70)

    # Step 1: Load model
    report_progress(1, 20, "loading_model")
    model = load_model(device=device)

    # Step 2: Load SAE
    report_progress(2, 20, "loading_sae")
    sae = load_sae(SAE_RELEASE, SAE_ID, device=device)

    # Run for each hierarchy
    all_hierarchy_results = {}

    for h_idx, hierarchy in enumerate(HIERARCHIES):
        h_start = time.time()

        # Step 3+: Load probe and data
        report_progress(3 + h_idx * 8, 20, f"loading_{hierarchy}")
        probe, class_labels = load_probe(hierarchy, layer=SAE_LAYER)
        cls_list = class_labels.tolist() if hasattr(class_labels, 'tolist') else list(class_labels)
        ravel_data = prepare_ravel_data(hierarchy)
        all_cities = ravel_data["cities"]
        all_labels = ravel_data["labels"]
        label_to_idx = {l: cls_list.index(l) for l in set(all_labels) if l in cls_list}

        # Step 4+: Discover absorber features
        report_progress(4 + h_idx * 8, 20, f"discovering_absorbers_{hierarchy}")
        absorber_results = discover_absorber_features(
            model, sae, probe, cls_list, label_to_idx,
            all_cities, all_labels, hierarchy, device=device,
        )

        # Step 5+: Run patching
        report_progress(5 + h_idx * 8, 20, f"patching_{hierarchy}")
        hierarchy_result = run_hierarchy_patching(
            model, sae, hierarchy, absorber_results, device=device,
        )

        h_elapsed = time.time() - h_start
        hierarchy_result["elapsed_minutes"] = h_elapsed / 60
        all_hierarchy_results[hierarchy] = hierarchy_result

        logger.info(f"\n  {hierarchy} completed in {h_elapsed/60:.1f} minutes")
        gc.collect()
        torch.cuda.empty_cache()

    # Final aggregation
    report_progress(18, 20, "final_aggregation")
    elapsed = time.time() - start_time

    # Cross-hierarchy comparison
    cross_hierarchy = {}
    for h, result in all_hierarchy_results.items():
        if result.get("status") == "completed":
            agg = result.get("aggregate", {})
            cross_hierarchy[h] = {
                "n_entities": result["n_entities_completed"],
                "total_absorbed": result["total_absorbed"],
                "primary_recovery": agg.get("mean_primary_rate", 0),
                "all_recovery": agg.get("mean_all_rate", 0),
                "control_recovery": agg.get("mean_control_rate", 0),
                "diff_primary": agg.get("mean_primary_rate", 0) - agg.get("mean_control_rate", 0),
                "wilcoxon_p": result.get("statistical_tests_primary", {}).get("wilcoxon", {}).get("p_value"),
                "cohens_d": result.get("statistical_tests_primary", {}).get("cohens_d", {}).get("value"),
            }

    # Interpretation
    any_significant = any(
        r.get("statistical_tests_primary", {}).get("wilcoxon", {}).get("significant_005", False)
        for r in all_hierarchy_results.values()
        if r.get("status") == "completed"
    )
    any_positive_diff = any(
        r.get("aggregate", {}).get("mean_primary_rate", 0) > r.get("aggregate", {}).get("mean_control_rate", 0)
        for r in all_hierarchy_results.values()
        if r.get("status") == "completed"
    )

    if any_significant and any_positive_diff:
        verdict = "CROSS_DOMAIN_CAUSAL_ABSORPTION_CONFIRMED"
    elif any_positive_diff:
        verdict = "EVIDENCE_FOR_CROSS_DOMAIN_ABSORPTION"
    else:
        verdict = "INSUFFICIENT_EVIDENCE"

    # Compare with iter_008 first-letter
    iter008_comparison = {}
    iter008_path = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_008/exp/results/phase0/activation_patching_full.json")
    if iter008_path.exists():
        try:
            iter008_data = json.loads(iter008_path.read_text())
            iter008_comparison = {
                "firstletter_recovery": iter008_data.get("aggregate", {}).get("mean_recovery_rate_child_absorbed", 0),
                "firstletter_control": iter008_data.get("aggregate", {}).get("mean_recovery_rate_control_absorbed", 0),
                "firstletter_p": iter008_data.get("statistical_tests", {}).get("wilcoxon_signed_rank", {}).get("p_value"),
                "firstletter_d": iter008_data.get("statistical_tests", {}).get("effect_size", {}).get("cohens_d"),
                "note": "iter_008 first-letter patching at L12 (different layer). "
                        "Cross-domain at L24. Direct comparison is qualitative.",
            }
        except Exception as e:
            iter008_comparison = {"error": str(e)}

    # Compile final output
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
        "n_discovery_contexts": N_DISCOVERY_CONTEXTS,
        "n_control_features": N_CONTROL_FEATURES,
        "hierarchies_tested": HIERARCHIES,
        "methodology_fix": {
            "pilot_bug": "Pilot zeroed features with highest POSITIVE cosine to probe "
                         "(features supporting correct class). Recovery was 0.05%.",
            "fix": "Now uses iter_008 contribution-based approach: contribution = "
                   "mean_activation * cosine_with_probe. Features with most NEGATIVE "
                   "contribution are identified as absorbers (they push prediction away "
                   "from true class). Zeroing absorbers should RECOVER correct prediction.",
        },
        "per_hierarchy": all_hierarchy_results,
        "cross_hierarchy_comparison": cross_hierarchy,
        "iter008_firstletter_comparison": iter008_comparison,
        "interpretation": {
            "verdict": verdict,
            "any_significant": any_significant,
            "any_positive_recovery_diff": any_positive_diff,
        },
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    # Save
    out_path = PHASE2_DIR / "activation_patching_crossdomain.json"
    out_path.write_text(json.dumps(output, indent=2, cls=NumpyEncoder, default=str))
    logger.info(f"\nSaved: {out_path}")

    # Also save to full directory
    full_path = RESULTS_DIR / "full" / "activation_patching_crossdomain_full.json"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(json.dumps(output, indent=2, cls=NumpyEncoder, default=str))

    # Summary markdown
    summary_md = generate_summary_md(output)
    md_path = PHASE2_DIR / "activation_patching_crossdomain_summary.md"
    md_path.write_text(summary_md)
    logger.info(f"Summary: {md_path}")

    # Cleanup
    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    # Build summary text
    summary_parts = [
        f"Phase 2.1 FULL cross-domain activation patching ({MODE}). "
        f"Hierarchies: {', '.join(HIERARCHIES)}. "
    ]
    for h, result in all_hierarchy_results.items():
        if result.get("status") == "completed":
            agg = result.get("aggregate", {})
            summary_parts.append(
                f"{h}: entities={result['n_entities_completed']}, "
                f"absorbed={result['total_absorbed']}, "
                f"primary_recovery={agg.get('mean_primary_rate', 0):.4f}, "
                f"control={agg.get('mean_control_rate', 0):.4f}. "
            )
            st = result.get("statistical_tests_primary", {})
            if "wilcoxon" in st and "p_value" in st["wilcoxon"]:
                summary_parts.append(f"Wilcoxon p={st['wilcoxon']['p_value']:.4f}. ")
            if "cohens_d" in st:
                summary_parts.append(f"d={st['cohens_d']['value']:.4f}. ")

    summary_parts.append(f"Verdict: {verdict}. Time: {elapsed/60:.1f}min.")
    summary_text = "".join(summary_parts)

    # Determine pass status
    pilot_pass = any_significant or any_positive_diff
    mark_done("success" if pilot_pass else "partial", summary_text)
    update_gpu_progress(elapsed, "completed")
    update_experiment_state("completed")

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETED: {summary_text}")
    logger.info(f"{'='*70}")

    return output


def generate_summary_md(results):
    """Generate human-readable markdown summary."""
    lines = [
        "# Phase 2.1 FULL: Cross-Domain Activation Patching",
        "",
        f"**Verdict**: {results.get('interpretation', {}).get('verdict', 'UNKNOWN')}",
        f"**Time**: {results.get('elapsed_minutes', 0):.1f} minutes",
        "",
        "## Methodology Fix",
        "",
        f"**Pilot bug**: {results.get('methodology_fix', {}).get('pilot_bug', '')}",
        f"**Fix**: {results.get('methodology_fix', {}).get('fix', '')}",
        "",
    ]

    for h in results.get("hierarchies_tested", []):
        h_result = results.get("per_hierarchy", {}).get(h, {})
        if h_result.get("status") != "completed":
            lines.extend([f"## {h}", "", f"Status: {h_result.get('status', 'unknown')}", ""])
            continue

        agg = h_result.get("aggregate", {})
        lines.extend([
            f"## {h}",
            "",
            f"- **Entities**: {h_result.get('n_entities_completed', 0)}",
            f"- **Total absorbed contexts**: {h_result.get('total_absorbed', 0)}",
            f"- **Primary absorber recovery**: {agg.get('mean_primary_rate', 0):.4f} "
            f"(CI [{agg.get('primary_bootstrap_ci', {}).get('ci_lower', 0):.3f}, "
            f"{agg.get('primary_bootstrap_ci', {}).get('ci_upper', 0):.3f}])",
            f"- **All absorbers recovery**: {agg.get('mean_all_rate', 0):.4f}",
            f"- **Control recovery**: {agg.get('mean_control_rate', 0):.4f}",
            f"- **Diff (primary-control)**: {agg.get('mean_primary_rate', 0) - agg.get('mean_control_rate', 0):+.4f}",
            "",
        ])

        # Stats
        st = h_result.get("statistical_tests_primary", {})
        if "wilcoxon" in st and "p_value" in st["wilcoxon"]:
            lines.append(f"- **Wilcoxon**: p={st['wilcoxon']['p_value']:.6f} "
                         f"(sig: {st['wilcoxon'].get('significant_005', False)})")
        if "cohens_d" in st:
            lines.append(f"- **Cohen's d**: {st['cohens_d']['value']:.4f} "
                         f"({st['cohens_d']['interpretation']})")

        # Per-class table
        pcp = h_result.get("per_class", {})
        if pcp:
            lines.extend([
                "", "### Per-Class Results", "",
                "| Class | Entities | Absorbed | Primary Rec | All Rec | Control |",
                "|-------|----------|----------|-------------|---------|---------|",
            ])
            for cls in sorted(pcp.keys()):
                d = pcp[cls]
                lines.append(
                    f"| {cls} | {d['n_entities']} | {d['total_absorbed']} | "
                    f"{d['primary_recovery_rate']:.4f} | {d['all_recovery_rate']:.4f} | "
                    f"{d['mean_control_rate']:.4f} |"
                )
        lines.append("")

    # Cross-hierarchy comparison
    cross = results.get("cross_hierarchy_comparison", {})
    if cross:
        lines.extend([
            "## Cross-Hierarchy Comparison", "",
            "| Hierarchy | Entities | Primary Rec | Control | Diff | Wilcoxon p | d |",
            "|-----------|----------|-------------|---------|------|-----------|---|",
        ])
        for h, data in cross.items():
            lines.append(
                f"| {h} | {data.get('n_entities', 0)} | "
                f"{data.get('primary_recovery', 0):.4f} | "
                f"{data.get('control_recovery', 0):.4f} | "
                f"{data.get('diff_primary', 0):+.4f} | "
                f"{data.get('wilcoxon_p', 'N/A')} | "
                f"{data.get('cohens_d', 'N/A')} |"
            )

    # iter_008 comparison
    i8 = results.get("iter008_firstletter_comparison", {})
    if i8 and "firstletter_recovery" in i8:
        lines.extend([
            "", "## Comparison with iter_008 First-Letter",
            "",
            f"- **First-letter recovery (L12)**: {i8['firstletter_recovery']:.4f}",
            f"- **First-letter control**: {i8['firstletter_control']:.4f}",
            f"- **First-letter Wilcoxon p**: {i8.get('firstletter_p', 'N/A')}",
            f"- **First-letter Cohen's d**: {i8.get('firstletter_d', 'N/A')}",
            f"- Note: {i8.get('note', '')}",
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
