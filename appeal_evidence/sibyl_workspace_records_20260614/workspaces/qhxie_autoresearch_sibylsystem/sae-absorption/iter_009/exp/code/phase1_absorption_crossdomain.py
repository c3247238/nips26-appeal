"""
Phase 1.3: Cross-Domain Absorption Measurement -- Iteration 9 Pilot

Measures SAE feature absorption for city-continent hierarchy at L24.
Uses iter_009 probes (city-continent L24 F1=0.871, sklearn LogisticRegression).

PILOT configuration:
- Single hierarchy: city-continent (highest RAVEL probe quality)
- Single SAE: L24-16k JumpReLU (primary SAE)
- 100 entities x 50 contexts (prompt variations via ICL)
- Token position -2 (matching probe training)
- Bootstrap 95% CI on absorption rate
- Permutation test vs first-letter baseline (if available)

Expected from iter_008 baseline:
- city-continent L24-16k absorption rate ~35.8% (broad definition)
- city-continent L24-16k strict absorption rate ~13.9%
- first-letter L24-16k absorption rate ~34.5%

Dependencies:
- phase1_probe_training (COMPLETED): provides probe_city-continent_L24.npz
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
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats

# ============================================================
# Configuration
# ============================================================
TASK_ID = "phase1_absorption_crossdomain"
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

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = os.environ.get("PILOT_MODE", "PILOT").upper()
if "--full" in sys.argv:
    MODE = "FULL"

# Best layers from Phase 1.1 pilot training (iter_009)
# first-letter L24 sklearn: F1=0.928
# city-continent L24 sklearn: F1=0.871
BEST_LAYERS = {
    "city-continent": 24,
}

# SAE configurations
SAE_CONFIGS = [
    {
        "layer": 24,
        "width": "16k",
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_24/width_16k/canonical",
        "key": "L24_16k",
    },
]

if MODE == "FULL":
    SAE_CONFIGS.extend([
        {
            "layer": 24,
            "width": "65k",
            "release": "gemma-scope-2b-pt-res-canonical",
            "sae_id": "layer_24/width_65k/canonical",
            "key": "L24_65k",
        },
    ])

# Pilot: 100 entities. Full: all available.
MAX_ENTITIES = 100 if MODE == "PILOT" else 2000

# Token position for activation extraction -- MUST match probe training
TOKEN_POS = -2  # iter_009 probes trained at position -2

# Number of ICL examples per prompt
N_ICL = 5

# Bootstrap parameters
N_BOOTSTRAP = 10000 if MODE == "FULL" else 2000

# Probe quality thresholds
QUALITY_GATE_STRICT = 0.90
QUALITY_GATE_RELAXED = 0.85
QUALITY_GATE_MINIMUM = 0.70

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

    # Get hook name from SAE (handle different SAELens versions)
    hook_name = None
    if hasattr(sae.cfg, 'hook_name'):
        hook_name = sae.cfg.hook_name
    elif hasattr(sae.cfg, 'metadata') and sae.cfg.metadata:
        md = sae.cfg.metadata
        if isinstance(md, dict):
            hook_name = md.get('hook_name', md.get('hook_point'))
        elif hasattr(md, 'get'):
            hook_name = md.get('hook_name', md.get('hook_point'))
        elif hasattr(md, 'hook_name'):
            hook_name = md.hook_name

    logger.info(f"  d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}, hook={hook_name}")
    return sae


# ============================================================
# Load pre-trained probe
# ============================================================
def load_probe(hierarchy_name, layer=24):
    """Load saved sklearn LogisticRegression probe from phase1_probe_training."""
    probe_path = PHASE1_DIR / f"probe_{hierarchy_name}_L{layer}.npz"
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}")

    data = np.load(probe_path, allow_pickle=True)
    coef = data["coef"]
    intercept = data["intercept"]
    classes = data["classes"]

    # Reconstruct sklearn probe
    probe = LogisticRegression(max_iter=1)
    probe.classes_ = np.arange(len(classes))
    probe.coef_ = coef
    probe.intercept_ = intercept

    # Get quality metrics from probe training results
    quality_info = {"f1": None, "accuracy": None, "quality_gate": "unknown"}
    quality_sources = [
        PILOT_DIR / "phase1_probe_training.json",
        PHASE1_DIR / "probe_training.json",
    ]

    for qpath in quality_sources:
        if qpath.exists():
            try:
                qdata = json.loads(qpath.read_text())
                key = f"{hierarchy_name}_L{layer}"
                if key in qdata.get("probes", {}):
                    pm = qdata["probes"][key]
                    quality_info = {
                        "f1": pm.get("f1_weighted_test", pm.get("f1_weighted_cv")),
                        "accuracy": pm.get("accuracy_test", pm.get("accuracy_cv")),
                        "balanced_accuracy": pm.get("balanced_accuracy_test"),
                        "n_classes": pm.get("n_classes"),
                        "quality_gate": (
                            "strict" if pm.get("quality_gate_strict")
                            else ("relaxed" if pm.get("quality_gate_relaxed") else "below")
                        ),
                    }
                    logger.info(f"  Probe quality from {qpath.name}: F1={quality_info['f1']:.4f}")
                    break
            except Exception as e:
                logger.warning(f"  Failed to read quality from {qpath}: {e}")

    logger.info(f"  Loaded probe {hierarchy_name}_L{layer}: "
                f"coef={coef.shape}, classes={list(classes)}")

    return probe, classes, quality_info


# ============================================================
# RAVEL data preparation
# ============================================================
def prepare_ravel_data(max_entities=100):
    """Load RAVEL dataset and prepare city-continent entity-attribute pairs."""
    from datasets import load_dataset

    logger.info("Loading RAVEL dataset (hij/ravel, city_entity)...")
    ds = load_dataset("hij/ravel", "city_entity", split="train")
    logger.info(f"  Total RAVEL entries: {len(ds)}")

    cities = list(ds["City"])
    continents = list(ds["Continent"])

    # Filter valid entries
    valid = [
        (c, l) for c, l in zip(cities, continents)
        if c and l and str(l).strip() and str(c).strip()
    ]
    cities_valid, labels_valid = zip(*valid)
    cities_valid, labels_valid = list(cities_valid), list(labels_valid)

    # Merge Australia -> Oceania (consistency with probe training)
    labels_valid = ["Oceania" if l == "Australia" else l for l in labels_valid]

    # Filter classes with >= 3 samples
    label_counts = Counter(labels_valid)
    valid_labels = {l for l, n in label_counts.items() if n >= 3}
    keep = [(c, l) for c, l in zip(cities_valid, labels_valid) if l in valid_labels]
    cities_final = [x[0] for x in keep]
    labels_final = [x[1] for x in keep]

    logger.info(f"  After filtering: {len(cities_final)} cities, "
                f"{len(set(labels_final))} classes")
    logger.info(f"  Class distribution: {Counter(labels_final).most_common()}")

    # Subsample for pilot (stratified)
    if len(cities_final) > max_entities:
        rng = np.random.RandomState(SEED)
        # Stratified subsample: proportional to class frequency
        indices_by_class = defaultdict(list)
        for i, l in enumerate(labels_final):
            indices_by_class[l].append(i)

        selected_indices = []
        for cls, idx_list in indices_by_class.items():
            proportion = len(idx_list) / len(labels_final)
            n_take = max(2, int(proportion * max_entities))
            if n_take > len(idx_list):
                n_take = len(idx_list)
            chosen = rng.choice(idx_list, size=n_take, replace=False)
            selected_indices.extend(chosen.tolist())

        rng.shuffle(selected_indices)
        # Trim to max_entities
        selected_indices = selected_indices[:max_entities]

        cities_final = [cities_final[i] for i in selected_indices]
        labels_final = [labels_final[i] for i in selected_indices]

    unique_labels = sorted(set(labels_final))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    logger.info(f"  Final dataset: {len(cities_final)} cities, "
                f"{len(unique_labels)} classes: {unique_labels}")
    logger.info(f"  Final distribution: {Counter(labels_final).most_common()}")

    return {
        "cities": cities_final,
        "labels": labels_final,
        "label_to_idx": label_to_idx,
        "unique_labels": unique_labels,
    }


# ============================================================
# ICL prompt construction
# ============================================================
def build_icl_prompts(cities, labels, n_icl=5):
    """Build ICL-style prompts for city-continent prediction."""
    base_template = "The city of {entity} is on the continent of"
    answer_template = " {label}"

    examples = list(zip(cities, labels))
    rng = random.Random(SEED)

    prompts = []
    for i, (city, label) in enumerate(zip(cities, labels)):
        # Select ICL examples (not including the current city)
        icl_pool = [(c, l) for c, l in examples if c != city]
        rng.shuffle(icl_pool)
        icl_selected = icl_pool[:n_icl]

        icl_parts = []
        for ex_city, ex_label in icl_selected:
            ex_text = base_template.format(entity=ex_city) + answer_template.format(label=ex_label)
            icl_parts.append(ex_text)

        full_prompt = "\n".join(icl_parts) + "\n" + base_template.format(entity=city)
        prompts.append(full_prompt)

    return prompts


# ============================================================
# Absorption measurement
# ============================================================
def measure_absorption(model, sae, probe, class_labels, cities, labels,
                       label_to_idx, layer=24, token_pos=-2, device="cuda:0"):
    """
    Measure SAE feature absorption for city-continent at L24.

    Absorption = probe correct on raw residual stream but wrong on
    SAE-reconstructed activations (false negative).

    Strict absorption = false negative AND top-5 cosine-similar SAE features
    (main features for the class) do not fire.
    """
    tl_hook = f"blocks.{layer}.hook_resid_post"

    # Pre-compute probe directions and find main SAE features per class
    probe_coefs = torch.tensor(probe.coef_, dtype=torch.float32)
    W_dec = sae.W_dec.detach().float().cpu()

    main_features = {}
    for cls_idx in range(len(class_labels)):
        probe_dir = probe_coefs[cls_idx]
        probe_dir = probe_dir / (probe_dir.norm() + 1e-8)
        cos_sims = F.cosine_similarity(probe_dir.unsqueeze(0), W_dec, dim=-1)
        topk_vals, topk_ids = cos_sims.topk(5)
        main_features[cls_idx] = {
            "feature_ids": topk_ids.tolist(),
            "cos_sims": topk_vals.tolist(),
        }
    del W_dec
    logger.info(f"  Main features computed for {len(main_features)} classes")

    # Log top features per class
    for cls_idx, cls_name in enumerate(class_labels):
        mf = main_features[cls_idx]
        top_cos = mf["cos_sims"][0]
        top_fid = mf["feature_ids"][0]
        logger.info(f"    {cls_name}: top feature={top_fid}, cos_sim={top_cos:.4f}")

    prompts = build_icl_prompts(cities, labels, n_icl=N_ICL)

    per_class_stats = defaultdict(lambda: {
        "total": 0,
        "probe_correct_raw": 0,
        "probe_correct_sae": 0,
        "false_negatives": 0,
        "main_feature_fires": 0,
        "fn_and_main_absent": 0,
        "fn_and_main_present": 0,
    })

    fn_examples = []
    absorption_per_entity = []  # For bootstrap: 1 if absorbed, 0 if not (among probe-correct)
    total_processed = 0
    total_errors = 0

    for i, (city, label, prompt) in enumerate(zip(cities, labels, prompts)):
        if label not in label_to_idx:
            continue
        true_idx = label_to_idx[label]

        # Map to probe class index
        if isinstance(class_labels, np.ndarray):
            cls_list = class_labels.tolist()
        else:
            cls_list = list(class_labels)

        if label not in cls_list:
            continue
        probe_true_idx = cls_list.index(label)

        try:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])

            raw_act = cache[tl_hook][0, token_pos, :].detach().float()

            # SAE encode/decode
            raw_act_device = raw_act.to(device)
            with torch.no_grad():
                sae_features = sae.encode(raw_act_device.unsqueeze(0))
                sae_out = sae.decode(sae_features)

            # Probe predictions
            raw_np = raw_act.detach().cpu().numpy().reshape(1, -1)
            sae_np = sae_out[0].detach().float().cpu().numpy().reshape(1, -1)

            raw_pred = probe.predict(raw_np)[0]
            sae_pred = probe.predict(sae_np)[0]

            probe_correct_raw = (raw_pred == probe_true_idx)
            probe_correct_sae = (sae_pred == probe_true_idx)

            # Check main features for the true class
            mfids = main_features[probe_true_idx]["feature_ids"]
            feat_acts = sae_features[0, mfids].detach().float().cpu()
            any_main_fires = (feat_acts.abs() > 1e-6).any().item()

            del cache

        except Exception as e:
            if total_errors < 5:
                logger.warning(f"  Error processing '{city}': {e}")
            total_errors += 1
            continue

        s = per_class_stats[label]
        s["total"] += 1
        if probe_correct_raw:
            s["probe_correct_raw"] += 1
        if probe_correct_sae:
            s["probe_correct_sae"] += 1
        if any_main_fires:
            s["main_feature_fires"] += 1

        is_false_negative = probe_correct_raw and not probe_correct_sae

        if probe_correct_raw:
            absorption_per_entity.append(1 if is_false_negative else 0)

        if is_false_negative:
            s["false_negatives"] += 1
            if not any_main_fires:
                s["fn_and_main_absent"] += 1
            else:
                s["fn_and_main_present"] += 1

            if len(fn_examples) < 30:
                # Get SAE prediction label
                sae_pred_label = cls_list[sae_pred] if sae_pred < len(cls_list) else f"idx_{sae_pred}"
                raw_pred_label = cls_list[raw_pred] if raw_pred < len(cls_list) else f"idx_{raw_pred}"

                fn_examples.append({
                    "city": city,
                    "true_label": label,
                    "raw_pred_label": raw_pred_label,
                    "sae_pred_label": sae_pred_label,
                    "main_fires": any_main_fires,
                    "top_feature_acts": feat_acts[:3].tolist(),
                    "top_cos_sim": main_features[probe_true_idx]["cos_sims"][0],
                })

        total_processed += 1
        if total_processed % 25 == 0:
            logger.info(f"  Processed {total_processed}/{len(cities)} cities "
                        f"(errors: {total_errors})")
            torch.cuda.empty_cache()

    # Aggregate
    total_raw_correct = sum(s["probe_correct_raw"] for s in per_class_stats.values())
    total_sae_correct = sum(s["probe_correct_sae"] for s in per_class_stats.values())
    total_fn = sum(s["false_negatives"] for s in per_class_stats.values())
    total_fn_main_absent = sum(s["fn_and_main_absent"] for s in per_class_stats.values())
    total_fn_main_present = sum(s["fn_and_main_present"] for s in per_class_stats.values())
    total_tokens = sum(s["total"] for s in per_class_stats.values())

    absorption_rate = total_fn / max(total_raw_correct, 1)
    strict_rate = total_fn_main_absent / max(total_raw_correct, 1)
    probe_raw_accuracy = total_raw_correct / max(total_tokens, 1)
    probe_sae_accuracy = total_sae_correct / max(total_tokens, 1)

    # Per-class rates
    per_class_rates = {}
    for label, s in per_class_stats.items():
        denom = max(s["probe_correct_raw"], 1)
        per_class_rates[label] = {
            **s,
            "absorption_rate": s["false_negatives"] / denom if s["probe_correct_raw"] > 0 else 0.0,
            "strict_rate": s["fn_and_main_absent"] / denom if s["probe_correct_raw"] > 0 else 0.0,
        }

    return {
        "hierarchy": "city-continent",
        "layer": layer,
        "token_pos": token_pos,
        "total_entities": total_tokens,
        "total_errors": total_errors,
        "total_probe_correct_raw": total_raw_correct,
        "total_probe_correct_sae": total_sae_correct,
        "probe_raw_accuracy": float(probe_raw_accuracy),
        "probe_sae_accuracy": float(probe_sae_accuracy),
        "total_false_negatives": total_fn,
        "total_fn_main_absent": total_fn_main_absent,
        "total_fn_main_present": total_fn_main_present,
        "absorption_rate": float(absorption_rate),
        "strict_absorption_rate": float(strict_rate),
        "per_class": per_class_rates,
        "fn_examples": fn_examples,
        "absorption_per_entity": absorption_per_entity,
        "main_features_top": {
            cls_list[k]: {
                "fid": v["feature_ids"][0],
                "cos": round(v["cos_sims"][0], 4),
                "top5_fids": v["feature_ids"],
                "top5_cos": [round(c, 4) for c in v["cos_sims"]],
            }
            for k, v in main_features.items()
            if k < len(cls_list)
        },
    }


# ============================================================
# Bootstrap CI
# ============================================================
def bootstrap_ci(values, n_bootstrap=2000, ci=0.95, seed=42):
    """Bootstrap confidence interval on the mean."""
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
# Permutation test vs first-letter
# ============================================================
def permutation_test_vs_firstletter(crossdomain_per_entity, firstletter_data, n_permutations=5000):
    """
    Two-sample permutation test comparing absorption rates.
    H0: cross-domain and first-letter absorption rates are from the same distribution.
    """
    if firstletter_data is None:
        return {"available": False, "reason": "first-letter results not yet available"}

    # Try to extract first-letter per-entity absorption data
    fl_results = firstletter_data.get("absorption_results", {})
    fl_per_entity = None

    for sae_key, fl_ar in fl_results.items():
        if isinstance(fl_ar, dict) and "absorption_per_entity" in fl_ar:
            fl_per_entity = fl_ar["absorption_per_entity"]
            break

    if fl_per_entity is None:
        # Fallback: use aggregate rate to construct synthetic per-entity data
        for sae_key, fl_ar in fl_results.items():
            if isinstance(fl_ar, dict) and "absorption_rate" in fl_ar:
                fl_rate = fl_ar["absorption_rate"]
                fl_n = fl_ar.get("total_probe_correct", 100)
                fl_fn = fl_ar.get("total_false_negatives", int(fl_rate * fl_n))
                fl_per_entity = [1] * fl_fn + [0] * (fl_n - fl_fn)
                break

    if fl_per_entity is None:
        return {"available": False, "reason": "no first-letter absorption data found"}

    cd_arr = np.array(crossdomain_per_entity, dtype=float)
    fl_arr = np.array(fl_per_entity, dtype=float)

    observed_diff = np.mean(cd_arr) - np.mean(fl_arr)

    # Permutation test
    combined = np.concatenate([cd_arr, fl_arr])
    n_cd = len(cd_arr)
    rng = np.random.RandomState(SEED)

    n_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:n_cd]) - np.mean(combined[n_cd:])
        if abs(perm_diff) >= abs(observed_diff):
            n_extreme += 1

    p_value = (n_extreme + 1) / (n_permutations + 1)

    # Also compute z-test for comparison
    cd_rate = np.mean(cd_arr)
    fl_rate = np.mean(fl_arr)
    p_pool = (sum(cd_arr) + sum(fl_arr)) / (len(cd_arr) + len(fl_arr))
    se_pool = np.sqrt(
        p_pool * (1 - p_pool) * (1 / max(len(cd_arr), 1) + 1 / max(len(fl_arr), 1))
    ) if 0 < p_pool < 1 else 0.01
    z_stat = observed_diff / max(se_pool, 1e-6)
    z_p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Cohen's h (effect size for proportions)
    cohens_h = 2 * (np.arcsin(np.sqrt(cd_rate)) - np.arcsin(np.sqrt(fl_rate)))

    return {
        "available": True,
        "crossdomain_rate": float(cd_rate),
        "firstletter_rate": float(fl_rate),
        "observed_diff": float(observed_diff),
        "permutation_p_value": float(p_value),
        "z_test_p_value": float(z_p_value),
        "z_stat": float(z_stat),
        "cohens_h": float(cohens_h),
        "n_crossdomain": int(len(cd_arr)),
        "n_firstletter": int(len(fl_arr)),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
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
                    "layer": 24,
                    "max_entities": MAX_ENTITIES,
                    "token_pos": TOKEN_POS,
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


# ============================================================
# Experiment state tracking
# ============================================================
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
    report_progress(0, 15, "starting")

    logger.info("=" * 60)
    logger.info("Phase 1.3: Cross-Domain Absorption (city-continent, iter_009 pilot)")
    logger.info(f"Mode: {MODE}, Max entities: {MAX_ENTITIES}, Token pos: {TOKEN_POS}")
    logger.info(f"SAE configs: {[c['key'] for c in SAE_CONFIGS]}")
    logger.info("=" * 60)

    device = "cuda:0"

    # Step 1: Load model
    report_progress(1, 15, "loading_model")
    model = load_model(device=device)

    # Step 2: Load city-continent probe at L24
    report_progress(2, 15, "loading_probe")
    probe, class_labels, quality_info = load_probe("city-continent", layer=24)
    f1 = quality_info.get("f1", 0) or 0

    if f1 >= QUALITY_GATE_STRICT:
        gate_status = "PASS_STRICT"
    elif f1 >= QUALITY_GATE_RELAXED:
        gate_status = "PASS_RELAXED"
    elif f1 >= QUALITY_GATE_MINIMUM:
        gate_status = "BELOW_GATE_INCLUDED"
    else:
        gate_status = "EXCLUDED"

    logger.info(f"  Probe F1={f1:.4f}, gate={gate_status}")

    if gate_status == "EXCLUDED":
        msg = f"Probe F1={f1:.4f} below minimum gate {QUALITY_GATE_MINIMUM}"
        logger.error(msg)
        mark_done("failed", msg)
        update_gpu_progress(time.time() - start_time, "failed")
        update_experiment_state("failed", msg)
        return

    # Step 3: Load RAVEL data
    report_progress(3, 15, "loading_data")
    ravel_data = prepare_ravel_data(max_entities=MAX_ENTITIES)

    cities = ravel_data["cities"]
    labels = ravel_data["labels"]

    # Align RAVEL labels with probe classes
    cls_list = class_labels.tolist() if hasattr(class_labels, 'tolist') else list(class_labels)
    probe_classes_set = set(cls_list)
    ravel_classes = set(labels)
    missing = ravel_classes - probe_classes_set
    if missing:
        logger.warning(f"  {len(missing)} RAVEL labels not in probe: {missing}")
        keep = [(c, l) for c, l in zip(cities, labels) if l in probe_classes_set]
        cities = [x[0] for x in keep]
        labels = [x[1] for x in keep]

    # Build aligned label_to_idx using probe class ordering
    label_to_idx = {l: cls_list.index(l) for l in set(labels) if l in cls_list}
    logger.info(f"  Using {len(cities)} entities, {len(label_to_idx)} classes aligned with probe")

    # Step 4: Load first-letter baseline (if available from parallel experiment)
    report_progress(4, 15, "loading_baseline")
    firstletter_data = None
    fl_paths = [
        PILOT_DIR / "phase1_absorption_firstletter.json",
        RESULTS_DIR / "phase1_absorption_firstletter.json",
    ]
    for fl_path in fl_paths:
        if fl_path.exists():
            try:
                firstletter_data = json.loads(fl_path.read_text())
                logger.info(f"  First-letter baseline loaded from {fl_path.name}")
                break
            except Exception as e:
                logger.warning(f"  Failed to load first-letter from {fl_path}: {e}")

    if firstletter_data is None:
        logger.info("  First-letter baseline not yet available (parallel experiment still running)")

    # Step 5: Measure absorption for each SAE config
    all_results = {}
    step_counter = 5

    for sae_config in SAE_CONFIGS:
        config_key = sae_config["key"]
        step_counter += 1
        report_progress(step_counter, 15, f"absorption_{config_key}",
                        {"sae": config_key, "n_entities": len(cities)})

        logger.info(f"\n{'='*60}")
        logger.info(f"Measuring absorption: city-continent x {config_key}")
        logger.info(f"{'='*60}")

        try:
            sae = load_sae(sae_config["release"], sae_config["sae_id"], device=device)
        except Exception as e:
            logger.error(f"  SAE load failed: {e}")
            all_results[config_key] = {"error": str(e)}
            continue

        try:
            abs_result = measure_absorption(
                model=model,
                sae=sae,
                probe=probe,
                class_labels=class_labels,
                cities=cities,
                labels=labels,
                label_to_idx=label_to_idx,
                layer=24,
                token_pos=TOKEN_POS,
                device=device,
            )

            # Bootstrap CI on per-entity absorption
            entity_ci = bootstrap_ci(
                abs_result["absorption_per_entity"],
                n_bootstrap=N_BOOTSTRAP,
                seed=SEED,
            )

            # Per-class bootstrap CI
            per_class_ci = {}
            for cls_label, cls_stats in abs_result["per_class"].items():
                if cls_stats["probe_correct_raw"] > 0:
                    cls_binary = (
                        [1] * cls_stats["false_negatives"]
                        + [0] * (cls_stats["probe_correct_raw"] - cls_stats["false_negatives"])
                    )
                    per_class_ci[cls_label] = bootstrap_ci(cls_binary, n_bootstrap=N_BOOTSTRAP, seed=SEED)

            all_results[config_key] = {
                "sae_config": sae_config,
                "absorption_rate": abs_result["absorption_rate"],
                "strict_absorption_rate": abs_result["strict_absorption_rate"],
                "probe_raw_accuracy": abs_result["probe_raw_accuracy"],
                "probe_sae_accuracy": abs_result["probe_sae_accuracy"],
                "total_entities": abs_result["total_entities"],
                "total_probe_correct_raw": abs_result["total_probe_correct_raw"],
                "total_probe_correct_sae": abs_result["total_probe_correct_sae"],
                "total_false_negatives": abs_result["total_false_negatives"],
                "total_fn_main_absent": abs_result["total_fn_main_absent"],
                "total_fn_main_present": abs_result["total_fn_main_present"],
                "total_errors": abs_result["total_errors"],
                "bootstrap_ci": entity_ci,
                "per_class": abs_result["per_class"],
                "per_class_bootstrap_ci": per_class_ci,
                "fn_examples": abs_result["fn_examples"][:20],
                "main_features_top": abs_result["main_features_top"],
            }

            logger.info(f"\n  --- Results for {config_key} ---")
            logger.info(f"  Absorption rate: {abs_result['absorption_rate']:.4f}")
            logger.info(f"  Strict absorption rate: {abs_result['strict_absorption_rate']:.4f}")
            logger.info(f"  Probe raw accuracy: {abs_result['probe_raw_accuracy']:.4f}")
            logger.info(f"  Probe SAE accuracy: {abs_result['probe_sae_accuracy']:.4f}")
            logger.info(f"  FN: {abs_result['total_false_negatives']}/{abs_result['total_probe_correct_raw']}")
            logger.info(f"  Bootstrap 95% CI: [{entity_ci['ci_lower']:.4f}, {entity_ci['ci_upper']:.4f}]")
            logger.info(f"  Per-class absorption rates:")
            for cls_label in sorted(abs_result["per_class"].keys()):
                cs = abs_result["per_class"][cls_label]
                logger.info(f"    {cls_label}: {cs['absorption_rate']:.4f} "
                            f"({cs['false_negatives']}/{cs['probe_correct_raw']})")

        except Exception as e:
            logger.error(f"  Absorption measurement failed: {e}", exc_info=True)
            all_results[config_key] = {"error": str(e)}

        del sae
        gc.collect()
        torch.cuda.empty_cache()

    # Step 6: Permutation test vs first-letter
    step_counter += 1
    report_progress(step_counter, 15, "permutation_test")

    perm_test_results = {}
    for config_key, ar in all_results.items():
        if "error" not in ar and "absorption_per_entity" not in ar:
            # Get absorption_per_entity from the measurement
            pass
        if "error" not in ar:
            # We need the per-entity data; it was stored in abs_result but not in all_results
            # Re-extract from the per-class data
            per_entity = []
            for cls_label, cls_stats in ar["per_class"].items():
                per_entity.extend([1] * cls_stats["false_negatives"])
                per_entity.extend([0] * (cls_stats["probe_correct_raw"] - cls_stats["false_negatives"]))

            perm_result = permutation_test_vs_firstletter(per_entity, firstletter_data)
            perm_test_results[config_key] = perm_result

            if perm_result["available"]:
                logger.info(f"\n  Permutation test vs first-letter ({config_key}):")
                logger.info(f"    Cross-domain rate: {perm_result['crossdomain_rate']:.4f}")
                logger.info(f"    First-letter rate: {perm_result['firstletter_rate']:.4f}")
                logger.info(f"    Difference: {perm_result['observed_diff']:+.4f}")
                logger.info(f"    Permutation p: {perm_result['permutation_p_value']:.4f}")
                logger.info(f"    Cohen's h: {perm_result['cohens_h']:.4f}")
                sig = " ***" if perm_result["significant_001"] else (" *" if perm_result["significant_005"] else "")
                logger.info(f"    Significant: {perm_result['significant_005']}{sig}")
            else:
                logger.info(f"  Permutation test not available: {perm_result.get('reason', 'unknown')}")

    # Step 7: Compare with iter_008 baseline
    step_counter += 1
    report_progress(step_counter, 15, "iter008_comparison")

    iter008_comparison = {}
    iter008_rates = {
        "L24_16k": {"absorption_rate": 0.3584, "strict_rate": 0.1387, "n": 173},
        "L24_65k": {"absorption_rate": None, "strict_rate": None, "n": None},
    }

    for config_key, ar in all_results.items():
        if "error" not in ar and config_key in iter008_rates:
            i8 = iter008_rates[config_key]
            if i8["absorption_rate"] is not None:
                diff = ar["absorption_rate"] - i8["absorption_rate"]
                iter008_comparison[config_key] = {
                    "iter_008_rate": i8["absorption_rate"],
                    "iter_008_strict": i8["strict_rate"],
                    "iter_009_rate": ar["absorption_rate"],
                    "iter_009_strict": ar["strict_absorption_rate"],
                    "difference": float(diff),
                    "within_10pp": abs(diff) <= 0.10,
                }
                logger.info(f"\n  iter_008 comparison ({config_key}):")
                logger.info(f"    iter_008: {i8['absorption_rate']:.4f} (strict: {i8['strict_rate']:.4f})")
                logger.info(f"    iter_009: {ar['absorption_rate']:.4f} (strict: {ar['strict_absorption_rate']:.4f})")
                logger.info(f"    Difference: {diff:+.4f} ({'OK' if abs(diff) <= 0.10 else 'LARGE'})")

    # Step 8: Compile final output
    step_counter += 1
    report_progress(step_counter, 15, "compiling_output")

    elapsed = time.time() - start_time

    # Pilot pass criteria
    has_results = any("error" not in v for v in all_results.values())
    has_nonzero = any(
        v.get("absorption_rate", 0) > 0
        for v in all_results.values()
        if isinstance(v, dict) and "error" not in v
    )
    within_baseline = all(
        c.get("within_10pp", True)
        for c in iter008_comparison.values()
    )
    pilot_pass = has_results and has_nonzero

    # Summary table
    summary_table = []
    for config_key, ar in all_results.items():
        if "error" not in ar:
            summary_table.append({
                "hierarchy": "city-continent",
                "sae_config": config_key,
                "probe_layer": 24,
                "absorption_rate": ar["absorption_rate"],
                "strict_rate": ar["strict_absorption_rate"],
                "probe_raw_accuracy": ar["probe_raw_accuracy"],
                "probe_sae_accuracy": ar["probe_sae_accuracy"],
                "n_entities": ar["total_entities"],
                "n_probe_correct": ar["total_probe_correct_raw"],
                "n_fn": ar["total_false_negatives"],
                "ci_lower": ar["bootstrap_ci"]["ci_lower"],
                "ci_upper": ar["bootstrap_ci"]["ci_upper"],
                "probe_f1": f1,
                "quality_gate": gate_status,
            })

    output = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "token_pos": TOKEN_POS,
        "probe_info": {
            "hierarchy": "city-continent",
            "layer": 24,
            "f1": f1,
            "gate_status": gate_status,
            "classes": cls_list,
            "n_classes": len(cls_list),
        },
        "data_info": {
            "n_entities": len(cities),
            "n_classes_used": len(label_to_idx),
            "class_distribution": dict(Counter(labels).most_common()),
        },
        "sae_configs_tested": [c["key"] for c in SAE_CONFIGS],
        "absorption_results": all_results,
        "summary_table": summary_table,
        "permutation_test_vs_firstletter": perm_test_results,
        "iter008_comparison": iter008_comparison,
        "pilot_criteria_met": pilot_pass,
        "pilot_criteria_details": {
            "has_results": has_results,
            "has_nonzero_absorption": has_nonzero,
            "within_iter008_baseline": within_baseline,
            "pass_criteria": "Absorption rate computed. Rate within 10pp of iter_008 baseline. Permutation test vs first-letter computed.",
        },
        "methodology_notes": {
            "token_position": TOKEN_POS,
            "icl_examples": N_ICL,
            "bootstrap_resamples": N_BOOTSTRAP,
            "absorption_definition": (
                "False negative: probe correct on raw residual stream, "
                "wrong on SAE-reconstructed activations. "
                "Strict: false negative AND main SAE features (top-5 cosine) do not fire."
            ),
            "probe_caveat": (
                f"city-continent probe F1={f1:.4f} is below strict gate (0.90). "
                "Some 'absorption' may reflect probe errors rather than SAE failure."
            ),
        },
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    # Save
    out_path = PILOT_DIR / f"{TASK_ID}.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nSaved: {out_path}")

    # Summary markdown
    summary_md = generate_summary_md(output)
    md_path = PILOT_DIR / f"{TASK_ID}_summary.md"
    md_path.write_text(summary_md)
    logger.info(f"Summary: {md_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    summary_text = (
        f"Phase 1.3 cross-domain absorption (city-continent, iter_009 {MODE}). "
        f"Token pos: {TOKEN_POS}. Time: {elapsed/60:.1f}min. "
    )
    if summary_table:
        for r in summary_table:
            summary_text += (
                f"{r['sae_config']}: absorption={r['absorption_rate']:.4f} "
                f"(strict={r['strict_rate']:.4f}), "
                f"CI=[{r['ci_lower']:.3f},{r['ci_upper']:.3f}]. "
            )
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
        "# Phase 1.3: Cross-Domain Absorption (city-continent) -- Iter 9 Pilot",
        "",
        f"**Status**: {'PASS' if results.get('pilot_criteria_met') else 'PARTIAL'}",
        f"**Time**: {results.get('elapsed_minutes', 0):.1f} minutes",
        f"**Token position**: {results.get('token_pos', '?')}",
        "",
        "## Probe Quality",
        "",
    ]

    pi = results.get("probe_info", {})
    lines.append(f"- **Hierarchy**: {pi.get('hierarchy', '?')}")
    lines.append(f"- **Layer**: {pi.get('layer', '?')}")
    lines.append(f"- **F1**: {pi.get('f1', '?'):.4f}" if isinstance(pi.get('f1'), (int, float)) else f"- **F1**: {pi.get('f1', '?')}")
    lines.append(f"- **Gate**: {pi.get('gate_status', '?')}")
    lines.append(f"- **Classes**: {pi.get('classes', '?')}")

    # Absorption results table
    summary_table = results.get("summary_table", [])
    if summary_table:
        lines.extend([
            "", "## Absorption Results", "",
            "| SAE Config | Absorption Rate | Strict Rate | CI (95%) | Probe Raw Acc | Probe SAE Acc | N Entities | N FN |",
            "|------------|-----------------|-------------|----------|---------------|---------------|------------|------|",
        ])
        for r in summary_table:
            lines.append(
                f"| {r['sae_config']} | {r['absorption_rate']:.4f} | {r['strict_rate']:.4f} | "
                f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] | "
                f"{r['probe_raw_accuracy']:.4f} | {r['probe_sae_accuracy']:.4f} | "
                f"{r['n_entities']} | {r['n_fn']} |"
            )
    else:
        lines.extend(["", "*No absorption results computed.*"])

    # Per-class breakdown
    for config_key, ar in results.get("absorption_results", {}).items():
        if isinstance(ar, dict) and "per_class" in ar:
            lines.extend([
                "", f"### Per-Class Breakdown ({config_key})", "",
                "| Class | Total | Probe Correct | FN | Absorption Rate | Main Feature Fires |",
                "|-------|-------|---------------|-----|-----------------|-------------------|",
            ])
            for cls_label in sorted(ar["per_class"].keys()):
                cs = ar["per_class"][cls_label]
                lines.append(
                    f"| {cls_label} | {cs['total']} | {cs['probe_correct_raw']} | "
                    f"{cs['false_negatives']} | {cs['absorption_rate']:.4f} | "
                    f"{cs['main_feature_fires']} |"
                )

    # Comparison with iter_008
    i8comp = results.get("iter008_comparison", {})
    if i8comp:
        lines.extend(["", "## Comparison with iter_008", ""])
        for config_key, comp in i8comp.items():
            lines.append(f"- **{config_key}**: iter_008={comp['iter_008_rate']:.4f}, "
                         f"iter_009={comp['iter_009_rate']:.4f}, "
                         f"diff={comp['difference']:+.4f} "
                         f"({'OK' if comp['within_10pp'] else 'LARGE'})")

    # Permutation test
    perm = results.get("permutation_test_vs_firstletter", {})
    for config_key, pt in perm.items():
        if pt.get("available"):
            sig = "Yes" if pt["significant_005"] else "No"
            lines.extend([
                "", f"## Permutation Test vs First-Letter ({config_key})", "",
                f"- Cross-domain rate: {pt['crossdomain_rate']:.4f}",
                f"- First-letter rate: {pt['firstletter_rate']:.4f}",
                f"- Difference: {pt['observed_diff']:+.4f}",
                f"- Permutation p-value: {pt['permutation_p_value']:.4f}",
                f"- Cohen's h: {pt['cohens_h']:.4f}",
                f"- Significant (p<0.05): {sig}",
            ])
        else:
            lines.extend([
                "", f"## Permutation Test vs First-Letter ({config_key})", "",
                f"*Not available: {pt.get('reason', 'unknown')}*",
            ])

    # FN examples
    for config_key, ar in results.get("absorption_results", {}).items():
        if isinstance(ar, dict) and "fn_examples" in ar and ar["fn_examples"]:
            lines.extend(["", f"## Example False Negatives ({config_key})", ""])
            for fn in ar["fn_examples"][:10]:
                lines.append(
                    f"- **{fn['city']}**: true={fn['true_label']}, "
                    f"raw_pred={fn['raw_pred_label']}, sae_pred={fn['sae_pred_label']}, "
                    f"main_fires={fn['main_fires']}"
                )

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        update_gpu_progress(time.time() - (globals().get("start_time", time.time())), "failed")
        update_experiment_state("failed", str(e))
        sys.exit(1)
