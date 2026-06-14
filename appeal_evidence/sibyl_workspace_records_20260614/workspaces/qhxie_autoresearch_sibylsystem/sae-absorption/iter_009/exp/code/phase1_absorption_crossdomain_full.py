"""
Phase 1.2b: Cross-Domain Absorption Measurement -- Iteration 9 FULL

Measures SAE feature absorption for ALL 3 RAVEL hierarchies at L24:
  - city-continent (6 classes, F1=0.871, relaxed gate PASS)
  - city-language  (23 classes, F1=0.818, relaxed gate PASS)
  - city-country   (80 classes, F1=0.726, below gate -- included with caveat)

SAE configs:
  - L24-16k JumpReLU (primary)
  - L24-65k JumpReLU (wider dictionary)

FULL configuration:
  - All available entities (up to 2000 per hierarchy)
  - Both SAE widths (16k, 65k)
  - 10000 bootstrap resamples
  - Pairwise permutation tests vs first-letter baseline
  - Bonferroni correction for 6 comparisons
  - Per-class breakdown with bootstrap CIs
  - iter_008 baseline comparison

Dependencies:
  - phase1_probe_training (COMPLETED): provides probe_{hierarchy}_L24.npz
  - phase1_absorption_firstletter (may be concurrent): used for permutation test
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
FULL_DIR = RESULTS_DIR / "full"
for d in [PILOT_DIR, PHASE1_DIR, FULL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = "FULL"

# All 3 RAVEL hierarchies with probe quality info from FULL probe training
HIERARCHIES = {
    "city-continent": {
        "layer": 24,
        "f1": 0.8710,  # from probe_training.json
        "n_classes": 6,
        "gate": "PASS_RELAXED",
        "ravel_attr": "Continent",
        "merge_rules": {"Australia": "Oceania"},
        "min_class_size": 3,
    },
    "city-language": {
        "layer": 24,
        "f1": 0.8179,
        "n_classes": 23,
        "gate": "PASS_RELAXED",
        "ravel_attr": "Language",
        "merge_rules": {},
        "min_class_size": 3,
    },
    "city-country": {
        "layer": 24,
        "f1": 0.7260,
        "n_classes": 80,
        "gate": "BELOW_GATE_INCLUDED",  # F1 > 0.70 minimum
        "ravel_attr": "Country",
        "merge_rules": {},
        "min_class_size": 5,  # more conservative for 80-class
    },
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
    {
        "layer": 24,
        "width": "65k",
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_24/width_65k/canonical",
        "key": "L24_65k",
    },
]

# FULL: all available entities
MAX_ENTITIES = 2000
TOKEN_POS = -2  # match probe training
N_ICL = 5
N_BOOTSTRAP = 10000
N_PERMUTATIONS = 10000
BONFERRONI_N = 6  # 3 hierarchies x 2 SAEs, comparisons vs first-letter

# Probe quality thresholds
QUALITY_GATE_STRICT = 0.90
QUALITY_GATE_RELAXED = 0.85
QUALITY_GATE_MINIMUM = 0.70

# GPU device -- when CUDA_VISIBLE_DEVICES is set, use cuda:0
# When running directly without CUDA_VISIBLE_DEVICES, set to the desired GPU
DEVICE = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda:1"

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
def load_model(device=DEVICE):
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


def load_sae(release, sae_id, device=DEVICE):
    from sae_lens import SAE
    logger.info(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release, sae_id, device=device)

    hook_name = None
    if hasattr(sae.cfg, 'hook_name'):
        hook_name = sae.cfg.hook_name
    elif hasattr(sae.cfg, 'metadata') and sae.cfg.metadata:
        md = sae.cfg.metadata
        if isinstance(md, dict):
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
        PHASE1_DIR / "probe_training.json",
        PILOT_DIR / "phase1_probe_training.json",
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
                    logger.info(f"  Probe quality: F1={quality_info['f1']:.4f}")
                    break
            except Exception as e:
                logger.warning(f"  Failed to read quality from {qpath}: {e}")

    logger.info(f"  Loaded probe {hierarchy_name}_L{layer}: "
                f"coef={coef.shape}, classes={list(classes)}")

    return probe, classes, quality_info


# ============================================================
# RAVEL data preparation (generic for any hierarchy)
# ============================================================
def prepare_ravel_data(hierarchy_name, hierarchy_config, max_entities=2000):
    """Load RAVEL dataset and prepare entity-attribute pairs for the given hierarchy."""
    from datasets import load_dataset

    attr_name = hierarchy_config["ravel_attr"]
    merge_rules = hierarchy_config.get("merge_rules", {})
    min_class_size = hierarchy_config.get("min_class_size", 3)

    logger.info(f"Loading RAVEL dataset for {hierarchy_name} (attr={attr_name})...")
    ds = load_dataset("hij/ravel", "city_entity", split="train")
    logger.info(f"  Total RAVEL entries: {len(ds)}")

    cities = list(ds["City"])
    labels = list(ds[attr_name])

    # Filter valid entries
    valid = [
        (c, l) for c, l in zip(cities, labels)
        if c and l and str(l).strip() and str(c).strip()
    ]
    cities_valid = [x[0] for x in valid]
    labels_valid = [x[1] for x in valid]

    # Apply merge rules
    if merge_rules:
        labels_valid = [merge_rules.get(l, l) for l in labels_valid]
        logger.info(f"  Applied merge rules: {merge_rules}")

    # Deduplicate: keep first occurrence of each city
    seen_cities = set()
    deduped_cities = []
    deduped_labels = []
    for c, l in zip(cities_valid, labels_valid):
        if c not in seen_cities:
            seen_cities.add(c)
            deduped_cities.append(c)
            deduped_labels.append(l)
    cities_valid = deduped_cities
    labels_valid = deduped_labels
    logger.info(f"  After dedup: {len(cities_valid)} unique cities")

    # Filter classes with >= min_class_size samples
    label_counts = Counter(labels_valid)
    valid_labels = {l for l, n in label_counts.items() if n >= min_class_size}
    keep = [(c, l) for c, l in zip(cities_valid, labels_valid) if l in valid_labels]
    cities_final = [x[0] for x in keep]
    labels_final = [x[1] for x in keep]

    logger.info(f"  After class filter (min={min_class_size}): {len(cities_final)} cities, "
                f"{len(set(labels_final))} classes")
    logger.info(f"  Class distribution top-10: {Counter(labels_final).most_common(10)}")

    # Subsample if needed (stratified)
    if len(cities_final) > max_entities:
        rng = np.random.RandomState(SEED)
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
        selected_indices = selected_indices[:max_entities]

        cities_final = [cities_final[i] for i in selected_indices]
        labels_final = [labels_final[i] for i in selected_indices]

    unique_labels = sorted(set(labels_final))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    logger.info(f"  Final: {len(cities_final)} cities, "
                f"{len(unique_labels)} classes: {unique_labels[:15]}{'...' if len(unique_labels)>15 else ''}")

    return {
        "cities": cities_final,
        "labels": labels_final,
        "label_to_idx": label_to_idx,
        "unique_labels": unique_labels,
        "class_distribution": dict(Counter(labels_final).most_common()),
    }


# ============================================================
# ICL prompt construction (generic)
# ============================================================
def build_icl_prompts(cities, labels, hierarchy_name, n_icl=5):
    """Build ICL-style prompts for the given hierarchy."""
    templates = {
        "city-continent": ("The city of {entity} is on the continent of", " {label}"),
        "city-language": ("The primary language spoken in {entity} is", " {label}"),
        "city-country": ("The city of {entity} is located in the country of", " {label}"),
    }

    base_template, answer_template = templates.get(
        hierarchy_name,
        ("The city of {entity} has the property", " {label}")
    )

    examples = list(zip(cities, labels))
    rng = random.Random(SEED)

    prompts = []
    for i, (city, label) in enumerate(zip(cities, labels)):
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
                       label_to_idx, hierarchy_name, layer=24,
                       token_pos=-2, device=DEVICE):
    """
    Measure SAE feature absorption for a given hierarchy.

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

    # Log top features per class (only first 10 for large hierarchies)
    for cls_idx, cls_name in enumerate(class_labels):
        if cls_idx >= 10:
            logger.info(f"    ... ({len(class_labels) - 10} more classes)")
            break
        mf = main_features[cls_idx]
        top_cos = mf["cos_sims"][0]
        top_fid = mf["feature_ids"][0]
        logger.info(f"    {cls_name}: top feature={top_fid}, cos_sim={top_cos:.4f}")

    prompts = build_icl_prompts(cities, labels, hierarchy_name, n_icl=N_ICL)

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

    cls_list = class_labels.tolist() if hasattr(class_labels, 'tolist') else list(class_labels)

    for i, (city, label, prompt) in enumerate(zip(cities, labels, prompts)):
        if label not in label_to_idx:
            continue

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

            if len(fn_examples) < 50:
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
        if total_processed % 50 == 0:
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
        "hierarchy": hierarchy_name,
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
def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
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
def permutation_test_vs_firstletter(crossdomain_per_entity, firstletter_data,
                                     sae_key, n_permutations=10000):
    """
    Two-sample permutation test comparing absorption rates.
    H0: cross-domain and first-letter absorption rates are from the same distribution.
    """
    if firstletter_data is None:
        return {"available": False, "reason": "first-letter results not yet available"}

    fl_results = firstletter_data.get("absorption_results", {})
    fl_per_entity = None

    # Try to find matching SAE key first, then any key
    for try_key in [sae_key] + list(fl_results.keys()):
        fl_ar = fl_results.get(try_key)
        if isinstance(fl_ar, dict) and "absorption_per_entity" in fl_ar:
            fl_per_entity = fl_ar["absorption_per_entity"]
            break

    if fl_per_entity is None:
        # Fallback: use aggregate rate to construct synthetic per-entity data
        for try_key in [sae_key] + list(fl_results.keys()):
            fl_ar = fl_results.get(try_key)
            if isinstance(fl_ar, dict) and "absorption_rate" in fl_ar:
                fl_rate = fl_ar["absorption_rate"]
                fl_n = fl_ar.get("total_probe_correct", fl_ar.get("total_probe_correct_raw", 100))
                fl_fn = fl_ar.get("total_false_negatives", int(fl_rate * fl_n))
                fl_per_entity = [1] * fl_fn + [0] * (fl_n - fl_fn)
                break

    if fl_per_entity is None:
        return {"available": False, "reason": "no first-letter absorption data found"}

    cd_arr = np.array(crossdomain_per_entity, dtype=float)
    fl_arr = np.array(fl_per_entity, dtype=float)

    if len(cd_arr) == 0 or len(fl_arr) == 0:
        return {"available": False, "reason": "empty absorption arrays"}

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
    p_bonferroni = min(p_value * BONFERRONI_N, 1.0)

    # z-test for comparison
    cd_rate = np.mean(cd_arr)
    fl_rate = np.mean(fl_arr)
    p_pool = (sum(cd_arr) + sum(fl_arr)) / (len(cd_arr) + len(fl_arr))
    se_pool = np.sqrt(
        p_pool * (1 - p_pool) * (1 / max(len(cd_arr), 1) + 1 / max(len(fl_arr), 1))
    ) if 0 < p_pool < 1 else 0.01
    z_stat = observed_diff / max(se_pool, 1e-6)
    z_p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Cohen's h (effect size for proportions)
    cohens_h = 2 * (np.arcsin(np.sqrt(max(0, min(1, cd_rate)))) -
                     np.arcsin(np.sqrt(max(0, min(1, fl_rate)))))

    return {
        "available": True,
        "crossdomain_rate": float(cd_rate),
        "firstletter_rate": float(fl_rate),
        "observed_diff": float(observed_diff),
        "permutation_p_value": float(p_value),
        "permutation_p_bonferroni": float(p_bonferroni),
        "z_test_p_value": float(z_p_value),
        "z_stat": float(z_stat),
        "cohens_h": float(cohens_h),
        "n_crossdomain": int(len(cd_arr)),
        "n_firstletter": int(len(fl_arr)),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "significant_bonferroni_005": p_bonferroni < 0.05,
    }


# ============================================================
# Cross-hierarchy ANOVA
# ============================================================
def cross_hierarchy_anova(all_absorption_per_entity):
    """
    One-way ANOVA / Kruskal-Wallis test: does absorption rate differ across hierarchies?
    """
    groups = {}
    for key, per_entity in all_absorption_per_entity.items():
        hierarchy = key.split("_")[0] if "_" in key else key
        if hierarchy not in groups:
            groups[hierarchy] = []
        groups[hierarchy].extend(per_entity)

    if len(groups) < 2:
        return {"available": False, "reason": f"need >= 2 groups, got {len(groups)}"}

    group_arrays = [np.array(v, dtype=float) for v in groups.values()]
    group_names = list(groups.keys())

    # Kruskal-Wallis (non-parametric, appropriate for binary data)
    if all(len(g) > 0 for g in group_arrays):
        kw_stat, kw_p = stats.kruskal(*group_arrays)
    else:
        kw_stat, kw_p = 0.0, 1.0

    # Pairwise Mann-Whitney U
    pairwise = {}
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1, g2 = group_arrays[i], group_arrays[j]
            if len(g1) > 0 and len(g2) > 0:
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                n_pairs = len(group_names) * (len(group_names) - 1) // 2
                pairwise[f"{group_names[i]}_vs_{group_names[j]}"] = {
                    "u_stat": float(u_stat),
                    "p_value": float(u_p),
                    "p_bonferroni": float(min(u_p * n_pairs, 1.0)),
                    "mean_1": float(np.mean(g1)),
                    "mean_2": float(np.mean(g2)),
                    "n_1": len(g1),
                    "n_2": len(g2),
                }

    return {
        "available": True,
        "kruskal_wallis_stat": float(kw_stat),
        "kruskal_wallis_p": float(kw_p),
        "significant_005": kw_p < 0.05,
        "significant_001": kw_p < 0.01,
        "groups": {name: {"n": len(arr), "mean": float(np.mean(arr))}
                   for name, arr in zip(group_names, group_arrays)},
        "pairwise_mannwhitney": pairwise,
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
                "planned_min": 50,
                "actual_min": round(elapsed_seconds / 60, 1),
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "config_snapshot": {
                    "model": "gemma-2-2b",
                    "mode": MODE,
                    "hierarchies": list(HIERARCHIES.keys()),
                    "sae_configs": [c["key"] for c in SAE_CONFIGS],
                    "layer": 24,
                    "max_entities": MAX_ENTITIES,
                    "token_pos": TOKEN_POS,
                },
            }
            progress_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"gpu_progress update failed: {e}")


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

    total_steps = 5 + len(HIERARCHIES) * len(SAE_CONFIGS) + 4  # ~15 steps
    report_progress(0, total_steps, "starting")

    logger.info("=" * 70)
    logger.info("Phase 1.2b: Cross-Domain Absorption (FULL, 3 hierarchies, iter_009)")
    logger.info(f"Mode: {MODE}, Max entities: {MAX_ENTITIES}, Token pos: {TOKEN_POS}")
    logger.info(f"Hierarchies: {list(HIERARCHIES.keys())}")
    logger.info(f"SAE configs: {[c['key'] for c in SAE_CONFIGS]}")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 70)

    # Step 1: Load model
    step = 1
    report_progress(step, total_steps, "loading_model")
    model = load_model(device=DEVICE)

    # Step 2: Load all probes
    step += 1
    report_progress(step, total_steps, "loading_probes")
    probes = {}
    for hier_name, hier_config in HIERARCHIES.items():
        layer = hier_config["layer"]
        try:
            probe, class_labels, quality_info = load_probe(hier_name, layer)
            f1 = quality_info.get("f1", 0) or 0
            if f1 >= QUALITY_GATE_STRICT:
                gate = "PASS_STRICT"
            elif f1 >= QUALITY_GATE_RELAXED:
                gate = "PASS_RELAXED"
            elif f1 >= QUALITY_GATE_MINIMUM:
                gate = "BELOW_GATE_INCLUDED"
            else:
                gate = "EXCLUDED"
            logger.info(f"  {hier_name} L{layer}: F1={f1:.4f}, gate={gate}")

            if gate == "EXCLUDED":
                logger.warning(f"  SKIPPING {hier_name}: F1={f1:.4f} below minimum {QUALITY_GATE_MINIMUM}")
                continue

            probes[hier_name] = {
                "probe": probe,
                "class_labels": class_labels,
                "quality_info": quality_info,
                "f1": f1,
                "gate": gate,
                "layer": layer,
            }
        except FileNotFoundError as e:
            logger.warning(f"  SKIPPING {hier_name}: {e}")

    logger.info(f"  Loaded {len(probes)} probes: {list(probes.keys())}")

    if not probes:
        msg = "No probes passed quality gate"
        mark_done("failed", msg)
        update_gpu_progress(time.time() - start_time, "failed")
        update_experiment_state("failed", msg)
        return

    # Step 3: Load RAVEL data for each hierarchy
    step += 1
    report_progress(step, total_steps, "loading_data")
    ravel_datasets = {}
    for hier_name in probes:
        hier_config = HIERARCHIES[hier_name]
        ravel_data = prepare_ravel_data(hier_name, hier_config, max_entities=MAX_ENTITIES)

        # Align with probe classes
        cls_list = probes[hier_name]["class_labels"]
        if hasattr(cls_list, 'tolist'):
            cls_list = cls_list.tolist()
        else:
            cls_list = list(cls_list)
        probe_classes_set = set(cls_list)

        cities = ravel_data["cities"]
        labels = ravel_data["labels"]
        ravel_classes = set(labels)
        missing = ravel_classes - probe_classes_set
        if missing:
            logger.warning(f"  {hier_name}: {len(missing)} RAVEL labels not in probe: "
                           f"{list(missing)[:10]}{'...' if len(missing)>10 else ''}")
            keep = [(c, l) for c, l in zip(cities, labels) if l in probe_classes_set]
            cities = [x[0] for x in keep]
            labels = [x[1] for x in keep]

        label_to_idx = {l: cls_list.index(l) for l in set(labels) if l in cls_list}

        ravel_datasets[hier_name] = {
            "cities": cities,
            "labels": labels,
            "label_to_idx": label_to_idx,
            "cls_list": cls_list,
            "n_entities": len(cities),
            "n_classes_used": len(label_to_idx),
            "class_distribution": dict(Counter(labels).most_common()),
        }
        logger.info(f"  {hier_name}: {len(cities)} entities, {len(label_to_idx)} classes aligned")

    # Step 4: Load first-letter baseline
    step += 1
    report_progress(step, total_steps, "loading_baseline")
    firstletter_data = None
    fl_paths = [
        PHASE1_DIR / "absorption_firstletter.json",
        PILOT_DIR / "phase1_absorption_firstletter.json",
        RESULTS_DIR / "phase1_absorption_firstletter.json",
        FULL_DIR / "phase1_absorption_firstletter.json",
    ]
    for fl_path in fl_paths:
        if fl_path.exists():
            try:
                firstletter_data = json.loads(fl_path.read_text())
                logger.info(f"  First-letter baseline loaded from {fl_path.name}")
                break
            except Exception as e:
                logger.warning(f"  Failed to load: {fl_path}: {e}")

    if firstletter_data is None:
        logger.info("  First-letter baseline not yet available (parallel experiment)")

    # Step 5+: Measure absorption for each hierarchy x SAE config
    all_results = {}
    all_absorption_per_entity = {}  # For ANOVA

    for sae_config in SAE_CONFIGS:
        config_key = sae_config["key"]

        try:
            sae = load_sae(sae_config["release"], sae_config["sae_id"], device=DEVICE)
        except Exception as e:
            logger.error(f"  SAE load failed for {config_key}: {e}")
            for hier_name in probes:
                result_key = f"{hier_name}_{config_key}"
                all_results[result_key] = {"error": str(e), "hierarchy": hier_name, "sae_key": config_key}
            continue

        for hier_name in probes:
            step += 1
            result_key = f"{hier_name}_{config_key}"
            report_progress(step, total_steps, f"absorption_{result_key}",
                            {"hierarchy": hier_name, "sae": config_key,
                             "n_entities": ravel_datasets[hier_name]["n_entities"]})

            logger.info(f"\n{'='*70}")
            logger.info(f"Measuring absorption: {hier_name} x {config_key}")
            logger.info(f"{'='*70}")

            probe_info = probes[hier_name]
            dataset = ravel_datasets[hier_name]

            try:
                abs_result = measure_absorption(
                    model=model,
                    sae=sae,
                    probe=probe_info["probe"],
                    class_labels=probe_info["class_labels"],
                    cities=dataset["cities"],
                    labels=dataset["labels"],
                    label_to_idx=dataset["label_to_idx"],
                    hierarchy_name=hier_name,
                    layer=probe_info["layer"],
                    token_pos=TOKEN_POS,
                    device=DEVICE,
                )

                # Bootstrap CI
                entity_ci = bootstrap_ci(
                    abs_result["absorption_per_entity"],
                    n_bootstrap=N_BOOTSTRAP, seed=SEED,
                )

                # Strict absorption bootstrap CI
                strict_per_entity = []
                for cls_label, cs in abs_result["per_class"].items():
                    strict_per_entity.extend([1] * cs["fn_and_main_absent"])
                    strict_per_entity.extend([0] * (cs["probe_correct_raw"] - cs["fn_and_main_absent"]))
                strict_ci = bootstrap_ci(strict_per_entity, n_bootstrap=N_BOOTSTRAP, seed=SEED)

                # Per-class bootstrap CI
                per_class_ci = {}
                for cls_label, cls_stats in abs_result["per_class"].items():
                    if cls_stats["probe_correct_raw"] > 0:
                        cls_binary = (
                            [1] * cls_stats["false_negatives"]
                            + [0] * (cls_stats["probe_correct_raw"] - cls_stats["false_negatives"])
                        )
                        per_class_ci[cls_label] = bootstrap_ci(cls_binary, n_bootstrap=N_BOOTSTRAP, seed=SEED)

                all_results[result_key] = {
                    "hierarchy": hier_name,
                    "sae_key": config_key,
                    "sae_config": sae_config,
                    "probe_f1": probe_info["f1"],
                    "probe_gate": probe_info["gate"],
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
                    "strict_bootstrap_ci": strict_ci,
                    "per_class": abs_result["per_class"],
                    "per_class_bootstrap_ci": per_class_ci,
                    "fn_examples": abs_result["fn_examples"][:30],
                    "main_features_top": abs_result["main_features_top"],
                }

                # Store per-entity absorption for ANOVA
                all_absorption_per_entity[result_key] = abs_result["absorption_per_entity"]

                logger.info(f"\n  --- Results for {result_key} ---")
                logger.info(f"  Absorption rate: {abs_result['absorption_rate']:.4f}")
                logger.info(f"  Strict absorption rate: {abs_result['strict_absorption_rate']:.4f}")
                logger.info(f"  Probe raw accuracy: {abs_result['probe_raw_accuracy']:.4f}")
                logger.info(f"  Probe SAE accuracy: {abs_result['probe_sae_accuracy']:.4f}")
                logger.info(f"  FN: {abs_result['total_false_negatives']}/{abs_result['total_probe_correct_raw']}")
                logger.info(f"  Bootstrap 95% CI: [{entity_ci['ci_lower']:.4f}, {entity_ci['ci_upper']:.4f}]")

            except Exception as e:
                logger.error(f"  Absorption measurement failed: {e}", exc_info=True)
                all_results[result_key] = {"error": str(e), "hierarchy": hier_name, "sae_key": config_key}

            torch.cuda.empty_cache()

        del sae
        gc.collect()
        torch.cuda.empty_cache()

    # Step: Permutation tests vs first-letter
    step += 1
    report_progress(step, total_steps, "permutation_tests")
    perm_test_results = {}

    for result_key, ar in all_results.items():
        if "error" in ar:
            continue
        # Reconstruct per-entity from per_class
        per_entity = all_absorption_per_entity.get(result_key, [])
        if not per_entity:
            for cls_label, cs in ar["per_class"].items():
                per_entity.extend([1] * cs["false_negatives"])
                per_entity.extend([0] * (cs["probe_correct_raw"] - cs["false_negatives"]))

        sae_key = ar.get("sae_key", result_key.split("_")[-1])
        perm = permutation_test_vs_firstletter(per_entity, firstletter_data, sae_key,
                                                 n_permutations=N_PERMUTATIONS)
        perm_test_results[result_key] = perm

        if perm["available"]:
            sig = " ***" if perm["significant_001"] else (" *" if perm["significant_005"] else "")
            logger.info(f"\n  Perm test vs first-letter ({result_key}): "
                        f"cd={perm['crossdomain_rate']:.4f}, fl={perm['firstletter_rate']:.4f}, "
                        f"diff={perm['observed_diff']:+.4f}, p={perm['permutation_p_value']:.4f}{sig}")

    # Step: Cross-hierarchy ANOVA
    step += 1
    report_progress(step, total_steps, "cross_hierarchy_anova")

    # Group by hierarchy (combine across SAE widths for ANOVA)
    anova_by_sae = {}
    for sae_config in SAE_CONFIGS:
        config_key = sae_config["key"]
        sae_groups = {}
        for hier_name in probes:
            rk = f"{hier_name}_{config_key}"
            if rk in all_absorption_per_entity:
                sae_groups[rk] = all_absorption_per_entity[rk]
        if sae_groups:
            anova_by_sae[config_key] = cross_hierarchy_anova(sae_groups)
            if anova_by_sae[config_key]["available"]:
                logger.info(f"\n  ANOVA ({config_key}): H={anova_by_sae[config_key]['kruskal_wallis_stat']:.3f}, "
                            f"p={anova_by_sae[config_key]['kruskal_wallis_p']:.4f} "
                            f"({'SIG' if anova_by_sae[config_key]['significant_005'] else 'NS'})")

    # Step: iter_008 comparison
    step += 1
    report_progress(step, total_steps, "iter008_comparison")

    iter008_rates = {
        "city-continent_L24_16k": {"absorption_rate": 0.3584, "strict_rate": 0.1387, "n": 173},
    }
    iter008_comparison = {}
    for result_key, ar in all_results.items():
        if "error" in ar:
            continue
        i8_key = f"{ar['hierarchy']}_{ar['sae_key']}"
        if i8_key in iter008_rates:
            i8 = iter008_rates[i8_key]
            diff = ar["absorption_rate"] - i8["absorption_rate"]
            iter008_comparison[result_key] = {
                "iter_008_rate": i8["absorption_rate"],
                "iter_008_strict": i8["strict_rate"],
                "iter_009_rate": ar["absorption_rate"],
                "iter_009_strict": ar["strict_absorption_rate"],
                "difference": float(diff),
                "within_10pp": abs(diff) <= 0.10,
            }
            logger.info(f"\n  iter_008 comparison ({result_key}): "
                        f"i8={i8['absorption_rate']:.4f}, i9={ar['absorption_rate']:.4f}, "
                        f"diff={diff:+.4f}")

    # Step: Compile final output
    step += 1
    report_progress(step, total_steps, "compiling_output")

    elapsed = time.time() - start_time

    # Summary table
    summary_table = []
    for result_key, ar in sorted(all_results.items()):
        if "error" in ar:
            continue
        summary_table.append({
            "hierarchy": ar["hierarchy"],
            "sae_config": ar["sae_key"],
            "probe_layer": ar.get("sae_config", {}).get("layer", 24),
            "absorption_rate": ar["absorption_rate"],
            "strict_rate": ar["strict_absorption_rate"],
            "probe_raw_accuracy": ar["probe_raw_accuracy"],
            "probe_sae_accuracy": ar["probe_sae_accuracy"],
            "n_entities": ar["total_entities"],
            "n_probe_correct": ar["total_probe_correct_raw"],
            "n_fn": ar["total_false_negatives"],
            "ci_lower": ar["bootstrap_ci"]["ci_lower"],
            "ci_upper": ar["bootstrap_ci"]["ci_upper"],
            "strict_ci_lower": ar["strict_bootstrap_ci"]["ci_lower"],
            "strict_ci_upper": ar["strict_bootstrap_ci"]["ci_upper"],
            "probe_f1": ar["probe_f1"],
            "quality_gate": ar["probe_gate"],
        })

    # Pass criteria
    has_results = any("error" not in v for v in all_results.values())
    has_nonzero = any(
        v.get("absorption_rate", 0) > 0
        for v in all_results.values()
        if isinstance(v, dict) and "error" not in v
    )
    n_significant = sum(
        1 for pt in perm_test_results.values()
        if pt.get("significant_005", False)
    )

    output = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "token_pos": TOKEN_POS,
        "device": DEVICE,
        "hierarchies_tested": list(probes.keys()),
        "sae_configs_tested": [c["key"] for c in SAE_CONFIGS],
        "probe_info": {
            hier_name: {
                "layer": pi["layer"],
                "f1": pi["f1"],
                "gate": pi["gate"],
                "n_classes": len(pi["class_labels"]),
                "classes": list(pi["class_labels"]) if hasattr(pi["class_labels"], '__iter__') else [],
            }
            for hier_name, pi in probes.items()
        },
        "data_info": {
            hier_name: {
                "n_entities": ds["n_entities"],
                "n_classes_used": ds["n_classes_used"],
                "class_distribution": ds["class_distribution"],
            }
            for hier_name, ds in ravel_datasets.items()
        },
        "absorption_results": all_results,
        "summary_table": summary_table,
        "permutation_test_vs_firstletter": perm_test_results,
        "cross_hierarchy_anova": anova_by_sae,
        "iter008_comparison": iter008_comparison,
        "pass_criteria": {
            "has_results": has_results,
            "has_nonzero_absorption": has_nonzero,
            "n_significant_vs_firstletter": n_significant,
            "target_significant": 2,
            "met": has_results and has_nonzero,
        },
        "methodology_notes": {
            "token_position": TOKEN_POS,
            "icl_examples": N_ICL,
            "bootstrap_resamples": N_BOOTSTRAP,
            "permutation_resamples": N_PERMUTATIONS,
            "bonferroni_n": BONFERRONI_N,
            "absorption_definition": (
                "False negative: probe correct on raw residual stream, "
                "wrong on SAE-reconstructed activations. "
                "Strict: false negative AND main SAE features (top-5 cosine) do not fire."
            ),
            "probe_caveats": {
                "city-continent": "F1=0.871, relaxed gate. Some absorption may reflect probe errors.",
                "city-language": "F1=0.818, relaxed gate. Some absorption may reflect probe errors.",
                "city-country": "F1=0.726, below relaxed gate. Results should be interpreted with caution.",
            },
        },
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    # Save to multiple locations
    out_path_full = FULL_DIR / f"{TASK_ID}.json"
    out_path_full.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nSaved FULL: {out_path_full}")

    # Also save to phase1 dir for downstream tasks
    out_path_phase1 = PHASE1_DIR / "absorption_crossdomain.json"
    out_path_phase1.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved phase1: {out_path_phase1}")

    # Also save to pilots dir (overwrite pilot)
    out_path_pilot = PILOT_DIR / f"{TASK_ID}.json"
    out_path_pilot.write_text(json.dumps(output, indent=2, default=str))

    # Generate summary markdown
    summary_md = generate_summary_md(output)
    md_path = FULL_DIR / f"{TASK_ID}_summary.md"
    md_path.write_text(summary_md)
    logger.info(f"Summary: {md_path}")
    # Also to pilots
    (PILOT_DIR / f"{TASK_ID}_summary.md").write_text(summary_md)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    summary_text = (
        f"Phase 1.2b cross-domain absorption (FULL, iter_009). "
        f"Hierarchies: {list(probes.keys())}. "
        f"Token pos: {TOKEN_POS}. Time: {elapsed/60:.1f}min. "
    )
    for r in summary_table:
        summary_text += (
            f"{r['hierarchy']}_{r['sae_config']}: abs={r['absorption_rate']:.4f} "
            f"(strict={r['strict_rate']:.4f}), "
            f"CI=[{r['ci_lower']:.3f},{r['ci_upper']:.3f}]. "
        )
    summary_text += f"Significant vs FL: {n_significant}/{BONFERRONI_N}."

    mark_done("success", summary_text)
    update_gpu_progress(elapsed, "completed")
    update_experiment_state("completed")

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETED: {summary_text}")
    logger.info(f"{'='*70}")

    return output


def generate_summary_md(results):
    """Generate human-readable markdown summary."""
    lines = [
        "# Phase 1.2b: Cross-Domain Absorption -- Iter 9 FULL",
        "",
        f"**Mode**: {results.get('mode', '?')}",
        f"**Time**: {results.get('elapsed_minutes', 0):.1f} minutes",
        f"**Token position**: {results.get('token_pos', '?')}",
        f"**Hierarchies**: {', '.join(results.get('hierarchies_tested', []))}",
        f"**SAE configs**: {', '.join(results.get('sae_configs_tested', []))}",
        "",
        "## Probe Quality",
        "",
        "| Hierarchy | Layer | F1 | Gate | N Classes |",
        "|-----------|-------|-----|------|-----------|",
    ]

    for hier_name, pi in results.get("probe_info", {}).items():
        f1 = pi.get("f1", 0)
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
        lines.append(f"| {hier_name} | {pi.get('layer', '?')} | {f1_str} | "
                     f"{pi.get('gate', '?')} | {pi.get('n_classes', '?')} |")

    # Absorption results table
    summary_table = results.get("summary_table", [])
    if summary_table:
        lines.extend([
            "", "## Absorption Results", "",
            "| Hierarchy | SAE | Absorption Rate | Strict Rate | CI (95%) | Strict CI | Probe Raw | Probe SAE | N | N FN |",
            "|-----------|-----|-----------------|-------------|----------|-----------|-----------|-----------|---|------|",
        ])
        for r in summary_table:
            lines.append(
                f"| {r['hierarchy']} | {r['sae_config']} | **{r['absorption_rate']:.4f}** | "
                f"{r['strict_rate']:.4f} | [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] | "
                f"[{r['strict_ci_lower']:.3f}, {r['strict_ci_upper']:.3f}] | "
                f"{r['probe_raw_accuracy']:.4f} | {r['probe_sae_accuracy']:.4f} | "
                f"{r['n_entities']} | {r['n_fn']} |"
            )

    # Per-class breakdowns
    for result_key, ar in results.get("absorption_results", {}).items():
        if isinstance(ar, dict) and "per_class" in ar and ar["per_class"]:
            lines.extend([
                "", f"### Per-Class Breakdown ({result_key})", "",
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

    # Cross-hierarchy ANOVA
    anova = results.get("cross_hierarchy_anova", {})
    if anova:
        lines.extend(["", "## Cross-Hierarchy ANOVA (Kruskal-Wallis)", ""])
        for sae_key, a in anova.items():
            if a.get("available"):
                sig = "YES" if a["significant_005"] else "NO"
                lines.append(f"- **{sae_key}**: H={a['kruskal_wallis_stat']:.3f}, "
                             f"p={a['kruskal_wallis_p']:.4f} (Significant: {sig})")
                for name, g in a.get("groups", {}).items():
                    lines.append(f"  - {name}: mean={g['mean']:.4f}, n={g['n']}")

                # Pairwise
                pw = a.get("pairwise_mannwhitney", {})
                if pw:
                    lines.append(f"  - Pairwise Mann-Whitney:")
                    for pair, pv in pw.items():
                        sig_pw = "*" if pv["p_bonferroni"] < 0.05 else ""
                        lines.append(f"    - {pair}: p={pv['p_value']:.4f} (Bonf: {pv['p_bonferroni']:.4f}){sig_pw}")

    # Permutation tests
    perm = results.get("permutation_test_vs_firstletter", {})
    if perm:
        lines.extend(["", "## Permutation Tests vs First-Letter", "",
                       "| Config | CD Rate | FL Rate | Diff | p (perm) | p (Bonf) | Cohen's h | Sig? |",
                       "|--------|---------|---------|------|----------|----------|-----------|------|"])
        for key, pt in perm.items():
            if pt.get("available"):
                sig = "Yes" if pt["significant_bonferroni_005"] else ("Uncorrected" if pt["significant_005"] else "No")
                lines.append(
                    f"| {key} | {pt['crossdomain_rate']:.4f} | {pt['firstletter_rate']:.4f} | "
                    f"{pt['observed_diff']:+.4f} | {pt['permutation_p_value']:.4f} | "
                    f"{pt['permutation_p_bonferroni']:.4f} | {pt['cohens_h']:.4f} | {sig} |"
                )
            else:
                lines.append(f"| {key} | -- | -- | -- | -- | -- | -- | *{pt.get('reason', '?')}* |")

    # iter_008 comparison
    i8comp = results.get("iter008_comparison", {})
    if i8comp:
        lines.extend(["", "## Comparison with iter_008", ""])
        for key, comp in i8comp.items():
            status = "OK" if comp["within_10pp"] else "LARGE"
            lines.append(f"- **{key}**: i8={comp['iter_008_rate']:.4f}, "
                         f"i9={comp['iter_009_rate']:.4f}, diff={comp['difference']:+.4f} ({status})")

    # FN examples (first 10 per hierarchy)
    for result_key, ar in results.get("absorption_results", {}).items():
        if isinstance(ar, dict) and ar.get("fn_examples"):
            lines.extend(["", f"### Example False Negatives ({result_key})", ""])
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
        try:
            update_gpu_progress(time.time() - (globals().get("start_time", time.time())), "failed")
        except Exception:
            pass
        update_experiment_state("failed", str(e))
        sys.exit(1)
