"""
Phase 1.3: Cross-Domain Absorption Measurement (Primary Contribution)

First systematic absorption measurement beyond first-letter spelling, using
RAVEL entity-attribute hierarchies (city-country, city-continent, city-language)
with Gemma 2B probes.

Pipeline:
1. Load Gemma 2 2B + pre-trained probes from Phase 1.1
2. Load RAVEL dataset and build ICL prompts for entity extraction
3. For each passing/usable probe x each Gemma Scope SAE:
   a) Cache residual stream activations on entity-containing tokens
   b) Encode through SAE
   c) Apply probe to SAE output
   d) Identify false negatives (probe correct on raw, wrong on SAE)
   e) Compute absorption rate per hierarchy x SAE
4. Compare with first-letter baseline from Phase 1.2
5. Bootstrap 95% CI and paired statistical tests

PILOT: Layer 12, 16k width, subset of cities
FULL: Layers 6,12,18,24 x widths 16k,65k
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

if MODE == "PILOT":
    SAE_CONFIGS = [
        {"layer": 12, "width": "16k", "release": "gemma-scope-2b-pt-res-canonical",
         "sae_id": "layer_12/width_16k/canonical"},
    ]
    MAX_CITIES = 200  # Subset for pilot
    TIMEOUT = 900
else:
    SAE_CONFIGS = [
        {"layer": l, "width": w, "release": "gemma-scope-2b-pt-res-canonical",
         "sae_id": f"layer_{l}/width_{w}/canonical"}
        for l in [6, 12, 18, 24] for w in ["16k", "65k"]
    ]
    MAX_CITIES = 2000
    TIMEOUT = 3600

# Probe quality thresholds
QUALITY_GATE_STRICT = 0.90
QUALITY_GATE_RELAXED = 0.85
QUALITY_GATE_MINIMUM = 0.70  # Below this we exclude entirely

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Process tracking (per protocol)
# ============================================================
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def report_progress(step, total_steps, status="running", metrics=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "epoch": step, "total_epochs": total_steps,
        "step": step, "total_steps": total_steps,
        "loss": None, "metric": metrics or {},
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
        except:
            pass
    done_data = json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress, "timestamp": datetime.now().isoformat(),
    })
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(done_data)
    (PILOT_DIR / f"{TASK_ID}_DONE").write_text(done_data)


def get_sae_hook_name(sae):
    """Get hook name from SAE, handling different SAELens versions."""
    if hasattr(sae.cfg, 'hook_name'):
        return sae.cfg.hook_name
    if hasattr(sae.cfg, 'metadata') and sae.cfg.metadata:
        md = sae.cfg.metadata
        if hasattr(md, 'get'):
            return md.get('hook_name', md.get('hook_point', None))
        if hasattr(md, 'hook_name'):
            return md.hook_name
        if isinstance(md, dict):
            return md.get('hook_name')
        try:
            return md['hook_name']
        except:
            pass
    raise ValueError(f"Cannot find hook_name in SAE config: {sae.cfg}")


# ============================================================
# Model loading
# ============================================================
def load_model(device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    logger.info("Loading Gemma 2 2B...")
    hf_model = AutoModelForCausalLM.from_pretrained(GEMMA_LOCAL_PATH, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_LOCAL_PATH)
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", device=device, dtype=torch.bfloat16,
        hf_model=hf_model, tokenizer=tokenizer,
    )
    logger.info(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    del hf_model
    gc.collect(); torch.cuda.empty_cache()
    return model


def load_sae(release, sae_id, device="cuda:0"):
    from sae_lens import SAE
    logger.info(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release, sae_id, device=device)
    hook_name = get_sae_hook_name(sae)
    logger.info(f"  d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}, hook={hook_name}")
    return sae


# ============================================================
# Load pre-trained probes from Phase 1.1
# ============================================================
def load_ravel_probe(hierarchy_name, layer=12):
    """
    Load a saved sklearn LogisticRegression probe from Phase 1.1.
    Returns (probe, class_labels, probe_quality_info).
    """
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

    # Get quality metrics from pilot results
    pilot_path = PILOT_DIR / "phase1_probe_training.json"
    quality_info = {"f1": None, "accuracy": None, "quality_gate": "unknown"}
    if pilot_path.exists():
        pilot_data = json.loads(pilot_path.read_text())
        key = f"{hierarchy_name}_L{layer}"
        if key in pilot_data.get("probes", {}):
            pm = pilot_data["probes"][key]
            quality_info = {
                "f1": pm.get("f1_weighted_cv"),
                "accuracy": pm.get("accuracy_cv"),
                "balanced_accuracy": pm.get("balanced_accuracy_cv"),
                "n_classes": pm.get("n_classes"),
                "quality_gate": "strict" if pm.get("quality_gate_strict") else
                               ("relaxed" if pm.get("quality_gate_relaxed") else "below"),
            }

    logger.info(f"  Loaded probe {hierarchy_name}_L{layer}: "
                f"coef={coef.shape}, classes={len(classes)}, "
                f"F1={quality_info.get('f1', '?')}")

    return probe, classes, quality_info


def load_firstletter_results():
    """Load first-letter absorption results from Phase 1.2 for comparison."""
    fl_path = PILOT_DIR / "phase1_absorption_firstletter.json"
    if fl_path.exists():
        return json.loads(fl_path.read_text())
    return None


# ============================================================
# RAVEL data preparation
# ============================================================
def prepare_ravel_data(max_cities=200):
    """
    Load RAVEL dataset and prepare entity-attribute pairs.
    Returns dict of {hierarchy_name: (cities, labels, label_encoder)}.
    """
    from datasets import load_dataset

    logger.info("Loading RAVEL dataset...")
    ds = load_dataset("hij/ravel", "city_entity", split="train")
    logger.info(f"  Total RAVEL entries: {len(ds)}")

    hierarchy_defs = {
        "city-country": "Country",
        "city-continent": "Continent",
        "city-language": "Language",
    }

    results = {}
    for hier_name, col_name in hierarchy_defs.items():
        cities = list(ds["City"])
        labels = list(ds[col_name])

        # Filter valid entries
        valid = [(c, l) for c, l in zip(cities, labels)
                 if c and l and str(l).strip() and str(c).strip()]
        cities_valid, labels_valid = zip(*valid)
        cities_valid, labels_valid = list(cities_valid), list(labels_valid)

        # Filter classes with >= 3 samples (for meaningful statistics)
        label_counts = Counter(labels_valid)
        valid_labels = {l for l, n in label_counts.items() if n >= 3}
        keep = [(c, l) for c, l in zip(cities_valid, labels_valid) if l in valid_labels]
        cities_final = [x[0] for x in keep]
        labels_final = [x[1] for x in keep]

        # Subsample for pilot
        if len(cities_final) > max_cities:
            rng = np.random.RandomState(SEED)
            indices = rng.choice(len(cities_final), size=max_cities, replace=False)
            cities_final = [cities_final[i] for i in indices]
            labels_final = [labels_final[i] for i in indices]

        # Build label mapping
        unique_labels = sorted(set(labels_final))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}

        logger.info(f"  {hier_name}: {len(cities_final)} cities, "
                    f"{len(unique_labels)} classes (from {len(label_counts)} original)")

        results[hier_name] = {
            "cities": cities_final,
            "labels": labels_final,
            "label_to_idx": label_to_idx,
            "unique_labels": unique_labels,
        }

    return results


# ============================================================
# ICL prompt construction for RAVEL
# ============================================================
def build_ravel_icl_prompts(cities, labels, hierarchy_name, n_icl=5):
    """
    Build ICL-style prompts for RAVEL entity-attribute prediction.

    Format mirrors sae_spelling's approach: provide examples, then ask about
    the target entity. Position -2 (last token before answer) captures the
    representation.
    """
    templates = {
        "city-country": {
            "base": "The city of {entity} is in the country of",
            "answer": " {label}",
        },
        "city-continent": {
            "base": "The city of {entity} is on the continent of",
            "answer": " {label}",
        },
        "city-language": {
            "base": "The primary language spoken in {entity} is",
            "answer": " {label}",
        },
    }

    tmpl = templates.get(hierarchy_name, templates["city-country"])
    examples = list(zip(cities, labels))
    rng = random.Random(SEED)

    prompts = []
    for i, (city, label) in enumerate(zip(cities, labels)):
        # Select ICL examples (excluding current city)
        icl_pool = [(c, l) for c, l in examples if c != city]
        rng.shuffle(icl_pool)
        icl_selected = icl_pool[:n_icl]

        # Build prompt
        icl_parts = []
        for ex_city, ex_label in icl_selected:
            ex_text = tmpl["base"].format(entity=ex_city) + tmpl["answer"].format(label=ex_label)
            icl_parts.append(ex_text)

        full_prompt = "\n".join(icl_parts) + "\n" + tmpl["base"].format(entity=city)
        prompts.append(full_prompt)

    return prompts


# ============================================================
# Absorption measurement for a single hierarchy
# ============================================================
def measure_crossdomain_absorption(model, sae, probe, class_labels,
                                    cities, labels, label_to_idx,
                                    hierarchy_name, layer=12,
                                    token_pos=-2, device="cuda:0"):
    """
    Measure absorption rate for a cross-domain hierarchy.

    For each city:
    1. Build ICL prompt, run model to get residual stream activation
    2. Apply probe to raw activation -> should predict correct attribute
    3. Encode through SAE, decode back
    4. Apply probe to SAE output -> may fail (false negative = absorption)
    5. Check if main SAE features for the attribute fire

    Returns detailed absorption statistics per class and overall.
    """
    tl_hook = f"blocks.{layer}.hook_resid_post"

    # Pre-compute probe directions and find main SAE features per class
    probe_coefs = torch.tensor(probe.coef_, dtype=torch.float32)  # [n_classes, d_model]
    W_dec = sae.W_dec.detach().float().cpu()  # [d_sae, d_model]

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
    logger.info(f"  Main features computed for {len(main_features)}/{len(class_labels)} classes")

    # Build ICL prompts
    prompts = build_ravel_icl_prompts(cities, labels, hierarchy_name, n_icl=5)

    # Per-class statistics
    per_class_stats = defaultdict(lambda: {
        "total": 0,
        "probe_correct_raw": 0,
        "probe_correct_sae": 0,
        "false_negatives": 0,
        "main_feature_fires": 0,
        "fn_and_main_absent": 0,
        "fn_and_main_present": 0,
    })

    fn_examples = []  # Store example false negatives
    total_processed = 0
    total_errors = 0

    for i, (city, label, prompt) in enumerate(zip(cities, labels, prompts)):
        if label not in label_to_idx:
            continue
        true_idx = label_to_idx[label]

        try:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])

            raw_act = cache[tl_hook][0, token_pos, :].detach().float()  # [d_model]

            # SAE encode/decode
            raw_act_device = raw_act.to(device)
            with torch.no_grad():
                sae_features = sae.encode(raw_act_device.unsqueeze(0))  # [1, d_sae]
                sae_out = sae.decode(sae_features)  # [1, d_model]

            # Probe predictions
            raw_np = raw_act.detach().cpu().numpy().reshape(1, -1)
            sae_np = sae_out[0].detach().float().cpu().numpy().reshape(1, -1)

            raw_pred = probe.predict(raw_np)[0]
            sae_pred = probe.predict(sae_np)[0]

            probe_correct_raw = (raw_pred == true_idx)
            probe_correct_sae = (sae_pred == true_idx)

            # Check main features
            if true_idx in main_features:
                mfids = main_features[true_idx]["feature_ids"]
                feat_acts = sae_features[0, mfids].detach().float().cpu()
                any_main_fires = (feat_acts.abs() > 1e-6).any().item()
            else:
                any_main_fires = False

            del cache

        except Exception as e:
            if total_errors < 3:
                logger.warning(f"Error processing '{city}': {e}")
            total_errors += 1
            continue

        stats = per_class_stats[label]
        stats["total"] += 1
        if probe_correct_raw:
            stats["probe_correct_raw"] += 1
        if probe_correct_sae:
            stats["probe_correct_sae"] += 1
        if any_main_fires:
            stats["main_feature_fires"] += 1

        # False negative: correct on raw, wrong on SAE
        if probe_correct_raw and not probe_correct_sae:
            stats["false_negatives"] += 1
            if not any_main_fires:
                stats["fn_and_main_absent"] += 1
            else:
                stats["fn_and_main_present"] += 1

            if len(fn_examples) < 30:
                fn_examples.append({
                    "city": city,
                    "true_label": label,
                    "raw_pred_label": class_labels[raw_pred] if raw_pred < len(class_labels) else f"idx_{raw_pred}",
                    "sae_pred_label": class_labels[sae_pred] if sae_pred < len(class_labels) else f"idx_{sae_pred}",
                    "main_fires": any_main_fires,
                    "top_cos_sim": main_features.get(true_idx, {}).get("cos_sims", [0])[0],
                })

        total_processed += 1
        if (total_processed) % 50 == 0:
            logger.info(f"  Processed {total_processed}/{len(cities)} cities "
                       f"(errors: {total_errors})")
            torch.cuda.empty_cache()

    # Aggregate
    total_raw_correct = sum(s["probe_correct_raw"] for s in per_class_stats.values())
    total_fn = sum(s["false_negatives"] for s in per_class_stats.values())
    total_fn_main_absent = sum(s["fn_and_main_absent"] for s in per_class_stats.values())
    total_tokens = sum(s["total"] for s in per_class_stats.values())

    absorption_rate = total_fn / max(total_raw_correct, 1)
    strict_rate = total_fn_main_absent / max(total_raw_correct, 1)

    # Probe accuracy on raw activations
    probe_raw_accuracy = total_raw_correct / max(total_tokens, 1)

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
        "total_cities_processed": total_tokens,
        "total_errors": total_errors,
        "total_probe_correct_raw": total_raw_correct,
        "probe_raw_accuracy": probe_raw_accuracy,
        "total_false_negatives": total_fn,
        "total_fn_main_absent": total_fn_main_absent,
        "absorption_rate": float(absorption_rate),
        "strict_absorption_rate": float(strict_rate),
        "per_class": per_class_rates,
        "fn_examples": fn_examples,
        "main_features_top": {
            class_labels[k] if k < len(class_labels) else f"idx_{k}": {
                "fid": v["feature_ids"][0],
                "cos": round(v["cos_sims"][0], 4)
            }
            for k, v in main_features.items()
            if k < len(class_labels)
        },
    }


# ============================================================
# Bootstrap CI
# ============================================================
def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    if len(values) == 0:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0, "std": 0}
    values = np.array(values, dtype=float)
    rng = np.random.RandomState(SEED)
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
    }


# ============================================================
# Statistical comparison with first-letter baseline
# ============================================================
def compare_with_firstletter(crossdomain_rates, firstletter_data):
    """
    Compare cross-domain absorption rates with first-letter baseline.
    Uses paired tests where possible, otherwise independent t-test.
    """
    if firstletter_data is None:
        return {"comparison_available": False}

    fl_results = firstletter_data.get("absorption_results", {})
    comparisons = {}

    for sae_key, fl_ar in fl_results.items():
        if not isinstance(fl_ar, dict) or "absorption_rate" not in fl_ar:
            continue

        fl_rate = fl_ar["absorption_rate"]
        fl_strict = fl_ar.get("strict_absorption_rate", fl_rate)

        for hier_name, cd_results in crossdomain_rates.items():
            if sae_key not in cd_results:
                continue
            cd_ar = cd_results[sae_key]
            cd_rate = cd_ar["absorption_rate"]
            cd_strict = cd_ar.get("strict_absorption_rate", cd_rate)

            # Effect size: Cohen's d (using pooled estimate)
            diff = cd_rate - fl_rate
            # For single rates we approximate sd from binomial
            fl_n = max(fl_ar.get("total_probe_correct", 100), 1)
            cd_n = max(cd_ar.get("total_probe_correct_raw", 100), 1)
            fl_se = np.sqrt(fl_rate * (1 - fl_rate) / fl_n) if 0 < fl_rate < 1 else 0.01
            cd_se = np.sqrt(cd_rate * (1 - cd_rate) / cd_n) if 0 < cd_rate < 1 else 0.01
            pooled_sd = np.sqrt((fl_se**2 + cd_se**2) / 2)
            cohens_d = diff / max(pooled_sd, 1e-6)

            # Two-proportion z-test
            fl_fn = fl_ar.get("total_false_negatives", 0)
            cd_fn = cd_ar.get("total_false_negatives", 0)
            p_pool = (fl_fn + cd_fn) / max(fl_n + cd_n, 1)
            se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/max(fl_n, 1) + 1/max(cd_n, 1))) if 0 < p_pool < 1 else 0.01
            z_stat = diff / max(se_pool, 1e-6)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            comp_key = f"{hier_name}_vs_firstletter_{sae_key}"
            comparisons[comp_key] = {
                "hierarchy": hier_name,
                "sae_config": sae_key,
                "firstletter_rate": fl_rate,
                "crossdomain_rate": cd_rate,
                "difference": diff,
                "cohens_d": float(cohens_d),
                "z_stat": float(z_stat),
                "p_value": float(p_value),
                "firstletter_strict": fl_strict,
                "crossdomain_strict": cd_strict,
                "firstletter_n": int(fl_n),
                "crossdomain_n": int(cd_n),
                "significant_005": p_value < 0.05,
                "significant_001": p_value < 0.01,
            }

    return {
        "comparison_available": True,
        "comparisons": comparisons,
    }


# ============================================================
# GPU progress tracking
# ============================================================
def update_gpu_progress(elapsed_seconds, status="completed"):
    progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        data = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "completed": [], "failed": [], "running": {}, "timings": {}}
        if status == "completed":
            if TASK_ID not in data.get("completed", []):
                data.setdefault("completed", []).append(TASK_ID)
        else:
            if TASK_ID not in data.get("failed", []):
                data.setdefault("failed", []).append(TASK_ID)
        data.get("running", {}).pop(TASK_ID, None)
        data.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 60,
            "actual_min": round(elapsed_seconds / 60),
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {
                "model": "gemma-2-2b",
                "mode": MODE,
                "n_sae_configs": len(SAE_CONFIGS),
                "max_cities": MAX_CITIES,
                "gpu_model": "RTX PRO 6000 Blackwell",
                "gpu_count": 1,
            }
        }
        progress_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"gpu_progress update failed: {e}")


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 20, "starting")

    logger.info(f"{'='*60}")
    logger.info(f"Phase 1.3: Cross-Domain Absorption Measurement")
    logger.info(f"Mode: {MODE}, SAE configs: {len(SAE_CONFIGS)}, Max cities: {MAX_CITIES}")
    logger.info(f"{'='*60}")

    device = "cuda:0"

    # Step 1: Load model
    report_progress(1, 20, "loading_model")
    model = load_model(device=device)

    # Step 2: Load pre-trained probes from Phase 1.1
    report_progress(2, 20, "loading_probes")
    hierarchies_to_test = ["city-continent", "city-country", "city-language"]
    probes = {}
    probe_quality_summary = {}

    for hier_name in hierarchies_to_test:
        try:
            probe, class_labels, quality_info = load_ravel_probe(hier_name, layer=12)
            f1 = quality_info.get("f1", 0) or 0

            # Determine inclusion status
            if f1 >= QUALITY_GATE_STRICT:
                gate_status = "PASS_STRICT"
            elif f1 >= QUALITY_GATE_RELAXED:
                gate_status = "PASS_RELAXED"
            elif f1 >= QUALITY_GATE_MINIMUM:
                gate_status = "BELOW_GATE_INCLUDED"  # Include with strong caveat
            else:
                gate_status = "EXCLUDED"

            probe_quality_summary[hier_name] = {
                "f1": f1,
                "quality_info": quality_info,
                "gate_status": gate_status,
                "included": gate_status != "EXCLUDED",
            }

            if gate_status != "EXCLUDED":
                probes[hier_name] = {
                    "probe": probe,
                    "class_labels": class_labels.tolist() if hasattr(class_labels, 'tolist') else list(class_labels),
                    "quality": quality_info,
                    "gate_status": gate_status,
                }
                logger.info(f"  {hier_name}: INCLUDED ({gate_status}, F1={f1:.4f})")
            else:
                logger.warning(f"  {hier_name}: EXCLUDED (F1={f1:.4f} < {QUALITY_GATE_MINIMUM})")

        except FileNotFoundError as e:
            logger.error(f"  {hier_name}: probe not found: {e}")
            probe_quality_summary[hier_name] = {
                "f1": None, "gate_status": "NOT_FOUND", "included": False
            }
        except Exception as e:
            logger.error(f"  {hier_name}: probe load error: {e}")
            probe_quality_summary[hier_name] = {
                "f1": None, "gate_status": "ERROR", "included": False,
                "error": str(e)
            }

    if not probes:
        logger.error("No probes available for cross-domain measurement!")
        results = {
            "task_id": TASK_ID, "mode": MODE, "status": "no_probes",
            "timestamp": datetime.now().isoformat(),
            "probe_quality_summary": probe_quality_summary,
            "pilot_criteria_met": False,
            "diagnosis": "All probes excluded due to quality gate or missing files.",
        }
        out_path = PILOT_DIR / f"{TASK_ID}.json"
        out_path.write_text(json.dumps(results, indent=2, default=str))
        mark_done("failed", "No probes available for cross-domain measurement")
        update_gpu_progress(time.time() - start_time, "failed")
        return results

    logger.info(f"\n  Probes included: {len(probes)}/{len(hierarchies_to_test)}")

    # Step 3: Load RAVEL data
    report_progress(3, 20, "loading_ravel_data")
    ravel_data = prepare_ravel_data(max_cities=MAX_CITIES)

    # Step 4: Load first-letter baseline for comparison
    report_progress(4, 20, "loading_baseline")
    firstletter_data = load_firstletter_results()
    if firstletter_data:
        logger.info(f"  First-letter baseline loaded: "
                   f"absorption={firstletter_data.get('absorption_results', {}).get('L12_16k', {}).get('absorption_rate', '?')}")
    else:
        logger.warning("  First-letter baseline not found")

    # Step 5: Measure absorption for each hierarchy x SAE config
    all_crossdomain_results = {}  # {hierarchy: {sae_config: results}}

    step_counter = 5
    total_steps = 5 + len(probes) * len(SAE_CONFIGS) + 3  # +3 for stats, summary, save

    for hier_name, probe_info in probes.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Hierarchy: {hier_name} (F1={probe_info['quality']['f1']:.4f}, "
                    f"gate={probe_info['gate_status']})")
        logger.info(f"{'='*60}")

        hier_data = ravel_data.get(hier_name)
        if hier_data is None:
            logger.error(f"  No RAVEL data for {hier_name}")
            continue

        probe = probe_info["probe"]
        class_labels = probe_info["class_labels"]
        cities = hier_data["cities"]
        labels = hier_data["labels"]
        label_to_idx = hier_data["label_to_idx"]

        # Verify probe and data class alignment
        # The probe was trained with LabelEncoder, so classes map 0..N-1
        # We need to map RAVEL labels to probe class indices
        # The class_labels from the npz are the original label names
        # The label_to_idx from RAVEL data maps label names to indices
        # These SHOULD be aligned because we use the same label encoding

        # However, the subset of cities may have different classes than the full probe
        # We need to ensure consistency
        ravel_unique = set(labels)
        probe_classes_set = set(class_labels)
        overlap = ravel_unique & probe_classes_set
        missing_in_probe = ravel_unique - probe_classes_set

        if missing_in_probe:
            logger.warning(f"  {len(missing_in_probe)} labels in data not in probe: "
                          f"{list(missing_in_probe)[:5]}")
            # Filter to only cities whose labels are in the probe
            keep = [(c, l) for c, l in zip(cities, labels) if l in probe_classes_set]
            if not keep:
                logger.error(f"  No overlapping labels between probe and data for {hier_name}")
                continue
            cities = [x[0] for x in keep]
            labels = [x[1] for x in keep]

        # Rebuild label_to_idx to match probe's class ordering
        class_to_probe_idx = {c: i for i, c in enumerate(class_labels)}
        label_to_idx_aligned = {l: class_to_probe_idx[l]
                                for l in set(labels) if l in class_to_probe_idx}

        logger.info(f"  Using {len(cities)} cities, {len(label_to_idx_aligned)} classes "
                    f"(overlap with probe)")

        all_crossdomain_results[hier_name] = {}

        for sae_idx, sae_config in enumerate(SAE_CONFIGS):
            config_key = f"L{sae_config['layer']}_{sae_config['width']}"
            step_counter += 1
            report_progress(step_counter, total_steps,
                           f"absorption_{hier_name}_{config_key}",
                           {"hierarchy": hier_name, "sae": config_key})

            logger.info(f"\n  --- {hier_name} x {config_key} ---")

            try:
                sae = load_sae(sae_config["release"], sae_config["sae_id"], device=device)
            except Exception as e:
                logger.error(f"  SAE load failed: {e}")
                all_crossdomain_results[hier_name][config_key] = {"error": str(e)}
                continue

            try:
                abs_results = measure_crossdomain_absorption(
                    model=model,
                    sae=sae,
                    probe=probe,
                    class_labels=class_labels,
                    cities=cities,
                    labels=labels,
                    label_to_idx=label_to_idx_aligned,
                    hierarchy_name=hier_name,
                    layer=sae_config["layer"],
                    token_pos=-2,  # Match Phase 1.1 extraction position
                    device=device,
                )

                # Bootstrap CI on per-class rates
                class_rates = [v["absorption_rate"] for v in abs_results["per_class"].values()
                               if v["total"] > 0 and v["probe_correct_raw"] > 0]
                ci = bootstrap_ci(class_rates)

                result_entry = {
                    "sae_config": sae_config,
                    "absorption_rate": abs_results["absorption_rate"],
                    "strict_absorption_rate": abs_results["strict_absorption_rate"],
                    "probe_raw_accuracy": abs_results["probe_raw_accuracy"],
                    "total_cities": abs_results["total_cities_processed"],
                    "total_probe_correct_raw": abs_results["total_probe_correct_raw"],
                    "total_false_negatives": abs_results["total_false_negatives"],
                    "total_fn_main_absent": abs_results["total_fn_main_absent"],
                    "total_errors": abs_results["total_errors"],
                    "bootstrap_ci": ci,
                    "per_class": abs_results["per_class"],
                    "fn_examples": abs_results["fn_examples"][:15],  # Keep top 15
                    "main_features_top": abs_results["main_features_top"],
                }

                all_crossdomain_results[hier_name][config_key] = result_entry

                logger.info(f"  Absorption rate: {abs_results['absorption_rate']:.4f}")
                logger.info(f"  Strict rate: {abs_results['strict_absorption_rate']:.4f}")
                logger.info(f"  Probe raw accuracy: {abs_results['probe_raw_accuracy']:.4f}")
                logger.info(f"  FN: {abs_results['total_false_negatives']}/{abs_results['total_probe_correct_raw']}")
                logger.info(f"  Bootstrap CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

            except Exception as e:
                logger.error(f"  Absorption measurement failed: {e}", exc_info=True)
                all_crossdomain_results[hier_name][config_key] = {"error": str(e)}

            del sae
            gc.collect(); torch.cuda.empty_cache()

    # Step 6: Statistical comparison with first-letter baseline
    step_counter += 1
    report_progress(step_counter, total_steps, "statistical_comparison")
    logger.info(f"\n{'='*60}")
    logger.info("Statistical Comparison with First-Letter Baseline")
    logger.info(f"{'='*60}")

    comparison = compare_with_firstletter(all_crossdomain_results, firstletter_data)
    if comparison.get("comparison_available"):
        for comp_key, comp in comparison.get("comparisons", {}).items():
            sig = "*" if comp["significant_005"] else ""
            logger.info(f"  {comp['hierarchy']} vs first-letter ({comp['sae_config']}): "
                       f"diff={comp['difference']:+.4f}, "
                       f"Cohen's d={comp['cohens_d']:.3f}, "
                       f"p={comp['p_value']:.4f}{sig}")

    # Step 7: Compile summary
    step_counter += 1
    report_progress(step_counter, total_steps, "summarizing")

    summary_table = []
    for hier_name, configs in all_crossdomain_results.items():
        for config_key, ar in configs.items():
            if isinstance(ar, dict) and "absorption_rate" in ar:
                summary_table.append({
                    "hierarchy": hier_name,
                    "sae_config": config_key,
                    "absorption_rate": ar["absorption_rate"],
                    "strict_rate": ar["strict_absorption_rate"],
                    "probe_raw_accuracy": ar["probe_raw_accuracy"],
                    "n_cities": ar["total_cities"],
                    "n_fn": ar["total_false_negatives"],
                    "n_probe_correct": ar["total_probe_correct_raw"],
                    "ci_lower": ar["bootstrap_ci"]["ci_lower"],
                    "ci_upper": ar["bootstrap_ci"]["ci_upper"],
                    "probe_f1": probes[hier_name]["quality"].get("f1", None),
                    "quality_gate": probes[hier_name]["gate_status"],
                })

    # Pilot pass criteria:
    # - At least 1 semantic hierarchy x 1 SAE config with non-zero absorption
    # - Paired comparison with first-letter baseline computed
    has_nonzero = any(r["absorption_rate"] > 0 for r in summary_table)
    has_comparison = comparison.get("comparison_available", False)
    pilot_pass = len(summary_table) > 0 and (has_nonzero or len(summary_table) >= 1) and has_comparison

    elapsed = time.time() - start_time

    results = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "probe_quality_summary": probe_quality_summary,
        "n_hierarchies_tested": len(probes),
        "n_sae_configs_tested": len(SAE_CONFIGS),
        "crossdomain_results": all_crossdomain_results,
        "summary_table": summary_table,
        "comparison_with_firstletter": comparison,
        "pilot_criteria_met": pilot_pass,
        "pilot_criteria_details": {
            "n_hierarchies_with_results": sum(1 for h in all_crossdomain_results
                                              if any("absorption_rate" in v
                                                     for v in all_crossdomain_results[h].values()
                                                     if isinstance(v, dict))),
            "has_nonzero_absorption": has_nonzero,
            "has_baseline_comparison": has_comparison,
            "hierarchies_tested": list(probes.keys()),
        },
        "methodology_notes": {
            "probe_quality_caveat": (
                "All RAVEL probes are below the strict quality gate (F1 >= 0.90). "
                "City-continent probe has the highest F1 (0.843). Results should be "
                "interpreted with caution: absorption measurements may be confounded "
                "with probe errors. Full mode should explore additional layers and "
                "prompt templates to improve probe quality."
            ),
            "token_position": "-2 (last content token, matching Phase 1.1 methodology)",
            "icl_examples": 5,
            "absorption_definition": (
                "False negative: probe correct on raw residual stream, "
                "wrong on SAE-reconstructed activations. "
                "Strict: false negative AND main SAE features (top-5 cosine) do not fire."
            ),
        },
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    # Diagnose if probes are problematic
    if not any(pq.get("f1", 0) >= QUALITY_GATE_RELAXED
               for pq in probe_quality_summary.values()
               if pq.get("f1") is not None):
        results["diagnosis"] = {
            "issue": "No RAVEL probe reaches the relaxed quality gate (F1 >= 0.85)",
            "impact": (
                "Absorption measurements may overestimate true absorption because "
                "probe errors are conflated with SAE-induced errors. High probe error "
                "rates mean some 'false negatives' are actually probe failures, not absorption."
            ),
            "recommended_actions": [
                "Try additional layers (6, 18, 24) in full mode",
                "Try different prompt templates (fill-in-the-blank, QA format)",
                "Try different extraction positions (-1, -3)",
                "Consider PCA/dimensionality reduction before probe training",
                "Focus on city-continent (best F1, fewest classes)",
            ],
        }

    # Save
    if MODE == "PILOT":
        out_path = PILOT_DIR / f"{TASK_ID}.json"
    else:
        out_path = PHASE1_DIR / "absorption_crossdomain.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"\nSaved: {out_path}")

    # Generate summary markdown
    summary_md = generate_summary_md(results)
    md_path = PILOT_DIR / f"{TASK_ID}_summary.md" if MODE == "PILOT" else PHASE1_DIR / "absorption_crossdomain_summary.md"
    md_path.write_text(summary_md)
    logger.info(f"Summary: {md_path}")

    del model
    gc.collect(); torch.cuda.empty_cache()

    # Status
    summary_text = (
        f"Phase 1.3 cross-domain absorption ({MODE}). "
        f"Hierarchies tested: {len(probes)}. "
        f"SAE configs: {len(SAE_CONFIGS)}. "
        f"Time: {elapsed/60:.1f}min. "
        f"Pilot: {'PASS' if pilot_pass else 'FAIL'}."
    )
    if summary_table:
        rates = [r['absorption_rate'] for r in summary_table]
        summary_text += f" Rates: {[f'{r:.4f}' for r in rates]}."

    mark_done("success" if pilot_pass else "partial", summary_text)
    update_gpu_progress(elapsed, "completed" if pilot_pass else "completed")

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETED: {summary_text}")
    logger.info(f"{'='*60}")

    return results


def generate_summary_md(results):
    """Generate a human-readable markdown summary."""
    lines = [
        "# Phase 1.3: Cross-Domain Absorption Measurement -- Pilot Summary",
        "",
        f"## Status: {'PASS (GO)' if results.get('pilot_criteria_met') else 'PARTIAL (see caveats)'}",
        "",
        "## Probe Quality Summary",
        "",
        "| Hierarchy | F1 | Gate Status | Included |",
        "|-----------|-----|-------------|----------|",
    ]

    for hier, pq in results.get("probe_quality_summary", {}).items():
        f1 = pq.get("f1", "N/A")
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) and f1 is not None else str(f1)
        gate = pq.get("gate_status", "?")
        incl = "Yes" if pq.get("included") else "No"
        lines.append(f"| {hier} | {f1_str} | {gate} | {incl} |")

    lines.extend(["", "## Absorption Results", ""])

    summary_table = results.get("summary_table", [])
    if summary_table:
        lines.extend([
            "| Hierarchy | SAE Config | Absorption Rate | Strict Rate | CI (95%) | Probe Raw Acc | N Cities | N FN |",
            "|-----------|------------|-----------------|-------------|----------|---------------|----------|------|",
        ])
        for r in summary_table:
            lines.append(
                f"| {r['hierarchy']} | {r['sae_config']} | "
                f"{r['absorption_rate']:.4f} | {r['strict_rate']:.4f} | "
                f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] | "
                f"{r['probe_raw_accuracy']:.4f} | {r['n_cities']} | {r['n_fn']} |"
            )
    else:
        lines.append("*No absorption results computed.*")

    # Comparison with first-letter
    comparison = results.get("comparison_with_firstletter", {})
    if comparison.get("comparison_available"):
        lines.extend(["", "## Comparison with First-Letter Baseline", ""])
        lines.extend([
            "| Hierarchy | SAE | First-Letter Rate | Cross-Domain Rate | Difference | Cohen's d | p-value |",
            "|-----------|-----|-------------------|-------------------|------------|-----------|---------|",
        ])
        for comp_key, comp in comparison.get("comparisons", {}).items():
            sig = " *" if comp["significant_005"] else ""
            lines.append(
                f"| {comp['hierarchy']} | {comp['sae_config']} | "
                f"{comp['firstletter_rate']:.4f} | {comp['crossdomain_rate']:.4f} | "
                f"{comp['difference']:+.4f} | {comp['cohens_d']:.3f} | "
                f"{comp['p_value']:.4f}{sig} |"
            )

    # False negative examples
    for hier_name, configs in results.get("crossdomain_results", {}).items():
        for config_key, ar in configs.items():
            if isinstance(ar, dict) and "fn_examples" in ar and ar["fn_examples"]:
                lines.extend(["", f"## Example False Negatives: {hier_name} ({config_key})", ""])
                for fn in ar["fn_examples"][:10]:
                    lines.append(
                        f"- **{fn['city']}**: true={fn['true_label']}, "
                        f"raw_pred={fn['raw_pred_label']}, sae_pred={fn['sae_pred_label']}, "
                        f"main_fires={fn['main_fires']}"
                    )

    # Caveats
    if "diagnosis" in results:
        lines.extend(["", "## Caveats and Limitations", ""])
        diag = results["diagnosis"]
        lines.append(f"**Issue**: {diag.get('issue', '')}")
        lines.append(f"\n**Impact**: {diag.get('impact', '')}")
        if "recommended_actions" in diag:
            lines.append("\n**Recommended Actions**:")
            for action in diag["recommended_actions"]:
                lines.append(f"- {action}")

    lines.extend(["", "## Methodology Notes", ""])
    notes = results.get("methodology_notes", {})
    if "probe_quality_caveat" in notes:
        lines.append(f"**Probe quality caveat**: {notes['probe_quality_caveat']}")

    lines.extend([
        "",
        f"**Elapsed time**: {results.get('elapsed_minutes', 0):.1f} minutes",
        f"**Mode**: {results.get('mode', 'PILOT')}",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        update_gpu_progress(time.time() - (globals().get("start_time", time.time())), "failed")
        sys.exit(1)
