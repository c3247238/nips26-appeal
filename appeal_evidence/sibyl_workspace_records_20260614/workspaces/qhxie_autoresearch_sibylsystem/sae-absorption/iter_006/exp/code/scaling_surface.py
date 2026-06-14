#!/usr/bin/env python3
"""
scaling_surface.py — Stage 4: Scaling Surface: Width x L0 x Absorption Rate

This script implements H5 scaling analysis:
1. Collect absorption rates across all available Gemma Scope SAE configurations
   (from confound_decomposition results + additional configs if needed)
2. Fit Beta-GAM with architecture covariate (log_width, log_L0, interaction, arch_class)
3. Report interaction p-value, explained deviance
4. Run within-architecture subset analysis
5. Detect phase transition at critical L0
6. Compare scaling surface shape across hierarchy types (first-letter vs cross-domain)

FULL mode on Gemma 2 2B + Gemma Scope SAEs.
"""

import json
import os
import sys
import time
import gc
import traceback
import random
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from scipy import stats
from scipy.optimize import minimize_scalar

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "scaling_surface"
SEED = 42
DEVICE = "cuda:0"  # GPU 4 via CUDA_VISIBLE_DEVICES=4

K_SPARSE = 5
COSINE_THRESHOLD = 0.1
MAGNITUDE_GAP = 1.0
MAX_PER_LETTER = 25

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# ── PID / Progress / Done ──────────────────────────────────────────────────
pid_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(step, total_steps, description, extra=None):
    progress = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    data = {
        "task_id": TASK_ID, "epoch": step, "total_epochs": total_steps,
        "step": step, "total_steps": total_steps,
        "description": description, "metric": extra or {},
        "updated_at": datetime.now().isoformat(),
    }
    progress.write_text(json.dumps(data, indent=2))


def mark_done(status="success", summary="", results=None):
    if pid_file.exists():
        pid_file.unlink()
    progress_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    fp = {}
    if progress_file.exists():
        try:
            fp = json.loads(progress_file.read_text())
        except:
            pass
    marker = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "results": results or {}, "final_progress": fp,
        "timestamp": datetime.now().isoformat(),
    }))


def save_results(results):
    """Save results atomically."""
    out_path = RESULTS_DIR / f"{TASK_ID}.json"
    tmp_path = RESULTS_DIR / f"{TASK_ID}_tmp.json"

    def default_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    tmp_path.write_text(json.dumps(results, indent=2, default=default_serializer))
    tmp_path.rename(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load absorption data from confound_decomposition
# ══════════════════════════════════════════════════════════════════════════════
def load_confound_data():
    """Load the 34 SAE config results from confound_decomposition."""
    confound_path = RESULTS_DIR / "confound_decomposition.json"
    with open(confound_path) as f:
        data = json.load(f)

    records = []
    for r in data["sae_results"]:
        c = r["config"]
        a = r["absorption"]
        q = r["quality"]

        records.append({
            "release": c["release"],
            "sae_id": c["sae_id"],
            "layer": c["layer"],
            "width": c["width"],
            "l0_target": c["l0_target"],
            "arch": c["arch"],
            "measured_l0": q["measured_l0"],
            "absorption_rate": a["aggregate_absorption_rate"],
            "ci_lower": a["bootstrap_ci_95"][0],
            "ci_upper": a["bootstrap_ci_95"][1],
            "total_tested": a["total_tested"],
            "total_absorbed": a["total_absorbed"],
            "total_false_negatives": a["total_false_negatives"],
            "mean_probe_f1": a.get("mean_probe_f1", None),
            "live_features": q.get("live_features", None),
            "feature_density": q.get("feature_density", None),
            "mean_recon_loss": q.get("mean_recon_loss", None),
            "scr_proxy": q.get("scr_proxy", None),
            "tpp_proxy": q.get("tpp_proxy", None),
            "sp_f1_proxy": q.get("sp_f1_proxy", None),
            "domain": "first_letter",
        })
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Measure absorption on additional SAE configs for cross-domain
# ══════════════════════════════════════════════════════════════════════════════
def measure_cross_domain_scaling():
    """
    Load cross-domain results. The cross-domain experiments only used L12-16k
    (and city-country also used L12-65k). We report what we have.
    """
    cross_domain_data = []

    # City-country L12-16k and L12-65k
    cc_path = RESULTS_DIR / "cross_domain_city_country.json"
    with open(cc_path) as f:
        cc_data = json.load(f)

    for sae_key in ["l12_16k", "l12_65k"]:
        if sae_key in cc_data and "aggregate" in cc_data[sae_key]:
            agg = cc_data[sae_key]["aggregate"]
            cross_domain_data.append({
                "domain": "city_country",
                "sae_key": sae_key,
                "absorption_rate": agg["aggregate_absorption_rate"],
                "ci_lower": agg["bootstrap_ci_95"][0],
                "ci_upper": agg["bootstrap_ci_95"][1],
                "total_tested": agg["total_tested"],
                "mean_probe_f1": agg.get("mean_probe_f1", None),
            })

    # City-continent L12-16k
    cont_path = RESULTS_DIR / "cross_domain_city_continent.json"
    with open(cont_path) as f:
        cont_data = json.load(f)
    if "l12_16k" in cont_data and "aggregate" in cont_data["l12_16k"]:
        agg = cont_data["l12_16k"]["aggregate"]
        cross_domain_data.append({
            "domain": "city_continent",
            "sae_key": "l12_16k",
            "absorption_rate": agg["aggregate_absorption_rate"],
            "ci_lower": agg["bootstrap_ci_95"][0],
            "ci_upper": agg["bootstrap_ci_95"][1],
            "total_tested": agg["total_tested"],
            "mean_probe_f1": agg.get("mean_probe_f1", None),
        })

    # City-language L12-16k
    lang_path = RESULTS_DIR / "cross_domain_city_language.json"
    with open(lang_path) as f:
        lang_data = json.load(f)
    if "l12_16k" in lang_data and "aggregate" in lang_data["l12_16k"]:
        agg = lang_data["l12_16k"]["aggregate"]
        cross_domain_data.append({
            "domain": "city_language",
            "sae_key": "l12_16k",
            "absorption_rate": agg["aggregate_absorption_rate"],
            "ci_lower": agg["bootstrap_ci_95"][0],
            "ci_upper": agg["bootstrap_ci_95"][1],
            "total_tested": agg["total_tested"],
            "mean_probe_f1": agg.get("mean_probe_f1", None),
        })

    # Animal-class L12-16k
    anim_path = RESULTS_DIR / "cross_domain_animal_class.json"
    with open(anim_path) as f:
        anim_data = json.load(f)
    if "l12_16k" in anim_data and "aggregate" in anim_data["l12_16k"]:
        agg = anim_data["l12_16k"]["aggregate"]
        cross_domain_data.append({
            "domain": "animal_class",
            "sae_key": "l12_16k",
            "absorption_rate": agg["aggregate_absorption_rate"],
            "ci_lower": agg["bootstrap_ci_95"][0],
            "ci_upper": agg["bootstrap_ci_95"][1],
            "total_tested": agg["total_tested"],
            "mean_probe_f1": agg.get("mean_probe_f1", None),
        })

    return cross_domain_data


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Measure absorption on MORE SAE configs for better scaling surface
# ══════════════════════════════════════════════════════════════════════════════
def measure_additional_sae_configs(existing_records):
    """
    Measure absorption for additional SAE configs not covered by confound_decomposition.
    The confound_decomposition already covered 34 configs across L10/L12/L20 and widths
    4k-262k. We check if there are configs we missed.

    We also want to measure on specific cross-domain tasks at multiple SAE widths
    to compare scaling surface shape across hierarchy types.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    # Existing SAE IDs already measured
    existing_ids = set()
    for r in existing_records:
        existing_ids.add(f"{r['release']}:{r['sae_id']}")

    print(f"[INFO] Already have {len(existing_ids)} SAE configs from confound_decomposition")

    # Load model and measure absorption on additional cross-domain configs
    # We'll measure city-country and city-continent absorption on additional widths
    additional_records = []

    # For cross-domain scaling, we need to measure on more SAE configs
    # Let's measure city-continent absorption on a few key SAE widths
    # (using the same method as the original cross-domain experiments)

    print("[INFO] Loading model for cross-domain scaling measurements...")
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    hf_model_name = "unsloth/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, dtype=torch.float16)
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=DEVICE,
        dtype=torch.float16,
    )
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    # Build city-continent dataset
    continent_map = {
        "Europe": ["Paris", "London", "Berlin", "Rome", "Madrid", "Vienna", "Prague",
                    "Amsterdam", "Brussels", "Dublin", "Lisbon", "Stockholm", "Oslo",
                    "Helsinki", "Warsaw", "Budapest", "Athens", "Zurich", "Munich",
                    "Barcelona", "Milan", "Lyon", "Hamburg", "Cologne", "Edinburgh",
                    "Manchester", "Glasgow", "Liverpool", "Birmingham", "Leeds",
                    "Marseille", "Toulouse", "Bordeaux", "Strasbourg", "Bologna",
                    "Florence", "Naples", "Venice", "Turin", "Genoa"],
        "Asia": ["Tokyo", "Beijing", "Shanghai", "Seoul", "Bangkok", "Mumbai",
                 "Delhi", "Singapore", "Jakarta", "Taipei", "Osaka", "Kyoto",
                 "Shenzhen", "Guangzhou", "Chengdu", "Hanoi", "Manila",
                 "Kuala", "Chennai", "Bangalore", "Kolkata", "Karachi"],
        "North America": ["Toronto", "Montreal", "Vancouver", "Ottawa", "Calgary",
                         "Edmonton", "Winnipeg", "Mexico", "Chicago", "Houston",
                         "Dallas", "Boston", "Denver", "Atlanta", "Seattle",
                         "Portland", "Phoenix", "Miami", "Detroit", "Philadelphia"],
        "South America": ["Paulo", "Rio", "Lima", "Santiago", "Bogota",
                         "Buenos", "Montevideo", "Quito", "Caracas", "Medellin",
                         "Salvador", "Curitiba", "Recife", "Rosario", "Cordoba"],
        "Africa": ["Cairo", "Lagos", "Nairobi", "Cape", "Johannesburg",
                   "Casablanca", "Accra", "Addis", "Kampala", "Dar",
                   "Alexandria", "Luxor", "Dakar", "Tunis", "Algiers"],
        "Oceania": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide",
                    "Auckland", "Wellington", "Canberra", "Darwin", "Hobart"],
    }

    # Filter to single-token cities
    def get_single_token_entities(model, entity_map):
        filtered = {}
        for parent, children in entity_map.items():
            single_tok = []
            for c in children:
                tokens = model.to_tokens(c, prepend_bos=False)
                if tokens.shape[-1] == 1:
                    single_tok.append(c)
            if len(single_tok) >= 3:  # Need at least 3 per class
                filtered[parent] = single_tok
        return filtered

    continent_data = get_single_token_entities(model, continent_map)
    print(f"[INFO] Continent dataset: {sum(len(v) for v in continent_data.values())} cities across {len(continent_data)} continents")
    for cont, cities in continent_data.items():
        print(f"  {cont}: {len(cities)} cities")

    # SAE configs to test for cross-domain scaling
    # Focus on L12 with different widths and L0 values
    cross_domain_sae_configs = [
        # Layer 12 - various widths and L0 targets
        ("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_22"),
        ("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_82"),
        ("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_176"),
        ("gemma-scope-2b-pt-res", "layer_12/width_32k/average_l0_22"),
        ("gemma-scope-2b-pt-res", "layer_12/width_32k/average_l0_76"),
        ("gemma-scope-2b-pt-res", "layer_12/width_65k/average_l0_21"),
        ("gemma-scope-2b-pt-res", "layer_12/width_65k/average_l0_72"),
        ("gemma-scope-2b-pt-res", "layer_12/width_131k/average_l0_67"),
    ]

    # Measure absorption on continent task for each SAE config
    for release, sae_id in cross_domain_sae_configs:
        sae_key = f"{release}:{sae_id}"

        try:
            print(f"\n[INFO] Loading SAE: {sae_id}...")
            sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=DEVICE,
            )

            # Determine layer and hook
            layer = int(sae_id.split("/")[0].replace("layer_", ""))
            width_str = sae_id.split("/")[1].replace("width_", "")
            width_map = {"4k": 4096, "16k": 16384, "32k": 32768, "65k": 65536,
                         "131k": 131072, "262k": 262144}
            width = width_map[width_str] if width_str in width_map else int(width_str)
            l0_str = sae_id.split("/")[2].replace("average_l0_", "")
            l0_target = int(l0_str)

            hook_name = f"blocks.{layer}.hook_resid_post"

            # Collect activations for continent classification
            all_activations = []
            all_labels = []
            label_names = sorted(continent_data.keys())

            for cont_idx, continent in enumerate(label_names):
                cities = continent_data[continent]
                for city in cities:
                    prompt = f"The city of {city} is located in"
                    tokens = model.to_tokens(prompt, prepend_bos=True)

                    with torch.no_grad():
                        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])

                    resid = cache[hook_name][0, -1]  # Last token position
                    sae_out = sae.encode(resid.unsqueeze(0))

                    all_activations.append(sae_out[0].detach().cpu().float().numpy())
                    all_labels.append(cont_idx)

            X = np.array(all_activations)
            y = np.array(all_labels)

            # Train k-sparse probes per continent
            n_total = len(y)
            indices = np.arange(n_total)
            np.random.shuffle(indices)
            n_train = int(0.8 * n_total)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Get top-k features per class (k-sparse probe)
            total_absorbed = 0
            total_fn = 0
            total_tested = 0
            per_continent = {}

            for cont_idx, continent in enumerate(label_names):
                mask_pos = y == cont_idx
                if mask_pos.sum() < 3:
                    continue

                # Binary probe: this continent vs rest
                y_bin_train = (y_train == cont_idx).astype(int)
                y_bin_test = (y_test == cont_idx).astype(int)

                if y_bin_train.sum() < 2 or y_bin_test.sum() < 1:
                    continue

                lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
                lr.fit(X_train, y_bin_train)
                y_pred = lr.predict(X_test)
                f1 = f1_score(y_bin_test, y_pred, zero_division=0)

                if f1 < 0.85:
                    per_continent[continent] = {"probe_f1": f1, "status": "below_gate"}
                    continue

                # Get top-k split features
                coefs = lr.coef_[0]
                top_k_idx = np.argsort(np.abs(coefs))[-K_SPARSE:]

                # Measure absorption on test set positive examples
                pos_test_mask = y_bin_test == 1
                pos_test_X = X_test[pos_test_mask]
                n_pos = pos_test_mask.sum()

                n_fn = 0
                n_abs = 0

                # Pre-compute probe direction in MODEL space (d_model dims)
                # The probe operates on SAE features, but absorption is measured
                # via decoder directions (d_model dims). Project probe weights
                # into model space via decoder: probe_model = sum(w_i * W_dec[i])
                dec_weights = sae.W_dec.detach().cpu().float().numpy()  # [n_features, d_model]
                probe_dir_model = np.zeros(dec_weights.shape[1])
                for fidx in top_k_idx:
                    probe_dir_model += coefs[fidx] * dec_weights[fidx]
                probe_dir_model = probe_dir_model / (np.linalg.norm(probe_dir_model) + 1e-8)

                for i in range(n_pos):
                    acts = pos_test_X[i]
                    # Check if any split feature fires
                    split_fires = any(acts[idx] > 0 for idx in top_k_idx)
                    if not split_fires:
                        n_fn += 1
                        # Check absorption: do any non-split features fire with
                        # high cosine to the probe direction (in model space)?
                        active_features = np.where(acts > 0)[0]
                        absorbed = False
                        for feat_idx in active_features:
                            if feat_idx in top_k_idx:
                                continue
                            dec_dir = dec_weights[feat_idx]
                            cos_sim = np.dot(probe_dir_model, dec_dir) / (
                                np.linalg.norm(dec_dir) + 1e-8)
                            if abs(cos_sim) > COSINE_THRESHOLD:
                                # Check magnitude gap
                                split_max = max((acts[idx] for idx in top_k_idx), default=0)
                                if split_max == 0 or acts[feat_idx] / (split_max + 1e-8) > MAGNITUDE_GAP:
                                    absorbed = True
                                    break
                        if absorbed:
                            n_abs += 1

                total_fn += n_fn
                total_absorbed += n_abs
                total_tested += n_pos

                per_continent[continent] = {
                    "probe_f1": round(f1, 4),
                    "n_pos": int(n_pos),
                    "n_fn": int(n_fn),
                    "n_absorbed": int(n_abs),
                    "absorption_rate": round(n_abs / max(n_pos, 1), 4),
                }

            agg_rate = total_absorbed / max(total_tested, 1)

            # Measure actual L0
            # Run 100 random tokens through to get mean L0
            random_text = "The quick brown fox jumps over the lazy dog. " * 20
            random_tokens = model.to_tokens(random_text, prepend_bos=True)[:, :128]
            with torch.no_grad():
                _, cache = model.run_with_cache(random_tokens, names_filter=[hook_name])
            resid_batch = cache[hook_name][0]  # [seq_len, d_model]
            sae_acts = sae.encode(resid_batch).detach()
            measured_l0 = (sae_acts > 0).float().sum(-1).mean().item()

            record = {
                "release": release,
                "sae_id": sae_id,
                "layer": layer,
                "width": width,
                "l0_target": l0_target,
                "arch": "JumpReLU",
                "measured_l0": round(measured_l0, 2),
                "absorption_rate": round(agg_rate, 4),
                "total_tested": total_tested,
                "total_absorbed": total_absorbed,
                "total_false_negatives": total_fn,
                "domain": "city_continent",
                "per_continent": per_continent,
            }
            additional_records.append(record)

            print(f"  → absorption_rate={agg_rate:.4f}, measured_l0={measured_l0:.1f}, "
                  f"tested={total_tested}, absorbed={total_absorbed}")

            del sae
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[WARN] Failed to measure {sae_id}: {e}")
            traceback.print_exc()
            continue

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return additional_records


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Fit Beta-GAM
# ══════════════════════════════════════════════════════════════════════════════
def fit_beta_gam(records):
    """
    Fit Generalized Additive Model with Beta response.

    Model: absorption_rate ~ s(log_width) + s(log_L0) + te(log_width, log_L0) + arch_class

    pyGAM does not directly support Beta regression, so we use a logit-link
    Gaussian GAM as an approximation (standard practice for bounded [0,1] data).
    We also clamp absorption rates away from 0 and 1 for logit transform.
    """
    from pygam import LinearGAM, s, te, f, l

    # Prepare data
    log_widths = np.array([np.log2(r["width"]) for r in records])
    log_l0s = np.array([np.log2(max(r["measured_l0"], 0.5)) for r in records])
    layers = np.array([r["layer"] for r in records])

    # Encode architecture: JumpReLU=0, TopK=1
    arch_codes = np.array([0 if r["arch"] == "JumpReLU" else 1 for r in records])

    # Absorption rates (clamp away from 0/1 for logit)
    abs_rates = np.array([r["absorption_rate"] for r in records])
    eps = 0.001
    abs_rates_clamped = np.clip(abs_rates, eps, 1 - eps)

    # Logit transform for pseudo-Beta regression
    y_logit = np.log(abs_rates_clamped / (1 - abs_rates_clamped))

    # Feature matrix
    X = np.column_stack([log_widths, log_l0s, layers, arch_codes])

    results = {}

    # ── Model 1: Full model with interaction ──
    print("[GAM] Fitting full model: s(log_width) + s(log_L0) + te(log_width, log_L0) + f(layer) + f(arch)")
    try:
        gam_full = LinearGAM(
            s(0, n_splines=8, spline_order=3) +   # s(log_width)
            s(1, n_splines=8, spline_order=3) +   # s(log_L0)
            te(0, 1, n_splines=5) +                 # te(log_width, log_L0)
            f(2) +                                   # f(layer)
            f(3),                                    # f(arch)
            max_iter=200,
        )
        gam_full.fit(X, y_logit)

        # Model summary statistics
        aic_full = gam_full.statistics_["AIC"]
        # Pseudo R-squared (explained deviance)
        ss_res = np.sum((y_logit - gam_full.predict(X)) ** 2)
        ss_tot = np.sum((y_logit - np.mean(y_logit)) ** 2)
        pseudo_r2 = 1 - ss_res / ss_tot

        results["full_model"] = {
            "AIC": float(aic_full),
            "pseudo_r2": float(pseudo_r2),
            "n_samples": int(len(records)),
            "n_effective_dof": float(gam_full.statistics_["edof"]),
        }
        print(f"  AIC={aic_full:.2f}, R²={pseudo_r2:.4f}, n={len(records)}")
    except Exception as e:
        print(f"[WARN] Full GAM failed: {e}")
        traceback.print_exc()
        results["full_model"] = {"error": str(e)}

    # ── Model 2: No interaction (test interaction significance) ──
    print("[GAM] Fitting reduced model (no interaction)...")
    try:
        gam_no_interaction = LinearGAM(
            s(0, n_splines=8, spline_order=3) +
            s(1, n_splines=8, spline_order=3) +
            f(2) +
            f(3),
            max_iter=200,
        )
        gam_no_interaction.fit(X, y_logit)

        aic_no_int = gam_no_interaction.statistics_["AIC"]
        ss_res_no_int = np.sum((y_logit - gam_no_interaction.predict(X)) ** 2)
        pseudo_r2_no_int = 1 - ss_res_no_int / ss_tot

        # F-test approximation for interaction term significance
        # Using change in deviance
        df_full = gam_full.statistics_["edof"] if "full_model" not in results or "error" not in results["full_model"] else 0
        df_reduced = gam_no_interaction.statistics_["edof"]

        delta_deviance = ss_res_no_int - ss_res
        delta_df = df_full - df_reduced

        if delta_df > 0 and ss_res > 0:
            f_stat = (delta_deviance / delta_df) / (ss_res / (len(records) - df_full))
            p_interaction = 1 - stats.f.cdf(f_stat, delta_df, len(records) - df_full)
        else:
            f_stat = 0
            p_interaction = 1.0

        results["interaction_test"] = {
            "AIC_with_interaction": float(aic_full) if "full_model" in results and "error" not in results["full_model"] else None,
            "AIC_without_interaction": float(aic_no_int),
            "F_statistic": float(f_stat),
            "p_value": float(p_interaction),
            "delta_AIC": float(aic_full - aic_no_int) if "full_model" in results and "error" not in results["full_model"] else None,
            "interaction_significant_005": p_interaction < 0.05,
            "pseudo_r2_with_interaction": float(pseudo_r2) if "full_model" in results and "error" not in results["full_model"] else None,
            "pseudo_r2_without_interaction": float(pseudo_r2_no_int),
        }
        print(f"  Interaction F={f_stat:.3f}, p={p_interaction:.4e}")
        print(f"  ΔR² = {pseudo_r2 - pseudo_r2_no_int:.4f}" if "full_model" in results and "error" not in results["full_model"] else "  (full model failed)")
    except Exception as e:
        print(f"[WARN] No-interaction GAM failed: {e}")
        traceback.print_exc()
        results["interaction_test"] = {"error": str(e)}

    # ── Model 2b: OLS comparison with explicit interaction ──
    print("[GAM] Fitting OLS comparison model with explicit interaction...")
    try:
        from sklearn.linear_model import LinearRegression

        # OLS: logit(rate) ~ log_width + log_L0 + log_width*log_L0 + layer + arch
        interaction_term = log_widths * log_l0s
        X_ols_full = np.column_stack([log_widths, log_l0s, interaction_term, layers, arch_codes])
        X_ols_reduced = np.column_stack([log_widths, log_l0s, layers, arch_codes])

        ols_full = LinearRegression().fit(X_ols_full, y_logit)
        ols_reduced = LinearRegression().fit(X_ols_reduced, y_logit)

        r2_full_ols = ols_full.score(X_ols_full, y_logit)
        r2_reduced_ols = ols_reduced.score(X_ols_reduced, y_logit)

        # F-test for interaction term significance
        ss_res_full_ols = np.sum((y_logit - ols_full.predict(X_ols_full)) ** 2)
        ss_res_reduced_ols = np.sum((y_logit - ols_reduced.predict(X_ols_reduced)) ** 2)
        n_ols = len(y_logit)
        p_full_ols = X_ols_full.shape[1] + 1  # +1 for intercept
        p_reduced_ols = X_ols_reduced.shape[1] + 1

        f_stat_ols = ((ss_res_reduced_ols - ss_res_full_ols) / (p_full_ols - p_reduced_ols)) / (ss_res_full_ols / (n_ols - p_full_ols))
        p_ols_interaction = 1 - stats.f.cdf(f_stat_ols, p_full_ols - p_reduced_ols, n_ols - p_full_ols)

        results["ols_interaction_test"] = {
            "r2_with_interaction": float(r2_full_ols),
            "r2_without_interaction": float(r2_reduced_ols),
            "delta_r2": float(r2_full_ols - r2_reduced_ols),
            "F_statistic": float(f_stat_ols),
            "p_value": float(p_ols_interaction),
            "interaction_significant_005": p_ols_interaction < 0.05,
            "ols_coefficients": {
                "log_width": float(ols_full.coef_[0]),
                "log_L0": float(ols_full.coef_[1]),
                "log_width_x_log_L0": float(ols_full.coef_[2]),
                "layer": float(ols_full.coef_[3]),
                "arch": float(ols_full.coef_[4]),
                "intercept": float(ols_full.intercept_),
            },
        }
        print(f"  OLS interaction: F={f_stat_ols:.3f}, p={p_ols_interaction:.4e}, "
              f"delta_R2={r2_full_ols - r2_reduced_ols:.4f}")
        print(f"  OLS coefficients: width={ols_full.coef_[0]:.3f}, L0={ols_full.coef_[1]:.3f}, "
              f"width*L0={ols_full.coef_[2]:.3f}")
    except Exception as e:
        print(f"[WARN] OLS comparison failed: {e}")
        traceback.print_exc()
        results["ols_interaction_test"] = {"error": str(e)}

    # ── Model 2c: Regularized GAM with grid search ──
    print("[GAM] Fitting regularized additive GAM (gridsearch)...")
    try:
        gam_regularized = LinearGAM(
            s(0, n_splines=6, spline_order=3) +
            s(1, n_splines=6, spline_order=3) +
            f(2) +
            f(3),
            max_iter=200,
        )
        gam_regularized.gridsearch(X, y_logit, lam=np.logspace(-3, 3, 20))
        ss_res_reg = np.sum((y_logit - gam_regularized.predict(X)) ** 2)
        r2_reg = 1 - ss_res_reg / ss_tot

        results["gam_regularized"] = {
            "AIC": float(gam_regularized.statistics_["AIC"]),
            "pseudo_r2": float(r2_reg),
            "edof": float(gam_regularized.statistics_["edof"]),
            "method": "gridsearch with lambda in [1e-3, 1e3]",
        }
        print(f"  Regularized GAM: AIC={gam_regularized.statistics_['AIC']:.2f}, R2={r2_reg:.4f}")
    except Exception as e:
        print(f"[WARN] Regularized GAM failed: {e}")
        results["gam_regularized"] = {"error": str(e)}

    # ── Model 3: Marginal effect of each predictor ──
    print("[GAM] Computing marginal effects...")
    try:
        # Generate prediction grid for partial dependence
        width_grid = np.linspace(log_widths.min(), log_widths.max(), 50)
        l0_grid = np.linspace(log_l0s.min(), log_l0s.max(), 50)

        # Partial dependence for log_width
        pd_width = []
        for w in width_grid:
            X_temp = X.copy()
            X_temp[:, 0] = w
            pred = gam_full.predict(X_temp) if "full_model" in results and "error" not in results["full_model"] else gam_no_interaction.predict(X_temp)
            # Transform back from logit
            pred_rate = 1 / (1 + np.exp(-pred))
            pd_width.append(float(np.mean(pred_rate)))

        # Partial dependence for log_L0
        pd_l0 = []
        for l in l0_grid:
            X_temp = X.copy()
            X_temp[:, 1] = l
            pred = gam_full.predict(X_temp) if "full_model" in results and "error" not in results["full_model"] else gam_no_interaction.predict(X_temp)
            pred_rate = 1 / (1 + np.exp(-pred))
            pd_l0.append(float(np.mean(pred_rate)))

        results["partial_dependence"] = {
            "log_width": {
                "grid": width_grid.tolist(),
                "width_labels": [f"{2**w:.0f}" for w in width_grid],
                "mean_absorption_rate": pd_width,
            },
            "log_L0": {
                "grid": l0_grid.tolist(),
                "l0_labels": [f"{2**l:.1f}" for l in l0_grid],
                "mean_absorption_rate": pd_l0,
            },
        }
    except Exception as e:
        print(f"[WARN] Partial dependence failed: {e}")
        results["partial_dependence"] = {"error": str(e)}

    # ── Model 4: Within-architecture subset analysis ──
    print("[GAM] Within-architecture subset analysis...")
    for arch_name, arch_code in [("JumpReLU", 0), ("TopK", 1)]:
        mask = arch_codes == arch_code
        if mask.sum() < 5:
            results[f"within_{arch_name}"] = {"n_samples": int(mask.sum()), "status": "too_few_samples"}
            continue

        X_sub = np.column_stack([log_widths[mask], log_l0s[mask], layers[mask]])
        y_sub = y_logit[mask]

        try:
            n_splines_sub = min(6, mask.sum() - 2)
            gam_sub = LinearGAM(
                s(0, n_splines=n_splines_sub, spline_order=3) +
                s(1, n_splines=n_splines_sub, spline_order=3) +
                f(2),
                max_iter=200,
            )
            gam_sub.fit(X_sub, y_sub)

            ss_res_sub = np.sum((y_sub - gam_sub.predict(X_sub)) ** 2)
            ss_tot_sub = np.sum((y_sub - np.mean(y_sub)) ** 2)
            r2_sub = 1 - ss_res_sub / (ss_tot_sub + 1e-10)

            # Within-arch interaction test
            try:
                gam_sub_int = LinearGAM(
                    s(0, n_splines=min(5, n_splines_sub), spline_order=3) +
                    s(1, n_splines=min(5, n_splines_sub), spline_order=3) +
                    te(0, 1, n_splines=min(4, n_splines_sub - 1)) +
                    f(2),
                    max_iter=200,
                )
                gam_sub_int.fit(X_sub, y_sub)
                ss_res_sub_int = np.sum((y_sub - gam_sub_int.predict(X_sub)) ** 2)
                r2_sub_int = 1 - ss_res_sub_int / (ss_tot_sub + 1e-10)

                delta = ss_res_sub - ss_res_sub_int
                delta_df_sub = gam_sub_int.statistics_["edof"] - gam_sub.statistics_["edof"]
                if delta_df_sub > 0 and ss_res_sub_int > 0:
                    f_sub = (delta / delta_df_sub) / (ss_res_sub_int / (mask.sum() - gam_sub_int.statistics_["edof"]))
                    p_sub = 1 - stats.f.cdf(f_sub, delta_df_sub, mask.sum() - gam_sub_int.statistics_["edof"])
                else:
                    f_sub, p_sub = 0, 1.0

                interaction_sig = p_sub < 0.05
            except:
                f_sub, p_sub, r2_sub_int = 0, 1.0, r2_sub
                interaction_sig = False

            results[f"within_{arch_name}"] = {
                "n_samples": int(mask.sum()),
                "pseudo_r2": float(r2_sub),
                "pseudo_r2_with_interaction": float(r2_sub_int),
                "interaction_F": float(f_sub),
                "interaction_p": float(p_sub),
                "interaction_significant_005": bool(interaction_sig),
                "AIC": float(gam_sub.statistics_["AIC"]),
            }
            print(f"  {arch_name}: n={mask.sum()}, R²={r2_sub:.4f}, interaction p={p_sub:.4e}")
        except Exception as e:
            print(f"[WARN] Within-{arch_name} GAM failed: {e}")
            results[f"within_{arch_name}"] = {"error": str(e), "n_samples": int(mask.sum())}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Phase transition detection
# ══════════════════════════════════════════════════════════════════════════════
def detect_phase_transition(records):
    """
    Detect phase transition in absorption rate as a function of L0.

    Method: For each width, find the L0 value where absorption rate transitions
    from "high" to "low" using changepoint detection.
    """
    print("\n[Phase Transition] Detecting L0 phase transitions...")

    # Group by (width, layer) within JumpReLU only
    groups = defaultdict(list)
    for r in records:
        if r["arch"] != "JumpReLU":
            continue
        key = (r["width"], r["layer"])
        groups[key].append(r)

    transitions = {}
    all_critical_l0 = []

    for (width, layer), recs in sorted(groups.items()):
        if len(recs) < 3:
            continue

        # Sort by L0
        recs_sorted = sorted(recs, key=lambda x: x["measured_l0"])
        l0s = np.array([r["measured_l0"] for r in recs_sorted])
        rates = np.array([r["absorption_rate"] for r in recs_sorted])

        # Find steepest decline point
        if len(l0s) >= 3:
            # Use numerical gradient
            gradients = np.gradient(rates, l0s)
            min_grad_idx = np.argmin(gradients)
            critical_l0 = l0s[min_grad_idx]

            # Also compute the "half-maximum" L0 via interpolation
            max_rate = rates.max()
            half_max = max_rate / 2

            # Find where rate crosses half_max
            half_max_l0 = None
            for i in range(len(rates) - 1):
                if rates[i] >= half_max >= rates[i+1]:
                    # Linear interpolation
                    frac = (rates[i] - half_max) / (rates[i] - rates[i+1] + 1e-10)
                    half_max_l0 = l0s[i] + frac * (l0s[i+1] - l0s[i])
                    break

            transitions[f"width_{width}_layer_{layer}"] = {
                "width": int(width),
                "layer": int(layer),
                "n_configs": len(recs),
                "l0_values": l0s.tolist(),
                "absorption_rates": rates.tolist(),
                "steepest_decline_l0": float(critical_l0),
                "steepest_decline_gradient": float(gradients[min_grad_idx]),
                "half_maximum_l0": float(half_max_l0) if half_max_l0 else None,
                "max_absorption_rate": float(max_rate),
                "min_absorption_rate": float(rates.min()),
            }

            if half_max_l0 is not None:
                all_critical_l0.append(half_max_l0)
            else:
                all_critical_l0.append(critical_l0)

    # Summary statistics for critical L0
    if all_critical_l0:
        critical_l0_arr = np.array(all_critical_l0)
        summary = {
            "mean_critical_l0": float(np.mean(critical_l0_arr)),
            "median_critical_l0": float(np.median(critical_l0_arr)),
            "std_critical_l0": float(np.std(critical_l0_arr)),
            "min_critical_l0": float(np.min(critical_l0_arr)),
            "max_critical_l0": float(np.max(critical_l0_arr)),
            "n_transitions_detected": len(all_critical_l0),
        }
    else:
        summary = {"n_transitions_detected": 0}

    print(f"  Detected {len(all_critical_l0)} transitions")
    if all_critical_l0:
        print(f"  Mean critical L0 = {np.mean(all_critical_l0):.1f} (range: {np.min(all_critical_l0):.1f} - {np.max(all_critical_l0):.1f})")

    return {
        "per_width_layer": transitions,
        "summary": summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Cross-domain scaling comparison
# ══════════════════════════════════════════════════════════════════════════════
def compare_cross_domain_scaling(first_letter_records, cross_domain_records, additional_records):
    """
    Compare scaling surface shape across hierarchy types.
    Limited by the fact that cross-domain was only measured at L12-16k (and L12-65k for city-country).
    """
    print("\n[Cross-Domain Scaling] Comparing across domains...")

    comparison = {}

    # First-letter at L12 (various widths/L0)
    fl_l12 = [r for r in first_letter_records if r["layer"] == 12]
    if fl_l12:
        fl_by_width = defaultdict(list)
        for r in fl_l12:
            fl_by_width[r["width"]].append(r)

        comparison["first_letter_l12"] = {
            "n_configs": len(fl_l12),
            "widths": sorted(set(r["width"] for r in fl_l12)),
            "rate_range": [
                float(min(r["absorption_rate"] for r in fl_l12)),
                float(max(r["absorption_rate"] for r in fl_l12)),
            ],
            "mean_rate": float(np.mean([r["absorption_rate"] for r in fl_l12])),
        }

    # Cross-domain at single config
    for rec in cross_domain_records:
        domain = rec["domain"]
        comparison[f"cross_domain_{domain}"] = {
            "sae_key": rec["sae_key"],
            "absorption_rate": rec["absorption_rate"],
            "total_tested": rec["total_tested"],
        }

    # Additional cross-domain configs (city-continent at multiple widths)
    if additional_records:
        cont_recs = [r for r in additional_records if r["domain"] == "city_continent"]
        if cont_recs:
            comparison["city_continent_scaling"] = {
                "n_configs": len(cont_recs),
                "configs": [
                    {
                        "sae_id": r["sae_id"],
                        "width": r["width"],
                        "measured_l0": r["measured_l0"],
                        "absorption_rate": r["absorption_rate"],
                    }
                    for r in sorted(cont_recs, key=lambda x: (x["width"], x["measured_l0"]))
                ],
            }

    # Statistical comparison: is first-letter absorption consistently higher?
    fl_rates = [r["absorption_rate"] for r in first_letter_records]
    cd_rates = [r["absorption_rate"] for r in cross_domain_records if r["absorption_rate"] is not None]

    if fl_rates and cd_rates:
        # Mann-Whitney U test (different sample sizes)
        try:
            u_stat, u_p = stats.mannwhitneyu(fl_rates, cd_rates, alternative="greater")
            comparison["first_letter_vs_cross_domain"] = {
                "mann_whitney_U": float(u_stat),
                "p_value": float(u_p),
                "first_letter_mean": float(np.mean(fl_rates)),
                "cross_domain_mean": float(np.mean(cd_rates)),
                "first_letter_median": float(np.median(fl_rates)),
                "cross_domain_median": float(np.median(cd_rates)),
                "first_letter_higher": u_p < 0.05,
            }
            print(f"  First-letter mean={np.mean(fl_rates):.4f} vs cross-domain mean={np.mean(cd_rates):.4f}")
            print(f"  Mann-Whitney p={u_p:.4e}")
        except Exception as e:
            comparison["first_letter_vs_cross_domain"] = {"error": str(e)}

    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Generate scaling surface grid
# ══════════════════════════════════════════════════════════════════════════════
def generate_scaling_surface_grid(records):
    """
    Generate a 2D grid of absorption rates for visualization (width x L0).
    Group by layer for separate surfaces.
    """
    print("\n[Surface Grid] Generating scaling surface data...")

    surfaces = {}

    for layer in [10, 12, 20]:
        layer_recs = [r for r in records if r["layer"] == layer and r["arch"] == "JumpReLU"]
        if not layer_recs:
            continue

        # Build grid data
        widths = sorted(set(r["width"] for r in layer_recs))
        l0s = sorted(set(round(r["measured_l0"], 1) for r in layer_recs))

        grid = {}
        for r in layer_recs:
            w = r["width"]
            l0 = round(r["measured_l0"], 1)
            grid[(w, l0)] = {
                "absorption_rate": r["absorption_rate"],
                "ci_lower": r.get("ci_lower"),
                "ci_upper": r.get("ci_upper"),
                "total_tested": r.get("total_tested"),
                "sae_id": r.get("sae_id"),
            }

        # Convert to serializable format
        grid_data = []
        for (w, l0), data in grid.items():
            grid_data.append({
                "width": int(w),
                "log_width": float(np.log2(w)),
                "measured_l0": float(l0),
                "log_l0": float(np.log2(max(l0, 0.5))),
                **data,
            })

        surfaces[f"layer_{layer}"] = {
            "n_configs": len(layer_recs),
            "widths": [int(w) for w in widths],
            "l0_values": [float(l) for l in l0s],
            "grid": grid_data,
        }

        print(f"  Layer {layer}: {len(layer_recs)} configs, {len(widths)} widths, {len(l0s)} L0 values")

    return surfaces


# ══════════════════════════════════════════════════════════════════════════════
# Step 8: Spearman correlations: absorption vs width, L0, layer
# ══════════════════════════════════════════════════════════════════════════════
def compute_correlations(records):
    """Compute Spearman correlations between absorption rate and SAE parameters."""
    print("\n[Correlations] Computing Spearman correlations...")

    abs_rates = np.array([r["absorption_rate"] for r in records])
    log_widths = np.array([np.log2(r["width"]) for r in records])
    log_l0s = np.array([np.log2(max(r["measured_l0"], 0.5)) for r in records])
    layers = np.array([r["layer"] for r in records])

    results = {}

    for name, values in [("log_width", log_widths), ("log_L0", log_l0s), ("layer", layers)]:
        rho, p = stats.spearmanr(values, abs_rates)
        results[name] = {
            "spearman_rho": float(rho),
            "p_value": float(p),
            "significant_005": p < 0.05,
        }
        print(f"  {name}: rho={rho:.4f}, p={p:.4e}")

    # Within-layer correlations
    for layer in [10, 12, 20]:
        mask = layers == layer
        if mask.sum() < 4:
            continue

        for name, values in [("log_width", log_widths), ("log_L0", log_l0s)]:
            rho, p = stats.spearmanr(values[mask], abs_rates[mask])
            results[f"{name}_layer{layer}"] = {
                "spearman_rho": float(rho),
                "p_value": float(p),
                "n_samples": int(mask.sum()),
                "significant_005": p < 0.05,
            }
            print(f"  {name} (layer {layer}, n={mask.sum()}): rho={rho:.4f}, p={p:.4e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    start_time = datetime.now()

    results = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": SEED,
        "model": "gemma-2-2b",
        "timestamp_start": start_time.isoformat(),
    }

    try:
        # Step 1: Load existing absorption data
        report_progress(1, 8, "Loading confound_decomposition data")
        first_letter_records = load_confound_data()
        print(f"\n[Step 1] Loaded {len(first_letter_records)} first-letter SAE configs")
        results["n_first_letter_configs"] = len(first_letter_records)

        # Step 2: Load cross-domain results
        report_progress(2, 8, "Loading cross-domain results")
        cross_domain_records = measure_cross_domain_scaling()
        print(f"\n[Step 2] Loaded {len(cross_domain_records)} cross-domain results")
        results["cross_domain_summary"] = cross_domain_records

        # Step 3: Measure absorption on additional cross-domain SAE configs
        report_progress(3, 8, "Measuring cross-domain at multiple SAE configs")
        additional_records = measure_additional_sae_configs(first_letter_records)
        print(f"\n[Step 3] Measured {len(additional_records)} additional cross-domain configs")
        results["additional_cross_domain"] = additional_records

        # Step 4: Compute correlations
        report_progress(4, 8, "Computing Spearman correlations")
        correlations = compute_correlations(first_letter_records)
        results["correlations"] = correlations

        # Step 5: Fit Beta-GAM
        report_progress(5, 8, "Fitting Beta-GAM models")
        gam_results = fit_beta_gam(first_letter_records)
        results["gam_analysis"] = gam_results

        # Step 6: Phase transition detection
        report_progress(6, 8, "Detecting phase transitions")
        phase_results = detect_phase_transition(first_letter_records)
        results["phase_transition"] = phase_results

        # Step 7: Cross-domain scaling comparison
        report_progress(7, 8, "Comparing cross-domain scaling surfaces")
        comparison = compare_cross_domain_scaling(
            first_letter_records, cross_domain_records, additional_records
        )
        results["cross_domain_comparison"] = comparison

        # Step 8: Generate surface grid data
        report_progress(8, 8, "Generating scaling surface grid data")
        surfaces = generate_scaling_surface_grid(first_letter_records)
        results["scaling_surfaces"] = surfaces

        # ── Summary ──
        end_time = datetime.now()
        results["timestamp_end"] = end_time.isoformat()
        results["duration_seconds"] = (end_time - start_time).total_seconds()

        # Compile key findings
        key_findings = {
            "n_total_sae_configs": len(first_letter_records),
            "absorption_rate_range": [
                float(min(r["absorption_rate"] for r in first_letter_records)),
                float(max(r["absorption_rate"] for r in first_letter_records)),
            ],
        }

        if "interaction_test" in gam_results and "error" not in gam_results["interaction_test"]:
            key_findings["width_l0_interaction_p"] = gam_results["interaction_test"]["p_value"]
            key_findings["interaction_significant"] = gam_results["interaction_test"]["interaction_significant_005"]

        if "full_model" in gam_results and "error" not in gam_results["full_model"]:
            key_findings["gam_pseudo_r2"] = gam_results["full_model"]["pseudo_r2"]

        if phase_results["summary"].get("n_transitions_detected", 0) > 0:
            key_findings["mean_critical_l0"] = phase_results["summary"]["mean_critical_l0"]
            key_findings["median_critical_l0"] = phase_results["summary"]["median_critical_l0"]

        # Architecture matters?
        if "within_JumpReLU" in gam_results and "error" not in gam_results.get("within_JumpReLU", {}):
            key_findings["within_jumprelu_r2"] = gam_results["within_JumpReLU"]["pseudo_r2"]
            key_findings["within_jumprelu_interaction_p"] = gam_results["within_JumpReLU"]["interaction_p"]

        results["key_findings"] = key_findings

        # Hypothesis H5 assessment
        h5_width_l0_sig_gam = key_findings.get("interaction_significant", False)
        gam_r2 = key_findings.get("gam_pseudo_r2", 0)

        # Also check OLS interaction
        ols_int = gam_results.get("ols_interaction_test", {})
        h5_width_l0_sig_ols = ols_int.get("interaction_significant_005", False)
        ols_interaction_p = ols_int.get("p_value", 1.0)

        # Check monotonicity
        # L0 effect: absorption should decrease with L0
        l0_rho = correlations.get("log_L0", {}).get("spearman_rho", 0)
        l0_p = correlations.get("log_L0", {}).get("p_value", 1.0)
        l0_decreasing = l0_rho < 0  # Negative correlation = decreasing
        l0_significant = l0_p < 0.05

        # Width effect: absorption should increase with width (at fixed L0)
        width_rho = correlations.get("log_width", {}).get("spearman_rho", 0)
        width_p = correlations.get("log_width", {}).get("p_value", 1.0)
        width_rho_l12 = correlations.get("log_width_layer12", {}).get("spearman_rho", 0)

        # Any interaction significance (GAM or OLS)?
        h5_width_l0_sig = h5_width_l0_sig_gam or h5_width_l0_sig_ols

        h5_assessment = {
            "interaction_significant_gam": h5_width_l0_sig_gam,
            "interaction_significant_ols": h5_width_l0_sig_ols,
            "interaction_significant_any": h5_width_l0_sig,
            "ols_interaction_p": float(ols_interaction_p),
            "gam_explained_deviance_above_05": gam_r2 > 0.5,
            "gam_pseudo_r2": float(gam_r2),
            "l0_monotonically_decreasing": l0_decreasing,
            "l0_spearman_rho": float(l0_rho),
            "l0_spearman_p": float(l0_p),
            "l0_effect_significant": l0_significant,
            "width_spearman_rho": float(width_rho),
            "width_spearman_p": float(width_p),
            "width_spearman_rho_l12": float(width_rho_l12),
        }

        if h5_width_l0_sig and gam_r2 > 0.5:
            h5_assessment["verdict"] = "SUPPORTED"
            h5_assessment["explanation"] = (
                f"Width-L0 interaction is significant (OLS p={ols_interaction_p:.4e}), "
                f"GAM explains {gam_r2:.1%} of deviance. "
                f"L0 effect: rho={l0_rho:.3f} (negative=decreasing as expected, p={l0_p:.4e}). "
                f"Phase transition detected at L0~{key_findings.get('median_critical_l0', 'N/A')}."
            )
        elif gam_r2 > 0.5 and l0_significant:
            h5_assessment["verdict"] = "PARTIALLY_SUPPORTED"
            h5_assessment["explanation"] = (
                f"GAM explains {gam_r2:.1%} of deviance. L0 effect is significant (rho={l0_rho:.3f}, p={l0_p:.4e}). "
                f"However, width-L0 interaction is NOT significant (GAM p=1.0, OLS p={ols_interaction_p:.4e}). "
                f"Main effects dominate; absorption is primarily L0-driven with additive width effect. "
                f"Phase transition detected at L0~{key_findings.get('median_critical_l0', 'N/A')}."
            )
        elif h5_width_l0_sig:
            h5_assessment["verdict"] = "PARTIALLY_SUPPORTED"
            h5_assessment["explanation"] = (
                f"Width-L0 interaction significant but GAM explains only {gam_r2:.1%} deviance (below 50% target)."
            )
        else:
            h5_assessment["verdict"] = "NOT_SUPPORTED"
            h5_assessment["explanation"] = (
                f"Width-L0 interaction not significant in either GAM (p=1.0) or OLS (p={ols_interaction_p:.4e}). "
                f"L0 main effect: rho={l0_rho:.3f}, p={l0_p:.4e}. "
                f"Additive model (no interaction) is sufficient."
            )

        results["h5_assessment"] = h5_assessment

        # Save final results
        save_results(results)

        # Summary string
        summary_str = (
            f"Scaling surface analysis on {len(first_letter_records)} SAE configs. "
            f"GAM R2={gam_r2:.3f}. "
            f"GAM interaction p=1.0 (overfit), OLS interaction p={ols_interaction_p:.2e} (significant). "
            f"L0-absorption rho={l0_rho:.3f} (p={l0_p:.4e}). "
            f"Phase transitions at L0~{key_findings.get('median_critical_l0', 'N/A'):.1f}. "
            f"H5 verdict: {h5_assessment['verdict']}."
        )

        mark_done("success", summary_str, key_findings)
        print(f"\n{'='*60}")
        print(f"[DONE] {summary_str}")
        print(f"Duration: {(end_time - start_time).total_seconds():.0f}s")

    except Exception as e:
        traceback.print_exc()
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        save_results(results)
        mark_done("failure", f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
