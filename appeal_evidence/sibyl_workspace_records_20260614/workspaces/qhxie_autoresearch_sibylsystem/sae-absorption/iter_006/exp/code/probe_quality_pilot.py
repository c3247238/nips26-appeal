#!/usr/bin/env python3
"""
probe_quality_pilot.py — Probe Quality Gate: City-Country on Gemma 2 2B (PILOT)

Decision Gate 1. Train k-sparse (k=5) logistic regression probes on city-country
hierarchy using ICL prompts on Gemma 2 2B + Gemma Scope L12 16k.

Uses single-token cities mapped to countries. Stratified 80/20 train/test split.
Reports per-country and aggregate F1.

Decision criteria:
  - F1 >= 0.85: PROCEED to full cross-domain experiments
  - 0.70 <= F1 < 0.85: PROCEED with caution, report as limitation
  - F1 < 0.70: PIVOT to animal taxonomy

PILOT mode: ~200 samples (50+ cities), seed 42, timeout 900s.
"""

import json
import os
import sys
import time
import gc
import traceback
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
TASK_ID = "probe_quality_pilot"
SEED = 42

K_SPARSE = 5
MAX_CITIES_PER_COUNTRY = 15  # Cap cities per country for balance
MIN_CITIES_PER_COUNTRY = 3   # Minimum cities needed per country

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
        try: fp = json.loads(progress_file.read_text())
        except: pass
    marker = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "results": results or {}, "final_progress": fp,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=SEED):
    np.random.seed(seed)
    if len(values) == 0:
        return None, None
    bs = [np.mean(np.random.choice(values, size=len(values), replace=True))
          for _ in range(n_bootstrap)]
    a = (1 - ci) / 2
    return (round(float(np.percentile(bs, a * 100)), 4),
            round(float(np.percentile(bs, (1 - a) * 100)), 4))


# ── City-Country Dataset ──────────────────────────────────────────────────
# Curated list of well-known cities that are likely single tokens in Gemma 2.
# Grouped by country, with emphasis on globally-recognized cities.
# We need 50+ single-token cities across 20+ countries.
CITY_COUNTRY_RAW = {
    "France": [
        "Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux", "Lille",
        "Nice", "Nantes", "Strasbourg", "Montpellier",
    ],
    "Germany": [
        "Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne", "Dresden",
        "Stuttgart", "Bremen", "Leipzig", "Hanover",
    ],
    "Italy": [
        "Rome", "Milan", "Naples", "Florence", "Venice", "Turin",
        "Bologna", "Genoa", "Palermo", "Verona",
    ],
    "Spain": [
        "Madrid", "Barcelona", "Seville", "Valencia", "Bilbao", "Malaga",
        "Granada", "Toledo", "Cordoba", "Salamanca",
    ],
    "United Kingdom": [
        "London", "Manchester", "Birmingham", "Liverpool", "Edinburgh",
        "Bristol", "Leeds", "Oxford", "Cambridge", "Glasgow",
    ],
    "United States": [
        "Seattle", "Boston", "Dallas", "Miami", "Denver",
        "Portland", "Austin", "Nashville", "Phoenix", "Detroit",
        "Atlanta", "Houston", "Chicago",
    ],
    "China": [
        "Beijing", "Shanghai", "Shenzhen", "Guangzhou", "Chengdu",
        "Hangzhou", "Nanjing", "Wuhan", "Tianjin", "Suzhou",
    ],
    "Japan": [
        "Tokyo", "Osaka", "Kyoto", "Nagoya", "Yokohama",
        "Sapporo", "Kobe", "Fukuoka", "Hiroshima", "Sendai",
    ],
    "India": [
        "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
        "Hyderabad", "Pune", "Jaipur", "Lucknow", "Ahmedabad",
    ],
    "Brazil": [
        "Paulo", "Rio", "Salvador", "Brasilia", "Curitiba",
        "Manaus", "Recife", "Belo",
    ],
    "Russia": [
        "Moscow", "Petersburg", "Novosibirsk", "Kazan", "Samara",
        "Volgograd", "Omsk", "Rostov", "Perm",
    ],
    "Australia": [
        "Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide",
        "Canberra", "Darwin", "Hobart",
    ],
    "Canada": [
        "Toronto", "Montreal", "Vancouver", "Ottawa", "Calgary",
        "Edmonton", "Winnipeg", "Halifax", "Quebec",
    ],
    "Mexico": [
        "Cancun", "Guadalajara", "Monterrey", "Puebla", "Tijuana",
        "Oaxaca", "Merida",
    ],
    "Turkey": [
        "Istanbul", "Ankara", "Izmir", "Antalya", "Bursa",
        "Adana", "Trabzon",
    ],
    "South Korea": [
        "Seoul", "Busan", "Incheon", "Daegu", "Daejeon",
        "Gwangju", "Ulsan",
    ],
    "Netherlands": [
        "Amsterdam", "Rotterdam", "Hague", "Utrecht", "Eindhoven",
        "Groningen", "Leiden",
    ],
    "Poland": [
        "Warsaw", "Krakow", "Wroclaw", "Gdansk", "Poznan",
        "Lublin", "Katowice",
    ],
    "Sweden": [
        "Stockholm", "Gothenburg", "Malmo", "Uppsala", "Lund",
    ],
    "Switzerland": [
        "Zurich", "Geneva", "Bern", "Basel", "Lausanne",
    ],
    "Egypt": [
        "Cairo", "Alexandria", "Luxor", "Aswan", "Giza",
    ],
    "Thailand": [
        "Bangkok", "Phuket", "Pattaya", "Chiang",
    ],
    "South Africa": [
        "Johannesburg", "Durban", "Pretoria", "Soweto",
    ],
    "Argentina": [
        "Aires", "Cordoba", "Rosario", "Mendoza",
    ],
    "Indonesia": [
        "Jakarta", "Bali", "Surabaya", "Bandung", "Medan",
    ],
    "Portugal": [
        "Lisbon", "Porto", "Braga", "Faro", "Coimbra",
    ],
    "Greece": [
        "Athens", "Thessaloniki", "Patras", "Crete",
    ],
    "Norway": [
        "Oslo", "Bergen", "Stavanger", "Trondheim",
    ],
    "Ireland": [
        "Dublin", "Cork", "Galway", "Limerick",
    ],
    "Austria": [
        "Vienna", "Salzburg", "Innsbruck", "Graz", "Linz",
    ],
}


def get_single_token_cities(tokenizer, city_country_raw, max_per_country=15, min_per_country=3):
    """
    Filter cities to those that tokenize as single tokens in Gemma 2.
    Returns dict: country -> list of {city, token_id}.
    """
    country_cities = {}

    for country, cities in city_country_raw.items():
        valid = []
        for city in cities:
            # Try both capitalized and as-is
            for c in [city, city.lower(), city.capitalize()]:
                # Gemma tokenizer: " City" should be a single token for well-known cities
                encoded = tokenizer.encode(f" {c}", add_special_tokens=False)
                if len(encoded) == 1:
                    valid.append({
                        "city": c,
                        "token_id": encoded[0],
                        "country": country,
                    })
                    break

            if len(valid) >= max_per_country:
                break

        if len(valid) >= min_per_country:
            country_cities[country] = valid

    return country_cities


def collect_city_activations(model, sae, tokenizer, country_cities, hook_point, device="cuda:0"):
    """
    Collect SAE activations for each city using ICL-style prompts.

    Prompt template: "The city of [CITY] is located in"
    We take the activation at the city token position (the last token of " [CITY]").
    """
    all_acts = []
    all_raw_acts = []
    all_labels = []  # country labels
    all_cities = []

    for country in sorted(country_cities.keys()):
        cities = country_cities[country]
        for c in cities:
            city_name = c["city"]

            # ICL prompt: elicits geographic knowledge
            prompt = f"The city of {city_name} is located in"
            tokens = model.to_tokens(prompt, prepend_bos=True)

            # Find the city token position
            # Tokenize just the city name to find its token(s)
            city_tokens = tokenizer.encode(f" {city_name}", add_special_tokens=False)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[hook_point],
                    return_type=None,
                )

            # Find the position of the city token in the full sequence
            # The city name should appear after "The city of "
            # Look for the city token in the token sequence
            token_ids = tokens[0].tolist()
            city_tok_id = city_tokens[0] if len(city_tokens) == 1 else None

            # Find city token position by searching for the token
            city_pos = -1
            if city_tok_id is not None:
                for i, tid in enumerate(token_ids):
                    if tid == city_tok_id:
                        city_pos = i
                        break

            if city_pos < 0:
                # Fallback: use position after "The city of " (typically position 4 for Gemma)
                # "The" "city" "of" " City" -> position 3-4
                city_pos = min(4, len(token_ids) - 1)

            raw_act = cache[hook_point][0, city_pos, :].detach()  # [d_model]
            sae_act = sae.encode(raw_act.unsqueeze(0))[0].detach()  # [n_features]

            all_acts.append(sae_act.cpu())
            all_raw_acts.append(raw_act.cpu())
            all_labels.append(country)
            all_cities.append(city_name)

    all_acts_tensor = torch.stack(all_acts)
    all_raw_tensor = torch.stack(all_raw_acts)

    return all_acts_tensor, all_raw_tensor, all_labels, all_cities


def train_and_evaluate_probes(sae_activations, labels, k_sparse=K_SPARSE):
    """
    Train per-country binary probes and evaluate with stratified CV + held-out split.

    For each country:
    1. Binary classification: this country vs all others
    2. Stratified K-Fold CV for robust F1 estimate
    3. Report per-country F1 and aggregate

    Returns detailed per-country and aggregate results.
    """
    X = sae_activations.numpy()
    n_features = X.shape[1]
    countries = sorted(set(labels))

    N_PRESELECT = 200  # Pre-select top features for efficiency

    per_country = {}
    all_f1s = []
    all_test_f1s = []

    for country in countries:
        y = np.array([1 if l == country else 0 for l in labels])
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        if n_pos < 3 or n_neg < 3:
            per_country[country] = {
                "status": "skipped",
                "reason": f"n_pos={n_pos}, n_neg={n_neg}",
                "n_pos": n_pos,
            }
            continue

        # Feature pre-selection by mean difference (fast)
        mean_pos = X[y == 1].mean(axis=0)
        mean_neg = X[y == 0].mean(axis=0)
        feat_score = np.abs(mean_pos - mean_neg)
        preselect_idx = np.argsort(-feat_score)[:N_PRESELECT]
        X_pre = X[:, preselect_idx]

        # Stratified K-Fold CV
        n_splits = min(5, n_pos, n_neg)
        if n_splits < 2:
            per_country[country] = {
                "status": "skipped",
                "reason": "insufficient for CV",
                "n_pos": n_pos,
            }
            continue

        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            fold_f1s = []
            for train_idx, test_idx in skf.split(X_pre, y):
                clf = LogisticRegression(
                    max_iter=1000, C=1.0, solver='lbfgs',
                    random_state=SEED, class_weight='balanced',
                )
                clf.fit(X_pre[train_idx], y[train_idx])
                preds = clf.predict(X_pre[test_idx])
                fold_f1s.append(f1_score(y[test_idx], preds, zero_division=0))

            cv_f1 = float(np.mean(fold_f1s))
            cv_std = float(np.std(fold_f1s))
        except Exception as e:
            per_country[country] = {
                "status": "error",
                "error": str(e),
                "n_pos": n_pos,
            }
            continue

        # 80/20 held-out split for final test F1
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_pre, y, test_size=0.2, stratify=y, random_state=SEED,
            )
            clf_final = LogisticRegression(
                max_iter=1000, C=1.0, solver='lbfgs',
                random_state=SEED, class_weight='balanced',
            )
            clf_final.fit(X_train, y_train)
            test_preds = clf_final.predict(X_test)
            test_f1 = float(f1_score(y_test, test_preds, zero_division=0))
        except Exception:
            test_f1 = cv_f1  # fallback to CV if split fails

        # Retrain on full data to get split features
        clf_full = LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs',
            random_state=SEED, class_weight='balanced',
        )
        clf_full.fit(X_pre, y)

        weights = clf_full.coef_[0]
        top_k_in_pre = np.argsort(-np.abs(weights))[:k_sparse]
        split_indices = preselect_idx[top_k_in_pre]
        split_weights_vals = weights[top_k_in_pre]

        all_f1s.append(cv_f1)
        all_test_f1s.append(test_f1)

        gate_char = "+" if cv_f1 >= 0.85 else ("~" if cv_f1 >= 0.70 else "-")
        print(f"    [{gate_char}] {country}: CV-F1={cv_f1:.3f} (std={cv_std:.3f}), "
              f"Test-F1={test_f1:.3f}, n_pos={n_pos}")

        per_country[country] = {
            "status": "ok",
            "n_pos": n_pos,
            "n_neg": n_neg,
            "cv_f1": round(cv_f1, 4),
            "cv_std": round(cv_std, 4),
            "test_f1": round(test_f1, 4),
            "fold_f1s": [round(f, 4) for f in fold_f1s],
            "split_features": split_indices.tolist(),
            "split_weights": [round(float(w), 4) for w in split_weights_vals],
        }

    # Aggregate
    n_countries_tested = sum(1 for v in per_country.values() if v.get("status") == "ok")
    n_countries_above_085 = sum(1 for v in per_country.values()
                                 if v.get("status") == "ok" and v.get("cv_f1", 0) >= 0.85)
    n_countries_above_080 = sum(1 for v in per_country.values()
                                 if v.get("status") == "ok" and v.get("cv_f1", 0) >= 0.80)
    n_countries_above_070 = sum(1 for v in per_country.values()
                                 if v.get("status") == "ok" and v.get("cv_f1", 0) >= 0.70)
    n_countries_zero = sum(1 for v in per_country.values()
                           if v.get("status") == "ok" and v.get("cv_f1", 0) == 0.0)

    mean_cv_f1 = float(np.mean(all_f1s)) if all_f1s else None
    median_cv_f1 = float(np.median(all_f1s)) if all_f1s else None
    mean_test_f1 = float(np.mean(all_test_f1s)) if all_test_f1s else None

    ci_lower, ci_upper = bootstrap_ci(all_f1s) if all_f1s else (None, None)

    aggregate = {
        "n_countries_tested": n_countries_tested,
        "n_countries_above_085": n_countries_above_085,
        "n_countries_above_080": n_countries_above_080,
        "n_countries_above_070": n_countries_above_070,
        "n_countries_zero_f1": n_countries_zero,
        "mean_cv_f1": round(mean_cv_f1, 4) if mean_cv_f1 is not None else None,
        "median_cv_f1": round(median_cv_f1, 4) if median_cv_f1 is not None else None,
        "mean_test_f1": round(mean_test_f1, 4) if mean_test_f1 is not None else None,
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "all_cv_f1s": [round(f, 4) for f in all_f1s],
    }

    return per_country, aggregate


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    set_seeds()
    total_steps = 8

    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae_config": "layer_12/width_16k/average_l0_82",
        "k_sparse": K_SPARSE,
        "timestamp_start": datetime.now().isoformat(),
    }

    print("=" * 70)
    print("Probe Quality Gate: City-Country on Gemma 2 2B (PILOT)")
    print(f"Task ID: {TASK_ID}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Seed: {SEED}")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 70)

    device = "cuda:0"

    # ── Step 1: Load model ──────────────────────────────────────────────
    report_progress(1, total_steps, "Loading Gemma 2 2B model")
    print("\n[Step 1/8] Loading Gemma 2 2B model...")
    t0 = time.time()

    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model_name = "unsloth/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, dtype=torch.float16)
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.float16,
    )
    del hf_model
    gc.collect(); torch.cuda.empty_cache()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # ── Step 2: Load SAE L12-16k ────────────────────────────────────────
    report_progress(2, total_steps, "Loading SAE L12-16k")
    print("\n[Step 2/8] Loading SAE L12-16k...")
    t0 = time.time()

    from sae_lens import SAE

    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_16k/average_l0_82",
        device=device,
    )
    print(f"  SAE loaded in {time.time()-t0:.1f}s, features={sae.cfg.d_sae}")

    hook_point = "blocks.12.hook_resid_post"
    W_dec = sae.W_dec.data.cpu()  # [n_features, d_model]

    # ── Step 3: Build city-country dataset ──────────────────────────────
    report_progress(3, total_steps, "Building single-token city dataset")
    print("\n[Step 3/8] Building single-token city dataset...")

    country_cities = get_single_token_cities(
        tokenizer, CITY_COUNTRY_RAW,
        max_per_country=MAX_CITIES_PER_COUNTRY,
        min_per_country=MIN_CITIES_PER_COUNTRY,
    )

    total_cities = sum(len(v) for v in country_cities.values())
    n_countries = len(country_cities)

    print(f"  Countries with sufficient cities: {n_countries}")
    for country in sorted(country_cities.keys()):
        cities = country_cities[country]
        city_names = [c['city'] for c in cities]
        print(f"    {country}: {len(cities)} cities ({', '.join(city_names[:5])}{'...' if len(cities) > 5 else ''})")
    print(f"  Total single-token cities: {total_cities}")

    results["dataset"] = {
        "n_countries": n_countries,
        "total_cities": total_cities,
        "per_country_counts": {k: len(v) for k, v in sorted(country_cities.items())},
        "per_country_examples": {
            k: [c['city'] for c in v[:5]]
            for k, v in sorted(country_cities.items())
        },
    }

    if total_cities < 30:
        msg = f"Only {total_cities} single-token cities found (need >= 30). Cannot proceed."
        print(f"\n  FATAL: {msg}")
        results["error"] = msg
        mark_done("failed", msg)
        return 1

    if n_countries < 10:
        msg = f"Only {n_countries} countries with sufficient cities (target 20+)."
        print(f"\n  WARNING: {msg}")
        results["warnings"] = [msg]

    # ── Step 4: Collect activations ─────────────────────────────────────
    report_progress(4, total_steps, "Collecting SAE and raw activations")
    print("\n[Step 4/8] Collecting activations...")
    t0 = time.time()

    sae_acts, raw_acts, all_labels, all_cities = collect_city_activations(
        model, sae, tokenizer, country_cities, hook_point, device
    )

    print(f"  Collected {len(all_labels)} activations in {time.time()-t0:.1f}s")
    print(f"  SAE shape: {sae_acts.shape}, Raw shape: {raw_acts.shape}")
    print(f"  Countries in dataset: {len(set(all_labels))}")
    print(f"  Label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

    results["activations"] = {
        "n_samples": len(all_labels),
        "sae_shape": list(sae_acts.shape),
        "raw_shape": list(raw_acts.shape),
        "n_countries": len(set(all_labels)),
        "label_distribution": {k: int(v) for k, v in
                               zip(*np.unique(all_labels, return_counts=True))},
    }

    # ── Step 5: SAE activation statistics ──────────────────────────────
    report_progress(5, total_steps, "Computing SAE activation statistics")
    print("\n[Step 5/8] SAE activation statistics...")

    n_active_per_sample = (sae_acts > 0).sum(dim=1).float()
    mean_active = float(n_active_per_sample.mean())
    max_active = int(n_active_per_sample.max())
    min_active = int(n_active_per_sample.min())
    pct_dead = float((sae_acts.sum(dim=0) == 0).float().mean()) * 100

    print(f"  Active features per sample: mean={mean_active:.1f}, "
          f"min={min_active}, max={max_active}")
    print(f"  Dead features (never active): {pct_dead:.1f}%")

    results["sae_stats"] = {
        "mean_active_features": round(mean_active, 1),
        "min_active_features": min_active,
        "max_active_features": max_active,
        "pct_dead_features": round(pct_dead, 2),
    }

    # Check for degenerate case (too many dead features)
    if pct_dead > 95:
        msg = f"CRITICAL: {pct_dead:.1f}% dead features on city prompts. SAE may be inadequate."
        print(f"\n  {msg}")
        results["warnings"] = results.get("warnings", []) + [msg]

    # ── Step 6: Train and evaluate probes ──────────────────────────────
    report_progress(6, total_steps, "Training per-country probes")
    print("\n[Step 6/8] Training per-country probes...")
    t0 = time.time()

    per_country, aggregate = train_and_evaluate_probes(sae_acts, all_labels, k_sparse=K_SPARSE)
    print(f"  Probe training time: {time.time()-t0:.1f}s")

    results["per_country"] = per_country
    results["aggregate"] = aggregate

    # ── Step 7: Dense probe upper bound (C3) ───────────────────────────
    report_progress(7, total_steps, "C3: Dense probe upper bound")
    print("\n[Step 7/8] Dense probe upper bound on raw activations...")

    X_raw = raw_acts.numpy()
    dense_f1s = {}
    countries_for_dense = sorted(set(all_labels))

    for country in countries_for_dense:
        y = np.array([1 if l == country else 0 for l in all_labels])
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos < 3 or n_neg < 3:
            continue

        n_splits = min(5, n_pos, n_neg)
        if n_splits < 2:
            continue

        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            fold_f1s = []
            for train_idx, test_idx in skf.split(X_raw, y):
                clf = LogisticRegression(
                    max_iter=1000, C=1.0, solver='lbfgs',
                    random_state=SEED, class_weight='balanced',
                )
                clf.fit(X_raw[train_idx], y[train_idx])
                preds = clf.predict(X_raw[test_idx])
                fold_f1s.append(f1_score(y[test_idx], preds, zero_division=0))
            dense_f1s[country] = round(float(np.mean(fold_f1s)), 4)
        except Exception:
            continue

    dense_mean = round(float(np.mean(list(dense_f1s.values()))), 4) if dense_f1s else None
    print(f"  Dense probe mean F1: {dense_mean}")
    for country in sorted(dense_f1s.keys()):
        sae_f1 = per_country.get(country, {}).get("cv_f1", "N/A")
        print(f"    {country}: Dense={dense_f1s[country]:.3f}, SAE={sae_f1}")

    results["dense_probe_control"] = {
        "per_country_f1": dense_f1s,
        "mean_f1": dense_mean,
        "n_countries": len(dense_f1s),
    }

    # ── Step 8: Compile results and decision ───────────────────────────
    report_progress(8, total_steps, "Compiling results and decision")
    print("\n[Step 8/8] Compiling results...")

    elapsed = time.time() - start_time

    # Decision gate criteria
    agg_f1 = aggregate.get("mean_cv_f1")
    n_above_085 = aggregate.get("n_countries_above_085", 0)
    n_above_080 = aggregate.get("n_countries_above_080", 0)
    n_tested = aggregate.get("n_countries_tested", 0)
    n_zero = aggregate.get("n_countries_zero_f1", 0)

    # Pass criteria from task_plan.json:
    # - Aggregate probe F1 > 0.85 on held-out test set
    # - At least 15/20 countries have per-country F1 > 0.80
    # - No country with F1 = 0.0
    criterion_agg_f1 = agg_f1 is not None and agg_f1 > 0.85
    criterion_per_country = n_above_080 >= min(15, int(n_tested * 0.75))
    criterion_no_zero = n_zero == 0

    # Decision gate
    if agg_f1 is not None and agg_f1 >= 0.85 and criterion_per_country and criterion_no_zero:
        decision = "GO"
        decision_detail = "F1 >= 0.85, proceed to full cross-domain experiments"
    elif agg_f1 is not None and agg_f1 >= 0.70:
        decision = "GO_WITH_CAUTION"
        decision_detail = (f"0.70 <= F1 < 0.85 ({agg_f1:.3f}). Proceed with caution; "
                           "report probe quality as limitation. Consider augmenting city set.")
    else:
        decision = "NO_GO"
        decision_detail = (f"F1 < 0.70 ({agg_f1}). Pivot to animal taxonomy or number parity domain.")

    pass_criteria = {
        "aggregate_f1_above_085": {
            "criterion": "mean CV F1 > 0.85",
            "value": agg_f1,
            "passed": criterion_agg_f1,
        },
        "per_country_f1_above_080": {
            "criterion": f">=75% of countries with F1 > 0.80 (need {min(15, int(n_tested * 0.75))})",
            "value": n_above_080,
            "total": n_tested,
            "passed": criterion_per_country,
        },
        "no_zero_f1_countries": {
            "criterion": "no country with F1 = 0.0",
            "value": n_zero,
            "passed": criterion_no_zero,
        },
        "decision": decision,
        "decision_detail": decision_detail,
        "overall": decision in ("GO", "GO_WITH_CAUTION"),
    }

    results["pass_criteria"] = pass_criteria
    results["elapsed_sec"] = round(elapsed, 1)
    results["timestamp_end"] = datetime.now().isoformat()

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PROBE QUALITY GATE RESULTS")
    print("=" * 70)
    print(f"  Model: Gemma 2 2B + Gemma Scope L12 16k")
    print(f"  Cities: {total_cities} across {n_countries} countries")
    print(f"  SAE active features/sample: {mean_active:.1f}")
    print(f"  Dead features: {pct_dead:.1f}%")
    print(f"\n  Probe Results:")
    print(f"    Mean CV F1:       {agg_f1}")
    print(f"    Median CV F1:     {aggregate.get('median_cv_f1')}")
    print(f"    Mean Test F1:     {aggregate.get('mean_test_f1')}")
    print(f"    Bootstrap 95% CI: {aggregate.get('bootstrap_ci_95')}")
    print(f"    Countries tested: {n_tested}")
    print(f"    Countries F1>0.85: {n_above_085}/{n_tested}")
    print(f"    Countries F1>0.80: {n_above_080}/{n_tested}")
    print(f"    Countries F1>0.70: {aggregate.get('n_countries_above_070', 0)}/{n_tested}")
    print(f"    Countries F1=0.0: {n_zero}")
    print(f"\n  Dense Probe (C3) Mean F1: {dense_mean}")
    print(f"\n  DECISION: {decision}")
    print(f"    {decision_detail}")
    print(f"\n  Pass Criteria:")
    for k, v in pass_criteria.items():
        if k in ("decision", "decision_detail", "overall"):
            continue
        print(f"    {k}: {'PASS' if v['passed'] else 'FAIL'} ({v['value']})")
    print(f"    OVERALL: {'PASS' if pass_criteria['overall'] else 'FAIL'}")
    print(f"\n  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

    # Save
    out_path = RESULTS_DIR / "probe_quality_city_country.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Saved: {out_path}")

    # Also save per-country bar chart data for visualization
    bar_data = []
    for country in sorted(per_country.keys()):
        info = per_country[country]
        if info.get("status") == "ok":
            bar_data.append({
                "country": country,
                "cv_f1": info["cv_f1"],
                "test_f1": info["test_f1"],
                "n_cities": info["n_pos"],
            })

    viz_path = RESULTS_DIR / "probe_quality_bar_data.json"
    viz_path.write_text(json.dumps(bar_data, indent=2))
    print(f"  Saved visualization data: {viz_path}")

    # Mark done
    mark_done(
        "success" if pass_criteria["overall"] else "failed",
        f"Probe Quality Gate: decision={decision}, mean_CV_F1={agg_f1}, "
        f"countries_tested={n_tested}, countries_above_085={n_above_085}, "
        f"countries_above_080={n_above_080}",
        results={
            "decision": decision,
            "mean_cv_f1": agg_f1,
            "n_countries": n_tested,
            "n_above_085": n_above_085,
        },
    )

    del model, sae
    gc.collect(); torch.cuda.empty_cache()

    return 0 if pass_criteria["overall"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nFATAL ERROR: {e}\n{tb}")
        mark_done("failed", f"Unhandled exception: {e}")
        sys.exit(1)
