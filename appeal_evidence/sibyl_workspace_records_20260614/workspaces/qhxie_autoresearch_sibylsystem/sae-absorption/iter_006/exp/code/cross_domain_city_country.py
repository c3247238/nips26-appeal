#!/usr/bin/env python3
"""
cross_domain_city_country.py — Cross-Domain Absorption: City -> Country (FULL)

Primary new domain for H1. Measures absorption rate on city-country hierarchy
using Gemma 2 2B + Gemma Scope SAEs.

Uses RAVEL-sourced single-token cities. Applies full Chanin et al. absorption
protocol with probe quality gate. Runs threshold sensitivity analysis (5x4 grid).
Includes shuffled country assignment control (C2) and random probe control (C1).
Runs on L12-16k and L12-65k.

MODE: FULL
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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
PILOT_RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
TASK_ID = "cross_domain_city_country"
SEED = 42

K_SPARSE = 5
# Cosine threshold calibrated for Gemma 2 2B d_model=2304
# Random baseline: mean=0.018, P95=0.053, P99=0.098
# Primary threshold: 0.1 (above P99 of random)
COSINE_THRESHOLD = 0.1
MAGNITUDE_GAP = 1.0
N_PRESELECT = 200  # Top features for probe training

# Threshold sensitivity grid
COSINE_THRESHOLDS = [0.01, 0.02, 0.025, 0.03, 0.05]
MAGNITUDE_GAPS = [0.5, 1.0, 1.5, 2.0]

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


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=SEED):
    np.random.seed(seed)
    if len(values) == 0:
        return None, None
    bs = [np.mean(np.random.choice(values, size=len(values), replace=True))
          for _ in range(n_bootstrap)]
    a = (1 - ci) / 2
    return (round(float(np.percentile(bs, a * 100)), 4),
            round(float(np.percentile(bs, (1 - a) * 100)), 4))


# ── City-Country Dataset ──────────────────────────────────────────────────
# Same curated dataset used in probe_quality_pilot, ensuring consistency
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
        "Guadalajara", "Monterrey", "Cancun", "Tijuana", "Oaxaca",
        "Puebla",
    ],
    "Argentina": [
        "Aires", "Cordoba", "Rosario", "Mendoza",
    ],
    "South Africa": [
        "Johannesburg", "Durban", "Pretoria",
    ],
    "Egypt": [
        "Cairo", "Alexandria", "Luxor",
    ],
    "Turkey": [
        "Istanbul", "Ankara", "Antalya", "Bursa",
    ],
    "Greece": [
        "Athens", "Thessaloniki", "Rhodes",
    ],
    "Thailand": [
        "Bangkok", "Phuket", "Pattaya", "Chiang",
    ],
    "Netherlands": [
        "Amsterdam", "Rotterdam", "Hague", "Utrecht", "Eindhoven",
        "Groningen", "Delft",
    ],
    "Sweden": [
        "Stockholm", "Gothenburg", "Malmo", "Uppsala",
    ],
    "Norway": [
        "Oslo", "Bergen", "Trondheim", "Stavanger",
    ],
    "Poland": [
        "Warsaw", "Krakow", "Gdansk", "Poznan",
    ],
    "Austria": [
        "Vienna", "Salzburg", "Innsbruck", "Graz", "Linz",
    ],
    "Switzerland": [
        "Zurich", "Geneva", "Bern", "Basel", "Lausanne",
    ],
    "Portugal": [
        "Lisbon", "Porto", "Faro", "Braga", "Coimbra",
    ],
    "Ireland": [
        "Dublin", "Cork", "Galway", "Limerick",
    ],
    "Indonesia": [
        "Jakarta", "Bali", "Surabaya", "Bandung", "Medan",
    ],
}


def get_single_token_cities(tokenizer, max_per_country=15, min_per_country=3):
    """Filter cities to those that tokenize as single tokens in Gemma 2."""
    country_cities = {}

    for country, candidates in CITY_COUNTRY_RAW.items():
        valid = []
        seen = set()
        for city in candidates:
            if city.lower() in seen:
                continue
            seen.add(city.lower())

            # Try the city name as-is and with space prefix
            for c in [city, city.lower()]:
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
    Collect SAE activations for each city using a simple prompt.
    Uses " [city]" prompt and extracts activation at the city token position.
    """
    all_sae_acts = []
    all_raw_acts = []
    all_labels = []
    all_cities = []

    for country in sorted(country_cities.keys()):
        cities = country_cities[country]
        for c in cities:
            prompt = f" {c['city']}"
            tokens = model.to_tokens(prompt, prepend_bos=True)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[hook_point],
                    return_type=None,
                )

            raw_act = cache[hook_point][0, -1, :].detach()  # [d_model]
            sae_act = sae.encode(raw_act.unsqueeze(0))[0].detach()  # [n_features]

            all_sae_acts.append(sae_act.cpu())
            all_raw_acts.append(raw_act.cpu())
            all_labels.append(country)
            all_cities.append(c['city'])

    sae_tensor = torch.stack(all_sae_acts)
    raw_tensor = torch.stack(all_raw_acts)
    return sae_tensor, raw_tensor, all_labels, all_cities


def analyze_absorption_per_country(
    sae_activations, labels, cities, W_dec,
    k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
    magnitude_gap=MAGNITUDE_GAP, verbose=True,
):
    """
    Full absorption analysis following Chanin et al. protocol, adapted for city-country.

    For each country:
    1. Train binary one-vs-rest logistic regression probe on SAE activations
    2. Identify "split features" = top-k features by absolute probe weight
    3. For correctly-classified positive tokens where ALL split features are inactive:
       these are false negatives (the SAE missed the country info)
    4. Among false negatives, check if any active feature absorbs the country info:
       - Its decoder direction aligns with the probe direction
       - cosine > threshold and activation > magnitude_gap relative to baseline
    """
    X = sae_activations.numpy()
    n_features = X.shape[1]
    countries = sorted(set(labels))

    per_country = {}
    all_probe_f1s = []
    total_tested = 0
    total_fn = 0
    total_absorbed = 0
    absorption_rates = []

    for country in countries:
        y = np.array([1 if l == country else 0 for l in labels])
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        if n_pos < 3 or n_neg < 3:
            per_country[country] = {"status": "skipped", "reason": f"n_pos={n_pos}", "n_pos": n_pos}
            continue

        n_splits = min(5, n_pos, n_neg)
        if n_splits < 2:
            per_country[country] = {"status": "skipped", "reason": "insufficient for CV", "n_pos": n_pos}
            continue

        # ── Feature pre-selection: top N_PRESELECT by mean difference ──
        mean_pos = X[y == 1].mean(axis=0)
        mean_neg = X[y == 0].mean(axis=0)
        feat_score = np.abs(mean_pos - mean_neg)
        preselect_idx = np.argsort(-feat_score)[:N_PRESELECT]
        X_pre = X[:, preselect_idx]

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

            probe_f1 = float(np.mean(fold_f1s))
            all_probe_f1s.append(probe_f1)
        except Exception as e:
            per_country[country] = {"status": "error", "error": str(e), "n_pos": n_pos}
            continue

        # ── Retrain on full data for analysis ──
        clf_full = LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs',
            random_state=SEED, class_weight='balanced',
        )
        clf_full.fit(X_pre, y)
        full_preds = clf_full.predict(X_pre)

        # ── Identify split features: top-k by absolute probe weight ──
        weights = clf_full.coef_[0]  # [N_PRESELECT]
        top_k_in_pre = np.argsort(-np.abs(weights))[:k_sparse]
        split_indices = preselect_idx[top_k_in_pre]
        split_weights = weights[top_k_in_pre]

        # Probe direction: weighted mean decoder direction of split features
        probe_direction = torch.zeros(W_dec.shape[1])
        for si, sw in zip(split_indices, split_weights):
            probe_direction += sw * W_dec[si]
        probe_direction = probe_direction / (probe_direction.norm() + 1e-8)

        # Baseline activation: mean activation of split features when they fire on positives
        pos_indices = np.where(y == 1)[0]
        active_vals = []
        for pi in pos_indices:
            for si in split_indices:
                v = sae_activations[pi, si].item()
                if v > 0:
                    active_vals.append(v)
        baseline_act = np.mean(active_vals) if active_vals else 1.0

        # ── Count false negatives and absorption ──
        n_tested_country = 0
        n_fn_country = 0
        n_abs_country = 0
        absorbed_examples = []

        for pi in pos_indices:
            if full_preds[pi] != 1:
                continue
            n_tested_country += 1

            split_acts = sae_activations[pi, split_indices]
            if (split_acts > 0).any().item():
                continue  # Split features cover this token

            # FALSE NEGATIVE: probe says yes but no split features fire
            n_fn_country += 1

            # ── Check absorption criteria ──
            active_mask = sae_activations[pi] > 0
            active_feat_indices = torch.where(active_mask)[0]

            absorbed = False
            absorbing_feats = []

            for fi in active_feat_indices:
                fi_int = fi.item()
                if fi_int in split_indices:
                    continue

                feat_dec = W_dec[fi_int]
                feat_norm = feat_dec.norm().item()
                if feat_norm < 1e-8:
                    continue

                cos_sim = torch.dot(feat_dec / feat_norm, probe_direction).item()
                feat_act = sae_activations[pi, fi_int].item()
                mag_ratio = feat_act / (baseline_act + 1e-8)

                if abs(cos_sim) > cosine_threshold and mag_ratio >= magnitude_gap:
                    absorbed = True
                    absorbing_feats.append({
                        "feature_idx": fi_int,
                        "cosine": round(cos_sim, 4),
                        "mag_ratio": round(mag_ratio, 4),
                        "activation": round(feat_act, 4),
                    })

            if absorbed:
                n_abs_country += 1
                absorbed_examples.append({
                    "city": cities[pi] if pi < len(cities) else "unknown",
                    "features": absorbing_feats[:3],
                })

        abs_rate = n_abs_country / n_tested_country if n_tested_country > 0 else float('nan')
        fn_rate = n_fn_country / n_tested_country if n_tested_country > 0 else float('nan')

        total_tested += n_tested_country
        total_fn += n_fn_country
        total_absorbed += n_abs_country

        if not np.isnan(abs_rate):
            absorption_rates.append(abs_rate)

        per_country[country] = {
            "status": "ok",
            "n_pos": n_pos,
            "n_tested": n_tested_country,
            "probe_f1": round(probe_f1, 4),
            "n_false_negatives": n_fn_country,
            "n_absorbed": n_abs_country,
            "absorption_rate": round(abs_rate, 4) if not np.isnan(abs_rate) else None,
            "false_negative_rate": round(fn_rate, 4) if not np.isnan(fn_rate) else None,
            "split_features": split_indices.tolist(),
            "split_weights": [round(float(w), 4) for w in split_weights],
            "absorbed_examples": absorbed_examples[:3],
        }

        if verbose:
            gate_char = "+" if probe_f1 > 0.85 else "-"
            print(f"    [{gate_char}] {country}: F1={probe_f1:.3f}, tested={n_tested_country}, "
                  f"FN={n_fn_country}, absorbed={n_abs_country}, "
                  f"rate={'N/A' if np.isnan(abs_rate) else f'{abs_rate:.3f}'}")

    # Aggregate
    agg_abs_rate = total_absorbed / total_tested if total_tested > 0 else None
    ci_lower, ci_upper = bootstrap_ci(absorption_rates) if absorption_rates else (None, None)
    countries_passing = sum(1 for r in per_country.values()
                           if r.get("status") == "ok" and r.get("probe_f1", 0) > 0.85)
    countries_tested = sum(1 for r in per_country.values() if r.get("status") == "ok")

    return {
        "per_country": per_country,
        "aggregate": {
            "total_tested": total_tested,
            "total_false_negatives": total_fn,
            "total_absorbed": total_absorbed,
            "aggregate_absorption_rate": round(agg_abs_rate, 4) if agg_abs_rate is not None else None,
            "aggregate_fn_rate": round(total_fn / total_tested, 4) if total_tested > 0 else None,
            "bootstrap_ci_95": [ci_lower, ci_upper],
            "mean_probe_f1": round(float(np.mean(all_probe_f1s)), 4) if all_probe_f1s else None,
            "median_probe_f1": round(float(np.median(all_probe_f1s)), 4) if all_probe_f1s else None,
            "countries_above_gate": countries_passing,
            "countries_tested": countries_tested,
        },
        "absorption_rates": [round(r, 4) for r in absorption_rates],
    }


def run_threshold_sensitivity(
    sae_activations, labels, cities, W_dec,
    k_sparse=K_SPARSE,
):
    """
    Run 5x4 threshold sensitivity grid.
    cosine: {0.01, 0.02, 0.025, 0.03, 0.05} x gap: {0.5, 1.0, 1.5, 2.0}
    """
    print("\n  Running threshold sensitivity analysis (5x4 grid)...")
    grid_results = {}

    for cos_th in COSINE_THRESHOLDS:
        for mag_gap in MAGNITUDE_GAPS:
            key = f"cos_{cos_th}_gap_{mag_gap}"
            result = analyze_absorption_per_country(
                sae_activations, labels, cities, W_dec,
                k_sparse=k_sparse, cosine_threshold=cos_th,
                magnitude_gap=mag_gap, verbose=False,
            )
            agg = result["aggregate"]
            rate = agg.get("aggregate_absorption_rate")
            grid_results[key] = {
                "cosine_threshold": cos_th,
                "magnitude_gap": mag_gap,
                "absorption_rate": rate,
                "total_tested": agg.get("total_tested"),
                "total_absorbed": agg.get("total_absorbed"),
                "total_fn": agg.get("total_false_negatives"),
            }
            print(f"    cos={cos_th}, gap={mag_gap}: rate={rate}")

    # Compute CV of absorption rates across grid
    rates = [v["absorption_rate"] for v in grid_results.values()
             if v["absorption_rate"] is not None]
    if rates and np.mean(rates) > 0:
        cv = round(float(np.std(rates) / np.mean(rates)), 4)
    else:
        cv = None

    return {
        "grid": grid_results,
        "cv": cv,
        "n_cells": len(grid_results),
        "rates_summary": {
            "min": round(min(rates), 4) if rates else None,
            "max": round(max(rates), 4) if rates else None,
            "mean": round(float(np.mean(rates)), 4) if rates else None,
            "std": round(float(np.std(rates)), 4) if rates else None,
        },
    }


def run_random_probe_control(sae_activations, labels, W_dec, k_sparse=K_SPARSE):
    """C1: Random probe control. Random features as split features -> expected ~0% absorption."""
    print("\n  Running C1: Random probe control...")
    np.random.seed(SEED + 100)

    n_features = sae_activations.shape[1]
    total_fn = 0
    total_tested = 0
    total_absorbed = 0

    for country in sorted(set(labels)):
        y = np.array([1 if l == country else 0 for l in labels])
        n_pos = int(y.sum())
        if n_pos < 3:
            continue

        # Random split features
        random_split = np.random.choice(n_features, size=k_sparse, replace=False)
        random_direction = W_dec[random_split].mean(dim=0)
        random_direction = random_direction / (random_direction.norm() + 1e-8)

        pos_indices = np.where(y == 1)[0]
        for pi in pos_indices:
            total_tested += 1
            split_acts = sae_activations[pi, random_split]
            if (split_acts > 0).any().item():
                continue
            total_fn += 1

            active_mask = sae_activations[pi] > 0
            active_feat_indices = torch.where(active_mask)[0]
            for fi in active_feat_indices:
                fi_int = fi.item()
                feat_dec = W_dec[fi_int]
                fn_v = feat_dec.norm().item()
                if fn_v < 1e-8:
                    continue
                cos = torch.dot(feat_dec / fn_v, random_direction).item()
                if abs(cos) > COSINE_THRESHOLD:
                    total_absorbed += 1
                    break

    abs_rate = total_absorbed / total_tested if total_tested > 0 else 0
    fn_rate = total_fn / total_tested if total_tested > 0 else 0
    print(f"    Random probe: tested={total_tested}, FN={total_fn} ({fn_rate:.3f}), "
          f"absorbed={total_absorbed} ({abs_rate:.3f})")
    return {
        "control_type": "C1_random_probe",
        "total_tested": total_tested,
        "total_false_negatives": total_fn,
        "false_negative_rate": round(fn_rate, 4),
        "total_absorbed": total_absorbed,
        "absorption_rate": round(abs_rate, 4),
    }


def run_shuffled_label_control(sae_activations, labels, cities, W_dec, k_sparse=K_SPARSE):
    """C2: Shuffled label control. Random city-country assignment -> expected ~0%."""
    print("\n  Running C2: Shuffled label control...")
    np.random.seed(SEED + 200)

    X = sae_activations.numpy()
    shuffled_labels = list(labels)
    random.seed(SEED + 200)
    random.shuffle(shuffled_labels)

    total_absorbed = 0
    total_tested = 0

    for country in sorted(set(shuffled_labels)):
        y = np.array([1 if l == country else 0 for l in shuffled_labels])
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos < 3 or n_neg < 3:
            continue

        try:
            mean_pos_s = X[y == 1].mean(axis=0)
            mean_neg_s = X[y == 0].mean(axis=0)
            pre_idx_s = np.argsort(-np.abs(mean_pos_s - mean_neg_s))[:N_PRESELECT]
            X_pre_s = X[:, pre_idx_s]

            clf = LogisticRegression(
                max_iter=1000, C=1.0, solver='lbfgs',
                random_state=SEED, class_weight='balanced',
            )
            clf.fit(X_pre_s, y)
            preds = clf.predict(X_pre_s)
        except Exception:
            continue

        weights = clf.coef_[0]
        top_k_pre_s = np.argsort(-np.abs(weights))[:k_sparse]
        split_idx = pre_idx_s[top_k_pre_s]

        probe_dir = torch.zeros(W_dec.shape[1])
        for si, wi in zip(split_idx, top_k_pre_s):
            probe_dir += weights[wi] * W_dec[si]
        probe_dir = probe_dir / (probe_dir.norm() + 1e-8)

        pos_indices = np.where(y == 1)[0]
        for pi in pos_indices:
            if preds[pi] != 1:
                continue
            total_tested += 1

            split_acts = sae_activations[pi, split_idx]
            if (split_acts > 0).any().item():
                continue

            active_mask = sae_activations[pi] > 0
            active_feat_indices = torch.where(active_mask)[0]
            for fi in active_feat_indices:
                fi_int = fi.item()
                feat_dec = W_dec[fi_int]
                fn_v = feat_dec.norm().item()
                if fn_v < 1e-8:
                    continue
                cos = torch.dot(feat_dec / fn_v, probe_dir).item()
                if abs(cos) > COSINE_THRESHOLD:
                    total_absorbed += 1
                    break

    rate = total_absorbed / total_tested if total_tested > 0 else 0
    print(f"    Shuffled labels: tested={total_tested}, absorbed={total_absorbed} ({rate:.3f})")
    return {
        "control_type": "C2_shuffled_labels",
        "total_tested": total_tested,
        "total_absorbed": total_absorbed,
        "absorption_rate": round(rate, 4),
    }


def run_dense_probe_control(raw_activations, labels):
    """C3: Dense probe on raw model activations (upper bound)."""
    print("\n  Running C3: Dense probe upper bound...")

    X_raw = raw_activations.numpy()
    dense_f1s = {}

    for country in sorted(set(labels)):
        y = np.array([1 if l == country else 0 for l in labels])
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

    mean_f1 = round(float(np.mean(list(dense_f1s.values()))), 4) if dense_f1s else None
    print(f"    Dense probe mean F1: {mean_f1}")
    return {
        "control_type": "C3_dense_probe",
        "per_country_f1": dense_f1s,
        "mean_f1": mean_f1,
        "n_countries": len(dense_f1s),
    }


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    set_seeds()
    total_steps = 12

    results = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": SEED,
        "model": "gemma-2-2b",
        "timestamp_start": datetime.now().isoformat(),
    }

    print("=" * 70)
    print("Cross-Domain Absorption: City -> Country (FULL)")
    print(f"Task ID: {TASK_ID}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Seed: {SEED}")
    print(f"Cosine threshold: {COSINE_THRESHOLD}")
    print(f"Magnitude gap: {MAGNITUDE_GAP}")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 70)

    device = "cuda:0"

    # ── Step 1: Load model ──────────────────────────────────────────────
    report_progress(1, total_steps, "Loading Gemma 2 2B model")
    print("\n[Step 1/12] Loading Gemma 2 2B model...")
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
    print("\n[Step 2/12] Loading SAE L12-16k...")
    t0 = time.time()

    from sae_lens import SAE

    sae_16k = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_16k/average_l0_82",
        device=device,
    )
    print(f"  SAE loaded in {time.time()-t0:.1f}s, features={sae_16k.cfg.d_sae}")

    hook_point = "blocks.12.hook_resid_post"
    W_dec_16k = sae_16k.W_dec.data.cpu()  # [n_features, d_model]

    # ── Step 3: Build city-country dataset ─────────────────────────────
    report_progress(3, total_steps, "Building city-country dataset")
    print("\n[Step 3/12] Building single-token city-country dataset...")

    country_cities = get_single_token_cities(tokenizer)

    total_cities = sum(len(v) for v in country_cities.values())
    print(f"  Countries with enough cities: {len(country_cities)}")
    print(f"  Total single-token cities: {total_cities}")
    for country in sorted(country_cities.keys()):
        c = country_cities[country]
        print(f"    {country}: {len(c)} cities (e.g., {', '.join(x['city'] for x in c[:5])})")

    results["dataset"] = {
        "n_countries": len(country_cities),
        "total_cities": total_cities,
        "per_country_counts": {k: len(v) for k, v in sorted(country_cities.items())},
        "per_country_examples": {k: [x['city'] for x in v[:5]]
                                  for k, v in sorted(country_cities.items())},
    }

    # ── Step 4: Collect activations (L12-16k) ──────────────────────────
    report_progress(4, total_steps, "Collecting SAE and raw activations (L12-16k)")
    print("\n[Step 4/12] Collecting activations (L12-16k)...")
    t0 = time.time()

    sae_acts_16k, raw_acts, all_labels, all_cities = collect_city_activations(
        model, sae_16k, tokenizer, country_cities, hook_point, device
    )

    print(f"  Collected {len(all_labels)} activations in {time.time()-t0:.1f}s")
    print(f"  SAE shape: {sae_acts_16k.shape}, Raw shape: {raw_acts.shape}")

    results["activations"] = {
        "n_samples": len(all_labels),
        "sae_shape": list(sae_acts_16k.shape),
        "raw_shape": list(raw_acts.shape),
        "n_countries": len(set(all_labels)),
        "label_distribution": {c: all_labels.count(c) for c in sorted(set(all_labels))},
    }

    # SAE stats
    mean_active = sae_acts_16k.gt(0).sum(dim=1).float().mean().item()
    pct_dead = (sae_acts_16k.gt(0).any(dim=0).float().mean().item()) * 100
    results["sae_stats_16k"] = {
        "mean_active_features": round(mean_active, 1),
        "pct_live_features": round(pct_dead, 2),
        "pct_dead_features": round(100 - pct_dead, 2),
    }
    print(f"  SAE stats: mean_active={mean_active:.1f}, live_features={pct_dead:.1f}%")

    # ── Step 5: Main absorption analysis (L12-16k) ─────────────────────
    report_progress(5, total_steps, "Running absorption analysis L12-16k")
    print("\n[Step 5/12] Absorption analysis on L12-16k...")
    t0 = time.time()

    l12_16k_results = analyze_absorption_per_country(
        sae_acts_16k, all_labels, all_cities, W_dec_16k,
        k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
        magnitude_gap=MAGNITUDE_GAP,
    )

    agg = l12_16k_results["aggregate"]
    print(f"\n  L12-16k Summary:")
    print(f"    Aggregate absorption rate: {agg['aggregate_absorption_rate']}")
    print(f"    Aggregate FN rate: {agg['aggregate_fn_rate']}")
    print(f"    CI 95%: {agg['bootstrap_ci_95']}")
    print(f"    Mean probe F1: {agg['mean_probe_f1']}")
    print(f"    Countries passing gate: {agg['countries_above_gate']}/{agg['countries_tested']}")
    print(f"    Time: {time.time()-t0:.1f}s")

    results["l12_16k"] = {
        "sae_id": "layer_12/width_16k/average_l0_82",
        "k_sparse": K_SPARSE,
        "cosine_threshold": COSINE_THRESHOLD,
        "magnitude_gap": MAGNITUDE_GAP,
        **l12_16k_results,
    }

    # ── Step 6: Threshold sensitivity analysis ─────────────────────────
    report_progress(6, total_steps, "Threshold sensitivity analysis (5x4 grid)")
    print("\n[Step 6/12] Threshold sensitivity analysis...")
    t0 = time.time()

    sensitivity = run_threshold_sensitivity(
        sae_acts_16k, all_labels, all_cities, W_dec_16k,
        k_sparse=K_SPARSE,
    )

    print(f"\n  Sensitivity Summary:")
    print(f"    CV: {sensitivity['cv']}")
    print(f"    Rate range: {sensitivity['rates_summary']['min']}-{sensitivity['rates_summary']['max']}")
    print(f"    Time: {time.time()-t0:.1f}s")

    results["threshold_sensitivity"] = sensitivity

    # ── Step 7: C1 Random probe control ─────────────────────────────────
    report_progress(7, total_steps, "C1: Random probe control")
    c1 = run_random_probe_control(sae_acts_16k, all_labels, W_dec_16k)
    results["controls"] = {"C1_random_probe": c1}

    # ── Step 8: C2 Shuffled label control ───────────────────────────────
    report_progress(8, total_steps, "C2: Shuffled label control")
    c2 = run_shuffled_label_control(sae_acts_16k, all_labels, all_cities, W_dec_16k)
    results["controls"]["C2_shuffled_labels"] = c2

    # ── Step 9: C3 Dense probe upper bound ──────────────────────────────
    report_progress(9, total_steps, "C3: Dense probe upper bound")
    c3 = run_dense_probe_control(raw_acts, all_labels)
    results["controls"]["C3_dense_probe"] = c3

    # ── Step 10: L12-65k width comparison ───────────────────────────────
    report_progress(10, total_steps, "L12-65k width comparison")
    print("\n[Step 10/12] Width comparison: L12-65k...")
    t0 = time.time()

    try:
        del sae_16k
        gc.collect(); torch.cuda.empty_cache()

        sae_65k = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res",
            sae_id="layer_12/width_65k/average_l0_72",
            device=device,
        )
        W_dec_65k = sae_65k.W_dec.data.cpu()
        print(f"  SAE L12-65k loaded, features={sae_65k.cfg.d_sae}")

        # Collect 65k activations
        acts_65k = []
        for city_name in all_cities:
            prompt = f" {city_name}"
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=[hook_point], return_type=None,
                )
            act = cache[hook_point][0, -1, :].detach()
            sae_act = sae_65k.encode(act.unsqueeze(0))[0].detach()
            acts_65k.append(sae_act.cpu())

        acts_65k_tensor = torch.stack(acts_65k)
        print(f"  65k activations collected: {acts_65k_tensor.shape}")

        l12_65k_results = analyze_absorption_per_country(
            acts_65k_tensor, all_labels, all_cities, W_dec_65k,
            k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
            magnitude_gap=MAGNITUDE_GAP,
        )

        agg_65k = l12_65k_results["aggregate"]
        print(f"\n  L12-65k Summary:")
        print(f"    Aggregate absorption rate: {agg_65k['aggregate_absorption_rate']}")
        print(f"    Mean probe F1: {agg_65k['mean_probe_f1']}")
        print(f"    Countries passing gate: {agg_65k['countries_above_gate']}/{agg_65k['countries_tested']}")

        # SAE stats for 65k
        mean_active_65k = acts_65k_tensor.gt(0).sum(dim=1).float().mean().item()
        pct_live_65k = (acts_65k_tensor.gt(0).any(dim=0).float().mean().item()) * 100

        results["l12_65k"] = {
            "sae_id": "layer_12/width_65k/average_l0_72",
            "k_sparse": K_SPARSE,
            "cosine_threshold": COSINE_THRESHOLD,
            "magnitude_gap": MAGNITUDE_GAP,
            "sae_stats": {
                "mean_active_features": round(mean_active_65k, 1),
                "pct_live_features": round(pct_live_65k, 2),
            },
            **l12_65k_results,
        }

        del sae_65k
        gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        tb = traceback.format_exc()
        results["l12_65k"] = {"status": "error", "error": str(e), "traceback": tb}
        print(f"  L12-65k failed: {e}")

    print(f"  Time: {time.time()-t0:.1f}s")

    # ── Step 11: Cross-reference with first-letter baseline ─────────────
    report_progress(11, total_steps, "Cross-referencing with first-letter baseline")
    print("\n[Step 11/12] Cross-referencing with first-letter baseline...")

    fl_path = PILOT_RESULTS_DIR / "first_letter_validation.json"
    if fl_path.exists():
        try:
            fl_data = json.loads(fl_path.read_text())
            fl_16k = fl_data.get("l12_16k", {}).get("aggregate", {})
            fl_rate = fl_16k.get("aggregate_absorption_rate")
            cc_rate = agg.get("aggregate_absorption_rate")

            results["cross_reference"] = {
                "first_letter_absorption_rate": fl_rate,
                "city_country_absorption_rate": cc_rate,
                "ratio": round(cc_rate / fl_rate, 4) if fl_rate and cc_rate and fl_rate > 0 else None,
                "first_letter_mean_probe_f1": fl_16k.get("mean_probe_f1"),
                "city_country_mean_probe_f1": agg.get("mean_probe_f1"),
            }
            print(f"  First-letter absorption rate: {fl_rate}")
            print(f"  City-country absorption rate: {cc_rate}")
            if fl_rate and cc_rate and fl_rate > 0:
                print(f"  Ratio (city-country / first-letter): {cc_rate / fl_rate:.2f}")
        except Exception as e:
            results["cross_reference"] = {"status": "error", "error": str(e)}
            print(f"  Cross-reference failed: {e}")
    else:
        results["cross_reference"] = {"status": "file_not_found"}
        print(f"  First-letter validation results not found at {fl_path}")

    # ── Step 12: Compile and save ───────────────────────────────────────
    report_progress(12, total_steps, "Compiling results")
    print("\n[Step 12/12] Compiling results...")

    elapsed = time.time() - start_time
    results["elapsed_sec"] = round(elapsed, 1)
    results["timestamp_end"] = datetime.now().isoformat()

    # Pass criteria evaluation
    l12_agg = results.get("l12_16k", {}).get("aggregate", {})
    countries_gate = l12_agg.get("countries_above_gate", 0)
    countries_total = l12_agg.get("countries_tested", 0)
    cc_abs_rate = l12_agg.get("aggregate_absorption_rate")
    c1_rate = results.get("controls", {}).get("C1_random_probe", {}).get("absorption_rate", 1.0)
    c2_rate = results.get("controls", {}).get("C2_shuffled_labels", {}).get("absorption_rate", 1.0)
    sens_cv = results.get("threshold_sensitivity", {}).get("cv")

    results["pass_criteria"] = {
        "probe_f1_above_085": {
            "criterion": ">=15/20 countries with probe F1 > 0.85",
            "value": countries_gate,
            "total": countries_total,
            "passed": countries_gate >= 15,
        },
        "absorption_measurable": {
            "criterion": "absorption rate measurable (non-NaN) for at least 10 countries",
            "value": sum(1 for r in l12_agg.get("bootstrap_ci_95", [None, None]) or [None, None]
                         if r is not None),
            "n_measurable": len(results.get("l12_16k", {}).get("absorption_rates", [])),
            "passed": len(results.get("l12_16k", {}).get("absorption_rates", [])) >= 10,
        },
        "shuffled_control": {
            "criterion": "shuffled control yields < 5%",
            "value": c2_rate,
            "passed": c2_rate < 0.05,
        },
        "threshold_sensitivity_cv": {
            "criterion": "threshold sensitivity CV < 0.5",
            "value": sens_cv,
            "passed": sens_cv is not None and sens_cv < 0.5,
        },
        "H1_test": {
            "criterion": "absorption rate >= 10% (H1 success) or < 5% (H1 falsified)",
            "value": cc_abs_rate,
            "verdict": ("SUPPORTED" if cc_abs_rate is not None and cc_abs_rate >= 0.10
                        else "PARTIALLY_SUPPORTED" if cc_abs_rate is not None and cc_abs_rate >= 0.05
                        else "FALSIFIED" if cc_abs_rate is not None
                        else "INCONCLUSIVE"),
        },
    }

    # Save to full results
    output_path = RESULTS_DIR / "cross_domain_city_country.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    # Also save to pilots dir for backward compatibility
    pilot_path = PILOT_RESULTS_DIR / "cross_domain_city_country.json"
    with open(pilot_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE: Cross-Domain City-Country Absorption")
    print(f"  Duration: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Aggregate absorption rate (L12-16k): {cc_abs_rate}")
    print(f"  CI 95%: {l12_agg.get('bootstrap_ci_95')}")
    print(f"  Countries passing probe gate: {countries_gate}/{countries_total}")
    print(f"  Shuffled control rate: {c2_rate}")
    print(f"  Threshold sensitivity CV: {sens_cv}")
    print(f"  H1 verdict: {results['pass_criteria']['H1_test']['verdict']}")
    print("=" * 70)

    # Update gpu_progress
    update_gpu_progress(elapsed, start_time)

    # Write DONE marker
    mark_done(
        status="success",
        summary=(f"City-country absorption rate={cc_abs_rate}, "
                 f"probe gate={countries_gate}/{countries_total}, "
                 f"shuffled control={c2_rate}, CV={sens_cv}"),
        results={
            "absorption_rate_16k": cc_abs_rate,
            "absorption_rate_65k": results.get("l12_65k", {}).get("aggregate", {}).get("aggregate_absorption_rate"),
            "countries_passing_gate": countries_gate,
            "H1_verdict": results["pass_criteria"]["H1_test"]["verdict"],
        },
    )


def update_gpu_progress(elapsed, start_time):
    """Update gpu_progress.json with timing and completion."""
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        if gp_path.exists():
            gp = json.loads(gp_path.read_text())
        else:
            gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

        # Add to completed
        if TASK_ID not in gp.get("completed", []):
            gp.setdefault("completed", []).append(TASK_ID)

        # Remove from running
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]

        # Record timing
        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 45,
            "actual_min": round(elapsed / 60),
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {
                "model": "gemma-2-2b",
                "sae_configs": ["L12-16k", "L12-65k"],
                "n_cities": "see results",
                "k_sparse": K_SPARSE,
                "cosine_threshold": COSINE_THRESHOLD,
                "magnitude_gap": MAGNITUDE_GAP,
                "gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "gpu_count": 1,
            },
        }

        gp_path.write_text(json.dumps(gp, indent=2))
        print(f"  gpu_progress.json updated")
    except Exception as e:
        print(f"  Warning: failed to update gpu_progress.json: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n!!! EXPERIMENT FAILED !!!\n{tb}")

        # Save error results
        error_results = {
            "task_id": TASK_ID,
            "status": "failed",
            "error": str(e),
            "traceback": tb,
            "timestamp": datetime.now().isoformat(),
        }
        output_path = RESULTS_DIR / "cross_domain_city_country.json"
        with open(output_path, 'w') as f:
            json.dump(error_results, f, indent=2)

        # Update gpu_progress with failure
        gp_path = WORKSPACE / "exp" / "gpu_progress.json"
        try:
            if gp_path.exists():
                gp = json.loads(gp_path.read_text())
            else:
                gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
            if TASK_ID not in gp.get("failed", []):
                gp.setdefault("failed", []).append(TASK_ID)
            if TASK_ID in gp.get("running", {}):
                del gp["running"][TASK_ID]
            gp_path.write_text(json.dumps(gp, indent=2))
        except: pass

        mark_done(status="failed", summary=str(e))
        sys.exit(1)
