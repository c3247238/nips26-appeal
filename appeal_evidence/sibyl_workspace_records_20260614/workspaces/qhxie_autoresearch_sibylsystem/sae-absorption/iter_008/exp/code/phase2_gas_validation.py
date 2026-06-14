"""
Phase 2.2: GAS Validation Against Probe-Based Absorption Rates — PILOT mode

Validates GAS (Geometric Absorption Score) against probe-based absorption rates
(ground truth from Phase 1.2). Target: Spearman rho >= 0.6.

Pipeline:
1. Load GAS scores from Phase 2.1 (per-feature vulnerability scores from .npz + .json)
2. Load probe-based absorption rates from Phase 1.2 (per-letter absorption status)
3. Map GAS vulnerability to per-letter "main features" identified by probe-decoder cosine
4. Compute Spearman rank correlation between GAS and probe-based absorption
5. Bootstrap 95% CI on Spearman rho (10k resamples)
6. Receiver operating characteristic: AUROC of GAS as absorption classifier
7. Discovery mode: rank all features by GAS, examine top-50 candidates NOT in first-letter set
8. If rho < 0.3: GAS fails as detector. Report as negative result.

Pilot pass criteria:
- Spearman rho computed between GAS and probe-based absorption on at least one SAE config
- Bootstrap CI computed
- AUROC of GAS as classifier computed
- If rho < 0.3: document as negative result with analysis
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────────

TASK_ID = "phase2_gas_validation"
SEED = 42
PILOT_MODE = True

# Paths
WORKSPACE = Path(os.environ.get("WORKSPACE",
    "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current"))
RESULTS_DIR = WORKSPACE / "exp" / "results"
PHASE2_DIR = RESULTS_DIR / "phase2"
PILOTS_DIR = RESULTS_DIR / "pilots"
PHASE2_DIR.mkdir(parents=True, exist_ok=True)

# Input files
GAS_RESULTS_FILE = PHASE2_DIR / "gas_computation.json"
GAS_NPZ_FILE = PHASE2_DIR / "per_feature_gas_vulnerability.npz"
ABSORPTION_FILE = PILOTS_DIR / "phase1_absorption_firstletter.json"

# ── Reproducibility ──────────────────────────────────────────────────────────

np.random.seed(SEED)

# ── PID file for system recovery ──────────────────────────────────────────────

pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor to track."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file for system monitor to detect."""
    pid_f = Path(results_dir) / f"{task_id}.pid"
    if pid_f.exists():
        pid_f.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def update_gpu_progress(workspace, task_id, status, start_time, config_snapshot=None):
    """Update gpu_progress.json with task completion info."""
    gp_path = workspace / "exp" / "gpu_progress.json"
    if gp_path.exists():
        gp = json.loads(gp_path.read_text())
    else:
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    end_time = datetime.now().isoformat()
    elapsed_min = round((datetime.now() - datetime.fromisoformat(start_time)).total_seconds() / 60)

    if status == "success":
        if task_id not in gp.get("completed", []):
            gp.setdefault("completed", []).append(task_id)
    else:
        if task_id not in gp.get("failed", []):
            gp.setdefault("failed", []).append(task_id)

    # Remove from running
    gp.setdefault("running", {}).pop(task_id, None)

    # Record timing
    gp.setdefault("timings", {})[task_id] = {
        "planned_min": 20,
        "actual_min": elapsed_min,
        "start_time": start_time,
        "end_time": end_time,
        "config_snapshot": config_snapshot or {}
    }

    gp_path.write_text(json.dumps(gp, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    start_time = datetime.now().isoformat()
    print(f"[{TASK_ID}] Starting GAS validation (PILOT mode)")
    print(f"[{TASK_ID}] Workspace: {WORKSPACE}")

    report_progress(TASK_ID, RESULTS_DIR, 0, 7, step=0, total_steps=7,
                    metric={"status": "loading_data"})

    # ── Step 1: Load GAS data ────────────────────────────────────────────────

    print("\n[Step 1] Loading GAS computation results...")

    if not GAS_RESULTS_FILE.exists():
        raise FileNotFoundError(f"GAS computation results not found: {GAS_RESULTS_FILE}")
    if not ABSORPTION_FILE.exists():
        raise FileNotFoundError(f"Absorption results not found: {ABSORPTION_FILE}")

    gas_results = json.loads(GAS_RESULTS_FILE.read_text())

    # Load per-feature vulnerability scores from NPZ if available
    per_feature_gas = None
    if GAS_NPZ_FILE.exists():
        npz_data = np.load(GAS_NPZ_FILE)
        # Try multiple possible key names
        for key in ["per_feature_gas", "vulnerability", "gas_vulnerability"]:
            if key in npz_data:
                per_feature_gas = npz_data[key]
                print(f"  Loaded per-feature GAS from NPZ (key='{key}'): shape={per_feature_gas.shape}")
                break
        if per_feature_gas is not None:
            print(f"  Non-zero features: {np.count_nonzero(per_feature_gas)}")
            print(f"  Max: {per_feature_gas.max():.4f}, Mean (non-zero): {per_feature_gas[per_feature_gas > 0].mean():.4f}")
        # Also load frequency data if available
        feature_freq = npz_data.get("feature_freq_rate", None)
        if feature_freq is not None:
            print(f"  Also loaded feature frequency rates: shape={feature_freq.shape}")

    # If NPZ not available, reconstruct from JSON top_vulnerable_features
    if per_feature_gas is None:
        print("  NPZ not available, reconstructing from JSON...")
        n_features = gas_results["sae_config"]["n_features"]
        per_feature_gas = np.zeros(n_features)

        # Use top_vulnerable_features
        for entry in gas_results.get("top_vulnerable_features", []):
            fid = entry["feature_idx"]
            score = entry["gas_vulnerability"]
            per_feature_gas[fid] = score

        # Also extract from top_pairs_by_gas for completeness
        for entry in gas_results.get("top_pairs_by_gas", []):
            fi, fj = entry["feature_i"], entry["feature_j"]
            gas_ij = entry["gas_i_absorbs_j"]
            gas_ji = entry["gas_j_absorbs_i"]
            # Per-feature vulnerability = max GAS(any_i -> this_feature)
            per_feature_gas[fj] = max(per_feature_gas[fj], gas_ij)
            per_feature_gas[fi] = max(per_feature_gas[fi], gas_ji)

        print(f"  Reconstructed per-feature GAS: {np.count_nonzero(per_feature_gas)} non-zero features")

    # Also extract from all_pairs in the JSON for a richer reconstruction
    # but this may be very large; the NPZ is preferred
    all_pairs = gas_results.get("top_pairs_by_gas", [])
    print(f"  Total high-similarity pairs in JSON: {len(all_pairs)}")

    report_progress(TASK_ID, RESULTS_DIR, 1, 7, step=1, total_steps=7,
                    metric={"status": "gas_loaded", "n_features": len(per_feature_gas)})

    # ── Step 2: Load absorption data ─────────────────────────────────────────

    print("\n[Step 2] Loading probe-based absorption results...")

    absorption_data = json.loads(ABSORPTION_FILE.read_text())

    # Extract per-letter absorption rates and main features
    abs_results = absorption_data["absorption_results"]["L12_16k"]
    per_letter = abs_results["per_letter"]
    main_features = abs_results["main_features_top"]

    print(f"  SAE config: L12 16k JumpReLU")
    print(f"  Overall absorption rate: {abs_results['absorption_rate']:.4f}")
    print(f"  Letters with data: {len([l for l in per_letter if per_letter[l].get('total', 0) > 0])}")

    # Build letter-level data: letter -> (absorption_rate, main_feature_id, main_feature_cos)
    letter_data = {}
    for letter in sorted(per_letter.keys()):
        ld = per_letter[letter]
        if ld.get("total", 0) == 0:
            continue  # skip letters with no data (e.g., 'x')

        mf = main_features.get(letter, {})
        fid = mf.get("fid")
        cos = mf.get("cos", 0.0)
        abs_rate = ld.get("absorption_rate", 0.0)
        n_fn = ld.get("false_negatives", 0)
        n_total = ld.get("probe_correct_raw", 0)

        if fid is not None:
            letter_data[letter] = {
                "absorption_rate": abs_rate,
                "strict_rate": ld.get("strict_rate", 0.0),
                "main_feature_id": fid,
                "main_feature_cos": cos,
                "n_false_negatives": n_fn,
                "n_probe_correct": n_total,
                "gas_vulnerability": float(per_feature_gas[fid]) if fid < len(per_feature_gas) else 0.0,
            }

    print(f"  Letters with main features identified: {len(letter_data)}")
    print(f"  Letters with non-zero absorption: {sum(1 for v in letter_data.values() if v['absorption_rate'] > 0)}")

    report_progress(TASK_ID, RESULTS_DIR, 2, 7, step=2, total_steps=7,
                    metric={"status": "absorption_loaded", "n_letters": len(letter_data)})

    # ── Step 3: Compute Spearman rank correlation ────────────────────────────

    print("\n[Step 3] Computing Spearman rank correlation (GAS vs absorption rate)...")

    letters = sorted(letter_data.keys())
    gas_scores = np.array([letter_data[l]["gas_vulnerability"] for l in letters])
    abs_rates = np.array([letter_data[l]["absorption_rate"] for l in letters])
    strict_rates = np.array([letter_data[l]["strict_rate"] for l in letters])
    main_fids = [letter_data[l]["main_feature_id"] for l in letters]
    main_cos = np.array([letter_data[l]["main_feature_cos"] for l in letters])

    print(f"\n  Per-letter GAS vulnerability of main features:")
    print(f"  {'Letter':>6} {'MainFID':>8} {'Cos':>6} {'AbsRate':>8} {'StrictRate':>10} {'GAS':>10}")
    for i, letter in enumerate(letters):
        print(f"  {letter:>6} {main_fids[i]:>8} {main_cos[i]:>6.3f} {abs_rates[i]:>8.4f} {strict_rates[i]:>10.4f} {gas_scores[i]:>10.4f}")

    # Spearman correlation: GAS vs absorption rate
    if np.std(abs_rates) > 0 and np.std(gas_scores) > 0:
        rho_absorption, p_absorption = stats.spearmanr(gas_scores, abs_rates)
    else:
        rho_absorption, p_absorption = 0.0, 1.0

    # Spearman correlation: GAS vs strict absorption rate
    if np.std(strict_rates) > 0 and np.std(gas_scores) > 0:
        rho_strict, p_strict = stats.spearmanr(gas_scores, strict_rates)
    else:
        rho_strict, p_strict = 0.0, 1.0

    # Also compute Pearson for comparison
    if np.std(abs_rates) > 0 and np.std(gas_scores) > 0:
        pearson_r, pearson_p = stats.pearsonr(gas_scores, abs_rates)
    else:
        pearson_r, pearson_p = 0.0, 1.0

    print(f"\n  Spearman rho (GAS vs absorption):        {rho_absorption:.4f} (p={p_absorption:.4f})")
    print(f"  Spearman rho (GAS vs strict absorption):  {rho_strict:.4f} (p={p_strict:.4f})")
    print(f"  Pearson r (GAS vs absorption):            {pearson_r:.4f} (p={pearson_p:.4f})")

    report_progress(TASK_ID, RESULTS_DIR, 3, 7, step=3, total_steps=7,
                    metric={"status": "spearman_computed", "rho": float(rho_absorption)})

    # ── Step 4: Bootstrap 95% CI ─────────────────────────────────────────────

    print("\n[Step 4] Bootstrap 95% CI on Spearman rho (10k resamples)...")

    n_bootstrap = 10000
    n_letters = len(letters)
    bootstrap_rhos = []

    rng = np.random.RandomState(SEED)
    for _ in range(n_bootstrap):
        idx = rng.choice(n_letters, n_letters, replace=True)
        g = gas_scores[idx]
        a = abs_rates[idx]
        if np.std(g) > 0 and np.std(a) > 0:
            r, _ = stats.spearmanr(g, a)
            bootstrap_rhos.append(r)
        else:
            bootstrap_rhos.append(0.0)

    bootstrap_rhos = np.array(bootstrap_rhos)
    ci_lower = float(np.percentile(bootstrap_rhos, 2.5))
    ci_upper = float(np.percentile(bootstrap_rhos, 97.5))
    bootstrap_mean = float(np.mean(bootstrap_rhos))
    bootstrap_std = float(np.std(bootstrap_rhos))

    print(f"  Bootstrap mean rho: {bootstrap_mean:.4f} (std={bootstrap_std:.4f})")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    report_progress(TASK_ID, RESULTS_DIR, 4, 7, step=4, total_steps=7,
                    metric={"status": "bootstrap_done", "ci_lower": ci_lower, "ci_upper": ci_upper})

    # ── Step 5: AUROC of GAS as absorption classifier ────────────────────────

    print("\n[Step 5] Computing AUROC of GAS as absorption classifier...")

    # Binary label: is this letter absorbed? (absorption_rate > 0)
    labels = (abs_rates > 0).astype(int)
    n_positive = labels.sum()
    n_negative = (1 - labels).sum()
    print(f"  Positive (absorbed) letters: {n_positive}, Negative: {n_negative}")

    # Compute ROC curve manually
    # Sort by GAS score descending
    sorted_indices = np.argsort(-gas_scores)
    sorted_labels = labels[sorted_indices]
    sorted_gas = gas_scores[sorted_indices]

    # TPR and FPR at each threshold
    tpr_list = [0.0]
    fpr_list = [0.0]
    tp, fp = 0, 0

    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / max(n_positive, 1))
        fpr_list.append(fp / max(n_negative, 1))

    tpr_array = np.array(tpr_list)
    fpr_array = np.array(fpr_list)

    # AUROC via trapezoidal rule
    auroc = float(np.trapz(tpr_array, fpr_array))

    print(f"  AUROC: {auroc:.4f}")

    # Also compute with strict absorption labels
    strict_labels = (strict_rates > 0).astype(int)
    n_strict_positive = strict_labels.sum()

    if n_strict_positive > 0 and n_strict_positive < len(strict_labels):
        sorted_strict = strict_labels[sorted_indices]
        tpr_s, fpr_s = [0.0], [0.0]
        tp_s, fp_s = 0, 0
        n_neg_s = (1 - strict_labels).sum()
        for i in range(len(sorted_strict)):
            if sorted_strict[i] == 1:
                tp_s += 1
            else:
                fp_s += 1
            tpr_s.append(tp_s / max(n_strict_positive, 1))
            fpr_s.append(fp_s / max(n_neg_s, 1))
        auroc_strict = float(np.trapz(tpr_s, fpr_s))
    else:
        auroc_strict = None
        tpr_s, fpr_s = [], []

    print(f"  AUROC (strict absorption): {auroc_strict}")

    # Also check: what if we use the main feature's cosine similarity with probe
    # as an alternative predictor? This is the baseline comparator.
    sorted_by_cos = np.argsort(-main_cos)
    cos_sorted_labels = labels[sorted_by_cos]
    tpr_c, fpr_c = [0.0], [0.0]
    tp_c, fp_c = 0, 0
    for i in range(len(cos_sorted_labels)):
        if cos_sorted_labels[i] == 1:
            tp_c += 1
        else:
            fp_c += 1
        tpr_c.append(tp_c / max(n_positive, 1))
        fpr_c.append(fp_c / max(n_negative, 1))
    auroc_cos_baseline = float(np.trapz(tpr_c, fpr_c))
    print(f"  AUROC (cos-similarity baseline): {auroc_cos_baseline:.4f}")

    # ROC curve data for output
    roc_curve = {
        "gas_roc": {
            "fpr": [round(float(x), 4) for x in fpr_array],
            "tpr": [round(float(x), 4) for x in tpr_array],
            "auroc": auroc
        },
        "cos_baseline_roc": {
            "fpr": [round(float(x), 4) for x in fpr_c],
            "tpr": [round(float(x), 4) for x in tpr_c],
            "auroc": auroc_cos_baseline
        }
    }
    if auroc_strict is not None:
        roc_curve["gas_strict_roc"] = {
            "fpr": [round(float(x), 4) for x in fpr_s],
            "tpr": [round(float(x), 4) for x in tpr_s],
            "auroc": auroc_strict
        }

    report_progress(TASK_ID, RESULTS_DIR, 5, 7, step=5, total_steps=7,
                    metric={"status": "auroc_computed", "auroc": auroc})

    # ── Step 6: Alternative GAS aggregation strategies ───────────────────────

    print("\n[Step 6] Testing alternative GAS aggregation strategies...")

    # Strategy 1: Instead of max GAS vulnerability per letter's main feature,
    # compute average GAS across ALL high-similarity neighbors of the main feature
    # This requires access to the pair data.

    # Strategy 2: Mean GAS across top-5 features per letter (not just top-1)
    # We don't have top-5 features per letter, but we can check top features
    # near the probe direction.

    # Strategy 3: Weighted GAS by cosine similarity (GAS * cos)
    weighted_gas = gas_scores * main_cos
    if np.std(weighted_gas) > 0 and np.std(abs_rates) > 0:
        rho_weighted, p_weighted = stats.spearmanr(weighted_gas, abs_rates)
    else:
        rho_weighted, p_weighted = 0.0, 1.0
    print(f"  Spearman rho (weighted GAS vs absorption): {rho_weighted:.4f} (p={p_weighted:.4f})")

    # Strategy 4: Log-transform GAS (reduces extreme outlier influence)
    log_gas = np.log1p(gas_scores)
    if np.std(log_gas) > 0 and np.std(abs_rates) > 0:
        rho_log, p_log = stats.spearmanr(log_gas, abs_rates)
    else:
        rho_log, p_log = 0.0, 1.0
    print(f"  Spearman rho (log(1+GAS) vs absorption): {rho_log:.4f} (p={p_log:.4f})")

    # Strategy 5: Use number of high-GAS neighbors instead of max
    # Count features with GAS > threshold pointing to each main feature
    # Reconstruct from pairs data
    neighbor_counts = {}
    for entry in gas_results.get("top_pairs_by_gas", []):
        fj = entry["feature_j"]
        fi = entry["feature_i"]
        gas_ij = entry["gas_i_absorbs_j"]
        gas_ji = entry["gas_j_absorbs_i"]
        if gas_ij > 1.0:
            neighbor_counts[fj] = neighbor_counts.get(fj, 0) + 1
        if gas_ji > 1.0:
            neighbor_counts[fi] = neighbor_counts.get(fi, 0) + 1

    gas_neighbor_counts = np.array([neighbor_counts.get(letter_data[l]["main_feature_id"], 0) for l in letters])
    if np.std(gas_neighbor_counts) > 0 and np.std(abs_rates) > 0:
        rho_neighbors, p_neighbors = stats.spearmanr(gas_neighbor_counts, abs_rates)
    else:
        rho_neighbors, p_neighbors = 0.0, 1.0
    print(f"  Spearman rho (n_high_GAS_neighbors vs absorption): {rho_neighbors:.4f} (p={p_neighbors:.4f})")

    # Strategy 6: Use the "inverse" direction -- check GAS vulnerability of the
    # letter feature itself (how much is it being absorbed BY other features)
    # This uses per_feature_gas[main_fid] which is max GAS(any -> main_fid)
    # This is what we already have. Let's also check the reverse:
    # How much does the main feature absorb others?
    # GAS(main_feature -> j) for all j
    absorber_scores = {}
    for entry in gas_results.get("top_pairs_by_gas", []):
        fi = entry["feature_i"]
        fj = entry["feature_j"]
        gas_ij = entry["gas_i_absorbs_j"]
        gas_ji = entry["gas_j_absorbs_i"]
        # fi as absorber
        if fi not in absorber_scores:
            absorber_scores[fi] = 0.0
        absorber_scores[fi] = max(absorber_scores[fi], gas_ij)
        # fj as absorber
        if fj not in absorber_scores:
            absorber_scores[fj] = 0.0
        absorber_scores[fj] = max(absorber_scores[fj], gas_ji)

    gas_absorber = np.array([absorber_scores.get(letter_data[l]["main_feature_id"], 0.0) for l in letters])
    if np.std(gas_absorber) > 0 and np.std(abs_rates) > 0:
        rho_absorber, p_absorber = stats.spearmanr(gas_absorber, abs_rates)
    else:
        rho_absorber, p_absorber = 0.0, 1.0
    print(f"  Spearman rho (max_absorber_GAS vs absorption): {rho_absorber:.4f} (p={p_absorber:.4f})")

    alternative_strategies = {
        "weighted_gas_x_cos": {"rho": float(rho_weighted), "p": float(p_weighted)},
        "log_gas": {"rho": float(rho_log), "p": float(p_log)},
        "n_high_gas_neighbors": {"rho": float(rho_neighbors), "p": float(p_neighbors)},
        "max_absorber_gas": {"rho": float(rho_absorber), "p": float(p_absorber)},
    }

    report_progress(TASK_ID, RESULTS_DIR, 6, 7, step=6, total_steps=7,
                    metric={"status": "alternatives_tested"})

    # ── Step 7: Discovery mode ───────────────────────────────────────────────

    print("\n[Step 7] Discovery mode: top-50 GAS candidates NOT in first-letter set...")

    first_letter_fids = set(letter_data[l]["main_feature_id"] for l in letters)
    print(f"  First-letter main feature IDs: {sorted(first_letter_fids)}")

    # Get top features by GAS vulnerability
    top_indices = np.argsort(-per_feature_gas)
    discovery_candidates = []
    for idx in top_indices:
        if len(discovery_candidates) >= 50:
            break
        if int(idx) not in first_letter_fids and per_feature_gas[idx] > 0:
            discovery_candidates.append({
                "feature_idx": int(idx),
                "gas_vulnerability": float(per_feature_gas[idx]),
            })

    print(f"  Top-50 non-first-letter candidates found: {len(discovery_candidates)}")
    if discovery_candidates:
        print(f"  Top 10:")
        for c in discovery_candidates[:10]:
            print(f"    Feature {c['feature_idx']:>6}: GAS = {c['gas_vulnerability']:.4f}")

    # ── Synthesis and assessment ──────────────────────────────────────────────

    print("\n[Synthesis] Assessing GAS validation results...")

    # Determine verdict
    target_rho = 0.6
    failure_rho = 0.3

    if abs(rho_absorption) >= target_rho:
        verdict = "STRONG_PASS"
        verdict_text = f"GAS achieves target rho >= {target_rho} ({rho_absorption:.4f})"
    elif abs(rho_absorption) >= failure_rho:
        verdict = "MODERATE_PASS"
        verdict_text = f"GAS correlation moderate (rho={rho_absorption:.4f}), between {failure_rho} and {target_rho}"
    else:
        verdict = "FAIL"
        verdict_text = f"GAS correlation below failure threshold (rho={rho_absorption:.4f} < {failure_rho})"

    # Diagnostic: why might GAS fail?
    diagnostics = []

    # Check 1: Are most letters zero-absorption? (limited signal)
    n_nonzero_abs = sum(1 for r in abs_rates if r > 0)
    if n_nonzero_abs < 5:
        diagnostics.append(
            f"Very few letters with non-zero absorption ({n_nonzero_abs}/{len(letters)}). "
            "Limited statistical power for correlation. "
            "The pilot sample uses only 352 test words per letter, "
            "and only L12-16k SAE -- more SAE configs and more words may increase signal."
        )

    # Check 2: Are high-GAS features the correct main features?
    n_high_gas_main = sum(1 for g in gas_scores if g > 1.0)
    diagnostics.append(
        f"Main letter features with GAS > 1.0: {n_high_gas_main}/{len(letters)}. "
        "GAS vulnerability of main features may be low because the main feature "
        "is the one that DOES the absorbing (high freq), not the one being absorbed."
    )

    # Check 3: GAS direction interpretation
    # GAS(i->j) = cos(d_i,d_j) * mismatch * freq(i)/freq(j)
    # High GAS(i->j) means i absorbs j (j is vulnerable).
    # For letter features: the main letter feature IS the parent being absorbed BY child features.
    # So we need GAS(child -> parent_letter_feature) to be high.
    # per_feature_gas[fid] = max_i GAS(i -> fid) -- this is correct:
    # it measures how much any feature absorbs the letter feature.
    diagnostics.append(
        "GAS direction check: per_feature_gas[fid] = max_i GAS(i -> fid). "
        "This measures vulnerability of the letter feature to absorption by any other feature. "
        "Higher GAS vulnerability should correlate with higher absorption rate. "
        "If it does NOT correlate, it may indicate that: "
        "(a) The main feature's GAS vulnerability is determined by geometry, not functional absorption; "
        "(b) Absorption happens through features NOT captured by the top-1 cosine match; or "
        "(c) The pilot sample (200 seqs, 25600 tokens) is too small for reliable co-activation statistics."
    )

    # Check 4: Is there a signal in the OPPOSITE direction?
    # (i.e., does the letter feature absorb OTHER features -- making it an absorber, not victim)
    if abs(rho_absorber) > abs(rho_absorption) and abs(rho_absorber) > 0.2:
        diagnostics.append(
            f"The absorber-direction GAS (rho={rho_absorber:.4f}) is stronger than "
            f"vulnerability-direction GAS (rho={rho_absorption:.4f}). "
            "This suggests the letter feature acts more as an absorber than a victim, "
            "which is consistent with the absorption mechanism: child features "
            "suppress the parent letter feature."
        )

    # Best strategy
    all_strategies = {
        "main_gas_vulnerability": {"rho": float(rho_absorption), "p": float(p_absorption)},
        **alternative_strategies
    }
    best_strategy = max(all_strategies.items(), key=lambda x: abs(x[1]["rho"]))
    print(f"\n  Best GAS aggregation strategy: {best_strategy[0]} (rho={best_strategy[1]['rho']:.4f})")
    print(f"  Verdict: {verdict} -- {verdict_text}")

    # ── Build output ─────────────────────────────────────────────────────────

    output = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "input_data": {
            "gas_source": str(GAS_RESULTS_FILE),
            "absorption_source": str(ABSORPTION_FILE),
            "sae_config": "L12_16k (Gemma Scope JumpReLU)",
            "n_letters_tested": len(letters),
            "n_letters_absorbed": int(n_nonzero_abs),
            "overall_absorption_rate": float(abs_results["absorption_rate"]),
        },
        "per_letter_detail": {
            letter: {
                "absorption_rate": float(letter_data[letter]["absorption_rate"]),
                "strict_rate": float(letter_data[letter]["strict_rate"]),
                "main_feature_id": int(letter_data[letter]["main_feature_id"]),
                "main_feature_cos": float(letter_data[letter]["main_feature_cos"]),
                "gas_vulnerability": float(letter_data[letter]["gas_vulnerability"]),
                "n_false_negatives": int(letter_data[letter]["n_false_negatives"]),
                "n_probe_correct": int(letter_data[letter]["n_probe_correct"]),
            }
            for letter in letters
        },
        "primary_correlation": {
            "metric": "Spearman rho (GAS vulnerability vs absorption rate)",
            "rho": float(rho_absorption),
            "p_value": float(p_absorption),
            "target": target_rho,
            "failure_threshold": failure_rho,
            "n_data_points": len(letters),
        },
        "strict_correlation": {
            "rho": float(rho_strict),
            "p_value": float(p_strict),
        },
        "pearson_correlation": {
            "r": float(pearson_r),
            "p_value": float(pearson_p),
        },
        "bootstrap_ci": {
            "n_resamples": n_bootstrap,
            "mean_rho": bootstrap_mean,
            "std_rho": bootstrap_std,
            "ci_lower_2_5": ci_lower,
            "ci_upper_97_5": ci_upper,
        },
        "auroc": {
            "gas_absorption": auroc,
            "gas_strict_absorption": auroc_strict,
            "cos_baseline": auroc_cos_baseline,
        },
        "roc_curves": roc_curve,
        "alternative_strategies": alternative_strategies,
        "best_strategy": {
            "name": best_strategy[0],
            "rho": best_strategy[1]["rho"],
            "p": best_strategy[1]["p"],
        },
        "discovery_candidates_top50": discovery_candidates,
        "diagnostics": diagnostics,
        "verdict": verdict,
        "verdict_text": verdict_text,
        "pilot_assessment": {
            "criterion_spearman_computed": {
                "target": "Spearman rho computed between GAS and probe-based absorption",
                "actual": f"rho={rho_absorption:.4f} (p={p_absorption:.4f})",
                "pass": True
            },
            "criterion_bootstrap_ci": {
                "target": "Bootstrap CI computed",
                "actual": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
                "pass": True
            },
            "criterion_auroc": {
                "target": "AUROC of GAS as classifier computed",
                "actual": f"AUROC={auroc:.4f}",
                "pass": True
            },
            "criterion_negative_result_documented": {
                "target": "If rho < 0.3, document as negative result",
                "actual": f"rho={rho_absorption:.4f}, verdict={verdict}",
                "pass": True,
                "note": "All results documented regardless of sign/magnitude"
            }
        },
        "overall_pilot_pass": True,  # All criteria met (computation successful)
        "timing": {
            "total_seconds": 0,  # Will be filled in
        }
    }

    # ── Save results ─────────────────────────────────────────────────────────

    # Save main results
    output_path = PHASE2_DIR / "gas_validation.json"
    elapsed = (datetime.now() - datetime.fromisoformat(start_time)).total_seconds()
    output["timing"]["total_seconds"] = round(elapsed, 2)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to: {output_path}")

    # Save summary markdown
    summary_path = PHASE2_DIR / "gas_validation_summary.md"
    summary_lines = [
        "# Phase 2.2: GAS Validation Against Probe-Based Absorption — PILOT Results",
        "",
        "## Configuration",
        f"- SAE: Gemma Scope L12 16k JumpReLU",
        f"- Letters tested: {len(letters)} (all with probe data)",
        f"- Letters with non-zero absorption: {n_nonzero_abs}",
        f"- Overall absorption rate: {abs_results['absorption_rate']:.4f}",
        f"- GAS source: Phase 2.1 (200 sequences, 25600 tokens)",
        "",
        "## Primary Result",
        f"- **Spearman rho (GAS vs absorption): {rho_absorption:.4f}** (p={p_absorption:.4f})",
        f"- Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]",
        f"- Target: rho >= {target_rho}, Failure threshold: rho < {failure_rho}",
        f"- **Verdict: {verdict}**",
        "",
        "## AUROC",
        f"- GAS AUROC (absorption detection): **{auroc:.4f}**",
        f"- GAS AUROC (strict absorption): {auroc_strict}",
        f"- Cosine baseline AUROC: {auroc_cos_baseline:.4f}",
        "",
        "## Alternative GAS Aggregation Strategies",
        "| Strategy | Spearman rho | p-value |",
        "|----------|-------------|---------|",
        f"| Main GAS vulnerability | {rho_absorption:.4f} | {p_absorption:.4f} |",
    ]
    for name, vals in alternative_strategies.items():
        summary_lines.append(f"| {name} | {vals['rho']:.4f} | {vals['p']:.4f} |")

    summary_lines.extend([
        "",
        f"**Best strategy: {best_strategy[0]}** (rho={best_strategy[1]['rho']:.4f})",
        "",
        "## Per-Letter Detail",
        "| Letter | AbsRate | StrictRate | MainFID | Cos | GAS Vuln |",
        "|--------|---------|------------|---------|-----|----------|",
    ])
    for letter in letters:
        ld = letter_data[letter]
        summary_lines.append(
            f"| {letter} | {ld['absorption_rate']:.4f} | {ld['strict_rate']:.4f} "
            f"| {ld['main_feature_id']} | {ld['main_feature_cos']:.3f} "
            f"| {ld['gas_vulnerability']:.4f} |"
        )

    summary_lines.extend([
        "",
        "## Diagnostics",
    ])
    for d in diagnostics:
        summary_lines.append(f"- {d}")

    summary_lines.extend([
        "",
        "## Discovery: Top 10 Non-First-Letter Candidates",
        "| Rank | Feature | GAS Vulnerability |",
        "|------|---------|-------------------|",
    ])
    for i, c in enumerate(discovery_candidates[:10]):
        summary_lines.append(f"| {i+1} | {c['feature_idx']} | {c['gas_vulnerability']:.4f} |")

    summary_lines.extend([
        "",
        f"## Timing: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)",
        "",
        "## Pilot Assessment: **ALL CRITERIA PASS**",
        "- [x] Spearman rho computed",
        "- [x] Bootstrap CI computed",
        "- [x] AUROC computed",
        "- [x] Negative/positive result documented",
    ])

    summary_path.write_text("\n".join(summary_lines))
    print(f"  Summary saved to: {summary_path}")

    # ── Mark done ────────────────────────────────────────────────────────────

    mark_task_done(TASK_ID, RESULTS_DIR, status="success",
                   summary=f"GAS validation: rho={rho_absorption:.4f}, AUROC={auroc:.4f}, verdict={verdict}")

    update_gpu_progress(WORKSPACE, TASK_ID, "success", start_time,
                        config_snapshot={
                            "mode": "PILOT",
                            "sae": "gemma-scope L12 16k",
                            "n_letters": len(letters),
                            "analysis_type": "CPU-only",
                        })

    print(f"\n{'='*60}")
    print(f"[{TASK_ID}] COMPLETE")
    print(f"  Verdict: {verdict}")
    print(f"  Spearman rho: {rho_absorption:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
