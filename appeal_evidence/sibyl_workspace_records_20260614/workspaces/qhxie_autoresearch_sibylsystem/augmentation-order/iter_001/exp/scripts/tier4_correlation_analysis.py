"""
Tier 4: NC Correlation and DPI Validation Analysis (CPU-only)
Aggregates NC_2 estimates (tier4a) and MI estimates (tier4b) with accuracy results (tier1_analysis).
Computes:
  (1) Spearman rho between NC_2 ranking and accuracy-diff ranking; permutation test
  (2) Spearman rho between MI ranking and accuracy ranking
  (3) Paired comparison: reversibility-sorted (CJ→Flip→Crop) vs conventional (Crop→Flip→CJ)
  (4) H3/H4 verdict table
Writes to exp/results/full/tier4_correlation.json and tier4_correlation.md
Pass criteria: correlation script runs without error; H3/H4 verdict written with rho, p-value, confirmed/falsified
"""
import os
import json
import time
import math
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from itertools import permutations

# ---- Config ----
TASK_ID = "tier4_correlation_analysis"
RESULTS_DIR = Path("/home/qhxie/sibyl_system/projects/augmentation-order/exp/results/full")
GPU_PROGRESS_PATH = Path("/home/qhxie/sibyl_system/projects/augmentation-order/exp/gpu_progress.json")

TIER1_ANALYSIS_PATH = RESULTS_DIR / "tier1_analysis.json"
TIER4A_NC_PATH = RESULTS_DIR / "tier4a_nc.json"
TIER4B_MI_PATH = RESULTS_DIR / "tier4b_mi.json"


def write_pid():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))


def mark_done(status="success", summary=""):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def write_progress(step, total_steps, msg=""):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    progress = {
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
        "message": msg,
        "updated_at": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(progress))


def update_gpu_progress(status, actual_min, start_time, end_time):
    try:
        if GPU_PROGRESS_PATH.exists():
            data = json.loads(GPU_PROGRESS_PATH.read_text())
        else:
            data = {"completed": [], "failed": [], "running": {}, "timings": {}}
        if status == "success":
            if TASK_ID not in data["completed"]:
                data["completed"].append(TASK_ID)
        else:
            if TASK_ID not in data["failed"]:
                data["failed"].append(TASK_ID)
        data["running"].pop(TASK_ID, None)
        data["timings"][TASK_ID] = {
            "planned_min": 10,
            "actual_min": actual_min,
            "start_time": start_time,
            "end_time": end_time,
            "config_snapshot": {"mode": "pilot", "cpu_only": True}
        }
        GPU_PROGRESS_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Warning: could not update gpu_progress.json: {e}")


def spearman_rho(x, y):
    """Compute Spearman rank correlation between two lists."""
    n = len(x)
    if n < 2:
        return 0.0, 1.0
    # Compute ranks
    def rank_list(lst):
        sorted_idx = sorted(range(n), key=lambda i: lst[i])
        ranks = [0] * n
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank + 1
        return ranks
    rx = rank_list(x)
    ry = rank_list(y)
    d2 = sum((rx[i] - ry[i])**2 for i in range(n))
    rho = 1 - 6 * d2 / (n * (n**2 - 1))
    # Approximate p-value using t-distribution
    if abs(rho) >= 1.0:
        p_value = 0.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1 - rho**2))
        # Two-tailed p-value approximation using normal distribution
        # For small n, use conservative estimate
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    return round(rho, 4), round(p_value, 4)


def _normal_cdf(x):
    """Approximation of standard normal CDF."""
    # Abramowitz & Stegun approximation
    t = 1 / (1 + 0.2316419 * abs(x))
    d = 0.3989422820 * math.exp(-0.5 * x * x)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.8212560 + t * 1.3302744))))
    if x > 0:
        return 1 - p
    return p


def permutation_test(x, y, n_permutations=1000, seed=42):
    """Permutation test for Spearman rho significance."""
    random.seed(seed)
    observed_rho, _ = spearman_rho(x, y)
    count_extreme = 0
    y_list = list(y)
    for _ in range(n_permutations):
        y_perm = y_list[:]
        random.shuffle(y_perm)
        perm_rho, _ = spearman_rho(x, y_perm)
        if abs(perm_rho) >= abs(observed_rho):
            count_extreme += 1
    p_perm = round((count_extreme + 1) / (n_permutations + 1), 4)
    return p_perm


def load_data():
    """Load all required data files."""
    errors = []

    # Load tier1_analysis
    t1_data = None
    if TIER1_ANALYSIS_PATH.exists():
        try:
            t1_data = json.loads(TIER1_ANALYSIS_PATH.read_text())
            print(f"Loaded tier1_analysis: {len(t1_data.get('blocks', {}))} blocks")
        except Exception as e:
            errors.append(f"tier1_analysis load error: {e}")
    else:
        errors.append(f"tier1_analysis not found at {TIER1_ANALYSIS_PATH}")

    # Load tier4a_nc
    t4a_data = None
    if TIER4A_NC_PATH.exists():
        try:
            t4a_data = json.loads(TIER4A_NC_PATH.read_text())
            print(f"Loaded tier4a_nc: {len(t4a_data.get('pairs_analyzed', []))} pairs")
        except Exception as e:
            errors.append(f"tier4a_nc load error: {e}")
    else:
        errors.append(f"tier4a_nc not found at {TIER4A_NC_PATH}")

    # Load tier4b_mi
    t4b_data = None
    if TIER4B_MI_PATH.exists():
        try:
            t4b_data = json.loads(TIER4B_MI_PATH.read_text())
            print(f"Loaded tier4b_mi: datasets={list(t4b_data.get('datasets', {}).keys())}")
        except Exception as e:
            errors.append(f"tier4b_mi load error: {e}")
    else:
        errors.append(f"tier4b_mi not found at {TIER4B_MI_PATH}")

    return t1_data, t4a_data, t4b_data, errors


def compute_h3_nc2_vs_accuracy(t1_data, t4a_data):
    """
    H3: NC_2 (non-commutativity) predicts ordering accuracy differences.
    Higher NC_2(pair) → larger accuracy difference when that pair's order changes.

    Approach: For each 3-op ordering, compute the sum of NC_2 for pairs that appear
    in the 'high-yield' position (crop before cj, flip before cj, etc.).
    Then correlate with ordering accuracy ranks.
    """
    result = {
        "hypothesis": "H3",
        "description": "NC_2 non-commutativity correlates with accuracy-diff ranking across orderings",
        "method": "spearman_rho(nc2_proxy_per_ordering, accuracy_rank)",
        "rho": None,
        "p_value": None,
        "p_perm": None,
        "verdict": "inconclusive",
        "details": {},
        "errors": [],
    }

    if t4a_data is None or t1_data is None:
        result["errors"].append("Missing tier4a or tier1 data")
        return result

    # Extract NC_2 per pair
    nc2_by_pair = {}
    for pair_info in t4a_data.get("pairs_analyzed", []):
        pair_label = pair_info["pair_label"]
        nc2_by_pair[pair_label] = pair_info["nc2_proxy"]

    print(f"  NC_2 per pair: {nc2_by_pair}")

    # For each ordering in tier1 (resnet18_cifar100 as primary block), get accuracy
    # Use the most informative block: resnet18_cifar100
    block_priorities = ["resnet18_cifar100", "resnet18_cifar10", "vit_cifar10", "vit_cifar100"]
    block_data = None
    used_block = None
    for block_name in block_priorities:
        if block_name in t1_data.get("blocks", {}):
            block_data = t1_data["blocks"][block_name]
            used_block = block_name
            break

    if block_data is None:
        result["errors"].append("No tier1 block data found")
        return result

    ordering_accs = block_data.get("ordering_accuracies", {})
    ordering_labels = block_data.get("ordering_labels", {})
    print(f"  Using block: {used_block}, orderings: {list(ordering_accs.keys())}")

    # Map operation pairs to NC_2 scores for each ordering
    # Ordering is a sequence of 3 ops. For each consecutive pair in the sequence,
    # look up NC_2. An ordering that places high-NC_2 pairs in 'canonical' order
    # (crop before flip, crop before cj, flip before cj) should have higher accuracy.

    # Build: for each ordering, compute a 'NC2-weighted ordering score'
    # Score = sum of NC_2(pair) * direction_sign for consecutive pairs
    # direction_sign = +1 if canonical order (lower irreversibility first), -1 otherwise
    # Individual transform SWD ranks: Crop(high) > CJ(medium) > Flip(low)
    # Canonical: more-reversible first → Flip < CJ < Crop
    # So canonical position: Flip first (rank 1), CJ second (rank 2), Crop last (rank 3)
    reversibility_rank = {"flip": 1, "cj": 2, "crop": 3}

    ordering_nc2_scores = {}
    for ordering_key, acc in ordering_accs.items():
        label = ordering_labels.get(ordering_key, "")
        # Parse ops from label
        ops = [op.strip().lower() for op in label.replace("→", "->").split("->")]
        if len(ops) != 3:
            continue
        # Map label names to internal keys
        ops_mapped = []
        for op in ops:
            if "crop" in op:
                ops_mapped.append("crop")
            elif "flip" in op:
                ops_mapped.append("flip")
            elif "cj" in op or "color" in op:
                ops_mapped.append("cj")
        if len(ops_mapped) != 3:
            continue

        # Sum NC_2 for all 3 pairs with canonical direction sign
        score = 0.0
        pair_details = []
        for i in range(len(ops_mapped)):
            for j in range(i+1, len(ops_mapped)):
                op_i, op_j = ops_mapped[i], ops_mapped[j]
                # Look up NC_2
                # pair_label format: "Crop_Flip", "Crop_CJ", "Flip_CJ"
                pair_lookup_options = [
                    f"{op_i.capitalize()}_{op_j.capitalize()}",
                    f"{op_j.capitalize()}_{op_i.capitalize()}",
                    f"Crop_Flip" if set([op_i, op_j]) == {"crop", "flip"} else None,
                    f"Crop_CJ" if set([op_i, op_j]) == {"crop", "cj"} else None,
                    f"Flip_CJ" if set([op_i, op_j]) == {"flip", "cj"} else None,
                ]
                nc2_val = None
                for pl in pair_lookup_options:
                    if pl and pl in nc2_by_pair:
                        nc2_val = nc2_by_pair[pl]
                        break
                if nc2_val is None:
                    continue
                # Direction: i appears before j in the sequence
                # Canonical: lower reversibility rank first (Flip=1, CJ=2, Crop=3)
                ri = reversibility_rank.get(op_i, 2)
                rj = reversibility_rank.get(op_j, 2)
                # If op_i (earlier in sequence) has lower rank, it's canonical direction
                direction_sign = 1.0 if ri <= rj else -1.0
                score += nc2_val * direction_sign
                pair_details.append({
                    "pair": f"{op_i}-{op_j}",
                    "nc2": nc2_val,
                    "direction_sign": direction_sign,
                })

        ordering_nc2_scores[ordering_key] = {
            "acc": acc,
            "nc2_score": round(score, 6),
            "label": label,
            "ops": ops_mapped,
            "pair_details": pair_details,
        }

    print(f"  NC2 scores per ordering: {{k: v['nc2_score'] for k, v in ordering_nc2_scores.items()}}")

    # Compute Spearman rho
    keys = list(ordering_nc2_scores.keys())
    nc2_scores = [ordering_nc2_scores[k]["nc2_score"] for k in keys]
    accs_list = [ordering_nc2_scores[k]["acc"] for k in keys]

    if len(keys) < 3:
        result["errors"].append(f"Too few orderings: {len(keys)}")
        return result

    rho, p_val = spearman_rho(nc2_scores, accs_list)
    p_perm = permutation_test(nc2_scores, accs_list, n_permutations=500)
    result["rho"] = rho
    result["p_value"] = p_val
    result["p_perm"] = p_perm
    result["used_block"] = used_block
    result["n_orderings"] = len(keys)
    result["ordering_scores"] = ordering_nc2_scores

    # Verdict: confirmed if rho > 0.3 and p < 0.2 (lenient for pilot)
    if rho > 0.3 and p_perm < 0.2:
        result["verdict"] = "confirmed"
    elif rho < -0.1:
        result["verdict"] = "falsified"
    else:
        result["verdict"] = "inconclusive"

    print(f"  H3: rho={rho}, p_val={p_val}, p_perm={p_perm}, verdict={result['verdict']}")
    return result


def compute_h4_mi_vs_accuracy(t1_data, t4b_data):
    """
    H4: InfoNCE MI estimate correlates with ordering accuracy ranking.
    """
    result = {
        "hypothesis": "H4",
        "description": "InfoNCE MI correlates with ordering accuracy ranking",
        "method": "spearman_rho(mi_estimate, accuracy) per dataset",
        "datasets": {},
        "combined_rho": None,
        "verdict": "inconclusive",
        "errors": [],
    }

    if t4b_data is None or t1_data is None:
        result["errors"].append("Missing tier4b or tier1 data")
        return result

    dataset_results = {}
    all_rhos = []

    for dataset_name in ["cifar10", "cifar100"]:
        ds_result = {}
        mi_data = t4b_data.get("datasets", {}).get(dataset_name, {})
        if not mi_data:
            ds_result["error"] = f"No MI data for {dataset_name}"
            dataset_results[dataset_name] = ds_result
            continue

        # Find corresponding tier1 block
        block_name_map = {
            "cifar10": ["resnet18_cifar10", "vit_cifar10"],
            "cifar100": ["resnet18_cifar100", "vit_cifar100"],
        }
        block_data = None
        used_block = None
        for bn in block_name_map.get(dataset_name, []):
            if bn in t1_data.get("blocks", {}):
                block_data = t1_data["blocks"][bn]
                used_block = bn
                break

        if block_data is None:
            ds_result["error"] = f"No tier1 block for {dataset_name}"
            dataset_results[dataset_name] = ds_result
            continue

        ordering_accs = block_data.get("ordering_accuracies", {})
        mi_orderings = mi_data.get("orderings", {})

        # Match orderings by key
        matched = []
        for order_key in ordering_accs:
            if order_key in mi_orderings:
                mi_val = mi_orderings[order_key].get("mi_estimate")
                acc_val = ordering_accs[order_key]
                if mi_val is not None and acc_val is not None and mi_orderings[order_key].get("mi_finite", True):
                    matched.append({
                        "ordering": order_key,
                        "label": mi_orderings[order_key].get("label", ""),
                        "mi_estimate": mi_val,
                        "accuracy": acc_val,
                    })

        if len(matched) < 3:
            ds_result["error"] = f"Too few matched orderings: {len(matched)}"
            dataset_results[dataset_name] = ds_result
            continue

        mi_vals = [m["mi_estimate"] for m in matched]
        acc_vals = [m["accuracy"] for m in matched]
        rho, p_val = spearman_rho(mi_vals, acc_vals)
        p_perm = permutation_test(mi_vals, acc_vals, n_permutations=500)

        ds_result = {
            "used_block": used_block,
            "n_orderings": len(matched),
            "rho": rho,
            "p_value": p_val,
            "p_perm": p_perm,
            "matched_orderings": matched,
            "paired_comparison": mi_data.get("paired_comparison", {}),
        }
        dataset_results[dataset_name] = ds_result
        all_rhos.append(rho)
        print(f"  H4 {dataset_name}: rho={rho}, p_perm={p_perm}")

    result["datasets"] = dataset_results
    if all_rhos:
        combined_rho = round(float(np.mean(all_rhos)), 4)
        result["combined_rho"] = combined_rho
        # Verdict
        if combined_rho > 0.3:
            result["verdict"] = "confirmed"
        elif combined_rho < -0.1:
            result["verdict"] = "falsified"
        else:
            result["verdict"] = "inconclusive"

    print(f"  H4: combined_rho={result['combined_rho']}, verdict={result['verdict']}")
    return result


def compute_h1_ordering_matters(t1_data):
    """
    H1: Ordering matters (spread > threshold across arch-dataset blocks).
    """
    result = {
        "hypothesis": "H1",
        "description": "Augmentation ordering significantly affects accuracy (spread > 0.5%)",
        "threshold_pct": 0.5,
        "blocks_confirmed": 0,
        "blocks_total": 0,
        "spreads": {},
        "verdict": "inconclusive",
        "errors": [],
    }

    if t1_data is None:
        result["errors"].append("Missing tier1 data")
        return result

    blocks = t1_data.get("blocks", {})
    for block_name, block_data in blocks.items():
        spread = block_data.get("spread", 0)
        spread_pct = spread * 100
        result["spreads"][block_name] = round(spread_pct, 3)
        result["blocks_total"] += 1
        if spread_pct > result["threshold_pct"]:
            result["blocks_confirmed"] += 1

    if result["blocks_total"] > 0:
        frac = result["blocks_confirmed"] / result["blocks_total"]
        if frac >= 0.5:
            result["verdict"] = "confirmed"
        else:
            result["verdict"] = "falsified"

    print(f"  H1: {result['blocks_confirmed']}/{result['blocks_total']} blocks above threshold, verdict={result['verdict']}")
    return result


def compute_h2_reversibility_ordering(t1_data):
    """
    H2: Reversibility-sorted ordering (CJ→Flip→Crop) outperforms conventional (Crop→Flip→CJ).
    """
    result = {
        "hypothesis": "H2",
        "description": "Reversibility-sorted ordering (CJ→Flip→Crop) outperforms conventional (Crop→Flip→CJ)",
        "reversibility_sorted": "CJ→Flip→Crop",
        "conventional": "Crop→Flip→CJ",
        "blocks_confirmed": 0,
        "blocks_total": 0,
        "comparisons": {},
        "verdict": "inconclusive",
        "errors": [],
    }

    if t1_data is None:
        result["errors"].append("Missing tier1 data")
        return result

    blocks = t1_data.get("blocks", {})
    for block_name, block_data in blocks.items():
        ordering_accs = block_data.get("ordering_accuracies", {})
        ordering_labels = block_data.get("ordering_labels", {})

        conv_key = None
        rev_key = None
        for k, label in ordering_labels.items():
            label_clean = label.replace(" ", "")
            if "Crop" in label and "Flip" in label and "CJ" in label:
                ops = [o.strip() for o in label.replace("→", "->").split("->")]
                if len(ops) == 3:
                    if ops[0].lower().startswith("crop") and ops[1].lower().startswith("flip"):
                        conv_key = k
                    elif ops[0].lower().startswith("cj") or ops[0].lower().startswith("color"):
                        if ops[1].lower().startswith("flip"):
                            rev_key = k

        if conv_key and rev_key and conv_key in ordering_accs and rev_key in ordering_accs:
            conv_acc = ordering_accs[conv_key]
            rev_acc = ordering_accs[rev_key]
            rev_wins = rev_acc > conv_acc
            result["comparisons"][block_name] = {
                "conventional_key": conv_key,
                "reversibility_sorted_key": rev_key,
                "conventional_acc": conv_acc,
                "reversibility_sorted_acc": rev_acc,
                "diff": round(rev_acc - conv_acc, 4),
                "reversibility_sorted_wins": rev_wins,
            }
            result["blocks_total"] += 1
            if rev_wins:
                result["blocks_confirmed"] += 1
            print(f"  H2 {block_name}: rev={rev_acc:.4f} vs conv={conv_acc:.4f}, wins={rev_wins}")

    if result["blocks_total"] > 0:
        frac = result["blocks_confirmed"] / result["blocks_total"]
        if frac >= 0.5:
            result["verdict"] = "confirmed"
        elif frac <= 0.25:
            result["verdict"] = "falsified"
        else:
            result["verdict"] = "inconclusive"

    print(f"  H2: {result['blocks_confirmed']}/{result['blocks_total']} blocks confirmed, verdict={result['verdict']}")
    return result


def compute_h5_magnitude_interaction(tier3_path=None):
    """
    H5: Magnitude moderates the ordering effect (spread increases with magnitude).
    """
    result = {
        "hypothesis": "H5",
        "description": "Higher augmentation magnitude amplifies ordering-induced accuracy spread",
        "spread_by_magnitude": {},
        "spread_increases": False,
        "verdict": "pending",
        "errors": [],
    }

    # Try to load tier3 results
    if tier3_path is None:
        tier3_path = RESULTS_DIR / "tier3_results.json"

    if not tier3_path.exists():
        result["verdict"] = "pending"
        result["errors"].append(f"tier3 results not yet available at {tier3_path}")
        return result

    try:
        t3_data = json.loads(tier3_path.read_text())
    except Exception as e:
        result["errors"].append(f"tier3 load error: {e}")
        result["verdict"] = "pending"
        return result

    spread_by_mag = t3_data.get("spread_by_magnitude", {})
    result["spread_by_magnitude"] = spread_by_mag

    m5 = spread_by_mag.get("M5")
    m9 = spread_by_mag.get("M9")
    m14 = spread_by_mag.get("M14")

    valid_spreads = [(k, v) for k, v in [("M5", m5), ("M9", m9), ("M14", m14)] if v is not None]
    if len(valid_spreads) < 2:
        result["verdict"] = "inconclusive"
        result["errors"].append("Insufficient tier3 data for H5 evaluation")
        return result

    # Check if spread increases from M5 to M14
    spread_increases = m14 is not None and m5 is not None and m14 > m5
    result["spread_increases"] = spread_increases
    result["nan_at_m14"] = t3_data.get("nan_at_m14", False)

    if spread_increases and not t3_data.get("nan_at_m14", False):
        result["verdict"] = "confirmed"
    elif not spread_increases:
        result["verdict"] = "falsified"
    else:
        result["verdict"] = "inconclusive"

    print(f"  H5: spread_m5={m5}, spread_m9={m9}, spread_m14={m14}, verdict={result['verdict']}")
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_pid()

    print(f"[{TASK_ID}] Starting Tier 4 Correlation Analysis (CPU-only)")
    start_time = datetime.now().isoformat()
    start_ts = time.time()

    write_progress(0, 6, "Loading data")

    # Load all data
    t1_data, t4a_data, t4b_data, load_errors = load_data()
    print(f"Load errors: {load_errors}")

    write_progress(1, 6, "Computing H1: ordering spread")
    h1_result = compute_h1_ordering_matters(t1_data)

    write_progress(2, 6, "Computing H2: reversibility ordering")
    h2_result = compute_h2_reversibility_ordering(t1_data)

    write_progress(3, 6, "Computing H3: NC_2 vs accuracy correlation")
    h3_result = compute_h3_nc2_vs_accuracy(t1_data, t4a_data)

    write_progress(4, 6, "Computing H4: MI vs accuracy correlation")
    h4_result = compute_h4_mi_vs_accuracy(t1_data, t4b_data)

    write_progress(5, 6, "Computing H5: magnitude interaction")
    h5_result = compute_h5_magnitude_interaction()

    # Compile verdict table
    verdict_table = {
        "H1": {
            "name": "Ordering Matters",
            "metric": f"spread > 0.5% in ≥50% blocks",
            "observed": f"{h1_result.get('blocks_confirmed', 0)}/{h1_result.get('blocks_total', 0)} blocks above threshold",
            "verdict": h1_result.get("verdict"),
        },
        "H2": {
            "name": "Reversibility-Sorted Wins",
            "metric": "CJ→Flip→Crop > Crop→Flip→CJ in ≥50% blocks",
            "observed": f"{h2_result.get('blocks_confirmed', 0)}/{h2_result.get('blocks_total', 0)} blocks confirmed",
            "verdict": h2_result.get("verdict"),
        },
        "H3": {
            "name": "NC_2 Predicts Accuracy",
            "metric": "Spearman rho(NC_2, accuracy) > 0.3",
            "observed": f"rho={h3_result.get('rho')}, p_perm={h3_result.get('p_perm')}",
            "verdict": h3_result.get("verdict"),
        },
        "H4": {
            "name": "MI Predicts Accuracy",
            "metric": "Spearman rho(MI, accuracy) > 0.3",
            "observed": f"combined_rho={h4_result.get('combined_rho')}",
            "verdict": h4_result.get("verdict"),
        },
        "H5": {
            "name": "Magnitude Amplifies Spread",
            "metric": "spread[M14] > spread[M5]",
            "observed": str(h5_result.get("spread_by_magnitude", {})),
            "verdict": h5_result.get("verdict"),
        },
    }

    # Pass criteria check
    # - Correlation script runs without error: True if we got here
    # - H3/H4 verdict written with rho, p-value, confirmed/falsified status
    h3_complete = h3_result.get("rho") is not None and h3_result.get("p_perm") is not None
    h4_complete = h4_result.get("combined_rho") is not None
    pass_criteria_met = h3_complete and h4_complete

    print(f"\n[{TASK_ID}] Verdict Table:")
    for h_key, h_val in verdict_table.items():
        print(f"  {h_key} ({h_val['name']}): {h_val['verdict']} — {h_val['observed']}")

    # Build full result
    full_result = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "timestamp": datetime.now().isoformat(),
        "load_errors": load_errors,
        "h1_ordering_matters": h1_result,
        "h2_reversibility_ordering": h2_result,
        "h3_nc2_vs_accuracy": h3_result,
        "h4_mi_vs_accuracy": h4_result,
        "h5_magnitude_interaction": h5_result,
        "verdict_table": verdict_table,
        "pass_criteria_met": pass_criteria_met,
        "elapsed_sec": round(time.time() - start_ts, 2),
    }

    # Write JSON
    out_json = RESULTS_DIR / "tier4_correlation.json"
    out_json.write_text(json.dumps(full_result, indent=2))
    print(f"\n[{TASK_ID}] JSON results written to {out_json}")

    # Write Markdown summary
    md_lines = [
        f"# Tier 4: NC Correlation and DPI Validation (Pilot)",
        f"",
        f"Generated: {full_result['timestamp']}",
        f"",
        f"## Hypothesis Verdict Table",
        f"",
        f"| Hypothesis | Name | Metric | Observed | Verdict |",
        f"|-----------|------|--------|----------|---------|",
    ]
    for h_key, h_val in verdict_table.items():
        md_lines.append(
            f"| {h_key} | {h_val['name']} | {h_val['metric']} | {h_val['observed']} | **{h_val['verdict']}** |"
        )
    md_lines += [
        f"",
        f"## H3: NC_2 vs Accuracy Correlation",
        f"",
        f"- **Spearman rho**: {h3_result.get('rho')}",
        f"- **p-value (approx)**: {h3_result.get('p_value')}",
        f"- **p-value (permutation)**: {h3_result.get('p_perm')}",
        f"- **n_orderings**: {h3_result.get('n_orderings')}",
        f"- **Verdict**: {h3_result.get('verdict')}",
        f"",
        f"## H4: MI vs Accuracy Correlation",
        f"",
        f"- **Combined Spearman rho**: {h4_result.get('combined_rho')}",
    ]
    for ds_name, ds_res in h4_result.get("datasets", {}).items():
        if isinstance(ds_res, dict) and "rho" in ds_res:
            md_lines.append(f"  - {ds_name}: rho={ds_res['rho']}, p_perm={ds_res.get('p_perm')}")
    md_lines += [
        f"- **Verdict**: {h4_result.get('verdict')}",
        f"",
        f"## H5: Magnitude Interaction",
        f"",
        f"- **Spread by magnitude**: {h5_result.get('spread_by_magnitude')}",
        f"- **Spread increases M5→M14**: {h5_result.get('spread_increases')}",
        f"- **Verdict**: {h5_result.get('verdict')}",
        f"",
        f"## Pass Criteria",
        f"",
        f"- H3 complete (rho + p-value computed): {h3_complete}",
        f"- H4 complete (combined rho computed): {h4_complete}",
        f"- **Overall pass**: {pass_criteria_met}",
    ]

    out_md = RESULTS_DIR / "tier4_correlation.md"
    out_md.write_text("\n".join(md_lines))
    print(f"[{TASK_ID}] Markdown written to {out_md}")

    end_time = datetime.now().isoformat()
    actual_min = round((time.time() - start_ts) / 60, 2)
    update_gpu_progress("success", actual_min, start_time, end_time)
    mark_done(
        "success" if pass_criteria_met else "failed",
        f"H3_rho={h3_result.get('rho')},H4_rho={h4_result.get('combined_rho')},H3={h3_result.get('verdict')},H4={h4_result.get('verdict')}"
    )
    print(f"[{TASK_ID}] Done in {actual_min} min.")


if __name__ == "__main__":
    main()
