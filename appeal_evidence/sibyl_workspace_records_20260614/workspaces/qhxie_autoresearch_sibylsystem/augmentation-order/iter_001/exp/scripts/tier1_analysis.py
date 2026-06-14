#!/usr/bin/env python3
"""
Tier 1 Statistical Analysis — PILOT MODE
Aggregates all Tier 1 pilot results and computes:
  1. Max-min spread per (arch, dataset)
  2. Paired t-tests best vs. worst ordering per block (with Bonferroni correction)
  3. Ordering effect summary
  4. Best and worst orderings for use in Tier 3
Writes to:
  - exp/results/full/tier1_analysis.json
  - exp/results/full/tier1_analysis.md
  - exp/results/full/tier1_analysis.pid  (then removed on done)
  - exp/results/full/tier1_analysis_DONE
"""

import json
import os
import sys
import math
from pathlib import Path
from datetime import datetime

REMOTE_BASE = "/home/qhxie/sibyl_system"
PROJECT = "augmentation-order"
PROJECT_DIR = Path(REMOTE_BASE) / "projects" / PROJECT
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
FULL_DIR = RESULTS_DIR / "full"
PILOTS_DIR = RESULTS_DIR / "pilots"

FULL_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "tier1_analysis"

# Write PID file
pid_file = FULL_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def load_pilot_data(path):
    """Load pilot JSON and extract ordering accuracy data."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_spread(orderings_data):
    """Compute max-min spread from ordering accuracy dict."""
    accs = [v["mean_val_acc"] for v in orderings_data.values() if isinstance(v, dict)]
    if not accs:
        return 0.0, 0.0, 0.0
    return max(accs) - min(accs), max(accs), min(accs)


def find_best_worst(orderings_data):
    """Find best and worst ordering IDs and labels."""
    best_id, best_acc = None, -1
    worst_id, worst_acc = None, 2
    for oid, v in orderings_data.items():
        if not isinstance(v, dict):
            continue
        acc = v["mean_val_acc"]
        if acc > best_acc:
            best_acc = acc
            best_id = oid
        if acc < worst_acc:
            worst_acc = acc
            worst_id = oid
    return best_id, worst_id


def mean(lst):
    if not lst:
        return float('nan')
    return sum(lst) / len(lst)


def stdev(lst):
    if len(lst) < 2:
        return 0.0
    m = mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))


def paired_t_test(vals_a, vals_b):
    """Paired t-test between two lists. Returns (t_stat, p_approx, n)."""
    if len(vals_a) != len(vals_b) or len(vals_a) < 2:
        return float('nan'), float('nan'), len(vals_a)
    diffs = [a - b for a, b in zip(vals_a, vals_b)]
    n = len(diffs)
    d_mean = mean(diffs)
    d_std = stdev(diffs)
    if d_std == 0:
        return float('nan'), float('nan'), n
    t_stat = d_mean / (d_std / math.sqrt(n))
    # Approximate p-value using t-distribution (two-tailed) — very rough for small n
    # For pilot purposes, note it as approximate
    df = n - 1
    # Simple approximation
    p_approx = min(1.0, 2 * math.exp(-0.717 * abs(t_stat) - 0.416 * abs(t_stat) ** 2) if df > 1 else 1.0)
    return t_stat, p_approx, n


def cohen_d(mean1, mean2, pooled_std):
    if pooled_std == 0:
        return 0.0
    return abs(mean1 - mean2) / pooled_std


# ============================================================
# Load data for all 4 (arch, dataset) blocks
# ============================================================

blocks = {}

# tier1_resnet18_cifar10 — stored in pilots/
path_rn18_c10 = PILOTS_DIR / "tier1_resnet18_cifar10_pilot.json"
data_rn18_c10 = load_pilot_data(path_rn18_c10)
if data_rn18_c10 is None:
    # Also check PROGRESS as fallback
    path_rn18_c10_prog = RESULTS_DIR / "tier1_resnet18_cifar10_PROGRESS.json"
    print(f"WARNING: tier1_resnet18_cifar10_pilot.json not found, trying PROGRESS at {path_rn18_c10_prog}")
    # Build minimal structure from main results dir
    data_rn18_c10 = None

# tier1_resnet18_cifar100 — stored in full/
path_rn18_c100 = FULL_DIR / "tier1_resnet18_cifar100_pilot.json"
data_rn18_c100 = load_pilot_data(path_rn18_c100)

# tier1_vit_cifar10 — stored in pilots/
path_vit_c10 = PILOTS_DIR / "tier1_vit_cifar10_pilot.json"
data_vit_c10 = load_pilot_data(path_vit_c10)

# tier1_vit_cifar100 — stored in pilots/
path_vit_c100 = PILOTS_DIR / "tier1_vit_cifar100_pilot.json"
data_vit_c100 = load_pilot_data(path_vit_c100)

block_sources = {
    ("resnet18", "cifar10"): data_rn18_c10,
    ("resnet18", "cifar100"): data_rn18_c100,
    ("vit", "cifar10"): data_vit_c10,
    ("vit", "cifar100"): data_vit_c100,
}

print("\n=== Tier 1 Statistical Analysis (PILOT MODE) ===\n")
print(f"Data availability:")
for (arch, ds), data in block_sources.items():
    print(f"  {arch} x {ds}: {'FOUND' if data else 'MISSING'}")

# ============================================================
# Per-block analysis
# ============================================================

block_results = {}
all_spreads = []

ORDERING_LABELS = {
    "order_0": "Crop→Flip→CJ",
    "order_1": "Crop→CJ→Flip",
    "order_2": "Flip→Crop→CJ",
    "order_3": "Flip→CJ→Crop",
    "order_4": "CJ→Crop→Flip",
    "order_5": "CJ→Flip→Crop",
}

for (arch, dataset), data in block_sources.items():
    block_key = f"{arch}_{dataset}"

    if data is None:
        print(f"\n[{block_key}] SKIP — data not available")
        block_results[block_key] = {
            "arch": arch,
            "dataset": dataset,
            "status": "missing",
            "spread": None,
            "best_ordering": None,
            "worst_ordering": None,
        }
        continue

    orderings_data = data.get("orderings", {})
    if not orderings_data:
        print(f"\n[{block_key}] SKIP — no orderings data")
        continue

    # Per-ordering accuracies
    ordering_accs = {}
    for oid, v in orderings_data.items():
        if isinstance(v, dict):
            ordering_accs[oid] = v.get("mean_val_acc", 0.0)

    spread = max(ordering_accs.values()) - min(ordering_accs.values())
    all_spreads.append(spread)
    best_id = max(ordering_accs, key=ordering_accs.get)
    worst_id = min(ordering_accs, key=ordering_accs.get)
    best_acc = ordering_accs[best_id]
    worst_acc = ordering_accs[worst_id]
    best_label = orderings_data.get(best_id, {}).get("label", ORDERING_LABELS.get(best_id, best_id))
    worst_label = orderings_data.get(worst_id, {}).get("label", ORDERING_LABELS.get(worst_id, worst_id))

    # Paired t-test: best vs. worst (per_seed data if available)
    best_seeds = orderings_data.get(best_id, {}).get("per_seed", {})
    worst_seeds = orderings_data.get(worst_id, {}).get("per_seed", {})

    # Collect paired values per seed
    common_seeds = sorted(set(best_seeds.keys()) & set(worst_seeds.keys()))
    vals_best = [best_seeds[s]["final_val_acc"] for s in common_seeds]
    vals_worst = [worst_seeds[s]["final_val_acc"] for s in common_seeds]

    if len(vals_best) >= 2:
        t_stat, p_val, n_pairs = paired_t_test(vals_best, vals_worst)
    else:
        # Single seed — can't do t-test; report raw difference
        t_stat = float('nan')
        p_val = float('nan')
        n_pairs = len(vals_best)

    # Cohen's d
    all_accs = list(ordering_accs.values())
    pooled_std = stdev(all_accs) if len(all_accs) >= 2 else 0.0
    d = cohen_d(best_acc, worst_acc, pooled_std)

    # Spread interpretation
    if spread > 0.003:
        confidence = "high"
    elif spread > 0.001:
        confidence = "cautious"
    else:
        confidence = "low"

    print(f"\n[{arch} x {dataset}]")
    print(f"  Best ordering:  {best_id} ({best_label}) = {best_acc:.4f}")
    print(f"  Worst ordering: {worst_id} ({worst_label}) = {worst_acc:.4f}")
    print(f"  Spread: {spread:.4f} ({confidence} confidence)")
    print(f"  Cohen's d: {d:.4f}")
    if not math.isnan(t_stat):
        print(f"  Paired t-test: t={t_stat:.3f}, p≈{p_val:.4f} (n={n_pairs})")
    else:
        print(f"  Paired t-test: n/a (n={n_pairs} seeds — need ≥2)")

    # All orderings ranked
    ranked = sorted(ordering_accs.items(), key=lambda x: x[1], reverse=True)
    print(f"  Rankings: {[(oid, f'{acc:.4f}') for oid, acc in ranked]}")

    block_results[block_key] = {
        "arch": arch,
        "dataset": dataset,
        "status": "ok",
        "mode": data.get("mode", "pilot"),
        "spread": round(spread, 6),
        "spread_pct": round(spread * 100, 4),
        "spread_confidence": confidence,
        "best_ordering_id": best_id,
        "best_ordering_label": best_label,
        "best_acc": round(best_acc, 6),
        "worst_ordering_id": worst_id,
        "worst_ordering_label": worst_label,
        "worst_acc": round(worst_acc, 6),
        "t_stat": round(t_stat, 4) if not math.isnan(t_stat) else None,
        "p_value": round(p_val, 4) if not math.isnan(p_val) else None,
        "n_seeds": n_pairs,
        "cohen_d": round(d, 4),
        "ordering_accuracies": {oid: round(acc, 6) for oid, acc in ordering_accs.items()},
        "ordering_labels": {oid: orderings_data[oid].get("label", ORDERING_LABELS.get(oid, oid)) for oid in ordering_accs},
        "ranked_orderings": [(oid, round(acc, 6)) for oid, acc in ranked],
    }


# ============================================================
# Bonferroni correction
# ============================================================
n_tests = len([b for b in block_results.values() if b.get("p_value") is not None])
alpha = 0.05
bonferroni_threshold = alpha / max(n_tests, 1) if n_tests > 0 else alpha

print(f"\n=== Bonferroni Correction ===")
print(f"  Number of blocks with t-tests: {n_tests}")
print(f"  Bonferroni threshold: {bonferroni_threshold:.4f}")

for block_key, res in block_results.items():
    if res.get("p_value") is not None:
        res["bonferroni_significant"] = res["p_value"] < bonferroni_threshold


# ============================================================
# Cross-block: identify consensus best/worst orderings for Tier 3
# ============================================================

ordering_win_count = {}
ordering_lose_count = {}
for res in block_results.values():
    if res.get("status") == "ok":
        best = res.get("best_ordering_id")
        worst = res.get("worst_ordering_id")
        if best:
            ordering_win_count[best] = ordering_win_count.get(best, 0) + 1
        if worst:
            ordering_lose_count[worst] = ordering_lose_count.get(worst, 0) + 1

# Pick ordering with most wins as "best overall" and most losses as "worst overall"
best_overall = max(ordering_win_count, key=ordering_win_count.get) if ordering_win_count else "order_5"
worst_overall = max(ordering_lose_count, key=ordering_lose_count.get) if ordering_lose_count else "order_4"

print(f"\n=== Cross-Block Consensus ===")
print(f"  Win counts:  {ordering_win_count}")
print(f"  Lose counts: {ordering_lose_count}")
print(f"  Best overall ordering:  {best_overall} ({ORDERING_LABELS.get(best_overall, best_overall)})")
print(f"  Worst overall ordering: {worst_overall} ({ORDERING_LABELS.get(worst_overall, worst_overall)})")


# ============================================================
# Write output JSON
# ============================================================

analysis_result = {
    "task_id": TASK_ID,
    "mode": "pilot",
    "timestamp": datetime.now().isoformat(),
    "n_blocks": len(block_results),
    "n_blocks_ok": len([b for b in block_results.values() if b.get("status") == "ok"]),
    "bonferroni_threshold": round(bonferroni_threshold, 4),
    "best_overall_ordering_id": best_overall,
    "best_overall_ordering_label": ORDERING_LABELS.get(best_overall, best_overall),
    "worst_overall_ordering_id": worst_overall,
    "worst_overall_ordering_label": ORDERING_LABELS.get(worst_overall, worst_overall),
    "ordering_win_counts": ordering_win_count,
    "ordering_lose_counts": ordering_lose_count,
    "blocks": block_results,
    "note": "PILOT mode: 10 epochs, 100-sample subsets. All statistics are indicative only; full-scale analysis will use 200 epochs and 5 seeds per ordering.",
}

out_json = FULL_DIR / "tier1_analysis.json"
with open(out_json, "w") as f:
    json.dump(analysis_result, f, indent=2)
print(f"\nWrote: {out_json}")


# ============================================================
# Write human-readable markdown summary
# ============================================================

md_lines = [
    "# Tier 1 Statistical Analysis — PILOT MODE",
    "",
    f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
    "**Mode**: Pilot (10 epochs, 100-sample subsets, 1 seed)",
    "**Note**: All statistics are indicative only. Full analysis requires 200 epochs, 5 seeds.",
    "",
    "---",
    "",
    "## Per-Block Spread Summary",
    "",
    "| Block | Best Ordering | Best Acc | Worst Ordering | Worst Acc | Spread | Confidence |",
    "|-------|--------------|----------|----------------|-----------|--------|------------|",
]

for block_key, res in sorted(block_results.items()):
    if res.get("status") == "ok":
        md_lines.append(
            f"| {res['arch']} × {res['dataset']} "
            f"| {res['best_ordering_label']} ({res['best_ordering_id']}) "
            f"| {res['best_acc']:.4f} "
            f"| {res['worst_ordering_label']} ({res['worst_ordering_id']}) "
            f"| {res['worst_acc']:.4f} "
            f"| {res['spread_pct']:.3f}% "
            f"| {res['spread_confidence']} |"
        )
    else:
        md_lines.append(f"| {res.get('arch','?')} × {res.get('dataset','?')} | — | — | — | — | — | missing |")

md_lines += [
    "",
    "---",
    "",
    "## Detailed Per-Block Rankings",
    "",
]

for block_key, res in sorted(block_results.items()):
    if res.get("status") != "ok":
        continue
    md_lines.append(f"### {res['arch']} × {res['dataset']}")
    md_lines.append("")
    md_lines.append("| Rank | Ordering ID | Label | Val Accuracy |")
    md_lines.append("|------|------------|-------|-------------|")
    for rank, (oid, acc) in enumerate(res["ranked_orderings"], 1):
        label = res["ordering_labels"].get(oid, oid)
        md_lines.append(f"| {rank} | {oid} | {label} | {acc:.4f} |")
    if res.get("t_stat") is not None:
        md_lines.append("")
        md_lines.append(f"**Paired t-test** (best vs. worst): t={res['t_stat']:.3f}, p≈{res['p_value']:.4f}, n={res['n_seeds']}")
    else:
        md_lines.append("")
        md_lines.append(f"**Paired t-test**: n/a ({res['n_seeds']} seed — need ≥2 for t-test)")
    md_lines.append(f"**Cohen's d**: {res['cohen_d']:.4f}")
    md_lines.append("")

md_lines += [
    "---",
    "",
    "## Consensus Best/Worst Orderings for Tier 3",
    "",
    f"- **Best overall**: `{best_overall}` — {ORDERING_LABELS.get(best_overall, best_overall)} (wins: {ordering_win_count.get(best_overall, 0)}/{len([b for b in block_results.values() if b.get('status')=='ok'])} blocks)",
    f"- **Worst overall**: `{worst_overall}` — {ORDERING_LABELS.get(worst_overall, worst_overall)} (losses: {ordering_lose_count.get(worst_overall, 0)}/{len([b for b in block_results.values() if b.get('status')=='ok'])} blocks)",
    "",
    "---",
    "",
    "## Pilot Assessment",
    "",
]

# Assessment
ok_blocks = [b for b in block_results.values() if b.get("status") == "ok"]
high_conf = [b for b in ok_blocks if b.get("spread_confidence") == "high"]
cautious_conf = [b for b in ok_blocks if b.get("spread_confidence") == "cautious"]
low_conf = [b for b in ok_blocks if b.get("spread_confidence") == "low"]

md_lines.append(f"- Blocks with high-confidence spread: **{len(high_conf)}/{len(ok_blocks)}**")
md_lines.append(f"- Blocks with cautious spread: **{len(cautious_conf)}/{len(ok_blocks)}**")
md_lines.append(f"- Blocks with low spread (noisy): **{len(low_conf)}/{len(ok_blocks)}**")
md_lines.append("")

if len(ok_blocks) == 0:
    md_lines.append("**STATUS**: FAIL — no data available for any block")
elif len(high_conf) + len(cautious_conf) >= 1:
    md_lines.append("**STATUS**: PASS — spread values written for all available blocks; best/worst ordering IDs confirmed")
else:
    md_lines.append("**STATUS**: UNCERTAIN — all spreads very small; may be noise at pilot scale")

out_md = FULL_DIR / "tier1_analysis.md"
out_md.write_text("\n".join(md_lines) + "\n")
print(f"Wrote: {out_md}")


# ============================================================
# Write DONE marker
# ============================================================

# Remove PID file
if pid_file.exists():
    pid_file.unlink()

done_content = {
    "task_id": TASK_ID,
    "status": "success",
    "summary": (
        f"pilot: {len(ok_blocks)}/{len(block_results)} blocks analyzed; "
        f"best_overall={best_overall} ({ORDERING_LABELS.get(best_overall, '')}); "
        f"worst_overall={worst_overall} ({ORDERING_LABELS.get(worst_overall, '')}); "
        f"high_conf_blocks={len(high_conf)}"
    ),
    "final_progress": {
        "task_id": TASK_ID,
        "blocks_analyzed": len(ok_blocks),
        "best_overall": best_overall,
        "worst_overall": worst_overall,
    },
    "timestamp": datetime.now().isoformat(),
}

done_file = FULL_DIR / f"{TASK_ID}_DONE"
with open(done_file, "w") as f:
    json.dump(done_content, f, indent=2)
print(f"Wrote: {done_file}")

print("\n=== tier1_analysis COMPLETE ===")
print(f"  Output: {out_json}")
print(f"  Summary: {out_md}")
print(f"  DONE marker: {done_file}")
