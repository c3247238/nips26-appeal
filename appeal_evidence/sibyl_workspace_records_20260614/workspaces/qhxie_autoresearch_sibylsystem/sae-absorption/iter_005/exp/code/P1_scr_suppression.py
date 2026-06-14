#!/usr/bin/env python3
"""
P1_scr_suppression: SCR Suppression Variable Diagnosis

Sequentially add covariates (width-only, layer-only, arch-only, L0-only)
and track how SCR partial correlation changes from bivariate (-0.431) to
full partial (-0.677). Identify which covariate produces the suppression jump.

This is a zero-GPU CPU analysis task on the iter_004 54-SAE dataset.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

# ============================================================
# Configuration
# ============================================================
WORKSPACE = Path(os.environ.get(
    "WORKSPACE",
    "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current"
))
DATA_PATH = Path(os.environ.get(
    "DATA_PATH",
    "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_004/exp/results/full/C3A_saebench_corr.json"
))
TASK_ID = "P1_scr_suppression"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

start_time = datetime.now()

# ============================================================
# PID file for system recovery
# ============================================================
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(task_id, results_dir, epoch, total_epochs,
                    step=0, total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file."""
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


# ============================================================
# Load data
# ============================================================
report_progress(TASK_ID, RESULTS_DIR, 0, 5, step=0, total_steps=5,
                metric={"phase": "loading_data"})

print(f"[{TASK_ID}] Loading data from {DATA_PATH}")
with open(DATA_PATH) as f:
    data = json.load(f)

records = data["raw_records"]
df = pd.DataFrame(records)
print(f"[{TASK_ID}] Loaded {len(df)} SAEs")

# ============================================================
# Prepare variables
# ============================================================
# Rename columns for consistency
df = df.rename(columns={
    "scr_score": "scr",
    "absorption_score": "absorption",
})

# Compute derived features
df["log_width"] = np.log2(df["width_int"])
# Filter to records that have SCR data
df_scr = df.dropna(subset=["scr"]).copy()
print(f"[{TASK_ID}] SAEs with SCR data: {len(df_scr)}")

# Prepare L0: filter to non-null L0 for L0-controlled analyses
df_scr_l0 = df_scr.dropna(subset=["l0"]).copy()
df_scr_l0["log_l0"] = np.log2(df_scr_l0["l0"])
print(f"[{TASK_ID}] SAEs with SCR + L0 data: {len(df_scr_l0)}")

# Encode arch_class as numeric
df_scr["arch_numeric"] = pd.Categorical(df_scr["arch_class"]).codes
df_scr_l0["arch_numeric"] = pd.Categorical(df_scr_l0["arch_class"]).codes

# Check if arch_class is constant in L0-filtered subset
n_arch_classes_l0 = df_scr_l0["arch_class"].nunique()
print(f"[{TASK_ID}] Number of arch_classes in L0-filtered SCR subset: {n_arch_classes_l0}")

report_progress(TASK_ID, RESULTS_DIR, 1, 5, step=1, total_steps=5,
                metric={"phase": "data_prepared", "n_scr": len(df_scr), "n_scr_l0": len(df_scr_l0)})


# ============================================================
# Helper: compute partial correlation using pingouin
# ============================================================
def compute_partial_corr(dataframe, x, y, covariates):
    """Compute partial Pearson correlation between x and y controlling for covariates."""
    if not covariates:
        # Bivariate correlation
        r, p = stats.pearsonr(dataframe[x], dataframe[y])
        return {"r": r, "p": p, "n": len(dataframe), "covariates": []}

    # Use pingouin for partial correlation
    result = pg.partial_corr(
        data=dataframe, x=x, y=y, covar=covariates, method="pearson"
    )
    r_val = result["r"].values[0]
    p_val = result["p_val"].values[0] if "p_val" in result.columns else result["p-val"].values[0]
    n_val = result["n"].values[0]
    return {"r": r_val, "p": p_val, "n": n_val, "covariates": covariates}


# ============================================================
# Step 1: Bivariate SCR-absorption correlation (full dataset with SCR)
# ============================================================
print(f"\n{'='*60}")
print(f"[{TASK_ID}] Step 1: Bivariate correlation (no covariates)")
print(f"{'='*60}")

bivariate = compute_partial_corr(df_scr, "absorption", "scr", [])
print(f"  Bivariate Pearson r = {bivariate['r']:.4f}, p = {bivariate['p']:.6f}, n = {bivariate['n']}")

report_progress(TASK_ID, RESULTS_DIR, 2, 5, step=2, total_steps=5,
                metric={"phase": "bivariate_done", "bivariate_r": bivariate["r"]})


# ============================================================
# Step 2: Sequential covariate addition on FULL dataset (with arch_class)
# ============================================================
print(f"\n{'='*60}")
print(f"[{TASK_ID}] Step 2: Sequential covariate addition (full 49-SAE dataset)")
print(f"{'='*60}")

# Define individual covariates and all combinations for sequential addition
individual_covariates = {
    "log_width_only": ["log_width"],
    "layer_only": ["layer"],
    "arch_only": ["arch_numeric"],
}

# Cumulative sequences
cumulative_sequences = [
    ("+ log_width", ["log_width"]),
    ("+ log_width + layer", ["log_width", "layer"]),
    ("+ log_width + layer + arch_class", ["log_width", "layer", "arch_numeric"]),
]

# Sequential addition results (full dataset)
sequential_results_full = []

# Bivariate
sequential_results_full.append({
    "step": "bivariate (no covariates)",
    "covariates": [],
    "partial_r": bivariate["r"],
    "partial_p": bivariate["p"],
    "n": bivariate["n"],
    "delta_from_bivariate": 0.0,
})
print(f"  bivariate: r = {bivariate['r']:.4f}")

# Individual single-covariate additions
for label, covs in individual_covariates.items():
    result = compute_partial_corr(df_scr, "absorption", "scr", covs)
    delta = result["r"] - bivariate["r"]
    sequential_results_full.append({
        "step": label,
        "covariates": covs,
        "partial_r": result["r"],
        "partial_p": result["p"],
        "n": result["n"],
        "delta_from_bivariate": delta,
    })
    print(f"  {label}: r = {result['r']:.4f}, delta = {delta:+.4f}")

# Cumulative addition
for label, covs in cumulative_sequences:
    result = compute_partial_corr(df_scr, "absorption", "scr", covs)
    delta = result["r"] - bivariate["r"]
    sequential_results_full.append({
        "step": f"cumulative {label}",
        "covariates": covs,
        "partial_r": result["r"],
        "partial_p": result["p"],
        "n": result["n"],
        "delta_from_bivariate": delta,
    })
    print(f"  cumulative {label}: r = {result['r']:.4f}, delta = {delta:+.4f}")


# ============================================================
# Step 3: Sequential covariate addition on L0-filtered dataset
# ============================================================
print(f"\n{'='*60}")
print(f"[{TASK_ID}] Step 3: Sequential covariate addition (L0-filtered dataset)")
print(f"{'='*60}")

# Bivariate on L0-filtered
bivariate_l0 = compute_partial_corr(df_scr_l0, "absorption", "scr", [])
print(f"  bivariate (L0-filtered): r = {bivariate_l0['r']:.4f}, n = {bivariate_l0['n']}")

# Individual single-covariate additions (with L0)
individual_covariates_l0 = {
    "log_width_only": ["log_width"],
    "layer_only": ["layer"],
    "log_l0_only": ["log_l0"],
}

# All permutations of addition order to trace each covariate's marginal effect
addition_orders = [
    # Standard order: width -> layer -> L0
    [
        ("+ log_width", ["log_width"]),
        ("+ log_width + layer", ["log_width", "layer"]),
        ("+ log_width + layer + log_l0", ["log_width", "layer", "log_l0"]),
    ],
    # L0 first
    [
        ("+ log_l0", ["log_l0"]),
        ("+ log_l0 + log_width", ["log_l0", "log_width"]),
        ("+ log_l0 + log_width + layer", ["log_l0", "log_width", "layer"]),
    ],
    # Layer first
    [
        ("+ layer", ["layer"]),
        ("+ layer + log_width", ["layer", "log_width"]),
        ("+ layer + log_width + log_l0", ["layer", "log_width", "log_l0"]),
    ],
]

sequential_results_l0 = []

# Bivariate
sequential_results_l0.append({
    "step": "bivariate (no covariates)",
    "covariates": [],
    "partial_r": bivariate_l0["r"],
    "partial_p": bivariate_l0["p"],
    "n": bivariate_l0["n"],
    "delta_from_bivariate": 0.0,
})

# Individual single-covariate results
for label, covs in individual_covariates_l0.items():
    result = compute_partial_corr(df_scr_l0, "absorption", "scr", covs)
    delta = result["r"] - bivariate_l0["r"]
    sequential_results_l0.append({
        "step": label,
        "covariates": covs,
        "partial_r": result["r"],
        "partial_p": result["p"],
        "n": result["n"],
        "delta_from_bivariate": delta,
    })
    print(f"  {label}: r = {result['r']:.4f}, delta = {delta:+.4f}")

# All addition orders
all_orders_results = {}
for i, order in enumerate(addition_orders):
    order_name = f"order_{i}"
    order_results = [{
        "step": "bivariate",
        "partial_r": bivariate_l0["r"],
        "delta_from_bivariate": 0.0,
    }]
    prev_r = bivariate_l0["r"]
    for label, covs in order:
        result = compute_partial_corr(df_scr_l0, "absorption", "scr", covs)
        delta = result["r"] - bivariate_l0["r"]
        marginal = result["r"] - prev_r
        order_results.append({
            "step": label,
            "covariates": covs,
            "partial_r": result["r"],
            "partial_p": result["p"],
            "n": result["n"],
            "delta_from_bivariate": delta,
            "marginal_change": marginal,
        })
        print(f"  [{order_name}] {label}: r = {result['r']:.4f}, delta = {delta:+.4f}, marginal = {marginal:+.4f}")
        prev_r = result["r"]
    all_orders_results[order_name] = order_results

report_progress(TASK_ID, RESULTS_DIR, 3, 5, step=3, total_steps=5,
                metric={"phase": "sequential_addition_done"})


# ============================================================
# Step 4: Pairwise covariate combinations (L0-filtered)
# ============================================================
print(f"\n{'='*60}")
print(f"[{TASK_ID}] Step 4: All pairwise covariate combinations")
print(f"{'='*60}")

from itertools import combinations

pairwise_vars = ["log_width", "layer", "log_l0"]
pairwise_results = {}

for r_count in range(1, len(pairwise_vars) + 1):
    for combo in combinations(pairwise_vars, r_count):
        covs = list(combo)
        label = " + ".join(covs)
        result = compute_partial_corr(df_scr_l0, "absorption", "scr", covs)
        delta = result["r"] - bivariate_l0["r"]
        pairwise_results[label] = {
            "covariates": covs,
            "partial_r": result["r"],
            "partial_p": result["p"],
            "n": result["n"],
            "delta_from_bivariate": delta,
        }
        print(f"  [{label}]: r = {result['r']:.4f}, delta = {delta:+.4f}")


# ============================================================
# Step 5: Identify suppression variable
# ============================================================
print(f"\n{'='*60}")
print(f"[{TASK_ID}] Step 5: Identify suppression variable")
print(f"{'='*60}")

# Suppression occurs when adding a covariate makes the partial r MORE negative
# (i.e., partial r moves further from 0, not closer)
# The bivariate r is -0.431 (negative), the full partial is -0.677 (more negative)
# So we look for the covariate that causes the largest NEGATIVE shift

# Single-covariate effects on L0-filtered data
single_effects = {}
for label, covs in individual_covariates_l0.items():
    result = compute_partial_corr(df_scr_l0, "absorption", "scr", covs)
    delta = result["r"] - bivariate_l0["r"]
    single_effects[label] = {
        "partial_r": result["r"],
        "delta": delta,
        "abs_delta": abs(delta),
    }

# Sort by magnitude of delta (negative = suppression enhancement)
sorted_effects = sorted(single_effects.items(), key=lambda x: x[1]["delta"])

print("\n  Single-covariate effects (sorted by delta, most negative first):")
for label, eff in sorted_effects:
    direction = "SUPPRESSION" if eff["delta"] < 0 else "CONFOUND REMOVAL"
    print(f"    {label}: delta = {eff['delta']:+.4f} ({direction})")

# Identify the primary suppression variable
suppression_variables = [(k, v) for k, v in sorted_effects if v["delta"] < 0]
primary_suppressor = suppression_variables[0] if suppression_variables else None

# Marginal effects: what does each covariate add AFTER the others?
print("\n  Marginal effects (adding last):")
marginal_effects = {}
for var in pairwise_vars:
    others = [v for v in pairwise_vars if v != var]
    # Without this variable
    without = compute_partial_corr(df_scr_l0, "absorption", "scr", others)
    # With all
    with_all = compute_partial_corr(df_scr_l0, "absorption", "scr", pairwise_vars)
    marginal = with_all["r"] - without["r"]
    marginal_effects[var] = {
        "partial_r_without": without["r"],
        "partial_r_with_all": with_all["r"],
        "marginal_effect": marginal,
    }
    direction = "SUPPRESSION" if marginal < 0 else "CONFOUND REMOVAL"
    print(f"    Adding {var} last: {without['r']:.4f} -> {with_all['r']:.4f}, marginal = {marginal:+.4f} ({direction})")

report_progress(TASK_ID, RESULTS_DIR, 4, 5, step=4, total_steps=5,
                metric={"phase": "suppression_identified"})


# ============================================================
# Step 6: Additional diagnostic - semi-partial correlations
# ============================================================
print(f"\n{'='*60}")
print(f"[{TASK_ID}] Step 6: Semi-partial correlation diagnostic")
print(f"{'='*60}")

# Semi-partial: control covariate's effect on one variable only
# This helps distinguish suppression from confounding
semi_partial_results = {}
for var in pairwise_vars:
    # Semi-partial: control var's effect on absorption only
    try:
        sp = pg.partial_corr(
            data=df_scr_l0, x="absorption", y="scr",
            x_covar=[var], method="pearson"
        )
        p_col = "p_val" if "p_val" in sp.columns else "p-val"
        semi_partial_results[f"control_{var}_on_absorption"] = {
            "r": sp["r"].values[0],
            "p": sp[p_col].values[0],
        }
        print(f"  Control {var} on absorption only: r = {sp['r'].values[0]:.4f}")
    except Exception as e:
        semi_partial_results[f"control_{var}_on_absorption"] = {"error": str(e)}
        print(f"  Control {var} on absorption: ERROR - {e}")

    try:
        sp2 = pg.partial_corr(
            data=df_scr_l0, x="absorption", y="scr",
            y_covar=[var], method="pearson"
        )
        p_col2 = "p_val" if "p_val" in sp2.columns else "p-val"
        semi_partial_results[f"control_{var}_on_scr"] = {
            "r": sp2["r"].values[0],
            "p": sp2[p_col2].values[0],
        }
        print(f"  Control {var} on SCR only:        r = {sp2['r'].values[0]:.4f}")
    except Exception as e:
        semi_partial_results[f"control_{var}_on_scr"] = {"error": str(e)}
        print(f"  Control {var} on SCR: ERROR - {e}")


# ============================================================
# Step 7: Correlation structure among covariates
# ============================================================
print(f"\n{'='*60}")
print(f"[{TASK_ID}] Step 7: Covariate correlation structure")
print(f"{'='*60}")

covariate_corrs = {}
vars_of_interest = ["absorption", "scr", "log_width", "layer", "log_l0"]
for i, v1 in enumerate(vars_of_interest):
    for v2 in vars_of_interest[i+1:]:
        r_val, p_val = stats.pearsonr(df_scr_l0[v1], df_scr_l0[v2])
        covariate_corrs[f"{v1}_vs_{v2}"] = {"r": r_val, "p": p_val}
        print(f"  {v1} vs {v2}: r = {r_val:.4f}, p = {p_val:.6f}")


# ============================================================
# Step 8: Explanation of suppression mechanism
# ============================================================
print(f"\n{'='*60}")
print(f"[{TASK_ID}] Step 8: Suppression mechanism analysis")
print(f"{'='*60}")

# Classical suppression: A variable S is a suppressor if
# 1. Adding S increases |r(X,Y)|
# 2. S correlates with X (predictor) more than with Y (outcome), or
#    S shares variance with X that is NOT related to Y, and removing
#    this irrelevant variance from X reveals the true X-Y relationship

# Check: which covariate is correlated with absorption but NOT with SCR?
absorption_corrs = {}
scr_corrs = {}
for var in pairwise_vars:
    r_abs, _ = stats.pearsonr(df_scr_l0["absorption"], df_scr_l0[var])
    r_scr, _ = stats.pearsonr(df_scr_l0["scr"], df_scr_l0[var])
    absorption_corrs[var] = r_abs
    scr_corrs[var] = r_scr
    ratio = abs(r_abs) / max(abs(r_scr), 0.001)
    print(f"  {var}: corr with absorption = {r_abs:.4f}, corr with SCR = {r_scr:.4f}, ratio = {ratio:.2f}")
    if abs(r_abs) > abs(r_scr):
        print(f"    -> More related to absorption than SCR (potential suppressor)")
    else:
        print(f"    -> More related to SCR than absorption (potential confound)")

# Determine suppression interpretation
interpretation_lines = []
if primary_suppressor:
    var_name = primary_suppressor[0].replace("_only", "")
    delta = primary_suppressor[1]["delta"]
    interpretation_lines.append(
        f"Primary suppressor: {var_name} (adding it shifts r by {delta:+.4f})"
    )
    interpretation_lines.append(
        "Suppression mechanism: this variable shares variance with absorption "
        "that is unrelated to SCR. Removing this irrelevant variance from "
        "absorption reveals a stronger true absorption-SCR relationship."
    )
else:
    interpretation_lines.append("No clear suppression variable identified.")

# Check all three orders to see consistency
# Identify which individual covariates consistently produce the biggest marginal effects
marginal_summary = {}
for var in pairwise_vars:
    # Get this variable's marginal effect from all three order analyses
    effects = []
    for order_name, order_results in all_orders_results.items():
        for step in order_results:
            if "covariates" in step and step["covariates"]:
                if var in step["covariates"] and "marginal_change" in step:
                    # Check if this is where the variable was added
                    prev_covs = [c for c in step["covariates"] if c != var]
                    if len(step["covariates"]) == len(prev_covs) + 1:
                        effects.append(step["marginal_change"])
    if effects:
        marginal_summary[var] = {
            "mean_marginal": np.mean(effects),
            "all_marginals": effects,
        }

report_progress(TASK_ID, RESULTS_DIR, 5, 5, step=5, total_steps=5,
                metric={"phase": "analysis_complete"})


# ============================================================
# Compile results
# ============================================================
end_time = datetime.now()
runtime_seconds = (end_time - start_time).total_seconds()

# Determine the definitive suppression variable
suppression_diagnosis = {
    "bivariate_r": bivariate_l0["r"],
    "full_partial_r": pairwise_results[" + ".join(pairwise_vars)]["partial_r"],
    "total_suppression_shift": pairwise_results[" + ".join(pairwise_vars)]["partial_r"] - bivariate_l0["r"],
}

# Find which single covariate produces biggest suppression
biggest_single_suppressor = None
biggest_delta = 0
for label, eff in single_effects.items():
    if eff["delta"] < biggest_delta:
        biggest_delta = eff["delta"]
        biggest_single_suppressor = label

suppression_diagnosis["biggest_single_suppressor"] = biggest_single_suppressor
suppression_diagnosis["biggest_single_delta"] = biggest_delta

# Find which marginal addition (adding last) produces biggest suppression
biggest_marginal_suppressor = None
biggest_marginal = 0
for var, eff in marginal_effects.items():
    if eff["marginal_effect"] < biggest_marginal:
        biggest_marginal = eff["marginal_effect"]
        biggest_marginal_suppressor = var

suppression_diagnosis["biggest_marginal_suppressor"] = biggest_marginal_suppressor
suppression_diagnosis["biggest_marginal_effect"] = biggest_marginal
suppression_diagnosis["interpretation"] = "\n".join(interpretation_lines)

# Pilot pass criteria: "Script identifies which covariate produces the suppression effect"
pilot_pass = biggest_single_suppressor is not None or biggest_marginal_suppressor is not None

result = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "timestamp": datetime.now().isoformat(),
    "data_source": str(DATA_PATH),
    "seed": SEED,
    "n_scr_total": len(df_scr),
    "n_scr_with_l0": len(df_scr_l0),
    "n_arch_classes_l0_filtered": n_arch_classes_l0,
    "arch_class_note": (
        f"After filtering to SAEs with known L0 + SCR, {n_arch_classes_l0} arch_class(es) remain. "
        + ("arch_class is constant and cannot serve as covariate." if n_arch_classes_l0 == 1
           else "arch_class is included as covariate.")
    ),
    "bivariate_correlations": {
        "full_dataset": {
            "n": bivariate["n"],
            "pearson_r": bivariate["r"],
            "pearson_p": bivariate["p"],
        },
        "l0_filtered": {
            "n": bivariate_l0["n"],
            "pearson_r": bivariate_l0["r"],
            "pearson_p": bivariate_l0["p"],
        },
    },
    "sequential_addition_full_dataset": sequential_results_full,
    "sequential_addition_l0_filtered": sequential_results_l0,
    "all_addition_orders": {
        k: [
            {key: (float(val) if isinstance(val, (np.floating, np.integer)) else val)
             for key, val in step.items()}
            for step in steps
        ]
        for k, steps in all_orders_results.items()
    },
    "all_pairwise_combinations": {
        k: {key: (float(val) if isinstance(val, (np.floating, np.integer)) else val)
            for key, val in v.items()}
        for k, v in pairwise_results.items()
    },
    "single_covariate_effects": {
        k: {key: float(val) if isinstance(val, (np.floating, np.integer)) else val
            for key, val in v.items()}
        for k, v in single_effects.items()
    },
    "marginal_effects_adding_last": {
        k: {key: float(val) if isinstance(val, (np.floating, np.integer)) else val
            for key, val in v.items()}
        for k, v in marginal_effects.items()
    },
    "semi_partial_correlations": {
        k: {key: float(val) if isinstance(val, (np.floating, np.integer)) else val
            for key, val in v.items()}
        for k, v in semi_partial_results.items()
    },
    "covariate_correlation_structure": {
        k: {key: float(val) if isinstance(val, (np.floating, np.integer)) else val
            for key, val in v.items()}
        for k, v in covariate_corrs.items()
    },
    "suppression_variable_correlations": {
        var: {
            "corr_with_absorption": float(absorption_corrs[var]),
            "corr_with_scr": float(scr_corrs[var]),
            "more_related_to": "absorption" if abs(absorption_corrs[var]) > abs(scr_corrs[var]) else "scr",
        }
        for var in pairwise_vars
    },
    "suppression_diagnosis": {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in suppression_diagnosis.items()
    },
    "marginal_effect_summary": {
        k: {key: (float(val) if isinstance(val, (np.floating, np.integer))
                  else [float(x) for x in val] if isinstance(val, list)
                  else val)
            for key, val in v.items()}
        for k, v in marginal_summary.items()
    },
    "pilot_pass_criteria": "Script identifies which covariate produces the suppression effect",
    "pilot_pass": pilot_pass,
    "runtime_seconds": runtime_seconds,
}

# Save results
output_path = RESULTS_DIR / f"{TASK_ID}.json"
with open(output_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\n[{TASK_ID}] Results saved to {output_path}")
print(f"[{TASK_ID}] Runtime: {runtime_seconds:.1f}s")
print(f"[{TASK_ID}] Pilot pass: {pilot_pass}")

# Print summary
print(f"\n{'='*60}")
print(f"SUMMARY: SCR Suppression Variable Diagnosis")
print(f"{'='*60}")
print(f"Bivariate absorption-SCR correlation (L0-filtered): {bivariate_l0['r']:.4f}")
print(f"Full partial (log_width + layer + log_l0):          {pairwise_results[' + '.join(pairwise_vars)]['partial_r']:.4f}")
print(f"Total suppression shift:                            {suppression_diagnosis['total_suppression_shift']:+.4f}")
print(f"")
print(f"Biggest single suppressor: {biggest_single_suppressor} (delta = {biggest_delta:+.4f})")
print(f"Biggest marginal (adding last): {biggest_marginal_suppressor} (marginal = {biggest_marginal:+.4f})")
print(f"")
print(f"Interpretation: {suppression_diagnosis['interpretation']}")

# Mark done
mark_task_done(TASK_ID, RESULTS_DIR, status="success",
               summary=f"SCR suppression analysis complete. Primary suppressor: {biggest_single_suppressor}. "
                       f"Shift from {bivariate_l0['r']:.4f} to {pairwise_results[' + '.join(pairwise_vars)]['partial_r']:.4f}.")
print(f"\n[{TASK_ID}] DONE marker written.")
