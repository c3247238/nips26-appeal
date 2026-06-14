#!/usr/bin/env python3
"""
Full: Cross-Layer Correlation Analysis
Aggregates absorption, steering, and probing data across layers.
Computes correlations and tests H3 (consistency across layers).
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

# Configuration
SEED = 42
RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/current/exp/results/full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Task identification
TASK_ID = "full_correlation"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

PID_FILE.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = {
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress))

def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except:
            pass
    marker = {
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }
    DONE_FILE.write_text(json.dumps(marker))

print(f"[{TASK_ID}] Starting cross-layer correlation analysis")
report_progress(0, 3, metric={"stage": "loading_data"})

# Load all experiment results
layers_data = {}

# Layer 10 (adapted from l12)
for layer, suffix in [(10, "l12"), (11, "l16")]:
    absorption_file = RESULTS_DIR / f"absorption_layer{layer}_32k.json"
    steering_file = RESULTS_DIR / f"steering_layer{layer}_32k.json"
    probing_file = RESULTS_DIR / f"probing_layer{layer}_32k.json"

    data = {"layer": layer, "suffix": suffix}

    if absorption_file.exists():
        with open(absorption_file) as f:
            data["absorption"] = json.load(f)

    if steering_file.exists():
        with open(steering_file) as f:
            data["steering"] = json.load(f)

    if probing_file.exists():
        with open(probing_file) as f:
            data["probing"] = json.load(f)

    layers_data[layer] = data

print(f"  Loaded data for layers: {list(layers_data.keys())}")

# Step 1: Compute correlations per layer
print(f"\nStep 1: Computing correlations per layer...")
report_progress(1, 3, metric={"stage": "computing_correlations"})

layer_correlations = {}

for layer, data in layers_data.items():
    if "absorption" not in data or "steering" not in data:
        continue

    abs_results = data["absorption"].get("results", {})
    steer_results = data["steering"].get("steering_results", {})

    # Collect paired data
    letters = sorted(set(abs_results.keys()) & set(steer_results.keys()))

    abs_rates = [abs_results[l]["absorption_rate"] for l in letters]
    mse_vals = [steer_results[l]["mean_mse"] for l in letters]
    cos_vals = [steer_results[l]["mean_cosine"] for l in letters]
    act_vals = [steer_results[l]["mean_main_activation"] for l in letters]

    # Correlations
    pearson_mse, p_mse = stats.pearsonr(abs_rates, mse_vals) if len(abs_rates) > 2 else (0.0, 1.0)
    pearson_cos, p_cos = stats.pearsonr(abs_rates, cos_vals) if len(abs_rates) > 2 else (0.0, 1.0)
    pearson_act, p_act = stats.pearsonr(abs_rates, act_vals) if len(abs_rates) > 2 else (0.0, 1.0)

    layer_correlations[layer] = {
        "n_letters": len(letters),
        "absorption_vs_mse": {"r": float(pearson_mse), "p": float(p_mse)},
        "absorption_vs_cosine": {"r": float(pearson_cos), "p": float(p_cos)},
        "absorption_vs_activation": {"r": float(pearson_act), "p": float(p_act)},
    }

    print(f"  Layer {layer}:")
    print(f"    Absorption vs MSE: r={pearson_mse:.3f}, p={p_mse:.3f}")
    print(f"    Absorption vs Cosine: r={pearson_cos:.3f}, p={p_cos:.3f}")
    print(f"    Absorption vs Activation: r={pearson_act:.3f}, p={p_act:.3f}")

# Step 2: Test H3 - Consistency across layers
print(f"\nStep 2: Testing H3 (consistency across layers)...")
report_progress(2, 3, metric={"stage": "testing_h3"})

if len(layer_correlations) >= 2:
    mse_rs = [layer_correlations[l]["absorption_vs_mse"]["r"] for l in layer_correlations]
    cos_rs = [layer_correlations[l]["absorption_vs_cosine"]["r"] for l in layer_correlations]
    act_rs = [layer_correlations[l]["absorption_vs_activation"]["r"] for l in layer_correlations]

    # Coefficient of variation
    def cv(vals):
        return np.std(vals) / abs(np.mean(vals)) if np.mean(vals) != 0 else float('inf')

    cv_mse = cv(mse_rs)
    cv_cos = cv(cos_rs)
    cv_act = cv(act_rs)

    h3_passes = {
        "mse_cv_lt_0.5": cv_mse < 0.5,
        "cosine_cv_lt_0.5": cv_cos < 0.5,
        "activation_cv_lt_0.5": cv_act < 0.5,
    }

    print(f"  CV of MSE correlations: {cv_mse:.3f} (pass={h3_passes['mse_cv_lt_0.5']})")
    print(f"  CV of Cosine correlations: {cv_cos:.3f} (pass={h3_passes['cosine_cv_lt_0.5']})")
    print(f"  CV of Activation correlations: {cv_act:.3f} (pass={h3_passes['activation_cv_lt_0.5']})")
else:
    h3_passes = {"insufficient_layers": True}
    cv_mse = cv_cos = cv_act = None
    print(f"  Insufficient layers for H3 test")

# Step 3: Aggregate summary
print(f"\nStep 3: Generating summary...")
report_progress(3, 3, metric={"stage": "summary"})

# Save results
output = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "layers_analyzed": list(layers_data.keys()),
    "layer_correlations": layer_correlations,
    "h3_test": {
        "cv_mse": float(cv_mse) if cv_mse is not None else None,
        "cv_cosine": float(cv_cos) if cv_cos is not None else None,
        "cv_activation": float(cv_act) if cv_act is not None else None,
        "passes": h3_passes,
    },
}

output_file = RESULTS_DIR / "correlation_analysis_full.json"
with open(output_file, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {output_file}")

# Summary markdown
summary_md = f"""# Cross-Layer Correlation Analysis

## Layers Analyzed
{', '.join(f'Layer {l}' for l in layers_data.keys())}

## Per-Layer Correlations

"""

for layer, corrs in layer_correlations.items():
    summary_md += f"""### Layer {layer}
- Absorption vs MSE: r={corrs['absorption_vs_mse']['r']:.3f}, p={corrs['absorption_vs_mse']['p']:.3f}
- Absorption vs Cosine: r={corrs['absorption_vs_cosine']['r']:.3f}, p={corrs['absorption_vs_cosine']['p']:.3f}
- Absorption vs Activation: r={corrs['absorption_vs_activation']['r']:.3f}, p={corrs['absorption_vs_activation']['p']:.3f}

"""

cv_mse_str = f"{cv_mse:.3f}" if cv_mse is not None else "N/A"
cv_cos_str = f"{cv_cos:.3f}" if cv_cos is not None else "N/A"
cv_act_str = f"{cv_act:.3f}" if cv_act is not None else "N/A"

summary_md += f"""## H3 Test (Consistency Across Layers)
- CV of MSE correlations: {cv_mse_str}
- CV of Cosine correlations: {cv_cos_str}
- CV of Activation correlations: {cv_act_str}

## H3 Passes
{json.dumps(h3_passes, indent=2)}
"""

summary_file = RESULTS_DIR / "correlation_analysis_full.md"
summary_file.write_text(summary_md)
print(f"Saved: {summary_file}")

summary_text = f"Correlation analysis complete. Layers: {list(layers_data.keys())}. H3 passes: {h3_passes}"

mark_done("success", summary_text)
print(f"\n{'='*60}")
print(f"Full correlation analysis COMPLETE")
print(f"{'='*60}")

if PID_FILE.exists():
    PID_FILE.unlink()
