#!/usr/bin/env python3
"""Final tasks: ablations + analysis."""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

from sae_lens.synthetic.synthetic_model import SyntheticModel
from sae_lens.synthetic.synthetic_sae_runner import (
    SyntheticSAERunner, SyntheticSAERunnerConfig,
    SyntheticModelConfig, eval_sae_on_synthetic_data, LoggingConfig,
)
from sae_lens.synthetic.hierarchy import HierarchyConfig
from sae_lens import StandardTrainingSAEConfig, TopKTrainingSAEConfig, SAE

NUM_FEATURES = 1024
HIDDEN_DIM = 256
FEATURES_EXPANSION = 8
TOTAL_ROOT_NODES = 32
BRANCHING_FACTOR = 4
MAX_DEPTH = 3
TRAINING_TOKENS = 2_000_000
BATCH_SIZE = 1024
EVAL_SAMPLES = 100_000
SEEDS = [42, 123, 456, 789, 1011]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mark_done(task_id, results_dir, status="success", summary=""):
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({"task_id": task_id, "status": status, "summary": summary, "timestamp": datetime.now().isoformat()}))


def compute_absorption(sae, synthetic_model, num_test=10000, seed_off=999):
    sae.eval()
    device = next(sae.parameters()).device
    decoder = sae.W_dec
    gt_features = synthetic_model.feature_dict.feature_vectors
    with torch.no_grad():
        cos_sim = torch.nn.functional.cosine_similarity(
            decoder.unsqueeze(1), gt_features.unsqueeze(0), dim=2
        ).abs()
        latent_to_feature = cos_sim.argmax(dim=1).cpu().numpy()
    feat_to_latents = {}
    for li, fi in enumerate(latent_to_feature):
        feat_to_latents.setdefault(int(fi), []).append(li)
    hierarchy = synthetic_model.hierarchy
    pairs = []
    def extract(n):
        for c in n.children:
            pairs.append((int(n.feature_index), int(c.feature_index)))
            extract(c)
    for r in hierarchy.roots:
        extract(r)
    if not pairs:
        return {"absorption_rate": 0.0, "num_pairs": 0}
    torch.manual_seed(seed_off)
    np.random.seed(seed_off)
    parent_fires = 0
    absorption_count = 0
    thresh = 0.05
    with torch.no_grad():
        bs = 1024
        nb = (num_test + bs - 1) // bs
        for _ in range(nb):
            hidden, feature_acts = synthetic_model.sample_with_features(bs)
            hidden = hidden.to(device)
            feature_acts = feature_acts.to(device)
            sae_latents = sae.encode(hidden)
            for pi, ci in pairs:
                pf = feature_acts[:, pi] > 0
                if pf.sum() == 0:
                    continue
                parent_fires += pf.sum().item()
                child_latents = feat_to_latents.get(ci, [])
                if not child_latents:
                    continue
                child_acts = sae_latents[pf][:, child_latents]
                child_fires = (child_acts > thresh).any(dim=1)
                absorption_count += child_fires.sum().item()
    rate = absorption_count / parent_fires if parent_fires > 0 else 0.0
    return {"absorption_rate": rate, "num_pairs": len(pairs)}


def run_ort_no_penalty(task_id, results_dir):
    """OrtSAE without orthogonality penalty = Standard L1."""
    print(f"\n{'='*60}")
    print(f"OrtSAE No Penalty Ablation (task: {task_id})")
    print(f"{'='*60}")
    all_results = {}
    start = time.time()
    for seed in SEEDS:
        hierarchy_cfg = HierarchyConfig(total_root_nodes=TOTAL_ROOT_NODES, branching_factor=BRANCHING_FACTOR, max_depth=MAX_DEPTH)
        model_cfg = SyntheticModelConfig(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, hierarchy=hierarchy_cfg, seed=seed)
        synthetic_model = SyntheticModel(cfg=model_cfg)
        synthetic_model.to(DEVICE)

        d_sae = HIDDEN_DIM * FEATURES_EXPANSION
        # Standard L1 with same config as OrtSAE baseline
        sae_cfg = StandardTrainingSAEConfig(d_in=HIDDEN_DIM, d_sae=d_sae, l1_coefficient=5e-3, device=DEVICE, apply_b_dec_to_input=False)
        logger_cfg = LoggingConfig(log_to_wandb=False)
        runner_cfg = SyntheticSAERunnerConfig(synthetic_model=model_cfg, sae=sae_cfg, training_samples=TRAINING_TOKENS, batch_size=BATCH_SIZE, lr=1e-3, device=DEVICE, eval_frequency=0, eval_samples=EVAL_SAMPLES, run_final_eval=True, logger=logger_cfg)
        runner = SyntheticSAERunner(cfg=runner_cfg)
        result = runner.run()

        eval_result = eval_sae_on_synthetic_data(sae=result.sae, feature_dict=synthetic_model.feature_dict, activations_generator=synthetic_model.activation_generator, num_samples=EVAL_SAMPLES, batch_size=BATCH_SIZE)
        absorption = compute_absorption(result.sae, synthetic_model, seed_off=seed+999)

        all_results[f"seed_{seed}"] = {
            "absorption_rate": absorption["absorption_rate"],
            "l0_sparsity": eval_result.sae_l0,
            "feature_recovery_mcc": eval_result.mcc,
        }
        print(f"  seed={seed}: absorption={absorption['absorption_rate']:.4f}, L0={eval_result.sae_l0:.1f}")

    elapsed = time.time() - start
    output = Path(results_dir) / f"{task_id}_results.json"
    with open(output, "w") as f:
        json.dump({"task_id": task_id, "results": all_results, "elapsed_seconds": elapsed, "timestamp": datetime.now().isoformat()}, f, indent=2)
    mark_done(task_id, results_dir, status="success", summary=f"Ort no penalty: {len(SEEDS)} seeds")
    print(f"Results: {output}, time={elapsed:.1f}s")


def run_topk_as_relu(task_id, results_dir):
    """TopK as ReLU+L1 (same sparsity level)."""
    print(f"\n{'='*60}")
    print(f"TopK-as-ReLU Ablation (task: {task_id})")
    print(f"{'='*60}")
    all_results = {}
    start = time.time()
    for seed in SEEDS:
        hierarchy_cfg = HierarchyConfig(total_root_nodes=TOTAL_ROOT_NODES, branching_factor=BRANCHING_FACTOR, max_depth=MAX_DEPTH)
        model_cfg = SyntheticModelConfig(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, hierarchy=hierarchy_cfg, seed=seed)
        synthetic_model = SyntheticModel(cfg=model_cfg)
        synthetic_model.to(DEVICE)

        d_sae = HIDDEN_DIM * FEATURES_EXPANSION
        # ReLU+L1 tuned for L0 ~50 (matching TopK k=50)
        sae_cfg = StandardTrainingSAEConfig(d_in=HIDDEN_DIM, d_sae=d_sae, l1_coefficient=2e-2, device=DEVICE, apply_b_dec_to_input=False)
        logger_cfg = LoggingConfig(log_to_wandb=False)
        runner_cfg = SyntheticSAERunnerConfig(synthetic_model=model_cfg, sae=sae_cfg, training_samples=TRAINING_TOKENS, batch_size=BATCH_SIZE, lr=1e-3, device=DEVICE, eval_frequency=0, eval_samples=EVAL_SAMPLES, run_final_eval=True, logger=logger_cfg)
        runner = SyntheticSAERunner(cfg=runner_cfg)
        result = runner.run()

        eval_result = eval_sae_on_synthetic_data(sae=result.sae, feature_dict=synthetic_model.feature_dict, activations_generator=synthetic_model.activation_generator, num_samples=EVAL_SAMPLES, batch_size=BATCH_SIZE)
        absorption = compute_absorption(result.sae, synthetic_model, seed_off=seed+999)

        all_results[f"seed_{seed}"] = {
            "absorption_rate": absorption["absorption_rate"],
            "l0_sparsity": eval_result.sae_l0,
            "feature_recovery_mcc": eval_result.mcc,
        }
        print(f"  seed={seed}: absorption={absorption['absorption_rate']:.4f}, L0={eval_result.sae_l0:.1f}")

    elapsed = time.time() - start
    output = Path(results_dir) / f"{task_id}_results.json"
    with open(output, "w") as f:
        json.dump({"task_id": task_id, "results": all_results, "elapsed_seconds": elapsed, "timestamp": datetime.now().isoformat()}, f, indent=2)
    mark_done(task_id, results_dir, status="success", summary=f"TopK as ReLU: {len(SEEDS)} seeds")
    print(f"Results: {output}, time={elapsed:.1f}s")


def run_analysis(task_id, results_dir):
    """Aggregate all results into statistical analysis."""
    print(f"\n{'='*60}")
    print(f"Statistical Analysis (task: {task_id})")
    print(f"{'='*60}")

    # Collect all variant results
    full_dir = Path(results_dir)
    variants = {}

    for variant_file in full_dir.glob("full_*_results.json"):
        with open(variant_file) as f:
            data = json.load(f)
        variant_name = data.get("variant", variant_file.stem)
        if "aggregated" in data:
            variants[variant_name] = data["aggregated"]

    # Also include ablations
    for abl_file in full_dir.glob("ablation_*_results.json"):
        with open(abl_file) as f:
            data = json.load(f)
        variants[data.get("task_id", abl_file.stem)] = data.get("results", {})

    # Compute summary statistics
    summary = {}
    for name, agg in variants.items():
        if isinstance(agg, dict) and "absorption_rate" in agg:
            summary[name] = {
                "absorption_mean": agg["absorption_rate"].get("mean", 0),
                "absorption_std": agg["absorption_rate"].get("std", 0),
                "l0_mean": agg.get("l0_sparsity", {}).get("mean", 0),
            }

    analysis = {
        "task_id": task_id,
        "variant_count": len(variants),
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }

    output = full_dir / f"{task_id}_results.json"
    with open(output, "w") as f:
        json.dump(analysis, f, indent=2)
    mark_done(task_id, results_dir, status="success", summary=f"Analysis: {len(variants)} variants")
    print(f"Analysis complete. {len(variants)} variants analyzed.")
    print(f"Results: {output}")
    for name, stats in summary.items():
        print(f"  {name}: absorption={stats['absorption_mean']:.4f} ± {stats['absorption_std']:.4f}, L0={stats['l0_mean']:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["ort_no_penalty", "topk_as_relu", "analysis", "all"], default="all")
    parser.add_argument("--results-dir", type=str, default="exp/results/full")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("ort_no_penalty", "all"):
        run_ort_no_penalty("ablation_ort_sae_no_penalty", results_dir)
    if args.task in ("topk_as_relu", "all"):
        run_topk_as_relu("ablation_topk_as_relu", results_dir)
    if args.task in ("analysis", "all"):
        run_analysis("analysis_statistics", results_dir)


if __name__ == "__main__":
    main()
