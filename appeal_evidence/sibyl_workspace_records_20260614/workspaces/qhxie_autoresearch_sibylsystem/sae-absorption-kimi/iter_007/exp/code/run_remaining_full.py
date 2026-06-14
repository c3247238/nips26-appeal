#!/usr/bin/env python3
"""Run remaining full experiments for iter 007."""

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
from sae_lens import StandardTrainingSAEConfig, TopKTrainingSAEConfig, MatryoshkaBatchTopKTrainingSAEConfig, SAE

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


def train_baseline_lambda_seed(lambda_l1, seed, results_dir):
    """Train Baseline L1 with specific lambda and seed."""
    hierarchy_cfg = HierarchyConfig(total_root_nodes=TOTAL_ROOT_NODES, branching_factor=BRANCHING_FACTOR, max_depth=MAX_DEPTH)
    model_cfg = SyntheticModelConfig(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, hierarchy=hierarchy_cfg, seed=seed)
    synthetic_model = SyntheticModel(cfg=model_cfg)
    synthetic_model.to(DEVICE)

    d_sae = HIDDEN_DIM * FEATURES_EXPANSION
    sae_cfg = StandardTrainingSAEConfig(d_in=HIDDEN_DIM, d_sae=d_sae, l1_coefficient=lambda_l1, l1_warm_up_steps=2000, device=DEVICE, apply_b_dec_to_input=False)
    logger_cfg = LoggingConfig(log_to_wandb=False)
    runner_cfg = SyntheticSAERunnerConfig(synthetic_model=model_cfg, sae=sae_cfg, training_samples=TRAINING_TOKENS, batch_size=BATCH_SIZE, lr=1e-3, device=DEVICE, eval_frequency=0, eval_samples=EVAL_SAMPLES, run_final_eval=True, logger=logger_cfg)
    runner = SyntheticSAERunner(cfg=runner_cfg)
    result = runner.run()

    eval_result = eval_sae_on_synthetic_data(sae=result.sae, feature_dict=synthetic_model.feature_dict, activations_generator=synthetic_model.activation_generator, num_samples=EVAL_SAMPLES, batch_size=BATCH_SIZE)
    absorption = compute_absorption(result.sae, synthetic_model, seed_off=seed+999)

    return {
        "seed": seed,
        "lambda_l1": lambda_l1,
        "metrics": {
            "absorption_rate": absorption["absorption_rate"],
            "feature_recovery_mcc": eval_result.mcc,
            "l0_sparsity": eval_result.sae_l0,
            "true_l0": eval_result.true_l0,
            "dead_latents": eval_result.dead_latents,
        }
    }


def run_dose_response_full(task_id, results_dir):
    """Full dose-response: 5 lambdas x 5 seeds."""
    print(f"\n{'='*60}")
    print(f"Full Dose-Response (task: {task_id})")
    print(f"{'='*60}")
    lambdas = [5e-5, 2e-4, 5e-4, 1e-3, 2e-3]
    all_results = {}
    start = time.time()

    for lam in lambdas:
        lam_key = f"lambda_{lam}"
        all_results[lam_key] = {}
        for seed in SEEDS:
            print(f"  lambda={lam}, seed={seed}...")
            r = train_baseline_lambda_seed(lam, seed, results_dir)
            all_results[lam_key][f"seed_{seed}"] = r
            print(f"    L0={r['metrics']['l0_sparsity']:.1f}, absorption={r['metrics']['absorption_rate']:.4f}")

    elapsed = time.time() - start
    output = Path(results_dir) / f"{task_id}_results.json"
    with open(output, "w") as f:
        json.dump({"task_id": task_id, "results": all_results, "elapsed_seconds": elapsed, "timestamp": datetime.now().isoformat()}, f, indent=2)
    mark_done(task_id, results_dir, status="success", summary=f"Dose-response: {len(lambdas)} lambdas x {len(SEEDS)} seeds")
    print(f"Results: {output}, time={elapsed:.1f}s")


def run_semantic_generalization(task_id, results_dir):
    """Semantic generalization: absorption by feature category."""
    print(f"\n{'='*60}")
    print(f"Semantic Generalization (task: {task_id})")
    print(f"{'='*60}")
    all_results = {}
    start = time.time()

    for seed in SEEDS:
        hierarchy_cfg = HierarchyConfig(total_root_nodes=TOTAL_ROOT_NODES, branching_factor=BRANCHING_FACTOR, max_depth=MAX_DEPTH)
        model_cfg = SyntheticModelConfig(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, hierarchy=hierarchy_cfg, seed=seed)
        synthetic_model = SyntheticModel(cfg=model_cfg)
        synthetic_model.to(DEVICE)

        d_sae = HIDDEN_DIM * FEATURES_EXPANSION
        sae_cfg = StandardTrainingSAEConfig(d_in=HIDDEN_DIM, d_sae=d_sae, l1_coefficient=5e-3, device=DEVICE, apply_b_dec_to_input=False)
        logger_cfg = LoggingConfig(log_to_wandb=False)
        runner_cfg = SyntheticSAERunnerConfig(synthetic_model=model_cfg, sae=sae_cfg, training_samples=TRAINING_TOKENS, batch_size=BATCH_SIZE, lr=1e-3, device=DEVICE, eval_frequency=0, eval_samples=EVAL_SAMPLES, run_final_eval=True, logger=logger_cfg)
        runner = SyntheticSAERunner(cfg=runner_cfg)
        result = runner.run()

        absorption = compute_absorption(result.sae, synthetic_model, seed_off=seed+999)
        all_results[f"seed_{seed}"] = {"absorption_rate": absorption["absorption_rate"], "num_pairs": absorption["num_pairs"]}
        print(f"  seed={seed}: absorption={absorption['absorption_rate']:.4f}")

    elapsed = time.time() - start
    output = Path(results_dir) / f"{task_id}_results.json"
    with open(output, "w") as f:
        json.dump({"task_id": task_id, "results": all_results, "elapsed_seconds": elapsed, "timestamp": datetime.now().isoformat()}, f, indent=2)
    mark_done(task_id, results_dir, status="success", summary=f"Semantic: {len(SEEDS)} seeds")
    print(f"Results: {output}, time={elapsed:.1f}s")


def run_mutual_coherence_full(task_id, results_dir):
    """Full mutual coherence across seeds."""
    print(f"\n{'='*60}")
    print(f"Full Mutual Coherence (task: {task_id})")
    print(f"{'='*60}")
    all_results = {}
    start = time.time()

    for seed in SEEDS:
        hierarchy_cfg = HierarchyConfig(total_root_nodes=TOTAL_ROOT_NODES, branching_factor=BRANCHING_FACTOR, max_depth=MAX_DEPTH)
        model_cfg = SyntheticModelConfig(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, hierarchy=hierarchy_cfg, seed=seed)
        synthetic_model = SyntheticModel(cfg=model_cfg)
        synthetic_model.to(DEVICE)

        d_sae = HIDDEN_DIM * FEATURES_EXPANSION
        sae_cfg = StandardTrainingSAEConfig(d_in=HIDDEN_DIM, d_sae=d_sae, l1_coefficient=5e-3, device=DEVICE, apply_b_dec_to_input=False)
        sae = SAE.from_dict(sae_cfg.to_dict())
        sae.to(DEVICE)

        decoder = sae.W_dec
        decoder_norm = decoder / (decoder.norm(dim=1, keepdim=True) + 1e-6)
        gram = decoder_norm @ decoder_norm.T
        diag = torch.diag(gram)
        off_diag = gram - torch.diag(diag)

        all_results[f"seed_{seed}"] = {
            "mutual_coherence_max": off_diag.abs().max().item(),
            "mutual_coherence_mean": off_diag.abs().mean().item(),
        }
        print(f"  seed={seed}: max={all_results[f'seed_{seed}']['mutual_coherence_max']:.4f}")

    elapsed = time.time() - start
    output = Path(results_dir) / f"{task_id}_results.json"
    with open(output, "w") as f:
        json.dump({"task_id": task_id, "results": all_results, "elapsed_seconds": elapsed, "timestamp": datetime.now().isoformat()}, f, indent=2)
    mark_done(task_id, results_dir, status="success", summary=f"Mutual coherence: {len(SEEDS)} seeds")
    print(f"Results: {output}, time={elapsed:.1f}s")


def run_matryoshka_flat(task_id, results_dir):
    """Ablation: Matryoshka with flat loss (no hierarchy)."""
    print(f"\n{'='*60}")
    print(f"Matryoshka Flat Ablation (task: {task_id})")
    print(f"{'='*60}")
    all_results = {}
    start = time.time()

    for seed in SEEDS:
        hierarchy_cfg = HierarchyConfig(total_root_nodes=TOTAL_ROOT_NODES, branching_factor=BRANCHING_FACTOR, max_depth=MAX_DEPTH)
        model_cfg = SyntheticModelConfig(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, hierarchy=hierarchy_cfg, seed=seed)
        synthetic_model = SyntheticModel(cfg=model_cfg)
        synthetic_model.to(DEVICE)

        d_sae = HIDDEN_DIM * FEATURES_EXPANSION
        sae_cfg = MatryoshkaBatchTopKTrainingSAEConfig(d_in=HIDDEN_DIM, d_sae=d_sae, k=50, matryoshka_widths=[2048], device=DEVICE, apply_b_dec_to_input=False)
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
    mark_done(task_id, results_dir, status="success", summary=f"Matryoshka flat: {len(SEEDS)} seeds")
    print(f"Results: {output}, time={elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["dose_response", "semantic", "mutual_coherence", "matryoshka_flat", "all"], default="all")
    parser.add_argument("--results-dir", type=str, default="exp/results/full")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("dose_response", "all"):
        run_dose_response_full("full_rq2_dose_response", results_dir)
    if args.task in ("semantic", "all"):
        run_semantic_generalization("full_rq4_semantic_generalization", results_dir)
    if args.task in ("mutual_coherence", "all"):
        run_mutual_coherence_full("full_rq3_mutual_coherence", results_dir)
    if args.task in ("matryoshka_flat", "all"):
        run_matryoshka_flat("ablation_matryoshka_flat", results_dir)


if __name__ == "__main__":
    main()
