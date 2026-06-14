#!/usr/bin/env python3
"""Extended pilot experiments for iter 007."""

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
SEED = 42
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
    return {"absorption_rate": rate, "num_pairs": len(pairs), "parent_fire_count": parent_fires, "absorption_count": absorption_count}


def compute_mse(sae, synthetic_model, num_samples=10000):
    sae.eval()
    device = next(sae.parameters()).device
    mse_sum = 0.0
    count = 0
    with torch.no_grad():
        bs = 1024
        nb = (num_samples + bs - 1) // bs
        for _ in range(nb):
            h, _ = synthetic_model.sample_with_features(bs)
            h = h.to(device)
            rec = sae.decode(sae.encode(h))
            mse = ((h - rec) ** 2).mean().item()
            mse_sum += mse * h.shape[0]
            count += h.shape[0]
    return mse_sum / count if count > 0 else 0.0


def train_baseline_lambda(lambda_l1, task_id, results_dir):
    """Train Baseline L1 with specific lambda."""
    start = time.time()
    print(f"\n{'='*60}")
    print(f"L0-Match Baseline: lambda={lambda_l1} (task: {task_id})")
    print(f"{'='*60}")

    hierarchy_cfg = HierarchyConfig(total_root_nodes=TOTAL_ROOT_NODES, branching_factor=BRANCHING_FACTOR, max_depth=MAX_DEPTH)
    model_cfg = SyntheticModelConfig(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, hierarchy=hierarchy_cfg, seed=SEED)
    synthetic_model = SyntheticModel(cfg=model_cfg)
    synthetic_model.to(DEVICE)

    d_sae = HIDDEN_DIM * FEATURES_EXPANSION
    sae_cfg = StandardTrainingSAEConfig(d_in=HIDDEN_DIM, d_sae=d_sae, l1_coefficient=lambda_l1, l1_warm_up_steps=2000, device=DEVICE, apply_b_dec_to_input=False)
    logger_cfg = LoggingConfig(log_to_wandb=False)
    runner_cfg = SyntheticSAERunnerConfig(synthetic_model=model_cfg, sae=sae_cfg, training_samples=TRAINING_TOKENS, batch_size=BATCH_SIZE, lr=1e-3, device=DEVICE, eval_frequency=0, eval_samples=EVAL_SAMPLES, run_final_eval=True, logger=logger_cfg)

    runner = SyntheticSAERunner(cfg=runner_cfg)
    result = runner.run()

    eval_result = eval_sae_on_synthetic_data(sae=result.sae, feature_dict=synthetic_model.feature_dict, activations_generator=synthetic_model.activation_generator, num_samples=EVAL_SAMPLES, batch_size=BATCH_SIZE)
    absorption = compute_absorption(result.sae, synthetic_model)
    mse = compute_mse(result.sae, synthetic_model)

    elapsed = time.time() - start
    results = {
        "task_id": task_id, "variant": f"Baseline_L1_lambda_{lambda_l1}", "lambda_l1": lambda_l1,
        "metrics": {
            "absorption_rate": absorption["absorption_rate"], "feature_recovery_mcc": eval_result.mcc,
            "reconstruction_mse": mse, "explained_variance": eval_result.explained_variance,
            "l0_sparsity": eval_result.sae_l0, "true_l0": eval_result.true_l0,
            "dead_latents": eval_result.dead_latents,
        },
        "timing": {"elapsed_seconds": elapsed}, "timestamp": datetime.now().isoformat(),
    }
    output = Path(results_dir) / f"{task_id}_lambda_{lambda_l1}.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  L0={eval_result.sae_l0:.1f}, absorption={absorption['absorption_rate']:.4f}, MSE={mse:.6f}, time={elapsed:.1f}s")
    return results


def run_l0_match(task_id, results_dir):
    """Sweep lambda to match L0 targets."""
    print(f"\n{'='*60}")
    print(f"L0-Matched Baseline Sweep (task: {task_id})")
    print(f"{'='*60}")
    lambdas = [5e-4, 1e-3, 5e-3, 1e-2, 2e-2]
    all_results = {}
    for lam in lambdas:
        r = train_baseline_lambda(lam, task_id, results_dir)
        all_results[f"lambda_{lam}"] = r

    combined = Path(results_dir) / f"{task_id}_results.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)
    mark_done(task_id, results_dir, status="success", summary=f"L0 sweep: {len(lambdas)} lambdas tested")
    print(f"Combined results: {combined}")


def run_dose_response(task_id, results_dir):
    """Fix architecture, vary sparsity via lambda."""
    print(f"\n{'='*60}")
    print(f"Dose-Response Sparsity Sweep (task: {task_id})")
    print(f"{'='*60}")
    lambdas = [5e-5, 2e-4, 5e-4, 1e-3, 2e-3]
    all_results = {}
    for lam in lambdas:
        r = train_baseline_lambda(lam, task_id, results_dir)
        all_results[f"lambda_{lam}"] = r

    combined = Path(results_dir) / f"{task_id}_results.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)
    mark_done(task_id, results_dir, status="success", summary=f"Dose-response: {len(lambdas)} levels")
    print(f"Combined results: {combined}")


def run_semantic_absorption(task_id, results_dir):
    """Compute absorption by feature category."""
    print(f"\n{'='*60}")
    print(f"Semantic Feature Absorption (task: {task_id})")
    print(f"{'='*60}")

    hierarchy_cfg = HierarchyConfig(total_root_nodes=TOTAL_ROOT_NODES, branching_factor=BRANCHING_FACTOR, max_depth=MAX_DEPTH)
    model_cfg = SyntheticModelConfig(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, hierarchy=hierarchy_cfg, seed=SEED)
    synthetic_model = SyntheticModel(cfg=model_cfg)
    synthetic_model.to(DEVICE)

    d_sae = HIDDEN_DIM * FEATURES_EXPANSION
    sae_cfg = StandardTrainingSAEConfig(d_in=HIDDEN_DIM, d_sae=d_sae, l1_coefficient=5e-3, l1_warm_up_steps=2000, device=DEVICE, apply_b_dec_to_input=False)
    logger_cfg = LoggingConfig(log_to_wandb=False)
    runner_cfg = SyntheticSAERunnerConfig(synthetic_model=model_cfg, sae=sae_cfg, training_samples=TRAINING_TOKENS, batch_size=BATCH_SIZE, lr=1e-3, device=DEVICE, eval_frequency=0, eval_samples=EVAL_SAMPLES, run_final_eval=True, logger=logger_cfg)
    runner = SyntheticSAERunner(cfg=runner_cfg)
    result = runner.run()

    absorption = compute_absorption(result.sae, synthetic_model)

    results = {
        "task_id": task_id, "variant": "semantic_baseline",
        "overall_absorption": absorption["absorption_rate"],
        "num_pairs": absorption["num_pairs"],
        "timestamp": datetime.now().isoformat(),
    }
    output = Path(results_dir) / f"{task_id}_results.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    mark_done(task_id, results_dir, status="success", summary=f"Semantic absorption: {absorption['absorption_rate']:.3f}")
    print(f"Results: {output}, absorption={absorption['absorption_rate']:.4f}")


def run_gpt2_validation(task_id, results_dir):
    """Load GPT-2 SAE and report basic stats."""
    print(f"\n{'='*60}")
    print(f"GPT-2 Validation (task: {task_id})")
    print(f"{'='*60}")
    try:
        from sae_lens import SAE
        sae = SAE.from_pretrained("gpt2-small-res-jb", "blocks.8.hook_resid_pre", device=DEVICE)
        results = {
            "task_id": task_id, "variant": "gpt2_sae",
            "loaded": True, "d_in": sae.cfg.d_in, "d_sae": sae.cfg.d_sae,
            "timestamp": datetime.now().isoformat(),
        }
        print(f"  Loaded GPT-2 SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    except Exception as e:
        results = {"task_id": task_id, "variant": "gpt2_sae", "loaded": False, "error": str(e), "timestamp": datetime.now().isoformat()}
        print(f"  Failed to load GPT-2 SAE: {e}")

    output = Path(results_dir) / f"{task_id}_results.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    mark_done(task_id, results_dir, status="success", summary="GPT-2 SAE loaded" if results.get("loaded") else "GPT-2 SAE load failed")
    print(f"Results: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["l0_match", "dose_response", "semantic", "gpt2", "all"], default="all")
    parser.add_argument("--results-dir", type=str, default="exp/results/pilots")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("l0_match", "all"):
        run_l0_match("pilot_rq1_l0_match", results_dir)
    if args.task in ("dose_response", "all"):
        run_dose_response("pilot_rq2_dose_response", results_dir)
    if args.task in ("semantic", "all"):
        run_semantic_absorption("pilot_rq4_semantic_absorption", results_dir)
    if args.task in ("gpt2", "all"):
        run_gpt2_validation("pilot_gpt2_validation", results_dir)


if __name__ == "__main__":
    main()
