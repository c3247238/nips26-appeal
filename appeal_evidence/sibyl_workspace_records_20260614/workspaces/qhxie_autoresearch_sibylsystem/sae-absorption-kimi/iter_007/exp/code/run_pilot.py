#!/usr/bin/env python3
"""
Pilot experiment script for SAE absorption component-isolated study.

Trains 4 SAE variants on synthetic hierarchical data:
- Baseline: Standard ReLU + L1
- TopK: Explicit k-sparsity (k=50)
- MultiScale: Matryoshka hierarchical decomposition
- Random control: Random decoder, no training

Measures ground-truth absorption rate using known parent-child relationships.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import numpy as np

from sae_lens.synthetic.synthetic_model import SyntheticModel
from sae_lens.synthetic.synthetic_sae_runner import (
    SyntheticSAERunner,
    SyntheticSAERunnerConfig,
    SyntheticModelConfig,
    eval_sae_on_synthetic_data,
    LoggingConfig,
)
from sae_lens.synthetic.hierarchy import HierarchyConfig
from sae_lens import (
    StandardTrainingSAEConfig,
    TopKTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAEConfig,
    GatedTrainingSAEConfig,
    SAE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_FEATURES = 1024
HIDDEN_DIM = 256
FEATURES_EXPANSION = 8  # d_sae = hidden_dim * expansion = 2048
TOTAL_ROOT_NODES = 32
BRANCHING_FACTOR = 4
MAX_DEPTH = 3
TRAINING_TOKENS = 2_000_000
BATCH_SIZE = 1024
EVAL_SAMPLES = 100_000
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def write_pid(task_id: str, results_dir: Path):
    """Write PID file for system recovery detection."""
    pid_file = results_dir / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))


def report_progress(task_id: str, results_dir: Path, epoch: int, total_epochs: int,
                    step: int = 0, total_steps: int = 0, loss: float = None, metric: dict = None):
    """Write progress file for system monitor."""
    progress = results_dir / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id: str, results_dir: Path, status: str = "success", summary: str = ""):
    """Write DONE marker file."""
    pid_file = results_dir / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = results_dir / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = results_dir / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def compute_absorption_rate(sae, synthetic_model, num_test_samples: int = 10000):
    """
    Compute ground-truth absorption rate using known parent-child hierarchy.

    Definition: For each parent feature p in the hierarchy, when p fires
    (activates), absorption occurs if the SAE latents corresponding to p's
    children also activate significantly. This means the SAE has "absorbed"
    the parent concept into child concepts.

    Method:
    1. For each SAE latent, find its best-matching ground-truth feature
    2. Sample natural activations from the synthetic model
    3. For each parent feature that fires, check if child-matching latents
       also fire (above threshold)
    4. Absorption rate = fraction of parent firings where child latents
       also fire significantly
    """
    sae.eval()
    device = next(sae.parameters()).device

    # Get decoder and GT feature vectors
    sae_decoder = sae.W_dec  # (d_sae, d_in)
    gt_features = synthetic_model.feature_dict.feature_vectors  # (num_features, hidden_dim)

    # Best-matching GT feature for each SAE latent
    with torch.no_grad():
        cos_sim = torch.nn.functional.cosine_similarity(
            sae_decoder.unsqueeze(1), gt_features.unsqueeze(0), dim=2
        ).abs()  # (d_sae, num_features)
        latent_to_feature = cos_sim.argmax(dim=1).cpu().numpy()  # (d_sae,)

    # Build feature -> list of latents mapping
    feature_to_latents = {}
    for latent_idx, feat_idx in enumerate(latent_to_feature):
        feat_idx = int(feat_idx)
        if feat_idx not in feature_to_latents:
            feature_to_latents[feat_idx] = []
        feature_to_latents[feat_idx].append(latent_idx)

    # Extract parent-child pairs from hierarchy
    hierarchy = synthetic_model.hierarchy
    parent_child_pairs = []

    def extract_pairs(node):
        for child in node.children:
            parent_child_pairs.append((int(node.feature_index), int(child.feature_index)))
            extract_pairs(child)

    for root in hierarchy.roots:
        extract_pairs(root)

    if len(parent_child_pairs) == 0:
        return {"absorption_rate": 0.0, "num_pairs": 0, "absorbed_pairs": 0}

    # Sample natural activations
    torch.manual_seed(SEED + 999)
    np.random.seed(SEED + 999)

    parent_fire_count = 0
    absorption_count = 0
    activation_threshold = 0.05

    with torch.no_grad():
        batch_size = 1024
        num_batches = (num_test_samples + batch_size - 1) // batch_size

        for _ in range(num_batches):
            # Sample from the model (natural firing, includes hierarchy effects)
            # sample_with_features returns (hidden_activations, feature_activations)
            hidden_acts, feature_acts = synthetic_model.sample_with_features(batch_size)
            hidden_acts = hidden_acts.to(device)
            feature_acts = feature_acts.to(device)

            # Encode through SAE
            sae_latents = sae.encode(hidden_acts)  # (batch, d_sae)

            # For each parent-child pair
            for parent_idx, child_idx in parent_child_pairs:
                # Check if parent fired in this batch
                parent_fires = feature_acts[:, parent_idx] > 0

                if parent_fires.sum() == 0:
                    continue

                parent_fire_count += parent_fires.sum().item()

                # Get child latents
                child_latents = feature_to_latents.get(child_idx, [])
                if len(child_latents) == 0:
                    continue

                # Check if child latents activate when parent fires
                child_activations = sae_latents[parent_fires][:, child_latents]
                child_fires = (child_activations > activation_threshold).any(dim=1)
                absorption_count += child_fires.sum().item()

    absorption_rate = absorption_count / parent_fire_count if parent_fire_count > 0 else 0.0

    return {
        "absorption_rate": absorption_rate,
        "num_pairs": len(parent_child_pairs),
        "parent_fire_count": parent_fire_count,
        "absorption_count": absorption_count,
    }


def compute_hedging_score(sae, synthetic_model):
    """
    Compute hedging score: fraction of SAE latents whose best-matching
    ground-truth feature has children in the hierarchy.

    A high hedging score means many latents are assigned to parent features
    (which are more likely to be "absorbed" into children).
    """
    sae.eval()
    sae_decoder = sae.W_dec
    gt_features = synthetic_model.feature_dict.feature_vectors

    with torch.no_grad():
        cos_sim = torch.nn.functional.cosine_similarity(
            sae_decoder.unsqueeze(1), gt_features.unsqueeze(0), dim=2
        ).abs()
        latent_to_feature = cos_sim.argmax(dim=1).cpu().numpy()

    # Find features with children
    hierarchy = synthetic_model.hierarchy
    features_with_children = set()

    def mark_parents(node):
        if node.children:
            features_with_children.add(int(node.feature_index))
        for child in node.children:
            mark_parents(child)

    for root in hierarchy.roots:
        mark_parents(root)

    hedging_count = sum(1 for f in latent_to_feature if int(f) in features_with_children)
    total_latents = len(latent_to_feature)

    return {
        "hedging_score": hedging_count / total_latents if total_latents > 0 else 0.0,
        "hedging_latents": hedging_count,
        "total_latents": total_latents,
    }


def compute_reconstruction_mse(sae, synthetic_model, num_samples: int = 10000):
    """Compute reconstruction MSE on synthetic data."""
    sae.eval()
    device = next(sae.parameters()).device

    mse_sum = 0.0
    count = 0

    with torch.no_grad():
        batch_size = 1024
        num_batches = (num_samples + batch_size - 1) // batch_size

        for _ in range(num_batches):
            hidden_acts, _ = synthetic_model.sample_with_features(batch_size)
            hidden_acts = hidden_acts.to(device)

            sae_latents = sae.encode(hidden_acts)
            reconstructed = sae.decode(sae_latents)

            mse = ((hidden_acts - reconstructed) ** 2).mean().item()
            mse_sum += mse * hidden_acts.shape[0]
            count += hidden_acts.shape[0]

    return mse_sum / count if count > 0 else 0.0


def train_sae_variant(variant_name: str, task_id: str, results_dir: Path,
                      architecture: str, sae_config, training_samples: int = TRAINING_TOKENS):
    """Train an SAE variant and evaluate it."""
    start_time = time.time()
    write_pid(task_id, results_dir)
    report_progress(task_id, results_dir, epoch=0, total_epochs=1, step=0, total_steps=1)

    print(f"\n{'='*60}")
    print(f"Training: {variant_name} (task: {task_id})")
    print(f"{'='*60}")

    # Create synthetic model with hierarchy
    hierarchy_cfg = HierarchyConfig(
        total_root_nodes=TOTAL_ROOT_NODES,
        branching_factor=BRANCHING_FACTOR,
        max_depth=MAX_DEPTH,
    )
    model_cfg = SyntheticModelConfig(
        num_features=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        hierarchy=hierarchy_cfg,
        seed=SEED,
    )

    print(f"Creating synthetic model...")
    synthetic_model = SyntheticModel(cfg=model_cfg)
    synthetic_model.to(DEVICE)
    print(f"  Features: {NUM_FEATURES}, Hidden dim: {HIDDEN_DIM}")
    print(f"  Hierarchy: {TOTAL_ROOT_NODES} roots, {BRANCHING_FACTOR} branch, depth {MAX_DEPTH}")

    # Create runner config (disable wandb)
    logger_cfg = LoggingConfig(log_to_wandb=False)
    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model_cfg,
        sae=sae_config,
        training_samples=training_samples,
        batch_size=BATCH_SIZE,
        lr=1e-3,
        device=DEVICE,
        eval_frequency=0,
        eval_samples=EVAL_SAMPLES,
        run_final_eval=True,
        logger=logger_cfg,
    )

    print(f"Training SAE ({architecture})...")
    print(f"  d_in={HIDDEN_DIM}, d_sae={sae_config.d_sae}")
    print(f"  Training samples: {training_samples:,}")
    print(f"  Batch size: {BATCH_SIZE}")

    runner = SyntheticSAERunner(cfg=runner_cfg)
    result = runner.run()

    print(f"Training complete!")

    # Run evaluation
    print(f"Running evaluation...")
    eval_result = eval_sae_on_synthetic_data(
        sae=result.sae,
        feature_dict=synthetic_model.feature_dict,
        activations_generator=synthetic_model.activation_generator,
        num_samples=EVAL_SAMPLES,
        batch_size=BATCH_SIZE,
    )

    # Compute absorption rate
    print(f"Computing absorption rate...")
    absorption = compute_absorption_rate(result.sae, synthetic_model)

    # Compute hedging score
    print(f"Computing hedging score...")
    hedging = compute_hedging_score(result.sae, synthetic_model)

    # Compute reconstruction MSE
    print(f"Computing reconstruction MSE...")
    mse = compute_reconstruction_mse(result.sae, synthetic_model)

    # Build results
    elapsed = time.time() - start_time
    results = {
        "task_id": task_id,
        "variant": variant_name,
        "architecture": architecture,
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "d_sae": sae_config.d_sae,
            "training_samples": training_samples,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "device": DEVICE,
        },
        "metrics": {
            "absorption_rate": absorption["absorption_rate"],
            "feature_recovery_mcc": eval_result.mcc,
            "reconstruction_mse": mse,
            "explained_variance": eval_result.explained_variance,
            "l0_sparsity": eval_result.sae_l0,
            "true_l0": eval_result.true_l0,
            "dead_latents": eval_result.dead_latents,
            "shrinkage": eval_result.shrinkage,
            "uniqueness": eval_result.uniqueness,
            "hedging_score": hedging["hedging_score"],
            "classification_precision": eval_result.classification.precision,
            "classification_recall": eval_result.classification.recall,
            "classification_f1": eval_result.classification.f1_score,
            "classification_accuracy": eval_result.classification.accuracy,
        },
        "absorption_details": absorption,
        "hedging_details": hedging,
        "timing": {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    output_file = results_dir / f"{task_id}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    # Print summary
    print(f"\n--- Results Summary ---")
    for k, v in results["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"  Time: {elapsed:.1f}s")

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"{variant_name}: absorption={absorption['absorption_rate']:.3f}, MCC={eval_result.mcc:.3f}")

    return results


def run_random_control(task_id: str, results_dir: Path):
    """Run random decoder control (no training)."""
    start_time = time.time()
    write_pid(task_id, results_dir)

    print(f"\n{'='*60}")
    print(f"Random Control (task: {task_id})")
    print(f"{'='*60}")

    # Create synthetic model
    hierarchy_cfg = HierarchyConfig(
        total_root_nodes=TOTAL_ROOT_NODES,
        branching_factor=BRANCHING_FACTOR,
        max_depth=MAX_DEPTH,
    )
    model_cfg = SyntheticModelConfig(
        num_features=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        hierarchy=hierarchy_cfg,
        seed=SEED,
    )

    print(f"Creating synthetic model...")
    synthetic_model = SyntheticModel(cfg=model_cfg)
    synthetic_model.to(DEVICE)

    # Create untrained SAE with random decoder
    d_sae = HIDDEN_DIM * FEATURES_EXPANSION
    sae_cfg = StandardTrainingSAEConfig(
        d_in=HIDDEN_DIM,
        d_sae=d_sae,
        device=DEVICE,
        apply_b_dec_to_input=False,
    )
    sae = SAE.from_dict(sae_cfg.to_dict())
    sae.to(DEVICE)

    # Evaluate
    print(f"Evaluating random decoder...")
    eval_result = eval_sae_on_synthetic_data(
        sae=sae,
        feature_dict=synthetic_model.feature_dict,
        activations_generator=synthetic_model.activation_generator,
        num_samples=EVAL_SAMPLES,
        batch_size=BATCH_SIZE,
    )

    absorption = compute_absorption_rate(sae, synthetic_model)
    hedging = compute_hedging_score(sae, synthetic_model)
    mse = compute_reconstruction_mse(sae, synthetic_model)

    elapsed = time.time() - start_time
    results = {
        "task_id": task_id,
        "variant": "random_control",
        "architecture": "random",
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "d_sae": d_sae,
            "seed": SEED,
        },
        "metrics": {
            "absorption_rate": absorption["absorption_rate"],
            "feature_recovery_mcc": eval_result.mcc,
            "reconstruction_mse": mse,
            "explained_variance": eval_result.explained_variance,
            "l0_sparsity": eval_result.sae_l0,
            "true_l0": eval_result.true_l0,
            "dead_latents": eval_result.dead_latents,
            "shrinkage": eval_result.shrinkage,
            "uniqueness": eval_result.uniqueness,
            "hedging_score": hedging["hedging_score"],
            "classification_precision": eval_result.classification.precision,
            "classification_recall": eval_result.classification.recall,
            "classification_f1": eval_result.classification.f1_score,
            "classification_accuracy": eval_result.classification.accuracy,
        },
        "timing": {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
        },
        "timestamp": datetime.now().isoformat(),
    }

    output_file = results_dir / f"{task_id}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    print(f"\n--- Results Summary ---")
    for k, v in results["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"Random: absorption={absorption['absorption_rate']:.3f}, MCC={eval_result.mcc:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["baseline", "topk", "multiscale", "random", "all"], default="all")
    parser.add_argument("--results-dir", type=str, default="exp/results/pilots")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    d_sae = HIDDEN_DIM * FEATURES_EXPANSION  # 2048

    all_results = {}

    if args.task in ("baseline", "all"):
        sae_cfg = StandardTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=d_sae,
            l1_coefficient=5e-3,
            l1_warm_up_steps=2000,
            device=DEVICE,
            apply_b_dec_to_input=False,
        )
        all_results["baseline"] = train_sae_variant(
            "Baseline ReLU", "pilot_baseline", results_dir,
            "standard", sae_cfg
        )

    if args.task in ("topk", "all"):
        sae_cfg = TopKTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=d_sae,
            k=50,
            device=DEVICE,
            apply_b_dec_to_input=False,
        )
        all_results["topk"] = train_sae_variant(
            "TopK (k=50)", "pilot_topk", results_dir,
            "topk", sae_cfg
        )

    if args.task in ("multiscale", "all"):
        sae_cfg = MatryoshkaBatchTopKTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=d_sae,
            k=50,
            matryoshka_widths=[512, 1024, 2048],
            device=DEVICE,
            apply_b_dec_to_input=False,
        )
        all_results["multiscale"] = train_sae_variant(
            "MultiScale (Matryoshka)", "pilot_multiscale", results_dir,
            "matryoshka_batchtopk", sae_cfg
        )

    if args.task in ("random", "all"):
        all_results["random"] = run_random_control("pilot_random_control", results_dir)

    # Save combined results
    combined_file = results_dir / "pilot_all_results.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_file}")


if __name__ == "__main__":
    main()
