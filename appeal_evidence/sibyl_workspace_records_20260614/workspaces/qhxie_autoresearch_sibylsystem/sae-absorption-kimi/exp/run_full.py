#!/usr/bin/env python3
"""
Full experiment script for SAE absorption component-isolated study.

Trains SAE variants on synthetic hierarchical data with 5 replicates (seeds 42-46).
Supports: baseline, topk, multiscale, orthogonality, gating, matryoshka, random.

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
# Constants (full scale)
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
SEEDS = [42, 123, 456, 789, 1011]  # 5 replicates
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def write_pid(task_id: str, results_dir: Path):
    """Write PID file for system recovery detection."""
    pid_file = results_dir / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))


def report_progress(task_id: str, results_dir: Path, replicate: int, total_replicates: int,
                    epoch: int = 0, total_epochs: int = 1, loss: float = None, metric: dict = None):
    """Write progress file for system monitor."""
    progress = results_dir / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "replicate": replicate,
        "total_replicates": total_replicates,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "loss": loss,
        "metric": metric or {},
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


def compute_absorption_rate(sae, synthetic_model, num_test_samples: int = 10000, seed_offset: int = 999):
    """Compute ground-truth absorption rate using known parent-child hierarchy."""
    sae.eval()
    device = next(sae.parameters()).device

    sae_decoder = sae.W_dec
    gt_features = synthetic_model.feature_dict.feature_vectors

    with torch.no_grad():
        # Chunked cosine similarity to avoid OOM
        chunk_size = 512
        d_sae = sae_decoder.shape[0]
        latent_to_feature = np.zeros(d_sae, dtype=np.int64)
        for i in range(0, d_sae, chunk_size):
            end = min(i + chunk_size, d_sae)
            chunk = sae_decoder[i:end]
            cos_sim_chunk = torch.nn.functional.cosine_similarity(
                chunk.unsqueeze(1), gt_features.unsqueeze(0), dim=2
            ).abs()
            latent_to_feature[i:end] = cos_sim_chunk.argmax(dim=1).cpu().numpy()

    feature_to_latents = {}
    for latent_idx, feat_idx in enumerate(latent_to_feature):
        feat_idx = int(feat_idx)
        if feat_idx not in feature_to_latents:
            feature_to_latents[feat_idx] = []
        feature_to_latents[feat_idx].append(latent_idx)

    hierarchy = synthetic_model.hierarchy
    parent_child_pairs = []

    def extract_pairs(node):
        for child in node.children:
            parent_child_pairs.append((int(node.feature_index), int(child.feature_index)))
            extract_pairs(child)

    for root in hierarchy.roots:
        extract_pairs(root)

    if len(parent_child_pairs) == 0:
        return {"absorption_rate": 0.0, "num_pairs": 0, "parent_fire_count": 0, "absorption_count": 0}

    torch.manual_seed(seed_offset)
    np.random.seed(seed_offset)

    parent_fire_count = 0
    absorption_count = 0
    activation_threshold = 0.05

    with torch.no_grad():
        batch_size = 1024
        num_batches = (num_test_samples + batch_size - 1) // batch_size

        for _ in range(num_batches):
            hidden_acts, feature_acts = synthetic_model.sample_with_features(batch_size)
            hidden_acts = hidden_acts.to(device)
            feature_acts = feature_acts.to(device)
            sae_latents = sae.encode(hidden_acts)

            for parent_idx, child_idx in parent_child_pairs:
                parent_fires = feature_acts[:, parent_idx] > 0
                if parent_fires.sum() == 0:
                    continue
                parent_fire_count += parent_fires.sum().item()
                child_latents = feature_to_latents.get(child_idx, [])
                if len(child_latents) == 0:
                    continue
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
    """Compute hedging score: fraction of latents assigned to parent features."""
    sae.eval()
    sae_decoder = sae.W_dec
    gt_features = synthetic_model.feature_dict.feature_vectors

    with torch.no_grad():
        # Chunked cosine similarity to avoid OOM
        chunk_size = 512
        d_sae = sae_decoder.shape[0]
        latent_to_feature = np.zeros(d_sae, dtype=np.int64)
        for i in range(0, d_sae, chunk_size):
            end = min(i + chunk_size, d_sae)
            chunk = sae_decoder[i:end]
            cos_sim_chunk = torch.nn.functional.cosine_similarity(
                chunk.unsqueeze(1), gt_features.unsqueeze(0), dim=2
            ).abs()
            latent_to_feature[i:end] = cos_sim_chunk.argmax(dim=1).cpu().numpy()

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


def train_single_replicate(variant_name: str, task_id: str, results_dir: Path,
                           architecture: str, sae_config, seed: int,
                           training_samples: int = TRAINING_TOKENS):
    """Train a single replicate of an SAE variant."""
    rep_id = f"{task_id}_seed{seed}"
    print(f"\n  --- Replicate {seed} ---")

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
        seed=seed,
    )

    synthetic_model = SyntheticModel(cfg=model_cfg)
    synthetic_model.to(DEVICE)

    # Create runner config
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

    runner = SyntheticSAERunner(cfg=runner_cfg)
    result = runner.run()

    # Evaluate
    eval_result = eval_sae_on_synthetic_data(
        sae=result.sae,
        feature_dict=synthetic_model.feature_dict,
        activations_generator=synthetic_model.activation_generator,
        num_samples=EVAL_SAMPLES,
        batch_size=BATCH_SIZE,
    )

    absorption = compute_absorption_rate(result.sae, synthetic_model, seed_offset=seed + 999)
    hedging = compute_hedging_score(result.sae, synthetic_model)
    mse = compute_reconstruction_mse(result.sae, synthetic_model)

    replicate_results = {
        "seed": seed,
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
    }

    # Save individual replicate
    rep_file = results_dir / f"{rep_id}_results.json"
    with open(rep_file, "w") as f:
        json.dump(replicate_results, f, indent=2)

    print(f"    absorption={absorption['absorption_rate']:.4f}, MCC={eval_result.mcc:.4f}, MSE={mse:.6f}, L0={eval_result.sae_l0:.1f}")

    return replicate_results


def train_sae_variant_full(variant_name: str, task_id: str, results_dir: Path,
                           architecture: str, sae_config, seeds: list = None,
                           training_samples: int = TRAINING_TOKENS):
    """Train an SAE variant with multiple replicates and aggregate results."""
    start_time = time.time()
    write_pid(task_id, results_dir)

    seeds = seeds or SEEDS
    print(f"\n{'='*60}")
    print(f"Full Experiment: {variant_name} (task: {task_id})")
    print(f"{'='*60}")
    print(f"  Architecture: {architecture}")
    print(f"  Replicates: {len(seeds)} (seeds: {seeds})")
    print(f"  d_in={HIDDEN_DIM}, d_sae={sae_config.d_sae}")
    print(f"  Training samples: {training_samples:,}")

    replicates = []
    for i, seed in enumerate(seeds):
        report_progress(task_id, results_dir, replicate=i+1, total_replicates=len(seeds))
        rep_results = train_single_replicate(
            variant_name, task_id, results_dir, architecture, sae_config, seed, training_samples
        )
        replicates.append(rep_results)

    # Aggregate statistics
    metrics_keys = replicates[0]["metrics"].keys()
    aggregated = {}
    for key in metrics_keys:
        values = [r["metrics"][key] for r in replicates]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

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
            "seeds": seeds,
            "device": DEVICE,
        },
        "replicates": replicates,
        "aggregated": aggregated,
        "timing": {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save combined results
    output_file = results_dir / f"{task_id}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Combined results saved to {output_file}")

    # Print summary
    print(f"\n  --- Aggregated Results ---")
    for key in ["absorption_rate", "feature_recovery_mcc", "reconstruction_mse", "l0_sparsity", "hedging_score"]:
        if key in aggregated:
            m = aggregated[key]
            print(f"    {key}: {m['mean']:.4f} ± {m['std']:.4f} (range: {m['min']:.4f} - {m['max']:.4f})")
    print(f"  Total time: {elapsed:.1f}s")

    summary = f"{variant_name}: absorption={aggregated['absorption_rate']['mean']:.3f}±{aggregated['absorption_rate']['std']:.3f}"
    mark_task_done(task_id, results_dir, status="success", summary=summary)

    return results


def run_random_control_full(task_id: str, results_dir: Path, seeds: list = None):
    """Run random decoder control with multiple seeds."""
    start_time = time.time()
    write_pid(task_id, results_dir)

    seeds = seeds or SEEDS
    print(f"\n{'='*60}")
    print(f"Random Control Full (task: {task_id})")
    print(f"{'='*60}")

    d_sae = HIDDEN_DIM * FEATURES_EXPANSION
    replicates = []

    for i, seed in enumerate(seeds):
        rep_id = f"{task_id}_seed{seed}"
        print(f"\n  --- Replicate {seed} ---")

        hierarchy_cfg = HierarchyConfig(
            total_root_nodes=TOTAL_ROOT_NODES,
            branching_factor=BRANCHING_FACTOR,
            max_depth=MAX_DEPTH,
        )
        model_cfg = SyntheticModelConfig(
            num_features=NUM_FEATURES,
            hidden_dim=HIDDEN_DIM,
            hierarchy=hierarchy_cfg,
            seed=seed,
        )

        synthetic_model = SyntheticModel(cfg=model_cfg)
        synthetic_model.to(DEVICE)

        sae_cfg = StandardTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=d_sae,
            device=DEVICE,
            apply_b_dec_to_input=False,
        )
        sae = SAE.from_dict(sae_cfg.to_dict())
        sae.to(DEVICE)

        eval_result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=synthetic_model.feature_dict,
            activations_generator=synthetic_model.activation_generator,
            num_samples=EVAL_SAMPLES,
            batch_size=BATCH_SIZE,
        )

        absorption = compute_absorption_rate(sae, synthetic_model, seed_offset=seed + 999)
        hedging = compute_hedging_score(sae, synthetic_model)
        mse = compute_reconstruction_mse(sae, synthetic_model)

        rep_results = {
            "seed": seed,
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
        }

        rep_file = results_dir / f"{rep_id}_results.json"
        with open(rep_file, "w") as f:
            json.dump(rep_results, f, indent=2)

        print(f"    absorption={absorption['absorption_rate']:.4f}, MCC={eval_result.mcc:.4f}")
        replicates.append(rep_results)

    # Aggregate
    metrics_keys = replicates[0]["metrics"].keys()
    aggregated = {}
    for key in metrics_keys:
        values = [r["metrics"][key] for r in replicates]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    elapsed = time.time() - start_time

    results = {
        "task_id": task_id,
        "variant": "random_control",
        "architecture": "random",
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "d_sae": d_sae,
            "seeds": seeds,
        },
        "replicates": replicates,
        "aggregated": aggregated,
        "timing": {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
        },
        "timestamp": datetime.now().isoformat(),
    }

    output_file = results_dir / f"{task_id}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Combined results saved to {output_file}")
    summary = f"Random: absorption={aggregated['absorption_rate']['mean']:.3f}±{aggregated['absorption_rate']['std']:.3f}"
    mark_task_done(task_id, results_dir, status="success", summary=summary)

    return results


# ---------------------------------------------------------------------------
# Orthogonality SAE (custom implementation)
# ---------------------------------------------------------------------------

class OrthogonalitySAE(torch.nn.Module):
    """Standard SAE with decoder orthogonality penalty."""

    def __init__(self, d_in: int, d_sae: int, ortho_lambda: float = 1e-3, device: str = "cuda"):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.ortho_lambda = ortho_lambda
        self.device = device

        # Encoder
        self.W_enc = torch.nn.Parameter(torch.randn(d_in, d_sae, device=device) / (d_in ** 0.5))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae, device=device))

        # Decoder
        self.W_dec = torch.nn.Parameter(torch.randn(d_sae, d_in, device=device) / (d_sae ** 0.5))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in, device=device))

        # Normalize decoder rows
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / (self.W_dec.data.norm(dim=1, keepdim=True) + 1e-6)

    def encode(self, x):
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc
        return torch.relu(pre_acts)

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        return self.decode(acts), acts

    def orthogonality_penalty(self):
        """Compute chunk-wise orthogonality penalty on decoder."""
        # Divide d_sae into chunks and penalize inner products within each chunk
        chunk_size = min(256, self.d_sae)
        num_chunks = (self.d_sae + chunk_size - 1) // chunk_size
        penalty = 0.0

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, self.d_sae)
            chunk = self.W_dec[start:end]  # (chunk_size, d_in)
            # Gram matrix of normalized rows
            gram = chunk @ chunk.T  # (chunk_size, chunk_size)
            # Penalize off-diagonal elements
            mask = 1.0 - torch.eye(gram.shape[0], device=gram.device)
            penalty += (gram * mask).abs().sum()

        return penalty / num_chunks


def train_orthogonality_sae(task_id: str, results_dir: Path, seeds: list = None,
                            ortho_lambda: float = 1e-3, training_samples: int = TRAINING_TOKENS):
    """Train Orthogonality SAE with custom training loop."""
    start_time = time.time()
    write_pid(task_id, results_dir)

    seeds = seeds or SEEDS
    print(f"\n{'='*60}")
    print(f"Full Experiment: Orthogonality SAE (task: {task_id})")
    print(f"{'='*60}")
    print(f"  Replicates: {len(seeds)} (seeds: {seeds})")
    print(f"  Orthogonality lambda: {ortho_lambda}")

    d_sae = HIDDEN_DIM * FEATURES_EXPANSION
    replicates = []

    for seed in seeds:
        rep_id = f"{task_id}_seed{seed}"
        print(f"\n  --- Replicate {seed} ---")

        hierarchy_cfg = HierarchyConfig(
            total_root_nodes=TOTAL_ROOT_NODES,
            branching_factor=BRANCHING_FACTOR,
            max_depth=MAX_DEPTH,
        )
        model_cfg = SyntheticModelConfig(
            num_features=NUM_FEATURES,
            hidden_dim=HIDDEN_DIM,
            hierarchy=hierarchy_cfg,
            seed=seed,
        )

        synthetic_model = SyntheticModel(cfg=model_cfg)
        synthetic_model.to(DEVICE)

        sae = OrthogonalitySAE(d_in=HIDDEN_DIM, d_sae=d_sae, ortho_lambda=ortho_lambda, device=DEVICE)

        # Training loop
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
        batch_size = BATCH_SIZE
        num_batches = training_samples // batch_size

        sae.train()
        for batch_idx in range(num_batches):
            hidden_acts, _ = synthetic_model.sample_with_features(batch_size)
            hidden_acts = hidden_acts.to(DEVICE)

            reconstructed, acts = sae(hidden_acts)
            recon_loss = ((hidden_acts - reconstructed) ** 2).mean()
            sparsity_loss = acts.mean()
            ortho_loss = sae.orthogonality_penalty()

            loss = recon_loss + 5e-3 * sparsity_loss + ortho_lambda * ortho_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Renormalize decoder
            with torch.no_grad():
                sae.W_dec.data = sae.W_dec.data / (sae.W_dec.data.norm(dim=1, keepdim=True) + 1e-6)

            if (batch_idx + 1) % 500 == 0:
                print(f"    Batch {batch_idx+1}/{num_batches}: loss={loss.item():.4f} "
                      f"(recon={recon_loss.item():.4f}, sparse={sparsity_loss.item():.4f}, ortho={ortho_loss.item():.4f})")

        # Evaluate using SAELens eval
        eval_result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=synthetic_model.feature_dict,
            activations_generator=synthetic_model.activation_generator,
            num_samples=EVAL_SAMPLES,
            batch_size=BATCH_SIZE,
        )

        absorption = compute_absorption_rate(sae, synthetic_model, seed_offset=seed + 999)
        hedging = compute_hedging_score(sae, synthetic_model)
        mse = compute_reconstruction_mse(sae, synthetic_model)

        rep_results = {
            "seed": seed,
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
        }

        rep_file = results_dir / f"{rep_id}_results.json"
        with open(rep_file, "w") as f:
            json.dump(rep_results, f, indent=2)

        print(f"    absorption={absorption['absorption_rate']:.4f}, MCC={eval_result.mcc:.4f}, MSE={mse:.6f}")
        replicates.append(rep_results)

    # Aggregate
    metrics_keys = replicates[0]["metrics"].keys()
    aggregated = {}
    for key in metrics_keys:
        values = [r["metrics"][key] for r in replicates]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    elapsed = time.time() - start_time

    results = {
        "task_id": task_id,
        "variant": "Orthogonality SAE",
        "architecture": "orthogonality",
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "d_sae": d_sae,
            "ortho_lambda": ortho_lambda,
            "training_samples": training_samples,
            "seeds": seeds,
        },
        "replicates": replicates,
        "aggregated": aggregated,
        "timing": {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
        },
        "timestamp": datetime.now().isoformat(),
    }

    output_file = results_dir / f"{task_id}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Combined results saved to {output_file}")
    summary = f"Orthogonality: absorption={aggregated['absorption_rate']['mean']:.3f}±{aggregated['absorption_rate']['std']:.3f}"
    mark_task_done(task_id, results_dir, status="success", summary=summary)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["baseline", "topk", "orthogonality",
                                           "gating", "matryoshka", "random", "all"],
                        default="all")
    parser.add_argument("--results-dir", type=str, default="exp/results/full")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1011",
                        help="Comma-separated list of seeds")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"Seeds: {seeds}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    d_sae = HIDDEN_DIM * FEATURES_EXPANSION

    if args.task in ("baseline", "all"):
        sae_cfg = StandardTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=d_sae,
            l1_coefficient=5e-3,
            l1_warm_up_steps=2000,
            device=DEVICE,
            apply_b_dec_to_input=False,
        )
        train_sae_variant_full(
            "Baseline ReLU", "full_baseline", results_dir,
            "standard", sae_cfg, seeds=seeds
        )

    if args.task in ("topk", "all"):
        sae_cfg = TopKTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=d_sae,
            k=50,
            device=DEVICE,
            apply_b_dec_to_input=False,
        )
        train_sae_variant_full(
            "TopK (k=50)", "full_topk", results_dir,
            "topk", sae_cfg, seeds=seeds
        )

    if args.task in ("orthogonality", "all"):
        train_orthogonality_sae(
            "full_orthogonality", results_dir, seeds=seeds, ortho_lambda=1e-3
        )

    if args.task in ("gating", "all"):
        sae_cfg = GatedTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=d_sae,
            l1_coefficient=5e-3,
            l1_warm_up_steps=2000,
            device=DEVICE,
            apply_b_dec_to_input=False,
        )
        train_sae_variant_full(
            "Gated SAE", "full_gating", results_dir,
            "gated", sae_cfg, seeds=seeds
        )

    if args.task in ("matryoshka", "all"):
        # Full Matryoshka = TopK + MultiScale + hierarchical loss
        # Use the same MatryoshkaBatchTopK config but with additional hierarchical loss
        sae_cfg = MatryoshkaBatchTopKTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=d_sae,
            k=50,
            matryoshka_widths=[512, 1024, 2048],
            device=DEVICE,
            apply_b_dec_to_input=False,
        )
        train_sae_variant_full(
            "Full Matryoshka", "full_matryoshka", results_dir,
            "matryoshka_batchtopk", sae_cfg, seeds=seeds
        )

    if args.task in ("random", "all"):
        run_random_control_full("full_random_control", results_dir, seeds=seeds)


if __name__ == "__main__":
    main()
