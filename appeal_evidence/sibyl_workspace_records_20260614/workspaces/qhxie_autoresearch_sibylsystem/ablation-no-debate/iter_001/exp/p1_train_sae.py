#!/usr/bin/env python3
"""
Pilot Experiment 2: Train SAEs on synthetic activations
for testing absorption under random baseline scrutiny.

Trains SAELens-style SAEs with different L0 targets on the
synthetic feature hierarchies.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import os

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
GPU_ID = 0
PILOT_SEED = 42
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"

torch.manual_seed(PILOT_SEED)
np.random.seed(PILOT_SEED)

class SimpleSAE(nn.Module):
    """
    Simplified SAE implementation for synthetic data.
    Uses TopK activation (standard for SAEs).
    """
    def __init__(self, d_model, d_sae, l0_target=32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l0_target = l0_target

        # Encoder: d_model -> d_sae
        self.W_encoder = nn.Linear(d_model, d_sae, bias=True)
        nn.init.xavier_uniform_(self.W_encoder.weight)

        # Decoder: d_sae -> d_model
        self.W_decoder = nn.Linear(d_sae, d_model, bias=False)
        nn.init.xavier_uniform_(self.W_decoder.weight)

        # Normalize decoder columns
        with torch.no_grad():
            self.normalize_decoder()

        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def normalize_decoder(self):
        """Normalize decoder columns to unit norm (SAE standard)."""
        norm = torch.norm(self.W_decoder.weight, dim=0, keepdim=True)
        self.W_decoder.weight.div_(norm + 1e-8)

    def forward(self, x, return_features=False):
        """
        Forward pass with TopK sparsity.
        """
        # Encode
        pre_acts = self.W_encoder(x) + self.b_enc
        acts = torch.relu(pre_acts)

        # TopK sparsity - keep only top k features per sample
        batch_size = acts.shape[0]
        k = min(self.l0_target, acts.shape[1])

        if k < acts.shape[1]:
            # Get top k values and indices for each sample
            topk_values, topk_indices = torch.topk(acts, k=k, dim=1)
            threshold = topk_values[:, -1:]  # Smallest value in top-k
            sparse_acts = torch.where(acts >= threshold, acts, torch.zeros_like(acts))
        else:
            sparse_acts = acts

        # Decode
        recon = self.W_decoder(sparse_acts) + self.b_dec

        if return_features:
            return recon, sparse_acts, acts
        return recon

    def get_active_features(self, x):
        """Get which features are active for input x."""
        with torch.no_grad():
            pre_acts = self.W_encoder(x) + self.b_enc
            acts = torch.relu(pre_acts)

            # Apply same TopK sparsity as forward
            batch_size = acts.shape[0]
            k = min(self.l0_target, acts.shape[1])

            if k < acts.shape[1]:
                topk_values, topk_indices = torch.topk(acts, k=k, dim=1)
                threshold = topk_values[:, -1:]
                sparse_acts = torch.where(acts >= threshold, acts, torch.zeros_like(acts))
            else:
                sparse_acts = acts

            return sparse_acts > 0, acts


def train_sae(train_data, d_model, d_sae, l0_target, n_epochs=100,
              batch_size=64, lr=1e-3, device=DEVICE, seed=42):
    """
    Train SAE on synthetic data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize model
    model = SimpleSAE(d_model, d_sae, l0_target).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_tensor = torch.FloatTensor(train_data).to(device)

    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        # Shuffle
        perm = torch.randperm(len(train_tensor))
        for i in range(0, len(train_tensor), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = train_tensor[batch_idx]

            optimizer.zero_grad()
            recon = model(batch)

            # L2 reconstruction loss
            loss = torch.nn.functional.mse_loss(recon, batch)

            loss.backward()
            optimizer.step()

            # Normalize decoder after update
            with torch.no_grad():
                model.normalize_decoder()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

    # Final metrics
    model.eval()
    with torch.no_grad():
        recon = model(train_tensor)
        final_mse = torch.nn.functional.mse_loss(recon, train_tensor).item()

        # Compute L0
        _, sparse_acts, _ = model.forward(train_tensor, return_features=True)
        avg_l0 = (sparse_acts > 0).float().sum(dim=-1).mean().item()

        # Compute explained variance
        total_var = torch.var(train_tensor).item()
        residual_var = torch.var(recon - train_tensor).item()
        explained_variance = 1 - (residual_var / (total_var + 1e-8))

    return {
        "model": model,
        "losses": losses,
        "final_mse": final_mse,
        "avg_l0": avg_l0,
        "explained_variance": explained_variance
    }


def main():
    print("=" * 60)
    print("Pilot Experiment 2: SAE Training on Synthetic Activations")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load synthetic data
    print("\n[1/3] Loading synthetic activations...")
    data_path = WORKSPACE / "data" / "pilot_activations.json"
    with open(data_path) as f:
        data = json.load(f)

    activations = np.array(data["activations"])
    d_model = data["d_model"]
    n_samples = data["n_samples"]
    print(f"  Loaded {n_samples} samples, d_model={d_model}")

    # SAE configurations (pilot: just L0=32, 1 seed)
    configs = [
        {"d_sae": 1024, "l0_target": 32, "seed": 42, "name": "L0_32"},
    ]

    results = {}
    os.makedirs(WORKSPACE / "exp" / "results" / "pilots", exist_ok=True)

    print("\n[2/3] Training SAEs...")
    for cfg in configs:
        print(f"\n  Training SAE: d_sae={cfg['d_sae']}, L0={cfg['l0_target']}, seed={cfg['seed']}")

        result = train_sae(
            train_data=activations,
            d_model=d_model,
            d_sae=cfg["d_sae"],
            l0_target=cfg["l0_target"],
            n_epochs=100,
            batch_size=64,
            lr=1e-3,
            device=DEVICE,
            seed=cfg["seed"]
        )

        cfg_name = cfg["name"]
        results[cfg_name] = {
            "config": cfg,
            "final_mse": result["final_mse"],
            "avg_l0": result["avg_l0"],
            "explained_variance": result["explained_variance"],
            "losses": result["losses"]
        }

        # Save model
        model_path = WORKSPACE / "exp" / "results" / "pilots" / f"sae_{cfg_name}.pt"
        torch.save({
            "model_state": result["model"].state_dict(),
            "config": cfg,
            "metrics": {
                "final_mse": result["final_mse"],
                "avg_l0": result["avg_l0"],
                "explained_variance": result["explained_variance"]
            }
        }, model_path)
        print(f"    Saved: {model_path}")

    print("\n[3/3] Computing pass criteria...")

    # Pass criteria from task_plan.json:
    # L0 ~ 30-35, CE loss recovered > 80% (we use explained variance)
    all_pass = True
    for name, res in results.items():
        l0 = res["avg_l0"]
        ev = res["explained_variance"]

        l0_ok = 25 <= l0 <= 40  # ~30-35 +/- 5
        ev_ok = ev > 0.80  # > 80% explained variance

        print(f"\n  {name}:")
        print(f"    L0: {l0:.1f} (expected ~30-35) -> {'PASS' if l0_ok else 'FAIL'}")
        print(f"    Explained variance: {ev:.3f} (expected > 0.80) -> {'PASS' if ev_ok else 'FAIL'}")

        if not (l0_ok and ev_ok):
            all_pass = False

    # Save results
    results_path = WORKSPACE / "exp" / "results" / "pilots" / "sae_training_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "task": "p1_train_sae",
            "pilot_samples": n_samples,
            "d_model": d_model,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "model"}
                       for k, v in results.items()},
            "all_pass": all_pass,
            "pass_criteria": {
                "l0_range": [25, 40],
                "explained_variance_min": 0.80
            }
        }, f, indent=2)
    print(f"\n  Saved: {results_path}")

    print("\n" + "=" * 60)
    print(f"PILOT PASS: {all_pass}")
    print("=" * 60)

    if all_pass:
        print("\n[SUCCESS] SAE training validated. Proceed to Task 3 (baselines).")
    else:
        print("\n[WARNING] Some criteria not met. Check L0 targets or training epochs.")

    return {"status": "success" if all_pass else "partial", "results": results}


if __name__ == "__main__":
    result = main()
