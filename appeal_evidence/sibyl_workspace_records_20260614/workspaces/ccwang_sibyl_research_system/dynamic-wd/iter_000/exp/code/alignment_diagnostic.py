"""
Tier 0 Alignment Diagnostic Tool.

Computes mini-batch and large-batch alignment at specified checkpoints.
Evaluates Pearson correlation between EMA-smoothed mini-batch and
large-batch alignment to validate H3 (mini-batch proxy reliability).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from data import get_dataloaders, get_large_batch_loader


def compute_alignment_single_batch(model, inputs, targets, loss_fn):
    """Compute alignment delta for a single batch."""
    model.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    dot_gw = 0.0
    norm_g_sq = 0.0
    norm_w_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.data
        w = p.data
        dot_gw += torch.dot(g.flatten(), w.flatten()).item()
        norm_g_sq += g.norm().item() ** 2
        norm_w_sq += w.norm().item() ** 2

    norm_g = math.sqrt(norm_g_sq)
    norm_w = math.sqrt(norm_w_sq)
    delta = abs(dot_gw) / (norm_g * norm_w + 1e-8)
    return delta


def compute_mini_batch_alignment(model, train_loader, loss_fn, device,
                                  num_batches=50):
    """Compute alignment from multiple mini-batches (B=128)."""
    model.eval()
    deltas = []
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        delta = compute_alignment_single_batch(model, inputs, targets, loss_fn)
        deltas.append(delta)
    model.train()
    return deltas


def compute_large_batch_alignment(model, large_loader, loss_fn, device):
    """Compute alignment using large batch (B=8192 or full)."""
    model.eval()
    # Accumulate gradients over the large batch
    model.zero_grad()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in large_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        break  # Just use first large batch

    dot_gw = 0.0
    norm_g_sq = 0.0
    norm_w_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.data
        w = p.data
        dot_gw += torch.dot(g.flatten(), w.flatten()).item()
        norm_g_sq += g.norm().item() ** 2
        norm_w_sq += w.norm().item() ** 2

    norm_g = math.sqrt(norm_g_sq)
    norm_w = math.sqrt(norm_w_sq)
    delta = abs(dot_gw) / (norm_g * norm_w + 1e-8)

    model.train()
    return delta


def ema_smooth(values, beta=0.99):
    """Apply EMA smoothing to a list of values."""
    smoothed = []
    ema = values[0] if values else 0.5
    for v in values:
        ema = beta * ema + (1 - beta) * v
        smoothed.append(ema)
    return smoothed


def pearson_r(x, y):
    """Compute Pearson correlation coefficient."""
    x = np.array(x)
    y = np.array(y)
    if len(x) < 2:
        return 0.0
    mx = x.mean()
    my = y.mean()
    sx = x.std()
    sy = y.std()
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return ((x - mx) * (y - my)).mean() / (sx * sy)
