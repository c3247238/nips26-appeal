"""Per-layer diagnostic logger for WD experiments.

Tracks rho_t, alpha_t, ||w||, ||g||, effective lambda_t per layer per epoch.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class DiagnosticLogger:
    """Log per-layer diagnostic metrics across training.

    Usage:
        logger = DiagnosticLogger(save_dir='exp/results/...', method='UDWDC', seed=42)
        for epoch in range(epochs):
            # ... training loop ...
            diagnostics = optimizer.get_diagnostics()
            logger.log_epoch(epoch, diagnostics, train_loss, test_loss, train_acc, test_acc)
        logger.save()
    """

    def __init__(self, save_dir: str, method: str, seed: int = 42,
                 model_name: str = '', dataset: str = ''):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.method = method
        self.seed = seed
        self.model_name = model_name
        self.dataset = dataset

        # Per-layer trajectory storage
        self.trajectories: Dict[str, Dict[str, List[float]]] = {}
        # Global metrics per epoch
        self.epoch_metrics: List[Dict] = []

    def log_epoch(self, epoch: int, diagnostics: Dict[str, Dict[str, float]],
                  train_loss: float, test_loss: float,
                  train_acc: float, test_acc: float,
                  lr: float = 0.0):
        """Log diagnostics for one epoch.

        Args:
            epoch: Current epoch number.
            diagnostics: Per-layer diagnostics from optimizer.get_diagnostics().
            train_loss: Training loss.
            test_loss: Test/validation loss.
            train_acc: Training accuracy (0-100).
            test_acc: Test/validation accuracy (0-100).
            lr: Current learning rate.
        """
        # Store per-layer trajectories
        for layer_name, metrics in diagnostics.items():
            if layer_name not in self.trajectories:
                self.trajectories[layer_name] = {
                    'rho_t': [], 'alpha_t': [], 'w_norm': [],
                    'g_norm': [], 'effective_wd': []
                }
            for key in ['rho_t', 'alpha_t', 'w_norm', 'g_norm', 'effective_wd']:
                self.trajectories[layer_name][key].append(metrics.get(key, 0.0))

        # Compute aggregate metrics
        rho_values = [m['rho_t'] for m in diagnostics.values()]
        alpha_values = [m['alpha_t'] for m in diagnostics.values()]
        wd_values = [m['effective_wd'] for m in diagnostics.values()]

        self.epoch_metrics.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'lr': lr,
            'mean_rho_t': float(np.mean(rho_values)) if rho_values else 0.0,
            'std_rho_t': float(np.std(rho_values)) if rho_values else 0.0,
            'mean_alpha_t': float(np.mean(alpha_values)) if alpha_values else 0.0,
            'mean_effective_wd': float(np.mean(wd_values)) if wd_values else 0.0,
            'total_wd_budget_epoch': float(np.sum(wd_values)) if wd_values else 0.0,
        })

    def save(self):
        """Save all logged data to JSON files."""
        # Save epoch metrics
        epoch_file = self.save_dir / f'{self.method}_seed{self.seed}_epochs.json'
        with open(epoch_file, 'w') as f:
            json.dump({
                'method': self.method,
                'seed': self.seed,
                'model': self.model_name,
                'dataset': self.dataset,
                'epochs': self.epoch_metrics,
            }, f, indent=2)

        # Save per-layer trajectories
        traj_file = self.save_dir / f'{self.method}_seed{self.seed}_trajectories.json'
        with open(traj_file, 'w') as f:
            json.dump({
                'method': self.method,
                'seed': self.seed,
                'model': self.model_name,
                'dataset': self.dataset,
                'layers': self.trajectories,
            }, f, indent=2)

        return str(epoch_file), str(traj_file)

    def get_total_wd_budget(self) -> float:
        """Compute total WD budget across all epochs."""
        return sum(e.get('total_wd_budget_epoch', 0.0) for e in self.epoch_metrics)

    def get_final_accuracy(self) -> float:
        """Get final test accuracy."""
        if self.epoch_metrics:
            return self.epoch_metrics[-1].get('test_acc', 0.0)
        return 0.0

    def get_generalization_gap(self) -> float:
        """Get final generalization gap (train_acc - test_acc)."""
        if self.epoch_metrics:
            last = self.epoch_metrics[-1]
            return last.get('train_acc', 0.0) - last.get('test_acc', 0.0)
        return 0.0
