"""
Diagnostics: Per-layer logging of r_t, delta_hat_t, lambda_t, weight norms.

Provides a DiagnosticsLogger that records per-layer metrics at configurable
intervals, and exports them to JSON for analysis and visualization.
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class DiagnosticsLogger:
    """Logs per-layer training diagnostics at configurable intervals.

    Tracks:
    - r_t: gradient-to-weight ratio ||g||/||w|| per layer
    - delta_hat_t: cosine similarity <g, w>/(||g|| ||w||) per layer
    - lambda_t: effective weight decay per layer
    - w_norm: weight norm per layer
    - g_norm: gradient norm per layer
    """

    def __init__(
        self,
        log_interval: int = 10,
        output_dir: Optional[str] = None,
        task_id: str = "default",
    ):
        self.log_interval = log_interval
        self.output_dir = Path(output_dir) if output_dir else None
        self.task_id = task_id

        # Storage: list of records per layer
        self._records: Dict[str, List[dict]] = defaultdict(list)
        # Epoch-level metrics
        self._epoch_records: List[dict] = []

    def log_step(
        self,
        model: nn.Module,
        step: int,
        epoch: int,
        effective_wds: Dict[str, float],
        wd_method=None,
    ):
        """Log per-layer diagnostics at the current step.

        Args:
            model: The training model
            step: Global step number
            epoch: Current epoch
            effective_wds: Dict of param_name -> effective WD value
            wd_method: WD method instance (for extracting internal state)
        """
        if step % self.log_interval != 0:
            return

        for name, param in model.named_parameters():
            if param.grad is None or param.dim() <= 1:
                continue

            with torch.no_grad():
                g = param.grad
                w = param.data
                g_norm = g.norm().item()
                w_norm = w.norm().item()
                eps = 1e-8

                # Ratio r_t
                r_t = g_norm / (w_norm + eps)

                # Cosine similarity delta_hat_t
                g_flat = g.reshape(-1)
                w_flat = w.reshape(-1)
                if g_norm < 1e-12 or w_norm < 1e-12:
                    delta_hat = 0.0
                else:
                    delta_hat = (torch.dot(g_flat, w_flat) / (g_norm * w_norm + eps)).item()

            # Effective WD
            eff_wd = effective_wds.get(name, 0.0)

            record = {
                "step": step,
                "epoch": epoch,
                "r_t": r_t,
                "delta_hat": delta_hat,
                "lambda_t": eff_wd,
                "w_norm": w_norm,
                "g_norm": g_norm,
            }

            # Add WD method internal state if available
            if wd_method is not None:
                method_diag = wd_method.get_diagnostics(name)
                if "r_star" in method_diag:
                    record["r_star"] = method_diag["r_star"]
                if "dev" in method_diag:
                    record["dev"] = method_diag["dev"]
                if "phi" in method_diag:
                    record["phi"] = method_diag["phi"]

            self._records[name].append(record)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float,
        test_acc: float,
        train_acc: Optional[float] = None,
        lr: Optional[float] = None,
        extra: Optional[dict] = None,
    ):
        """Log epoch-level metrics."""
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        if train_acc is not None:
            record["train_acc"] = train_acc
        if lr is not None:
            record["lr"] = lr
        if extra:
            record.update(extra)

        self._epoch_records.append(record)

    def get_layer_records(self, layer_name: str) -> List[dict]:
        """Get all diagnostic records for a specific layer."""
        return self._records.get(layer_name, [])

    def get_epoch_records(self) -> List[dict]:
        """Get all epoch-level records."""
        return self._epoch_records

    def get_summary_stats(self) -> dict:
        """Compute summary statistics across layers."""
        summary = {}
        for name, records in self._records.items():
            if not records:
                continue
            r_values = [r["r_t"] for r in records]
            delta_values = [r["delta_hat"] for r in records]
            wd_values = [r["lambda_t"] for r in records]

            summary[name] = {
                "r_t_mean": sum(r_values) / len(r_values),
                "r_t_std": _std(r_values),
                "r_t_min": min(r_values),
                "r_t_max": max(r_values),
                "delta_hat_mean": sum(delta_values) / len(delta_values),
                "delta_hat_std": _std(delta_values),
                "lambda_t_mean": sum(wd_values) / len(wd_values),
                "lambda_t_std": _std(wd_values),
                "n_records": len(records),
            }

        return summary

    def save(self, path: Optional[str] = None):
        """Save all diagnostics to JSON files.

        Creates:
        - {path}/diagnostics_layers.json: per-layer step-level data
        - {path}/diagnostics_epochs.json: epoch-level data
        - {path}/diagnostics_summary.json: summary statistics
        """
        save_dir = Path(path) if path else self.output_dir
        if save_dir is None:
            raise ValueError("No output directory specified")

        save_dir.mkdir(parents=True, exist_ok=True)

        # Per-layer data (can be large; save only summary for very long runs)
        layer_data = {}
        for name, records in self._records.items():
            # Subsample if too many records (>10K per layer)
            if len(records) > 10000:
                step = len(records) // 10000
                layer_data[name] = records[::step]
            else:
                layer_data[name] = records

        with open(save_dir / "diagnostics_layers.json", "w") as f:
            json.dump(layer_data, f, indent=2)

        with open(save_dir / "diagnostics_epochs.json", "w") as f:
            json.dump(self._epoch_records, f, indent=2)

        with open(save_dir / "diagnostics_summary.json", "w") as f:
            json.dump(self.get_summary_stats(), f, indent=2)

    def reset(self):
        """Reset all records."""
        self._records = defaultdict(list)
        self._epoch_records = []


class ProgressReporter:
    """Reports training progress for system monitor integration."""

    def __init__(self, task_id: str, results_dir: str):
        self.task_id = task_id
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()

    def write_pid(self):
        """Write PID file for system recovery detection."""
        pid_file = self.results_dir / f"{self.task_id}.pid"
        pid_file.write_text(str(os.getpid()))

    def report_progress(
        self,
        epoch: int,
        total_epochs: int,
        step: int = 0,
        total_steps: int = 0,
        loss: Optional[float] = None,
        metric: Optional[dict] = None,
    ):
        """Write progress file for system monitor."""
        from datetime import datetime

        progress = self.results_dir / f"{self.task_id}_PROGRESS.json"
        data = {
            "task_id": self.task_id,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": step,
            "total_steps": total_steps,
            "loss": loss,
            "metric": metric or {},
            "updated_at": datetime.now().isoformat(),
        }
        progress.write_text(json.dumps(data))

    def mark_done(self, status: str = "success", summary: str = "",
                  results: Optional[dict] = None):
        """Write DONE marker file."""
        from datetime import datetime

        # Clean up PID file
        pid_file = self.results_dir / f"{self.task_id}.pid"
        if pid_file.exists():
            pid_file.unlink()

        # Read final progress
        progress_file = self.results_dir / f"{self.task_id}_PROGRESS.json"
        final_progress = {}
        if progress_file.exists():
            try:
                final_progress = json.loads(progress_file.read_text())
            except (json.JSONDecodeError, ValueError):
                pass

        marker = self.results_dir / f"{self.task_id}_DONE"
        data = {
            "task_id": self.task_id,
            "status": status,
            "summary": summary,
            "final_progress": final_progress,
            "results": results or {},
            "timestamp": datetime.now().isoformat(),
        }
        marker.write_text(json.dumps(data, indent=2))


def _std(values: list) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return var ** 0.5
