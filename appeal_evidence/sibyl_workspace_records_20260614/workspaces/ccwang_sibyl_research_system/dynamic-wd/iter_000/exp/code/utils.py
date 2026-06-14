"""
Shared utility functions for the AADWD experiment suite.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def write_pid_file(task_id, results_dir):
    """Write PID file for system recovery detection."""
    pid_file = Path(results_dir) / f"{task_id}.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    return pid_file


def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor to track."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file for system monitor to detect."""
    # Clean up PID file
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    # Merge final progress if available
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    # Write DONE marker
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def update_gpu_progress(workspace_path, task_id, status="completed",
                        planned_min=0, actual_min=0, start_time=None,
                        end_time=None, config_snapshot=None):
    """Update gpu_progress.json after task completion."""
    progress_file = Path(workspace_path) / "exp" / "gpu_progress.json"
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    # Read existing
    if progress_file.exists():
        with open(progress_file) as f:
            data = json.loads(f.read())
    else:
        data = {"completed": [], "failed": [], "running": {}, "timings": {}}

    # Ensure keys exist
    data.setdefault("completed", [])
    data.setdefault("failed", [])
    data.setdefault("running", {})
    data.setdefault("timings", {})

    # Update status
    if status == "completed":
        if task_id not in data["completed"]:
            data["completed"].append(task_id)
    elif status == "failed":
        if task_id not in data["failed"]:
            data["failed"].append(task_id)

    # Remove from running
    data["running"].pop(task_id, None)

    # Record timing
    timing = {
        "planned_min": planned_min,
        "actual_min": actual_min,
    }
    if start_time:
        timing["start_time"] = start_time
    if end_time:
        timing["end_time"] = end_time
    if config_snapshot:
        timing["config_snapshot"] = config_snapshot
    data["timings"][task_id] = timing

    with open(progress_file, 'w') as f:
        json.dump(data, f, indent=2)
