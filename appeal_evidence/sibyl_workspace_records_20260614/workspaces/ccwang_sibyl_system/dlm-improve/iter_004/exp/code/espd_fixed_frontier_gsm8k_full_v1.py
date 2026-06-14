#!/usr/bin/env python3
"""Compatibility wrapper for the ESPD fixed-frontier full-scale GSM8K control."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.workspace_root or os.environ.get(
        "SIBYL_REMOTE_PROJECT_ROOT",
        "/home/ccwang/sibyl_system/projects/dlm-improve",
    )
    os.environ["SIBYL_REMOTE_PROJECT_ROOT"] = project_root
    os.environ.setdefault("SIBYL_SETUP_DIR", f"{project_root}/current/exp/full_scale_espd_v1/setup")
    os.environ.setdefault("SIBYL_ARMS_DIR", f"{project_root}/current/exp/full_scale_espd_v1/arms")
    os.environ.setdefault("SIBYL_ESPD_RESULT_JSON", "espd_gsm8k_full_v1.json")
    os.environ.setdefault("SIBYL_GSM8K_CONTROLS_RESULT_JSON", "gsm8k_controls_full_v1.json")
    # The fixed-frontier control needs extra full-frontier forwards and OOMed at 52 on full GSM8K.
    os.environ.setdefault("SIBYL_FORCE_BATCH_SIZE", "48")

    results_dir = Path(os.environ.get("SIBYL_RESULTS_DIR", f"{project_root}/exp/results"))
    for suffix in (".pid", "_PROGRESS.json", "_DONE", ".json", "_gpu_profile.json"):
        target = results_dir / f"espd_fixed_frontier_gsm8k_full_v1{suffix}"
        if target.exists():
            target.unlink()

    sys.argv = [
        "build_espd_fixed_frontier_control.py",
        "--task-id",
        "espd_fixed_frontier_gsm8k_full_v1",
        "--arm-name",
        "espd_fixed_frontier",
    ]

    from build_espd_fixed_frontier_control import main as run_fixed_frontier

    return run_fixed_frontier()


if __name__ == "__main__":
    raise SystemExit(main())
