#!/usr/bin/env python3
"""Compatibility wrapper for the ESPD full-scale GSM8K task."""

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
    os.environ.setdefault("SIBYL_SUMMARY_BASENAME", "espd_gsm8k_full_v1")
    # 57 and 54 both OOMed on the retained runtime contract; cap future relaunches at 52.
    os.environ.setdefault("SIBYL_FORCE_BATCH_SIZE", "52")

    results_dir = Path(os.environ.get("SIBYL_RESULTS_DIR", f"{project_root}/exp/results"))
    for suffix in (".pid", "_PROGRESS.json", "_DONE", ".json", "_gpu_profile.json"):
        target = results_dir / f"espd_gsm8k_full_v1{suffix}"
        if target.exists():
            target.unlink()

    sys.argv = [
        "implement_espd.py",
        "--task-id",
        "espd_gsm8k_full_v1",
        "--arm-name",
        "espd",
    ]

    from implement_espd import main as run_espd

    return run_espd()


if __name__ == "__main__":
    raise SystemExit(main())
