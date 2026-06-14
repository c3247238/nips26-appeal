#!/usr/bin/env python3
"""Compatibility wrapper for the full-scale GSM8K shared controls task."""

from __future__ import annotations

import argparse
import os


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
    os.environ.setdefault("SIBYL_TASK_ID", "refresh_gsm8k_controls_full_v1")
    os.environ.setdefault("SIBYL_RESULT_FILE", "gsm8k_controls_full_v1.json")

    from refresh_gsm8k_controls_v2 import main as run_refresh

    return run_refresh()


if __name__ == "__main__":
    raise SystemExit(main())
