#!/usr/bin/env python3
"""Collect small Sibyl workspace evidence files for the NeurIPS appeal.

The collector is intentionally conservative: it copies text, code,
configuration, logs, and small structured result files, while excluding model
weights, checkpoints, virtual environments, git object stores, caches, and
large generated artifacts. It also writes manifests so the evidence package is
auditable without pretending to include every raw byte.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_OUTPUT = Path("/home/ccwang/paper/sibyl-nips-apple/nips26-latex/appeal_evidence/sibyl_workspace_records_20260614")

SOURCES = [
    ("ccwang_sibyl_system", Path("/home/ccwang/sibyl-system/workspaces")),
    ("ccwang_sibyl_research_system", Path("/home/ccwang/sibyl-research-system/workspaces")),
    ("qhxie_autoresearch_sibylsystem", Path("/home/qhxie/AutoResearch-SibylSystem/workspaces")),
]

INCLUDED_WORKSPACES = {
    "attention-sink-comparison",
    "ablation-mem-negative",
    "ablation-mem-positive",
    "ablation-mem-weakened",
    "ablation-no-debate",
    "ablation-no-revision",
    "ablation-no-validation",
    "augmentation-order",
    "dlm-acceleration",
    "dlm-improve",
    "dynamic-wd",
    "lewm-generalization",
    "sae-absorption",
    "sae-absorption-kimi",
    "sae-absorption-minimax",
    "sae-feature-absorption-and-steering-minimization",
    "ttt-dlm",
}

ALLOWED_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".out",
    ".py",
    ".sh",
    ".sty",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

ALLOWED_FILENAMES = {
    "Makefile",
    "README",
    "README.md",
    "CLAUDE.md",
    "GEMINI.md",
    "AGENTS.md",
}

EXCLUDED_PARTS = {
    ".cache",
    ".claude",
    ".conda",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".serena",
    ".venv",
    "__pycache__",
    "checkpoints",
    "downloaded_saes",
    "lark_sync",
    "node_modules",
    "wandb",
}

EXCLUDED_SUFFIXES = {
    ".7z",
    ".bin",
    ".ckpt",
    ".dat",
    ".dll",
    ".dylib",
    ".egg",
    ".gz",
    ".h5",
    ".hdf5",
    ".joblib",
    ".npy",
    ".npz",
    ".onnx",
    ".pack",
    ".parquet",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".so",
    ".tar",
    ".tgz",
    ".zip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_excluded_part(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def is_allowed_file(path: Path) -> bool:
    if path.name in ALLOWED_FILENAMES:
        return True
    return path.suffix.lower() in ALLOWED_SUFFIXES


def is_known_large_artifact(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in EXCLUDED_SUFFIXES:
        return True
    lowered = str(path).lower()
    return any(token in lowered for token in ("/checkpoint", "/model", "/dataset", "/downloaded"))


def git_summary(workspace: Path) -> dict[str, str]:
    if not (workspace / ".git").exists():
        return {}

    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(workspace), *args],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except Exception:
            return ""

    return {
        "head": run_git(["rev-parse", "HEAD"]),
        "head_subject": run_git(["log", "-1", "--pretty=%h %aI %an %s"]),
        "status_short": run_git(["status", "--short"]),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect(output: Path, max_bytes: int, dry_run: bool) -> None:
    copied: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    workspaces_out = output / "workspaces"
    manifests_out = output / "manifests"
    if not dry_run:
        workspaces_out.mkdir(parents=True, exist_ok=True)
        manifests_out.mkdir(parents=True, exist_ok=True)

    for label, root in SOURCES:
        if not root.exists():
            summaries.append({"source_label": label, "source_root": str(root), "status": "missing"})
            continue

        for workspace in sorted(p for p in root.iterdir() if p.is_dir()):
            if workspace.name not in INCLUDED_WORKSPACES:
                summaries.append(
                    {
                        "source_label": label,
                        "workspace": workspace.name,
                        "source_path": str(workspace),
                        "status": "skipped_non_evidence_workspace",
                    }
                )
                continue

            workspace_key = f"{label}/{workspace.name}"
            copied_count = copied_bytes = excluded_count = excluded_bytes = seen_count = 0
            git_info = git_summary(workspace)

            for dirpath, dirnames, filenames in os.walk(workspace):
                current = Path(dirpath)
                dirnames[:] = [
                    name
                    for name in dirnames
                    if name not in EXCLUDED_PARTS and not has_excluded_part(current / name)
                ]

                for filename in filenames:
                    source = current / filename
                    rel = source.relative_to(workspace)
                    seen_count += 1

                    try:
                        stat = source.stat()
                    except OSError as exc:
                        excluded.append(
                            {
                                "source_label": label,
                                "workspace": workspace.name,
                                "source_path": str(source),
                                "size_bytes": "",
                                "reason": f"stat_failed:{exc}",
                            }
                        )
                        continue

                    reason = ""
                    if has_excluded_part(rel):
                        reason = "excluded_path_part"
                    elif is_known_large_artifact(rel):
                        reason = "known_large_or_binary_artifact"
                    elif stat.st_size > max_bytes:
                        reason = f"over_{max_bytes}_bytes"
                    elif not is_allowed_file(rel):
                        reason = "extension_not_in_evidence_whitelist"

                    if reason:
                        if reason != "extension_not_in_evidence_whitelist" or stat.st_size > max_bytes:
                            excluded.append(
                                {
                                    "source_label": label,
                                    "workspace": workspace.name,
                                    "source_path": str(source),
                                    "size_bytes": stat.st_size,
                                    "reason": reason,
                                }
                            )
                            excluded_count += 1
                            excluded_bytes += stat.st_size
                        continue

                    dest = workspaces_out / label / workspace.name / rel
                    if not dry_run:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, dest)

                    copied.append(
                        {
                            "source_label": label,
                            "workspace": workspace.name,
                            "source_path": str(source),
                            "relative_path": str(rel),
                            "dest_path": str(dest.relative_to(output)),
                            "size_bytes": stat.st_size,
                            "sha256": sha256_file(source),
                        }
                    )
                    copied_count += 1
                    copied_bytes += stat.st_size

            summaries.append(
                {
                    "source_label": label,
                    "workspace": workspace.name,
                    "source_path": str(workspace),
                    "status": "scanned",
                    "seen_files": seen_count,
                    "copied_files": copied_count,
                    "copied_bytes": copied_bytes,
                    "excluded_manifested_files": excluded_count,
                    "excluded_manifested_bytes": excluded_bytes,
                    "git_head": git_info.get("head", ""),
                    "git_head_subject": git_info.get("head_subject", ""),
                    "git_status_short": git_info.get("status_short", ""),
                }
            )

    if dry_run:
        print(json.dumps({"dry_run": True, "copied_files": len(copied), "copied_bytes": sum(int(r["size_bytes"]) for r in copied)}, indent=2))
        return

    write_csv(
        manifests_out / "copied_files.csv",
        copied,
        ["source_label", "workspace", "source_path", "relative_path", "dest_path", "size_bytes", "sha256"],
    )
    write_csv(
        manifests_out / "excluded_large_or_binary_files.csv",
        excluded,
        ["source_label", "workspace", "source_path", "size_bytes", "reason"],
    )
    write_csv(
        manifests_out / "workspace_summary.csv",
        summaries,
        [
            "source_label",
            "workspace",
            "source_path",
            "status",
            "seen_files",
            "copied_files",
            "copied_bytes",
            "excluded_manifested_files",
            "excluded_manifested_bytes",
            "git_head",
            "git_head_subject",
            "git_status_short",
        ],
    )

    (manifests_out / "collection_policy.json").write_text(
        json.dumps(
            {
                "max_file_bytes": max_bytes,
                "source_roots": [{"label": label, "path": str(path)} for label, path in SOURCES],
                "included_workspaces": sorted(INCLUDED_WORKSPACES),
                "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
                "allowed_filenames": sorted(ALLOWED_FILENAMES),
                "excluded_path_parts": sorted(EXCLUDED_PARTS),
                "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"copied_files": len(copied), "copied_bytes": sum(int(r["size_bytes"]) for r in copied), "output": str(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    collect(args.output, args.max_bytes, args.dry_run)


if __name__ == "__main__":
    main()
