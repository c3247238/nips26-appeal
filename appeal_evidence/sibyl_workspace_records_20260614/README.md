# Sibyl Workspace Evidence Records

This directory collects small, auditable records from local Sibyl workspaces for
the NeurIPS 2026 position-paper appeal. The goal is to expose the underlying
workspace traces behind the paper's process-evidence claims without uploading
model weights, checkpoints, virtual environments, downloaded datasets, git
object stores, or large binary artifacts.

## Source Roots

- `/home/ccwang/sibyl-system/workspaces`
- `/home/ccwang/sibyl-research-system/workspaces`
- `/home/qhxie/AutoResearch-SibylSystem/workspaces`

## Included Workspace Records

The collected workspaces include the main experimental and diagnostic workspaces
used to support the Sibyl position paper:

- `dynamic-wd`
- `dlm-acceleration`
- `sae-absorption`
- `sae-absorption-kimi`
- `sae-absorption-minimax`
- `augmentation-order`
- `lewm-generalization`
- `ablation-mem-negative`
- `ablation-mem-positive`
- `ablation-mem-weakened`
- `ablation-no-debate`
- `ablation-no-revision`
- `ablation-no-validation`
- `dlm-improve`
- `ttt-dlm`
- `attention-sink-comparison`
- `sae-feature-absorption-and-steering-minimization`

## What Was Copied

The collector copies small text, code, configuration, log, and structured result
files, including:

- planning, reflection, critic, supervisor, and decision records
- workspace context, idea, methodology, and literature notes
- paper drafts, LaTeX source, outlines, and review artifacts
- Python and shell scripts used for experiments or analysis
- small JSON, JSONL, CSV, TSV, YAML, TOML, log, and text outputs

The per-file copy threshold used for this package is 128 KiB.

## What Was Excluded

The package intentionally excludes large or non-auditable runtime artifacts:

- `.git`, `.venv`, `.serena`, caches, and Python bytecode
- downloaded checkpoints, model weights, and `downloaded_saes`
- tensor dumps, NumPy arrays, pickles, archives, PDFs, PNGs, and large binaries
- files over 128 KiB
- smoke-test, toy-test, and old backup workspaces that are not part of the
  paper's evidence chain

Excluded large or binary files are summarized in
`manifests/excluded_large_or_binary_files.csv` when they are relevant enough to
record.

## Manifests

- `manifests/copied_files.csv`: every copied file, original path, destination,
  size, and SHA-256 hash.
- `manifests/workspace_summary.csv`: per-workspace copied file counts, copied
  bytes, and available local git HEAD metadata.
- `manifests/excluded_large_or_binary_files.csv`: large or binary artifacts that
  were not copied.
- `manifests/collection_policy.json`: machine-readable collection policy.

The collection script is `scripts/collect_workspace_evidence.py`.
