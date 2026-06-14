#!/usr/bin/env python3
"""Build the iteration-4 runtime contract v2 for screening pilots."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

TASK_ID = "setup_runtime_contract_v2"
RESULT_FILE = "iteration4_runtime_contract_v2.json"
DEFAULT_MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
PROMPT_MAX_TOKENS = 512
GENERATION_TOKENS = 256
TOTAL_SEQ_LEN = PROMPT_MAX_TOKENS + GENERATION_TOKENS
SEED = 42


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def report_progress(results_dir: Path, *, step: int, total_steps: int, metric: dict[str, Any]) -> None:
    write_json(
        results_dir / f"{TASK_ID}_PROGRESS.json",
        {
            "task_id": TASK_ID,
            "epoch": 1,
            "total_epochs": 1,
            "step": step,
            "total_steps": total_steps,
            "loss": None,
            "metric": metric,
            "updated_at": now_iso(),
        },
    )


def mark_done(results_dir: Path, *, status: str, summary: str, extra: dict[str, Any] | None = None) -> None:
    pid_file = results_dir / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()

    final_progress: dict[str, Any] = {}
    progress_file = results_dir / f"{TASK_ID}_PROGRESS.json"
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            final_progress = {}

    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": now_iso(),
        "result_path": str(results_dir / RESULT_FILE),
    }
    if extra:
        payload.update(extra)

    write_json(results_dir / f"{TASK_ID}_DONE", payload)


def log_line(results_dir: Path, message: str) -> None:
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    with (results_dir / f"{TASK_ID}.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def locate_task_metadata(task_plan: dict[str, Any]) -> dict[str, Any]:
    for task in task_plan.get("tasks", []):
        if task.get("id") == TASK_ID:
            return dict(task)
    return {}


def compatibility_runtime_path(setup_dir: Path) -> Path:
    return setup_dir / "runtime_contract.json"


def compatibility_batch_probe_path(setup_dir: Path) -> Path:
    return setup_dir / "batch_probe.json"


def compatibility_env_check_path(setup_dir: Path) -> Path:
    return setup_dir / "env_check.json"


def compatibility_shared_gpu_profile(results_dir: Path) -> Path:
    return results_dir / "shared_runtime_probe_gpu_profile.json"


def compatibility_shared_progress(results_dir: Path) -> Path:
    return results_dir / "shared_runtime_probe_PROGRESS.json"


def setup_roots(workspace_root: Path) -> list[Path]:
    roots = [workspace_root / "exp" / "pilot_evidence_closure_v1" / "setup"]
    current_setup = workspace_root / "current" / "exp" / "pilot_evidence_closure_v1" / "setup"
    if current_setup not in roots:
        roots.append(current_setup)
    return roots


def existing_manifest_source(workspace_root: Path, roots: list[Path]) -> Path | None:
    candidates = [
        *(root / "sample_manifest.json" for root in roots),
        workspace_root / "iter_003" / "exp" / "pilot_evidence_closure_v1" / "setup" / "sample_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def attention_backend(engine: Any) -> str:
    model = getattr(engine, "model", None)
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "module"):
        config = getattr(model.module, "config", None)
    return getattr(config, "_attn_implementation", None) or "eager"


def compile_enabled(engine: Any) -> bool:
    model = getattr(engine, "model", None)
    return hasattr(model, "_orig_mod") or (
        hasattr(model, "module") and hasattr(model.module, "_orig_mod")
    )


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("torch", "transformers", "datasets", "numpy", "accelerate"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "NOT_INSTALLED"
    versions["flash_attn"] = (
        "available" if importlib.util.find_spec("flash_attn") is not None else "NOT_INSTALLED"
    )
    versions["ptxas"] = shutil.which("ptxas") or "NOT_FOUND"
    return versions


def gpu_env() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "flash_attn_available": importlib.util.find_spec("flash_attn") is not None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "gpu_name": props.name,
                "vram_total_mb": round(props.total_memory / 1024**2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return payload


def benchmark_forward(engine: Any, *, batch_size: int, seq_len: int, repeats: int = 3) -> dict[str, Any]:
    input_ids = torch.full(
        (batch_size, seq_len),
        engine.pad_token_id,
        dtype=torch.long,
        device=engine.device,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        engine.model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            engine.model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    mean_sec = sum(timings) / len(timings)
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "mean_forward_sec": round(mean_sec, 4),
        "tokens_per_sec": round(batch_size * seq_len / mean_sec, 2),
    }


def probe_runtime_path(model_path: str, *, prefer_flash: bool, prefer_compile: bool) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from batched_dlm_utils import BatchedDLMInference

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    engine = BatchedDLMInference(
        model_path=model_path,
        device="cuda",
        use_flash_attn=prefer_flash,
        use_compile=prefer_compile,
    )
    backend = attention_backend(engine)
    compile_ok = compile_enabled(engine)
    safe_batch = int(
        engine.find_max_batch_size(
            gen_length=GENERATION_TOKENS,
            prompt_max_len=PROMPT_MAX_TOKENS,
            lo=1,
            hi=256,
            safety_margin=0.9,
        )
    )
    throughput = benchmark_forward(
        engine,
        batch_size=max(1, safe_batch),
        seq_len=TOTAL_SEQ_LEN,
        repeats=3,
    )
    peak_vram_mb = round(torch.cuda.max_memory_allocated(engine.device) / 1024**2)
    props = torch.cuda.get_device_properties(engine.device)
    result = {
        "requested_flash_attention": prefer_flash,
        "requested_compile": prefer_compile,
        "resolved_backend": backend,
        "compile_enabled": compile_ok,
        "safe_batch_size": safe_batch,
        "peak_vram_mb": peak_vram_mb,
        "gpu_name": props.name,
        "vram_total_mb": round(props.total_memory / 1024**2),
        "throughput": throughput,
        "duration_sec": round(time.perf_counter() - started, 2),
    }
    del engine
    torch.cuda.empty_cache()
    gc.collect()
    return result


def choose_best_path(paths: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        paths,
        key=lambda row: (
            int(row.get("safe_batch_size", 0)),
            float((row.get("throughput") or {}).get("tokens_per_sec", 0.0)),
            int(bool(row.get("compile_enabled"))),
        ),
    )


def build_runtime_contract(
    *,
    existing_contract: dict[str, Any],
    best_path: dict[str, Any],
    task_meta: dict[str, Any],
    workspace_root: Path,
) -> dict[str, Any]:
    runtime = dict(existing_contract.get("runtime", {}))
    runtime.update(
        {
            "model_path": existing_contract.get("model_path", DEFAULT_MODEL_PATH),
            "resolved_attn_implementation": best_path["resolved_backend"],
            "requested_attn_implementation": (
                "flash_attention_2" if best_path["requested_flash_attention"] else "eager"
            ),
            "flash_attention_available": importlib.util.find_spec("flash_attn") is not None,
            "torch_compile_enabled": bool(best_path["compile_enabled"]),
            "safe_batch_size": int(best_path["safe_batch_size"]),
            "tokens_per_sec": float(best_path["throughput"]["tokens_per_sec"]),
            "peak_vram_mb": int(best_path["peak_vram_mb"]),
            "probe_total_seq_len": TOTAL_SEQ_LEN,
        }
    )

    auxiliary_fields = [
        "extra_forward_passes",
        "auxiliary_overhead_sec",
        "frontier_selection_overhead_sec",
        "syntax_guard_overhead_sec",
        "uplift_estimation_overhead_sec",
    ]

    contract = {
        "task_id": TASK_ID,
        "timestamp": now_iso(),
        "workspace_root": str(workspace_root),
        "model_family": existing_contract.get("model_family", "LLaDA-8B-Instruct"),
        "model_path": existing_contract.get("model_path", DEFAULT_MODEL_PATH),
        "tokenizer_name_or_path": existing_contract.get(
            "tokenizer_name_or_path",
            DEFAULT_MODEL_PATH,
        ),
        "temperature": existing_contract.get("temperature", 1.0),
        "prompt_template_policy": existing_contract.get(
            "prompt_template_policy",
            "Frozen across all screening arms",
        ),
        "postprocess_policy": existing_contract.get(
            "postprocess_policy",
            "Frozen across all screening arms",
        ),
        "left_padding": True,
        "prompt_max_tokens": PROMPT_MAX_TOKENS,
        "generation_tokens": GENERATION_TOKENS,
        "selection_manifest_path": existing_contract.get(
            "selection_manifest_path",
            "current/exp/pilot_evidence_closure_v1/setup/sample_manifest.json",
        ),
        "dataset_sources": existing_contract.get(
            "dataset_sources",
            {
                "gsm8k": "/home/ccwang/sibyl_system/shared/datasets/gsm8k",
                "mbpp": "/home/ccwang/sibyl_system/shared/datasets/mbpp",
            },
        ),
        "runtime": runtime,
        "batch_probe_summary": {
            "min_safe_batch_size": int(best_path["safe_batch_size"]),
            "max_safe_batch_size": int(best_path["safe_batch_size"]),
        },
        "auxiliary_ledger_fields": auxiliary_fields,
        "runtime_contract_policy": {
            "backend_compile_parity_required": True,
            "batch_mode": "auto_detect",
            "phase": "iteration4_screening_v1",
        },
        "task_meta": {
            "id": task_meta.get("id", TASK_ID),
            "expected_output": task_meta.get("expected_output", f"exp/results/{RESULT_FILE}"),
            "estimated_minutes": task_meta.get("estimated_minutes"),
            "pass_criteria": task_meta.get("pilot", {}).get("pass_criteria", ""),
        },
    }
    return contract


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    results_dir = workspace_root / "exp" / "results"
    plan_dir = workspace_root / "plan"
    idea_dir = workspace_root / "idea"
    setup_dirs = setup_roots(workspace_root)
    primary_setup_dir = setup_dirs[0]

    results_dir.mkdir(parents=True, exist_ok=True)
    for setup_dir in setup_dirs:
        setup_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")

    task_plan = load_json(plan_dir / "task_plan.json")
    task_meta = locate_task_metadata(task_plan)
    methodology_text = load_text(plan_dir / "methodology.md")
    proposal_text = load_text(idea_dir / "proposal.md")
    existing_contract = load_json(compatibility_runtime_path(primary_setup_dir))

    report_progress(
        results_dir,
        step=1,
        total_steps=4,
        metric={
            "phase": "bootstrap",
            "task_found": bool(task_meta),
            "plan_present": (plan_dir / "task_plan.json").exists(),
            "methodology_present": bool(methodology_text),
            "proposal_present": bool(proposal_text),
        },
    )
    log_line(results_dir, "开始建立 runtime contract v2，并检查当前迭代 planning 资产。")

    if not torch.cuda.is_available():
        summary = "CUDA 不可用，无法建立 runtime contract v2。"
        write_json(results_dir / RESULT_FILE, {"task_id": TASK_ID, "status": "failed", "summary": summary})
        mark_done(results_dir, status="failed", summary=summary)
        return 1

    env_info = {
        "packages": package_versions(),
        "gpu": gpu_env(),
        "model_path_exists": Path(args.model_path).exists(),
    }
    for setup_dir in setup_dirs:
        write_json(compatibility_env_check_path(setup_dir), env_info)
    report_progress(
        results_dir,
        step=2,
        total_steps=4,
        metric={
            "phase": "env_checked",
            "flash_attn_available": env_info["gpu"].get("flash_attn_available", False),
            "ptxas": env_info["packages"].get("ptxas", "NOT_FOUND"),
        },
    )

    runtime_paths: list[dict[str, Any]] = []
    try:
        flash_available = bool(env_info["gpu"].get("flash_attn_available", False))
        candidate_paths = (
            [(True, False), (True, True)] if flash_available else [(False, False), (False, True)]
        )
        for prefer_flash, prefer_compile in candidate_paths:
            label = (
                f"{'flash_attention_2' if prefer_flash else 'eager'}|compile={prefer_compile}"
            )
            log_line(results_dir, f"探测 runtime path: {label}")
            path_result = probe_runtime_path(
                args.model_path,
                prefer_flash=prefer_flash,
                prefer_compile=prefer_compile,
            )
            runtime_paths.append(path_result)
            report_progress(
                results_dir,
                step=3,
                total_steps=4,
                metric={
                    "phase": "runtime_probe",
                    "runtime_path": label,
                    "resolved_backend": path_result["resolved_backend"],
                    "compile_enabled": path_result["compile_enabled"],
                    "safe_batch_size": path_result["safe_batch_size"],
                    "tokens_per_sec": path_result["throughput"]["tokens_per_sec"],
                },
            )
    except Exception as exc:  # noqa: BLE001
        summary = f"runtime path 探测失败：{exc!r}"
        write_json(
            results_dir / RESULT_FILE,
            {
                "task_id": TASK_ID,
                "status": "failed",
                "summary": summary,
                "traceback": traceback.format_exc(),
                "env_info": env_info,
            },
        )
        mark_done(results_dir, status="failed", summary=summary)
        return 1

    best_path = choose_best_path(runtime_paths)
    compile_candidates = [row for row in runtime_paths if row["resolved_backend"] == best_path["resolved_backend"]]
    parity = {
        "backend_consistent": all(
            row["resolved_backend"] == best_path["resolved_backend"] for row in runtime_paths
        ),
        "compile_variant_present": any(row["requested_compile"] for row in compile_candidates),
        "compile_enabled_best_path": bool(best_path["compile_enabled"]),
        "safe_batch_delta_vs_noncompile": None,
    }
    noncompile_peer = next(
        (
            row
            for row in compile_candidates
            if not row["requested_compile"]
        ),
        None,
    )
    if noncompile_peer is not None:
        parity["safe_batch_delta_vs_noncompile"] = int(best_path["safe_batch_size"]) - int(
            noncompile_peer["safe_batch_size"]
        )

    runtime_contract = build_runtime_contract(
        existing_contract=existing_contract,
        best_path=best_path,
        task_meta=task_meta,
        workspace_root=workspace_root,
    )
    result_payload = {
        "task_id": TASK_ID,
        "status": "success",
        "summary": (
            f"已冻结 runtime contract v2：backend={best_path['resolved_backend']}，"
            f"compile={'on' if best_path['compile_enabled'] else 'off'}，"
            f"safe_batch={best_path['safe_batch_size']}，"
            f"tps={best_path['throughput']['tokens_per_sec']}"
        ),
        "checked_at": now_iso(),
        "workspace_root": str(workspace_root),
        "env_info": env_info,
        "runtime_paths": runtime_paths,
        "best_path": best_path,
        "parity": parity,
        "auxiliary_ledger_fields": runtime_contract["auxiliary_ledger_fields"],
        "evidence_files": {
            "task_plan": str(plan_dir / "task_plan.json"),
            "methodology": str(plan_dir / "methodology.md"),
            "proposal": str(idea_dir / "proposal.md"),
            "runtime_contract": str(compatibility_runtime_path(primary_setup_dir)),
        },
    }

    write_json(results_dir / RESULT_FILE, result_payload)
    batch_probe_payload = {
        "task_id": TASK_ID,
        "timestamp": now_iso(),
        "probe_start_batch": 256,
        "probe_method": "binary_search_down_on_oom",
        "rows": [
            {
                "bucket": "screening_runtime",
                "prompt_tokens": PROMPT_MAX_TOKENS,
                "gen_tokens": GENERATION_TOKENS,
                "runtime_path": row["resolved_backend"],
                "compile_enabled": row["compile_enabled"],
                "safe_batch_size": row["safe_batch_size"],
                "peak_vram_mb": row["peak_vram_mb"],
                "tokens_per_sec": row["throughput"]["tokens_per_sec"],
            }
            for row in runtime_paths
        ],
        "buckets": {
            f"{row['resolved_backend']}|compile={int(bool(row['compile_enabled']))}": {
                "safe_batch_size": row["safe_batch_size"],
                "peak_vram_mb": row["peak_vram_mb"],
                "tokens_per_sec": row["throughput"]["tokens_per_sec"],
                "compile_enabled": row["compile_enabled"],
                "attention_backend": row["resolved_backend"],
            }
            for row in runtime_paths
        },
    }
    for setup_dir in setup_dirs:
        write_json(compatibility_runtime_path(setup_dir), runtime_contract)
        write_json(compatibility_batch_probe_path(setup_dir), batch_probe_payload)

    manifest_src = existing_manifest_source(workspace_root, setup_dirs)
    if manifest_src is not None:
        manifest_payload = json.loads(manifest_src.read_text(encoding="utf-8"))
        for setup_dir in setup_dirs:
            write_json(setup_dir / "sample_manifest.json", manifest_payload)

    gpu_profile = {
        "task_id": TASK_ID,
        "gpu_name": best_path["gpu_name"],
        "vram_total_mb": best_path["vram_total_mb"],
        "max_batch_size": best_path["safe_batch_size"],
        "peak_vram_mb": best_path["peak_vram_mb"],
        "utilization_pct": round(
            100.0 * float(best_path["peak_vram_mb"]) / max(1.0, float(best_path["vram_total_mb"])),
            1,
        ),
        "attention_backend": best_path["resolved_backend"],
        "compile_tested": bool(best_path["compile_enabled"]),
        "tokens_per_sec": best_path["throughput"]["tokens_per_sec"],
        "prompt_max_tokens": PROMPT_MAX_TOKENS,
        "generation_tokens": GENERATION_TOKENS,
    }
    write_json(results_dir / f"{TASK_ID}_gpu_profile.json", gpu_profile)
    write_json(compatibility_shared_gpu_profile(results_dir), gpu_profile)
    write_json(
        compatibility_shared_progress(results_dir),
        {
            "task_id": "shared_runtime_probe",
            "epoch": 1,
            "total_epochs": 1,
            "step": 1,
            "total_steps": 1,
            "loss": None,
            "metric": {
                "phase": "done",
                "safe_batch_size": best_path["safe_batch_size"],
                "tokens_per_sec": best_path["throughput"]["tokens_per_sec"],
                "attn_backend": best_path["resolved_backend"],
                "compile_ok": bool(best_path["compile_enabled"]),
            },
            "updated_at": now_iso(),
        },
    )

    report_progress(
        results_dir,
        step=4,
        total_steps=4,
        metric={
            "phase": "done",
            "resolved_backend": best_path["resolved_backend"],
            "compile_enabled": bool(best_path["compile_enabled"]),
            "safe_batch_size": best_path["safe_batch_size"],
            "tokens_per_sec": best_path["throughput"]["tokens_per_sec"],
        },
    )
    mark_done(results_dir, status="success", summary=result_payload["summary"])
    log_line(results_dir, result_payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
