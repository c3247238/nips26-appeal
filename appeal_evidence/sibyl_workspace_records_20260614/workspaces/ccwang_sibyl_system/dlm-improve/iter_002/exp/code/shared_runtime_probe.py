#!/usr/bin/env python3
"""Remote runtime probe for iter_002 honest-compute experiments.

This script is designed to run on cs8000d and write Sibyl-compatible
PID / PROGRESS / DONE markers under the remote project root layout:
  /home/ccwang/sibyl_system/projects/dlm-improve/exp/results/
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

TASK_ID = "shared_runtime_probe"
PROJECT_ROOT = Path("/home/ccwang/sibyl_system/projects/dlm-improve")
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
RESULTS_PATH = RESULTS_DIR / "runtime_probe_iter2.json"
GPU_PROFILE_PATH = RESULTS_DIR / f"{TASK_ID}_gpu_profile.json"
LOG_PATH = RESULTS_DIR / f"{TASK_ID}.log"
REGISTRY_PATH = Path("/home/ccwang/sibyl_system/shared/registry.json")
DEFAULT_MODEL_PATH = Path("/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct")
VISIBLE_GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
SEED = 42


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def report_progress(epoch: int, total_epochs: int, metric: dict | None = None) -> None:
    progress_path = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    write_json(
        progress_path,
        {
            "task_id": TASK_ID,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": 0,
            "total_steps": 0,
            "loss": None,
            "metric": metric or {},
            "updated_at": now_iso(),
        },
    )


def mark_done(status: str, summary: str) -> None:
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    final_progress = {}
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            final_progress = {}
    write_json(
        RESULTS_DIR / f"{TASK_ID}_DONE",
        {
            "task_id": TASK_ID,
            "status": status,
            "summary": summary,
            "final_progress": final_progress,
            "timestamp": now_iso(),
        },
    )


def resolve_model_path() -> tuple[Path, dict]:
    info = {
        "registry_path": str(REGISTRY_PATH),
        "default_model_path": str(DEFAULT_MODEL_PATH),
        "shared_hint": "shared/checkpoints/llada-8b",
    }
    if REGISTRY_PATH.exists():
        try:
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            ckpt = registry.get("checkpoints", {}).get("llada_8b_instruct", {})
            target = ckpt.get("target")
            if target and Path(target).exists():
                info["registry_target_used"] = target
                return Path(target), info
        except (json.JSONDecodeError, OSError) as exc:
            info["registry_error"] = repr(exc)
    if DEFAULT_MODEL_PATH.exists():
        info["fallback_used"] = str(DEFAULT_MODEL_PATH)
        return DEFAULT_MODEL_PATH, info
    raise FileNotFoundError("No usable LLaDA-8B-Instruct model path found on remote host")


def flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def datasets_available() -> dict:
    dataset_root = Path("/home/ccwang/sibyl_system/shared/datasets/gsm8k")
    return {
        "path": str(dataset_root),
        "exists": dataset_root.exists(),
    }


def build_dummy_inputs(batch_size: int, seq_len: int, pad_token_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask


def try_load_model(model_path: Path, attn_impl: str | None, use_compile: bool) -> tuple[object, object, dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs).to("cuda:0").eval()
    compiled = False
    compile_error = None
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            compiled = True
        except Exception as exc:  # noqa: BLE001
            compile_error = repr(exc)
    meta = {
        "requested_attn_impl": attn_impl or "default",
        "resolved_attn_backend": getattr(getattr(model, "config", None), "_attn_implementation", None) or attn_impl or "default",
        "compile_requested": use_compile,
        "compile_enabled": compiled,
        "compile_error": compile_error,
        "load_sec": round(time.time() - t0, 2),
        "pad_token_id": tokenizer.pad_token_id,
    }
    return model, tokenizer, meta


def probe_max_batch(model: object, pad_token_id: int, seq_len: int, lo: int = 1, hi: int = 128, safety_margin: float = 0.9) -> tuple[int, int]:
    best = lo
    peak = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            inputs, attention_mask = build_dummy_inputs(mid, seq_len, pad_token_id, "cuda:0")
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=inputs, attention_mask=attention_mask)
            torch.cuda.synchronize()
            peak = max(peak, int(torch.cuda.max_memory_allocated(0) / 1024**2))
            best = mid
            lo = mid + 1
        except torch.cuda.OutOfMemoryError:
            hi = mid - 1
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    return max(1, int(best * safety_margin)), peak


def benchmark_forward(model: object, pad_token_id: int, batch_size: int, seq_len: int, repeats: int = 3) -> dict:
    inputs, attention_mask = build_dummy_inputs(batch_size, seq_len, pad_token_id, "cuda:0")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, attention_mask=attention_mask)
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=inputs, attention_mask=attention_mask)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - t0)
    mean_sec = sum(elapsed) / len(elapsed)
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "mean_forward_sec": round(mean_sec, 4),
        "tokens_per_sec": round(batch_size * seq_len / mean_sec, 2),
    }


def dataparallel_smoke(model_path: Path, pad_token_id: int) -> dict:
    if torch.cuda.device_count() < 2:
        return {"supported": False, "reason": "fewer_than_2_visible_gpus"}
    from transformers import AutoModelForCausalLM

    result = {"supported": False, "visible_gpu_count": torch.cuda.device_count()}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
        ).eval()
        wrapped = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
        inputs, attention_mask = build_dummy_inputs(2, 256, pad_token_id, "cuda:0")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            wrapped(input_ids=inputs, attention_mask=attention_mask)
        torch.cuda.synchronize()
        result.update(
            {
                "supported": True,
                "mode": "DataParallel",
                "batch_smoke": 2,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
    finally:
        torch.cuda.empty_cache()
        gc.collect()
    return result


def collect_gpu_info() -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": torch.cuda.is_available(),
        "visible_devices": VISIBLE_GPUS,
        "device_count": torch.cuda.device_count(),
        "gpu_name": props.name,
        "vram_total_mb": round(props.total_memory / 1024**2),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "flash_attn_available": flash_attn_available(),
    }


def run_runtime_path(model_path: Path, attn_impl: str | None, use_compile: bool) -> dict:
    torch.manual_seed(SEED)
    started = time.time()
    model, tokenizer, meta = try_load_model(model_path, attn_impl=attn_impl, use_compile=use_compile)
    pad_token_id = tokenizer.pad_token_id
    safe_batch, peak_vram_mb = probe_max_batch(model, pad_token_id=pad_token_id, seq_len=768)
    bench = benchmark_forward(model, pad_token_id=pad_token_id, batch_size=max(1, safe_batch), seq_len=768)
    path_result = {
        "runtime_path": f"{meta['resolved_attn_backend']}|compile={meta['compile_enabled']}",
        "attn_backend": meta["resolved_attn_backend"],
        "compile_requested": meta["compile_requested"],
        "compile_enabled": meta["compile_enabled"],
        "compile_error": meta["compile_error"],
        "load_sec": meta["load_sec"],
        "safe_batch_size": safe_batch,
        "peak_vram_mb": peak_vram_mb,
        "throughput": bench,
        "duration_sec": round(time.time() - started, 2),
    }
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return path_result


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()), encoding="utf-8")
    report_progress(0, 4, {"phase": "init", "visible_devices": VISIBLE_GPUS})

    results = {
        "task_id": TASK_ID,
        "timestamp": now_iso(),
        "status": "running",
        "seed": SEED,
        "visible_devices": VISIBLE_GPUS,
        "paths": [],
    }

    try:
        report_progress(1, 4, {"phase": "env_check"})
        model_path, model_info = resolve_model_path()
        gpu_info = collect_gpu_info()
        dataset_info = datasets_available()
        results["model_resolution"] = model_info
        results["gpu_info"] = gpu_info
        results["dataset_info"] = dataset_info

        requested_paths = []
        if gpu_info["flash_attn_available"]:
            requested_paths.append(("flash_attention_2", False))
            requested_paths.append(("flash_attention_2", True))
        else:
            requested_paths.append(("eager", False))
            requested_paths.append(("eager", True))

        report_progress(2, 4, {"phase": "runtime_paths", "requested_paths": [f"{a}|compile={c}" for a, c in requested_paths]})
        for attn_impl, use_compile in requested_paths:
            path_result = run_runtime_path(model_path, attn_impl=attn_impl, use_compile=use_compile)
            results["paths"].append(path_result)
            report_progress(
                2,
                4,
                {
                    "phase": "runtime_paths",
                    "last_path": path_result["runtime_path"],
                    "safe_batch_size": path_result["safe_batch_size"],
                    "tokens_per_sec": path_result["throughput"]["tokens_per_sec"],
                },
            )

        report_progress(3, 4, {"phase": "multi_gpu_smoke"})
        pad_token_id = 0
        if results["paths"]:
            path0 = results["paths"][0]
            # Reuse tokenizer-derived pad id from path probe metadata is not persisted;
            # eos/pad value does not matter for this smoke test.
            pad_token_id = 0
        dp_result = dataparallel_smoke(model_path, pad_token_id=pad_token_id)
        results["multi_gpu_smoke"] = dp_result

        best_path = max(results["paths"], key=lambda item: (item["safe_batch_size"], item["throughput"]["tokens_per_sec"]))
        results["recommended_runtime_path"] = best_path["runtime_path"]
        results["recommended_batch_size"] = best_path["safe_batch_size"]
        results["status"] = "success" if best_path["safe_batch_size"] >= 16 else "failed"
        results["pass"] = best_path["safe_batch_size"] >= 16
        results["summary"] = (
            f"path={best_path['runtime_path']}, batch={best_path['safe_batch_size']}, "
            f"backend={best_path['attn_backend']}, tps={best_path['throughput']['tokens_per_sec']}, "
            f"multi_gpu={'ok' if dp_result.get('supported') else 'no'}"
        )

        gpu_profile = {
            "gpu_name": gpu_info["gpu_name"],
            "vram_total_mb": gpu_info["vram_total_mb"],
            "max_batch_size": best_path["safe_batch_size"],
            "vram_used_mb": best_path["peak_vram_mb"],
            "utilization_pct": round(100.0 * best_path["peak_vram_mb"] / max(1, gpu_info["vram_total_mb"]), 1),
            "attention_backend": best_path["attn_backend"],
            "compile_enabled": best_path["compile_enabled"],
            "runtime_path": best_path["runtime_path"],
            "visible_devices": VISIBLE_GPUS,
        }
        write_json(GPU_PROFILE_PATH, gpu_profile)
        write_json(RESULTS_PATH, results)
        report_progress(
            4,
            4,
            {
                "phase": "done",
                "recommended_runtime_path": best_path["runtime_path"],
                "recommended_batch_size": best_path["safe_batch_size"],
                "tokens_per_sec": best_path["throughput"]["tokens_per_sec"],
                "multi_gpu_supported": dp_result.get("supported", False),
            },
        )
        mark_done("success" if results["pass"] else "failed", results["summary"])
        return 0 if results["pass"] else 1
    except Exception as exc:  # noqa: BLE001
        results["status"] = "failed"
        results["pass"] = False
        results["error"] = repr(exc)
        write_json(RESULTS_PATH, results)
        report_progress(4, 4, {"phase": "failed", "error": repr(exc)})
        mark_done("failed", repr(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
