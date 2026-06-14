#!/usr/bin/env python3
"""Build iter_003 pilot manifest and runtime contract on the remote host."""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

TASK_ID = "setup_runtime_contract"
PROJECT_ROOT = Path(os.environ.get("SIBYL_REMOTE_PROJECT_ROOT", "/home/ccwang/sibyl_system/projects/dlm-improve"))
RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
SETUP_DIR = PROJECT_ROOT / "current" / "exp" / "pilot_evidence_closure_v1" / "setup"
MODEL_PATH = Path("/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct")
GSM8K_PATH = Path("/home/ccwang/sibyl_system/shared/datasets/gsm8k")
MBPP_PATH = Path("/home/ccwang/sibyl_system/shared/datasets/mbpp")
SEED = 42
MASK_TOKEN_ID = 126336
GEN_LENGTH = 128
PRESCAN_POOL = 100
PROMPT_MAX_TOKENS = 512


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def progress_path() -> Path:
    return RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"


def pid_path() -> Path:
    return RESULTS_DIR / f"{TASK_ID}.pid"


def done_path() -> Path:
    return RESULTS_DIR / f"{TASK_ID}_DONE"


def log_path() -> Path:
    return RESULTS_DIR / f"{TASK_ID}.log"


def log(msg: str) -> None:
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    with log_path().open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def report_progress(epoch: int, total_epochs: int, metric: dict | None = None) -> None:
    write_json(
        progress_path(),
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


def mark_done(status: str, summary: str, extra: dict | None = None) -> None:
    try:
        pid_path().unlink(missing_ok=True)
    except TypeError:
        if pid_path().exists():
            pid_path().unlink()
    final_progress = {}
    if progress_path().exists():
        try:
            final_progress = json.loads(progress_path().read_text(encoding="utf-8"))
        except Exception:
            final_progress = {}
    payload = {
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": now_iso(),
    }
    if extra:
        payload.update(extra)
    write_json(done_path(), payload)


def package_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return f"NOT_INSTALLED ({exc.__class__.__name__})"
    return getattr(mod, "__version__", "unknown")


def load_model() -> tuple[object, object, dict]:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    flash_available = importlib.util.find_spec("flash_attn") is not None
    requested_attn = "flash_attention_2" if flash_available else "sdpa"
    compile_enabled = False
    compile_error = None

    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=requested_attn,
        ).to("cuda:0").eval()
    except Exception as exc:
        compile_error = f"attn_fallback:{exc!r}"
        requested_attn = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=requested_attn,
        ).to("cuda:0").eval()

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            compile_enabled = True
        except Exception as exc:
            compile_error = repr(exc)

    torch.cuda.synchronize()
    meta = {
        "model_path": str(MODEL_PATH),
        "load_sec": round(time.time() - t0, 2),
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": getattr(tokenizer, "mask_token_id", None) or MASK_TOKEN_ID,
        "requested_attn_implementation": requested_attn,
        "resolved_attn_implementation": getattr(getattr(model, "config", None), "_attn_implementation", None) or requested_attn,
        "flash_attention_available": flash_available,
        "torch_compile_enabled": compile_enabled,
        "torch_compile_error": compile_error,
        "model_vram_mb": round(torch.cuda.memory_allocated(0) / 1024**2),
    }
    return model, tokenizer, meta


def left_pad(batch_tokens: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x in batch_tokens)
    input_ids = torch.full((len(batch_tokens), max_len), pad_token_id, dtype=torch.long, device="cuda:0")
    attention_mask = torch.zeros((len(batch_tokens), max_len), dtype=torch.long, device="cuda:0")
    for i, toks in enumerate(batch_tokens):
        input_ids[i, max_len - len(toks):] = torch.tensor(toks, dtype=torch.long, device="cuda:0")
        attention_mask[i, max_len - len(toks):] = 1
    return input_ids, attention_mask


def entropy_scores(model: object, tokenizer: object, prompts: list[str], batch_size: int = 4) -> list[float]:
    scores: list[float] = []
    for offset in range(0, len(prompts), batch_size):
        chunk = prompts[offset: offset + batch_size]
        token_lists = []
        for prompt in chunk:
            toks = tokenizer(prompt, truncation=True, max_length=PROMPT_MAX_TOKENS, add_special_tokens=True)["input_ids"]
            toks = toks + [MASK_TOKEN_ID] * 8
            token_lists.append(toks)
        input_ids, attention_mask = left_pad(token_lists, tokenizer.pad_token_id)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -8:, :]
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean(dim=-1)
        scores.extend(float(x) for x in entropy.detach().cpu())
        del input_ids, attention_mask, outputs, logits, probs, entropy
        torch.cuda.empty_cache()
    return scores


def safe_batch_probe(model: object, pad_token_id: int, total_len: int, lo: int = 1, hi: int = 32) -> tuple[int, int]:
    best = lo
    peak = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            gc.collect()
            torch.cuda.empty_cache()
            dummy = torch.full((mid, total_len), pad_token_id, dtype=torch.long, device="cuda:0")
            mask = torch.ones_like(dummy)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=dummy, attention_mask=mask)
            torch.cuda.synchronize()
            peak = max(peak, int(torch.cuda.max_memory_allocated(0) / 1024**2))
            best = mid
            lo = mid + 1
        except torch.cuda.OutOfMemoryError:
            hi = mid - 1
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    return best, peak


def prompt_for_gsm8k(row: dict) -> str:
    return row["question"].strip()


def prompt_for_mbpp(row: dict) -> str:
    return row["text"].strip()


def manifest_rows(dataset_name: str, dataset, prompt_builder, model: object, tokenizer: object) -> list[dict]:
    rng = random.Random(SEED if dataset_name == "gsm8k" else SEED + 1)
    candidate_indices = list(range(len(dataset)))
    rng.shuffle(candidate_indices)
    candidate_indices = candidate_indices[: min(PRESCAN_POOL, len(candidate_indices))]
    candidate_rows = [dataset[i] for i in candidate_indices]
    prompts = [prompt_builder(row) for row in candidate_rows]
    scores = entropy_scores(model, tokenizer, prompts, batch_size=4)

    enriched = []
    for local_idx, (row, prompt, score, source_idx) in enumerate(zip(candidate_rows, prompts, scores, candidate_indices)):
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=PROMPT_MAX_TOKENS, add_special_tokens=True)["input_ids"])
        enriched.append(
            {
                "source_index": int(source_idx),
                "prompt": prompt,
                "prompt_length": int(prompt_len),
                "observer_entropy_precomputed": round(float(score), 6),
                "row": row,
            }
        )

    enriched.sort(key=lambda item: item["observer_entropy_precomputed"], reverse=True)
    high = enriched[:25]
    remainder = enriched[25:]
    mid_cut = max(1, len(remainder) // 2)
    mid = remainder[:mid_cut]
    low = remainder[mid_cut:]
    selected = high + mid[:12] + low[:13]
    selected = selected[:50]

    def bucket(score: float) -> str:
        if selected and score >= sorted([x["observer_entropy_precomputed"] for x in selected], reverse=True)[24]:
            return "high_entropy"
        if selected and score <= sorted([x["observer_entropy_precomputed"] for x in selected])[12]:
            return "low_entropy"
        return "mid_entropy"

    rows = []
    for rank, item in enumerate(selected):
        row = item["row"]
        rows.append(
            {
                "sample_id": f"{dataset_name}_test_{item['source_index']:04d}",
                "dataset": dataset_name,
                "source_split": "test",
                "source_index": item["source_index"],
                "prompt": item["prompt"],
                "prompt_length": item["prompt_length"],
                "observer_entropy_precomputed": item["observer_entropy_precomputed"],
                "difficulty_bucket": bucket(item["observer_entropy_precomputed"]),
                "selection_reason": "top_entropy_prescan" if rank < 25 else "balanced_mid_low_prescan",
                "reference": row.get("answer") or row.get("code") or "",
                "test_list": row.get("test_list") or [],
                "challenge_test_list": row.get("challenge_test_list") or [],
            }
        )
    return rows


def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SETUP_DIR.mkdir(parents=True, exist_ok=True)
    pid_path().write_text(str(os.getpid()), encoding="utf-8")

    total_epochs = 4
    try:
        report_progress(1, total_epochs, {"phase": "env_check"})
        env_check = {
            "task_id": TASK_ID,
            "timestamp": now_iso(),
            "python": sys.version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "datasets": {
                "gsm8k": {"path": str(GSM8K_PATH), "exists": GSM8K_PATH.exists()},
                "mbpp": {"path": str(MBPP_PATH), "exists": MBPP_PATH.exists()},
            },
            "packages": {
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "datasets": package_version("datasets"),
                "accelerate": package_version("accelerate"),
                "flash_attn": package_version("flash_attn"),
            },
            "gpu": {
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "unavailable",
                "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2) if torch.cuda.is_available() else 0,
                "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            },
        }
        write_json(SETUP_DIR / "env_check.json", env_check)
        log("env_check.json written")

        report_progress(2, total_epochs, {"phase": "load_model"})
        model, tokenizer, runtime_meta = load_model()
        gsm8k = load_from_disk(str(GSM8K_PATH))
        mbpp = load_from_disk(str(MBPP_PATH))
        log("datasets and model loaded")

        report_progress(3, total_epochs, {"phase": "manifest"})
        gsm8k_rows = manifest_rows("gsm8k", gsm8k, prompt_for_gsm8k, model, tokenizer)
        mbpp_rows = manifest_rows("mbpp", mbpp, prompt_for_mbpp, model, tokenizer)
        sample_manifest = {
            "task_id": TASK_ID,
            "timestamp": now_iso(),
            "seed": SEED,
            "total_samples": len(gsm8k_rows) + len(mbpp_rows),
            "selection_policy": {
                "per_dataset_samples": 50,
                "prescan_pool": PRESCAN_POOL,
                "high_entropy_target_pct": 50,
                "remaining_mix": "balanced_mid_low",
            },
            "rows": gsm8k_rows + mbpp_rows,
        }
        write_json(SETUP_DIR / "sample_manifest.json", sample_manifest)
        log("sample_manifest.json written")

        report_progress(4, total_epochs, {"phase": "batch_probe"})
        prompt_lengths = [row["prompt_length"] for row in sample_manifest["rows"]]
        p50 = sorted(prompt_lengths)[len(prompt_lengths) // 2]
        p90 = sorted(prompt_lengths)[math.floor(len(prompt_lengths) * 0.9) - 1]
        probe_buckets = [
            {"name": "median_prompt", "prompt_tokens": int(p50)},
            {"name": "p90_prompt", "prompt_tokens": int(p90)},
            {"name": "max_prompt", "prompt_tokens": int(max(prompt_lengths))},
        ]
        batch_probe_rows = []
        for bucket in probe_buckets:
            safe_batch, peak_mb = safe_batch_probe(model, tokenizer.pad_token_id, min(PROMPT_MAX_TOKENS, bucket["prompt_tokens"]) + GEN_LENGTH)
            batch_probe_rows.append(
                {
                    "bucket": bucket["name"],
                    "prompt_tokens": bucket["prompt_tokens"],
                    "gen_tokens": GEN_LENGTH,
                    "safe_batch_size": int(max(1, math.floor(safe_batch * 0.9))),
                    "peak_vram_mb": peak_mb,
                    "gpu_id": 0,
                    "attention_backend": runtime_meta["resolved_attn_implementation"],
                    "compile_enabled": runtime_meta["torch_compile_enabled"],
                }
            )
        batch_probe = {
            "task_id": TASK_ID,
            "timestamp": now_iso(),
            "probe_start_batch": 32,
            "probe_method": "binary_search_down_on_oom",
            "rows": batch_probe_rows,
        }
        write_json(SETUP_DIR / "batch_probe.json", batch_probe)

        runtime_contract = {
            "task_id": TASK_ID,
            "timestamp": now_iso(),
            "model_family": "LLaDA-8B-Instruct",
            "model_path": str(MODEL_PATH),
            "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", str(MODEL_PATH)),
            "temperature": 1.0,
            "prompt_template_policy": "Frozen across all arms",
            "postprocess_policy": "Frozen across all arms",
            "left_padding": True,
            "prompt_max_tokens": PROMPT_MAX_TOKENS,
            "generation_tokens": GEN_LENGTH,
            "selection_manifest_path": "current/exp/pilot_evidence_closure_v1/setup/sample_manifest.json",
            "dataset_sources": {
                "gsm8k": str(GSM8K_PATH),
                "mbpp": str(MBPP_PATH),
            },
            "runtime": runtime_meta,
            "batch_probe_summary": {
                "min_safe_batch_size": min(row["safe_batch_size"] for row in batch_probe_rows),
                "max_safe_batch_size": max(row["safe_batch_size"] for row in batch_probe_rows),
            },
        }
        write_json(SETUP_DIR / "runtime_contract.json", runtime_contract)
        log("runtime_contract.json and batch_probe.json written")

        summary = (
            f"built 100-sample manifest (50 GSM8K / 50 MBPP), "
            f"attn={runtime_meta['resolved_attn_implementation']}, "
            f"compile={runtime_meta['torch_compile_enabled']}, "
            f"safe_batch_range={runtime_contract['batch_probe_summary']['min_safe_batch_size']}-"
            f"{runtime_contract['batch_probe_summary']['max_safe_batch_size']}"
        )
        mark_done("success", summary, extra={"artifact_dir": str(SETUP_DIR)})
        log(summary)
        return 0
    except Exception as exc:
        write_json(
            SETUP_DIR / "env_check.json",
            {
                "task_id": TASK_ID,
                "timestamp": now_iso(),
                "status": "failed",
                "error": repr(exc),
            },
        )
        log(f"FAILED: {exc!r}")
        mark_done("failed", f"{TASK_ID} failed: {exc!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
