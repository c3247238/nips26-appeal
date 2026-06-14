#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/Users/cwan0785/sibyl-system")
SSH_USER = os.environ.get("SIBYL_SUPERVISOR_SSH_USER", "ccwang")
SSH_HOST = os.environ.get("SIBYL_SUPERVISOR_SSH_HOST", "10.163.141.179")


def run(cmd: list[str], env: dict[str, str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def run_cli(env: dict[str, str], *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return run(
        [str(REPO_ROOT / ".venv" / "bin" / "python3"), "-m", "sibyl.cli", *args],
        env,
        timeout=timeout,
    )


def run_ssh(remote_cmd: str, env: dict[str, str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    target = f"{SSH_USER}@{SSH_HOST}"
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            target,
            remote_cmd,
        ],
        env,
        timeout=timeout,
    )


def parse_json_output(cp: subprocess.CompletedProcess[str], fallback: Any) -> Any:
    if cp.returncode != 0:
        return fallback
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        return fallback


def append_log(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def common_compile_failure(statuses: dict[str, Any]) -> bool:
    errors = []
    for task in statuses.values():
        done_payload = task.get("done_payload") or {}
        summary = str(done_payload.get("summary", "")) or str(task.get("error", ""))
        errors.append(summary)
    if not errors:
        return False
    return all("CUBLAS_STATUS_INVALID_VALUE" in err for err in errors)


def task_statuses(task_ids: list[str], remote_project_dir: str, env: dict[str, str]) -> dict[str, Any]:
    results_dir = f"{remote_project_dir}/exp/results"
    task_json = json.dumps(task_ids)
    remote_cmd = (
        "python3 - <<'PY'\n"
        "import json, os\n"
        f"base = {results_dir!r}\n"
        f"tasks = json.loads({task_json!r})\n"
        "payload = {}\n"
        "for task_id in tasks:\n"
        "    done_path = os.path.join(base, f'{task_id}_DONE')\n"
        "    progress_path = os.path.join(base, f'{task_id}_PROGRESS.json')\n"
        "    pid_path = os.path.join(base, f'{task_id}.pid')\n"
        "    log_path = os.path.join(base, f'{task_id}.launch.log')\n"
        "    item = {\n"
        "        'done_exists': os.path.exists(done_path),\n"
        "        'progress_exists': os.path.exists(progress_path),\n"
        "        'pid_exists': os.path.exists(pid_path),\n"
        "        'log_exists': os.path.exists(log_path),\n"
        "    }\n"
        "    if os.path.exists(done_path):\n"
        "        try:\n"
        "            item['done_payload'] = json.load(open(done_path, encoding='utf-8'))\n"
        "        except Exception as exc:\n"
        "            item['done_payload'] = {'read_error': repr(exc)}\n"
        "    if os.path.exists(progress_path):\n"
        "        try:\n"
        "            item['progress_payload'] = json.load(open(progress_path, encoding='utf-8'))\n"
        "        except Exception as exc:\n"
        "            item['progress_payload'] = {'read_error': repr(exc)}\n"
        "    if os.path.exists(log_path):\n"
        "        try:\n"
        "            lines = open(log_path, encoding='utf-8', errors='ignore').read().splitlines()\n"
        "            item['log_tail'] = lines[-60:]\n"
        "        except Exception as exc:\n"
        "            item['log_tail'] = [f'read_error: {exc!r}']\n"
        "    payload[task_id] = item\n"
        "print(json.dumps(payload, ensure_ascii=False))\n"
        "PY"
    )
    cp = run_ssh(remote_cmd, env, timeout=180)
    return parse_json_output(cp, {})


def gpu_poll(workspace: str, env: dict[str, str]) -> tuple[list[int], str]:
    cp = run_ssh(
        "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits",
        env,
        timeout=60,
    )
    raw = cp.stdout.strip()
    if cp.returncode == 0 and raw:
        record = run_cli(
            env,
            "record-gpu-poll",
            workspace,
            "--nvidia-smi-output",
            raw,
            "--source",
            "experiment_supervisor_sidecar",
            timeout=120,
        )
        parsed = parse_json_output(record, {})
        return list(parsed.get("free_gpus", [])), raw
    return [], raw


def summarize_failures(statuses: dict[str, Any]) -> dict[str, Any]:
    failures: dict[str, Any] = {}
    for task_id, info in statuses.items():
        done_payload = info.get("done_payload") or {}
        progress_payload = info.get("progress_payload") or {}
        metric = progress_payload.get("metric") or {}
        failures[task_id] = {
            "done_status": done_payload.get("status", ""),
            "done_summary": done_payload.get("summary", ""),
            "progress_phase": metric.get("phase", ""),
            "progress_error": metric.get("error", ""),
            "progress_updated_at": progress_payload.get("updated_at", ""),
        }
    return failures


def all_done(statuses: dict[str, Any], task_ids: list[str]) -> bool:
    return bool(task_ids) and all(bool(statuses.get(task_id, {}).get("done_exists")) for task_id in task_ids)


def any_failed(statuses: dict[str, Any]) -> bool:
    for info in statuses.values():
        done_payload = info.get("done_payload") or {}
        if str(done_payload.get("status", "")).lower() not in {"", "success", "completed"}:
            return True
        progress_payload = info.get("progress_payload") or {}
        metric = progress_payload.get("metric") or {}
        if str(metric.get("phase", "")).lower() == "failed":
            return True
    return False


def make_owner_id() -> str:
    return f"exp-supervisor-{int(time.time())}-{socket.gethostname()}-{random.randint(1000, 9999)}"


def main() -> int:
    if len(sys.argv) != 14:
        print(
            "usage: experiment_supervisor_sidecar.py WORKSPACE MODE SSH_SERVER REMOTE_BASE "
            "REMOTE_ENV TASK_IDS_CSV SUP_POLL GPU_POLL THRESHOLD MAX_GPUS AGGR AGGR_PCT OWNER_ID",
            file=sys.stderr,
        )
        return 2

    (
        workspace,
        mode,
        ssh_server,
        remote_base,
        remote_env,
        task_ids_csv,
        sup_poll,
        gpu_poll_interval,
        threshold_mb,
        max_gpus,
        aggressive,
        aggressive_pct,
        owner_id,
    ) = sys.argv[1:14]

    workspace_path = Path(workspace)
    log_path = workspace_path / "exp" / "experiment_supervisor_sidecar.log"
    env = os.environ.copy()
    env["SIBYL_STATE_DIR"] = str(Path("/Users/cwan0785/sibyl-system/workspaces/dlm-improve/.sibyl"))
    env["SIBYL_LANGUAGE"] = "zh"
    remote_project_dir = f"{remote_base}/projects/dlm-improve"
    task_ids = [task_id.strip() for task_id in task_ids_csv.split(",") if task_id.strip()]
    wake_sent = False

    claim = run_cli(
        env,
        "experiment-supervisor-claim",
        workspace,
        "--owner",
        owner_id,
        "--stale-after",
        "900",
    )
    claim_payload = parse_json_output(claim, {})
    if not claim_payload.get("should_start"):
        append_log(
            log_path,
            {
                "ts": time.time(),
                "event": "claim_rejected",
                "payload": claim_payload,
            },
        )
        return 0

    while True:
        snapshot = parse_json_output(
            run_cli(env, "experiment-supervisor-snapshot", workspace, timeout=120),
            {},
        )
        experiment_status = snapshot.get("experiment_status", {})
        running_count = int(experiment_status.get("running_count", 0) or 0)
        pending_count = int(experiment_status.get("pending_count", 0) or 0)
        total_tasks = int(experiment_status.get("total_tasks", 0) or 0)
        completed_count = int(experiment_status.get("completed_count", 0) or 0)

        free_gpus, raw_gpu = gpu_poll(workspace, env)
        statuses = task_statuses(task_ids, remote_project_dir, env)
        target_done = all_done(statuses, task_ids)
        target_failed = target_done and any_failed(statuses)

        actions = ["snapshot", "gpu_poll"]
        if target_done:
            actions.append("remote_done_confirmed")

        recommendations: list[str] = []
        summary = f"后台监督中：{completed_count}/{total_tasks} 完成，running={running_count}，pending={pending_count}"

        if target_done and not wake_sent:
            failures = summarize_failures(statuses)
            if target_failed:
                recommendations = [
                    "主系统应立即停止继续派发同一 runtime contract 下的后续 pilot 臂",
                    "优先修复 pilot_standard_arm 的 torch.compile 路径，至少增加 eager fallback 后再重跑",
                    "重新核对 gpu_count=2 / DataParallel 假设；当前标准臂实现只显式使用单个 cuda device",
                ]
                if common_compile_failure(statuses):
                    recommendations[1] = "两个基线均在 torch.compile 路径触发同一 CUBLAS 错误；应先禁用 compile 或加入 eager fallback 再重跑"
                notify = run_cli(
                    env,
                    "experiment-supervisor-notify-main",
                    workspace,
                    "--owner",
                    owner_id,
                    "--kind",
                    "needs_main_system",
                    "--summary",
                    "run_dnb64_baseline 与 run_dnb84_matched_compute 均已结束但同样失败；需主系统立即修复 runtime/计划漂移",
                    "--details-json",
                    json.dumps(
                        {
                            "mode": mode,
                            "task_ids": task_ids,
                            "ssh_server_arg": ssh_server,
                            "effective_ssh_target": f"{SSH_USER}@{SSH_HOST}",
                            "failures": failures,
                            "free_gpus": free_gpus,
                            "gpu_poll_threshold_mb": int(threshold_mb),
                            "gpu_poll_interval_sec": int(gpu_poll_interval),
                            "common_compile_failure": common_compile_failure(statuses),
                            "evidence_path": str(log_path),
                        },
                        ensure_ascii=False,
                    ),
                    "--actions-json",
                    json.dumps(actions, ensure_ascii=False),
                    "--recommendations-json",
                    json.dumps(recommendations, ensure_ascii=False),
                    "--urgency",
                    "critical",
                    "--requires-main-system",
                    timeout=120,
                )
                actions.append("notify_main_failure")
                summary = "两个 DNB pilot 基线已确认失败；已唤醒主系统处理 compile/CUDA 与资源计划漂移"
                wake_sent = notify.returncode == 0
            else:
                recommendations = ["主系统应尽快重新评估 experiment_wait 并派发后续排队任务"]
                notify = run_cli(
                    env,
                    "experiment-supervisor-notify-main",
                    workspace,
                    "--owner",
                    owner_id,
                    "--kind",
                    "resolution",
                    "--summary",
                    "run_dnb64_baseline 与 run_dnb84_matched_compute 已完成；主系统可立即继续下一批实验",
                    "--details-json",
                    json.dumps(
                        {
                            "task_ids": task_ids,
                            "free_gpus": free_gpus,
                            "effective_ssh_target": f"{SSH_USER}@{SSH_HOST}",
                        },
                        ensure_ascii=False,
                    ),
                    "--actions-json",
                    json.dumps(actions, ensure_ascii=False),
                    "--recommendations-json",
                    json.dumps(recommendations, ensure_ascii=False),
                    "--urgency",
                    "high",
                    timeout=120,
                )
                actions.append("notify_main_resolution")
                summary = "两个目标任务已确认完成；已提醒主系统继续派发后续实验"
                wake_sent = notify.returncode == 0

        heartbeat = run_cli(
            env,
            "experiment-supervisor-heartbeat",
            workspace,
            "--owner",
            owner_id,
            "--summary",
            summary,
            "--actions-json",
            json.dumps(actions, ensure_ascii=False),
            "--recommendations-json",
            json.dumps(recommendations, ensure_ascii=False),
            timeout=120,
        )

        append_log(
            log_path,
            {
                "ts": time.time(),
                "summary": summary,
                "actions": actions,
                "recommendations": recommendations,
                "heartbeat_ok": heartbeat.returncode == 0,
                "target_done": target_done,
                "target_failed": target_failed,
                "free_gpus": free_gpus,
                "raw_gpu": raw_gpu,
            },
        )

        if pending_count == 0 and running_count == 0:
            run_cli(
                env,
                "experiment-supervisor-release",
                workspace,
                "--owner",
                owner_id,
                "--status",
                "idle",
                "--summary",
                "无运行或排队实验，后台监督退出",
                timeout=120,
            )
            return 0

        time.sleep(int(sup_poll))


if __name__ == "__main__":
    raise SystemExit(main())
