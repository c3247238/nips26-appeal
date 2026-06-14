#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, env=env, text=True, capture_output=True)


def main() -> int:
    if len(sys.argv) != 14:
        print("usage: sidecar.py WORKSPACE MODE SSH_SERVER REMOTE_BASE REMOTE_ENV TASK_IDS_CSV SUP_POLL GPU_POLL THRESHOLD MAX_GPUS AGGR AGGR_PCT OWNER_ID", file=sys.stderr)
        return 2

    (
        workspace,
        mode,
        ssh_server,
        remote_base,
        remote_env,
        task_ids_csv,
        sup_poll,
        gpu_poll,
        threshold_mb,
        max_gpus,
        aggressive,
        aggressive_pct,
        owner_id,
    ) = sys.argv[1:15]

    workspace_path = Path(workspace)
    repo_root = Path("/Users/cwan0785/sibyl-system")
    state_dir = Path("/Users/cwan0785/sibyl-system/workspaces/dlm-improve/.sibyl")
    log_path = workspace_path / "exp" / "experiment_supervisor_sidecar.log"
    env = os.environ.copy()
    env["SIBYL_STATE_DIR"] = str(state_dir)

    last_wake_kind = ""

    while True:
        snap = run(
            [
                str(repo_root / ".venv" / "bin" / "python3"),
                "-m",
                "sibyl.cli",
                "experiment-supervisor-snapshot",
                workspace,
            ],
            env,
        )
        if snap.returncode != 0:
            log_path.write_text(f"snapshot failed:\n{snap.stderr}\n", encoding="utf-8")
            break

        data = json.loads(snap.stdout)
        exp = data.get("experiment_status", {})
        drift = data.get("drift", {})
        pending = int(exp.get("pending_count", 0) or 0)
        running = int(exp.get("running_count", 0) or 0)
        total = int(exp.get("total_tasks", 0) or 0)
        completed = int(exp.get("completed_count", 0) or 0)
        gpu_poll_age = exp.get("gpu_poll_age_sec")

        actions = ["snapshot"]
        recommendations: list[str] = []
        summary = f"后台监督中：{completed}/{total} 完成，running={running}，pending={pending}"

        if pending or running:
            dispatch = run(
                [
                    str(repo_root / ".venv" / "bin" / "python3"),
                    "-m",
                    "sibyl.cli",
                    "dispatch",
                    workspace,
                ],
                env,
            )
            if dispatch.returncode == 0:
                try:
                    payload = json.loads(dispatch.stdout)
                except json.JSONDecodeError:
                    payload = {"dispatch": [], "reason": "unparseable"}
                reason = payload.get("reason", "")
                if payload.get("dispatch"):
                    actions.append("dispatch")
                    summary += f"，已派发 {len(payload['dispatch'])} 个 assignment"
                elif reason:
                    actions.append(f"dispatch:{reason}")
            else:
                actions.append("dispatch_failed")
                recommendations.append("检查 dispatch 失败原因")

        if gpu_poll_age is None or int(gpu_poll_age) > int(gpu_poll):
            recommendations.append("需要主系统修复 detached supervisor 的 GPU 刷新通路（当前仅已完成首次 MCP 刷新）")
            if last_wake_kind != "needs_main_system":
                details = {
                    "ssh_server_arg": ssh_server,
                    "fallback_mode": "default_mcp_only",
                    "gpu_poll_age_sec": gpu_poll_age,
                    "task_ids_csv": task_ids_csv,
                    "remote_base": remote_base,
                    "remote_env": remote_env,
                    "gpu_poll_interval_sec": int(gpu_poll),
                    "reason": "detached supervisor cannot perform MCP-only GPU refresh; cs8000d alias missing in current MCP config",
                }
                notify = run(
                    [
                        str(repo_root / ".venv" / "bin" / "python3"),
                        "-m",
                        "sibyl.cli",
                        "experiment-supervisor-notify-main",
                        workspace,
                        "--owner",
                        owner_id,
                        "--kind",
                        "needs_main_system",
                        "--summary",
                        "后台监督可维持 snapshot/dispatch，但 GPU 刷新通路仍依赖主系统修复 cs8000d/MCP 别名漂移",
                        "--details-json",
                        json.dumps(details, ensure_ascii=False),
                        "--actions-json",
                        json.dumps(actions, ensure_ascii=False),
                        "--recommendations-json",
                        json.dumps(recommendations, ensure_ascii=False),
                        "--urgency",
                        "high",
                        "--requires-main-system",
                    ],
                    env,
                )
                actions.append("notify_main")
                summary += "；已唤醒主系统处理 GPU 刷新漂移"
                last_wake_kind = "needs_main_system" if notify.returncode == 0 else last_wake_kind

        hb = run(
            [
                str(repo_root / ".venv" / "bin" / "python3"),
                "-m",
                "sibyl.cli",
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
            ],
            env,
        )
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "at": time.time(),
                        "summary": summary,
                        "actions": actions,
                        "recommendations": recommendations,
                        "heartbeat_ok": hb.returncode == 0,
                        "gpu_poll_age_sec": gpu_poll_age,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if total and completed >= total and running == 0 and pending == 0:
            run(
                [
                    str(repo_root / ".venv" / "bin" / "python3"),
                    "-m",
                    "sibyl.cli",
                    "experiment-supervisor-release",
                    workspace,
                    "--owner",
                    owner_id,
                    "--status",
                    "idle",
                    "--summary",
                    "实验任务已完成，后台监督退出",
                ],
                env,
            )
            break

        time.sleep(int(sup_poll))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
