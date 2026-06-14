from __future__ import annotations

import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENT_STAGES = {"pilot_experiments", "experiment_cycle"}


def _repo_root() -> Path:
    env_root = os.environ.get("SIBYL_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "sibyl" / "cli.py").exists():
            return parent
    raise RuntimeError("Unable to locate Sibyl repo root")


def _workspace_root(workspace_path: Path) -> Path:
    return workspace_path.parent if workspace_path.name == "current" else workspace_path


def _project_name(workspace_path: Path) -> str:
    return _workspace_root(workspace_path).name


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def _run_json(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    proc = _run(cmd, cwd=cwd, env=env)
    stdout = proc.stdout.strip()
    if not stdout:
        return {}
    return json.loads(stdout)


def _run_ssh(ssh_target: str, remote_cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            ssh_target,
            remote_cmd,
        ],
        check=check,
    )


def _resolve_ssh_target(raw_target: str) -> str:
    candidates = [raw_target]
    if raw_target == "cs8000d":
        candidates.extend(["CityU-ccwang", "ccwang@10.163.141.179"])
    for candidate in candidates:
        proc = _run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                candidate,
                "hostname",
            ],
            check=False,
        )
        if proc.returncode == 0:
            return candidate
    return raw_target


def _record_gpu_poll(
    repo_root: Path,
    env: dict[str, str],
    workspace_path: Path,
    ssh_target: str,
) -> tuple[list[int], str]:
    proc = _run_ssh(
        ssh_target,
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits",
        check=False,
    )
    raw_output = proc.stdout.strip()
    if not raw_output:
        return [], ""
    payload = _run_json(
        [
            sys.executable,
            "-m",
            "sibyl.cli",
            "record-gpu-poll",
            str(workspace_path),
            "--nvidia-smi-output",
            raw_output,
            "--source",
            "experiment_supervisor_monitor",
        ],
        cwd=repo_root,
        env=env,
    )
    return list(payload.get("free_gpus", [])), raw_output


def _remote_task_state(ssh_target: str, remote_project_dir: str, task_id: str) -> dict:
    remote_cmd = f"""
set -e
cd {shlex.quote(remote_project_dir)}
done_path=exp/results/{task_id}_DONE
pid_path=exp/results/{task_id}.pid
progress_path=exp/results/{task_id}_PROGRESS.json
if [ -f "$done_path" ]; then
  echo "__DONE__"
  cat "$done_path"
  exit 0
fi
if [ -f "$pid_path" ]; then
  pid=$(cat "$pid_path")
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "__ALIVE__:$pid"
    if [ -f "$progress_path" ]; then
      cat "$progress_path"
    fi
    exit 0
  fi
  echo "__DEAD__:$pid"
  exit 0
fi
echo "__NO_PID__"
"""
    proc = _run_ssh(ssh_target, remote_cmd, check=False)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {"state": "unknown", "raw": proc.stdout}
    header = lines[0].strip()
    if header == "__DONE__":
        payload = {}
        if len(lines) > 1:
            try:
                payload = json.loads("\n".join(lines[1:]))
            except json.JSONDecodeError:
                payload = {}
        return {"state": "done", "payload": payload}
    if header.startswith("__ALIVE__:"):
        progress = {}
        if len(lines) > 1:
            try:
                progress = json.loads("\n".join(lines[1:]))
            except json.JSONDecodeError:
                progress = {}
        return {"state": "alive", "pid": header.split(":", 1)[1], "progress": progress}
    if header.startswith("__DEAD__:"):
        return {"state": "dead", "pid": header.split(":", 1)[1]}
    if header == "__NO_PID__":
        return {"state": "no_pid"}
    return {"state": "unknown", "raw": proc.stdout}


def _maybe_upload_remote_script(
    workspace_path: Path,
    ssh_target: str,
    remote_project_dir: str,
    task_id: str,
) -> bool:
    local_script = workspace_path / "exp" / "code" / f"{task_id}.py"
    if not local_script.exists():
        return False
    remote_script = f"{remote_project_dir}/exp/code/{task_id}.py"
    check_proc = _run_ssh(
        ssh_target,
        f"test -f {shlex.quote(remote_script)}",
        check=False,
    )
    if check_proc.returncode == 0:
        return True
    _run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            str(local_script),
            f"{ssh_target}:{remote_script}",
        ],
        check=False,
    )
    verify = _run_ssh(
        ssh_target,
        f"test -f {shlex.quote(remote_script)}",
        check=False,
    )
    return verify.returncode == 0


def _launch_task(
    ssh_target: str,
    remote_project_dir: str,
    remote_env_cmd: str,
    task_id: str,
    gpu_ids: list[int],
) -> bool:
    remote_script = f"exp/code/{task_id}.py"
    gpu_csv = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    remote_cmd = (
        f"cd {shlex.quote(remote_project_dir)} && "
        f"test -f {shlex.quote(remote_script)} && "
        f"nohup bash -lc {shlex.quote(f'cd {remote_project_dir} && export CUDA_VISIBLE_DEVICES={gpu_csv} && {remote_env_cmd} python {remote_script} --workspace-root {remote_project_dir}')} "
        f"> exp/results/{task_id}.log 2>&1 < /dev/null & sleep 2 && "
        f"test -f exp/results/{task_id}.pid"
    )
    proc = _run_ssh(ssh_target, remote_cmd, check=False)
    return proc.returncode == 0


def _notify_main(
    repo_root: Path,
    env: dict[str, str],
    workspace_path: Path,
    owner_id: str,
    *,
    kind: str,
    summary: str,
    details: dict,
    actions: list[str],
    recommendations: list[str],
    requires_main: bool = False,
    urgency: str = "high",
) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "sibyl.cli",
            "experiment-supervisor-notify-main",
            str(workspace_path),
            "--owner",
            owner_id,
            "--kind",
            kind,
            "--summary",
            summary,
            "--details-json",
            json.dumps(details, ensure_ascii=False),
            "--actions-json",
            json.dumps(actions, ensure_ascii=False),
            "--recommendations-json",
            json.dumps(recommendations, ensure_ascii=False),
            "--urgency",
            urgency,
            *([] if not requires_main else ["--requires-main-system"]),
        ],
        cwd=repo_root,
        env=env,
        check=False,
    )


def _heartbeat(
    repo_root: Path,
    env: dict[str, str],
    workspace_path: Path,
    owner_id: str,
    summary: str,
    actions: list[str],
    recommendations: list[str],
) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "sibyl.cli",
            "experiment-supervisor-heartbeat",
            str(workspace_path),
            "--owner",
            owner_id,
            "--summary",
            summary,
            "--actions-json",
            json.dumps(actions, ensure_ascii=False),
            "--recommendations-json",
            json.dumps(recommendations, ensure_ascii=False),
        ],
        cwd=repo_root,
        env=env,
        check=False,
    )


def main(argv: list[str]) -> int:
    if len(argv) != 12:
        raise SystemExit(
            "usage: experiment_supervisor_monitor.py <workspace> <mode> <ssh_server> "
            "<remote_base> <remote_env_cmd> <task_ids_csv> <supervisor_poll_sec> "
            "<gpu_poll_sec> <gpu_free_threshold_mb> <max_gpus> <aggressive_mode> "
            "<aggressive_threshold_pct>"
        )

    workspace_path = Path(argv[0]).expanduser()
    mode = argv[1]
    raw_ssh_target = argv[2]
    remote_base = argv[3]
    remote_env_cmd = argv[4]
    task_ids_csv = argv[5]
    supervisor_poll_sec = int(argv[6])
    _gpu_poll_sec = int(argv[7])
    _gpu_free_threshold_mb = int(argv[8])
    _max_gpus = int(argv[9])
    _aggressive_mode = argv[10].lower() == "true"
    _aggressive_threshold_pct = int(argv[11])

    repo_root = _repo_root()
    os.chdir(repo_root)
    env = os.environ.copy()
    env.setdefault("SIBYL_ROOT", str(repo_root))
    env.setdefault("SIBYL_LANGUAGE", "zh")
    env.setdefault("SIBYL_STATE_DIR", str(workspace_path / ".sibyl_runtime_state"))
    Path(env["SIBYL_STATE_DIR"]).mkdir(parents=True, exist_ok=True)

    ssh_target = _resolve_ssh_target(raw_ssh_target)
    project = _project_name(workspace_path)
    remote_project_dir = f"{remote_base}/projects/{project}"
    task_ids = [task_id for task_id in task_ids_csv.split(",") if task_id]
    owner_id = f"exp-supervisor-{int(time.time())}-{random.randint(1000, 9999)}"

    claim = _run_json(
        [
            sys.executable,
            "-m",
            "sibyl.cli",
            "experiment-supervisor-claim",
            str(workspace_path),
            "--owner",
            owner_id,
            "--stale-after",
            "900",
        ],
        cwd=repo_root,
        env=env,
    )
    if not claim.get("should_start", False):
        return 0

    try:
        while True:
            snapshot = _run_json(
                [
                    sys.executable,
                    "-m",
                    "sibyl.cli",
                    "experiment-supervisor-snapshot",
                    str(workspace_path),
                ],
                cwd=repo_root,
                env=env,
            )
            stage = str(snapshot.get("stage", "")).strip()
            experiment_status = snapshot.get("experiment_status", {})
            running_tasks = list(experiment_status.get("running_tasks", []))
            pending_count = int(experiment_status.get("pending_count", 0) or 0)
            remaining_work = bool(running_tasks or pending_count)

            if stage not in EXPERIMENT_STAGES and not remaining_work:
                break

            free_gpus, _ = _record_gpu_poll(repo_root, env, workspace_path, ssh_target)
            actions: list[str] = ["snapshot", f"mode={mode}", f"ssh={ssh_target}"]
            recommendations: list[str] = []

            for task in running_tasks:
                task_id = str(task.get("task_id", "")).strip()
                if not task_id:
                    continue
                state = _remote_task_state(ssh_target, remote_project_dir, task_id)
                if state.get("state") == "alive":
                    actions.append(f"{task_id}:alive")
                    continue
                if state.get("state") == "done":
                    actions.append(f"{task_id}:done")
                    continue

                actions.append(f"{task_id}:{state.get('state', 'unknown')}")
                _run(
                    [
                        sys.executable,
                        "-m",
                        "sibyl.cli",
                        "requeue-experiment-task",
                        str(workspace_path),
                        task_id,
                        "--reason",
                        f"runtime_drift_{state.get('state', 'unknown')}",
                    ],
                    cwd=repo_root,
                    env=env,
                    check=False,
                )

                launch_ok = False
                if _maybe_upload_remote_script(workspace_path, ssh_target, remote_project_dir, task_id):
                    launch_ok = _launch_task(
                        ssh_target,
                        remote_project_dir,
                        remote_env_cmd,
                        task_id,
                        list(task.get("gpu_ids", [])),
                    )
                if launch_ok:
                    actions.append(f"{task_id}:relaunch_ok")
                    _notify_main(
                        repo_root,
                        env,
                        workspace_path,
                        owner_id,
                        kind="resolution",
                        summary=f"{task_id} 运行状态漂移已修复并重启",
                        details={"task_id": task_id, "remote_state": state.get("state", "unknown")},
                        actions=["requeue-experiment-task", "remote relaunch"],
                        recommendations=["主系统尽快 drain wake 并刷新 experiment_wait 状态"],
                    )
                else:
                    recommendations.append(f"需要主系统接管重启 {task_id}")
                    _notify_main(
                        repo_root,
                        env,
                        workspace_path,
                        owner_id,
                        kind="needs_main_system",
                        summary=f"{task_id} 运行状态漂移，后台无法自动重启",
                        details={"task_id": task_id, "remote_state": state.get("state", "unknown")},
                        actions=["requeue-experiment-task"],
                        recommendations=["调用 sibyl-experimenter 重新派发该任务"],
                        requires_main=True,
                        urgency="critical",
                    )

            if pending_count and free_gpus:
                dispatch = _run_json(
                    [
                        sys.executable,
                        "-m",
                        "sibyl.cli",
                        "dispatch",
                        str(workspace_path),
                    ],
                    cwd=repo_root,
                    env=env,
                )
                if dispatch.get("dispatch"):
                    actions.append("dispatch")
                    for assignment in dispatch.get("dispatch", []):
                        task_batch = assignment.get("task_ids", [])
                        gpu_ids = assignment.get("gpu_ids", [])
                        for task_id in task_batch:
                            launch_ok = False
                            if _maybe_upload_remote_script(workspace_path, ssh_target, remote_project_dir, task_id):
                                launch_ok = _launch_task(
                                    ssh_target,
                                    remote_project_dir,
                                    remote_env_cmd,
                                    task_id,
                                    list(gpu_ids),
                                )
                            if launch_ok:
                                actions.append(f"{task_id}:launched")
                            else:
                                recommendations.append(f"需要主系统派发 {task_id}")
                                _notify_main(
                                    repo_root,
                                    env,
                                    workspace_path,
                                    owner_id,
                                    kind="needs_main_system",
                                    summary=f"{task_id} 已进入调度队列，但后台无法自动启动",
                                    details={"task_id": task_id, "gpu_ids": gpu_ids},
                                    actions=["dispatch"],
                                    recommendations=["调用 sibyl-experimenter / sibyl-server-experimenter 启动任务"],
                                    requires_main=True,
                                    urgency="high",
                                )

            summary = (
                f"stage={stage or 'unknown'} running={len(running_tasks)} "
                f"pending={pending_count} free_gpus={free_gpus}"
            )
            _heartbeat(
                repo_root,
                env,
                workspace_path,
                owner_id,
                summary,
                actions,
                recommendations,
            )
            time.sleep(max(30, supervisor_poll_sec))
    finally:
        _run(
            [
                sys.executable,
                "-m",
                "sibyl.cli",
                "experiment-supervisor-release",
                str(workspace_path),
                "--owner",
                owner_id,
                "--status",
                "idle",
                "--summary",
                "后台监督退出",
            ],
            cwd=repo_root,
            env=env,
            check=False,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
