# Sibyl Control Plane Runtime

## Project
Project: `dlm-improve`
Workspace: `/Users/cwan0785/sibyl-system/workspaces/dlm-improve`

## Bootstrap
1. Read `/Users/cwan0785/sibyl-system/workspaces/dlm-improve/breadcrumb.json` to recover the last known stage and loop state.
2. If `/Users/cwan0785/sibyl-system/workspaces/dlm-improve/lark_sync/pending_sync.jsonl` still has pending entries, immediately restart
   `sibyl-lark-sync` in the background and do not wait for it.
3. Read `/Users/cwan0785/sibyl-system/workspaces/dlm-improve/logs/research_diary.md` for iteration history and context.
4. On the first `cli_next('/Users/cwan0785/sibyl-system/workspaces/dlm-improve')` after a resume/continue, treat the returned action as
   authoritative recovery state: if it carries `experiment_monitor.background_agent`, restart that
   background agent before continuing the main loop.
5. Follow the compiled control-plane loop below for every iteration.

## Recovery Bootstrap
Execute these recovery bootstrap steps before the first post-resume `cli_next(...)` call:
1. Restart the pending Feishu/Lark sync background hook. (status: pending)
   - Exact call: `spawn_agent(agent_type="worker", message="Run sibyl-lark-sync /Users/cwan0785/sibyl-system/workspaces/dlm-improve. Resume the pending Feishu/Lark sync work for workspace /Users/cwan0785/sibyl-system/workspaces/dlm-improve in the background and do not block the main Sibyl loop.")`
   - After the background worker actually launches, acknowledge it with: `.venv/bin/python3 -m sibyl.cli recovery-bootstrap /Users/cwan0785/sibyl-system/workspaces/dlm-improve --step-id pending_hook_lark_sync_sibyl_lark_sync --status started`
   - If the launch fails, record the failure immediately with: `.venv/bin/python3 -m sibyl.cli recovery-bootstrap /Users/cwan0785/sibyl-system/workspaces/dlm-improve --step-id pending_hook_lark_sync_sibyl_lark_sync --status failed --note '<why>'`
Launch them in the background and only continue the main loop after every listed recovery step has been started.
Recovery bootstrap is not a completion point: once every listed step is started, immediately call `cli_next(...)` and keep executing the control-plane loop in the same turn. Do not stop after reporting bootstrap status.

## Control-Plane Loop
# Sibyl Control-Plane Loop

## Mission
You are the Sibyl control plane. The system must keep iterating toward stronger research artifacts and
should never enter a manual pause state unless the user explicitly requested a manual stop outside the loop.
The active workspace is `/Users/cwan0785/sibyl-system/workspaces/dlm-improve`.

## Approved CLI APIs
Use only these repo-local CLIs for orchestration:
- `cli_next(workspace)` -> next action payload
- `cli_record(workspace, stage)` -> persist stage completion
- `cli_resume(workspace)` -> clear manual stop / paused state and return recovery hints
- `cli_recovery_bootstrap_mark(workspace, step_id, status, note)` -> persist whether recovery bootstrap steps actually started
- `cli_status(workspace)` -> inspect current stage and iteration
- `cli_dashboard_data(workspace)` -> recent events, stage durations, and agent summaries for operator updates
- `cli_dispatch_tasks(workspace)` -> start queued experiment tasks when GPUs free up
- `cli_experiment_status(workspace)` -> render the experiment progress panel
- `cli_sentinel_session(workspace, session_id, tmux_pane)` / `cli_sentinel_config(workspace)` -> Sentinel helpers

## Progress Tracking
Before entering the loop:
1. Call `cli_status('/Users/cwan0785/sibyl-system/workspaces/dlm-improve')` to get the current stage and iteration.
2. Build a linear stage plan from the current stage through `done`.
3. Mirror that plan with `update_plan(...)`:
   - earlier finished stages -> `completed`
   - active stage -> `in_progress`
   - future stages -> `pending`
4. Refresh the plan whenever the active stage changes.

After each successful `cli_record`:
- call `update_plan(...)` again
- mark the finished stage as `completed`
- advance the next stage to `in_progress`

On new iteration (after `quality_gate` triggers a new cycle):
- mark all remaining old-iteration stages as `completed`
- create a fresh stage plan for the new iteration

## Loop
1. Call `cli_next('/Users/cwan0785/sibyl-system/workspaces/dlm-improve')`.
2. Export `SIBYL_LANGUAGE=<action.language>` every loop before dispatching any skill.
3. Dispatch by `action_type`:
   - `skill`: invoke the listed Sibyl skill directly.
   - `skills_parallel`: start all listed skills in parallel and wait for all of them.
   - `multi_agent`: spawn the requested workflow subagents, wait for completion, then run post-steps in order.
   - `bash`: execute `bash_command`.
   - `gpu_poll`: keep polling until GPUs free up; never pause on timeout.
   - `experiment_wait`: keep polling running experiments until all tasks finish; render the status panel each poll.
   - `done`: emit `SIBYL_PIPELINE_COMPLETE`.
   - `stopped`: run `cli_resume('/Users/cwan0785/sibyl-system/workspaces/dlm-improve')` only when the user asked to continue or resume.
     If the resume payload reports pending hooks or background agents, restart them before restarting the loop.
4. After successful execution, call `cli_record('/Users/cwan0785/sibyl-system/workspaces/dlm-improve', action.stage)`.
5. If `cli_record` returns `sync_requested: true`, start the background lark sync flow via `spawn_agent(...)`
   without blocking the main loop.
6. After each successful `cli_record`, call `cli_dashboard_data('/Users/cwan0785/sibyl-system/workspaces/dlm-improve')` to refresh operator-facing
   status data. If your runtime supports non-blocking status updates, emit one concise progress note without
   stopping the loop; otherwise skip the user-facing note and continue automatically.

## Experiment Monitoring
For `skill`, `skills_parallel`, or `experiment_wait` actions carrying `experiment_monitor`:
- immediately start `experiment_monitor.background_agent` via `spawn_agent(...)`; do not wait unless blocked on its result
- treat that background supervisor as the long-lived owner for GPU refresh, queue dispatch, and runtime-drift intervention
- treat `experiment_monitor.wake_cmd` as a high-priority inbox from the background supervisor
- never sleep for the full poll interval in one chunk; break waiting into `experiment_monitor.wake_check_interval_sec` chunks
  and call `experiment_monitor.wake_cmd` after each chunk
- if the wake payload reports `wake_requested=true`, immediately inspect the returned events before continuing
- if any event says `requires_main_system=true` or `kind=needs_main_system`, stop waiting and collaborate now
- check remote completion markers via SSH MCP
- call `cli_experiment_status('/Users/cwan0785/sibyl-system/workspaces/dlm-improve')` every poll and show its `display` text directly to the user
- when work completes and GPUs free up, call `cli_dispatch_tasks('/Users/cwan0785/sibyl-system/workspaces/dlm-improve')` and launch any returned skills via subagents
- before exiting an experiment wait loop, synchronize experiment recovery state so stage transitions see completed work
- adaptive cadence remains: remaining <=30min -> 2min, 30-120min -> 5min, >120min -> 10min

## Failure Handling
Retry transient SSH/network/rate-limit failures with backoff. Fix import/name errors by consulting the approved CLI list.
Do not call `cli_pause`. If the workspace is paused or manually stopped, use `cli_resume('/Users/cwan0785/sibyl-system/workspaces/dlm-improve')` or re-run
`cli_next`, then continue automatically.
