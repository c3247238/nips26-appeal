# dlm-improve Experiment Supervisor Overlay

- Flag under-utilization aggressively for this project: if an inference-heavy task is running with `batch_size=1` or obviously low VRAM usage, treat that as a configuration problem to repair, not a harmless default.
- Use `current/exp/results/pilot_summary.json` as the authority for whether a candidate deserves more budget. `cand_mgcd` and `cand_dsg` currently do not.
- Prefer relaunches that improve throughput first: batching, compatible attention optimization, `torch.compile`, and multi-GPU sharding across independent method arms.
- If monitor state and remote completion markers disagree, prefer the remote `_DONE` markers and wake the main system with the discrepancy only after preserving the evidence.
