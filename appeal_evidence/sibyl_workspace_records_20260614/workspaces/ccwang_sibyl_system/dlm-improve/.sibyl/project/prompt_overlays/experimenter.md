# dlm-improve Experimenter Overlay

- Before any long evaluation run, probe the largest safe batch or eval batch. Avoid `batch_size=1` unless the runtime proves batching impossible.
- Read `current/exp/results/pilot_summary.json` before launching new work. `cand_mgcd` and `cand_dsg` should not be escalated into a full benchmark without a materially revised hypothesis.
- Prefer throughput fixes early: batched left-padding for variable-length prompts, `flash_attention_2` when supported, `torch.compile` when stable, and multi-GPU sharding across independent method arms.
- When local monitor state drifts from reality, trust remote `_DONE` markers and remote result artifacts over stale local running-state.
