# dlm-improve Server Experimenter Overlay

- The server-side Codex prompt must enforce batched evaluation by default for this project; do not accept `batch_size=1` as a lazy fallback.
- Read `current/exp/results/pilot_summary.json` before spending new full-benchmark budget. `cand_mgcd` and `cand_dsg` currently require a pivot or a materially revised hypothesis before escalation.
- Prefer server-side throughput improvements that are easy to verify: batched left-padding, `flash_attention_2` when compatible, `torch.compile`, and multi-GPU sharding across independent methods.
- Write remote `_DONE` markers faithfully and treat them as the completion authority if local supervisor state drifts.
