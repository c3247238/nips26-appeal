# dlm-improve Planner Overlay

- Treat `current/exp/results/pilot_summary.json` as the authoritative latest screening signal.
- Iteration-4 screening showed `cand_mgcd` and `cand_dsg` failed the sham-control gate. Do not schedule them into a full benchmark again unless the hypothesis or repair object has materially changed.
- New planning should prioritize either:
  - a clean pivot back to idea refinement, or
  - a cheap discriminative follow-up for a revised candidate such as BSR / a new repair object.
- For inference-heavy tasks on RTX PRO 6000, default to `max_batch_size_hint: auto-detect` and assume batched evaluation, not `batch_size=1`.
- Explicitly plan throughput improvements when relevant: left-padding batched prompts, `flash_attention_2` if compatible, `torch.compile` where stable, and multi-GPU sharding when multiple independent method arms can run in parallel.
