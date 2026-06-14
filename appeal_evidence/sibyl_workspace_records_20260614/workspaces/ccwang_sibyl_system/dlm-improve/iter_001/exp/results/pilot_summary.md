# DCD Optional Pilot Summary

- Task: `baseline_dcd_optional`
- Verdict: `NO_GO`
- Reason: the low-cost `DCD-lite-64` pilot completed cleanly, but accuracy was only `0.26` on the 100-sample GSM8K pilot, well below the already reproduced `Standard-64` / `DNB-84` pilot accuracy of `0.36`.
- Compute profile: `actual_nfe=65.0`, `latency_sec=157.41`, `tokens_per_sec=162.63`, `batch_size=57`, `attention_backend=eager`
- Decision: keep the result as appendix/concurrent-work evidence, but do not spend more engineering time on a faithful DCD reimplementation unless the paper narrative later requires a stricter head-to-head baseline.
