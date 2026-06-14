# C3C Safety Probe Pilot Summary

**Timestamp:** 2026-04-14T18:24:51.471248
**Model:** gpt2-small
**SAE:** cfg_L8_24k_narrow (gpt2-small-res-jb / blocks.8.hook_resid_pre)
**Mean absorption:** 0.044

## Dataset
- Harmful prompts: 20 (AdvBench-style)
- Benign prompts: 20 (OpenWebText-style)
- Total: 40

## Dense Probe Results
- AUC per fold: [1.0, 1.0, 1.0]
- AUC mean: 1.0000 ± 0.0000
- Has NaN: False

## Pass Criteria
- Logistic regression trains: True
- AUC > 0.50: True (AUC=1.0000)
- No NaN in CV scores: True
- **Overall: True**

## GO / NO-GO: **GO**

**Runtime:** 8.2s