# Pilot Claim Gate Summary

- 结论分支: `cand_negative_audit_pivot`
- `CARD-84` overall accuracy: 0.18
- `RAND-84` overall accuracy: 0.17
- GSM8K `card84_vs_dnb84` net repaired: 7
- GSM8K `card84_vs_rand84` net repaired: 1
- MBPP harm delta (`CARD-84` minus `DNB-84`, both against `DNB-64`): -1

## Main Readout

- `DNB-64`: accuracy=0.17, gsm8k=0.3, mbpp=0.04
- `DNB-84`: accuracy=0.1, gsm8k=0.18, mbpp=0.02
- `CARD-84`: accuracy=0.18, gsm8k=0.32, mbpp=0.04
- `RAND-84`: accuracy=0.17, gsm8k=0.3, mbpp=0.04

## Gate Details

- `gsm8k_card_beats_dnb84_by_2_net_repaired`: True
- `gsm8k_card_beats_rand84_by_2_net_repaired`: False
- `mbpp_card_harm_not_worse_than_dnb84_plus_3`: True
- `artifacts_joinable`: True

## Artifacts

- `per_sample_audit.csv`: exp/pilot_evidence_closure_v1/analysis/per_sample_audit.csv
- `transition_matrix.csv`: exp/pilot_evidence_closure_v1/analysis/transition_matrix.csv
- `decision.json`: exp/pilot_evidence_closure_v1/analysis/decision.json
- `claim_to_asset_map.json`: exp/pilot_evidence_closure_v1/analysis/claim_to_asset_map.json
- `code_failure_modes.md`: exp/pilot_evidence_closure_v1/analysis/code_failure_modes.md
- `runtime_contract.json`: exp/pilot_evidence_closure_v1/setup/runtime_contract.json

## Runtime Contract

- attention backend: `unknown`
- flash attention available: `None`
- compile enabled in production arms: `False`