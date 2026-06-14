# Result Summary For Iteration 3

## Active framing

Iteration 3 no longer treats `CARD-84` as a winning controller method. The active paper object is an **audited negative case study** built from the completed pilot evidence bundle.

## Core readout

- `DNB-64`: overall `0.17`, GSM8K `0.30`, MBPP `0.04`
- `DNB-84`: overall `0.10`, GSM8K `0.18`, MBPP `0.02`
- `CARD-84`: overall `0.18`, GSM8K `0.32`, MBPP `0.04`
- `RAND-84`: overall `0.17`, GSM8K `0.30`, MBPP `0.04`

## Interpretation constraints

- `CARD-84` shows a localized GSM8K signal against compute-matched `DNB-84`.
- That signal does **not** cleanly separate from budget-matched `RAND-84`.
- Entropy can be described as a `risk marker`, not a validated targeting rule.
- The evidence scope is an audited slice with current-only artifacts.

## Decision gates

- `CARD-84` minus `DNB-84` on GSM8K `net_repaired`: `+7`
- `CARD-84` minus `RAND-84` on GSM8K `net_repaired`: `+1`
- `gsm8k_card_beats_rand84_by_2_net_repaired`: `False`
- `artifacts_joinable`: `True`

## Packaging outputs already available

- `analysis/packaging/validation_report.json`
- `analysis/packaging/claim_scope_map.json`
- `analysis/packaging/main_results_table.csv`
- `analysis/packaging/repair_harm_table.csv`
- `analysis/packaging/harm_profile_table.csv`
- `analysis/packaging/figure_specs.md`
- `analysis/packaging/followup_gaps.md`

## Working conclusion

The strongest contribution is not a positive controller claim. It is a reviewer-safe negative-case package showing that matched-compute improvement over `DNB-84` is insufficient once a stronger sham control is added.
