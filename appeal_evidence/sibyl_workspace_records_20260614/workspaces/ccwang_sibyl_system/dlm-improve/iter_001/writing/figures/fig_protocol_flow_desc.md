# Protocol Flow: Honest Compute, Observer-Controller Mismatch, and Task-Dependent Failure

## Purpose
Describe a single flow diagram that explains the paper's protocol rather than any one controller. The figure should make three layers explicit:

1. runtime-fair comparison on GSM8K;
2. signal auditing that separates observer quality from controller gain;
3. boundary testing on HumanEval to expose task-dependent failure.

## Layout
Use a left-to-right, three-stage flow on a full-width canvas.

### Stage 1: Honest-compute shortlist
- Box title: `Matched-compute reasoning shortlist`
- Show the six GSM8K methods from `diag_compute_curve_gsm8k.json`:
  - `Standard-64`
  - `DNB-84`
  - `Prophet-64`
  - `CORE-proxy-64`
  - `Entropy-Revise-64+3`
  - `TIGER-Instability-64+3`
- Under the box, list the protocol metadata that travel with every point:
  - `accuracy`
  - `nominal_nfe`
  - `actual_nfe`
  - `latency_sec`
  - `tokens_per_sec`
  - `batch_size`
  - `attention_backend`
  - `compile_enabled`
- Data-backed annotation:
  - `CORE-proxy-64`: nominal NFE `64`, actual NFE `69`, latency `482.953s`, batch size `1`, compile `false`
  - `Entropy-Revise-64+3`: nominal NFE `67`, actual NFE `68`, latency `210.67s`, batch size `32`, compile `false`
  - `Standard-64`: actual NFE `64`, latency `157.04s`, batch size `115`, compile `true`
- Callout text: `Headline step counts are not enough; one explicit reorder appears when actual compute is used.`

### Arrow 1
- Label: `Audit comparisons under actual compute`
- Small note below arrow: `max nominal-vs-actual compute gap = 7.81%`

### Stage 2: Observer vs controller audit
- Box title: `Signal audit`
- Split this box into two columns:
  - left column = `Observer quality`
  - right column = `Controller gain`
- Populate from `diag_signal_gap_audit.json`:
  - `calibration`: diagnostic score `0.6225`, control effectiveness `0.00`
  - `entropy`: diagnostic score `0.4140`, control effectiveness `0.00`
  - `instability`: diagnostic score `0.0555`, control effectiveness `0.01`
- Add a small footer row with the screening results:
  - `Random revise = 0.37`
  - `Entropy revise = 0.37`
  - `Instability revise = 0.38`
- Callout text: `The best observer is not the best controller under the tested policies.`

### Arrow 2
- Label: `Stress-test the controller story on a structured-output task`

### Stage 3: Code boundary
- Box title: `HumanEval boundary test`
- Show three method slots from `diag_humaneval_guard_boundary.json`:
  - `Standard`
  - `Entropy/TIGER Ungated Revision`
  - `Gated TIGER`
- For each slot, display three numbers:
  - `pass@1`
  - `syntax_failure_rate`
  - `runtime_failure_rate`
- Data-backed values:
  - `Standard`: `pass@1 0.06`, `syntax 0.46`, `runtime 0.48`
  - `Ungated`: `pass@1 0.02`, `syntax 0.48`, `runtime 0.50`
  - `Gated TIGER`: `pass@1 0.04`, `syntax 0.28`, `runtime 0.68`
- Add a guard badge near `Gated TIGER`:
  - `gate_open_rate = 0.14`
  - `syntax_guard_avg_ms = 0.089`
- Callout text: `Syntax repair is shallow; execution success does not recover.`

## Visual emphasis
- Use neutral gray-blue styling for protocol boxes and reserve accent colors for the three claims:
  - honest compute = red highlight
  - observer/controller mismatch = green-blue split
  - code boundary failure = orange/red
- Keep the arrows thick and explanatory; this figure is a protocol map, not a method diagram.
- Prefer short numeric annotations instead of prose-heavy paragraphs inside the boxes.

## Caption intent
`Our evaluation protocol tracks runtime-fair metadata, audits signal quality separately from control gain, and then tests whether any apparent revision benefit survives a structured-output boundary task.`
