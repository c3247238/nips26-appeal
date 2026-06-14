# Paper Outline: A Stronger Sham Control Rewrites a Small-Gain Story in Training-Free DLM Revision

## Title Options

1. **A Stronger Sham Control Rewrites a Small-Gain Story in Training-Free DLM Revision**
2. **An Audited Negative Case for Entropy-Guided Revision in Diffusion Language Models**
3. **When a Localized Revision Signal Fails a Stronger Sham Control: An Audited Negative Case in DLMs**

**Recommended title**: Option 1. It foregrounds the paper object, stays faithful to the audited slice, and does not overclaim a new method or protocol.

## One-Sentence Thesis

This paper is an audited negative case study showing that a localized GSM8K repair signal for entropy-guided revision survives a compute-matched baseline but fails to cleanly separate from a budget-matched sham control, so the most credible contribution is not a winning controller but a minimal audit template for interpreting small gains in training-free DLM revision.

## Positioning and Guardrails

- Frame the paper as an **audited negative case study with a minimal auditable template**, not as a new decoding method paper.
- Treat `CARD-84` as the audited intervention, not as the winning inference method.
- Describe entropy as a **risk marker**, not as a validated targeting rule.
- State scope limits early: this is an **audited slice with current-only artifacts**, not a broad benchmark-generalization claim.
- Treat MBPP as a **harm profile / failure-localization boundary**, not as a mechanism proof.
- Keep runtime language modest: the logged eager runtime path is a disclosure object, not a systems contribution.

## Section 1: Introduction

### Key content

- Motivate the problem: training-free DLM revision often produces small, tempting gains that are easy to over-interpret when the control structure is weak.
- Introduce the paper’s central claim: a compute-matched active control is not enough when the observed signal can also be mimicked by a budget-matched sham control.
- Preview the paper object and downscope:
  - this is an audited negative case, not a positive controller paper;
  - the supporting contribution is a minimal audit template;
  - the main lesson is interpretive, not algorithmic.

### Evidence to foreshadow

- `CARD-84` improves over `DNB-84` on GSM8K (`net_repaired = +7`).
- `CARD-84` does not cleanly separate from `RAND-84` on GSM8K (`net_repaired = +1`).
- MBPP does not separate `CARD-84` from `RAND-84` (`0.04` vs `0.04`), reinforcing a harm-profile interpretation rather than a success story.

### Transition

Move from the field-level risk of over-interpreting small gains to the audit structure needed to interpret them credibly.

## Section 2: Audited Slice and Minimal Audit Template

### Key content

- Define the audited slice:
  - 100 total examples;
  - 4 arms: `DNB-64`, `DNB-84`, `CARD-84`, `RAND-84`;
  - current-only, joinable artifact bundle.
- Explain the role of each arm:
  - `DNB-64` as the lower-compute reference;
  - `DNB-84` as the compute-matched active control;
  - `CARD-84` as the audited entropy-guided revision arm;
  - `RAND-84` as the budget-matched sham control.
- Present the minimal audit template:
  - matched-compute control;
  - sham control;
  - sample-level audit;
  - current-only artifact closure.

### Evidence to include

- `validation_report.json`
- `claim_to_asset_map.json`
- `per_sample_audit.csv`
- `transition_matrix.csv`

### Transition

Once the audit template is defined, the paper can present the one result that initially looks positive and then show how the stronger control changes its interpretation.

## Section 3: Main Audited Result on GSM8K

### Key content

- Present the localized GSM8K signal as the main empirical fact.
- Show why `CARD-84 > DNB-84` alone would tempt a positive interpretation.
- Immediately bring in `RAND-84` to show why that positive interpretation does not survive the stronger control.

### Evidence to include

- From `main_results_table.csv`:
  - `DNB-84` GSM8K accuracy = `0.18`
  - `CARD-84` GSM8K accuracy = `0.32`
  - `RAND-84` GSM8K accuracy = `0.30`
- From `repair_harm_table.csv`:
  - `card84_vs_dnb84,gsm8k -> net_repaired = 7`
  - `card84_vs_rand84,gsm8k -> net_repaired = 1`

### Main argument

The compute-matched active control confirms a localized signal, but the sham control prevents that signal from being written up as validated controller efficacy.

### Transition

After showing that the signal exists but cannot be upgraded into a method claim, move to what the code-side evidence does and does not add.

## Section 4: Code-Side Behavior as Harm Profile, Not Mechanism

### Key content

- Reframe MBPP as boundary evidence.
- Show that code behavior should be interpreted as harm localization and failure typing, not as proof of a general mechanism.
- Explain why equality between `CARD-84` and `RAND-84` on MBPP further constrains any method-forward narrative.

### Evidence to include

- From `main_results_table.csv`:
  - `CARD-84` MBPP accuracy = `0.04`
  - `RAND-84` MBPP accuracy = `0.04`
- From `harm_profile_table.csv`:
  - `CARD-84` concentrates failures in `NameError`, `SyntaxError`, and `IndentationError`
  - The distribution differs from `DNB-84` and `RAND-84`, but not in a way that supports a clean mechanism story
- From `code_failure_modes.md`

### Transition

With the reasoning-side signal and code-side boundary both established, the paper can state what the current evidence actually supports and where the wording ceiling must sit.

## Section 5: What the Current Evidence Supports

### Key content

- State the explicit claim ceiling:
  - localized signal against a compute-matched control;
  - no clean separation from a sham control;
  - entropy as a risk marker;
  - audited-slice scope only.
- Convert the negative result into a constructive contribution:
  - the value of stronger sham controls in small-gain settings;
  - the value of sample-level and current-only asset closure.
- Explain why the trajectory addon was intentionally skipped by the negative pivot.

### Evidence to include

- `claim_scope_map.json`
- `proposal.md`
- `idea_validation_decision.json`

### Transition

After stating the claim ceiling, the discussion can broaden to why this negative case matters to future DLM revision work.

## Section 6: Discussion and Implications

### Key content

- Argue that the real lesson is interpretive hygiene: small gains in DLM revision need stronger controls before they deserve method-forward language.
- Explain why the minimal audit template is the supporting contribution:
  - it does not establish a new paradigm;
  - it shows a minimal evidence structure that prevents overclaiming.
- Clarify the future-work boundary:
  - stronger sham controls are allowed as optional follow-up;
  - reopening the old controller-family mainline is not.

### Evidence to include

- `followup_gaps.md`
- `validation_report.json`
- `claim_scope_map.json`

### Transition

Close by restating that the contribution is narrower but more credible than the original positive controller story.

## Section 7: Related Work

### Key content

- Position the paper against training-free DLM revision and control papers that emphasize gains without equally strong sham controls.
- Contrast this work with calibration- or entropy-centered narratives that conflate prediction-side observability with intervention-side validity.
- Explain that the paper contributes a negative-case audit framing rather than a stronger controller family.

### Placement note

Keep the section compact and use it to sharpen the paper’s contribution margin: the paper is not more novel because it has a new method, but because it documents how a stronger control rewrites a plausible positive interpretation.

## Section 8: Conclusion

### Key content

- Restate the paper object:
  - a localized signal existed;
  - a stronger sham control changed its interpretation;
  - the final object is an audited negative case.
- Reiterate the supporting contribution:
  - a minimal auditable template for small-gain interpretation in training-free DLM revision.
- End with the most durable takeaway:
  - future papers should not treat compute-matched controls as sufficient when a budget-matched sham control can still rewrite the story.

## Figure & Table Plan

### Figure 1: The Minimal Audit Template for Small-Gain Revision Claims (Section: Audited Slice and Minimal Audit Template)
- **Purpose**: Introduce the paper’s core conceptual object before any numbers appear.
- **Type**: flow_chart
- **Content**: A four-block schematic showing `compute-matched control`, `sham control`, `sample-level audit`, and `current-only artifact closure`, with arrows into the final claim ceiling.
- **Key takeaway**: The paper’s contribution is an audit structure that changes interpretation, not a new controller.
- **Generation**: manual_diagram
- **Data source**: `current/idea/proposal.md`, `current/exp/pilot_evidence_closure_v1/analysis/packaging/validation_report.json`, `current/exp/pilot_evidence_closure_v1/analysis/packaging/claim_scope_map.json`

### Table 1: Main Audited Results Across the Four Arms (Section: Main Audited Result on GSM8K)
- **Purpose**: Show the headline reasoning and code numbers together in the audited slice.
- **Type**: comparison_table
- **Content**: `arm`, `dataset`, `accuracy`, `correct_count`, `avg_nfe`, `avg_tokens_changed`, `batch_size`, `attention_backend`, `compile_enabled`.
- **Key takeaway**: `CARD-84` beats `DNB-84` on GSM8K but does not separate from `RAND-84` strongly enough to support a positive controller claim.
- **Generation**: data_table
- **Data source**: `current/exp/pilot_evidence_closure_v1/analysis/packaging/main_results_table.csv`

### Figure 2: Repair and Harm Counts Reframe the GSM8K Signal (Section: Main Audited Result on GSM8K)
- **Purpose**: Make the shift from positive signal to audited negative interpretation visually obvious.
- **Type**: stacked_bar
- **Content**: `fixed`, `harmed`, `unchanged_correct`, `unchanged_wrong` for `dnb84_vs_dnb64`, `card84_vs_dnb64`, `rand84_vs_dnb64`, `card84_vs_dnb84`, and `card84_vs_rand84`.
- **Key takeaway**: The key contrast is not only `CARD-84 > DNB-84`, but also that `CARD-84` does not open a decisive gap over `RAND-84`.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `current/exp/pilot_evidence_closure_v1/analysis/packaging/repair_harm_table.csv`

### Figure 3: MBPP Harm Profile by Failure Mode (Section: Code-Side Behavior as Harm Profile, Not Mechanism)
- **Purpose**: Show why the code-side result belongs in harm localization rather than mechanism validation.
- **Type**: bar_chart
- **Content**: Failure-mode counts for each arm, emphasizing `NameError`, `SyntaxError`, and `IndentationError`.
- **Key takeaway**: Code-side evidence localizes harm and failure structure, but does not support a general controller-success story.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `current/exp/pilot_evidence_closure_v1/analysis/packaging/harm_profile_table.csv`

### Table 2: Claim Ceiling and Required Disclosures (Section: What the Current Evidence Supports)
- **Purpose**: Convert the paper’s wording policy into a visible scientific object.
- **Type**: table
- **Content**: Allowed claims, forbidden claims, and required disclosures.
- **Key takeaway**: The paper’s strongest contribution depends on disciplined claim boundaries.
- **Generation**: data_table
- **Data source**: `current/exp/pilot_evidence_closure_v1/analysis/packaging/claim_scope_map.json`

### Appendix Table A: Artifact Integrity and Lineage Checklist (Section: Appendix)
- **Purpose**: Demonstrate that the negative case is current-only, joinable, and auditable.
- **Type**: table
- **Content**: Artifact path, existence, current-only status, notes.
- **Key takeaway**: The result is backed by a self-contained evidence bundle rather than ad hoc narrative selection.
- **Generation**: data_table
- **Data source**: `current/exp/pilot_evidence_closure_v1/analysis/packaging/validation_report.json`
