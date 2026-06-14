# Planning Critique

## Verdict

The planning documents made the right strategic pivot, but they still contain stronger promises than the delivered artifacts and paper currently satisfy. The biggest planning failure is not direction. It is closure discipline: some obligations were locked in, partially executed, and then left in an ambiguous pilot state instead of being either fully canonicalized or explicitly downscoped.

## Strong Planning Decisions

- The methodology correctly refused to reopen the hero-controller search (`plan/methodology.md:5-10`).
- The proposal correctly elevated bucket audit to the main evidence lane and runtime lineage to the support lane (`idea/proposal.md:79-105`).
- The seed spot-check was correctly treated as a quality gate rather than a decorative appendix (`idea/proposal.md:146-154`).

These decisions gave the workspace a coherent identity.

## Planning Failures

### 1. Locked deliverables were not reconciled with emitted artifacts

The proposal explicitly locks in:

- `exp/results/benefit_bucket_audit.json`
- `exp/results/benefit_bucket_examples.json`

at `idea/proposal.md:137-148`.

But the workspace currently exposes only:

- `exp/results/benefit_bucket_audit_pilot.json`
- `exp/results/benefit_bucket_examples_pilot.json`

This is not a cosmetic mismatch. It breaks the promised paper-to-artifact contract and makes the result package look half-promoted.

### 2. The plan promised richer bucket analysis than the paper now uses

The proposal says Priority 1 must include per-bucket averages for `tokens_changed`, average signal value, correctness alignment, and representative examples (`idea/proposal.md:158-166`). The bucket artifact contains enough raw data to compute some of this, but the manuscript does not surface it. That is a planning closure issue: the analysis obligation was identified correctly and then not fully landed in the paper.

### 3. `Runtime-first setup` was only partially carried through to the headline table

`plan/methodology.md:46-54` says runtime setup should happen before new experiments and should prevent polluted comparisons. The runtime probe succeeded, but the runtime fairness matrix still imports headline rows from an older artifact with different batch sizes and compile states. Planning should have forced a binary choice:

- either regenerate the comparison under the audited runtime path,
- or explicitly label the old rows as historical context.

Right now it does neither cleanly.

### 4. The backup path is better matched to the evidence than the main framing

The alternatives file says `cand_minimal_scope` should be activated if the observer/controller evidence remains thin (`idea/alternatives.md:3-14`). That condition is effectively true. The current paper has strong bucket evidence, decent runtime protocol evidence, and weak direct evidence for the observer-controller split. Planning should therefore either:

- promote `cand_minimal_scope` to the official scope,
- or schedule one targeted experiment to justify the stronger frame.

Instead, the workspace sits between those two choices.

## Planning Recommendation

The planning layer should now do one of the following and say so explicitly:

1. **Close as minimal scope.**
   Treat the final paper as `honest compute + bucket audit + minimal seed closure`, and remove stronger observer-controller rhetoric from the plan and manuscript.

2. **Open one last matched test.**
   Add one direct experiment that actually evaluates the observer-controller claim under matched runtime conditions.

Anything else is planning ambiguity, not scientific ambition.

## Concrete Planning Repairs

- Promote or alias `_pilot` artifacts to the canonical names promised in the proposal.
- Add an explicit decision note stating whether the project is now on `cand_minimal_scope`.
- Update the methodology so the runtime-fairness matrix is described as either historical context or audited evidence, but not both.
- Add one line in the planning docs that translates `coverage = 100%` into absolute sample count whenever the slice is smaller than the full benchmark.
