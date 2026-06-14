# Experiment Critique

## Core judgment

The experiment logic is appropriate for the paper's claim ceiling. The decisive strength is not absolute performance, but the fact that the sham-control comparison changes the interpretation of the active-control win.

## Main weaknesses

1. The evidence comes from a 100-example audited slice, not a benchmark-wide estimate. That is acceptable for a negative-case audit, but the limitation must stay front and center.
2. The selection policy includes a high-entropy prescan, which is sensible for auditing vulnerable samples but also invites selection-sensitivity concerns if the paper sounds too general.
3. The runtime contract records later batch probing and compile attempts, while the reported arm metrics remain eager-path audited runs at batch 8. This is scientifically fine, but reviewer-facing explanation must stay extremely clear.

## Concrete suggestion

Add one compact sentence in the methods or discussion making the experiment logic explicit: this paper tests interpretive attribution on a fixed audited slice, not average-case benchmark superiority.
