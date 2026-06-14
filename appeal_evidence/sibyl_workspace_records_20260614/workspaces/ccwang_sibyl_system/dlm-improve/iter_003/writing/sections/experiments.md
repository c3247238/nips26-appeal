# Experiments

## Experimental setup

The audited package fixes one seed (`42`) and evaluates four arms on a current-only slice of 100 examples. The slice contains 50 GSM8K items and 50 MBPP items, chosen from a prescan pool of 100 per dataset with a 50% high-entropy target and a balanced mid/low-entropy remainder. This is therefore an audit-oriented evaluation rather than a benchmark-wide estimate. All arms share the same model family, prompt template, post-processing policy, generation budget, and logged runtime lineage.

The runtime disclosure is intentionally precise. A setup-time batch probe found a safe batch ceiling of 28 across representative prompt buckets under the eager attention backend. The packaged experimental arms, however, were executed with batch size 8, and Table 1 reports the recorded arm-level settings exactly as logged. This mismatch is not hidden because the point of the paper is not to optimize a systems stack; it is to preserve a reviewer-auditable lineage for the negative-case interpretation.

## Main audited results

Table 1 reports the headline results across the four arms. Two facts matter most. First, `CARD-84` improves over `DNB-84` on GSM8K, moving from 0.18 to 0.32 accuracy and increasing the number of correct audited GSM8K solutions from 9 to 16. Second, `CARD-84` does not cleanly separate from the sham control `RAND-84`, which reaches 0.30 accuracy and 15 correct GSM8K solutions. The positive reading survives the active-control comparison but not the sham-control comparison.

Table 1. Main audited results on the 100-sample current-only slice.

| Arm | GSM8K accuracy | MBPP accuracy | Total correct | Average NFE | Avg. tokens changed | Batch size | Attention backend | Compile enabled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `DNB-64` | 0.30 | 0.04 | 17 | 64.0 | 0.00 | 8 | eager | False |
| `DNB-84` | 0.18 | 0.02 | 10 | 83.5 | 0.00 | 8 | eager | False |
| `CARD-84` | **0.32** | 0.04 | **18** | 68.0 | **9.06** | 8 | eager | False |
| `RAND-84` | 0.30 | **0.04** | 17 | 67.0 | 0.62 | 8 | eager | False |

Accuracy alone does not show why the interpretation changes, so we rely on outcome decomposition. As shown in Figure 2, `CARD-84` repairs 10 GSM8K samples and harms 3 relative to `DNB-84`, yielding a net repaired margin of +7. Against `RAND-84`, however, it repairs 1 sample and harms 0, yielding only +1 net repaired. The localized signal is therefore real in the sense that it survives the compute-matched active control, but it is not strong enough to support a validated controller claim once the sham control is added.

**Figure 2.** GSM8K repair and harm counts for the key audited comparisons. The stacked bars visualize fixed, harmed, unchanged-correct, and unchanged-wrong samples, while the net repaired annotations make clear that the decisive contrast is `CARD-84` against `RAND-84`, not just `CARD-84` against `DNB-84`.

## Code-side boundary evidence

MBPP should be read as boundary evidence rather than as a secondary success story. `CARD-84` and `RAND-84` both obtain 0.04 accuracy on the 50 audited MBPP examples, while `DNB-84` reaches only 0.02. This means the code-side slice does not separate the audited intervention from the sham control, even though `CARD-84` changes many more tokens on average. The most defensible reading is not that entropy-guided revision discovered a code mechanism, but that the current intervention and the sham baseline share a similar ceiling on this audited code slice.

Figure 3 sharpens this reading. `CARD-84` concentrates failures in `NameError` (29), `SyntaxError` (13), and `IndentationError` (4), whereas `RAND-84` concentrates in `NameError` (22), `SyntaxError` (20), and `IndentationError` (3). The difference is real enough to localize harm patterns, yet not decisive enough to justify a mechanism claim. In particular, the heavier `NameError` concentration for `CARD-84` and the identical MBPP accuracy of `CARD-84` and `RAND-84` argue against upgrading the code results into evidence for a generally better controller.

**Figure 3.** MBPP failure-mode counts across the four audited arms. The figure is used as a harm-profile visualization rather than as proof of a controller-specific mechanism, because `CARD-84` and `RAND-84` remain tied in audited MBPP accuracy.

## Result summary

The experimental picture is therefore asymmetrical by design. There is enough signal to reject the trivial claim that only extra denoising budget matters, because `CARD-84` clearly beats `DNB-84` on GSM8K. There is not enough signal to claim validated entropy-guided revision, because `RAND-84` closes almost all of that gap on the same audited slice. The negative conclusion is not that nothing happened. It is that the stronger control determines the strongest safe interpretation.

<!-- FIGURES
- Figure 2: gen_repair_harm.py, repair_harm.pdf — Stacked GSM8K repair-harm chart with net repaired annotations.
- Figure 3: gen_mbpp_harm_profile.py, mbpp_harm_profile.pdf — Grouped bar chart for MBPP failure-mode counts.
- Table 1: inline — Main audited results across the four arms.
-->
