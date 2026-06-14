# Discussion

## What the current evidence supports

The most important outcome of this iteration is not a new decoding recipe but a clean claim boundary. The audited slice supports a localized reasoning-side signal for `CARD-84` against `DNB-84`, yet it does not support a validated controller narrative because that signal does not survive a stronger sham control. This distinction is easy to blur when papers emphasize only compute-matched comparisons or aggregate accuracy. Our evidence bundle keeps the two apart.

Table 2 states the resulting claim ceiling explicitly.

Table 2. Allowed and disallowed interpretations for the current audited bundle.

| Category | Safe statement |
| --- | --- |
| Allowed claim | `CARD-84` shows a localized GSM8K signal against the compute-matched `DNB-84` baseline. |
| Allowed claim | That signal does not cleanly separate from budget-matched random targeting. |
| Allowed claim | Entropy is best treated here as a risk marker for vulnerable samples. |
| Required disclosure | Scope is limited to the audited slice and current-only artifacts. |
| Required disclosure | `CARD-84` fails the stricter sham-control gate against `RAND-84`. |
| Required disclosure | The trajectory add-on was intentionally skipped by the negative pivot. |
| Forbidden claim | Entropy-guided revision reliably improves DLM reasoning. |
| Forbidden claim | `CARD-84` is the winning inference method. |
| Forbidden claim | The present evidence establishes a new DLM protocol paradigm. |
| Forbidden claim | The code/reasoning divergence has been mechanistically explained. |

This table is not merely a stylistic appendix to the paper. It is the scientific endpoint of the audit. Once the sham-control comparison is known, those forbidden statements become incompatible with the evidence, no matter how appealing the `CARD-84 > DNB-84` contrast may look in isolation.

## Why the negative case still matters

A negative result becomes useful when it blocks a plausible overclaim under realistic conditions. That is exactly what happened here. Without `RAND-84`, the current slice would likely have been written up as a modest but real success for entropy-guided revision. The stronger audit shows that this would have been premature. In that sense, the paper's contribution is a minimal audit template for small-gain DLM studies: if a claimed intervention survives only a compute-matched active control but not a budget-matched sham control, the appropriate object is an audited negative case, not a near-win method paper.

This lesson is especially relevant for DLM inference work because the field now sits in a regime where test-time knobs are plentiful and small improvements are common. Once that regime is reached, the methodological burden shifts. The question is no longer just whether a controller can move a metric. It is whether the movement remains attributable after stronger controls, sample-level accounting, and runtime disclosure are enforced.

## Why MBPP is a boundary, not a mechanism

The code-side results reinforce the paper's caution. `CARD-84` and `RAND-84` tie on audited MBPP accuracy, and the remaining differences reside in failure composition rather than net outcome. That is enough to motivate harm localization, but not enough to support a cross-task mechanism story. The paper therefore treats MBPP as a boundary condition: it tells us that the reasoning-side signal does not automatically transport to code and that token-change intensity alone is not evidence of targeted revision quality.

## Future work boundary

The follow-up list remains intentionally narrow. Reviewer-driven future work may include a stronger sham control, an external-prior reinterpretation of similar small-gain DLM papers, or a more compressed MBPP failure taxonomy for visualization. What this iteration does not allow is reopening the original controller-family mainline, rerunning the completed pilot by default, or writing the current result as if it were one experiment away from turning into a positive method paper. The negative pivot is a scientific decision, not a temporary posture.

## Implication for DLM revision studies

The broad implication is modest but durable: in training-free DLM revision, compute-matched controls should not be treated as sufficient when a budget-matched sham control can still rewrite the conclusion. That lesson is narrower than a new algorithmic contribution, but it is also more trustworthy. In the present paper, credibility comes from what the audit prevented us from saying.

<!-- FIGURES
- None
-->
