# A Stronger Sham Control Rewrites a Small-Gain Story in Training-Free DLM Revision

## Abstract

Training-free revision in diffusion language models (DLMs) can produce small gains that are easy to over-interpret when the control structure is weak. We report an audited negative case in which an entropy-guided revision arm, `CARD-84`, appears promising against a compute-matched active control but fails to cleanly separate from a budget-matched sham control. On a current-only audited slice of 100 examples, `CARD-84` improves GSM8K accuracy from 0.18 to 0.32 relative to `DNB-84`, yielding a net repaired margin of +7, yet reaches only a +1 net repaired margin against the sham arm `RAND-84`, which reaches 0.30 accuracy on the same slice. On MBPP, `CARD-84` and `RAND-84` tie at 0.04 accuracy, constraining any mechanism-forward interpretation to a harm-profile reading rather than controller efficacy. The resulting contribution is not a new winning inference method. Instead, we document a minimal audit template for small-gain DLM revision claims: a compute-matched active control, a budget-matched sham control, sample-level repair/harm accounting, and current-only artifact closure. The main lesson is interpretive. In a small-gain regime, compute-matched baselines alone are not sufficient when a stronger sham control can still rewrite the story.

## 1. Introduction

Training-free DLM inference has shifted from proving that non-autoregressive denoising can work to asking which test-time interventions are worth trusting in practice [RADD; DPad; Prophet]. In this regime, modest gains are common, but their meaning is often unstable. A method can outperform a lower-budget or compute-matched baseline and still fail to justify a stronger controller claim once more demanding controls are introduced. The present paper is about that failure mode.

Our starting point was a plausible positive result. On an audited GSM8K slice, the entropy-guided revision arm `CARD-84` reaches 0.32 accuracy, compared with 0.18 for the compute-matched active control `DNB-84`. At the sample level, this corresponds to 10 fixes and 3 harms relative to `DNB-84`, or a net repaired margin of +7. Had the analysis stopped there, the natural interpretation would have been that an observer-side entropy signal can be converted into a successful training-free revision controller. The stronger audit performed in this iteration shows that such a write-up would be overstated.

The decisive comparison is to a budget-matched sham control. On the same GSM8K slice, `RAND-84` reaches 0.30 accuracy, and the net repaired margin of `CARD-84` over `RAND-84` is only +1. Once this sham control is introduced, the earlier positive story no longer survives. The right object is therefore not a method paper, a protocol manifesto, or a "near-win" rescue narrative. It is an audited negative case showing that a localized signal against an active control can still fail the stronger question of attribution.

This paper makes three contributions.

1. It documents an audited negative case in which a localized reasoning-side signal survives a compute-matched comparison but not a budget-matched sham control.
2. It packages that negative case through a minimal audit template: compute-matched active control, sham control, sample-level repair/harm accounting, and current-only artifact closure.
3. It turns the empirical lesson into a claim ceiling for training-free DLM revision: entropy may be treated here as a risk marker for vulnerable samples, but not as a validated targeting rule [ActCab+CoDec, 2024; Confidence over Time, 2026].

This framing matters because recent DLM and test-time-control literature increasingly operates in small-gain settings where implementation, stopping, and targeting choices can move the headline [DPad; Prism; Sampler-Centric Evaluation]. In those settings, a compute-matched baseline is necessary but insufficient. It rules out the trivial explanation that more budget alone caused the gain, yet it does not rule out the possibility that a generic or weakly structured perturbation would have produced nearly the same effect.

The rest of the paper proceeds accordingly. Section 2 defines the audited slice and the minimal audit template. Section 3 positions the paper against DLM inference, uncertainty-guided control, and sampler-centric evaluation. Section 4 reports the main audited results with Table 1, Figure 2, and Figure 3. Section 5 states the claim ceiling and explains why the result remains scientifically useful even as a negative case. Section 6 concludes. Appendix A provides a compact artifact-integrity checklist for the current-only evidence bundle.

## 2. Audited Slice and Minimal Audit Template

The paper studies a fixed audited slice rather than a benchmark-wide estimate. The slice contains 100 current-only samples, with 50 GSM8K reasoning problems and 50 MBPP coding problems, all traced through `sample_manifest.json`. The selection policy combines a high-entropy prescan with a balanced mid/low-entropy remainder, making the slice suitable for bounded interpretive auditing rather than for estimating population-level benchmark performance. All claims in this manuscript are restricted to that audited slice.

We compare four fixed inference arms:

- `DNB-64`: lower-budget reference arm.
- `DNB-84`: compute-matched active control.
- `CARD-84`: audited entropy-guided revision arm.
- `RAND-84`: budget-matched sham control.

The arm roles matter more than the raw metrics. `DNB-84` tests whether the signal survives a stronger budget than `DNB-64`. `RAND-84` tests whether the same signal survives a sham intervention with approximately matched budget. The paper's negative pivot follows from the second test.

Figure 1 defines the minimal audit template used throughout the manuscript.

[Figure 1 here: `writing/figures/audit_template.pdf`.]

**Figure 1.** Minimal audit template for small-gain DLM revision claims. A localized gain is first tested against a compute-matched active control, then against a budget-matched sham control, and finally restricted by sample-level audit and current-only artifact closure. In the present case, the positive interpretation fails at the sham-control stage.

The runtime contract is kept explicit but modest. The setup records a left-padded LLaDA-8B-Instruct configuration with temperature 1.0, a 512-token prompt budget, and 128 generation tokens. The setup probe found a safe batch ceiling of 28 under the eager attention path, but the already-completed audited arms in the current bundle were preserved at batch size 8 rather than rerun at the higher probe ceiling so that the paper would remain tied to a fixed current-only artifact bundle. A compile attempt was recorded during setup, yet the experimental arm metrics remain on the eager path. We treat these runtime details as disclosure objects rather than contributions.

At the outcome level, we use four mutually exclusive sample-level categories for any pairwise comparison: `fixed`, `harmed`, `unchanged_correct`, and `unchanged_wrong`. For dataset \(D\), the summary quantity of interest is the net repaired count,
\[
\Delta_{\text{repair}}(a,b;D) = \mathrm{Fix}(a,b;D) - \mathrm{Harm}(a,b;D).
\]
This quantity is what lets the paper distinguish between an apparently positive contrast and a robustly attributable one. In the present bundle, \(\Delta_{\text{repair}}(\text{CARD-84}, \text{DNB-84}; \text{GSM8K}) = 7\), while \(\Delta_{\text{repair}}(\text{CARD-84}, \text{RAND-84}; \text{GSM8K}) = 1\).

Finally, the method includes a claim policy. `CARD-84` may be described as an audited intervention arm, not as a winning inference method. Entropy may be described as a risk marker for vulnerable samples, not as a validated targeting rule. The skipped trajectory add-on is a deliberate boundary of the negative pivot, preventing the paper from reopening a rescue narrative that the packaged decision already rejected.

## 3. Related Work

Early DLM work established the viability of discrete diffusion text generation, while recent work has moved toward inference-time engineering and test-time scaling [RADD, 2023; LLaDA 1.5, 2025; DPad, 2025; Prophet, 2025; Prism, 2026]. This literature provides many ways to reduce denoising cost, alter stopping behavior, or guide revision without retraining. What it provides less often is a reviewer-facing recipe for deciding when a small downstream gain deserves method-forward language.

Uncertainty-guided control makes the gap sharper. In neighboring autoregressive settings, confidence-like signals have been used for guided decoding, critique, and compute allocation [ActCab+CoDec; Don’t Think Twice!; COREA]. The consistent lesson is that predictive observer signals do not automatically validate the intervention that consumes them. A signal can be informative and still fail to establish controlled efficacy. That is exactly the distinction our audited negative case makes concrete inside a DLM revision setting.

Sampler-centric evaluation adds a second caution. Recent work asks whether apparent DLM improvements should be attributed to the denoiser, the sampler, or the surrounding runtime and argues that aggregate performance alone can blur those sources [Sampler-Centric Evaluation]. Our paper is aligned with that concern, but its contribution is narrower: we do not propose a general sampler correctness framework. Instead, we show how a minimal evidence bundle can prevent a plausible small-gain result from being written up too aggressively.

The contribution gap of this paper is therefore specific. Relative to DLM inference papers, we do not offer a stronger controller family. Relative to uncertainty-guided decoding papers, we do not offer a new observer. Relative to sampler-evaluation papers, we do not offer broader formalism. We offer a reviewer-facing audit template demonstrating how a sham control can materially rewrite the interpretation of a small claimed gain.

## 4. Main Audited Results

The audited package fixes one seed and evaluates the four arms on the same current-only 100-sample slice. Table 1 reports the headline numbers. The most tempting positive comparison is `CARD-84` versus `DNB-84` on GSM8K: 0.32 versus 0.18 accuracy, corresponding to 16 versus 9 correct solutions on the 50 audited GSM8K examples.

Table 1. Main audited results on the 100-sample current-only slice.

| Arm | GSM8K accuracy | MBPP accuracy | Total correct | Average NFE | Avg. tokens changed | Batch size | Attention backend | Compile enabled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `DNB-64` | 0.30 | 0.04 | 17 | 64.0 | 0.00 | 8 | eager | False |
| `DNB-84` | 0.18 | 0.02 | 10 | 83.5 | 0.00 | 8 | eager | False |
| `CARD-84` | **0.32** | 0.04 | **18** | 68.0 | **9.06** | 8 | eager | False |
| `RAND-84` | 0.30 | **0.04** | 17 | 67.0 | 0.62 | 8 | eager | False |

Accuracy alone does not capture the interpretive shift, so Figure 2 decomposes the sample-level outcomes. `CARD-84` repairs 10 GSM8K samples and harms 3 relative to `DNB-84`, giving a net repaired margin of +7. Against `RAND-84`, however, it repairs only 1 sample and harms none, giving +1. The localized signal is therefore auditable in the active-control comparison, but it is not strong enough to support a validated controller claim once the sham control is taken seriously.

[Figure 2 here: `writing/figures/repair_harm.pdf`.]

**Figure 2.** GSM8K repair and harm counts for the key audited comparisons. The decisive comparison is `CARD-84` against the sham control `RAND-84`, not just `CARD-84` against the compute-matched `DNB-84` arm.

The code-side slice further narrows the interpretation. On MBPP, `CARD-84` and `RAND-84` both obtain 0.04 accuracy on the 50 audited examples, whereas `DNB-84` reaches 0.02. This means the code-side bundle does not separate the audited intervention from the sham control, even though `CARD-84` changes many more tokens on average. The safest reading is therefore boundary-setting rather than mechanism confirmation.

Figure 3 makes that boundary concrete. `CARD-84` concentrates MBPP failures in `NameError` (29), `SyntaxError` (13), and `IndentationError` (4), while `RAND-84` concentrates in `NameError` (22), `SyntaxError` (20), and `IndentationError` (3). These differences are useful for harm localization, but they do not establish a general controller mechanism, especially because the overall MBPP accuracy remains tied.

[Figure 3 here: `writing/figures/mbpp_harm_profile.pdf`.]

**Figure 3.** MBPP failure-mode counts across the four audited arms. The figure is used to localize harm patterns, not to justify a mechanism-forward success story.

The result is therefore asymmetrical by construction. There is enough evidence to reject the trivial explanation that only extra denoising budget matters, because `CARD-84` clearly outperforms `DNB-84` on the audited GSM8K slice. There is not enough evidence to claim validated entropy-guided revision, because the budget-matched sham control closes almost all of that gap on the same slice.

## 5. Discussion and Claim Ceiling

The most important outcome of this iteration is a clean claim boundary. The current evidence supports the statement that `CARD-84` shows a localized GSM8K signal against the compute-matched active control. It does not support the stronger statement that entropy-guided revision is validated as a controller, because the decisive sham-control quantity is only +1 net repaired on GSM8K, below the preset gate captured in the packaged decision artifact.

Table 2 converts this claim boundary into a visible scientific object.

Table 2. Allowed and disallowed interpretations for the current audited bundle.

| Category | Safe statement |
| --- | --- |
| Allowed claim | `CARD-84` shows a localized GSM8K signal against the compute-matched `DNB-84` baseline. |
| Allowed claim | That localized signal does not cleanly separate from budget-matched random targeting. |
| Allowed claim | Entropy is best treated here as a risk marker for vulnerable samples. |
| Required disclosure | Scope is limited to the audited slice and current-only artifacts. |
| Required disclosure | `CARD-84` fails the stricter sham-control gate against `RAND-84` (+1 net repaired on GSM8K). |
| Required disclosure | The trajectory add-on was intentionally skipped by the negative pivot. |
| Forbidden claim | Entropy-guided revision reliably improves DLM reasoning. |
| Forbidden claim | `CARD-84` is the winning inference method. |
| Forbidden claim | The present evidence establishes a new DLM protocol paradigm. |
| Forbidden claim | The code/reasoning divergence has been mechanistically explained. |

This table also explains why the negative result remains useful. The paper blocks a plausible overclaim under realistic DLM conditions [DPad; Prism]. Without the sham control, the present slice would likely have been written up as a modest positive method paper. With the sham control, it becomes a minimal audit template for small-gain DLM revision claims.

MBPP reinforces the same lesson from another direction. The tied code accuracy of `CARD-84` and `RAND-84`, combined with different failure distributions, is enough to motivate harm localization but not enough to support cross-task transport of the observer signal. In other words, a risk marker on the reasoning side does not automatically yield a portable revision rule on the coding side.

The future-work boundary is intentionally narrow. Stronger sham controls, external-prior reinterpretation of related DLM papers, and more compressed harm-profile visualizations remain plausible follow-ups. What this iteration explicitly disallows is reopening the abandoned controller-family mainline or writing the current result as if it were one experiment away from becoming a positive method claim.

## 6. Conclusion

This paper reports an audited negative case for training-free DLM revision. On an audited 100-sample slice, `CARD-84` gains a +7 net repaired margin over `DNB-84` on GSM8K but only +1 over the budget-matched sham control `RAND-84`. That contrast is enough to preserve a localized signal and enough to reject a validated controller narrative.

The supporting contribution is a minimal audit template composed of a compute-matched active control, a sham control, sample-level repair/harm accounting, and explicit claim-to-asset lineage. In the present case, that template produces a narrower but more credible paper object: entropy acts as a risk marker for vulnerable samples, while the stronger sham control prevents the result from being written up as a winning inference method.

The broader lesson is simple. In small-gain, test-time-controlled DLM settings, compute-matched baselines alone are not sufficient when a budget-matched sham control can still rewrite the conclusion.

## Appendix A. Artifact Integrity Checklist

Table A1. Representative current-only artifacts supporting the audited negative case.

| Artifact | Exists | Current-only | Role in the paper |
| --- | --- | --- | --- |
| `setup/sample_manifest.json` | Yes | Yes | Defines the audited 100-sample slice. |
| `setup/runtime_contract.json` | Yes | Yes | Logs the runtime path and disclosure conditions. |
| `analysis/per_sample_audit.csv` | Yes | Yes | Supports sample-level repair and harm accounting. |
| `analysis/transition_matrix.csv` | Yes | Yes | Supports the GSM8K pairwise comparison claims. |
| `analysis/claim_to_asset_map.json` | Yes | Yes | Maps allowed and forbidden claims to supporting assets. |
| `analysis/code_failure_modes.md` | Yes | Yes | Supports the MBPP harm-profile discussion. |
| `analysis/decision.json` | Yes | Yes | Records the failed sham-control gate and selected negative pivot. |

## Figures and Tables

- Figure 1: `audit_template.pdf` — Minimal audit template showing how the sham control and artifact closure enforce the claim ceiling.
- Figure 2: `repair_harm.pdf` — GSM8K repair/harm chart showing that the sham-control comparison rewrites the interpretation.
- Figure 3: `mbpp_harm_profile.pdf` — MBPP failure-mode profile showing boundary evidence rather than mechanism validation.
- Table 1: inline — Main audited results across the four arms.
- Table 2: inline — Claim ceiling, required disclosures, and forbidden interpretations.
- Table A1: inline — Representative artifact-integrity checklist for the current-only evidence bundle.

## Draft References

- **RADD:** *A Reparameterized Discrete Diffusion Model for Text Generation* (2023).
- **LLaDA 1.5:** *LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models* (2025).
- **DPad:** *DPad: Efficient Diffusion Language Models with Suffix Dropout* (2025).
- **Prophet:** *Diffusion Language Models Know the Answer Before Decoding* (2025).
- **Prism:** *Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models* (2026).
- **Sampler-Centric Evaluation:** *Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models* (2026).
- **ActCab+CoDec:** *Enhancing Language Model Factuality via Activation-Based Confidence Calibration and Guided Decoding* (2024).
- **Don’t Think Twice!:** *Over-Reasoning Impairs Confidence Calibration* (2025).
- **Confidence over Time:** *Confidence Calibration with Temporal Logic for Large Language Model Reasoning* (2026).
- **COREA:** *Confidence-Calibrated Small-Large Language Model Collaboration for Cost-Efficient Reasoning* (2026).
