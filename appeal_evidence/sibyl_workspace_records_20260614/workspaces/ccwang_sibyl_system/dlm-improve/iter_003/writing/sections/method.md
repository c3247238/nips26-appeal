# Method

## Paper object and audited slice

This paper studies a fixed audited slice rather than a broad benchmark sweep. The slice contains 100 current-only samples, with 50 GSM8K reasoning problems and 50 MBPP coding problems, all traced to the manifest in `sample_manifest.json`. The slice was constructed by combining a high-entropy prescan with a balanced mid/low-entropy remainder, so the audit intentionally emphasizes difficult or vulnerable cases without claiming benchmark representativeness. In our notation, the audited set is \(\mathcal{S}\), each sample \(s \in \mathcal{S}\) belongs to either GSM8K or MBPP, and each sample carries a precomputed observer-side entropy score \(u(s)\) used only for audit stratification.

The paper compares four fixed inference arms:

| Arm | Role in the audit | Average NFE | Average tokens changed |
| --- | --- | ---: | ---: |
| `DNB-64` | Lower-budget reference arm | 64.0 | 0.00 |
| `DNB-84` | Compute-matched active control | 83.5 | 0.00 |
| `CARD-84` | Audited entropy-guided revision arm | 68.0 | 9.06 |
| `RAND-84` | Budget-matched sham control | 67.0 | 0.62 |

The role assignment is the key methodological choice. `DNB-84` tells us whether the localized signal can survive a stronger budget than `DNB-64`, but `RAND-84` tells us whether that signal survives a sham intervention with roughly matched budget. The paper's negative pivot follows from the second comparison, not the first.

As shown in Figure 1, the study uses a minimal audit template rather than a new controller protocol. The template is deliberately compact: observe a small gain, compare it to a compute-matched control, then compare it again to a budget-matched sham control, and finally restrict claims to whatever survives sample-level and artifact-level auditing.

**Figure 1.** Minimal audit template for small-gain DLM revision claims. The rendered paper figure should depict how a localized gain is filtered through a compute-matched active control, a sham control, sample-level auditing, and current-only artifact closure before a final claim ceiling is stated.

## Runtime contract and evidence lineage

All comparisons inherit a frozen prompt template and frozen post-processing policy. The runtime contract records a left-padded setup with a 512-token prompt budget, 128 generation tokens, and temperature 1.0 on the LLaDA-8B-Instruct model family. The setup probe also records that the requested attention implementation resolved to the eager path and that flash attention was unavailable in the current environment. A setup-time compile attempt was logged, but the audited arm metrics remain tied to the explicit eager runtime path reported in the packaged results. We treat these runtime details as disclosure objects, not as contributions.

The evidence lineage is equally explicit. Each claim in the paper maps to a current-only artifact: `transition_matrix.csv` supports the GSM8K repair comparison, `code_failure_modes.md` supports the MBPP harm-profile reading, and `claim_to_asset_map.json` records the allowed and disallowed claim scopes. This mapping is important because the paper is not trying to generalize from an unbounded run history. It is documenting exactly one auditable bundle and enforcing a strict claim ceiling from that bundle outward.

## Outcome decomposition

The sample-level audit classifies every pairwise comparison into four mutually exclusive outcomes:

- `fixed`: arm \(a\) is correct where arm \(b\) is incorrect.
- `harmed`: arm \(a\) is incorrect where arm \(b\) is correct.
- `unchanged_correct`: both arms are correct.
- `unchanged_wrong`: both arms are incorrect.

For dataset \(D\), the summary quantity of interest is the net repaired count,
\[
\Delta_{\text{repair}}(a,b;D) = \mathrm{Fix}(a,b;D) - \mathrm{Harm}(a,b;D).
\]
This quantity is more informative than accuracy alone in a small audited slice because it reveals whether an apparent accuracy gap comes from genuine repairs, from avoiding harms, or from a combination that is too fragile to support strong language. In the present paper, \(\Delta_{\text{repair}}(\text{CARD-84}, \text{DNB-84}; \text{GSM8K}) = 7\) establishes the localized signal, while \(\Delta_{\text{repair}}(\text{CARD-84}, \text{RAND-84}; \text{GSM8K}) = 1\) enforces the negative interpretation.

## Claim policy

The method section also fixes the wording policy that governs the rest of the manuscript:

- `CARD-84` may be described as an audited intervention arm, not as the winning inference method.
- The precomputed entropy signal \(u(s)\) may be described as a risk marker, not as a validated targeting rule.
- All conclusions are limited to the audited slice and current-only artifacts.
- The skipped trajectory add-on is a deliberate boundary of the negative pivot, not an unfinished subresult hidden for later rescue.

These rules are methodological, not editorial. They are part of the evidence design because they determine which interpretations remain legal once the stronger sham-control comparison is accounted for.

<!-- FIGURES
- Figure 1: audit_template_desc.md — Flow-diagram specification for the minimal audit template and claim-ceiling logic.
-->
