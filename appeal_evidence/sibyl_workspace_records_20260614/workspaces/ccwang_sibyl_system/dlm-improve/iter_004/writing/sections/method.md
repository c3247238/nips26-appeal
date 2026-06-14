# Method

## Problem Reframing

The starting point of this work is a reframing rather than a clean-slate method invention. Earlier iteration-4 evidence suggested that entropy is informative as an observer-side signal, but that entropy-guided semantic revision does not reliably outperform a matched random control. This creates a distinction between two roles: an **observer** that detects risk, and a **controller** that determines what semantic update should be made. Our central hypothesis is that entropy is better suited to the former role than to the latter.

Accordingly, we define the target problem as **compute routing under a fixed denoising family**. Given an input instance $x$, we first run a shared draft trajectory $d(x)$ for 64 denoising steps. At draft completion, we compute normalized token entropy values $H_i(x)$ over positions and retain a frontier $F(x)$ consisting of the top-$\\rho$ positions, where $\\rho \\approx 0.1211$ in the current full-scale study. Additional denoising is then restricted to this active frontier for at most $T_{\\max}=3$ extra steps.

## Entropy-Routed Candidate

The candidate method, `cand_espd`, uses draft-time entropy to decide where further compute should be spent. The method has four fixed ingredients in the current study:

1. **Frontier scoring rule**: use normalized draft-time token entropy as the routing score.
2. **Retention ratio**: keep the top-$\\rho$ positions as the active frontier.
3. **Stopping rule**: stop frontier-only revision once the masked-frontier entropy ratio falls below $\\tau=0.85$, subject to at least one frontier-only step.
4. **Revision budget**: apply at most $T_{\\max}=3$ frontier-only revision steps after the shared draft.

Importantly, the method does not claim to know what semantic repair should be made at each position. Instead, it claims only that some positions deserve more compute than others. This is why we refer to the method as entropy-routed compute rather than entropy-guided semantic control.

As shown conceptually in Figure 1, the candidate and the sham share the same draft, the same broad budget family, and the same frontier-only revision structure. The tested claim is therefore not whether frontier compute helps in general, but whether entropy-based frontier placement provides a better trade-off than a frontier of the same nominal size placed without entropy routing.

**Figure 1 (fig1_entropy_routed_compute_desc.md).** Entropy-routed compute versus the matched fixed-frontier sham. Both methods share the same 64-step draft and revision budget family, but the candidate uses draft-time entropy to place additional compute, whereas the sham uses a fixed frontier of the same nominal size.

## Fixed-Frontier Sham

The key control is `ESPD-FixedFrontier`, a matched sham designed to preserve the frontier budget family while removing entropy-based frontier placement. The sham uses the same nominal retention ratio and the same stopping family, but the frontier itself is placed deterministically rather than by draft-time uncertainty. This control is intentionally stronger than a generic baseline because it preserves the most obvious budget-related degrees of freedom while disabling the claimed routing mechanism.

The sham is crucial because small-gain dLLM results are easy to over-interpret. If a candidate only beats a weak shared baseline, it remains unclear whether the gain comes from the claimed mechanism, a generic extra-compute effect, or an implementation side effect. By comparing the candidate to a matched fixed-frontier control, we can ask a narrower and more meaningful question: does dynamic frontier placement itself matter?

## Shared Controls and Paired Outcome Accounting

In addition to the fixed-frontier sham, we compare against two shared controls: `CARD-84` and `RAND-84`. The former uses entropy within a standard revision budget family, whereas the latter uses random targeting within the same broad compute regime. These shared controls serve two purposes. First, they anchor the candidate in the broader revision family rather than only in a single sham comparison. Second, they provide a diagnostic check on the old entropy-as-controller framing: if entropy were already a strong semantic targeting rule, `CARD-84` would be expected to clearly outperform `RAND-84`.

To avoid relying only on aggregate accuracy, we also report paired repair/harm decompositions. For a paired comparison between two methods, we record repaired cases $R$, harmed cases $H$, and the net repaired count $\\Delta_{\\text{repair}} = R - H$. This decomposition is not a substitute for statistical confidence intervals, but it makes the structure of a small gain more transparent.

## Runtime Accounting

Because this paper studies a small-gain intervention claim, runtime accounting is part of the method rather than an implementation footnote. We separately log draft and frontier behavior, effective batch size, peak VRAM, wall-clock latency, and auxiliary overhead associated with frontier selection and stopping checks. The goal is not to claim that the current implementation is globally optimal, but to ensure that the candidate, sham, and shared controls are compared under a visible and auditable execution envelope.

<!-- FIGURES
- Figure 1: fig1_entropy_routed_compute_desc.md — Conceptual diagram of the candidate and the matched fixed-frontier sham
-->
