# Result Debate Verdict: ComposeAccel (Updated)

**Date**: 2026-04-14 (post-tau=0.0 comparison experiment)
**Synthesizer**: sibyl-result-synthesizer

---

## Score: 6.5 / 10

**What justifies 6.5**: A genuinely novel composability framework (first systematic pairwise orthogonality study of MDM acceleration), one strong mechanistically-grounded result (M1+CD-SSD super-multiplicative synergy, Ortho=1.385, 5.13x combined speedup), an internal control condition isolating the synergy mechanism (M1+naive-T16 Ortho=0.949), and a failure-mode atlas with 4 characterized modes — all of which are novel and uncontested in the literature.

**What holds it back**: The headline Ortho result rests on 2 seeds and 15% of the benchmark — statistically insufficient for publication. CD-SSD standalone is dominated by SSD (lossless 3.46x vs. 35% accuracy retention at 3.40x). The M1 implementation gap (1.38x vs. published 15–26x) is unexplained. The REFINE phase KV-cache ablation (the decisive mechanistic test) has not been run. Raw combined speedup (5.13x) is 5–11x below current SOTA combinations.

**Score upgrade from 6.0**: The tau=0.0 comparison experiment converted what was an unresolved "fatal flaw" into a mechanistic control condition. M1+naive-T16 Ortho=0.949 vs. M1+CD-SSD Ortho=1.385 provides the first controlled isolation of the REFINE phase's synergistic contribution — this is stronger than what the original synthesis had.

---

## Key Conclusion

**The composability framework and failure-mode atlas are the real contributions. The frozen-token REFINE mechanism is the paper's scientific story.**

The core finding is: MDM acceleration composability is strongly bimodal. One pair (KV-cache + step-budget allocation via CD-SSD REFINE) achieves super-multiplicative synergy (Ortho=1.385) through a specific frozen-token cache exploitation mechanism. All other pairs (AR guidance + KV-cache, AR guidance + CD-SSD) destructively interfere through mask-state coupling. Naive step reduction composes near-multiplicatively (Ortho=0.949) but not super-multiplicatively — the REFINE phase's frozen-token structure is what creates the amplification.

The paper must be written as an *analysis paper* where CD-SSD is the study vehicle, not as a methods paper claiming CD-SSD as a production contribution. The tau=0.0 finding is not a weakness to hide — it is a controlled experiment that isolates the REFINE phase's synergistic contribution and should be presented as such.

---

## 5 Verified Facts

1. **M1+CD-SSD super-multiplicative synergy is cross-seed consistent**: Ortho=1.385 (seed 42: 1.292, seed 123: 1.478). Combined speedup 5.13x exceeds product of individual speedups (4.69x). The mechanism — frozen-token KV anchors in CD-SSD's REFINE phase — is supported by the control condition: M1+naive-T16 achieves only Ortho=0.949, demonstrating the REFINE phase's specific contribution.

2. **tau=0.0 paradox resolved — CD-SSD is a step-budget allocation mechanism**: CD-SSD(tau=0.0) produces identical accuracy to naive-T16 (42.0% GSM8K) at slightly lower throughput (7.12x vs. 7.56x). The experiment's own decision logic: "CD-SSD can be reframed as a step-reduction method." CD-SSD's REFINE phase (tau=0.9) recovers ~4.5pp accuracy at cost of ~3x throughput reduction, AND creates frozen-token anchors enabling Ortho=1.385 vs. 0.949.

3. **MDM composability exhibits a stark bimodal pattern**: Of 3 testable pairs, 1 synergizes (Ortho=1.385) and 2 show severe destructive interference (Ortho=0.493 and 0.301). The gap between synergy and best-case interference is 0.892 — no plausible noise model explains this as a gradient with sparse sampling.

4. **M2 (adaptive step scheduling) is structurally incompatible with discrete MDMs**: LLaDA's per-step mask dependencies make DDIM-style step skipping impossible. Accuracy collapse is abrupt and monotonic: 76% retention at 2x step jump → 3.2% at 6x+. This generalizable negative finding belongs in the failure atlas.

5. **Coding benchmarks (HumanEval 2.4%, MBPP 0.0%) are statistically uninformative** for LLaDA-8B-Instruct. All aggregate QAS and accuracy retention metrics using these benchmarks are contaminated by 0/0 → 1.0 mapping artifacts. Primary analysis must exclude coding benchmarks.

---

## 3 Critical Risks

1. **Statistical insufficiency of headline result**: Ortho=1.385 rests on 200 GSM8K samples and 2 seeds. Full-scale (3 seeds, 1319 GSM8K samples) could shift the mean Ortho significantly given the 13.4% inter-seed variance. The 22% baseline TPS discrepancy between sessions (seed 42: 25.4 tok/s vs. seeds 123/456: 33.9 tok/s) may additionally inflate Ortho for seed 42. If full-scale Ortho < 1.0, the "super-multiplicative" headline must be downgraded.

2. **REFINE ablation not yet run**: The paper's mechanistic claim — that frozen tokens in the REFINE phase are the synergy locus — is supported by the M1+naive-T16 control condition (Ortho=0.949) but not directly validated. Without the ablation (disable M1 in REFINE phase only), the mechanism remains a story, not an experimentally confirmed finding.

3. **CD-SSD dominated by SSD**: SSD (arXiv:2510.04147) achieves lossless 3.46x speedup; CD-SSD achieves 3.40x at 35% accuracy retention. Without a direct comparison or a result showing CD-SSD uniquely composes with M1 (which requires SSD+M1 data), reviewers will question why CD-SSD is used as the study vehicle.

---

## Action Plan

**Decision: PROCEED**

Three mandatory experiments (~14 GPU-hours) before paper submission:

| Priority | Experiment | GPU-hours | Rationale |
|---------|-----------|-----------|-----------|
| **1st** | P2: REFINE KV-cache ablation (M1 disabled in REFINE only) | 2h | Validates frozen-token mechanism claim; most information per GPU-hour |
| **2nd** | P1: Full-scale M1+CD-SSD (3 seeds, full GSM8K + MATH500) | 8h | Statistical foundation for headline claim; must resolve before any submission |
| **3rd** | P4: SSD+M1 composability (conditional on P1 go) | 4h | Competitive differentiation; either outcome strengthens the paper |

**Immediate action (today)**: Update paper.md to reflect tau=0.0 resolution — reframe CD-SSD as step-budget allocation mechanism, add M1+naive-T16 Ortho=0.949 as control condition.

**Go/No-Go thresholds after P1**:
- Ortho mean ≥ 1.0 → NeurIPS 2026 primary target. "Super-multiplicative synergy" headline.
- Ortho mean ∈ [0.85, 1.0) → NeurIPS 2026 with hedged language. "Strongly orthogonal composition."
- Ortho mean < 0.85 → EMNLP 2026 main or NeurIPS workshop. Composability framework + failure atlas as contribution.

**Abstract correction required immediately**: Remove any reference to claimed speedups not in the experimental data. Replace with: "5.13x combined speedup via super-multiplicative synergy (Ortho=1.385) between KV-cache and step-budget-allocation-based self-speculative denoising, against a background of catastrophic interference in all other method pairs, with a failure-mode atlas characterizing four destructive interference patterns."
