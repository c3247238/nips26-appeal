# Section Critique: Introduction

**Section**: Introduction (intro.md)
**Reviewer**: Section Critic
**Date**: 2026-03-10

---

## Overall Score: 7 / 10

The introduction is technically thorough and covers the necessary ground -- it motivates the problem, identifies the information island bottleneck, presents two proposed methods, previews key results including negative findings, and outlines the paper structure. However, it suffers from several issues that reduce its effectiveness as an opening section.

---

## Strengths

1. **Clear problem identification**: The "information island problem" is well-named and well-explained. The formal description of how distributional information is discarded after argmax is immediately understandable.

2. **Strong motivation through negative results**: The systematic cataloguing of why existing approaches fail (remasking, DTA, best-of-N) before presenting the proposed methods is effective. It establishes that the problem is non-trivial and that naive solutions do not work.

3. **Quantitative grounding throughout**: Claims are backed by specific numbers (e.g., DMI 9.3% vs vanilla 4.7%, A-CFG 12.5%, DTA 4.8%). This gives the reader concrete anchors.

4. **Honest reporting of negative results**: Dedicating space to failure modes (JSD stability, temporal scheduling, method non-composability) is scientifically valuable and differentiates the paper.

5. **Good contextualization with concurrent work**: References to MetaState (Xia et al., 2026), LRD, ReMix, and EvoToken-DLM show awareness of the broader landscape.

---

## Weaknesses and Specific Improvement Suggestions

### CRITICAL

**C1. Excessive length and density (paragraphs 4-7)**
The introduction is approximately 1,800 words -- roughly 2x the typical length for a top-venue ML paper introduction. Paragraphs 4-7 essentially preview the entire results section, including specific accuracy numbers, ablation outcomes, and failure mode analysis. This level of detail belongs in the results/discussion sections, not the introduction.

*Suggestion*: Cut paragraphs 5-7 (the detailed negative results enumeration and contributions list) by at least 50%. Summarize the key positive result (A-CFG generalizes, BSD provides information-theoretic insight) and the key negative result (methods do not compose; naive scaling fails) in 2-3 sentences each. Move the detailed enumeration to the beginning of Section 4 or Section 5.

**C2. Weak high-level motivation in the opening**
The first paragraph jumps directly into MDLM technical details without establishing why a reader should care. There is no connection to broader impact -- e.g., why parallel/non-autoregressive generation matters for real applications, or why inference-time scaling is practically important beyond benchmark scores.

*Suggestion*: Add 2-3 sentences at the very beginning connecting to a broader motivation: the practical limitations of autoregressive generation (latency, lack of self-correction), and why MDLMs represent a paradigm shift worth investing in.

### HIGH

**H1. Confusing method preview -- BSD vs DMI relationship unclear on first read**
Paragraph 4 introduces DMI as a "critical insight from our prior iteration" and then paragraph 5 introduces BSD as a generalization of DMI. But the reader has not yet been told what DMI is when they encounter its results. The narrative jumps between DMI results, related work results (LRD, ReMix), and the transition to BSD/A-CFG too rapidly.

*Suggestion*: Restructure: (a) State the insight that mask embedding poverty is the bottleneck, (b) briefly describe DMI as the simplest fix, (c) state DMI's result, (d) then introduce BSD and A-CFG as the full methods. Remove the enumeration of LRD/ReMix/EvoToken from the introduction -- these belong in Related Work.

**H2. Pilot-scale results presented without adequate caveats**
The introduction prominently features results from Countdown-16 and GSM8K-16 pilots (n=16) alongside Countdown-500 results (n=500) without clearly distinguishing their statistical reliability. The experiments section itself notes that "statistical tests have very low power at this sample size (minimum detectable effect ~25pp)," yet the introduction presents 12.5% and 37.5% pilot numbers as headline results.

*Suggestion*: Clearly label which results are pilot-scale when presenting them in the introduction. Add a brief caveat such as "on a 16-sample pilot (to be validated at scale)" or prioritize the Countdown-500 full-scale results and relegate pilot numbers to a parenthetical.

**H3. The contributions list (paragraph 7) is too granular for an introduction**
Four numbered contributions with sub-details make this read like an abstract rather than flowing narrative prose. Contributions (3) and (4) are essentially sub-points of the overall narrative.

*Suggestion*: Compress to 2-3 high-level contributions stated in prose form, not a numbered list. For example: "We contribute BSD and A-CFG as complementary training-free methods, establish A-CFG as the most promising general tool, and provide a systematic mapping of failure modes that constrains the design space."

### MEDIUM

**M1. Figure 1 description is appended awkwardly**
The figure description appears as a blockquote at the end of the section, after the paper outline paragraph. This breaks reading flow and suggests the figure placement was an afterthought.

*Suggestion*: Integrate the Figure 1 reference into the narrative -- ideally right after the information island problem is defined (paragraph 3), with a forward reference like "(Figure 1, left)." The figure description itself should be a proper caption, not a blockquote.

**M2. "Prior iteration" language is informal and self-referential**
Phrases like "A critical insight emerged from our prior iteration" and "our prior iteration" appear multiple times. This is appropriate for internal research diary language but not for a published paper. The reader does not know what "prior iteration" means.

*Suggestion*: Either cite DMI as a separate prior work (if published) or describe it as "a preliminary investigation" or "initial experiments." Remove all "iteration" language.

**M3. The paper outline paragraph (paragraph 8) is mechanical**
"Section 2 surveys... Section 3 presents... Section 4 reports..." is boilerplate that adds little value in a well-structured paper. Many top venues discourage or omit this.

*Suggestion*: Either remove entirely or compress to a single sentence: "We present our methods in Section 3, evaluate them in Section 4, and discuss failure modes in Section 5."

**M4. Missing comparison framing against AR inference-time scaling**
The introduction mentions AR inference-time scaling (chain-of-thought, best-of-N, tree search) but does not quantify the gap. How much does inference-time scaling help AR models on the same benchmarks? This would contextualize whether MDLM results are promising or disappointing.

*Suggestion*: Add one sentence comparing the magnitude of AR inference-time scaling gains on Countdown or GSM8K to ground the reader's expectations.

---

## Summary of Recommended Actions

| Priority | Issue | Action |
|----------|-------|--------|
| CRITICAL | C1 | Cut introduction length by 30-40%, move detailed results to later sections |
| CRITICAL | C2 | Add broader motivation in opening paragraph |
| HIGH | H1 | Restructure DMI/BSD/A-CFG narrative flow |
| HIGH | H2 | Clearly distinguish pilot vs full-scale results |
| HIGH | H3 | Compress contributions list to prose |
| MEDIUM | M1 | Integrate Figure 1 reference into narrative |
| MEDIUM | M2 | Remove "prior iteration" language |
| MEDIUM | M3 | Compress or remove paper outline paragraph |
| MEDIUM | M4 | Add AR scaling comparison for context |

---

## Verdict

The introduction contains all necessary content but needs significant restructuring for a top-venue submission. The core narrative (information island problem -> DMI insight -> BSD + A-CFG as solutions -> key results + failure modes) is sound, but it is buried under excessive detail that should be deferred to later sections. With trimming and reorganization, this could reach 8.5-9/10.
