# Writing Critique — Iteration 4

## Overall Assessment: MAJOR REVISION NEEDED

The paper is well-structured and honestly reports negative results, which is commendable. However, it suffers from a critical gap between its ambitious framing and its thin evidentiary base, along with several presentation issues that would likely lead to rejection at a top venue.

---

## 1. Title/Abstract Mismatch with Results

**Severity: CRITICAL**

The title "Beyond Remasking: Continuous Belief States and Classifier-Free Guidance for Inference-Time Reasoning" frames BSD and A-CFG as co-equal contributions. However:

- BSD's best result is 6.2% on Countdown-16 (n=16, 1/16 correct) — this is **one additional correct sample** over vanilla. The 95% CI is [0.0%, 18.8%], fully consistent with noise.
- A-CFG's results are also on n=16 pilot data.
- The only statistically validated result is DMI at 9.3% on Countdown-500 — a method from iteration 3 that is presented as a minor baseline.

The abstract should either: (a) honestly foreground DMI as the primary validated contribution and frame BSD/A-CFG as promising pilot-scale directions, or (b) acknowledge upfront that the main contributions are negative results and design space mapping.

## 2. Overclaiming on Pilot Data

**Severity: CRITICAL**

The paper repeatedly presents pilot-scale results (n=16) alongside full-scale results (n=500) without sufficient visual/textual separation. Specific problems:

- "A-CFG achieves Countdown +12.5pp and GSM8K +12.5pp" — this is 2/16 correct vs 0/16 on Countdown. The bootstrap 95% CI includes zero. Stating "+12.5pp" without immediately qualifying the sample size is misleading.
- Section 4.4 title "GSM8K Generalization" suggests confirmed generalization. It should be "GSM8K Generalization (Pilot, n=16)".
- The entropy-accuracy correlation r=0.78, p<0.001 on n=16 is suspicious — the p-value for a Pearson correlation with n=16 and r=0.78 should be approximately p=0.0004, which is plausible but the claim "p < 0.001" obscures the extreme fragility of this estimate with so few data points. With a binary outcome variable (correct/incorrect), this is really a point-biserial correlation, and 2 correct out of 16 means the estimate is driven by those 2 data points.

## 3. Framing Inconsistencies

**Severity: HIGH**

- The proposal (proposal.md) targets 14-18% accuracy for BSD and BSD+RACFG. The paper observes 6.2%. The discussion (Section 5.4) rationalizes this as "the bottleneck is real but narrow" without honestly confronting the gap between prediction and result.
- Section 5.5 claims methods map "what does and does not work" — but the only thing validated to work is DMI, which predates this iteration. BSD and A-CFG remain unvalidated at scale.
- The paper claims BSD "generalizes" DMI (Table 1, Section 3.2). But BSD (6.2%) performs worse than DMI (12.5%) on the same pilot benchmark. A method that strictly underperforms its "special case" is not a successful generalization.

## 4. Missing Full-Scale Validation

**Severity: CRITICAL**

The paper explicitly acknowledges this but structures the narrative as if pilot results are confirmatory. For a top venue:
- BSD needs Countdown-500 × 3 seeds evaluation
- A-CFG needs Countdown-500 × 3 seeds evaluation
- GSM8K needs at minimum 100+ samples per method
- The compute-fair comparison needs full-scale data (pilot shows vanilla step-scaling is competitive)

Without these, the paper is an extended pilot study, not a contribution paper.

## 5. Narrative Structure Issues

**Severity: MEDIUM**

- The paper introduces RACFG in the proposal/methodology as a key contribution, then immediately reveals it failed completely (0% everywhere). This wastes reader attention. Consider relegating RACFG to an appendix and leading with what works.
- The BSD+A-CFG combination failure is discussed in Section 5.1 with a post-hoc rationalization about "representational mismatch." This is speculative — with n=16, the combination scoring 1/16 vs A-CFG at 2/16 could easily be noise.
- Too many hypotheses (H1-H11) for a paper with such limited data. Several hypotheses (H3, H5, H6, H7) are "falsified" based on n=16 pilot data where any single sample flip would change the conclusion.

## 6. Missing Comparisons

**Severity: HIGH**

- No comparison with concurrent work mentioned in related work: wd1 (GSM8K 84.5%), HEX (GSM8K 88.1%), MetaState. These are mentioned as baselines in the proposal but absent from experiments.
- Dream-7B vanilla at 29.6% GSM8K (from summary.md) vs pilot 25.0% (from paper) — discrepancy not explained (different sample sets).
- A-CFG is credited as an original contribution but it is a direct application of the published A-CFG method (arXiv 2505.20199) to a different model. The novelty claim should be stated more carefully.

## 7. Statistical Presentation

**Severity: MEDIUM**

- McNemar test is applied to DMI vs vanilla on Countdown-500 but not to the pilot comparisons (correctly). However, the paper should explicitly compute the minimum detectable effect size at n=16 to show what the pilot *cannot* detect.
- The Bonferroni correction is mentioned but never applied because no pairwise test at pilot scale reaches significance.
- Bootstrap CIs for pilot accuracy: [-12.5%, 37.5%] for A-CFG on GSM8K — this *includes* substantial negative effects. The paper should state this explicitly rather than emphasizing the point estimate.

## 8. Figures

**Severity: MEDIUM**

- All figures are described via markdown placeholders, not actual rendered figures. This is understandable for a draft but makes evaluation difficult.
- Figure 1 inset shows "Accuracy comparison: Vanilla 4.7%, DMI 9.3%, BSD 6.2-12.5%, A-CFG 12.5%" — mixing full-scale (4.7%, 9.3%) and pilot (6.2%, 12.5%) numbers in the same figure is misleading.

## 9. Prose Quality

**Severity: LOW**

- Generally well-written and clear
- Some redundancy between Sections 3 and 4 in describing methods
- The related work section is comprehensive but could be more concise
- "Information island problem" is a good coinage but used too frequently (10+ times)
