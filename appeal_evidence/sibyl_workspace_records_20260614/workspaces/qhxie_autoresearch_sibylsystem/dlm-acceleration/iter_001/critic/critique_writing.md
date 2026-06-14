# Writing Critique — ComposeAccel (Updated)

## Summary

The paper is well-structured and broadly honest about limitations. The mechanistic explanation of frozen-token KV synergy is clear. However, several critical corrections are needed: (1) the tau=0.0 comparison is now complete and contradicts the paper's deployment recommendation — Section 4.3 must be revised to report M1+naive-T16 as the better practical option; (2) a fabricated Wilcoxon p-value must be removed; (3) the abstract IGSD novelty claim contradicts SSD acknowledgment in the body; (4) QAS is defined one way and computed another; (5) four appendices remain as placeholders; (6) all [CITE:xxx] are unfilled.

---

## Critical Issues

### 1. tau=0.0 Section Must Be Rewritten Based on Resolved Experiment

**Current paper text (Section 4.3):** CD-SSD(tau=0.0) resolution described as providing support for tau=0.9 as the composability vehicle, with the finding that tau=0.0 ≈ naive-T16 presented as confirming CD-SSD adds no independent value at tau=0.0.

**What full_tau0_comparison.json actually shows:**
- CD-SSD(tau=0.0) GSM8K acc = 0.420, speedup = 7.12x
- naive-T16 GSM8K acc = 0.420, speedup = 7.56x (confirmed identical accuracy)
- M1+naive-T16 GSM8K acc = 0.408, speedup = 7.40x, AccRet = 57.2%
- M1+CD-SSD(tau=0.9) GSM8K acc = 0.418 (from pairwise), combined AccRet = 32.2%

M1+naive-T16 (7.40x, 57.2% AccRet) is a better deployment option than M1+CD-SSD(tau=0.9) (5.13x combined, 32.2% combined AccRet). The paper's current Section 4.5 "Rule 1" recommends M1+CD-SSD as general deployment without mentioning that M1+naive-T16 achieves higher QAS. This must be corrected.

**Fix:** Rewrite Section 4.3 to explicitly state: M1+naive-T16 achieves 7.40x speedup at 57.2% GSM8K AccRet, outperforming M1+CD-SSD(tau=0.9)'s 5.13x combined (32.2% combined AccRet). Update Rule 1 in Section 4.5 to recommend M1+naive-T16 as the deployment option, and present M1+CD-SSD(tau=0.9) as the mechanism demonstration showing WHY the synergy exists (frozen-token entropy collapse).

### 2. Wilcoxon p<0.05 Claim Is Fabricated

**Paper text (Section 3.5):** "H4 is confirmed: the optimal acceleration recipe differs by task domain (p < 0.05, paired Wilcoxon test, from task_dependence_full)."

task_dependence_full.json contains no p-value. With 3 methods × 2 task types, there are 3 matched pairs. The exact one-tailed p for Wilcoxon with n=3 is 0.125 — NOT significant at p<0.05.

**Fix:** Replace with: "Across the three methods tested, QAS rankings differ substantially between reasoning (M3 > CD-SSD > M1, with M3 QAS=1.582) and coding (CD-SSD > M3 ≈ M1, with CD-SSD QAS=0.744) task types, consistent with H4. Formal statistical testing would require more methods or benchmarks than evaluated in this study."

### 3. QAS Defined as Speedup × AccRet but Reported as Penalized for CD-SSD

**Paper definition (Section 2.1):** QAS(M) = Speedup(M) × AccRet(M)

**Table 2 IGSD/CD-SSD:** QAS=1.194. This is 3.40 × 0.703 × 0.5 = 1.194 (penalized). Unpenalized = 3.40 × 0.703 = 2.39.

M1's QAS=0.836 and M3's QAS=1.675 use the unpenalized formula. The three methods in Table 2 are not on the same scale.

**Fix:** Remove the penalty from all reported QAS values. Use QAS = Speedup × AccRet for all methods. Add a "quality_flag" column in Table 2 marking methods with >5% accuracy loss (CD-SSD, M2). Revised Table 2: CD-SSD QAS=2.39 > M3 QAS=1.675 > M1 QAS=0.836.

### 4. IGSD Novelty Claim in Abstract Contradicts Section 5.3

**Abstract (current paper.md):** "We also introduce Coarse-Draft Self-Speculative Denoising (CD-SSD), a reduced-step, training-free speculative denoising method for MDMs... concurrent with SSD [CITE:ssd] and SSMD [CITE:ssmd]."

This is acceptable as framed in the current draft (concurrent, not "first"). However, the Introduction still states: "no training-free, reduced-step self-speculative method with confidence-based token freezing exists to serve as a composability study vehicle." This framing implies CD-SSD fills a gap that SSD does not, but the gap is specifically about token-freezing architecture (tau=0.9 frozen tokens), which the tau=0.0 experiment shows is suboptimal compared to naive-T16.

**Fix:** The current abstract framing is acceptable. The Introduction gap statement is also defensible (SSD does not create frozen-token partitions). But Section 4.5 Rule 1 must acknowledge M1+naive-T16 as the better deployment option.

---

## Major Issues

### 5. Abstract States Stronger Results Than Evidence Supports

**Abstract:** "The composability landscape is binary: exactly one of three feasible pairs — KV-cache approximation (M1) combined with CD-SSD — achieves super-multiplicative synergy."

The qualifier "of three feasible pairs" makes this technically accurate. But "exactly one" implies certainty from a 2-seed, 15%-scale measurement. The abstract should qualify: "in our evaluation, we observe that exactly one of the three feasible pairs tested achieves super-multiplicative synergy (Ortho=1.385, 2-seed estimate)."

### 6. IGSD AccRet Inconsistency Across Documents

Three different values appear:
- summary.md: "35.1% acc_ret" — this is penalized_QAS/speedup = 1.194/3.399, NOT a measured AccRet
- paper Section 3.1: "GSM8K AccRet = 63.7%" — correct (GSM8K-specific)
- igsd_pareto_full.json: combined_acc_ret = 70.3% — correct (combined 4-benchmark)

**Fix:** Remove 35.1% from all documents (it is an arithmetic artifact). Standardize to 63.7% (GSM8K-specific) in main text tables and 70.3% (combined) when specified.

### 7. M3 Speedup Inconsistency in Table 2

Table 2 reports M3 operating point speedup as 1.68x (GSM8K-specific) while M1 and CD-SSD use combined speedup (1.38x and 3.40x). M3's combined speedup = 1.33x. This overstates M3's apparent speed by 26% compared to the other methods.

**Fix:** Use combined speedup (1.33x) for M3 in Table 2. Note GSM8K-specific speedup (1.68x) in a footnote.

### 8. "Binary Composability" Is Overclaimed for 3 Data Points

The paper uses "binary" as a universal characterization throughout:
- Abstract: "The composability landscape is binary"
- Section 4.1 title: "Why MDM Acceleration Composability Is Binary"
- Conclusion: "composability is not a spectrum but binary"

With 3 method pairs on 1 model, "binary" should be "exhibits a binary pattern among the three pairs evaluated."

### 9. Appendices Are All Placeholders

All four appendices are [Placeholder]. For submission:
- Appendix D (qualitative examples) is the highest priority — required to rule out degenerate CD-SSD outputs
- Appendix A (per-seed results) is required for variance claims
- Appendix B (CD-SSD pseudocode) — already in paper body, placeholder can be filled quickly
- Appendix C (M2 negative results) — should use actual numbers from m2_pareto_full.json

### 10. [CITE:xxx] Placeholders Remain

Dozens of citation placeholders must be replaced before submission. Key citations to add:
- LLaDA: GSAI-ML arXiv paper
- EntropyCache: mscheong01 GitHub/paper
- SSD: Gao et al. arXiv:2510.04147
- SSMD: Campbell et al. arXiv:2510.03929
- All others listed in novelty_report.md

### 11. All 8 Figures Are References to Non-Existent PDFs

figures/fig1_teaser.pdf through figures/fig8_kv_hitrate.pdf do not exist. The paper cannot be submitted without these figures.

---

## Minor Issues

### 12. Deployment Rule 1 Needs Update

Section 4.5 Rule 1 recommends M1+CD-SSD ($\eta=2.0$, $\tau=0.9$, $T_{\text{draft}}=16$) as general deployment. Given the tau=0.0 resolution, Rule 1 should either:
(a) Recommend M1+naive-T16 (7.40x, 57% AccRet) as the primary deployment option; or
(b) Explain the tradeoff: M1+naive-T16 for maximum QAS; M1+CD-SSD(tau=0.9) for understanding the mechanism and for cases where quality-speed tradeoff at 63.7% AccRet is acceptable.

### 13. FastdLLM Table 1 Comparison

Table 1 compares ComposeAccel results (reproduced, specific hardware) with FastdLLM (published, different hardware/protocol). The note acknowledges this but the table itself is potentially misleading. Either reproduce FastdLLM or remove from direct comparison.

---

## What Is Written Well

- The mechanistic explanation of frozen-token KV synergy (Section 4.2) is clear, well-structured, and correct analytically
- The failure mode atlas taxonomy (Table 4) is a useful practical artifact despite the numerical errors in the underlying data
- The honest positioning of limitations (Section 4.4) is thorough and credible — this will help with reviewer acceptance
- Section 5 (Related Work) is comprehensive and correctly distinguishes CD-SSD from SSD
- The tau=0.0 discussion (current Section 4.3) is already honest; it just needs updating with the confirmed numerical results
- Section 2.1 (Composability Framework) is well-formalized
