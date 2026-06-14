# Writing Critique -- ComposeAccel iter_002

## Overall Assessment: 6/10

The writing is technically competent, well-structured, and unusually honest for an ML paper. The AR comparison, negative results reporting, and transparent acknowledgment of M1's implementation gap are strengths that most papers lack. However, the paper has a fundamental framing problem: the abstract and contributions promise more than the evidence delivers. Two missing figures, unresolved editorial artifacts, and a structural mismatch between pilot-scale data and full-scale claims weaken the manuscript.

---

## What Works Well

### Intellectual Honesty

The paper voluntarily reports:
- AR baseline (Qwen2.5-7B) dominates on both speed and accuracy (Section 5.2, Table 7)
- M1 speedup is projected, not measured, with explicit kernel-level failure documentation (Section 3.2)
- M2 (step scheduling) is a structural NO_GO, reported as a negative result (Section 6.5)
- IGSD confidence gate "provides zero measurable benefit at this operating point" (Section 4.4)
- M1+M3 shows destructive interference, overturning earlier pilot results (Section 6.1)

This level of honesty is rare and would be valued by reviewers. The paper does not claim DLM acceleration parity with AR -- instead framing the contribution as "mapping the composition design space." This is the correct framing.

### Clear Metric Definitions

Section 3.1 defines QAS, Ortho, AccRet, and the combined metric concisely with mathematical notation and an interpretation table. The thresholds (Ortho > 1.0 synergy, 0.8-1.0 near-orthogonal, < 0.8 interference) are explicit and consistently applied throughout the paper.

### Specific, Quantified Claims

Nearly every claim includes a specific number: "1.16x measured speedup" not "modest speedup"; "58.9% accuracy retention" not "significant degradation"; "15.2x framework overhead" not "substantial overhead." This specificity makes claims verifiable.

---

## Critical Issues

### C1: Abstract/Contributions Mismatch with Evidence

The abstract states: "M1+IGSD achieves near-orthogonal composition (Ortho = 0.96, 2.75x speedup at 58.9% accuracy retention on GSM8K)." A reader interprets this as "combining KV caching with step distillation yields 2.75x speedup." In reality:
- M1 contributes 1.16x (negligible) and IGSD contributes ~2.8x (almost all the speedup)
- Ortho=0.96 means "IGSD is not harmed by M1's overhead," not "two methods synergize"
- The 100-sample, single-seed data behind this claim has wide uncertainty

Contribution 3 frames IGSD as "A 50-line training-free step scheduler that partitions denoising into draft and refine phases using inter-step KL divergence." Section 4.4 shows the partition is inert (tau=0.0 = tau=0.9 at T_draft=32). The contribution text and the evidence contradict each other within the same paper.

**Fix**: Rewrite the abstract to lead with the interference finding (the strongest, most reliable result) rather than the M1+IGSD composition. Rewrite Contribution 3 to acknowledge the null ablation result.

### C2: Two Missing Figures Are Critical

Figure 2 (architecture diagram) has a [TODO] placeholder at line 231. This is the key visual for method comprehension. Without it, a reader must reconstruct the IGSD pipeline from text alone. A specification exists (fig2_architecture_desc.md) but was never rendered.

Figure 7 (KL divergence profile) has a [TODO] at line 324. This figure would visualize the falsification of H6 (inverted-U hypothesis). Raw data exists (igsd_kl_profiles_raw.json). The ablation analysis in Section 4.4 describes the KL profile in text ("monotonically decreasing, not the inverted-U shape") but the evidence is text-only.

Both figures have existing data or specifications. Their absence is an execution gap, not a design gap.

---

## Major Issues

### M1: Strength of Claims Does Not Match Scale of Evidence

Throughout the paper, composition verdicts are stated as definitive findings:
- "M1+IGSD is near-orthogonal" (from 100 samples, 1 seed)
- "M1+M3 shows destructive interference" (from 100 samples, 1 seed)
- "M3+IGSD is task-dependent" (from 100 samples, 1 seed)

The three-way results have 3-seed validation (a strength), but pairwise results do not. The paper should use hedged language for pilot-scale findings: "Pilot data suggests M1+IGSD is near-orthogonal (Ortho = 0.96, N=100, seed 42); full-scale validation is needed."

Compare with the batch sensitivity section (5.3), which is appropriately brief for its pilot-scale data. The pairwise sections should have similar hedging.

### M2: Section 5.3 (Batch Sensitivity) Is a Stub

Three sentences with no baseline comparison, no figure, and no interpretation. The numbers "batch=4: accuracy 50%, TPS 56" are uninterpretable without batch=4 baseline TPS. If the batch=4 baseline is 200 TPS (as expected from 4x parallelism), then 56 TPS is a 0.28x slowdown, which would be a major finding. If the batch=4 baseline is 60 TPS, it is a minor slowdown. The reader cannot tell.

Either expand this section with baseline measurements and Figure 8 (planned in outline), or move it entirely to the appendix with a note that batch sensitivity analysis is incomplete.

### M3: Table 3 Mixes Incompatible Sample Sizes

M1 (N=1319), IGSD (N=200), M3 (N=100) in the same table. The N column exists but a reader scanning QAS values will compare M1 QAS=0.98 against M3 QAS=1.69 without recognizing the 13x sample size difference. M3's higher QAS has narrower precision but this is not flagged.

**Fix**: Add a table footnote or use visual grouping (horizontal lines, bold, or shading) to separate sample-size groups. Or simply note: "Caution: M3 results are from 100-sample evaluation and carry wider uncertainty than M1 results."

### M4: Related Work Overclaims Novelty

Section 2.3 states: "For DLM inference acceleration, zero composition data exists." This is true for systematic factorial studies, but FlashDLM explicitly combines FreeCache with AR supervision -- a two-method composition. The paper acknowledges FlashDLM in the same section but distinguishes it as "one pre-designed combination." The claim should be: "No published work provides systematic pairwise composition analysis across DLM acceleration families."

### M5: Conclusion Repeats Abstract

The conclusion (Section 7) largely restates the abstract and key results without adding new insight. The three design principles (Section 6.3) are the most valuable output and should be featured more prominently in the conclusion. The conclusion should also explicitly state the statistical limitations of pilot-scale pairwise data.

---

## Minor Issues

### m1: Editorial Artifacts (Flagged by Review)
- Line 47: Parenthetical note about section renumbering
- Lines 233-237, 328-337: HTML comment blocks
- Lines 442-457: "Figures and Tables" index section
- Lines 231, 324: [TODO] tags

All must be removed before submission.

### m2: N_gen Undefined
First appears in the CHR formula (line 147). Never formally defined. Add: "where N_gen denotes the number of generation-token positions (excluding the prompt)."

### m3: r_accept vs. alpha Confusion
Table 3 uses r_accept (accept rate) while Section 3.3 text uses alpha (frozen fraction). notation.md defines both but their relationship is unclear. If they are the same quantity measured at different points, state this explicitly.

### m4: "Over 20 Training-Free Acceleration Methods"
Line 11. The paper should count them precisely if making this claim in the introduction. Table 1 lists 11. The text later mentions additional methods (KLASS, PRR, etc.). Counting explicitly would strengthen the opening.

---

## Structural Suggestions

1. **Lead the abstract with the interference finding**: "We find that composition of training-free DLM acceleration methods predominantly produces interference, not synergy. Of three pairwise combinations studied, one shows near-orthogonal composition, one is task-dependent, and one exhibits destructive interference." This is more honest and more interesting than the current framing.

2. **Add a "Statistical Limitations" subsection to Section 6.4**: Currently limitations mentions "pilot-scale" in one sentence. A dedicated subsection should state sample sizes, acknowledge the M1+M3 pilot/full-scale discrepancy, and report confidence intervals or at minimum standard errors for key Ortho values.

3. **Restructure Contributions to reflect reality**:
   - C1: Composition taxonomy with three distinct regimes (strong, supported by all data)
   - C2: Honest AR comparison establishing the remaining gap (strong)
   - C3: Empirical finding that naive step truncation is as effective as confidence-partitioned IGSD (moderately strong, interesting negative result)
   - C4: QAS/Ortho framework as methodological conventions for future work (minor)

4. **Move HumanEval to supplementary material**: Currently it occupies space in the experimental setup (Section 3.5) and is referenced in multiple places, but the 2.4% baseline makes it uninformative. A single sentence ("HumanEval results are in Appendix X; the 2.4% baseline provides insufficient signal for meaningful comparison") suffices in the main text.
