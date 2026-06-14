# Experiment Result Analysis

## Key Results Summary

All results are from PILOT mode (10 epochs, 100-sample subsets for Tier 1, single seed). No paired t-tests were possible (all t_stat fields are null).

**Tier 1 (3-op permutation, 6 orderings x 2 architectures x 2 datasets):**
- ViT-Small CIFAR-10: 2.32% spread (Flip->CJ->Crop: 0.1970 vs Crop->CJ->Flip: 0.1738)
- ResNet-18 CIFAR-10: 0.96% spread (CJ->Flip->Crop: 0.1097 vs CJ->Crop->Flip: 0.1001)
- ResNet-18 CIFAR-100: 0.88% spread (Flip->Crop->CJ: 0.4663 vs CJ->Flip->Crop: 0.4575)
- ViT-Small CIFAR-100: 0.25% spread (Flip->Crop->CJ: 0.0289 vs CJ->Flip->Crop: 0.0264)

**Hypothesis verdicts (pilot-only, all provisional):**
- H1 (ordering matters): 3/4 blocks with spread > 0.5% -- labeled "confirmed" but based on single seed
- H2 (reversibility-sorted wins): 2/4 blocks -- labeled "confirmed" but indistinguishable from chance at n=1
- H3 (NC_2 predicts accuracy): rho = -0.20, p = 0.68 -- **falsified** (wrong sign)
- H4 (MI/DPI): combined rho = -0.057, with contradictory signs across datasets -- **inconclusive**
- H5 (magnitude scaling): M5=0.35%, M9=0.88%, M14=0.00% -- **falsified** (non-monotonic, not monotonic)

**Tier 2 (6-op category ordering, CIFAR-10 5k pilot):**
- Interleaved P->G: 0.2939 vs all-geometric-first: 0.2038 -- 9.01% spread (strongest signal but highly suspect at pilot scale)

**Baselines (30 epochs, full dataset -- NOT comparable to ordering results):**
- Conventional: 91.91% (CIFAR-10 RN18) vs ordering experiments at 10-11% (CIFAR-10 RN18)
- This asymmetry renders direct comparison invalid

## Debate Perspectives Summary

- **Optimist**: Ordering effects are real and practically meaningful. The 2.32% ViT spread and 9.01% category-level spread are novel findings. Flip-first orderings consistently rank high; Crop-first consistently underperforms. The MI dataset-dependent sign flip is itself an interesting finding about task granularity. Recommends proceeding with full-scale validation, expects spreads to persist.

- **Skeptic**: Two fatal flaws: (1) all "confirmed" verdicts rest on 1 seed, 10 epochs, 100 samples with no statistical tests possible; (2) ResNet-18 CIFAR-10 accuracies (10-11%) are at random chance, and ViT CIFAR-100 (2.6-2.9%) is barely above random -- ordering comparisons at these accuracy levels are meaningless. The best ordering is inconsistent across blocks (characteristic of noise). The 9% Tier 2 spread is an extraordinary claim requiring extraordinary evidence. Demands full Tier 1 (200 epochs, 5 seeds) before any claim is made.

- **Strategist**: PROCEED with narrative reframing. H1 has strong signal in 3/4 blocks. The theoretical narrative must be rebuilt: NC bound and DPI principle both failed, so the paper must pivot from "theory-validated experiment" to "empirical discovery with theoretical analysis." Recommends Full Tier 1 (P1) + Full Tier 2 (P2) + class-level analysis (free) + abbreviated Tier 3. Total revised budget: 44.5 GPU-h, under the original 51 GPU-h estimate.

- **Comparativist**: No concurrent work threatens novelty as of April 2026. The question is novel and fills an explicitly documented gap. However, current evidence is from severely under-trained models and cannot support publishable claims. 2/3 theoretical predictions failed. Venue projection ranges from NeurIPS (if spread > 1% at convergence + theory recovery) to negative-results workshop (if spread < 0.3%). Full Tier 1 is the prerequisite for any submission.

- **Methodologist**: Reproducibility score 2/5. All hypothesis verdicts are premature. The asymmetric training conditions between orderings (10 epochs, 100 samples) and baselines (30 epochs, full dataset) make the main results table misleading. Cohen's d values are mathematically vacuous with n=1. Selection bias in Tier 3 (best/worst orderings chosen from noisy pilot). The methodology is well-conceived but execution is pilot-only.

- **Revisionist**: H2 was silently redefined from "architecture-differential sensitivity" to "reversibility-sorted wins." The original H2 remains unanswered. The best ordering flips between datasets: CIFAR-100 consistently favors geometric-first across both architectures, while CIFAR-10 shows no consistent pattern. Proposes three new hypotheses: (NH1) task difficulty modulates optimal ordering, (NH2) magnitude-ordering spread is inverted-U, (NH3) feature-space NC_2 recovers predictive power. Concludes: ordering matters but WHY it matters is not captured by either theoretical framework.

## Analysis

### 1. Method Feasibility

The core experimental method (controlled factorial permutation study) is sound and well-implemented. The infrastructure for running all tiers is validated. The pilot demonstrates that the system can execute all planned experiments. **The method itself is not in question -- only the scale of execution.**

### 2. Performance

At pilot scale, the results are ambiguous. Two of four architecture-dataset blocks (ResNet-18 CIFAR-10 at 10-11%, ViT-Small CIFAR-100 at 2.6-2.9%) show accuracy near random chance, making any ordering comparison in those blocks uninterpretable. The remaining two blocks (ViT-Small CIFAR-10: 2.32% spread at 17-20% accuracy; ResNet-18 CIFAR-100: 0.88% spread at 45-47% accuracy) show plausible ordering effects at above-chance accuracy, but these are single-seed estimates from undertrained models. No baseline comparison is meaningful because training conditions differ by 500x in data size and 3x in epochs.

### 3. Improvement Headroom

There is substantial improvement headroom through full-scale execution:
- Moving from 10 epochs/100 samples/1 seed to 200 epochs/full dataset/5 seeds will resolve all statistical ambiguities
- The Tier 2 category-level signal (9.01%) is the highest-potential finding; even shrinking 3-4x would yield a 2-3% practical contribution
- The narrative can be strengthened by reframing falsified theoretical predictions as findings rather than failures
- New hypotheses (NH1: task-difficulty modulation, NH3: feature-space NC_2) provide unexplored angles
- Budget allows 44.5 GPU-h of remaining experiments within the original 51 GPU-h allocation

### 4. Time-Cost Tradeoff

Continuing to full-scale experiments is clearly more efficient than pivoting:
- The experimental infrastructure is fully built and validated
- Full Tier 1 requires ~20 GPU-h (~10h wall-clock on 2 GPUs) -- a modest investment
- A pivot would require new idea generation, new experimental design, new infrastructure -- conservatively 3-5 days before reaching the same point
- The novelty window is open with no concurrent threats
- Even a null result at full scale is publishable (negative-results track + NC falsification)
- The backup plans (variance decomposition, class-level analysis) can be pursued in parallel with the main study at near-zero marginal cost

### 5. Critical Objections

The skeptic raises two genuinely fatal concerns about the CURRENT state:
1. Near-random-chance accuracy in 2/4 blocks makes those blocks uninterpretable
2. Single-seed results with no statistical tests cannot distinguish signal from noise

However, both concerns are addressable -- they are artifacts of pilot scale, not fundamental design flaws. Full-scale experiments directly resolve both: 200 epochs will push all blocks well above chance, and 5 seeds will enable paired t-tests with Bonferroni correction.

The methodologist's concern about asymmetric training conditions is also valid but self-correcting: full-scale ordering experiments will use the same training budget as baselines.

No perspective identified a fundamental flaw that would prevent full-scale experiments from producing credible results.

## Decision Rationale

**PROCEED** for the following evidence-backed reasons:

1. **The research question is novel and the gap is real.** All six perspectives agree unanimously. Two published surveys explicitly identify augmentation ordering as an open question. No concurrent work threatens priority as of April 2026. This alone justifies completing the study.

2. **The pilot evidence, while insufficient for publication, is consistent with a real effect in the blocks above random chance.** The 2.32% spread on ViT-Small CIFAR-10 and 0.88% on ResNet-18 CIFAR-100 occur at above-chance accuracy levels and motivate full-scale validation. The Tier 2 category-level 9.01% spread, even if it shrinks 3-4x, would be a significant practical finding.

3. **All pilot weaknesses are addressable through planned full-scale experiments, not a fundamental redesign.** The path from 3.5/10 to 7.5/10 quality is clear: full Tier 1 (20 GPU-h) + Tier 2 confirmation (1.5 GPU-h) + narrative reframing.

4. **The theoretical falsifications (H3, H5) are themselves publishable contributions.** The NC_2 measure's failure to predict ordering quality, and the non-monotonic magnitude relationship, are novel findings that add scientific value to the paper regardless of H1's outcome.

5. **The cost of proceeding (~44.5 GPU-h, ~23h wall-clock) is modest compared to the cost of pivoting** (new idea generation + design + infrastructure, 3-5 days minimum). The existing infrastructure, experimental framework, and novelty position are assets that would be abandoned by a pivot.

6. **The study is designed to be publishable regardless of outcome direction.** If H1 confirms at full scale, the paper has a strong empirical contribution. If H1 is null, the paper becomes "the first rigorous answer to a documented open question" plus NC/H5 falsification -- publishable at a workshop or negative-results track.

The mandatory condition for proceeding: **the narrative must be restructured from "theory-validated experiment" to "empirical discovery with negative theoretical results."** The NC bound and DPI reversibility principle cannot be the paper's leading contributions given the pilot falsifications.

## DECISION: PROCEED
