# Result Debate Synthesis: Augmentation Ordering Study

**Synthesizer**: Senior Research Director
**Date**: 2026-04-02
**Input**: 6 result debate perspectives (Optimist, Skeptic, Strategist, Methodologist, Comparativist, Revisionist)

---

## 1. Consensus Map

The following points command broad agreement across all 6 perspectives (high-confidence conclusions):

### 1.1 All results are pilot-only — no finding is statistically confirmed

**Unanimous**: Every perspective flags that Tier 1 experiments used 10 epochs, 100-sample training subsets, and a single seed. No paired t-tests were possible (t_stat is null for all blocks). The Cohen's d values reported (2.27–2.71) are mathematically vacuous with n=1 seed. All "confirmed" verdicts are provisional pilot findings, not publication-quality conclusions.

### 1.2 The research question is novel and the gap is unoccupied

**Unanimous**: No prior work isolates augmentation operation ordering as the sole independent variable in a controlled factorial experiment. Two independent surveys (Cheung & Yeung, IEEE TNNLS 2023; Yang et al., KAIS 2023) explicitly identify ordering as an open question. No concurrent work threatens the novelty window as of April 2026.

### 1.3 Full-scale Tier 1 is the single highest-priority action

**Unanimous**: 6 orderings × 2 architectures × 2 datasets × 5 seeds × 200 epochs. Without this, nothing is publication-quality. All perspectives converge on this as the non-negotiable prerequisite.

### 1.4 H3 (NC_2 predicts accuracy) is the clearest falsification in the pilot data

**Unanimous**: Spearman rho = −0.20, p = 0.68. The theoretical Wasserstein non-commutativity measure does not predict accuracy rankings, and the sign is in the wrong direction. Even if NC_2 estimation had more power (larger samples, exact W_2 rather than SWD proxy), the conceptual gap between pixel-space distributional distance and feature-space learning dynamics is real. The comparativist, revisionist, skeptic, methodologist, and strategist all agree this theory does not drive the paper.

### 1.5 H5 (magnitude monotonically amplifies ordering spread) is falsified

**Unanimous**: M5 = 0.35%, M9 = 0.88%, M14 = 0.00%. The collapse at M14 is a robust signal that extreme augmentation creates a noise floor where ordering is irrelevant. The non-monotonic pattern (inverted U) is agreed upon across all perspectives, though the skeptic notes it could also reflect insufficient training duration at M14.

### 1.6 The Tier 2 category-level pilot signal (9.01% spread) is the single strongest number but requires validation

**Unanimous**: All perspectives acknowledge the interleaved P→G ordering (29.39%) vs. all-geometric-first (20.38%) gap is striking. All perspectives also flag it as potentially inflated by 5k-sample, 10-epoch, 1-seed pilot conditions, and call for full-scale validation before any claim is made.

---

## 2. Conflict Resolution

### Conflict 1: Is the ordering effect real or noise? (Optimist vs. Skeptic/Methodologist)

**Optimist's position**: 3/4 blocks show spread > 0.5% at pilot; Tier 0 showed 2.68% spread. The direction is real.

**Skeptic/Methodologist's position**: ResNet-18 CIFAR-10 accuracies (10–11%) are near random chance. You cannot extract ordering signal from models that have not learned. Ranking reversals between Tier 0 and Tier 1 further undermine confidence.

**Judgment**: The skeptic and methodologist land on a fatal methodological issue that the optimist underweights. The 10–11% accuracy on CIFAR-10 is at the 10% random-chance floor for 10 classes; differences at this accuracy regime genuinely cannot be attributed to augmentation ordering. However, CIFAR-100 ResNet-18 (45.75%–46.63%) and ViT-Small CIFAR-10 (17.38%–19.70%) are above chance, and the 0.88% and 2.32% spreads there are more interpretable. **The signal is plausible but unconfirmed. H1 remains a credible working hypothesis, not a confirmed finding.** The 2/4-block ranking (excluding the clearly near-random CIFAR-10 ResNet-18 block) gives moderate confidence in the direction, pending full-scale validation.

### Conflict 2: Is H2 (DPI reversibility principle) confirmed or noise? (Optimist vs. Skeptic/Revisionist)

**Optimist's position**: 2/4 blocks show reversibility-sorted ordering outperforming conventional. The 2/4 win rate is expected for a nuanced principle.

**Skeptic's position**: 2/4 is exactly chance level. The wins are in the near-random-accuracy regime; the losses are at marginally better accuracy. There is no consistent pattern.

**Revisionist's finding**: H2 was silently redefined from "architecture-differential ordering preferences" to "reversibility-sorted wins" — a different hypothesis. The original H2 (CNNs sensitive to photometric, ViTs sensitive to geometric) is unanswered.

**Judgment**: The optimist overstates. With n=1 seed at near-chance accuracy, the 2/4 win rate is indistinguishable from chance. The best ordering flipping between blocks (CJ→Flip→Crop best on CIFAR-10 ResNet, Flip→Crop→CJ best on CIFAR-100 both architectures) is more consistent with noise than with a principled effect. **H2 (reversibility-sorted) is inconclusive, not confirmed.** The original architecture-differential question remains open and is the more scientifically interesting formulation.

### Conflict 3: Is the MI/DPI dataset sign-flip informative or noise? (Optimist vs. Skeptic)

**Optimist's position**: CIFAR-10 rho = +0.54, CIFAR-100 rho = −0.66 suggests a genuine task-granularity interaction where MI preservation helps for coarse tasks and hurts for fine-grained ones.

**Skeptic's position**: With 6 data points and non-significant p-values, opposite signs are indistinguishable from random noise. The InfoNCE estimator is unreliable at pilot scale.

**Judgment**: The sign-flip pattern is intriguing but statistically unsupported. The revisionist's note that the best ordering on CIFAR-100 is consistently spatial-first (across both architectures) is a separate observation that independently points toward a difficulty-dependent mechanism. Together, these form a promising new hypothesis (NH1: task difficulty modulates optimal ordering) worth testing — but the current MI evidence is not credible enough to treat as the theoretical explanation.

### Conflict 4: Can the paper proceed to writing without full-scale Tier 1? (Strategist vs. Methodologist/Comparativist)

**Strategist's position**: PROCEED with narrative reframing. Tier 2 pilot signal justifies proceeding to full experiments.

**Methodologist/Comparativist's position**: The paper has zero credible evidence in the current state. Full Tier 1 is the prerequisite.

**Judgment**: Both perspectives agree on the same action (run full Tier 1). The disagreement is about framing. **The correct stance is: PROCEED to full-scale experiments immediately, but do not proceed to writing until full Tier 1 results are in hand.** The narrative reframing recommended by the strategist is correct and should be applied to the writing phase.

---

## 3. Result Quality Score

**Score: 3.5 / 10**

**Justification**:
- Positive contributors (+3.5): The research question is novel and well-designed (+1.5). The pilot infrastructure is functional and the experimental design is sound (+1.0). Tier 2 pilot shows a tantalizing 9.01% signal that motivates full-scale validation (+0.5). The falsification of H3 and H5 are clean results that provide scientific value even as negative findings (+0.5).
- Negative contributors (−6.5): All results are from a severely underpowered pilot with 1 seed, 10 epochs, and 100-sample subsets (−2.0). Two of four architecture-dataset blocks have accuracy at or near random chance, making ordering comparisons uninterpretable (−2.0). All hypothesis verdicts are premature: H1/H2 are labeled "confirmed" without statistical power; H3/H5 are labeled "falsified" with underpowered proxy metrics (−1.5). Baseline comparisons are invalid due to asymmetric training conditions (ordering: 10 epochs/100 samples vs. baselines: 30 epochs/full dataset) (−1.0).

**Path to a 7.5/10**: Full-scale Tier 1 with 5 seeds + paired t-tests + at least 2/4 blocks showing statistically significant spread > 0.5%.

---

## 4. Key Findings

The following statements represent what the current evidence reliably supports:

1. **Augmentation ordering produces detectable accuracy differences at pilot scale, but these are unconfirmed at full training.** The 2.32% spread on ViT-Small CIFAR-10 and 0.88% on ResNet-18 CIFAR-100 are the most credible numbers; the 0.96% on ResNet-18 CIFAR-10 is near random chance and should be discounted. Whether these differences survive 200 epochs and 5 seeds is the open empirical question.

2. **Both main theoretical frameworks (NC_2 Wasserstein bound and DPI/MI reversibility) fail to predict pilot ordering rankings.** This is a useful negative result that reshapes the paper's theoretical contribution: the NC formalism correctly identifies that operations do not commute, but cannot tell you which ordering is better for learning. The gap between pixel-space distributional distance and feature-space learning dynamics is real and understudied.

3. **Augmentation magnitude has a non-monotonic relationship with ordering sensitivity.** M5 spread < M9 spread > M14 spread (which collapses to 0%). Extreme augmentation creates a noise floor that overwhelms ordering differences. This is a clean, interpretable finding from the magnitude sweep.

4. **The best ordering is dataset-dependent.** CJ→Flip→Crop wins on CIFAR-10 ResNet-18, but Flip→Crop→CJ wins on CIFAR-100 for both architectures. This pattern, if confirmed at full scale, is more consistent with a task-difficulty-dependent mechanism than with any universal principle derived from the transforms alone.

5. **Category-level ordering (geometric vs. photometric interleaving) shows a 9.01% pilot spread — the largest signal in the study — but it is also the most likely to be inflated by undertraining.** This is the highest-variance finding: if validated, it is the headline result; if it collapses at full scale, the paper's practical contribution narrows significantly.

---

## 5. Methodology Gaps

Critical experimental improvements needed before any submission:

### Gap 1 (BLOCKING): Full Tier 1 with 5 seeds and 200 epochs
- **Priority**: BLOCKING for all other work
- **Detail**: 6 orderings × 2 architectures × 2 datasets × 5 seeds × 200 epochs = 120 runs (~20 GPU-h)
- **Required output**: Paired t-tests (same seed, different ordering) with Bonferroni correction; Cohen's d from between-seed variance; training curves to observe whether ordering effects emerge early and persist or emerge late
- **Pass criteria**: At least 2/4 blocks with paired t-test p < 0.05 after correction and spread > 0.3%

### Gap 2 (CRITICAL): Match training conditions between orderings and baselines
- All ordering experiments must be run at full-dataset, full-epoch scale before comparison with baselines
- The current Table 1 (ordering results at 10–11% vs. baselines at 92%) is methodologically unsound and will immediately disqualify the paper upon review

### Gap 3 (HIGH): Tier 2 category-ordering confirmation pilot
- Before committing 18 GPU-h to full Tier 2, run a 1.5-GPU-h confirmation: full CIFAR-10, 50 epochs, 2 seeds, ResNet-18 only
- This will reveal whether the 9.01% pilot signal is a real effect or an early-training artifact

### Gap 4 (MODERATE): NC_2 recomputation with larger samples and correct metric
- Re-run Tier 4a with 10k samples and 1000 projections (as originally planned, vs. 100/100 in the pilot)
- Optionally test feature-space NC_2 (NH3): compute SWD in penultimate-layer feature space of a pretrained ResNet to test whether the pixel-space NC_2 failure is a proxy issue or a conceptual failure

### Gap 5 (MODERATE): H2 original formulation needs dedicated test
- The architecture-differential hypothesis (CNNs sensitive to photometric ordering, ViTs sensitive to geometric) was silently replaced in the final summary
- Full Tier 1 results can answer this with a three-way interaction analysis (architecture × ordering-type × dataset)
- This is the more scientifically interesting version of H2

---

## 6. Competitive Position

**Novelty of question**: Strong. No prior work studies augmentation ordering as the sole independent variable in a controlled factorial experiment. Two published surveys explicitly cite this as an open question.

**Novelty of theory**: Moderate, diminished by pilot data. The NC bound and DPI principle are novel theoretical contributions, but both frameworks fail to predict empirical outcomes in the pilot. The paper's theoretical contribution will need to pivot to: (a) explanation of WHY the NC formalism fails, or (b) a new empirical regularity (task-difficulty-dependent ordering) that motivates a revised theoretical frame.

**Concurrent work risk**: Low. The closest related work (Li et al. 2024, tree-structured augmentation composition) addresses structural topology, not ordering within a fixed sequence. No paper found that directly scoops this work as of April 2026.

**Contribution margin vs. SOTA**: Unknown until full-scale results. Conservative estimate from pilot data:
- If Tier 1 spread holds at 0.5–1.5% at convergence: ~0.5–1.5 pp above conventional ordering, comparable to TrivialAugment's margin over baseline
- If Tier 2 category signal holds at 2–3%: headline practical contribution comparable to RandAugment margins
- If both Tier 1 and Tier 2 shrink below 0.3%: workshop-level result

**Venue projection** (contingent on full-scale results confirming H1):
- Spread > 1% at convergence + at least partial theoretical recovery: NeurIPS/ICML (strong empirical + theoretical framing)
- Spread 0.5–1% + NC still fails: ICCV/ECCV or AAAI
- Spread 0.3–0.5%: CVPR Workshop or negative-results track
- Spread < 0.3%: NeurIPS negative-results workshop

---

## 7. Hypothesis Update

| Hypothesis | Original Status | Revised Status | Action |
|---|---|---|---|
| H1: Ordering affects accuracy | "Confirmed" (premature) | **Credible working hypothesis** | Run full Tier 1 to resolve |
| H2 (DPI/reversibility): reversibility-sorted wins | "Confirmed" (premature) | **Inconclusive** | Test at full scale; restore original architecture-differential formulation |
| H2 (original): Architecture-differential sensitivity | Not tested | **Open question** | Full Tier 1 with 3-way interaction test |
| H3: NC_2 predicts accuracy ranking | "Falsified" | **Falsified (medium confidence)** | Recompute with larger samples; test feature-space variant; frame as negative result if confirmed |
| H4: MI/DPI principle | "Inconclusive" | **Inconclusive, likely spurious** | The sign-flip across datasets is unexplained; do not build theoretical story on this until full-scale MI estimation is possible |
| H5: Magnitude monotonically amplifies spread | "Falsified" | **Falsified (high confidence)** | Accept falsification; reframe as inverted-U effect (NH2); add 2 intermediate magnitude levels to confirm |
| NH1: Task difficulty modulates optimal ordering | New | **Promising hypothesis** | CIFAR-100 data (geometric-first wins both architectures) is consistent; test via difficulty-varied experiment |
| NH2: Magnitude-ordering spread is inverted-U | New | **Consistent with pilot data** | Test with M=1,3,5,7,9,11,14 sweep |
| NH3: Feature-space NC_2 recovers predictive power | New | **Mechanistic bridge hypothesis** | Test with frozen pretrained backbone; low cost (<1 GPU-h) |

---

## 8. Action Plan

### VERDICT: PROCEED — with mandatory pipeline restructuring

The research question is novel, the gap is real, and the pilot evidence (while insufficient) is consistent with a real ordering effect on the blocks that have moved beyond random chance. The theoretical frameworks have largely failed, but this failure is itself scientifically valuable and clearly reportable. There is no evidence of fundamental design flaws in the experimental setup that would prevent full-scale validation from yielding credible results.

### Phase 1: Unblock (Immediate, before any other work)

**Step 1A — Tier 2 confirmation pilot** (1.5 GPU-h, ~45 min wall-clock)
- Full CIFAR-10, 50 epochs, 2 seeds, ResNet-18, 5 category orderings
- Decision gate: if spread > 2%, proceed to full Tier 2; if < 1%, deprioritize Tier 2 and focus resources on Tier 1
- Rationale: The 9% pilot signal is the highest-variance claim; resolve its fate cheaply before committing 18 GPU-h

**Step 1B — Full Tier 1** (20 GPU-h, ~10h on 2 GPUs)
- 6 orderings × 2 architectures × 2 datasets × 5 seeds × 200 epochs
- Run Tier 2 confirmation pilot first; if gate passes, run Tier 1 and Tier 2 in parallel
- Required outputs: per-block paired t-tests, Cohen's d from between-seed variance, training curves per ordering, final accuracy rankings with confidence intervals

### Phase 2: Secondary experiments (after Tier 1 in hand)

**Step 2A — Full Tier 2 if gate passes** (18 GPU-h)
- 5 category orderings × 2 architectures × 2 datasets × 5 seeds × 200 epochs

**Step 2B — Class-level analysis on Tier 1 results** (0 GPU-h)
- Free analysis; per-class accuracy tracking on CIFAR-100 may reveal class-conditional ordering effects
- Adds depth to the paper at zero marginal cost

**Step 2C — Abbreviated Tier 3** (4 GPU-h, not 12)
- Run M5, M7, M9, M11, M14 with 2 seeds
- Sufficient to establish the inverted-U curve shape; report magnitude non-monotonicity as a finding
- Skip full 5-seed Tier 3 (save 8 GPU-h)

**Step 2D — NC_2 re-estimation** (<1 GPU-h, CPU)
- Recompute with 10k samples and 1000 projections
- Additionally test feature-space NC_2 (NH3) using frozen ResNet-18 backbone features
- Decision: if rho > 0.4, partial theory recovery; if still negative, report as confirmed falsification

### Phase 3: Narrative construction (after full-scale results)

**Mandatory narrative reframe** (regardless of full-scale results):
- Drop "theory-validated-by-experiment" as the primary framing
- Lead with empirical discovery: ordering produces measurable accuracy differences; ViTs more sensitive than CNNs; category-level ordering dominates 3-op permutation effects
- Report NC_2 and H5 falsifications as findings, not failures: "The standard non-commutativity measure does not predict which orderings are better for learning; we show WHY and propose a refined measure"
- If task-difficulty dependence (NH1) is confirmed, promote to main finding with theoretical interpretation

**Paper structure (provisional)**:
1. Introduction: documented gap + pilot evidence that ordering matters
2. Theoretical analysis: NC bound (define and analyze), DPI principle (define and analyze), show why pixel-space NC fails as predictor
3. Experiments: Tier 2 category ordering (headline result, if validated), Tier 1 permutation study, magnitude interaction, architecture comparison
4. Analysis: Task-difficulty modulation, feature-space NC_2, why NC fails
5. Conclusion: Practical recommendations, limitations, future directions

### Resource summary

| Task | GPU-h | Priority |
|------|-------|----------|
| Tier 2 confirmation pilot | 1.5 | Immediate (gate decision) |
| Full Tier 1 | 20 | P1 (BLOCKING) |
| Full Tier 2 (if gate passes) | 18 | P2 |
| Class-level analysis | 0 | P3 (after Tier 1) |
| Abbreviated Tier 3 | 4 | P4 |
| NC_2 re-estimation + NH3 | 1 | P5 |
| **Total** | **44.5** | Under original 51 GPU-h budget |

---

## 9. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Tier 1 spread shrinks below 0.3% at full scale | Low | CRITICAL (nullifies main claim) | Design statistical test in advance; if null, pivot to "first principled null result + NC falsification" framing for workshop submission |
| Tier 2 9% signal collapses at full scale | Medium-High | HIGH (loses headline result) | Tier 2 confirmation pilot (Step 1A) provides early warning at low cost |
| NC_2 remains negative with larger samples | Medium | LOW (already framed as negative result) | Pursue feature-space NC_2 (NH3) as alternative; if both fail, the NC falsification is itself publishable |
| Ordering inconsistency across blocks persists at full scale | Medium | MODERATE (weakens universal recommendation) | Frame dataset-conditioned recommendation as finding; test NH1 (task-difficulty modulation) |
| Reviewer dismissal as "just an ablation study" | Medium | MODERATE | Lead with surveys' explicit identification of gap; provide information-theoretic framing; cite practitioner impact |
