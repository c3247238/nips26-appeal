# Strategist Analysis: Augmentation Ordering Study

**Date**: 2026-04-02
**Stage**: Post-pilot result debate
**Verdict**: **PROCEED** — with significant reframing of the theoretical narrative

---

## 1. Signal Strength Assessment

| Result | Signal Strength | Metric Delta | Justification |
|--------|----------------|--------------|---------------|
| H1: Ordering affects accuracy | **Strong** | 0.25%–2.32% spread across 4 blocks | 3/4 arch-dataset blocks exceed 0.5% spread. The 2.32% spread for ViT-Small on CIFAR-10 is large enough to survive scaling to full training. Effect direction is consistent across seeds in Tier 0. |
| H2: Reversibility-sorted ordering outperforms conventional | **Moderate** | Wins 2/4 blocks | Mixed signal. CJ-first wins on ResNet-18/CIFAR-10 (0.96%) but Flip-first wins on CIFAR-100 for both architectures. The DPI story is partially supported but not clean. |
| H3: NC_2 predicts accuracy ranking | **Noise** | rho=-0.200, p=0.683 | The Wasserstein NC proxy does not predict which orderings perform best. The SWD-based NC_2 values are distinguishable across pairs (Crop-CJ > Crop-Flip > Flip-CJ) but this ranking does not map to accuracy differences. This is a clear falsification. |
| H4: MI correlates with accuracy | **Weak** | combined rho=-0.057 | Contradictory sign across datasets (rho=+0.54 CIFAR-10 vs rho=-0.66 CIFAR-100). Neither is significant. The InfoNCE MI estimator may be too noisy at pilot scale, but the opposite signs are a bad sign. |
| H5: Magnitude amplifies ordering spread | **Noise** | M14 spread=0.00 | Clear falsification. At high magnitude (M14), aggressive augmentation overwhelms ordering effects entirely — both orderings converge to identical accuracy (0.245). The non-monotonic pattern (M5=0.35%, M9=0.88%, M14=0.00%) suggests a "sweet spot" rather than monotonic scaling. |
| Tier 2: Category-level ordering (6 ops) | **Strong** | 9.01% spread | Interleaved P-G ordering (0.2939) massively outperforms all-geometric-first (0.2038). This is the single strongest signal in the entire pilot. However, this is a 5k-sample, 10-epoch pilot on ResNet-18 only — the 9% spread is suspiciously large and likely inflated by the underpowered setting. |

---

## 2. Opportunity Cost Analysis

| Direction | GPU Cost | Expected Info Gain | Info Gain per GPU-hour |
|-----------|----------|-------------------|----------------------|
| Full Tier 1 (3-op, 200 epochs, 5 seeds, 4 blocks) | 20 GPU-h | **High** — definitively resolves H1 and H2 with statistical power | 1.0 (reference) |
| Full Tier 2 (6-op category ordering, 200 epochs) | 18 GPU-h | **Very High** — the 9% pilot signal is the strongest finding; validating it at scale would be the paper's headline result | 1.5 |
| Tier 3 full (magnitude interaction, 3 levels × 5 seeds) | 12 GPU-h | **Low** — H5 is already falsified; full run will confirm null with more precision but adds little narrative value | 0.3 |
| Tier 4 full (NC + MI, 10k samples) | 1 GPU-h | **Low-Medium** — H3 is falsified; the full NC measurement confirms the negative result but the SWD proxy itself may be the wrong metric | 0.5 |
| Revised NC metric (e.g., KL-based, feature-space NC) | 2 GPU-h | **Medium** — if SWD is the wrong proxy, a feature-space NC might rescue H3; but this is speculative | 0.5 |
| Class-level analysis (Backup B, zero extra training) | 0 GPU-h | **Medium** — could reveal that aggregate null masks meaningful per-class effects; free analysis on existing runs | infinite (free) |

---

## 3. Decision Matrix

| Direction | Signal | GPU Cost | Risk | Expected Outcome |
|-----------|--------|----------|------|-----------------|
| Full Tier 1 (3-op factorial) | Strong (H1) | 20h | Low | Confirms H1 with publication-quality stats. Expected spread 0.3-1.5% after 200 epochs. Necessary for any paper. |
| Full Tier 2 (6-op category ordering) | Strong (9% pilot) | 18h | **Medium-High** — the 9% may collapse at scale | If validated at even 2-3%, becomes the paper's strongest result. High variance risk: 10-epoch pilot on 5k samples is severely underpowered. |
| Full Tier 3 (magnitude) | Noise | 12h | Low (will confirm null) | Saves 12 GPU-hours for reallocation. Report H5 falsification with pilot data + brief confirmation run (2 seeds instead of 5). |
| Revised theory (rescue H3) | Noise for current metric | 2h | High (speculative) | Only worth attempting if the paper pivots to "theory-grounded" framing. With H3 falsified, the NC bound becomes a negative contribution. |
| Class-level analysis (Backup B) | Unknown | 0h | Low | Free analysis that could add a section to the paper regardless of aggregate results. |

**Dominant strategy**: Full Tier 1 + Full Tier 2 + Class-level analysis + Abbreviated Tier 3

---

## 4. PIVOT vs PROCEED Verdict

### **PROCEED** — but with mandatory narrative reframing

**Rationale**:
- H1 (ordering matters) has strong signal across 3/4 blocks, with the largest effect (2.32%) on ViT — this alone is a publishable finding that fills a documented gap.
- Tier 2 category-level ordering shows a 9% pilot spread — even if this shrinks 3-4x at full scale, a 2-3% effect from simply reordering augmentation categories is a strong practical contribution.
- The study is designed to be publishable regardless of outcome direction.

**However, the theoretical narrative must be rebuilt**:
- H3 (NC_2 predicts accuracy) is falsified. The Wasserstein NC bound — pitched as a core novel contribution in the proposal — does not work empirically. The paper cannot lead with this.
- H5 (magnitude scaling) is falsified. The interaction story is weaker than hoped.
- H4 (MI correlation) is inconclusive with contradictory signs across datasets.

**The original proposal sells 3 theoretical contributions (NC bound, DPI principle, algebraic classification). Two are falsified/inconclusive. The paper must pivot from "theory-validated-by-experiment" to "empirical-discovery-with-theoretical-analysis".**

---

## 5. Recommended Next Steps (Priority Order)

### Priority 1: Full Tier 1 — 3-op factorial (20 GPU-h, ~10h wall-clock)
- 6 orderings x 2 architectures x 2 datasets x 5 seeds x 200 epochs
- This is the non-negotiable core of the paper. Without it, nothing is publication-quality.
- **Expected outcome**: H1 confirmed with paired t-tests and effect sizes. Ordering spread of 0.3-1.5% after full training (smaller than pilot's 0.96-2.32% because pilot is underpowered at 10 epochs).

### Priority 2: Full Tier 2 — 6-op category ordering (18 GPU-h, ~9h wall-clock)
- 5 category orderings x 2 architectures x 2 datasets x 5 seeds x 200 epochs
- The pilot's 9% spread is the strongest signal. Even if it shrinks to 2-3% at full scale, this is the **headline finding**: interleaved photometric-geometric ordering beats the universal geometric-first convention by a meaningful margin.
- **Risk management**: Run a quick Tier 2 confirmation pilot first (full CIFAR-10, 50 epochs, 2 seeds, ResNet-18 only) to check if the 9% signal survives before committing 18 GPU-hours. Cost: ~1.5 GPU-h, 45 min wall-clock.

### Priority 3: Class-level analysis (0 GPU-h)
- Run on Tier 1 results once available.
- Per-class accuracy tracking on CIFAR-100 to detect class-conditional ordering effects.
- Zero additional training cost. Adds depth to the paper.

### Priority 4: Abbreviated Tier 3 — magnitude confirmation (4 GPU-h instead of 12)
- Run only M5 and M9 levels (skip M14 which converges to null) with 2 seeds instead of 5.
- Sufficient to report "moderate magnitude shows largest ordering sensitivity; extreme magnitude overwhelms ordering effects" as a brief section.

### Deprioritize:
- **Full Tier 4** (NC measurement at 10k scale): Only run after Tier 1/2 if there is remaining GPU budget. The falsified NC correlation means this is supporting material, not a core contribution.
- **Revised NC metric**: Do not invest in rescuing H3 at this stage. If the paper pivots to empirical framing, the NC falsification becomes a valuable negative result ("standard non-commutativity measures do not predict ordering sensitivity").

---

## 6. Narrative Reframing Strategy

### Original pitch (no longer viable as primary framing):
"Theory-grounded study: NC bound + DPI principle predict ordering effects"

### Recommended new pitch:
**"Order Matters: The First Systematic Study of Augmentation Pipeline Sequencing"**

Lead with the empirical discovery:
1. **Main finding**: Augmentation ordering produces 0.5-2.5% accuracy differences (Tier 1), and category-level ordering (geometric vs. photometric) produces even larger effects (Tier 2). The conventional geometric-first ordering is suboptimal.
2. **Architecture interaction**: ViTs are more sensitive to ordering than CNNs (2.32% vs 0.96% on CIFAR-10). This is architecturally grounded — patchification amplifies spatial ordering effects.
3. **Negative theoretical result**: The Wasserstein non-commutativity measure does NOT predict which orderings are better (rho=-0.20), despite correctly identifying that transforms do not commute. This reveals a gap between "these operations don't commute" and "this ordering is better for learning."
4. **Partial DPI support**: Reversibility-sorted ordering wins in 2/4 blocks, providing suggestive but not definitive support for the information-theoretic principle.
5. **Practical recommendation**: Use interleaved photometric-geometric ordering or reversibility-sorted ordering instead of the universal geometric-first default.

### Paper structure adjustment:
- Move the NC bound from "core contribution" to "Section 3: Theoretical Analysis" with an honest assessment of its empirical limitations
- Lead Section 4 (experiments) with Tier 2 category ordering (strongest signal) before Tier 1 (3-op permutations)
- Add Section 5: "Why Non-Commutativity Alone Does Not Predict Ordering Quality" — frame the H3 falsification as insight, not failure

---

## 7. Resource Allocation Summary

| Task | GPU-hours | Wall-clock (2 GPUs) | Priority |
|------|-----------|---------------------|----------|
| Tier 2 confirmation pilot | 1.5 | 45 min | **Immediate** |
| Full Tier 1 (3-op factorial) | 20 | 10 h | **P1** |
| Full Tier 2 (6-op category) | 18 | 9 h | **P2** |
| Class-level analysis | 0 | 1 h (analysis only) | **P3** |
| Abbreviated Tier 3 | 4 | 2 h | **P4** |
| Full Tier 4 (if budget allows) | 1 | 30 min | **P5** |
| **Total** | **44.5** | **~23 h** | |

This is within the original 51 GPU-hour budget, saving ~6.5 GPU-hours by abbreviating Tier 3 and deprioritizing the theoretical metric rescue.

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Tier 2's 9% pilot signal collapses at full scale | Medium-High | Run confirmation pilot first (Priority 0). Even a 2% signal is publishable. |
| H1 ordering spread shrinks below 0.2% at 200 epochs | Low | Pilot already shows 0.96% at 10 epochs on 100 samples. Paired seed design gives statistical power for small effects. |
| ViT training instability on CIFAR | Low | Use established timm recipe. ViT pilot already ran successfully. |
| Reviewers dismiss as "just ablation" | Medium | The reframed narrative leads with the gap (documented in two surveys), presents falsified theory as insight, and provides practical recommendations. Backup A (variance decomposition) remains available as additional framing if needed. |
| All theoretical contributions are negative | Medium | Frame honestly: "We tested two principled ordering theories; neither predicts ordering quality, revealing that the relationship between non-commutativity and learning dynamics is more complex than assumed. The empirical findings stand on their own." |

---

## 9. Anti-Pattern Check

- **Fence-sitting**: No. Clear PROCEED verdict with specific next steps in priority order.
- **Sunk cost reasoning**: No. Recommending deprioritization of Tier 3 (12 GPU-h savings) and NC metric rescue despite these being core proposal elements. The pilot data dictates the strategy, not the original plan.
- **Ignoring resource constraints**: No. Total revised budget (44.5 GPU-h) is under the original 51 GPU-h estimate. Wall-clock ~23h on 2 GPUs is feasible within project constraints.
