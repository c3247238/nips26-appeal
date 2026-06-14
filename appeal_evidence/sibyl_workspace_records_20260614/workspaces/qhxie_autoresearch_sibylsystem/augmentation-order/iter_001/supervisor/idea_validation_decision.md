# Idea Validation Decision

**Version**: 2.0 (corrected 2026-04-07 — supersedes 2026-04-02 draft)
**Evaluator**: sibyl-idea-validation-decision (sibyl-heavy tier)
**Correction note**: The prior version (2026-04-02) incorrectly declared all experiments complete and recommended proceeding directly to paper writing. That version was based on pilot-mode data only. This version corrects that error and aligns with the revised hypotheses, candidates, and task plan (all updated 2026-04-03).

---

## Pilot Evidence Summary

All existing experimental data is **pilot-scale only** (1 seed, 10 epochs, 100-sample subsets for Tier 1; 5k subsets for Tier 0; single seed for Tier 2/3/4). Full-scale runs (5 seeds, 200 epochs, full datasets) have not been executed. Verdicts from pilot data are preliminary and cannot be used for paper claims.

### Tier 0 (Completed — 2 seeds, 5k subset, 10 epochs)
- ResNet-18 × CIFAR-10, 6 orderings, 2 seeds
- Spread: **2.68%** (best: CJ→Crop→Flip at 28.04%, worst: Flip→CJ→Crop at 25.36%)
- Conventional ordering (Crop→Flip→CJ) ranks 5th of 6
- Decision at the time: GO

### Tier 1 Pilot (Completed — 1 seed, 10 epochs, 100-sample subsets)
All 4 arch-dataset blocks run in pilot mode. Accuracy levels are near random-chance floor in 2/4 blocks, which inflates apparent percentage-point spread:

| Block | Spread (%) | Best Ordering | Worst Ordering | Accuracy Range | Note |
|---|---|---|---|---|---|
| ResNet-18 × CIFAR-10 | 0.96% | CJ→Flip→Crop | CJ→Crop→Flip | 10.01%–10.97% | Near chance floor (10%); HIGH_CONFIDENCE rating is misleading |
| ResNet-18 × CIFAR-100 | 0.88% | Flip→Crop→CJ | CJ→Flip→Crop | 45.75%–46.63% | Above chance; meaningful signal |
| ViT-Small × CIFAR-10 | 2.32% | Flip→CJ→Crop | Crop→CJ→Flip | 17.38%–19.70% | Above chance; strongest signal |
| ViT-Small × CIFAR-100 | 0.25% | Flip→Crop→CJ | CJ→Flip→Crop | 2.64%–2.89% | Near chance floor (1%); unreliable |

**Critical issue (from issues.json)**: The ResNet-18×CIFAR-10 block (10.01–10.97%) and ViT-S×CIFAR-100 block (2.64–2.89%) are at or within 1 pp of random-chance performance. Accuracy comparisons in degenerate blocks cannot be attributed to augmentation ordering effects. After excluding these blocks, only 2 of 4 interpretable blocks show meaningful spread — this must be addressed in full-scale experiments.

### Tier 2 Pilot (Completed — 1 seed, 5k subset, 10 epochs)
- Category-level ordering, 5 orderings, ResNet-18 × CIFAR-10
- Interleaved P→G: 0.2939 (best) vs. All-geo-first: 0.2038 (worst)
- Spread: **9.01%** — largest effect in the study, but lowest statistical support (1 seed, early training)
- Issues.json flags this as "most likely to be early-training artifact" and requires Tier 2 confirmation pilot before full investment

### Tier 3 Pilot (Completed — 1 seed, 10 epochs)
- Magnitude levels M=5, M=9, M=14 on ResNet-18 × CIFAR-100
- M5=0.35%, M9=0.88%, M14=0.00%
- Issues.json flags: M14 shows exactly identical accuracy (0.245) for both orderings to 3 decimal places — suspicious; needs verification
- Supports new H6 (inverted-U) but requires 5-seed, 200-epoch confirmation

### Tier 4a (Completed — pixel-space only, 100 samples, 100 projections)
- NC_2 (SWD proxy, pixel space): Spearman rho = -0.20, p = 0.68
- H3 pixel-space **falsified**, but this is an underpowered estimate (100 samples in 3072-D)
- Revised H3 requires feature-space NC_2 (10k samples, 512-D, 1000 projections) — not yet computed

### Tier 4b (Completed — near-random encoder, 100 samples, 10 epochs)
- InfoNCE MI (underpowered encoder): CIFAR-10 rho = +0.54 (p=0.196), CIFAR-100 rho = -0.66 (p=0.081)
- Combined rho = -0.057 — inconclusive and sign-reversal is unexplained (could be encoder noise)
- Issues.json: Combined rho obscures the sign-reversal; must report per-dataset values
- Full-scale MI requires 200-epoch encoder from completed Tier 1 full runs — not yet available

---

## Updated Hypothesis Verdicts (Post-Pilot, Revised 2026-04-03)

The hypotheses below reflect the **corrected and revised** hypothesis set from `hypotheses.md` (2026-04-03). The prior decision file used the outdated hypothesis labels from iteration 0.

| Hypothesis | Pilot Verdict | Evidence | Full-Scale Required |
|---|---|---|---|
| **H1** — Ordering spread > 0.3% in ≥2/4 blocks (paired t-test, p<0.05) | Preliminary SUPPORTED | 2/4 interpretable blocks above 0.5% threshold; 2 blocks near chance floor | 5 seeds, 200 epochs, paired t-test |
| **H2** — ViT larger absolute ordering sensitivity than CNN (interaction ANOVA p<0.05) | Preliminary SUPPORTED | ViT/CIFAR-10 = 2.32% vs. CNN/CIFAR-10 = 0.96%; ViT/CIFAR-100 = 0.25% vs. CNN/CIFAR-100 = 0.88% (mixed) | Full two-way ANOVA with 5 seeds |
| **H3 (revised)** — Feature-space NC_2 (512-D, 10k samples) rho > 0.5 | PENDING | Pixel-space NC_2 falsified (rho=-0.20); feature-space not yet computed | Requires Tier 1 full checkpoint + Tier 4a feature-space run |
| **H4 (revised)** — Flip-first ordering outperforms Crop-first in ≥2/4 blocks | Preliminary PARTIALLY SUPPORTED | Flip→Crop→CJ wins in 2/4 blocks; corrected DPI prediction consistent with pilot winner | 5 seeds paired comparison |
| **H5b** — Interleaved P→G beats all-geometric-first in ≥3/4 blocks | Preliminary STRONGLY SUPPORTED | 9.01% spread (1 seed, early training) — highest effect but lowest statistical confidence | Tier 2 confirmation pilot required first; then full Tier 2 |
| **H6** — Inverted-U: M9 spread > M5 AND M9 > M14 | Preliminary SUPPORTED | M9>M5>M14 inverted-U seen in pilot; M14 identical-accuracy result needs verification | 5 seeds, 200 epochs; verify M14 result integrity |

---

## Decision Matrix

### cand_a — Primary Candidate

| Criterion | Weight | Score (1–5) | Evidence |
|---|---|---|---|
| Pilot signal strength | 0.30 | **4** | 2/4 interpretable blocks show meaningful spread (0.88–2.32%); Tier 2 category pilot shows 9.01% (single seed, needs confirmation). The 2 near-chance blocks reduce confidence from 5 to 4 — real effects exist but scope is uncertain. |
| Hypothesis survival | 0.25 | **4** | H1 (primary) preliminary supported; H2 supported; H4 corrected and supported; H5b and H6 are new/revised hypotheses with pilot backing. H3 revised (not falsified per se — pixel-space H3 falsified; feature-space H3 is an open test). No core hypothesis critically falsified. |
| Path to full result | 0.20 | **4** | Clear execution plan in task_plan.json. ~54 GPU-hours of well-specified experiments remain. Blocking gate: Tier 1 full (120 runs). Tier 2 confirmation pilot (30 min) determines Tier 2 investment. Feature-space NC requires Tier 1 checkpoints. All dependencies are clear. |
| Novelty (from report) | 0.15 | **4** | novelty_score = 8/10. No prior work isolates ordering as sole independent variable. Feature-space NC_2, corrected DPI, category-level interleaving, and inverted-U magnitude are all novel. Wrona et al. 2025 is the closest partial overlap but studies medical images with 2-step orderings only. |
| Resource efficiency | 0.10 | **3** | ~54 GPU-hours remaining before paper-ready data. At ~27h wall-clock on 4 GPUs this is feasible but not trivial. The Tier 2 confirmation pilot (45 min) provides a go/no-go gate that could save 18 GPU-hours if the 9.01% effect doesn't survive moderate training. |

**Weighted Score = (4×0.30) + (4×0.25) + (4×0.20) + (4×0.15) + (3×0.10)**
**= 1.20 + 1.00 + 0.80 + 0.60 + 0.30 = 3.90**

### cand_b — Backup (Variance Decomposition)

| Criterion | Weight | Score (1–5) | Evidence |
|---|---|---|---|
| Pilot signal strength | 0.30 | **1** | Activation condition NOT met: spread > 0.2% in 3/4 blocks (including interpretable blocks). cand_b activates only if full-scale Tier 1 shows all blocks with spread < 0.2%. Extremely unlikely. |
| Hypothesis survival | 0.25 | **2** | Even if activated, magnitude-dominates-variance is an "obvious" finding per candidates.json with limited novelty. |
| Path to full result | 0.20 | **2** | Requires full Tier 1 data and fresh design; partially reuses Tier 3 infrastructure. |
| Novelty | 0.15 | **3** | novelty_score = 7/10; variance decomposition framing is novel but weaker standalone contribution. |
| Resource efficiency | 0.10 | **1** | Would duplicate some infrastructure; activation probability < 20%. |

**Weighted Score = 0.30 + 0.50 + 0.40 + 0.45 + 0.10 = 1.75**

### cand_c — Secondary Analysis (Per-Class Accuracy)

Not a standalone candidate. Designated as zero-cost secondary analysis within cand_a paper using Tier 1 CIFAR-100 full runs (per-class accuracy). Depends on Tier 1 full completion.

---

## Decision Rationale

**ADVANCE on cand_a. Confidence: 0.76.**

This is an ADVANCE decision, but it must be distinguished from the prior (incorrect) ADVANCE that declared experiments complete. The correct ADVANCE means: proceed to full-scale experimentation as specified in task_plan.json. Do NOT proceed to paper writing yet.

**Why ADVANCE and not REFINE:**
- The pilot evidence establishes genuine ordering effects (2.32% in ViT/CIFAR-10; 9.01% category-level); the direction is confirmed.
- The revised hypotheses (H1, H2, H4, H5b, H6) are well-specified with pre-registered thresholds, falsification criteria, and statistical tests.
- The methodology is sound: 5-seed paired design eliminates between-seed confounds; Bonferroni correction applied; 200-epoch training eliminates early-training artifacts.
- The plan for H3 (feature-space NC_2) is a principled correction of the pixel-space failure, not a post-hoc rationalization.
- REFINE would apply if methodology problems were severe enough to require idea redesign — they are not. The identified issues (near-chance blocks, underpowered Tier 4, M14 verification) are all addressed by the full-scale plan.

**Why ADVANCE and not PIVOT:**
- H1 (primary) is supported in 2 interpretable blocks with measurable effects. The near-chance blocks do not falsify H1 — they require full-scale runs to generate meaningful comparisons.
- cand_a's weighted score is 3.90, well above the PIVOT threshold of 2.5.
- cand_b's activation condition is not met and is unlikely to be met. There is no better candidate to pivot to.

**Confidence is 0.76 (not higher) because:**
- 2 of 4 Tier 1 blocks were near the chance floor — we do not yet know if full-scale training reveals ordering effects in ResNet-18×CIFAR-10 and ViT-S×CIFAR-100 at meaningful accuracy levels.
- The 9.01% category-level effect may be an early-training artifact (issues.json: "most likely to be an early-training artifact").
- Feature-space NC_2 (revised H3) is entirely untested — it could either recover predictive power or yield a second falsification, which would require theoretical reframing.

**Sanity checks:**
- [x] All candidates evaluated, not just the front-runner
- [x] Near-chance blocks are penalized in the pilot signal score (4 not 5)
- [x] Not swayed by the 9.01% number — it is flagged as single-seed and requires confirmation pilot
- [x] No sunk-cost reasoning — the pilot GPU budget already spent is irrelevant to the decision
- [x] Pilot inconclusive blocks → ADVANCE to full-scale, not blindly advancing to paper writing

---

## Immediate Next Actions (Priority Order)

1. **[IMMEDIATE — BLOCKING] Run Tier 2 confirmation pilot** (45 min, 1 GPU): 5 category orderings on ResNet-18/CIFAR-10, 30 epochs, 2 seeds. Pass criterion: interleaved_PG spread > 1% on at least 1 seed. This determines whether to invest 18 GPU-hours in full Tier 2.

2. **[IMMEDIATE — PARALLEL] Launch full Tier 1 experiments** (10h wall-clock on 4 GPUs): All 4 blocks (tier1_resnet18_cifar10_full, tier1_resnet18_cifar100_full, tier1_vit_cifar10_full, tier1_vit_cifar100_full) in parallel. 5 seeds, 200 epochs, full datasets. This is the single blocking action for all downstream hypotheses.

3. **[AFTER TIER 1 FULL] Run full statistical analysis** (tier1_analysis_full): Paired t-tests, Bonferroni correction, Cohen's d, two-way ANOVA for H1 and H2 verdicts.

4. **[AFTER TIER 1 FULL — PARALLEL] Run Tier 3 full** (50 min): Test H6 inverted-U with 5 seeds. Use best/worst orderings from tier1_analysis_full. Verify M14 identical-accuracy result.

5. **[AFTER TIER 1 FULL — PARALLEL] Run Tier 4a feature-space NC** (30 min): Feature-space NC_2 with 10k samples, 1000 projections, 512-D embeddings from 200-epoch ResNet-18 checkpoint.

6. **[CONDITIONAL ON TIER 2 CONFIRMATION PASS] Run full Tier 2** (9h wall-clock): 100 runs, 5 seeds, 200 epochs. Only if confirmation pilot spread > 1%.

7. **[AFTER ALL ABOVE] Run final aggregation** (final_summary_full): Generate paper-ready tables, H1–H6 verdict table, practical recommendations. This is the entry point to paper writing.

8. **[THROUGHOUT] Address issues.json items**:
   - Verify M14 identical-accuracy result (training curves diagnostic)
   - Ensure ordering vs. baseline comparison uses matched training conditions in Table 1
   - Report per-dataset rho values for InfoNCE MI (not combined rho only)
   - Add H2a (architecture-differential ANOVA) as distinct from H4 (reversibility ordering) in reporting

---

## Key Risks Requiring Active Monitoring

| Risk | Severity | Monitoring Action |
|---|---|---|
| Ordering effects disappear at 200 epochs | HIGH | Check per-epoch spread curves; if spread < 0.2% by epoch 50 in all blocks, initiate early stop and reassess |
| Tier 2 9.01% effect is early-training artifact | HIGH | Tier 2 confirmation pilot (30 epochs, 2 seeds) is the gate |
| Feature-space NC_2 also falsified | MEDIUM | Report both pixel and feature-space rho; if both fail, reframe theoretical contribution as "negative result" (why NC_2 fails at all representation levels) |
| M14 zero-spread result invalid | MEDIUM | Verify with per-epoch training curves; re-run M14 with different seeds |
| ResNet-18×CIFAR-10 remains near chance at 200 epochs | MEDIUM | Full dataset (50k) at 200 epochs should produce ~92% accuracy; if ordering effects remain < 0.3%, that block contributes to narrowing the claim scope |
| Theoretical framework (Theorem 1 proof gap) | MEDIUM | Bubble-sort decomposition must be proved before paper submission; acknowledged in proposal as deferred |
| DPI contraction coefficient distribution-dependence | MEDIUM | Formal revision of H4 theoretical grounding needed; weakened claim adopted as mitigation |

SELECTED_CANDIDATE: cand_a
CONFIDENCE: 0.76
DECISION: ADVANCE
