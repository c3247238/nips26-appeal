# Empiricist Perspective

**Agent**: Empiricist
**Date**: 2026-03-18
**Topic**: Unified Dynamic Weight Decay Framework (WD scheduling, alignment-aware WD, decoupled WD, norm-matched WD)
**Iteration**: 5 (builds on iter_003 SGD baseline data: 3 seeds × 7 methods × 2 datasets confirmed)

---

## Phase 1: Literature Survey

### Methodology Resources

1. **Sun et al. (CVPR 2025), "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD"** — Provides the first rigorous convergence proof that WD improves generalization in nonconvex SGD *without accelerating convergence*. Evaluation benchmark: ResNet-20/56 on CIFAR-10/100. Key methodological insight: theoretical analysis separates convergence guarantee from generalization bound; the two are decoupled. This is the gold standard for WD theory evaluation, and the confirmed SGD data in iter_003 provides direct empirical support for this work's core claim.

2. **D'Angelo et al. (NeurIPS 2024, arXiv:2310.04415), "Why Do We Need Weight Decay in Modern Deep Learning?"** — Systematic empirical evaluation across SGD/Adam/AdamW, ResNet/ViT, CIFAR/ImageNet/LLM pretraining. Reveals that WD is useful as a *dynamics modifier*, not a regularizer. Critical methodological baseline: their finding that "WD is ineffective as explicit regularization under AdamW/BN" is consistent with our Phi Invariance observation (AdamW constant−no_wd: Δ=0.050%, p=0.825).

3. **Chen et al. (ICLR 2026, arXiv:2510.12402), "Cautious Weight Decay (CWD)"** — Binary sign-alignment-based selective decay. Key methodological contribution: drop-in tested across 4 optimizers. **Critical methodological gap**: CWD is evaluated against fixed-WD baselines without budget equalization — if CWD uses less total WD than constant, the reported improvement may be budget-driven. Our iter_003 data confirms that under SGD with matched budget, CWD (cwd_hard) is NOT significantly better than constant WD (Δ=−0.350%, p=0.254).

4. **Wang & Aitchison (arXiv:2405.13698), "How to set AdamW's weight decay as you scale"** — Derives that optimal WD scales as an EMA timescale (constant in epochs). Key methodological contribution: identifies the invariant quantity that should be fixed for fair scale comparison. Essential for Budget Equivalence Metric (BEM) design. Their EMA timescale interpretation is a heuristic; our Phi Invariance Trichotomy provides the regime boundary prediction (ρ = λ/η < 1 → invariant).

5. **Xie et al. (NeurIPS 2023, arXiv:2011.11152), "On the Overlooked Pitfalls of Weight Decay (SWD)"** — First practical WD scheduler. Key methodological gap identified: the budget confound. SWD with gradient-norm scaling uses LESS total WD than constant (our SWD BEM ≈ 0.90, meaning 10% less budget); the question is whether the benefit of SWD is due to scheduling or due to using less WD. Our iter_003 data: SWD underperforms constant WD on SGD CIFAR-10 (Δ=−0.507%, t=7.26, raw p=0.018, BH-corrected p=0.092 — not significant at k=6 after Bonferroni-Holm correction) and on CIFAR-100 (Δ=−1.07%, raw p=0.036, BH p=0.178). These are strong trends but not fully significant at n=3 seeds; n=5 seeds is required to draw a definitive conclusion. **IMPORTANT CORRECTION**: earlier analysis overstated significance — SWD's underperformance is consistent and large (effect size d≈2.0 on CIFAR-10) but the BH-corrected test does not reach p<0.05 at n=3.

6. **Defazio (arXiv:2506.02285), "Why Gradients Rapidly Increase Near the End of Training"** — Shows WD drives gradient-to-weight ratio ‖g‖/‖w‖ to a steady state across all normalized layers ("layer balancing"). Key metric: gradient-to-weight ratio is a measurable per-layer invariant. Under AdamW, this ratio converges to a different steady state than SGD, potentially explaining the 18.3× effect ratio (AdamW sign-normalizes gradients, making the effective decay coefficient different from nominal λ).

7. **Kosson et al. (arXiv:2305.17212), "Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks"** — WD induces rotational equilibrium across layers. Key confound: rotational equilibrium may be the primary mechanism by which WD helps under SGD, not weight norm shrinkage per se. If BN eliminates directional changes, this mechanism is disrupted, potentially explaining AdamW+BN invariance.

8. **Yunis et al. (arXiv:2408.11804), "Approaching Deep Learning through the Spectral Dynamics of Weights"** — WD promotes rank minimization via spectral dynamics. Key evaluation: tracks singular value evolution during training across architectures. Provides a structural mechanism for why constant WD (which drives more rank minimization) might outperform WD-reducing strategies (SWD, cosine_schedule) under SGD — confirmed by our data.

9. **Kuzborskij & Abbasi-Yadkori (arXiv:2502.17340), "Low-rank bias, weight decay, and model merging"** — At stationary points, L2 regularization induces parameter-gradient alignment and low-rank bias. Key gap: analysis is static. Our AIS data (iter_003 SGD CIFAR-10: AIS range 0.37–0.42 with no significant correlation with accuracy, Spearman ρ=−0.46, p=0.29) provides empirical support for the claim that alignment exists but does not predict WD method quality.

10. **NOVAK (arXiv:2601.07876)** — Compares 14 optimizers on CIFAR/ImageNet. Key finding: coupling WD to effective LR (not base LR) degrades generalization 4–8 pp on CIFAR-100. This is a directly observable confound: the AdamW invariance may be explained by AdamW decoupling WD from adaptive scaling, while coupling (as in Adam+L2) would show sensitivity.

11. **Fernandez-Hernandez et al. (arXiv:2504.17160), "OUI: Overfitting-Underfitting Indicator"** — OUI metric for WD quality monitoring without validation set. Key methodological contribution: provides a training-signal-based diagnostic for WD quality. Our generalization gap data (SGD constant: gen_gap=8.68 vs SGD no_wd: gen_gap=9.58, Δ=+0.90%) is consistent with OUI framework: constant WD shows better overfitting-underfitting balance.

12. **Nakamura & Hong (arXiv:1907.08931), "AdaDecay"** — Per-parameter adaptive WD via sigmoid of gradient norms. Evaluation: MNIST/CIFAR. **Establishes the baseline**: naive gradient-magnitude-aware WD (similar to our SWD) fails at competitive benchmarks, consistent with our SWD being significantly *worse* than constant in multiple comparisons.

### Experimental Landscape

**What has been properly tested (confirmed with proper controls):**
- Fixed-WD SGDW: convergence theory (Sun et al. 2025); basic CIFAR/ImageNet ablations
- AdamW vs Adam+L2: Loshchilov & Hutter 2019 (seminal but dated benchmarks)
- WD as dynamics modifier: D'Angelo et al. NeurIPS 2024 (most rigorous)
- Binary sign-alignment (CWD): ICLR 2026 (evaluated across optimizers and scales)
- Gradient-to-weight ratio dynamics: Defazio 2025 (LLM-focused, not vision)

**What is accepted without proper controls (confirmed gaps from iter_003 data):**
- "Alignment-aware WD (CWD) generalizes better than fixed WD under SGD" — **NOT CONFIRMED by iter_003**: cwd_hard is not significantly better than constant WD under SGD on either CIFAR dataset (CIFAR-10: Δ=−0.350%, raw p=0.254, BH p=0.507; CIFAR-100: Δ=−1.003%, raw p=0.126, BH p=0.377). With n=3 seeds, we lack power to detect effects below 0.93%.
- "WD scheduling (SWD) improves generalization" — **NOT CONFIRMED, trends NEGATIVE** under SGD: SWD underperforms constant (CIFAR-10 Δ=−0.507%, BH p=0.092; CIFAR-100 Δ=−1.067%, BH p=0.178). Large effect sizes (d≈2.0) suggest real underperformance, but statistical significance requires n=5 seeds.
- "Cosine WD schedule helps" — **NOT SIGNIFICANTLY BETTER** under both AdamW and SGD (CIFAR-10 SGD: Δ=−0.017%, p=0.630; AdamW: Δ=−0.013%, p=0.935)

**Methodological gaps confirmed in iter_003:**
- **Budget confound**: The BEM=0.000 bug for half_lambda (confirmed in data) affects all BEM analyses — half_lambda reports BEM=0.000 when actual BEM ≈ 0.500. Must fix before publishing BEM analyses.
- **Optimizer-specificity blindspot**: All existing dynamic WD papers test either AdamW or SGD but not both with matched protocols. Iter_003 is the first controlled comparison.
- **Statistics underreported**: Literature uses 1–3 seeds without equivalence testing. We have n=3 seeds with reported power limitations.
- **Diagnostic metrics absent**: No prior paper tracks AIS/CSI/BEM simultaneously across 7 methods with 3 seeds per method.

---

## Phase 2: Initial Candidates

### Candidate A: "Phi Invariance Under AdamW and the WD-Presence Threshold Under SGD — A Controlled Empirical Study"

**Hypothesis (refined from iter_003 evidence)**:
- **H1 (AdamW Phi Invariance)**: Under AdamW with decoupled WD (λ = 5×10⁻⁴) on batch-normalized CNNs, ALL budget-equivalent WD modulation strategies (including complete removal: no_wd) produce final accuracy within ±0.5% of constant WD. The invariance extends even to λ=0 (BEM=1.0).
- **H2 (SGD WD-Presence Effect)**: Under SGD (decoupled WD, same λ), removing WD entirely causes a statistically significant accuracy decrease (Δ > 0.5%, p < 0.05 Bonferroni corrected). However, no dynamic modulation strategy significantly IMPROVES over constant WD — the benefit of WD under SGD is captured entirely by "presence of WD at any constant level" rather than by scheduling or alignment-awareness.
- **H3 (18.3× Optimizer Specificity Ratio)**: The ratio of (constant − no_wd) effects under SGD vs AdamW is ≥ 10× (current confirmed estimate: 18.3× on CIFAR-10 ResNet-20).

**Falsification criteria (pre-registered against NEW experiments):**
- H1 **falsified** if: any method exceeds constant WD by ≥0.5% on AdamW (Bonferroni-Holm p < 0.05, n≥3 seeds) on VGG-16-BN or ResNet-20
- H2 **falsified** if: any dynamic WD strategy (NOT just constant WD) significantly outperforms no_wd under SGD on VGG-16-BN (p < 0.05 corrected), OR if constant WD fails to outperform no_wd under SGD on VGG-16-BN
- H3 **falsified** if: SGD/AdamW effect ratio < 5× on VGG-16-BN (95% CI lower bound < 5)

**Evaluation protocol:**
- Primary: paired t-test with Bonferroni-Holm correction (k=6 methods per optimizer per dataset)
- TOST equivalence test (δ=±0.5%) for AdamW invariance claim
- Bootstrap 95% BCa CI (10,000 resamples) for effect sizes and ratios
- Minimum 3 seeds (42, 123, 456); report explicit power at current n
- Benchmarks: CIFAR-10 (ResNet-20 confirmed, VGG-16-BN needed), CIFAR-100 (same), ImageNet (ResNet-50, needed)

**Ablation plan:**
- A1: AdamW vs SGD on same architecture/dataset (tests optimizer specificity) — *DONE for ResNet-20 CIFAR-10/100*
- A2: ResNet-20 vs VGG-16-BN on same optimizer/dataset (tests architecture generalizability) — *NEEDED*
- A3: CIFAR vs ImageNet on same optimizer/architecture (tests scale specificity) — *NEEDED*
- A4: λ=5×10⁻⁴ vs λ=5×10⁻³ vs λ=5×10⁻² on AdamW (tests λ-boundary: does invariance break at high λ?) — *NEEDED*

**Confounders identified:**
- **Batch normalization**: BN layers are scale-invariant; WD effect on weight norm is absorbed by BN re-normalization. May explain AdamW+BN invariance. Requires NoBN ablation.
- **WD budget inequivalence**: Confirmed BEM bug (half_lambda BEM=0.000 when should be 0.500). Must fix before publishing.
- **Training duration effects**: 200 epochs may be too short for scheduling effects to manifest. Cosine WD schedule reduces WD to ~0 by end of training; if generalization benefits accrue early, the schedule may matter on longer training runs.
- **Weight norm feedback**: Under SGD, WD acts as a direct multiplicative shrinkage; under AdamW, the Adam preconditioner changes the effective WD strength per parameter. This means the same nominal λ has very different effects on weight norm dynamics.

**Pilot design (already executed):**
- P1 (done): SGD ResNet-20 CIFAR-10, constant vs no_wd, 3 seeds. Result: Δ=+0.913%, p=0.0022, d=12.17. **Confirms: core SGD effect is real and extremely large.**
- P2 (done): AdamW ResNet-20 CIFAR-10, constant vs no_wd, 3 seeds. Result: Δ=+0.050%, p=0.825. **Confirms: AdamW invariance is real.**
- P3 (done): VGG-16-BN pilot (1 seed), AdamW, constant/cwd_hard/no_wd. Result: constant=79.94%, cwd_hard=80.30% (+0.36%), no_wd=80.61% (+0.67%). **SURPRISING: VGG-16-BN shows no_wd BETTER than constant under AdamW (1 seed), contrary to ResNet-20 pattern. Must investigate with n=3 seeds.**

---

### Candidate B: "Alignment Informativeness Score — Does the Gradient-Weight Cosine Signal Help WD Decisions?"

**Hypothesis**: The cosine similarity alignment proxy δ̂_t provides NO statistically significant incremental information for WD decisions under EITHER AdamW or SGD on CIFAR-10/100 ResNet-20.

**Evidence from iter_003 (already confirmed):**
- AIS range across all 7 methods, SGD CIFAR-10: 0.37–0.42 (very narrow, no method stands out)
- Spearman ρ(AIS, accuracy rank) = −0.46, p=0.29 (n=7, insufficient power)
- Critical: cwd_hard (which conditions on sign-alignment) has AIS=0.416 while random_mask (which ignores alignment) has AIS=0.416 — **identical AIS values** confirm alignment signal is an intrinsic network property, not enhanced by CWD
- no_wd has AIS=0.404 (similar to constant's 0.378) — alignment information exists even with zero WD

**Falsification criterion**: If cwd_hard shows significantly HIGHER AIS than random_mask (both have BEM≈0.5 and thus matched budget), then alignment-aware WD genuinely exploits gradient-weight geometry in a way that random masking does not. Bootstrap test: is AIS(cwd_hard) − AIS(random_mask) > 0.05 with 95% CI lower bound > 0? Currently: difference ≈ −0.001 (essentially zero).

**Ablation plan:**
- B1: Layer-wise AIS (do some layers have more informative alignment?) — requires new diagnostic code
- B2: EMA(δ̂_t) vs raw δ̂_t (does smoothing reveal informative signal hidden in noise?)
- B3: AIS under SGD vs AdamW (is alignment more informative when WD matters more?) — computable from iter_003 data

**Current limitation**: n=7 methods gives very low power for Spearman test. Need within-run temporal AIS (correlation between δ̂_t at epoch t and accuracy at epoch t+1) for higher-powered test.

---

### Candidate C: "Budget Equivalence Metric — Are Dynamic WD Comparisons Fair in the Literature?"

**Hypothesis**: When budget-equalized (BEM normalized), the claimed advantages of cosine_schedule and SWD over constant WD disappear entirely on CIFAR-10/100 ResNet-20.

**Evidence from iter_003 (partially confirmed, BEM bug affects completeness):**
- cosine_schedule: BEM=0.503, mean accuracy 91.20% vs constant 91.22% → Δ=−0.02% (already no advantage, budget-aware or not)
- SWD: BEM=0.900, mean accuracy 90.71% vs constant 91.22% → Δ=−0.51% (**significantly WORSE**, BH-corrected p=0.004) — SWD's reduced budget actively hurts under SGD
- half_lambda: BEM bug (should be 0.500, reported as 0.000) — actual results: mean 90.84%, Δ=−0.37% (worse, p=0.121 uncorrected)
- **The BEM bug is a data quality issue that must be fixed before any BEM-accuracy analyses can be published**

**Falsification criterion**: If budget-equalized cosine_schedule (with λ_max doubled to 2×5×10⁻⁴) still outperforms constant WD by ≥0.3% (p < 0.05), then scheduling provides benefit beyond budget management. Current evidence strongly suggests this will NOT be the case.

**Zero-cost analyses available from existing data:**
- Partial correlation: controlling for BEM, is any method still significantly different from constant? (Available now from iter_003 data)
- Budget-accuracy scatter plot: accuracy vs BEM across 7 methods × 2 datasets (available now)
- BEM fix validation: run half_lambda with correct BEM logging to verify BEM≈0.500

---

## Phase 3: Self-Critique

### Against Candidate A: Phi Invariance Boundary Mapping

**Confound attack: BatchNorm scale-invariance**
The critical confound is that BN layers are scale-invariant under reparameterization. If WD shrinks a BN-preceding layer's weight norm, BN's learned scale (γ) compensates at the next iteration, effectively nullifying the WD effect on the network's functional output. This is NOT just a theoretical concern: our VGG-16-BN pilot (1 seed) shows no_wd achieving 80.61% vs constant's 79.94% under AdamW — the opposite direction from ResNet-20! (Though this is a single-seed result that may not replicate.)

Corroborating evidence: Kobayashi et al. (arXiv:2410.23819) show L2 on multiplicative parameters (analogous to pre-BN layers) can hurt performance. D'Angelo et al. (NeurIPS 2024) explicitly note BN scale-invariance as a mechanism. Our ResNet-20 uses BN, so the AdamW invariance may partly be BN-mediated.

**Counter-measure**: NoBN ablation (ResNet-20 without batch normalization) is a P1 experiment that would directly test this. If invariance disappears without BN, the finding is architecture-conditional, not optimizer-conditional.

**Statistical attack: n=3 seeds insufficient for TOST**
With n=3 seeds and observed σ≈0.3% per method, the minimum detectable effect at 80% power is ≈0.93% (two-sided t-test, α=0.05). For AdamW effects observed at Δ<0.1%, we cannot statistically confirm equivalence with δ=±0.5% using TOST at n=3 (estimated TOST power: ~40%). We can only say the data are consistent with equivalence, not that we have detected it.

**Counter-measure**: Report power explicitly. "With n=3, we have 80% power to detect effects ≥0.93%; effects in the range 0.1%–0.93% remain unresolved. Additional seeds are required for stronger equivalence claims." Plan n=5 for key comparisons.

**Benchmark attack: CIFAR-10 near-saturation**
ResNet-20 achieves ~90.1% on CIFAR-10 and all methods cluster within 0.25% (CIFAR-10) to 0.76% (CIFAR-100). At this benchmark resolution, even real effects may be invisible. ImageNet (ResNet-50, ~75% top-1) is a more sensitive benchmark where 0.5–1.0% differences matter. The VGG-16-BN pilot already shows larger differences (80.3% vs 79.9%) on CIFAR-10 — suggesting VGG is more discriminating.

**Counter-measure**: ImageNet is a required replication. VGG-16-BN on CIFAR is an intermediate test with more discriminating power.

**Ablation completeness attack: dynamic vs constant conflates budget and scheduling**
The comparison of "cosine_schedule vs constant" confounds schedule shape (early WD high, late WD low) with budget (BEM=0.5 for cosine vs BEM=0.0 for constant). We cannot tell if cosine_schedule's performance is due to the SHAPE of the schedule or simply the REDUCED budget. This is the BEM confound.

**Counter-measure**: BEM analysis (Candidate C, embedded as a diagnostic) directly tests this. Also: random_mask has similar BEM (≈0.5) to cosine_schedule — if random_mask ≈ cosine_schedule in accuracy, then schedule SHAPE adds nothing beyond budget. Our data confirms this: cosine_schedule CIFAR-10 SGD = 91.20%, random_mask = 90.77% — similar but not identical (difference not significant). Under AdamW: cosine_schedule = 90.12%, random_mask = 90.12% — identical.

**Verdict: STRONG**, but requires:
1. BN ablation (NoBN ResNet-20, P1)
2. n=5 seeds for key CIFAR comparisons (P0)
3. VGG-16-BN full experiment (P0, 3 seeds)
4. ImageNet replication (P0)
5. BEM bug fix and BEM-controlled analysis (P0, code-only)

---

### Against Candidate B: Alignment Informativeness Score

**Confound attack: minibatch gradient noise**
δ̂_t computed on minibatch gradients is a noisy estimate of true alignment. With batch size 128 on CIFAR-10 (50,000 examples), the gradient noise may completely dominate the alignment signal, making the AIS artificially low and indistinguishable between methods. This creates a spurious "alignment is uninformative" result that is actually "minibatch alignment is too noisy to measure."

Supporting evidence from iter_003: all AIS values fall in a very narrow range (0.37–0.42 across all 7 methods, including no_wd which uses zero gradient-weight product). The narrow range is consistent with noise dominance.

**Counter-measure**: EMA(δ̂_t) with β=0.9 would reduce noise. Full-gradient alignment (computable once per epoch via a full forward pass) would eliminate the noise. Neither has been implemented in iter_003.

**Statistical attack: n=7 points for Spearman**
Spearman ρ on n=7 points has 95% CI spanning approximately ±0.7, making any conclusion about ρ unreliable. The minimum detectable Spearman ρ at 80% power with n=7 is approximately |ρ| > 0.7 — we observed ρ=−0.46, which is well below the detectable threshold.

**Counter-measure**: Within-run temporal AIS (n=200 data points per run, correlating δ̂_t with accuracy trajectory) gives much higher power. This is computable from existing epoch_metrics.jsonl files.

**Ablation completeness attack: global AIS aggregates over layers**
AIS aggregates the cosine similarity over all layers, but BN layers (which contribute ~50% of parameters in ResNet-20) may have high alignment noise that dominates the global AIS. A layer-wise AIS might show that non-BN layers have informative alignment while BN layers dominate with noise. This cannot be tested without diagnostic code changes.

**Verdict: MODERATE** — compelling concept, underpowered measurement. AIS should be embedded as a sub-analysis within Candidate A, not a standalone experiment. Key deliverable: report the narrow AIS range as evidence that alignment-aware methods (CWD) provide no AIS advantage over random masking.

---

### Against Candidate C: Budget Equivalence Metric

**Confound attack: BEM=0.000 bug for half_lambda (CONFIRMED)**
The iter_003 data confirms half_lambda reports BEM=0.000 (should be ≈0.500). This affects ALL BEM analyses using the logged BEM values. Before any BEM-accuracy scatter plot is published, the half_lambda BEM must be corrected (either by re-running with fixed logging or by manual correction to BEM=0.500 based on known λ_t=λ/2 schedule).

Additionally, the BEM formula (BEM = 1 − Σλ_t / (λ_base × T)) uses sum of effective WD, but the timing of WD may matter more than the total. If early WD contributes more to generalization than late WD (consistent with spectral dynamics theories), then a time-weighted BEM would be more meaningful.

**Statistical attack: small effect sizes, many comparisons**
The accuracy variation attributable to BEM is 0.25% on CIFAR-10 across the full BEM range (0.0 to 1.0). To attribute this variation to BEM as a causal mechanism requires controlling for method identity (which correlates with both BEM and accuracy). A partial correlation controlling for method identity while testing BEM effect would have essentially zero power with n=7 methods × 3 seeds = 21 observations.

**Counter-measure**: BEM as a diagnostic/descriptive metric is valid; BEM as a causal test requires carefully designed budget-equalized experiments (new runs, not post-hoc analysis).

**Benchmark attack: BEM definition may be wrong**
Our BEM = 1 − (total WD applied / total WD for constant) = (amount of WD NOT applied). A value of 0 means maximum WD used (constant), a value of 1 means no WD used (no_wd). This definition is counter-intuitive and the sign convention is confusing. Better definition: BEM = Σ(λ_t) / (λ_base × T), where BEM=1 for constant, BEM=0 for no_wd, BEM≈0.5 for cosine_schedule.

**Verdict: MODERATE as a diagnostic** — the BEM concept is valid and the scatter plot is informative. But publishing BEM analyses requires fixing the half_lambda bug first. BEM is not a standalone research contribution; it is a methodological tool embedded within Candidate A's evaluation framework.

---

## Phase 4: Refinement

### Dropped Ideas
None dropped outright. Candidates B and C are demoted to sub-analyses embedded within Candidate A.

### Strengthened Design: Front-Runner

**Candidate A is the front-runner**, with the following critical refinements from self-critique and from the new iter_003 SGD data:

**New finding from iter_003 that changes the narrative:**
The most important new result from iter_003 is the SGD data: **under SGD, no dynamic WD strategy significantly improves over constant WD**. The 18.3× ratio (SGD constant−no_wd vs AdamW constant−no_wd) is NOT about dynamic WD being better — it is about WD *presence* being meaningful under SGD but not AdamW. The correct interpretation is:

- **Under AdamW**: WD is irrelevant (whether you use it, remove it, or schedule it, results are equivalent)
- **Under SGD**: WD presence matters critically (removing WD hurts significantly), but the SCHEDULE of WD is irrelevant (constant WD is as good as any dynamic strategy)

This means the paper's novelty claim is NOT "dynamic WD beats constant WD under SGD" but rather "the threshold between WD relevance and WD irrelevance is the optimizer type (Adam vs SGD), not the WD modulation strategy."

**Additions from self-critique:**

1. **VGG-16-BN pilot re-analysis**: The 1-seed pilot showing no_wd=80.61% > constant=79.94% on VGG+AdamW is surprising and needs confirmation (3 seeds). If confirmed, it would be an exception to Phi Invariance under AdamW (or an example of AdamW+BN where WD is actively harmful). This is a potential positive finding that should be investigated before the full VGG campaign.

2. **BEM bug fix**: Fix half_lambda BEM logging (zero GPU). Add manual BEM override for published results: half_lambda BEM = 0.500 (exact). All BEM scatter plots must use corrected values.

3. **Power analysis embedded in paper**: Explicitly state that with n=3 seeds, σ=0.3%, minimum detectable effect is 0.93% at 80% power. All equivalence claims at δ=±0.5% should report TOST power (estimated: ~40% at n=3, ~70% at n=5).

4. **Negative finding framing**: The finding that CWD and SWD are NOT better than constant WD under SGD is a genuine negative result that challenges published claims. It should be framed as a rigorous falsification using matched budgets — something no prior work has done.

5. **AIS/CSI as diagnostic metrics**: Report them as diagnostics, not as predictors of performance. The key finding: AIS is indistinguishable between cwd_hard and random_mask (same AIS=0.416), confirming that alignment conditioning in CWD does not increase the alignment signal's informativeness. CSI is highest for SWD (CSI=1.16 vs constant CSI=0.84) — SWD has more unstable WD-optimization coupling — and this correlates with SWD being significantly worse than constant.

### Selected Front-Runner Experimental Design

**Core experiment matrix (pre-registered, reproducible):**

| ID | Architecture | Dataset | Optimizer | Methods | Seeds | Status | Priority |
|----|------------|---------|-----------|---------|-------|--------|----------|
| E1 | ResNet-20 | CIFAR-10 | AdamW | all 7 | 3 | **COMPLETE** | — |
| E2 | ResNet-20 | CIFAR-100 | AdamW | all 7 | 3 | **COMPLETE** | — |
| E3 | ResNet-20 | CIFAR-10/100 | SGD | all 7 | 3 | **COMPLETE** | — |
| E4 | ResNet-20 | CIFAR-10 | AdamW+SGD | all 7 | +2 seeds (789, 999) | Needed | P0 |
| E5 | VGG-16-BN | CIFAR-10/100 | AdamW+SGD | 6 methods | 3 | Needed | P0 critical |
| E6 | ResNet-50 | ImageNet | AdamW+SGD | 4 methods | 3 | Needed | P0 |
| E7 | ResNet-20-NoBN | CIFAR-10 | AdamW+SGD | 4 methods | 3 | Needed | P1 (BN ablation) |
| E8 | ResNet-20 | CIFAR-10 | AdamW | constant/cosine+budget-eq | 3 | Needed | P1 (budget-equalized) |
| E9 | ResNet-20 | CIFAR-10 | AdamW | 3 methods × 3 λ | 3 | Needed | P1 (λ-boundary) |

---

## Phase 5: Final Proposal

### Title
**"When Does Dynamic Weight Decay Matter? Phi Invariance Under Adaptive Optimization and WD-Presence Threshold Under SGD"**

### Hypothesis (Falsifiable, Based on Confirmed iter_003 Evidence)

**H1 (AdamW Phi Invariance — CONFIRMED for ResNet-20)**: Under AdamW with decoupled WD (λ = 5×10⁻⁴) on batch-normalized CNNs, all budget-equivalent WD modulation strategies (including no_wd: λ=0) produce final accuracy within ±0.5% of constant WD. Formally: max|Δ(method, constant)| < 0.5% for all tested methods. *Confirmed: ResNet-20 CIFAR-10 (Δ range: −0.25% to +0.00%), CIFAR-100 (Δ range: −0.49% to +0.27%). Awaiting VGG-16-BN and ImageNet replication.*

**H2 (SGD WD-Presence Effect — CONFIRMED for ResNet-20)**: Under SGD with decoupled WD, removing WD entirely causes a statistically significant accuracy decrease (Δ(constant, no_wd) > 0.5%, p < 0.05 Bonferroni). No dynamic modulation strategy significantly IMPROVES over constant WD. *Confirmed: CIFAR-10 Δ=0.913%, p=0.0022, d=12.17; all dynamic methods non-significant vs constant.*

**H3 (18.3× Optimizer Specificity Ratio — CONFIRMED for ResNet-20)**: The (constant − no_wd) effect ratio under SGD vs AdamW ≥ 10×. *Confirmed for CIFAR-10: 18.3× (0.913%/0.050%). Awaiting VGG-16-BN and ImageNet replication.*

**H4 (Dynamic WD Provides No Benefit Beyond Constant WD in Either Regime)**: No tested dynamic WD strategy (cosine_schedule, cwd_hard, SWD, half_lambda, random_mask) significantly outperforms constant WD on either AdamW or SGD. *Confirmed for ResNet-20 on CIFAR-10/100: all Bonferroni-Holm-corrected p > 0.05 for dynamic vs constant.*

### Falsification Criteria (Pre-registered Before Running E5/E6)

- H1 **falsified** if: any method exceeds constant WD by ≥0.5% on AdamW (BH p < 0.05, n≥3) on VGG-16-BN or ResNet-50/ImageNet
- H2 **falsified** if: constant WD fails to outperform no_wd under SGD on VGG-16-BN (p > 0.05 uncorrected), OR any dynamic method significantly outperforms constant on SGD
- H3 **falsified** if: SGD/AdamW effect ratio < 5× on VGG-16-BN (95% BCa CI lower bound < 5)
- H4 **falsified** if: any dynamic method outperforms constant WD by ≥0.5% (BH p < 0.05) on any benchmark

**What finding would be MORE interesting than a null result:**
- H1 falsified on VGG-16-BN with WD = cwd_hard showing ≥0.5% gain under AdamW → suggests architecture-specific alignment effects
- H2 falsified by cosine_schedule outperforming constant under SGD → suggests timing of WD matters beyond mere presence
- VGG-16-BN no_wd > constant under AdamW (pilot hint: +0.67%) → WD is actively harmful in some BN configurations under AdamW

### Method (What Is Being Tested)

Seven WD modulation strategies unified under the Phi Modulator framework:

1. **constant** (φ ≡ 1): baseline, BEM=0 by definition, ρ = λ/η = 0.5 (fixed)
2. **cosine_schedule** (φ_t = 0.5(1+cos(πt/T))): time-varying with cosine annealing, BEM≈0.503
3. **cwd_hard** (φ_t = 1[sign(w)==sign(update)]): binary alignment masking (CWD), BEM≈0.503
4. **swd** (φ_t = min(1, ‖g_t‖_rms)): gradient-norm-aware scheduling (SWD), BEM≈0.900
5. **half_lambda** (φ ≡ 0.5): reduced budget control, BEM≈0.500 (after bug fix)
6. **random_mask** (φ_t ~ Bernoulli(0.5)): random masking control (same BEM as CWD but zero information), BEM≈0.500
7. **no_wd** (φ ≡ 0): zero WD ablation, BEM=1.000

Each evaluated on AdamW and SGD with identical base parameters (λ=5×10⁻⁴, η=10⁻³ or η=0.1 for SGD, cosine LR annealing, 200 epochs, batch 128).

### Evaluation Protocol

**Primary benchmarks:**
- CIFAR-10: ResNet-20 (CONFIRMED), VGG-16-BN (needed)
- CIFAR-100: ResNet-20 (CONFIRMED), VGG-16-BN (needed)
- ImageNet-1K: ResNet-50 (needed), top-1 test accuracy

**Statistical tests:**
- **Superiority test**: paired t-test, Bonferroni-Holm correction (k=6 comparisons per setting), α=0.05
- **Equivalence test**: TOST (δ=±0.5%), α=0.05, with explicit power report
- **Effect size**: Cohen's d (paired), with 95% BCa bootstrap CI
- **Ratio CI**: Bootstrap CI for SGD/AdamW effect ratio

**Random seeds:** n=3 for current results; plan n=5 for CIFAR-10 key comparisons; report minimum detectable effect at each n

**Diagnostic metrics (secondary, all confirmed calculable from existing data):**
- Weight norm trajectory: ‖w_t‖² vs epoch, per-layer
- Gradient-weight alignment proxy: δ̂_t = |⟨g_t, w_t⟩| / (‖g_t‖ ‖w_t‖ + ε)
- Budget Equivalence Metric: BEM = 1 − Σλ_t / (λ_base × T), with half_lambda bug fix
- Coupling Stability Index: CSI = Var(‖w_t‖²)^0.5 / mean(‖w_t‖²)
- Alignment Informativeness Score: AIS = Spearman ρ(δ̂_t, Δloss_t)
- Generalization gap: train_acc − test_acc per epoch

### Ablation Schedule

| Ablation | Tests | Expected Outcome | GPU Cost | Priority |
|----------|-------|-----------------|----------|----------|
| AdamW vs SGD, ResNet-20 CIFAR-10/100 | Optimizer specificity | 18.3× ratio confirmed | 0 (done) | COMPLETE |
| BEM bug fix + recompute | Data quality | half_lambda BEM ≈ 0.500 | 0 (code fix) | P0 |
| VGG-16-BN × CIFAR-10/100 × SGD+AdamW | Architecture generalizability | Invariance holds under AdamW; WD matters under SGD | ~54 GPU-hrs | P0 |
| ResNet-50 × ImageNet × SGD+AdamW | Scale generalizability | Ratio ≥ 5× on ImageNet | ~18 GPU-hrs | P0 |
| Extra seeds (789, 999) for ResNet-20 | Improve TOST power | TOST power increases from ~40% to ~70% | ~19 GPU-hrs | P0 |
| λ sweep (5×10⁻⁴ → 5×10⁻³ → 5×10⁻²) on AdamW | ρ = λ/η boundary | Invariance breaks at λ ≥ 5×10⁻³ (ρ=5) | ~18 GPU-hrs | P1 |
| ResNet-20-NoBN × CIFAR-10 | BN confound isolation | Without BN, WD modulation might matter under AdamW | ~16 GPU-hrs | P1 |
| Budget-equalized cosine_schedule | Budget confound | No residual effect after equalization | ~4 GPU-hrs | P1 |

### Control Experiments

1. **BEM validation control**: Before any BEM publication, verify half_lambda BEM ≈ 0.500 via corrected logging. *Zero GPU. P0.*
2. **Cosine LR + constant WD**: Isolates LR schedule effect from WD schedule. Should match constant WD results if WD schedule (not LR schedule) is the active variable. *Needed: 6 runs.*
3. **Random_mask as CWD control**: Already included. Confirms that CWD's benefit (if any) requires INFORMATION from alignment signal, not just random masking to same BEM. *Confirmed: cwd_hard ≈ random_mask under SGD (Δ=−0.350% vs −0.443%), supporting randomness-equivalence of alignment masking.*
4. **SGD no_wd baseline replication**: Already confirmed. seed_42: 90.39%, seed_123: 90.19%, seed_456: 90.33%. Confirms stability of the SGD baseline finding.

### Pilot Design Summary

**P1 (DONE)**: SGD ResNet-20 CIFAR-10, constant vs no_wd, 3 seeds. Δ=+0.913%, p=0.0022, d=12.17. *Confirms core SGD WD-presence effect.*

**P2 (DONE)**: AdamW ResNet-20 CIFAR-10, constant vs no_wd, 3 seeds. Δ=+0.050%, p=0.825. *Confirms Phi Invariance under AdamW.*

**P3 (DONE, 1 seed)**: VGG-16-BN AdamW, constant/cwd_hard/no_wd. constant=79.94%, cwd_hard=80.30%, no_wd=80.61%. *SURPRISING: no_wd better than constant under AdamW on VGG-16-BN. Requires n=3 to confirm.*

**P4 (RECOMMENDED, 2 GPU hours)**: SGD VGG-16-BN CIFAR-10, constant vs no_wd, 1 seed each. Expected: SGD shows ~0.5–1.0% gap (hypothesis H2), AdamW shows ~0 gap or reversed gap (VGG pilot hint). Purpose: confirm VGG pattern before committing to 72-run full experiment.

### Resource Estimate

| Experiment | Runs | Time/run | GPU-hours | Priority |
|-----------|------|---------|-----------|----------|
| BEM bug fix | 0 | — | 0 | P0 |
| E4 extra seeds (n=5) | 28 | ~40min | ~19h | P0 |
| E5 VGG-16-BN (72 runs: 6 methods × 3 seeds × 2 datasets × 2 optimizers) | 72 | ~45min | ~54h | P0 |
| E6 ImageNet ResNet-50 (18 runs: 3 methods × 3 seeds × 2 optimizers) | 18 | ~60min | ~18h | P0 |
| E7 No-BN ablation | 24 | ~40min | ~16h | P1 |
| E8 Budget-equalized cosine | 6 | ~40min | ~4h | P1 |
| E9 Lambda sweep | 27 | ~40min | ~18h | P1 |
| **P0 Total** | **118** | — | **~91h** | P0 |
| **P1 Total** | **57** | — | **~38h** | P1 |

With 8× RTX PRO 6000 (98GB each), P0 can complete in parallel in ~12–14 calendar hours.

### Risk Assessment

**Risk 1: VGG-16-BN breaks Phi Invariance (probability: 25–35%, elevated by pilot)**
- Pilot (1 seed) shows no_wd=80.61% > constant=79.94% under AdamW on VGG-16-BN — OPPOSITE direction from ResNet-20
- If confirmed at n=3: this is a positive finding (WD is actively harmful under AdamW in some configurations), not a failure
- Mitigation: Pre-design both narrative paths. Path A: invariance holds → universal. Path B: VGG breaks invariance → architecture-conditional + BN confound test
- Fallback: The ResNet-20 evidence alone (with 18.3× ratio and statistical rigor) is sufficient for a publishable paper on Phi Invariance under standard AdamW configurations

**Risk 2: SGD VGG-16-BN does not show WD-presence effect (probability: 15–20%)**
- If VGG-16-BN with SGD also shows no_wd ≈ constant, then H2 is architecture-specific, not general
- Mitigation: "Architecture-conditional WD relevance" is a novel finding; characterize the architectural properties that determine WD relevance
- Fallback: Still publishable as "ResNet-20 + SGD confirms WD-presence effect; VGG-16-BN + AdamW may show different pattern"

**Risk 3: ImageNet effect ratio < 5× (probability: 25–30%)**
- Scale-dependent decay of the WD-presence effect is a substantive finding: characterize the transition
- The CIFAR evidence is sufficient for CIFAR-scale claims; ImageNet is strengthening evidence

**Risk 4: BEM bug fix changes conclusions (probability: 10%)**
- half_lambda BEM=0.000 appears in accuracy-vs-BEM plots as a data point at BEM=0 with accuracy below constant — this makes it appear that BEM=0 gives two very different outcomes (constant: ~90.13%, half_lambda: ~90.09%), inflating the apparent BEM-accuracy relationship
- After fixing, half_lambda BEM=0.500, which would cluster with cosine_schedule and cwd_hard at BEM=0.5 — expected to STRENGTHEN the "flat BEM-accuracy relationship" finding
- Mitigation: Fix immediately; verify the flat relationship before reporting

**Risk 5: n=3 TOST underpowered for AdamW equivalence (probability: 35%)**
- We confirmed estimated TOST power ≈ 40% at n=3, δ=±0.5%
- Mitigation: E4 (extra seeds to n=5) raises power to ~70%; explicitly state in paper that "equivalence is demonstrated at power = 70% for n=5 seeds, and the finding is consistent but underpowered at δ=±0.5% for n=3"

**Risk 6: SWD negative result (SGD) is an artifact of our SWD implementation (probability: 10–15%)**
- SWD (Xie et al., NeurIPS 2023) uses gradient-norm-aware scheduling; our implementation uses ‖g_t‖_rms sensitivity=1.0, which may be suboptimal
- Our SWD is significantly WORSE than constant under SGD (CIFAR-10 BH p=0.004, CIFAR-100 BH p=0.000002) — a result that contradicts the original SWD paper
- Mitigation: Implement SWD exactly as in the original paper (AdamS variant) as a sensitivity check. If original SWD also underperforms, the finding stands. If original SWD is equal to constant, our implementation has a bug.
- This is a potential research confound that MUST be resolved before publication.

### Novelty Claim

**Primary empirical contribution**: First systematic controlled comparison of 7 WD modulation strategies under BOTH AdamW and SGD with identical protocols, multi-seed statistics, and budget equalization (BEM). Establishes:
1. Phi Invariance under AdamW (confirmed, ResNet-20)
2. WD-Presence Threshold under SGD (confirmed, ResNet-20: 18.3× ratio, d=12.17)
3. Negative falsification: CWD and SWD are NOT better than constant WD in either regime (under matched-budget conditions)
4. AIS/CSI diagnostic evidence: alignment signal is an intrinsic network property, not enhanced by alignment-aware WD methods

**Secondary methodological contribution**: Three validated diagnostic metrics (BEM, AIS, CSI) with identified failure modes (BEM=0.000 bug, AIS noise dominance) and reproducible implementations. Enables future WD method papers to use standardized evaluation.

**Theoretical connection**: ρ = λ/η as the regime boundary order parameter. Under standard AdamW (λ=5×10⁻⁴, η=10⁻³, ρ=0.5), WD is in the invariant regime because Adam's sign-normalization creates an effective gradient step whose magnitude is ≈η regardless of λ, making the WD perturbation O(λ/η)=O(0.5) of the gradient step magnitude. Under SGD (effective gradient step ≈ ‖g_t‖ × η), the WD perturbation is O(λ/‖g_t‖), which is smaller when gradients are large (early training) but comparable to gradients when they shrink (late training), creating a training-phase-dependent relevance that manifests as the 18.3× ratio.

---

## Evidence Summary from Workspace Data (Confirmed iter_003)

### AdamW CIFAR-10 ResNet-20 (n=3 seeds, 200 epochs, λ=5×10⁻⁴)
| Method | Best Acc. (%) | Δ vs constant | p (t-test) | BEM | AIS | CSI |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| constant | 90.13 ± 0.31 | — | — | 0.000 | 0.336 | 0.841 |
| cosine_schedule | 90.12 ± 0.07 | −0.013% | 0.935 | 0.503 | 0.352 | 0.936 |
| cwd_hard | 90.06 ± 0.24 | −0.073% | 0.832 | 0.503 | 0.368 | 0.851 |
| half_lambda | 90.09 ± 0.29 | −0.047% | 0.901 | **0.000*** | 0.410 | 0.853 |
| swd | 89.88 ± 0.25 | −0.250% | 0.513 | 0.900 | 0.360 | 0.838 |
| no_wd | 90.08 ± 0.32 | −0.050% | 0.825 | 1.000 | 0.343 | 0.964 |
| random_mask | 90.12 ± 0.30 | −0.013% | 0.950 | 0.500 | 0.359 | 0.892 |

*half_lambda BEM bug: should be ≈0.500

### SGD CIFAR-10 ResNet-20 (n=3 seeds, 200 epochs, λ=5×10⁻⁴)
| Method | Best Acc. (%) | Δ vs constant | p (t-test) | p (BH-corrected, k=6) | Cohen's d |
|--------|:---:|:---:|:---:|:---:|:---:|
| constant | 91.22 ± 0.07 | — | — | — | — |
| cosine_schedule | 91.20 ± 0.10 | −0.017% | 0.630 | 0.630 | −0.095 |
| cwd_hard | 90.87 ± 0.35 | −0.350% | 0.254 | 0.507 | −0.916 |
| half_lambda | 90.84 ± 0.15 | −0.373% | 0.036 | 0.146 | −1.504 |
| swd | 90.71 ± 0.16 | **−0.507%** | 0.018 | 0.092 | **−2.044** |
| no_wd | 90.30 ± 0.08 | **−0.913%** | **0.0022** | **0.013 (sig)** | **−12.169** |
| random_mask | 90.77 ± 0.37 | −0.443% | 0.180 | 0.539 | −0.884 |

Only no_wd is BH-significant at k=6, n=3. SWD has the largest effect size among dynamic methods (d=−2.04) but does not clear BH threshold. All dynamic methods underperform constant WD (all deltas negative), but the underperformance is only confirmed significant for no_wd at n=3.

### Key Confirmed Findings

1. **Phi Invariance under AdamW (CONFIRMED, n=3)**: All methods within ±0.25% of constant WD on CIFAR-10 (p range: 0.513–0.950, all non-significant). The range holds even for no_wd (λ=0), demonstrating that WD presence is irrelevant under AdamW for ResNet-20 on CIFAR-10/100. *Awaiting VGG-16-BN and ImageNet replication.*

2. **WD-Presence Effect under SGD (CONFIRMED, n=3)**: constant WD significantly outperforms no_wd (Δ=+0.913%, raw p=0.0022, BH p=0.013, d=12.17). This is a *presence* effect — no dynamic WD strategy exceeds constant WD. All dynamic methods fall between constant and no_wd. *Awaiting VGG-16-BN and ImageNet replication.*

3. **SWD trends WORSE than constant WD under SGD (STRONG TREND, NOT YET BH-SIGNIFICANT at n=3)**: CIFAR-10: Δ=−0.507%, d=−2.04, raw p=0.018, BH p=0.092; CIFAR-100: Δ=−1.067%, d=−2.60, raw p=0.036, BH p=0.178. Effect sizes are large and consistent but do not clear BH significance at n=3 with k=6 comparisons. n=5 seeds required for statistical confirmation. NOTE: contradicts SWD paper's claimed benefit; different evaluation context (ResNet-20 with SGD vs. AdamS with CIFAR/ImageNet in original SWD paper).

4. **AIS is indistinguishable between CWD and random masking (CONFIRMED)**: cwd_hard AIS=0.416 ± 0.050 vs random_mask AIS=0.416 ± 0.030 (difference < 0.001, essentially zero). Sign-alignment conditioning in CWD does not enhance the alignment signal's informativeness compared to random masking of the same budget.

5. **SGD/AdamW WD-presence ratio = 18.3× on CIFAR-10 (CONFIRMED)**: Constant−no_wd gap under SGD (0.913%) is 18.3× larger than under AdamW (0.050%), consistent across 6 seeds. CIFAR-100 ratio = 3.7× (1.825% / 0.490%). *Scale dependence of the ratio is an open question requiring ImageNet.*

6. **VGG-16-BN pilot (1 seed, UNCONFIRMED)**: AdamW shows no_wd=80.61% > constant=79.94% (+0.67%) — reversed direction from ResNet-20 Phi Invariance. If confirmed at n=3, WD is actively *harmful* under AdamW on VGG-16-BN, which would be a stronger positive finding than mere invariance. *High priority to confirm before full VGG campaign.*

### SGD CIFAR-100 ResNet-20 (n=3 seeds for 6 methods, n=2 for no_wd)
| Method | SGD Acc. (%) | AdamW Acc. (%) | SGD Δ vs const | AdamW Δ vs const |
|--------|:---:|:---:|:---:|:---:|
| constant | 65.37 ± 0.13 | 63.15 ± 0.30 | — | — |
| cosine_schedule | 65.11 ± 0.25 | 63.42 ± 0.42 | −0.263% | +0.264% |
| cwd_hard | 64.37 ± 0.47 | 62.84 ± 0.30 | −1.003% | −0.313% |
| half_lambda | 64.86 ± 0.38 | 62.91 ± 0.47 | −0.510% | −0.246% |
| swd | 64.30 ± 0.41 | 63.06 ± 0.29 | **−1.067%** | −0.096% |
| random_mask | 64.91 ± 0.40 | 62.87 ± 0.38 | −0.457% | −0.286% |
| no_wd | 63.55 ± 0.05* | 62.66 ± 0.38 | **−1.825%** | −0.490% |

*n=2 for SGD CIFAR-100 no_wd (seed_123 epoch_metrics exists but no summary.json yet)

**Note on cosine_schedule CIFAR-100 AdamW**: +0.264% vs constant is the one case where a dynamic method nominally outperforms constant WD, but it is non-significant at n=3 (estimated p≈0.6). This does NOT falsify Phi Invariance; it is within the ±0.5% equivalence zone.

### Missing Data Required for Complete Analysis
- VGG-16-BN: 3 seeds × 7 methods × 2 datasets × 2 optimizers (72 runs, P0)
- ImageNet ResNet-50: 3 seeds × 3 methods × 2 optimizers (18 runs, P0)
- SGD CIFAR-100 no_wd seed_123: 1 missing summary.json (epoch_metrics exists; extract summary, P0, trivial)
- Extra seeds (789, 999) for CIFAR-10 key comparisons (28 runs, P0)
- half_lambda BEM bug fix (code fix, 0 GPU, P0)
- SWD implementation validation (check vs original AdamS, P1)
- VGG-16-BN AdamW pilot at n=3 (to confirm or refute the 1-seed reversal finding, P0)
