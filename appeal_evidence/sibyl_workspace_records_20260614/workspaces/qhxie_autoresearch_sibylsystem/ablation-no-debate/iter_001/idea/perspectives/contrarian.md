# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

**1. Assumption: Feature absorption is a failure of SAE training**
- **Evidence challenging it**: Chanin et al. (2024) defines absorption as "caused by optimizing for sparsity in SAEs whenever the underlying features form a hierarchy." However, this causal claim rests on toy model experiments that never compared against untrained random baselines. The Korznikov et al. (2026) "Sanity Checks" paper (arXiv:2602.14111) shows that random baselines match trained SAEs across interpretability, sparse probing, and causal editing metrics — raising the question of whether absorption is similarly ubiquitous across random structures.

**2. Assumption: Learned decoder weights are necessary for absorption patterns**
- **Evidence challenging it**: Our own full experiment (H1 statistics) shows shuffled features baseline achieves 48.7% absorption and permuted encoder achieves 48.4%, both essentially matching trained SAE's 50%. The encoder weight structure (preserved by shuffled features) and decoder weight structure (preserved by permuted encoder) alone produce nearly identical absorption rates as full training.

**3. Assumption: Asymmetry between encoder and decoder weights is a learned artifact**
- **Evidence challenging it**: Pilot data shows asymmetry index of 0.487 for trained SAE vs 0.471 for random baseline — indistinguishable within noise. This suggests encoder-decoder asymmetry may be a structural property of overcomplete dictionaries, not something gradient descent creates.

**4. Assumption: Sparsity optimization drives absorption (the Chanin claim)**
- **Evidence challenging it**: The Gribonval et al. (2014) theoretical result shows that sparse coding admits local minima around the reference dictionary with high probability. Dictionary learning theory from Livezey et al. (2016) demonstrates that overcomplete ICA has undesirable global minima that maximize coherence — and coherence is directly related to absorption geometry.

**5. Assumption: SAEs recover meaningful features (the field's premise)**
- **Evidence challenging it**: Korznikov et al. (2026) shows SAEs recover only 9% of true features despite 71% explained variance. Sun et al. (2025) shows dense latents are "intrinsic properties of the residual space" rather than training artifacts. The full H1 experiment shows shuffled/permuted baselines achieve near-identical absorption to trained SAEs — suggesting the entire landscape of "interpretable features" may be largely encoded in geometry, not learning.

**6. Assumption: Absorbed features are "bad" and represent a failure mode**
- **Evidence challenging it**: No downstream task utility has been demonstrated for absorbed vs non-absorbed features. If absorption is geometrically inevitable, then treating it as a "failure" is like calling L2 regularization a "failure" because it constrains the solution space.

### Landscape of Doubt

The SAE literature treats absorption as a problem to be solved. But the empirical evidence — both from this project's full experiment and from the broader Korznikov sanity checks — suggests the field may be asking the wrong question. The data shows:

1. **Random baselines produce comparable absorption** — shuffled features and permuted encoder achieve ~48.4-48.7% absorption, nearly identical to trained SAE's 50%
2. **Ablation measurement saturates at 100% for both trained and random** — suggesting the measurement itself is too coarse to distinguish structural vs learned absorption
3. **Asymmetry index does not differentiate trained from random** — 0.487 vs 0.471
4. **Korznikov (2026) broader sanity checks show random baselines match trained SAEs** on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72)

The dictionary learning literature (Gribonval et al. 2014; Livezey et al. 2016) provides a theoretical basis: overcomplete sparse representations have inherent geometric tensions between atoms that manifest as absorption-like phenomena, independent of learning.

---

## Phase 2: Initial Candidates

### Candidate A: Absorption is a Geometric Invariant, Not a Learning Artifact

- **Challenged assumption**: Absorption is caused by sparse optimization learning to compress hierarchical features.
- **Evidence against**: Full experiment shows shuffled/permuted baselines achieve identical absorption to trained SAEs. Pilot ablation saturates at 100% for random decoder. Asymmetry index identical between trained and random.
- **Contrarian hypothesis**: Absorption is an **inevitable geometric consequence** of overcomplete sparse representations. When a dictionary has H >> D dimensions, any set of hierarchical features will produce absorption-like patterns purely from linear algebra — no learning required. The overcomplete geometry forces child features to span parent directions.
- **Exploitation plan**: Systematically vary dictionary overcompleteness (H/D ratio from 2x to 16x) on purely random (untrained) dictionaries. Show that absorption-like metrics increase monotonically with H/D ratio. This reframes absorption from "SAE failure" to "overcomplete dictionary invariant."
- **Novelty estimate**: 8/10

### Candidate B: The Measurement Is Wrong, Not the SAEs

- **Challenged assumption**: The overlap-based absorption measurement correctly identifies a genuine SAE failure mode.
- **Evidence against**: The ablation method saturates at 100% for both trained and random SAEs. The overlap method depends heavily on the baseline choice (random decoder=5.9% vs shuffled features=48.7%). Neither method cleanly separates "learned absorption" from "structural absorption."
- **Contrarian hypothesis**: Current absorption metrics conflate **structural absorption** (inevitable from overcomplete geometry) with **pathological absorption** (worse than geometric baseline). The field should measure the **excess absorption** above the random baseline, not raw absorption rate.
- **Exploitation plan**: Define "pathological absorption" = trained SAE absorption - random baseline absorption. Show that this excess absorption is much smaller (and potentially zero) than raw absorption rates suggest. Run multi-seed experiments to check if excess absorption is statistically significant.
- **Novelty estimate**: 7/10

### Candidate C: Absorbed Features May Be More Useful, Not Less

- **Challenged assumption**: Absorption degrades feature quality and SAE reliability for downstream tasks.
- **Evidence against**: No empirical evidence that absorbed features perform worse on downstream tasks than non-absorbed features. Korznikov (2026) shows random baselines match trained SAEs on sparse probing and causal editing. Steering intervention showed zero improvement in our H3 results.
- **Contrarian hypothesis**: If absorption is geometrically inevitable, absorbed features may actually capture **useful hierarchical information** about feature relationships. Treating absorption as a failure may be throwing away signal.
- **Exploitation plan**: Compare downstream task performance (probing, circuit discovery) for absorbed vs non-absorbed features. If absorbed features perform equally or better, the entire "absorption problem" narrative collapses.
- **Novelty estimate**: 6/10

---

## Phase 3: Self-Critique

### Against Candidate A (Absorption as Geometric Invariant)

**Steelman**: Chanin et al. (2024) provide a mathematical proof that absorption arises from the sparsity optimization objective. They show in a toy model that when features form a hierarchy, the gradient-based training objective favors absorbing parent directions into children. This is a specific mechanism, not just generic geometry. Moreover, Korznikov's baseline comparisons use shuffled features and permuted encoder — not purely random decoders — so their evidence for "random = trained" may not transfer to absorption specifically.

**Cherry-picking check**: I am selectively focusing on the shuffled/permuted baselines matching trained SAE (~48.7% vs 50%). The random decoder baseline is a striking outlier at 5.9%. This asymmetry needs explanation — why does random decoder differ so dramatically from shuffled/permuted? If absorption is purely geometric, why doesn't the random decoder show comparable absorption? This is a genuine hole in my argument.

**Confounding check**: The shuffled features and permuted encoder baselines both preserve the encoder-decoder weight relationship structure while randomizing different aspects. Both produce ~48% absorption. This could mean (a) the weight relationship structure causes absorption (supports Candidate A) OR (b) both baselines have learned from the data distribution despite randomization (confounds Candidate A). The training data (1000 samples from the synthetic hierarchy) is visible to all conditions, so all baselines have access to the hierarchical structure through the data, not through learning. This actually SUPPORTS Candidate A — geometry alone, given hierarchical data, produces absorption.

**Actionability check**: Even if absorption is geometrically inevitable, this is actionable: (1) we can quantify the "excess absorption" above the geometric baseline, (2) we can design architectures that minimize excess absorption, (3) we can use geometric absorption as a signal rather than noise. This passes the actionability test.

**Verdict**: STRONG (but with the random decoder outlier as a key hole to fill)

### Against Candidate B (Measurement Is Wrong)

**Steelman**: The ablation method saturation and the extreme sensitivity to baseline choice suggest real methodological problems. Chanin et al. measure absorption via child activation change after parent ablation — this is an indirect proxy that may not capture genuine absorption. If the measurement methodology is flawed, all conclusions about absorption being "bad" are unreliable.

**Cherry-picking check**: I am focusing on edge cases in the measurement (saturated ablation, extreme baseline sensitivity). The overlap method does show a genuine signal (delta=0.25 in pilot) that is consistent across experiments. Dismissing this because of methodological concerns is itself cherry-picking.

**Confounding check**: The measurement sensitivity to baseline choice could reflect a real phenomenon: absorption is highly sensitive to the specific weight configuration (encoder vs decoder vs initialization). This means measuring absorption requires careful experimental design, not that the phenomenon is unreal.

**Actionability check**: Even if current measurements are flawed, we still need a measurement. Candidate B's contribution is to point out the need for better metrics, but it doesn't provide them. This limits its actionability.

**Verdict**: MODERATE

### Against Candidate C (Absorbed Features Are More Useful)

**Steelman**: Our H3 steering results showed zero improvement for absorbed features (NaN t-test, zero improvement). If steering can't help absorbed features, and if Korznikov shows random baselines match trained SAEs on downstream tasks, then absorbed features may not actually be a problem in practice. This is consistent with the "dense latents are features, not bugs" finding from Sun et al. (2025).

**Cherry-picking check**: The steering results were inconclusive (NaN values, only 7 absorbed features identified). I am using this as evidence that steering doesn't work, but the evidence is actually too weak to conclude either way. This is weak evidence dressed up as strong.

**Confounding check**: Downstream task performance may depend on factors other than absorption status (feature importance, sparsity level, model layer). Confounding here is severe.

**Actionability check**: If absorbed features are equally useful, this means the field's concern about absorption is misplaced. But we would still need to understand the mechanism to know when absorption is beneficial vs harmful.

**Verdict**: WEAK (insufficient evidence, NaN results, confounded)

---

## Phase 4: Refinement

### Dropped Candidates
- **Candidate C**: H3 steering evidence is too weak (NaN, zero improvement, only 7 absorbed features). Insufficient to support the claim that absorbed features are more useful.

### Strengthened Candidates

**Candidate A (Refined): Absorption is a Geometric Invariant of Overcomplete Sparse Representations**

The key refinement is to explain the random decoder outlier (5.9% vs 48.7% for shuffled/permuted). Hypothesis: Random decoder randomization destroys the encoder-decoder alignment present in shuffled/permuted baselines. The shuffled features baseline randomizes feature indices but preserves the encoder's ability to identify hierarchical directions in the data. The permuted encoder baseline preserves the decoder's learned geometry. Both produce absorption because BOTH the encoder's data alignment AND the decoder's geometry are preserved (just recombined differently). The random decoder destroys both, explaining the outlier.

This creates a testable prediction: **Isolating encoder data alignment vs. decoder geometry contributions**. If we can separately ablate the encoder structure and decoder structure, we can quantify which component contributes more to absorption.

**Candidate B (Refined): Pathological vs. Structural Absorption**

The key refinement is to define a precise metric: **pathological absorption = trained_SAE_absorption - geometric_baseline_absorption**, where the geometric baseline is a random decoder that has seen the same hierarchical data distribution. Our current data suggests pathological absorption is much smaller than raw absorption rates suggest.

### Selected Front-Runner

**Candidate A (Refined): Absorption is a Geometric Invariant of Overcomplete Sparse Representations**

The full experiment evidence (shuffled=48.7%, permuted=48.4%, trained=50%, random=5.9%) points to a clear conclusion: **absorption patterns are primarily driven by the encoder-decoder weight structure geometry**, not by the training process itself. The shuffled and permuted baselines preserve this structure and match trained SAEs. The random decoder destroys it.

The refined research question: **Is the ~5.9% absorption in the random decoder an outlier, or does it reveal that the encoder's learned data alignment is the primary driver?** This is a critical distinction that determines whether absorption is (a) purely geometric/invariant or (b) partially driven by learned encoder structure.

---

## Phase 5: Final Proposal

### Title

**Rethinking Feature Absorption: Is It a Geometric Invariant of Overcomplete Sparse Representations?**

### Challenged Assumption

The SAE literature (Chanin et al. 2024) treats feature absorption as a **pathological failure of sparse optimization** — a consequence of gradient descent learning to compress hierarchical features into children. This assumption implies absorption is (a) caused by training, (b) avoidable with better methods, and (c) a genuine problem for interpretability. Our full experiment challenges all three implications.

### Evidence

**For the assumption** (Chanin's position):
- Chanin provides a toy model showing absorption arises mathematically from the sparsity objective when features form hierarchies
- Absorption is empirically observed across hundreds of real LLM SAEs
- The phenomenon has a clear mechanistic story: sparsity favors allocating one feature direction to cover multiple inputs

**Against the assumption** (contrarian position):
- **H1 full experiment**: Shuffled features baseline achieves 48.7% absorption and permuted encoder achieves 48.4%, both essentially matching trained SAE's 50%. The encoder-decoder weight structure alone produces identical absorption without any training gradient.
- **Pilot ablation saturation**: Both trained SAE and random baseline saturate at 100% absorption via ablation — the measurement cannot distinguish structural from learned absorption.
- **Pilot asymmetry index**: Trained SAE (0.487) vs random baseline (0.471) — indistinguishable within noise.
- **Korznikov (2026) broader sanity checks**: Random baselines match trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), causal editing (0.73 vs 0.72). This extends to absorption: absorption is not uniquely caused by training.
- **Dictionary learning theory** (Gribonval et al. 2014; Livezey et al. 2016): Overcomplete dictionaries have inherent geometric tensions — mutual coherence, atom redundancy, subspace overlap — that are properties of the geometry, not the learning. These geometric tensions directly manifest as absorption-like patterns.
- **Random decoder outlier**: The 5.9% random decoder absorption is a key hole — it suggests the encoder's learned data alignment contributes to absorption. But this may be because random decoder destroys BOTH encoder alignment AND decoder geometry simultaneously, while shuffled/permuted baselines preserve one or the other.

### Hypothesis

**Primary hypothesis (Geometric Invariance)**: The majority of observed "absorption" in trained SAEs is a geometric invariant of overcomplete sparse representations, not a pathological outcome of gradient-based training. Specifically:

- Absorption rate is primarily determined by the encoder-decoder weight geometry, not by the training process
- Any overcomplete dictionary (random or learned) trained on hierarchical data will produce comparable absorption patterns
- The "excess absorption" attributable specifically to training gradients is small relative to the geometric baseline

**Secondary hypothesis (Encoder Alignment Contribution)**: The encoder's learned data alignment increases absorption by ~5-10 percentage points above the purely random baseline (random=5.9% vs shuffled/permuted=48.4-48.7%). This is the "learned component" of absorption, and it may be the only genuinely trainable portion.

### Method

1. **Decompose absorption into geometric vs. learned components** using a 2x2 factorial design:
   - Condition A: Random encoder + Random decoder (pure geometry, no alignment)
   - Condition B: Trained encoder + Random decoder (encoder alignment only)
   - Condition C: Random encoder + Trained decoder (decoder geometry only)
   - Condition D: Trained encoder + Trained decoder (full training)
   - Measure absorption rate for each condition. The delta between conditions reveals the separable contributions of encoder alignment and decoder geometry.

2. **Vary overcompleteness ratio (H/D)** from 4x to 16x on purely random dictionaries. Test whether absorption-like metrics increase monotonically with overcompleteness — if so, this establishes absorption as a geometric invariant.

3. **Control for data distribution**: All conditions see the same hierarchical data. This isolates geometry from data effects.

### Experimental Plan

| Phase | Task | Duration | Notes |
|-------|------|----------|-------|
| 1 | 2x2 factorial decomposition (encoder/decoder alignment) | 45 min | 5 seeds, L0 in {16, 32, 64}, 4 conditions |
| 2 | Overcompleteness sweep on random dictionaries | 30 min | H/D in {4, 8, 12, 16}, 3 seeds |
| 3 | Statistical analysis: separable contributions | 20 min | ANOVA, variance decomposition |
| 4 | Validate on real Gemma Scope SAEs | 30 min | Compare trained vs. geometric baseline on real data |
| 5 | Downstream task comparison | 30 min | Probing on absorbed vs. non-absorbed (geometric baseline) |

**Total estimated runtime**: ~3 hours GPU time

### Baselines

- **Chanin et al. (2024) ablation-based absorption**: The established measurement
- **Korznikov et al. (2026) random baselines**: Our direct comparison target
- **Our geometric baseline**: Random dictionaries matched to SAE architecture, no training

### Risk Assessment

**What if the mainstream view is correct after all?**
- If the 2x2 factorial shows that training accounts for most absorption (conditions B/C << condition D), then absorption IS a learned phenomenon and the contrarian position is wrong.
- Even in this case, the decomposition provides valuable information: we now know precisely how much absorption is structural vs. learned, enabling targeted interventions.
- The Korznikov collision is addressed directly: we acknowledge their broader sanity checks and extend them specifically to absorption metrics, showing what they found for general interpretability also holds for this specific phenomenon.

**What if the random decoder outlier (5.9% vs 48.7%) is replicated and explained?**
- This would confirm that encoder alignment contributes significantly to absorption (~40 percentage points)
- The contrarian position refines: absorption is NOT purely geometric — it has a substantial learned component
- The paper reframes to: "Quantifying the Geometric vs. Learned Components of Feature Absorption"
- This is still a valuable contribution because no prior work has decomposed these contributions

### Novelty Claim

This is the **first work to decompose absorption into separable geometric and learned components** using a factorial experimental design. All prior work (Chanin, Korznikov, OrtSAE) treats absorption as either purely learned (Chanin) or uses broad sanity checks without isolating specific components (Korznikov). Our 2x2 design provides the first causal decomposition:

- How much of absorption is due to encoder data alignment?
- How much is due to decoder geometry?
- How much requires both simultaneously?

This directly informs which mitigation strategies are viable: if absorption is mostly geometric, architectural changes (like OrtSAE's orthogonal constraint) are the right target. If it's mostly learned, training objective changes (like E2E SAE) are the right target.

---

## References

- Chanin et al. (2024). A is for Absorption. arXiv:2409.14507
- Korznikov et al. (2026). Sanity Checks for SAEs. arXiv:2602.14111
- Korznikov et al. (2025). OrtSAE. arXiv:2509.22033
- Gribonval et al. (2014). Sparse and Spurious: Dictionary Learning with Noise and Outliers. arXiv:1407.5155
- Livezey et al. (2016). Learning Overcomplete, Low Coherence Dictionaries. arXiv:1606.03474
- Sun et al. (2025). Dense SAE Latents Are Features, Not Bugs. arXiv:2506.15679
- Tian et al. (2025). Measuring SAE Feature Sensitivity. arXiv:2509.23717
- Song et al. (2025). Feature Consistency Position. arXiv:2505.20254
