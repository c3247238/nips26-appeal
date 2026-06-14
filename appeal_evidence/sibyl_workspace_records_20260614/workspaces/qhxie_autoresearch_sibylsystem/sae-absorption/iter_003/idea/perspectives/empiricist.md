# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024). A is for Absorption. arXiv:2409.14507 (NeurIPS 2025)** — Defines the canonical absorption metric: k-sparse probing to identify false negatives, integrated-gradients ablation to identify absorbing latents, cosine similarity threshold (0.025) and gap criterion (≥1.0) to confirm absorption. Key methodological pitfall: requires known probe directions; absorption rate may undercount partial absorption cases. Reports 15–35% absorption across Gemma Scope 16k/65k. Code: github.com/lasr-spelling/sae-spelling.

2. **Karvonen et al. (2025). SAEBench. arXiv:2503.09532** — The current gold-standard benchmark for SAEs. Absorptions metric code included. Critical methodological finding: proxy metrics (CE loss, sparsity L0) do NOT reliably predict practical performance, including absorption. This is the core empirical basis for why a rigorous absorption study is needed: existing measurements may be artifacts of metric confounds. Provides timing baseline: ~65 min per SAE full evaluation on an RTX 3090.

3. **Gao et al. (2024). Scaling and Evaluating Sparse Autoencoders (TopK). arXiv:2406.04093 (ICLR 2025)** — Establishes the sparsity-fidelity evaluation framework. Key for our experiments: shows that at low L0, TopK SAEs have significantly worse feature absorption than L1 SAEs (SAEBench finding). Critical confound: SAE architecture type (L1 vs. TopK vs. JumpReLU) and the target L0 are interrelated — fair comparison must hold one fixed.

4. **Tian et al. (2025). Measuring Sparse Autoencoder Feature Sensitivity. arXiv:2509.23717** — Frames absorption as a special case of poor feature sensitivity. Provides complementary metric that does not require known probe directions (uses nearest-neighbor activation consistency). Key methodological insight: many features appear interpretable from their top activating examples but fail on broader input distributions — absorption is not the only cause of this sensitivity failure. Must include sensitivity as a control metric.

5. **Korznikov et al. (2025). OrtSAE. arXiv:2509.22033** — Introduces three distinct absorption metrics beyond Chanin's rate: mean absorption fraction, full-absorption score, and feature splits count. These three metrics are complementary — a feature can have low absorption rate (Chanin metric) but high mean absorption fraction if partial absorption is common. Any rigorous absorption study must report all three to avoid cherry-picking.

6. **Korznikov et al. (2026). Sanity Checks for SAEs. arXiv:2602.14111** — Shows SAEs recover only 9% of true features in synthetic settings and that random baselines match trained SAEs on interpretability benchmarks. Critical methodological baseline: any absorption study must demonstrate that measured "absorption" is not simply noise or random probe failure. The 9% recovery rate is the sobering lower bound that any claimed improvement must surpass.

7. **Narayanaswamy et al. (2026). Masked Regularization. arXiv:2604.06495** — Reduces absorption via token masking during training. Methodological insight: disrupting co-occurrence patterns during training reduces absorption, directly implying that co-occurrence is a causal driver of absorption. Provides a "manipulation check" for any causal claim about co-occurrence causing absorption.

8. **Chanin, Dulka, & Garriga-Alonso (2025). Feature Hedging. arXiv:2505.11756** — Identifies hedging as the absorption-dual failure mode (narrow SAEs merge correlated features; wide SAEs absorb them). Critical confound for any absorption experiment: when absorption rate decreases (e.g., in wider SAEs), hedging may increase. A study that claims to "reduce absorption" without measuring hedging may be measuring a redistribution of failure, not a genuine improvement.

9. **Li et al. (2025). Understanding SAE Scaling in Feature Manifolds. arXiv:2509.02565** — Shows Gemma Scope SAE latent activation frequencies decay with slope ≈ −0.74 (Zipfian-like). Directly relevant: the parent feature's frequency relative to child features in the co-occurrence graph determines the frequency ratio ρ that the theoretical perspective identifies as the key predictor of absorption. Provides empirical frequency data without additional experiments.

10. **Bussmann et al. (2025). Matryoshka SAE. arXiv:2503.17547 (ICML 2025)** — Best SAE architecture on SAEBench absorption metric. Key for controlled comparison: reports absorption reduction while acknowledging trade-off with feature hedging in inner levels. Methodologically important: any comparison of absorption across architectures must use the same base model, same layers, and same L0 — none of the current papers do this.

11. **Song et al. (2025). Feature Consistency in SAEs. arXiv:2505.20254** — Shows SAE features are inconsistent across training runs; proposes PW-MCC metric. Critical for experimental design: if absorption rate is measured from a single SAE training run, the result may be an artifact of initialization. Any absorption study should report absorption rates across multiple SAE training seeds, or use pre-trained SAEs with known stable configurations (Gemma Scope).

12. **DeepMind Safety Research Team (2025). Negative Results for SAEs on Downstream Tasks. Medium** — Found that 1-sparse SAE probes fail catastrophically at detecting harmful intent while dense linear probes succeed. The highest-stakes empirical finding motivating absorption research. Critical for experimental design: a rigorously designed study should replicate and extend this finding, measuring whether the probe performance gap is specifically driven by absorption (measured absorption rate of the relevant latents) vs. other causes.

### Experimental Landscape

**What has been properly tested:**
- Absorption in the first-letter spelling task (Chanin et al., validated on hundreds of Gemma Scope and Llama SAEs): absorption rate 15–35% across widths; higher for wider/more sparse SAEs.
- Architectural comparison on SAEBench: Matryoshka SAE reduces absorption; TopK/JumpReLU worsen it.
- OrtSAE: 65% absorption reduction via orthogonality penalty; independently validated.
- ATM-SAE: best reported absorption scores (0.0068 vs. TopK 0.1402 on Gemma-2-2B).

**What is accepted without sufficient evidence:**
- "Absorption is caused by co-occurrence" — Chanin et al. demonstrate the mechanism in a 2-feature toy model, but no controlled experiment manipulates co-occurrence statistics in real LLM SAEs while holding other variables constant.
- "Wider SAEs have higher absorption" — directional evidence exists but no study controls for the confound that wider Gemma Scope SAEs were also trained with different L0 targets.
- "Absorption generalizes beyond the first-letter task" — universally assumed but never systematically tested with proper negative controls (non-hierarchical feature pairs with matched frequency ratios).
- "Absorption is the main driver of the SAE probe vs. linear probe performance gap" — DeepMind finding is correlational; no one has decomposed the gap into absorption-attributable vs. other-attributable fractions.

**Methodological gaps (where the field is most vulnerable):**
- No controlled experiment isolating co-occurrence frequency as the causal driver of absorption while controlling for feature direction correlation, SAE width, and L0.
- No study uses proper negative controls (non-hierarchical feature pairs matched on frequency ratio) to confirm that the absorption metric is measuring hierarchy-specific failure, not generic feature unreliability.
- No statistical power analysis: are absorption rate differences across configurations statistically significant, or within noise bounds given the small number of test letters (26)?
- No study replicates absorption measurement from multiple SAE random seeds on the same configuration to establish measurement reliability.
- Absorption rate and feature sensitivity (Tian et al.) have not been compared on the same SAEs — it is unknown whether they measure the same construct or complementary ones.

---

## Phase 2: Initial Candidates

### Candidate A: Causal Isolation of Co-occurrence as the Driver of Absorption

**Core hypothesis:** Feature absorption rate for a (parent, child) feature pair is causally driven by the co-occurrence frequency ratio ρ = p_child/p_parent, and this effect is *separable from* (a) feature direction correlation (cosine similarity between decoder vectors), (b) SAE width, and (c) mean sparsity L0.

**Falsification criterion (decided before seeing results):** If, in a controlled analysis that holds ρ fixed while varying direction correlation, absorption rate does NOT vary significantly with direction correlation (one-way ANOVA, p > 0.05), then the "direction correlation is a confound" claim is refuted. If absorption rate does not increase monotonically with ρ when holding all other variables fixed (Spearman ρ < 0.3), the core hypothesis is refuted.

**Evaluation protocol:**
- Dataset: First-letter spelling task on Gemma Scope (Gemma-2-2B), reusing sae-spelling codebase directly.
- For each of the 26 letters, compute: (1) ρ = letter frequency / mean token-starting-with-letter frequency from OpenWebText; (2) decoder cosine similarity between the identified parent latent and child latents; (3) SAE width W; (4) mean sparsity L0.
- Use Gemma Scope SAEs at 4 widths (1k, 4k, 16k, 65k) × 4 layers (8, 12, 16, 20) × 2 architectures (JumpReLU baseline, TopK).
- For each SAE, measure absorption rate per letter using the canonical Chanin metric.
- Statistical analysis: partial correlation of absorption rate with ρ, controlling for cosine similarity, W, and L0. Report partial Spearman ρ with bootstrap 95% CI (1000 iterations).

**Ablation plan:**
- Ablation 1: Fix W = 16k, vary ρ (use 26 letters as natural ρ sweep). Tests: does ρ predict absorption?
- Ablation 2: Fix ρ ≈ 0.3 (select letters with similar frequency ratio), vary W across 4 widths. Tests: does width independently predict absorption, or is W effect mediated by L0?
- Ablation 3: Fix W and ρ, measure decoder cosine similarity between parent and child latents. Tests: is direction correlation an additional driver?
- Ablation 4 (negative control): Select non-hierarchical feature pairs with ρ matched to the letter pairs. Apply the same absorption metric. Tests: does the metric detect absorption only for hierarchical pairs, or for any co-occurring features?

**Confounders identified:**
1. *Width-L0 confound:* Gemma Scope SAEs at different widths have different L0 targets. Must hold L0 constant across widths by selecting SAEs from the Gemma Scope suite with matched L0 values (multiple SAEs per width are available with different sparsity levels).
2. *Architecture confound:* JumpReLU vs. TopK SAEs may differ in absorption due to training dynamics, not just the activation function. Must report architecture as a covariate.
3. *Layer confound:* Different layers represent different information types; features at early layers may have different co-occurrence structure than late layers. Must include layer as a fixed effect.
4. *Token frequency confound:* Letter frequency itself may correlate with how well the SAE represents that feature — high-frequency letters may have better-trained features. Must include raw letter frequency as a control variable.

**Pilot design (< 15 min):**
Load Gemma Scope 16k JumpReLU SAE at layer 12 of Gemma-2-2B (one SAE, ~2 min to load). Compute ρ for letters A, E, I, O, U from OpenWebText token counts (5 min). Run canonical absorption measurement for each of the 5 letters (~1-2 min per letter). Plot absorption rate vs. ρ. If Spearman ρ > 0.5 for this 5-letter sample, the trend is promising and the full experiment is justified. Total: ~15 min.

---

### Candidate B: Cross-Domain Generalization of Absorption with Proper Negative Controls

**Core hypothesis:** Feature absorption generalizes beyond the first-letter spelling task to semantically richer hierarchies (entity type: ANIMAL ⊃ DOG; syntactic: VERB ⊃ IRREGULAR-PAST-TENSE), with absorption rate positively correlated with the parent-child frequency ratio ρ; and absorption is NOT present (or significantly lower) for non-hierarchical feature pairs with matched frequency ratios.

**Falsification criterion (decided before seeing results):** If absorption rate for the entity-type hierarchy is not significantly higher than for the matched negative-control (non-hierarchical) pair (paired t-test, p > 0.1 with Bonferroni correction), the "absorption generalizes" claim is refuted. If absorption rate does not positively correlate with ρ across hierarchy types (Spearman ρ < 0.2), the hypothesis is refuted.

**Evaluation protocol:**
- Primary metric: Chanin et al. canonical absorption rate (sae-spelling code, directly reused).
- Secondary metric: Feature sensitivity (Tian et al., arXiv:2509.23717) — measure sensitivity for the same feature pairs to check whether absorption and sensitivity measure the same construct.
- Benchmarks: Gemma Scope 16k JumpReLU SAE at layer 12 (fixed across all hierarchy types for fair comparison).
- Statistical test: Paired t-test on absorption rates between hierarchy-present and hierarchy-absent (negative control) conditions, matched on ρ. Minimum 3 probe training seeds per hierarchy type.

**Ablation plan:**
- Ablation 1: Entity-type hierarchy (ANIMAL ⊃ DOG). Construct 200 sentences per class using WordNet + templated generation.
- Ablation 2: Syntactic hierarchy (VERB ⊃ IRREGULAR-PAST-TENSE). Use Penn Treebank/Universal Dependencies annotated data.
- Ablation 3: First-letter baseline (from Chanin et al.) — exact replication to validate our pipeline produces identical numbers to the published paper.
- Ablation 4: Negative control. Two co-occurring but non-hierarchical features: "contains the letter E" AND "is a content word" — these co-occur at the same frequency ratio as letter A ⊃ A-starting token pairs, but neither implies the other. If the metric is hierarchy-specific, this should show near-zero absorption.
- Ablation 5: Frequency-matched negative control. For each positive hierarchy pair, construct a non-hierarchical pair with the same ρ by selecting random co-occurring features from the SAE's activation statistics.

**Confounders identified:**
1. *Probe training confound:* Linear probe accuracy affects which "should have fired" examples are identified. Must ensure probe accuracy > 80% for all hierarchy types before running absorption measurement; report probe accuracy alongside absorption rate.
2. *Hierarchy knowledge confound:* We know the first-letter hierarchy by construction; entity-type hierarchy relies on WordNet, which may not match what the LM learned. Must validate that the LM actually represents the proposed hierarchy by checking that a linear probe correctly identifies parent and child concepts at the tested layer.
3. *Dataset construction confound:* Sentences constructed to probe entity-type hierarchies may have distributional artifacts (e.g., all ANIMAL sentences mentioning pets) that create spurious co-occurrence with other features. Must validate probe activation distributions look similar to naturally occurring tokens for that concept.
4. *Measurement metric confound:* The Chanin metric uses an integrated-gradients attribution with a fixed threshold (cosine similarity 0.025). This threshold was calibrated for the first-letter task and may not transfer to other hierarchy types. Must report sensitivity analysis over the threshold.

**Pilot design (< 15 min):**
Load Gemma Scope 16k SAE at layer 12. Construct 50 ANIMAL sentences and 50 non-ANIMAL sentences using templates ("The [ANIMAL] ran through the forest." vs. "The scientist analyzed the data."). Train LR probe on residual stream activations. Check probe accuracy (>80% → proceed). Measure absorption rate using the sae-spelling pipeline. Total: ~12-15 min for this minimal slice.

---

### Candidate C: Decomposing the SAE-vs-Linear-Probe Performance Gap via Absorption Attribution

**Core hypothesis:** The performance gap between 1-sparse SAE probes and dense linear probes on downstream classification tasks (specifically harmful intent detection, as documented by DeepMind 2025) is primarily attributable to feature absorption: the relevant "harmful intent" features are absorbed into more specific context features (roleplay framing, instruction-following tone) that frequently co-occur with harmful content.

**Falsification criterion (decided before seeing results):** If measured absorption rate for the top SAE latents associated with the "harmful intent" feature is < 15% (below the baseline first-letter absorption rate), the "absorption is primary driver" claim is refuted. If the SAE-vs-linear-probe gap does not decrease significantly when absorbed latents are manually "un-absorbed" (i.e., added to the active latent set), the causal claim is refuted.

**Evaluation protocol:**
- Dataset: AdvBench or ToxiGen (400 examples: 200 harmful + 200 harmless), balanced.
- Metrics: F1 score, precision, recall for (a) 1-sparse SAE probe, (b) k-sparse SAE ensemble (k = 1, 5, 10, 20), (c) dense linear probe on residual stream activations.
- Statistical test: McNemar test for paired binary classification (harmful vs. harmless) comparing SAE probe and linear probe. Bootstrap 95% CI on the F1 gap.
- For absorption attribution: run the Chanin absorption metric adapted to the "harmful intent" probe direction (instead of a letter probe). Measure absorption rate. For tokens where the harmful-intent latent fails to fire but the probe correctly classifies, manually add the identified "absorbing latents" to the active set and re-run the downstream classifier. Measure F1 improvement from this intervention.

**Ablation plan:**
- Ablation 1: Replicate the SAE-vs-linear-probe gap on the standard task (confirming DeepMind finding on our specific dataset).
- Ablation 2: k-sparse SAE ensemble probe (k increases from 1 to 20). Tests: does adding more latents close the gap (indicating the information is present in SAE but requires more budget)?
- Ablation 3: Absorption attribution intervention. For false-negative tokens (where the harmful-intent latent fails to fire), identify absorbing latents and manually activate the parent latent. Measure the change in probe F1.
- Ablation 4: Control intervention. For the same false-negative tokens, activate a randomly chosen high-frequency latent (not identified by absorption analysis). Should show no improvement, confirming the specificity of the absorption attribution.

**Confounders identified:**
1. *Harmful content definition confound:* "Harmful intent" is not a single feature — it may be represented differently across different categories of harm (violence, fraud, manipulation). Must report results stratified by harm category if possible.
2. *Layer selection confound:* Harmful intent may be most linearly represented at a different layer than layer 12. Must search for the layer with highest linear probe accuracy and use that layer consistently.
3. *Probe vs. absorption directionality confound:* The "harmful intent" feature may not have a single clean direction in the residual stream — it may be distributed across multiple directions. If the linear probe identifies a direction that is not represented by any single SAE latent, this would produce a gap for reasons unrelated to absorption.
4. *Dataset contamination confound:* Harmful content examples may have stylistic features (e.g., specific tokenization patterns from jailbreak prompts) that are tracked by high-frequency SAE latents independently of harmful intent. This could produce apparent absorption where the actual cause is distribution shift.

**Pilot design (< 15 min):**
Load Gemma Scope 16k SAE at layer 12. Use 50 AdvBench examples (25 harmful + 25 harmless). Extract residual stream activations and SAE latent activations. Train dense LR probe (2 min). Train 1-sparse SAE probe (1 min). Compare F1 scores. If gap > 10% (F1), run the full absorption analysis. Total: ~12 min.

---

## Phase 3: Self-Critique

### Against Candidate A

**Confound attack:** The most severe confound is the width-L0 coupling in Gemma Scope. Wider Gemma Scope SAEs were trained to have higher L0 (more features firing per token on average). If absorption increases with both width AND L0, we cannot separate their effects from the standard Gemma Scope suite. However, Gemma Scope releases multiple SAEs at each (width, layer) with different sparsity levels — this allows partial confound control by selecting SAEs with matched L0 across different widths. This requires careful SAE selection during the pilot.

**Statistical attack:** With 26 letters across 32 SAE configurations, the total sample is 832 (letter, SAE) pairs. This is sufficient for a partial correlation analysis. However, absorption rate per letter is itself estimated from a limited number of test examples (the first-letter task uses ~300-500 tokens per letter from the test vocabulary). The standard error of absorption rate estimates is approximately sqrt(AR(1-AR)/n) ≈ sqrt(0.25/400) ≈ 0.025 — about 2.5 percentage points. This means we can detect absorption rate differences of ~5% at p < 0.05. This is sufficient to detect the hypothesized effect (15% vs. 25% across configurations) but leaves little margin for noisy letters with few test tokens (e.g., rare letters like X, Z, Q may have fewer than 100 test tokens, making their estimates unreliable). Solution: restrict analysis to the 15-20 most common letters with >200 test tokens; report token count alongside absorption estimates.

**Benchmark attack:** The first-letter spelling task is a narrow proxy that Chanin et al. explicitly chose for its convenient hierarchy structure. The key question — whether absorption rate measured on this task predicts absorption on other tasks — is precisely what Candidate B tests. Candidate A should not overclaim generalizability.

**Ablation completeness attack:** The proposed ablation design includes 4 ablations, but Ablations 1-3 may be confounded: changing width (Ablation 2) changes both the SAE's capacity AND the typical L0, so it is not a clean isolation of width. To address this, Ablation 2 should use the Gemma Scope suite to identify two SAEs of the same width but different L0 (available in Gemma Scope) to separately measure the L0 effect and the width effect. This requires an additional data selection step but is entirely within the training-free constraint.

**Verdict:** STRONG. The experiment is feasible, the falsification criterion is pre-specified, and the confounders are identifiable and partially controllable. The main risk is that the partial correlation is weak (many confounders reduce the net signal), which would be an honest null result worth reporting.

---

### Against Candidate B

**Confound attack:** The most critical confound is probe quality. If the LR probe for entity-type ANIMAL achieves only 65% accuracy (vs. 92% for letter A), then apparent differences in absorption rate between hierarchy types may reflect probe quality differences, not actual hierarchy-specific absorption. Mitigation: report probe accuracy as a covariate; restrict analysis to hierarchy types where probe accuracy > 80%; include a "matched probe quality" analysis pairing hierarchy types with similar probe accuracies.

**Statistical attack:** We have fewer independent data points in Candidate B than in Candidate A: only 3-4 hierarchy types, not 26 letters. This severely limits statistical power. With 4 hierarchy types and 4 measurement replicates per type (using different probe training seeds), we have 16 data points — barely enough for a paired t-test. The experiment is most convincing if the effect is large (>15 percentage point difference between hierarchical and non-hierarchical conditions) and consistent across seeds. Pre-register the expected effect size and the power calculation before running.

**Benchmark attack:** The entity-type hierarchy (ANIMAL ⊃ DOG) is a "toy" semantic hierarchy that may not be learned in a clean hierarchically separable way by LLMs. LLMs may represent "ANIMAL" as a diffuse property across many layers rather than as a single localized feature at any layer. If this is the case, the LR probe for ANIMAL may be detecting a distributed representation rather than a localized feature, and the absorption analysis (which relies on identifying a single probe direction) may not apply. Mitigation: validate the feature geometry of ANIMAL representations using the Geometry of Concepts framework (arXiv:2410.19750) before running the absorption experiment.

**Ablation completeness attack:** The negative control design (Ablation 4 and 5) is the key to the entire experiment. If the negative control does NOT show near-zero absorption despite matched frequency ratios, the entire absorption metric is uninterpretable as a hierarchy-specific measure. The experiment is structured correctly to test this, but we must ensure the negative control is genuinely non-hierarchical. "Contains letter E" and "is a content word" may in fact be hierarchically related in some weak sense (content words tend to be longer and contain more vowels). A stronger negative control would be two features whose co-occurrence is due to topical clustering (e.g., "mentions weather" and "mentions time of day" — they co-occur in weather reports but neither implies the other).

**Verdict:** STRONG, but the statistical power concern is real. The experiment should be positioned primarily as exploratory/qualitative (showing whether the phenomenon exists beyond the first-letter task) rather than quantitative (comparing absorption rates precisely across hierarchy types).

---

### Against Candidate C

**Confound attack:** The performance gap between SAE probes and linear probes has multiple potential causes beyond absorption: (1) "dark matter" — SAE reconstruction error that is linearly predictable from input (Engels et al., arXiv:2410.14670, ~50% of error); (2) Feature inconsistency — the "harmful intent" latent may not be reliably identified by k-sparse probing; (3) polysemanticity of harmful-intent latents — even without absorption, a latent that fires on both harmful and harmless inputs would produce a poor probe. The intervention test (Ablation 3) is designed to directly test the absorption attribution, but requires the assumption that "manually activating the absorbing latent" faithfully simulates what would happen if the SAE had not absorbed it. This assumption may fail if the absorbing latent and the parent latent encode different directions that downstream probes treat differently.

**Statistical attack:** With 400 examples, the McNemar test is well-powered for detecting F1 gaps > 5% (power > 0.8 at α = 0.05). However, the intervention test (Ablation 3) will only have a subset of the 400 examples where absorption is detected, potentially as few as 50-80 examples. A McNemar test on 60 examples has power ~0.5 for a 5% gap — marginally underpowered. Mitigation: use AdvBench supplemented with synthetic harmful examples to increase the test set to 600-800 examples.

**Benchmark attack:** AdvBench contains adversarial prompts designed to elicit refusals — it may not represent the distribution of harmful content that safety probes would encounter in deployment. The generalizability of findings to other harmful-content benchmarks (ToxiGen, WildChat harmful subset) is unknown. Mitigation: test on at least two distinct harmful-content datasets.

**Ablation completeness attack:** Ablation 3 tests whether "un-absorbing" improves performance. But the control (Ablation 4) only randomly activates one latent. A more informative control would activate the latent that is MOST correlated with harmful content (other than the absorbing latent) — this tests whether any high-activation latent improves the probe, or whether the absorbing latent specifically does.

**Verdict:** MODERATE. The causal claim (absorption drives the gap) is hard to establish rigorously because the intervention test is imperfect. The experiment is most defensible as: "absorption accounts for X% of the false-negative cases where the SAE probe fails but the linear probe succeeds, and adding the absorbed parent latent to the active set reduces this to Y%." This framing makes the empirical contribution concrete without overclaiming causality.

---

## Phase 4: Refinement

### Dropped Ideas

None dropped — all three received STRONG or MODERATE verdicts. Candidate C is retained but repositioned as a *downstream consequence analysis* rather than the primary contribution. The causal claim is explicitly weakened to a characterization of how much of the gap is absorption-attributable.

### Strengthened Survivors

**Candidate A — Strengthened:**
1. **Width-L0 separation**: Add an explicit Ablation 2b: for the same width (16k), select two Gemma Scope SAEs at different target L0 values (e.g., L0≈40 and L0≈80 for the 16k width). This isolates the L0 effect at fixed width. Similarly, select two SAEs from different widths (1k and 16k) with matched L0 (if such SAEs exist in Gemma Scope). This converts the confounded observation into a 2x2 factorial design: {width: 1k, 16k} × {L0: low, high}.
2. **Pre-specified falsification**: Formally pre-register the analysis plan and expected effect sizes before data collection (even as a timestamped note in the paper). This prevents post-hoc cherry-picking of which letters show the effect.
3. **Three-metric reporting**: Report ALL three KronSAE absorption metrics (Chanin rate, mean absorption fraction, full-absorption score) for each SAE to detect cases where architectures differ on one metric but not another.
4. **Multiple seeds check**: For the primary SAE (Gemma Scope 16k, layer 12), verify that absorption rates measured from the same pre-trained weights are deterministic (they should be, since Gemma Scope weights are fixed). Report this explicitly.

**Candidate B — Strengthened:**
1. **Probe quality gate**: Do not proceed with the full absorption measurement for any hierarchy type where LR probe accuracy < 80% at the selected layer. Report probe accuracy as a primary quality metric alongside absorption rate.
2. **Effect size pre-specification**: Pre-specify the minimum expected difference between hierarchical and non-hierarchical conditions (10 percentage points, based on the 15-35% range for first-letter vs. expected ~5% for non-hierarchical). Report Cohen's d alongside p-values.
3. **Replication of Chanin baseline**: The first ablation must exactly replicate Chanin et al.'s reported absorption rates (15-35%) on Gemma Scope 16k at the same layers. If our replication deviates by >5 percentage points, stop and debug before proceeding.
4. **Stronger negative control**: Use corpus-derived co-occurrence statistics to select negative-control feature pairs that have identical ρ to the positive hierarchy pairs but confirmed to lack a logical implication relationship. Check this using WordNet hypernym paths: if both features appear on a common hypernym path, reject from the negative control.

**Candidate C — Strengthened:**
1. **Two datasets**: Test on both AdvBench and ToxiGen (both publicly available) to assess generalizability.
2. **Absorption attribution table**: For the top 20 false-negative cases (where the harmful-intent latent fails but the linear probe succeeds), report which specific latents are absorbing and what their Neuronpedia annotations suggest about their semantics. This provides a mechanistic narrative.
3. **Quantitative attribution framing**: Report as "X% of the SAE false-negative rate is absorption-attributable" rather than "absorption drives the gap."
4. **Ablation 4 improvement**: Replace random latent activation with "most-similar-non-absorbing latent" as the control, to test whether the improvement is specific to the absorbed parent direction.

### Selection of Front-Runner

**Candidate A + Candidate B form the primary contribution** (they are methodologically complementary and together answer: "how severe is absorption, and does it generalize?"). **Candidate C is the downstream motivation section** showing why the quantitative characterization matters.

**Rationale for selecting Candidate A as primary front-runner:**
1. The most rigorous empirical experiment: the 2×2 factorial design (width × L0) with pre-specified falsification criteria addresses the main confound head-on.
2. Entirely training-free: uses only forward passes through existing Gemma Scope SAEs and the sae-spelling measurement code.
3. The falsification criterion is the sharpest: if partial correlation of absorption rate with ρ is < 0.3 after controlling for width, L0, and layer, the co-occurrence hypothesis is refuted.
4. Produces the key quantitative artifact that the field needs: a table of absorption rates across configurations with statistical uncertainty estimates.

---

## Phase 5: Final Proposal

### Title
Isolating the Drivers of Feature Absorption in SAEs: Controlled Experiments on Co-occurrence Frequency, Hierarchy Type, and Downstream Probe Reliability

### Hypothesis (Precisely Falsifiable)
**H1 (Primary):** Co-occurrence frequency ratio ρ = p_child/p_parent is the primary predictor of absorption rate (Spearman ρ > 0.4, p < 0.05), after controlling for SAE width, L0, layer, and decoder cosine similarity, using 26 letters × multiple Gemma Scope configurations as data points.

**H2 (Generalization):** Absorption is present for semantically richer hierarchies (entity-type: ANIMAL ⊃ DOG; syntactic: VERB ⊃ IRREGULAR-PAST-TENSE) but NOT for non-hierarchical feature pairs with matched frequency ratios (absorption rate difference > 10 percentage points, p < 0.05 by paired t-test with 3+ probe training seeds).

**H3 (Downstream consequence):** At least 25% of the cases where the SAE probe fails but the dense linear probe succeeds (false-negative rate of the SAE probe) can be attributed to absorption, as measured by: identifying the absorbing latent for each false-negative token and confirming that manually adding the absorbed parent latent to the active set converts the classification from false-negative to true-positive.

### Falsification Criteria (Pre-Specified Before Data Collection)

| Hypothesis | Falsified if |
|---|---|
| H1 | Partial Spearman ρ (absorption rate ~ ρ, controlled for W, L0, layer, ε) < 0.3, OR the relationship reverses direction for any architecture type |
| H2 | Absorption rate for entity-type hierarchy is not significantly > negative control (p > 0.1 Bonferroni-corrected), OR absorption rate does not vary significantly across hierarchy types (one-way ANOVA p > 0.1) |
| H3 | Fraction of false-negatives explained by absorption < 15% (below the baseline first-letter absorption rate), OR the intervention (adding absorbed parent latent) does not significantly improve F1 (McNemar p > 0.1) |

### Method
Training-free analysis using existing pre-trained Gemma Scope SAEs (Gemma-2-2B). The core measurement pipeline is the sae-spelling codebase (Chanin et al., MIT-licensed) extended with:
1. A 2×2 factorial design varying width (1k vs. 16k) and L0 (low vs. high) at matched conditions from Gemma Scope.
2. Three additional hierarchy types beyond first-letter spelling (entity-type, syntactic, negative-control).
3. Downstream consequence analysis using AdvBench + ToxiGen.

### Evaluation Protocol

**Primary Benchmarks:**
- Gemma Scope SAEs on Gemma-2-2B (standardized, publicly available, training-free)
- First-letter spelling task (Chanin et al. canonical protocol, replication + extension)
- AdvBench (200 harmful + 200 harmless) and ToxiGen (200 toxic + 200 non-toxic) for downstream analysis

**Metrics with Statistical Test Plan:**
1. Absorption rate per (letter, SAE): Chanin et al. metric. SE ≈ sqrt(AR(1-AR)/n_tokens). Report mean ± bootstrap 95% CI (1000 iterations).
2. Mean absorption fraction + full-absorption score (KronSAE metrics): report alongside Chanin rate to detect partial absorption.
3. Feature sensitivity (Tian et al.): computed for the same feature pairs; compare sensitivity vs. absorption correlation.
4. Primary statistical tests: Partial Spearman correlation (H1), Paired t-test (H2), McNemar test (H3).
5. Correction for multiple comparisons: Bonferroni correction across the 26 letters for any per-letter significance claims.
6. Effect size: Cohen's d for all t-tests; r for all correlations.

**Number of random seeds:**
- SAE weights: Fixed (Gemma Scope pre-trained weights are deterministic).
- Probe training: 5 random seeds per hierarchy type (for Candidate B). Report mean absorption rate ± std across seeds.
- Bootstrap: 1000 resamples for all CI estimates.

### Ablation Schedule

| Ablation | What it tests | Expected outcome |
|---|---|---|
| 0. Replication | Reproduce Chanin et al. first-letter absorption rates on Gemma Scope 16k | Must match 15–35%; if not, stop and debug |
| 1. ρ sweep (26 letters, fixed W=16k, fixed L0) | Does absorption rate increase with frequency ratio ρ? | Spearman ρ > 0.4 |
| 2a. Width effect (W=1k vs. 16k at matched L0) | Does width independently increase absorption? | Higher absorption at W=16k |
| 2b. L0 effect (low vs. high L0 at fixed W=16k) | Does lower L0 (more sparsity pressure) independently increase absorption? | Higher absorption at lower L0 |
| 3. Architecture comparison (JumpReLU vs. TopK at matched W and L0) | Do architectures differ in absorption at controlled settings? | TopK worse than JumpReLU per SAEBench |
| 4. Decoder cosine similarity partial effect | Does feature direction correlation predict absorption after controlling for ρ? | Higher cosine similarity → higher absorption |
| 5. Entity-type hierarchy (hierarchical condition) | Does absorption generalize to semantic hierarchies? | Absorption rate > 15% |
| 6. Negative control (frequency-matched, non-hierarchical) | Is absorption metric hierarchy-specific? | Absorption rate < 5% |
| 7. Syntactic hierarchy (additional hierarchy type) | Does absorption appear for syntactic hierarchy? | Absorption rate > 10% |
| 8. Downstream probe gap (AdvBench) | How large is the 1-sparse SAE probe vs. linear probe gap? | F1 gap > 10% |
| 9. Absorption attribution intervention | What fraction of false-negatives is absorption-attributable? | > 25% of false-negatives attributable |
| 10. Random-latent control for intervention | Does the improvement from Ablation 9 require the specific absorbed parent latent? | Random latent addition does NOT improve F1 |

### Control Experiments

**Control 1: Negative-control hierarchy.** A non-hierarchical feature pair (e.g., "is a plural noun" and "refers to an animal" — co-occurring in text but neither implies the other) with the same frequency ratio as the ANIMAL ⊃ DOG hierarchy. If the absorption metric fires for this control, the metric is not hierarchy-specific and all absorption rate estimates are suspect.

**Control 2: Probe accuracy-matched comparison.** For Candidate B, pair each hierarchy type with a dummy probe trained on random labels from the same distribution. If the dummy probe produces similar absorption rate estimates (it should not, given the threshold in the metric), the metric is measuring noise.

**Control 3: Replication on a second model.** Repeat the primary analysis (Candidate A, ablations 0-4) on GPT-2-small with EleutherAI SAEs. If absorption rates are similar (within 10 percentage points), the result generalizes beyond Gemma-2-2B. If they differ substantially, the finding is model-specific.

**Control 4: Measurement reliability.** For the Gemma Scope 16k SAE at layer 12, run the absorption metric 3 times with identical inputs (it should be deterministic; if not, identify source of stochasticity and fix). Report that measurement is deterministic as a reliability check.

**Control 5: Alternative explanation test.** The "dark matter" phenomenon (Engels et al., arXiv:2410.14670) shows ~50% of SAE reconstruction error is linearly predictable from input. For the downstream consequence analysis (Candidate C), measure the reconstruction error for false-negative tokens and test whether it is higher than average — if so, "dark matter" may be an alternative explanation for probe failures.

### Pilot Design (< 15 minutes)

1. **Minutes 1-3:** Load Gemma Scope 16k JumpReLU SAE at layer 12 of Gemma-2-2B via SAELens (single SAE, ~1.5 GB).
2. **Minutes 3-7:** Compute ρ for letters A, E, I, O, T using 10k tokens from OpenWebText (letter frequency and token-starting-with-letter frequency; simple tokenizer count).
3. **Minutes 7-14:** Run sae-spelling absorption metric for each of the 5 letters (~1.5 min per letter).
4. **Minute 14-15:** Plot absorption rate vs. ρ for the 5 letters. Check if Spearman ρ > 0.3.

**Go/no-go criterion:** If Spearman ρ > 0.3 for the 5-letter pilot, proceed to full experiment. If ρ < 0.3, investigate whether the confounds (L0 variation across Gemma Scope widths) need to be addressed first.

### Resource Estimate

| Component | Resources | Time |
|---|---|---|
| Pilot (5 letters, 1 SAE) | 1 GPU (≥16 GB VRAM), CPU for corpus stats | ~15 min |
| Candidate A full sweep (32 SAE configs × 20 letters) | 1 GPU, parallelizable | ~8-12 hrs total (≤1 hr per config) |
| Candidate B probe construction (3 hierarchy types) | CPU for dataset construction | ~2 hrs |
| Candidate B absorption measurement (3 hierarchy types × 5 seeds) | 1 GPU | ~3-4 hrs |
| Candidate C downstream analysis | 1 GPU for activation collection | ~2 hrs |
| GPT-2 replication (Control 3) | 1 GPU | ~2 hrs |
| Total | 1 GPU | ~18-20 GPU-hours |

**Model sizes:** Gemma-2-2B (~5 GB bfloat16) + 16k SAE (~0.15 GB) = ~5.2 GB. Fits on a single 16 GB GPU.

**All experiments are training-free:** No gradient computation; only forward passes and logistic regression (minutes on CPU).

### Risk Assessment

**Risk 1: Width and L0 are fully confounded in Gemma Scope (no matched SAEs available).**
- Probability: 30%. Gemma Scope releases multiple SAEs per (width, layer) with different sparsity targets, but the L0 ranges for 1k and 16k SAEs may not overlap.
- Impact: High. The 2×2 factorial design becomes infeasible; can only report the confounded width effect.
- Mitigation: Pre-check Gemma Scope HuggingFace listings for L0 ranges across widths before committing to the factorial design. If no matched L0 SAEs exist, use partial Spearman correlation as the primary analysis and acknowledge the confound explicitly.
- Time cost: 10 minutes to check HuggingFace listings; do this during the pilot.

**Risk 2: The entity-type hierarchy is not linearly represented at a single residual stream layer.**
- Probability: 40%. LLMs may represent "ANIMAL" as a multi-layer distributed property. Linear probe accuracy may be < 80%.
- Impact: Medium. Candidate B may fail for the entity-type hierarchy while succeeding for the syntactic hierarchy, which has a cleaner linear representation.
- Mitigation: Test LR probe accuracy at layers 8, 12, 16, 20 before running the full absorption analysis. Use the layer with highest probe accuracy. If accuracy < 80% at all layers, replace entity-type with a more concretely linear hierarchy (e.g., "capitalized word" ⊃ "first word in sentence" — simpler and more likely to be linearly separable).

**Risk 3: Absorption metric is not sensitive to the proposed confounders (all absorption rates are similar regardless of ρ, W, L0).**
- Probability: 25%. Chanin et al. showed directional trends but the effect size may be small.
- Impact: Medium. The null result (absorption rate is relatively constant) is itself a publishable finding with significant implications.
- Mitigation: The pre-specified falsification criteria make the null result interpretable. Report null results explicitly with power analysis showing we had sufficient power to detect the hypothesized effect size.

**Risk 4: sae-spelling code is incompatible with the current SAELens version or Gemma Scope format.**
- Probability: 20%. Both repositories are actively maintained and may have breaking changes.
- Impact: Low (2-4 hours to debug). A known engineering risk.
- Mitigation: Pin to the version of SAELens used in the sae-spelling codebase; check the GitHub issues for known compatibility problems before starting.

**Risk 5: The downstream consequence analysis (Candidate C) shows negligible absorption for harmful intent features.**
- Probability: 35%. The DeepMind finding may reflect other SAE failure modes (dark matter, polysemanticity) rather than absorption specifically.
- Impact: Low. Candidate C is a secondary analysis; a null result here strengthens the paper by ruling out absorption as the primary cause.
- Mitigation: Treat Candidate C as exploratory and explicitly report absorption rate for harmful-intent features regardless of whether it matches H3. Frame it as "how much is absorption-attributable?" not as a confirmatory test.

### Novelty Claim

**What specific empirical questions are being answered for the first time:**

1. **Is co-occurrence frequency ratio a causal (or at least strongly confound-controlled) predictor of absorption rate?** Current evidence is entirely observational and confounded. The 2×2 factorial design (width × L0, at controlled conditions) provides the most rigorous test yet of whether width and L0 have *independent* effects on absorption, and whether ρ predicts absorption after controlling for both.

2. **Does the absorption metric detect hierarchy-specific failure, or generic feature unreliability?** No existing paper includes a properly matched negative control (non-hierarchical feature pair with matched frequency ratio). The negative control in Candidate B directly tests the hierarchy-specificity of the metric for the first time.

3. **What fraction of the SAE probe vs. linear probe performance gap is absorption-attributable?** DeepMind documented the gap; no one has quantitatively decomposed it. The intervention test (adding absorbed parent latents) provides the first direct attribution.

**Why this is not already done:** The field has been moving fast on architectural innovations (Matryoshka, OrtSAE, ATM, masked regularization) without stopping to rigorously characterize the phenomenon being addressed. This creates a situation where papers claim to "reduce absorption" without establishing what drives absorption in the first place. The empiricist contribution is to build the evidentiary foundation that justifies and guides the architectural work.

**Expected contribution level:** NeurIPS/ICLR workshop paper with potential for main track if the downstream consequence analysis (Candidate C) produces a strong, clear quantitative finding. The contribution is measurement-focused rather than algorithmic — analogous to the SAEBench paper's contribution to the field.

### Sources

- [Chanin et al. (2024). A is for Absorption. arXiv:2409.14507](https://arxiv.org/abs/2409.14507)
- [Karvonen et al. (2025). SAEBench. arXiv:2503.09532](https://arxiv.org/abs/2503.09532)
- [Gao et al. (2024). Scaling and Evaluating Sparse Autoencoders. arXiv:2406.04093](https://arxiv.org/abs/2406.04093)
- [Tian et al. (2025). Measuring SAE Feature Sensitivity. arXiv:2509.23717](https://arxiv.org/abs/2509.23717)
- [Korznikov et al. (2025). OrtSAE. arXiv:2509.22033](https://arxiv.org/abs/2509.22033)
- [Korznikov et al. (2026). Sanity Checks for SAEs. arXiv:2602.14111](https://arxiv.org/abs/2602.14111)
- [Narayanaswamy et al. (2026). Masked Regularization. arXiv:2604.06495](https://arxiv.org/abs/2604.06495)
- [Chanin, Dulka, & Garriga-Alonso (2025). Feature Hedging. arXiv:2505.11756](https://arxiv.org/abs/2505.11756)
- [Li et al. (2025). SAE Scaling with Feature Manifolds. arXiv:2509.02565](https://arxiv.org/abs/2509.02565)
- [Bussmann et al. (2025). Matryoshka SAE. arXiv:2503.17547](https://arxiv.org/abs/2503.17547)
- [Song et al. (2025). Feature Consistency in SAEs. arXiv:2505.20254](https://arxiv.org/abs/2505.20254)
- [Engels et al. (2024). Dark Matter of SAEs. arXiv:2410.14670](https://arxiv.org/abs/2410.14670)
- [DeepMind Safety Research Team (2025). Negative Results for SAEs on Downstream Tasks](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)
- [Shu et al. (2024). Unified Theory of SDL. arXiv:2512.05534](https://arxiv.org/abs/2512.05534)
- [SAEBench interactive results](https://www.neuronpedia.org/sae-bench)
- [sae-spelling codebase](https://github.com/lasr-spelling/sae-spelling)
- [Gemma Scope HuggingFace](https://huggingface.co/google/gemma-scope)
