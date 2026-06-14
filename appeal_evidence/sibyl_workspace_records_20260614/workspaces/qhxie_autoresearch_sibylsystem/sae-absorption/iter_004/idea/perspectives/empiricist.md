# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **Chanin et al. (2024), arXiv:2409.14507 (NeurIPS 2025)** — Defines the canonical absorption metric via k-sparse probing + integrated-gradients ablation. Key evaluation insight: absorption is measured on the first-letter task only; the metric is explicitly acknowledged as a *conservative lower bound* on the true absorption rate; threshold for absorption detection (cosine > 0.025 with probe, magnitude ≥ 1.0 above second candidate) was set by hand, not by cross-validation. The 15–35% figure is a measurement artifact of this threshold, not a natural quantity.

2. **Karvonen et al. (2025), "SAEBench," arXiv:2503.09532** — Establishes 8-metric evaluation suite including absorption; tests 200+ SAEs across 8 architectures. Critical finding: proxy metrics (CE loss, L0, sparsity) do not reliably predict practical performance. Absorption metric is one of 8 and is acknowledged to capture only one failure mode. Runs on RTX 3090 in ~65 min per SAE full suite. Enables direct comparison baselines without re-running.

3. **Korznikov et al. (2026), "Sanity Checks for SAEs," arXiv:2602.14111** — SAEs recover only 9% of true synthetic features; random baselines match trained SAEs on interpretability, sparse probing, and causal editing tasks. Crucial falsification benchmark: any proposed contribution must clear this "do we beat random?" baseline or explain why the gap doesn't matter for the specific claim.

4. **SynthSAEBench (2026), arXiv:2602.14687** — Controlled synthetic data with known ground-truth feature hierarchy, correlation, Zipfian firing; logistic probes reach 0.974 F1 while best SAEs substantially underperform. Provides the only setting where absorption can be measured against a known ground truth rather than a proxy task. Enables controlled ablation of individual causal variables (hierarchy depth, frequency ratio, co-occurrence strength) independently.

5. **Tian et al. (2025), "Measuring SAE Feature Sensitivity," arXiv:2509.23717** — Frames absorption as a special case of poor feature sensitivity; scalable evaluation method; many interpretable features have poor sensitivity even when activation examples appear monosemantic. Provides an orthogonal, complementary metric to the Chanin et al. absorption rate — evaluating a different aspect of the same failure mode.

6. **Li et al. (2025), "ATM SAE," arXiv:2510.08855** — Achieves mean absorption score 0.0068 vs. TopK 0.1402 and JumpReLU 0.0114 on Gemma-2-2B — a 20× reduction. However, evaluated on *one* model (Gemma-2-2B) and *one* architecture family only. This is a suspicious 95% improvement that demands reproduction and adversarial testing. The claim has not been independently replicated on GPT-2 or Llama 3.2, making it potentially an outlier result or a training confound.

7. **DeepMind Safety Research Team (2025), Medium blog post** — Dense linear probes dramatically outperform 1-sparse SAE probes on harmful intent detection. Critical evaluation insight: the downstream task (safety-relevant detection) is the real test, and absorption is only the proposed *mechanism* for SAE failure. No paper has directly verified that reducing absorption specifically improves performance on this downstream task — the causal chain from absorption rate to safety probe performance is assumed, not tested.

8. **Narayanaswamy et al. (2026), "Masked Regularization," arXiv:2604.06495** — Reduces absorption by disrupting co-occurrence patterns during training. Evaluation insight: no head-to-head comparison on the same model/layer/L0 settings against Matryoshka SAE, OrtSAE, or ATM SAE. Each mitigation paper uses different base configurations, making it impossible to rank them on absorption reduction per unit of additional compute or per unit of reconstruction quality lost.

9. **Song et al. (2025), "Feature Consistency," arXiv:2505.20254** — Reveals SAE features are inconsistent across training runs (53% overlap for TopK SAEs). Critical confound for absorption measurement: if absorption patterns differ across training runs, the absorption rate for a specific feature letter may be a function of the random seed rather than the SAE configuration. No paper has reported absorption rate variance across seeds.

10. **LessWrong, "Looking for Feature Absorption Automatically"** — Negative result: cosine similarity of ablation effect vectors does NOT produce a bimodal distribution distinguishing absorbed from non-absorbed pairs. Cannot distinguish absorbed pairs from non-absorbed pairs by downstream causal similarity. This is an important empirical prior against purely geometry-based detection approaches.

11. **Bussmann et al. (2025), "Matryoshka SAE," arXiv:2503.17547 (ICML 2025)** — Best on SAEBench absorption metric. However, Chanin & Dulka (arXiv:2505.11756) show Matryoshka SAEs *trade* absorption for hedging — improving on the absorption metric while worsening on the hedging failure mode. This is a classic case of Goodhart's Law: optimizing for the absorption metric does not improve overall feature quality.

12. **Engels et al. (2024), "Dark Matter of SAEs," arXiv:2410.14670** — ~50% of the reconstruction error vector is linearly predictable from input but not captured by SAE features. Evaluation insight: absorption measurement uses residuals that may include dark matter signal, potentially inflating false-negative counts and the apparent absorption rate.

### Experimental Landscape

**What has been properly tested:**
- Absorption exists in all tested SAE architectures (Chanin et al., 2024) — replicated across Gemma Scope, Llama 3.2 1B, Qwen2 0.5B
- Absorption rate is 15–35% under the Chanin metric on the first-letter task (one proxy domain only)
- Lower L0 → higher absorption, wider SAEs → higher absorption (empirical trend, no controlled experiment)
- Matryoshka SAEs reduce the absorption metric score on SAEBench (but at cost of hedging)
- ATM SAE achieves lowest reported absorption score on Gemma-2-2B (one model, not replicated)

**What is accepted without proper controls:**
- The claim that "absorption is worse with TopK than L1 SAEs" — this finding comes from SAEBench but the comparison is across different base configurations; L0 is not held constant across architecture comparisons
- The claim that wider SAEs have higher absorption — the confound that wider SAEs are trained with lower L0/D ratio has not been controlled for
- The causal chain that "absorption → safety failure" — assumed from DeepMind's result but not directly tested by measuring absorption rate of specific safety-relevant features and correlating with downstream task failure rate
- ATM SAE's 20× absorption reduction — not independently replicated or tested beyond Gemma-2-2B

**Where methodological gaps are most critical:**
- No controlled experiment varying a single factor (L0, width, layer, architecture) while holding all others fixed to isolate its causal effect on absorption rate
- No study using multiple random seeds per configuration to quantify absorption rate variance
- No experiment verifying that absorption reduction (by any method) translates to improved performance on the downstream safety tasks that motivated absorption research in the first place
- No replication of ATM SAE's extraordinary absorption reduction claim on a second model

---

## Phase 2: Initial Candidates

### Candidate A: The Controlled Absorption Audit — Isolating Individual Causal Factors

**Core hypothesis:** The commonly reported empirical relationships between SAE hyperparameters and absorption rate (L0 → absorption, width → absorption) are confounded. Each relationship has been observed across SAE configurations where multiple factors vary simultaneously. A properly controlled experiment, varying only one factor at a time, will reveal that the reported trends are at least partially spurious — and may identify a factor that explains most of the variance but has been overlooked.

**Specific falsifiable predictions (decided before seeing the results):**
1. After controlling for L0/D ratio (relative sparsity), the effect of absolute SAE width on absorption rate will be reduced to non-significance (p > 0.05).
2. After controlling for training steps, the effect of SAE architecture (TopK vs. L1) on absorption rate will be reduced by at least 50% compared to uncontrolled comparison.
3. Absorption rate variance across 3 random seeds for the same SAE configuration will exceed 5 percentage points for at least one layer, indicating that reported rates are partly seed-dependent rather than configuration-determined.

**Falsification criterion:** If all three predictions are wrong — i.e., width effect persists after L0/D control, architecture effect persists after training-step control, and seed variance is < 2pp — then the conventional view (width and architecture are genuine drivers) is upheld. This is also the result I would expect if the conventional view is correct.

**Evaluation protocol:**
- Primary benchmark: Chanin et al. first-letter task absorption rate (the canonical metric, directly comparable to existing literature)
- Models: Gemma 2 2B with Gemma Scope SAEs (multiple widths, multiple L0 settings, multiple layers available out-of-the-box from HuggingFace)
- Statistical test: ANOVA with controlled factor structure; bootstrap confidence intervals on each coefficient; paired t-test for seed variance
- Number of seeds: 3 minimum per configuration (Gemma Scope provides multiple training seeds at some configurations)
- Pre-registration: All three specific predictions stated above decided before running experiments

**Ablation plan:**
- Ablation 1 (isolate L0 effect): Hold width=16k, layer=12 fixed; vary L0 over {10, 20, 50, 100, 200}. This is the cleanest test of the L0-absorption relationship — first time it has been done with all other factors held constant.
- Ablation 2 (isolate width effect): Hold L0=50, layer=12 fixed; vary width over {1k, 4k, 16k, 65k, 131k}. CRITICAL: check whether Gemma Scope SAEs at different widths actually achieve the same L0 on the same test distribution — if not, the width effect is confounded with L0.
- Ablation 3 (isolate architecture): Compare L1, TopK, JumpReLU SAEs at matched L0 (using L0 as the matching variable, not L1 coefficient) on the same base model. SAEBench provides pre-computed absorption scores — use these to check if architecture effect disappears when L0 is matched.
- Ablation 4 (seed variance): Use 3 seeds at the single configuration (16k, L0=50, layer=12) where Gemma Scope provides multiple checkpoints. Measure absorption rate per letter per seed; report within-configuration variance.

**Confounders identified:**
- L0/D ratio: wider SAEs may be trained to lower absolute L0, creating a spurious width effect
- Training duration: longer-trained SAEs show higher absorption (SAEBench finding for JumpReLU); not controlled in existing comparisons
- Layer: early vs. late layer SAEs have different feature complexity; not controlled in cross-layer comparisons
- Dark matter: absorption measurement uses reconstruction residuals that contain dark matter signal, potentially inflating false-negative counts

**Pilot design (< 15 min):** Load Gemma Scope 16k and 131k SAEs for Gemma 2 2B at layer 12. Check whether they achieve the same empirical L0 on a 500-token sample. If the 131k SAE achieves L0=50 and the 16k SAE achieves L0=30, the "width effect" is actually an L0 effect. This single measurement decides whether the controlled-width ablation is even possible with existing data.

---

### Candidate B: Does Reducing Absorption Actually Help? The Downstream Causal Chain Test

**Core hypothesis:** The dominant research framing assumes: absorption → reduced feature recall → degraded downstream task performance. This causal chain has never been directly tested. Multiple papers report absorption reduction as a goal in itself, but none measure whether absorption reduction actually improves performance on the downstream safety tasks (harmful intent detection, concept erasure, circuit analysis) that motivate absorption research.

**Specific falsifiable predictions (decided before seeing the results):**
1. Absorption rate (Chanin metric) will be negatively correlated (r < -0.3) with downstream safety probe F1 across a set of matched SAEs varying in architecture and absorption rate.
2. Matryoshka SAEs, which have the lowest absorption scores, will outperform standard SAEs on downstream safety-relevant linear probe tasks by at least 10%.
3. If prediction 2 fails: ATM SAE's 20× absorption reduction will also not improve safety probe performance — suggesting absorption reduction as measured by the Chanin metric is decoupled from the downstream task.

**Falsification criterion:** If absorption rate is uncorrelated with downstream safety probe performance (|r| < 0.1), the causal chain assumption is falsified and the entire research framing needs reconsideration. This is the result the DeepMind blog strongly hints at but does not directly test.

**Evaluation protocol:**
- Primary benchmarks: SAEBench (absorption score, SCR task, RAVEL task) — these are already computed for 200+ SAEs; no new experiments required for the correlation analysis
- Safety task: linear probe accuracy on a harmful intent detection dataset (e.g., AdvBench prompts + benign prompts) using SAE features vs. raw residual stream; measure 1-sparse SAE probe accuracy vs. dense linear probe accuracy for the same feature set
- Metrics: Pearson and Spearman correlation between absorption rate and downstream task performance; p-values with Bonferroni correction for 3 SAE tasks
- Number of seeds: use the 200+ SAEs in SAEBench as the sample (each is a different configuration/seed); correlation is measured across this population

**Ablation plan:**
- Ablation 1 (absorption-performance correlation): Using existing SAEBench data (no new experiments), regress absorption_score ~ downstream_task_performance for each of the 4 interpretability tasks in SAEBench. This is a pure analysis, not a new experiment.
- Ablation 2 (matched comparison): Select 5 SAEs with lowest absorption scores (Matryoshka, ATM if available) and 5 SAEs with highest absorption scores (JumpReLU, standard TopK) matched on model, layer, and as closely as possible on L0. Compare their downstream performance on RAVEL (a causal intervention task).
- Ablation 3 (safety gap): Measure the performance gap between 1-sparse SAE probes and dense linear probes for harmful intent detection on 3 matched SAEs (low/medium/high absorption). Does the performance gap decrease as absorption decreases?

**Confounders identified:**
- SAEs optimized for low absorption may sacrifice reconstruction fidelity, which itself degrades downstream performance independent of absorption
- The SAEBench correlation may reflect other latent factors (model layer, SAE width) rather than absorption causally
- The 200+ SAEs in SAEBench are not at identical base configurations — partial confounding is unavoidable; control for model and layer in the regression

**Pilot design (< 15 min):** Download the SAEBench results CSV from neuronpedia.org/sae-bench. Compute the Pearson correlation between the "absorption" column and each of the downstream task performance columns. If |r| < 0.1 for all tasks, the causal chain hypothesis is falsified immediately and this is the most important result in the paper.

---

### Candidate C: ATM SAE Replication and Generalization — Testing an Extraordinary Claim

**Core hypothesis:** The ATM SAE paper reports a 20× absorption reduction (0.0068 vs. TopK 0.1402) on Gemma-2-2B — an extraordinary claim that has not been independently replicated. This improvement is suspicious under the evidence contract: >30% improvement on a simple baseline requires validation before accepting as a real finding. The likely explanations for such a large reduction are: (a) genuine mechanism (adaptive temporal masking actually disrupts absorption-inducing co-occurrence patterns), (b) evaluation-metric gaming (ATM SAE is optimized in a way that specifically reduces the Chanin metric without genuinely improving feature recall), or (c) distribution mismatch (ATM SAE is evaluated on a test distribution more favorable to its training procedure).

**Specific falsifiable predictions (decided before seeing the results):**
1. ATM SAE's 20× absorption reduction will replicate on GPT-2 small SAEs (a different model family) with a reduction of at least 5× (effect size > 0).
2. ATM SAE will outperform standard TopK SAE on at least 3 of the 4 SAEBench downstream tasks (RAVEL, SCR, sparse probing, unlearning) at the same absorption-reduced setting — i.e., the absorption reduction translates to downstream improvement.
3. If prediction 2 fails: the absorption metric used to evaluate ATM SAE is partially decoupled from the downstream tasks, and the "improvement" is metric-specific rather than genuine.

**Falsification criterion:** If ATM SAE's absorption reduction is < 2× on GPT-2 small, the original result is likely a model-specific artifact. If the absorption reduction does not improve downstream tasks, the metric is not capturing what matters.

**Evaluation protocol:**
- Primary benchmark: First-letter absorption rate (Chanin metric) on GPT-2 small SAE + ATM-equivalent training procedure (or using existing ATM SAE checkpoints if released by the authors)
- Secondary benchmark: SAEBench 4-task suite on whatever ATM SAE checkpoints are available
- Metrics: Absorption rate, false-negative rate, downstream task scores
- Statistical test: Bootstrap CI on the absorption rate difference; one-sided t-test for replication claim
- Key question: Does ATM SAE's absorption reduction persist when measured on a different model at comparable L0?

**Ablation plan:**
- Ablation 1 (metric-only vs. full replication): Run Chanin et al. absorption metric on ATM SAE activations on both the original test set and a novel test set with different token co-occurrence statistics. If ATM SAE improves on the original but not the novel test set, the improvement is test-set-specific.
- Ablation 2 (component isolation): The ATM mechanism has two components — temporal masking AND importance scoring. Ablate each component independently to determine which contributes to absorption reduction.
- Ablation 3 (downstream transfer): Measure whether ATM SAE's absorbed features on the first-letter task correspond to features that fail on safety-relevant downstream tasks — directly testing whether the absorption metric and downstream failure are measuring the same thing.

**Confounders identified:**
- The ATM SAE paper uses Gemma-2-2B; if ATM is tuned to Gemma's activation statistics, the result may not generalize
- The absorption score metric uses a specific threshold (cosine > 0.025, magnitude ≥ 1.0 above second candidate); ATM SAE training may have reduced the magnitude of absorbing features below this threshold without actually preventing absorption
- Training duration: ATM SAE may be trained for more steps than the TopK baseline

**Pilot design (< 15 min):** Download ATM SAE activation statistics for Gemma-2-2B (if available) or use the reported activation frequency statistics from the paper. Check whether the ATM SAE has fewer features with high cosine similarity to the first-letter probe directions — this would indicate genuine absorption reduction vs. metric gaming.

---

## Phase 3: Self-Critique

### Against Candidate A

**Confound attack:** The main vulnerability is that Gemma Scope may not provide SAEs at exactly matched L0 across different widths. If the 131k SAE can only be run at L0=100 while the 16k SAE runs at L0=50, the controlled comparison breaks down. Check: Gemma Scope SAEs are trained at fixed dictionary sizes with varying L1 penalties, producing different empirical L0 values; the same empirical L0 may not be achievable across all widths.

*Mitigation:* Instead of requiring exact L0 matching, choose a range of L0 values and plot absorption rate as a function of L0 separately for each width. If the curves are parallel (same slope, different intercept), width has a genuine effect after L0 control. If they overlap, width has no effect beyond L0.

**Statistical attack:** ANOVA on 30 SAE configurations (5 widths × 3 L0 × 2 layers) with 26 letters per SAE gives n=780 observations, but they are not independent (letters from the same SAE share reconstruction residuals and SAE weights). The effective sample size is closer to 30 (number of SAEs). With 5 predictors in a regression on n=30 observations, the test is severely underpowered for detecting small effects. Prediction 3 (seed variance) requires at least 3 seeds × 1 configuration, which is feasible only if Gemma Scope provides multiple seeds at the same configuration.

*Mitigation:* Focus on detecting large effects (Δabsorption > 10pp). Use mixed-effects models with SAE as a random effect to account for letter-level clustering. For seed variance, verify that Gemma Scope provides 3+ checkpoints at matched configurations before committing to this ablation.

**Benchmark attack:** Using only the first-letter task restricts the finding to syntactic features. If the controlled-factor relationships found here don't generalize to semantic features, the practical value is limited.

*Mitigation:* This is an acceptable limitation for a paper focused on causal identification. Frame it explicitly: "we provide the first controlled identification of the L0 and width effects on absorption rate, validated on the first-letter benchmark; generalization to other feature hierarchies is left for future work."

**Ablation completeness attack:** Ablation 3 (architecture comparison) relies on existing SAEBench data. The SAEBench data does not perfectly control for L0 across architectures (different architectures were evaluated at different settings). This makes the architecture ablation partially confounded.

*Mitigation:* Clearly flag SAEBench-based results as "exploratory/observational" rather than "controlled" in the paper. Run at least one controlled architecture comparison (TopK vs. L1 at matched L0=50, same base model) as the primary architecture evidence.

**Verdict: STRONG** — The methodological gaps it targets (no controlled experiments, no seed variance reported) are real and important. The pre-registered predictions are specific and falsifiable. The main risk is data availability for the controlled comparison, addressable by the pilot.

---

### Against Candidate B

**Confound attack:** SAEBench correlates absorption with downstream performance, but absorption rate and downstream performance are both correlated with SAE architecture quality (better architectures have lower absorption AND better downstream performance). The correlation may be mediated by architecture quality rather than absorption directly.

*Mitigation:* Partial correlation controlling for SAE architecture class (TopK/L1/Matryoshka/etc.) and SAE width. If the correlation disappears when architecture is controlled, absorption is not the causal factor.

**Statistical attack:** With 200+ SAEs in SAEBench, even a weak correlation (r=0.1) will be statistically significant. The relevant threshold for "meaningful" is r > 0.3 (explaining > 10% of variance). We need to pre-specify the minimum effect size that would be considered practically meaningful, not just statistically significant.

*Mitigation:* Pre-specify r > 0.3 as the threshold for a "meaningful" correlation (already done in the falsifiable prediction above).

**Benchmark attack:** SAEBench's "safety" tasks (SCR and RAVEL) measure spurious correlation removal and ravel disentanglement — these are related to but not identical to harmful intent detection. The DeepMind finding specifically concerns harmful intent, which may not be captured by SAEBench's task set.

*Mitigation:* Supplement the SAEBench correlation with one direct harmful intent probe experiment (select 3 SAEs covering the absorption range and measure 1-sparse vs. dense probe accuracy on AdvBench-style prompts). This is a pilot experiment, not a comprehensive study.

**Ablation completeness attack:** Ablation 2 (matched comparison of low vs. high absorption SAEs) selects the 5 lowest and 5 highest absorption SAEs. These may differ on many dimensions beyond absorption rate. Even matching on L0 and width may not control for all confounders.

*Mitigation:* Be explicit about residual confounding and frame this as correlational evidence rather than causal proof. The causal claim requires a more controlled design (train SAEs with and without an absorption-reducing intervention while holding all other factors constant), which is beyond the scope of the training-free analysis constraint.

**Verdict: STRONG** — The core analysis (SAEBench correlation between absorption and downstream tasks) is essentially zero-effort (reanalyze existing data) and provides the most critical empirical test of the field's central assumption. The main limitation is that the causal inference is weak (correlation only), but this is exactly what the paper should claim: "the assumed causal chain has not been empirically tested; here is the first evidence on whether it holds."

---

### Against Candidate C

**Confound attack:** ATM SAE's training procedure differs from TopK/L1 in multiple ways beyond temporal masking (different optimizer settings, different warmup schedule, different effective L0). The reported 20× improvement may reflect these confounds rather than the masking mechanism.

*Mitigation:* The pilot experiment (checking ATM SAE activation frequency statistics) can quickly determine whether the improvement is genuine or metric-specific. If ATM SAE's activation frequencies on the test distribution look substantially different from TopK (fewer high-frequency latents, different co-occurrence structure), the improvement is plausibly genuine but from the distribution shift rather than the masking mechanism per se.

**Statistical attack:** A single model replication (GPT-2 small) is underpowered for confirming a 20× result. If the true reduction is only 5×, GPT-2 replication might show 3-7× (high variance), which might not reach statistical significance.

*Mitigation:* Focus on the sign of the effect (does ATM outperform TopK at all?) rather than the magnitude. Any improvement that is statistically significant at p < 0.05 on GPT-2 supports the claim of generalization.

**Benchmark attack:** The ATM SAE paper reports absorption_score as a single number, not broken down by letter or layer. If ATM reduces absorption for common letters (A, E, T) but not rare letters, the aggregate improvement may mask no improvement for the features most relevant to rare semantic hierarchies.

*Mitigation:* Always report per-letter absorption rates, not just aggregate scores. This is essential for any honest evaluation.

**Ablation completeness attack:** Ablating each ATM component (temporal masking vs. importance scoring) requires access to ATM SAE training code, which may not be publicly available. If the code is not available, this ablation is not feasible.

*Mitigation:* If ATM training code is not available, focus on the evaluation-only experiments (absorption metric on different test distributions). The component ablation becomes a suggested future experiment.

**Verdict: STRONG** — Testing an extraordinary claim through independent replication is the most fundamental scientific contribution. The 20× reduction is the most suspicious empirical claim in the absorption literature and testing it is high-value regardless of outcome.

---

## Phase 4: Refinement

**Dropped:** No candidate is dropped in full. All three address distinct, critical empirical gaps.

**Priority ranking:** B > A > C

**Rationale:**
- Candidate B is the highest-impact with lowest implementation cost (reanalyze existing SAEBench data). It directly tests the field's central motivating assumption and can be completed in hours without new experiments.
- Candidate A provides the controlled experimental evidence that has never existed for the basic causal claims. It requires ~15 GPU-hours of data collection but produces clean, unconfounded results.
- Candidate C provides replication of the strongest claim in the mitigation literature. It is important but secondary to establishing whether the downstream impact exists at all (Candidate B).

**Strenghtened design decisions:**

For **Candidate B**:
- Expand the downstream task set: use SAEBench RAVEL + SCR + Sparse Probing (all available for 200+ SAEs) rather than just absorption. Measure the full correlation matrix between absorption and all downstream metrics.
- Add the harmful intent detection pilot: 3 SAEs (low/medium/high absorption) × 1-sparse vs. dense probe on AdvBench-style prompts. This directly connects to the DeepMind finding.
- Pre-register minimum effect sizes: r > 0.3 for meaningful correlation, 10pp improvement for matched comparison.

For **Candidate A**:
- Add a null result reporting commitment: if the pilot shows L0 is not matched across widths in Gemma Scope, report this as the finding and redirect to a within-architecture L0-sweep instead of a cross-width comparison.
- Strengthen the seed variance component: check whether Gemma Scope provides 3+ checkpoints at the same configuration. If not, use the 3 seeds reported in the ATM SAE paper for Gemma-2-2B (Table 1 error bars suggest they tested multiple runs).

For **Candidate C**:
- Reduce scope: focus on the evaluation-only replication (does the Chanin metric on different test distributions show the same 20× improvement?) before attempting full training-based replication.
- Add a critical diagnostic: measure whether ATM SAE produces fewer high-cosine-similarity decoder direction pairs (the OrtSAE diagnostic) — if yes, ATM is genuinely learning more orthogonal features; if no, the absorption metric improvement may be an artifact.

**Selected front-runner:** Candidate B + Candidate A, with B as the anchor finding and A as the controlled causal evidence that explains what B finds.

The integrated story: **"Does Reducing Absorption Help? Controlled Experiments and the Downstream Impact of SAE Feature Absorption."**

This paper combines: (1) the first controlled causal identification of absorption rate determinants (Candidate A), (2) the first direct test of whether absorption reduction translates to downstream improvement (Candidate B), and as a byproduct (3) a replication/stress test of the ATM SAE extraordinary claim (Candidate C, reduced scope).

---

## Phase 5: Final Proposal

### Title
**Does Absorption Reduction Improve Interpretability? Controlled Experiments on Causal Factors and Downstream Impact of SAE Feature Absorption**

### Hypothesis
Feature absorption in SAEs as measured by the Chanin et al. metric is (H1) significantly confounded by the L0/D ratio rather than being causally driven by absolute SAE width, and (H2) weakly or not correlated with downstream interpretability task performance — meaning absorption reduction as currently measured may not translate to the interpretability improvements that motivate the research.

**Secondary hypothesis (H3):** ATM SAE's reported 20× absorption reduction is a metric-specific artifact that does not generalize across model families or test distributions.

### Falsification Criteria (pre-specified before running experiments)
- **H1 falsified if:** After controlling for L0/D ratio, the width effect on absorption rate remains significant (p < 0.05, effect size > 10pp)
- **H2 falsified if:** Pearson r between absorption rate and downstream SAEBench task performance is r < -0.3 (absorption predicts downstream performance meaningfully)
- **H3 falsified if:** ATM SAE's absorption reduction replicates on GPT-2 small with ≥ 5× improvement AND downstream tasks also show ≥ 10pp improvement

### Method

**Component 1: Controlled Absorption Audit (Candidate A)**

Setting: Gemma 2 2B with Gemma Scope pre-trained SAEs; first-letter spelling task via sae-spelling codebase.

Experimental design (full factorial where data permits):
- Vary L0 at fixed width and layer: widths {16k}, layers {12}, L0 settings available in Gemma Scope for this configuration
- Vary width at fixed L0: identify L0 setting achievable at multiple widths (16k, 65k, 131k) and run absorption measurement at the matched L0
- 3 random seeds where Gemma Scope provides multiple checkpoints at the same configuration
- All 26 letters per SAE; report mean + 95% CI per letter

Measurement: sae-spelling k-sparse probing + integrated gradients absorption metric (Chanin et al.). Record: (layer, width, empirical_L0, letter, absorption_rate, false_negative_rate, seed).

Analysis: ANOVA with empirical_L0 as covariate. Primary coefficient of interest: partial effect of width on absorption rate after controlling for L0. Bootstrap CI on each coefficient.

**Component 2: Downstream Impact Analysis (Candidate B)**

Stage 1 — SAEBench correlation analysis (zero GPU-hours, existing data):
- Download SAEBench results for all 200+ SAEs from neuronpedia.org/sae-bench
- Compute Pearson and Spearman correlations between absorption_score column and each downstream metric column (RAVEL, SCR, Sparse Probing, Unlearning)
- Partial correlation controlling for model, layer, and architecture class
- Report: correlation matrix, partial correlations, p-values with Bonferroni correction

Stage 2 — Matched low/high absorption comparison (2 GPU-hours):
- Select top-5 lowest absorption SAEs (candidates: Matryoshka) and top-5 highest absorption SAEs (candidates: JumpReLU, standard TopK) from SAEBench, matched on model=Gemma 2 2B, layer∈{12,20}
- For each matched pair: measure RAVEL task performance directly (RAVEL code is publicly available in SAEBench; task requires ~30 min per SAE pair)
- Statistical test: paired t-test on RAVEL score difference; one-sided test for H2

Stage 3 — Safety probe pilot (1 GPU-hour):
- Select 3 SAEs (lowest, median, highest absorption from the Gemma 2 2B SAEBench results)
- Construct a binary classification task: harmful intent vs. benign prompts (50 harmful from AdvBench, 50 benign from OpenWebText)
- Train 1-sparse SAE probe (select one latent; ridge regression over SAE activations with L0=1 constraint) vs. dense linear probe on residual stream activations
- Report: ROC-AUC for 1-sparse SAE probe vs. dense probe, by absorption level of SAE

**Component 3: ATM SAE Replication (Candidate C, reduced scope)**

Stage 1 — Test distribution stress test (30 min):
- Run the Chanin et al. absorption metric on the ATM SAE (if checkpoints available) on: (a) standard test set from OpenWebText (same as original paper), (b) adversarially constructed test set where parent-token co-occurrence statistics differ from training distribution
- If absorption score improves on (a) but not (b): metric-specific artifact
- If absorption score improves on both: genuine mechanism

Stage 2 — Diagnostic check (30 min):
- Compute pairwise decoder cosine similarity distribution for ATM SAE vs. TopK SAE at matched L0
- If ATM SAE has fewer high-cosine pairs (< 0.25): genuine orthogonalization is occurring
- If distributions are similar: the activation mechanism rather than the feature geometry is responsible

### Evaluation Protocol

**Primary benchmarks:**
- Chanin et al. first-letter task absorption rate (canonical, directly comparable to existing literature)
- SAEBench RAVEL task performance (standardized causal intervention benchmark, publicly available)
- SAEBench SCR (spurious correlation removal, relevance to safety-adjacent tasks)

**Secondary metrics:**
- False-negative rate (complement of true positive rate for individual letter features)
- 1-sparse SAE probe AUC on harmful intent classification (pilot)
- Seed-to-seed absorption rate variance (novel measurement)

**Statistical plan:**
- ANOVA for controlled factor experiment (Component 1): Bonferroni-corrected for 4 predictors
- Pearson correlation for SAEBench analysis (Component 2, Stage 1): r threshold = 0.3; Bonferroni for 4 downstream tasks
- Paired t-test for matched comparison (Component 2, Stage 2): one-sided, α = 0.05
- All bootstrap CIs at 95%; 1000 resamples
- Sample sizes: Component 1: ~30 SAE configurations × 26 letters = 780 observations (n_SAE=30); Component 2 Stage 1: 200+ SAEs; Component 2 Stage 2: 10 SAEs (5 pairs)

**Minimum 3 random seeds:** Applied to Component 1 where Gemma Scope checkpoints permit. If only 1 seed is available per configuration, report this limitation explicitly.

### Ablation Schedule

| Ablation | Tests | Expected Outcome if H1 True | Expected Outcome if H1 False |
|---|---|---|---|
| A1: L0 sweep at fixed width | Does L0 alone explain absorption? | R² > 0.8 with L0 only | Low R² with L0 only |
| A2: Width at matched L0 | Does width effect persist after L0 control? | Width coefficient p > 0.05 | Width coefficient p < 0.05 |
| A3: Architecture at matched L0 | Does architecture effect persist after L0 control? | Architecture reduced or non-significant | Architecture remains significant |
| A4: Seed variance | Is absorption rate stable across seeds? | >5pp variance at some configuration | <2pp variance everywhere |

| Ablation | Tests | Expected Outcome if H2 True | Expected Outcome if H2 False |
|---|---|---|---|
| B1: SAEBench correlation | Is absorption correlated with downstream tasks? | |r| < 0.1 for all tasks | r < -0.3 for some tasks |
| B2: Matched SAE comparison | Do low-absorption SAEs outperform downstream? | No significant difference (p > 0.1) | p < 0.05, >10pp improvement |
| B3: Safety probe gap | Does absorption predict probe gap? | Gap uncorrelated with absorption | Gap correlated (r < -0.3) |

### Control Experiments

1. **Reconstruct Chanin et al. baseline:** Re-run the original absorption measurement on one Gemma Scope 16k SAE at layer 12 and confirm the absorption rate matches the 15–35% reported in the paper. This validates the pipeline before running the controlled experiments. If the pipeline cannot reproduce the original result, stop and debug before proceeding.

2. **Dark matter confound control:** For each SAE in Component 1, measure the "dark matter" fraction (fraction of residual error linearly predictable from input, as in Engels et al. 2024) and include it as a covariate in the ANOVA. If dark matter fraction varies systematically with L0 or width, it may confound the absorption measurement.

3. **Random baseline:** Include a random SAE (with randomly initialized weights) in the SAEBench correlation analysis (Component 2, Stage 1). The "Sanity Checks" paper (arXiv:2602.14111) shows random SAEs match trained SAEs on some metrics — verify this is not true for the absorption metric (absorption rate for random SAE should be near 100%, not 15–35%).

4. **Probe stability control:** For each letter in the absorption measurement, check whether the k-sparse probe training is stable (coefficient of variation < 20% across 5 probe training seeds). Unstable probes would make the absorption measurement noisy and the controlled experiment conclusions unreliable.

### Pilot Design (< 15 min)

**Critical pilot:** Load Gemma Scope 16k and 131k SAEs for Gemma 2 2B at layer 12. Pass a 500-token sample from OpenWebText through both SAEs. Record the empirical L0 (mean number of active latents per token) for each SAE. Check whether both SAEs achieve the same empirical L0 on the same test distribution.

**Decision rule:** If empirical L0 differs by > 20% between 16k and 131k SAEs, the controlled width comparison is not possible with existing data (the width effect is confounded with L0). In this case, shift to a within-architecture L0 sweep only (ablation A1). If empirical L0 is within 20%, proceed with the full controlled design.

This pilot takes < 15 min (SAE loading: 5 min; forward pass on 500 tokens: 5 min; L0 measurement: 2 min) and completely determines the feasibility of the main experiment.

### Resource Estimate

| Task | GPU | Wall-clock | Notes |
|---|---|---|---|
| Pipeline validation (reproduce Chanin) | 1× A100 | 20 min | Single SAE, 26 letters |
| Pilot: L0 matching check | 1× A100 | 15 min | Two SAEs, 500 tokens |
| Component 1: Controlled audit | 1× A100 | 8 hrs | 30 SAEs × 26 letters |
| Component 2, Stage 1: SAEBench correlation | None | 1 hr | Data download + analysis |
| Component 2, Stage 2: Matched RAVEL | 1× A100 | 2 hrs | 10 SAEs × 1 task |
| Component 2, Stage 3: Safety pilot | 1× A100 | 1 hr | 3 SAEs × 1 task |
| Component 3: ATM stress test | 1× A100 | 1 hr | 1-2 SAEs |
| **Total** | 1× A100 | **~13 hrs** | All training-free |

Model: Gemma 2 2B (5 GB VRAM); SAEs: 16k–131k latents (0.1–1.3 GB VRAM); all fits on single A100 40GB.

### Risk Assessment

**Risk 1: Pilot shows L0 is not matched across widths in Gemma Scope (high probability, ~50%).**
- Consequence: Cannot run the controlled width ablation (Ablation A2).
- Mitigation: The L0 sweep at fixed width (Ablation A1) is still fully valid and addresses H1. Report the L0 matching failure as a methodological finding in itself: "Gemma Scope SAEs at different widths achieve different empirical L0 values, making cross-width absorption comparisons in prior literature confounded." This is itself a meaningful negative result.

**Risk 2: SAEBench correlation (Stage 1) shows r < -0.3 for some downstream task — H2 is falsified (moderate probability, ~30%).**
- Consequence: This is actually the best outcome for the field: absorption does matter for downstream performance. The paper reframes as: "we establish that absorption reduction improves downstream performance, and provide the first controlled characterization of what factors drive absorption rate."
- Mitigation: If H2 is falsified, strengthen Candidate A as the primary contribution and add the implication: "given that absorption does predict downstream performance, the controlled identification of its causal drivers (Candidate A) is essential for designing interventions."

**Risk 3: ATM SAE checkpoints are not publicly available (moderate probability, ~40%).**
- Consequence: Candidate C is limited to the diagnostic check on decoder geometry.
- Mitigation: Use the OrtSAE absorption diagnostic (cosine similarity distribution) as a proxy to estimate whether ATM-like orthogonalization is occurring. If not, report as a recommended future experiment.

**Risk 4: The SAEBench data for 200+ SAEs does not have good covariate information (model, layer, L0) for partial correlation analysis.**
- Consequence: Component 2 Stage 1 produces uncontrolled correlations that cannot be causally interpreted.
- Mitigation: Filter to a homogeneous subset (Gemma 2 2B, layer 12 only) where covariate variation is minimized. Accept smaller sample size (~20 SAEs) for cleaner interpretation.

**Risk 5: Probe training instability makes absorption rate noisy.**
- Consequence: The controlled experiments have inflated noise, reducing statistical power.
- Mitigation: This is caught by the probe stability control (Control Experiment 4). If CV > 20%, aggregate across multiple probe seeds before reporting absorption rates.

### Novelty Claim

**What is being established for the first time:**

1. **The first controlled experiment identifying causal drivers of absorption rate.** Prior work shows absorption correlates with L0 and width, but no experiment controls for confounders. We provide the first pre-registered, controlled causal identification of which factors genuinely drive absorption rate, with explicit null results where factors are not causal.

2. **The first direct test of the causal chain: absorption rate → downstream interpretability task performance.** The motivation for all absorption reduction research is that absorption degrades downstream tasks. This has been assumed but never directly measured. We provide the first empirical test of this assumption using existing SAEBench data plus targeted new experiments.

3. **The first quantification of absorption rate variance across random seeds.** If absorption rate varies by > 5pp across seeds for the same SAE configuration, then reported absorption rates (all single-seed in existing literature) may be unreliable measurements. This would challenge the reliability of comparative absorption results across papers.

4. **A replication test of the strongest absorption reduction claim.** ATM SAE's 20× improvement is the most extraordinary result in the absorption literature and has not been independently replicated. We provide the first independent evaluation of this claim.

**Specific empirical question being answered for the first time:** Does reducing the SAE absorption metric score actually improve downstream interpretability and safety-task performance, or is it an example of Goodhart's Law — where optimizing the metric stops it from being a good measure of what it was supposed to capture?

**Target venue:** NeurIPS 2026 or ICLR 2027, mechanistic interpretability track. The combination of controlled experiments and the downstream impact analysis matches the methodological rigor expected at these venues for interpretability evaluation work.
