# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **[Chanin et al., 2024. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507, NeurIPS 2025]** -- The canonical paper defining feature absorption. Proves absorption in a toy model with hierarchical features. Introduces the probe-directed ablation metric. Finds 15-35% absorption rate across all tested SAEs. All evaluation restricted to the first-letter spelling task. Critical foundation but narrowly validated.

2. **[Karvonen et al., 2025. "SAEBench." arXiv:2503.09532]** -- 200+ SAEs, 8 metrics including absorption. Key empirical finding: proxy metrics (CE loss, L0) do not predict practical performance. Matryoshka SAEs dominate on absorption but this is confounded with width and L0. Essential evaluation infrastructure.

3. **[Tang et al., 2025. "A Unified Theory of Sparse Dictionary Learning." arXiv:2512.05534]** -- Casts all SDL variants as piecewise biconvex optimization. Proves absorption arises as spurious local minima. Proposes feature anchoring but only validated on synthetic benchmarks. The theoretical anchor for understanding when absorption is unavoidable.

4. **[Geiger et al., 2025. "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability." JMLR 26(1-63)]** -- Formal framework for evaluating when a high-level causal model is a faithful abstraction of a low-level model. Unifies activation patching, circuit analysis, DAS, and SAEs under one mathematical vocabulary. Critical insight: absorption is a failure of causal faithfulness -- the SAE's feature-level description fails to be a valid causal abstraction of the model's computation for absorbed features.

5. **[Westphal et al., 2025. "A Generalized Information Bottleneck Theory of Deep Learning." arXiv:2509.26327]** -- Reformulates the IB principle through synergy: the information obtainable only through joint processing of features. PMI-based reweighting scheme. Feature-wise synergy decomposition penalizes representations relying too heavily on individual features. Novel lens: absorption concentrates information in single latents, maximizing redundancy and minimizing synergy.

6. **[Cui et al., 2025. "On the Limits of Sparse Autoencoders." arXiv:2506.15963]** -- First closed-form theoretical analysis showing SAEs generally fail to recover ground truth monosemantic features unless features are extremely sparse. Proposes reweighted SAE (WSAE). Establishes a theoretical impossibility result that constrains what any absorption mitigation can achieve.

7. **[Leask et al., 2025. "Sparse Autoencoders Do Not Find Canonical Units of Analysis." arXiv:2502.04878, ICLR 2025]** -- Meta-SAEs decompose SAE latents into meta-latents. Novel latents in larger SAEs capture information missed by smaller ones. Shows SAE features are neither complete nor atomic. Directly relevant: meta-SAE decomposition may reveal the internal structure of absorbing latents.

8. **[Chanin & Garriga-Alonso, 2025. "Sparse but Wrong." arXiv:2508.16560]** -- Incorrect L0 causes feature mixing. Too-low L0 triggers hedging, too-high finds degenerate solutions. Most open-source SAEs have L0 that is too low. Critical confound: much observed "absorption" may actually be L0-induced hedging masquerading as hierarchy-driven absorption.

9. **[Mir, 2026. "Renormalization-Group Principles for Deep Neural Architectures." Research Square]** -- Formal correspondence between network depth and RG scale transformations. Each layer implements learned coarse-graining that contracts Fisher information geometry. Three falsifiable hypotheses about scale-structure in neural representations. Fresh theoretical angle: absorption as a failure of coarse-graining across representation scales.

10. **[Ivanov et al., 2026. "Spectral Superposition: A Theory of Feature Geometry." arXiv:2602.02224]** -- Spectral theory for feature geometry via the frame operator F = WW^T. Analyzes how features allocate norm across eigenspaces. New mathematical tools for understanding decoder weight structure and its relationship to representational capacity.

11. **[Sardy et al., 2025. "Validation-Free Sparse Learning: A Phase Transition Approach." arXiv:2411.17180]** -- Demonstrates phase transitions in probability of exact support recovery for sparse learners. HarderLASSO achieves sharp phase transition at quantile universal threshold. Structural analogy: absorption may exhibit a sharp phase transition as a function of SAE capacity-to-hierarchy-complexity ratio.

12. **[Narayanaswamy et al., 2026. "Masked Regularization for SAEs." arXiv:2604.06495]** -- Token masking during training disrupts co-occurrence patterns, reducing absorption and improving OOD robustness. Very recent. Validates that co-occurrence statistics are the proximal driver of absorption formation during training.

### Landscape Summary

After four iterations, the project has established that: (1) the Lotka-Volterra competitive exclusion framework provides a useful vocabulary but its predictive power is weak (H1 F1=0.128, H2 partial R^2=0.0006); (2) the strongest finding is the correlation between absorption and downstream SAE quality (H3: r=-0.595, partial r=-0.661 for sparse probing), but this is confounded with SAE width and L0; (3) a taxonomy of absorption types was developed but the 92.3% classification rate is inflated by measurement artifacts.

The critical insight from iter_4 that has not been fully exploited: the H3 finding -- that absorption correlates with downstream task performance -- is potentially the most impactful contribution, but only if the width/L0 confound is resolved. The iter_4 reflection explicitly states that reaching 7.0+ requires zero-GPU confound control analysis, not new experiments.

Meanwhile, the broader field has advanced in three directions that the project has not yet leveraged: (1) causal abstraction theory now provides a formal framework for defining what it means for an SAE to "faithfully represent" a feature hierarchy -- absorption can be rigorously defined as a failure of causal faithfulness; (2) the information bottleneck / synergy perspective offers a principled way to measure whether a latent is carrying "too much" information (absorbing multiple concepts) vs. the right amount; (3) phase transition theory from sparse recovery and compressed sensing provides mathematical tools for predicting when absorption becomes inevitable as a function of problem parameters.

The four previous innovator proposals have explored: (a) tree-structured correction at inference time, (b) rate-distortion theory of absorption, (c) co-occurrence graph topology, and (d) unsupervised detection via co-activation geometry (CMI + cosine). All have been creative but share a common weakness: they propose new methods without first establishing the empirical foundations that would validate whether those methods address the right problem. The iter_4 experiment results show that the most impactful direction is not a new detection method or architectural fix, but rather establishing rigorous empirical scaling laws and confound-controlled causal claims about absorption's downstream impact.

---

## Phase 2: Initial Candidates

### Candidate A: Absorption Phase Diagrams -- Mapping the Critical Boundaries Where Absorption Becomes Inevitable

- **Hypothesis**: There exists a sharp phase transition in absorption rate as a function of the ratio R = (SAE effective capacity) / (feature hierarchy complexity), where effective capacity is D/L0 (dictionary size divided by mean sparsity) and hierarchy complexity is measured by the number of implication pairs in the feature hierarchy weighted by their frequency ratio. Below a critical R*, absorption is near-zero; above R*, absorption rate plateaus at a model-dependent maximum. This phase transition can be mapped empirically using the 400+ Gemma Scope SAEs (spanning widths 1k to 1M and L0 from 9 to 445) combined with the existing sae-spelling absorption metric, producing the first absorption phase diagram -- analogous to the Donoho-Tanner phase diagram in compressed sensing. Crucially, this framework predicts that the iter_4 H3 finding (absorption correlates with downstream quality) is a consequence of operating on one side of the phase boundary, and that the width/L0 confound resolves naturally once the phase diagram is conditioned on R rather than width or L0 independently.

- **Cross-domain insight**: The Donoho-Tanner phase transition from compressed sensing establishes that sparse signal recovery undergoes a sharp transition: below a critical measurement-to-sparsity ratio, exact recovery is impossible; above it, recovery succeeds with high probability. The phase boundary is a smooth curve in the (delta=n/p, rho=k/n) plane. In SAE absorption, the analogous parameters are: delta = D/d (overcomplete ratio, dictionary size / model dimension) and rho = L0/D (fraction of active features). The absorption phase diagram maps the boundary in (delta, rho) space where absorption transitions from avoidable to inevitable. The key insight from recent phase transition work (Sardy et al., arXiv:2411.17180) is that this boundary is not smooth -- it is sharp, and the sharpness enables precise prediction. Moreover, the "price of sparsity" result (arXiv:2509.01809) shows that reducing measurement density (analogous to lower L0) inflates the required sample size by a factor that depends on the sparsity structure -- directly explaining why lower-L0 SAEs show higher absorption.

- **Evidence for**: (a) Chanin et al. empirically observe that absorption increases with lower L0 and wider SAEs -- directional consistency with a phase boundary that depends on D/L0. (b) Gemma Scope provides 400+ SAEs spanning 3 orders of magnitude in width, enabling a dense empirical sweep. (c) The unified SDL theory (arXiv:2512.05534) proves absorption arises as spurious minima, suggesting a critical capacity threshold below which the absorbing solution is the global minimum. (d) SAE scaling laws (arXiv:2406.04093) demonstrate that CE loss follows clean power laws in width and L0 -- absorption may follow similarly clean scaling but has never been measured. (e) The iter_4 H3 finding (r=-0.595) covers 54 Gemma Scope SAEs with known width, L0, and absorption scores -- directly enabling the phase diagram computation.

- **Novelty estimate**: 9/10 -- No paper has constructed an absorption phase diagram. The directional observation (wider/sparser SAEs have more absorption) is known but has never been formalized as a phase transition with a critical boundary. The compressed sensing phase diagram analogy is novel to the SAE field. The specific innovation of conditioning on R = D/L0 as the single control parameter (rather than width and L0 independently) is new and resolves the iter_4 confound.

### Candidate B: Causal Abstraction Score for Absorption -- Measuring Feature Faithfulness Without Probes

- **Hypothesis**: Feature absorption can be rigorously defined and measured as a failure of causal abstraction faithfulness, where the SAE's feature-level causal model fails to predict the effect of intervening on the base model's activations. Specifically, for a feature hierarchy with parent P and child C, absorption manifests as: intervening to set the child feature's activation to zero in the SAE changes the base model's output prediction for P-related tokens, while intervening to set the parent feature's activation to zero does NOT change the output prediction for C-related tokens -- the asymmetric causal influence signature. This "Causal Absorption Score" (CAS) can be computed entirely from the base model, the SAE, and a text corpus, without any probe training or labeled features. It measures the degree to which the SAE's feature decomposition is causally faithful to the model's computation. CAS achieves AUROC > 0.80 against Chanin et al.'s probe-based absorption labels on the first-letter task, while being applicable to any feature hierarchy without pre-specification.

- **Cross-domain insight**: Causal abstraction theory (Geiger et al., JMLR 2025) formalizes the relationship between a high-level causal model (the SAE's feature decomposition) and a low-level causal model (the base model's computation) through "interchange interventions" -- swapping activations between inputs and measuring whether the high-level model's predictions about the effect match reality. This is exactly the test needed for absorption: if the SAE claims features P and C are separate, then intervening on P should have an effect on P-related predictions independent of C. Absorption violates this -- the causal effect of P is mediated through C's latent. The DAS (Distributed Alignment Search) methodology from the causal abstraction literature provides the algorithmic template: search for the rotation of SAE latent space that maximizes causal faithfulness for a given task, then measure how much the default (unrotated) SAE features fall short.

- **Evidence for**: (a) Causal abstraction has been successfully applied to study feature representations in language models (Geiger et al., 2024, 2025). (b) The iter_4 experiments already used DAS (H4) to measure whether absorption-reducing interventions have the expected causal effect -- the 42.3% positive slope rate suggests DAS captures something real about absorption but with high variance. (c) Anthropic's circuit tracing work (Lindsey et al., 2025) demonstrates that SAE features can anchor causal claims about model computation -- absorption undermines exactly these causal claims. (d) The "Non-Linear Representation Dilemma" paper (NeurIPS 2025 Spotlight) shows that unconstrained causal abstraction is trivially satisfiable -- linearity constraints (which SAEs naturally enforce) prevent the trivial solution, making CAS non-degenerate.

- **Novelty estimate**: 8/10 -- No paper has framed absorption as a failure of causal abstraction faithfulness. The iter_4 experiments used DAS but as one test among many, not as the central theoretical framework. The causal abstraction literature has not been applied to absorption specifically. The key novelty is the asymmetric causal influence signature (intervening on child changes parent-related predictions but not vice versa) as a probe-free absorption detector. However, some of the DAS machinery is already established.

### Candidate C: The Absorption-Quality Causal Chain -- Establishing Causation Beyond Correlation Through Controlled Width-Stratified Analysis

- **Hypothesis**: The iter_4 H3 finding (absorption rate correlates with downstream SAE quality at r=-0.595) reflects a genuine causal mechanism, not a confound with SAE width or L0. This can be established through three analyses, all training-free: (1) Width-stratified partial correlations -- within fixed-width groups (16k, 65k, 131k, 1M), absorption still correlates with quality at |r| > 0.3; (2) Mediation analysis -- absorption mediates the effect of L0 on downstream quality (Sobel test p < 0.05), meaning L0's negative effect on quality operates through increased absorption; (3) Dose-response curve -- across the 54 Gemma Scope SAEs, the absorption-quality relationship follows a monotonic dose-response pattern that is robust to propensity score matching on width and L0. If all three hold, absorption is established as a causal quality indicator for SAEs -- the first such result in the field, directly actionable for SAE selection and hyperparameter tuning.

- **Cross-domain insight**: Epidemiology's Bradford Hill criteria for establishing causation from observational data provide the framework: (1) strength of association (r=-0.595 is substantial), (2) consistency (across multiple quality metrics: sparse probing, RAVEL, SCR), (3) specificity (absorption predicts quality better than L0 alone), (4) temporality (absorption is measured on the SAE, quality is measured on downstream tasks), (5) biological gradient / dose-response (monotonic absorption-quality curve), (6) plausibility (absorption causes false negatives in feature detection, which degrades any task relying on feature completeness). The key transplant is the mediation analysis framework from causal epidemiology: rather than just showing correlation, decompose the total effect of L0 on quality into the direct effect and the absorption-mediated indirect effect. If the indirect effect is significant, absorption is a genuine causal mechanism.

- **Evidence for**: (a) The iter_4 data already shows r=-0.595 (partial r=-0.661 controlling for width and layer) across 54 Gemma Scope SAEs. (b) SAEBench provides absorption scores and 7 other quality metrics for these SAEs, enabling multivariate analysis. (c) The width/L0 confound is well-characterized: all 5 high-absorption SAEs are 1M width, all 5 low-absorption are 16k/65k. This is the BLOCKING issue identified in iter_4 reflection. (d) Mediation analysis is well-established in epidemiology and psychology but has not been applied to SAE evaluation. (e) The zero-GPU cost makes this the highest-ROI direction.

- **Novelty estimate**: 7/10 -- The epidemiological causal framework applied to SAE quality is novel. The specific analyses (width-stratification, mediation, dose-response) have not been performed on absorption data. However, this is primarily a methodological contribution (better statistical analysis of existing data) rather than a conceptual breakthrough. Its strength is practical impact: if the causal chain holds, it directly informs SAE selection criteria.

---

## Phase 3: Self-Critique

### Against Candidate A

- **Prior work attack**: Searched for "absorption phase transition SAE capacity threshold" and "sparse autoencoder scaling law absorption width sparsity." The iter_2 innovator proposed a "Rate-Distortion Theory of Absorption" with a scaling law AR(W, L0, alpha), and the iter_3 innovator proposed "Absorption Scaling Laws" with a sigmoid parameterization. Both proposed scaling laws. However, neither proposed a phase diagram with a sharp phase transition -- they assumed smooth scaling laws (power laws, sigmoids). The Donoho-Tanner phase transition is qualitatively different from a smooth scaling law: it predicts a critical boundary where absorption jumps from near-zero to high, rather than smoothly increasing. No SAE paper has tested whether absorption exhibits a sharp phase transition.

- **Methodological attack**: The empirical measurement requires dense coverage of the (D/d, L0/D) parameter space. Gemma Scope SAEs are not uniformly distributed in this space -- they cluster at specific width/L0 combinations chosen by DeepMind. The phase boundary may fall in a poorly covered region. Additionally, the Chanin et al. absorption metric was designed for the first-letter task, which provides only 26 binary classification targets -- the effective sample size for estimating absorption rate per SAE configuration may be too small for detecting a sharp transition.

- **Theoretical attack**: The Donoho-Tanner phase transition assumes random measurement matrices, while SAE encoders are learned. The learning process may smooth out any sharp transition through adaptive optimization. Moreover, the "hierarchy complexity" parameter is hard to define without ground-truth knowledge of the feature hierarchy, creating a circularity problem for the phase diagram's independent variable.

- **Scalability attack**: Computing absorption rate for 400+ SAEs on the first-letter task requires running the full sae-spelling pipeline for each SAE. Even if each takes 5-10 minutes, the full sweep is 30-60 GPU-hours -- well beyond the 1-hour constraint unless SAEBench precomputed scores are available.

- **Verdict**: MODERATE -- The conceptual novelty (sharp phase transition vs. smooth scaling) is genuine. But the practical execution faces three risks: (1) SAEBench already provides absorption scores for many Gemma Scope SAEs, making the empirical measurement potentially trivial; (2) the phase transition may not be sharp, in which case this reduces to the iter_2/3 scaling law proposals; (3) the hierarchy complexity parameter may be intractable. The saving grace is that SAEBench precomputed data makes the empirical test cheap, and even a smooth phase diagram would be a novel contribution.

### Against Candidate B

- **Prior work attack**: Searched for "causal abstraction sparse autoencoder absorption faithfulness intervention." Found: (1) The iter_4 experiments already used DAS to measure absorption-related causal effects (H4: 42.3% positive slopes). (2) Geiger et al.'s JMLR 2025 paper defines causal abstraction but does not apply it to absorption. (3) The "Non-Linear Representation Dilemma" (NeurIPS 2025 Spotlight) shows that unconstrained causal abstraction is trivially satisfiable -- but with linear constraints, it becomes meaningful. No paper has formally framed absorption as a causal abstraction failure.

- **Methodological attack**: The causal abstraction framework requires specifying a high-level causal model to test against. For the first-letter task, this is clear (letter -> token), but for arbitrary feature hierarchies, the high-level model is unknown. This creates the same probe-dependency problem as Chanin et al.'s metric -- you need to know the hierarchy to test faithfulness.

- **Theoretical attack**: The CAS score depends on the specific intervention (which latent to zero out) and the specific target variable (which output to measure). Different choices of target variable may give contradictory absorption scores. Unlike the Chanin et al. metric, which has a well-defined binary outcome (absorbed or not), CAS is a continuous measure whose threshold for "absorbed" is arbitrary.

- **Scalability attack**: Interchange interventions require running the base model on many input pairs, which is O(N^2) in the number of test examples. For efficiency, this must be amortized, but the amortization introduces approximation error.

- **Verdict**: MODERATE -- The theoretical framing is elegant and addresses a real gap (connecting absorption to causal abstraction). But the practical execution is complex, the probe-dependency problem resurfaces for general hierarchies, and the iter_4 DAS experiments already provide partial validation. The main contribution would be the theoretical framing rather than a new method.

### Against Candidate C

- **Prior work attack**: Searched for "SAE absorption downstream quality causal mediation analysis stratified." Found: No paper has performed mediation analysis on the absorption-quality relationship. The iter_4 H3 finding is the first systematic correlation result. The SAEBench paper measures both absorption and quality metrics but does not analyze the causal structure. No prior work conflict.

- **Methodological attack**: The 54-SAE dataset has severe distributional issues: 5 high-absorption SAEs are all 1M width, 5 low-absorption are all 16k/65k. Width-stratified analysis within these groups may have n < 10 per stratum, yielding insufficient power for partial correlations. The mediation analysis assumes linear relationships and may fail if the absorption-quality relationship is nonlinear or threshold-based. Propensity score matching with n=54 total observations is statistically fragile.

- **Theoretical attack**: Mediation analysis from observational data cannot establish causation without the sequential ignorability assumption (no unmeasured confounders). Training procedure differences (learning rate, epochs, random seed) are unmeasured confounders that could drive both absorption and quality. The Bradford Hill criteria are heuristics, not formal causal inference methods.

- **Scalability attack**: All analyses operate on the existing 54-SAE dataset. No scalability issue. The main limitation is statistical power, not computation.

- **Verdict**: STRONG -- Despite the methodological limitations (small n, potential confounders), this candidate has three critical strengths: (1) it directly addresses the iter_4 BLOCKING issue (width/L0 confound); (2) it requires zero GPU resources; (3) if successful, it transforms the absorption-quality correlation from a suspicious finding into an actionable causal claim. The methodological attacks are real but can be addressed: bootstrap for small-sample inference, sensitivity analysis for unmeasured confounders (Rosenbaum bounds), and acknowledging the observational limitation transparently. The statistical power issue is the most serious -- but even null results (stratified correlations are insignificant) would be informative, ruling out the causal claim and pointing to width as the true driver.

---

## Phase 4: Refinement

### Dropped Ideas

- None dropped outright. All three candidates are MODERATE or STRONG. However, priorities differ sharply in terms of ROI.

### Strengthened Ideas

**Candidate C (Absorption-Quality Causal Chain)** strengthened as follows:
- **Added sensitivity analysis**: Rosenbaum bounds to quantify how large an unmeasured confounder would need to be to explain away the absorption-quality correlation. If Gamma > 2.0 (meaning a hidden confounder would need to double the odds of high absorption to explain the result), the causal claim is robust.
- **Added bootstrap inference**: BCa bootstrap CIs (10,000 resamples) for all stratified correlations, addressing the small-n concern.
- **Added L0 as explicit covariate**: The iter_4 reflection's top priority is "L0 as covariate in H3 partial correlation." This is absorbed into Candidate C as Step 1.
- **Expanded scope**: Beyond width-stratified correlation, added the analysis of whether the iter_4 taxonomy results (92.3% classification rate) survive when conditioned on L0 and recomputed with proper comparison tokens (addressing the n_comparison_tokens=0 artifact).

**Candidate A (Phase Diagram)** strengthened by:
- **Using SAEBench precomputed data**: Instead of running the full sae-spelling pipeline for 400+ SAEs, use the existing SAEBench absorption scores (available for 200+ SAEs). This eliminates the computational concern.
- **Relaxed the "sharp transition" claim**: Instead of predicting a Donoho-Tanner-style sharp boundary, propose to empirically characterize the absorption rate surface in (log(D), log(L0)) space and test for non-linearity (interaction effects) that would indicate a phase-boundary-like structure, even if not perfectly sharp.

**Candidate B (Causal Abstraction Score)** deferred as a theoretical contribution for a later iteration -- the practical execution is complex and the iter_4 DAS results already provide partial validation. The conceptual framing (absorption as causal faithfulness failure) should be included in the paper's discussion section regardless.

### Additional Evidence Found

- **SAEBench interactive data**: The SAEBench website (neuronpedia.org/sae-bench) provides precomputed absorption scores, sparse probing accuracy, SCR scores, and unlearning scores for 200+ Gemma Scope SAEs. This data is freely available and eliminates the need for new absorption computation.
- **Iter_4 H3 raw data**: The project already has the 54-SAE dataset with absorption scores and all downstream metrics in JSON format. The confound analysis can be performed immediately.
- **Chanin & Garriga-Alonso (2025)**: Their L0 analysis provides a proxy metric for "correct L0" that can be used as an additional covariate in the mediation analysis, addressing the confound between L0 choice and absorption.

### Selected Front-Runner

**Candidate C: The Absorption-Quality Causal Chain** combined with elements of **Candidate A (Phase Diagram)**.

Rationale:
1. **Directly addresses the iter_4 BLOCKING issue**: The reflection explicitly states that reaching 7.0+ requires resolving the width/L0 confound. This is the highest-priority action item, and it requires zero GPU resources.
2. **Highest ROI**: Zero GPU cost, operates entirely on existing data, and the result (causal vs. confounded) determines whether the paper's strongest finding (H3) stands or falls. No other direction has this binary, high-stakes outcome.
3. **Novel methodological contribution**: Mediation analysis, propensity matching, and Rosenbaum sensitivity analysis have never been applied to SAE evaluation. Transplanting these epidemiological methods to SAE quality assessment is a genuine cross-domain contribution.
4. **Natural extension to phase diagram**: If the causal chain holds, the phase diagram (Candidate A) provides the scaling law that governs the causal mechanism. If it fails, the phase diagram explains why (absorption is confounded with width/L0 and the actual causal structure is simpler).
5. **Produces actionable guidance**: If absorption causally drives quality degradation, this immediately informs SAE selection criteria: prefer SAEs with low absorption score, not just good CE loss or sparsity. This is directly relevant to the community's need (highlighted by DeepMind's deprioritization of SAEs due to downstream task failures).

---

## Phase 5: Final Proposal

### Title

Beyond Correlation: Establishing Feature Absorption as a Causal Quality Indicator for Sparse Autoencoders via Epidemiological Methods

### Hypothesis

The negative correlation between SAE feature absorption rate and downstream task performance (sparse probing, RAVEL, spurious correlation removal) observed across 54 Gemma Scope SAEs (r=-0.595, Chanin absorption metric) reflects a genuine causal mechanism, not a confound with SAE width or L0 sparsity. This is established through three convergent lines of evidence: (1) width-stratified partial correlations remain significant (|r| > 0.3 within at least 2 of 4 width groups: 16k, 65k, 131k, 1M); (2) formal mediation analysis shows absorption mediates > 30% of L0's effect on downstream quality (indirect effect / total effect > 0.30); (3) sensitivity analysis confirms the result is robust to moderate unmeasured confounders (Rosenbaum Gamma > 1.5).

### Motivation

The iter_4 experiments produced the strongest finding yet in the absorption literature: a systematic negative correlation (r=-0.595, partial r=-0.661 controlling for width and layer) between absorption rate and three independent downstream quality metrics across 54 Gemma Scope SAEs. This is potentially the first evidence that absorption is not just a theoretical curiosity but a practical quality indicator for SAE selection.

However, the iter_4 reflection identified a critical confound: all 5 high-absorption SAEs are 1M width (L0 16-58), while all 5 low-absorption SAEs are 16k/65k (L0 137-297). It is possible that SAE width directly drives both absorption rate AND downstream quality, with absorption being an epiphenomenon rather than a causal mediator. Resolving this confound is essential: if absorption is merely a proxy for width, then mitigating absorption (via Matryoshka SAE, OrtSAE, etc.) would not improve downstream performance -- only increasing width would help. Conversely, if absorption is a genuine causal mechanism, then absorption mitigation is a high-value research direction for the entire mechanistic interpretability community.

The cross-domain methodological contribution is equally important: the SAE evaluation community currently lacks rigorous tools for establishing causal claims from observational SAE comparison data. Epidemiology has developed sophisticated methods (mediation analysis, propensity matching, sensitivity analysis) for exactly this problem -- extracting causal insights from non-randomized observational studies. Transplanting these methods to SAE evaluation establishes a methodological precedent that extends beyond absorption to any SAE quality claim made from cross-architecture comparisons.

### Method

**Step 1: L0 as Covariate (Direct answer to iter_4 BLOCKING issue)**

Using the existing 54-SAE dataset (Gemma Scope SAEs with SAEBench absorption and quality scores):
- Add log(L0) as a covariate in the partial correlation of absorption vs. sparse probing accuracy (currently controlling only for log(width) and layer).
- Compute the partial correlation r(absorption, sparse_probing | log(width), layer, log(L0)).
- If this drops below |0.3|, the L0 confound is confirmed and the finding weakens.
- Also compute for RAVEL and SCR.

**Step 2: Width-Stratified Analysis**

Partition the 54 SAEs into width groups (16k, 65k, 131k, 1M). Within each group:
- Compute Spearman correlation between absorption and each quality metric.
- Report BCa bootstrap 95% CIs (10,000 resamples) for each within-group correlation.
- If the correlation is significant (CI excludes 0) in at least 2 of 4 width groups, the within-width effect is established.
- If the correlation is only significant at 1M (where there are enough SAEs with varying L0), report this honestly as "the effect is detectable only at the widest scale."

**Step 3: Mediation Analysis**

Test the causal path: L0 -> Absorption -> Downstream Quality.
- Total effect: regress quality on log(L0) controlling for log(width) and layer.
- Direct effect: regress quality on log(L0) + absorption controlling for log(width) and layer.
- Indirect effect = total - direct. Proportion mediated = indirect / total.
- Test significance via Sobel test and bootstrap CI on the indirect effect (10,000 resamples).
- If proportion mediated > 0.30 with bootstrap CI excluding 0, absorption is a significant mediator.

**Step 4: Sensitivity Analysis (Rosenbaum Bounds)**

Compute Rosenbaum's Gamma for the matched-pair comparison:
- Match high-absorption and low-absorption SAEs on width (exact match) and L0 (nearest neighbor within caliper of 0.2 SD).
- For each matched pair, compute the quality difference.
- Compute the Gamma value at which the Wilcoxon signed-rank test on matched differences first becomes non-significant.
- Gamma > 1.5 indicates moderate robustness to unmeasured confounders; Gamma > 2.0 indicates strong robustness.

**Step 5: Absorption Phase Surface (Extension)**

Using SAEBench data for 200+ SAEs:
- Plot absorption rate as a function of (log(D), log(L0)) in a 2D heatmap.
- Fit a GAM (Generalized Additive Model) to the absorption surface: absorption ~ s(log(D)) + s(log(L0)) + ti(log(D), log(L0)).
- Test the interaction term ti(log(D), log(L0)): if significant, the absorption surface has non-linear structure that cannot be explained by width or L0 independently.
- If a sharp boundary is visible in the surface (high-absorption region separated from low-absorption by a narrow transition zone), this provides the first empirical evidence for phase-transition-like behavior.

**Step 6: Reconciling the 92.3% Taxonomy Rate**

Recompute the iter_4 taxonomy classification using proper comparison tokens (not the global fallback):
- For each letter with n_comparison_tokens=0, use tokens from the same frequency band as the target letter's tokens (matched on log-frequency, not same letter).
- Recompute Type II classification with this proper baseline.
- Report the corrected classification rate alongside the original 92.3%.

### Experimental Plan

**Pilot (10-15 min)**: Load the iter_4 H3 dataset (54 SAEs with absorption and quality scores). Compute the L0-controlled partial correlation (Step 1). If the correlation drops below |0.2|, the causal chain hypothesis is unlikely and we pivot to characterizing width as the true driver. This is the critical go/no-go decision point.

**Main experiment 1 (20 min)**: Steps 2-4 -- full width-stratified, mediation, and sensitivity analysis on the 54-SAE dataset. All computed in Python with scipy, statsmodels, and custom bootstrap code. Zero GPU required.

**Main experiment 2 (30 min)**: Step 5 -- absorption phase surface using SAEBench precomputed data (200+ SAEs). Requires downloading SAEBench data and fitting a GAM. Zero GPU required (may need to download data from Hugging Face, but no model inference).

**Main experiment 3 (20 min)**: Step 6 -- recompute taxonomy with proper comparison tokens. Requires re-running the taxonomy classification code from iter_4 with modified comparison token selection. May need GPT-2 Small model loaded for activation caching (~2GB VRAM), but the classification logic runs on CPU.

**What would falsify the hypothesis**: 
- If the L0-controlled partial correlation drops below |0.2| for all three quality metrics, the causal chain is falsified -- absorption is an epiphenomenon of L0.
- If width-stratified correlations are non-significant in all 4 groups, the effect is purely ecological (between-width, not within-width).
- If the mediation indirect effect bootstrap CI includes 0, absorption is not a significant mediator.
- If Rosenbaum Gamma < 1.2, the result is fragile and likely driven by unmeasured confounders.

### Resource Estimate

All experiments are training-free and operate on existing precomputed data.
- Memory: Minimal (Python analysis code, ~1GB RAM for dataframes).
- Compute: Steps 1-4 require < 5 min CPU (bootstrap is the bottleneck, trivially parallelized). Step 5 requires downloading SAEBench data (~100MB) and fitting a GAM (~10 min CPU). Step 6 requires GPT-2 Small (~500MB VRAM) for activation caching on ~5000 tokens (~5 min on any GPU).
- Total: well within the 1-hour constraint. Most time is analysis, not computation.

### Risk Assessment

**Risk 1**: The L0-controlled partial correlation kills the finding (drops below |0.2|).
- Mitigation: This is actually the most informative outcome. It establishes that width/L0, not absorption, drives quality -- a strong negative result that redirects the field. Report it prominently and analyze what width/L0 confound structure looks like, since this would invalidate the entire class of absorption mitigation approaches.

**Risk 2**: Sample sizes within width strata are too small for reliable inference (n < 10 per group).
- Mitigation: Use nonparametric methods (Spearman, bootstrap) rather than parametric tests. Report exact p-values and CIs even for small samples. Supplement with the 200+ SAE SAEBench dataset for Step 5, which has larger per-stratum samples.

**Risk 3**: The mediation analysis fails because the absorption-quality relationship is nonlinear.
- Mitigation: Use Baron-Kenny mediation with a generalized link (logistic for binary quality outcomes) rather than assuming linearity. Also report nonparametric mediation via bootstrap without parametric assumptions.

### Novelty Claim

This work provides three novel contributions:

1. **First rigorous confound control for the absorption-quality relationship**: Previous work (including iter_4 H3) reported correlations between absorption and downstream quality, but the width/L0 confound was not controlled. This is the first analysis to disentangle absorption's independent contribution via stratification, partial correlation with L0, and mediation analysis.

2. **First application of epidemiological causal methods to SAE evaluation**: Mediation analysis, propensity matching, Rosenbaum sensitivity analysis, and Bradford Hill criteria have not been applied to SAE quality assessment. This establishes a methodological precedent for rigorous causal claims in the SAE evaluation literature, which currently relies on correlation-based comparisons.

3. **First absorption phase surface**: The 2D heatmap of absorption rate as a function of (log(D), log(L0)) across 200+ SAEs provides the first empirical characterization of how absorption varies jointly with the two most important SAE hyperparameters. Whether or not it exhibits a sharp phase transition, this phase surface is immediately actionable for SAE hyperparameter selection.

Supporting evidence for novelty: (a) The iter_4 reflection explicitly identifies the width/L0 confound as the BLOCKING issue preventing score improvement -- no prior work has addressed it. (b) A literature search for "SAE mediation analysis" and "SAE propensity matching" returns zero relevant results. (c) The phase surface extends the scaling law direction (proposed in iters 2-3) by using precomputed SAEBench data (not available to earlier iterations) and focusing on the empirical surface rather than a parametric functional form.
