# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Feature absorption is the "most fundamental" SAE failure mode**
   - Evidence challenging it: DeepMind's 2025 blog post deprioritizing SAE research found that SAE probes underperform simple linear probes on downstream tasks, but this gap persists even for SAEs with low absorption scores (e.g., Matryoshka SAEs). The "Sanity Checks for Sparse Autoencoders" paper (Korznikov et al., arXiv:2602.14111) shows that random baselines match trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72) -- problems that have nothing to do with absorption. The SAE failure mode hierarchy may be inverted: the truly fundamental problem is that SAEs do not reliably recover ground-truth features at all (only 9% in synthetic settings), making absorption a downstream symptom rather than a root cause.

2. **Assumption: The first-letter spelling task is a valid proxy for measuring absorption**
   - Evidence challenging it: The first-letter spelling task is an artificial, constructed hierarchy where the model is prompted with "X has the first letter:" -- this is not a feature hierarchy the model naturally represents. Chanin et al. themselves note that the absorption metric requires known probe directions and only detects cases where a single absorber dominates. The "Looking for feature absorption automatically" LessWrong post (2024) explicitly reports that proposed automatic detection methods "don't work in practice." The feature sensitivity paper (Tian et al., arXiv:2509.23717) frames absorption as a special case of poor sensitivity, suggesting the metric may be capturing a broader measurement problem rather than a specific mechanistic phenomenon.

3. **Assumption: Absorption severity increases with sparsity and is monotonically bad**
   - Evidence challenging it: "Sparse but Wrong" (Chanin & Garriga-Alonso, arXiv:2508.16560) shows that incorrect L0 causes feature hedging -- a DIFFERENT failure mode that looks superficially similar to absorption in measurements. If L0 is too low, the SAE mixes correlated features; if too high, it finds degenerate solutions. Most commonly used SAEs have L0 that is too low, meaning what is measured as "absorption" may actually be feature hedging from incorrect sparsity. The Matryoshka SAE paper shows that reducing absorption comes at the cost of increased hedging, suggesting these may be two sides of the same measurement coin rather than distinct phenomena.

4. **Assumption: Absorption creates systematic "holes" in feature recall that undermine interpretability**
   - Evidence challenging it: The position paper "Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts" (arXiv:2506.23845) argues SAEs are not meant to be classifiers. The absorption metric evaluates SAEs as classifiers (does the "starts with S" latent fire on all S-words?), but SAEs may be doing something valuable even when individual latents have holes -- they enumerate a large set of relatively monosemantic features for discovery purposes. DeepMind's team argues that "a decent approximation to the model's computation might well be good enough" for many practical goals.

5. **Assumption: The rate-distortion framing (from the current proposal) is novel and insightful**
   - Evidence challenging it: Chanin et al. (2024) already stated that "for each parent-child feature relation, a vanilla SAE with feature absorption can represent both features with +1 L0, while an SAE without feature absorption would require +2 L0." This IS the rate-distortion argument -- formalizing it does not add explanatory power. The unified SDL theory paper (arXiv:2512.05534) already provides a principled theoretical framework for absorption via piecewise biconvex optimization, and the Tilde Research blog "The Rate Distortion Dance of Sparse Autoencoders" has already explored rate-distortion interpretations of SAE sparsity-reconstruction tradeoffs. The proposed "absorption threshold" formula (lambda > sin^2(theta)) is elegant but may be vacuously true -- if the SAE has already optimized decoder angles to facilitate absorption, measuring those angles post-hoc to "predict" absorption is circular.

6. **Assumption: A probe-free absorption detection method (ASI) is feasible and useful**
   - Evidence challenging it: The proposed ASI = cos^2(theta) x (freq_p / freq_c) computed from decoder geometry is essentially measuring decoder column similarity -- which already correlates with feature cosine similarity by construction. High cosine similarity between decoder columns means the SAE has learned similar directions, which could indicate many things besides absorption (e.g., feature splitting, polysemanticity, noise). Without probe-based ground truth, there is no way to know whether high-ASI pairs are absorbed pairs, split pairs, or simply co-occurring features that the SAE represents similarly. The "Looking for feature absorption automatically" post already tried a related approach (finding latents with similar causal effects that don't co-activate) and reported that it does not work in practice.

7. **Assumption: Phase transitions and hysteresis in absorption are detectable and meaningful**
   - Evidence challenging it: The Gemma Scope SAE suite provides a limited number of discrete L0 settings -- typically 3-5 per width -- which is far too few data points to distinguish a smooth curve from a phase transition. The proposed hysteresis test (fine-tune a SAE for 500 steps with reduced sparsity) confounds the question: 500 steps is insufficient to fully retrain the decoder, so failure to reverse absorption proves nothing about whether the absorbed state is metastable vs. simply that 500 steps is too few.

8. **Assumption: Cross-domain absorption characterization will reveal new insights**
   - Evidence challenging it: The absorption metric's reliance on probe directions means that extending to new domains (entity type, geographic, grammatical) requires building entirely new probe-based evaluation pipelines. Each domain introduces its own confounds: entity-type hierarchies depend on how the model represents entities (which may not be hierarchical in activation space), geographic knowledge may be stored differently than syntactic patterns, and POS tag hierarchies may not align with the model's internal representations. SynthSAEBench (arXiv:2602.14687) already shows that logistic probes achieve 0.974 F1 while SAEs substantially underperform in synthetic settings with known hierarchies -- this suggests the problem is not domain-specific but fundamental to the SAE objective.

### Landscape of Doubt

The most striking pattern across the literature is a growing disconnect between the SAE research community's focus on understanding and mitigating specific failure modes (absorption, hedging, splitting) and the more fundamental finding that SAEs may not recover meaningful features at all. The "Sanity Checks" paper (2026) shows random baselines match trained SAEs on standard evaluation metrics. "Sparse but Wrong" (2025) shows incorrect L0 alone produces incorrect features. "On the Limits of Sparse Autoencoders" (2025) proves SAEs generally fail to recover ground truth features unless features are extremely sparse. Meanwhile, the community continues to develop elaborate architectures (Matryoshka, OrtSAE, ATM, KronSAE) to fix absorption -- a specific failure mode that may be merely one visible symptom of the deeper problem that SAEs are the wrong tool for the job.

The current proposal doubles down on absorption as the central object of study, proposing an elaborate theoretical framework (rate-distortion optimality), a novel metric (ASI), and ambitious empirical characterization (cross-domain, phase transitions). But this approach assumes that understanding absorption will lead to better SAEs. The contrarian evidence suggests the opposite: understanding absorption better may simply demonstrate more precisely that absorption is an inherent, unavoidable property of sparsity optimization under any realistic feature structure -- confirming what practitioners already suspect, without yielding actionable improvements.

The strongest contrarian signal comes from the transcoders literature: skip transcoders (Paulo et al., arXiv:2501.18823) reduce absorption as a side effect of mapping MLP inputs to outputs, without any absorption-specific optimization. This suggests absorption is an artifact of the SAE paradigm (encoding and reconstructing activations), not a fundamental property of neural network features. If the field is moving toward transcoders and crosscoders anyway -- as Anthropic's attribution graphs work (Lindsey et al., 2025) strongly suggests -- then an elaborate theory of SAE-specific absorption becomes an intellectual curiosity rather than a practical contribution.

---

## Phase 2: Initial Candidates

### Candidate A: "The Absorption Mirage: Feature Absorption is Primarily a Measurement Artifact of Probe-Dependent Evaluation, Not a Mechanistic SAE Failure"

- **Challenged assumption**: Feature absorption is a real, mechanistic failure mode where specific SAE latents fail to fire on inputs where they should.
- **Evidence against**: (1) The absorption metric requires pre-specifying which features to look for via probes, creating a self-fulfilling prophecy -- if the probe finds a feature that the SAE represents differently, that counts as "absorption." (2) "Sparse but Wrong" shows that incorrect L0 produces incorrect features that would also trigger the absorption metric (hedging masquerading as absorption). (3) The "Looking for feature absorption automatically" effort failed, suggesting that what probes detect as absorption may not correspond to a structurally detectable pattern. (4) Random baselines match trained SAEs on evaluation metrics, suggesting the "features" themselves are not as meaningful as assumed.
- **Contrarian hypothesis**: At least 50% of what the Chanin et al. metric measures as "absorption" is actually a mix of (a) feature hedging from incorrect L0, (b) the model's actual representation not matching the linear probe's assumed hierarchy, and (c) the artificial nature of the first-letter task. True mechanistic absorption (where the SAE could represent the feature but chose not to due to sparsity incentives) accounts for a minority of measured cases.
- **Exploitation plan**: Decompose the absorption metric into sub-categories by checking whether each "absorbed" case also exhibits hedging signatures (as defined in Chanin & Garriga-Alonso, 2025). Run the metric at the "correct" L0 (as identified by the Sparse but Wrong proxy metric) vs. the standard L0. Measure absorption on features that the model actually uses in computation (via causal intervention) vs. features that merely correlate with probe directions.
- **Novelty estimate**: 7/10

### Candidate B: "When Feature Absorption is Feature Wisdom: The SAE's Implicit Regularization Against Spurious Hierarchies"

- **Challenged assumption**: Feature absorption is always bad for interpretability and downstream tasks.
- **Evidence against**: (1) The first-letter task is an artificial hierarchy -- the model may correctly recognize that first-letter membership is not a causally important feature and deprioritize it. (2) Transcoders, which map inputs to outputs (capturing computation rather than representation), show reduced absorption as a side effect -- because computational features naturally avoid representing causally unimportant hierarchies. (3) DeepMind's position paper argues that imperfect feature decomposition may be "good enough" for practical goals. (4) The unified SDL theory paper shows absorption is inherent to optimal sparse coding -- perhaps because it IS optimal for the downstream use of these representations (compact coding of what matters).
- **Contrarian hypothesis**: For at least some feature hierarchies, absorption represents the SAE correctly learning that the parent feature is not independently useful for the model's computation. The model itself does not maintain a clean "starts with S" feature that fires on all S-words because the model does not need such a feature for next-token prediction. The SAE is faithfully reflecting this computational structure, not distorting it.
- **Exploitation plan**: Compare absorption rates for hierarchies that the model demonstrably uses in computation (identified via causal tracing / logit lens) vs. hierarchies that are merely present in the representation (identified by linear probes). Hypothesis: absorption is higher for "representational but not computational" features. Test on safety-relevant features where the model does use the feature computationally (e.g., toxicity detection in chat models).
- **Novelty estimate**: 8/10

### Candidate C: "Rethinking Absorption: Why the Real Problem is SAE Non-Identifiability, Not Feature Hierarchy"

- **Challenged assumption**: Feature absorption is caused by hierarchical feature structure interacting with sparsity penalties.
- **Evidence against**: (1) Feature inconsistency (Song et al., arXiv:2505.20254) shows SAEs trained on the same data learn different feature sets, suggesting the decomposition is not unique. If the decomposition itself is non-unique, then "absorption" of specific features is an artifact of one particular decomposition, not a general property. (2) The sanity checks paper shows random baselines match trained SAEs, suggesting the "features" that get absorbed may not be well-defined in the first place. (3) Non-linear features (Engels et al., 2024 -- "Not All Language Model Features Are Linear") mean the linear representation hypothesis underlying SAEs is wrong for some features; absorption of features that are not linear to begin with is meaningless. (4) "On the Limits of SAEs" shows fundamental identifiability failures that are more general than absorption.
- **Contrarian hypothesis**: Feature absorption is a surface manifestation of the deeper problem of SAE non-identifiability. The same data admits multiple equally valid sparse decompositions, and what we call "absorption" is simply one decomposition's way of representing the same information. Fixing absorption (via Matryoshka, OrtSAE, etc.) is equivalent to choosing a different point in the equivalence class of valid decompositions, not recovering "true" features.
- **Exploitation plan**: Train multiple SAEs with different random seeds on the same data. For each pair, compute the PW-MCC consistency metric (Song et al., 2025). Then measure absorption rates in each. Hypothesis: features that are absorbed in one SAE may be perfectly represented in another, and the intersection (features absorbed across all seeds) is much smaller than the union. This would demonstrate absorption is seed-dependent, i.e., a decomposition artifact rather than a structural necessity.
- **Novelty estimate**: 9/10

---

## Phase 3: Self-Critique

### Against Candidate A: "The Absorption Mirage"

- **Steelman**: Chanin et al.'s toy model analysis provides a clean mathematical proof that absorption MUST occur in a two-layer network with hierarchical features and a sparsity penalty. This is not a measurement artifact -- it is a provable property of the optimization landscape. The integrated-gradients ablation used to detect absorption is causal, not correlational: when the absorbing latent is ablated, the model's ability to perform the first-letter task on the affected tokens is destroyed. This rules out mere probe-direction mismatch.
- **Cherry-picking check**: The "random baselines match trained SAEs" result from Korznikov et al. (2026) has been debated -- the AutoInterp metric may simply be too weak to detect meaningful differences between random and trained features. Other metrics (SAEBench absorption task, RAVEL) do show clear differences. The "Sparse but Wrong" paper focuses on hedging, not absorption, and the two phenomena have different signatures (hedging involves mixing correlated features; absorption involves completely suppressing one feature in favor of another).
- **Confounding check**: The claim that "incorrect L0 causes what looks like absorption" is partially valid but does not explain absorption at correct L0. If we run the absorption metric at the L0 identified by the Sparse but Wrong proxy, absorption rates decrease but do not disappear. The core phenomenon persists.
- **Actionability check**: If the metric is partially confounded, the action item is to deconfound it -- which is itself a contribution. However, if the "true" absorption rate after deconfounding is small, the paper becomes a negative result that undermines the field rather than advancing it.
- **Verdict**: MODERATE -- The basic phenomenon is provably real (toy model), but the quantitative severity measured on real LLMs likely includes confounds from hedging and L0 misspecification. Candidate A overstates the case by claiming >50% is artifact.

### Against Candidate B: "When Absorption is Wisdom"

- **Steelman**: The strongest argument against this candidate is that absorption also occurs for features that the model demonstrably uses in computation. In the first-letter task, the model can correctly predict the first letter -- meaning it DOES compute this feature -- yet the SAE fails to represent it faithfully. If the model uses the feature, the SAE should capture it. Furthermore, DeepMind's negative results on safety-relevant probing show that absorption of safety features (which the model clearly "knows" about) leads to practically catastrophic failures.
- **Cherry-picking check**: The argument that "maybe the model doesn't really use first-letter features" is undermined by the fact that Chanin et al. verified causal mediation -- ablating the first-letter latents degrades model performance on the task. The model does use some representation of first-letter membership, even if it is not a single clean feature direction.
- **Confounding check**: The distinction between "representational but not computational" features vs. "computational" features is hard to operationalize cleanly. Causal tracing identifies features important for specific tasks, but a feature may be computationally important for some tasks and not others. The "wisdom" interpretation requires showing that absorbed features are consistently unimportant, which is hard to establish comprehensively.
- **Actionability check**: Even if some absorption is "wise," the paper needs to show this leads to a better understanding or a practical improvement. A result that says "some absorption is fine, some isn't, and we can't always tell which is which" is not very useful.
- **Verdict**: MODERATE -- The insight that not all absorption is equally harmful is likely correct and under-appreciated, but the "wisdom" framing is too strong. A softer version ("absorption severity correlates with feature computational importance") is more defensible and still novel.

### Against Candidate C: "Non-Identifiability, Not Hierarchy"

- **Steelman**: Chanin et al.'s toy model proves that absorption is not merely a decomposition choice -- in the two-feature hierarchical setting, the UNIQUE optimal solution under sparsity exhibits absorption. There is no alternative decomposition at the same L0 that avoids absorption. The PW-MCC consistency metric (Song et al., 2025) shows that TopK SAEs achieve 0.80 consistency -- high enough that most features ARE consistent across runs, making the "it's just a different decomposition" argument weaker than it sounds. Furthermore, the unified SDL theory paper (arXiv:2512.05534) proves that spurious minima (including absorbed solutions) are true structural properties of the optimization problem, not artifacts of initialization.
- **Cherry-picking check**: The 0.80 PW-MCC means 20% of features are inconsistent -- and these could be exactly the features that exhibit absorption. This is worth testing but does not support the strong claim that absorption is "primarily" a decomposition artifact.
- **Confounding check**: Even if absorption varies across seeds, the phenomenon could be real for each decomposition. Analogy: different clustering algorithms produce different cluster assignments, but that doesn't mean clustering is meaningless -- it means the data has ambiguity in cluster boundaries. Similarly, absorption may be "real" in each decomposition while the specific absorbed features differ.
- **Actionability check**: If the experiment shows that the intersection of absorbed features across seeds is small, this IS actionable: it suggests ensemble methods (aggregating features across multiple SAE training runs) could eliminate absorption without architectural changes. This would be a highly practical contribution.
- **Verdict**: STRONG -- The hypothesis is testable, the experiments are feasible (multiple seed training + cross-seed absorption comparison), and both outcomes (small intersection = absorption is seed-dependent; large intersection = absorption is structural) are informative. The connection to ensemble methods provides a concrete actionable direction.

---

## Phase 4: Refinement

### Dropped

**Candidate A ("Absorption Mirage")** -- The toy model proof from Chanin et al. is too strong to dismiss absorption entirely as a measurement artifact. While the quantitative severity is likely overstated due to L0 confounds, the claim that >50% is artifact is not defensible. The kernel of truth (L0 misspecification inflates absorption measurements) is better incorporated as a methodological control within a broader study.

### Strengthened

**Candidate B (softened to "Absorption Heterogeneity")** -- Dropped the strong "wisdom" framing. Retained the core insight that not all absorption is equally harmful, and that the model's own use of a feature should predict whether its absorption is problematic. This becomes a supporting analysis rather than the main contribution.

**Candidate C (selected as front-runner)** -- Strengthened significantly:

1. **Precision on the claim**: Rather than "absorption is not real," the refined claim is: "The severity and practical impact of absorption is grossly overestimated because (a) measurements are confounded by L0 misspecification (a deconfounding control is needed), (b) the phenomenon is partially seed-dependent (ensemble methods can reduce it), and (c) the field evaluates absorption on artificial hierarchies that may not reflect the model's actual computational structure."

2. **Constructive proposal**: Instead of just critique, propose a three-pronged "Absorption Stress Test" protocol:
   - (i) Run absorption measurement at the "correct" L0 (using the Sparse but Wrong proxy metric) to deconfound hedging
   - (ii) Measure absorption across multiple random seeds and report the seed-invariant absorption rate (features absorbed in >80% of seeds)
   - (iii) Validate that absorbed features are actually used by the model via causal intervention, filtering out "phantom absorption" of features the model doesn't compute with

3. **Additional corroboration**: The CE-Bench paper (arXiv:2509.00691) shows that SAE interpretability scores increase with latent width, suggesting that given enough capacity, the SAE can represent all features -- meaning absorption at a given width may simply reflect insufficient capacity rather than a fundamental sparsity-hierarchy interaction. This is testable: does absorption disappear at sufficiently high width-to-feature ratios?

4. **Ensemble method**: If seed-dependent absorption is confirmed, propose a simple ensemble method: train N SAEs with different seeds, take the union of features, and use the intersection of activation patterns to identify robust features. Test whether this ensemble has lower absorption than any individual SAE. This is a training-free analysis if multiple pre-trained SAEs at the same configuration exist (which they do via the Gemma Scope suite -- different random seeds at the same width/L0 configuration).

---

## Phase 5: Final Proposal

### Title

**Rethinking Feature Absorption: Deconfounding Measurement, Quantifying Seed-Dependence, and Ensemble Methods for Robust SAE Features**

### Challenged Assumption

The conventional wisdom holds that feature absorption is a fundamental, mechanistic failure mode of SAEs caused by the interaction of hierarchical feature structure with sparsity optimization. The community treats absorption as a structural property of a given SAE configuration that must be mitigated through architectural innovation (Matryoshka, OrtSAE, ATM, etc.). The current proposal extends this view by proposing rate-distortion optimality as a theoretical framework.

We challenge the assumption that absorption, as currently measured, reflects a single coherent phenomenon that is both structural and practically important. We argue that the measured absorption rate is a composite of at least three distinct phenomena that have been conflated: (1) true sparsity-driven absorption (the phenomenon Chanin et al. describe), (2) feature hedging masquerading as absorption due to incorrect L0 settings, and (3) seed-dependent decomposition artifacts where the specific features absorbed differ across training runs. The practical impact of the remaining "true, structural" absorption after deconfounding these factors may be substantially smaller than current estimates suggest.

### Evidence

**For the conventional view:**
- Chanin et al. (2024) provide a toy model proof that absorption is a provable consequence of sparsity optimization under hierarchical features (arXiv:2409.14507)
- The unified SDL theory paper (arXiv:2512.05534) shows absorption arises from spurious minima in the biconvex optimization landscape
- Absorption is detected across every tested SAE, architecture, and model
- DeepMind's safety team found SAE probes fail catastrophically on safety tasks, with absorption as a contributing factor

**Against the conventional view (or more precisely, against the claim that measured absorption rates reflect the true severity):**
- "Sparse but Wrong" (arXiv:2508.16560) shows incorrect L0 produces feature hedging that would trigger the absorption metric. Most commonly used SAEs have L0 that is too low
- "Sanity Checks" (arXiv:2602.14111) shows random baselines match trained SAEs on interpretability, sparse probing, and causal editing -- if the "features" themselves are not reliably meaningful, measuring their absorption is questionable
- Feature inconsistency across seeds (Song et al., arXiv:2505.20254; PW-MCC = 0.80 for TopK) suggests 20% of features are seed-dependent, and these may overlap with absorbed features
- The absorption metric is probe-dependent and task-specific: no automatic detection method has been shown to work (confirmed by community attempts on LessWrong)
- "Use SAEs to Discover Unknown Concepts" (arXiv:2506.23845) argues SAEs should not be evaluated as classifiers, which is exactly what the absorption metric does
- Matryoshka SAEs reduce absorption but increase hedging (arXiv:2505.11756), suggesting a conservation law where "fixing" one artifact reveals another

### Hypothesis

**H1 (Deconfounding):** At least 30% of measured absorption on the first-letter spelling task is attributable to L0 misspecification (feature hedging) rather than true sparsity-driven absorption. Specifically, measuring absorption at the L0 identified by the "Sparse but Wrong" proxy metric will reduce the measured absorption rate by >=30% relative to the standard L0 setting.

Falsification: Absorption rate at the corrected L0 is within 10% of the rate at the standard L0 (i.e., L0 correction has negligible effect).

**H2 (Seed-dependence):** The set of features exhibiting absorption varies substantially across random seeds. The seed-invariant absorption rate (features absorbed in >=80% of seeds, controlling for configuration) is less than 50% of the absorption rate measured on any single seed.

Falsification: Seed-invariant absorption rate exceeds 80% of single-seed rate (i.e., absorption is highly consistent across seeds).

**H3 (Ensemble reduction):** An ensemble of K>=3 SAEs (same architecture, different seeds), where a feature is considered "active" if it fires in ANY member of the ensemble, achieves lower absorption rate than any individual SAE, without architectural changes.

Falsification: Ensemble absorption rate is not significantly lower than single-SAE rate (p > 0.05, paired test across hierarchy types).

**H4 (Computational relevance):** Absorption rates are higher for features that linear probes detect but that are not causally important for the model's next-token prediction (as measured by logit lens attribution < median) compared to features that are both probe-detectable and causally important.

Falsification: No significant difference in absorption rates between causally important and causally unimportant features (effect size d < 0.2).

### Method

All experiments are training-free analysis on pre-trained SAEs. Primary models: Gemma 2 2B (Gemma Scope SAEs) and GPT-2 Small (SAELens SAEs).

**Experiment 1: Deconfounding absorption from hedging (45 min)**
- Use the "Sparse but Wrong" proxy metric to identify the correct L0 for Gemma Scope SAEs at widths 16k and 65k on layer 12
- Measure absorption via the Chanin et al. metric at (a) the standard L0 and (b) the "correct" L0
- Report the difference and decompose absorbed cases into those that persist (true absorption) vs. those that disappear (hedging masquerading as absorption)

**Experiment 2: Cross-seed absorption consistency (45 min)**
- Use multiple Gemma Scope SAEs at the same configuration (if available with different seeds) OR train 5 SAEs on GPT-2 Small with SAELens at the same hyperparameters but different seeds (training is fast for GPT-2 small: ~10 min per SAE)
- For each seed, compute the set of absorbed feature pairs on the first-letter task
- Compute Jaccard similarity of absorbed sets across all seed pairs
- Report the seed-invariant absorption rate (features absorbed in >=4/5 seeds)

**Experiment 3: Ensemble absorption reduction (30 min)**
- For the 5 SAEs from Experiment 2, construct an ensemble: for each input token, a feature is considered "active" if it fires in any of the 5 SAEs
- Measure absorption rate for the ensemble representation vs. individual SAEs
- Report the absorption reduction and any cost in reconstruction quality

**Experiment 4: Computational relevance filter (45 min)**
- For features with known absorption status (from Chanin et al. first-letter task), use logit lens attribution to classify each feature as "computationally important" (high attribution to correct next-token prediction) vs. "merely representational" (low attribution)
- Compare absorption rates between the two groups
- If absorbed features are predominantly in the "merely representational" category, this supports the "absorption reflects computational irrelevance" hypothesis

**Experiment 5: Width scaling of absorption (30 min)**
- Measure absorption rate across Gemma Scope widths (16k, 32k, 65k, 131k) at matched L0
- Test whether absorption scales as O(1/width) -- if so, absorption is a capacity problem, not a fundamental sparsity-hierarchy interaction
- Extrapolate to predict the width at which absorption drops below noise floor

### Baselines

- **Standard absorption measurement** (Chanin et al., 2024) on the first-letter spelling task at default L0 settings -- the conventional baseline against which all our measurements are compared
- **No special architectures**: We deliberately avoid comparing Matryoshka, OrtSAE, etc. because our thesis is that the basic phenomenon is overestimated, not that it needs new architectures to fix
- **Shuffled-label control**: For all absorption measurements, include a permutation test with randomized feature labels to establish the noise floor

### Risk Assessment

**What if the conventional view turns out to be correct?**

If H1 and H2 are both falsified (L0 correction has negligible effect AND absorption is highly seed-consistent), this is strong evidence that absorption is indeed a structural, robust phenomenon worthy of dedicated study. In this case, the paper becomes a rigorous validation of the conventional view -- confirming that absorption is real, not confounded, and not an artifact of decomposition non-uniqueness. This is still a valuable contribution, as the field currently lacks such a thorough validation. We would pivot the framing to: "We set out to debunk feature absorption and instead confirmed it is every bit as real and concerning as claimed -- here are the rigorous controls that prove it."

If H3 is falsified (ensembles do not help), this rules out a simple practical mitigation and supports the theoretical work on architectural solutions. If H4 is falsified (no difference between computationally important and unimportant features), this is the worst outcome for our hypothesis and the strongest evidence for the conventional view -- absorption is indiscriminate and affects even the features the model relies on.

### Novelty Claim

The specific insight is that measured absorption rates conflate at least three distinct phenomena (sparsity-driven absorption, L0-driven hedging, and seed-dependent decomposition artifacts) that have never been disentangled. No prior work has (a) controlled for L0 misspecification when measuring absorption, (b) quantified seed-dependence of absorption, or (c) tested whether ensemble methods can reduce absorption without architectural changes. Even if our contrarian hypothesis is partially wrong, the deconfounding methodology is a contribution to the field's measurement standards.

### Connection to the Current Proposal

This contrarian perspective does not reject the current proposal ("When Sparsity Eats Its Young") but challenges its implicit assumption that absorption is well-measured and well-defined enough to build a theoretical framework on. We suggest adding the following experiments as controls within the current proposal:

1. **Mandatory L0 control**: Before any cross-domain absorption measurement (Phase C), first identify the "correct" L0 and report absorption at both standard and corrected L0
2. **Multi-seed absorption**: For at least one SAE configuration, report absorption across 3+ random seeds and include seed-invariant rate alongside single-seed rate
3. **Computational importance filter**: For the ASI metric (Phase D), validate that high-ASI pairs correspond to features the model actually uses, not just features that probes can detect

These controls would strengthen the proposal regardless of whether our contrarian hypothesis holds: if absorption survives deconfounding, the theory is validated more rigorously. If it does not, the paper has identified a major methodological issue that the community needs to address before building elaborate theoretical frameworks on top of potentially inflated measurements.
