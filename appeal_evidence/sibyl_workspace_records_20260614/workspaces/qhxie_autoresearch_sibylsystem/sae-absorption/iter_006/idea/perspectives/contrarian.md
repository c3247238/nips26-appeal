# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Feature absorption is a well-defined, isolable phenomenon distinct from other SAE failures.**
   - Evidence challenging it:
     - Chanin & Garriga-Alonso (2025, arXiv:2508.16560) show most open-source SAEs have incorrect L0, causing feature hedging that manifests identically to absorption (features failing to fire where expected). No study has formally decomposed observed "absorption" into hierarchy-driven absorption vs. L0-induced hedging vs. reconstruction error vs. metric sensitivity.
     - Korznikov et al. (2026, arXiv:2602.14111) demonstrate that random baselines match trained SAEs on standard evaluation metrics (interpretability 0.87 vs 0.90, sparse probing 0.69 vs 0.72, causal editing 0.73 vs 0.72), suggesting the baseline against which absorption is defined may itself be unreliable.
     - The LessWrong post "Looking for feature absorption automatically" (Aug 2025, Ehrenborg, Riggs, Nardo) attempted unsupervised absorption detection via causal-effect cosine similarity and **failed completely** -- the similarity distribution was non-bimodal, suggesting absorption may not be a discrete, detectable event but a continuous artifact of the sparsity-reconstruction tradeoff.

2. **Assumption: The first-letter spelling task provides a generalizable proxy for absorption in real interpretability applications.**
   - Evidence challenging it:
     - The first-letter task has uniquely clean properties: 26 non-overlapping, exhaustive classes; perfect ground truth; purely syntactic; extremely low hierarchy depth (exactly 2 levels). Real feature hierarchies (entity types, safety-relevant concepts like "deception") are messy, overlapping, multi-level, and contextual.
     - The "Resurrecting the Salmon" paper (arXiv:2508.09363) argues that SAEBench's use of "word starts with x" features is "not useful for evaluating domain-specific feature absorption," suggesting the benchmark is measuring something specific to its setup, not a general property.
     - Our own iteration 1 data showed cross-domain absorption rates of 0.1-1.7% using proxy metrics on knowledge hierarchies -- orders of magnitude lower than the 15-35% reported on first-letter tasks. This could mean either (a) the proxy metric is inadequate, or (b) first-letter absorption rates genuinely do not generalize.

3. **Assumption: Absorption is a pathology to be eliminated rather than a rational compression strategy.**
   - Evidence challenging it:
     - Chanin et al. (2024) themselves acknowledge: "any solution eliminating absorption will have worse L0 vs. variance-explained tradeoff (since absorption saves one L0 per parent-child pair)." Absorption is locally optimal for the sparsity-reconstruction objective.
     - The "Sparsity is the enemy of feature extraction" post (LessWrong, May 2025, 7vik, chanind, Garriga-alonso) proves analytically that absorption doesn't affect reconstruction and increases sparsity, showing it is a rational response to the optimization objective.
     - If the model itself uses hierarchical, overlapping representations (which Anthropic's "On the Biology of a Large Language Model" strongly suggests), then an SAE that absorbs parent into child features may actually be reflecting the model's own computational structure rather than distorting it. The model itself does not "fire" separate neurons for "starts with S" and "snake" -- it computes them compositionally.

4. **Assumption: The linear representation hypothesis holds sufficiently well that absorption (defined in terms of linear probe directions) is meaningful.**
   - Evidence challenging it:
     - Engels et al. (ICLR 2025, arXiv:2405.14860) demonstrate irreducible multi-dimensional features (circular representations for days/months) that fundamentally violate the one-dimensional linear assumption underlying both SAEs and the absorption metric.
     - Hindupur et al. (NeurIPS 2025, arXiv:2503.01822, "Projecting Assumptions") show that SAEs impose structural assumptions about concept geometry, and no existing SAE architecture simultaneously handles nonlinear separability and concept heterogeneity. The absorption metric presupposes that the "true" feature is a single linear direction -- but if concepts are nonlinearly separable or multi-dimensional, the entire framing is wrong.
     - Barin Pacela et al. (March 2026, arXiv:2603.28744, "Stop Probing, Start Coding") prove that under superposition, linearly separable concepts in latent space become nonlinearly separable in activation space, and the SAE encoder is fundamentally insufficient for accurate sparse inference (amortisation gap).

5. **Assumption: SAE features are stable enough objects for absorption to be a meaningful, reproducible measurement.**
   - Evidence challenging it:
     - Song et al. (2025, arXiv:2505.20254) show SAE features are inconsistent across training runs: TopK SAEs achieve only 0.80 PW-MCC, meaning 20% of features differ. If the features themselves are not reproducible, measuring their "absorption" may be measuring noise.
     - Leask et al. (ICLR 2025, arXiv:2502.04878) show SAE features are neither canonical nor atomic: meta-SAEs decompose single features into sub-features, and novel latents in larger SAEs capture information missed by smaller ones. If features are not atomic units, absorption (one "feature" eating another) may be a conceptual error -- there may be no stable parent feature to be absorbed in the first place.

6. **Assumption: Studying absorption in SAEs is the right level of analysis for improving mechanistic interpretability.**
   - Evidence challenging it:
     - DeepMind Safety Research (March 2025) deprioritized SAE research entirely, finding that dense linear probes dramatically outperform SAE probes on safety-relevant downstream tasks. This suggests the entire SAE paradigm, not just absorption, may be the wrong tool.
     - Paulo et al. (2025, arXiv:2501.18823) show transcoders beat SAEs on interpretability with Pareto improvement on reconstruction. Transcoders sidestep absorption by construction (mapping inputs to outputs rather than encoding/decoding activations).
     - The "Use SAEs to Discover Unknown Concepts, Not to Act on Known Concepts" position paper (arXiv:2506.23845, 2025) argues SAEs have a comparative advantage only for concept enumeration/discovery, not for acting on known concepts. Absorption is primarily a problem when acting on known concepts (detecting whether a feature fires). For discovery (finding what features exist), absorption may be irrelevant.

### Landscape of Doubt

The literature reveals a field in deep tension. Feature absorption was identified in 2024 as a clear, empirically demonstrable failure mode of SAEs. But the landscape in 2025-2026 has evolved dramatically:

**The ground beneath the absorption narrative has shifted.** The linear representation hypothesis that underpins the absorption metric has been partially falsified (multi-dimensional features, nonlinear separability). The features whose absorption we measure are not stable across training runs. Random baselines match trained SAEs on standard metrics. The only unsupervised absorption detection attempt failed. And the one evaluation domain used (first-letter spelling) is artificially clean in ways that real interpretability applications are not.

**The deeper question** is whether "feature absorption" is even the right frame. Perhaps what Chanin et al. observed is better understood as: (a) the inevitable consequence of compressing a continuous manifold of feature hierarchies into discrete, overcomplete sparse codes, (b) evidence that the one-dimensional linear assumption is wrong for hierarchical concepts, or (c) a measurement artifact of a metric that assumes probe directions define ground truth when probes themselves have systematic biases.

The community has been treating absorption as a bug to fix. The contrarian view is that absorption may be a symptom of much deeper problems with the SAE paradigm -- problems that architectural tweaks (Matryoshka, OrtSAE, masking) cannot fundamentally resolve.


## Phase 2: Initial Candidates

### Candidate A: "Feature Absorption Is Not Real" -- Decomposing Absorption into Confounds

- **Challenged assumption**: Feature absorption is a distinct, identifiable phenomenon caused by hierarchical features and sparsity optimization.
- **Evidence against**: (1) The only unsupervised detection attempt failed, suggesting absorption may not have a clear signal. (2) L0-induced hedging and reconstruction error both produce identical symptoms (features failing to fire). (3) The Chanin metric's thresholds (cosine > 0.025, magnitude gap >= 1.0) were never subjected to sensitivity analysis. (4) Iteration 1 found raw absorption rates of 92.3% on GPT-2 that collapsed to 19.2% with frequency matching, suggesting the metric is highly sensitive to confounds.
- **Contrarian hypothesis**: The majority of what is currently measured as "absorption" is actually a mixture of L0-induced hedging, metric sensitivity to threshold choices, frequency confounds, and inherent probe error -- not a discrete phenomenon caused by feature hierarchy.
- **Exploitation plan**: Systematically decompose measured absorption into component causes via controlled ablations (L0 correction, threshold sweeps, frequency matching, random baseline comparison), showing that after removing confounds, the residual "true absorption" is much smaller than reported.
- **Novelty estimate**: 7/10

### Candidate B: "Absorption Is the Model Speaking" -- Absorption as Evidence Against the One-Dimensional Linear Representation Hypothesis

- **Challenged assumption**: Absorption is a failure of SAEs to represent features that exist as clean linear directions in the model.
- **Evidence against**: (1) Multi-dimensional features exist causally (Engels et al., ICLR 2025). (2) Hierarchical concepts are nonlinearly separable under superposition (Barin Pacela et al., 2026). (3) The "Projecting Assumptions" paper (Hindupur et al., NeurIPS 2025) proves SAEs impose geometric assumptions that may not match concept geometry. (4) The model itself represents "starts with S" and "snake" compositionally, not as independent firing units.
- **Contrarian hypothesis**: What we call "absorption" is actually evidence that the underlying model representations for hierarchical concepts are not one-dimensional linear directions. The SAE is not "failing to recover" a linear feature -- the linear feature never existed. Instead, the model represents hierarchical concepts as regions in a multi-dimensional manifold, and the SAE's linear probe creates the illusion of a missing feature.
- **Exploitation plan**: Test whether concepts with high absorption rates in SAEs also show evidence of multi-dimensional or nonlinear representation (using the Engels et al. methodology for finding multi-dimensional features). If "absorbed" concepts are systematically the ones that violate the one-dimensional linear assumption, this reframes absorption from SAE failure to probe failure.
- **Novelty estimate**: 9/10

### Candidate C: "The Absorption Metric Measures Itself" -- A Critical Evaluation of Absorption Measurement

- **Challenged assumption**: The Chanin et al. absorption metric reliably measures a real phenomenon.
- **Evidence against**: (1) The metric requires known probe directions -- it can only find absorption for concepts the researcher already knows about. (2) The thresholds are arbitrary and never sensitivity-tested. (3) Absorption rates vary from 0.1% to 92.3% depending on confound controls (our iteration 1). (4) The metric is defined for exactly one task (first-letter spelling) and has not been validated on any other domain. (5) Random baselines performing comparably to trained SAEs (Korznikov et al., 2026) means the features against which absorption is measured may themselves be unreliable.
- **Contrarian hypothesis**: The absorption metric is measuring a combination of probe error, threshold sensitivity, and the inherent imprecision of projecting hierarchical concepts onto one-dimensional directions -- not a discrete phenomenon. The metric's apparent robustness across SAE architectures is because it is measuring properties of the task and the probe, not properties of the SAE.
- **Exploitation plan**: Conduct a rigorous meta-evaluation of the absorption metric itself: threshold sensitivity analysis (how much does the rate change across a 5x4 grid?), probe quality auditing (what is the probe error rate, and does it correlate with absorption?), random-feature baseline (what absorption rate do random directions in the SAE show?), and cross-task validation (does absorption rate for the same concept measured in different ways agree?).
- **Novelty estimate**: 8/10


## Phase 3: Self-Critique

### Against Candidate A: "Feature Absorption Is Not Real"

- **Steelman**: Chanin et al. (2024) provide a clean toy model proof that absorption is a mathematical consequence of sparsity optimization with hierarchical features. This is not an empirical observation -- it is a theorem. The fact that the specific measured rate is sensitive to confounds does not mean the underlying phenomenon is fake. Additionally, multiple architectural interventions (Matryoshka, OrtSAE, ATM, tied weights) independently reduce the measured phenomenon, which would be unlikely if it were purely a measurement artifact. The Matryoshka SAE specifically targets the hierarchical structure and achieves the best absorption scores on SAEBench, which is strong evidence that the phenomenon is real and caused by hierarchy.
- **Cherry-picking check**: I am emphasizing negative and critical results (random baselines, failed unsupervised detection, metric sensitivity) while downplaying the positive evidence: absorption is consistently present across all SAE architectures, models (Gemma 2, Llama 3.2, Qwen2, GPT-2), and widths. The convergence of independent groups finding the same phenomenon is strong evidence. The iteration 1 finding of 92.3% raw absorption was an implementation issue (lack of frequency matching), not evidence against absorption.
- **Confounding check**: It is true that L0 confounds could explain some measured absorption. But Chanin et al. show absorption even in toy models where L0 is correctly set, and the absorption metric specifically attributes false negatives to absorbing latents (not just any failure to fire). The confound story does not explain the integrated-gradients attribution step.
- **Actionability check**: If my claim is "absorption mostly isn't real," the practical consequence is "stop trying to fix it" -- which is not very actionable and potentially harmful if absorption actually matters for safety applications.
- **Verdict**: WEAK. The toy model proof is hard to argue against. The phenomenon is real; the question is about its magnitude in practice and whether the metric accurately captures it.

### Against Candidate B: "Absorption Is the Model Speaking"

- **Steelman**: The one-dimensional linear representation hypothesis is well-supported for many concepts. Burns et al. (2022) found truth directions, Kim et al. found concept directions, and Anthropic's scaling monosemanticity found thousands of interpretable linear features. Multi-dimensional features (Engels et al.) are the exception, not the rule -- found for periodic concepts (days, months) but not for general semantic features. The first-letter task ("starts with S") is almost certainly a linear feature; it's hard to argue it requires multi-dimensional representation.
- **Cherry-picking check**: I am selectively citing the multi-dimensional feature work (which found ~3 clear examples) while ignoring the thousands of linear features found by Anthropic, OpenAI, and others. The vast majority of SAE features do appear to be approximately one-dimensional.
- **Confounding check**: Even if hierarchical concepts have multi-dimensional structure, this doesn't mean absorption isn't happening. Both could be true: the model has multi-dimensional structure AND the SAE absorbs parent features into children. These are not mutually exclusive.
- **Actionability check**: This proposal leads to a testable prediction (absorbed concepts should show multi-dimensional structure). However, the first-letter task is the primary absorption benchmark, and "starts with S" is very likely a linear feature. If the strongest absorption is in the most linear features, my hypothesis is falsified.
- **Verdict**: MODERATE. The core insight (the LRH may be wrong for hierarchical concepts) is well-grounded, but the specific prediction (absorption correlates with multi-dimensionality) is testable and may fail for the canonical absorption examples. The more defensible version is weaker: absorption is worse than reported because the linear probe defines a false ground truth for some fraction of concepts.

### Against Candidate C: "The Absorption Metric Measures Itself"

- **Steelman**: The Chanin metric is not just checking whether a feature fires -- it uses integrated-gradients attribution to identify the specific absorbing latent and verifies that this latent has significant projection onto the probe direction. This is a multi-step verification that is harder to explain as pure probe artifact. Additionally, the metric has been independently reproduced (SAEBench), adopted by multiple papers (OrtSAE, KronSAE, ATM), and produces consistent rank-orderings of SAE architectures. If it were measuring noise, different groups would get inconsistent rankings.
- **Cherry-picking check**: I am emphasizing the 92.3% vs 19.2% discrepancy from iteration 1 as evidence of metric fragility, but this is an implementation issue (missing frequency matching), not evidence that the metric is wrong when properly applied. The Chanin metric as specified includes these controls.
- **Confounding check**: The metric could be consistently measuring something real but different from what is claimed. For instance, it could be measuring "the degree to which hierarchical features are represented compositionally vs. independently" -- which is a real property of the model-SAE interaction, just framed differently.
- **Actionability check**: A rigorous meta-evaluation of the metric is extremely actionable and valuable regardless of whether absorption is "real." The community needs to know whether the standard metric is robust, and this would be a foundational contribution.
- **Verdict**: STRONG. The meta-evaluation is independently valuable, the threshold sensitivity analysis fills a clear gap, and even if the metric turns out to be robust, demonstrating that is an important result. The combination of threshold sensitivity + random baseline + probe quality audit would be the first systematic validation of the absorption metric.


## Phase 4: Refinement

### Dropped: Candidate A
The "absorption is not real" claim is too strong. The toy model proof establishes that absorption is a mathematical property of sparsity optimization under hierarchy. Trying to argue it doesn't exist is fighting a theorem. The more nuanced version (the magnitude is overstated due to confounds) is subsumed by Candidate C.

### Strengthened: Candidate C (front-runner, reframed)
The meta-evaluation angle is the most defensible and valuable contrarian position. I strengthen it by:

1. **Making the critique more precise**: The claim is not "absorption doesn't exist" but "we don't actually know how much absorption occurs in practice because the standard metric has never been validated." The gap between the metric as described and the metric as deployed (our iteration 1 showed order-of-magnitude sensitivity to implementation details) is itself a publishable finding.

2. **Turning the critique constructive**: Rather than just attacking the metric, I propose a rigorous metric validation protocol and, crucially, a confound decomposition framework that separates true hierarchy-driven absorption from (a) L0-induced hedging, (b) probe error, (c) threshold sensitivity, and (d) reconstruction error. This decomposition would be the first of its kind.

3. **Independent corroboration**: The "random baselines match SAEs" result (Korznikov et al., 2026) provides independent evidence that standard SAE evaluation metrics may be measuring something other than what they claim. If trained SAEs don't beat random baselines on standard metrics, then measuring "absorption" in trained SAEs (which assumes the trained SAE has learned meaningful features) rests on shaky ground.

4. **Incorporating Candidate B elements**: The strongest element of Candidate B (the linear probe defines a potentially false ground truth for concepts that may not be purely linear) becomes a specific ablation: measure probe residual error and test whether high-residual concepts show higher apparent absorption.

### Selected front-runner: "Rethinking Absorption Measurement: A Confound Decomposition and Metric Validation Study"


## Phase 5: Final Proposal

### Title
When Is Absorption Real? Decomposing and Validating the Feature Absorption Metric for Sparse Autoencoders

### Challenged Assumption
The community assumes that the Chanin et al. (2024) absorption metric reliably measures a distinct phenomenon (feature absorption) caused by sparsity optimization under feature hierarchy. This metric, with its specific thresholds and procedures, is the basis for evaluating absorption across SAE architectures (SAEBench), motivating new architectures (Matryoshka, OrtSAE, ATM), and framing absorption as a key obstacle to SAE-based interpretability. Yet the metric has never been subjected to systematic validation: its thresholds have no published sensitivity analysis, its confounds have not been formally decomposed, and its generalizability beyond the first-letter task is untested.

### Evidence

**For the assumption (absorption is real and the metric works):**
- Chanin et al. (2024) provide a clean toy model proof that sparsity optimization under hierarchy necessarily produces absorption.
- The metric produces consistent rank-orderings across SAE architectures (SAEBench: Matryoshka best, JumpReLU worst).
- Multiple independent groups (OrtSAE, KronSAE, ATM) adopt the metric and report consistent findings.
- Architectural interventions specifically targeting hierarchy (Matryoshka) improve absorption scores.

**Against the assumption (the metric may be unreliable):**
- The only attempt at unsupervised absorption detection failed (LessWrong, Aug 2025) -- the signal is non-bimodal.
- Our iteration 1 found raw absorption rates ranging from 0.1% to 92.3% depending on implementation details (frequency matching).
- The thresholds (cosine > 0.025, magnitude gap >= 1.0) were set without published sensitivity analysis.
- L0-induced hedging (Chanin & Garriga-Alonso, 2025) produces identical symptoms to absorption but requires different mitigation.
- Random SAE baselines match trained SAEs on standard evaluation metrics (Korznikov et al., 2026), suggesting the features being measured may not be reliable reference points.
- Feature inconsistency across training runs (Song et al., 2025: PW-MCC ~0.80) means absorption measurements may vary by 20% due to initialization randomness alone.
- Probes themselves have systematic biases: many interpretable features have poor sensitivity (Tian et al., 2025), and SAE probes underperform dense linear probes (DeepMind, 2025; arXiv:2502.16681).

### Hypothesis
The measured absorption rate in practice is dominated by confounds (L0-induced hedging, probe error, threshold sensitivity, frequency artifacts) rather than by true hierarchy-driven absorption. After systematic confound decomposition, the residual "pure" absorption rate will be substantially lower than reported (15-35%), and the relative ranking of SAE architectures on absorption may change when confounds are properly controlled.

### Method
A training-free, analysis-only study using existing pre-trained SAEs (Gemma Scope on Gemma 2 2B) and the canonical absorption metric codebase (sae-spelling), structured as four experiments:

**Experiment 1: Threshold Sensitivity Analysis (est. 30 min)**
- Apply the Chanin metric on Gemma Scope SAEs (layer 12, width 16k/65k) for first-letter task.
- Sweep cosine threshold in {0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05} and magnitude gap in {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}.
- Compute: absorption rate at each (cosine, gap) pair; coefficient of variation across the grid; Spearman rank correlation of SAE architectures across threshold settings.
- **Success criterion**: If CV > 0.5 or rank correlation < 0.7 across reasonable threshold ranges, the metric is threshold-sensitive, and published absorption rates depend heavily on arbitrary choices.

**Experiment 2: Confound Decomposition (est. 45 min)**
- For each letter and SAE, decompose false-negative cases into:
  (a) **Probe error**: cases where the logistic regression probe itself is wrong (cross-validated probe error rate on held-out data).
  (b) **L0-mismatch hedging**: cases where the Chanin & Garriga-Alonso (2025) L0 diagnostic indicates the SAE operates at incorrect L0 for the relevant feature cluster.
  (c) **Frequency artifact**: cases attributable to low-frequency tokens (compare absorption rate on frequency-matched vs. unmatched token sets, following our iteration 1 finding).
  (d) **Residual absorption**: false negatives not attributable to (a), (b), or (c), with positive integrated-gradients attribution to an absorbing latent.
- Report the fraction of measured absorption attributable to each component.

**Experiment 3: Random Baseline Absorption (est. 20 min)**
- Compute "absorption rate" for random directions in the SAE decoder space (sample 100 random unit vectors as pseudo-probe directions).
- If random directions show non-trivial absorption rates (>5%), this indicates the metric partially measures properties of the SAE geometry rather than a real phenomenon.
- Also compute absorption for SAEs with randomly shuffled encoder weights (preserving decoder) to test whether the encoder's learned structure drives absorption.

**Experiment 4: Cross-Architecture Stability Under Controlled Confounds (est. 30 min)**
- After applying confound controls from Experiment 2, re-rank SAE architectures on residual absorption rate.
- Compare the ranking to the published SAEBench ranking (Matryoshka > p-anneal > L1 > TopK > JumpReLU).
- If rankings change substantially, the current "best architecture for absorption" claims are confound-driven.

### Experimental Plan
- **Model**: Gemma 2 2B (via TransformerLens + SAELens).
- **SAEs**: Gemma Scope SAEs at layers 10, 12, 20 (residual stream); widths 16k, 65k. Both JumpReLU (Gemma Scope) and any available Matryoshka/TopK variants.
- **Task**: First-letter spelling (canonical, to directly compare with published results).
- **Code base**: Fork sae-spelling (MIT license) for metric computation; SAELens for SAE loading and activation extraction; TransformerLens for model access.
- **Hardware**: Single GPU (Gemma 2 2B fits in 16GB). Target <= 1 hour per experiment, 2-3 hours total.
- **Seed**: Fixed (42) for any random operations.

### Baselines
- **Chanin et al. (2024) published absorption rates**: 15-35% on Gemma Scope 16k/65k. This is the primary comparison.
- **SAEBench architecture rankings**: The established ranking of architectures by absorption score.
- **Random direction baseline**: Establishes the floor for the absorption metric.
- **Probe-only baseline**: Absorption rate attributable purely to probe error.

### Risk Assessment
**If the mainstream view turns out to be correct** (i.e., the metric is robust, confounds are small, and absorption rates are stable across thresholds): This is still a valuable contribution. Demonstrating that the metric is robust and that confounds account for <10% of measured absorption would be the first systematic validation of the field's primary measurement tool, strengthening rather than undermining the absorption literature. The paper would then be framed as "validating and strengthening the absorption metric" rather than "debunking it."

**If absorption rates are indeed dominated by confounds**: This would be a high-impact finding that forces the community to re-evaluate published results and develop better-controlled metrics. It would motivate a second-generation absorption metric with built-in confound correction.

**Either outcome is publishable and valuable.** The field needs this meta-evaluation regardless of the result.

### Novelty Claim
The specific insight is that the absorption metric, despite widespread adoption, has never been validated as a measurement instrument. By treating the metric itself as the object of study (rather than using it uncritically to study SAE architectures), we provide the first:
- Threshold sensitivity analysis of the Chanin absorption metric
- Formal confound decomposition separating hierarchy-driven absorption from L0 hedging, probe error, and frequency artifacts
- Random baseline analysis establishing the metric's floor
- Cross-architecture re-evaluation under controlled confounds

This is the interpretability equivalent of a methods paper in empirical sciences: before trusting a measurement, you validate the instrument.
