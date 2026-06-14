# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Feature absorption is a well-defined, distinct phenomenon with clear boundaries**
   - Evidence challenging it: Chanin & Garriga-Alonso (2025, arXiv:2508.16560) demonstrate that most open-source SAEs have L0 that is too low, causing feature hedging -- a phenomenon that manifests identically to absorption (features failing to fire where they should). The "Sparsity vs Reconstruction Tradeoff Illusion" (LessWrong, 2025) argues that the standard L0 framing is fundamentally wrong, and what we call "absorption" may be substantially contaminated by hedging artifacts. No study has systematically disentangled the two.
   - Sources: [Sparse but Wrong (arXiv:2508.16560)](https://arxiv.org/abs/2508.16560); [The Sparsity vs Reconstruction Tradeoff Illusion (LessWrong)](https://www.lesswrong.com/posts/RwWrkGnncSCqryrSZ/the-sparsity-vs-reconstruction-tradeoff-illusion); [Feature Hedging (arXiv:2505.11756)](https://arxiv.org/abs/2505.11756)

2. **Assumption: The first-letter spelling task is a valid proxy for measuring absorption in general**
   - Evidence challenging it: The entire canonical absorption measurement (Chanin et al., 2024) relies on a single, narrow, syntactic task where ground-truth hierarchy is artificially clean. The first-letter task has an unusually crisp parent-child structure (letter membership strictly implies token identity) that may not resemble the messy, probabilistic, overlapping hierarchies in semantic features. No evidence exists that absorption rates on first-letter tasks predict absorption rates on knowledge, reasoning, or safety-relevant features. This is essentially a one-task benchmark masquerading as a general phenomenon.
   - Sources: [A is for Absorption (arXiv:2409.14507)](https://arxiv.org/abs/2409.14507); [SAEBench (arXiv:2503.09532)](https://arxiv.org/abs/2503.09532)

3. **Assumption: Absorption is a problem worth solving within the SAE paradigm**
   - Evidence challenging it: DeepMind's safety team deprioritized SAE research entirely after finding dense linear probes dramatically outperform SAE probes on safety tasks (2025). Korznikov et al. (2026, arXiv:2602.14111) show SAEs recover only 9% of true features in synthetic settings, with random baselines matching trained SAEs on interpretability, sparse probing, and causal editing. If SAEs fundamentally cannot recover model features, then "fixing absorption" is optimizing a broken tool.
   - Sources: [DeepMind Negative Results (Medium, 2025)](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9); [Sanity Checks for SAEs (arXiv:2602.14111)](https://arxiv.org/abs/2602.14111)

4. **Assumption: Sparsity is compatible with faithful feature recovery**
   - Evidence challenging it: The LessWrong post "Sparsity is the enemy of feature extraction" (2025) proves analytically that sparsity always increases absorption when hierarchical features exist -- the derivative of sparsity-absorption is always negative. Cui et al. (arXiv:2506.15963) provide a closed-form theoretical analysis showing SAEs generally fail to recover ground truth features unless features are extremely sparse. This suggests absorption is not a bug but an inherent, mathematically unavoidable property of sparsity-based decomposition.
   - Sources: [Sparsity is the Enemy of Feature Extraction (LessWrong, 2025)](https://www.lesswrong.com/posts/RxtaNrynuzPf4JePE/sparsity-is-the-enemy-of-feature-extraction-ft-absorption); [On the Limits of SAEs (arXiv:2506.15963)](https://arxiv.org/abs/2506.15963)

5. **Assumption: SAE features are the right unit of analysis for mechanistic interpretability**
   - Evidence challenging it: Leask et al. (ICLR 2025, arXiv:2502.04878) demonstrate SAE features are neither canonical nor atomic. Engels et al. (ICLR 2025, arXiv:2405.14860) discover irreducible multi-dimensional features (circular representations for days/months) that SAEs fundamentally cannot represent as single latents. Transcoders (Paulo et al., 2025) produce significantly more interpretable features. The "strong feature hypothesis" may be wrong -- features may not decompose into neat one-dimensional linear directions at all.
   - Sources: [SAEs Do Not Find Canonical Units (arXiv:2502.04878)](https://arxiv.org/abs/2502.04878); [Not All Features Are Linear (arXiv:2405.14860)](https://arxiv.org/abs/2405.14860); [Transcoders Beat SAEs (arXiv:2501.18823)](https://arxiv.org/abs/2501.18823); [The Strong Feature Hypothesis Could Be Wrong (Alignment Forum)](https://www.alignmentforum.org/posts/tojtPCCRpKLSHBdpn/the-strong-feature-hypothesis-could-be-wrong)

6. **Assumption: Feature absorption can be measured without knowing the "true features" in advance**
   - Evidence challenging it: The canonical absorption metric requires pre-specified probe directions -- you must already know what feature you're looking for. This creates a fundamental circularity: you need an interpretable feature decomposition to measure whether your interpretable feature decomposition is broken. Feature sensitivity (Tian et al., arXiv:2509.23717) attempts to address this but still requires knowing which features to test. No truly unsupervised absorption detection method exists.
   - Sources: [Feature Sensitivity (arXiv:2509.23717)](https://arxiv.org/abs/2509.23717); [A is for Absorption (arXiv:2409.14507)](https://arxiv.org/abs/2409.14507)

7. **Assumption: Mitigating absorption will make SAEs practically useful**
   - Evidence challenging it: Even the best-performing absorption mitigation (ATM SAE, absorption score 0.0068) has not been shown to improve downstream task performance. DeepMind found that the gap between SAE probes and linear probes on safety tasks is not attributable solely to absorption -- SAEs have multiple compounding failure modes. The "discovery vs. action" position paper (arXiv:2506.23845) argues SAEs are useful for concept discovery but not for acting on known concepts, regardless of absorption.
   - Sources: [Use SAEs to Discover, Not to Act (arXiv:2506.23845)](https://arxiv.org/abs/2506.23845); [ATM SAE (arXiv:2510.08855)](https://arxiv.org/abs/2510.08855); [Interpretability without Actionability (arXiv:2603.18353)](https://arxiv.org/abs/2603.18353)

8. **Assumption: "Studying absorption" is the right level of granularity for a research contribution**
   - Evidence challenging it: Multiple recent papers suggest the real problem is much broader. Song et al. (2025) show features are inconsistent across runs. Korznikov et al. (2026) show features match random baselines. The unified SDL theory (arXiv:2512.05534) treats absorption as just one manifestation of a general identifiability failure. Focusing on absorption specifically may be like studying one symptom of a systemic disease.
   - Sources: [Feature Consistency (arXiv:2505.20254)](https://arxiv.org/abs/2505.20254); [Unified Theory SDL (arXiv:2512.05534)](https://arxiv.org/abs/2512.05534); [Sanity Checks (arXiv:2602.14111)](https://arxiv.org/abs/2602.14111)

### Landscape of Doubt

The landscape of SAE research in early 2026 reveals a field in crisis. The dominant narrative -- that SAEs decompose neural network activations into human-interpretable monosemantic features -- is under systematic attack from multiple directions:

**Theoretical collapse**: Two independent theoretical analyses (Cui et al., 2025; unified SDL theory, 2024) show SAEs generally fail to recover true features. The sparsity objective that defines SAEs is mathematically incompatible with faithful feature recovery under feature hierarchy.

**Empirical embarrassment**: Random baselines match trained SAEs on standard metrics (Korznikov et al., 2026). Linear probes dramatically outperform SAE-based probes on downstream tasks (DeepMind, 2025). Features are not consistent across training runs (Song et al., 2025). The gap between what SAEs promise and what they deliver is widening.

**Paradigm shift underway**: Transcoders and crosscoders produce more interpretable features without the absorption problem. Anthropic's attribution graphs bypass per-layer SAEs entirely. The field's most prominent researcher (Neel Nanda) has publicly stated the most ambitious vision of mechanistic interpretability "is probably dead."

**The absorption-specific paradox**: Feature absorption is simultaneously (a) theoretically proven to be an inherent consequence of sparsity, (b) only measured on one narrow task, (c) confounded with at least two other phenomena (hedging and incorrect L0), and (d) shown to be just one of many failure modes that collectively render SAEs unreliable. A project that "studies absorption" risks studying a well-characterized symptom of a tool that may not work at all.


## Phase 2: Initial Candidates

### Candidate A: Absorption Is Mostly an L0 Artifact -- Disentangling True Absorption from Feature Hedging

- **Challenged assumption**: Feature absorption, as currently measured, represents a distinct and well-defined failure mode caused specifically by hierarchical feature structure.
- **Evidence against**: Chanin & Garriga-Alonso (2025) show most open-source SAEs have L0 that is too low, causing feature hedging that manifests identically to absorption. The "Sparsity vs Reconstruction Tradeoff" is an illusion -- there is a single correct L0 for each distribution, and deviations produce incorrect features. No study has controlled for L0 correctness when measuring absorption. The 15-35% absorption rate reported by Chanin et al. (2024) was measured on SAEs with potentially incorrect L0, meaning an unknown fraction of that rate is actually hedging.
- **Contrarian hypothesis**: The majority (>50%) of what is currently measured as "absorption" in standard SAEs is actually L0-induced feature hedging. True hierarchical absorption, while theoretically real, accounts for a smaller fraction of observed feature failure than the community believes.
- **Exploitation plan**: (1) Use Chanin & Garriga-Alonso's proxy metric to find the correct L0 for Gemma Scope SAEs, (2) retrain or select SAEs at the correct L0, (3) re-measure absorption with the canonical metric, (4) quantify how much "absorption" disappears when L0 is corrected. If the reduction is >50%, this fundamentally reframes the problem and redirects mitigation efforts from architectural solutions toward simply choosing correct L0.
- **Novelty estimate**: 8/10

### Candidate B: The First-Letter Task Is a Misleading Proxy -- Absorption Severity Is Task-Dependent and Overestimated

- **Challenged assumption**: Absorption rates measured on the first-letter spelling task generalize to other feature hierarchies, and the severity of absorption is roughly constant across domains.
- **Evidence against**: The first-letter task has an unusually clean, deterministic hierarchy (token identity strictly implies first letter). Real-world semantic hierarchies are probabilistic and overlapping (a "city" is usually but not always associated with a "country"). No cross-domain validation exists. Domain-specific SAEs (trained on chat data vs. pretraining data) show dramatically different feature quality. The absorption metric requires known probe directions, biasing measurement toward features with clean hierarchical structure -- precisely the features most vulnerable to absorption.
- **Contrarian hypothesis**: Absorption is severe on the first-letter task because that task has an unusually sharp hierarchical structure. On fuzzier, real-world hierarchies (entity types, sentiment-topic, knowledge hierarchies), absorption rates will be significantly lower (<10%) because the probabilistic overlap reduces the sparsity incentive to merge parent-child features.
- **Exploitation plan**: (1) Design absorption measurements for 3-4 feature hierarchies with varying "sharpness" (deterministic to probabilistic), (2) measure absorption rates on each using the same SAEs, (3) correlate absorption severity with hierarchy sharpness (mutual information between parent and child feature). If absorption drops sharply on fuzzier hierarchies, this explains why DeepMind found SAEs useful for some tasks but not others, and reframes absorption as a task-dependent rather than universal problem.
- **Novelty estimate**: 7/10

### Candidate C: SAE Absorption Research Is Studying a Dead Paradigm -- Transcoders Already Solve the Problem

- **Challenged assumption**: Studying and mitigating absorption within the SAE framework is a productive research direction that will yield impactful results.
- **Evidence against**: Transcoders (Paulo et al., 2025) produce significantly more interpretable features and reduce absorption as a side effect, without needing specialized mitigation. Anthropic's attribution graphs (Lindsey et al., 2025) use cross-layer transcoders and have already produced the most impressive mechanistic interpretability results to date. DeepMind has deprioritized SAE research. Random baselines match SAEs on standard metrics. The field's leading researchers are pivoting away from SAEs. Neel Nanda publicly stated the ambitious MI vision "is probably dead."
- **Contrarian hypothesis**: The absorption problem does not need a solution within the SAE paradigm because the paradigm itself is being superseded. Instead of fixing SAEs, we should quantify exactly when and why transcoders avoid absorption, providing the theoretical understanding needed to guide the paradigm transition.
- **Exploitation plan**: (1) Systematically compare absorption rates between SAEs and transcoders on matched settings using the first-letter task and extended hierarchies, (2) identify the theoretical mechanism by which the transcoder's input-output mapping avoids absorption, (3) determine whether there exist feature hierarchies where even transcoders exhibit absorption-like failures. This produces a "why transcoders work" theory paper rather than a "how to fix SAEs" paper.
- **Novelty estimate**: 6/10


## Phase 3: Self-Critique

### Against Candidate A: Absorption Is Mostly an L0 Artifact

**Steelman the conventional view**: Chanin et al. (2024) prove absorption in a toy model where L0 is not a confound -- the theoretical result holds regardless of L0 setting. Their toy model has the correct L0, and absorption still occurs because it is optimal under the sparsity objective. The "Sparsity is the enemy" post (LessWrong, 2025) proves analytically that the sparsity-absorption derivative is always negative, meaning absorption increases monotonically with any sparsity incentive, even at "correct" L0. Furthermore, the Chanin et al. first-letter experiments use multiple SAEs with varying L0, and absorption is present across the entire range. This suggests L0 correction will reduce but not eliminate absorption.

**Cherry-picking check**: I may be overweighting the "Sparse but Wrong" paper's L0 findings. While that paper shows most SAEs have incorrect L0, it does not claim absorption is entirely an L0 artifact -- it focuses on hedging specifically. The absorption and hedging mechanisms are distinct: absorption requires hierarchical features (parent-child structure), while hedging requires correlated features (any correlation). They can co-occur but have different causal pathways.

**Confounding check**: There could be a third factor: SAE training instability. If SAEs with incorrect L0 also have other training pathologies (dead neurons, non-convergence), the observed "hedging" might actually be a training artifact rather than a stable phenomenon. The recovery after L0 correction might reflect improved training rather than elimination of hedging per se.

**Actionability check**: If confirmed, this would be highly actionable -- it would redirect mitigation efforts from expensive architectural changes (OrtSAE, Matryoshka) toward simply choosing the correct L0, which is much cheaper. However, if the effect size is small (<20%), the paper would have limited impact.

**Verdict**: MODERATE. The confound is real and worth quantifying, but the theoretical results showing absorption is inherent to sparsity (even at correct L0) limit how much of the observed absorption can be explained away as hedging. The most likely outcome is that L0 correction reduces absorption by 20-40%, not >50% as my contrarian hypothesis claims. This is still a valuable finding but less dramatic than hoped.

### Against Candidate B: First-Letter Task Is Misleading

**Steelman the conventional view**: The first-letter task was chosen precisely because it provides clean ground truth for measurement. The SAEBench absorption metric uses this task and has been validated across 200+ SAEs (Karvonen et al., 2025). While the hierarchy is crisp, Chanin et al. found absorption in every single SAE they tested, across multiple models (Gemma, Llama, Qwen) and configurations. The universality of the finding suggests it is not a task-specific artifact. Furthermore, the theoretical argument (sparsity incentivizes absorption whenever features co-occur hierarchically) applies to any hierarchy, not just deterministic ones. In fact, fuzzy hierarchies might exhibit *more* absorption in some cases because the SAE has more "room" to redistribute activations.

**Cherry-picking check**: I am selectively emphasizing the narrowness of the first-letter task while ignoring that it is the only task where absorption can be cleanly measured. If absorption is lower on fuzzier tasks, that could also be because the metric fails to detect it (the probe is less accurate on fuzzy hierarchies, reducing the false-negative detection rate). Lower measured absorption might reflect metric weakness, not less absorption.

**Confounding check**: Hierarchy "sharpness" is confounded with other properties. Deterministic hierarchies (first letter) tend to involve syntactic features, while probabilistic hierarchies (entity type) tend to involve semantic features. Differences in absorption rates might reflect syntactic vs. semantic feature representation rather than hierarchy sharpness per se.

**Actionability check**: Even if confirmed, this finding is somewhat "deflationary" -- showing the problem is smaller than thought is useful but less publishable than showing how to fix it. A pure "absorption is overestimated" paper might struggle at top venues unless paired with a constructive proposal.

**Verdict**: MODERATE. The measurement limitation is real, but the confound between metric sensitivity and true absorption rate makes this hard to resolve cleanly. The risk of concluding "absorption is lower" when actually "our metric is worse" is substantial. Needs careful experimental design with multiple independent metrics.

### Against Candidate C: Dead Paradigm

**Steelman the conventional view**: SAEs remain the dominant tool in the community. Gemma Scope has 400+ SAEs, SAEBench evaluates 200+ SAEs, and the vast majority of mechanistic interpretability papers still use SAEs. Anthropic's attribution graphs are impressive but use Claude 3.5 Haiku (not publicly accessible) with internal transcoders. Open-source transcoders are far less mature than SAEs. Anthropic themselves used SAE features in their biology paper -- if SAEs were truly dead, Anthropic would not be using them. Furthermore, the claim that transcoders "solve" absorption is unverified -- no systematic absorption measurement has been done on transcoders. They might have different but equally severe failure modes. DeepMind deprioritized SAE research for pragmatic reasons (SAE probes < linear probes on their specific task), not because they believe the paradigm is theoretically bankrupt.

**Cherry-picking check**: I am selectively citing pessimistic statements (Nanda's "ambitious MI is probably dead") while ignoring that the same researchers continue to publish SAE papers and the field was named a "breakthrough technology for 2026" by MIT Technology Review. The pessimism is about the most ambitious claims, not about SAEs being useless.

**Confounding check**: The success of transcoders at Anthropic may be confounded with their massive compute budget and access to frontier models. Transcoders might only be superior in very large-scale settings, while SAEs remain competitive for the academic-scale experiments this project targets.

**Actionability check**: A "transcoders vs. SAEs on absorption" comparison paper would be useful, but it essentially advocates abandoning the project's core focus (SAE absorption) in favor of a different research direction. This is a strategic pivot rather than a constructive contribution to the absorption literature.

**Verdict**: WEAK as a standalone research direction for this project. The "SAEs are dead" framing is too provocative and insufficiently supported. SAEs clearly have value for some tasks. However, the underlying insight -- that understanding *why* transcoders avoid absorption could inform SAE architecture improvements -- has merit and could be incorporated as one component of a broader study.


## Phase 4: Refinement

### Dropped

**Candidate C** (Dead Paradigm / Transcoders) is dropped. The steelman test revealed that the "SAEs are dead" framing is unsupported -- SAEs remain the community standard, Anthropic continues to use them, and transcoders have not been systematically evaluated for absorption. The contrarian angle of "stop studying SAEs" is provocative without being constructive.

### Strengthened Survivors

**Candidate A** (L0 Artifact) was refined to be more precise:

- **Sharpened claim**: Rather than claiming >50% of absorption is hedging, the refined claim is: "A significant and currently unmeasured fraction of observed absorption in standard SAEs is confounded with L0-induced hedging, and no existing study controls for this confound."
- **Constructive addition**: Instead of just showing the confound exists, propose a **decomposition methodology** that separates L0-induced hedging from true hierarchical absorption. This involves training SAEs at the empirically correct L0 (using Chanin & Garriga-Alonso's proxy metric), measuring residual absorption, and attributing the reduction to hedging.
- **Additional corroboration**: The feature hedging paper (Chanin et al., 2025, arXiv:2505.11756) explicitly notes that Matryoshka SAEs trade absorption for hedging -- confirming that the two phenomena interact and that "fixing" one can worsen the other. This supports the need for joint characterization.
- **Key search finding**: The "Sparsity vs Reconstruction Tradeoff Illusion" post argues there is exactly one correct L0, and any deviation produces wrong features. If this is correct, then the entire body of absorption research -- which measures absorption "across a range of L0 values" -- is confounded by design.

**Candidate B** (Task-Dependent Absorption) was refined to become an essential complement to Candidate A:

- **Sharpened claim**: "The canonical absorption metric is simultaneously too narrow (one task) and too lenient (does not control for L0), and these two limitations compound to produce unreliable absorption rate estimates."
- **Constructive addition**: Design a **multi-task absorption benchmark** that spans hierarchy sharpness levels (deterministic: first-letter; semi-deterministic: city-country; probabilistic: entity-type; approximate: sentiment-topic). Use this benchmark to establish whether absorption severity actually varies with hierarchy structure.
- **Risk mitigation**: To address the concern that lower measured absorption on fuzzier tasks might reflect metric weakness rather than less absorption, use multiple independent metrics: (1) the canonical Chanin et al. probe-based metric, (2) KronSAE's mean absorption fraction, (3) a new sensitivity-based metric inspired by Tian et al. (2025). If all three metrics agree that absorption is lower, the conclusion is robust.

### Selected Front-Runner

**Candidates A and B are merged into a single unified proposal**: "Rethinking SAE Absorption: Disentangling the L0 Confound and Task-Dependent Artifact." This unified framing is stronger because:

1. The L0 confound (Candidate A) and the task-dependence question (Candidate B) are complementary facets of the same meta-question: "How much of reported absorption is a real, generalizable phenomenon, and how much is an artifact of how we measure it?"
2. The merged proposal produces a comprehensive recharacterization of absorption that the community needs before investing further in mitigation.
3. It is contrarian but constructive: it does not claim absorption is fake, but rather that our current understanding is systematically biased by measurement choices.


## Phase 5: Final Proposal

### Title

**When Absorption Shrinks: Disentangling L0 Confounds and Task Artifacts in SAE Feature Absorption**

### Challenged Assumption

The community assumes that the 15-35% absorption rate reported by Chanin et al. (2024) across Gemma Scope SAEs represents the true prevalence of hierarchical feature absorption -- a phenomenon where child features absorb parent features due to the sparsity incentive. This rate is treated as a stable, generalizable property of SAEs that motivates architectural solutions (Matryoshka, OrtSAE, ATM, masked regularization).

We challenge this assumption on two grounds:
1. **L0 confound**: The reported absorption rates were measured on SAEs with uncorrected (likely too low) L0, meaning an unknown fraction of "absorption" is actually L0-induced feature hedging -- a distinct phenomenon with different causes and different solutions.
2. **Task artifact**: All absorption measurements use a single task (first-letter spelling) with an unusually deterministic hierarchy, providing no evidence that absorption rates generalize to the fuzzier hierarchies found in semantic, knowledge, or safety-relevant features.

### Evidence

**For the conventional view (absorption is real and severe):**
- Chanin et al. (2024) prove absorption exists in toy models under correct theoretical conditions
- Absorption is found in every tested SAE across Gemma, Llama, and Qwen architectures
- The analytical result that sparsity-absorption derivative is always negative (LessWrong, 2025) means any sparsity incentive produces some absorption
- Matryoshka SAEs and OrtSAE demonstrate that architectural changes can reduce measured absorption, suggesting it is a real phenomenon amenable to intervention

**Against the conventional view (absorption is overestimated and confounded):**
- Most open-source SAEs have L0 that is too low (Chanin & Garriga-Alonso, 2025), causing hedging that manifests as features failing to fire where they should -- identical to absorption's observable signature
- No absorption study controls for L0 correctness. The "sparsity-reconstruction tradeoff" framing (measuring across L0 values) is itself an illusion (Chanin & Garriga-Alonso, 2025) -- only one L0 value is correct
- The first-letter task has a uniquely clean hierarchy where the sparsity incentive to absorb is maximal. Probabilistic hierarchies provide weaker incentive because the parent-child co-occurrence is not deterministic
- SAEs recover only 9% of true features even in synthetic settings (Korznikov et al., 2026), suggesting the features being "absorbed" may not be well-recovered in the first place
- DeepMind's safety task failures (2025) are attributed loosely to "absorption and other issues" but the specific causal contribution of absorption vs. other failure modes has never been quantified

### Hypothesis

**Precise claim**: When SAEs are operated at empirically correct L0 and evaluated across hierarchies of varying sharpness, the measured absorption rate will be substantially lower (by 30-60%) than the canonical 15-35% range. Specifically:
- At correct L0, the reduction will be 15-30% (absolute), attributable to the elimination of hedging-as-absorption confound
- On probabilistic hierarchies (e.g., city-country, entity-type), the residual absorption rate after L0 correction will be <10%, compared to 15-35% on the first-letter task
- The interaction will be multiplicative: L0 correction + fuzzy hierarchy together will reduce measured absorption to <5% for probabilistic feature hierarchies

### Method

**Phase 1: L0 Confound Quantification (training-free analysis)**
1. Select 10-15 Gemma Scope SAEs (varying width, layer) that overlap with Chanin et al.'s absorption measurements
2. Apply the "correct L0" proxy metric from Chanin & Garriga-Alonso (2025) to estimate the empirically correct L0 for each SAE
3. Compare the nominal L0 of each SAE to its correct L0 -- quantify the L0 gap
4. For SAEs where correct L0 is achievable by adjusting the activation threshold (JumpReLU allows post-hoc threshold adjustment), measure absorption at both nominal and corrected L0
5. Compute: Absorption_hedging_fraction = (Absorption_nominal - Absorption_corrected) / Absorption_nominal

**Phase 2: Multi-Task Absorption Benchmark**
1. Design 4 feature hierarchy tasks with varying sharpness:
   - **Deterministic**: First-letter spelling (baseline, replicating Chanin et al.)
   - **Semi-deterministic**: City to country mapping (using Gemma 2B's knowledge; most cities map uniquely to a country, but some names are shared)
   - **Probabilistic**: Entity type hierarchy (e.g., "scientist" implies "person" but not vice versa; overlap is statistical not deterministic)
   - **Approximate**: Sentiment-topic association (e.g., "positive sentiment" loosely co-occurs with certain topics but the relationship is noisy)
2. For each task: train logistic regression probes, apply the canonical absorption metric, and measure absorption rate
3. Additionally apply KronSAE's mean absorption fraction metric and the sensitivity metric (Tian et al., 2025) as cross-validation
4. Compute hierarchy sharpness (mutual information between parent and child feature activation) and correlate with absorption rate

**Phase 3: Joint Analysis**
1. Cross the two factors: measure absorption at (nominal L0, corrected L0) x (4 hierarchy tasks) = 8 conditions per SAE
2. Run 2-way ANOVA or equivalent to decompose absorption variance into L0 effect, hierarchy effect, and interaction
3. Produce absorption scaling law: Absorption_rate = f(L0_gap, hierarchy_MI)

### Experimental Plan

All experiments are training-free, using existing pre-trained SAEs from Gemma Scope:
- **Model**: Gemma 2 2B (via Gemma Scope SAEs)
- **SAEs**: JumpReLU SAEs at widths 16k and 65k, layers 3, 8, 15, 20 (replicating Chanin et al. layer selection)
- **Software**: SAELens + TransformerLens + sae-spelling (absorption metric code)
- **Compute**: Each task x SAE combination takes ~10-20 minutes on a single GPU for probe training + absorption measurement
- **Total budget**: ~15 hours of single-GPU compute, well within constraints
- **Pilot**: First-letter absorption at nominal vs. corrected L0 on 3 SAEs (estimated 30-45 minutes)

### Baselines

1. **Canonical absorption rates** from Chanin et al. (2024) -- replicate exactly before modifying
2. **Standard L0 SAEs** (nominal L0) -- the "without correction" baseline
3. **Random baseline** -- compute the absorption rate for random probe directions to establish a chance level
4. **Dense linear probe performance** -- the achievable upper bound that SAE probes should approach if absorption were eliminated

No strawman baselines. The comparison is between the community's current understanding of absorption (measured at nominal L0, first-letter task only) and the refined understanding after controlling for confounds (corrected L0, multi-task).

### Risk Assessment

**What if the mainstream view is correct?**
- If L0 correction reduces absorption by <10%: This means the confound is minor, and the community's absorption estimates are roughly accurate. We report this as a null result, validating the canonical measurements. The multi-task component still provides value by establishing generalizability.
- If absorption rates are equally high across all hierarchy types: This means absorption is truly universal and not task-dependent. Again, a null result that strengthens the case for architectural mitigation. The contribution becomes the first systematic cross-domain validation.
- If both effects are small: The paper becomes a "validation and extension" paper rather than a "recharacterization" paper -- less exciting but still publishable as the first multi-task absorption benchmark.

**Probability the mainstream view is broadly correct**: ~40%. The theoretical arguments for absorption being inherent to sparsity are strong. But even if absorption is real, the magnitude estimates are almost certainly biased by L0 confounds.

**Fallback**: If neither the L0 confound nor task-dependence produces large effects, pivot the paper to focus on the **interaction between absorption and hedging** as SAE width and L0 co-vary. This is itself an unstudied question (Gap 3 in the literature survey) and produces a useful empirical contribution regardless of effect sizes.

### Novelty Claim

The specific insight this work contributes: **The reported severity of SAE feature absorption is systematically inflated by two uncontrolled confounds -- incorrect L0 (causing hedging-as-absorption) and an unrepresentative evaluation task (with unusually sharp hierarchical structure).** No prior work has:
1. Measured absorption at empirically correct L0
2. Measured absorption on any task beyond first-letter spelling
3. Quantified the fraction of "absorption" that is actually hedging
4. Established whether absorption severity varies with hierarchy properties

This is the first comprehensive audit of absorption measurement validity, and it reframes the research priority from "how to fix absorption" to "how severe is absorption actually, and for which features does it matter?"
