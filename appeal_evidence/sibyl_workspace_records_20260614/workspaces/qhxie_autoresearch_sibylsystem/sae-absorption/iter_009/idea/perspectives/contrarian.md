# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Feature absorption is a well-defined, measurable phenomenon distinct from other SAE failure modes.**
   - Evidence challenging it: The canonical absorption metric (Chanin et al., arXiv:2409.14507) requires pre-specified probe directions and has only been validated on the first-letter spelling task. Chanin & Garriga-Alonso (arXiv:2508.16560) show that incorrect L0 causes feature hedging that manifests identically to absorption (features failing to fire where they should). No study has systematically disentangled how much of observed "absorption" is actually L0-induced hedging. The feature sensitivity paper (Tian et al., arXiv:2509.23717) reframes absorption as merely a special case of poor feature sensitivity, suggesting absorption may not be a distinct phenomenon at all.
   - Sources: Chanin et al. 2024 (arXiv:2409.14507), Chanin & Garriga-Alonso 2025 (arXiv:2508.16560), Tian et al. 2025 (arXiv:2509.23717)

2. **Assumption: The first-letter spelling task is a valid proxy for absorption in general.**
   - Evidence challenging it: The first-letter task is syntactic and extremely low-level. The "hierarchy" (letter membership > specific token) is an artifact of how English orthography interacts with tokenization, not a naturally occurring semantic hierarchy. "Resurrecting the Salmon" (arXiv:2508.09363) explicitly criticizes SAEBench's use of "word starts with x" as non-representative for domain-specific absorption evaluation. No one has shown that the 15-35% absorption rates on first-letter tasks generalize to any other feature hierarchy. The task may be pathologically easy for absorption to occur because letter-membership is deterministically predictable from token identity -- a property most real feature hierarchies do not share.
   - Sources: arXiv:2508.09363, SAEBench (arXiv:2503.09532)

3. **Assumption: Absorption is caused by the sparsity objective and can be "fixed" by better architectures.**
   - Evidence challenging it: Chanin et al. themselves note that absorption is an "effective strategy" from reconstruction perspective -- it encodes parent+child with +1 L0 instead of +2 L0. BatchTopK SAEs (which have NO L1 penalty) still show clear absorption because it improves reconstruction loss at a given k (arXiv:2412.06410, LessWrong post "Sparsity is the enemy of feature extraction"). This means absorption is not a bug of the L1 penalty but a fundamental consequence of ANY sparsity constraint. The "fix" would require worse reconstruction quality, which may be a worse tradeoff than living with absorption. The unified SDL theory (arXiv:2512.05534) calls absorption a consequence of piecewise biconvex optimization, suggesting it is a mathematical inevitability under standard formulations.
   - Sources: Chanin et al. 2024, Bussmann 2024 (arXiv:2412.06410), unified SDL theory (arXiv:2512.05534)

4. **Assumption: SAE features are real computational primitives that absorption disrupts.**
   - Evidence challenging it: Korznikov et al. 2026 (arXiv:2602.14111) show SAEs recover only 9% of true features in synthetic settings and random baselines match fully-trained SAEs on interpretability metrics. Heap et al. 2025 (arXiv:2501.17727) show SAE auto-interpretability scores cannot distinguish trained models from randomly initialized ones. Leask et al. (ICLR 2025, arXiv:2502.04878) show SAE features are neither canonical nor atomic. Song et al. (arXiv:2505.20254) show SAE features are inconsistent across training runs. Enkhbayar 2025 (arXiv:2511.11711) shows only 25% of top SAE features genuinely encode task-relevant information. If SAE features are not real computational primitives in the first place, then "absorption" may be studying artifacts of a fundamentally flawed decomposition method.
   - Sources: arXiv:2602.14111, arXiv:2501.17727, arXiv:2502.04878, arXiv:2505.20254, arXiv:2511.11711

5. **Assumption: Mitigating absorption will make SAEs useful for downstream tasks.**
   - Evidence challenging it: DeepMind (March 2025) found dense linear probes dramatically outperform SAE probes with 10-40% performance degradation even without specifically attributing the gap to absorption. Wu et al. (arXiv:2506.23845) show SAEs fail to outperform simple baselines (logistic regression, prompting, even bag-of-words) on concept detection and steering. Matryoshka SAEs reduce absorption to ~0.03 (from ~0.29) but the literature survey shows no evidence this absorption reduction translates into improved downstream task performance. The absorption rate improved by 10x, but has anyone shown this actually matters?
   - Sources: DeepMind Safety Research blog (2025), Wu et al. (arXiv:2506.23845), Bussmann et al. (arXiv:2503.17547)

6. **Assumption: The linear representation hypothesis holds sufficiently for SAEs to be the right tool.**
   - Evidence challenging it: Engels et al. (ICLR 2025, arXiv:2405.14860) discover irreducible multi-dimensional features (circular representations for days/months). A March 2026 paper (arXiv:2603.28744) shows linear probes and SAEs fail at compositional generalization, with dictionary learning identified as the binding constraint. Engels' 2025 MIT thesis provides "the first definitive evidence of causal, multi-dimensional features, thereby refuting the one-dimensional linear representation hypothesis." If the LRH is wrong for important classes of features, then SAEs may fundamentally misrepresent model computation, and absorption research is optimizing within a broken paradigm.
   - Sources: Engels et al. (arXiv:2405.14860), arXiv:2603.28744, Engels MIT thesis 2025

7. **Assumption: Studying absorption in a training-free setting (using pre-trained SAEs) is methodologically sound.**
   - Evidence challenging it: Chanin & Garriga-Alonso (arXiv:2508.16560) show most open-source SAEs have L0 that is too low, meaning they are trained with systematically wrong hyperparameters. If the SAEs we are analyzing were trained incorrectly, any absorption measurements on them conflate true absorption with training artifacts. Domain-specific SAEs (arXiv:2508.09363) show broad-domain training produces fundamentally different (worse) latents than domain-focused training. The "training-free" constraint means we are studying the failure modes of potentially misconfigured SAEs, not the inherent limitations of the SAE paradigm.
   - Sources: arXiv:2508.16560, arXiv:2508.09363

8. **Assumption: Absorption is one of the most important open problems in SAE interpretability.**
   - Evidence challenging it: Neel Nanda (September 2025) stated "the most ambitious vision of mechanistic interpretability I once dreamed of is probably dead." DeepMind deprioritized SAE research entirely. The field is fragmenting: Anthropic uses attribution graphs (not raw SAE features), alternatives like transcoders and chain-of-thought monitoring are emerging, and the fundamental question of whether "true features" even exist remains open. Studying absorption may be optimizing deck chairs on the Titanic if SAEs themselves are being abandoned by major labs.
   - Sources: DeepMind Safety Research blog (2025), Neel Nanda's September 2025 update, mechanistic interpretability 2026 status report

### Landscape of Doubt

The landscape reveals a striking paradox: the SAE community has invested enormous effort studying feature absorption (producing Matryoshka SAEs, OrtSAE, ATM, KronSAE, masked regularization), yet the foundational premises of this research program are under severe strain:

1. **The measurement problem**: The absorption metric requires knowing which features to look for (probe directions), making it fundamentally circular for open-ended interpretability. You can only measure absorption of features you already know about.

2. **The generalization problem**: All absorption evidence comes from one extremely narrow task (first-letter spelling). The 15-35% absorption rate has never been replicated on any other feature hierarchy. The community has built an entire mitigation literature around a single benchmark.

3. **The relevance problem**: Even if absorption is reduced to near-zero (as ATM claims with 0.0068), no one has demonstrated this translates into improved downstream performance on the tasks that actually matter (safety, circuit discovery, concept steering).

4. **The paradigm problem**: If SAE features are not canonical (ICLR 2025), not consistent across runs (Song et al. 2025), not distinguishable from random baselines on auto-interpretability (Heap et al. 2025), and only 25% genuinely task-relevant (Enkhbayar 2025), then absorption research is studying failure modes of a tool whose basic validity is in question.

5. **The inevitability problem**: Absorption is mathematically optimal under any sparsity constraint. "Fixing" it requires either worse reconstruction or abandoning sparsity -- either of which undermines the core value proposition of SAEs.

---

## Phase 2: Initial Candidates

### Candidate A: Absorption Is a Measurement Artifact, Not a Real Phenomenon

- **Challenged assumption**: Feature absorption is a distinct, well-defined failure mode of SAEs that occurs at 15-35% rates across SAE configurations.
- **Evidence against**: (1) The absorption metric conflates at least three distinct phenomena: true hierarchy-driven absorption, L0-induced hedging, and feature inconsistency across runs. (2) The metric requires known probe directions, creating a selection bias toward features where hierarchical structure is obvious. (3) The single benchmark task (first-letter spelling) has unique properties (deterministic parent-child co-occurrence) that may inflate absorption rates compared to typical feature hierarchies. (4) SAEBench's modified metric detects "absorbing latents" based on cosine similarity thresholds, where threshold choices are arbitrary and affect rates dramatically.
- **Contrarian hypothesis**: Much of what is measured as "absorption" is actually (a) incorrect L0 causing hedging/mixing, (b) features that were never reliably learned in the first place (only 25% of SAE features are task-relevant per knockoff analysis), or (c) an artifact of the first-letter task's deterministic co-occurrence structure. True hierarchy-driven absorption may be far rarer than reported.
- **Exploitation plan**: Systematically decompose measured absorption rates into component causes using controlled experiments: vary L0 to separate hedging from true absorption, compare absorption rates across tasks with different co-occurrence statistics (deterministic vs. probabilistic hierarchies), and use knockoff-based FDR control to exclude non-genuine features before measuring absorption.
- **Novelty estimate**: 8/10

### Candidate B: Absorption Is Optimal Behavior, Not a Bug

- **Challenged assumption**: Feature absorption is a problem that needs to be fixed for SAEs to be useful.
- **Evidence against**: (1) Chanin et al. themselves note absorption is an "effective strategy" saving +1 L0 per parent-child pair. (2) BatchTopK SAEs absorb even without L1 penalty because it improves reconstruction. (3) No study has shown that reducing absorption improves downstream task performance. (4) DeepMind's negative results found 10-40% degradation with SAE-reconstructed activations, but this gap persists whether absorption is high or low. (5) The human visual system exhibits analogous "absorption" (categorical perception absorbs fine-grained distinctions), and this is considered functional, not pathological.
- **Contrarian hypothesis**: Absorption is the SAE correctly learning that hierarchical features are informationally redundant in context. The parent feature "starts with S" adds zero information when "snake" is already active, because the model's computation does not need both. Absorption reflects the model's actual computational structure, not a failure to recover it. The real problem is that researchers expect SAEs to recover a feature ontology that does not match how the model actually computes.
- **Exploitation plan**: Test whether absorbed features are actually used by the model's computation (via causal intervention), or whether they are "decoration" -- features that linear probes can extract but the model does not rely on. If the model's computation already implicitly encodes the parent feature within the child feature, then absorption is faithful representation, not failure.
- **Novelty estimate**: 9/10

### Candidate C: The Entire Absorption Literature Is Studying the Wrong Problem

- **Challenged assumption**: Understanding and mitigating absorption within SAEs will advance mechanistic interpretability.
- **Evidence against**: (1) SAEs recover only 9% of true features in synthetic settings (Korznikov et al. 2026). (2) SAE features are non-canonical (Leask et al. ICLR 2025) and inconsistent across runs (Song et al. 2025). (3) Auto-interpretability scores cannot distinguish trained models from random ones (Heap et al. 2025). (4) DeepMind deprioritized SAE research entirely. (5) Neel Nanda: "the most ambitious vision of mechanistic interpretability I once dreamed of is probably dead." (6) 75% of SAE features are false positives per knockoff analysis (Enkhbayar 2025). (7) The linear representation hypothesis has been partially falsified (Engels et al. ICLR 2025).
- **Contrarian hypothesis**: The absorption literature is a case of "streetlight effect" -- the community studies absorption because they have a metric for it (the first-letter task), not because it is the most important failure mode. The real problems (non-canonicality, inconsistency, 91% missed features, multi-dimensional feature misrepresentation) are far more fundamental, and solving absorption would not meaningfully advance SAE utility. Research effort should shift to understanding WHEN and WHETHER SAE decomposition is valid at all.
- **Exploitation plan**: Conduct a systematic "importance ranking" of SAE failure modes by measuring how much each one contributes to the SAE-vs-baseline performance gap. Compare: (a) absorption-corrected SAEs vs. baseline, (b) consistency-corrected SAEs vs. baseline, (c) width-corrected SAEs (addressing 91% miss rate) vs. baseline. Show that absorption is a minor contributor compared to other failure modes.
- **Novelty estimate**: 7/10

---

## Phase 3: Self-Critique

### Against Candidate A: "Absorption Is a Measurement Artifact"

- **Steelman for the conventional view**: Chanin et al. provide a clean toy model proof that hierarchical features mathematically cause absorption under sparsity optimization. This is not a measurement artifact -- it is a theorem. The first-letter task was chosen specifically because ground truth is fully known, making false positives impossible (you know with certainty whether a token starts with a given letter). The metric is conservative by design: it only flags absorption when a single latent dominates and the primary latents are completely silent. Even if L0 confounding explains *some* measured absorption, the toy model proof guarantees that *some* true absorption occurs for any SAE with hierarchical features.
- **Cherry-picking check**: I am selectively emphasizing the L0 confound and measurement limitations while downplaying the theoretical proof. The proof is architecture-agnostic and applies to any sparsity-inducing method. However, the proof shows absorption *can* occur, not that 15-35% of features are actually absorbed. The quantitative rates may indeed be inflated by confounds.
- **Confounding check**: The L0 confound is real and documented. Chanin & Garriga-Alonso (2025) show most open-source SAEs have L0 that is too low. This is a genuine confound, not one I am fabricating. However, the toy model proof holds even at correct L0.
- **Actionability check**: If absorption rates are inflated by confounds, this has major implications: mitigation architectures may be solving a smaller problem than believed, and the field should invest in correct L0 selection rather than architectural changes. This is actionable.
- **Verdict**: MODERATE. The theoretical proof is solid, but the quantitative measurements are genuinely confounded. The claim should be refined: "absorption exists but its measured prevalence is likely inflated by L0 confounds and task-specific artifacts."

### Against Candidate B: "Absorption Is Optimal Behavior"

- **Steelman for the conventional view**: Even if absorption is informationally optimal, it creates real problems for practical applications. Circuit discovery requires knowing which features are active on which inputs; if a feature has systematic blind spots, circuits will be incomplete. Safety monitoring requires high recall; a "deception" feature that does not fire on deception-in-context-X is dangerous. Anthropic's circuit tracing work (Lindsey et al. 2025) demonstrates that reliable features enable powerful mechanistic understanding -- absorption directly undermines this. The human categorical perception analogy is misleading because SAEs are supposed to *decompose* representations, not reproduce their structure.
- **Cherry-picking check**: I am emphasizing the reconstruction optimality of absorption while ignoring concrete downstream failures. The DeepMind safety team specifically cited absorption as contributing to the SAE-vs-probe gap. However, they did not quantify absorption's specific contribution, and the gap persists even for Matryoshka SAEs with near-zero absorption.
- **Confounding check**: The claim "no study has shown reducing absorption improves downstream performance" is strong but hard to verify definitively. Matryoshka SAEs do improve on SAEBench metrics beyond just absorption (RAVEL, sparse probing, spurious correlation removal). These improvements may be partially driven by absorption reduction. However, the causal link is unclear.
- **Actionability check**: Testing whether the model's computation actually relies on parent features (vs. child features encoding parent information implicitly) is a clean, testable experiment. If confirmed, this would fundamentally reframe absorption from "failure mode" to "faithful representation."
- **Verdict**: STRONG. The core insight -- that absorption may reflect computational structure rather than failure -- is testable and would be highly impactful if validated. The practical objection (safety recall) is real but does not invalidate the theoretical point.

### Against Candidate C: "The Entire Absorption Literature Is Studying the Wrong Problem"

- **Steelman for the conventional view**: The absorption literature is not isolated -- it connects to core questions about SAE reliability for safety. Anthropic successfully used SAE features for circuit tracing (Lindsey et al. 2025), proving that when features are reliable, mechanistic understanding follows. Absorption is the most cleanly characterized failure mode, making it the natural entry point for systematic improvement. Even if other failure modes are larger, absorption is the one we can measure and potentially fix first. "Don't let the perfect be the enemy of the good."
- **Cherry-picking check**: I am presenting the most pessimistic evidence (9% feature recovery, DeepMind deprioritization, Nanda's pessimism) while ignoring the successes (Anthropic's circuit tracing, MIT Technology Review's "breakthrough technology 2026" designation, SAEs for concept discovery). The field is genuinely split, not uniformly negative.
- **Confounding check**: The 9% feature recovery number is from synthetic settings that may not capture real LLM structure. The knockoff analysis (25% genuine features) was done on Pythia-70M for sentiment, a specific and potentially unfavorable setting.
- **Actionability check**: A systematic importance ranking of failure modes would be valuable but extremely difficult to execute. Defining "contribution to the performance gap" requires controlling for interactions between failure modes, which may be intractable.
- **Verdict**: WEAK. While the underlying concern (SAEs may not work at all) is legitimate, the proposal to rank-order failure modes is methodologically challenging, and dismissing absorption research wholesale ignores its connections to the broader reliability question.

---

## Phase 4: Refinement

### Dropped: Candidate C

Candidate C ("absorption literature is studying the wrong problem") did not survive the steelman test. The core claim -- that other failure modes matter more -- is plausible but (a) hard to test cleanly, (b) risks being merely destructive without offering a constructive alternative, and (c) ignores that absorption research serves as a concrete entry point into SAE reliability questions. The "streetlight effect" critique has merit as a framing concern but is insufficient for a standalone research contribution.

### Strengthened: Candidate B (Primary) -- "When Absorption Is Faithful"

The core contrarian insight deserves sharpening: **absorption is not a failure to recover features but a faithful representation of how the model actually uses features**. The precise claim:

- **Precise failure condition**: Absorption is only a *problem* when the parent feature is causally used independently of the child feature in the model's computation. If the model never uses "starts with S" independently when "snake" is active (because "snake" already routes through S-related circuits), then absorption is the SAE correctly reflecting that the parent feature's causal role is subsumed.

- **Testable prediction**: For feature pairs where absorption occurs, causal intervention (ablating the absorbed direction) will have minimal effect on model behavior, because the child feature already carries the parent information downstream. Conversely, for feature pairs where the parent has independent causal roles, absorption will cause measurable circuit-level failures.

- **Constructive proposal**: Instead of trying to eliminate absorption universally, develop a *diagnostic* that distinguishes "benign absorption" (parent feature is computationally redundant in context) from "pathological absorption" (parent feature has independent causal effects that are lost). This would be far more useful than blanket absorption reduction.

**Additional corroboration**:
- The balanced Matryoshka SAE finding (Chanin et al. 2025) that absorption and hedging trade off against each other suggests there is no "free lunch" -- reducing absorption mechanically increases hedging. This is consistent with the hypothesis that some absorption is structurally necessary.
- Anthropic's circuit tracing work shows that what matters is whether features enable correct circuit identification, not whether features have perfect recall. A feature with 85% recall due to benign absorption may be perfectly adequate for circuit discovery if the 15% absorbed cases are ones where the circuit does not use that feature.

### Strengthened: Candidate A (Secondary) -- "Decomposing the Absorption Rate"

The confounding critique is refined into a constructive proposal:

- **Precise claim**: The reported 15-35% absorption rate is an upper bound that conflates at least three distinct mechanisms: (1) true hierarchy-driven absorption (mathematically proven), (2) L0-induced hedging (empirically documented), and (3) feature non-existence (the SAE never learned the parent feature reliably). Decomposing this rate would provide the first accurate estimate of how much absorption is "real."
- **Method**: Use Chanin & Garriga-Alonso's L0 proxy metric to identify SAEs at approximately correct L0. Measure absorption only on these SAEs. Compare with absorption measured at systematically wrong L0 (too low, too high). The difference estimates the L0-confounding component. Additionally, use the knockoff-based FDR method (Enkhbayar 2025) to exclude non-genuine features before computing absorption rates.

### Selected Front-Runner: Candidate B

**Title**: "When Absorption Is Faithful: Distinguishing Computational Redundancy from Feature Recovery Failure in Sparse Autoencoders"

The proposal reframes absorption from a universal problem to a sometimes-problem, provides a diagnostic framework, and generates concrete testable predictions. This is constructive (not just critical), grounded in theory (the reconstruction optimality argument), and experimentally feasible (causal interventions on pre-trained SAEs).

---

## Phase 5: Final Proposal

### Title
**Rethinking Feature Absorption: When SAEs Faithfully Represent Computational Redundancy**

### Challenged Assumption
The SAE community assumes that feature absorption -- where a parent feature fails to fire because its information is absorbed into a co-occurring child feature -- is uniformly pathological and should be minimized. This assumption drives a substantial mitigation literature (Matryoshka SAEs, OrtSAE, ATM, KronSAE, masked regularization) that aims to reduce absorption rates toward zero.

### Evidence: Both For and Against the Assumption

**Evidence that absorption is problematic:**
- Chanin et al. (arXiv:2409.14507) provide a clean toy model proof that absorption occurs under sparsity optimization with hierarchical features
- DeepMind cited feature absorption among reasons for SAE underperformance on safety-relevant downstream tasks
- Absorption creates systematic false negatives that undermine the use of SAE features as reliable classifiers
- Safety applications (deception detection) require high recall; absorption directly reduces recall

**Evidence that absorption may be benign or even faithful:**
- Absorption is an "effective strategy" that encodes parent+child with +1 L0 instead of +2 L0 (Chanin et al.)
- BatchTopK SAEs absorb even without L1 penalty because it improves reconstruction loss -- suggesting absorption reflects genuine informational structure
- No study has demonstrated that reducing absorption (e.g., Matryoshka SAE's 10x improvement) translates into improved downstream task performance
- The balanced Matryoshka SAE finding shows absorption and hedging trade off, suggesting some absorption may be structurally necessary
- Linear probes can extract the "absorbed" information from the child feature's activation, suggesting the information is present but re-encoded, not lost
- The model may not independently use the parent feature when the child feature is active, making absorption a correct reflection of computational redundancy

### Hypothesis
Feature absorption is a mixture of two fundamentally different phenomena that current metrics conflate:

1. **Benign absorption**: The parent feature is computationally redundant in the presence of the child feature. The model's circuits do not independently use the parent direction when the child is active. The SAE correctly represents this by not wasting a latent on redundant information.

2. **Pathological absorption**: The parent feature has independent causal effects in the model's computation that are lost when absorbed. The SAE incorrectly sacrifices causally important information for sparsity.

The ratio of benign-to-pathological absorption varies by feature hierarchy type, model layer, and SAE configuration. Current mitigation efforts treat all absorption as pathological, which is wasteful and may introduce new failure modes (hedging).

### Method
**Training-free analysis using pre-trained SAEs from Gemma Scope and SAEBench.**

**Experiment 1: Causal Absorption Diagnostic (core contribution)**
For each identified absorption instance (using Chanin et al.'s metric):
1. Identify the absorbing child latent and the absorbed parent direction
2. Perform causal intervention: ablate the parent direction component from the child latent's decoder vector
3. Measure downstream effect on model behavior (logit change for relevant tokens)
4. If ablation has minimal effect: classify as benign absorption (parent information is redundant)
5. If ablation significantly degrades model performance: classify as pathological absorption

This produces the first decomposition of absorption into benign vs. pathological components.

**Experiment 2: Decomposing Absorption Rate Confounds**
1. Select SAEs at varying L0 levels on Gemma-2-2B (using SAEBench's pre-trained SAEs at 6 sparsity levels)
2. Use Chanin & Garriga-Alonso's proxy metric to identify SAEs near "correct" L0
3. Measure absorption rate at correct L0 vs. too-low L0 vs. too-high L0
4. Estimate: (a) L0-confounding component, (b) true hierarchy-driven absorption, (c) feature non-existence component (using knockoff FDR or similar quality filter)
5. Report decomposed absorption rates: how much of the 15-35% headline number survives after removing confounds?

**Experiment 3: Cross-Domain Absorption Character**
Extend beyond first-letter task to feature hierarchies with different co-occurrence structures:
- First-letter task (deterministic co-occurrence, baseline)
- Entity type hierarchy (city > country, probabilistic co-occurrence)
- Part-of-speech hierarchy (verb > tense, semi-deterministic)
- Semantic hierarchy (animal > species, variable co-occurrence)

For each domain, measure total absorption rate AND benign/pathological ratio. The contrarian prediction: absorption rates will be much lower in domains with probabilistic (rather than deterministic) co-occurrence, because the model's computation more often needs both parent and child features independently.

**Experiment 4: Do Mitigation Architectures Help Where It Matters?**
Compare standard SAE vs. Matryoshka SAE vs. OrtSAE on:
- Total absorption rate (existing metric)
- Pathological absorption rate (our new diagnostic)
- Downstream task performance (sparse probing, circuit discovery fidelity)

The contrarian prediction: Matryoshka SAEs dramatically reduce total absorption but achieve most of their downstream improvement by reducing *pathological* absorption specifically, while much of the "absorbed" features they recover are computationally redundant (benign).

### Experimental Plan
- **Models**: Gemma-2-2B (primary), GPT-2-small (secondary, for reproducibility)
- **SAEs**: Gemma Scope 16k and 65k SAEs (JumpReLU), SAEBench pre-trained SAEs (7 architectures x 3 widths x 6 sparsities on Gemma-2-2B layer 12), Matryoshka SAEs from Gemma Scope 2
- **Tools**: SAELens v6, TransformerLens, sae-spelling metric code (adapted), SAEBench evaluation framework
- **Compute**: All experiments training-free; causal interventions require forward passes only; target ~45 minutes per experiment task on single GPU
- **Key libraries**: Integrated gradients for attribution, linear probes for ground truth, TransformerLens hooks for causal interventions

### Baselines
- Standard absorption metric (Chanin et al.) on first-letter task as baseline measurement
- Matryoshka SAE as the current "best" absorption mitigation (absorption ~0.03)
- Dense linear probe performance as the ceiling for what information is extractable

The baselines are NOT strawmen: Matryoshka SAE is the ICML 2025 state-of-the-art, and Chanin et al.'s metric is the NeurIPS 2025 standard.

### Risk Assessment
**If the mainstream view turns out to be correct** (i.e., most absorption is pathological):
- The causal diagnostic framework is still valuable as a new evaluation tool that provides more actionable information than the binary "absorbed/not-absorbed" metric
- The L0 confound decomposition (Experiment 2) is independently useful regardless of whether absorption is mostly benign or pathological
- The cross-domain analysis (Experiment 3) fills a recognized gap (Gap 2 and Gap 6 in the literature survey) regardless of the benign/pathological ratio

**If our contrarian hypothesis is wrong**: We still contribute (1) a novel causal diagnostic framework for absorption, (2) the first decomposition of absorption rates into component causes, (3) the first cross-domain absorption characterization. These contributions stand even if the benign/pathological ratio is different from what we predict.

### Novelty Claim
The specific insight is that feature absorption has been treated as a monolithic failure mode, but it is actually a mixture of computationally faithful representation (benign) and genuine feature recovery failure (pathological). No prior work has:
1. Distinguished benign from pathological absorption using causal intervention
2. Decomposed measured absorption rates into confounding components (L0, hedging, non-existence)
3. Tested whether the model's computation actually requires the absorbed feature independently
4. Measured whether absorption-reduction architectures specifically reduce *pathological* absorption

This reframing converts absorption from "a problem to minimize" into "a signal to interpret" -- benign absorption tells us about the model's computational structure, while pathological absorption identifies genuine interpretability gaps.
