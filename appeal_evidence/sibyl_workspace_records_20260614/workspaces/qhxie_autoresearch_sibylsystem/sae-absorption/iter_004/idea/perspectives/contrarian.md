# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption**: Feature absorption is primarily caused by the L1/sparsity penalty acting on hierarchical features.
   - Evidence challenging it: Research confirms that absorption emerges from the *interaction* between sparsity pressure and co-occurrence statistics in training data — not sparsity alone. Chanin et al. (2024) themselves note "the specific values of L1 penalty and feature co-occurrence rates/magnitudes will lead to different levels of absorption." The same phenomenon occurs with TopK SAEs (which have no L1 penalty) at low sparsity budgets (SAEBench finding). This means the conventional framing — "fix sparsity, fix absorption" — is incomplete.
   - Sources: Chanin et al. arXiv:2409.14507; SAEBench arXiv:2503.09532; Feature Hedging arXiv:2505.11756

2. **Assumption**: The canonical absorption metric (15–35% absorption rate on first-letter task) accurately characterizes the severity and scope of the phenomenon.
   - Evidence challenging it: The metric is explicitly acknowledged as a **conservative underestimate**. It only fires when a *single* SAE latent dominates ablation effect — missing partial absorption (where the main latent still fires but weakly) and distributed absorption (where multiple latents share absorbing responsibility). SAEBench extended the metric precisely because manual inspection revealed many absorption cases the original metric missed entirely. The metric also fails past layer 17 in Gemma 2 2B because it relies on causal ablation of early-layer source tokens.
   - Sources: SAEBench arXiv:2503.09532; Chanin et al. arXiv:2409.14507; LessWrong post "Looking for feature absorption automatically"

3. **Assumption**: Feature absorption is a well-defined, clean phenomenon distinct from other SAE failure modes (hedging, inconsistency, dark matter).
   - Evidence challenging it: The boundaries between failure modes blur on inspection. Absorption and hedging are caused by the same incentive (sparsity optimization under feature correlation) but manifest in opposite capacity regimes. SAEBench shows these metrics are somewhat correlated. Feature inconsistency across seeds (53% overlap for TopK SAEs; arXiv:2505.20254) suggests that some "absorbed" features in one training run may be "present but inconsistently located" rather than systematically suppressed. The nonconvexity of SAE training means some absorption patterns may be artifacts of local optima rather than structural optima.
   - Sources: Feature Consistency arXiv:2505.20254; Feature Hedging arXiv:2505.11756; EleutherAI blog on seed similarity

4. **Assumption**: Reducing absorption improves downstream interpretability and practical utility.
   - Evidence challenging it: DeepMind's safety team found that even without explicitly fixing absorption, dense linear probes dramatically outperform 1-sparse SAE probes on harmful intent detection — suggesting that the absorption problem, even if solved, may not close the critical gap with direct probing methods. Korznikov et al. (arXiv:2602.14111) showed SAEs recover only 9% of true synthetic features, and random baselines match trained SAEs on interpretability, sparse probing, and causal editing tasks. If absorption reduction leads to better absorption scores but similar practical failure on downstream tasks, it may be fixing the wrong thing.
   - Sources: DeepMind Medium post; Sanity Checks arXiv:2602.14111; "Are Sparse Autoencoders Useful?" arXiv:2502.16681

5. **Assumption**: Superposition is the correct theoretical frame for understanding why LLMs produce polysemantic neurons, and SAEs are the right tool to decompose it.
   - Evidence challenging it: A April 2026 paper (arXiv:2604.00443) shows 18–36% of SAE features blend word senses due to a lexical identity confound — neurons that activate for "bank" (financial) and "bank" (river) are labeled polysemantic but may simply be processing the shared word form before disambiguation. This lexical confound inflates measured polysemanticity and calls into question how much of what we measure as "absorption" is a genuine compression artifact vs. a linguistic phenomenon being mischaracterized by our metrics.
   - Sources: arXiv:2604.00443 (Polysemanticity or Polysemy?)

6. **Assumption**: Studying absorption on the first-letter spelling task is a valid proxy for understanding absorption of safety-relevant features (deception, harmful intent, bias).
   - Evidence challenging it: The first-letter task creates an artificial, known-ground-truth hierarchy (letter membership ⊃ specific token) that may be structurally unlike real safety-relevant feature hierarchies. In linguistic terms, the first-letter feature hierarchy is perfectly regular and almost entirely determined by surface form — no semantic ambiguity. Real safety features (e.g., "harmful intent") co-occur with contextual features in ways that are far more stochastic, context-dependent, and semantically complex. DeepMind found that the features required for harmful intent detection do not emerge from SAEs trained on pretraining data at all — the issue may be training data mismatch rather than absorption per se.
   - Sources: DeepMind Medium post; "Resurrecting the Salmon" arXiv:2508.09363; SAEBench (evaluation methodology section)

7. **Assumption**: Architectural solutions (Matryoshka SAE, OrtSAE, ATM SAE) that reduce the absorption metric genuinely solve the interpretability problem posed by absorption.
   - Evidence challenging it: Matryoshka SAEs reduce absorption but worsen feature hedging (Chanin & Dulka, arXiv:2505.11756). ATM SAE achieves best reported absorption score (0.0068 vs TopK 0.1402) but is evaluated only on Gemma-2-2B; generalization is unknown. Critically, none of the architecture papers demonstrate that their absorption reduction translates to downstream task improvements *beyond* what dense probes already achieve. The absorption metric and the downstream task metrics may be only loosely coupled.
   - Sources: Feature Hedging arXiv:2505.11756; ATM SAE arXiv:2510.08855; Sanity Checks arXiv:2602.14111

8. **Assumption**: We understand when and why absorption occurs — the toy model proof is sufficient theoretical grounding.
   - Evidence challenging it: Toy model results showing absorption don't straightforwardly transfer to real LLMs. Researchers tried solutions that worked in toy settings and found they still detected problems on Gemma-2-2b. "It could be possible that in LLMs, underlying parent/child features follow specific patterns that reduce the severity of absorption, but it's hard to say anything with certainty about true features in LLMs." The unified SDL theory (arXiv:2512.05534) provides a piecewise biconvex framework but feature anchoring (its proposed remedy) is only validated on synthetic benchmarks.
   - Sources: LessWrong "Broken Latents"; arXiv:2512.05534; Feature Hedging arXiv:2505.11756

### Landscape of Doubt

The absorption literature suffers from three intertwined problems:

**Problem 1: Metric-to-phenomenon validity gap.** The canonical absorption rate is a conservative underestimate of one specific type of failure, measured on one artificial task, at a subset of model layers. The community treats 15–35% as a characterization of the phenomenon's severity when it may dramatically understate it — or may measure something subtly different from what matters for downstream applications.

**Problem 2: Fixing the wrong variable.** The dominant research framing assumes: absorption is caused by sparsity optimization → design better training objectives/architectures → reduce absorption → better interpretability. But the causal chain breaks at step 3: multiple papers now show that absorption-reduced SAEs do not meaningfully outperform baselines on the downstream tasks that motivated SAE research. The bottleneck may be elsewhere (training data mismatch, the non-existence of clean "true features" in real LLMs, or the fundamental limits proven by Cui et al. arXiv:2506.15963).

**Problem 3: The measurement confound.** What we call "absorption" may actually be a mixture of: (a) genuine sparsity-pressure-driven absorption of hierarchical features, (b) lexical polysemy being misclassified as polysemanticity (arXiv:2604.00443), (c) training instability across seeds producing different local optima, and (d) the measurement artifact of using a conservative ablation threshold. No paper has attempted to decompose the absorption rate into these components.

---

## Phase 2: Initial Candidates

### Candidate A: The Absorption Metric Itself is the Problem — Not SAEs

**Challenged assumption**: The 15–35% absorption rate accurately and comprehensively measures a real failure mode that architectures should be optimized to minimize.

**Evidence against it**: The metric is conservative by design and misses partial and distributed absorption. SAEBench extended the metric after finding absorption patterns the original missed. The metric fails past layer 17 in Gemma 2 2B. Critically, researchers trying to automatically detect absorption via related methods (cosine similarity of ablation effect vectors) found poor correlation with subjective absorption judgments ("that is not currently the case"). The 15–35% number is widely cited but may be an artifact of measurement design rather than a reliable empirical floor.

**Contrarian hypothesis**: The reported absorption rate is a lower bound on an approximately known form of absorption but ignores at least three other types: (a) partial absorption (main feature fires weakly while absorber compensates), (b) distributed absorption (multiple latents jointly absorb), and (c) cross-layer absorption (information passed to later layers before absorption is measurable). A comprehensive measurement would show absorption affecting 60–90%+ of hierarchically related feature pairs, fundamentally changing our estimate of the problem's scope.

**Exploitation plan**: Develop a richer absorption measurement framework that directly quantifies partial and distributed absorption using the fractional approach from SAEBench, extended with cross-layer tracking. Apply it to the same Gemma Scope SAEs studied by Chanin et al. and show the true absorption rate is dramatically higher than reported. This reframes the entire field's understanding of how severe the problem actually is.

**Novelty estimate**: 7/10

---

### Candidate B: Absorption Is a Symptom of a Deeper Measurement Ontology Problem — "True Features" Don't Exist

**Challenged assumption**: Feature absorption is a failure of SAE training to recover pre-existing "true" monosemantic features that are ontologically present in the LLM.

**Evidence against it**: Multiple converging results: (1) SAEs recover only 9% of synthetic ground-truth features (arXiv:2602.14111); (2) different training runs converge to fundamentally different feature sets (53% overlap; arXiv:2505.20254); (3) random baselines match trained SAEs on interpretability metrics; (4) SAEs on randomly initialized models yield apparently interpretable features (Heap et al. 2025); (5) Polysemanticity or Polysemy (arXiv:2604.00443) shows 18–36% of SAE features may be linguistic artifacts rather than computational features. DeepMind itself noted "SAEs lack a ground truth of the 'true' features in language models."

**Contrarian hypothesis**: Feature absorption, as currently framed, presupposes that there exists a set of "true" monosemantic features that SAEs should be recovering. But LLMs may not have a unique, determinate set of features to be recovered. Instead, the feature decomposition is underdetermined — many different sparse dictionaries achieve equivalent reconstruction with equivalent behavior modification capacity. In this view, absorption is not a bug in recovering the correct features; it is a symptom of the conceptual error in assuming a unique correct decomposition exists. The right research question is not "how do we reduce absorption?" but "how do we characterize the equivalence class of valid decompositions?"

**Exploitation plan**: Formally characterize the absorption-permuted equivalence class of SAE solutions using the piecewise biconvex framework (arXiv:2512.05534). Show empirically (on GPT-2/Gemma-2B SAEs) that absorbed and non-absorbed decompositions have equivalent downstream behavioral predictions, supporting the view that absorption reflects decomposition ambiguity rather than error. Propose a new evaluation criterion based on downstream causal sufficiency rather than feature identity.

**Novelty estimate**: 8/10

---

### Candidate C: The Root Cause Is Training Data, Not Sparsity — Absorption Is a Corpus Statistics Artifact

**Challenged assumption**: Absorption is fundamentally caused by sparsity optimization under feature hierarchy, and changing the sparsity objective or architecture is the right fix.

**Evidence against it**: Absorption occurs with both L1 SAEs and TopK SAEs (no L1 penalty). The trigger condition is not sparsity per se but the co-occurrence of dense and sparse features in the training corpus. "Domain-specific SAEs" (arXiv:2508.09363) that train on task-relevant data rather than generic pretraining corpora show dramatically better feature recovery. DeepMind found that SAEs trained on pretraining data fail to encode task-relevant latents (like refusal behavior) entirely — a training data mismatch problem, not a sparsity problem. The "masked regularization" fix (arXiv:2604.06495) works by disrupting co-occurrence patterns during training, which is essentially addressing the corpus statistics issue.

**Contrarian hypothesis**: Feature absorption is not primarily a sparsity optimization problem — it is a corpus statistics problem. The frequency and co-occurrence structure of the training corpus determines which features get absorbed into which. This means the right solution is corpus engineering (targeted data selection, co-occurrence disruption) rather than architecture engineering. Moreover, absorption patterns in SAEs trained on generic web text systematically reflect the statistical structure of natural language co-occurrence rather than the computational structure of the LLM's representations.

**Exploitation plan**: Empirically test whether corpus statistics predict absorption patterns. Compute token co-occurrence statistics from the SAE training corpus and predict which first-letter features will be absorbed into which child features based on statistical co-occurrence. If corpus statistics are predictive, train SAEs on modified corpora where absorbing co-occurrences are disrupted (via strategic data resampling or augmentation) and show that absorption rates decrease. Compare to architecture-based fixes on the same base model. Runnable training-free analysis of existing Gemma Scope SAEs + OpenWebText corpus statistics.

**Novelty estimate**: 7.5/10

---

## Phase 3: Self-Critique

### Against Candidate A

**Steelman (conventional view)**: The Chanin et al. absorption metric is conservative by explicit design — and the SAEBench extension precisely addresses partial and distributed absorption. The 15–35% rate is not presented as a ceiling but as a measurement of a specific phenomenon type. The original authors note the metric is likely an underestimate. SAEBench's "absorption fraction" metric captures partial absorption. So the field is already aware of the limitation and has begun addressing it.

**Cherry-picking check**: The evidence that the metric underestimates absorption is itself cherry-picked from critiques. The metric has been validated through careful manual inspection of ablation experiments and confirmed causal relationships. SAEBench's extension shows the original metric is correlated with the more comprehensive approach.

**Confounding check**: Part of the gap between metric-measured absorption and "true" absorption may reflect genuine partial absorption that the metric misses vs. noise in the ablation procedure. Not all missed cases are genuine absorption.

**Actionability check**: Showing the metric underestimates absorption is valuable primarily as a methodological contribution, not a new research direction. It doesn't point toward a solution.

**Verdict**: MODERATE. The insight that absorption is systematically underestimated is valid and important, but the field is already aware of this. A paper purely about "the metric is worse than you thought" has limited novelty. More powerful as a component of a broader story.

---

### Against Candidate B

**Steelman (conventional view)**: The "true features don't exist" position is a philosophical claim that doesn't negate the practical value of absorption research. Even if the decomposition is underdetermined, some decompositions are more behaviorally useful than others. The question "which decompositions allow reliable causal intervention?" is empirically answerable. Absorption research is asking a well-posed question: why do SAE features have systematic false negatives?

**Cherry-picking check**: The evidence for underdetermination (seed inconsistency, random baselines) is strong, but ignores strong evidence the other way: SAEs reliably find features that are functionally meaningful, allow causal interventions, and generalize across models (e.g., universal features across GPT-2 variants). The "9% of features recovered" result is on synthetic data where ground truth is known — real LLMs don't have a canonical ground truth to compare against.

**Confounding check**: Seed inconsistency (53% overlap) could reflect that 47% of features are genuinely variant (due to local optima) — but it could also reflect that these features are genuinely present in the data but captured differently. Not all inconsistency implies non-existence.

**Actionability check**: "Features don't uniquely exist" is a hard philosophical claim that leads to "evaluate by downstream causal sufficiency" — but this is essentially what SAEBench already does with its downstream task metrics.

**Verdict**: MODERATE-WEAK. The philosophical challenge to the "true features" framing is important background but doesn't easily translate into a falsifiable research proposal stronger than existing work. Risk of being a "gotcha" paper without a constructive alternative.

---

### Against Candidate C

**Steelman (conventional view)**: Corpus statistics play a role in absorption, but the toy model proof shows that absorption would occur even in idealized settings where co-occurrence is perfectly regular and the SAE architecture is simple. The mathematical condition is clear: when dense feature A co-occurs with sparse feature B, the SAE has a sparsity incentive to fold A into B. This is a property of the training *objective*, not the data distribution per se. Masked regularization works precisely because it disrupts the signal that drives the sparsity-motivated absorption.

**Cherry-picking check**: The observation that TopK SAEs (no L1) also produce absorption is strong evidence that the phenomenon is not purely an L1 artifact. However, this is also consistent with the "corpus statistics" framing — any sparsity objective will be incentivized by co-occurrence in the data. The corpus framing and the objective framing are not mutually exclusive; they are dual perspectives.

**Confounding check**: If absorption is corpus statistics → predicts absorption from co-occurrence, we need to rule out that the co-occurrence prediction simply reflects the feature hierarchy (which is what the toy model says causes absorption anyway). The corpus statistics prediction may be circular — co-occurrence IS the feature hierarchy.

**Actionability check**: Strong. Corpus statistics analysis of existing SAEs is training-free, fast, and directly actionable. If co-occurrence predicts absorption patterns, this provides a principled basis for corpus engineering as a mitigation strategy complementary to architecture engineering. This has not been empirically tested.

**Verdict**: STRONG. The corpus statistics causal mechanism is plausible, testable with existing data (training-free), and leads to a constructive proposal that is genuinely novel.

---

## Phase 4: Refinement

**Dropped**: Candidate B in its strong form ("true features don't exist"). The philosophical challenge, while important, is too speculative for a focused experimental paper and risks being dismissed as hand-waving. It works better as a framing critique in the introduction.

**Strengthened Candidate A**: Rather than simply showing the metric underestimates absorption, fold this into a more constructive proposal: develop a **multi-type absorption taxonomy** that distinguishes full absorption, partial absorption, distributed absorption, and cross-layer absorption — and show empirically that each type has different causal structure and different implications for downstream task performance. This is more actionable than "the metric is bad" — it gives the community a richer vocabulary for what absorption actually is.

**Strengthened Candidate C**: After steelmanning, this is the front-runner. Refined hypothesis: **corpus co-occurrence statistics are predictively sufficient to explain which specific feature pairs exhibit absorption, even when controlling for feature hierarchy depth and SAE architecture.** This is testable: (1) extract token co-occurrence statistics from OpenWebText, (2) predict which first-letter features are absorbed into which child tokens, (3) compare to observed absorption patterns in Gemma Scope SAEs. If corpus statistics explain absorption patterns better than or as well as feature hierarchy depth, it shifts the causal story toward the data side and away from the objective side.

**Additional corroboration for Candidate C**: The masked regularization paper (arXiv:2604.06495) works by disrupting co-occurrence patterns during training — this is indirect evidence that co-occurrence is the proximal cause. Domain-specific SAEs (arXiv:2508.09363) show that training data selection dramatically changes which features appear, consistent with corpus statistics driving feature structure.

**Merged front-runner**: Combine A (richer absorption taxonomy + measurement) and C (corpus statistics causation). The proposal is: *absorption patterns are predictable from corpus co-occurrence statistics, but the existing metric only measures one type of absorption and therefore underestimates the true scope of a data-driven rather than architecture-driven phenomenon.*

---

## Phase 5: Final Proposal

**Title**: "When Did Absorption Happen? Corpus Co-Occurrence Statistics Predict and Explain Feature Absorption in Sparse Autoencoders"

*(Alternative constructive framing: "Data-Driven Feature Absorption: How Corpus Co-Occurrence Shapes SAE Failures")*

---

**Challenged Assumption**: Feature absorption in SAEs is primarily an architectural/optimization artifact caused by sparsity penalty incentives acting on feature hierarchies — and is therefore best addressed by changing the training objective or architecture.

**Evidence For the Conventional View**:
- Toy model proof (Chanin et al.) shows absorption occurs in minimal settings with hierarchical features and any sparsity objective
- Absorption is reduced by architectural changes that change the objective: Matryoshka SAE (hierarchy-aware loss), OrtSAE (orthogonality penalty), masked regularization (co-occurrence disruption during training)
- ATM SAE achieves near-zero absorption score (0.0068) by changing the training dynamics, supporting that the training procedure is the lever

**Evidence Against the Conventional View (or complicating it)**:
- Absorption occurs with both L1 and TopK objectives — the sparsity mechanism is not uniquely responsible
- The masked regularization fix works by disrupting *co-occurrence patterns*, not by reducing sparsity pressure per se — this is a corpus statistics intervention
- Domain-specific SAEs trained on task-relevant data show dramatically better feature recovery, suggesting training corpus composition determines feature quality
- The co-occurrence statistics of the training corpus should predict, for any sparsity objective, which feature pairs will experience absorption — this has never been directly tested
- The canonical absorption rate (15–35%) is a conservative underestimate that misses partial and distributed absorption types

---

**Hypothesis**: Corpus co-occurrence statistics are a **sufficient explanatory variable** for predicting which specific feature pairs exhibit absorption, and the pattern of absorption reflects the statistical structure of the training corpus more than the depth of the feature hierarchy. Specifically: for any two features A (dense/general) and B (sparse/specific), the absorption rate for A→B is predicted by the pointwise mutual information (PMI) of their co-occurring tokens in the training corpus, controlling for feature hierarchy depth.

**Secondary hypothesis**: The commonly reported 15–35% absorption rate substantially understates the true failure, because it measures only full absorption (complete suppression of the parent feature) while ignoring partial absorption and distributed absorption — which account for a large fraction of missed feature activations.

---

**Method**:

1. **Absorption taxonomy development**: Define and operationalize three absorption types using Gemma Scope SAEs on Gemma 2 2B:
   - Type I (Full): Main feature latent completely suppressed; single absorbing latent (existing metric)
   - Type II (Partial): Main feature latent fires but weakly (<50% of expected magnitude); absorbing latent partially compensates
   - Type III (Distributed): Main feature suppressed; multiple latents jointly absorb with no single dominant absorber

2. **Comprehensive measurement**: Apply the taxonomy to Gemma Scope 16k/65k SAEs on the first-letter task and on 2–3 additional feature hierarchies (entity type, syntactic role) using the sae-spelling codebase extended with fractional absorption scoring from SAEBench.

3. **Corpus statistics prediction**: Extract token co-occurrence statistics from OpenWebText (the SAE training corpus):
   - Compute PMI for all token pairs that activate the same SAE latent
   - For each first-letter feature hierarchy (letter L ⊃ token T), compute the co-occurrence strength of T with other L-bearing tokens in the training data
   - Fit a logistic regression predicting Type I/II/III absorption from: (a) co-occurrence PMI, (b) feature hierarchy depth, (c) SAE width, (d) L0

4. **Causal test**: Test whether high-PMI pairs predict absorption across multiple SAE configurations (different widths: 1k, 4k, 16k, 65k from Gemma Scope) and across two model families (Gemma 2B, GPT-2). If corpus statistics predict absorption cross-architecturally, it supports the data-driven framing.

5. **Ablation**: Compare absorption rates of features in high-PMI vs. low-PMI corpus regions (naturally occurring variation in co-occurrence) to confirm the causal direction.

---

**Experimental Plan**:

All experiments are training-free — using existing Gemma Scope SAEs (available on HuggingFace) and existing training corpora (OpenWebText). No GPU-intensive training required.

| Experiment | Time Estimate | What It Tests |
|---|---|---|
| Absorption taxonomy operationalization on Gemma Scope 16k | 30 min | Can we measure Type II and III absorption reliably? |
| Comprehensive absorption measurement (3 feature hierarchies × 4 SAE widths) | 60 min | What is the true absorption rate under comprehensive measurement? |
| Co-occurrence statistics extraction (OpenWebText) | 30 min | Extract PMI features for prediction |
| Corpus statistics prediction model | 30 min | Does PMI predict absorption beyond hierarchy depth? |
| Cross-model validation (GPT-2 SAEs) | 45 min | Does the corpus statistics finding generalize? |

Total: ~3.5 hours of compute, well within training-free analysis budget. Each sub-experiment can be run in <1 hour.

---

**Baselines**:
- Conventional absorption metric (Chanin et al.) — shows how much the comprehensive measurement changes the picture
- Feature hierarchy depth as predictor — compare corpus statistics PMI prediction against pure hierarchy-depth prediction
- SAE width as predictor — existing finding that wider SAEs have higher absorption (SAEBench) as a confound to control

---

**Risk Assessment**:

*If the corpus statistics prediction fails to outperform feature hierarchy depth*: This would support the conventional toy-model explanation. The paper still contributes the absorption taxonomy (Types I/II/III) and the comprehensive measurement showing higher true absorption rates. We would conclude absorption is primarily a hierarchy/objective phenomenon and the corpus statistics are a downstream proxy for hierarchy.

*If co-occurrence PMI is perfectly collinear with hierarchy depth*: These two explanations may be empirically indistinguishable because the hierarchy IS defined by co-occurrence structure. We would need a more nuanced test using non-hierarchical but high-PMI feature pairs as a control condition.

*If absorption types I/II/III don't cleanly separate*: The taxonomy may be a continuum rather than discrete types. We would report the fractional absorption distribution and still show the comprehensive rate exceeds the standard metric.

*Broader risk*: If downstream task performance doesn't improve even with comprehensive absorption measurement, this reinforces the DeepMind finding that absorption research may be measuring the wrong failure mode — a finding that is itself publishable and important.

---

**Novelty Claim**: The specific insight being established: **feature absorption in SAEs is predictably structured by the training corpus's co-occurrence statistics, not primarily by feature hierarchy topology or sparsity objective design**. This reframes the intervention target from "architecture/objective engineering" to "corpus engineering" — and provides, for the first time, a quantitative predictive model of which specific feature pairs will exhibit absorption in any SAE trained on a given corpus. Separately, establishing a comprehensive absorption taxonomy reveals that the reported 15–35% rate is a measurement artifact that understates the true failure scope by a factor estimated to be 2–5x.

---

## Summary of Contrarian Insight

The field has converged on "absorption is an objective problem → fix the objective." The deeper truth may be: **absorption is a data problem — it encodes what the training corpus treats as substitutable**. This means the correct diagnostic is corpus analysis, not architecture analysis; and the correct fix may be corpus curation, not architecture change. The bonus insight is that our measurement of absorption has been systematically too conservative, so the field may have been arguing about a 15% problem when the true failure rate is 40–80%.

These two claims together — "we've been measuring it wrong" and "we've been fixing the wrong thing" — form the core of a genuinely provocative but evidence-grounded contribution.
