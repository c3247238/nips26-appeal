# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Feature absorption is a well-defined, distinct phenomenon that deserves its own research program.**
   - Evidence challenging it: The canonical absorption metric (Chanin et al., 2024) relies on a manually-tuned cosine similarity threshold of 0.025 and an ablation effect gap threshold of 1.0, both chosen from "manual inspection of the data." SAEBench modified the metric to use a different detection criterion (latent contribution to probe direction) because the original metric fails in later layers. Tian et al. (2025) subsume absorption as a special case of the broader "feature sensitivity" concept. Feature hedging (Chanin & Dulka, 2025) produces the same observable symptom (features failing to fire) but has opposite encoder effects. The "Sanity Checks" paper (Korznikov et al., 2026) shows random baselines match trained SAEs on the very metrics used to evaluate absorption mitigation. If our measurement tools cannot distinguish trained from random features, how confident can we be that "absorption" is a coherent, distinct failure mode rather than an artifact of measurement methodology?
   - Sources: arXiv:2409.14507 (thresholds), arXiv:2509.23717 (sensitivity), arXiv:2505.11756 (hedging), arXiv:2602.14111 (random baselines), arXiv:2503.09532 (SAEBench metric modification)

2. **Assumption: The first-letter spelling task is a valid proxy for absorption in general.**
   - Evidence challenging it: Every absorption study, every SAEBench evaluation, every architecture comparison uses the first-letter spelling task. This is a syntactic, low-level feature hierarchy (letter membership > specific token) that exists primarily because of how tokenization works, not because of deep semantic structure. No study has demonstrated absorption on semantically rich hierarchies (entity types, reasoning chains, safety-relevant features). The first-letter task has the unusual property that ground-truth labels are trivially available for all tokens in the vocabulary --- a property that does not hold for the feature hierarchies that actually matter for interpretability. If absorption is primarily a phenomenon of shallow syntactic hierarchies, the entire research program may be solving a problem that does not matter for the safety applications that motivate it.
   - Sources: arXiv:2409.14507 (original metric), arXiv:2503.09532 (SAEBench), arXiv:2508.09363 (domain-specific SAEs noting first-letter is not useful for domain-specific absorption)

3. **Assumption: Feature absorption is caused by the sparsity penalty (L1/TopK).**
   - Evidence challenging it: Chanin et al. themselves note that BatchTopK SAEs, which lack L1 loss entirely, show clear absorption patterns. The explanation given is that absorption "improves reconstruction loss at a given k." This is a crucial admission: absorption may not be a sparsity artifact but rather a fundamental consequence of the reconstruction objective combined with finite dictionary width. If absorption occurs whenever reconstruction with fewer active features is possible (regardless of the sparsity mechanism), then the entire framing of absorption as a "sparsity pathology" is misleading, and architectures that modify sparsity (Matryoshka, ATM, OrtSAE) may be treating a symptom rather than the disease.
   - Sources: arXiv:2409.14507 (BatchTopK absorption), arXiv:2503.17547 (Matryoshka SAE trade-offs), arXiv:2412.06410 (BatchTopK architecture)

4. **Assumption: SAE features should be "monosemantic" and "atomic" --- absorption violates this ideal.**
   - Evidence challenging it: Leask et al. (ICLR 2025) show SAE features are neither canonical nor atomic. Engels et al. (ICLR 2025) discover irreducible multi-dimensional features. Song et al. (2025) show features are inconsistent across training runs. Korznikov et al. (2026) show random baselines match trained features on interpretability. Ma et al. (NeurIPS 2025) show auto-generated explanations of SAE features are overly broad and fail to account for polysemanticity. If the "monosemantic ideal" is itself wrong --- if LLMs do not internally represent information as sparse, one-dimensional, monosemantic atoms --- then absorption is not a pathology of SAEs but a natural consequence of trying to force a one-dimensional basis onto inherently multi-dimensional, non-atomic, context-dependent representations.
   - Sources: arXiv:2502.04878 (non-canonical), arXiv:2405.14860 (multi-dimensional), arXiv:2505.20254 (inconsistency), arXiv:2602.14111 (random baselines), OpenReview OJAW2mHVND (explanation quality)

5. **Assumption: Reducing absorption rate will make SAEs more useful for downstream tasks.**
   - Evidence challenging it: DeepMind (2025) deprioritized SAE research after finding dense linear probes dramatically outperform SAE probes on safety-relevant tasks, with 10-40% performance degradation from SAE-reconstructed activations. Wu et al. (2025) showed SAEs fail to outperform simple baselines on concept steering and detection. Korznikov et al. (2026) showed random baselines match SAEs on causal editing. Crucially, no study has shown that reducing absorption (e.g., using Matryoshka SAEs with 0.03 absorption rate instead of BatchTopK with 0.29) actually closes the gap between SAE probes and dense probes on downstream tasks. The Matryoshka SAE paper shows absorption reduction but does not demonstrate that this translates into improved downstream task performance compared to dense probes. The gap may be due to the information bottleneck inherent in sparse decomposition, not absorption specifically.
   - Sources: DeepMind Medium blog (2025), arXiv:2506.23845 (SAEs for discovery not action), arXiv:2602.14111 (random baselines), arXiv:2503.17547 (Matryoshka)

6. **Assumption: More sophisticated SAE architectures (Matryoshka, OrtSAE, ATM) are solving absorption.**
   - Evidence challenging it: The feature hedging paper (Chanin & Dulka, 2025) explicitly shows that Matryoshka SAEs trade absorption for hedging. Matryoshka inner levels suffer from feature hedging because they are effectively narrow SAEs. OrtSAE does not fully eliminate absorption. ATM has been evaluated only on Gemma-2-2B. The balanced Matryoshka SAE attempts to balance the absorption-hedging trade-off via a compound multiplier, but this is a manual hyperparameter. No architecture has been shown to eliminate absorption without introducing a new failure mode or degrading reconstruction. The fundamental tension identified by Chanin et al. --- that absorption saves 1 L0 per parent-child pair --- suggests this may be an inherent trade-off, not a solvable problem.
   - Sources: arXiv:2505.11756 (hedging trade-off), arXiv:2509.22033 (OrtSAE limitations), arXiv:2510.08855 (ATM limited evaluation), arXiv:2503.17547 (Matryoshka FVU penalty)

7. **Assumption: Feature absorption is the primary obstacle to SAE-based interpretability.**
   - Evidence challenging it: The Unified SDL Theory paper (arXiv:2512.05534) identifies multiple pathologies (dead neurons, feature splitting, absorption, spurious minima) as co-occurring failure modes of the same optimization landscape. Dark matter (Engels et al., 2024) accounts for ~50% of reconstruction error and is linearly predictable. Feature inconsistency across training runs (Song et al., 2025) means features are not reproducible regardless of absorption. The "Sanity Checks" paper calls the entire learned-feature paradigm into question. If SAEs fail at the more basic task of learning meaningful features in the first place, optimizing for absorption reduction is premature optimization.
   - Sources: arXiv:2512.05534 (unified theory), arXiv:2410.14670 (dark matter), arXiv:2505.20254 (inconsistency), arXiv:2602.14111 (sanity checks)

8. **Assumption: Anthropic's successful circuit tracing validates the SAE feature paradigm despite absorption.**
   - Evidence challenging it: Anthropic's attribution graph work (Lindsey et al., 2025) uses cross-layer transcoders (CLTs), not SAEs. The replacement model substitutes transcoders for MLPs, and Paulo et al. (2025) showed transcoders produce more interpretable features than SAEs while having similar absorption behavior. The fact that Anthropic's most celebrated interpretability result does not use SAEs undermines the narrative that improving SAE absorption will unlock the same kind of analysis. Furthermore, the CLT replacement model matches the underlying model's outputs in only ~50% of cases, suggesting even this approach has substantial limitations.
   - Sources: transformer-circuits.pub/2025/attribution-graphs (Anthropic), arXiv:2501.18823 (transcoders beat SAEs)

### Landscape of Doubt

The evidence reveals a troubling picture for the "feature absorption as primary research target" narrative. Feature absorption is measured by a single metric family that uses arbitrary thresholds, has been validated on a single narrow task (first-letter spelling), and whose reduction has not been shown to improve downstream utility. The phenomenon occurs even without L1 sparsity (in BatchTopK), challenging the standard causal explanation. New architectures that reduce absorption introduce other failure modes (hedging) or degrade reconstruction. The most critical recent results --- random baselines matching SAEs, probes outperforming SAEs, features being non-canonical and inconsistent --- suggest the field's foundational assumptions may be wrong, making absorption mitigation potentially premature.

The contrarian reading: **the field has been hyperfocused on absorption because it has a clean metric and a clean toy model, not because it is the most important problem.** The real problem may be that SAEs, as currently conceived, are the wrong tool for feature extraction from LLMs, and absorption is merely one symptom of this deeper misalignment.

---

## Phase 2: Initial Candidates

### Candidate A: "The Absorption Illusion: Is Feature Absorption a Measurement Artifact of the First-Letter Task?"

- **Challenged assumption**: Feature absorption is a robust, generalizable phenomenon that occurs across all feature hierarchies and is a primary obstacle to SAE-based interpretability.
- **Evidence against**: (1) All absorption measurements use the first-letter task, which has unique properties (trivial ground truth, syntactic hierarchy, tokenization-dependent) not shared by the semantic hierarchies that matter for interpretability. (2) The absorption metric uses manually-tuned thresholds (cosine similarity > 0.025, ablation gap >= 1.0) and was modified by SAEBench because it failed in later layers. (3) Initial ReLU SAE results showed decreased absorption, but this was later attributed to dead features, not genuine absorption reduction --- showing the metric is confounded by unrelated SAE properties. (4) No study has demonstrated absorption on semantic feature hierarchies (entity types, knowledge, safety features).
- **Contrarian hypothesis**: Measured "absorption rates" of 15-35% are inflated by task-specific confounds and do not predict how SAEs handle the hierarchical features that actually matter for interpretability. The first-letter task may systematically overestimate absorption because it involves features that are maximally hierarchical (every token has exactly one first letter, creating a perfect parent-child structure) and maximally easy to detect via linear probes.
- **Exploitation plan**: Construct absorption measurement tasks for semantic hierarchies (city/country, animal/species, sentiment/topic) using the same probe-based methodology as Chanin et al. Measure whether absorption rates generalize. If they do not, this fundamentally undermines the current research program.
- **Novelty estimate**: 7/10

### Candidate B: "Absorption is Not About Sparsity: Reconstruction Loss as the True Culprit"

- **Challenged assumption**: Feature absorption is caused by the sparsity penalty (L1, L0 regularization, TopK constraint) and can be fixed by modifying the sparsity mechanism.
- **Evidence against**: (1) BatchTopK SAEs show clear absorption patterns despite having no L1 penalty. (2) Chanin et al. explicitly note absorption "improves reconstruction loss at a given k." (3) The unified SDL theory (arXiv:2512.05534) shows absorption arises from spurious minima in the biconvex optimization landscape, which exists regardless of the sparsity mechanism. (4) Feature anchoring, the proposed fix from the unified theory, operates on identifiability, not sparsity. (5) All architectures that reduce absorption (Matryoshka, OrtSAE, ATM) do so by modifying training dynamics or decoder structure, not by removing the sparsity constraint.
- **Contrarian hypothesis**: Absorption is fundamentally a property of the MSE reconstruction objective under finite dictionary width, not the sparsity penalty. Any SAE that represents a parent-child pair with fewer dictionary elements than the full hierarchy requires will exhibit absorption, regardless of how sparsity is implemented. The field's focus on sparsity-based solutions is a red herring.
- **Exploitation plan**: Ablation study: train SAEs with identical architectures but systematically vary (a) sparsity mechanism (L1, TopK, BatchTopK, no sparsity), (b) reconstruction objective (MSE, cosine, KL), and (c) dictionary width. Show that absorption correlates with reconstruction objective and width, not sparsity type. This would redirect mitigation efforts.
- **Novelty estimate**: 8/10

### Candidate C: "Absorption as Optimal Compression: Why Fixing It May Be Harmful"

- **Challenged assumption**: Feature absorption is a pathology that needs to be fixed. Reducing absorption will improve SAE utility.
- **Evidence against**: (1) Chanin et al. prove that absorption saves 1 L0 per parent-child pair, meaning absorption-free SAEs have strictly worse sparsity-reconstruction trade-offs. (2) Matryoshka SAEs, which reduce absorption, have worse FVU than vanilla SAEs at the same L0. (3) No study has shown that reducing absorption improves downstream task performance relative to dense probes. (4) DeepMind's negative results suggest the problem is the sparse decomposition paradigm itself, not absorption specifically. (5) Wu et al. (2025) argue SAEs are best for concept discovery, not downstream action --- if so, absorption matters less because discovery tasks are tolerant of false negatives. (6) From an MDL (minimum description length) perspective (Ayonrinde et al., 2024), absorption may represent genuinely optimal compression of the model's representations.
- **Contrarian hypothesis**: Feature absorption is an optimal strategy for sparse coding of hierarchically structured information. Eliminating it necessarily degrades the sparsity-fidelity trade-off without proportional improvement in downstream utility. The field should accept absorption as a feature (pun intended) of sparse representation and develop downstream methods that are robust to it, rather than trying to eliminate it.
- **Exploitation plan**: (1) Formalize the information-theoretic optimality of absorption using rate-distortion theory. (2) Show empirically that reducing absorption (via Matryoshka, OrtSAE) does not close the gap with dense probes on downstream tasks. (3) Propose and evaluate "absorption-aware" downstream methods that account for known parent-child relationships. This reframes the research agenda from "fix absorption" to "work around absorption."
- **Novelty estimate**: 9/10

---

## Phase 3: Self-Critique

### Against Candidate A: "The Absorption Illusion"

- **Steelman**: The first-letter task was chosen precisely because it provides clean ground truth, making it a rigorous test. The fact that absorption occurs across all tested architectures (L1, TopK, BatchTopK, JumpReLU), all tested models (Gemma 2, Llama 3.2, Qwen2, GPT-2), and all tested widths strongly suggests it is a robust phenomenon, not a task-specific artifact. Furthermore, the mechanistic explanation (sparsity gain from absorbing parent into child) is architecture-agnostic and should apply to any feature hierarchy. The toy model proof in Chanin et al. provides theoretical grounding. SAEBench modified the metric to work across all layers, which strengthens rather than weakens the generality claim.
- **Cherry-picking check**: I am emphasizing the metric's limitations while downplaying the consistency of the finding across diverse settings. The threshold sensitivity concern is valid but could be addressed with a robustness analysis rather than dismissing the phenomenon.
- **Confounding check**: The confound between dead features and reduced absorption (in the ReLU SAE case) is a legitimate methodological concern, but it was identified and corrected by the SAEBench team. This actually shows the field is self-correcting.
- **Actionability check**: If absorption does not generalize to semantic hierarchies, this is a high-impact finding that redirects the field. If it does generalize, the cross-domain measurement itself is a valuable contribution. Either way, the experiments are actionable.
- **Verdict**: MODERATE. The core claim that first-letter results may not generalize is reasonable, but dismissing the phenomenon entirely goes too far given the theoretical grounding and cross-architecture consistency.

### Against Candidate B: "Absorption is Not About Sparsity"

- **Steelman**: The strongest evidence that sparsity matters is the Matryoshka SAE result: by modifying the loss structure (nested prefix losses), absorption drops from 0.29 to 0.03. This is a 10x reduction from a change in how sparsity is structured, not from changing the reconstruction objective. ATM achieves absorption score 0.0068 through temporal masking, which is fundamentally about feature selection (sparsity). OrtSAE reduces absorption through decoder orthogonality, which constrains how sparsity is achieved. These successes suggest sparsity mechanism modifications can dramatically reduce absorption, even if sparsity is not the sole cause.
- **Cherry-picking check**: I am overweighting the BatchTopK result (which shows absorption without L1) while underweighting the Matryoshka result (which shows dramatic absorption reduction via loss structure changes). Both are relevant, but the Matryoshka result is arguably more informative because it demonstrates a practical solution.
- **Confounding check**: The BatchTopK absorption finding could be explained differently: BatchTopK still has a sparsity mechanism (the top-k constraint), it just implements it differently than L1. The claim that "no sparsity penalty" exists in BatchTopK is misleading --- the k constraint is a hard sparsity constraint. Absorption occurring in all sparse coding methods is actually consistent with "sparsity causes absorption."
- **Actionability check**: If confirmed, this would redirect mitigation research from sparsity modifications to reconstruction objective modifications --- a concrete and actionable shift.
- **Verdict**: MODERATE. The nuance between "sparsity penalty" and "sparsity constraint" weakens the claim. However, the insight that absorption serves reconstruction, not just sparsity, remains valuable and under-explored.

### Against Candidate C: "Absorption as Optimal Compression"

- **Steelman**: The strongest argument against this view is Anthropic's circuit tracing work. When features are reliable (as shown in "On the Biology of a Large Language Model"), they enable powerful mechanistic understanding. Absorption directly undermines this: if a "deception" feature has absorbed exceptions for specific contexts, circuit tracing will miss those contexts. The practical cost of absorption is not just lower probe accuracy --- it is fundamentally about trust. If you cannot trust that a safety-relevant feature fires when it should, the entire interpretability program is compromised. The Matryoshka SAE result shows absorption can be reduced 10x with only a "minor reconstruction penalty" --- suggesting the optimality claim is weak.
- **Cherry-picking check**: I am emphasizing the theoretical optimality of absorption while downplaying the practical cost. The "absorption-aware downstream methods" proposal is speculative and may be harder to build than fixing absorption at the source.
- **Confounding check**: The DeepMind negative results may reflect many SAE limitations beyond absorption (dark matter, inconsistency, non-canonicality). Attributing the full SAE-vs-probe gap to "the sparse paradigm" rather than absorption is speculative.
- **Actionability check**: The rate-distortion formalization is genuinely novel. The empirical test (does absorption reduction close the gap with dense probes?) is concrete and falsifiable. The "absorption-aware methods" direction is interesting but vague. Overall, this candidate produces actionable research questions.
- **Verdict**: STRONG. The information-theoretic framing is novel, the empirical questions are crisp, and the contrarian position is defensible even if absorption reduction does help somewhat. The key insight --- that the field has not established whether absorption reduction actually improves downstream utility --- is a genuine blind spot.

---

## Phase 4: Refinement

### Dropped

**Candidate A** is weakened by the steelman: the toy model proof and cross-architecture consistency suggest absorption is a real phenomenon, not just a measurement artifact. However, the narrow task concern remains valid and is incorporated into the strengthened proposal below.

### Strengthened

**Candidate B** survives in attenuated form. The distinction between "sparsity penalty" and "sparsity constraint" is important --- I concede that BatchTopK has a sparsity mechanism, just not L1. However, the core insight --- that absorption serves reconstruction quality, not just sparsity minimization --- remains under-explored and is folded into the front-runner.

**Candidate C** is the front-runner, strengthened as follows:

1. **More precise claim**: Absorption is a locally optimal strategy in the reconstruction-sparsity landscape. Current mitigations (Matryoshka, OrtSAE, ATM) achieve absorption reduction by moving to a different region of this landscape that trades FVU for interpretability, but no evidence shows this trade-off improves downstream utility compared to dense probes. The claim is not that absorption is globally optimal, but that the field has not demonstrated that eliminating it is worth the cost.

2. **Constructive proposal**: Rather than arguing "do not fix absorption," propose a diagnostic framework that asks: (a) for a given downstream task, how much of the SAE-vs-probe performance gap is attributable to absorption specifically? (b) does reducing absorption (via Matryoshka/OrtSAE) close this gap? (c) can absorption-aware methods (that explicitly model parent-child feature relationships) achieve comparable performance without architectural changes?

3. **Additional corroboration**: Wu et al. (2025, arXiv:2506.23845) argue SAEs are best suited for concept discovery, not downstream action. If discovery is the primary use case, absorption matters less because discovery is tolerant of systematic false negatives (you still discover the concept exists, you just may miss some instances). This aligns with the emerging consensus after DeepMind's deprioritization.

4. **Cross-domain validation**: Extend the analysis to semantic feature hierarchies (city/country, animal/species) to test whether absorption rates are comparable. If absorption is lower for semantic hierarchies (which are less "perfectly" hierarchical than first-letter), this supports the view that the first-letter task overestimates the problem.

### Selected Front-Runner: Candidate C (strengthened)

---

## Phase 5: Final Proposal

### Title
"Rethinking Feature Absorption: When Optimal Compression Meets Interpretability Demands"

### Challenged Assumption
The field assumes that reducing feature absorption rates will make SAEs more useful for mechanistic interpretability and safety applications. Multiple architectural innovations (Matryoshka SAE, OrtSAE, ATM, masked regularization) have been proposed to reduce absorption, and absorption has been called "one of the most important open problems in SAE-based interpretability." However, no study has demonstrated that reducing absorption actually closes the performance gap between SAE features and dense probes on downstream tasks.

### Evidence

**For the assumption (that absorption matters):**
- Chanin et al. (2024) provide a clean toy model proof and empirical measurements showing 15-35% absorption rates across diverse SAEs.
- Anthropic's circuit tracing work demonstrates the practical value of reliable features for mechanistic understanding.
- The absorption mechanism is theoretically grounded: sparsity gain from absorbing parent into child features.
- Matryoshka SAEs reduce absorption 10x (from 0.29 to 0.03) and improve on multiple SAEBench metrics.

**Against the assumption (that fixing absorption helps downstream):**
- DeepMind deprioritized SAE research because dense probes dramatically outperform SAE probes on safety tasks, with 10-40% performance degradation from SAE-reconstructed activations. Absorption was cited as one factor, but not the primary one.
- Wu et al. (2025) show SAEs fail to outperform simple baselines on concept detection and steering, arguing SAEs are for discovery, not action.
- Korznikov et al. (2026) show random baselines match trained SAEs on interpretability, sparse probing, and causal editing.
- Matryoshka SAEs reduce absorption but have worse FVU than vanilla SAEs --- and no study shows this trade-off improves downstream task performance compared to dense probes.
- Chanin et al. themselves note absorption saves 1 L0 per parent-child pair, making absorption-free SAEs strictly worse on the sparsity-reconstruction frontier.
- Feature hedging (Chanin & Dulka, 2025) shows Matryoshka SAEs trade absorption for hedging, suggesting a conservation law of SAE failure modes.
- The first-letter spelling task, used by all absorption studies, has unique properties (perfect hierarchy, trivial ground truth) that may inflate absorption severity relative to real-world feature hierarchies.

### Hypothesis
Feature absorption is a locally optimal compression strategy for hierarchical features that cannot be eliminated without degrading the sparsity-reconstruction trade-off. The current research program of reducing absorption rates is premature because: (a) it has not been demonstrated that absorption reduction improves downstream utility; (b) the measurement task (first-letter spelling) may overestimate the severity of absorption for semantically meaningful hierarchies; and (c) the field should instead develop absorption-aware methods that exploit known feature hierarchy structure.

### Method

**Part 1: Does absorption reduction improve downstream utility?** (training-free, uses pre-trained SAEs)
- Compare vanilla BatchTopK SAEs vs. Matryoshka SAEs (which have 10x lower absorption) on downstream tasks: sparse probing on classification benchmarks, concept steering, safety-relevant feature detection (deceptive intent, harmful content).
- Use Gemma Scope SAEs and Gemma Scope 2 Matryoshka SAEs on Gemma 2 2B.
- Measure: (a) SAE probe accuracy vs. dense probe accuracy for each architecture, (b) the gap between SAE and dense probes, (c) whether absorption reduction correlates with gap reduction.
- Key prediction: If absorption is the primary problem, Matryoshka SAEs should substantially close the gap with dense probes. If the gap persists, absorption is not the bottleneck.

**Part 2: Cross-domain absorption measurement** (training-free)
- Construct absorption measurement tasks for semantic hierarchies:
  - City/Country hierarchy (e.g., "Paris" should fire both "city" and "France" features)
  - Animal/Species hierarchy (e.g., "eagle" should fire both "bird" and "eagle" features)
  - Sentiment/Topic hierarchy (e.g., a negative movie review should fire both "negative sentiment" and "movie" features)
- Adapt the Chanin et al. probe-based methodology to these tasks.
- Measure absorption rates and compare to first-letter task baselines.
- Key prediction: Absorption rates will be lower for semantic hierarchies because they are less "perfectly" hierarchical (features are more distributed, less binary).

**Part 3: Absorption-aware downstream methods** (training-free)
- Given known absorption patterns (identified via probes), construct "corrected" feature activations that re-activate absorbed parent features when child features fire.
- Evaluate whether this simple post-hoc correction recovers much of the SAE-vs-probe gap.
- If effective, this demonstrates that absorption can be handled at inference time without architectural changes.

**Part 4: Information-theoretic analysis** (theoretical)
- Formalize absorption as a rate-distortion trade-off: given a hierarchical feature structure with known co-occurrence statistics, derive the minimum number of active features needed to encode the hierarchy at a given reconstruction fidelity.
- Show that absorption arises naturally as the optimal encoding strategy when L0 is constrained below the hierarchy depth.
- Derive predictions for absorption rate as a function of L0, dictionary width, and hierarchy depth.

### Experimental Plan
- **Models**: Gemma 2 2B (primary), GPT-2 Small (secondary, for reproducibility)
- **SAEs**: Gemma Scope JumpReLU SAEs (vanilla, high absorption), Gemma Scope 2 Matryoshka SAEs (low absorption), SAEBench SAEs across architectures
- **Tasks**: First-letter spelling (baseline), city/country, animal/species, sentiment/topic (novel)
- **Metrics**: Absorption rate (Chanin et al.), sparse probing accuracy, concept steering effectiveness, SAE-vs-probe gap
- **Compute**: All experiments training-free; uses pre-trained SAEs and models. Single A100 sufficient. Target 1 hour per experiment.
- **Key ablation**: Vary L0 systematically (using SAEBench SAEs at multiple sparsities) to test whether absorption rate is predictable from L0 and hierarchy depth.

### Baselines
- Dense linear probes (the gold standard that SAEs fail to match)
- Vanilla SAEs without absorption mitigation
- Matryoshka SAEs (SOTA on absorption reduction)
- Random baseline SAEs (Korznikov et al., 2026) to establish minimum performance threshold

### Risk Assessment
**If the mainstream view is correct (absorption reduction helps downstream):**
- Matryoshka SAEs will substantially close the SAE-vs-probe gap on downstream tasks.
- The cross-domain experiments will show comparable absorption rates across all hierarchy types.
- The absorption-aware correction will be unnecessary because architectural fixes suffice.
- In this case, the paper becomes a thorough validation of the importance of absorption research, with the novel contributions being the cross-domain measurements and the information-theoretic analysis. This is still publishable.

**If our contrarian view is correct (absorption is not the bottleneck):**
- The SAE-vs-probe gap persists even with Matryoshka SAEs.
- Semantic hierarchies show lower absorption than the first-letter task.
- The absorption-aware correction provides partial but limited improvement.
- In this case, the paper provides the first evidence that the absorption research program is misdirected, and proposes a more productive alternative framing. This is high-impact.

**Mitigation**: The proposal is designed to produce valuable results regardless of the outcome. The cross-domain absorption measurements are novel regardless of what they show. The information-theoretic analysis provides the first formal treatment of absorption as a compression phenomenon.

### Novelty Claim
The specific insight: the SAE interpretability community has invested heavily in reducing absorption rates without demonstrating that this reduction translates to improved downstream utility. This paper provides (a) the first empirical test of whether absorption reduction closes the SAE-vs-probe gap, (b) the first cross-domain absorption measurements beyond the first-letter task, (c) the first information-theoretic formalization of absorption as optimal compression, and (d) the first evaluation of post-hoc absorption correction as an alternative to architectural mitigation.
