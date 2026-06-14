# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Feature absorption is a well-defined, distinct phenomenon that can be cleanly measured.**
   - Evidence challenging it: The canonical absorption metric (Chanin et al., 2024; arXiv:2409.14507) requires pre-specified probe directions, making it circular -- you must already know what features "should" exist to detect that they are missing. Chanin & Garriga-Alonso (2025; arXiv:2508.16560) show that most open-source SAEs have incorrect L0, causing feature hedging that manifests identically to absorption (features fail to fire where they should). The "Feature Hedging" paper (arXiv:2505.11756) explicitly shows absorption and hedging trade off against each other, suggesting they may be entangled manifestations of a single underlying capacity limitation rather than distinct pathologies. Tian et al. (2025; arXiv:2509.23717) frame absorption as merely a special case of poor feature sensitivity, suggesting the phenomenon may be less specific than claimed.

2. **Assumption: The first-letter spelling task is a valid and generalizable proxy for absorption in real-world interpretability applications.**
   - Evidence challenging it: The first-letter task is uniquely artificial. No language model actually needs to represent "starts with S" as a computational feature for next-token prediction -- this is a task imposed by the researcher, not one the model naturally performs. The Chanin et al. paper acknowledges this limitation explicitly: "our results use one model (Gemma-2-2b), one SAE architecture (JumpReLU), and one task (the first letter identification task)." Furthermore, the first-letter hierarchy is the simplest possible hierarchy (depth-2, clean binary membership), while real semantic hierarchies are noisy, multi-level, and overlapping. No study has demonstrated absorption on semantically rich features (knowledge, reasoning, safety-relevant behaviors). The SAEBench absorption evaluation itself acknowledges this is "not useful for evaluating domain-specific feature absorption" (from the SAEBench documentation).

3. **Assumption: Feature absorption is always harmful and should be minimized.**
   - Evidence challenging it: Chanin et al. themselves note in the Matryoshka SAE discussion that "for each parent-child feature relation, a vanilla SAE with feature absorption can represent both features with +1 L0, while an SAE without feature absorption would require +2 L0. Any solution that removes feature absorption will then likely have worse variance explained against L0." This means absorption is the OPTIMAL strategy under the sparsity-reconstruction objective. The LessWrong post "Sparsity is the enemy of feature extraction (ft. absorption)" (7vik et al., 2025) proves analytically that "increasing sparsity always increases absorption" -- absorption is not a bug but a mathematically inevitable consequence of the objective function doing exactly what we asked it to do. The MDL-SAE framework (Ayonrinde et al., 2024; arXiv:2410.11179) shows that absorption reduces description length by eliminating redundant parent-child encoding, which is desirable from an information-theoretic perspective.

4. **Assumption: SAEs would be useful for safety-critical interpretability if only we could fix absorption.**
   - Evidence challenging it: DeepMind's deprioritization (March 2025) found 10-40% performance degradation from SAE-reconstructed activations, with absorption being only ONE of many contributing factors. Korznikov et al. (2026; arXiv:2602.14111) show SAEs recover only 9% of true features in synthetic settings, with random baselines matching fully-trained SAEs. Basu et al. (2026; arXiv:2603.18353) found that clamping 3,695 SAE features associated with hazard cases produced "exactly zero change in model outputs" -- a devastating finding suggesting SAE features may not be causally relevant regardless of absorption. Wu et al. (2025; arXiv:2506.23845) show SAEs fail to outperform simple baselines on concept detection and steering. Even with perfect absorption-free SAEs, these deeper problems remain.

5. **Assumption: Architectural solutions (Matryoshka, OrtSAE, ATM) are making meaningful progress toward solving absorption.**
   - Evidence challenging it: The absorption-hedging tradeoff (Chanin et al., 2025; arXiv:2505.11756) shows Matryoshka SAEs trade absorption for hedging -- they have not solved the problem but merely shifted it. ATM SAE (arXiv:2510.08855) reports absorption score 0.0068 but is evaluated only on Gemma-2-2B on the first-letter task -- the same narrow benchmark. No architecture has been evaluated on absorption in knowledge, reasoning, or safety domains. SAE features are inconsistent across training runs (~30% shared between seeds; Song et al., 2025; arXiv:2505.20254), suggesting the features themselves are not stable objects that can be meaningfully "absorbed." Pacela et al. (2025; arXiv:2603.28744) show SAEs suffer from a fundamental "amortisation gap" that persists across all architectures and hyperparameter settings.

6. **Assumption: The linear representation hypothesis (LRH) -- that model features are linear directions -- is correct, and absorption is a failure of SAEs to recover these linear features.**
   - Evidence challenging it: Engels et al. (ICLR 2025; arXiv:2405.14860) discover irreducible multi-dimensional features (circular representations for days/months) that refute the one-dimensional LRH. Leask et al. (ICLR 2025; arXiv:2502.04878) show SAE features are neither canonical nor atomic -- meta-SAEs decompose "atoms" into sub-features, and larger SAEs find features missed by smaller ones. If the LRH is wrong, then "absorption" may not be a failure to recover real features but rather an artifact of forcing non-linear, multi-dimensional representations into a linear sparse framework that is fundamentally inappropriate.

### Landscape of Doubt

The research community has built an increasingly elaborate theory of feature absorption on remarkably thin empirical foundations. The entire absorption literature rests on a single controlled experiment -- the first-letter spelling task -- which is arguably the most artificial possible test case for interpretability. The phenomenon is measured with a metric that requires exactly the kind of supervised knowledge that SAE-based interpretability is supposed to render unnecessary. Meanwhile, the broader SAE enterprise is facing an existential crisis: DeepMind has deprioritized SAE research, SAEs fail to beat random baselines on synthetic data, and clamping thousands of supposedly meaningful SAE features produces zero behavioral change in models.

The most troubling finding for the absorption research program is not any single negative result, but the combination of:
- Absorption is **optimal** under the sparsity-reconstruction objective (proven analytically)
- Fixing absorption through architectural changes **trades it for hedging** (empirically demonstrated)
- Even if SAE features had perfect recall, **they do not causally control model behavior** (Basu et al., 2026)
- The very concept of "the feature that should have fired" **presumes the LRH**, which is increasingly challenged

This suggests that the feature absorption research program may be studying a symptom of a much deeper problem -- the inappropriateness of the entire sparse linear decomposition framework for understanding neural network representations -- while treating it as if it were a tractable engineering problem that better architectures can solve.

## Phase 2: Initial Candidates

### Candidate A: Absorption Is Not the Problem -- Disentangling Measurement Artifacts from Real Interpretability Failures

- **Challenged assumption**: That feature absorption, as currently measured, is a distinct and important phenomenon rather than a conflation of multiple measurement artifacts.
- **Evidence against**: (1) The absorption metric conflates true hierarchy-driven absorption with L0-induced hedging (Gap 9 in literature). (2) The first-letter task is not representative of real interpretability applications. (3) "Poor feature sensitivity" (Tian et al.) subsumes absorption as a special case, suggesting absorption is not a distinct phenomenon. (4) The metric requires supervised probe directions, creating an unfalsifiable circular evaluation.
- **Contrarian hypothesis**: What is called "absorption" is actually a mixture of at least three distinct phenomena -- L0-induced hedging, capacity-limited feature merging, and genuine hierarchy-driven absorption -- and the standard metric cannot distinguish between them. Furthermore, the first-letter task massively overestimates the practical importance of absorption because it tests features the model barely represents (graphemic information is weakly encoded compared to semantic features). A proper decomposition of "absorption" into its component causes would reveal that true hierarchy-driven absorption is rare and relatively benign, while the more damaging component is incorrect L0 selection (which is fixable by choosing the right hyperparameters).
- **Exploitation plan**: Systematically decompose observed feature failures into component causes using controlled experiments. Measure the same features across SAEs with matched L0 (controlling for hedging), across different widths (controlling for capacity), and only then attribute residual failures to true absorption. Compare absorption rates on the first-letter task (weak representation) vs. knowledge hierarchies (strong representation) to test whether absorption severity correlates with representation strength.
- **Novelty estimate**: 7/10

### Candidate B: Absorption as Rational Compression -- When "Failing" to Recover Features Is Actually the Right Answer

- **Challenged assumption**: That absorption should be minimized or eliminated.
- **Evidence against**: (1) Absorption is analytically optimal under the SAE objective (proven in the "Sparsity is the enemy" LessWrong post, 2025). (2) MDL-SAE theory shows absorption reduces description length, which IS desirable. (3) Matryoshka SAEs that reduce absorption sacrifice reconstruction quality. (4) For downstream tasks like circuit analysis (Lindsey et al., 2025), what matters is not whether individual features fire correctly but whether the full SAE reconstruction preserves the model's computational graph -- and absorption preserves reconstruction perfectly.
- **Contrarian hypothesis**: Feature absorption is an intelligent compression strategy, not a failure. When the model computes "starts with S" information as part of the token "short" representation, the model itself is "absorbing" the letter-level feature into the token-level representation. The SAE is faithfully mirroring the model's own computational structure. The problem is not that the SAE absorbs features -- the problem is that we expected SAEs to decompose representations into a taxonomy that the model itself does not maintain. Attempting to force SAEs to avoid absorption would actually make them LESS faithful to the model's true representational structure.
- **Exploitation plan**: Test whether "absorbed" features in the SAE correspond to features that the MODEL ITSELF represents in a non-decomposed way. Use probing experiments at different model layers to determine whether the model maintains separate representations for "starts with S" and individual tokens, or whether these are already merged. If the model itself does not maintain separate linear representations for parent features, then absorption is not an SAE failure but a correct reflection of model internals.
- **Novelty estimate**: 8/10

### Candidate C: The First-Letter Illusion -- Absorption Is an Artifact of Testing on Weakly Represented Features

- **Challenged assumption**: That the first-letter spelling task provides generalizable evidence about SAE feature absorption.
- **Evidence against**: (1) The authors themselves acknowledge the task is narrow and limited to one model/architecture. (2) First-letter information is a secondary feature of token representations -- models do not primarily encode graphemic information. (3) SAEBench documentation acknowledges the absorption benchmark's limitations for domain-specific evaluation. (4) No replication has been done on semantically primary features (entity types, sentiment, factual knowledge). (5) The hierarchical structure of the first-letter task (26 parent features, ~1000 child features) has an extremely high branching factor, which is atypical of natural semantic hierarchies.
- **Contrarian hypothesis**: The absorption rates of 15-35% reported by Chanin et al. are dramatically inflated by the choice of a task where the "parent" feature (letter membership) is weakly represented relative to "child" features (token identity). On features that the model robustly and primarily represents -- such as entity types, named entities, or factual attributes -- absorption rates will be an order of magnitude lower because the model has stronger, more independent representations for both parent and child features. The first-letter task creates a worst-case scenario for absorption because (a) the parent feature is weakly encoded, (b) the child features perfectly predict the parent, and (c) the hierarchy has unnaturally high branching factor. This would mean the entire alarm about absorption threatening SAE-based interpretability is founded on unrepresentative evidence.
- **Exploitation plan**: Measure absorption on knowledge hierarchies (city->country, entity->type) using RAVEL and sae-entities datasets. Compare absorption rates to the first-letter baseline. If absorption is dramatically lower on semantic features, the paper's central contribution is demonstrating that absorption severity is feature-dependent and that the alarm was overblown.
- **Novelty estimate**: 8/10

## Phase 3: Self-Critique & Adversarial Testing

### Against Candidate A (Measurement Artifact Decomposition)

- **Steelman the conventional view**: Chanin et al.'s toy model proof is mathematically airtight -- under the stated assumptions (hierarchical features, sparsity penalty), absorption provably occurs. The empirical evidence extends across multiple models (Gemma, Llama, Qwen) and hundreds of SAEs. The phenomenon is real, not an artifact. Even if L0 and hedging contribute to observed failures, the analytical proof guarantees that true hierarchy-driven absorption exists as a distinct mechanism.
- **Cherry-picking check**: I am emphasizing the confounds (L0, hedging) while downplaying that the Chanin et al. proof specifically isolates the hierarchy mechanism independent of these confounds. The toy model has no hedging (it has sufficient width) and correct L0, yet absorption still occurs. This is genuine evidence that absorption is distinct from hedging.
- **Confounding check**: The claim that "what we call absorption is really three things" may be true, but even if true, each component is still a real problem. Decomposing the phenomenon into three sub-problems does not make any of them go away. The practical question is whether decomposition leads to better solutions -- and it might, since each component has a different fix (hedging: increase width; L0: tune L0; hierarchy absorption: architecture changes).
- **Actionability check**: If the decomposition shows that most observed "absorption" is actually hedging or incorrect L0, this would redirect community effort toward simpler fixes (tuning hyperparameters) rather than complex architectural changes. This IS actionable and valuable.
- **Verdict**: MODERATE -- The decomposition is worthwhile but does not invalidate absorption as a real phenomenon. The strongest version of this idea is not "absorption is not real" but "absorption is real but its practical impact is exaggerated by confounding with more mundane problems."

### Against Candidate B (Absorption as Rational Compression)

- **Steelman the conventional view**: The safety argument is compelling -- if we want to use SAE features to detect deception, and the "deception" feature has been absorbed into token-specific features, we cannot reliably detect deception. This is a real problem regardless of whether absorption is "rational." The 7vik et al. (2025) post makes this explicit: "a feature for deception, with absorption, might actually have learned 'deception-except-deception-in-2027.'" From the safety perspective, a feature with arbitrary exceptions is useless, no matter how compression-optimal it is.
- **Cherry-picking check**: I am emphasizing the compression optimality while ignoring the devastating downstream consequences. The fact that absorption preserves reconstruction is irrelevant if reconstruction is not the goal -- the goal is interpretability, and absorbed features are by definition less interpretable. Furthermore, the "model mirrors absorption" hypothesis is speculative: I have no direct evidence that the model's own representations merge parent and child features in the way the SAE does.
- **Confounding check**: My argument that the model "itself absorbs" parent features into child features is a specific empirical claim that needs testing. It is plausible that the model maintains separate representations (linear probes with high accuracy suggest this) that the SAE fails to recover. If linear probes can extract "starts with S" with high accuracy from the residual stream, then the information IS separately represented in the model, and the SAE is failing to capture it.
- **Actionability check**: If absorption IS rational compression, the implication is that we should stop trying to fix absorption and instead develop new interpretability tools that work WITH absorbed features (e.g., meta-SAEs, multi-feature explanations). This is actionable but defeatist -- it abandons the goal of monosemantic feature extraction.
- **Verdict**: MODERATE -- The "absorption is rational" framing is intellectually provocative and partially correct (absorption IS optimal under the SAE loss), but the practical argument (absorption undermines the use case for SAEs) remains strong. The strongest version of this idea is not "absorption is fine" but "we are optimizing for the wrong loss, and fixing the loss is more important than fixing absorption."

### Against Candidate C (First-Letter Illusion)

- **Steelman the conventional view**: The Chanin et al. authors explicitly state they "suspect that feature absorption is just easiest to find for token-aligned features but conceptually could occur any time a similar structure exists between features." The toy model proof makes no assumptions about the specific features -- it applies to ANY hierarchical feature pair with non-zero co-occurrence. Multiple independent lines of evidence (Matryoshka SAE reducing absorption from 0.29 to 0.03, OrtSAE reducing by 70%) confirm absorption is a real phenomenon that mitigations address, not a measurement artifact.
- **Cherry-picking check**: I am emphasizing the narrowness of the first-letter evaluation while ignoring that the toy model proof is task-independent. The proof shows that sparsity + hierarchy => absorption, full stop. The first-letter task is merely a convenient evaluation setting, not the only domain where absorption occurs. Searching for independent evidence: the feature sensitivity paper (Tian et al., 2025) finds poor sensitivity (a generalization of absorption) across diverse feature types, not just first-letter features.
- **Confounding check**: My hypothesis that absorption rates will be lower on semantic features rests on the assumption that "stronger representation" implies "less absorption." But the sparsity incentive for absorption depends on co-occurrence frequency, not representation strength. Entity-type features (e.g., "is a city") may have HIGHER co-occurrence with child features (specific cities) than first-letter features do, which would predict MORE absorption, not less. This is the key risk to my hypothesis.
- **Actionability check**: If absorption IS lower on semantic features, this has major implications -- it means the community can deprioritize absorption mitigation and focus on other SAE failure modes. If absorption is NOT lower, this confirms that the phenomenon generalizes and strengthens the case for mitigation research. Either way, the experiment produces valuable information.
- **Verdict**: STRONG -- This is the most testable contrarian hypothesis. The experiment is well-defined, feasible within the training-free constraint, and produces an informative result regardless of the outcome. The risk to the hypothesis (that semantic features may have HIGHER co-occurrence and thus MORE absorption) makes the experiment even more interesting.

## Phase 4: Refinement

### Dropped: Candidate A (Measurement Artifact Decomposition)
While the decomposition of absorption into components is valuable, the toy model proof establishes that true hierarchy-driven absorption is a real, distinct mechanism. The self-critique revealed that decomposition does not invalidate the phenomenon but rather refines our understanding of its relative contribution. This is a useful supporting analysis but not a paper-carrying idea.

### Dropped: Candidate B (Absorption as Rational Compression)
The "absorption is rational" framing, while intellectually interesting, falls apart under the safety argument. Even if absorption is loss-optimal, it undermines the interpretability use case that motivates SAE research. Furthermore, the key empirical claim (model internals mirror absorption) is speculative and requires evidence that may not exist. However, the insight about loss function design is retained as a supporting element.

### Strengthened: Candidate C (First-Letter Illusion) -> Evolved into "When Does Absorption Actually Matter? A Cross-Domain Characterization"

The core contrarian claim is refined based on the self-critique:

**Original claim**: Absorption is dramatically lower on semantic features.

**Refined claim**: Absorption severity is strongly predicted by the representational strength and independence of the parent feature relative to its children -- a property that varies dramatically across domains. The first-letter task represents a worst-case scenario (weak parent, strong children, high co-occurrence), while knowledge hierarchies may exhibit qualitatively different absorption patterns. The field's perception of absorption as a universal, severe problem is driven by evaluation on this worst-case scenario.

**Additional corroboration found**:
- The RAVEL benchmark (integrated into SAEBench) already provides entity-attribute hierarchies (city->country, city->continent, city->language) that have been used to evaluate SAEs, but NOT specifically for absorption measurement. This provides ready-made infrastructure.
- The sae-entities work (Ferran et al., ICLR 2025) shows entity recognition features exist in Gemma SAEs as paired known/unknown latents across cities, movies, songs -- confirming that the features needed for cross-domain absorption measurement are present in existing SAEs.
- The feature sensitivity paper (Tian et al., 2025) shows sensitivity varies across feature types, supporting the hypothesis that absorption-like phenomena are feature-dependent.
- DeepMind's deprioritization was specifically motivated by SAE failures on SAFETY features (harmful intent detection) -- but nobody has measured absorption specifically on safety features. If absorption rates differ between syntactic features and safety features, this is crucial information for the field.

**Key insight retained from Candidate B**: The reason absorption varies across domains is that it reflects the OBJECTIVE FUNCTION's interaction with the specific feature hierarchy structure. Absorption is not a fixed pathology but a function of (1) the co-occurrence statistics between parent and child features, (2) the relative representational strength of parent vs. child features in the model's activations, and (3) the sparsity pressure. All three vary across domains. The first-letter task is an extreme point in this space, not a representative one.

**Key insight retained from Candidate A**: Any cross-domain absorption measurement must control for confounds (L0, hedging) that are known to inflate apparent absorption rates. The experiment design should include matched L0 controls and explicit hedging measurement alongside absorption measurement.

## Phase 5: Final Proposal

### Title: Rethinking Feature Absorption: When and Why Sparsity-Induced Feature Merging Actually Matters

### Challenged Assumption
The field treats feature absorption as measured in 15-35% rates on the first-letter spelling task (Chanin et al., 2024) as evidence of a universal, severe SAE pathology that undermines mechanistic interpretability. This perception drives significant architectural research (Matryoshka SAE, OrtSAE, ATM, masked regularization). However, this entire research program rests on evaluation with a single, arguably unrepresentative task. No study has established whether absorption generalizes to the semantic and safety-relevant features that actually motivate SAE-based interpretability.

### Evidence

**For the mainstream view (absorption is universal and severe):**
- Chanin et al.'s toy model proof applies to ANY hierarchical feature pair with co-occurrence, independent of the specific features involved
- Absorption is found in every SAE architecture tested (L1, TopK, JumpReLU, across Gemma, Llama, Qwen)
- SAEBench confirms that TopK and JumpReLU significantly worsen absorption
- The "Sparsity is the enemy" proof shows absorption is analytically guaranteed under sparsity + hierarchy
- OrtSAE and Matryoshka SAE achieve measurable absorption reduction, confirming the phenomenon is real and addressable

**Against the mainstream view (absorption may be overestimated and context-dependent):**
- All empirical absorption measurements use only the first-letter spelling task
- First-letter features are weakly represented (graphemic information is secondary to semantic content in LLMs)
- The absorption metric requires supervised probes, creating circular evaluation
- The first-letter hierarchy has unusual properties: depth-2, branching factor ~40 (1000 tokens / 26 letters), near-perfect co-occurrence (a token ALWAYS starts with exactly one letter)
- Feature hedging (incorrect L0) produces identical symptoms to absorption and is prevalent in open-source SAEs
- Tian et al. (2025) show feature sensitivity (a generalization of absorption) varies across feature types
- Dense linear probes achieve high accuracy on both parent and child features, suggesting the model maintains separate representations that SAEs fail to recover -- but the degree of separation may vary across domains
- The "rational compression" perspective shows absorption is loss-optimal, suggesting it may be less severe when the underlying features are more clearly separated in the model's representation space

### Hypothesis
Feature absorption severity is governed by three domain-specific factors that vary dramatically across different types of semantic hierarchies:
1. **Parent-child co-occurrence rate**: Higher co-occurrence creates stronger sparsity incentive to absorb (first-letter task: ~100% co-occurrence; city-country: high but variable; sentiment-topic: low)
2. **Parent feature representational independence**: When the parent feature is strongly and independently represented (not just as the sum of its children), absorption is harder because the encoder must actively suppress a strong signal (syntactic features like POS may be more independent than graphemic features like first-letter)
3. **Feature hierarchy branching factor**: Higher branching factors create more child features competing to absorb the parent, increasing absorption probability (first-letter: ~40; knowledge hierarchies: variable, often lower)

The first-letter task sits at the extreme of all three factors (near-100% co-occurrence, weak parent representation, high branching), making it a worst-case scenario that overestimates absorption severity for most practical applications.

### Method
A training-free, cross-domain absorption characterization using existing pretrained SAEs and established evaluation infrastructure.

**Phase 1: Cross-Domain Absorption Measurement (core experiment)**
1. Adapt the sae-spelling absorption metric to three additional feature hierarchies with known structure:
   - **Knowledge hierarchy**: city -> country (via RAVEL dataset; parent = "located in France", children = specific French cities). This tests absorption on strongly represented factual features with moderate branching factor.
   - **Taxonomic hierarchy**: entity -> entity type (via sae-entities code; parent = "is a person", children = specific people). This tests absorption on high-level categorical features.
   - **Attribute hierarchy**: city -> language (via RAVEL; parent = "language is French", children = French-speaking cities). This tests absorption with partial co-occurrence (not all cities in a country share the same language at the country-wide level; cross-country language sharing).
2. Run on 5-8 Gemma Scope pretrained SAEs (varying width: 16k, 65k; varying layers: early, middle, late) plus SAEBench pretrained SAEs if available.
3. For each domain-hierarchy, train LR probes for parent features and compute absorption rate using the Chanin et al. metric.
4. Compare absorption rates across domains and to the first-letter baseline.

**Phase 2: Confound Control**
1. For each measurement, also compute feature hedging score (using feature-hedging-paper code) to disentangle hedging from true absorption.
2. Match SAEs by L0 when comparing across architectures to control for the L0 confound.
3. Measure parent feature probe accuracy separately (how well does a linear probe extract the parent feature from the residual stream?) to quantify "parent representational strength."

**Phase 3: Predictive Model**
1. For each domain-hierarchy, compute the three proposed predictive factors: (a) parent-child co-occurrence rate, (b) parent feature probe accuracy (representational strength), (c) branching factor.
2. Fit a regression model predicting absorption rate from these factors.
3. If the model has high predictive power (R^2 > 0.7), this provides a practical tool for predicting absorption severity on new domains without running the full metric.

### Experimental Plan
- **Models**: Gemma 2 2B (primary, via Gemma Scope SAEs), GPT-2 Small (secondary, for replication on a different architecture)
- **SAEs**: Gemma Scope JumpReLU SAEs at widths 16k and 65k, layers 6, 12, 20 (covering early-mid-late); SAEBench pretrained SAEs on Gemma-2-2B layer 12 (multiple architectures if accessible)
- **Datasets**: First-letter spelling task (baseline, from sae-spelling), RAVEL (city-country, city-language), sae-entities entity type data
- **Pilot**: Single SAE (Gemma Scope 16k, layer 12), single hierarchy (city->country). Estimate: 15-20 minutes.
- **Full experiment**: 5-8 SAEs x 4 hierarchies x (absorption + hedging + probe accuracy). Estimate: 4-6 hours total, parallelizable across SAEs. Each individual run stays under 1 hour.
- **Infrastructure**: sae-spelling (absorption metric), SAELens (SAE loading), TransformerLens (activation extraction), RAVEL dataset, sae-entities code. All MIT/Apache licensed and actively maintained.

### Baselines
- **First-letter absorption rate** (Chanin et al.): The standard 15-35% baseline. Our experiment must replicate this on the same SAEs before extending to other domains.
- **Dense linear probe accuracy**: If probes achieve high accuracy on parent features, this confirms the information is present in the model and absorption is an SAE failure. If probe accuracy is low, absorption may reflect the model's own representational choices.
- **Random baseline absorption rate**: Compute "absorption rate" for random probe directions to establish a false-positive floor for the metric.

### Risk Assessment
**What if the mainstream view turns out to be correct?**
If absorption rates are equally high (15-35%) across all domains, this STRENGTHENS the case for absorption mitigation research by providing the first cross-domain evidence of the phenomenon's universality. The paper would then contribute:
- First evidence that absorption generalizes beyond the first-letter task
- Identification of which semantic hierarchies are most vulnerable
- A quantitative analysis of the factors (co-occurrence, representational strength, branching) that predict absorption severity

In this scenario, the contrarian hypothesis is falsified, but the experiment still produces a high-value publication because the field urgently needs cross-domain absorption data regardless of the outcome.

**What if absorption rates vary dramatically across domains?**
This is the contrarian prediction. If absorption is low (< 5%) on knowledge/semantic hierarchies, the paper challenges the urgency of absorption mitigation research for practical interpretability applications and suggests the community should focus on other SAE failure modes (hedging, inconsistency, lack of causality) that may be more impactful for safety-relevant features.

**What if the measurement itself fails?**
The absorption metric may not transfer cleanly from first-letter features to semantic features if (a) semantic features are more polysemantic or (b) the hierarchy structure is noisier. This risk is mitigated by including probe accuracy and hedging measurements, which provide interpretable results even if the absorption metric has limited precision on new domains.

### Novelty Claim
This would be the first systematic measurement of feature absorption outside the first-letter spelling task, providing the cross-domain evidence that the field needs to assess whether absorption is a universal SAE pathology or a phenomenon whose severity is strongly domain-dependent. The contrarian framing -- that the alarm about absorption may be overblown because it is based on an unrepresentative benchmark -- challenges the current research direction while being constructively falsifiable. Regardless of outcome, the experiment closes the critical Gap 2 (absorption only studied on first-letter task) and Gap 6 (absorption in hierarchically rich domains is unstudied) identified in the literature survey.
