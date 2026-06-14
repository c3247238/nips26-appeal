# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

Below are 10 resources organized by the mainstream assumption or claim they challenge.

1. **Assumption: "Feature absorption is a clearly defined, well-measured failure mode of SAEs"**
   - Evidence challenging it:
     - Chanin et al. (2024/2409.14507) — the authors themselves acknowledge "our probe-based metric is quite noisy." The absorption measurement procedure requires: (a) training a supervised LR probe, (b) running k-sparse probing to identify feature splits, and (c) integrated-gradients ablation. The metric is undefined as applied to most features because it requires knowing in advance what concept a feature "should" encode. This is methodologically circular: you first assume you know what a feature represents, then declare it "absorbed" when it doesn't behave that way.
     - SAEBench (arXiv:2503.09532) explicitly identifies feature absorption measurement as one of 8 metrics and notes that "proxy metrics do not reliably predict practical performance." SAEBench itself evaluates absorption only on first-letter tasks — a deliberate limitation acknowledged in the paper.
     - Tian et al. (arXiv:2509.23717) — "Measuring Sparse Autoencoder Feature Sensitivity" — shows that the absorption metric misses a significant dimension of the problem: many features that score well on the absorption metric still have poor sensitivity (they activate correctly but inconsistently on semantically similar inputs). This means the absorption metric is not a complete operationalization of the underlying phenomenon.

2. **Assumption: "Feature absorption is caused by hierarchical co-occurrence and sparsity optimization"**
   - Evidence challenging it:
     - Chanin et al. Section A.7 only tests the co-occurrence hypothesis in a toy model. The causal mechanism in real LLMs is inferred, not directly established. Multiple confounds exist: (a) frequency distributions (Zipfian co-occurrence naturally creates hierarchical patterns), (b) geometry of the residual stream (high-dimensional interference), (c) training corpus statistics (domain-specific texts over-represent certain hierarchies).
     - The unified SDL theory (arXiv:2512.05534) explains absorption as a "spurious minimum" of piecewise biconvex optimization — but this is a necessary, not sufficient, condition. Many spurious minima are never found; whether the optimizer finds the absorption minima depends on initialization, learning rate, batch size, and data order in ways that have not been empirically characterized.
     - Korznikov et al. (arXiv:2602.14111) "Sanity Checks for Sparse Autoencoders" (2026) show that SAEs recover only ~9% of true features in synthetic settings where ground truth is known. This suggests absorption might be a small contributor to a much larger systematic failure that is not caused by feature hierarchy at all — the real culprit may simply be insufficient capacity and the fundamental non-identifiability of dictionary learning.

3. **Assumption: "Absorption rates of 15–35% are representative of how bad the problem is across the board"**
   - Evidence challenging it:
     - The entire empirical absorption literature uses the first-letter spelling task (Chanin et al., 2024). "Resurrecting the Salmon" (arXiv:2508.09363, 2025) explicitly states "SAEBench evaluates feature absorption by using features for 'word starts with x', which is *not useful for evaluating domain-specific feature absorption*." The first-letter task has an artificial, symmetric feature hierarchy (all 26 letters, each letter covering similar fractions of the vocabulary). Real semantic hierarchies are far more imbalanced (e.g., country→city is a 195:10,000+ imbalance).
     - There is no published evidence that absorption rates of 15–35% extend to: knowledge-type features (country→city), reasoning features (premise→conclusion), safety-relevant features (harmful intent→specific type of harm), or non-English settings.
     - The absorption metric requires known ground-truth feature labels. For the vast majority of SAE features — which are discovered unsupervised — no such labels exist. The measured absorption rate therefore samples only the most human-legible, structurally clean features and almost certainly underestimates or mischaracterizes absorption in the general population of features.

4. **Assumption: "Wider and more sparse SAEs make absorption worse, and this is a monotonic relationship"**
   - Evidence challenging it:
     - SAEBench results show this relationship holds for most architectures but breaks for Matryoshka SAEs. The relationship between width, sparsity, and absorption is architecture-dependent in ways that suggest the mechanism is not simply "more features = more opportunities to absorb."
     - ATM SAE (arXiv:2510.08855) achieves absorption scores of 0.0068 vs. TopK 0.1402 on Gemma-2-2B — a 20× improvement — using per-latent importance tracking that disrupts the temporal dynamics of absorption. If absorption were simply caused by width and sparsity, importance tracking should not have such a dramatic effect. This suggests the optimizer's trajectory, not just its final capacity, drives absorption.
     - The Feature Hedging paper (arXiv:2505.11756) shows a complementary regime: narrow SAEs have less absorption but more hedging. This absorption-hedging tradeoff is not predicted by the standard "wider = more absorption" story and suggests that the underlying mechanism involves a tradeoff between two failure modes that are not simultaneously measurable by the absorption metric alone.

5. **Assumption: "Absorption is bad for interpretability"**
   - Evidence challenging it:
     - Chanin et al.'s own framing is ambiguous. They show absorbed features have lower recall on the first-letter probe, but they do not show that the absorbing feature (e.g., the token-specific feature "lion" that absorbs "starts with L") is less interpretable than the absorbed feature. In fact, token-aligned features that absorb general features may be *more* interpretable to humans (a feature for "lion" is more semantically meaningful than a feature for "starts with L").
     - The position paper "Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts" (arXiv:2506.23845, 2025) explicitly argues that SAEs' strength is *discovery*, not *reliable detection*. From the discovery perspective, absorption may be revealing something true about how the model compresses information: the model has a "lion" concept that subsumes "starts with L" in context. This is not a failure — it is information about the model's actual internal structure.
     - DeepMind's "Sanity Checks" paper (arXiv:2602.14111) shows that random baselines match fully-trained SAEs on interpretability, sparse probing, and causal editing tasks. If even the overall SAE interpretation is unreliable at that level, singling out absorption as the critical problem may be missing the forest for the trees.

6. **Assumption: "Mitigation methods (OrtSAE, Matryoshka, ATM) genuinely solve absorption"**
   - Evidence challenging it:
     - OrtSAE claims 65% absorption reduction, but the base rate comparison uses the same first-letter task metric. If the metric is noisy and architecture-dependent (as described above for MP-SAEs), the 65% claim may reflect the metric's sensitivity to OrtSAE's particular encoder structure rather than genuine absorption elimination.
     - Matryoshka SAE reduces absorption but introduces feature hedging in its inner levels (Chanin & Dulka, 2025). This is a trade, not a fix.
     - ATM SAE achieves 0.0068 vs. TopK 0.1402, but has only been evaluated on Gemma-2-2B and one layer type. No replication across other models, layers, or feature hierarchies has been published.
     - The Matching Pursuit SAE example from SAEBench is instructive: an architecture explicitly designed to handle feature hierarchy scored *worse* on the absorption metric, not better. This suggests there may be something about the absorption metric itself that is sensitive to architectural choices in ways that do not track with actual absorption.

7. **Assumption: "Absorption is primarily a training-time phenomenon fixable by architecture changes"**
   - Evidence challenging it:
     - Mechanistic Interpretability 2026 status report notes "chaotic dynamics in deep networks: steering vectors become completely unpredictable after just O(log(1/ε)) layers." If the residual stream itself has chaotic properties, absorption may be a symptom of fundamental non-linearity in how features are composed — not a training artifact at all.
     - The linear representation hypothesis has been partially falsified (Csordás et al., 2024, found "onion" nonlinear representations). If representations are not actually linear, then absorption is not a fixable training artifact but a fundamental category error in using linear dictionary learning to decompose non-linear representations.

8. **Assumption: "The absorption phenomenon is unique to SAEs and could be avoided with better training"**
   - Evidence challenging it:
     - Transcoders and skip transcoders (arXiv:2501.18823) — designed on a fundamentally different input-output paradigm — still exhibit comparable feature absorption behavior. "SAEs and transcoders were found to have similar feature absorption behavior." This suggests absorption is a property of the data manifold and the sparsity objective, not of the specific encoder/decoder architecture.

9. **Assumption: "The causal structure of absorption is well understood: sparse features absorb dense features"**
   - Evidence challenging it:
     - The integrated-gradients ablation used to confirm absorption is causally important makes a hidden assumption: that the absorbed direction's causal effect on model output is mediated *through* the SAE's reconstruction, not around it via the SAE error term ("dark matter"). Engels et al. (arXiv:2410.14670) show that ~50% of the SAE error vector is linearly predictable from input — meaning the model's computation does not pass entirely through the SAE features. An absorbed feature might actually operate *in the error term* rather than in any SAE latent, which would make the standard absorption detection method (ablating SAE latents) systematically mislead about whether information is "absorbed" at all.

10. **Assumption: "Absorption is the primary reason dense probes outperform SAE probes on safety tasks"**
    - Evidence challenging it:
      - DeepMind's 2025 blog describes SAE probe failure on harmful intent detection, citing absorption as one explanation. But the same paper notes that even training chat-specific SAEs (which should reduce absorption for safety features) "closed about half the gap but was still worse than linear probes." This residual gap after addressing corpus mismatch suggests that absorption is not the sole driver — fundamental probe sparsity (1-sparse probing is severely limited) and feature geometry may be the larger contributors.

### Landscape of Doubt

The SAE feature absorption field has a structural problem: the canonical measurement procedure (Chanin et al., 2024) uses a single, artificial proxy task (first-letter spelling) to establish absorption rates that are then treated as representative of a general pathology. Virtually every downstream claim — that 15–35% absorption rates are problematic, that wider SAEs are worse, that OrtSAE reduces absorption by 65% — rests on this one task, one metric, one probe-based measurement procedure.

The deeper issue is that "absorption" as currently defined is not a property of an SAE in isolation — it is a relational property that only makes sense given a prior belief about what concept a latent "should" encode. For unsupervised features (the vast majority of any SAE's dictionary), this prior doesn't exist. The field has operationalized absorption for a tiny, specially constructed subset of features and then generalized as if it were a global characterization.

Meanwhile, evidence accumulates that SAEs have a far more fundamental problem: even in controlled synthetic settings with known ground truth, SAEs recover only ~9% of true features (arXiv:2602.14111). Absorption (a nuanced secondary failure mode) may be a distraction from this primary catastrophic failure.

---

## Phase 2: Initial Candidates

### Candidate A: The Absorption Metric Is a Measurement Artifact of the Probe-Dependence Problem

- **Challenged assumption**: Feature absorption is a real, measurable, and general failure mode of SAEs, characterized by the Chanin et al. metric.
- **Evidence against it**: The metric is defined relative to a linear probe trained on the same residual stream the SAE is meant to decompose. Absorption is declared whenever the probe succeeds but the SAE feature fails — but the probe may be picking up on directions that are genuinely multi-feature (not the single "general" concept the researcher intends), making the "absorption" label misleading. The metric is explicitly acknowledged to be "quite noisy" by the authors. The MP-SAE example shows the metric can give anti-intuitive results for architectures with different sparsity mechanisms.
- **Contrarian hypothesis**: A significant fraction of reported "absorption" cases are not genuine feature absorption (where one latent takes over another's role) but rather measurement artifacts caused by the probe encoding a mixture of concepts that the SAE correctly disentangles — the probe is the imprecise instrument, not the SAE.
- **Exploitation plan**: Design controlled experiments where the ground truth of feature composition is known (using synthetic activations or activation patching setups). Measure how often the absorption metric fires when there is genuine hierarchical co-occurrence vs. when there is only superficial co-occurrence without true semantic implication. Compare probe-based absorption measurement with activation-patching-based causal measurement. If they diverge significantly, the probe-based metric is partially spurious.
- **Novelty estimate**: 7/10 (no paper has directly compared probe-based vs. causal-ablation-based absorption detection at scale)

### Candidate B: Absorption Is a Signal, Not a Failure — It Reveals Model Compression Strategy

- **Challenged assumption**: Feature absorption is undesirable because it reduces recall of general features.
- **Evidence against it**: The position paper (arXiv:2506.23845) argues SAEs are better for discovery than acting on known concepts. From the model's perspective, if "lion" always implies "starts with L," encoding a "lion" feature that subsumes the "starts with L" information is *more* efficient, not less. Absorption may be the SAE correctly recovering the model's actual compression strategy. The model may genuinely not have a separate "starts with L" representation for the token "lion" — it has a "lion" representation that happens to project onto the "starts with L" direction.
- **Contrarian hypothesis**: Absorption is not a failure mode but a faithful recovery of the model's internal hierarchical compression. The interpretability failure lies not in the SAE but in our expectation that models should have flat, non-hierarchical feature sets. By trying to eliminate absorption, the field may be forcing SAEs to learn representations that are *more* interpretable to humans but *less* faithful to the model's actual computations.
- **Exploitation plan**: Behavioral validation experiments: For tokens where absorption is detected (SAE doesn't fire the "general" feature, but "specific" feature fires), test whether the model's downstream behavior is fully explained by the specific feature alone. If yes, absorption is not a failure — the general feature is genuinely redundant in that context. If no, absorption is a real pathology.
- **Novelty estimate**: 8/10 (the framing of absorption as potentially faithful is novel; existing literature uniformly frames it as pathological)

### Candidate C: Absorption Rates Are Task-Specific Artifacts and Are Not Predictive of Real-World Interpretability Failures

- **Challenged assumption**: The 15–35% absorption rates measured on first-letter tasks are representative of absorption severity across feature types, and this drives downstream interpretability failures.
- **Evidence against it**: All published absorption measurements use the first-letter spelling task — a task specifically chosen because it has a clean, symmetric, well-controlled feature hierarchy. Real semantic hierarchies (safety features, knowledge features) are far more imbalanced, contextually rich, and less predictable. "Resurrecting the Salmon" (2025) explicitly states that the absorption metric is not useful for domain-specific evaluation. Furthermore, DeepMind's failure on harmful intent detection was only "about half" explained by absorption-related factors even after controlling for corpus mismatch — suggesting absorption rates from first-letter tasks are poor predictors of the safety-relevant failure.
- **Contrarian hypothesis**: The SAE absorption literature has been studying a narrow, atypical feature hierarchy (letters of the alphabet) that happens to be easy to measure, and the results do not generalize to the semantically rich feature types that actually matter for interpretability applications. The field may be optimizing for the wrong target.
- **Exploitation plan**: Measure absorption rates on a diverse set of feature hierarchies using the SAEBench/sae-spelling infrastructure: (a) alphabet (current gold standard), (b) word frequency classes (common/rare), (c) syntactic role hierarchies (noun/verb/adjective), (d) semantic category hierarchies (animal/mammal/dog), (e) safety-relevant hierarchies (harmful intent/specific harm type). Compare absorption rates across these types and correlate with downstream task performance. If alphabet-derived rates don't correlate with semantically rich hierarchy rates, the standard metric is misleading.
- **Novelty estimate**: 9/10 (this is the most straightforwardly testable claim, with a clear null hypothesis, and fills Gap 2 from the literature survey in a contrarian direction)

---

## Phase 3: Self-Critique

### Against Candidate A (Probe-Based Absorption as Measurement Artifact)

- **Steelman of conventional view**: Chanin et al. do validate the absorption metric causally using integrated-gradients ablation, specifically testing that the probe direction in the absorbing feature is causally responsible for model output. They show that removing the probe direction from the absorbing latent eliminates its ablation effect. This is not purely correlational — there is a genuine causal signal. Furthermore, the metric does not just ask "does the probe fire?" — it uses ablation to confirm the SAE latent is causally important. This is stronger than a simple probe comparison.
- **Cherry-picking check**: The evidence I cited (MP-SAE giving anti-intuitive results, probe noise acknowledgment) represents edge cases. The overwhelming majority of published results using the metric do find consistent patterns across architectures (L1, TopK, JumpReLU, Gated), models (Gemma, Llama, Qwen), and layers. The metric's consistency across such diverse conditions suggests it is measuring something real.
- **Confounding check**: The circularity concern (probe trained on same activations) is partially addressed by the two-stage design: the probe direction is established first, then the SAE latents are evaluated against it. The probe is not trained to maximize separation among SAE latents; it is trained on raw model activations. So the SAE is evaluated against an external reference, not against itself.
- **Actionability check**: Even if I'm right that the metric has noise and measurement artifacts, demonstrating this would require the same controlled experiments that would also confirm genuine absorption cases. The research contribution would be limited to metric validation, not a new insight about interpretability.
- **Verdict**: MODERATE. The probe-based metric has real methodological limitations, but the causal validation addresses the strongest circularity concern. The MP-SAE anomaly is real and unexplained, but it is one data point. The insight that the metric may not generalize to non-L1 architectures is valid but narrow.

### Against Candidate B (Absorption as Faithful Recovery of Model Compression)

- **Steelman of conventional view**: If absorption were faithful recovery, we would expect the model's downstream behavior to be fully explained by the absorbing latent (e.g., the "lion" feature). But Chanin et al. show that there are cases where the SAE probe fails to detect "starts with L" even when the model's behavior clearly depends on this property (e.g., in the first-letter task, the model correctly predicts the letter but the SAE latent doesn't fire). This means the model's computation is not fully explained by the absorbing latent — the model does have a general feature that the SAE is failing to represent separately.
- **Cherry-picking check**: The absorption cases that are easiest to measure are precisely those where we have an independent behavioral test (the first-letter task). These may be atypical in that the behavioral signal is particularly clean. In cases where behavior is more distributed (which is most real-world tasks), the model's use of general vs. specific features may be genuinely ambiguous, making it impossible to determine whether absorption is "faithful" or "pathological."
- **Confounding check**: The model's computation may use both the general feature direction and the specific feature direction simultaneously, with the general feature information encoded in the absorbing latent rather than a separate latent. This would look like "faithful recovery" from the behavioral perspective but would make causal attribution using individual latents impossible — which is the actual problem for interpretability.
- **Actionability check**: If absorption is "faithful," the implication is that we should change how we think about what SAEs should represent, not that we should fix SAEs. This is a conceptual contribution but has fewer actionable implications than Candidate C.
- **Verdict**: WEAK. The steelman is strong. Chanin et al.'s first-letter behavioral test does distinguish cases where the model relies on the general feature but the SAE fails to represent it. Absorption is not purely faithful recovery — there is genuine information loss.

### Against Candidate C (Absorption Rates as Task-Specific Artifacts)

- **Steelman of conventional view**: The first-letter task was chosen precisely because it provides a clean, controllable ground truth. The argument for using it is methodological: you need to know what concept a feature "should" encode before you can measure whether it was absorbed. The clean hierarchy (letter-membership ⊃ token) provides this ground truth. If you use semantically richer hierarchies, you lose the clean ground truth. The fact that semantic hierarchies are harder to measure doesn't mean the first-letter results are invalid — it means they represent a lower bound on the problem.
- **Cherry-picking check**: I cited DeepMind's result that "about half" the harmful intent gap remains after controlling for corpus mismatch. But "about half" still means absorption explains a substantial fraction of the safety failure. I may be downplaying a significant real-world impact by focusing on the residual gap.
- **Confounding check**: The comparison of absorption rates across different feature hierarchies (Candidate C's experiment) has a confound: different feature types activate with different frequencies, have different co-occurrence patterns, and are represented in different layers. Controlling for all these variables while varying feature type is extremely difficult. A null result (no difference in absorption rates across feature types) could be a genuine finding or could reflect insufficient statistical power.
- **Actionability check**: If true, this is highly actionable: it would redirect the entire absorption research program toward semantically rich hierarchies and away from the spelling task. It would also call into question all existing architectural comparisons that use the spelling task as the primary absorption metric.
- **Verdict**: STRONG. The steelman is partially effective (the first-letter task gives a clean ground truth), but the core concern stands: a clean ground truth in a controlled, artificial setting does not guarantee generalizability. The published evidence that semantic domain shifts change SAE behavior (refusal features, domain-specific SAEs) strongly suggests that absorption rates measured on alphabetical tasks may not predict absorption rates in semantically rich domains. This is the most impactful and testable contrarian claim.

---

## Phase 4: Refinement

**Dropped**: Candidate B (Absorption as faithful recovery) did not survive the steelman test. The behavioral evidence from Chanin et al. demonstrates genuine information loss cases, not merely a different representation choice.

**Strengthened Candidate A**: The probe-based measurement artifact concern is real but secondary. The more precise critique: the metric is architecturally biased. For architectures with different sparsity mechanisms (e.g., TopK vs. L1), the mapping between "sparsity benefit from absorption" and "latent activation patterns" differs fundamentally. The MP-SAE negative result provides direct evidence. This should be reformulated as: the absorption metric may systematically mis-rank architectures that use non-L1 sparsity, and any head-to-head architectural comparison using this metric may have confounded results. This is worth flagging as a methodological caveat rather than as a standalone research contribution.

**Strengthened Candidate C**: Combine with the additional corroboration from:
- Resurrecting the Salmon (2025): explicit statement that first-letter metric is "not useful for evaluating domain-specific feature absorption"
- SAEBench itself: "evaluates feature absorption by using features for 'word starts with x'" — a single feature type
- Task-specific probe failure literature: SAE probes generalize worse OOD than linear probes across diverse datasets (Kantamneni et al., 2025; Smith et al., 2025)
- The SynthSAEBench (arXiv:2602.14687) shows SAEs substantially underperform direct probes with controlled synthetic hierarchies — but crucially, this benchmark uses Zipfian firing distributions and correlation structures that differ from the uniform spelling task

The refined contrarian hypothesis for Candidate C: **Absorption rate as measured by the first-letter task is not a reliable predictor of absorption severity in semantically rich or safety-relevant feature hierarchies, because the two hierarchy types differ in frequency imbalance, co-occurrence density, and contextual modulation. Existing architectural comparisons using this metric may select for "absorption-resistant on letters" rather than "absorption-resistant in general."**

**Selected front-runner**: Candidate C, refined.

**Additional corroboration from Phase 3 searches**: The Tian et al. feature sensitivity paper (arXiv:2509.23717) shows that features with good absorption scores (they activate correctly) still have poor sensitivity (they activate inconsistently). This suggests that absorption-focused evaluation misses a related but distinct failure mode. Combined with the cross-domain concern, this strengthens the case that the field's evaluation infrastructure is systematically skewed toward measurable-but-unrepresentative failure modes.

---

## Phase 5: Final Proposal

### Title: When the Ruler Is the Problem: Feature Absorption Rates from Alphabetical Tasks Do Not Predict Absorption in Semantically Rich Hierarchies

**Challenged assumption**: The 15–35% feature absorption rates measured via the first-letter spelling task (Chanin et al., 2024) are representative of the severity and structure of feature absorption across the feature types most relevant to mechanistic interpretability — specifically, semantic concept hierarchies and safety-relevant features.

**Evidence for the conventional view**: The first-letter task provides a clean, well-controlled ground truth (every token has a unique first letter, every letter covers a measurable fraction of the vocabulary). The task has been replicated across multiple models (Gemma 2, Llama 3.2, Qwen 2) and architectures (L1, TopK, JumpReLU, Gated). The absorption rates discovered (15–35%) are consistent across these settings, suggesting the phenomenon is systematic and general.

**Evidence against the conventional view**:
1. "Resurrecting the Salmon" (2025, arXiv:2508.09363) explicitly states: "SAEBench evaluates feature absorption by using features for 'word starts with x', which is not useful for evaluating domain-specific feature absorption."
2. The first-letter task creates a maximally symmetric hierarchy: all 26 letters, each covering ~3–5% of the vocabulary. Real semantic hierarchies are asymmetric: "animal" covers hundreds of thousands of tokens; "pangolin" covers a handful. This frequency imbalance changes the sparsity incentive for absorption fundamentally.
3. DeepMind's harmful intent detection failure was only about half-explained by corpus mismatch; the residual gap persists even in chat-trained SAEs, suggesting absorption in safety-relevant hierarchies is structurally different from alphabet absorption.
4. SAE probes generalize worse OOD than simple linear probes across diverse datasets (Kantamneni et al., 2025; Smith et al., 2025) — but this gap varies across task types in ways the absorption metric does not predict.
5. The Matching Pursuit SAE (explicitly designed for hierarchical features) scores worse on the absorption metric despite being theoretically better suited to handle hierarchy. This suggests the metric is not reliably measuring the construct it purports to measure.

**Hypothesis**: Feature absorption rates in alphabetically organized hierarchies (letter ⊃ token) are systematically lower bounds on absorption rates in semantically imbalanced hierarchies (common concept ⊃ rare concept) because: (a) alphabetical hierarchies have more balanced parent-child frequency ratios (~1:26 split), while semantic hierarchies have more extreme ratios (1:1000+); (b) the sparsity incentive for absorption scales with the frequency imbalance — the more imbalanced, the stronger the absorption pressure; (c) the standard metric cannot detect absorption in unsupervised features (those without known ground-truth labels), which constitute the vast majority of any SAE dictionary.

The corollary is that existing architectural comparisons (Matryoshka vs. OrtSAE vs. ATM SAE) may rank architectures correctly on "absorption in alphabetical tasks" while having no predictive validity for "absorption in safety-relevant semantic hierarchies." The community's current optimization target may be the wrong target.

**Method**: Training-free analysis of existing pre-trained SAEs (Gemma Scope, GPT-2 SAEs via SAELens). Four feature hierarchy types will be tested using the same integrated-gradients ablation framework as Chanin et al.:
1. Alphabetical (first-letter, control): letters ⊃ tokens
2. Frequency class: top-1000 vocabulary items vs. long-tail items
3. Syntactic category: part-of-speech categories ⊃ specific tokens
4. Semantic category: broad categories (animal, country, color) ⊃ specific instances

For each hierarchy type, measure: (a) absorption rate using the Chanin et al. procedure adapted to the relevant probe direction, (b) frequency ratio of parent-to-child concepts, (c) co-occurrence density in the training corpus, (d) downstream probe performance gap between dense linear probe and 1-sparse SAE probe.

**Experimental plan**:

*Pilot (10–15 minutes)*: Using a pre-loaded Gemma Scope SAE on Gemma 2 2B layer 12 (available via SAELens), measure absorption rate for 5 semantic category hierarchies (animal, country, color, profession, vehicle) using the existing sae-spelling codebase adapted to a semantic probe. Compare to the first-letter baseline from published results. Estimate whether the expected direction (semantic > alphabetical absorption rates) is supported.

*Main experiment (≤ 1 hour per hierarchy type)*:
- Load pre-trained Gemma Scope SAE (16k width, layer 12, GPT-2 residual stream)
- For each of the 4 hierarchy types: train LR probes on held-out activation caches, run k-sparse probing to find feature splits, compute absorption rates using integrated-gradients ablation
- Compute frequency imbalance metric for each hierarchy type from the OpenWebText corpus
- Regress absorption rate against frequency imbalance across hierarchy types

*Analysis (training-free)*: Test whether frequency imbalance predicts absorption rate better than hierarchy type alone. If semantic hierarchies show higher absorption rates, this directly challenges the generalizability of the alphabet-derived baselines.

**Baselines**: The standard absorption rate from Chanin et al. (15–35% on Gemma Scope 16k), measured on the same SAE for direct comparison. Additionally, measure feature sensitivity (Tian et al. metric, arXiv:2509.23717) alongside the absorption rate for each hierarchy type to test whether the two metrics diverge across hierarchy types.

**Risk assessment**: If the null hypothesis holds (alphabetical absorption rates do generalize to semantic hierarchies), the positive contribution would be validating the existing metric's generalizability — a useful empirical contribution even as a "negative" result for the contrarian hypothesis. The main risk is that semantic hierarchies are harder to construct in a controlled way, making it difficult to ensure the probe quality is comparable to the alphabet case. This can be mitigated by focusing on highly controllable semantic hierarchies (e.g., using WordNet hypernymy relations to construct ground-truth parent-child pairs) rather than ad-hoc choices.

**Secondary risk**: The finding might be dismissed as "of course semantic hierarchies are harder — that's why we use the spelling task." The key rebuttal is that if the metric doesn't generalize, then the entire literature of architectural comparisons (Matryoshka vs. OrtSAE vs. ATM) is making design decisions that may not carry over to the use cases that motivated the research.

**Novelty claim**: The specific insight that frequency imbalance between parent and child features in a hierarchy is a predictive variable for absorption severity — and that the spelling task's symmetric structure specifically avoids the high-imbalance regime most relevant to safety applications — is entirely absent from the current literature. This would be the first systematic cross-hierarchy absorption study, filling Gap 2 and Gap 7 from the literature survey in a contrarian direction.

**Connections to broader research agenda**: This finding, if confirmed, would reframe the entire absorption mitigation effort. Rather than asking "which architecture minimizes absorption on the spelling task?", the field should ask "which architecture minimizes absorption on imbalanced semantic hierarchies?" This shift may invert the ranking of existing methods (e.g., ATM SAE's 20× improvement on spelling might not translate to semantic hierarchies, or Matryoshka SAE's improvement might be larger on semantic hierarchies than reported).
