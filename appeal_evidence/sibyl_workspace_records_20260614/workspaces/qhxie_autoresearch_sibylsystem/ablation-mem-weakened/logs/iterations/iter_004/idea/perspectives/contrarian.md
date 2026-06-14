# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

#### 1. Assumption: "Feature absorption is a bug that needs fixing"
**Widely held belief**: Absorption is a failure mode of SAEs that undermines interpretability, and reducing absorption should improve downstream utility.
**Evidence challenging it**:
- Kantamneni et al. (ICML 2025) "Are Sparse Autoencoders Useful?" --- SAEs underperform simple baselines on downstream probing tasks regardless of absorption levels. If SAEs are not useful for downstream tasks, fixing absorption may not matter.
- Wang et al. (ICLR 2026) "Does Higher Interpretability Imply Better Utility?" --- Weak correlation (tau_b ~ 0.3) between interpretability scores and steering utility. Reducing absorption improves interpretability metrics but may not improve practical utility.
- The project's own data: Feature U (24.2% absorption) still achieves 100% steering success. Precision remains 1.0 universally. EC50 shows no efficiency degradation.
- "Sanity Checks for Sparse Autoencoders" (Korznikov et al., arXiv:2602.14111, 2026) --- Frozen/random SAE baselines achieve comparable performance to trained SAEs on Gemma-2-2B and Llama-3-8B across interpretability, sparse probing, and causal editing metrics. If random baselines match trained SAEs, absorption may be an artifact of massive dictionary sizes rather than a meaningful phenomenon.
- DeepMind Safety Research (Sharkey et al., 2025) --- "Negative Results for Sparse Autoencoders on Downstream Tasks and Deprioritising SAE Research." Institutional deprioritization based on disappointing practical results.

#### 2. Assumption: "Decoder correlations reveal competitive suppression"
**Widely held belief**: W_dec^T W_dec captures meaningful competitive dynamics between features, and high decoder correlation indicates problematic feature entanglement.
**Evidence challenging it**:
- "Evaluating and Designing Sparse Autoencoders by Approximating Quasi-Orthogonality" (arXiv:2503.24277, 2025) --- Questions whether strict orthogonality is the right goal; proposes quasi-orthogonality as more realistic. Natural feature structures (days of week, months, hierarchical concepts) are inherently non-orthogonal.
- OrtSAE paper (Korznikov et al., 2025) acknowledges: "If non-orthogonal geometries were widespread, our penalty would force an incorrect basis and degrade reconstruction." They verify this isn't happening but only via reconstruction fidelity --- a weak test.
- "Disentangling Dense Embeddings with Sparse Autoencoders" (arXiv:2408.00657, 2024) --- Mean encoder-decoder cosine similarity of 0.57; encoder and decoder solve fundamentally different problems (discrimination vs. reconstruction). Forcing decoder orthogonality may help interpretability but forces unnatural representations.
- "Base Models Know How to Reason" (arXiv:2510.07364, 2025) --- Found decoder cosine similarities "consistently very high (near 1.0)" across SAE configurations, leading them to abandon decoder orthogonality as a metric for taxonomy independence.
- The LCA-SAE correspondence is exact only for tied-weight SAEs. The project's SAE (gpt2-small-res-jb) uses untied weights. The correspondence is definitional, not empirical --- its scientific value depends entirely on validated predictions (H6).

#### 3. Assumption: "The Local Inhibition Graph provides a novel theoretical contribution"
**Widely held belief**: Connecting LCA neuroscience to SAE absorption is a genuinely novel theoretical insight that strengthens the paper's intellectual contribution.
**Evidence challenging it**:
- The correspondence W_dec^T W_dec = G_LCA is exact by definition for tied-weight SAEs. It is a mathematical identity, not an empirical discovery. Its value depends entirely on whether it generates validated predictions.
- Rozell et al. (2008) LCA has ~2000 citations but zero applications to LLM SAEs. This may indicate the connection is obvious-in-hindsight, not profound.
- "Interpretable Representations in Neural Networks" (arXiv:2503.01824, 2025) critiques that "simple sparse autoencoders cannot achieve optimal code recovery" and notes the "amortization gap" as a critical problem. The LCA framework does not address this gap.
- The "rebranding" critique: The LIG reframes decoder correlations (already computed in many SAE papers) as an "inhibition graph" without adding new computational content.

#### 4. Assumption: "Training-free analysis of pretrained SAEs is sufficient for meaningful contributions"
**Widely held belief**: Since the project is training-free, analyzing existing pretrained SAEs (GemmaScope, GPT-2 SAEs) can yield publishable insights.
**Evidence challenging it**:
- "Sanity Checks" (Korznikov et al., 2026) shows frozen/random baselines match trained SAEs. This raises the stakes for any training-free analysis: if random SAEs look similar to trained ones, training-free analysis must rigorously distinguish signal from artifact.
- "Transcoders Beat Sparse Autoencoders for Interpretability" (Paulo et al., 2025) --- Transcoders achieve Pareto dominance over SAEs. The field may be moving away from SAEs entirely, making SAE-specific absorption analysis less relevant.
- Self-ablating transformers (arXiv:2505.00509, 2025) provide built-in interpretability without external decomposition, challenging the fundamental need for SAEs.
- Neel Nanda's assessment: "Sparse autoencoders are useful but overhyped" --- useful for unknown concept discovery, but "simple probes outperform SAEs" for finding known concepts.

#### 5. Assumption: "Precision-recall asymmetry is a deep finding requiring theoretical explanation"
**Widely held belief**: The observation that absorption affects recall but not precision is a meaningful pattern that the inhibition framework explains.
**Evidence challenging it**:
- Precision = 1.0 only holds at k >= 5. At k=1, precision_mean = 0.897. The "invariance" is parameter-dependent.
- The project's precision metric uses a specific probe construction (max-activating examples). Different probe constructions might yield different precision values.
- The precision-recall decomposition is post-hoc --- it was not pre-registered as a primary analysis. The apparent asymmetry may be a consequence of how the metrics are constructed rather than a deep property of absorption.
- In the LCA framework, inhibition from child to parent should reduce parent activation (recall loss). But inhibition could also cause false positives if the parent fires when the child fires for unrelated inputs. The framework does not rule this out.

#### 6. Assumption: "The field values absorption research"
**Widely held belief**: Feature absorption is an important problem in mechanistic interpretability, and papers addressing it are valued by the community.
**Evidence challenging it**:
- Dan Hendrycks (Nov 2025): "Sparse autoencoders (SAEs): Struggled to compress activations in a meaningful or robust way, with DeepMind reportedly scaling back work on them."
- "Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts" (arXiv:2506.23845, 2025) --- Argues SAEs are misapplied; useful only for discovery, not action.
- "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders" (Wu et al., 2025) --- Simple baselines beat SAEs at steering.
- The project's result debate score: 3/10. The field may view absorption as a solved-or-unsolvable problem that is no longer a priority.

### Landscape of Doubt

The SAE field is experiencing a credibility crisis in 2025-2026. Multiple high-profile negative results (Kantamneni, Wang, Korznikov "Sanity Checks", DeepMind deprioritization, transcoder superiority) challenge whether SAEs are a productive research direction at all. The Local Inhibition Graph proposal risks being a sophisticated answer to a question nobody cares about anymore. If the field pivots to transcoders or self-ablating transformers, an SAE-specific absorption framework becomes intellectually orphaned.

The most damaging critique for this project is the "Sanity Checks" paper: if random SAE baselines match trained SAEs on standard metrics, then absorption may be an artifact of dictionary size and sparsity constraints rather than a meaningful failure of feature learning. Any absorption study must directly address this challenge with random baseline comparisons.

---

## Phase 2: Initial Candidates

### Candidate A: "Absorption is Benign: Feature Absorption as Optimal Compression, Not a Bug"

**Challenged assumption**: Absorption is a failure mode that undermines interpretability and should be reduced.

**Evidence against it**:
- Project's own data: 100% steering success even at 24.2% absorption. Precision = 1.0 universally. No EC50 degradation.
- Wang et al. (ICLR 2026): Weak correlation (~0.3) between interpretability and utility.
- Kantamneni et al. (ICML 2025): SAEs underperform baselines regardless of absorption.
- Rate-distortion theory: Absorption is the optimal strategy for a sparsity-constrained encoder when features are hierarchically correlated. The SAE is doing exactly what it should do.

**Contrarian hypothesis**: Absorption is not a bug but an optimal compression strategy. The SAE correctly identifies that parent features are redundant given child features (due to hierarchical co-occurrence) and suppresses them to maintain sparsity. This is rate-distortion optimal: the parent feature's information is preserved in the child feature's decoder direction, so no information is lost --- only distributed differently.

**Exploitation plan**: Reframe the project's null results (H1-H4 falsified) as positive evidence for the "absorption is optimal compression" hypothesis. The precision-recall asymmetry is explained: precision is preserved because the decoder direction still points to the correct concept (information preserved); recall is reduced because the encoder suppresses redundant activations (optimal compression). The paper becomes "Feature Absorption as Optimal Compression: Evidence that SAEs Correctly Handle Hierarchical Features."

**Novelty estimate**: 7/10. Directly challenges the dominant narrative. Requires careful framing to avoid seeming like a "gotcha" paper.

### Candidate B: "The Emperor's New Graph: Why Decoder Correlations Do Not Reveal Inhibition"

**Challenged assumption**: W_dec^T W_dec captures competitive suppression dynamics, and the Local Inhibition Graph is a meaningful diagnostic tool.

**Evidence against it**:
- Decoder correlations conflate feature composition (bad), natural correlation (neutral/good), and superposition math (inevitable).
- "Base Models Know How to Reason" (2025): Decoder cosine similarities near 1.0 across configurations; authors abandoned orthogonality as a metric.
- The LCA-SAE correspondence is exact only for tied weights. The project's SAE uses untied weights.
- OrtSAE acknowledges: "If non-orthogonal geometries were widespread, our penalty would force an incorrect basis."
- The "rebranding" critique: The LIG adds no new computation beyond what SAE researchers already compute.

**Contrarian hypothesis**: Decoder correlations primarily reflect (a) the overcomplete nature of SAE dictionaries (mathematically inevitable), (b) natural semantic correlations in language (desirable), and (c) reconstruction-driven basis overlap (necessary). They do not primarily reflect competitive suppression. The LIG is a reframing, not a discovery.

**Exploitation plan**: Test H6 (graph edges predict absorption) and show that precision@k is only marginally above chance even after controlling for (a) feature frequency correlation and (b) semantic similarity. Show that a simple baseline using co-occurrence statistics outperforms the inhibition graph. The paper becomes "Decoder Correlations Do Not Reveal Competitive Suppression: A Critical Analysis of the Local Inhibition Graph Hypothesis."

**Novelty estimate**: 6/10. Important if true, but risks being seen as purely destructive without a constructive alternative.

### Candidate C: "SAEs Are the Wrong Target: Absorption is a Symptom, Not the Disease"

**Challenged assumption**: Feature absorption in SAEs is a worthwhile research target.

**Evidence against it**:
- "Sanity Checks" (Korznikov et al., 2026): Random SAE baselines match trained SAEs. Absorption may be an artifact.
- Transcoders Pareto-dominate SAEs (Paulo et al., 2025).
- DeepMind deprioritized SAE research (Sharkey et al., 2025).
- Self-ablating transformers provide built-in interpretability without external decomposition.
- Kantamneni et al.: SAEs underperform baselines on downstream tasks.

**Contrarian hypothesis**: The SAE paradigm itself is flawed, and absorption is merely one symptom of a deeper problem: SAEs reconstruct activations without learning meaningful features. The community should pivot to transcoders or alternative interpretability methods rather than refining SAE diagnostics.

**Exploitation plan**: This is a meta-critique rather than a constructive proposal. Use the project's data to show that (a) absorption rates correlate with dictionary size (artifact signal), (b) random SAEs exhibit similar "absorption-like" patterns, and (c) transcoder features do not exhibit absorption. The paper becomes "Feature Absorption is a Symptom of SAE Obsolescence: Evidence from Random Baselines and Transcoder Comparisons."

**Novelty estimate**: 5/10. Important argument but very risky --- could be dismissed as nihilistic or premature.

---

## Phase 3: Self-Critique

### Against Candidate A ("Absorption is Optimal Compression")

**Steelman**: Chanin et al. (2024) proved absorption is a logical consequence of sparsity loss under hierarchical features. This is a mathematical theorem, not an empirical claim. The "optimal compression" framing is consistent with their proof --- absorption is the SAE's optimal response to the training objective. The project's data (precision = 1.0, steering still works) supports this. Rate-distortion theory provides a principled framework.

**Cherry-picking check**: Am I selectively citing the project's null results while ignoring the one partially significant finding (H1b: r=-0.431 at layer 8, uncorrected p=0.028)? The delta-corrected steering correlation suggests absorption does have some negative effect, at least at layer 8. The "optimal compression" framing must explain this.

**Confounding check**: Could the null results be due to insufficient statistical power rather than genuine absence of effect? The project tested 26 features --- a small sample. With more features, significant correlations might emerge. The "absorption is benign" claim is premature without a power analysis.

**Actionability check**: Even if absorption is optimal compression, this insight is only useful if it guides better SAE design or usage. The "optimal compression" framing suggests practitioners should accept absorption as inevitable --- but this is not actionable advice. It could be seen as a "gotcha" paper that proves the conventional wisdom is wrong without offering a path forward.

**Verdict**: MODERATE. The optimal compression framing is theoretically coherent and supported by evidence. But it risks being a "gotcha" paper unless it offers constructive guidance (e.g., "when to trust absorbed features" or "how to design around absorption").

### Against Candidate B ("Decoder Correlations Do Not Reveal Inhibition")

**Steelman**: The LCA-SAE correspondence is mathematically exact for tied-weight SAEs. Even for untied weights, the decoder correlation matrix is a well-defined quantity that captures geometric relationships between feature directions. If graph edges predict absorption pairs (H6), the framework is validated regardless of interpretive debates about "inhibition." The neuroscience analogy is a communication tool, not the scientific claim.

**Cherry-picking check**: Am I selectively citing papers that question orthogonality while ignoring the empirical success of orthogonality penalties? OrtSAE reduces absorption by 65% using orthogonality constraints. This is strong evidence that decoder geometry matters. The "decoder correlations are meaningless" claim contradicts this empirical success.

**Confounding check**: The high decoder cosine similarities reported in "Base Models Know How to Reason" may be specific to their experimental setup (residual stream SAEs, specific model). JumpReLU SAEs (GemmaScope) or TopK SAEs may have different correlation structures. Generalizing from one setup is risky.

**Actionability check**: If decoder correlations do not reveal inhibition, what do they reveal? A purely destructive critique is less valuable than the LIG's constructive diagnostic tool. The critique needs an alternative interpretation of decoder correlations to be publishable.

**Verdict**: MODERATE. The critique has merit but risks being purely destructive. It would be stronger if paired with an alternative interpretation of decoder correlations (e.g., "decoder correlations measure semantic similarity, not competitive suppression").

### Against Candidate C ("SAEs Are the Wrong Target")

**Steelman**: The SAE field is vibrant and growing. SAEBench (200+ SAEs), GemmaScope (comprehensive pretrained SAEs), and ongoing architectural innovation (Matryoshka, OrtSAE, ATM) show sustained community investment. One negative result (Kantamneni) and one institutional decision (DeepMind) do not invalidate the entire paradigm. Transcoders have their own limitations (different objective, scaling challenges on Gemma 2). The "SAEs are dead" claim is premature.

**Cherry-picking check**: Am I overweighting recent negative results while ignoring positive results? Anthropic's Base64 findings, Golden Gate Claude, and circuit tracing successes demonstrate genuine SAE value. The "Sanity Checks" paper itself frames its findings as contributing to "more rigorous evaluation standards" rather than a final verdict.

**Confounding check**: The project's training-free constraint may bias toward negative results. Training SAEs with absorption-aware objectives (Gap 4 in literature) might yield different conclusions. The "SAEs are wrong" claim may reflect the project's constraints, not SAEs' inherent limitations.

**Actionability check**: This candidate offers no constructive path forward. It is a meta-critique that, even if correct, does not help practitioners. NeurIPS/ICML reviewers would likely reject it as premature and unconstructive.

**Verdict**: WEAK. The evidence is insufficient to declare SAEs obsolete, and the candidate offers no constructive alternative.

---

## Phase 4: Refinement

### Dropped
- **Candidate C (SAEs are the wrong target)**: Verdict WEAK. Insufficient evidence to declare the paradigm obsolete, and no constructive alternative offered.

### Strengthened
- **Candidate A (Absorption is optimal compression)**: Strengthened by incorporating the rate-distortion framework explicitly. The key insight: absorption is not "benign" (which sounds like apologia) but "optimal" (which sounds like principled analysis). The precision-recall asymmetry is explained as information redistribution, not information loss. Added constructive guidance: when absorption is acceptable (high-precision tasks) vs. problematic (high-recall tasks).
- **Candidate B (Decoder correlations do not reveal inhibition)**: Strengthened by proposing an alternative interpretation: decoder correlations measure semantic similarity and basis overlap, not competitive suppression. The critique becomes constructive: "Decoder Correlations Measure Semantic Geometry, Not Inhibition: A Reinterpretation."

### Additional Corroboration
- Searched for "rate distortion sparse coding" and found that the rate-distortion interpretation of sparse coding is well-established (Olshausen & Field, 1996; Rozell et al., 2008). The "absorption as optimal compression" framing connects to this established literature.
- Searched for "decoder correlation semantic similarity" and found that "Disentangling Dense Embeddings" (2024) explicitly links decoder similarity to semantic similarity: "features with lower max similarity to others have more similar encoder-decoder representations." This supports the alternative interpretation in Candidate B.

### Selected Front-Runner
**Candidate A: "Feature Absorption as Optimal Compression"** is selected as the front-runner contrarian position. It is provocative but grounded in evidence, theoretically coherent, and offers constructive guidance. It directly challenges the dominant "absorption is a bug" narrative while explaining the project's null results as positive findings.

---

## Phase 5: Final Proposal

### Title
"Rethinking Feature Absorption: Evidence that SAEs Correctly Compress Hierarchical Features"

Alternative: "Feature Absorption as Optimal Compression: When Sparse Autoencoders Do the Right Thing"

### Challenged Assumption
The dominant narrative in the SAE literature (Chanin et al., 2024; Matryoshka SAE; OrtSAE) frames feature absorption as a failure mode: parent features are "wrongly" suppressed by child features, creating interpretability illusions. This assumption underlies all absorption mitigation research.

### Evidence

**For the assumption (absorption is a bug)**:
- Chanin et al. proved absorption is a logical consequence of sparsity loss under hierarchical features.
- Absorption creates arbitrary false negatives, undermining reliability for high-stakes applications (bias detection, deception monitoring).
- Multiple architectural solutions (Matryoshka, OrtSAE, ATM, Balance Matryoshka) successfully reduce absorption.

**Against the assumption (absorption is optimal compression)**:
- The project's data: 100% steering success at 24.2% absorption; precision = 1.0 universally; no EC50 degradation.
- Wang et al. (ICLR 2026): Weak correlation (~0.3) between interpretability and utility. Reducing absorption improves metrics, not practical outcomes.
- Kantamneni et al. (ICML 2025): SAEs underperform baselines regardless of absorption levels.
- Rate-distortion theory: A sparsity-constrained encoder should suppress redundant activations. Parent features are redundant given child features (due to hierarchical co-occurrence). The SAE is doing exactly what rate-distortion optimal compression requires.
- "Sanity Checks" (Korznikov et al., 2026): Random SAE baselines match trained SAEs, suggesting absorption patterns may be structural (dictionary size + sparsity) rather than learned failures.

### Hypothesis
Feature absorption is not a failure mode but an optimal compression strategy. The SAE correctly redistributes parent feature information into child feature directions to maintain sparsity under hierarchical co-occurrence. Precision is preserved because decoder directions retain the correct semantic content; recall is reduced because the encoder optimally suppresses redundant activations. The interpretability "illusion" is actually an interpretability "trade-off" --- one that the SAE navigates correctly given its constraints.

### Method

**Experiment 1: Random Baseline Comparison**
- Construct random SAE baselines (frozen decoder, frozen encoder) following Korznikov et al. (2026).
- Measure absorption rates on random baselines vs. trained SAE.
- Prediction: Random baselines exhibit absorption-like patterns (structural artifact), but trained SAEs show systematic differences in which features are absorbed.
- Time: ~30 min.

**Experiment 2: Information-Theoretic Analysis**
- Compute mutual information I(parent feature; child feature) for absorption pairs.
- Test whether absorption rate correlates with redundancy (mutual information / parent entropy).
- Prediction: Higher redundancy -> higher absorption rate (optimal compression).
- Time: ~30 min.

**Experiment 3: Rate-Distortion Frontier**
- For a fixed SAE, sweep sparsity levels and measure (reconstruction error, absorption rate, steering utility).
- Plot the Pareto frontier.
- Prediction: Absorption increases as sparsity increases, but steering utility remains on the frontier (no degradation).
- Time: ~30 min.

**Experiment 4: Transcoder Comparison**
- Compare absorption patterns in SAEs vs. transcoders on the same model/layer.
- Prediction: Transcoders (different objective) show different absorption patterns, confirming absorption is objective-dependent, not a universal failure.
- Time: ~1 hour (if transcoder checkpoints available).

### Experimental Plan
All experiments use small models (GPT-2 Small, Gemma-2-2B) and are training-free (analysis of pretrained SAEs). Target <=1 hour per experiment.

### Baselines
1. **Chanin et al. absorption metric**: The mainstream method for detecting absorption.
2. **Random SAE baseline**: Following Korznikov et al. (2026), to test whether absorption is structural or learned.
3. **Non-hierarchical feature control**: Test absorption on features without hierarchical relationships (predicted: low absorption).

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Random baselines show identical absorption patterns | Medium | High | If true, absorption is purely structural. Paper becomes "Absorption is a Structural Artifact, Not a Learned Failure." |
| Information-theoretic analysis shows no redundancy-absorption correlation | Medium | Medium | Would falsify the optimal compression hypothesis. Fallback: descriptive analysis of when absorption occurs. |
| Steering utility degrades at high sparsity | Medium | Medium | Would contradict the "no degradation" claim. Report the Pareto frontier honestly. |
| Community rejects the reframing as apologetics | Medium | High | Frame as "understanding the mechanism" not "defending the status quo." Emphasize constructive guidance. |
| Transcoder comparison unavailable | High | Low | Primary experiments do not depend on transcoders. |

### Novelty Claim
The specific insight is: **feature absorption is rate-distortion optimal compression, not a failure mode.** This reframes the dominant narrative, explains the project's null results, and provides constructive guidance for when absorption is acceptable vs. problematic. The connection to rate-distortion theory grounds the claim in established information theory rather than ad hoc apologia.

---

## Synthesis with Project Context

### How This Contrarian Position Interacts with the Local Inhibition Graph

The contrarian position and the LIG are not mutually exclusive --- they are complementary perspectives:

| Aspect | Local Inhibition Graph | Absorption as Optimal Compression |
|--------|------------------------|-----------------------------------|
| **Framing** | Mechanistic (HOW absorption happens) | Functional (WHY absorption happens) |
| **Core claim** | Decoder correlations predict absorption pairs | Absorption is optimal given sparsity constraints |
| **Valence** | Diagnostic (identify at-risk features) | Reframing (absorption is not a bug) |
| **Constructive contribution** | Training-free diagnostic tool | Guidance on when absorption matters |

If H6 (graph edges predict absorption) succeeds, the LIG provides the mechanism; the optimal compression framing provides the justification. Together: "Absorption is optimal compression, and the inhibition graph identifies where it occurs."

If H6 fails, the optimal compression framing still stands (absorption is structural/inevitable, not predicted by correlations), and the paper pivots to the rate-distortion analysis.

### Integration with Existing Data

The project's existing data directly supports the optimal compression framing:

| Finding | Optimal Compression Explanation |
|---------|--------------------------------|
| Precision = 1.0 universally | Decoder directions preserve semantic content; no information loss |
| Recall varies widely | Encoder optimally suppresses redundant activations |
| Feature U (24.2% abs) still steers 100% | Information is redistributed, not destroyed |
| EC50 shows no efficiency degradation | Compression does not degrade capability, only coverage |
| Delta-corrected correlation at layer 8 | Layer 8 has stronger hierarchical structure = more optimal compression |
| H1-H4 null results | Absorption does not degrade performance because it is optimal |

### Key Warnings for the Project

1. **The "Sanity Checks" challenge must be addressed head-on.** Any paper on SAE absorption in 2026 must include random baseline comparisons. Without them, reviewers will dismiss the work as artifact-chasing.

2. **The precision invariance claim is fragile.** It only holds at k >= 5. At k=1, precision drops to 0.897. Always qualify this claim.

3. **The "exact" LCA correspondence is overstated.** It is exact for tied weights; the project's SAE uses untied weights. Use "approximate" or "motivated by" instead of "exact."

4. **The field is skeptical of SAE value.** A paper that says "absorption is fine actually" risks being seen as apologetics. The framing must be "understanding when and why absorption occurs" not "absorption is not a problem."

5. **The result debate score (3/10) is a warning.** The field may not value absorption research regardless of framing. The paper needs a genuinely novel theoretical or methodological contribution beyond reframing.
