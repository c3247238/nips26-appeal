# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption**: Feature absorption is a critical problem that degrades SAE interpretability quality
   - Evidence challenging it: Chanin et al. (2024) defines absorption precisely, but only shows it's "common" (0.49 rate in Matryoshka). Does not prove absorbed features cause downstream interpretability failures.
   - Source: arXiv:2409.14507

2. **Assumption**: Sparsity optimization drives absorption as an unavoidable Pareto trade-off
   - Evidence challenging it: SynthSAEBench (2026) shows MP-SAEs exploit superposition noise without learning true features — absorption may be a training artifact, not an inherent property of sparsity
   - Source: arXiv:2602.14687

3. **Assumption**: Architectural innovations (Matryoshka, HSAE, OrtSAE) meaningfully reduce absorption and improve interpretability
   - Evidence challenging it: SAEBench (2025) reveals proxy metrics (reconstruction, L0) do NOT predict practical performance. Reduction in absorption metric may not correlate with better downstream interpretability.
   - Source: arXiv:2503.09532

4. **Assumption**: Feature sensitivity (Hu et al., 2025) measures something different from absorption
   - Evidence challenging it: Sensitivity declines with SAE width, but nobody has proven low-sensitivity features are the absorbed ones. The correlation between absorption and sensitivity is assumed but unmeasured (Gap 5 in literature).
   - Source: arXiv:2509.23717

5. **Assumption**: Feature consistency across training runs is desirable for reproducibility
   - Evidence challenging it: PW-MCC (Song et al., 2025) emphasizes consistency, but if absorbed features are inconsistent, maybe inconsistency IS the ground truth signal
   - Source: arXiv:2505.20254

6. **Assumption**: Hierarchical feature structures are the correct model of LLM representations
   - Evidence challenging it: Multiple architectures converge on hierarchy as solution (Matryoshka, HSAE, Feature Forest), but this could be confirmation bias. What if superposition is the correct model and we're forcing hierarchy onto it?
   - Source: arXiv:2503.17547, arXiv:2602.11881

7. **Assumption**: Absorption measurement via ablation is the gold standard
   - Evidence challenging it: Ablation only works for layers 0-17 in Gemma 2 2B. Deep attention layers have moved info to final token positions — we literally cannot measure absorption there. The field is optimizing for measurable layers while ignoring unmeasurable ones.
   - Source: literature.md Gap 1

8. **Assumption**: Broader-domain SAEs produce representations vulnerable to absorption and nonlinear error
   - Evidence challenging it: Till et al. (2025) propose domain-specific SAEs as mitigation, but this just shifts the problem — now we need domain-specific everything, and domain boundaries are fuzzy.
   - Source: arXiv:2508.09363

### Landscape of Doubt

The SAE field has a collective blind spot: **absorption is defined and measured by the same ablation-based methodology, but this methodology only works for early layers**. We are:
- Optimizing absorption metrics for layers where we CAN measure it
- Ignoring absorption in deep layers where we CANNOT measure it
- Assuming absorbed features matter for downstream tasks without direct evidence
- Converging on hierarchical architectures without controlled comparison

The community has moved from "visualization-based interpretability" to "metric-based interpretability" — but the metrics themselves may be misleading proxies.

---

## Phase 2: Initial Candidates

### Candidate A: "Ablation-Dependent Absorption Metrics Are a Comforting Illusion"

- **Challenged assumption**: Feature absorption is measurable and comparable across architectures via ablation-based metrics
- **Evidence against**: Ablation only works for layers 0-17 in Gemma 2 2B (Chanin et al.). For deeper layers, we have NO ground-truth absorption measurement. The field is optimizing what it CAN measure while ignoring ~40% of layers.
- **Contrarian hypothesis**: All cross-architecture absorption comparisons (Matryoshka vs JumpReLU vs TopK) are incomplete and potentially misleading because they exclude deep layers. We may be choosing architectures based on incomplete data.
- **Exploitation plan**: Demonstrate that encoder-decoder asymmetry (a training-free signal) correlates with absorption in early layers AND extends to deep layers where ablation fails. Prove that absorbed features in deep layers show distinctive encoder-decoder behavior patterns.
- **Novelty estimate**: 8/10 — nobody has systematically addressed the measurement gap in deep layers

### Candidate B: "Absorption Doesn't Actually Matter for Interpretability Downstream"

- **Challenged assumption**: Reduced absorption → better interpretability → better downstream utility (circuit analysis, model editing, safety)
- **Evidence against**: SAEBench shows proxy metrics don't predict practical performance. Zero papers have demonstrated that absorbed features cause failures in real interpretability tasks (circuit discovery, concept erasure, steering).
- **Contrarian hypothesis**: Absorption is a metric artifact that correlates poorly with actual interpretability failures. Absorbed features may still be useful; absorbed direction may be dominated by "close enough" features with acceptable fidelity.
- **Exploitation plan**: Design controlled experiments: identify absorbed vs non-absorbed features using ablation, then test both on downstream tasks (steering accuracy, circuit completeness, model editing success rate). Compare whether absorbed features systematically fail.
- **Novelty estimate**: 7/10 — this is the elephant in the room nobody has tested

### Candidate C: "The Hierarchy Assumption Is Forced onto Superposition"

- **Challenged assumption**: LLM representations naturally have hierarchical structure, and SAE architectures should model this
- **Evidence against**: Multiple architectures (Matryoshka, HSAE, Feature Forest) independently converge on hierarchy — but convergence on a hypothesis is a sign of confirmation bias, not validation. MP-SAEs exploit superposition noise without learning true features (SynthSAEBench 2026).
- **Contrarian hypothesis**: Superposition (not hierarchy) is the correct model of LLM representations. Forcing hierarchical structure onto superposition may create artificial constraints that actually hurt feature decomposition quality.
- **Exploitation plan**: Train a plain ReLU SAE on SynthSAEBench ground-truth features with known hierarchical structure. Test whether Matryoshka-style nested training actually improves ground-truth feature recovery vs simply fitting better pseudo-labels.
- **Novelty estimate**: 6/10 — meta-analysis of convergence bias

---

## Phase 3: Self-Critique

### Against Candidate A (Ablation-Dependent Illusion)

- **Steelman**: The ablation method is the only ground-truth measurement we have. Encoder-decoder asymmetry is a proxy that could miss subtle absorption. We should optimize what we can measure rather than abandon measurement entirely.
- **Cherry-picking check**: I'm arguing that deep layer absorption matters without direct evidence. Maybe deep layers simply don't have the same absorption patterns (info moved to final token positions means absorption effects are diluted).
- **Confounding check**: The measurement gap in deep layers might not matter if most interpretability work uses early/mid layers (where circuits are more traceable).
- **Actionability check**: If I prove the measurement is incomplete, what do we do? We can't do ablation in deep layers. Encoder-decoder asymmetry might help but isn't proven. This could be a "gotcha" without a solution.
- **Verdict**: MODERATE — the problem is real but actionability is unclear. Needs more work on training-free deep-layer detection before it becomes a full proposal.

### Against Candidate B (Absorption Doesn't Matter)

- **Steelman**: This is the most important question to answer. If absorption doesn't matter for downstream tasks, we're optimizing for the wrong thing.
- **Cherry-picking check**: I'm focusing on the absence of evidence for absorption harm while ignoring the theoretical argument for why absorption SHOULD matter. If specific features are absorbed, steering those features should fail — this is testable.
- **Confounding check**: Maybe absorbed features matter for SOME downstream tasks (safety concept detection) but not others (circuit discovery). Need to test multiple downstream scenarios, not just one.
- **Actionability check**: If absorption doesn't matter, the field should focus on other properties (sensitivity, consistency). If it DOES matter, we need better mitigation. Either way we get a publishable result.
- **Verdict**: STRONG — direct test of the field's core assumption. Could be a "gotcha" paper if I'm right, or a definitive paper if I'm wrong (either result matters).

### Against Candidate C (Forced Hierarchy)

- **Steelman**: The convergence of Matryoshka, HSAE, and Feature Forest on hierarchy is powerful evidence. Three independent teams finding the same structure suggests it's real, not a bias artifact.
- **Cherry-picking check**: I'm cherry-picking the MP-SAE result (exploits superposition) while ignoring the substantial gains from hierarchical SAEs on SAEBench absorption metrics. Maybe MP-SAE exploits noise in the specific benchmark but hierarchy is still correct for real representations.
- **Confounding check**: Convergence could reflect shared training data (OpenWebText) and shared architectural priors, not ground truth.
- **Actionability check**: "Hierarchy is wrong" is a strong claim that needs strong evidence. Current evidence is circumstantial.
- **Verdict**: WEAK — the steelman is strong. Better to test the hierarchy assumption indirectly through Candidate B's downstream experiments.

---

## Phase 4: Refinement

### Dropped
- **Candidate C**: Too weak on steelman. The convergence evidence is compelling. Deferred to background concern.

### Strengthened
- **Candidate B**: High actionability. If I'm right, the field is optimizing wrong metrics. If I'm wrong, we learn exactly what absorption costs. Design cleaner experiments.

### Additional Corroboration for Candidate B
- Search for any paper that directly links absorption to downstream task failure
- Check if any SAE-based steering experiments report differential success based on feature type
- Look for evidence that absorbed safety-relevant features fail safety detection

### Selected Front-Runner
**Candidate B: "Absorption Doesn't Actually Matter for Interpretability Downstream"**

This is the highest-value question because:
1. It directly challenges the field's implicit assumption
2. It has clear experimental design (absorbed vs non-absorbed feature comparison on downstream tasks)
3. Either result is publishable and impactful
4. It's the most "contrarian for insight" — proving the field wrong would reshape research priorities

---

## Phase 5: Final Proposal

### Title
**"Rethinking Feature Absorption: Ablation Metrics Don't Predict Downstream Interpretability Utility"**

### Challenged Assumption
Feature absorption in SAEs (measured by ablation-based absorption rate) degrades the utility of SAE features for downstream interpretability tasks. The field assumes absorbed features are "lost" and must be recovered through architectural innovations.

### Evidence (Honestly Presented)

**For the assumption**:
- Absorption is precisely defined and reliably measured in early layers
- Multiple architectures (Matryoshka, HSAE, OrtSAE) reduce absorption and improve proxy metrics
- Theoretical argument: if feature A is absorbed into feature B, then steering A should fail because the direction is dominated by B

**Against the assumption**:
- SAEBench shows proxy metrics (reconstruction, L0, even absorption rate) do NOT predict practical interpretability performance
- No paper has directly tested whether absorbed features fail downstream tasks
- Absorbed features may still be "close enough" for many applications
- The ablation metric measures encoder behavior, not decoder utility

### Hypothesis
Feature absorption (as measured by ablation rate) is NOT a reliable predictor of downstream interpretability task failure. Absorbed features may remain usable for steering, circuit analysis, and concept detection because:
1. Partial feature activation may be sufficient for many tasks
2. Absorbed direction may still correlate with intended behavior
3. The ablation metric captures encoder sparsity optimization, not decoder fidelity

### Method
1. **Feature classification**: Use ablation-based method on Gemma 2 2B SAEs (layers 0-17) to classify features as "absorbed" vs "non-absorbed"
2. **Downstream task battery**: Test both feature types on:
   - Steering accuracy: Can we shift model behavior by activating/deactivating features?
   - Circuit completeness: Do absorbed features form less complete circuits?
   - Concept erasure: Does ablating absorbed features affect concept detection differently?
   - Model editing: Is targeted editing success rate different for absorbed features?
3. **Controlled comparison**: Match absorbed/non-absorbed features on confounders (activation frequency, magnitude, uniqueness)
4. **Validation**: Replicate on SynthSAEBench ground-truth features with known absorption status

### Experimental Plan

| Task | Model | Duration | Metric |
|------|-------|----------|--------|
| Feature classification (absorbed vs not) | Gemma 2 2B (layers 0-17) | 15 min | Absorption rate threshold |
| Steering accuracy test | GPT-2 small + SAE | 20 min | Behavior change correlation |
| Circuit completeness comparison | GPT-2 small | 20 min | Circuit edge coverage |
| Concept erasure test | GPT-2 small | 15 min | Concealment success rate |

Total pilot time: ~70 minutes (split into 4 independent tasks)

### Baselines
- Absorption rate metric (Chanin et al.)
- SAEBench proxy metrics (reconstruction, L0)
- Feature sensitivity (Hu et al.)

### Risk Assessment
- **Risk if I'm wrong**: Absorbed features DO fail downstream. Then absorption matters, and the field's architectural innovations are justified. This is a definitive negative result that clarifies research priorities.
- **Risk if I'm right**: We stop optimizing for absorption and focus on other metrics. Some downstream tasks may suffer slightly. But we save the community from over-engineering solutions to the wrong problem.
- **Mitigation**: Design experiments to be conclusive either way. Test multiple downstream tasks so we don't just flip the binary — we get granular understanding of when/where absorption matters.

### Novelty Claim
This is the first paper to directly test whether absorption metrics predict downstream utility. All prior work assumes absorption is harmful; none has measured actual downstream task degradation. A negative result (absorption doesn't matter) would be a landmark contribution that reshapes how the field measures SAE quality.