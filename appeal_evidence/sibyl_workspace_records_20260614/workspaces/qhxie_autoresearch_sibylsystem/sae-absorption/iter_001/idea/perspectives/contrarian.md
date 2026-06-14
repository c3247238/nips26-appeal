# Contrarian Perspective (Iteration 3 — Post-Experimental)

## Context

This is a post-experimental contrarian review. Unlike iterations 1-2, which critiqued the proposal before experiments ran, this iteration evaluates the actual results from Phases 0-4 alongside the result debate synthesis (score: 5.5/10). My prior warnings about EDA false positives, cross-domain probe quality, and the diagnostic-only framing have been largely vindicated by the experimental outcomes. This iteration focuses on: what do the actual results tell us, what should the paper claim, and what blind spots remain.

---

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: EDA (encoder-decoder alignment) is a reliable absorption detector**
   - Evidence challenging it: The paper's own results show AUROC passes the 0.65 threshold on only 2/6 Gemma Scope configs (L5-16k: 0.698, L12-16k: 0.776). Two configs show sub-chance AUROC (L12-65k: 0.468, L19-16k: 0.458). The system's own pre-registered go/no-go gate declared NO_GO.
   - Additional: D-EDA, the enhanced variant designed to separate absorption from polysemanticity, underperforms scalar EDA on 5/6 configs. The core theoretical promise — that residual decomposition can distinguish absorption from polysemanticity — is empirically falsified.
   - Sources: Experiment results from Phase 1 validation; result debate synthesis (all 6 perspectives unanimously agree on EDA inconsistency)

2. **Assumption: Feature absorption is a well-defined, cleanly measurable phenomenon**
   - Evidence challenging it: Korznikov et al. (2026, arXiv:2602.14111) show SAEs recover only 9% of true features in synthetic settings, with random baselines matching trained SAEs on interpretability metrics. If SAEs cannot even identify features reliably, the premise that we can precisely detect which features are "absorbed" is questionable.
   - Additional: Kantamneni & Engels (ICML 2025, arXiv:2502.16681) show SAE probes underperform logistic regression in every regime. "Many settings where SAE probes were thought to be helpful turn out not to be when compared to stronger baselines."
   - Sources: [Sanity Checks for SAEs](https://arxiv.org/abs/2602.14111), [Are SAEs Useful?](https://arxiv.org/pdf/2502.16681)

3. **Assumption: Absorption is caused by the sparsity penalty interacting with hierarchical features**
   - Evidence challenging it: Costa et al. (2025, arXiv:2506.03093) demonstrate that MP-SAE — which uses iterative matching pursuit encoding rather than a feedforward encoder — naturally handles hierarchical features without absorption. This implies the root cause is the amortized inference gap (single-pass linear encoding), not the sparsity objective per se.
   - Additional: O'Neill, Gumran & Klindt (ICML 2025) prove SAE encoders are suboptimal sparse decoders — the linear-nonlinear architecture fundamentally lacks the expressivity for optimal sparse recovery of hierarchical features.
   - Sources: [MP-SAE: From Flat to Hierarchical](https://arxiv.org/html/2506.03093v2), [Evaluating SAEs: From Shallow Design to MP](https://arxiv.org/html/2506.05239)

4. **Assumption: Studying absorption in SAEs is still a high-impact research direction**
   - Evidence challenging it: Neel Nanda (Google DeepMind mech interp team lead) stated in September 2025 that SAEs are "overrated" and that "the most ambitious vision of mech interp he once dreamed of is probably dead." DeepMind deprioritized SAE research in March 2025 after finding SAE probes fail catastrophically on safety tasks. Dan Hendrycks (xAI) wrote "the returns from interpretability have been roughly nonexistent."
   - Counter-evidence: Anthropic continues aggressive SAE scaling. SAEs detected emergent misalignment signals. MIT Technology Review named mech interp a "breakthrough technology for 2026." The field is polarized, not dead.
   - Sources: [DeepMind Negative Results](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9), [Neel Nanda Interview](https://forum.effectivealtruism.org/posts/za2oHe8HBtcYNnN7C/neel-nanda-mechanistic-interpretability), [MI 2026 Status Report](https://gist.github.com/bigsnarfdude/629f19f635981999c51a8bd44c6e2a54)

5. **Assumption: The first-letter spelling task is a representative benchmark for absorption**
   - Evidence challenging it: The paper's own cross-domain results show a negative correlation between first-letter absorption rates and RAVEL absorption rates (rho = -0.43 to -0.20). If the supposedly canonical benchmark anti-correlates with richer semantic hierarchies, it is at best non-representative and at worst misleading.
   - Sources: Phase 3 experimental results; result debate synthesis section 2B

6. **Assumption: EDA detects absorption specifically (not just polysemanticity)**
   - Evidence challenging it: The polysemanticity-stratified AUROC of 0.922 (vs. 0.776 overall) at L12-16k is interpretable two ways: either EDA detects absorption better among polysemantic latents, or EDA primarily detects polysemanticity and absorption merely correlates with it. Since D-EDA fails to separate the two signals, the confound remains unresolved.
   - Sources: Phase 1 experimental results; methodologist and skeptic debate contributions

7. **Assumption: Cross-domain absorption exists and is meaningful**
   - Evidence challenging it: RAVEL probes were trained on the wrong model (Qwen2.5-0.5B instead of Gemma 2B). Probe accuracy was below the pre-registered 85% gate. The absolute absorption rates (0.11% continent, 1.75% country) are at the noise floor. The high intra-RAVEL coherence (rho = 0.924) could reflect systematic probe failure patterns rather than genuine absorption structure.
   - Sources: Phase 3 experimental results; methodologist's probe-quality analysis

8. **Assumption: Training-free correction (ITAC) is feasible for absorption**
   - Evidence challenging it: ITAC achieves 3.14% mean FN reduction vs. the 20% pre-registered target. 75% of absorbed latents are "early" type (decoder-absent), structurally ineligible for ITAC. This is a 7x miss on the primary hypothesis.
   - Sources: Phase 2 ITAC experimental results; revisionist's structural analysis

### Landscape of Doubt

The experimental results have revealed a fundamental tension at the heart of this research: **the phenomenon we aim to diagnose (absorption) may not be cleanly separable from the tool we use to diagnose it (SAEs) or from the related phenomena (polysemanticity, probe failure, model mismatch) that contaminate every measurement**.

The broader field landscape reinforces this concern. SAE research is in a legitimacy crisis: the dominant tool cannot beat simple baselines on downstream tasks, its features are inconsistent across training runs, and the most prestigious mech interp team has publicly walked away from it. Against this backdrop, a paper that diagnoses a specific failure mode of SAEs must contend with the meta-question: **does it matter how exactly SAEs fail if they fail broadly?**

The most provocative finding from the experiments — the negative cross-paradigm correlation — suggests that feature absorption may be fundamentally context-dependent. What looks like absorption on first-letter tasks may not look like absorption on semantic tasks, and vice versa. This undermines the notion of "absorption" as a unified phenomenon with a single underlying mechanism.

---

## Phase 2: Initial Candidates

### Candidate A: "The Amortization Gap Thesis — Absorption is an Encoding Problem, Not a Sparsity Problem"

- **Challenged assumption**: The dominant narrative (Chanin et al. 2024, Tang et al. 2025) frames absorption as a consequence of sparsity optimization interacting with hierarchical features. All proposed architectural mitigations (OrtSAE, Matryoshka, masked regularization) target the sparsity-hierarchy interaction.
- **Evidence against**: MP-SAE (Costa et al. 2025) uses standard sparsity objectives but replaces the feedforward encoder with iterative matching pursuit — and avoids absorption on hierarchical features. The "Broken Latents" LessWrong post shows tied SAEs (encoder = decoder transpose) solve absorption in overcomplete settings. O'Neill et al. (ICML 2025) prove SAE encoders are provably suboptimal for sparse recovery. Together: the real culprit is the single-pass linear encoder, not the sparsity penalty.
- **Contrarian hypothesis**: Feature absorption is primarily an amortization gap artifact. The feedforward encoder cannot select the optimal sparse code for hierarchical features in a single pass. Replace the encoder with any iterative scheme (matching pursuit, iterative thresholding, or even just 2-3 encoder passes with residual feedback) and absorption largely disappears — regardless of the sparsity objective.
- **Exploitation plan**: Compare standard SAE, tied SAE, and MP-SAE on identical dictionaries and sparsity levels across first-letter and RAVEL hierarchies. If MP-SAE shows near-zero absorption while standard SAE shows 15-35%, the amortization gap thesis is confirmed. This would redirect the entire absorption mitigation literature from architectural changes (OrtSAE, Matryoshka) to encoding changes (iterative inference).
- **Novelty estimate**: 8/10

### Candidate B: "Absorption is Optimal Compression — Rethinking the Failure Narrative"

- **Challenged assumption**: The field treats absorption as a failure mode to be mitigated. Every paper on the topic frames it negatively.
- **Evidence against**: Chanin et al. themselves note that eliminating absorption worsens the L0-FVU tradeoff. Bereska et al. (2025, arXiv:2512.13568) frame superposition through rate-distortion theory — absorption saves bits without increasing reconstruction distortion. The paper's own finding that 75% of absorption is "early" type (decoder-absent) suggests that for the majority of cases, the SAE never allocated capacity to the parent feature because it was informationally redundant given the child features. This is not a failure — it is efficient compression.
- **Contrarian hypothesis**: For most downstream tasks, absorption does not degrade performance because the "absorbed" information is recoverable from the child features that absorbed it. The cases where absorption is genuinely harmful are narrow (1-sparse probing, exact feature attribution) and reflect a mismatch between the evaluation metric and the SAE's actual representational strategy.
- **Exploitation plan**: Measure whether "absorbed" parent information is recoverable from child feature activations via a simple linear readout. If a probe on child-feature activations can predict the parent concept with >90% accuracy, then absorption is lossless compression, not information loss. Compare SAE performance on causal editing tasks with and without absorption correction — if correction doesn't help, absorption is benign.
- **Novelty estimate**: 7/10

### Candidate C: "When Does Absorption Actually Matter? A Downstream Impact Calibration"

- **Challenged assumption**: The current paper (and the entire absorption literature) measures absorption in isolation without connecting it to downstream consequences.
- **Evidence against**: DeepMind's negative results show SAE probes fail on safety tasks. But is absorption the reason? Nobody has quantified the causal chain: absorption → feature loss → downstream task degradation. It is entirely possible that SAEs fail for other reasons (polysemanticity, insufficient width, domain mismatch) and absorption is a minor contributor.
- **Contrarian hypothesis**: Absorption accounts for less than 20% of the performance gap between SAE probes and dense linear probes on downstream safety tasks. The dominant failure modes are insufficient coverage (dead features), polysemanticity, and training-distribution mismatch — not absorption specifically.
- **Exploitation plan**: On a safety-relevant downstream task (e.g., harmful intent detection using RAVEL-style probing), decompose the SAE-vs-linear-probe performance gap into components attributable to: (a) absorption, (b) dead features, (c) polysemanticity, (d) residual. Use ITAC to correct for absorption and measure the gap reduction. If ITAC-corrected SAE probes still lose to linear probes by a large margin, absorption is not the bottleneck.
- **Novelty estimate**: 9/10

---

## Phase 3: Self-Critique & Adversarial Testing

### Against Candidate A (Amortization Gap Thesis)

- **Steelman for the conventional view**: Tang et al. (2025) prove that absorption corresponds to spurious local minima of the biconvex loss, which is a property of the optimization landscape — not the encoder. Even with an optimal encoder, if the decoder has converged to an absorption-compatible configuration, the optimal sparse code on that decoder will still exhibit absorption-like behavior. The issue may be in the learned dictionary, not the encoding.
- **Cherry-picking check**: MP-SAE has been validated primarily on synthetic data and small models (MNIST, toy hierarchies). Real LLM hierarchies may be qualitatively different. The MP-SAE ICML paper does not report the Chanin et al. absorption metric on first-letter tasks, so the claim of "near-zero absorption" is extrapolated, not directly measured.
- **Confounding check**: MP-SAE uses a different dictionary learning procedure (not just a different encoder). The reduced absorption could come from the dictionary, not the encoding scheme.
- **Actionability check**: If true, the implication is clear — use iterative encoding methods. But MP-SAE is 10-50x slower than standard SAE encoding, which may be prohibitive for large-scale deployment.
- **Verdict**: MODERATE. The amortization gap is a plausible contributing factor but likely not the sole cause. The confound between dictionary quality and encoder quality needs to be resolved by using the SAME dictionary with different encoding methods.

### Against Candidate B (Absorption as Optimal Compression)

- **Steelman for conventional view**: Even if parent information is recoverable from child activations, this requires the downstream user to KNOW that recovery is needed and to implement the readout. The entire point of SAE features is that they should be individually interpretable. "Feature X fires on concept Y" is the core promise. "Feature X fires on concept Y unless concept Z is also present, in which case you need to read out Y from Z's activation" fundamentally undermines the interpretability proposition.
- **Cherry-picking check**: The 75% early absorption finding comes from RAVEL with wrong-model probes. The true rate of early vs. late absorption on properly validated hierarchies is unknown.
- **Confounding check**: "Recoverability" via a linear readout on child features is a lower bar than monosemantic feature firing. The interpretability community needs features that fire on their concept, not features from which the concept can be linearly decoded — that's just a linear probe with extra steps.
- **Actionability check**: If true, the implication is "stop trying to fix absorption." This is actionable but potentially harmful if absorption does cause subtle failures in specific critical applications (e.g., deception detection where a deception-adjacent feature absorbs the deception signal).
- **Verdict**: MODERATE. The compression perspective is theoretically sound but misses the interpretability contract. However, it is a valuable reframing that the paper should explicitly address.

### Against Candidate C (Downstream Impact Calibration)

- **Steelman for conventional view**: Measuring absorption's downstream impact requires fixing absorption (via ITAC or similar) and measuring the improvement. But ITAC achieves only 3% FN reduction — too weak to cause a measurable downstream difference. The experiment is therefore unfalsifiable: a null result could mean absorption doesn't matter OR that ITAC doesn't work.
- **Cherry-picking check**: I'm selecting DeepMind's negative results as my primary target, but there are other downstream tasks where SAEs outperform baselines (Templeton et al. 2024 found safety-relevant features in Claude 3 Sonnet that enabled successful steering). The failure may be task-specific, not SAE-generic.
- **Confounding check**: Decomposing a performance gap into "absorption vs. polysemanticity vs. dead features" requires disentangling highly correlated failure modes, which may be infeasible in practice.
- **Actionability check**: High — this directly answers "does absorption matter?" But the experimental design is complex and may exceed the 1-hour-per-experiment budget.
- **Verdict**: STRONG as a research question. WEAK as a feasible experiment within this paper's scope. The ITAC insufficiency makes the causal isolation impractical.

---

## Phase 4: Refinement

### Dropped

- **Candidate C** (downstream impact calibration) is dropped as a standalone proposal. The ITAC failure means we lack a clean tool to isolate absorption's causal effect on downstream performance. However, the core insight — that the paper measures absorption without measuring its consequences — should be carried forward as a framing critique.

### Strengthened

**Candidate A (Amortization Gap Thesis)** — strengthened and refined:

The key experiment is a **controlled dictionary comparison**: take the SAME pre-trained decoder dictionary from a Gemma Scope SAE and run three encoding methods on the same data:
1. Standard feedforward encoder (original SAE)
2. Orthogonal matching pursuit on the decoder dictionary (no learned encoder)
3. 2-pass encoder: standard encoder followed by residual correction
Measure absorption rates under all three encoders with identical dictionaries. This eliminates the decoder-quality confound. If absorption drops by >50% with iterative encoding at the same sparsity level, the amortization gap thesis is confirmed.

**Candidate B (Absorption as Compression)** — strengthened into a secondary contribution:

Add a "recoverability analysis" to the cross-domain anatomy: for each absorbed parent feature, compute the mutual information between the parent concept and the activations of the absorbing child features. If I(parent; child_activations) is high (comparable to I(parent; parent_activations) in the non-absorbed case), then absorption is approximately lossless. Report this alongside the absorption rates to contextualize whether absorption is truly an information loss problem or merely a format problem (information present but not in the expected latent).

### Selected Front-Runner

**Candidate A is the front-runner**, with elements of Candidate B as secondary analysis.

The amortization gap thesis is the most provocative, evidence-based contrarian position that can be supported by the current experimental results and recent literature. It redirects the field from "fix the sparsity objective" to "fix the encoder," which has concrete and actionable implications.

---

## Phase 5: Final Proposal

### Title

**Rethinking Feature Absorption: Evidence that Amortized Encoding, Not Sparsity, Drives the Dominant Failure Mode**

### Challenged Assumption

The dominant narrative treats feature absorption as a consequence of the sparsity penalty optimizing for hierarchical feature compression. All major mitigations (OrtSAE, Matryoshka SAE, KronSAE, masked regularization) target the sparsity-hierarchy interaction by modifying the loss function, architecture, or training procedure.

### Evidence (For and Against the Mainstream View)

**Evidence for the conventional view (sparsity causes absorption):**
- Chanin et al. (2024): formal proof that delta-absorption monotonically decreases sparsity loss for hierarchical feature pairs
- Tang et al. (2025): unified biconvex framework proves absorption corresponds to spurious partial minima of the SAE loss
- SAEBench (2025): TopK and JumpReLU SAEs (stronger sparsity enforcement) show more absorption than L1 SAEs
- All SAE architectures tested show absorption, regardless of encoder details, as long as sparsity is enforced

**Evidence against (amortization gap is the real cause):**
- MP-SAE (Costa et al. 2025): standard L1 sparsity but iterative encoding; avoids absorption on hierarchical features
- "Broken Latents" (Chanin et al., LessWrong Dec 2024): tied SAEs (encoder = decoder.T) eliminate absorption in overcomplete settings — tying is an encoder constraint, not a sparsity constraint
- O'Neill et al. (ICML 2025): prove SAE encoders are provably suboptimal sparse decoders; the amortization gap is formally characterized
- The paper's own ITAC failure: ITAC attempts to correct at inference time by re-encoding with residual feedback — an extremely crude version of iterative encoding. Even this crude version shows directional improvement, suggesting the encoding bottleneck is real
- The 75% early absorption rate: early absorption means the parent feature's decoder direction was never learned — this could reflect the encoder's inability to separate parent from child during the joint encoder-decoder optimization
- The EDA metric itself measures encoder-decoder misalignment. If absorption were purely a decoder/loss problem, there would be no reason for the encoder to misalign. The fact that EDA has signal proves the encoder is involved.

### Hypothesis

Feature absorption is primarily caused by the amortization gap in single-pass feedforward SAE encoders, which cannot compute the optimal sparse code for hierarchical features. The sparsity penalty is a necessary but not sufficient condition; it creates the incentive for absorption, but the encoder limitation is what makes absorption the actual outcome. Under iterative encoding, the same sparsity penalty produces substantially less absorption because the encoder can sequentially resolve parent-child ambiguity.

### Method

**Experiment 1: Controlled Dictionary Encoding Comparison** (target: 1 hour)
1. Load a Gemma Scope SAE (L12, 16k) with its pre-trained decoder dictionary D.
2. Run three encoding methods on 10,000 tokens from OpenWebText:
   - Standard: original feedforward encoder
   - OMP: orthogonal matching pursuit on D (no learned encoder, same sparsity K)
   - 2-Pass: standard encoder + one residual correction pass
3. Compute absorption rates using adapted Chanin et al. metric for all three encoders with the same D.
4. Compute EDA-equivalents for OMP and 2-Pass (encoder direction = selected dictionary atoms).
5. Statistical test: paired Wilcoxon signed-rank on per-letter absorption rates.

**Experiment 2: Recoverability Analysis** (target: 30 min)
1. For each absorbed parent feature identified in Phase 3:
   - Extract the set of child features that are active when the parent should fire but does not
   - Train a simple 1-layer probe: parent_concept ~ child_activations
   - Report recovery accuracy vs. the original parent probe accuracy
2. Compute I(parent; child_activations) / I(parent; original_probe_activations) as a "compression efficiency" ratio
3. If ratio > 0.9 for majority of cases: absorption is approximately lossless compression

**Experiment 3: Cross-Paradigm Absorption Coherence** (target: 30 min, uses existing data)
1. Use the negative cross-paradigm correlation (first-letter vs RAVEL rho = -0.43) as evidence that absorption is context-dependent, not a stable SAE property
2. Compute absorption rate variance across hierarchies for each SAE config
3. If intra-SAE variance across hierarchies > inter-SAE variance within the same hierarchy: absorption is hierarchy-dependent (supporting the encoding hypothesis — different hierarchies challenge the encoder differently)

### Baselines

- Standard SAE absorption rates (from existing Chanin et al. measurements)
- MP-SAE published results on hierarchical features (if available; otherwise defer as related work)
- Tied SAE "Broken Latents" results as a reference point for the encoder-tying ablation

### Risk Assessment

**If the mainstream view turns out to be correct after all:**
- The controlled dictionary experiment would show absorption rates are identical across encoding methods (standard, OMP, 2-Pass) at the same sparsity level. This would be a clean null result confirming that absorption is a pure loss-landscape property.
- This null result would itself be a valuable contribution — it would rule out the encoder as a cause and focus the field firmly on decoder-side and loss-side mitigations.
- The recoverability analysis and cross-paradigm analysis are independent of the encoding hypothesis and provide value regardless.

**If the encoding hypothesis is confirmed:**
- Immediate practical implication: use 2-pass or iterative encoding for deployed SAEs when absorption matters (safety auditing, circuit analysis)
- Theoretical implication: Tang et al.'s biconvex analysis describes the loss landscape correctly but the landscape is navigated differently by different encoders — the spurious minima are avoidable with iterative methods
- Mitigation implication: OrtSAE and Matryoshka may be partially solving the wrong problem; the simpler fix is to improve the encoder

### Novelty Claim

The specific insight is: **absorption can be separated into a "loss incentive" component (driven by sparsity + hierarchy, well-characterized by existing theory) and an "encoder realizability" component (driven by the amortization gap, not yet characterized)**. No prior work performs the controlled dictionary experiment that disentangles these two components. The recoverability analysis is the first to ask whether absorption is lossy or lossless compression. The cross-paradigm coherence analysis is the first to test whether absorption is a stable SAE property or a hierarchy-dependent encoding failure.

---

## Critical Assessment of the Current Proposal

Given the actual experimental results, here are my updated ratings and recommendations for the paper as it currently stands:

### 1. EDA/D-EDA: 4/10 (downgraded from 6 in pre-experimental estimate)

**What happened**: The pre-registered go/no-go gate failed. EDA passes on 2/6 configs. D-EDA fails everywhere. The polysemanticity confound is unresolved.

**What it should be in the paper**: A supplementary analysis section, not a main contribution. The theoretical lower bound is sound and can be presented as a theorem, but the empirical validation does not support "EDA as a practical absorption detector." The honest framing: "We derive a weight-based lower bound on absorption and show it has significant signal in favorable regimes (mid-layer, narrow SAEs) but insufficient discriminative power for general deployment."

### 2. Cross-Domain Characterization: 7/10 (stable)

**What happened**: Absorption exists in RAVEL hierarchies (all 18 measurements above 3x random baseline; intra-RAVEL rho = 0.924). But probes were on the wrong model, and absolute rates are unreliable.

**What it should be in the paper**: The primary contribution, presented as an existence proof with strong coherence evidence and an honest caveat about probe model mismatch. The negative cross-paradigm correlation (first-letter vs. RAVEL) should be presented as a genuinely surprising finding rather than buried, because it challenges the assumption that absorption is a unified phenomenon.

### 3. Three-Subtype Taxonomy: 5/10 (downgraded from 6)

**What happened**: The late > early EDA ordering holds. The partial subtype ordering fails. The 75% early absorption rate is the key finding.

**What it should be in the paper**: Refocus the taxonomy around the early/late distinction only (drop "partial" or acknowledge it as a data limitation). The headline finding is that 75% of absorption is decoder-absent (early), which has major implications: it means most absorption cannot be fixed at inference time and requires retraining or architectural changes. This is a more impactful framing than the taxonomy itself.

### 4. ITAC: 2/10 (severely downgraded)

**What happened**: 3% FN reduction vs. 20% target. Structurally inapplicable to 75% of cases.

**What it should be in the paper**: Either dropped entirely or presented in one paragraph as a negative result that confirms the dominance of early absorption. Do not present ITAC as a contribution.

### 5. Overall Paper Viability: 5/10

The paper needs a major restructuring around the findings that actually hold:
1. **Cross-domain absorption exists** (primary)
2. **Most absorption is early-type** (secondary, high-impact practical finding)
3. **EDA has theoretical grounding but limited practical power** (theoretical contribution)
4. **First-letter absorption does not predict RAVEL absorption** (surprising negative finding)

The ambitious four-contribution framing should be simplified to a two-contribution framing with clean, defensible claims.

---

## Key References

| Reference | Source | Critical Point |
|-----------|--------|---------------|
| Costa et al. (2025) | arXiv:2506.03093 | MP-SAE avoids absorption via iterative encoding |
| O'Neill, Gumran & Klindt (2025) | ICML 2025 | SAE encoder provably suboptimal; amortization gap characterized |
| Korznikov et al. (2026) | arXiv:2602.14111 | SAEs recover 9% of features; random baselines competitive |
| Kantamneni & Engels (2025) | arXiv:2502.16681 | SAE probes underperform LR in every regime |
| DeepMind Safety Team (2025) | Medium blog post | SAE research deprioritized; linear probes outperform |
| Nanda (2025) | EA Forum interview | SAEs "overrated"; ambitious mech interp vision "probably dead" |
| Chanin et al. (Dec 2024) | LessWrong "Broken Latents" | Tied SAEs eliminate absorption in overcomplete settings |
| Tang et al. (2025) | arXiv:2512.05534 | Absorption as spurious local minima of biconvex loss |
| Bereska et al. (2025) | arXiv:2512.13568 | Superposition as lossy compression; absorption saves bits |
| Michaud et al. (2025) | arXiv:2509.02565 | Feature manifolds; nonlinear feature structures in LLMs |
| Song et al. (2025) | arXiv:2505.20254 | SAE features inconsistent across training runs |
| Narayanaswamy et al. (2026) | arXiv:2604.06495 | Masked regularization — latest mitigation, field moving fast |
| Casper (2024) | LessWrong | SAE work relies on "streetlight demos" without competitive baselines |
| Hendrycks (2025) | Essay | "Returns from interpretability have been roughly nonexistent" |
