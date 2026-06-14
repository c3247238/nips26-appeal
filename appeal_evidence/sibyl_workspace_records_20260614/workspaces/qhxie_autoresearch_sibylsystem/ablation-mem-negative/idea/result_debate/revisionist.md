# Revisionist Analysis: What the Data Tells Us About Our Mental Model

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|------------|---------|--------------|------------|
| **H1**: UAD detects absorption with F1 >= 0.6 | **Confirmed** | F1=0.725 (full), F1=0.704 (pilot). Precision=0.543, Recall=1.0. Exceeds threshold by +0.104. | High |
| **H2**: Cross-model generalization (F1 >= 0.55) | **Inconclusive** | Blocked: Gemma-2B gated, Pythia SAE unavailable. Zero evidence. | N/A |
| **H3**: DFDA recovers >10% with <0.01% params | **Partially Confirmed** | 8/8 pairs positive, 99.5% mean improvement, 0.004% param ratio. BUT metric is artifactual (near-zero prediction). | Low |
| **H4**: End-to-end pipeline improves probing | **Not Tested** | No end-to-end experiment executed. | N/A |
| **H-S1**: Collision rate correlates with true absorption (r > 0.5) | **Inconclusive** | True absorption validation (E9) not executed. Cannot assess. | N/A |
| **H-S2**: True absorption negatively correlates with probing (r < -0.3) | **Inconclusive** | Requires H-S1 first. No data. | N/A |
| **H-E1**: Semantic hierarchy generalization | **Not Tested** | WordNet experiments not executed. | N/A |
| **H-E2**: Multi-concept "super-absorbers" | **Tentatively Supported** | Feature 18486 absorbs 5 letters (c, i, o, p, u). Cluster analysis suggests multi-parent patterns. | Low |
| **H-E3**: Absorption severity spectrum | **Not Tested** | UAD confidence scores not correlated with suppression severity. | N/A |

**Inconclusive resolution**: H2 requires cross-model validation (Gemma-2B, Pythia-2.8B). H3 requires a non-artifactual metric. H4 requires end-to-end pipeline execution. H-S1/H-S2 require Chanin protocol implementation.

---

## 2. Surprise Analysis: Results That Deviate >20% from Expectations

### Surprise 1: The Original Primary Hypotheses (H1-H4 CAAB) Were All Noise

**Deviation**: Expected strong signals for cross-architecture absorption variation, causal downstream links, sparsity monotonicity, and layer patterns. Got |r| < 0.11, p > 0.86 across the board.

**Wrong assumption**: We assumed that "collision rate" was a meaningful proxy for feature absorption, and that systematic variation across architectures/sparsities/layers would emerge. The data shows these are either underpowered (n=5-6) or the proxy metric itself is flawed.

**What this tells us**: Our mental model conflated "feature collision" (multiple concepts sharing a latent) with "feature absorption" (parent feature suppressed when child co-occurs). These are structurally different phenomena. Collision rate measures polysemanticity; absorption measures hierarchical suppression. The 4x difference between JumpReLU and TopK (15.4% vs 3.8%) is real but its interpretation as "absorption variation" was unsupported.

### Surprise 2: The Exploratory Hypotheses (UAD/DFDA) Outperformed the Primary Ones

**Deviation**: UAD and DFDA were explicitly marked as "exploratory" with lower expected contribution. UAD achieved F1=0.725 (vs 0.6 threshold), and DFDA achieved 99.5% on 8 pairs (vs 10% threshold). Both exceeded thresholds by massive margins.

**Wrong assumption**: We assumed the primary contributions (CAAB cross-architecture benchmark, causal assessment) were the safer, more established directions, while UAD/DFDA were high-risk gambles. The opposite was true.

**What this tells us**: The research planning process overweighted "systematic comparison" (which sounds rigorous) and underweighted "novel method development" (which sounds risky). The lesson: in a field with immature metrics (like SAE absorption), novel methods that address fundamental gaps may be more robust than systematic comparisons built on shaky proxies.

### Surprise 3: DFDA's 99.5% Improvement Is Artifactual, Not Real

**Deviation**: Expected meaningful MSE improvement reflecting true absorption recovery. Got near-100% improvement because the MLP learns to predict near-zero parent values in child-dominant positions.

**Wrong assumption**: We assumed that "MSE improvement on parent feature activation" was a valid proxy for "absorption compensation." The baseline MSE is high because the parent feature is naturally near-zero in child-dominant positions; predicting near-zero trivially achieves massive percentage improvement.

**What this tells us**: Our mental model of what DFDA "compensates" was wrong. DFDA does NOT recover absorbed parent activations on examples where the parent should fire. It predicts near-zero values on examples where the parent is already near-zero. The metric was measuring the wrong thing. This is a classic case of optimizing a proxy that diverges from the true objective.

### Surprise 4: Perfect Multi-Seed Consistency (std = 0.000)

**Deviation**: Expected some variance across seeds (typical ML methods show std > 0.05). Got identical F1=0.725 across all 3 seeds with zero variance.

**Wrong assumption**: We treated seed variation as a robustness test. In reality, the SAE is fixed (not retrained per seed), and UAD operates on the co-occurrence matrix derived from the SAE's activations on a fixed corpus. The "seed" only affects the random subset of tokens analyzed, and 15,000 tokens is sufficient to stabilize co-occurrence statistics.

**What this tells us**: UAD's determinism is a feature, not a bug -- but it is determinism given a fixed SAE, not robustness to SAE variation. The robustness claim was overstated. The correct framing is: "UAD is stable across sampling variation for a fixed SAE," not "UAD is robust across random initializations."

### Surprise 5: The Proposal's Contribution Hierarchy Was Completely Inverted

**Deviation**: Original proposal positioned CAAB and causal assessment as primary contributions (confidence: high), with UAD/DFDA as exploratory (confidence: medium). The evidence inverts this: UAD/DFDA are the only viable contributions; CAAB/causal are noise.

**Wrong assumption**: We assumed that "systematic benchmark" and "causal analysis" were inherently more rigorous and publishable than "novel method." This reflected a bias toward breadth over depth.

**What this tells us**: In early-stage research, the value of a systematic comparison depends entirely on the validity of the metrics used. If the metrics are proxies (not ground truth), systematic comparison amplifies noise rather than signal. Novel methods that address metric validity (UAD eliminates the supervised requirement) may be more valuable even with narrower validation.

---

## 3. Mental Model Revision

**Revision 1**: We assumed that "feature collision" (shared latents across concepts) and "feature absorption" (parent suppression under child co-occurrence) were closely related phenomena measurable by the same proxy. The data suggests they are distinct: collision rate varies with dictionary size and training data (pilot: 30.8% at d_SAE=3,072; full: 3.8% at d_SAE=16,384), while absorption is a specific hierarchical relationship detectable via co-occurrence conditional probability. The 8x pilot-full difference in collision rate was never explained in the original analysis.

**Revision 2**: We assumed that DFDA's MSE improvement metric measured "absorption recovery." The data shows it measures "near-zero prediction in child-dominant positions." The MLP learns to output near-zero because that is the dominant value of the parent feature when the child fires. True absorption recovery would require measuring parent feature activation on examples where (a) the parent SHOULD fire (ground truth), (b) the child DOES fire, and (c) the parent does NOT fire in the baseline SAE. DFDA has not been tested on this scenario.

**Revision 3**: We assumed that research contributions should be ranked by "systematicness" (benchmarks > methods). The data shows that in a field with immature ground truth, a novel method that eliminates a fundamental limitation (supervision requirement) can be more valuable than a systematic comparison built on proxy metrics. The CAAB experiments produced noise because the proxy metric (collision rate) was not validated against true absorption. UAD succeeded because it addressed a well-defined gap with a clear success criterion (F1 vs supervised labels).

---

## 4. Reframing Test

**Original research question**: "Can we systematically quantify feature absorption across SAE architectures and establish its causal impact on downstream interpretability?"

**Would we frame it the same way today?** No.

**Revised research question**: "Can feature absorption be detected without ground-truth parent features or supervised probe directions, and if so, can absorbed parent activations be recovered at inference time without retraining?"

**Why the reframing**: The original question assumed (a) absorption is quantifiable via proxy metrics (collision rate), and (b) causal impact is measurable with existing tools. Both assumptions were falsified by the data. The revised question focuses on what the data actually supports: unsupervised detection (UAD works) and training-free compensation (DFDA is conceptually sound but metrically flawed). The revised question is narrower but answerable.

**What is lost**: The ambition of "systematic quantification" and "causal impact assessment." These are valuable goals but require validated metrics first.

**What is gained**: A clear, falsifiable contribution with genuine novelty (first unsupervised absorption detection) and a concrete next step (fix DFDA metric, validate cross-model).

---

## 5. New Hypotheses from Surprising Results

### NH1: UAD Precision Is the Bottleneck, Not Recall

**Observation**: UAD achieves perfect recall (1.0) but only 0.543 precision. This means it finds ALL true absorbed pairs but flags many false positives.

**Hypothesis**: A post-hoc filtering step (e.g., based on feature interpretability scores or activation pattern analysis) can improve UAD precision from 0.54 to >0.7 without sacrificing recall.

**Falsification**: If precision cannot exceed 0.6 with any filtering method, the false positive rate is structural (co-occurrence clustering detects correlation, not just absorption).

**Experiment**: Test 3 filtering strategies on UAD output: (a) LLM-as-judge interpretability score threshold, (b) activation sparsity ratio filter, (c) hierarchical consistency check (parent should be more general than child).

### NH2: The "Super-Absorber" Phenomenon Is General

**Observation**: Feature 18486 absorbs 5 first-letter concepts simultaneously. This is not a binary parent-child relationship.

**Hypothesis**: Multi-parent absorption is common in SAEs, and UAD's cluster-based approach is uniquely suited to detect it (unlike supervised methods that test one parent at a time).

**Falsification**: If cluster analysis on Gemma-2B and Pythia-2.8B shows <5% of absorbed features have >2 parents, the phenomenon is rare.

**Experiment**: Run UAD on 3 models, measure cluster size distribution. Compare to Chanin protocol (which only tests binary parent-child pairs).

### NH3: DFDA's True Value Is in Parent-Positive Recovery, Not Child-Dominant MSE

**Observation**: DFDA's 99.5% MSE improvement is artifactual (near-zero prediction). The real test is whether DFDA recovers parent activations on examples where the parent SHOULD fire but doesn't.

**Hypothesis**: When evaluated on parent-positive examples (ground truth from Chanin protocol), DFDA recovers >20% of missed parent activations.

**Falsification**: If DFDA recovers <5% of missed parent activations on parent-positive examples, the compensation mechanism is ineffective for the intended use case.

**Experiment**: Redefine DFDA evaluation metric. For each absorbed pair, identify examples where (a) child fires, (b) parent should fire (Chanin ground truth), (c) parent does NOT fire in baseline SAE. Measure fraction of "missed" activations recovered by DFDA.

---

## 6. Anti-Pattern Check

| Anti-Pattern | Status | Evidence |
|--------------|--------|----------|
| Post-hoc rationalization | **AVOIDED** | H2-H4 zero results were reported honestly, not explained away. Reflection.md explicitly states "H2-H4 all zero results" and does not claim they are "meaningful negative findings." |
| Hypothesis creep | **PRESENT** | The original H1-H4 (cross-architecture, causal, sparsity, layer) were demoted to supplementary but still retained in hypotheses.md. They should be dropped entirely or reframed as "exploratory observations." |
| Ignoring the original question | **AVOIDED** | The revised proposal directly addresses the evidence and reframes the question honestly. The original CAAB/causal directions are acknowledged as failures, not silently dropped. |

---

## 7. Bottom Line

The data forces a fundamental revision of our research mental model:

1. **Proxy metrics are dangerous**: Collision rate is not absorption. Systematic comparison built on unvalidated proxies produces noise, not insight.

2. **Novel methods beat systematic comparisons when metrics are immature**: UAD (a novel method) produced a clear signal; CAAB (a systematic comparison) produced noise. The field needs better methods before it needs better benchmarks.

3. **DFDA's metric was measuring the wrong thing**: The 99.5% improvement is not "absorption recovery" but "near-zero prediction." This is not a minor quibble -- it undermines the entire compensation claim. DFDA needs a new evaluation protocol before it can be claimed as a contribution.

4. **The contribution hierarchy must be inverted**: UAD alone is a solid methodological contribution. DFDA is promising but unvalidated. CAAB/causal assessment should be dropped or reframed as "lessons learned."

**Recommended paper framing**: "Unsupervised Feature Absorption Detection in Sparse Autoencoders: A Co-Occurrence Clustering Approach" with UAD as the sole primary contribution, DFDA as preliminary work, and honest discussion of why systematic comparison failed.
