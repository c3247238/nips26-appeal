# 8 Discussion

## 8.1 Practical Implications for SAE Evaluation

Single-task, single-layer absorption benchmarks understate the problem's complexity. First-letter absorption at layer 12 ranges from 0.7% to 3.4% across four SAE architectures (Table 4), yet the same metric at layer 24 reaches 25.5--34.5% on the same model (Figure 3). Any evaluation that measures absorption at one layer and reports it as a property of the SAE systematically mischaracterizes the failure mode. SAEBench (Karvonen et al., 2025) and the sae-spelling benchmark (Chanin et al., 2024) both operate at a single layer; our results demonstrate that the layer choice alone can shift the measured rate by 15x.

Cross-domain variation compounds this problem. At layer 24, first-letter absorption (34.5% on 16k-width, 25.5% on 65k-width) differs significantly from city-country (18.5%, 12.7%) and city-language (13.6%, 13.6%), with 4 of 6 pairwise comparisons reaching statistical significance (Kruskal-Wallis p=0.005; pairwise permutation p=0.0001 to 0.015). A benchmark that reports "15--35% absorption" based on first-letter spelling at one layer conveys a false impression of precision: the actual rate depends on which hierarchy the user cares about and where in the network it is measured.

These findings have direct consequences for architecture comparison. The four architectures tested at layer 12 (JumpReLU 16k/65k, BatchTopK 16k, Matryoshka 32k) show no significant absorption difference (Kruskal-Wallis p=0.87), while the hierarchy effect is significant (p=0.005). Hierarchy type explains more variance in absorption than architecture choice. Current architecture comparisons for absorption resistance -- including those for Matryoshka SAE (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), ATM-SAE (Li et al., 2025), and masked regularization (Narayanaswamy et al., 2026) -- all benchmark on first-letter only. If the architecture ranking changes across hierarchy types, as our data at layer 12 tentatively suggest (BatchTopK shows lowest city-continent absorption at 13.5% while Matryoshka shows lowest city-country and city-language at 35.3%), then single-task rankings may mislead practitioners.

The practical recommendation is straightforward: absorption evaluation should report rates across at least two hierarchy types at multiple layers, with probe quality documented alongside each measurement.

## 8.2 Layer-Position Mechanism Hypothesis

The absorption surge at layer 24 -- where first-letter rates jump from 2.2--9.2% at layers 6--18 to 25.5--34.5% (Figure 3) -- coincides with the final residual stream position in Gemma 2 2B's 26-layer architecture. This pattern is consistent with a representation-sharpening mechanism: at earlier layers, the residual stream carries distributed representations that SAEs can encode without strong hierarchical competition; at layer 24, the representation sharpens toward specific token predictions, concentrating parent-child information overlap and creating the conditions for child features to absorb parent features.

Two pieces of evidence support this interpretation. First, RAVEL probe quality peaks at layer 24 for all three entity-attribute hierarchies (city-continent F1=0.843, city-country F1=0.789, city-language F1=0.823), indicating that factual knowledge is most linearly accessible at this layer. Second, the absorption-hedging decomposition shows that L24 has a lower proportion of absorbed (vs. hedged) false negatives compared to earlier layers -- 50.0% absorbed at L24-16k versus 100.0% at L6-16k -- suggesting that late-layer absorption competes with hedging as both failure modes intensify.

An alternative explanation is that layer 24 SAEs have different effective $L_0$ values or reconstruction quality that drives the absorption increase. We cannot fully rule this out with the current data, as we did not measure reconstruction error per layer. Disentangling the representation-sharpening hypothesis from confounds in SAE training quality at different layers requires controlled experiments that are beyond the scope of this paper.

## 8.3 The Probe Quality Confound

Probe quality correlates strongly with false negative rate ($\rho$=-0.756, p<0.001), and this confound must be acknowledged as the primary limitation on interpreting cross-domain absorption rates. First-letter probes achieve F1=0.97 at all layers tested, creating a large denominator (nearly all tokens correctly classified in the raw condition) from which false negatives can be detected. RAVEL probes peak at F1=0.79--0.84, meaning that many genuine misclassifications in the raw condition are invisible to the absorption measurement pipeline.

The direction of this bias is unambiguous: lower probe quality underestimates the denominator (correctly classified raw tokens), which inflates the false negative count from probe error and simultaneously reduces the absorption rate by mixing genuine absorption with probe-induced false negatives. The net effect on absolute rates is unclear -- it could either inflate or deflate the measured absorption rate depending on the relative magnitudes of these opposing forces. What is clear is that the quantitative cross-domain rates (13.6--35.8%) carry meaningful uncertainty due to this confound, and that the 15x layer-dependent variation on first-letter (where probes are near-perfect) is the cleanest finding.

The threshold sensitivity analysis provides a partial reassurance: across a 5x4 grid of cosine thresholds (0.01--0.05) and magnitude gaps (0.5--2.0), false negatives remain constant at n=87 and absorption rates vary only from 11.8% to 15.1% (CV=0.077). Absorption is structural, not an artifact of detection threshold choice. Probe quality, not detection thresholds, is the binding constraint.

## 8.4 Broader Context: Absorption as Intrinsic to Sparse Coding

The activation patching result -- 32.5% recovery when child features are zeroed versus 1.5% for magnitude-matched controls ($\Delta$RR=0.310, 95% CI [0.213, 0.421], Wilcoxon p=0.000218, Cohen's d=1.33) -- provides the first interventional causal evidence that feature absorption reflects genuine feature suppression, not merely a measurement artifact. 16 of 19 words with detected absorption show positive recovery, with effects spanning both common tokens (yaitu: 45.5% recovery, raw accuracy 1.0) and rare tokens (conmigo: 100% recovery, raw accuracy 0.4).

Combined with the layer dependence finding, this suggests that absorption is an intrinsic property of SAE encoding under sparsity constraints. The absorption-reconstruction tradeoff is fundamental: eliminating absorption for a parent-child pair requires the SAE to activate both the parent and child latents, costing +1 $L_0$ per pair. At layer 24, where representations are sharpest and hierarchical overlap is maximal, this cost is paid most frequently. The three failed unsupervised detectors (GAS: $\rho$=0.116; CMI: $\rho$=0.044; Absorption Tax ranking: $\rho$=-0.20) reinforce that absorption is not a simple geometric or information-theoretic property of the decoder -- it emerges from the interaction between encoder dynamics, sparsity pressure, and input statistics.

This framing has a specific implication for the field: absorption cannot be eliminated by architectural changes alone (architecture effect p=0.87), but may be mitigated by operating at higher $L_0$ or by developing training objectives that explicitly penalize parent-child suppression. The tightened hedging decomposition supports this: at base $L_0$=22, only 7.9% of false negatives exhibit strict hedging (parent recovery at 8x $L_0$), while 86.2% resolve through compensatory features. The widely-cited ~98% hedging rate conflates parent recovery with the near-tautological observation that expanding the dictionary by 8x resolves most false negatives through any available feature.

## 8.5 Limitations

**RAVEL probes below strict quality gate.** The best RAVEL probes reach F1=0.843 (city-continent at layer 24), below the strict 0.90 quality gate. Absolute cross-domain absorption rates carry quantitative uncertainty. The qualitative finding -- that rates differ significantly across hierarchies -- is supported by the Kruskal-Wallis test (p=0.005), but the exact magnitude of each rate should be interpreted with caution.

**Activation patching restricted to first-letter.** All 25 patching targets are first-letter absorption pairs at layer 12 (probe F1=0.883). Cross-domain activation patching was not attempted because no RAVEL probe passes the 0.85 quality gate required for reliable causal inference. Whether the causal mechanism generalizes to entity-attribute hierarchies remains unknown.

**Architecture comparison at layer 12 only.** The four-architecture comparison is limited to layer 12, the only layer with SAEs available for all architectures (Gemma Scope JumpReLU + SAEBench BatchTopK and Matryoshka). At layer 12, RAVEL probes are at their worst (F1=0.52--0.69), making the cross-domain architecture comparison doubly confounded. The architecture null result (p=0.87) may reflect genuine equivalence or insufficient statistical power (N=16 observations, minimum detectable effect is large).

**Single model family.** All experiments use Gemma 2 2B with Gemma Scope JumpReLU SAEs as the primary evaluation target. Generalization to other model families (Llama, Mistral), model scales (9B, 27B), and independently trained SAEs has not been tested. The 15x layer variation and cross-domain effects may be specific to Gemma 2 2B's architecture or Gemma Scope's training procedure.

**Training-free evaluation only.** This study evaluates pretrained SAEs at inference time. Whether retraining SAEs with hierarchy-aware losses, modified sparsity schedules, or explicit anti-absorption regularization would change the observed layer and hierarchy dependence is an open question.

## 8.6 Future Directions

**Degraded-probe ablation.** Injecting calibrated noise into first-letter probes to simulate RAVEL-level quality (F1=0.80--0.85) would quantify the probe-quality confound directly. If absorption rates on degraded first-letter probes match RAVEL rates, the cross-domain variation is largely a probe artifact; if they remain distinct, the hierarchy effect is genuine.

**Cross-domain activation patching.** Improving RAVEL probes above the 0.85 quality gate -- through better prompt templates, larger entity sets, or probe architectures beyond logistic regression -- would enable activation patching on entity-attribute hierarchies. The causal evidence currently rests entirely on first-letter features.

**Multi-model validation.** Gemma 2 9B and 27B have Gemma Scope SAEs available at multiple layers. Replicating the layer-dependent absorption pattern across model scales would establish whether the layer 24 spike is universal or specific to the 2B model. Cross-family replication on Llama 3.1 with independently trained SAEs (e.g., from SAEBench) would further strengthen generalizability.

**Unsupervised detection.** All three unsupervised detectors tested here fail (GAS, CMI, Absorption Tax). Absorption currently requires supervised probe-based measurement. Developing detectors that capture encoder competitive exclusion dynamics -- rather than decoder geometry alone -- is an open challenge. Activation patching at scale could provide training signal for a learned absorption predictor.

**Safety-relevant feature hierarchies.** The practical motivation for absorption research is that safety-relevant features (deception, manipulation, harmful intent) live in knowledge and reasoning space, not spelling space. Extending absorption measurement to safety-relevant hierarchies -- if suitable probes can be developed -- would directly connect this work to AI safety.

<!-- FIGURES
- None
-->
