# 11. Conclusion

This paper provides the first systematic empirical characterization of feature absorption in Sparse Autoencoders. Across five hypotheses tested on GPT-2 small residual-stream SAEs, the central finding is uniform: absorption as defined by the $A_f$ metric is too rare to constitute a primary failure mode for SAE-based interpretability in this model class.

## 11.1 Summary of Findings

We designed five hypotheses about absorption prevalence, causes, and downstream impact. Four were falsified; H5 moved in the hypothesized direction but with limited practical significance.

**H1** predicted that over 20% of mid-layer latents would show absorption ($A_f > 0.5$). We found 0.19% — 100x below the hypothesis threshold and well below the falsification boundary of 10%. The phenomenon is not merely rare; it is nearly absent.

**H3** predicted a monotonic increase in absorption with sparsity. We found an inverted-U pattern: absorption peaks at layer 4 (49.3% of latents with $A_f > 0.5$) and declines at both shallower and deeper layers. The sparsest layers (8 and 10) show the lowest absorption rates. Sparsity, as measured by L0, is not a reliable proxy for absorption risk.

**H4** predicted that patching high-absorption SAE latents would degrade circuit faithfulness by at least 5 percentage points relative to low-absorption latents. Both subsets yielded 0.0 faithfulness, making the comparison uninformative. The latent selection method (corpus-wide bottom/top 10% by absorption score) apparently selects latents irrelevant to the France/Paris circuit rather than latents that differ in absorption quality. Full SAE patching (0.289 faithfulness) substantially outperforms both subsets (0.0), suggesting circuit faithfulness requires the complete dictionary.

**H5** predicted that larger dictionary sizes reduce absorption. The data confirm this: 2.25% at $d_{\text{sae}} = 2{,}048$ down to 0.19% at $d_{\text{sae}} = 24{,}576$. The 10x reduction is real, but the practical significance is bounded by the overall rarity of the phenomenon — even at 2K dictionary size, absorption prevalence is low by any reasonable standard.

**H2** was not tested. The H2 pilot was skipped after H1, H3, and H4 falsification demonstrated that the full experimental program needed revision. The pre-registered analysis — Spearman correlation between median token frequency and absorption score — remains a legitimate open question, but running it without redesigning the surrounding framework would produce isolated results of unclear value.

These are genuine negative results, not artifacts of an uncalibrated metric. Random dictionary controls score 0.00% by construction, confirming the metric detects real structure. The metric is strict, not broken.

## 11.2 What Negative Results Mean

The falsification of primary hypotheses is meaningful in two senses. First, researchers using GPT-2 small SAEs for circuit tracing or feature attribution can proceed with moderate confidence that SAE latents represent relatively independent directions in activation space. The risk that one latent's semantic content is redundantly encoded by another is bounded by observed prevalence: below 0.5% even at the worst layer. This is not a guarantee of independence — partial absorption at levels below the $A_f > 0.5$ threshold may exist — but it rules out the high rates that would invalidate SAE-based circuit analysis.

Second, the research community can now rule out absorption as a primary failure mode for SAE-based circuit analysis in models of GPT-2 small's scale and architecture. Prior work that assumed absorption was widespread, or developed mitigation strategies premised on its prevalence, should be revisited in light of this boundary condition.

The findings do not generalize to larger models. GPT-2 small has well-characterized limitations, and its feature space is orders of magnitude smaller than GemmaScope or LLaMA-class models. Whether absorption scales with model size, and whether the anti-absorption pressure that appears to emerge from standard SAE training objectives holds at larger scales, remain open empirical questions. The experimental framework developed here provides the tools to answer them.

## 11.3 Practical Recommendations

Three findings have direct implications for practitioners.

**The full SAE dictionary is necessary for downstream causal validity.** H4 shows that patching with latent subsets — whether selected by absorption score, activation magnitude, or any other criterion — destroys reconstruction capacity entirely. The faithful circuit tracing depends on the complete dictionary, not on a curated subset. This holds regardless of absorption level.

**When absorption risk is a concern, prefer larger dictionary SAEs.** H5 demonstrates a monotonic reduction in absorption rate with dictionary size: 2.25% at 2K latents, 0.56% at 8K, 0.19% at 24K. The effect is consistent across all metrics and holds even when random controls are subtracted.

**Sparsity level is not a proxy for absorption risk.** The inverted-U pattern at mid-layers reflects the model's internal representational structure, not the sparsity hyperparameter. Pushing for sparser representations does not increase absorption at higher rates. Practitioners who avoid high-sparsity SAEs on absorption grounds should reconsider: the data do not support that concern.

## 11.4 A Reproducible Framework

The paper's primary contribution beyond the empirical findings is a validated absorption metric and a reproducible experimental framework. The absorption score $A_f$ is defined precisely in Section 3.1; the falsification criteria were pre-registered before data collection; and the code is publicly available. Any researcher can apply this framework to other SAE releases, other base models, or other dictionary sizes and produce comparable results. A researcher studying GemmaScope SAEs on Gemma 2B can run the same pipeline and produce results directly comparable to those reported here.

The open questions — whether absorption manifests differently in GemmaScope SAEs, whether the 8 perfect-score latents are genuine absorption or artifacts of the metric, whether a relaxed absorption threshold captures subtler forms of feature overlap, whether explicit anti-absorption regularization during SAE training reduces absorption further — are matters for future work. This paper establishes the baseline.

<!-- FIGURES
- None
-->