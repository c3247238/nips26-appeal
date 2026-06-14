# Discussion

### 5.1 Collision Rate as a Poor Proxy

Collision rates differ 4x by architecture (15.4% vs. 3.8%), yet correlate weakly with downstream performance ($\rho_S$ = 0.10). This suggests the field may be conflating *detectable collision* with *harmful absorption*. Not all collisions impair interpretability---some may reflect benign feature compression that preserves semantic content. CAAB uses collision rate as the primary metric, but collision rate is measurable and may not measure harm.

### 5.2 Why H2-H4 Failed

Three hypotheses yielded near-zero correlations:
- **H2 (causal impact)**: Collision may not harm probing because SAE features are overcomplete (more features than dimensions)---even if one feature is suppressed, overlapping features preserve information.
- **H3 (sparsity monotonicity)**: Collision rate is invariant to $k$ because it depends on feature semantic overlap, not activation count.
- **H4 (layer depth)**: Collision rate is driven by concept co-occurrence in data, not layer position.

Alternatively, all three may reflect **underpowered designs**---with $n$ = 5--6, our study had approximately 20% power to detect a medium effect size ($r$ = 0.5) at $\alpha$ = 0.05. Small sample sizes and single-seed experiments cannot detect subtle effects. These null results are also consistent with superposition theory [Elhage et al., 2022]: if features are encoded in overlapping directions, some degree of collision is inevitable and may not impair downstream decoding.

### 5.3 UAD as a Promising Direction

UAD's F1 = 0.704 with perfect recall is the study's most promising direction. It eliminates the need for predefined hierarchies, enabling absorption detection on any concept set. Limitations include: (1) 54.3% precision means nearly half of detected collisions are false positives; (2) validation on larger models (Gemma-2B+) is needed; (3) the top-10% co-occurrence threshold is heuristic and may not generalize.

### 5.4 DFDA Feasibility

DFDA's 11.1% per-pair residual MSE improvement with negligible parameter overhead (388 parameters, <0.4% of SAE parameters) demonstrates that inference-time mitigation is feasible. The key challenge is generalization: DFDA is trained per-feature and may not transfer across models. If it does not generalize, the contribution is limited to a proof-of-concept.

### 5.5 Confounds and Limitations

1. **Pretrained vs. trained confound**: JumpReLU is pretrained on Gemma data while TopK is trained on OpenWebText. This affects all E1 comparisons.
2. **Proxy metric**: CAAB uses collision rate, not true absorption (requires hierarchy labels).
3. **Single model**: All experiments on GPT-2 Small; Gemma-2-2B experiments blocked by API issues.
4. **Single seed**: No multi-seed replication for statistical robustness.
5. **Dead features**: 89--99% dead feature ratio in trained SAEs indicates training problems that may confound results. How this affects collision rate measurements is unclear.

### 5.6 Future Work

1. Cross-model UAD validation (Gemma-2B, Pythia-2.8B)
2. Multi-seed replication with bootstrap CIs
3. True absorption detection per Chanin et al. protocol
4. DFDA generalization across models
5. Steering efficacy experiments (not measured due to time constraints)

Each limitation maps to a specific future work item, creating a coherent research agenda.

---
