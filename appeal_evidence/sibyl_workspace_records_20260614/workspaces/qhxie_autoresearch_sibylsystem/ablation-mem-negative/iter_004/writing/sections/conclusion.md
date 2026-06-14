# 7 Conclusion

We conducted the first systematic empirical evaluation of unsupervised co-occurrence clustering for detecting feature absorption in sparse autoencoders. Our findings are clear: UAD achieves F1 = 0.00048---detecting exactly 1 true positive out of 4,155 candidate pairs---a performance numerically identical to random sampling within clusters. All ablation variants fail, with the best alternative (K-means) achieving F1 = 0.0037, still far below any practical threshold.

Through careful root-cause analysis, we identified why: absorption features in tested token-disjoint hierarchies are mutually exclusive at the token level. A feature that activates on "three" never activates on "four," because these tokens represent different child concepts. Co-occurrence clustering searches for features that fire together, but absorption features fire on different tokens. This is a structural mismatch, not a methodological shortcoming that better tuning could fix.

Our investigation also yields a constructive contribution. We demonstrate that collision rate---the Jaccard overlap of top-$K$ activating features---exhibits strong internal consistency (Spearman $r = 0.869$, $n = 56$, 95% CI $[0.780, 0.938]$). This validates the structural coherence of our operationalization and suggests that top-$K$ feature overlap is a meaningful quantity, even if co-occurrence clustering is the wrong tool for exploiting it.

**Call to action.** We propose decoder weight similarity as the most promising alternative for unsupervised absorption detection. Unlike co-occurrence, decoder similarity directly measures structural relationships in the SAE's weight space and does not require token-level co-occurrence. Testing this hypothesis is the natural next step.

Negative results prevent wasted effort. By documenting that co-occurrence clustering is fundamentally unsuitable for absorption detection in token-disjoint hierarchies, we help the community focus on approaches with theoretical grounding. Honest reporting of what does not work is as valuable as reporting what does.
