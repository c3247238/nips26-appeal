# Conclusion

We presented the first cross-architecture study of feature absorption in Sparse Autoencoders, evaluating six hypotheses. Our key findings are:

1. **Architecture matters, with caveats**: Collision rates differ 4x between JumpReLU (15.4%) and TopK (3.8%), but this comparison confounds architecture with training data (pretrained JumpReLU vs. trained TopK).

2. **Collision rate is a poor proxy for absorption harm**: The near-zero correlation between collision rate and sparse probing accuracy ($\rho_S$ = 0.10, $p$ = 0.870) indicates that collision rate does not predict downstream task impairment, though our study's statistical power was limited.

3. **Unsupervised detection is feasible**: UAD achieves F1 = 0.704 with perfect recall, enabling absorption detection without labeled hierarchies.

4. **SAE-retraining-free mitigation is feasible**: DFDA improves per-pair residual MSE by 11.1% with 388 parameters.

This work has limitations: collision rate is a proxy metric, experiments use a single seed on GPT-2 Small, and the pretrained-vs-trained confound affects E1 comparisons. By providing scalable tools for absorption detection and lightweight mitigation, we support the development of more reliable interpretability methods for AI safety applications.

The SAE community should refocus: rather than treating all absorption as harmful, we need better discrimination between benign compression and problematic feature loss. UAD provides a scalable tool for this discrimination, and DFDA offers a lightweight mitigation path. Collision rate is a poor proxy for absorption harm; unsupervised detection and lightweight mitigation offer more promising paths.

---
