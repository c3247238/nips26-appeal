# Related Work

## Feature Absorption: From Discovery to Benchmark

Chanin et al. (2024) introduced *feature absorption* as a systematic failure mode in sparse autoencoders (SAEs). Their key observation is that when hierarchical features co-occur---for example, the first letter of a word and the full word spelling---the sparsity penalty can cause the general "parent" feature to be encoded through the specific "child" feature. This makes the parent effectively invisible in the SAE's latent space, undermining the core goal of finding atomic, interpretable features. Chanin et al. also introduced the canonical first-letter absorption benchmark, which has since become the standard comparability metric in absorption research.

The community responded quickly with architectural innovations designed to reduce absorption. Korznikov et al. (2025) proposed **OrtSAE**, which enforces orthogonality between decoder directions via a cosine-similarity penalty and reports a 65% reduction in absorption. Bussmann et al. (2025) introduced **Matryoshka SAEs**, which train nested dictionaries of increasing size so that smaller dictionaries learn general concepts and larger ones learn specifics; they claim roughly a 10x reduction in absorption. Narayanaswamy et al. (2026) showed that masked regularization during training disrupts co-occurrence patterns and improves robustness. Li & Ren (2025) proposed adaptive temporal masking (ATM), a probabilistic feature-selection mechanism that outperforms TopK and JumpReLU on absorption metrics. These results have created a strong impression that absorption is a fixable bug, and that the next architectural tweak may eliminate it.

## The Skeptical Turn: Tradeoffs, Theory, and Limits

More recent work casts doubt on this optimistic framing. Chanin et al. (2025) proved that narrower SAEs reduce absorption but increase **feature hedging**---an opposite pathology in which latents incorrectly mix correlated features. This establishes a direct tension between two desirable properties: low absorption and monosemanticity. Cui et al. (2025) proved theoretically that standard SAEs generally fail to recover ground-truth monosemantic features under realistic correlation structures. Roy et al. (2026) showed "catastrophic interpretability collapse" under aggressive sparsification, suggesting that pushing too hard on sparsity can destroy interpretability utility altogether. Kantamneni et al. (2025) found that SAE probes underperform simple logistic-regression baselines on downstream tasks, raising questions about whether SAE features are as useful as assumed.

These theoretical and empirical limits suggest that absorption may not be an isolated bug but rather an intrinsic consequence of the sparsity objective and dictionary geometry. Yet no prior work has tested this hypothesis with a systematic, multi-objective evaluation across the full suite of SAE quality metrics. Individual papers compare architectures on absorption *and* reconstruction, but they do not frame the comparison as a Pareto analysis in which no architecture dominates across all objectives. Our work fills this gap.

## Benchmarks and Evaluation Frameworks

The rise of systematic SAE evaluation has been driven by two major benchmarks. **SAEBench** (Karvonen et al., 2025) is now the dominant comprehensive benchmark, covering 8 evaluations including feature absorption, sparse probing, RAVEL causality, spurious correlation removal (SCR), targeted probe perturbation (TPP), and unlearning. It releases over 200 trained SAEs across 7 architectures, making large-scale meta-analysis feasible. **CE-Bench** (Gulko et al., 2025) offers a lightweight contrastive alternative that correlates well with SAEBench without requiring an external LLM judge. More recently, **SynthSAEBench** (Chanin & Garriga-Alonso, 2026) provides synthetic data with ground-truth features and hierarchy, enabling controlled ablations.

Despite this progress, a major limitation remains: the canonical absorption metric is tied to the first-letter spelling task. Karvonen et al. (2025) explicitly note that supervised metrics are "fundamentally limited by the availability of ground truth data." No prior work has proposed a fully task-agnostic absorption metric using automated hierarchy discovery. Our Experiment 3 pilots such a metric and tests its correlation with the canonical benchmark.

## Relationship to This Paper

Our contribution is distinct from the prior work in three ways. First, we conduct the first systematic, training-free, multi-objective Pareto evaluation of absorption-mitigation methods using existing pretrained checkpoints. Second, we quantify absorption's unique causal effect on downstream interpretability utility via the largest-scale controlled meta-analysis to date (314 SAEBench checkpoints). Third, we pilot and validate a task-agnostic absorption metric, testing whether the canonical first-letter benchmark generalizes beyond its original domain. Together, these contributions reframe the SAE research agenda from "fixing absorption" to "navigating unavoidable tradeoffs."

<!-- FIGURES
- None
-->
