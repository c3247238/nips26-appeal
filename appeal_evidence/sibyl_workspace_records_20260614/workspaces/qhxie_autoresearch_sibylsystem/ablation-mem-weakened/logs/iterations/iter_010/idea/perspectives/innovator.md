# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

Based on the comprehensive literature survey already compiled in `context/literature.md`, the following papers are most relevant to the innovator's ideation:

1. **Chanin et al., 2024. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507** — Foundational work defining absorption and proving it is a logical consequence of sparsity loss under hierarchical features. This establishes the phenomenon and its theoretical basis.

2. **Tang et al., 2025. "On the Theoretical Foundation of Sparse Dictionary Learning in Mechanistic Interpretability." arXiv:2512.05534** — First theoretical explanation for absorption as spurious local minima; introduces "feature anchoring." This provides formal theoretical grounding.

3. **Li et al., 2025. "Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training." arXiv:2510.08855** — Shows ~40% reduction in absorption via temporal EMA tracking. Demonstrates that absorption is reducible through training modifications.

4. **Korznikov et al., 2025. "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033** — Reduces absorption by 65% via decoder orthogonality constraint. Architectural solution to the absorption problem.

5. **Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547** — Reduces absorption from 0.49 to 0.05 via nested architecture. Establishes the nested/hierarchical approach to absorption.

6. **Wang et al., ICLR 2026. "Does Higher Interpretability Imply Better Utility?" arXiv:2510.03659** — Shows weak correlation (~0.3) between interpretability and steering utility. Consistent with the project's null results on absorption-impact downstream tasks.

7. **Sanity Checks for SAEs, 2026. arXiv:2602.14111** — Frozen/random baselines match trained SAEs on multiple metrics. This is the closest prior art; our work extends by focusing on absorption metrics specifically.

8. **Various, 2026. "From Atoms to Trees: Building a Structured Feature Forest with Hierarchical SAEs." arXiv:2602.11881** — Jointly learns SAEs and parent-child relationships; recovers semantic hierarchies. Represents the cutting-edge of hierarchical SAE approaches.

### Landscape Summary

The SAE field has converged on understanding absorption as a real but nuanced phenomenon. Key developments:

1. **Architectural solutions abound**: Matryoshka, OrtSAE, ATM, H-SAE all reduce absorption through different mechanisms (nested dictionaries, orthogonality, temporal masking, hierarchical semantics). This suggests absorption is not a fundamental blocker but a tractable engineering problem.

2. **Theoretical foundations emerging**: Tang et al.'s work on spurious local minima and rate-distortion theory provides formal grounding for why absorption occurs and when it can be mitigated.

3. **The utility question is open**: Wang et al. show interpretability-utility correlation is weak (~0.3), consistent with our null results. The field is questioning whether absorption reduction actually improves downstream tasks.

4. **Sanity Checks challenge**: The 2026 "Sanity Checks" paper raises the fundamental question: do trained SAEs outperform random baselines on ANY metric? Our H7 finding directly addresses this for absorption metrics specifically.

5. **Gap in the literature**: No prior work has specifically compared trained vs. random SAEs on the Chanin absorption differential correlation metric. The Sanity Checks paper addresses general metrics but NOT absorption metrics specifically. This is our primary novelty gap.

---

## Phase 2: Initial Candidates

Given the project's maturity (iteration 9, all experiments complete, cand_g established), I generate candidates that represent genuine innovation possibilities or refinements, not radical pivots.

### Candidate A: Information-Theoretic Absorption Decomposition

- **Hypothesis**: The Chanin absorption metric can be decomposed into three independent information-theoretic components: (1) geometric artifact (random SAE baseline), (2) structural artifact (training-independent dictionary properties), and (3) genuine absorption (training-dependent feature hierarchy effect). The genuine component is small relative to artifacts.

- **Cross-domain insight**: This draws from rate-distortion theory in information science and signal processing. In compression, the distinction between artifact and signal is formalized via excess distortion measures. The absorption phenomenon mirrors this: some "absorption" is distortion from overcomplete representation, not meaningful feature merging.

- **Why it might work**: Our H7 already shows random SAEs have 8x higher absorption than trained SAEs. An information-theoretic decomposition would quantify how much of the remaining absorption in trained SAEs is artifact vs. genuine. If genuine absorption is small, it explains why absorption doesn't degrade downstream tasks (H1-H4 null results).

- **Rough novelty estimate**: 6/10 — Rate-distortion theory is well-established (Shannon 1959, cover & thomas 2006), but applying it specifically to decompose the Chanin absorption metric is novel. The decomposition methodology itself would be a contribution.

### Candidate B: Absorption as Feature Routing (Evolution of cand_g)

- **Hypothesis**: Feature absorption is not a failure mode but an information routing mechanism where child features selectively suppress parent feature activation to reduce redundancy. This is analogous to predictive coding in neuroscience where top-down predictions suppress bottom-up signals.

- **Cross-domain insight**: Predictive coding (Clark 2013, Friston 2005) proposes that neural circuits route information by suppressing expected signals, only propagating prediction errors. The SAE absorption dynamic—where child features fire while parent features are suppressed—mirrors this: absorption may represent "prediction error routing" where specific features capture what general features missed.

- **Why it might work**: H5 shows precision=1.0 (decoder directions preserve semantic content) but recall varies. The routing framing explains this: absorbed features still encode correct information but in a more specific context (child dominates when parent+child co-occur, maximizing sparsity).

- **Rough novelty estimate**: 5/10 — Predictive coding analogies have been explored in other neural network interpretability contexts. The specific connection to SAE absorption is novel but derivative of established theory.

### Candidate C: Absorption Benchmark Gap Analysis

- **Hypothesis**: The current absorption evaluation landscape has a critical gap: no systematic comparison of absorption rates across different SAE architectures (ReLU, TopK, JumpReLU, Gated, Matryoshka, OrtSAE) using identical evaluation protocols and the same underlying model. Our GPT-2 Small analysis is informative but limited.

- **Cross-domain insight**: This is a meta-scientific contribution rather than cross-domain. The gap exists because each paper evaluates on its own setup; no unified comparison framework exists. This mirrors the "benchmark fragmentation" problem in other ML fields (NLP, CV).

- **Why it might work**: SAEBench (Karvonen et al.) attempts this but includes 200+ SAEs with heterogeneous evaluation protocols. A focused absorption comparison with standardized protocols would reveal architecture-dependent absorption patterns that inform whether architectural solutions (Matryoshka, OrtSAE) actually reduce genuine absorption or just metric artifacts.

- **Rough novelty estimate**: 4/10 — SAEBench already attempted unified evaluation. The specificity to absorption metrics is novel but the meta-science approach is established.

---

## Phase 3: Self-Critique

### Against Candidate A (Information-Theoretic Decomposition)

- **Prior work attack**: Rate-distortion theory for dictionary learning has been explored (e.g., Gribonval & Rauhut, 2008; Tosic & Frossard, 2011). However, these works focus on general dictionary learning, not SAE-specific absorption metrics. The Chanin differential correlation metric is unique to the absorption literature. Applying standard IT decomposition to this specific metric is novel.
- **Methodological attack**: The decomposition would require estimating mutual information I(parent; child | input), which is notoriously difficult in high dimensions. Finite sample effects could dominate. The decomposition might be too noisy to yield actionable insights.
- **Theoretical attack**: The three-component decomposition (geometric, structural, genuine) is intuitive but may not map cleanly to estimable quantities. The boundaries between "geometric artifact" and "structural artifact" may be blurry.
- **Scalability attack**: The decomposition requires fitting the random SAE baseline carefully. Our existing H7 establishes the random SAE baseline, but extending to a full decomposition with confidence intervals requires substantial additional analysis.
- **Verdict**: MODERATE — The idea is theoretically sound and addresses a real gap (quantifying genuine vs. artifact absorption). However, the implementation complexity is high and the marginal contribution over existing H7 findings may be limited.

### Against Candidate B (Absorption as Feature Routing)

- **Prior work attack**: Predictive coding has been applied to neural networks (e.g., Lotter et al. 2016, Kreutzer et al. 2022). The analogy to SAE absorption is intriguing but not clearly novel. The Friston community has explored related ideas without specific connection to SAEs.
- **Methodological attack**: Testing the routing hypothesis would require showing that absorbed features fire in a complementary pattern to parent features (suppression rather than co-activation). Our existing data (p_11 + absorption = 1.0) is consistent but definitional, not causal evidence for routing.
- **Theoretical attack**: The predictive coding analogy may be superficial. SAEs optimize a reconstruction + sparsity objective, not a predictive coding objective. The structural correspondence is metaphorical, not mechanistic.
- **Scalability attack**: The routing framing might only apply to cases where parent-child relationships are clearly defined (hierarchical features). Many SAE features do not have obvious parent-child structure.
- **Verdict**: WEAK — The analogy is intriguing but speculative. It does not clearly outperform the existing rate-distortion optimal compression framing (cand_g) and adds complexity without clear empirical payoff.

### Against Candidate C (Absorption Benchmark Gap Analysis)

- **Prior work attack**: SAEBench (Karvonen et al., ICML 2025) already provides a comprehensive benchmark including absorption. Our focused absorption comparison would be a subset of what SAEBench already attempted.
- **Methodological attack**: Running a systematic architecture comparison would require significant computational resources and time. The project constraints (training-free, 1-hour experiments) may not accommodate this.
- **Theoretical attack**: The gap analysis is descriptive, not explanatory. It would document what architectures reduce absorption but not why. This is valuable but less intellectually compelling than mechanistic understanding.
- **Scalability attack**: A proper benchmark comparison would require standardizing across 6+ architectures, multiple model sizes, and multiple layers. This is a multi-year research program, not a single project.
- **Verdict**: WEAK — The benchmark gap is real but addressing it exceeds the current project scope. The finding would be "Matryoshka reduces absorption more than ReLU" which is already known from individual papers.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate B (Feature Routing)** dropped because: The predictive coding analogy is speculative and does not clearly outperform the existing rate-distortion framing. No empirical test distinguishes routing from compression without additional experiments.

- **Candidate C (Benchmark Gap Analysis)** dropped because: Exceeds project scope; SAEBench already addresses this at a high level; would require multi-architecture evaluation that contradicts the training-free, single-model project constraints.

### Strengthened Ideas

- **Candidate A (Information-Theoretic Decomposition)**: Strengthened by considering that the existing H7 (trained < random) already establishes the foundation for decomposing absorption into artifact vs. genuine components. The key question is: how much of trained SAE absorption is genuine vs. artifact? If our decomposition shows that even "genuine" absorption is metric-sensitive, this would strengthen the paper's core claim that absorption is mostly benign.

### Additional Evidence Found

From the existing literature and project findings:

1. The Chanin absorption metric is a differential correlation measure, not a direct absorption count. This means the metric itself has properties (sensitivity to dictionary geometry) that could be formally characterized.

2. Our H5 finding (precision=1.0, recall varies) is consistent across all layers. This suggests the decoder geometry is preserved even when encoder activation is suppressed—supporting the "optimal compression" framing.

3. Feature U (24.2% absorption, 100% steering success) is a concrete example where absorption is demonstrably benign for downstream tasks.

### Selected Front-Runner

**Candidate A (Information-Theoretic Absorption Decomposition)** is selected as the innovator's contribution because:

1. It directly addresses the unresolved question: how much of measured absorption is genuine vs. artifact?
2. It leverages our existing H7 findings (random SAE baseline) rather than requiring new experiments
3. The decomposition framework is novel and reusable by the community
4. If successful, it provides a formal answer to "is absorption real or a metric artifact?" that transcends our specific findings

However, I note that **cand_g (Optimal Compression Framing)** remains the overall project front-runner because:
- All experiments are already complete for cand_g
- All 6 perspectives have validated cand_g
- cand_g's novelty (7/10) is already confirmed
- Candidate A would require additional analysis that may not fit the current project timeline

**Recommendation**: The innovator perspective contributes Candidate A as a potential extension or future work direction, but the project should proceed with cand_g as the main contribution.

---

## Phase 5: Final Proposal

### Title

**Information-Theoretic Decomposition of SAE Feature Absorption: Quantifying Genuine vs. Artifact Components**

### Hypothesis

The Chanin differential correlation absorption metric can be decomposed into three orthogonal components: (1) a geometric artifact term proportional to dictionary overcompleteness, (2) a structural artifact term reflecting training-independent dictionary geometry, and (3) a genuine absorption term capturing actual parent-child feature merging. The genuine component is small relative to artifacts, explaining why absorption does not degrade downstream task performance.

### Motivation

Our H7 finding (trained SAEs show 8x lower absorption than random SAEs) raises a critical question: how much of the remaining absorption in trained SAEs is genuine feature merging vs. residual structural artifact? If most measured absorption is artifact, it explains the null results (H1-H4): absorption doesn't degrade tasks because it doesn't meaningfully alter feature geometry, only activation patterns.

### Method

1. **Random SAE baseline extension**: Use the existing H7 framework to characterize absorption in random SAEs with varying dictionary sizes and encoder/decoder initialization strategies.

2. **Geometric artifact quantification**: Compute absorption as a function of dictionary overcompleteness ratio (n_latents / n_inputs) in random SAEs. Fit a power law: artifact_abs = alpha * (overcompleteness)^beta.

3. **Structural artifact quantification**: Characterize absorption as a function of decoder condition number and encoder-decoder alignment in trained SAEs vs. random SAEs.

4. **Genuine absorption estimation**: Subtract estimated artifact contributions from total measured absorption. Compare genuine absorption to downstream task metrics.

### Experimental Plan

| Experiment | Method | Metric |
|---|---|---|
| Geometric artifact curve | Random SAE absorption vs. overcompleteness ratio | Power law fit (alpha, beta) |
| Structural artifact analysis | Absorption vs. decoder condition number, trained vs. random | Correlation and residual analysis |
| Genuine absorption estimation | Total - geometric - structural | Residual absorption fraction |
| Downstream validation | Correlation of genuine absorption with H1-H4 metrics | Pearson r, significance |

### Resource Estimate

- **Computational**: ~50 GPU-hours for random SAE sweeps across dictionary sizes (0.5x to 4x overcomplete)
- **Time**: ~1 week for full decomposition analysis
- **Model sizes**: GPT-2 Small (primary), Gemma-2-2B (validation if time permits)

### Risk Assessment

1. **Risk**: The three-component decomposition may not be identifiable (components may be entangled)
   - **Mitigation**: Use regularization or constraints to ensure identifiability; report sensitivity analysis

2. **Risk**: The genuine absorption component may still correlate with artifact components
   - **Mitigation**: Report uncertainty bounds; use bootstrap confidence intervals

3. **Risk**: The decomposition may require more time than available
   - **Mitigation**: Treat as future work if project timeline is constrained; cand_g remains main contribution

### Novelty Claim

First information-theoretic decomposition of the Chanin absorption metric into artifact vs. genuine components. First formal quantification of how much measured absorption is attributable to structural artifacts vs. actual feature merging. This provides a principled answer to "is absorption real?" that transcends single-model, single-metric analyses.

### Relationship to cand_g

This proposal (Candidate A) is an **extension** of cand_g, not a replacement. If implemented:
- It strengthens cand_g's "optimal compression" framing by providing formal decomposition
- It answers the implicit reviewer question: "is your null result because absorption doesn't matter, or because your metric is flawed?"
- It provides a reusable framework for future absorption studies

If not implemented due to time constraints, cand_g remains the strong front-runner with the H7 finding as its primary novelty contribution.
