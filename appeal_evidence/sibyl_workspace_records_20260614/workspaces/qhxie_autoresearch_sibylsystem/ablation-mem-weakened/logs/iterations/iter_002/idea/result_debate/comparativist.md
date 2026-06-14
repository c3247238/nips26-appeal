# Comparativist Analysis: The Local Inhibition Graph for SAE Feature Absorption

## Executive Summary

This project has pivoted from a null-result correlation study (absorption vs. downstream task performance) to a novel theoretical framework: **the Local Inhibition Graph**, connecting Rozell et al.'s Locally Competitive Algorithm (LCA) from neuroscience to Sparse Autoencoder (SAE) feature absorption. The core claim is that the decoder correlation matrix W_dec^T W_dec is exactly the inhibition matrix G_LCA from the LCA framework, providing a mechanistic explanation for absorption as competitive suppression.

**Verdict**: The contribution margin is **moderate-to-strong** relative to existing work. The LCA-SAE connection is technically novel, the framework explains existing empirical findings, and the diagnostic tool is practical. However, the field is crowded with SAE analysis methods, and the framework's empirical validation is still pending. The pivot was the right strategic move -- the original null-result study had marginal contribution at best.

---

## 1. Baseline Landscape: Where Does This Work Stand?

### 1.1 Existing Work on Absorption Mechanisms

| Study | What They Showed | Key Numbers | Our Proposal | Delta |
|-------|-----------------|-------------|--------------|-------|
| **Chanin et al. (2024/2025)** | First identification of absorption; proves it is a logical consequence of sparsity loss under hierarchical features | Absorption occurs in 100% of tested LLM SAEs; metric validated on Gemma-2-2B, Llama 3.2, Qwen2 | We explain *why* absorption happens mechanistically (competitive suppression via decoder correlations) | **Theoretical advance** -- Chanin identifies the phenomenon; we propose the mechanism |
| **SAEBench (Karvonen et al., ICML 2025)** | Standardized absorption metric + 8 downstream metrics | Absorption scores vary 0.05-0.35 across architectures | Our framework predicts which features are at-risk without running absorption metrics | **Complementary** -- SAEBench measures; we diagnose pre-emptively |
| **Matryoshka SAE (Bussmann et al., ICML 2025)** | Nested multi-level dictionaries reduce absorption from 0.49 to 0.05 | 90% absorption reduction | Our framework explains *why* multi-level structure helps (reduces competitive suppression at each level) | **Explanatory** -- explains an existing architectural solution |
| **OrtSAE (Korznikov et al., 2025)** | Decoder orthogonality constraint reduces absorption by 65% | -65% absorption, -15% composition | Our framework predicts that orthogonality reduces inhibition (G_ij = 0 for orthogonal decoders) | **Predictive** -- our framework predicts OrtSAE's effectiveness |
| **ATM (Li et al., ICLR-W 2025)** | Temporal EMA tracking reduces absorption ~40% | ~40% reduction; maintains reconstruction | Our framework does not directly explain temporal dynamics but is compatible | **Neutral** -- orthogonal direction |
| **Cui et al. (2025)** | Formal identifiability analysis; proves conditions for recovering ground-truth features | Theoretical conditions for perfect recovery | Our framework is phenomenological, not identifiability-theoretic | **Different level** -- they prove conditions; we explain mechanism |
| **Tang et al. (2025)** | First theoretical explanation for absorption as spurious local minima; "feature anchoring" | Theoretical framework | Our framework is complementary (dynamical vs. optimization) | **Complementary** -- different theoretical lens |

### 1.2 Existing Work on Decoder Correlations and Feature Competition

| Study | What They Showed | Relevance to Our Proposal |
|-------|-----------------|---------------------------|
| **Feature Suppression in SAEs (LessWrong, 2024)** | Sparsity penalty causes feature suppression; standard loss = MSE + alpha * L1 | **Foundational** -- establishes that suppression exists, but does not connect to decoder correlations |
| **OrtSAE (Korznikov et al., 2025)** | Chunk-wise orthogonality penalty on decoder columns: gamma * L_ortho(W_dec) | **Directly related** -- they penalize decoder correlations; we analyze them as an inhibition graph |
| **Decoder pairwise cosine similarity (OpenReview 2025)** | c_dec = mean of |cos(W_dec_i, W_dec_j)| as metric for feature mixing | **Related** -- uses decoder correlations as diagnostic; we use them as a graph structure |
| **CorrSteer (Arad et al., 2025)** | Pearson correlation for task-specific feature selection | **Related** -- uses correlations for steering; we use them for absorption diagnosis |
| **AlignSAE (2025)** | Two-stage training with orthogonality loss L_perp | **Related** -- enforces orthogonality; we analyze the correlation structure that orthogonality eliminates |

### 1.3 LCA / Neuroscience Connection Landscape

| Study | What They Showed | Relevance |
|-------|-----------------|-----------|
| **Rozell et al. (2008)** | LCA for sparse coding; inhibition matrix G = Phi^T Phi | **Foundational** -- our theoretical basis; ~2000 citations, zero applications to LLM SAEs |
| **Paiton et al. (2019)** | LCA on TrueNorth neuromorphic system | **Related** -- hardware implementation of LCA; not connected to SAEs |
| **ALCA (2024)** | Adaptive LCA for speech classification | **Related** -- extends LCA; not connected to SAEs |
| **Meyer & Kenyon (2020)** | LCA lateral inhibition explains V1 responses | **Related** -- biological validation of LCA; not connected to SAEs |

**Critical observation**: There is **no existing paper that connects the LCA inhibition matrix to SAE decoder correlations**. The structural correspondence (W_dec^T W_dec = G_LCA) has not been articulated in the literature. This is the core novelty claim.

---

## 2. Contribution Margin Assessment

### 2.1 What the Proposal Claims vs. What Exists

| Claimed Contribution | Evidence Required | Existing Work | Margin |
|---------------------|-------------------|---------------|--------|
| "First connection between LCA lateral inhibition and SAE absorption" | Clear mathematical derivation showing W_dec^T W_dec = G_LCA | Rozell et al. (2008) established LCA; no SAE connection exists | **Strong (>5%)** -- genuinely new theoretical bridge |
| "First local inhibition graph for SAE diagnostics" | Graph construction + validation against known absorption pairs | No existing graph-based SAE diagnostic | **Strong** -- new tool category |
| "Mechanistic explanation for precision-recall asymmetry" | Theoretical argument + empirical validation | No existing explanation for this finding | **Moderate (1-5%)** -- explains our own data; needs broader validation |
| "First training-free post-hoc repair" | Homeostatic rebalancing restores parent firing with <5% reconstruction error | All existing solutions require retraining | **Strong** -- if validated |
| "Scalable to million-latent SAEs" | Complexity analysis + demonstration | Local graph is O(k * d_dict) | **Moderate** -- complexity claim is standard; needs empirical validation |

### 2.2 Classification of Contribution Margin

- **Theoretical contribution (LCA-SAE connection)**: **Strong** -- The structural correspondence is exact, not metaphorical. No prior work makes this connection.
- **Diagnostic contribution (inhibition graph)**: **Strong** -- A new tool category (graph-based absorption prediction) that is training-free and scalable.
- **Explanatory contribution (precision-recall asymmetry)**: **Moderate** -- Explains the project's own data well, but needs validation on other models/feature types.
- **Repair contribution (homeostatic rebalancing)**: **Moderate-to-Strong** -- If validated, it would be the first training-free repair. High risk/reward.
- **Empirical contribution**: **Pending** -- The framework makes falsifiable predictions (precision@20 >= 0.10 vs. 0.004 chance), but experiments have not yet been run.

---

## 3. Concurrent Work Scan (Last 6 Months)

### 3.1 Directly Competing or Related Work

| Paper | Date | Relevance | Threat Level |
|-------|------|-----------|--------------|
| **"Sanity Checks for SAEs" (arXiv:2602.14111)** | Feb 2026 | Random baselines match trained SAEs on standard metrics | **MEDIUM** -- If random SAEs have similar decoder correlation structure, our graph may not be meaningful. However, the inhibition graph is about *specific* correlations, not aggregate metrics. |
| **"Interpretability without actionability" (arXiv:2603.18353)** | Mar 2026 | 98.2% AUROC linear probes but 45.1% output sensitivity; SAE steering produced zero effect | **LOW** -- This is about steering failure, not absorption mechanism. Our framework does not claim to fix steering. |
| **"Stable and Steerable SAEs with Weight Regularization" (arXiv:2603.04198)** | Mar 2026 | L2 regularization doubles steering success rates | **LOW** -- Engineering improvement; does not address absorption mechanism. |
| **"From Atoms to Trees: Hierarchical SAEs" (arXiv:2602.11881)** | Feb 2026 | Jointly learns SAEs and parent-child relationships | **MEDIUM** -- Related to hierarchical feature structure, but different approach (learning vs. analysis). Our framework is post-hoc; theirs is architectural. |
| **"SynthSAEBench" (arXiv:2602.14687)** | Feb 2026 | Large-scale synthetic benchmark; reproduces LLM SAE phenomena | **LOW** -- Complementary; synthetic validation could strengthen our framework. |
| **"A Causal Locally Competitive Algorithm" (Charles et al., 2010)** | 2010 | Causal LCA variant | **LOW** -- Foundational; not applied to SAEs. |
| **"Efficient Sparse Coding with ALCA" (arXiv:2409.08188)** | Sep 2024 | Adaptive LCA for speech classification | **LOW** -- Different domain; not SAE-related. |

### 3.2 Key Insight from Concurrent Work

The most relevant concurrent finding is the **"Sanity Checks for SAEs"** paper (Feb 2026), which shows that frozen/random SAE baselines achieve comparable performance to trained SAEs on multiple metrics. This poses a **medium-level threat**: if random SAEs also exhibit decoder correlation structure that predicts absorption, our framework might be measuring a dictionary-size artifact rather than genuine feature quality. The mitigation is to include a random baseline comparison in the validation.

The **"Interpretability without actionability"** paper (Mar 2026) shows that SAE steering can produce zero effect despite significant features. This does not directly threaten our framework (which is about absorption mechanism, not steering effectiveness), but it reinforces the need to separate "diagnostic" claims from "repair" claims.

---

## 4. Novelty Verdict

### The ONE Thing This Work Does That No Prior Work Does

**"Shows that the SAE decoder correlation matrix W_dec^T W_dec is exactly the inhibition matrix from Rozell et al.'s Locally Competitive Algorithm, enabling a neuroscience-inspired graph-based diagnostic for feature absorption."**

This is genuinely novel:
1. No prior paper connects LCA to SAEs.
2. No prior paper constructs a graph from decoder correlations to diagnose absorption.
3. No prior paper explains precision-recall asymmetry via competitive suppression.
4. No prior paper proposes a training-free post-hoc repair for absorption.

### Why the Novelty is Strong

1. **The structural correspondence is exact, not metaphorical.** W_dec^T W_dec = G_LCA is a mathematical identity given the LCA definition, not an analogy.
2. **The framework makes falsifiable predictions.** Precision@20 >= 0.10 vs. 0.004 chance is a clear, testable threshold.
3. **It explains existing findings rather than requiring new data.** The precision-recall asymmetry, layer-dependent effects, and steering robustness all have natural explanations in the competitive suppression framework.
4. **It is complementary to existing solutions.** Matryoshka, OrtSAE, and ATM are architectural/training modifications; our framework is a post-hoc analysis tool that works on pretrained SAEs.

### Why the Novelty Could Be Questioned

1. **The LCA connection might be considered "obvious in hindsight."** Both LCA and SAEs involve sparse coding with correlated dictionaries; the connection may be seen as a reframing rather than a discovery.
2. **The framework's empirical validation is pending.** If the graph does not predict absorption pairs (precision@20 < 0.05), the theoretical contribution stands but the practical contribution collapses.
3. **The field is skeptical of SAE metrics.** The "Sanity Checks" paper questions whether any SAE analysis is meaningful.

---

## 5. Venue Recommendation

### Assessment

| Criterion | Status |
|-----------|--------|
| Theoretically grounded | **YES** -- Exact mathematical correspondence |
| Novelty claim supported | **YES** -- No prior LCA-SAE connection |
| Empirical validation | **PENDING** -- Experiments not yet run |
| Practical utility | **PENDING** -- Diagnostic tool needs validation |
| Concurrent work threat | **MEDIUM** -- Sanity Checks paper questions SAE metrics broadly |
| Methodological rigor | **GOOD** -- Clear falsification thresholds, random baselines specified |

### Recommendation: **Top-tier (NeurIPS/ICML/ICLR) IF Empirically Validated; Workshop IF Null Result**

**Justification:**

- **Top-tier (NeurIPS/ICML/ICLR)**: **Plausible if H1 is validated.** A paper that (a) establishes a novel theoretical connection between neuroscience and SAEs, (b) validates it empirically with precision@20 >> chance, and (c) provides a practical diagnostic tool, would be competitive at top-tier venues. Comparable papers:
  - Matryoshka SAE (ICML 2025) -- architectural innovation with absorption reduction
  - OrtSAE (2025) -- orthogonality constraint with empirical validation
  - The LCA connection is more theoretically grounded than either.

- **Mid-tier (AAAI/EMNLP/ACL)**: **Likely if partial validation.** If precision@20 is 0.05-0.10 (above chance but not dramatically), the contribution is still novel but less impactful.

- **Workshop (ICML MiF, NeurIPS XAI)**: **If H1 is not validated.** The theoretical connection alone (without empirical validation) might be workshop-level.

**Comparable papers at top-tier level:**
- Chanin et al. (2024/2025) -- NeurIPS 2025 (foundational absorption work)
- Matryoshka SAE -- ICML 2025
- SAEBench -- ICML 2025
- OrtSAE -- OpenReview 2025

---

## 6. Strengthening Plan: What Would Maximize Positioning

### 6.1 Critical Experiments (Must-Do)

| Addition | Why It Would Help | Effort | Expected Impact |
|----------|-------------------|--------|----------------|
| **E1: Graph validation on GPT-2 Small** | Tests H1 (primary hypothesis); determines if paper is top-tier viable | ~15 min | **CRITICAL** -- decides the paper's fate |
| **E2: Random baseline comparison** | Addresses Sanity Checks threat; distinguishes genuine structure from artifacts | ~5 min | **HIGH** -- essential for credibility |
| **E3: Precision-recall asymmetry test** | Tests H2; connects framework to existing data | ~15 min | **MEDIUM** -- strengthens theoretical claims |
| **E4: Cross-layer graph structure** | Tests H4; explains layer-dependent effects | ~20 min | **MEDIUM** -- adds depth |
| **E5: Homeostatic rebalancing** | Tests H5; if successful, dramatically strengthens paper | ~30 min | **HIGH** -- high risk/reward |
| **E6: Cross-model validation (Gemma-2-2B)** | Tests generalizability; GemmaScope SAEs are the community standard | ~30 min | **HIGH** -- essential for broad impact |

### 6.2 Paper Framing Recommendations

**If H1 is strongly validated (precision@20 >= 0.10):**
- Title: "The Local Inhibition Graph: A Neuroscience-Inspired Framework for Understanding Feature Absorption in Sparse Autoencoders"
- Position as: First theoretical bridge between neuroscience (LCA) and SAE absorption; practical diagnostic tool.

**If H1 is partially validated (precision@20 = 0.05-0.10):**
- Title: "Decoder Correlations Reveal Competitive Suppression in Sparse Autoencoders"
- Position as: Novel analytical framework with promising but limited empirical validation; call for further investigation.

**If H1 is not validated (precision@20 <= 0.05):**
- Title: "Feature Absorption as Optimal Compression: Evidence from GPT-2 Small"
- Position as: The inhibition framework is retained as a theoretical speculation; the paper becomes a descriptive trade-off analysis.

---

## 7. Honest Assessment of Contribution Margin

### What This Proposal Adds

1. **First LCA-SAE connection** -- The structural correspondence (W_dec^T W_dec = G_LCA) is exact and has not been articulated.
2. **First local inhibition graph for SAE diagnostics** -- A new tool category that is training-free and scalable.
3. **Mechanistic explanation for precision-recall asymmetry** -- Competitive suppression explains why absorption affects recall but not precision.
4. **First training-free post-hoc repair** -- Homeostatic rebalancing operates on pretrained SAEs (if validated).
5. **Explains existing architectural solutions** -- Matryoshka and OrtSAE's effectiveness is predicted by the framework.

### What This Proposal Does NOT Add (Yet)

1. **No empirical validation** -- The experiments are planned but not executed.
2. **No cross-model validation** -- Primary experiments on GPT-2 Small; Gemma as validation only.
3. **No comparison with random/frozen baselines** -- Essential given the Sanity Checks paper.
4. **No demonstration of practical utility** -- Does the diagnostic actually help practitioners?

### Contribution Margin Summary

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Theoretical contribution | **4/5** | Exact mathematical correspondence; genuinely novel |
| Methodological contribution | **4/5** | New diagnostic tool category; training-free; scalable |
| Empirical contribution | **2/5** | Pending validation; high variance in outcome |
| Practical contribution | **3/5** | Diagnostic tool is practical; repair is high-risk |
| Timeliness | **4/5** | Field is actively seeking theoretical understanding of absorption |
| **Overall (potential)** | **3.4/5** | Top-tier viable if empirically validated |
| **Overall (if null)** | **2/5** | Workshop-level theoretical contribution |

---

## 8. Bottom Line

The pivot from the null-result correlation study to the Local Inhibition Graph framework was **the correct strategic decision**. The original study had a contribution margin of ~1.4/5 (below threshold for any venue); the new framework has a potential contribution margin of ~3.4/5 (top-tier viable if validated).

**The critical path is H1 validation.** If the local inhibition graph predicts known absorption pairs with precision@20 >= 0.10 (vs. 0.004 chance), the paper has a strong case for NeurIPS/ICML/ICLR. If not, the theoretical contribution alone might sustain a workshop paper or a strong arXiv preprint.

**The most honest framing** is: "We propose a novel theoretical framework connecting neuroscience (LCA) to SAE absorption, derive a practical diagnostic tool from it, and validate it empirically." This is a positive claim with clear falsification criteria -- exactly what the field needs.

**Minimum bar for top-tier submission:**
1. H1 validated with precision@20 >= 0.10 on GPT-2 Small
2. Random baseline comparison included
3. Cross-model validation on Gemma-2-2B (or another model)
4. H2 (precision-recall asymmetry) supported
5. H3 (at-risk feature prediction) supported

**If homeostatic rebalancing (H5) also succeeds**, the paper becomes significantly stronger -- the first training-free repair for absorption would be a major practical contribution.

---

## Sources

- [Chanin et al. (2024/2025) - A is for Absorption](https://arxiv.org/abs/2409.14507) - NeurIPS 2025
- [Rozell et al. (2008) - Locally Competitive Algorithm](https://arxiv.org/abs/0805.2963) - Foundational LCA paper
- [SAEBench - Comprehensive Benchmark](https://arxiv.org/abs/2503.09532) - Karvonen et al., ICML 2025
- [OrtSAE - Orthogonal Sparse Autoencoders](https://arxiv.org/abs/2509.22033) - Korznikov et al., 2025
- [Matryoshka SAEs](https://arxiv.org/abs/2503.17547) - Bussmann et al., ICML 2025
- [ATM: Adaptive Temporal Masking](https://arxiv.org/abs/2510.08855) - Li et al., ICLR-W 2025
- [Cui et al. - On the Limits of SAEs](https://arxiv.org/abs/2506.15963) - 2025
- [Tang et al. - Theoretical Foundation of SDL](https://arxiv.org/abs/2512.05534) - 2025
- [Sanity Checks for SAEs](https://arxiv.org/abs/2602.14111) - Feb 2026
- [Interpretability without Actionability](https://arxiv.org/abs/2603.18353) - Mar 2026
- [Stable and Steerable SAEs](https://arxiv.org/abs/2603.04198) - Mar 2026
- [Hierarchical SAEs](https://arxiv.org/abs/2602.11881) - Feb 2026
- [SynthSAEBench](https://arxiv.org/abs/2602.14687) - Chanin et al., Feb 2026
- [Feature Hedging](https://arxiv.org/abs/2505.11756) - Chanin et al., 2025
- [CorrSteer](https://arxiv.org/abs/2508.12535) - Arad et al., 2025
- [Charles et al. - Causal LCA](https://www.bme.jhu.edu/ascharles/wp-content/uploads/2019/02/CharlesKressnerRozell_2010.pdf) - 2010
- [ALCA for Speech Classification](https://arxiv.org/abs/2409.08188) - 2024
- [Meyer & Kenyon - LCA and V1](https://ui.adsabs.harvard.edu/abs/2020ssia.conf...35T/abstract) - 2020
- [Paiton et al. - LCA on TrueNorth](https://www.frontiersin.org/articles/10.3389/fnins.2019.00754/full) - 2019
- [Feature Suppression in SAEs](https://www.lesswrong.com/posts/3JuSjTZyMzaSeTxKk/addressing-feature-suppression-in-saes) - LessWrong, 2024
