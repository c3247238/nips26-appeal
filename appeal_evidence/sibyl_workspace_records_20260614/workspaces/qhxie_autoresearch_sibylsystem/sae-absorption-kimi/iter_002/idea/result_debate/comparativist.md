# Comparativist Analysis: Positioning the SAEBench Absorption Construct-Validity Study

**Date:** 2026-04-16  
**Workspace:** `/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current`  
**Source Results:** `exp/results/full/statistical_analysis_summary.json`

---

## 1. Baseline Landscape: What Already Exists

### 1.1 Foundational Work

| Method / Paper | Year | Core Contribution | Absorption Metric Scope |
|---------------|------|-------------------|------------------------|
| Chanin et al., "A is for Absorption" | 2024 | Introduced feature absorption; first-letter probe protocol | First-letter only |
| SAEBench (Karvonen et al.) | 2025 | Standardized 8-eval benchmark including absorption | First-letter only |
| Matryoshka SAEs (Bussmann et al.) | 2025 | Nested dictionaries; absorption ~0.05 vs BatchTopK ~0.49 | First-letter only |
| OrtSAE (Korznikov et al.) | 2025 | Orthogonality penalty; -65% absorption on SAEBench | First-letter only |
| HSAE (Zhan et al.) | 2026 | Explicit tree-structured constraints | First-letter only |

**Key observation:** Every architecture paper that reports absorption improvements uses the *same* first-letter benchmark. No prior work systematically tests whether first-letter absorption predicts semantic-hierarchy absorption.

### 1.2 Published Absorption Numbers (First-Letter, SAEBench)

From the literature, the canonical first-letter absorption scores on Gemma-2-2B (layer 12, dict size 65k, L0 ~40) are approximately:

| Architecture | Absorption Rate (lower = better) | Source |
|-------------|----------------------------------|--------|
| Matryoshka SAE | ~0.05 | Bussmann et al. (2025) |
| OrtSAE | ~0.17 reduction from BatchTopK (~0.32 effective) | Korznikov et al. (2025) |
| BatchTopK | ~0.49 | SAEBench |
| Standard ReLU | ~0.45–0.55 | SAEBench |
| TopK / JumpReLU | Worse than ReLU at low L0 | SAEBench |

Our experiment uses **Pythia-160M** (not Gemma-2-2B), so direct numerical comparison is confounded by model scale and layer. However, the *relative* architecture ranking on first-letter in our data (BatchTopK highest, TopK highest, Gated/PAnneal lowest) partially aligns with SAEBench findings for TopK/JumpReLU underperforming on absorption.

---

## 2. Contribution Margin: Our Results vs. The Field

### 2.1 What We Actually Found

From `statistical_analysis_summary.json`:

| Hypothesis | Result | Verdict |
|-----------|--------|---------|
| **H1 (Construct Validity)** | Pearson r = 0.463, 95% CI [-0.389, 0.981] | **INCONCLUSIVE** |
| **H2 (Hierarchy Specificity)** | Semantic absorption (0.235) < Non-hierarchy control (0.331), t = -4.75, p = 0.003 | **REJECTED** |
| **H3 (Robustness)** | r stable across tau_fs (0.01, 0.03, 0.05) ~0.47 | **INCONCLUSIVE** |

### 2.2 Honest Assessment of Contribution Margin

**This is not a "we beat SOTA" paper.** This is a **methodological diagnostic** paper. The contribution margin must be judged on:

1. **Novelty of the question asked** (first construct-validity test of a dominant benchmark).
2. **Implications of the answer** (inconclusive-to-negative results challenge a field-wide assumption).

The *empirical delta* is not a percentage improvement over a baseline. It is the **absence of a predicted correlation** (r < 0.6) and the **reversal of hierarchy specificity** (non-hierarchy control scores are *higher* than semantic hierarchy, opposite to expectation).

**Classification:** The result is a **moderate-to-strong methodological contribution** if framed as a falsification study, but a **marginal empirical contribution** if judged purely on effect-size magnitude.

---

## 3. Concurrent Work Scan

### 3.1 Directly Related Recent Papers

| Paper | Date | Relevance | Threat Level |
|-------|------|-----------|--------------|
| "Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data" (arXiv:2602.14687) | Feb 2026 | Argues SAEBench metrics are "indirect proxies"; calls for ground-truth validation | **Moderate** — overlaps in critique spirit, but does not test semantic hierarchy absorption specifically |
| "Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures" (2025) | 2025 | Shows hierarchical architectures reduce absorption; uses first-letter benchmark | **Low** — complementary; they assume the metric is valid |
| "Do Sparse Autoencoders Generalize? A Case Study in Answerability" (ICML 2025 Workshop) | 2025 | Shows inconsistent OOD generalization of SAE features on real semantic tasks | **Moderate** — supports our broader narrative that SAE feature quality metrics may not generalize |
| HSAE (Zhan et al., arXiv:2602.11881) | 2026 | Hierarchical SAE with explicit parent-child constraints | **Low** — they report SAEBench absorption improvements, but do not validate the metric itself |

### 3.2 Threat Assessment

- **No concurrent paper has performed the exact experiment we ran** (first-letter vs. semantic-hierarchy absorption correlation across multiple architectures).
- However, **multiple 2025–2026 papers are questioning SAEBench metric validity** from different angles (synthetic data, OOD generalization, indirect proxies). This means the *narrative space* is becoming more crowded.
- The **construct-validity question is still novel** at the time of writing, but the window may close within 6–12 months as the community intensifies benchmark criticism.

---

## 4. Novelty Verdict

> **"What is the ONE thing this work does that no prior work does?"**

**Answer:** It is the first study to empirically test whether the dominant SAEBench feature-absorption metric—used to validate Matryoshka SAEs, OrtSAE, and HSAE—actually generalizes from first-letter tasks to real semantic hierarchies.

**Verdict:** The novelty is **genuine but narrow**. It is a "first systematic test of X" paper, not a "we invented X" paper. The value lies in the **falsifiability of the design** and the **stakes of the question**, not in a new algorithm or architecture.

---

## 5. Venue Recommendation

| Criterion | Assessment |
|-----------|------------|
| Contribution margin | Moderate (methodological diagnostic, not algorithmic advance) |
| Novelty | Genuine but narrow; "first test of X" |
| Experimental rigor | High (controlled conditions, multiple architectures, bootstrap CIs, replication on GPT-2) |
| Result strength | Mixed/negative (inconclusive correlation, rejected specificity) — honest but not headline-grabbing |
| Timeliness | High (absorption is a central SAE topic in 2025–2026) |

### Recommended Venues

1. **Primary target: ICLR Workshop on Mechanistic Interpretability** or **NeurIPS MI Workshop**
   - Workshop is appropriate because the contribution is a focused methodological finding rather than a full algorithmic paper.
   - The SAE community is highly active at these workshops.

2. **Stretch target: EMNLP Findings** or **AAAI**
   - If the paper is expanded with stronger theoretical framing and additional ablations, it could reach a mid-tier conference.
   - The negative result + benchmark critique angle fits well with EMNLP's increasing interest in evaluation validity.

3. **Not recommended for top-tier main conference (NeurIPS/ICML/ICLR)** without additional work
   - A single negative-result correlation study on a small model (Pythia-160M) with inconclusive statistics is unlikely to clear the empirical bar for a top-tier ML venue.

---

## 6. Strengthening Plan: 3 Specific Additions

### 6.1 Add a Larger-Scale Replication (Gemma-2-2B)

**Why:** All SOTA absorption papers report on Gemma-2-2B. Pythia-160M is a valid pilot, but reviewers will ask whether the inconclusive correlation holds on the model where Matryoshka/OrtSAE claims were made.

**What:** Run first-letter + semantic-hierarchy absorption on 3–4 Gemma-2-2B SAEs (Standard, TopK, Matryoshka, OrtSAE if available). This directly anchors the study to the SOTA benchmark setting.

### 6.2 Add a Causal Ablation or Intervention

**Why:** The current study is purely correlational (probe-based). Chanin et al. (2024) used causal ablations to confirm absorption. Adding even a lightweight causal check (e.g., latent ablation on semantic parent features) would elevate the paper from "metric correlation" to "mechanistic validation."

**What:** For 2–3 SAEs, perform activation patching or latent suppression on semantic parent latents and measure downstream impact on child-feature prediction.

### 6.3 Expand the Semantic-Hierarchy Dataset

**Why:** The current study uses 10 WordNet parent-child pairs. A reviewer could argue this is too small to draw general conclusions. Also, some hierarchies showed very low absorption (fruit: 0.097) while others were high (container: 0.390), suggesting concept-dependent variance.

**What:** Expand to 20–30 hierarchies, stratified by concept type (natural kinds, artifacts, social categories). Report per-category absorption variance and test whether first-letter correlation improves within specific concept classes.

---

## 7. Summary

| Dimension | Score / Verdict |
|-----------|-----------------|
| Baseline comparison | Well-grounded in SAEBench literature |
| Contribution margin | Moderate (methodological, not algorithmic) |
| Concurrent work threat | Low-to-moderate; no direct overlap yet |
| Novelty | Genuine: first construct-validity test of SAEBench absorption |
| Venue recommendation | **Workshop (ICLR/NeurIPS MI)** as-is; **EMNLP Findings / AAAI** with Gemma-2B replication |
| Biggest weakness | Small model (Pythia-160M), inconclusive correlation, rejected H2 |
| Biggest strength | Asks a high-stakes question no one else has asked; rigorous experimental design |

---

## Sources

- [SAEBench: A Comprehensive Benchmark for Sparse Autoencoders](https://arxiv.org/abs/2503.09532)
- ["A is for Absorption" (Chanin et al., 2024)](https://arxiv.org/abs/2409.14507)
- [Matryoshka SAEs (Bussmann et al., 2025)](https://arxiv.org/abs/2503.17547)
- [OrtSAE (Korznikov et al., 2025)](https://arxiv.org/abs/2509.22033)
- [HSAE / "From Atoms to Trees" (Zhan et al., 2026)](https://arxiv.org/abs/2602.11881)
- ["Do Sparse Autoencoders Generalize?" (ICML 2025 Workshop)](https://arxiv.org/abs/2502.19964)
- ["Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data" (2026)](https://arxiv.org/abs/2602.14687)
