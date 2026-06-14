# Supervisor Review: Feature Absorption in Sparse Autoencoders

**Score: 6 / 10** | **Verdict: REVISE**
**Date: 2026-05-01**

---

## Executive Summary

The paper introduces a legitimate research question (feature absorption in SAEs) with a reproducible metric and honestly reported negative results. The central finding — an inverted-U layer-dependence pattern with absorption peaking at mid-layers (49.3% at layer 4 vs 0.19% at layer 8) — is a genuine discovery worth reporting. However, the paper cannot pass a quality gate at this score due to four critical issues: (1) the H4 conclusion is drawn from a fundamentally flawed experiment that cannot support it; (2) the 8 perfect-score latents (each firing on exactly 100 tokens = number of sequences) strongly suggest positional artifacts; (3) H2 was never tested despite layer 4 having 260x more absorbed latents; (4) only pilot-scale (n=100) experiments are reported, not the stated full-scale (n=1024).

The honest negative result reporting is the paper's strongest aspect. The contribution framing as "boundary conditions for SAE trust" is weakened by 4 of 5 hypotheses failing or being skipped.

---

## Dimension Scores

| Dimension | Score | Assessment |
|-----------|-------|------------|
| **Novelty** | 6 | The absorption metric and the layer-dependence finding are new contributions. However, the framing as a "falsification study" with 4/5 hypotheses falsified/skipped limits the novelty impact. The contribution is more modest than the abstract suggests. |
| **Soundness** | 5.5 | The metric is well-defined and validated against random controls. However, H4's causal conclusion is unsupported by the experiment design. The H1 internal inconsistency (layer 4 confirms vs layer 8 falsifies) creates ambiguity. |
| **Experiments** | 5 | Pilot-scale (n=100) results only, not the stated full-scale (n=1024). H2 untested. H4 experiment design flaw. No seed ablation. 8 perfect-score latents uninvestigated. |
| **Reproducibility** | 6 | Metric definition is precise. Code not attached but methodology is detailed enough. All 5 figures exist in writing/latex/figures/ and compile correctly. H2/H6 pending analysis is a gap. |

---

## Critical Issues

### 1. H4 Conclusion Is Unsupported (Critical, Soundness)

**Finding**: The paper states that "absorption does not predict circuit-level causal importance" based on H4 results. This is wrong. The experiment compared **10% latent subsets** (bottom/top 10% by absorption) at layer 8, but **both subsets yielded 0.0 faithfulness**. Zeroing 90% of latents destroys reconstruction capacity entirely — this tests dictionary completeness, not absorption quality.

**The correct experiment** — compare **full SAE at layer 4** (49.3% absorption) vs **full SAE at layer 8** (0.19% absorption) on the France/Paris circuit — was **never conducted**.

**Evidence**: h4_pilot_output.log (both low/high absorption subsets = 0.0), pilot_summary.md (Section on H4: "A better approach would be to compare full SAEs at different layers").

**Suggestion**: Remove the H4 causal conclusion. Run the correctly designed experiment or frame H4 as future work. Current data only supports "subset patching destroys signal due to dictionary incompleteness," not "absorption level predicts importance."

---

### 2. 8 Perfect-Score Latents Likely Positional Artifacts (Critical, Experiment)

**Finding**: 8 latents achieve `Af = 1.0` (maximum absorption), each firing on **exactly 100 tokens** — precisely the number of sequences in the pilot corpus. This regularity is a telltale sign of a positional artifact (e.g., position embedding-related features firing at the same absolute position across all sequences), not genuine semantic absorption.

**Evidence**: h5_pilot_output.log: "Top 5 most absorbed latents (dict=24576): 1. Subsampled idx 6955: score=1.0000, activations=100". The paper acknowledges "suspicious uniformity" in Section 6.5 but does not investigate.

**Note**: Always-on features (>90% of tokens) are excluded from analysis as trivial co-firers. These 8 latents (0.78% of tokens each, exactly 100 per sequence) may warrant the same treatment.

**Suggestion**: Compute token position consistency across sequences for each perfect-score latent. If confirmed as positional artifacts, exclude from primary analysis and report in Appendix. This directly affects the validity of the Af>0.5 threshold, especially at layer 4 where 25.1% of latents score exactly 1.0.

---

## Major Issues

### 3. H2 Untested Despite Adequate Power at Layer 4 (Major, Experiment)

**Finding**: The paper cites "insufficient absorption variance at layer 8" as reason for not testing H2. This is self-defeating: layer 4 has 49.3% absorption (~12,000 latents with Af>0.5) — **260x more data** than layer 8 (46 latents). The early termination rationale is incoherent.

**Evidence**: h3_pilot_PROGRESS.json layer_results (layer 4: 49.3%, layer 8: 0.19%). The pilot_summary.md explicitly states: "layer 4 has 260x more absorbed latents than layer 8."

**Suggestion**: Run H2 analysis on layer 4 where absorption prevalence is highest. Alternatively, use a continuous absorption measure (mean RVE across all activating tokens) rather than binary threshold to increase sensitivity.

---

### 4. Only Pilot-Scale Experiments Reported (Major, Experiment)

**Finding**: The methodology (Section 4.2) specifies "1,024 sequences for full experiments." All results use only 100 sequences. The 10x scale gap is material for rare phenomenon characterization.

**Evidence**: pilot_summary.json confirms n=100 only. writing/outline.md states "Full experiment with 1,024 sequences may show different patterns."

**Suggestion**: Either run full-scale experiments (n=1024) or explicitly scope the paper as a pilot study with commitment to full-scale replication. Do not present pilot results as if they are full-scale findings.

---

### 5. H1 Internal Inconsistency (Major, Analysis)

**Finding**: The hypothesis predicts ">20% prevalence at layers 4-10." At layer 8: 0.19% (falsified). At layer 4: 49.3% (not falsified). The paper claims "H1 falsified" but layer 4 results contradict rather than support that conclusion.

**Note**: The abstract's phrasing "falsifying the hypothesis at that layer" is accurate given the layer-8-specific claim. However, Section 4.3 and 5.6 should more explicitly address why the collective criterion (<10% across layers 4-10) is satisfied by layer 8 alone when layer 4 exceeds 20%.

**Suggestion**: Restate H1 precisely: specify layer 8 as the primary test layer, or redefine to predict layer-specific absorption rates. The layer discrepancy (0.19% vs 49.3%) is itself a finding consistent with the H3 inverted-U pattern — state this explicitly.

---

## Minor Issues

### 6. No Seed Ablation (Minor, Experiment)

**Finding**: All experiments use seed=42 with no confirmation across seeds. The dramatic layer difference (49.3% vs 0.19%) could be a seed artifact.

**Suggestion**: Run key experiments (H1 layer comparison, H3 inverted-U) with 2-3 seeds before finalizing.

---

### 7. H5 Framing Overstates Practical Significance (Minor, Analysis)

**Finding**: "10-fold reduction" (2K→24K) is directionally correct but 97.75% vs 99.81% non-absorbed. The paper acknowledges this in Section 5.4 but abstract framing persists.

**Suggestion**: Lead with absolute numbers: "97.75% non-absorbed at 2K vs 99.81% at 24K" to avoid overstating a 2 percentage point difference.

---

## What Works Well

1. **Exemplary negative result reporting**: The paper correctly labels H4 as "uninformative" rather than "falsified," acknowledging the design flaw. Numbers are specific and not rounded away.

2. **Precise metric definition**: The absorption score formula (5 steps: activating tokens → top-5 co-firers → partial reconstruction → RVE → threshold) is fully reproducible. Validation against random controls is good scientific practice.

3. **Layer-dependence as central contribution**: The 100x gap between layer 4 and layer 8 is a real finding — absorption is not uniformly rare but peaks at mid-layers. This inverted-U pattern is the paper's strongest contribution.

4. **Figures are present and compile**: All 5 figures (gen_pipeline.pdf, fig1_layer_absorption.pdf, fig2_dict_size.pdf, fig4_layer4_histogram.pdf, fig3_faithfulness.pdf) exist in writing/latex/figures/ and the LaTeX compiles without missing figure warnings. This resolves the previous review's critical production issue.

5. **Random control validation**: Consistent reporting of random dictionary baselines (0.00% at all sizes) confirms the metric detects genuine structure.

---

## Evidence Cross-Validation

| Claim | Paper Location | Source Data | Status |
|-------|---------------|-------------|--------|
| H1 layer 8: 0.19% (46/24,576) | Section 5.1 | h1_pilot.json | Confirmed |
| H1 layer 4: 49.3% | Section 5.2 (H3 context) | h3_pilot_PROGRESS.json | Confirmed |
| H3 inverted-U: layer 4 peak (49.3%), layer 8 (20.9%) | Table 2 | h3_pilot_PROGRESS.json | Confirmed |
| H4: raw residual = 0.400, SAE all = 0.289, subsets = 0.000 | Section 5.3 | h4_pilot_output.log | Confirmed |
| H5: 2K=2.25%, 24K=0.19% | Section 5.4 | h5_pilot_output.log | Confirmed |
| 8 perfect-score latents, each 100 tokens | Section 5.1 | h5_pilot_output.log | Confirmed |
| Figures exist and compile | All figure refs | writing/latex/figures/*.pdf | Confirmed |

All paper numbers match source data. No fabrication detected.

---

## Risks

1. H4 conclusion drawn from incorrectly designed experiment may not survive reviewer scrutiny
2. 8 perfect-score latents may be positional artifacts that invalidate Af>0.5 threshold at high absorption levels
3. H1 internal inconsistency (layer 4 vs layer 8 contradiction) undermines the falsification claim
4. H2 untested undermines "systematic study" framing — paper studies 5 hypotheses but only tests 4
5. Pilot-scale (n=100) results may differ materially at full scale (n=1024) for rare phenomena
6. No seed ablation means layer discrepancy (49.3% vs 0.19%) could be artifact

---

## What Would Raise the Score to ~7.5

1. **Run H4 correctly**: compare full SAE at layer 4 vs layer 8 on France/Paris circuit (not latent subsets)
2. **Investigate or exclude** the 8 perfect-score latents as positional artifacts
3. **Run H2 on layer 4** where n~12,000 provides adequate power
4. **Run full-scale (n=1024) experiments**
5. **Add seed ablation** for H1 layer comparison

These five steps address all critical and major issues and would move the score to approximately 7.5.

---

## Verdict Rationale

**Score: 6 — Borderline Reject (Top 40%)**

The paper has a legitimate research contribution (the layer-dependence finding) but cannot pass at this score because:
- H4's central claim is unsupported by the experiment actually conducted
- 8 perfect-score latents remain uninvestigated despite being potential positional artifacts
- H2 (critical path) was never tested
- Only pilot-scale experiments are reported

The honest negative result reporting and the reproducible metric are genuine strengths. With the five corrective steps above, the paper could reach 7.5 (Borderline Accept).