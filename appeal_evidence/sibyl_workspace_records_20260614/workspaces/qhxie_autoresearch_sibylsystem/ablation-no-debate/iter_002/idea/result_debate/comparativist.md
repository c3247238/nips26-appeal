# Comparative Analysis: Encoder-Driven Feature Absorption in SAEs

## 1. Baseline Landscape

### Top Existing Methods on Absorption-Related Benchmarks

| Method | Year | Venue | Absorption Metric | Key Result | Our Comparable Result |
|--------|------|-------|-------------------|------------|----------------------|
| **Chanin et al.** | 2024/2025 | NeurIPS 2025 | Absorption rate (ablation-based) | 0.49 on Gemma-2-2B (BatchTopK) | N/A (different metric) |
| **Matryoshka SAE** (Bussmann et al.) | 2025 | arXiv | Absorption rate | 0.49 -> 0.05 (10x reduction) | N/A (architectural solution) |
| **OrtSAE** (Korznikov et al.) | 2025 | OpenReview | Absorption rate | 65% reduction at L0=70 | N/A (training-time modification) |
| **HSAE** (Luo et al.) | 2026 | arXiv | Absorption score | Substantially outperforms baselines | N/A (hierarchical architecture) |
| **AdaptiveK SAE** | 2025 | arXiv | SAEBench absorption | Best absorption on SAEBench | N/A (dynamic sparsity) |
| **JumpReLU SAE** | 2024 | arXiv | SAEBench absorption | 0.0114 vs TopK 0.1402 | N/A (different architecture) |
| **SAEBench** (Karvonen et al.) | 2025 | ICML 2025 | 8-metric benchmark | Proxy metrics do not predict absorption | Confirmed (L0-absorption relationship is inverse) |

**Critical Note**: Our work does not directly report on the standard "absorption rate" metric from Chanin et al. (which requires causal ablation). Instead, we measure "multi-child proportional absorption" via Jaccard overlap on synthetic data. This is a **different metric on different data**, making direct numerical comparison impossible.

### What We Can Compare

| Dimension | Chanin et al. | Our Work |
|-----------|--------------|----------|
| **Metric** | Ablation-based absorption rate | Jaccard overlap (multi-child proportional) |
| **Data** | Real LLM SAEs (Gemma-2-2B) | Synthetic hierarchical data |
| **Mechanism claim** | Sparsity penalty drives absorption | **Encoder alignment drives absorption** (new claim) |
| **Decoder role** | Implicit (both encoder and decoder trained jointly) | **Explicitly shown negligible** (0.011 effect vs 0.843 encoder effect) |
| **Depth limitation** | Layers 0-17 only | N/A (synthetic, all "layers" equivalent) |

## 2. Contribution Margin

### Quantitative Deltas

| Finding | Our Result | Prior Understanding | Delta | Classification |
|---------|-----------|---------------------|-------|----------------|
| Encoder vs decoder effect | Encoder: 0.843, Decoder: 0.011 (80x ratio) | Both contribute jointly | **New decomposition** | Strong (new mechanistic insight) |
| Trained vs random SAE absorption | 0.477 vs 0.033 (p=3.85e-10) | Known that trained > random | Confirms with multi-seed | Moderate (replication with new metric) |
| Hierarchy strength -> absorption | Monotonic increase (0.416 -> 0.544) | Known qualitatively | Quantified relationship | Moderate |
| L0 sparsity -> absorption | **Inverse** relationship (L0=20: 0.552, L0=50: 0.419) | Hypothesized positive | **Opposite finding** | Strong (surprising result) |
| Safety feature absorption | No difference (p=0.989) | Not tested | Negative result | Marginal (informs scope) |
| Steering sensitivity | No difference (ratio=0.97x, p=0.936) | Hypothesized higher for absorbed | **Robust negative result** | Moderate (falsifies hypothesis) |
| Generalization | Perfect train/test correlation (r=0.998) | Not tested | Confirms stability | Moderate |

### Assessment

- **Strong contributions**: The encoder-driven mechanism finding (80x ratio) is genuinely new and challenges the prevailing narrative that absorption is a joint encoder-decoder phenomenon. The inverse L0-absorption relationship is surprising and counter to the hypothesis.
- **Moderate contributions**: Multi-seed validation, hierarchy strength quantification, and steering negative result add robustness but do not fundamentally change the field's understanding.
- **Marginal contributions**: Safety analysis negative result is informative but narrow in scope.

## 3. Concurrent Work Scan

### Papers from Last 6 Months Addressing Absorption Mechanisms

| Paper | Date | Key Claim | Overlap with Our Work | Threat Level |
|-------|------|-----------|----------------------|--------------|
| **Oursland (2026)** — Decoder-Free SAE | Jan 2026 | Derives SAE from first principles; eliminates decoder entirely; encoder-decoder asymmetry enables absorption | **Directly related**: Our finding that decoder contributes negligibly aligns with Oursland's theoretical elimination of decoder | Medium — Oursland goes further by removing decoder entirely, but is theoretical; we provide empirical factorial decomposition |
| **Luo et al. (2026)** — HSAE | Feb 2026 | Hierarchical structure mitigates absorption | Related: We test hierarchy strength effect; they propose architectural solution | Low — Different approach (architecture vs mechanism) |
| **"Sanity Checks for SAEs"** (arXiv:2602.14111) | Feb 2026 | Questions whether SAEs beat random baselines on many metrics | Related: Our random baseline (0.033) vs trained (0.477) shows clear separation | Low — Our result supports SAEs being distinguishable from random |
| **SynthSAEBench** (Chanin et al., 2026) | Feb 2026 | Ground-truth synthetic benchmark for SAE evaluation | Related: We use synthetic data; they provide standardized benchmark | Low — Complementary; we could validate on their benchmark |
| **"Use SAEs to Discover Unknown Concepts, Not to Act on Known Concepts"** (arXiv:2506.23845) | June 2025 | SAEs underperform baselines on downstream tasks | Related: Our steering negative result aligns with this reframing | Low — Our result is consistent with their broader claim |
| **DeepMind Negative Results** (Smith et al., March 2025) | Mar 2025 | SAEs fail on downstream tasks; deprioritized SAE research | Related: Our steering and safety negative results align | Low — Consistent with broader negative results trend |

### Key Threat: Oursland (2026)

The most relevant concurrent work is **Oursland's "Deriving Decoder-Free Sparse Autoencoders from First Principles"** (arXiv:2601.06478, January 2026). This paper:
- Theoretically derives that encoder-decoder asymmetry is the root cause of absorption
- Proposes eliminating the decoder entirely (decoder-free SAE)
- Achieves 93.4% probe accuracy vs 90.3% for standard SAE with half the parameters

**Our work's relationship**: We provide empirical evidence (via 2x2 factorial) that the decoder contributes negligibly to absorption (effect = 0.011 vs encoder = 0.843), which is consistent with Oursland's theoretical claim. However, Oursland goes further by proposing a solution (remove decoder), while we only diagnose the mechanism. This means our contribution is **partially preempted** — the "encoder matters more than decoder" insight is now supported by both theory (Oursland) and our empirics.

## 4. Novelty Verdict

**The ONE thing this work does that no prior work does:**

> **Provides causal empirical evidence (via controlled 2x2 factorial decomposition) that feature absorption is driven by encoder alignment, not decoder geometry, overturning the implicit assumption that absorption is a joint encoder-decoder phenomenon.**

### Novelty Strength Assessment

| Aspect | Verdict |
|--------|---------|
| **Encoder-driven claim** | Novel — Prior work (Chanin et al.) treats absorption as sparsity-penalty-driven with implicit encoder-decoder joint optimization; no prior work explicitly decomposes and isolates encoder vs decoder contributions |
| **Factorial methodology** | Novel — 2x2 factorial (random/trained encoder x random/trained decoder) is a new experimental design for SAE mechanism analysis |
| **Inverse L0-absorption finding** | Novel — Counter to hypothesis and prior expectations; suggests fewer active features force more absorption |
| **Metric used** | Not novel — Jaccard overlap is standard; multi-child proportional absorption is adapted from existing concepts |
| **Synthetic data** | Not novel — Standard approach for controlled SAE experiments |

**Overall novelty**: Moderate-to-strong. The core mechanistic insight is genuinely new, but the concurrent Oursland (2026) paper provides theoretical support for a similar conclusion, reducing the exclusivity of our contribution.

## 5. Venue Recommendation

### Recommendation: **Workshop or Mid-Tier Conference (AAAI/EMNLP/Findings)**

| Venue Tier | Fit | Justification |
|------------|-----|---------------|
| **Top-tier (NeurIPS/ICML/ICLR)** | Poor | Contribution margin is moderate — a single mechanistic insight with synthetic data only is unlikely to compete with full benchmark evaluations (SAEBench, SynthSAEBench) or architectural innovations (Matryoshka, HSAE, OrtSAE) |
| **Mid-tier (AAAI, EMNLP, COLING)** | Good | The encoder-driven mechanism is a solid empirical finding; negative results (steering, safety) add methodological value; suitable for venues that value rigorous empirical analysis |
| **Workshop (MEI, XAI, SAE-dedicated)** | Best | The narrow scope (single mechanism on synthetic data) and partial overlap with concurrent work make this ideal for a workshop; allows focused discussion with the MI community |
| **Findings/Short papers** | Good | The core result (encoder effect 80x decoder) is concise and impactful enough for a short paper format |

### Comparable Papers at Recommended Venues

- **"A is for Absorption"** (Chanin et al.) — NeurIPS 2025 (full paper with hundreds of SAEs, real models, new metric definition)
- **"Measuring Sparse Autoencoder Feature Sensitivity"** (Hu et al.) — arXiv 2025 (similar scope: single new metric, empirical evaluation)
- **"Feature Hedging"** (Chanin & Garriga-Alonso) — arXiv 2025 (narrow scope, single phenomenon)
- **"Incorporating Hierarchical Semantics in SAE Architectures"** (Muchane et al.) — arXiv 2025 (architectural, smaller scale than Matryoshka/HSAE)

Our work is closer in scope to Hu et al. or Muchane et al. than to Chanin et al. (which is a flagship paper).

## 6. Strengthening Plan

### 2-3 Specific Additions to Maximize Positioning

#### Strengthening 1: Validate on Real SAEs (Critical)

**What**: Replicate the factorial decomposition on pretrained Gemma Scope or GPT-2 SAEs using a training-free proxy for absorption.

**Why**: Our entire result is on synthetic data (d_model=128, synthetic hierarchy). The field's gold standard is real LLM SAEs. Without real-model validation, the result may be dismissed as a synthetic artifact.

**How**:
- Load Gemma Scope SAEs (layer 5, 16k width) via SAELens
- Use encoder-decoder cosine similarity asymmetry as proxy for absorption (following Chanin et al.'s suggestion)
- Compare encoder-only vs decoder-only vs full SAE modifications
- Expected runtime: ~1 hour

**Impact**: Would elevate the paper from "synthetic mechanism" to "validated mechanism across synthetic and real settings."

#### Strengthening 2: Connect to Standard Absorption Metric (High Value)

**What**: Measure the correlation between our Jaccard-based "multi-child proportional absorption" and Chanin et al.'s ablation-based "absorption rate" on the same features.

**Why**: Our metric is non-standard. Without validation against the established metric, reviewers will question whether we're measuring the same phenomenon.

**How**:
- Use sae-spelling dataset (first-letter classification) on Gemma-2-2B layers 0-17
- Compute both metrics on the same feature set
- Report Pearson/Spearman correlation
- Expected runtime: ~1 hour

**Impact**: Would validate our metric and enable direct comparison with Chanin et al.'s baseline numbers (0.49 absorption rate).

#### Strengthening 3: Test on Multiple Architectures (Moderate Value)

**What**: Replicate H_Mech factorial on TopK, JumpReLU, and ReLU SAEs (not just TopK).

**Why**: Our result is architecture-specific (TopK only). If the encoder-driven mechanism holds across architectures, the claim is much stronger.

**How**:
- Train 3 architectures x 4 conditions x 2 seeds = 24 SAEs on synthetic data
- Compare encoder effect ratios across architectures
- Expected runtime: ~2 hours

**Impact**: Would generalize the claim from "TopK-specific" to "universal SAE property."

### Risk Assessment for Strengthening

| Strengthening | Effort | Impact | Risk |
|---------------|--------|--------|------|
| Real SAE validation | High | Very High | Gemma Scope loading may fail; may need fallback to GPT-2 |
| Standard metric correlation | Medium | High | Requires layers 0-17 (ablation limitation); may show weak correlation |
| Multi-architecture | Medium | Medium | Training 24 SAEs may exceed time budget; may show architecture-dependent effects |

## 7. Honest Assessment of Limitations

### Critical Limitations

1. **Synthetic data only**: All experiments use synthetic hierarchical data (d_model=128). No validation on real LLM representations.
2. **Non-standard metric**: "Multi-child proportional absorption" (Jaccard overlap) is not the established "absorption rate" from Chanin et al. Direct comparison is impossible.
3. **Single architecture**: All results are for TopK SAEs. Generalization to JumpReLU, ReLU, Gated SAEs is untested.
4. **Small scale**: d_sae=4096, d_model=128. Real SAEs are d_sae=16k-1M, d_model=768-8192.
5. **Partially preempted**: Oursland (2026) provides theoretical support for encoder-decoder asymmetry, reducing exclusivity.

### What This Work Does NOT Show

- It does not show that encoder-driven absorption occurs in real LLMs
- It does not propose a solution to absorption (only diagnoses the mechanism)
- It does not validate against the standard absorption rate metric
- It does not test across SAE architectures
- It does not address deep-layer absorption (ablation limitation)

## 8. Bottom Line

**Contribution**: A rigorously demonstrated mechanistic insight (encoder drives absorption, decoder is negligible) with solid experimental design (factorial, multi-seed, ablations), but limited to synthetic data and a non-standard metric.

**Position in field**: The insight is valuable and novel, but the limited scope (synthetic only, single architecture) and partial overlap with concurrent work (Oursland 2026) make it better suited for a workshop or mid-tier venue than a top-tier conference.

**Path to strengthening**: Real-model validation is the single most impactful addition. Without it, the paper risks being dismissed as a synthetic-only result.

---

## Sources

- [A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](https://arxiv.org/abs/2409.14507) (Chanin et al., NeurIPS 2025)
- [SAEBench: A Comprehensive Benchmark for Sparse Autoencoders](https://arxiv.org/abs/2503.09532) (Karvonen et al., ICML 2025)
- [Learning Multi-Level Features with Matryoshka Sparse Autoencoders](https://arxiv.org/abs/2503.17547) (Bussmann et al., 2025)
- [OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features](https://arxiv.org/abs/2509.22033) (Korznikov et al., 2025)
- [From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders](https://arxiv.org/abs/2602.11881) (Luo et al., 2026)
- [Deriving Decoder-Free Sparse Autoencoders from First Principles](https://arxiv.org/abs/2601.06478) (Oursland, 2026)
- [SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data](https://arxiv.org/abs/2602.14687) (Chanin et al., 2026)
- [Negative Results for Sparse Autoencoders On Downstream Tasks](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9) (Smith et al., DeepMind, 2025)
- [Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts](https://arxiv.org/abs/2506.23845) (2025)
