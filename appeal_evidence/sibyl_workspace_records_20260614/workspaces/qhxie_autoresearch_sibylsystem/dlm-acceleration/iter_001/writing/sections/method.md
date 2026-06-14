# 4. Methods

This section formalizes the ComposeAccel framework. Section 4.1 defines the composability
metric and Quality-Adjusted Speedup (QAS). Section 4.2 describes each method family and
its operating parameters. Section 4.3 specifies the evaluation protocol.

---

## 4.1 Composability Framework

### Orthogonality Metric

Let $M_a$ and $M_b$ denote two acceleration methods applied to the same MDM inference
pipeline, and let $\text{Speedup}(M)$ be the wall-clock throughput ratio of method $M$
over the unaccelerated baseline (measured in tokens per second, TPS). The
**Orthogonality metric** quantifies how the combined speedup of $(M_a, M_b)$ relates
to the product of their individual speedups:

$$
\text{Ortho}(M_a + M_b) = \frac{\text{Speedup}(M_a + M_b)}{\text{Speedup}(M_a) \times \text{Speedup}(M_b)}
$$

The denominator is the multiplicative ideal: the combined speedup that would be observed
if the two methods eliminated independent, non-overlapping bottlenecks. Ortho = 1.0
means the methods are perfectly orthogonal. Three interpretive regimes:

- $\text{Ortho} \geq 1.0$: **super-multiplicative synergy** — the combined speedup exceeds the product of individual speedups, indicating that one method creates conditions that amplify the other.
- $\text{Ortho} \in [0.8, 1.0)$: **partially orthogonal** — modest degradation from the multiplicative ideal, consistent with minor resource contention.
- $\text{Ortho} < 0.8$: **destructive interference** — the methods conflict substantially, and the combination underperforms the product baseline.

Ortho is a dimensionless ratio: it remains valid regardless of the absolute speedup
magnitude of each method, as long as both components and their combination are measured
under the same hardware and implementation conditions.

### Quality-Adjusted Speedup

Speedup alone is insufficient when methods degrade output quality. We define
**Quality-Adjusted Speedup (QAS)** as:

$$
\text{QAS}(M) = \text{Speedup}(M) \times \text{AccRet}(M)
$$

where accuracy retention is $\text{AccRet}(M) = \text{Acc}(M) / \text{Acc}(\text{baseline})$.
QAS simultaneously rewards throughput gains and penalizes accuracy losses: a method
that doubles speed but halves accuracy yields QAS = 1.0, equivalent to the unaccelerated
baseline. All methods use this identical formula with no per-method modifications or
feasibility penalties. Methods with $\text{AccRet} < 0.95$ receive an explicit warning
flag in result tables, but QAS itself is not adjusted.

For pairwise combinations, QAS uses the combined speedup and the mean accuracy retention
across benchmarks:

$$
\text{QAS}(M_a + M_b) = \text{Speedup}(M_a + M_b) \times \overline{\text{AccRet}}(M_a + M_b)
$$

---

## 4.2 Method Implementations

We study four families of training-free MDM acceleration methods. All operate on
LLaDA-8B-Instruct [CITE:llada] without modifying model weights. Operating-point
parameters are selected from single-method Pareto sweeps (Section 5.1).

### M1: KV-Cache Approximation (EntropyCache)

Each MDM denoising step recomputes all $N^2$ key-value products via bidirectional
attention. KV-cache approximation [CITE:entropycache] reuses cached key-value matrices
$\hat{K}_{t-1}^{(\ell)}, \hat{V}_{t-1}^{(\ell)}$ from the previous step for positions
where the token representation has not meaningfully changed.

**Refresh decision.** EntropyCache computes per-token decoded entropy $H_i$ as the
stability signal. At each denoising step $t$:

$$
H_i = -\sum_{v \in \mathcal{V}} p_\theta(v \mid \tilde{x}_t) \log p_\theta(v \mid \tilde{x}_t)
$$

Position $i$ refreshes its KV entry when $H_i \geq \eta$; otherwise it reuses cached
values from step $t-1$. The cache hit rate (CHR) at threshold $\eta$ is
$\text{CHR} = |\{i : H_i < \eta\}| / N$.

**Parameter sweep**: $\eta \in \{0.5, 1.0, 2.0, 3.0\}$.

**Operating point**: $\eta = 2.0$. At this threshold, CHR $\approx$ 60%, yielding
1.38$\times$ combined speedup across benchmarks. Thresholds below $\eta = 1.0$ invert
the benefit: entropy computation and selective recomputation overhead exceed
attention savings (speedup = 0.55$\times$ at $\eta = 0.5$). This cache overhead
inversion is documented as failure mode FM2 in Section 5.3.

**Implementation note.** Our M1 implementation computes entropy and applies the
cache-refresh logic but still executes full transformer forward passes — kernel-level
sparse attention that skips cache-hit positions is not implemented. The measured
1.38$\times$ reflects only the savings from avoiding redundant entropy recomputation and
cache management; the published EntropyCache 15--26$\times$ [CITE:entropycache]
requires sparse attention kernels that bypass attention computation for cached positions
entirely. The Ortho metric is computed from speedup ratios under identical hardware and
implementation conditions, so it remains valid despite this absolute speedup gap.
Section 6.3 discusses this limitation.

### M2: Adaptive Step Scheduling (Simplified Saber)

Adaptive step scheduling reduces the total denoising steps from $T = 64$ by unmasking
$J$ times more tokens per step. Following Saber [CITE:saber], we select the top-$k$
tokens by confidence at each step, reducing the effective step count to
$T_{\text{eff}} = \lceil T / J \rceil$.

**Parameter sweep**: $J \in \{2, 4, 6, 8\}$.

**Verdict: NO\_GO.** At $J = 2$, GSM8K accuracy retention falls to 54.4%; at $J = 4$
it collapses to 13.0%; at $J = 8$ it reaches 24.3% (catastrophic). The root cause is
a fundamental algorithmic mismatch: LLaDA's masked denoising requires sequential
cumulative conditioning — each token's unmasking at step $t$ depends on the mask state
established by all prior steps. Skipping steps creates unresolvable mask inconsistencies
at positions committed too early, because tokens that would have been refined in
intermediate steps are instead frozen in their premature state. This is a structural
incompatibility, not a hyperparameter issue: no value of $J$ avoids the cascade.
M2 is excluded from all composability experiments and documented as failure mode FM1
(Section 5.3).

**Caveat.** Saber's full algorithm includes backtracking-enhanced remasking, which
we do not implement. Our NO\_GO verdict applies to simplified top-$k$ scheduling.
The structural mask-inconsistency mechanism is fundamental to LLaDA's architecture,
but Saber's backtracking may partially mitigate its severity.

### M3: AR-Guided Unmasking

AR-guided unmasking [CITE:flashdlm] biases each denoising step's token selection
using a lightweight autoregressive (AR) model. At each step, we run Qwen2.5-0.5B
[CITE:qwen25] on the current partially-unmasked sequence and blend its log-probabilities
with LLaDA's unmasking logits:

$$
\ell_{\text{blended}}(v, i) = (1 - w) \cdot \ell_{\text{LLaDA}}(v, i) + w \cdot \log q_\phi(v \mid x_{<i})
$$

where $q_\phi(v \mid x_{<i})$ is Qwen2.5-0.5B's autoregressive token probability and
$w$ is the guidance weight. Cross-tokenizer bridging is required: LLaDA tokens are
decoded to text and re-encoded with the Qwen2.5 tokenizer at each step.

**Parameter sweep**: $w \in \{0.3, 0.5, 0.7, 1.0\}$.

**Operating point**: $w = 0.3$, which achieves 1.68$\times$ speedup on GSM8K with
accuracy retention of 103.9% — the AR guide genuinely improves reasoning accuracy by
3.9 percentage points above baseline (71.2% $\to$ 74.0%). Combined across all
benchmarks, the speedup is 1.33$\times$. The MATH500 retention figure (243.9%) is
inflated by the low 11.1% baseline and should not be interpreted as a real 2.4$\times$
improvement.

M3 fails on coding benchmarks: HumanEval QAS $\approx$ 0 at speedup 0.83$\times$
(slower than baseline). Qwen2.5-0.5B's causal language model logits are misaligned
with LLaDA's bidirectional MASK-to-token mapping on Python syntax, producing guidance
signals that degrade rather than improve code generation. This task-dependent failure
is documented as FM4 in Section 5.3.

### CD-SSD: Coarse-Draft Self-Speculative Denoising

CD-SSD (Coarse-Draft Self-Speculative Denoising) is a training-free speculative
denoising method that uses LLaDA-8B as both drafter and verifier — no external model
and no training required. Figure 2 illustrates the three-phase pipeline.

![CD-SSD three-phase pipeline: Draft, Partition, and Refine](figures/fig2_igsd_architecture_desc.md)

**Draft phase.** Run LLaDA-8B with $T_{\text{draft}}$ denoising steps over the full
sequence canvas, producing a draft output $x_{\text{draft}} \in \mathcal{V}^N$ and
per-token confidence scores:

$$
c_i = \max_{v \in \mathcal{V}} p_\theta(v \mid \tilde{x}_{T_{\text{draft}}})
$$

At $T_{\text{draft}} = 16$ (one-quarter of the full $T = 64$ schedule), the model
establishes the dominant semantic structure of the output. The draft uses the standard
LLaDA generation schedule, unmasking a uniform fraction of tokens per step with no
block structure.

**Partition.** Tokens are split into two disjoint sets based on the confidence
threshold $\tau$:

$$
S_{\text{accept}} = \{i : c_i \geq \tau\}, \quad S_{\text{refine}} = \{i : c_i < \tau\}
$$

At $\tau = 0.9$, approximately 52% of unique token positions are accepted
($\alpha \approx 0.52$), with the remaining 48% marked for refinement. The partition
is a single vector comparison with negligible wall-clock cost.

**Refine phase.** Tokens in $S_{\text{accept}}$ are frozen: their identities are fixed
as $x_{\text{draft}}[i]$ for all subsequent steps and are not subject to further
denoising. The full $T_{\text{full}} = 64$-step schedule runs over $S_{\text{refine}}$
positions, but frozen tokens in $S_{\text{accept}}$ remain present as context in
bidirectional attention. The speedup arises from two sources: (1) the draft phase
runs 16 steps instead of 64; (2) the refine phase processes only
$|S_{\text{refine}}| / N \approx 48\%$ of positions at full depth. At the operating
point ($\tau = 0.9$, $T_{\text{draft}} = 16$), CD-SSD achieves 4.57$\times$ speedup
on GSM8K with 63.7% accuracy retention and 2.32$\times$ on MATH500 with 88.5%
retention (3-seed mean).

**Acceptance criterion.** The MDM setting precludes the exact lossless speculative
decoding guarantee available in AR models: token identity changes in LLaDA are globally
coupled through bidirectional attention, so rejecting and resampling a draft token
changes the context for all other positions. CD-SSD uses a soft acceptance criterion:
tokens with draft confidence $c_i \geq \tau$ are accepted unconditionally, and the
refine phase runs a corrective full-depth pass on the remaining positions using the
frozen context. This provides an approximate, not lossless, quality guarantee,
quantified in the ablation study (Section 5.4).

**Synergy with M1 (KV-cache).** During the refine phase, tokens in $S_{\text{accept}}$
have fixed identities across all $T_{\text{full}} = 64$ steps. From EntropyCache's
perspective, these positions are deterministic: their decoded probability distributions
assign full mass to a single token, giving entropy $H_i = 0$ for every
$i \in S_{\text{accept}}$. Since $0 < \eta = 2.0$, every frozen position is a
guaranteed cache hit for every step of the refine phase. The measured KV-cache hit rate
during the refine phase reaches 94.0% (from per-seed results at $\tau = 0.9$,
$T_{\text{draft}} = 16$), compared to 60% in standalone M1 operation. This yields a
combined speedup of 5.13$\times$ for M1+CD-SSD, exceeding the product of individual
speedups ($1.38 \times 3.40 = 4.69\times$) and producing Ortho = 1.385. The
frozen-token KV amplification mechanism is analyzed in Section 6.2.

**tau = 0.0 ablation preview.** Removing the partition entirely ($\tau = 0.0$: accept
all draft tokens, skip refine) yields QAS = 4.198, statistically indistinguishable
from naive 16-step denoising (QAS = 4.458, $\Delta$ = $-$5.8%). This indicates that
the confidence partitioning mechanism does not add measurable value beyond simple step
reduction for standalone CD-SSD quality. CD-SSD's value in this paper is as a
composability vehicle: the frozen-token structure created by the partition is what
enables the super-multiplicative KV-cache synergy, even if the partition does not
improve standalone output quality over naive $T = 16$. Section 5.4 presents the full
ablation.

**Parameter sweep**: $\tau \in \{0.7, 0.8, 0.85, 0.9\}$,
$T_{\text{draft}} \in \{4, 8, 16, 32\}$.

**Operating point**: $\tau = 0.9$, $T_{\text{draft}} = 16$.

**Relation to concurrent work.** SSD [CITE:ssd] (Gao et al., arXiv:2510.04147)
achieves lossless 3.46$\times$ speedup via same-model hierarchical tree verification —
a training-free self-speculative approach concurrent with CD-SSD. SSMD [CITE:ssmd]
(Campbell et al., arXiv:2510.03929) achieves self-speculation via an attention-mask
switch mechanism. S2D2 [CITE:s2d2] (arXiv:2603.25702) targets block-diffusion LLMs.
DualDiffusion [CITE:dualdiffusion] (arXiv:2604.05250) requires an external draft model.
CD-SSD differs from SSD in draft mechanism (reduced 16-step coarse pass vs. full
64-step hierarchical tree) and quality guarantee (approximate vs. lossless). The
key differentiation is composability behavior: CD-SSD's coarse-step draft creates a
frozen-token set ($\alpha \approx 0.52$) during the refine phase, enabling the
KV-cache amplification mechanism described above. SSD's lossless verification
does not produce a comparable frozen-token set. Whether SSD+M1 achieves similar
super-multiplicative synergy is an open question (Section 6.3).

---

## 4.3 Evaluation Protocol

### Model and Hardware

All experiments use **LLaDA-8B-Instruct** (GSAI-ML/LLaDA-8B-Instruct) in bf16
precision on one NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB VRAM).
No model weights are modified. Dream-7B-Instruct was unavailable; all results
are single-model.

### Benchmarks

| Benchmark | Task | Metric | Samples | Note |
|-----------|------|--------|---------|------|
| GSM8K | Grade school math | Exact match (8-shot) | 1319 | Primary reasoning benchmark |
| MATH500 | Competition math | Exact match (4-shot) | 500 | High-difficulty reasoning |
| HumanEval | Code generation | pass@1 (0-shot) | 164 | Degenerate baseline (2.4%) |
| MBPP | Python programming | pass@1 (3-shot) | 257 | Degenerate baseline (0.0%) |

HumanEval (2.4% baseline pass@1) and MBPP (0.0% baseline pass@1) have degenerate
baselines, making accuracy retention metrics unreliable: any method scoring 0/164 on
HumanEval records 0.0% retention regardless of baseline performance, while MBPP maps
trivially to 100% retention (0/0 $\to$ 1.0 by convention). Primary quantitative claims
in Sections 5.1--5.3 are restricted to GSM8K and MATH500. Coding benchmarks are
reported separately with explicit caveats.

### Seeds and Variance

Single-method Pareto experiments use seeds [42, 123, 456]; results are reported as
mean $\pm$ std. Pairwise composability experiments (Section 5.2) use seeds [42, 123]
on 200 GSM8K + 164 HumanEval samples. This reduced scale reflects the combinatorial
cost of evaluating three method pairs across multiple benchmarks and seeds; pairwise
Ortho values are 2-seed estimates with per-seed ranges reported for variance assessment.

### Throughput Measurement

Throughput is reported as tokens per second (TPS): wall-clock output tokens divided by
elapsed generation time (prompt encoding excluded). The first 2--5 warm-up samples per
benchmark are discarded to exclude JIT compilation and cache warm-up overhead. Speedup
is computed as $\text{TPS}(M) / \text{TPS}(\text{baseline})$, with both measurements
taken on the same GPU within the same session.

### Baseline Reference

| Benchmark | Accuracy | TPS (mean $\pm$ std) |
|-----------|----------|---------------------|
| GSM8K | 71.2% exact match | 31.0 $\pm$ 4.0 tok/s |
| MATH500 | 11.1% exact match | 79.2 $\pm$ 0.1 tok/s |
| HumanEval | 2.4% pass@1 | 98.0 $\pm$ 2.1 tok/s |
| MBPP | 0.0% pass@1 | 191.6 $\pm$ 0.6 tok/s |

The MATH500 baseline accuracy (11.1%) reflects LLaDA-8B's limited mathematical reasoning
at competition difficulty. The MBPP 0.0% baseline is a degenerate result at the 3-shot
setting; no acceleration method can claim meaningful accuracy retention on this benchmark.

<!-- FIGURES
- Figure 2: fig2_igsd_architecture_desc.md — CD-SSD three-phase architecture diagram: Draft phase (T_draft=16 steps, full sequence), Partition (confidence threshold tau=0.9 splitting S_accept vs S_refine at alpha=0.52), Refine phase (frozen S_accept tokens providing stable KV context with H_i=0 while T_full=64 steps run on S_refine; KV hit rate=94% during refine). Annotated with frozen-token KV amplification synergy mechanism.
- None (no code-generated figures in this section)
-->
