# Critique: Method

## Summary Assessment

The Method section is technically precise, well-motivated, and largely complete. Each subsection delivers its core definition, motivating argument, and connection to the broader paper. The main weaknesses are (1) a systematic notation inconsistency between the method text and the established notation.md (specifically, `W_e` vs. `W_enc`), (2) a missing Figure 3 architecture diagram promised by both the outline and section 4.3's logical flow, (3) a questionable causal story in section 3.1 that elevates a "tends to" pattern to a mechanism without citing direct gradient-level evidence, and (4) a Jaccard filter condition in section 3.2 that silently differs from notation.md's definition. These are fixable; the section's core argument is sound.

## Score: 7/10

**Justification**: The mechanistic motivation is specific and testable, the computation claim (under 0.1 seconds) is concrete, and the pre-committed falsification criterion in section 3.3 is unusually rigorous for a ML paper. The score is held back by notation drift that will confuse readers who consult notation.md, the absence of Figure 3 (a promised visual that would make the 3.3 setup considerably clearer), and the aggressive causal language in 3.1 ("the encoder responds by inflating…") which overstates what the gradient competition argument actually proves. A 8/10 requires: (a) notation harmonized with notation.md, (b) Figure 3 either placed or explicitly flagged as "see Figure 3", and (c) the causal language in 3.1 softened to match the empirical framing that follows.

---

## Critical Issues

### Issue 1: Systematic notation inconsistency — `W_e` vs. `W_enc`

- **Location**: Section 3.1, Definition and Mechanistic Motivation paragraphs; also Section 3.3 Condition A
- **Quote**: `"For SAE latent $j$ with encoder weight matrix row $\mathbf{w}_{e,j} \in \mathbb{R}^{d_\text{model}}$"` (3.1); `"$\mathbf{z} = \text{ReLU}(W_e \mathbf{x} + \mathbf{b}_e)$"` (3.3)
- **Problem**: notation.md defines the encoder weight matrix as `W_enc` (with rows `W_enc_j`) and the encoder bias as `b_enc`. The method text uses `W_e`, `\mathbf{w}_{e,j}`, and `\mathbf{b}_e` throughout — all inconsistent with notation.md. A reader who looks up the glossary after reading 3.1 will encounter a symbol mismatch. Experiments section 4.1 already uses the `EncNorm` shorthand consistently, but the method section that defines it uses different base notation.
- **Fix**: Replace `W_e` → `W_\text{enc}`, `\mathbf{w}_{e,j}` → `\mathbf{w}_{\text{enc},j}`, `\mathbf{b}_e` → `\mathbf{b}_\text{enc}` throughout. Cross-check section 3.3 Condition A encoding formula for the same drift.

### Issue 2: Figure 3 is referenced in the outline but absent from the text

- **Location**: Section 3.3 (Amortization Gap Controlled Experiment); outline Figure Plan (Figure 3: Amortization Gap Experiment Architecture)
- **Quote**: The outline specifies: "Figure 3: Architecture diagram — fixed decoder vs. varied encoder; absorption measurement pipeline." The method section contains no figure reference at all.
- **Problem**: The OMP experiment design (fixed decoder, varied encoder) is non-obvious. A reader following the prose must mentally construct the two-branch pipeline. The outline planned Figure 3 precisely to address this. Its absence means the reader must hold the control design in working memory across a dense paragraph; the text would be considerably clearer if it contained "see Figure 3" and the figure appeared here.
- **Fix**: Either (a) add the figure inline with a `\ref{fig:omp-design}` pointer in the text, or (b) insert a placeholder `[Figure 3: Architecture diagram — feedforward vs. OMP, fixed decoder]` and explicitly note in the submission that this figure is to be added before camera-ready. The text at the start of section 3.3 (before the Setup subsection) is the natural anchoring point: "Figure 3 illustrates the two conditions."

---

## Major Issues

### Issue 3: Causal language in 3.1 asserts mechanism not yet established at this point in the paper

- **Location**: Section 3.1, Mechanistic Motivation paragraph
- **Quote**: `"The encoder responds by inflating $\|\mathbf{w}_{e,j}\|_2$: a larger encoder weight increases the dot product $\mathbf{w}_{e,j}^\top \mathbf{x}$ for on-distribution inputs, partially counteracting the suppressive gradient from $c$. Absorbed features therefore tend to develop elevated encoder norms as a training-time signature of this gradient competition."`
- **Problem**: The language "the encoder responds by inflating" presents a training-time mechanism as established fact. The paper's actual evidence for this is observational: EncNorm is elevated for absorbed features. The mechanism (gradient competition → encoder weight inflation) is a theoretical prediction from Tang et al.'s partial minimum account, not something directly measured in this paper. The word "therefore" implies logical derivation, but this is post-hoc interpretation of empirical correlation. A reviewer at NeurIPS will flag this as overclaiming.
- **Fix**: Reframe as prediction from theory, not established mechanism: "Under the sparsity landscape account \citep{tang2025partial}, the training-time gradient competition from child $c$ prevents $j$'s encoder direction from converging normally. One predicted consequence is that $\|\mathbf{w}_{e,j}\|_2$ becomes inflated as the encoder attempts to counteract the suppressive gradient. This predicts EncNorm$(j)$ > EncNorm for non-absorbed features, which we test empirically." This preserves the scientific logic while accurately attributing the mechanism to Tang et al. and flagging it as a prediction being tested rather than a proven fact.

### Issue 4: Jaccard filter condition in section 3.2 silently differs from notation.md

- **Location**: Section 3.2, Definition
- **Quote**: `"the max is over all latents with higher activation frequency than $j$"` in the text vs. `"max_{i: f_i > 3f_j} O_jaccard(j, i)"` in notation.md
- **Problem**: notation.md's `O_jaccard(j)` is defined as the maximum over latents with activation frequency at least 3 times that of j (`f_i > 3f_j`). The method section's formula says "all latents with higher activation frequency than j" (i.e., `f_k > f_j` with threshold 1×, not 3×). These are different operationalizations. Whichever is correct for the actual experiment, the discrepancy will confuse readers trying to reproduce results.
- **Fix**: Harmonize with notation.md. If the actual experiment used `f_k > f_j` (the method text version), update notation.md to match and add a note explaining why the 1× threshold was chosen over 3×. If the experiment used `f_i > 3f_j` (the notation.md version), fix the method text formula to read `\{k : f_k > 3 f_j\}`.

### Issue 5: Absorption measurement in section 3.3 uses an abbreviated description that may not match experiments

- **Location**: Section 3.3, Absorption Measurement paragraph
- **Quote**: `"A feature $j$ is counted as absorbed for letter $\ell$ if (a) its decoder direction aligns with the $\ell$-probe direction (cosine similarity $\geq 0.30$) and (b) $z_j = 0$ on $\ell$-positive tokens."`
- **Problem**: Experiments section 3.4 describes IG-based gold labels using the Chanin FeatureAbsorptionCalculator for the Standard SAE. But the absorption measurement in 3.3 describes a proxy method (decoder alignment + zero activation) rather than the full IG pipeline. This creates ambiguity: does the OMP experiment use IG labels or the proxy? The proxy method is described for TopK-32k labels in 3.4 — so the same abbreviated description may apply to different label types in different contexts. This needs explicit disambiguation.
- **Fix**: Replace the abbreviated description with: "We use the Chanin et al.\ Integrated Gradients pipeline (same labels as Section 3.4 Standard SAE gold labels: $n_\text{pos} = 18$, $n_\text{neg} = 24{,}558$). A feature $j$ is counted as absorbed for letter $\ell$ if the IG pipeline confirms concept presence but $z_j = 0$." This makes the OMP experiment's label source explicit and links it to the evaluation setup described in 3.4.

---

## Minor Issues

- **Section 3.1, Computation paragraph**: The operation count `$O(d_\text{SAE} \cdot d_\text{model})$` and `$O(65536 \times 768) \approx 50$M` is correct but mislabeled. This is for a 65k SAE, not the 24k SAE used in most experiments. A reader might be confused. Fix: either use the 24k example (`$O(24576 \times 768) \approx 18.9$M`) or explicitly state "for a 65k-width SAE such as Gemma Scope."

- **Section 3.1, last sentence of Definition**: `"Unlike EDA (which measures the angular divergence between encoder and decoder directions), EncNorm requires only the encoder weight matrix — it does not require the decoder to be unit-normed…"` — the phrase "encoder-decoder pairing to be geometrically interpretable" is vague. Fix: replace with "it does not assume any angular relationship between the encoder row $\mathbf{w}_{\text{enc},j}$ and decoder column $\mathbf{d}_j$."

- **Section 3.2, Complementarity paragraph**: `"Spearman $\rho(\text{EncNorm}, O_\text{Jaccard}) = 0.044$ (near-zero)"` — this number appears again in experiments (4.2), but here it is stated without context (10k tokens, Standard SAE). The number is the same but should be anchored: "computed on 10,000 OpenWebText tokens using the Standard/L1 SAE, Spearman $\rho = 0.044$." This prevents the reader from wondering if this is a claimed general result or a specific measurement.

- **Section 3.3, pre-committed criterion**: The criterion states `$\geq 80\%$` as the falsification threshold. The word "pre-committed" is used without citing where this was pre-committed (the proposal). Add a brief parenthetical: "(pre-committed in the experimental proposal before running experiments; see supplementary)."

- **Section 3.4, hook confound note**: Correctly identifies the confound. However, this note appears in "Experimental Setup" after methods 3.1–3.3 have already used the two SAEs. The note would be more effective placed in section 3.4 under a separate subheading "Confounds and Controls" rather than buried in the Note formatting.

- **Glossary violation — "OMP oracle"**: notation.md/glossary.md states "Orthogonal Matching Pursuit (OMP) on first use." Section 3.3 uses "OMP oracle" before defining OMP. Fix: write out "Orthogonal Matching Pursuit (OMP)" at first use in section 3.3, then "OMP oracle" thereafter.

---

## Visual Element Assessment

- [ ] **Figures/tables match outline plan**: PARTIAL FAIL. The outline's Figure & Table Plan specifies Figure 2 (encoder norm histogram, method section) and Figure 3 (architecture diagram, method section). Neither is referenced in the current method text. Figure 2's content (histogram, Cohen's d=0.971, mean absorbed=3.26 vs. overall=2.58) is described in the outline but the actual method text does not contain a `\ref{fig:enc-norm-hist}` pointer. Figure 3 is entirely absent.
- [ ] **All visuals referenced before appearance**: FAIL. No figures are referenced at all in the current method section. Even if figures exist elsewhere, the method text must contain "see Figure X" references.
- [ ] **Captions are self-explanatory**: N/A — no captions present in section text.
- [ ] **No text-heavy sections that need visual support**: Section 3.3 is the densest. It describes a two-condition experiment (feedforward vs. OMP) with a fixed decoder and measurement pipeline. Without Figure 3, the reader must hold the design in memory. This section urgently needs the architecture diagram.

**Action required**: Add at minimum two figure references in the method section:
- Section 3.1: "Figure 2 shows the encoder norm distribution for absorbed vs.\ non-absorbed latents."
- Section 3.3 (before Setup): "Figure 3 illustrates the experimental design: both conditions share a fixed pre-trained decoder; only the encoding procedure varies."

---

## What Works Well

1. **Pre-committed falsification criterion (section 3.3)**: The explicit statement "Pre-committed criterion: if OMP achieves $\geq 80\%$ of feedforward absorption rate, H2 is falsified" is genuinely rigorous. Most ML papers state criteria retrospectively. This earns credibility.

2. **Complementarity framing (section 3.2)**: Leading with the Spearman $\rho = 0.044$ result to motivate the two-signal audit is exactly the right order — evidence first, implication second. The paragraph is a model of the paper's stated writing quality rules.

3. **Computational cost claim (section 3.1)**: `"under 0.1 seconds on CPU"` with the explicit flop count (`$O(65536 \times 768) \approx 50$M`) is specific and verifiable. This is the kind of concrete, practical detail that makes a methods contribution usable.
