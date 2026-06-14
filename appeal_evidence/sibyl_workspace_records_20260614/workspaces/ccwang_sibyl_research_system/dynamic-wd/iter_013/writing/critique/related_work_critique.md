# Critique: Related Work Section

**Score: 6 / 10**

---

## Summary Assessment

The Related Work section is structurally clean and internally consistent with the paper's narrative. The four-category taxonomy (scheduling, alignment-aware, norm-constrained, ratio dynamics) maps naturally onto EqWD's positioning claims. The comparison table is useful and the "Positioning EqWD" subsection does the job of differentiating the contribution. However, the section suffers from shallow coverage across all four categories, overconfident characterizations of baselines that contradict experimental results, several missing foundational citations, and a theoretical framing that is not sufficiently skeptical about EqWD's own limitations. These issues would likely draw reviewer scrutiny at a top venue.

---

## Issue 1: Literature Coverage is Sparse and Uneven (Major)

**Weight decay scheduling** receives only two citations beyond AdamW (Xie et al. 2023 and Naganuma et al. 2026). This is a well-studied area, and the coverage is insufficient for a paper claiming to unify the field. Missing works that reviewers will expect:

- **Warmup + cosine LR schedules** implicitly define a WD schedule via the coupling $r^* = \lambda/\gamma$; the relationship between learning rate schedules and WD effectiveness (Gotmare et al., 2018; Smith & Topin, 2019, super-convergence) is not discussed, even though the paper's own analysis shows that EqWD responds at LR schedule transitions.
- **L2 vs. decoupled WD**: The distinction between L2 regularization and decoupled WD (Loshchilov & Hutter, 2019) is handled in one sentence, but its implications for Adam-based optimizers are understated given that the paper uses SGDW throughout. Reviewers will ask why AdamW variants are not evaluated.
- **Layer-wise WD**: Parameter-group-specific WD in Transformers (e.g., excluding bias and normalization layers from WD, as in BERT/ViT) is a standard technique that is directly related to per-layer granularity, yet it goes unmentioned.
- **WD and flat minima**: Work connecting weight decay to sharpness-aware minima (SAM, Foret et al. 2021; ASAM, Kwon et al. 2021) is absent. These are prominent recent methods that use gradient-norm information to find flatter optima and are relevant to the claim that EqWD improves stability.

**Alignment-aware** category: Only CWD and one theoretical paper (Sun et al. 2025) are cited. Missing:
- **Cautious optimizers** (Luo et al., 2024), which apply a similar alignment mask to gradient updates rather than weight decay and have attracted significant recent attention. This is a direct conceptual competitor.
- **Lion optimizer** (Chen et al. 2023), which uses sign-based updates and is related to the binary-mask philosophy of CWD; its relationship to alignment-aware regularization should be noted.

**Norm-constrained** category: Three works are cited. Missing:
- **MaxNorm regularization** (Hinton et al., 2012; Srivastava et al., 2014 dropout paper): the conceptual predecessor of per-parameter norm constraints, whose absence makes the category history appear to start in 2024.
- **Weight clipping** and **spectral normalization** (Miyato et al., 2018): norm-based techniques that bear directly on the CPR line of work.
- **AdaDecay** or similar adaptive weight decay schemes from NLP fine-tuning (e.g., Zhuang et al., 2022, or LoRA-WD variants): increasingly relevant given EqWD's claimed applicability to large models.

**Ratio dynamics** category: This is the most directly relevant category and receives only four citations. The coverage is adequate here, but the Chou (2025) citation deserves a sentence connecting its $\gamma^2$ scaling law to EqWD's $r^* = \lambda/\gamma$ formula, since the two are closely related and a reviewer may ask whether they are consistent.

---

## Issue 2: Characterization of SWD Overstates EqWD's Advantage (Major)

The section states SWD "discards weight norm information, making its schedule sensitive to gradient noise and yielding higher seed-to-seed variance." The gradient noise claim is asserted rather than shown; the variance claim is then supported by experimental results (SWD std = 0.40% vs. EqWD 0.20%). However, this characterization is misleading:

- The actual accuracy gap between EqWD and SWD on ImageNet is **+0.23%** (72.27% vs. 72.04%). This is a modest margin on 45-epoch training, and reviewers familiar with ResNet-50 convergence will question whether 45 epochs is long enough for WD methods to show stable differences (standard training is 90 epochs).
- The reduced variance of EqWD over SWD is presented as a core advantage in both the intro and the related work, but the related work section does not explain *why* ratio-based modulation should reduce seed variance. The argument should be made explicitly or deferred to the theory section.
- Describing SWD as "heuristic" in the comparison table while labeling EqWD as grounded in "Defazio 2025 + Sun 2025" is accurate but incomplete: EqWD's Proposition 1 is definitional (it follows trivially from the formula) and Theorem 1 is labeled "informal." The asymmetric labeling ("heuristic" vs. cited theory) slightly inflates EqWD's theoretical standing.

---

## Issue 3: CWD Characterization is Unfair and Potentially Inaccurate (Moderate)

The section claims CWD "underperforms fixed weight decay (71.39% vs. 71.89%)" and attributes this to the binary signal becoming "noisy at scale." This is a strong claim that deserves more nuance:

- The experiments use 45 epochs. CWD's masking mechanism may need more training steps to take effect on ImageNet, so the comparison may be configuration-dependent rather than a principled failure of the approach.
- The binary vs. continuous signal distinction is well-taken, but CWD was designed for a specific setting. Saying it "becomes noisy at scale" without a careful diagnostic (e.g., what fraction of parameters have the mask activated as a function of scale?) is speculative.
- The characterization "decaying weights in the direction of gradient descent is counterproductive" is the paper's paraphrase of CWD's motivation, but CWD's actual argument is more nuanced. The description risks being uncharitable in a way reviewers from the CWD team would object to.

---

## Issue 4: AlphaDecay Is Treated Superficially (Moderate)

AlphaDecay (He et al. 2025) is mentioned in one clause but dismissed as "not per-iteration adaptive." This is an important method because it establishes per-module spectral-density-guided WD for large language models, which is conceptually related to EqWD's per-layer design. The dismissal is technically correct but academically insufficient:

- The discussion should acknowledge that spectral density is a richer per-layer signal than the gradient-to-weight ratio, and explain why EqWD's simpler ratio signal is preferred (computational cost, adaptivity during training, LLM-free applicability).
- The related work does not mention whether EqWD was evaluated on LLMs or only vision models. If AlphaDecay targets LLMs and EqWD targets vision, the competitive positioning is moot and should be framed as complementary rather than competing.

---

## Issue 5: The Comparison Table Is Incomplete (Moderate)

Table 1 (method comparison) has four rows: SWD, CWD, CPR, and EqWD. AlphaDecay and AdamWN (mentioned in the text) are excluded without explanation. The table also omits the "Theory" column entry for CPR as "Lagrangian duality," which is accurate but does not capture the per-matrix constraint nature of CPR. More importantly:

- **FixedWD** and **AdamW** should appear in the table as anchor rows, even with "N/A" entries, to ground the comparison for readers not expert in the subfield.
- The "Per-Layer" column is binary (Yes/No), but the granularity differences are more nuanced: CPR is per-matrix, CWD is per-parameter, SWD is global, EqWD is per-layer-group. Using a single binary column conflates meaningfully different granularities.
- There is no column for "Computational Overhead" despite the intro claiming EqWD adds only ~2% overhead. This would be a useful differentiator.

---

## Issue 6: Missing Broader Context on Weight Decay Theory (Minor)

The section focuses narrowly on methods that modulate weight decay dynamically, but provides no discussion of:

- **Why weight decay works at all**: the implicit regularization interpretation (weight decay as a Gaussian prior / MAP estimation), the implicit bias interpretation (bias toward low-norm solutions), and the recent implicit regularization of SGD literature (Barrett & Dherin, 2021; Smith et al., 2021). EqWD's theoretical claim that ratio deviation identifies "transitional phases" would be better situated against this broader understanding of what weight decay does.
- **Connection to learning rate / weight decay equivalence**: Under certain training assumptions, scaling $\lambda$ by $1/\gamma$ is equivalent to standard L2 regularization (Zhang et al., 2019). This is directly relevant to the ratio $r^* = \lambda/\gamma$ that EqWD tracks.

---

## Issue 7: Citations for 2025-2026 Papers Are Not Verifiable (Minor, but Procedural Risk)

CWD (chen2026cwd), Naganuma et al. (2026), and Sun et al. (2025) are cited as published or in-press works, but the dates suggest they are either preprints or works-in-progress at the time of writing (current date: March 2026). The related work section should clarify the publication status of these citations (arXiv preprint vs. conference proceedings) and ensure the correct citation format is used, since misrepresenting preprints as peer-reviewed works is a common reason for reviewer concern.

---

## Specific Improvement Suggestions

1. **Expand the scheduling subsection** to cover SAM/ASAM, warmup-cosine WD interaction, and layer-exclusion practices in Transformers. Even brief mentions signal to reviewers that the authors are aware of the full landscape.

2. **Add cautious optimizers** (Luo et al. 2024) to the alignment-aware subsection. They are a prominent competitor to the alignment-masking idea and their omission is conspicuous.

3. **Revisit the SWD characterization**: replace the unsubstantiated "gradient noise" claim with a more precise statement about what the ratio signal adds over gradient-norm-only scheduling. Cite the ratio trajectory analysis from Section 4.3 as evidence.

4. **Temper the CWD criticism**: acknowledge that CWD's underperformance in your setting may reflect experimental setup (45 epochs, no hyperparameter tuning parity check) rather than a fundamental limitation.

5. **Expand AlphaDecay discussion** to two sentences that articulate the domain boundary (LLM vs. vision) and explain why per-layer ratio is preferred over spectral density for your setting.

6. **Extend the comparison table** with FixedWD as an anchor row, a Computational Overhead column, and more granular per-layer entries.

7. **Add a paragraph on weight decay theory foundations** (implicit regularization, MAP interpretation, LR-WD coupling) before the method-specific subsections to ground the contribution in the broader theoretical literature.

8. **Clarify publication status** of 2025-2026 citations throughout.

9. **Address the 45-epoch training caveat**: either justify why 45 epochs is sufficient to distinguish WD methods or add a supplementary experiment at 90 epochs for at least EqWD vs. FixedWD.

---

## Verdict

The section is serviceable but would not pass review at a top-tier venue (NeurIPS/ICML/ICLR) without significant additions. The core positioning argument is clear and the taxonomy is logical, but the coverage gaps, overconfident baseline characterizations, and thin theoretical grounding (relative to what the intro promises) collectively weaken the paper's credibility. With the additions listed above, the section could reach a score of 8/10.
