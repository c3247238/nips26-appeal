# Related Work Critique

## Score: 5/10

## Strengths

- The section is cleanly organized into three topically coherent subsections: (1) weight decay theory and practice, (2) dynamic/adaptive regularization, and (3) gradient-parameter alignment. This tripartite structure maps naturally onto the paper's three main contributions (budget equivalence, LR-WD coupling, alignment signal characterization).
- The treatment of L2 vs. decoupled weight decay (AdamW / Loshchilov & Hutter 2019) is accurate and appropriately brief. Tying the regularization formulation to the paper's own notation early in the section sets up the LR-WD coupling analysis cleanly.
- The placement of Xie et al. (2024) as the direct theoretical predecessor is correct and well-motivated. The paper's extension to time-varying WD and empirical characterization of delta_t is clearly differentiated from what Xie et al. prove.
- The van der Berg et al. (2017) scale-free perspective is a genuinely insightful connection: it provides an independent theoretical lens that is consistent with the budget equivalence result, strengthening the paper's narrative without overclaiming.
- The CWD (Liu et al., 2025) discussion is appropriately scoped—it acknowledges the method's motivation, then points directly to the cross-architecture instability finding. This is one of the few places the section does real positioning work.
- The self-critical tone in the alignment subsection ("The negative answer we obtain...does not contradict the theoretical importance of the alignment condition") is scientifically honest and useful for setting reviewer expectations.

---

## Issues (by priority)

### Critical

1. **Missing literature on LR scheduling and WD-LR interaction (the paper's second and arguably most practically significant finding).** The LR-WD coupling result is flagged by the paper's own abstract and intro as a "structural necessity." Yet the Related Work section has no subsection or even a paragraph dedicated to learning rate scheduling. Foundational references—Loshchilov & Hutter SGDR (2017), Smith's cyclical LR (2017), cosine annealing, multi-step milestone schedules—are absent. More importantly, the interaction between LR schedules and regularizer behavior under milestone drops is the exact mechanism exploited in Proposition 1 (decoupling instability). Prior work that has observed or analyzed this interaction (e.g., You et al., LARS/LAMB; Gotmare et al. on warm restarts; He et al. on the original milestone schedule for ResNets) needs coverage. Without this grounding, the LR-WD coupling section in the paper appears to emerge from nowhere.

2. **No coverage of weight norm / effective learning rate analyses relevant to budget equivalence.** Budget equivalence—the paper's headline result—claims that the cumulative sum of lambda_t is the sufficient statistic. This connects directly to a substantial body of work on: (a) parameter norm evolution under L2 regularization (e.g., Krogh & Hertz 1992; van Laarhoven 2017 on norm bounds); (b) Lyu & Li (2020) on gradient descent's implicit bias toward max-margin solutions with weight decay; (c) work on the sharpness-flatness tradeoff and its relation to weight norm (e.g., Foret et al. 2021, SAM; Andriushchenko & Flammarion 2022). If the cumulative weight decay budget controls the final weight norm and hence generalization, these are directly relevant predecessors. Their absence leaves the budget equivalence claim theoretically orphaned in the related work.

3. **Implicit bias and generalization literature is almost completely absent.** The paper's central claim concerns generalization performance (test accuracy). There is a rich theoretical literature on how weight decay affects the implicit bias of SGD (Gunasekar et al., 2017; Soudry et al., 2018; Lyu & Li 2020; Damian et al., 2021) and how it interacts with the loss landscape to produce flat minima (Keskar et al., 2017; Wu et al., 2020). The Related Work section touches on implicit regularization only briefly via Blanc et al. (2020) and Li et al. (2021), but these are cited for gradient covariance structure—not for the WD-generalization connection that the paper's contributions directly address. A reviewer will immediately flag this gap.

4. **The Xie et al. (2024) reference is the single theoretical anchor for the alignment framework, but no verification of its key claims is offered in the related work.** Is this a published paper, an arXiv preprint, or workshop paper? What are its limitations? Does the "necessary and sufficient" characterization hold under all the assumptions made in this paper? The Related Work should briefly note the setting of Xie et al. (fixed LR vs. decaying LR, convex vs. nonconvex) to clarify exactly what the current paper extends and what it does not extend.

### Major

5. **No engagement with WD tuning / sensitivity literature.** A key implicit claim of the paper is that tuning a constant lambda is sufficient. There is substantial empirical literature on WD sensitivity, including: Nakkiran et al. (2021) on double descent and regularization; Zhang et al. (2019) on algorithmic regularization; Recht et al. (2019) on distribution shift and WD. The paper cites zhang2019algorithmic in the intro but does not discuss it in Related Work. If the paper is arguing that constant WD is optimal, it should engage with studies that have found WD sensitivity to be high (and explain why those findings are consistent with budget equivalence).

6. **The "Dynamic and Adaptive Regularization" subsection conflates methods that operate at very different levels of abstraction.** LARS/LAMB (layer-wise LR scaling) are optimizer variants; CWD is a coordinate-wise WD variant; meta-learning approaches (Baydin et al., Franceschi et al.) are bilevel optimization methods; curriculum dropout is a dropout schedule. Grouping these under "dynamic regularization" obscures the relevant distinctions. More specifically, the subsection should carve out a clearer comparison for methods that adapt regularization based on training-time signals (alignment, gradient statistics), since that is the class to which AADWD belongs. The current structure buries this positioning.

7. **Missing recent work on warmup, cooldown, and WD annealing in large-scale training.** The paper focuses on ResNet/CIFAR, but the broader context motivating the question is large-scale training where hyperparameter scheduling is standard practice. Recent work on Chinchilla scaling laws (Hoffmann et al., 2022), muP (Yang et al., 2022), and WD scheduling in LLM pretraining (e.g., weight decay warmup in GPT-style training) is highly relevant to the practical motivation. The paper could legitimately limit its scope to SGD/image classification, but this should be explicitly stated as a scope limitation in Related Work, not ignored.

8. **Gradient-parameter alignment subsection cites Fort et al. (2019) and Gur-Ari et al. (2018) for gradient alignment and Hessian alignment, but these are only loosely connected to the paper's use of delta_t.** The references are technically accurate but the connection is not made explicit. The section should more directly note that the gradient-parameter cosine similarity (as opposed to gradient-gradient or Hessian-gradient alignment) has not been studied as an adaptive WD signal in prior work—this is the genuine novelty being positioned.

9. **CWD citation (Liu et al., 2025) at ICLR 2025 is noted as "recently proposed," but no discussion of why CWD's theoretical motivation predicts the opposite of the observed instability.** CWD is designed to *avoid* interference between WD and gradient directions, which is exactly the kind of alignment-aware design this paper tests. A deeper engagement with CWD's failure mode—and why this failure mode is consistent with the paper's budget equivalence and LR-WD coupling findings—would strengthen the positioning considerably and make the CWD finding more than an incidental observation.

### Minor

10. **The Related Work section lacks a brief transitional conclusion** that synthesizes the three subsections and states what gap the paper fills. NeurIPS reviewers expect the final sentences of Related Work to explicitly position the paper: "Unlike all prior work, we show X." Currently the section ends abruptly with the alignment subsection's self-reflective paragraph, which is a good paragraph but does not function as a section summary.

11. **Krogh & Hertz (1991) is cited only for "long history as an explicit regularizer."** This is accurate but thin. The classical Krogh & Hertz analysis provides a rigorous solution for L2 regularization in linear networks that is directly relevant to the weight norm evolution argument in Proposition 2. A one-sentence connection would strengthen the theoretical grounding.

12. **The "scheduled regularization" paragraph (curriculum dropout, RandAugment) reads as padding.** These methods are not meaningfully related to weight decay scheduling and their inclusion dilutes the section. If the purpose is to show that "scheduling other hyperparameters has shown only varying success," that point can be made in one sentence, pointing back to the intro.

13. **Chen et al. (2019) and Zhuang et al. (2022) are cited for nonconvex convergence but not described.** What do they prove? How does their analysis relate to the paper's Theorem 1? Without a brief description, these citations serve only as credential-padding.

14. **No discussion of Fisher-Rao regularization, spectral norm regularization, or other norm-based alternatives** that a reviewer might ask about as comparators to L2/WD. Even a brief acknowledgment that the paper focuses specifically on L2/decoupled WD and not other norm-based regularizers would forestall reviewer objections.

---

## Specific Suggestions

1. **Add a subsection on "Learning Rate Scheduling and LR-WD Interaction"** (4-6 sentences). Cite the milestone schedule (He et al., 2016), cosine annealing (Loshchilov & Hutter, 2017), and cyclical LR (Smith, 2017). Briefly note that the interaction between abrupt LR drops and fixed WD has been studied empirically but not formally characterized for adaptive WD schemes. This directly sets up the LR-WD coupling finding.

2. **Add 2-3 sentences on implicit bias and WD-norm theory** in the first subsection ("Weight Decay: Theory and Practice"). Cite Lyu & Li (2020) on gradient flow implicit bias with WD and Foret et al. (2021) on the connection between weight norm and sharpness. This grounds the budget equivalence claim theoretically.

3. **Restructure the "Dynamic and Adaptive Regularization" subsection** around a clearer taxonomy: (a) architecture-level adaptive regularization (LARS/LAMB), (b) coordinate-level adaptive regularization (CWD), (c) schedule-level adaptive regularization (meta-learning approaches, curriculum methods). Make explicit that AADWD belongs to category (c) and that no prior work in category (c) has used gradient-parameter alignment as the scheduling signal.

4. **Expand the CWD discussion** (2-3 additional sentences) to explain why CWD's theoretical motivation—avoiding regularization-gradient conflict—is precisely the hypothesis the paper tests. This makes the CWD instability finding theoretically motivated rather than an empirical curiosity.

5. **Add an explicit gap statement at the end of the section**, e.g.: "In summary, while prior work has established when weight decay improves convergence in theory and has explored adaptive regularization in various forms, no prior work has (a) empirically characterized the alignment quantity delta_t under standard training, (b) demonstrated the structural necessity of LR-WD coupling for adaptive WD schemes, or (c) established budget equivalence as a sufficient statistic for generalization. This paper fills all three gaps."

6. **Replace the broad "chen2019convergence, zhuang2022understanding" citations with brief descriptions** of what each proves and how it relates to this paper's Theorem 1. If they are not directly relevant, remove them.

7. **Add a forward pointer from Related Work to the method section** where each positioning claim is validated. E.g., "We verify this in Section 4.2" after claiming that delta_t is always near zero. This is especially important for claims about CWD instability, which a reviewer may want to fact-check against the experimental section.

8. **Explicitly state the paper's scope limitation**: the analysis is restricted to SGD with momentum and milestone/cosine LR schedules on image classification benchmarks. This preempts reviewer objections about generalization to Adam, transformer architectures, or LLM pretraining, without requiring the paper to cover those settings.
