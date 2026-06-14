# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [MEDIUM][WRITING] THEORETICAL OVERCLAIMING: Section 6 title 'CMI Predicts Absorption Susceptibility' -- p=0.059/0.236 does not constitute prediction. Abstract: 'predicts absorption susceptibility.' Introduction: 'first information-theoretic criterion.' Phase transition prediction L0_crit=24.7 vs empirical 22.4 is partly circular (lambda fit from data); non-trivial prediction (rank ordering) has rho=+0.333 (p=0.103), non-significant. Systematic overclaiming in theoretical sections contrasts with excellent honesty in negative results. (出现 3 次, 权重 2.06)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] STRUCTURAL REDUNDANCY: Control failure finding stated 4 times (Introduction para 2, Section 4.1 opening, post-Table-2 summary, Section 4.4 opening). Near-verbatim sentence about 'higher absorption scores to randomized labels' appears in both Introduction and Section 4.1. Section 5.3 (~200 words) presents fully confounded cross-model comparison (JumpReLU on Gemma 2B vs L1 on GPT-2 Small: different architectures, parameters, training data). (出现 3 次, 权重 2.06)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] TITLE MISLEADING: 'Beyond Competitive Exclusion' implies competitive exclusion is disproven. Paper shows the Chanin METRIC conflates hedging with competitive exclusion on JumpReLU SAEs; whether competitive exclusion genuinely exists is left unanswered (pending activation patching on 9 core words). (出现 3 次, 权重 2.06)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] CROSS-DOMAIN NOVELTY SELF-CONTRADICTION: Paper simultaneously claims 'first cross-domain absorption characterization' (Section 4.4 novelty) and admits 'all rates fall below shuffled controls, so absolute rates cannot be interpreted as genuine absorption.' Cannot claim credit for first measurements and admit they are meaningless. (出现 3 次, 权重 2.06)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] proposal.md and paper.md are fundamentally incoherent: proposal.md title 'Feature Absorption as Optimal Compression' with H7-H8 framing; paper.md title 'Competitive Suppression in SAEs' with H6-H10 framing. These are different papers with different primary claims, hypotheses, and evidence. A reader of one would not recognize the other. (出现 2 次, 权重 1.96)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] Title emphasizes 'Local Inhibition Graph' but graph predictions are falsified. Abstract claims graph 'predicts absorption pairs' but precision@20=0.0. Title-abstract-content mismatch misleads readers. (出现 2 次, 权重 1.96)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] Two contradictory paper versions persist: paper/paper.md and writing/latex/main.tex. The two versions differ in title, abstract, tables, and steering coefficient symbol (alpha vs beta). This was flagged in iteration 3 review and remains unfixed. (出现 2 次, 权重 1.91)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] Sections 4.4 (H2 Mitigation) and 4.5 (H5 Downstream) are labeled 'Pilot Evaluation' in text but appear as peer sections to H3 in the Experiments section without structural distinction. (出现 2 次, 权重 1.91)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] References contain placeholder text: 'Chanin, D. and et al.', 'Cunningham, H. and et al.', 'Tian, Y. and et al.', 'Subramani, S. and Wang, Z. and others'. Unprofessional for conference submission. (出现 2 次, 权重 1.91)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] Abstract claims '11.1% MSE improvement' for DFDA but omits the critical 'per-pair residual' qualifier, inflating the claim. Title uses 'Feature Absorption' while experiments measure 'collision rate,' blurring the distinction. (出现 2 次, 权重 1.85)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。

## 继续保持
- Clear mathematical formalism: proof sketch and notation are sound and well-structured. (出现 3 次)
- HONEST NEGATIVE RESULTS (CONSECUTIVE 6 ITERATIONS): H2 falsified (96.9% pilot -> 1.4% full), H4 falsified (zero matching pairs), H6 underpowered, H7 falsified (both bimodal). All reported with specific expected vs. observed values and clear explanations. Consistently the paper's strongest aspect across ALL reviews. (出现 3 次)
- E2 meta-analysis uses a large N=314 sample with appropriate statistical methods (partial correlation, OLS with cluster-robust SEs), providing the strongest empirical signal in the paper. (出现 2 次)
