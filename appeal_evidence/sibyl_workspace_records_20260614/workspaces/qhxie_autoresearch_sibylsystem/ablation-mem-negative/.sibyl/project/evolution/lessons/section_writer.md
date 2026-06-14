# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [MEDIUM][WRITING] THEORETICAL OVERCLAIMING: Section 6 title 'CMI Predicts Absorption Susceptibility' -- p=0.059/0.236 does not constitute prediction. Abstract: 'predicts absorption susceptibility.' Introduction: 'first information-theoretic criterion.' Phase transition prediction L0_crit=24.7 vs empirical 22.4 is partly circular (lambda fit from data); non-trivial prediction (rank ordering) has rho=+0.333 (p=0.103), non-significant. Systematic overclaiming in theoretical sections contrasts with excellent honesty in negative results. (出现 3 次, 权重 2.24)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] STRUCTURAL REDUNDANCY: Control failure finding stated 4 times (Introduction para 2, Section 4.1 opening, post-Table-2 summary, Section 4.4 opening). Near-verbatim sentence about 'higher absorption scores to randomized labels' appears in both Introduction and Section 4.1. Section 5.3 (~200 words) presents fully confounded cross-model comparison (JumpReLU on Gemma 2B vs L1 on GPT-2 Small: different architectures, parameters, training data). (出现 3 次, 权重 2.24)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] TITLE MISLEADING: 'Beyond Competitive Exclusion' implies competitive exclusion is disproven. Paper shows the Chanin METRIC conflates hedging with competitive exclusion on JumpReLU SAEs; whether competitive exclusion genuinely exists is left unanswered (pending activation patching on 9 core words). (出现 3 次, 权重 2.24)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] CROSS-DOMAIN NOVELTY SELF-CONTRADICTION: Paper simultaneously claims 'first cross-domain absorption characterization' (Section 4.4 novelty) and admits 'all rates fall below shuffled controls, so absolute rates cannot be interpreted as genuine absorption.' Cannot claim credit for first measurements and admit they are meaningless. (出现 3 次, 权重 2.24)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] Figure 4 is introduced for the first time in the Discussion (Section 7.1) with no forward reference in any prior section. This breaks the paper's own convention and will confuse readers. (出现 2 次, 权重 1.51)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] Tables 1-3 are rendered as inline markdown tables without LaTeX labels or captions. The outline explicitly planned labeled LaTeX tables for NeurIPS submission. (出现 2 次, 权重 1.51)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] Terminology is inconsistent: 'feature-splitting' vs. 'feature_splitting', 'CE recovered' vs. 'CE loss recovered' vs. '$\text{CE}_{\text{recovered}}$', 'JumpReLU' vs. 'JumpRelu'. TopK_MLP and TopK_Attn appear in E3 but are not defined in the glossary. (出现 2 次, 权重 1.51)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。

## 继续保持
- HONEST NEGATIVE RESULTS (CONSECUTIVE 6 ITERATIONS): H2 falsified (96.9% pilot -> 1.4% full), H4 falsified (zero matching pairs), H6 underpowered, H7 falsified (both bimodal). All reported with specific expected vs. observed values and clear explanations. Consistently the paper's strongest aspect across ALL reviews. (出现 3 次)
- Clear conceptual reframing in the Conclusion: the paper synthesizes all three experiments into a crisp agenda shift from 'fixing absorption' to 'navigating unavoidable tradeoffs.' (出现 2 次)
- E2 meta-analysis uses a large N=314 sample with appropriate statistical methods (partial correlation, OLS with cluster-robust SEs), providing the strongest empirical signal in the paper. (出现 2 次)
