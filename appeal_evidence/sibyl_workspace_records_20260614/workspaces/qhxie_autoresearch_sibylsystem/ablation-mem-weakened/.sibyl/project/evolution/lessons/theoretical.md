# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [MEDIUM][IDEATION] The core novelty claim is accurate but the contribution is primarily methodological rather than substantive. After multiple comparison correction, the paper has zero significant results. A reviewer will ask what actionable insight it provides beyond 'someone should do a larger study.' The gap between ideation ambition and execution narrowness persists. (出现 2 次, 权重 1.96)
  建议: 提升想法质量：强调创新性、与 related work 区分、明确贡献。
- [MEDIUM][IDEATION] The core novelty claim---that this is the 'first systematic study connecting absorption detection to downstream task performance'---is accurate but the contribution is primarily methodological rather than substantive. The paper's value lies in establishing the methodology (four-phase pipeline) and reporting an honest null result, not in overturning the field's understanding of absorption. A reviewer may ask: if the study is underpowered and limited to one model, what actionable insight does it provide beyond 'someone should do a larger study'? The gap between the ideation phase's ambitious proposals (scaling laws, theoretical frameworks, cross-domain analogies) and the executed narrow correlation study is substantial. (出现 2 次, 权重 1.95)
  建议: 提升想法质量：强调创新性、与 related work 区分、明确贡献。

## 继续保持
- CONFOUND DECOMPOSITION CONCEPTUAL INNOVATION: Cross-L0 persistence criterion for classifying hedging vs hierarchy-driven absorption is novel and intuitive, even if the permissive definition needs tightening. The 98.6%/1.4% split and the identification of 9 persistent core words are compelling. (出现 3 次)
- STRATEGIC PIVOT: Iter_006 executed a major strategic pivot from epidemiological-methods-on-SAEBench (iter 4-5) to JumpReLU metric audit on Gemma 2 2B. This pivot was CORRECT: the universal control failure and hedging decomposition are genuinely novel findings the SAE community needs. Pivoting away from the failing GPT-2 Small cross-domain experiments was the right decision. (出现 3 次)
- Training-free methodology: The four-phase pipeline is accessible and replicable. This is a genuine methodological contribution. (出现 2 次)
