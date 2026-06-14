# idea_debate - Iteration 3
**Date**: 2026-03-08

## 产出文件
- idea/perspectives/{innovator,pragmatist,theoretical,contrarian,interdisciplinary,empiricist}.md
- idea/proposal.md（最终综合提案）
- idea/alternatives.md（3个备选方案）
- idea/hypotheses.md（10个可检验假设）
- codex/idea_debate_review.md（Codex独立审查）

## 核心结论
最终方案: **Denoising-Time Adaptation (DTA)** -- 将DLM去噪迭代转化为显式TTT，通过LoRA在线更新实现参数级自适应。
- 4级消融层次: vanilla → DMI(embedding记忆) → SCP(自矛盾探测) → DTA(参数适应)
- 目标: Dream-7B + LLaDA-8B, Countdown/GSM8K/MBPP
- 预算: ~84 GPU·h, 4周

## Codex审查 (6/10)
- 理论包装过强，VDTA EM框架需降调为"交替优化视角"
- 需加入可信token子集过滤、回滚机制、compute-matched baseline
