# 项目: ttt-dlm

## 研究主题
遮蔽扩散语言模型的推理时计算扩展（ReMask-Retry / TTT / TCR）

## 背景与动机
遮蔽扩散语言模型（MDLMs）通过迭代去噪生成文本，但一旦 token 被确定就无法修改。本项目研究推理时计算扩展策略（ReMask-Retry、TTT、Best-of-N）能否提升生成质量。

本项目已完成 18 轮迭代，关键发现：
- TTT（6 个变体）：统计不显著（p=0.88）
- Best-of-N：完全无效（3x 计算量反而 +6.9%）
- ReMask-Retry：PPL 提升但由文本退化（重复）驱动
- 0.6B 模型：PPL "提升"是重复伪影，多样性下降
- LLaDA-8B：无退化但 PPL 恶化（+31.5%）
- TCR on Dream-7B（第 18 轮）：mean PPL 改善（30.3→17.9）但 median 未变

基于当前的结果，我希望你进一步去探索：
1. 是否能将 TTT 的结构、算法、理论或思路引入 DLM 中，让DLM 也拥有记忆能力，且 DLM 推理时自带多轮迭代相较 AR 无需额外的 TTT 过程。
2. 是否能将 TTT 作为插入层插入到现有的 DLM 中，为其直接提供推理时记忆能力

你要广泛的探索和查看相关领域的最新进展，获取新的 insight、idea 和想法，不必拘泥于框架，我们要做最新最好的工作，我们的目的是推动领域向前，发出高质量文章

## 初始想法

1. **轨迹一致性重遮蔽（TCR）** on Dream-7B — 用轨迹稳定性替代置信度选择重遮蔽目标（第 18 轮有初步结果）
2. **指标批判论文** — 重构为"PPL 何时无法评估 DLM"（EMNLP Findings）
3. **大模型验证** — 在 Dream-7B / LLaDA-8B 上测试，模型容量足够时方法是否有效
4. **基于训练的方法** — PRISM 风格的轻量适配器，学习 token 质量分数

## 关键参考文献
- ReMDM (arxiv 2503.00307): 原则性重遮蔽采样器
- PRISM (arxiv 2510.01384): 学习的逐 token 质量分数
- Soft-Masked Diffusion (arxiv 2510.17206): 软混合替代二值遮蔽
- Self-Rewarding SMC (arxiv 2602.01849): 平行粒子轨迹与重采样
- CoRe (arxiv 2602.04096): 上下文鲁棒重遮蔽
- ProSeCo (arxiv 2602.11590): 渐进式自纠正训练
- Dream 7B (arxiv 2508.15487): 最强开源 DLM
- LLaDA: 8B 遮蔽扩散模型
- MDLM (Sahoo et al., 2024): 吸收态离散扩散
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States
- Test-Time Learning for Large Language Models
- Test-Time Training with KV Binding Is Secretly Linear
- Titans: Learning to Memorize at Test Time

## 可用资源
- GPU: 4x on cs8000d，自行选取空闲 GPU
- 服务器: cs8000d
- 远程路径: /home/ccwang/sibyl_system

## 实验约束
- 实验类型: training-free（优先），轻量训练可接受，lora 可接受
- 模型规模: 0.6B (Qwen3), 7B (Dream), 8B (LLaDA)，以及小于8B 的其他模型
- 要同时报告 PPL 和多样性指标（第 15 轮教训）
- 必须定性检查生成文本（第 15 轮教训）
- ppl 等类似指标只能作为前期感性的参考指标，实际衡量模型性能需要到测试模型能力的 benchmark 上去做测试，初期实验要选用能够较快推理完成的 benchmark 来进行测试

## 目标产出
- 论文（NIPS,ICML,ICLR 等顶级会议正文）
- 以 nips 模板为例，页数不能少于 8 页，可适当将补充内容放到附录中

## 特殊需求
- 18 轮迭代的历史数据在 logs/iterations/ 和 logs/research_diary.md
- 所有实验代码在 exp/code/（ACA-DLM v1-v11, TCR-Dream v1-v3）
- 现有论文草稿在 writing/paper.md（ReMask-Retry 负面结果框架）
- 核心教训：PPL 可被重复文本 gaming，务必用多样性指标验证
- Conda 环境: sibyl_ttt-dlm（远程服务器）
