# Round 2 Idea Debate Synthesis

## 背景

`idea_validation_decision` 已经基于 pilot 证据做出 `PIVOT`：

- `TIGER-64+3` 在 `GSM8K` 上仅与 `Entropy-Revise-64+3` 打平
- `CORE-proxy-64` 明显更强
- `task gating` 只能部分修复 `HumanEval` 语法失败，且 reasoning 略掉点

因此，round 2 的 `idea_debate` 不再讨论“如何继续救 TIGER”，而是讨论：

**PIVOT 之后，哪一种新 framing 最值得进入 planning。**

## 六个视角的共识

1. `cand_diag` 应成为主候选
2. calibration 应保留为诊断工具，而不是主方法驱动
3. `code` 只能作为 boundary / appendix，不应再占据主 headline
4. honest compute accounting 本身就是贡献的一部分
5. proposal 必须显式承认 `TIGER` 没有形成方法胜利

## 最终选择

### 选中的主候选

`cand_diag`

### 工作标题方向

**When Revision Helps Diffusion LMs, When It Hurts, and Why Honest Compute Accounting Changes the Story**

### 主张

1. `revision benefit` 强烈依赖任务结构
2. `actual compute` 匹配会改变 training-free DLM inference 方法的相对排序
3. calibration correction 有诊断价值，但不足以单独解释或驱动质量收益
4. code 上的 cheap syntax guard 只能部分修复局部 revision 的结构性失败

## 对 planning 的直接输入

1. 以 `GSM8K` 为主 benchmark
2. 只在需要验证主叙事稳定性时扩展到 `MATH500`
3. `HumanEval` 仅作为 boundary appendix
4. 方法集锁定为：
   - `Standard-64`
   - `DNB-84`
   - `Prophet-64`
   - `CORE-proxy-64`
   - `Entropy-Revise-64+3`
   - `TIGER-64+3`

## 决议

round 2 `idea_debate` 的结果不是生成一个新的方法名词，而是把研究对象重新定义成一篇 **compute-normalized diagnostic / benchmark** 论文。接下来应进入 `planning`，围绕 `cand_diag` 设计最小但足够有力的验证与写作路线。
