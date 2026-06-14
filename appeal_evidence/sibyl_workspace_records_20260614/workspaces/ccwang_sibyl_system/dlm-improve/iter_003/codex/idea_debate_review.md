# Codex 独立评审 - idea_debate

**评审时间**: 2026-03-12  
**模型**: Codex (GPT-5)  
**方式**: 由于专用 Codex MCP 路径不可用，本评审基于 workspace 内现有 `context`、`perspectives` 与 `debate` 材料独立完成。  
**说明**: 官方角色流程提到的 `idea/proposal.md` 在当前 workspace 中未找到，因此以下判断不假设存在一个已综合完成的最终提案。

## 总评

这轮 debate 的质量明显高于上一轮，最重要的进展不是“又提出了一个更聪明的新 controller”，而是**大多数视角已经共同接受：iteration 3 的主问题必须从方法扩张收缩为证据闭环、claim hygiene 和 reviewer-friendly audit bundle**。  

我的独立评分是：**8.2/10**。  

这个分数不代表“已经找到高把握 acceptance 配方”，而代表：**debate 已经把错误方向基本清掉，也初步形成了一个可信的 surviving spine。** 目前最值得推进的不是更大的方法故事，而是一个边界清楚的 DLM diagnostic / protocol paper。

## 最强 surviving direction

**最强 surviving direction** 是：

**以 `pragmatist` 为主轴，把当前工作收束成“training-free DLM 的可审计评估与最小选择性修订协议”论文；以 `CARD-84` 仅作为 reference controller，而不是新方法家族；再把 `empiricist` 的 matched-compute / sham-control / runtime-contract 审判层强制并入，最后只吸收 `theoretical` 与 `interdisciplinary` 中最可测、最克制的 framing 与 trajectory 指标。**

我支持这个方向的原因很简单：

1. 它最符合当前证据密度。现有资产更适合支撑 `observer / controller / runtime` 拆分、bucket audit、runtime fairness、task dependence，而不适合支撑“我们又找到一个更强 DLM controller”。
2. 它最符合 debate 共识。多个 critique 虽然措辞不同，但都在把主轴往 `pragmatist` 收拢，同时要求必须嵌入 `empiricist` 的硬负控。
3. 它最能避免再次 narrative overreach。`innovator` 的 selective-compute 语言有启发，但现在拿来做主轴，风险仍然高于收益。
4. 它即使产出偏负，也仍然可写。只要能证明某些 gain 只在何种 task / runtime / control 条件下才可信，这篇 paper 仍有成立空间。

## 我建议的 synthesis 结构

最稳的整合方式不是“选一个赢家，别的都扔掉”，而是分层吸收：

1. **主轴**: `pragmatist`
   - audit bundle
   - reference controller
   - claim-to-asset lineage
   - runtime fairness
   - task dependence

2. **必须并入的判决门**: `empiricist`
   - `DNB-84` matched-compute
   - `RAND-84` sham control
   - current-only `runtime_contract.json`
   - `self_check.json` 或等价自检资产

3. **只能作为 framing 的配件**: `theoretical`
   - 保留 observer/controller/runtime 三层语言
   - 只保留“高 observer 分数样本在 matched-compute 下是否表现出更高净修复率”这种可测句子
   - 不要把它抬成“我们建立了识别性理论”

4. **只能作为测量增强的配件**: `interdisciplinary`
   - 可吸收 `time-to-first-correct / time-to-stable-correct / relapse risk`
   - 不要把 SDT / survival analysis 升格成论文主语言

5. **只能降格使用的语言资源**: `innovator`
   - 可在 discussion/future work 中保留少量 selective-compute / eligibility wording
   - 不能反过来驱动当前 paper 的主问题定义

## 最大残余风险

**最大的残余风险** 不是实验不够花，而是 synthesis 会再次把一个“协议主线 + reference controller + 硬负控审判”的收缩稿，悄悄写回“方法主线 + selective-compute insight + task-general gain”。

更具体地说，我最担心三种滑坡：

1. **把 evidence closure 写成 identification closure**  
   做了 matched-compute 和 sham-control，不等于已经完成 observer/controller/sampler 的严格识别。

2. **把 reference controller 写成 validated controller**  
   `CARD-84` 当前最多是 paper 内的 reference component，不是已经被充分证明的通用策略。

3. **把 task dependence 写成免罪符**  
   “reasoning 有 gain、code 会塌”只有在 failure mode 被明确定位后才是结果；否则它只是吸收负结果的修辞。

如果这一风险没有被压住，这轮 debate 即使在内部看起来很成熟，最终也仍会在摘要、引言和结论处重复 iteration 2 的错误。

## synthesizer 绝对不能 overclaim 的内容

下面这些话，当前证据阶段**不能写**，或者只能在极强限定下写：

1. 不能写“我们提出了新的 DLM controller family”。
2. 不能写“calibration / entropy improves DLM generation”这类泛化因果句。
3. 不能把 `n=96` 或 `n=100` audited slice 的结果外推成总体 benchmark headline。
4. 不能把 `observer useful` 直接写成 `controller valid`。
5. 不能把 matched-compute / sham-control 存活写成“机制已识别”。
6. 不能把 reasoning 上的局部 gain 写成 code、reasoning、general generation 的统一机制。
7. 不能把 runtime fairness 卡片的完整性写成方法贡献本身已经成立的替代物。
8. 不能把 trajectory 指标或跨学科术语写成主要新理论，除非它们对应的可测对象和证据已经单独站稳。

## 我认为最安全、也最有力的 paper-level claim

如果下一轮实验通过最关键的 reference-validity 审判，我认为最安全的主张上限应当是：

> 我们提出并验证了一套面向 training-free DLM inference claims 的可审计协议：它将 observer、reference controller 与 runtime contract 显式拆开，并表明某些 selective revision gain 只有在 matched-compute、sham-control 与 task-specific audit 下才值得被有限度地解释为 targeting value。

注意这里的关键词应该是：

- `protocol`
- `reference controller`
- `limited targeting value`
- `task-specific`
- `audited`

而不是：

- `new controller`
- `general gain`
- `calibration-driven improvement`
- `identified mechanism`

## 最后的独立判断

如果只能给 synthesizer 一句话建议，我会给这句：

**请把这篇 paper 写成“怎样让一个 DLM 小 gain 变得可信”，而不是“怎样把一个小 gain 重新包装得更大”。**

当前 debate 已经足够清楚地说明：

- 主轴应当是 `pragmatist`
- 生死门必须是 `empiricist`
- 语言纪律由 `theoretical` 提供
- 测量增强最多吸收 `interdisciplinary`
- `innovator` 现在不能反客为主

只要 synthesis 守住这个顺序，这轮 idea debate 就是成功的；一旦顺序反过来，项目会再次回到“概念听起来更大，但证据边界更糊”的旧路上。
