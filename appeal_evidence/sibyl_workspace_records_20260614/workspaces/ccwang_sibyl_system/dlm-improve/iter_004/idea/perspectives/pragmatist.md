# Pragmatist Perspective

## 结论先行

这轮我不支持继续把 `MGCD` 或 `DSG` 当成 mainline。真实 pilot 证据已经足够明确：

- `MGCD-lite` 在 `GSM8K audited slice` 上相对 `RAND-84` 的 `net_repaired=-4`，`MBPP audited slice` 上是 `-2`
- `DSG` 在 `GSM8K audited slice` 上相对 `RAND-84` 仅 `0`，`MBPP audited slice` 上是 `-1`
- `pilot_plan.json` 的 fail branch 已触发，当前正确动作是回到 `idea_debate`，而不是继续追逐旧的 front-runner 叙事

所以实用主义视角下，这一轮应该做两件事：

1. 承认 `MGCD / DSG` 已被 screening pilot 筛掉，停止追加 narrative budget
2. 把 serious candidate pool 收缩到 **2-3 个更容易实现、更容易证伪、且更贴近现有开源生态** 的方向

我建议保留的 pool 是：

- `cand_espd`: `Entropy-Stable Parallel Decoding`
- `cand_bsr`: `Boundary-Stable Revision`
- `cand_agd_lite`: `AR-Guided Diffusion Lite`

其中我的 pragmatist front-runner 不是 `MGCD`，而是 `cand_espd`。

## 我接受哪些已有证据

### 必须承认的负结果

- iteration 4 的 screening pilot 已经完成，不是“还没跑”
- `MGCD-lite` 与 `DSG` 都没有穿透 `RAND-84`
- `cand_bsr` 这轮没有执行，所以它只能作为下一轮重新立题的对象，不能被拿来替 `MGCD` 续命

### 仍然值得复用的正信号

来自项目长期 memory 的可信信号没有被这轮负结果推翻：

- `SC-ECE=0.22`，说明 DLM 的置信度确实可读
- raw entropy 与错误有明显相关性，`Pearson r=0.44`
- revision 曾带来 `+5.3%` 改善
- `CARD-84` 明显优于 `DNB-84`，说明“有选择地修”优于“均匀多跑”
- temperature annealing 已被否证，`isothermal T=1.0` 仍应保留
- `128 steps < 64 steps` 的非单调现象支持“自适应步数/自适应冻结”而不是“盲目更多步”

对我来说，这些信号的组合含义很直接：

- `observer signal` 是真的
- 但 `memory-heavy controller story` 没有站住
- 下轮更值得赌的是 **低复杂度的 acceptance / freezing / caching / boundary object**

## 定向文献与工程扫描

说明：

- 本轮按技能要求先做了定向检索
- 当前 runtime 没有可用的 Google Scholar MCP，因此我用 **arXiv + 官方 GitHub / project page / official PDF** 替代
- 我只保留与“training-free 提速 / 不改训练的性能提升 / LLaDA 可复用工程实现”直接相关的条目

### 1. LLaDA 官方信息已经明确承认“效率还有很大优化空间”

- LLaDA 官方仓库明确写到，当前采样慢于 AR baseline 的原因包括：固定上下文长度、还不能充分利用 `KV-cache`、且步数减少会掉性能；同时官方 README 直接点名 `block diffusion`、`Fast-dLLM`、`dLLM-Cache` 作为可适配方向  
  来源: https://github.com/ML-GSAI/LLaDA
- 同一仓库在 2025-10 已补充 batch inference 支持和 `lm-evaluation-harness` evaluation code，这说明“先换更高效 inference substrate，再谈新算法”是符合 upstream 演进方向的  
  来源: https://github.com/ML-GSAI/LLaDA

### 2. Speed-up 方向已经有多个 training-free、且直接支持 LLaDA/Dream 的开源实现

- `dLLM-Cache` 提出 training-free adaptive caching，针对“prompt 静态 + response 局部动态”做长区间 prompt cache 和部分 response update，报告在 `LLaDA 8B` / `Dream 7B` 上可达 `up to 9.1x speedup`，且声称多数任务无性能损失  
  来源: https://arxiv.org/abs/2506.06295  
  代码: https://github.com/maomaocun/dLLM-cache
- `Fast-dLLM` 直接面向 `LLaDA` / `Dream`，把 `KV cache for block-wise decoding` 与 `confidence-aware parallel decoding` 结合起来；仓库 README 给出多组 `2x-6x` 以上加速，并展示在 `GSM8K` / `HumanEval` 上可通过阈值调节 speed-quality tradeoff  
  来源: https://github.com/NVlabs/Fast-dLLM
- `dInfer` 是更像“推理框架层”的工作，支持 batched inference、LLaDA family、`lm-eval-harness`、以及 vLLM / SGLang backend；它不是单一方法，而是说明“先把运行时系统换对”本身就是实用路线的一部分  
  来源: https://github.com/inclusionAI/dInfer

### 3. Block / semi-autoregressive 路线说明“并行生成 + 可控质量损失”是成熟设计轴

- `SSD-LM` 很早就证明 semi-autoregressive diffusion language modeling 是一条成立的 design axis，核心思想就是按 block 生成而不是整句全局重算  
  来源: https://arxiv.org/abs/2210.17432
- `Block Diffusion (BD3-LMs)` 则把这种思路推进到更强的统一框架：通过 block size 在 AR 与 diffusion 间做插值，并明确把“quality vs sampling efficiency trade-off”当作一等公民  
  来源: https://arxiv.org/abs/2503.09573  
  代码: https://github.com/kuleshov-group/bd3lms

### 4. “Forecast-then-verify” 已经在 diffusion 家族里反复被证明有效

- 图像/视频 DiT 社区里，`token-wise feature caching`、`relational feature caching`、`SpeCa` 等工作都在做 training-free acceleration，核心不是更复杂建模，而是 **预测哪些位置/层/步值得重新算，哪些可以复用**  
  代表来源: https://arxiv.org/abs/2410.05317, https://arxiv.org/abs/2509.11628
- `FLASHDLM` 更直接：一方面用 `FreeCache` 做 shrinking active window 的 KV reuse，另一方面用小 AR model 只做 agreement signal，不直接纠错，声称在 `Dream-7B-Instruct` / `LLaDA-8B-Instruct` 上可以有明显 speedup，同时尽量保质量  
  来源: https://openreview.net/pdf/f715cbedc778ae22b0f270f0ffd038694a2a7913.pdf
- `Speculative Diffusion Decoding` 证明 diffusion drafter + AR verifier 这个范式本身是可行的，并报告比 vanilla generation 高得多的速度收益；虽然它的 setting 不完全等同于我们现在的 LLaDA revision pipeline，但它给出了很强的工程 prior  
  来源: https://neurips2024-enlsp.github.io/papers/paper_68.pdf

### 5. 如果目标是 code / structured generation，constraint 本身就是 cheap win

- `Constrained Decoding of Diffusion LLMs with Context-Free Grammars` 已经给出 `LLaDA` / `Dream-Coder` 上的通用 CFG constrained decoding，实现上保证 syntactic correctness，并报告 functional correctness 可提升 `up to 7%`，且额外开销不大  
  来源: https://github.com/eth-sri/constrained-diffusion

这条线不适合作为主线 candidate，因为它解释不了 `GSM8K`，但对 `MBPP/HumanEval/JSON` 很可能是低成本 bonus。

## 我的候选池与排序

## Candidate 1

### `cand_espd`: Entropy-Stable Parallel Decoding

这是我的 pragmatist front-runner。

#### 核心想法

不再发明新的 memory graph，也不引入第二个大模型；而是把项目里已经成立的信号直接工程化：

- raw entropy 有用
- late-step 过度改写会伤害质量
- denoising 非单调，说明并不是每一步都值得全量重算

具体做法：

- 每步记录 token 的 `top-1 prediction`、`raw entropy`、以及是否连续 `k=2` 步保持不变
- 只有同时满足“低熵 + 连续稳定”的 token/span 才被永久接受
- 已接受区域冻结，不再参与后续 full recompute
- 剩余高不确定区域继续走 diffusion
- 结合 prefix / region cache，只对 active region 做重算

#### 为什么它比 `MGCD` 更值得先做

- 不需要 dual draft
- 不需要 tri-state memory graph
- 不引入“是不是只是 hidden compute”这一大块解释包袱
- 和现有 `CARD` 代码路径最接近，改动半径小
- 同时有机会拿到两种收益：
  - 减少 wall-clock
  - 减少 late-step harm

#### 最小实现

- 在当前 `CARD` / standard denoise loop 上增加 `temporal stability buffer`
- 增加 span merge 与 boundary lock
- 增加 active-region shrinking
- 记录：
  - `stable_token_ratio`
  - `stable_span_count`
  - `active_region_len`
  - `cache_hit_ratio`
  - `wall_clock_sec`

#### 预期收益

- 目标不是论文级夸张数字，而是先拿到 **可信的小胜**
- 我会把 success 定义为以下二者之一：
  - `>=1.5x` wall-clock speedup，同时准确率下降 `<=1pp`
  - 或 wall-clock 基本不变，但相对 `CARD-84` / `RAND-84` 在 `GSM8K audited slice` 上出现真实 separation

#### 风险

- 过早冻结，导致 reasoning chain 被锁死
- 稳定性规则太保守，最后几乎没有提速
- 只在短输出有用，长输出收益下降

#### 成功概率

`0.55`

#### 单个 pilot 任务预算

- 实现: `0.5-1.0` 天
- 单个 50-sample audited slice: `20-40` 分钟
- 前提: 必须使用最大 safe batch，而不是 `batch_size=1`

## Candidate 2

### `cand_bsr`: Boundary-Stable Revision

我支持继续保留 `BSR`，但只把它当成 **quality-first object candidate**，不再当 `MGCD` 的附庸。

#### 核心想法

这条线只回答一个更小的问题：

> 失败到底是因为 token-wise revision object 太碎，还是因为真的缺 memory？

`BSR` 的最小版本应该故意保持朴素：

- 单轨信号
- island 聚合
- boundary locking
- 限制 revision 只发生在 disputed spans
- 不上 dual draft，不上 memory graph

#### 为什么它还值得活着

- `cand_bsr` 本轮没跑过，所以它不是被证伪，只是没被测试
- 它正好是 `MGCD` 必需性的 clean control
- 如果 `BSR` 都过不了 `RAND-84`，那就可以更有力地说“不是 memory 没建好，而是 repair object 这条线整体不成立”

#### 我对它的要求

- 不许写成“大而全新方法”
- 不许偷偷接入 hidden compute
- 必须与 `CARD-84`、`RAND-84` 做 matched-compute 对比

#### Success 条件

- 相对 `RAND-84` 至少出现小幅正 separation
- 或与 `CARD-84` 同 NFE 下更低 harm、更好 wall-clock

#### 风险

- 其实只是 `CARD` 的 span-level 重命名
- `DSG` 已经失败，说明单轨信息可能根本撑不起 controller

#### 成功概率

`0.35`

#### 单个 pilot 任务预算

- 实现: `1-1.5` 天
- 单个 50-sample audited slice: `30-45` 分钟

## Candidate 3

### `cand_agd_lite`: AR-Guided Diffusion Lite

这是高风险高回报的第三候选，只应在前两个 candidate 至少有一点信号时继续推进。

#### 核心想法

借鉴 `FLASHDLM` / `SpecDiff` 的 “draft-then-verify / agreement-only guidance”：

- 大 DLM 负责并行提出候选 token
- 一个小 AR 模型只做 agreement signal
- 只有一致的 token 才被放行
- AR guider 不负责重写文本，只负责决定“这一步能安全 unmask 多快”

实作上我只接受小 guider：

- `Qwen2.5-0.5B` 或 `1.5B`
- 不能再上一个 7B guider，否则 compute neutrality 失去意义

#### 为什么保留，但不排第一

- 外部 prior 很强，training-free，也有公开代码路径
- 但它天然带入 hidden compute 风险
- 一旦评价口径不严，很容易出现“表面更快，实际只是把 compute 挪到 guider”

#### Success 条件

- 用 matched wall-clock 或 matched total FLOPs 做比较
- 至少拿到：
  - 明显快于 vanilla / `CARD`
  - 且准确率不掉太多
- 如果 agreement rate 很低，立即 kill

#### 风险

- guider 额外开销抵消收益
- agreement signal 对数学题有用、对代码没用，或反过来
- 引入第二模型会让实验 pipeline 更脆弱

#### 成功概率

`0.40`

#### 单个 pilot 任务预算

- 实现: `1.5-2.0` 天
- 单个 50-sample audited slice: `40-60` 分钟

## 我明确不建议作为 mainline 的方向

### `MGCD` 续命版

不建议。原因很简单：它已经拿到了真负结果，不是 launch failure。

### `DSG` 续命版

也不建议。它已经提供了“更轻但不分离”的证据，继续加预算只会重复这条事实。

### 纯系统优化当作 research contribution

我支持做系统优化，但不支持把“终于把 batch 调大了、打开 flash attention 了”包装成研究 idea 本身。

它们应该是 **所有 candidate 的共用 runtime floor**：

- batch size probe 到最大 safe 值
- 尽量启用 `flash_attention_2`
- `torch.compile`
- 评估 `dInfer` / `lm-eval-harness`
- 多 GPU 并行拆方法

这些是必须先做的 hygiene，不是 novelty。

### CFG constrained decoding 作为主线

不建议作为当前主线，因为它对 `GSM8K` 帮助有限，解释面太窄。

但如果后续要救 `MBPP/HumanEval`，它是非常好的 task-specific add-on。

## 推荐的执行顺序

### P0. 统一 runtime floor

先做一次不是论文贡献、但必须完成的系统对齐：

- safe batch size probe
- 开 `flash_attention_2` / `torch.compile`
- 同时记录 `wall_clock_sec`、`tokens_per_sec`、`peak_vram_mb`
- 如果 `dInfer` 可以直接接 LLaDA-8B-Instruct，就把它作为 speed control baseline

如果这一步不做，后面所有“提速”结论都容易失真。

### P1. `cand_espd` on GSM8K audited slice

这是我最推荐的第一枪，因为：

- 代码改动最小
- 最贴现有正信号
- 最容易在 `<=1 hour` 内得到 go/no-go

### P2. `cand_bsr` on GSM8K audited slice

若 `ESPD` 只有 speed gain 没有 quality gain，就补 `BSR`，测试 object-level repair 是否仍有生机。

### P3. `cand_agd_lite`

只有当前两者里至少有一个出现真实 signal 时才推进。否则它太容易把系统复杂度拉高。

### P4. MBPP / HumanEval 扩展

- 若目标是 code correctness，优先把 `CFG constrained decoding` 作为 add-on，而不是主线候选
- 若目标仍是 reasoning，先跑 `GSM8K`，不要让 benchmark 维度过早扩散

## 清晰的 kill rule

为避免再次发生 front-runner drift，我建议写死 kill rule：

- 如果 `cand_espd` 不能带来 `>=1.5x` speedup，且也不能超过 `RAND-84`，kill
- 如果 `cand_bsr` 不能超过 `RAND-84`，kill
- 如果 `cand_agd_lite` 在 matched wall-clock 下没有净收益，kill
- 任一候选若必须依赖额外隐藏 compute 才看起来更强，kill

## Pragmatist Bottom Line

这轮我支持的不是“更复杂的 DLM controller”，而是：

- **一个 quality-first 的小对象候选**: `BSR`
- **一个最有工程胜算的 speed/performance 候选**: `ESPD`
- **一个有文献支持但需要严格控 confound 的高回报候选**: `AR-Guided Diffusion Lite`

如果只允许保留两个 serious candidates，我会选：

1. `cand_espd`
2. `cand_bsr`

如果必须选一个 pragmatist front-runner，我选 `cand_espd`，理由不是它最“理论漂亮”，而是它最符合这轮真实证据：

- 旧的复杂对象已经输过一次
- 真正仍然活着的信号来自 entropy / stability / adaptive compute
- 公开生态已经证明 caching、parallel decoding、agreement-based verification 是可以直接落地的

这才是现在最值得下注的 training-free 路线。
