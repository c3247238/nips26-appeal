# Codex 独立评审 - result_debate

**评审时间**: 2026-03-14  
**模型**: Codex (替代独立审查通道)

## 评审意见

### 1. 你们仍然低估了“proposal 与最新 full-scale 证据错位”这件事的严重性

当前 workspace 已经在 result_debate 中明确承认 `cand_espd` 才是被 full-scale 支持的 serious line，但 `proposal.md` 依然把 `cand_bsr` 写成 quality front-runner。这不仅是 narrative 不整洁，而是会直接影响后续 `experiment_decision` 与写作阶段的逻辑一致性。  
如果你们继续带着这个错位进入下一阶段，后面每一个 reviewer-friendly artifact 都会被迫同时服务两个相互竞争的主线，最后谁都讲不清楚。

### 2. 你们对 fixed-frontier sham 的依赖很重，但还没有把它做成 reviewer 难以攻击的控制

当前主 story 几乎全部依赖 `cand_espd > ESPD-FixedFrontier`。但 sham 仍存在明显非机制差异：

- `peak_vram_mb`: `15289` vs `41973`
- `effective_batch_size`: `54` vs `52`
- `auxiliary_overhead_sec`: `2569.81` vs `3049.62`
- `avg_tokens_changed`: `4.16` vs `0.29`

这意味着 reviewer 完全可以说：你们证明的不是 entropy routing 更优，而是“这个 fixed-frontier 实现更差”。  
如果你们下一步不主动补一个更 matched 的 sham，这会成为整篇稿子最致命的单点脆弱处。

### 3. 当前所有人都在说“claim scope 要收窄”，但还没有把它变成结构化 artifact

这是一个危险信号。  
当一个项目进入“我们都知道该更诚实”阶段，如果没有立即生成以下结构化工件，后续写作阶段几乎一定会反弹回更夸张的 narrative：

1. `claim_boundary.md`
2. `runtime_lineage_table.json/md`
3. `primary_endpoint.md`

你们现在缺的不是更多讨论，而是把“不能说什么”落成文件。否则写作时会自然回到更顺手的强 claim。

### 4. 结果辩论已经足够说明：iteration 4 的真正科学问题变了

当前最值得强调的不是 `cand_espd` 有多强，而是：

> 旧问题“entropy 能否直接做 semantic controller”已经被削弱；新问题变成“entropy 是否适合作为 compute router，以及这一 gain 是否在更严格控制下仍成立”。  

这不是文字游戏，而是研究问题的重写。  
如果 `experiment_decision` 阶段还沿着旧问题组织下一步动作，你们会浪费一轮迭代。

### 5. 你们还没有充分利用当前结果的真正优势：它很适合做“bounded contribution paper”

很多项目死在“效果不够大”。但当前这条线其实有一个别的优势：  
它非常适合被写成一篇 **小增益、强控制、强归因纪律** 的稿子。  

这类稿子的赢法不是扩大结果，而是把以下三件事做到 reviewer 没法挑：

- claim boundary 非常清楚
- sham control 足够强
- runtime attribution 十分透明

如果你们继续把资源投在“能不能再涨 0.5pp”，而不是把这三件事补齐，可能会错过当前结果最适合的发表路径。

## 评分

**6.5 / 10**

### 评分理由

- `+` full-scale evidence 真实存在，不是空想
- `+` result_debate 已经正确识别出了主线错位与 bounded-claim 需求
- `-` 关键 sham 仍不够 matched
- `-` proposal / evidence 不一致尚未结构化修复
- `-` reviewer-facing claim-boundary artifact 仍缺失

## 建议

1. 在进入 `experiment_decision` 前，先生成一份 `claim_boundary.md`，明确：
   - 当前支持什么
   - 当前不支持什么
   - 哪些 hypothesis 仍是 untested / inconclusive
2. 立刻规划一个更 matched 的 fixed-frontier sham 或 accounting-control sham。
3. 不要把下一轮优先级放在“提升 headline accuracy”，而应放在：
   - 外推验证
   - runtime-lineage artifact
   - stronger sham
4. 在所有后续阶段，默认把 `cand_espd` 视作当前主线，把 `cand_bsr` 视作等待重新证明自己的 challenger。
