# Result Debate Synthesis

## 1. Consensus Map

六个视角在以下几点上高度一致：

1. `cand_espd` 是当前 iteration 4 **唯一被 full-scale 结果直接支持**的 serious line。`GSM8K` 全量 `1319` 样本上，candidate accuracy `0.4041`，高于 `CARD-84=0.3965`、`RAND-84=0.3980`，bundle verdict 为 `ADVANCE`。
2. 当前最硬的机制证据来自 matched sham。`ESPD-FixedFrontier` 在相同 `frontier_ratio=0.1211` 下只有 `0.3988` accuracy、`105.73 tok/s`，而 `cand_espd` 达到 `124.42 tok/s`，速度优势 `+18.69 tok/s`，质量优势 `+0.53pp`。
3. 这组结果不支持把 entropy 继续写成 semantic controller；它更像 **routing / stopping / compute allocation** 信号。
4. 当前信号是真实但温和的。shared controls 本身已非常接近，candidate 只领先 `+0.61pp ~ +0.76pp`，因此不能写成压倒性突破。
5. proposal 里 object-level 主线 (`cand_bsr`) 并未被这轮 full-scale 结果直接验证。当前 evidence 已经落后于旧 narrative 排序，必须改写。

## 2. Conflict Resolution

主要分歧集中在两点：

### 分歧 A：当前结果到底有多强？

- `optimist` 认为这已经足够把 `cand_espd` 升级为 serious speed-line mainline。
- `skeptic` 与 `methodologist` 指出：差值很小，统计上站不稳，fixed-frontier sham 也不是完全 matched。

**综合判断：**

这是一条 **weak-to-moderate positive result**。它已经超过“只是 pilot 幻觉”的门槛，但还没有达到“机制已被显著证成”的门槛。  
因此正确表述不是“breakthrough”，而是：

> 在统一 runtime contract 下，entropy-routed compute 相对 matched fixed-frontier sham 展现出可信但幅度不大的 gain，值得继续推进。

### 分歧 B：下一步应继续 `cand_espd`，还是回到 `cand_bsr`?

- `strategist` 主张 narrowed `PROCEED`，以 `cand_espd` 为当前主线。
- `revisionist` 认为真正的关键是把 proposal 从 object-first narrative 改成 evidence-first narrative。

**综合判断：**

两者并不冲突。正确动作不是大 pivot，而是 **内部叙事 pivot + 外部执行 proceed**：

- 叙事上：`cand_espd` 升为当前主线，`cand_bsr` 下调为 challenger
- 执行上：继续推进 `cand_espd`，但必须附带最小方法学补强

## 3. Result Quality Score

**6.7 / 10**

### 给分理由

- `+` full-scale `GSM8K`，不是小样本 pilot
- `+` shared controls 与 matched sham 同时存在
- `+` 结果方向清晰，至少把 entropy 的角色从 semantic controller 重定位到了 router
- `-` 效果幅度小，统计显著性不足
- `-` runtime-lineage 仍不够 reviewer-friendly
- `-` fixed-frontier sham 不是完全 matched，routing 与 stopping 尚未完全拆开
- `-` proposal 主线与最新 full evidence 出现错位

## 4. Key Findings

1. `cand_espd` 在 full-scale `GSM8K` 上给出了当前 workspace 中最强的新实证信号，但它是**温和增益**，不是大幅超越。
2. `ESPD-FixedFrontier` 无法复现 candidate 的质量/速度组合，说明 “frontier 存在” 不够，**frontier 的动态位置选择** 很可能才是关键。
3. `CARD-84` 没有稳定优于 `RAND-84`，再次表明 entropy 更适合作为 observer/routing signal，而非直接 semantic targeting rule。
4. 当前最值得讲的不是 benchmark SOTA，而是 **bounded positive result + attribution discipline**。
5. iteration 4 的 narrative 必须从 object-first 改成 evidence-first；否则写作会继续落后于数据。

## 5. Methodology Gaps

1. 缺少 multi-seed stability。
2. 缺少 plain draft / no-revision baseline。
3. 缺少 reviewer-friendly runtime-lineage artifact，把 raw wall-clock、equal-quality speed、batch、VRAM、extra forwards、auxiliary overhead 放进同一张表。
4. 缺少 routing / stopping 的正交拆分：
   - `entropy frontier + fixed 3 steps`
   - `random frontier + entropy stopping`
5. fixed-frontier sham 仍未匹配 `tokens_changed` 与 `auxiliary_overhead_sec`，不能算完全干净的机制控制。
6. 缺少跨任务验证，尤其是 proposal 已承诺的 `MBPP` / structured-output robustness。

## 6. Competitive Position

- 对整个 `GSM8K` SOTA 而言，当前结果没有 headline 竞争力。
- 对 **training-free discrete diffusion LM test-time intervention / compute-routing** 这一窄赛道而言，当前结果有真实 contribution margin：
  - 邻近工作已经在 2025-2026 年进入 search / guided decoding / memory augmentation 阶段
  - 当前工作提供的是一条更保守但更干净的 speed-line attribution result
- 若后续再补一个外推验证与一份 runtime-lineage artifact，这条线可以从“诚实的小结果”升级为“有明确 contribution margin 的 bounded submission”。

## 7. Hypothesis Update

- `H1 Local Repair Object`: **inconclusive**
- `H2 Entropy-As-Routing`: **supported, but modestly**
- `H3 Benefit-Gating`: **untested**
- `H4 Runtime-Parity Robustness`: **partially supported**
- `H5 Structured-Code Robustness`: **untested**

最重要的心智模型更新是：

> entropy 的主要价值目前不在“决定该怎么修”，而在“决定哪里还值得继续花 compute、哪里可以提前停”。

## 8. Action Plan

### P1. 立刻执行

1. 把 iteration 4 主线重写为 `cand_espd` 当前领跑、`cand_bsr` 为 challenger
2. 生成 reviewer-friendly runtime-lineage artifact
3. 明确 paper-facing claim scope：bounded positive speed-line，不是 mechanism breakthrough

### P2. 下一组实验

1. 做一个关键外推验证：
   - 第二个 benchmark 或第二个 DLM
   - 必须保留 `RAND-84`、`CARD-84`、`ESPD-FixedFrontier`
2. 做一个 routing/stopping 拆分 ablation：
   - `entropy frontier + fixed steps`
   - `random frontier + entropy stopping`

### P3. 保留但降级

1. `cand_bsr` 只保留低成本 continuation option
2. 在 `cand_espd` 外推结果回来前，不开启新的 object-line full-scale

## Final Recommendation

**结论：`PROCEED`, 但必须是 evidence-first、scope-narrowed、methodology-aware 的 PROCEED。**
