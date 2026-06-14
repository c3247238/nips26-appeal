# 反对者视角：别再救 `MGCD / DSG`，把下一轮改成“少做事、更硬对照、系统级加速”

## 检索说明

- 已按角色要求先读 `current/exp/results/pilot_summary.json` 与 `current/supervisor/idea_validation_decision.json`。
- 本次 runtime 没有 Google Scholar MCP；按 shared prompt 的 fallback 规则，改用 `arXiv + Web` 搜索替代，并优先使用 primary-source arXiv 页面。

## 先说结论

我反对继续把 `MGCD` 或 `DSG` 当成还值得追加 GPU 预算的 serious candidate。

当前最硬的内部证据已经足够明确：

- `cand_mgcd` 在 `GSM8K audited slice` 相对 `RAND-84` 的 `net_repaired=-4`，在 `MBPP audited slice` 是 `-2`。
- `cand_dsg` 在 `GSM8K audited slice` 相对 `RAND-84` 只有 `0`，在 `MBPP audited slice` 是 `-1`。
- `current/supervisor/idea_validation_decision.json` 已明确给出 `PIVOT`，并要求把 `cand_mgcd` 从 front-runner 降级、围绕 `repair object` 重新组织候选池。

所以这一轮真正该反对的，不是某个具体 trick，而是整个默认叙事：

> “observer signal 已经有信息量，所以只要再加更复杂的 controller / memory / draft-consensus，就应当自然穿透 sham control。”

我认为这句话现在最可能是错的。

更可能成立的反命题是：

> 在 training-free DLM 上，下一轮有效增益也许来自 **更少的 intervention**、**更粗粒度的 repair object / commitment policy**、以及 **更系统级的缓存与调度优化**，而不是更重的局部控制器。

## 我反对的三个主流假设

### 1. 假设一：`uncertainty / entropy` 信号一旦存在，更强 stateful controller 就应该能赢

我不认同。

内部证据已经在反着说话：

- `iter_004/exp/results/summary.md` 明确写了：`entropy / calibration signal 可以作为 observer`，但 `token-wise revision` 仍未证明能稳定转化为 `controller gain`。
- 本轮 screening 里，`MGCD` 比 `DSG` 更复杂，却没有穿透 `RAND-84`；这更像 “hidden compute + richer story”，不像 “causal controller”。

外部反证也一致：

- Zhu et al. 发现很多 calibration method 对 failure prediction 反而“无用甚至有害”，问题在于它们恶化了 correct/incorrect 之间的 confidence separation，而不是自动提升 downstream trust decision（[Rethinking Confidence Calibration for Failure Prediction, 2023](https://arxiv.org/abs/2303.02970)）。
- Tang et al. 指出离散 diffusion sampler 的 headline metric 改善并不自动等于 correct sampling；即使在 oracle denoiser 下，few-step sampler 也可能不正确（[Is Your Diffusion Sampler Actually Correct?, 2026](https://arxiv.org/abs/2602.19619)）。

### 我的反方向

把 `BSR` 升级为真正的一号候选，但要换 framing：

- 不是 `MGCD-lite` 的轻量附属物。
- 而是一个独立的 **minimal object-level falsification candidate**。
- 核心问题不再是 “memory graph 是否更强”，而是 “把 repair object 从 token 提升到 span / boundary，是否已经足够”。

这条线更 contrarian，因为它直接押注：

> 之前失败的不是 state 不够复杂，而是我们一开始就选错了 intervention object。

### 成本与成功率

- Pilot 计算成本：低到中。延续当前 `84-step matched-compute` 设定即可，单卡就能跑完 audited slice；工程复杂度明显低于 `MGCD`。
- 若引入辅助器，必须限制在 `BERT-base` / `Qwen-0.5B` 级别，避免再次把 attribution 污染成“大模型偷偷救场”。
- 我给它的成功率：`0.35`。

不高，但高于继续救 `MGCD`。

---

### 2. 假设二：token-wise revision 就是 training-free DLM 的正确干预粒度

我也不认同。

当前 workspace 自己已经在提醒我们：`token-wise revision` 这条线没有被验证。

更关键的是，近期文献的失败模式和这个判断高度一致：

- Shu et al. 认为 block-based diffusion 的核心问题是 **Boundary-Induced Context Truncation**，也就是 token 在边界附近过早 commit；他们的 training-free `Deferred Commitment Decoding` 不是去做更复杂的 token scoring，而是让高不确定 token 晚一点再定（[Deferred Commitment Decoding, 2026](https://arxiv.org/abs/2601.02076)）。
- Wang et al. 进一步指出 block diffusion 会出现 irreversible commitment 导致的 stagnation，需要可逆 backtracking 才能恢复（[Reversible Diffusion Decoding, 2026](https://arxiv.org/abs/2602.00150)）。
- 社区讨论里也反复出现同一担忧：diffusion 看起来像“能回头修错”，但实际上并没有显式 error-correction mechanism；如果 reverse path 偏离分布，单纯多做几步并不等于能纠正它（[LocalLLaMA discussion on Block Diffusion](https://www.reddit.com/r/LocalLLaMA/comments/1ja5pf9/block_diffusion_interpolating_between/)）。

### 我的反方向

保留一个新的 serious candidate，但不要再沿用 `MGCD / DSG` 的语言，而是单独立题为：

`DCD-lite` 或 `Late-Commit Revision`

核心原则：

- 不再为每个 token 设计更聪明的 controller。
- 改成对 **span / block 的 commit 时机** 动手。
- 低不确定区域尽早冻结。
- 高不确定边界延后提交，必要时允许一次轻量 backtrack。

这条线的反常识之处在于：

> 也许 DLM 的问题不是“不会选要改哪个 token”，而是“改得太早、锁得太早”。

### 成本与成功率

- Pilot 计算成本：中。比 `BSR` 稍高，但仍明显低于再做 `MGCD` 的多轨 draft / memory machinery。
- 如果实现得克制，可以直接沿用当前 runtime contract，不需要新的训练。
- 我给它的成功率：`0.40`，但目标要写窄：
  - `0.40` 概率拿到 **相近 wall-clock 下的小幅质量增益**；
  - `0.25` 概率同时拿到 **质量不降 + NFE/latency 改善**。

---

### 3. 假设三：training-free speed improvement 的主战场应该是更聪明的 revision heuristic

这是我最反对的一条。

如果用户目标里 “提升推理速度” 与 “提升性能” 二选一都可以成立，那么最 contrarian 的判断其实是：

> DLM 的下一波 training-free speed gain，可能主要来自 cache / systems / schedule reuse，而不是新的 repair controller。

外部证据非常强：

- Peng et al. 明确指出：在实践中 open-source DLM 往往仍然比 AR 慢，而且像 dual cache、parallel decoding 这样的加速技巧在大 batch 下收益会缩小；他们甚至把现有效率评测方式本身都当成问题对象（[How Efficient Are Diffusion Language Models?, 2025](https://arxiv.org/abs/2510.18480)）。
- Ma et al. 的 `dKV-Cache` 直接把 bottleneck 定义为 “DLM 无法像 AR 那样用 KV cache”，并报告 training-free 的 `2-10x` 推理加速（[dKV-Cache, 2025](https://arxiv.org/abs/2505.15781)）。
- Hu et al. 的 `FlashDLM` 也不是靠更复杂的 token repair，而是 `FreeCache + lightweight AR guidance`，报告平均 `12.14x` end-to-end speedup（[FlashDLM, 2025](https://arxiv.org/abs/2505.21467)）。

这几篇 paper 的共同点很重要：

- 它们都把问题重心放在 **execution envelope**。
- 而不是假设“只要 controller 更聪明，速度/质量就会一起变好”。

### 我的反方向

把 “速度线” 单独升格成 serious candidate，而不是继续附属于 `MGCD / DSG / BSR`：

`Cache-First Speed Lane`

核心原则：

- 第一优先级：最大 batch size + cache reuse + compile / attention backend。
- 第二优先级：若必须加入 guidance，只允许 `Qwen-0.5B` 级别的轻量 side model。
- 第三优先级：把指标从 `NFE` 改成 `wall-clock latency / tokens-per-second / peak VRAM / hidden auxiliary FLOPs`。

这条线背后的反命题是：

> 对 DLM 来说，最快的 training-free improvement 也许不是“更会修”，而是“更少重算”。

### 成本与成功率

- Pilot 计算成本：低到中，主要是工程集成与 profiling，而不是长时间 benchmark。
- 最适合当前项目约束，因为用户已经明确要求最大 batch size、flash attention、compile、多卡并行优先。
- 我给它的成功率：
  - 若目标只是 **显著提速**：`0.60`
  - 若目标是 **提速且几乎不掉点**：`0.30`

## 我建议保留的 2-3 个 serious candidates

### 1. `BSR++`：主候选

- 角色：新的 object-level primary candidate
- 原因：它最直接回答 “是不是一开始就选错了 repair object”
- 允许的主张上限：只有当它超过 `RAND-84`，才配继续活下去

### 2. `DCD-lite / Late-Commit Revision`：第二候选

- 角色：quality/speed bridge candidate
- 原因：它挑战 token-wise intervention 这个更深的默认假设
- 允许的主张上限：只准写成 commitment policy，不准重新包装成“大而全 controller”

### 3. `Cache-First Speed Lane`：独立速度候选

- 角色：若用户更看重 speed，就把它与 `BSR++` 并列，而不是附属
- 原因：它最贴合外部文献目前已经显著奏效的 training-free 路线
- 允许的主张上限：必须用 wall-clock 说话，不能只报 step reduction

## 我建议明确降级或删除的方向

- `MGCD`：降级为 **已筛掉候选**，只允许作为失败先例或对照，不再作为 front-runner。
- `DSG`：降级为 **已筛掉候选**。它“更轻但仍不分离”，这不是 positive signal。
- `LMG`：继续 deferred。training-free 证据没站稳前，不要把训练补丁拉进来搅浑 attribution。

## 我对下一轮 pilot gate 的底线要求

1. 新候选在任何 full benchmark 之前，必须先穿透 `RAND-84`。
2. 速度线必须报告 `wall-clock`，不能只报 `NFE`。
3. 所有 auxiliary compute 都必须显式入账，尤其是 draft、cache refresh、small verifier、AR guide。
4. 若再使用 side model，只允许 `BERT-base` / `Qwen-0.5B` 这类小模型。
5. 下一轮文本里不允许再出现 `MGCD front-runner` 话术。

## 最终态度

我支持继续做，但前提不是“把 `MGCD` 改得更聪明”，而是把研究主语改掉：

> 从“如何把 observer signal 变成更强 controller”  
> 改成  
> “training-free DLM 到底需要更少的 intervention、更晚的 commitment，还是更强的 cache/system support？”

这才是一个真正经过 `PIVOT` 之后应有的 contrarian candidate pool。

## 参考来源

### 内部证据

- `current/exp/results/pilot_summary.json`
- `current/supervisor/idea_validation_decision.json`
- `current/exp/results/summary.md`

### 外部证据

- Tang et al., 2026, *Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models*  
  https://arxiv.org/abs/2602.19619
- Zhu et al., 2023, *Rethinking Confidence Calibration for Failure Prediction*  
  https://arxiv.org/abs/2303.02970
- Shu et al., 2026, *Deferred Commitment Decoding for Diffusion Language Models*  
  https://arxiv.org/abs/2601.02076
- Wang et al., 2026, *Reversible Diffusion Decoding for Diffusion Language Models*  
  https://arxiv.org/abs/2602.00150
- Peng et al., 2025, *How Efficient Are Diffusion Language Models? A Critical Examination of Efficiency Evaluation Practices*  
  https://arxiv.org/abs/2510.18480
- Ma et al., 2025, *dKV-Cache: The Cache for Diffusion Language Models*  
  https://arxiv.org/abs/2505.15781
- Hu et al., 2025, *FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion*  
  https://arxiv.org/abs/2505.21467
- LocalLLaMA discussion on Block Diffusion  
  https://www.reddit.com/r/LocalLLaMA/comments/1ja5pf9/block_diffusion_interpolating_between/
