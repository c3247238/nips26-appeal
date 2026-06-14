# Revisionist

## 1. Hypothesis Audit

### H1: Local Repair Object Hypothesis

**结论：`inconclusive`。**

- H1 的主语是 `cand_bsr`
- 本轮 full-scale 实际测试的是 `cand_espd`
- 因此最新数据不能支持“repair object 已被证明选对了”，也不能继续把 `cand_bsr` 当成已被结果背书的 front-runner

### H2: Entropy-As-Routing Hypothesis

**结论：`supported`, 但只是温和支持。**

- `cand_espd=0.4041`
- `ESPD-FixedFrontier=0.3988`
- `speed_gain_vs_fixed_frontier=+18.69 tok/s`
- gate `routing_story_survives_sham_control=true`

这组结果支持 entropy 更像 **routing / stopping signal**，而不是 semantic controller。

### H3: Benefit-Gating Hypothesis

**结论：`untested`。**

- 本轮没有 `cand_ugr` 新结果
- 不能继续在 headline 中占主位

### H4: Runtime-Parity Robustness Hypothesis

**结论：`partially supported`。**

- 主要比较已在统一 contract 下完成
- 但仍存在 batch 差异与 compile-off caveat
- 这只能支撑“当前 contract 下成立”，不能支撑“最优工程实现下也成立”

### H5: Structured-Code Robustness Hypothesis

**结论：`untested`。**

- 当前只有 `GSM8K`
- 没有 `MBPP` / syntax-guard 新证据

## 2. Surprise Analysis

- 真正跑出正信号的不是 proposal 中的 quality front-runner `cand_bsr`，而是 speed-line `cand_espd`。
- `ESPD-FixedFrontier` 不仅质量更低，速度也明显更慢，这比“just a small sham gap”更强。
- `RAND-84` 与 `CARD-84` 几乎打平，且 `RAND-84` 略优，再次削弱了“entropy 直接决定 semantic revision object”这条旧叙事。
- `cand_espd` 的 gain 不是靠显著降低 NFE 得到的，更像是在近似 compute 带内更聪明地花 compute。

## 3. Mental-Model Update

我们需要把旧心智模型：

> entropy 是 revision controller 的直接语义信号

改写成：

> entropy 更像 risk detector / compute router。它未必可靠地告诉我们“该怎么修”，但它可能可靠地告诉我们“哪里还值得继续花 compute，哪里可以提前停”。

更具体地说：

1. `observer != controller` 仍必须分开  
2. routing 比 semantic targeting 更接近当前真实增益来源  
3. 当前更像 speed/compute allocation 问题，而不是 object-level semantics 已经解决的问题  
4. object-level proposal 仍有理论吸引力，但证据上尚未接管主线

## 4. Reframing Proposals

### Reframing A

从“对象线主、速度线辅”改成“速度线已证、对象线待证”。

- `cand_espd` 从 backup 升级为当前唯一被 full-scale 结果直接支持的 serious line
- `cand_bsr` 从 front-runner 下调为 next mechanism challenger

### Reframing B

把研究问题改成：

> 如何把 entropy 从 weak controller 改造成 useful router

这比“entropy 到底是不是对的语义信号”更贴近数据。

### Reframing C

把 fixed-frontier sham 写成核心机制证据，而不是附属 sanity check。当前最有价值的不是 `+0.61pp`，而是：same frontier ratio、same retained-step family、same contract 下，fixed-frontier 仍复现不了 candidate。

## 5. New Hypotheses

- `NH1`: entropy 的主要价值不是决定“修哪里”，而是决定“哪里还值得继续算”
- `NH2`: `cand_espd` 的优势主要来自 dynamic frontier placement 与 early stopping 的联合作用
- `NH3`: 当前 `cand_espd` 的研究上限可能被 auxiliary overhead 卡住；若大幅压低 frontier bookkeeping，speed-line 可能更强
- `NH4`: object-level line 未来必须证明的不只是“比 random 好一点”，而是“值得压过 routing line 的预算优先级”
- `NH5`: `RAND-84 ≈ CARD-84` 暗示 semantic targeting 在当前 DLM regime 里可能天然偏弱

## 6. Pivot vs Iterate Recommendation

**建议：`ITERATE`, 但要对 proposal 主线做一次明确的内部 pivot。**

- 不需要再做大 pivot，因为 `cand_espd` 已给出 full-scale 正信号
- 但也不能继续沿用当前 proposal 的排序，因为最新数据并不支持 `cand_bsr` 仍是默认主角

### 具体决策

1. 短期主线改为 `cand_espd`
2. `cand_bsr` 保留，但降为 challenger
3. result debate 的中心问题改成：
   - `cand_espd` 是否足够强，值得进入 paper-level speed-line claim？
   - object-level line 需要拿出什么级别的证据，才有资格重新夺回主线？

最终最诚实的结论是：

> 当前不应再维持 object-first narrative，而应改成 evidence-first narrative：先承认 `cand_espd` 是 iteration 4 唯一被 full-scale 结果直接支持的 serious candidate，再决定 object-level line 如何重新证明自己。
