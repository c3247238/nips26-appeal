# theoretical on seed

## 总判断

从 theoretical 视角看，`proposal_seed_round2.md` 已经明显比前两轮更稳，最大的优点是：

- 不再试图把 paper 写回 generic method paper
- 已经把 `observer/controller split` 压成低承诺命题
- 已经明确 bucket 与 runtime fairness 的支撑分工

但最后一轮仍需要再做一次“降满度”处理。  
原因很简单：当前最强的资产仍然是 **机制证据 + protocol discipline**，而不是一个可广泛推广的一般理论。因此最终版本必须继续强调：

- **这是 scoped diagnostic claim**
- **不是一般性理论定理**
- **不是新的 standard-setting manifesto**

---

## 1) 主命题是否足够可证伪且不过满

### 结论

**基本可证伪，但还可以再收紧半档。**

当前主命题是：

> 在当前 tested policies 与 current evidence 下，`observer quality != controller gain`；因此 DLM revision 不应只按 aggregate gain 报告，而应同时报告 observer/controller split、bucket-level outcomes 与 realized compute fairness。

这比前面版本已经好很多，因为它具备三点：

1. 有明确适用范围：
   - `current tested policies`
   - `current evidence`
2. 有可证伪对象：
   - 如果 observer quality 与 controller gain 在现有设置下高度一致，这个命题就站不住
3. 有对应的验证资产：
   - bucket audit
   - signal / observer-controller protocol
   - runtime fairness matrix

### 仍建议再降一档的地方

问题出在主命题的后半句：

> “因此 DLM revision 不应只按 aggregate gain 报告……”

这句话容易从经验判断滑向 **规范性普遍要求**。  
目前资产足以支持：

- “在我们的设置里，只报 aggregate gain 会隐藏关键差异”

但还不完全足以支持：

- “DLM revision 一般都必须这样报”

### 更稳的推荐写法

建议改成：

> 在当前 tested policies 与 evidence scope 下，aggregate gain 不足以解释 revision 的真实效应结构；observer/controller split、bucket-level outcomes 与 realized compute fairness 提供了更完整的诊断视角。

这个版本更稳，因为它把“必须”降成了“在当前证据下更充分的解释方式”。

---

## 2) observer/controller split 应留在 framing 还是 contribution bullet

### 结论

**应保留在 framing，并以弱形式进入 contribution bullet，但不能作为独立 hero contribution。**

### 为什么不能只留在 framing

如果完全只留在 framing，风险是：

- 它会变成 intro 里的语言资源
- reviewer 会觉得这只是叙事组织，不是研究产出

而实际上，它确实承载了一个真正有价值的对象区分：

- diagnostic quality
- realized intervention gain

这个区分不应被完全降成写作修辞。

### 为什么又不能升成独立 hero bullet

因为当前证据还不够支持把它写成：

- “我们提出了一个新的理论框架”
- “我们建立了 observer/controller 的一般分解原理”

这会明显 evidence outrun。

### 最稳的放置方式

我建议：

- 在 **framing / intro / problem statement** 中把它作为主问题定义
- 在 **contribution bullet** 中只保留一个弱版本，比如：

> We show that, in our tested setup, observer quality and controller gain should be evaluated separately rather than inferred from aggregate revision gains alone.

对应中文理解就是：

- 这是一个 **paper-specific falsification-style finding**
- 不是通用理论口号

所以答案不是“只放 framing”或“强行放贡献第一条”，而是：

> **以 framing 为主，贡献列表中保留一个低承诺版本。**

---

## 3) 哪些 claim 仍需降调

下面几类 claim 仍然建议继续降调。

### Claim A：把 observer/controller split 写成普适规范

仍需避免的写法：

- DLM revision 必须统一采用 observer/controller split
- 我们建立了 revision 的标准评价框架

建议保留的写法：

- 在当前研究对象下，这一区分解释了为何 aggregate gain 不够
- 我们的证据支持将二者分开报告

### Claim B：把 bucket 写成完整机制 taxonomy

仍需避免的写法：

- 我们建立了完整 failure taxonomy
- 我们系统揭示了 revision 的所有失效机制

建议保留的写法：

- 我们通过 `fixed / harmed / no-effect` 解释 aggregate gain 的主要来源结构
- 我们观察到 reasoning / code 边界上的 recoverability 差异

### Claim C：把 runtime-lineage protocol 写成独立 benchmark standard

仍需避免的写法：

- 我们提出了新的 DLM inference fairness standard
- 我们解决了 inference comparison 的可比性问题

建议保留的写法：

- 我们提供了 reviewer-auditable runtime-lineage artifact，以减少 implementation confound
- 我们展示 realized compute fairness 会改变结论解释

### Claim D：把 seed spot-check 写成稳定性证明

如果 `seed_sensitivity_spotcheck` 最终进入 proposal deliverables，也必须降调。

仍需避免：

- 我们证明 headline ordering 稳定

建议保留：

- 我们做最小稳健性 spot-check，检查 headline delta 的方向一致性

### Claim E：abstract 里 headline claim 数量过多

当前最危险的是把 abstract 写成三条并列大 claim：

1. observer/controller split
2. bucket mechanism
3. runtime fairness protocol

如果三条都写得很重，会显得每一条都没被充分打透。  
我建议 abstract 最多保留 **2 个 headline claim**，第三条只作为 supporting phrase 出现。

---

## 4) 最终推荐结构

### 我建议的最终结构是：一条主命题，两层支撑

#### A. 主命题层

**主命题：aggregate revision gain 不足以解释真实效应结构。**

更具体地说：

- observer quality 与 controller gain 不应被默认等同

这就是整篇 paper 的问题意识与 framing 中心。

#### B. 主证据层

**Benefit-Bucket / Recoverability Analysis**

这是最核心的实证支撑，因为它直接回答：

- 修了谁
- 伤了谁
- 对谁无效

没有这一层，主命题会显得抽象。

#### C. 可信度层

**Runtime-Lineage / Honest-Compute Protocol**

这是保证结论不被 implementation confound 冲掉的必要条件。  
它应该紧贴主证据层出现，而不是独立悬浮成一条“我们很规范”的平行贡献。

### 对 contribution bullets 的推荐排列

我建议最终只写成 **2+1** 结构：

1. **主 finding**
   - 在 tested setup 下，aggregate gain 不能充分代表 revision 的真实效应结构，observer quality 与 controller gain 需要分开解释
2. **主 evidence**
   - 通过 benefit-bucket / recoverability analysis，展示 gain 来自哪些 fixed / harmed / no-effect 样本结构
3. **supporting protocol**
   - 提供 runtime-lineage / honest-compute artifacts，确保上述结论在 realized compute 下可审计

这里第三条必须明显弱于前两条。

### abstract 的推荐结构

我建议 abstract 只保留两条 headline：

1. **主发现**：
   - aggregate gain 不足以解释 revision 效应
2. **主证据**：
   - bucket-level analysis 揭示 fixed / harmed / no-effect 结构

然后用一句 support 句带过：

- runtime-lineage artifacts ensure compute-normalized interpretation

不要让 abstract 看起来像三篇小 paper 拼在一起。

---

## 对 proposal seed 的最终修改建议

### 建议保留

- serious execution candidates 只保留 2 个：
  1. `Benefit-Bucket / Recoverability Analysis`
  2. `Runtime-Lineage / Honest-Compute Protocol`
- `Observer-Controller Split` 不作为独立执行 lane

### 建议微调

- 把主命题里的 “不应只按 aggregate gain 报告” 改成
  - “aggregate gain 不足以解释”
  - 或 “aggregate gain alone can be misleading under current evidence”

### 建议明确写死

- `Observer-Controller Split`
  - 是 **problem framing + weak contribution**
  - 不是独立 execution candidate
  - 不是一般性理论框架

---

## 一句话结论

当前 seed 已经接近可用；最后需要做的不是换方向，而是继续降调：  
**把 observer/controller split 保留为 framing 主轴和弱贡献，把 bucket 作为主证据，把 runtime-lineage 作为必要护城河，这样最不容易 evidence outrun。**
