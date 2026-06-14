# 方法论审计报告 — 第 4 轮迭代 (BSD + RACFG)

**角色**: Methodologist — 审查实验方法论的内部效度、外部效度、评估协议合理性和可重现性。

---

## 总体评估

**严重程度**: 高
**核心问题**: 所有第 4 轮方法（BSD、RACFG、A-CFG、组合）的"full-scale"评估实际上仍停留在 **PILOT 规模（n=16, 单 seed）**，未完成 task_plan.json 中规定的 Countdown-500 × 3 seeds 全规模验证。所有声称的方法比较（verdict: GO/NO-GO）均基于极端小样本，统计功效极低。

---

## 1. 基线公平性审计

### 1.1 基线选择 — 部分合理

**合理之处**:
- Vanilla Dream-7B 是自然基线
- DMI（iter_003 验证过的最佳方法，9.3% Countdown-500）作为内部强基线
- ReMDM-conf 和 RCR 作为 remasking 类基线

**严重问题**:

**(a) 基线结果跨迭代不一致**
- iter_003 的 Countdown-500: vanilla=4.7%, DMI=9.3%（16 个 pilot prompts 中 vanilla=0%, DMI=12.5%）
- iter_004 的 Countdown-16: vanilla=0.0%, DMI=12.5%（某些评估中 DMI=0.0%）
- 不同评估运行中，**同一方法在相同 16 样本上结果波动剧烈**（如 compute_fair 中 DMI=0.0%，而 fullscale_bsd 中 DMI=12.5%）。这暗示实验可能未严格控制随机种子，或者评估管线存在非确定性因素。

**(b) 基线未充分调优**
- A-CFG 是从 arXiv 2505.20199 复现的，但仅测试了 w∈{1.0, 1.5, 2.0} 三个权重值
- 原论文在 LLaDA-8B 上报告 GSM8K 73.5%，但本实验在 Dream-7B 上 GSM8K-16 仅 37.5%（最佳 A-CFG），未讨论这一巨大差距是模型差异还是实现差异
- DMI 的 alpha=0.3 是 iter_003 的最佳值，但在 iter_004 的不同 prompt set 上未重新验证

**(c) 外部基线缺失**
- MetaState、CORE、Self-Rewarding SMC 均在 proposal 中列为竞争基线，但实验中**完全没有实现或比较**
- 这使得论文的竞争力定位（"substantially outperforming existing remasking baselines"）无法得到实验支撑

### 1.2 结论

**基线公平性评分: 4/10** — 基线选择方向正确，但跨运行不一致性暴露了实验控制问题，且关键外部基线完全缺失。

---

## 2. 评估指标适当性

### 2.1 正面变化

iter_004 从 PPL（困惑度）转向 **benchmark accuracy** 作为主指标，这是对 iter_003 经验教训的正确应用。iter_003 的 full_results.json 显示 PPL 改善（pilot 阶段 -16% 到 -25%）在 full-scale 验证中完全消失（p>0.25），证实 PPL 不是可靠的推理质量指标。

### 2.2 仍存在的问题

**(a) Countdown 准确率作为主指标的局限性**
- 在 n=16 下，准确率的量化精度为 6.25%（1/16），所有方法的准确率都是 0%, 6.25%, 12.5% 之一
- 这种粗粒度无法区分方法间的真实差异
- 正确/错误的二分法丢失了"接近正确"的信息（如模型是否至少找到了正确的运算组合但答案错误）

**(b) 辅助指标未被充分利用**
- rep-2/3, distinct-1/2/3 在所有评估中都在正常范围（无退化），但这些指标的方差极小，基本无区分力
- 信念熵轨迹（entropy trajectory）是有意义的分析维度，但目前仅作为 diagnostic 而非比较指标

**(c) GSM8K 评估不充分**
- 仅 16 样本的 pilot，未进行 full-scale 1319 样本评估
- Bootstrap 95% CI 宽达 50pp（[-12.5%, +37.5%]），完全无法得出结论

**(d) MBPP 完全未评估**
- proposal 和 methodology 均将 MBPP 列为评估基准之一，但 iter_004 未对任何新方法运行 MBPP

### 2.3 结论

**指标适当性评分: 6/10** — 从 PPL 转向 accuracy 是重要进步，但 n=16 下 accuracy 的量化精度太低，且缺少部分预定基准。

---

## 3. 评估协议审计

### 3.1 数据泄露与训练/测试污染

**未检测到数据泄露风险**。所有方法均为 training-free（无参数训练，DTA 在 iter_004 已被放弃），Countdown 问题是动态生成的，GSM8K 使用标准 test split。

### 3.2 超参数选择偏差 — 严重

**(a) Pilot-to-full 的超参数过拟合风险**
- 所有消融实验（BSD k, alpha schedule, RACFG remask_pct, schedule type）均在 n=16 pilot 上完成
- 最佳配置（BSD: k=0.75, alpha=linear(0.1→0.8); A-CFG: fixed w=1.5）是从 16 个样本中选出的
- **iter_003 已经证明了 pilot-to-full 过拟合的严重性**: entropy_r20_mean 在 pilot 中 PPL -24.9%，full-scale 中 -0.5%；tcr_r30_s32 pilot PPL -22.9%，full-scale -2.8%。这些 24pp 的 pilot-to-full 差距是方法论上的警示信号。
- iter_004 的 BSD 和 A-CFG 的消融决策直接复制了这一有缺陷的模式：在 n=16 上选择超参数，然后声称"GO"

**(b) 消融实验的区分力不足**
- BSD alpha schedule 消融：linear, cosine, constant(0.3) 全部得到 6.2% accuracy，constant(0.5) 也是 6.2%——四个配置结果完全相同
- 这意味着"最佳配置"的选择实际上是**随机的**（从四个相同结果中任选一个），不具备方法论意义

**(c) 评估集在 pilot 阶段的不稳定性**
- fullscale_bsd 中 vanilla=0.0%，但 bsd_racfg_combo 中 vanilla=6.2%
- fullscale_bsd 中 DMI=12.5%，但 compute_fair 中 DMI=0.0%
- 这些波动表明 16 样本评估集在不同运行间的随机性极大，或者评估管线存在非确定性（如生成时未固定 seed、评估 prompt set 不同）

### 3.3 计算量公平比较 — 设计合理但执行不充分

Compute-fair 比较的设计思路正确（BSD ~1.1x vs vanilla 1.1x steps, RACFG ~2x vs vanilla 2x steps），但：
- n=16 下的 Pareto 分析结果显示 vanilla 在所有计算预算下都是 Pareto 最优，这与 BSD/A-CFG 的预期效果完全矛盾
- 如果这一结论在 full-scale 中成立，则论文的核心贡献不复存在

### 3.4 Decision Gates 执行审计

Methodology 定义了三个 decision gates，但执行中存在问题：

| Gate | 条件 | 实际结果 | 执行是否合理 |
|------|------|---------|-------------|
| Gate 1 | A-CFG < vanilla → switch to LLaDA-8B | A-CFG (w=1.5) = 12.5% > vanilla 0% → GO | 合理，但基于 2/16 vs 0/16 |
| Gate 1 | BSD OOD collapse → activate fallback | 无崩溃 → GO | 合理 |
| Gate 2 | 若 BSD 和 RACFG 均 < vanilla+3pp → pivot | BSD 6.2% > 0% vanilla，但差距仅 1 样本 | **应触发 pivot，但被忽略** |
| Gate 3 | Combo < max(BSD, RACFG) → report negative | Combo 6.2% < max 12.5% → **NO-GO 正确识别** | 合理 |

**关键问题**: Gate 2 的"vanilla + 3pp"条件在 n=16 下不可靠（1 个样本的差异 = 6.25pp），但 RACFG 全线 0% 的结果本应更早触发 pivot。实际上 RACFG 已在 remask_pct ablation 后 pivot 为 A-CFG，这一决策是正确的，但 pivot 后的 A-CFG 仍被标记为"RACFG"，导致术语混淆。

### 3.5 结论

**评估协议评分: 3/10** — 无数据泄露，但超参数过拟合风险极高（iter_003 已实证），同一基线在不同运行间结果不一致，decision gates 基于统计功效不足的数据做出。

---

## 4. 消融完整性

### 4.1 已完成的消融

| 消融 | 状态 | 样本量 | 区分力 |
|------|------|--------|--------|
| BSD k ∈ {T/4, T/2, 3T/4} | 完成 | 16 | **无**（0%, 0%, 6.2%，无法区分 T/2 和 T/4） |
| BSD alpha schedule (4 configs) | 完成 | 16 | **无**（四个配置全部 6.2%） |
| RACFG remask_pct (3 × 2 configs) | 完成 | 16 | **有**（全部 0%，清晰否定） |
| RACFG schedule (4 × 2 configs) | 完成 | 16 | **有**（仅 fixed 有效，scheduled 全部 0%） |
| RACFG vs A-CFG | 完成 | 16 | **有**（0% vs 12.5%，方向清晰） |

### 4.2 未完成但在 methodology 中规定的消融

- **BSD tau schedule 消融**: 未单独测试
- **RACFG guidance weight (w_base) 独立消融**: 仅在 schedule ablation 中附带测试
- **组合参数消融（BSD k × A-CFG w）**: 仅测试了一个配置

### 4.3 消融的方法论问题

- 消融的串行依赖结构（k ablation → alpha ablation → fullscale）意味着早期消融的错误会传递到后续实验
- 在 n=16 下，BSD k ablation 中 k=0.75 的"最佳"是基于 1/16 vs 0/16 的差异（p=1.0），完全不可靠
- 如果 full-scale 中 k=T/2 实际更好，整个后续管线（alpha ablation、fullscale_bsd、combination）都基于了错误的 k 值

### 4.4 结论

**消融完整性评分: 5/10** — 消融覆盖面足够，但样本量导致绝大多数消融无区分力，且串行依赖放大了早期选择的不确定性。

---

## 5. 可重现性评估

### 5.1 正面因素

- 代码模块化清晰（`bsd_racfg/bsd.py`, `racfg.py`, `eval_harness.py`）
- 超参数完整记录（每个实验的 JSON 结果文件包含完整配置）
- 硬件信息详细（NVIDIA RTX PRO 6000 Blackwell, 98GB VRAM）
- 依赖版本锁定（PyTorch 2.10.0+cu128, Transformers 4.57.6）

### 5.2 可重现性风险

**(a) 随机种子控制不确定**
- 前述同一方法在不同评估运行中结果不一致（DMI 在不同 summary 中 accuracy 从 0% 到 12.5% 波动）
- 可能原因：生成时的非确定性（CUDA 非确定性操作、不同的 GPU 分配）
- Methodology 规定 seed=42 for pilot，但实际实验是否严格执行有疑问

**(b) Countdown 问题集不固定**
- 不同 task 的 pilot 使用的 16 个 Countdown 问题可能不同（"16 prompts"来源未统一）
- iter_003 pilot 使用"简单科学问题"，iter_004 使用"自生成 Countdown 问题"——评估集不同使跨迭代比较无效

**(c) 评估方法可能存在差异**
- iter_003 使用 GPT-2 PPL 作为指标，iter_004 使用 accuracy——同一方法的跨迭代比较需要在相同指标下进行
- 准确率的评估标准（精确匹配？容差？）未在 methodology 中明确定义

### 5.3 结论

**可重现性评分: 6/10** — 代码和配置记录良好，但随机种子控制和评估集一致性存在疑问，跨运行结果波动暗示重现性风险。

---

## 6. 具体建议：加强实验方法论

### 优先级 1: 必须在论文提交前完成

1. **完成 full-scale 评估（Countdown-500 × 3 seeds）**
   这是最关键的缺失。n=16 的所有 verdict（GO/NO-GO）都不可靠。iter_003 已证明 pilot-to-full 差距可达 24pp。必须对 BSD (best config)、A-CFG (w=1.5)、DMI、vanilla 完成 Countdown-500 × seeds {42, 123, 456} 评估。

2. **固定评估 prompt set**
   创建一个标准化的 Countdown-500 问题集并版本控制。所有方法必须在完全相同的问题上评估。消除跨运行结果不一致的根源。

3. **GSM8K full-scale 评估**
   至少对最佳方法和 vanilla 完成 GSM8K-1319 × 1 seed。A-CFG 在 pilot 中 GSM8K-16 表现最佳（37.5%），这需要在 full-scale 中验证。

### 优先级 2: 显著提升论文质量

4. **bootstrap 重采样的 full-scale 统计分析**
   在 n=500 × 3 seeds 数据上运行 McNemar + Bonferroni + Bootstrap 95% CI。目前的统计分析框架已搭建好（statistical_tests task 已验证），只需要数据。

5. **compute-fair 比较的 full-scale 重做**
   pilot 中 vanilla 在所有计算预算下 Pareto 最优是一个危险信号。如果 full-scale 仍然如此，论文需要诚实报告。

6. **添加至少一个外部基线**
   CORE 或 HEX 的复现（或至少引用其论文数字进行间接比较）。MetaState 需要训练，不切实际，但 CORE 是 training-free 的，应该可以复现。

### 优先级 3: 增强严谨性

7. **消融在 n=100 上重做**
   BSD k 和 alpha 消融在 n=16 上完全无区分力。使用 Countdown-100（单 seed）重做关键消融，至少获得 1pp 量化精度。

8. **报告 pilot-to-full 校准比**
   参照 iter_003 的经验，计算 pilot-to-full 效果保持率。如果 iter_003 的保持率约为 0%（pilot 改善在 full-scale 消失），那么 iter_004 的 pilot 结果需要加上明确的不确定性折扣。

9. **RACFG 失败的理论分析**
   JSD stability 在 Dream-7B 上无效（stability ~0.997）这一发现本身有方法论价值。建议增加对 Dream-7B 去噪过程中 logit 分布动态的定量分析（如不同步数的平均 JSD），解释为什么 cross-step stability 在这一模型上不适用。

---

## 7. 关键假设检验状态总结

| 假设 | 预测 | 当前状态 | 方法论信心 |
|------|------|---------|-----------|
| H1: BSD ≥ 14% Countdown-500 | BSD 显著优于 DMI | **未验证**（仅 pilot 6.2%, n=16） | 极低 |
| H2: RACFG ≥ 15% Countdown-500 | RACFG 显著优于 vanilla | **否定**（JSD 信号在 Dream-7B 无效，pivot 为 A-CFG） | 高（清晰否定） |
| H3: BSD+RACFG ≥ 18% | 组合产生协同增益 | **否定**（combo 6.2% < max 12.5%，NO-GO） | 中（n=16 但方向清晰） |
| H4: Cross-step JSD > single-step confidence | JSD 优于 confidence re-masking | **否定**（0% vs 12.5%，H5 falsified） | 高（清晰否定） |
| H5: CFG temporal scheduling > fixed | 调度优于固定权重 | **否定**（fixed 12.5%, scheduled 全 0%） | 中（n=16 但一致） |
| H6: BSD entropy monotonically decreasing | 信念向量熵单调递减 | **支持**（15/16 monotonic, rho=-0.95） | 中（有意义但非核心） |

**方法论结论**: 6 个核心假设中，4 个被否定（H2-H5），1 个未验证（H1），1 个被支持但非核心（H6）。能否产出正面论文完全取决于 BSD 和 A-CFG 在 full-scale 评估中的表现。

---

## 8. 跨迭代方法论反思

iter_004 面临的方法论困境与 iter_003 高度一致：

1. **Pilot 过于乐观**: iter_003 的 pilot PPL 改善（-16% 到 -25%）在 full-scale 中消失。iter_004 的 pilot accuracy 改善（0%→6.2% 或 0%→12.5%）同样可能在 full-scale 中消失。

2. **n=16 的统计噪声**: 1 个样本的差异 = 6.25pp，2 个样本 = 12.5pp。在这种粒度下，方法间的差异几乎不可能与随机噪声区分。

3. **超参数消融的假精度**: 在 n=16 上做 6-8 配置的消融，然后声称找到了"最佳"配置，这在统计上是不合理的。

**建议**: 如果计算资源允许，跳过 pilot 阶段，直接在 Countdown-100（单 seed）上做初步验证，然后在 Countdown-500（3 seeds）上做最终评估。中间的 n=16 pilot 阶段增加了管线复杂度但几乎不提供有用信号。

---

*报告生成时间: 2026-03-10*
*审计覆盖: iter_004 全部 17 个实验任务（setup_env 至 statistical_tests）*
*数据来源: exp/results/pilots/, exp/results/full/, plan/methodology.md, plan/task_plan.json, idea/proposal.md*
