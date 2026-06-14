# 实用主义者视角：迭代 5 研究提案

**角色**: 工程可行性优先，关注可执行性、时间成本、风险控制
**日期**: 2026-04-14
**焦点**: 宽度/L0 混淆的可操作修复 + 分类学膨胀的根治方案

---

## 一、问题诊断

迭代 4 的分数从 5.5 跃升至 6.5，但两个 BLOCKING 级问题阻止进一步提升：

### 问题 1：H3 宽度/L0 混淆（EXP001）

**现状**：论文最强发现（absorption 与下游质量的相关性 r=-0.595）建立在 54 个 Gemma Scope SAE 的全样本相关上。但匹配比较中，5 个高吸收 SAE 全部为 1M 宽度（L0 16-58），5 个低吸收 SAE 全部为 16k/65k（L0 137-297）。偏相关控制了 log(width)、layer、arch_class，但**未包含 L0 作为协变量**。SCR 偏相关从 r=-0.431 跳至 partial r=-0.677，是典型抑制变量效应。

**核心风险**：如果在宽度层内（16k-only, 65k-only, 1M-only）相关性消失，H3 退化为生态谬误："宽 SAE 同时具有高吸收和低质量，但吸收不是因果中介。"

### 问题 2：分类学膨胀（EXP002）

**现状**：92.3% 综合吸收率在摘要、引言、结论中作为主要发现呈现。但论文自身的 suspicious_flags 承认 Type II 率"CRITICAL[ly] likely inflated"。几乎所有字母的 n_comparison_tokens=0，迫使回退到全局均值基线。88.5% 的 Type II 率（23/26 字母）是回退逻辑的产物，不是部分吸收的证据。仅 Type I（3.8%）和 Chanin 复现率（80.8%）经过验证。

**核心风险**：任何审稿人在读完方法后都会注意到 92.3% 的不可信性。这不是可以用对冲语言修补的问题，而是需要实验层面的根治。

---

## 二、提案 A（推荐）：零 GPU 混淆控制分析 + 有根据的分类学重跑

### A1：L0 协变量控制（零 GPU，2-3 小时）

**操作**：
1. 从 C3A 结果 JSON 中提取 54 个 SAE 的 L0 值（已有数据，不需新实验）
2. 将 L0 加入偏相关模型，计算 absorption vs. sparse_probing / RAVEL / SCR / unlearning 的偏相关（控制 log(width) + layer + arch_class + log(L0)）
3. 计算宽度层内分层相关：
   - 16k-only（约 15 个 SAE）：absorption vs. 4 个下游指标
   - 65k-only（约 15 个 SAE）：同上
   - 1M-only（约 18 个 SAE）：同上
4. SCR 抑制变量顺序诊断：逐个控制 width-only、layer-only、arch-only、L0-only，定位 SCR 膨胀来源

**代码实现**：
```python
# 核心逻辑：从现有 C3A JSON 加载，用 pingouin 计算偏相关
import pingouin as pg
import pandas as pd

df = pd.read_json("iter_004/exp/results/full/C3A_saebench_correlation_results.json")
# 加入 L0 协变量
partial_r = pg.partial_corr(data=df, x='absorption_score', y='sparse_probing',
                            covar=['log_width', 'layer', 'arch_class', 'log_L0'])
# 宽度层内分层
for width in ['16k', '65k', '1M']:
    tier = df[df['width_class'] == width]
    if len(tier) >= 8:  # 最小样本量
        r = pg.corr(tier['absorption_score'], tier['sparse_probing'])
```

**判定准则**（预注册）：
- 如果 L0 控制后 absorption-sparse_probing 偏相关 r < -0.3 且 p < 0.05：混淆不致命，H3 存活
- 如果宽度层内至少 2/3 个层级显示 r < -0.2：H3 在层级内成立
- 如果两项均不满足：H3 需要根本性重构，论文叙事转向"宽度-质量关联"

**时间**：2-3 小时纯分析，零 GPU。这是整个迭代中 ROI 最高的任务。

### A2：聚类标准误 PMI 回归（零 GPU，1 小时）

**操作**：
1. 用 letter-level cluster（26 个聚类）重跑 C2C PMI 回归
2. 同时报告 HC3 和聚类 SE 结果
3. 如果 L0 在聚类下不显著（p > 0.05），修改讨论为"仅 layer 是吸收率的稳健预测因子"
4. 考虑 beta 回归或零膨胀模型（skewness=5.186 表明 OLS 不适用）

**代码实现**：
```python
import statsmodels.api as sm
# 聚类 SE
model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['letter']})
# Beta 回归
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Tweedie
```

**时间**：1 小时纯分析。

### A3：分类学重跑（1-2 GPU 小时）

**核心改进**：用 sae-spelling 地面真实父特征 ID 替换启发式父特征识别。

**操作**：
1. 从 sae-spelling 的 Chanin et al. IG（集成梯度）标签中提取已知的吸收对（父特征 ID + 被吸收子特征 ID）
2. 对于每个字母，用已知父特征的实际激活数据计算 comparison_tokens（而非回退到全局统计）
3. 重跑 Type II 分类：magnitude_ratio 基于真实父特征激活分布，而非全局均值
4. 如果 sae-spelling 标签不直接提供父特征 ID，可从 C1B 的 probe 数据中反推：probe 正确但 SAE 漏检的 token 集合即为 comparison_tokens 的候选集

**判定准则**：
- 报告三个数字：Type I 验证率、Type II 修正率、Chanin 复现率
- 92.3% 在所有出现位置标记为 "上界（upper bound）"
- 摘要主打数字改为 Type I + Chanin 验证率

**时间**：1-2 GPU 小时（需要加载 SAE 并计算父特征激活）。

### A4：Figure 1 生成（1 小时）

**操作**：从 fig_lv_framework_desc.md 生成 LV 框架概念图。用 matplotlib + networkx 而非 TikZ（更快迭代）。

**时间**：1 小时。

---

## 三、提案 B（补充）：跨域吸收度量拓展

### B1：RAVEL 知识层次结构吸收测量

**动机**：所有吸收测量仅限首字母拼写任务。RAVEL 提供了天然的实体-属性层次结构（城市->国家->大洲），且 SAE-RAVEL 已经显示 SAE 在解缠方面的失败。但没有人将这个失败**定性为吸收**。

**操作**：
1. 加载 RAVEL 城市数据集（3000+ 城市，country/continent/language 属性）
2. 构造探测模板："{City} is located in the country:" 等
3. 在 Gemma 2 2B 上训练 LR probes（TransformerLens 提取 residual stream 激活）
4. 移植 sae-spelling 吸收度量到知识层次结构：
   - k-sparse probing 找特征分裂（通用，直接移植）
   - false-negative 识别（通用）
   - 集成梯度消融（需将 SpellingGrader 替换为 KnowledgeGrader，约 200 行代码）
5. 测量 Gemma Scope 16k/65k SAE 在层 8-17 上的城市-国家吸收率
6. 与同一 SAE 上的首字母吸收率做并列比较

**时间**：Pilot 15 分钟（1 个 SAE, 1 层, 3 个高频国家 + 3 个低频国家），完整实验 4-6 GPU 小时。

**风险评估**：
- 探针质量不足 → 低风险（RAVEL 先前工作报告 >90% 准确率）
- 吸收阈值不迁移 → 中风险（需做阈值扫描）
- 吸收率接近零 → 低风险（本身是有价值的负面结果）

**我对这个方向的工程判断**：基础设施完备（sae-spelling + RAVEL + SAELens + Gemma Scope 全部现成），代码适配量可控（约 200 行），单 GPU 可完成，且填补文献明确指出的 Gap 2 和 Gap 6。即使吸收率与首字母任务显著不同，结果也有发表价值。

### B2：L0 敏感性分析（利用现有 SAE）

**操作**：
1. 在 Gemma Scope 中找同一层、不同 L0 的 SAE（利用 JumpReLU 阈值差异产生的 L0 变化）
2. 在这些 SAE 上测量吸收率 + 特征对冲率
3. 画 L0 vs 吸收率曲线，识别吸收与对冲的交叉点

**时间**：1-2 GPU 小时（不训练新 SAE，只用现有的）。

---

## 四、执行优先级与时间分配

| 优先级 | 任务 | 类型 | 预估时间 | 阻断级别 |
|--------|------|------|----------|----------|
| P0 | A1: L0 协变量 + 宽度层内分层 | 零 GPU 分析 | 2-3h | BLOCKING |
| P0 | A2: 聚类 SE 回归 | 零 GPU 分析 | 1h | BLOCKING |
| P1 | A3: 分类学重跑 | GPU 实验 | 1-2h | BLOCKING |
| P1 | 写作修正：92.3% 标为上界 | 写作 | 0.5h | BLOCKING |
| P1 | 写作修正："validated" -> "quality-correlated" | 写作 | 0.5h | BLOCKING |
| P2 | A4: Figure 1 生成 | 图表 | 1h | HIGH |
| P2 | B1: RAVEL 知识层次吸收 pilot | GPU 实验 | 0.25h | MAJOR |
| P3 | B1: RAVEL 完整实验 | GPU 实验 | 4-6h | MAJOR |
| P3 | B2: L0 敏感性 | GPU 实验 | 1-2h | MODERATE |
| P4 | 安全探针移至附录 | 写作 | 0.5h | MODERATE |
| P4 | 标题修改 | 写作 | 15min | MODERATE |

**总时间预算**：P0+P1 = 5-7 小时（其中零 GPU 3-4 小时），P2 = 1.25 小时，P3 = 5-8 小时。

---

## 五、决策树

```
A1 完成
├── H3 在 L0 控制 + 层内分层后存活（r < -0.3, p < 0.05）
│   ├── 继续 H3 为论文核心贡献
│   ├── 执行 A3 + B1 扩展论文广度
│   └── 目标分数：7.5+
│
└── H3 在控制后消失
    ├── 论文核心需要重构
    ├── 可能方向：
    │   ├── "宽度-质量"相关性（去掉吸收作为中介）
    │   ├── 转向 B1 跨域吸收作为主贡献
    │   └── 诚实报告 H3 是生态谬误
    └── 目标分数：6.5-7.0（比当前不会更差）
```

---

## 六、对其他视角的预期回应

**理论派可能要求**：建立吸收率的闭合形式预测模型。**我的回应**：在缺乏充分控制的经验数据之前，理论模型是无锚的推测。先做 A1 确认 H3 的可靠性，再谈理论。

**创新派可能要求**：无监督吸收检测方法。**我的回应**：iter_004 的 H1（LV 检测器，F1=0.128）已经证明这条路不通。ROC-AUC=0.148 意味着反预测。在没有新的信号源之前，不应该在这个方向上投入更多时间。

**怀疑派可能质疑**：RAVEL 跨域实验（B1）是否只是换了一个任务重跑同样的度量。**我的回应**：这恰好是社区最需要的验证。文献中至少 5 篇论文（Chanin et al., SAEBench, SynthSAEBench, OrtSAE, KronSAE）明确呼吁将吸收测量扩展到首字母以外的领域。"换任务重跑"在实验科学中叫做"外部效度验证"，这是论文可发表的必要条件。

**方法论者可能要求**：更严格的混淆控制设计（如工具变量、断点回归）。**我的回应**：54 个 SAE 的观测数据不支持复杂因果推断方法。偏相关 + 层内分层是样本量允许的最严格控制。如果社区未来提供 200+ 个匹配控制的 SAE 数据集，再考虑更强的因果方法。

---

## 七、工程可行性总结

**最核心的判断**：A1（L0 协变量控制）是整个迭代 5 中影响最大、成本最低的单一任务。它的结果决定论文是继续 H3 叙事还是需要结构性重构。在 A1 完成之前，任何新实验或写作修改都是在不确定的地基上建房子。

**执行顺序的铁律**：A1 -> A2 -> （如果 H3 存活）A3 + B1 pilot -> 写作修正 -> B1 完整实验 -> B2

**不应该做的事**：
1. 不要在 A1 完成前开始新的 GPU 实验（浪费时间在可能需要重构的框架上）
2. 不要尝试新的无监督检测方法（H1 已经证伪，ROC-AUC < 0.5）
3. 不要训练新的 SAE（利用 Gemma Scope 现有预训练模型，保持 training-free 定位）
4. 不要花时间美化 92.3% 的表述——用实验（A3）修正数字，而不是用语言修饰问题
