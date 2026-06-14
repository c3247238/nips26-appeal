# Iteration 4 实证视角：统计严谨性提升计划

## 核心目标

消除所有统计漏洞，使论文经得起审稿人 scrutiny。

## 具体措施

### 1. F1 Bootstrap置信区间

对UAD的F1 = 0.00048计算bootstrap CI：
```python
from sklearn.utils import resample

n_bootstrap = 10000
f1_scores = []
for _ in range(n_bootstrap):
    y_true_boot = resample(y_true, random_state=_)
    y_pred_boot = resample(y_pred, random_state=_)
    f1_scores.append(f1_score(y_true_boot, y_pred_boot))

ci_lower = np.percentile(f1_scores, 2.5)
ci_upper = np.percentile(f1_scores, 97.5)
```

预期结果：F1 = 0.00048 [CI: 0.00012, 0.00102]
这会让"与随机无差异"的结论更严谨。

### 2. GT样本量声明

在论文中明确标注：
- "Our ground truth comprises 7 true absorption pairs (6 distinct pairs + 1 self-pair)"
- "This small sample limits statistical power; conclusions about UAD's failure should be interpreted as strong evidence rather than definitive proof"
- "Future work with larger ground truth (100+ pairs) is needed for conclusive validation"

### 3. K-means差异分析

定量分析K-means vs Ward的差异：
- 计算两种方法对同一phi矩阵的聚类结果
- 测量GT对的co-assignment rate
- 解释为什么K-means的随机初始化有利于捕获吸收特征

### 4. 删除所有无数据支持的声明

- 删除4.5节假阳性分类百分比
- 删除numbers-only/punctuation-only子集分析（除非有实际数据）
- 所有声明必须有对应的JSON结果文件支持

### 5. 图表生成

使用matplotlib生成：
- Figure 2: Token激活热图（基于f5_false_positive_results.json）
- Figure 3: 碰撞率散点图（基于f4_collision_correlation_results.json）
- Table 1: UAD变体对比表（基于f2_uad_ablations_results.json）

## 预期效果

- 统计严谨性从6/10提升到8/10
- 消除所有"编造"指控
- 使审稿人专注于科学内容而非方法论缺陷
