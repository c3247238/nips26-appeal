# Iteration 4 方法论者视角：验证改进效果

## 改进验证框架

每项改进都必须可验证：

### 可验证的改进

| 改进 | 验证方法 | 通过标准 |
|------|----------|----------|
| 删除编造声明 | 全文搜索 "manual inspection"、"78%"、"15%"、"7%" | 0 命中 |
| 修正数据不匹配 | 对比 experiments.md 与 f4 JSON | 数字一致 |
| 统一术语 | 全文搜索 "Spearman rho" | 0 命中 |
| 添加 bootstrap CI | 检查 experiments.md 第 4.1 节 | 包含 CI |
| 生成图表 | 检查 figures/ 目录 | 3+ 个文件 |

### 不可完全验证的改进

| 改进 | 验证方法 | 局限 |
|------|----------|------|
| 重构碰撞率叙事 | 请独立读者判断 | 主观 |
| 软化普遍声明 | 请独立读者判断 | 主观 |
| K-means 分析 | 检查数据一致性 | 解释性 |

## 质量控制流程

1. **自查**：作者逐项检查改进清单
2. **数据验证**：所有数字与 JSON 结果文件交叉验证
3. **文本检查**：禁止模式扫描（编造声明、不一致术语）
4. **编译检查**：LaTeX 编译无错误

## 建议

使用自动化脚本进行文本检查：
```bash
grep -r "manual inspection" writing/
grep -r "Spearman rho" writing/
grep -r "completely cannot" writing/
grep -r "universal" writing/
```
