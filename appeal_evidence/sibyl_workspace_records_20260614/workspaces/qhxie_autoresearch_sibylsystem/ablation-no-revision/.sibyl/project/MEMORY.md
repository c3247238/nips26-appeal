# Project Memory

记录这个项目特有的长期约束、偏好和已确认决策。

- 这里只写项目层信息，不要重复系统层规则
- 长期有效的约束优先，例如数据集、资源预算、论文目标、禁用方案
- 当系统策略与项目特殊要求冲突时，在这里明确写出覆盖规则

## 项目总结 (ablation-no-revision)

### 主题
研究 LLM 推理中的"消融而非修正"问题：训练模型在单次推理中产生正确答案。

### 状态
- 阶段: pilot_experiments (Round 3)
- 阻塞: **无 API Key** - Round 3 API-based 实验需要 DeepSeek/OpenAI API Key

### 关键发现

#### Round 1: EDW-Step-DPO
- 深度加权 DPO 训练损失相同（0.6922-0.6927）- 方法无效
- H3 超加性假设被证伪
- 决策: PIVOT to CCAR

#### Round 2: CCAR
- DeepSeek-Math-7B 准确率仅 26%（预期 40%+）- H1 证伪
- Step-DPO 训练损失 0.694 > 0.5 - H2 证伪
- GPU 训练因 Blackwell 不兼容而失败（H3/H4 阻塞）

#### 硬件突破
- **PyTorch 2.11.0 已安装，支持 RTX PRO 6000 Blackwell (sm_120)**
- 测试确认所有训练操作正常工作
- 含义: Round 2 CCAR 实验可以重新运行

#### Round 3: Training-Free API Inference
- 提案: Answer Consistency + Ranked Voting
- 新颖性: 5/10 (与 RASC 碰撞)
- 备份: Activation Energy Theory (8/10)
- **阻塞: 无 DeepSeek/OpenAI API Key**

### 实验假设

| Round | Hypothesis | Status | Result |
|-------|------------|--------|--------|
| 1 | H3: EDW 超加性 | 证伪 | 准确率无超加性 |
| 2 | H1: 40%+ baseline | 证伪 | 实际 26% |
| 2 | H2: 损失 <0.5 | 证伪 | 实际 0.694 |
| 3 | H1: 一致性预测正确性 | 待测试 | Correlation > 0.3 |
| 3 | H2: 排序投票 > 多数投票 | 待测试 | Delta > 2% |
| 3 | H3: >40% token 减少 | 待测试 | 需要完整实验 |

### 下一步
1. 配置 API Key 后运行 Round 3 实验
2. 或使用 PyTorch 2.11.0 重新运行 Round 2 CCAR 实验
