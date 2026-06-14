# Project Memory

记录这个项目特有的长期约束、偏好和已确认决策。

- 这里只写项目层信息，不要重复系统层规则
- 长期有效的约束优先，例如数据集、资源预算、论文目标、禁用方案
- 当系统策略与项目特殊要求冲突时，在这里明确写出覆盖规则

## 研究目标（Iteration 3+ 更新）

**统一动态权重衰减框架（Unified Dynamic Weight Decay Framework）**：
1. 将 WD scheduling、alignment-aware WD、decoupled WD、norm-matched WD 等方法统一到一套理论框架
2. 提出标准化评测体系（Budget Equivalence Metric, Coupling Stability Index, Alignment Informativeness Score）
3. 系统性数学推导和大规模可视化分析，揭示核心问题并指明发展方向
4. 大量公式论证 + 可视化分析图表

## 实验约束

- **数据集**: CIFAR-10, CIFAR-100, **ImageNet**（用户明确要求，最高优先级）
  - **CIFAR 数据集太简单**，仅靠 CIFAR 实验无法充分验证方法的有效性，论文说服力不足
  - **必须加入 ImageNet 规模的实验**进行训练、测试和评估，这是提升论文质量的关键
  - ImageNet 实验结果应作为核心贡献的主要支撑证据
  - **ImageNet 数据路径**: `/home/ccwang/dataset/imagenet-1k`（HuggingFace parquet 格式，294 train shards + 28 test shards）
- **架构**: ResNet-20, VGG-16-BN（CIFAR）, ResNet-50, ResNet-101（ImageNet）, 考虑 Vision Transformer
- **compute_backend**: local（8x RTX PRO 6000 Blackwell, 98GB each）
- **多种子**: 所有核心实验必须使用至少 3 个种子（42, 123, 456），报告 mean ± std
- **时间预算覆盖**: ImageNet 实验可能需要更长时间（4-8 小时），覆盖默认 1 小时限制
- **质量提升方向**: 当前分数 7.0/8.0 停滞，ImageNet 实验是突破瓶颈的关键路径

## 论文质量检查规则（强制）

1. **图片内容视觉验证**: 生成的所有图片（PNG/PDF）必须调用视觉能力（Read tool 读取图片文件）检查内容——坐标轴标签、图例、数据一致性、颜色区分度、有无截断/空白/渲染异常。
2. **最终 PDF 图片插入验证**: LaTeX 编译生成 PDF 后，必须逐页读取 PDF（Read tool + pages 参数）视觉确认每张图片正确显示、位置正常、caption/label 正确渲染、分辨率足够。
