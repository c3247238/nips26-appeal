# Project Memory

记录这个项目特有的长期约束、偏好和已确认决策。

- 这里只写项目层信息，不要重复系统层规则
- 长期有效的约束优先，例如数据集、资源预算、论文目标、禁用方案
- 当系统策略与项目特殊要求冲突时，在这里明确写出覆盖规则

## 实验约束（用户指定）

- **最大 batch size**: 所有实验必须尽可能使用最大 batch size（根据 GPU 显存动态调整），充分利用 98GB VRAM
- **性能优化**: 新实验脚本应启用 Flash Attention (`attn_implementation="flash_attention_2"`)、`torch.compile`、`device_map="auto"` 多 GPU 并行
- **已知问题**: `train_dal_linear.py` 等脚本 META_BATCH=2 太小、只用单 GPU、无 Flash Attention — 后续任务需修复
