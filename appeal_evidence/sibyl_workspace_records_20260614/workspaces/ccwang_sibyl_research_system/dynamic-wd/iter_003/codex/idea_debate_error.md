# Codex MCP 调用错误 - idea_debate

**时间**: 2026-03-18
**错误类型**: MCP 工具不可用
**详情**: `mcp__codex__codex` 工具在当前 session 中未注册。Codex MCP server 未配置或未启动。

## 影响

Codex 独立第三方审查无法执行。已生成本地替代审查写入 `idea_debate_review.md`。

## 建议

1. 检查 `~/.codex/config.toml` 是否存在且 `OPENAI_API_KEY` 已设置
2. 确认 `codex mcp-server` stdio 已在 MCP 配置中注册
3. 重新启动 session 后重试
