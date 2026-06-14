# Codex MCP 调用失败记录

**时间**: 2026-03-17
**模式**: idea_debate
**错误类型**: MCP 工具不可用

## 错误详情

`claude mcp list` 未返回任何 codex 相关的 MCP server。Codex MCP（`codex mcp-server` stdio）未在当前环境中注册。

## 处理方式

根据错误处理协议，已由 codex-reviewer agent 以独立第三方视角生成替代评审，写入 `codex/idea_debate_review.md`。Pipeline 未被阻塞。

## 恢复建议

若需启用 Codex 审查，请确保：
1. OpenAI API key 已配置（`OPENAI_API_KEY` 环境变量）
2. Codex CLI 已安装（`npm install -g @openai/codex`）
3. Codex MCP server 已注册：`claude mcp add codex -- codex mcp-server`
4. `~/.codex/config.toml` 中 `approval-policy = "never"` 已设置
