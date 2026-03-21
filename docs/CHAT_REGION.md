# 对话模式：国内（Qwen）/ 国外（Gemini）

## 行为

| 模式 | 后端 | 说明 |
|------|------|------|
| **国内** `china` | 阿里云 DashScope **Qwen-Omni**（`omni_client.stream_chat`） | 流式文本 + **服务端 Omni 音频**（眼镜扬声器） |
| **国外** `international` | Google **Gemini**（`gemini_client.stream_chat_gemini`） | 流式**文本**；**无** Qwen Omni 音频，手机转发 TTS 仍可用 |

## 配置

- 环境变量 **`AIGLASS_CHAT_REGION`**：`china`（默认）或 `international`（及别名：`gemini`、`国外` 等，见 `get_chat_region()`）。
- **Gemini**：`GEMINI_API_KEY`（可选，未设置则用 `gemini_client.py` 内默认占位）；`GEMINI_MODEL`（默认 `gemini-1.5-flash`）；`GEMINI_API_BASE`（默认 Google v1beta models 根 URL）。
- **Qwen**：仍使用 `DASHSCOPE_API_KEY`（`omni_client.py`）。

## HTTP 切换（无需改代码）

```bash
# 查询
curl -s http://127.0.0.1:8081/api/chat-region

# 设为国外（Gemini）
curl -s -X POST http://127.0.0.1:8081/api/chat-region \
  -H "Content-Type: application/json" \
  -d '{"region":"international"}'

# 设回国内（Qwen）
curl -s -X POST http://127.0.0.1:8081/api/chat-region \
  -H "Content-Type: application/json" \
  -d '{"region":"china"}'
```

## 安全提示

请勿将 **Gemini API Key** 长期硬编码在仓库；生产环境请用 `.env` 设置 `GEMINI_API_KEY` 并轮换已泄露的密钥。

## 为什么「prompt 里写了字数」仍可能偏长？

- **不是接口调错**：Qwen 用 `messages` 里的 `system` + `user`；Gemini 用 `systemInstruction` + `contents`，用法符合文档。
- **大模型对「精确字数」是软约束**，不会像程序一样严格计数；中英混用时「30 字」若只按中文理解，英文会仍偏长——已用 `AIGLASS_CHAT_BRIEF_WORDS_EN` 单独约束英文词数。
- **Qwen-Omni 同时开文本+音频**时，模型有时为对齐播报而略加长；代码里已把简短要求**同时**写在 system 与 **user 文本前缀**里，提高遵循率。
- 仍偏长时可调：`.env` 里减小 `AIGLASS_CHAT_MAX_TOKENS`、降低 `AIGLASS_CHAT_TEMPERATURE`。
