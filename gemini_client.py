# gemini_client.py
# -*- coding: utf-8 -*-
"""
Google Gemini 流式对话（多模态：文本 + JPEG），与 omni_client.OmniStreamPiece 对齐（仅文本增量，无 Omni 音频）。
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from omni_client import OmniStreamPiece

# 与前端/文档约定一致；可用环境变量覆盖
GEMINI_BASE = os.getenv(
    "GEMINI_API_BASE",
    "https://generativelanguage.googleapis.com/v1beta/models",
)
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
# 优先使用环境变量 GEMINI_API_KEY；未设置时使用下列默认值（请在生产环境改为 .env）
DEFAULT_GEMINI_API_KEY = "AIzaSyDmZhc6zFN_amglkj_EgF1B-pP2In2dbJo"


def _gemini_api_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if key:
        return key
    return DEFAULT_GEMINI_API_KEY


def _content_list_to_gemini_parts(content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 OpenAI 风格 multimodal content 转为 Gemini parts。"""
    parts: List[Dict[str, Any]] = []
    for item in content_list:
        t = item.get("type")
        if t == "text":
            parts.append({"text": item.get("text") or ""})
        elif t == "image_url":
            url = (item.get("image_url") or {}).get("url") or ""
            if "base64," in url:
                b64 = url.split("base64,", 1)[-1].strip()
                mime = "image/jpeg"
                if "image/png" in url:
                    mime = "image/png"
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    return parts


def _extract_text_from_chunk(obj: Dict[str, Any]) -> str:
    out = []
    try:
        for c in obj.get("candidates") or []:
            content = c.get("content") or {}
            for p in content.get("parts") or []:
                if "text" in p:
                    out.append(p["text"])
    except Exception:
        pass
    return "".join(out)


async def stream_chat_gemini(
    content_list: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> AsyncGenerator[OmniStreamPiece, None]:
    """
    Gemini streamGenerateContent（SSE），产出 OmniStreamPiece（仅 text_delta；audio_b64 恒为 None）。
    """
    api_key = _gemini_api_key()
    m = model or GEMINI_DEFAULT_MODEL
    url = f"{GEMINI_BASE.rstrip('/')}/{m}:streamGenerateContent"
    parts = _content_list_to_gemini_parts(content_list)
    if not parts:
        parts = [{"text": ""}]

    body: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
    }
    if system_prompt and system_prompt.strip():
        body["systemInstruction"] = {"parts": [{"text": system_prompt.strip()}]}
    if max_output_tokens is None:
        max_output_tokens = int(os.getenv("AIGLASS_CHAT_MAX_TOKENS", "120"))
    if temperature is None:
        temperature = float(os.getenv("AIGLASS_CHAT_TEMPERATURE", "0.2"))
    body["generationConfig"] = {
        "maxOutputTokens": max_output_tokens,
        "temperature": temperature,
    }

    params = {"key": api_key}
    timeout = aiohttp.ClientTimeout(total=300, sock_read=120)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, params=params, json=body) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                raise RuntimeError(f"Gemini HTTP {resp.status}: {err_text[:500]}")

            buffer = b""
            accumulated = ""
            async for chunk in resp.content.iter_chunked(4096):
                if not chunk:
                    continue
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(b"data:"):
                        line = line[5:].strip()
                    if line == b"[DONE]":
                        break
                    try:
                        obj = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    if "error" in obj:
                        msg = obj.get("error", {})
                        raise RuntimeError(f"Gemini API error: {msg}")
                    piece = _extract_text_from_chunk(obj)
                    if not piece:
                        continue
                    # 流式可能是「累计全文」或「仅增量」，两种都兼容
                    if piece.startswith(accumulated):
                        delta = piece[len(accumulated) :]
                        accumulated = piece
                    else:
                        delta = piece
                        accumulated += piece
                    if delta:
                        yield OmniStreamPiece(text_delta=delta, audio_b64=None)

            if buffer.strip():
                try:
                    line = buffer.strip()
                    if line.startswith(b"data:"):
                        line = line[5:].strip()
                    obj = json.loads(line.decode("utf-8"))
                    piece = _extract_text_from_chunk(obj)
                    if piece:
                        if piece.startswith(accumulated):
                            delta = piece[len(accumulated) :]
                        else:
                            delta = piece
                        if delta:
                            yield OmniStreamPiece(text_delta=delta, audio_b64=None)
                except Exception:
                    pass
