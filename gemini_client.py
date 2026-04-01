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
from aiohttp_socks import ProxyConnector

from omni_client import OmniStreamPiece

# 与前端/文档约定一致；可用环境变量覆盖
GEMINI_BASE = os.getenv(
    "GEMINI_API_BASE",
    "https://generativelanguage.googleapis.com/v1beta/models",
)
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# 优先使用环境变量 GEMINI_API_KEY；未设置时使用下列默认值（请在生产环境改为 .env）


def _gemini_api_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if key:
        return key
    return ""


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


def _pick_proxy_url() -> Optional[str]:
    """
    代理优先级：
    1) GEMINI_PROXY_URL（推荐，支持 socks5:// / socks5h:// / http://）
    2) HTTPS_PROXY / HTTP_PROXY（兼容）
    """
    p = (os.getenv("GEMINI_PROXY_URL") or "").strip()
    if p:
        return p
    p = (os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or "").strip()
    return p or None


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

    # 官方流式建议使用 alt=sse，返回标准 SSE 行（data: {...}）
    params = {"key": api_key, "alt": "sse"}
    req_timeout_s = float(os.getenv("GEMINI_TOTAL_TIMEOUT_SEC", "60"))
    read_timeout_s = float(os.getenv("GEMINI_READ_TIMEOUT_SEC", "20"))
    timeout = aiohttp.ClientTimeout(total=req_timeout_s, sock_read=read_timeout_s)

    proxy_url = _pick_proxy_url()
    connector = None
    request_kwargs: Dict[str, Any] = {}
    if proxy_url:
        low = proxy_url.lower()
        if low.startswith(("socks5://", "socks5h://", "socks4://", "socks4a://")):
            connector = ProxyConnector.from_url(proxy_url)
        else:
            # aiohttp 原生支持 http/https 代理
            request_kwargs["proxy"] = proxy_url

    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
        url = f"{GEMINI_BASE.rstrip('/')}/{m}:streamGenerateContent"
        try:
            resp = await session.post(url, params=params, json=body, **request_kwargs)
        except TimeoutError:
            raise RuntimeError(f"Gemini request timeout @ {m}") from None
        except aiohttp.ServerTimeoutError:
            raise RuntimeError(f"Gemini request timeout @ {m}") from None
        except Exception as e:
            raise RuntimeError(f"Gemini request failed @ {m}: {e}") from None

        async with resp:
            if resp.status != 200:
                err_text = await resp.text()
                if resp.status == 408:
                    raise RuntimeError(f"Gemini request timeout @ {m}: {err_text[:300]}")
                raise RuntimeError(f"Gemini HTTP {resp.status} @ {m}: {err_text[:500]}")
            if proxy_url:
                print(f"[GEMINI] stream opened model={m}, status={resp.status}, proxy={proxy_url}", flush=True)
            else:
                print(f"[GEMINI] stream opened model={m}, status={resp.status}, proxy=direct", flush=True)

            buffer = b""
            accumulated = ""
            saw_any_event = False
            last_finish_reason: Optional[str] = None
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
                    saw_any_event = True
                    if "error" in obj:
                        msg = obj.get("error", {})
                        raise RuntimeError(f"Gemini API error: {msg}")
                    try:
                        cands = obj.get("candidates") or []
                        if cands:
                            fr = cands[0].get("finishReason")
                            if fr:
                                last_finish_reason = str(fr)
                    except Exception:
                        pass
                    # 官方文档：当被安全策略拦截时，可能只有 promptFeedback 无 candidates
                    if "promptFeedback" in obj and not (obj.get("candidates") or []):
                        pf = obj.get("promptFeedback") or {}
                        br = pf.get("blockReason")
                        if br:
                            raise RuntimeError(f"Gemini prompt blocked: {br}")
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
                    saw_any_event = True
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

            # 避免“静默空响应”：明确给上层错误原因，便于日志定位
            if not accumulated.strip():
                reason = f", finish_reason={last_finish_reason}" if last_finish_reason else ""
                if saw_any_event:
                    raise RuntimeError(f"Gemini returned no text content{reason}")
                raise RuntimeError("Gemini stream returned no events")
            print(f"[GEMINI] stream done chars={len(accumulated)} finish={last_finish_reason}", flush=True)
