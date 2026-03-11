import os
import io
import urllib.request
import wave
import audioop
from typing import Any, Dict

import dashscope

from audio_player import enqueue_external_pcm


# --- Qwen 官方 TTS 配置 ---
_DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 默认使用北京区域 API，如需新加坡可在环境变量中覆盖
dashscope.base_http_api_url = os.getenv(
    "DASHSCOPE_HTTP_API_URL",
    "https://dashscope.aliyuncs.com/api/v1",
)

QWEN_TTS_MODEL = os.getenv("QWEN_TTS_MODEL", "qwen3-tts-flash")
QWEN_TTS_VOICE = os.getenv("QWEN_TTS_VOICE", "Cherry")
QWEN_TTS_LANG = os.getenv("QWEN_TTS_LANG", "English")  # 地图指令多为英文


def _tts_once_sync(text: str) -> None:
    """
    同步调用 Qwen 官方 TTS，一次性取完整音频：
    1) 调 DashScope TTS，拿到音频 URL
    2) 下载 WAV
    3) 转 8kHz 单声道 PCM16
    4) 丢进统一播放队列
    """
    if not text:
        return
    if not _DASHSCOPE_API_KEY:
        print("[NAV_TTS] 未设置 DASHSCOPE_API_KEY，无法调用 Qwen TTS")
        return

    try:
        resp: Dict[str, Any] = dashscope.MultiModalConversation.call(
            model=QWEN_TTS_MODEL,
            api_key=_DASHSCOPE_API_KEY,
            text=text,
            voice=QWEN_TTS_VOICE,
            language_type=QWEN_TTS_LANG,
        )
    except Exception as e:
        print(f"[NAV_TTS] 调用 Qwen TTS 失败: {e}")
        return

    try:
        audio_info = (resp or {}).get("output", {}).get("audio") or {}
        audio_url = audio_info.get("url")
        if not audio_url:
            print(f"[NAV_TTS] Qwen TTS 未返回 audio.url: {resp}")
            return
    except Exception as e:
        print(f"[NAV_TTS] 解析 Qwen TTS 响应失败: {e}")
        return

    try:
        with urllib.request.urlopen(audio_url) as f:
            wav_bytes = f.read()
    except Exception as e:
        print(f"[NAV_TTS] 下载 TTS 音频失败: {e}")
        return

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_f:
            channels = wav_f.getnchannels()
            sampwidth = wav_f.getsampwidth()
            framerate = wav_f.getframerate()
            frames = wav_f.readframes(wav_f.getnframes())

        if channels == 2:
            frames = audioop.tomono(frames, sampwidth, 1, 0)
        if framerate != 8000:
            frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
        if sampwidth != 2:
            print(
                f"[NAV_TTS] Qwen TTS 返回采样位宽={sampwidth}，期望16bit，可能音色有限制"
            )

        enqueue_external_pcm(frames)
    except Exception as e:
        print(f"[NAV_TTS] 处理 TTS WAV 失败: {e}")


async def synth_nav_tts_and_enqueue(text: str) -> None:
    """
    异步封装：在后台线程里调用 Qwen 官方 TTS，并把结果丢进统一播放队列。
    """
    if not text:
        return
    import asyncio

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _tts_once_sync, text)

