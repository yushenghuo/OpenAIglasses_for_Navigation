# audio_stream.py
# -*- coding: utf-8 -*-
import os
import asyncio
from dataclasses import dataclass
from typing import Optional, Set, List, Tuple, Any, Dict
from fastapi import Request
from fastapi.responses import StreamingResponse

# ===== 下行 WAV 流基础参数 =====
STREAM_SR = 8000  # 改为8kHz，ESP32支持
STREAM_CH = 1
STREAM_SW = 2
BYTES_PER_20MS_16K = STREAM_SR * STREAM_SW * 20 // 1000  # 320B (8kHz)

# ===== 音频输出目标：local=电脑本机播放 | esp32 | phone | both =====
AUDIO_OUTPUT = os.getenv("AUDIO_OUTPUT", "local").strip().lower()
if AUDIO_OUTPUT not in ("local", "esp32", "phone", "both"):
    AUDIO_OUTPUT = "local"

# ===== 本机播放（PyAudio）=====
_local_pyaudio = None
_local_stream = None

def _get_local_audio_stream():
    """懒加载：打开本机默认声卡播放（8kHz 单声道 16bit）。"""
    global _local_pyaudio, _local_stream
    if _local_stream is not None:
        return _local_stream
    try:
        import pyaudio
        _local_pyaudio = pyaudio.PyAudio()
        _local_stream = _local_pyaudio.open(
            format=pyaudio.paInt16,
            channels=STREAM_CH,
            rate=STREAM_SR,
            output=True,
            frames_per_buffer=BYTES_PER_20MS_16K // 2,
        )
        return _local_stream
    except Exception as e:
        print(f"[AUDIO] 本机播放不可用: {e}")
        return None

# ===== AI 播放任务总闸 =====
current_ai_task: Optional[asyncio.Task] = None

async def cancel_current_ai():
    """取消当前大模型语音任务，并等待其退出。"""
    global current_ai_task
    task = current_ai_task
    current_ai_task = None
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

def is_playing_now() -> bool:
    t = current_ai_task
    return (t is not None) and (not t.done())

# ===== /stream.wav 连接管理 =====
@dataclass(frozen=True)
class StreamClient:
    q: asyncio.Queue
    abort_event: asyncio.Event

stream_clients: "Set[StreamClient]" = set()
# 手机端专用连接（/stream_phone.wav），用于 darksight 等手机 app 播放
stream_clients_phone: "Set[StreamClient]" = set()
STREAM_QUEUE_MAX = 96  # 小缓冲，避免积压

def _wav_header_unknown_size(sr=16000, ch=1, sw=2) -> bytes:
    import struct
    byte_rate = sr * ch * sw
    block_align = ch * sw
    data_size = 0x7FFFFFF0
    riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16,
        1, ch, sr, byte_rate, block_align, sw * 8,
        b"data", data_size
    )

async def hard_reset_audio(reason: str = ""):
    """
    **一键清场**：丢弃所有播放器连接（abort_event置位）+ 取消当前AI任务。
    这样旧的音频不会再有任何去处，也没有任何任务继续产出。
    """
    # 1) 断开所有正在播放的 HTTP 连接（ESP32 + 手机）
    for sc in list(stream_clients):
        try:
            sc.abort_event.set()
        except Exception:
            pass
    stream_clients.clear()
    for sc in list(stream_clients_phone):
        try:
            sc.abort_event.set()
        except Exception:
            pass
    stream_clients_phone.clear()

    # 2) 取消当前AI任务
    await cancel_current_ai()

    # 3) 日志
    if reason:
        print(f"[HARD-RESET] {reason}")

async def broadcast_pcm16_realtime(pcm16: bytes):
    """按配置：本机播放 / 推送给 ESP32 / 手机。默认仅在电脑上播放，不发给 ESP32。"""
    try:
        import sync_recorder
        sync_recorder.record_audio(pcm16, text="[Omni对话]")
    except Exception:
        pass

    send_to_esp32 = AUDIO_OUTPUT in ("esp32", "both")
    send_to_phone = AUDIO_OUTPUT in ("phone", "both")
    play_local = AUDIO_OUTPUT == "local"

    loop = asyncio.get_event_loop()
    next_tick = loop.time()
    off = 0
    while off < len(pcm16):
        take = min(BYTES_PER_20MS_16K, len(pcm16) - off)
        piece = pcm16[off:off + take]

        if play_local and piece:
            stream = _get_local_audio_stream()
            if stream:
                try:
                    await loop.run_in_executor(None, lambda p=piece: stream.write(p))
                except Exception:
                    pass

        if send_to_esp32:
            dead: List[StreamClient] = []
            for sc in list(stream_clients):
                if sc.abort_event.is_set():
                    dead.append(sc)
                    continue
                try:
                    if sc.q.full():
                        try: sc.q.get_nowait()
                        except Exception: pass
                    sc.q.put_nowait(piece)
                except Exception:
                    dead.append(sc)
            for sc in dead:
                try: stream_clients.discard(sc)
                except Exception: pass

        if send_to_phone:
            dead_phone: List[StreamClient] = []
            for sc in list(stream_clients_phone):
                if sc.abort_event.is_set():
                    dead_phone.append(sc)
                    continue
                try:
                    if sc.q.full():
                        try: sc.q.get_nowait()
                        except Exception: pass
                    sc.q.put_nowait(piece)
                except Exception:
                    dead_phone.append(sc)
            for sc in dead_phone:
                try: stream_clients_phone.discard(sc)
                except Exception: pass

        next_tick += 0.020
        now = loop.time()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
        else:
            next_tick = now
        off += take

# ===== FastAPI 路由注册器 =====
def register_stream_route(app):
    @app.get("/stream.wav")
    async def stream_wav(_: Request):
        # —— 强制单连接（或少数连接），先拉闸所有旧连接 ——
        for sc in list(stream_clients):
            try: sc.abort_event.set()
            except Exception: pass
        stream_clients.clear()

        q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)
        abort_event = asyncio.Event()
        sc = StreamClient(q=q, abort_event=abort_event)
        stream_clients.add(sc)

        async def gen():
            yield _wav_header_unknown_size(STREAM_SR, STREAM_CH, STREAM_SW)
            try:
                while True:
                    if abort_event.is_set():
                        break
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if abort_event.is_set():
                        break
                    if chunk is None:
                        break
                    if chunk:
                        yield chunk
            finally:
                stream_clients.discard(sc)
        return StreamingResponse(gen(), media_type="audio/wav")

    @app.get("/stream_phone.wav")
    async def stream_phone_wav(_: Request):
        """手机端专用语音流，供 darksight 等 app 或手机浏览器播放。"""
        for sc in list(stream_clients_phone):
            try: sc.abort_event.set()
            except Exception: pass
        stream_clients_phone.clear()

        q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)
        abort_event = asyncio.Event()
        sc = StreamClient(q=q, abort_event=abort_event)
        stream_clients_phone.add(sc)

        async def gen():
            yield _wav_header_unknown_size(STREAM_SR, STREAM_CH, STREAM_SW)
            try:
                while True:
                    if abort_event.is_set():
                        break
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if abort_event.is_set():
                        break
                    if chunk is None:
                        break
                    if chunk:
                        yield chunk
            finally:
                stream_clients_phone.discard(sc)
        return StreamingResponse(gen(), media_type="audio/wav")