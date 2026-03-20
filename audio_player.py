# audio_player.py
# 处理预录音频文件的播放，通过ESP32扬声器输出

import os
import wave
import json
import asyncio
import threading
import queue
import time
import re
import sys
from typing import Any, Callable, Dict, Optional
from audio_stream import broadcast_pcm16_realtime
from audio_compressor import compressed_audio_cache, AudioCompressor

# 导入录制器（避免循环导入，在需要时动态导入）
_recorder_imported = False
_sync_recorder = None

def _get_recorder():
    """延迟导入录制器"""
    global _recorder_imported, _sync_recorder
    if not _recorder_imported:
        try:
            import sync_recorder as sr
            _sync_recorder = sr
            _recorder_imported = True
        except Exception as e:
            print(f"[AUDIO] 无法导入录制器: {e}")
            _recorder_imported = True  # 标记已尝试，避免重复
    return _sync_recorder

# 兼容旧工程中的示例音频（保留）；Mac/跨平台使用项目下的 music 目录
AUDIO_BASE_DIR = os.getenv("AUDIO_BASE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "music"))

# 新增：voice 目录与映射表
# 使用脚本所在目录的 voice 文件夹，避免工作目录问题
VOICE_DIR = os.getenv("VOICE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice"))
VOICE_MAP_FILE = os.path.join(VOICE_DIR, "map.zh-CN.json")

# 音频文件映射（将合并 voice 映射）
AUDIO_MAP = {
    "检测到物体": os.path.join(AUDIO_BASE_DIR, "音频1.wav"),
    "向上": os.path.join(AUDIO_BASE_DIR, "音频2.wav"),
    "向下": os.path.join(AUDIO_BASE_DIR, "音频3.wav"),
    "向左": os.path.join(AUDIO_BASE_DIR, "音频4.wav"),
    "向右": os.path.join(AUDIO_BASE_DIR, "音频5.wav"),
    "OK": os.path.join(AUDIO_BASE_DIR, "音频6.wav"),
    "向前": os.path.join(AUDIO_BASE_DIR, "音频7.wav"),
    "后退": os.path.join(AUDIO_BASE_DIR, "音频8.wav"),
    "拿到物体": os.path.join(AUDIO_BASE_DIR, "音频9.wav"),
}

# 音频缓存，避免重复读取
_audio_cache = {}

# 音频播放队列和工作线程 - 使用优先级队列
_audio_queue = queue.PriorityQueue(maxsize=10)
_audio_priority = 0  # 递增的优先级计数器
_worker_thread = None
_worker_loop = None
_is_playing = False  # 标记是否正在播放音频
_playing_lock = threading.Lock()  # 播放锁
_initialized = False
_last_play_ts = 0.0  # 记录上次播放结束时间，用于决定预热静音长度

# 语音转发模式：服务端不本地播报，仅将TTS事件转发给手机端
_tts_forward_only = os.getenv("AIGLASS_TTS_FORWARD_ONLY", "1").strip().lower() in ("1", "true", "yes", "on")
_tts_forwarder: Optional[Callable[[Dict[str, Any]], None]] = None


def set_tts_forwarder(forwarder: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """注册 TTS 事件转发器（通常由 app_main 注入，转发到 iPhone websocket）。"""
    global _tts_forwarder
    _tts_forwarder = forwarder


def _enqueue_pcm(pcm_data: bytes):
    """将已是 8kHz 单声道 PCM16 的数据放入播放队列（线程安全）。"""
    global _audio_queue, _audio_priority

    if not _initialized:
        initialize_audio_system()

    if not pcm_data:
        return

    # 队列实时策略与 play_audio_threadsafe 一致
    queue_size = _audio_queue.qsize()
    with _playing_lock:
        currently_playing = _is_playing

    if queue_size > 0 and not currently_playing:
        print(f"[AUDIO] 清空队列（当前{queue_size}个），播放最新语音")
        _audio_queue = queue.PriorityQueue(maxsize=10)
    elif queue_size > 1 and currently_playing:
        print(f"[AUDIO] 队列积压({queue_size}个)，清空以保持实时")
        _audio_queue = queue.PriorityQueue(maxsize=10)

    try:
        _audio_priority += 1
        _audio_queue.put_nowait((_audio_priority, pcm_data))
        if queue_size >= 1:
            print(f"[AUDIO] 播放队列当前大小: {queue_size + 1}")
    except queue.Full:
        print("[AUDIO] 队列满，丢弃一条 TTS 语音")
        return

def load_wav_file(filepath):
    """加载WAV文件并返回PCM数据（自动转换为8kHz）"""
    if filepath in _audio_cache:
        return _audio_cache[filepath]
    
    # 使用压缩缓存
    if os.getenv("AIGLASS_COMPRESS_AUDIO", "1") == "1":
        compressed_data = compressed_audio_cache.load_and_compress(filepath)
        if compressed_data:
            # 存储压缩后的数据
            _audio_cache[filepath] = compressed_data
            return compressed_data
    
    # 原始加载方式（不压缩）
    try:
        with wave.open(filepath, 'rb') as wav:
            # 检查音频格式
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            
            if channels != 1:
                print(f"[AUDIO] 警告: {filepath} 不是单声道，将只使用第一个声道")
            if sampwidth != 2:
                print(f"[AUDIO] 警告: {filepath} 不是16位音频")
            
            # 读取所有帧
            frames = wav.readframes(wav.getnframes())
            
            # 如果是立体声，只取左声道
            if channels == 2:
                import audioop
                frames = audioop.tomono(frames, sampwidth, 1, 0)
            
            # 统一转换为8kHz（使用ratecv保证音调和速度不变）
            if framerate != 8000:
                import audioop
                frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
                print(f"[AUDIO] 重采样: {filepath} {framerate}Hz -> 8000Hz")
            
            _audio_cache[filepath] = frames
            return frames
            
    except Exception as e:
        print(f"[AUDIO] 加载音频文件失败 {filepath}: {e}")
        return None

def _merge_voice_map():
    """读取 voice/map.zh-CN.json 并合并到 AUDIO_MAP"""
    try:
        if not os.path.exists(VOICE_MAP_FILE):
            print(f"[AUDIO] 未找到映射文件: {VOICE_MAP_FILE}")
            return
        with open(VOICE_MAP_FILE, "r", encoding="utf-8") as f:
            m = json.load(f)
        added = 0
        for text, info in (m or {}).items():
            files = (info or {}).get("files") or []
            if not files:
                continue
            fname = files[0]
            fpath = os.path.join(VOICE_DIR, fname)
            if os.path.exists(fpath):
                AUDIO_MAP[text] = fpath
                added += 1
            else:
                print(f"[AUDIO] 映射文件缺失: {fpath}")
        print(f"[AUDIO] 已合并 voice 映射 {added} 条")
    except Exception as e:
        print(f"[AUDIO] 读取 voice 映射失败: {e}")

def preload_all_audio():
    """预加载所有音频文件到内存"""
    print("[AUDIO] 开始预加载音频文件...")
    loaded_count = 0
    
    # 【暂时禁用变速】因为需要修改缓存机制
    # 需要加速的音频列表（斑马线相关）
    # speedup_keywords = ["斑马线", "画面"]
    # speedup_factor = 1.3  # 加速30%
    
    for audio_key, filepath in AUDIO_MAP.items():
        if os.path.exists(filepath):
            # 【修复】暂时使用默认速度加载
            # need_speedup = any(keyword in audio_key for keyword in speedup_keywords)
            # speed = speedup_factor if need_speedup else 1.0
            
            data = load_wav_file(filepath)  # 使用默认参数
            if data:
                loaded_count += 1
                # if need_speedup:
                #     print(f"[AUDIO] 加载（加速{speedup_factor}x）: {audio_key}")
        else:
            # 降低噪声输出
            pass
    print(f"[AUDIO] 预加载完成，共加载 {loaded_count} 个音频文件")

def _audio_worker():
    """音频播放工作线程"""
    global _worker_loop
    
    # 尝试设置线程优先级（Windows特定）
    try:
        import ctypes
        import sys
        if sys.platform == "win32":
            # 设置线程为高优先级
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(),
                1  # THREAD_PRIORITY_ABOVE_NORMAL
            )
            print("[AUDIO] 设置音频线程为高优先级")
    except Exception as e:
        print(f"[AUDIO] 设置线程优先级失败: {e}")
    
    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)
    
    async def process_queue():
        while True:
            try:
                # 从优先级队列获取数据
                priority_data = await asyncio.get_event_loop().run_in_executor(None, _audio_queue.get, True)
                if priority_data is None:
                    break
                # 解包优先级和实际音频数据
                if isinstance(priority_data, tuple) and len(priority_data) == 2:
                    _, audio_data = priority_data
                else:
                    audio_data = priority_data
                await _broadcast_audio_optimized(audio_data)
            except Exception as e:
                print(f"[AUDIO] 工作线程错误: {e}")
    
    _worker_loop.run_until_complete(process_queue())

async def _broadcast_audio_optimized(pcm_data: bytes):
    """优化的音频广播：单次调用由底层按20ms节拍发送，移除重复节拍和Python层sleep"""
    global _last_play_ts, _is_playing
    try:
        # 设置播放标志
        with _playing_lock:
            _is_playing = True
        # 此时 pcm_data 应该已经是解压后的16位PCM数据了（8kHz）
        now = time.monotonic()
        idle_sec = now - (_last_play_ts or now)
        # 首次或长时间空闲后，预热更长静音；否则小静音
        lead_ms = 160 if idle_sec > 3.0 else 60
        tail_ms = 40

        lead_silence = b'\x00' * (lead_ms * 8000 * 2 // 1000)  # 8k * 2B
        tail_silence = b'\x00' * (tail_ms * 8000 * 2 // 1000)

        # 完整音频数据（包含静音）
        full_audio = lead_silence + pcm_data + tail_silence
        
        # 注意：录制在 broadcast_pcm16_realtime 中统一完成，避免重复

        # 单次调用交给底层 pacing（20ms节拍在 broadcast_pcm16_realtime 内部实现）
        await broadcast_pcm16_realtime(full_audio)

        _last_play_ts = time.monotonic()
    except Exception as e:
        print(f"[AUDIO] 广播音频失败: {e}")
    finally:
        # 清除播放标志
        with _playing_lock:
            _is_playing = False

def initialize_audio_system():
    """初始化音频系统"""
    global _initialized, _worker_thread, _last_play_ts
    
    if _initialized:
        return
    
    # 先合并 voice 映射，再预加载
    _merge_voice_map()
    preload_all_audio()
    
    _worker_thread = threading.Thread(target=_audio_worker, daemon=True)
    _worker_thread.start()
    _initialized = True
    _last_play_ts = 0.0
    
    # 显示压缩统计
    if os.getenv("AIGLASS_COMPRESS_AUDIO", "1") == "1":
        stats = compressed_audio_cache.get_compression_stats()
        print(f"[AUDIO] 音频压缩统计:")
        print(f"  - 文件数: {stats['files_cached']}")
        print(f"  - 原始大小: {stats['total_original_size'] / 1024:.1f} KB")
        print(f"  - 压缩后: {stats['total_compressed_size'] / 1024:.1f} KB")
        print(f"  - 压缩率: {stats['compression_ratio']:.1%}")
        print(f"  - 节省: {stats['bytes_saved'] / 1024:.1f} KB")
    
    print("[AUDIO] 音频系统初始化完成（预加载+工作线程）")


def play_audio_threadsafe(audio_key):
    """线程安全的音频播放函数"""
    global _audio_queue, _audio_priority
    
    if not _initialized:
        initialize_audio_system()
    
    if audio_key not in AUDIO_MAP:
        print(f"[AUDIO] 未知的音频键: {audio_key}")
        return
    
    filepath = AUDIO_MAP[audio_key]
    pcm_data = _audio_cache.get(filepath)
    if pcm_data is None:
        print(f"[AUDIO] 音频未在缓存中: {audio_key}")
        return
    
    # 如果是压缩的数据，先解压
    if pcm_data and len(pcm_data) > 5 and pcm_data[0] in [0x01, 0x02]:
        pcm_data = compressed_audio_cache.decompress(pcm_data)
        if not pcm_data:
            print(f"[AUDIO] 解压失败: {audio_key}")
            return
    
    _enqueue_pcm(pcm_data)

# 全局语音节流
_last_voice_time = 0
_last_voice_text = ""
_voice_cooldown = 1.0  # 相同语音至少间隔1秒

# 语音优先级定义
VOICE_PRIORITY = {
    'obstacle': 100,     # 障碍物 - 最高优先级
    'direction': 50,     # 转向/平移 - 中等优先级  
    'straight': 10,      # 保持直行 - 最低优先级
    'other': 30          # 其他 - 默认优先级
}


_CN_TO_EN: Dict[str, str] = {
    # -- obstacle --
    "前方有人，注意避让。": "Person ahead, watch out.",
    "前方有车，注意避让。": "Car ahead, watch out.",
    "前方有自行车，停一下。": "Bicycle ahead, stop.",
    "前方有摩托车，停一下。": "Motorcycle ahead, stop.",
    "前方有公交车，停一下。": "Bus ahead, stop.",
    "前方有卡车，停一下。": "Truck ahead, stop.",
    "前方有电瓶车，停一下。": "Scooter ahead, stop.",
    "前方有婴儿车，停一下。": "Stroller ahead, stop.",
    "前方有狗，停一下。": "Dog ahead, stop.",
    "前方有动物，停一下。": "Animal ahead, stop.",
    "前方有障碍物，注意避让。": "Obstacle ahead, watch out.",
    # -- no-path / stop --
    "请停下，前方无路": "Stop, no path ahead.",
    "请停下": "Please stop.",
    "前方无路": "No path ahead.",
    # -- direction / navigation --
    "保持直行": "Keep going straight.",
    "向左移动一点": "Move slightly to the left.",
    "向右移动一点": "Move slightly to the right.",
    "稍微向左靠一点": "Move a bit to the left.",
    "稍微向右靠一点": "Move a bit to the right.",
    "请向左微调方向": "Adjust slightly to the left.",
    "请向右微调方向": "Adjust slightly to the right.",
    "向左": "Move left.",
    "向右": "Move right.",
    "左转": "Turn left.",
    "右转": "Turn right.",
    "左移": "Shift left.",
    "右移": "Shift right.",
    # -- onboarding / calibration --
    "方向已对正！现在校准位置。": "Direction aligned! Now calibrating position.",
    "请向左转动。": "Please turn left.",
    "请向右转动。": "Please turn right.",
    "校准完成！您已在盲道上，开始前行。": "Calibration complete! You are on the path, start walking.",
    "请向左平移。": "Please shift left.",
    "请向右平移。": "Please shift right.",
    "请向前移动，让盲道更清晰。": "Move forward to get a clearer path.",
    # -- maneuvering / avoidance --
    "检测到已移动，开始对准新方向。": "Movement detected, aligning to new direction.",
    "已对准新路径，请向前直行。": "Aligned to new path, go straight ahead.",
    "好的，请停下侧移。": "OK, stop shifting.",
    "已回到盲道。": "Back on the path.",
    "向右平移，对准盲道": "Shift right to align with the path.",
    "向左平移，对准盲道": "Shift left to align with the path.",
    "丢失路径，重新搜索。": "Path lost, searching again.",
    "避让完成，已回到盲道。": "Avoidance complete, back on the path.",
    "向前直行几步越过障碍物。然后说'好了'。": "Walk a few steps forward past the obstacle.",
    "请继续向左平移。": "Keep shifting left.",
    "请继续向右平移。": "Keep shifting right.",
    "请向左微调，对准盲道。": "Adjust left to align with the path.",
    "请向右微调，对准盲道。": "Adjust right to align with the path.",
    # -- crosswalk --
    "已对准, 准备切换过马路模式。": "Aligned, ready to switch to crossing mode.",
    "绿灯稳定，开始通行。": "Green light stable, start crossing.",
    "正在等待绿灯…": "Waiting for green light.",
    "正在接近斑马线，为您对准方向。": "Approaching crosswalk, aligning direction.",
    "斑马线已对准，继续前行。": "Crosswalk aligned, keep going.",
    "斑马线到了可以过马路": "Crosswalk reached, you can cross now.",
    "远处发现斑马线，继续直行。": "Crosswalk spotted ahead, keep going straight.",
    # -- crosswalk awareness (with dynamic position) --
    "远处发现斑马线,在画面左侧": "Crosswalk spotted far ahead, on the left.",
    "远处发现斑马线,在画面中间": "Crosswalk spotted far ahead, in the center.",
    "远处发现斑马线,在画面右侧": "Crosswalk spotted far ahead, on the right.",
    "正在靠近斑马线,在画面左侧": "Approaching crosswalk, on the left.",
    "正在靠近斑马线,在画面中间": "Approaching crosswalk, in the center.",
    "正在靠近斑马线,在画面右侧": "Approaching crosswalk, on the right.",
    "接近斑马线,在画面左侧": "Near crosswalk, on the left.",
    "接近斑马线,在画面中间": "Near crosswalk, in the center.",
    "接近斑马线,在画面右侧": "Near crosswalk, on the right.",
    # -- crossing mode (from app_main) --
    "过马路模式已启动。": "Crossing mode started.",
    "启动过马路模式失败，请稍后重试。": "Failed to start crossing mode, try again later.",
    "已停止导航。": "Navigation stopped.",
    # -- traffic light --
    "红灯，请等待。": "Red light, please wait.",
    "绿灯，可以通行。": "Green light, you can go.",
    "黄灯，注意。": "Yellow light, be careful.",
}

_CN_SUBSTR_EN = [
    ("前方有", "注意避让", lambda mid: f"{mid} ahead, watch out."),
    ("前方有", "停一下", lambda mid: f"{mid} ahead, stop."),
    ("路径被挡住", "侧平移", lambda _: "Path blocked, shift to the side."),
    ("请继续向", "平移", lambda d: f"Keep shifting {d}."),
    ("请向", "微调，对准盲道", lambda d: f"Adjust {d} to align with the path."),
    ("请向", "平移。", lambda d: f"Please shift {d}."),
    ("请向", "转动。", lambda d: f"Please turn {d}."),
]

_CN_DIR = {"左": "left", "右": "right", "前": "forward"}


def _translate_for_mobile(text: str) -> str:
    """Translate known Chinese navigation phrases to English for iPhone TTS."""
    t = (text or "").strip()
    if not t:
        return t
    # 1) exact match
    if t in _CN_TO_EN:
        return _CN_TO_EN[t]
    base = t.rstrip("。.!！?？…")
    if base in _CN_TO_EN:
        return _CN_TO_EN[base]
    if base + "。" in _CN_TO_EN:
        return _CN_TO_EN[base + "。"]
    # 2) substring / pattern match for dynamic f-string phrases
    for start_kw, end_kw, fmt in _CN_SUBSTR_EN:
        if start_kw in t and end_kw in t:
            after_start = t.split(start_kw, 1)[1]
            mid_raw = after_start.split("，")[0].split(",")[0].split(end_kw)[0].strip()
            mid_en = _CN_DIR.get(mid_raw, mid_raw)
            return fmt(mid_en)
    # 3) no CJK → already English
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in t)
    if not has_cjk:
        return t
    # 4) last resort: keyword-based rough translation
    if "路径特征提取失败" in t:
        return ""
    if "前方无路" in t or "无路" in t:
        return "Stop, no path ahead."
    if "请停下" in t:
        return "Please stop."
    if "斑马线" in t and "发现" in t:
        return "Crosswalk spotted ahead."
    if "斑马线" in t and ("靠近" in t or "接近" in t):
        return "Approaching crosswalk."
    if "斑马线" in t and "过马路" in t:
        return "Crosswalk reached, you can cross now."
    if "斑马线" in t:
        return "Crosswalk detected."
    if "盲道" in t and "回到" in t:
        return "Back on the path."
    if "盲道" in t:
        return "Aligning to the path."
    if "直行" in t or "直走" in t:
        return "Keep going straight."
    if "向左" in t:
        return "Move left."
    if "向右" in t:
        return "Move right."
    if "左转" in t:
        return "Turn left."
    if "右转" in t:
        return "Turn right."
    if "左" in t and ("平移" in t or "移动" in t):
        return "Shift left."
    if "右" in t and ("平移" in t or "移动" in t):
        return "Shift right."
    if "停" in t:
        return "Please stop."
    if "避让" in t or "障碍" in t:
        return "Obstacle ahead, watch out."
    if "校准" in t:
        return "Calibrating position."
    if "对准" in t or "对正" in t:
        return "Aligning direction."
    if "绿灯" in t:
        return "Green light."
    if "红灯" in t:
        return "Red light, please wait."
    # truly unknown — return as-is (iOS auto-detects language)
    print(f"[AUDIO] untranslated CN text for mobile: {t}")
    return t


def _classify_tts_channel_and_priority(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if any(k in t for k in ("左转", "右转", "直行", "到达", "导航",
                            "Turn left", "Turn right", "straight", "aligned", "Calibration")):
        return {"channel": "navigation", "priority": 120}
    if any(k in t for k in ("障碍", "停一下", "避让", "危险",
                            "ahead", "stop", "watch out", "Obstacle")):
        return {"channel": "obstacle", "priority": 140}
    if any(k in t for k in ("盲道", "人行道", "斑马线", "过马路", "红灯", "绿灯", "黄灯",
                            "crosswalk", "Crosswalk", "crossing", "green light", "path")):
        return {"channel": "blind_path", "priority": 110}
    return {"channel": "server_tts", "priority": VOICE_PRIORITY.get("other", 30)}

# 新增：根据中文提示文案直接播放（会做轻度规范化与降级）
def play_voice_text(text: str):
    """
    传入中文提示，自动匹配 voice 映射并播放。
    - 尝试原文
    - 尝试补全/去除句末标点（。.!！?？）
    - 若包含“前方有…注意避让”但未命中，降级到“前方有障碍物，注意避让。”
    """
    global _last_voice_time, _last_voice_text

    if not text:
        return
    if not _initialized:
        initialize_audio_system()

    current_time = time.time()
    if text == _last_voice_text and current_time - _last_voice_time < _voice_cooldown:
        return

    def _forward(text_for_mobile: str, resolved_key: Optional[str] = None) -> bool:
        if not (_tts_forward_only and _tts_forwarder):
            return False
        en_text = _translate_for_mobile(text_for_mobile)
        cls = _classify_tts_channel_and_priority(en_text)
        payload = {
            "type": "tts_event",
            "text": en_text,
            "source": "server",
            "channel": cls["channel"],
            "priority": int(cls["priority"]),
            "timestamp": time.time(),
            "resolved_key": resolved_key or "",
        }
        try:
            _tts_forwarder(payload)
            return True
        except Exception as e:
            print(f"[AUDIO] 转发TTS到手机失败: {e}")
            return False

    norm_text = text.strip()
    if "路径特征提取失败" in norm_text:
        return
    if "向左移动" in norm_text:
        norm_text = "向左"
    elif "向右移动" in norm_text:
        norm_text = "向右"

    candidates = [norm_text]
    if norm_text[-1:] not in ("。", "！", "!", "？", "?", "."):
        candidates.append(norm_text + "。")
    else:
        t2 = norm_text.rstrip("。.!！?？")
        if t2 and t2 != norm_text:
            candidates.append(t2)

    for ck in candidates:
        if ck in AUDIO_MAP:
            if _forward(ck, resolved_key=ck):
                _last_voice_text = text
                _last_voice_time = current_time
                return
            play_audio_threadsafe(ck)
            _last_voice_text = text
            _last_voice_time = current_time
            return

    if ("前方有" in norm_text) and ("注意避让" in norm_text):
        fallback = "前方有障碍物，注意避让。"
        if fallback in AUDIO_MAP:
            if _forward(fallback, resolved_key=fallback):
                _last_voice_text = text
                _last_voice_time = current_time
                return
            play_audio_threadsafe(fallback)
            _last_voice_text = text
            _last_voice_time = current_time
            return

    base = norm_text.rstrip("。.!！?？")
    if base in AUDIO_MAP:
        if _forward(base, resolved_key=base):
            _last_voice_text = text
            _last_voice_time = current_time
            return
        play_audio_threadsafe(base)
        _last_voice_text = text
        _last_voice_time = current_time
        return
    if base + "。" in AUDIO_MAP:
        full = base + "。"
        if _forward(full, resolved_key=full):
            _last_voice_text = text
            _last_voice_time = current_time
            return
        play_audio_threadsafe(full)
        _last_voice_text = text
        _last_voice_time = current_time
        return

    # 无预录映射时，仍下发给 iPhone 端（由 iOS TTS 中心兜底合成）。
    if _forward(norm_text):
        _last_voice_text = text
        _last_voice_time = current_time
        return
    print(f"[AUDIO] 未找到匹配语音: {text}")

# 兼容旧接口
play_audio_on_esp32 = play_audio_threadsafe


def enqueue_external_pcm(pcm_data: bytes):
    """
    供云端 TTS 等模块使用：将 8kHz 单声道 PCM16 放入统一播放队列。
    """
    # 在“仅转发到手机”模式下，不在服务端播放云端 PCM。
    if _tts_forward_only:
        return
    _enqueue_pcm(pcm_data)