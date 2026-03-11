# app_main.py
# -*- coding: utf-8 -*-
import os, sys, time, json, asyncio, base64, audioop
from typing import Any, Dict, Optional, Tuple, List, Callable, Set, Deque
from collections import deque
from dataclasses import dataclass
import re
# 在其它 import 之后加：
from navigation_master import NavigationMaster, OrchestratorResult 
# 新增：导入盲道导航器
from workflow_blindpath import BlindPathNavigator
# 新增：导入过马路导航器
from workflow_crossstreet import CrossStreetNavigator
# 高德/谷歌地图导航（从 darksight 迁移）
from maps_navigation import (
    geocode_address,
    get_walking_route,
    precompute_route_polyline,
    match_position_to_route,
    get_instruction_for_position,
    get_display_info_for_position,
    get_navigation_api_key,
    get_navigation_provider,
    wgs84_to_gcj02,
    set_navigation_provider,
    OFF_ROUTE_DISTANCE_M,
    MAX_ACCURACY_FOR_OFF_ROUTE_M,
    ACCURACY_TOO_BAD_M,
    OFF_ROUTE_CONSECUTIVE_FOR_REPLAN,
    MIN_PROGRESS_BEFORE_OFF_ROUTE_M,
    REPLAN_COOLDOWN_MS,
)
from navigation_destination import parse_navigation_destination
import nav_tts
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
import cv2
import numpy as np
from obstacle_detector_client import ObstacleDetectorClient

import torch  # 添加这行
from device_utils import get_device


import bridge_io
import threading
# ---- Windows 事件循环策略 ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---- .env ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- 本地 Whisper ASR 配置（连续流式 + 唤醒词）----
SAMPLE_RATE  = 16000
CHUNK_MS     = 20
BYTES_CHUNK  = SAMPLE_RATE * CHUNK_MS // 1000 * 2

# Whisper 模型配置：优先使用环境变量，否则默认使用项目下 model/whisper 目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", os.path.join(BASE_DIR, "model", "whisper"))
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
_whisper_model = None
try:
    import whisper  # 需要: pip install -U openai-whisper
except Exception as e:
    whisper = None
    print(f"[ASR] 未安装 whisper，本地 ASR 将不可用: {e}")


def normalize_asr_text(text: str) -> str:
    """
    简单的英文 ASR 文本归一化：
    - 去首尾空格
    - 去掉结尾标点
    - 全部转小写
    - 去掉开头常见语气词（uh, um, well, you know, like, okay 等）
    """
    if not text:
        return ""
    t = text.strip()
    # 去掉结尾标点
    t = re.sub(r"[\.!?，。！？]+$", "", t)
    t = t.strip()
    t_lower = t.lower()
    # 去掉开头语气词
    filler_prefixes = [
        "uh ", "um ", "well ", "you know ", "like ",
        "okay ", "ok ", "so ", "uh,", "um,", "well,"
    ]
    changed = True
    while changed:
        changed = False
        for f in filler_prefixes:
            if t_lower.startswith(f):
                cut = len(f)
                t_lower = t_lower[cut:].lstrip()
                changed = True

    # 常见误识别容错映射（英文）
    corrections = {
        "star navigation": "start navigation",
    }
    if t_lower in corrections:
        t_lower = corrections[t_lower]

    return t_lower.strip()

# ---- 引入我们的模块 ----
from audio_stream import (
    register_stream_route,         # 挂 /stream.wav
    broadcast_pcm16_realtime,      # 实时向连接分发 16k PCM
    hard_reset_audio,              # 音频+AI 播放总闸
    BYTES_PER_20MS_16K,
    is_playing_now,
    current_ai_task,
)
from omni_client import stream_chat, OmniStreamPiece
from asr_core import (
    ASRCallback,
    set_current_recognition,
    stop_current_recognition,
)
from audio_player import initialize_audio_system, play_voice_text, play_audio_threadsafe
from cloud_nav_tts import synth_nav_tts_and_enqueue

# ---- 同步录制器 ----
import sync_recorder
import signal
import atexit

# ---- IMU UDP ----
UDP_IP   = "0.0.0.0"
UDP_PORT = 12345

app = FastAPI()

# ====== 状态与容器 ======
app.mount("/static", StaticFiles(directory="static"), name="static")

ui_clients: Dict[int, WebSocket] = {}
current_partial: str = ""
recent_finals: List[str] = []
RECENT_MAX = 50
last_frames: Deque[Tuple[float, bytes]] = deque(maxlen=10)
latest_mobile_nav: str = ""  # 最新手机导航指令

camera_viewers: Set[WebSocket] = set()
esp32_camera_ws: Optional[WebSocket] = None
imu_ws_clients: Set[WebSocket] = set()
esp32_audio_ws: Optional[WebSocket] = None

# 【新增】盲道 / 可通行区域导航相关全局变量
blind_path_navigator = None
navigation_active = False
# segformer 分割模型打包：{"processor": ..., "model": ..., "device": ...}
segformer_bundle = None
obstacle_detector = None

# 【高德/谷歌地图导航】从 darksight 迁移
latest_user_position: Optional[Tuple[float, float]] = None  # (lng, lat) WGS84
latest_user_position_accuracy: Optional[float] = None  # 米
map_nav_route = None  # AmapWalkingRoute
map_nav_polyline = None  # RoutePolylinePrecomputed
map_nav_active = False
map_nav_destination_text: str = ""
map_match_index = 0
map_match_s = 0.0
off_route_count = 0
last_replan_time_ms = 0.0
last_amap_spoken = ""
last_amap_spoken_time = 0.0
last_amap_step_index = 0
map_nav_poll_task: Optional[asyncio.Task] = None
AMAP_POLL_INTERVAL_SEC = 5.0

# 【新增】过马路导航相关全局变量
cross_street_navigator = None
cross_street_active = False
orchestrator = None  # 新增

# 【新增】omni对话状态标志
omni_conversation_active = False  # 标记omni对话是否正在进行
omni_previous_nav_state = None  # 保存omni激活前的导航状态，用于恢复

# 【新增】模型加载函数
def load_navigation_models():
    """加载导航相关模型：SegFormer 可通行区域分割 + 障碍物检测"""
    global segformer_bundle, obstacle_detector

    try:
        # 加载 SegFormer 语义分割模型（可通行区域）
        try:
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except Exception as e:
            print(f"[NAVIGATION] 导入 transformers/SegFormer 失败，无法加载可通行区域模型: {e}")
        else:
            segformer_dir = os.getenv(
                "SEGFORMER_MODEL_DIR",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "segformer-b1-ade"),
            )
            if os.path.exists(segformer_dir):
                try:
                    print(f"[NAVIGATION] 尝试从本地目录加载 SegFormer: {segformer_dir}")
                    processor = AutoImageProcessor.from_pretrained(segformer_dir)
                    model = SegformerForSemanticSegmentation.from_pretrained(segformer_dir)
                    dev = get_device()
                    model.to(dev)
                    segformer_bundle = {
                        "processor": processor,
                        "model": model,
                        "device": dev,
                    }
                    print(f"[NAVIGATION] SegFormer 分割模型加载成功，设备: {dev}")
                    print(f"[NAVIGATION] SegFormer 类别数: {model.config.num_labels}")
                except Exception as e:
                    segformer_bundle = None
                    print(f"[NAVIGATION] SegFormer 模型加载失败: {e}")
            else:
                print(f"[NAVIGATION] 未找到 SegFormer 模型目录: {segformer_dir}")
            
        # 【修改开始】使用 ObstacleDetectorClient 替代直接的 YOLO
        obstacle_model_path = os.getenv("OBSTACLE_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "yoloe-11l-seg.pt"))
        print(f"[NAVIGATION] 尝试加载障碍物检测模型: {obstacle_model_path}")
        
        if os.path.exists(obstacle_model_path):
            print(f"[NAVIGATION] 障碍物检测模型文件存在，开始加载...")
            try:
                # 使用 ObstacleDetectorClient 封装的 YOLO-E
                obstacle_detector = ObstacleDetectorClient(model_path=obstacle_model_path)
                print(f"[NAVIGATION] ========== YOLO-E 障碍物检测器加载成功 ==========")
                
                # 检查模型是否成功加载
                if hasattr(obstacle_detector, 'model') and obstacle_detector.model is not None:
                    print(f"[NAVIGATION] YOLO-E 模型已初始化")
                    print(f"[NAVIGATION] 模型设备: {next(obstacle_detector.model.parameters()).device}")
                else:
                    print(f"[NAVIGATION] 警告：YOLO-E 模型初始化异常")
                
                # 检查白名单是否成功加载
                if hasattr(obstacle_detector, 'WHITELIST_CLASSES'):
                    print(f"[NAVIGATION] 白名单类别数: {len(obstacle_detector.WHITELIST_CLASSES)}")
                    print(f"[NAVIGATION] 白名单前10个类别: {', '.join(obstacle_detector.WHITELIST_CLASSES[:10])}")
                else:
                    print(f"[NAVIGATION] 警告：白名单类别未定义")
                
                # 检查文本特征是否成功预计算
                if hasattr(obstacle_detector, 'whitelist_embeddings') and obstacle_detector.whitelist_embeddings is not None:
                    print(f"[NAVIGATION] YOLO-E 文本特征已预计算")
                    print(f"[NAVIGATION] 文本特征张量形状: {obstacle_detector.whitelist_embeddings.shape if hasattr(obstacle_detector.whitelist_embeddings, 'shape') else '未知'}")
                else:
                    print(f"[NAVIGATION] 警告：YOLO-E 文本特征未预计算")
                
                # 测试障碍物检测功能
                print(f"[NAVIGATION] 开始测试 YOLO-E 检测功能...")
                try:
                    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    # 在测试图像中画一个白色矩形，模拟一个物体
                    cv2.rectangle(test_img, (200, 200), (400, 400), (255, 255, 255), -1)
                    
                    # 测试检测（不提供 path_mask）
                    test_results = obstacle_detector.detect(test_img)
                    print(f"[NAVIGATION] YOLO-E 检测测试成功!")
                    print(f"[NAVIGATION] 测试检测结果数: {len(test_results)}")
                    
                    if len(test_results) > 0:
                        print(f"[NAVIGATION] 测试检测到的物体:")
                        for i, obj in enumerate(test_results):
                            print(f"  - 物体 {i+1}: {obj.get('name', 'unknown')}, "
                                  f"面积比例: {obj.get('area_ratio', 0):.3f}, "
                                  f"位置: ({obj.get('center_x', 0):.0f}, {obj.get('center_y', 0):.0f})")
                except Exception as e:
                    print(f"[NAVIGATION] YOLO-E 检测测试失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                print(f"[NAVIGATION] ========== YOLO-E 障碍物检测器加载完成 ==========")
                
            except Exception as e:
                print(f"[NAVIGATION] 障碍物检测器加载失败: {e}")
                import traceback
                traceback.print_exc()
                obstacle_detector = None
        else:
            print(f"[NAVIGATION] 警告：找不到障碍物检测模型文件: {obstacle_model_path}")
        
    except Exception as e:
        print(f"[NAVIGATION] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()

# 在程序启动时加载模型
print("[NAVIGATION] 开始加载导航模型...")
load_navigation_models()
print(f"[NAVIGATION] 模型加载完成 - SegFormer: {segformer_bundle is not None}")

# 【新增】启动同步录制
print("[RECORDER] 启动同步录制系统...")
sync_recorder.start_recording()
print("[RECORDER] 录制系统已启动，将自动保存视频和音频")

# 【新增】注册退出处理器，确保Ctrl+C时保存录制文件
def cleanup_on_exit():
    """程序退出时的清理工作"""
    print("\n[SYSTEM] 正在关闭录制器...")
    try:
        sync_recorder.stop_recording()
        print("[SYSTEM] 录制文件已保存")
    except Exception as e:
        print(f"[SYSTEM] 关闭录制器时出错: {e}")

def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    print("\n[SYSTEM] 收到中断信号，正在安全退出...")
    cleanup_on_exit()
    import sys
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
atexit.register(cleanup_on_exit)  # 正常退出时也调用

print("[RECORDER] 已注册退出处理器 - Ctrl+C时会自动保存录制文件")



# 【新增】预加载红绿灯检测模型（避免进入WAIT_TRAFFIC_LIGHT状态时卡顿）
try:
    import trafficlight_detection
    print("[TRAFFIC_LIGHT] 开始预加载红绿灯检测模型...")
    if trafficlight_detection.init_model():
        print("[TRAFFIC_LIGHT] 红绿灯检测模型预加载成功")
        # 执行一次测试推理，完全预热模型
        try:
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = trafficlight_detection.process_single_frame(test_img)
            print("[TRAFFIC_LIGHT] 模型预热完成")
        except Exception as e:
            print(f"[TRAFFIC_LIGHT] 模型预热失败: {e}")
    else:
        print("[TRAFFIC_LIGHT] 红绿灯检测模型预加载失败")
except Exception as e:
    print(f"[TRAFFIC_LIGHT] 红绿灯模型预加载出错: {e}")

# ============== 关键：系统级"硬重置"总闸 =================
interrupt_lock = asyncio.Lock()

# （手部识别/找物体已移除，保留变量避免其余代码报错）
yolomedia_running = False
yolomedia_sending_frames = False

async def ui_broadcast_raw(msg: str):
    dead = []
    for k, ws in list(ui_clients.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(k)
    for k in dead:
        ui_clients.pop(k, None)


async def ui_broadcast_partial(text: str):
    global current_partial
    current_partial = text
    await ui_broadcast_raw("PARTIAL:" + text)

async def ui_broadcast_final(text: str):
    global current_partial, recent_finals
    current_partial = ""
    recent_finals.append(text)
    if len(recent_finals) > RECENT_MAX:
        recent_finals = recent_finals[-RECENT_MAX:]
    await ui_broadcast_raw("FINAL:" + text)
    print(f"[ASR/AI FINAL] {text}", flush=True)

async def full_system_reset(reason: str = ""):
    """
    回到刚启动后的状态：
    1) 停播 + 取消AI任务 + 切断所有/stream.wav（hard_reset_audio）
    2) 停止 ASR 实时识别流（关键）
    3) 清 UI 状态
    4) 清最近相机帧（避免把旧帧又拼进下一轮）
    5) 告知 ESP32：RESET（可选）
    """
    # 1) 音频&AI
    await hard_reset_audio(reason or "full_system_reset")

    # 2) ASR
    await stop_current_recognition()

    # 3) UI
    global current_partial, recent_finals
    current_partial = ""
    recent_finals = []

    # 4) 相机帧
    try:
        last_frames.clear()
    except Exception:
        pass

    # 5) 通知 ESP32
    try:
        if esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
            await esp32_audio_ws.send_text("RESET")
    except Exception:
        pass

    print("[SYSTEM] full reset done.", flush=True)


async def start_map_navigation(destination: str):
    """请求步行路线并开启地图导航播报轮询（高德/谷歌）。"""
    global map_nav_route, map_nav_polyline, map_nav_active, map_nav_destination_text
    global map_match_index, map_match_s, off_route_count, last_replan_time_ms
    global last_amap_spoken, last_amap_spoken_time, last_amap_step_index, map_nav_poll_task
    global latest_user_position
    key = get_navigation_api_key()
    if not key:
        play_voice_text("Please set AMAP_WEB_SERVICE_KEY or GOOGLE_MAPS_API_KEY for map navigation.")
        return
    if not latest_user_position:
        play_voice_text("Please send your location from the phone first.")
        return
    lng, lat = latest_user_position
    origin = (lng, lat)
    dest_coord = None
    if re.match(r"^-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?$", destination.strip()):
        parts = destination.strip().split(",")
        dest_coord = (float(parts[0].strip()), float(parts[1].strip()))
    else:
        dest_coord = await geocode_address(destination, key)
    if not dest_coord:
        play_voice_text("Could not find the destination. Check address or network.")
        return
    provider = get_navigation_provider()
    if provider == "amap":
        origin = wgs84_to_gcj02(lng, lat)
    route = await get_walking_route(origin, dest_coord, key)
    if not route or not route.steps:
        play_voice_text("No walking route found.")
        return
    poly = precompute_route_polyline(route)
    map_nav_route = route
    map_nav_polyline = poly
    map_nav_active = True
    map_nav_destination_text = destination
    nav_tts.set_map_nav_active(True)
    map_match_index = 0
    map_match_s = 0.0
    off_route_count = 0
    last_replan_time_ms = time.time() * 1000
    last_amap_spoken = ""
    last_amap_spoken_time = 0.0
    last_amap_step_index = 0
    # 地图导航播报改为云端：仅推送文本给前端/手机，由云端TTS负责播放
    await ui_broadcast_final(f"[MAP] Start navigate to {destination}.")
    if map_nav_poll_task is None or map_nav_poll_task.done():
        map_nav_poll_task = asyncio.create_task(_map_nav_poll_loop())
    print(f"[MAP_NAV] Started to {destination}, steps={len(route.steps)}")


async def _map_nav_poll_loop():
    """轮询：用最新位置做 map matching + 指令播报 + 偏离重规划。"""
    global map_nav_route, map_nav_polyline, map_nav_active, map_match_index, map_match_s
    global off_route_count, last_replan_time_ms, last_amap_spoken, last_amap_spoken_time, last_amap_step_index
    global latest_user_position, latest_user_position_accuracy
    while map_nav_active and map_nav_route and map_nav_polyline:
        await asyncio.sleep(AMAP_POLL_INTERVAL_SEC)
        if not map_nav_active or not latest_user_position:
            continue
        try:
            lng, lat = latest_user_position
            raw_position = (lng, lat)
            provider = get_navigation_provider()
            if provider == "amap":
                lng, lat = wgs84_to_gcj02(lng, lat)
            position = (lng, lat)
            horizontal_accuracy_m = latest_user_position_accuracy
            route = map_nav_route
            poly = map_nav_polyline
            match = match_position_to_route(poly, map_match_index, map_match_s, position)
            if match.off_route_distance > OFF_ROUTE_DISTANCE_M:
                if horizontal_accuracy_m is not None and horizontal_accuracy_m > ACCURACY_TOO_BAD_M:
                    off_route_count = 0
                elif (
                    match.matched_s >= MIN_PROGRESS_BEFORE_OFF_ROUTE_M
                    and (horizontal_accuracy_m is None or horizontal_accuracy_m <= MAX_ACCURACY_FOR_OFF_ROUTE_M)
                ):
                    off_route_count += 1
            else:
                off_route_count = 0
            if off_route_count >= OFF_ROUTE_CONSECUTIVE_FOR_REPLAN:
                now_ms = time.time() * 1000
                if now_ms - last_replan_time_ms < REPLAN_COOLDOWN_MS:
                    off_route_count = 0
                    map_match_index = match.matched_index
                    map_match_s = match.matched_s
                else:
                    off_route_count = 0
                    last_replan_time_ms = now_ms
                    key = get_navigation_api_key()
                    if key:
                        if provider == "amap":
                            origin_for_route = wgs84_to_gcj02(raw_position[0], raw_position[1])
                        else:
                            origin_for_route = raw_position
                        new_route = await get_walking_route(origin_for_route, route.destination, key)
                        if new_route and new_route.steps:
                            map_nav_route = new_route
                            map_nav_polyline = precompute_route_polyline(new_route)
                            map_match_index = 0
                            map_match_s = 0.0
                            # 仅推送文本，交给云端TTS/手机播报
                            await ui_broadcast_final("[MAP] Route recalculated.")
                    continue
            map_match_index = match.matched_index
            map_match_s = match.matched_s
            position_for_display = match.matched_point
            now_ms = time.time() * 1000
            result = get_instruction_for_position(
                route,
                position_for_display,
                last_amap_spoken,
                last_amap_spoken_time,
                last_amap_step_index,
            )
            if not result:
                continue
            last_amap_spoken = result.text
            last_amap_spoken_time = now_ms
            last_amap_step_index = result.step_index
            # 地图指令文本推送给前端/手机
            await ui_broadcast_final(f"[MAP] {result.text}")

            # 进入斑马线 / 红绿灯相关模式时，暂停地图语音播报，仅更新内部地图状态
            if orchestrator:
                try:
                    cur_state = orchestrator.get_state()
                except Exception:
                    cur_state = None
                if cur_state in ("CROSSING", "WAIT_TRAFFIC_LIGHT", "TRAFFIC_LIGHT_DETECTION"):
                    continue

            # 使用 Qwen 云端 TTS 合成，并放入统一音频队列
            try:
                await synth_nav_tts_and_enqueue(result.text)
                nav_tts.mark_map_speech()
            except Exception as e:
                print(f"[MAP_NAV] cloud TTS failed: {e}")

            if result.kind == "arrived":
                map_nav_active = False
                map_nav_route = None
                map_nav_polyline = None
                nav_tts.set_map_nav_active(False)
                await ui_broadcast_final("[MAP] You have arrived at your destination.")
                return
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[MAP_NAV] poll error: {e}")


# ========= 手部识别/找物体已移除，保留空函数避免调用报错 =========
def start_yolomedia_with_target(target_name: str):
    pass

def stop_yolomedia():
    pass

# ========= 自定义的 start_ai_with_text，支持识别特殊命令 =========
async def start_ai_with_text_custom(user_text: str):
    """扩展版的AI启动函数，支持识别特殊命令"""
    global navigation_active, blind_path_navigator, cross_street_active, cross_street_navigator, orchestrator
    text_lower = (user_text or "").lower()
    
    # 在导航 / 红绿灯模式下，只有特定英文关键词才进入 omni 对话
    if orchestrator:
        current_state = orchestrator.get_state()
        # 如果在导航模式或红绿灯检测模式（非CHAT模式）
        if current_state not in ["CHAT", "IDLE"]:
            # 允许进入对话的大致英文触发词
            allowed_keywords = ["please check", "could you see", "what is this", "recognize"]
            is_allowed_query = any(k in text_lower for k in allowed_keywords)
            
            # 导航控制相关英文关键词
            nav_control_keywords = [
                "start crossing", "end crossing",
                "start navigation", "blind path", "stop navigation", "end navigation",
                "start traffic light", "check traffic light", "stop traffic light",
            ]
            is_nav_control = any(k in text_lower for k in nav_control_keywords)
            
            # 如果既不是允许的查询，也不是导航控制命令，则丢弃
            if not is_allowed_query and not is_nav_control:
                mode_name = "traffic-light" if current_state == "TRAFFIC_LIGHT_DETECTION" else "navigation"
                print(f"[{mode_name} mode] discard non-dialog utterance: {user_text}")
                return  # 直接丢弃，不进入omni
    
    # English: crossing commands
    if "start crossing" in text_lower or "help me cross" in text_lower:
        if orchestrator:
            orchestrator.start_crossing()
            print(f"[CROSS_STREET] crossing mode started, state: {orchestrator.get_state()}")
            play_voice_text("Crossing mode started.")
            await ui_broadcast_final("[SYSTEM] Crossing mode started.")
        else:
            print("[CROSS_STREET] WARNING: navigation orchestrator not initialized!")
            play_voice_text("Failed to start crossing mode. Please try again later.")
            await ui_broadcast_final("[SYSTEM] Navigation system is not ready.")
        return
    
    if "end crossing" in text_lower or "finish crossing" in text_lower:
        if orchestrator:
            orchestrator.stop_navigation()
            print(f"[CROSS_STREET] crossing stopped, state: {orchestrator.get_state()}")
            play_voice_text("Navigation stopped.")
            await ui_broadcast_final("[SYSTEM] Crossing mode stopped.")
        else:
            await ui_broadcast_final("[SYSTEM] Navigation system is not running.")
        return
    
    # 【兼容旧中文】检查是否是过马路相关命令 - 使用orchestrator控制
    if "开始过马路" in user_text or "帮我过马路" in user_text:
        if orchestrator:
            orchestrator.start_crossing()
            print(f"[CROSS_STREET] 过马路模式已启动，状态: {orchestrator.get_state()}")
            # 播放启动语音并广播到UI
            play_voice_text("过马路模式已启动。")
            await ui_broadcast_final("[系统] 过马路模式已启动")
        else:
            print("[CROSS_STREET] 警告：导航统领器未初始化！")
            play_voice_text("启动过马路模式失败，请稍后重试。")
            await ui_broadcast_final("[系统] 导航系统未就绪")
        return
    
    if "过马路结束" in user_text or "结束过马路" in user_text:
        if orchestrator:
            orchestrator.stop_navigation()
            print(f"[CROSS_STREET] 导航已停止，状态: {orchestrator.get_state()}")
            # 播放停止语音并广播到UI
            play_voice_text("已停止导航。")
            await ui_broadcast_final("[系统] 过马路模式已停止")
        else:
            await ui_broadcast_final("[系统] 导航系统未运行")
        return
    
    # English: traffic light detection commands
    if "start traffic light" in text_lower or "check traffic light" in text_lower:
        try:
            import trafficlight_detection
            
            if orchestrator:
                orchestrator.start_traffic_light_detection()
                print(f"[TRAFFIC] switch to traffic-light mode, state: {orchestrator.get_state()}")
            
            success = trafficlight_detection.init_model()
            trafficlight_detection.reset_detection_state()
            
            if success:
                await ui_broadcast_final("[SYSTEM] Traffic light detection started.")
            else:
                await ui_broadcast_final("[SYSTEM] Failed to load traffic light model.")
        except Exception as e:
            print(f"[TRAFFIC] failed to start traffic-light detection: {e}")
            await ui_broadcast_final(f"[SYSTEM] Failed to start: {e}")
        return
    
    if "stop traffic light" in text_lower:
        try:
            if orchestrator:
                orchestrator.stop_navigation()
                print(f"[TRAFFIC] traffic-light detection stopped, back to {orchestrator.get_state()} mode")
            await ui_broadcast_final("[SYSTEM] Traffic light detection stopped.")
        except Exception as e:
            print(f"[TRAFFIC] failed to stop traffic-light detection: {e}")
            await ui_broadcast_final(f"[SYSTEM] Failed to stop: {e}")
        return
    
    # 【兼容旧中文】检查是否是红绿灯检测命令 - 实现与盲道导航互斥
    if "检测红绿灯" in user_text or "看红绿灯" in user_text:
        try:
            import trafficlight_detection
            
            # 切换orchestrator到红绿灯检测模式（暂停盲道导航）
            if orchestrator:
                orchestrator.start_traffic_light_detection()
                print(f"[TRAFFIC] 切换到红绿灯检测模式，状态: {orchestrator.get_state()}")
            
            # 【改进】使用主线程模式而不是独立线程，避免掉帧
            success = trafficlight_detection.init_model()  # 只初始化模型，不启动线程
            trafficlight_detection.reset_detection_state()  # 重置状态
            
            if success:
                await ui_broadcast_final("[系统] 红绿灯检测已启动")
            else:
                await ui_broadcast_final("[系统] 红绿灯模型加载失败")
        except Exception as e:
            print(f"[TRAFFIC] 启动红绿灯检测失败: {e}")
            await ui_broadcast_final(f"[系统] 启动失败: {e}")
        return
    
    if "停止检测" in user_text or "停止红绿灯" in user_text:
        try:
            # 恢复到对话模式
            if orchestrator:
                orchestrator.stop_navigation()  # 回到CHAT模式
                print(f"[TRAFFIC] 红绿灯检测停止，恢复到{orchestrator.get_state()}模式")
            
            await ui_broadcast_final("[系统] 红绿灯检测已停止")
        except Exception as e:
            print(f"[TRAFFIC] 停止红绿灯检测失败: {e}")
            await ui_broadcast_final(f"[系统] 停止失败: {e}")
        return
    
    # 【高德/谷歌】「导航到 XXX」：解析目的地并请求步行路线，需手机先上报位置
    dest = parse_navigation_destination(user_text or "")
    if dest:
        if latest_user_position is None:
            play_voice_text("Please send your location from the phone to start navigation.")
            await ui_broadcast_final("[SYSTEM] Send location from phone (e.g. JSON with lat, lng) to start map navigation.")
            return
        asyncio.create_task(start_map_navigation(dest))
        # 同时开启可通行区域导航（地图+盲道一起）
        if orchestrator:
            orchestrator.start_blind_path_navigation()
            print(f"[NAVIGATION] map + blind-path started to {dest}, state: {orchestrator.get_state()}")
        await ui_broadcast_final(f"[SYSTEM] Map and blind-path navigation started to {dest}.")
        return
    
    # English: navigation commands（仅可通行区域，无目的地）
    if "start navigation" in text_lower or "blind path" in text_lower or "help me navigate" in text_lower:
        if orchestrator:
            orchestrator.start_blind_path_navigation()
            print(f"[NAVIGATION] blind-path navigation started, state: {orchestrator.get_state()}")
            await ui_broadcast_final("[SYSTEM] Blind-path navigation started.")
        else:
            print("[NAVIGATION] WARNING: navigation orchestrator not initialized!")
            await ui_broadcast_final("[SYSTEM] Navigation system is not ready.")
        return
    
    if "stop navigation" in text_lower or "end navigation" in text_lower:
        if orchestrator:
            orchestrator.stop_navigation()
            print(f"[NAVIGATION] navigation stopped, state: {orchestrator.get_state()}")
            await ui_broadcast_final("[SYSTEM] Blind-path navigation stopped.")
        else:
            await ui_broadcast_final("[SYSTEM] Navigation system is not running.")
        # 若正在地图导航，也一并停止
        global map_nav_active
        if map_nav_active:
            map_nav_active = False
            nav_tts.set_map_nav_active(False)
            play_voice_text("Map navigation stopped.")
        return
    
    # English: extra navigation commands delegated to orchestrator
    nav_cmd_keywords = [
        "start crossing", "end crossing",
        "start navigation", "blind path", "stop navigation", "end navigation",
        "go now", "cross now", "continue",
    ]
    if any(k in text_lower for k in nav_cmd_keywords):
        if orchestrator:
            orchestrator.on_voice_command(user_text)
            await ui_broadcast_final("[SYSTEM] Navigation mode updated.")
        else:
            await ui_broadcast_final("[SYSTEM] Navigation orchestrator is not initialized.")
        return    

    # 【修改】omni对话开始时，切换到CHAT模式
    global omni_conversation_active, omni_previous_nav_state
    omni_conversation_active = True
    
    # 保存当前导航状态并切换到CHAT模式
    if orchestrator:
        current_state = orchestrator.get_state()
        # 只有在导航模式下才需要保存和切换
        if current_state not in ["CHAT", "IDLE"]:
            omni_previous_nav_state = current_state
            orchestrator.force_state("CHAT")
            print(f"[OMNI] 对话开始，从{current_state}切换到CHAT模式")
        else:
            omni_previous_nav_state = None
            print(f"[OMNI] 对话开始（当前已在{current_state}模式）")
    
    # 如果不是特殊命令，执行原有的AI对话逻辑
    await start_ai_with_text(user_text)

# ========= Omni 播放启动 =========
async def start_ai_with_text(user_text: str):
    """硬重置后，开启新的 AI 语音输出。"""
    async def _runner():
        txt_buf: List[str] = []
        rate_state = None

        # 组装（图像+文本）
        content_list = []
        if last_frames:
            try:
                _, jpeg_bytes = last_frames[-1]
                img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            except Exception:
                pass
        content_list.append({"type": "text", "text": user_text})

        try:
            async for piece in stream_chat(content_list, voice="Cherry", audio_format="wav"):
                # 文本增量（仅 UI）
                if piece.text_delta:
                    txt_buf.append(piece.text_delta)
                    try:
                        await ui_broadcast_partial("[AI] " + "".join(txt_buf))
                    except Exception:
                        pass

                # 音频分片：Omni 返回 24k (PCM16) 的 wav audio.data（Base64）；下行需要 8k PCM16
                if piece.audio_b64:
                    try:
                        pcm24 = base64.b64decode(piece.audio_b64)
                    except Exception:
                        pcm24 = b""
                    if pcm24:
                        # 24k → 8k (使用ratecv保证音调和速度不变)
                        pcm8k, rate_state = audioop.ratecv(pcm24, 2, 1, 24000, 8000, rate_state)
                        pcm8k = audioop.mul(pcm8k, 2, 0.60)
                        if pcm8k:
                            await broadcast_pcm16_realtime(pcm8k)

        except asyncio.CancelledError:
            # 被新一轮打断
            raise
        except Exception as e:
            try:
                await ui_broadcast_final(f"[AI] 发生错误：{e}")
            except Exception:
                pass
        finally:
            # 【修改】标记omni对话结束，恢复之前的导航模式
            global omni_conversation_active, omni_previous_nav_state
            omni_conversation_active = False
            
            # 恢复之前的导航状态
            if orchestrator and omni_previous_nav_state:
                orchestrator.force_state(omni_previous_nav_state)
                print(f"[OMNI] 对话结束，恢复到{omni_previous_nav_state}模式")
                omni_previous_nav_state = None
            else:
                print(f"[OMNI] 对话结束（无需恢复导航状态）")
            
            # 自然结束时，给当前连接一个 "完结" 信号
            from audio_stream import stream_clients  # 局部导入，避免环依赖
            for sc in list(stream_clients):
                if not sc.abort_event.is_set():
                    try: sc.q.put_nowait(b"\x00"*BYTES_PER_20MS_16K)  # 一帧静音
                    except Exception: pass
                    try: sc.q.put_nowait(None)
                    except Exception: pass

            final_text = ("".join(txt_buf)).strip() or "（空响应）"
            try:
                await ui_broadcast_final("[AI] " + final_text)
            except Exception:
                pass

    # 真正启动前先硬重置，保证**绝无**旧音频残留
    await hard_reset_audio("start_ai_with_text")
    loop = asyncio.get_running_loop()
    from audio_stream import current_ai_task as _task_holder  # 读写模块内全局
    from audio_stream import __dict__ as _as_dict
    # 设置模块内的 current_ai_task
    task = loop.create_task(_runner())
    _as_dict["current_ai_task"] = task

# ---------- 页面 / 健康 ----------
@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/phone", response_class=HTMLResponse)
def phone_player():
    """手机端语音播报页：打开后自动连接 /stream_phone.wav 播放 AI 语音。"""
    with open(os.path.join("templates", "phone_player.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"


@app.post("/api/command")
async def api_command(request: Request):
    """网页指令输入：与语音控制使用相同的关键词匹配逻辑，适用于所有状态。"""
    print("[API] POST /api/command received", flush=True)
    try:
        body = await request.json()
        text = (body.get("text") or "").strip()
        if not text:
            print("[API] command rejected: text is empty", flush=True)
            return {"ok": False, "error": "text is empty"}
        print(f"[API] executing command: {text!r}", flush=True)
        await start_ai_with_text_custom(text)
        print("[API] command done", flush=True)
        return {"ok": True, "text": text}
    except Exception as e:
        print(f"[API] command error: {e}", flush=True)
        return {"ok": False, "error": str(e)}


# 注册 /stream.wav
register_stream_route(app)

# ---------- WebSocket：WebUI 文本（ASR/AI 状态推送） ----------
@app.websocket("/ws_ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    ui_clients[id(ws)] = ws
    try:
        init = {
            "partial": current_partial,
            "finals": recent_finals[-10:],
            "mobile_nav": latest_mobile_nav,
        }
        await ws.send_text("INIT:" + json.dumps(init, ensure_ascii=False))
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        ui_clients.pop(id(ws), None)

SILENCE_20MS = b"\x00" * BYTES_PER_20MS_16K

# ---------- WebSocket：ESP32 音频入口（本地 Whisper，按键分段） ----------
@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    global esp32_audio_ws, _whisper_model
    esp32_audio_ws = ws
    await ws.accept()
    print("\n[AUDIO] client connected (local Whisper segment)")

    streaming = False
    pcm_chunks: List[bytes] = []

    async def stop_rec(send_notice: Optional[str] = None):
        """本地 ASR 模式下，仅重置状态并可选通知设备。"""
        nonlocal streaming
        streaming = False
        pcm_chunks.clear()
        await set_current_recognition(None)
        if send_notice:
            try:
                await ws.send_text(send_notice)
            except Exception:
                pass

    try:
        while True:
            if WebSocketState and ws.client_state != WebSocketState.CONNECTED:
                break
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if "Cannot call \"receive\"" in str(e):
                    break
                raise

            if "text" in msg and msg["text"] is not None:
                raw = (msg["text"] or "").strip()
                cmd = raw.upper()

                if cmd == "START":
                    print("[AUDIO] START received (local Whisper segment)")
                    await stop_rec()
                    streaming = True
                    await ui_broadcast_partial("（已开始接收音频…）")
                    try:
                        await ws.send_text("OK:STARTED")
                    except Exception:
                        pass

                elif cmd == "STOP":
                    print("[AUDIO] STOP received (local Whisper segment)")
                    streaming = False
                    full_pcm = b"".join(pcm_chunks)
                    pcm_chunks.clear()

                    # PCM 为空或 Whisper 不可用，直接结束
                    if whisper is None or not full_pcm:
                        await stop_rec(send_notice="OK:STOPPED")
                    else:
                        async def run_whisper_and_handle(pcm: bytes):
                            global _whisper_model
                            try:
                                # 简单时长检查：至少 0.5 秒再识别，避免 0 长度错误
                                if len(pcm) < int(SAMPLE_RATE * 2 * 0.5):
                                    await ui_broadcast_final("[ASR] Utterance too short, please press the button and speak again.")
                                    return

                                if _whisper_model is None:
                                    print(f"[WHISPER] 加载模型: {WHISPER_MODEL_NAME} (dir={WHISPER_MODEL_DIR})")
                                    os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)
                                    _whisper_model = whisper.load_model(
                                        WHISPER_MODEL_NAME,
                                        download_root=WHISPER_MODEL_DIR,
                                    )
                                import numpy as _np
                                audio_np = _np.frombuffer(pcm, dtype=_np.int16).astype(_np.float32) / 32768.0
                                loop = asyncio.get_running_loop()
                                result = await loop.run_in_executor(
                                    None,
                                    lambda: _whisper_model.transcribe(audio_np, language="en")
                                )
                                raw_text = (result.get("text") or "").strip()
                                norm_text = normalize_asr_text(raw_text)
                                if norm_text:
                                    await ui_broadcast_final(f"[ASR] {norm_text}")
                                    async with interrupt_lock:
                                        await start_ai_with_text_custom(norm_text)
                                else:
                                    await ui_broadcast_final("[ASR] Could not understand, please press the button and speak again.")
                            except Exception as e:
                                print(f"[WHISPER] 识别失败: {e}")
                                try:
                                    await ui_broadcast_final(f"[系统] 本地语音识别失败: {e}")
                                except Exception:
                                    pass
                            finally:
                                await stop_rec(send_notice="OK:STOPPED")

                        asyncio.create_task(run_whisper_and_handle(full_pcm))

                elif raw.startswith("PROMPT:"):
                    text = raw[len("PROMPT:"):].strip()
                    if text:
                        async with interrupt_lock:
                            await start_ai_with_text_custom(text)
                        await ws.send_text("OK:PROMPT_ACCEPTED")
                    else:
                        await ws.send_text("ERR:EMPTY_PROMPT")

            elif "bytes" in msg and msg["bytes"] is not None:
                if streaming and msg["bytes"]:
                    pcm_chunks.append(msg["bytes"])

    except Exception as e:
        print(f"\n[WS ERROR] {e}")
    finally:
        await stop_rec()
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            esp32_audio_ws = None
        print("[WS] connection closed")

# ---------- WebSocket：手机导航指令入口（ws://.../ws/mobile-nav） ----------
@app.websocket("/ws/mobile-nav")
async def ws_mobile_nav(ws: WebSocket):
    global latest_mobile_nav, latest_user_position, latest_user_position_accuracy
    await ws.accept()
    print("[MOBILE_NAV] client connected")
    try:
        while True:
            msg = await ws.receive_text()
            raw = (msg or "").strip()
            if not raw:
                continue
            # 支持 JSON 上报位置：
            # 1) {"lat": 31.2, "lng": 121.5, "accuracy": 10}
            # 2) {"location": {"lat": 31.2, "lng": 121.5}, "accuracy": 10}
            # 3) {"position": {"lat": 31.2, "lng": 121.5, "accuracy": 10, ...}, "type": "nav_state", ...}（手机导航完整状态）
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    # 兼容多种字段名：location / position / 顶层 lat/lng
                    loc = data.get("location") or data.get("position") or data
                    if isinstance(loc, dict):
                        lat = loc.get("lat")
                        lng = loc.get("lng")
                        acc = loc.get("accuracy", data.get("accuracy"))
                    else:
                        lat = data.get("lat")
                        lng = data.get("lng")
                        acc = data.get("accuracy")
                    if lat is not None and lng is not None:
                        latest_user_position = (float(lng), float(lat))
                        latest_user_position_accuracy = acc
                        try:
                            await ui_broadcast_raw("MOBILE_NAV:location " + str(latest_user_position))
                        except Exception:
                            pass
                        continue
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            latest_mobile_nav = raw
            try:
                await ui_broadcast_raw("MOBILE_NAV:" + latest_mobile_nav)
            except Exception:
                pass
    except WebSocketDisconnect:
        print("[MOBILE_NAV] disconnected")
    except Exception as e:
        print(f"[MOBILE_NAV] error: {e}")

# ---------- WebSocket：ESP32 相机入口（JPEG 二进制） ----------
@app.websocket("/ws/camera")
async def ws_camera_esp(ws: WebSocket):
    global esp32_camera_ws, blind_path_navigator, cross_street_navigator, cross_street_active, navigation_active, orchestrator
    if esp32_camera_ws is not None:
        await ws.close(code=1013)
        return
    esp32_camera_ws = ws
    await ws.accept()
    print("[CAMERA] ESP32 connected")
    
    # 【新增】初始化盲道导航器
    if blind_path_navigator is None and segformer_bundle is not None:
        blind_path_navigator = BlindPathNavigator(segformer_bundle, obstacle_detector)
        print("[NAVIGATION] 可通行区域导航器已初始化 (SegFormer)")
    else:
        if blind_path_navigator is not None:
            print("[NAVIGATION] 导航器已存在，无需重新初始化")
        elif segformer_bundle is None:
            print("[NAVIGATION] 警告：SegFormer 模型未加载，无法初始化导航器")
    
    # 【新增】初始化过马路导航器
    if cross_street_navigator is None:
        if segformer_bundle:
            # TODO: 将来如需在过马路导航中使用 SegFormer，可在这里接入；
            # 当前暂时保持简化：只要有分割模型，就初始化一个最小 navigator 占位。
            cross_street_navigator = CrossStreetNavigator(
                seg_model=None,
                coco_model=None,
                obs_model=None
            )
            print("[CROSS_STREET] 过马路导航器已初始化（占位，尚未使用 SegFormer）")
        else:
            print("[CROSS_STREET] 错误：缺少分割模型，无法初始化过马路导航器")
    
    if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
        orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)
        print("[NAV MASTER] 统领状态机已初始化（托管模式）")
    frame_counter = 0  # 添加帧计数器
    
    # 用队列解耦「接收」和「处理」：单独任务持续从 WS 收帧，主循环从队列取帧处理，
    # 避免处理（SegFormer/导航）阻塞时 TCP 缓冲区满导致 ESP32 send 失败并断连
    frame_queue = asyncio.Queue(maxsize=3)
    
    async def drain_camera_ws():
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") in ("websocket.close", "websocket.disconnect"):
                    try:
                        frame_queue.put_nowait(None)
                    except asyncio.QueueFull:
                        frame_queue.get_nowait()
                        frame_queue.put_nowait(None)
                    return
                if msg.get("bytes"):
                    try:
                        frame_queue.put_nowait(msg["bytes"])
                    except asyncio.QueueFull:
                        frame_queue.get_nowait()
                        frame_queue.put_nowait(msg["bytes"])
        except WebSocketDisconnect:
            try:
                frame_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        except Exception as e:
            print(f"[CAMERA] drain task error: {e}")
            try:
                frame_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
    
    drain_task = asyncio.create_task(drain_camera_ws())
    
    try:
        while True:
            data = await frame_queue.get()
            if data is None:
                break
            frame_counter += 1

            # 【新增】录制原始帧
            try:
                sync_recorder.record_frame(data)
            except Exception as e:
                if frame_counter % 100 == 0:  # 避免日志刷屏
                    print(f"[RECORDER] 录制帧失败: {e}")

            try:
                last_frames.append((time.time(), data))
            except Exception:
                pass

            # 推送到bridge_io（供其它模块使用）
            bridge_io.push_raw_jpeg(data)

            # 【调试】检查导航条件
            if frame_counter % 30 == 0:
                state_dbg = orchestrator.get_state() if orchestrator else "N/A"
                print(f"[NAVIGATION DEBUG] 帧:{frame_counter}, state={state_dbg}")

            # 统一解码（添加更严格的异常处理）
            try:
                arr = np.frombuffer(data, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                # 验证解码结果
                if bgr is None or bgr.size == 0:
                    if frame_counter % 30 == 0:
                        print(f"[JPEG] 解码失败：数据长度={len(data)}")
                    bgr = None
            except Exception as e:
                if frame_counter % 30 == 0:
                    print(f"[JPEG] 解码异常: {e}")
                bgr = None

            # 【托管】交给统领状态机处理画面
            if orchestrator and bgr is not None:
                current_state = orchestrator.get_state()
                out_img = bgr
                try:
                    # 红绿灯检测模式：在主线程中直接处理
                    if current_state == "TRAFFIC_LIGHT_DETECTION":
                        import trafficlight_detection
                        result = trafficlight_detection.process_single_frame(bgr, ui_broadcast_callback=ui_broadcast_final)
                        out_img = result['vis_image'] if result['vis_image'] is not None else bgr
                    else:
                        # 其他模式：正常的导航处理
                        res = orchestrator.process_frame(bgr)
                        if res.guidance_text:
                            try:
                                nav_tts.play_voice_text_for_nav(res.guidance_text)
                                await ui_broadcast_final(f"[导航] {res.guidance_text}")
                            except Exception:
                                pass
                        out_img = res.annotated_image if res.annotated_image is not None else bgr
                except Exception as e:
                    if frame_counter % 100 == 0:
                        print(f"[NAV MASTER] 处理帧时出错: {e}")

                # 广播图像
                if camera_viewers and out_img is not None:
                    ok, enc = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        jpeg_data = enc.tobytes()
                        dead = []
                        for viewer_ws in list(camera_viewers):
                            try:
                                await viewer_ws.send_bytes(jpeg_data)
                            except Exception:
                                dead.append(viewer_ws)
                        for d in dead:
                            camera_viewers.discard(d)
                # 已托管，进入下一帧
                continue

            # 【回退】未解码成功或无统领器时，按原始画面回传
            if camera_viewers:
                try:
                    if bgr is None:
                        arr = np.frombuffer(data, dtype=np.uint8)
                        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            jpeg_data = enc.tobytes()
                            dead = []
                            for viewer_ws in list(camera_viewers):
                                try:
                                    await viewer_ws.send_bytes(jpeg_data)
                                except Exception:
                                    dead.append(viewer_ws)
                            for ws in dead:
                                camera_viewers.discard(ws)
                except Exception as e:
                    print(f"[CAMERA] Broadcast error: {e}")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CAMERA ERROR] {e}")
    finally:
        drain_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        esp32_camera_ws = None
        print("[CAMERA] ESP32 disconnected")

# ---------- WebSocket：浏览器订阅相机帧 ----------
@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    await ws.accept()
    camera_viewers.add(ws)
    print(f"[VIEWER] Browser connected. Total viewers: {len(camera_viewers)}", flush=True)
    try:
        while True:
            # 保持连接活跃
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("[VIEWER] Browser disconnected", flush=True)
    finally:
        try: 
            camera_viewers.remove(ws)
        except Exception: 
            pass
        print(f"[VIEWER] Removed. Total viewers: {len(camera_viewers)}", flush=True)

# ---------- WebSocket：浏览器订阅 IMU ----------
@app.websocket("/ws")
async def ws_imu(ws: WebSocket):
    await ws.accept()
    imu_ws_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        imu_ws_clients.discard(ws)

async def imu_broadcast(msg: str):
    if not imu_ws_clients: return
    dead = []
    for ws in list(imu_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        imu_ws_clients.discard(ws)

# ---------- 服务端 IMU 估计（原样保留） ----------
from math import atan2, hypot, pi
GRAV_BETA   = 0.98
STILL_W     = 0.4
YAW_DB      = 0.08
YAW_LEAK    = 0.2
ANG_EMA     = 0.15
AUTO_REZERO = True
USE_PROJ    = True
FREEZE_STILL= True
G     = 9.807
A_TOL = 0.08 * G
gLP = {"x":0.0, "y":0.0, "z":0.0}
gOff= {"x":0.0, "y":0.0, "z":0.0}
BIAS_ALPHA = 0.002
yaw  = 0.0
Rf = Pf = Yf = 0.0
ref = {"roll":0.0, "pitch":0.0, "yaw":0.0}
holdStart = 0.0
isStill   = False
last_ts_imu = 0.0
last_wall = 0.0
imu_store: List[Dict[str, Any]] = []

def _wrap180(a: float) -> float:
    a = a % 360.0
    if a >= 180.0: a -= 360.0
    if a < -180.0: a += 360.0
    return a

def process_imu_and_maybe_store(d: Dict[str, Any]):
    global gLP, gOff, yaw, Rf, Pf, Yf, ref, holdStart, isStill, last_ts_imu, last_wall

    t_ms = float(d.get("ts", 0.0))
    now_wall = time.monotonic()
    if t_ms <= 0.0:
        t_ms = (now_wall * 1000.0)
    if last_ts_imu <= 0.0 or t_ms <= last_ts_imu or (t_ms - last_ts_imu) > 3000.0:
        dt = 0.02
    else:
        dt = (t_ms - last_ts_imu) / 1000.0
    last_ts_imu = t_ms

    ax = float(((d.get("accel") or {}).get("x", 0.0)))
    ay = float(((d.get("accel") or {}).get("y", 0.0)))
    az = float(((d.get("accel") or {}).get("z", 0.0)))
    wx = float(((d.get("gyro")  or {}).get("x", 0.0)))
    wy = float(((d.get("gyro")  or {}).get("y", 0.0)))
    wz = float(((d.get("gyro")  or {}).get("z", 0.0)))

    gLP["x"] = GRAV_BETA * gLP["x"] + (1.0 - GRAV_BETA) * ax
    gLP["y"] = GRAV_BETA * gLP["y"] + (1.0 - GRAV_BETA) * ay
    gLP["z"] = GRAV_BETA * gLP["z"] + (1.0 - GRAV_BETA) * az
    gmag = hypot(gLP["x"], gLP["y"], gLP["z"]) or 1.0
    gHat = {"x": gLP["x"]/gmag, "y": gLP["y"]/gmag, "z": gLP["z"]/gmag}

    roll  = (atan2(az, ay)   * 180.0 / pi)
    pitch = (atan2(-ax, ay)  * 180.0 / pi)

    aNorm = hypot(ax, ay, az); wNorm = hypot(wx, wy, wz)
    nearFlat = (abs(roll) < 2.0 and abs(pitch) < 2.0)
    stillCond = (abs(aNorm - G) < A_TOL) and (wNorm < STILL_W)

    if stillCond:
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0: isStill = True
        gOff["x"] = (1.0 - BIAS_ALPHA)*gOff["x"] + BIAS_ALPHA*wx
        gOff["y"] = (1.0 - BIAS_ALPHA)*gOff["y"] + BIAS_ALPHA*wy
        gOff["z"] = (1.0 - BIAS_ALPHA)*gOff["z"] + BIAS_ALPHA*wz
    else:
        holdStart = 0.0; isStill = False

    if USE_PROJ:
        yawdot = ((wx - gOff["x"])*gHat["x"] + (wy - gOff["y"])*gHat["y"] + (wz - gOff["z"])*gHat["z"])
    else:
        yawdot = (wy - gOff["y"])

    if abs(yawdot) < YAW_DB: yawdot = 0.0
    if FREEZE_STILL and stillCond: yawdot = 0.0

    yaw = _wrap180(yaw + yawdot * dt)

    if (YAW_LEAK > 0.0) and nearFlat and stillCond and abs(yaw) > 0.0:
        step = YAW_LEAK * dt * (-1.0 if yaw > 0 else (1.0 if yaw < 0 else 0.0))
        if abs(yaw) <= abs(step): yaw = 0.0
        else: yaw += step

    global Rf, Pf, Yf, ref, last_wall
    Rf = ANG_EMA * roll  + (1.0 - ANG_EMA) * Rf
    Pf = ANG_EMA * pitch + (1.0 - ANG_EMA) * Pf
    Yf = ANG_EMA * yaw   + (1.0 - ANG_EMA) * Yf

    if AUTO_REZERO and nearFlat and (wNorm < STILL_W):
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0:
            ref.update({"roll": Rf, "pitch": Pf, "yaw": Yf})
            isStill = True

    R = _wrap180(Rf - ref["roll"])
    P = _wrap180(Pf - ref["pitch"])
    Y = _wrap180(Yf - ref["yaw"])

    now_wall = time.monotonic()
    if last_wall <= 0.0 or (now_wall - last_wall) >= 0.100:
        last_wall = now_wall
        item = {
            "ts": t_ms/1000.0,
            "angles": {"roll": R, "pitch": P, "yaw": Y},
            "accel":  {"x": ax, "y": ay, "z": az},
            "gyro":   {"x": wx, "y": wy, "z": wz},
        }
        imu_store.append(item)

# ---------- UDP 接收 IMU 并转发 ----------
class UDPProto(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        print(f"[UDP] listening on {UDP_IP}:{UDP_PORT}")
    def datagram_received(self, data, addr):
        try:
            s = data.decode('utf-8', errors='ignore').strip()
            d = json.loads(s)
            if 'ts' not in d and 'timestamp_ms' in d:
                d['ts'] = d.pop('timestamp_ms')
            process_imu_and_maybe_store(d)
            asyncio.create_task(imu_broadcast(json.dumps(d)))
        except Exception:
            pass



# === 新增：注册给 bridge_io 的发送回调（把 JPEG 广播给 /ws/viewer） ===
@app.on_event("startup")
async def on_startup_register_bridge_sender():
    # 保存主线程的事件循环
    main_loop = asyncio.get_event_loop()
    
    def _sender(jpeg_bytes: bytes):
        try:
            if main_loop.is_closed():
                return
            async def _broadcast():
                if not camera_viewers:
                    return
                dead = []
                for ws in list(camera_viewers):
                    try:
                        await ws.send_bytes(jpeg_bytes)
                    except Exception as e:
                        dead.append(ws)
                for ws in dead:
                    try:
                        camera_viewers.discard(ws)
                    except Exception:
                        pass
            asyncio.run_coroutine_threadsafe(_broadcast(), main_loop)
        except Exception as e:
            if "Event loop is closed" not in str(e):
                print(f"[DEBUG] _sender error: {e}", flush=True)

    bridge_io.set_sender(_sender)

@app.on_event("startup")
async def on_startup_init_audio():
    """启动时初始化音频系统"""
    # 在后台线程中初始化，避免阻塞启动
    def _init():
        try:
            initialize_audio_system()
        except Exception as e:
            print(f"[AUDIO] 初始化失败: {e}")
    
    threading.Thread(target=_init, daemon=True).start()

@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(lambda: UDPProto(), local_addr=(UDP_IP, UDP_PORT))

@app.on_event("shutdown")
async def on_shutdown():
    """应用关闭时的清理工作"""
    print("[SHUTDOWN] 开始清理资源...")
    
    # 停止YOLO媒体处理
    stop_yolomedia()
    
    # 停止音频和AI任务
    await hard_reset_audio("shutdown")
    
    print("[SHUTDOWN] 资源清理完成")

# app_main.py —— 在文件里已有的 @app.on_event("startup") 之后，再加一个新的 startup 钩子


# --- 导出接口（可选） ---
def get_last_frames():
    return last_frames

def get_camera_ws():
    return esp32_camera_ws

if __name__ == "__main__":
    uvicorn.run(
        app, host="172.20.10.10", port=8081,
        log_level="warning", access_log=False,
        loop="asyncio", workers=1, reload=False
    )
