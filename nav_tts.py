# nav_tts.py
# 地图导航与可通行区域/红绿灯/斑马线播报的节流：地图播报时其他播报静音

import time
from audio_player import play_voice_text

# 地图最近一次播报时间（用于抑制其他 TTS）
_last_map_speech_time = 0.0
# 地图导航是否开启（由 app_main 同步）
_map_nav_active = False

# 地图播报后抑制其他播报的时长（秒）
MAP_SUPPRESS_OTHERS_SEC = 4.0

# 同一条导航语最短间隔（秒），避免一刻不停播报
_last_nav_text = ""
_last_nav_text_time = 0.0
NAV_SAME_TEXT_COOLDOWN_SEC = 5.0


def set_map_nav_active(active: bool) -> None:
    global _map_nav_active
    _map_nav_active = active


def mark_map_speech() -> None:
    """地图播报后调用，用于节流其他播报。"""
    global _last_map_speech_time
    _last_map_speech_time = time.time()


def should_suppress_nav_tts() -> bool:
    """地图导航开启且最近刚播报过时，应抑制可通行区域/红绿灯/斑马线播报。"""
    if not _map_nav_active:
        return False
    return (time.time() - _last_map_speech_time) < MAP_SUPPRESS_OTHERS_SEC


def play_voice_text_for_nav(text: str) -> None:
    """
    播放可通行区域/红绿灯/斑马线等导航相关 TTS。
    若地图导航正在播报或刚播报过，则静音（不播放）。
    同一条文案在 NAV_SAME_TEXT_COOLDOWN_SEC 内不重复播报。
    """
    if not (text and text.strip()):
        return
    if should_suppress_nav_tts():
        return
    global _last_nav_text, _last_nav_text_time
    now = time.time()
    if text.strip() == _last_nav_text and (now - _last_nav_text_time) < NAV_SAME_TEXT_COOLDOWN_SEC:
        return  # 同一条语在冷却期内不重复播
    _last_nav_text = text.strip()
    _last_nav_text_time = now
    play_voice_text(text)
