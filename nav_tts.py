# nav_tts.py
# 地图导航与可通行区域/红绿灯/斑马线播报：地图开启时抑制其它导航 TTS；
# 导航语音串行入队（见 audio_player）；抑制左右/转向指令短时来回跳。

import os
import time
from typing import Optional

from audio_player import play_voice_text

# 地图最近一次播报时间（用于抑制其他 TTS）
_last_map_speech_time = 0.0
# 地图导航是否开启（由 app_main 同步）
_map_nav_active = False

# 地图播报后抑制其他播报的时长（秒）
MAP_SUPPRESS_OTHERS_SEC = 4.0

# 相反横向/转向指令最短间隔（秒），避免「左—右—左」抖动播报
_NAV_MIN_FLIP_SEC = float(os.getenv("AIGLASS_NAV_MIN_LATERAL_FLIP_SEC", "2.0"))

_last_committed_bucket: Optional[str] = None
_last_bucket_commit_time: float = 0.0


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


def _nav_action_bucket(text: str) -> Optional[str]:
    """粗分导航动作桶，用于防左右/转向对跳。非方向类返回 None。"""
    t = (text or "").strip()
    if not t:
        return None
    if "左转" in t:
        return "turn_l"
    if "右转" in t:
        return "turn_r"
    lat_l_kw = (
        "左移", "向左平", "请向左平", "请继续向左", "向左微", "稍向左",
        "向左移动", "稍微向左", "请向左微",
    )
    lat_r_kw = (
        "右移", "向右平", "请向右平", "请继续向右", "向右微", "稍向右",
        "向右移动", "稍微向右", "请向右微",
    )
    if any(k in t for k in lat_l_kw) or t == "向左":
        return "lat_l"
    if any(k in t for k in lat_r_kw) or t == "向右":
        return "lat_r"
    return None


def _is_opposite_bucket(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b or a == b:
        return False
    pairs = {frozenset({"lat_l", "lat_r"}), frozenset({"turn_l", "turn_r"})}
    return frozenset({a, b}) in pairs


def _should_suppress_flip(new_bucket: Optional[str]) -> bool:
    global _last_committed_bucket, _last_bucket_commit_time
    if new_bucket is None:
        return False
    now = time.time()
    if _is_opposite_bucket(new_bucket, _last_committed_bucket):
        if (now - _last_bucket_commit_time) < _NAV_MIN_FLIP_SEC:
            return True
    return False


def _commit_bucket(bucket: Optional[str]) -> None:
    global _last_committed_bucket, _last_bucket_commit_time
    if bucket is None:
        return
    _last_committed_bucket = bucket
    _last_bucket_commit_time = time.time()


def play_voice_text_for_nav(text: str) -> bool:
    """
    播放可通行区域/红绿灯/斑马线等导航相关 TTS。
    地图导航正在播报或刚播报过时静音；短时内禁止与上一句相反的左/右或转向对跳。
    使用 nav_serial 入队，保证本地播放时播完再接下一条。
    返回是否已成功下发（与 play_voice_text 一致）。
    """
    if not (text and text.strip()):
        return False
    if should_suppress_nav_tts():
        return False

    b = _nav_action_bucket(text)
    if _should_suppress_flip(b):
        return False

    ok = play_voice_text(text, nav_serial=True)
    if ok:
        if b is None:
            _release_flip_after_neutral_play()
        else:
            _commit_bucket(b)
    return bool(ok)


def reset_nav_flip_state() -> None:
    """盲道导航重置时可调用，避免跨会话残留桶状态。"""
    global _last_committed_bucket, _last_bucket_commit_time
    _last_committed_bucket = None
    _last_bucket_commit_time = 0.0


def _release_flip_after_neutral_play() -> None:
    """直行/停/障碍等非横向句播完后释放对向锁，避免接下来合法的「左/右」仍被旧桶挡住。"""
    global _last_committed_bucket, _last_bucket_commit_time
    _last_committed_bucket = None
    _last_bucket_commit_time = 0.0
