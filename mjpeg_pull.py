# mjpeg_pull.py
# 从 HTTP MJPEG 流（如 ESP32 CameraWebServer /stream）拉取帧并推入 asyncio.Queue + bridge_io
# 用法：AIGLASS_CAMERA_SOURCE=mjpeg 且 AIGLASS_MJPEG_URL=http://<ESP32_IP>:81/stream 时由 app_main 启动
# 参考：docs/CAMERA_STREAMING_ALTERNATIVES.md

import threading
import time
from typing import Optional, Dict, Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"


def pull_mjpeg_worker(
    url: str,
    packet_queue: "asyncio.Queue",
    loop: "asyncio.AbstractEventLoop",
    stop_event: threading.Event,
    push_bridge_io: bool = True,
) -> None:
    """
    在后台线程中拉取 MJPEG，解析出每帧 JPEG，通过 loop.call_soon_threadsafe 放入 packet_queue。
    若 push_bridge_io 为 True，每帧同时调用 bridge_io.push_raw_jpeg。
    """
    import bridge_io
    buf = b""
    reconnect_delay = 2.0
    while not stop_event.is_set():
        try:
            req = Request(url, headers={"User-Agent": "AIGlass-MJPEG-Pull/1.0"})
            with urlopen(req, timeout=10) as f:
                while not stop_event.is_set():
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        a = buf.find(SOI)
                        b = buf.find(EOI)
                        if a == -1 or b == -1 or b < a:
                            if len(buf) > 2 * 1024 * 1024:
                                buf = buf[-1024 * 1024 :]
                            break
                        jpg = buf[a : b + len(EOI)]
                        buf = buf[b + len(EOI) :]
                        recv_ts = time.time()
                        packet = {"data": jpg, "recv_ts": recv_ts, "capture_ts_ms": None}
                        if push_bridge_io:
                            try:
                                bridge_io.push_raw_jpeg(jpg)
                            except Exception:
                                pass
                        try:
                            loop.call_soon_threadsafe(packet_queue.put_nowait, packet)
                        except Exception:
                            pass
        except (URLError, HTTPError, OSError) as e:
            if not stop_event.is_set():
                print(f"[MJPEG-PULL] {e}, reconnect in {reconnect_delay}s")
        except Exception as e:
            if not stop_event.is_set():
                print(f"[MJPEG-PULL] error: {e}")
        if not stop_event.is_set():
            time.sleep(reconnect_delay)


def start_mjpeg_pull(
    url: str,
    packet_queue: "asyncio.Queue",
    loop: "asyncio.AbstractEventLoop",
) -> threading.Event:
    """
    启动 MJPEG 拉流线程；返回 stop_event，用于在关闭时停止线程。
    """
    stop_event = threading.Event()
    t = threading.Thread(
        target=pull_mjpeg_worker,
        args=(url, packet_queue, loop, stop_event),
        kwargs={"push_bridge_io": True},
        daemon=True,
    )
    t.start()
    return stop_event
