# 相机 UDP 传输（ESP32 → Python）

## 开关对应关系

| 组件 | 说明 |
|------|------|
| 固件 `compile/compile.ino` **`CAMERA_USE_UDP_ONLY`** | `1` = 仅 UDP，不连相机 WebSocket；`0` = 恢复 `/ws/camera` |
| 环境变量 **`AIGLASS_CAMERA_UDP`** | `1`（默认）= 在 **UDP 18500** 收流并走导航管线；`0` = 仅 WebSocket 相机 |

两者需一致：UDP 模式固件 `1` + `AIGLASS_CAMERA_UDP=1`；纯 WebSocket 则固件 `0` + `AIGLASS_CAMERA_UDP=0`。

## 协议

- 小端 `uint16`：`frame_id`、`chunk_id`、`total_chunks`，后跟最多 **1024** 字节 JPEG 片段（见 `CAM_UDP_CHUNK_PAYLOAD`）。

## 运行

1. `SERVER_HOST` 指向运行 `app_main.py` 的机器 IP。
2. 本机放行 **UDP 18500**（与 IMU 的 12345 不同端口）。
3. `python3 app_main.py`，浏览器仍用 **`/ws/viewer`** 看画面。

## 可选 Node 桥接

`tools/udp_camera_bridge/` 可单独用 Node 收 UDP 并做 MJPEG/WS 预览（不经过 Python），与 Python **不要同时**占用 18500。
