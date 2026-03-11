# 项目运行方式与 Mac + 手机热点 + 手机播报 改造说明

## 一、项目整体说明

本项目是一个**面向视障人士的智能眼镜导航与辅助系统**，由三部分组成：

1. **服务端（本机 Python）**：在 MacBook 上运行的 `app_main.py`，提供 HTTP/WebSocket 服务，负责视频处理、语音识别、AI 对话、导航逻辑和语音合成/流式语音下发。
2. **ESP32-CAM 眼镜端**：摄像头、麦克风、扬声器、IMU，通过 WiFi 连接服务端，上传视频/音频、接收 TTS 播放（可选关闭）。
3. **浏览器监控端**：访问 `http://<服务端IP>:8081` 查看实时画面、状态、IMU 等。

---

## 二、代码运行方式（数据流）

### 2.1 启动入口

- **唯一入口**：`python app_main.py`
- 会加载 `.env`（含 `DASHSCOPE_API_KEY`）、导航模型（盲道 yolo-seg、障碍物 yoloe）、红绿灯/物品检测等，然后启动 **FastAPI + uvicorn**，监听 `0.0.0.0:8081`，UDP 监听 `0.0.0.0:12345`（IMU）。

### 2.2 视频流

```
ESP32 摄像头 → WebSocket /ws/camera (JPEG)
  → bridge_io.push_raw_jpeg()
  → 若未在寻物：orchestrator.process_frame()（盲道/过马路/红绿灯等）
  → 若在寻物：yolomedia 线程
  → 处理后帧经 bridge_io 回调 → WebSocket /ws/viewer → 浏览器 Canvas
```

### 2.3 音频上行（语音识别）

```
ESP32 麦克风 (PCM16 16kHz) → WebSocket /ws_audio
  → asr_core（DashScope Paraformer 实时 ASR）
  → 识别结果 → start_ai_with_text_custom()
  → 解析指令：导航/过马路/红绿灯/找物/对话
```

### 2.4 音频下行（TTS / AI 语音播报）——当前与改造后

**当前逻辑：**

- TTS 预录（`audio_player.play_voice_text`）或 Qwen-Omni 流式语音 → 转为 8kHz 单声道 PCM16  
- → `broadcast_pcm16_realtime()`（`audio_stream.py`）  
- → 只发给**当前连接 `GET /stream.wav` 的客户端**，即 **ESP32**  
- ESP32 用 HTTP 长连接拉取 `/stream.wav`，I2S 扬声器播放  

**改造后（语音只返回手机）：**

- 新增 `GET /stream_phone.wav`，仅供**手机**连接（如 darksight app 或手机浏览器）。
- `broadcast_pcm16_realtime()` 改为只向 `stream_phone` 客户端推送（或通过配置选择推送到 ESP32 / 手机 / 两者）。
- ESP32 固件增加**关闭 TTS 播放**的选项，不再连接 `/stream.wav`，语音仅在手机端播放。

### 2.5 IMU

```
ESP32 → UDP 发到 服务端 IP:12345
  → process_imu_and_maybe_store()
  → WebSocket /ws → 前端 visualizer.js (Three.js) 显示姿态
```

---

## 三、ESP32 与服务端的交互方式

| 方式 | 端点 | 方向 | 说明 |
|------|------|------|------|
| WebSocket | `/ws/camera` | ESP32 → 服务端 | 推送 JPEG 视频帧 |
| WebSocket | `/ws_audio` | ESP32 → 服务端 | 推送 PCM16 麦克风；可收文本指令 START/STOP/RESET 等 |
| HTTP | `GET /stream.wav` | 服务端 → ESP32 | 当前：TTS/Omni 语音流，ESP32 I2S 播放；改造后可关闭 |
| HTTP | `GET /stream_phone.wav` | 服务端 → 手机 | 新增：仅手机连接，用于手机播报 |
| UDP | `服务端IP:12345` | ESP32 → 服务端 | IMU 数据 |

**ESP32 固件内需配置的常量（`compile/compile.ino`）：**

- `WIFI_SSID` / `WIFI_PASS`：WiFi 热点名称与密码  
- `SERVER_HOST` / `SERVER_PORT`：服务端 IP 与端口（如 MacBook 在手机热点下的 IP、8081）  
- `UDP_HOST` / `UDP_PORT`：与上面一致（同机 12345）  

---

## 四、在 MacBook Air 上运行需要的修改

### 4.1 模型与资源路径（改为项目相对路径）

Mac 上没有 `C:\Users\...`，需统一改为**项目根目录下的相对路径**（如 `model/xxx.pt`），这样在任意机器上都能跑。具体修改见下文「六、具体修改清单」。

### 4.2 环境与依赖

- Python 3.9–3.11，`pip install -r requirements.txt`
- 模型文件放到项目下 `model/` 目录（见 README 与 ModelScope 链接）
- `.env` 中配置 `DASHSCOPE_API_KEY`
- MacBook 无 NVIDIA GPU 时，PyTorch 使用 CPU（会较慢，但可跑）

### 4.3 手机热点 + MacBook + ESP32 组网

1. 手机开热点，MacBook 和 ESP32 都连该热点。  
2. 在 MacBook 上查看本机在该热点下的 IP，例如：
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```
   记下类似 `192.168.x.x` 或 `10.0.0.x` 的地址。  
3. 在 `compile/compile.ino` 中：
   - `WIFI_SSID` / `WIFI_PASS` = 手机热点名称与密码  
   - `SERVER_HOST` = **MacBook 的 IP**（上一步得到的）  
   - `SERVER_PORT` = 8081  
   - `UDP_HOST` = 同上，`UDP_PORT` = 12345  
4. 重新烧录 ESP32 固件。  
5. Mac 防火墙放行 8081（TCP）和 12345（UDP）。

---

## 五、语音只返回手机、由 darksight 播放的设计

### 5.1 目标

- 语音（TTS + Omni 流式回复）**不**再发到 ESP32 扬声器，而是**只**发到手机。  
- 手机通过 **darksight app**（或你指定的播放方式）播放 `http://<MacBook_IP>:8081/stream_phone.wav`。

### 5.2 服务端

- **新增** `GET /stream_phone.wav`：  
  - 与 `/stream.wav` 协议一致（WAV 头 + 8kHz 单声道 PCM16 chunked 流），但使用**独立的连接集合** `stream_clients_phone`。  
- **广播逻辑**：  
  - 增加配置项 `AUDIO_OUTPUT`：`esp32` | `phone` | `both`。  
  - `phone`：只向 `stream_clients_phone` 推送（即只给连接 `/stream_phone.wav` 的客户端，通常是手机）。  
  - `esp32`：保持原样，只向 `stream_clients`（即连接 `/stream.wav` 的 ESP32）推送。  
  - `both`：同时推送到两边。  
- 你当前需求设为 `phone` 即可。

### 5.3 ESP32

- 增加编译开关（如 `#define ENABLE_TTS_PLAYBACK 0`）。  
- 当为 0 时：**不**调用 `startStreamWav()`，即不连接 `GET /stream.wav`，不占用下行带宽，也不在眼镜端播放。  
- 当为 1 时：保持现有行为（连接 `/stream.wav` 并 I2S 播放）。

### 5.4 手机 / darksight app

- **方式 A**：darksight app 支持“播放网络音频流”时，直接播放 URL：  
  `http://<MacBook_IP>:8081/stream_phone.wav`  
  （&lt;MacBook_IP&gt; 与 ESP32 里填的 `SERVER_HOST` 一致。）  
- **方式 B**：若暂无 app，可用**手机浏览器**打开播报页：  
  `http://<MacBook_IP>:8081/phone`  
  该页面会自动连接 `/stream_phone.wav` 并播放，适合临时使用或给 darksight 做参考。  
- 注意：`/stream_phone.wav` 是**长连接、持续流**，有语音时才有数据；播放端需支持流式 WAV 或 chunked 音频。

### 5.5 流程小结

1. 手机开热点，MacBook 与 ESP32 连上；MacBook 运行 `python app_main.py`。  
2. 服务端 `.env` 设置 `AUDIO_OUTPUT=phone`。  
3. ESP32 固件设置 `ENABLE_TTS_PLAYBACK 0` 并烧录，不再连接 `/stream.wav`。  
4. 手机连接同一热点后，darksight app（或我们提供的播放页）打开 `http://<MacBook_IP>:8081/stream_phone.wav` 并播放。  
5. 用户对眼镜说话 → ESP32 上传 `/ws_audio` → ASR → AI → TTS/Omni → 只推送到 `stream_phone` → 手机播放。

---

## 六、具体修改清单（摘要）

- **`audio_stream.py`**：新增 `stream_clients_phone`、`/stream_phone.wav` 路由；`broadcast_pcm16_realtime()` 根据 `AUDIO_OUTPUT` 选择推送到 `stream_clients` 和/或 `stream_clients_phone`；`hard_reset_audio` 同时清空 phone 连接。  
- **`app_main.py`**：无逻辑改动，仅依赖 `audio_stream` 的注册；模型/资源路径改为项目相对路径（见下）。  
- **`compile/compile.ino`**：`WIFI_SSID`/`WIFI_PASS`/`SERVER_HOST`/`UDP_HOST` 改为你的热点与 MacBook IP；增加 `ENABLE_TTS_PLAYBACK`，为 0 时不调用 `startStreamWav()`。  
- **Mac 默认路径**：`app_main.py`、`audio_player.py`、`yolomedia.py`、`trafficlight_detection.py`、`yoloe_backend.py` 中默认路径改为基于项目根目录的 `model/...`、`music/`、`voice/` 等，保证在 Mac 上直接可运行。  
- **可选**：在 `templates/` 下提供 `phone_player.html`，用 `<audio>` 播放 `/stream_phone.wav`；并增加路由 `GET /phone` 供手机浏览器打开。

按上述修改后，即可在 MacBook 上通过手机热点连接 ESP32，并将语音仅返回到手机由 darksight app 播放。

---

## 七、快速检查清单（Mac + 手机热点 + 手机播报）

| 步骤 | 位置 | 操作 |
|------|------|------|
| 1 | MacBook | 手机开热点，Mac 连上，记下 Mac 的 IP（如 192.168.43.xxx） |
| 2 | .env | 添加 AUDIO_OUTPUT=phone（语音只发手机） |
| 3 | compile.ino | WIFI_SSID/PASS = 热点；SERVER_HOST/UDP_HOST = Mac 的 IP；ENABLE_TTS_PLAYBACK = 0 |
| 4 | ESP32 | 烧录固件，上电后连热点并连上服务端 |
| 5 | MacBook | 运行 python app_main.py，防火墙放行 8081、12345 |
| 6 | 手机 | 浏览器打开 http://Mac的IP:8081/phone，或 darksight app 播放 stream_phone.wav |
