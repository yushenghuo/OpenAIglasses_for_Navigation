# iOS TTS 调度中心接入说明

本文档用于 `darksight` iOS 端接入服务端统一语音事件（`/ws/mobile-nav`）。

## 1. WebSocket 协议

服务端会在 `ws://<server>/ws/mobile-nav` 推送两类消息：

- `TTS_CAPABILITY:<json>`
- `TTS_EVENT:<json>`

### 1.1 TTS_CAPABILITY

```json
{
  "type": "tts_capability",
  "version": 1,
  "mode": "iphone_scheduler",
  "channels": ["navigation", "blind_path", "obstacle", "server_tts"]
}
```

### 1.2 TTS_EVENT

```json
{
  "type": "tts_event",
  "channel": "navigation",
  "source": "server",
  "text": "Turn right in 20 meters.",
  "priority": 120,
  "interrupt": false,
  "timestamp": 1710000000000,
  "dedupe_key": "map_step_4_turn_approach"
}
```

字段约定：

- `channel`: `navigation | blind_path | obstacle | server_tts`
- `priority`: 数值越大优先级越高（建议阈值：`obstacle > navigation > blind_path > server_tts`）
- `interrupt`: 是否允许打断低优先级语音
- `dedupe_key`: 去重键（短时间内重复消息丢弃）

## 2. iOS 调度中心建议

建议在 iOS 侧新增一个 `TTSSchedulerCenter`，并由它统一决定是否播报：

- **节流**：最短播报间隔 1.5-2.5s
- **去重**：同 `dedupe_key` 在 8-12s 内不重复
- **优先级抢占**：高优先级可打断低优先级
- **合并**：两条导航消息在 1-2s 内可合并成一句

## 3. Swift 参考实现（可直接迁移到 darksight）

```swift
import AVFoundation
import Foundation

struct TTSEvent: Decodable {
    let type: String
    let channel: String
    let source: String
    let text: String
    let priority: Int
    let interrupt: Bool
    let timestamp: Int64
    let dedupe_key: String?
}

final class TTSSchedulerCenter {
    static let shared = TTSSchedulerCenter()

    private let synthesizer = AVSpeechSynthesizer()
    private var queue: [TTSEvent] = []
    private var lastSpeakAt: TimeInterval = 0
    private var recentKeys: [String: TimeInterval] = [:]
    private let minInterval: TimeInterval = 1.8
    private let dedupeWindow: TimeInterval = 10.0

    func enqueue(_ event: TTSEvent) {
        guard shouldAccept(event) else { return }

        if event.interrupt, let current = currentPriority(), event.priority > current {
            synthesizer.stopSpeaking(at: .immediate)
        }

        queue.append(event)
        queue.sort { lhs, rhs in
            if lhs.priority != rhs.priority { return lhs.priority > rhs.priority }
            return lhs.timestamp < rhs.timestamp
        }
        drainIfPossible()
    }

    private func shouldAccept(_ event: TTSEvent) -> Bool {
        let now = Date().timeIntervalSince1970
        if let key = event.dedupe_key, let ts = recentKeys[key], now - ts < dedupeWindow {
            return false
        }
        if let key = event.dedupe_key {
            recentKeys[key] = now
        }
        return true
    }

    private func currentPriority() -> Int? {
        return nil
    }

    private func drainIfPossible() {
        let now = Date().timeIntervalSince1970
        guard !queue.isEmpty else { return }
        guard now - lastSpeakAt >= minInterval else { return }
        guard !synthesizer.isSpeaking else { return }

        let event = queue.removeFirst()
        let utter = AVSpeechUtterance(string: event.text)
        utter.voice = AVSpeechSynthesisVoice(language: "en-US")
        utter.rate = 0.47
        synthesizer.speak(utter)
        lastSpeakAt = now
    }
}
```

> 建议在 `AVSpeechSynthesizerDelegate` 回调 `didFinish` 中继续调用 `drainIfPossible()`，实现队列连续播报。

## 4. WebSocket 接收示例（darksight）

```swift
func handleWebSocketText(_ raw: String) {
    if raw.hasPrefix("TTS_EVENT:") {
        let json = String(raw.dropFirst("TTS_EVENT:".count))
        if let data = json.data(using: .utf8),
           let event = try? JSONDecoder().decode(TTSEvent.self, from: data) {
            TTSSchedulerCenter.shared.enqueue(event)
        }
    }
}
```

## 5. 与服务端职责边界

- **服务端**：保留现有 TTS 节流/优先级，统一下发 `TTS_EVENT`
- **iOS 端**：最终播报仲裁（导航/盲道/障碍物 + 本地业务语音）

这样可保证即使未来在 iOS 新增地图导航语音，也能通过同一调度中心统一管理。
