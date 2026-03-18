import io
import os
import re
import shutil
import audioop
import urllib.request
import wave
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

import dashscope


ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = ROOT / "voice"


def load_env_from_dotenv() -> None:
    dotenv_path = ROOT / ".env"
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_PUNCT_RE = re.compile(r"[，,。.!！?？…]+$")


def _strip_punct(s: str) -> str:
    s = (s or "").strip()
    s = _PUNCT_RE.sub("", s)
    return s.strip()


def zh_to_en_full(text_zh: str) -> str:
    """
    将中文导航提示尽量“完整地”翻译成英文（不刻意简化）。
    说明：
    - 这里不用 LLM 翻译，避免额外网络/模型依赖；
    - 覆盖本项目导航/避障/斑马线/红绿灯/校准常见句式；
    - 未覆盖的会回退到比较中性的英文（并打印提示，便于你后续补规则）。
    """
    t = _strip_punct(text_zh)
    if not t:
        return "OK."

    # 固定句（尽量保留信息量）
    fixed = {
        "保持直行": "Keep straight.",
        "保持直行，靠近盲道": "Keep straight and move toward the tactile path.",
        "丢失路径，重新搜索": "Path lost. Searching again.",
        "请停下": "Please stop.",
        "请停下，前方无路": "Please stop. No path ahead.",
        "前方无路": "No path ahead.",
        "切换到盲道导航": "Switching to tactile-path navigation.",
        "已到盲道跟前，切换到盲道导航": "At the tactile path. Switching to tactile-path navigation.",
        "路径太远，请继续靠近": "The path is too far. Please come closer.",
        "检测到已移动，开始对准新方向": "Movement detected. Aligning to the new direction.",
        "已对准新路径，请向前直行": "Aligned to the new path. Please walk forward.",
        "校准完成！您已在盲道上，开始前行": "Calibration complete. You are on the tactile path. Start walking.",
        "方向已对正！现在校准位置": "Direction aligned. Now calibrating position.",
        "请向前移动，让盲道更清晰": "Please move forward so the tactile path is clearer.",
        "开始通行": "Start crossing.",
        "绿灯稳定，开始通行": "Green light is stable. Start crossing.",
        "正在等待绿灯": "Waiting for the green light.",
        "绿灯快没了": "Green light is ending soon.",
        "红灯": "Red light.",
        "黄灯": "Yellow light.",
        "绿灯": "Green light.",
        "已对准, 准备切换过马路模式": "Aligned. Ready to switch to crossing mode.",
        "斑马线已对准，继续前行": "Crosswalk aligned. Continue forward.",
        "正在接近斑马线，为您对准方向": "Approaching the crosswalk. Aligning your direction.",
        "斑马线已在跟前，进入红绿灯判定模式": "At the crosswalk. Checking the traffic light.",
        "过马路结束，准备上人行道": "Crossing complete. Preparing to step onto the sidewalk.",
        "远处有盲道，继续前行": "Tactile path ahead. Continue forward.",
        "已停止导航": "Navigation stopped.",
        "过马路模式已启动": "Crossing mode started.",
        "在画面左侧": "It is on the left.",
        "在画面中间": "It is in the center.",
        "在画面右侧": "It is on the right.",
    }
    if t in fixed:
        return fixed[t]

    # ---------- 通用词表（用于兜底拼句，保证“每一条都能翻译成英文”） ----------
    obj_map = {
        "盲道": "tactile path",
        "人行道": "sidewalk",
        "可通行区域": "walkable area",
        "斑马线": "crosswalk",
        "红绿灯": "traffic light",
        "新路径": "new path",
        "路径": "path",
        "方向": "direction",
        "位置": "position",
        "障碍物": "obstacle",
        "前方": "ahead",
        "左侧": "left side",
        "右侧": "right side",
    }

    def _map_obj(s: str) -> str:
        s = _strip_punct(s)
        for k, v in obj_map.items():
            if k in s:
                return v
        return ""

    def _has_any(subs: tuple[str, ...]) -> bool:
        return any(x in t for x in subs)

    # 方向/动作类（保留“请/继续/微调”等语气）
    if t.startswith("请继续向") and "平移" in t:
        if "右" in t:
            return "Please keep moving right."
        if "左" in t:
            return "Please keep moving left."
    if t.startswith("请向") and ("平移" in t or "移动" in t):
        if "右" in t:
            return "Please move right."
        if "左" in t:
            return "Please move left."
    if t.startswith("请向") and ("微调" in t):
        if "右" in t:
            return "Please adjust slightly to the right."
        if "左" in t:
            return "Please adjust slightly to the left."
    if t.startswith("请向") and ("转动" in t or "转" in t):
        if "右" in t:
            return "Please turn right."
        if "左" in t:
            return "Please turn left."
    if t in ("左转", "右转", "左移", "右移"):
        return {"左转": "Turn left.", "右转": "Turn right.", "左移": "Move left.", "右移": "Move right."}[t]

    # 没有“请”的短口令：右转一点 / 左转一点 / 向右平移 / 向左平移 / 向右移动 / 向左移动 / 稍微向右调整…
    if _has_any(("右转一点", "右转点", "向右转一点", "右转一下")):
        return "Turn right a bit."
    if _has_any(("左转一点", "左转点", "向左转一点", "左转一下")):
        return "Turn left a bit."

    if t in ("向右平移", "向右移动", "右移一点", "右移点", "右移一下"):
        return "Move right."
    if t in ("向左平移", "向左移动", "左移一点", "左移点", "左移一下"):
        return "Move left."

    if _has_any(("稍微向右", "向右一点", "向右靠", "向右调整")) and _has_any(("调整", "靠", "一点", "稍微", "微调")):
        return "Move a bit right."
    if _has_any(("稍微向左", "向左一点", "向左靠", "向左调整")) and _has_any(("调整", "靠", "一点", "稍微", "微调")):
        return "Move a bit left."

    # “到达转弯处，向X平移”
    if t.startswith("到达转弯处") and ("向左" in t or "向右" in t):
        if "向左" in t:
            return "At the turn. Move left."
        return "At the turn. Move right."

    # 路径被挡住/避障
    if "路径被挡住" in t or "被挡住" in t:
        if "左" in t:
            return "Path blocked. Move left."
        if "右" in t:
            return "Path blocked. Move right."
        return "Path blocked."
    if t.startswith("向前直行几步越过障碍物"):
        return "Walk forward a few steps to pass the obstacle. Then say: done."

    # 对准类：向右平移，对准盲道 / 请向右微调，对准人行道
    if ("对准" in t) and (_has_any(("平移", "移动", "微调", "调整"))):
        target = _map_obj(t.split("对准", 1)[-1])
        if not target:
            target = "the path"
        if "右" in t:
            if "微调" in t or "调整" in t:
                return f"Adjust slightly right to align with the {target}."
            return f"Move right to align with the {target}."
        if "左" in t:
            if "微调" in t or "调整" in t:
                return f"Adjust slightly left to align with the {target}."
            return f"Move left to align with the {target}."
        return f"Align with the {target}."

    # 转弯提示 “前方有左/右转弯，继续直行”
    if "前方有左转弯" in t and "继续直行" in t:
        return "Left turn ahead. Keep straight."
    if "前方有右转弯" in t and "继续直行" in t:
        return "Right turn ahead. Keep straight."

    # 障碍物：前方有X（停一下/注意避让）
    if t.startswith("前方有"):
        obj = t.replace("前方有", "")
        obj = obj.replace("，注意避让", "").replace("，停一下", "")
        obj = _strip_punct(obj)
        obj_map = {
            "人": "a person",
            "车": "a car",
            "公交车": "a bus",
            "卡车": "a truck",
            "自行车": "a bicycle",
            "摩托车": "a motorcycle",
            "电瓶车": "an e-scooter",
            "婴儿车": "a stroller",
            "狗": "a dog",
            "动物": "an animal",
            "障碍物": "an obstacle",
        }
        en_obj = obj_map.get(obj, "an obstacle")
        if "停一下" in t:
            return f"{en_obj.capitalize()} ahead. Please stop."
        if "注意避让" in t:
            return f"{en_obj.capitalize()} ahead. Please avoid."
        return f"{en_obj.capitalize()} ahead."

    # 障碍物：左/右侧有X（停一下）
    if t.startswith("左侧有") or t.startswith("右侧有"):
        side = "Left" if t.startswith("左侧有") else "Right"
        obj = t.replace("左侧有", "").replace("右侧有", "")
        obj = obj.replace("，停一下", "")
        obj = _strip_punct(obj)
        obj_map = {
            "人": "person",
            "车": "car",
            "公交车": "bus",
            "卡车": "truck",
            "自行车": "bicycle",
            "摩托车": "motorcycle",
            "电瓶车": "e-scooter",
            "婴儿车": "stroller",
            "狗": "dog",
            "动物": "animal",
            "障碍物": "obstacle",
        }
        en_obj = obj_map.get(obj, "obstacle")
        if "停一下" in t:
            return f"{side} {en_obj}. Please stop."
        return f"{side} {en_obj}."

    # 斑马线（远处/靠近/到达）
    if "斑马线" in t:
        if "远处发现" in t:
            return "Crosswalk detected ahead."
        if "正在靠近" in t or "接近" in t:
            return "Approaching the crosswalk."
        if "可以过马路" in t:
            return "At the crosswalk. You may cross."
        return "Crosswalk."

    # ---------- 最后兜底：拼一个“可执行”的英文句子，避免任何一条变成 OK/Continue ----------
    # 规则：尽量识别方向+动作；识别不到则给 “Proceed.”（仍是导航语境）
    side = ""
    if "左" in t and "右" not in t:
        side = "left"
    elif "右" in t and "左" not in t:
        side = "right"

    action = ""
    if "转" in t:
        action = "turn"
    elif "平移" in t or "移动" in t:
        action = "move"
    elif "微调" in t or "调整" in t:
        action = "adjust"
    elif "直行" in t or "前进" in t:
        action = "go"

    if action == "turn" and side:
        return f"Turn {side}."
    if action in ("move", "adjust") and side:
        return f"{'Adjust' if action=='adjust' else 'Move'} {side}."
    if action == "go":
        return "Proceed."
    return "Proceed."


def tts_to_wav_by_qwen(text: str, wav_path: Path) -> tuple[float, int]:
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is missing")

    dashscope.base_http_api_url = os.getenv(
        "DASHSCOPE_HTTP_API_URL",
        "https://dashscope.aliyuncs.com/api/v1",
    )

    # 强制使用女生音色 + 英文，避免被环境变量覆盖成男声/中文
    # 优先使用 Instruct 模型，通过 instructions 直接控制语速/播报风格，避免后处理重采样导致音色变差。
    use_instruct = os.getenv("QWEN_TTS_USE_INSTRUCT", "1").strip().lower() in ("1", "true", "yes", "on")
    # 默认强制使用 instruct 模型；若你确实要覆盖，设置 QWEN_TTS_MODEL 显式指定
    model_name = os.getenv("QWEN_TTS_MODEL") or ("qwen3-tts-instruct-flash" if use_instruct else "qwen3-tts-flash")
    instructions = None
    optimize_instructions = None
    if use_instruct:
        # 面向盲人导航提示：语速快、清晰、少情绪、少停顿
        # 注意：该参数仅对 qwen3-tts-instruct-* 系列生效
        instructions = os.getenv(
            "QWEN_TTS_INSTRUCTIONS",
            "Speak at 1.5x normal speed. Fast speech rate, clear articulation, short pauses, navigation guidance style.",
        )
        optimize_instructions = os.getenv("QWEN_TTS_OPTIMIZE_INSTRUCTIONS", "1").strip().lower() in ("1", "true", "yes", "on")

    print(f"[TTS] model={model_name} voice=Cherry lang=English instruct={bool(instructions)}")

    resp = dashscope.MultiModalConversation.call(
        model=model_name,
        api_key=api_key,
        text=text,
        voice="Cherry",
        language_type="English",
        instructions=instructions,
        optimize_instructions=optimize_instructions,
    )

    audio_url = ((resp or {}).get("output", {}).get("audio") or {}).get("url")
    if not audio_url:
        raise RuntimeError(f"Qwen TTS did not return audio url: {resp}")

    wav_bytes = urllib.request.urlopen(audio_url, timeout=30).read()

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        src_rate = wf.getframerate()
        raw_frames = wf.readframes(wf.getnframes())

    if not raw_frames:
        raise RuntimeError(f"Invalid empty wav for text={text!r}")

    if channels == 2:
        raw_frames = audioop.tomono(raw_frames, sampwidth, 1, 0)
    elif channels != 1:
        raise RuntimeError(f"Unsupported channels={channels} for text={text!r}")

    # 统一到 16k/16bit/mono
    if src_rate != 16000:
        raw_frames, _ = audioop.ratecv(raw_frames, sampwidth, 1, src_rate, 16000, None)
        src_rate = 16000
    if sampwidth != 2:
        raw_frames = audioop.lin2lin(raw_frames, sampwidth, 2)
        sampwidth = 2

    # 默认不做后处理变速（重采样会损音色）；如确实需要，可显式开启
    if os.getenv("QWEN_TTS_ENABLE_RESAMPLE_SPEEDUP", "0").strip().lower() in ("1", "true", "yes", "on"):
        # 语速加快（兜底方案，会略升音调）
        try:
            speedup = float(os.getenv("QWEN_TTS_SPEEDUP", "1.5"))
        except Exception:
            speedup = 1.5
        if speedup and speedup > 1.01:
            target_rate = int(round(16000 / speedup))
            target_rate = max(8000, min(16000, target_rate))
            raw_frames, _ = audioop.ratecv(raw_frames, 2, 1, 16000, target_rate, None)

    with wave.open(str(wav_path), "wb") as out_wf:
        out_wf.setnchannels(1)
        out_wf.setsampwidth(2)
        out_wf.setframerate(16000)
        out_wf.writeframes(raw_frames)

    # 优先用 ffmpeg 的 atempo 做“无变调”加速（比重采样好听）
    # 默认启用，倍率默认 1.5
    use_ffmpeg = os.getenv("QWEN_TTS_USE_FFMPEG_ATEMPO", "1").strip().lower() in ("1", "true", "yes", "on")
    atempo = os.getenv("QWEN_TTS_ATEMPO", "2.0").strip()
    if use_ffmpeg:
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            # atempo 支持 0.5~2.0；1.5 在范围内
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(wav_path),
                    "-filter:a",
                    f"atempo={atempo}",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    tmp_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            shutil.move(tmp_path, str(wav_path))
            print(f"[TTS] applied ffmpeg atempo={atempo}")
        except Exception:
            pass

    with wave.open(str(wav_path), "rb") as check_wf:
        duration = check_wf.getnframes() / float(check_wf.getframerate())
        if duration <= 0.0:
            raise RuntimeError(f"Written wav duration invalid for text={text!r}")

    return duration, wav_path.stat().st_size


def iter_voice_wavs() -> list[Path]:
    wavs: list[Path] = []
    if not VOICE_DIR.exists():
        return wavs
    for p in VOICE_DIR.rglob("*"):
        if not p.is_file():
            continue
        # 跳过备份目录
        if "_backup_zh" in p.parts:
            continue
        if p.suffix.lower() == ".wav":
            wavs.append(p)
    return sorted(wavs)


def main() -> None:
    load_env_from_dotenv()

    one = os.getenv("VOICE_ONE_FILE", "").strip()

    wavs = iter_voice_wavs()
    if not wavs:
        print("No wav files found under voice/.")
        return

    if one:
        wavs = [p for p in wavs if p.name == one]
        if not wavs:
            print(f"No matching wav under voice/ for VOICE_ONE_FILE={one!r}")
            return

    backup_dir = VOICE_DIR / "_backup_zh" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0
    skipped = 0

    for wav_path in wavs:
        zh_text = wav_path.stem
        en_text = zh_to_en_full(zh_text)
        print(f"[ONE] {wav_path.name} | zh={zh_text} | en={en_text} | voice=Cherry | speedup={os.getenv('QWEN_TTS_SPEEDUP','1.5')}")

        # 备份原始文件
        try:
            shutil.copy2(wav_path, backup_dir / wav_path.name)
        except Exception:
            pass

        try:
            duration, size = tts_to_wav_by_qwen(en_text, wav_path)
            ok += 1
            if ok % 20 == 0:
                print(f"[OK] {ok}/{len(wavs)} last={wav_path.name} <= {en_text} ({duration:.2f}s, {size} bytes)")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {wav_path.name} <= {en_text}: {e}")

    print(f"\nDone. success={ok}, failed={fail}, skipped={skipped}, total={len(wavs)}")
    print(f"Backup: {backup_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

