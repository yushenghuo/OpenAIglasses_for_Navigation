import io
import os
import audioop
import urllib.request
import wave
from pathlib import Path

import dashscope


ROOT = Path(__file__).resolve().parents[1]
MUSIC_DIR = ROOT / "music"


# 对短导航指令做固定英文，保证术语一致性；若新增词条会自动回退到文件名
EN_TEXT_BY_BASENAME = {
    "向上": "Move up.",
    "向下": "Move down.",
    "向左": "Move left.",
    "向右": "Move right.",
    "向前": "Move forward.",
    "向后": "Move backward.",
    "在画面左侧": "It is on the left side of the view.",
    "在画面中间": "It is in the center of the view.",
    "在画面右侧": "It is on the right side of the view.",
    "远处发现斑马线": "Crosswalk detected in the distance.",
    "正在靠近斑马线": "Approaching the crosswalk.",
    "接近斑马线": "Near the crosswalk.",
    "斑马线到了可以过马路": "You are at the crosswalk. It is safe to cross.",
    "已对中": "Aligned.",
    "找到啦": "Found it.",
    "拿到啦": "Got it.",
    "红灯": "Red light.",
    "绿灯": "Green light.",
    "黄灯": "Yellow light.",
    "音频1": "Object detected.",
    "音频2": "Move up.",
    "音频3": "Move down.",
    "音频4": "Move left.",
    "音频5": "Move right.",
    "音频6": "OK.",
    "音频7": "Move forward.",
    "音频8": "Step back.",
    "音频9": "Object acquired.",
}


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


def tts_to_wav_by_qwen(text: str, wav_path: Path) -> tuple[float, int]:
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is missing")

    dashscope.base_http_api_url = os.getenv(
        "DASHSCOPE_HTTP_API_URL",
        "https://dashscope.aliyuncs.com/api/v1",
    )

    resp = dashscope.MultiModalConversation.call(
        model=os.getenv("QWEN_TTS_MODEL", "qwen3-tts-flash"),
        api_key=api_key,
        text=text,
        voice=os.getenv("QWEN_TTS_VOICE", "Cherry"),
        language_type=os.getenv("QWEN_TTS_LANG", "English"),
    )

    audio_url = ((resp or {}).get("output", {}).get("audio") or {}).get("url")
    if not audio_url:
        raise RuntimeError(f"Qwen TTS did not return audio url: {resp}")

    wav_bytes = urllib.request.urlopen(audio_url, timeout=30).read()

    # 某些云端 WAV 头里的 nframes 是占位值，这里读真实字节并重新封装标准 WAV
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        src_rate = wf.getframerate()
        raw_frames = wf.readframes(wf.getnframes())

    if not raw_frames:
        raise RuntimeError(f"Invalid empty wav for text={text!r}")

    if channels == 2:
        raw_frames = audioop.tomono(raw_frames, sampwidth, 1, 0)
        channels = 1
    elif channels != 1:
        raise RuntimeError(f"Unsupported channels={channels} for text={text!r}")

    if src_rate != 16000:
        raw_frames, _ = audioop.ratecv(raw_frames, sampwidth, 1, src_rate, 16000, None)
        src_rate = 16000

    if sampwidth != 2:
        raw_frames = audioop.lin2lin(raw_frames, sampwidth, 2)
        sampwidth = 2

    with wave.open(str(wav_path), "wb") as out_wf:
        out_wf.setnchannels(1)
        out_wf.setsampwidth(2)
        out_wf.setframerate(16000)
        out_wf.writeframes(raw_frames)

    with wave.open(str(wav_path), "rb") as check_wf:
        duration = check_wf.getnframes() / float(check_wf.getframerate())
        if duration <= 0.0:
            raise RuntimeError(f"Written wav duration invalid for text={text!r}")

    return duration, wav_path.stat().st_size


def main() -> None:
    load_env_from_dotenv()

    txt_stems = {p.stem for p in MUSIC_DIR.glob("*.txt")}
    mapped_stems = set(EN_TEXT_BY_BASENAME.keys())
    target_stems = sorted(txt_stems | mapped_stems)
    if not target_stems:
        print("No prompts found in music directory.")
        return

    ok_count = 0
    fail_count = 0

    for stem in target_stems:
        en_text = EN_TEXT_BY_BASENAME.get(stem) or stem
        wav_path = MUSIC_DIR / f"{stem}.WAV"
        try:
            duration, size = tts_to_wav_by_qwen(en_text, wav_path)
            ok_count += 1
            print(
                f"[OK] {wav_path.relative_to(ROOT)} <= {en_text} "
                f"(dur={duration:.3f}s, size={size})"
            )
        except Exception as e:
            fail_count += 1
            print(f"[FAIL] {wav_path.relative_to(ROOT)} <= {en_text}: {e}")

    print(
        f"\nDone. success={ok_count}, failed={fail_count}, total={len(target_stems)} "
        f"(txt={len(txt_stems)}, mapped={len(mapped_stems)})"
    )


if __name__ == "__main__":
    main()
