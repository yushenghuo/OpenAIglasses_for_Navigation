# navigation_destination.py
# 从 ASR 整句中抽取导航目的地（中英文、多模式、停用词过滤）
# 从 darksight/utils/navigationDestination.ts 迁移

import re
from typing import Optional

MIN_PLACE_LENGTH = 2
MAX_PLACE_LENGTH = 80

ENGLISH_STOPWORDS = {"to", "two", "too", "the", "a", "an", "me", "my", "please", "go", "get", "take", "need", "want"}
CHINESE_STOPWORDS = {"到", "去", "请", "帮", "我", "要", "想", "能", "给", "带", "往", "在"}


def _trim_trailing_punctuation(s: str) -> str:
    return re.sub(r"[.?!,。？！，、\s]+$", "", s).strip()


def _strip_leading_english_stopwords(s: str) -> str:
    t = s.strip()
    lower = t.lower()
    if lower.startswith("the "):
        return t[4:].strip()
    if lower.startswith("a "):
        return t[2:].strip()
    if lower.startswith("an "):
        return t[3:].strip()
    return t


def _strip_leading_chinese_stopwords(s: str) -> str:
    t = s.strip()
    for w in ["请带我去", "带我去", "我要去", "想去", "去", "到", "请"]:
        if t.startswith(w):
            t = t[len(w) :].strip()
            break
    return t


def _is_valid_place_name(place: str) -> bool:
    if not place or len(place) < MIN_PLACE_LENGTH:
        return False
    if len(place) > MAX_PLACE_LENGTH:
        return False
    lower = place.lower().strip()
    if lower in ENGLISH_STOPWORDS:
        return False
    parts = lower.split()
    if len(parts) <= 2 and parts and parts[0] in ENGLISH_STOPWORDS:
        return False
    if re.match(r"^\d+$", place.strip()):
        return False
    if place.strip() in CHINESE_STOPWORDS:
        return False
    if place.strip()[:1] in CHINESE_STOPWORDS:
        rest = place.strip()[1:].strip()
        if len(rest) < 2:
            return False
    return True


def _normalize_input(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\bnavigate\s+two\b", "navigate to", t, flags=re.I)
    t = re.sub(r"\bnavigate\s+2\s+", "navigate to ", t, flags=re.I)
    t = re.sub(r"\bnavigation\s+two\b", "navigation to", t, flags=re.I)
    t = re.sub(r"\bgo\s+two\b", "go to", t, flags=re.I)
    return t.strip()


def _normalize_place(raw: str, lang: str) -> str:
    s = _trim_trailing_punctuation(raw)
    s = _strip_leading_english_stopwords(s) if lang == "en" else _strip_leading_chinese_stopwords(s)
    return s.strip()


# 英文模式：(pattern, 'en')
EN_PATTERNS = [
    (re.compile(r"navigate\s+to\s+(.+)", re.I), "en"),
    (re.compile(r"navigation\s+to\s+(.+)", re.I), "en"),
    (re.compile(r"navigate\s+(.+)", re.I), "en"),
    (re.compile(r"go\s+to\s+(.+)", re.I), "en"),
    (re.compile(r"take\s+me\s+to\s+(.+)", re.I), "en"),
    (re.compile(r"get\s+me\s+to\s+(.+)", re.I), "en"),
    (re.compile(r"directions?\s+to\s+(.+)", re.I), "en"),
    (re.compile(r"walk(?:ing)?\s+to\s+(.+)", re.I), "en"),
    (re.compile(r"drive\s+to\s+(.+)", re.I), "en"),
    (re.compile(r"(?:i\s+)?(?:want\s+to\s+)?go\s+to\s+(.+)", re.I), "en"),
    (re.compile(r"start\s+navigate\s+to\s+(.+)", re.I), "en"),
    (re.compile(r"start\s+navigation\s+to\s+(.+)", re.I), "en"),
]

ZH_PATTERNS = [
    (re.compile(r"导航\s*[到至]\s*(.+)", re.I), "zh"),
    (re.compile(r"导航\s+(.+)", re.I), "zh"),
    (re.compile(r"去\s+(.+)", re.I), "zh"),
    (re.compile(r"到\s+(.+)", re.I), "zh"),
    (re.compile(r"带?\s*我\s*去\s*(.+)", re.I), "zh"),
    (re.compile(r"我要?\s*去\s*(.+)", re.I), "zh"),
    (re.compile(r"想去\s*(.+)", re.I), "zh"),
]


def parse_navigation_destination(text: str) -> Optional[str]:
    """
    从 ASR 整句文本中抽取导航目的地。
    支持中英文、多模式、ASR 容错（navigate two -> navigate to）、停用词过滤。
    Returns:
        地名字符串，无效或未识别时返回 None。
    """
    t = _normalize_input(text)
    if not t:
        return None
    for pat, lang in EN_PATTERNS:
        m = pat.search(t)
        if m and m.group(1):
            place = _normalize_place(m.group(1).strip(), lang)
            if _is_valid_place_name(place):
                return place
    for pat, lang in ZH_PATTERNS:
        m = pat.search(t)
        if m and m.group(1):
            place = _normalize_place(m.group(1).strip(), lang)
            if _is_valid_place_name(place):
                return place
    return None
