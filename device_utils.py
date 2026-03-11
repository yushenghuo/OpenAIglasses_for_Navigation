# device_utils.py
# -*- coding: utf-8 -*-
"""
统一推理设备选择：优先 CUDA（NVIDIA） > MPS（Apple Silicon GPU） > CPU。
在 Mac 上无 NVIDIA 显卡时自动使用 Apple 的 Metal GPU（MPS），比纯 CPU 快不少。
"""
import os
import torch

# 允许环境变量强制指定设备（cuda:0 / mps / cpu）
_env_device = os.getenv("AIGLASS_DEVICE", "").strip().lower()


def get_device() -> str:
    """返回当前推荐推理设备：cuda:0 | mps | cpu"""
    if _env_device:
        if _env_device.startswith("cuda") and torch.cuda.is_available():
            return _env_device if ":" in _env_device else "cuda:0"
        if _env_device == "mps" and _get_mps_available():
            return "mps"
        if _env_device == "cpu":
            return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if _get_mps_available():
        return "mps"
    return "cpu"


def _get_mps_available() -> bool:
    """Apple Silicon / Metal GPU 是否可用"""
    if not hasattr(torch.backends, "mps"):
        return False
    if not getattr(torch.backends.mps, "is_available", lambda: False)():
        return False
    if not getattr(torch.backends.mps, "is_built", lambda: False)():
        return False
    return True


def is_cuda() -> bool:
    return get_device().startswith("cuda")


def is_mps() -> bool:
    return get_device() == "mps"


def get_amp_device_type():
    """
    返回 autocast 使用的 device_type：'cuda' | 'mps' | None。
    None 表示不使用 autocast（如 CPU）。
    """
    dev = get_device()
    if dev.startswith("cuda"):
        return "cuda"
    if dev == "mps":
        return "mps"
    return None


# 模块加载时打印一次，方便确认
_DEVICE = get_device()
if _DEVICE == "mps":
    print("[DEVICE] 使用 Apple Silicon GPU (MPS) 加速推理")
elif _DEVICE.startswith("cuda"):
    print(f"[DEVICE] 使用 NVIDIA GPU ({_DEVICE}) 加速推理")
else:
    print("[DEVICE] 使用 CPU 推理（无 CUDA/MPS）")
