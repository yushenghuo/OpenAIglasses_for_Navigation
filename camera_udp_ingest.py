# -*- coding: utf-8 -*-
"""
ESP32 相机 UDP 分片重组，协议与 compile.ino 一致：
小端 uint16: frame_id, chunk_id, total_chunks，后跟 JPEG 片段。
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, Optional

HEADER = 6
FRAME_TIMEOUT_MS = float(os.getenv("CAMERA_UDP_FRAME_TIMEOUT_MS", "100"))
MAX_FRAME_BYTES = int(os.getenv("CAMERA_UDP_MAX_FRAME_BYTES", str(512 * 1024)))
MAX_INFLIGHT = 4


class CameraUDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, frame_queue: asyncio.Queue) -> None:
        self.frame_queue = frame_queue
        self.inflight: Dict[int, dict] = {}

    def datagram_received(self, data: bytes, addr) -> None:  # noqa: ANN001
        self._on_packet(data)

    def _cleanup_inflight(self, now: float) -> None:
        cutoff = now - FRAME_TIMEOUT_MS / 1000.0
        stale = [fid for fid, asm in self.inflight.items() if asm["t0"] < cutoff]
        for fid in stale:
            del self.inflight[fid]
        while len(self.inflight) > MAX_INFLIGHT:
            oldest_fid = min(self.inflight.keys(), key=lambda f: self.inflight[f]["t0"])
            del self.inflight[oldest_fid]

    def _on_packet(self, data: bytes) -> None:
        if len(data) <= HEADER:
            return
        frame_id = int.from_bytes(data[0:2], "little")
        chunk_id = int.from_bytes(data[2:4], "little")
        total_chunks = int.from_bytes(data[4:6], "little")
        payload = data[HEADER:]
        if total_chunks == 0 or chunk_id >= total_chunks:
            return
        now = time.time()
        self._cleanup_inflight(now)
        if frame_id not in self.inflight:
            self.inflight[frame_id] = {"total": total_chunks, "chunks": {}, "t0": now}
        asm = self.inflight[frame_id]
        if asm["total"] != total_chunks:
            asm["total"] = total_chunks
            asm["chunks"].clear()
            asm["t0"] = now
        if chunk_id not in asm["chunks"]:
            asm["chunks"][chunk_id] = payload
        self._try_finish(frame_id, asm)

    def _try_finish(self, frame_id: int, asm: dict) -> None:
        t = asm["total"]
        if len(asm["chunks"]) < t:
            return
        for i in range(t):
            if i not in asm["chunks"]:
                return
        parts = [asm["chunks"][i] for i in range(t)]
        total_len = sum(len(p) for p in parts)
        if total_len > MAX_FRAME_BYTES:
            del self.inflight[frame_id]
            return
        jpeg = b"".join(parts)
        del self.inflight[frame_id]
        if len(jpeg) < 2 or jpeg[0] != 0xFF or jpeg[1] != 0xD8:
            return
        recv_ts = time.time()
        packet = {
            "data": jpeg,
            "recv_ts": recv_ts,
            "frame_id": frame_id,
            "capture_ts_ms": None,
            "jpeg_len": len(jpeg),
        }
        try:
            self.frame_queue.put_nowait(packet)
        except asyncio.QueueFull:
            try:
                self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.frame_queue.put_nowait(packet)
            except Exception:
                pass


async def start_udp_camera_listener(frame_queue: asyncio.Queue, port: Optional[int] = None):
    port = port or int(os.getenv("CAMERA_UDP_PORT", "18500"))
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: CameraUDPProtocol(frame_queue),
        local_addr=("0.0.0.0", port),
    )
    print(f"[CAMERA-UDP] listening on 0.0.0.0:{port}", flush=True)
    return transport
