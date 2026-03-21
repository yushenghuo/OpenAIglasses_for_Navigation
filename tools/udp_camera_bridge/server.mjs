/**
 * 独立 UDP→JPEG 预览（与 Python 二选一，勿同占 18500）
 */
import dgram from "node:dgram";
import http from "node:http";
import { WebSocketServer } from "ws";

const UDP_PORT = Number(process.env.CAM_UDP_PORT || 18500);
const HTTP_PORT = Number(process.env.CAM_HTTP_PORT || 8082);
const HEADER = 6;
const FRAME_TIMEOUT_MS = 100;
const MAX_FRAME_BYTES = 512 * 1024;
const MAX_INFLIGHT = 4;

const inflight = new Map();
const mjpegClients = new Set();
let latestJpeg = null;

function cleanupStale(now = Date.now()) {
  for (const [fid, asm] of inflight) {
    if (now - asm.t0 > FRAME_TIMEOUT_MS) inflight.delete(fid);
  }
}
setInterval(cleanupStale, 50);

function trimInflight() {
  while (inflight.size > MAX_INFLIGHT) {
    let oldest = Infinity;
    let oid = -1;
    for (const [fid, asm] of inflight) {
      if (asm.t0 < oldest) {
        oldest = asm.t0;
        oid = fid;
      }
    }
    if (oid >= 0) inflight.delete(oid);
    else break;
  }
}

function pushMjpeg(res, jpeg) {
  if (res.writableEnded) return;
  try {
    res.write(
      `--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${jpeg.length}\r\n\r\n`
    );
    res.write(jpeg);
    res.write("\r\n");
  } catch (_) {
    mjpegClients.delete(res);
  }
}

const server = http.createServer((req, res) => {
  if (req.url === "/health") {
    res.end("ok");
    return;
  }
  if (req.url === "/mjpeg" || req.url.startsWith("/mjpeg?")) {
    res.writeHead(200, {
      "Content-Type": "multipart/x-mixed-replace; boundary=frame",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    mjpegClients.add(res);
    if (latestJpeg) pushMjpeg(res, latestJpeg);
    req.on("close", () => mjpegClients.delete(res));
    return;
  }
  res.writeHead(404);
  res.end();
});

const wss = new WebSocketServer({ server, path: "/ws/latest" });
wss.on("connection", (ws) => {
  if (latestJpeg) {
    try {
      ws.send(latestJpeg);
    } catch (_) {}
  }
});

function tryFinish(fid, asm) {
  const t = asm.total;
  for (let i = 0; i < t; i++) {
    if (!asm.chunks.has(i)) return;
  }
  const parts = [];
  for (let i = 0; i < t; i++) parts.push(asm.chunks.get(i));
  const jpeg = Buffer.concat(parts);
  inflight.delete(fid);
  if (jpeg.length < 2 || jpeg[0] !== 0xff || jpeg[1] !== 0xd8) return;
  if (jpeg.length > MAX_FRAME_BYTES) return;
  latestJpeg = jpeg;
  for (const c of wss.clients) {
    if (c.readyState === 1) {
      try {
        c.send(jpeg);
      } catch (_) {}
    }
  }
  for (const r of mjpegClients) pushMjpeg(r, jpeg);
}

const sock = dgram.createSocket("udp4");
sock.on("message", (msg) => {
  if (msg.length <= HEADER) return;
  const frameId = msg.readUInt16LE(0);
  const chunkId = msg.readUInt16LE(2);
  const totalChunks = msg.readUInt16LE(4);
  const payload = msg.subarray(HEADER);
  if (totalChunks === 0 || chunkId >= totalChunks) return;
  let asm = inflight.get(frameId);
  if (!asm) {
    asm = { total: totalChunks, chunks: new Map(), t0: Date.now() };
    inflight.set(frameId, asm);
    trimInflight();
  } else if (asm.total !== totalChunks) {
    asm.total = totalChunks;
    asm.chunks.clear();
    asm.t0 = Date.now();
  }
  if (!asm.chunks.has(chunkId)) asm.chunks.set(chunkId, Buffer.from(payload));
  tryFinish(frameId, asm);
});

sock.bind(UDP_PORT, () => console.log(`[bridge] UDP :${UDP_PORT}`));
server.listen(HTTP_PORT, "0.0.0.0", () =>
  console.log(`[bridge] http://0.0.0.0:${HTTP_PORT}/mjpeg  ws://...:${HTTP_PORT}/ws/latest`)
);
