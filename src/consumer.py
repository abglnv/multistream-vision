import asyncio
import logging
from typing import Optional

import cv2
import numpy as np
from aiohttp import web

logger = logging.getLogger(__name__)

TARGET_H, TARGET_W = 640, 640
GRID_COLS = 3

_latest_jpeg: bytes = b""


class InferenceEngine:
    def __init__(self, model_path: str = "models/yolov8n.onnx"):
        self.model_path = model_path
        self.session = None
        self._load()

    def _load(self) -> None:
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            inp = self.session.get_inputs()[0]
            logger.info(f"ONNX model loaded — input: {inp.name} {inp.shape}")
        except Exception as exc:
            logger.warning(f"Running in stub mode (no model): {exc}")

    def preprocess(self, frames: list[np.ndarray]) -> np.ndarray:
        batch = []
        for f in frames:
            resized = cv2.resize(f, (TARGET_W, TARGET_H))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
            batch.append(chw)
        return np.stack(batch)

    def infer(self, batch: np.ndarray) -> np.ndarray:
        if self.session is None:
            return np.zeros((batch.shape[0], 84, 8400), dtype=np.float32)

        name = self.session.get_inputs()[0].name
        return self.session.run(None, {name: batch})[0]


def draw_boxes(frame: np.ndarray, boxes: np.ndarray, stream_id: int) -> np.ndarray:
    import time
    h, w = frame.shape[:2]
    sx, sy = w / TARGET_W, h / TARGET_H
    out = frame.copy()
    for box in boxes:
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * sx)
        y1 = int((cy - bh / 2) * sy)
        x2 = int((cx + bw / 2) * sx)
        y2 = int((cy + bh / 2) * sy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ts = time.strftime("%H:%M:%S")
    cv2.putText(out, f"cam{stream_id} | {len(boxes)} det | {ts}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def make_grid(frames: dict[int, np.ndarray], n_streams: int) -> np.ndarray:
    cell_h, cell_w = 360, 640
    cols = GRID_COLS
    rows = (n_streams + cols - 1) // cols
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for idx in range(n_streams):
        r, c = divmod(idx, cols)
        y1, y2 = r * cell_h, (r + 1) * cell_h
        x1, x2 = c * cell_w, (c + 1) * cell_w
        if idx in frames:
            grid[y1:y2, x1:x2] = cv2.resize(frames[idx], (cell_w, cell_h))
    return grid


async def inference_consumer(
    queues: list[asyncio.Queue],
    engine: InferenceEngine,
    stop_event: asyncio.Event,
) -> None:
    logger.info("consumer started")
    latest_frames: dict[int, np.ndarray] = {}

    while not stop_event.is_set():
        frames: list[Optional[np.ndarray]] = []
        for q in queues:
            try:
                frames.append(q.get_nowait())
            except asyncio.QueueEmpty:
                frames.append(None)

        live = [(i, f) for i, f in enumerate(frames) if f is not None]
        if not live:
            await asyncio.sleep(0.01)
            continue

        indices, live_frames = zip(*live)
        batch = await asyncio.to_thread(engine.preprocess, list(live_frames))
        output = await asyncio.to_thread(engine.infer, batch)

        for i, frame in enumerate(live_frames):
            latest_frames[indices[i]] = draw_boxes(frame, np.empty((0, 4)), indices[i])

        preds = output.transpose(0, 2, 1)

        try:
            import nms_cuda
            for i, pred in enumerate(preds):
                boxes_np = pred[:, :4].astype(np.float32)
                scores_np = pred[:, 4:].max(axis=1).astype(np.float32)
                keep = nms_cuda.run_nms(boxes_np, scores_np, iou_threshold=0.45)
                kept_boxes = boxes_np[keep.astype(bool)]
                stream_id = indices[i]
                logger.debug(f"stream {stream_id}: {len(kept_boxes)} detections")
                latest_frames[stream_id] = draw_boxes(live_frames[i], kept_boxes, stream_id)
        except ImportError:
            pass 
        except Exception as exc:
            logger.error(f"NMS error: {exc}")  

        if latest_frames:
            grid = await asyncio.to_thread(make_grid, latest_frames, len(queues))
            _, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 70])
            global _latest_jpeg
            _latest_jpeg = buf.tobytes()

        await asyncio.sleep(0)

    logger.info("consumer stopped")


async def mjpeg_handler(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse(headers={
        "Content-Type": "multipart/x-mixed-replace; boundary=frame"
    })
    await resp.prepare(request)
    while True:
        if _latest_jpeg:
            await resp.write(
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _latest_jpeg + b"\r\n"
            )
        await asyncio.sleep(0.05)
    return resp


async def _index(request: web.Request) -> web.Response:
    html = """<!DOCTYPE html><html>
<head><title>Vision Watchdog</title>
<style>body{background:#111;margin:0;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:100vh}</style></head>
<body><img src="/stream"></body></html>"""
    return web.Response(text=html, content_type="text/html")


async def start_web(host: str = "0.0.0.0", port: int = 8080) -> None:
    app = web.Application()
    app.router.add_get("/", _index)
    app.router.add_get("/stream", mjpeg_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info(f"Viewer → http://localhost:{port}/")
