import asyncio
import concurrent.futures
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent.parent / "cuda"))

from .tracker import COCO_NAMES, SCALE_M_PER_PX, StreamTracker, TrackResult

logger = logging.getLogger(__name__)

TARGET_H, TARGET_W = 640, 640
GRID_COLS = 3

_latest_jpeg: bytes = b""

_inference_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="infer"
)


class InferenceEngine:
    def __init__(self, model_path: str = "models/yolov8n.onnx"):
        self.model_path = model_path
        self.session = None
        self._load()

    def _load(self) -> None:
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if p in available]
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            inp = self.session.get_inputs()[0]
            logger.info(f"ONNX model loaded ({providers[0]}) — input: {inp.name} {inp.shape}")
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
        return np.stack([
            self.session.run(None, {name: frame[None]})[0][0]
            for frame in batch
        ])


_TRAIL_LEN = 15


def _track_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 37) % 180
    bgr = cv2.cvtColor(np.array([[[hue, 200, 220]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_tracks(
    frame: np.ndarray,
    tracks: list[TrackResult],
    stream_id: int,
    history: dict[int, deque],
) -> np.ndarray:
    import time
    out = frame.copy()

    for tid, pts in history.items():
        trail = list(pts)[-_TRAIL_LEN:]
        color = _track_color(tid)
        for j in range(1, len(trail)):
            cv2.line(out,
                     (int(trail[j - 1][0]), int(trail[j - 1][1])),
                     (int(trail[j][0]), int(trail[j][1])),
                     color, 1)

    for t in tracks:
        color = _track_color(t.track_id)
        cv2.rectangle(out, (t.x1, t.y1), (t.x2, t.y2), color, 2)
        name = COCO_NAMES.get(t.class_id, "obj")
        cv2.putText(out, f"#{t.track_id} {name}", (t.x1, max(t.y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    ts = time.strftime("%H:%M:%S")
    cv2.putText(out, f"cam{stream_id} | {len(tracks)} trk | {ts}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def draw_overlay(frame: np.ndarray, stream_id: int) -> np.ndarray:
    import time
    out = frame.copy()
    ts = time.strftime("%H:%M:%S")
    cv2.putText(out, f"cam{stream_id} | {ts}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def draw_speed_hud(grid: np.ndarray, all_tracks: dict[int, list[TrackResult]]) -> np.ndarray:
    entries = [
        (sid, t.track_id, COCO_NAMES.get(t.class_id, "obj"), t.speed_kmh)
        for sid, tracks in sorted(all_tracks.items())
        for t in sorted(tracks, key=lambda t: t.speed_kmh, reverse=True)
    ]
    if not entries:
        return grid

    line_h, pad = 20, 8
    panel_w = 215
    panel_h = line_h * min(len(entries), 25) + pad * 2 + 20
    x0 = grid.shape[1] - panel_w - 10
    y0 = 10

    out = grid.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

    cv2.putText(out, f"Speed  km/h  (scale={SCALE_M_PER_PX:.3f}m/px)",
                (x0 + pad, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    for i, (sid, tid, name, spd) in enumerate(entries[:25]):
        y = y0 + 20 + pad + i * line_h
        color = (0, 220, 0) if spd < 90 else (0, 165, 255) if spd < 130 else (0, 0, 255)
        cv2.putText(out, f"cam{sid} #{tid:<3d} {name:<5}  {spd:5.0f}",
                    (x0 + pad, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

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


async def _run_in_inference_pool(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_inference_pool, fn, *args)


async def inference_consumer(
    queues: list[asyncio.Queue],
    engine: InferenceEngine,
    stop_event: asyncio.Event,
) -> None:
    global _latest_jpeg
    logger.info("consumer started")
    latest_frames: dict[int, np.ndarray] = {}
    trackers: dict[int, StreamTracker] = {i: StreamTracker() for i in range(len(queues))}

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

        try:
            batch = await _run_in_inference_pool(engine.preprocess, list(live_frames))
            output = await _run_in_inference_pool(engine.infer, batch)

            for i, frame in enumerate(live_frames):
                latest_frames[indices[i]] = draw_overlay(frame, indices[i])

            preds = output.transpose(0, 2, 1)

            all_tracks: dict[int, list[TrackResult]] = {}
            try:
                import nms_cuda
                for i, pred in enumerate(preds):
                    boxes_np = pred[:, :4].astype(np.float32)
                    scores_np = pred[:, 4:].max(axis=1).astype(np.float32)
                    class_ids_np = pred[:, 4:].argmax(axis=1).astype(np.int32)

                    keep = nms_cuda.run_nms(boxes_np, scores_np, iou_threshold=0.45, conf_threshold=0.25)
                    mask = keep.astype(bool)
                    kept_boxes = boxes_np[mask]
                    kept_scores = scores_np[mask]
                    kept_class_ids = class_ids_np[mask]

                    stream_id = indices[i]
                    tracker = trackers[stream_id]
                    tracks = tracker.update(
                        kept_boxes, kept_scores, kept_class_ids,
                        live_frames[i].shape[:2],
                    )
                    all_tracks[stream_id] = tracks
                    logger.debug(f"stream {stream_id}: {len(kept_boxes)} det, {len(tracks)} tracks")
                    latest_frames[stream_id] = draw_tracks(
                        live_frames[i], tracks, stream_id, tracker.history,
                    )
            except ImportError:
                pass
            except Exception as exc:
                logger.error(f"NMS/tracker error: {exc}")

            if latest_frames:
                snap = dict(latest_frames)
                snap_tracks = dict(all_tracks)
                n = len(queues)

                def _encode():
                    g = make_grid(snap, n)
                    g = draw_speed_hud(g, snap_tracks)
                    _, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    return buf.tobytes()

                _latest_jpeg = await _run_in_inference_pool(_encode)

        except Exception as exc:
            logger.error(f"consumer loop error: {exc}", exc_info=True)

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
