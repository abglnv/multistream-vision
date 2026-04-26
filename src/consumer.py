import asyncio
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET_H, TARGET_W = 640, 640


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
            chw = np.transpose(rgb.astype(np.float32) / 255.0, (2,0,1))
            batch.append(chw)
        return np.stack(batch)
        
    def infer(self, batch: np.ndarray) -> np.ndarray:
        if self.session is None:
            return np.zeros((batch.shape[0], 84, 8400), dtype=np.float32)
        
        name = self.session.get_inputs()[0].name 
        return self.session.run(None, {name: batch})[0]


async def inference_consumer(
    queues: list[asyncio.Queue],
    engine: InferenceEngine,
    stop_event: asyncio.Event,
) -> None:
    logger.info("consumer started")

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

        preds = output.transpose(0, 2, 1)          

        try:
            import nms_cuda
            for i, pred in enumerate(preds):
                boxes_np = pred[:, :4].astype(np.float32)          # [8400, 4]
                scores_np = pred[:, 4:].max(axis=1).astype(np.float32)  # [8400]
                keep = nms_cuda.run_nms(boxes_np, scores_np, iou_threshold=0.45)
                kept_boxes = boxes_np[keep.astype(bool)]
                logger.debug(f"stream {indices[i]}: {len(kept_boxes)} detections after NMS")
        except ImportError:
            logger.debug(f"batch {list(indices)} → {output.shape} (nms_cuda not available)")
        await asyncio.sleep(0)
    logger.info("consumer stopped")