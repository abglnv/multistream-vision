from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import supervision as sv

VEHICLE_CLASSES = frozenset([2, 3, 5, 7])  # car, motorcycle, bus, truck
COCO_NAMES = {2: "car", 3: "moto", 5: "bus", 7: "truck"}

_MODEL_SIZE = 640
_HISTORY_LEN = 20  


@dataclass
class TrackResult:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    velocity_px_s: float


class StreamTracker:
    def __init__(self, fps: float = 25.0):
        self.fps = fps
        self._byte = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=int(fps),
        )
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=_HISTORY_LEN))

    def update(
        self,
        boxes_cxcywh: np.ndarray,  # [K, 4] model-space (0..640)
        scores: np.ndarray,         # [K]
        class_ids: np.ndarray,      # [K] int
        frame_hw: tuple[int, int],  # (H, W) of original frame
    ) -> list[TrackResult]:
        h, w = frame_hw
        sx, sy = w / _MODEL_SIZE, h / _MODEL_SIZE

        detections = self._build_detections(boxes_cxcywh, scores, class_ids, sx, sy)
        tracked = self._byte.update_with_detections(detections)

        results = []
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            cid = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            self.history[tid].append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            results.append(TrackResult(tid, x1, y1, x2, y2, cid, self._velocity(tid)))

        return results

    def _build_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        sx: float,
        sy: float,
    ) -> sv.Detections:
        if len(boxes) == 0:
            return sv.Detections.empty()

        mask = np.isin(class_ids, list(VEHICLE_CLASSES))
        if not mask.any():
            return sv.Detections.empty()

        b = boxes[mask]
        cx, cy, bw, bh = b.T
        xyxy = np.stack([
            (cx - bw / 2) * sx,
            (cy - bh / 2) * sy,
            (cx + bw / 2) * sx,
            (cy + bh / 2) * sy,
        ], axis=1).astype(np.float32)

        return sv.Detections(
            xyxy=xyxy,
            confidence=scores[mask],
            class_id=class_ids[mask].astype(int),
        )

    def _velocity(self, track_id: int) -> float:
        pts = self.history[track_id]
        if len(pts) < 2:
            return 0.0
        arr = np.array(pts)
        return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).mean() * self.fps)
