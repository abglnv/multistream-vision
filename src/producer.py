import asyncio
import logging

import cv2

logger = logging.getLogger(__name__)

QUEUE_MAXSIZE = 2


async def stream_producer(
    stream_id: int,
    url: str,
    queue: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    cap: cv2.VideoCapture | None = None

    def _open() -> cv2.VideoCapture | None:
        c = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        return c if c.isOpened() else None

    def _read():
        assert cap is not None
        return cap.read()

    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            cap = await asyncio.to_thread(_open)
            if cap is None:
                logger.warning(f"[{stream_id}] can't open, retrying in 3s")
                await asyncio.sleep(3)
                continue
            logger.info(f"[{stream_id}] connected")

        try:
            ret, frame = await asyncio.to_thread(_read)
        except Exception as e:
            logger.error(f"[{stream_id}] read error: {e}")
            cap.release()
            cap = None
            continue

        if not ret or frame is None:
            logger.warning(f"[{stream_id}] stream ended, reconnecting")
            cap.release()
            cap = None
            continue

        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(frame)

    if cap:
        cap.release()
    logger.info(f"[{stream_id}] producer stopped")
