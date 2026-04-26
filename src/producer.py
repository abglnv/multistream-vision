import asyncio
import logging
import os
import time

import cv2

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "referer;https://kaztoll.kz/"
    "|user_agent;Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

logger = logging.getLogger(__name__)

QUEUE_MAXSIZE = 2


def _stamped(url: str) -> str:
    ts = int(time.time() * 1000)
    base = url.split("?")[0]
    return f"{base}?t={ts}"


async def stream_producer(
    stream_id: int,
    url: str,
    queue: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    cap: cv2.VideoCapture | None = None

    def _open() -> cv2.VideoCapture | None:
        c = cv2.VideoCapture(_stamped(url), cv2.CAP_FFMPEG)
        c.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        return c if c.isOpened() else None

    def _read():
        assert cap is not None 
        return cap.read()
    
    while not stop_event.is_set():
        # reconnect 
        if cap is None or not cap.isOpened(): 
            cap = await asyncio.to_thread(_open)
            if cap is None: 
                logger.warning(f"{stream_id} producer can't open video stream; retrying in 3 seconds")
                await asyncio.sleep(3)
                continue 
            logger.info(f"cap {stream_id} connected")

        try: 
            ret, frame = await asyncio.to_thread(_read)
        except Exception as e:
            logger.error(f"[{stream_id}] read error: {e}")
            cap = None 
            continue 

        if not ret or frame is None:
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