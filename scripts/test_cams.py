import asyncio
import sys
import time

import cv2

BASE_URL = "https://kaztoll.kz/livestream"
STREAM_NAMES = [
    "cam1.mp4", "cam2.mp4", "cam3.mp4", "cam4.mp4",
    "jjvezd.mp4", "arshaly.mp4", "anar.mp4",
    "oshagandy.mp4", "ttvezd.mp4", "kunaev.mp4",
]
TIMEOUT = 15.0 


async def probe(name: str) -> dict:
    url = f"{BASE_URL}/{name}"
    result = {"name": name, "url": url, "status": "unknown",
              "resolution": None, "fps": None, "latency_ms": None}

    def _read_one_frame():
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            return None, None, None
        t0 = time.time()
        ret, frame = cap.read()
        latency = (time.time() - t0) * 1000
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return (frame, fps, latency) if ret else (None, None, latency)

    try:
        frame, fps, latency = await asyncio.wait_for(
            asyncio.to_thread(_read_one_frame), timeout=TIMEOUT
        )
        if frame is not None:
            h, w = frame.shape[:2]
            result.update(
                status="LIVE",
                resolution=f"{w}x{h}",
                fps=round(fps, 1),
                latency_ms=round(latency, 0),
            )
        else:
            result["status"] = "NO_FRAME"
    except asyncio.TimeoutError:
        result["status"] = f"TIMEOUT>{TIMEOUT}s"
    except Exception as exc:
        result["status"] = f"ERROR:{exc}"

    return result


def render_table(results: list[dict]) -> None:
    col = {"name": 20, "status": 16, "resolution": 12, "fps": 6, "latency_ms": 10}
    header = (
        f"{'Stream':<{col['name']}} {'Status':<{col['status']}} "
        f"{'Resolution':<{col['resolution']}} {'FPS':<{col['fps']}} {'Latency(ms)':<{col['latency_ms']}}"
    )
    sep = "-" * len(header)
    print(f"\n{header}\n{sep}")
    for r in results:
        marker = "✓" if r["status"] == "LIVE" else "✗"
        print(
            f"{r['name']:<{col['name']}} "
            f"{marker} {r['status']:<{col['status'] - 2}} "
            f"{str(r['resolution'] or '-'):<{col['resolution']}} "
            f"{str(r['fps'] or '-'):<{col['fps']}} "
            f"{str(r['latency_ms'] or '-'):<{col['latency_ms']}}"
        )
    live = sum(1 for r in results if r["status"] == "LIVE")
    print(f"\n{live}/{len(results)} streams alive\n")


async def main():
    print(f"Probing {len(STREAM_NAMES)} kaztoll.kz streams (timeout={TIMEOUT}s each)…")
    results = await asyncio.gather(*[probe(n) for n in STREAM_NAMES])
    render_table(list(results))

    if not any(r["status"] == "LIVE" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
