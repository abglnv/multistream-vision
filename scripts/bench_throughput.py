import argparse
import asyncio
import concurrent.futures
import time
from collections import defaultdict

import cv2
import numpy as np

from src.consumer import InferenceEngine
from src.streams import STREAMS

QUEUE_MAXSIZE = 2
WARMUP_S      = 5     # seconds before we start counting


async def _producer(stream_id, url, queue, stop_event, counters, executor):
    loop = asyncio.get_running_loop()
    cap  = None

    def _open():
        c = cv2.VideoCapture()
        c.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000)
        ok = c.open(url, cv2.CAP_FFMPEG)
        return c if ok else None

    def _read():
        return cap.read()

    def _release():
        if cap:
            cap.release()

    try:
        while not stop_event.is_set():
            if cap is None or not cap.isOpened():
                cap = await loop.run_in_executor(executor, _open)
                if cap is None:
                    await asyncio.sleep(3)
                    continue

            try:
                ret, frame = await loop.run_in_executor(executor, _read)
            except Exception:
                await loop.run_in_executor(executor, _release)
                cap = None
                continue

            if not ret or frame is None:
                await loop.run_in_executor(executor, _release)
                cap = None
                continue

            counters["frames"][stream_id] += 1

            if queue.full():
                try:
                    queue.get_nowait()
                    counters["dropped"][stream_id] += 1
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(frame)
    finally:
        await loop.run_in_executor(executor, _release)


async def _consumer(queues, stop_event, engine, inf_pool, stats):
    loop = asyncio.get_running_loop()

    while not stop_event.is_set():
        frames = []
        for q in queues:
            try:
                frames.append(q.get_nowait())
            except asyncio.QueueEmpty:
                frames.append(None)

        live = [(i, f) for i, f in enumerate(frames) if f is not None]
        if not live:
            await asyncio.sleep(0.005)
            continue

        _, live_frames = zip(*live)

        t0    = time.perf_counter()
        batch = await loop.run_in_executor(inf_pool, engine.preprocess, list(live_frames))
        stats["pre_ms"].append((time.perf_counter() - t0) * 1000)

        t1     = time.perf_counter()
        _      = await loop.run_in_executor(inf_pool, engine.infer, batch)
        stats["inf_ms"].append((time.perf_counter() - t1) * 1000)

        stats["batches"]  += 1
        stats["consumed"] += len(live_frames)


async def run(n_streams: int, duration: int):
    urls     = STREAMS[:n_streams]
    queues   = [asyncio.Queue(maxsize=QUEUE_MAXSIZE) for _ in urls]
    stop     = asyncio.Event()
    counters = {"frames": defaultdict(int), "dropped": defaultdict(int)}
    stats    = {"pre_ms": [], "inf_ms": [], "batches": 0, "consumed": 0}

    engine   = InferenceEngine()
    inf_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="infer")
    executors = [
        concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"stream{i}")
        for i in range(n_streams)
    ]

    producers = [
        asyncio.create_task(_producer(i, url, queues[i], stop, counters, executors[i]))
        for i, url in enumerate(urls)
    ]
    consumer = asyncio.create_task(_consumer(queues, stop, engine, inf_pool, stats))

    print(f"Connecting to {n_streams} streams… (warmup {WARMUP_S}s)")
    await asyncio.sleep(WARMUP_S)

    # reset after warmup
    for d in counters.values():
        d.clear()
    stats.update(pre_ms=[], inf_ms=[], batches=0, consumed=0)

    print(f"Measuring for {duration}s…\n")
    t_start = time.perf_counter()
    await asyncio.sleep(duration)
    elapsed = time.perf_counter() - t_start

    stop.set()
    await asyncio.gather(*producers, consumer, return_exceptions=True)
    inf_pool.shutdown(wait=False)
    for ex in executors:
        ex.shutdown(wait=False)

    # ── report ───────────────────────────────────────────────────────────────
    print(f"{'━'*58}")
    print(f"  Duration : {elapsed:.1f}s    Streams : {n_streams}")
    print(f"{'━'*58}")

    print(f"\n  {'Stream':<10} {'Captured':>9} {'Dropped':>9} {'FPS':>8}  {'Drop%':>7}")
    print(f"  {'─'*50}")
    total_cap = total_drop = 0
    for i in range(n_streams):
        cap_  = counters["frames"][i]
        drop_ = counters["dropped"][i]
        fps_  = cap_ / elapsed
        pct_  = 100 * drop_ / max(cap_ + drop_, 1)
        print(f"  stream{i:<4} {cap_:>9} {drop_:>9} {fps_:>8.1f}  {pct_:>6.1f}%")
        total_cap  += cap_
        total_drop += drop_

    print(f"  {'─'*50}")
    total_fps  = total_cap / elapsed
    total_pct  = 100 * total_drop / max(total_cap + total_drop, 1)
    print(f"  {'TOTAL':<10} {total_cap:>9} {total_drop:>9} {total_fps:>8.1f}  {total_pct:>6.1f}%")

    if stats["pre_ms"]:
        pre = np.array(stats["pre_ms"])
        inf = np.array(stats["inf_ms"])
        con = stats["consumed"] / elapsed
        print(f"\n  Consumer  ({stats['batches']} batches, {stats['consumed']} frames total)")
        print(f"  {'─'*50}")
        print(f"  {'Metric':<24} {'mean':>8}  {'min':>8}  {'p99':>8}")
        print(f"  {'─'*50}")
        print(f"  {'Preprocess (ms)':<24} {pre.mean():>8.2f}  {pre.min():>8.2f}  {np.percentile(pre,99):>8.2f}")
        print(f"  {'Infer (ms)':<24} {inf.mean():>8.2f}  {inf.min():>8.2f}  {np.percentile(inf,99):>8.2f}")
        print(f"  {'End-to-end (ms)':<24} {(pre+inf).mean():>8.2f}  {(pre+inf).min():>8.2f}  {np.percentile(pre+inf,99):>8.2f}")
        print(f"  {'Consumer FPS':<24} {con:>8.1f}")

    print(f"\n{'━'*58}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams",  type=int, default=8)
    ap.add_argument("--duration", type=int, default=60)
    args = ap.parse_args()
    asyncio.run(run(args.streams, args.duration))


if __name__ == "__main__":
    main()
