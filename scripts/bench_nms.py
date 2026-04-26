import time
import numpy as np

WARMUP = 10
RUNS   = 100
IOU_TH = 0.45


# ── pure-NumPy reference ────────────────────────────────────────────────────

def numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return np.array(keep, dtype=np.int32)


# ── timing helper ───────────────────────────────────────────────────────────

def bench(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    a = np.array(t)
    return a.mean(), a.min(), np.percentile(a, 99)


def row(label, np_r, cu_r):
    speedup = np_r[0] / cu_r[0]
    print(
        f"  {label:<22}"
        f"  {np_r[0]:>8.2f}ms  {np_r[1]:>8.2f}ms  {np_r[2]:>8.2f}ms"
        f"  │  {cu_r[0]:>8.2f}ms  {cu_r[1]:>8.2f}ms  {cu_r[2]:>8.2f}ms"
        f"  │  {speedup:>6.1f}x"
    )


def make_data(n, rng):
    boxes  = rng.random((n, 4), dtype=np.float32) * np.array([640, 640, 200, 200], dtype=np.float32)
    scores = rng.random(n, dtype=np.float32)
    return boxes, scores


# ── main ────────────────────────────────────────────────────────────────────

def main():
    try:
        import nms_cuda
    except ImportError:
        print("ERROR: nms_cuda not found — compile it first")
        return

    rng = np.random.default_rng(42)

    header = (
        f"  {'':22}  {'':>8}   {'NumPy':^27}   {'':3}  {'CUDA':^27}   {'':6}\n"
        f"  {'Label':<22}  {'mean':>8}    {'min':>8}    {'p99':>8}  │"
        f"  {'mean':>8}    {'min':>8}    {'p99':>8}  │  {'speedup':>6}"
    )

    # ── 1. single-stream, varying N boxes ───────────────────────────────────
    print("\n━━━  Single-stream NMS  (varying number of boxes)  ━━━")
    print(header)
    print("  " + "─" * 100)

    for n in [1_000, 8_400, 25_000, 100_000, 500_000]:
        boxes, scores = make_data(n, rng)
        np_r  = bench(lambda b=boxes, s=scores: numpy_nms(b, s, IOU_TH))
        cu_r  = bench(lambda b=boxes, s=scores: nms_cuda.run_nms(b, s, IOU_TH))
        row(f"{n:>9,} boxes", np_r, cu_r)

    # ── 2. multi-stream batch, fixed 8400 boxes (YOLOv8 output size) ────────
    print("\n━━━  Multi-stream batch NMS  (8 400 boxes/stream)  ━━━")
    print(header)
    print("  " + "─" * 100)

    for batch in [1, 2, 4, 8]:
        data = [make_data(8_400, rng) for _ in range(batch)]

        def np_batch(d=data):
            for b, s in d:
                numpy_nms(b, s, IOU_TH)

        def cu_batch(d=data):
            for b, s in d:
                nms_cuda.run_nms(b, s, IOU_TH)

        np_r = bench(np_batch)
        cu_r = bench(cu_batch)
        row(f"{batch} stream{'s' if batch>1 else ''} × 8 400", np_r, cu_r)

    print()


if __name__ == "__main__":
    main()
