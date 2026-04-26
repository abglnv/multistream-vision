# multistream vision 

Async pipeline that ingests 8 simultaneous RTSP streams, runs YOLOv8n inference on every frame batch, and serves a live MJPEG grid over HTTP. NMS is accelerated by a custom CUDA kernel exposed via Pybind11.

---

## Architecture

```
 RTSP streams (mediamtx)
   stream0 ──► Producer 0 (dedicated thread) ──► Queue 0 ─┐
   stream1 ──► Producer 1 (dedicated thread) ──► Queue 1 ─┤
   ...                                                      ├──► Consumer ──► MJPEG /stream
   streamn ──► Producer {n} (dedicated thread) ──► Queue {n} ─┘     (inference pool, 4 threads)
```

- Each producer runs in its own `ThreadPoolExecutor(max_workers=1)` — FFMPEG operations never share a thread pool, eliminating cross-stream interference.
- The consumer uses a separate 4-thread inference pool so `cap.read()` calls never starve preprocessing or inference.
- Queues are capped at 2 frames — old frames are dropped under load, RAM stays flat.

---

## Stack

| Layer               | Tool                           |
| ------------------- | ------------------------------ |
| Async orchestration | Python `asyncio`             |
| Stream ingest       | OpenCV + FFMPEG (RTSP)         |
| Inference           | YOLOv8n → ONNX Runtime (CUDA) |
| NMS                 | Custom CUDA kernel + Pybind11  |
| Stream server       | mediamtx                       |
| Web output          | aiohttp MJPEG                  |

---

## Setup

**Requirements:** Python 3.11+, `uv`, `ffmpeg`, `mediamtx`, CUDA toolkit (for NMS kernel)

```bash
# install deps
uv sync

# build CUDA NMS kernel
nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC -o libnms.so nms.cu
uv run python setup.py build_ext --inplace

# export YOLOv8n to ONNX
uv run pip install ultralytics
uv run yolo export model=yolov8n.pt format=onnx
mv yolov8n.onnx models/
```

---

## Usage

```bash
# 1. download and slice source videos (one-time)
./scripts/download_videos.sh

# 2. start 8 looped RTSP streams
./scripts/start_streams.sh

# 3. run the pipeline
uv run python -m src.main --streams 8

# open in browser
open http://localhost:8080
```

---

## Benchmarks

> Tested on NVIDIA GTX 1650 8 GB.

### NMS — NumPy vs CUDA (single stream)

| Boxes | NumPy | CUDA | Speedup |
|------:|------:|-----:|--------:|
| 1 000 | 14.33 ms | 1.34 ms | **10.7×** |
| 8 400 | 145.35 ms | 4.88 ms | **29.8×** |
| 25 000 | 529.61 ms | 11.54 ms | **45.9×** |
| 100 000 | 5 571.57 ms | 88.16 ms | **63.2×** |

### Throughput (2 streams, 60 s)

| Stream | FPS | Drop rate |
|--------|----:|----------:|
| stream0 | 24.9 | 38.8% |
| stream1 | 23.6 | 39.5% |
| **Total** | **48.4** | **39.1%** |

Consumer pipeline (per batch):

| Stage | mean | min | p99 |
|-------|-----:|----:|----:|
| Preprocess | 23.48 ms | 2.89 ms | 39.87 ms |
| Infer (ONNX) | 86.01 ms | 39.98 ms | 115.64 ms |
| End-to-end | 109.49 ms | 44.63 ms | 103.57 ms |
| Consumer FPS | 17.2 | | |

### Reproduce

```bash
# throughput
uv run python scripts/bench_throughput.py --streams 8 --duration 60

# NMS
uv run python scripts/bench_nms.py

# memory leak check
uv run mprof run --interval 60 python -m src.main --streams 8
mprof plot -o memory.png
explorer.exe memory.png
```

---

## Watchdog

The pipeline runs as a `systemd` service with `Restart=always` and `WatchdogSec=10`. A separate `psutil`-based watchdog monitors RSS and CPU usage and sends `SIGKILL` if thresholds are exceeded.

```bash
# install as systemd service
sudo cp deploy/watchdog.service /etc/systemd/system/
sudo systemctl enable --now watchdog
```
