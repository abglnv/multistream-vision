#!/usr/bin/env bash
# Spin up N looped RTSP streams from the videos/ directory.
# Each stream gets a different clip (cycles if N > number of clips).
# Requires: mediamtx (brew install mediamtx), ffmpeg
#
# Usage: ./scripts/start_streams.sh [num_streams]
#   ./scripts/start_streams.sh        # 8 streams (default)
#   ./scripts/start_streams.sh 4

set -euo pipefail

N="${1:-8}"
VID_DIR="videos"

if ! command -v mediamtx &>/dev/null; then
    echo "mediamtx not found — install with: brew install mediamtx"
    exit 1
fi
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found — install with: brew install ffmpeg"
    exit 1
fi

mapfile -t CLIPS < <(find "$VID_DIR" -maxdepth 1 -name "*.mp4" | sort)
if [[ ${#CLIPS[@]} -eq 0 ]]; then
    echo "No .mp4 files found in $VID_DIR/"
    echo "Run: ./scripts/download_videos.sh"
    exit 1
fi

echo "Found ${#CLIPS[@]} clip(s) in $VID_DIR/, starting $N streams..."

mediamtx &
MEDIAMTX_PID=$!
sleep 1

FFMPEG_PIDS=()
for i in $(seq 0 $((N - 1))); do
    clip="${CLIPS[$((i % ${#CLIPS[@]}))]}"
    ffmpeg -re -stream_loop -1 -i "$clip" \
        -c copy -f rtsp \
        "rtsp://localhost:8554/stream${i}" \
        -loglevel error &
    FFMPEG_PIDS+=($!)
    echo "  stream${i} → rtsp://localhost:8554/stream${i}  ($(basename "$clip"))"
done

echo ""
echo "$N streams live. Ctrl-C to stop."

cleanup() {
    echo ""
    echo "Stopping..."
    kill "${FFMPEG_PIDS[@]}" 2>/dev/null || true
    kill "$MEDIAMTX_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait
