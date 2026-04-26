#!/usr/bin/env bash
# Spin up N looped RTSP streams split across multiple mediamtx servers.
#
# Usage: ./scripts/start_streams.sh [num_streams] [streams_per_server]
#   ./scripts/start_streams.sh        # 8 streams, 2 per server (4 servers)
#   ./scripts/start_streams.sh 4 2    # 4 streams, 2 per server (2 servers)

set -euo pipefail

N="${1:-8}"
PER_SERVER="${2:-2}"
VID_DIR="videos"
BASE_RTSP=8554
BASE_HLS=8888

if ! command -v mediamtx &>/dev/null; then echo "mediamtx not found"; exit 1; fi
if ! command -v ffmpeg &>/dev/null; then echo "ffmpeg not found"; exit 1; fi

mapfile -t CLIPS < <(find "$VID_DIR" -maxdepth 1 -name "*.mp4" | sort)
if [[ ${#CLIPS[@]} -eq 0 ]]; then
    echo "No .mp4 files found in $VID_DIR/ — run ./scripts/download_videos.sh"
    exit 1
fi

N_SERVERS=$(( (N + PER_SERVER - 1) / PER_SERVER ))
echo "Starting $N streams across $N_SERVERS mediamtx servers ($PER_SERVER per server)..."

MEDIAMTX_PIDS=()
FFMPEG_PIDS=()

wait_for_port() {
    local port=$1
    for _ in $(seq 1 20); do
        nc -z localhost "$port" 2>/dev/null && return 0
        sleep 0.3
    done
    echo "ERROR: mediamtx did not start on port $port — check /tmp/mediamtx_*.log"
    return 1
}

for s in $(seq 0 $((N_SERVERS - 1))); do
    rtsp_port=$(( BASE_RTSP + s ))
    hls_port=$(( BASE_HLS + s ))
    log="/tmp/mediamtx_${rtsp_port}.log"

    cfg=$(mktemp /tmp/mediamtx_XXXXXX.yml)
    rtp_port=$(( 8000 + s * 2 ))
    rtcp_port=$(( rtp_port + 1 ))
    cat > "$cfg" <<EOF
rtspAddress: :${rtsp_port}
rtpAddress: :${rtp_port}
rtcpAddress: :${rtcp_port}
hls: true
hlsAddress: :${hls_port}
rtmp: false
webrtc: false
srt: false
api: false
metrics: false
pprof: false
paths:
  all_others:
EOF

    mediamtx "$cfg" >"$log" 2>&1 &
    MEDIAMTX_PIDS+=($!)
    wait_for_port "$rtsp_port"
    echo "  mediamtx :${rtsp_port} (HLS :${hls_port}) started"
done

for i in $(seq 0 $((N - 1))); do
    server=$(( i / PER_SERVER ))
    rtsp_port=$(( BASE_RTSP + server ))
    clip="${CLIPS[$((i % ${#CLIPS[@]}))]}"

    ffmpeg -re -stream_loop -1 -i "$clip" \
        -c copy -f rtsp \
        "rtsp://localhost:${rtsp_port}/stream${i}" \
        -loglevel error &
    FFMPEG_PIDS+=($!)
    echo "  stream${i} → rtsp://localhost:${rtsp_port}/stream${i}  ($(basename "$clip"))"
done

echo ""
echo "$N streams live. Ctrl-C to stop."

cleanup() {
    echo "Stopping..."
    kill "${FFMPEG_PIDS[@]}" 2>/dev/null || true
    kill "${MEDIAMTX_PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait
