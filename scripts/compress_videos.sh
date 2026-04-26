#!/usr/bin/env bash
# Compress all .mp4 files in videos/ in-place using H.264 CRF encoding.
# Usage: ./scripts/compress_videos.sh [crf]   (default CRF=28)

set -euo pipefail

CRF="${1:-28}"
VID_DIR="$(dirname "$0")/../videos"

if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found"
    exit 1
fi

mapfile -t CLIPS < <(find "$VID_DIR" -maxdepth 1 -name "*.mp4" | sort)
if [[ ${#CLIPS[@]} -eq 0 ]]; then
    echo "No .mp4 files found in $VID_DIR/"
    exit 1
fi

echo "Compressing ${#CLIPS[@]} clip(s) with CRF=$CRF..."

for src in "${CLIPS[@]}"; do
    tmp="${src%.mp4}_tmp.mp4"
    echo "  $(basename "$src")"
    ffmpeg -fflags +discardcorrupt -i "$src" \
        -c:v libx264 -crf "$CRF" -preset fast \
        -c:a aac -b:a 128k \
        -movflags +faststart \
        "$tmp" -y -loglevel error
    mv "$tmp" "$src"
done

echo "Done."
