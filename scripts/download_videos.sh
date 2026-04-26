# Usage: ./scripts/download_videos.sh

set -euo pipefail

declare -A SOURCES=(
    ["https://youtu.be/wqctLW0Hb_0"]=2
    ["https://youtu.be/QuUxHIVUoaY"]=2
    ["https://youtu.be/PNCJQkvALVc"]=2
    ["https://youtu.be/TW3EH4cnFZo"]=4
)

SEG_SECS=900   # 15 minutes
OUT="videos"
mkdir -p "$OUT"

if ! command -v yt-dlp &>/dev/null; then
    echo "yt-dlp not found — install with: brew install yt-dlp"
    exit 1
fi
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found — install with: brew install ffmpeg"
    exit 1
fi

counter=0
src_idx=0

for url in "${!SOURCES[@]}"; do
    n_segs="${SOURCES[$url]}"
    tmp="$OUT/_tmp_${src_idx}.mp4"
    src_idx=$((src_idx + 1))

    echo ""
    echo "[$src_idx/4] Downloading → $url"
    yt-dlp -f "bv[ext=mp4]+ba[ext=m4a]/best[ext=mp4]/best" \
        --merge-output-format mp4 \
        -o "$tmp" "$url"

    echo "  Splitting into $n_segs × 15-min clips..."
    for seg in $(seq 0 $((n_segs - 1))); do
        out=$(printf "%s/video_%02d.mp4" "$OUT" "$counter")
        ffmpeg -ss $((seg * SEG_SECS)) -i "$tmp" \
            -t $SEG_SECS -c copy -avoid_negative_ts 1 \
            "$out" -y -loglevel error
        echo "    → $out"
        counter=$((counter + 1))
    done

    rm "$tmp"
done

echo ""
echo "Done — $counter clips in $OUT/"
echo "Run: ./scripts/start_streams.sh"
